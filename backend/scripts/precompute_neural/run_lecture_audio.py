#!/usr/bin/env python
"""Run TRIBE v2 over a chunked lecture recording, one chunk at a time.

    python run_lecture_audio.py --job-dir ~/profe-jobs/<job_id>

Unlike run.py (one slide, one synthesized clip) this feeds *actual* recorded audio to TRIBE's
own get_events_dataframe(audio_path=...), which transcribes it with whisperx and produces a
genuine time-course of predicted response. There is no slide alignment: we have no
slide-transition timestamps for a recording, and inventing them would be worse than not having
them.

The job directory is prepared by the ProFe backend and looks like:

    <job-dir>/manifest.json      what to run: chunk list, order, start offsets
    <job-dir>/chunks/chunk_NN.*  the audio, already cut into short pieces
    <job-dir>/out/chunk_NN/      written here: metrics.json, timecourse.json, four surfaces
    <job-dir>/status.json        rewritten after every chunk, so the backend can poll

Chunks are short because they have to be: TRIBE's text encoder contextualises every word
against the whole preceding transcript, so a 25-minute file in one pass had a 30-hour ETA.
Each chunk starts its own context, which is also why the first seconds of each one read low.

Resumable: a chunk whose metrics.json already exists is skipped, so a crash costs at most the
chunk in flight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from profe.neural import SURFACE_VIEWS, assert_may_run_tribe, processing_ratio  # noqa: E402

from regions import fetch_region_masks, global_field_power, region_drive  # noqa: E402


def write_status(job_dir: Path, **fields) -> None:
    """Rewritten after every chunk. The backend polls this file and nothing else, so it is
    written whole each time rather than appended to -- a half-written line would read as a
    corrupt job rather than as a job in progress."""
    path = job_dir / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"updated_at": time.time(), **fields}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _segment_offset_seconds(segment) -> float | None:
    """Segment objects come from tribev2/neuralset, not from us. Try the plausible attribute
    names rather than assuming one, and fall back to None so the caller spaces them evenly
    instead of inventing a timestamp from a wrong attribute."""
    for attr in ("offset", "start", "onset", "t0"):
        if hasattr(segment, attr):
            try:
                return float(getattr(segment, attr))
            except (TypeError, ValueError):
                pass
    return None


def render_surfaces(preds: np.ndarray, masks: dict, out_dir: Path) -> None:
    """Four cortical views, averaged over the chunk. No title drawn on the image: the UI names
    the panel, and text baked into a PNG cannot be restyled, translated or selected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    per_vertex = preds.mean(axis=0)
    n_left = len(masks["visual"]) // 2
    specs = {
        "lateral_left": (fsaverage["infl_left"], fsaverage["sulc_left"], per_vertex[:n_left], "lateral"),
        "medial_left": (fsaverage["infl_left"], fsaverage["sulc_left"], per_vertex[:n_left], "medial"),
        "lateral_right": (fsaverage["infl_right"], fsaverage["sulc_right"], per_vertex[n_left:], "lateral"),
        "medial_right": (fsaverage["infl_right"], fsaverage["sulc_right"], per_vertex[n_left:], "medial"),
    }
    vmax = float(np.abs(per_vertex).max()) or 1.0
    for name in SURFACE_VIEWS:
        mesh, bg, values, view = specs[name]
        fig = plotting.plot_surf_stat_map(
            mesh, values, bg_map=bg, hemi="left" if "left" in name else "right",
            view=view, cmap="cold_hot", vmax=vmax, symmetric_cbar=True, colorbar=True,
        )
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        plt.close(fig)


def run_chunk(model, masks: dict, audio: Path, out_dir: Path, start_s: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = model.get_events_dataframe(audio_path=str(audio))
    preds, segments = model.predict(events)
    preds = np.asarray(preds)

    language_drive = region_drive(preds, masks["language"])
    visual_drive = region_drive(preds, masks["visual"])
    try:
        ratio = processing_ratio(language_drive, visual_drive)
    except ValueError:
        ratio = None

    metrics = {
        "language_drive": language_drive,
        "visual_drive": visual_drive,
        "processing_ratio": ratio,
        "gfp_negative_baseline": global_field_power(preds),
        "source": "real recorded lecture audio, not synthesized narration",
        "n_segments": int(preds.shape[0]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Per-segment time course. visual_drive is carried because the format is shared with the
    # per-slide output, but an audio-only run gives the visual regions nothing to respond to:
    # it is noise around zero, and the UI plots language drive alone for that reason.
    offsets = [_segment_offset_seconds(s) for s in segments]
    if all(o is not None for o in offsets):
        times = [float(o) for o in offsets]
    else:
        times = list(np.linspace(0, preds.shape[0], preds.shape[0], endpoint=False))
    rows = []
    for i in range(preds.shape[0]):
        ld = region_drive(preds[i : i + 1], masks["language"])
        vd = region_drive(preds[i : i + 1], masks["visual"])
        try:
            r = processing_ratio(ld, vd)
        except ValueError:
            r = None
        rows.append({"t_s": times[i], "language_drive": ld, "visual_drive": vd, "processing_ratio": r})
    (out_dir / "timecourse.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    render_surfaces(preds, masks, out_dir)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="recompute chunks that already have metrics.json")
    args = ap.parse_args()

    job = args.job_dir.expanduser()
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    out_root = job / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    assert_may_run_tribe()  # refuses if PROFE_PROCESS=server; the web app never runs this
    write_status(job, state="starting", done=0, total=len(chunks), chunk=None, error=None)

    try:
        from tribev2 import TribeModel

        write_status(job, state="loading_model", done=0, total=len(chunks), chunk=None, error=None)
        model = TribeModel.from_pretrained("facebook/tribev2", device="auto")
        masks = fetch_region_masks()

        done = 0
        for c in chunks:
            out_dir = out_root / c["name"]
            if (out_dir / "metrics.json").is_file() and not args.force:
                done += 1
                write_status(job, state="running", done=done, total=len(chunks), chunk=c["name"], error=None)
                continue
            write_status(job, state="running", done=done, total=len(chunks), chunk=c["name"], error=None)
            run_chunk(model, masks, job / "chunks" / c["file"], out_dir, c["start_s"])
            done += 1
            write_status(job, state="running", done=done, total=len(chunks), chunk=c["name"], error=None)

        write_status(job, state="complete", done=done, total=len(chunks), chunk=None, error=None)
        print("done.")
    except Exception as e:  # noqa: BLE001 - the backend has to see why, not just that
        write_status(job, state="failed", done=locals().get("done", 0), total=len(chunks),
                     chunk=None, error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
