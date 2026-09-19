#!/usr/bin/env python
"""Resumable CLI: run TRIBE v2 on a saved deck's slides and cache the result for the web
app. Read README.md before running this -- it needs its own environment, a CUDA GPU, and a
HuggingFace token with the gated Llama-3.2-3B license accepted.

Usage:
    python run.py --run-id sample-llm-serving --out ../../fixtures/neural
    python run.py --run-id sample-llm-serving --out ../../fixtures/neural --force
    python run.py --run-id sample-llm-serving --out ../../fixtures/neural --slides 3,7,12

Resumable: a slide with an existing metrics.json is skipped unless --force is passed, so a
crash partway through a deck costs at most the slide that was in flight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sightline.neural import SURFACE_VIEWS, OVERLAY_LABEL, assert_may_run_tribe, processing_ratio  # noqa: E402
from sightline.store import RunStore, BUNDLED_RUNS_DIR, data_dir  # noqa: E402

from narrate import synthesize  # noqa: E402
from regions import fetch_region_masks, global_field_power, region_drive  # noqa: E402


def _build_video(slide_png: Path, duration_s: float, out_path: Path) -> Path:
    """A trivial "video": the slide's own image held static for the narration's duration.
    Expect the V-JEPA2 video encoder to contribute little here (spec 5) -- this input is a
    still image, not a movie. That is the disclosed, expected shape of this pipeline, not
    a bug to fix."""
    from moviepy.editor import ImageClip

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ImageClip(str(slide_png)).set_duration(max(duration_s, 1.0)).write_videofile(
        str(out_path), fps=1, audio=False, logger=None
    )
    return out_path


def _call_tribe(model, video_path: Path, audio_path: Path, transcript: str) -> np.ndarray:
    """The one function that actually talks to TRIBE. Written from the research pass
    described in README.md ("What's verified vs. what needs confirming"), not from a
    completed live run -- no CUDA GPU was available to test against while writing this.
    Confirm the call shape against github.com/facebookresearch/tribev2's README and
    tribe_demo.ipynb before a real run; this is the only place that needs to change if it
    differs."""
    response = model.predict(video_path=video_path, audio_path=audio_path, transcript=transcript)
    response = np.asarray(response)
    if response.ndim != 2 or response.shape[1] != 20484:
        raise RuntimeError(
            f"unexpected TRIBE output shape {response.shape}; expected (T, 20484). "
            "The model's real output shape or call signature may differ from what "
            "_call_tribe() assumes -- check the actual tribev2 repo."
        )
    return response


def _render_surface(response: np.ndarray, masks: dict, out_dir: Path) -> None:
    """One PNG per SURFACE_VIEWS entry, cold_hot colormap symmetric about zero, with the
    "predicted, not measured" disclosure burned into the image itself (spec 9: "on the
    image itself, not in a tab")."""
    from nilearn import datasets, plotting

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    per_vertex = response.mean(axis=0)  # time-averaged predicted response, one value/vertex
    n_left = len(masks["visual"]) // 2  # left/right halves are concatenated identically on load

    specs = {
        "lateral_left": (fsaverage["infl_left"], fsaverage["sulc_left"], per_vertex[:n_left], "lateral"),
        "medial_left": (fsaverage["infl_left"], fsaverage["sulc_left"], per_vertex[:n_left], "medial"),
        "lateral_right": (fsaverage["infl_right"], fsaverage["sulc_right"], per_vertex[n_left:], "lateral"),
        "medial_right": (fsaverage["infl_right"], fsaverage["sulc_right"], per_vertex[n_left:], "medial"),
    }
    vmax = float(np.abs(per_vertex).max()) or 1.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (mesh, bg, values, view) in specs.items():
        fig = plotting.plot_surf_stat_map(
            mesh, values, bg_map=bg, hemi="left" if "left" in name else "right",
            view=view, cmap="cold_hot", vmax=vmax, symmetric_cbar=True, colorbar=True,
            title=OVERLAY_LABEL,
        )
        fig.savefig(out_dir / f"{name}.png", dpi=120)
        import matplotlib.pyplot as plt

        plt.close(fig)


def run_slide(run_id: str, index: int, text: str, image_png_path: Path, out_dir: Path, model, masks: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        narration = synthesize(text, tmp / "narration.mp3")
        video_path = _build_video(image_png_path, narration.duration_s, tmp / "slide.mp4")
        response = _call_tribe(model, video_path, narration.audio_path, narration.transcript)

    language_drive = region_drive(response, masks["language"])
    visual_drive = region_drive(response, masks["visual"])
    try:
        ratio = processing_ratio(language_drive, visual_drive)
    except ValueError:
        ratio = None
    metrics = {
        "language_drive": language_drive,
        "visual_drive": visual_drive,
        "processing_ratio": ratio,
        "gfp_negative_baseline": global_field_power(response),
        "narration_transcript": narration.transcript,
    }
    slide_dir = out_dir / run_id / str(index)
    slide_dir.mkdir(parents=True, exist_ok=True)
    (slide_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _render_surface(response, masks, slide_dir)
    return metrics


def _login_to_huggingface() -> None:
    """The text encoder (meta-llama/Llama-3.2-3B) is gated: this needs a HF token from an
    account that has accepted its license (README.md step 4). Reads HF_TOKEN from this
    directory's own .env -- deliberately not the repo-root .env, and never committed
    (git-ignored, same pattern as ANTHROPIC_API_KEY there)."""
    from dotenv import dotenv_values
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN") or dotenv_values(HERE / ".env").get("HF_TOKEN")
    if not token:
        sys.exit(
            "No HF_TOKEN found (checked the environment and .env in this directory). "
            "TRIBE v2's text encoder is a gated model; see README.md step 4."
        )
    login(token=token, add_to_git_credential=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="a saved run's id under data/history/ (or a bundled sample)")
    parser.add_argument("--out", required=True, type=Path, help="output root, e.g. ../../fixtures/neural")
    parser.add_argument("--slides", help="comma-separated slide indices; default is every slide in the run")
    parser.add_argument("--force", action="store_true", help="recompute even if metrics.json already exists")
    args = parser.parse_args()

    assert_may_run_tribe()  # refuses to proceed if SIGHTLINE_PROCESS=server is set
    _login_to_huggingface()

    store = RunStore(data_dir() / "history", bundled=[BUNDLED_RUNS_DIR])
    slides = store.load_slides(args.run_id)
    wanted = {int(s) for s in args.slides.split(",")} if args.slides else {s.index for s in slides}

    print(f"loading TRIBE v2 (this downloads ~8-10GB of weights on first run)...")
    from tribev2 import TribeModel  # confirm this import path against the real repo

    model = TribeModel.from_pretrained("facebook/tribev2", device="auto")

    print("fetching the Destrieux atlas for region masks (cached after first run)...")
    masks = fetch_region_masks()

    for slide in slides:
        if slide.index not in wanted:
            continue
        out_file = args.out / args.run_id / str(slide.index) / "metrics.json"
        if out_file.is_file() and not args.force:
            print(f"slide {slide.index}: already cached, skipping (--force to redo)")
            continue
        print(f"slide {slide.index}: narrating, encoding, and running TRIBE (6-13 min)...")
        image_path = store.image_path(args.run_id, slide.index)
        metrics = run_slide(args.run_id, slide.index, slide.text, image_path, args.out, model, masks)
        print(f"slide {slide.index}: processing_ratio={metrics['processing_ratio']}")

    print("done.")


if __name__ == "__main__":
    main()
