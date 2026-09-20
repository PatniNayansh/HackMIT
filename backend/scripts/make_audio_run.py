#!/usr/bin/env python
"""Turn a directory of TRIBE chunk output into an audio run the app can show.

    python backend/scripts/make_audio_run.py <chunk-dir> --id <id> --title "..." [--course ...]

<chunk-dir> is what the GX10 writes: one chunk_NN/ per chunk, each with metrics.json,
timecourse.json and four cortical surface PNGs. This reads them, stitches the per-chunk time
courses into one series, and writes

    backend/fixtures/audio/<id>/timecourse.json
    backend/fixtures/audio/<id>/surfaces/chunk_NN.png

Only `language_drive` survives into the plotted series. These runs have no visual input at all
-- they are audio -- so `visual_drive` is model noise around zero and `processing_ratio` divides
by it. Plotting either would be drawing a line through nothing.

Per-second values are jittery enough to hide the shape, so the series is a rolling mean sampled
every few seconds. Chunk boundaries are carried through because each chunk begins its own
context (the transcript is re-encoded from scratch), so the model is cold for a few seconds
after each one -- visible as a dip, and better explained than smoothed away.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "backend" / "fixtures" / "audio"
SURFACE_VIEW = "lateral_left"  # the language regions these runs are about face this way
WINDOW_S = 15
STRIDE_S = 5
CHUNK_SECONDS = 120  # what the GX10 splits with: ffmpeg -segment_time 120

# TRIBE's predictions are NOT one row per second, and the runner's `t_s` is a row index rather
# than a timestamp (its fallback spaces rows evenly over the row count when the segment objects
# expose no offset). A full 120-second chunk comes back as 240 rows, so a row is half a second.
# Reading `t_s` as seconds put every duration out by a factor of two -- the 8-minute lecture
# read as 16. The rate is derived below from the longest chunk rather than hard-coded, so a
# future change in TRIBE's sampling does not silently repeat the mistake.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chunk_dir", type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--course", default="")
    ap.add_argument("--venue", default="")
    ap.add_argument("--chunks-total", type=int, default=None,
                    help="chunks the whole recording would need, when only some were run")
    args = ap.parse_args()

    chunks = sorted(p for p in args.chunk_dir.iterdir() if p.is_dir() and (p / "timecourse.json").is_file())
    if not chunks:
        raise SystemExit(f"no chunk_NN/ directories with a timecourse under {args.chunk_dir}")

    out = OUT_ROOT / args.id
    (out / "surfaces").mkdir(parents=True, exist_ok=True)

    rows_per_chunk = [len(json.loads((d / "timecourse.json").read_text(encoding="utf-8"))) for d in chunks]
    seconds_per_row = CHUNK_SECONDS / max(rows_per_chunk)

    points: list[tuple[float, float]] = []
    starts: list[float] = []
    surfaces = []
    offset = 0.0
    for d, n_rows in zip(chunks, rows_per_chunk):
        segs = json.loads((d / "timecourse.json").read_text(encoding="utf-8"))
        starts.append(offset)
        points += [(offset + i * seconds_per_row, s["language_drive"]) for i, s in enumerate(segs)]
        chunk_len = n_rows * seconds_per_row
        src = d / f"{SURFACE_VIEW}.png"
        if src.is_file():
            shutil.copy2(src, out / "surfaces" / f"{d.name}.png")
            surfaces.append({
                "chunk": d.name,
                "from_min": round(offset / 60, 1),
                "to_min": round((offset + chunk_len) / 60, 1),
            })
        offset += chunk_len

    half = max(1, int(round((WINDOW_S / seconds_per_row) / 2)))
    smoothed = [
        (t, st.mean(v for _, v in points[max(0, i - half) : i + half + 1]))
        for i, (t, _) in enumerate(points)
    ]
    stride = max(1, int(round(STRIDE_S / seconds_per_row)))
    series = [{"t": round(t, 1), "v": round(v, 5)} for t, v in smoothed[::stride]]

    (out / "timecourse.json").write_text(
        json.dumps(
            {
                "id": args.id,
                "title": args.title,
                "course": args.course,
                "venue": args.venue,
                "source": "real recorded lecture audio, run through TRIBE v2 with no synthesis",
                "duration_s": round(points[-1][0]),
                "chunk_starts_s": [round(s) for s in starts],
                "chunks_run": len(chunks),
                "chunks_total": args.chunks_total or len(chunks),
                "smoothing_window_s": WINDOW_S,
                "seconds_per_segment": round(seconds_per_row, 4),
                "n_segments": len(points),
                "surface_view": SURFACE_VIEW,
                "surfaces": surfaces,
                "series": series,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{args.id}: {len(points)} segments -> {len(series)} points, "
          f"{points[-1][0] / 60:.1f} min, {len(surfaces)} surfaces -> {out}")


if __name__ == "__main__":
    main()
