#!/usr/bin/env python
"""Turn the GX10's real-lecture audio run into the one series the UI plots.

    python backend/scripts/make_lecture_timecourse.py

Reads tribe_results/umass_lecture_audio_chunks/ (chunks 0-3 of a real recorded lecture, run
through TRIBE v2 with no synthesis) and writes backend/fixtures/lecture_timecourse.json.

Only `language_drive` survives. That run had no visual input at all -- it is audio -- so
`visual_drive` there is model noise around zero and `processing_ratio` divides by it. Plotting
either would be drawing a line through nothing.

Per-second values are jittery enough to hide the shape, so the series is a 15 s rolling mean
sampled every 10 s: 958 segments become 96 points. Chunk starts are carried through because
each chunk begins its own context (the transcript is re-encoded from scratch), so the model is
cold for a few seconds after each one.
"""

from __future__ import annotations

import json
import shutil
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "tribe_results" / "umass_lecture_audio_chunks"
OUT = REPO_ROOT / "backend" / "fixtures" / "lecture_timecourse.json"
SURF_OUT = REPO_ROOT / "backend" / "fixtures" / "lecture_surfaces"
# The left lateral view: the language regions this run is about are on this face of the cortex.
SURFACE_VIEW = "lateral_left"
WINDOW_S = 15
STRIDE_S = 10


def main() -> None:
    chunks = sorted(SRC.glob("chunk_*"))
    if not chunks:
        raise SystemExit(f"no chunks under {SRC}")

    points: list[tuple[float, float]] = []
    starts: list[float] = []
    offset = 0.0
    for d in chunks:
        segs = json.loads((d / "timecourse.json").read_text(encoding="utf-8"))
        starts.append(offset)
        points += [(s["t_s"] + offset, s["language_drive"]) for s in segs]
        offset += segs[-1]["t_s"] + 1

    half = WINDOW_S // 2
    smoothed = [
        (t, st.mean(v for _, v in points[max(0, i - half) : i + half + 1]))
        for i, (t, _) in enumerate(points)
    ]
    series = [{"t": round(t, 1), "v": round(v, 5)} for t, v in smoothed[::STRIDE_S]]

    # One cortical view per chunk, so the climb can be seen and not only read off a line. Each
    # render carries its own colour scale (TRIBE renders symmetric about that chunk's own peak),
    # which is why the caption says so rather than inviting a pixel-to-pixel comparison.
    SURF_OUT.mkdir(parents=True, exist_ok=True)
    surfaces = []
    for i, d in enumerate(chunks):
        src = d / f"{SURFACE_VIEW}.png"
        if not src.is_file():
            continue
        shutil.copy2(src, SURF_OUT / f"{d.name}.png")
        surfaces.append({
            "chunk": d.name,
            "from_min": round(starts[i] / 60),
            "to_min": round((starts[i + 1] if i + 1 < len(starts) else points[-1][0]) / 60),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "title": "M1b: Summarizing data",
                "course": "CEE 260 / MIE 273, Probability and statistics in civil engineering",
                "venue": "UMass Amherst",
                "source": "real recorded lecture audio, run through TRIBE v2 with no synthesis",
                "duration_s": round(points[-1][0]),
                "chunk_starts_s": [round(s) for s in starts],
                "chunks_run": len(chunks),
                "chunks_total": 13,
                "smoothing_window_s": WINDOW_S,
                "n_segments": len(points),
                "surface_view": SURFACE_VIEW,
                "surfaces": surfaces,
                "series": series,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(points)} segments -> {len(series)} points, {points[-1][0] / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
