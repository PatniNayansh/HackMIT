"""Crop the burned-in title band off the cortical renders that already exist.

    python backend/scripts/crop_render_titles.py

The renders were produced with a matplotlib title across the top. The UI names the panel
instead, so the band is dead space repeating what the heading already says -- and it is the
only text in the app rendered as pixels, which means it cannot be restyled, translated or
selected. `run.py` no longer draws it; this fixes up the PNGs that predate that change.

Crops the app's own fixtures only. tribe_results/ keeps the renders exactly as the GX10
produced them: that directory is the record of the run, not a UI asset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    REPO_ROOT / "backend" / "fixtures" / "neural",
    REPO_ROOT / "backend" / "fixtures" / "lecture_surfaces",
    REPO_ROOT / "backend" / "fixtures" / "audio",
]
WHITE = 250


def title_band(a: np.ndarray) -> tuple[int, int] | None:
    """The first block of ink, when a taller block of ink (the figure) follows a clear gap."""
    rows = np.where((a < WHITE).any(axis=(1, 2)))[0]
    if not len(rows):
        return None
    bands, start = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] != rows[i - 1] + 1:
            bands.append((int(start), int(rows[i - 1])))
            start = rows[i]
    bands.append((int(start), int(rows[-1])))
    if len(bands) < 2:
        return None  # nothing but the figure: already cropped, or never had a title
    return bands[0][0], bands[1][0]  # title top, figure top


MARGIN = 24


def crop(path: Path) -> bool:
    """Normalise a render to its ink: drop the title band if there is one, and trim the dead
    whitespace either way. Renders made before the title was removed and renders made after it
    should end up the same shape, or they read as two different sizes side by side in the UI.
    Idempotent: a render already at MARGIN is left exactly as it is."""
    im = Image.open(path).convert("RGB")
    a = np.array(im)
    rows = np.where((a < WHITE).any(axis=(1, 2)))[0]
    if not len(rows):
        return False

    band = title_band(a)
    figure_top = band[1] if band else int(rows[0])
    top = max(0, figure_top - MARGIN)
    bottom = min(im.height, int(rows[-1]) + MARGIN)
    if (top, bottom) == (0, im.height):
        return False
    im.crop((0, top, im.width, bottom)).save(path)
    return True


def main() -> None:
    done = skipped = 0
    for root in TARGETS:
        for p in sorted(root.rglob("*.png")) if root.is_dir() else []:
            if crop(p):
                done += 1
            else:
                skipped += 1
    print(f"cropped {done} render(s), left {skipped} alone (no title band)")
    if not done and not skipped:
        sys.exit("no renders found; nothing to do")


if __name__ == "__main__":
    main()
