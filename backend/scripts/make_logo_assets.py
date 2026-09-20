#!/usr/bin/env python
"""Turn the ProFe wordmark into the assets the app needs.

    python backend/scripts/make_logo_assets.py [source.jpeg]

The source is a JPEG on a near-white background, so it has no transparency: dropped into the
dark theme as-is it sits in a white box. This keys that background out to alpha and writes:

    frontend/img/profe-wordmark.png   the full wordmark, trimmed to its ink, for the header
    frontend/img/profe-mark.png       the "P" alone, square, for the favicon and small sizes

The wordmark is 3.65:1. Squeezing that into a 16px favicon leaves it unreadable, which is why
the square mark exists rather than a stretched copy -- aspect ratio is never altered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = Path.home() / "Downloads" / "Profe Logo.jpeg"
OUT_DIR = REPO_ROOT / "frontend" / "img"
# Background is #f4f8f9-ish; the ink is deep navy through teal, all far darker than this.
BG_FLOOR = 232


def keyed(im: Image.Image) -> Image.Image:
    """Near-white to transparent, with the alpha ramped over the antialiased edge so the
    letterforms keep their curves instead of gaining a jagged halo."""
    a = np.array(im.convert("RGB")).astype(np.float32)
    lightness = a.max(axis=2)
    alpha = np.clip((BG_FLOOR - lightness) / 40.0, 0, 1) * 255
    out = np.dstack([a, alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def trimmed(im: Image.Image) -> Image.Image:
    box = im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return im.crop(box)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_file():
        raise SystemExit(f"logo not found: {src}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    word = trimmed(keyed(Image.open(src)))
    word.save(OUT_DIR / "profe-wordmark.png")

    # The "P": the first glyph, padded to a square so nothing is squashed at favicon sizes.
    w, h = word.size
    p = word.crop((0, 0, int(w * 0.215), h))
    side = max(p.size)
    mark = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    mark.paste(p, ((side - p.width) // 2, (side - p.height) // 2))
    mark.resize((512, 512), Image.LANCZOS).save(OUT_DIR / "profe-mark.png")

    print(f"wordmark {word.size} -> {OUT_DIR / 'profe-wordmark.png'}")
    print(f"mark     512x512 -> {OUT_DIR / 'profe-mark.png'}")


if __name__ == "__main__":
    main()
