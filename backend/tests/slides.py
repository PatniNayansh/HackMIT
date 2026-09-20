"""Two hand-written slides where the right answer is known in advance.

CLEAR      - plain-language, self-contained. All three audiences should converge on the
             presenter's point, and none should need a term defined.
JARGON     - dense with undefined field-specific terms. The expert should get it; the
             novice cannot. This is the classic expert blind spot.

`PLANTED_JARGON` is the list of terms deliberately left undefined on the JARGON slide; the
gate checks that the novice's term_gap recovers them.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from profe.audiences import DeckProfile, SlideInput

CLEAR_PROFILE = DeckProfile(domain="quarterly business reporting", adjacent_field="general management")
CLEAR_TEXT = (
    "[title] Q3 sales grew 20% over Q2\n"
    "[body] Revenue: $5.0M in Q2, $6.0M in Q3\n"
    "[body] Growth came from more customers renewing their subscriptions\n"
    "[body] Next quarter: keep the current plan"
)
CLEAR_INTENT = "Q3 sales grew 20% over Q2 because more customers renewed."

JARGON_PROFILE = DeckProfile(
    domain="LLM inference serving systems", adjacent_field="distributed systems and databases"
)
JARGON_TEXT = (
    "[title] PagedAttention + speculative decoding: 2.4x goodput at p99\n"
    "[body] KV-cache fragmentation: <4% waste via block tables\n"
    "[body] Draft model acceptance α = 0.78 at γ = 5\n"
    "[body] Continuous batching; chunked prefill off to hold the TTFT SLO"
)
# Written the way a presenter in the field would write it: in the field's own register.
JARGON_INTENT = (
    "Our serving system delivers 2.4x higher goodput at p99 latency by pairing "
    "PagedAttention-style KV-cache management with speculative decoding."
)
PLANTED_JARGON = [
    "PagedAttention", "speculative decoding", "goodput", "p99", "KV-cache", "block tables",
    "Draft model acceptance", "γ", "Continuous batching", "chunked prefill", "TTFT", "SLO",
]


def render_png(text: str, size: tuple[int, int] = (1280, 720)) -> bytes:
    """Plain white slide with the text laid out top to bottom. Enough to exercise the
    vision path; it makes no claim to look like a real deck."""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.load_default(size=44)
    body_font = ImageFont.load_default(size=32)
    y = 70
    for line in text.splitlines():
        role, _, content = line.partition("] ")
        font = title_font if role == "[title" else body_font
        draw.text((80, y), content, fill="black", font=font)
        y += 110 if role == "[title" else 80
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def clear_slide(index: int = 1, with_image: bool = True) -> SlideInput:
    return SlideInput(index, CLEAR_TEXT, render_png(CLEAR_TEXT) if with_image else None)


def jargon_slide(index: int = 1, with_image: bool = True) -> SlideInput:
    return SlideInput(index, JARGON_TEXT, render_png(JARGON_TEXT) if with_image else None)
