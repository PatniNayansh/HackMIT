"""Run the real audiences on one of the two hand-written test slides and print what each
said. Works from any directory.

    make demo                 # jargon slide
    make demo SLIDE=clear
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "tests"))

import slides  # noqa: E402

from sightline.audiences import AudienceEngine, FileCache  # noqa: E402
from sightline.divergence import score_slide  # noqa: E402
from sightline.llm import OpenAIClient  # noqa: E402

SLIDES = {
    "jargon": (slides.jargon_slide, slides.JARGON_PROFILE, slides.JARGON_INTENT),
    "clear": (slides.clear_slide, slides.CLEAR_PROFILE, slides.CLEAR_INTENT),
}


async def main(name: str) -> None:
    make_slide, profile, intent = SLIDES[name]
    engine = AudienceEngine(OpenAIClient(), FileCache(BACKEND / ".cache" / "audiences"))
    readings = await engine.read_slide(make_slide(), profile)
    s = score_slide(intent, {p: r.response for p, r in readings.items()}, slide_index=1)

    print(f"declared intent: {intent}\n")
    for p, r in readings.items():
        print(
            f"[{p}]  alignment={s.intent_alignment[p].value:.2f}  confidence={r.response.confidence:.2f}  "
            f"{'cached' if r.cached else f'{r.latency_s:.1f}s'}  ({r.model})\n"
            f"  takeaway:   {r.response.takeaway}\n"
            f"  unresolved: {r.response.unresolved_terms}\n"
            f"  questions:  {r.response.questions}\n"
            f"  claim:      {r.response.inferred_claim}\n"
        )
    print(f"audience_divergence = {s.audience_divergence.value:.3f}")
    print(f"blind_spot_score    = {s.blind_spot_score.value:.3f}")
    print(f"term_gap            = {list(s.term_gap.terms)}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "jargon"
    if name not in SLIDES:
        sys.exit(f"usage: try_slide.py [{'|'.join(SLIDES)}]")
    try:
        asyncio.run(main(name))
    except TypeError as e:  # the SDK's message when no credential resolves
        if "authentication" not in str(e):
            raise
        sys.exit("No API key found. Put OPENAI_API_KEY=... in the repo-root .env (see .env.example).")
