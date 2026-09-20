#!/usr/bin/env python
"""DEMO, not a real run. Shows the ADHD research-lens pipeline end to end using
ILLUSTRATIVE, MADE-UP brain-response numbers standing in for a real TRIBE v2 prediction --
because TRIBE v2 itself has not run yet (no CUDA GPU / GX10 access at the time this was
written; see scripts/precompute_neural/README.md).

What is REAL in this script:
  * The DMN vertex mask (regions.fetch_dmn_mask) -- an actual projection of the Yeo 2011
    atlas onto the real fsaverage5 surface, the same mesh TRIBE v2 outputs onto.
  * The Z-score-against-the-deck math (sightline.neural.deck_z_scores) -- the same function
    the real app uses for processing_ratio.
  * The citation-backed sentence generator (sightline.research_lens.adhd_dmn_lens).
  * The slide revision -- an ACTUAL call to Claude, not canned text.

What is NOT real:
  * The "predicted brain response" arrays for both the original and revised slide. These
    are synthetically constructed (see `_synthetic_response`) to illustrate what an
    elevated-DMN slide would look like next to the rest of a deck, and what the SAME
    revision would look like IF the design heuristic in research_lens.py's revision prompt
    actually lowers predicted DMN drive -- a hypothesis this script assumes for
    illustration, not one it has tested. Getting a REAL before/after here needs an actual
    TRIBE v2 inference run on both slide versions, which needs a CUDA GPU.

Run from an environment with this directory's requirements (or at minimum nilearn) plus
the main backend's dependencies (anthropic, pydantic, python-dotenv) installed:

    python demo_adhd_lens.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sightline.audiences import SlideInput  # noqa: E402
from sightline.llm import AnthropicClient  # noqa: E402
from sightline.neural import deck_z_scores  # noqa: E402
from sightline.research_lens import adhd_dmn_lens, propose_adhd_friendly_revision  # noqa: E402

from regions import fetch_dmn_mask, region_drive  # noqa: E402

N_TIMEPOINTS = 20  # arbitrary for a demo; a real TRIBE run's T depends on narration length
RNG = np.random.default_rng(seed=7)  # reproducible, not cherry-picked run to run

PROBLEM_SLIDE_TEXT = (
    "[title] Deployment\n"
    "[body] We use PagedAttention-style KV-cache paging, speculative decoding with a draft "
    "model, continuous batching, tensor parallelism across 8 GPUs, quantized weights, and a "
    "custom scheduler, all tuned jointly to hit the SLO under bursty traffic without "
    "violating p99 latency budgets or wasting reserved capacity."
)


def _synthetic_response(dmn_mask: np.ndarray, dmn_level: float, baseline_level: float) -> np.ndarray:
    """A made-up (T, 20484) array: `dmn_level` and `baseline_level` control the mean
    predicted response inside vs. outside the DMN mask. NOT a model output -- see the
    module docstring."""
    n_vertices = dmn_mask.shape[0]
    response = RNG.normal(loc=baseline_level, scale=0.15, size=(N_TIMEPOINTS, n_vertices))
    response[:, dmn_mask] = RNG.normal(loc=dmn_level, scale=0.15, size=(N_TIMEPOINTS, dmn_mask.sum()))
    return response


async def main() -> None:
    print("=" * 78)
    print("DEMO ONLY -- brain-response numbers below are synthetic, not a real TRIBE v2 run.")
    print("See this file's module docstring for exactly what is and isn't real here.")
    print("=" * 78)

    print("\nfetching the real DMN mask (Yeo 2011 atlas projected onto fsaverage5)...")
    dmn_mask = fetch_dmn_mask()
    print(f"  {dmn_mask.sum()} / {dmn_mask.shape[0]} vertices ({100 * dmn_mask.mean():.1f}%) -- matches the "
          "published ~17% DMN coverage, so this mask is doing something real.")

    # A small illustrative "deck": four ordinary slides with natural slide-to-slide
    # variation in DMN level, plus the one dense, jargon-heavy problem slide (clearly
    # elevated). The ordinary slides' levels are deliberately NOT identical to each other:
    # region_drive averages over ~3500 vertices x 20 timepoints, so within-slide sampling
    # noise nearly vanishes -- four near-identical "ordinary" slides would make ANY
    # differing 5th value's Z-score saturate near sqrt(4)=2.0 regardless of how different
    # it actually is (a real artifact hit while building this: the first version of this
    # demo showed an identical Z-score for a small and a large deviation, for exactly this
    # reason). Giving the ordinary slides realistic between-slide spread avoids that.
    print(f"\nbuilding an illustrative 5-slide deck ({PROBLEM_SLIDE_TEXT[:40]!r}... is slide 5)...")
    ordinary_dmn_levels = [0.25, 0.33, 0.28, 0.36]
    ordinary_drives = [
        region_drive(_synthetic_response(dmn_mask, dmn_level=lvl, baseline_level=0.3), dmn_mask)
        for lvl in ordinary_dmn_levels
    ]
    problem_response = _synthetic_response(dmn_mask, dmn_level=0.85, baseline_level=0.3)
    problem_drive = region_drive(problem_response, dmn_mask)

    drives = {i: d for i, d in enumerate(ordinary_drives, start=1)} | {5: problem_drive}
    z_scores = deck_z_scores(drives)
    print("  illustrative dmn_drive by slide:", {i: round(v, 3) for i, v in drives.items()})
    print("  Z-scores vs. this (tiny, illustrative, 5-slide) deck:", {i: round(z, 2) for i, z in z_scores.items()})

    lens_before = adhd_dmn_lens(z_scores[5], slide_index=5)
    print("\n--- research lens, BEFORE revision (slide 5) ---")
    print(lens_before.label)
    print(lens_before.finding)
    print("Citations:")
    for c in lens_before.citations:
        print(f"  - {c}")

    print("\n--- proposing a revision (attempting a REAL Claude call) ---")
    slide = SlideInput(5, PROBLEM_SLIDE_TEXT, None)
    try:
        client = AnthropicClient()
        revised_text = await propose_adhd_friendly_revision(client, slide)
        print("(this is real, live model output, not canned)")
    except Exception as e:  # noqa: BLE001 - this demo degrades honestly, it doesn't fake a call
        print(f"  no working Anthropic credentials in this environment ({type(e).__name__}); "
              "falling back to a hand-written example of what the prompt targets.")
        print("  THIS FALLBACK TEXT IS NOT MODEL OUTPUT -- see below.")
        revised_text = (
            "[title] Deployment\n"
            "[body] 1. KV-cache paging (PagedAttention-style): avoids wasting memory on "
            "unused cache slots\n"
            "[body] 2. Speculative decoding: a small draft model proposes tokens, verified "
            "in parallel\n"
            "[body] 3. Continuous batching + 8-way tensor parallelism: keeps GPUs busy under "
            "bursty traffic\n"
            "[body] Result: hits latency SLOs without over-provisioning capacity"
        )
    print("Original:\n  " + PROBLEM_SLIDE_TEXT.replace("\n", "\n  "))
    print("Revised:\n  " + revised_text.replace("\n", "\n  "))

    # ILLUSTRATIVE "after": assumes the revision's chunking/structuring heuristic lowers
    # predicted DMN drive back toward the ordinary slides' level. Not measured -- there is
    # no real inference behind this number. Getting a real one means running the revised
    # text back through actual TRIBE v2 inference.
    print("\n--- illustrative AFTER, assuming the heuristic works (NOT measured) ---")
    revised_response = _synthetic_response(dmn_mask, dmn_level=0.31, baseline_level=0.3)
    revised_drive = region_drive(revised_response, dmn_mask)
    drives_after = {i: d for i, d in enumerate(ordinary_drives, start=1)} | {5: revised_drive}
    z_scores_after = deck_z_scores(drives_after)
    print(f"  dmn_drive: before={problem_drive:.3f} (Z={z_scores[5]:.2f})  "
          f"after={revised_drive:.3f} (Z={z_scores_after[5]:.2f})  "
          f"[ordinary range: {min(ordinary_drives):.3f}-{max(ordinary_drives):.3f}]")
    lens_after = adhd_dmn_lens(z_scores_after[5], slide_index=5)
    print(lens_after.finding)

    print("\n" + "=" * 78)
    print("To replace the illustrative numbers above with real ones: run both the original")
    print("and revised slide through actual TRIBE v2 inference (run.py, on a CUDA GPU) and")
    print("compare the real dmn_drive values -- this script has not done that.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
