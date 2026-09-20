#!/usr/bin/env python
"""DEMO, not a real run. Shows all three research lenses (ADHD, depression, dyslexia)
side by side on an illustrative deck, using MADE-UP brain-response numbers standing in for
a real TRIBE v2 prediction -- because TRIBE v2 itself has not run yet (no CUDA GPU / GX10
access at the time this was written; see scripts/precompute_neural/README.md).

What is REAL in this script:
  * Both vertex masks (regions.fetch_region_masks: language via Destrieux, DMN via Yeo 2011
    projected onto the actual fsaverage5 surface) -- the same masks a real precompute run
    uses for processing_ratio and dmn_drive.
  * The Z-score-against-the-deck math (sightline.neural.deck_z_scores).
  * All three citation-backed sentence generators (sightline.research_lens).
  * The slide revisions -- an ACTUAL call to Claude when credentials are available; this
    demo checks rather than assumes a key is configured (a real gap this script found in an
    earlier version: the app's own /api/health endpoint says "can_call_model: true" even
    with no working key, since it only checks that the client object constructs).

What is NOT real:
  * Every "predicted brain response" array. Synthetically constructed (see
    `_synthetic_response`) to illustrate two DIFFERENT kinds of problem slide next to a
    normal deck, and what each would look like if a literature-adjacent revision heuristic
    worked as hypothesized. Getting a real before/after needs an actual TRIBE v2 inference
    run on both slide versions, on a CUDA GPU.

What this demo actually shows, in order:
  1. A dense, jargon-heavy slide with elevated DMN drive -- flags BOTH the ADHD and
     depression lenses (they share a signal; see research_lens.py's module docstring for
     why that's by design, not a coincidence to paper over).
  2. A sparse, under-explained slide with low language-network drive -- flags the dyslexia
     lens, which runs the OPPOSITE statistical direction from the other two.
  3. A revision for each, and the illustrative (not measured) "after" numbers.

Run from an environment with this directory's requirements (or at minimum nilearn) plus
the main backend's dependencies (anthropic, pydantic, python-dotenv) installed:

    python demo_research_lenses.py
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
from sightline.research_lens import (  # noqa: E402
    adhd_dmn_lens,
    depression_dmn_lens,
    dyslexia_language_lens,
    propose_adhd_friendly_revision,
    propose_dyslexia_friendly_revision,
)

from regions import fetch_region_masks  # noqa: E402

N_TIMEPOINTS = 20  # arbitrary for a demo; a real TRIBE run's T depends on narration length
RNG = np.random.default_rng(seed=7)  # reproducible, not cherry-picked run to run

DENSE_SLIDE_TEXT = (
    "[title] Deployment\n"
    "[body] We use PagedAttention-style KV-cache paging, speculative decoding with a draft "
    "model, continuous batching, tensor parallelism across 8 GPUs, quantized weights, and a "
    "custom scheduler, all tuned jointly to hit the SLO under bursty traffic without "
    "violating p99 latency budgets or wasting reserved capacity."
)
SPARSE_SLIDE_TEXT = "[title] Results\n[body] Goodput: 2.4x. p99: within SLO."


def _region_mean(response: np.ndarray, mask: np.ndarray) -> float:
    return float(response[:, mask].mean())


def _synthetic_response(masks: dict, dmn_level: float, language_level: float, baseline_level: float) -> np.ndarray:
    """A made-up (T, 20484) array with independently controllable DMN and language levels
    -- NOT a model output. See the module docstring."""
    n_vertices = masks["dmn"].shape[0]
    response = RNG.normal(loc=baseline_level, scale=0.15, size=(N_TIMEPOINTS, n_vertices))
    response[:, masks["dmn"]] = RNG.normal(loc=dmn_level, scale=0.15, size=(N_TIMEPOINTS, masks["dmn"].sum()))
    response[:, masks["language"]] = RNG.normal(
        loc=language_level, scale=0.15, size=(N_TIMEPOINTS, masks["language"].sum())
    )
    return response


async def _revise(prompt_fn, client_available: bool, slide: SlideInput, fallback_text: str, fallback_reason: str) -> tuple[str, bool]:
    if not client_available:
        print(f"  no working Anthropic credentials; using a hand-written fallback ({fallback_reason}).")
        print("  THIS FALLBACK TEXT IS NOT MODEL OUTPUT.")
        return fallback_text, False
    try:
        client = AnthropicClient()
        revised = await prompt_fn(client, slide)
        print("  (this is real, live model output, not canned)")
        return revised, True
    except Exception as e:  # noqa: BLE001 - this demo degrades honestly, it doesn't fake a call
        print(f"  Claude call failed ({type(e).__name__}); using a hand-written fallback ({fallback_reason}).")
        print("  THIS FALLBACK TEXT IS NOT MODEL OUTPUT.")
        return fallback_text, False


async def main() -> None:
    print("=" * 78)
    print("DEMO ONLY -- brain-response numbers below are synthetic, not a real TRIBE v2 run.")
    print("See this file's module docstring for exactly what is and isn't real here.")
    print("=" * 78)

    print("\nfetching the real region masks (Destrieux language mask + Yeo-2011-on-fsaverage5 DMN mask)...")
    masks = fetch_region_masks()
    print(f"  language: {masks['language'].sum()} vertices  |  DMN: {masks['dmn'].sum()} vertices "
          f"({100 * masks['dmn'].mean():.1f}% -- matches the published ~17% DMN coverage)")

    # An illustrative 5-slide deck: 3 ordinary slides with realistic between-slide spread on
    # BOTH signals, plus the two problem slides. Ordinary slides deliberately are NOT
    # identical to each other -- see demo_adhd_lens.py's original version (superseded by
    # this file) for the small-N Z-score artifact that caused when they were.
    print("\nbuilding an illustrative 5-slide deck...")
    ordinary = [
        _synthetic_response(masks, dmn_level=lvl_d, language_level=lvl_l, baseline_level=0.3)
        for lvl_d, lvl_l in [(0.25, 0.55), (0.33, 0.62), (0.28, 0.58)]
    ]
    dense_response = _synthetic_response(masks, dmn_level=0.85, language_level=0.58, baseline_level=0.3)
    sparse_response = _synthetic_response(masks, dmn_level=0.3, language_level=0.18, baseline_level=0.3)
    all_responses = {1: ordinary[0], 2: ordinary[1], 3: ordinary[2], 4: dense_response, 5: sparse_response}

    dmn_drives = {i: _region_mean(r, masks["dmn"]) for i, r in all_responses.items()}
    language_drives = {i: _region_mean(r, masks["language"]) for i, r in all_responses.items()}
    dmn_z = deck_z_scores(dmn_drives)
    language_z = deck_z_scores(language_drives)
    print("  dmn_drive Z-scores:     ", {i: round(z, 2) for i, z in dmn_z.items()})
    print("  language_drive Z-scores:", {i: round(z, 2) for i, z in language_z.items()})

    client_available = False
    try:
        AnthropicClient()  # construction alone doesn't prove a working key -- see above
        import os

        client_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    except Exception:
        pass

    # ----------------------------------------------------------- slide 4: dense/jargon-heavy
    print("\n" + "-" * 78)
    print("SLIDE 4 (dense, jargon-heavy) -- elevated DMN drive")
    print("-" * 78)
    adhd = adhd_dmn_lens(dmn_z[4], slide_index=4)
    depression = depression_dmn_lens(dmn_z[4], slide_index=4)
    print(f"\n[{adhd.population}] {adhd.finding}")
    print(f"\n[{depression.population}] {depression.finding}")
    print("\nBoth lenses fired from the SAME dmn_drive_z -- independently cited, not stacked evidence.")

    print("\nproposing a revision (attempting a REAL Claude call)...")
    dense_fallback = (
        "[title] Deployment\n"
        "[body] 1. KV-cache paging (PagedAttention-style): avoids wasting memory on unused cache slots\n"
        "[body] 2. Speculative decoding: a small draft model proposes tokens, verified in parallel\n"
        "[body] 3. Continuous batching + 8-way tensor parallelism: keeps GPUs busy under bursty traffic\n"
        "[body] Result: hits latency SLOs without over-provisioning capacity"
    )
    revised_dense, _ = await _revise(
        propose_adhd_friendly_revision, client_available, SlideInput(4, DENSE_SLIDE_TEXT, None),
        dense_fallback, "chunked into numbered steps",
    )
    print("Original:\n  " + DENSE_SLIDE_TEXT.replace("\n", "\n  "))
    print("Revised:\n  " + revised_dense.replace("\n", "\n  "))

    revised_dense_response = _synthetic_response(masks, dmn_level=0.31, language_level=0.58, baseline_level=0.3)
    dmn_drives_after = dict(dmn_drives) | {4: _region_mean(revised_dense_response, masks["dmn"])}
    dmn_z_after = deck_z_scores(dmn_drives_after)
    print(f"\nillustrative (NOT measured) AFTER: dmn_drive Z {dmn_z[4]:.2f} -> {dmn_z_after[4]:.2f}")
    print(adhd_dmn_lens(dmn_z_after[4], slide_index=4).finding)

    # ------------------------------------------------------------- slide 5: sparse/underexplained
    print("\n" + "-" * 78)
    print("SLIDE 5 (sparse, under-explained) -- low language-network drive")
    print("-" * 78)
    dyslexia = dyslexia_language_lens(language_z[5], slide_index=5)
    print(f"\n[{dyslexia.population}] {dyslexia.finding}")

    print("\nproposing a revision (attempting a REAL Claude call)...")
    sparse_fallback = (
        "[title] Results\n"
        "[body] The system now handles 2.4 times more requests per second.\n"
        "[body] It still answers within the required time limit (p99 latency, the slowest "
        "1 in 100 requests) even under heavy load."
    )
    revised_sparse, _ = await _revise(
        propose_dyslexia_friendly_revision, client_available, SlideInput(5, SPARSE_SLIDE_TEXT, None),
        sparse_fallback, "spelled out the abbreviations in plain words",
    )
    print("Original:\n  " + SPARSE_SLIDE_TEXT.replace("\n", "\n  "))
    print("Revised:\n  " + revised_sparse.replace("\n", "\n  "))

    revised_sparse_response = _synthetic_response(masks, dmn_level=0.3, language_level=0.56, baseline_level=0.3)
    language_drives_after = dict(language_drives) | {5: _region_mean(revised_sparse_response, masks["language"])}
    language_z_after = deck_z_scores(language_drives_after)
    print(f"\nillustrative (NOT measured) AFTER: language_drive Z {language_z[5]:.2f} -> {language_z_after[5]:.2f}")
    print(dyslexia_language_lens(language_z_after[5], slide_index=5).finding)

    print("\n" + "=" * 78)
    print("To replace the illustrative numbers above with real ones: run all four slide")
    print("versions through actual TRIBE v2 inference (run.py, on a CUDA GPU) and compare")
    print("the real dmn_drive/language_drive values -- this script has not done that.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
