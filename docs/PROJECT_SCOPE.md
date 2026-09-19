# SIGHTLINE — Project Scope

_HackMIT 2026 · 9.19–9.20 · Active development on `tribev2-implementation`_

This is a working snapshot of what SIGHTLINE is, what's built, and what's left — derived
from `docs/SIGHTLINE_spec.md` and a read of the codebase as of the `tribev2-implementation`
branch. Treat the spec as the source of truth for intent; treat this as the source of truth
for **status**.

## 1. What it is

SIGHTLINE reads a presentation the way three different audiences would. Three simulated
personas — **novice**, **peer**, **expert** — differ *only* in prior knowledge. Each reads
every slide independently and reports what they took away. A metrics layer measures how far
apart those readings are, and how far each is from what the presenter says they meant.

**One sentence:** we show you the wrong conclusion your audience will draw, and which slide
caused it.

This is not a delivery/clarity coach (filler words, pacing, eye contact — a crowded,
commodity category already served by Yoodli, Orai, PowerPoint Speaker Coach, etc.). The
differentiated claim is the divergence measurement itself: no existing tool checks whether
*different viewers* extract *different points* from the same slide.

## 2. Non-negotiable design rules

These come from the spec and constrain every feature below. Violating one is a bug, not a
style choice.

1. **Audiences perform, never rate.** No `clarity: 7/10` anywhere. Output is a takeaway, a
   confidence, and unresolved terms — all checkable against the slide text.
2. **Every number decomposes into its source text.** Clicking a score must show the raw
   takeaways that produced it.
3. **No scalar engagement metric, ever.** A July 2026 paper (arXiv 2607.01400) showed the
   published scalar readout from brain-encoding models is a null result against real
   attention data. If a single "engagement score" appears anywhere, it's a bug.
4. **Neural output is labeled "predicted — simulated, not measured"** on the image itself,
   not tucked into a tab.
5. **Ratios within a deck, never absolutes across decks.** A slide is ranked against the
   rest of its own deck; a bare number never implies a good/bad direction on its own.

## 3. Current status

Status is reported against the spec's own build order (§7), which is the intended sequence.

| # | Component | Status | Notes |
|---|---|---|---|
| 1 | `audiences.py` + `divergence.py` (+ tests) | ✅ Done | Three personas, strict-JSON schema, per-persona memory, file cache keyed by `(slide_hash, persona, context_hash)`. Divergence metrics: `intent_alignment`, `audience_divergence` (headline), `blind_spot_score`, `term_gap`. |
| 2 | Slide detail screen, PDF ingest | ✅ Done | `ingest.py` (pymupdf) — **PDF only**. Slide detail UI shows three audience cards + cross-audience metrics, all click-through to source text. |
| 3 | `saliency.py` + `scanpath.py` | ❌ Not started | Bottom-up attention (SpectralResidual/DeepGaze/Stub) and greedy-max scanpath — no code exists. |
| 4 | `diagnose.py` rules | 🟡 Contract only | `FIXTURE_BACKED = True`; returns the same checked-in fixture for every slide. UI badges it `sample data`. No real recommendation engine yet. |
| 5 | Blind-spot view | 🟡 Partial | Gap-ranking and divergence-ranking exist inside the **deck overview** page, not as the spec's dedicated screen. |
| 6 | `/validate` route | ❌ Not started | No human-validation endpoint or `validation/{deck_id}.json` logging. |
| 7 | pptx ingest | ❌ Not started | `ingest.py` explicitly rejects `.pptx` today (tested behavior, not an oversight). |
| 8 | `fix.py` (revise-and-rescore loop) | ❌ Not started | No before/after Fix view. |
| 9 | Neural layer (TRIBE, precomputed) | ❌ Not started | No `NeuralModel` protocol, no cached-neural path, no cortical-surface rendering. |

**Also not yet built:** a standalone Methods panel screen (spec §10's required disclosures
aren't surfaced anywhere in the UI yet).

**What's solid:** the entire live path — the actual product, per the spec's own framing
("the live path is the product; the neural layer is the proof-of-concept centerpiece"). Deck
upload → subfield inference → run → streaming per-slide results → deck rollup (distributions,
ranks, recurring-term table, narrative arc) all work end-to-end, backed by real tests
(`backend/tests/`) and a bundled offline sample run for demo-safety.

## 4. Architecture

```
backend/sightline/
  audiences.py    3 personas, prompting, caching, isolation guarantees
  divergence.py   embedding-based metrics (local sentence-transformers, no API call)
  ingest.py       PDF → per-slide text + rendered PNG; one-shot subfield inference
  deck.py         pure-arithmetic deck rollup (no model calls)
  runner.py       runs a stored deck slide-by-slide, tolerant of a single persona failing
  store.py        plain-filesystem run history (data/history/<run_id>/), no DB
  server.py       FastAPI: upload, start, poll (streaming = polling), replay
  diagnose.py     recommendations CONTRACT ONLY — fixture-backed stub
frontend/
  index.html, js/*.js, style.css   plain HTML/ES modules, no build step
```

Tech stack per spec §8: Python FastAPI + plain JS frontend (spec says React/Vite; current
implementation is vanilla JS/ES modules instead — a deviation worth flagging if the spec is
being treated as binding). Saliency/neural stack described in the spec (OpenCV
SpectralResidual, DeepGaze IIE, nilearn/TRIBE) is entirely unimplemented.

**Windows note:** the `Makefile` assumes macOS/Linux (`.venv/bin`, `python3.11`, `chflags`)
and will need adaptation to run natively here.

## 5. Out of scope / explicitly deferred

- A scalar "engagement score" of any kind — permanently out of scope, not just deferred (see
  rule 3 above).
- Live neural inference — 6–13 min GPU/slide makes this structurally impossible; precomputed
  only, on two bundled sample decks.
- Validating simulated audiences against real humans beyond the lightweight on-site protocol
  in spec §6 (two slides, ~8 hackers who know the topic, ~8 who don't).

## 6. Suggested next step

Per the spec's cut-order guidance ("cut from the bottom... never the audiences"), and given
steps 1–2 are done, the natural next build-order items are **saliency.py + scanpath.py**
(step 3) or hardening the Blind-spot view into its own screen (step 5) before touching
anything neural or fix-loop related.
