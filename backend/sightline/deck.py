"""Per-slide result records and the deck-level rollup.

`SlideResult` is the unit the store persists, the server sends and the UI renders. It is a
plain JSON-shaped dict on purpose: every value on screen must be traceable to something in it,
so there is no second representation to drift out of sync.

`rollup` is arithmetic over already-computed slide results. It makes no model call and adds no
claim that is not a count, a rank, a quantile or a restatement of one. Notes (`DeckRollup
["notes"]`) are templated from those numbers and carry the evidence they were built from.

Three things this module deliberately does not do:
  * It never turns a single slide's number into a verdict. Every metric is reported as a
    position inside this deck's own distribution (`distributions` / `per_slide[*].position`),
    because the level on one slide swings between runs by about as much as slides differ from
    each other. Trust separation between slides, never the level.
  * It never clamps or repairs a bad model response. A persona whose output failed validation
    is recorded as an error (`error_record`) and that slide gets no metrics.
  * It never scores the expert against the reference. Alignment is measured against an intent
    inferred from the expert's own reading, so the expert's alignment is 1.0 by construction:
    it is carried in the payload flagged `definitional`, and excluded from the distributions.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np

from .audiences import PERSONAS, AudienceReading, AudienceResponseError, CacheMiss, Persona
from .divergence import Embedder, normalize_term, score_slide
from .intent import EXPERT_IS_DEFINITIONAL

# Comparing one slide to "the rest of the deck" needs a rest. Below this the UI says so
# instead of ranking three slides against each other.
MIN_SLIDES_FOR_COMPARISON = 5
# How many slides the "hardest for a newcomer" note names.
TOP_HARDEST_SLIDES = 3
# A term is a deck-wide vocabulary problem once the novice fails on it on this many slides.
RECURRING_TERM_MIN_SLIDES = 2


# ------------------------------------------------------------------------- slide records


class SlideResult(TypedDict):
    index: int
    text: str  # what the personas were shown, verbatim
    # persona -> reading_record(...) or error_record(...); an "ok" flag tells them apart
    readings: dict[str, dict[str, Any]]
    # the intent inferred from the expert's reading (intent.SlideIntent.to_dict()); None if the
    # expert's reading is unavailable
    slide_intent: dict[str, Any] | None
    # build_metrics(...), or None if any persona failed
    metrics: dict[str, Any] | None
    metrics_error: str | None
    scored_by: str | None  # name of the embedding model behind the metrics
    timing: dict[str, float]  # seconds: personas, intent


def plain(obj: Any) -> Any:
    """Round-trip through JSON so a record held in memory is identical to the one reloaded from
    disk (tuples become lists, numpy scalars fail loudly here rather than in the server)."""
    return json.loads(json.dumps(obj))


def reading_record(r: AudienceReading) -> dict[str, Any]:
    return {
        "ok": True,
        **r.response.model_dump(),
        "model": r.model,
        "cached": r.cached,
        "latency_s": round(r.latency_s, 3),
        "attempts": r.attempts,
        "slide_hash": r.slide_hash,
    }


def error_record(exc: BaseException) -> dict[str, Any]:
    """A persona that produced no usable reading. The reason is kept verbatim so a model that
    misread the task (for instance a confidence outside [0, 1]) is visible, not smoothed over."""
    if isinstance(exc, AudienceResponseError):
        return {
            "ok": False,
            "error": {"kind": "invalid_response", "message": exc.attempts[-1], "attempts": list(exc.attempts)},
        }
    if isinstance(exc, CacheMiss):
        return {"ok": False, "error": {"kind": "offline_cache_miss", "message": str(exc), "attempts": []}}
    return {
        "ok": False,
        "error": {"kind": "call_failed", "message": f"{type(exc).__name__}: {exc}", "attempts": []},
    }


def build_metrics(
    intent: str, responses: Mapping[Persona, Any], embedder: Embedder, slide_index: int
) -> dict[str, Any]:
    """Alignment of each persona's takeaway to the inferred intent, via step 1's `score_slide`
    (the reference string is the only thing that changed). Its divergence, pairwise and
    blind-spot outputs are dropped: blind-spot, expert minus novice, reduces to 1 - novice
    alignment once the expert is the reference, so it carries nothing the novice alignment
    does not, and raw divergence is not legible to a presenter.

    The expert's alignment is not the cosine `score_slide` computed. It is 1.0 by definition."""
    sd = score_slide(intent, responses, embedder, slide_index)
    alignment = {p: asdict(sd.intent_alignment[p]) for p in PERSONAS}
    if EXPERT_IS_DEFINITIONAL:
        alignment["expert"] = {
            "value": 1.0,
            "definitional": True,
            "inputs": {"intent": intent, "expert": sd.takeaways["expert"]},
        }
    return plain(
        {
            "intent": intent,
            "takeaways": dict(sd.takeaways),
            "intent_alignment": alignment,
            "term_gap": asdict(sd.term_gap),
        }
    )


# ---------------------------------------------------------------------------- the rollup


class DeckRollup(TypedDict):
    n_slides: int
    n_scored: int
    unscored: list[int]  # slide indices with no metrics (a persona failed)
    comparable: bool  # enough slides for "relative to the deck" to mean anything
    min_slides_for_comparison: int
    # metric key -> {n, min, q1, median, q3, max, comparable, values: [{slide, value}]}
    distributions: dict[str, dict[str, Any]]
    # one row per slide: raw values and rank inside each distribution
    per_slide: list[dict[str, Any]]
    hardest: list[dict[str, Any]]  # slides, hardest for the novice first
    terms: list[dict[str, Any]]  # novice-unresolved terms, most slides first
    arc: list[dict[str, Any]]  # alignment to the inferred intent per persona, in slide order
    definitional: list[str]  # personas in `arc` that are the reference, not measured
    notes: list[dict[str, Any]]  # restatements of the above; each carries its evidence


def _distribution(values: Mapping[int, float]) -> dict[str, Any]:
    vs = np.array(list(values.values()), dtype=float)
    q1, med, q3 = (float(x) for x in np.percentile(vs, [25, 50, 75]))
    return {
        "n": len(vs),
        "min": float(vs.min()),
        "q1": q1,
        "median": med,
        "q3": q3,
        "max": float(vs.max()),
        "comparable": len(vs) >= MIN_SLIDES_FOR_COMPARISON,
        "values": [{"slide": i, "value": float(v)} for i, v in sorted(values.items())],
    }


def _ranks(values: Mapping[int, float]) -> dict[int, int]:
    """1 = highest. Ties share a rank."""
    return {i: 1 + sum(1 for w in values.values() if w > v) for i, v in values.items()}


def _definitional(m: Mapping[str, Any], persona: str) -> bool:
    return bool(m["intent_alignment"][persona].get("definitional"))


def _slide_values(r: SlideResult) -> dict[str, float]:
    """Every per-slide number the deck view compares, keyed by metric name. A persona whose
    alignment is definitional (the reference) has no measured value and is left out."""
    out: dict[str, float] = {}
    for p in PERSONAS:
        reading = r["readings"].get(p, {})
        if reading.get("ok"):
            out[f"unresolved_count.{p}"] = float(len(reading["unresolved_terms"]))
    m = r["metrics"]
    if m:
        out["term_gap_count"] = float(len(m["term_gap"]["terms"]))
        for p in PERSONAS:
            if not _definitional(m, p):
                out[f"intent_alignment.{p}"] = float(m["intent_alignment"][p]["value"])
    return out


def _term_table(results: Sequence[SlideResult]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for r in results:
        novice = r["readings"].get("novice", {})
        if not novice.get("ok"):
            continue
        for raw in novice["unresolved_terms"]:
            key = normalize_term(raw)
            if not key:
                continue
            entry = by_key.setdefault(key, {"key": key, "occurrences": []})
            if all(o["slide"] != r["index"] for o in entry["occurrences"]):
                entry["occurrences"].append({"slide": r["index"], "as_written": raw})
    table = []
    for e in by_key.values():
        spellings = Counter(o["as_written"] for o in e["occurrences"])
        slides = [o["slide"] for o in e["occurrences"]]
        table.append(
            {
                "term": spellings.most_common(1)[0][0],
                "key": e["key"],
                "count": len(slides),
                "slides": slides,
                "first_slide": min(slides),
                "occurrences": e["occurrences"],
            }
        )
    table.sort(key=lambda t: (-t["count"], t["first_slide"], t["key"]))
    return table


def _join(items: Sequence[Any]) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _notes(
    terms: list[dict[str, Any]], hardest: list[dict[str, Any]], comparable: bool
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    recurring = [t for t in terms if t["count"] >= RECURRING_TERM_MIN_SLIDES]
    if recurring:
        slides = sorted({s for t in recurring for s in t["slides"]})
        first = min(t["first_slide"] for t in recurring)
        shown = [f"\u201c{t['term']}\u201d" for t in recurring[:5]]
        more = f" and {len(recurring) - 5} more" if len(recurring) > 5 else ""
        notes.append(
            {
                "id": "recurring_terms",
                "text": (
                    f"{len(slides)} slides share {len(recurring)} terms the novice could not resolve "
                    f"({_join(shown)}{more}). Define them once, early; the first is needed by slide {first}."
                ),
                "evidence": {
                    "kind": "terms",
                    "terms": [{k: t[k] for k in ("term", "count", "slides", "occurrences")} for t in recurring],
                    "slides": slides,
                },
            }
        )
    if comparable and hardest:
        top = hardest[:TOP_HARDEST_SLIDES]
        notes.append(
            {
                "id": "hardest_slides",
                "text": (
                    f"Slide{'s' if len(top) > 1 else ''} {_join([h['slide'] for h in top])} "
                    f"{'are' if len(top) > 1 else 'is'} the hardest for a newcomer in this deck. "
                    "That is a ranking within this deck, not a verdict on any one slide."
                ),
                "evidence": {"kind": "hardest", "slides": [h["slide"] for h in top], "rows": top},
            }
        )
    return notes


def rollup(results: Sequence[SlideResult]) -> DeckRollup:
    results = sorted(results, key=lambda r: r["index"])
    values = {r["index"]: _slide_values(r) for r in results}
    keys = sorted({k for v in values.values() for k in v})

    distributions: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[int, int]] = {}
    for k in keys:
        series = {i: v[k] for i, v in values.items() if k in v}
        distributions[k] = _distribution(series)
        ranks[k] = _ranks(series)

    per_slide = [
        {
            "slide": r["index"],
            "values": values[r["index"]],
            "position": {
                k: {"rank": ranks[k][r["index"]], "of": distributions[k]["n"]}
                for k in keys
                if r["index"] in ranks[k]
            },
        }
        for r in results
    ]

    scored = [r for r in results if r["metrics"]]
    comparable = len(scored) >= MIN_SLIDES_FOR_COMPARISON

    # Hardest for a newcomer: lowest novice alignment to the slide's inferred intent first, ties
    # broken by more novice-unresolved terms. Alignment is not shown in the row: the order is the
    # finding, and the level on one slide is not trustworthy.
    hardest = sorted(
        (
            {
                "slide": i,
                "novice_alignment": v["intent_alignment.novice"],
                "novice_unresolved": int(v["unresolved_count.novice"]),
            }
            for i, v in values.items()
            if "intent_alignment.novice" in v
        ),
        key=lambda h: (h["novice_alignment"], -h["novice_unresolved"], h["slide"]),
    )
    for n, h in enumerate(hardest, start=1):
        h["rank"] = n

    terms = _term_table(results)

    arc = []
    for r in results:
        m = r["metrics"]
        arc.append({"slide": r["index"], **{p: (float(m["intent_alignment"][p]["value"]) if m else None) for p in PERSONAS}})
    definitional = [p for p in PERSONAS if scored and all(_definitional(r["metrics"], p) for r in scored)]

    return {
        "n_slides": len(results),
        "n_scored": len(scored),
        "unscored": [r["index"] for r in results if not r["metrics"]],
        "comparable": comparable,
        "min_slides_for_comparison": MIN_SLIDES_FOR_COMPARISON,
        "distributions": distributions,
        "per_slide": per_slide,
        "hardest": hardest,
        "terms": terms,
        "arc": arc,
        "definitional": definitional,
        "notes": _notes(terms, hardest, comparable),
    }
