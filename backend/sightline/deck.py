"""Per-slide result records and the deck-level rollup.

`SlideResult` is the unit the store persists, the server sends and the UI renders. It is a
plain JSON-shaped dict on purpose: every value on screen must be traceable to something in it,
so there is no second representation to drift out of sync.

`rollup` is arithmetic over already-computed slide results. It makes no model call and adds no
claim that is not a count, a rank, a quantile or a restatement of one. Notes (`DeckRollup
["notes"]`) are templated from those numbers and carry the evidence they were built from.

Two things this module deliberately does not do, both from the step 1 gate report:
  * It never turns a single slide's number into a verdict. Every metric is reported as a
    position inside this deck's own distribution (`distributions` / `per_slide[*].position`),
    because run-to-run swing on one slide (about +/-0.3 on the blind-spot score) is as large
    as the differences between slides. Trust separation between slides, never the level.
  * It never clamps or repairs a bad model response. A persona whose output failed validation
    is recorded as an error (`error_record`) and that slide gets no metrics.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping, Sequence, TypedDict

import numpy as np

from .audiences import PERSONAS, AudienceReading, AudienceResponseError, CacheMiss, Persona
from .divergence import normalize_term

# Comparing one slide to "the rest of the deck" needs a rest. Below this the UI says so
# instead of ranking three slides against each other.
MIN_SLIDES_FOR_COMPARISON = 5
# How many slides the "highest divergence relative to the deck" list names.
TOP_DIVERGENCE_SLIDES = 3
# A term is a deck-wide vocabulary problem once the novice fails on it on this many slides.
RECURRING_TERM_MIN_SLIDES = 2


# ------------------------------------------------------------------------- slide records


class SlideResult(TypedDict):
    index: int
    text: str  # what the personas were shown, verbatim
    # persona -> reading_record(...) or error_record(...); an "ok" flag tells them apart
    readings: dict[str, dict[str, Any]]
    # SlideDivergence.to_dict(), or None if any persona failed
    metrics: dict[str, Any] | None
    metrics_error: str | None
    scored_by: str | None  # name of the embedding model behind the metrics


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
    gap_ranking: list[dict[str, Any]]  # slides, widest novice-expert alignment gap first
    divergence_top: list[dict[str, Any]]  # highest divergence relative to this deck ([] if not comparable)
    terms: list[dict[str, Any]]  # novice-unresolved terms, most slides first
    arc: list[dict[str, Any]]  # alignment to intent per persona, in slide order
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


def _slide_values(r: SlideResult) -> dict[str, float]:
    """Every per-slide number the deck view compares, keyed by metric name."""
    out: dict[str, float] = {}
    for p in PERSONAS:
        reading = r["readings"].get(p, {})
        if reading.get("ok"):
            out[f"confidence.{p}"] = float(reading["confidence"])
            out[f"unresolved_count.{p}"] = float(len(reading["unresolved_terms"]))
    if "confidence.novice" in out and "confidence.expert" in out:
        out["confidence_gap"] = out["confidence.expert"] - out["confidence.novice"]
    m = r["metrics"]
    if m:
        out["audience_divergence"] = float(m["audience_divergence"]["value"])
        out["blind_spot_score"] = float(m["blind_spot_score"]["value"])  # expert - novice alignment
        out["term_gap_count"] = float(len(m["term_gap"]["terms"]))
        for p in PERSONAS:
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
    terms: list[dict[str, Any]], gap_ranking: list[dict[str, Any]], comparable: bool
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    recurring = [t for t in terms if t["count"] >= RECURRING_TERM_MIN_SLIDES]
    if recurring:
        slides = sorted({s for t in recurring for s in t["slides"]})
        first = min(t["first_slide"] for t in recurring)
        shown = [f"“{t['term']}”" for t in recurring[:5]]
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
    if comparable and gap_ranking:
        top = gap_ranking[:3]
        notes.append(
            {
                "id": "widest_gaps",
                "text": (
                    f"Slide{'s' if len(top) > 1 else ''} {_join([g['slide'] for g in top])} show"
                    f"{'' if len(top) > 1 else 's'} the widest novice–expert alignment gap in this deck. "
                    "That is a ranking within this deck, not a verdict on any one slide."
                ),
                "evidence": {"kind": "gap", "slides": [g["slide"] for g in top], "gaps": top},
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

    # blind_spot_score is expert alignment minus novice alignment, so this ranks the slides on
    # which the novice reads furthest from the intent relative to the expert.
    gap_ranking = sorted(
        (
            {"slide": i, "gap": v["blind_spot_score"], "rank": ranks["blind_spot_score"][i]}
            for i, v in values.items()
            if "blind_spot_score" in v
        ),
        key=lambda g: (-g["gap"], g["slide"]),
    )

    div_ranking = sorted(
        (
            {"slide": i, "value": v["audience_divergence"], "rank": ranks["audience_divergence"][i]}
            for i, v in values.items()
            if "audience_divergence" in v
        ),
        key=lambda d: (-d["value"], d["slide"]),
    )
    divergence_top = div_ranking[:TOP_DIVERGENCE_SLIDES] if comparable else []

    terms = _term_table(results)

    arc = [
        {
            "slide": r["index"],
            **{p: (values[r["index"]].get(f"intent_alignment.{p}")) for p in PERSONAS},
        }
        for r in results
    ]

    return {
        "n_slides": len(results),
        "n_scored": len(scored),
        "unscored": [r["index"] for r in results if not r["metrics"]],
        "comparable": comparable,
        "min_slides_for_comparison": MIN_SLIDES_FOR_COMPARISON,
        "distributions": distributions,
        "per_slide": per_slide,
        "gap_ranking": gap_ranking,
        "divergence_top": divergence_top,
        "terms": terms,
        "arc": arc,
        "notes": _notes(terms, gap_ranking, comparable),
    }
