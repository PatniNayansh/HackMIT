"""Three-tier reading of what a slide demands of its reader.

    Self-contained     A newcomer recovers the intended point.
    Background needed  Needs some familiarity with the field; a newcomer drifts.
    Expert-gated       Only a specialist recovers the intended point.

Inputs, in order of weight: the novice's alignment to the slide's inferred intent, the peer's
alignment to it, and the novice's unresolved-term count.

Position comes first from the DECK: each input is turned into a percentile among this deck's
scored slides, and the thresholds below apply to that percentile. The step 1 gate found that
separation between slides is trustworthy and the level on one slide is not, so a tier is a
statement about where a slide sits in its own deck. Only when there are too few slides to
compare (a single-slide view, or a deck early in a run) do the absolute thresholds apply, and the
record says which basis it used.

EVERY THRESHOLD LIVES IN `CONFIG`, below. Tune them there against real decks; nothing else in
the code holds a cut-off.
"""

from __future__ import annotations

from typing import Any, Mapping

CONFIG: dict[str, Any] = {
    # Compare a slide with its deck only once the deck has this many scored slides.
    "min_slides_for_relative": 5,
    # Relative thresholds: percentile within this deck's scored slides, 0 (lowest) to 1 (highest).
    "relative": {
        # The novice "aligns" if their alignment is at or above this percentile: the upper half.
        "novice_aligned_min_pct": 0.50,
        # The peer "aligns" unless they sit in the deck's lowest fifth.
        "peer_aligned_min_pct": 0.20,
        # Novice unresolved terms are "low" at or below this percentile of the deck's counts.
        "novice_terms_low_max_pct": 0.50,
    },
    # Absolute fallback, used only when the deck is too small to compare with itself. Cosine
    # levels on one slide move between runs, so treat these as a rough single-slide guide.
    "absolute": {
        "novice_aligned_min": 0.55,
        "peer_aligned_min": 0.40,
        "novice_terms_low_max": 2,
    },
}

# Field-wise comparator (compare.py). The claim's entailment state and the concept/result checks are
# categorical, so they are read directly; only the novice's unresolved-term count is a level, and it
# keeps the deck-relative reading (percentile among the deck's scored slides) with an absolute fallback.
CONFIG["fieldwise"] = {
    "min_slides_for_relative": 5,
    "relative": {"novice_terms_low_max_pct": 0.50},
    "absolute": {"novice_terms_low_max": 2},
}

TIERS: dict[str, dict[str, str]] = {
    "self_contained": {
        "label": "Self-contained",
        "meaning": "A newcomer recovers the intended point.",
    },
    "background_needed": {
        "label": "Background needed",
        "meaning": "Needs some familiarity with the field; a newcomer drifts.",
    },
    "expert_gated": {
        "label": "Expert-gated",
        "meaning": "Only a specialist recovers the intended point.",
    },
}


def percentile(value: float, population: list[float]) -> float:
    """Where `value` sits among `population`, 0 to 1: the share of the population below it, with
    ties counted as half. Higher is higher. A lone value is 0.5."""
    less = sum(1 for w in population if w < value)
    equal = sum(1 for w in population if w == value)
    return (less + 0.5 * equal) / len(population)


def _ordinal(pct: float) -> str:
    n = round(pct * 100)
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _decide(novice_aligned: bool, peer_aligned: bool, terms_low: bool) -> str:
    if not novice_aligned and not peer_aligned:
        return "expert_gated"
    if novice_aligned and peer_aligned and terms_low:
        return "self_contained"
    return "background_needed"  # the peer aligns and the novice does not, or the novice aligns but is left with many unknown terms


def assign_tiers(
    inputs: Mapping[int, Mapping[str, float]], config: Mapping[str, Any] = CONFIG
) -> dict[int, dict[str, Any]]:
    """`inputs[slide]` = {novice_alignment, peer_alignment, novice_unresolved}. Returns a tier
    record per slide: the tier, its label and meaning, the basis (relative or absolute), and
    every check with the value, the threshold it met or missed, and a plain-words line."""
    relative = len(inputs) >= config["min_slides_for_relative"]
    pops = {k: [v[k] for v in inputs.values()] for k in ("novice_alignment", "peer_alignment", "novice_unresolved")}
    out: dict[int, dict[str, Any]] = {}
    for slide, v in inputs.items():
        checks: list[dict[str, Any]] = []
        if relative:
            r = config["relative"]
            n_pct = percentile(v["novice_alignment"], pops["novice_alignment"])
            p_pct = percentile(v["peer_alignment"], pops["peer_alignment"])
            t_pct = percentile(v["novice_unresolved"], pops["novice_unresolved"])
            n_ok, p_ok, t_ok = n_pct >= r["novice_aligned_min_pct"], p_pct >= r["peer_aligned_min_pct"], t_pct <= r["novice_terms_low_max_pct"]
            n = len(inputs)
            checks = [
                {"name": "novice_aligned", "passed": n_ok, "value": v["novice_alignment"], "percentile": n_pct,
                 "threshold": r["novice_aligned_min_pct"],
                 "text": f"Novice alignment is at the {_ordinal(n_pct)} percentile of these {n} slides; it counts as aligned at the {_ordinal(r['novice_aligned_min_pct'])} or above."},
                {"name": "peer_aligned", "passed": p_ok, "value": v["peer_alignment"], "percentile": p_pct,
                 "threshold": r["peer_aligned_min_pct"],
                 "text": f"Peer alignment is at the {_ordinal(p_pct)} percentile of these {n} slides; it counts as aligned at the {_ordinal(r['peer_aligned_min_pct'])} or above."},
                {"name": "novice_terms_low", "passed": t_ok, "value": v["novice_unresolved"], "percentile": t_pct,
                 "threshold": r["novice_terms_low_max_pct"],
                 "text": f"The novice’s unresolved-term count is at the {_ordinal(t_pct)} percentile of these {n} slides; it counts as low at the {_ordinal(r['novice_terms_low_max_pct'])} or below."},
            ]
        else:
            a = config["absolute"]
            n_ok, p_ok, t_ok = v["novice_alignment"] >= a["novice_aligned_min"], v["peer_alignment"] >= a["peer_aligned_min"], v["novice_unresolved"] <= a["novice_terms_low_max"]
            checks = [
                {"name": "novice_aligned", "passed": n_ok, "value": v["novice_alignment"], "threshold": a["novice_aligned_min"],
                 "text": f"Novice alignment counts as aligned at {a['novice_aligned_min']:.2f} or above."},
                {"name": "peer_aligned", "passed": p_ok, "value": v["peer_alignment"], "threshold": a["peer_aligned_min"],
                 "text": f"Peer alignment counts as aligned at {a['peer_aligned_min']:.2f} or above."},
                {"name": "novice_terms_low", "passed": t_ok, "value": v["novice_unresolved"], "threshold": a["novice_terms_low_max"],
                 "text": f"The novice’s unresolved-term count counts as low at {a['novice_terms_low_max']} or fewer."},
            ]
        key = _decide(n_ok, p_ok, t_ok)
        out[slide] = {
            "tier": key,
            **TIERS[key],
            "basis": "relative" if relative else "absolute",
            "n_slides": len(inputs),
            "min_slides_for_relative": config["min_slides_for_relative"],
            "checks": checks,
        }
    return out


_REACH_WORDS = {"ok": "reached it", "partial": "reached part of it", "fail": "did not reach it"}


def _field_line(persona: str, field: str, comp: Mapping[str, Any]) -> tuple[bool, str]:
    """(counts as reached, plain-words line) for one scored field of one audience."""
    who = persona.capitalize()
    exp, aud = comp.get("expert"), comp.get("audience")
    o = comp.get("outcome")
    if field == "claim":
        line = {
            "equivalent": f"{who} claim is equivalent to the expert's.",
            "over-claimed": f"{who} claim over-generalises the expert's (it implies the expert's).",
            "under-specified": f"{who} claim is a weaker version of the expert's (under-specified).",
            "divergent": f"{who} claim and the expert's do not imply each other (divergent).",
            "absent": f"{who} takeaway stated no general claim (absent).",
        }[o]
        return o in ("equivalent", "over-claimed"), line
    if o == "absent":
        return False, f"{who} takeaway did not reach the {field}; the expert reached \u201c{exp}\u201d."
    return o in ("match", "near"), (
        f"{who} {field} \u201c{aud}\u201d {'matches' if o in ('match', 'near') else 'does not match'} the expert\u2019s \u201c{exp}\u201d."
    )


def assign_tiers_fieldwise(
    inputs: Mapping[int, Mapping[str, Any]], config: Mapping[str, Any] = CONFIG
) -> dict[int, dict[str, Any]]:
    """`inputs[slide]` = {profile, comparisons (novice, peer), reach (novice, peer), novice_unresolved}.
    A slide whose profile has no scored field has no structure to judge by and gets no tier.
    Everything runs only over the fields the expert populated (the slide profile)."""
    cfg = config["fieldwise"]
    relative = len(inputs) >= cfg["min_slides_for_relative"]
    counts = [v["novice_unresolved"] for v in inputs.values()]
    out: dict[int, dict[str, Any]] = {}
    for slide, v in inputs.items():
        profile = v["profile"]
        if not profile["scored"]:
            continue
        checks: list[dict[str, Any]] = []
        for persona in ("novice", "peer"):
            for f in profile["scored"]:
                ok, line = _field_line(persona, f, v["comparisons"][persona][f])
                checks.append({"name": f"{persona}_{f}", "persona": persona, "field": f, "passed": ok, "text": line})
        n_ok = v["reach"]["novice"] == "ok"
        p_ok = v["reach"]["peer"] in ("ok", "partial")
        checks.insert(0, {"name": "novice_reached", "passed": n_ok, "text": f"Over this slide's scored fields ({', '.join(profile['scored'])}), the novice {_REACH_WORDS[v['reach']['novice']]}." + (" A novice counts as aligned only if all were reached." if not n_ok else "")})
        checks.insert(1, {"name": "peer_reached", "passed": p_ok, "text": f"The peer {_REACH_WORDS[v['reach']['peer']]}. A peer counts as aligned unless none were reached."})
        n = v["novice_unresolved"]
        if relative:
            pct = percentile(n, counts)
            t_ok = pct <= cfg["relative"]["novice_terms_low_max_pct"]
            checks.append({"name": "novice_terms_low", "passed": t_ok, "value": n, "percentile": pct, "threshold": cfg["relative"]["novice_terms_low_max_pct"],
                           "text": f"The novice\u2019s unresolved-term count is at the {_ordinal(pct)} percentile of these {len(inputs)} slides; it counts as low at the {_ordinal(cfg['relative']['novice_terms_low_max_pct'])} or below."})
        else:
            t_ok = n <= cfg["absolute"]["novice_terms_low_max"]
            checks.append({"name": "novice_terms_low", "passed": t_ok, "value": n, "threshold": cfg["absolute"]["novice_terms_low_max"],
                           "text": f"The novice\u2019s unresolved-term count counts as low at {cfg['absolute']['novice_terms_low_max']} or fewer."})
        key = _decide(n_ok, p_ok, t_ok)
        out[slide] = {
            "tier": key, **TIERS[key], "basis": "relative" if relative else "absolute",
            "comparator": "fieldwise", "n_slides": len(inputs), "min_slides_for_relative": cfg["min_slides_for_relative"], "checks": checks,
        }
    return out
