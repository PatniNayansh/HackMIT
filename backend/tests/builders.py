"""Builders for slide results with controlled numbers, made through the REAL `score_slide` so
the record shapes cannot drift from what step 1 produces."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from sightline.audiences import PERSONAS, AudienceReading, AudienceResponse
from sightline.deck import SlideResult, build_metrics, expert_takeaway_intent, reading_record

INTENT = "INTENT"


class AngleEmbedder:
    """'INTENT' -> (1, 0); 'cos=0.30' -> a unit vector whose cosine with the intent is 0.30.
    Lets a test say exactly what each persona's intent alignment is."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            c = 1.0 if t == INTENT else float(t.split("cos=")[1].split()[0])
            rows.append((c, math.sqrt(max(0.0, 1 - c * c))))
        return np.array(rows)


def slide_result(
    index: int,
    align: tuple[float, float] = (0.5, 0.5),
    conf: tuple[float, float, float] = (0.5, 0.5, 0.5),
    terms: tuple[Sequence[str], Sequence[str], Sequence[str]] = ((), (), ()),
    text: str = "[title] A slide",
) -> SlideResult:
    """A scored slide. `align` is (novice, peer) alignment to the slide's intent, which is the
    expert's takeaway verbatim; the expert is the reference, so its own alignment is definitional
    (see deck.build_metrics)."""
    cos = (*align, 1.0)
    responses = {
        p: AudienceResponse(
            takeaway=f"cos={a:.2f} takeaway of {p}",
            confidence=c,
            unresolved_terms=list(t),
            questions=[],
            inferred_claim=f"{p} claim",
        )
        for p, a, c, t in zip(PERSONAS, cos, conf, terms)
    }
    readings = {
        p: reading_record(AudienceReading(p, index, f"hash{index}", r, "fake-model", False, 1.0))
        for p, r in responses.items()
    }
    return {
        "index": index,
        "text": text,
        "readings": readings,
        "slide_intent": expert_takeaway_intent(responses["expert"]),
        "metrics": build_metrics(responses["expert"].takeaway, responses, AngleEmbedder(), index),
        "metrics_error": None,
        "scored_by": "angle-embedder",
        "timing": {"personas_s": 1.0},
    }


class HashEmbedder:
    """Deterministic stand-in for the sentence-transformers model: same text, same vector."""

    model_name = "hash-embedder"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import hashlib

        rows = []
        for t in texts:
            raw = np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8).astype(float) - 127.5
            rows.append(raw / np.linalg.norm(raw))
        return np.array(rows)


# ------------------------------------------------------------------ field-wise (compare.py) slides


def fw_slide_result(
    index: int,
    novice: dict | None = None,
    peer: dict | None = None,
    expert: dict | None = None,
    verdicts: dict | None = None,
    unresolved: tuple = ((), (), ()),
    text: str = "[title] A slide",
    image: str | None = None,
) -> SlideResult:
    """A scored slide under the field-wise comparator, built through the REAL `build_fieldwise_metrics`
    so the payload shape cannot drift. Fields default to a slide where everyone reached everything."""
    from sightline.compare import build_fieldwise_metrics

    full = {"concept": "isotope", "claim": "Isotopes are atoms of one element with different neutron counts.", "result": "6", "vehicle": "carbon-12 and carbon-14"}
    fields = {"novice": novice if novice is not None else dict(full), "peer": peer if peer is not None else dict(full), "expert": expert if expert is not None else dict(full)}
    yes = {"evaluated": True, "expert_entails_audience": True, "audience_entails_expert": True, "rationale": "Same idea.", "quote": "atoms of one element", "quote_verified": True}
    structured = {"fields": fields, "verdicts": {"novice": yes, "peer": yes, **(verdicts or {})},
                  "meta": {"model": "fake-sonnet", "attempts": 1, "latency_s": 0.1, "dropped": [], "retried_because": []}}
    takeaways = {p: f"takeaway of {p} on slide {index}" for p in PERSONAS}
    responses = {
        p: AudienceResponse(takeaway=takeaways[p], confidence=0.5, unresolved_terms=list(u), questions=[], inferred_claim=f"{p} claim")
        for p, u in zip(PERSONAS, unresolved)
    }
    readings = {p: reading_record(AudienceReading(p, index, f"hash{index}", r, "fake-model", False, 1.0)) for p, r in responses.items()}
    return {
        "index": index, "text": text, "readings": readings,
        "slide_intent": expert_takeaway_intent(responses["expert"]),
        "metrics": build_fieldwise_metrics(takeaways, structured, {p: list(u) for p, u in zip(PERSONAS, unresolved)}, image),
        "metrics_error": None, "scored_by": "fake-sonnet", "timing": {"personas_s": 1.0, "structuring_s": 1.0},
        "image_content": {"text": image, "model": "fake-haiku", "source": "vision", "machine_generated": True} if image else None,
    }
