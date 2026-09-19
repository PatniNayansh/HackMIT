"""Builders for slide results with controlled numbers, made through the REAL `score_slide` so
the record shapes cannot drift from what step 1 produces."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from sightline.audiences import PERSONAS, AudienceReading, AudienceResponse
from sightline.deck import SlideResult, plain, reading_record
from sightline.divergence import score_slide

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
    align: tuple[float, float, float] = (0.5, 0.5, 0.5),
    conf: tuple[float, float, float] = (0.5, 0.5, 0.5),
    terms: tuple[Sequence[str], Sequence[str], Sequence[str]] = ((), (), ()),
    text: str = "[title] A slide",
) -> SlideResult:
    responses = {
        p: AudienceResponse(
            takeaway=f"cos={a:.2f} takeaway of {p}",
            confidence=c,
            unresolved_terms=list(t),
            questions=[],
            inferred_claim=f"{p} claim",
        )
        for p, a, c, t in zip(PERSONAS, align, conf, terms)
    }
    readings = {
        p: reading_record(AudienceReading(p, index, f"hash{index}", r, "fake-model", False, 1.0))
        for p, r in responses.items()
    }
    metrics = score_slide(INTENT, responses, AngleEmbedder(), index)
    return {
        "index": index,
        "text": text,
        "readings": readings,
        "metrics": plain(metrics.to_dict()),
        "metrics_error": None,
        "scored_by": "angle-embedder",
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
