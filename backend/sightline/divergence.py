"""Divergence scoring: how far apart are what three audiences took away, and how far are
they from what the presenter meant.

Every metric is a `Metric`: a number together with the raw strings that produced it, so
the UI can answer "where does this number come from?" by showing the text. Nothing here
calls an LLM; similarity comes from a local sentence-transformers model.

Known limitation (read before trusting the headline): sentence embeddings measure how
similar two sentences are in TOPIC and phrasing, not whether they make the same claim.
Takeaways that are on-topic but contradictory ("throughput rose 2.4x" vs "throughput fell
2.4x") can score as similar. The metric is strong at what this product needs most, a
novice who could not extract the point at all, and weak at subtle misreadings. Swapping
in an NLI / cross-encoder scorer behind the `Embedder`-style seam is the planned upgrade.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Protocol, Sequence

import numpy as np

from .audiences import PERSONAS, AudienceResponse, Persona

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, d) float array, each row L2-normalised."""
        ...


class SentenceTransformerEmbedder:
    """Local, CPU, no API. Loads from the on-disk cache first so it works with no network
    after the first download; falls back to downloading only if the model is absent."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(self.model_name, local_files_only=True)
        except OSError:
            return SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._model = self._load()
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float64,
        )


_default: Embedder | None = None


def default_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = SentenceTransformerEmbedder()
    return _default


# -------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Metric:
    """A number and the raw text that produced it. `inputs` maps a label
    ("intent", "novice", ...) to the verbatim string that was embedded."""

    value: float
    inputs: Mapping[str, str]


@dataclass(frozen=True)
class TermGap:
    """Terms the novice could not resolve that the expert could. `terms` keeps the novice's
    spelling; the two lists it was computed from are kept so the UI can show both."""

    terms: tuple[str, ...]
    novice_unresolved: tuple[str, ...]
    expert_unresolved: tuple[str, ...]


@dataclass(frozen=True)
class SlideDivergence:
    slide_index: int
    intent: str
    takeaways: Mapping[Persona, str]
    # cosine(takeaway, declared intent), one per persona
    intent_alignment: Mapping[Persona, Metric]
    # mean of the three pairwise cosine distances. THE HEADLINE METRIC.
    audience_divergence: Metric
    # cosine distance for each pair, keyed in PERSONAS order: novice-peer, novice-expert, peer-expert
    pairwise_distance: Mapping[tuple[Persona, Persona], Metric]
    # expert alignment - novice alignment; large positive = the expert blind spot
    blind_spot_score: Metric
    term_gap: TermGap

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairwise_distance"] = {f"{a}-{b}": m for (a, b), m in d["pairwise_distance"].items()}
        return d


# ---------------------------------------------------------------------------- scoring


def normalize_term(term: str) -> str:
    """Case, width, hyphen/underscore/slash and surrounding punctuation are not
    differences of meaning ("KV-cache" == "kv cache"). Inner symbols are kept so "C++"
    does not collapse into "C"."""
    t = unicodedata.normalize("NFKC", term).casefold()
    t = re.sub(r"[-_/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" \t\"'“”‘’.,;:!?()[]{}")
    return t


def compute_term_gap(novice_unresolved: Sequence[str], expert_unresolved: Sequence[str]) -> TermGap:
    expert_norm = {normalize_term(t) for t in expert_unresolved}
    seen: set[str] = set()
    gap: list[str] = []
    for term in novice_unresolved:
        n = normalize_term(term)
        if n and n not in expert_norm and n not in seen:
            seen.add(n)
            gap.append(term)
    return TermGap(tuple(gap), tuple(novice_unresolved), tuple(expert_unresolved))


def score_slide(
    intent: str,
    responses: Mapping[Persona, AudienceResponse],
    embedder: Embedder | None = None,
    slide_index: int = 0,
) -> SlideDivergence:
    if not intent.strip():
        raise ValueError("declared intent is empty; alignment is undefined without it")
    missing = [p for p in PERSONAS if p not in responses]
    if missing:
        raise ValueError(f"missing audience responses: {missing}")

    embedder = embedder or default_embedder()
    takeaways = {p: responses[p].takeaway for p in PERSONAS}
    vecs = embedder.embed([intent, *(takeaways[p] for p in PERSONAS)])
    v_intent, v = vecs[0], dict(zip(PERSONAS, vecs[1:]))

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    alignment = {
        p: Metric(cos(v_intent, v[p]), {"intent": intent, p: takeaways[p]}) for p in PERSONAS
    }
    pairwise = {
        (a, b): Metric(1.0 - cos(v[a], v[b]), {a: takeaways[a], b: takeaways[b]})
        for a, b in combinations(PERSONAS, 2)
    }
    divergence = Metric(
        float(np.mean([m.value for m in pairwise.values()])), dict(takeaways)
    )
    blind_spot = Metric(
        alignment["expert"].value - alignment["novice"].value,
        {"intent": intent, "novice": takeaways["novice"], "expert": takeaways["expert"]},
    )
    return SlideDivergence(
        slide_index=slide_index,
        intent=intent,
        takeaways=takeaways,
        intent_alignment=alignment,
        audience_divergence=divergence,
        pairwise_distance=pairwise,
        blind_spot_score=blind_spot,
        term_gap=compute_term_gap(
            responses["novice"].unresolved_terms, responses["expert"].unresolved_terms
        ),
    )
