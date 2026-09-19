"""divergence.py.

Part 1 checks the arithmetic exactly, using an embedder with hand-set vectors.
Part 2 runs the REAL local embedding model on hand-written takeaways. Those takeaways are
my hypothesis of what each audience would say, not LLM output: they show the metric can
separate a converging slide from a diverging one, and nothing more. Whether the LLM
personas actually produce such takeaways is the live gate's job.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from slides import CLEAR_INTENT, JARGON_INTENT

from sightline.audiences import AudienceResponse
from sightline.divergence import compute_term_gap, normalize_term, score_slide


class VectorEmbedder:
    def __init__(self, table: dict[str, list[float]]):
        self.table = {k: np.array(v, float) / np.linalg.norm(v) for k, v in table.items()}

    def embed(self, texts):
        return np.stack([self.table[t] for t in texts])


def resp(takeaway: str, terms: list[str] | None = None) -> AudienceResponse:
    return AudienceResponse(
        takeaway=takeaway, confidence=0.5, unresolved_terms=terms or [], questions=[], inferred_claim="c"
    )


# ------------------------------------------------------------------ exact arithmetic


def test_metrics_from_known_geometry():
    emb = VectorEmbedder(
        {
            "INTENT": [1, 0, 0],
            "N": [0, 1, 0],  # cosine 0 with intent
            "P": [0.6, 0.8, 0],  # cosine 0.6 with intent, 0.8 with N
            "E": [1, 0, 0],  # cosine 1 with intent, 0 with N, 0.6 with P
        }
    )
    out = score_slide("INTENT", {"novice": resp("N"), "peer": resp("P"), "expert": resp("E")}, emb, 4)

    assert out.slide_index == 4
    assert out.intent_alignment["novice"].value == pytest.approx(0.0)
    assert out.intent_alignment["peer"].value == pytest.approx(0.6)
    assert out.intent_alignment["expert"].value == pytest.approx(1.0)

    assert out.pairwise_distance[("novice", "peer")].value == pytest.approx(0.2)
    assert out.pairwise_distance[("novice", "expert")].value == pytest.approx(1.0)
    assert out.pairwise_distance[("peer", "expert")].value == pytest.approx(0.4)
    assert out.audience_divergence.value == pytest.approx((0.2 + 1.0 + 0.4) / 3)

    assert out.blind_spot_score.value == pytest.approx(1.0)


def test_identical_takeaways_have_zero_divergence_and_zero_blind_spot():
    emb = VectorEmbedder({"INTENT": [1, 1], "SAME": [1, 1]})
    r = resp("SAME")
    out = score_slide("INTENT", {"novice": r, "peer": r, "expert": r}, emb)
    assert out.audience_divergence.value == pytest.approx(0.0, abs=1e-9)
    assert out.blind_spot_score.value == pytest.approx(0.0, abs=1e-9)
    assert all(m.value == pytest.approx(1.0) for m in out.intent_alignment.values())


def test_blind_spot_is_negative_when_the_novice_gets_it_and_the_expert_does_not():
    emb = VectorEmbedder({"INTENT": [1, 0], "GOOD": [1, 0], "BAD": [0, 1]})
    out = score_slide("INTENT", {"novice": resp("GOOD"), "peer": resp("GOOD"), "expert": resp("BAD")}, emb)
    assert out.blind_spot_score.value == pytest.approx(-1.0)


# --------------------------------------------------------------- decomposability


def test_every_number_carries_the_raw_text_that_produced_it():
    emb = VectorEmbedder({"INTENT": [1, 0], "N-text": [0, 1], "P-text": [1, 1], "E-text": [1, 0]})
    out = score_slide(
        "INTENT", {"novice": resp("N-text"), "peer": resp("P-text"), "expert": resp("E-text")}, emb
    )
    assert out.takeaways == {"novice": "N-text", "peer": "P-text", "expert": "E-text"}
    assert dict(out.audience_divergence.inputs) == out.takeaways
    assert dict(out.intent_alignment["peer"].inputs) == {"intent": "INTENT", "peer": "P-text"}
    assert dict(out.blind_spot_score.inputs) == {"intent": "INTENT", "novice": "N-text", "expert": "E-text"}
    assert dict(out.pairwise_distance[("novice", "expert")].inputs) == {"novice": "N-text", "expert": "E-text"}


def test_result_serialises_to_json_for_the_api():
    emb = VectorEmbedder({"I": [1, 0], "a": [1, 0], "b": [0, 1]})
    out = score_slide("I", {"novice": resp("a", ["x"]), "peer": resp("b"), "expert": resp("a")}, emb)
    d = json.loads(json.dumps(out.to_dict()))
    assert d["pairwise_distance"]["novice-expert"]["inputs"] == {"novice": "a", "expert": "a"}
    assert d["term_gap"]["terms"] == ["x"]


def test_rejects_missing_intent_or_persona():
    emb = VectorEmbedder({"I": [1, 0], "a": [1, 0]})
    with pytest.raises(ValueError, match="intent"):
        score_slide("  ", {"novice": resp("a"), "peer": resp("a"), "expert": resp("a")}, emb)
    with pytest.raises(ValueError, match="peer"):
        score_slide("I", {"novice": resp("a"), "expert": resp("a")}, emb)


# --------------------------------------------------------------------- term gap


def test_term_gap_is_novice_minus_expert_after_normalisation():
    gap = compute_term_gap(
        novice_unresolved=["KV-cache", "PagedAttention", "p99.", "Foo", "foo"],
        expert_unresolved=["kv cache", "  P99 "],
    )
    assert gap.terms == ("PagedAttention", "Foo")  # novice spelling, deduplicated
    assert gap.novice_unresolved == ("KV-cache", "PagedAttention", "p99.", "Foo", "foo")
    assert gap.expert_unresolved == ("kv cache", "  P99 ")


def test_term_normalisation_keeps_meaningful_symbols():
    assert normalize_term("C++") != normalize_term("C")
    assert normalize_term("“KV_cache”.") == "kv cache"
    assert compute_term_gap([], ["x"]).terms == ()


def test_term_gap_ignores_terms_only_the_expert_flagged():
    assert compute_term_gap(["TTFT"], ["TTFT", "coined-name"]).terms == ()


# --------------------------------------------- real embedder, hand-written takeaways

CLEAR_TAKEAWAYS = {
    "novice": "Third-quarter sales rose 20% over the second quarter because more customers renewed.",
    "peer": "Q3 revenue is up 20% from Q2 thanks to customers renewing their subscriptions.",
    "expert": "Sales went up about 20% last quarter, and renewals are the reason.",
}
JARGON_TAKEAWAYS = {
    "novice": "Something about a computer system being faster, but I can't tell what was measured or how the improvement was achieved.",
    "peer": "A serving system is more efficient under load, probably through better memory management and batching, though I can't follow the specific techniques.",
    "expert": "The system reaches 2.4x goodput at p99 latency by combining PagedAttention-style KV-cache management with speculative decoding.",
}


@pytest.mark.embedding
def test_real_embedder_separates_converging_from_diverging_slides(real_embedder):
    clear = score_slide(CLEAR_INTENT, {p: resp(t) for p, t in CLEAR_TAKEAWAYS.items()}, real_embedder)
    jargon = score_slide(JARGON_INTENT, {p: resp(t) for p, t in JARGON_TAKEAWAYS.items()}, real_embedder)

    report = (
        f"clear: div={clear.audience_divergence.value:.3f} "
        f"align={ {p: round(m.value, 3) for p, m in clear.intent_alignment.items()} } "
        f"blind={clear.blind_spot_score.value:.3f}\n"
        f"jargon: div={jargon.audience_divergence.value:.3f} "
        f"align={ {p: round(m.value, 3) for p, m in jargon.intent_alignment.items()} } "
        f"blind={jargon.blind_spot_score.value:.3f}"
    )
    # Absolute divergence is NOT near zero even for real paraphrases (~0.4 measured), so only
    # relative comparisons are meaningful. Thresholds are set from these hand-written proxies.
    assert jargon.audience_divergence.value > clear.audience_divergence.value + 0.1, report
    assert all(m.value > 0.6 for m in clear.intent_alignment.values()), report
    assert abs(clear.blind_spot_score.value) < 0.15, report
    assert jargon.blind_spot_score.value > 0.2, report
    a = jargon.intent_alignment
    assert a["expert"].value > a["peer"].value > a["novice"].value, report


@pytest.mark.embedding
def test_real_embedder_known_limitation_contradiction_reads_as_similar(real_embedder):
    """Documents the weakness stated in divergence.py: a contradictory takeaway is on-topic,
    so it embeds close to the intent. If this ever fails because the model now separates
    them, delete the caveat, not the test."""
    same_topic = score_slide(
        "Throughput increased 2.4x under the new scheduler.",
        {p: resp("Throughput decreased 2.4x under the new scheduler.") for p in ("novice", "peer", "expert")},
        real_embedder,
    )
    assert same_topic.intent_alignment["novice"].value > 0.85
