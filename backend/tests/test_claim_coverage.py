"""Claim comparison by proposition coverage (compare.py): the states, how each is reached, the
provenance rule and the tripwire. All offline, from hand-written fixtures and scripted model replies."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from profe import compare as C
from profe.compare import (
    build_fieldwise_metrics, chart_value, clean_propositions, coverage_state, judge_coverage,
    quoted_word_for_word, reask_contradiction, structure_slide,
)
from profe.llm import LLMError
from profe.tiers import CONFIG

# ------------------------------------------------------------------------- the live bug, slide 5
EXPERT_CLAIM = "Cost-benefit/opportunity cost analysis applies to the college decision, challenging the standard case for college."
NOVICE_CLAIM = "Cost-benefit thinking applies to the decision of going to college by weighing benefits against opportunity costs."
P1 = "Cost-benefit / opportunity-cost analysis applies to the college decision"
P2 = "It challenges the standard case for college"

NOBODY = {"concept": None, "claim": None, "result": None, "vehicle": None}


def fields(claim, concept=None, result=None):
    return {"concept": concept, "claim": claim, "result": result, "vehicle": None}


class Scripted:
    model = "fake-sonnet"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user": user_text, "schema": schema})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def judgement(*items, extras=(), note="n"):
    """items are (id, status, evidence)."""
    return {"judgements": [{"id": i, "status": s, "evidence": e} for i, s, e in items], "extra_assertions": list(extras), "note": note}


def raw_reply(expert_claim, novice_claim, peer_claim, props, novice_cov, peer_cov):
    return {
        "fields": {"novice": fields(novice_claim), "peer": fields(peer_claim), "expert": fields(expert_claim)},
        "propositions": [{"id": i, "text": t} for i, t in props],
        "coverage": {"novice": novice_cov, "peer": peer_cov},
    }


TAKEAWAYS = {"novice": "novice takeaway", "peer": "peer takeaway", "expert": "expert takeaway"}


async def run(expert_claim, novice_claim, props, novice_cov, peer_claim=None, peer_cov=None, embedder=None, extra_replies=()):
    peer_claim = peer_claim if peer_claim is not None else expert_claim
    peer_cov = peer_cov if peer_cov is not None else judgement(*((i, "covered", peer_claim) for i, _ in props))
    client = Scripted(raw_reply(expert_claim, novice_claim, peer_claim, props, novice_cov, peer_cov), *extra_replies)
    got = await structure_slide(client, TAKEAWAYS, embedder=embedder)
    m = build_fieldwise_metrics(TAKEAWAYS, got, {"novice": [], "peer": [], "expert": []})
    return client, got, m


# ----------------------------------------------------------- F1. THE REGRESSION CASE (written first)


async def test_the_college_case_is_under_specified_the_novice_took_the_method_but_missed_the_argument():
    """Reported `divergent` before. The novice covers p1 (the method) and omits p2 (the argument): partial
    coverage, and the miss is the finding."""
    cov = judgement(("p1", "covered", "Cost-benefit thinking applies to the decision of going to college"), ("p2", "omitted", ""),
                    note="Novice recovered the method but not the argument.")
    _, got, m = await run(EXPERT_CLAIM, NOVICE_CLAIM, [("p1", P1), ("p2", P2)], cov)

    claim = m["comparisons"]["novice"]["claim"]
    assert claim["outcome"] == "under-specified" and claim["outcome"] != "divergent"
    props = {p["id"]: p for p in claim["coverage"]["propositions"]}
    assert props["p1"]["status"] == "covered" and props["p1"]["evidence"] == "Cost-benefit thinking applies to the decision of going to college"
    assert props["p2"]["status"] == "omitted" and props["p2"]["evidence"] is None
    assert claim["coverage"]["missed"] == [P2] and claim["coverage"]["covered"] == 1 and claim["coverage"]["total"] == 2
    assert claim["coverage"]["extra_assertions"] == []  # "by weighing benefits against opportunity costs" is a definition, not a new assertion
    assert claim["coverage"]["note"] == "Novice recovered the method but not the argument."
    assert m["chart"]["novice"] == {"value": 0.5, "covered": 1, "total": 2, "state": "under-specified", "thin": True}  # only the claim is scored here, so the profile is thin


def test_the_prompt_carries_the_worked_example_and_the_paraphrase_rule_verbatim():
    s = C._SYSTEM
    for text in (
        "Two statements express the same proposition when a reader who understood one would assent to the other.",
        'Standard definitions of named concepts count as the same proposition: "cost-benefit analysis" and "weighing benefits against costs" are one proposition stated two ways, not two propositions.',
        "Restating a concept's definition is not a new assertion.",
        "Adding a NEW claim is not paraphrase. A different quantity, a different direction of effect, a different scope, or an assertion the expert's claim does not make is not covered.",
    ):
        assert " ".join(text.split()) in " ".join(s.split()), text
    assert "WORKED EXAMPLE" in s and EXPERT_CLAIM in s and NOVICE_CLAIM in s and "p1 covered" in s and "p2 omitted" in s
    assert "Novice recovered the method but not the argument." in s
    assert "words copied EXACTLY from THAT VIEWER'S CLAIM" in s and "the status is omitted" in s
    assert "entail" not in s.lower()  # the question is coverage now, not two-way entailment


# ----------------------------------------------------------------------- F2. definitional paraphrase


async def test_a_definition_of_the_concept_is_covered_not_an_extra_assertion():
    expert = "Cost-benefit analysis applies to the decision."
    novice = "Weighing benefits against costs applies to the decision."
    cov = judgement(("p1", "covered", "Weighing benefits against costs applies to the decision"))
    client, _, m = await run(expert, novice, [("p1", "Cost-benefit analysis applies to the decision")], cov)
    claim = m["comparisons"]["novice"]["claim"]
    assert claim["outcome"] == "equivalent" and claim["coverage"]["extra_assertions"] == []
    # the instruction that makes the model see this is in the prompt it was sent
    assert "cost-benefit analysis" in client.calls[0]["system"] and "weighing benefits against costs" in client.calls[0]["system"]


# ------------------------------------------------------------------------------- F3. negation


async def test_negation_is_divergent():
    expert = "Cost-benefit thinking applies to the college decision."
    novice = "Cost-benefit thinking does not apply to the college decision."
    cov = judgement(("p1", "contradicted", "does not apply to the college decision"))
    _, _, m = await run(expert, novice, [("p1", "Cost-benefit thinking applies to the college decision")], cov)
    claim = m["comparisons"]["novice"]["claim"]
    assert claim["outcome"] == "divergent" and claim["coverage"]["contradicted"] == ["Cost-benefit thinking applies to the college decision"]
    assert m["chart"]["novice"]["value"] == 0.0 and m["reach"]["novice"] == "fail"


# ------------------------------------------------------------------------- F4. wrong quantity


async def test_a_wrong_quantity_is_divergent():
    expert = "The opportunity cost of choosing Tyler is $50."
    novice = "The opportunity cost of choosing Tyler is $40."
    cov = judgement(("p1", "contradicted", "is $40"))
    _, _, m = await run(expert, novice, [("p1", "The opportunity cost of choosing Tyler is $50")], cov)
    assert m["comparisons"]["novice"]["claim"]["outcome"] == "divergent"
    assert m["comparisons"]["novice"]["claim"]["coverage"]["propositions"][0]["evidence"] == "is $40"


# --------------------------------------------------------------- F5. extra assertion, full coverage


async def test_full_coverage_plus_an_unsupported_extra_assertion_is_over_claimed():
    expert = "Cost-benefit thinking applies to the college decision."
    novice = "Cost-benefit thinking applies to the college decision and to every major life choice."
    cov = judgement(("p1", "covered", "Cost-benefit thinking applies to the college decision"), extras=["every major life choice"])
    _, _, m = await run(expert, novice, [("p1", "Cost-benefit thinking applies to the college decision")], cov)
    claim = m["comparisons"]["novice"]["claim"]
    assert claim["outcome"] == "over-claimed" and claim["coverage"]["extra_assertions"] == ["every major life choice"]
    assert m["chart"]["novice"]["value"] == 1.0  # everything the expert asserted was covered


# ------------------------------------------------------------------------ F6. empty audience claim


async def test_an_empty_audience_claim_is_absent_and_the_model_is_not_asked_about_it():
    assert coverage_state("A claim.", None, None) == "absent"  # no coverage is consulted for a reader with no claim
    assert coverage_state(None, None, None) is None and coverage_state(None, "A claim.", None) is None

    # Through the pipeline: whatever the model says about a reader whose claim is null is ignored.
    bogus = judgement(("p1", "contradicted", "words that were never written"))
    _, got, m = await run(EXPERT_CLAIM, None, [("p1", P1), ("p2", P2)], bogus)
    claim = m["comparisons"]["novice"]["claim"]
    assert claim["outcome"] == "absent" and claim["status"] == "gap"
    assert [p["status"] for p in claim["coverage"]["propositions"]] == ["omitted", "omitted"] and claim["coverage"]["missed"] == [P1, P2]
    assert m["chart"]["novice"]["value"] == 0.0

    # and the tripwire's re-ask makes NO call when there is nothing to re-judge
    client = Scripted()
    assert await reask_contradiction(client, EXPERT_CLAIM, "", [{"id": "p1", "text": P1}]) is None
    assert await reask_contradiction(client, "", NOVICE_CLAIM, [{"id": "p1", "text": P1}]) is None
    assert client.calls == []


# ------------------------------------------------------------- F7. contradiction outranks coverage


def test_two_covered_and_one_contradicted_is_divergent_not_partial_credit():
    claim = "Aa bb. Cc dd. Ee ff."
    props = clean_propositions([{"id": "p1", "text": "a"}, {"id": "p2", "text": "b"}, {"id": "p3", "text": "c"}], "expert claim")
    cov = judge_coverage(judgement(("p1", "covered", "Aa bb"), ("p2", "covered", "Cc dd"), ("p3", "contradicted", "Ee ff")), props, claim)
    assert coverage_state("expert claim", claim, cov) == "divergent"
    assert chart_value("divergent", cov) == 0.0  # divergent is 0.0 regardless of what else was covered
    assert sum(p["status"] == "covered" for p in cov["propositions"]) == 2


# ------------------------------------------------------------------------------- F8. the tripwire


class Vectors:
    """A stand-in for the local sentence-embedding model: chosen strings get chosen cosines."""

    def __init__(self, cosine):
        self.cosine, self.calls = cosine, []

    def embed(self, texts):
        self.calls.append(list(texts))
        c = self.cosine
        return np.array([[1.0, 0.0], [c, float(np.sqrt(1 - c * c))]])


EXPERT = "Cost-benefit thinking applies to the college decision."
NEAR = "Cost-benefit thinking applies to the college decision, mostly."
DIVERGENT_FIRST = judgement(("p1", "contradicted", "mostly"))
CLEAN = judgement(("p1", "covered", "Cost-benefit thinking applies to the college decision"), note="On reflection nothing is contradicted.")
STILL = judgement(("p1", "contradicted", "mostly"), note="Contradicts.")
PROPS = [("p1", "Cost-benefit thinking applies to the college decision")]


async def test_a_forced_divergent_with_high_cosine_triggers_exactly_one_reask_and_downgrades_if_nothing_is_contradicted():
    client, got, m = await run(EXPERT, NEAR, PROPS, DIVERGENT_FIRST, embedder=Vectors(0.9), extra_replies=[CLEAN, CLEAN, CLEAN])
    assert len(client.calls) == 2  # the one structuring call and EXACTLY one re-ask, never more
    assert "NAME it" in client.calls[1]["system"] and "give its id and quote the exact words" in client.calls[1]["system"]
    assert NEAR in client.calls[1]["user"] and EXPERT in client.calls[1]["user"]
    (trip,) = got["meta"]["tripwire"]
    assert (trip["persona"], trip["first"], trip["outcome"], trip["after"]) == ("novice", "divergent", "downgraded", "equivalent") and trip["cosine"] == 0.9 and trip["limit"] == 0.75
    assert m["comparisons"]["novice"]["claim"]["outcome"] == "equivalent"  # downgraded per the coverage table, and logged


async def test_when_the_reask_confirms_the_contradiction_the_state_stands_and_it_is_still_only_one_reask():
    client, got, m = await run(EXPERT, NEAR, PROPS, DIVERGENT_FIRST, embedder=Vectors(0.9), extra_replies=[STILL, STILL])
    assert len(client.calls) == 2
    assert got["meta"]["tripwire"][0]["outcome"] == "confirmed"
    assert m["comparisons"]["novice"]["claim"]["outcome"] == "divergent"


async def test_the_tripwire_stays_quiet_when_the_claims_are_not_close():
    client, got, m = await run(EXPERT, NEAR, PROPS, DIVERGENT_FIRST, embedder=Vectors(0.4))
    assert len(client.calls) == 1 and got["meta"]["tripwire"] == [] and m["comparisons"]["novice"]["claim"]["outcome"] == "divergent"


async def test_the_tripwire_only_looks_at_divergent_pairs_so_the_common_path_costs_nothing():
    emb = Vectors(0.99)
    client, got, _ = await run(EXPERT, NEAR, PROPS, judgement(("p1", "covered", "Cost-benefit thinking applies")), embedder=emb)
    assert len(client.calls) == 1 and emb.calls == []  # not even the local model was consulted


async def test_without_an_embedder_there_is_no_tripwire():
    client, got, _ = await run(EXPERT, NEAR, PROPS, DIVERGENT_FIRST, embedder=None)
    assert len(client.calls) == 1 and got["meta"]["tripwire"] == []


async def test_a_failed_reask_keeps_the_first_judgement_and_says_so():
    client, got, m = await run(EXPERT, NEAR, PROPS, DIVERGENT_FIRST, embedder=Vectors(0.9), extra_replies=[LLMError("down")])
    assert got["meta"]["tripwire"][0]["outcome"] == "reask_failed" and m["comparisons"]["novice"]["claim"]["outcome"] == "divergent"


def test_the_tripwire_threshold_lives_in_the_same_config_dict_as_the_tier_thresholds():
    assert CONFIG["tripwire"] == {"cosine": 0.75}


# ---------------------------------------------------------------- the table in B, row by row


def cov_of(*statuses, extras=()):
    props = [{"id": f"p{i}", "text": f"prop {i}"} for i in range(1, len(statuses) + 1)]
    return {"propositions": [{**p, "status": s, "evidence": "x" if s != "omitted" else None} for p, s in zip(props, statuses)],
            "extra_assertions": list(extras)}


@pytest.mark.parametrize("statuses,extras,state", [
    (("covered",), (), "equivalent"),
    (("covered", "covered", "covered"), (), "equivalent"),
    (("covered", "covered"), ("more",), "over-claimed"),
    (("covered", "omitted"), (), "under-specified"),
    (("covered", "omitted", "omitted"), (), "under-specified"),
    (("covered", "omitted"), ("more",), "under-specified"),          # extras only matter when everything is covered
    (("omitted", "omitted"), (), "absent"),
    (("omitted",), ("more",), "absent"),
    (("contradicted",), (), "divergent"),
    (("covered", "covered", "contradicted"), (), "divergent"),      # contradiction is checked FIRST
    (("covered", "contradicted"), ("more",), "divergent"),
    (("omitted", "contradicted"), (), "divergent"),
])
def test_the_state_table(statuses, extras, state):
    assert coverage_state("expert", "audience", cov_of(*statuses, extras=extras)) == state


def test_divergent_appears_only_where_a_proposition_is_contradicted():
    """Exhaustive over every combination of up to three propositions and either extras setting."""
    for n in (1, 2, 3):
        for statuses in itertools.product(C.STATUSES, repeat=n):
            for extras in ((), ("more",)):
                state = coverage_state("expert", "audience", cov_of(*statuses, extras=extras))
                assert (state == "divergent") == ("contradicted" in statuses), (statuses, extras, state)
                if state == "equivalent":
                    assert set(statuses) == {"covered"} and not extras
                if state == "absent":
                    assert "covered" not in statuses


def test_the_state_names_are_unchanged_and_each_has_a_meaning():
    assert C.STATES == ("equivalent", "over-claimed", "under-specified", "divergent", "absent") and set(C.STATE_MEANING) == set(C.STATES)


# ------------------------------------------------------------- the chart value is a real quantity


@pytest.mark.parametrize("statuses,state,value", [
    (("covered", "covered", "omitted"), "under-specified", 0.6667), (("covered", "omitted"), "under-specified", 0.5),
    (("covered", "covered", "covered"), "equivalent", 1.0), (("omitted", "omitted"), "absent", 0.0),
    (("covered", "covered", "contradicted"), "divergent", 0.0),  # divergent is 0.0 whatever else was covered
])
def test_chart_value_is_propositions_covered_over_total(statuses, state, value):
    assert chart_value(state, cov_of(*statuses)) == value


def test_a_slide_with_no_expert_claim_has_nothing_to_count():
    assert chart_value(None, None) is None and chart_value("absent", {"propositions": [], "extra_assertions": []}) is None


async def test_the_metrics_carry_covered_and_total_for_the_tooltip_and_no_ordinal_at_all():
    cov = judgement(("p1", "covered", "Cost-benefit thinking applies to the decision of going to college"), ("p2", "omitted", ""))
    _, _, m = await run(EXPERT_CLAIM, NOVICE_CLAIM, [("p1", P1), ("p2", P2)], cov)
    assert m["chart"]["novice"]["covered"] == 1 and m["chart"]["novice"]["total"] == 2
    assert m["chart"]["peer"]["value"] == 1.0 and m["chart"]["expert"] == {"value": 1.0, "covered": 2, "total": 2, "state": "equivalent", "thin": True, "definitional": True}
    assert "ordinal" not in m and not hasattr(C, "ORDINAL")  # the invented 1.0 / 0.66 / 0.33 / 0.0 is gone


# ------------------------------------------------------------- G3. the provenance rule, word for word


def test_a_proposition_is_never_covered_without_a_span_found_word_for_word_in_the_audience_claim():
    props = clean_propositions([{"id": "p1", "text": "a"}], "expert claim")
    claim = "Cost-benefit thinking applies to the college decision."
    ok = judge_coverage(judgement(("p1", "covered", "cost-benefit thinking applies")), props, claim)
    assert ok["propositions"][0]["status"] == "covered"  # case and spacing do not matter
    for bad in ("Cost benefit thinking applies", "thinking applies to college", "Cost-benefit thinking is applied", "", None, "applies to the college decision, always"):
        got = judge_coverage(judgement(("p1", "covered", bad)), props, claim)
        assert got["propositions"][0]["status"] == "omitted" and got["propositions"][0]["evidence"] is None, bad
        assert got["downgraded"] and got["downgraded"][0]["claimed"] == "covered"


def test_a_contradiction_needs_a_real_span_too_so_divergent_needs_positive_evidence():
    props = clean_propositions([{"id": "p1", "text": "a"}], "expert claim")
    got = judge_coverage(judgement(("p1", "contradicted", "words the reader never wrote")), props, "The reader wrote something else.")
    assert coverage_state("expert claim", "The reader wrote something else.", got) == "absent"  # not divergent


def test_a_span_must_sit_on_word_boundaries():
    assert quoted_word_for_word("cost", "the cost is high") and not quoted_word_for_word("cost", "the costly choice")
    assert quoted_word_for_word("is $40", "The cost is $40.") and not quoted_word_for_word("is $4", "The cost is $40.")


def test_extra_assertions_are_quoted_from_the_audience_claim_or_dropped():
    props = clean_propositions([{"id": "p1", "text": "a"}], "expert claim")
    claim = "Cost-benefit thinking applies here and everywhere."
    got = judge_coverage(judgement(("p1", "covered", "Cost-benefit thinking applies here"), extras=["and everywhere", "a claim never made"]), props, claim)
    assert got["extra_assertions"] == ["and everywhere"] and got["dropped_extras"] == ["a claim never made"]
    assert coverage_state("e", claim, got) == "over-claimed"


def test_an_unknown_status_or_a_missing_judgement_is_omitted_never_covered():
    props = clean_propositions([{"id": "p1", "text": "a"}, {"id": "p2", "text": "b"}], "expert claim")
    got = judge_coverage({"judgements": [{"id": "p1", "status": "fully agrees", "evidence": "x"}], "extra_assertions": [], "note": ""}, props, "x y z")
    assert [p["status"] for p in got["propositions"]] == ["omitted", "omitted"]


def test_propositions_are_renumbered_joined_by_the_models_own_ids_and_bounded():
    props = clean_propositions([{"id": "a", "text": "first"}, {"id": "b", "text": "second"}], "claim")
    assert [(p["id"], p["orig"]) for p in props] == [("p1", "a"), ("p2", "b")]
    got = judge_coverage(judgement(("b", "covered", "second thing"), ("a", "omitted", "")), props, "the second thing happens")
    assert [(p["id"], p["status"]) for p in got["propositions"]] == [("p1", "omitted"), ("p2", "covered")]
    assert clean_propositions([{"id": "p1", "text": "x"}], None) == []  # no expert claim, no propositions
    with pytest.raises(C.StructuringError, match="no propositions"):
        clean_propositions([], "a present claim")
    with pytest.raises(C.StructuringError, match="too many"):
        clean_propositions([{"id": f"p{i}", "text": "x"} for i in range(C.MAX_PROPOSITIONS + 1)], "claim")


async def test_an_unusable_decomposition_is_retried_once():
    good = raw_reply(EXPERT, EXPERT, EXPERT, PROPS, judgement(("p1", "covered", EXPERT)), judgement(("p1", "covered", EXPERT)))
    bad = {**good, "propositions": []}
    c = Scripted(bad, good)
    got = await structure_slide(c, TAKEAWAYS)
    assert got["meta"]["attempts"] == 2 and "no propositions" in got["meta"]["retried_because"][0]


def test_the_schema_has_a_status_enum_and_stays_inside_the_apis_nullable_limit():
    j = C._JUDGEMENT_SCHEMA["properties"]["status"]
    assert j["enum"] == ["covered", "omitted", "contradicted"]

    def unions(node):
        if isinstance(node, dict):
            return (1 if "anyOf" in node or isinstance(node.get("type"), list) else 0) + sum(unions(v) for v in node.values())
        return sum(unions(v) for v in node) if isinstance(node, list) else 0

    assert unions(C.STRUCTURE_SCHEMA) <= 16 and unions(C._COVERAGE_SCHEMA) == 0
