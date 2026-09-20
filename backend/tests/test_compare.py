"""compare.py: the field-wise comparator, from hand-written fixtures. No model is called except
through fakes that return exactly the JSON a test wants."""

from __future__ import annotations

import pytest

from profe import compare as C
from profe.compare import (
    FileStructureCache, StructuringError, build_fieldwise_metrics, clean_fields, compare_concept, compare_field,
    compare_result, example_bound, figure_dependent, null_if_blank, slide_profile, structure_slide,
)
from profe.llm import LLMError

# ---- the live example this rework exists for: the expert names the principle and gives the answer;
# ---- the novice only describes the task.
EXPERT_T = "Opportunity cost is the value of the next best alternative you give up: choosing Tyler at $150 over Doja Cat at $100 costs $50 of foregone benefit."
NOVICE_T = "The slide compares tickets to see Tyler at $150 versus Doja Cat at $100 and works out the price difference."
PEER_T = "Opportunity cost, the benefit foregone from the next best alternative, is $50 when Tyler at $150 is chosen over Doja Cat."
TAKEAWAYS = {"novice": NOVICE_T, "peer": PEER_T, "expert": EXPERT_T}

EXPERT_F = {"concept": "opportunity cost", "claim": "Opportunity cost is the value of the next best alternative that is given up.",
            "result": "$50", "vehicle": "Tyler at $150 over Doja Cat at $100"}
NOVICE_F = {"concept": None, "claim": None, "result": None, "vehicle": "tickets to see Tyler at $150 versus Doja Cat at $100"}
PEER_F = {"concept": "Opportunity cost", "claim": "Opportunity cost is the benefit foregone from the next best alternative.",
          "result": "$50", "vehicle": "Tyler at $150 chosen over Doja Cat"}


def props_of(expert):
    """The expert claim as ONE proposition (the whole claim), enough for tests that are not about decomposition."""
    return C.clean_propositions([{"id": "p1", "text": expert["claim"]}], expert["claim"]) if expert.get("claim") else []


def coverage_of(fields, props, statuses=None, extras=()):
    """What structure_slide produces for one audience: nothing without a claim (no model consulted),
    all-omitted with a claim of None, otherwise the checked judgements. Evidence is the audience's own claim."""
    if not props:
        return None
    if fields.get("claim") is None:
        return C.all_omitted(props)
    statuses = statuses or ["covered"] * len(props)
    raw = {"judgements": [{"id": p["orig"], "status": st, "evidence": fields["claim"] if st != "omitted" else ""} for p, st in zip(props, statuses)],
           "extra_assertions": list(extras), "note": "n"}
    return C.judge_coverage(raw, props, fields["claim"])


def structured(novice=NOVICE_F, peer=PEER_F, expert=EXPERT_F, novice_statuses=None, peer_statuses=None, extras=((), ())):
    props = props_of(expert)
    return {"fields": {"novice": novice, "peer": peer, "expert": expert}, "propositions": props,
            "coverage": {"novice": coverage_of(novice, props, novice_statuses, extras[0]), "peer": coverage_of(peer, props, peer_statuses, extras[1])},
            "meta": {"model": "fake-sonnet", "attempts": 1, "latency_s": 0.1, "dropped": [], "tripwire": []}}


def metrics(**kw):
    image = kw.pop("image", None)
    return build_fieldwise_metrics(TAKEAWAYS, structured(**kw), {"novice": ["p99"], "peer": [], "expert": []}, image)


# ------------------------------------------------------------------------- result: exact match


@pytest.mark.parametrize("a,b,outcome", [
    ("$50", "$50", "match"), ("$50", "50", "match"), ("$50.00", "$50", "match"), (" $ 50 ", "$50", "match"),
    ("1,200", "$1200", "match"), ("50 dollars", "$50", "match"), ("50%", "50%", "match"), ("2.4x", "2.4X", "match"),
    ("$50", "$60", "mismatch"), ("$50", "$5", "mismatch"), ("50%", "50", "mismatch"), ("four", "4", "mismatch"),
    # extracted results are phrases: it is the quantity they state that is matched exactly
    ("2.4x improvement", "2.4x", "match"), ("2.4x higher request throughput", "2.4x more requests per second", "match"),
    ("2.4x improvement", "2.4 times faster", "match"), ("2.4x p99 goodput improvement", "2.4x", "match"),
    ("2.4x improvement", "3x improvement", "mismatch"), ("about $50 of foregone benefit", "$50", "match"),
    ("2.4x improvement", "a big improvement", "mismatch"), ("a big improvement", "a big improvement", "match"),
    ("p99 of 500 ms", "500ms", "match"), ("KV2 cache", "cache", "mismatch"),  # a digit inside a name is not a quantity
])
def test_result_is_an_exact_match_after_trivial_normalisation(a, b, outcome):
    got = compare_result(a, b)
    assert got["outcome"] == outcome and got["comparator"] == "exact"  # no model, no similarity


# ------------------------------------------------------------------------ concept: identity then fuzzy


@pytest.mark.parametrize("a,b,outcome,comparator", [
    ("opportunity cost", "Opportunity Cost", "match", "identity"),
    ("opportunity cost", "opportunity costs", "match", "identity"),
    ("cost of opportunity", "opportunity cost of", "near", "fuzzy"),          # same words, another order
    ("opportunity cost", "cost of the next best alternative", "near", "synonym"),  # listed synonym
    ("price elasticity", "price elasticty", "near", "fuzzy"),                  # a typo: small edit distance
    ("opportunity cost", "sunk cost", "mismatch", "fuzzy"),
    ("supply", "demand", "mismatch", "fuzzy"),
    ("p99 goodput", "goodput", "near", "fuzzy"),                                # one names the other more narrowly
    ("PagedAttention's block table mechanism", "block tables", "near", "fuzzy"),
    ("PagedAttention's block table mechanism", "block-table memory management", "near", "fuzzy"),  # shares most of its words
    ("KV-cache, memory fragmentation, p99 tail latency", "memory fragmentation", "match", "identity"),  # a list: the best pair decides
    ("KV-cache, memory fragmentation, p99 tail latency", "fragmented memory", "mismatch", "fuzzy"),
    ("throughput", "latency", "mismatch", "fuzzy"),
    ("cost of capital", "opportunity cost", "mismatch", "fuzzy"),               # one shared word is not enough
])
def test_concept_is_identity_first_then_a_fuzzy_or_synonym_fallback(a, b, outcome, comparator):
    got = compare_concept(a, b)
    assert (got["outcome"], got["comparator"]) == (outcome, comparator)


# ------------------------------------------------------------------------- the null table


@pytest.mark.parametrize("field", ["concept", "result"])
def test_null_table_row_1_both_null_is_excluded_not_a_gap(field):
    assert compare_field(field, None, None)["status"] == "excluded"


@pytest.mark.parametrize("field,e,a", [("concept", "opportunity cost", "opportunity cost"), ("result", "$50", "$50")])
def test_null_table_row_2_both_present_are_compared_with_the_fields_comparator(field, e, a):
    got = compare_field(field, e, a)
    assert got["status"] == "compared" and got["outcome"] == "match"


@pytest.mark.parametrize("field", ["concept", "result", "claim"])
def test_null_table_row_3_expert_present_audience_null_is_the_gap(field):
    got = compare_field(field, "something", None)
    assert got["status"] == "gap" and got["outcome"] == "absent"


@pytest.mark.parametrize("field", ["concept", "result", "claim"])
def test_null_table_row_4_audience_introduced_something_is_over_reach_and_not_penalised(field):
    got = compare_field(field, None, "something the expert never said")
    assert got["status"] == "over_reach" and "outcome" not in got  # surfaced quietly, never scored


def test_a_slide_with_no_result_and_no_vehicle_is_never_a_gap_on_account_of_those_absences():
    expert = {"concept": "isotope", "claim": "Isotopes are atoms of one element with different neutron counts.", "result": None, "vehicle": None}
    novice = dict(expert)
    m = build_fieldwise_metrics({"novice": "n", "peer": "p", "expert": "e"}, structured(novice=novice, peer=dict(expert), expert=expert), {"novice": [], "peer": [], "expert": []})
    assert m["slide_profile"]["fields"] == ["concept", "claim"]
    for aud in ("novice", "peer"):
        assert m["comparisons"][aud]["result"]["status"] == "excluded"
        assert m["comparisons"][aud]["vehicle"]["status"] == "not_scored"
        assert m["reach"][aud] == "ok" and m["chart"][aud]["value"] == 1.0
    assert m["findings"] == []


# ---------------------------------------------------------------- vehicle: excluded from the score


def test_vehicle_is_recorded_and_shown_and_never_scored():
    for vehicle in (None, "a completely different example", EXPERT_F["vehicle"]):
        peer = {**PEER_F, "vehicle": vehicle}
        m = metrics(peer=peer)
        assert m["comparisons"]["peer"]["vehicle"]["status"] == "not_scored"
        assert m["comparisons"]["peer"]["vehicle"]["audience"] == vehicle  # recorded
        assert m["reach"]["peer"] == "ok" and m["chart"]["peer"]["value"] == 1.0  # ...and no effect on any score
    # even when the audience's vehicle is null and the expert's is present, that is not a gap
    assert metrics(peer={**PEER_F, "vehicle": None})["comparisons"]["peer"]["vehicle"]["status"] == "not_scored"


def test_the_tyler_versus_doja_cat_penalty_is_gone_deterministically():
    """The novice's takeaway shares proper nouns with the expert's. Under cosine that produced a
    mid-range score; here the vehicle is simply not in the comparison, and what remains is that the
    novice never reached the principle."""
    m = metrics()
    novice = m["comparisons"]["novice"]
    assert novice["concept"]["status"] == "gap" and novice["result"]["status"] == "gap" and novice["claim"]["outcome"] == "absent"
    assert m["reach"]["novice"] == "fail" and m["chart"]["novice"]["value"] == 0.0
    assert m["reach"]["peer"] == "ok"


# ------------------------------------------------------------------------------- slide profile


def test_the_slide_profile_is_defined_by_the_experts_populated_fields():
    p = slide_profile(EXPERT_F)
    assert p["fields"] == ["concept", "claim", "result", "vehicle"] and p["scored"] == ["concept", "claim", "result"] and p["thin"] is False
    assert p["text"] == "This slide names a principle, works through an example and reaches a numeric result."
    d = slide_profile({"concept": None, "claim": "A definition.", "result": None, "vehicle": None})
    assert d["text"] == "This slide states a claim; no example, no worked result." and d["thin"] is True
    assert slide_profile({f: None for f in C.FIELDS})["text"] == "This slide has no extractable structure."
    assert slide_profile({"concept": "x", "claim": None, "result": "the answer", "vehicle": None})["text"] == "This slide names a principle and reaches a stated result; no example."


# ------------------------------------------------------------------- example-bound (G1)


def test_example_bound_fires_for_the_novice_in_the_screenshot_case_and_quotes_its_evidence():
    m = metrics()
    (f,) = [x for x in m["findings"] if x["id"] == "example_bound"]
    assert f["audience"] == "novice"
    assert f["text"] == "Novice takeaway is example-bound: describes the example but never names the principle or reaches the answer."
    assert f["evidence"][0] == {"persona": "novice", "label": "Novice takeaway, verbatim", "text": NOVICE_T}
    assert {"persona": "expert", "label": "Expert concept, from its takeaway", "text": "opportunity cost"} in f["evidence"]
    assert not [x for x in m["findings"] if x["audience"] == "peer" and x["id"] == "example_bound"]


@pytest.mark.parametrize("aud,fires", [
    ({"concept": None, "claim": None, "result": None, "vehicle": "x"}, True),
    ({"concept": "opportunity cost", "claim": None, "result": None, "vehicle": "x"}, False),   # named the principle
    ({"concept": None, "claim": "A general point.", "result": None, "vehicle": "x"}, False),   # stated it generally
    ({"concept": None, "claim": None, "result": "$50", "vehicle": "x"}, False),                # reached the answer
])
def test_example_bound_rule(aud, fires):
    profile = slide_profile(EXPERT_F)
    assert example_bound(aud, EXPERT_F, profile) is fires


def test_example_bound_ignores_a_result_the_slide_does_not_have():
    expert = {"concept": "isotope", "claim": "Same element, different neutrons.", "result": None, "vehicle": None}
    aud = {"concept": None, "claim": None, "result": None, "vehicle": "carbon-12 and carbon-14"}
    assert example_bound(aud, expert, slide_profile(expert)) is True  # result is not in the profile, so its absence is not required...
    assert example_bound({**aud, "result": "6"}, expert, slide_profile(expert)) is True  # ...and one the expert never had does not excuse the miss
    text = [x for x in build_fieldwise_metrics(TAKEAWAYS, structured(novice=aud, expert=expert, peer=aud), {"novice": [], "peer": [], "expert": []})["findings"]][0]["text"]
    assert "reaches the answer" not in text


def test_example_bound_needs_a_principle_for_the_audience_to_have_missed():
    nothing = {f: None for f in C.FIELDS}
    assert example_bound(nothing, nothing, slide_profile(nothing)) is False


# ------------------------------------------------------------------ figure-dependent (G2)

FIG = ("Two lines on axes labelled Price and Quantity: one slopes downward, one slopes upward. They cross at a point marked P*, Q*. "
       "A shaded triangle sits above the crossing point.")


def test_figure_dependent_fires_when_the_expert_claim_references_the_figure_and_the_audience_claim_does_not():
    expert = {"claim": "Price and quantity settle where the crossing point is marked.", **{f: None for f in ("concept", "result", "vehicle")}}
    novice = {"claim": "Markets are complicated things.", **{f: None for f in ("concept", "result", "vehicle")}}
    got = figure_dependent(novice, expert, FIG)
    assert got and "price" in got["figure_terms"]
    assert figure_dependent({**novice, "claim": None}, expert, FIG)  # a null claim references nothing


def test_figure_dependent_does_not_fire_when_the_audience_engages_or_the_figure_is_not_substantial():
    expert = {"claim": "Price and quantity settle where the crossing point is marked.", "concept": None, "result": None, "vehicle": None}
    engaged = {"claim": "The price and quantity are set at the crossing.", "concept": None, "result": None, "vehicle": None}
    assert figure_dependent(engaged, expert, FIG) is None
    assert figure_dependent({"claim": None}, expert, "A chart.") is None  # not substantial
    assert figure_dependent({"claim": None}, expert, None) is None
    assert figure_dependent({"claim": None}, {"claim": "Something unrelated entirely."}, FIG) is None  # the expert does not use the figure either
    generic = {"claim": "The lines and axes are labelled.", "concept": None, "result": None, "vehicle": None}
    assert figure_dependent({"claim": None}, generic, FIG) is None  # sharing only generic figure words is not referencing it


def test_the_figure_finding_uses_the_specs_wording_and_quotes_both_claims():
    expert = {**{f: None for f in C.FIELDS}, "claim": "Price and quantity settle where the crossing point is marked."}
    novice = {**{f: None for f in C.FIELDS}, "claim": "Markets are complicated."}
    fs = C.build_findings({"novice": "n", "peer": "p", "expert": "e"}, {"novice": novice, "peer": expert, "expert": expert}, slide_profile(expert), FIG)
    (f,) = [x for x in fs if x["id"] == "figure_dependent"]
    assert f["audience"] == "novice"
    assert f["text"] == "The substance of this slide is in the figure, and the novice reading does not engage with it: the figure is carrying meaning it does not label."
    assert [e["text"] for e in f["evidence"]] == ["Markets are complicated.", expert["claim"]]


# --------------------------------------------- extraction: nulls, grounding, one call, no inference


def test_blank_and_placeholder_values_are_null_never_a_value():
    for bad in ("", "  ", "N/A", "n/a", "none", "None", "null", "unknown", "not stated", "-", "—", None):
        assert null_if_blank(bad) is None
    assert null_if_blank("  opportunity   cost ") == "opportunity cost"


def test_clean_fields_turns_placeholders_into_null():
    got, dropped = clean_fields({"concept": "N/A", "claim": "", "result": "none", "vehicle": "unknown"}, "The slide lists three things.")
    assert got == {"concept": None, "claim": None, "result": None, "vehicle": None} and dropped == []


def test_a_concept_the_persona_never_named_is_dropped_and_recorded():
    got, dropped = clean_fields({"concept": "opportunity cost", "claim": None, "result": None, "vehicle": None}, NOVICE_T)
    assert got["concept"] is None
    assert dropped == [{"field": "concept", "value": "opportunity cost", "reason": "the takeaway does not name this concept"}]


def test_a_result_the_persona_never_reached_is_dropped_but_one_it_wrote_is_kept():
    assert clean_fields({"result": "$75", "concept": None, "claim": None, "vehicle": None}, NOVICE_T)[0]["result"] is None
    assert clean_fields({"result": "$150", "concept": None, "claim": None, "vehicle": None}, NOVICE_T)[0]["result"] == "$150"
    assert clean_fields({"result": "$50", "concept": None, "claim": None, "vehicle": None}, "It costs $50.00 in total")[0]["result"] == "$50"


def test_a_restated_claim_may_paraphrase_but_not_smuggle_in_new_terms():
    ok, _ = clean_fields({"claim": "The cheaper option is worth choosing.", "concept": None, "result": None, "vehicle": None}, "Choosing the cheaper option is worth it.")
    assert ok["claim"]
    bad, dropped = clean_fields({"claim": "Stochastic eigenvalue blockchain arbitrage hedging.", "concept": None, "result": None, "vehicle": None}, NOVICE_T)
    assert bad["claim"] is None and "adds terms" in dropped[0]["reason"]


def test_a_vehicle_must_be_what_the_takeaway_describes():
    assert clean_fields({"vehicle": "Tyler at $150 versus Doja Cat", "concept": None, "claim": None, "result": None}, NOVICE_T)[0]["vehicle"]
    assert clean_fields({"vehicle": "a bakery selling sourdough loaves", "concept": None, "claim": None, "result": None}, NOVICE_T)[0]["vehicle"] is None


class Scripted:
    model = "fake-haiku"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user": user_text, "image": image_png, "schema": schema})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(novice=NOVICE_F, peer=PEER_F, expert=EXPERT_F, novice_cov=None, peer_cov=None, propositions=None):
    """What the model returns: fields, the expert claim's propositions, and each audience's judgements."""
    props = propositions if propositions is not None else ([{"id": "p1", "text": expert["claim"]}] if expert["claim"] else [])

    def cov(f, given):
        if given is not None:
            return given
        if not f["claim"] or not props:
            return {"judgements": [], "extra_assertions": [], "note": ""}
        return {"judgements": [{"id": "p1", "status": "covered", "evidence": f["claim"]}], "extra_assertions": [], "note": "Same."}

    return {"fields": {"novice": novice, "peer": peer, "expert": expert}, "propositions": props,
            "coverage": {"novice": cov(novice, novice_cov), "peer": cov(peer, peer_cov)}}


async def test_one_call_per_slide_takes_all_three_takeaways_and_never_the_image():
    c = Scripted(reply())
    got = await structure_slide(c, TAKEAWAYS)
    (call,) = c.calls
    for t in TAKEAWAYS.values():
        assert t in call["user"]
    assert call["image"] is None and set(call["schema"]["properties"]) == {"fields", "propositions", "coverage"}  # clearly separated sections
    assert got["fields"]["novice"] == NOVICE_F and got["meta"]["attempts"] == 1 and got["meta"]["model"] == "fake-haiku"
    assert [p["text"] for p in got["propositions"]] == [EXPERT_F["claim"]]


async def test_the_prompt_forbids_inference_and_placeholders_and_demands_general_claims():
    c = Scripted(reply())
    await structure_slide(c, TAKEAWAYS)
    system = c.calls[0]["system"]
    for rule in ("ONLY from what that takeaway says", "Never infer a concept", "Never supply a result", "NEVER an empty string", '"N/A"',
                 "in GENERAL terms", "claim is null and the example goes in vehicle", "Treat all three takeaways identically"):
        assert rule in system, rule


async def test_an_extractor_that_fills_blanks_is_corrected_in_code():
    filled = {"concept": "N/A", "claim": "", "result": "none", "vehicle": NOVICE_F["vehicle"]}
    got = await structure_slide(Scripted(reply(novice=filled)), TAKEAWAYS)
    assert got["fields"]["novice"] == {"concept": None, "claim": None, "result": None, "vehicle": NOVICE_F["vehicle"]}





async def test_coverage_for_a_claim_the_grounding_check_nulled_is_ignored():
    # the model invented a claim full of new terms; grounding drops it, so the reader has no claim and is absent
    hallucinated = {**PEER_F, "claim": "Stochastic eigenvalue blockchain arbitrage hedging."}
    got = await structure_slide(Scripted(reply(peer=hallucinated)), TAKEAWAYS)
    assert got["fields"]["peer"]["claim"] is None
    m = build_fieldwise_metrics(TAKEAWAYS, got, {"novice": [], "peer": [], "expert": []})
    assert m["comparisons"]["peer"]["claim"]["outcome"] == "absent"


def test_structurings_are_cached_by_the_three_takeaways(tmp_path):
    cache = FileStructureCache(tmp_path)
    assert cache.get(TAKEAWAYS) is None
    cache.put(TAKEAWAYS, structured())
    assert cache.get(TAKEAWAYS)["meta"]["model"] == "fake-sonnet"
    assert cache.get({**TAKEAWAYS, "novice": "a different takeaway"}) is None


def test_the_schema_stays_inside_the_apis_limit_on_nullable_parameters():
    """A live call was rejected once for using 20 union-typed parameters against a limit of 16."""
    def unions(node):
        if isinstance(node, dict):
            return (1 if "anyOf" in node or isinstance(node.get("type"), list) else 0) + sum(unions(v) for v in node.values())
        if isinstance(node, list):
            return sum(unions(v) for v in node)
        return 0

    assert unions(C.STRUCTURE_SCHEMA) <= 16


def test_the_payload_is_plain_json_and_carries_the_texts_every_state_traces_to():
    import json

    m = metrics()
    json.dumps(m)
    assert m["comparator"] == "fieldwise" and m["takeaways"] == TAKEAWAYS and m["intent"] == EXPERT_T
    claim = m["comparisons"]["peer"]["claim"]
    assert claim["expert"] == EXPERT_F["claim"] and claim["audience"] == PEER_F["claim"]
    assert claim["meaning"] == C.STATE_MEANING["equivalent"] and claim["outcome"] == "equivalent"
    assert claim["coverage"]["propositions"][0]["evidence"] == PEER_F["claim"]  # the quoted span it traces to
    assert m["claim_comparison"] == "coverage" and m["term_gap"]["novice_unresolved"] == ["p99"] and "verdicts" not in m
