"""Tiers, the hardest-slides ranking and the arc, from the field-wise signals."""

from __future__ import annotations

import json

import pytest

from profe import compare as C
from profe.deck import rollup
from profe.tiers import CONFIG, assign_tiers_fieldwise

from builders import fw_slide_result

FULL = {"concept": "isotope", "claim": "Isotopes are atoms of one element with different neutron counts.", "result": "6", "vehicle": "carbon-12 and carbon-14"}
NONE = {"concept": None, "claim": None, "result": None, "vehicle": "carbon-12 and carbon-14"}


def tiers_for(results):
    r = rollup(results)
    return {p["slide"]: p["tier"]["tier"] for p in r["per_slide"] if "tier" in p}


def deck(n=6, **kw):
    return [fw_slide_result(i, **kw) for i in range(1, n + 1)]


def test_self_contained_when_the_novice_and_peer_reach_the_point_and_terms_are_low():
    assert set(tiers_for(deck(6)).values()) == {"self_contained"}


def test_background_needed_when_the_peer_reaches_it_and_the_novice_does_not():
    results = deck(5) + [fw_slide_result(6, novice=dict(NONE))]
    assert tiers_for(results)[6] == "background_needed"


def test_expert_gated_when_neither_the_novice_nor_the_peer_reaches_it():
    results = deck(5) + [fw_slide_result(6, novice=dict(NONE), peer=dict(NONE))]
    assert tiers_for(results)[6] == "expert_gated"


def test_an_under_specified_peer_still_counts_as_aligned_but_an_under_specified_novice_does_not():
    r = deck(5) + [fw_slide_result(6, novice=dict(FULL, result=None, concept=None), peer=dict(FULL, result=None, concept=None), states={"novice": "under-specified", "peer": "under-specified"})]
    assert tiers_for(r)[6] == "background_needed"  # novice: partial (not aligned); peer: partial (aligned)
    both_fail = deck(5) + [fw_slide_result(6, novice=dict(NONE), peer=dict(FULL, claim=None, concept=None, result=None))]
    assert tiers_for(both_fail)[6] == "expert_gated"


def test_an_over_claiming_reader_reached_the_point():
    r = deck(5) + [fw_slide_result(6, states={"novice": "over-claimed"})]
    assert tiers_for(r)[6] == "self_contained"


def test_a_novice_who_reaches_it_but_meets_many_unknown_terms_still_needs_background():
    many = tuple(f"t{i}" for i in range(9))
    r = [fw_slide_result(i, unresolved=((), (), ())) for i in range(1, 6)] + [fw_slide_result(6, unresolved=(many, (), ()))]
    assert tiers_for(r)[6] == "background_needed" and tiers_for(r)[1] == "self_contained"


def test_only_the_fields_the_expert_populated_are_read_a_missing_result_is_not_a_gap():
    """The expert has no result and no vehicle. Nobody is faulted for lacking them."""
    expert = {"concept": "isotope", "claim": FULL["claim"], "result": None, "vehicle": None}
    aud = dict(expert)
    r = deck(5) + [fw_slide_result(6, novice=aud, peer=aud, expert=expert)]
    assert tiers_for(r)[6] == "self_contained"
    m = r[-1]["metrics"]
    assert m["slide_profile"]["scored"] == ["concept", "claim"]
    checks = rollup(r)["per_slide"][-1]["tier"]["checks"]
    assert not any("result" in c["name"] for c in checks)  # the checks are over the profile's fields only


def test_over_reach_is_not_penalised_in_the_tier():
    expert = {"concept": "isotope", "claim": FULL["claim"], "result": None, "vehicle": None}
    novice = dict(expert, result="12")  # the novice produced a result the expert did not
    r = deck(5) + [fw_slide_result(6, novice=novice, expert=expert, peer=dict(expert))]
    assert r[-1]["metrics"]["comparisons"]["novice"]["result"]["status"] == "over_reach"
    assert tiers_for(r)[6] == "self_contained"


def test_a_slide_with_no_scored_field_has_no_tier():
    blank = {"concept": None, "claim": None, "result": None, "vehicle": "just an example"}
    r = deck(5) + [fw_slide_result(6, novice=dict(blank), peer=dict(blank), expert=dict(blank))]
    assert 6 not in tiers_for(r) and len(tiers_for(r)) == 5


def test_a_short_deck_uses_the_absolute_term_threshold_and_says_so():
    r = rollup(deck(3))
    tier = r["per_slide"][0]["tier"]
    assert tier["basis"] == "absolute" and tier["comparator"] == "fieldwise"
    assert tier["checks"][-1]["name"] == "novice_terms_low" and "percentile" not in tier["checks"][-1]
    assert rollup(deck(6))["per_slide"][0]["tier"]["basis"] == "relative"


def test_every_check_is_plain_words_and_names_the_text_it_reads():
    tier = rollup(deck(5) + [fw_slide_result(6, novice=dict(NONE))])["per_slide"][-1]["tier"]
    lines = [c["text"] for c in tier["checks"]]
    assert any("Novice claim covers none of the expert's 2 propositions (absent)." == t for t in lines)
    assert any("did not reach the concept; the expert reached “isotope”." in t for t in lines)
    assert any("did not reach the result; the expert reached “6”." in t for t in lines)


def test_thresholds_for_the_terms_check_live_in_the_one_config_dict():
    assert set(CONFIG["fieldwise"]) == {"min_slides_for_relative", "relative", "absolute"}


# -------------------------------------------------------------------------------- hardest


def test_hardest_slides_are_ranked_by_the_novices_rung_then_missing_fields_then_terms():
    r = rollup([
        fw_slide_result(1),                                                                # equivalent (1.0)
        fw_slide_result(2, states={"novice": "under-specified"}),                           # 1 of 2 propositions (0.5)
        fw_slide_result(3, novice=dict(NONE)),                                              # absent (0.0), 3 gaps
        fw_slide_result(4, novice=dict(FULL, claim=None)),                                  # absent (0.0), 1 gap... claim absent only
        fw_slide_result(5, states={"novice": "over-claimed"}),                                # all covered plus extra (1.0)
        fw_slide_result(6, novice=dict(NONE), unresolved=(("a", "b", "c"), (), ())),        # absent, 3 gaps, more terms
    ])["hardest"]
    # values: 6, 3, 4 are 0.0 (absent); 2 is 0.5; 1 and 5 are 1.0. Ties break on missing fields, then terms.
    assert [h["slide"] for h in r] == [6, 3, 4, 2, 1, 5]
    assert [h["novice_state"] for h in r] == ["absent", "absent", "absent", "under-specified", "equivalent", "over-claimed"]
    assert [h["novice_value"] for h in r] == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
    assert (r[3]["novice_covered"], r[3]["novice_total"]) == (1, 2)
    assert r[0]["novice_gaps"] == 3 and r[2]["novice_gaps"] == 1 and r[0]["novice_unresolved"] == 3


# ----------------------------------------------------------------------------------- arc


def test_the_arc_is_propositions_covered_over_total_and_the_expert_is_at_the_top_by_definition():
    r = rollup([
        fw_slide_result(1),
        fw_slide_result(2, states={"novice": "under-specified", "peer": "over-claimed"}),
        fw_slide_result(3, novice=dict(NONE), states={"peer": "divergent"}),
    ])
    arc = r["arc"]
    assert [(a["novice"], a["peer"], a["expert"]) for a in arc] == [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (0.0, 0.0, 1.0)]  # propositions covered / total; divergent is 0
    assert arc[1]["counts"]["novice"] == {"covered": 1, "total": 2} and arc[0]["counts"]["peer"] == {"covered": 2, "total": 2}
    assert arc[2]["states"] == {"novice": "absent", "peer": "divergent", "expert": "equivalent"}
    assert r["definitional"] == ["expert"] and r["comparator"] == "fieldwise" and r["arc_kind"] == "coverage"


def test_a_thin_slide_is_marked_and_its_profile_named_not_treated_as_low_comprehension():
    thin = {"concept": None, "claim": "Isotopes are atoms of one element with different neutron counts.", "result": None, "vehicle": None}
    r = rollup([fw_slide_result(1), fw_slide_result(2, novice=dict(thin), peer=dict(thin), expert=dict(thin))])
    a1, a2 = r["arc"]
    assert a1["thin"] is False and a2["thin"] is True
    assert a2["profile"] == "This slide states a claim; no example, no worked result."
    assert a2["novice"] == 1.0  # a thin slide where the novice covered everything is at the top, not penalised for what it lacks


def test_a_slide_whose_expert_made_no_claim_has_nothing_to_count_and_no_point():
    no_claim = {"concept": "isotope", "claim": None, "result": None, "vehicle": None}
    r = rollup([fw_slide_result(1), fw_slide_result(2, novice=dict(no_claim), peer=dict(no_claim), expert=dict(no_claim))])
    assert r["arc"][1]["novice"] is None and r["arc"][1]["peer"] is None and r["arc"][1]["expert"] is None
    assert [h["slide"] for h in r["hardest"]] == [1]  # nothing to rank it by


def test_a_slide_without_a_scored_field_has_no_point_on_the_arc():
    blank = {"concept": None, "claim": None, "result": None, "vehicle": "x"}
    (a,) = rollup([fw_slide_result(1, novice=dict(blank), peer=dict(blank), expert=dict(blank))])["arc"]
    assert a["novice"] is None and a["peer"] is None


def test_the_fieldwise_rollup_has_no_scalar_alignment_and_is_plain_json():
    r = rollup(deck(6))
    json.dumps(r)
    assert not [k for k in r["distributions"] if k.startswith("intent_alignment")]
    assert {"unresolved_count.novice", "unresolved_count.peer", "unresolved_count.expert"} <= set(r["distributions"])
    assert r["comparable"] is True and r["notes"] and all(n["id"] in ("recurring_terms", "hardest_slides") for n in r["notes"])


def test_the_deck_relative_reading_is_kept_for_terms():
    r = rollup(deck(6, unresolved=(("a", "b"), (), ())))
    pos = r["per_slide"][0]["position"]["unresolved_count.novice"]
    assert pos["of"] == 6 and r["distributions"]["unresolved_count.novice"]["comparable"] is True


def test_a_failed_slide_stays_out_of_every_field_wise_series():
    bad = fw_slide_result(2)
    bad["metrics"], bad["metrics_error"] = None, "not computed: the field-wise structuring call failed"
    r = rollup([fw_slide_result(1), bad])
    assert r["unscored"] == [2] and [h["slide"] for h in r["hardest"]] == [1]
    assert r["arc"][1] == {"slide": 2, "novice": None, "peer": None, "expert": None}
