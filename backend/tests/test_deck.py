from __future__ import annotations

import json

import pytest

from sightline import deck
from sightline.audiences import AudienceResponseError, CacheMiss
from sightline.deck import error_record, rollup

from builders import slide_result


def make_deck(n=6, overrides=None):
    """Slide i has expert alignment 0.9 and novice alignment 0.9 - 0.05*i, so the novice-expert
    gap grows with the slide number. `overrides[i]` are extra keyword args for slide i."""
    overrides = overrides or {}
    return [
        slide_result(i, align=(0.9 - 0.05 * i, 0.8, 0.9), **overrides.get(i, {})) for i in range(1, n + 1)
    ]


def test_gap_ranking_puts_the_widest_novice_expert_gap_first():
    r = rollup(make_deck())
    assert [g["slide"] for g in r["gap_ranking"]] == [6, 5, 4, 3, 2, 1]
    assert r["gap_ranking"][0]["gap"] == pytest.approx(0.30)
    assert r["gap_ranking"][0]["rank"] == 1


def test_every_value_is_a_plain_json_number():
    json.dumps(rollup(make_deck()))  # numpy scalars would raise here


def test_position_is_rank_within_this_deck_with_ties_sharing_a_rank():
    d = [slide_result(i, conf=(0.5, 0.5, c)) for i, c in enumerate((0.9, 0.9, 0.7, 0.5, 0.5), 1)]
    r = rollup(d)
    ranks = {p["slide"]: p["position"]["confidence.expert"] for p in r["per_slide"]}
    assert [ranks[i]["rank"] for i in range(1, 6)] == [1, 1, 3, 4, 4]
    assert all(v["of"] == 5 for v in ranks.values())


def test_distribution_keeps_the_per_slide_values_it_was_computed_from():
    dist = rollup(make_deck())["distributions"]["intent_alignment.expert"]
    assert dist["n"] == 6 and dist["median"] == pytest.approx(0.9)
    assert [v["slide"] for v in dist["values"]] == [1, 2, 3, 4, 5, 6]


def test_small_decks_are_not_ranked_against_themselves():
    r = rollup(make_deck(n=deck.MIN_SLIDES_FOR_COMPARISON - 1))
    assert r["comparable"] is False
    assert r["divergence_top"] == []
    assert not any(n["id"] == "widest_gaps" for n in r["notes"])


def test_divergence_top_names_the_highest_slides_of_a_big_enough_deck():
    # spread of takeaway angles => spread of divergence; slide 3 is the most divergent
    d = [slide_result(i, align=a) for i, a in enumerate(
        [(0.5, 0.5, 0.5), (0.5, 0.6, 0.5), (0.1, 0.5, 0.95), (0.5, 0.5, 0.6), (0.4, 0.5, 0.5), (0.5, 0.5, 0.5)], 1)]
    r = rollup(d)
    assert r["comparable"] is True
    assert r["divergence_top"][0]["slide"] == 3
    assert len(r["divergence_top"]) == deck.TOP_DIVERGENCE_SLIDES


def test_terms_are_ranked_by_how_many_slides_the_novice_fails_on():
    d = make_deck(
        4,
        {
            1: {"terms": (["KV-cache", "TTFT"], [], [])},
            2: {"terms": (["kv cache"], [], [])},  # same term, different spelling
            3: {"terms": (["KV-cache", "goodput"], [], [])},
            4: {"terms": ([], ["TTFT"], [])},  # peer's terms are not the novice's
        },
    )
    terms = rollup(d)["terms"]
    assert [(t["key"], t["count"]) for t in terms] == [("kv cache", 3), ("ttft", 1), ("goodput", 1)]  # ties: earliest slide first
    kv = terms[0]
    assert kv["slides"] == [1, 2, 3] and kv["first_slide"] == 1
    assert kv["term"] == "KV-cache"  # the spelling most slides used
    assert [o["as_written"] for o in kv["occurrences"]] == ["KV-cache", "kv cache", "KV-cache"]


def test_a_term_listed_twice_on_one_slide_counts_once():
    d = [slide_result(1, terms=(["TTFT", "ttft"], [], []))]
    assert rollup(d)["terms"][0]["count"] == 1


def test_recurring_terms_produce_a_note_that_carries_its_evidence():
    d = make_deck(
        3, {1: {"terms": (["SLO"], [], [])}, 2: {"terms": (["SLO", "TTFT"], [], [])}, 3: {"terms": (["TTFT"], [], [])}}
    )
    (note,) = rollup(d)["notes"]
    assert note["id"] == "recurring_terms"
    assert "3 slides share 2 terms" in note["text"] and "by slide 1" in note["text"]
    assert {t["term"] for t in note["evidence"]["terms"]} == {"SLO", "TTFT"}


def test_no_recurring_terms_means_no_advice():
    assert rollup(make_deck(3, {1: {"terms": (["A"], [], [])}, 2: {"terms": (["B"], [], [])}}))["notes"] == []


def test_arc_follows_slide_order_with_alignment_per_persona():
    arc = rollup(make_deck(3))["arc"]
    assert [a["slide"] for a in arc] == [1, 2, 3]
    assert arc[0]["novice"] == pytest.approx(0.85) and arc[0]["expert"] == pytest.approx(0.9)


def test_a_failed_persona_leaves_the_slide_unscored_but_keeps_its_other_readings():
    good = slide_result(1)
    bad = slide_result(2)
    bad["readings"]["novice"] = error_record(
        AudienceResponseError("novice", ["confidence must be in [0, 1], got 1.7"])
    )
    bad["metrics"], bad["metrics_error"] = None, "novice response invalid"

    r = rollup([good, bad])

    assert r["unscored"] == [2] and r["n_scored"] == 1
    row = next(p for p in r["per_slide"] if p["slide"] == 2)
    assert "confidence.expert" in row["values"]  # peer/expert readings still count
    assert "confidence.novice" not in row["values"] and "audience_divergence" not in row["values"]
    assert [g["slide"] for g in r["gap_ranking"]] == [1]
    assert r["arc"][1] == {"slide": 2, "novice": None, "peer": None, "expert": None}


def test_rollup_of_nothing_is_empty_not_an_error():
    r = rollup([])
    assert r["n_slides"] == 0 and r["gap_ranking"] == [] and r["terms"] == [] and r["notes"] == []


def test_input_order_does_not_matter():
    d = make_deck()
    assert rollup(list(reversed(d))) == rollup(d)


# ------------------------------------------------------------------------ error records


def test_invalid_response_error_keeps_every_attempt_verbatim():
    e = error_record(AudienceResponseError("expert", ["first problem", "confidence must be in [0, 1], got 1.7"]))
    assert e["ok"] is False
    assert e["error"]["kind"] == "invalid_response"
    assert e["error"]["message"] == "confidence must be in [0, 1], got 1.7"
    assert e["error"]["attempts"] == ["first problem", "confidence must be in [0, 1], got 1.7"]


def test_other_failures_are_classified():
    assert error_record(CacheMiss("nothing cached"))["error"]["kind"] == "offline_cache_miss"
    e = error_record(RuntimeError("boom"))["error"]
    assert e["kind"] == "call_failed" and e["message"] == "RuntimeError: boom"
