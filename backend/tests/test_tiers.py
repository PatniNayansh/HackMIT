from __future__ import annotations

import copy

import pytest

from profe import deck, tiers
from profe.tiers import CONFIG, assign_tiers, percentile

from builders import slide_result


def deck_inputs(rows):
    return {i: {"novice_alignment": n, "peer_alignment": p, "novice_unresolved": t} for i, (n, p, t) in enumerate(rows, 1)}


def test_every_threshold_lives_in_the_one_config_dict():
    """A cut-off written into the code would be untunable. The only numeric literals allowed in
    the logic are the 0.5 tie weight and the percentile/ordinal arithmetic."""
    assert set(CONFIG) == {"min_slides_for_relative", "relative", "absolute", "fieldwise", "tripwire"}
    assert set(CONFIG["relative"]) == {"novice_aligned_min_pct", "peer_aligned_min_pct", "novice_terms_low_max_pct"}
    assert set(CONFIG["absolute"]) == {"novice_aligned_min", "peer_aligned_min", "novice_terms_low_max"}


def test_percentile_is_the_share_below_with_ties_as_half():
    pop = [1, 2, 3, 3, 5]
    assert percentile(1, pop) == pytest.approx(0.1) and percentile(3, pop) == pytest.approx(0.6)
    assert percentile(5, pop) == pytest.approx(0.9) and percentile(7, [7]) == 0.5


# Slides 1-7 spread across the deck: a slide's tier depends on where it sits among the others.
DECK = deck_inputs([
    (0.34, 0.59, 3),   # 1  novice lowest, peer mid
    (0.42, 0.43, 2),   # 2  novice low, peer lowish
    (0.47, 0.29, 10),  # 3  peer lowest, many terms
    (0.60, 0.70, 3),   # 4
    (0.69, 0.68, 3),   # 5
    (0.70, 0.87, 1),   # 6  everything high, few terms
    (0.60, 0.64, 0),   # 7
])


def test_tiers_are_assigned_from_position_in_the_deck():
    t = assign_tiers(DECK)
    assert {i: r["tier"] for i, r in t.items()} == {
        1: "background_needed", 2: "background_needed", 3: "expert_gated", 4: "background_needed",
        5: "background_needed", 6: "self_contained", 7: "self_contained",
    }
    assert all(r["basis"] == "relative" and r["n_slides"] == 7 for r in t.values())


def test_the_same_numbers_get_a_different_tier_in_a_different_deck():
    """The level on one slide is not trusted; only its place among its deck's slides is."""
    low = deck_inputs([(0.30, 0.30, 5)] * 4 + [(0.20, 0.20, 9)])   # a deck where 0.30 is the norm
    high = deck_inputs([(0.30, 0.30, 5)] + [(0.60, 0.60, 1)] * 4)  # a deck where 0.30 is far below the rest
    assert assign_tiers(low)[1]["tier"] != assign_tiers(high)[1]["tier"]


def test_a_slide_that_falls_below_its_decks_novice_and_peer_is_expert_gated():
    rows = [(0.7, 0.7, 1)] * 5 + [(0.2, 0.2, 8)]
    assert assign_tiers(deck_inputs(rows))[6]["tier"] == "expert_gated"


def test_a_novice_who_aligns_but_meets_many_unknown_terms_still_needs_background():
    rows = [(0.5, 0.5, 1), (0.5, 0.5, 1), (0.5, 0.5, 2), (0.5, 0.5, 2), (0.9, 0.9, 9), (0.4, 0.5, 1)]
    assert assign_tiers(deck_inputs(rows))[5]["tier"] == "background_needed"


def test_a_short_deck_falls_back_to_absolute_thresholds_and_says_so():
    t = assign_tiers(deck_inputs([(0.7, 0.7, 0), (0.45, 0.6, 3), (0.2, 0.2, 9)]))
    assert [t[i]["tier"] for i in (1, 2, 3)] == ["self_contained", "background_needed", "expert_gated"]
    assert all(r["basis"] == "absolute" for r in t.values())
    assert all("percentile" not in c for r in t.values() for c in r["checks"])


def test_a_single_slide_view_uses_the_absolute_fallback():
    (only,) = assign_tiers(deck_inputs([(0.7, 0.7, 0)])).values()
    assert only["tier"] == "self_contained" and only["basis"] == "absolute"


def test_thresholds_can_be_tuned_from_the_config_alone():
    cfg = copy.deepcopy(CONFIG)
    cfg["relative"]["novice_aligned_min_pct"] = 0.99  # almost nobody counts as aligned
    baseline = assign_tiers(DECK)
    stricter = assign_tiers(DECK, cfg)
    assert baseline[6]["tier"] == "self_contained" and stricter[6]["tier"] == "background_needed"


def test_each_record_explains_every_check_in_plain_words():
    r = assign_tiers(DECK)[3]
    assert [c["name"] for c in r["checks"]] == ["novice_aligned", "peer_aligned", "novice_terms_low"]
    assert r["label"] == "Expert-gated" and r["meaning"] == "Only a specialist recovers the intended point."
    assert "percentile of these 7 slides" in r["checks"][0]["text"] and r["checks"][0]["passed"] is False
    assert r["checks"][2]["value"] == 10


def test_the_rollup_attaches_a_tier_to_each_scored_slide_and_nothing_to_unscored_ones():
    good = [slide_result(i, align=(0.3 + 0.08 * i, 0.5 + 0.05 * i), terms=([str(x) for x in range(8 - i)], [], [])) for i in range(1, 7)]
    bad = slide_result(7)
    bad["metrics"], bad["metrics_error"] = None, "not computed"
    r = deck.rollup([*good, bad])
    by_slide = {p["slide"]: p for p in r["per_slide"]}
    assert all(by_slide[i]["tier"]["tier"] in tiers.TIERS for i in range(1, 7))
    assert "tier" not in by_slide[7]
