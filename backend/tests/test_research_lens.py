"""research_lens.py: citation-backed hypotheses layered on top of dmn_drive. Pure logic --
no model call, no neural data -- so this is fully testable independent of whether TRIBE v2
has actually been run yet.
"""

from __future__ import annotations

from sightline.research_lens import (
    ADHD_DMN_CITATIONS,
    ADHD_DMN_Z_THRESHOLD,
    AUTISM_ISC_CITATIONS,
    DEPRESSION_DMN_CITATIONS,
    DEPRESSION_DMN_Z_THRESHOLD,
    DYSLEXIA_LANGUAGE_CITATIONS,
    DYSLEXIA_LANGUAGE_Z_THRESHOLD,
    ResearchLens,
    adhd_dmn_lens,
    depression_dmn_lens,
    dyslexia_language_lens,
)


def test_elevated_dmn_drive_yields_the_adhd_hypothesis_with_citations():
    lens = adhd_dmn_lens(dmn_drive_z=1.8, slide_index=4)
    assert lens.population == "ADHD"
    assert "Slide 4" in lens.finding
    assert "1.80" in lens.finding
    assert "research-grounded hypothesis" in lens.finding
    assert lens.citations == ADHD_DMN_CITATIONS
    assert len(lens.citations) >= 3


def test_low_dmn_drive_makes_no_claim_in_either_direction():
    lens = adhd_dmn_lens(dmn_drive_z=0.1, slide_index=2)
    assert "no elevated" in lens.finding
    assert "hypothesis for where ADHD viewers" not in lens.finding
    # still cited, so the reader can see what threshold and literature this is judged against
    assert lens.citations == ADHD_DMN_CITATIONS


def test_threshold_boundary_is_exclusive_not_inclusive():
    at_threshold = adhd_dmn_lens(dmn_drive_z=ADHD_DMN_Z_THRESHOLD, slide_index=1)
    just_above = adhd_dmn_lens(dmn_drive_z=ADHD_DMN_Z_THRESHOLD + 0.01, slide_index=1)
    assert "no elevated" in at_threshold.finding
    assert "research-grounded hypothesis" in just_above.finding


def test_label_names_the_population_and_disclaims_simulation():
    lens = adhd_dmn_lens(dmn_drive_z=2.0, slide_index=1)
    assert "ADHD" in lens.label
    assert "not a simulation" in lens.label


def test_to_dict_is_json_shaped_and_round_trips_citations():
    import json

    lens = adhd_dmn_lens(dmn_drive_z=1.5, slide_index=3)
    d = json.loads(json.dumps(lens.to_dict()))
    assert d["population"] == "ADHD"
    assert d["citations"] == list(ADHD_DMN_CITATIONS)


def test_no_computed_autism_lens_exists_only_a_citation_for_the_methods_panel():
    """Deliberate absence, not an oversight: see the module docstring for why an ISC
    finding can't be derived from TRIBE v2's single deterministic mean. This test exists so
    that if someone adds `autism_isc_lens(...)` later without reading the docstring, at
    least one test fails and asks them to justify it."""
    import sightline.research_lens as module

    assert not hasattr(module, "autism_isc_lens")
    assert AUTISM_ISC_CITATIONS  # the citation itself is still kept, just not computed from


def test_result_is_immutable():
    lens = adhd_dmn_lens(dmn_drive_z=2.0, slide_index=1)
    assert isinstance(lens, ResearchLens)
    try:
        lens.population = "something else"  # type: ignore[misc]
        assert False, "ResearchLens should be frozen"
    except AttributeError:
        pass


# ------------------------------------------------------------------------ depression lens


def test_elevated_dmn_drive_yields_the_depression_hypothesis_with_its_own_citations():
    lens = depression_dmn_lens(dmn_drive_z=1.8, slide_index=4)
    assert lens.population == "depression"
    assert "Slide 4" in lens.finding
    assert "research-grounded hypothesis" in lens.finding
    assert lens.citations == DEPRESSION_DMN_CITATIONS
    assert lens.citations != ADHD_DMN_CITATIONS  # distinct, independently-cited population


def test_depression_low_dmn_drive_makes_no_claim():
    lens = depression_dmn_lens(dmn_drive_z=0.1, slide_index=2)
    assert "no elevated" in lens.finding
    assert lens.citations == DEPRESSION_DMN_CITATIONS


def test_depression_threshold_boundary_is_exclusive():
    at_threshold = depression_dmn_lens(dmn_drive_z=DEPRESSION_DMN_Z_THRESHOLD, slide_index=1)
    just_above = depression_dmn_lens(dmn_drive_z=DEPRESSION_DMN_Z_THRESHOLD + 0.01, slide_index=1)
    assert "no elevated" in at_threshold.finding
    assert "research-grounded hypothesis" in just_above.finding


def test_adhd_and_depression_agree_on_the_same_slide_since_they_share_a_signal():
    """Not a coincidence to hide: both lenses read the same dmn_drive_z, by design (see the
    module docstring). A slide that trips one trips the other, with separate citations."""
    z = 2.5
    adhd = adhd_dmn_lens(z, slide_index=1)
    depression = depression_dmn_lens(z, slide_index=1)
    assert "research-grounded hypothesis" in adhd.finding
    assert "research-grounded hypothesis" in depression.finding
    assert adhd.population != depression.population
    assert set(adhd.citations).isdisjoint(depression.citations)


# --------------------------------------------------------------------------- dyslexia lens


def test_low_language_drive_yields_the_dyslexia_hypothesis():
    lens = dyslexia_language_lens(language_drive_z=-1.8, slide_index=6)
    assert lens.population == "dyslexia"
    assert "Slide 6" in lens.finding
    assert "-1.80" in lens.finding
    assert "underactivation" in lens.finding
    assert lens.citations == DYSLEXIA_LANGUAGE_CITATIONS


def test_dyslexia_runs_the_opposite_direction_from_the_dmn_lenses():
    """High language drive is NOT the citable direction for dyslexia -- underactivation is
    the published finding, so a high Z-score must make no claim, same discipline as the
    "no signal" branch of the DMN lenses but on the opposite side of zero."""
    high = dyslexia_language_lens(language_drive_z=2.0, slide_index=1)
    assert "no unusually low" in high.finding
    assert "research-grounded hypothesis" not in high.finding


def test_dyslexia_threshold_boundary_is_exclusive():
    at_threshold = dyslexia_language_lens(language_drive_z=DYSLEXIA_LANGUAGE_Z_THRESHOLD, slide_index=1)
    just_below = dyslexia_language_lens(language_drive_z=DYSLEXIA_LANGUAGE_Z_THRESHOLD - 0.01, slide_index=1)
    assert "no unusually low" in at_threshold.finding
    assert "underactivation" in just_below.finding


def test_dyslexia_label_names_the_population_and_disclaims_simulation():
    lens = dyslexia_language_lens(language_drive_z=-2.0, slide_index=1)
    assert "dyslexia" in lens.label
    assert "not a simulation" in lens.label
