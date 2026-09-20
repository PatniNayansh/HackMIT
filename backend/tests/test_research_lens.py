"""research_lens.py: citation-backed hypotheses layered on top of dmn_drive. Pure logic --
no model call, no neural data -- so this is fully testable independent of whether TRIBE v2
has actually been run yet.
"""

from __future__ import annotations

from sightline.research_lens import (
    ADHD_DMN_CITATIONS,
    ADHD_DMN_Z_THRESHOLD,
    AUTISM_ISC_CITATIONS,
    ResearchLens,
    adhd_dmn_lens,
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
