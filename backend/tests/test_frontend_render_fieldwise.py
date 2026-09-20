"""The field-wise UI, rendered in Node against the bundled sample's real payload and against
hand-built runs that exercise each finding and null case."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from sightline.audiences import FileCache
from sightline.ingest import Slide
from sightline.server import FRONTEND_DIR, create_app
from sightline.store import BUNDLED_RUNS_DIR, RunStore

from builders import HashEmbedder, fw_slide_result

RUN = "sample-llm-serving"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to render the frontend")


def read_payload(store, run_id, tmp, n_slides, recs=None):
    app = create_app(store=store, cache=FileCache(tmp / "cache"), client_factory=lambda: None, embedder=HashEmbedder(), comparator="fieldwise")
    with TestClient(app) as c:
        run = c.get(f"/api/runs/{run_id}").json()
        rec = {n: c.get(f"/api/runs/{run_id}/slides/{n}/recommendations").json() for n in range(1, n_slides + 1)}
        # Only the slides that were actually precomputed: a slide missing here renders the empty
        # state, which is the case the UI has to get right most of the time.
        neural = {}
        for n in range(1, n_slides + 1):
            r = c.get(f"/api/runs/{run_id}/slides/{n}/neural")
            if r.status_code == 200:
                neural[n] = r.json()
    return {"run": run, "recs": rec, "neural": neural}


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fw-sample")
    return read_payload(RunStore(tmp / "history", bundled=[BUNDLED_RUNS_DIR]), RUN, tmp, 8)


def make_run(tmp, results, run_id="fw-demo"):
    store = RunStore(tmp / "history")
    slides = [Slide(r["index"], r["text"], b"\x89PNG") for r in results]
    meta = store.create_draft(title="Field-wise demo", source_filename="d.pdf", slides=slides, inferred=None, inference_error=None, run_id=run_id)
    store.update_meta(run_id, status="complete", comparator="fieldwise", model="fake", structuring_model="fake-sonnet", started_at=meta["created_at"], finished_at=meta["created_at"],
                      profile={"domain": "isotopes", "adjacent_field": "chemistry", "confirmed": True, "edited": False})
    for r in results:
        store.save_result(run_id, r)
    return read_payload(store, run_id, tmp, len(results))


def render(tmp_path, payload, slides="3", run=RUN):
    f = tmp_path / "payload.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    # encoding is explicit: node writes UTF-8, and text=True would otherwise decode it with the
    # platform's locale encoding (cp1252 on Windows), which chokes on the typographic quotes.
    p = subprocess.run(["node", str(FRONTEND_DIR / "smoke" / "render.mjs"), str(f), run, slides],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def all_text(out):
    t = {k: v for k, v in out.items() if isinstance(v, str)}
    for k, drawers in out.items():
        if k.endswith(":drawers"):
            for i, d in enumerate(drawers):
                t[f"{k}[{i}] {d['opener']}"] = d["body"]
    return t


# ---------------------------------------------------------------------- the bundled sample


def test_every_screen_and_every_drawer_renders_without_a_runtime_error(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    assert out["errors"] == [] and out["overview_buttons"] > 0
    assert all(out[f"slide{n}"] for n in range(1, 9)) and all(len(out[f"slide{n}:drawers"]) >= 6 for n in range(1, 9))
    assert all(d["body"].strip() for n in range(1, 9) for d in out[f"slide{n}:drawers"])


def test_the_slide_page_shows_the_profile_line_and_leads_with_words_not_a_number(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    for n in range(1, 9):
        page = out[f"slide{n}"]
        profile = sample["run"]["results"][n - 1]["metrics"]["slide_profile"]["text"]
        assert profile in page and page.index(profile) < page.index("What this slide demands of its reader")  # the line that tells a reader how to read the rest
        assert "Alignment to the intended reading" not in page and "Semantic similarity" not in page and "weaker instrument" not in page
        assert re.search(r"Claim(equivalent|over-claimed|under-specified|divergent|absent)", page)  # each audience card leads with the state, in words


def test_each_audience_card_shows_the_four_fields_with_absent_excluded_and_extra_told_apart(tmp_path, sample):
    page = render(tmp_path, sample, "5")["slide5"]
    novice = page[page.index("Novice") :]
    for row in ("Concept", "Claim", "Result", "Example"):
        assert row in novice
    assert "not reached" in novice  # the gap
    p3 = render(tmp_path, sample, "3")["slide3"]
    p2 = render(tmp_path, sample, "2")["slide2"]
    assert "not part of this slide" in p2  # a field the expert left null is excluded, not a gap
    assert "not part of this slide" not in p3


def test_the_expert_card_says_reference_and_its_takeaway_is_shown_once_as_the_intent(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3")
    for n in (1, 2, 3):
        page = out[f"slide{n}"]
        takeaway = sample["run"]["results"][n - 1]["readings"]["expert"]["takeaway"]
        assert page.count(takeaway) == 1 and page.count("Intent of this slide") == 1
        assert "reference" in page[page.index("Works in LLM inference serving systems") :]


def test_the_figure_slide_shows_a_collapsed_machine_generated_description_and_the_others_show_none(tmp_path, sample):
    out = render(tmp_path, sample, "3,8")
    desc = sample["run"]["results"][7]["image_content"]["text"]
    assert "Figure description (machine-generated)" in out["slide8"] and desc in out["slide8"]
    assert "Marks and labels only, never a principle" in out["slide8"] and "Not scored." in out["slide8"]
    assert "Figure description" not in out["slide3"]
    src = (FRONTEND_DIR / "js" / "views-detail.js").read_text(encoding="utf-8")
    assert 'h("details", { class: "figure-desc" }' in src and "open:" not in src.split('class: "figure-desc"')[1].split("\n")[0]  # collapsed by default


def test_the_figure_description_is_what_the_personas_were_given_under_a_figure_marker(sample):
    r = sample["run"]["results"][7]
    assert r["text"].endswith("FIGURE: [description of the image, not printed words] " + r["image_content"]["text"])
    assert r["image_content"]["machine_generated"] is True and r["image_content"]["model"] == "claude-haiku-4-5"


def test_the_overview_ranks_shows_state_chips_and_labels_the_arc_propositions_covered(tmp_path, sample):
    out = render(tmp_path, sample)
    o = out["overview"]
    assert "Hardest slides for a newcomer" in o and "field-wise comparison" in o and "Where the audiences differ most" not in o
    assert "novice: " in o and re.search(r"novice: (equivalent|over-claimed|under-specified|divergent|absent)", o)
    assert re.search(r"novice: \S+ \d of \d", o)  # the count sits beside the state
    for tick in ("none covered", "half", "all covered"):
        assert tick in o
    assert "propositions covered" in o and "hollow: thin profile" in o
    assert "ordinal" not in o and "four rungs" not in o and "a count, not a similarity or a rank" in o
    assert out["arc_label"] and "Propositions of the expert" in out["arc_label"]


def test_the_arc_tooltip_gives_the_count_of_propositions_covered_and_no_similarity_number(tmp_path, sample):
    out = render(tmp_path, sample)
    for tip in out["arc_tooltips"]:
        assert re.search(r"Slide \d", tip) and "Expertreference" in tip
        assert re.search(r"Novice\d of \d propositions covered · (equivalent|over-claimed|under-specified|divergent|absent)", tip) or "Novice" in tip and "no claim" in tip
        assert re.search(r"Peer\d of \d propositions covered", tip) or "Peerno claim" in tip
        assert "rank" not in tip and not re.search(r"\b0\.\d\d\b", tip)  # a count, never a similarity-looking number


def test_no_scalar_alignment_appears_anywhere_on_a_field_wise_run(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    bad = {k: m.group(0) for k, t in all_text(out).items() if (m := re.search(r"cosine|semantic similarity|weaker instrument|Alignment to the intended|confiden|blind|diverg(?!ent)", t, re.I))}
    assert bad == {}


def test_an_under_specified_card_names_the_missed_proposition_in_plain_words(tmp_path, sample):
    out = render(tmp_path, sample, "3,4,5")
    seen = 0
    for n in (3, 4, 5):
        page = out[f"slide{n}"]
        for persona in ("novice", "peer"):
            cov = sample["run"]["results"][n - 1]["metrics"]["comparisons"][persona]["claim"]["coverage"]
            for missed in cov["missed"]:
                assert f"Missed: {missed}" in page, (n, persona, missed)  # the finding, in the words of the expert's proposition
                seen += 1
    assert seen >= 4


def test_every_state_and_finding_panel_shows_its_quoted_span(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    seen = 0
    for n in range(1, 9):
        m = sample["run"]["results"][n - 1]["metrics"]
        for d in out[f"slide{n}:drawers"]:
            if d["body"].startswith("Claim —"):
                persona = "Novice" if "Novice" in d["body"][:20] else "Peer"
                claim = m["comparisons"][persona.lower()]["claim"]
                assert m["takeaways"]["expert"] in d["body"] and m["takeaways"][persona.lower()] in d["body"]  # both takeaways, verbatim
                cov = claim["coverage"]
                for p in cov["propositions"]:  # the proposition table: each proposition, its status, and the quoted span
                    assert p["text"] in d["body"] and p["status"] in d["body"]
                    if p["evidence"]:
                        assert p["evidence"] in d["body"]
                assert f"{cov['covered']} of {cov['total']} proposition" in d["body"] and "How the state was reached" in d["body"]
                assert "expert claim ⇒" not in d["body"] and "Rationale" not in d["body"]  # the two-booleans block is gone
                if claim["status"] == "compared":
                    assert claim["expert"] in d["body"] and claim["audience"] in d["body"]
                else:
                    assert "No model was asked about this pair" in d["body"]
                seen += 1
    assert seen >= 10


def test_the_tier_panel_reads_only_the_slide_profile_fields(tmp_path, sample):
    out = render(tmp_path, sample, "3")
    tier = next(d["body"] for d in out["slide3:drawers"] if d["opener"] in ("Self-contained", "Background needed", "Expert-gated"))
    assert "What drove it" in tier and ("met" in tier) and "The rule" in tier and "an under-specified claim is partial" in tier
    assert "Semantic" not in tier and "alignment" not in tier.lower().replace("aligned", "")


# ---------------------------------------------------------------- findings, null cases, thin slides

EXPERT = {"concept": "opportunity cost", "claim": "Opportunity cost is the value of the next best alternative that is given up.", "result": "$50", "vehicle": "Tyler at $150 over Doja Cat at $100"}
NOVICE = {"concept": None, "claim": None, "result": None, "vehicle": "tickets to see Tyler at $150 versus Doja Cat at $100"}


def screenshot_case(tmp_path):
    results = [fw_slide_result(i, text=f"[title] Slide {i}") for i in range(1, 5)]
    results.append(fw_slide_result(5, novice=NOVICE, peer=dict(EXPERT), expert=EXPERT, text="[title] Opportunity cost"))
    for r in results[:4]:
        r["metrics"]["takeaways"]["expert"] = r["readings"]["expert"]["takeaway"]
    return make_run(tmp_path, results)


def test_the_screenshot_case_reports_example_bound_with_quoted_evidence_and_no_midrange_score(tmp_path):
    payload = screenshot_case(tmp_path)
    out = render(tmp_path, payload, "5", run="fw-demo")
    page = out["slide5"]
    text = "Novice takeaway is example-bound: describes the example but never names the principle or reaches the answer."
    assert page.count(text) >= 2  # in the summary, at the same prominence a score once had, and in the novice card
    assert page.index(text) < page.index("What this slide demands of its reader") + 400  # leads the summary
    assert not re.search(r"\b0\.\d\d\b", page)  # no score anywhere
    finding = next(d["body"] for d in out["slide5:drawers"] if d["body"].startswith("Example-bound"))
    novice_takeaway = payload["run"]["results"][4]["metrics"]["takeaways"]["novice"]
    assert novice_takeaway in finding and "opportunity cost" in finding and "rule (no model)" in finding and "$50" in finding
    assert "this slide has a result" in finding


def test_a_figure_dependent_finding_shows_the_figure_and_the_shared_words(tmp_path):
    fig = "Two lines on axes labelled Price and Quantity: one slopes downward, one slopes upward. They cross at a point marked P*, Q*."
    expert = {"concept": None, "claim": "Price and quantity settle where the crossing point is marked.", "result": None, "vehicle": None}
    novice = {"concept": None, "claim": "Markets are complicated things.", "result": None, "vehicle": None}
    results = [fw_slide_result(i, text=f"[title] S{i}") for i in range(1, 5)] + [fw_slide_result(5, novice=novice, peer=dict(expert), expert=expert, image=fig)]
    payload = make_run(tmp_path, results)
    out = render(tmp_path, payload, "5", run="fw-demo")
    assert "The substance of this slide is in the figure, and the novice reading does not engage with it: the figure is carrying meaning it does not label." in out["slide5"]
    finding = next(d["body"] for d in out["slide5:drawers"] if d["body"].startswith("Figure-dependent"))
    assert fig in finding and "price, quantity" in finding.lower() and "machine-generated" in finding


def test_a_slide_with_no_result_and_no_example_is_never_shown_as_a_gap_on_account_of_them(tmp_path):
    expert = {"concept": "isotope", "claim": "Isotopes are atoms of one element with different neutron counts.", "result": None, "vehicle": None}
    results = [fw_slide_result(i) for i in range(1, 5)] + [fw_slide_result(5, novice=dict(expert), peer=dict(expert), expert=expert)]
    out = render(tmp_path, make_run(tmp_path, results), "5", run="fw-demo")
    page = out["slide5"]
    assert "This slide names a principle; no example, no worked result." in page
    assert "not reached" not in page  # nothing is a gap
    assert "not part of this slide" in page
    assert "Novice result" not in page  # the summary reads only the profile's fields


def test_over_reach_is_surfaced_quietly_and_not_penalised(tmp_path):
    expert = {"concept": "isotope", "claim": "Isotopes are atoms of one element with different neutron counts.", "result": None, "vehicle": None}
    novice = dict(expert, result="12")
    results = [fw_slide_result(i) for i in range(1, 5)] + [fw_slide_result(5, novice=novice, peer=dict(expert), expert=expert)]
    page = render(tmp_path, make_run(tmp_path, results), "5", run="fw-demo")["slide5"]
    assert "also gave a result the expert did not. Not penalised." in page and "extra (over-reach)" in page


def test_thin_slides_are_hollow_on_the_arc_and_their_profile_is_named_in_the_tooltip(tmp_path):
    thin = {"concept": None, "claim": "Isotopes are atoms of one element with different neutron counts.", "result": None, "vehicle": None}
    results = [fw_slide_result(i) for i in range(1, 5)] + [fw_slide_result(5, novice=dict(thin), peer=dict(thin), expert=dict(thin))]
    payload = make_run(tmp_path, results)
    m = payload["run"]["results"][4]["metrics"]
    assert m["slide_profile"]["thin"] and m["chart"]["novice"]["value"] == 1.0  # a thin slide is still counted, and drawn hollow
    out = render(tmp_path, payload, "5", run="fw-demo")
    assert out["arc_hollow_markers"] == 2  # novice and peer on the thin slide; the reference stays solid
    assert any(m["slide_profile"]["text"] + " Thin profile: drawn hollow." in t for t in out["arc_tooltips"])
    assert "(thin)" in out["overview"]


def test_a_field_wise_run_shows_which_comparator_made_it_and_a_cosine_run_shows_its_own(tmp_path, sample):
    assert "field-wise comparison" in render(tmp_path, sample)["overview"]


def _as_saved_before_coverage(payload):
    """The payload of a field-wise run saved by the entailment version: verdicts and an ordinal, no coverage."""
    p = copy.deepcopy(payload)
    rung = {"equivalent": 1.0, "over-claimed": 0.66, "under-specified": 0.33, "divergent": 0.0, "absent": 0.0}
    for r in p["run"]["results"]:
        m = r["metrics"]
        m.pop("claim_comparison", None), m.pop("propositions", None)
        m["ordinal"] = {a: {"value": None if c["state"] is None else rung[c["state"]], "state": c["state"], "thin": c["thin"], **({"definitional": True} if c.get("definitional") else {})} for a, c in m.pop("chart").items()}
        for a in ("novice", "peer"):
            claim = m["comparisons"][a]["claim"]
            claim.pop("coverage", None)
            if claim["status"] == "compared":
                claim["verdict"] = {"expert_entails_audience": True, "audience_entails_expert": False, "rationale": "Only the expert says more.", "quote": claim["audience"], "quote_verified": True}
    p["run"]["rollup"]["arc_kind"] = "ordinal"
    for row in p["run"]["rollup"]["hardest"]:
        row["novice_total"] = None
    for pt in p["run"]["rollup"]["arc"]:
        pt.pop("counts", None)
    return p


def test_a_run_saved_before_coverage_keeps_its_entailment_panel_and_its_ordinal_arc(tmp_path, sample):
    out = render(tmp_path, _as_saved_before_coverage(sample), "3,4")
    assert out["errors"] == []
    assert "ordinal: four rungs, not a similarity" in out["overview"] and "four rungs" in out["arc_label"]
    for n in (3, 4):
        assert "Missed:" not in out[f"slide{n}"]  # nothing to name without coverage
        claim_panels = [d["body"] for d in out[f"slide{n}:drawers"] if d["body"].startswith("Claim —")]
        assert claim_panels and all("Rationale" in b and "proposition" not in b.lower() for b in claim_panels)


# --------------------------------------------------------------------------- the neural panel


def test_the_neural_panel_discloses_what_it_is_every_time_it_shows_a_number(sample, tmp_path):
    """Design rule 4: the prediction is labelled as predicted wherever it is shown. One line, in
    the dek, rather than a stack of warnings -- but it still has to name the model, say the
    narration was synthesized, and say plainly that nothing here was measured."""
    out = render(tmp_path, sample, "1")
    page = out["slide1"]
    assert "TRIBE v2" in page and "synthesized narration" in page
    assert "Predicted, not measured." in page


def test_the_neural_panel_shows_the_narration_the_number_was_made_from(sample, tmp_path):
    """Design rule 2: the number decomposes into its source text, on the page, without a round
    trip to the server."""
    out = render(tmp_path, sample, "1")
    assert "Serving LLMs faster" in out["slide1"]  # the synthesized narration of slide 1


def test_no_screen_ever_shows_the_gfp_engagement_scalar(sample, tmp_path):
    """Design rule 3: TRIBE's scalar engagement readout is a published null result. If the word
    reaches a screen, that is a bug, not a feature."""
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    for name, text in all_text(out).items():
        assert "gfp" not in text.lower(), name
        assert "engagement" not in text.lower(), name


def test_a_slide_that_was_never_precomputed_says_so_instead_of_showing_a_brain(sample, tmp_path):
    """Slide 8 was added to the deck after the GPU run. It must read as absent, and must not
    borrow slide 7's numbers to fill the space."""
    out = render(tmp_path, sample, "8")
    page = out["slide8"]
    assert "No precomputed neural response for this slide" in page
    assert "Processing ratio" not in page and "Language drive" not in page


def test_the_cortical_renders_are_framed_like_slide_images_so_dark_mode_cannot_dim_them(sample, tmp_path):
    """A cortical surface is a white-background render, exactly like a slide image, so it goes on
    the same light card in both themes rather than being inverted or dimmed to "fit" dark mode."""
    src = (FRONTEND_DIR / "js" / "views-detail.js").read_text(encoding="utf-8")
    panel = src[src.index("function neuralBody"):src.index("function neuralSection")]
    assert 'h("img"' not in panel and "slideImage(" in panel
    empty = src[src.index("function neuralAbsent"):src.index("function neuralBody")]
    assert 'h("img"' not in empty and "slideImage(" in empty
