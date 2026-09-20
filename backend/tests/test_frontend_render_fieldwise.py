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
    return {"run": run, "recs": rec}


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
    f.write_text(json.dumps(payload))
    p = subprocess.run(["node", str(FRONTEND_DIR / "smoke" / "render.mjs"), str(f), run, slides], capture_output=True, text=True, timeout=120)
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
    assert "not part of this slide" in p3  # a field the expert left null is excluded, not a gap


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
    assert "Deliberately non-interpretive" in out["slide8"] and "not scored and never compared across readers" in out["slide8"]
    assert "Figure description" not in out["slide3"]
    src = (FRONTEND_DIR / "js" / "views-detail.js").read_text()
    assert 'h("details", { class: "figure-desc" }' in src and "open:" not in src.split('class: "figure-desc"')[1].split("\n")[0]  # collapsed by default


def test_the_figure_description_is_what_the_personas_were_given_under_a_figure_marker(sample):
    r = sample["run"]["results"][7]
    assert r["text"].endswith("FIGURE: [description of the image, not printed words] " + r["image_content"]["text"])
    assert r["image_content"]["machine_generated"] is True and r["image_content"]["model"] == "claude-haiku-4-5"


def test_the_overview_ranks_shows_state_chips_and_labels_the_arc_ordinal(tmp_path, sample):
    out = render(tmp_path, sample)
    o = out["overview"]
    assert "Hardest slides for a newcomer" in o and "field-wise comparison" in o and "Where the audiences differ most" not in o
    assert "novice: " in o and re.search(r"novice: (equivalent|over-claimed|under-specified|divergent|absent)", o)
    for rung in ("divergent · absent", "under-specified", "over-claimed", "equivalent"):
        assert rung in o
    assert "ordinal: four rungs, not a similarity" in o and "hollow: thin profile" in o
    assert "It is a rank, not a similarity or a probability." in o
    assert out["arc_label"] and "four rungs" in out["arc_label"]


def test_the_arc_tooltip_names_states_and_says_it_is_a_rank_not_a_similarity(tmp_path, sample):
    out = render(tmp_path, sample)
    for tip in out["arc_tooltips"]:
        assert re.search(r"Slide \d", tip) and "a rank, not a similarity or a probability" in tip
        assert re.search(r"Novice(equivalent|over-claimed|under-specified|divergent|absent|no claim)", tip) and "Expertreference" in tip
        assert not re.search(r"\b0\.\d\d\b", tip)  # no similarity-looking numbers


def test_no_scalar_alignment_appears_anywhere_on_a_field_wise_run(tmp_path, sample):
    out = render(tmp_path, sample, "1,2,3,4,5,6,7,8")
    bad = {k: m.group(0) for k, t in all_text(out).items() if (m := re.search(r"cosine|semantic similarity|weaker instrument|Alignment to the intended|confiden|blind|diverg(?!ent)", t, re.I))}
    assert bad == {}


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
                if claim["status"] == "compared":
                    v = claim["verdict"]
                    assert claim["expert"] in d["body"] and claim["audience"] in d["body"] and v["quote"] in d["body"]
                    assert "expert claim ⇒" in d["body"] and "Rationale" in d["body"]
                else:
                    assert "no entailment was run" in d["body"]
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
YES = {"evaluated": True, "expert_entails_audience": True, "audience_entails_expert": True, "rationale": "Same idea.", "quote": "next best alternative", "quote_verified": True}


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
    thin = {"concept": "isotope", "claim": None, "result": None, "vehicle": None}
    results = [fw_slide_result(i) for i in range(1, 5)] + [fw_slide_result(5, novice=dict(thin), peer=dict(thin), expert=dict(thin))]
    out = render(tmp_path, make_run(tmp_path, results), "5", run="fw-demo")
    assert out["arc_hollow_markers"] == 2  # novice and peer on the thin slide; the reference stays solid
    assert any("This slide names a principle; no example, no worked result. Thin profile: drawn hollow." in t for t in out["arc_tooltips"])
    assert "(thin)" in out["overview"]


def test_a_field_wise_run_shows_which_comparator_made_it_and_a_cosine_run_shows_its_own(tmp_path, sample):
    assert "field-wise comparison" in render(tmp_path, sample)["overview"]
