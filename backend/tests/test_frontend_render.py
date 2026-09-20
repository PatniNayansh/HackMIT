"""Renders the real frontend views in Node (frontend/smoke/render.mjs, a tiny DOM stand-in)
against the bundled sample run's real payload, and checks what each screen says. It cannot judge
layout or colour; it does catch runtime errors and pins what text is, and is not, on screen."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from sightline.audiences import FileCache
from sightline.server import FRONTEND_DIR, create_app
from sightline.store import BUNDLED_RUNS_DIR, RunStore

from builders import HashEmbedder

RUN = "sample-llm-serving"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to render the frontend")


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render")
    app = create_app(
        store=RunStore(tmp / "history", bundled=[BUNDLED_RUNS_DIR]), cache=FileCache(tmp / "cache"),
        client_factory=lambda: None, intent_client_factory=lambda: None, embedder=HashEmbedder(),
    )
    with TestClient(app) as c:
        run = c.get(f"/api/runs/{RUN}").json()
        recs = {n: c.get(f"/api/runs/{RUN}/slides/{n}/recommendations").json() for n in range(1, 8)}
    return {"run": run, "recs": recs}


def render(tmp_path, payload, slides="3"):
    f = tmp_path / "payload.json"
    f.write_text(json.dumps(payload))
    p = subprocess.run(["node", str(FRONTEND_DIR / "smoke" / "render.mjs"), str(f), RUN, slides],
                       capture_output=True, text=True, timeout=90)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def everything(out):
    texts = {k: v for k, v in out.items() if isinstance(v, str)}
    for k, drawers in out.items():
        if k.endswith(":drawers"):
            for i, d in enumerate(drawers):
                texts[f"{k}[{i}]"] = d["body"]
    return texts


def test_the_sample_replays_its_recommendations_offline_for_every_slide(payload):
    assert all(r["available"] and r["cached"] for r in payload["recs"].values())  # no key, no calls


def test_the_screens_render_without_a_runtime_error(tmp_path, payload):
    out = render(tmp_path, payload, "1,3,6")
    assert out["errors"] == []
    assert out["overview_buttons"] > 0 and all(out[f"slide{n}"] for n in (1, 3, 6))


def test_the_overview_ranks_by_novice_difficulty_and_has_no_differ_most_section(tmp_path, payload):
    o = render(tmp_path, payload)["overview"]
    assert "Hardest slides for a newcomer" in o and "Narrative arc" in o
    assert "Where the audiences differ most" not in o and "Slides ranked by novice" not in o
    for label in ("Self-contained", "Background needed", "Expert-gated"):
        assert label in o
    # each row: takeaway, tier chip, novice unresolved-term count
    assert o.count("Novice takeaway") >= 5 and o.count("Novice unresolved terms") >= 5


def test_confidence_and_blind_spot_appear_nowhere_in_the_ui(tmp_path, payload):
    out = render(tmp_path, payload, "1,2,3,4,5,6,7")
    hits = {k: m.group(0) for k, t in everything(out).items() if (m := re.search(r"confiden|blind|diverg", t, re.I))}
    assert hits == {}


def test_every_slide_page_leads_with_a_tier_and_has_exactly_one_attributed_intent_block(tmp_path, payload):
    out = render(tmp_path, payload, "1,2,3,4,5,6,7")
    declared = payload["run"]["meta"]["intent"]
    assert declared  # the sample has a declared intent, so its absence below means something
    for n in range(1, 8):
        page = out[f"slide{n}"]
        intent = payload["run"]["results"][n - 1]["slide_intent"]["text"]
        assert "What this slide demands of its reader" in page
        assert page.index("What this slide demands of its reader") < page.index("Takeaway, verbatim")  # summary above the cards
        # one intent block, under the slide image (before the summary and the text well), attributed
        assert page.count("Intent of this slide") == 1 and page.count(intent) == 1
        assert page.index("Intent of this slide") < page.index("What this slide demands of its reader")
        assert (intent + "Inferred from the expert reading") in page  # the attribution follows the sentence
        assert page.count("Inferred from the expert reading") == 1
        # the retired band is gone, and the presenter's declared intent is not on the slide page
        assert "What this slide is trying to establish" not in page
        assert declared not in page and "declared intent" not in page.lower()


def test_the_declared_intent_is_still_stored_and_still_shown_on_the_overview(tmp_path, payload):
    out = render(tmp_path, payload)
    assert payload["run"]["meta"]["intent"] in out["overview"]


def test_the_expert_card_says_reference_instead_of_a_score(tmp_path, payload):
    page = render(tmp_path, payload)["slide3"]
    expert = page[page.index("Expert") :]
    assert "Alignment to the intended reading" in expert and "reference" in expert
    assert "The intended reading is derived from this expert interpretation, so it defines the baseline rather than scoring against it." in expert
    assert not re.search(r"reference.{0,40}1\.00|1\.00", expert)  # no 1.00 dressed up as a score


def test_recommendations_quote_evidence_and_are_grouped_by_audience(tmp_path, payload):
    page = render(tmp_path, payload)["slide3"]
    recs = payload["recs"][3]["recommendations"]
    for audience in ("novice", "peer"):
        for r in recs[audience]:
            assert r["bullet"] in page and r["evidence"] in page
    assert page.count("What to change") == 2  # novice and peer, never the expert
    assert recs["expert_flagged"] and "Expert also flagged" in page
    assert "sample data" in page  # the run is a bundled sample; the recommendations are not
    assert "checked-in fixture" not in page


def test_every_number_opens_a_drawer_showing_real_text(tmp_path, payload):
    out = render(tmp_path, payload, "3")
    drawers = out["slide3:drawers"]
    assert len(drawers) >= 10
    r = payload["run"]["results"][2]
    for d in drawers:
        assert d["body"].strip(), d["opener"]
    by_opener = {d["opener"]: d["body"] for d in drawers}
    assert r["metrics"]["takeaways"]["novice"] in by_opener[f"{r['metrics']['intent_alignment']['novice']['value']:.2f}"]
    assert r["slide_intent"]["text"] in by_opener[f"{r['metrics']['intent_alignment']['peer']['value']:.2f}"]
    assert "definitional, not measured" in by_opener["reference"]
    tier = next(v for k, v in by_opener.items() if k in ("Self-contained", "Background needed", "Expert-gated"))
    assert "not met" in tier or "met" in tier
    assert r["metrics"]["takeaways"]["expert"] in tier  # the three takeaways drive it, and are shown


def test_a_saved_run_from_before_this_revision_opens_in_a_reduced_form(tmp_path, payload):
    legacy = copy.deepcopy(payload)
    legacy["run"]["meta"]["legacy"] = True
    out = render(tmp_path, legacy)
    o = out["overview"]
    assert out["errors"] == []
    assert "Saved in the earlier format" in o and "Hardest slides" not in o and "Narrative arc" not in o
    assert "Terms the novice could not resolve" in o  # what is still valid keeps showing
