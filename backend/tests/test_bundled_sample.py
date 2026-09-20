"""The checked-in sample run and the frontend files, exercised exactly as shipped: through the
real bundled directory, with no client, no key and no network."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from sightline.audiences import PERSONAS, FileCache
from sightline.server import FRONTEND_DIR, create_app
from sightline.store import BUNDLED_RUNS_DIR, RunStore

from builders import HashEmbedder

RUN = "sample-llm-serving"


@pytest.fixture
def client(tmp_path):
    app = create_app(
        store=RunStore(tmp_path / "history", bundled=[BUNDLED_RUNS_DIR]),
        cache=FileCache(tmp_path / "cache"),
        client_factory=lambda: None,  # no key
        embedder=HashEmbedder(),
    )
    with TestClient(app) as c:
        yield c


def test_the_sample_is_listed_flagged_and_complete(client):
    (row,) = [r for r in client.get("/api/runs").json() if r["run_id"] == RUN]
    assert row["sample"] is True and row["status"] == "complete" and row["slide_count"] == 7
    assert row["model"] and row["title"]


def test_the_sample_reproduces_the_full_review_payload_offline(client):
    body = client.get(f"/api/runs/{RUN}").json()
    assert body["meta"]["status"] == "complete" and body["pending"] == []
    assert len(body["results"]) == 7 and body["rollup"]["n_scored"] == 7 and body["rollup"]["comparable"]
    assert body["rollup"]["terms"] and body["rollup"]["arc"] and body["rollup"]["hardest"]
    assert body["rollup"]["definitional"] == ["expert"] and body["meta"]["legacy"] is False
    for i in range(1, 8):
        assert client.get(f"/api/runs/{RUN}/slides/{i}.png").status_code == 200


def test_every_metric_in_the_sample_traces_back_to_the_text_that_produced_it(client):
    """A slide's intent IS the expert's stored takeaway, and every alignment was computed from
    exactly that string and the persona's own stored takeaway."""
    body = client.get(f"/api/runs/{RUN}").json()
    for r in body["results"]:
        m, si = r["metrics"], r["slide_intent"]
        takeaways = {p: r["readings"][p]["takeaway"] for p in PERSONAS}
        assert m["takeaways"] == takeaways
        assert si == {"text": takeaways["expert"], "source": "expert_takeaway"}
        assert m["intent"] == si["text"] == r["readings"]["expert"]["takeaway"]  # shown == measured, character for character
        for p in ("novice", "peer"):
            assert m["intent_alignment"][p]["inputs"] == {"intent": si["text"], p: takeaways[p]}
        expert = m["intent_alignment"]["expert"]  # the reference: 1.0 by definition, and labelled so
        assert expert["value"] == 1.0 and expert["definitional"] is True
        assert m["term_gap"]["novice_unresolved"] == r["readings"]["novice"]["unresolved_terms"]
        assert not ({"audience_divergence", "blind_spot_score", "pairwise_distance"} & set(m))
        # confidence is still stored exactly as the model reported it (debugging, history)
        assert all(0.0 <= r["readings"][p]["confidence"] <= 1.0 for p in PERSONAS)


def test_the_sample_cannot_be_run_again_or_modified(client):
    r = client.post(f"/api/runs/{RUN}/start", json={"intent": "x", "domain": "y", "adjacent_field": "z"})
    assert r.status_code == 403


# ------------------------------------------------------------------------------ frontend


def test_the_server_serves_the_frontend(client):
    index = client.get("/")
    assert index.status_code == 200 and "/static/js/main.js" in index.text
    assert client.get("/static/style.css").headers["cache-control"] == "no-cache"


def test_every_module_the_frontend_imports_exists():
    js = FRONTEND_DIR / "js"
    modules = sorted(js.glob("*.js"))
    assert modules
    for path in modules:
        for target in re.findall(r'from "\./([\w-]+\.js)"', path.read_text()):
            assert (js / target).is_file(), f"{path.name} imports missing {target}"


def test_the_frontend_never_builds_html_from_strings():
    """Slide text and model output are untrusted; the DOM helpers only ever create text nodes."""
    for path in (FRONTEND_DIR / "js").glob("*.js"):
        assert "innerHTML" not in path.read_text(), path.name
        assert "insertAdjacentHTML" not in path.read_text(), path.name
