"""The checked-in sample run and the frontend files, exercised exactly as shipped: through the
real bundled directory, with no client, no key and no network."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from profe import compare as C
from profe.audiences import PERSONAS, FileCache
from profe.server import FRONTEND_DIR, create_app
from profe.store import BUNDLED_RUNS_DIR, RunStore

from builders import HashEmbedder

RUN = "sample-llm-serving"
INTERPRETIVE = r"\b(shows?|demonstrat\w*|illustrat\w*|represent\w*|indicat\w*|implies|suggest\w*|equilibrium|surplus|therefore|because|would|better|worse)\b"


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
    assert row["sample"] is True and row["status"] == "complete" and row["slide_count"] == 8
    assert row["model"] and row["title"]


def test_the_sample_reproduces_the_full_review_payload_offline(client):
    body = client.get(f"/api/runs/{RUN}").json()
    assert body["meta"]["status"] == "complete" and body["pending"] == []
    assert body["meta"]["comparator"] == "fieldwise" and body["rollup"]["comparator"] == "fieldwise" and body["meta"]["legacy"] is False
    assert len(body["results"]) == 8 and body["rollup"]["n_scored"] == 8 and body["rollup"]["comparable"]
    assert body["rollup"]["terms"] and body["rollup"]["arc"] and body["rollup"]["hardest"] and body["rollup"]["definitional"] == ["expert"]
    for i in range(1, 9):
        assert client.get(f"/api/runs/{RUN}/slides/{i}.png").status_code == 200
        rec = client.get(f"/api/runs/{RUN}/slides/{i}/recommendations").json()
        assert rec["available"] and rec["cached"]  # saved with the run: no key needed


def test_a_slides_intent_is_the_expert_takeaway_and_the_fieldwise_payload_is_self_consistent(client):
    """Every stored comparison can be recomputed from the stored fields alone (the null table, the
    comparators, the states from the stored verdicts), so nothing on screen is unexplained."""
    body = client.get(f"/api/runs/{RUN}").json()
    for r in body["results"]:
        m = r["metrics"]
        assert r["slide_intent"] == {"text": r["readings"]["expert"]["takeaway"], "source": "expert_takeaway"} and m["intent"] == r["slide_intent"]["text"]
        assert m["takeaways"] == {p: r["readings"][p]["takeaway"] for p in PERSONAS}
        fields = m["fields"]
        assert m["slide_profile"] == C.slide_profile(fields["expert"])
        for aud in ("novice", "peer"):
            for f in C.FIELDS:
                stored = m["comparisons"][aud][f]
                fresh = C.compare_field(f, fields["expert"][f], fields[aud][f])
                assert stored["status"] == fresh["status"], (r["index"], aud, f)
                if f in ("concept", "result") and stored["status"] == "compared":
                    assert stored["outcome"] == fresh["outcome"]
            claim = m["comparisons"][aud]["claim"]
            if claim["status"] in ("compared", "gap"):
                cov = claim["coverage"]
                assert [p["text"] for p in cov["propositions"]] == [p["text"] for p in m["propositions"]]
                assert claim["outcome"] == C.coverage_state(claim["expert"], claim["audience"], cov)  # the state is derived from the stored judgements
                assert (claim["outcome"] == "divergent") == any(p["status"] == "contradicted" for p in cov["propositions"])
                for p in cov["propositions"]:
                    if p["status"] in ("covered", "contradicted"):  # provenance: a span found word for word in the reader's claim
                        assert p["evidence"] and C.quoted_word_for_word(p["evidence"], claim["audience"]), (r["index"], aud, p)
                    else:
                        assert not p["evidence"]
                assert m["chart"][aud]["value"] == C.chart_value(claim["outcome"], cov)
                assert cov["covered"] == sum(p["status"] == "covered" for p in cov["propositions"]) and cov["total"] == len(m["propositions"])
            assert m["comparisons"][aud]["vehicle"]["status"] == "not_scored"  # never scored
        assert m["chart"]["expert"]["definitional"] is True and "ordinal" not in m and "intent_alignment" not in m
        assert m["claim_comparison"] == "coverage" and body["rollup"]["arc_kind"] == "coverage"
        assert not ({"audience_divergence", "blind_spot_score", "pairwise_distance"} & set(m))


def test_every_finding_in_the_sample_is_backed_by_quoted_text(client):
    for r in client.get(f"/api/runs/{RUN}").json()["results"]:
        for f in r["metrics"]["findings"]:
            assert f["evidence"] and all(e["text"] for e in f["evidence"])
            if f["id"] == "example_bound":
                assert f["evidence"][0]["text"] == r["metrics"]["takeaways"][f["audience"]]


def test_only_the_figure_slide_was_described_and_the_description_is_neutral(client):
    results = client.get(f"/api/runs/{RUN}").json()["results"]
    described = [r["index"] for r in results if r["image_content"]]
    assert described == [8]  # a text-only slide costs nothing extra
    ic = results[7]["image_content"]
    assert ic["source"] == "vision" and ic["machine_generated"] is True and ic["model"] == "claude-haiku-4-5"
    assert not re.search(INTERPRETIVE, ic["text"], re.I)  # marks, labels, axes, values: no meaning drawn from them
    assert 'labeled "p99 latency (ms)"' in ic["text"] or "p99 latency (ms)" in ic["text"]
    assert results[7]["text"].endswith("FIGURE: [description of the image, not printed words] " + ic["text"])
    assert all("FIGURE:" not in r["text"] for r in results[:7])


# ----------------------------------------------------------------------------- neural
# The checked-in neural fixtures are real TRIBE v2 output from a GX10 run, not a stand-in.
# They cover slides 1-7; slide 8 was added to the deck afterwards and was never run, which
# is what the UI's empty state exists for.


def test_the_bundled_neural_fixtures_are_the_narration_of_the_slides_they_belong_to(client, tmp_path):
    """Design rule 2: every number decomposes into the text that produced it. If the deck is ever
    regenerated without re-running TRIBE, the transcripts drift off the slides and this catches
    it before the UI quietly shows one slide's brain next to another slide's words."""
    store = RunStore(tmp_path / "history", bundled=[BUNDLED_RUNS_DIR])
    text = {s.index: s.text for s in store.load_slides(RUN)}
    for n in range(1, 8):
        spoken = client.get(f"/api/runs/{RUN}/slides/{n}/neural").json()["narration_transcript"]
        for line in text[n].splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                line = line.split("]", 1)[1].strip()  # drop the [title]/[body] layout prefix
            assert line in spoken, f"slide {n}: narration is missing {line!r}"


def test_the_sample_serves_real_neural_output_for_the_slides_that_were_run(client):
    roll = client.get(f"/api/runs/{RUN}/neural").json()
    assert roll["n_slides"] == 8 and roll["scored"] == [1, 2, 3, 4, 5, 6, 7]
    # Z-scores are relative to this deck (design rule 5: ranks within a deck, never absolutes),
    # so they must centre on zero however the underlying ratios are scaled.
    zs = [row["processing_ratio_z"] for row in roll["per_slide"]]
    assert abs(sum(zs)) < 1e-9 and max(zs) > 0 > min(zs)


def test_the_slide_that_was_never_run_404s_instead_of_borrowing_another_slides_brain(client):
    assert client.get(f"/api/runs/{RUN}/slides/8/neural").status_code == 404
    assert client.get(f"/api/runs/{RUN}/slides/8/neural/lateral_left.png").status_code == 404


def test_the_gfp_null_result_is_stored_but_never_served(client):
    """Design rule 3: the scalar engagement readout is a published null result. It stays in the
    fixture so the Methods panel can reproduce it next to the citation, and never reaches a
    response on its own."""
    on_disk = json.loads((BUNDLED_RUNS_DIR.parent / "neural" / RUN / "1" / "metrics.json").read_text(encoding="utf-8"))
    assert "gfp_negative_baseline" in on_disk
    body = client.get(f"/api/runs/{RUN}/slides/1/neural").json()
    assert "gfp_negative_baseline" not in body


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
        for target in re.findall(r'from "\./([\w-]+\.js)"', path.read_text(encoding="utf-8")):
            assert (js / target).is_file(), f"{path.name} imports missing {target}"


def test_the_frontend_never_builds_html_from_strings():
    """Slide text and model output are untrusted; the DOM helpers only ever create text nodes."""
    for path in (FRONTEND_DIR / "js").glob("*.js"):
        assert "innerHTML" not in path.read_text(encoding="utf-8"), path.name
        assert "insertAdjacentHTML" not in path.read_text(encoding="utf-8"), path.name
