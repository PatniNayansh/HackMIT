from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf
import pytest
from fastapi.testclient import TestClient

from sightline.audiences import FileCache
from sightline.server import create_app
from sightline.store import RunStore

from builders import HashEmbedder
from conftest import FakeLLM, sentinel_payload


def pdf_bytes(tmp_path, pages=3) -> bytes:
    doc = pymupdf.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=960, height=540)
        page.insert_text((60, 90), f"Slide {i} title", fontsize=36)
        page.insert_text((60, 200), "PagedAttention cuts KV-cache waste", fontsize=18)
    p = tmp_path / "talk.pdf"
    doc.save(p)
    return p.read_bytes()


class Fake:
    """Client factory whose LLM answers the subfield question and the persona questions."""

    def __init__(self, responder=None):
        self.llm = FakeLLM(responder)
        self.available = True
        self.recs_calls = 0
        self.recs_delay = 0.0
        self.recs_fail = False

    async def _complete(self, *, system, user_text, image_png, schema, max_tokens=4096):
        if "Name the field it belongs to" in system:
            return {"domain": "LLM inference serving", "adjacent_field": "distributed systems"}
        if "You help a presenter fix one slide" in system:
            self.recs_calls += 1
            if self.recs_fail:
                raise ConnectionError("down")
            if self.recs_delay:
                await asyncio.sleep(self.recs_delay)
            term = re.search(r"<novice_report>.*?Terms it could not resolve: (.*?)\n", user_text, re.S).group(1)
            return {
                "novice": [{"bullet": f"Define {term} on first use", "evidence": term}],
                "peer": [{"bullet": "Invented change with no evidence", "evidence": "nobody said this"}],
                "expert_flagged": [],
            }
        return await FakeLLM.complete_json(
            self.llm, system=system, user_text=user_text, image_png=image_png, schema=schema
        )

    def __call__(self):
        if not self.available:
            return None
        self.llm.complete_json = self._complete  # type: ignore[method-assign]
        return self.llm


@pytest.fixture
def env(tmp_path):
    store = RunStore(tmp_path / "history")
    factory = Fake()

    def make_app(client_factory=factory, comparator="cosine"):
        return create_app(
            comparator=comparator,
            store=store,
            cache=FileCache(tmp_path / "cache"),
            client_factory=client_factory,
            embedder=HashEmbedder(),
            frontend_dir=tmp_path / "no-frontend",
        )

    return store, factory, make_app, tmp_path


def wait_done(client, run_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["meta"]["status"] in ("complete", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {body['meta']}")


def upload(client, tmp_path, pages=3):
    r = client.post("/api/upload", files={"file": ("My talk.pdf", pdf_bytes(tmp_path, pages), "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


START = {"intent": "Show the speedup.", "domain": "LLM serving", "adjacent_field": "databases"}


def test_upload_prefills_an_unconfirmed_subfield_and_holds_the_deck_as_a_draft(env):
    store, _, make_app, tmp = env
    with TestClient(make_app()) as c:
        d = upload(c, tmp)
        assert d["title"] == "My talk" and d["slide_count"] == 3 and d["status"] == "draft"
        assert d["profile_inferred"] == {"domain": "LLM inference serving", "adjacent_field": "distributed systems"}
        assert d["profile"] is None  # nothing is confirmed by inference alone
        assert c.get("/api/runs").json() == []  # a draft is not history
        assert c.get(f"/api/runs/{d['run_id']}/slides/2.png").headers["content-type"] == "image/png"


def test_upload_without_a_key_still_works_and_says_the_subfield_is_up_to_you(env):
    _, factory, make_app, tmp = env
    factory.available = False
    with TestClient(make_app()) as c:
        d = upload(c, tmp)
        assert d["profile_inferred"] is None and "Enter it yourself" in d["profile_inference_error"]
        assert d["can_call_model"] is False


@pytest.mark.parametrize("name,payload,code", [("deck.pptx", b"x", 415), ("bad.pdf", b"not a pdf", 422)])
def test_bad_uploads_are_rejected_with_a_reason(env, name, payload, code):
    _, _, make_app, _ = env
    with TestClient(make_app()) as c:
        r = c.post("/api/upload", files={"file": (name, payload, "application/octet-stream")})
        assert r.status_code == code and r.json()["detail"]


@pytest.mark.parametrize("body", [{**START, "domain": ""}, {**START, "adjacent_field": " "}])
def test_start_requires_both_subfields(env, body):
    _, _, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp)["run_id"]
        r = c.post(f"/api/runs/{run_id}/start", json=body)
        assert r.status_code == 422
        assert c.get(f"/api/runs/{run_id}").json()["meta"]["status"] == "draft"


@pytest.mark.parametrize("intent", [None, "   ", "Land the 2.4x."])
def test_the_declared_intent_is_optional(env, intent):
    _, _, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=1)["run_id"]
        body = {k: v for k, v in {**START, "intent": intent}.items() if not (k == "intent" and intent is None)}
        assert c.post(f"/api/runs/{run_id}/start", json=body).status_code == 202
        done = wait_done(c, run_id)
    assert done["meta"]["status"] == "complete"
    assert done["meta"]["intent"] == (intent.strip() or None if intent else None)


def test_a_run_streams_slide_by_slide_and_ends_complete(env):
    _, factory, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=4)["run_id"]
        assert c.post(f"/api/runs/{run_id}/start", json=START).status_code == 202
        body = wait_done(c, run_id)

        assert body["meta"]["status"] == "complete" and body["pending"] == []
        assert [r["index"] for r in body["results"]] == [1, 2, 3, 4]
        assert body["meta"]["profile"] == {
            "domain": "LLM serving", "adjacent_field": "databases", "confirmed": True, "edited": True,
        }
        assert body["rollup"]["n_scored"] == 4 and body["meta"]["legacy"] is False
        assert all(r["slide_intent"]["source"] == "expert_takeaway" for r in body["results"])
        assert all(r["slide_intent"]["text"] == r["readings"]["expert"]["takeaway"] == r["metrics"]["intent"] for r in body["results"])
        assert len(factory.llm.calls) == 4 * 3  # three persona calls a slide, nothing else in the batch
        # `since` returns only what has landed after slide 2
        later = c.get(f"/api/runs/{run_id}?since=2").json()
        assert [r["index"] for r in later["results"]] == [3, 4]
        assert later["results"][0]["image_url"].endswith("/slides/3.png")


def test_partial_state_reports_which_slides_are_still_pending(env):
    store, _, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=3)["run_id"]
        from builders import slide_result

        store.save_result(run_id, slide_result(1))
        store.update_meta(run_id, status="running")
        body = c.get(f"/api/runs/{run_id}").json()
        # no live task owns this "running" run, so it is reported as interrupted, not running
        assert body["meta"]["status"] == "interrupted"
        assert body["pending"] == [2, 3] and len(body["results"]) == 1


def test_a_saved_run_reopens_with_no_client_no_key_and_no_model_calls(env):
    store, factory, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        before = wait_done(c, run_id)
    calls_after_run = len(factory.llm.calls)

    factory.available = False  # the key is gone, the network is gone
    with TestClient(make_app(client_factory=factory)) as c2:
        rows = c2.get("/api/runs").json()
        assert [(r["run_id"], r["title"], r["slide_count"], r["model"], r["status"]) for r in rows] == [
            (run_id, "My talk", 3, "fake-model", "complete")
        ]
        after = c2.get(f"/api/runs/{run_id}").json()
        assert c2.get(f"/api/runs/{run_id}/slides/1.png").status_code == 200

    assert after == before  # the full review payload, byte for byte
    assert len(factory.llm.calls) == calls_after_run


def test_an_invalid_persona_reply_reaches_the_ui_as_an_error_not_a_number(env):
    store, factory, make_app, tmp = env
    factory.llm.responder = lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 1.7} if p == "expert" else {})}
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=2)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        body = wait_done(c, run_id)
    bad = body["results"][0]["readings"]["expert"]
    assert bad["ok"] is False and "1.7" in bad["error"]["message"] and "confidence" not in bad
    assert body["results"][0]["metrics"] is None and body["rollup"]["unscored"] == [1, 2]
    assert body["results"][0]["slide_intent"] is None  # no expert reading, no intended reading


def test_a_failed_run_can_be_started_again_from_scratch(env):
    store, factory, make_app, tmp = env
    factory.available = False
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        failed = wait_done(c, run_id)
        assert failed["meta"]["status"] == "failed" and "offline" in failed["meta"]["error"]

        factory.available = True
        assert c.post(f"/api/runs/{run_id}/start", json=START).status_code == 202
        assert wait_done(c, run_id)["meta"]["status"] == "complete"


def test_starting_twice_is_refused(env):
    _, factory, make_app, tmp = env
    factory.llm.delay = 0.3
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp)["run_id"]
        assert c.post(f"/api/runs/{run_id}/start", json=START).status_code == 202
        assert c.post(f"/api/runs/{run_id}/start", json=START).status_code == 409
        wait_done(c, run_id, timeout=15)


def test_unknown_and_hostile_run_ids_are_404(env):
    _, _, make_app, _ = env
    with TestClient(make_app()) as c:
        for path in ("/api/runs/nope", "/api/runs/..%2f..%2fetc", "/api/runs/nope/slides/1.png"):
            assert c.get(path).status_code == 404


def test_bundled_samples_list_open_and_refuse_to_start(env, tmp_path):
    import json

    bundled = tmp_path / "bundled" / "sample-1"
    bundled.mkdir(parents=True)
    (bundled / "run.json").write_text(json.dumps(
        {"run_id": "sample-1", "title": "Sample", "status": "complete", "sample": True, "slide_count": 0,
         "created_at": "2026-01-01T00:00:00+00:00", "model": "fixture"}))
    store = RunStore(tmp_path / "history", bundled=[tmp_path / "bundled"])
    app = create_app(store=store, cache=FileCache(tmp_path / "c"), client_factory=lambda: None,
                     embedder=HashEmbedder(), frontend_dir=tmp_path / "none")
    with TestClient(app) as c:
        assert c.get("/api/runs").json()[0]["sample"] is True
        assert c.get("/api/runs/sample-1").status_code == 200
        assert c.post("/api/runs/sample-1/start", json=START).status_code == 403


def test_health_reports_whether_new_runs_are_possible(env):
    _, factory, make_app, _ = env
    with TestClient(make_app()) as c:
        assert c.get("/api/health").json()["can_call_model"] is True
        factory.available = False
        assert c.get("/api/health").json()["can_call_model"] is False


# ----------------------------------------------------------------------- recommendations


def test_recommendations_are_lazy_generated_on_first_open_and_then_served_from_disk(env):
    _, factory, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=2)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        wait_done(c, run_id)
        assert factory.recs_calls == 0  # the batch run never asked for recommendations

        first = c.get(f"/api/runs/{run_id}/slides/1/recommendations").json()
        again = c.get(f"/api/runs/{run_id}/slides/1/recommendations").json()

    assert factory.recs_calls == 1  # one call for the slide that was opened, none for slide 2
    assert first["available"] and first["cached"] is False and again["cached"] is True
    recs = first["recommendations"]
    assert recs["novice"] == [{"audience": "novice", "bullet": "Define SENTINEL-NOVICE-1 term on first use", "evidence": "SENTINEL-NOVICE-1 term"}]
    assert recs["peer"] == [] and recs["meta"]["dropped_without_evidence"] == 1  # the invented bullet never reaches the UI
    assert again["recommendations"] == recs


def test_saved_recommendations_replay_offline_with_no_key(env):
    _, factory, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=2)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        wait_done(c, run_id)
        saved = c.get(f"/api/runs/{run_id}/slides/1/recommendations").json()["recommendations"]
    calls = factory.recs_calls

    factory.available = False
    with TestClient(make_app(client_factory=factory)) as c2:
        replay = c2.get(f"/api/runs/{run_id}/slides/1/recommendations").json()
        other = c2.get(f"/api/runs/{run_id}/slides/2/recommendations").json()  # never opened before saving

    assert replay == {"available": True, "cached": True, "recommendations": saved}
    assert factory.recs_calls == calls
    assert other["available"] is False and "No API key" in other["reason"]


def test_opening_the_same_slide_twice_at_once_makes_one_call(env):
    _, factory, make_app, tmp = env
    factory.recs_delay = 0.4
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=1)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        wait_done(c, run_id)
        with ThreadPoolExecutor(2) as pool:
            a, b = pool.map(lambda _: c.get(f"/api/runs/{run_id}/slides/1/recommendations").json(), range(2))
    assert factory.recs_calls == 1 and a["available"] and b["available"]


def test_a_slide_with_an_invalid_persona_reply_has_no_recommendations(env):
    _, factory, make_app, tmp = env
    factory.llm.responder = lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 9} if p == "peer" else {})}
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=1)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        wait_done(c, run_id)
        out = c.get(f"/api/runs/{run_id}/slides/1/recommendations").json()
    assert out["available"] is False and factory.recs_calls == 0


def test_a_failed_recommendation_call_is_a_502_and_is_not_saved(env):
    store, factory, make_app, tmp = env
    with TestClient(make_app()) as c:
        run_id = upload(c, tmp, pages=1)["run_id"]
        c.post(f"/api/runs/{run_id}/start", json=START)
        wait_done(c, run_id)
        factory.recs_fail = True
        r = c.get(f"/api/runs/{run_id}/slides/1/recommendations")
    assert r.status_code == 502 and store.load_recs(run_id, 1) is None
