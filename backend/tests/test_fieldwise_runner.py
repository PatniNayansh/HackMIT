"""The field-wise path through the runner and the server, and the COMPARATOR flag that chooses it."""

from __future__ import annotations

import asyncio
import re

import pymupdf
import pytest
from fastapi.testclient import TestClient

from sightline import server as server_mod
from sightline.audiences import DeckProfile, FileCache
from sightline.compare import FileStructureCache
from sightline.ingest import Slide
from sightline.runner import TolerantEngine, run_deck
from sightline.server import COMPARATORS, create_app, default_comparator
from sightline.store import RunStore

from builders import HashEmbedder
from conftest import FakeLLM, sentinel_payload

PROFILE = DeckProfile("LLM serving", "distributed systems")


class Structurer:
    """Stands in for the structuring model. Every viewer's claim is its own takeaway, and every pair
    'entails' both ways, unless a scenario says otherwise."""

    model = "fake-sonnet"

    def __init__(self, override=None, delay=0.0):
        self.calls, self.override, self.delay = [], override, delay

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append(user_text)
        if self.delay:
            await asyncio.sleep(self.delay)
        tk = {p: re.search(rf"<{p}_takeaway>\n(.*?)\n</{p}_takeaway>", user_text, re.S).group(1) for p in ("novice", "peer", "expert")}
        fields = {p: {"concept": None, "claim": tk[p], "result": None, "vehicle": None} for p in tk}
        yes = {"evaluated": True, "expert_entails_audience": True, "audience_entails_expert": True, "rationale": "same", "quote": tk["expert"][:9]}
        out = {"fields": fields, "verdicts": {"novice": yes, "peer": yes}}
        if self.override:
            out = self.override(out, tk)
        return out


def start(store, n=3, image_content_on=()):
    slides = [
        Slide(i, f"[title] slide {i}", b"\x89PNG", {"text": "Two lines cross at a marked point.", "model": "fake-haiku", "source": "vision", "machine_generated": True} if i in image_content_on else None)
        for i in range(1, n + 1)
    ]
    meta = store.create_draft(title="t", source_filename="t.pdf", inferred=None, inference_error=None, slides=slides)
    store.update_meta(meta["run_id"], intent=None, status="running",
                      profile={"domain": PROFILE.domain, "adjacent_field": PROFILE.adjacent_field, "confirmed": True})
    return meta["run_id"]


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "history")


async def test_a_fieldwise_run_stores_fields_comparisons_and_findings_with_one_structuring_call_per_slide(store):
    run_id = start(store)
    llm, st = FakeLLM(), Structurer()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder(), comparator="fieldwise", structuring_client=st)

    meta = store.load_meta(run_id)
    assert meta["status"] == "complete" and meta["comparator"] == "fieldwise" and meta["structuring_model"] == "fake-sonnet"
    results = store.load_results(run_id)
    assert [r["index"] for r in results] == [1, 2, 3]  # saved in order, though structuring overlaps the next slide
    assert len(st.calls) == 3 and len(llm.calls) == 9  # 3 persona calls + ONE structuring call per slide
    m = results[1]["metrics"]
    assert m["comparator"] == "fieldwise" and m["intent"] == "SENTINEL-EXPERT-2 takeaway"
    assert m["fields"]["novice"]["claim"] == "SENTINEL-NOVICE-2 takeaway"
    assert m["comparisons"]["novice"]["claim"]["outcome"] == "equivalent" and m["ordinal"]["expert"]["definitional"] is True
    assert "intent_alignment" not in m  # no cosine anywhere on this path
    assert set(results[1]["timing"]) == {"personas_s", "structuring_s"}
    assert results[1]["slide_intent"] == {"text": "SENTINEL-EXPERT-2 takeaway", "source": "expert_takeaway"}


async def test_each_slides_structuring_overlaps_the_next_slides_persona_calls(store):
    """Per-slide time stays near max(personas, structuring), not their sum."""
    run_id = start(store, n=4)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await run_deck(store, run_id, TolerantEngine(FakeLLM(delay=0.15)), HashEmbedder(), comparator="fieldwise", structuring_client=Structurer(delay=0.15))
    elapsed = loop.time() - t0
    assert store.load_meta(run_id)["status"] == "complete" and len(store.load_results(run_id)) == 4
    assert elapsed < 4 * 0.15 * 2 - 0.15  # a strictly sequential run would take about 1.2 s; overlapped is about 0.75 s


async def test_a_failed_structuring_call_leaves_only_that_slide_unscored(store):
    run_id = start(store)
    calls = {"n": 0}

    def flaky(out, tk):
        calls["n"] += 1
        if calls["n"] == 2:  # the second slide's answer is missing a verdict for a pair with both claims, on both attempts
            out["verdicts"]["peer"] = {"evaluated": False, "expert_entails_audience": False, "audience_entails_expert": False, "rationale": "", "quote": ""}
        return out

    class Flaky(Structurer):
        async def complete_json(self, **kw):
            return await super().complete_json(**kw)

    st = Structurer(override=lambda out, tk: flaky(out, tk))
    # slide 2's two attempts (calls 2 and 3) must both fail: make call 3 fail as well
    orig = st.override

    def both(out, tk):
        out = orig(out, tk)
        if calls["n"] == 3:
            out["verdicts"]["peer"] = {"evaluated": False, "expert_entails_audience": False, "audience_entails_expert": False, "rationale": "", "quote": ""}
        return out

    st.override = both
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), comparator="fieldwise", structuring_client=st)
    r1, r2, r3 = store.load_results(run_id)
    assert store.load_meta(run_id)["status"] == "complete"
    assert r2["metrics"] is None and "structuring call failed" in r2["metrics_error"] and "verdict missing" in r2["metrics_error"]
    assert r2["readings"]["novice"]["ok"] and r1["metrics"] and r3["metrics"]  # the readings survive; the neighbours are fine


async def test_offline_a_cached_deck_reruns_field_wise_with_no_client_at_all(store, tmp_path):
    pc, sc = FileCache(tmp_path / "cache"), FileStructureCache(tmp_path / "structured")
    first = start(store)
    await run_deck(store, first, TolerantEngine(FakeLLM(), pc), HashEmbedder(), comparator="fieldwise", structuring_client=Structurer(), structure_cache=sc)
    second = start(store)
    await run_deck(store, second, TolerantEngine(None, pc, offline=True), HashEmbedder(), comparator="fieldwise", structuring_client=None, structure_cache=sc)
    a, b = store.load_results(first), store.load_results(second)
    assert store.load_meta(second)["status"] == "complete" and [r["metrics"]["fields"] for r in a] == [r["metrics"]["fields"] for r in b]


async def test_with_no_structuring_client_and_nothing_cached_the_slide_is_unscored_with_a_reason(store):
    run_id = start(store, n=1)
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), comparator="fieldwise", structuring_client=None)
    (r,) = store.load_results(run_id)
    assert r["metrics"] is None and "no model client" in r["metrics_error"] and all(x["ok"] for x in r["readings"].values())


async def test_the_figure_description_reaches_every_persona_under_a_figure_marker_and_is_never_compared(store):
    run_id = start(store, n=2, image_content_on=(2,))
    llm, st = FakeLLM(), Structurer()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder(), comparator="fieldwise", structuring_client=st)
    r1, r2 = store.load_results(run_id)
    for persona in ("novice", "peer", "expert"):  # the SAME description goes to all three
        assert "FIGURE: [description of the image, not printed words] Two lines cross at a marked point." in llm.calls_by(persona)[1].user_text
        assert "FIGURE:" not in llm.calls_by(persona)[0].user_text  # a slide with no figure gets none
    assert r2["text"].endswith("FIGURE: [description of the image, not printed words] Two lines cross at a marked point.")
    assert r2["image_content"]["text"] == "Two lines cross at a marked point." and r1["image_content"] is None
    assert "image_content" not in str(r2["metrics"]["comparisons"])  # slide-level context: never in a comparison


# --------------------------------------------------------------------- the COMPARATOR flag


def test_the_flag_defaults_to_fieldwise_and_reads_the_environment(monkeypatch):
    assert COMPARATORS == ("cosine", "fieldwise") and server_mod.COMPARATOR in COMPARATORS
    monkeypatch.delenv("SIGHTLINE_COMPARATOR", raising=False)
    assert default_comparator() == "fieldwise"
    monkeypatch.setenv("SIGHTLINE_COMPARATOR", "cosine")
    assert default_comparator() == "cosine"  # the one-line rollback, no code change
    monkeypatch.setenv("SIGHTLINE_COMPARATOR", "Fieldwise ")
    assert default_comparator() == "fieldwise"
    monkeypatch.setenv("SIGHTLINE_COMPARATOR", "nonsense")
    with pytest.raises(ValueError, match="SIGHTLINE_COMPARATOR"):
        default_comparator()
    with pytest.raises(ValueError):
        create_app(comparator="nonsense")


def pdf_bytes(tmp_path, pages=2) -> bytes:
    doc = pymupdf.open()
    for i in range(1, pages + 1):
        page = doc.new_page(width=960, height=540)
        page.insert_text((60, 90), f"Slide {i} title", fontsize=36)
        page.insert_text((60, 200), "PagedAttention cuts KV-cache waste", fontsize=18)
    p = tmp_path / "talk.pdf"
    doc.save(p)
    return p.read_bytes()


def make_app(tmp_path, comparator, llm=None, structurer=None):
    llm = llm or FakeLLM()

    class Main:
        model = "fake-model"

        async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
            if "Name the field it belongs to" in system:
                return {"domain": "LLM serving", "adjacent_field": "databases"}
            return await llm.complete_json(system=system, user_text=user_text, image_png=image_png, schema=schema)

    st = structurer or Structurer()
    return create_app(
        store=RunStore(tmp_path / "history"), cache=FileCache(tmp_path / "cache"), client_factory=lambda: Main(),
        helper_client_factory=lambda: None, structuring_client_factory=lambda: st, embedder=HashEmbedder(),
        frontend_dir=tmp_path / "none", comparator=comparator,
    ), llm, st


def run_through_server(c, tmp_path):
    run_id = c.post("/api/upload", files={"file": ("t.pdf", pdf_bytes(tmp_path), "application/pdf")}).json()["run_id"]
    c.post(f"/api/runs/{run_id}/start", json={"domain": "LLM serving", "adjacent_field": "databases"})
    import time
    for _ in range(200):
        body = c.get(f"/api/runs/{run_id}").json()
        if body["meta"]["status"] in ("complete", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish")


@pytest.mark.parametrize("comparator", ["cosine", "fieldwise"])
def test_the_flag_flips_between_the_two_paths_without_a_code_change(tmp_path, comparator):
    app, llm, st = make_app(tmp_path, comparator)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["comparator"] == comparator
        body = run_through_server(c, tmp_path)
    assert body["meta"]["status"] == "complete" and body["meta"]["comparator"] == comparator
    assert body["rollup"]["comparator"] == comparator
    for r in body["results"]:
        assert r["metrics"]["comparator"] == comparator
        assert ("intent_alignment" in r["metrics"]) == (comparator == "cosine")
        assert ("comparisons" in r["metrics"]) == (comparator == "fieldwise")
    assert len(st.calls) == (2 if comparator == "fieldwise" else 0)  # the cosine path never touches the structuring model


def test_a_saved_run_keeps_the_comparator_that_made_it_when_the_flag_flips(tmp_path):
    app, _, _ = make_app(tmp_path, "cosine")
    with TestClient(app) as c:
        run_id = run_through_server(c, tmp_path)["meta"]["run_id"]
    other, _, _ = make_app(tmp_path, "fieldwise")  # same history directory, flag now flipped
    with TestClient(other) as c2:
        body = c2.get(f"/api/runs/{run_id}").json()
        assert c2.get("/api/health").json()["comparator"] == "fieldwise"
    assert body["meta"]["comparator"] == "cosine" and body["rollup"]["comparator"] == "cosine"
    assert all("intent_alignment" in r["metrics"] for r in body["results"])


# ------------------------------------------------------------ figure descriptions at upload


def chart_pdf(tmp_path) -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page(width=960, height=540)
    p1.insert_text((60, 90), "Text only", fontsize=36)
    p1.insert_textbox(pymupdf.Rect(60, 150, 900, 300), "A slide made only of ordinary body text, nothing drawn on it at all.", fontsize=20)
    p2 = doc.new_page(width=960, height=540)
    p2.insert_text((60, 60), "Latency", fontsize=30)
    p2.draw_line((120, 470), (860, 470)); p2.draw_line((120, 470), (120, 150))
    for i in range(1, 9):
        p2.draw_line((120 + i * 90, 466), (120 + i * 90, 474))
    p2.draw_polyline([(120, 460), (400, 400), (860, 150)])
    f = tmp_path / "chart.pdf"
    doc.save(f)
    return f.read_bytes()


class FigureDescriber:
    model = "fake-haiku"

    def __init__(self):
        self.calls = 0

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls += 1
        return {"has_figure": True, "description": "One line rises from left to right on axes labelled Requests and Latency."}


def test_upload_describes_only_the_figure_slides_once_and_the_run_hands_the_text_to_every_persona(tmp_path):
    describer, llm = FigureDescriber(), FakeLLM()
    app, _, st = make_app(tmp_path, "fieldwise", llm=llm)
    app = create_app(store=RunStore(tmp_path / "history"), cache=FileCache(tmp_path / "cache"), client_factory=lambda: None,
                     helper_client_factory=lambda: describer, structuring_client_factory=lambda: st, embedder=HashEmbedder(),
                     frontend_dir=tmp_path / "none", comparator="fieldwise")
    with TestClient(app) as c:
        up = c.post("/api/upload", files={"file": ("chart.pdf", chart_pdf(tmp_path), "application/pdf")}).json()
        assert describer.calls == 1  # the text-only slide sent nothing
        again = c.post("/api/upload", files={"file": ("chart.pdf", chart_pdf(tmp_path), "application/pdf")}).json()
        assert describer.calls == 1  # the same deck is described once: the second upload came from the cache
    import json
    recs = json.loads((tmp_path / "history" / up["run_id"] / "input.json").read_text())
    assert recs[0]["image_content"] is None and recs[0]["figure_signals"]["carries_figure"] is False
    assert recs[1]["figure_signals"]["carries_figure"] is True and recs[1]["image_content"]["text"].startswith("One line rises")
    assert recs[1]["image_content"]["machine_generated"] is True
    assert json.loads((tmp_path / "history" / again["run_id"] / "input.json").read_text())[1]["image_content"] == recs[1]["image_content"]


def test_with_no_helper_client_an_upload_still_works_and_the_figure_is_simply_not_described(tmp_path):
    app, _, _ = make_app(tmp_path, "fieldwise")  # make_app gives no helper client
    with TestClient(app) as c:
        up = c.post("/api/upload", files={"file": ("chart.pdf", chart_pdf(tmp_path), "application/pdf")})
    assert up.status_code == 200
    import json
    recs = json.loads((tmp_path / "history" / up.json()["run_id"] / "input.json").read_text())
    assert recs[1]["figure_signals"]["carries_figure"] is True and recs[1]["image_content"] is None
