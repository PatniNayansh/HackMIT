from __future__ import annotations

import pytest

from sightline.audiences import PERSONAS, AudienceEngine, DeckProfile, FileCache
from sightline.deck import rollup
from sightline.ingest import Slide
from sightline.runner import TolerantEngine, run_deck
from sightline.store import RunStore

from builders import HashEmbedder
from conftest import FakeLLM, sentinel_payload

PROFILE = DeckProfile("LLM serving", "distributed systems")


def start(store, n=3, intent="Show the speedup."):
    meta = store.create_draft(
        title="t", source_filename="t.pdf", inferred=None, inference_error=None,
        slides=[Slide(i, f"[title] slide {i}", b"\x89PNG") for i in range(1, n + 1)],
    )
    store.update_meta(
        meta["run_id"], intent=intent, status="running",
        profile={"domain": PROFILE.domain, "adjacent_field": PROFILE.adjacent_field, "confirmed": True},
    )
    return meta["run_id"]


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "history")


async def test_every_slide_is_stored_with_three_readings_and_metrics(store):
    run_id = start(store)
    llm = FakeLLM()

    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())

    meta = store.load_meta(run_id)
    assert meta["status"] == "complete" and meta["error"] is None
    assert meta["model"] == "fake-model" and meta["embedding_model"] == "hash-embedder"
    results = store.load_results(run_id)
    assert [r["index"] for r in results] == [1, 2, 3]
    r = results[1]
    assert set(r["readings"]) == set(PERSONAS) and all(x["ok"] for x in r["readings"].values())
    assert r["readings"]["novice"]["takeaway"] == "SENTINEL-NOVICE-2 takeaway"
    # provenance survives: the metric carries the exact strings it was computed from
    assert r["metrics"]["intent_alignment"]["novice"]["inputs"] == {
        "intent": "Show the speedup.", "novice": "SENTINEL-NOVICE-2 takeaway",
    }
    assert r["scored_by"] == "hash-embedder" and r["text"] == "[title] slide 2"


async def test_personas_still_carry_only_their_own_notes_between_slides(store):
    run_id = start(store)
    llm = FakeLLM()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())

    slide3_novice = llm.calls_by("novice")[2].user_text
    assert "SENTINEL-NOVICE-1" in slide3_novice and "SENTINEL-NOVICE-2" in slide3_novice
    assert "SENTINEL-EXPERT" not in slide3_novice and "SENTINEL-PEER" not in slide3_novice
    assert "Show the speedup" not in slide3_novice  # the declared intent is never shown to a persona


async def test_out_of_range_confidence_is_recorded_as_an_error_never_clamped(store):
    run_id = start(store)

    def responder(persona, n, user_text):
        p = sentinel_payload(persona, user_text)
        if persona == "novice" and "Slide 2." in user_text:
            p["confidence"] = 1.7
        return p

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), HashEmbedder())

    assert store.load_meta(run_id)["status"] == "complete"  # a bad reply is not a failed run
    r1, r2, r3 = store.load_results(run_id)
    bad = r2["readings"]["novice"]
    assert bad["ok"] is False and bad["error"]["kind"] == "invalid_response"
    assert "1.7" in bad["error"]["message"] and len(bad["error"]["attempts"]) == 2
    assert "confidence" not in bad  # not clamped to 1.0, not kept as a value
    assert r2["readings"]["peer"]["ok"] and r2["readings"]["expert"]["ok"]  # the other two survive
    assert r2["metrics"] is None and "novice" in r2["metrics_error"]
    assert r1["metrics"] and r3["metrics"]
    assert rollup([r1, r2, r3])["unscored"] == [2]


async def test_a_persona_that_failed_has_no_note_for_that_slide(store):
    run_id = start(store)
    llm = FakeLLM(lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 9} if p == "novice" and "Slide 1." in u else {})})
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())
    slide3_novice = llm.calls_by("novice")[-1].user_text
    assert "SENTINEL-NOVICE-1" not in slide3_novice and "SENTINEL-NOVICE-2" in slide3_novice


async def test_when_every_persona_fails_for_infrastructure_reasons_the_run_stops(store):
    run_id = start(store)

    def responder(persona, n, user_text):
        raise ConnectionError("api down")

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), HashEmbedder())

    meta = store.load_meta(run_id)
    assert meta["status"] == "failed" and "api down" in meta["error"] and "slide 1" in meta["error"]
    assert store.load_results(run_id) == []  # no slide of wall-to-wall errors was stored


async def test_a_saved_deck_reruns_with_no_client_and_no_key_from_the_cache(store, tmp_path):
    cache = FileCache(tmp_path / "cache")
    first = start(store)
    await run_deck(store, first, TolerantEngine(FakeLLM(), cache), HashEmbedder())

    second = start(store)
    engine = TolerantEngine(None, cache, offline=True)  # no client at all
    await run_deck(store, second, engine, HashEmbedder())

    assert store.load_meta(second)["status"] == "complete"
    assert [r["metrics"]["intent_alignment"] for r in store.load_results(second)] == [
        r["metrics"]["intent_alignment"] for r in store.load_results(first)
    ]
    assert all(x["cached"] for r in store.load_results(second) for x in r["readings"].values())


async def test_offline_with_nothing_cached_fails_the_run_with_a_clear_reason(store, tmp_path):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(None, FileCache(tmp_path / "empty"), offline=True), HashEmbedder())
    meta = store.load_meta(run_id)
    assert meta["status"] == "failed" and "offline" in meta["error"]


def test_tolerant_engine_is_still_the_step_one_engine():
    assert issubclass(TolerantEngine, AudienceEngine)
