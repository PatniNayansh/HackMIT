from __future__ import annotations

import re

import pytest

from sightline.audiences import PERSONAS, AudienceEngine, DeckProfile, FileCache
from sightline.deck import rollup
from sightline.ingest import Slide
from sightline.intent import FileIntentCache
from sightline.runner import TolerantEngine, run_deck
from sightline.store import RunStore

from builders import HashEmbedder
from conftest import FakeLLM, sentinel_payload

PROFILE = DeckProfile("LLM serving", "distributed systems")


class IntentEcho:
    """Stands in for the Haiku client: 'rephrases' the expert takeaway it is shown by returning
    it. Records what it was sent."""

    model = "fake-haiku"

    def __init__(self):
        self.calls = []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append(user_text)
        takeaway = re.search(r"<expert_takeaway>\n(.*?)\n</expert_takeaway>", user_text, re.S).group(1)
        return {"intent": takeaway}


def start(store, n=3, intent=None):
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


async def test_every_slide_is_stored_with_readings_an_inferred_intent_and_metrics(store):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), IntentEcho())

    meta = store.load_meta(run_id)
    assert meta["status"] == "complete" and meta["error"] is None
    assert meta["model"] == "fake-model" and meta["embedding_model"] == "hash-embedder"
    assert meta["intent_model"] == "fake-haiku" and meta["schema_version"] == 2
    results = store.load_results(run_id)
    assert [r["index"] for r in results] == [1, 2, 3]

    r = results[1]
    assert set(r["readings"]) == set(PERSONAS) and all(x["ok"] for x in r["readings"].values())
    assert r["readings"]["novice"]["takeaway"] == "SENTINEL-NOVICE-2 takeaway"
    # the intent is the one derived from the EXPERT reading, and it is what novice and peer are measured against
    assert r["slide_intent"]["source"] == "model" and r["slide_intent"]["text"] == "SENTINEL-EXPERT-2 takeaway."
    assert r["slide_intent"]["derived_from"]["takeaway"] == "SENTINEL-EXPERT-2 takeaway"
    m = r["metrics"]
    assert m["intent"] == "SENTINEL-EXPERT-2 takeaway."
    assert m["intent_alignment"]["novice"]["inputs"] == {"intent": m["intent"], "novice": "SENTINEL-NOVICE-2 takeaway"}
    assert m["intent_alignment"]["expert"] == {
        "value": 1.0, "definitional": True, "inputs": {"intent": m["intent"], "expert": "SENTINEL-EXPERT-2 takeaway"},
    }
    assert r["scored_by"] == "hash-embedder" and r["text"] == "[title] slide 2"
    assert set(r["timing"]) == {"personas_s", "intent_s"}


async def test_the_intent_model_sees_only_the_experts_reading(store):
    run_id = start(store, intent="The presenter's own sentence.")
    intent = IntentEcho()
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), intent)
    assert len(intent.calls) == 3  # one per slide, after that slide's personas
    for call in intent.calls:
        assert "SENTINEL-EXPERT" in call
        assert "SENTINEL-NOVICE" not in call and "SENTINEL-PEER" not in call
        assert "presenter's own sentence" not in call  # the declared intent is not an input to anything


async def test_the_declared_intent_is_optional_and_stored_but_not_used(store):
    with_it, without = start(store, intent="Land the 2.4x."), start(store)
    for run_id in (with_it, without):
        await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), IntentEcho())
    assert store.load_meta(with_it)["intent"] == "Land the 2.4x." and store.load_meta(without)["intent"] is None
    a, b = store.load_results(with_it), store.load_results(without)
    assert [r["metrics"] for r in a] == [r["metrics"] for r in b]  # alignment is identical either way


async def test_personas_still_carry_only_their_own_notes_between_slides(store):
    run_id = start(store)
    llm = FakeLLM()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder(), IntentEcho())

    slide3_novice = llm.calls_by("novice")[2].user_text
    assert "SENTINEL-NOVICE-1" in slide3_novice and "SENTINEL-NOVICE-2" in slide3_novice
    assert "SENTINEL-EXPERT" not in slide3_novice and "SENTINEL-PEER" not in slide3_novice


async def test_out_of_range_confidence_is_recorded_as_an_error_never_clamped(store):
    run_id = start(store)

    def responder(persona, n, user_text):
        p = sentinel_payload(persona, user_text)
        if persona == "novice" and "Slide 2." in user_text:
            p["confidence"] = 1.7
        return p

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), HashEmbedder(), IntentEcho())

    assert store.load_meta(run_id)["status"] == "complete"  # a bad reply is not a failed run
    r1, r2, r3 = store.load_results(run_id)
    bad = r2["readings"]["novice"]
    assert bad["ok"] is False and bad["error"]["kind"] == "invalid_response"
    assert "1.7" in bad["error"]["message"] and len(bad["error"]["attempts"]) == 2
    assert "confidence" not in bad  # not clamped to 1.0, not kept as a value
    assert r2["readings"]["peer"]["ok"] and r2["readings"]["expert"]["ok"]  # the other two survive
    assert r2["metrics"] is None and "novice" in r2["metrics_error"]
    assert r2["slide_intent"]["text"]  # the expert was fine, so its intended reading still exists
    assert r1["metrics"] and r3["metrics"]
    assert rollup([r1, r2, r3])["unscored"] == [2]


async def test_a_bad_expert_reply_means_no_intent_and_no_metrics_for_that_slide(store):
    run_id = start(store)
    llm = FakeLLM(lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 9} if p == "expert" and "Slide 2." in u else {})})
    intent = IntentEcho()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder(), intent)
    r2 = store.load_results(run_id)[1]
    assert r2["slide_intent"] is None and r2["metrics"] is None
    assert "defines the intended reading" in r2["metrics_error"]
    assert len(intent.calls) == 2  # no intent call for the slide whose expert reply was unusable


async def test_a_persona_that_failed_has_no_note_for_that_slide(store):
    run_id = start(store)
    llm = FakeLLM(lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 9} if p == "novice" and "Slide 1." in u else {})})
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder(), IntentEcho())
    slide3_novice = llm.calls_by("novice")[-1].user_text
    assert "SENTINEL-NOVICE-1" not in slide3_novice and "SENTINEL-NOVICE-2" in slide3_novice


async def test_with_no_intent_client_the_experts_claim_stands_in_and_says_so(store):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder(), None)
    r = store.load_results(run_id)[0]
    assert r["slide_intent"]["source"] == "template" and r["slide_intent"]["reason"] == "no model client available"
    assert r["slide_intent"]["text"] == "SENTINEL-EXPERT-1 claim."
    assert store.load_meta(run_id)["intent_model"] == "template (expert claim)"


async def test_when_every_persona_fails_for_infrastructure_reasons_the_run_stops(store):
    run_id = start(store)

    def responder(persona, n, user_text):
        raise ConnectionError("api down")

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), HashEmbedder(), IntentEcho())

    meta = store.load_meta(run_id)
    assert meta["status"] == "failed" and "api down" in meta["error"] and "slide 1" in meta["error"]
    assert store.load_results(run_id) == []  # no slide of wall-to-wall errors was stored


async def test_a_saved_deck_reruns_with_no_client_and_no_key_from_the_caches(store, tmp_path):
    cache, intents = FileCache(tmp_path / "cache"), FileIntentCache(tmp_path / "intents")
    first = start(store)
    await run_deck(store, first, TolerantEngine(FakeLLM(), cache), HashEmbedder(), IntentEcho(), intents)

    second = start(store)
    await run_deck(store, second, TolerantEngine(None, cache, offline=True), HashEmbedder(), None, intents)  # no clients at all

    assert store.load_meta(second)["status"] == "complete"
    a, b = store.load_results(first), store.load_results(second)
    assert [r["metrics"] for r in a] == [r["metrics"] for r in b]
    assert all(r["slide_intent"]["cached"] and r["slide_intent"]["source"] == "model" for r in b)
    assert all(x["cached"] for r in b for x in r["readings"].values())


async def test_offline_with_nothing_cached_fails_the_run_with_a_clear_reason(store, tmp_path):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(None, FileCache(tmp_path / "empty"), offline=True), HashEmbedder())
    meta = store.load_meta(run_id)
    assert meta["status"] == "failed" and "offline" in meta["error"]


def test_tolerant_engine_is_still_the_step_one_engine():
    assert issubclass(TolerantEngine, AudienceEngine)
