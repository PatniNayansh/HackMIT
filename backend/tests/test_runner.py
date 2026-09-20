from __future__ import annotations

import numpy as np
import pytest

from sightline.audiences import PERSONAS, AudienceEngine, DeckProfile, FileCache
from sightline.deck import rollup
from sightline.ingest import Slide
from sightline.runner import TolerantEngine, run_deck as _run_deck
from sightline.store import RunStore

from builders import HashEmbedder
from conftest import FakeLLM, sentinel_payload

PROFILE = DeckProfile("LLM serving", "distributed systems")


async def run_deck(*args, **kwargs):
    """These tests cover the ORIGINAL cosine path, which the COMPARATOR flag keeps reachable
    (tests/test_fieldwise_runner.py covers the field-wise path)."""
    return await _run_deck(*args, comparator="cosine", **kwargs)


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


class RecordingEmbedder(HashEmbedder):
    """Remembers every string it was asked to embed, so a test can see what alignment was measured against."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def embed(self, texts):
        self.seen.append(list(texts))
        return super().embed(texts)


async def test_every_slide_is_stored_with_readings_the_experts_takeaway_as_intent_and_metrics(store):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder())

    meta = store.load_meta(run_id)
    assert meta["status"] == "complete" and meta["error"] is None
    assert meta["model"] == "fake-model" and meta["embedding_model"] == "hash-embedder" and meta["schema_version"] == 2
    results = store.load_results(run_id)
    assert [r["index"] for r in results] == [1, 2, 3]

    r = results[1]
    assert set(r["readings"]) == set(PERSONAS) and all(x["ok"] for x in r["readings"].values())
    assert r["readings"]["novice"]["takeaway"] == "SENTINEL-NOVICE-2 takeaway"
    assert r["slide_intent"] == {"text": "SENTINEL-EXPERT-2 takeaway", "source": "expert_takeaway"}
    m = r["metrics"]
    assert m["intent"] == "SENTINEL-EXPERT-2 takeaway"
    assert m["intent_alignment"]["novice"]["inputs"] == {"intent": m["intent"], "novice": "SENTINEL-NOVICE-2 takeaway"}
    assert m["intent_alignment"]["expert"] == {
        "value": 1.0, "definitional": True, "inputs": {"intent": m["intent"], "expert": "SENTINEL-EXPERT-2 takeaway"},
    }
    assert r["scored_by"] == "hash-embedder" and r["text"] == "[title] slide 2"
    assert set(r["timing"]) == {"personas_s"}


ODD_TAKEAWAYS = [
    "throughput rose 2.4x—maybe more...",                  # ellipsis, dash, no full stop, lower case
    "  Leading and doubled  spaces stay as returned  ",          # the persona model's own spacing
    'She said "yes" & left … (a long one: ' + "and on " * 40 + ")",  # quotes, ampersand, very long
]


@pytest.mark.parametrize("takeaway", ODD_TAKEAWAYS)
async def test_the_string_shown_as_the_intent_is_the_string_alignment_was_measured_against(store, takeaway):
    """The page says the slide's intent IS the expert takeaway, so the displayed string, the stored
    reading and the string that was actually embedded for alignment must be identical, character for
    character: no truncation, no trailing-period fix, no re-casing."""
    run_id = start(store, n=1)
    emb = RecordingEmbedder()

    def responder(persona, n, user_text):
        p = sentinel_payload(persona, user_text)
        if persona == "expert":
            p["takeaway"] = takeaway
        return p

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), emb)

    (r,) = store.load_results(run_id)
    shown = r["slide_intent"]["text"]
    stored_reading = r["readings"]["expert"]["takeaway"]
    assert shown == stored_reading == r["metrics"]["intent"] == r["metrics"]["takeaways"]["expert"]
    assert shown == r["metrics"]["intent_alignment"]["novice"]["inputs"]["intent"] == r["metrics"]["intent_alignment"]["peer"]["inputs"]["intent"]
    assert emb.seen[0][0] == shown  # the first string embedded is the reference, and it is exactly what the page shows
    for c in (".", "…", "—", "&", '"'):
        if c in takeaway.strip():
            assert c in shown  # nothing was normalised away
    assert shown == takeaway.strip()  # only the persona validation's own strip, done before storage


async def test_there_is_no_extra_model_call_the_batch_is_three_persona_calls_a_slide(store):
    run_id = start(store)
    llm = FakeLLM()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())
    assert len(llm.calls) == 9 and {len(llm.calls_by(p)) for p in PERSONAS} == {3}


async def test_the_declared_intent_is_optional_and_stored_but_not_used(store):
    with_it, without = start(store, intent="Land the 2.4x."), start(store)
    for run_id in (with_it, without):
        await run_deck(store, run_id, TolerantEngine(FakeLLM()), HashEmbedder())
    assert store.load_meta(with_it)["intent"] == "Land the 2.4x." and store.load_meta(without)["intent"] is None
    a, b = store.load_results(with_it), store.load_results(without)
    assert [r["metrics"] for r in a] == [r["metrics"] for r in b]  # alignment is identical either way


async def test_personas_still_carry_only_their_own_notes_between_slides(store):
    run_id = start(store)
    llm = FakeLLM()
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())

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

    await run_deck(store, run_id, TolerantEngine(FakeLLM(responder)), HashEmbedder())

    assert store.load_meta(run_id)["status"] == "complete"  # a bad reply is not a failed run
    r1, r2, r3 = store.load_results(run_id)
    bad = r2["readings"]["novice"]
    assert bad["ok"] is False and bad["error"]["kind"] == "invalid_response"
    assert "1.7" in bad["error"]["message"] and len(bad["error"]["attempts"]) == 2
    assert "confidence" not in bad  # not clamped to 1.0, not kept as a value
    assert r2["readings"]["peer"]["ok"] and r2["readings"]["expert"]["ok"]  # the other two survive
    assert r2["metrics"] is None and "novice" in r2["metrics_error"]
    assert r2["slide_intent"]["text"] == "SENTINEL-EXPERT-2 takeaway"  # the expert was fine, so the intent still exists
    assert r1["metrics"] and r3["metrics"]
    assert rollup([r1, r2, r3])["unscored"] == [2]


async def test_a_bad_expert_reply_means_no_intent_and_no_metrics_for_that_slide(store):
    run_id = start(store)
    llm = FakeLLM(lambda p, n, u: {**sentinel_payload(p, u), **({"confidence": 9} if p == "expert" and "Slide 2." in u else {})})
    await run_deck(store, run_id, TolerantEngine(llm), HashEmbedder())
    r2 = store.load_results(run_id)[1]
    assert r2["slide_intent"] is None and r2["metrics"] is None
    assert "defines the intended reading" in r2["metrics_error"]


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
    await run_deck(store, second, TolerantEngine(None, cache, offline=True), HashEmbedder())  # no client at all

    assert store.load_meta(second)["status"] == "complete"
    a, b = store.load_results(first), store.load_results(second)
    assert [r["metrics"] for r in a] == [r["metrics"] for r in b]
    assert [r["slide_intent"] for r in a] == [r["slide_intent"] for r in b]
    assert all(x["cached"] for r in b for x in r["readings"].values())


async def test_offline_with_nothing_cached_fails_the_run_with_a_clear_reason(store, tmp_path):
    run_id = start(store)
    await run_deck(store, run_id, TolerantEngine(None, FileCache(tmp_path / "empty"), offline=True), HashEmbedder())
    meta = store.load_meta(run_id)
    assert meta["status"] == "failed" and "offline" in meta["error"]


def test_tolerant_engine_is_still_the_step_one_engine():
    assert issubclass(TolerantEngine, AudienceEngine)
