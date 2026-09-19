"""Mechanics of audiences.py with a fake LLM: isolation, caching, concurrency, schema.

These prove the plumbing. They say nothing about whether the personas behave like
different audiences; that is the live gate in test_gate_live.py.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeLLM, sentinel_payload
from pydantic import ValidationError
from slides import CLEAR_PROFILE, JARGON_PROFILE, clear_slide, jargon_slide

from sightline.audiences import (
    PERSONAS,
    RESPONSE_SCHEMA,
    AudienceEngine,
    AudienceResponse,
    AudienceResponseError,
    CacheMiss,
    FileCache,
    SlideInput,
    build_prompt,
    slide_hash,
)

VALID = {
    "takeaway": "Sales grew.",
    "confidence": 0.9,
    "unresolved_terms": [],
    "questions": [],
    "inferred_claim": "The plan works.",
}


# ------------------------------------------------------------------------- schema


def test_response_has_exactly_the_five_spec_fields():
    assert set(AudienceResponse.model_fields) == {
        "takeaway", "confidence", "unresolved_terms", "questions", "inferred_claim",
    }


def test_json_schema_matches_pydantic_model():
    assert set(RESPONSE_SCHEMA["properties"]) == set(AudienceResponse.model_fields)
    assert set(RESPONSE_SCHEMA["required"]) == set(AudienceResponse.model_fields)
    assert RESPONSE_SCHEMA["additionalProperties"] is False


def test_no_ratings_anywhere_in_the_backend():
    """Design rule: audiences perform comprehension, they never rate."""
    banned_fields = {"score", "rating", "clarity", "grade", "rank"}
    assert not banned_fields & set(AudienceResponse.model_fields)
    pattern = re.compile(r"\b\d+(\.\d+)?\s*/\s*10\b|\bclarity\b\s*[:=]\s*\d", re.I)
    for path in Path(__file__).parent.parent.joinpath("sightline").glob("*.py"):
        assert not pattern.search(path.read_text()), f"rating-style output in {path.name}"


@pytest.mark.parametrize(
    "patch",
    [
        {"confidence": 7},  # a 0-10 rating smuggled in
        {"confidence": -0.1},
        {"takeaway": "  "},
        {"inferred_claim": ""},
        {"clarity": 7},  # extra field
    ],
)
def test_invalid_responses_are_rejected(patch):
    with pytest.raises(ValidationError):
        AudienceResponse.model_validate({**VALID, **patch})


def test_terms_are_stripped_and_deduplicated_case_insensitively():
    r = AudienceResponse.model_validate(
        {**VALID, "unresolved_terms": [" KV-cache ", "kv-cache", "", "TTFT"]}
    )
    assert r.unresolved_terms == ["KV-cache", "TTFT"]


# ----------------------------------------------------------------------- prompts


def test_each_persona_gets_a_distinct_knowledge_prompt():
    slide = jargon_slide()
    prompts = {p: build_prompt(p, slide, JARGON_PROFILE)[0] for p in PERSONAS}
    assert len(set(prompts.values())) == 3
    for p in PERSONAS:
        assert f"Your background: {p.upper()}." in prompts[p]
        assert JARGON_PROFILE.domain in prompts[p]
    assert JARGON_PROFILE.adjacent_field in prompts["peer"]
    assert JARGON_PROFILE.adjacent_field not in prompts["expert"]


async def test_slide_text_and_image_reach_the_model():
    llm = FakeLLM()
    slide = clear_slide()
    await AudienceEngine(llm).read_slide(slide, CLEAR_PROFILE)
    assert len(llm.calls) == 3
    for call in llm.calls:
        assert call.image_png == slide.image_png and slide.image_png
        assert "Q3 sales grew 20% over Q2" in call.user_text
        assert call.schema is RESPONSE_SCHEMA


# --------------------------------------------------------------------- isolation


async def test_a_persona_never_sees_another_personas_output():
    llm = FakeLLM()
    slides = [clear_slide(i, with_image=False) for i in (1, 2, 3)]
    await AudienceEngine(llm).read_deck(slides, CLEAR_PROFILE)

    assert len(llm.calls) == 9
    for call in llm.calls:
        seen = call.system + call.user_text
        for other in PERSONAS:
            if other != call.persona:
                assert f"SENTINEL-{other.upper()}" not in seen, (
                    f"{call.persona} prompt leaked {other} output"
                )


async def test_running_context_is_each_personas_own_notes():
    llm = FakeLLM()
    slides = [clear_slide(i, with_image=False) for i in (1, 2, 3)]
    await AudienceEngine(llm).read_deck(slides, CLEAR_PROFILE)

    for p in PERSONAS:
        first, second, third = llm.calls_by(p)
        assert "first slide" in first.user_text
        assert f"SENTINEL-{p.upper()}-1 takeaway" in second.user_text
        assert f"SENTINEL-{p.upper()}-1 takeaway" in third.user_text
        assert f"SENTINEL-{p.upper()}-2 takeaway" in third.user_text


# ------------------------------------------------------------------ concurrency


async def test_three_personas_run_concurrently():
    llm = FakeLLM(delay=0.2)
    started = time.perf_counter()
    await AudienceEngine(llm).read_slide(clear_slide(with_image=False), CLEAR_PROFILE)
    elapsed = time.perf_counter() - started
    assert llm.max_in_flight == 3
    assert elapsed < 0.5  # sequential would be ~0.6


# ------------------------------------------------------------------------ caching


async def test_second_run_is_free_and_identical(tmp_path):
    cache = FileCache(tmp_path / "c")
    slide = clear_slide(with_image=False)

    llm1 = FakeLLM()
    first = await AudienceEngine(llm1, cache).read_slide(slide, CLEAR_PROFILE)
    assert len(llm1.calls) == 3 and not any(r.cached for r in first.values())

    llm2 = FakeLLM()
    second = await AudienceEngine(llm2, cache).read_slide(slide, CLEAR_PROFILE)  # new engine
    assert llm2.calls == []
    assert all(r.cached for r in second.values())
    assert all(r.attempts == 0 for r in second.values())
    assert {p: r.response for p, r in first.items()} == {p: r.response for p, r in second.items()}
    assert second["novice"].model == "fake-model"  # provenance survives the cache


@pytest.mark.parametrize(
    "change",
    [
        lambda s: replace(s, text=s.text + "\n[body] one more line"),
        lambda s: replace(s, image_png=b"different-image"),
        lambda s: replace(s, index=s.index + 1),
    ],
    ids=["text", "image", "index"],
)
async def test_cache_misses_when_the_slide_changes(tmp_path, change):
    cache = FileCache(tmp_path)
    slide = clear_slide()
    await AudienceEngine(FakeLLM(), cache).read_slide(slide, CLEAR_PROFILE)
    llm = FakeLLM()
    await AudienceEngine(llm, cache).read_slide(change(slide), CLEAR_PROFILE)
    assert len(llm.calls) == 3


async def test_cache_misses_when_the_deck_profile_changes(tmp_path):
    cache = FileCache(tmp_path)
    slide = clear_slide(with_image=False)
    await AudienceEngine(FakeLLM(), cache).read_slide(slide, CLEAR_PROFILE)
    llm = FakeLLM()
    await AudienceEngine(llm, cache).read_slide(slide, JARGON_PROFILE)
    assert len(llm.calls) == 3


async def test_changed_memory_invalidates_only_that_persona(tmp_path):
    cache = FileCache(tmp_path)
    slide = clear_slide(with_image=False)
    await AudienceEngine(FakeLLM(), cache).read_slide(slide, CLEAR_PROFILE)
    llm = FakeLLM()
    await AudienceEngine(llm, cache).read_slide(
        slide, CLEAR_PROFILE, memory={"novice": ((0, "I was lost earlier."),)}
    )
    assert [c.persona for c in llm.calls] == ["novice"]


async def test_bundled_read_only_cache_serves_the_demo_offline(tmp_path):
    bundled, runtime = tmp_path / "bundled", tmp_path / "runtime"
    slide = jargon_slide()
    await AudienceEngine(FakeLLM(), FileCache(bundled)).read_slide(slide, JARGON_PROFILE)
    before = sorted(p.name for p in bundled.iterdir())

    engine = AudienceEngine(None, FileCache(runtime, read_only_dirs=[bundled]), offline=True)
    out = await engine.read_slide(slide, JARGON_PROFILE)

    assert all(r.cached for r in out.values())
    assert sorted(p.name for p in bundled.iterdir()) == before  # never written
    assert not runtime.exists()


async def test_offline_miss_fails_immediately_and_never_calls_the_llm(tmp_path):
    llm = FakeLLM()
    engine = AudienceEngine(llm, FileCache(tmp_path), offline=True)
    with pytest.raises(CacheMiss):
        await engine.read_slide(clear_slide(with_image=False), CLEAR_PROFILE)
    assert llm.calls == []


async def test_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    slide = clear_slide(with_image=False)
    cache = FileCache(tmp_path)
    await AudienceEngine(FakeLLM(), cache).read_slide(slide, CLEAR_PROFILE)
    for f in tmp_path.glob("*.json"):
        f.write_text("{not json")
    llm = FakeLLM()
    await AudienceEngine(llm, cache).read_slide(slide, CLEAR_PROFILE)
    assert len(llm.calls) == 3


# ------------------------------------------------------------------ retry / errors


async def test_invalid_output_is_retried_once_with_the_rejection_reason(tmp_path):
    def responder(persona, nth, user_text):
        if persona == "novice" and nth == 1:
            return {**VALID, "confidence": 7}  # a 0-10 rating: must be rejected
        return sentinel_payload(persona, user_text)

    llm = FakeLLM(responder)
    out = await AudienceEngine(llm, FileCache(tmp_path)).read_slide(
        clear_slide(with_image=False), CLEAR_PROFILE
    )
    novice_calls = llm.calls_by("novice")
    assert len(novice_calls) == 2 and len(llm.calls_by("peer")) == 1
    assert "rejected" in novice_calls[1].user_text and "confidence" in novice_calls[1].user_text
    assert out["novice"].response.confidence == 0.5
    assert {p: r.attempts for p, r in out.items()} == {"novice": 2, "peer": 1, "expert": 1}


async def test_persistent_invalid_output_raises_and_is_never_cached(tmp_path):
    llm = FakeLLM(lambda p, n, u: {**VALID, "confidence": 7} if p == "expert" else sentinel_payload(p, u))
    cache = FileCache(tmp_path)
    slide = clear_slide(with_image=False)
    with pytest.raises(AudienceResponseError) as e:
        await AudienceEngine(llm, cache).read_slide(slide, CLEAR_PROFILE)
    assert e.value.persona == "expert" and len(e.value.attempts) == 2
    assert cache.get(slide_hash(slide, CLEAR_PROFILE), "expert", "x") is None
    assert not [f for f in tmp_path.glob("*__expert__*")]


def test_slide_input_cannot_carry_declared_intent():
    """The answer key must be structurally unreachable by the audiences."""
    assert not any("intent" in f for f in SlideInput.__dataclass_fields__)
