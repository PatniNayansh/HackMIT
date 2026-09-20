"""fix.py: the revise-and-rescore loop.

Uses AngleEmbedder (see builders.py) so each persona's alignment to the intent is an exact,
hand-set number, and improvement (or its absence) can be asserted exactly rather than just
"some real model happened to like the revision better."
"""

from __future__ import annotations

import pytest

from profe.audiences import PERSONAS, AudienceEngine, DeckProfile, SlideInput
from profe.divergence import score_slide
from profe.fix import propose_revision, run_fix
from profe.llm import LLMError

from builders import INTENT, AngleEmbedder
from conftest import FakeLLM

ORIGINAL_TEXT = "[title] Serving faster\n[body] Uses UNDEFINED_TERM without explanation"
REVISED_TEXT = "[title] Serving faster\n[body] Explains UNDEFINED_TERM clearly"


class FixClient:
    """Routes persona-shaped calls (system prompt identifies a persona) to a real FakeLLM;
    routes fix.py's own revision-proposal call (identified by its own marker line in the
    system prompt) to `revision_payload`. Two different call *shapes* sharing one client,
    the same way the real Anthropic client serves both from the app."""

    model = "fake-model"

    def __init__(self, responder=None, revision_payload=None):
        self.audiences = FakeLLM(responder)
        self.revision_payload = revision_payload or {
            "revised_text": REVISED_TEXT,
            "rationale": "defined UNDEFINED_TERM",
        }
        self.revision_calls: list[str] = []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        if "You revise ONE presentation slide" in system:
            self.revision_calls.append(user_text)
            return self.revision_payload
        return await self.audiences.complete_json(
            system=system, user_text=user_text, image_png=image_png, schema=schema, max_tokens=max_tokens
        )


def _responder(cos_by_persona: dict, terms_by_persona: dict):
    """A persona responder keyed off whether the ORIGINAL or REVISED slide text is in the
    prompt, so the same three personas can be asked to read both without a second engine."""

    def respond(persona, nth, user_text):
        cos, terms = (
            (cos_by_persona["after"][persona], terms_by_persona["after"][persona])
            if REVISED_TEXT in user_text
            else (cos_by_persona["before"][persona], terms_by_persona["before"][persona])
        )
        return {
            "takeaway": f"cos={cos:.2f} reading",
            "confidence": 0.6,
            "unresolved_terms": terms,
            "questions": [],
            "inferred_claim": "c",
        }

    return respond


async def _read_and_score(client, slide, profile) -> tuple:
    engine = AudienceEngine(client, cache=None)
    readings = await engine.read_slide(slide, profile)
    responses = {p: readings[p].response for p in PERSONAS}
    before = score_slide(INTENT, responses, AngleEmbedder(), slide.index)
    return engine, before, responses["novice"].unresolved_terms


@pytest.mark.asyncio
async def test_a_revision_that_closes_the_gap_is_reported_as_improved():
    responder = _responder(
        cos_by_persona={"before": {"novice": 0.10, "peer": 0.50, "expert": 0.90}, "after": {"novice": 0.85, "peer": 0.88, "expert": 0.90}},
        terms_by_persona={"before": {"novice": ["UNDEFINED_TERM"], "peer": [], "expert": []}, "after": {"novice": [], "peer": [], "expert": []}},
    )
    client = FixClient(responder)
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    profile = DeckProfile("LLM serving", "distributed systems")
    engine, before, novice_terms = await _read_and_score(client, slide, profile)

    result = await run_fix(client, engine, AngleEmbedder(), slide, profile, INTENT, before, novice_terms)

    assert result.improved is True
    assert result.revised_text == REVISED_TEXT
    assert result.audience_divergence_delta < 0
    assert result.term_gap_delta == -1  # one novice-only term resolved, none introduced
    assert result.newly_unresolved_terms == ()
    assert result.before.audience_divergence.value > result.after.audience_divergence.value


@pytest.mark.asyncio
async def test_a_revision_that_does_not_help_is_reported_honestly_not_hidden():
    # The "revision" is asked for, but the audiences read it exactly as divergently as the
    # original -- e.g. the model reworded without actually resolving anything.
    same_cos = {"novice": 0.10, "peer": 0.50, "expert": 0.90}
    same_terms = {"novice": ["UNDEFINED_TERM"], "peer": [], "expert": []}
    responder = _responder(
        cos_by_persona={"before": same_cos, "after": same_cos},
        terms_by_persona={"before": same_terms, "after": same_terms},
    )
    client = FixClient(responder)
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    profile = DeckProfile("LLM serving", "distributed systems")
    engine, before, novice_terms = await _read_and_score(client, slide, profile)

    result = await run_fix(client, engine, AngleEmbedder(), slide, profile, INTENT, before, novice_terms)

    assert result.improved is False
    assert result.audience_divergence_delta == pytest.approx(0.0, abs=1e-9)
    # The failed attempt is still fully reported, not swallowed:
    assert result.revised_text == REVISED_TEXT
    assert result.after.audience_divergence.value == pytest.approx(result.before.audience_divergence.value)


@pytest.mark.asyncio
async def test_improvement_requires_both_metrics_not_just_one():
    # Divergence falls, but a NEW term goes unresolved: a real tradeoff, not a clean win.
    responder = _responder(
        cos_by_persona={"before": {"novice": 0.10, "peer": 0.50, "expert": 0.90}, "after": {"novice": 0.85, "peer": 0.88, "expert": 0.90}},
        terms_by_persona={
            "before": {"novice": ["UNDEFINED_TERM"], "peer": [], "expert": []},
            "after": {"novice": ["A_NEW_TERM"], "peer": [], "expert": []},
        },
    )
    client = FixClient(responder)
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    profile = DeckProfile("LLM serving", "distributed systems")
    engine, before, novice_terms = await _read_and_score(client, slide, profile)

    result = await run_fix(client, engine, AngleEmbedder(), slide, profile, INTENT, before, novice_terms)

    assert result.audience_divergence_delta < 0  # divergence did improve...
    assert result.term_gap_delta == 0  # ...and the raw COUNT looks unchanged...
    assert result.newly_unresolved_terms == ("A_NEW_TERM",)  # ...but it's a swap, not a fix
    assert result.improved is False


@pytest.mark.asyncio
async def test_the_prompt_names_the_unresolved_terms_and_the_gap_size():
    client = FixClient(_responder(
        cos_by_persona={"before": {"novice": 0.1, "peer": 0.5, "expert": 0.9}, "after": {"novice": 0.9, "peer": 0.9, "expert": 0.9}},
        terms_by_persona={"before": {"novice": ["KV-cache"], "peer": [], "expert": []}, "after": {"novice": [], "peer": [], "expert": []}},
    ))
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    profile = DeckProfile("LLM serving", "distributed systems")
    engine, before, novice_terms = await _read_and_score(client, slide, profile)

    await run_fix(client, engine, AngleEmbedder(), slide, profile, INTENT, before, novice_terms)

    assert len(client.revision_calls) == 1
    prompt = client.revision_calls[0]
    assert "KV-cache" in prompt
    assert f"{before.blind_spot_score.value:.2f}" in prompt
    assert ORIGINAL_TEXT in prompt


@pytest.mark.asyncio
async def test_propose_revision_rejects_an_empty_revised_text():
    client = FixClient(revision_payload={"revised_text": "  ", "rationale": "nothing"})
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    with pytest.raises(LLMError, match="empty"):
        await propose_revision(client, slide, INTENT, ["UNDEFINED_TERM"], 0.5)


@pytest.mark.asyncio
async def test_revised_reading_carries_no_memory_of_the_original_slide():
    """The revised slide is read cold: fix.py must not pass the original's readings back in
    as persona memory, or a persona could "remember" resolving a term it never actually saw
    defined on the revision itself."""
    seen_memories = []

    class MemorySpyClient(FixClient):
        async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
            if "You revise ONE presentation slide" not in system:
                seen_memories.append(user_text)
            return await super().complete_json(
                system=system, user_text=user_text, image_png=image_png, schema=schema, max_tokens=max_tokens
            )

    client = MemorySpyClient(_responder(
        cos_by_persona={"before": {"novice": 0.1, "peer": 0.5, "expert": 0.9}, "after": {"novice": 0.9, "peer": 0.9, "expert": 0.9}},
        terms_by_persona={"before": {"novice": ["X"], "peer": [], "expert": []}, "after": {"novice": [], "peer": [], "expert": []}},
    ))
    slide = SlideInput(1, ORIGINAL_TEXT, None)
    profile = DeckProfile("LLM serving", "distributed systems")
    engine, before, novice_terms = await _read_and_score(client, slide, profile)

    await run_fix(client, engine, AngleEmbedder(), slide, profile, INTENT, before, novice_terms)

    revised_calls = [m for m in seen_memories if REVISED_TEXT in m]
    assert revised_calls and all("first slide" in m for m in revised_calls)
