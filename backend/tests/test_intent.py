from __future__ import annotations

import pytest

from sightline import intent as intent_mod
from sightline.audiences import AudienceResponse
from sightline.intent import (
    FileIntentCache, cache_key, infer_slide_intent, template_intent, unsupported_terms,
)

EXPERT = AudienceResponse(
    takeaway="The slide claims that pairing PagedAttention with speculative decoding lifts goodput 2.4x at p99.",
    confidence=0.9,
    unresolved_terms=[],
    questions=["What is the baseline?"],
    inferred_claim="Combining PagedAttention and speculative decoding delivers 2.4x higher goodput at p99 latency.",
)
SLIDE = "[title] PagedAttention + speculative decoding: 2.4x goodput at p99\n[body] KV-cache waste under 4%"


class Client:
    model = "claude-haiku-4-5"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user": user_text, "image": image_png})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"intent": r}


async def test_the_model_sentence_is_used_and_records_where_it_came_from():
    c = Client("Pairing PagedAttention with speculative decoding raises goodput 2.4x at p99.")
    out = await infer_slide_intent(EXPERT, SLIDE, c)
    assert out.source == "model" and out.model == "claude-haiku-4-5" and out.reason is None
    assert out.text == "Pairing PagedAttention with speculative decoding raises goodput 2.4x at p99."
    assert out.derived_from == {"takeaway": EXPERT.takeaway, "inferred_claim": EXPERT.inferred_claim}


async def test_the_model_sees_only_the_experts_takeaway_and_claim_never_the_slide_or_image():
    c = Client("Pairing PagedAttention with speculative decoding raises goodput 2.4x at p99.")
    await infer_slide_intent(EXPERT, SLIDE, c)
    (call,) = c.calls
    assert EXPERT.takeaway in call["user"] and EXPERT.inferred_claim in call["user"]
    assert "KV-cache" not in call["user"] and "What is the baseline" not in call["user"]
    assert call["image"] is None


async def test_a_sentence_that_adds_a_term_is_retried_then_accepted():
    c = Client("Quantum annealing improves goodput.", "Speculative decoding with PagedAttention raises goodput.")
    out = await infer_slide_intent(EXPERT, SLIDE, c)
    assert out.source == "model" and out.attempts == 2
    assert "annealing" in c.calls[1]["user"]  # the retry says which words were rejected


async def test_two_bad_sentences_fall_back_to_the_experts_claim_and_say_why():
    c = Client("Quantum annealing improves goodput.", "Blockchain sharding improves goodput.")
    out = await infer_slide_intent(EXPERT, SLIDE, c)
    assert out.source == "template" and out.model is None
    assert out.text == "Combining PagedAttention and speculative decoding delivers 2.4x higher goodput at p99 latency."
    assert "Blockchain" in out.reason and out.attempts == 2


async def test_an_api_failure_degrades_to_the_template_instead_of_failing_the_slide():
    out = await infer_slide_intent(EXPERT, SLIDE, Client(ConnectionError("down")))
    assert out.source == "template" and "ConnectionError" in out.reason


async def test_no_client_means_template():
    out = await infer_slide_intent(EXPERT, SLIDE, None)
    assert out.source == "template" and out.reason == "no model client available"


async def test_template_mode_never_calls_the_model(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_INTENT_MODE", "template")
    c = Client("unused")
    out = await infer_slide_intent(EXPERT, SLIDE, c)
    assert out.source == "template" and c.calls == []


async def test_vocabulary_comes_from_the_slide_too(tmp_path):
    # "KV-cache" is on the slide, not in the expert's two texts: allowed.
    c = Client("PagedAttention keeps KV-cache waste low while speculative decoding raises goodput.")
    assert (await infer_slide_intent(EXPERT, SLIDE, c)).source == "model"


async def test_cached_intents_are_reused_with_no_client(tmp_path):
    cache = FileIntentCache(tmp_path)
    c = Client("Speculative decoding with PagedAttention raises goodput 2.4x.")
    first = await infer_slide_intent(EXPERT, SLIDE, c, cache=cache)
    again = await infer_slide_intent(EXPERT, SLIDE, None, cache=cache)  # no key, no client
    assert again.text == first.text and again.cached is True and again.source == "model"
    assert again.model == "claude-haiku-4-5"  # provenance survives even though there is no client to ask


async def test_template_results_are_never_cached(tmp_path):
    cache = FileIntentCache(tmp_path)
    await infer_slide_intent(EXPERT, SLIDE, None, cache=cache)
    assert cache.get(cache_key(EXPERT)) is None


def test_unsupported_terms_ignores_function_words_plurals_and_generic_verbs():
    src = [EXPERT.takeaway, EXPERT.inferred_claim]
    assert unsupported_terms("The slide shows that speculative decoding delivers higher goodput.", src) == []
    assert unsupported_terms("Pairing the two lifts goodputs.", src) == []  # ordinary synonyms, plural of a source term
    assert unsupported_terms("Uses blockchain at p99 and 9.9x.", src) == ["blockchain", "9.9x"]
    assert unsupported_terms("It beats vLLM and TTFT.", src) == ["vLLM", "TTFT"]  # coined or capitalised terms


def test_template_is_the_claim_verbatim_as_a_sentence():
    e = AudienceResponse(takeaway="t", confidence=0.5, unresolved_terms=[], questions=[], inferred_claim='"a claim without a stop"')
    assert template_intent(e) == "A claim without a stop."
