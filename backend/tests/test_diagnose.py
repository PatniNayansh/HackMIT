from __future__ import annotations

import pytest

from sightline import diagnose
from sightline.diagnose import RecommendationsUnavailable, quoted_in, recommend
from sightline.intent import SlideIntent
from sightline.llm import LLMError

from builders import slide_result

INTENT = {"text": "Speculative decoding with PagedAttention raises goodput 2.4x."}


def slide(**terms):
    r = slide_result(1, text="[title] PagedAttention + speculative decoding\n[body] Draft model acceptance alpha = 0.78 at gamma = 5")
    r["readings"]["novice"].update(
        takeaway="The presenter combines two techniques to speed something up by 2.4x.",
        inferred_claim="Two techniques together make it faster.",
        questions=["What is goodput?", "Faster than what baseline?"],
        unresolved_terms=["goodput", "PagedAttention", "acceptance alpha"],
    )
    r["readings"]["peer"].update(unresolved_terms=["gamma"], questions=[], takeaway="A speedup from two methods.")
    r["readings"]["expert"].update(unresolved_terms=[], takeaway="Combining the methods lifts goodput.")
    return r


class Client:
    model = "fake-sonnet"

    def __init__(self, reply=None, *, fail=0):
        self.reply, self.fail, self.calls = reply or {}, fail, []

    async def complete_json(self, *, system, user_text, image_png, schema, max_tokens=4096):
        self.calls.append({"system": system, "user": user_text, "image": image_png, "schema": schema})
        if self.fail:
            self.fail -= 1
            raise LLMError("response was not valid JSON")
        return self.reply


GOOD = {
    "novice": [
        {"bullet": "Define goodput on first use", "evidence": "goodput"},
        {"bullet": "State the baseline the 2.4x is measured against", "evidence": "Faster than what baseline?"},
    ],
    "peer": [{"bullet": "Say what gamma is, or drop it from the slide", "evidence": "gamma"}],
    "expert_flagged": [],
}


async def test_bullets_that_quote_their_own_persona_are_kept():
    out = await recommend(slide(), INTENT, client=Client(GOOD))
    assert [b["bullet"] for b in out["novice"]] == ["Define goodput on first use", "State the baseline the 2.4x is measured against"]
    assert out["peer"] == [{"audience": "peer", "bullet": "Say what gamma is, or drop it from the slide", "evidence": "gamma"}]
    assert out["meta"]["model"] == "fake-sonnet" and out["meta"]["dropped_without_evidence"] == 0
    assert all(b["audience"] == "novice" for b in out["novice"])


async def test_a_bullet_with_no_evidence_is_dropped_not_shown():
    reply = {**GOOD, "novice": [
        {"bullet": "Define goodput on first use", "evidence": "goodput"},
        {"bullet": "Replace the second bullet with a diagram", "evidence": ""},
        {"bullet": "Cut the acknowledgements section", "evidence": "a phrase nobody wrote"},
    ]}
    out = await recommend(slide(), INTENT, client=Client(reply))
    assert [b["bullet"] for b in out["novice"]] == ["Define goodput on first use"]
    assert out["meta"]["dropped_without_evidence"] == 2


async def test_evidence_must_come_from_that_audiences_report_or_the_slide():
    # 'gamma' is the PEER's unresolved term, not the novice's, and is on the slide text only as 'gamma = 5'
    reply = {**GOOD, "novice": [{"bullet": "Explain the peer's confusion", "evidence": "What is goodput?"},
                                {"bullet": "Say what gamma is on this slide", "evidence": "Draft model acceptance alpha = 0.78 at gamma = 5"},
                                {"bullet": "Quote somebody else's report", "evidence": "Combining the methods lifts goodput."}]}
    out = await recommend(slide(), INTENT, client=Client(reply))
    assert [b["bullet"] for b in out["novice"]] == ["Explain the peer's confusion", "Say what gamma is on this slide"]  # slide text counts; the expert's report does not


async def test_generic_advice_is_dropped_even_when_it_quotes_something():
    reply = {**GOOD, "novice": [{"bullet": "Simplify the language", "evidence": "goodput"},
                                {"bullet": "Make it clearer", "evidence": "goodput"},
                                {"bullet": "Define goodput on first use", "evidence": "goodput"}]}
    out = await recommend(slide(), INTENT, client=Client(reply))
    assert [b["bullet"] for b in out["novice"]] == ["Define goodput on first use"]


async def test_at_most_three_bullets_per_audience_and_no_duplicates():
    many = [{"bullet": f"Define term number {i} on first use", "evidence": "goodput"} for i in range(6)]
    reply = {**GOOD, "novice": [many[0], many[0], *many[1:]]}
    out = await recommend(slide(), INTENT, client=Client(reply))
    assert len(out["novice"]) == diagnose.MAX_PER_AUDIENCE == 3
    assert len({b["bullet"] for b in out["novice"]}) == 3


async def test_the_expert_gets_no_recommendations_even_if_the_model_offers_some():
    reply = {**GOOD, "expert": [{"bullet": "Fix the expert", "evidence": "goodput"}]}
    out = await recommend(slide(), INTENT, client=Client(reply))
    assert set(out) == {"novice", "peer", "expert_flagged", "meta"}


async def test_terms_the_expert_could_not_resolve_are_always_flagged_from_its_own_report():
    s = slide()
    s["readings"]["expert"]["unresolved_terms"] = ["alpha"]
    out = await recommend(s, INTENT, client=Client(GOOD))
    assert out["expert_flagged"] == [{"note": "The expert could not resolve “alpha”.", "evidence": "alpha"}]


async def test_a_contradiction_the_model_finds_needs_a_quote_from_the_expert_report():
    s = slide()
    s["readings"]["expert"]["takeaway"] = "The slide claims 2.4x but never says against what."
    reply = {**GOOD, "expert_flagged": [
        {"note": "The 2.4x has no stated baseline.", "evidence": "never says against what"},
        {"note": "Invented worry.", "evidence": "something the expert never wrote"},
    ]}
    out = await recommend(s, INTENT, client=Client(reply))
    assert out["expert_flagged"] == [{"note": "The 2.4x has no stated baseline.", "evidence": "never says against what"}]
    assert out["meta"]["dropped_without_evidence"] == 1


async def test_the_model_sees_the_reports_and_intent_but_never_confidence_or_the_image():
    c = Client(GOOD)
    await recommend(slide(), SlideIntent("Speculative decoding raises goodput.", "model", "m", None, 1, 0.1, False, {}), client=c)
    (call,) = c.calls
    assert "Speculative decoding raises goodput." in call["user"] and "What is goodput?" in call["user"]
    assert "confidence" not in call["user"].lower() and call["image"] is None
    assert "never advise on the expert" in call["system"]


async def test_one_retry_on_an_unusable_reply_then_the_error_surfaces():
    ok = await recommend(slide(), INTENT, client=Client(GOOD, fail=1))
    assert ok["novice"]
    with pytest.raises(LLMError):
        await recommend(slide(), INTENT, client=Client(GOOD, fail=2))


async def test_a_slide_with_a_failed_persona_has_nothing_to_recommend_from():
    s = slide()
    s["readings"]["peer"] = {"ok": False, "error": {"kind": "invalid_response", "message": "x", "attempts": []}}
    with pytest.raises(RecommendationsUnavailable, match="peer"):
        await recommend(s, INTENT, client=Client(GOOD))


def test_quotes_ignore_case_spacing_and_accept_ellipses():
    src = ["The  presenter combines two techniques to speed something up by 2.4x."]
    assert quoted_in("the presenter COMBINES two techniques", src)
    assert quoted_in("presenter combines ... speed something up", src)
    assert not quoted_in("presenter combines ... slow something down", src)
    assert not quoted_in("", src) and not quoted_in("...", src)
