"""THE GATE. Run with `make gate`.

Two hand-written slides with known answers, sent through the real LLM personas and the
real embedding model:

  clear  -> the three audiences converge on the presenter's point
  jargon -> the novice diverges from the expert, and the term gap names the planted jargon

If these do not behave, the premise of the product (that simulated audiences separated
only by prior knowledge produce measurably different readings) is unsound, and nothing
built on top of it means anything. So this file FAILS, loudly, when it cannot run; it
never skips. It always calls the live model (fresh cache) so it is a real test each time.

Thresholds are relative wherever possible. Calibration history (Claude Sonnet 5, effort low, 5 runs,
`make gate-repeat`): clear-slide blind spot ranged -0.28..0.00 and jargon 0.15..0.48, so an
absolute blind-spot magnitude on ONE slide is not stable (two of five runs failed on it) and
was removed. What is stable, and asserted, is the separation between the slides, the sign of
the novice-vs-expert gap, and the checkable outputs (term gap, confidence, unresolved terms).
Every assertion message includes the raw text.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from slides import (
    CLEAR_INTENT,
    CLEAR_PROFILE,
    JARGON_INTENT,
    JARGON_PROFILE,
    PLANTED_JARGON,
    clear_slide,
    jargon_slide,
)

from profe.audiences import AudienceEngine, FileCache
from profe.divergence import default_embedder, normalize_term, score_slide
from profe.llm import OpenAIClient

pytestmark = pytest.mark.live

REPORT_PATH = Path(__file__).parent.parent / ".cache" / "gate_last_run.json"


def _fail_loudly(why: str, e: BaseException) -> None:
    pytest.fail(
        "\n\n=== PROFE GATE COULD NOT RUN ===\n"
        f"{why}: {e!r}\n"
        "The gate is the go/no-go test for the whole project and is never skipped.\n"
        "Put OPENAI_API_KEY=... in the repo-root .env (see .env.example), then re-run `make gate`.\n",
        pytrace=False,
    )


@pytest.fixture(scope="module")
def results():
    try:
        client = OpenAIClient()
    except Exception as e:  # noqa: BLE001
        _fail_loudly("cannot construct the OpenAI client", e)

    with tempfile.TemporaryDirectory() as tmp:
        engine = AudienceEngine(client, FileCache(tmp), offline=False)

        async def run():
            return await asyncio.gather(
                engine.read_slide(clear_slide(), CLEAR_PROFILE),
                engine.read_slide(jargon_slide(), JARGON_PROFILE),
            )

        try:
            clear_readings, jargon_readings = asyncio.run(run())
        except Exception as e:  # noqa: BLE001
            _fail_loudly("live persona call failed", e)

    embedder = default_embedder()
    clear = score_slide(CLEAR_INTENT, {p: r.response for p, r in clear_readings.items()}, embedder, 1)
    jargon = score_slide(JARGON_INTENT, {p: r.response for p, r in jargon_readings.items()}, embedder, 1)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "model": client.model,
                "effort": client.effort,
                "usage": client.usage_log,
                "attempts": {
                    "clear": {p: r.attempts for p, r in clear_readings.items()},
                    "jargon": {p: r.attempts for p, r in jargon_readings.items()},
                },
                "latency_s": {
                    "clear": {p: round(r.latency_s, 2) for p, r in clear_readings.items()},
                    "jargon": {p: round(r.latency_s, 2) for p, r in jargon_readings.items()},
                },
                "clear": {"scores": clear.to_dict(), "responses": {p: r.response.model_dump() for p, r in clear_readings.items()}},
                "jargon": {"scores": jargon.to_dict(), "responses": {p: r.response.model_dump() for p, r in jargon_readings.items()}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return clear, jargon, clear_readings, jargon_readings


def _report(name, sd, readings) -> str:
    lines = [f"--- {name}: divergence={sd.audience_divergence.value:.3f} blind_spot={sd.blind_spot_score.value:.3f}"]
    for p, r in readings.items():
        lines.append(
            f"  {p:6s} align={sd.intent_alignment[p].value:.3f} conf={r.response.confidence:.2f}\n"
            f"         takeaway: {r.response.takeaway}\n"
            f"         unresolved: {r.response.unresolved_terms}"
        )
    return "\n".join(lines)


def _matches(planted: str, term: str) -> bool:
    a, b = normalize_term(planted), normalize_term(term)
    return a in b or b in a


def test_clear_slide_all_three_audiences_converge(results):
    clear, _, clear_r, _ = results
    msg = _report("clear", clear, clear_r)
    assert all(m.value >= 0.5 for m in clear.intent_alignment.values()), msg
    assert len(clear.term_gap.terms) <= 1, msg


def test_jargon_slide_novice_is_lost_and_expert_is_not(results):
    _, jargon, _, jargon_r = results
    msg = _report("jargon", jargon, jargon_r)
    assert jargon.intent_alignment["expert"].value >= 0.5, msg
    assert jargon.intent_alignment["novice"].value < jargon.intent_alignment["expert"].value, msg
    assert jargon_r["novice"].response.confidence < jargon_r["expert"].response.confidence, msg


def test_jargon_slide_diverges_more_than_clear_slide(results):
    clear, jargon, clear_r, jargon_r = results
    msg = _report("clear", clear, clear_r) + "\n" + _report("jargon", jargon, jargon_r)
    assert jargon.audience_divergence.value > clear.audience_divergence.value + 0.05, msg
    assert jargon.blind_spot_score.value > clear.blind_spot_score.value + 0.1, msg


def test_term_gap_recovers_the_planted_jargon(results):
    _, jargon, _, jargon_r = results
    found = [p for p in PLANTED_JARGON if any(_matches(p, t) for t in jargon.term_gap.terms)]
    msg = f"found {len(found)}/{len(PLANTED_JARGON)}: {found}\n" + _report("jargon", jargon, jargon_r)
    assert len(found) >= 4, msg


def test_slide_latency_is_within_the_5_second_budget(results):
    """Design rule: upload -> results in under 5 s per slide. The three personas run in
    parallel, so a slide costs as long as its slowest persona. This is the LLM share only;
    ingest and saliency come on top, so the real margin is thinner than this assertion."""
    _, _, clear_r, jargon_r = results
    slowest = {
        name: max((r.latency_s, p) for p, r in readings.items())
        for name, readings in (("clear", clear_r), ("jargon", jargon_r))
    }
    assert all(t < 5.0 for t, _ in slowest.values()), (
        f"slowest persona per slide (seconds, persona): {slowest}. "
        "A retry after invalid output doubles a persona's time: check the report's `attempts`."
    )


def test_expert_persona_is_not_faking_confusion_about_its_own_field(results):
    _, jargon, _, jargon_r = results
    assert len(jargon_r["expert"].response.unresolved_terms) <= 3, _report("jargon", jargon, jargon_r)
