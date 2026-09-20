"""Runs a stored deck through the three audiences and writes each slide's result as it lands.

This is the glue between the step 1 engine and the store. It adds no analysis of its own.

Two behaviours differ from `AudienceEngine.iter_deck`, both because the UI must show a bad
model response rather than hide it:

  * One persona failing does not discard the other two. `read_slide` gathers all three and
    raises on the first error, so a single invalid reply would lose the slide. Here each
    persona's outcome is kept: a reading, or the reason there is none.
  * A slide with any failed persona is stored WITHOUT metrics. A confidence outside [0, 1] is
    an error to display, never a value to clamp, and a comparison that is missing a side is
    not computed.

After the three personas finish for a slide, a cheap model call (see `intent`) rephrases the
EXPERT's reading into that slide's intended reading, and the other two are measured against it.
The expert is the reference, so its own alignment is definitional (see `deck.build_metrics`).

A run stops early only when a slide fails for reasons that are not the model's output (no key,
API down, nothing cached while offline), since every later slide would fail the same way.
"""

from __future__ import annotations

import asyncio
import time
from typing import Mapping

from .audiences import (
    MAX_MEMORY,
    PERSONAS,
    AudienceEngine,
    AudienceReading,
    DeckProfile,
    Memory,
    Persona,
    SlideInput,
    slide_hash,
)
from .deck import SlideResult, build_metrics, error_record, reading_record
from .divergence import Embedder
from .intent import IntentCache, infer_slide_intent
from .llm import LLMClient
from .store import RunStore, now_iso

# Failures that say nothing about the model's reading of a slide.
_INFRASTRUCTURE_ERRORS = ("call_failed", "offline_cache_miss")


class TolerantEngine(AudienceEngine):
    async def read_slide_tolerant(
        self, slide: SlideInput, profile: DeckProfile, memory: Mapping[Persona, Memory]
    ) -> dict[Persona, AudienceReading | Exception]:
        sh = slide_hash(slide, profile)
        outcomes = await asyncio.gather(
            *(self._read_one(p, slide, profile, sh, memory.get(p, ())) for p in PERSONAS),
            return_exceptions=True,
        )
        for o in outcomes:
            if isinstance(o, BaseException) and not isinstance(o, Exception):
                raise o  # cancellation is not a persona failure
        return dict(zip(PERSONAS, outcomes))


def _embedder_name(embedder: Embedder) -> str:
    return getattr(embedder, "model_name", type(embedder).__name__)


async def _build(
    index: int,
    text: str,
    outcomes: Mapping[Persona, AudienceReading | Exception],
    embedder: Embedder,
    intent_client: LLMClient | None,
    intent_cache: IntentCache | None,
    personas_s: float,
) -> SlideResult:
    readings = {
        p: reading_record(o) if isinstance(o, AudienceReading) else error_record(o)
        for p, o in outcomes.items()
    }
    expert = outcomes["expert"]
    slide_intent = None
    intent_s = 0.0
    if isinstance(expert, AudienceReading):
        t0 = time.perf_counter()
        slide_intent = (await infer_slide_intent(expert.response, text, intent_client, cache=intent_cache)).to_dict()
        intent_s = time.perf_counter() - t0

    failed = [p for p, o in outcomes.items() if not isinstance(o, AudienceReading)]
    metrics, metrics_error, scored_by = None, None, None
    if failed:
        metrics_error = f"not computed: no usable reading from {', '.join(failed)}"
        if "expert" in failed:
            metrics_error += " (the expert\u2019s reading defines the intended reading)"
    else:
        responses = {p: o.response for p, o in outcomes.items() if isinstance(o, AudienceReading)}
        # The embedding model is local and CPU-bound; keep it off the event loop so polling stays live.
        metrics = await asyncio.to_thread(build_metrics, slide_intent["text"], responses, embedder, index)
        scored_by = _embedder_name(embedder)
    return {
        "index": index,
        "text": text,
        "readings": readings,
        "slide_intent": slide_intent,
        "metrics": metrics,
        "metrics_error": metrics_error,
        "scored_by": scored_by,
        "timing": {"personas_s": round(personas_s, 3), "intent_s": round(intent_s, 3)},
    }


async def run_deck(
    store: RunStore,
    run_id: str,
    engine: TolerantEngine,
    embedder: Embedder,
    intent_client: LLMClient | None = None,
    intent_cache: IntentCache | None = None,
) -> None:
    """Run every slide of a stored deck, in order. Never raises for a failed run: the outcome is
    in the run's metadata. Cancellation is recorded and re-raised."""
    meta = store.load_meta(run_id)
    profile = DeckProfile(meta["profile"]["domain"], meta["profile"]["adjacent_field"])
    store.update_meta(
        run_id,
        status="running",
        started_at=now_iso(),
        model=engine.client.model if engine.client else None,
        embedding_model=_embedder_name(embedder),
        error=None,
    )
    models: set[str] = set()
    intent_sources: set[str] = set()
    try:
        memory: dict[Persona, Memory] = {p: () for p in PERSONAS}
        for slide in store.load_slides(run_id):
            t0 = time.perf_counter()
            outcomes = await engine.read_slide_tolerant(slide.to_input(), profile, memory)
            personas_s = time.perf_counter() - t0

            errors = [error_record(o)["error"] for o in outcomes.values() if isinstance(o, Exception)]
            if len(errors) == len(PERSONAS) and all(e["kind"] in _INFRASTRUCTURE_ERRORS for e in errors):
                raise RuntimeError(f"slide {slide.index}: {errors[0]['message']}")

            result = await _build(slide.index, slide.text, outcomes, embedder, intent_client, intent_cache, personas_s)
            store.save_result(run_id, result)
            if result["slide_intent"]:
                si = result["slide_intent"]
                intent_sources.add(si["model"] if si["source"] == "model" else "template (expert claim)")
            for p, o in outcomes.items():
                if isinstance(o, AudienceReading):
                    models.add(o.model)
                    memory[p] = (*memory[p], (slide.index, o.response.takeaway))[-MAX_MEMORY:]
    except asyncio.CancelledError:
        store.update_meta(run_id, status="failed", error="interrupted: the server stopped mid-run", finished_at=now_iso())
        raise
    except Exception as e:  # noqa: BLE001 - recorded on the run for the UI to show
        store.update_meta(run_id, status="failed", error=str(e) or type(e).__name__, finished_at=now_iso())
        return
    store.update_meta(
        run_id,
        status="complete",
        finished_at=now_iso(),
        model=", ".join(sorted(models)) or meta.get("model"),
        intent_model=", ".join(sorted(intent_sources)) or None,
    )
