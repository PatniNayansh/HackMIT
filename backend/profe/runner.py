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

A slide's intent is the EXPERT's takeaway, verbatim (see `deck.expert_takeaway_intent`). The
expert is the reference, so its own standing is definitional.

Two comparators, chosen by the `comparator` argument (the server's COMPARATOR flag) and recorded on
the run:
  * "cosine":    step 1's whole-takeaway cosine against the expert's takeaway (`deck.build_metrics`).
                 Nothing else is called: the batch is the three persona calls per slide.
  * "fieldwise": `compare.py`. After the three personas, ONE cheap structuring call turns the three
                 takeaways into fields and proposition coverage; everything after that is pure code.

A run stops early only when a slide fails for reasons that are not the model's output (no key,
API down, nothing cached while offline), since every later slide would fail the same way.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

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
from .compare import FileStructureCache, StructuringError, build_fieldwise_metrics, structure_slide
from .deck import (
    TITLE_SLIDE_INDEX,
    SlideResult,
    build_metrics,
    error_record,
    expert_takeaway_intent,
    plain,
    reading_record,
    title_slide_result,
)
from .divergence import Embedder
from .llm import LLMClient, LLMError
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
    personas_s: float,
    *,
    comparator: str = "fieldwise",
    structuring_client: LLMClient | None = None,
    structure_cache: FileStructureCache | None = None,
    image_content: dict[str, Any] | None = None,
) -> SlideResult:
    readings = {
        p: reading_record(o) if isinstance(o, AudienceReading) else error_record(o)
        for p, o in outcomes.items()
    }
    expert = outcomes["expert"]
    slide_intent = expert_takeaway_intent(expert.response) if isinstance(expert, AudienceReading) else None

    failed = [p for p, o in outcomes.items() if not isinstance(o, AudienceReading)]
    metrics, metrics_error, scored_by = None, None, None
    timing = {"personas_s": round(personas_s, 3)}
    if failed:
        metrics_error = f"not computed: no usable reading from {', '.join(failed)}"
        if "expert" in failed:
            metrics_error += " (the expert\u2019s reading defines the intended reading)"
    elif comparator == "cosine":
        responses = {p: o.response for p, o in outcomes.items() if isinstance(o, AudienceReading)}
        # The embedding model is local and CPU-bound; keep it off the event loop so polling stays live.
        metrics = await asyncio.to_thread(build_metrics, slide_intent["text"], responses, embedder, index)
        scored_by = _embedder_name(embedder)
    else:
        takeaways = {p: o.response.takeaway for p, o in outcomes.items() if isinstance(o, AudienceReading)}
        unresolved = {p: o.response.unresolved_terms for p, o in outcomes.items() if isinstance(o, AudienceReading)}
        structured = structure_cache.get(takeaways) if structure_cache else None
        if structured is None and structuring_client is not None:
            t0 = time.perf_counter()
            try:
                structured = await structure_slide(structuring_client, takeaways, embedder=embedder)
                if structure_cache:
                    structure_cache.put(takeaways, structured)
            except (StructuringError, LLMError) as e:
                metrics_error = f"not computed: the field-wise structuring call failed ({e})"
            timing["structuring_s"] = round(time.perf_counter() - t0, 3)
        elif structured is None:
            metrics_error = "not computed: no model client is available for the field-wise comparison and nothing was cached"
        if structured is not None:
            metrics = build_fieldwise_metrics(takeaways, structured, unresolved, (image_content or {}).get("text"))
            scored_by = structured["meta"]["model"]
    return {
        "index": index,
        "text": text,
        "readings": readings,
        "slide_intent": slide_intent,
        "metrics": metrics,
        "metrics_error": metrics_error,
        "scored_by": scored_by,
        "timing": timing,
        "image_content": image_content,
    }


async def run_deck(
    store: RunStore,
    run_id: str,
    engine: TolerantEngine,
    embedder: Embedder,
    *,
    comparator: str = "fieldwise",
    structuring_client: LLMClient | None = None,
    structure_cache: FileStructureCache | None = None,
    skip_title_slide: bool = True,
) -> None:
    """Run every slide of a stored deck, in order. Never raises for a failed run: the outcome is
    in the run's metadata. Cancellation is recorded and re-raised.

    `skip_title_slide` is the product default: a deck's first slide is its title card, so no
    persona reads it and no model call is made for it. Tests that are about the runner's own
    mechanics rather than about decks turn it off, because a two-slide fixture has no title."""
    meta = store.load_meta(run_id)
    profile = DeckProfile(meta["profile"]["domain"], meta["profile"]["adjacent_field"])
    store.update_meta(
        run_id,
        status="running",
        started_at=now_iso(),
        comparator=comparator,
        structuring_model=getattr(structuring_client, "model", None) if comparator == "fieldwise" else None,
        model=engine.client.model if engine.client else None,
        embedding_model=_embedder_name(embedder),
        error=None,
    )
    models: set[str] = set()
    pending: asyncio.Task | None = None  # the previous slide's build: its structuring call overlaps the next slide's persona calls

    async def land(task: asyncio.Task) -> None:
        store.save_result(run_id, await task)

    try:
        memory: dict[Persona, Memory] = {p: () for p in PERSONAS}
        for slide in store.load_slides(run_id):
            if skip_title_slide and slide.index == TITLE_SLIDE_INDEX:
                # Saved so the slide keeps its place and its number, and so the UI can show it;
                # no persona sees it, so this costs nothing and the deck's stats never see it.
                if pending is not None:
                    await land(pending)
                    pending = None
                store.save_result(run_id, plain(title_slide_result(slide)))
                continue
            t0 = time.perf_counter()
            outcomes = await engine.read_slide_tolerant(slide.to_input(), profile, memory)
            personas_s = time.perf_counter() - t0

            errors = [error_record(o)["error"] for o in outcomes.values() if isinstance(o, Exception)]
            if len(errors) == len(PERSONAS) and all(e["kind"] in _INFRASTRUCTURE_ERRORS for e in errors):
                raise RuntimeError(f"slide {slide.index}: {errors[0]['message']}")

            # Slide k's results are saved (in order) once slide k+1's persona calls are done.
            if pending is not None:
                await land(pending)
            pending = asyncio.ensure_future(
                _build(
                    slide.index, slide.to_input().text, outcomes, embedder, personas_s,
                    comparator=comparator, structuring_client=structuring_client,
                    structure_cache=structure_cache, image_content=slide.image_content,
                )
            )
            for p, o in outcomes.items():
                if isinstance(o, AudienceReading):
                    models.add(o.model)
                    memory[p] = (*memory[p], (slide.index, o.response.takeaway))[-MAX_MEMORY:]
        if pending is not None:
            await land(pending)
            pending = None
    except asyncio.CancelledError:
        if pending is not None:
            pending.cancel()
        store.update_meta(run_id, status="failed", error="interrupted: the server stopped mid-run", finished_at=now_iso())
        raise
    except Exception as e:  # noqa: BLE001 - recorded on the run for the UI to show
        if pending is not None:
            pending.cancel()
        store.update_meta(run_id, status="failed", error=str(e) or type(e).__name__, finished_at=now_iso())
        return
    store.update_meta(
        run_id,
        status="complete",
        finished_at=now_iso(),
        model=", ".join(sorted(models)) or meta.get("model"),
    )
