"""The thin HTTP layer: upload, start a run, poll it, replay history.

Streaming is polling. A run writes each slide's result to disk as it completes (see `store`);
`GET /api/runs/<id>?since=<n>` returns whatever has landed after slide n plus a fresh deck
rollup, and the page asks again every second or so. There is no in-memory run state to lose,
so a reload mid-run, opening a saved run and watching a live one are the same code path.

Nothing under `GET /api/runs...` calls a model, so a saved run replays with no network and no
API key. Only `POST /api/upload` (subfield inference) and `POST /api/runs/<id>/start` can, and
both degrade to a clear message when there is no key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

# Marks this process as the FastAPI server, so profe.neural.TribeNeural (6-13 GPU-min
# per slide) refuses to run here even if something imports and calls it by mistake.
os.environ["PROFE_PROCESS"] = "server"

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ingest
from .audiences import DeckProfile, FileCache
from .compare import FileStructureCache
from .diagnose import RecommendationsUnavailable, recommend
from .deck import rollup
from .divergence import Embedder, default_embedder
from .llm import LLMClient, OpenAIClient
from .neural import SURFACE_VIEWS, CachedNeural, NeuralNotCached
from .runner import TolerantEngine, run_deck
from .store import (
    BACKEND,
    BUNDLED_RUNS_DIR,
    REPO_ROOT,
    ReadOnlyRun,
    RunNotFound,
    SCHEMA_VERSION,
    RunStore,
    data_dir,
)

FRONTEND_DIR = REPO_ROOT / "frontend"

# THE FLAG. Which comparator produces a run's numbers: "fieldwise" (compare.py) or "cosine" (the
# original whole-takeaway cosine, divergence.py, left intact). Flip it with no code change:
#     PROFE_COMPARATOR=cosine make dev
# It applies to runs STARTED after it changes; a saved run keeps and shows the comparator that made it.
COMPARATORS = ("cosine", "fieldwise")


def default_comparator() -> str:
    value = os.environ.get("PROFE_COMPARATOR", "fieldwise").strip().lower()
    if value not in COMPARATORS:
        raise ValueError(f"PROFE_COMPARATOR must be one of {COMPARATORS}, got {value!r}")
    return value


COMPARATOR = default_comparator()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def default_client_factory() -> LLMClient | None:
    """None when no credential resolves (the client raises TypeError), so the app still starts and
    saved runs still open."""
    try:
        return OpenAIClient()
    except (TypeError, ImportError):
        return None


def default_helper_client_factory() -> LLMClient | None:
    """The cost-tier model for the figure describer (`helper` in llm.CONFIG): describing marks and
    labels is extraction, not judgement."""
    try:
        return OpenAIClient(role="helper")
    except (TypeError, ImportError):
        return None


def default_structuring_client_factory() -> LLMClient | None:
    """The field-wise structuring + proposition-coverage call (`structuring` in llm.CONFIG). On
    Claude the cost tier misjudged the direction of a paraphrase on hand-written fixtures and the
    balanced tier did not, and a false gap is exactly what this comparator exists to avoid; the
    role is measured again on the current models (see the README). Override the model with
    PROFE_STRUCTURING_MODEL."""
    try:
        return OpenAIClient(role="structuring")
    except (TypeError, ImportError):
        return None


class StartRequest(BaseModel):
    intent: str | None = None  # the presenter's declared intent: optional, stored, never used for alignment
    domain: str
    adjacent_field: str


def create_app(
    *,
    store: RunStore | None = None,
    cache: FileCache | None = None,
    client_factory: Callable[[], LLMClient | None] = default_client_factory,
    helper_client_factory: Callable[[], LLMClient | None] = default_helper_client_factory,
    structuring_client_factory: Callable[[], LLMClient | None] = default_structuring_client_factory,
    comparator: str | None = None,
    skip_title_slide: bool = True,
    embedder: Embedder | None = None,
    neural_cache: CachedNeural | None = None,
    frontend_dir: Path = FRONTEND_DIR,
) -> FastAPI:
    store = store or RunStore(data_dir() / "history", bundled=[BUNDLED_RUNS_DIR])
    cache = cache or FileCache(data_dir() / "cache" / "audiences", read_only_dirs=[BACKEND / "bundled_cache"])
    comparator = comparator or COMPARATOR
    if comparator not in COMPARATORS:
        raise ValueError(f"comparator must be one of {COMPARATORS}, got {comparator!r}")
    figure_cache = ingest.FileFigureCache(cache.write_dir.parent / "figures")
    structure_cache = FileStructureCache(cache.write_dir.parent / "structured")
    # Precomputed only, and only ever for the bundled sample decks (spec 5, 9): there is no
    # writable neural store, unlike run history or the audience cache.
    neural_cache = neural_cache or CachedNeural(BACKEND / "fixtures" / "neural")
    active: dict[str, asyncio.Task] = {}
    in_flight: dict[tuple[str, int], asyncio.Future] = {}
    embedder_ref: dict[str, Embedder] = {}

    def get_embedder() -> Embedder:
        if "e" not in embedder_ref:
            embedder_ref["e"] = embedder or default_embedder()
        return embedder_ref["e"]

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        # Load the local embedding model in the background so the first run does not pay for it.
        warm = asyncio.create_task(asyncio.to_thread(lambda: get_embedder().embed(["warm-up"])))
        warm.add_done_callback(lambda t: t.exception())  # a missing model surfaces on the first run
        yield
        for task in list(active.values()):
            task.cancel()
        await asyncio.gather(*active.values(), return_exceptions=True)

    app = FastAPI(title="ProFe", lifespan=lifespan)

    @app.exception_handler(RunNotFound)
    async def _not_found(_, exc: RunNotFound):
        return JSONResponse({"detail": f"run not found: {exc}"}, status_code=404)

    @app.exception_handler(ReadOnlyRun)
    async def _read_only(_, exc: ReadOnlyRun):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    def public_meta(meta: dict[str, Any]) -> dict[str, Any]:
        # A run that says "running" with no live task behind it was cut off by a restart.
        meta = {**meta, "legacy": meta.get("schema_version", 1) < SCHEMA_VERSION}
        if meta["status"] == "running" and meta["run_id"] not in active:
            return {**meta, "status": "interrupted"}
        return meta

    def with_image(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {**result, "image_url": f"/api/runs/{run_id}/slides/{result['index']}.png"}

    # ---------------------------------------------------------------------------- API

    @app.get("/api/health")
    def health():
        client = client_factory()
        return {
            "can_call_model": client is not None,
            "comparator": comparator,
            "model": client.model if client else None,
            "offline": os.environ.get("PROFE_OFFLINE") == "1",
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in ingest.supported_extensions():
            raise HTTPException(415, f"unsupported file type {suffix or '(none)'!r}; upload a PDF")
        # mkstemp + close-before-reopen, not NamedTemporaryFile: on Windows a file opened
        # for writing is locked against being opened again by path (pymupdf.open below)
        # until the write handle is closed, which NamedTemporaryFile's context manager
        # only does at the very end of the `with` block.
        fd, tmp_name = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as tmp:
                size = 0
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1 << 20)} MB")
                    tmp.write(chunk)
            try:
                slides = await asyncio.to_thread(ingest.parse, Path(tmp_name))
            except ingest.IngestError as e:
                raise HTTPException(422, str(e)) from e
        finally:
            try:
                os.unlink(tmp_name)
            except PermissionError:
                pass  # pymupdf can keep its own handle open after a failed parse on
                # Windows; a leaked temp file the OS reaps later beats crashing the request

        inferred: DeckProfile | None = None
        error: str | None = None
        client = client_factory()

        async def infer() -> None:
            nonlocal inferred, error
            if client is None:
                error = "No API key is configured, so the subfield could not be inferred. Enter it yourself."
                return
            try:
                inferred = await ingest.infer_profile(client, slides)
            except Exception as e:  # noqa: BLE001 - shown to the presenter, who can type it in
                error = f"Could not infer the subfield ({type(e).__name__}). Enter it yourself."

        # Figure descriptions are made ONCE per deck, here, in parallel with the subfield: a
        # text-only deck sends nothing, and the personas never make this call themselves.
        described: list[ingest.Slide] = slides
        async def describe() -> None:
            nonlocal described
            described = await ingest.describe_figures(helper_client_factory(), slides, figure_cache)

        await asyncio.gather(infer(), describe())
        slides = described

        meta = store.create_draft(
            title=Path(name).stem or "Untitled deck",
            source_filename=name,
            slides=slides,
            inferred=inferred,
            inference_error=error,
        )
        return {**public_meta(meta), "can_call_model": client is not None}

    @app.post("/api/runs/{run_id}/start", status_code=202)
    async def start(run_id: str, body: StartRequest):
        meta = store.load_meta(run_id)  # 404 if unknown
        if store.is_read_only(run_id):
            raise ReadOnlyRun(f"{run_id} is a bundled sample and cannot be run again")
        intent, domain, adjacent = (body.intent or "").strip() or None, body.domain.strip(), body.adjacent_field.strip()
        if not domain or not adjacent:
            raise HTTPException(422, "Both the deck's subfield and the adjacent field are required.")
        if run_id in active:
            raise HTTPException(409, "This run is already in progress.")
        if meta["status"] not in ("draft", "failed", "running"):  # "running" here = an orphan
            raise HTTPException(409, f"A {meta['status']} run cannot be started again.")

        inferred = meta.get("profile_inferred")
        edited = not inferred or (inferred["domain"], inferred["adjacent_field"]) != (domain, adjacent)
        if meta["status"] != "draft":
            store.clear_results(run_id)
        store.update_meta(
            run_id,
            intent=intent,
            profile={"domain": domain, "adjacent_field": adjacent, "confirmed": True, "edited": edited},
            status="running",
            error=None,
            started_at=None,
            finished_at=None,
        )
        engine = TolerantEngine(client_factory(), cache)
        task = asyncio.create_task(
            run_deck(
                store, run_id, engine, get_embedder(), comparator=comparator,
                structuring_client=structuring_client_factory() if comparator == "fieldwise" else None,
                structure_cache=structure_cache, skip_title_slide=skip_title_slide,
            )
        )
        active[run_id] = task
        task.add_done_callback(lambda _: active.pop(run_id, None))
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/runs")
    def list_runs():
        return [
            {k: m.get(k) for k in ("run_id", "title", "created_at", "slide_count", "model", "status", "sample")}
            | {"status": public_meta(m)["status"]}
            for m in store.list_runs()
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, since: int = 0):
        meta = public_meta(store.load_meta(run_id))
        results = store.load_results(run_id)
        done = {r["index"] for r in results}
        return {
            "meta": meta,
            "results": [with_image(run_id, r) for r in results if r["index"] > since],
            "rollup": rollup(results),
            "pending": [i for i in range(1, meta["slide_count"] + 1) if i not in done],
            "image_urls": [f"/api/runs/{run_id}/slides/{i}.png" for i in range(1, meta["slide_count"] + 1)],
        }

    @app.get("/api/runs/{run_id}/slides/{index}.png")
    def slide_image(run_id: str, index: int):
        return FileResponse(
            store.image_path(run_id, index), media_type="image/png", headers={"Cache-Control": "max-age=3600"}
        )

    @app.get("/api/runs/{run_id}/slides/{index}/recommendations")
    async def recommendations(run_id: str, index: int):
        """Recommendations for one slide, made the first time the slide is opened and saved with
        the run. A saved run replays them from disk with no key. Concurrent opens share one call."""
        cached = store.load_recs(run_id, index)
        if cached is not None:
            return {"available": True, "cached": True, "recommendations": cached}
        result = next((r for r in store.load_results(run_id) if r["index"] == index), None)
        if result is None:
            raise HTTPException(404, f"slide {index} has no result yet")
        if not result.get("slide_intent"):
            return {"available": False, "reason": "This slide has no inferred intent to work from."}
        client = client_factory()
        if client is None:
            return {"available": False, "reason": "No API key is configured, and no recommendations were saved for this slide."}

        key = (run_id, index)
        if key not in in_flight:
            async def generate():
                try:
                    recs = await recommend(result, result["slide_intent"], client=client)
                    try:
                        store.save_recs(run_id, index, recs)
                    except ReadOnlyRun:
                        pass  # a bundled sample: show it, do not write into the repo
                    return recs
                finally:
                    in_flight.pop(key, None)

            in_flight[key] = asyncio.ensure_future(generate())
        try:
            recs = await asyncio.shield(in_flight[key])
        except RecommendationsUnavailable as e:
            return {"available": False, "reason": str(e).capitalize() + "."}
        except Exception as e:  # noqa: BLE001 - shown next to a retry button
            raise HTTPException(502, f"Could not generate recommendations ({type(e).__name__}).") from e
        return {"available": True, "cached": False, "recommendations": recs}

    # ------------------------------------------------------------------------- neural
    # Precomputed only (spec 5, 9): these never trigger a model call, only ever read what
    # scripts/precompute_neural/run.py already wrote under backend/fixtures/neural/. A run
    # with nothing cached there (every run right now, until that CLI is actually run on a
    # CUDA machine) 404s per slide and reports zero scored slides at the deck level -- the
    # UI's job is to show a clear empty state for that, never a placeholder brain.

    @app.get("/api/runs/{run_id}/neural")
    def neural_rollup(run_id: str):
        meta = store.load_meta(run_id)  # 404 if the run itself is unknown
        return neural_cache.deck_rollup(run_id, list(range(1, meta["slide_count"] + 1)))

    @app.get("/api/runs/{run_id}/slides/{index}/neural")
    def neural_slide(run_id: str, index: int):
        store.load_meta(run_id)  # 404 if the run itself is unknown
        if not neural_cache.has(run_id, index):
            raise HTTPException(404, f"no precomputed neural result for slide {index}")
        m = neural_cache.metrics(run_id, index)
        return {
            "slide": index,
            "language_drive": m.language_drive,
            "visual_drive": m.visual_drive,
            "processing_ratio": m.processing_ratio,
            "narration_transcript": m.narration_transcript,
            # gfp_negative_baseline is deliberately withheld here: design rule 3 forbids
            # showing it as a finding. It belongs only in the Methods panel (not built
            # yet), labeled next to the null result it reproduces -- never on its own.
            "views": {v: f"/api/runs/{run_id}/slides/{index}/neural/{v}.png" for v in SURFACE_VIEWS},
        }

    @app.get("/api/runs/{run_id}/slides/{index}/neural/{view}.png")
    def neural_image(run_id: str, index: int, view: str):
        try:
            path = neural_cache.image_path(run_id, index, view)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except NeuralNotCached as e:
            raise HTTPException(404, str(e)) from e
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    # ------------------------------------------------------------------------- lecture
    # One real recorded lecture, run through TRIBE with no synthesis (see
    # scripts/make_lecture_timecourse.py). Static: read once from the fixture, never computed.

    LECTURE = BACKEND / "fixtures" / "lecture_timecourse.json"

    LECTURE_SURFACES = BACKEND / "fixtures" / "lecture_surfaces"

    @app.get("/api/lecture")
    def lecture():
        if not LECTURE.is_file():
            raise HTTPException(404, "no lecture timecourse has been generated")
        body = json.loads(LECTURE.read_text(encoding="utf-8"))
        for v in body.get("surfaces", []):
            v["url"] = f"/api/lecture/surfaces/{v['chunk']}.png"
        return body

    @app.get("/api/lecture/surfaces/{chunk}.png")
    def lecture_surface(chunk: str):
        path = (LECTURE_SURFACES / f"{chunk}.png").resolve()
        if not path.is_file() or LECTURE_SURFACES.resolve() not in path.parents:
            raise HTTPException(404, f"no cortical render for {chunk}")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    # -------------------------------------------------------------------------- pages

    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

        @app.middleware("http")
        async def _revalidate_static(request, call_next):
            # No build step means no hashed filenames, so make the browser recheck each time.
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
