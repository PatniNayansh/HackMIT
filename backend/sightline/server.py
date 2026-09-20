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
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import diagnose, ingest
from .audiences import DeckProfile, FileCache
from .intent import FileIntentCache, intent_model
from .deck import rollup
from .divergence import Embedder, default_embedder
from .llm import AnthropicClient, LLMClient
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
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def default_client_factory() -> LLMClient | None:
    """None when no credential resolves (the SDK raises TypeError), so the app still starts and
    saved runs still open."""
    try:
        return AnthropicClient()
    except (TypeError, ImportError):
        return None


def default_intent_client_factory() -> LLMClient | None:
    """The small model that rephrases the expert's reading (Haiku by default)."""
    try:
        return AnthropicClient(model=intent_model())
    except (TypeError, ImportError):
        return None


class StartRequest(BaseModel):
    intent: str | None = None  # the presenter's declared intent: optional, stored, not used for alignment
    domain: str
    adjacent_field: str


def create_app(
    *,
    store: RunStore | None = None,
    cache: FileCache | None = None,
    client_factory: Callable[[], LLMClient | None] = default_client_factory,
    intent_client_factory: Callable[[], LLMClient | None] = default_intent_client_factory,
    embedder: Embedder | None = None,
    frontend_dir: Path = FRONTEND_DIR,
) -> FastAPI:
    store = store or RunStore(data_dir() / "history", bundled=[BUNDLED_RUNS_DIR])
    cache = cache or FileCache(data_dir() / "cache" / "audiences", read_only_dirs=[BACKEND / "bundled_cache"])
    intent_cache = FileIntentCache(data_dir() / "cache" / "intents") if cache is None else FileIntentCache(cache.write_dir.parent / "intents")
    active: dict[str, asyncio.Task] = {}
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

    app = FastAPI(title="Sightline", lifespan=lifespan)

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
            "model": client.model if client else None,
            "offline": os.environ.get("SIGHTLINE_OFFLINE") == "1",
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in ingest.supported_extensions():
            raise HTTPException(415, f"unsupported file type {suffix or '(none)'!r}; upload a PDF")
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            size = 0
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file is larger than {MAX_UPLOAD_BYTES // (1 << 20)} MB")
                tmp.write(chunk)
            tmp.flush()
            try:
                slides = await asyncio.to_thread(ingest.parse, Path(tmp.name))
            except ingest.IngestError as e:
                raise HTTPException(422, str(e)) from e

        inferred: DeckProfile | None = None
        error: str | None = None
        client = client_factory()
        if client is None:
            error = "No API key is configured, so the subfield could not be inferred. Enter it yourself."
        else:
            try:
                inferred = await ingest.infer_profile(client, slides)
            except Exception as e:  # noqa: BLE001 - shown to the presenter, who can type it in
                error = f"Could not infer the subfield ({type(e).__name__}). Enter it yourself."

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
        task = asyncio.create_task(run_deck(store, run_id, engine, get_embedder(), intent_client_factory(), intent_cache))
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

    @app.get("/api/runs/{run_id}/slides/{index}/findings")
    def findings(run_id: str, index: int):
        results = store.load_results(run_id)
        result = next((r for r in results if r["index"] == index), None)
        if result is None:
            raise HTTPException(404, f"slide {index} has no result yet")
        return {
            "fixture": diagnose.FIXTURE_BACKED,  # the UI badges these as sample data
            "findings": diagnose.diagnose(result, rollup(results)),
        }

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
