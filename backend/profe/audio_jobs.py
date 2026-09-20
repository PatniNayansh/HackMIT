"""The audio half of a review: chunk it, ship it to the GX10, watch it run.

Deliberately separate from the deck pipeline, because it must never be able to take the deck
down with it. Slides are the product; audio is optional, runs on a machine that may be asleep
or on another network, and takes minutes on a GPU. Every failure in here is recorded as a state
on the job and returned to the page -- nothing raises into the request that started it, and
nothing here touches the run's own status.

State lives in one small file per run, beside the audio, so a `make dev` reload does not lose
a job that is still running on the GX10.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import gx10
from .audio import AudioError, duration_seconds, manifest, split, write_manifest

# Local phases, in order. The remote ones ("running", "complete", "failed") come from the
# status file the runner writes on the GX10 and are merged in on read.
CHUNKING = "chunking"
UPLOADING = "uploading"
QUEUED = "queued"
RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"
UNAVAILABLE = "unavailable"


@dataclass
class AudioJob:
    run_id: str
    source_name: str
    state: str = CHUNKING
    duration_s: float = 0.0
    chunks_total: int = 0
    chunks_sent: int = 0
    chunks_done: int = 0
    error: str | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["blocking"] = self.state in (CHUNKING, UPLOADING)
        return d


def job_root(data_dir: Path, run_id: str) -> Path:
    return data_dir / "audio" / run_id


def _state_path(root: Path) -> Path:
    return root / "state.json"


def save(root: Path, job: AudioJob) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _state_path(root).write_text(json.dumps(job.as_dict(), indent=2), encoding="utf-8")


def load(root: Path) -> AudioJob | None:
    p = _state_path(root)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    d.pop("blocking", None)
    known = {f for f in AudioJob.__dataclass_fields__}
    return AudioJob(**{k: v for k, v in d.items() if k in known})


async def prepare_and_launch(root: Path, run_id: str, source: Path) -> AudioJob:
    """Chunk, ship, start. Runs in a background task; every failure lands on the job rather
    than on the request. The GX10 being unreachable is `unavailable`, not `failed`: nothing
    went wrong, the machine simply is not there, and the deck is unaffected either way."""
    job = AudioJob(run_id=run_id, source_name=source.name, state=CHUNKING, detail="Splitting the recording")
    save(root, job)

    try:
        job.duration_s = await asyncio.to_thread(duration_seconds, source)
        chunks = await asyncio.to_thread(split, source, root / "chunks")
        job.chunks_total = len(chunks)
        job.state, job.detail = UPLOADING, f"0 of {len(chunks)} chunks"
        save(root, job)

        mpath = write_manifest(root / "chunks", manifest(chunks, source_name=source.name))

        def ship() -> None:
            cfg = gx10.config()
            remote = gx10.job_dir(run_id)
            gx10.ssh(f"rm -rf {remote} && mkdir -p {remote}/chunks {remote}/out", cfg=cfg)
            for i, c in enumerate(chunks, 1):
                gx10.scp_to(c.path, f"{remote}/chunks/", cfg=cfg)
                job.chunks_sent = i
                job.detail = f"{i} of {len(chunks)} chunks"
                save(root, job)
            gx10.scp_to(mpath, f"{remote}/manifest.json", cfg=cfg)

        await asyncio.to_thread(ship)

        job.state, job.detail = QUEUED, "Starting on the GX10"
        save(root, job)
        await asyncio.to_thread(gx10.launch, run_id)

        job.state, job.detail = RUNNING, f"0 of {job.chunks_total} chunks"
        save(root, job)

    except gx10.Gx10Unavailable as e:
        job.state, job.error = UNAVAILABLE, str(e)
        job.detail = "The GX10 is not reachable; the deck is unaffected."
        save(root, job)
    except (AudioError, gx10.Gx10Error) as e:
        job.state, job.error, job.detail = FAILED, str(e), "The audio could not be processed."
        save(root, job)
    except Exception as e:  # noqa: BLE001 - an optional extra must never escape into the run
        job.state, job.error = FAILED, f"{type(e).__name__}: {e}"
        job.detail = "The audio could not be processed."
        save(root, job)
    return job


async def refresh(root: Path, job: AudioJob) -> AudioJob:
    """Merge in what the GX10 says. Only meaningful once the job is actually over there."""
    if job.state not in (QUEUED, RUNNING):
        return job
    try:
        remote = await asyncio.to_thread(gx10.status, job.run_id)
    except (gx10.Gx10Unavailable, gx10.Gx10Error) as e:
        # A blip in the network is not a failed run: the job is still on the GX10, and the next
        # poll will find it. Say so rather than declaring the whole thing dead.
        job.detail = f"Lost contact with the GX10 ({e}); still polling."
        return job

    state = remote.get("state")
    if state in ("running", "starting", "loading_model"):
        job.state = RUNNING
        job.chunks_done = int(remote.get("done") or 0)
        job.chunks_total = int(remote.get("total") or job.chunks_total)
        job.detail = ("Loading the model" if state == "loading_model"
                      else f"{job.chunks_done} of {job.chunks_total} chunks")
    elif state == "complete":
        job.state, job.chunks_done = COMPLETE, int(remote.get("done") or job.chunks_total)
        job.detail = f"{job.chunks_done} chunks"
    elif state == "failed":
        job.state, job.error = FAILED, remote.get("error") or "the run failed on the GX10"
        job.detail = "The run failed on the GX10."
    save(root, job)
    return job


async def collect(root: Path, job: AudioJob, dest: Path) -> int:
    """Pull finished chunk output back. Safe to call more than once."""
    if job.state != COMPLETE:
        return 0
    names = await asyncio.to_thread(gx10.collect, job.run_id, dest)
    return len(names)


def discard(root: Path) -> None:
    """Forget a job entirely, chunks and all. Used when the presenter removes the audio."""
    shutil.rmtree(root, ignore_errors=True)
