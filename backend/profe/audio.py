"""Cutting lecture audio into chunks small enough for TRIBE to finish.

This is a cost limit, not a size limit. TRIBE v2's text encoder builds *contextualised* word
embeddings: every word is re-encoded against the whole transcript before it, through a 3B-parameter
Llama. Cost therefore grows with total document length, not with word count. A single slide of
3-25 words finishes in seconds; a 25-minute lecture (~12,300 word events) was still grinding
through the embedding stage after 30 minutes with a 30-hour ETA before it was killed.

Two-minute chunks each start their own short context, which restores the fast per-item rate.
No overlap: each chunk starts cold regardless, so overlapping would pay twice for the same
seconds and buy nothing. Not silence-based: lecture pauses are short and irregular, so chunk
lengths -- and with them the cost bound that is the entire point -- stop being predictable.

The cost is real and visible in the output: a sentence spanning a boundary is encoded with no
lead-in, which reads as a brief dip in the first seconds of each chunk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

CHUNK_SECONDS = 120

# What a browser will hand us from a lecture recording. ffmpeg copies the stream rather than
# re-encoding, so the chunk format always matches the source.
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".oga", ".flac", ".webm", ".mp4", ".mpeg", ".mpga"}


class AudioError(RuntimeError):
    """Something went wrong preparing the audio. The message is shown to the presenter."""


@dataclass(frozen=True)
class Chunk:
    index: int
    path: Path
    start_s: float

    @property
    def name(self) -> str:
        return f"chunk_{self.index:02d}"


def ffmpeg_exe() -> str:
    """The bundled binary if it is installed, else one on PATH. imageio-ffmpeg ships a static
    build per platform, which is how this works on a laptop with no system ffmpeg -- the same
    trick the GX10 uses to get ffmpeg without a sudo password."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any import or lookup failure means "fall back"
        found = shutil.which("ffmpeg")
        if not found:
            raise AudioError(
                "No ffmpeg available to split the audio. Install the backend's dependencies "
                "(`make setup`), which include imageio-ffmpeg."
            ) from None
        return found


def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=600)
    except FileNotFoundError as e:
        raise AudioError(f"could not run ffmpeg: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioError("ffmpeg took too long and was stopped") from e


def duration_seconds(path: Path) -> float:
    """Length in seconds, read from the container. 0.0 when ffmpeg cannot tell us -- the caller
    treats that as "unknown" rather than as "empty", because a zero-length claim about a file
    that clearly has bytes in it is more likely a probe failure than the truth."""
    exe = ffmpeg_exe()
    proc = _run([exe, "-i", str(path), "-f", "null", "-"])
    for line in reversed((proc.stderr or "").splitlines()):
        if "time=" in line:
            stamp = line.split("time=", 1)[1].split(" ", 1)[0]
            try:
                h, m, s = stamp.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                continue
    return 0.0


def split(src: Path, out_dir: Path, *, seconds: int = CHUNK_SECONDS) -> list[Chunk]:
    """Cut `src` into fixed-length chunks under `out_dir`, returning them in order.

    `-c copy` means the stream is copied, never re-encoded: it is fast, lossless, and keeps the
    source's own format. The trade-off is that cuts land on the nearest frame boundary rather
    than exactly on the second, which is well inside the tolerance of something already being
    summarised over two-minute windows.
    """
    if not src.is_file():
        raise AudioError(f"no audio file at {src}")
    if src.suffix.lower() not in AUDIO_SUFFIXES:
        raise AudioError(f"unsupported audio type {src.suffix or '(none)'!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"chunk_*{src.suffix}"):
        stale.unlink()

    pattern = str(out_dir / f"chunk_%02d{src.suffix}")
    proc = _run([
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-f", "segment", "-segment_time", str(seconds), "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    ])
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not split the audio: {(proc.stderr or '').strip()[:300]}")

    paths = sorted(out_dir.glob(f"chunk_*{src.suffix}"))
    if not paths:
        raise AudioError("ffmpeg produced no chunks; the file may be empty or not audio")
    return [Chunk(index=i, path=p, start_s=float(i * seconds)) for i, p in enumerate(paths)]


def manifest(chunks: list[Chunk], *, source_name: str, seconds: int = CHUNK_SECONDS) -> dict:
    """What the GX10 is told about the job. Written beside the chunks and shipped with them, so
    the remote side needs no arguments beyond a directory."""
    return {
        "source_name": source_name,
        "chunk_seconds": seconds,
        "chunks": [{"index": c.index, "name": c.name, "file": c.path.name, "start_s": c.start_s} for c in chunks],
    }


def write_manifest(out_dir: Path, data: dict) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
