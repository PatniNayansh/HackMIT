"""Run history on the plain filesystem. No database.

    data/history/<run_id>/
        run.json          deck metadata: title, dates, status, model, intent, subfield, ...
        input.json        [{index, text}] exactly what each persona was shown as text
        slides/001.png    the rendered slide images they were shown
        results/001.json  one SlideResult per completed slide: all three persona readings
                          (or the reason one failed) and every metric with its provenance

Everything the review UI shows is in these files, so opening a saved run needs no network and
no API key: the server reads the directory and nothing else. Runs are written slide by slide as
they complete, which is also what lets the UI stream and lets a reload mid-run pick up where it
was. `status` says how far a run got; it is never inferred from the files.

Bundled runs (the read-only fixtures shipped in the repo) are looked up after the writable
root and can never be written to, mirroring the read-only layer of the audience cache.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .audiences import PROMPT_VERSION, DeckProfile
from .deck import SlideResult
from .ingest import Slide

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
BUNDLED_RUNS_DIR = BACKEND / "fixtures" / "runs"


def data_dir() -> Path:
    return Path(os.environ.get("SIGHTLINE_DATA_DIR") or REPO_ROOT / "data")


# 2: per-slide inferred intent; alignment measured against it; confidence, blind-spot and
# divergence no longer part of the results. Runs saved before that have no version and open in a
# reduced view (see server.public_meta).
SCHEMA_VERSION = 2

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# draft -> running -> complete | failed. "interrupted" is reported (never stored) for a run
# that says "running" but that no live process owns.
STATUSES = ("draft", "running", "complete", "failed")


class RunNotFound(KeyError):
    pass


class ReadOnlyRun(PermissionError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: a poll never sees half a file


class RunStore:
    def __init__(self, root: str | Path, bundled: Sequence[str | Path] = ()):
        self.root = Path(root)
        self.bundled = [Path(b) for b in bundled]

    # ---------------------------------------------------------------- locating a run

    def _dir(self, run_id: str) -> tuple[Path, bool]:
        """(directory, read_only). Raises RunNotFound, including for ids that could escape."""
        if not RUN_ID.match(run_id):
            raise RunNotFound(run_id)
        for base, read_only in ((self.root, False), *((b, True) for b in self.bundled)):
            d = base / run_id
            if (d / "run.json").is_file():
                return d, read_only
        raise RunNotFound(run_id)

    def _writable(self, run_id: str) -> Path:
        d, read_only = self._dir(run_id)
        if read_only:
            raise ReadOnlyRun(f"{run_id} is a bundled sample and cannot be modified")
        return d

    def is_read_only(self, run_id: str) -> bool:
        return self._dir(run_id)[1]

    # ---------------------------------------------------------------------- writing

    @staticmethod
    def new_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)

    def create_draft(
        self,
        *,
        title: str,
        source_filename: str,
        slides: Sequence[Slide],
        inferred: DeckProfile | None,
        inference_error: str | None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist an uploaded deck before the run starts, so the presenter can confirm the
        intent and subfield against it. A draft is not history until it has been started."""
        run_id = run_id or self.new_run_id()
        if not RUN_ID.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        d = self.root / run_id
        for s in slides:
            (d / "slides").mkdir(parents=True, exist_ok=True)
            (d / "slides" / f"{s.index:03d}.png").write_bytes(s.image_png)
        _write_json(d / "input.json", [{"index": s.index, "text": s.text} for s in slides])
        meta = {
            "run_id": run_id,
            "title": title,
            "source_filename": source_filename,
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "schema_version": SCHEMA_VERSION,
            "status": "draft",
            "slide_count": len(slides),
            "intent": None,
            # What the presenter confirmed, and what the model suggested. Kept apart so the UI
            # can say "unconfirmed" until the presenter has looked at it.
            "profile": None,
            "profile_inferred": (
                {"domain": inferred.domain, "adjacent_field": inferred.adjacent_field} if inferred else None
            ),
            "profile_inference_error": inference_error,
            "model": None,
            "intent_model": None,
            "embedding_model": None,
            "prompt_version": PROMPT_VERSION,
            "error": None,
            "sample": False,
        }
        _write_json(d / "run.json", meta)
        return meta

    def update_meta(self, run_id: str, **fields: Any) -> dict[str, Any]:
        d = self._writable(run_id)
        meta = json.loads((d / "run.json").read_text())
        unknown = set(fields) - set(meta)
        if unknown:
            raise KeyError(f"not run metadata: {sorted(unknown)}")
        meta.update(fields)
        _write_json(d / "run.json", meta)
        return meta

    def clear_results(self, run_id: str) -> None:
        """Forget earlier results before a failed run is started again."""
        for p in (self._writable(run_id) / "results").glob("*.json"):
            p.unlink()

    def save_result(self, run_id: str, result: SlideResult) -> None:
        _write_json(self._writable(run_id) / "results" / f"{result['index']:03d}.json", result)

    # ---------------------------------------------------------------------- reading

    def load_meta(self, run_id: str) -> dict[str, Any]:
        d, _ = self._dir(run_id)
        return json.loads((d / "run.json").read_text())

    def load_slides(self, run_id: str) -> list[Slide]:
        d, _ = self._dir(run_id)
        return [
            Slide(rec["index"], rec["text"], (d / "slides" / f"{rec['index']:03d}.png").read_bytes())
            for rec in json.loads((d / "input.json").read_text())
        ]

    def load_results(self, run_id: str) -> list[SlideResult]:
        d, _ = self._dir(run_id)
        out: list[SlideResult] = []
        for p in sorted((d / "results").glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue  # half-written by a crash; the slide simply reads as pending
        return out

    def image_path(self, run_id: str, index: int) -> Path:
        d, _ = self._dir(run_id)
        p = d / "slides" / f"{index:03d}.png"
        if not p.is_file():
            raise RunNotFound(f"{run_id}/slide {index}")
        return p

    def list_runs(self, *, include_drafts: bool = False) -> list[dict[str, Any]]:
        """Newest first. A directory that cannot be read is skipped, not fatal: one corrupt
        run must not take the history page down."""
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for base in (self.root, *self.bundled):
            if not base.is_dir():
                continue
            for d in base.iterdir():
                if d.name in seen or not (d / "run.json").is_file():
                    continue
                try:
                    meta = json.loads((d / "run.json").read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                seen.add(d.name)
                if meta.get("status") == "draft" and not include_drafts:
                    continue
                rows.append(meta)
        rows.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        return rows
