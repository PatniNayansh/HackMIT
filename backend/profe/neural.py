"""Neural layer: predicted cortical response to a narrated slide, from TRIBE v2 (Meta AI,
github.com/facebookresearch/tribev2). PRECOMPUTED ONLY. Read this whole docstring before
touching anything here -- every constraint below is load-bearing, not decoration.

What we extract, and what we refuse to extract (spec 5, 2.2):
  * We report WHERE the predicted response sits -- language-region drive vs. visual-region
    drive -- never HOW MUCH of it there is in aggregate. `processing_ratio` is that spatial
    readout: language_drive / visual_drive, reported as a Z-score against the rest of the
    deck. It is a proposed readout, not a validated metric; every place it is shown must
    say so.
  * A single scalar collapsed from the whole predicted response (global field power, GFP,
    a stand-in for "engagement") is a documented null result: arXiv 2607.01400 found it
    does not correlate with real attention data (pooled partial r = 0.058, p = 0.23) across
    six cortical networks. That number must NEVER be presented as a finding. It exists here
    only as `gfp_negative_baseline`, a labeled negative baseline for the Methods panel,
    shown next to the null result it reproduces -- never on its own.

Hard constraints this module enforces structurally, not just documents:
  * TRIBE needs narrated (audio + transcript) input; a silent slide gives it almost
    nothing (spec 5). `scripts/precompute_neural/narrate.py` synthesizes narration for
    silent decks before TRIBE ever sees them.
  * Inference costs 6-13 GPU-minutes per slide (real cost, confirmed against the actual
    TRIBE v2 repo: three transformer backbones, one of them "giant"-sized, CUDA-only, no
    documented CPU or ROCm path). Live inference inside the request/response cycle is not
    an option. `TribeNeural` -- the class that would load and run the real model -- lives
    entirely in `scripts/precompute_neural/`, in its own virtualenv (TRIBE pins
    numpy==2.2.6 and torch<2.7, which would collide with this package's own dependencies).
    It is not importable from here at all. `assert_may_run_tribe()` is the guard the CLI
    calls before doing so; the FastAPI process never calls it.
  * TRIBE was trained on movie-watching fMRI; narrated slides are out of distribution. These
    are predicted values, never measured ones -- true of every number this module returns, and
    the reason the UI names the panel a prediction rather than a reading.

This module is everything the FastAPI process is allowed to touch: the output contract
(`NeuralModel`, as documentation of what the CLI must produce), the pure arithmetic
(`processing_ratio`, `deck_z_scores`), and `CachedNeural`, a read-only loader for what the
CLI already computed. It imports no ML library and makes no GPU call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

N_VERTICES_FSAVERAGE5 = 20484  # fsaverage5 cortical surface mesh, both hemispheres combined
TR_SECONDS = 1.0

# Rendered surface views a precomputed slide must have, per spec 9 (screen 5): lateral and
# medial, both hemispheres.
SURFACE_VIEWS: tuple[str, ...] = ("lateral_left", "medial_left", "lateral_right", "medial_right")


class NeuralModel(Protocol):
    """The contract `scripts/precompute_neural/` must satisfy. Documentation, not a type
    the FastAPI process ever constructs -- see the module docstring."""

    def predict(self, video_path: Path, audio_path: Path, transcript: str) -> np.ndarray:
        """(T, 20484) predicted cortical response on fsaverage5, one row per 1s TR."""
        ...


class NotRunnableHere(RuntimeError):
    """Something tried to run TRIBE inference outside the standalone precompute CLI."""


def assert_may_run_tribe() -> None:
    """The one check standing between "someone imported the wrong thing" and 6-13 minutes
    of GPU time firing inside a web request. `scripts/precompute_neural/run.py` calls this
    before touching the model; nothing under `profe/` ever does."""
    if os.environ.get("PROFE_PROCESS") == "server":
        raise NotRunnableHere(
            "TRIBE inference cannot run inside the FastAPI process: it costs 6-13 minutes "
            "of GPU time per slide. Run scripts/precompute_neural/run.py (on a CUDA "
            "machine, e.g. the ASUS GX10) and let the web app read the result through "
            "CachedNeural instead."
        )


# ---------------------------------------------------------------------------- arithmetic


@dataclass(frozen=True)
class SlideNeuralMetrics:
    slide_index: int
    language_drive: float  # mean predicted response over language-network vertices
    visual_drive: float  # mean predicted response over visual-network vertices
    processing_ratio: float  # language_drive / visual_drive
    gfp_negative_baseline: float  # see module docstring: NEVER a finding, methods-panel only
    narration_transcript: str  # the synthesized narration TRIBE actually heard (spec rule 2:
    # every number must decompose to its source text; this is this number's source)


def processing_ratio(language_drive: float, visual_drive: float) -> float:
    """language-region drive over visual-region drive. Undefined (and refused, not
    divided-by-zero-into-inf) when there is no visual drive to compare against."""
    if visual_drive == 0:
        raise ValueError("visual_drive is 0; processing_ratio is undefined")
    return language_drive / visual_drive


def deck_z_scores(ratios: Mapping[int, float]) -> dict[int, float]:
    """Each slide's processing_ratio expressed as a Z-score against this deck's own mean
    (spec 5: "expressed as a Z-score relative to the deck mean"). A deck of one slide, or
    one with zero variance, has no meaningful spread: every slide gets 0.0 rather than a
    division by zero standing in for "no signal"."""
    if not ratios:
        return {}
    values = np.array(list(ratios.values()), dtype=float)
    mean, std = float(values.mean()), float(values.std())
    if std < 1e-12:
        return {i: 0.0 for i in ratios}
    return {i: (v - mean) / std for i, v in ratios.items()}


# -------------------------------------------------------------------------- cached reader


class NeuralNotCached(KeyError):
    """No precomputed neural result for this run/slide. The caller must show a clear empty
    state (spec 9, screen 5: "never a placeholder brain"), not fabricate one."""


class CachedNeural:
    """Reads what `scripts/precompute_neural/run.py` already wrote to disk. The only neural
    path the FastAPI process may use -- see the module docstring. Read-only: this process
    never writes here.

    Layout, one directory per scored slide:
        <root>/<run_id>/<slide_index>/metrics.json   SlideNeuralMetrics, plain dict
        <root>/<run_id>/<slide_index>/<view>.png      one per name in SURFACE_VIEWS
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def has(self, run_id: str, slide_index: int) -> bool:
        return (self.root / run_id / str(slide_index) / "metrics.json").is_file()

    def metrics(self, run_id: str, slide_index: int) -> SlideNeuralMetrics:
        p = self.root / run_id / str(slide_index) / "metrics.json"
        if not p.is_file():
            raise NeuralNotCached(f"no cached neural result for {run_id}/slide {slide_index}")
        d = json.loads(p.read_text(encoding="utf-8"))
        return SlideNeuralMetrics(
            slide_index=slide_index,
            language_drive=d["language_drive"],
            visual_drive=d["visual_drive"],
            processing_ratio=d["processing_ratio"],
            gfp_negative_baseline=d["gfp_negative_baseline"],
            narration_transcript=d.get("narration_transcript", ""),
        )

    def scored_slides(self, run_id: str) -> list[int]:
        d = self.root / run_id
        if not d.is_dir():
            return []
        return sorted(int(p.name) for p in d.iterdir() if (p / "metrics.json").is_file())

    def image_path(self, run_id: str, slide_index: int, view: str) -> Path:
        if view not in SURFACE_VIEWS:
            raise ValueError(f"unknown surface view {view!r}; must be one of {SURFACE_VIEWS}")
        p = self.root / run_id / str(slide_index) / f"{view}.png"
        if not p.is_file():
            raise NeuralNotCached(f"no cached {view} image for {run_id}/slide {slide_index}")
        return p

    def deck_rollup(self, run_id: str, slide_indices: Sequence[int]) -> dict[str, object]:
        """Deck-level view for the neural screen: per-slide metrics plus each one's
        processing_ratio Z-score against the rest of this (precomputed-only) deck."""
        scored = self.scored_slides(run_id)
        per_slide = {i: self.metrics(run_id, i) for i in scored}
        z = deck_z_scores({i: m.processing_ratio for i, m in per_slide.items()})
        return {
            "n_slides": len(slide_indices),
            "n_scored": len(scored),
            "scored": scored,
            "per_slide": [
                {
                    "slide": i,
                    "language_drive": m.language_drive,
                    "visual_drive": m.visual_drive,
                    "processing_ratio": m.processing_ratio,
                    "processing_ratio_z": z[i],
                }
                for i, m in sorted(per_slide.items())
            ],
        }
