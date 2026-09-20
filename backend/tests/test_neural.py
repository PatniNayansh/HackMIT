"""neural.py: the arithmetic and the cached-result reader -- everything the FastAPI process
is allowed to touch. No torch, no GPU, no network: the real TRIBE model lives entirely in
scripts/precompute_neural/, in its own environment, and is not importable from here.
"""

from __future__ import annotations

import json
import os

import pytest

from profe.neural import (
    SURFACE_VIEWS,
    CachedNeural,
    NeuralNotCached,
    NotRunnableHere,
    SlideNeuralMetrics,
    assert_may_run_tribe,
    deck_z_scores,
    processing_ratio,
)

# --------------------------------------------------------------------------- arithmetic


def test_processing_ratio_is_language_over_visual():
    assert processing_ratio(3.0, 1.5) == pytest.approx(2.0)


def test_processing_ratio_rejects_zero_visual_drive_rather_than_returning_inf():
    with pytest.raises(ValueError, match="undefined"):
        processing_ratio(1.0, 0.0)


def test_deck_z_scores_are_standard_scores_against_the_deck_mean():
    z = deck_z_scores({1: 1.0, 2: 2.0, 3: 3.0})
    assert z[2] == pytest.approx(0.0)
    assert z[1] == pytest.approx(-z[3])
    assert z[1] < z[2] < z[3]


def test_deck_z_scores_of_empty_deck_is_empty():
    assert deck_z_scores({}) == {}


def test_deck_z_scores_with_zero_variance_is_all_zero_not_nan():
    z = deck_z_scores({1: 5.0, 2: 5.0, 3: 5.0})
    assert z == {1: 0.0, 2: 0.0, 3: 0.0}


def test_deck_z_scores_single_slide_is_zero():
    assert deck_z_scores({7: 42.0}) == {7: 0.0}


# ------------------------------------------------------------------------------- guard


def test_tribe_refuses_to_run_inside_the_server_process(monkeypatch):
    monkeypatch.setenv("PROFE_PROCESS", "server")
    with pytest.raises(NotRunnableHere, match="6-13 minutes"):
        assert_may_run_tribe()


def test_tribe_guard_is_silent_outside_the_server_process(monkeypatch):
    monkeypatch.delenv("PROFE_PROCESS", raising=False)
    assert_may_run_tribe() is None


# --------------------------------------------------------------------------- CachedNeural


def _write_metrics(root, run_id, slide, language_drive, visual_drive, gfp=0.01, transcript="the slide, read aloud"):
    d = root / run_id / str(slide)
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "language_drive": language_drive,
                "visual_drive": visual_drive,
                "processing_ratio": language_drive / visual_drive,
                "gfp_negative_baseline": gfp,
                "narration_transcript": transcript,
            }
        ),
        encoding="utf-8",
    )
    return d


def test_cached_neural_round_trips_metrics(tmp_path):
    _write_metrics(tmp_path, "run1", 3, language_drive=4.0, visual_drive=2.0)
    cache = CachedNeural(tmp_path)
    assert cache.has("run1", 3)
    m = cache.metrics("run1", 3)
    assert m == SlideNeuralMetrics(
        slide_index=3,
        language_drive=4.0,
        visual_drive=2.0,
        processing_ratio=2.0,
        gfp_negative_baseline=0.01,
        narration_transcript="the slide, read aloud",
    )


def test_cached_neural_missing_slide_raises_not_a_placeholder(tmp_path):
    cache = CachedNeural(tmp_path)
    assert not cache.has("nope", 1)
    with pytest.raises(NeuralNotCached):
        cache.metrics("nope", 1)


def test_cached_neural_missing_image_raises(tmp_path):
    d = _write_metrics(tmp_path, "run1", 1, 1.0, 1.0)
    cache = CachedNeural(tmp_path)
    with pytest.raises(NeuralNotCached):
        cache.image_path("run1", 1, "lateral_left")
    (d / "lateral_left.png").write_bytes(b"\x89PNG\r\n")
    assert cache.image_path("run1", 1, "lateral_left") == d / "lateral_left.png"


def test_cached_neural_image_path_rejects_an_unknown_view(tmp_path):
    cache = CachedNeural(tmp_path)
    with pytest.raises(ValueError, match="unknown surface view"):
        cache.image_path("run1", 1, "top_down")


def test_scored_slides_of_an_unknown_run_is_empty_not_an_error(tmp_path):
    assert CachedNeural(tmp_path).scored_slides("nope") == []


def test_deck_rollup_reports_z_scores_relative_to_scored_slides_only(tmp_path):
    _write_metrics(tmp_path, "run1", 1, language_drive=1.0, visual_drive=1.0)  # ratio 1.0
    _write_metrics(tmp_path, "run1", 2, language_drive=3.0, visual_drive=1.0)  # ratio 3.0
    cache = CachedNeural(tmp_path)
    out = cache.deck_rollup("run1", slide_indices=[1, 2, 3])
    assert out["n_slides"] == 3
    assert out["n_scored"] == 2
    assert out["scored"] == [1, 2]
    per_slide = {row["slide"]: row for row in out["per_slide"]}
    assert per_slide[1]["processing_ratio_z"] < per_slide[2]["processing_ratio_z"]


def test_surface_views_cover_both_hemispheres_and_both_angles():
    assert set(SURFACE_VIEWS) == {"lateral_left", "medial_left", "lateral_right", "medial_right"}
