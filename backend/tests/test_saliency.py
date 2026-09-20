"""saliency.py.

The one synthetic-image case the spec build order calls out by name: "one white square on
black -> one fixation at its center." `StubSaliency` is used here (not
`SpectralResidualSaliency`) so this test needs no OpenCV install and is exact and
dependency-free; SpectralResidual is covered separately, skipped when cv2 is unavailable.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from profe.saliency import DeepGazeSaliency, StubSaliency, compute_saliency, decode_png, default_saliency_model
from profe.scanpath import compute_scanpath


def _square_on_black(size: int = 201, square: int = 21, fill=(255, 255, 255)) -> np.ndarray:
    """A `square`x`square` block, centred exactly (both `size` and `square` are odd, so
    there is a single centre pixel), on an otherwise black image."""
    assert size % 2 == 1 and square % 2 == 1, "keep both odd so the square has an exact centre pixel"
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    r = square // 2
    img[c - r : c + r + 1, c - r : c + r + 1] = fill
    return img


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def test_white_square_on_black_saliency_peaks_at_its_center():
    img = _square_on_black(size=201, square=21)
    sal = StubSaliency(blur_sigma_px=12.0).compute(img)
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    assert (x, y) == (100, 100)


def test_white_square_on_black_yields_one_fixation_at_its_center():
    """The exact case named in the spec build order: saliency + scanpath together produce
    a single fixation sitting on the square's centre."""
    img = _square_on_black(size=201, square=21)
    sal = StubSaliency(blur_sigma_px=12.0).compute(img)
    fixations = compute_scanpath(sal, n_fixations=1)
    assert len(fixations) == 1
    assert (fixations[0].x, fixations[0].y) == (100, 100)


def test_off_center_square_moves_the_fixation_with_it():
    img = np.zeros((201, 201, 3), dtype=np.uint8)
    img[40:61, 140:161] = 255  # a 21x21 square centred at (150, 50)
    sal = StubSaliency(blur_sigma_px=12.0).compute(img)
    fixations = compute_scanpath(sal, n_fixations=1)
    assert (fixations[0].x, fixations[0].y) == (150, 50)


def test_saliency_map_is_normalised_to_unit_interval():
    img = _square_on_black()
    sal = StubSaliency().compute(img)
    assert sal.min() == pytest.approx(0.0, abs=1e-9)
    assert sal.max() == pytest.approx(1.0, abs=1e-9)


def test_uniform_image_has_no_saliency_anywhere():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    sal = StubSaliency().compute(img)
    assert np.allclose(sal, 0.0)


def test_rejects_a_non_rgb_image():
    with pytest.raises(ValueError, match="RGB"):
        StubSaliency().compute(np.zeros((10, 10)))


def test_compute_saliency_decodes_png_bytes_end_to_end():
    png = _png_bytes(_square_on_black())
    sal = compute_saliency(png, model=StubSaliency())
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    assert (x, y) == (100, 100)
    assert sal.shape == (201, 201)


def test_decode_png_round_trips_shape_and_content():
    img = _square_on_black(size=51, square=11)
    decoded = decode_png(_png_bytes(img))
    assert decoded.shape == img.shape
    assert np.array_equal(decoded, img)


def test_default_saliency_model_is_spectral_residual():
    assert default_saliency_model().name == "spectral_residual"


# ------------------------------------------------------------- DeepGaze: clear failure


def test_deepgaze_without_weights_raises_a_clear_startup_error(monkeypatch):
    monkeypatch.delenv("PROFE_DEEPGAZE_WEIGHTS", raising=False)
    with pytest.raises(RuntimeError, match="DeepGaze IIE is not available"):
        DeepGazeSaliency()


def test_deepgaze_error_names_the_working_alternatives(monkeypatch):
    monkeypatch.delenv("PROFE_DEEPGAZE_WEIGHTS", raising=False)
    with pytest.raises(RuntimeError, match="SpectralResidualSaliency"):
        DeepGazeSaliency()


# --------------------------------------------------------- SpectralResidual, if present


def test_spectral_residual_highlights_the_square_over_the_uniform_background():
    cv2 = pytest.importorskip("cv2", reason="OpenCV (with the saliency module) is not installed")
    if not hasattr(cv2, "saliency"):
        pytest.skip("this OpenCV build has no saliency module (needs opencv-contrib)")
    from profe.saliency import SpectralResidualSaliency

    img = _square_on_black(size=201, square=41)
    sal = SpectralResidualSaliency().compute(img)
    assert sal.shape == (201, 201)
    inside = sal[90:111, 90:111].mean()
    outside = sal[:20, :20].mean()
    assert inside > outside
