"""Bottom-up visual saliency: where does the eye land first, from image structure alone.

`SaliencyModel` is a protocol with three implementations behind it, per the spec:
  * `SpectralResidualSaliency` -- OpenCV's spectral-residual method. CPU, well under a
    second, no model weights to download. This is the default and the live demo path.
  * `DeepGazeSaliency` -- DeepGaze IIE, a trained deep saliency model. Not wired up yet:
    it raises a clear startup error rather than silently falling back to something else,
    so a missing integration is loud, never a quietly wrong saliency map.
  * `StubSaliency` -- deterministic, pure numpy (no cv2, no weights): a Gaussian-blurred
    grayscale intensity map. For tests, and for any environment where OpenCV's saliency
    module is unavailable.

Saliency measures bottom-up visual attention -- what draws the eye first because of
contrast, edges and color -- not comprehension and not aesthetic quality. That is a
required Methods-panel disclosure (spec 10); nothing in this module should be presented
as either of those.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Protocol

import numpy as np


class SaliencyModel(Protocol):
    name: str

    def compute(self, image: np.ndarray) -> np.ndarray:
        """`image`: (H, W, 3) uint8, RGB. Returns an (H, W) float64 map normalised to
        [0, 1], at the same resolution as the input."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-12:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def _check_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape}")


class SpectralResidualSaliency:
    """OpenCV's StaticSaliencySpectralResidual. CPU, near-instant, zero model weights.
    Default and demo path."""

    name = "spectral_residual"

    def compute(self, image: np.ndarray) -> np.ndarray:
        _check_rgb(image)
        import cv2

        bgr = np.ascontiguousarray(image[:, :, ::-1])  # RGB -> BGR: what cv2 expects
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal = detector.computeSaliency(bgr)
        if not ok:
            raise RuntimeError("OpenCV spectral-residual saliency computation failed")
        return _normalize(sal)


class DeepGazeSaliency:
    """DeepGaze IIE. Deliberately not integrated in this build: model loading and
    inference are not implemented. Per spec 8 ("DeepGaze IIE if weights present, else
    clear startup error"), this raises immediately and says so, rather than pretending
    to work or silently falling back to a different model. Kept as a class, not deleted,
    so the `SaliencyModel` contract stays complete and the real integration has a home.
    """

    name = "deepgaze"

    def __init__(self, weights_path: str | Path | None = None):
        configured = weights_path or os.environ.get("PROFE_DEEPGAZE_WEIGHTS")
        self.weights_path = Path(configured) if configured else None
        if self.weights_path is None or not self.weights_path.is_file():
            raise RuntimeError(
                "DeepGaze IIE is not available: no weights file at "
                f"{self.weights_path or '(PROFE_DEEPGAZE_WEIGHTS is unset)'}. "
                "DeepGaze integration is not implemented in this build; use "
                "SpectralResidualSaliency (the default) or StubSaliency instead."
            )
        raise NotImplementedError(
            "DeepGaze IIE weights were found, but model loading and inference are not "
            "implemented yet. This class exists to keep the SaliencyModel contract "
            "complete; wire the real model in here."
        )

    def compute(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - unreachable
        raise NotImplementedError


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _convolve_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1D convolution along one axis, edges replicated before convolving (not zero-padded):
    zero-padding would darken a blurred image's borders even when it is perfectly uniform,
    fabricating saliency that is not there. `mode="valid"` on the padded array is exactly
    `mode="same"` on the original, without that artifact."""
    radius = (len(kernel) - 1) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=padded)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian blur, pure numpy. No scipy/cv2 dependency, so `StubSaliency`
    never depends on anything that could be the thing under test elsewhere."""
    if sigma <= 0:
        return arr.astype(np.float64)
    k = _gaussian_kernel1d(sigma)
    out = _convolve_axis(arr.astype(np.float64), k, axis=1)
    out = _convolve_axis(out, k, axis=0)
    return out


class StubSaliency:
    """Deterministic, dependency-free: a Gaussian-blurred grayscale intensity map. Not a
    real saliency model, and never the default -- only for tests and for environments
    without OpenCV's saliency module."""

    name = "stub"

    def __init__(self, blur_sigma_px: float = 12.0):
        self.blur_sigma_px = blur_sigma_px

    def compute(self, image: np.ndarray) -> np.ndarray:
        _check_rgb(image)
        gray = image.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        return _normalize(_gaussian_blur(gray, self.blur_sigma_px))


_MODELS: dict[str, type] = {
    "spectral_residual": SpectralResidualSaliency,
    "deepgaze": DeepGazeSaliency,
    "stub": StubSaliency,
}


def default_saliency_model() -> SaliencyModel:
    return SpectralResidualSaliency()


def decode_png(image_png: bytes) -> np.ndarray:
    """PNG bytes (as stored for a slide) -> (H, W, 3) uint8 RGB array."""
    from PIL import Image

    with Image.open(io.BytesIO(image_png)) as img:
        return np.array(img.convert("RGB"))


def compute_saliency(image_png: bytes, model: SaliencyModel | None = None) -> np.ndarray:
    """The entry point callers use: slide PNG bytes -> (H, W) saliency map in [0, 1]."""
    model = model or default_saliency_model()
    return model.compute(decode_png(image_png))
