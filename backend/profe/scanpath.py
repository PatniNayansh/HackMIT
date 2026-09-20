"""Scanpath approximation: greedy-max fixation selection with inhibition-of-return.

Turns a saliency map into an ordered sequence of fixations -- where the eye lands first,
second, third... by repeatedly taking the most salient remaining point, then suppressing
a region around it (a zeroed-out sigma=60px Gaussian, per spec 8) so the same point is not
picked again. This is the classic Itti-Koch winner-take-all-plus-inhibition-of-return
scheme, simplified: no foveal falloff, no saccade cost, no visual working memory beyond
"already looked here."

THIS IS NOT A VALIDATED SCANPATH MODEL. It approximates one. Say so wherever it is shown
(spec 3 and 10) -- it is a fast, deterministic proxy for fixation order, not a claim about
how a real viewer's eyes actually move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_N_FIXATIONS = 6
DEFAULT_IOR_SIGMA_PX = 60.0


@dataclass(frozen=True)
class Fixation:
    order: int  # 1-based; 1 = first fixation
    x: int  # column, in the saliency map's own pixel coordinates
    y: int  # row
    value: float  # the map's value at this point, before any inhibition was applied

    def to_dict(self) -> dict:
        return {"order": self.order, "x": self.x, "y": self.y, "value": self.value}


def _inhibition(shape: tuple[int, int], cx: int, cy: int, sigma: float) -> np.ndarray:
    """A Gaussian centred at (cx, cy), 1.0 at the centre falling off with sigma. Multiplying
    the saliency map by (1 - this) suppresses that region without a hard cutoff."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def compute_scanpath(
    saliency_map: np.ndarray,
    n_fixations: int = DEFAULT_N_FIXATIONS,
    ior_sigma_px: float = DEFAULT_IOR_SIGMA_PX,
) -> list[Fixation]:
    """Greedy max + inhibition-of-return. Deterministic for a deterministic saliency map:
    a tie is broken in reading order (top row first, then left to right), since
    `np.argmax` returns the first occurrence in row-major order."""
    if saliency_map.ndim != 2:
        raise ValueError(f"saliency_map must be 2D (H, W), got shape {saliency_map.shape}")
    if n_fixations < 0:
        raise ValueError(f"n_fixations must be >= 0, got {n_fixations}")

    h, w = saliency_map.shape
    remaining = saliency_map.astype(np.float64).copy()
    fixations: list[Fixation] = []
    n = min(n_fixations, h * w)
    for i in range(1, n + 1):
        y, x = np.unravel_index(np.argmax(remaining), remaining.shape)
        fixations.append(Fixation(order=i, x=int(x), y=int(y), value=float(saliency_map[y, x])))
        remaining *= 1.0 - _inhibition(remaining.shape, int(x), int(y), ior_sigma_px)
    return fixations
