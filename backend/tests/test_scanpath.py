"""scanpath.py: greedy-max fixation selection with inhibition-of-return.

Uses hand-built saliency maps (not real images) so the algorithm is tested in isolation
from saliency.py. The one required synthetic-image case from the spec build order --
"one white square on black -> one fixation at its center" -- lives in test_saliency.py,
since it exercises the saliency map itself, not just the fixation-picking algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from sightline.scanpath import Fixation, compute_scanpath


def _bump(shape: tuple[int, int], cx: int, cy: int, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    return amplitude * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))


def test_single_bump_yields_one_fixation_at_its_peak():
    m = _bump((100, 100), cx=30, cy=70, sigma=10)
    fixations = compute_scanpath(m, n_fixations=1)
    assert fixations == [Fixation(order=1, x=30, y=70, value=pytest.approx(1.0))]


def test_two_separated_bumps_are_visited_strongest_first_in_order():
    m = _bump((200, 200), cx=40, cy=40, sigma=8, amplitude=1.0) + _bump((200, 200), cx=160, cy=150, sigma=8, amplitude=0.6)
    fixations = compute_scanpath(m, n_fixations=2, ior_sigma_px=20)
    assert [f.order for f in fixations] == [1, 2]
    assert (fixations[0].x, fixations[0].y) == (40, 40)
    assert (fixations[1].x, fixations[1].y) == (160, 150)


def test_inhibition_of_return_prevents_picking_the_same_peak_twice():
    # A single, wide bump: without IOR the second fixation would land on the same peak.
    m = _bump((200, 200), cx=100, cy=100, sigma=30)
    fixations = compute_scanpath(m, n_fixations=3, ior_sigma_px=15)
    points = [(f.x, f.y) for f in fixations]
    assert len(set(points)) == 3, f"expected 3 distinct fixations, got {points}"
    # Each later fixation should sit further from the first than the last (monotonically
    # spreading outward as the centre gets suppressed).
    dists = [((x - 100) ** 2 + (y - 100) ** 2) ** 0.5 for x, y in points]
    assert dists[0] < dists[1] <= dists[2] + 1e-9


def test_fixation_order_matches_reading_order_on_exact_ties():
    # Two pixels tied at the global max: argmax (and therefore this function) must be
    # deterministic, breaking ties in row-major (reading) order.
    m = np.zeros((10, 10))
    m[2, 8] = 1.0
    m[2, 1] = 1.0  # same row, earlier column: this one wins the tie
    fixations = compute_scanpath(m, n_fixations=1)
    assert (fixations[0].x, fixations[0].y) == (1, 2)


def test_requesting_more_fixations_than_pixels_is_capped_not_an_error():
    m = np.array([[1.0]])
    fixations = compute_scanpath(m, n_fixations=6)
    assert len(fixations) == 1


def test_zero_fixations_requested_returns_empty():
    m = _bump((20, 20), cx=10, cy=10, sigma=3)
    assert compute_scanpath(m, n_fixations=0) == []


def test_rejects_a_non_2d_map():
    with pytest.raises(ValueError, match="2D"):
        compute_scanpath(np.zeros((5, 5, 3)))


def test_fixation_to_dict_round_trips_plain_values():
    f = Fixation(order=1, x=5, y=7, value=0.42)
    assert f.to_dict() == {"order": 1, "x": 5, "y": 7, "value": 0.42}
