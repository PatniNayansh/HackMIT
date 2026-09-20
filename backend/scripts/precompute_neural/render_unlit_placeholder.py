#!/usr/bin/env python
"""One-off: render the "unlit" brain (no stat map, anatomy only) exactly the way TRIBE v2
itself does, for the frontend's neural empty state -- so what a viewer sees there is the
real thing with nothing predicted on it yet, not an approximation of it.

Matched against github.com/facebookresearch/tribev2's own rendering code
(tribev2/plotting/cortical_pv.py, PlotBrainPyvista, the class tribev2/plotting/__init__.py
actually aliases PlotBrain to -- the notebook-facing entry point), confirmed by inspecting
the repo directly:
  * Mesh: nilearn's fsaverage5 (same source we already used), HALF-inflated --
    coords = 0.5 * inflated + 0.5 * pial (BasePlotBrain.get_mesh, inflate="half" default).
    Plain inflated (what this script used before) is not what TRIBE v2 renders.
  * Coloring with no stat_map: bg_norm = (bg_map - min) / (max - min); bg_rgb = 1 - bg_norm
    per channel (bg_darkness=0) -- inverted grayscale sulcal depth, not nilearn's own
    Greys-colormap-plus-shading combination.
  * Renderer: PyVista (VTK), off-screen, smooth_shading=True, ambient=0.3, white
    background -- not matplotlib's Poly3DCollection, which shades differently.
  * Camera for a "left" lateral view: view_vector([-1, 0, 0], viewup=[0, 0, 1]).

Not part of the app's runtime: run this once, commit the resulting PNG to frontend/img/.
Needs pyvista + nilearn installed ad hoc (`pip install pyvista nilearn`) -- neither is a
dependency of backend/ or of this directory's requirements.txt; both are heavier installs
this repo otherwise avoids at runtime for a static build-time asset.

    python render_unlit_placeholder.py ../../../frontend/img/brain-unlit.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from nilearn import datasets, surface


def half_inflate(fsaverage, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """coords, faces for the half-inflated mesh TRIBE v2 actually renders on."""
    infl_coords, faces = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
    pial_coords, _ = surface.load_surf_mesh(fsaverage[f"pial_{hemi}"])
    coords = 0.5 * infl_coords + 0.5 * pial_coords
    return coords, faces


def bg_rgb_from_sulc(sulc: np.ndarray) -> np.ndarray:
    """TRIBE v2's exact "no stat_map" coloring: inverted, normalised sulcal depth,
    broadcast to RGB. (bg_darkness=0, its default.)"""
    bg_norm = (sulc - sulc.min()) / (sulc.max() - sulc.min() + 1e-8)
    return np.column_stack([1 - bg_norm] * 3)


def to_pyvista_mesh(coords: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    # PyVista's face format: each row is [n_points, i0, i1, i2, ...].
    n = faces.shape[0]
    pv_faces = np.column_stack([np.full(n, 3), faces]).ravel()
    return pv.PolyData(coords, pv_faces)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output.png>")
    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    coords, faces = half_inflate(fsaverage, "left")
    sulc = surface.load_surf_data(fsaverage["sulc_left"])
    colors = bg_rgb_from_sulc(sulc)

    mesh = to_pyvista_mesh(coords, faces)
    mesh["colors"] = colors

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=[900, 900])
    pl.add_mesh(mesh, scalars="colors", rgb=True, smooth_shading=True, ambient=0.3)
    pl.set_background("white")
    pl.view_vector([-1, 0, 0], viewup=[0, 0, 1])  # TRIBE v2's own "left" lateral camera
    pl.camera.zoom(1.3)
    pl.screenshot(str(out_path), transparent_background=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
