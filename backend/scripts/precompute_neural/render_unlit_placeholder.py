#!/usr/bin/env python
"""One-off: render the real fsaverage5 cortical surface mesh with only anatomical shading
(sulcal depth) and NO stat map -- the "unlit" brain -- for the frontend's neural empty
state. This is a real anatomical mesh, the same one TRIBE v2's predicted response lands on
(fsaverage5, spec 8), rendered with nothing overlaid on it, so it reads unmistakably as
"no data" rather than a fake prediction.

Not part of the app's runtime: run this once, commit the resulting PNG to
frontend/img/, and nilearn/matplotlib never need to be installed for the FastAPI app
itself. Needs nilearn + matplotlib installed (not a listed dependency of backend/ or of
this directory's requirements.txt -- both are heavier ML-adjacent installs this repo
otherwise avoids at runtime; install them ad hoc to run this script, e.g.
`pip install nilearn matplotlib`).

    python render_unlit_placeholder.py ../../../frontend/img/brain-unlit.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, surface


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output.png>")
    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One hemisphere, not a two-hemisphere montage: nilearn's single fixed-direction
    # light source shades the left and right lateral views very differently (one washes
    # out), which isn't worth fighting for a decorative placeholder. A single lateral
    # view of the right hemisphere is still real fsaverage5 anatomy, still unlit (no
    # stat_map -- bg_map, sulcal depth, is the only thing coloring it), and reads fine on
    # its own as "a brain."
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    sulc_right = surface.load_surf_data(fsaverage["sulc_right"])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4.2))
    plotting.plot_surf(
        fsaverage["infl_right"], bg_map=sulc_right, hemi="right", view="lateral",
        cmap="Greys", avg_method="mean", axes=ax, figure=fig,
    )
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_box_aspect(None, zoom=1.5)  # crop in: default leaves a lot of dead margin
    fig.patch.set_alpha(0.0)
    fig.savefig(out_path, dpi=160, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    rgba = plt.imread(out_path)
    opaque_fraction = (rgba[..., 3] > 0.05).mean() if rgba.shape[-1] == 4 else 1.0
    print(f"wrote {out_path} ({rgba.shape[1]}x{rgba.shape[0]}, {opaque_fraction:.1%} non-transparent)")


if __name__ == "__main__":
    main()
