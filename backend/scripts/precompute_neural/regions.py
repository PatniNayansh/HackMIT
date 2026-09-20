"""Language vs. visual vertex masks on fsaverage5, and the GFP negative-baseline readout.

The spec's question per slide (5): "does it drive language regions, or only early visual?"
That needs a mapping from the model's 20484 fsaverage5 vertices to "language network" and
"visual network" -- TRIBE's own output does not come with one built in.

We use the Destrieux 2010 atlas (nilearn.datasets.fetch_atlas_surf_destrieux), which labels
each vertex with an anatomical gyrus/sulcus name, and group by name substring:
  * visual:   occipital gyri/sulci and the calcarine sulcus (primary visual cortex, V1)
  * language: superior temporal gyrus/sulcus and inferior frontal gyrus (classic
              Wernicke's/Broca's-area territory)

This is a first-pass, defensible-but-unvalidated grouping by anatomical proxy, not a
functional localizer run on this data. Say so wherever processing_ratio is shown (spec 5:
"a proposed readout, not a validated metric") -- this module is exactly the part that
makes it a proxy rather than a measurement.
"""

from __future__ import annotations

import numpy as np

# Destrieux label substrings, matched case-sensitively against the atlas's own naming
# (e.g. "G_occipital_middle", "S_calcarine", "G_temp_sup-Lateral", "G_front_inf-Opercular").
_VISUAL_SUBSTRINGS = ("occipital", "calcarine", "cuneus", "lingual")
_LANGUAGE_SUBSTRINGS = ("temp_sup", "front_inf", "pariet_inf-Supramar")


def _label_names(labels: list[bytes] | list[str]) -> list[str]:
    return [l.decode() if isinstance(l, bytes) else l for l in labels]


def _mask_from_substrings(vertex_labels: np.ndarray, label_names: list[str], substrings: tuple[str, ...]) -> np.ndarray:
    matching_ids = {i for i, name in enumerate(label_names) if any(s in name for s in substrings)}
    return np.isin(vertex_labels, list(matching_ids))


def fetch_region_masks() -> dict[str, np.ndarray]:
    """(left mask, right mask) concatenated to one fsaverage5-length boolean array each,
    for "visual" and "language". Downloads the Destrieux atlas via nilearn on first call
    (cached under ~/nilearn_data after that -- consistent with this project's "fully
    offline after first download" convention for the sentence-transformers model)."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    names = _label_names(atlas["labels"])
    vertex_labels = np.concatenate([atlas["map_left"], atlas["map_right"]])

    return {
        "visual": _mask_from_substrings(vertex_labels, names, _VISUAL_SUBSTRINGS),
        "language": _mask_from_substrings(vertex_labels, names, _LANGUAGE_SUBSTRINGS),
        "dmn": fetch_dmn_mask(),
    }


# Yeo et al. 2011 (J Neurophysiol), the standard 7-network functional parcellation.
# Network 7 is their own Default Mode Network label -- this is the field's own numbering,
# not a choice made here. ("Background" occupies index 0: medial-wall / non-cortex
# vertices the parcellation doesn't cover.)
_DMN_NETWORK_INDEX = 7


def fetch_dmn_mask() -> np.ndarray:
    """A boolean mask over the same 20484 fsaverage5 vertices TRIBE v2 outputs, True where
    Yeo 2011's 7-network atlas assigns a vertex to the Default Mode Network.

    Unlike `fetch_region_masks`'s Destrieux-based masks, the Yeo atlas ships as a MNI152
    VOLUME, not a surface parcellation -- there is no off-the-shelf fsaverage5-surface
    version in nilearn. `nilearn.surface.vol_to_surf` (nearest_most_frequent, the standard
    pattern in nilearn's own docs for projecting a volumetric atlas onto a surface mesh) is
    used to bring it onto the same mesh. This is one more layer of approximation than the
    Destrieux masks have (a volume-to-surface projection, not a native surface
    parcellation) -- say so anywhere DMN drive is shown, on top of the existing
    "proposed readout, not a validated metric" disclosure.

    Verified empirically before shipping (not just assumed to work): projecting onto
    fsaverage5 produces exactly 20484 labelled vertices, and network 7 (DMN) covers ~17%
    of cortex -- both consistent with the published parcellation, not an empty or
    all-cortex mask.
    """
    from nilearn import datasets, surface

    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    proj_left = np.asarray(
        surface.vol_to_surf(yeo.maps, fsaverage["pial_left"], interpolation="nearest_most_frequent")
    ).squeeze()
    proj_right = np.asarray(
        surface.vol_to_surf(yeo.maps, fsaverage["pial_right"], interpolation="nearest_most_frequent")
    ).squeeze()
    network = np.round(np.concatenate([proj_left, proj_right])).astype(int)
    return network == _DMN_NETWORK_INDEX


def region_drive(response: np.ndarray, mask: np.ndarray) -> float:
    """Mean predicted response over a region's vertices, averaged over time. `response` is
    (T, 20484); `mask` is a boolean array of length 20484."""
    if response.shape[1] != mask.shape[0]:
        raise ValueError(f"response has {response.shape[1]} vertices, mask has {mask.shape[0]}")
    if not mask.any():
        raise ValueError("region mask is empty; the atlas label substrings matched nothing")
    return float(response[:, mask].mean())


def global_field_power(response: np.ndarray) -> float:
    """The scalar "engagement" readout arXiv 2607.01400 found does not correlate with real
    attention data (spec 2.2). Standard deviation across vertices at each timepoint (the
    classic GFP definition), averaged over time to one number. NEVER present this as a
    finding -- it is stored only so the Methods panel can reproduce the null result next to
    its citation, per spec design rule 3."""
    return float(response.std(axis=1).mean())
