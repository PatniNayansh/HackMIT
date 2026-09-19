# Neural precompute CLI

Runs TRIBE v2 (Meta AI, [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2))
on a narrated slide and writes the result to `backend/fixtures/neural/<run_id>/<slide>/`,
where the FastAPI app's `sightline.neural.CachedNeural` reads it. This is the **only** place
in the repo that calls TRIBE — see `sightline/neural.py`'s module docstring for why it must
never run inside the web app (6–13 minutes of GPU time per slide, CUDA-only).

## Before you run anything

1. **This needs its own environment**, separate from `backend/.venv`. TRIBE v2 pins
   `numpy==2.2.6` exactly and `torch<2.7`; installing that into the main app's venv would
   break it.
   ```bash
   cd backend/scripts/precompute_neural
   python3.11 -m venv .venv        # TRIBE v2 requires Python >=3.11
   source .venv/bin/activate       # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Needs a CUDA GPU.** Confirmed no supported AMD/ROCm or CPU path — CPU technically
   runs but is expected to take hours per slide, not minutes. Run this on the ASUS GX10.
3. **Clone and install TRIBE v2 itself** (not on PyPI as of the last check):
   ```bash
   git clone https://github.com/facebookresearch/tribev2
   pip install -e tribev2
   ```
4. **Accept the gated Llama-3.2-3B license and get a HuggingFace token.** TRIBE v2's text
   encoder is `meta-llama/Llama-3.2-3B`, gated on HuggingFace. Nobody but the person running
   this can do this step:
   - Log into HuggingFace, accept the license at the `meta-llama/Llama-3.2-3B` model page.
   - Generate a token with read access to gated repos.
   - Either `huggingface-cli login` interactively, or put it in a `.env` file in this
     directory (`HF_TOKEN=hf_...`) -- git-ignored, `run.py` loads it automatically. Never
     put a token in a file that isn't git-ignored, and never in a commit or a chat.
5. **License note:** TRIBE v2's weights are CC-BY-NC-4.0 — non-commercial use only. Fine
   for a hackathon demo; flag it if this project goes anywhere past that.

## What `run.py` actually does, per slide

1. Reads the slide's extracted text and PNG from a saved run (`data/history/<run_id>/`).
2. `narrate.py` synthesizes spoken narration from that text (gTTS) — decks in this app are
   silent PDFs, and TRIBE gets almost nothing from a silent input (spec §5). This is
   synthetic narration, not a real presenter; that is disclosed in the Methods panel, not
   just here.
3. Builds a trivial "video" from the synthesized audio: the slide's own image held for the
   narration's duration (moviepy). Expect the video/V-JEPA2 encoder to contribute little
   here — a static image is not a movie — signal should come mostly from the audio and
   transcript encoders. That is the expected, disclosed shape of this input, not a bug.
4. Runs TRIBE v2 inference → `(T, 20484)` predicted response on the fsaverage5 surface.
5. `regions.py` aggregates that response over language- and visual-network vertices
   (Destrieux atlas via nilearn) into `language_drive` / `visual_drive`, and separately
   computes `gfp_negative_baseline` (global field power — the metric the null-result paper
   used; stored ONLY for the Methods panel, never surfaced as a finding).
6. Renders the four required surface views (lateral/medial × left/right) via
   `nilearn.plotting.plot_surf_stat_map`, colormap `cold_hot`, symmetric about zero, with
   "Predicted response — simulated, not measured" burned into each image.
7. Writes `metrics.json` + the four PNGs to `backend/fixtures/neural/<run_id>/<slide>/`.

**Resumable**: re-running skips any slide that already has a `metrics.json`, unless
`--force` is passed. A crash partway through a deck loses at most the slide in progress.

```bash
python run.py --run-id sample-llm-serving --out ../../fixtures/neural
python run.py --run-id sample-llm-serving --out ../../fixtures/neural --force  # redo all
python run.py --run-id sample-llm-serving --out ../../fixtures/neural --slides 3,7,12
```

## What's verified vs. what needs confirming against the real repo

Confirmed by research against the actual TRIBE v2 repo before this was written: the model
architecture, its dependency pins, the CUDA-only requirement, and that
`TribeModel.from_pretrained("facebook/tribev2")` is the loading entry point.

**Not independently verified** — `run.py`'s `_call_tribe()` is written from that same
research pass, not from having actually run the model (no CUDA GPU was available to test
against). Before a real run, check `_call_tribe()` against the real repo's README and
`tribe_demo.ipynb` and adjust the call signature if it differs — it is isolated into that
one function precisely so that fix is small and contained.
