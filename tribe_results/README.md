# TRIBE v2 precompute results

Real TRIBE v2 (Meta's brain-encoding model) runs on the GX10, organized by source.

## sample_deck/
The bundled `sample-llm-serving` sample deck (7 slides, "Serving LLMs faster"). Each
`slide_NN/` has 4 brain-surface PNGs (lateral/medial x left/right) and `metrics.json`
(language_drive, visual_drive, processing_ratio, gfp_negative_baseline,
narration_transcript). Narration was synthesized (gTTS) from the slide's extracted text,
then transcribed back (whisperx) for word-level timing -- see
docs/GX10_SETUP.md for why TRIBE needs that round trip even for text input.

## umass_lecture_slides/
A real user-supplied deck: "CEE 260/MIE 273: Probability and Statistics in Civil
Engineering, M1b: Summarizing Data" (Prof. Oke, UMass Amherst), 16 slides. Same format as
sample_deck/, same synthetic-narration approach (no real audio was used per-slide here).

## umass_lecture_audio_chunks/
The *real* recorded lecture audio (`lecture.mp3`, ~25 min) for the same UMass course, run
through TRIBE v2 directly -- no synthesis. A first attempt fed the whole 25-minute file
in one continuous pass, but TRIBE v2's text encoder computes *contextualized* word
embeddings (each word re-encodes the entire preceding transcript), so cost grows with
document length -- the full file was still running the word-embedding stage after 30+
minutes with an ETA over 30 hours. Killed and replaced with this chunked approach: the
audio was split into 2-minute segments (`chunk_00`..`chunk_12`, via ffmpeg), each run
independently so each chunk starts its own short context. Only chunks 0-3 (the first 8
minutes) were run before stopping deliberately -- chunks 4-12 were never started.

Each `chunk_NN/` has:
  - `metrics.json` -- whole-chunk averaged language_drive / visual_drive / processing_ratio.
    **`visual_drive` and `processing_ratio` are not meaningful here** -- this run has no
    visual input at all (audio-only), so `visual_drive` is pure model noise around zero;
    only `language_drive` reflects real signal.
  - `timecourse.json` / `timecourse.png` -- per-segment language_drive / visual_drive across
    the chunk's ~2 minutes (segment timestamps are exact if TRIBE's segment objects exposed
    an offset/start attribute; otherwise evenly approximated across the chunk's duration).
  - the same 4 brain-surface PNGs, averaged over the chunk.

## Reading these results

Every number here is **predicted by a model, not measured from a real brain** -- that's
disclosed on the images themselves ("Predicted response — simulated, not measured").
Compare `processing_ratio` as ranks *within* one source (e.g. across umass_lecture_slides'
16 slides), never as an absolute score, and never call it "engagement" -- the project spec
(`docs/SIGHTLINE_spec.md`) explicitly forbids that framing, because TRIBE v2's own scalar
engagement readout was a null result in the published paper (arXiv 2607.01400,
r ≈ 0.058, n.s.). `gfp_negative_baseline` in each metrics.json is stored only so that
result can be reproduced next to its citation, never shown as a finding. For the audio
chunks specifically, only `language_drive` is interpretable (see above).
