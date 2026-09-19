# Sightline

Sightline reads a presentation the way three different audiences would. Three model personas
(a **novice**, a **peer** and an **expert**), separated only by what they already know, each read
every slide and report what they took away. A metrics layer then measures how far apart those
readings are, and how far each is from what the presenter said they meant.

Every number on screen is clickable and resolves to the exact text that produced it.

## Setup

Needs Python 3.11 and, to start a *new* review, an Anthropic API key. Saved runs open without one.

```bash
cp .env.example .env        # then put your key in ANTHROPIC_API_KEY=
make setup                  # creates .venv and installs the backend (idempotent)
make dev                    # http://localhost:8000
```

`make dev PORT=9000` changes the port. `make test` runs the offline suite (no key needed);
`make gate` runs the live go/no-go check against the real model.

### Using it

1. **Upload** a PDF. The deck's subfield is inferred once by the model and shown as
   `unconfirmed`; edit it, and the adjacent field the peer comes from, then write the deck's
   **declared intent** (one sentence, required). Start the review.
2. **Deck overview** fills in as slides land (about 5 s each, in order). It shows the terms the
   novice could not resolve across the deck, slides ranked by novice-expert gap, the slides
   where the audiences differ most relative to this deck, and the alignment arc.
3. **Slide detail**: the slide on the left, three audience cards on the right. Click any number
   for the texts it was computed from. The *Recommendations* tab is a contract-only fixture and
   is badged `sample data`.
4. **History** is the front page's list of saved runs. Runs save automatically, slide by slide,
   to `data/history/<run_id>/` as plain JSON and images. Opening one makes no API calls, so it
   works with no network and no key. The bundled **Sample: serving LLMs faster** run works the
   same way and is your demo insurance.

### Layout

| Path | What it is |
|---|---|
| `backend/sightline/audiences.py`, `divergence.py`, `llm.py` | Step 1: the personas, the metrics, the only code that talks to the Anthropic SDK |
| `backend/sightline/ingest.py` | PDF to per-slide records behind `parse(path)`; subfield inference |
| `backend/sightline/deck.py` | Deck rollup: pure arithmetic over per-slide results, no model call |
| `backend/sightline/runner.py`, `store.py` | Runs a deck and saves each slide as it lands |
| `backend/sightline/server.py` | FastAPI: upload, start, poll, replay. Streaming is polling |
| `backend/sightline/diagnose.py` | Recommendations **contract only**; returns a checked-in fixture |
| `frontend/` | Plain HTML, CSS and ES modules. No build step |
| `backend/fixtures/runs/` | Bundled, read-only sample runs (`make sample` rebuilds them; about 22 API calls) |

## What these numbers do and do not mean

These rules come from the step 1 gate report. The UI is built to follow them, and so should
anything you build on it.

- **Confidence is the model's self-report, not a measurement.** Each card says so. It is how
  sure the persona claims to be that it understood the presenter; nothing checks that claim.
- **One slide's blind-spot score is not a verdict.** The same slide swings by about 0.3 between
  runs. Trust how slides *separate* from each other inside a deck, never the level on one slide.
  That is why every value is shown against the deck's own distribution, and why a deck needs
  five scored slides before any slide is called high or low relative to it.
- **Absolute divergence means little on its own.** A bare number never appears with an implied
  good or bad direction; it always sits next to the deck's median, range and rank.
- **There are no ratings.** No stars, no letter grades, no 0-100 score, no traffic lights. The
  personas return none, and the UI does not invent a scoring layer. (Recommendation severity is
  shown as a label, not a colour.)
- **Every number resolves to text.** If a value cannot show what produced it, it is not shown.
- **A bad model reply is an error, not a value.** A confidence outside 0-1 (or any reply that
  fails validation) is shown as a visible error, never clamped or repaired. The slide gets no
  cross-audience metrics, and the overview lists which slides were affected.
- **Semantic similarity is the weaker instrument.** It measures topic and phrasing, not whether
  two takeaways make the same claim. Where the count of unresolved terms and the confidence
  disagree with divergence, believe the term count and the confidence: they separated the
  audiences most cleanly in testing, so they get the visual weight.
- **The personas are simulated readers.** They say what someone with a given background would
  probably take from a slide. They are not your audience, and one run is one sample.
