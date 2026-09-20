# Sightline

Sightline reads a presentation the way three different audiences would. Three model personas
(a **novice**, a **peer** and an **expert**), separated only by what they already know, each read
every slide and report what they took away. The expert's takeaway, verbatim, is the slide's
intent, and the novice and the peer are measured against exactly that. From that, each slide gets
a tier (what it demands of its reader) and concrete, evidence-quoting edits for the novice and the
peer.

> **Live: the expert's `takeaway` is the alignment reference**, and it is the very string shown as
> the slide's intent, so what the page shows is what was measured. `intent.py`, which rephrases
> it, is kept in the repo but is not used by the pipeline.

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
   `unconfirmed`; edit it, and the adjacent field the peer comes from. You may also give a
   **declared intent** (one sentence, optional): it is stored and shown on the deck overview, and
   is not used for alignment or shown on the slide page. Start the review.
2. **Deck overview** fills in as slides land (about 6 s each, in order). It lists the *hardest
   slides for a newcomer*, the terms the novice could not resolve across the deck, and the
   narrative arc: how far each audience falls below the intended reading as the deck goes on.
3. **Slide detail**: the slide on the left, on a light card in both themes. Directly under it, one
   block, *Intent of this slide*, shows the expert persona's takeaway **verbatim** (exactly as
   returned: no truncation or tidying); that is the string the novice and peer were measured
   against. On the right, the slide's **tier** and what it was read from, then three audience
   cards. The expert card does not repeat its takeaway. The novice and peer cards each carry *What
   to change*: concrete edits, each quoting the persona's report or the slide. They are made the
   first time you open a slide (about 5 s) and saved with the run. Click any number for the texts it
   was computed from.
4. **History** is the front page's list of saved runs. Runs save automatically, slide by slide,
   to `data/history/<run_id>/` as plain JSON and images. Opening one makes no API calls, so it
   works with no network and no key, and recommendations you opened before saving replay too. The
   bundled **Sample: serving LLMs faster** run works the same way and is your demo insurance.
   Runs saved before per-slide intent existed open in a reduced form with a notice. Runs saved
   while the intent was a rephrased sentence still show that sentence (with its attribution), because
   it is what their alignment was measured against.

### Themes

Light and dark. The header toggle switches on every screen; until you use it the page follows
your system setting. The choice is saved in `localStorage` (guarded: it throws in private windows,
and then lasts for that page only) and applied by a small blocking script in `index.html`'s
`<head>`, so there is no flash of the wrong theme. Every colour is a token in
`frontend/style.css`; slide images always sit on a light neutral card and are never filtered,
inverted or dimmed, so they look as they will when projected.

### Layout

| Path | What it is |
|---|---|
| `backend/sightline/audiences.py`, `divergence.py`, `llm.py` | Step 1: the personas, the metrics, the only code that talks to the Anthropic SDK |
| `backend/sightline/ingest.py` | PDF to per-slide records behind `parse(path)`; subfield inference |
| `backend/sightline/intent.py` | **Kept, unused.** Rephrases the expert's takeaway into a sentence (Haiku, checked). Not in the pipeline: the takeaway itself is the intent |
| `backend/sightline/tiers.py` | The three tiers. **Every threshold is in `CONFIG` at the top** |
| `backend/sightline/deck.py` | Deck rollup: pure arithmetic over per-slide results, no model call |
| `backend/sightline/diagnose.py` | Recommendations: one Sonnet call per slide, generated lazily, evidence enforced in code |
| `backend/sightline/runner.py`, `store.py` | Runs a deck and saves each slide as it lands |
| `backend/sightline/server.py` | FastAPI: upload, start, poll, replay. Streaming is polling |
| `frontend/` | Plain HTML, CSS and ES modules. No build step. `frontend/smoke/render.mjs` renders the real views in Node for the tests |
| `backend/fixtures/runs/` | Bundled, read-only sample runs (`make sample` rebuilds them; about 36 API calls) |

## Tiers

A tier says what a slide demands of its reader. It is a label, not a grade, and it is drawn by
fill weight, never by a red/amber/green colour.

| Tier | Meaning |
|---|---|
| **Self-contained** | A newcomer recovers the intended point. |
| **Background needed** | Needs some familiarity with the field; a newcomer drifts. |
| **Expert-gated** | Only a specialist recovers the intended point. |

Assigned from, in order of weight: the novice's alignment to the slide's intent, the
peer's alignment to it, and the novice's unresolved-term count.

- **Self-contained**: novice and peer both align, and the novice's unresolved terms are low.
- **Background needed**: the peer aligns and the novice does not, or the novice aligns but meets
  many unknown terms.
- **Expert-gated**: neither the novice nor the peer aligns.

**Relative to the deck first.** Each input becomes a percentile among the deck's scored slides,
and the thresholds apply to that percentile. Only when a deck has fewer than five scored slides
(or in a single-slide view) are the absolute thresholds used, and the tier's provenance panel says
which basis it used. Because tiers are relative, they can shift while a run is still streaming, and
some slides of any deck will land in the upper tiers: a tier says where a slide sits among *its
own deck's* slides. All thresholds are in one dict, `CONFIG` in `backend/sightline/tiers.py`; tune
them there against real decks.

## The expert baseline is definitional, not measured

A slide's intent is the expert's own takeaway, and the novice and the peer are measured against
it. So the expert's alignment is 1.0 by construction. It is a definition, not a score, and the UI
never shows it as one: the expert card says **reference**, with a note, and the arc draws the expert
on the reference line, thicker than the measured series and labelled as definitional today. The
series is kept so that when the expert becomes an independently measured model, the chart, the
payload and the legend do not change shape (`EXPERT_IS_DEFINITIONAL` in `intent.py`). The blind-spot
score (expert minus novice alignment) was retired for the same reason: with the expert pinned to the
reference it equals one minus the novice's alignment and adds nothing.

## Latency

The batch is three Sonnet persona calls per slide and nothing else, because the intent is the
expert's takeaway rather than a further model call. Measured on the bundled 7-slide sample: **3.7 s
per slide on average, 4.6 s at worst** (the earlier design, which added a Haiku call to rephrase the
takeaway, measured 5.6 s and 9.7 s). Recommendations are a separate Sonnet call made only when a
slide is opened (about 5 s), then cached with the run.

## What these numbers do and do not mean

- **Alignment is semantic similarity, the weaker instrument.** It measures topic and phrasing,
  not whether two sentences make the same claim. Where the unresolved-term count disagrees with
  alignment, believe the term count: it separated the audiences most cleanly in testing. The
  overview's ranking orders by alignment first, as specified, and shows the term count beside
  every row for that reason.
- **The level on one slide is not a verdict.** The same slide moves between runs by about as
  much as slides differ from each other. Trust how slides *separate* inside a deck. That is why
  every value is shown against the deck's own distribution, and why a deck needs five scored
  slides before any slide is compared with it.
- **A bare number never implies good or bad.** It always sits next to the deck's median, range
  and rank, and there are no ratings: no stars, letter grades, 0-100 scores or traffic lights.
- **Every number resolves to text.** If a value cannot show what produced it, it is not shown.
- **A bad model reply is an error, not a value.** A reply that fails validation, such as a
  confidence outside 0-1, is shown as a visible error and never clamped or repaired. That slide
  gets no alignment, tier or recommendations, and the overview lists which slides were affected.
  (Confidence itself is still stored in each run's JSON for debugging; the UI no longer shows it.)
- **A recommendation always quotes.** Each bullet carries a verbatim quote from that persona's
  report or the slide; a bullet whose quote cannot be found word for word is dropped, not shown.
  There are none for the expert: it defines the reference. What the expert itself flags, an
  unresolved term or a claim the slide does not support, is shown separately as *Expert also
  flagged*.
- **The personas are simulated readers.** They say what someone with a given background would
  probably take from a slide. They are not your audience, and one run is one sample.
