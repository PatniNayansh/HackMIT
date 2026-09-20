# Sightline

Sightline reads a presentation the way three different audiences would. Three model personas
(a **novice**, a **peer** and an **expert**), separated only by what they already know, each read
every slide and report what they took away. The expert's takeaway, verbatim, is the slide's
intent, and the novice and the peer are measured against exactly that. From that, each slide gets
a tier (what it demands of its reader) and concrete, evidence-quoting edits for the novice and the
peer.

> **Live comparator: `fieldwise`** (`compare.py`). Each takeaway is structured into fields and each
> field is compared with the comparator that suits it; the claim is checked by entailment and reported
> as a *state*, not a score. The original whole-takeaway cosine (`divergence.py`) is intact and one
> setting away: `SIGHTLINE_COMPARATOR=cosine make dev`. See [Comparators](#comparators).
>
> The expert's `takeaway`, verbatim, is the slide's intent under both, and it is the very string
> shown under the slide, so what the page shows is what was measured. `intent.py`, which rephrases
> it, is kept in the repo but is not used by the pipeline.

Every value on screen is clickable and resolves to the exact text that produced it.

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
   narrative arc: how each audience's reading of the claim compares with the expert's as the deck
   goes on (an ordinal with four rungs, labelled as such).
3. **Slide detail**: the slide on the left, on a light card in both themes. Directly under it, one
   block, *Intent of this slide*, shows the expert persona's takeaway **verbatim**; if the slide
   has a chart or diagram, a collapsed *Figure description (machine-generated)* sits beneath it. On
   the right, a line saying what shape the slide has (*"This slide names a principle and works to
   a numeric result."*), then the slide's **findings** and **tier**, then three audience cards.
   Each novice and peer card leads with the **state** of their claim in words, then the four fields
   beside the expert's, then unresolved terms and *What to change*: concrete edits, each quoting
   the persona's report or the slide (made the first time you open a slide, about 5 s, and saved
   with the run). Click any state, field, finding or number for the texts it came from.
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
| `backend/sightline/audiences.py`, `divergence.py`, `llm.py` | Step 1: the personas, the cosine metrics, the only code that talks to the Anthropic SDK. Untouched |
| `backend/sightline/compare.py` | **The field-wise comparator**: structuring call, null table, comparators, entailment states, findings |
| `backend/sightline/ingest.py` | PDF to per-slide records behind `parse(path)`; subfield inference; which slides carry a figure and its neutral description |
| `backend/sightline/intent.py` | **Kept, unused.** Rephrases the expert's takeaway into a sentence (Haiku, checked). Not in the pipeline: the takeaway itself is the intent |
| `backend/sightline/tiers.py` | The three tiers, for either comparator. **Every threshold is in `CONFIG` at the top** |
| `backend/sightline/deck.py` | Deck rollup: pure arithmetic over per-slide results, no model call |
| `backend/sightline/diagnose.py` | Recommendations: one Sonnet call per slide, generated lazily, evidence enforced in code |
| `backend/sightline/runner.py`, `store.py` | Runs a deck and saves each slide as it lands |
| `backend/sightline/server.py` | FastAPI: upload, start, poll, replay. Streaming is polling |
| `frontend/` | Plain HTML, CSS and ES modules. No build step. `frontend/smoke/render.mjs` renders the real views in Node for the tests |
| `backend/fixtures/runs/` | Bundled, read-only sample runs (`make sample` rebuilds them; about 50 API calls) |

## Tiers

A tier says what a slide demands of its reader. It is a label, not a grade, and it is drawn by
fill weight, never by a red/amber/green colour.

| Tier | Meaning |
|---|---|
| **Self-contained** | A newcomer recovers the intended point. |
| **Background needed** | Needs some familiarity with the field; a newcomer drifts. |
| **Expert-gated** | Only a specialist recovers the intended point. |

Under the cosine comparator, assigned from, in order of weight: the novice's alignment to the slide's
intent, the peer's alignment to it, and the novice's unresolved-term count.

- **Self-contained**: novice and peer both align, and the novice's unresolved terms are low.
- **Background needed**: the peer aligns and the novice does not, or the novice aligns but meets
  many unknown terms.
- **Expert-gated**: neither the novice nor the peer aligns.

Under the field-wise comparator the same three tiers are read from the new signals, **over the
fields in the slide's profile only**: the claim's state, whether the concept and the result were
reached, and the novice's unresolved-term count. The novice counts as aligned only if they reached
everything the slide has; the peer counts as aligned unless they reached none of it (an
under-specified claim is partial).

**Relative to the deck first.** Each input becomes a percentile among the deck's scored slides,
and the thresholds apply to that percentile. Only when a deck has fewer than five scored slides
(or in a single-slide view) are the absolute thresholds used, and the tier's provenance panel says
which basis it used. Because tiers are relative, they can shift while a run is still streaming, and
some slides of any deck will land in the upper tiers: a tier says where a slide sits among *its
own deck's* slides. (Under the field-wise comparator the states are categorical and are read directly; only the
novice's term count is a level, and it keeps this deck-relative reading.) All thresholds are in one dict, `CONFIG` in `backend/sightline/tiers.py`; tune
them there against real decks.

## Comparators

`SIGHTLINE_COMPARATOR` (or `create_app(comparator=...)`) chooses how a run's numbers are made, per
run, and each saved run remembers and shows the one that made it. Flipping it changes runs started
afterwards; nothing is re-run and nothing else changes.

| Value | Path | Model calls per slide |
|---|---|---|
| **`fieldwise`** (live) | `compare.py` | three persona calls, then one structuring call (Sonnet 5), plus one figure description at upload if the slide carries a figure (Haiku 4.5) |
| `cosine` | `divergence.py`, whole-takeaway cosine against the expert's takeaway | three persona calls |

**Why not cosine over the prose.** Takeaways that mean the same thing differ in wording, so cosine
reaches its number largely through shared proper nouns and phrasing, and says nothing about whether
the reader understood. Standardising the prose and re-running cosine makes it worse: the shared
scaffolding dominates and every pair lands near 1. So the takeaways are standardised **into fields**
and each field gets its own comparator.

**The fields.** Per audience, from each persona's takeaway (one call per slide, all three at once):
`concept` (the named principle), `claim` (what it establishes, in general terms, the example
stripped out), `result` (the numeric or factual outcome), `vehicle` (the example it leaned on). An
absent field is `null`, never a placeholder, and the extraction is checked in code: a concept the
persona never named, or a result it never reached, is dropped and recorded rather than kept.
`vehicle` is shown and **never scored**, so a reader is not penalised for talking about the
example. The slide-level `image_content` (below) is a different thing and is never compared.

**Null is a first-class value.** The expert defines the slide's shape: a slide has a field if and
only if the expert's takeaway populated it (its `slide_profile`, shown on the slide page). Per
field, reader against expert:

| Expert | Reader | Meaning |
|---|---|---|
| null | null | Not part of this slide. Excluded, never a gap |
| present | present | Compared with the field's comparator |
| present | null | **The gap**: the reader did not reach this part |
| null | present | `over-reach`: shown quietly, never penalised |

**Comparators.** `result`: exact match on the quantity it states (`2.4x improvement` and `2.4 times
faster` both state 2.4x), no model. `concept`: identity, then near matches (same words reordered, a
listed synonym, one name containing the other, mostly shared words, a small edit distance). `claim`:
bidirectional entailment, reported as one of four states, or `absent`:

| State | Meaning |
|---|---|
| **equivalent** | Both directions entail: same understanding |
| **over-claimed** | The reader's claim entails the expert's, not the reverse: over-generalised |
| **under-specified** | The expert's claim entails the reader's, not the reverse: a weaker version |
| **divergent** | Neither entails the other: likely a misconception |
| **absent** | The reader stated no general claim (no entailment is run) |

Every state traces to a quoted span in its provenance panel, next to both claims and both takeaways.

**Two findings**, deterministic field logic with no model, shown as prominently as a score once
was, and passed to the recommender as triggers: **example-bound** (no general claim, no concept, and
no result where the slide has one: the reader attached to the example, not the principle) and
**figure-dependent** (the expert's claim uses the figure and the reader's uses nothing in it).

**The narrative arc** maps the states to an ordinal purely so the chart can be drawn: equivalent
1.0, over-claimed 0.66, under-specified 0.33, divergent or absent 0.0. It is labelled ordinal, not
continuous; it is a rank with four rungs, not a similarity or a probability. A slide whose profile
has at most one scored field is drawn with a hollow marker and its profile is named in the tooltip:
a thin slide is not a low-comprehension slide.

**`image_content` is machine-generated and deliberately non-interpretive.** At upload, one Haiku 4.5
vision call per figure-bearing slide (chosen by image area, vector-path count, or almost no text
beside graphics, so a text-only deck costs nothing extra; cached, so once per deck) describes marks,
labels, axes, values and arrangement, and names no principle, draws no conclusion and expands no
acronym. The same string is given to all three personas under a `FIGURE:` marker, and shown on the
slide page, collapsed. If it explains what the figure *means* it has done the expert's job for the
novice and the readings converge, so a description that keeps interpreting is rejected. It is never
scored.

**The personas also receive the rendered slide image**, as they always have, so a persona can
recognise a familiar diagram from the picture whatever the description says. On a textbook
supply-and-demand chart even the novice persona named "supply and demand" and "equilibrium", with
the image and, in a test, with only the description (the labels and the lines' slopes were enough).
A persona is a model role-playing a background; it cannot un-know a well-known diagram. The
description does not cause this (it names no principle), and a less familiar figure will separate
the readers more. `SIGHTLINE_FIGURE_INPUT=description_only` withholds the image on slides that
have a description, so the description is the only way the figure reaches them; the default is
`image+description`.

**Models.** Structuring and entailment run on **Sonnet 5**, not Haiku: on hand-written directional
fixtures Haiku 4.5 called a plain paraphrase "under-specified" (3 of 4 correct) while Sonnet 5 got
4 of 4, and a false gap is exactly what this comparator exists to avoid. Set
`SIGHTLINE_STRUCTURING_MODEL=claude-haiku-4-5` to trade accuracy for speed.

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

Per slide: three Sonnet persona calls, then (field-wise) one Sonnet structuring call that overlaps
the *next* slide's persona calls, so the per-slide time is close to the slower of the two, not their
sum. Measured on the bundled 8-slide sample: **5.4 s and 6.9 s per slide wall-clock** on two runs
(personas about 3.5 to 4.5 s; structuring about 4.7 to 6.5 s, occasionally 12 to 15 s). That is at
the edge of a 7 s budget; if runs are slower for you, `SIGHTLINE_COMPARATOR=cosine` drops the
structuring call, or `SIGHTLINE_STRUCTURING_MODEL=claude-haiku-4-5` shortens it at some cost in
accuracy. The cosine path is three persona calls a slide (about 3.7 s). Figure descriptions are made
once at upload, in parallel with the subfield inference. Recommendations are a separate Sonnet call
made only when a slide is opened (about 5 s), then cached with the run.

## What these numbers do and do not mean

- **(Cosine comparator only.) Alignment is semantic similarity, the weaker instrument.** It measures topic and phrasing,
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
