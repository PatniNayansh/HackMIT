# Sightline

Sightline reads a presentation the way three different audiences would. Three model personas
(a **novice**, a **peer** and an **expert**), separated only by what they already know, each read
every slide and report what they took away. The expert's takeaway, verbatim, is the slide's
intent, and the novice and the peer are measured against exactly that. From that, each slide gets
a tier (what it demands of its reader) and concrete, evidence-quoting edits for the novice and the
peer.

> **Live comparator: `fieldwise`** (`compare.py`). Each takeaway is structured into fields and each
> field is compared with the comparator that suits it; the claim is checked by proposition coverage and
> reported as a *state*, not a score. The original whole-takeaway cosine (`divergence.py`) is intact and one
> setting away: `SIGHTLINE_COMPARATOR=cosine make dev`. See [Comparators](#comparators).
>
> The expert's `takeaway`, verbatim, is the slide's intent under both, and it is the very string
> shown under the slide, so what the page shows is what was measured. `intent.py`, which rephrases
> it, is kept in the repo but is not used by the pipeline.

Every value on screen is clickable and resolves to the exact text that produced it.

## Setup

Needs Python 3.11 and, to start a *new* review, an OpenAI API key. Saved runs open without one.

```bash
cp .env.example .env        # then put your key in OPENAI_API_KEY=
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
   goes on (the number of the expert's propositions each audience covered).
3. **Slide detail**: the slide on the left, on a light card in both themes. Directly under it, one
   block, *Intent of this slide*, shows the expert persona's takeaway **verbatim**; if the slide
   has a chart or diagram, a collapsed *Figure description (machine-generated)* sits beneath it. On
   the right, a line saying what shape the slide has (*"This slide names a principle and works to
   a numeric result."*), then the slide's **findings** and **tier**, then three audience cards.
   Each novice and peer card leads with the **state** of their claim in words, then the four fields
   beside the expert's, then unresolved terms and *What to change*: concrete edits, each quoting
   the persona's report or the slide (made the first time you open a slide, about 5 s, and saved
   with the run). Click any state, field, finding or number for the texts it came from. Last on
   the page, a **predicted neural response** to the slide, if one was precomputed for it — see
   [The neural layer](#the-neural-layer).
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
| `backend/sightline/audiences.py`, `divergence.py`, `llm.py` | Step 1: the personas, the cosine metrics, step 1's code. `llm.py` was rewritten for OpenAI (see [Models](#models)); the other two are untouched |
| `backend/sightline/compare.py` | **The field-wise comparator**: structuring call, null table, comparators, proposition coverage and its states, findings |
| `backend/sightline/ingest.py` | PDF to per-slide records behind `parse(path)`; subfield inference; which slides carry a figure and its neutral description |
| `backend/sightline/intent.py` | **Kept, unused.** Rephrases the expert's takeaway into a sentence (the `intent` model in `llm.CONFIG`). Not in the pipeline: the takeaway itself is the intent |
| `backend/sightline/tiers.py` | The three tiers, for either comparator. **Every threshold is in `CONFIG` at the top** |
| `backend/sightline/deck.py` | Deck rollup: pure arithmetic over per-slide results, no model call |
| `backend/sightline/diagnose.py` | Recommendations: one call per slide on the persona-tier model, generated lazily, evidence enforced in code |
| `backend/sightline/runner.py`, `store.py` | Runs a deck and saves each slide as it lands |
| `backend/sightline/server.py` | FastAPI: upload, start, poll, replay. Streaming is polling |
| `frontend/` | Plain HTML, CSS and ES modules. No build step. `frontend/smoke/render.mjs` renders the real views in Node for the tests |
| `backend/sightline/neural.py` | The neural layer's read-only half: the output contract, the arithmetic, and `CachedNeural`. Imports no ML library and makes no GPU call |
| `backend/scripts/precompute_neural/` | The other half: the CLI that actually runs TRIBE, in its own virtualenv on a CUDA machine. See [The neural layer](#the-neural-layer) |
| `backend/sightline/saliency.py`, `scanpath.py` | **Library only, not wired into the app.** Bottom-up visual saliency (OpenCV spectral residual) and a greedy fixation order over it |
| `backend/sightline/fix.py` | **Library only, not wired into the app.** The revise-and-rescore loop: propose a revision, have the same three audiences read it cold, rescore with the same metrics |
| `backend/sightline/research_lens.py` | **Library only, not wired into the app.** Pairs `dmn_drive` (deck-relative) with cited fMRI findings for named populations. Reads as a hypothesis, never as a diagnosis |
| `docs/` | The spec (the source of truth for scope), the design system the UI follows, the GX10 runbook, and the neurodivergent fine-tune research record |
| `backend/fixtures/runs/` | Bundled, read-only sample runs (`make sample` rebuilds them; about 50 API calls. `backend/scripts/restructure_sample.py` re-derives only the comparison and recommendations from the stored readings; about 16 calls) |

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
fields in the slide's profile only**: the claim's state (how many of the expert's propositions were covered), whether the concept and the result were
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
| **`fieldwise`** (live) | `compare.py` | three persona calls, then one structuring call (`structuring` role), plus one figure description at upload if the slide carries a figure (`helper` role) |
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
**proposition coverage**, reported as one of five states.

The expert's claim is broken into 1 to 3 atomic propositions (in the same structuring call, so no
extra call and no extra model). For each proposition the reader's claim is judged `covered`,
`omitted` or `contradicted`, and any assertion in the reader's claim that the expert's does not
make is listed as an extra. A proposition is `covered` or `contradicted` **only with a span found
word for word in the reader's claim**; the code checks this, and a judgement without such a span is
downgraded to `omitted` and recorded. The state is then derived from the judgements, in this order:

| Condition | State |
|---|---|
| Any proposition `contradicted` (checked first: it outranks everything) | **divergent**: likely a misconception |
| All `covered`, no extra assertions | **equivalent**: same understanding |
| All `covered`, plus an assertion the expert did not make | **over-claimed**: over-generalised |
| Some `covered`, none `contradicted` | **under-specified**: the reader got part of the point |
| None `covered` (including a reader who states no general claim) | **absent** |

`divergent` therefore needs positive evidence of a contradiction; it is no longer the fallthrough
for "no clean match". A reader who covers two propositions and contradicts a third is divergent,
not partly right.

**Coverage permits definitional paraphrase, and does not permit new assertions.** The prompt states
the rule: two statements are the same proposition when a reader who understood one would assent to
the other, and a standard definition of a named concept counts as the same proposition ("cost-benefit
analysis" and "weighing benefits against costs" are one proposition stated two ways). A new claim, a
different quantity for the same measure, a different direction or scope, or an assertion the expert
did not make is not covered. A reader who is vaguer than the expert has omitted the proposition; only
a negation or a conflicting value contradicts it. The prompt carries a worked example: expert "cost-benefit
analysis applies to the college decision, challenging the standard case for college", novice "cost-benefit
thinking applies to the decision of going to college by weighing benefits against opportunity costs":
p1 covered, p2 omitted, so **under-specified**, and the slide page says
"Missed: *it challenges the standard case for college*": the novice took away the method and missed the argument.

Where a slide is under-specified, or the reader is `absent`, the slide page names each missed proposition on the
audience card, the provenance panel shows the proposition table (each proposition, its status, the
quoted span for the ones that were covered, and the extras), and the missed propositions go to the
recommender as a trigger. Every state traces to a quoted span in its provenance panel, next to both claims and
both takeaways.

**A local tripwire, never a scorer.** `sentence-transformers` already runs locally for the
original comparator. If the model calls a pair `divergent` while the two claims sit above a cosine of
`TRIPWIRE` (0.75, in `CONFIG["tripwire"]` in `tiers.py` beside the tier thresholds), the model is
asked once more, told to name the contradicted proposition. If none comes back contradicted the state
is downgraded per the table above, and the disagreement is logged in the run's `structuring.tripwire`
and shown in the state's provenance panel. This costs nothing on the common path and fires only on
suspicious pairs. Similarity is never used to score anything: it cannot see negation, and negation
is where misconceptions live (the step-1 gate measured a contradiction at 0.94 against the claim it
contradicts).

**Two findings**, deterministic field logic with no model, shown as prominently as a score once
was, and passed to the recommender as triggers: **example-bound** (no general claim, no concept, and
no result where the slide has one: the reader attached to the example, not the principle) and
**figure-dependent** (the expert's claim uses the figure and the reader's uses nothing in it).

**The narrative arc** plots a real quantity: propositions covered / propositions in the expert's
claim, with a `divergent` slide drawn at 0 whatever else was covered. "2 of 3 propositions covered" is
countable and traces to quoted text; the axis reads "propositions covered" and the tooltip gives the
count. It is not a similarity or a probability. A slide whose profile has at most one scored field is
drawn with a hollow marker and its profile is named in the tooltip: a thin slide is not a low-comprehension
slide. Runs saved before coverage keep their own four-rung ordinal chart and entailment panel; they are
not rewritten.

**`image_content` is machine-generated and deliberately non-interpretive.** At upload, one cost-tier
vision call (the `helper` role) per figure-bearing slide (chosen by image area, vector-path count, or almost no text
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

## The expert baseline is definitional, not measured

A slide's intent is the expert's own takeaway, and the novice and the peer are measured against
it. So the expert's alignment is 1.0 by construction. It is a definition, not a score, and the UI
never shows it as one: the expert card says **reference**, with a note, and the arc draws the expert
on the reference line, thicker than the measured series and labelled as definitional today. The
series is kept so that when the expert becomes an independently measured model, the chart, the
payload and the legend do not change shape (`EXPERT_IS_DEFINITIONAL` in `intent.py`). The blind-spot
score (expert minus novice alignment) was retired for the same reason: with the expert pinned to the
reference it equals one minus the novice's alignment and adds nothing.

## Models

`backend/sightline/llm.py` is the only module that talks to an LLM SDK (OpenAI, through the Responses
API with strict structured outputs). The model and the reasoning effort for every role are in the one
`CONFIG` dict at the top of that file, and each can be overridden from the environment (`.env.example`).
Effort is sent explicitly on every call: OpenAI's own default is `medium`, where the Claude setup ran at
`low`, so leaving it out would spend more and run slower without saying so.

| Role | Model | Effort | Used for |
|---|---|---|---|
| `persona` | `gpt-5.6-terra` | low | the three audience readers, and the profile inference and recommendations that share their client |
| `structuring` | `gpt-5.6-luna` | low | field extraction and proposition coverage (one call per slide) |
| `helper` | `gpt-5.6-luna` | low | the figure describer (vision, at upload) |
| `intent` | `gpt-5.6-luna` | low | `intent.py`, kept but unused |

Callers hand `llm.py` a JSON schema and get a plain dict back. The reply is validated inside `llm.py` (a
Pydantic model built from the same schema) and returned through `.model_dump()`, and nothing is repaired: a
confidence of 1.4 is handed on as 1.4 and rejected by the caller, never clamped. A refusal, a truncation, a
failed call, or a reply that is not valid JSON or does not match the schema raises `LLMError`. An API error
that quotes part of a key is scrubbed before it can reach a log, a retry note or the UI. Token counts
(`input_tokens`, `output_tokens`, which includes reasoning) are read from `response.usage` and logged as before.

**The persona gate was measured on Claude and has been re-measured on OpenAI.** The step-1 gate figures
(persona separation on the two hand-written slides) were measured on Claude Sonnet 5 at low effort, and the
same gate failed on Haiku 4.5, where the novice read like the expert: persona separation depends on the model.
`make gate-repeat N=5` on `gpt-5.6-terra`, five runs each (the criteria are in `tests/test_gate_live.py`):

| Gate test | terra, low (the default) | terra, medium |
|---|---|---|
| clear slide: all three audiences converge | 5/5 | 5/5 |
| jargon slide: the novice is lost, the expert is not | 3/5 | 5/5 |
| jargon slide diverges more than the clear slide | 3/5 | 3/5 |
| term gap recovers the planted jargon | 5/5 | 5/5 |
| within the 5 s per-slide budget | 5/5 (slowest call 3.5 s) | 2/5 (slowest call 7.0 s) |
| the expert is not faking confusion | 5/5 | 5/5 |

On Claude at low effort the calibration note in `tests/test_gate_live.py` records these separation tests as
stable, so this is a weaker result, not a like-for-like one. The two settings trade separation against latency
(`medium` reads the jargon slide more like a newcomer, and blows the 5 s budget on three runs of five). Nothing
above `terra` (`sol`, `astra`) was tried. The default in `CONFIG` is `low`, as specified; `medium` is one edit
away.

**Everything else on this page that quotes a latency or a model comparison was measured on Claude** (the bundled
sample was generated by Claude Sonnet 5 and Haiku 4.5 and is stored as such; it is read-only and does not need a
key). On OpenAI, one structuring call takes about 4.5 to 5 s on `luna` and about 2.5 s on `terra` at low effort,
and the college-decision regression case (below) came out `under-specified` with the missed proposition named 3
of 3 times on each. The disk cache does not key on the model, so cached persona readings from Claude stay valid
and offline mode keeps working.

## Latency

Per slide: three persona calls, then (field-wise) one structuring call that overlaps the *next* slide's
persona calls, so the per-slide time is close to the slower of the two, not their sum. Measured on Claude
on the bundled 8-slide sample: **5.4 s and 6.9 s per slide wall-clock** on two runs. Not yet re-measured
end to end on OpenAI; the gate's persona calls took 1.8 to 3.5 s at low effort (see above).
`SIGHTLINE_COMPARATOR=cosine` drops the structuring call. Figure descriptions are made once at upload, in
parallel with the subfield inference. The tripwire's re-ask, one more structuring-sized call, happens only for
a `divergent` pair that is also close in wording, which is rare. Recommendations are a separate call made
only when a slide is opened, then cached with the run.

## Look and feel

The interface follows `docs/DESIGN_SYSTEM.md`: an academic/editorial system, closer to a
university-press monograph than to a dashboard. Serif headings (Source Serif 4), sans body
(Source Sans 3), small uppercase labels in the margin, a 960px measure, corners capped at 4px,
and **no elevation at all** — separation is a 1px rule, never a shadow. Warm paper in light,
an archival reading room in dark.

Two places deviate from the system, both because a test caught them:

- Its secondary ink (`#5C5C5C`) misses the **7:1** floor this app holds quoted persona text to,
  so `--ink-2` is `#4e4c47`.
- Its tan and brick sit 23° apart, and the three persona series must stay **40° apart** to be
  told by hue rather than by position. The series are ochre, brick and forest — 49°, 6° and 156°.

The rest is enforced from the tokens themselves in `tests/test_theme.py`, including two tests
that exist to keep the system honest: depth is never an elevation layer, and nothing is
pill-shaped or rounder than 4px. The slide page is the one screen allowed a wider measure
(1240px), because neither the artefact nor the analysis reads well squeezed into half of 960.

## A real lecture, read by the model

The front page carries one artefact that is not a deck at all: 16 minutes of a real recorded
lecture (CEE 260 / MIE 273, UMass Amherst) run straight through TRIBE with no synthesis —
the language-drive timecourse with its peak marked, and the left lateral cortex for each
four-minute chunk beneath it.

Only language drive is plotted. That run had no visual input, so visual drive in it is model
noise around zero and `processing_ratio` divides by it; the chart the data cannot support is
the one that is not drawn. Regenerate it with:

```bash
python backend/scripts/make_lecture_timecourse.py
```

## The neural layer

A slide page also shows a **predicted** cortical response to that slide, from
[TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta AI), a brain-encoding model trained
on movie-watching fMRI. It is precomputed, never live, and every constraint below is enforced in
code rather than left to discipline.

**It is precomputed because it has to be.** One slide costs 6-13 GPU-minutes across three
transformer backbones, CUDA only. `sightline/neural.py` therefore contains no ML import at all: it
is the output contract, the arithmetic, and a read-only loader. The half that runs the model lives
in `backend/scripts/precompute_neural/`, in a separate virtualenv (TRIBE pins `numpy==2.2.6` and
`torch<2.7`, which cannot coexist with this package), and calls `assert_may_run_tribe()` first —
which refuses outright when `SIGHTLINE_PROCESS=server`, the variable `server.py` sets on import.
Nothing a web request can reach can start a GPU run.

**What it reports, and what it refuses to report.** The readout is *where* the predicted response
sits — language-region drive against visual-region drive, as `processing_ratio`, expressed as a
Z-score against the rest of its own deck. It is a proposed readout, not a validated metric, and it
says so wherever it appears. What it will not report is a single scalar collapsed from the whole
response as a stand-in for "engagement": that is a documented null result (arXiv 2607.01400, pooled
partial *r* = 0.058, *p* = 0.23 against real attention data). It stays on disk as
`gfp_negative_baseline`, for a Methods panel to reproduce next to its citation, and never reaches
an API response or a screen. A test enforces both halves of that.

**The bundled sample carries a real run.** `backend/fixtures/neural/sample-llm-serving/` is output
from an actual GX10 run: slides 1-7, four cortical surface renders and a `metrics.json` each. Slide
8 was added to the deck later and was never run, so it 404s and the page says so — the empty state
is an unlit surface, which stands for the absence and never for a result.

To compute more, on a CUDA machine (see `docs/GX10_SETUP.md` for the full path):

```bash
cd backend/scripts/precompute_neural     # its own venv, not the repo's
python run.py --run-id sample-llm-serving --out ../../fixtures/neural
```

It is resumable: a slide with a `metrics.json` is skipped unless `--force` is passed, so a crash
costs at most the slide in flight.

## What these numbers do and do not mean

- **(Cosine comparator only.) Alignment is semantic similarity, the weaker instrument.** It measures topic and phrasing,
  not whether two sentences make the same claim. Where the unresolved-term count disagrees with
  alignment, believe the term count: it separated the audiences most cleanly in testing. The
  overview's ranking orders by alignment first, as specified, and shows the term count beside
  every row for that reason.
- **A proposition is all or nothing.** Coverage counts whole propositions, so a reader who gets most of one
  (says \"an improvement in some throughput metric\" where the expert says \"a 2.4x improvement in p99
  goodput\") has omitted it, and the count does not give partial credit. How finely the expert's claim is split
  changes the count, and the model's judgement of \"the same proposition\" varies a little between runs; read
  the quoted spans in the provenance panel rather than the fraction alone.
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
- **The neural layer is a prediction about an average brain, not a measurement of yours.** TRIBE
  v2 was trained on people watching films, so narrated slides are out of distribution for it, and
  the narration it heard is synthetic text-to-speech of the slide's own text — not a real presenter,
  with none of the emphasis, pauses or asides a real talk has. The released model has no
  subject-conditioning at inference either: it produces one group-average output for everyone.
  Compare `processing_ratio` as ranks within one deck; the level on one slide means nothing on its
  own.
