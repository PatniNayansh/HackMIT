// The drawer that answers "where does this number come from?".
//
// Everything here is read from the run payload: the exact strings a metric was computed from,
// the other side of the comparison, and the raw arithmetic. Nothing is derived that the payload
// does not already carry. `numBtn` is how a number gets onto the screen at all: a value with no
// provenance spec has no way to be rendered.
import { h, f2, ordinal, stateChip, STATE_MEANING, PERSONAS, LABEL, COLOR } from "./dom.js";
import { strip } from "./charts.js";

const KEY_LABEL = {
  term_gap_count: "Terms only the novice missed",
};
function keyLabel(key) {
  if (KEY_LABEL[key]) return KEY_LABEL[key];
  const [base, persona] = key.split(".");
  return `${{ unresolved_count: "Unresolved terms", intent_alignment: "Alignment to the intended reading" }[base]} \u2014 ${LABEL[persona]}`;
}
const ALIGN_CAVEAT = "Alignment is semantic similarity to the intended reading: the weaker instrument. It reads topic and phrasing, not whether the claim was understood. Trust where a slide sits among the deck\u2019s slides, not the level on one slide.";
const KEY_CAVEAT = {};
for (const p of ["novice", "peer"]) KEY_CAVEAT[`intent_alignment.${p}`] = ALIGN_CAVEAT;

const slideOf = (state, n) => state.results.get(n);

// Counts print as counts; every other metric is a two-decimal number.
const isCount = (key) => key.startsWith("unresolved_count") || key === "term_gap_count";
export const fmtFor = (key) => (isCount(key) ? (x) => (Number.isInteger(x) ? String(x) : x.toFixed(1)) : f2);

// -------------------------------------------------------------------- the drawer

let stack = [];
let drawerEl = null;
let lastFocus = null;

export function closeDrawer() {
  stack = [];
  drawerEl?.remove();
  drawerEl = null;
  document.removeEventListener("keydown", onKey);
  lastFocus?.focus?.();
  lastFocus = null;
}
function onKey(e) { if (e.key === "Escape") closeDrawer(); }

export function openProvenance(spec, state) {
  if (!drawerEl) { stack = []; lastFocus = document.activeElement; }
  stack.push(spec);
  paintDrawer(state);
}

function paintDrawer(state) {
  const spec = stack[stack.length - 1];
  const { title, body } = build(spec, state);
  const closeBtn = h("button", { class: "btn small", "aria-label": "Close", on: { click: closeDrawer } }, "Close");
  const node = h("div", null,
    h("div", { class: "scrim", on: { click: closeDrawer } }),
    h("aside", { class: "drawer", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("header", null,
        stack.length > 1 && h("button", { class: "btn small", on: { click: () => { stack.pop(); paintDrawer(state); } } }, "← Back"),
        h("h2", null, title),
        closeBtn),
      h("div", { class: "body" }, ...body)));
  drawerEl?.remove();
  drawerEl = node;
  document.body.append(node);
  closeBtn.focus();
  document.removeEventListener("keydown", onKey);
  document.addEventListener("keydown", onKey);
}

// ------------------------------------------------- numbers that carry their provenance

export function numBtn(text, spec, state, label) {
  return h("button", {
    class: "num", type: "button", "aria-label": label || `${text}. Show where this comes from.`,
    on: { click: (e) => { e.stopPropagation(); openProvenance(spec, state); } },
  }, text);
}

/** "median 0.55, range 0.20–0.71" and "3rd highest of 12", each opening the deck's values. */
export function cmpLine(state, key, slide, fmt = fmtFor(key)) {
  const dist = state.rollup?.distributions?.[key];
  const pos = state.rollup?.per_slide?.find((p) => p.slide === slide)?.position?.[key];
  if (!dist || !pos) return h("span", { class: "cmp" }, "No comparison available");
  if (dist.min === dist.max) return h("span", { class: "cmp" }, `Deck: ${fmt(dist.min)} on every slide read so far`);
  if (!dist.comparable) {
    return h("span", { class: "cmp" }, `No deck comparison yet: needs ${state.rollup.min_slides_for_comparison} scored slides, has ${dist.n}. `,
      numBtn("See values", { kind: "dist", key, slide }, state, "See the values so far"));
  }
  return h("span", { class: "cmp" },
    "Deck: ", numBtn(`median ${fmt(dist.median)}, range ${fmt(dist.min)}–${fmt(dist.max)}`, { kind: "dist", key, slide }, state), " · ",
    numBtn(`${ordinal(pos.rank)} highest of ${pos.of}`, { kind: "dist", key, slide }, state));
}

export function deckStrip(state, key, slide, color) {
  const dist = state.rollup?.distributions?.[key];
  if (!dist || !dist.comparable || !dist.values.some((v) => v.slide === slide)) return null;
  return strip(dist, slide, color);
}

// ------------------------------------------------------------------- shared pieces

function highlight(text, terms) {
  const lower = text.toLowerCase();
  const ranges = [];
  for (const t of terms) {
    const needle = t.toLowerCase();
    if (!needle) continue;
    for (let i = lower.indexOf(needle); i >= 0; i = lower.indexOf(needle, i + needle.length)) ranges.push([i, i + needle.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]); else merged.push([...r]);
  }
  const out = [];
  let at = 0;
  for (const [a, b] of merged) {
    out.push(text.slice(at, a), h("mark", { class: "hl" }, text.slice(a, b)));
    at = b;
  }
  out.push(text.slice(at));
  return out;
}
const missing = (text, terms) => terms.filter((t) => !text.toLowerCase().includes(t.toLowerCase()));

/** True when this slide's metrics came from the field-wise comparator (compare.py). Runs made by
 *  the original cosine comparator keep their own panels below. */
export const isFieldwise = (r) => r?.metrics?.comparator === "fieldwise";

/** True when this slide's intent IS the expert's takeaway (the current pipeline). Runs saved by the
 *  previous version measured alignment against a rephrased sentence instead, and keep showing it. */
export const usesTakeawayIntent = (r) => r.slide_intent?.source === "expert_takeaway";

/** A labelled, verbatim text box with a persona-coloured dot: the one "green-dot block" component. */
export function src(label, text, persona) {
  return h("div", { class: "src" },
    h("div", { class: "who" }, persona && h("span", { class: `dot ${persona}` }), label),
    h("p", null, text));
}
function srcNodes(label, nodes, persona) {
  return h("div", { class: "src" },
    h("div", { class: "who" }, persona && h("span", { class: `dot ${persona}` }), label),
    h("p", null, ...nodes));
}
const section = (title, ...kids) => h("section", null, h("h3", null, title), ...kids);
const calc = (text) => h("div", { class: "calc" }, text);
const val = (metric) => f2(metric.value);
const slideLink = (state, n, text) => h("a", { href: `#/run/${state.meta.run_id}/slide/${n}`, on: { click: closeDrawer } }, text ?? `Slide ${n}`);

function deckSection(state, key, slide, color) {
  return section("In this deck",
    h("div", { class: "stack" },
      deckStrip(state, key, slide, color),
      cmpLine(state, key, slide),
      KEY_CAVEAT[key] && h("p", { class: "caveat" }, KEY_CAVEAT[key])));
}

// ------------------------------------------------------------------------- builders

function build(spec, state) {
  const b = BUILDERS[spec.kind];
  return b(spec, state);
}

const REFERENCE_NOTE = "The intended reading is derived from this expert interpretation, so it defines the baseline rather than scoring against it.";

function derivation(si) {
  if (!si) return "No intended reading was produced for this slide.";
  const lines = [
    `written by: ${si.source === "model" ? si.model : "template (the expert\u2019s claim, framing removed)"}`,
    si.reason ? `why the template: ${si.reason}` : null,
    si.source === "model" ? `model calls: ${si.attempts}${si.cached ? " (served from cache)" : ""}` : null,
  ].filter(Boolean);
  return lines.join("\n");
}

function intentSources(r) {
  const si = r.slide_intent;
  return h("div", { class: "stack" },
    src("Intended reading, inferred from the expert reading (verbatim)", si.text),
    src("Expert takeaway it was derived from (verbatim)", si.derived_from.takeaway, "expert"),
    src("Expert claim it was derived from (verbatim)", si.derived_from.inferred_claim, "expert"));
}

// ------------------------------------------------- field-wise (compare.py) panels
// Everything here is read from the run payload: the claims and takeaways compared, the two directional
// answers and the quoted span that decided a state, the field values and how each comparator saw them.

const FIELD_NAME = { concept: "concept", claim: "claim", result: "result", vehicle: "vehicle (example)" };
const NULL_ROW = {
  excluded: (f) => `Neither the expert\u2019s takeaway nor this reader\u2019s had a ${f}, so it is not part of this slide. It is left out of the comparison and is never a gap.`,
  gap: (f) => `The expert\u2019s takeaway had a ${f} and this reader\u2019s did not: the reader did not reach this part. This is the gap, and it is a finding.`,
  over_reach: (f) => `This reader gave a ${f} that the expert\u2019s takeaway did not. It is surfaced quietly and never penalised: usually a misreading, occasionally a real catch, always worth a look.`,
  not_scored: () => "The example each reader leaned on to get to the point. It is recorded and shown, and never scored: a reader who talks about the example is not penalised for it.",
};

function statusWord(c) {
  if (c.status === "compared") return c.outcome ?? "compared";
  return { excluded: "not part of this slide", gap: "not reached", over_reach: "over-reach", not_scored: "not scored" }[c.status];
}
export { statusWord };

const yes = (b) => (b ? "yes" : "no");

const fw = {
  claimState(slide, persona, r) {
    const m = r.metrics, c = m.comparisons[persona].claim, v = c.verdict;
    const who = LABEL[persona];
    const dropped = (m.structuring.dropped || []).filter((d) => d.persona === persona || d.persona === "expert");
    return {
      title: `Claim \u2014 ${who}, slide ${slide}`,
      body: [
        h("div", null, stateChip(c.outcome, { large: true })),
        h("p", null, STATE_MEANING[c.outcome]),
        c.status === "gap"
          ? h("p", { class: "muted" }, "The reader\u2019s takeaway states no general claim, so no entailment was run: an absent claim is the strongest form of under-specified.")
          : h("p", { class: "muted" }, "Entailment is checked in both directions and reported as a state, not a score. This is the check the alignment number never made: whether the takeaway is right about the point."),
        section("The two claims compared",
          h("div", { class: "stack" },
            srcNodes("Expert claim (its takeaway, restated in general terms)", v?.quote_verified ? highlight(c.expert, [v.quote]) : [c.expert], "expert"),
            c.audience
              ? srcNodes(`${who} claim (its takeaway, restated in general terms)`, v?.quote_verified ? highlight(c.audience, [v.quote]) : [c.audience], persona)
              : src(`${who} takeaway`, "(states no general claim)", persona))),
        v ? section("The verdict", h("div", { class: "stack" },
          calc(`expert claim \u21d2 ${who.toLowerCase()} claim: ${yes(v.expert_entails_audience)}\n${who.toLowerCase()} claim \u21d2 expert claim: ${yes(v.audience_entails_expert)}\nmodel: ${m.structuring.model}`),
          v.rationale ? src("Rationale", v.rationale) : null,
          v.quote ? src(v.quote_verified ? "Quoted span that decided it (found word for word in the claims above)" : "Quoted span (not found word for word; both claims are shown in full above)", v.quote) : null)) : null,
        section("The takeaways they were extracted from (verbatim)",
          h("div", { class: "stack" }, src("Expert takeaway", m.takeaways.expert, "expert"), src(`${who} takeaway`, m.takeaways[persona], persona))),
        dropped.length ? h("p", { class: "caveat" }, `Fields the takeaways did not support were left empty, not filled: ${dropped.map((d) => `${d.persona} ${d.field} (${d.reason})`).join("; ")}.`) : null,
      ],
    };
  },

  field(slide, persona, field, r) {
    const m = r.metrics, c = m.comparisons[persona][field], who = LABEL[persona];
    const exp = m.takeaways.expert, mine = m.takeaways[persona];
    const detail = c.status === "compared" && c.comparator !== "entailment"
      ? calc(field === "result"
          ? `exact match after trivial normalisation (currency symbol, spaces, commas)\nexpert: ${c.normalised[0]}\n${who.toLowerCase()}: ${c.normalised[1]}\n=> ${c.outcome}`
          : `identity, then a fuzzy or synonym fallback\nexpert: ${c.normalised[0]}\n${who.toLowerCase()}: ${c.normalised[1]}\n${c.how}\n=> ${c.outcome}`)
      : null;
    return {
      title: `${FIELD_NAME[field][0].toUpperCase() + FIELD_NAME[field].slice(1)} \u2014 ${who}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, statusWord(c)),
        h("p", null, c.status === "compared" ? (c.outcome === "match" ? "The two values are the same." : c.outcome === "near" ? "The two values are close: same words in another order, a listed synonym, or a small edit." : "The two values differ.") : NULL_ROW[c.status](field)),
        section("The values",
          h("div", { class: "stack" },
            src("Expert", c.expert ?? "(none: the expert\u2019s takeaway had no " + field + ")", "expert"),
            src(who, c.audience ?? "(none: the takeaway had no " + field + ")", persona))),
        detail ? section("Raw computation", detail) : null,
        section("Where they come from (verbatim takeaways)",
          h("div", { class: "stack" },
            srcNodes("Expert takeaway", c.expert ? highlight(exp, [c.expert]) : [exp], "expert"),
            srcNodes(`${who} takeaway`, c.audience ? highlight(mine, [c.audience]) : [mine], persona))),
      ],
    };
  },

  finding(slide, index, r) {
    const f = r.metrics.findings[index];
    const m = r.metrics;
    const rule = f.id === "example_bound"
      ? calc([
          `rule (no model): the reader attached to the example, not the principle`,
          `the expert had a principle to miss: concept ${JSON.stringify(m.fields.expert.concept)}, claim ${m.fields.expert.claim ? "present" : "none"}`,
          `${f.audience} concept:  ${JSON.stringify(f.facts.concept)}`,
          `${f.audience} claim:    ${JSON.stringify(f.facts.claim)}  (a claim stated only in terms of the example is extracted as none)`,
          f.facts.result_in_profile ? `${f.audience} result:   ${JSON.stringify(m.fields[f.audience].result)}  (this slide has a result)` : `result: not part of this slide, so not required`,
        ].join("\n"))
      : calc([
          `rule (no model): the substance is in the figure and this reader did not engage with it`,
          `words the expert\u2019s claim shares with the figure description: ${f.facts.figure_terms.join(", ")}`,
          `words the ${f.audience}\u2019s claim shares with it: none`,
        ].join("\n"));
    return {
      title: `${f.id === "example_bound" ? "Example-bound" : "Figure-dependent"} \u2014 ${LABEL[f.audience]}, slide ${slide}`,
      body: [
        h("p", { class: "finding-text" }, f.text),
        h("p", { class: "muted" }, "A finding is deterministic field logic on the extracted fields, with no model call. It is not a score."),
        section("Why it fired", rule),
        section("The quoted evidence", h("div", { class: "stack" }, ...f.evidence.map((e) => src(e.label, e.text, e.persona)))),
        f.id === "figure_dependent" && r.image_content?.text
          ? section("The figure description (machine-generated; deliberately non-interpretive)", src("Given to all three readers under FIGURE:", r.image_content.text))
          : null,
      ],
    };
  },

  profile(slide, r) {
    const m = r.metrics, p = m.slide_profile, ef = m.fields.expert;
    const values = Object.entries(ef).filter(([, v]) => v);
    return {
      title: `The shape of this slide \u2014 slide ${slide}`,
      body: [
        h("p", null, p.text),
        h("p", { class: "muted" }, "The expert defines which fields this slide has: a slide has a result if and only if the expert\u2019s takeaway reached one. Every comparison, tier and finding here reads only these fields. A field the expert left out is not part of the slide, so a reader lacking it is never marked down for it."),
        srcNodes("Expert takeaway, with what was extracted highlighted", highlight(m.takeaways.expert, values.filter(([k]) => k !== "claim").map(([, v]) => v)), "expert"),
        section("Extracted from it", h("div", { class: "stack" }, ...Object.entries(ef).map(([k, v]) =>
          src(FIELD_NAME[k] + (p.scored.includes(k) ? " (scored)" : k === "vehicle" ? " (shown, never scored)" : ""), v ?? "(none: not part of this slide)", "expert")))),
        p.thin ? h("p", { class: "caveat" }, "This slide has at most one scored field, so it is drawn with a hollow marker on the arc: a thin slide is not a low-comprehension slide.") : null,
      ],
    };
  },

  reference(slide, r) {
    const m = r.metrics, p = m.slide_profile;
    return {
      title: `Reference \u2014 Expert, slide ${slide}`,
      body: [
        h("div", null, h("div", { class: "big" }, "reference"), h("span", { class: "pill outline" }, "definitional, not measured")),
        h("p", null, REFERENCE_NOTE),
        h("p", { class: "muted" }, "The expert\u2019s takeaway is the slide\u2019s intent, and its extracted fields define which fields the slide has. There is nothing to score it against. It stays in the payload and on the arc so they keep their shape for when the expert is measured independently."),
        section("What it defined", h("div", { class: "stack" },
          src("Expert takeaway, verbatim: the intent of this slide", m.takeaways.expert, "expert"),
          src("The slide\u2019s shape", p.text, "expert"))),
      ],
    };
  },

  tier(slide, r, t) {
    const m = r.metrics;
    return {
      title: `Tier \u2014 slide ${slide}`,
      body: [
        h("div", null, h("span", { class: `tier-chip large ${t.tier}` }, t.label)),
        h("p", null, t.meaning),
        h("p", { class: "muted" }, t.basis === "relative"
          ? `The claim states and the concept and result checks are read directly. Only the novice\u2019s unresolved-term count is a level, and it is read against the ${t.n_slides} scored slides of this deck.`
          : `The claim states and the concept and result checks are read directly. This deck has ${t.n_slides} scored slide${t.n_slides === 1 ? "" : "s"}, fewer than the ${t.min_slides_for_relative} needed to read a count against its deck, so the term count uses a rough absolute threshold.`),
        section("What drove it", h("div", { class: "stack" }, ...t.checks.map((c) => h("div", { class: "src" }, h("div", { class: "who" }, c.passed ? "\u2713 met" : "\u2717 not met"), h("p", null, c.text))))),
        section("The slide\u2019s shape", h("p", null, m.slide_profile.text)),
        section("The takeaways (verbatim)", h("div", { class: "stack" },
          src("Expert takeaway, the reference", m.takeaways.expert, "expert"),
          src("Novice takeaway", m.takeaways.novice, "novice"),
          src("Peer takeaway", m.takeaways.peer, "peer"),
          src(`Terms the novice could not resolve (${m.term_gap.novice_unresolved.length})`, m.term_gap.novice_unresolved.join(", ") || "None", "novice"))),
        section("The rule", calc("Over the fields the expert populated (claim, concept, result):\nSelf-contained:    novice reaches all of them, peer reaches them, novice unresolved terms low\nBackground needed: peer reaches them and the novice does not (or the novice does but meets many unknown terms)\nExpert-gated:      neither reaches them\n(peer counts as reaching them unless it reached none; an under-specified claim is partial)")),
      ],
    };
  },
};

const BUILDERS = {
  alignment({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const m = r.metrics.intent_alignment[persona];
    if (m.definitional) return BUILDERS.reference({ slide }, state);
    const tk = usesTakeawayIntent(r);
    return {
      title: `Alignment to the intended reading \u2014 ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, val(m)),
        h("p", { class: "muted" }, "How close, in topic and wording, this persona\u2019s takeaway is to what the slide is trying to establish. It does not check that the takeaway is right."),
        section("The two texts compared",
          h("div", { class: "stack" },
            tk ? src("Intent of this slide: the expert takeaway (verbatim)", m.inputs.intent, "expert")
               : src("Intended reading, inferred from the expert reading (verbatim)", m.inputs.intent),
            src(`${LABEL[persona]} takeaway (verbatim)`, m.inputs[persona], persona))),
        section("Raw computation", calc(`cosine similarity of two sentence embeddings\nmodel: ${r.scored_by}\ncos(intent, ${persona} takeaway) = ${m.value.toFixed(4)}`)),
        tk ? null : section("Where the intended reading came from", calc(derivation(r.slide_intent))),
        deckSection(state, `intent_alignment.${persona}`, slide, COLOR[persona]),
      ],
    };
  },

  reference({ slide }, state) {
    const r = slideOf(state, slide);
    if (isFieldwise(r)) return fw.reference(slide, r);
    const tk = usesTakeawayIntent(r);
    return {
      title: `Reference \u2014 Expert, slide ${slide}`,
      body: [
        h("div", null, h("div", { class: "big" }, "reference"), h("span", { class: "pill outline" }, "definitional, not measured")),
        h("p", null, REFERENCE_NOTE),
        h("p", { class: "muted" }, tk
          ? "Its alignment is 1.0 by construction: the slide\u2019s intent is this persona\u2019s own takeaway, so there is nothing to score it against. It is shown so the chart, the payload and the legend keep their shape for when the expert is measured independently."
          : "Its alignment is 1.0 by construction: the intended reading was generated from this persona\u2019s own reading, so there is nothing to score it against. It is shown so the chart, the payload and the legend keep their shape for when the expert is measured independently."),
        tk ? section("What it defined", src("Expert takeaway, verbatim: this is the intent of this slide", r.slide_intent.text, "expert")) : null,
        !tk && r.slide_intent ? section("What it defined", intentSources(r)) : null,
        !tk && r.slide_intent ? section("How the intended reading was written", calc(derivation(r.slide_intent))) : null,
      ],
    };
  },

  tier({ slide }, state) {
    const r = slideOf(state, slide);
    const t = state.rollup.per_slide.find((p) => p.slide === slide).tier;
    if (t.comparator === "fieldwise") return fw.tier(slide, r, t);
    const m = r.metrics;
    const tk = usesTakeawayIntent(r);
    return {
      title: `Tier \u2014 slide ${slide}`,
      body: [
        h("div", null, h("span", { class: `tier-chip large ${t.tier}` }, t.label)),
        h("p", null, t.meaning),
        h("p", { class: "muted" }, t.basis === "relative"
          ? `Read against the ${t.n_slides} slides of this deck that have been scored. The same numbers can land in another tier in another deck: only a slide\u2019s place among its own deck\u2019s slides is trusted.`
          : `This deck has ${t.n_slides} scored slide${t.n_slides === 1 ? "" : "s"}, fewer than the ${t.min_slides_for_relative} needed to compare a slide with its deck, so rough absolute thresholds were used. Levels on one slide move between runs; treat this as a guide.`),
        section("What drove it",
          h("div", { class: "stack" }, ...t.checks.map((c) => h("div", { class: "src" },
            h("div", { class: "who" }, c.passed ? "\u2713 met" : "\u2717 not met"),
            h("p", null, c.text))))),
        section("The inputs, with their texts",
          h("div", { class: "stack" },
            tk ? src("Intent of this slide = the expert takeaway, the reference (verbatim)", m.intent, "expert")
               : src("Intended reading, inferred from the expert reading", m.intent),
            src(`Novice takeaway \u2014 alignment ${f2(m.intent_alignment.novice.value)}`, m.takeaways.novice, "novice"),
            src(`Peer takeaway \u2014 alignment ${f2(m.intent_alignment.peer.value)}`, m.takeaways.peer, "peer"),
            tk ? null : src("Expert takeaway (the reference)", m.takeaways.expert, "expert"),
            src(`Terms the novice could not resolve (${m.term_gap.novice_unresolved.length})`, m.term_gap.novice_unresolved.join(", ") || "None", "novice"))),
        section("The rule", calc("Self-contained:    novice and peer both align, novice unresolved terms low\nBackground needed: peer aligns, novice does not (or the novice aligns but meets many unknown terms)\nExpert-gated:      neither novice nor peer aligns")),
      ],
    };
  },

  state({ slide, persona }, state) { return fw.claimState(slide, persona, slideOf(state, slide)); },
  field({ slide, persona, field }, state) { return fw.field(slide, persona, field, slideOf(state, slide)); },
  finding({ slide, index }, state) { return fw.finding(slide, index, slideOf(state, slide)); },
  profile({ slide }, state) { return fw.profile(slide, slideOf(state, slide)); },

  unresolved({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const rd = r.readings[persona];
    const gap = persona === "novice" && r.metrics ? r.metrics.term_gap : null;
    const absent = missing(r.text, rd.unresolved_terms);
    return {
      title: `Unresolved terms \u2014 ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, String(rd.unresolved_terms.length)),
        h("p", { class: "muted" }, "Terms this persona said it could not resolve, exactly as it wrote them. It is told to leave out terms the slide defines and terms someone with its background knows."),
        rd.unresolved_terms.length
          ? section("The terms", h("p", null, rd.unresolved_terms.join(" \u00b7 ")))
          : section("The terms", h("p", { class: "muted" }, "It reported none.")),
        srcNodes("Slide text as the personas saw it (matches highlighted)", highlight(r.text || "(no extractable text on this slide; the persona read the image)", rd.unresolved_terms)),
        absent.length ? h("p", { class: "caveat" }, `Not found word-for-word in the slide text: ${absent.join(", ")}. The persona may have read them from the image.`) : null,
        gap && section("Novice versus expert",
          h("div", { class: "stack" },
            src("Missed by the novice and not by the expert", gap.terms.length ? gap.terms.join(", ") : "None", "novice"),
            src("Expert also could not resolve", gap.expert_unresolved.length ? gap.expert_unresolved.join(", ") : "None", "expert"))),
        deckSection(state, `unresolved_count.${persona}`, slide, COLOR[persona]),
      ],
    };
  },

  dist({ key, slide }, state) {
    const dist = state.rollup.distributions[key];
    const sorted = [...dist.values].sort((a, b) => b.value - a.value || a.slide - b.slide);
    const span = Math.max(1e-9, ...sorted.map((v) => Math.abs(v.value)));
    const mine = slide != null && dist.values.find((v) => v.slide === slide);
    const rank = mine ? state.rollup.per_slide.find((p) => p.slide === slide).position[key].rank : null;
    return {
      title: keyLabel(key),
      body: [
        mine ? h("div", { class: "big" }, fmtFor(key)(mine.value)) : null,
        h("p", { class: "muted" }, `The value on every slide scored so far (${dist.n} of ${state.meta.slide_count}). Position is always read inside this deck; it says nothing about any other deck.`),
        calc([
          mine ? `rank of slide ${slide} = 1 + (number of slides with a higher value) = ${rank} of ${dist.n}` : null,
          `median ${dist.median.toFixed(4)}   quartiles ${dist.q1.toFixed(4)} – ${dist.q3.toFixed(4)}`,
          `range ${dist.min.toFixed(4)} – ${dist.max.toFixed(4)}`,
        ].filter(Boolean).join("\n")),
        h("div", { class: "vals" }, ...sorted.map((v) => h("a", { class: `vrow${v.slide === slide ? " here" : ""}`, href: `#/run/${state.meta.run_id}/slide/${v.slide}`, on: { click: closeDrawer } },
          h("span", null, `Slide ${v.slide}`),
          h("span", { class: "bar" }, h("i", { style: `width:${Math.max(2, (Math.abs(v.value) / span) * 100)}%` })),
          h("span", null, fmtFor(key)(v.value))))),
        KEY_CAVEAT[key] ? h("p", { class: "caveat" }, KEY_CAVEAT[key]) : null,
      ],
    };
  },

  term({ key }, state) {
    const t = state.rollup.terms.find((x) => x.key === key);
    return {
      title: `“${t.term}” — unresolved by the novice on ${t.count} slide${t.count === 1 ? "" : "s"}`,
      body: [
        h("p", { class: "muted" }, "Each slide below is one where the novice listed this term (spelling as the novice wrote it, case and hyphens ignored when matching)."),
        ...t.occurrences.map((o) => {
          const r = slideOf(state, o.slide);
          return h("div", { class: "src" },
            h("div", { class: "who" }, slideLink(state, o.slide), ` · novice wrote “${o.as_written}”`),
            h("p", null, ...highlight(r?.text || "", [o.as_written])));
        }),
      ],
    };
  },

  note({ id }, state) {
    const n = state.rollup.notes.find((x) => x.id === id);
    const ev = n.evidence;
    return {
      title: "Where this note comes from",
      body: [
        h("p", null, n.text),
        h("p", { class: "muted" }, "It is written from the numbers below and adds no reading of its own."),
        ev.kind === "terms"
          ? h("div", { class: "stack" }, ...ev.terms.map((t) => h("div", { class: "src" },
              h("div", { class: "who" }, `“${t.term}” · novice unresolved on ${t.count} slides`),
              h("p", null, ...t.slides.flatMap((s, i) => [i ? ", " : "", slideLink(state, s)])))))
          : h("div", { class: "vals" }, ...ev.rows.map((g) => h("a", { class: "vrow", href: `#/run/${state.meta.run_id}/slide/${g.slide}`, on: { click: closeDrawer } },
              h("span", null, `Slide ${g.slide}`), h("span", null, g.novice_state ? `novice: ${g.novice_state}` : `novice alignment ${f2(g.novice_alignment)}`), h("span", null, `${g.novice_unresolved} terms`)))),
      ],
    };
  },
};
