// The drawer that answers "where does this number come from?".
//
// Everything here is read from the run payload: the exact strings a metric was computed from,
// the other side of the comparison, and the raw arithmetic. Nothing is derived that the payload
// does not already carry. `numBtn` is how a number gets onto the screen at all: a value with no
// provenance spec has no way to be rendered.
import { h, f2, ordinal, PERSONAS, LABEL, COLOR } from "./dom.js";
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

function src(label, text, persona) {
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

const BUILDERS = {
  alignment({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const m = r.metrics.intent_alignment[persona];
    if (m.definitional) return BUILDERS.reference({ slide }, state);
    return {
      title: `Alignment to the intended reading \u2014 ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, val(m)),
        h("p", { class: "muted" }, "How close, in topic and wording, this persona\u2019s takeaway is to what the slide is trying to establish. It does not check that the takeaway is right."),
        section("The two texts compared",
          h("div", { class: "stack" },
            src("Intended reading, inferred from the expert reading (verbatim)", m.inputs.intent),
            src(`${LABEL[persona]} takeaway (verbatim)`, m.inputs[persona], persona))),
        section("Raw computation", calc(`cosine similarity of two sentence embeddings\nmodel: ${r.scored_by}\ncos(intended reading, ${persona} takeaway) = ${m.value.toFixed(4)}`)),
        section("Where the intended reading came from", calc(derivation(r.slide_intent))),
        deckSection(state, `intent_alignment.${persona}`, slide, COLOR[persona]),
      ],
    };
  },

  reference({ slide }, state) {
    const r = slideOf(state, slide);
    return {
      title: `Reference \u2014 Expert, slide ${slide}`,
      body: [
        h("div", null, h("div", { class: "big" }, "reference"), h("span", { class: "pill outline" }, "definitional, not measured")),
        h("p", null, REFERENCE_NOTE),
        h("p", { class: "muted" }, "Its alignment is 1.0 by construction: the intended reading was generated from this persona\u2019s own reading, so there is nothing to score it against. It is shown so the chart, the payload and the legend keep their shape for when the expert is measured independently."),
        r.slide_intent ? section("What it defined", intentSources(r)) : null,
        r.slide_intent ? section("How the intended reading was written", calc(derivation(r.slide_intent))) : null,
      ],
    };
  },

  tier({ slide }, state) {
    const r = slideOf(state, slide);
    const t = state.rollup.per_slide.find((p) => p.slide === slide).tier;
    const m = r.metrics;
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
            src("Intended reading, inferred from the expert reading", m.intent),
            src(`Novice takeaway \u2014 alignment ${f2(m.intent_alignment.novice.value)}`, m.takeaways.novice, "novice"),
            src(`Peer takeaway \u2014 alignment ${f2(m.intent_alignment.peer.value)}`, m.takeaways.peer, "peer"),
            src("Expert takeaway (the reference)", m.takeaways.expert, "expert"),
            src(`Terms the novice could not resolve (${m.term_gap.novice_unresolved.length})`, m.term_gap.novice_unresolved.join(", ") || "None", "novice"))),
        section("The rule", calc("Self-contained:    novice and peer both align, novice unresolved terms low\nBackground needed: peer aligns, novice does not (or the novice aligns but meets many unknown terms)\nExpert-gated:      neither novice nor peer aligns")),
      ],
    };
  },

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
              h("span", null, `Slide ${g.slide}`), h("span", null, `novice alignment ${f2(g.novice_alignment)}`), h("span", null, `${g.novice_unresolved} terms`)))),
      ],
    };
  },
};
