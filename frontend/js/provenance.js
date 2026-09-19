// The drawer that answers "where does this number come from?".
//
// Everything here is read from the run payload: the exact strings a metric was computed from,
// the other side of the comparison, and the raw arithmetic. Nothing is derived that the payload
// does not already carry. `numBtn` is how a number gets onto the screen at all: a value with no
// provenance spec has no way to be rendered.
import { h, f2, ordinal, PERSONAS, LABEL, COLOR } from "./dom.js";
import { strip } from "./charts.js";

const KEY_LABEL = {
  audience_divergence: "Audience divergence",
  blind_spot_score: "Blind-spot score (expert alignment − novice alignment)",
  term_gap_count: "Terms only the novice missed",
  confidence_gap: "Confidence gap (expert − novice)",
};
function keyLabel(key) {
  if (KEY_LABEL[key]) return KEY_LABEL[key];
  const [base, persona] = key.split(".");
  return `${{ confidence: "Confidence (self-reported)", unresolved_count: "Unresolved terms", intent_alignment: "Alignment to intent" }[base]} — ${LABEL[persona]}`;
}
const KEY_CAVEAT = {
  blind_spot_score: "The same slide swings by about ±0.3 from run to run. Trust how slides separate from each other in a deck, never the level on one slide.",
  audience_divergence: "Absolute divergence means little on its own. Semantic similarity measures topic and phrasing, not whether two takeaways make the same claim; it is the weaker instrument here.",
};
for (const p of PERSONAS) KEY_CAVEAT[`intent_alignment.${p}`] = "Semantic similarity to the intent is the weaker instrument: it separates readings by topic and wording, not by whether the claim was understood.";
for (const p of PERSONAS) KEY_CAVEAT[`confidence.${p}`] = "Confidence is the model's own report of how sure it is. It is not measured against anything.";

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
  const node = h("div", null,
    h("div", { class: "scrim", on: { click: closeDrawer } }),
    h("aside", { class: "drawer", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("header", null,
        stack.length > 1 && h("button", { class: "btn small", on: { click: () => { stack.pop(); paintDrawer(state); } } }, "← Back"),
        h("h2", null, title),
        h("button", { class: "btn small", "aria-label": "Close", on: { click: closeDrawer } }, "Close")),
      h("div", { class: "body" }, ...body)));
  drawerEl?.remove();
  drawerEl = node;
  document.body.append(node);
  node.querySelector("header .btn:last-child").focus();
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

const BUILDERS = {
  alignment({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const m = r.metrics.intent_alignment[persona];
    return {
      title: `Alignment to intent — ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, val(m)),
        h("p", { class: "muted" }, "How close, in topic and wording, this persona’s takeaway is to your declared intent. It does not check that the takeaway is right."),
        section("The two texts compared",
          h("div", { class: "stack" }, src("Declared intent (verbatim)", m.inputs.intent), src(`${LABEL[persona]} takeaway (verbatim)`, m.inputs[persona], persona))),
        section("Raw computation", calc(`cosine similarity of two sentence embeddings\nmodel: ${r.scored_by}\ncos(intent, ${persona} takeaway) = ${m.value.toFixed(4)}`)),
        deckSection(state, `intent_alignment.${persona}`, slide, COLOR[persona]),
      ],
    };
  },

  confidence({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const rd = r.readings[persona];
    return {
      title: `Confidence — ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", null, h("div", { class: "big" }, f2(rd.confidence)), h("span", { class: "pill outline" }, "self-reported by the model")),
        h("p", { class: "muted" }, "The model’s own answer, in the same reply as its takeaway, to how sure it is that it understood what the presenter meant. It is a statement about itself; nothing measures it."),
        section("The reply it came with",
          h("div", { class: "stack" },
            src("Takeaway (verbatim)", rd.takeaway, persona),
            src("Claim it thinks you want believed (verbatim)", rd.inferred_claim, persona),
            rd.questions.length ? src("Questions it would ask", rd.questions.map((q) => `• ${q}`).join("\n"), persona) : null,
            src("Terms it could not resolve", rd.unresolved_terms.length ? rd.unresolved_terms.join(", ") : "None", persona))),
        section("Where the reply came from", calc(
          `model: ${rd.model}\nserved from cache: ${rd.cached ? "yes" : "no"}\nmodel calls it took: ${rd.attempts}\nlatency: ${rd.latency_s}s\nslide hash: ${rd.slide_hash}`)),
        deckSection(state, `confidence.${persona}`, slide, COLOR[persona]),
      ],
    };
  },

  unresolved({ slide, persona }, state) {
    const r = slideOf(state, slide);
    const rd = r.readings[persona];
    const gap = persona === "novice" && r.metrics ? r.metrics.term_gap : null;
    const absent = missing(r.text, rd.unresolved_terms);
    return {
      title: `Unresolved terms — ${LABEL[persona]}, slide ${slide}`,
      body: [
        h("div", { class: "big" }, String(rd.unresolved_terms.length)),
        h("p", { class: "muted" }, "Terms this persona said it could not resolve, exactly as it wrote them. It is told to leave out terms the slide defines and terms someone with its background knows."),
        rd.unresolved_terms.length
          ? section("The terms", h("p", null, rd.unresolved_terms.join(" · ")))
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

  termgap({ slide }, state) {
    const r = slideOf(state, slide);
    const gap = r.metrics.term_gap;
    return {
      title: `Terms only the novice missed — slide ${slide}`,
      body: [
        h("div", { class: "big" }, String(gap.terms.length)),
        h("p", { class: "muted" }, "Terms the novice could not resolve that the expert did not list, after ignoring case, hyphens and surrounding punctuation."),
        srcNodes("Slide text as the personas saw it (matches highlighted)", highlight(r.text || "(no extractable text)", gap.terms)),
        section("Both lists it was computed from",
          h("div", { class: "stack" },
            src("Novice could not resolve", gap.novice_unresolved.join(", ") || "None", "novice"),
            src("Expert could not resolve", gap.expert_unresolved.join(", ") || "None", "expert"),
            src("Difference", gap.terms.join(", ") || "None"))),
        deckSection(state, "term_gap_count", slide, "var(--ink)"),
      ],
    };
  },

  divergence({ slide }, state) {
    const r = slideOf(state, slide);
    const m = r.metrics;
    const pairs = Object.entries(m.pairwise_distance);
    return {
      title: `Audience divergence — slide ${slide}`,
      body: [
        h("div", { class: "big" }, val(m.audience_divergence)),
        h("p", { class: "muted" }, "The mean of the three pairwise distances (1 − cosine similarity) between what the novice, peer and expert took away."),
        section("The three takeaways", h("div", { class: "stack" }, ...PERSONAS.map((p) => src(`${LABEL[p]} (verbatim)`, m.takeaways[p], p)))),
        section("The three distances",
          h("div", { class: "stack" }, ...pairs.map(([k, pm]) => {
            const [a, b] = k.split("-");
            return h("div", { class: "src" }, h("div", { class: "who" }, `${LABEL[a]} ↔ ${LABEL[b]}`), h("p", null, `distance ${pm.value.toFixed(4)}`));
          }))),
        section("Raw computation", calc(`(${pairs.map(([, pm]) => pm.value.toFixed(4)).join(" + ")}) / ${pairs.length} = ${m.audience_divergence.value.toFixed(4)}\nmodel: ${r.scored_by}`)),
        deckSection(state, "audience_divergence", slide, "var(--ink)"),
      ],
    };
  },

  blind({ slide }, state) {
    const r = slideOf(state, slide);
    const m = r.metrics;
    const e = m.intent_alignment.expert.value, n = m.intent_alignment.novice.value;
    return {
      title: `Blind-spot score — slide ${slide}`,
      body: [
        h("div", { class: "big" }, val(m.blind_spot_score)),
        h("p", { class: "muted" }, "The expert’s alignment to your intent minus the novice’s. Positive means the expert’s reading sits closer to what you meant than the novice’s does."),
        section("The three texts",
          h("div", { class: "stack" },
            src("Declared intent (verbatim)", m.blind_spot_score.inputs.intent),
            src("Novice takeaway (verbatim)", m.blind_spot_score.inputs.novice, "novice"),
            src("Expert takeaway (verbatim)", m.blind_spot_score.inputs.expert, "expert"))),
        section("Raw computation", calc(`expert alignment ${e.toFixed(4)} − novice alignment ${n.toFixed(4)} = ${m.blind_spot_score.value.toFixed(4)}\nmodel: ${r.scored_by}`)),
        deckSection(state, "blind_spot_score", slide, "var(--ink)"),
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
          : h("div", { class: "vals" }, ...ev.gaps.map((g) => h("a", { class: "vrow", href: `#/run/${state.meta.run_id}/slide/${g.slide}`, on: { click: closeDrawer } },
              h("span", null, `Slide ${g.slide}`), h("span", null, `${ordinal(g.rank)} of ${state.rollup.distributions.blind_spot_score.n}`), h("span", null, f2(g.gap))))),
      ],
    };
  },
};
