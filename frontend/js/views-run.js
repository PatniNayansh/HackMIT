import { h, mount, f2, when, describe, warnIcon, infoIcon, PERSONAS, LABEL, COLOR } from "./dom.js";
import { getJSON } from "./api.js";
import { watch } from "./run.js";
import { arcChart } from "./charts.js";
import { numBtn, cmpLine, deckStrip, openProvenance } from "./provenance.js";

const runHref = (state, n) => `#/run/${state.meta.run_id}${n ? `/slide/${n}` : ""}`;

function statusPill(meta) {
  if (meta.status === "complete") return h("span", { class: "pill" }, "complete");
  if (meta.status === "running") return h("span", { class: "pill outline" }, h("span", { class: "spinner" }), "running");
  return h("span", { class: "pill bad" }, meta.status);
}

function subfieldLine(meta) {
  const p = meta.profile;
  if (!p) return null;
  return h("p", { class: "muted small" },
    `Subfield: ${p.domain} · peer comes from: ${p.adjacent_field} `,
    h("span", { class: "pill" }, p.edited ? "edited by presenter" : "as suggested, confirmed by presenter"));
}

/** Shared loading / error frame. Returns true if the caller should stop painting. */
function guard(root, state) {
  if (state.error && !state.loaded) {
    root.replaceChildren(h("div", { class: "banner bad" }, warnIcon(), h("div", null, state.error, " ", h("a", { href: "#/" }, "Back to start"))));
    return true;
  }
  if (!state.loaded) {
    root.replaceChildren(h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading run…"));
    return true;
  }
  if (state.meta.status === "draft") { location.hash = `#/run/${state.meta.run_id}/setup`; return true; }
  return false;
}

function runBanners(state) {
  const { meta, rollup } = state;
  const out = [];
  if (meta.sample) out.push(h("div", { class: "banner sample" }, infoIcon(), h("div", null, h("strong", null, "sample data. "), "This is a bundled sample run for demonstration, not a review of your own deck.")));
  if (meta.status === "failed" || meta.status === "interrupted") {
    out.push(h("div", { class: "banner bad" }, warnIcon(), h("div", null,
      h("strong", null, meta.status === "interrupted" ? "This run was cut off. " : "This run stopped. "),
      meta.error || "The server stopped before it finished.",
      state.results.size ? ` The ${state.results.size} slide${state.results.size === 1 ? "" : "s"} below finished before that.` : "",
      !meta.sample && [" ", h("a", { href: `#/run/${meta.run_id}/setup` }, "Set up and try again")])));
  }
  if (rollup.unscored.length) {
    out.push(h("div", { class: "banner bad" }, warnIcon(), h("div", null,
      h("strong", null, `Bad model response on ${rollup.unscored.length} slide${rollup.unscored.length === 1 ? "" : "s"}: `),
      ...rollup.unscored.flatMap((n, i) => [i ? ", " : "", h("a", { href: runHref(state, n) }, String(n))]),
      ". Those replies failed validation, so they are shown as errors, not repaired, and no comparison was computed for those slides.")));
  }
  return out;
}

// -------------------------------------------------------------------- overview

export function overview(root, runId) {
  let showAllTerms = false, showAllGaps = false;

  const paint = (state) => {
    if (guard(root, state)) return;
    const { meta, rollup } = state;
    const total = meta.slide_count, done = state.results.size;
    document.title = `${meta.title} — Sightline`;

    const head = h("div", { class: "page-head" },
      h("div", null,
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Saved runs"), "›", meta.title),
        h("h1", null, meta.title),
        h("div", { class: "meta-line small" },
          statusPill(meta), h("span", null, when(meta.created_at)), h("span", null, `${total} slides`),
          h("span", { class: "mono" }, meta.model || "no model"), meta.sample && h("span", { class: "pill sample" }, "sample data")),
        meta.status === "running" && h("div", null,
          h("div", { class: "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": total, "aria-valuenow": done },
            h("i", { style: `width:${(done / total) * 100}%` })),
          h("p", { class: "muted small", style: "margin-top:4px" }, `Read ${done} of ${total} slides. Each takes about 5 seconds; the overview fills in as they land.`))));

    const intentBox = h("div", { class: "card" },
      h("div", { class: "intent-box" }, h("div", { class: "small muted" }, "Declared intent"), h("p", null, meta.intent)),
      h("div", { style: "margin-top:8px" }, subfieldLine(meta)));

    // Streaming film strip: each slide shows as soon as its result lands.
    const nextPending = state.pending[0];
    const filmstrip = h("div", { class: "filmstrip" }, ...state.imageUrls.map((url, i) => {
      const n = i + 1, r = state.results.get(n);
      const img = h("div", { class: "thumb" }, h("img", { src: url, alt: `Slide ${n}`, loading: "lazy" }));
      if (!r) return h("div", { class: `tile pending${meta.status === "running" && n === nextPending ? " next" : ""}` }, img, h("div", { class: "cap" }, h("b", null, n), h("span", null, "pending")));
      const bad = !r.metrics;
      return h("a", { class: "tile", href: runHref(state, n) }, img,
        h("div", { class: "cap" }, h("b", null, n), bad ? h("span", { class: "pill bad" }, "bad response") : h("span", null, "read")));
    }));

    const parts = [head, ...runBanners(state), intentBox,
      h("section", { class: "section" }, h("header", null, h("h2", null, "Slides")), filmstrip)];

    if (!done) {
      parts.push(h("p", { class: "empty" }, meta.status === "running" ? "Waiting for the first slide…" : "No slide finished."));
      mount(root, ...parts);
      return;
    }

    // Notes: restatements of the numbers below, each opening its evidence.
    if (rollup.notes.length) {
      parts.push(h("section", { class: "section" },
        h("header", null, h("h2", null, "What the numbers say")),
        h("p", { class: "lede" }, "Written from the counts and ranks below; each opens the evidence it was built from."),
        h("div", { class: "card" }, ...rollup.notes.map((n) => h("div", { class: "note" }, infoIcon(), numBtn(n.text, { kind: "note", id: n.id }, state, `${n.text} Show the evidence.`))))));
    }

    // Terms: the novice's deck-wide vocabulary problem. Counts, so they carry the visual weight.
    const terms = rollup.terms;
    const shown = showAllTerms ? terms : terms.slice(0, 8);
    parts.push(h("section", { class: "section" },
      h("header", null, h("h2", null, "Terms the novice could not resolve, across the deck")),
      h("p", { class: "lede" }, "A term that stays unresolved on several slides is a vocabulary problem for the whole deck, not for one slide."),
      terms.length
        ? h("div", { class: "card" },
            ...shown.map((t) => h("div", { class: "term-row" },
              h("span", { class: "term" }, t.term),
              h("span", { class: "term-bar" }, ...t.slides.map((n) => h("a", { href: runHref(state, n), title: `Slide ${n}`, "aria-label": `Slide ${n}` }, n))),
              h("span", { class: "right tnum", style: "text-align:right" }, numBtn(`${t.count} slide${t.count === 1 ? "" : "s"}`, { kind: "term", key: t.key }, state)))),
            terms.length > 8 && h("div", { style: "margin-top:10px" }, h("button", { class: "btn small", on: { click: () => { showAllTerms = !showAllTerms; paint(state); } } }, showAllTerms ? "Show fewer" : `Show all ${terms.length} terms`)))
        : h("p", { class: "empty" }, "The novice listed no unresolved terms on the slides read so far.")));

    // Ranking + relative divergence.
    const rankRow = (g, key, factsFor) => {
      const n = g.slide;
      return h("div", { class: "rank-row" },
        h("a", { href: runHref(state, n), "aria-label": `Open slide ${n}` }, h("img", { src: state.imageUrls[n - 1], alt: "" }), h("div", { class: "small", style: "margin-top:4px;font-weight:600" }, `Slide ${n}`)),
        h("div", { class: "stack" }, h("div", { class: "rank-facts" }, ...factsFor(n)), deckStrip(state, key, n), cmpLine(state, key, n)));
    };
    const fact = (k, v, weak) => h("div", { class: `fact${weak ? " weak" : ""}` }, h("span", { class: "k" }, k), h("span", { class: "v" }, v));
    const gapFacts = (n) => {
      const r = state.results.get(n), c = (p) => r.readings[p];
      return [
        fact("Novice unresolved terms", numBtn(String(c("novice").unresolved_terms.length), { kind: "unresolved", slide: n, persona: "novice" }, state)),
        fact("Confidence, novice → expert", h("span", null, numBtn(f2(c("novice").confidence), { kind: "confidence", slide: n, persona: "novice" }, state), " → ", numBtn(f2(c("expert").confidence), { kind: "confidence", slide: n, persona: "expert" }, state))),
        fact("Novice–expert gap", numBtn(f2(rollup.gap_ranking.find((g) => g.slide === n).gap), { kind: "blind", slide: n }, state), true),
      ];
    };
    const gaps = showAllGaps ? rollup.gap_ranking : rollup.gap_ranking.slice(0, 5);
    const divFacts = (n) => [
      fact("Novice unresolved terms", numBtn(String(state.results.get(n).readings.novice.unresolved_terms.length), { kind: "unresolved", slide: n, persona: "novice" }, state)),
      fact("Audience divergence", numBtn(f2(state.results.get(n).metrics.audience_divergence.value), { kind: "divergence", slide: n }, state), true),
    ];
    parts.push(h("div", { class: "section two-col" },
      h("section", null,
        h("header", { style: "display:block" }, h("h2", null, "Slides ranked by novice–expert gap")),
        h("p", { class: "lede" }, "Widest first. The gap is how much closer the expert’s reading sits to your intent than the novice’s. It is a ranking inside this deck: the same slide moves by about ±0.3 between runs, so read the order, not the number."),
        rollup.gap_ranking.length
          ? h("div", { class: "card" }, ...gaps.map((g) => rankRow(g, "blind_spot_score", gapFacts)),
              rollup.gap_ranking.length > 5 && h("div", { style: "margin-top:10px" }, h("button", { class: "btn small", on: { click: () => { showAllGaps = !showAllGaps; paint(state); } } }, showAllGaps ? "Show fewer" : `Show all ${rollup.gap_ranking.length} slides`)))
          : h("p", { class: "empty" }, "No slide has all three readings yet.")),
      h("section", null,
        h("header", { style: "display:block" }, h("h2", null, "Where the audiences differ most")),
        h("p", { class: "lede" }, "Highest audience divergence relative to the rest of this deck. Semantic similarity is the weaker instrument here; the term counts and confidence beside it separated the audiences more cleanly."),
        rollup.comparable
          ? h("div", { class: "card" }, ...rollup.divergence_top.map((d) => rankRow(d, "audience_divergence", divFacts)))
          : h("p", { class: "empty" }, `A slide can only be high relative to its deck once ${rollup.min_slides_for_comparison} slides are scored; ${rollup.n_scored} so far.`))));

    parts.push(h("section", { class: "section" },
      h("header", null, h("h2", null, "Narrative arc")),
      h("p", { class: "lede" }, "Alignment to your declared intent, per audience, in slide order. Look for where one line drops away from the others. Semantic similarity is the weaker instrument, so compare shapes across slides rather than levels."),
      h("div", { class: "card" }, arcChart(rollup, {
        onPoint: (slide, persona) => openProvenance({ kind: "alignment", slide, persona }, state),
        tableNumber: (slide, persona, v) => numBtn(f2(v), { kind: "alignment", slide, persona }, state),
      }))));

    mount(root, ...parts);
  };
  return watch(runId, paint);
}

// ---------------------------------------------------------------- slide detail

let tab = "readings";

function audienceCard(state, r, persona) {
  const rd = r.readings[persona];
  const n = r.index;
  const who = h("div", { class: "aud-head" },
    h("span", { class: "who" }, h("span", { class: `dot ${persona}` }), LABEL[persona]),
    h("span", { class: "muted small" }, describe(persona, state.meta.profile)));

  // A reply that failed validation, or a stored confidence outside [0, 1], is shown as an
  // error. It is never clamped and never rendered as a number.
  const badConfidence = rd.ok && !(typeof rd.confidence === "number" && rd.confidence >= 0 && rd.confidence <= 1);
  if (!rd.ok || badConfidence) {
    const message = rd.ok ? `Confidence ${rd.confidence} is outside 0–1.` : rd.error.message;
    const attempts = rd.ok ? [] : rd.error.attempts;
    return h("article", { class: `aud ${persona} bad` }, who,
      h("div", { class: "aud-body" },
        h("div", { class: "banner bad", style: "margin-top:8px" }, warnIcon(), h("div", null,
          h("strong", null, "Bad response, not used. "), message,
          h("p", { class: "small", style: "margin-top:4px" }, "The model’s reply failed validation. It is shown as an error rather than clamped or repaired, and no comparison involving this audience was computed for this slide."))),
        attempts.length > 0 && h("details", null, h("summary", null, `What the model returned (${attempts.length} attempt${attempts.length === 1 ? "" : "s"})`),
          h("ul", null, ...attempts.map((a) => h("li", { class: "mono" }, a))))));
  }

  const color = COLOR[persona];
  const terms = rd.unresolved_terms;
  const align = r.metrics?.intent_alignment[persona];
  const row = (cls, label, valueNode, ...right) => h("div", { class: `metric-row ${cls}` },
    h("div", { class: "left" }, h("span", { class: "lbl" }, label), valueNode),
    h("div", { class: "right" }, ...right));

  return h("article", { class: `aud ${persona}` }, who,
    h("div", { class: "aud-body" },
      h("div", { class: "take-label" }, "Takeaway, verbatim"),
      h("blockquote", { class: "take" }, `“${rd.takeaway}”`),
      // Unresolved terms and confidence separated the audiences best, so they carry the weight.
      row("strong", "Unresolved terms", h("span", { class: "val" }, numBtn(String(terms.length), { kind: "unresolved", slide: n, persona }, state)),
        deckStrip(state, `unresolved_count.${persona}`, n, color), cmpLine(state, `unresolved_count.${persona}`, n),
        terms.length ? h("span", { class: "preview" }, terms.slice(0, 4).join(" · "), terms.length > 4 ? ` +${terms.length - 4} more` : "") : null),
      row("strong", "Confidence", h("span", { class: "val" }, numBtn(f2(rd.confidence), { kind: "confidence", slide: n, persona }, state)),
        h("span", { class: "self" }, "The model’s self-report, not a measurement"),
        deckStrip(state, `confidence.${persona}`, n, color), cmpLine(state, `confidence.${persona}`, n)),
      align
        ? row("weak", "Alignment to intent", h("span", { class: "val" }, numBtn(f2(align.value), { kind: "alignment", slide: n, persona }, state)),
            deckStrip(state, `intent_alignment.${persona}`, n, color), cmpLine(state, `intent_alignment.${persona}`, n),
            h("span", { class: "cmp" }, "Semantic similarity: the weaker instrument."))
        : row("weak", "Alignment to intent", h("span", { class: "val" }, "—"), h("span", { class: "cmp" }, "Not computed for this slide.")),
      h("details", null, h("summary", null, "More from this reading"),
        h("div", { class: "stack", style: "margin-top:8px" },
          h("div", null, h("div", { class: "take-label" }, "Claim it thinks you want believed"), h("p", null, rd.inferred_claim)),
          h("div", null, h("div", { class: "take-label" }, "Questions it would need answered"),
            rd.questions.length ? h("ul", null, ...rd.questions.map((q) => h("li", null, q))) : h("p", { class: "muted" }, "None"))))));
}

function crossAudience(state, r) {
  const n = r.index;
  if (!r.metrics) {
    return h("div", { class: "banner bad rel" }, warnIcon(), h("div", null, h("strong", null, "Cross-audience metrics not computed. "), r.metrics_error || ""));
  }
  const m = r.metrics;
  const row = (title, valueNode, key, color, caveat) => h("div", { class: "rel-row" },
    h("div", null, h("div", { class: "small muted" }, title), h("div", { class: "val" }, valueNode), caveat && h("p", { class: "caveat" }, caveat)),
    h("div", { class: "stack" }, deckStrip(state, key, n, color), cmpLine(state, key, n)));
  return h("section", { class: "rel" },
    h("h2", { style: "margin-bottom:2px" }, "Across the three audiences"),
    h("p", { class: "muted small" }, "Each value is shown against the rest of this deck, never as a verdict on this slide."),
    row("Terms only the novice missed", numBtn(String(m.term_gap.terms.length), { kind: "termgap", slide: n }, state), "term_gap_count", "var(--ink)",
      "Unresolved by the novice and not by the expert."),
    row("Audience divergence", numBtn(f2(m.audience_divergence.value), { kind: "divergence", slide: n }, state), "audience_divergence", "var(--ink)",
      "Semantic similarity is the weaker instrument. It reads topic and phrasing, not whether the claims match."),
    row("Blind-spot score", numBtn(f2(m.blind_spot_score.value), { kind: "blind", slide: n }, state), "blind_spot_score", "var(--ink)",
      "Expert minus novice alignment. About ±0.3 run-to-run swing on one slide: trust the separation between slides, not this level."));
}

function findingsPanel(runId, n) {
  const box = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading…"));
  getJSON(`/api/runs/${encodeURIComponent(runId)}/slides/${n}/findings`).then(({ fixture, findings }) => {
    box.replaceChildren(
      fixture && h("div", { class: "banner sample", style: "margin-bottom:12px" }, infoIcon(), h("div", null,
        h("span", { class: "pill sample" }, "sample data"), " ",
        "These findings come from a checked-in fixture. They do not describe this slide, and the recommendation engine is not built yet.")),
      ...findings.map((f) => h("article", { class: "finding" },
        h("div", { style: "display:flex;gap:8px;align-items:center;flex-wrap:wrap" },
          h("span", { class: `pill ${{ high: "solid", medium: "outline", low: "" }[f.severity]}` }, `${f.severity} severity`),
          fixture && h("span", { class: "pill sample" }, "sample data")),
        h("p", { style: "margin-top:8px;font-weight:600" }, f.trigger),
        h("dl", null,
          h("div", null, h("dt", null, "Evidence, verbatim"), h("dd", null, h("blockquote", null, f.evidence))),
          h("div", null, h("dt", null, "Suggestion"), h("dd", null, f.suggestion)),
          h("div", null, h("dt", null, "What this cannot tell you"), h("dd", { class: "muted" }, f.confidence_note))))));
  }).catch((e) => box.replaceChildren(h("p", { class: "err" }, e.message)));
  return box;
}

function neuralPanel(runId, n) {
  const box = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading…"));
  getJSON(`/api/runs/${encodeURIComponent(runId)}/slides/${n}/neural`).then((d) => {
    box.replaceChildren(
      h("div", { class: "banner sample" }, infoIcon(), h("div", null,
        h("strong", null, d.overlay_label), " ",
        "From a brain-encoding model (TRIBE v2, Meta AI) run on synthesized narration of this slide’s text — not a real presenter reading it, and not a brain measurement.")),
      h("div", { class: "rel-row" },
        h("div", null,
          h("div", { class: "small muted" }, "Processing ratio (language drive ÷ visual drive)"),
          h("div", { class: "val" }, f2(d.processing_ratio)),
          h("p", { class: "caveat" }, "A proposed readout, not a validated metric. Compare its shape across slides in this deck, not the level on this one.")),
        h("div", null,
          h("div", { class: "small muted" }, "Language drive"), h("div", null, f2(d.language_drive)),
          h("div", { class: "small muted", style: "margin-top:8px" }, "Visual drive"), h("div", null, f2(d.visual_drive)))),
      h("details", null, h("summary", null, "Narration this prediction was made from"),
        h("div", { class: "textbox" }, d.narration_transcript)),
      h("div", { class: "surface-grid" }, ...Object.entries(d.views).map(([name, url]) =>
        h("figure", null, h("img", { src: url, alt: name, loading: "lazy" }),
          h("figcaption", { class: "small muted" }, name.replace(/_/g, " "))))));
  }).catch((e) => {
    box.replaceChildren(e.status === 404
      ? h("div", { class: "neural-empty" },
          h("img", { class: "neural-empty-brain", src: "/static/img/brain-unlit.png", alt: "", "aria-hidden": "true" }),
          h("p", { class: "empty" }, "No precomputed neural data for this slide. The neural layer runs offline on a GPU, only for the bundled sample decks — see docs/SIGHTLINE_spec.md §5."))
      : h("p", { class: "err" }, e.message));
  });
  return box;
}

const DETAIL_TABS = [
  ["readings", "Audience readings", null],
  ["neural", "Neural (predicted)", "predicted"],
  ["recs", "Recommendations", "sample data"],
];

export function detail(root, runId, n) {
  return watch(runId, (state) => {
    if (guard(root, state)) return;
    const { meta } = state;
    const total = meta.slide_count;
    const r = state.results.get(n);
    document.title = `Slide ${n} — ${meta.title} — Sightline`;

    if (n < 1 || n > total) { root.replaceChildren(h("p", { class: "err" }, `This deck has ${total} slides.`), h("a", { href: runHref(state) }, "Back to the overview")); return; }

    const chips = h("nav", { class: "filmbar", "aria-label": "Slides" }, ...Array.from({ length: total }, (_, i) => {
      const k = i + 1, rr = state.results.get(k);
      if (k === n) return h("span", { class: "chip here", "aria-current": "page" }, k);
      if (!rr) return h("span", { class: "chip pending", title: "Not read yet" }, k);
      return h("a", { class: `chip${rr.metrics ? "" : " unscored"}`, href: runHref(state, k), title: rr.metrics ? `Slide ${k}` : `Slide ${k}: bad model response` }, k);
    }));
    const prev = n > 1 && state.results.has(n - 1), next = n < total && state.results.has(n + 1);
    const head = h("div", { class: "page-head" },
      h("div", null,
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Saved runs"), "›", h("a", { href: runHref(state) }, meta.title), "›", `Slide ${n}`),
        h("h1", null, `Slide ${n} of ${total}`)),
      h("div", { style: "display:flex;gap:8px" },
        h("a", { class: "btn", href: runHref(state), style: "text-decoration:none" }, "Overview"),
        prev ? h("a", { class: "btn", href: runHref(state, n - 1), style: "text-decoration:none" }, "← Previous") : h("button", { class: "btn", disabled: true }, "← Previous"),
        next ? h("a", { class: "btn", href: runHref(state, n + 1), style: "text-decoration:none" }, "Next →") : h("button", { class: "btn", disabled: true }, "Next →")));

    if (!r) {
      mount(root, head, chips, ...runBanners(state),
        h("div", { class: "card" }, meta.status === "running"
          ? h("p", null, h("span", { class: "spinner" }), ` Slide ${n} has not been read yet. It will appear here as soon as it lands.`)
          : h("p", { class: "muted" }, `Slide ${n} has no result: the run ended before reaching it.`)));
      return;
    }

    const left = h("div", { class: "slide-col" },
      h("img", { class: "slide-img", src: state.imageUrls[n - 1], alt: `Slide ${n}` }),
      h("div", { class: "intent-box" }, h("div", { class: "small muted" }, "Declared intent (the audiences never saw this)"), h("p", null, meta.intent)),
      h("details", null, h("summary", { class: "small muted", style: "cursor:pointer" }, "Text the audiences were given, alongside the image"),
        h("div", { class: "textbox" }, r.text || "(no extractable text on this slide)")));

    const tabs = h("div", { class: "tabs", role: "tablist" },
      ...DETAIL_TABS.map(([id, label, badge]) => h("button", {
        class: "tab", role: "tab", "aria-selected": String(tab === id),
        on: { click: () => { tab = id; detailPaint(); } },
      }, label, badge && h("span", { class: "pill sample" }, badge))));
    const panel = h("div", null);
    const detailPaint = () => {
      tabs.querySelectorAll(".tab").forEach((b, i) => b.setAttribute("aria-selected", String(tab === DETAIL_TABS[i][0])));
      panel.replaceChildren(...(
        tab === "readings" ? [...PERSONAS.map((p) => audienceCard(state, r, p)), crossAudience(state, r)]
        : tab === "neural" ? [neuralPanel(runId, n)]
        : [findingsPanel(runId, n)]
      ));
    };
    detailPaint();

    mount(root, head, chips, ...runBanners(state), h("div", { class: "detail" }, left, h("div", null, tabs, panel)));
  });
}
