import { h, mount, f2, when, infoIcon, slideImage, stateChip } from "./dom.js";
import { watch } from "./run.js";
import { arcChart } from "./charts.js";
import { numBtn, openProvenance } from "./provenance.js";
import { runHref, statusPill, subfieldLine, tierChip, guard, runBanners } from "./run-common.js";

const firstLine = (text) => (text.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? text);

export function overview(root, runId) {
  let showAllTerms = false, showAllHardest = false;

  const paint = (state) => {
    if (guard(root, state)) return;
    const { meta, rollup } = state;
    const total = meta.slide_count, done = state.results.size;
    document.title = `${meta.title} — ProFe`;

    const head = h("div", { class: "page-head" },
      h("div", null,
        h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Saved runs"), "›", meta.title),
        h("h1", null, meta.title),
        h("div", { class: "meta-line small" },
          statusPill(meta), h("span", null, when(meta.created_at)), h("span", null, `${total} slides`),
          state.rollup.comparator && !meta.legacy && h("span", { class: "pill", title: "Which comparator produced this run\u2019s numbers" }, state.rollup.comparator === "fieldwise" ? "field-wise comparison" : "cosine comparison"),
          h("span", { class: "mono" }, meta.model || "no model"),
          meta.sample && h("span", { class: "pill sample" }, "sample data")),
        meta.status === "running" && h("div", null,
          h("div", { class: "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": total, "aria-valuenow": done },
            h("i", { style: `width:${(done / total) * 100}%` })),
          h("p", { class: "muted small", style: "margin-top:4px" }, `Read ${done} of ${total} slides. Each takes about 6 seconds; the overview fills in as they land.`))));

    const context = (meta.intent || meta.profile) && h("div", { class: "card" },
      meta.intent && h("div", { class: "intent-box" }, h("div", { class: "small muted" }, "Your declared intent (stored; alignment is measured against each slide’s own intent, the expert’s takeaway)"), h("p", null, meta.intent)),
      h("div", { style: meta.intent ? "margin-top:8px" : "" }, subfieldLine(meta)));

    // Streaming film strip: each slide shows as soon as its result lands.
    const nextPending = state.pending[0];
    const filmstrip = h("div", { class: "filmstrip" }, ...state.imageUrls.map((url, i) => {
      const n = i + 1, r = state.results.get(n);
      const img = h("div", { class: "thumb slide-card" }, h("img", { src: url, alt: `Slide ${n}`, loading: "lazy" }));
      if (!r) return h("div", { class: `tile pending${meta.status === "running" && n === nextPending ? " next" : ""}` }, img, h("div", { class: "cap" }, h("b", null, n), h("span", null, "pending")));
      // A saved-before-tiers run has nowhere to go: its slide page was retired with the metrics it showed.
      // The title card has no tier because nothing read it -- not a bad response, just not analysed.
      const label = r.title_slide ? h("span", { class: "pill" }, "title slide")
        : meta.legacy ? h("span", null, "read")
        : (r.metrics ? tierMini(state, n) : h("span", { class: "pill bad" }, "bad response"));
      const tag = meta.legacy ? "div" : "a";
      return h(tag, { class: "tile", href: meta.legacy ? null : runHref(state, n) }, img, h("div", { class: "cap" }, h("b", null, n), label));
    }));

    const parts = [head, ...runBanners(state), context,
      h("section", { class: "section" }, h("header", null, h("h2", null, "Slides")), filmstrip)];

    if (!done) {
      parts.push(h("p", { class: "empty" }, meta.status === "running" ? "Waiting for the first slide…" : "No slide finished."));
      mount(root, ...parts);
      return;
    }

    if (!meta.legacy && rollup.notes.length) {
      parts.push(h("section", { class: "section" },
        h("header", null, h("h2", null, "What the numbers say")),
        h("p", { class: "lede" }, "Each one opens the evidence it came from."),
        h("div", { class: "card" }, ...rollup.notes.map((n) => h("div", { class: "note" }, infoIcon(), numBtn(n.text, { kind: "note", id: n.id }, state, `${n.text} Show the evidence.`))))));
    }

    if (!meta.legacy) parts.push(hardestSection(state, showAllHardest, () => { showAllHardest = !showAllHardest; paint(state); }));

    // Terms: the novice's deck-wide vocabulary problem.
    const terms = rollup.terms;
    const shown = showAllTerms ? terms : terms.slice(0, 8);
    parts.push(h("section", { class: "section" },
      h("header", null, h("h2", null, "Terms the novice could not resolve, across the deck")),
      h("p", { class: "lede" }, "A term missed on several slides is a deck problem, not a slide problem."),
      terms.length
        ? h("div", { class: "card" },
            ...shown.map((t) => h("div", { class: "term-row" },
              h("span", { class: "term" }, t.term),
              h("span", { class: "term-bar" }, ...t.slides.map((n) => h("a", { href: meta.legacy ? null : runHref(state, n), title: `Slide ${n}`, "aria-label": `Slide ${n}` }, n))),
              h("span", { class: "right tnum", style: "text-align:right" }, numBtn(`${t.count} slide${t.count === 1 ? "" : "s"}`, { kind: "term", key: t.key }, state)))),
            terms.length > 8 && h("div", { style: "margin-top:10px" }, h("button", { class: "btn small", on: { click: () => { showAllTerms = !showAllTerms; paint(state); } } }, showAllTerms ? "Show fewer" : `Show all ${terms.length} terms`)))
        : h("p", { class: "empty" }, "The novice listed no unresolved terms on the slides read so far.")));

    if (!meta.legacy && rollup.arc.some((a) => a.novice != null)) {
      parts.push(h("section", { class: "section" },
        h("header", null, h("h2", null, "Narrative arc")),
        h("p", { class: "lede" }, rollup.comparator === "fieldwise"
          ? (rollup.arc_kind === "coverage"
              ? "Propositions of the expert\u2019s claim each audience covered, slide by slide. Where the novice line drops and stays down, a newcomer was lost."
              : "Where each audience\u2019s reading of the claim lands against the expert\u2019s: equivalent, over-claimed, under-specified, divergent or absent.")
          : "How far each audience falls below the intended reading as the deck goes on."),
        h("div", { class: "card" }, arcChart(rollup, {
          onPoint: (slide, persona) => openProvenance(rollup.comparator === "fieldwise"
            ? (rollup.definitional.includes(persona) ? { kind: "reference", slide } : { kind: "state", slide, persona })
            : { kind: "alignment", slide, persona }, state),
          tableNumber: (slide, persona, v, isReference) => (isReference
            ? numBtn("reference", { kind: "reference", slide }, state, "reference. Show why this is definitional.")
            : rollup.comparator === "fieldwise"
              ? (() => {
                  const pt = rollup.arc.find((a) => a.slide === slide), ct = pt.counts?.[persona];
                  return numBtn(`${rollup.arc_kind === "coverage" && ct?.total ? `${ct.covered} of ${ct.total} \u00b7 ` : ""}${pt.states[persona] ?? "\u2014"}`, { kind: "state", slide, persona }, state);
                })()
              : numBtn(f2(v), { kind: "alignment", slide, persona }, state)),
        }))));
    }

    mount(root, ...parts);
  };
  return watch(runId, paint);
}

/** A tier as a quiet label under a thumbnail. Clicking the thumbnail opens the slide, so this
 *  one is not a button: the chip on the slide page and the ranking is the clickable one. */
function tierMini(state, n) {
  const t = state.rollup.per_slide.find((p) => p.slide === n)?.tier;
  return t ? h("span", { class: `tier-mini ${t.tier}` }, t.label) : h("span", null, "read");
}

function hardestSection(state, showAll, toggle) {
  const { rollup, meta } = state;
  const rows = showAll ? rollup.hardest : rollup.hardest.slice(0, 5);
  return h("section", { class: "section" },
    h("header", null, h("h2", null, "Hardest slides for a newcomer")),
    h("p", { class: "lede" },
      rollup.comparator === "fieldwise"
        ? (rollup.arc_kind === "coverage"
          ? "Propositions covered first, then fields missed, then unresolved terms. A ranking inside this deck."
          : "Where the novice’s claim lands first, then fields missed, then unresolved terms. A ranking inside this deck.")
        : "How far the novice falls from the slide’s point, then unresolved terms. A ranking inside this deck.",
      meta.status === "running" && " Tiers can shift as more slides arrive."),
    rows.length
      ? h("div", { class: "card" },
          ...rows.map((row) => {
            const n = row.slide, r = state.results.get(n);
            const take = r.readings.novice.takeaway;
            return h("div", { class: "rank-row wide" },
              h("a", { href: runHref(state, n), "aria-label": `Open slide ${n}` }, slideImage(state.imageUrls[n - 1], ""), h("div", { class: "small", style: "margin-top:4px;font-weight:600" }, `Slide ${n}`)),
              h("div", { class: "rank-take" }, h("div", { class: "take-label" }, "Novice takeaway"), h("div", { class: "clamp" }, `“${firstLine(take)}”`)),
              h("div", { class: "rank-tier" }, tierChip(state, n), rollup.comparator === "fieldwise" && row.novice_state ? h("span", { class: "rank-state" }, "novice: ", stateChip(row.novice_state), row.novice_total ? ` ${row.novice_covered} of ${row.novice_total}` : "") : null),
              h("div", { class: "fact" }, h("span", { class: "k" }, "Novice unresolved terms"),
                h("span", { class: "v" }, numBtn(String(row.novice_unresolved), { kind: "unresolved", slide: n, persona: "novice" }, state))));
          }),
          rollup.hardest.length > 5 && h("div", { style: "margin-top:10px" }, h("button", { class: "btn small", on: { click: toggle } }, showAll ? "Show fewer" : `Show all ${rollup.hardest.length} slides`)))
      : h("p", { class: "empty" }, "No slide has all three readings yet."));
}
