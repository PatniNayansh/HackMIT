import { h, mount, f2, describe, warnIcon, infoIcon, slideImage, stateChip, STATE_MEANING, meaningOf, PERSONAS, LABEL, COLOR } from "./dom.js";
import { getJSON } from "./api.js";
import { watch } from "./run.js";
import { numBtn, cmpLine, deckStrip, src, usesTakeawayIntent, isFieldwise, openProvenance } from "./provenance.js";
import { runHref, tierChip, guard, runBanners } from "./run-common.js";

const REFERENCE_NOTE = "This reading defines the baseline; it is not scored against it.";

// ------------------------------------------------------- recommendations, lazily fetched
// Opening a slide is what asks the server for its recommendations; the server saves them with
// the run, so a reopened or replayed slide gets them from disk. One entry per slide, shared by
// every repaint while a run is still streaming.

const recs = new Map();

function useRecs(runId, n, paint) {
  const key = `${runId}:${n}`;
  let e = recs.get(key);
  if (!e || e.status === "error") {
    e = { status: "loading", data: null, message: "", waiters: new Set() };
    recs.set(key, e);
    getJSON(`/api/runs/${encodeURIComponent(runId)}/slides/${n}/recommendations`)
      .then((body) => Object.assign(e, body.available ? { status: "ok", data: body.recommendations } : { status: "unavailable", message: body.reason }))
      .catch((err) => Object.assign(e, { status: "error", message: err.message }))
      .finally(() => e.waiters.forEach((fn) => fn(e)));
  }
  e.waiters.add(paint);
  if (e.status !== "loading") queueMicrotask(() => paint(e));
  else paint(e);
  return () => e.waiters.delete(paint);
}

function recsView(persona, e, retry) {
  if (e.status === "loading") return h("p", { class: "muted small" }, h("span", { class: "spinner" }), " Working out what to change…");
  if (e.status === "error") return h("div", null, h("p", { class: "err" }, e.message), h("button", { class: "btn small", on: { click: retry } }, "Try again"));
  if (e.status === "unavailable") return h("p", { class: "muted small" }, e.message);
  const items = e.data[persona];
  if (!items.length) return h("p", { class: "muted small" }, "Nothing to change for this audience: they had little trouble.");
  return h("ul", { class: "recs" }, ...items.map((r) => h("li", null,
    h("div", { class: "bullet" }, r.bullet),
    h("blockquote", null, r.evidence))));
}

function expertFlags(reading, e) {
  // Terms the expert could not resolve come straight from its reading, so they show at once; the
  // model may add a contradiction once the recommendations are in.
  const flags = e.status === "ok" ? e.data.expert_flagged
    : reading.unresolved_terms.map((t) => ({ note: `The expert could not resolve “${t}”.`, evidence: t }));
  if (!flags.length) return null;
  return h("div", { class: "flag" }, warnIcon(), h("div", null,
    h("strong", null, "Expert also flagged"),
    h("ul", null, ...flags.map((f) => h("li", null, f.note, " ", h("span", { class: "muted" }, "“", f.evidence, "”"))))));
}

// ----------------------------------------------------- the neural layer, lazily fetched
// Precomputed only (spec 5): the server reads what scripts/precompute_neural/ already wrote on a
// CUDA machine, and 404s when there is nothing there, so nothing on this page can trigger the
// 6-13 GPU-minutes-per-slide model call. One fetch per slide, shared by every repaint while a run
// is still streaming.

const neural = new Map();

function fetchNeural(runId, n) {
  const key = `${runId}:${n}`;
  if (!neural.has(key)) {
    neural.set(key, getJSON(`/api/runs/${encodeURIComponent(runId)}/slides/${n}/neural`)
      .then((data) => ({ status: "ok", data }))
      .catch((err) => (err.status === 404 ? { status: "absent" } : { status: "error", message: err.message })));
  }
  return neural.get(key);
}

/** The empty state is an inert, unlit fsaverage5 surface: the same render TRIBE produces with no
 *  response laid over it. Nothing is lit because nothing was predicted -- it stands in for the
 *  absence, never for a result (spec 9, screen 5). */
function neuralAbsent() {
  return h("div", { class: "neural-empty" },
    slideImage("/static/img/brain-unlit.png", "", "neural-empty-brain"),
    h("p", { class: "empty" },
      "No precomputed neural response for this slide. The neural layer runs offline on a GPU, only for the bundled sample decks."));
}

function neuralBody(d) {
  return h("div", null,
    h("div", { class: "rel-row" },
      h("div", null,
        h("div", { class: "label-xs" }, "Processing ratio"),
        h("div", { class: "val" }, f2(d.processing_ratio)),
        h("p", { class: "caveat" }, "Language drive ÷ visual drive. Read across this deck, not on one slide.")),
      h("div", null,
        h("div", { class: "label-xs" }, "Language drive"), h("div", null, f2(d.language_drive)),
        h("div", { class: "label-xs", style: "margin-top:10px" }, "Visual drive"), h("div", null, f2(d.visual_drive)))),
    h("details", null,
      h("summary", { class: "small muted", style: "cursor:pointer" }, "Narration this prediction was made from"),
      h("div", { class: "textbox" }, d.narration_transcript)),
    h("div", { class: "surface-grid" }, ...Object.entries(d.views).map(([name, url]) =>
      h("figure", null,
        slideImage(url, `Predicted response, ${name.replace(/_/g, " ")}`),
        h("figcaption", { class: "small muted" }, name.replace(/_/g, " "))))));
}

function neuralSection(runId, n) {
  const box = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading…"));
  fetchNeural(runId, n).then((e) => box.replaceChildren(
    e.status === "ok" ? neuralBody(e.data)
    : e.status === "absent" ? neuralAbsent()
    : h("p", { class: "err" }, e.message)));
  return h("section", { class: "card neural" },
    h("h3", null, "Predicted neural response"),
    // The one disclosure this panel makes, in the dek position rather than as a warning: TRIBE
    // predicts an average cortical response, and the slide was read to it by a speech synthesiser.
    h("p", { class: "dek" }, "TRIBE v2 (Meta AI), from synthesized narration of this slide. ",
      h("strong", null, "Predicted, not measured.")),
    box);
}


// ------------------------------------------------------------------ the cards

function badCard(persona, rd, profile) {
  const message = rd.ok ? `Confidence ${rd.confidence} is outside 0–1.` : rd.error.message;
  const attempts = rd.ok ? [] : rd.error.attempts;
  return h("article", { class: `aud ${persona} bad` },
    h("div", { class: "aud-head" }, h("span", { class: "who" }, h("span", { class: `dot ${persona}` }), LABEL[persona]), h("span", { class: "muted small" }, describe(persona, profile))),
    h("div", { class: "aud-body" },
      h("div", { class: "banner bad", style: "margin-top:8px" }, warnIcon(), h("div", null,
        h("strong", null, "Bad response, not used. "), message,
        h("p", { class: "small", style: "margin-top:4px" }, "The model’s reply failed validation. It is shown as an error rather than clamped or repaired, and no alignment or tier was computed for this slide."))),
      attempts.length > 0 && h("details", null, h("summary", null, `What the model returned (${attempts.length} attempt${attempts.length === 1 ? "" : "s"})`),
        h("ul", null, ...attempts.map((a) => h("li", { class: "mono" }, a))))));
}

function metricRow(cls, label, valueNode, ...right) {
  return h("div", { class: `metric-row ${cls}` },
    h("div", { class: "left" }, h("span", { class: "lbl" }, label), valueNode),
    h("div", { class: "right" }, ...right));
}

function audienceCard(state, r, persona, recsSlot) {
  const rd = r.readings[persona];
  const n = r.index;
  if (isFieldwise(r) && rd.ok) return fieldwiseCard(state, r, persona, recsSlot);
  // The expert's takeaway is the slide's intent and is shown once, under the slide image; the
  // card does not repeat it. (Runs saved before that measured against a rephrased sentence, so
  // their expert card keeps its takeaway: it is not the text shown as the intent.)
  const takeawayShownAsIntent = persona === "expert" && usesTakeawayIntent(r);
  // The persona's reply is validated when it is read, so a stored value outside range is only
  // possible in a damaged file. Either way it is an error to show, never a number to trust.
  const damaged = rd.ok && !(typeof rd.confidence === "number" && rd.confidence >= 0 && rd.confidence <= 1);
  if (!rd.ok || damaged) return badCard(persona, rd, state.meta.profile);

  const color = COLOR[persona];
  const terms = rd.unresolved_terms;
  const align = r.metrics?.intent_alignment[persona];
  const rows = [
    metricRow("strong", "Unresolved terms", h("span", { class: "val" }, numBtn(String(terms.length), { kind: "unresolved", slide: n, persona }, state)),
      deckStrip(state, `unresolved_count.${persona}`, n, color), cmpLine(state, `unresolved_count.${persona}`, n),
      terms.length ? h("span", { class: "preview" }, terms.slice(0, 4).join(" · "), terms.length > 4 ? ` +${terms.length - 4} more` : "") : null),
  ];
  if (align?.definitional) {
    // The expert IS the reference. The word stands where the others show a number.
    rows.push(metricRow("strong", "Alignment to the intended reading",
      h("span", { class: "val ref" }, numBtn("reference", { kind: "reference", slide: n }, state, "reference. Show why this is definitional.")),
      h("span", { class: "cmp" }, REFERENCE_NOTE)));
  } else if (align) {
    rows.push(metricRow("strong", "Alignment to the intended reading", h("span", { class: "val" }, numBtn(f2(align.value), { kind: "alignment", slide: n, persona }, state)),
      deckStrip(state, `intent_alignment.${persona}`, n, color), cmpLine(state, `intent_alignment.${persona}`, n),
      h("span", { class: "cmp" }, "Semantic similarity: the weaker instrument.")));
  } else {
    rows.push(metricRow("weak", "Alignment to the intended reading", h("span", { class: "val" }, "—"), h("span", { class: "cmp" }, "Not computed for this slide.")));
  }

  return h("article", { class: `aud ${persona}` },
    h("div", { class: "aud-head" }, h("span", { class: "who" }, h("span", { class: `dot ${persona}` }), LABEL[persona]), h("span", { class: "muted small" }, describe(persona, state.meta.profile))),
    h("div", { class: "aud-body" },
      takeawayShownAsIntent
        ? h("p", { class: "muted small", style: "margin: 6px 0 4px" }, "Its takeaway is the intent of this slide, shown under the slide image.")
        : [h("div", { class: "take-label" }, "Takeaway, verbatim"), h("blockquote", { class: "take" }, `“${rd.takeaway}”`)],
      ...rows,
      recsSlot && h("div", { class: "recs-slot" }, h("h4", null, "What to change"), recsSlot),
      h("details", null, h("summary", null, "More from this reading"),
        h("div", { class: "stack", style: "margin-top:8px" },
          h("div", null, h("div", { class: "take-label" }, "Claim it thinks you want believed"), h("p", null, rd.inferred_claim)),
          h("div", null, h("div", { class: "take-label" }, "Questions it would need answered"),
            rd.questions.length ? h("ul", null, ...rd.questions.map((q) => h("li", null, q))) : h("p", { class: "muted" }, "None"))))));
}

// ------------------------------------------------------------ field-wise (compare.py) cards

const FIELD_LABEL = { concept: "Concept", claim: "Claim", result: "Result", vehicle: "Example" };
const STATUS_TEXT = { excluded: "not part of this slide", gap: "not reached", over_reach: "extra (over-reach)", not_scored: "shown, not scored" };

function statusCell(state, r, persona, field, c) {
  const word = c.status === "compared"
    ? (c.outcome === "match" ? "matches" : c.outcome === "near" ? "near match" : c.outcome === "mismatch" ? "differs" : c.outcome)
    : STATUS_TEXT[c.status];
  const cls = c.status === "gap" ? "gap" : c.status === "excluded" || c.status === "not_scored" ? "quiet" : "";
  return numBtn(word, { kind: field === "claim" && c.status !== "excluded" ? "state" : "field", slide: r.index, persona, field }, state, `${FIELD_LABEL[field]}: ${word}. Show where this comes from.`);
}

/** The four fields for one reader, beside the expert's. Absent, excluded and extra are visibly different things. */
function fieldsTable(state, r, persona) {
  const m = r.metrics, comps = persona === "expert" ? null : m.comparisons[persona];
  const val = (v) => (v ? h("span", { class: "fv" }, v) : h("span", { class: "fv none" }, "—"));
  return h("table", { class: "fields" },
    h("thead", null, h("tr", null, h("th", null, "Field"), persona !== "expert" && h("th", null, "Expert"), h("th", null, persona === "expert" ? "Extracted from its takeaway" : `${LABEL[persona]}`), persona !== "expert" && h("th", null, "Reading"))),
    h("tbody", null, ...["concept", "claim", "result", "vehicle"].map((f) => {
      const c = comps?.[f];
      const inProfile = m.slide_profile.fields.includes(f);
      return h("tr", { class: `${c ? c.status : ""}${persona === "expert" && !inProfile ? " excluded" : ""}` },
        h("th", { scope: "row" }, FIELD_LABEL[f]),
        persona !== "expert" && h("td", null, val(c.expert)),
        h("td", null, val(persona === "expert" ? m.fields.expert[f] : c.audience)),
        persona !== "expert" && h("td", null, statusCell(state, r, persona, f, c)));
    })));
}

/** The finding in plain words: the propositions of the expert's claim this reader missed (and any it
 *  contradicted). The state chip beside it opens the table behind them. Nothing for runs saved before coverage. */
function missedLines(state, r, persona) {
  const cov = r.metrics.comparisons[persona].claim.coverage;
  if (!cov) return [];
  const line = (label, texts, cls) => texts.length
    ? h("p", { class: `missed ${cls}` }, h("strong", null, label), " ", ...texts.flatMap((t, i) => [i ? "; " : null, h("em", null, t)]))
    : null;
  return [line("Contradicts:", cov.contradicted, "contradicts"), line("Missed:", cov.missed, "omitted")].filter(Boolean);
}

function findingsFor(state, r, persona) {
  const idx = r.metrics.findings.map((f, i) => [f, i]).filter(([f]) => f.audience === persona);
  return idx.map(([f, i]) => h("div", { class: `finding-callout ${f.id}` },
    infoIcon(),
    numBtn(f.text, { kind: "finding", slide: r.index, index: i }, state, `${f.text} Show the rule and the quoted evidence.`)));
}

function fieldwiseCard(state, r, persona, recsSlot) {
  const rd = r.readings[persona], n = r.index, m = r.metrics;
  const head = h("div", { class: "aud-head" },
    h("span", { class: "who" }, h("span", { class: `dot ${persona}` }), LABEL[persona]),
    h("span", { class: "muted small" }, describe(persona, state.meta.profile)));
  const terms = rd.unresolved_terms;
  const termsRow = metricRow("strong", "Unresolved terms", h("span", { class: "val" }, numBtn(String(terms.length), { kind: "unresolved", slide: n, persona }, state)),
    deckStrip(state, `unresolved_count.${persona}`, n, COLOR[persona]), cmpLine(state, `unresolved_count.${persona}`, n),
    terms.length ? h("span", { class: "preview" }, terms.slice(0, 4).join(" \u00b7 "), terms.length > 4 ? ` +${terms.length - 4} more` : "") : null);

  if (persona === "expert") {
    return h("article", { class: "aud expert" }, head,
      h("div", { class: "aud-body" },
        h("p", { class: "muted small", style: "margin: 6px 0 4px" }, "Its takeaway is the intent of this slide, shown under the slide image."),
        metricRow("strong", "Reading of this slide", h("span", { class: "val ref" }, numBtn("reference", { kind: "reference", slide: n }, state, "reference. Show why this is definitional.")),
          h("span", { class: "cmp" }, REFERENCE_NOTE)),
        h("div", { class: "fields-wrap" }, h("div", { class: "take-label" }, "What its takeaway names (this defines the slide’s shape)"), fieldsTable(state, r, "expert")),
        termsRow,
        h("details", null, h("summary", null, "More from this reading"),
          h("div", { class: "stack", style: "margin-top:8px" },
            h("div", null, h("div", { class: "take-label" }, "Questions it would need answered"),
              rd.questions.length ? h("ul", null, ...rd.questions.map((q) => h("li", null, q))) : h("p", { class: "muted" }, "None"))))));
  }

  const claim = m.comparisons[persona].claim;
  const over = ["concept", "claim", "result"].filter((f) => m.comparisons[persona][f].status === "over_reach");
  return h("article", { class: `aud ${persona}` }, head,
    h("div", { class: "aud-body" },
      // Words lead: the state of the claim, then anything the field logic found. No scalar.
      claim.status === "excluded"
        ? h("div", { class: "lead-row" }, h("span", { class: "muted small" }, "This slide states no general claim, so there is no claim to compare."))
        : h("div", { class: "lead-row" },
            h("span", { class: "lead-label" }, "Claim"),
            h("button", { class: "state-btn", type: "button", "aria-label": `${claim.outcome}. Show the propositions and the quoted spans behind it.`, on: { click: (e) => { e.stopPropagation(); openProvenance({ kind: "state", slide: n, persona }, state); } } }, stateChip(claim.outcome, { large: true })),
            h("span", { class: "muted small" }, meaningOf(claim))),
      ...missedLines(state, r, persona),
      ...findingsFor(state, r, persona),
      h("div", { class: "take-label", style: "margin-top:12px" }, "Takeaway, verbatim"),
      h("blockquote", { class: "take" }, `“${rd.takeaway}”`),
      h("div", { class: "fields-wrap" }, fieldsTable(state, r, persona)),
      over.length ? h("p", { class: "muted small" }, `Beyond the expert: this reader also gave a ${over.join(" and a ")} the expert did not. Not penalised.`) : null,
      termsRow,
      recsSlot && h("div", { class: "recs-slot" }, h("h4", null, "What to change"), recsSlot),
      h("details", null, h("summary", null, "More from this reading"),
        h("div", { class: "stack", style: "margin-top:8px" },
          h("div", null, h("div", { class: "take-label" }, "Claim it thinks you want believed"), h("p", null, rd.inferred_claim)),
          h("div", null, h("div", { class: "take-label" }, "Questions it would need answered"),
            rd.questions.length ? h("ul", null, ...rd.questions.map((q) => h("li", null, q))) : h("p", { class: "muted" }, "None"))))));
}

/** Headline, field-wise: what the field logic found (findings), the tier, and what the tier was read from. */
function fieldwiseSummary(state, r) {
  const n = r.index, m = r.metrics;
  const t = state.rollup.per_slide.find((p) => p.slide === n)?.tier;
  const findings = m.findings.map((f, i) => [f, i]);
  const basis = t && (t.basis === "relative"
    ? `The terms check is read against the other slides of this deck (${t.n_slides} scored so far).`
    : `Too few slides scored to read the terms check against this deck (needs ${t.min_slides_for_relative}, has ${t.n_slides}): a rough absolute threshold was used.`);
  const stateRow = (persona) => {
    const c = m.comparisons[persona].claim;
    if (c.status === "excluded") return null;
    const cov = c.coverage, gaps = cov ? [...cov.contradicted, ...cov.missed] : [];
    const detail = cov ? `${cov.covered} of ${cov.total} propositions covered.${gaps.length ? ` ${cov.contradicted.length ? "Contradicts" : "Missed"}: \u201c${gaps.join("\u201d; \u201c")}\u201d` : ""}` : meaningOf(c);
    return metricRow("strong", `${LABEL[persona]} claim`, h("span", { class: "val" }, numBtn(c.outcome, { kind: "state", slide: n, persona }, state)), h("span", { class: "cmp" }, detail));
  };
  const presence = (persona, field) => {
    const c = m.comparisons[persona][field];
    if (!m.slide_profile.scored.includes(field)) return null; // only fields in the slide profile are read
    return metricRow("strong", `${LABEL[persona]} ${field}`, h("span", { class: "val" }, numBtn(c.status === "gap" ? "not reached" : c.outcome === "mismatch" ? "differs" : "reached", { kind: "field", slide: n, persona, field }, state)),
      h("span", { class: "cmp" }, c.status === "gap" ? `The expert reached \u201c${c.expert}\u201d.` : c.outcome === "mismatch" ? `Expert \u201c${c.expert}\u201d, reader \u201c${c.audience}\u201d.` : `\u201c${c.audience}\u201d`));
  };
  return h("section", { class: "summary card" },
    h("div", { class: "small muted" }, "What this slide demands of its reader"),
    findings.length ? h("div", { class: "findings" }, ...findings.map(([f, i]) => h("div", { class: `finding-callout lead ${f.id}` }, warnIcon(),
      h("div", null, h("strong", null, f.id === "example_bound" ? "Example-bound. " : "Figure-dependent. "),
        numBtn(f.text, { kind: "finding", slide: n, index: i }, state, `${f.text} Show the rule and the quoted evidence.`))))) : null,
    t
      ? [h("div", { class: "summary-head" }, tierChip(state, n, { large: true }), h("span", { class: "tier-meaning" }, t.meaning)),
         h("p", { class: "muted small" }, basis, state.meta.status === "running" && " Tiers are read against the slides read so far and can shift as more arrive.")]
      : h("div", { class: "banner", style: "margin-top:8px" }, infoIcon(), h("div", null, h("strong", null, "No tier. "), "The expert’s takeaway gave this slide no concept, claim or result to judge a reader by.")),
    h("div", { class: "small muted", style: "margin-top:12px" }, "What it was read from"),
    stateRow("novice"), stateRow("peer"),
    ...["concept", "result"].map((f) => presence("novice", f)),
    metricRow("strong", "Novice unresolved terms", h("span", { class: "val" }, numBtn(String(r.readings.novice.unresolved_terms.length), { kind: "unresolved", slide: n, persona: "novice" }, state)),
      deckStrip(state, "unresolved_count.novice", n, COLOR.novice), cmpLine(state, "unresolved_count.novice", n)));
}

/** "This slide names a principle and works to a numeric result." How to read everything below it. */
function profileLine(state, r) {
  if (!isFieldwise(r)) return null;
  return h("p", { class: "profile-line" }, numBtn(r.metrics.slide_profile.text, { kind: "profile", slide: r.index }, state, `${r.metrics.slide_profile.text} Show how this shape was read.`));
}

/** The figure on the slide, described neutrally at ingest. Collapsed, and labelled as machine-generated. */
function figureDetails(r) {
  const ic = r.image_content;
  if (!ic) return null;
  if (!ic.text) return h("p", { class: "muted small" }, `A description of this slide’s figure could not be produced (${ic.error || "no usable description"}). The readers still saw the image itself.`);
  return h("details", { class: "figure-desc" },
    h("summary", { class: "small muted", style: "cursor:pointer" }, "Figure description (machine-generated)"),
    h("div", { class: "src" }, h("div", { class: "who" }, `Written by ${ic.model || "a model"} from the slide image; given to all three readers under FIGURE:`), h("p", null, ic.text)),
    h("p", { class: "caveat" }, "Marks and labels only, never a principle — so the readers are not handed the expert’s job. Not scored."));
}

/** The headline: which tier the slide falls in, and the three numbers it was read from. */
function summary(state, r) {
  if (isFieldwise(r)) return fieldwiseSummary(state, r);
  const n = r.index;
  const t = state.rollup.per_slide.find((p) => p.slide === n)?.tier;
  if (!r.metrics || !t) {
    return h("section", { class: "summary card" },
      h("div", { class: "small muted" }, "What this slide demands of its reader"),
      h("div", { class: "banner bad", style: "margin-top:8px" }, warnIcon(), h("div", null, h("strong", null, "No tier. "), r.metrics_error || "Not computed for this slide.")));
  }
  const m = r.metrics;
  const basis = t.basis === "relative"
    ? `Read against the other slides of this deck (${t.n_slides} scored so far).`
    : `Too few slides scored to compare with this deck (needs ${t.min_slides_for_relative}, has ${t.n_slides}): rough absolute thresholds used.`;
  const row = (label, valueNode, key, color, ...extra) => metricRow("strong", label, h("span", { class: "val" }, valueNode),
    deckStrip(state, key, n, color), cmpLine(state, key, n), ...extra);
  return h("section", { class: "summary card" },
    h("div", { class: "small muted" }, "What this slide demands of its reader"),
    h("div", { class: "summary-head" }, tierChip(state, n, { large: true }), h("span", { class: "tier-meaning" }, t.meaning)),
    h("p", { class: "muted small" }, basis, state.meta.status === "running" && " Tiers are read against the slides read so far and can shift as more arrive."),
    h("div", { class: "small muted", style: "margin-top:12px" }, "What it was read from"),
    row("Novice alignment", numBtn(f2(m.intent_alignment.novice.value), { kind: "alignment", slide: n, persona: "novice" }, state), "intent_alignment.novice", COLOR.novice),
    row("Peer alignment", numBtn(f2(m.intent_alignment.peer.value), { kind: "alignment", slide: n, persona: "peer" }, state), "intent_alignment.peer", COLOR.peer),
    row("Novice unresolved terms", numBtn(String(r.readings.novice.unresolved_terms.length), { kind: "unresolved", slide: n, persona: "novice" }, state), "unresolved_count.novice", COLOR.novice));
}

/** The ONE place the slide page states what the slide is for. In the current pipeline the slide's
 *  intent IS the expert persona's takeaway, and alignment was measured against exactly this string,
 *  so it is shown verbatim (rendered as returned: no truncation, no re-casing, no added full stop),
 *  in the same green-dot block used in the provenance panels.
 *
 *  Runs saved by the previous version were measured against a rephrased sentence instead. For
 *  those, the page must show the string that was actually measured, with its attribution, so it
 *  never claims one thing while the metrics used another. */
function intentBlock(r) {
  const si = r.slide_intent;
  if (!si) {
    return h("div", { class: "intent-box" },
      h("div", { class: "intent-title" }, "Intent of this slide"),
      h("p", { class: "intent-text muted" }, "No intent for this slide: the expert’s reply was unusable, and the intent is the expert’s takeaway."));
  }
  if (usesTakeawayIntent(r)) {
    return h("div", { class: "intent-box takeaway" },
      h("div", { class: "intent-title" }, "Intent of this slide"),
      h("div", { class: "intent-src" }, src("Expert takeaway, verbatim", si.text, "expert")));
  }
  return h("div", { class: "intent-box" },
    h("div", { class: "intent-title" }, "Intent of this slide"),
    h("p", { class: "intent-text" }, si.text),
    h("p", { class: "intent-attr" }, "Inferred from the expert reading"),
    si.source === "template" && h("p", { class: "muted small" },
      `Built directly from the expert’s claim rather than rephrased by the intent model (${si.reason}).`));
}

export function detail(root, runId, n) {
  let cleanups = [];
  const off = watch(runId, (state) => {
    cleanups.forEach((fn) => fn());
    cleanups = [];
    if (guard(root, state)) return;
    const { meta } = state;
    if (meta.legacy) { location.hash = `#/run/${runId}`; return; }
    const total = meta.slide_count;
    const r = state.results.get(n);
    document.title = `Slide ${n} — ${meta.title} — Sightline`;

    if (n < 1 || n > total) { mount(root, h("p", { class: "err" }, `This deck has ${total} slides.`), h("a", { href: runHref(state) }, "Back to the overview")); return; }

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
      slideImage(state.imageUrls[n - 1], `Slide ${n}`),
      intentBlock(r),
      figureDetails(r),
      h("details", null, h("summary", { class: "small muted", style: "cursor:pointer" }, "Text the audiences were given, alongside the image"),
        h("div", { class: "textbox" }, r.text || "(no extractable text on this slide)")));

    // Novice and peer cards get a slot the recommendations fill; the expert gets none.
    const slots = { novice: h("div"), peer: h("div") };
    const flagSlot = h("div");
    const canRecommend = r.metrics && r.slide_intent && PERSONAS.every((p) => r.readings[p].ok);
    const cards = PERSONAS.map((p) => audienceCard(state, r, p, canRecommend ? slots[p] : null));
    if (canRecommend) {
      const paint = (e) => {
        for (const p of ["novice", "peer"]) slots[p].replaceChildren(recsView(p, e, () => { recs.delete(`${runId}:${n}`); cleanups.forEach((fn) => fn()); cleanups = [useRecs(runId, n, paint)]; }));
        flagSlot.replaceChildren(...[expertFlags(r.readings.expert, e)].filter(Boolean));
      };
      cleanups.push(useRecs(runId, n, paint));
    } else if (r.readings.expert.ok) {
      flagSlot.replaceChildren(...[expertFlags(r.readings.expert, { status: "loading" })].filter(Boolean));
    }

    mount(root, head, chips, ...runBanners(state),
      h("div", { class: "detail" }, left,
        h("div", null, flagSlot, profileLine(state, r), summary(state, r), ...cards, neuralSection(runId, n))));
  });
  return () => { cleanups.forEach((fn) => fn()); off(); };
}
