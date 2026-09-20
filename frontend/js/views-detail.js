import { h, mount, f2, describe, warnIcon, infoIcon, slideImage, PERSONAS, LABEL, COLOR } from "./dom.js";
import { getJSON } from "./api.js";
import { watch } from "./run.js";
import { numBtn, cmpLine, deckStrip, src, usesTakeawayIntent } from "./provenance.js";
import { runHref, tierChip, guard, runBanners } from "./run-common.js";

const REFERENCE_NOTE = "The intended reading is derived from this expert interpretation, so it defines the baseline rather than scoring against it.";

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

/** The headline: which tier the slide falls in, and the three numbers it was read from. */
function summary(state, r) {
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
        h("div", null, flagSlot, summary(state, r), ...cards)));
  });
  return () => { cleanups.forEach((fn) => fn()); off(); };
}
