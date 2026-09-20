// Pieces shared by the overview and the slide page.
import { h, warnIcon, infoIcon } from "./dom.js";
import { openProvenance } from "./provenance.js";

export const runHref = (state, n) => `#/run/${state.meta.run_id}${n ? `/slide/${n}` : ""}`;

export function statusPill(meta) {
  if (meta.status === "complete") return h("span", { class: "pill" }, "complete");
  if (meta.status === "running") return h("span", { class: "pill outline" }, h("span", { class: "spinner" }), "running");
  return h("span", { class: "pill bad" }, meta.status);
}

export function subfieldLine(meta) {
  const p = meta.profile;
  if (!p) return null;
  return h("p", { class: "muted small" },
    `Subfield: ${p.domain} · peer comes from: ${p.adjacent_field} `,
    h("span", { class: "pill" }, p.edited ? "edited by presenter" : "as suggested, confirmed by presenter"));
}

/** The tier as a clickable chip. Tiers are labels for what a slide demands of its reader, not a
 *  grade: they differ by fill weight, never by a red/amber/green colour. */
export function tierChip(state, slide, { large = false } = {}) {
  const t = state.rollup.per_slide.find((p) => p.slide === slide)?.tier;
  if (!t) return null;
  return h("button", {
    class: `tier-chip ${t.tier}${large ? " large" : ""}`, type: "button",
    "aria-label": `${t.label}. Show how this tier was assigned.`,
    on: { click: (e) => { e.stopPropagation(); openProvenance({ kind: "tier", slide }, state); } },
  }, t.label);
}

/** Shared loading / error frame. Returns true if the caller should stop painting. */
export function guard(root, state) {
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

export function runBanners(state) {
  const { meta, rollup } = state;
  const out = [];
  if (meta.sample) out.push(h("div", { class: "banner sample" }, infoIcon(), h("div", null, h("strong", null, "sample data. "), "This is a bundled sample run for demonstration, not a review of your own deck.")));
  if (meta.legacy) {
    out.push(h("div", { class: "banner" }, infoIcon(), h("div", null,
      h("strong", null, "Saved in the earlier format. "),
      "This run measured alignment against one declared intent for the whole deck and reported confidence, blind-spot and divergence. Those views were retired, so it opens in a reduced form with the slides and the terms the novice could not resolve. ",
      "Start a new review of the same deck to get per-slide intended readings, tiers and recommendations.")));
  }
  if (meta.status === "failed" || meta.status === "interrupted") {
    out.push(h("div", { class: "banner bad" }, warnIcon(), h("div", null,
      h("strong", null, meta.status === "interrupted" ? "This run was cut off. " : "This run stopped. "),
      meta.error || "The server stopped before it finished.",
      state.results.size ? ` The ${state.results.size} slide${state.results.size === 1 ? "" : "s"} below finished before that.` : "",
      !meta.sample && [" ", h("a", { href: `#/run/${meta.run_id}/setup` }, "Set up and try again")])));
  }
  if (!meta.legacy && rollup.unscored.length) {
    out.push(h("div", { class: "banner bad" }, warnIcon(), h("div", null,
      h("strong", null, `Bad model response on ${rollup.unscored.length} slide${rollup.unscored.length === 1 ? "" : "s"}: `),
      ...rollup.unscored.flatMap((n, i) => [i ? ", " : "", h("a", { href: runHref(state, n) }, String(n))]),
      ". Those replies failed validation, so they are shown as errors, not repaired, and no alignment or tier was computed for those slides.")));
  }
  return out;
}
