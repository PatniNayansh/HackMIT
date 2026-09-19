// Tiny DOM helpers. Every string goes in as a text node, never as HTML, so slide text, model
// output and file names cannot inject markup.

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "value" || k === "checked" || k === "disabled") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, kids);
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else el.setAttribute(k, v);
  }
  append(el, kids);
  return el;
}

/** root.replaceChildren, minus the falsy values that native replaceChildren would print as text. */
export function mount(root, ...nodes) {
  root.replaceChildren(...nodes.flat(Infinity).filter((n) => n != null && n !== false));
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

// ---------------------------------------------------------------- formatting

export const f2 = (x) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(2);

export function ordinal(n) {
  const r = n % 100;
  if (r >= 11 && r <= 13) return `${n}th`;
  return `${n}${["th", "st", "nd", "rd"][n % 10] || "th"}`;
}

export function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// ------------------------------------------------------------------- personas

export const PERSONAS = ["novice", "peer", "expert"];
export const LABEL = { novice: "Novice", peer: "Peer", expert: "Expert" };
export const COLOR = { novice: "var(--c-novice)", peer: "var(--c-peer)", expert: "var(--c-expert)" };

export function describe(persona, profile) {
  if (!profile) return "";
  return {
    novice: `No training in ${profile.domain}`,
    peer: `Works in ${profile.adjacent_field}, not ${profile.domain}`,
    expert: `Works in ${profile.domain}`,
  }[persona];
}

// ---------------------------------------------------------------------- icons

export function warnIcon() {
  return s("svg", { width: 18, height: 18, viewBox: "0 0 20 20", "aria-hidden": "true" },
    s("path", { d: "M10 2.5 18.5 17h-17L10 2.5Z", fill: "none", stroke: "currentColor", "stroke-width": 1.6, "stroke-linejoin": "round" }),
    s("path", { d: "M10 8v4.2", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round" }),
    s("circle", { cx: 10, cy: 14.6, r: 0.9, fill: "currentColor" }));
}

export function infoIcon() {
  return s("svg", { width: 18, height: 18, viewBox: "0 0 20 20", "aria-hidden": "true" },
    s("circle", { cx: 10, cy: 10, r: 7.5, fill: "none", stroke: "currentColor", "stroke-width": 1.6 }),
    s("path", { d: "M10 9v5", stroke: "currentColor", "stroke-width": 1.6, "stroke-linecap": "round" }),
    s("circle", { cx: 10, cy: 6.4, r: 0.9, fill: "currentColor" }));
}

// An inert, unlit outline -- never colored or filled like a real prediction. Only ever
// used as an empty-state illustration when there is no precomputed neural data to show
// (spec 9 warns against a "placeholder brain" reading as a result; this stays plainly
// inert -- no colormap, no activation, no numbers -- so it cannot be mistaken for one).
export function brainIcon() {
  return s("svg", { width: 84, height: 68, viewBox: "0 0 100 80", "aria-hidden": "true" },
    s("path", {
      d: "M28 10 C15 10 8 20 8 32 C8 40 12 46 10 52 C8 58 14 64 22 64 C24 70 32 74 40 72 C46 76 56 76 62 72 C72 74 82 68 82 58 C90 56 92 46 86 40 C90 32 86 20 76 16 C74 8 62 4 52 8 C46 4 34 4 28 10 Z",
      fill: "none", stroke: "currentColor", "stroke-width": 2.2, "stroke-linejoin": "round",
    }),
    s("path", { d: "M40 14 C40 26 34 30 34 40 C34 50 42 52 40 62", fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round" }),
    s("path", { d: "M58 12 C60 24 68 26 66 38 C64 48 56 50 58 62", fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round" }),
    s("path", { d: "M46 18 C48 30 44 34 48 44", fill: "none", stroke: "currentColor", "stroke-width": 1.3, "stroke-linecap": "round" }));
}
