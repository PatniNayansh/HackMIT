// Renders the real overview and slide views in Node against a real API, using a tiny DOM stand-in,
// and prints what each screen says as JSON. It cannot judge layout or colour; it catches runtime
// errors and lets tests assert what text is (and is not) on each screen.
//
//   node render.mjs http://localhost:8000 <run_id> <slide,slide,...>
//   node render.mjs payload.json           (offline: {"run": {...}, "recs": {"<slide>": {...}}})
import fs from "node:fs";

class Node_ {
  constructor() { this.childNodes = []; this.parentNode = null; }
  append(...kids) { for (const k of kids) { const n = k instanceof Node_ ? k : new Text_(k); n.parentNode = this; this.childNodes.push(n); } }
  replaceChildren(...kids) { this.childNodes = []; this.append(...kids); }
  remove() { if (this.parentNode) this.parentNode.childNodes = this.parentNode.childNodes.filter((c) => c !== this); }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
}
class Text_ extends Node_ { constructor(t) { super(); this.data = String(t); } get textContent() { return this.data; } }
class El extends Node_ {
  constructor(tag) { super(); this.tagName = tag; this.attrs = {}; this.style = {}; this.listeners = {}; this.className = ""; this.value = ""; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
  removeEventListener() {}
  focus() {}
  click() { (this.listeners.click || []).forEach((fn) => fn({ stopPropagation() {}, preventDefault() {} })); }
  getBoundingClientRect() { return { left: 0, width: 1000 }; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] ?? null; }
  querySelectorAll(sel) {
    const out = [];
    const walk = (n) => { for (const c of n.childNodes) { if (c instanceof El) { if (matches(c, sel)) out.push(c); walk(c); } } };
    walk(this);
    return out;
  }
  get children() { return this.childNodes.filter((c) => c instanceof El); }
}
function matches(el, sel) {
  return sel.split(",").some((s) => {
    s = s.trim();
    if (s.startsWith(".")) return s.slice(1).split(".").every((c) => (el.className?.baseVal ?? el.attrs.class ?? el.className ?? "").split(/\s+/).includes(c));
    return el.tagName === s;
  });
}
const doc = new El("document");
doc.createElement = (t) => new El(t);
doc.createElementNS = (_, t) => new El(t);
doc.createTextNode = (t) => new Text_(t);
doc.body = new El("body");
doc.addEventListener = () => {};
doc.removeEventListener = () => {};
doc.activeElement = null;
globalThis.Node = Node_;
globalThis.document = doc;
globalThis.location = { hash: "" };
globalThis.window = globalThis;
globalThis.scrollTo = () => {};

// `class` set through setAttribute lands in attrs; normalise so matches() sees it either way.
const origSet = El.prototype.setAttribute;
El.prototype.setAttribute = function (k, v) { origSet.call(this, k, v); if (k === "class") this.className = String(v); };

const [target, runId, slideList] = process.argv.slice(2);
let route;
if (target.startsWith("http")) {
  const real = globalThis.fetch;
  globalThis.fetch = (url, opts) => real(target + url, opts);
} else {
  const payload = JSON.parse(fs.readFileSync(target, "utf8"));
  route = (url) => {
    if (/\/recommendations$/.test(url)) {
      const n = url.match(/slides\/(\d+)\/recommendations/)[1];
      return payload.recs[n] ?? { available: false, reason: "not saved" };
    }
    return payload.run;
  };
  globalThis.fetch = async (url) => ({ ok: true, json: async () => route(url) });
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const text = (node) => node.textContent.replace(/\s+/g, " ").trim();

const { overview } = await import("../js/views-overview.js");
const { detail } = await import("../js/views-detail.js");
const { openProvenance, closeDrawer } = await import("../js/provenance.js");
const { currentState } = await import("../js/run.js");

const id = runId ?? JSON.parse(fs.readFileSync(target, "utf8")).run.meta.run_id;
const out = { errors: [] };
const errors = console.error;
console.error = (...a) => out.errors.push(a.map(String).join(" "));

async function screen(name, view, ...args) {
  const root = new El("main");
  const off = view(root, id, ...args);
  await wait(150);
  out[name] = text(root);
  out[`${name}:html`] = root;
  return { root, off };
}

const ov = await screen("overview", overview);
out.overview_buttons = ov.root.querySelectorAll("button").length;
// Hover the narrative arc: record what its tooltip says at a few positions, and count hollow markers.
const arcSvg = ov.root.querySelectorAll("svg").find((v) => /across slide order|four rungs/.test(v.attrs["aria-label"] ?? ""));
if (arcSvg) {
  out.arc_tooltips = [];
  for (const x of [120, 500, 900]) {
    (arcSvg.listeners.pointermove || []).forEach((fn) => fn({ clientX: x }));
    const tip = arcSvg.parentNode.querySelectorAll(".tip")[0];
    out.arc_tooltips.push(tip ? text(tip) : "");
  }
  out.arc_hollow_markers = arcSvg.querySelectorAll("circle").filter((c) => c.attrs.fill === "var(--surface)").length;
  out.arc_label = arcSvg.attrs["aria-label"];
}
ov.off?.();

const slides = (slideList ?? "1").split(",").map(Number);
for (const n of slides) {
  const s = await screen(`slide${n}`, detail, n);
  // open every provenance drawer the slide offers and record what it says
  const drawers = [];
  for (const btn of s.root.querySelectorAll("button")) {
    if (!btn.className.includes("num") && !btn.className.includes("tier-chip")) continue;
    closeDrawer();
    btn.click();
    drawers.push({ opener: text(btn), body: text(doc.body) });
  }
  closeDrawer();
  out[`slide${n}:drawers`] = drawers;
  s.off?.();
}
for (const k of Object.keys(out)) if (k.endsWith(":html")) delete out[k];
console.error = errors;
// Exit only once the (large) output has been flushed to a pipe.
process.stdout.write(JSON.stringify(out), () => process.exit(0));
