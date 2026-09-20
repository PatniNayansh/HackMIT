import { h, s, f2, PERSONAS, LABEL, COLOR } from "./dom.js";

// ------------------------------------------------------------- strip vs the deck
// A one-slide number is only ever shown against the deck's own distribution: every other
// slide is a grey dot, the middle half of the deck is a faint band, this slide is the big dot.

const STRIP_W = 200, STRIP_H = 26, PAD = 8;

export function strip(dist, slide, color = "var(--ink)") {
  const mine = dist.values.find((v) => v.slide === slide);
  const lo = dist.min, hi = dist.max;
  const x = (v) => (hi === lo ? STRIP_W / 2 : PAD + ((v - lo) / (hi - lo)) * (STRIP_W - 2 * PAD));
  const y = STRIP_H / 2;
  return s("svg", {
    class: "strip", viewBox: `0 0 ${STRIP_W} ${STRIP_H}`, preserveAspectRatio: "xMinYMid meet", role: "img",
    "aria-label": `This slide ${f2(mine.value)}. Deck median ${f2(dist.median)}, range ${f2(lo)} to ${f2(hi)}, ${dist.n} slides.`,
  },
    s("line", { class: "axis", x1: PAD, x2: STRIP_W - PAD, y1: y, y2: y }),
    s("line", { class: "iqr", x1: x(dist.q1), x2: x(dist.q3), y1: y, y2: y }),
    s("line", { class: "median", x1: x(dist.median), x2: x(dist.median), y1: y - 7, y2: y + 7 }),
    ...dist.values.filter((v) => v.slide !== slide).map((v) => s("circle", { class: "other", cx: x(v.value), cy: y, r: 3 })),
    s("circle", { class: "here", cx: x(mine.value), cy: y, r: 5.5, fill: color }),
  );
}

// -------------------------------------------------------------- narrative arc
// Alignment to each slide's inferred intent, per persona, in slide order. The reference is the
// flat dashed line at 1.0. Novice and peer are measured; the expert IS the reference (the intent
// was derived from its reading), so it lies on the line, and is drawn and labelled as such. The
// space between each series and the reference is shaded: that gap is what is being read.

const W = 1000, H = 320, M = { l: 44, r: 108, t: 28, b: 30 };

export function arcChart(rollup, { onPoint, tableNumber }) {
  const pts = rollup.arc.filter((p) => PERSONAS.some((k) => p[k] != null));
  const definitional = new Set(rollup.definitional || []);
  const measured = PERSONAS.filter((k) => !definitional.has(k));
  const vals = pts.flatMap((p) => measured.map((k) => p[k])).filter((v) => v != null);
  if (!vals.length) return h("p", { class: "empty" }, "No slide has been scored yet.");

  const lo = Math.min(0, Math.floor(Math.min(...vals) * 4) / 4);
  const hi = 1;
  const n = pts.length;
  const X = (i) => (n === 1 ? (M.l + W - M.r) / 2 : M.l + (i / (n - 1)) * (W - M.l - M.r));
  const Y = (v) => H - M.b - ((Math.min(v, hi) - lo) / (hi - lo)) * (H - M.t - M.b);
  const kind = (k) => (definitional.has(k) ? "definitional today" : "measured");

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Alignment to each slide\u2019s intended reading for the novice, peer and expert across slide order, against the reference line at 1.0" });

  const ticks = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.25) ticks.push(t);
  svg.append(s("g", { class: "grid" }, ...ticks.map((t) => s("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }))));
  svg.append(...ticks.map((t) => s("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end" }, f2(t))));
  svg.append(s("g", { class: "axis" }, s("line", { x1: M.l, x2: W - M.r, y1: Y(lo), y2: Y(lo) })));

  const every = Math.ceil(n / 15);
  pts.forEach((p, i) => {
    if (i % every === 0 || i === n - 1) svg.append(s("text", { x: X(i), y: H - 8, "text-anchor": "middle" }, p.slide));
  });

  // Shade the gap between each measured series and the reference, per run of consecutive slides.
  for (const k of measured) {
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        const top = run.map((i) => `${X(i).toFixed(1)},${Y(hi).toFixed(1)}`);
        const bottom = [...run].reverse().map((i) => `${X(i).toFixed(1)},${Y(pts[i][k]).toFixed(1)}`);
        svg.append(s("polygon", { points: [...top, ...bottom].join(" "), fill: COLOR[k], opacity: 0.1 }));
      }
      run = [];
    };
    pts.forEach((p, i) => (p[k] == null ? flush() : run.push(i)));
    flush();
  }

  // The reference: flat, dashed, at 1.0, labelled.
  svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(hi), y2: Y(hi), stroke: "var(--ink-2)", "stroke-width": 1.5, "stroke-dasharray": "6 5" }));
  svg.append(s("text", { x: M.l, y: Y(hi) - 8, style: "fill: var(--ink-2); font-size: 12.5px" }, "inferred intent (reference)"));

  // Series lines: 2px, broken where a persona has no value for a slide. Same weight for all three.
  for (const k of PERSONAS) {
    let d = "", pen = false;
    pts.forEach((p, i) => {
      if (p[k] == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(p[k]).toFixed(1)}`;
      pen = true;
    });
    svg.append(s("path", { d, fill: "none", stroke: COLOR[k], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  }

  // Direct labels at the right edge; nudged apart, with leader lines when they would collide.
  const ends = PERSONAS.map((k) => {
    const i = pts.map((p) => p[k]).findLastIndex((v) => v != null);
    return i < 0 ? null : { k, x: X(i), y: Y(pts[i][k]) };
  }).filter(Boolean).sort((a, b) => a.y - b.y);
  let last = -Infinity;
  for (const e of ends) { e.ly = Math.max(e.y, last + 15); last = e.ly; }
  for (const e of ends) {
    if (Math.abs(e.ly - e.y) > 1) svg.append(s("line", { x1: e.x + 6, y1: e.y, x2: W - M.r + 8, y2: e.ly, stroke: "var(--baseline)", "stroke-width": 1 }));
    svg.append(s("text", { x: W - M.r + 12, y: e.ly + 4, style: "fill: var(--ink-2); font-size: 12.5px" }, `${LABEL[e.k]}${definitional.has(e.k) ? " (reference)" : ""}`));
  }

  // Markers: 8px+ dots with a 2px surface ring; a 24px transparent target opens the provenance.
  for (const k of PERSONAS) {
    pts.forEach((p, i) => {
      if (p[k] == null) return;
      svg.append(s("circle", { cx: X(i), cy: Y(p[k]), r: 4.5, fill: COLOR[k], stroke: "var(--surface)", "stroke-width": 2, "pointer-events": "none" }));
    });
  }
  const cross = s("line", { y1: M.t, y2: H - M.b, stroke: "var(--baseline)", "stroke-width": 1, style: "display:none", "pointer-events": "none" });
  svg.append(cross);
  for (const k of PERSONAS) {
    pts.forEach((p, i) => {
      if (p[k] == null) return;
      svg.append(s("circle", {
        class: "mk", cx: X(i), cy: Y(p[k]), r: 12, fill: "transparent", tabindex: 0, role: "button",
        "aria-label": definitional.has(k)
          ? `${LABEL[k]}, slide ${p.slide}, reference (definitional). Show why.`
          : `${LABEL[k]}, slide ${p.slide}, alignment ${f2(p[k])}. Show the text behind it.`,
        on: {
          click: () => onPoint(p.slide, k),
          keydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPoint(p.slide, k); } },
        },
      }));
    });
  }

  // Crosshair + one tooltip listing every persona at the nearest slide.
  const tip = h("div", { class: "tip" });
  const wrap = h("div", { class: "chart-wrap chart" }, svg, tip);
  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(n === 1 ? 0 : ((px - M.l) / (W - M.l - M.r)) * (n - 1))));
    const p = pts[i];
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.style.display = "";
    tip.replaceChildren(
      h("div", { class: "h" }, `Slide ${p.slide}`),
      ...PERSONAS.map((k) => h("div", { class: "r" },
        h("span", { class: "k" }, h("i", { style: `border-color:${COLOR[k]}` }), LABEL[k]),
        h("b", null, p[k] == null ? "not scored" : definitional.has(k) ? "reference" : f2(p[k])))),
      h("div", { class: "foot" }, "Click a dot for the text behind it"),
    );
    tip.style.display = "block";
    const cx = (X(i) / W) * box.width;
    tip.style.left = `${Math.min(cx + 14, box.width - 180)}px`;
    tip.style.top = "8px";
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerleave", () => { cross.style.display = "none"; tip.style.display = "none"; });

  const legend = h("div", { class: "legend" },
    ...PERSONAS.map((k) => h("span", null, h("i", { style: `border-color:${COLOR[k]}` }), LABEL[k], h("em", { class: "legend-kind" }, ` (${kind(k)})`))),
    h("span", null, h("i", { class: "ref-key" }), "inferred intent (reference)"));

  const note = definitional.size
    ? h("p", { class: "caveat", style: "margin-top:8px" },
        `${[...definitional].map((k) => LABEL[k]).join(" and ")} sits on the reference at 1.0 because the intended reading is derived from that reading: it is a definition, not a measurement. It is drawn anyway so the chart, the payload and the legend keep their shape when the expert is measured independently.`)
    : null;

  const table = h("details", { class: "table-view" },
    h("summary", null, "Show as a table"),
    h("div", { class: "scroll" },
      h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Slide"), ...PERSONAS.map((k) => h("th", { class: "right" }, LABEL[k])))),
        h("tbody", null, ...pts.map((p) => h("tr", null,
          h("td", null, p.slide),
          ...PERSONAS.map((k) => h("td", { class: "right tnum" },
            p[k] == null ? "not scored" : tableNumber(p.slide, k, p[k], definitional.has(k))))))))));

  return h("div", null, legend, wrap, note, table);
}
