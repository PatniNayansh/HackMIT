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
// Alignment to declared intent for each persona across slide order: three lines, one axis.

const W = 1000, H = 300, M = { l: 44, r: 96, t: 14, b: 30 };

export function arcChart(rollup, { onPoint, tableNumber }) {
  const pts = rollup.arc;
  const vals = pts.flatMap((p) => PERSONAS.map((k) => p[k])).filter((v) => v != null);
  if (!vals.length) return h("p", { class: "empty" }, "No slide has been scored yet.");

  const lo = Math.min(0, Math.floor(Math.min(...vals) * 2) / 2);
  const hi = Math.max(1, Math.ceil(Math.max(...vals) * 2) / 2);
  const n = pts.length;
  const X = (i) => (n === 1 ? (M.l + W - M.r) / 2 : M.l + (i / (n - 1)) * (W - M.l - M.r));
  const Y = (v) => H - M.b - ((v - lo) / (hi - lo)) * (H - M.t - M.b);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Alignment to declared intent for the novice, peer and expert across slide order" });

  const ticks = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.5) ticks.push(t);
  svg.append(s("g", { class: "grid" }, ...ticks.map((t) => s("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }))));
  svg.append(...ticks.map((t) => s("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end" }, f2(t))));
  svg.append(s("g", { class: "axis" }, s("line", { x1: M.l, x2: W - M.r, y1: Y(lo), y2: Y(lo) })));

  const every = Math.ceil(n / 15);
  pts.forEach((p, i) => {
    if (i % every === 0 || i === n - 1) svg.append(s("text", { x: X(i), y: H - 8, "text-anchor": "middle" }, p.slide));
  });

  // Lines: 2px, broken where a persona has no value for a slide.
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
    svg.append(s("text", { x: W - M.r + 12, y: e.ly + 4, style: "fill: var(--ink-2); font-size: 12.5px" }, LABEL[e.k]));
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
        "aria-label": `${LABEL[k]}, slide ${p.slide}, alignment ${f2(p[k])}. Show the text behind it.`,
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
        h("b", null, p[k] == null ? "not scored" : f2(p[k])))),
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
    ...PERSONAS.map((k) => h("span", null, h("i", { style: `border-color:${COLOR[k]}` }), LABEL[k])));

  const table = h("details", { class: "table-view" },
    h("summary", null, "Show as a table"),
    h("div", { class: "scroll" },
      h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Slide"), ...PERSONAS.map((k) => h("th", { class: "right" }, LABEL[k])))),
        h("tbody", null, ...pts.map((p) => h("tr", null,
          h("td", null, p.slide),
          ...PERSONAS.map((k) => h("td", { class: "right tnum" }, p[k] == null ? "not scored" : tableNumber(p.slide, k, p[k])))))))));

  return h("div", null, legend, wrap, table);
}
