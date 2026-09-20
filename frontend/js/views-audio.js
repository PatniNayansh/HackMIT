// The audio run: one real recorded lecture read by the model, on its own page rather than
// squeezed into a panel on the front page. It is a run like any other -- it just took audio
// instead of a deck, so it has a timecourse where a deck run has slides.
import { h, mount } from "./dom.js";
import { getJSON } from "./api.js";
import { lectureChart } from "./charts.js";
import { slideImage } from "./dom.js";

export const audioHref = (id) => `#/audio/${id}`;

export function audioRun(root, id) {
  document.title = "Lecture audio — ProFe";
  mount(root, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading the audio run…"));

  getJSON(`/api/audio/${encodeURIComponent(id)}`).then((d) => {
    const mins = Math.round(d.duration_s / 60);
    mount(root,
      h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Start"), "›", "Lecture audio"),
      h("div", { class: "page-head" }, h("div", null,
        h("h1", null, d.title),
        h("p", { class: "sub" }, d.course))),
      h("div", { class: "meta-line" },
        h("span", { class: "pill" }, "audio run"),
        h("span", null, d.venue),
        h("span", null, `${mins} min`),
        h("span", null, `${d.chunks_run} of ${d.chunks_total} chunks`),
        h("span", { class: "mono" }, "TRIBE v2")),

      h("section", { class: "section" },
        h("header", null, h("h2", null, "Language-region drive over the lecture")),
        h("p", { class: "lede" }, "Recorded audio, straight through the model. No slides, no synthesis."),
        h("div", { class: "card" },
          lectureChart(d),
          h("p", { class: "caveat" },
            "Smoothed over ", String(d.smoothing_window_s),
            " seconds. Rules mark the chunk boundaries, where the model starts a fresh context."))),

      d.surfaces && d.surfaces.length && h("section", { class: "section" },
        h("header", null, h("h2", null, "Cortex, chunk by chunk")),
        h("p", { class: "lede" }, "Left lateral cortex per chunk. Each panel has its own scale."),
        h("div", { class: "card" },
          h("div", { class: "lecture-surfaces" },
            ...d.surfaces.map((v) => h("figure", null,
              slideImage(v.url, `Cortical response, minutes ${v.from_min} to ${v.to_min}`),
              h("figcaption", { class: "label-xs" }, `${v.from_min}–${v.to_min} min`)))))),

      h("section", { class: "section" },
        h("header", null, h("h2", null, "How this run was made")),
        h("div", { class: "card" },
          h("div", { class: "lecture-facts" },
            h("div", { class: "fact" }, h("div", { class: "k" }, "Audio read"), h("div", { class: "v" }, `${mins} min`)),
            h("div", { class: "fact" }, h("div", { class: "k" }, "1-second segments"), h("div", { class: "v" }, String(d.n_segments))),
            h("div", { class: "fact" }, h("div", { class: "k" }, "Chunks run"), h("div", { class: "v" }, `${d.chunks_run} of ${d.chunks_total}`)),
            h("div", { class: "fact" }, h("div", { class: "k" }, "Smoothing"), h("div", { class: "v" }, `${d.smoothing_window_s} s`))))));
  }).catch((e) => mount(root,
    h("p", { class: "err" }, e.message),
    h("a", { href: "#/" }, "Back to start")));

  return () => {};
}
