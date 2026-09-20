import { h, mount, when, warnIcon, infoIcon, slideImage } from "./dom.js";
import { api, getJSON, postJSON } from "./api.js";
import { forget } from "./run.js";
import { AUDIO_RUN_HREF } from "./views-audio.js";
import { AUDIO_CHUNK_S, sendAudioChunk } from "./audio-upload.js";

function statusPill(status) {
  if (status === "complete") return h("span", { class: "pill" }, "complete");
  if (status === "running") return h("span", { class: "pill outline" }, h("span", { class: "spinner" }), "running");
  return h("span", { class: "pill bad" }, status);
}

// ------------------------------------------------------------------------ home

// Upload is a small state machine, because every step of it is something the person waits on
// and therefore something they have to be able to watch.
//
// The steps are the requests that actually happen, not a prettier list of them. /api/upload
// parses the slides, describes the figures and infers the subfield in ONE call, so it is one
// step: splitting it into three would mean animating progress the client cannot see. Chunking
// is genuinely per-chunk, so that one counts.
const STEPS = [
  ["deck", "Reading the deck", "Slides, figures and the subfield"],
  ["audio", "Chunking audio", "Two-minute segments, sent one at a time"],
  ["review", "Reading it three ways", "Novice, peer and expert, one slide at a time"],
];

/** `states` maps each step key to done | now | queued | skipped. `detail` replaces the sub-line
 *  of whichever step is running. Steps keep their height throughout, so the panel never
 *  reflows as it advances: only the mark and the sub-line change. */
function progressPanel(states, detail = "") {
  return h("div", { class: "progress-panel", role: "status", "aria-live": "polite" },
    ...STEPS.map(([key, title, sub]) => {
      const state = states[key] || "queued";
      return h("div", { class: `pstep ${state}` },
        h("span", { class: "pstep-mark", "aria-hidden": "true" },
          state === "now" ? h("span", { class: "spinner" }) : state === "done" ? "\u2713" : ""),
        h("div", null,
          h("div", { class: "pstep-title" }, title),
          h("div", { class: "pstep-sub" },
            state === "skipped" ? "No audio provided"
              : state === "now" && detail ? detail
              : sub)));
    }));
}

const fmtDuration = (s) => {
  if (!isFinite(s) || s <= 0) return "";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
};

/** Audio duration, read in the browser so the person sees it before anything is sent. */
function readDuration(file) {
  return new Promise((resolve) => {
    const el = document.createElement("audio");
    if (!el || !URL.createObjectURL) return resolve(0);
    const url = URL.createObjectURL(file);
    el.preload = "metadata";
    el.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(el.duration || 0); };
    el.onerror = () => { URL.revokeObjectURL(url); resolve(0); };
    el.src = url;
  });
}


// -------------------------------------------------------------------- history, on request
// Not linked from anywhere on purpose. The front page is an entry point, not an index, but a
// presentation needs to reach the runs that are already on disk without uploading anything.
// Two ways in, because a live demo should not depend on hitting a small target: the invisible
// button in the corner below, and Shift+R from any screen (see main.js).

export const HISTORY_HREF = "#/runs";

/** A real, focusable button that happens to be invisible until you hover or tab to it. Not
 *  display:none and not aria-hidden: it stays operable by keyboard, which is what makes it a
 *  reliable way in rather than a trick that might not work on the night. */
export function demoDoor() {
  return h("a", {
    class: "demo-door", href: HISTORY_HREF, title: "History (Shift+R)",
    "aria-label": "Open history",
  }, "\u00b7");
}

export function savedRuns(root) {
  document.title = "History \u2014 ProFe";
  const body = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading\u2026"));
  mount(root,
    h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Start"), "\u203a", "History"),
    h("div", { class: "page-head" }, h("div", null,
      h("h1", null, "History"),
      h("p", { class: "sub" }, "Every run on disk. Opening one makes no model calls."))),
    h("section", { class: "card" }, body));

  getJSON("/api/runs").then((runs) => {
    if (!runs.length) {
      body.replaceChildren(h("p", { class: "empty" }, "No saved runs yet."));
      return;
    }
    body.replaceChildren(h("table", { class: "rows" },
      h("thead", null, h("tr", null, h("th", null, "Deck"), h("th", null, "Date"),
        h("th", { class: "right" }, "Slides"), h("th", null, "Model"), h("th", null, "Status"))),
      h("tbody", null, ...runs.map((r) => h("tr", {
        tabindex: 0, on: {
          click: () => { location.hash = `#/run/${r.run_id}`; },
          keydown: (e) => { if (e.key === "Enter") location.hash = `#/run/${r.run_id}`; },
        },
      },
        h("td", null, h("strong", null, r.title), r.sample && h("span", null, " ", h("span", { class: "pill sample" }, "sample data"))),
        h("td", { class: "muted" }, when(r.created_at)),
        h("td", { class: "right tnum" }, r.slide_count),
        h("td", { class: "mono" }, r.model || "\u2014"),
        h("td", null, statusPill(r.status)))))));
  }).catch((e) => body.replaceChildren(h("p", { class: "err" }, e.message)));

  return () => {};
}


/** Everything that has been run, deck or audio, newest first. The audio run is not in
 *  /api/runs -- it took a recording rather than a PDF and has no slides to stream -- so it is
 *  fetched alongside and folded in, because from here it is simply another thing we ran. */
function historySection() {
  const body = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading\u2026"));

  const row = ({ href, title, kind, when: date, slides, model, status }) =>
    h("a", { class: "hrow", href },
      h("div", { class: "hrow-main" },
        h("div", { class: "hrow-title" }, title),
        h("div", { class: "hrow-meta" }, kind, date && [" \u00b7 ", date], slides && [" \u00b7 ", slides])),
      h("div", { class: "hrow-right" },
        model && h("span", { class: "mono small muted" }, model),
        status));

  Promise.allSettled([getJSON("/api/runs"), getJSON("/api/lecture")]).then(([runsR, lectureR]) => {
    const rows = [];
    if (lectureR.status === "fulfilled") {
      const d = lectureR.value;
      rows.push(row({
        href: AUDIO_RUN_HREF, title: d.title, kind: "Lecture audio",
        when: `${d.venue}`, slides: `${Math.round(d.duration_s / 60)} min`,
        model: "TRIBE v2", status: h("span", { class: "pill" }, "complete"),
      }));
    }
    if (runsR.status === "fulfilled") {
      for (const r of runsR.value) {
        rows.push(row({
          href: `#/run/${r.run_id}`, title: r.title,
          kind: r.sample ? "Slides \u00b7 sample" : "Slides",
          when: when(r.created_at), slides: `${r.slide_count} slides`,
          model: r.model || null, status: statusPill(r.status),
        }));
      }
    }
    body.replaceChildren(...(rows.length ? rows : [h("p", { class: "empty" }, "Nothing has been run yet.")]));
  });

  return h("section", { class: "section history" },
    h("header", null, h("h2", null, "History")),
    h("p", { class: "lede" }, "Everything run so far. Opening one makes no model calls."),
    h("div", { class: "card" }, body));
}

export function home(root) {
  document.title = "ProFe";
  const banner = h("div");
  const uploadErr = h("p", { class: "err", role: "alert" });

  const slideInput = h("input", { type: "file", accept: "application/pdf,.pdf" });
  const audioInput = h("input", { type: "file", accept: "audio/*,.mp3,.m4a,.wav,.aac,.ogg,.flac" });
  const drop = h("div", { class: "drop", tabindex: 0, role: "button", "aria-label": "Choose a PDF to review" });
  const audioRow = h("div", { class: "audio-row" });
  const stage = h("div", { class: "upload-stage" });
  const go = h("button", { class: "btn primary", disabled: true }, "Proceed");

  // Nothing starts on its own any more: the deck is held here until Proceed is pressed.
  let deck = null;      // { file, run_id, slide_count }
  let audio = null;     // { file, duration_s, chunks, state }
  let busy = false;
  let busyAudio = false;

  function audioLine() {
    if (!audio) {
      return audioRow.replaceChildren(
        h("button", { class: "btn small", on: { click: () => audioInput.click() } }, "Add lecture audio"),
        h("span", { class: "hint" }, "Optional. The deck is analysed with or without it."),
        audioInput);
    }
    const d = fmtDuration(audio.duration_s);
    audioRow.replaceChildren(
      h("span", { class: "file-chip" },
        h("span", { class: "file-name" }, audio.file.name),
        d && h("span", { class: "muted" }, " \u00b7 ", d),
        audio.chunks ? h("span", { class: "muted" }, ` \u00b7 ${audio.chunks} chunks`) : null),
      h("button", {
        class: "btn small", disabled: busy,
        on: { click: () => { audio = null; audioInput.value = ""; paint(); } },
      }, "Remove"),
      audioInput);
  }

  function paint() {
    audioLine();
    go.disabled = busy || !deck;
    go.replaceChildren(busy
      ? h("span", null, h("span", { class: "spinner" }), " Working\u2026")
      : h("span", null, deck ? "Proceed" : "Add a deck to continue", deck ? " \u2192" : ""));
  }

  const idle = () => {
    drop.classList.remove("has-file");
    drop.replaceChildren(
      h("strong", null, "Drop a PDF here, or click to choose one"),
      h("span", { class: "hint" }, "Slides are required. About 5 seconds a slide."),
      slideInput);
    stage.replaceChildren();
    paint();
  };

  const deckChosen = () => {
    drop.classList.add("has-file");
    drop.replaceChildren(
      h("span", { class: "file-chip" }, h("span", { class: "file-name" }, deck.file.name),
        h("span", { class: "muted" }, ` \u00b7 ${deck.slide_count} slides`)),
      h("button", {
        class: "btn small", on: { click: (e) => { e.stopPropagation(); deck = null; slideInput.value = ""; idle(); } },
      }, "Choose another"),
      slideInput);
    stage.replaceChildren();
    paint();
  };

  // ---- the deck: uploaded as soon as it is chosen, but the run does NOT start ----------
  async function takeDeck(file) {
    uploadErr.textContent = "";
    if (!file) return;
    busy = true; paint();
    drop.replaceChildren(h("strong", null, h("span", { class: "spinner" }), ` Reading ${file.name}\u2026`));
    drop.style.pointerEvents = "none";
    const audioState = () => (!audio ? "skipped" : audio.chunks && !busyAudio ? "done" : "queued");
    stage.replaceChildren(progressPanel({ deck: "now", audio: audioState(), review: "queued" }, `Reading ${file.name}`));
    try {
      const form = new FormData();
      form.append("file", file);
      const run = await api("/api/upload", { method: "POST", body: form });
      forget(run.run_id);
      deck = { file, run_id: run.run_id, slide_count: run.slide_count ?? run.meta?.slide_count ?? 0 };
      busy = false;
      drop.style.pointerEvents = "";
      deckChosen();
    } catch (e) {
      busy = false;
      uploadErr.textContent = e.message;
      drop.style.pointerEvents = "";
      idle();
    }
  }

  // ---- the audio: measured and chunked in the browser, then handed to the GX10 ---------
  async function takeAudio(file) {
    uploadErr.textContent = "";
    if (!file) return;
    busy = true;
    audio = { file, duration_s: 0, chunks: 0 };
    paint();
    busyAudio = true;
    const deckState = () => (deck ? "done" : "queued");
    stage.replaceChildren(progressPanel({ deck: deckState(), audio: "now", review: "queued" }, `Measuring ${file.name}\u2026`));
    audio.duration_s = await readDuration(file);
    audio.chunks = Math.max(1, Math.ceil(audio.duration_s / AUDIO_CHUNK_S));
    for (let i = 1; i <= audio.chunks; i++) {
      stage.replaceChildren(progressPanel({ deck: deckState(), audio: "now", review: "queued" }, `Chunk ${i} of ${audio.chunks}`));
      await sendAudioChunk(file, i, audio.chunks);
    }
    busyAudio = false;
    busy = false;
    stage.replaceChildren();
    paint();
  }

  const chooseDeck = () => slideInput.click();
  drop.addEventListener("click", (e) => { if (e.target !== slideInput && !deck && !busy) chooseDeck(); });
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !deck && !busy) { e.preventDefault(); chooseDeck(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); if (!busy) drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (!busy) takeDeck(e.dataTransfer.files[0]); });
  slideInput.addEventListener("change", () => takeDeck(slideInput.files[0]));
  audioInput.addEventListener("change", () => takeAudio(audioInput.files[0]));

  go.addEventListener("click", () => {
    if (!deck || busy) return;
    stage.replaceChildren(progressPanel(
      { deck: "done", audio: audio ? "done" : "skipped", review: "now" }, "Setting up the three audiences"));
    location.hash = `#/run/${deck.run_id}/setup`;
  });

  idle();

  root.replaceChildren(
    h("p", { class: "home-lede" }, "See how a novice, a peer and an expert would each read your slides, and where they part ways."),
    banner,
    h("section", { class: "card upload-card" },
      drop, audioRow, stage, uploadErr,
      h("div", { class: "go-row" }, go)),
    historySection(),
    demoDoor());

  getJSON("/api/health").then((hl) => {
    if (!hl.can_call_model) {
      banner.className = "banner";
      banner.style.marginBottom = "16px";
      banner.append(infoIcon(), h("div", null,
        h("strong", null, "No API key found. "),
        "Starting a new review needs OPENAI_API_KEY in the repo-root .env (see .env.example)."));
    }
  }).catch(() => {});

  return () => {};
}

// ----------------------------------------------------------------------- setup

export async function setup(root, runId) {
  document.title = "Set up review — ProFe";
  root.replaceChildren(h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading…"));
  let run;
  try {
    run = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (e) {
    root.replaceChildren(h("p", { class: "err" }, e.message), h("a", { href: "#/" }, "Back to start"));
    return;
  }
  const { meta } = run;
  if (!["draft", "failed", "interrupted"].includes(meta.status)) { location.hash = `#/run/${runId}`; return; }

  const inferred = meta.profile_inferred;
  const prior = meta.profile;
  const start = {
    intent: meta.intent || "",
    domain: (prior || inferred)?.domain || "",
    adjacent: (prior || inferred)?.adjacent_field || "",
  };
  const intent = h("textarea", { id: "intent", placeholder: "e.g. Our serving system delivers 2.4x higher goodput at p99 latency.", value: start.intent, required: true });
  const domain = h("input", { type: "text", id: "domain", value: start.domain, placeholder: "e.g. LLM inference serving systems" });
  const adjacent = h("input", { type: "text", id: "adjacent", value: start.adjacent, placeholder: "e.g. distributed systems and databases" });
  const go = h("button", { class: "btn primary", type: "button" }, "Start review");
  const err = h("p", { class: "err", role: "alert" });
  const badge = h("span", { class: "badge" });

  const edited = () => !inferred || domain.value.trim() !== inferred.domain || adjacent.value.trim() !== inferred.adjacent_field;
  function refresh() {
    const ready = intent.value.trim() && domain.value.trim() && adjacent.value.trim();
    go.disabled = !ready;
    if (!inferred) badge.replaceChildren();
    else if (edited()) badge.replaceChildren(h("span", { class: "pill" }, "edited by you"));
    else badge.replaceChildren(h("span", { class: "pill unconfirmed" }, "unconfirmed"), h("span", { class: "hint" }, "suggested by the model; check it before you start"));
  }
  [intent, domain, adjacent].forEach((el) => el.addEventListener("input", refresh));

  go.addEventListener("click", async () => {
    err.textContent = "";
    go.disabled = true;
    go.replaceChildren(h("span", { class: "spinner" }), " Starting…");
    try {
      await postJSON(`/api/runs/${encodeURIComponent(runId)}/start`, {
        intent: intent.value, domain: domain.value, adjacent_field: adjacent.value,
      });
      forget(runId);
      location.hash = `#/run/${runId}`;
    } catch (e) {
      err.textContent = e.message;
      go.textContent = "Start review";
      refresh();
    }
  });

  mount(root,
    h("div", { class: "crumbs" }, h("a", { href: "#/" }, "Start"), "›", meta.title),
    h("div", { class: "page-head" }, h("div", null, h("h1", null, meta.title), h("p", { class: "sub" }, `${meta.slide_count} slides read from ${meta.source_filename}`))),
    meta.status !== "draft" && meta.error && h("div", { class: "banner bad", style: "margin-bottom:16px" }, warnIcon(), h("div", null, h("strong", null, "The last attempt did not finish. "), meta.error)),
    h("section", { class: "card stack" },
      h("div", { class: "thumbs-strip" }, ...run.image_urls.map((u, i) => slideImage(u, `Slide ${i + 1}`))),
      h("div", { class: "form-grid" },
        h("label", { class: "field" },
          h("span", { class: "label" }, "Declared intent", h("span", { class: "pill" }, "required")),
          intent,
          h("span", { class: "hint" }, "What is this deck supposed to land? The audiences never see it.")),
        h("label", { class: "field" },
          h("span", { class: "label" }, "Deck subfield ", badge),
          domain,
          h("span", { class: "hint" }, "The exact subfield the expert works in.")),
        meta.profile_inference_error && h("p", { class: "banner" }, infoIcon(), meta.profile_inference_error),
        h("label", { class: "field" },
          h("span", { class: "label" }, "Adjacent field"),
          adjacent,
          h("span", { class: "hint" }, "Technical, but has never worked in the subfield.")),
        h("div", null, go), err)));
  refresh();
  intent.focus();
}
