import { h, mount, when, warnIcon, infoIcon } from "./dom.js";
import { api, getJSON, postJSON } from "./api.js";
import { forget } from "./run.js";

function statusPill(status) {
  if (status === "complete") return h("span", { class: "pill" }, "complete");
  if (status === "running") return h("span", { class: "pill outline" }, h("span", { class: "spinner" }), "running");
  return h("span", { class: "pill bad" }, status);
}

// ------------------------------------------------------------------------ home

export function home(root) {
  document.title = "Sightline";
  const banner = h("div");
  const drop = h("div", { class: "drop", tabindex: 0, role: "button", "aria-label": "Choose a PDF to review" });
  const fileInput = h("input", { type: "file", accept: "application/pdf,.pdf" });
  const uploadErr = h("p", { class: "err", role: "alert" });
  const historyBody = h("div", null, h("p", { class: "empty" }, h("span", { class: "spinner" }), " Loading saved runs…"));

  const idle = () => drop.replaceChildren(
    h("strong", null, "Drop a PDF here, or click to choose one"),
    h("span", { class: "hint" }, "PDF only for now. Each slide is read three times, one slide after another, about 5 seconds a slide."),
    fileInput);

  async function upload(file) {
    uploadErr.textContent = "";
    if (!file) return;
    drop.replaceChildren(h("strong", null, h("span", { class: "spinner" }), ` Reading ${file.name}…`),
      h("span", { class: "hint" }, "Extracting the slides and suggesting the deck’s subfield."));
    drop.style.pointerEvents = "none";
    try {
      const form = new FormData();
      form.append("file", file);
      const run = await api("/api/upload", { method: "POST", body: form });
      forget(run.run_id);
      location.hash = `#/run/${run.run_id}/setup`;
    } catch (e) {
      uploadErr.textContent = e.message;
      drop.style.pointerEvents = "";
      idle();
    }
  }
  const choose = () => fileInput.click();
  drop.addEventListener("click", (e) => { if (e.target !== fileInput) choose(); });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files[0]); });
  fileInput.addEventListener("change", () => upload(fileInput.files[0]));
  idle();

  root.replaceChildren(
    h("div", { class: "page-head" }, h("div", null,
      h("h1", null, "Review a presentation"),
      h("p", { class: "sub" }, "See how a novice, a peer and an expert would each read your slides, and where they part ways."))),
    banner,
    h("div", { class: "home-grid" },
      h("section", { class: "card" }, h("h2", { style: "margin-bottom:12px" }, "New review"), drop, uploadErr),
      h("section", { class: "card" }, h("h2", { style: "margin-bottom:8px" }, "Saved runs"), historyBody)));

  getJSON("/api/health").then((hl) => {
    if (!hl.can_call_model) {
      banner.className = "banner";
      banner.style.marginBottom = "16px";
      banner.append(infoIcon(), h("div", null,
        h("strong", null, "No API key found. "),
        "Saved runs still open and replay offline. Starting a new review needs ANTHROPIC_API_KEY in the repo-root .env (see .env.example)."));
    }
  }).catch(() => {});

  getJSON("/api/runs").then((runs) => {
    if (!runs.length) {
      historyBody.replaceChildren(h("p", { class: "empty" }, "No saved runs yet. Upload a deck and its review is saved here automatically."));
      return;
    }
    historyBody.replaceChildren(h("table", { class: "rows" },
      h("thead", null, h("tr", null, h("th", null, "Deck"), h("th", null, "Date"), h("th", { class: "right" }, "Slides"), h("th", null, "Model"), h("th", null, "Status"))),
      h("tbody", null, ...runs.map((r) => h("tr", {
        tabindex: 0, on: {
          click: () => { location.hash = `#/run/${r.run_id}`; },
          keydown: (e) => { if (e.key === "Enter") location.hash = `#/run/${r.run_id}`; },
        },
      },
        h("td", null, h("strong", null, r.title), r.sample && h("span", null, " ", h("span", { class: "pill sample" }, "sample data"))),
        h("td", { class: "muted" }, when(r.created_at)),
        h("td", { class: "right tnum" }, r.slide_count),
        h("td", { class: "mono" }, r.model || "—"),
        h("td", null, statusPill(r.status)))))));
  }).catch((e) => historyBody.replaceChildren(h("p", { class: "err" }, e.message)));

  return () => {};
}

// ----------------------------------------------------------------------- setup

export async function setup(root, runId) {
  document.title = "Set up review — Sightline";
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
      h("div", { class: "thumbs-strip" }, ...run.image_urls.map((u, i) => h("img", { src: u, alt: `Slide ${i + 1}`, loading: "lazy" }))),
      h("div", { class: "form-grid" },
        h("label", { class: "field" },
          h("span", { class: "label" }, "Declared intent", h("span", { class: "pill" }, "required")),
          intent,
          h("span", { class: "hint" }, "One sentence: what is this deck supposed to land? Alignment is measured against it, and the audiences never see it.")),
        h("label", { class: "field" },
          h("span", { class: "label" }, "Deck subfield ", badge),
          domain,
          h("span", { class: "hint" }, "The exact subfield the expert works in.")),
        meta.profile_inference_error && h("p", { class: "banner" }, infoIcon(), meta.profile_inference_error),
        h("label", { class: "field" },
          h("span", { class: "label" }, "Adjacent field"),
          adjacent,
          h("span", { class: "hint" }, "Where the peer comes from: technical, but has never worked in the subfield. Set once here so all three audiences share one definition.")),
        h("div", null, go), err)));
  refresh();
  intent.focus();
}
