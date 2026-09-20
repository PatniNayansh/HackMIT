import { closeDrawer } from "./provenance.js";
import { initTheme } from "./theme.js";
import { home, setup, savedRuns } from "./views-home.js";
import { overview } from "./views-overview.js";
import { detail } from "./views-detail.js";
import { audioRun } from "./views-audio.js";

const root = document.getElementById("app");
let cleanup = null;

const ROUTES = [
  [/^#\/?$/, () => home(root)],
  [/^#\/audio\/([\w-]+)$/, (id) => audioRun(root, id)],
  [/^#\/runs$/, () => savedRuns(root)],
  [/^#\/run\/([\w-]+)\/setup$/, (id) => setup(root, id)],
  [/^#\/run\/([\w-]+)\/slide\/(\d+)$/, (id, n) => detail(root, id, Number(n))],
  [/^#\/run\/([\w-]+)$/, (id) => overview(root, id)],
];

async function route() {
  closeDrawer();
  if (typeof cleanup === "function") cleanup();
  cleanup = null;
  const hash = location.hash || "#/";
  for (const [re, view] of ROUTES) {
    const m = hash.match(re);
    if (m) {
      cleanup = await view(...m.slice(1));
      break;
    }
  }
  document.querySelectorAll(".topbar nav a").forEach((a) =>
    a.toggleAttribute("aria-current", a.getAttribute("href") === "#/" && hash === "#/"));
  window.scrollTo(0, 0);
}

// Shift+R opens the saved runs from any screen. Ignored while typing, so it can never eat a
// capital R in the intent box; kept out of the visible UI on purpose (see views-home.demoDoor).
addEventListener("keydown", (e) => {
  if (!e.shiftKey || e.key !== "R" || e.ctrlKey || e.metaKey || e.altKey) return;
  const el = document.activeElement;
  if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
  e.preventDefault();
  location.hash = "#/runs";
});

initTheme();
addEventListener("hashchange", route);
route();
