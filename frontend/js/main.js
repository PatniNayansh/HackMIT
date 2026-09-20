import { closeDrawer } from "./provenance.js";
import { initTheme } from "./theme.js";
import { home, setup } from "./views-home.js";
import { overview } from "./views-overview.js";
import { detail } from "./views-detail.js";
import { audioRun } from "./views-audio.js";

const root = document.getElementById("app");
let cleanup = null;

const ROUTES = [
  [/^#\/?$/, () => home(root)],
  [/^#\/audio$/, () => audioRun(root)],
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

initTheme();
addEventListener("hashchange", route);
route();
