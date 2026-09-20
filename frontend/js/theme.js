// Light/dark theme. The choice is applied to <html data-theme> by a small blocking script in
// index.html's <head> BEFORE first paint (so there is no flash of the wrong theme); this module
// only wires up the header toggle and follows the system setting until the user has chosen.
//
// localStorage throws in private windows and when site data is blocked, so every access is wrapped:
// a failed read means "no saved choice" and a failed write means the choice lasts for this page only.
const KEY = "sightline-theme";
const root = document.documentElement;
const query = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

function stored() {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch (_) {
    return null;
  }
}

function save(theme) {
  try {
    localStorage.setItem(KEY, theme);
  } catch (_) { /* nowhere to keep it; the page still works */ }
}

const system = () => (query && query.matches ? "dark" : "light");
const current = () => (root.getAttribute("data-theme") === "dark" ? "dark" : "light");

function apply(theme, button) {
  root.setAttribute("data-theme", theme);
  if (button) {
    button.setAttribute("aria-pressed", String(theme === "dark"));
    button.setAttribute("aria-label", theme === "dark" ? "Dark theme. Switch to light." : "Light theme. Switch to dark.");
    button.title = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  }
}

export function initTheme() {
  const button = document.getElementById("theme-toggle");
  apply(root.getAttribute("data-theme") || stored() || system(), button);
  button?.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    apply(next, button);
    save(next);
  });
  // Until the user picks a theme, follow the system setting as it changes.
  const follow = () => { if (!stored()) apply(system(), button); };
  if (query?.addEventListener) query.addEventListener("change", follow);
  else if (query?.addListener) query.addListener(follow);
}
