// Exercises theme.js with a stand-in DOM: a localStorage that throws (private window), one that
// works, and a system preference, and prints what the page would end up with as JSON.
class Attr { constructor() { this.a = {}; } setAttribute(k, v) { this.a[k] = String(v); } getAttribute(k) { return this.a[k] ?? null; } }
function setup({ stored, storageThrows, systemDark, preset }) {
  const root = new Attr();
  if (preset) root.setAttribute("data-theme", preset);
  const button = Object.assign(new Attr(), { listeners: {}, title: "", addEventListener(ev, fn) { this.listeners[ev] = fn; }, click() { this.listeners.click(); } });
  const mq = { matches: systemDark, listeners: [], addEventListener(_, fn) { this.listeners.push(fn); } };
  const store = { ...(stored ? { "profe-theme": stored } : {}) };
  globalThis.document = { documentElement: root, getElementById: () => button };
  globalThis.window = { matchMedia: () => mq };
  globalThis.localStorage = storageThrows
    ? { getItem() { throw new Error("SecurityError"); }, setItem() { throw new Error("QuotaExceededError"); } }
    : { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
  return { root, button, mq, store };
}

const results = {};
let n = 0;
async function scenario(name, opts, act) {
  const env = setup(opts);
  const mod = await import(`../js/theme.js?${n++}`); // a fresh module (and matchMedia) per scenario
  mod.initTheme();
  const before = env.root.getAttribute("data-theme");
  const pressedBefore = env.button.getAttribute("aria-pressed");
  if (act) act(env);
  results[name] = { before, pressedBefore, after: env.root.getAttribute("data-theme"), stored: env.store["profe-theme"] ?? null, pressed: env.button.getAttribute("aria-pressed") };
}

await scenario("system dark, nothing saved", { systemDark: true });
await scenario("system light, nothing saved", { systemDark: false });
await scenario("saved choice beats the system", { stored: "light", systemDark: true });
await scenario("toggle overrides the system and persists", { systemDark: true }, (e) => e.button.click());
await scenario("private window: storage throws, page still works", { storageThrows: true, systemDark: false }, (e) => e.button.click());
await scenario("head script already set it", { preset: "dark", systemDark: false });
await scenario("follows the system until a choice is made", { systemDark: false }, (e) => { e.mq.matches = true; e.mq.listeners.forEach((f) => f()); });
await scenario("stops following once the user has chosen", { systemDark: false }, (e) => { e.button.click(); e.mq.matches = false; e.mq.listeners.forEach((f) => f()); });
process.stdout.write(JSON.stringify(results), () => process.exit(0));
