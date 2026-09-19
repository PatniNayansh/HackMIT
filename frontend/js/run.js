// One live view of a run. The server writes each slide's result to disk as it lands; this polls
// for whatever is new and hands listeners the merged state. Saved runs go through the same path
// and simply stop polling after the first response.
import { getJSON } from "./api.js";

const POLL_MS = 1000;
let current = null;

function signature(state) {
  const m = state.meta;
  if (!m) return null;
  return `${m.status}|${m.error}|${state.results.size}|${m.finished_at}`;
}

// A view that throws while painting is a bug in the view, not a failed request: log it loudly
// instead of letting it masquerade as a network error.
function notify(run) {
  for (const fn of run.listeners) {
    try {
      fn(run.state);
    } catch (e) {
      console.error("view failed to render", e);
    }
  }
}

async function poll(run) {
  if (run !== current || run.polling) return;
  run.polling = true;
  run.timer = null;
  try {
    const since = Math.max(0, ...run.state.results.keys());
    const body = await getJSON(`/api/runs/${encodeURIComponent(run.id)}?since=${since}`);
    if (run !== current) return;
    const before = run.state ? signature(run.state) : null;
    for (const r of body.results) run.state.results.set(r.index, r);
    Object.assign(run.state, {
      meta: body.meta, rollup: body.rollup, pending: body.pending, imageUrls: body.image_urls, loaded: true, error: null,
    });
    if (before !== signature(run.state) || !run.notified) {
      run.notified = true;
      notify(run);
    }
  } catch (e) {
    if (run !== current) return;
    run.state.error = e.message;
    notify(run);
  }
  run.polling = false;
  if (run === current && run.listeners.size && (run.state.error || run.state.meta?.status === "running")) {
    run.timer = setTimeout(() => poll(run), run.state.error ? 2500 : POLL_MS);
  }
}

/** Watch a run. `listener(state)` fires on the first load and whenever a slide lands or the
 *  status changes. Returns an unsubscribe function. Pass fresh=true after (re)starting a run. */
export function watch(runId, listener, { fresh = false } = {}) {
  if (!current || current.id !== runId || fresh) {
    if (current) clearTimeout(current.timer);
    current = {
      id: runId, listeners: new Set(), notified: false, timer: null, polling: false,
      state: { meta: null, results: new Map(), rollup: null, pending: [], imageUrls: [], loaded: false, error: null },
    };
    poll(current);
  } else if (current.state.loaded) {
    // Already have data (e.g. moving from the overview to a slide): paint at once, keep polling.
    queueMicrotask(() => { try { listener(current.state); } catch (e) { console.error("view failed to render", e); } });
    if (current.state.meta.status === "running") poll(current); // no-op if already polling
  }
  const run = current;
  run.listeners.add(listener);
  return () => {
    run.listeners.delete(listener);
    if (run.listeners.size === 0) {
      clearTimeout(run.timer);
      run.timer = null;
    }
  };
}

export function currentState() {
  return current?.state ?? null;
}

/** Drop the cached view of a run (after it is started or restarted) so the next watch is fresh. */
export function forget(runId) {
  if (current?.id === runId) {
    clearTimeout(current.timer);
    current = null;
  }
}
