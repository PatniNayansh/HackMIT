// Lecture audio, on its way to the GX10.
//
// The browser's part is small on purpose. It hands the file to our own backend, which cuts it
// into two-minute chunks with ffmpeg and ships them over SSH to the GX10 -- there is no HTTP
// endpoint on that machine, and there is not meant to be one. The page then polls one status
// endpoint and draws what it says.
//
// WHY TWO MINUTES. TRIBE v2's text encoder builds contextualised word embeddings: every word is
// re-encoded against the whole transcript before it, through a 3B-parameter Llama. Cost grows
// with document length, not word count. A 25-minute lecture in one pass was still embedding
// words after 30 minutes with a 30-hour ETA. Two-minute chunks each start their own short
// context, which restores the fast rate. No overlap -- each chunk starts cold anyway, so
// overlapping pays twice for the same seconds. Not silence-based -- lecture pauses are short and
// irregular, so chunk lengths, and the cost bound that is the point, stop being predictable.
//
// The cost is visible in the output: a sentence spanning a boundary is encoded with no lead-in,
// which reads as a brief dip in the first seconds of each chunk.

export const AUDIO_CHUNK_S = 120;

const url = (runId) => `/api/runs/${encodeURIComponent(runId)}/audio`;

/** Hand the recording to our backend. It answers as soon as the file has landed; everything
 *  after that -- chunking, shipping, running -- is watched through pollAudio. */
export async function sendAudio(runId, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url(runId), { method: "POST", body: form });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function audioStatus(runId) {
  const res = await fetch(url(runId));
  if (!res.ok) throw new Error(`could not read the audio job (${res.status})`);
  return res.json();
}

export async function discardAudio(runId) {
  try { await fetch(url(runId), { method: "DELETE" }); } catch (_) { /* local cleanup only */ }
}

/** Poll until the job reaches a resting state. `onUpdate` is called with every status, so the
 *  page can draw progress; the returned function stops the polling. */
export function pollAudio(runId, onUpdate, { intervalMs = 2500 } = {}) {
  let stopped = false;
  const done = new Set(["complete", "failed", "unavailable", "none"]);
  (async () => {
    while (!stopped) {
      let s;
      try {
        s = await audioStatus(runId);
      } catch (e) {
        // A poll that fails is not a job that failed: the work is on another machine and keeps
        // going. Report it and try again rather than declaring the run dead.
        onUpdate({ state: "running", detail: e.message, transient: true });
        await new Promise((r) => setTimeout(r, intervalMs * 2));
        continue;
      }
      onUpdate(s);
      if (done.has(s.state)) return;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  })();
  return () => { stopped = true; };
}
