// Lecture audio, on its way to the GX10.
//
// CHUNKING. The audio is cut into fixed two-minute segments with no overlap before anything is
// sent. This is not a size limit -- it is a cost limit. TRIBE v2's text encoder builds
// *contextualised* word embeddings, so every word is re-encoded against the whole transcript
// before it, and cost grows faster than length. Feeding a 25-minute lecture in one pass was
// still in the word-embedding stage after 30 minutes with an ETA over 30 hours. Two-minute
// chunks each start their own short context, which keeps the cost bound flat and lets the run
// be resumed a chunk at a time.
//
// No overlap: each chunk starts a cold context regardless, so overlapping would pay twice for
// the same seconds and buy nothing. Not silence-based: lecture pauses are short and irregular,
// so chunk lengths -- and the cost bound that is the whole point -- become unpredictable.
//
// The cost is a real one and worth knowing about: a sentence spanning a boundary is encoded
// with no lead-in, which reads as a brief dip in the first seconds of each chunk.

export const AUDIO_CHUNK_S = 120;

// --------------------------------------------------------------------------- THE GX10 CALL
//
// STUB. The GX10 has no HTTP endpoint yet: today the neural layer reaches this app as files on
// disk, written by scripts/precompute_neural/run.py over SSH. Wiring this up needs the URL,
// the auth method, the request shape and whether it answers synchronously or hands back a job
// to poll -- none of which are decided. Everything on the page above this line is real; this
// one function is the only thing pretending, and it pretends by waiting rather than by
// inventing a result.
//
// When the endpoint exists, replace the body and nothing else.

export async function sendAudioChunk(file, index, total) {
  await new Promise((resolve) => setTimeout(resolve, 450));
  return { chunk: index, of: total, accepted: true };
}
