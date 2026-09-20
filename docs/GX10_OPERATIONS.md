# GX10 operations reference

Everything learned hands-on while actually running TRIBE v2 on the GX10 — access,
environment quirks, and how to run/monitor/kill jobs. `GX10_SETUP.md` is the "set this up
from scratch" walkthrough; this is the "here's what actually breaks and why" reference for
anyone (or any agent) operating it afterward.

## Hardware and OS

- Model: **NVIDIA GB10** ("DGX Spark" class) — Blackwell architecture, **aarch64/ARM64**,
  not x86_64. This matters: most prebuilt CUDA wheels on PyPI only target x86_64.
- OS: Ubuntu 24.04.4 LTS.
- Driver 580.159.03, reports CUDA 13.0 capability.
- 124GB RAM, plenty of headroom — none of this workload has come close to using it.
- Hostname: `gx10-f575`.

## Access

- IP: `10.189.105.191`, user: `asus`.
- SSH key-based auth is already set up: this machine's `~/.ssh/id_rsa.pub` is in the GX10's
  `~/.ssh/authorized_keys`. Plain `ssh asus@10.189.105.191 "<cmd>"` works with no password.
- If key auth ever breaks (new machine, key rotated), password auth is available. It was
  used once, live, to bootstrap the key (a short-lived Python/paramiko script appended the
  pubkey to `authorized_keys`, then deleted itself).
- `sudo` on the GX10 needs its own password prompt (same value as SSH, at least so far), and
  **non-interactive SSH commands don't get a real TTY**, so plain `sudo` won't prompt
  correctly — use `sudo -S` and pipe the password in instead.
- Credentials (`GX10_HOST`, `GX10_SSH_USER`, `GX10_SSH_PASSWORD`) live in the repo-root
  `.env` (git-ignored — never commit it, never put the actual value in a tracked file like
  this one). To power off:
  ```bash
  source .env  # or read the values another way; don't hardcode them into a command
  ssh $GX10_SSH_USER@$GX10_HOST "echo '$GX10_SSH_PASSWORD' | sudo -S shutdown -h now"
  ```
- Before shutting down: confirm nothing is running (`pgrep -af 'run\.py|run_lecture_audio|whisperx|uvx'`)
  and flush disk state (`sync`). Then the shutdown is safe; the SSH connection dropping
  ("Connection closed by remote host") is the expected signal it worked. Verify with a
  `ConnectTimeout=5` ssh attempt — "Connection timed out" confirms it's actually off.
- **Another Claude Code session may be using this same machine/repo concurrently.** Check
  `who` on the GX10 and `ListAgents` locally before assuming exclusive access — a locked
  file, an unexpected process, or a renamed directory may be someone else's in-progress
  work, not a bug. Investigate before deleting or overwriting.

## Repo layout on the GX10

- Cloned at `~/SightLine` (the GitHub repo and this checkout are still named
  `SightLine`/`HackMIT` even after the project was renamed to **ProFe** — only the local
  Windows checkout and some in-repo Python package names have been renamed so far; don't
  assume the GX10 path has caught up without checking).
- The GitHub repo (`PatniNayansh/HackMIT`) is **private** — an anonymous `git clone` over
  HTTPS fails with no credentials configured on the GX10. Fastest way to get local changes
  onto the GX10 without setting up GitHub auth there: stream a `git archive` of the current
  branch straight over SSH:
  ```bash
  git archive --format=tar HEAD | ssh asus@10.189.105.191 "mkdir -p ~/SightLine && tar -x -C ~/SightLine"
  ```
  This only sends the committed tree (respects `.gitignore`-like exclusion via git's own
  tracked-files list) — untracked local files won't come along; `scp` those individually.
- The neural precompute venv lives at `backend/scripts/precompute_neural/.venv` — isolated
  from anything else, per `requirements.txt`'s own warning comment.

## Environment gotchas (each cost real time to find — don't rediscover them)

1. **Blackwell needs newer torch than the repo pins.** `requirements.txt` pins
   `torch>=2.5.1,<2.7` (matching TRIBE v2's own `pyproject.toml`), but PyTorch only added
   Blackwell (`sm_100`) GPU support in **2.7+**, built against CUDA 12.8, with aarch64/SBSA
   wheels on the `cu128` index. Installing the pinned range on this machine silently gives
   a CPU-only build (no aarch64 wheel exists on the `cu124` index, so pip falls back to a
   plain PyPI CPU wheel without erroring). Fix: install `torch==2.7.1` / `torchvision==0.22.1`
   from `--index-url https://download.pytorch.org/whl/cu128` instead, then `pip install -e
   tribev2 --no-deps` (skip TRIBE v2's own dependency resolution so it doesn't downgrade
   torch back down) and manually install its other listed deps at their pinned versions.
   Verify with `python -c "import torch; print(torch.cuda.is_available())"` before trusting
   any run — a silent CPU fallback looks like success until inference is impossibly slow.

2. **Non-interactive SSH shells don't source `~/.bashrc`.** Ubuntu's default `.bashrc`
   starts with a guard that returns immediately for non-interactive shells, so anything it
   would normally set up (like `uv`/`uvx`'s PATH entry at `~/.local/bin`) is silently
   missing in `ssh host "some command"` invocations, even though the identical command
   works fine when typed interactively. Always `export PATH=$HOME/.local/bin:$PATH`
   explicitly at the start of any non-interactive remote command that needs `uv`/`uvx`, and
   pass it through explicitly to any `nohup`'d background process too
   (`nohup env PATH=$HOME/.local/bin:$PATH python ...`), since the child inherits whatever
   `os.environ` looked like at launch — not a re-sourced shell.

3. **No CUDA `ctranslate2` build for aarch64.** TRIBE v2 shells out to `uvx whisperx` for
   transcription, which uses `ctranslate2` as its backend. There's no aarch64 CUDA wheel for
   it on PyPI, so `--device cuda` fails with `ValueError: This CTranslate2 package was not
   compiled with CUDA support`. This is hardcoded inside `tribev2/tribev2/eventstransforms.py`
   (`device = "cuda" if torch.cuda.is_available() else "cpu"`, `compute_type = "float16"`),
   not something exposed as a config option — the fix was patching that vendored file
   directly (after `git clone`ing tribev2 fresh) to force `device = "cpu"` and
   `compute_type = "int8"` for the whisperx call specifically. TRIBE's own model inference
   still runs on GPU; only this one transcription subprocess is CPU-bound.

4. **No `ffmpeg`, no passwordless `sudo` to install it.** `whisperx`/`ctranslate2` need
   `ffmpeg` to decode audio. Rather than requiring a sudo password for `apt-get install`,
   `imageio_ffmpeg` (already a dependency, bundles a static ffmpeg binary for the current
   platform) was symlinked into PATH:
   `ln -sf <venv>/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-* ~/.local/bin/ffmpeg`

5. **Slide indices are 1-based, not 0-based.** `--slides 0` silently matches nothing (the
   run loop just does nothing and exits "done" with zero output, no error) — check
   `input.json`'s actual `index` values, or just omit `--slides` to run everything.

6. **The real TRIBE v2 API differs from what a documentation-only read would guess.**
   `TribeModel.get_events_dataframe()` takes **exactly one** of `text_path` / `audio_path` /
   `video_path` (raises `ValueError` if zero or more than one given — you cannot combine a
   silent slide image with separate narration audio in one call), and `.predict(events)`
   takes that dataframe directly, not raw path/transcript kwargs. Even `text_path` doesn't
   skip audio — it does `gTTS` synthesis internally then the same whisperx transcription
   round-trip, because TRIBE fundamentally needs word-level *timing* (it predicts a brain
   response time series, so text needs a time axis from somewhere). There is no way to feed
   TRIBE v2 pure untimed text.

7. **Long continuous audio is computationally infeasible in one pass.** TRIBE v2's text
   encoder computes *contextualized* word embeddings — every single word re-encodes the
   entire preceding transcript through the 3B-parameter Llama model, so cost grows with
   total document length, not just word count. A single slide (3–25 words) finishes in
   seconds. A 25-minute lecture (~12,300 word-level events) was still grinding through the
   embedding stage after 30+ minutes with a **30+ hour ETA** before being killed. Fix:
   split long audio into short (~2 minute) independent chunks with ffmpeg's segment muxer
   (`ffmpeg -i in.mp3 -f segment -segment_time 120 -c copy chunk_%02d.mp3`) and run each
   chunk through its own `get_events_dataframe`/`predict` call — each chunk then gets its
   own short context, reproducing the fast per-slide rate (~2-5 sec/item instead of
   ~9-10 sec/item and climbing).

8. **HuggingFace login: prefer the cached token over asking again.** `huggingface-cli
   login` (or `huggingface_hub.login(token=...)`) caches the token at
   `~/.cache/huggingface/token`; anything using `huggingface_hub` afterward (including a
   fresh script) picks it up automatically via `HfApi().whoami()` — no need to re-supply
   `HF_TOKEN` every run. Gated access to `meta-llama/Llama-3.2-3B` requires the account
   owner to have accepted its license on huggingface.co first; nobody else can do that step,
   and a 403/`GatedRepoError` means either that wasn't done yet or the token lacks gated-repo
   read scope — don't try to work around it, it needs the account owner.

9. **The precompute venv needs a few backend-only packages too**, if you use
   `sightline.store.RunStore`/`sightline.ingest.parse` (or their renamed `profe.*`
   equivalents) directly from a script in this venv (e.g. to ingest a new deck) — `pymupdf`
   specifically isn't in `requirements.txt` (it's a `backend/pyproject.toml` dependency,
   deliberately kept out of the isolated precompute env), so install it separately if a
   script imports those modules.

## Running things

**Per-slide precompute** (the normal path, via `backend/scripts/precompute_neural/run.py`):
```bash
export PATH=$HOME/.local/bin:$PATH
cd ~/SightLine/backend/scripts/precompute_neural && source .venv/bin/activate
python run.py --run-id <run_id> --out ../../fixtures/neural --slides 1,2,3   # or omit --slides for all
```
Resumable — a slide with an existing `metrics.json` is skipped unless `--force`.

**Ingesting a new deck** (PDF → the run-store format `run.py` reads) without needing the
full FastAPI app or an Anthropic key — `ingest.parse()`/`profe.ingest.parse()` needs no LLM
call for basic slide extraction:
```python
import sys; sys.path.insert(0, "backend")
from profe.ingest import parse          # or sightline.ingest, depending on rename state
from profe.store import RunStore, data_dir
slides = parse("some_deck.pdf")
RunStore(data_dir() / "history").create_draft(
    title="...", source_filename="some_deck.pdf", slides=slides,
    inferred=None, inference_error=None,
)
```

**Real continuous audio** (not per-slide synthetic narration) — only tractable in short
clips per the contextual-embedding cost issue above. See a `run_lecture_audio.py`-style
script (feeds `audio_path` directly, no per-slide split) — chunk first with ffmpeg if the
source is longer than a few minutes.

## Backgrounding, monitoring, and killing jobs

- **Fully detach or the SSH command hangs**, even with `nohup ... &`. Redirect *all three*
  streams: `nohup cmd > log 2>&1 < /dev/null & disown`. Missing `< /dev/null` leaves stdin
  attached to the SSH session's pty, which can make the launching `ssh` invocation itself
  hang or time out even though the background job did start correctly — always double-check
  with a **separate** `ssh ... "pgrep -af <name>"` call rather than trusting the launch
  command's own exit status.
- Monitor with `ps -p <pid> -o pid,etime,cmd`, `pgrep -af <pattern>`, `tail -c N logfile`,
  and check for the expected output file (`metrics.json`, etc.) as the real completion
  signal — a tqdm progress bar's ETA can be wildly optimistic early on (cold caches, model
  loading) or pessimistic once it settles, so don't fully trust it without a data point or
  two to confirm the steady-state rate.
- A `for f in ...; do python job.py --input $f; done` batch loop run via `nohup bash -c
  '...' & disown` can be stopped **without killing the currently-running job**: kill the
  wrapper `bash -c` PIDs (the loop control), and the already-launched child process (the
  current iteration) keeps running to completion independently (it gets reparented, not
  killed) — the loop just won't proceed to the next iteration once its parent is gone. This
  is how "let chunk N finish, but don't start chunk N+1" was done.
- `pkill -9 -f <pattern>` for a clean kill by process name when you don't have exact PIDs.

## Transferring results back

`scp` per file/dir works fine for pulling `metrics.json`/PNGs back to the local machine.
For a lot of files, loop over a known directory structure rather than trying to `scp -r`
an entire fixtures tree in one shot (slower to debug if one file fails partway).
