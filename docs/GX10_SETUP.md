# GX10 setup: running TRIBE v2 precompute

Instructions for whoever (or whichever agent) is SSHed into the GX10 to set up and run
Meta's TRIBE v2 brain-encoding model ([github.com/facebookresearch/tribev2](https://github.com/facebookresearch/tribev2))
for PROFE's neural precompute step (`backend/scripts/precompute_neural/`).

This must run in **complete isolation** from any other Python environment on this machine.
TRIBE v2 pins `numpy==2.2.6` exactly and `torch<2.7`, which will break anything else
installed globally or in another venv. Do not `pip install` anything system-wide or into
an existing venv at any point below.

## 0. Locate or clone the repo

Check whether the PROFE repo already exists on this machine (e.g.
`find / -maxdepth 4 -iname "SightLine" -type d 2>/dev/null` or ask the user for the path).
If it's not present, clone it:

```bash
git clone https://github.com/PatniNayansh/HackMIT.git SightLine
cd SightLine
git checkout tribev2-implementation
```

If it already exists, `cd` into it, `git fetch`, and make sure you're on
`tribev2-implementation` (`git checkout tribev2-implementation && git pull`). Do not commit
or push anything from this session unless explicitly told to.

## 1. Confirm you have a CUDA GPU

```bash
nvidia-smi
```

If this fails or shows no GPU, stop and report back — everything past this point assumes
CUDA is available (CPU-only would take hours per slide instead of minutes).

## 2. Create the isolated venv

```bash
cd backend/scripts/precompute_neural
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # confirm it reports 3.11.x
```

If `python3.11` isn't found, check what's installed (`ls /usr/bin/python3*` or
`pyenv versions`) and report back rather than substituting a different version — TRIBE v2
requires >=3.11.

## 3. Log into Hugging Face

You'll need a Hugging Face access token (a string starting with `hf_`) with read access to
gated repos. This requires the account owner to have already accepted the license at
[huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) —
nobody else can do that step. Don't echo the token back or write it to any file/log.

```bash
pip install huggingface_hub
huggingface-cli login
# paste the token when prompted; answer "y" if asked about git credential storage
```

Then verify gated access actually works before installing anything else:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('meta-llama/Llama-3.2-3B'))"
```

- If this prints model metadata → proceed.
- If it raises a 403 or `GatedRepoError` → stop and report back. It means either the
  license wasn't accepted on huggingface.co yet, or the token doesn't have gated-repo read
  access. Don't try workarounds (mirrors, re-downloading elsewhere) — this needs the account
  owner to fix it on the HF website.

## 4. Install the rest of the pinned dependencies

Still inside the activated `.venv`, from `backend/scripts/precompute_neural`:

```bash
pip install -r requirements.txt
```

This installs a CUDA 12.4 torch build, the exact numpy pin, `transformers`, `nilearn`,
`moviepy`, `gtts`, etc. All pins are deliberate — do not loosen them if something conflicts;
report the conflict instead.

## 5. Clone and install TRIBE v2 itself

It's not on PyPI, so install from source, outside the `backend/` tree:

```bash
cd ../../..            # back to SightLine repo root
git clone https://github.com/facebookresearch/tribev2
pip install -e tribev2
```

## 6. Run one slide as a smoke test

```bash
cd backend/scripts/precompute_neural
python run.py --run-id sample-llm-serving --out ../../fixtures/neural --slides 0
```

Expected: after downloading ~8–10GB of weights (first run only), it produces
`backend/fixtures/neural/sample-llm-serving/0/metrics.json` plus 4 PNGs.

**If it errors inside `_call_tribe()` in `run.py`:** that function was written from
documentation research, not a live test run, so its call signature may not match the real
TRIBE v2 API. Open the real repo's `README.md` and `tribe_demo.ipynb` (in the `tribev2`
clone from step 5), compare against how `run.py` calls `TribeModel.from_pretrained(...)`
and any inference call, and fix only that function — don't restructure anything else in
`run.py`.

## 7. Report back

Report: whether the smoke test succeeded, the exact error text if not, and what (if
anything) was changed in `_call_tribe()`. Don't run `--force` on a full deck or touch other
branches/files until the smoke-test result is reviewed.

## License note

TRIBE v2's weights are CC-BY-NC-4.0 — non-commercial use only. Fine for a hackathon demo;
worth flagging if this project goes anywhere past that.
