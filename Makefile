# One virtualenv for the whole project, at the repo root. Never install into a global Python.
VENV := .venv

# This repo is developed on both macOS and Windows (the neural precompute runs on a Linux GPU box
# besides), so the venv layout is detected rather than assumed: Windows puts the binaries in
# Scripts/ and has no `python3.11` on PATH -- the py launcher is how you ask for a version there.
ifeq ($(OS),Windows_NT)
  PY        := $(VENV)/Scripts
  BOOTSTRAP := py -3.11
else
  PY        := $(VENV)/bin
  BOOTSTRAP := python3.11
endif

# Extra pytest arguments, e.g.  make test ARGS="tests/test_audiences.py -v"
ARGS ?=
# Which hand-written slide `make demo` runs: jargon | clear
SLIDE ?= jargon
# How many times `make gate-repeat` runs the live gate
N ?= 5
# Port for `make dev`
PORT ?= 8000

.PHONY: setup test gate gate-repeat demo dev sample clean-venv

# Idempotent: creates .venv if missing, then (re)installs the backend and its dev deps.
setup: $(PY)/activate
	$(PY)/pip install -q -e "backend[dev]"
	# macOS marks the editable-install .pth file "hidden" and Python then ignores it,
	# which makes `import profe` fail outside backend/. Clear the flag.
	@chflags -R nohidden $(VENV) 2>/dev/null || true
	@$(PY)/python -c "import profe; print('profe importable from', profe.__file__)"

$(PY)/activate:
	$(BOOTSTRAP) -m venv $(VENV)

# Offline and deterministic: fake LLM + the local embedding model. No API key needed.
test: setup
	cd backend && ../$(PY)/pytest $(ARGS)

# The go/no-go check for the whole project. Calls the live LLM; fails loudly if it cannot.
gate: setup
	cd backend && ../$(PY)/pytest -m live -o addopts="" -v -s

# Run the live gate N times (fresh cache each) and report metric spread, latency and pass rates.
gate-repeat: setup
	$(PY)/python backend/scripts/gate_repeat.py $(N)

# Run the real audiences on one hand-written slide and print each one's reading.
demo: setup
	$(PY)/python backend/scripts/try_slide.py $(SLIDE)

# The review UI: FastAPI serves the API and the plain HTML/JS frontend from one port. Saved runs
# open with no API key; starting a new review needs OPENAI_API_KEY in .env.
dev: setup
	@echo "ProFe: http://localhost:$(PORT)"
	$(PY)/python -m uvicorn profe.server:app --app-dir backend --port $(PORT) --reload --reload-dir backend/profe

# Rebuild the bundled sample run (backend/fixtures/runs/) with the real model. ~22 API calls.
sample: setup
	$(PY)/python backend/scripts/make_sample_run.py

clean-venv:
	rm -rf $(VENV)
