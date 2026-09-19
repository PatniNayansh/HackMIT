# One virtualenv for the whole project, at the repo root. Never install into a global Python.
VENV := .venv
PY   := $(VENV)/bin

# Extra pytest arguments, e.g.  make test ARGS="tests/test_audiences.py -v"
ARGS ?=
# Which hand-written slide `make demo` runs: jargon | clear
SLIDE ?= jargon
# How many times `make gate-repeat` runs the live gate
N ?= 5

.PHONY: setup test gate gate-repeat demo clean-venv

# Idempotent: creates .venv if missing, then (re)installs the backend and its dev deps.
setup: $(VENV)/bin/activate
	$(PY)/pip install -q -e "backend[dev]"
	# macOS marks the editable-install .pth file "hidden" and Python then ignores it,
	# which makes `import sightline` fail outside backend/. Clear the flag.
	@chflags -R nohidden $(VENV) 2>/dev/null || true
	@$(PY)/python -c "import sightline; print('sightline importable from', sightline.__file__)"

$(VENV)/bin/activate:
	python3.11 -m venv $(VENV)

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

clean-venv:
	rm -rf $(VENV)
