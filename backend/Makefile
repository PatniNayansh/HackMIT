# Convenience only: forward to the repo-root Makefile so `make test` also works from backend/.
# Variables given on the command line (ARGS=, SLIDE=, N=, PORT=) are passed through automatically.
TARGETS := setup test gate gate-repeat demo dev sample clean-venv

.PHONY: $(TARGETS)
$(TARGETS):
	@$(MAKE) --no-print-directory -C .. $@
