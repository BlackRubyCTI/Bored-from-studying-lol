PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8080

.PHONY: help venv install check run test lint fmt clean smoke

help:
	@echo "Hyper-Cluster Omega"
	@echo "  make install   create the venv and install dependencies"
	@echo "  make check     validate the configuration"
	@echo "  make run       start the head node on :$(PORT)"
	@echo "  make test      run the full test-suite"
	@echo "  make smoke     start the node, probe it, stop it"
	@echo "  make clean     remove caches"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install -q --upgrade pip setuptools wheel
	$(BIN)/python -m pip install -q -r requirements.txt

check:
	$(BIN)/python -m head_node.serve --check

run:
	$(BIN)/python -m head_node.serve --host 0.0.0.0 --port $(PORT)

test:
	$(BIN)/python -m pytest tests/ -q

lint:
	$(BIN)/python -m ruff check core head_node tests || true

fmt:
	$(BIN)/python -m ruff format core head_node tests || true

smoke:
	$(BIN)/python scripts/smoke_m1.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
