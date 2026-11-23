PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8080
WORKER_PORT ?= 8081
DASHBOARD_PORT ?= 8082

.PHONY: help venv install check run run-worker dashboard test lint fmt smoke smoke-m1 smoke-m2 smoke-m3 smoke-m4 smoke-m5 smoke-m6 worker-bootstrap worker-check install-all clean

help:
	@echo "Hyper-Cluster Omega"
	@echo "  make install          create the venv and install dependencies"
	@echo "  make check            validate the configuration"
	@echo "  make run              start the head node on :$(PORT)"
	@echo "  make run-worker       start a mesh worker on :$(WORKER_PORT)"
	@echo "  make dashboard        start the mobile Mission Control on :$(DASHBOARD_PORT)"
	@echo "  make worker-check     validate the worker environment (no changes)"
	@echo "  make worker-bootstrap provision a worker from scratch (zero-touch)"
	@echo "  make test             run the full test-suite"
	@echo "  make smoke            run the Milestone 1 smoke test"
	@echo "  make smoke-m2         run the Milestone 2 smoke test"
	@echo "  make smoke-m3         run the Milestone 3 smoke test (auto-scaler + spawner)"
	@echo "  make smoke-m4         run the Milestone 4 smoke test (memory + self-evolution)"
	@echo "  make smoke-m5         run the Milestone 5 smoke test (mobile mission control)"
	@echo "  make smoke-m6         run the Milestone 6 smoke test (truth + real jail)"
	@echo "  make install-all      run the interactive installer (the phone setup)"
	@echo "  make lint             run ruff over the whole project"
	@echo "  make clean            remove caches"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install -q --upgrade pip setuptools wheel
	$(BIN)/python -m pip install -q -r requirements.txt

check:
	$(BIN)/python -m head_node.serve --check

run:
	$(BIN)/python -m head_node.serve --host 0.0.0.0 --port $(PORT)

run-worker:
	$(BIN)/python -m worker.worker_agent

dashboard:
	$(BIN)/python -m mobile.dashboard --host 0.0.0.0 --port $(DASHBOARD_PORT)

worker-check:
	bash worker/bootstrap.sh --check

worker-bootstrap:
	bash worker/bootstrap.sh --install

test:
	$(BIN)/python -m pytest tests/ -q

lint:
	$(BIN)/python -m ruff check core head_node worker tools memory training mobile tests scripts

fmt:
	$(BIN)/python -m ruff format core head_node worker tools memory training mobile tests scripts

smoke: smoke-m1

smoke-m1:
	$(BIN)/python scripts/smoke_m1.py

smoke-m2:
	$(BIN)/python scripts/smoke_m2.py

smoke-m3:
	$(BIN)/python scripts/smoke_m3.py

smoke-m4:
	$(BIN)/python scripts/smoke_m4.py

smoke-m5:
	$(BIN)/python scripts/smoke_m5.py

smoke-m6:
	$(BIN)/python scripts/smoke_m6.py

install-all:
	bash install_all.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
