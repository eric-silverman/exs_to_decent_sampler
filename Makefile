PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python

.DEFAULT_GOAL := help

$(BIN)/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: help venv install upgrade sync compile run freeze check clean

help:
	@echo "Targets:"
	@echo "  venv     - create .venv and upgrade pip"
	@echo "  install  - install deps from requirements.txt"
	@echo "  upgrade  - upgrade deps listed in requirements.txt"
	@echo "  sync     - pip-tools sync if available; else pip install"
	@echo "  compile  - pip-compile requirements.in -> requirements.txt"
	@echo "  run      - run main script with venv"
	@echo "  freeze   - write exact deps to requirements.lock"
	@echo "  check    - print Pillow version via venv"
	@echo "  clean    - remove caches and build outputs"

venv: $(BIN)/activate

install: venv
	$(PIP) install -r requirements.txt

upgrade: venv
	$(PIP) install --upgrade -r requirements.txt

sync: venv
	@if [ -x "$(BIN)/pip-sync" ]; then \
		$(BIN)/pip-sync requirements.txt; \
	else \
		echo "pip-tools not installed; falling back to 'pip install -r requirements.txt'"; \
		$(PIP) install -r requirements.txt; \
	fi

compile: venv
	@if [ -x "$(BIN)/pip-compile" ]; then \
		$(BIN)/pip-compile --output-file=requirements.txt requirements.in; \
	else \
		echo "pip-tools not installed; install with '$(PIP) install pip-tools'"; \
	fi

run: venv
	$(PY) exs_to_decent_sampler.py

freeze: venv
	$(PIP) freeze > requirements.lock

check: venv
	@$(PY) - <<-'PY'
	import PIL
	print("Pillow", PIL.__version__)
	PY

clean:
	rm -rf __pycache__/ .pytest_cache/ build/ dist/ *.egg-info/
