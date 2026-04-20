VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install dev lint format type arch test coverage audit ci clean

venv:
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)

install: venv
	$(PIP) install -e .

dev: venv
	$(PIP) install -e ".[dev]"

lint:
	$(VENV_DIR)/bin/ruff check .
	$(VENV_DIR)/bin/ruff format --check .

format:
	$(VENV_DIR)/bin/ruff check --fix .
	$(VENV_DIR)/bin/ruff format .

type:
	$(VENV_DIR)/bin/mypy src/aicache/domain src/aicache/application

arch:
	$(VENV_DIR)/bin/lint-imports

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing

audit:
	$(VENV_DIR)/bin/pip-audit

ci: lint arch coverage
	@echo "Skipping strict mypy — warn-only in Phase 0; run \`make type\` to see current errors."

clean:
	rm -rf build dist *.egg-info .aicache .pytest_cache .ruff_cache .mypy_cache .coverage
