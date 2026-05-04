.PHONY: install dev lint format type test check clean

# Use poetry if it's on PATH; fall back to plain python/pip otherwise.
RUN := $(shell command -v poetry >/dev/null 2>&1 && echo "poetry run" || echo "python -m")

install:
	poetry install --with dev || pip install -e ".[dev]"

dev: install

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy src

test:
	pytest

check: lint type test

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
