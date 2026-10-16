PY := .venv/bin

.PHONY: install test lint typecheck check report oos

install:            ## create the venv and install with dev + data extras
	python3 -m venv .venv && $(PY)/pip install -e ".[dev,yahoo]"

test:               ## run the test suite with coverage
	$(PY)/pytest -q --cov=regime_engine --cov-report=term-missing:skip-covered

lint:
	$(PY)/ruff check src tests scripts && $(PY)/ruff format --check src tests scripts

typecheck:
	$(PY)/mypy

check: lint typecheck test   ## everything CI runs

report:             ## SPY ablation + tear sheet + README figures (~3 min)
	$(PY)/regime-engine report --config config/spy_daily.toml --out reports --runs runs --figures docs/images

oos:                ## pre-registered out-of-sample study (docs/OUT_OF_SAMPLE.md)
	$(PY)/python scripts/run_out_of_sample.py --out reports/oos
