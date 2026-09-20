# Thin wrappers around scripts/ and pytest. Run `pip install -e .` first (or just rely on
# PYTHONPATH below, which makes the scripts importable from a plain checkout).
PYTHON ?= python
export PYTHONPATH := src

.PHONY: data test quick train evaluate lint clean

data:  ## download and verify UCI Adult into data/raw/adult
	$(PYTHON) scripts/download_data.py --summary

test:
	$(PYTHON) -m pytest

quick: data  ## smoke test: 6000 rows, 1 seed, about a minute and a half on two CPU cores
	$(PYTHON) scripts/train_encoders.py --config configs/quick.yaml
	$(PYTHON) scripts/evaluate.py --config configs/quick.yaml

train: data  ## main experiment, stage 1: embeddings for 5 seeds x all methods (slow on CPU)
	$(PYTHON) scripts/train_encoders.py --config configs/full.yaml

evaluate:  ## main experiment, stage 2: probes, tables, figures -> results/full/
	$(PYTHON) scripts/evaluate.py --config configs/full.yaml

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests
	mypy

clean:  ## remove run artefacts and caches (keeps data/ and results/)
	rm -rf runs .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
