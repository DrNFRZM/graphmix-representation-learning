# Reproduction check

The installation was checked in a fresh Python 3.12.14 virtual environment on
Linux, CPU only. PyTorch was installed from the CPU wheel index, followed by
`pip install -e ".[dev]"`. `pip check` found no dependency conflicts.

| Check | Outcome |
| --- | --- |
| `python -m pytest` | 171 tests passed |
| `make data` | Downloaded and checked both UCI Adult files; 48,842 original rows, 42,468 after cleaning |
| `make quick` | Completed training, evaluation, diagnostics, and figures for the 19 configured methods and the supervised reference |
| `make lint` | Ruff checks and formatting passed; mypy checked 14 source files successfully |
| Committed full-run CSVs | Recomputed the headline means, sample SDs, and record-value paired difference from per-seed metrics; no duplicate metric rows |

The tested numerical stack was NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6,
scikit-learn 1.9.1, PyTorch 2.14.1+cpu, PyG 2.8.0.post1, and Matplotlib 3.11.2.
Tests used one BLAS/OpenMP thread; the quick run used two. It processed a
6,000-row subset with seed 0. This is a smoke run, not a repetition of the
five-seed full experiment. The committed full-run results and their original
software versions remain unchanged.

## Type-checker compatibility

The unconstrained development install initially selected mypy 2.4.0, which
failed internally during `make lint`. mypy 1.20.2 completed the check. The
development requirement now stays on the 1.x series.

The old mypy setting forced a Python 3.11 target even in a Python 3.12
environment. With the newer NumPy stubs selected on Python 3.12, that produced
a syntax error in the dependency's type declarations. mypy now uses the active
interpreter version; Ruff still checks source syntax against Python 3.11,
and the CI workflow still runs Python 3.11. No experiment code was changed.
