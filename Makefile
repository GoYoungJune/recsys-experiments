PYTHON ?= python

.PHONY: install data test run

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) scripts/run_experiments.py
