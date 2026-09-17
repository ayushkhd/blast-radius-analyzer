# Developer entry points. Every target runs inside the uv-managed virtualenv.
#
#   make install    create .venv with all extras and dev tools
#   make data       check that the scanner exports are in place and intact
#   make ingest     build the index artifact from the scanner exports
#   make serve      run the API and UI on http://localhost:8000
#   make eval       print the retrieval and end-to-end metrics table
#   make fmt        rewrite files to the house style (isort + pyink)
#   make lint       fail on style drift or pylint findings
#   make typecheck  mypy over the package
#   make test       pytest
#   make check      everything CI runs

.PHONY: install data ingest serve eval fmt lint typecheck test check

PY_DIRS := blast_radius tests
DATA_DIR ?= data

install:
	uv sync --all-extras

data:
	@test -f $(DATA_DIR)/asset_data_scrubbed.json \
	  && test -f $(DATA_DIR)/vulns_data_scrubbed.json \
	  || { echo "Put asset_data_scrubbed.json and vulns_data_scrubbed.json" \
	       "in $(DATA_DIR)/ (see README, 'Getting the data')."; exit 1; }
	cd $(DATA_DIR) && shasum -a 256 -c $(CURDIR)/data.sha256

ingest: data
	uv run blast-radius ingest --data-dir $(DATA_DIR)

serve:
	uv run blast-radius serve

eval:
	uv run blast-radius eval

fmt:
	uv run isort $(PY_DIRS)
	uv run pyink $(PY_DIRS)

lint:
	uv run isort --check-only --diff $(PY_DIRS)
	uv run pyink --check --diff $(PY_DIRS)
	uv run pylint $(PY_DIRS)

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
