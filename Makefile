.PHONY: setup sample smoke download prepare benchmark test
PYTHON ?= python3
RUN = $(PYTHON) -m marketing_uplift.pipeline

setup:
	$(PYTHON) -m pip install -e '.[dev]'

sample:
	$(RUN) sample --source data/smoke/synthetic-smoke.csv.gz --rows 50000

smoke: sample
	$(RUN) all --source data/smoke/synthetic-smoke.csv.gz --full data/smoke/full.parquet --sample data/smoke/model_sample.parquet --output outputs/smoke/metrics.json --sample-rows 30000 --outcome visit

download:
	$(RUN) download --i-accept-license

prepare:
	$(RUN) prepare --sample-rows 1000000

benchmark:
	$(RUN) benchmark --outcome visit --budget-fraction 0.10

test:
	$(PYTHON) -m pytest -q
