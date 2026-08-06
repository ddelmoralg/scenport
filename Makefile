.PHONY: help install data forecast backtest exact tables figures all test lint clean

CONFIG ?= configs/base.yaml
PY     ?= python

help:
	@echo "make install    create the conda environment and install the package"
	@echo "make data       download and build the panel and universe (needs network)"
	@echo "make forecast   simulate scenarios for every CPU forecaster, locally"
	@echo "make backtest   run the allocator grid over the stored scenarios"
	@echo "make exact      MILP optimality gaps and the scaling sweep"
	@echo "make tables     build the CSV/LaTeX tables"
	@echo "make figures    build the PDF figures"
	@echo "make all        data -> forecast -> backtest -> tables -> figures"
	@echo "make test       run the test suite"
	@echo
	@echo "On the cluster use the slurm/ scripts instead of forecast/backtest;"
	@echo "these targets run everything in one process and take days."

install:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n scenport pip install -e ".[dev]"

data:
	$(PY) scripts/fetch_data.py --config $(CONFIG)

forecast:
	@for f in rw rw_nodrift arima ets garch prophet prophet_rmse lgbm; do \
		echo "== $$f"; \
		$(PY) scripts/run_forecasts.py --config $(CONFIG) --forecaster $$f || exit 1; \
	done

backtest:
	@for f in rw rw_nodrift arima ets garch prophet prophet_rmse lgbm; do \
		$(PY) scripts/run_backtest.py --config $(CONFIG) --forecaster $$f --allocator all || exit 1; \
	done

exact:
	$(PY) scripts/run_exact.py gaps --config $(CONFIG) --forecaster garch --every 6
	$(PY) scripts/run_exact.py scaling --config $(CONFIG)

tables:
	$(PY) scripts/aggregate.py --config $(CONFIG)

figures:
	$(PY) scripts/make_figures.py --config $(CONFIG)

all: data forecast backtest tables figures

test:
	pytest -q

lint:
	ruff check scenport scripts tests
	ruff format --check scenport scripts tests

clean:
	rm -rf runs artifacts logs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
