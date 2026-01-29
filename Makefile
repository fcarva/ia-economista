.PHONY: install install-dev dashboard backtest train lint format test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

dashboard:
	streamlit run dashboard/Home.py

backtest:
	python scripts/run_backtest.py

train:
	python scripts/train.py

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

test:
	pytest
