.PHONY: install test lint typecheck clean run-traditional run-traditional-large

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check .

typecheck:
	mypy src/

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f logs.txt

run-traditional:
	python -m src.cli traditional --config configs/traditional_hi_small.toml --models all

run-traditional-large:
	python -m src.cli traditional --config configs/traditional_hi_large.toml --models all
