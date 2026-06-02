.PHONY: install test lint clean run-traditional run-traditional-large fetch-results

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f logs.txt

run-traditional:
	python -m src.cli traditional --config configs/traditional_hi_small.toml --models all

run-traditional-large:
	python -m src.cli traditional --config configs/traditional_hi_large.toml --models all

run-gnn:
	python -m src.cli gnn --config configs/gnn_hi_small.toml

run-gnn-large:
	python -m src.cli gnn --config configs/gnn_hi_large.toml

run-gnn-grid:
	python -m src.cli gnn-grid --config configs/gnn_hi_small.toml

run-gnn-grid-large:
	python -m src.cli gnn-grid --config configs/gnn_hi_large.toml

# Downloads MLflow results from HF Hub and exports them to outputs/results/<experiment>_results.csv
# Usage: make fetch-results EXPERIMENT=traditional_HI-Small
fetch-results:
	python -m src.cli fetch-results --experiment $(EXPERIMENT)
