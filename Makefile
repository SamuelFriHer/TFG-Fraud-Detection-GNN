.PHONY: install test lint typecheck clean

# Instala las dependencias del proyecto
install:
	pip install -r requirements.txt

# Ejecuta todos los tests con pytest
test:
	PYTHONPATH=. pytest tests/ -v

# Pasa el linter ruff para revisar la calidad del código
lint:
	ruff check .

# Pasa el typechecker de Mypy para validar los tipos definidos
typecheck:
	mypy src/

# Limpia directorios de caché para forzar ejecuciones limpias
clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
