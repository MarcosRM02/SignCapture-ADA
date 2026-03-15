# Variables
PYTHON = python3
PIP = pip

.PHONY: help install test lint bronze silver gold clean

# Meta-comando por defecto (se ejecuta al escribir solo 'make')
help:
	@echo "Comandos disponibles:"
	@echo "  make install  - Instala dependencias desde requirements.txt"
	@echo "  make test     - Ejecuta las pruebas con pytest"
	@echo "  make lint     - Revisa el estilo de código con ruff"
	@echo "  make bronze   - Ejecuta la pipeline Bronze"
	@echo "  make silver   - Ejecuta la pipeline Silver"
	@echo "  make gold     - Ejecuta la pipeline Gold"
	@echo "  make clean    - Elimina archivos temporales de Python"

# Comandos de configuración
install:
	$(PIP) install -r requirements.txt

# Comandos de Calidad
test:
	pytest tests/ -v

lint:
	ruff check src/

# Pipelines de Datos
bronze:
	$(PYTHON) -m pipelines.run_bronze

silver:
	$(PYTHON) -m pipelines.run_silver

gold:
	$(PYTHON) -m pipelines.run_gold

# Limpieza
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete