# Makefile para TPV Prediction

.PHONY: help install clean data dataset train train-all compare individual backtest baseline

help:
	@echo "Comandos disponibles:"
	@echo "  make install     - Instalar dependencias"
	@echo "  make data        - Procesar datos crudos"
	@echo "  make dataset     - Generar dataset con features"
	@echo "  make train       - Entrenar modelo LightGBM (default)"
	@echo "  make train-all   - Entrenar LightGBM + CatBoost + XGBoost"
	@echo "  make compare     - Comparar los 3 modelos"
	@echo "  make individual  - Entrenar modelos individuales por comercio"
	@echo "  make backtest    - Ejecutar backtesting global (LightGBM)"
	@echo "  make baseline    - Evaluar modelo baseline (media movil)"
	@echo "  make clean       - Limpiar archivos temporales"

install:
	pip install -r requirements.txt

data:
	python scripts/run_data_processing.py

dataset:
	python scripts/run_dataset_generation.py --dias-pred 28

train:
	python scripts/run_training.py --model-type lightgbm

train-all:
	python scripts/run_training.py --model-type all

compare:
	python scripts/compare_models.py

individual:
	python scripts/run_individual.py --model-type lightgbm

backtest:
	python scripts/run_backtesting.py --model-type lightgbm --modo global \
		--fechas 2025-07-01 2025-08-01 2025-09-01 2025-10-01 2025-11-01 2025-12-01

baseline:
	python scripts/evaluar_media_movil.py \
		--fechas 2025-07-01 2025-08-01 2025-09-01 2025-10-01 2025-11-01 2025-12-01

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete