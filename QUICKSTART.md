# Inicio Rápido — TPV Prediction

## Instalación en 3 Pasos

### 1. Clonar o Descargar
```bash
git clone <tu-repo-url>
cd tpv-prediction
```

### 2. Instalar Dependencias
```bash
# Opción A: pip (recomendado)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt # Correr 2 veces en caso de que salga error

# Opción B: conda
conda env create -f environment.yml
conda activate tpv-prediction
```

### 3. Preparar Datos
```bash
# Colocar el archivo de datos en:
# data/base_proyecto_ml_v2.parquet
```

---

## Uso en 3 Comandos

```bash
# 1. Procesar datos
python scripts/run_data_processing.py

# 2. Generar dataset
python scripts/run_dataset_generation.py --dias-pred 28 --forzar

# 3. Entrenar modelo
python scripts/run_training.py --model-type lightgbm --dias-pred 28 --dias-benchmark 7 --fecha-corte 2025-12-04 --usar-optuna --optuna-trials 20
```

Listo! Tu modelo está entrenado en `models/`. Junto al archivo del modelo se generan automáticamente:
- `*_feature_importance.csv` — importancia porcentual de todas las variables
- `*_feature_importance.pdf` — gráfico con las 10 variables más importantes

---

## Uso Avanzado

Las siguientes son ejecuciones simples con parámetros predefinidos. Para ver la ejecución completa y posibles valores de las variables de ejecución revisar la documentación o utilizar el comando help (ejecución de ejemplo al final de este documento).

### Generar dataset para diferentes horizontes

Recordar poner "--forzar" de ser necesario para reescribir la data antigua.

```bash
python scripts/run_dataset_generation.py --dias-pred 7
python scripts/run_dataset_generation.py --dias-pred 14
python scripts/run_dataset_generation.py --dias-pred 28
```

### Entrenar los 3 modelos y comparar
```bash
python scripts/run_training.py --model-type all
# o directamente:
python scripts/compare_models.py
```

### Regenerar dataset (forzar)
```bash
python scripts/run_dataset_generation.py --forzar
```

### Backtesting con un modelo
```bash
# Nota: fecha_corte es el PRIMER día del test
python scripts/run_backtesting.py --model-type lightgbm --modo global `
    --fechas 2025-10-01 2025-11-01 2025-12-01
```

### Backtesting con todos los modelos
```bash
python scripts/run_backtesting.py --model-type all --modo global `
    --fechas 2025-10-01 2025-11-01 2025-12-01
```

### Modelos individuales por comercio
```bash
python scripts/run_individual.py --model-type lightgbm --fecha-corte 2026-01-01
```


### Predecir con data nueva (modelo ya entrenado)
```bash
# Paso 1: actualizar data cruda y reprocesar
python scripts/run_data_processing.py

# Paso 2: predecir (genera features internamente)
python scripts/run_prediction.py \n    --model-path models/catboost_global_todos_2025-12-04_28dias.cbm \n    --fecha-corte 2026-03-05
# Output: data/predicciones/predicciones_2026-03-05_catboost_global_todos_2025-12-04_28dias.{csv,pdf}
```

Opciones adicionales:
```bash
# Usar la ultima fecha disponible (sin especificar fecha)
python scripts/run_prediction.py --model-path models/catboost_global_todos_2025-12-04_28dias.cbm

# Guardar tambien el dataset de features en data/
python scripts/run_prediction.py --model-path models/catboost_global_todos_2025-12-04_28dias.cbm --guardar-features
```

### Evaluar modelo baseline (Media Móvil)
```bash
# Evaluar el modelo de media móvil actual
python scripts/evaluar_media_movil.py `
    --fechas 2025-10-15 2025-11-15 2025-12-15
```

---

## Usar en Código Python

```python
from src.models.predict import cargar_modelo, predecir
from src.data.loader import load_training_dataset

# Cargar
df, _ = load_training_dataset('data/dataset_entrenamiento_28dias_todos.parquet')
modelo = cargar_modelo('models/lgbm_global_todos_2026-01-01_28dias.txt')

# Predecir
preds = predecir(modelo, df)
```

---

## Estructura de Archivos

```
tpv-prediction/
├── config/         # Configuración (comercios, eventos, parámetros)
├── data/           # Datos (colocar archivos aquí)
├── models/         # Modelos entrenados (generado automáticamente)
├── results/        # Resultados de backtesting (generado)
├── data/predicciones/ # Outputs de run_prediction.py (CSV + PDF)
├── src/            # Código fuente
├── scripts/        # Scripts ejecutables (incluye run_prediction.py)
└── notebooks/      # Notebooks Jupyter originales
```

---

## Documentación Completa

- `README.md` — Documentación principal y referencia de argumentos
- `ESTRUCTURA.md` — Arquitectura detallada del proyecto
- `MODELOS_MULTIPLE.md` — Guía comparativa de los 3 modelos
- `GUIA_COMPLETA_MODELOS.md` — Casos de uso, backtesting y estrategias
- `PARAMETROS_EXPLICADOS.md` — Parámetros y sus equivalencias entre modelos

---

## Problemas Comunes

**`FileNotFoundError: base_proyecto_ml_v2.parquet`**
→ Colocar el archivo en `data/base_proyecto_ml_v2.parquet`

**`No module named 'lightgbm'` / `'catboost'` / `'xgboost'`**
→ Ejecutar `pip install -r requirements.txt`

**`No se encontró dataset_entrenamiento_*.parquet`**
→ Ejecutar primero `python scripts/run_dataset_generation.py --dias-pred 28`

**`FileNotFoundError: dataset_inicial_limpio_{modo}.parquet`**
→ Verificar que `MODO_COMERCIOS` en `config/config.py` coincide con el modo usado al procesar datos

**`No se encontró el modelo`**
→ Entrenar primero con `python scripts/run_training.py`

**`No se encontro dataset_inicial_limpio_{modo}.parquet`** (en run_prediction.py)
→ Ejecutar `python scripts/run_data_processing.py`

**`Dataset insuficiente`**
→ La `fecha_corte` + `dias_benchmark` supera el rango del dataset. Usar una fecha más temprana.

---

## Ayuda

```bash
python scripts/run_training.py --help
python scripts/run_individual.py --help
python scripts/run_backtesting.py --help
python scripts/compare_models.py --help
python scripts/evaluar_media_movil.py --help
python scripts/run_prediction.py --help
```