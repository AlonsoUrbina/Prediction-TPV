# Guía Completa: Modelos Globales e Individuales con Backtesting

Notar que esta es una guía de ejecución simple. Para revisar la totalidad de parámetros verificar en los archivos "run" de "scripts".

## Metodología de Corte Temporal

Todos los modos usan el mismo esquema de 4 zonas para evitar leakage:

```
── TRAIN ───── VALIDACIÓN (dias_pred) ─── GAP (dias_pred) ─── TEST (dias_benchmark) ──
           ↑ val_inicio                ↑ gap_inicio        ↑ fecha_corte             ↑ fin_test
```

- **`fecha_corte`** es el **primer día** (o único) del período de test.
- El **GAP** excluye las filas cuya ventana de target se solapa con el test.
- La **validación** se usa exclusivamente para early stopping (y overfitting), puede quitarse de ser necesario.

---

## 1. Modelos Globales

Un solo modelo entrenado con datos de **todos los comercios**.

### Uso Simple

```bash
# LightGBM Global
python scripts/run_training.py --model-type lightgbm --dias-pred 28 --dias-benchmark 7 --fecha-corte 2025-11-01

# CatBoost Global
python scripts/run_training.py --model-type catboost --dias-pred 28 --dias-benchmark 7 --fecha-corte 2025-11-01

# XGBoost Global
python scripts/run_training.py --model-type xgboost --dias-pred 28 --dias-benchmark 7 --fecha-corte 2025-11-01

# Todos a la vez
python scripts/run_training.py --model-type all --dias-pred 28 --dias-benchmark 7 --fecha-corte 2025-11-01
```

**Nota sobre datasets:**
```bash
# Generar dataset primero (si no existe)
# Nota: el archivo generado incluye el sufijo del MODO_COMERCIOS configurado en config.py
python scripts/run_dataset_generation.py --dias-pred 28

# Luego entrenar
python scripts/run_training.py --model-type catboost --fecha-corte 2025-10-01
```



### Archivos generados

Al entrenar cada modelo se generan automáticamente tres archivos en `models/`:

```
models/
├── lgbm_global_{modo}_2026-01-01_28dias.txt
├── lgbm_global_{modo}_2026-01-01_28dias_feature_importance.csv
├── lgbm_global_{modo}_2026-01-01_28dias_feature_importance.pdf
├── catboost_global_{modo}_2026-01-01_28dias.cbm
├── catboost_global_{modo}_2026-01-01_28dias_feature_importance.csv
├── catboost_global_{modo}_2026-01-01_28dias_feature_importance.pdf
├── xgboost_global_{modo}_2026-01-01_28dias.json
├── xgboost_global_{modo}_2026-01-01_28dias_feature_importance.csv
└── xgboost_global_{modo}_2026-01-01_28dias_feature_importance.pdf
```

El CSV contiene las columnas `rank`, `feature`, `importancia` e `importancia_pct` (porcentaje sobre el total), ordenadas de mayor a menor importancia. El PDF muestra un gráfico horizontal de barras con las 10 variables más importantes.

---

## 2. Modelos Individuales

Un modelo específico para **cada comercio**.

### Uso

```bash
# LightGBM Individual
python scripts/run_individual.py `
    --model-type lightgbm `
    --fecha-corte 2026-01-01 `
    --dias-pred 28 `
    --dias-benchmark 7 `
    --usar-optuna `
    --guardar-modelos `
    --no-verbose


# CatBoost Individual
python scripts/run_individual.py `
    --model-type catboost `
    --fecha-corte 2026-01-01 `
    --dias-pred 28 `
    --dias-benchmark 7

# XGBoost Individual
python scripts/run_individual.py `
    --model-type xgboost `
    --fecha-corte 2026-01-01 `
    --dias-pred 28 `
    --dias-benchmark 7

# Con optimización Optuna
python scripts/run_individual.py `
    --model-type catboost `
    --fecha-corte 2026-01-01 `
    --usar-optuna
```

---

## 3. Backtesting Global

Evalúa un modelo **global** en múltiples fechas de corte para medir estabilidad temporal.

### Uso

```bash
# Todos los modelos a la vez (RECOMENDADO para comparar)
python scripts/run_backtesting.py `
    --model-type all `
    --modo global `
    --dias-pred 28 `
    --dias-benchmark 7 `
    --fechas 2025-07-01 2025-08-01 2025-09-01 2025-10-01 2025-11-01 2025-12-01

# LightGBM Backtesting Global
python scripts/run_backtesting.py `
    --model-type lightgbm `
    --modo global `
    --dias-pred 28 `
    --dias-benchmark 7 `
    --fechas 2025-07-01 2025-08-01 2025-09-01 2025-10-01 2025-11-01 2025-12-01

# CatBoost Backtesting Global
python scripts/run_backtesting.py `
    --model-type catboost `
    --modo global `
    --fechas 2025-07-01 2025-08-01 2025-09-01

# XGBoost Backtesting Global
python scripts/run_backtesting.py `
    --model-type xgboost `
    --modo global `
    --fechas 2025-07-01 2025-08-01 2025-09-01
```

**Nota:** Con `--model-type all`, genera 9 PDFs (3 por modelo: métricas, porcentual, miles de millones)

---

## 4. Backtesting Individual

Evalúa modelos **individuales** en múltiples fechas.

### Uso

```bash
# LightGBM Backtesting Individual
python scripts/run_backtesting.py `
    --model-type lightgbm `
    --modo individual `
    --dias-pred 28 `
    --dias-benchmark 7 `
    --fechas 2025-07-01 2025-08-01 2025-09-01

# CatBoost con Optuna
python scripts/run_backtesting.py `
    --model-type catboost `
    --modo individual `
    --usar-optuna `
    --fechas 2025-07-01 2025-08-01 2025-09-01
```

### Archivos generados
```
results/
├── backtesting_individual_lightgbm_metricas_28dias.pdf
├── backtesting_individual_lightgbm_porcentual_28dias.pdf
├── backtesting_individual_lightgbm_cantidad_28dias.pdf
├── backtesting_individual_catboost_metricas_28dias.pdf
├── backtesting_individual_catboost_porcentual_28dias.pdf
└── backtesting_individual_catboost_cantidad_28dias.pdf
```

---

## Comparación de Configuraciones Por Modelo

| Modelo       | Velocidad  | Precisión | Mejor para                         |
|--------------|------------|-----------|------------------------------------|
| **LightGBM** | Más rápido | Buena     | Datasets grandes, iteración rápida |
| **CatBoost** | Medio      | Mejor     | Máxima precisión                   |
| **XGBoost**  | Medio      | Buena     | Balance y estabilidad              |

---

## Casos de Uso Recomendados

### Caso 1: Exploración Inicial

```bash
# 1. Generar dataset (si no existe)
python scripts/run_dataset_generation.py --dias-pred 28 --forzar

# 2. Comparar los 3 modelos globales
python scripts/compare_models.py

# 3. Backtesting del mejor con todos los modelos
python scripts/run_backtesting.py --model-type all --modo global `
    --fechas 2025-10-01 2025-11-01 2025-12-01
```

### Caso 2: Máxima Precisión por Comercio

```bash
# 1. Generar dataset (si no existe)
python scripts/run_dataset_generation.py --dias-pred 28

# 2. Baseline global
python scripts/run_training.py --model-type catboost

# 3. Modelos individuales con optimización
python scripts/run_individual.py --model-type catboost --usar-optuna
```

### Caso 3: Validación Temporal Completa (12 meses)

```bash
python scripts/run_backtesting.py `
    --model-type all `
    --modo global `
    --fechas 2025-01-01 2025-02-01 2025-03-01 2025-04-01 `
             2025-05-01 2025-06-01 2025-07-01 2025-08-01 `
             2025-09-01 2025-10-01 2025-11-01 2025-12-01
```

### Caso 4: Estrategia Híbrida (individual para TOP, global para el resto)

```python
# Pseudocódigo
comercios_top = ['MERCPAGO', 'DLOCAL', 'PAYU']

for comercio in df['nombre_comercio'].unique():
    if comercio in comercios_top:
        prediccion = modelo_individual[comercio].predict(...)
    else:
        prediccion = modelo_global.predict(...)
```

---

## Argumentos Disponibles

### `run_dataset_generation.py` (NUEVO en v5.0)
```
--dias-pred INT           # Horizonte de predicción (default: config.DIAS_PRED)
--forzar                  # Forzar regeneración aunque el dataset exista
```

### `run_training.py`
```
--model-type {lightgbm,catboost,xgboost,all}
--dias-pred INT           # Horizonte de predicción (default: config.DIAS_PRED)
--dias-benchmark INT      # Días del período de test (default: config.DIAS_BENCHMARK)
--fecha-corte YYYY-MM-DD  # Primer día del test (hacia adelante)
```

### `run_individual.py`
```
--model-type {lightgbm,catboost,xgboost}
--fecha-corte YYYY-MM-DD  # Primer día del test
--dias-pred INT
--dias-benchmark INT      # Días del período de test (default: config.DIAS_BENCHMARK)
--usar-optuna
--no-verbose
--guardar-modelos
```

### `run_backtesting.py`
```
--model-type {lightgbm,catboost,xgboost,all}  # Ahora soporta 'all'
--modo {global,individual}
--dias-pred INT
--dias-benchmark INT      # Días del período de test (default: 7)
--usar-optuna             # Solo para modo individual
--fechas YYYY-MM-DD ...   # Lista de fechas de corte (primer día del test)
```

### `compare_models.py`
```
--dias-pred INT
--dias-benchmark INT      # Días del período de test (default: config.DIAS_BENCHMARK)
--fecha-corte YYYY-MM-DD  # Primer día del test
--sin-graficos
```

---

## 5. Evaluación de Baseline (Media Móvil)

Evalúa el modelo actual de **Media Móvil** para establecer un baseline de comparación.

**Modelo:** `prediccion_mes = m × x + n`
- **m** = Promedio de los últimos 28 días
- **x** = Días restantes del mes
- **n** = TPV acumulado hasta ayer

### Uso

```bash
# Evaluar media móvil en fechas específicas
python scripts/evaluar_media_movil.py `
    --fechas 2025-10-15 2025-11-15 2025-12-15

# Comparar con modelos ML (después de ejecutar backtesting)
python scripts/comparar_con_media_movil.py `
    --fechas 2025-10-15 2025-11-15 2025-12-15
```

### Flujo Completo de Comparación

```bash
# 1. Establecer baseline
python scripts/evaluar_media_movil.py `
    --fechas 2025-07-15 2025-08-15 2025-09-15 `
             2025-10-15 2025-11-15 2025-12-15

# 2. Evaluar modelos ML
python scripts/run_backtesting.py `
    --model-type all `
    --modo global `
    --fechas 2025-07-15 2025-08-15 2025-09-15 `
             2025-10-15 2025-11-15 2025-12-15

# 3. Comparar resultados
python scripts/comparar_con_media_movil.py `
    --fechas 2025-07-15 2025-08-15 2025-09-15 `
             2025-10-15 2025-11-15 2025-12-15
```


## Troubleshooting

**"No se encontró dataset_entrenamiento_*_{modo}.parquet"**
```bash
# Generar dataset primero (el nombre incluye el MODO_COMERCIOS de config.py)
python scripts/run_dataset_generation.py --dias-pred 28
```

**"Dataset insuficiente"**
→ La `fecha_corte + dias_benchmark` supera el rango del dataset. Usar una fecha más temprana.

**Nota sobre fecha_corte (v5.0):**
→ `fecha_corte` es el **PRIMER día del test**, no el último. Los días de benchmark van hacia adelante.
```bash
# Ejemplo: fecha_corte = 2025-10-01, dias_benchmark = 7
# → Test: 2025-10-01 al 2025-10-07 (7 días hacia adelante)
```

**Backtesting muy lento**
```bash
# Opción 1: Reducir número de fechas
--fechas 2025-10-01 2025-11-01 2025-12-01

# Opción 2: Ejecutar un modelo a la vez (en lugar de --model-type all)
--model-type catboost
```

**Modelos individuales sin memoria**
```python
# Aumentar filtro mínimo en train_individual.py
if len(df_cliente) < 200:  # en lugar de 60
    continue
```