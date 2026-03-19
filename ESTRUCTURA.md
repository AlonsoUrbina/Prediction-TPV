# Estructura del Proyecto Predicción TPV

## Árbol de Directorios

```
tpv-prediction/
├── config/
│   └── config.py                  # Parámetros, comercios, mapping RUTs, Cyber
├── data/                          # Datos (no versionados)
│   ├── base_proyecto_ml_v2.parquet          <- datos crudos (colocar aquí y no cambiar nombre)
│   ├── dataset_inicial_limpio_{modo}.parquet       <- generado por run_data_processing.py
│   ├── dataset_entrenamiento_Xdias_{modo}.parquet  <- generado por run_dataset_generation.py
│   ├── encoder_comercios_{modo}.joblib      <- generado por run_dataset_generation.py
│   └── predicciones/                        <- outputs de run_prediction.py
├── guardar_backtesting/           # Almacenamiento externo manual de backtesting
├── models/                        # Modelos entrenados (no versionados)
│   └── individual/                # Modelos individuales por comercio
├── notebooks/                     # Notebooks originales de referencia
│   ├── Data.ipynb                 # Procesamiento y limpieza
│   └── LightGBM.ipynb             # Modelado original
├── scripts/
│   ├── run_data_processing.py     # Paso 1: limpieza de datos crudos
│   ├── run_dataset_generation     # Paso 2: generar el dataset con features
│   ├── run_backtesting.py         # Paso 3: Evaluación en múltiples fechas
│   ├── run_training.py            # Paso 4: entrenamiento de modelos globales (a partir de resultados de backtesting)
│   ├── run_individual.py          # Paso 5: entrenamiento de modelos individuales
│   ├── compare_models.py          # Comparación de algoritmos (Opcional para fecha fija)
│   ├── evaluar_media_movil.py     # Evaluación del modelo media móvil
│   └── run_prediction.py          # Prediccion con data nueva
├── src/
│   ├── data/
│   │   ├── loader.py             # Carga y guardado de datos
│   │   └── preprocessing.py      # Pipeline de limpieza
│   ├── features/
│   │   └── feature_engineering.py # generar_dataset() - pipeline de features
│   └── models/
│       ├── train.py               # LightGBM global
│       ├── train_catboost.py      # CatBoost global
│       ├── train_xgboost.py       # XGBoost global
│       ├── train_individual.py    # Modelos individuales (LGB/CAT/XGB)
│       ├── backtesting.py         # Backtesting global e individual
│       └── predict.py             # Predicción con modelos guardados
├── tests/
├── requirements.txt
├── environment.yml
└── Makefile
```

## Descripción de Módulos

### config/config.py

Configuración centralizada del proyecto. Variables principales:

```python
MAPPING_COMERCIO            # dict RUT -> nombre comercio
MODO_COMERCIOS              # 'todos' o 'algunos' (controla COMERCIOS_A_MANTENER)
COMERCIOS_TODOS             # lista completa de comercios
COMERCIOS_ALGUNOS           # subconjunto de comercios
COMERCIOS_A_MANTENER        # derivado automaticamente segun MODO_COMERCIOS
MCC_INVALIDOS               # Algunos MCC inválidos en la data [0, 34]
MIN_INSTANCIAS_COMERCIO     # Instancias mínimas para considerar un comercio
CYBER_EVENTS                # lista de fechas de CyberDays/BlackFridays...
COMERCIOS_ANTIGUEDAD        # fechas de primera transacción por comercio
DIAS_PRED                   # horizonte de predicción

# Parámetros de modelos (separados por modo)
LIGHTGBM_PARAMS_GLOBAL        LIGHTGBM_PARAMS_INDIVIDUAL
CATBOOST_PARAMS_GLOBAL        CATBOOST_PARAMS_INDIVIDUAL
XGBOOST_PARAMS_GLOBAL         XGBOOST_PARAMS_INDIVIDUAL
NUM_BOOST_ROUND_GLOBAL        NUM_BOOST_ROUND_INDIVIDUAL
EARLY_STOPPING_ROUNDS_GLOBAL  EARLY_STOPPING_ROUNDS_INDIVIDUAL
```

### src/data/loader.py

- `load_raw_data(filepath)`: Carga parquet, retorna DataFrame o None
- `load_training_dataset(filepath)`: Carga dataset + encoder
- `save_dataset(df, filepath)`: Guarda en parquet
- `save_encoder(encoder, filepath)`: Guarda LabelEncoder con joblib

### src/data/preprocessing.py

Pipeline completo aplicado en `run_data_processing.py`:

1. `drop_unnecessary_columns()` — elimina columnas no usadas
2. `drop_nulls()` — `df.dropna()`
3. `map_merchant_names()` — `id_comercio` (RUT) -> `nombre_comercio`
4. `filter_merchants_by_name()` — filtra por `COMERCIOS_A_MANTENER`
5. `filter_merchants_by_min_instances()` — elimina comercios con < 100 filas
6. `clean_mcc()` — elimina MCC 0, 34 y nulos
7. `convert_dtypes()` — convierte tipos

Función principal: `preprocess_data(df, columns_to_drop, mapping_comercio, merchants_to_keep, min_instances, mcc_invalidos)`

### src/features/feature_engineering.py

Función única `generar_dataset(df_raw, df_antiguedad_ref, dias_prediccion, cybers_list, para_prediccion=False)`.

El parametro `para_prediccion` controla si se descartan las filas sin target:
- `False` (default, entrenamiento): hace `dropna(subset=['tpv_futuro'])`, solo filas con target valido
- `True` (prediccion): mantiene todos los rows incluyendo los mas recientes donde `tpv_futuro=NaN`

Implementación directa del notebook LightGBM.ipynb. Pasos internos:

1. Label encoding de `nombre_comercio` -> `id_comercio_num`
2. Cálculo de `antiguedad_meses` (merge con tabla de fechas de primera TX)
3. Calendario de feriados chilenos y Cyber Events (features futuras)
4. Agregación diaria por `(id_comercio_num, mcc, fecha_trx)`
5. Relleno de grilla completa (skeleton de todas las combinaciones posibles)
6. Variables cíclicas: `mes_sin/cos`, `dia_semana_sin/cos`, `dia_mes_sin/cos`, `dias_rest_mes_actual`
7. Rolling windows: acumulados, medias, volatilidad, aceleración, ticket promedio, mix de pago
8. Target: `tpv_futuro` (suma forward `dias_prediccion` días), `tpv_futuro_año_anterior`

### src/models/

**common.py**

Utilidades compartidas por todos los módulos de entrenamiento:
- `preparar_datos(df)` — separa X e y
- `calcular_fechas(fecha_corte, dias_pred, dias_benchmark)` — calcula los límites temporales de cada zona
- `guardar_importancia_variables(modelo, model_type, model_filepath)` — extrae la importancia de variables (en porcentaje), guarda CSV con todas las variables y PDF con las 10 más importantes junto al archivo del modelo

**train.py / train_catboost.py / train_xgboost.py**

Cada archivo expone:
- `entrenar_modelo_global(df, fecha_corte, dias_pred, params)` -> `(modelo, métricas)`
- `guardar_modelo_*(modelo, filepath)`
- `cargar_modelo_*(filepath)`

**train_individual.py**

- `entrenar_modelo_individual(df, fecha_corte, encoder, dias_val, dias_benchmark, model_type, guardar_modelos, models_dir, ...)` -> DataFrame con predicciones
- `guardar_modelo_individual(model, nombre_comercio, model_type, directorio)` -> Path
- `cargar_modelo_individual(nombre_comercio, model_type, directorio)` -> modelo
- `directorio_modelos_individuales(models_dir, model_type, fecha_corte, dias_val)` -> Path

Los modelos individuales se guardan en:
`models/individual/{model_type}_{fecha_corte}_{dias_val}dias/{COMERCIO}.{ext}`

**backtesting.py**

- `ejecutar_backtesting_global(df, fechas_corte, encoder, dias_testeo, model_type)` -> DataFrame resultados
- `ejecutar_backtesting_individual(df, fechas_corte, encoder, dias_testeo, model_type, usar_optuna)` -> (df_pct, df_monto)

**predict.py**

Interfaz unificada para los 3 modelos:
- `cargar_modelo(filepath, model_type=None)` — infiere tipo por extensión (.txt/.cbm/.json)
- `predecir(modelo, df, model_type=None)` — infiere tipo por clase del objeto
- `cargar_y_predecir(filepath, df)` — carga y predice en un paso

## Ejecución Completa

> **Nota:** El sufijo `_{modo}` en los nombres de archivo corresponde al valor de `MODO_COMERCIOS` en `config/config.py` (`todos` o `algunos`). Cambiar esa variable es suficiente para alternar entre el modelo global completo y el parcial.

```bash
# 1. Preparar datos
cp datos.parquet data/base_proyecto_ml_v2.parquet
python scripts/run_data_processing.py
# -> genera data/dataset_inicial_limpio_{modo}.parquet

# 2. Generar dataset con features
python scripts/run_dataset_generation.py --dias-pred 28
# -> genera data/dataset_entrenamiento_28dias_{modo}.parquet y data/encoder_comercios_{modo}.joblib

# 3. Entrenar modelos globales
python scripts/run_training.py --model-type all --fecha-corte 2026-01-01
# -> genera models/lgbm_global_{modo}_*.txt, models/catboost_global_{modo}_*.cbm, etc.

# 4. Comparar modelos
python scripts/compare_models.py

# 5. Backtesting global (ahora soporta --model-type all)
python scripts/run_backtesting.py --modo global --model-type all `
    --fechas 2025-07-01 2025-08-01 2025-09-01
# -> genera 9 PDFs en results/ (3 por modelo)

# 6. Modelos individuales (opcional)
python scripts/run_individual.py --model-type catboost --guardar-modelos
# -> genera models/individual/catboost_2026-01-01_28dias/MERCPAGO.cbm, ...

# 7. Backtesting individual (opcional)
python scripts/run_backtesting.py --modo individual --model-type catboost `
    --fechas 2025-10-01 2025-11-01

# 8. Predecir con data nueva (modelos ya entrenados)
python scripts/run_data_processing.py              # actualizar dataset_inicial_limpio_{modo}.parquet
python scripts/run_prediction.py --model-path models/catboost_global_todos_2025-12-04_28dias.cbm  --fecha-corte 2026-03-05
# -> genera data/predicciones/predicciones_2026-03-05_catboost_global_todos_2025-12-04_28dias.{csv,pdf}

# 9. Evaluar baseline de media movil (opcional)
python scripts/evaluar_media_movil.py `
    --fechas 2025-10-01 2025-11-01 2025-12-01

```

**Nota importante sobre fecha_corte:**
- `fecha_corte` es el **PRIMER día del test**, no el último
- Los días de benchmark van **hacia adelante** desde fecha_corte
- Ejemplo: `--fecha-corte 2025-10-01 --dias-benchmark 7`
  - Test: 2025-10-01 al 2025-10-07 (7 días hacia adelante)

## Diferencias Global vs Individual

| Aspecto                  | Global               | Individual                     |
|--------------------------|----------------------|--------------------------------|
| Cantidad de modelos      | 1                    | N (uno por comercio)           |
| Agrupación en train      | todos los comercios  | cada comercio por separado     |
| tweedie_variance_power   | 1.0                  | 1.6                            |
| Iteraciones / early stop | 10000 / 500          | 10000 / 500                    |
| Complejidad del modelo   | Alta (256 hojas)     | Baja (15 hojas, profundidad 4) |
| Regularización L2        | 0.1                  | 2.0                            |
| n_jobs / nthread         | -1 (todos los cores) | 1 (un thread por modelo)       |
| Tiempo de entrenamiento  | Horas                | Minutos                        |
| Guardado en disco        | Siempre              | Con flag --guardar-modelos     |

