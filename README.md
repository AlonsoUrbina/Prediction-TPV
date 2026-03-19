# Predicción de TPV para PSPs

> Sistema de Machine Learning para predecir el Volumen Total de Pagos (TPV) de comercios PSP (Proveedores de Servicios de Pago). Soporta múltiples algoritmos de gradient boosting, backtesting temporal estricto y generación automática de reportes PDF.

---

## Índice

- [¿Qué hace este proyecto?](#qué-hace-este-proyecto)
- [Metodología](#metodología)
- [Features del modelo](#features-del-modelo)
- [Modelos disponibles](#modelos-disponibles)
- [Instalación](#instalación)
- [Pipeline de uso](#pipeline-de-uso)
- [Backtesting y evaluación](#backtesting-y-evaluación)
- [Modelos individuales por comercio](#modelos-individuales-por-comercio)
- [Configuración avanzada](#configuración-avanzada)
- [Predicción con modelos entrenados](#predicción-con-modelos-entrenados)
- [Uso programático](#uso-programático)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Troubleshooting](#troubleshooting)
- [Licencia y contacto](#licencia-y-contacto)

---

## ¿Qué hace este proyecto?

Los PSPs procesan transacciones de múltiples comercios simultáneamente. Predecir cuánto va a transaccionar cada comercio en los próximos días permite anticipar flujos de pago, detectar desviaciones anormales respecto al comportamiento esperado y tomar decisiones operativas con antelación.

Este sistema toma datos históricos de transacciones diarias por comercio, construye features temporales y de comportamiento, y entrena modelos de gradient boosting para estimar el TPV agregado para un horizonte configurable de N días.

### Otras capacidades del sistema


| Capacidad                 | Descripción                                                                                    |
|---------------------------|------------------------------------------------------------------------------------------------|
| Backtesting temporal      | Evaluación en múltiples fechas de corte históricas, simulando condiciones reales de producción |
| Comparación de algoritmos | LightGBM, CatBoost y XGBoost bajo las mismas condiciones y features                            |
| Baseline de media móvil   | Modelo de referencia actual para cuantificar el beneficio real del ML                          |
| Reportes PDF              | Tablas de métricas y comparativas por comercio generadas automáticamente                       |


---

## Metodología

### Formulación del problema

El problema se formula como una **regresión supervisada sobre series de tiempo**. Los datos se dividen en cuatro conjuntos estrictamente ordenados en el tiempo:

1. **Entrenamiento**: todos los registros históricos hasta `fecha_corte - gap - dias_validacion`. Es la mayor parte de los datos y el modelo aprende sobre este conjunto.
2. **Validación**: período de N días posteriores al entrenamiento. Se usa para el **early stopping** del gradient boosting: el entrenamiento se detiene cuando el error de validación deja de mejorar, evitando el overfitting sin necesidad de fijar el número de árboles a mano.
2. **Gap**: período de `DIAS_PRED` días entre la validación y el conjunto de test. Es necesario porque los features de lag miran hacia atrás hasta `DIAS_PRED` días. Esto sirve únicamente para el testeo de los modelos, es decir, backtesting. Para entrenamientos formales puede ser útil quitar esto.
4. **Test**: los `DIAS_BENCHMARK` días inmediatamente posteriores a `fecha_corte`. Es el conjunto de evaluación final sobre el cual se reportan las métricas.


Este diseño garantiza que el modelo nunca accede a datos futuros en el entrenamiento y que el error reportado es una estimación real del rendimiento.

### Función de regresión: Tweedie

Los tres algoritmos usan distribuciones **Tweedie**. Esto es apropiado porque:

- El TPV es siempre no-negativo (la distribución Tweedie respeta este límite naturalmente)
- La distribución del TPV es **asimétrica con cola derecha pesada**: eventos como CyberDay o Black Friday generan peaks de volumen muy superiores a los días ordinarios
- La pérdida Tweedie penaliza los errores en escala proporcional, más adecuado que el error cuadrático cuando los volúmenes varían órdenes de magnitud entre comercios

El parámetro `tweedie_variance_power` controla el balance entre distribuciones:
- **1.0** → comportamiento Poisson (adecuado cuando hay días con cero transacciones)
- **2.0** → comportamiento Gamma (adecuado para distribuciones con colas muy pesadas)

### Backtesting temporal estricto

```
Corte 1:  |--- Train 1 ---|  Test  |
Corte 2:  |----- Train 2 -----|  Test  |
Corte 3:  |------- Train 3 -------|  Test  |

Cada fecha de corte agrega más datos al entrenamiento.
El test tiene siempre DIAS_BENCHMARK días inmediatamente despues del corte.
```

Evaluar en múltiples fechas de corte permite estimar la **variabilidad del error** y detectar si el modelo se degrada en ciertos períodos (por ejemplo, meses con eventos especiales que generan distribuciones de TPV muy distintas a las de entrenamiento).

---

## Features del modelo

El módulo `src/features/feature_engineering.py` genera todas las features a partir de los datos históricos. Todas se calculan **usando solo información disponible antes de la fecha de corte**, evitando data leakage.

### Features temporales cíclicas

| Feature                            | Descripción                       |
|------------------------------------|-----------------------------------|
| `mes_sin`, `mes_cos`               | Ciclicidad del mes del año        |
| `dia_semana_sin`, `dia_semana_cos` | Ciclicidad del día de la semana   |
| `dia_mes_sin`, `dia_mes_cos`       | Ciclicidad del día dentro del mes |
| `dias_rest_mes_actual`             | Días restantes en el mes actual   |

> **Nota sobre encoding cíclico:** El día 1 y el día 7 de la semana son adyacentes en el tiempo, pero un encoding lineal no captura esta relación. El encoding senoidal (sin/cos) proyecta el tiempo en un círculo, preservando la continuidad cíclica.

### Features de eventos futuros

| Feature              | Descripción                                                                       |
|----------------------|-----------------------------------------------------------------------------------|
| `q_feriados_futuros` | Cantidad de feriados chilenos en los próximos N días                              |
| `flag_cyber_futuro`  | Flag: hay algún Cyber Event en los próximos N días (configurables en `config.py`) |

### Features de TPV rolling

| Feature               | Descripción                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `tpv_acumulado_7d`    | Suma de TPV de los últimos 7 días                                                             |
| `tpv_acumulado_15d`   | Suma de TPV de los últimos 15 días                                                            |
| `tpv_acumulado_30d`   | Suma de TPV de los últimos 30 días                                                            |
| `tpv_acumulado_Xd`    | Suma de TPV de los últimos N días (horizonte de predicción)                                   |
| `media_movil_7d`      | Media de TPV de los últimos 7 días                                                            |
| `media_movil_30d`     | Media de TPV de los últimos 30 días                                                           |
| `volatilidad_tpv_30d` | Desviación estándar del TPV en los últimos 30 días                                            |
| `aceleracion_tpv`     | Ratio `media_movil_7d / (media_movil_30d + 1)`: captura aceleración o desaceleración reciente |

### Features de transacciones rolling

| Feature               | Descripción                                         |
|-----------------------|-----------------------------------------------------|
| `cantidad_tx_7d`      | Suma de transacciones de los últimos 7 días         |
| `cantidad_tx_30d`     | Suma de transacciones de los últimos 30 días        |
| `media_tx_7d`         | Media de transacciones de los últimos 7 días        |
| `media_tx_30d`        | Media de transacciones de los últimos 30 días       |
| `aceleracion_tx`      | Ratio `media_tx_7d / (media_tx_30d + 1)`            |
| `ticket_promedio_7d`  | TPV promedio por transacción en los últimos 7 días  |
| `ticket_promedio_30d` | TPV promedio por transacción en los últimos 30 días |

### Features de mix de pago (% últimos 30 días)

| Feature                 | Descripción                                    |
|-------------------------|------------------------------------------------|
| `pct_debito_30d`        | Proporción del TPV en tarjetas de débito       |
| `pct_credito_30d`       | Proporción del TPV en tarjetas de crédito      |
| `pct_visa_30d`          | Proporción del TPV en red Visa                 |
| `pct_mastercard_30d`    | Proporción del TPV en red Mastercard           |
| `pct_internacional_30d` | Proporción del TPV de tarjetas internacionales |
| `pct_presente_30d`      | Proporción del TPV con tarjeta presente        |

### Features de comercio

| Feature              | Descripción                                                          |
|----------------------|----------------------------------------------------------------------|
| `id_comercio_num`    | ID numérico del comercio (label encoding de `nombre_comercio`)       |
| `mcc`                | Código de categoría del comercio (MCC)                               |
| `antiguedad_meses`   | Meses desde la primera transacción del comercio                      |

### Feature de referencia histórica anual

| Feature                    | Descripción                                                                    |
|----------------------------|--------------------------------------------------------------------------------|
| `tpv_futuro_año_anterior`  | TPV futuro del mismo período hace exactamente 365 días; captura estacionalidad |

---

## Modelos disponibles

Los tres algoritmos comparten los mismos features, la misma función de pérdida (Tweedie) y la misma interfaz de entrenamiento, lo que hace su comparación directa y justa.

### Comparativa

| Aspecto                      | LightGBM              | CatBoost                       | XGBoost                       |
|------------------------------|-----------------------|--------------------------------|-------------------------------|
| Velocidad de entrenamiento   | Muy alta              | Media                          | Alta                          |
| Uso de memoria               | Bajo                  | Medio                          | Medio                         |
| Precisión sin tuning         | Buena                 | Muy buena                      | Buena                         |
| Robustez a overfitting       | Alta                  | Muy alta                       | Alta                          |
| Parámetros a ajustar         | Muchos                | Pocos                          | Muchos                        |
| Formato del modelo guardado  | `.txt`                | `.cbm`                         | `.json`                       |
| Implementación               | `src/models/train.py` | `src/models/train_catboost.py` | `src/models/train_xgboost.py` |

### LightGBM

Usa crecimiento de árbol hoja a hoja (leaf-wise), lo que lo hace más eficiente que los métodos nivel a nivel en datasets grandes. Su bajo consumo de memoria lo hace útil para iteraciones rápidas de experimentación.

### CatBoost

Usa _ordered boosting_: construye los árboles evaluando cada muestra con un modelo entrenado solo con las muestras anteriores, lo que reduce el sesgo y permite obtener buena precisión con menos tuning.

### XGBoost

Buen balance entre velocidad y precisión, con amplia documentación y comunidad. Útil como referencia comparativa.

---

## Instalación

### Requisitos previos

- Python 3.8 o superior
- Archivo de datos de entrada: `base_proyecto_ml_v2.parquet`
- ~4 GB de RAM recomendados para el dataset completo

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd tpv-prediction

# 2a. Entorno virtual con pip (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 2b. Alternativa: entorno conda
conda env create -f environment.yml
conda activate tpv-prediction

# 3. Colocar los datos en la carpeta data/
cp /ruta/a/base_proyecto_ml_v2.parquet data/
```

---

## Pipeline de uso

El flujo de trabajo completo se divide en 5 pasos secuenciales. Cada paso genera un artefacto que es la entrada del siguiente.

```
base_proyecto_ml_v2.parquet
     |
     v
[1] run_data_processing.py    ->  dataset_inicial_limpio_{modo}.parquet
     |
     v
[2] run_dataset_generation.py ->  dataset_entrenamiento_28dias_{modo}.parquet
     |
     v
[3] run_backtesting.py        ->  reportes PDF en results/
     |
     v
[4] run_training.py           ->  modelo.{txt|cbm|json}
     |
     v (cuando hay data nueva)
[5] run_prediction.py         ->  data/predicciones/ (CSV + PDF)
```

### Paso 1 — Procesamiento de datos

```bash
python scripts/run_data_processing.py
```

Lee `base_proyecto_ml_v2.parquet`, aplica filtros de calidad (MCCs inválidos, comercios con datos insuficientes), mapea RUTs a nombres de comercio y guarda el dataset limpio en `data/`.

### Paso 2 — Generación de features

```bash
python scripts/run_dataset_generation.py --dias-pred 28
```

Construye el dataset de entrenamiento con todas las features. El parámetro `--dias-pred` define el horizonte de predicción. Genera `dataset_entrenamiento_28dias_{modo}.parquet` (donde `{modo}` es el valor de `MODO_COMERCIOS` en `config.py`).

### Paso 3 — Entrenamiento

```bash
# Entrenar con un algoritmo especifico
python scripts/run_training.py --model-type lightgbm
python scripts/run_training.py --model-type catboost
python scripts/run_training.py --model-type xgboost

# Entrenar y comparar los tres simultaneamente
python scripts/compare_models.py
```

Los modelos se guardan en `models/` con el formato `{algoritmo}_{modo}_{fecha_corte}_{horizonte}.{ext}`. Junto con cada modelo se generan automáticamente un **CSV** con la importancia porcentual de todas las variables (`*_feature_importance.csv`) y un **PDF** con las 10 más importantes (`*_feature_importance.pdf`).

### Paso 4 — Backtesting

```bash
python scripts/run_backtesting.py `
    --model-type catboost `
    --modo global `
    --fechas 2025-07-01 2025-08-01 2025-09-01 `
             2025-10-01 2025-11-01 2025-12-01
```

Genera tres reportes PDF en `results/`:

| Archivo                                      | Contenido                                                      |
|----------------------------------------------|----------------------------------------------------------------|
| `*_metricas_28dias.pdf`                      | Resumen de MAE, RMSE, MAPE y R² por comercio y fecha de corte  |
| `*_por_comercio_porcentual_28dias.pdf`       | Error porcentual por comercio                                  |
| `*_por_comercio_miles_millones_28dias.pdf`   | Error absoluto en miles de millones                            |

### Usando Makefile

```bash
make data       # Paso 1
make dataset    # Paso 2 (28 dias por defecto)
make train      # Paso 3 con LightGBM
make backtest   # Paso 4
make help       # Ver todos los comandos disponibles
```

---
## Backtesting y evaluación

### Métricas reportadas

| Métrica  | Descripción                                                                                |
|----------|--------------------------------------------------------------------------------------------|
| **MAE**  | Error absoluto promedio, en la misma unidad que el TPV                                     |
| **RMSE** | Penaliza más los errores grandes; sensible a predicciones muy erróneas                     |
| **MAPE** | Error porcentual promedio; permite comparar entre comercios de distinto tamaño             |
| **R²**   | Proporción de la varianza explicada por el modelo (1.0 = perfecto, <0 = peor que la media) |

### Comparación con baseline

Para cuantificar la ganancia real del ML sobre un método simple, se puede evaluar el baseline de media móvil:

```bash
python scripts/evaluar_media_movil.py `
    --fechas 2025-10-01 2025-11-01 2025-12-01
```

Esto genera métricas del modelo de referencia (predecir que el TPV futuro = promedio de los últimos N días) bajo las mismas condiciones de evaluación que el modelo de ML.

---

## Modelos individuales por comercio

Además del modelo global, el sistema permite entrenar un modelo independiente para cada comercio. Esto es útil cuando:

- Los comercios tienen patrones de comportamiento muy distintos entre sí
- Se quiere máxima precisión para comercios con suficientes datos históricos
- El modelo global introduce demasiado ruido por la heterogeneidad entre comercios

Considerar que es necesaria una gran cantidad de data y un comportamiento razonable por parte del comercio para que este tipo de modelos obtengan resultados favorables. Realizar uso responsable.

```bash
python scripts/run_individual.py `
    --model-type catboost `
    --fecha-corte 2025-11-01 `
    --guardar-modelos
```

El script itera sobre los comercios en `COMERCIOS_A_MANTENER`, entrena un modelo independiente para cada uno y genera métricas individuales. Con `--guardar-modelos`, los modelos se persisten en `models/`.

---

## Configuración avanzada

Todos los parámetros del sistema se centralizan en `config/config.py`. Los cambios en este archivo afectan a todos los scripts sin necesidad de tocar el código fuente.

### Parámetros principales

```python
# Horizonte de prediccion (dias hacia adelante)
DIAS_PRED = 28

# Tamano del conjunto de test en backtesting
DIAS_BENCHMARK = 7

# Seleccion de comercios incluidos en el modelo global.
# Cambiar solo esta linea para alternar entre los dos conjuntos predefinidos:
# 'todos'   -> COMERCIOS_TODOS (todos los comercios disponibles)
# 'algunos' -> COMERCIOS_ALGUNOS (subconjunto reducido para modelos parciales)
MODO_COMERCIOS = 'todos'

# Fechas de eventos especiales (CyberDay, Black Friday, etc.)
# Usadas para construir la feature es_cyber_event.
CYBER_EVENTS = ['2025-06-02', '2025-06-03', '2025-11-28', ...]
```

### Parámetros de los algoritmos

Los diccionarios `LIGHTGBM_PARAMS_GLOBAL`, `CATBOOST_PARAMS_GLOBAL` y `XGBOOST_PARAMS_GLOBAL` (en `config/config.py`) permiten ajustar el comportamiento de cada modelo:

| Parámetro                      | Efecto                                                                                    | Valor por defecto |
|--------------------------------|-------------------------------------------------------------------------------------------|-------------------|
| `num_leaves` (LightGBM)        | Complejidad del modelo; valores altos capturan más patrones pero riesgo de overfitting    | 192               |
| `depth` (CatBoost/XGBoost)     | Profundidad máxima del árbol; equivalente a `num_leaves`                                  | 6                 |
| `learning_rate`                | Tamaño del paso en el descenso de gradiente; valores bajos requieren más iteraciones      | 0.03              |
| `min_data_in_leaf`             | Mínimo de muestras en una hoja; valores altos reducen el overfitting                      | 100               |
| `feature_fraction`             | Fracción de features usadas por árbol; reduce correlación entre árboles                   | 0.6               |
| `lambda_l1` / `lambda_l2`      | Regularización L1/L2 para penalizar pesos grandes                                         | 1.0 / 3.0         |


---

## Predicción con modelos entrenados

Una vez entrenado el modelo, para generar predicciones sobre **datos nuevos** se usa `run_prediction.py`. Este script genera los features internamente desde `dataset_inicial_limpio_{modo}.parquet` (sin necesidad de correr `run_dataset_generation.py`) y guarda el resultado como CSV y PDF.

### Flujo para predecir con data nueva

```bash
# Paso 1: Actualizar la data cruda con las transacciones mas recientes
# (reemplazar data/base_proyecto_ml_v2.parquet con el parquet actualizado)

# Paso 2: Reprocesar datos
python scripts/run_data_processing.py

# Paso 3: Predecir
python scripts/run_prediction.py --model-path models/catboost_global_todos_2025-12-04_28dias.cbm --fecha-corte 2026-03-05
```

Si no se pasa `--fecha-corte`, se usa automaticamente la ultima fecha disponible en el dataset (lo cual simplifica la implementación).

### Argumentos de run_prediction.py

| Argumento              | Default                    | Descripción                                                                                |
|------------------------|----------------------------|--------------------------------------------------------------------------------------------|
| `--model-path`         | (requerido)                | Ruta al modelo (.cbm, .txt o .json)                                                        |
| `--fecha-corte`        | última fecha disponible    | Fecha de inicio de la ventana de predicción (YYYY-MM-DD)                                   |
| `--dias-pred`          | `DIAS_PRED` de config      | Horizonte en días                                                                          |
| `--guardar-features`   | False                      | Si se pasa, guarda el dataset de features en `data/features_prediccion_{dias}dias.parquet` |

### Salida

Los resultados se guardan en `data/predicciones/` con el patrón `predicciones_{fecha}_{nombre_modelo}.{csv|pdf}`.

El CSV tiene una fila por comercio: `nombre_comercio`, `tpv_predicho`. Las fechas de la ventana de prediccion se incluyen en el titulo del PDF y en el nombre del archivo. El PDF es una tabla formateada con los mismos datos mas una fila de TOTAL.

> **Nota tecnica:** `run_prediction.py` llama a `generar_dataset(..., para_prediccion=True)`, lo que evita que se descarten los rows mas recientes (que tienen `tpv_futuro=NaN`). Las predicciones se agregan sumando por comercio, ya que el dataset tiene un row por combinacion `(comercio, MCC)`.

---
## Uso programático

### Cargar un modelo guardado y predecir

```python
from src.models.predict import cargar_modelo, predecir
from src.data.loader import load_training_dataset

# Cargar dataset de entrenamiento
df, encoder = load_training_dataset('data/dataset_entrenamiento_28dias_todos.parquet')

# Cargar modelo previamente entrenado
modelo = cargar_modelo('models/catboost_global_todos_2025-11-01_28dias.cbm')

# Generar predicciones
predicciones = predecir(modelo, df)
```

### Entrenar un modelo desde Python

```python
from src.models.train_catboost import entrenar_modelo_catboost, guardar_modelo_catboost
from src.data.loader import load_training_dataset

df, _ = load_training_dataset('data/dataset_entrenamiento_28dias_todos.parquet')

modelo, metricas = entrenar_modelo_catboost(
    df,
    fecha_corte='2025-11-01',
    dias_pred=28,
    dias_benchmark=7
)

guardar_modelo_catboost(modelo, 'models/mi_modelo.cbm')
print(metricas)
```

---

## Estructura del proyecto

```
tpv-prediction/
├── config/
│   └── config.py                      # Configuracion global: parametros, rutas, comercios, eventos
├── src/
│   ├── data/
│   │   ├── loader.py                  # Carga de datasets procesados
│   │   └── preprocessing.py           # Limpieza y transformacion de datos crudos
│   ├── features/
│   │   └── feature_engineering.py     # Construccion de los 60+ features
│   ├── models/
│   │   ├── common.py                  # Utilidades compartidas (preparar_datos, guardar_importancia_variables, ...)
│   │   ├── train.py                   # Entrenamiento LightGBM
│   │   ├── train_catboost.py          # Entrenamiento CatBoost
│   │   ├── train_xgboost.py           # Entrenamiento XGBoost
│   │   ├── train_individual.py        # Modelos por comercio
│   │   ├── backtesting.py             # Logica de evaluacion temporal
│   │   └── predict.py                 # Carga de modelos y prediccion
│   └── utils/
│       └── helpers.py                 # Funciones auxiliares
├── scripts/
│   ├── run_data_processing.py         # Paso 1: Limpieza de datos crudos
│   ├── run_dataset_generation.py      # Paso 2: Generacion de features
│   ├── run_training.py                # Paso 3: Entrenamiento del modelo global
│   ├── run_backtesting.py             # Paso 4: Backtesting y generacion de PDFs
│   ├── run_individual.py              # Entrenamiento de modelos por comercio
│   ├── compare_models.py              # Comparacion de los tres algoritmos
│   ├── evaluar_media_movil.py         # Evaluacion del baseline
│   └── run_prediction.py              # Prediccion con data nueva (CSV + PDF)
├── data/                              # Datos de entrada (no versionados)
├── models/                            # Modelos entrenados (no versionados)
├── results/                           # Reportes PDF generados (no versionados)
├── notebooks/
│   └── Data.ipynb                     # Notebook de exploracion de datos
├── .gitignore
├── requirements.txt
├── environment.yml
├── Makefile
└── README.md
```

> Las carpetas `data/` y `models/` contienen archivos `.gitkeep` para que git las rastree aunque estén vacías. Los datos, modelos y reportes generados están en `.gitignore`.

---
## Troubleshooting

### No se encuentra el dataset de entrenamiento

**Error:** `FileNotFoundError: dataset_entrenamiento_*.parquet`

**Causa:** El dataset de features no ha sido generado todavía.

```bash
python scripts/run_dataset_generation.py --dias-pred 28
```

### Dataset insuficiente para la fecha de corte

**Error:** `Dataset insuficiente` o conjunto de test vacío.

**Causa:** La `fecha_corte` elegida más los `DIAS_BENCHMARK` días de test supera el último día disponible en el dataset.

**Solución:** Usar una `fecha_corte` más temprana que deje al menos `DIAS_BENCHMARK` días de datos después.

### Dependencias faltantes

```bash
pip install lightgbm catboost xgboost
# O reinstalar el entorno completo:
pip install -r requirements.txt
```

### Entrenamiento muy lento

Para usar GPU si está disponible:

```python
# En config/config.py:
CATBOOST_PARAMS_GLOBAL['task_type'] = 'GPU'
LIGHTGBM_PARAMS_GLOBAL['device'] = 'gpu'
```

### Overfitting (R² alto en train, bajo en test)

Ajustar en `config/config.py`:

1. Reducir `num_leaves` (LightGBM) o `depth` (CatBoost/XGBoost)
2. Aumentar `lambda_l1` y `lambda_l2` (regularización)
3. Reducir `feature_fraction` (ej: 0.5)
4. Aumentar `min_data_in_leaf` (ej: 150)

### Prediccion: no se encuentra dataset_inicial_limpio.parquet

**Error:** `Error: No se encontro dataset_inicial_limpio.parquet`

**Causa:** No se ha procesado la data cruda todavia.

```bash
python scripts/run_data_processing.py
```

### Sobre `fecha_corte`

`fecha_corte` es el **primer día del conjunto de test**, no el último día de entrenamiento. Los `DIAS_BENCHMARK` días de evaluación se cuentan hacia adelante desde ese punto:

```bash
--fecha-corte 2025-10-01 --dias-benchmark 7
# Train: todos los datos anteriores a 2025-10-01
# Test:  2025-10-01 al 2025-10-07 (7 dias)
```

---

## Licencia y contacto

Este proyecto está bajo la **Licencia MIT**. Ver el archivo `LICENSE` para más detalles.

**Autor:** Alonso Urbina
**Email:** alonso.urbina@ug.uchile.cl

---

*Última actualización: Marzo 2026*

