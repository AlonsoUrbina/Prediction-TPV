"""
Configuración global del proyecto.
Este archivo contiene todas las rutas, parámetros y configuraciones necesarias para el correcto
funcionamiento del proyecto. Aquí se definen las rutas para los datos y modelos, 
los parámetros para los modelos de machine learning, así como otras configuraciones relevantes.
De ser necesario, realizar los cambios en este archivo. En particular, cambiar:
- DIAS_PRED: número de días a predecir (horizonte de predicción).
- DIAS_BENCHMARK: número de días para test (benchmark).
- COLUMNS_TO_DROP: columnas a eliminar del dataset original (podría no ser necesario).
- COMERCIOS_A_MANTENER: lista de comercios a mantener en el dataset (Podrían eliminarse o 
agregarse modelos según sesgo).
- MAPPING_COMERCIO: diccionario para mapear RUTs a nombres de comercio.
- CYBER_EVENTS: lista de fechas de eventos cibernéticos (cyber events) a considerar en el análisis.
- LIGHTGBM_PARAMS_GLOBAL, CATBOOST_PARAMS_GLOBAL, XGBOOST_PARAMS_GLOBAL: diccionarios con los parámetros 
globales para cada modelo.
- LIGHTGBM_PARAMS_INDIVIDUAL, CATBOOST_PARAMS_INDIVIDUAL, XGBOOST_PARAMS_INDIVIDUAL: diccionarios 
con los parámetros individuales para cada modelo.
"""
from pathlib import Path

# === RUTAS DEL PROYECTO ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# === PARÁMETROS DEL MODELO ===
DIAS_PRED = 28  # Horizonte de predicción
# Consideración. Los días de benchmark son para medir errores generales del modelo.
# Para verificar qué pasaría realmente con el modelo en producción, lo ideal es medir en un
# único día, que debería ser el día siguiente al último día posible del dataset.
# Esto ya se contempla en el script 
# de backtesting, no es necesario poner 1.
DIAS_BENCHMARK = 7  # Días para test

# === COLUMNAS A ELIMINAR DEL DATASET ORIGINAL ===
COLUMNS_TO_DROP = ['id_sucursal', 'fecha_devolucion', 'tipo_tx', 'merchant_neto']

# === MAPEO RUT -> NOMBRE COMERCIO ===
MAPPING_COMERCIO = {
    '76516950-K': 'MERCPAGO',
    '76795561-8': 'HAULMER SPA',
    '76923783-6': 'DLOCAL',
    '99584660-8': 'PAYU',
    '76389778-8': 'EBANX CHILE S.A.',
    '76400485-K': 'FUDO',
    '77126383-6': 'TOKU SPA',
    '76211425-9': 'PEDIDOS YA',
    '76772379-2': 'PPRO',
    '77401986-3': 'PAGSEGURO',
    '77005329-3': 'NUVEI CHILE SPA',
    '77784488-1': 'PAYSCAN SPA',
    '77367796-4': 'INDRIVE',
    '76237019-0': 'CABIFY',
    '76478111-2': 'VIRTUALPOS',
    '76837223-3': 'RAPPI',
    '77101928-5': 'SUMUP CHILE PAYMENTS S.A',
    '77733347-K': 'RECIBOPAGOS SPA',
    '77427651-3': 'EBANX',
    '77125361-K': 'DELIVERY HERO STORES CHILE SPA',
    '78057888-2': 'COPEC',
    '77332933-8': 'CLPRO SPA',
    '97023000-9': 'ITAU CORPBANCA',
    '77152076-6': 'PAGSMILE',
    '76830014-3': 'FLOW S.A.',
    '76584147-K': 'PAYKU SPA'
}

# === FILTROS DE LIMPIEZA ===
MCC_INVALIDOS = [0, 34]          # MCCs a eliminar por ser inválidos
MIN_INSTANCIAS_COMERCIO = 100    # Mínimo de filas por comercio para conservarlo (poca data)

# === COMERCIOS A MANTENER ===
# Cambiar MODO_COMERCIOS para alternar entre los dos conjuntos sin comentar/descomentar.
#   'todos'   -> los 25 comercios del portfolio completo
#   'algunos' -> subconjunto optimo para el modelo global (menor sesgo)
MODO_COMERCIOS = 'algunos'  # <-- CAMBIAR AQUI: 'todos' o 'algunos'

COMERCIOS_TODOS = [
    'FLOW S.A.', 'PAYU', 'CABIFY', 'SUMUP CHILE PAYMENTS S.A',
    'CLPRO SPA', 'PAYSCAN SPA', 'RECIBOPAGOS SPA', 'MERCPAGO',
    'PEDIDOS YA', 'PAGSEGURO', 'FUDO', 'HAULMER SPA', 'DLOCAL',
    'RAPPI', 'EBANX CHILE S.A.', 'EBANX', 'DELIVERY HERO STORES CHILE SPA',
    'NUVEI CHILE SPA', 'TOKU SPA', 'ITAU CORPBANCA', 'PAGSMILE',
    'INDRIVE', 'PAYKU SPA', 'PPRO', 'VIRTUALPOS'
]

COMERCIOS_ALGUNOS = [
    'EBANX CHILE S.A.', 'PAYU', 'NUVEI CHILE SPA', 'DLOCAL', 'PEDIDOS YA',
    'CABIFY', 'MERCPAGO', 'TOKU SPA', 'FUDO', 'INDRIVE'
]

COMERCIOS_A_MANTENER = COMERCIOS_TODOS if MODO_COMERCIOS == 'todos' else COMERCIOS_ALGUNOS

# === CYBERDAYs ===
CYBER_EVENTS = [
    '2022-05-30', '2022-05-31', '2022-06-01',
    '2022-10-03', '2022-10-04', '2022-10-05', '2022-11-25',
    '2023-05-29', '2023-05-30', '2023-05-31',
    '2023-10-02', '2023-10-03', '2023-10-04', '2023-11-24',
    '2024-06-03', '2024-06-04', '2024-06-05',
    '2024-09-30', '2024-10-01', '2024-10-02', '2024-11-29',
    '2025-06-02', '2025-06-03', '2025-06-04',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-11-28',
    '2026-06-02', '2026-06-03', '2026-06-04',
    '2026-09-30', '2026-10-01', '2026-10-02', '2026-11-28'
]

# === ANTIGÜEDAD DE COMERCIOS ===
# Agregar de ser necesario a futuro, se puede usar la fecha de primera transacción para calcular la antigüedad de cada comercio.
COMERCIOS_ANTIGUEDAD = {
    'nombre_fantasia': [
        'FLOW S.A.', 'PAYU', 'CABIFY', 'SUMUP CHILE PAYMENTS S.A',
        'CLPRO SPA', 'PAYSCAN SPA', 'RECIBOPAGOS SPA', 'MERCPAGO',
        'PEDIDOS YA', 'PAGSEGURO', 'FUDO', 'HAULMER SPA', 'DLOCAL',
        'RAPPI', 'EBANX CHILE S.A.', 'EBANX', 'DELIVERY HERO STORES CHILE SPA',
        'NUVEI CHILE SPA', 'TOKU SPA', 'ITAU CORPBANCA', 'PAGSMILE',
        'INDRIVE', 'PAYKU SPA', 'PPRO', 'VIRTUALPOS'
    ],
    'fecha_primera_tx': [
        '28-03-2025', '08-03-2022', '18-09-2021', '05-10-2022',
        '31-01-2023', '01-11-2024', '03-10-2025', '11-11-2021',
        '30-09-2021', '07-12-2023', '19-12-2023', '14-03-2025',
        '24-04-2021', '19-03-2021', '23-07-2021', '17-03-2022',
        '17-08-2022', '08-10-2020', '19-12-2022', '13-03-2023',
        '02-08-2023', '23-10-2023', '08-05-2024', '14-02-2025', '02-10-2025'
    ]
}

# === PARÁMETROS MODELOS GLOBALES ===

# LightGBM Global
LIGHTGBM_PARAMS_GLOBAL = {
    'objective': 'tweedie',         # Intentar no cambiar, si se cambia que sea a regression.
    'tweedie_variance_power': 1.2,  # Mientras más cercano a 1.0 mejor para datos con muchos ceros, mientras más cercano a 2.0 mejor para datos con colas pesadas.
    'metric': 'rmse',               # RMSE es la métrica de evaluación, se puede cambiar a 'mae' o 'tweedie' si se desea.
    'num_leaves': 192,              # Mientras más alto mejor para capturar complejidad, pero más riesgo de overfitting. 256 es un buen punto de partida, considerar 128.
    'max_depth': -1,                # NO CAMBIAR
    'min_data_in_leaf': 100,        # Mientras más alto mejor para evitar overfitting, pero más riesgo de underfitting. 50 es un buen punto de partida, considerar 100.
    'lambda_l1': 1.0,               # NO CAMBIAR
    'lambda_l2': 3.0,               # NO CAMBIAR
    'min_gain_to_split': 0.1,       # Si se cambia que sea a 0.0 o 0.2.
    'learning_rate': 0.05,          # NO CAMBIAR
    'feature_fraction': 0.6,        # Mientras más bajo mejor para evitar overfitting, pero más riesgo de underfitting. Entre 0.3 a 0.8.
    'bagging_fraction': 0.8,        # Mientras más bajo mejor para evitar overfitting, pero más riesgo de underfitting. Entre 0.5 a 0.9.
    'bagging_freq': 1,              # NO CAMBIAR
    'boosting_type': 'gbdt',        # NO CAMBIAR
    'verbosity': -1,                # NO CAMBIAR
    'n_jobs': -1,                   # NO CAMBIAR
    'seed': 42,                     # NO CAMBIAR
    'force_col_wise': True          # NO CAMBIAR
}


# # CatBoost Global
CATBOOST_PARAMS_GLOBAL = {
    'loss_function': 'Tweedie:variance_power=1.1',  # Equivalente a tweedie_variance_power en LightGBM
    'eval_metric': 'RMSE',
    'iterations': 2000,                             # Equivalente a NUM_BOOST_ROUND_GLOBAL
    'learning_rate': 0.03,                          # NO CAMBIAR
    'depth': 6,                                     # Equivalente a profundidad del árbol (LightGBM usa num_leaves=128 ≈ depth 7-8)
    'l2_leaf_reg': 5.0,                             # Equivalente a lambda_l2 en LightGBM
    'min_data_in_leaf': 100,                        # Equivalente directo
    'subsample': 0.8,                               # Equivalente a bagging_fraction
    'rsm': 0.7,                                     # Equivalente a feature_fraction (random subspace method)
    'random_seed': 42,                              # NO CAMBIAR
    'verbose': False,                               # NO CAMBIAR
    'early_stopping_rounds': 200,                   # Equivalente a EARLY_STOPPING_ROUNDS_GLOBAL
    'task_type': 'CPU',                             # NO CAMBIAR
    'thread_count': -1,                             # NO CAMBIAR (equivalente a n_jobs)
    'bootstrap_type': 'Bernoulli',                  # Equivalente al bagging de LightGBM
    'allow_writing_files': False                    # NO CAMBIAR
}

# XGBoost Global
XGBOOST_PARAMS_GLOBAL = {
    'objective': 'reg:tweedie',                     # Intentar no cambiar
    'tweedie_variance_power': 1.1,                  # Equivalente directo a LightGBM
    'eval_metric': 'rmse',                          # RMSE es la métrica
    'max_depth': 6,                                 # Equivalente a num_leaves=128 en LightGBM (2^8=256)
    'learning_rate': 0.03,                          # NO CAMBIAR (equivalente a eta)
    'subsample': 0.8,                               # Equivalente a bagging_fraction
    'colsample_bytree': 0.7,                        # Equivalente a feature_fraction
    'min_child_weight': 100,                        # Equivalente a min_data_in_leaf (controla mínimo de muestras)
    'reg_alpha': 1.0,                               # Equivalente a lambda_l1
    'reg_lambda': 3.0,                              # Equivalente a lambda_l2
    'gamma': 0.1,                                   # Equivalente a min_gain_to_split
    'seed': 42,                                     # NO CAMBIAR
    'tree_method': 'hist',                          # NO CAMBIAR (más rápido)
    'verbosity': 0,                                 # NO CAMBIAR
    'nthread': -1                                   # NO CAMBIAR (equivalente a n_jobs)
}

NUM_BOOST_ROUND_GLOBAL = 3000
EARLY_STOPPING_ROUNDS_GLOBAL = 200


# === PARÁMETROS MODELOS INDIVIDUALES ===

# LightGBM Individual
LIGHTGBM_PARAMS_INDIVIDUAL = {
    'objective': 'tweedie', 
    'tweedie_variance_power': 1.2,
    'metric': 'rmse',
    'num_leaves': 15,
    'max_depth': 4,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.5,
    'lambda_l2': 3.0,
    'min_gain_to_split': 0.1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1,
    'n_jobs': 4,
    'seed': 42
}

# CatBoost Individual
CATBOOST_PARAMS_INDIVIDUAL = {
    'loss_function': 'Tweedie:variance_power=1.2',
    'eval_metric': 'RMSE',
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 2.0,
    'min_data_in_leaf': 20,
    'subsample': 0.8,
    'rsm': 0.8,
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'thread_count': 4,
    'bootstrap_type': 'Bernoulli'
}

# XGBoost Individual
XGBOOST_PARAMS_INDIVIDUAL = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.2,
    'eval_metric': 'rmse',
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 20,
    'reg_alpha': 0.5,
    'reg_lambda': 3.0,
    'gamma': 0.1,
    'seed': 42,
    'tree_method': 'hist',
    'verbosity': 0,
    'nthread': 4
}

NUM_BOOST_ROUND_INDIVIDUAL = 1000
EARLY_STOPPING_ROUNDS_INDIVIDUAL = 50

# === WARNINGS ===
SUPPRESS_WARNINGS = True # Si se desea suprimir warnings

# Crear carpetas si no existen
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
