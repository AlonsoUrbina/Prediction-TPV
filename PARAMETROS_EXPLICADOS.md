# Equivalencia de Parámetros entre Modelos

Los valores que aparecen a continuación son los configurados en `config/config.py`. Existen dos conjuntos: **globales** (un modelo para todos los comercios) e **individuales** (un modelo por comercio, más simple para evitar overfitting con menos datos).

---

## Parámetros Clave y su Importancia

### 1. **Objective: Tweedie**

```python
# LightGBM Global
'objective': 'tweedie'
'tweedie_variance_power': 1.2

# CatBoost Global
'loss_function': 'Tweedie:variance_power=1.1'

# XGBoost Global
'objective': 'reg:tweedie'
'tweedie_variance_power': 1.1
```

**¿Por qué Tweedie?**

- TPV es siempre positivo (Tweedie respeta este límite)
- Distribución asimétrica con colas pesadas (CyberDay, Black Friday generan peaks)
- `variance_power` entre 1.0 y 2.0: 1.0 = Poisson (muchos ceros), 2.0 = Gamma (colas muy pesadas)
- Los valores 1.1-1.2 son un balance apropiado para TPV monetario

---

### 2. **Complejidad del árbol**

```python
# LightGBM Global
'num_leaves': 192   # hojas por árbol (leaf-wise)
'max_depth': -1     # sin límite de profundidad

# CatBoost Global
'depth': 6          # nivel a nivel, aprox 2^6 = 64 hojas

# XGBoost Global
'max_depth': 6      # misma lógica que CatBoost
```

LightGBM crece hoja a hoja (leaf-wise) -> `num_leaves` es el control principal.
CatBoost y XGBoost crecen nivel a nivel -> `depth` controla la profundidad.

**Modelos individuales** (menos datos por comercio):
```python
'num_leaves': 15   # LightGBM Individual
'max_depth': 4     # CatBoost / XGBoost Individual
```

---

### 3. **Learning rate**

```python
# LightGBM Global
'learning_rate': 0.05

# CatBoost Global / XGBoost Global
'learning_rate': 0.03

# Todos los individuales
'learning_rate': 0.03
```

Learning rate bajo -> el modelo aprende más despacio pero generaliza mejor. Compensa con más iteraciones (`NUM_BOOST_ROUND_GLOBAL = 3000`, `iterations = 2000`).

---

### 4. **Regularización L1 y L2**

```python
# LightGBM Global
'lambda_l1': 1.0
'lambda_l2': 3.0

# CatBoost Global (solo L2; ordered boosting ya regulariza internamente)
'l2_leaf_reg': 5.0

# XGBoost Global
'reg_alpha': 1.0    # L1
'reg_lambda': 3.0   # L2
```

- **L1**: puede llevar pesos a 0 (feature selection implícita)
- **L2**: reduce magnitud de pesos sin zeroing (preferido para suavizar)
- CatBoost usa mayor l2 (5.0) porque su ordered boosting ya reduce sesgo

**Individuales** (más regularización porque hay menos datos):
```python
'lambda_l1': 0.5, 'lambda_l2': 3.0   # LightGBM
'l2_leaf_reg': 2.0                     # CatBoost
'reg_alpha': 0.5, 'reg_lambda': 3.0   # XGBoost
```

---

### 5. **min_data_in_leaf / min_child_weight**

```python
# Globales (dataset grande, muchos datos por hoja posibles)
'min_data_in_leaf': 100   # LightGBM / CatBoost
'min_child_weight': 100   # XGBoost

# Individuales (dataset por comercio es mucho menor)
'min_data_in_leaf': 20   # LightGBM / CatBoost
'min_child_weight': 20   # XGBoost
```

Evita hojas con muy pocos datos -> previene overfitting y ruido.

---

### 6. **Bagging (feature_fraction y bagging_fraction)**

```python
# LightGBM Global
'feature_fraction': 0.6    # 60% de features por árbol
'bagging_fraction': 0.8    # 80% de datos por árbol
'bagging_freq': 1

# CatBoost Global
'rsm': 0.7                 # Random Subspace Method (equivalente a feature_fraction)
'subsample': 0.8
'bootstrap_type': 'Bernoulli'

# XGBoost Global
'colsample_bytree': 0.7   # equivalente a feature_fraction
'subsample': 0.8

# Individuales (menos agresivo porque hay menos datos)
'feature_fraction': 0.8, 'bagging_fraction': 0.8   # LightGBM
'rsm': 0.8, 'subsample': 0.8                        # CatBoost
'colsample_bytree': 0.8, 'subsample': 0.8           # XGBoost
```

Cada árbol ve subconjuntos distintos de features y datos. Reduce correlación entre árboles y el modelo generaliza mejor.

---

### 7. **Iteraciones y early stopping**

```python
# Globales
NUM_BOOST_ROUND_GLOBAL = 3000          # máximo de árboles
EARLY_STOPPING_ROUNDS_GLOBAL = 200     # parar si no mejora en 200 rondas

# CatBoost Global (en el dict de parámetros)
'iterations': 2000
'early_stopping_rounds': 200

# Individuales
NUM_BOOST_ROUND_INDIVIDUAL = 1000
EARLY_STOPPING_ROUNDS_INDIVIDUAL = 50
```

Con early stopping el modelo se detiene antes si la métrica de validación no mejora.

---

### 8. **min_gain_to_split / gamma**

```python
# LightGBM Global
'min_gain_to_split': 0.1

# XGBoost Global (equivalente directo)
'gamma': 0.1
```

Umbral mínimo de ganancia para hacer un split. Previene splits con poca información real.

---

## Tabla de Equivalencias Completa (Modelos Globales)


| Parámetro | LightGBM | CatBoost | XGBoost |
|-----------|----------|----------|---------|
| **Función objetivo** | `objective: tweedie` | `loss_function: Tweedie` | `objective: reg:tweedie` |
| **Tweedie power** | `tweedie_variance_power: 1.2` | `variance_power=1.1` | `tweedie_variance_power: 1.1` |
| **Complejidad** | `num_leaves: 192` | `depth: 6` | `max_depth: 6` |
| **Learning rate** | `learning_rate: 0.05` | `learning_rate: 0.03` | `learning_rate: 0.03` |
| **Regularización L2** | `lambda_l2: 3.0` | `l2_leaf_reg: 5.0` | `reg_lambda: 3.0` |
| **Regularización L1** | `lambda_l1: 1.0` | -- | `reg_alpha: 1.0` |
| **Min datos hoja** | `min_data_in_leaf: 100` | `min_data_in_leaf: 100` | `min_child_weight: 100` |
| **Feature sampling** | `feature_fraction: 0.6` | `rsm: 0.7` | `colsample_bytree: 0.7` |
| **Data sampling** | `bagging_fraction: 0.8` | `subsample: 0.8` | `subsample: 0.8` |
| **Iteraciones max** | `NUM_BOOST_ROUND: 3000` | `iterations: 2000` | `NUM_BOOST_ROUND: 3000` |
| **Early stopping** | `200 rounds` | `early_stopping_rounds: 200` | `200 rounds` |
| **Min gain/split** | `min_gain_to_split: 0.1` | -- | `gamma: 0.1` |


---

## Tabla de Equivalencias (Modelos Individuales)

| Parámetro | LightGBM | CatBoost | XGBoost |
|-----------|----------|----------|---------|
| **Tweedie power** | `1.2` | `1.2` | `1.2` |
| **Complejidad** | `num_leaves: 15` | `depth: 4` | `max_depth: 4` |
| **Learning rate** | `0.03` | `0.03` | `0.03` |
| **Regularización L2** | `lambda_l2: 3.0` | `l2_leaf_reg: 2.0` | `reg_lambda: 3.0` |
| **Regularización L1** | `lambda_l1: 0.5` | -- | `reg_alpha: 0.5` |
| **Min datos hoja** | `min_data_in_leaf: 20` | `min_data_in_leaf: 20` | `min_child_weight: 20` |
| **Feature sampling** | `feature_fraction: 0.8` | `rsm: 0.8` | `colsample_bytree: 0.8` |
| **Data sampling** | `bagging_fraction: 0.8` | `subsample: 0.8` | `subsample: 0.8` |
| **Iteraciones max** | `NUM_BOOST_ROUND: 1000` | `iterations: 1000` | `NUM_BOOST_ROUND: 1000` |
| **Early stopping** | `50 rounds` | `early_stopping_rounds: 50` | `50 rounds` |

---

## Por qué estos parámetros para TPV

### Características del problema:

1. **Valores continuos positivos** -> Tweedie con variance_power 1.1-1.2
2. **Gran rango de valores** (miles a millones) -> regularización moderada-alta
3. **Dataset grande (globales)** -> alta complejidad (`num_leaves=192`, `depth=6`)
4. **Dataset pequeño (individuales)** -> complejidad reducida (`num_leaves=15`, `depth=4`)
5. **Muchas features** -> bagging de features (0.6-0.7 globales, 0.8 individuales)
6. **Convergencia estable** -> learning rate bajo con muchas iteraciones

### Lo que NO deberías cambiar:

- `objective / loss_function` -- Tweedie es específico para TPV
- `bagging_freq: 1` -- necesario para bagging en LightGBM
- `bootstrap_type: Bernoulli` -- necesario para subsample en CatBoost
- `tree_method: hist` -- método rápido de XGBoost
- Seeds -- para reproducibilidad

### Lo que SÍ puedes experimentar:

- `learning_rate` -- 0.01 para más precisión, 0.1 para más velocidad
- `num_leaves / depth` -- subir para más complejidad, bajar si hay overfitting
- `lambda_l1 / lambda_l2` -- subir si hay overfitting, bajar si hay underfitting
- `feature_fraction / bagging_fraction` -- bajar si hay overfitting
- `tweedie_variance_power` -- 1.0 si hay muchos ceros, 1.5+ si colas muy pesadas

---

## Cómo optimizar parámetros

```python
import optuna
from src.models.train_catboost import entrenar_modelo_catboost

def objective(trial):
    params = {
        'loss_function': 'Tweedie:variance_power=1.1',  # NO cambiar
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),
    }
    modelo, metricas = entrenar_modelo_catboost(df, '2026-01-01', params=params)
    return metricas['rmse']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Mejores parámetros:', study.best_params)
```

---

## Casos Especiales

### Si tienes overfitting:

```python
'lambda_l1': 2.0,              # subir
'lambda_l2': 5.0,              # subir
'min_data_in_leaf': 200,       # subir
'feature_fraction': 0.5,       # bajar
'bagging_fraction': 0.7,       # bajar
'num_leaves': 128,             # bajar
```

### Si tienes underfitting:

```python
'num_leaves': 256,             # subir
'max_depth': -1,               # sin límite (LightGBM)
'min_data_in_leaf': 50,        # bajar
'lambda_l1': 0.0,              # bajar
'lambda_l2': 1.0,              # bajar
'learning_rate': 0.05,         # subir
```

---

## Referencias

- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost Parameters](https://catboost.ai/en/docs/references/training-parameters/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Tweedie Distribution](https://en.wikipedia.org/wiki/Tweedie_distribution)

