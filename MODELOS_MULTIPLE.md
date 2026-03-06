#  Guía de Modelos: LightGBM, CatBoost y XGBoost

##  Resumen

El repositorio soporta **3 algoritmos de gradient boosting**:

| Modelo    | Velocidad   | Precisión   | Uso GPU | Mejor para                        |
|-----------|-------------|-------------|---------|-----------------------------------|
| LightGBM  | Muy rápido  | Buena       | Sí      | Datasets grandes, rapidez         |
| CatBoost  | Rápido      | Muy buena   | Sí      | Variables categóricas, robustez   |
| XGBoost   | Normal      | Buena       | Sí      | Balance, estabilidad              |

---

##  Uso Rápido

### Entrenar un modelo específico

```bash
# LightGBM (default)
python scripts/run_training.py --model-type lightgbm

# CatBoost
python scripts/run_training.py --model-type catboost

# XGBoost
python scripts/run_training.py --model-type xgboost
```

### Entrenar los 3 modelos y comparar

```bash
# Entrenar todos
python scripts/run_training.py --model-type all

# O usar script de comparación
python scripts/compare_models.py
```

### Predicción con modelo específico

```python
# LightGBM
from src.models.predict import cargar_modelo, predecir
modelo = cargar_modelo('models/lgbm_global_todos_2026-01-01_28dias.txt')
preds = predecir(modelo, df)

# CatBoost
from src.models.train_catboost import cargar_modelo_catboost, predecir_catboost
modelo = cargar_modelo_catboost('models/catboost_global_todos_2026-01-01_28dias.cbm')
preds = predecir_catboost(modelo, df)

# XGBoost
from src.models.train_xgboost import cargar_modelo_xgboost, predecir_xgboost
modelo = cargar_modelo_xgboost('models/xgboost_global_todos_2026-01-01_28dias.json')
preds = predecir_xgboost(modelo, df)
```

---

##  Comparación Detallada

### LightGBM

**Ventajas:**
-  **Más rápido** para datasets grandes
-  Menor uso de memoria
-  Excelente para datos densos
-  Altamente configurable

**Desventajas:**
- Requiere más tuning de hiperparámetros
- Menos robusto a outliers

**Cuándo usar:**
- Tienes millones de filas
- Necesitas resultados rápidos
- Tienes experiencia tuneando modelos

**Parámetros clave (globales):**
```python
LIGHTGBM_PARAMS_GLOBAL = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.2,
    'num_leaves': 192,          # Complejidad del árbol
    'learning_rate': 0.05,      # Velocidad de aprendizaje
    'feature_fraction': 0.6,    # % de features por árbol
    'bagging_fraction': 0.8,    # % de datos por árbol
    'min_data_in_leaf': 100,
    'lambda_l1': 1.0,
    'lambda_l2': 3.0,
}
```

---

### CatBoost

**Ventajas:**
-  **Mejor precisión** out-of-the-box
-  Muy robusto a overfitting
-  Excelente con variables categóricas
-  Menos tuning necesario

**Desventajas:**
- Más lento que LightGBM
- Mayor uso de memoria

**Cuándo usar:**
- Quieres la mejor precisión posible
- Tienes muchas variables categóricas
- No tienes tiempo para tunear
- Datasets pequeños/medianos

**Parámetros clave (globales):**
```python
CATBOOST_PARAMS_GLOBAL = {
    'loss_function': 'Tweedie:variance_power=1.1',
    'depth': 6,                 # Profundidad del árbol
    'learning_rate': 0.03,      # Velocidad de aprendizaje
    'l2_leaf_reg': 5.0,         # Regularización L2
    'min_data_in_leaf': 100,
    'subsample': 0.8,
    'rsm': 0.7,                 # Random subspace (feature_fraction)
    'iterations': 2000,
    'task_type': 'CPU',
}
```

---

### XGBoost

**Ventajas:**
-  **Buen balance** velocidad/precisión
-  Muy popular en competiciones
-  Amplia documentación
-  Muchas herramientas disponibles

**Desventajas:**
- No tan rápido como LightGBM
- No tan preciso como CatBoost

**Cuándo usar:**
- Quieres un modelo confiable
- Necesitas reproducibilidad
- Prefieres herramientas maduras
- Balance entre todo

**Parámetros clave (globales):**
```python
XGBOOST_PARAMS_GLOBAL = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.1,
    'max_depth': 6,             # Profundidad máxima
    'learning_rate': 0.03,      # Velocidad de aprendizaje
    'subsample': 0.8,           # % de datos por árbol
    'colsample_bytree': 0.7,    # % de features por árbol
    'min_child_weight': 100,
    'reg_alpha': 1.0,           # L1
    'reg_lambda': 3.0,          # L2
    'gamma': 0.1,
}
```

---

##  Casos de Uso Específicos

### Caso 1: Maximizar Precisión
```bash
# Usar CatBoost con tuning
python scripts/run_training.py \
    --model-type catboost \
    --dias-pred 28
```

### Caso 2: Maximizar Velocidad
```bash
# Usar LightGBM
python scripts/run_training.py \
    --model-type lightgbm \
    --dias-pred 28
```

### Caso 3: Comparar y Elegir el Mejor
```bash
# Entrenar los 3 y comparar automáticamente
python scripts/compare_models.py --dias-pred 28

# Genera:
# - Tabla comparativa
# - Gráficos
# - Recomendación del mejor
```

### Caso 4: Ensemble (Combinar los 3)
```python
# Cargar los 3 modelos
from src.models.predict import cargar_modelo, predecir
from src.models.train_catboost import cargar_modelo_catboost, predecir_catboost
from src.models.train_xgboost import cargar_modelo_xgboost, predecir_xgboost

lgb = cargar_modelo('models/lgbm_global_todos_2026-01-01_28dias.txt')
cat = cargar_modelo_catboost('models/catboost_global_todos_2026-01-01_28dias.cbm')
xgb = cargar_modelo_xgboost('models/xgboost_global_todos_2026-01-01_28dias.json')

# Predecir con cada uno
pred_lgb = predecir(lgb, df)
pred_cat = predecir_catboost(cat, df)
pred_xgb = predecir_xgboost(xgb, df)

# Ensemble: Promedio ponderado
prediccion_final = (
    0.4 * pred_lgb +   # 40% LightGBM
    0.4 * pred_cat +   # 40% CatBoost
    0.2 * pred_xgb     # 20% XGBoost
)
```

---

##  Personalización de Parámetros

### Editar en config.py

```python
# config/config.py

# Para LightGBM más preciso (pero más lento)
LIGHTGBM_PARAMS_GLOBAL = {
    'num_leaves': 256,           # Mayor (actual: 192)
    'learning_rate': 0.01,       # Menor (actual: 0.05)
    'feature_fraction': 0.8,     # Mayor (actual: 0.6)
    ...
}

# Para CatBoost más rápido (pero menos preciso)
CATBOOST_PARAMS_GLOBAL = {
    'depth': 4,                  # Menor (actual: 6)
    'learning_rate': 0.1,        # Mayor (actual: 0.03)
    ...
}

# Para XGBoost con GPU
XGBOOST_PARAMS_GLOBAL = {
    'tree_method': 'gpu_hist',   # GPU (actual: hist)
    ...
}
```

---

##  Interpretación de Resultados

### Script de Comparación

```bash
python scripts/compare_models.py
```

**Output:**
```
 RESULTADOS COMPARATIVOS (no realista)

1⃣ Métricas Principales:
model       rmse        r2      error_pct  best_iteration
LightGBM    154321.8    0.8456  1.25       457
CatBoost    148902.3    0.8612  1.18       523
XGBoost     151234.7    0.8534  1.21       489

2⃣ Rankings:
 Mejor RMSE: CatBoost ($148,902)
 Mejor R²: CatBoost (0.8612)
 Mejor Error %: CatBoost (1.18%)

 RECOMENDACIÓN
 Modelo recomendado: CatBoost
```

### Archivos Generados

```
models/
 lgbm_global_{modo}_2026-01-01_28dias.txt
 catboost_global_{modo}_2026-01-01_28dias.cbm
 xgboost_global_{modo}_2026-01-01_28dias.json
 comparacion_modelos_2026-01-01_28dias.csv

comparaciones/
 comparacion_modelos_2026-01-01_28dias.png
```

---

##  Optimización de Hiperparámetros

### Con Optuna (para cualquier modelo)

```python
import optuna
from src.models.train_catboost import entrenar_modelo_catboost

def objective(trial):
    # Sugerir hiperparámetros
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
    }
    
    # Entrenar
    modelo, metricas = entrenar_modelo_catboost(
        df, fecha_corte='2026-01-01', params=params
    )
    
    # Retornar métrica a optimizar
    return metricas['rmse']

# Optimizar
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Mejores parámetros:", study.best_params)
```

---

##  Consejos Prácticos

### 1. Elige según tu caso

- **Producción rápida**: LightGBM
- **Máxima precisión**: CatBoost
- **Más estable**: XGBoost
- **No estás seguro**: Entrena los 3 y compara

### 2. Considera recursos

```python
# Si tienes GPU
CATBOOST_PARAMS_GLOBAL['task_type'] = 'GPU'
XGBOOST_PARAMS_GLOBAL['tree_method'] = 'gpu_hist'
# LightGBM detecta GPU automáticamente
```

### 3. Datasets pequeños vs grandes

- **< 100k filas**: CatBoost o XGBoost
- **> 100k filas**: LightGBM o XGBoost
- **> 1M filas**: LightGBM

### 4. Categorías

Si tienes muchas categorías:
1. **Primera opción**: CatBoost (las maneja automáticamente)
2. **Segunda opción**: LightGBM con encoding manual

---

##  Troubleshooting

### Error: "No module named 'catboost'"

```bash
pip install catboost>=1.2
```

### Error: "No module named 'xgboost'"

```bash
pip install xgboost>=2.0.0
```

### CatBoost muy lento

```python
# En config.py
CATBOOST_PARAMS_GLOBAL['task_type'] = 'GPU'  # Si tienes GPU
# O
CATBOOST_PARAMS_GLOBAL['depth'] = 4  # Reducir profundidad
```

### Diferencias entre modelos muy grandes

Esto es normal. Cada algoritmo aprende diferente:
- CatBoost suele ser más conservador
- LightGBM más agresivo
- XGBoost intermedio

---

##  Recursos Adicionales

### Documentación Oficial

- **LightGBM**: https://lightgbm.readthedocs.io/
- **CatBoost**: https://catboost.ai/docs/
- **XGBoost**: https://xgboost.readthedocs.io/

### Tutoriales Recomendados

- [Comparación práctica](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost)
- [Tuning de hiperparámetros](https://neptune.ai/blog/optuna-vs-hyperopt)

---

##  Checklist de Uso

- [ ] Instalar dependencias nuevas
- [ ] Entrenar al menos 2 modelos
- [ ] Comparar resultados
- [ ] Elegir el mejor según métricas
- [ ] Guardar modelo elegido
- [ ] Documentar decisión

---

##  Resumen Ejecutivo

**Para empezar rápido:**
```bash
# 1. Comparar los 3
python scripts/compare_models.py

# 2. Ver cuál es mejor
cat models/comparacion_modelos_*.csv

# 3. Usar el mejor
# (ver resultado de la comparación)
```

**Para máxima precisión:**
```bash
python scripts/run_training.py --model-type catboost
```

**Para máxima velocidad:**
```bash
python scripts/run_training.py --model-type lightgbm
```

**Para balance:**
```bash
python scripts/run_training.py --model-type xgboost
```

---


