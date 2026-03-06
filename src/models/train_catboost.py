"""
Módulo para entrenamiento de modelos CatBoost
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from src.models.common import preparar_datos, calcular_fechas


class _ProgressCallback:
    """Callback de progreso para CatBoost usando tqdm."""
    def __init__(self, iterations: int):
        self.pbar = tqdm(total=iterations, desc="Entrenando CatBoost",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} iter')
    
    def after_iteration(self, info):
        self.pbar.update(1)
        return True  # continuar entrenamiento
    
    def __del__(self):
        if hasattr(self, 'pbar'):
            self.pbar.close()


def optimizar_hiperparametros_catboost(X_train, y_train, X_val, y_val,
                                      params_base: dict,
                                      n_trials: int = 20) -> dict:
    """
    Optimiza hiperparámetros de CatBoost usando Optuna.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        params_base: Parámetros base del modelo
        n_trials: Número de iteraciones de Optuna
        
    Returns:
        Diccionario con los mejores parámetros encontrados
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    def objective(trial):
        trial_params = params_base.copy()
        
        trial_params['bootstrap_type'] = 'Bayesian'
        if 'subsample' in trial_params:
            del trial_params['subsample']
        
        # Hiperparámetros a optimizar
        trial_params['depth'] = trial.suggest_int('depth', 4, 10)
        trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        trial_params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 0.1, 5.0)
        trial_params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        trial_params['random_strength'] = trial.suggest_float('random_strength', 0.0, 2.0)
        trial_params['iterations'] = 2000
        
        # Entrenar modelo trial
        model = CatBoostRegressor(**trial_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
            plot=False,
            use_best_model=True,
            early_stopping_rounds=200
        )
        
        # Evaluar en validación
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    print("\nOptimizando hiperparámetros con Optuna...")
    print(f"   Trials: {n_trials}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n   Mejor RMSE: ${study.best_value:,.0f}")
    print(f"   Mejores parámetros:")
    for key, value in study.best_params.items():
        print(f"      {key}: {value}")
    
    # Combinar parámetros base con los mejores encontrados
    best_params = {**params_base, **study.best_params}

    # Ajustes específicos para el modelo final
    best_params['bootstrap_type'] = 'Bayesian'
    if 'subsample' in best_params:
        del best_params['subsample']


    return best_params


def entrenar_modelo_catboost(df: pd.DataFrame,
                             fecha_corte: str,
                             dias_pred: int = 28,
                             dias_benchmark: int = 7,
                             params: dict = None,
                             usar_optuna: bool = False,
                             optuna_trials: int = 20) -> Tuple:
    """
    Entrena modelo CatBoost global con separación correcta train/val/gap/test.

    Args:
        df:             DataFrame con features y target.
        fecha_corte:    Primer día del período de test ('YYYY-MM-DD').
        dias_pred:      Horizonte de predicción. Define también ancho del gap y validación.
        dias_benchmark: Número de días del período de test.
        params:         Parámetros CatBoost (usa config por defecto si es None).
        usar_optuna:    Si True, optimiza hiperparámetros con Optuna.
        optuna_trials:  Número de trials de Optuna (default: 20).

    Returns:
        Tupla (modelo, métricas).
    """
    from config.config import CATBOOST_PARAMS_GLOBAL

    if params is None:
        params = CATBOOST_PARAMS_GLOBAL.copy()

    fechas = calcular_fechas(fecha_corte, dias_pred, dias_benchmark)
    f_val_ini  = fechas['fecha_val_inicio']
    f_val_fin  = fechas['fecha_val_fin']
    f_gap_ini  = fechas['fecha_gap_inicio']
    f_gap_fin  = fechas['fecha_gap_fin']
    f_test     = fechas['fecha_inicio_test']
    f_fin      = fechas['fecha_fin_test']

    print(f"\nEntrenando modelo CatBoost global...")
    print(f"   Fecha corte    : {fecha_corte}  |  Días pred: {dias_pred}  |  Benchmark: {dias_benchmark}")
    if usar_optuna:
        print(f"   Optuna         : Activado ({optuna_trials} trials)")
    print(f"   Train          : inicio → {f_val_ini.date()}")
    print(f"   Validación     : {f_val_ini.date()} → {f_val_fin.date()}  ({dias_pred} días)")
    print(f"   Gap (excluido) : {f_gap_ini.date()} → {f_gap_fin.date()}  ({dias_pred} días)")
    print(f"   Test           : {f_test.date()} → {f_fin.date()}  ({dias_benchmark} días)")

    max_fecha = df['fecha_trx'].max()
    if max_fecha < f_fin:
        raise ValueError(
            f"Dataset insuficiente: llega hasta {max_fecha.date()} "
            f"pero el test requiere hasta {f_fin.date()}."
        )

    mask_train = df['fecha_trx'] < f_val_ini
    mask_val   = (df['fecha_trx'] >= f_val_ini) & (df['fecha_trx'] <= f_val_fin)
    mask_test  = (df['fecha_trx'] >= f_test) & (df['fecha_trx'] <= f_fin)

    df_train = df[mask_train]
    df_val   = df[mask_val]
    df_test  = df[mask_test].copy()

    X_train, y_train = preparar_datos(df_train)
    X_val,   y_val   = preparar_datos(df_val)
    X_test,  y_test  = preparar_datos(df_test)

    print(f"   Filas  →  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    if len(X_test) == 0:
        raise ValueError("No hay filas en el período de test.")
    if len(X_val) == 0:
        raise ValueError(
            "No hay filas en el período de validación. "
            "Verifica que el dataset sea suficientemente largo."
        )

    # Optimizar hiperparámetros si se solicita
    if usar_optuna:
        params = optimizar_hiperparametros_catboost(
            X_train, y_train, X_val, y_val,
            params_base=params,
            n_trials=optuna_trials
        )

    # Pools — early stopping sobre val_pool, nunca sobre test
    train_pool = Pool(X_train, y_train)
    val_pool   = Pool(X_val,   y_val)

    modelo = CatBoostRegressor(**params)
    progress = _ProgressCallback(params.get('iterations', 10000))
    modelo.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False,
        callbacks=[progress]
    )

    y_pred = modelo.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    metricas = {
        'rmse'          : rmse,
        'r2'            : r2,
        'tpv_real'      : float(y_test.sum()),
        'tpv_pred'      : float(y_pred.sum()),
        'best_iteration': modelo.get_best_iteration(),
    }

    print(f"\n   Resultados CatBoost (test):")
    print(f"   RMSE          : ${rmse:,.0f}")
    print(f"   R²            : {r2:.4f}")
    print(f"   Best iteration: {metricas['best_iteration']}")
    print(f"   TPV Real      : ${metricas['tpv_real']:,.0f}")
    print(f"   TPV Pred      : ${metricas['tpv_pred']:,.0f}")

    return modelo, metricas


def guardar_modelo_catboost(modelo, filepath: str):
    """Guarda modelo CatBoost en disco."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(str(filepath))
    print(f"Modelo CatBoost guardado: {filepath}")


def cargar_modelo_catboost(filepath: str):
    """Carga modelo CatBoost desde disco."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    modelo = CatBoostRegressor()
    modelo.load_model(str(filepath))
    print(f"Modelo CatBoost cargado: {filepath}")
    return modelo


def predecir_catboost(modelo, df: pd.DataFrame) -> pd.Series:
    """Genera predicciones con modelo CatBoost."""
    cols_drop = ['fecha_trx', 'id_comercio_num', 'tpv_futuro']
    X = df.drop(columns=cols_drop, errors='ignore')
    return pd.Series(modelo.predict(X), index=df.index)