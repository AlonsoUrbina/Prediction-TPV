"""
Módulo para entrenamiento de modelos LightGBM (globales)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
from src.models.common import preparar_datos, calcular_fechas


def _crear_callback_progreso(num_boost_round: int):
    """Crea callback de tqdm para mostrar progreso del entrenamiento."""
    pbar = tqdm(total=num_boost_round, desc="Entrenando LightGBM", 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} iter')
    
    def callback(env):
        pbar.update(1)
        if env.iteration == env.end_iteration - 1:
            pbar.close()
    
    callback.order = 0
    return callback


def optimizar_hiperparametros_lightgbm(X_train, y_train, X_val, y_val,
                                      params_base: dict,
                                      n_trials: int = 20) -> dict:
    """
    Optimiza hiperparámetros de LightGBM usando Optuna.
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    def objective(trial):
        trial_params = params_base.copy()
        
        # Hiperparámetros a optimizar
        trial_params['num_leaves'] = trial.suggest_int('num_leaves', 64, 512)
        trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        trial_params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.0, 2.0)
        trial_params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.0, 2.0)
        trial_params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.6, 1.0)
        trial_params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.6, 1.0)
        trial_params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 20, 100)

        trial_params['feature_pre_filter'] = False  # Para evitar que LightGBM descarte features antes de la optimización
    
        
        # Entrenar modelo trial
        model = lgb.train(
            trial_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        
        # Evaluar en validación
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
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
    return best_params


def entrenar_modelo_global(df: pd.DataFrame,
                           fecha_corte: str,
                           dias_pred: int = 28,
                           dias_benchmark: int = 7,
                           params: dict = None,
                           usar_optuna: bool = False,
                           optuna_trials: int = 20) -> Tuple:
    """
    Entrena modelo LightGBM global con separación correcta train/val/gap/test.

    Args:
        df:             DataFrame con features y target.
        fecha_corte:    Primer día del período de test ('YYYY-MM-DD').
        dias_pred:      Horizonte de predicción (días). Define también el ancho
                        del gap y de la validación.
        dias_benchmark: Número de días del período de test.
        params:         Parámetros LightGBM (usa config por defecto si es None).
        usar_optuna:    Si True, optimiza hiperparámetros con Optuna.
        optuna_trials:  Número de trials de Optuna (default: 20).

    Returns:
        Tupla (modelo, métricas).
    """
    from config.config import (LIGHTGBM_PARAMS_GLOBAL,
                               NUM_BOOST_ROUND_GLOBAL,
                               EARLY_STOPPING_ROUNDS_GLOBAL)

    if params is None:
        params = LIGHTGBM_PARAMS_GLOBAL.copy()

    fechas = calcular_fechas(fecha_corte, dias_pred, dias_benchmark)
    f_val_ini  = fechas['fecha_val_inicio']
    f_val_fin  = fechas['fecha_val_fin']
    f_gap_ini  = fechas['fecha_gap_inicio']
    f_gap_fin  = fechas['fecha_gap_fin']
    f_test     = fechas['fecha_inicio_test']
    f_fin      = fechas['fecha_fin_test']

    print(f"\nEntrenando modelo LightGBM global...")
    print(f"   Fecha corte    : {fecha_corte}  |  Días pred: {dias_pred}  |  Benchmark: {dias_benchmark}")
    if usar_optuna:
        print(f"   Optuna         : Activado ({optuna_trials} trials)")
    print(f"   Train          : inicio → {f_val_ini.date()}")
    print(f"   Validación     : {f_val_ini.date()} → {f_val_fin.date()}  ({dias_pred} días)")
    print(f"   Gap (excluido) : {f_gap_ini.date()} → {f_gap_fin.date()}  ({dias_pred} días)")
    print(f"   Test           : {f_test.date()} → {f_fin.date()}  ({dias_benchmark} días)")

    # Verificar que el dataset llega hasta el fin del test
    max_fecha = df['fecha_trx'].max()
    if max_fecha < f_fin:
        raise ValueError(
            f"Dataset insuficiente: llega hasta {max_fecha.date()} "
            f"pero el test requiere hasta {f_fin.date()}."
        )

    # Máscaras
    mask_train = df['fecha_trx'] < f_val_ini
    mask_val   = (df['fecha_trx'] >= f_val_ini) & (df['fecha_trx'] <= f_val_fin)
    # GAP: [f_gap_ini, f_gap_fin] → se omite completamente
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
        params = optimizar_hiperparametros_lightgbm(
            X_train, y_train, X_val, y_val,
            params_base=params,
            n_trials=optuna_trials
        )

    # Datasets LightGBM — early stopping sobre VAL, no sobre test
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    modelo = lgb.train(
        params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND_GLOBAL,
        valid_sets=[val_data],
        valid_names=['VALIDACION'],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS_GLOBAL, verbose=False),
            _crear_callback_progreso(NUM_BOOST_ROUND_GLOBAL)
        ]
    )

    # Evaluar en test
    y_pred = modelo.predict(X_test, num_iteration=modelo.best_iteration)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    metricas = {
        'rmse'          : rmse,
        'r2'            : r2,
        'tpv_real'      : float(y_test.sum()),
        'tpv_pred'      : float(y_pred.sum()),
        'best_iteration': modelo.best_iteration,
    }

    print(f"\n   Resultados LightGBM (test):")
    print(f"   RMSE          : ${rmse:,.0f}")
    print(f"   R²            : {r2:.4f}")
    print(f"   Best iteration: {modelo.best_iteration}")
    print(f"   TPV Real      : ${metricas['tpv_real']:,.0f}")
    print(f"   TPV Pred      : ${metricas['tpv_pred']:,.0f}")

    return modelo, metricas


def guardar_modelo(modelo, filepath: str):
    """Guarda modelo LightGBM en disco."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    modelo.save_model(str(filepath))
    print(f"Modelo LightGBM guardado: {filepath}")