"""
Módulo para entrenar modelos individuales (uno por cada comercio)
Soporta: LightGBM, CatBoost, XGBoost
"""
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path
from tqdm import tqdm
import gc
from typing import Optional


# Extensión de archivo por tipo de modelo
_EXTENSION = {
    'lightgbm': '.txt',
    'catboost': '.cbm',
    'xgboost':  '.json',
}


def _nombre_a_archivo(nombre_comercio: str) -> str:
    """Sanitiza el nombre del comercio para usarlo como nombre de archivo."""
    return re.sub(r'[^\w\-]', '_', nombre_comercio).strip('_')


def guardar_modelo_individual(model, nombre_comercio: str, model_type: str, directorio: Path) -> Path:
    """
    Guarda un modelo individual en disco.

    Estructura generada:
        directorio/
            MERCPAGO.txt        (lightgbm)
            DLOCAL.cbm          (catboost)
            PAYU.json           (xgboost)

    Args:
        model: Modelo entrenado
        nombre_comercio: Nombre del comercio (se sanitiza para el nombre de archivo)
        model_type: 'lightgbm', 'catboost' o 'xgboost'
        directorio: Path de la carpeta donde guardar

    Returns:
        Path del archivo guardado
    """
    directorio.mkdir(parents=True, exist_ok=True)
    ext = _EXTENSION[model_type]
    filepath = directorio / f"{_nombre_a_archivo(nombre_comercio)}{ext}"

    if model_type == 'lightgbm':
        model.save_model(str(filepath))
    elif model_type == 'catboost':
        model.save_model(str(filepath))
    elif model_type == 'xgboost':
        model.save_model(str(filepath))

    return filepath


def cargar_modelo_individual(nombre_comercio: str, model_type: str, directorio: Path):
    """
    Carga un modelo individual desde disco.

    Args:
        nombre_comercio: Nombre del comercio tal como se usó al guardar
        model_type: 'lightgbm', 'catboost' o 'xgboost'
        directorio: Path de la carpeta que contiene los modelos

    Returns:
        Modelo cargado
    """
    ext = _EXTENSION[model_type]
    filepath = directorio / f"{_nombre_a_archivo(nombre_comercio)}{ext}"

    if not filepath.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")

    if model_type == 'lightgbm':
        model = lgb.Booster(model_file=str(filepath))
    elif model_type == 'catboost':
        model = CatBoostRegressor()
        model.load_model(str(filepath))
    elif model_type == 'xgboost':
        model = xgb.Booster()
        model.load_model(str(filepath))
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    return model


def directorio_modelos_individuales(models_dir: Path, model_type: str,
                                    fecha_corte: str, dias_val: int) -> Path:
    """
    Devuelve (y crea si hace falta) el directorio estándar para modelos individuales.

    Ejemplo: models/individual/lightgbm_2026-01-01_28dias/
    """
    folder = models_dir / 'individual' / f'{model_type}_{fecha_corte}_{dias_val}dias'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def entrenar_modelo_individual(df_final: pd.DataFrame,
                               fecha_corte: str,
                               encoder=None,
                               dias_val: int = 28,
                               dias_benchmark: int = 7,
                               model_type: str = 'lightgbm',
                               guardar_modelos: bool = False,
                               models_dir: Optional[Path] = None,
                               verbose: bool = True,
                               pbar_position: int = 0,
                               usar_optuna: bool = False) -> Optional[pd.DataFrame]:
    """
    Entrena un modelo individual para CADA comercio.

    Args:
        df_final: DataFrame con features
        fecha_corte: Fecha de corte para test
        encoder: LabelEncoder de comercios
        dias_val: Días de predicción
        dias_benchmark: Días para ventana de test
        model_type: 'lightgbm', 'catboost' o 'xgboost'
        guardar_modelos: Si True, persiste cada modelo en disco
        models_dir: Carpeta raíz de modelos. Si None usa config.MODELS_DIR.
                    Los modelos se guardan en:
                    models_dir/individual/{model_type}_{fecha_corte}_{dias_val}dias/
        verbose: Mostrar información detallada
        pbar_position: Posición de barra de progreso (para backtesting anidado)
        usar_optuna: Optimizar hiperparámetros con Optuna por comercio

    Returns:
        DataFrame con predicciones de todos los comercios
    """
    # Resolver directorio de modelos
    if guardar_modelos:
        if models_dir is None:
            from config.config import MODELS_DIR
            models_dir = MODELS_DIR
        output_dir = directorio_modelos_individuales(models_dir, model_type, fecha_corte, dias_val)
        if verbose:
            print(f"Modelos individuales se guardarán en: {output_dir}")

    # Importar configuración según tipo de modelo
    if model_type == 'lightgbm':
        from config.config import (LIGHTGBM_PARAMS_INDIVIDUAL, 
                                   NUM_BOOST_ROUND_INDIVIDUAL, 
                                   EARLY_STOPPING_ROUNDS_INDIVIDUAL)
        BASE_PARAMS = LIGHTGBM_PARAMS_INDIVIDUAL.copy()
        NUM_ROUNDS = NUM_BOOST_ROUND_INDIVIDUAL
        EARLY_STOP = EARLY_STOPPING_ROUNDS_INDIVIDUAL
    elif model_type == 'catboost':
        from config.config import CATBOOST_PARAMS_INDIVIDUAL
        BASE_PARAMS = CATBOOST_PARAMS_INDIVIDUAL.copy()
        NUM_ROUNDS = BASE_PARAMS.get('iterations', 10000)
        EARLY_STOP = BASE_PARAMS.get('early_stopping_rounds', 500)
    elif model_type == 'xgboost':
        from config.config import (XGBOOST_PARAMS_INDIVIDUAL,
                                   NUM_BOOST_ROUND_INDIVIDUAL,
                                   EARLY_STOPPING_ROUNDS_INDIVIDUAL)
        BASE_PARAMS = XGBOOST_PARAMS_INDIVIDUAL.copy()
        NUM_ROUNDS = NUM_BOOST_ROUND_INDIVIDUAL
        EARLY_STOP = EARLY_STOPPING_ROUNDS_INDIVIDUAL
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")
    
    # Configuración de fechas
    fecha_inicio_test = pd.to_datetime(fecha_corte)  # fecha_corte es INICIO de test
    fecha_fin_test = fecha_inicio_test + pd.Timedelta(days=dias_benchmark - 1)
    
    # Gap termina el día anterior al test
    fecha_gap_fin = fecha_inicio_test - pd.Timedelta(days=1)
    fecha_gap_inicio = fecha_gap_fin - pd.Timedelta(days=dias_val - 1)
    
    # Validación termina el día anterior al gap
    fecha_val_fin = fecha_gap_inicio - pd.Timedelta(days=1)
    fecha_val_inicio = fecha_val_fin - pd.Timedelta(days=dias_val - 1)
    
    max_fecha_dataset = df_final['fecha_trx'].max()
    
    # Verificar datos suficientes
    if max_fecha_dataset < fecha_fin_test:
        if verbose:
            print(f"Error: Horizonte de datos insuficiente.")
            print(f"   Dataset llega hasta: {max_fecha_dataset.date()}")
            print(f"   Se requiere hasta: {fecha_fin_test.date()}")
            print(f"   Faltan {(fecha_fin_test - max_fecha_dataset).days} días")
        return None
    
    if verbose:
        print(f"Test        : {fecha_inicio_test.date()} al {fecha_fin_test.date()} ({dias_benchmark} días)")
        print(f"Gap         : {fecha_gap_inicio.date()} al {fecha_gap_fin.date()} ({dias_val} días)")
        print(f"Validación  : {fecha_val_inicio.date()} al {fecha_val_fin.date()} ({dias_val} días)")
        print(f"Train       : Inicio hasta {fecha_val_inicio.date()}")
    
    # Comercios únicos
    ids_unicos = df_final['id_comercio_num'].unique()
    list_predicciones = []
    
    # Configurar Optuna
    if usar_optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Barra de progreso
    if verbose:
        desc_bar = f"Modelos {model_type.upper()} {fecha_corte}"
        iterator = tqdm(ids_unicos, desc=desc_bar, position=pbar_position, leave=(pbar_position == 0))
    else:
        iterator = ids_unicos
    
    # Entrenar modelo para cada comercio
    for id_comercio in iterator:
        # Filtrar comercio
        df_cliente = df_final[df_final['id_comercio_num'] == id_comercio].copy()
        df_cliente = df_cliente.sort_values('fecha_trx')
        
        # Filtro: mínimo 60 observaciones
        if len(df_cliente) < 60:
            continue
        
        # Splits
        mask_train = df_cliente['fecha_trx'] < fecha_val_inicio
        mask_val = (df_cliente['fecha_trx'] >= fecha_val_inicio) & \
                   (df_cliente['fecha_trx'] <= fecha_val_fin)
        mask_test = (df_cliente['fecha_trx'] >= fecha_inicio_test) & \
                    (df_cliente['fecha_trx'] <= fecha_fin_test)
        
        df_train_loc = df_cliente[mask_train]
        df_val_loc = df_cliente[mask_val]
        df_test_loc = df_cliente[mask_test].copy()
        
        # Verificar datos suficientes
        if len(df_train_loc) < 10 or len(df_test_loc) == 0:
            continue
        
        # Preparar matrices
        cols_drop = ['fecha_trx', 'tpv_futuro', 'id_comercio_num']
        cols_drop = [c for c in cols_drop if c in df_cliente.columns]
        
        X_train = df_train_loc.drop(columns=cols_drop)
        y_train = df_train_loc['tpv_futuro']
        
        # Si no hay TPV, predecir 0
        if y_train.sum() == 0:
            df_test_loc['prediccion_individual'] = 0.0
            list_predicciones.append(df_test_loc)
            continue
        
        X_val = df_val_loc.drop(columns=cols_drop)
        y_val = df_val_loc['tpv_futuro']
        X_test = df_test_loc.drop(columns=cols_drop)
        
        # Parámetros locales
        local_params = BASE_PARAMS.copy()
        
        # Entrenar modelo según tipo
        try:
            if model_type == 'lightgbm':
                model = _entrenar_lightgbm_individual(
                    X_train, y_train, X_val, y_val, X_test,
                    local_params, usar_optuna, len(df_train_loc), NUM_ROUNDS, EARLY_STOP
                )
            elif model_type == 'catboost':
                model = _entrenar_catboost_individual(
                    X_train, y_train, X_val, y_val, X_test,
                    local_params, usar_optuna, len(df_train_loc), NUM_ROUNDS, EARLY_STOP
                )
            elif model_type == 'xgboost':
                model = _entrenar_xgboost_individual(
                    X_train, y_train, X_val, y_val, X_test,
                    local_params, usar_optuna, len(df_train_loc), NUM_ROUNDS, EARLY_STOP
                )
            
            # Predecir
            if model_type == 'lightgbm':
                df_test_loc['prediccion_individual'] = model.predict(X_test)
            elif model_type == 'catboost':
                df_test_loc['prediccion_individual'] = model.predict(X_test)
            elif model_type == 'xgboost':
                dtest = xgb.DMatrix(X_test)
                df_test_loc['prediccion_individual'] = model.predict(dtest)

            # Guardar modelo en disco si se solicitó
            if guardar_modelos and encoder is not None:
                try:
                    nombre = encoder.inverse_transform([id_comercio])[0]
                    guardar_modelo_individual(model, nombre, model_type, output_dir)
                except Exception as e_save:
                    if verbose:
                        print(f"\nAviso: no se pudo guardar modelo de comercio {id_comercio}: {e_save}")

            list_predicciones.append(df_test_loc)
            
        except Exception as e:
            if verbose:
                print(f"\nError en comercio {id_comercio}: {e}")
            df_test_loc['prediccion_individual'] = 0.0
            list_predicciones.append(df_test_loc)
            continue
    
    # Consolidar predicciones
    if not list_predicciones:
        if verbose:
            print("No se generaron predicciones")
        return None
    
    df_test_final = pd.concat(list_predicciones, ignore_index=True)
    
    # Métricas globales
    if verbose:
        tpv_real = df_test_final['tpv_futuro'].sum()
        tpv_pred = df_test_final['prediccion_individual'].sum()
        error_pct = abs(tpv_pred - tpv_real) / tpv_real * 100

        print(f"\nResultados Modelos Individuales {model_type.upper()}:")
        print(f"   Comercios procesados: {len(list_predicciones)}")
        print(f"   TPV Real: ${tpv_real:,.0f}")
        print(f"   TPV Pred: ${tpv_pred:,.0f}")
        print(f"   Error %: {error_pct:.2f}%")
        if guardar_modelos:
            print(f"   Modelos guardados en: {output_dir}")
    
    return df_test_final


def _entrenar_lightgbm_individual(X_train, y_train, X_val, y_val, X_test, params, usar_optuna, n_samples, num_rounds, early_stop):
    """Entrena modelo LightGBM individual"""
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [lgb.Dataset(X_val, label=y_val, reference=train_data)] if len(X_val) > 0 else [train_data]
    
    if usar_optuna and n_samples > 150:
        import optuna
        
        def objective(trial):
            trial_params = params.copy()
            trial_params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.0, 2.0)
            trial_params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.0, 5.0)
            trial_params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.0, 5.0)
            trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1)
            trial_params['num_leaves'] = trial.suggest_int('num_leaves', 7, 31)
            
            model = lgb.train(
                trial_params,
                train_data,
                valid_sets=valid_sets,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            preds = model.predict(X_val) if len(X_val) > 0 else model.predict(X_train)
            y_eval = y_val if len(y_val) > 0 else y_train
            rmse = np.sqrt(mean_squared_error(y_eval, preds))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        best_params = {**params, **study.best_params}
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=num_rounds,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )
    else:
        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=num_rounds,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )
    
    return model


def _entrenar_catboost_individual(X_train, y_train, X_val, y_val, X_test, params, usar_optuna, n_samples, num_rounds, early_stop):
    """Entrena modelo CatBoost individual"""
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val) if len(X_val) > 0 else None
    
    if usar_optuna and n_samples > 150:
        import optuna
        
        def objective(trial):
            trial_params = params.copy()
            trial_params['depth'] = trial.suggest_int('depth', 3, 6)
            trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1)
            trial_params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 0.5, 5.0)
            trial_params['iterations'] = 1000
            trial_params['early_stopping_rounds'] = 100
            
            model = CatBoostRegressor(**trial_params)
            model.fit(train_pool, eval_set=eval_pool, verbose=False, plot=False)
            
            preds = model.predict(X_val) if len(X_val) > 0 else model.predict(X_train)
            y_eval = y_val if len(y_val) > 0 else y_train
            rmse = np.sqrt(mean_squared_error(y_eval, preds))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        best_params = {**params, **study.best_params}
        best_params['iterations'] = num_rounds
        best_params['early_stopping_rounds'] = early_stop
        model = CatBoostRegressor(**best_params)
        model.fit(train_pool, eval_set=eval_pool, verbose=False, plot=False, use_best_model=True)
    else:
        local_params = params.copy()
        local_params['iterations'] = num_rounds
        local_params['early_stopping_rounds'] = early_stop
        model = CatBoostRegressor(**local_params)
        model.fit(train_pool, eval_set=eval_pool, verbose=False, plot=False, use_best_model=True)
    
    return model


def _entrenar_xgboost_individual(X_train, y_train, X_val, y_val, X_test, params, usar_optuna, n_samples, num_rounds, early_stop):
    """Entrena modelo XGBoost individual"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val) if len(X_val) > 0 else dtrain
    evals = [(dval, 'val')] if len(X_val) > 0 else [(dtrain, 'train')]
    
    if usar_optuna and n_samples > 150:
        import optuna
        
        def objective(trial):
            trial_params = params.copy()
            trial_params['max_depth'] = trial.suggest_int('max_depth', 3, 6)
            trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1)
            trial_params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 5.0)
            trial_params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 5.0)
            
            model = xgb.train(
                trial_params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=100,
                verbose_eval=False
            )
            
            preds = model.predict(dval)
            y_eval = y_val if len(y_val) > 0 else y_train
            rmse = np.sqrt(mean_squared_error(y_eval, preds))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        best_params = {**params, **study.best_params}
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stop,
            verbose_eval=False
        )
    else:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stop,
            verbose_eval=False
        )
    
    return model
