"""
Módulo para predicciones con modelos entrenados
Soporta: LightGBM, CatBoost, XGBoost
"""
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import pandas as pd
from pathlib import Path


def cargar_modelo(filepath: str, model_type: str = None):
    """
    Carga modelo desde disco
    
    Args:
        filepath: Ruta al archivo del modelo
        model_type: Tipo de modelo ('lightgbm', 'catboost', 'xgboost')
                   Si es None, se infiere de la extensión del archivo
    
    Returns:
        Modelo cargado
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    
    # Inferir tipo de modelo de la extensión si no se especifica
    if model_type is None:
        extension = filepath.suffix
        if extension == '.txt':
            model_type = 'lightgbm'
        elif extension == '.cbm':
            model_type = 'catboost'
        elif extension == '.json':
            model_type = 'xgboost'
        else:
            raise ValueError(f"No se pudo inferir el tipo de modelo de la extensión: {extension}")
    
    # Cargar según tipo
    if model_type == 'lightgbm':
        modelo = lgb.Booster(model_file=str(filepath))
        print(f"Modelo LightGBM cargado: {filepath}")
    elif model_type == 'catboost':
        modelo = CatBoostRegressor()
        modelo.load_model(str(filepath))
        print(f"Modelo CatBoost cargado: {filepath}")
    elif model_type == 'xgboost':
        modelo = xgb.Booster()
        modelo.load_model(str(filepath))
        print(f"Modelo XGBoost cargado: {filepath}")
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    return modelo


def predecir(modelo, df: pd.DataFrame, model_type: str = None) -> pd.Series:
    """
    Genera predicciones con el modelo
    
    Args:
        modelo: Modelo entrenado (LightGBM, CatBoost o XGBoost)
        df: DataFrame con features (debe tener la misma estructura que el entrenamiento)
        model_type: Tipo de modelo ('lightgbm', 'catboost', 'xgboost')
                   Si es None, se intenta inferir del tipo de objeto
    
    Returns:
        Series con predicciones
    """
    # Inferir tipo de modelo del objeto si no se especifica
    if model_type is None:
        if isinstance(modelo, lgb.Booster):
            model_type = 'lightgbm'
        elif isinstance(modelo, CatBoostRegressor):
            model_type = 'catboost'
        elif isinstance(modelo, xgb.Booster):
            model_type = 'xgboost'
        else:
            raise ValueError(f"No se pudo inferir el tipo de modelo del objeto: {type(modelo)}")
    
    # Preparar datos (eliminar columnas que no son features)
    cols_drop = ['fecha_trx', 'id_comercio_num', 'tpv_futuro']
    cols_to_drop = [c for c in cols_drop if c in df.columns]
    
    if cols_to_drop:
        X = df.drop(columns=cols_to_drop)
    else:
        X = df.copy()
    
    # Predecir según tipo de modelo
    if model_type == 'lightgbm':
        predicciones = modelo.predict(X)
    elif model_type == 'catboost':
        predicciones = modelo.predict(X)
    elif model_type == 'xgboost':
        dmatrix = xgb.DMatrix(X)
        predicciones = modelo.predict(dmatrix)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    return pd.Series(predicciones, index=df.index)


def cargar_y_predecir(filepath: str, df: pd.DataFrame, model_type: str = None) -> pd.Series:
    """
    Función de conveniencia que carga el modelo y hace predicciones en un solo paso
    
    Args:
        filepath: Ruta al archivo del modelo
        df: DataFrame con features
        model_type: Tipo de modelo (opcional, se infiere de la extensión)
    
    Returns:
        Series con predicciones
    """
    modelo = cargar_modelo(filepath, model_type)
    return predecir(modelo, df, model_type)


def cargar_modelos_individuales(directorio: str, model_type: str = None) -> dict:
    """
    Carga todos los modelos individuales de un directorio.
    
    Args:
        directorio: Ruta al directorio con modelos individuales
                   (ej: models/individual/catboost_2026-01-01_28dias/)
        model_type: Tipo de modelo (opcional, se infiere del primer archivo)
    
    Returns:
        Diccionario {nombre_comercio: modelo}
        
    Ejemplo:
        >>> modelos = cargar_modelos_individuales('models/individual/catboost_2026-01-01_28dias/')
        >>> pred_mercpago = predecir(modelos['MERCPAGO'], df_mercpago)
    """
    directorio = Path(directorio)
    
    if not directorio.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directorio}")
    
    # Inferir tipo de modelo del primer archivo si no se especifica
    if model_type is None:
        archivos = list(directorio.glob('*'))
        if archivos:
            ext = archivos[0].suffix
            if ext == '.txt':
                model_type = 'lightgbm'
            elif ext == '.cbm':
                model_type = 'catboost'
            elif ext == '.json':
                model_type = 'xgboost'
    
    # Cargar todos los modelos
    modelos = {}
    extensiones = {'.txt', '.cbm', '.json'}
    
    for archivo in directorio.iterdir():
        if archivo.suffix in extensiones:
            nombre_comercio = archivo.stem  # Nombre sin extensión
            try:
                modelos[nombre_comercio] = cargar_modelo(str(archivo), model_type)
                print(f"  Cargado: {nombre_comercio}")
            except Exception as e:
                print(f"  Error al cargar {nombre_comercio}: {e}")
    
    print(f"\nTotal modelos cargados: {len(modelos)}")
    return modelos


def predecir_con_modelos_individuales(modelos: dict, df: pd.DataFrame, 
                                      encoder, model_type: str = None) -> pd.DataFrame:
    """
    Genera predicciones usando modelos individuales.
    
    Args:
        modelos: Diccionario {nombre_comercio: modelo}
        df: DataFrame con features (debe incluir 'id_comercio_num')
        encoder: LabelEncoder para convertir id_comercio_num a nombres
        model_type: Tipo de modelo (opcional)
    
    Returns:
        DataFrame original con columna 'prediccion_individual' agregada
    """
    df = df.copy()
    df['prediccion_individual'] = 0.0
    
    for id_comercio in df['id_comercio_num'].unique():
        try:
            nombre = encoder.inverse_transform([int(id_comercio)])[0]
            
            if nombre in modelos:
                mask = df['id_comercio_num'] == id_comercio
                df_comercio = df[mask]
                
                predicciones = predecir(modelos[nombre], df_comercio, model_type)
                df.loc[mask, 'prediccion_individual'] = predicciones.values
        except Exception as e:
            print(f"Error al predecir para comercio {id_comercio}: {e}")
    
    return df
