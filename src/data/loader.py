"""
Módulo para carga de datos
Contiene funciones para:
- Cargar datos crudos desde parquet
- Cargar dataset de entrenamiento con encoder
- Guardar dataset procesado en parquet
- Guardar encoder con joblib
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Carga datos desde archivo parquet
    
    Args:
        filepath: Ruta al archivo parquet
        
    Returns:
        DataFrame con los datos o None si hay error
    """
    try:
        df = pd.read_parquet(filepath)
        print(f" Datos cargados: {len(df):,} filas, {len(df.columns)} columnas")
        return df
    except FileNotFoundError:
        print(f" Archivo no encontrado: {filepath}")
        return None
    except Exception as e:
        print(f" Error al leer archivo: {e}")
        return None


def load_training_dataset(filepath: str, encoder_path: str = None) -> tuple:
    """
    Carga dataset de entrenamiento y encoder
    
    Args:
        filepath: Ruta al archivo del dataset
        encoder_path: Ruta al encoder (opcional; por defecto busca encoder_comercios.joblib junto al dataset)
        
    Returns:
        Tupla (df, encoder). Lanza RuntimeError si hay error.
    """
    import joblib
    
    try:
        df = pd.read_parquet(filepath)
        if encoder_path is None:
            encoder_path = Path(filepath).parent / 'encoder_comercios.joblib'
        encoder = joblib.load(encoder_path)
        print(f" Dataset cargado: {df.shape}")
        return df, encoder
    except Exception as e:
        raise RuntimeError(f" Error al cargar dataset: {e}") from e


def save_dataset(df: pd.DataFrame, filepath: str, compression='snappy'):
    """
    Guarda DataFrame en formato parquet
    
    Args:
        df: DataFrame a guardar
        filepath: Ruta destino
        compression: Tipo de compresión ('snappy', 'gzip', etc.)
    """
    try:
        # Crear directorio si no existe
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, index=False, compression=compression)
        print(f" Dataset guardado: {filepath}")
        print(f"   Shape: {df.shape}")
    except Exception as e:
        raise RuntimeError(f" Error al guardar dataset en {filepath}: {e}") from e


def save_encoder(encoder, filepath: str):
    """
    Guarda encoder usando joblib
    
    Args:
        encoder: LabelEncoder a guardar
        filepath: Ruta destino
    """
    import joblib
    
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, filepath)
        print(f" Encoder guardado: {filepath}")
    except Exception as e:
        raise RuntimeError(f" Error al guardar encoder en {filepath}: {e}") from e
