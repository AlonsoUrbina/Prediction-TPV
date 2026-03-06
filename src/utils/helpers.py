"""
Utilidades y funciones auxiliares
"""
import pandas as pd
import warnings


def configure_warnings(suppress: bool = True):
    """Configura warnings"""
    if suppress:
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)


def print_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """Imprime información del dataset"""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumnas:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    print(f"\nNulos:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
