"""
Utilidades compartidas entre los módulos de entrenamiento de modelos.
"""
import pandas as pd
from typing import Tuple


def preparar_datos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X e y para entrenamiento."""
    cols_drop = ["fecha_trx", "id_comercio_num", "tpv_futuro"]
    df_clean = df.dropna(subset=["tpv_futuro"]).copy()
    X = df_clean.drop(columns=cols_drop, errors="ignore")
    y = df_clean["tpv_futuro"]
    return X, y


def calcular_fechas(fecha_corte: str, dias_pred: int, dias_benchmark: int) -> dict:
    """
    Calcula los limites temporales de cada zona.

    Args:
        fecha_corte:    Primer dia del periodo de test (YYYY-MM-DD).
        dias_pred:      Horizonte de prediccion (= ancho del gap y de la validacion).
        dias_benchmark: Numero de dias del periodo de test (hacia adelante desde fecha_corte).

    Returns:
        Diccionario con las fechas de cada corte.
    """
    fecha_inicio_test = pd.to_datetime(fecha_corte)
    fecha_fin_test    = fecha_inicio_test + pd.Timedelta(days=dias_benchmark - 1)

    fecha_gap_fin     = fecha_inicio_test - pd.Timedelta(days=1)
    fecha_gap_inicio  = fecha_gap_fin - pd.Timedelta(days=dias_pred - 1)

    fecha_val_fin     = fecha_gap_inicio - pd.Timedelta(days=1)
    fecha_val_inicio  = fecha_val_fin - pd.Timedelta(days=dias_pred - 1)

    return {
        "fecha_val_inicio" : fecha_val_inicio,
        "fecha_val_fin"    : fecha_val_fin,
        "fecha_gap_inicio" : fecha_gap_inicio,
        "fecha_gap_fin"    : fecha_gap_fin,
        "fecha_inicio_test": fecha_inicio_test,
        "fecha_fin_test"   : fecha_fin_test,
    }
