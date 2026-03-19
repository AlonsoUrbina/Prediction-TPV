"""
Utilidades compartidas entre los módulos de entrenamiento de modelos.
"""
import pandas as pd
from pathlib import Path
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


def guardar_importancia_variables(modelo, model_type: str, model_filepath: str):
    """
    Extrae la importancia de variables del modelo entrenado, guarda un CSV
    con la importancia porcentual de todas las variables y un PDF con las 10
    más importantes.

    Args:
        modelo:         Modelo entrenado (LightGBM, CatBoost o XGBoost Booster).
        model_type:     Tipo de modelo ('lightgbm', 'catboost', 'xgboost').
        model_filepath: Ruta del archivo del modelo guardado (con extensión).
                        Los archivos de importancia se guardan en la misma
                        carpeta con el mismo nombre base.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = Path(model_filepath).with_suffix('')

    # Extraer importancia según framework
    if model_type == 'lightgbm':
        feature_names = modelo.feature_name()
        importancias  = modelo.feature_importance(importance_type='gain')
        df_imp = pd.DataFrame({'feature': feature_names, 'importancia': importancias.astype(float)})

    elif model_type == 'catboost':
        feature_names = modelo.feature_names_
        importancias  = modelo.get_feature_importance()
        df_imp = pd.DataFrame({'feature': feature_names, 'importancia': importancias.astype(float)})

    elif model_type == 'xgboost':
        scores = modelo.get_score(importance_type='gain')
        df_imp = pd.DataFrame(list(scores.items()), columns=['feature', 'importancia'])
        df_imp['importancia'] = df_imp['importancia'].astype(float)

    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    # Calcular importancia porcentual y ordenar
    total = df_imp['importancia'].sum()
    df_imp['importancia_pct'] = (df_imp['importancia'] / total * 100).round(4)
    df_imp = df_imp.sort_values('importancia', ascending=False).reset_index(drop=True)
    df_imp.insert(0, 'rank', range(1, len(df_imp) + 1))

    # Guardar CSV
    csv_path = str(base) + '_feature_importance.csv'
    df_imp.to_csv(csv_path, index=False)
    print(f"   Importancia variables (CSV) : {csv_path}")

    # Guardar PDF con top 10
    top10 = df_imp.head(10).sort_values('importancia_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10['feature'], top10['importancia_pct'], color='steelblue')

    for bar, val in zip(bars, top10['importancia_pct']):
        ax.text(
            bar.get_width() + top10['importancia_pct'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%',
            va='center',
            fontsize=9
        )

    ax.set_xlabel('Importancia (%)')
    ax.set_title(f'Top 10 Variables Más Importantes  ({model_type.upper()})')
    ax.set_xlim(0, top10['importancia_pct'].max() * 1.18)
    plt.tight_layout()

    pdf_path = str(base) + '_feature_importance.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   Importancia variables (PDF) : {pdf_path}")
