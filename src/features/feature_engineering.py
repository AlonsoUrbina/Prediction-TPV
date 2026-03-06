"""
Módulo para generación de features (feature engineering).
Contiene la función principal generar_dataset() que toma el DataFrame limpio y genera todas
las features necesarias para el modelado, incluyendo:
- Procesamiento de identidad (encoding de comercios)
- Cálculo de antigüedad
- Procesamiento de eventos (feriados, cyber events)
- Agregaciones diarias y features derivados (rolling windows, ciclos, etc.)
- Agregado de variable target (TPV futuro)
"""
import pandas as pd
import numpy as np
import holidays
import gc
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional


def generar_dataset(df_raw: pd.DataFrame,
                    df_antiguedad_ref: pd.DataFrame,
                    dias_prediccion: int = 28,
                    cybers_list: Optional[List[str]] = None,
                    para_prediccion: bool = False) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Toma el DataFrame después de la limpieza y devuelve el DataFrame final
    listo para entrenar el modelo.

    Parámetros:
    - df_raw: DataFrame original con variables limpias.
    - df_antiguedad_ref: DataFrame con fechas de primera transacción por comercio.
    - dias_prediccion: Número de días para las variables futuras.
    - cybers_list: Lista de fechas de Cyber Events.

    Retorna:
    - df_final: DataFrame listo para modelar con todas las features generadas.
    - encoder: LabelEncoder ajustado sobre nombre_comercio.
    """

    df = df_raw.copy()

    # Asegurar formato fecha en datetime
    df['fecha_trx'] = pd.to_datetime(df['fecha_trx'])


    # --- Procesamiento de Identidad ---

    df['nombre_comercio'] = df['nombre_comercio'].astype(str).str.upper().str.strip()
    encoder = LabelEncoder()
    df['id_comercio_num'] = encoder.fit_transform(df['nombre_comercio'])
    # MCC a entero (rellenar nulos con 0, aunque idealmente no deberia haber nulos)
    df['mcc'] = df['mcc'].fillna(0).astype(int)


    # --- Procesamiento de antiguedad ---

    df_ref = df_antiguedad_ref[['nombre_fantasia', 'fecha_primera_tx']].copy()
    df_ref.columns = ['nombre_comercio', 'fecha_primera_tx']
    # Convertir a datetime (dayfirst=True para formato CL dia-mes-año)
    df_ref['fecha_primera_tx'] = pd.to_datetime(df_ref['fecha_primera_tx'], dayfirst=True)
    df = df.merge(df_ref, on='nombre_comercio', how='left')
    dias_float = (df['fecha_trx'] - df['fecha_primera_tx']) / np.timedelta64(1, 'D')
    # Conversion a meses truncados
    df['antiguedad_meses'] = (dias_float / 30.44).fillna(-1).astype(int)
    df.drop(columns=['fecha_primera_tx', 'nombre_comercio'], errors='ignore', inplace=True)
    del df_ref
    gc.collect()


    # --- Procesamiento de eventos ---

    fecha_min = df['fecha_trx'].min()
    fecha_max = df['fecha_trx'].max() + pd.Timedelta(days=dias_prediccion + 10)

    año_min, año_max = fecha_min.year, fecha_max.year
    cl_holidays = holidays.CL(years=range(año_min, año_max + 1))
    feriados_dt = pd.to_datetime(sorted(list(cl_holidays.keys())))
    cybers_dt = pd.to_datetime(cybers_list) if cybers_list else pd.DatetimeIndex([])

    df_cal = pd.DataFrame({'fecha': pd.date_range(start=fecha_min, end=fecha_max, freq='D')})
    df_cal['es_feriado'] = df_cal['fecha'].isin(feriados_dt).astype(int)
    df_cal['es_cyber'] = df_cal['fecha'].isin(cybers_dt).astype(int)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=dias_prediccion)
    df_cal['q_feriados_futuros'] = df_cal['es_feriado'].shift(-1).rolling(window=indexer).sum().fillna(0)
    df_cal['flag_cyber_futuro'] = df_cal['es_cyber'].shift(-1).rolling(window=indexer).max().fillna(0)

    cols_cal = ['fecha', 'q_feriados_futuros', 'flag_cyber_futuro']
    df = df.merge(df_cal[cols_cal], left_on='fecha_trx', right_on='fecha', how='left')
    df.drop(columns=['fecha'], inplace=True)

    del df_cal, feriados_dt, cybers_dt
    gc.collect()


    # --- Agregacion diaria y generacion de features ---

    # Normalizacion de texto
    cat_prod = df['categoria_producto'].astype(str).str.upper().str.strip()
    cat_prod = cat_prod.str.replace('É','E').str.replace('Á','A').str.replace('Í','I')
    marca = df['marca_liq'].astype(str).str.upper().str.strip()
    presencia = df['tarjeta_presente'].astype(str).str.upper().str.strip()

    df['is_debito'] = np.where(cat_prod.str.contains('DEBITO'), df['tpv'], 0)
    df['is_credito'] = np.where(cat_prod.str.contains('CREDITO'), df['tpv'], 0)
    df['is_visa'] = np.where(marca.str.contains('VISA'), df['tpv'], 0)
    df['is_master'] = np.where(marca.str.contains('MASTERCARD'), df['tpv'], 0)
    df['is_inter'] = np.where(df['nacionalidad'] == 'Internacional', df['tpv'], 0)
    df['is_presente'] = np.where(presencia == 'SI', df['tpv'], 0)

    df.drop(columns=['categoria_producto', 'marca_liq', 'nacionalidad'], inplace=True)

    reglas = {
        'tpv': 'sum', 'cantidad_tx': 'sum',
        'is_debito': 'sum', 'is_credito': 'sum',
        'is_visa': 'sum', 'is_master': 'sum', 'is_inter': 'sum', 'is_presente': 'sum',
        'antiguedad_meses': 'max',
        'q_feriados_futuros': 'first', 'flag_cyber_futuro': 'first'
    }
    df_day = df.groupby(['id_comercio_num', 'mcc', 'fecha_trx']).agg(reglas).reset_index()

    del df, cat_prod, marca, presencia
    gc.collect()


    # --- Rellenar huecos en el esquema completo ---

    pares = df_day[['id_comercio_num', 'mcc']].drop_duplicates()
    rango_fechas = pd.DataFrame({'fecha_trx': pd.date_range(fecha_min, df_day['fecha_trx'].max(), freq='D')})

    pares['key'] = 1
    rango_fechas['key'] = 1
    skeleton = pd.merge(pares, rango_fechas, on='key').drop('key', axis=1)
    df_full = pd.merge(skeleton, df_day, on=['id_comercio_num', 'mcc', 'fecha_trx'], how='left')

    del pares, rango_fechas, skeleton, df_day
    gc.collect()

    cols_vol = ['tpv', 'cantidad_tx', 'is_debito', 'is_credito',
                'is_visa', 'is_master', 'is_inter', 'is_presente']
    df_full[cols_vol] = df_full[cols_vol].fillna(0)

    mes = df_full['fecha_trx'].dt.month
    dia_sem = df_full['fecha_trx'].dt.dayofweek
    dia_mes = df_full['fecha_trx'].dt.day
    dias_en_mes = df_full['fecha_trx'].dt.days_in_month

    df_full['dias_rest_mes_actual'] = (dias_en_mes - dia_mes).astype(int)

    df_full['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    df_full['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    df_full['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7)
    df_full['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7)
    df_full['dia_mes_sin'] = np.sin(2 * np.pi * dia_mes / dias_en_mes)
    df_full['dia_mes_cos'] = np.cos(2 * np.pi * dia_mes / dias_en_mes)

    # Rellenar antiguedad por ID
    df_full['antiguedad_meses'] = (
        df_full.groupby('id_comercio_num')['antiguedad_meses']
        .transform(lambda x: x.ffill().bfill())
        .fillna(-1).astype(int)
    )
    df_full['antiguedad_meses'] = (
        df_full.groupby(['id_comercio_num', 'fecha_trx'])['antiguedad_meses']
        .transform('max')
    )

    # Rellenar variables de calendario por fecha
    cols_temp = ['dias_rest_mes_actual', 'mes_sin', 'mes_cos', 'dia_semana_sin',
                 'q_feriados_futuros', 'flag_cyber_futuro']
    for col in cols_temp:
        df_full[col] = df_full.groupby('fecha_trx')[col].transform('first').fillna(0)


    # --- Variables agregadas con rolling windows ---

    df_full = df_full.sort_values(['id_comercio_num', 'mcc', 'fecha_trx']).reset_index(drop=True)
    g = df_full.groupby(['id_comercio_num', 'mcc'])

    tpv = g['tpv']
    tx  = g['cantidad_tx']

    # TPV Rolling
    df_full['tpv_acumulado_7d']  = tpv.rolling(7).sum().values
    df_full['tpv_acumulado_15d'] = tpv.rolling(15).sum().values
    df_full['tpv_acumulado_30d'] = tpv.rolling(30).sum().values
    df_full['tpv_acumulado_Xd']  = tpv.rolling(dias_prediccion).sum().values

    # Medias
    df_full['media_movil_7d']  = tpv.rolling(7).mean().values
    df_full['media_movil_30d'] = tpv.rolling(30).mean().values
    df_full['media_tx_7d']     = tx.rolling(7).mean().values
    df_full['media_tx_30d']    = tx.rolling(30).mean().values

    # Volatilidad y aceleracion
    df_full['volatilidad_tpv_30d'] = tpv.rolling(30).std().values
    df_full['aceleracion_tpv'] = df_full['media_movil_7d'] / (df_full['media_movil_30d'] + 1)
    df_full['aceleracion_tx']  = df_full['media_tx_7d'] / (df_full['media_tx_30d'] + 1)

    # Cantidad rolling
    df_full['cantidad_tx_7d']  = tx.rolling(7).sum().values
    df_full['cantidad_tx_30d'] = tx.rolling(30).sum().values

    # Ticket promedio
    df_full['ticket_promedio_7d']  = (
        df_full['tpv_acumulado_7d'] / df_full['cantidad_tx_7d'].replace(0, np.nan)
    ).fillna(0)
    df_full['ticket_promedio_30d'] = (
        df_full['tpv_acumulado_30d'] / df_full['cantidad_tx_30d'].replace(0, np.nan)
    ).fillna(0)

    # Mix (porcentajes a 30d)
    total_30 = df_full['tpv_acumulado_30d'].replace(0, np.nan)

    def calc_pct_and_drop(col_name):
        res = (g[col_name].rolling(30).sum().values / total_30).astype(np.float32)
        df_full.drop(columns=[col_name], inplace=True)
        return res

    df_full['pct_debito_30d']        = calc_pct_and_drop('is_debito')
    df_full['pct_credito_30d']       = calc_pct_and_drop('is_credito')
    df_full['pct_visa_30d']          = calc_pct_and_drop('is_visa')
    df_full['pct_mastercard_30d']    = calc_pct_and_drop('is_master')
    df_full['pct_internacional_30d'] = calc_pct_and_drop('is_inter')
    df_full['pct_presente_30d']      = calc_pct_and_drop('is_presente')

    # Target
    indexer_fwd = pd.api.indexers.FixedForwardWindowIndexer(window_size=dias_prediccion)
    df_full['tpv_futuro']            = g['tpv'].shift(-1).rolling(window=indexer_fwd).sum().values
    df_full['tpv_futuro_año_anterior'] = g['tpv_futuro'].shift(365).values

    # Solo filas con target valido (omitir en modo prediccion: los ultimos rows no tienen target)
    if para_prediccion:
        df_final = df_full
    else:
        df_final = df_full.dropna(subset=['tpv_futuro'])

    del df_full, g, tpv, tx
    gc.collect()

    return df_final, encoder
