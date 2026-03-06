"""
Módulo para limpieza y preprocesamiento de datos
Contiene todas las transformaciones aplicadas antes del feature engineering
"""
import pandas as pd
from typing import Dict, List, Optional


def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Elimina columnas innecesarias del DataFrame.
    Corresponde a:
        df.drop(columns=['id_sucursal', 'fecha_devolucion', 'tipo_tx', 'merchant_neto'])
    """
    df_clean = df.copy()
    dropped = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean.drop(columns=dropped, inplace=True)
    if dropped:
        print(f"  Columnas eliminadas: {dropped}")
    return df_clean


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con valores nulos.
    Corresponde a:
        df.dropna(inplace=True)
    """
    n_antes = len(df)
    df_clean = df.dropna()
    n_despues = len(df_clean)
    n_eliminadas = n_antes - n_despues
    print(f"  Filas nulas eliminadas: {n_eliminadas:,} ({n_eliminadas / n_antes * 100:.2f}%)")
    print(f"  Filas restantes: {n_despues:,}")
    return df_clean


def map_merchant_names(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Crea la columna nombre_comercio a partir del RUT (id_comercio) y elimina la columna original.
    Corresponde a:
        df['nombre_comercio'] = df['id_comercio'].map(mapping_comercio).fillna(df['id_comercio'])
        df.drop(columns=['id_comercio'], inplace=True)
    
    Si un RUT no está en el mapping, conserva el valor original de id_comercio como nombre.
    """
    if 'id_comercio' not in df.columns:
        print("  Columna 'id_comercio' no encontrada, se omite el mapeo")
        return df

    df_clean = df.copy()
    n_antes = df_clean['id_comercio'].nunique()

    df_clean['nombre_comercio'] = df_clean['id_comercio'].map(mapping).fillna(df_clean['id_comercio'])
    df_clean.drop(columns=['id_comercio'], inplace=True)

    n_mapeados = df_clean['nombre_comercio'].isin(mapping.values()).sum()
    print(f"  RUTs únicos en data: {n_antes}")
    print(f"  Registros mapeados a nombre conocido: {n_mapeados:,}")
    print(f"  Comercios únicos resultantes: {df_clean['nombre_comercio'].nunique()}")
    return df_clean


def filter_merchants_by_name(df: pd.DataFrame, merchants_to_keep: List[str]) -> pd.DataFrame:
    """
    Conserva únicamente los comercios especificados en la lista.
    Corresponde a la selección de COMERCIOS_A_MANTENER en config.py.
    """
    if 'nombre_comercio' not in df.columns:
        print("  Columna 'nombre_comercio' no encontrada")
        return df

    n_antes = len(df)
    comercios_antes = df['nombre_comercio'].nunique()
    df_clean = df[df['nombre_comercio'].isin(merchants_to_keep)].copy()
    n_despues = len(df_clean)
    comercios_despues = df_clean['nombre_comercio'].nunique()

    print(f"  Comercios: {comercios_antes} -> {comercios_despues}")
    print(f"  Filas: {n_antes:,} -> {n_despues:,}")
    return df_clean


def filter_merchants_by_min_instances(df: pd.DataFrame, min_instances: int = 100) -> pd.DataFrame:
    """
    Elimina comercios con menos de min_instances filas en el dataset.
    Corresponde a:
        comercios_a_eliminar = distribucion_comercios[distribucion_comercios < 100].index
        df = df[~df['nombre_comercio'].isin(comercios_a_eliminar)]
    """
    if 'nombre_comercio' not in df.columns:
        print("  Columna 'nombre_comercio' no encontrada")
        return df

    distribucion = df['nombre_comercio'].value_counts()
    comercios_a_eliminar = distribucion[distribucion < min_instances].index.tolist()

    if comercios_a_eliminar:
        print(f"  Comercios eliminados por tener menos de {min_instances} instancias:")
        for c in comercios_a_eliminar:
            print(f"    - {c}: {distribucion[c]} filas")
    else:
        print(f"  Todos los comercios tienen al menos {min_instances} instancias")

    df_clean = df[~df['nombre_comercio'].isin(comercios_a_eliminar)].copy()
    return df_clean


def clean_mcc(df: pd.DataFrame, mcc_invalidos: List[int] = None) -> pd.DataFrame:
    """
    Limpia la columna mcc:
      1. Convierte a numérico (errores -> NaN)
      2. Elimina MCCs inválidos (0, 34) y nulos
    Corresponde a:
        df['mcc'] = pd.to_numeric(df['mcc'], errors='coerce').fillna(0).astype(int)
        df = df[~df['mcc'].isin([0, 34])]
        df = df.dropna(subset=['mcc'])
    """
    if 'mcc' not in df.columns:
        print("  Columna 'mcc' no encontrada")
        return df

    if mcc_invalidos is None:
        mcc_invalidos = [0, 34]

    df_clean = df.copy()

    # Convertir a numérico
    df_clean['mcc'] = pd.to_numeric(df_clean['mcc'], errors='coerce')

    # Contar y reportar lo que se elimina
    n_antes = len(df_clean)
    mask_invalidos = df_clean['mcc'].isin(mcc_invalidos) | df_clean['mcc'].isna()
    n_invalidos = mask_invalidos.sum()
    pct = n_invalidos / n_antes * 100
    print(f"  Registros con MCC inválido ({mcc_invalidos} o nulo): {n_invalidos:,} ({pct:.2f}%)")

    # Eliminar
    df_clean = df_clean[~df_clean['mcc'].isin(mcc_invalidos)]
    df_clean = df_clean.dropna(subset=['mcc'])

    # Convertir a int
    df_clean['mcc'] = df_clean['mcc'].astype(int)

    print(f"  Filas restantes: {len(df_clean):,}")
    print(f"  MCCs únicos válidos: {df_clean['mcc'].nunique()}")
    return df_clean


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte los tipos de datos de las columnas principales.
    """
    df_clean = df.copy()

    if 'fecha_trx' in df_clean.columns:
        df_clean['fecha_trx'] = pd.to_datetime(df_clean['fecha_trx'], format='%Y-%m-%d', errors='coerce')
        print(f"  Rango fechas: {df_clean['fecha_trx'].min().date()} a {df_clean['fecha_trx'].max().date()}")

    if 'tpv' in df_clean.columns:
        df_clean['tpv'] = pd.to_numeric(df_clean['tpv'], errors='coerce')

    if 'cantidad_tx' in df_clean.columns:
        df_clean['cantidad_tx'] = pd.to_numeric(df_clean['cantidad_tx'], errors='coerce')

    return df_clean


def preprocess_data(df: pd.DataFrame,
                    columns_to_drop: Optional[List[str]] = None,
                    mapping_comercio: Optional[Dict[str, str]] = None,
                    merchants_to_keep: Optional[List[str]] = None,
                    min_instances: int = 100,
                    mcc_invalidos: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento. Aplica en orden:
      1. Eliminar columnas innecesarias
      2. Eliminar filas con nulos
      3. Mapear id_comercio -> nombre_comercio
      4. Filtrar comercios por lista blanca (COMERCIOS_A_MANTENER)
      5. Eliminar comercios con pocas instancias
      6. Limpiar MCC inválidos
      7. Convertir tipos de datos

    Args:
        df: DataFrame crudo
        columns_to_drop: Columnas a eliminar
        mapping_comercio: Diccionario RUT -> nombre
        merchants_to_keep: Lista blanca de comercios a conservar
        min_instances: Mínimo de filas por comercio
        mcc_invalidos: Lista de MCCs a eliminar

    Returns:
        DataFrame limpio listo para feature engineering
    """
    df_clean = df.copy()

    print(f"\nShape inicial: {df_clean.shape}")

    # 1. Eliminar columnas innecesarias
    if columns_to_drop:
        print("\n[1/7] Eliminando columnas innecesarias...")
        df_clean = drop_unnecessary_columns(df_clean, columns_to_drop)

    # 2. Eliminar nulos
    print("\n[2/7] Eliminando filas con valores nulos...")
    df_clean = drop_nulls(df_clean)

    # 3. Mapear nombres de comercio
    if mapping_comercio:
        print("\n[3/7] Mapeando id_comercio -> nombre_comercio...")
        df_clean = map_merchant_names(df_clean, mapping_comercio)

    # 4. Filtrar comercios por lista blanca
    if merchants_to_keep:
        print("\n[4/7] Filtrando comercios por lista blanca...")
        df_clean = filter_merchants_by_name(df_clean, merchants_to_keep)

    # 5. Eliminar comercios con pocas instancias
    print(f"\n[5/7] Eliminando comercios con menos de {min_instances} instancias...")
    df_clean = filter_merchants_by_min_instances(df_clean, min_instances)

    # 6. Limpiar MCC
    print("\n[6/7] Limpiando columna MCC...")
    df_clean = clean_mcc(df_clean, mcc_invalidos)

    # 7. Convertir tipos
    print("\n[7/7] Convirtiendo tipos de datos...")
    df_clean = convert_dtypes(df_clean)

    print(f"\nShape final: {df_clean.shape}")

    return df_clean


def get_merchant_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el share de TPV por comercio.
    """
    if 'nombre_comercio' not in df.columns or 'tpv' not in df.columns:
        print("  Columnas necesarias no encontradas")
        return pd.DataFrame()

    df_share = df.groupby('nombre_comercio')['tpv'].sum().reset_index()
    tpv_total = df_share['tpv'].sum()
    df_share['pct_del_total'] = df_share['tpv'] / tpv_total
    df_share = df_share.sort_values('pct_del_total', ascending=False)
    df_share['pct_acumulado'] = df_share['pct_del_total'].cumsum()
    df_share['pct_del_total_label'] = (df_share['pct_del_total'] * 100).map('{:,.2f}%'.format)
    return df_share