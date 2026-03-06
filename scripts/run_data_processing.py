#!/usr/bin/env python3
"""
Script para ejecutar procesamiento de datos.
Pasos que ejecuta (en orden):
  1. Cargar base_proyecto_ml_v2.parquet
  2. Eliminar columnas innecesarias
  3. Eliminar filas con nulos
  4. Mapear id_comercio -> nombre_comercio
  5. Filtrar comercios por lista blanca (COMERCIOS_A_MANTENER)
  6. Eliminar comercios con menos de MIN_INSTANCIAS_COMERCIO filas
  7. Limpiar MCCs invalidos (0, 34, nulos)
  8. Convertir tipos de datos
  9. Guardar dataset_inicial_limpio.parquet

Consideraciones adicionales: el dataset inicial tiene el nombre 'base_proyecto_ml_v2.parquet',
cambiar el nombre en el código si se usa otro nombre para el archivo de entrada. Además, recordar
que los parámetros de preprocesamiento (columnas a eliminar, comercios a mantener) se configuran en config/config.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, save_dataset
from src.data.preprocessing import preprocess_data
from config.config import (
    DATA_DIR,
    COLUMNS_TO_DROP,
    MAPPING_COMERCIO,
    COMERCIOS_A_MANTENER,
    MIN_INSTANCIAS_COMERCIO,
    MCC_INVALIDOS,
    MODO_COMERCIOS
)


def main():
    print("=" * 60)
    print("PROCESAMIENTO DE DATOS TPV")
    print("=" * 60)

    # 1. Cargar datos crudos
    print("\n[1] Cargando datos crudos...")
    raw_file = DATA_DIR / 'base_proyecto_ml_v2.parquet' # Aquí cambiar si el archivo de entrada tiene otro nombre
    df = load_raw_data(raw_file)

    if df is None:
        print("Error al cargar datos. Abortando.")
        return

    print(f"  Shape inicial: {df.shape}")

    # 2. Pipeline completo de preprocesamiento
    print("\n[2] Ejecutando pipeline de preprocesamiento...")
    df_clean = preprocess_data(
        df,
        columns_to_drop=COLUMNS_TO_DROP,
        mapping_comercio=MAPPING_COMERCIO,
        merchants_to_keep=COMERCIOS_A_MANTENER,
        min_instances=MIN_INSTANCIAS_COMERCIO,
        mcc_invalidos=MCC_INVALIDOS
    )

    # 3. Guardar
    print("\n[3] Guardando dataset limpio...")
    output_file = DATA_DIR / f'dataset_inicial_limpio_{MODO_COMERCIOS}.parquet'
    save_dataset(df_clean, output_file)

    print(f"\nProcesamiento completado.")
    print(f"  Archivo: {output_file}")
    print(f"  Shape final: {df_clean.shape}")


if __name__ == "__main__":
    main()