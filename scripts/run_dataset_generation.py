#!/usr/bin/env python3
"""
Script para generar dataset de entrenamiento sin ejecutar el entrenamiento.
Útil cuando solo se quiere preparar los datos para backtesting o exploración.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, save_dataset, save_encoder
from src.features.feature_engineering import generar_dataset
from config.config import DATA_DIR, DIAS_PRED, CYBER_EVENTS, COMERCIOS_ANTIGUEDAD, MODO_COMERCIOS
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Generar dataset de entrenamiento')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                        help='Días de predicción (horizonte)')
    parser.add_argument('--forzar', action='store_true',default=False,
                        help='Forzar regeneración aunque el dataset exista')
    args = parser.parse_args()

    print("=" * 60)
    print(" GENERACIÓN DE DATASET TPV")
    print("=" * 60)
    print(f"Dias prediccion : {args.dias_pred}")
    print(f"Modo comercios  : {MODO_COMERCIOS}")

    # Verificar si el archivo existe
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_{MODO_COMERCIOS}.parquet'
    
    if dataset_file.exists() and not args.forzar:
        print(f"\n Dataset ya existe: {dataset_file}")
        print("   Usa --forzar para regenerar")
        return

    # Cargar datos limpios
    print(f"\n Cargando datos limpios...")
    df_raw = load_raw_data(DATA_DIR / f'dataset_inicial_limpio_{MODO_COMERCIOS}.parquet')
    
    if df_raw is None:
        print(" Error: No se encontró dataset_inicial_limpio.parquet")
        print("   Ejecuta run_data_processing.py primero")
        return

    # Generar dataset con features
    print(f"\n Generando features...")
    df_antiguedad = pd.DataFrame(COMERCIOS_ANTIGUEDAD)
    df_final, encoder = generar_dataset(
        df_raw, df_antiguedad,
        dias_prediccion=args.dias_pred,
        cybers_list=CYBER_EVENTS
    )

    # Guardar
    print(f"\n Guardando dataset...")
    save_dataset(df_final, dataset_file)
    save_encoder(encoder, DATA_DIR / f'encoder_comercios_{MODO_COMERCIOS}.joblib')

    print(f"\n Dataset generado exitosamente!")
    print(f"   Archivo: {dataset_file}")
    print(f"   Shape: {df_final.shape}")
    print(f"\n Ahora puedes ejecutar:")
    print(f"   - Entrenamiento: python scripts/run_training.py --dias-pred {args.dias_pred}")
    print(f"   - Backtesting: python scripts/run_backtesting.py --dias-pred {args.dias_pred}")


if __name__ == "__main__":
    main()
