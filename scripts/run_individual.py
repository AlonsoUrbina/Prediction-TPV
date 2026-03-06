#!/usr/bin/env python3
"""
Script para entrenar modelos individuales (uno por comercio)
Soporta: LightGBM, CatBoost, XGBoost
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_dataset
from src.models.train_individual import entrenar_modelo_individual
from config.config import DATA_DIR, DIAS_PRED, MODO_COMERCIOS


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos individuales')
    parser.add_argument('--model-type', type=str, default='lightgbm',
                       choices=['lightgbm', 'catboost', 'xgboost'],
                       help='Tipo de modelo')
    parser.add_argument('--fecha-corte', type=str, default='2026-01-01',
                       help='Fecha de corte para test')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                       help='Días de predicción')
    parser.add_argument('--dias-benchmark', type=int, default=7,
                       help='Días para ventana de test')
    parser.add_argument('--usar-optuna', action='store_true',
                       help='Usar Optuna para optimizar hiperparámetros')
    parser.add_argument('--no-verbose', action='store_true',
                       help='No mostrar información detallada')
    parser.add_argument('--guardar-modelos', action='store_true',
                       help='Guardar cada modelo individual en models/individual/{model_type}_{fecha}_{dias}dias/')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"MODELOS INDIVIDUALES - {args.model_type.upper()}")
    print("=" * 70)
    print(f"Fecha corte: {args.fecha_corte}")
    print(f"Días predicción: {args.dias_pred}")
    print(f"Días benchmark: {args.dias_benchmark}")
    print(f"Optuna: {'Activado' if args.usar_optuna else 'Desactivado'}")
    print(f"Guardar modelos: {'Si' if args.guardar_modelos else 'No'}")
    
    # Cargar datos
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_{MODO_COMERCIOS}.parquet'
    
    if not dataset_file.exists():
        print(f"\nError: No se encontró {dataset_file}")
        print("   Ejecuta primero: python scripts/run_training.py --regenerar")
        return
    
    print(f"\nCargando dataset...")
    encoder_file = DATA_DIR / f"encoder_comercios_{MODO_COMERCIOS}.joblib"
    df, encoder = load_training_dataset(dataset_file, encoder_path=encoder_file)
    
    # Entrenar modelos individuales
    df_resultados = entrenar_modelo_individual(
        df,
        fecha_corte=args.fecha_corte,
        encoder=encoder,
        dias_val=args.dias_pred,
        dias_benchmark=args.dias_benchmark,
        model_type=args.model_type,
        guardar_modelos=args.guardar_modelos,
        verbose=not args.no_verbose,
        pbar_position=0,
        usar_optuna=args.usar_optuna
    )
    
    if df_resultados is None:
        print("\nNo se generaron predicciones")
        return
    
    # Análisis por comercio
    print("\n" + "=" * 70)
    print("ANÁLISIS POR COMERCIO")
    print("=" * 70)
    
    df_por_comercio = df_resultados.groupby('id_comercio_num').agg({
        'tpv_futuro': 'sum',
        'prediccion_individual': 'sum'
    }).reset_index()
    
    df_por_comercio['error_abs'] = abs(
        df_por_comercio['prediccion_individual'] - df_por_comercio['tpv_futuro']
    )
    df_por_comercio['error_pct'] = (
        df_por_comercio['error_abs'] / df_por_comercio['tpv_futuro'] * 100
    )
    
    # Agregar nombres
    df_por_comercio['nombre'] = encoder.inverse_transform(
        df_por_comercio['id_comercio_num']
    )
    
    # Top 10 mejores
    print("\nTop 10 comercios con MENOR error:")
    top_10_mejores = df_por_comercio.nsmallest(10, 'error_pct')
    for idx, row in top_10_mejores.iterrows():
        print(f"   {row['nombre']:30} | Error: {row['error_pct']:5.2f}% | "
              f"Real: ${row['tpv_futuro']:>12,.0f} | "
              f"Pred: ${row['prediccion_individual']:>12,.0f}")
    
    # Top 10 peores
    print("\nTop 10 comercios con MAYOR error:")
    top_10_peores = df_por_comercio.nlargest(10, 'error_pct')
    for idx, row in top_10_peores.iterrows():
        print(f"   {row['nombre']:30} | Error: {row['error_pct']:5.2f}% | "
              f"Real: ${row['tpv_futuro']:>12,.0f} | "
              f"Pred: ${row['prediccion_individual']:>12,.0f}")

    print("\nModelos individuales completados!")


if __name__ == "__main__":
    main()
