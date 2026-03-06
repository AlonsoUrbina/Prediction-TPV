#!/usr/bin/env python3
"""
Script para ejecutar entrenamiento de modelos.
Soporta: LightGBM, CatBoost, XGBoost.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_dataset
from config.config import (
    DATA_DIR, MODELS_DIR, DIAS_PRED, DIAS_BENCHMARK, MODO_COMERCIOS
)
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción TPV')
    parser.add_argument('--model-type', type=str, default='lightgbm',
                        choices=['lightgbm', 'catboost', 'xgboost', 'all'],
                        help='Tipo de modelo a entrenar')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                        help='Días de predicción (horizonte)')
    parser.add_argument('--dias-benchmark', type=int, default=DIAS_BENCHMARK,
                        help='Días del período de test/benchmark')
    parser.add_argument('--fecha-corte', type=str, default='2025-11-01',
                        help='Primer día del período de test (YYYY-MM-DD)')
    parser.add_argument('--usar-optuna', action='store_true',
                        help='Optimizar hiperparámetros con Optuna')
    parser.add_argument('--optuna-trials', type=int, default=20,
                        help='Número de trials de Optuna (default: 20)')
    args = parser.parse_args()

    print("=" * 60)
    print(" ENTRENAMIENTO DE MODELO TPV")
    print("=" * 60)
    print(f"Modelo(s)       : {args.model_type.upper()}")
    print(f"Fecha corte     : {args.fecha_corte}")
    print(f"Días predicción : {args.dias_pred}")
    print(f"Días benchmark  : {args.dias_benchmark}")
    if args.usar_optuna:
        print(f"Optuna          : Activado ({args.optuna_trials} trials)")

    # 1. Cargar dataset (ya no lo genera automáticamente)
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_{MODO_COMERCIOS}.parquet'

    if not dataset_file.exists():
        print(f"\nError: No se encontró el dataset")
        print(f"   {dataset_file}")
        print(f"\nGenera el dataset primero con:")
        print(f"   python scripts/run_dataset_generation.py --dias-pred {args.dias_pred}")
        return

    print(f"\nCargando dataset existente...")
    encoder_file = DATA_DIR / f"encoder_comercios_{MODO_COMERCIOS}.joblib"
    df_final, encoder = load_training_dataset(dataset_file, encoder_path=encoder_file)

    # 2. Determinar qué modelos entrenar
    models_to_train = ['lightgbm', 'catboost', 'xgboost'] \
        if args.model_type == 'all' else [args.model_type]

    # 3. Entrenar cada modelo
    resultados = []

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f" Entrenando: {model_type.upper()}")
        print('='*60)

        if model_type == 'lightgbm':
            from src.models.train import entrenar_modelo_global, guardar_modelo
            modelo, metricas = entrenar_modelo_global(
                df_final,
                fecha_corte=args.fecha_corte,
                dias_pred=args.dias_pred,
                dias_benchmark=args.dias_benchmark,
                usar_optuna=args.usar_optuna,
                optuna_trials=args.optuna_trials
            )
            model_file = MODELS_DIR / f'lgbm_global_{MODO_COMERCIOS}_{args.fecha_corte}_{args.dias_pred}dias.txt'
            guardar_modelo(modelo, model_file)

        elif model_type == 'catboost':
            from src.models.train_catboost import entrenar_modelo_catboost, guardar_modelo_catboost
            modelo, metricas = entrenar_modelo_catboost(
                df_final,
                fecha_corte=args.fecha_corte,
                dias_pred=args.dias_pred,
                dias_benchmark=args.dias_benchmark,
                usar_optuna=args.usar_optuna,
                optuna_trials=args.optuna_trials
            )
            model_file = MODELS_DIR / f'catboost_global_{MODO_COMERCIOS}_{args.fecha_corte}_{args.dias_pred}dias.cbm'
            guardar_modelo_catboost(modelo, model_file)

        elif model_type == 'xgboost':
            from src.models.train_xgboost import entrenar_modelo_xgboost, guardar_modelo_xgboost
            modelo, metricas = entrenar_modelo_xgboost(
                df_final,
                fecha_corte=args.fecha_corte,
                dias_pred=args.dias_pred,
                dias_benchmark=args.dias_benchmark,
                usar_optuna=args.usar_optuna,
                optuna_trials=args.optuna_trials
            )
            model_file = MODELS_DIR / f'xgboost_global_{MODO_COMERCIOS}_{args.fecha_corte}_{args.dias_pred}dias.json'
            guardar_modelo_xgboost(modelo, model_file)

        metricas['model_type'] = model_type
        metricas['model_file'] = str(model_file)
        resultados.append(metricas)

    # 4. Resumen final
    if len(resultados) > 1:
        print("\n" + "="*60)
        print(" COMPARACIÓN DE MODELOS")
        print("="*60)

        df_comp = pd.DataFrame(resultados)
        df_comp['error_pct'] = abs(df_comp['tpv_pred'] - df_comp['tpv_real']) \
                               / df_comp['tpv_real'] * 100

        print("\nMétricas:")
        print(df_comp[['model_type', 'rmse', 'r2', 'error_pct']].to_string(index=False))

        print("\nMejor modelo por métrica:")
        print(f"   Menor RMSE : {df_comp.loc[df_comp['rmse'].idxmin(),  'model_type'].upper()}")
        print(f"   Mayor R²   : {df_comp.loc[df_comp['r2'].idxmax(),    'model_type'].upper()}")

        comp_file = MODELS_DIR / f'comparacion_modelos_{args.fecha_corte}_{args.dias_pred}dias.csv'
        df_comp.to_csv(comp_file, index=False)
        print(f"\nComparación guardada en: {comp_file}")

    print("\nEntrenamiento completado!")


if __name__ == "__main__":
    main()