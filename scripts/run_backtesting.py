#!/usr/bin/env python3
"""
Script para ejecutar backtesting de modelos
Soporta modelos globales e individuales con LightGBM, CatBoost y XGBoost
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_dataset
from src.models.backtesting import ejecutar_backtesting_global, ejecutar_backtesting_individual
from config.config import DATA_DIR, DIAS_PRED, MODO_COMERCIOS


def main():
    parser = argparse.ArgumentParser(description='Ejecutar backtesting de modelos')
    parser.add_argument('--model-type', type=str, default='lightgbm',
                       choices=['lightgbm', 'catboost', 'xgboost', 'all'],
                       help='Tipo de modelo para backtesting (o "all" para los 3)')
    parser.add_argument('--modo', type=str, default='global',
                       choices=['global', 'individual'],
                       help='Modo de entrenamiento (global o individual por comercio)')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                       help='Días de predicción')
    parser.add_argument('--dias-benchmark', type=int, default=7,
                       help='Días para ventana de test')
    parser.add_argument('--usar-optuna', action='store_true',
                       help='Usar Optuna para modelos individuales')
    parser.add_argument('--fechas', type=str, nargs='+',
                       default=[
                           '2025-07-01', '2025-08-01', '2025-09-01',
                           '2025-10-01', '2025-11-01', '2025-12-01'
                       ],
                       help='Fechas de corte para backtesting')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BACKTESTING DE MODELOS")
    print("=" * 70)
    print(f"Modelo: {args.model_type.upper()}")
    print(f"Modo: {args.modo.upper()}")
    print(f"Fechas: {len(args.fechas)} fechas de corte")
    print(f"Días predicción: {args.dias_pred}")
    print(f"Días benchmark: {args.dias_benchmark}")
    if args.modo == 'individual':
        print(f"Optuna: {'Activado' if args.usar_optuna else 'Desactivado'}")
    
    # Cargar datos
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_{MODO_COMERCIOS}.parquet'
    
    if not dataset_file.exists():
        print(f"\nError: No se encontró {dataset_file}")
        print(f"   Ejecuta primero: python scripts/run_dataset_generation.py --dias-pred {args.dias_pred}")
        return
    
    print(f"\nCargando dataset...")
    encoder_file = DATA_DIR / f"encoder_comercios_{MODO_COMERCIOS}.joblib"
    df, encoder = load_training_dataset(dataset_file, encoder_path=encoder_file)
    
    # Determinar qué modelos ejecutar
    models_to_run = ['lightgbm', 'catboost', 'xgboost'] if args.model_type == 'all' else [args.model_type]
    
    # Ejecutar backtesting para cada modelo
    for model_type in models_to_run:
        if len(models_to_run) > 1:
            print("\n" + "=" * 70)
            print(f" EJECUTANDO: {model_type.upper()}")
            print("=" * 70)
        
        # Ejecutar backtesting según modo
        if args.modo == 'global':
            df_metricas, df_comercios_pct, df_comercios_monto = ejecutar_backtesting_global(
                df,
                fechas_corte=args.fechas,
                encoder=encoder,
                dias_testeo=args.dias_pred,
                dias_benchmark=args.dias_benchmark,
                model_type=model_type
            )
            
            # Mostrar resumen
            print("\n" + "=" * 70)
            print(f"RESUMEN FINAL - {model_type.upper()}")
            print("=" * 70)
            print("\nEstadísticas por métrica:")
            print(f"  RMSE: ${df_metricas['rmse'].mean():,.0f} ± ${df_metricas['rmse'].std():,.0f}")
            print(f"  R²: {df_metricas['r2'].mean():.4f} ± {df_metricas['r2'].std():.4f}")
            print(f"  Error %: {df_metricas['error_pct'].mean():.2f}% ± {df_metricas['error_pct'].std():.2f}%")
            
            print("\nMejor fecha (menor RMSE):")
            mejor = df_metricas.loc[df_metricas['rmse'].idxmin()]
            print(f"  {mejor['fecha_corte']}: RMSE=${mejor['rmse']:,.0f}, R²={mejor['r2']:.4f}")
            
            print("\nPeor fecha (mayor RMSE):")
            peor = df_metricas.loc[df_metricas['rmse'].idxmax()]
            print(f"  {peor['fecha_corte']}: RMSE=${peor['rmse']:,.0f}, R²={peor['r2']:.4f}")
            
            if not df_comercios_pct.empty:
                print(f"\nComercios analizados: {len(df_comercios_pct)}")
                print("\nTop 5 comercios con menor error promedio absoluto:")
                top5 = df_comercios_pct.head(5)
                for _, row in top5.iterrows():
                    print(f"  {row['nombre_comercio']:30} | Error: {row['PROM ABS']:.2f}%")
            
        else:  # individual
            df_metricas, df_pct, df_monto = ejecutar_backtesting_individual(
                df,
                fechas_corte=args.fechas,
                encoder=encoder,
                dias_testeo=args.dias_pred,
                dias_benchmark=args.dias_benchmark,
                model_type=model_type,
                usar_optuna=args.usar_optuna
            )
            
            if not df_metricas.empty:
                print("\n" + "=" * 70)
                print(f"RESUMEN FINAL - {model_type.upper()}")
                print("=" * 70)
                print("\nEstadísticas por métrica:")
                print(f"  RMSE: ${df_metricas['rmse'].mean():,.0f} ± ${df_metricas['rmse'].std():,.0f}")
                print(f"  R²: {df_metricas['r2'].mean():.4f} ± {df_metricas['r2'].std():.4f}")
                print(f"  Error %: {df_metricas['error_pct'].mean():.2f}% ± {df_metricas['error_pct'].std():.2f}%")
                
            if not df_pct.empty:
                print(f"\nComercios evaluados: {len(df_pct)}")
                print(f"\nTop 5 comercios con menor error:")
                top5 = df_pct.head(5)
                for _, row in top5.iterrows():
                    print(f"  {row['nombre_comercio']:30} | MAE: {row['MAE']:.2f}%")
    
    print("\nBacktesting completado!")


if __name__ == "__main__":
    main()
