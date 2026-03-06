#!/usr/bin/env python3
"""
Script para generar predicciones con modelos entrenados sobre data nueva.

Flujo:
  1. Carga data limpia (dataset_inicial_limpio.parquet)
  2. Genera features con generar_dataset(..., para_prediccion=True)
     -> mantiene los rows mas recientes aunque tpv_futuro sea NaN
  3. Filtra al ultimo dia disponible por comercio (o la fecha indicada)
  4. Predice con el modelo global cargado
  5. Suma predicciones por comercio (hay un row por MCC)
  6. Guarda resultados en data/predicciones/ como CSV y PDF

Uso:
  python scripts/run_prediction.py \
      --model-path models/catboost_global_2025-12-04_28dias.cbm \
      --fecha-corte 2026-03-05
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data, save_dataset
from src.features.feature_engineering import generar_dataset
from src.models.predict import cargar_modelo, predecir
from src.models.backtesting import _crear_pdf_tabla
from config.config import DATA_DIR, DIAS_PRED, CYBER_EVENTS, COMERCIOS_ANTIGUEDAD, MODO_COMERCIOS


def main():
    parser = argparse.ArgumentParser(description='Predecir TPV con modelo entrenado')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Ruta al modelo entrenado (.cbm, .txt o .json)')
    parser.add_argument('--fecha-corte', type=str, default=None,
                        help='Fecha de prediccion YYYY-MM-DD (default: ultima fecha disponible)')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                        help='Horizonte de prediccion en dias')
    parser.add_argument('--guardar-features', action='store_true',
                        help='Guardar el dataset de features en data/features_prediccion_{dias}dias.parquet')
    args = parser.parse_args()

    print("=" * 60)
    print(" PREDICCION TPV CON MODELO ENTRENADO")
    print("=" * 60)
    print(f"Modelo      : {args.model_path}")
    print(f"Fecha corte : {args.fecha_corte or '(ultima disponible)'}")
    print(f"Horizonte   : {args.dias_pred} dias")

    # --- 1. Cargar data limpia ---
    data_path = DATA_DIR / f'dataset_inicial_limpio_{MODO_COMERCIOS}.parquet'
    print(f"\n Cargando data limpia: {data_path}")
    df_raw = load_raw_data(data_path)
    if df_raw is None:
        print(" Error: No se encontro dataset_inicial_limpio.parquet")
        print("   Ejecuta primero: python scripts/run_data_processing.py")
        sys.exit(1)

    # --- 2. Generar features (modo prediccion: sin dropna del target) ---
    print(f"\n Generando features (modo prediccion)...")
    df_antiguedad = pd.DataFrame(COMERCIOS_ANTIGUEDAD)
    df_features, encoder = generar_dataset(
        df_raw, df_antiguedad,
        dias_prediccion=args.dias_pred,
        cybers_list=CYBER_EVENTS,
        para_prediccion=True
    )
    print(f"   Dataset generado: {df_features.shape}")

    # Guardar features si se solicita
    if args.guardar_features:
        feat_path = DATA_DIR / f'features_prediccion_{args.dias_pred}dias.parquet'
        save_dataset(df_features, feat_path)

    # --- 3. Determinar fecha de corte y filtrar ---
    if args.fecha_corte:
        fecha_corte = pd.to_datetime(args.fecha_corte)
        df_pred = df_features[df_features['fecha_trx'] == fecha_corte]
        if df_pred.empty:
            # Si no hay datos exactos para esa fecha, usar la mas cercana anterior
            df_pred = df_features[df_features['fecha_trx'] <= fecha_corte]
            if df_pred.empty:
                print(f" Error: No hay datos disponibles hasta {args.fecha_corte}")
                sys.exit(1)
            fecha_corte = df_pred['fecha_trx'].max()
            df_pred = df_features[df_features['fecha_trx'] == fecha_corte]
            print(f"   Fecha exacta no disponible. Usando: {fecha_corte.date()}")
    else:
        fecha_corte = df_features['fecha_trx'].max()
        df_pred = df_features[df_features['fecha_trx'] == fecha_corte]

    fecha_fin_ventana = fecha_corte + pd.Timedelta(days=args.dias_pred - 1)
    print(f"   Ventana predicha: {fecha_corte.date()} al {fecha_fin_ventana.date()} ({args.dias_pred} dias)")
    print(f"   Rows a predecir   : {len(df_pred)} (comercio x MCC)")

    # --- 4. Cargar modelo y predecir ---
    print(f"\n Cargando modelo...")
    modelo = cargar_modelo(args.model_path)

    print(f"\n Generando predicciones...")
    pred_series = predecir(modelo, df_pred)

    # --- 5. Agregar por comercio (sumar por id_comercio_num) ---
    df_result = df_pred[['id_comercio_num']].copy()
    df_result['tpv_predicho'] = pred_series.values

    df_agrupado = df_result.groupby('id_comercio_num', as_index=False)['tpv_predicho'].sum()

    # Decodificar nombre del comercio
    df_agrupado['nombre_comercio'] = encoder.inverse_transform(
        df_agrupado['id_comercio_num'].astype(int)
    )

    # Ordenar y seleccionar columnas finales
    df_salida = df_agrupado[
        ['nombre_comercio', 'tpv_predicho']
    ].sort_values('tpv_predicho', ascending=False).reset_index(drop=True)

    print(f"\n Predicciones generadas para {len(df_salida)} comercios")

    # --- 6. Guardar CSV y PDF ---
    output_dir = DATA_DIR / 'predicciones'
    output_dir.mkdir(parents=True, exist_ok=True)

    model_stem = Path(args.model_path).stem
    fecha_str = str(fecha_corte.date())
    base_name = f"predicciones_{fecha_str}_{model_stem}"

    # CSV
    csv_path = output_dir / f"{base_name}.csv"
    df_salida.to_csv(csv_path, index=False)
    print(f"\n CSV guardado: {csv_path}")

    # PDF
    pdf_path = output_dir / f"{base_name}.pdf"
    titulo = (
        f"Prediccion TPV | Ventana: {fecha_str} al {fecha_fin_ventana.date()} ({args.dias_pred} dias) | "
        f"Modelo: {model_stem}"
    )

    # Construir tabla para PDF: headers + filas
    headers = ['Comercio', 'TPV Predicho ($)']
    pdf_data = [headers]
    for _, row in df_salida.iterrows():
        pdf_data.append([
            row['nombre_comercio'],
            f"${row['tpv_predicho']:,.0f}"
        ])
    # Fila TOTAL
    total = df_salida['tpv_predicho'].sum()
    pdf_data.append(['TOTAL', f"${total:,.0f}"])

    _crear_pdf_tabla(
        data=pdf_data,
        titulo=titulo,
        filepath=str(pdf_path),
        orientacion='portrait'
    )
    print(f" PDF guardado: {pdf_path}")

    # Mostrar resumen en consola
    print(f"\n{'='*60}")
    print(f" RESUMEN - Top 5 comercios")
    print(f"{'='*60}")
    for _, row in df_salida.head(5).iterrows():
        print(f"  {row['nombre_comercio']:35} ${row['tpv_predicho']:>15,.0f}")
    print(f"  {'TOTAL':35} ${total:>15,.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
