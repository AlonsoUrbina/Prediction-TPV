#!/usr/bin/env python3
"""
Script para comparar LightGBM, CatBoost y XGBoost
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_dataset
from src.models.train import entrenar_modelo_global
from src.models.train_catboost import entrenar_modelo_catboost
from src.models.train_xgboost import entrenar_modelo_xgboost
from config.config import DATA_DIR, DIAS_PRED, MODO_COMERCIOS


def _crear_pdf_tabla(data: List[List],
                     titulo: str,
                     filepath: str,
                     color_header: tuple = (52, 73, 94),
                     orientacion: str = 'portrait'):
    pagesize = landscape(A4) if orientacion == 'landscape' else A4
    """Crea un PDF con una tabla estilizada.

    Args:
        data: Lista de listas, donde la primera sublista es el encabezado.
        titulo: Título del documento.
        filepath: Ruta donde se guardará el PDF.
        color_header: Tupla RGB para el fondo del encabezado (default: azul oscuro).
        orientacion: 'portrait' o 'landscape' (default: 'portrait').
    """
    
    doc = SimpleDocTemplate(
        filepath,
        pagesize=pagesize,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.75*inch,
        bottomMargin=0.5*inch
    )
    
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    elements.append(Paragraph(titulo, title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    num_cols = len(data[0])
    available_width = 10 * inch if orientacion == 'landscape' else 7 * inch
    
    col_widths = [2.0*inch] + [(available_width - 2.0*inch) / (num_cols - 1)] * (num_cols - 1)
    
    table = Table(data, colWidths=col_widths, repeatRows=1)
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(
            color_header[0]/255, color_header[1]/255, color_header[2]/255
        )),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ])
    
    for i in range(1, len(data)):
        if i % 2 == 0:
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.Color(0.95, 0.95, 0.95))
    
    # Resaltar fila del mejor modelo
    for i, row in enumerate(data):
        if i > 0 and '⭐' in str(row[0]):
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.Color(0.9, 0.95, 0.9))
            table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
    
    table.setStyle(table_style)
    elements.append(table)
    
    doc.build(elements)


def comparar_modelos(df: pd.DataFrame, 
                    fecha_corte: str, 
                    dias_pred: int = 28,
                    dias_benchmark: int = 7,
                    guardar_graficos: bool = True):
    """
    Compara los 3 modelos de gradient boosting
    
    Args:
        df: DataFrame con features
        fecha_corte: Fecha de corte para test
        dias_pred: Días de predicción
        dias_benchmark: Días de test
        guardar_graficos: Si True, guarda gráficos comparativos
        
    Returns:
        DataFrame con comparación de métricas
    """
    print("\n" + "="*70)
    print(" COMPARACIÓN DE MODELOS: LightGBM vs CatBoost vs XGBoost")
    print("="*70)
    
    resultados = []
    modelos_entrenados = {}
    
    # === LightGBM ===
    print("\n" + "-"*70)
    print(" Entrenando LightGBM...")
    print("-"*70)
    modelo_lgb, metricas_lgb = entrenar_modelo_global(
        df, fecha_corte, dias_pred, dias_benchmark
    )
    metricas_lgb['model'] = 'LightGBM'
    resultados.append(metricas_lgb)
    modelos_entrenados['LightGBM'] = modelo_lgb
    
    # === CatBoost ===
    print("\n" + "-"*70)
    print(" Entrenando CatBoost...")
    print("-"*70)
    modelo_cat, metricas_cat = entrenar_modelo_catboost(
        df, fecha_corte, dias_pred, dias_benchmark
    )
    metricas_cat['model'] = 'CatBoost'
    resultados.append(metricas_cat)
    modelos_entrenados['CatBoost'] = modelo_cat
    
    # === XGBoost ===
    print("\n" + "-"*70)
    print(" Entrenando XGBoost...")
    print("-"*70)
    modelo_xgb, metricas_xgb = entrenar_modelo_xgboost(
        df, fecha_corte, dias_pred, dias_benchmark
    )
    metricas_xgb['model'] = 'XGBoost'
    resultados.append(metricas_xgb)
    modelos_entrenados['XGBoost'] = modelo_xgb
    
    # === Análisis Comparativo ===
    df_comp = pd.DataFrame(resultados)
    df_comp['error_pct'] = abs(df_comp['tpv_pred'] - df_comp['tpv_real']) / df_comp['tpv_real'] * 100
    df_comp['error_abs'] = abs(df_comp['tpv_pred'] - df_comp['tpv_real'])
    
    print("\n" + "="*70)
    print(" RESULTADOS COMPARATIVOS")
    print("="*70)
    
    # Tabla de métricas
    print("\nMetricas Principales:")
    print("-" * 70)
    cols_mostrar = ['model', 'rmse', 'r2', 'error_pct', 'best_iteration']
    print(df_comp[cols_mostrar].to_string(index=False))
    
    # Rankings
    print("\nRankings:")
    print("-" * 70)
    print(f" Mejor RMSE (menor error): {df_comp.loc[df_comp['rmse'].idxmin(), 'model']}")
    print(f"   RMSE: ${df_comp['rmse'].min():,.0f}")
    print(f"\n Mejor R² (mayor ajuste): {df_comp.loc[df_comp['r2'].idxmax(), 'model']}")
    print(f"   R²: {df_comp['r2'].max():.4f}")
    print(f"\n Mejor Error % (menor error relativo): {df_comp.loc[df_comp['error_pct'].idxmin(), 'model']}")
    print(f"   Error %: {df_comp['error_pct'].min():.2f}%")
    
    # Estadísticas adicionales
    print("\nTPV Real vs Predicho:")
    print("-" * 70)
    for _, row in df_comp.iterrows():
        diff = row['tpv_pred'] - row['tpv_real']
        signo = "+" if diff > 0 else ""
        print(f"{row['model']:10} | Real: ${row['tpv_real']:>15,.0f} | "
              f"Pred: ${row['tpv_pred']:>15,.0f} | "
              f"Diff: {signo}${diff:>15,.0f}")
    
    # Velocidad de convergencia
    print("\nVelocidad de Convergencia:")
    print("-" * 70)
    if 'best_iteration' in df_comp.columns:
        for _, row in df_comp.iterrows():
            print(f"{row['model']:10} | {row['best_iteration']:>4} iteraciones")
    
    # === Generar PDF con Tabla Comparativa ===
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Calcular score ponderado para determinar mejor modelo
    df_comp['rmse_norm'] = (df_comp['rmse'] - df_comp['rmse'].min()) / (df_comp['rmse'].max() - df_comp['rmse'].min() + 1e-10)
    df_comp['r2_norm'] = 1 - df_comp['r2']
    df_comp['error_norm'] = (df_comp['error_pct'] - df_comp['error_pct'].min()) / (df_comp['error_pct'].max() - df_comp['error_pct'].min() + 1e-10)
    
    # Score ponderado (50% RMSE, 30% R², 20% Error%)
    df_comp['score_total'] = (
        0.5 * df_comp['rmse_norm'] + 
        0.3 * df_comp['r2_norm'] + 
        0.2 * df_comp['error_norm']
    )
    
    mejor_modelo = df_comp.loc[df_comp['score_total'].idxmin(), 'model']
    
    # Preparar datos para PDF
    titulo = f"Comparación Modelos {fecha_corte} {dias_pred} días"
    tabla_data = [['MODELO', 'RMSE (B)', 'R²', 'Error %', 'TPV Real (B)', 'TPV Pred (B)', 'Iteraciones']]
    
    for _, row in df_comp.sort_values('score_total').iterrows():
        nombre_modelo = row['model']
        if nombre_modelo == mejor_modelo:
            nombre_modelo = f"{nombre_modelo} ⭐"
        
        fila = [
            nombre_modelo,
            f"${row['rmse']/1e9:.2f}B",
            f"{row['r2']:.4f}",
            f"{row['error_pct']:.2f}%",
            f"${row['tpv_real']/1e9:.2f}B",
            f"${row['tpv_pred']/1e9:.2f}B",
            f"{int(row['best_iteration'])}"
        ]
        tabla_data.append(fila)
    
    # Generar PDF
    pdf_file = output_dir / f'comparacion_modelos_{fecha_corte}_{dias_pred}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_data,
        titulo=titulo,
        filepath=str(pdf_file),
        color_header=(149, 117, 205),
        orientacion='landscape'
    )
    
    print(f"\nPDF generado: {pdf_file}")
    
    # === Gráficos Comparativos ===
    if guardar_graficos:
        print("\nGenerando gráficos comparativos...")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Modelos: LightGBM vs CatBoost vs XGBoost', 
                     fontsize=16, fontweight='bold')
        
        # RMSE
        ax1 = axes[0, 0]
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax1.bar(df_comp['model'], df_comp['rmse'], color=colors_bar, alpha=0.7)
        ax1.set_ylabel('RMSE ($)', fontsize=12)
        ax1.set_title('Error RMSE\n(Menor es mejor)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        # R²
        ax2 = axes[0, 1]
        bars = ax2.bar(df_comp['model'], df_comp['r2'], color=colors_bar, alpha=0.7)
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('R² Score\n(Mayor es mejor)', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Error Porcentual
        ax3 = axes[1, 0]
        bars = ax3.bar(df_comp['model'], df_comp['error_pct'], color=colors_bar, alpha=0.7)
        ax3.set_ylabel('Error %', fontsize=12)
        ax3.set_title('Error Porcentual\n(Menor es mejor)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10)
        
        # Iteraciones hasta convergencia
        ax4 = axes[1, 1]
        if 'best_iteration' in df_comp.columns:
            bars = ax4.bar(df_comp['model'], df_comp['best_iteration'], color=colors_bar, alpha=0.7)
            ax4.set_ylabel('Iteraciones', fontsize=12)
            ax4.set_title('Iteraciones hasta Convergencia\n(Menor = más rápido)', 
                         fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Guardar gráfico
        grafico_file = output_dir / f'comparacion_modelos_{fecha_corte}_{dias_pred}dias_graficos.png'
        plt.savefig(grafico_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {grafico_file}")
        plt.close()
    
    # === Guardar CSV ===
    csv_file = output_dir / f'comparacion_modelos_{fecha_corte}_{dias_pred}dias.csv'
    df_comp.to_csv(csv_file, index=False)
    print(f"CSV guardado: {csv_file}")
    
    # === Recomendación ===
    print("\n" + "="*70)
    print(" RECOMENDACIÓN")
    print("="*70)
    
    print(f"\nModelo recomendado: {mejor_modelo}")
    print("\nCriterio: Score ponderado (50% RMSE, 30% R², 20% Error%)")
    print("\nScores totales (menor es mejor):")
    for _, row in df_comp.sort_values('score_total').iterrows():
        print(f"   {row['model']:10} : {row['score_total']:.4f}")
    
    return df_comp


def main():
    parser = argparse.ArgumentParser(description='Comparar modelos de gradient boosting')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                       help='Días de predicción')
    parser.add_argument('--dias-benchmark', type=int, default=7,
                       help='Días de test')
    parser.add_argument('--fecha-corte', type=str, default='2026-01-01',
                       help='Fecha de corte para test')
    parser.add_argument('--sin-graficos', action='store_true',
                       help='No generar gráficos')
    args = parser.parse_args()
    
    # Cargar datos
    print("Cargando dataset...")
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_{MODO_COMERCIOS}.parquet'
    
    if not dataset_file.exists():
        print(f"\nError: No se encontró {dataset_file}")
        print(f"Ejecuta primero: python scripts/run_dataset_generation.py --dias-pred {args.dias_pred}")
        return
    
    encoder_file = DATA_DIR / f"encoder_comercios_{MODO_COMERCIOS}.joblib"
    df, encoder = load_training_dataset(dataset_file, encoder_path=encoder_file)
    
    # Comparar modelos
    df_resultados = comparar_modelos(
        df,
        fecha_corte=args.fecha_corte,
        dias_pred=args.dias_pred,
        dias_benchmark=args.dias_benchmark,
        guardar_graficos=not args.sin_graficos
    )
    
    print("\nComparación completada!")
    print("\nArchivos generados en results/:")
    print(f"  - comparacion_modelos_{args.fecha_corte}_{args.dias_pred}dias.pdf")
    print(f"  - comparacion_modelos_{args.fecha_corte}_{args.dias_pred}dias.csv")
    if not args.sin_graficos:
        print(f"  - comparacion_modelos_{args.fecha_corte}_{args.dias_pred}dias_graficos.png")


if __name__ == "__main__":
    main()
