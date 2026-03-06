#!/usr/bin/env python3
"""
Script para evaluar el modelo de Media Móvil (baseline actual).
Genera PDF profesional con métricas en el mismo formato que backtesting.

Modelo: prediccion_mes = m * x + n
Donde:
- m = promedio de los últimos 28 días
- x = días restantes del mes (desde fecha_prediccion hasta fin de mes)
- n = TPV acumulado del mes hasta ayer
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_training_dataset
from config.config import DATA_DIR, DIAS_PRED


def _formatear_fecha_columna(fecha_str: str) -> str:
    """Convierte '2025-07-01' a '01-07'."""
    try:
        fecha = pd.to_datetime(fecha_str)
        return fecha.strftime('%d-%m')
    except:
        return fecha_str


def _crear_pdf_tabla(data: List[List],
                     titulo: str,
                     filepath: str,
                     color_header: tuple = (46, 125, 50),
                     orientacion: str = 'landscape'):
    """Crea un PDF con una tabla (mismo formato que backtesting).
    
    Args:
        data: Lista de listas, donde la primera sublista es el encabezado.
        titulo: Título del documento.
        filepath: Ruta donde se guardará el PDF.
        color_header: Tupla RGB para el fondo del encabezado (default: verde).
        orientacion: 'portrait' o 'landscape' (default: 'landscape').    
    """
    pagesize = landscape(A4) if orientacion == 'landscape' else A4
    
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
    
    col_widths = [1.5*inch] + [(available_width - 1.5*inch) / (num_cols - 1)] * (num_cols - 1)
    
    table = Table(data, colWidths=col_widths, repeatRows=1)
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(
            color_header[0]/255, color_header[1]/255, color_header[2]/255
        )),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ])
    
    for i in range(1, len(data)):
        if i % 2 == 0:
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.Color(0.95, 0.95, 0.95))
    
    for i, row in enumerate(data):
        if row[0] in ['TOTAL', 'PROMEDIO']:
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.Color(0.3, 0.3, 0.3))
            table_style.add('TEXTCOLOR', (0, i), (-1, i), colors.whitesmoke)
            table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
    
    table.setStyle(table_style)
    elements.append(table)
    
    doc.build(elements)


def calcular_prediccion_media_movil(df_diario: pd.DataFrame, 
                                    fecha_prediccion: str,
                                    dias_horizonte: int = 28) -> Dict:
    """
    Calcula la predicción usando media móvil y extrae el TPV Real usando
    exactamente la misma lógica que el backtesting de Machine Learning.

    Args:
        df_diario: DataFrame con columnas 'fecha_trx', 'tpv' y 'tpv_futuro'.
        fecha_prediccion: Fecha para la cual se quiere hacer la predicción (YYYY-MM-DD).
        dias_horizonte: Días a predecir (horizonte), default 28.
    """
    fecha_pred = pd.to_datetime(fecha_prediccion)
    
    # Media móvil de los últimos 28 días (usando el 'tpv' diario pasado)
    fecha_inicio_ventana = fecha_pred - pd.Timedelta(days=28)
    fecha_fin_ventana = fecha_pred - pd.Timedelta(days=1)
    
    mask_ventana = (df_diario['fecha_trx'] >= fecha_inicio_ventana) & (df_diario['fecha_trx'] <= fecha_fin_ventana)
    tpv_ventana = df_diario[mask_ventana]['tpv'].sum()
    m = tpv_ventana / 28  # Promedio diario
    
    # Predicción: media móvil × días a predecir
    prediccion = m * dias_horizonte
    
    # TPV REAL: Tomamos directamente la suma de 'tpv_futuro' para esa fecha exacta
    # (Esto es idéntico a lo que hace el script de ML en el "Día 1")
    mask_hoy = df_diario['fecha_trx'] == fecha_pred
    tpv_real = float(df_diario[mask_hoy]['tpv_futuro'].sum())
    
    fecha_fin_real = fecha_pred + pd.Timedelta(days=dias_horizonte - 1)
    
    return {
        'fecha_prediccion': fecha_prediccion,
        'fecha_inicio_horizonte': fecha_pred.date(),
        'fecha_fin_horizonte': fecha_fin_real.date(),
        'dias_horizonte': dias_horizonte,
        'media_movil_28d': m,
        'tpv_predicho': prediccion,
        'tpv_real': tpv_real,
        'error_absoluto': abs(prediccion - tpv_real),
        'error_porcentual': abs(prediccion - tpv_real) / tpv_real * 100 if tpv_real > 0 else 0
    }


def evaluar_media_movil(df: pd.DataFrame, 
                       fechas_prediccion: List[str],
                       dias_pred: int = DIAS_PRED ,
                       verbose: bool = True) -> pd.DataFrame:
    """Evalúa el modelo de media móvil en múltiples fechas."""
    
    # Asegurarnos de que fecha_trx sea datetime para evitar problemas de formato
    df['fecha_trx'] = pd.to_datetime(df['fecha_trx'])
    
    # Agrupamos sumando tanto el TPV del día como el TPV Futuro (target)
    df_diario = df.groupby('fecha_trx').agg({
        'tpv': 'sum',
        'tpv_futuro': 'sum'
    }).reset_index()
    
    resultados = []
    
    iterator = tqdm(fechas_prediccion, desc="Evaluando Media Móvil") if verbose else fechas_prediccion
    
    for fecha in iterator:
        try:
            resultado = calcular_prediccion_media_movil(df_diario, fecha, dias_horizonte=dias_pred)
            resultados.append(resultado)
            
            if verbose:
                print(f"\nFecha: {fecha}")
                print(f"  Media móvil (28d): ${resultado['media_movil_28d']/1e9:,.2f}B/día")
                print(f"  Horizonte: {resultado['dias_horizonte']} días")
                print(f"  Período: {resultado['fecha_inicio_horizonte']} a {resultado['fecha_fin_horizonte']}")
                print(f"  Predicción: ${resultado['tpv_predicho']/1e9:,.2f}B")
                print(f"  Real: ${resultado['tpv_real']/1e9:,.2f}B")
                print(f"  Error: {resultado['error_porcentual']:.2f}%")
        
        except Exception as e:
            print(f"Error en fecha {fecha}: {e}")
            continue
    
    df_resultados = pd.DataFrame(resultados)
    return df_resultados


def generar_pdf_metricas(df_resultados: pd.DataFrame, 
                        dias_pred: int,
                        output_dir: Path = None) -> None:
    """Genera PDF con tabla de métricas.
    
    Args:
    - df_resultados: DataFrame con columnas 'fecha_prediccion', 'tpv_real', 'tpv_predicho', 'error_absoluto', 'error_porcentual'.
    - dias_pred: Días de predicción (horizonte).
    - output_dir: Directorio donde se guardará el PDF. Si es None, se guardará en 'results/'.
    """
    if output_dir is None:
        output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Calcular métricas agregadas
    rmse = np.sqrt(np.mean(df_resultados['error_absoluto']**2))
    mae = df_resultados['error_absoluto'].mean()
    mape = df_resultados['error_porcentual'].mean()
    
    y_true = df_resultados['tpv_real']
    y_pred = df_resultados['tpv_predicho']
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print("\n" + "="*70)
    print("RESULTADOS - MODELO MEDIA MÓVIL")
    print("="*70)
    print(f"\nMétricas agregadas ({len(df_resultados)} fechas):")
    print(f"  RMSE: ${rmse/1e9:.2f}B")
    print(f"  MAE:  ${mae/1e9:.2f}B")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\nTPV Real Promedio: ${df_resultados['tpv_real'].mean()/1e9:.2f}B")
    print(f"TPV Pred Promedio: ${df_resultados['tpv_predicho'].mean()/1e9:.2f}B")
    
    # Preparar datos para tabla PDF (formato igual a backtesting)
    fechas_ordenadas = sorted(df_resultados['fecha_prediccion'].tolist())
    cols_fechas = [_formatear_fecha_columna(f) for f in fechas_ordenadas]
    
    titulo = f"Backtesting Media Móvil Métricas {dias_pred} días"
    tabla_data = [['MÉTRICA'] + cols_fechas + ['PROMEDIO']]
    
    # RMSE (en miles de millones)
    fila_rmse = ['RMSE (B)']
    for fecha in fechas_ordenadas:
        error_abs = df_resultados.loc[df_resultados['fecha_prediccion']==fecha, 'error_absoluto'].values[0]
        fila_rmse.append(f"${error_abs/1e9:.2f}B")
    fila_rmse.append(f"${rmse/1e9:.2f}B")
    tabla_data.append(fila_rmse)
    
    # Error %
    fila_err = ['Error %']
    for fecha in fechas_ordenadas:
        err = df_resultados.loc[df_resultados['fecha_prediccion']==fecha, 'error_porcentual'].values[0]
        fila_err.append(f"{err:.2f}%")
    fila_err.append(f"{mape:.2f}%")
    tabla_data.append(fila_err)
    
    # TPV Real
    fila_real = ['TPV Real (B)']
    for fecha in fechas_ordenadas:
        real = df_resultados.loc[df_resultados['fecha_prediccion']==fecha, 'tpv_real'].values[0]
        fila_real.append(f"${real/1e9:.2f}B")
    fila_real.append(f"${y_true.mean()/1e9:.2f}B")
    tabla_data.append(fila_real)
    
    # TPV Pred
    fila_pred = ['TPV Pred (B)']
    for fecha in fechas_ordenadas:
        pred = df_resultados.loc[df_resultados['fecha_prediccion']==fecha, 'tpv_predicho'].values[0]
        fila_pred.append(f"${pred/1e9:.2f}B")
    fila_pred.append(f"${y_pred.mean()/1e9:.2f}B")
    tabla_data.append(fila_pred)
    
    # Generar PDF
    pdf_file = output_dir / f'backtesting_media_movil_metricas_{dias_pred}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_data,
        titulo=titulo,
        filepath=str(pdf_file),
        color_header=(255, 152, 0),  # Naranja para diferenciar de ML
        orientacion='landscape'
    )
    
    print(f"\nPDF generado: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelo de Media Móvil (baseline actual)'
    )
    parser.add_argument('--fechas', type=str, nargs='+', required=True,
                       help='Fechas de predicción (YYYY-MM-DD)')
    parser.add_argument('--dias-pred', type=int, default=DIAS_PRED,
                       help='Días de predicción del dataset')
    parser.add_argument('--no-verbose', action='store_true',
                       help='No mostrar información detallada')
    args = parser.parse_args()
    
    print("="*70)
    print("EVALUACIÓN - MODELO MEDIA MÓVIL")
    print("="*70)
    print(f"Fechas a evaluar: {len(args.fechas)}")
    print(f"Fechas: {args.fechas}")
    
    # Cargar datos
    dataset_file = DATA_DIR / f'dataset_entrenamiento_{args.dias_pred}dias_todos.parquet'
    
    if not dataset_file.exists():
        print(f"\nError: No se encontró {dataset_file}")
        print(f"Ejecuta primero: python scripts/run_dataset_generation.py --dias-pred {args.dias_pred}")
        return
    
    print(f"\nCargando dataset...")
    encoder_file = DATA_DIR / "encoder_comercios_todos.joblib"
    df, _ = load_training_dataset(dataset_file, encoder_path=encoder_file)
    
    # Evaluar modelo
    df_resultados = evaluar_media_movil(
        df,
        fechas_prediccion=args.fechas,
        dias_pred=args.dias_pred,
        verbose=not args.no_verbose
    )
    
    # Generar PDF
    generar_pdf_metricas(df_resultados, args.dias_pred)
    
    print("\nEvaluación completada!")


if __name__ == "__main__":
    main()