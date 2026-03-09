"""
Módulo para backtesting de modelos.
Permite evaluar modelos en múltiples fechas de corte.
Soporta: LightGBM, CatBoost, XGBoost (modo global e individual).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import gc
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


# Mapeo de meses (para nombres de columnas)
MESES_ESP = {
    1: 'ENE', 2: 'FEB', 3: 'MAR', 4: 'ABR', 5: 'MAY', 6: 'JUN',
    7: 'JUL', 8: 'AGO', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DIC'
}


def _formatear_fecha_columna(fecha_str: str, usar_mes_esp: bool = False) -> str:
    """
    Convierte fecha de '2025-07-01' a '01-07' o 'JUL' según se indique.
    
    Args:
        fecha_str: Fecha en formato 'YYYY-MM-DD'.
        usar_mes_esp: Si True, devuelve 'JUL' en lugar de '01-07'.
    
    Returns:
        Fecha en formato 'DD-MM' o nombre del mes.
    """
    try:
        fecha = pd.to_datetime(fecha_str)
        if usar_mes_esp:
            return MESES_ESP[fecha.month]
        else:
            return fecha.strftime('%d-%m')
    except Exception as e:
        print(f"Error formateando fecha '{fecha_str}': {e}")
        return fecha_str  # Devolver sin formatear en caso de error


def _crear_pdf_tabla(data: List[List],
                     titulo: str,
                     filepath: str,
                     color_header: tuple = (52, 152, 219),
                     orientacion: str = 'landscape'):
    """
    Crea un PDF con una tabla profesional.
    
    Args:
        data:         Lista de listas con los datos (primera fila = headers).
        titulo:       Título a mostrar encima de la tabla.
        filepath:     Ruta donde guardar el PDF.
        color_header: Color RGB para la cabecera.
        orientacion:  'landscape' o 'portrait'.
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
    
    # Resaltar fila TOTAL
    for i, row in enumerate(data):
        if row[0] == 'TOTAL':
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.Color(0.3, 0.3, 0.3))
            table_style.add('TEXTCOLOR', (0, i), (-1, i), colors.whitesmoke)
            table_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')
    
    # Resaltar columnas PROM y PROM ABS si existen
    if 'PROM' in data[0]:
        col_prom = data[0].index('PROM')
        for i in range(1, len(data)):
            if data[i][0] != 'TOTAL':
                table_style.add('BACKGROUND', (col_prom, i), (col_prom, i), colors.Color(0.89, 0.91, 0.91))
                table_style.add('FONTNAME', (col_prom, i), (col_prom, i), 'Helvetica-Bold')
    
    if 'PROM ABS' in data[0]:
        col_promabs = data[0].index('PROM ABS')
        for i in range(1, len(data)):
            if data[i][0] != 'TOTAL':
                table_style.add('BACKGROUND', (col_promabs, i), (col_promabs, i), colors.Color(0.93, 0.99, 0.98))
                table_style.add('FONTNAME', (col_promabs, i), (col_promabs, i), 'Helvetica-Bold')
    
    table.setStyle(table_style)
    elements.append(table)
    
    doc.build(elements)


def ejecutar_backtesting_global(df: pd.DataFrame,
                                fechas_corte: List[str],
                                encoder,
                                dias_testeo: int = 28,
                                dias_benchmark: int = 7,
                                model_type: str = 'lightgbm') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta backtesting con modelo GLOBAL en múltiples fechas.
    Genera 3 PDFs:
      1. Tabla de métricas globales (RMSE, R², Error%, TPV Real/Pred)
      2. Tabla por comercio - Errores Porcentuales
      3. Tabla por comercio - Diferencias Promedio Diarias en Miles de Millones (B)

    Args:
        df:            DataFrame con features.
        fechas_corte:  Lista de fechas de inicio del período de test.
        encoder:       LabelEncoder de comercios.
        dias_testeo:   Días de predicción (horizonte).
        dias_benchmark:Número de días del período de test.
        model_type:    'lightgbm', 'catboost' o 'xgboost'.

    Returns:
        Tupla (df_metricas, df_comercios_pct, df_comercios_monto)
    """
    print(f"\nBACKTESTING - Modelo Global {model_type.upper()}")
    print(f"   Fechas a evaluar  : {len(fechas_corte)}")
    print(f"   Días de predicción: {dias_testeo}")
    print(f"   Días de benchmark : {dias_benchmark}")

    if model_type == 'lightgbm':
        from src.models.train import entrenar_modelo_global
    elif model_type == 'catboost':
        from src.models.train_catboost import entrenar_modelo_catboost as entrenar_modelo_global
    elif model_type == 'xgboost':
        from src.models.train_xgboost import entrenar_modelo_xgboost as entrenar_modelo_global
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    resultados_metricas = {}
    datos_por_comercio = []

    for fecha in tqdm(fechas_corte, desc="Backtesting Global"):
        try:
            modelo, metricas = entrenar_modelo_global(
                df,
                fecha_corte=fecha,
                dias_pred=dias_testeo,
                dias_benchmark=dias_benchmark,
            )

            # Métricas globales del período completo (promedio de dias_benchmark)
            resultados_metricas[fecha] = {
                'rmse'     : metricas['rmse'],
                'r2'       : metricas['r2'],
                'tpv_real' : metricas['tpv_real'],
                'tpv_pred' : metricas['tpv_pred'],
                'error_pct': abs(metricas['tpv_pred'] - metricas['tpv_real'])
                             / metricas['tpv_real'] * 100,
            }

            # Generar predicciones para datos de test
            from src.models.predict import predecir
            fecha_dt = pd.to_datetime(fecha)
            fecha_fin_test = fecha_dt + pd.Timedelta(days=dias_benchmark - 1)
            
            mask_test = (df['fecha_trx'] >= fecha_dt) & (df['fecha_trx'] <= fecha_fin_test)
            df_test = df[mask_test].copy()
            
            # Calcular métricas solo para el primer día (fecha_corte)
            mask_primer_dia = df['fecha_trx'] == fecha_dt
            df_primer_dia = df[mask_primer_dia].copy()
            
            if len(df_primer_dia) > 0:
                from sklearn.metrics import mean_squared_error, r2_score
                preds_dia = predecir(modelo, df_primer_dia, model_type=model_type)
                df_primer_dia['prediccion'] = preds_dia
                
                y_real_dia = df_primer_dia['tpv_futuro']
                y_pred_dia = df_primer_dia['prediccion']
                
                rmse_dia = np.sqrt(mean_squared_error(y_real_dia, y_pred_dia))
                r2_dia = r2_score(y_real_dia, y_pred_dia)
                tpv_real_dia = float(y_real_dia.sum())
                tpv_pred_dia = float(y_pred_dia.sum())
                error_pct_dia = abs(tpv_pred_dia - tpv_real_dia) / tpv_real_dia * 100 if tpv_real_dia > 0 else 0
                
                resultados_metricas[fecha]['rmse_dia1'] = rmse_dia
                resultados_metricas[fecha]['r2_dia1'] = r2_dia
                resultados_metricas[fecha]['tpv_real_dia1'] = tpv_real_dia
                resultados_metricas[fecha]['tpv_pred_dia1'] = tpv_pred_dia
                resultados_metricas[fecha]['error_pct_dia1'] = error_pct_dia
            
            if len(df_test) > 0:
                predicciones = predecir(modelo, df_test, model_type=model_type)
                df_test['prediccion'] = predicciones
                
                # Agrupar por comercio
                resumen = df_test.groupby('id_comercio_num').agg({
                    'tpv_futuro': 'sum',
                    'prediccion': 'sum'
                }).reset_index()
                
                resumen['fecha_corte'] = fecha
                resumen['diff'] = (resumen['prediccion'] - resumen['tpv_futuro']) / dias_benchmark
                resumen['error_pct'] = (resumen['diff'] * dias_benchmark / resumen['tpv_futuro'].replace(0, np.nan)) * 100
                resumen['error_pct'] = resumen['error_pct'].fillna(0)
                
                datos_por_comercio.append(resumen)

            del modelo
            gc.collect()

        except Exception as e:
            print(f"\nError en fecha {fecha}: {e}")
            continue

    # ========== TABLA 1: MÉTRICAS GLOBALES ==========
    df_metricas = pd.DataFrame(resultados_metricas).T
    df_metricas.index.name = 'fecha_corte'
    df_metricas = df_metricas.reset_index()

    print(f"\nMÉTRICAS GLOBALES:")
    print(f"   RMSE Promedio   : ${df_metricas['rmse'].mean()/1e9:.2f}B")
    print(f"   R² Promedio     : {df_metricas['r2'].mean():.4f}")
    print(f"   Error % Promedio: {df_metricas['error_pct'].mean():.2f}%")

    # ========== TABLA 2 y 3: POR COMERCIO ==========
    if datos_por_comercio:
        df_comercios_all = pd.concat(datos_por_comercio, ignore_index=True)
        
        # Matrices pivot: comercio x fecha
        df_comercios_pct = df_comercios_all.pivot_table(
            index='id_comercio_num',
            columns='fecha_corte',
            values='error_pct',
            aggfunc='first'
        ).reset_index()
        
        df_comercios_monto = df_comercios_all.pivot_table(
            index='id_comercio_num',
            columns='fecha_corte',
            values='diff',
            aggfunc='first'
        ).reset_index()
        
        # Agregar nombres de comercio
        def agregar_nombres(df_matrix):
            if encoder:
                try:
                    df_matrix['nombre_comercio'] = encoder.inverse_transform(
                        df_matrix['id_comercio_num'].astype(int).values
                    )
                except Exception:
                    df_matrix['nombre_comercio'] = "ID_" + df_matrix['id_comercio_num'].astype(str)
            else:
                df_matrix['nombre_comercio'] = "ID_" + df_matrix['id_comercio_num'].astype(str)
            return df_matrix
        
        df_comercios_pct = agregar_nombres(df_comercios_pct)
        df_comercios_monto = agregar_nombres(df_comercios_monto)
        
        # Obtener columnas de fechas y renombrarlas a nombres de meses
        cols_fechas = [c for c in df_comercios_pct.columns if c not in ['id_comercio_num', 'nombre_comercio']]
        
        # Ordenar columnas cronológicamente
        cols_fechas_ordenadas = sorted(cols_fechas, key=lambda x: pd.to_datetime(x))
        
        # Mapeo fecha -> día-mes (01-01, 01-02, etc.)
        col_mapping = {fecha: _formatear_fecha_columna(fecha, usar_mes_esp=False) for fecha in cols_fechas_ordenadas}
        
        df_comercios_pct.rename(columns=col_mapping, inplace=True)
        df_comercios_monto.rename(columns=col_mapping, inplace=True)
        
        cols_meses = [col_mapping[f] for f in cols_fechas_ordenadas]
        
        # Calcular promedios
        df_comercios_pct['PROM'] = df_comercios_pct[cols_meses].mean(axis=1)
        df_comercios_pct['PROM ABS'] = df_comercios_pct[cols_meses].abs().mean(axis=1)
        
        df_comercios_monto['PROM'] = df_comercios_monto[cols_meses].mean(axis=1)
        
        # Ordenar por PROM ABS
        df_comercios_pct = df_comercios_pct.sort_values('PROM ABS', ascending=True).reset_index(drop=True)
        df_comercios_monto = df_comercios_monto.sort_values('PROM', key=abs, ascending=False).reset_index(drop=True)
        
        # Reordenar columnas
        df_comercios_pct = df_comercios_pct[['nombre_comercio'] + cols_meses + ['PROM', 'PROM ABS']]
        df_comercios_monto = df_comercios_monto[['nombre_comercio'] + cols_meses + ['PROM']]
        
        # Fila TOTAL - usar métricas globales reales, no promedios de comercios
        # Para cada fecha, calcular el error global real
        total_pct_valores = []
        total_monto_valores = []
        
        for fecha_orig in cols_fechas_ordenadas:
            metricas_fecha = resultados_metricas[fecha_orig]
            tpv_real = metricas_fecha['tpv_real']
            tpv_pred = metricas_fecha['tpv_pred']
            
            # Error porcentual global
            error_pct_global = ((tpv_pred - tpv_real) / tpv_real * 100) if tpv_real > 0 else 0
            total_pct_valores.append(error_pct_global)
            
            # Diferencia absoluta (promedio diario)
            diff_total = (tpv_pred - tpv_real) / dias_benchmark
            total_monto_valores.append(diff_total)
        
        # Promedios para columnas PROM y PROM ABS
        prom_pct = sum(total_pct_valores) / len(total_pct_valores)
        prom_abs_pct = sum(abs(x) for x in total_pct_valores) / len(total_pct_valores)
        prom_monto = sum(total_monto_valores) / len(total_monto_valores)
        
        total_pct = ['TOTAL'] + total_pct_valores + [prom_pct, prom_abs_pct]
        total_monto = ['TOTAL'] + total_monto_valores + [prom_monto]
    else:
        df_comercios_pct = pd.DataFrame()
        df_comercios_monto = pd.DataFrame()

    # ========== GENERAR PDFs ==========
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    model_name = model_type.capitalize()

    # PDF 1: Métricas globales
    titulo_metricas = f"Backtesting Global {model_name} Métricas ({dias_testeo} días)"
    tabla_metricas = [['MÉTRICA'] + [_formatear_fecha_columna(f) for f in fechas_corte] + ['PROMEDIO']]
    
    # --- Métricas del PERÍODO COMPLETO (promedio de dias_benchmark) ---
    fila_rmse = ['RMSE Período (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'rmse'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_rmse.append(f"${df_metricas['rmse'].mean()/1e9:.2f}B")
    tabla_metricas.append(fila_rmse)
    
    fila_r2 = ['R2 Período'] + [f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'r2'].values[0]:.4f}" for f in fechas_corte]
    fila_r2.append(f"{df_metricas['r2'].mean():.4f}")
    tabla_metricas.append(fila_r2)
    
    fila_err = ['Error % Período'] + [f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'error_pct'].values[0]:.2f}%" for f in fechas_corte]
    fila_err.append(f"{df_metricas['error_pct'].mean():.2f}%")
    tabla_metricas.append(fila_err)
    
    # TPV en miles de millones (B)
    fila_real = ['TPV Real (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_real'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_real.append(f"${df_metricas['tpv_real'].mean()/1e9:.2f}B")
    tabla_metricas.append(fila_real)
    
    fila_pred = ['TPV Pred (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_pred'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_pred.append(f"${df_metricas['tpv_pred'].mean()/1e9:.2f}B")
    tabla_metricas.append(fila_pred)
    
    # --- Métricas del PRIMER DÍA (solo fecha_corte) ---
    if 'rmse_dia1' in df_metricas.columns:
        tabla_metricas.append(['--- DÍA 1 ---'] + ['---'] * len(fechas_corte) + ['---'])
        
        fila_rmse_d1 = ['RMSE Día 1 (B)'] + [
            f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'rmse_dia1'].values[0]/1e9:.2f}B" 
            if not pd.isna(df_metricas.loc[df_metricas['fecha_corte']==f, 'rmse_dia1'].values[0]) 
            else "N/A" 
            for f in fechas_corte
        ]
        fila_rmse_d1.append(f"${df_metricas['rmse_dia1'].mean()/1e9:.2f}B")
        tabla_metricas.append(fila_rmse_d1)
        
        fila_r2_d1 = ['R2 Día 1'] + [
            f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'r2_dia1'].values[0]:.4f}" 
            if not pd.isna(df_metricas.loc[df_metricas['fecha_corte']==f, 'r2_dia1'].values[0]) 
            else "N/A" 
            for f in fechas_corte
        ]
        fila_r2_d1.append(f"{df_metricas['r2_dia1'].mean():.4f}")
        tabla_metricas.append(fila_r2_d1)
        
        fila_err_d1 = ['Error % Día 1'] + [
            f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'error_pct_dia1'].values[0]:.2f}%" 
            if not pd.isna(df_metricas.loc[df_metricas['fecha_corte']==f, 'error_pct_dia1'].values[0]) 
            else "N/A" 
            for f in fechas_corte
        ]
        fila_err_d1.append(f"{df_metricas['error_pct_dia1'].mean():.2f}%")
        tabla_metricas.append(fila_err_d1)
        
        fila_real_d1 = ['TPV Real Día 1 (B)'] + [
            f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_real_dia1'].values[0]/1e9:.2f}B" 
            if not pd.isna(df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_real_dia1'].values[0]) 
            else "N/A" 
            for f in fechas_corte
        ]
        fila_real_d1.append(f"${df_metricas['tpv_real_dia1'].mean()/1e9:.2f}B")
        tabla_metricas.append(fila_real_d1)
        
        fila_pred_d1 = ['TPV Pred Día 1 (B)'] + [
            f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_pred_dia1'].values[0]/1e9:.2f}B" 
            if not pd.isna(df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_pred_dia1'].values[0]) 
            else "N/A" 
            for f in fechas_corte
        ]
        fila_pred_d1.append(f"${df_metricas['tpv_pred_dia1'].mean()/1e9:.2f}B")
        tabla_metricas.append(fila_pred_d1)
    
    pdf_metricas = output_dir / f'backtesting_global_{model_type}_metricas_{dias_testeo}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_metricas,
        titulo=titulo_metricas,
        filepath=str(pdf_metricas),
        color_header=(41, 128, 185),
        orientacion='landscape'
    )
    
    print(f"\nPDFs generados:")
    print(f"   [1] Métricas globales: {pdf_metricas}")

    # PDF 2 y 3: Por comercio
    if not df_comercios_pct.empty:
        # PDF 2: Porcentual
        titulo_pct = f"Backtesting Global {model_name} Por Comercio Porcentual {dias_testeo} días"
        tabla_pct = [['COMERCIO'] + cols_meses + ['PROM', 'PROM ABS']]
        
        for _, row in df_comercios_pct.iterrows():
            fila = [row['nombre_comercio']]
            fila += [f"{row[m]:+.1f}%" if pd.notna(row[m]) else "-" for m in cols_meses]
            fila += [f"{row['PROM']:+.1f}%", f"{row['PROM ABS']:.1f}%"]
            tabla_pct.append(fila)
        
        fila_total_pct = ['TOTAL']
        fila_total_pct += [f"{v:+.1f}%" for v in total_pct[1:-2]]
        fila_total_pct += [f"{total_pct[-2]:+.1f}%", f"{total_pct[-1]:.1f}%"]
        tabla_pct.append(fila_total_pct)
        
        pdf_pct = output_dir / f'backtesting_global_{model_type}_por_comercio_porcentual_{dias_testeo}dias.pdf'
        _crear_pdf_tabla(
            data=tabla_pct,
            titulo=titulo_pct,
            filepath=str(pdf_pct),
            color_header=(52, 152, 219),
            orientacion='landscape'
        )
        print(f"   [2] Por comercio (%%): {pdf_pct}")
        
        # PDF 3: Cantidad en Miles de Millones (B) - Promedio Diario
        titulo_monto = f"Backtesting Global {model_name} Por Comercio Miles Millones Promedio Diario {dias_testeo} días"
        tabla_monto = [['COMERCIO'] + cols_meses + ['PROM']]
        
        for _, row in df_comercios_monto.iterrows():
            fila = [row['nombre_comercio']]
            fila += [f"{row[m]/1e9:+.1f}B" if pd.notna(row[m]) else "-" for m in cols_meses]
            fila += [f"{row['PROM']/1e9:+.1f}B"]
            tabla_monto.append(fila)
        
        fila_total_monto = ['TOTAL']
        fila_total_monto += [f"{v/1e9:+.1f}B" for v in total_monto[1:-1]]
        fila_total_monto += [f"{total_monto[-1]/1e9:+.1f}B"]
        tabla_monto.append(fila_total_monto)
        
        pdf_monto = output_dir / f'backtesting_global_{model_type}_por_comercio_cantidad_{dias_testeo}dias.pdf'
        _crear_pdf_tabla(
            data=tabla_monto,
            titulo=titulo_monto,
            filepath=str(pdf_monto),
            color_header=(39, 174, 96),
            orientacion='landscape'
        )
        print(f"   [3] Por comercio (B, promedio diario): {pdf_monto}")
    
    return df_metricas, df_comercios_pct, df_comercios_monto


def ejecutar_backtesting_individual(df: pd.DataFrame,
                                    fechas_corte: List[str],
                                    encoder,
                                    dias_testeo: int = 28,
                                    dias_benchmark: int = 7,
                                    model_type: str = 'lightgbm',
                                    usar_optuna: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta backtesting con modelos INDIVIDUALES (uno por comercio).
    Genera 3 PDFs:
      1. Tabla de métricas globales (RMSE, R², Error%, TPV Real/Pred)
      2. Tabla por comercio - Errores Porcentuales
      3. Tabla por comercio - Diferencias Promedio Diarias en Miles de Millones (B)

    Args:
        df:            DataFrame con features.
        fechas_corte:  Lista de fechas de corte.
        encoder:       LabelEncoder de comercios.
        dias_testeo:   Días de predicción (horizonte).
        dias_benchmark:Días de benchmark.
        model_type:    'lightgbm', 'catboost' o 'xgboost'.
        usar_optuna:   Si True, optimiza hiperparámetros con Optuna.

    Returns:
        Tupla (df_metricas, df_pct, df_monto) con resultados.
    """
    from src.models.train_individual import entrenar_modelo_individual

    print(f"\nBACKTESTING - Modelos Individuales {model_type.upper()}")
    print(f"   Fechas a evaluar  : {len(fechas_corte)}")
    print(f"   Días de predicción: {dias_testeo}")
    print(f"   Días de benchmark : {dias_benchmark}")
    if usar_optuna:
        print("   Optimizacion Optuna: ACTIVADA")

    resultados = []
    resultados_metricas = {}

    for fecha in tqdm(fechas_corte, desc="Backtesting Individual"):
        df_preds = entrenar_modelo_individual(
            df,
            fecha_corte=fecha,
            encoder=encoder,
            dias_val=dias_testeo,
            dias_benchmark=dias_benchmark,
            model_type=model_type,
            usar_optuna=usar_optuna,
            verbose=False,
            pbar_position=1
        )

        if df_preds is not None and not df_preds.empty:
            # Métricas globales
            tpv_real_total = df_preds['tpv_futuro'].sum()
            tpv_pred_total = df_preds['prediccion_individual'].sum()
            
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(df_preds['tpv_futuro'], df_preds['prediccion_individual']))
            r2 = r2_score(df_preds['tpv_futuro'], df_preds['prediccion_individual'])
            
            resultados_metricas[fecha] = {
                'rmse': rmse,
                'r2': r2,
                'tpv_real': tpv_real_total,
                'tpv_pred': tpv_pred_total,
                'error_pct': abs(tpv_pred_total - tpv_real_total) / tpv_real_total * 100
            }
            
            # Calcular métricas por comercio
            resumen = df_preds.groupby('id_comercio_num').agg({
                'tpv_futuro': 'sum',
                'prediccion_individual': 'sum'
            }).reset_index()
            
            resumen['fecha_corte'] = fecha
            resumen['diff'] = (resumen['prediccion_individual'] - resumen['tpv_futuro']) / dias_benchmark
            resumen['error_pct'] = (resumen['diff'] * dias_benchmark / resumen['tpv_futuro'].replace(0, np.nan)) * 100
            resumen['error_pct'] = resumen['error_pct'].fillna(0)
            
            resultados.append(resumen)

    if not resultados:
        print("\nNo se obtuvieron resultados.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(resultados, ignore_index=True)

    # Matrices pivot
    df_matrix_pct = df_all.pivot_table(
        index='id_comercio_num',
        columns='fecha_corte',
        values='error_pct',
        aggfunc='first'
    ).reset_index()

    df_matrix_monto = df_all.pivot_table(
        index='id_comercio_num',
        columns='fecha_corte',
        values='diff',
        aggfunc='first'
    ).reset_index()

    # Agregar nombres
    def agregar_nombres(df_matrix):
        if encoder:
            try:
                df_matrix['nombre_comercio'] = encoder.inverse_transform(
                    df_matrix['id_comercio_num'].astype(int).values
                )
            except Exception:
                df_matrix['nombre_comercio'] = "ID_" + df_matrix['id_comercio_num'].astype(str)
        else:
            df_matrix['nombre_comercio'] = "ID_" + df_matrix['id_comercio_num'].astype(str)
        return df_matrix

    df_matrix_pct = agregar_nombres(df_matrix_pct)
    df_matrix_monto = agregar_nombres(df_matrix_monto)

    # Columnas de fechas
    cols_fechas = [c for c in df_matrix_pct.columns if c not in ['id_comercio_num', 'nombre_comercio']]
    cols_fechas_ordenadas = sorted(cols_fechas, key=lambda x: pd.to_datetime(x))
    
    # Mapeo a día-mes (01-01, 01-02)
    col_mapping = {fecha: _formatear_fecha_columna(fecha, usar_mes_esp=False) for fecha in cols_fechas_ordenadas}
    df_matrix_pct.rename(columns=col_mapping, inplace=True)
    df_matrix_monto.rename(columns=col_mapping, inplace=True)
    
    cols_meses = [col_mapping[f] for f in cols_fechas_ordenadas]

    # Reordenar
    df_matrix_pct = df_matrix_pct[['nombre_comercio'] + cols_meses]
    df_matrix_monto = df_matrix_monto[['nombre_comercio'] + cols_meses]

    # MAE
    df_matrix_pct['MAE'] = df_matrix_pct[cols_meses].abs().mean(axis=1)
    df_matrix_monto['MAE'] = df_matrix_monto[cols_meses].abs().mean(axis=1)

    # Ordenar
    df_final_pct = df_matrix_pct.sort_values('MAE', ascending=True).reset_index(drop=True)
    df_final_monto = df_matrix_monto.sort_values('MAE', ascending=False).reset_index(drop=True)

    # Fila TOTAL
    total_pct_valores = [df_final_pct[col].mean() for col in cols_meses]
    mae_pct = df_final_pct['MAE'].mean()
    total_pct = ['TOTAL'] + [f"{v:+.1f}%" for v in total_pct_valores] + [f"{mae_pct:.1f}%"]

    total_monto_valores = [df_final_monto[col].mean() for col in cols_meses]
    mae_monto = df_final_monto['MAE'].mean()
    total_monto = ['TOTAL'] + [f"{v/1e9:+.1f}B" for v in total_monto_valores] + [f"{mae_monto/1e9:.1f}B"]

    # PDFs
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    model_name = model_type.capitalize()

    # ========== PDF 1: Métricas Globales ==========
    df_metricas = pd.DataFrame(resultados_metricas).T
    df_metricas.index.name = 'fecha_corte'
    df_metricas = df_metricas.reset_index()
    
    print(f"\nMÉTRICAS GLOBALES:")
    print(f"   RMSE Promedio   : ${df_metricas['rmse'].mean()/1e9:.2f}B")
    print(f"   R² Promedio     : {df_metricas['r2'].mean():.4f}")
    print(f"   Error % Promedio: {df_metricas['error_pct'].mean():.2f}%")
    
    titulo_metricas = f"Backtesting Individual {model_name} Métricas {dias_testeo} días"
    tabla_metricas = [['MÉTRICA'] + [_formatear_fecha_columna(f) for f in fechas_corte] + ['PROMEDIO']]
    
    fila_rmse = ['RMSE (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'rmse'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_rmse.append(f"${df_metricas['rmse'].mean()/1e9:.2f}B")
    tabla_metricas.append(fila_rmse)
    
    fila_r2 = ['R2'] + [f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'r2'].values[0]:.4f}" for f in fechas_corte]
    fila_r2.append(f"{df_metricas['r2'].mean():.4f}")
    tabla_metricas.append(fila_r2)
    
    fila_err = ['Error %'] + [f"{df_metricas.loc[df_metricas['fecha_corte']==f, 'error_pct'].values[0]:.2f}%" for f in fechas_corte]
    fila_err.append(f"{df_metricas['error_pct'].mean():.2f}%")
    tabla_metricas.append(fila_err)
    
    fila_real = ['TPV Real (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_real'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_real.append(f"${df_metricas['tpv_real'].mean()/1e9:.2f}B")
    tabla_metricas.append(fila_real)
    
    fila_pred = ['TPV Pred (B)'] + [f"${df_metricas.loc[df_metricas['fecha_corte']==f, 'tpv_pred'].values[0]/1e9:.2f}B" for f in fechas_corte]
    fila_pred.append(f"${df_metricas['tpv_pred'].mean()/1e9:.2f}B   ")
    tabla_metricas.append(fila_pred)
    
    pdf_metricas = output_dir / f'backtesting_individual_{model_type}_metricas_{dias_testeo}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_metricas,
        titulo=titulo_metricas,
        filepath=str(pdf_metricas),
        color_header=(41, 128, 185),
        orientacion='landscape'
    )
    
    print(f"\nPDFs generados:")
    print(f"   [1] Métricas globales: {pdf_metricas}")

    # ========== PDF 2: Porcentual ==========
    titulo_pct = f"Backtesting Individual {model_name} Porcentual {dias_testeo} días"
    tabla_pct = [['COMERCIO'] + cols_meses + ['MAE']]

    for _, row in df_final_pct.iterrows():
        fila = [row['nombre_comercio']]
        fila += [f"{row[col]:+.1f}%" if pd.notna(row[col]) else "-" for col in cols_meses]
        fila += [f"{row['MAE']:.1f}%"]
        tabla_pct.append(fila)

    tabla_pct.append(total_pct)

    pdf_pct = output_dir / f'backtesting_individual_{model_type}_porcentual_{dias_testeo}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_pct,
        titulo=titulo_pct,
        filepath=str(pdf_pct),
        color_header=(52, 152, 219),
        orientacion='landscape'
    )
    print(f"   [2] Por comercio (%%): {pdf_pct}")

    # ========== PDF 3: Cantidad ==========
    titulo_monto = f"Backtesting Individual {model_name} Cantidad {dias_testeo} días"
    tabla_monto = [['COMERCIO'] + cols_meses + ['MAE']]

    for _, row in df_final_monto.iterrows():
        fila = [row['nombre_comercio']]
        fila += [f"{row[col]/1e9:+.1f}B" if pd.notna(row[col]) else "-" for col in cols_meses]
        fila += [f"{row['MAE']/1e9:.1f}B"]
        tabla_monto.append(fila)

    tabla_monto.append(total_monto)

    pdf_monto = output_dir / f'backtesting_individual_{model_type}_cantidad_{dias_testeo}dias.pdf'
    _crear_pdf_tabla(
        data=tabla_monto,
        titulo=titulo_monto,
        filepath=str(pdf_monto),
        color_header=(39, 174, 96),
        orientacion='landscape'
    )
    print(f"   [3] Por comercio (B, promedio diario): {pdf_monto}")

    return df_metricas, df_final_pct, df_final_monto