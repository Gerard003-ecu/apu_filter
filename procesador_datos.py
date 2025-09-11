import pandas as pd
import os

def process_files(presupuesto_path, apus_path, insumos_path):
    """
    Procesa los archivos de presupuesto, APU e insumos para consolidar los datos.

    Args:
        presupuesto_path (str): Ruta al archivo de presupuesto (Excel).
        apus_path (str): Ruta al archivo de APUs (Excel).
        insumos_path (str): Ruta al archivo de insumos (Excel).

    Returns:
        pandas.DataFrame: Un DataFrame consolidado con los costos del proyecto.
    """
    try:
        # --- 1. Cargar y limpiar el archivo de Presupuesto ---
        # Se asume que los encabezados están en la fila 10.
        df_presupuesto = pd.read_excel(presupuesto_path, engine='openpyxl', header=9)
        df_presupuesto = df_presupuesto.rename(columns={
            'ÍTEM': 'CODIGO_APU',
            'DESCRIPCIÓN': 'DESCRIPCION_APU',
            'CANT.': 'CANTIDAD_PRESUPUESTO',
            'VR. UNITARIO': 'VR_UNITARIO_PRESUPUESTO'
        })
        # Seleccionar solo las columnas necesarias
        df_presupuesto = df_presupuesto[['CODIGO_APU', 'DESCRIPCION_APU', 'CANTIDAD_PRESUPUESTO']]
        df_presupuesto.dropna(subset=['CODIGO_APU'], inplace=True)
        # Convertir a tipos de datos numéricos, tratando errores.
        df_presupuesto['CANTIDAD_PRESUPUESTO'] = pd.to_numeric(df_presupuesto['CANTIDAD_PRESUPUESTO'], errors='coerce')


        # --- 2. Cargar y limpiar el archivo de Insumos ---
        # Se asume que los encabezados están en la fila 9.
        df_insumos = pd.read_excel(insumos_path, engine='openpyxl', header=8)
        df_insumos = df_insumos.rename(columns={
            'CÓDIGO': 'CODIGO_INSUMO',
            'VR.UNITARIO': 'VR_UNITARIO_INSUMO'
        })
        # Seleccionar solo las columnas necesarias
        df_insumos = df_insumos[['CODIGO_INSUMO', 'VR_UNITARIO_INSUMO']]
        df_insumos.dropna(subset=['CODIGO_INSUMO'], inplace=True)
        df_insumos['VR_UNITARIO_INSUMO'] = pd.to_numeric(df_insumos['VR_UNITARIO_INSUMO'], errors='coerce')


        # --- 3. Cargar y procesar el archivo de APUs ---
        # Este archivo es complejo. Se asume que el código del APU se encuentra en una fila
        # y los insumos en las filas siguientes, hasta el próximo código de APU.
        df_apu_raw = pd.read_excel(apus_path, engine='openpyxl', header=None)
        apus_data = []
        current_apu_code = None

        for i, row in df_apu_raw.iterrows():
            # Se asume que una fila de definición de APU tiene un valor en la primera celda que empieza con "APU:"
            cell_value = str(row[0])
            if 'APU:' in cell_value:
                current_apu_code = cell_value.split(':')[1].strip()
                continue

            # Se asume que las filas de insumos tienen valores en columnas específicas
            # Columna 0: CODIGO_INSUMO, Columna 4: CANTIDAD_APU
            if current_apu_code and pd.notna(row[0]) and pd.notna(row[4]) and isinstance(row[4], (int, float)):
                apus_data.append({
                    'CODIGO_APU': current_apu_code,
                    'CODIGO_INSUMO': str(row[0]),
                    'CANTIDAD_APU': float(row[4])
                })

        df_apus = pd.DataFrame(apus_data)
        df_apus.dropna(subset=['CODIGO_APU', 'CODIGO_INSUMO', 'CANTIDAD_APU'], inplace=True)


        # --- 4. Combinar (Merge) los DataFrames ---
        # Unir APUs con Insumos para obtener el costo de cada componente.
        df_merged = pd.merge(df_apus, df_insumos, on='CODIGO_INSUMO', how='left')

        # Calcular el costo total para cada insumo dentro de su APU.
        df_merged['COSTO_INSUMO_EN_APU'] = df_merged['CANTIDAD_APU'] * df_merged['VR_UNITARIO_INSUMO']

        # Agrupar por APU para calcular el valor unitario total de cada APU.
        df_apu_costos = df_merged.groupby('CODIGO_APU').agg(
            VR_UNITARIO_CALCULADO=('COSTO_INSUMO_EN_APU', 'sum')
        ).reset_index()

        # Unir los costos calculados de APU con el presupuesto.
        df_final = pd.merge(df_presupuesto, df_apu_costos, on='CODIGO_APU', how='left')


        # --- 5. Calcular Costos Finales y dar formato al resultado ---
        # Calcular el valor total por cada ítem del presupuesto.
        df_final['VALOR_TOTAL'] = df_final['CANTIDAD_PRESUPUESTO'] * df_final['VR_UNITARIO_CALCULADO']

        # Añadir la columna 'ZONA' como se solicitó.
        df_final['ZONA'] = ''

        # Renombrar y seleccionar las columnas para el DataFrame final.
        df_final = df_final.rename(columns={
            'CODIGO_APU': 'Código APU',
            'DESCRIPCION_APU': 'Descripción',
            'VALOR_TOTAL': 'Valor Total'
        })

        return df_final[['Código APU', 'Descripción', 'Valor Total', 'ZONA']]

    except Exception as e:
        print(f"Ocurrió un error al procesar los archivos: {e}")
        # En caso de error, devolver un DataFrame vacío para evitar que la aplicación falle.
        return pd.DataFrame()
