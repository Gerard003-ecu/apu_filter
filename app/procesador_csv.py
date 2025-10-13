import logging

import numpy as np
import pandas as pd

from .data_validator import validate_and_clean_data
from .report_parser import ReportParser
from .utils import (
    clean_apu_code,
    find_and_rename_columns,
    normalize_text,
    safe_read_dataframe,
)

logger = logging.getLogger(__name__)

# Estas funciones ahora viven aquí temporalmente para evitar dependencias circulares
# TODO: Mover a un módulo de utils de dataframes o similar
def group_and_split_description(df):
    """Conserva la descripción original y divide la principal en dos si es necesario.

    Esta función crea una copia de la columna 'DESCRIPCION_APU' en
    'original_description' para preservar el texto completo. Luego, si encuentra
    un separador ' / ', divide 'DESCRIPCION_APU' en una parte principal y una
    secundaria.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna 'DESCRIPCION_APU'.

    Returns:
        pd.DataFrame: El DataFrame modificado con las nuevas columnas
                      'original_description' y 'descripcion_secundaria'.
    """
    df_grouped = df.copy()
    df_grouped["original_description"] = df_grouped["DESCRIPCION_APU"]

    # Dividir la descripción solo si contiene ' / '
    if "DESCRIPCION_APU" in df_grouped.columns:
        split_desc = df_grouped["DESCRIPCION_APU"].str.split(" / ", n=1, expand=True)
        df_grouped["DESCRIPCION_APU"] = split_desc[0]
        if split_desc.shape[1] > 1:
            df_grouped["descripcion_secundaria"] = split_desc[1]
        else:
            df_grouped["descripcion_secundaria"] = ""

    return df_grouped


def process_presupuesto_csv(path: str, config: dict) -> pd.DataFrame:
    """Lee y procesa el archivo CSV del presupuesto.

    Detecta automáticamente la fila de encabezado, renombra las columnas según
    la configuración, limpia los códigos de APU y convierte las cantidades a
    formato numérico.

    Args:
        path (str): La ruta al archivo CSV del presupuesto.
        config (dict): El diccionario de configuración de la aplicación.

    Returns:
        pd.DataFrame: Un DataFrame limpio y procesado del presupuesto, o un
                      DataFrame vacío si ocurre un error.
    """
    # Leer sin encabezado para tener control total
    df = safe_read_dataframe(path, header=None)
    if df is None:
        return pd.DataFrame()

    try:
        # Buscar la fila que actúa como encabezado
        header_row_index = -1
        for i, row in df.head(10).iterrows():  # Buscar en las primeras 10 filas
            row_str = " ".join(row.astype(str).str.upper())
            if "ITEM" in row_str and "DESCRIPCION" in row_str and "CANT" in row_str:
                header_row_index = i
                break

        if header_row_index != -1:
            # Asignar la fila encontrada como el nuevo encabezado
            df.columns = df.iloc[header_row_index]
            # Eliminar todas las filas hasta el encabezado (inclusive)
            df = df.iloc[header_row_index + 1 :].reset_index(drop=True)
        else:
            logger.warning(
                "No se encontró una fila de encabezado válida en el presupuesto."
            )
            return pd.DataFrame()

        # Ahora, proceder con el renombrado estándar
        column_map = config.get("presupuesto_column_map", {})
        df = find_and_rename_columns(df, column_map)

        if "CODIGO_APU" not in df.columns:
            logger.error(
                "La columna 'CODIGO_APU' no se pudo crear después de buscar el encabezado."
            )
            return pd.DataFrame()

        # El resto de la lógica de limpieza es la misma
        df["CODIGO_APU"] = df["CODIGO_APU"].astype(str).apply(clean_apu_code)
        df = df[df["CODIGO_APU"].notna() & (df["CODIGO_APU"] != "")]

        cantidad_str = (
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", ".", regex=False)
        )
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(cantidad_str, errors="coerce")

        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]

    except Exception as e:
        logger.error(f"Error procesando el archivo de presupuesto: {e}", exc_info=True)
        return pd.DataFrame()


def process_insumos_csv(file_path):
    """Procesa el archivo CSV de insumos con formato no estándar.

    Parsea un archivo donde los datos están agrupados bajo encabezados de 'Grupo'
    y extrae la información relevante, normalizando descripciones y valores.

    Args:
        file_path (str): La ruta al archivo CSV de insumos.

    Returns:
        pd.DataFrame: Un DataFrame con los datos de insumos procesados, o un
                      DataFrame vacío si ocurre un error.
    """
    try:
        with open(file_path, "r", encoding="latin1") as f:
            lines = f.readlines()

        records = []
        current_group = None
        header = None

        for line in lines:
            parts = [p.strip().replace('"', "") for p in line.strip().split(";")]
            if not any(parts):
                continue
            if parts[0].startswith("G"):
                current_group = parts[1]
                header = None
                continue
            if "CODIGO" in parts[0] and "DESCRIPCION" in parts[1]:
                header = ["CODIGO", "DESCRIPCION", "UND", "CANT.", "VR. UNIT."]
                continue
            if header and current_group:
                record = {"GRUPO_INSUMO": current_group}
                for i, col in enumerate(header):
                    record[col] = parts[i] if i < len(parts) else None
                records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.rename(
            columns={
                "DESCRIPCION": "DESCRIPCION_INSUMO",
                "VR. UNIT.": "VR_UNITARIO_INSUMO",
            },
            inplace=True,
        )
        final_cols = ["GRUPO_INSUMO", "DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"]
        df = df[final_cols]
        # Forzar la conversión a string antes de reemplazar la coma
        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"].astype(str).str.replace(",", "."), errors="coerce"
        )
        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        return df.dropna(subset=["DESCRIPCION_INSUMO"])
    except Exception as e:
        logger.error(f"Error procesando el archivo de insumos: {e}")
        return pd.DataFrame()


def process_all_files(presupuesto_path, apus_path, insumos_path, config):
    """Orquesta el procesamiento completo de los archivos de entrada.

    Esta función es el punto de entrada principal que invoca a la lógica de
    procesamiento interna.

    Args:
        presupuesto_path (str): Ruta al archivo de presupuesto.
        apus_path (str): Ruta al archivo de APUs.
        insumos_path (str): Ruta al archivo de insumos.
        config (dict): Diccionario de configuración de la aplicación.

    Returns:
        dict: Un diccionario que contiene todos los DataFrames procesados y
              listas de datos, o un diccionario de error si falla.
    """
    return _do_processing(presupuesto_path, apus_path, insumos_path, config)


def _do_processing(presupuesto_path, apus_path, insumos_path, config):
    """Lógica central para procesar, unificar y calcular todos los datos.

    Realiza la carga, limpieza, unión y cálculos sobre los datos de presupuesto,
    APUs e insumos para generar las estructuras de datos finales que utilizará
    la aplicación.

    Args:
        presupuesto_path (str): Ruta al archivo de presupuesto.
        apus_path (str): Ruta al archivo de APUs.
        insumos_path (str): Ruta al archivo de insumos.
        config (dict): Diccionario de configuración.

    Returns:
        dict: Un diccionario con los datos procesados listos para ser usados.
    """
    logger.info(
        f"Iniciando procesamiento: {presupuesto_path}, {apus_path}, {insumos_path}"
    )

    # 1. Cargar todos los datos base
    df_presupuesto = process_presupuesto_csv(presupuesto_path, config)
    df_insumos = process_insumos_csv(insumos_path)
    parser = ReportParser(apus_path)
    df_apus_raw = parser.parse()

    if df_presupuesto.empty or df_insumos.empty or df_apus_raw.empty:
        return {"error": "Fallo al cargar uno o más archivos de datos."}

    # 2. Unificar y calcular costos de insumos
    df_merged = pd.merge(
        df_apus_raw, df_insumos, on="NORMALIZED_DESC", how="left", suffixes=("_apu", "")
    )
    df_merged["TIPO_INSUMO"] = df_merged["CATEGORIA"]

    # Coalesce: Usar la descripción del insumo maestro si existe, si no, usar la del APU.
    # Esto es crucial para no perder descripciones de mano de obra o insumos no mapeados.
    df_merged["DESCRIPCION_INSUMO"] = df_merged["DESCRIPCION_INSUMO"].fillna(
        df_merged["DESCRIPCION_INSUMO_apu"]
    )

    # La unidad del insumo siempre vendrá del reporte de APU, así que renombramos la columna.
    df_merged.rename(columns={"UNIDAD_INSUMO_apu": "UNIDAD_INSUMO"}, inplace=True)

    # Calcular el costo real de cada insumo en el APU
    costo_insumo = np.where(
        df_merged["VR_UNITARIO_INSUMO"].notna(),
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_INSUMO"],
        df_merged["VALOR_TOTAL_APU"]
    )
    df_merged["COSTO_INSUMO_EN_APU"] = pd.Series(costo_insumo).fillna(0)

    # Calcular el precio unitario final de cada insumo
    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_INSUMO"].fillna(
        df_merged["PRECIO_UNIT_APU"]
    )
    cantidad_safe = df_merged["CANTIDAD_APU"].replace(0, 1)
    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_FINAL"].fillna(
        df_merged["COSTO_INSUMO_EN_APU"] / cantidad_safe
    )

    # 3. Agregar costos por APU y categoría
    df_apu_costos = (
        df_merged.groupby(["CODIGO_APU", "CATEGORIA"])["COSTO_INSUMO_EN_APU"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    cost_cols = ["MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"]
    for col in cost_cols:
        if col not in df_apu_costos.columns:
            df_apu_costos[col] = 0

    # 4. Calcular valores unitarios y clasificar APUs
    df_apu_costos["VALOR_SUMINISTRO_UN"] = df_apu_costos["MATERIALES"]
    df_apu_costos["VALOR_INSTALACION_UN"] = (
        df_apu_costos["MANO DE OBRA"] + df_apu_costos["EQUIPO"]
    )
    df_apu_costos["VALOR_CONSTRUCCION_UN"] = df_apu_costos[cost_cols].sum(axis=1)

    def classify_apu(row):
        # ... (la función classify_apu no necesita cambios)
        costo_total = row["VALOR_CONSTRUCCION_UN"]
        if costo_total == 0:
            return "Indefinido"
        porcentaje_mo_eq = (
            (row.get("MANO DE OBRA", 0) + row.get("EQUIPO", 0)) / costo_total
        ) * 100
        if porcentaje_mo_eq > 75:
            return "Instalación"
        porcentaje_materiales = (row.get("MATERIALES", 0) / costo_total) * 100
        if porcentaje_materiales > 75 and porcentaje_mo_eq < 10:
            return "Suministro"
        if porcentaje_materiales > 50 and porcentaje_mo_eq > 10:
            return "Suministro (Pre-fabricado)"
        return "Obra Completa"

    df_apu_costos["tipo_apu"] = df_apu_costos.apply(classify_apu, axis=1)

    # 5. Calcular tiempo y rendimiento
    df_tiempo = (
        df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
        .groupby("CODIGO_APU")["CANTIDAD_APU"]
        .sum()
        .reset_index()
    )
    df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)

    df_rendimiento = (
        df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
        .groupby("CODIGO_APU")["RENDIMIENTO"]
        .sum()
        .reset_index()
    )
    df_rendimiento.rename(columns={"RENDIMIENTO": "RENDIMIENTO_DIA"}, inplace=True)

    # 6. Construir el DataFrame final para el presupuesto (df_final)
    df_final = pd.merge(df_presupuesto, df_apu_costos, on="CODIGO_APU", how="left")
    df_final = pd.merge(df_final, df_tiempo, on="CODIGO_APU", how="left")
    df_final = group_and_split_description(df_final) # Esto crea 'original_description'

    # 7. Construir el DataFrame final para el estimador (df_processed_apus)
    df_apu_descriptions = df_apus_raw[
        ["CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU"]
    ].drop_duplicates()
    df_processed_apus = pd.merge(
        df_apu_costos, df_apu_descriptions, on="CODIGO_APU", how="left"
    )
    df_processed_apus = pd.merge(
        df_processed_apus, df_tiempo, on="CODIGO_APU", how="left"
    )
    df_processed_apus = pd.merge(
        df_processed_apus, df_rendimiento, on="CODIGO_APU", how="left"
    )
    df_processed_apus.rename(columns={"UNIDAD_APU": "UNIDAD"}, inplace=True)
    df_processed_apus["DESC_NORMALIZED"] = normalize_text(
        df_processed_apus["DESCRIPCION_APU"]
    )
    df_processed_apus = group_and_split_description(
        df_processed_apus
    )  # Esto crea 'original_description'

    # 8. Preparar diccionarios de salida y sanitizar
    df_merged.rename(
        columns={"VR_UNITARIO_FINAL": "VR_UNITARIO", "COSTO_INSUMO_EN_APU": "VR_TOTAL"},
        inplace=True,
    )

    apus_detail = df_merged.to_dict("records")

    insumos_dict = {
        name.strip(): group[["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"]]
        .dropna()
        .to_dict("records")
        for name, group in df_insumos.groupby("GRUPO_INSUMO")
        if name and isinstance(name, str)
    }

    result_dict = {
        "presupuesto": df_final.to_dict("records"),
        "insumos": insumos_dict,
        "apus_detail": apus_detail,
        "all_apus": df_apus_raw.to_dict("records"),
        "raw_insumos_df": df_insumos, # Pasamos el DF directamente para el validador
        "processed_apus": df_processed_apus.to_dict("records"),
    }

    logger.info("Iniciando Agente de Validación de Datos...")
    validated_result = validate_and_clean_data(result_dict)
    logger.info("Validación completada.")

    # El validador ya no necesita el DataFrame, lo convertimos a dict para el resto de la app
    validated_result['raw_insumos_df'] = df_insumos.to_dict("records")

    logger.info("--- Procesamiento completado ---")
    return validated_result
