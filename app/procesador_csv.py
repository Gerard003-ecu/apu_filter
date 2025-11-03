import logging

import numpy as np
import pandas as pd

from .apu_processor import APUProcessor
from .data_validator import validate_and_clean_data
from .report_parser_crudo import ReportParserCrudo
from .utils import (
    clean_apu_code,
    find_and_rename_columns,
    normalize_text_series,
    safe_read_dataframe,
)

logger = logging.getLogger(__name__)

# ==================== FUNCIONES AUXILIARES ====================

def group_and_split_description(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva la descripción original y la divide en principal y secundaria.

    Args:
        df (pd.DataFrame): El DataFrame que contiene la columna 'DESCRIPCION_APU'.

    Returns:
        pd.DataFrame: El DataFrame con la descripción original conservada y
                      dividida en 'DESCRIPCION_APU' y 'descripcion_secundaria'.
    """
    df_grouped = df.copy()
    df_grouped["original_description"] = df_grouped["DESCRIPCION_APU"]

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

    Args:
        path (str): La ruta al archivo CSV del presupuesto.
        config (dict): La configuración de la aplicación.

    Returns:
        pd.DataFrame: Un DataFrame con los datos del presupuesto procesados.
    """
    df = safe_read_dataframe(path, header=None)
    if df is None:
        return pd.DataFrame()

    try:
        # Buscar la fila que actúa como encabezado
        header_row_index = -1
        for i, row in df.head(10).iterrows():
            row_str = " ".join(row.astype(str).str.upper())
            if "ITEM" in row_str and "DESCRIPCION" in row_str and "CANT" in row_str:
                header_row_index = i
                break

        if header_row_index != -1:
            df.columns = df.iloc[header_row_index]
            df = df.iloc[header_row_index + 1 :].reset_index(drop=True)
        else:
            logger.warning("No se encontró una fila de encabezado válida en el presupuesto.")
            return pd.DataFrame()

        column_map = config.get("presupuesto_column_map", {})
        df = find_and_rename_columns(df, column_map)

        if "CODIGO_APU" not in df.columns:
            logger.error(
                "La columna 'CODIGO_APU' no se pudo crear después de buscar el encabezado."
            )
            return pd.DataFrame()

        df["CODIGO_APU"] = df["CODIGO_APU"].astype(str).apply(clean_apu_code)
        df = df[df["CODIGO_APU"].notna() & (df["CODIGO_APU"] != "")]

        cantidad_str = (
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", ".", regex=False)
        )
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(cantidad_str, errors="coerce")

        # 🔥 VALIDACIÓN CRÍTICA: Eliminar duplicados de CODIGO_APU
        duplicates = df[df.duplicated(subset=["CODIGO_APU"], keep=False)]
        if not duplicates.empty:
            logger.warning(
                f"⚠️ Se encontraron {len(duplicates)} filas duplicadas en "
                f"presupuesto por CODIGO_APU. Se conservará la primera ocurrencia."
            )
            logger.debug(
                f"Códigos duplicados: {duplicates['CODIGO_APU'].unique().tolist()}"
            )
            df = df.drop_duplicates(subset=["CODIGO_APU"], keep="first")

        logger.info(f"✅ Presupuesto cargado: {len(df)} APUs únicos")
        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]

    except Exception as e:
        logger.error(f"Error procesando el archivo de presupuesto: {e}", exc_info=True)
        return pd.DataFrame()


def process_insumos_csv(file_path: str) -> pd.DataFrame:
    """Procesa el archivo CSV de insumos con formato no estándar.

    Args:
        file_path (str): La ruta al archivo CSV de insumos.

    Returns:
        pd.DataFrame: Un DataFrame con los datos de insumos procesados.
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

        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"].astype(str).str.replace(",", "."), errors="coerce"
        )
        df["DESCRIPCION_INSUMO_NORM"] = normalize_text_series(df["DESCRIPCION_INSUMO"])

        df = df.dropna(subset=["DESCRIPCION_INSUMO"])

        # 🔥 VALIDACIÓN CRÍTICA: Verificar duplicados en descripciones normalizadas
        duplicates = df[df.duplicated(subset=["DESCRIPCION_INSUMO_NORM"], keep=False)]
        if not duplicates.empty:
            logger.warning(
                f"⚠️ Se encontraron {len(duplicates)} insumos con descripciones "
                f"normalizadas duplicadas. Esto podría causar problemas en el merge. "
                f"Se conservará el de mayor precio."
            )
            # Conservar el de mayor precio para cada descripción normalizada
            df = df.sort_values("VR_UNITARIO_INSUMO", ascending=False).drop_duplicates(
                subset=["DESCRIPCION_INSUMO_NORM"], keep="first"
            )

        logger.info(f"✅ Insumos cargados: {len(df)} insumos únicos")
        return df

    except Exception as e:
        logger.error(f"Error procesando el archivo de insumos: {e}")
        return pd.DataFrame()


def process_all_files(
    presupuesto_path: str, apus_path: str, insumos_path: str, config: dict
) -> dict:
    """Orquesta el procesamiento completo de los archivos de entrada.

    Args:
        presupuesto_path (str): La ruta al archivo del presupuesto.
        apus_path (str): La ruta al archivo de APUs.
        insumos_path (str): La ruta al archivo de insumos.
        config (dict): La configuración de la aplicación.

    Returns:
        dict: Un diccionario con los datos procesados.
    """
    return _do_processing(presupuesto_path, apus_path, insumos_path, config)


# ==================== LÓGICA PRINCIPAL ====================

def _calculate_apu_costs_and_metadata(df_merged: pd.DataFrame) -> tuple:
    """Calcula costos, clasifica APUs y extrae metadatos.

    Args:
        df_merged (pd.DataFrame): El DataFrame fusionado con los datos de APU.

    Returns:
        tuple: Una tupla con los DataFrames de costos de APU, tiempo y rendimiento.
    """
    # ========== 4. AGREGAR COSTOS POR APU Y CATEGORÍA (VALIDADO) ==========
    logger.info("📊 Agregando costos por APU y categoría...")

    df_apu_costos = (
        df_merged.groupby(["CODIGO_APU", "CATEGORIA"])["COSTO_INSUMO_EN_APU"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    logger.info(f"✅ df_apu_costos creado: {len(df_apu_costos)} APUs únicos")

    cost_cols = ["MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"]
    for col in cost_cols:
        if col not in df_apu_costos.columns:
            df_apu_costos[col] = 0

    # ========== 5. CALCULAR VALORES UNITARIOS ==========
    df_apu_costos["VALOR_SUMINISTRO_UN"] = df_apu_costos["MATERIALES"]
    df_apu_costos["VALOR_INSTALACION_UN"] = (
        df_apu_costos["MANO DE OBRA"] + df_apu_costos["EQUIPO"]
    )
    df_apu_costos["VALOR_CONSTRUCCION_UN"] = df_apu_costos[cost_cols].sum(axis=1)

    # 🔥 VALIDACIÓN MEJORADA: Verificar valores unitarios razonables
    valor_max = df_apu_costos["VALOR_CONSTRUCCION_UN"].max()
    valor_avg = df_apu_costos["VALOR_CONSTRUCCION_UN"].mean()
    valor_std = df_apu_costos["VALOR_CONSTRUCCION_UN"].std()

    logger.info(f"📈 Estadísticas de costos unitarios: Max=${valor_max:,.2f}, Avg=${valor_avg:,.2f}")

    # Detectar outliers usando método estadístico
    outlier_threshold = valor_avg + (3 * valor_std)
    outliers = df_apu_costos[df_apu_costos["VALOR_CONSTRUCCION_UN"] > outlier_threshold]

    if not outliers.empty:
        logger.warning(f"⚠️ Se detectaron {len(outliers)} APUs con costos anómalos")
        for _, outlier in outliers.iterrows():
            logger.warning(
                f" APU {outlier['CODIGO_APU']}: ${outlier['VALOR_CONSTRUCCION_UN']:,.2f} "
                f"(MO: {outlier.get('MANO DE OBRA', 0):,.2f}, MAT: {outlier.get('MATERIALES', 0):,.2f})"
            )

    # ========== 6. CLASIFICAR APUs ==========
    def classify_apu(row):
        costo_total = row["VALOR_CONSTRUCCION_UN"]
        if costo_total == 0:
            return "Indefinido"

        mo_cost = row.get("MANO DE OBRA", 0)
        eq_cost = row.get("EQUIPO", 0)
        mat_cost = row.get("MATERIALES", 0)

        porcentaje_mo_eq = ((mo_cost + eq_cost) / costo_total) * 100
        porcentaje_materiales = (mat_cost / costo_total) * 100

        if porcentaje_mo_eq > 75:
            return "Instalación"

        if porcentaje_materiales > 75 and porcentaje_mo_eq < 15:
            return "Suministro"

        if porcentaje_materiales > 65 and porcentaje_mo_eq > 15:
            return "Suministro (Pre-fabricado)"

        return "Obra Completa"

    df_apu_costos["tipo_apu"] = df_apu_costos.apply(classify_apu, axis=1)

    # ========== 7. CALCULAR TIEMPO Y RENDIMIENTO ==========
    df_mo_data = df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"].copy()

    df_tiempo = (
        df_mo_data.groupby("CODIGO_APU")["CANTIDAD_APU"]
        .sum()
        .reset_index()
    )
    df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)

    if "RENDIMIENTO" in df_mo_data.columns:
        df_rendimiento = (
            df_mo_data.groupby("CODIGO_APU")["RENDIMIENTO"]
            .sum()
            .reset_index()
        )
        df_rendimiento.rename(columns={"RENDIMIENTO": "RENDIMIENTO_DIA"}, inplace=True)
    else:
        df_rendimiento = pd.DataFrame(columns=["CODIGO_APU", "RENDIMIENTO_DIA"])


    return df_apu_costos, df_tiempo, df_rendimiento


def _do_processing(
    presupuesto_path: str, apus_path: str, insumos_path: str, config: dict
) -> dict:
    """Lógica central para procesar, unificar y calcular todos los datos.

    Args:
        presupuesto_path (str): La ruta al archivo del presupuesto.
        apus_path (str): La ruta al archivo de APUs.
        insumos_path (str): La ruta al archivo de insumos.
        config (dict): La configuración de la aplicación.

    Returns:
        dict: Un diccionario con los datos procesados.
    """
    logger.info("=" * 80)
    logger.info("🚀 Iniciando procesamiento de archivos...")
    logger.info(f"   Presupuesto: {presupuesto_path}")
    logger.info(f"   APUs: {apus_path}")
    logger.info(f"   Insumos: {insumos_path}")
    logger.info("=" * 80)

    # ========== 1. CARGAR DATOS BASE ==========
    df_presupuesto = process_presupuesto_csv(presupuesto_path, config)
    df_insumos = process_insumos_csv(insumos_path)

    # Etapa 1: Extracción Cruda
    parser = ReportParserCrudo(apus_path)
    raw_records = parser.parse_to_raw()

    # Etapa 2: Procesamiento con Lógica de Negocio
    processor = APUProcessor(raw_records)
    df_apus_raw = processor.process_all()

    if df_presupuesto.empty or df_insumos.empty or df_apus_raw.empty:
        logger.error("❌ Fallo al cargar uno o más archivos de datos.")
        return {"error": "Fallo al cargar uno o más archivos de datos."}

    logger.info("📊 Datos cargados:")
    logger.info(f"   - Presupuesto: {len(df_presupuesto)} APUs")
    logger.info(f"   - Insumos maestros: {len(df_insumos)} insumos")
    logger.info(
        f"   - APUs detallados: {len(df_apus_raw)} registros de insumos en APUs"
    )

    # ========== 2. MERGE APUs con INSUMOS (CRÍTICO) ==========
    logger.info("🔗 Iniciando merge de APUs con catálogo de insumos...")

    # Normalizar descripción en df_apus_raw para el merge
    df_apus_raw['DESCRIPCION_INSUMO_NORM'] = normalize_text_series(df_apus_raw['DESCRIPCION_INSUMO'])

    rows_before = len(df_apus_raw)
    df_merged = pd.merge(
        df_apus_raw,
        df_insumos,
        on="DESCRIPCION_INSUMO_NORM",
        how="left",
        suffixes=("_apu", ""),
        validate="m:1"  # 🔥 VALIDACIÓN: Múltiples APUs a 1 insumo (evita explosión)
    )
    rows_after = len(df_merged)

    if rows_after != rows_before:
        logger.error(
            f"❌ EXPLOSIÓN CARTESIANA DETECTADA EN MERGE APUs-INSUMOS: "
            f"{rows_before} → {rows_after} filas (+{rows_after - rows_before})"
        )
        problematic = (
            df_apus_raw.groupby("DESCRIPCION_INSUMO_NORM")
            .size()
            .loc[lambda x: x > 1]
            .sort_values(ascending=False)
            .head(10)
        )
        logger.error(
            f"Descripciones normalizadas con más ocurrencias:\n{problematic}"
            )
        logger.warning(
            "⚠️ Continuando con precaución..."
            )

    df_merged["TIPO_INSUMO"] = df_merged["CATEGORIA"]
    df_merged["DESCRIPCION_INSUMO"] = df_merged["DESCRIPCION_INSUMO"].fillna(
        df_merged["DESCRIPCION_INSUMO_apu"]
    )
    df_merged.rename(columns={"UNIDAD_INSUMO_apu": "UNIDAD_INSUMO"}, inplace=True)

    # ========== 3. CALCULAR COSTOS DE INSUMOS (VALIDADO) ==========
    logger.info("💰 Calculando costos de insumos...")
    qty_stats = df_merged["CANTIDAD_APU"].describe()
    logger.debug(f"Estadísticas de CANTIDAD_APU:\n{qty_stats}")
    if df_merged["CANTIDAD_APU"].max() > 1e6:
        logger.warning(
            f"⚠️ Cantidad extremadamente alta detectada: {df_merged['CANTIDAD_APU'].max()}"
        )

    precio_stats = df_merged["VR_UNITARIO_INSUMO"].describe()
    logger.debug(f"Estadísticas de VR_UNITARIO_INSUMO:\n{precio_stats}")
    costo_insumo = np.where(
        df_merged["VR_UNITARIO_INSUMO"].notna(),
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_INSUMO"],
        df_merged["VALOR_TOTAL_APU"]
    )
    df_merged["COSTO_INSUMO_EN_APU"] = pd.Series(costo_insumo).fillna(0)
    costo_max = df_merged["COSTO_INSUMO_EN_APU"].max()
    if costo_max > 1e9:
        logger.error(f"❌ COSTO ANÓMALO DETECTADO: {costo_max:,.2f}")
        anomalies = df_merged[df_merged["COSTO_INSUMO_EN_APU"] > 1e9].head(10)
        logger.error(
            "Registros anómalos:\n%s",
            anomalies[
                [
                    "CODIGO_APU",
                    "DESCRIPCION_INSUMO",
                    "CANTIDAD_APU",
                    "VR_UNITARIO_INSUMO",
                    "COSTO_INSUMO_EN_APU",
                ]
            ],
        )

    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_INSUMO"].fillna(
        df_merged["PRECIO_UNIT_APU"]
    )
    cantidad_safe = df_merged["CANTIDAD_APU"].replace(0, 1)
    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_FINAL"].fillna(
        df_merged["COSTO_INSUMO_EN_APU"] / cantidad_safe
    )

    # ========== 4-7. CALCULAR COSTOS Y METADATOS DE APU ==========
    df_apu_costos, df_tiempo, df_rendimiento = _calculate_apu_costs_and_metadata(
        df_merged
    )

    # ========== 8. MERGE FINAL CON PRESUPUESTO (CRÍTICO) ==========
    logger.info("🔗 Realizando merge final: presupuesto + costos...")
    logger.info(
        f"   - df_presupuesto: {len(df_presupuesto)} filas, "
        f"{df_presupuesto['CODIGO_APU'].nunique()} APUs únicos"
    )
    logger.info(
        f"   - df_apu_costos: {len(df_apu_costos)} filas, "
        f"{df_apu_costos['CODIGO_APU'].nunique()} APUs únicos"
    )

    rows_before_final = len(df_presupuesto)
    try:
        df_final = pd.merge(
            df_presupuesto,
            df_apu_costos,
            on="CODIGO_APU",
            how="left",
            validate="1:1",  # 🔥 VALIDACIÓN CRÍTICA: Solo permite relación 1:1
        )
    except pd.errors.MergeError:
        logger.error(
            "❌ EXPLOSIÓN CARTESIANA DETECTADA: Se encontraron APUs duplicados "
            "al intentar unir con el presupuesto."
        )
        dupes = df_apu_costos[
            df_apu_costos.duplicated("CODIGO_APU", keep=False)
        ].sort_values("CODIGO_APU")
        logger.error(f"APUs duplicados en la lista de costos:\n{dupes}")
        return {
            "error": "Explosión cartesiana detectada en merge final. "
            "Revise los datos de APU, existen duplicados."
        }

    rows_after_final = len(df_final)
    if rows_after_final != rows_before_final:
        logger.error(
            f"❌ EXPLOSIÓN CARTESIANA EN MERGE FINAL: "
            f"{rows_before_final} → {rows_after_final} filas"
        )
        return {
            "error": "Explosión cartesiana detectada en merge final. Revise los datos."
        }

    logger.info(f"✅ Merge exitoso: {len(df_final)} filas (sin duplicación)")

    df_final = pd.merge(df_final, df_tiempo, on="CODIGO_APU", how="left")
    df_final = group_and_split_description(df_final)

    # 🔥 CALCULAR VALORES TOTALES (CON VALIDACIÓN)
    logger.info("💵 Calculando valores totales del presupuesto...")
    if "CANTIDAD_PRESUPUESTO" not in df_final.columns:
        logger.error("❌ Falta columna CANTIDAD_PRESUPUESTO en df_final")
        return {"error": "Error en estructura de datos: falta CANTIDAD_PRESUPUESTO"}

    df_final["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(
        df_final["CANTIDAD_PRESUPUESTO"], errors="coerce"
    ).fillna(0)
    logger.debug(
        "Estadísticas de CANTIDAD_PRESUPUESTO:\n%s",
        df_final["CANTIDAD_PRESUPUESTO"].describe(),
    )

    df_final["VALOR_SUMINISTRO_TOTAL"] = (
        df_final["VALOR_SUMINISTRO_UN"] * df_final["CANTIDAD_PRESUPUESTO"]
    )
    df_final["VALOR_INSTALACION_TOTAL"] = (
        df_final["VALOR_INSTALACION_UN"] * df_final["CANTIDAD_PRESUPUESTO"]
    )
    df_final["VALOR_CONSTRUCCION_TOTAL"] = (
        df_final["VALOR_CONSTRUCCION_UN"] * df_final["CANTIDAD_PRESUPUESTO"]
    )

    total_construccion = df_final["VALOR_CONSTRUCCION_TOTAL"].sum()
    logger.info(f"💰 COSTO TOTAL CONSOLIDADO: ${total_construccion:,.2f}")
    if total_construccion > 1e11:
        logger.error(f"❌ COSTO TOTAL ANORMALMENTE ALTO: ${total_construccion:,.2f}")
        top_contributors = df_final.nlargest(10, "VALOR_CONSTRUCCION_TOTAL")[
            [
                "CODIGO_APU",
                "DESCRIPCION_APU",
                "CANTIDAD_PRESUPUESTO",
                "VALOR_CONSTRUCCION_UN",
                "VALOR_CONSTRUCCION_TOTAL",
            ]
        ]
        logger.error("Top 10 APUs que más contribuyen:\n%s", top_contributors)
        return {
            "error": f"Costo total anormalmente alto: ${total_construccion:,.2f}. "
            f"Revise los datos."
        }

    # ========== 9. CONSTRUIR df_processed_apus ==========
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
    df_processed_apus["DESC_NORMALIZED"] = normalize_text_series(
        df_processed_apus["DESCRIPCION_APU"]
    )
    df_processed_apus = group_and_split_description(df_processed_apus)

    # ========== 10. PREPARAR DICCIONARIOS DE SALIDA ==========
    df_merged.rename(
        columns={"vr_unitario_final": "vr_unitario", "costo_insumo_en_apu": "vr_total"},
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
        "raw_insumos_df": df_insumos,
        "processed_apus": df_processed_apus.to_dict("records"),
    }

    # ========== 11. VALIDACIÓN FINAL ==========
    logger.info("🤖 Iniciando Agente de Validación de Datos...")
    validated_result = validate_and_clean_data(result_dict)
    logger.info("✅ Validación completada.")

    validated_result["raw_insumos_df"] = df_insumos.to_dict("records")

    logger.info("=" * 80)
    logger.info("🎉 Procesamiento completado exitosamente")
    logger.info(f"   - APUs en presupuesto: {len(df_final)}")
    logger.info(f"   - Costo total: ${total_construccion:,.2f}")
    logger.info("=" * 80)

    return validated_result
