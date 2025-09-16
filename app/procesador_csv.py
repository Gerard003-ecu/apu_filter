import csv
import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import process

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def encontrar_mejor_coincidencia(texto, opciones, umbral=70):
    """Encuentra la mejor coincidencia de texto en una lista de opciones usando fuzzy matching."""
    if not texto or not opciones:
        return None
    coincidencia, puntaje = process.extractOne(texto, opciones)
    return coincidencia if puntaje >= umbral else None


def detect_delimiter(file_path: str) -> str:
    """
    Detecta el delimitador (',' o ';') de un archivo CSV usando csv.Sniffer.
    """
    try:
        with open(file_path, "r", encoding="latin1") as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=";,")
            return dialect.delimiter
    except (csv.Error, FileNotFoundError):
        # Fallback a ; si el sniffer falla o el archivo no existe
        return ";"


def safe_read_dataframe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Lee un archivo de datos (CSV o Excel) de forma segura y eficiente.
    - Para CSV, prueba explícitamente diferentes delimitadores y codificaciones.
    - Para Excel, usa openpyxl.
    - Convierte los tipos de datos a formatos eficientes con PyArrow.
    """
    file_extension = os.path.splitext(path)[1].lower()
    df = None

    if file_extension == ".csv":
        encodings = ["latin1", "utf-8", "iso-8859-1", "windows-1252"]
        delimiters = [";", ","]

        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    # Usar un manejador de contexto para asegurar que el archivo se cierre
                    with open(path, "r", encoding=encoding) as f:
                        # Pasamos el objeto de archivo a pandas, no la ruta
                        # Si se especifica explícitamente que no hay cabecera,
                        # debemos manejar CSVs "irregulares" proporcionando
                        # nombres de columna para evitar que pandas infiera
                        # el número de columnas de la primera fila.
                        kwargs_for_csv = kwargs.copy()
                        if (
                            kwargs_for_csv.get("header") is None
                            and "header" in kwargs_for_csv
                        ):
                            kwargs_for_csv["names"] = range(50)

                        temp_df = pd.read_csv(
                            f,
                            encoding=encoding,
                            delimiter=delimiter,
                            decimal=".",
                            on_bad_lines="warn",
                            **kwargs_for_csv,
                        )

                        # Si el DataFrame tiene más de una columna
                        # asumimos que el delimitador es correcto
                        if temp_df.shape[1] > 1:
                            logger.info(
                                f"Éxito al leer"
                                f"{os.path.basename(path)} con encoding '{encoding}'"
                                f"y delimitador '{delimiter}'"
                            )
                            df = temp_df
                            break  # Salir del bucle de encoding
                except Exception:
                    # Continuar al siguiente intento si hay cualquier error
                    continue
            if df is not None:
                break  # Salir del bucle de delimiter

    elif file_extension == ".xlsx":
        try:
            df = pd.read_excel(path, engine="openpyxl", **kwargs)
            logger.info(f"Éxito al leer Excel {path}")
        except Exception as e:
            logger.error(f"Error fatal al leer el archivo Excel {path}: {e}", exc_info=True)
            return None
    else:
        logger.error(f"Formato de archivo no soportado para {path}")
        return None

    if df is None:
        logger.error(f"No se pudo leer {os.path.basename(path)} con ninguna combinación.")
        return None

    try:
        # Optimización con PyArrow
        df = df.convert_dtypes(dtype_backend="pyarrow")
        logger.info(f"Tipos de datos optimizados con PyArrow para {path}")
    except Exception as e:
        logger.warning(f"No se pudo convertir los tipos a PyArrow para {path}: {e}")

    return df


def normalize_text(series, remove_accents=True, remove_special_chars=True):
    """Limpia y estandariza texto para hacer un 'join' fiable."""
    normalized = series.astype(str).str.lower().str.strip()

    if remove_accents:
        normalized = normalized.apply(unidecode)

    if remove_special_chars:
        # Mantener sólo caracteres alfanuméricos y espacios
        normalized = normalized.str.replace(r"[^a-z0-9\s]", "", regex=True)

    # Reducir múltiples espacios a uno solo
    normalized = normalized.str.replace(r"\s+", " ", regex=True)

    return normalized


def find_and_rename_columns(df, column_map, fuzzy_match=False):
    """
    Busca columnas que contengan un texto clave y las renombra.
    Soporta coincidencia exacta y fuzzy.
    """
    rename_dict = {}
    df.columns = df.columns.str.strip()

    for new_name, keywords in column_map.items():
        if not isinstance(keywords, list):
            keywords = [keywords]

        found_col = None
        for keyword in keywords:
            # Coincidencia exacta
            for col in df.columns:
                if keyword.lower() == col.lower():
                    found_col = col
                    break

            # Coincidencia parcial si no se encontró exacta
            if not found_col:
                for col in df.columns:
                    if keyword.lower() in col.lower():
                        found_col = col
                        break

            if found_col:
                rename_dict[found_col] = new_name
                break

    df = df.rename(columns=rename_dict)

    # Verificar columnas faltantes
    missing = [name for name in column_map.keys() if name not in df.columns]
    if missing:
        logger.warning(f"Columnas no encontradas: {missing}")

    return df


def load_config():
    """Carga el archivo de configuración JSON."""
    try:
        # Asumimos que config.json está en el mismo directorio que este script
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("El archivo config.json no fue encontrado.")
        return {}
    except json.JSONDecodeError:
        logger.error("Error al decodificar config.json.")
        return {}


config = load_config()


def process_presupuesto_csv(path: str) -> pd.DataFrame:
    """Lee y limpia el archivo presupuesto (CSV o Excel) de forma robusta."""
    df = safe_read_dataframe(path)
    if df is None:
        return pd.DataFrame()

    try:
        column_map = config.get("presupuesto_column_map", {})
        df = find_and_rename_columns(df, column_map)

        if "CODIGO_APU" not in df.columns:
            logger.warning("La columna 'CODIGO_APU' no se encontró en el presupuesto.")
            return pd.DataFrame()

        df["CODIGO_APU"] = df["CODIGO_APU"].astype(str).str.strip()
        df = df[df["CODIGO_APU"].str.contains(r",", na=False)]
        # Limpiar y convertir la columna de cantidad
        cantidad_str = (
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", ".", regex=False)
        )
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(cantidad_str, errors="coerce")

        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]
    except Exception as e:
        logger.error(f"Error procesando presupuesto.csv: {e}")
        return pd.DataFrame()


def process_insumos_csv(path: str) -> pd.DataFrame:
    """
    Lee y limpia el archivo insumos.csv, manejando correctamente los grupos
    y los delimitadores inconsistentes mediante una lectura manual línea por línea.
    """
    data_rows = []
    current_group = "INDEFINIDO"
    header_columns = []
    desc_index, vr_unit_index = -1, -1

    try:
        # Primero, detectar el delimitador
        delimiter = detect_delimiter(path)
        if not delimiter:
            return pd.DataFrame()

        with open(path, "r", encoding="latin1") as f:
            # Usar csv.reader para manejar correctamente los campos entre comillas
            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            for raw_parts in reader:
                if not raw_parts or not any(p.strip() for p in raw_parts):
                    continue

                parts = [p.strip() for p in raw_parts]
                line_for_check = "".join(parts)

                # Detectar y actualizar el grupo actual
                if len(parts) > 1 and parts[0].upper().startswith("G") and parts[1]:
                    current_group = parts[1]
                    continue

                # Identificar y capturar la línea de encabezado
                if (
                    "CODIGO" in line_for_check.upper()
                    and "DESCRIPCION" in line_for_check.upper()
                ):
                    header_columns = [p.upper() for p in parts]
                    desc_index = next(
                        (i for i, col in enumerate(header_columns) if "DESCRIPCION" in col),
                        -1,
                    )
                    vr_unit_index = next(
                        (i for i, col in enumerate(header_columns) if "VR. UNIT" in col), -1
                    )
                    continue

                # Procesar líneas de datos solo si hemos encontrado el encabezado
                if (
                    desc_index != -1
                    and vr_unit_index != -1
                    and len(parts) > max(desc_index, vr_unit_index)
                ):
                    description = parts[desc_index]
                    vr_unit_str = parts[vr_unit_index]

                    if description and vr_unit_str:
                        data_rows.append(
                            {
                                "DESCRIPCION_INSUMO": description,
                                "VR_UNITARIO_INSUMO": vr_unit_str,
                                "GRUPO_INSUMO": current_group,
                            }
                        )

        if not data_rows:
            logger.warning(f"No se extrajeron filas de datos de insumos de {path}")
            return pd.DataFrame()

        df = pd.DataFrame(data_rows)
        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df.dropna(subset=["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"], inplace=True)
        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        df = df.drop_duplicates(subset=["NORMALIZED_DESC"], keep="first")
        return df[
            ["NORMALIZED_DESC", "VR_UNITARIO_INSUMO", "GRUPO_INSUMO", "DESCRIPCION_INSUMO"]
        ]

    except Exception as e:
        logger.error(f"Error procesando insumos.csv manualmente: {e}")
        return pd.DataFrame()


def get_file_hash(path: str) -> str:
    """Calcula el hash de un archivo para usar como clave de caché."""
    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        logger.error(f"No se pudo encontrar el archivo para hashear: {path}")
        return ""
    return hash_md5.hexdigest()


def process_apus_csv_v2(path: str) -> pd.DataFrame:
    """
    Parsea un archivo de APUs con formato de "reporte de texto" utilizando una
    lógica de contexto explícita y robusta. VERSIÓN REFACTORIZADA.
    """
    apus_data = []
    current_context = {"apu_code": None, "apu_desc": None, "category": "INDEFINIDO"}
    category_keywords = config.get("category_keywords", {})
    potential_apu_desc = ""

    def to_numeric_safe(s: str):
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            s_cleaned = re.sub(r"[^\d,\.]", "", s).replace(",", ".").strip()
            return pd.to_numeric(s_cleaned, errors="coerce")
        return pd.NA

    def parse_data_line(parts, context):
        description = parts[0].strip()
        valor_total = next(
            (to_numeric_safe(p) for p in reversed(parts) if pd.notna(to_numeric_safe(p))),
            0.0,
        )
        cantidad = to_numeric_safe(parts[2]) if len(parts) > 2 else 0.0
        precio_unit = to_numeric_safe(parts[4]) if len(parts) > 4 else 0.0

        if (
            (pd.isna(precio_unit) or precio_unit == 0)
            and valor_total > 0
            and pd.notna(cantidad)
            and cantidad > 0
        ):
            precio_unit = valor_total / cantidad
        if (
            (pd.isna(valor_total) or valor_total == 0)
            and pd.notna(cantidad)
            and cantidad > 0
            and pd.notna(precio_unit)
        ):
            valor_total = cantidad * precio_unit

        return {
            "CODIGO_APU": context["apu_code"],
            "DESCRIPCION_APU": context["apu_desc"],
            "DESCRIPCION_INSUMO": description,
            "UNIDAD": parts[1] if len(parts) > 1 else "UND",
            "CANTIDAD_APU": cantidad if pd.notna(cantidad) else 0,
            "PRECIO_UNIT_APU": precio_unit if pd.notna(precio_unit) else 0,
            "VALOR_TOTAL_APU": valor_total if pd.notna(valor_total) else 0,
            "CATEGORIA": context["category"],
        }

    try:
        delimiter = detect_delimiter(path)
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Prioridad 1: Identificar la línea de ITEM
                if "ITEM:" in line.upper():
                    match = re.search(r"ITEM:\s*([\d,\.]*)", line.upper())
                    if match and match.group(1).strip():
                        apu_code = match.group(1).strip().rstrip('.,')

                        # El contexto se consolida aquí
                        current_context["apu_code"] = apu_code
                        current_context["apu_desc"] = potential_apu_desc
                        current_context["category"] = "INDEFINIDO" # Resetear categoría para el nuevo APU
                        potential_apu_desc = ""  # Limpiar para el próximo APU
                    continue

                # Usar csv.reader para manejar correctamente las comillas
                parts = next(csv.reader([line], delimiter=delimiter, quotechar='"'))
                parts = [p.strip() for p in parts]
                if not any(parts):
                    continue

                # Prioridad 2: Identificar línea de CATEGORÍA
                line_content_for_check = "".join(parts).upper()
                is_category = len(parts) < 4 and line_content_for_check in category_keywords
                if is_category:
                    current_context["category"] = category_keywords[line_content_for_check]
                    continue

                # Prioridad 3: Identificar línea de DATOS
                is_data_line = (
                    current_context["apu_code"] is not None and
                    len(parts) > 2 and
                    pd.notna(to_numeric_safe(parts[2]))
                )
                if is_data_line:
                    # Si la primera columna está vacía, es un offset y hay que desfasar
                    if not parts[0] and len(parts) > 1:
                        data_row = parse_data_line(parts[1:], current_context)
                    else:
                        data_row = parse_data_line(parts, current_context)

                    if data_row:
                        apus_data.append(data_row)
                    continue

                # Fallback: Tratar como línea de DESCRIPCIÓN
                if parts and parts[0]:
                    # Acumular en caso de descripciones multilínea
                    potential_apu_desc = (potential_apu_desc + " " + parts[0]).strip()

    except Exception as e:
        logger.error(f"Error fatal procesando APUs en {path}: {e}", exc_info=True)
        return pd.DataFrame()

    if not apus_data:
        logger.warning(f"No se extrajeron datos de APU de {path}")
        return pd.DataFrame()

    df = pd.DataFrame(apus_data)
    df["CODIGO_APU"] = df["CODIGO_APU"].str.strip()
    df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
    return df


def group_and_split_description(df: pd.DataFrame) -> pd.DataFrame:
    if "DESCRIPCION_APU" not in df.columns:
        df["grupo"] = "Ítems Varios"
        df["DESCRIPCION_APU"] = ""
        return df

    def get_group_and_short_desc(desc):
        if not isinstance(desc, str):
            return "Ítems Varios", ""
        cal_match = re.match(r"(.*? CAL\.? ?\d+)", desc, re.IGNORECASE)
        if cal_match:
            group = cal_match.group(1).strip()
            short_desc = desc[len(cal_match.group(0)) :].strip()
            short_desc = re.sub(
                r"^(de|en|con|para)\s", "", short_desc, flags=re.IGNORECASE
            ).strip()
            return group, short_desc
        words = desc.split()
        if len(words) > 4:
            return " ".join(words[:4]), " ".join(words[4:])
        return desc, ""

    df["original_description"] = df["DESCRIPCION_APU"]
    df[["grupo", "descripcion_corta"]] = df["original_description"].apply(
        lambda x: pd.Series(get_group_and_short_desc(x))
    )
    df["DESCRIPCION_APU"] = df["descripcion_corta"]
    df.drop(columns=["descripcion_corta"], inplace=True)
    return df


def _do_processing(presupuesto_path, apus_path, insumos_path):
    logger.info(f"Procesando archivos: {presupuesto_path}, {apus_path}, {insumos_path}")
    df_presupuesto = process_presupuesto_csv(presupuesto_path)
    df_insumos = process_insumos_csv(insumos_path)
    df_apus = process_apus_csv_v2(apus_path)
    if df_presupuesto.empty or df_insumos.empty or df_apus.empty:
        return {"error": "Uno o más archivos no pudieron ser procesados."}
    df_merged = pd.merge(
        df_apus, df_insumos, on="NORMALIZED_DESC", how="left", suffixes=("_apu", "")
    )
    df_merged["DESCRIPCION_INSUMO"] = df_merged["DESCRIPCION_INSUMO_apu"]
    df_merged["COSTO_INSUMO_EN_APU"] = np.where(
        df_merged["VR_UNITARIO_INSUMO"].notna(),
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_INSUMO"],
        df_merged["VALOR_TOTAL_APU"],
    )
    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_INSUMO"].fillna(
        df_merged["PRECIO_UNIT_APU"]
    )
    mask_recalc = (df_merged["VR_UNITARIO_FINAL"].isna()) | (
        df_merged["VR_UNITARIO_FINAL"] == 0
    )
    cantidad_safe = df_merged["CANTIDAD_APU"].replace(0, 1)
    df_merged.loc[mask_recalc, "VR_UNITARIO_FINAL"] = (
        df_merged.loc[mask_recalc, "COSTO_INSUMO_EN_APU"] / cantidad_safe
    )
    df_merged["COSTO_INSUMO_EN_APU"] = df_merged["COSTO_INSUMO_EN_APU"].fillna(0)
    df_apu_costos_categoria = (
        df_merged.groupby(["CODIGO_APU", "CATEGORIA"])["COSTO_INSUMO_EN_APU"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    cost_cols = ["MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"]
    for col in cost_cols:
        if col not in df_apu_costos_categoria.columns:
            df_apu_costos_categoria[col] = 0
    df_apu_costos_categoria["VALOR_SUMINISTRO_UN"] = df_apu_costos_categoria["MATERIALES"]
    df_apu_costos_categoria["VALOR_INSTALACION_UN"] = (
        df_apu_costos_categoria["MANO DE OBRA"] + df_apu_costos_categoria["EQUIPO"]
    )
    df_apu_costos_categoria["VALOR_CONSTRUCCION_UN"] = df_apu_costos_categoria[
        cost_cols
    ].sum(axis=1)

    # --- INICIO: Clasificación de APU por tipo ---
    def classify_apu(row):
        """Clasifica un APU basado en la proporción de sus costos."""
        costo_total = row["VALOR_CONSTRUCCION_UN"]
        if costo_total == 0:
            return "Indefinido"

        costo_materiales = row.get("MATERIALES", 0)
        costo_mano_obra = row.get("MANO DE OBRA", 0)
        costo_equipo = row.get("EQUIPO", 0)

        porcentaje_materiales = (costo_materiales / costo_total) * 100
        porcentaje_mo_eq = ((costo_mano_obra + costo_equipo) / costo_total) * 100

        # Aplicar las siguientes reglas en orden:
        # 1. Si Costo de Mano de Obra + Equipo > 75%, clasificar como "Instalación".
        if porcentaje_mo_eq > 75:
            return "Instalación"
        # 2. Si Costo de Materiales > 75% Y Costo de Mano de Obra + Equipo < 10%,
        # clasificar como "Suministro".
        if porcentaje_materiales > 75 and porcentaje_mo_eq < 10:
            return "Suministro"
        # 3. Si Costo de Materiales > 50% Y Costo de Mano de Obra + Equipo > 10%,
        # clasificar como "Suministro (Pre-fabricado)".
        if porcentaje_materiales > 50 and porcentaje_mo_eq > 10:
            return "Suministro (Pre-fabricado)"
        # 4. En cualquier otro caso, clasificar como "Obra Completa".
        return "Obra Completa"

    df_apu_costos_categoria["tipo_apu"] = df_apu_costos_categoria.apply(classify_apu, axis=1)
    # --- FIN: Clasificación de APU por tipo ---

    df_tiempo = (
        df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
        .groupby("CODIGO_APU")["CANTIDAD_APU"]
        .sum()
        .reset_index()
    )
    df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)
    df_final = pd.merge(df_presupuesto, df_apu_costos_categoria, on="CODIGO_APU", how="left")
    df_final = pd.merge(df_final, df_tiempo, on="CODIGO_APU", how="left")
    final_cols_to_fill = [
        "VALOR_SUMINISTRO_UN",
        "VALOR_INSTALACION_UN",
        "VALOR_CONSTRUCCION_UN",
        "TIEMPO_INSTALACION",
    ]
    for col in final_cols_to_fill:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
        else:
            df_final[col] = 0
    df_final = group_and_split_description(df_final)
    logger.info(
        f"Diagnóstico: {df_final['VALOR_CONSTRUCCION_UN'].notna().sum()} de"
        f" {len(df_final)} ítems del presupuesto encontraron su costo."
    )
    df_merged_renamed = df_merged.rename(
        columns={
            "DESCRIPCION_INSUMO": "DESCRIPCION",
            "CANTIDAD_APU": "CANTIDAD",
            "UNIDAD": "UNIDAD",
            "VR_UNITARIO_FINAL": "VALOR_UNITARIO",
            "COSTO_INSUMO_EN_APU": "VALOR_TOTAL",
        }
    )
    apus_detail = {
        n: g[
            [
                "DESCRIPCION",
                "UNIDAD",
                "CANTIDAD",
                "VALOR_UNITARIO",
                "VALOR_TOTAL",
                "CATEGORIA",
            ]
        ].to_dict("records")
        for n, g in df_merged_renamed.groupby("CODIGO_APU")
    }
    insumos_dict = {}
    if "GRUPO_INSUMO" in df_insumos.columns:
        df_insumos_renamed = df_insumos.rename(
            columns={
                "DESCRIPCION_INSUMO": "Descripción",
                "VR_UNITARIO_INSUMO": "Vr Unitario",
            }
        )
        for name, group in df_insumos_renamed.groupby("GRUPO_INSUMO"):
            if name and isinstance(name, str):
                insumos_dict[name.strip()] = (
                    group[["Descripción", "Vr Unitario"]].dropna().to_dict("records")
                )
    logger.info("--- Procesamiento completado ---")

    # Reemplazar NaN por None para evitar problemas de serialización JSON
    df_final = df_final.replace({np.nan: None})
    df_merged_renamed = df_merged_renamed.replace({np.nan: None})
    df_apus = df_apus.replace({np.nan: None})
    df_insumos = df_insumos.replace({np.nan: None})

    # Actualizar los diccionarios después de la limpieza
    apus_detail = {
        n: g[
            [
                "DESCRIPCION",
                "UNIDAD",
                "CANTIDAD",
                "VALOR_UNITARIO",
                "VALOR_TOTAL",
                "CATEGORIA",
            ]
        ]
        .replace({np.nan: None})
        .to_dict("records")
        for n, g in df_merged_renamed.groupby("CODIGO_APU")
    }

    insumos_dict = {}
    if "GRUPO_INSUMO" in df_insumos.columns:
        df_insumos_renamed = df_insumos.rename(
            columns={
                "DESCRIPCION_INSUMO": "Descripción",
                "VR_UNITARIO_INSUMO": "Vr Unitario",
            }
        )
        for name, group in df_insumos_renamed.groupby("GRUPO_INSUMO"):
            if name and isinstance(name, str):
                insumos_dict[name.strip()] = (
                    group[["Descripción", "Vr Unitario"]]
                    .replace({np.nan: None})
                    .dropna()
                    .to_dict("records")
                )

    return {
        "presupuesto": df_final.to_dict("records"),
        "insumos": insumos_dict,
        "apus_detail": apus_detail,
        "all_apus": df_apus.to_dict("records"),
        "raw_insumos_df": df_insumos.to_dict("records"),
    }


@lru_cache(maxsize=10)
def _cached_csv_processing(
    presupuesto_hash, apus_hash, insumos_hash, presupuesto_path, apus_path, insumos_path
):
    logger.info("Cache miss.")
    return _do_processing(presupuesto_path, apus_path, insumos_path)


def process_all_files(
    presupuesto_path: str, apus_path: str, insumos_path: str, use_cache: bool = True
):
    if not use_cache:
        _cached_csv_processing.cache_clear()
        logger.info("Caché limpiado forzosamente y procesando sin caché.")
        return _do_processing(presupuesto_path, apus_path, insumos_path)
    try:
        file_hashes = (
            get_file_hash(presupuesto_path),
            get_file_hash(apus_path),
            get_file_hash(insumos_path),
        )
        if any(h == "" for h in file_hashes):
            return {"error": "No se pudo generar el hash para uno o más archivos."}
    except Exception as e:
        logger.error(f"Error al generar hashes de archivos: {e}")
        return {"error": f"Error al generar hashes: {e}"}
    result = _cached_csv_processing(*file_hashes, presupuesto_path, apus_path, insumos_path)
    info = _cached_csv_processing.cache_info()
    logger.info(f"Cache info: hits={info.hits}, misses={info.misses}, size={info.currsize}")
    return result


def calculate_estimate(
    params: Dict[str, str], data_store: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Calcula una estimación de costo basada en la lógica de negocio de
    "Suministro + Instalación".
    """
    log = []
    required_params = ["tipo", "material", "cuadrilla"]
    missing_params = [p for p in required_params if p not in params or not params[p]]
    if missing_params:
        error_msg = f"Parámetros requeridos faltantes o vacíos: {missing_params}"
        logger.warning(error_msg)
        return {"error": error_msg, "log": [error_msg]}

    def _normalize(text: str) -> str:
        return unidecode(str(text)).upper()

    all_apus_list = data_store.get("all_apus", [])
    if not all_apus_list:
        return {
            "error": "No hay datos de APU disponibles.",
            "log": ["Error: all_apus no encontrado en data_store."],
        }

    df_apus = pd.DataFrame(all_apus_list)
    df_apus_unique = df_apus.drop_duplicates(subset=["DESCRIPCION_APU"]).copy()
    df_apus_unique["DESC_NORMALIZED"] = df_apus_unique["DESCRIPCION_APU"].apply(_normalize)

    # --- Parámetros y Mapeo ---
    tipo = params.get("tipo", "").upper()
    material = params.get("material", "").upper()
    cuadrilla = params.get("cuadrilla", "0")
    log.append(f"Parámetros de entrada: {params}")

    param_map = config.get("param_map", {})
    material_mapped = _normalize(param_map.get("material", {}).get(material, material))
    tipo_mapped = _normalize(param_map.get("tipo", {}).get(tipo, tipo))
    log.append(f"Parámetros mapeados: material='{material_mapped}', tipo='{tipo_mapped}'")

    # --- 1. Búsqueda de Suministro (Refactorizada con Fuzzy-Matching) ---
    log.append("\n--- BÚSQUEDA DE SUMINISTRO ---")

    # Filtrar primero por APUs que parezcan de suministro
    df_suministro_options = df_apus_unique[
        df_apus_unique["DESC_NORMALIZED"].str.contains("SUMINISTRO|SOLO", na=False)
    ]
    opciones_suministro = df_suministro_options["DESC_NORMALIZED"].tolist()

    log.append(
        f"Buscando la mejor coincidencia para '{material_mapped}' en {len(opciones_suministro)} opciones de suministro."
    )
    mejor_coincidencia = encontrar_mejor_coincidencia(material_mapped, opciones_suministro)

    suministro_candidates = pd.DataFrame()  # Inicializar como vacío
    if mejor_coincidencia:
        suministro_candidates = df_apus_unique[
            df_apus_unique["DESC_NORMALIZED"] == mejor_coincidencia
        ]
        log.append(f"Mejor coincidencia encontrada: '{mejor_coincidencia}'")
    else:
        log.append("No se encontró una buena coincidencia para el suministro.")

    log.append(f"Encontrados {len(suministro_candidates)} candidatos para suministro.")

    apu_suministro_desc = ""
    valor_suministro = 0.0
    if not suministro_candidates.empty:
        apu_suministro_desc = suministro_candidates.iloc[0]["DESCRIPCION_APU"]
        suministro_items_df = df_apus[df_apus["DESCRIPCION_APU"] == apu_suministro_desc]
        valor_suministro = suministro_items_df[
            suministro_items_df["CATEGORIA"] == "MATERIALES"
        ]["VALOR_TOTAL_APU"].sum()
        log.append(
            f"APU de Suministro seleccionado: '{apu_suministro_desc}'"
            f" -> Valor: ${valor_suministro:,.0f}"
        )
    else:
        log.append(
            "ADVERTENCIA: No se encontró un APU de suministro directo. "
            "Iniciando fallback a insumos."
        )
        raw_insumos_data = data_store.get("raw_insumos_df")
        if raw_insumos_data:
            df_insumos = pd.DataFrame(raw_insumos_data)
            if not df_insumos.empty:
                # Buscar el material normalizado en la lista de insumos
                insumo_match = df_insumos[
                    df_insumos["NORMALIZED_DESC"].str.contains(material_mapped, na=False)
                ]
                if not insumo_match.empty:
                    valor_suministro = insumo_match.iloc[0]["VR_UNITARIO_INSUMO"]
                    apu_suministro_desc = (
                        f"Insumo: {insumo_match.iloc[0]['DESCRIPCION_INSUMO']}"
                    )
                    log.append(
                        f"ÉXITO (Fallback): Insumo encontrado: '{apu_suministro_desc}'"
                        f" -> Valor: ${valor_suministro:,.0f}"
                    )
                else:
                    log.append(
                        "ERROR (Fallback): Material no encontrado en lista de insumos."
                    )
            else:
                log.append("ERROR (Fallback): El dataframe de insumos está vacío.")
        else:
            log.append("ERROR (Fallback): El dataframe de insumos no está disponible.")

    # --- 2. Búsqueda de Instalación (Refactorizada con Filtros Secuenciales) ---
    log.append("\n--- BÚSQUEDA DE INSTALACIÓN ---")
    # Paso A: Filtrar por "MANO DE OBRA" Y "INSTALACION"
    df_inst = df_apus_unique[
        df_apus_unique["DESC_NORMALIZED"].str.contains("MANO DE OBRA|MANO OBRA")
        & df_apus_unique["DESC_NORMALIZED"].str.contains("INSTALACION")
    ].copy()
    log.append(
        f"Paso A: {len(df_inst)} APUs encontrados con 'MANO DE OBRA' e 'INSTALACION'."
    )

    # Paso B: Filtrar por material
    df_inst = df_inst[df_inst["DESC_NORMALIZED"].str.contains(material_mapped)].copy()
    log.append(f"Paso B: {len(df_inst)}APUs restantes tras filtrar por '{material_mapped}'.")

    # Paso C: Filtrar por cuadrilla
    cuadrilla_term = f"CUADRILLA DE {cuadrilla}"
    df_inst = df_inst[df_inst["DESC_NORMALIZED"].str.contains(cuadrilla_term)].copy()
    log.append(f"Paso C: {len(df_inst)} APUs restantes tras filtrar por '{cuadrilla_term}'.")

    if df_inst.empty:
        msg = "No se encontró un APU de instalación que coincida con todos los criterios."
        log.append(f"ERROR: {msg}")
        return {"error": msg, "log": "\n".join(log)}

    # Tomar el primer resultado
    apu_instalacion_desc = df_inst.iloc[0]["DESCRIPCION_APU"]
    log.append(f"ÉXITO: APU de Instalación seleccionado: '{apu_instalacion_desc}'")

    # --- 3. Cálculo de Costos y Tiempos ---
    log.append("\n--- CÁLCULO DE COSTOS Y TIEMPO ---")
    instalacion_items_df = df_apus[df_apus["DESCRIPCION_APU"] == apu_instalacion_desc]
    costos_instalacion = (
        instalacion_items_df.groupby("CATEGORIA")["VALOR_TOTAL_APU"]
        .sum()
        .reindex(["MANO DE OBRA", "EQUIPO"], fill_value=0)
    )
    valor_instalacion = costos_instalacion["MANO DE OBRA"] + costos_instalacion["EQUIPO"]
    tiempo_instalacion = instalacion_items_df[
        instalacion_items_df["CATEGORIA"] == "MANO DE OBRA"
    ]["CANTIDAD_APU"].sum()

    log.append(
        f"Costo Instalación Base (MO+EQ): ${valor_instalacion:,.0f} "
        f"| Tiempo: {tiempo_instalacion:.4f} días/un"
    )

    # --- 4. Ajustes Adicionales (Ej. Izaje, Seguridad, Zona) ---
    # (La lógica de ajustes se mantiene, pero se podría refactorizar en el futuro)
    # ...

    log.append(
        f"\n--- RESULTADO FINAL ---\n"
        f"Valor Suministro: ${valor_suministro:,.0f}\n"
        f"Valor Instalación: ${valor_instalacion:,.0f}"
    )

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": f"Suministro: {apu_suministro_desc or 'N/A'} | "
        f"Instalación: {apu_instalacion_desc}",
        "log": "\n".join(log),
    }
