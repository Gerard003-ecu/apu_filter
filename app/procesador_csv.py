import csv
import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fuzzywuzzy import process
from unidecode import unidecode

from .report_parser import ReportParser
from .utils import clean_apu_code

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def encontrar_mejor_coincidencia(texto, opciones, umbral=70):
    """Encuentra la mejor coincidencia de texto
    en una lista de opciones usando fuzzy matching.
    """
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
    """Normaliza texto sin perder información crítica (#, -)."""
    normalized = series.astype(str).str.lower().str.strip()

    if remove_accents:
        normalized = normalized.apply(unidecode)

    if remove_special_chars:
        # Conservar # y -
        normalized = normalized.str.replace(r"[^a-z0-9\s#\-]", "", regex=True)

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


def to_numeric_safe(s):
    if isinstance(s, (int, float)):
        return s
    if isinstance(s, str):
        s_cleaned = s.replace(".", "").replace(",", ".").strip()
        return pd.to_numeric(s_cleaned, errors="coerce")
    return pd.NA


def parse_data_line(parts: List[str], current_category: str) -> Optional[Dict]:
    """
    Parsea una línea de datos de un APU (insumo) y devuelve un diccionario con los datos.
    Contiene lógica especializada para ítems de "MANO DE OBRA".
    """
    description = parts[0]
    labor_prefixes = ["M.O.", "SISO", "INGENIERO"]
    is_mano_de_obra = current_category == "MANO DE OBRA" or any(
        description.upper().startswith(p) for p in labor_prefixes
    )

    if is_mano_de_obra and len(parts) >= 6:
        # Lógica especializada para Mano de Obra
        valor_total = to_numeric_safe(parts[5])
        precio_unitario_jornal = to_numeric_safe(parts[3])
        rendimiento = to_numeric_safe(parts[4])

        cantidad = 0.0
        if pd.notna(rendimiento) and rendimiento != 0:
            cantidad = 1 / rendimiento

        # El "Jornal Total" es el verdadero precio unitario para la mano de obra
        precio_unit = precio_unitario_jornal

    else:
        # Lógica para Materiales y Otros (Fallback)
        cantidad = to_numeric_safe(parts[2])
        precio_unit = to_numeric_safe(parts[4])
        valor_total = to_numeric_safe(parts[5])

        # Si el valor total no está, pero sí la cantidad y el precio, se calcula
        if pd.notna(cantidad) and pd.notna(precio_unit) and pd.isna(valor_total):
            valor_total = cantidad * precio_unit

    if pd.notna(valor_total):
        return {
            "DESCRIPCION_INSUMO": description,
            "UNIDAD": parts[1],
            "CANTIDAD_APU": cantidad if pd.notna(cantidad) else 0,
            "PRECIO_UNIT_APU": precio_unit if pd.notna(precio_unit) else 0,
            "VALOR_TOTAL_APU": valor_total if pd.notna(valor_total) else 0,
            "CATEGORIA": current_category,
        }
    return None


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

        df["CODIGO_APU"] = df["CODIGO_APU"].astype(str).apply(clean_apu_code)
        df = df[df["CODIGO_APU"].notna() & (df["CODIGO_APU"] != "")]

        # Limpiar y convertir cantidad
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
    Lee y limpia insumos.csv, detectando encabezados SAGUT automáticamente
    y soportando variantes en nombres de columnas.
    """
    data_rows = []
    current_group = "INDEFINIDO"
    desc_index, vr_unit_index = -1, -1

    try:
        delimiter = detect_delimiter(path)
        if not delimiter:
            return pd.DataFrame()

        with open(path, "r", encoding="latin1") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"')
            for parts in reader:
                if not parts or not any(p.strip() for p in parts):
                    continue

                parts = [p.strip() for p in parts]
                line_for_check = " ".join(parts).upper()

                # Detectar grupos
                if len(parts) > 1 and parts[0].upper().startswith("G") and parts[1]:
                    current_group = parts[1]
                    continue

                # Detectar encabezados SAGUT (más flexible)
                if "DESCRIPCION" in line_for_check and "VR" in line_for_check:
                    header_map = {col.upper(): idx for idx, col in enumerate(parts)}
                    desc_index = header_map.get("DESCRIPCION", -1)
                    vr_unit_index = next(
                        (i for k, i in header_map.items() if "VR" in k and "UNIT" in k),
                        -1,
                    )
                    continue

                # Procesar filas de datos
                if desc_index != -1 and vr_unit_index != -1:
                    if len(parts) > max(desc_index, vr_unit_index):
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
            logger.warning(f"No se extrajeron filas de insumos de {path}")
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
        logger.error(f"Error procesando insumos.csv: {e}", exc_info=True)
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
    logger.debug(
        f"df_presupuesto creado. Columnas:"
        f"{df_presupuesto.columns.tolist()}. Filas: {len(df_presupuesto)}"
        )

    df_insumos = process_insumos_csv(insumos_path)
    logger.debug(
        f"df_insumos creado. Columnas:"
        f"{df_insumos.columns.tolist()}. Filas: {len(df_insumos)}"
        )

    # Usar el nuevo ReportParser
    parser = ReportParser(apus_path)
    df_apus = parser.parse()
    logger.debug(f"Contenido de df_apus después del parsing:\n{df_apus.to_string()}")
    logger.debug(
        f"df_apus creado por ReportParser."
        f"Columnas: {df_apus.columns.tolist()}. Filas: {len(df_apus)}"
        )


    if df_presupuesto.empty or df_insumos.empty or df_apus.empty:
        return {"error": "Uno o más archivos no pudieron ser procesados."}

    try:
        # --- SECCIÓN CRÍTICA ---
        logger.info("Iniciando merge de df_apus y df_insumos...")
        df_merged = pd.merge(
            df_apus, df_insumos, on="NORMALIZED_DESC", how="left", suffixes=("_apu", "")
        )
        logger.info("Merge completado exitosamente.")

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
        df_costos = df_apu_costos_categoria
        df_costos["VALOR_SUMINISTRO_UN"] = df_costos["MATERIALES"]
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

        df_apu_costos_categoria["tipo_apu"] = df_apu_costos_categoria.apply(
            classify_apu, axis=1
            )
        # --- FIN: Clasificación de APU por tipo ---

        df_tiempo = (
            df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
            .groupby("CODIGO_APU")["CANTIDAD_APU"]
            .sum()
            .reset_index()
        )
        df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)

        # --- INICIO: Creación del DataFrame de APUs Procesados ---
        # Unificar toda la información de APUs en un solo DataFrame para el estimador.
        df_apu_descriptions = df_apus[["CODIGO_APU", "DESCRIPCION_APU"]].drop_duplicates()
        df_processed_apus = pd.merge(
            df_apu_costos_categoria, df_apu_descriptions, on="CODIGO_APU", how="left"
        )
        df_processed_apus = pd.merge(
            df_processed_apus, df_tiempo, on="CODIGO_APU", how="left"
        )
        df_processed_apus["DESCRIPCION_APU"] = df_processed_apus[
            "DESCRIPCION_APU"
        ].fillna("")
        # Limpiar NaNs en columnas clave para evitar errores en cálculos posteriores
        fill_zero_cols = [
            "VALOR_SUMINISTRO_UN",
            "VALOR_INSTALACION_UN",
            "VALOR_CONSTRUCCION_UN",
            "TIEMPO_INSTALACION",
        ]
        for col in fill_zero_cols:
            if col in df_processed_apus.columns:
                df_processed_apus[col] = df_processed_apus[col].fillna(0)

        # Aplicar la agrupación al DataFrame de búsqueda para el estimador
        df_processed_apus = group_and_split_description(df_processed_apus)
        # Ahora creamos la columna normalizada desde la descripción original completa
        df_processed_apus["DESC_NORMALIZED"] = normalize_text(
            df_processed_apus["original_description"]
        )
        # --- FIN: Creación del DataFrame de APUs Procesados ---

        df_final = pd.merge(
            df_presupuesto, df_apu_costos_categoria, on="CODIGO_APU", how="left"
        )
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
                "unidad": "UNIDAD",
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
        df_processed_apus = df_processed_apus.replace({np.nan: None})

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
            "processed_apus": df_processed_apus.to_dict("records"),
        }

    except Exception as e:
        logger.error(
            f"ERROR CRÍTICO en el merge o cálculo en _do_processing: {e}",exc_info=True
            )
        return {"error": f"Fallo en la etapa de procesamiento de datos: {e}"}


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


