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

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def safe_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """Lee un CSV de forma segura con múltiples intentos y codificaciones."""
    encodings = ["latin1", "utf-8", "iso-8859-1", "windows-1252"]
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Error leyendo {path} con encoding {encoding}: {e}")

    logger.error(f"No se pudo leer {path} con ningún encoding conocido")
    return None


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
    """Lee y limpia el archivo presupuesto.csv de forma robusta."""
    df = safe_read_csv(path, delimiter=";", skipinitialspace=True)
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
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]
    except Exception as e:
        logger.error(f"Error procesando presupuesto.csv: {e}")
        return pd.DataFrame()


def process_insumos_csv(path: str) -> pd.DataFrame:
    """
    Lee y limpia el archivo insumos.csv, manejando correctamente los grupos
    mediante una lectura manual línea por línea.
    """
    data_rows = []
    current_group = "INDEFINIDO"
    header_found = False

    try:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(";")]
                if len(parts) > 1 and parts[0].upper().startswith("G") and parts[1]:
                    current_group = parts[1]
                    continue
                if "CODIGO" in line and "DESCRIPCION" in line:
                    header_found = True
                    header_columns = [p.strip() for p in line.split(";")]
                    desc_index = next(
                        (
                            i
                            for i, col in enumerate(header_columns)
                            if "DESCRIPCION" in col
                        ),
                        -1,
                    )
                    vr_unit_index = next(
                        (
                            i
                            for i, col in enumerate(header_columns)
                            if "VR. UNIT" in col
                        ),
                        -1,
                    )
                    continue
                if header_found and desc_index != -1 and vr_unit_index != -1:
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
            return pd.DataFrame()
        df = pd.DataFrame(data_rows)
        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df.dropna(subset=["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"], inplace=True)
        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        df = df.drop_duplicates(subset=["NORMALIZED_DESC"], keep="first")
        return df[
            [
                "NORMALIZED_DESC",
                "VR_UNITARIO_INSUMO",
                "GRUPO_INSUMO",
                "DESCRIPCION_INSUMO",
            ]
        ]
    except Exception as e:
        logger.error(f"Error procesando insumos.csv: {e}")
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
    """Versión mejorada del parser de APUs con mejor manejo de estados."""
    apus_data = []
    current_context = {
        "apu_code": None,
        "apu_desc": None,
        "category": "INDEFINIDO",
        "last_non_empty_line": None,
    }
    category_keywords = config.get("category_keywords", {})

    def to_numeric_safe(s):
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            s = s.replace(".", "").replace(",", ".").strip()
            return pd.to_numeric(s, errors="coerce")
        return pd.NA

    def parse_data_line(parts, context):
        description = parts[0]
        cantidad = to_numeric_safe(parts[2])
        precio_unit = to_numeric_safe(parts[4])
        valor_total = to_numeric_safe(parts[5])

        ### CAMBIO CLAVE 1: Lógica para extraer VALOR_TOTAL_APU de forma robusta ###
        # Si el valor_total no se pudo parsear de la columna 6 (índice 5),
        # intenta buscarlo en la última columna no vacía, que es un caso común.
        if pd.isna(valor_total):
            last_numeric_part = next(
                (
                    to_numeric_safe(p)
                    for p in reversed(parts)
                    if to_numeric_safe(p) is not pd.NA
                ),
                pd.NA,
            )
            if pd.notna(last_numeric_part):
                valor_total = last_numeric_part

        if pd.isna(cantidad) and "%" in parts[2]:
            jornal_total = to_numeric_safe(parts[3])
            if pd.notna(valor_total) and pd.notna(jornal_total) and jornal_total > 0:
                cantidad = valor_total / jornal_total
            else:
                cantidad = 0
            precio_unit = jornal_total

        if pd.notna(valor_total):
            if (
                (pd.isna(cantidad) or cantidad == 0)
                and pd.notna(precio_unit)
                and precio_unit > 0
            ):
                cantidad = valor_total / precio_unit
            if (
                (pd.isna(precio_unit) or precio_unit == 0)
                and pd.notna(cantidad)
                and cantidad > 0
            ):
                precio_unit = valor_total / cantidad

            return {
                "CODIGO_APU": context["apu_code"],
                "DESCRIPCION_APU": context["apu_desc"],
                "DESCRIPCION_INSUMO": description,
                "CANTIDAD_APU": cantidad if pd.notna(cantidad) else 0,
                "PRECIO_UNIT_APU": precio_unit if pd.notna(precio_unit) else 0,
                "VALOR_TOTAL_APU": valor_total,
                "CATEGORIA": context["category"],
            }
        return None

    def parse_line(line, context):
        line = line.strip()
        if not line:
            return None, context
        clean_line = line.replace(";", "").strip().upper()
        if clean_line in category_keywords:
            context["category"] = category_keywords[clean_line]
            return None, context
        if "ITEM:" in line.upper():
            match = re.search(r"ITEM:\s*([\d,\.]*)", line.upper())
            if match:
                context["apu_code"] = match.group(1).strip()
            if context["last_non_empty_line"]:
                context["apu_desc"] = (
                    context["last_non_empty_line"].split(";")[0].strip()
                )
            return None, context
        context["last_non_empty_line"] = line
        if ";" in line and context["apu_code"]:
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 6 and parts[0]:
                return parse_data_line(parts, context), context
        return None, context

    try:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                data_row, current_context = parse_line(line, current_context)
                if data_row:
                    apus_data.append(data_row)
        if not apus_data:
            return pd.DataFrame()
        df = pd.DataFrame(apus_data)
        df["CODIGO_APU"] = df["CODIGO_APU"].str.strip()
        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        return df
    except Exception as e:
        logger.error(f"Error procesando apus.csv con v2: {e}")
        return pd.DataFrame()


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
    df_apu_costos_categoria["VALOR_SUMINISTRO_UN"] = df_apu_costos_categoria[
        "MATERIALES"
    ]
    df_apu_costos_categoria["VALOR_INSTALACION_UN"] = (
        df_apu_costos_categoria["MANO DE OBRA"] + df_apu_costos_categoria["EQUIPO"]
    )
    df_apu_costos_categoria["VALOR_CONSTRUCCION_UN"] = df_apu_costos_categoria[
        cost_cols
    ].sum(axis=1)
    df_tiempo = (
        df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
        .groupby("CODIGO_APU")["CANTIDAD_APU"]
        .sum()
        .reset_index()
    )
    df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)
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
    df_merged["Vr Unitario"] = df_merged["VR_UNITARIO_FINAL"].fillna(0)
    df_merged["Vr Total"] = df_merged["COSTO_INSUMO_EN_APU"]
    df_merged_renamed = df_merged.rename(
        columns={"DESCRIPCION_INSUMO": "Descripción", "CANTIDAD_APU": "Cantidad"}
    )
    apus_detail = {
        n: g[
            ["Descripción", "Cantidad", "Vr Unitario", "Vr Total", "CATEGORIA"]
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
    return {
        "presupuesto": df_final.to_dict("records"),
        "insumos": insumos_dict,
        "apus_detail": apus_detail,
        "all_apus": df_apus.to_dict("records"),
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
    result = _cached_csv_processing(
        *file_hashes, presupuesto_path, apus_path, insumos_path
    )
    info = _cached_csv_processing.cache_info()
    logger.info(
        f"Cache info: hits={info.hits}, misses={info.misses}, size={info.currsize}"
    )
    return result


def calculate_estimate(
    params: Dict[str, str], data_store: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    log = []
    required_params = ["tipo", "material", "cuadrilla"]
    missing_params = [p for p in required_params if p not in params or not params[p]]
    if missing_params:
        error_msg = f"Parámetros requeridos faltantes o vacíos: {missing_params}"
        logger.warning(error_msg)
        return {"error": error_msg, "log": [error_msg]}

    def _normalize(text: str) -> str:
        return unidecode(str(text)).upper()

    param_map = config.get("param_map", {})
    all_apus_list = data_store.get("all_apus", [])
    if not all_apus_list:
        return {
            "error": "No hay datos de APU disponibles.",
            "log": ["Error: all_apus no encontrado en data_store."],
        }

    df_apus = pd.DataFrame(all_apus_list)
    df_apus_unique = df_apus.drop_duplicates(subset=["DESCRIPCION_APU"])
    apus_detail = data_store.get("apus_detail", {})

    tipo = params.get("tipo", "").upper()
    material = params.get("material", "").upper()
    cuadrilla = params.get("cuadrilla", "0")
    izaje = params.get("izaje", "manual").lower()
    seguridad = params.get("seguridad", "normal").lower()
    zona = params.get("zona", "zona 0").lower()
    log.append(f"Parámetros: {params}")

    material_mapped = param_map.get("material", {}).get(material, material)
    tipo_mapped = param_map.get("tipo", {}).get(tipo, tipo)

    search_terms = ["MANO DE OBRA", tipo_mapped, material_mapped]
    if cuadrilla != "0":
        search_terms.append(f"CUADRILLA DE {cuadrilla}")

    search_terms_normalized = [_normalize(term) for term in search_terms]
    log.append(f"Términos de búsqueda normalizados: {search_terms_normalized}")

    df_apus_unique["DESC_NORMALIZED"] = df_apus_unique["DESCRIPCION_APU"].apply(
        _normalize
    )

    apu_mo_code = None
    apu_mo_desc = ""

    ### CAMBIO CLAVE 2: Buscar en la lista de APUs, no en el presupuesto ###
    # La fuente correcta de todas las descripciones de APU es el dataframe de APUs.
    for _, apu_row in df_apus_unique.iterrows():
        desc_normalized = apu_row["DESC_NORMALIZED"]
        if all(term in desc_normalized for term in search_terms_normalized):
            apu_mo_code = apu_row.get("CODIGO_APU")
            apu_mo_desc = apu_row.get("DESCRIPCION_APU")
            log.append(
                f"ÉXITO: APU de M.O. encontrado:"
                f" '{apu_mo_desc}' (Código: {apu_mo_code})"
            )
            break

    if not apu_mo_code:
        log.append(
            "ERROR: No se encontró un APU de mano de obra"
            "que coincida con los criterios."
        )
        return {
            "error": "No se encontró un APU de mano de obra"
            "que coincida con los criterios.",
            "log": "\n".join(log),
        }

    costos_base = {"MATERIALES": 0.0, "MANO DE OBRA": 0.0, "EQUIPO": 0.0, "OTROS": 0.0}
    tiempo_instalacion = 0.0
    apu_items = apus_detail.get(apu_mo_code, [])
    if not apu_items:
        return {
            "error": f"El código APU {apu_mo_code} no tiene detalle.",
            "log": "\n".join(log),
        }

    for insumo in apu_items:
        categoria = insumo.get("CATEGORIA", "OTROS")
        costo = insumo.get("Vr Total", 0)
        if categoria in costos_base:
            costos_base[categoria] += costo
        if categoria == "MANO DE OBRA":
            tiempo_instalacion += insumo.get("Cantidad", 0)

    valor_suministro = costos_base["MATERIALES"]
    valor_instalacion = costos_base["MANO DE OBRA"] + costos_base["EQUIPO"]
    log.append(
        f"Costos base:"
        f" Suministro=${valor_suministro:,.0f}, Instalación=${valor_instalacion:,.0f}"
    )
    log.append(f"Tiempo base: {tiempo_instalacion:.4f} días/un")

    if izaje == "grúa":
        costo_grua_por_dia = 0.0
        # La búsqueda de la grúa debe hacerse en la lista completa de APUs
        for _, apu_row in df_apus_unique.iterrows():
            if "alquiler grua" in _normalize(apu_row.get("DESCRIPCION_APU", "")):
                codigo_grua = apu_row.get("CODIGO_APU")
                detalle_grua = apus_detail.get(codigo_grua, [])
                costo_total_apu_grua = sum(d.get("Vr Total", 0) for d in detalle_grua)
                costo_grua_por_dia = costo_total_apu_grua
                log.append(
                    f"APU de Grúa encontrado. Costo/día: ${costo_grua_por_dia:,.0f}"
                )
                break
        if costo_grua_por_dia > 0:
            costo_adicional_grua = costo_grua_por_dia * tiempo_instalacion
            valor_instalacion += costo_adicional_grua
            log.append(
                f"Ajuste por grúa"
                f" ({tiempo_instalacion:.4f} días): +${costo_adicional_grua:,.0f}"
            )
        else:
            log.append(
                "ADVERTENCIA: Izaje por grúa seleccionado"
                "pero no se encontró APU de grúa."
            )

    if seguridad == "alta":
        costo_mo = costos_base["MANO DE OBRA"]
        incremento_seguridad = costo_mo * 0.15
        valor_instalacion += incremento_seguridad
        log.append(
            f"Ajuste por seguridad alta (15% de M.O.): +${incremento_seguridad:,.0f}"
        )

    zona_factor = {"zona 0": 1.0, "zona 1": 1.05, "zona 2": 1.10, "zona 3": 1.15}
    factor = zona_factor.get(zona, 1.0)
    if factor > 1.0:
        costo_original_instalacion = valor_instalacion
        valor_instalacion *= factor
        incremento_zona = valor_instalacion - costo_original_instalacion
        log.append(
            f"Ajuste por {zona} ({(factor - 1) * 100:.0f}%): +${incremento_zona:,.0f}"
        )

    log.append(
        f"Valores finales:"
        f" Suministro=${valor_suministro:,.0f}, Instalación=${valor_instalacion:,.0f}"
    )

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": apu_mo_desc,
        "log": "\n".join(log),
    }
