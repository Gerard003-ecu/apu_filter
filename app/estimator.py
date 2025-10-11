import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from .procesador_csv import config, normalize_text

logger = logging.getLogger(__name__)

def _find_best_match(df_pool: pd.DataFrame, keywords: List[str], log: List[str]) -> Optional[pd.Series]:
    """
    Encuentra el mejor APU que coincida con una lista de palabras clave.
    Divide la descripción en palabras para una coincidencia más precisa.
    """
    if df_pool.empty or not keywords:
        return None

    log.append(f"Iniciando búsqueda con keywords: {keywords}")

    # Paso 1: Búsqueda estricta (todas las palabras clave)
    log.append("--- Iniciando Búsqueda Estricta ---")
    for _, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        desc_words = set(desc_normalized.split())
        log.append(f"Estricta: Probando '{desc_normalized}'. Palabras: {desc_words}. Buscando: {keywords}")
        if all(keyword in desc_words for keyword in keywords):
            log.append(f"  --> Coincidencia estricta encontrada: '{desc_normalized}'")
            return apu

    # Paso 2: Búsqueda flexible (cualquier palabra clave) si la estricta falla
    log.append("--- Iniciando Búsqueda Flexible ---")
    for _, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        desc_words = set(desc_normalized.split())
        log.append(f"Flexible: Probando '{desc_normalized}'. Palabras: {desc_words}. Buscando: {keywords}")
        if any(keyword in desc_words for keyword in keywords):
            log.append(f"  --> Coincidencia flexible encontrada: '{desc_normalized}'")
            return apu

    log.append("  --> No se encontraron coincidencias.")
    return None


def calculate_estimate(params: Dict[str, str], data_store: Dict) -> Dict[str, Union[str, float, List[str]]]:
    log = []
    # ... (validación de parámetros como antes) ...

    processed_apus_list = data_store.get("processed_apus", [])
    if not processed_apus_list:
        # ... (manejo de error) ...
        return {"error": "No hay datos de APU procesados disponibles."}
    df_apus = pd.DataFrame(processed_apus_list)

    material = params.get("material", "").upper()
    cuadrilla = params.get("cuadrilla", "0")
    log.append(f"Parámetros de entrada: {params}")

    param_map = config.get("param_map", {})
    material_mapped = param_map.get("material", {}).get(material, material)
    log.append(f"Parámetros mapeados: material='{material_mapped}'")

    # --- Búsqueda de Suministro ---
    log.append("\n--- BÚSQUEDA DE SUMINISTRO ---")
    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    supply_keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()

    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_apus[df_apus["tipo_apu"].isin(supply_types)]

    apu_encontrado = _find_best_match(df_suministro_pool, supply_keywords, log)
    if apu_encontrado is not None:
        valor_suministro = apu_encontrado["VALOR_SUMINISTRO_UN"]
        apu_suministro_desc = apu_encontrado["original_description"]

    # --- Búsqueda de Instalación ---
    log.append("\n--- BÚSQUEDA DE INSTALACIÓN ---")
    valor_instalacion = 0.0
    tiempo_instalacion = 0.0
    apu_instalacion_desc = "No encontrado"

    # Normalizar el material y la cuadrilla por separado para crear una lista de palabras clave
    install_keywords_base = normalize_text(pd.Series([material_mapped])).iloc[0].split()

    if cuadrilla and cuadrilla != "0":
        # Añadir el número de la cuadrilla como una palabra clave separada
        install_keywords = install_keywords_base + [str(cuadrilla)]
    else:
        install_keywords = install_keywords_base

    log.append(f"Palabras clave de instalación: {install_keywords}")

    df_instalacion_pool = df_apus[df_apus["tipo_apu"] == "Instalación"]

    apu_encontrado = _find_best_match(df_instalacion_pool, install_keywords, log)
    if apu_encontrado is not None:
        valor_instalacion = apu_encontrado["VALOR_INSTALACION_UN"]
        tiempo_instalacion = apu_encontrado["TIEMPO_INSTALACION"]
        apu_instalacion_desc = apu_encontrado["original_description"] # Usar la descripción original

    # --- Resultado ---
    valor_construccion = valor_suministro + valor_instalacion
    # ... (resto de la función para devolver el diccionario) ...
    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": f"Suministro: {apu_suministro_desc} | Instalación: {apu_instalacion_desc}",
        "log": "\n".join(log),
    }