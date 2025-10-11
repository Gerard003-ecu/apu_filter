import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from .procesador_csv import config, normalize_text

logger = logging.getLogger(__name__)

def _find_apu_by_keywords(df_pool: pd.DataFrame, keywords: List[str], log: List[str]) -> Optional[pd.Series]:
    """Función auxiliar para buscar un APU por palabras clave."""
    log.append(f"Buscando con palabras clave: {keywords} en {len(df_pool)} APUs.")
    for _, apu in df_pool.iterrows():
        desc_to_search = apu.get("DESC_NORMALIZED", "")
        log.append(f"  ... comparando con: '{desc_to_search}'")
        if all(keyword in desc_to_search for keyword in keywords):
            log.append("  --> ¡Coincidencia encontrada!")
            return apu
    log.append("  --> No se encontraron coincidencias.")
    return None

def calculate_estimate(params: Dict[str, str], data_store: Dict) -> Dict[str, Union[str, float, List[str]]]:
    log = []
    # ... (validación de parámetros como antes) ...

    processed_apus_list = data_store.get("processed_apus", [])
    if not processed_apus_list:
        # ... (manejo de error como antes) ...
        return {
            "valor_suministro": 0,
            "valor_instalacion": 0,
            "valor_construccion": 0,
            "tiempo_instalacion": 0,
            "apu_encontrado": "Error: 'processed_apus' no encontrado en data_store.",
            "log": "Error: 'processed_apus' no encontrado en data_store.",
        }
    df_apus = pd.DataFrame(processed_apus_list)

    material = params.get("material", "").upper()
    log.append(f"Parámetros de entrada: {params}")

    param_map = config.get("param_map", {})
    material_mapped = param_map.get("material", {}).get(material, material)
    log.append(f"Parámetros mapeados: material='{material_mapped}'")

    keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()

    # --- 1. Búsqueda de Suministro ---
    log.append("\n--- BÚSQUEDA DE SUMINISTRO ---")
    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"

    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_apus[df_apus["tipo_apu"].isin(supply_types)]

    apu_encontrado = _find_apu_by_keywords(df_suministro_pool, keywords, log)
    if apu_encontrado is not None:
        valor_suministro = apu_encontrado["VALOR_SUMINISTRO_UN"]
        apu_suministro_desc = apu_encontrado["original_description"]
        log.append(f"APU de Suministro encontrado: '{apu_suministro_desc}'. Valor: ${valor_suministro:,.0f}")
    else:
        log.append("No se encontró APU de suministro. Fallback a insumos no implementado.")

    # --- 2. Búsqueda de Instalación ---
    log.append("\n--- BÚSQUEDA DE INSTALACIÓN ---")
    valor_instalacion = 0.0
    tiempo_instalacion = 0.0
    apu_instalacion_desc = "No encontrado"

    df_instalacion_pool = df_apus[df_apus["tipo_apu"] == "Instalación"]

    apu_encontrado = _find_apu_by_keywords(df_instalacion_pool, keywords, log)
    if apu_encontrado is not None:
        valor_instalacion = apu_encontrado["VALOR_INSTALACION_UN"]
        tiempo_instalacion = apu_encontrado["TIEMPO_INSTALACION"]
        apu_instalacion_desc = apu_encontrado["original_description"]
        log.append(f"APU de Instalación encontrado: '{apu_instalacion_desc}'. Valor: ${valor_instalacion:,.0f}")
    else:
        log.append("No se encontró APU de instalación.")

    # --- 3. Devolver Resultado ---
    # ... (resto de la función como antes) ...
    valor_construccion = valor_suministro + valor_instalacion
    log.append(
        f"\n--- RESULTADO COMPUESTO ---\n"
        f"Valor Suministro: ${valor_suministro:,.0f}\n"
        f"Valor Instalación: ${valor_instalacion:,.0f}\n"
        f"Valor Construcción: ${valor_construccion:,.0f}"
    )

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": f"Suministro: {apu_suministro_desc} | Instalación: {apu_instalacion_desc}",
        "log": "\n".join(log),
    }
