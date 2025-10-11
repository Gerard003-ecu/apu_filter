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

    # --- Búsqueda de Instalación (Nueva Lógica) ---
    log.append("\n--- BÚSQUEDA DE INSTALACIÓN (LÓGICA DE 2 PASOS) ---")
    valor_instalacion = 0.0
    tiempo_instalacion = 0.0
    rendimiento = 0.0
    costo_diario_cuadrilla = 0.0
    apu_tarea_desc = "No encontrado"
    apu_cuadrilla_desc = "No encontrado"

    # 1. Búsqueda del APU de Tarea (Rendimiento)
    log.append("\n--- 1. BÚSQUEDA APU DE TAREA (RENDIMIENTO) ---")
    task_keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()
    log.append(f"Palabras clave de tarea: {task_keywords}")

    df_instalacion_pool = df_apus[df_apus["tipo_apu"] == "Instalación"]
    apu_tarea = _find_best_match(df_instalacion_pool, task_keywords, log)

    if apu_tarea is not None:
        tiempo_instalacion = apu_tarea["TIEMPO_INSTALACION"]
        apu_tarea_desc = apu_tarea["original_description"]
        if tiempo_instalacion > 0:
            rendimiento = 1 / tiempo_instalacion
        log.append(f"APU de Tarea encontrado: '{apu_tarea_desc}'")
        log.append(f"  -> Tiempo Instalación: {tiempo_instalacion:.4f} días/un")
        log.append(f"  -> Rendimiento: {rendimiento:.2f} un/día")
    else:
        log.append("No se encontró APU de Tarea.")

    # 2. Búsqueda del APU de Cuadrilla (Costo Diario) - Lógica Específica
    log.append("\n--- 2. BÚSQUEDA APU DE CUADRILLA (COSTO) ---")
    if cuadrilla and cuadrilla != "0":
        # Crear la frase de búsqueda específica y normalizarla
        cuadrilla_phrase = f"cuadrilla de {cuadrilla}"
        normalized_phrase = normalize_text(pd.Series([cuadrilla_phrase])).iloc[0]
        log.append(f"Frase de búsqueda de cuadrilla: '{normalized_phrase}'")

        apu_cuadrilla = None
        # Búsqueda por subcadena exacta en la descripción normalizada
        for _, apu in df_apus.iterrows():
            desc_normalized = apu.get("DESC_NORMALIZED", "")
            if normalized_phrase in desc_normalized:
                apu_cuadrilla = apu
                log.append(f"  --> Coincidencia de cuadrilla encontrada: '{desc_normalized}'")
                break  # Tomar la primera coincidencia

        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = apu_cuadrilla["VALOR_CONSTRUCCION_UN"]
            apu_cuadrilla_desc = apu_cuadrilla["original_description"]
            log.append(f"APU de Cuadrilla encontrado: '{apu_cuadrilla_desc}'")
            log.append(f"  -> Costo Diario: ${costo_diario_cuadrilla:,.2f}")
        else:
            log.append("No se encontró APU de Cuadrilla.")
    else:
        log.append("No se especificó cuadrilla, se omite la búsqueda de costo de cuadrilla.")

    # 3. Cálculo del Costo de Instalación
    log.append("\n--- 3. CÁLCULO COSTO DE INSTALACIÓN ---")
    if rendimiento > 0:
        valor_instalacion = costo_diario_cuadrilla / rendimiento
        log.append(f"Cálculo: (Costo Diario / Rendimiento) = (${costo_diario_cuadrilla:,.2f} / {rendimiento:.2f}) = ${valor_instalacion:,.2f}")
    else:
        valor_instalacion = 0.0
        log.append("El rendimiento es 0, por lo que el costo de instalación es 0.")


    # --- Resultado Final ---
    log.append("\n--- RESULTADO FINAL ---")
    valor_construccion = valor_suministro + valor_instalacion
    log.append(f"Valor Construcción: (Suministro + Instalación) = (${valor_suministro:,.2f} + ${valor_instalacion:,.2f}) = ${valor_construccion:,.2f}")

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": f"Suministro: {apu_suministro_desc} | Tarea: {apu_tarea_desc} | Cuadrilla: {apu_cuadrilla_desc}",
        "log": "\n".join(log),
    }