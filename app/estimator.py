import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from .utils import normalize_text

logger = logging.getLogger(__name__)

def _find_best_match(df_pool: pd.DataFrame, keywords: List[str], log: List[str], strict: bool = False) -> Optional[pd.Series]:
    """
    Encuentra el mejor APU que coincida con una lista de palabras clave.
    Soporta búsqueda estricta (todas las keywords) y flexible (cualquiera).
    """
    if df_pool.empty or not keywords:
        return None

    log.append(f"Iniciando búsqueda con keywords: {keywords} (Strict={strict})")

    # Búsqueda estricta (todas las palabras clave)
    log.append("--- Iniciando Búsqueda Estricta ---")
    for _, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        desc_words = set(desc_normalized.split())
        if all(keyword in desc_words for keyword in keywords):
            log.append(f"  --> Coincidencia estricta encontrada: '{desc_normalized}'")
            return apu

    if strict:
        log.append("  --> No se encontraron coincidencias (búsqueda estricta finalizada).")
        return None

    # Búsqueda flexible (cualquier palabra clave) si la estricta falla
    log.append("--- Iniciando Búsqueda Flexible ---")
    for _, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        desc_words = set(desc_normalized.split())
        if any(keyword in desc_words for keyword in keywords):
            log.append(f"  --> Coincidencia flexible encontrada: '{desc_normalized}'")
            return apu

    log.append("  --> No se encontraron coincidencias.")
    return None


def calculate_estimate(params: Dict[str, str], data_store: Dict, config: Dict) -> Dict[str, Union[str, float, List[str]]]:
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

    # --- Búsqueda de Instalación (Nueva Lógica de Cálculo) ---
    log.append("\n--- CÁLCULO DE INSTALACIÓN (NUEVO ALGORITMO) ---")
    valor_instalacion = 0.0
    apu_cuadrilla_desc = "No encontrada"
    apu_tarea_desc = "No encontrado"
    costo_diario_cuadrilla = 0.0
    rendimiento_m2_por_dia = 0.0
    costo_equipo_por_m2 = 0.0

    # Paso A: Encontrar el Costo Diario de la Cuadrilla
    log.append("\n--- Paso A: Encontrar Costo Diario de Cuadrilla ---")
    if cuadrilla and cuadrilla != "0":
        search_term = f"cuadrilla de {cuadrilla}"
        keywords_cuadrilla = normalize_text(pd.Series([search_term])).iloc[0].split()
        log.append(f"Buscando cuadrilla con keywords: {keywords_cuadrilla} y UNIDAD: DIA")

        # Usar una copia para evitar SettingWithCopyWarning
        df_cuadrilla_pool = df_apus[df_apus["UNIDAD"].astype(str).str.upper() == "DIA"].copy()

        apu_cuadrilla = _find_best_match(df_cuadrilla_pool, keywords_cuadrilla, log, strict=True)

        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = apu_cuadrilla["VALOR_CONSTRUCCION_UN"]
            apu_cuadrilla_desc = apu_cuadrilla["original_description"]
            log.append(f"  --> Cuadrilla encontrada: '{apu_cuadrilla_desc}'. Costo/día: ${costo_diario_cuadrilla:,.0f}")
        else:
            log.append("  --> No se encontró APU para la cuadrilla especificada con UNIDAD: DIA.")
    else:
        log.append("  --> No se especificó cuadrilla, costo diario es $0.")

    # Paso B: Encontrar Rendimiento y Costos de Equipo para la Tarea
    log.append("\n--- Paso B: Encontrar Rendimiento y Costo de Equipo de Tarea ---")
    task_keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()
    log.append(f"Buscando tarea con keywords: {task_keywords}")

    # Buscar en APUs que sean principalmente de instalación
    df_instalacion_pool = df_apus[df_apus["tipo_apu"].isin(["Instalación", "Obra Completa"])]
    apu_tarea = _find_best_match(df_instalacion_pool, task_keywords, log)

    if apu_tarea is not None:
        apu_tarea_desc = apu_tarea["original_description"]
        apu_code = apu_tarea["CODIGO_APU"]
        costo_equipo_por_m2 = apu_tarea.get("EQUIPO", 0.0)
        log.append(f"  --> Tarea encontrada: '{apu_tarea_desc}' (Código: {apu_code})")

        # Nueva lógica para calcular rendimiento desde el desglose
        # 'apus_detail' ahora es una lista plana de registros, no un diccionario.
        apus_detail_records = data_store.get("apus_detail", [])
        df_apus_detail = pd.DataFrame(apus_detail_records)

        if not df_apus_detail.empty:
            apu_details = df_apus_detail[df_apus_detail["CODIGO_APU"] == apu_code]
            mano_de_obra = apu_details[apu_details["TIPO_INSUMO"] == "MANO DE OBRA"]

            if not mano_de_obra.empty:
                tiempo_total_por_unidad = mano_de_obra["CANTIDAD_APU"].sum()
                log.append(f"      - Insumos 'MANO DE OBRA' encontrados: {len(mano_de_obra)}")
                log.append(f"      - Tiempo total sumado (Cantidades): {tiempo_total_por_unidad:.4f}")

                if tiempo_total_por_unidad > 0:
                    rendimiento_m2_por_dia = 1 / tiempo_total_por_unidad
                    log.append(f"      - Rendimiento calculado: 1 / {tiempo_total_por_unidad:.4f} = {rendimiento_m2_por_dia:.2f} un/día")
                else:
                    log.append("      - Tiempo total es 0, no se puede calcular rendimiento.")
            else:
                log.append("      - No se encontraron insumos de 'MANO DE OBRA' para este APU.")
        else:
            log.append("      - No se encontró el detalle de APUs (apus_detail) para calcular el rendimiento.")

        log.append(f"      - Costo Equipo por Unidad: ${costo_equipo_por_m2:,.2f}")
    else:
        log.append("  --> No se encontró APU de tarea coincidente.")

    # Paso C: El Cálculo Final del Costo de Instalación
    log.append("\n--- Paso C: Cálculo Final del Costo de Instalación ---")
    costo_mo_por_m2 = 0.0
    if rendimiento_m2_por_dia > 0:
        costo_mo_por_m2 = costo_diario_cuadrilla / rendimiento_m2_por_dia
        log.append(f"Costo MO por m² = (Costo Diario Cuadrilla / Rendimiento) = ${costo_diario_cuadrilla:,.2f} / {rendimiento_m2_por_dia:.2f} = ${costo_mo_por_m2:,.2f}")
    else:
        log.append("Costo MO por m² = $0 (Rendimiento es 0, se evita división por cero).")

    valor_instalacion = costo_mo_por_m2 + costo_equipo_por_m2
    log.append(f"Valor Instalación Final = (Costo MO por m² + Costo Equipo por m²) = ${costo_mo_por_m2:,.2f} + ${costo_equipo_por_m2:,.2f} = ${valor_instalacion:,.2f}")


    # --- Resultado Final ---
    log.append("\n--- RESULTADO FINAL ---")
    valor_construccion = valor_suministro + valor_instalacion
    log.append(f"Valor Construcción: (Suministro + Instalación) = (${valor_suministro:,.2f} + ${valor_instalacion:,.2f}) = ${valor_construccion:,.2f}")

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "rendimiento_m2_por_dia": rendimiento_m2_por_dia,
        "apu_encontrado": f"Suministro: {apu_suministro_desc} | Tarea: {apu_tarea_desc} | Cuadrilla: {apu_cuadrilla_desc}",
        "log": "\n".join(log),
    }
