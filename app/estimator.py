import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .utils import normalize_text

logger = logging.getLogger(__name__)

def _calculate_match_score(
    desc_words: set, keywords: List[str]
) -> Tuple[int, float]:
    """
    Calcula un puntaje de coincidencia para un APU.

    Returns:
        Tuple[int, float]: (número de palabras coincidentes, porcentaje de coincidencia)
    """
    matches = sum(1 for keyword in keywords if keyword in desc_words)
    percentage = (matches / len(keywords) * 100) if keywords else 0
    return matches, percentage


def _find_best_match(
    df_pool: pd.DataFrame,
    keywords: List[str],
    log: List[str],
    strict: bool = False,
    min_match_percentage: float = 30.0
) -> Optional[pd.Series]:
    if df_pool.empty or not keywords:
        log.append("  --> Pool vacío o sin keywords, retornando None.")
        return None

    log.append("--- INICIO DEPURACIÓN DETALLADA DE BÚSQUEDA ---")
    log.append(f"Buscando con keywords: {keywords}")
    log.append(f"  - Modo: {'ESTRICTO' if strict else f'FLEXIBLE (umbral: {min_match_percentage}%)'}")
    log.append(f"  - APUs en pool: {len(df_pool)}")
    log.append("  - Primeras 5 descripciones en el pool:")
    for desc in df_pool.head(5)['original_description']:
        log.append(f"    - {desc}")

    best_candidate = None
    highest_score = -1

    for _, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        desc_words = set(desc_normalized.split())

        matches = sum(1 for keyword in keywords if keyword in desc_words)

        if matches > 0:
            percentage = (matches / len(keywords)) * 100 if keywords else 0
            log.append(f"  - Candidato: '{desc_normalized}' -> Coincidencias: {matches}/{len(keywords)} ({percentage:.1f}%)")

            if percentage > highest_score:
                highest_score = percentage
                best_candidate = apu

    log.append("--- FIN DEPURACIÓN DETALLADA ---")

    if best_candidate is None:
        log.append("  ❌ No se encontraron candidatos con ninguna coincidencia.")
        return None

    log.append(f"  🏆 Mejor candidato encontrado: '{best_candidate.get('original_description')}' con un {highest_score:.1f}% de coincidencia.")

    # Aplicar criterios de selección
    if strict:
        if highest_score >= 100:
            log.append("  ✅ Éxito: Coincidencia estricta encontrada.")
            return best_candidate
        else:
            log.append("  ❌ Fallo: No se alcanzó el 100% requerido por el modo estricto.")
            return None
    else:
        if highest_score >= min_match_percentage:
            log.append(f"  ✅ Éxito: Supera el umbral flexible de {min_match_percentage}%.")
            return best_candidate
        else:
            log.append(f"  ❌ Fallo: No supera el umbral flexible de {min_match_percentage}%.")
            return None


def calculate_estimate(
    params: Dict[str, str], data_store: Dict, config: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    log = []

    # Validación de parámetros
    processed_apus_list = data_store.get("processed_apus", [])
    if not processed_apus_list:
        error_msg = "No hay datos de APU procesados disponibles."
        log.append(f"❌ ERROR: {error_msg}")
        return {"error": error_msg, "log": "\n".join(log)}

    df_apus = pd.DataFrame(processed_apus_list)

    material = params.get("material", "").upper()
    cuadrilla = params.get("cuadrilla", "0")
    log.append(f"📝 Parámetros de entrada: {params}")

    param_map = config.get("param_map", {})
    material_mapped = param_map.get("material", {}).get(material, material)
    log.append(f"🔄 Material mapeado: '{material}' → '{material_mapped}'")

    # --- Búsqueda de Suministro ---
    log.append("\n" + "="*60)
    log.append("🔍 BÚSQUEDA DE SUMINISTRO")
    log.append("="*60)

    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    supply_keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()

    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_apus[df_apus["tipo_apu"].isin(supply_types)]

    # Búsqueda flexible con 25% mínimo de coincidencia
    apu_encontrado = _find_best_match(
        df_suministro_pool,
        supply_keywords,
        log,
        strict=False,
        min_match_percentage=25.0
    )

    if apu_encontrado is not None:
        valor_suministro = apu_encontrado["VALOR_SUMINISTRO_UN"]
        apu_suministro_desc = apu_encontrado["original_description"]
        log.append(f"\n💰 Valor suministro: ${valor_suministro:,.2f}")
    else:
        log.append("\n⚠️ No se encontró APU de suministro.")

    # --- Búsqueda de Instalación ---
    log.append("\n" + "="*60)
    log.append("🔧 CÁLCULO DE INSTALACIÓN")
    log.append("="*60)

    valor_instalacion = 0.0
    apu_cuadrilla_desc = "No encontrada"
    apu_tarea_desc = "No encontrado"
    costo_diario_cuadrilla = 0.0
    rendimiento_m2_por_dia = 0.0
    costo_equipo_por_m2 = 0.0

    # Paso A: Encontrar el Costo Diario de la Cuadrilla
    log.append("\n📍 Paso A: Costo Diario de Cuadrilla")
    log.append("-" * 40)

    if cuadrilla and cuadrilla != "0":
        search_term = f"cuadrilla de {cuadrilla}"
        keywords_cuadrilla = normalize_text(pd.Series([search_term])).iloc[0].split()

        df_cuadrilla_pool = df_apus[
            df_apus["UNIDAD"].astype(str).str.upper() == "DIA"
        ].copy()

        # Búsqueda estricta para cuadrillas
        apu_cuadrilla = _find_best_match(
            df_cuadrilla_pool,
            keywords_cuadrilla,
            log,
            strict=True
        )

        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = apu_cuadrilla["VALOR_CONSTRUCCION_UN"]
            apu_cuadrilla_desc = apu_cuadrilla["original_description"]
            log.append(f"💰 Costo/día: ${costo_diario_cuadrilla:,.2f}")
        else:
            log.append("⚠️ No se encontró cuadrilla con UNIDAD: DIA.")
    else:
        log.append("ℹ️ No se especificó cuadrilla, costo diario es $0.")

    # Paso B: Encontrar Rendimiento y Costos de Equipo
    log.append("\n📍 Paso B: Rendimiento y Costo de Equipo")
    log.append("-" * 40)

    task_keywords = normalize_text(pd.Series([material_mapped])).iloc[0].split()

    df_instalacion_pool = df_apus[
        df_apus["tipo_apu"].isin(["Instalación", "Obra Completa"])
    ]

    # Búsqueda flexible con 30% mínimo
    apu_tarea = _find_best_match(
        df_instalacion_pool,
        task_keywords,
        log,
        strict=False,
        min_match_percentage=30.0
    )

    if apu_tarea is not None:
        apu_tarea_desc = apu_tarea["original_description"]
        apu_code = apu_tarea["CODIGO_APU"]
        costo_equipo_por_m2 = apu_tarea.get("EQUIPO", 0.0)
        log.append(f"\n📦 APU Tarea: {apu_code}")

        # Calcular rendimiento desde el desglose
        apus_detail_records = data_store.get("apus_detail", [])
        df_apus_detail = pd.DataFrame(apus_detail_records)

        if not df_apus_detail.empty:
            apu_details = df_apus_detail[df_apus_detail["CODIGO_APU"] == apu_code]
            mano_de_obra = apu_details[apu_details["TIPO_INSUMO"] == "MANO DE OBRA"]

            if not mano_de_obra.empty:
                tiempo_total_por_unidad = mano_de_obra["CANTIDAD_APU"].sum()
                log.append(f"⏱️ Tiempo total MO: {tiempo_total_por_unidad:.4f}")

                if tiempo_total_por_unidad > 0:
                    rendimiento_m2_por_dia = 1 / tiempo_total_por_unidad
                    log.append(f"📊 Rendimiento: {rendimiento_m2_por_dia:.2f} un/día")
                else:
                    log.append("⚠️ Tiempo total es 0.")
            else:
                log.append("⚠️ No hay insumos de MANO DE OBRA.")
        else:
            log.append("⚠️ No hay detalle de APUs disponible.")

        log.append(f"🔧 Costo Equipo: ${costo_equipo_por_m2:,.2f}/un")
    else:
        log.append("⚠️ No se encontró APU de tarea.")

    # Paso C: Cálculo Final
    log.append("\n📍 Paso C: Cálculo Final")
    log.append("-" * 40)

    costo_mo_por_m2 = 0.0
    if rendimiento_m2_por_dia > 0:
        costo_mo_por_m2 = costo_diario_cuadrilla / rendimiento_m2_por_dia
        log.append(
            f"💵 Costo MO/un = ${costo_diario_cuadrilla:,.2f} / {rendimiento_m2_por_dia:.2f} "
            f"= ${costo_mo_por_m2:,.2f}"
        )
    else:
        log.append("💵 Costo MO/un = $0 (rendimiento es 0)")

    valor_instalacion = costo_mo_por_m2 + costo_equipo_por_m2
    log.append(
        f"🔨 Instalación = ${costo_mo_por_m2:,.2f} + ${costo_equipo_por_m2:,.2f} "
        f"= ${valor_instalacion:,.2f}"
    )

    # --- Resultado Final ---
    log.append("\n" + "="*60)
    log.append("📊 RESULTADO FINAL")
    log.append("="*60)

    valor_construccion = valor_suministro + valor_instalacion
    log.append(f"💰 Suministro:    ${valor_suministro:,.2f}")
    log.append(f"🔨 Instalación:   ${valor_instalacion:,.2f}")
    log.append(f"{'─'*40}")
    log.append(f"💵 TOTAL:         ${valor_construccion:,.2f}")

    apu_encontrado_str = (
        f"Suministro: {apu_suministro_desc} | "
        f"Tarea: {apu_tarea_desc} | "
        f"Cuadrilla: {apu_cuadrilla_desc}"
    )

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "rendimiento_m2_por_dia": rendimiento_m2_por_dia,
        "apu_encontrado": apu_encontrado_str,
        "log": "\n".join(log),
    }
