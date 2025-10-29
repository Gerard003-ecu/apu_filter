import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .utils import normalize_text

logger = logging.getLogger(__name__)


def _calculate_match_score(
    desc_words: set, keywords: List[str]
) -> Tuple[int, float]:
    """Calcula un puntaje de coincidencia para un APU.

    Args:
        desc_words (set): Un conjunto de palabras de la descripción.
        keywords (List[str]): Una lista de palabras clave a buscar.

    Returns:
        Tuple[int, float]: Una tupla con el número de palabras coincidentes
                           y el porcentaje de coincidencia.
    """
    matches = sum(1 for keyword in keywords if keyword in desc_words)
    percentage = (matches / len(keywords) * 100) if keywords else 0
    return matches, percentage


def _find_best_match(
    df_pool: pd.DataFrame,
    keywords: List[str],
    log: List[str],
    strict: bool = False,
    min_match_percentage: float = 30.0,
    match_mode: str = 'words'
) -> Optional[pd.Series]:
    """Encuentra la mejor coincidencia de APU para una lista de palabras clave."""
    if df_pool.empty or not keywords:
        log.append("  --> Pool vacío o sin keywords, retornando None.")
        return None

    log.append(f"  🔍 Buscando: {' '.join(keywords)}")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    modo_str = "ESTRICTO (100%)" if strict else f"FLEXIBLE (≥{min_match_percentage}%)"
    log.append(f" ⚙️ Modo: {modo_str} | Estrategia: {match_mode}")

    best_match = None
    best_score = 0
    best_percentage = 0.0
    candidates = []

    for idx, apu in df_pool.iterrows():
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        if pd.isna(desc_normalized):
            desc_normalized = ""

        matches, percentage = 0, 0
        if match_mode == 'words':
            desc_words = set(desc_normalized.split())
            matches, percentage = _calculate_match_score(desc_words, keywords)
        elif match_mode == 'substring':
            keyword_str = ' '.join(keywords)
            if keyword_str in desc_normalized:
                matches = len(keywords)
                percentage = 100.0

        if matches > 0:
            candidates.append({
                'description': apu.get("original_description", ""),
                'matches': matches,
                'percentage': percentage,
                'apu': apu
            })

            if matches > best_score:
                best_match = apu
                best_score = matches
                best_percentage = percentage

    candidates.sort(key=lambda x: (x['matches'], x['percentage']), reverse=True)

    if candidates:
        log.append(f"  📋 Top {min(3, len(candidates))} candidatos:")
        for i, candidate in enumerate(candidates[:3], 1):
            log.append(
                f"    {i}. [{candidate['matches']}/{len(keywords)}] "
                f"({candidate['percentage']:.0f}%) - {candidate['description'][:60]}..."
            )

    if strict and best_percentage == 100.0:
        log.append("  ✅ Match ESTRICTO encontrado!")
        return best_match
    elif not strict and best_percentage >= min_match_percentage:
        log.append(f"  ✅ Match FLEXIBLE encontrado ({best_percentage:.0f}%)")
        return best_match
    else:
        log.append(f"  ❌ Sin match válido (mejor: {best_percentage:.0f}%)")
        return None


def calculate_estimate(
    params: Dict[str, str], data_store: Dict, config: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    """Busca componentes de forma atómica y ensambla el resultado."""
    log = []
    log.append("🕵️ ESTIMADOR DETECTIVE INICIADO")
    log.append("="*70)

    # ============================================
    # 1. CARGA DE DATOS
    # ============================================
    processed_apus_list = data_store.get("processed_apus", [])
    if not processed_apus_list:
        error_msg = "No hay datos de APU procesados disponibles."
        log.append(f"❌ ERROR: {error_msg}")
        return {"error": error_msg, "log": "\n".join(log)}

    df_processed_apus = pd.DataFrame(processed_apus_list)
    log.append(f"📚 Datos cargados: {len(df_processed_apus)} APUs disponibles")

    material = params.get("material", "").upper()
    cuadrilla = params.get("cuadrilla", "0")
    zona = params.get("zona", "ZONA 0")
    izaje = params.get("izaje", "MANUAL")
    seguridad = params.get("seguridad", "NORMAL")
    log.append(f"📝 Material: '{material}' | Cuadrilla: '{cuadrilla}'")
    log.append(f"📝 Config: Zona='{zona}', Izaje='{izaje}', Seguridad='{seguridad}'")

    param_map = config.get("param_map", {})
    material_mapped = param_map.get("material", {}).get(material, material)
    if material != material_mapped:
        log.append(f"🔄 Material mapeado: '{material}' → '{material_mapped}'")

    material_keywords = normalize_text(material_mapped).split()

    # ============================================
    # 2. BÚSQUEDA DE SUMINISTRO
    # ============================================
    log.append("\n" + "="*70)
    log.append("🎯 BÚSQUEDA #1: SUMINISTRO")
    log.append("-"*70)
    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_processed_apus[
        df_processed_apus["tipo_apu"].isin(supply_types)
    ].copy()
    log.append(f"📦 Pool de suministros: {len(df_suministro_pool)} APUs")
    apu_suministro = _find_best_match(
        df_suministro_pool, material_keywords, log, strict=False, min_match_percentage=25.0
    )
    if apu_suministro is not None:
        valor_suministro = apu_suministro.get("VALOR_SUMINISTRO_UN", 0.0)
        apu_suministro_desc = apu_suministro.get("original_description", "")
        log.append(f"💰 Valor encontrado: ${valor_suministro:,.2f}")
    else:
        log.append("⚠️ No se encontró suministro")

    # ============================================
    # 3. BÚSQUEDA DE CUADRILLA
    # ============================================
    log.append("\n" + "="*70)
    log.append("🎯 BÚSQUEDA #2: CUADRILLA")
    log.append("-"*70)
    costo_diario_cuadrilla = 0.0
    apu_cuadrilla_desc = "No encontrada"
    if cuadrilla and cuadrilla != "0":
        df_cuadrilla_pool = df_processed_apus[
            df_processed_apus["UNIDAD"].astype(str).str.upper() == "DIA"
        ].copy()
        log.append(f"👥 Pool de cuadrillas: {len(df_cuadrilla_pool)} APUs con UNIDAD=DIA")
        cuadrilla_mapped = param_map.get("cuadrilla", {}).get(cuadrilla, cuadrilla)
        search_term = f"cuadrilla {cuadrilla_mapped}"
        cuadrilla_keywords_norm = normalize_text(search_term).split()
        apu_cuadrilla = _find_best_match(
            df_cuadrilla_pool, cuadrilla_keywords_norm, log, strict=True, match_mode='substring'
        )
        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0)
            apu_cuadrilla_desc = apu_cuadrilla.get("original_description", "")
            log.append(f"💰 Costo diario: ${costo_diario_cuadrilla:,.2f}")
        else:
            log.append("⚠️ No se encontró cuadrilla exacta")
    else:
        log.append("ℹ️ Sin cuadrilla especificada")

    # ============================================
    # 4. BÚSQUEDA DE TAREA (RENDIMIENTO Y EQUIPO)
    # ============================================
    log.append("\n" + "="*70)
    log.append("🎯 BÚSQUEDA #3: TAREA (RENDIMIENTO Y EQUIPO)")
    log.append("-"*70)
    rendimiento_dia = 0.0
    costo_equipo = 0.0
    apu_tarea_desc = "No encontrado"
    df_tarea_pool = df_processed_apus[df_processed_apus["tipo_apu"] == "Instalación"].copy()
    log.append(f"🔧 Pool de instalación: {len(df_tarea_pool)} APUs")
    apu_tarea = _find_best_match(
        df_tarea_pool, material_keywords, log, strict=False, min_match_percentage=30.0
    )
    if apu_tarea is not None:
        apu_tarea_desc = apu_tarea.get("original_description", "")
        costo_equipo = apu_tarea.get("EQUIPO", 0.0)
        apu_code = apu_tarea.get("CODIGO_APU", "")
        log.append(f"📊 APU encontrado: {apu_code}")
        apus_detail_list = data_store.get("apus_detail", [])
        if apus_detail_list:
            df_detail = pd.DataFrame(apus_detail_list)
            apu_details = df_detail[df_detail["CODIGO_APU"] == apu_code]
            mano_obra = apu_details[apu_details["TIPO_INSUMO"] == "MANO DE OBRA"]
            if not mano_obra.empty:
                tiempo_total = mano_obra["CANTIDAD_APU"].sum()
                if tiempo_total > 0:
                    rendimiento_dia = 1 / tiempo_total
                    log.append(f"⏱️ Rendimiento calculado: {rendimiento_dia:.2f} un/día")
    else:
        log.append("⚠️ No se encontró tarea de instalación")

    if apu_tarea is None and apu_suministro is not None:
        apu_tarea_desc = f"INSTALACION {material_mapped}"
        log.append(f"✅ Tarea sintética creada: '{apu_tarea_desc}'")
        rendimiento_dia = 0.0
        costo_equipo = 0.0

    # ============================================
    # 5. CÁLCULO FINAL CON REGLAS DE NEGOCIO
    # ============================================
    log.append("\n" + "="*70)
    log.append("🧮 CÁLCULO FINAL CON REGLAS DE NEGOCIO")
    log.append("-"*70)
    rules = config.get("estimator_rules", {})
    factor_zona = rules.get("factores_zona", {}).get(zona, 1.0)
    costo_adicional_izaje = rules.get("costo_adicional_izaje", {}).get(izaje, 0)
    factor_seguridad = rules.get("factor_seguridad", {}).get(seguridad, 1.0)
    costo_mo_base = 0.0
    if rendimiento_dia > 0:
        costo_mo_base = costo_diario_cuadrilla / rendimiento_dia
    costo_mo_ajustado = costo_mo_base * factor_seguridad
    valor_instalacion = (
        (costo_mo_ajustado + costo_equipo) * factor_zona + costo_adicional_izaje
    )
    valor_construccion = valor_suministro + valor_instalacion

    # ============================================
    # RESUMEN FINAL
    # ============================================
    log.append("\n" + "="*70)
    log.append("📊 RESUMEN EJECUTIVO")
    log.append("="*70)
    log.append(f"📦 Suministro: ${valor_suministro:,.2f} ({apu_suministro_desc[:50]}...)")
    log.append(f"🔨 Instalación:   ${valor_instalacion:,.2f}")
    log.append(f"💰 TOTAL:         ${valor_construccion:,.2f}")

    apu_encontrado_str = (
        f"Suministro: {apu_suministro_desc} | "
        f"Tarea: {apu_tarea_desc} | "
        f"Cuadrilla: {apu_cuadrilla_desc}"
    )

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "rendimiento_m2_por_dia": rendimiento_dia,
        "apu_encontrado": apu_encontrado_str,
        "log": "\n".join(log),
    }
