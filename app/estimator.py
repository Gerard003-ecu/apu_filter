import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .utils import normalize_text

logger = logging.getLogger(__name__)


def _calculate_match_score(
    desc_words: set, keywords: List[str]
) -> Tuple[int, float]:
    """
    Calcula el puntaje de coincidencia entre una descripción normalizada y palabras clave.

    Args:
        desc_words (set): Conjunto de palabras de la descripción normalizada.
        keywords (List[str]): Lista de palabras clave a buscar (no vacía).

    Returns:
        Tuple[int, float]: Número de palabras coincidentes y porcentaje de cobertura.
    """
    if not keywords:
        return 0, 0.0

    matches = sum(1 for keyword in keywords if keyword in desc_words)
    percentage = (matches / len(keywords)) * 100.0
    return matches, percentage


def _find_best_match(
    df_pool: pd.DataFrame,
    keywords: List[str],
    log: List[str],
    strict: bool = False,
    min_match_percentage: float = 30.0,
    match_mode: str = 'words'
) -> Optional[pd.Series]:
    """
    Encuentra la mejor coincidencia de APU para una lista de palabras clave.
    Soporta modos: 'words' (coincidencia exacta de palabras) y 'substring' (contiene toda la cadena).

    Args:
        df_pool (pd.DataFrame): DataFrame con APUs procesados.
        keywords (List[str]): Palabras clave a buscar.
        log (List[str]): Lista de mensajes de log (mutables).
        strict (bool): Si True, requiere 100% de coincidencia.
        min_match_percentage (float): Umbral mínimo de coincidencia para modo flexible.
        match_mode (str): 'words' o 'substring'.

    Returns:
        pd.Series o None: El APU con mejor coincidencia, o None si no cumple criterios.
    """
    # Validación inicial de entradas
    if not isinstance(df_pool, pd.DataFrame):
        log.append("  ❌ ERROR: df_pool no es un DataFrame.")
        return None
    if df_pool.empty:
        log.append("  --> Pool vacío, retornando None.")
        return None
    if not keywords or not any(k.strip() for k in keywords):
        log.append("  --> Keywords vacías o nulas, retornando None.")
        return None

    # Normalizar keywords (limpiar espacios, convertir a minúsculas)
    keywords_clean = [k.strip().lower() for k in keywords if k.strip()]
    if not keywords_clean:
        log.append("  --> Después de limpiar, keywords vacías, retornando None.")
        return None

    log.append(f"  🔍 Buscando: {' '.join(keywords_clean)}")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    modo_str = "ESTRICTO (100%)" if strict else f"FLEXIBLE (≥{min_match_percentage:.0f}%)"
    log.append(f" ⚙️ Modo: {modo_str} | Estrategia: {match_mode}")

    best_match = None
    best_score = -1
    best_percentage = -1.0
    candidates = []

    for idx, apu in df_pool.iterrows():
        # Obtener descripción normalizada con manejo seguro
        desc_normalized = apu.get("DESC_NORMALIZED", "")
        if pd.isna(desc_normalized) or not isinstance(desc_normalized, str):
            desc_normalized = ""

        desc_normalized = desc_normalized.strip().lower()
        if not desc_normalized:
            continue  # Saltar descripciones vacías

        matches = 0
        percentage = 0.0

        if match_mode == 'words':
            desc_words = set(desc_normalized.split())
            matches, percentage = _calculate_match_score(desc_words, keywords_clean)
        elif match_mode == 'substring':
            keyword_str = ' '.join(keywords_clean)
            if keyword_str in desc_normalized:
                matches = len(keywords_clean)
                percentage = 100.0
        else:
            log.append(f"  ⚠️ Modo '{match_mode}' no soportado, saltando este APU.")
            continue

        if matches == 0:
            continue

        # Guardar candidato
        original_desc = apu.get("original_description", "").strip()
        if not original_desc:
            original_desc = "Descripción no disponible"

        candidates.append({
            'description': original_desc,
            'matches': matches,
            'percentage': percentage,
            'apu': apu
        })

        # Actualizar mejor coincidencia (prioriza matches, luego por %)
        if matches > best_score or (matches == best_score and percentage > best_percentage):
            best_match = apu
            best_score = matches
            best_percentage = percentage

    # Ordenar candidatos por relevancia
    candidates.sort(key=lambda x: (x['matches'], x['percentage']), reverse=True)

    # Mostrar top 3 candidatos
    if candidates:
        top_n = min(3, len(candidates))
        log.append(f"  📋 Top {top_n} candidatos:")
        for i, cand in enumerate(candidates[:top_n], 1):
            desc_snippet = cand['description'][:60] + "..." if len(cand['description']) > 60 else cand['description']
            log.append(
                f"    {i}. [{cand['matches']}/{len(keywords_clean)}] "
                f"({cand['percentage']:.0f}%) - {desc_snippet}"
            )

    # Decisión final
    if strict:
        if best_percentage == 100.0:
            log.append("  ✅ Match ESTRICTO encontrado!")
            return best_match
        else:
            log.append(f"  ❌ No se encontró match estricto (mejor: {best_percentage:.0f}%)")
            return None
    else:
        if best_percentage >= min_match_percentage:
            log.append(f"  ✅ Match FLEXIBLE encontrado ({best_percentage:.0f}%)")
            return best_match
        else:
            log.append(f"  ❌ Sin match válido (mejor: {best_percentage:.0f}% | umbral: {min_match_percentage:.0f}%)")
            return None


def calculate_estimate(
    params: Dict[str, str], data_store: Dict, config: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Estima el costo de construcción desglosado en suministro, cuadrilla y tarea.
    Usa coincidencias semánticas en APU para inferir valores.

    Args:
        params (Dict): Parámetros de entrada: material, cuadrilla, zona, izaje, seguridad.
        data_store (Dict): Almacén de datos: processed_apus, apus_detail.
        config (Dict): Configuración: param_map, estimator_rules.

    Returns:
        Dict: Resultado estimado con valores, descripciones y log detallado.
    """
    log: List[str] = []
    log.append("🕵️ ESTIMADOR DETECTIVE INICIADO")
    log.append("=" * 70)

    # ============================================
    # 1. CARGA Y VALIDACIÓN DE DATOS
    # ============================================
    processed_apus_list = data_store.get("processed_apus", [])
    if not isinstance(processed_apus_list, list):
        error_msg = "❌ ERROR: 'processed_apus' no es una lista."
        log.append(error_msg)
        return {"error": error_msg, "log": "\n".join(log)}

    if not processed_apus_list:
        error_msg = "❌ ERROR: No hay datos de APU procesados disponibles."
        log.append(error_msg)
        return {"error": error_msg, "log": "\n".join(log)}

    try:
        df_processed_apus = pd.DataFrame(processed_apus_list)
    except Exception as e:
        error_msg = f"❌ ERROR al convertir processed_apus a DataFrame: {str(e)}"
        log.append(error_msg)
        return {"error": error_msg, "log": "\n".join(log)}

    log.append(f"📚 Datos cargados: {len(df_processed_apus)} APUs disponibles")

    # Validar columnas críticas
    required_cols = {"DESC_NORMALIZED", "original_description", "tipo_apu", "VALOR_SUMINISTRO_UN", "UNIDAD", "EQUIPO", "CODIGO_APU"}
    missing_cols = required_cols - set(df_processed_apus.columns)
    if missing_cols:
        error_msg = f"❌ Columnas faltantes en APUs: {missing_cols}"
        log.append(error_msg)
        return {"error": error_msg, "log": "\n".join(log)}

    # Extraer parámetros con validación
    material = (params.get("material", "") or "").strip().upper()
    cuadrilla = (params.get("cuadrilla", "0") or "").strip()
    zona = (params.get("zona", "ZONA 0") or "").strip()
    izaje = (params.get("izaje", "MANUAL") or "").strip()
    seguridad = (params.get("seguridad", "NORMAL") or "").strip()

    log.append(f"📝 Material: '{material}' | Cuadrilla: '{cuadrilla}'")
    log.append(f"📝 Config: Zona='{zona}', Izaje='{izaje}', Seguridad='{seguridad}'")

    # Mapeo de material
    param_map = config.get("param_map", {})
    material_mapped = (param_map.get("material", {}).get(material, material) or "").strip()
    if material_mapped != material:
        log.append(f"🔄 Material mapeado: '{material}' → '{material_mapped}'")

    # Normalizar keywords de material
    material_keywords = normalize_text(material_mapped).split()
    if not material_keywords:
        log.append("⚠️ Material mapeado resultó en keywords vacías. La búsqueda de suministro y tarea será inefectiva.")

    # ============================================
    # 2. BÚSQUEDA DE SUMINISTRO
    # ============================================
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #1: SUMINISTRO")
    log.append("-" * 70)

    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"

    # Filtrar solo tipos de suministro
    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_processed_apus[
        df_processed_apus["tipo_apu"].isin(supply_types)
    ].copy()

    log.append(f"📦 Pool de suministros: {len(df_suministro_pool)} APUs")

    apu_suministro = _find_best_match(
        df_suministro_pool,
        material_keywords,
        log,
        strict=False,
        min_match_percentage=25.0
    )

    if apu_suministro is not None:
        valor_suministro = float(apu_suministro.get("VALOR_SUMINISTRO_UN", 0.0) or 0.0)
        apu_suministro_desc = str(apu_suministro.get("original_description", "")).strip()
        log.append(f"💰 Valor encontrado: ${valor_suministro:,.2f}")
    else:
        log.append("⚠️ No se encontró suministro")

    # ============================================
    # 3. BÚSQUEDA DE CUADRILLA
    # ============================================
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #2: CUADRILLA")
    log.append("-" * 70)

    costo_diario_cuadrilla = 0.0
    apu_cuadrilla_desc = "No encontrada"

    if cuadrilla and cuadrilla != "0":
        # Filtrar por UNIDAD == "DIA" (case-insensitive)
        df_cuadrilla_pool = df_processed_apus[
            df_processed_apus["UNIDAD"].astype(str).str.upper().str.strip() == "DIA"
        ].copy()

        log.append(f"👥 Pool de cuadrillas: {len(df_cuadrilla_pool)} APUs con UNIDAD=DIA")

        cuadrilla_mapped = (param_map.get("cuadrilla", {}).get(cuadrilla, cuadrilla) or "").strip()
        search_term = f"cuadrilla {cuadrilla_mapped}"
        cuadrilla_keywords_norm = normalize_text(search_term).split()

        apu_cuadrilla = _find_best_match(
            df_cuadrilla_pool,
            cuadrilla_keywords_norm,
            log,
            strict=False,
            min_match_percentage=50.0,
            match_mode='substring'  # Buscar la frase completa
        )

        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = float(apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0) or 0.0)
            apu_cuadrilla_desc = str(apu_cuadrilla.get("original_description", "")).strip()
            log.append(f"💰 Costo diario: ${costo_diario_cuadrilla:,.2f}")
        else:
            log.append("⚠️ No se encontró cuadrilla exacta")
    else:
        log.append("ℹ️ Sin cuadrilla especificada")

    # ============================================
    # 4. BÚSQUEDA DE TAREA (RENDIMIENTO Y EQUIPO)
    # ============================================
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #3: TAREA (RENDIMIENTO Y EQUIPO)")
    log.append("-" * 70)

    rendimiento_dia = 0.0
    costo_equipo = 0.0
    apu_tarea_desc = "No encontrado"

    df_tarea_pool = df_processed_apus[df_processed_apus["tipo_apu"] == "Instalación"].copy()
    log.append(f"🔧 Pool de instalación: {len(df_tarea_pool)} APUs")

    apu_tarea = _find_best_match(
        df_tarea_pool,
        material_keywords,
        log,
        strict=False,
        min_match_percentage=40.0
    )

    if apu_tarea is not None:
        apu_tarea_desc = str(apu_tarea.get("original_description", "")).strip()
        costo_equipo = float(apu_tarea.get("EQUIPO", 0.0) or 0.0)
        apu_code = str(apu_tarea.get("CODIGO_APU", "")).strip()

        log.append(f"📊 APU encontrado: {apu_code}")

        # Buscar rendimiento desde apus_detail
        apus_detail_list = data_store.get("apus_detail", [])
        if isinstance(apus_detail_list, list) and apus_detail_list:
            try:
                df_detail = pd.DataFrame(apus_detail_list)
                required_detail_cols = {"CODIGO_APU", "TIPO_INSUMO", "CANTIDAD_APU"}
                missing_detail = required_detail_cols - set(df_detail.columns)
                if missing_detail:
                    log.append(f"⚠️ Columnas faltantes en apus_detail: {missing_detail}")
                else:
                    apu_details = df_detail[df_detail["CODIGO_APU"].astype(str).str.strip() == apu_code]
                    mano_obra = apu_details[apu_details["TIPO_INSUMO"].astype(str).str.strip() == "MANO DE OBRA"]
                    if not mano_obra.empty:
                        tiempo_total = mano_obra["CANTIDAD_APU"].sum()
                        if tiempo_total > 0:
                            rendimiento_dia = 1.0 / tiempo_total
                            log.append(f"⏱️ Rendimiento calculado: {rendimiento_dia:.2f} un/día")
            except Exception as e:
                log.append(f"⚠️ Error procesando apus_detail: {str(e)}")
        else:
            log.append("ℹ️ No hay datos de apus_detail para calcular rendimiento")
    else:
        log.append("⚠️ No se encontró tarea de instalación")

    # Si no se encontró tarea, crear una sintética basada en el material
    if apu_tarea is None and apu_suministro is not None:
        apu_tarea_desc = f"INSTALACION {material_mapped}"
        log.append(f"✅ Tarea sintética creada: '{apu_tarea_desc}'")
        rendimiento_dia = 0.0
        costo_equipo = 0.0

    # ============================================
    # 5. CÁLCULO FINAL CON REGLAS DE NEGOCIO
    # ============================================
    log.append("\n" + "=" * 70)
    log.append("🧮 CÁLCULO FINAL CON REGLAS DE NEGOCIO")
    log.append("-" * 70)

    rules = config.get("estimator_rules", {})
    if not isinstance(rules, dict):
        log.append("⚠️ Configuración 'estimator_rules' no es un dict. Usando valores por defecto.")
        rules = {}

    factor_zona = rules.get("factores_zona", {}).get(zona, 1.0)
    costo_adicional_izaje = rules.get("costo_adicional_izaje", {}).get(izaje, 0)
    factor_seguridad = rules.get("factor_seguridad", {}).get(seguridad, 1.0)

    # Calcular costo de mano de obra base
    costo_mo_base = 0.0
    if rendimiento_dia > 0:
        costo_mo_base = costo_diario_cuadrilla / rendimiento_dia
    else:
        log.append("⚠️ Rendimiento = 0 → No se puede calcular costo de mano de obra base.")

    costo_mo_ajustado = costo_mo_base * factor_seguridad
    valor_instalacion = (
        (costo_mo_ajustado + costo_equipo) * factor_zona + costo_adicional_izaje
    )
    valor_construccion = valor_suministro + valor_instalacion

    # ============================================
    # RESUMEN FINAL
    # ============================================
    log.append("\n" + "=" * 70)
    log.append("📊 RESUMEN EJECUTIVO")
    log.append("=" * 70)
    log.append(f"📦 Suministro: ${valor_suministro:,.2f} ({apu_suministro_desc[:50]}{'...' if len(apu_suministro_desc) > 50 else ''})")
    log.append(f"🔨 Instalación:   ${valor_instalacion:,.2f}")
    log.append(f"💰 TOTAL:         ${valor_construccion:,.2f}")

    apu_encontrado_str = (
        f"Suministro: {apu_suministro_desc} | "
        f"Tarea: {apu_tarea_desc} | "
        f"Cuadrilla: {apu_cuadrilla_desc}"
    )

    # Retornar resultados limpios y tipados
    return {
        "valor_suministro": float(valor_suministro),
        "valor_instalacion": float(valor_instalacion),
        "valor_construccion": float(valor_construccion),
        "rendimiento_m2_por_dia": float(rendimiento_dia),
        "apu_encontrado": apu_encontrado_str,
        "log": "\n".join(log),
    }
