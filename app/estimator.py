import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from flask import current_app
from sentence_transformers import SentenceTransformer

from .utils import normalize_text

logger = logging.getLogger(__name__)


def _calculate_match_score(desc_words: set, keywords: List[str]) -> Tuple[int, float]:
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


def _find_best_keyword_match(
    df_pool: pd.DataFrame,
    keywords: List[str],
    log: List[str],
    strict: bool = False,
    min_match_percentage: float = 30.0,
    match_mode: str = "words",
) -> Optional[pd.Series]:
    """
    Encuentra la mejor coincidencia de APU para una lista de palabras clave.
    Soporta modos: 'words' (coincidencia exacta de palabras) y 'substring'
    (contiene toda la cadena).

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

        if match_mode == "words":
            desc_words = set(desc_normalized.split())
            matches, percentage = _calculate_match_score(desc_words, keywords_clean)
        elif match_mode == "substring":
            keyword_str = " ".join(keywords_clean)
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

        candidates.append(
            {
                "description": original_desc,
                "matches": matches,
                "percentage": percentage,
                "apu": apu,
            }
        )

        # Actualizar mejor coincidencia (prioriza matches, luego por %)
        if matches > best_score or (matches == best_score and percentage > best_percentage):
            best_match = apu
            best_score = matches
            best_percentage = percentage

    # Ordenar candidatos por relevancia
    candidates.sort(key=lambda x: (x["matches"], x["percentage"]), reverse=True)

    # Mostrar top 3 candidatos
    if candidates:
        top_n = min(3, len(candidates))
        log.append(f"  📋 Top {top_n} candidatos:")
        for i, cand in enumerate(candidates[:top_n], 1):
            desc_snippet = (
                cand["description"][:60] + "..."
                if len(cand["description"]) > 60
                else cand["description"]
            )
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
            log.append(
                f"  ❌ Sin match válido (mejor: {best_percentage:.0f}% | "
                f"umbral: {min_match_percentage:.0f}%)"
            )
            return None


def _find_best_semantic_match(
    df_pool: pd.DataFrame,
    query_text: str,
    log: List[str],
    min_similarity: float = 0.5,
    top_k: int = 5,
) -> Optional[pd.Series]:
    """
    Encuentra la mejor coincidencia semántica para un texto de consulta.

    Utiliza un índice FAISS y embeddings de `sentence-transformers` cargados
    en la aplicación Flask para encontrar los APUs más relevantes.

    Args:
        df_pool (pd.DataFrame): DataFrame con APUs procesados a considerar.
        query_text (str): Texto de consulta (ej. "muro de ladrillo").
        log (List[str]): Lista de mensajes de log (mutable).
        min_similarity (float): Umbral mínimo de similitud de coseno para
                                considerar una coincidencia válida.
        top_k (int): Número de vecinos a buscar en el índice FAISS.

    Returns:
        pd.Series o None: El APU con mejor coincidencia, o None si no cumple criterios.
    """
    # --- 1. Obtener artefactos de búsqueda semántica desde la app ---
    model: Optional[SentenceTransformer] = current_app.config.get("EMBEDDING_MODEL")
    faiss_index = current_app.config.get("FAISS_INDEX")
    id_map: Optional[dict] = current_app.config.get("ID_MAP")

    if not all([model, faiss_index, id_map]):
        log.append(
            "  ❌ ERROR: Faltan artefactos de búsqueda semántica. "
            "La función está desactivada."
        )
        return None

    # --- 2. Validación de entradas ---
    if not isinstance(df_pool, pd.DataFrame) or df_pool.empty:
        log.append("  --> Pool de APUs vacío, retornando None.")
        return None
    if not query_text or not query_text.strip():
        log.append("  --> Texto de consulta vacío, retornando None.")
        return None

    log.append(f"  🧠 Búsqueda Semántica: '{query_text}'")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    log.append(f"  ⚙️ Umbral de similitud: {min_similarity:.2f}")

    # --- 3. Generar embedding para la consulta ---
    try:
        query_embedding = model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception as e:
        log.append(f"  ❌ ERROR al generar embedding para la consulta: {e}")
        return None

    # --- 4. Buscar en el índice FAISS ---
    try:
        # FAISS espera un array 2D de float32
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), k=top_k)
    except Exception as e:
        log.append(f"  ❌ ERROR durante la búsqueda en FAISS: {e}")
        return None

    # --- 5. Procesar y filtrar resultados ---
    candidates = []
    if indices.size == 0 or distances.size == 0:
        log.append("  --> No se encontraron resultados en el índice.")
        return None

    # Iterar sobre los resultados encontrados
    for i in range(len(indices[0])):
        faiss_idx = indices[0][i]
        similarity = distances[0][i]

        # Obtener el CODIGO_APU desde el mapeo
        apu_code = id_map.get(str(faiss_idx))
        if apu_code is None:
            log.append(f"  ⚠️ Índice FAISS {faiss_idx} no encontrado en el mapa de IDs.")
            continue

        # Buscar el APU en el DataFrame del pool actual
        apu_match = df_pool[df_pool["CODIGO_APU"] == apu_code]

        if not apu_match.empty:
            candidates.append(
                {
                    "apu": apu_match.iloc[0],
                    "similarity": float(similarity),
                }
            )

    if not candidates:
        log.append("  --> Ninguno de los candidatos del índice estaba en el pool filtrado.")
        return None

    # --- 6. Log y selección final ---
    # Ordenar candidatos por similitud
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    log.append(f"  📋 Top {len(candidates)} candidatos encontrados:")
    for i, cand in enumerate(candidates, 1):
        apu = cand["apu"]
        desc = apu.get("original_description", "N/A")
        desc_snippet = desc[:70] + "..." if len(desc) > 70 else desc
        log.append(
            f"    {i}. Sim: {cand['similarity']:.3f} | "
            f"Código: {apu.get('CODIGO_APU')} | Desc: {desc_snippet}"
        )

    # Seleccionar el mejor candidato que cumpla el umbral
    best_candidate = candidates[0]
    if best_candidate["similarity"] >= min_similarity:
        log.append(
            f"  ✅ Coincidencia encontrada con similitud "
            f"({best_candidate['similarity']:.3f}) >= {min_similarity:.2f}"
        )
        return best_candidate["apu"]
    else:
        log.append(
            f"  ❌ Sin coincidencia válida. Mejor similitud "
            f"({best_candidate['similarity']:.3f}) < umbral ({min_similarity:.2f})"
        )
        return None


def calculate_estimate(
    params: Dict[str, str], data_store: Dict, config: Dict
) -> Dict[str, Union[str, float, List[str]]]:
    """
    Estima el costo de construcción con una estrategia de búsqueda híbrida.
    Prioriza la búsqueda semántica y recurre a la búsqueda por palabras clave.
    """
    log: List[str] = ["🕵️ ESTIMADOR HÍBRIDO INICIADO"]
    log.append("=" * 70)

    # --- 1. Carga y Validación de Datos ---
    df_processed_apus = pd.DataFrame(data_store.get("processed_apus", []))
    if df_processed_apus.empty:
        return {"error": "No hay datos de APU procesados.", "log": "\n".join(log)}

    material = (params.get("material", "") or "").strip().upper()
    cuadrilla = (params.get("cuadrilla", "0") or "").strip()
    # ... (resto de la extracción de parámetros)

    param_map = config.get("param_map", {})
    material_mapped = (param_map.get("material", {}).get(material, material) or "").strip()
    material_keywords = normalize_text(material_mapped).split()

    thresholds = config.get("estimator_thresholds", {})
    min_sim_suministro = thresholds.get("min_semantic_similarity_suministro", 0.30)
    min_sim_tarea = thresholds.get("min_semantic_similarity_tarea", 0.40)
    min_kw_cuadrilla = thresholds.get("min_keyword_match_percentage_cuadrilla", 50.0)

    # --- 2. Búsqueda de Suministro (Híbrida) ---
    log.append("\n" + "=" * 70 + "\n🎯 BÚSQUEDA #1: SUMINISTRO\n" + "-" * 70)
    supply_types = ["Suministro", "Suministro (Pre-fabricado)"]
    df_suministro_pool = df_processed_apus[df_processed_apus["tipo_apu"].isin(supply_types)]

    apu_suministro = _find_best_semantic_match(
        df_pool=df_suministro_pool,
        query_text=material_mapped,
        log=log,
        min_similarity=min_sim_suministro,
    )

    if apu_suministro is None:
        log.append("\n  --> Búsqueda semántica sin éxito. Recurriendo a palabras clave...")
        apu_suministro = _find_best_keyword_match(
            df_suministro_pool, material_keywords, log, strict=False
        )

    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    if apu_suministro is not None:
        valor_suministro = float(apu_suministro.get("VALOR_SUMINISTRO_UN", 0.0) or 0.0)
        apu_suministro_desc = str(apu_suministro.get("original_description", "")).strip()
        log.append(f"💰 Valor Suministro: ${valor_suministro:,.2f}")

    # --- 3. Búsqueda de Cuadrilla ---
    # (La búsqueda de cuadrilla sigue siendo por palabras clave,
    # ya que es más precisa para ese caso)
    log.append("\n" + "=" * 70 + "\n🎯 BÚSQUEDA #2: CUADRILLA\n" + "-" * 70)
    costo_diario_cuadrilla = 0.0
    apu_cuadrilla_desc = "No encontrada"
    if cuadrilla and cuadrilla != "0":
        df_cuadrilla_pool = df_processed_apus[
            df_processed_apus["UNIDAD"].astype(str).str.upper().str.strip() == "DIA"
        ]
        search_term = f"cuadrilla {cuadrilla}"
        cuadrilla_keywords = normalize_text(search_term).split()
        apu_cuadrilla = _find_best_keyword_match(
            df_cuadrilla_pool, cuadrilla_keywords, log, min_match_percentage=min_kw_cuadrilla
        )
        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = float(
                apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0) or 0.0
            )
            apu_cuadrilla_desc = str(apu_cuadrilla.get("original_description", "")).strip()
            log.append(f"💰 Costo Cuadrilla: ${costo_diario_cuadrilla:,.2f}/día")

    # --- 4. Búsqueda de Tarea (Híbrida) ---
    log.append("\n" + "=" * 70 + "\n🎯 BÚSQUEDA #3: TAREA (RENDIMIENTO)\n" + "-" * 70)
    df_tarea_pool = df_processed_apus[df_processed_apus["tipo_apu"] == "Instalación"]

    apu_tarea = _find_best_semantic_match(
        df_pool=df_tarea_pool,
        query_text=material_mapped,
        log=log,
        min_similarity=min_sim_tarea,
    )

    if apu_tarea is None:
        log.append("\n  --> Búsqueda semántica sin éxito. Recurriendo a palabras clave...")
        apu_tarea = _find_best_keyword_match(
            df_tarea_pool, material_keywords, log, strict=False
        )

    rendimiento_dia = 0.0
    costo_equipo = 0.0
    apu_tarea_desc = "No encontrado"
    if apu_tarea is not None:
        apu_tarea_desc = str(apu_tarea.get("original_description", "")).strip()
        costo_equipo = float(apu_tarea.get("EQUIPO", 0.0) or 0.0)
        apu_code = str(apu_tarea.get("CODIGO_APU", "")).strip()

        # Calcular rendimiento desde apus_detail
        apus_detail_list = data_store.get("apus_detail", [])
        if apus_detail_list:
            df_detail = pd.DataFrame(apus_detail_list)
            mano_obra = df_detail[
                (df_detail["CODIGO_APU"] == apu_code)
                & (df_detail["TIPO_INSUMO"] == "MANO DE OBRA")
            ]
            tiempo_total = mano_obra["CANTIDAD_APU"].sum()
            if tiempo_total > 0:
                rendimiento_dia = 1.0 / tiempo_total
                log.append(f"⏱️ Rendimiento: {rendimiento_dia:.2f} un/día")

    # --- 5. Cálculo Final y Resultados ---
    log.append("\n" + "=" * 70 + "\n🧮 CÁLCULO FINAL\n" + "-" * 70)
    # ... (Lógica de cálculo de costos con reglas de negocio, sin cambios)
    zona = (params.get("zona", "ZONA 0") or "").strip()
    izaje = (params.get("izaje", "MANUAL") or "").strip()
    seguridad = (params.get("seguridad", "NORMAL") or "").strip()
    rules = config.get("estimator_rules", {})
    factor_zona = rules.get("factores_zona", {}).get(zona, 1.0)
    costo_adicional_izaje = rules.get("costo_adicional_izaje", {}).get(izaje, 0)
    factor_seguridad = rules.get("factor_seguridad", {}).get(seguridad, 1.0)

    costo_mo_base = costo_diario_cuadrilla / rendimiento_dia if rendimiento_dia > 0 else 0
    costo_mo_ajustado = costo_mo_base * factor_seguridad
    valor_instalacion = (
        costo_mo_ajustado + costo_equipo
    ) * factor_zona + costo_adicional_izaje
    valor_construccion = valor_suministro + valor_instalacion

    log.append(f"💰 TOTAL CONSTRUCCIÓN: ${valor_construccion:,.2f}")

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "valor_construccion": valor_construccion,
        "rendimiento_m2_por_dia": rendimiento_dia,
        "apu_encontrado": (
            f"Suministro: {apu_suministro_desc} | "
            f"Tarea: {apu_tarea_desc} | "
            f"Cuadrilla: {apu_cuadrilla_desc}"
        ),
        "log": "\n".join(log),
    }
