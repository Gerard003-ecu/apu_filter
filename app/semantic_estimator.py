"""
Microservicio: Semantic Estimator (El Asesor Táctico)
Estrato DIKW: TACTICS (Nivel 2)

Responsabilidad: Alojamiento del espacio vectorial continuo (FAISS) y 
resolución de ambigüedades semánticas. Actúa como el motor de inferencia
desacoplado del Plano de Control.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.financial_engine import (
    CapitalAssetPricing,
    FinancialConfig,
    RealOptionsAnalyzer,
    RiskQuantifier,
)
from app.utils import normalize_text
from models.probability_models import run_monte_carlo_simulation
from app.tools_interface import MICRegistry

logger = logging.getLogger("SemanticEstimator")

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

MAX_POOL_SIZE_FOR_ITERATION = 50000
MAX_CANDIDATES_TO_TRACK = 1000
MAX_FAISS_TOP_K = 500
MIN_DESCRIPTION_LENGTH = 2
MAX_KEYWORDS_COUNT = 50

MIN_DATA_QUALITY_THRESHOLD = 0.3
MIN_VALID_APUS_FOR_ESTIMATION = 10

REQUIRED_COLUMNS_APU_CRITICAL = ["CODIGO_APU", "DESC_NORMALIZED"]
REQUIRED_COLUMNS_APU_OPTIONAL = ["original_description", "tipo_apu", "UNIDAD"]

REQUIRED_COLUMNS_DETAIL_CRITICAL = ["CODIGO_APU", "TIPO_INSUMO"]
REQUIRED_COLUMNS_DETAIL_OPTIONAL = ["CANTIDAD_APU"]

DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_MIN_MATCH_PERCENTAGE = 30.0
DEFAULT_TOP_K = 5
DEFAULT_ZONA = "ZONA 0"
DEFAULT_IZAJE = "MANUAL"
DEFAULT_SEGURIDAD = "NORMAL"

# ============================================================================
# ENUMS
# ============================================================================

class MatchMode(str, Enum):
    WORDS = "words"
    SUBSTRING = "substring"

class TipoAPU(str, Enum):
    SUMINISTRO = "Suministro"
    SUMINISTRO_PREFABRICADO = "Suministro (Pre-fabricado)"
    INSTALACION = "Instalación"

class TipoInsumo(str, Enum):
    MANO_OBRA = "MANO DE OBRA"
    EQUIPO = "EQUIPO"
    MATERIAL = "MATERIAL"

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class DerivationDetails:
    match_method: str
    confidence_score: float
    source: str
    reasoning: str

@dataclass
class MatchCandidate:
    apu: pd.Series
    description: str
    matches: int
    percentage: float
    similarity: Optional[float] = None
    details: Optional[DerivationDetails] = None

@dataclass
class SearchArtifacts:
    model: SentenceTransformer
    faiss_index: Any
    id_map: Dict[str, str]

@dataclass
class DataQualityMetrics:
    total_records: int = 0
    valid_records: int = 0
    missing_descriptions: int = 0
    missing_codes: int = 0
    empty_numeric_fields: int = 0
    quality_score: float = 0.0

    def is_acceptable(self, threshold: float = MIN_DATA_QUALITY_THRESHOLD) -> bool:
        return self.quality_score >= threshold

# ============================================================================
# UTILS & LOGGING
# ============================================================================

def safe_float_conversion(value: Any, default: float = 0.0, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        if isinstance(value, str):
            value_clean = value.strip().replace(",", "").replace(" ", "")
            if not value_clean or value_clean in ("-", "N/A", "NA", "null", "None"):
                return default
            result = float(value_clean)
        else:
            result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        if min_value is not None and result < min_value:
            result = min_value
        if max_value is not None and result > max_value:
            result = max_value
        return result
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        float_val = safe_float_conversion(value, float(default))
        result = int(float_val)
        if min_value is not None and result < min_value: result = min_value
        if max_value is not None and result > max_value: result = max_value
        return result
    except Exception:
        return default

def get_safe_column_value(row: pd.Series, column: str, default: Any = "", expected_type: type = str) -> Any:
    value = row.get(column, default)
    if pd.isna(value): return default
    if not isinstance(value, expected_type):
        try:
            if expected_type == str: return str(value).strip()
            elif expected_type == float: return safe_float_conversion(value, default)
            elif expected_type == int: return safe_int_conversion(value, default)
        except Exception:
            return default
    return value if not (isinstance(value, str) and not value.strip()) else default

def _log_top_candidates(candidates: List[MatchCandidate], log: List[str], top_n: int = 3, keywords_count: int = 0) -> None:
    if not candidates:
        log.append("  📋 No se encontraron candidatos.")
        return

    display_count = min(top_n, len(candidates))
    log.append(f"  📋 Top {display_count} candidatos:")

    for i, cand in enumerate(candidates[:display_count], 1):
        desc_snippet = f"{cand.description[:60]}..." if len(cand.description) > 60 else cand.description
        if cand.similarity is not None:
            log.append(f"    {i}. Sim: {cand.similarity:.3f} | Código: {get_safe_column_value(cand.apu, 'CODIGO_APU', 'N/A')} | Desc: {desc_snippet}")
        else:
            log.append(f"    {i}. [{cand.matches}/{keywords_count}] ({cand.percentage:.0f}%) - {desc_snippet}")

# ============================================================================
# SEARCH LOGIC
# ============================================================================

def validate_search_artifacts(search_artifacts: Optional[SearchArtifacts], log: List[str], require_all: bool = True) -> Tuple[bool, str]:
    if search_artifacts is None:
        msg = "SearchArtifacts es None"
        log.append(f"  ❌ {msg}")
        return (False, msg)

    if search_artifacts.model is None:
        msg = "Modelo de embeddings no disponible"
        log.append(f"  ❌ {msg}")
        if require_all: return (False, msg)

    if search_artifacts.faiss_index is None:
        msg = "Índice FAISS no disponible"
        log.append(f"  ❌ {msg}")
        if require_all: return (False, msg)

    if search_artifacts.id_map is None:
        msg = "Mapa de IDs no disponible"
        log.append(f"  ❌ {msg}")
        if require_all: return (False, msg)

    return (True, "")

def assess_data_quality(df: pd.DataFrame, critical_columns: List[str], optional_columns: List[str], log: List[str]) -> DataQualityMetrics:
    metrics = DataQualityMetrics()
    if not isinstance(df, pd.DataFrame): return metrics
    metrics.total_records = len(df)
    if metrics.total_records == 0: return metrics

    missing_critical = set(critical_columns) - set(df.columns)
    if missing_critical:
        log.append(f"  ❌ Columnas críticas faltantes: {missing_critical}")
        return metrics

    valid_count = 0
    for _, row in df.iterrows():
        is_valid = True
        codigo = row.get("CODIGO_APU", "")
        if not codigo or (isinstance(codigo, str) and not codigo.strip()):
            metrics.missing_codes += 1
            is_valid = False
        desc = row.get("DESC_NORMALIZED", "")
        if not desc or (isinstance(desc, str) and len(desc.strip()) < MIN_DESCRIPTION_LENGTH):
            metrics.missing_descriptions += 1
            is_valid = False
        if is_valid: valid_count += 1

    metrics.valid_records = valid_count
    metrics.quality_score = valid_count / metrics.total_records if metrics.total_records > 0 else 0.0
    return metrics

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame", strict: bool = False) -> Tuple[bool, List[str], List[str]]:
    if not isinstance(df, pd.DataFrame): return (False, [], list(required_columns))
    present = [c for c in required_columns if c in df.columns]
    missing = [c for c in required_columns if c not in df.columns]
    return (len(missing) == 0 if strict else True, present, missing)

def _calculate_match_score(desc_words: set, keywords: List[str]) -> Tuple[int, float]:
    if not desc_words or not keywords: return 0, 0.0
    matches = sum(1 for k in keywords if k in desc_words)
    percentage = (matches / len(keywords) * 100.0) if keywords else 0.0
    return matches, percentage

def _find_best_keyword_match(df_pool: pd.DataFrame, keywords: List[str], log: List[str], strict: bool = False, min_match_percentage: float = DEFAULT_MIN_MATCH_PERCENTAGE, match_mode: str = MatchMode.WORDS) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
    if not isinstance(df_pool, pd.DataFrame) or df_pool.empty:
        log.append("  ⚠️ Pool vacío, no hay datos para buscar.")
        return None, None

    if not keywords:
        log.append("  ⚠️ Keywords vacías.")
        return None, None

    # Check match_mode
    try:
        mode = MatchMode(match_mode) if isinstance(match_mode, str) else match_mode
    except ValueError:
        log.append(f"  ⚠️ Modo '{match_mode}' no válido. Usando '{MatchMode.WORDS}'")
        mode = MatchMode.WORDS

    # Check DESC_NORMALIZED
    if "DESC_NORMALIZED" not in df_pool.columns:
        if "original_description" in df_pool.columns:
            df_pool = df_pool.copy()
            df_pool["DESC_NORMALIZED"] = df_pool["original_description"].apply(lambda x: normalize_text(str(x)) if pd.notna(x) else "")
        else:
            log.append("  ❌ ERROR: Columna 'DESC_NORMALIZED' no encontrada")
            return None, None

    keywords_clean = [k.strip().lower() for k in keywords if isinstance(k, str) and len(k.strip()) >= 2]
    # Limit keywords
    if len(keywords_clean) > MAX_KEYWORDS_COUNT:
        log.append(f"  ⚠️ Demasiadas keywords ({len(keywords_clean)}), usando primeras {MAX_KEYWORDS_COUNT}")
        keywords_clean = keywords_clean[:MAX_KEYWORDS_COUNT]

    if not keywords_clean:
        log.append("  ⚠️ Después de limpieza, no hay keywords válidas.")
        return None, None

    pool_size = len(df_pool)
    log.append(f"  🔍 Buscando: '{' '.join(keywords_clean)}'")
    log.append(f"  📊 Pool size: {pool_size} APUs")

    candidates = []

    for _, apu in df_pool.iterrows():
        desc = str(apu.get("DESC_NORMALIZED", "")).strip().lower()
        if len(desc) < MIN_DESCRIPTION_LENGTH: continue

        matches = 0
        percentage = 0.0
        method_name = "KEYWORD"

        if mode == MatchMode.WORDS:
            desc_words = set(desc.split())
            matches, percentage = _calculate_match_score(desc_words, keywords_clean)
        elif mode == MatchMode.SUBSTRING:
            kw_str = " ".join(keywords_clean)
            if kw_str in desc:
                matches = len(keywords_clean)
                percentage = 100.0
                method_name = "EXACT_SUBSTRING"
            else:
                matches = sum(1 for kw in keywords_clean if kw in desc)
                percentage = (matches / len(keywords_clean) * 100.0) if keywords_clean else 0.0
                method_name = "PARTIAL_SUBSTRING"

        if matches == 0: continue

        original_desc = get_safe_column_value(apu, "original_description", "Sin descripción")

        details = DerivationDetails(method_name, percentage/100.0, "Histórico", f"Match: {percentage:.0f}% ({matches} palabras)")

        candidate = MatchCandidate(apu, original_desc, matches, percentage, details=details)
        candidates.append(candidate)

        if strict and percentage == 100.0:
            log.append("  ⚡ Early exit: Match perfecto encontrado")
            _log_top_candidates([candidate], log, 1, len(keywords_clean))
            return apu, details

    candidates.sort(key=lambda x: (x.percentage, x.matches), reverse=True)
    _log_top_candidates(candidates, log, 3, len(keywords_clean))

    if not candidates:
        log.append("  ❌ Sin match válido (no se encontraron candidatos)")
        return None, None

    best_match = candidates[0]

    if strict:
        log.append(f"  ❌ No se encontró match estricto. Mejor coincidencia: {best_match.percentage:.0f}%")
        return None, None

    if best_match.percentage >= min_match_percentage:
        log.append(f"  ✅ Match FLEXIBLE encontrado ({best_match.percentage:.0f}% ≥ {min_match_percentage:.0f}%)")
        return best_match.apu, best_match.details
    else:
        log.append(f"  ❌ Sin match válido. Mejor: {best_match.percentage:.0f}% | Umbral: {min_match_percentage:.0f}%")
        return None, None

def _find_best_semantic_match(df_pool: pd.DataFrame, query_text: str, search_artifacts: SearchArtifacts, log: List[str], min_similarity: float = DEFAULT_MIN_SIMILARITY, top_k: int = DEFAULT_TOP_K) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
    valid, msg = validate_search_artifacts(search_artifacts, log)
    if not valid:
        log.append(f"  ❌ Búsqueda semántica deshabilitada: {msg}")
        return None, None

    if not isinstance(df_pool, pd.DataFrame):
        log.append(f"  ❌ df_pool no es DataFrame (tipo: {type(df_pool).__name__})")
        return None, None

    if df_pool.empty:
        log.append("  ⚠️ Pool de APUs vacío para búsqueda semántica.")
        return None, None

    if "CODIGO_APU" not in df_pool.columns:
        log.append("  ❌ ERROR: Columna 'CODIGO_APU' no encontrada en df_pool.")
        return None, None

    if not query_text or not isinstance(query_text, str):
        log.append("  ⚠️ Texto de consulta vacío o inválido.")
        return None, None

    query_clean = query_text.strip()
    if len(query_clean) < MIN_DESCRIPTION_LENGTH:
        log.append(f"  ⚠️ Texto de consulta muy corto: '{query_clean}'")
        return None, None

    log.append(f"  🧠 Búsqueda Semántica: '{query_clean[:50]}...'")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    log.append(f"  ⚙️ Umbral similitud: {min_similarity:.2f} | Top-K: {top_k}")

    try:
        embedding = search_artifacts.model.encode([query_clean], convert_to_numpy=True, normalize_embeddings=True)
        if embedding is None or embedding.size == 0: return None, None

        D, I = search_artifacts.faiss_index.search(embedding.astype(np.float32), k=min(top_k, search_artifacts.faiss_index.ntotal))

        pool_codes = set(df_pool["CODIGO_APU"].astype(str).str.strip())

        candidates = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            sim = float(D[0][i])

            if sim < 0 or sim > 1.0: sim = min(max(sim, 0.0), 1.0)

            code = search_artifacts.id_map.get(str(idx))
            if not code: continue
            code = str(code).strip()

            if code not in pool_codes: continue

            matches = df_pool[df_pool["CODIGO_APU"].astype(str).str.strip() == code]
            if matches.empty: continue

            apu = matches.iloc[0]
            original_desc = get_safe_column_value(apu, "original_description", "Sin descripción")

            details = DerivationDetails("SEMANTIC", sim, "Vector Database", f"Similitud semántica: {sim:.3f}")
            candidates.append(MatchCandidate(apu, original_desc, 0, 0.0, similarity=sim, details=details))

        if not candidates:
            log.append("  ⚠️ Ningún resultado de FAISS coincide con el pool filtrado")
            return None, None

        candidates.sort(key=lambda x: x.similarity if x.similarity is not None else 0.0, reverse=True)
        _log_top_candidates(candidates, log, 3)

        if candidates and candidates[0].similarity >= min_similarity:
            log.append(f"  ✅ Coincidencia semántica encontrada: {candidates[0].similarity:.3f}")
            return candidates[0].apu, candidates[0].details
        else:
            best_sim = candidates[0].similarity if candidates else 0.0
            log.append(f"  ❌ Sin coincidencia válida. Mejor similitud: {best_sim:.3f}")

    except Exception as e:
        logger.error(f"Error en búsqueda semántica: {e}")
        error_msg = str(e)
        if "Model error" in error_msg or "encode" in error_msg:
             log.append(f"  ❌ ERROR al generar embedding: {type(e).__name__}: {e}")
        elif "FAISS" in error_msg:
             log.append(f"  ❌ ERROR en búsqueda FAISS ({type(e).__name__}): {e}")
        else:
             log.append(f"  ❌ Error en búsqueda semántica: {e}")

    return None, None

def _calculate_historical_average(df_processed_apus: pd.DataFrame, material_keywords: List[str], log: List[str]) -> Tuple[Optional[pd.Series], Optional[DerivationDetails], float]:
    try:
        sig_keywords = [k for k in material_keywords if len(k) > 3]
        if not sig_keywords: return None, None, 0.0

        regex = "|".join(sig_keywords)
        if "DESC_NORMALIZED" not in df_processed_apus.columns: return None, None, 0.0

        mask = df_processed_apus["DESC_NORMALIZED"].str.contains(regex, case=False, regex=True, na=False)
        candidates = df_processed_apus[mask].copy()

        if "RENDIMIENTO_DIA" in candidates.columns:
            candidates = candidates[candidates["RENDIMIENTO_DIA"].apply(lambda x: safe_float_conversion(x) > 0)]

        if candidates.empty: return None, None, 0.0

        avg_rend = candidates["RENDIMIENTO_DIA"].apply(lambda x: safe_float_conversion(x)).mean()

        apu_synth = pd.Series({
            "CODIGO_APU": "EST-AVG",
            "original_description": f"Promedio ({len(candidates)} items)",
            "RENDIMIENTO_DIA": avg_rend,
            "EQUIPO": 0.0
        })

        details = DerivationDetails("HISTORICAL_AVERAGE", min(0.5, len(candidates)/10), "Promedio Histórico", f"Promedio de {len(candidates)} items")
        return apu_synth, details, avg_rend

    except Exception as e:
        logger.error(f"Error histórico: {e}")
        return None, None, 0.0

def _calculate_rendimiento_from_detail(apu_code: str, data_store: Dict[str, Any], log: List[str]) -> float:
    log.append("\n  📊 Calculando rendimiento desde detalle...")
    details = data_store.get("apus_detail", [])
    if not details:
        log.append("  ⚠️ No hay datos de detalle de APUs disponibles")
        return 0.0

    try:
        df = pd.DataFrame(details)
        if "CANTIDAD_APU" not in df.columns or "TIPO_INSUMO" not in df.columns:
            log.append("  ⚠️ Columnas faltantes en detalle")
            return 0.0

        mo = df[
            (df["CODIGO_APU"].astype(str).str.strip() == str(apu_code).strip()) &
            (df["TIPO_INSUMO"] == TipoInsumo.MANO_OBRA.value)
        ]

        if mo.empty:
            log.append("  ⚠️ No se encontró mano de obra para este APU")
            return 0.0

        total_time = sum(safe_float_conversion(c) for c in mo["CANTIDAD_APU"])
        if total_time > 0:
            rend = 1.0 / total_time
            log.append(f"  ⏱️ Rendimiento: {rend:.4f} un/día")
            return rend
        else:
            log.append("  ⚠️ Tiempo total de mano de obra es cero")
            return 0.0

    except Exception as e:
        log.append(f"  ❌ ERROR al calcular rendimiento: {e}")
        logger.error(f"Error rendimiento: {e}")
        return 0.0

def calculate_estimate(params: Dict[str, Any], data_store: Dict[str, Any], config: Dict[str, Any], search_artifacts: SearchArtifacts) -> Dict[str, Any]:
    log = ["🕵️ ESTIMADOR HÍBRIDO INICIADO (TACTICS STRATUM)"]
    derivation_details = {"suministro": None, "tarea": None, "cuadrilla": None}

    # 1. Validación Básica
    processed_apus = data_store.get("processed_apus", [])
    if not processed_apus: return {"error": "No hay datos de APU procesados (processed_apus vacío)", "log": "\n".join(log)}

    try:
        df_apus = pd.DataFrame(processed_apus) if isinstance(processed_apus, list) else processed_apus.copy()
    except Exception:
        return {"error": "Invalid APU data", "log": "\n".join(log)}

    # 2. Extracción Parámetros
    material = str(params.get("material", "")).strip().upper()
    if not material: return {"error": "El parámetro 'material' es obligatorio y no puede estar vacío", "log": "\n".join(log)}

    cuadrilla = str(params.get("cuadrilla", "0")).strip()
    zona = str(params.get("zona", DEFAULT_ZONA)).strip()
    izaje = str(params.get("izaje", DEFAULT_IZAJE)).strip()
    seguridad = str(params.get("seguridad", DEFAULT_SEGURIDAD)).strip()

    # Mapeo y Keywords
    param_map = config.get("param_map", {}) if isinstance(config.get("param_map"), dict) else {}
    material_mapped = param_map.get("material", {}).get(material, material)
    keywords = [k for k in normalize_text(material_mapped).split() if len(k) >= 2]

    # Umbrales
    thresholds = config.get("estimator_thresholds", {}) if isinstance(config.get("estimator_thresholds"), dict) else {}
    min_sim_sum = safe_float_conversion(thresholds.get("min_semantic_similarity_suministro", 0.3))
    min_sim_tarea = safe_float_conversion(thresholds.get("min_semantic_similarity_tarea", 0.4))
    min_kw_cuadrilla = safe_float_conversion(thresholds.get("min_keyword_match_percentage_cuadrilla", 50.0))

    # 3. Búsqueda Suministro
    log.append("🎯 Búsqueda Suministro")
    apu_sum, det_sum = None, None
    supply_types = [TipoAPU.SUMINISTRO.value, TipoAPU.SUMINISTRO_PREFABRICADO.value]

    df_sum_pool = df_apus[df_apus["tipo_apu"].isin(supply_types)].copy() if "tipo_apu" in df_apus.columns else df_apus.copy()

    if not df_sum_pool.empty:
        apu_sum, det_sum = _find_best_semantic_match(df_sum_pool, material_mapped, search_artifacts, log, min_similarity=min_sim_sum)
        if apu_sum is None:
            log.append("\n  🔄 Fallback: Búsqueda por palabras clave...")
            apu_sum, det_sum = _find_best_keyword_match(df_sum_pool, keywords, log)

    if det_sum: derivation_details["suministro"] = det_sum.__dict__
    val_sum = safe_float_conversion(apu_sum.get("VALOR_SUMINISTRO_UN", 0.0)) if apu_sum is not None else 0.0

    # 4. Búsqueda Cuadrilla
    log.append("\n" + "=" * 70)
    log.append("🎯 Búsqueda Cuadrilla")
    log.append("=" * 70)
    costo_dia_cuadrilla = 0.0
    apu_cuadrilla, det_cuadrilla = None, None

    if safe_int_conversion(cuadrilla) > 0:
        df_cuad_pool = df_apus[df_apus["UNIDAD"].astype(str).str.upper().str.strip() == "DIA"].copy() if "UNIDAD" in df_apus.columns else df_apus.copy()
        if not df_cuad_pool.empty:
            kw_cuad = normalize_text(f"cuadrilla {cuadrilla}").split()
            apu_cuadrilla, det_cuadrilla = _find_best_keyword_match(df_cuad_pool, kw_cuad, log, min_match_percentage=min_kw_cuadrilla)
            if apu_cuadrilla is not None:
                costo_dia_cuadrilla = safe_float_conversion(apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0))

    if det_cuadrilla: derivation_details["cuadrilla"] = det_cuadrilla.__dict__

    # 5. Búsqueda Tarea (Rendimiento)
    log.append("\n" + "=" * 70)
    log.append("🎯 Búsqueda Tarea")
    log.append("=" * 70)
    apu_tarea, det_tarea = None, None
    rendimiento = 0.0
    costo_equipo = 0.0

    df_tarea_pool = df_apus[df_apus["tipo_apu"] == TipoAPU.INSTALACION.value].copy() if "tipo_apu" in df_apus.columns else df_apus.copy()

    if not df_tarea_pool.empty:
        apu_tarea, det_tarea = _find_best_semantic_match(df_tarea_pool, material_mapped, search_artifacts, log, min_similarity=min_sim_tarea)
        if apu_tarea is None:
            log.append("\n  🔄 Fallback: Búsqueda por palabras clave...")
            apu_tarea, det_tarea = _find_best_keyword_match(df_tarea_pool, keywords, log)

    if apu_tarea is None:
        log.append("\n  ⚠️ Tarea específica no encontrada. Intentando promedio de históricos...")
        apu_tarea, det_tarea, rendimiento = _calculate_historical_average(df_apus, keywords, log)

    if det_tarea: derivation_details["tarea"] = det_tarea.__dict__

    if apu_tarea is not None:
        costo_equipo = safe_float_conversion(apu_tarea.get("EQUIPO", 0.0))
        if rendimiento <= 0 and str(apu_tarea.get("CODIGO_APU")) != "EST-AVG":
            rendimiento = _calculate_rendimiento_from_detail(str(apu_tarea.get("CODIGO_APU")), data_store, log)

    # 6. Cálculo Final
    rules = config.get("estimator_rules", {}) if isinstance(config.get("estimator_rules"), dict) else {}
    f_zona = safe_float_conversion(rules.get("factores_zona", {}).get(zona, 1.0), 1.0)
    c_izaje = safe_float_conversion(rules.get("costo_adicional_izaje", {}).get(izaje, 0.0))
    f_seguridad = safe_float_conversion(rules.get("factor_seguridad", {}).get(seguridad, 1.0), 1.0)

    costo_mo_base = (costo_dia_cuadrilla / rendimiento) if rendimiento > 0 else 0.0
    costo_mo_ajustado = costo_mo_base * f_seguridad

    val_instalacion = (costo_mo_ajustado + costo_equipo) * f_zona + c_izaje
    val_construccion = val_sum + val_instalacion

    # 7. Análisis Financiero (Opcional)
    financial_analysis = {}
    try:
        # Intento simple de Monte Carlo si hay APUs encontrados
        codes = [c for c in [
            str(apu_sum.get("CODIGO_APU")) if apu_sum is not None else None,
            str(apu_tarea.get("CODIGO_APU")) if apu_tarea is not None else None
        ] if c and c not in ["N/A", "EST-AVG"]]

        if codes:
            all_insumos = data_store.get("apus_detail", [])
            insumos = [i for i in all_insumos if str(i.get("CODIGO_APU")) in codes]
            if insumos:
                mc = run_monte_carlo_simulation(insumos)
                stats = mc.get("statistics", {})
                financial_analysis["monte_carlo_mean"] = stats.get("mean", val_construccion)
                financial_analysis["monte_carlo_std"] = stats.get("std_dev", 0)
    except Exception as e:
        logger.error(f"Error financiero: {e}")

    return {
        "valor_suministro": round(val_sum, 2),
        "valor_instalacion": round(val_instalacion, 2),
        "valor_construccion": round(val_construccion, 2),
        "rendimiento_m2_por_dia": round(rendimiento, 4),
        "costo_equipo": round(costo_equipo, 2),
        "costo_mano_obra": round(costo_mo_ajustado, 2),
        "apu_suministro_codigo": str(apu_sum.get("CODIGO_APU")) if apu_sum is not None else "N/A",
        "apu_tarea_codigo": str(apu_tarea.get("CODIGO_APU")) if apu_tarea is not None else "N/A",
        "apu_cuadrilla_codigo": str(apu_cuadrilla.get("CODIGO_APU")) if apu_cuadrilla is not None else "N/A",
        "factores_aplicados": {"zona": f_zona, "seguridad": f_seguridad, "izaje": c_izaje},
        "financial_analysis": financial_analysis,
        "derivation_details": derivation_details,
        "log": "\n".join(log)
    }

class SemanticEstimatorService:
    """
    Agente autónomo que gobierna la búsqueda vectorial y estimación de costos.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.artifacts: Optional[SearchArtifacts] = None
        self.is_ready = False
        # Carga perezosa en hilo separado para no bloquear el arranque (timeout prevention)
        import threading
        self._loading_thread = threading.Thread(target=self._load_tensor_space, daemon=True)
        self._loading_thread.start()

    def _load_tensor_space(self) -> None:
        """
        [Método Físico] Carga los modelos masivos en memoria aislada.
        Libera al servidor web (Gunicorn/Flask) de esta carga térmica.
        """
        logger.info("Iniciando ignición del Espacio Vectorial (FAISS) en hilo background...")
        try:
            # Configuración extraída del config_rules.json o defaults
            emb_meta = self.config.get("embedding_metadata", {})
            model_name = emb_meta.get("model_name", "all-MiniLM-L6-v2") if isinstance(emb_meta, dict) else "all-MiniLM-L6-v2"

            from pathlib import Path
            import json

            embeddings_dir = Path(__file__).parent / "embeddings"
            index_path = embeddings_dir / "faiss.index"
            map_path = embeddings_dir / "id_map.json"

            if not index_path.exists() or not map_path.exists():
                logger.warning("Artefactos FAISS no encontrados. Búsqueda semántica deshabilitada.")
                return

            model = SentenceTransformer(model_name)
            index = faiss.read_index(str(index_path))
            
            with open(map_path, 'r', encoding='utf-8') as f:
                id_map = json.load(f)

            self.artifacts = SearchArtifacts(model=model, faiss_index=index, id_map=id_map)
            self.is_ready = True
            logger.info(f"✅ Espacio Vectorial cargado. Dimensión: {index.d}, Vectores: {index.ntotal}")

        except Exception as e:
            logger.critical(f"❌ Fallo al materializar el espacio vectorial: {e}")
            self.is_ready = False

    def project_semantic_match(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Vector de la MIC] Recibe una descripción humana y la mapea a un insumo formal.
        """
        telemetry: Optional[TelemetryContext] = context.get("telemetry_context")
        if telemetry: telemetry.start_step("semantic_projection")

        if not self.is_ready or not self.artifacts:
            return {"success": False, "error": "Asesor Semántico no inicializado."}

        query_text = payload.get("query_text", "")
        df_pool_data = payload.get("df_pool")

        try:
            df_pool = pd.DataFrame(df_pool_data) if isinstance(df_pool_data, list) else df_pool_data
            if df_pool is None or df_pool.empty:
                return {"success": False, "error": "Pool vacío"}

            apu, details = _find_best_semantic_match(df_pool, query_text, self.artifacts, [])
            
            if telemetry: telemetry.end_step("semantic_projection", "success")
            
            if apu is not None:
                return {
                    "success": True,
                    "matched_id": str(apu.get("CODIGO_APU")),
                    "confidence": details.confidence_score if details else 0.0,
                    "details": details.__dict__ if details else None,
                    "stratum": Stratum.TACTICS.name
                }
            return {"success": False, "error": "No match found"}

        except Exception as e:
            if telemetry: telemetry.record_error("semantic_projection", str(e))
            return {"success": False, "error": str(e)}

    def calculate_dynamic_estimate(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Vector de la MIC] Ejecuta la estimación de costos delegada.
        """
        telemetry: Optional[TelemetryContext] = context.get("telemetry_context")
        
        try:
            params = payload.get("params", {})
            data_store = payload.get("data_store", {})
            
            # Invocamos la lógica interna
            result = calculate_estimate(
                params=params,
                data_store=data_store,
                config=self.config,
                search_artifacts=self.artifacts
            )
            
            return {"success": True, "estimate": result, "stratum": Stratum.TACTICS.name}

        except Exception as e:
            logger.error(f"Fallo en cálculo táctico: {e}")
            if telemetry: telemetry.record_error("tactical_estimate", str(e))
            return {"success": False, "error": str(e)}

    def register_in_mic(self, mic: MICRegistry) -> None:
        """
        Registra las capacidades de este agente en la Matriz de Interacción Central.
        """
        mic.register_vector(
            service_name="semantic_match",
            stratum=Stratum.TACTICS,
            handler=self.project_semantic_match
        )
        mic.register_vector(
            service_name="tactical_estimate",
            stratum=Stratum.TACTICS,
            handler=self.calculate_dynamic_estimate
        )
        logger.info("✅ Vectores Tácticos Semánticos registrados en la MIC.")
