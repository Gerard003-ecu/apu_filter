import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import normalize_text

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES Y ENUMERACIONES
# ============================================================================


class MatchMode(str, Enum):
    """Modos de coincidencia soportados."""

    WORDS = "words"
    SUBSTRING = "substring"


class TipoAPU(str, Enum):
    """Tipos de APU disponibles."""

    SUMINISTRO = "Suministro"
    SUMINISTRO_PREFABRICADO = "Suministro (Pre-fabricado)"
    INSTALACION = "Instalación"


class TipoInsumo(str, Enum):
    """Tipos de insumo en APU."""

    MANO_OBRA = "MANO DE OBRA"
    EQUIPO = "EQUIPO"
    MATERIAL = "MATERIAL"


# Columnas requeridas en DataFrames
REQUIRED_COLUMNS_APU = [
    "CODIGO_APU",
    "DESC_NORMALIZED",
    "original_description",
    "tipo_apu",
    "UNIDAD",
]

REQUIRED_COLUMNS_DETAIL = ["CODIGO_APU", "TIPO_INSUMO", "CANTIDAD_APU"]

# Valores por defecto
DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_MIN_MATCH_PERCENTAGE = 30.0
DEFAULT_TOP_K = 5
DEFAULT_ZONA = "ZONA 0"
DEFAULT_IZAJE = "MANUAL"
DEFAULT_SEGURIDAD = "NORMAL"

# Rangos de validación
MIN_SIMILARITY_RANGE = (0.0, 1.0)
MIN_MATCH_PERCENTAGE_RANGE = (0.0, 100.0)
TOP_K_RANGE = (1, 100)


# ============================================================================
# CLASES DE DATOS
# ============================================================================


@dataclass
class DerivationDetails:
    """Detalles del razonamiento para la coincidencia encontrada (White Box)."""

    match_method: str  # "SEMANTIC", "KEYWORD", "EXACT"
    confidence_score: float  # 0.0 - 1.0 (o 0-100 para KEYWORD, se normalizará)
    source: str  # De dónde vino el dato
    reasoning: str  # Frase explicativa


@dataclass
class MatchCandidate:
    """Representa un candidato de coincidencia."""

    apu: pd.Series
    description: str
    matches: int
    percentage: float
    similarity: Optional[float] = None
    details: Optional[DerivationDetails] = None


@dataclass
class SearchArtifacts:
    """Artefactos necesarios para búsqueda semántica."""

    model: SentenceTransformer
    faiss_index: Any
    id_map: Dict[str, str]


# ============================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN Y CONVERSIÓN
# ============================================================================


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Convierte un valor a float de forma segura.

    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.

    Returns:
        float: Valor convertido o default.
    """
    if value is None or pd.isna(value):
        return default

    try:
        result = float(value)
        return result if not np.isnan(result) and not np.isinf(result) else default
    except (ValueError, TypeError):
        logger.warning(f"No se pudo convertir '{value}' a float, usando default: {default}")
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Convierte un valor a int de forma segura.

    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.

    Returns:
        int: Valor convertido o default.
    """
    if value is None or pd.isna(value):
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"No se pudo convertir '{value}' a int, usando default: {default}")
        return default


def validate_dataframe_columns(
    df: pd.DataFrame, required_columns: List[str], df_name: str = "DataFrame"
) -> bool:
    """
    Valida que un DataFrame contenga las columnas requeridas.

    Args:
        df: DataFrame a validar.
        required_columns: Lista de columnas requeridas.
        df_name: Nombre del DataFrame para mensajes de error.

    Returns:
        bool: True si todas las columnas están presentes.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{df_name} no es un DataFrame válido.")
        return False

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.warning(
            f"{df_name} no contiene las columnas requeridas: {missing_cols}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
        return False

    return True


def validate_numeric_range(
    value: float, valid_range: Tuple[float, float], param_name: str
) -> bool:
    """
    Valida que un valor numérico esté dentro de un rango válido.

    Args:
        value: Valor a validar.
        valid_range: Tupla (min, max) del rango válido.
        param_name: Nombre del parámetro para mensajes.

    Returns:
        bool: True si el valor está en el rango.
    """
    min_val, max_val = valid_range
    if not (min_val <= value <= max_val):
        logger.warning(
            f"{param_name}={value} está fuera del rango válido [{min_val}, {max_val}]"
        )
        return False
    return True


def get_safe_column_value(
    row: pd.Series, column: str, default: Any = "", expected_type: type = str
) -> Any:
    """
    Obtiene un valor de una columna de forma segura con tipo esperado.

    Args:
        row: Serie de pandas (fila).
        column: Nombre de la columna.
        default: Valor por defecto.
        expected_type: Tipo esperado del valor.

    Returns:
        Valor de la columna o default si no existe/es inválido.
    """
    value = row.get(column, default)

    if pd.isna(value):
        return default

    if not isinstance(value, expected_type):
        try:
            if expected_type == str:
                return str(value).strip()
            elif expected_type == float:
                return safe_float_conversion(value, default)
            elif expected_type == int:
                return safe_int_conversion(value, default)
        except Exception:
            return default

    return value if not (isinstance(value, str) and not value.strip()) else default


# ============================================================================
# FUNCIONES DE BÚSQUEDA Y MATCHING
# ============================================================================


def _calculate_match_score(desc_words: set, keywords: List[str]) -> Tuple[int, float]:
    """
    Calcula el puntaje de coincidencia entre una descripción normalizada y palabras clave.

    Args:
        desc_words: Conjunto de palabras de la descripción normalizada.
        keywords: Lista de palabras clave a buscar.

    Returns:
        Tuple[int, float]: Número de palabras coincidentes y porcentaje de cobertura.
    """
    # Validación de entradas
    if not isinstance(desc_words, set):
        logger.warning("desc_words no es un set, convirtiendo...")
        desc_words = set(desc_words) if desc_words else set()

    if not desc_words or not keywords:
        return 0, 0.0

    # Calcular coincidencias
    matches = sum(1 for keyword in keywords if keyword and keyword in desc_words)

    # Prevenir división por cero
    total_keywords = len(keywords)
    percentage = (matches / total_keywords * 100.0) if total_keywords > 0 else 0.0

    return matches, percentage


def _create_match_candidate(
    apu: pd.Series,
    matches: int,
    percentage: float,
    keywords_count: int,
    method: str = "KEYWORD",
) -> Optional[MatchCandidate]:
    """
    Crea un candidato de coincidencia validado.

    Args:
        apu: Serie con datos del APU.
        matches: Número de coincidencias.
        percentage: Porcentaje de cobertura.
        keywords_count: Total de keywords buscadas.
        method: Método de coincidencia.

    Returns:
        MatchCandidate o None si los datos no son válidos.
    """
    original_desc = get_safe_column_value(apu, "original_description", "Sin descripción")

    if not original_desc or original_desc == "Sin descripción":
        logger.debug(f"APU {apu.get('CODIGO_APU', 'UNKNOWN')} sin descripción válida")

    details = DerivationDetails(
        match_method=method,
        confidence_score=percentage / 100.0,
        source="Histórico Procesado",
        reasoning=f"Coincidencia de {matches}/{keywords_count} palabras clave ({percentage:.0f}%)",
    )

    return MatchCandidate(
        apu=apu,
        description=original_desc,
        matches=matches,
        percentage=percentage,
        details=details,
    )


def _log_top_candidates(
    candidates: List[MatchCandidate], log: List[str], top_n: int = 3, keywords_count: int = 0
) -> None:
    """
    Registra los mejores candidatos en el log.

    Args:
        candidates: Lista de candidatos.
        log: Lista de mensajes de log.
        top_n: Número de candidatos a mostrar.
        keywords_count: Total de keywords para contexto.
    """
    if not candidates:
        log.append("  📋 No se encontraron candidatos.")
        return

    display_count = min(top_n, len(candidates))
    log.append(f"  📋 Top {display_count} candidatos:")

    for i, cand in enumerate(candidates[:display_count], 1):
        desc_snippet = (
            f"{cand.description[:60]}..." if len(cand.description) > 60 else cand.description
        )

        if cand.similarity is not None:
            log.append(
                f"    {i}. Sim: {cand.similarity:.3f} | "
                f"Código: {get_safe_column_value(cand.apu, 'CODIGO_APU', 'N/A')} | "
                f"Desc: {desc_snippet}"
            )
        else:
            log.append(
                f"    {i}. [{cand.matches}/{keywords_count}] "
                f"({cand.percentage:.0f}%) - {desc_snippet}"
            )


def _find_best_keyword_match(
    df_pool: pd.DataFrame,
    keywords: List[str],
    log: List[str],
    strict: bool = False,
    min_match_percentage: float = DEFAULT_MIN_MATCH_PERCENTAGE,
    match_mode: str = MatchMode.WORDS,
) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
    """
    Encuentra la mejor coincidencia de APU para una lista de palabras clave.
    Soporta modos: 'words' (coincidencia exacta de palabras) y 'substring'
    (contiene toda la cadena).

    Args:
        df_pool: DataFrame con APUs procesados.
        keywords: Palabras clave a buscar.
        log: Lista de mensajes de log (mutable).
        strict: Si True, requiere 100% de coincidencia.
        min_match_percentage: Umbral mínimo de coincidencia para modo flexible.
        match_mode: 'words' o 'substring'.

    Returns:
        Tuple[Optional[pd.Series], Optional[DerivationDetails]]: El APU con mejor coincidencia y sus detalles.
    """
    # ========== VALIDACIÓN DE ENTRADAS ==========
    if not isinstance(df_pool, pd.DataFrame):
        log.append("  ❌ ERROR: df_pool no es un DataFrame.")
        return None, None

    if df_pool.empty:
        log.append("  ⚠️ Pool vacío, no hay datos para buscar.")
        return None, None

    if not keywords or not any(k.strip() for k in keywords if isinstance(k, str)):
        log.append("  ⚠️ Keywords vacías o inválidas.")
        return None, None

    # Validar que match_mode sea válido
    try:
        mode = MatchMode(match_mode)
    except ValueError:
        log.append(f"  ❌ ERROR: Modo '{match_mode}' no válido. Usando '{MatchMode.WORDS}'.")
        mode = MatchMode.WORDS

    # Validar rango de min_match_percentage
    if not validate_numeric_range(
        min_match_percentage, MIN_MATCH_PERCENTAGE_RANGE, "min_match_percentage"
    ):
        min_match_percentage = DEFAULT_MIN_MATCH_PERCENTAGE

    # ========== PREPARACIÓN DE KEYWORDS ==========
    keywords_clean = [
        k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()
    ]

    if not keywords_clean:
        log.append("  ⚠️ Después de limpieza, no hay keywords válidas.")
        return None, None

    # ========== LOG INICIAL ==========
    log.append(f"  🔍 Buscando: '{' '.join(keywords_clean)}'")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    modo_str = "ESTRICTO (100%)" if strict else f"FLEXIBLE (≥{min_match_percentage:.0f}%)"
    log.append(f"  ⚙️ Modo: {modo_str} | Estrategia: {mode.value}")

    # ========== PROCESAMIENTO DE CANDIDATOS ==========
    best_match = None
    best_percentage = -1.0
    best_details = None
    candidates: List[MatchCandidate] = []

    for idx, apu in df_pool.iterrows():
        # Obtener descripción normalizada de forma segura
        desc_normalized = get_safe_column_value(apu, "DESC_NORMALIZED", "")

        if not desc_normalized:
            continue

        desc_normalized = desc_normalized.strip().lower()
        matches = 0
        percentage = 0.0
        method_name = "KEYWORD"

        # Calcular matches según el modo
        if mode == MatchMode.WORDS:
            desc_words = set(desc_normalized.split())
            matches, percentage = _calculate_match_score(desc_words, keywords_clean)
            method_name = "KEYWORD"

        elif mode == MatchMode.SUBSTRING:
            keyword_str = " ".join(keywords_clean)
            if keyword_str in desc_normalized:
                matches = len(keywords_clean)
                percentage = 100.0
                method_name = "EXACT_SUBSTRING"
            else:
                # Búsqueda parcial: cuántas keywords están como substring
                matches = sum(1 for kw in keywords_clean if kw in desc_normalized)
                percentage = (
                    (matches / len(keywords_clean) * 100.0) if keywords_clean else 0.0
                )
                method_name = "PARTIAL_SUBSTRING"

        if matches == 0:
            continue

        # Crear y guardar candidato
        candidate = _create_match_candidate(
            apu, matches, percentage, len(keywords_clean), method=method_name
        )

        if candidate:
            candidates.append(candidate)

            # Actualizar mejor coincidencia
            if percentage > best_percentage:
                best_match = apu
                best_percentage = percentage
                best_details = candidate.details

    # ========== ORDENAR Y MOSTRAR CANDIDATOS ==========
    candidates.sort(key=lambda x: (x.percentage, x.matches), reverse=True)
    _log_top_candidates(candidates, log, top_n=3, keywords_count=len(keywords_clean))

    # ========== DECISIÓN FINAL ==========
    if strict:
        if best_percentage == 100.0:
            log.append("  ✅ Match ESTRICTO encontrado (100%)!")
            return best_match, best_details
        else:
            log.append(
                f"  ❌ No se encontró match estricto. "
                f"Mejor coincidencia: {best_percentage:.0f}%"
            )
            return None, None
    else:
        if best_percentage >= min_match_percentage:
            log.append(
                f"  ✅ Match FLEXIBLE encontrado ({best_percentage:.0f}% ≥ "
                f"{min_match_percentage:.0f}%)"
            )
            return best_match, best_details
        else:
            log.append(
                f"  ❌ Sin match válido. Mejor: {best_percentage:.0f}% | "
                f"Umbral: {min_match_percentage:.0f}%"
            )
            return None, None


def _find_best_semantic_match(
    df_pool: pd.DataFrame,
    query_text: str,
    search_artifacts: SearchArtifacts,
    log: List[str],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
    """
    Encuentra la mejor coincidencia semántica para un texto de consulta.

    Utiliza un índice FAISS y embeddings de sentence-transformers para
    encontrar los APUs más relevantes semánticamente.

    Args:
        df_pool: DataFrame con APUs procesados a considerar.
        query_text: Texto de consulta (ej. "muro de ladrillo").
        search_artifacts: Artefactos de búsqueda (modelo, índice, mapa).
        log: Lista de mensajes de log (mutable).
        min_similarity: Umbral mínimo de similitud de coseno [0.0-1.0].
        top_k: Número de vecinos a buscar en el índice FAISS.

    Returns:
        Tuple[Optional[pd.Series], Optional[DerivationDetails]]: El APU con mejor coincidencia y sus detalles.
    """
    # ========== VALIDACIÓN DE ARTEFACTOS ==========
    if (
        not search_artifacts
        or not search_artifacts.model
        or not search_artifacts.faiss_index
    ):
        log.append(
            "  ❌ ERROR: Artefactos de búsqueda semántica no disponibles. "
            "Búsqueda desactivada."
        )
        return None, None

    # ========== VALIDACIÓN DE ENTRADAS ==========
    if not isinstance(df_pool, pd.DataFrame) or df_pool.empty:
        log.append("  ⚠️ Pool de APUs vacío para búsqueda semántica.")
        return None, None

    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        log.append("  ⚠️ Texto de consulta vacío o inválido.")
        return None, None

    # Validar columna CODIGO_APU
    if "CODIGO_APU" not in df_pool.columns:
        log.append("  ❌ ERROR: Columna 'CODIGO_APU' no encontrada en df_pool.")
        return None, None

    # Validar rangos numéricos
    if not validate_numeric_range(min_similarity, MIN_SIMILARITY_RANGE, "min_similarity"):
        log.append(f"  ⚠️ min_similarity ajustado a {DEFAULT_MIN_SIMILARITY}")
        min_similarity = DEFAULT_MIN_SIMILARITY

    if not validate_numeric_range(top_k, TOP_K_RANGE, "top_k"):
        log.append(f"  ⚠️ top_k ajustado a {DEFAULT_TOP_K}")
        top_k = DEFAULT_TOP_K

    top_k = max(1, min(int(top_k), 100))  # Asegurar rango seguro

    # ========== LOG INICIAL ==========
    query_clean = query_text.strip()
    log.append(f"  🧠 Búsqueda Semántica: '{query_clean}'")
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    log.append(f"  ⚙️ Umbral similitud: {min_similarity:.2f} | Top-K: {top_k}")

    # ========== GENERAR EMBEDDING ==========
    try:
        query_embedding = search_artifacts.model.encode(
            [query_clean],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if query_embedding is None or query_embedding.size == 0:
            log.append("  ❌ ERROR: Embedding generado está vacío.")
            return None, None

    except Exception as e:
        log.append(f"  ❌ ERROR al generar embedding: {type(e).__name__}: {str(e)}")
        logger.exception("Error en generación de embedding")
        return None, None

    # ========== BÚSQUEDA EN FAISS ==========
    try:
        # Asegurar tipo correcto para FAISS
        query_vector = query_embedding.astype(np.float32)

        # Ajustar top_k si es mayor que el tamaño del índice
        index_size = search_artifacts.faiss_index.ntotal
        actual_k = min(top_k, index_size)

        if actual_k == 0:
            log.append("  ❌ ERROR: Índice FAISS vacío.")
            return None, None

        distances, indices = search_artifacts.faiss_index.search(query_vector, k=actual_k)

        if distances is None or indices is None:
            log.append("  ❌ ERROR: FAISS retornó resultados nulos.")
            return None, None

    except Exception as e:
        log.append(f"  ❌ ERROR en búsqueda FAISS: {type(e).__name__}: {str(e)}")
        logger.exception("Error durante búsqueda en FAISS")
        return None, None

    # ========== PROCESAMIENTO DE RESULTADOS ==========
    candidates: List[MatchCandidate] = []

    if indices.size == 0 or distances.size == 0:
        log.append("  ⚠️ FAISS no retornó resultados.")
        return None, None

    # Crear conjunto de códigos APU del pool para búsqueda rápida
    pool_apu_codes = set(df_pool["CODIGO_APU"].astype(str))

    for i in range(len(indices[0])):
        faiss_idx = int(indices[0][i])
        similarity = float(distances[0][i])

        # Validar que la similitud esté en rango razonable
        if not (0.0 <= similarity <= 1.0):
            log.append(
                f"  ⚠️ Similitud fuera de rango para índice {faiss_idx}: {similarity:.3f}"
            )
            continue

        # Obtener código APU desde el mapa
        apu_code = search_artifacts.id_map.get(str(faiss_idx))

        if apu_code is None:
            logger.debug(f"Índice FAISS {faiss_idx} no encontrado en id_map")
            continue

        # Verificar si el APU está en el pool actual (optimizado)
        if apu_code not in pool_apu_codes:
            continue

        # Buscar el APU en el DataFrame
        apu_matches = df_pool[df_pool["CODIGO_APU"] == apu_code]

        if apu_matches.empty:
            continue

        apu = apu_matches.iloc[0]
        original_desc = get_safe_column_value(apu, "original_description", "Sin descripción")

        details = DerivationDetails(
            match_method="SEMANTIC",
            confidence_score=similarity,
            source="Vector Database (FAISS)",
            reasoning=f"Coincidencia semántica alta con '{original_desc[:50]}...'",
        )

        candidate = MatchCandidate(
            apu=apu,
            description=original_desc,
            matches=0,  # No aplica en semántica
            percentage=0.0,  # No aplica en semántica
            similarity=similarity,
            details=details,
        )
        candidates.append(candidate)

    if not candidates:
        log.append("  ⚠️ Ningún resultado de FAISS coincide con el pool filtrado.")
        return None, None

    # ========== ORDENAR Y MOSTRAR CANDIDATOS ==========
    candidates.sort(key=lambda x: x.similarity if x.similarity else 0.0, reverse=True)
    _log_top_candidates(candidates, log, top_n=3)

    # ========== SELECCIÓN FINAL ==========
    best_candidate = candidates[0]

    if best_candidate.similarity and best_candidate.similarity >= min_similarity:
        log.append(
            f"  ✅ Coincidencia semántica encontrada: "
            f"{best_candidate.similarity:.3f} ≥ {min_similarity:.2f}"
        )
        return best_candidate.apu, best_candidate.details
    else:
        sim_value = best_candidate.similarity if best_candidate.similarity else 0.0
        log.append(
            f"  ❌ Sin coincidencia válida. Mejor similitud: "
            f"{sim_value:.3f} < {min_similarity:.2f}"
        )
        return None, None


# ============================================================================
# FUNCIÓN PRINCIPAL DE ESTIMACIÓN
# ============================================================================


def calculate_estimate(
    params: Dict[str, str],
    data_store: Dict[str, Any],
    config: Dict[str, Any],
    search_artifacts: SearchArtifacts,
) -> Dict[str, Union[str, float, List[str], Dict]]:
    """
    Estima el costo de construcción con una estrategia de búsqueda híbrida.

    Prioriza la búsqueda semántica y recurre a la búsqueda por palabras clave
    si la primera no produce resultados satisfactorios.

    Args:
        params: Diccionario con parámetros de entrada (material, cuadrilla, etc.)
        data_store: Diccionario con datos procesados (APUs, detalles, etc.)
        config: Diccionario con configuración (umbrales, mapeos, reglas)
        search_artifacts: Artefactos de búsqueda semántica inyectados.

    Returns:
        Dict con resultados de estimación: costos, APUs encontrados, log, y derivation_details.
    """
    log: List[str] = ["🕵️ ESTIMADOR HÍBRIDO INICIADO"]
    log.append("=" * 70)

    derivation_details = {"suministro": None, "tarea": None, "cuadrilla": None}

    # ========== VALIDACIÓN Y CARGA DE DATOS ==========
    log.append("\n📦 CARGA Y VALIDACIÓN DE DATOS")
    log.append("-" * 70)

    # Validar y cargar APUs procesados
    df_processed_apus = pd.DataFrame(data_store.get("processed_apus", []))

    if df_processed_apus.empty:
        error_msg = "No hay datos de APU procesados disponibles."
        log.append(f"  ❌ ERROR: {error_msg}")
        return {"error": error_msg, "log": "\n".join(log)}

    log.append(f"  ✓ APUs cargados: {len(df_processed_apus)} registros")

    # Validar columnas requeridas (solo advertencia, no bloquear)
    has_required_cols = validate_dataframe_columns(
        df_processed_apus, REQUIRED_COLUMNS_APU, "df_processed_apus"
    )
    if not has_required_cols:
        log.append("  ⚠️ Algunas columnas esperadas no están presentes")

    # ========== EXTRACCIÓN Y VALIDACIÓN DE PARÁMETROS ==========
    log.append("\n🎛️ EXTRACCIÓN DE PARÁMETROS")
    log.append("-" * 70)

    # Parámetros principales
    material = (params.get("material", "") or "").strip().upper()
    cuadrilla = (params.get("cuadrilla", "0") or "0").strip()
    zona = (params.get("zona", DEFAULT_ZONA) or DEFAULT_ZONA).strip().upper()
    izaje = (params.get("izaje", DEFAULT_IZAJE) or DEFAULT_IZAJE).strip().upper()
    seguridad = (
        (params.get("seguridad", DEFAULT_SEGURIDAD) or DEFAULT_SEGURIDAD).strip().upper()
    )

    log.append(f"  • Material: '{material}'")
    log.append(f"  • Cuadrilla: '{cuadrilla}'")
    log.append(f"  • Zona: '{zona}'")
    log.append(f"  • Izaje: '{izaje}'")
    log.append(f"  • Seguridad: '{seguridad}'")

    if not material:
        error_msg = "El parámetro 'material' es obligatorio y no puede estar vacío."
        log.append(f"  ❌ ERROR: {error_msg}")
        return {"error": error_msg, "log": "\n".join(log)}

    # Obtener mapeo de parámetros y configuración
    param_map = config.get("param_map", {})
    material_mapped = (
        param_map.get("material", {}).get(material, material) or material
    ).strip()

    log.append(f"  • Material mapeado: '{material_mapped}'")

    # Generar keywords normalizadas
    try:
        material_keywords = normalize_text(material_mapped).split()
    except Exception as e:
        log.append(f"  ⚠️ Error al normalizar material: {e}")
        material_keywords = material_mapped.lower().split()

    log.append(f"  • Keywords: {material_keywords}")

    # ========== EXTRACCIÓN DE UMBRALES ==========
    log.append("\n⚙️ CONFIGURACIÓN DE UMBRALES")
    log.append("-" * 70)

    thresholds = config.get("estimator_thresholds", {})

    min_sim_suministro = safe_float_conversion(
        thresholds.get("min_semantic_similarity_suministro", 0.30), 0.30
    )
    min_sim_tarea = safe_float_conversion(
        thresholds.get("min_semantic_similarity_tarea", 0.40), 0.40
    )
    min_kw_cuadrilla = safe_float_conversion(
        thresholds.get("min_keyword_match_percentage_cuadrilla", 50.0), 50.0
    )

    # Validar rangos
    min_sim_suministro = max(0.0, min(1.0, min_sim_suministro))
    min_sim_tarea = max(0.0, min(1.0, min_sim_tarea))
    min_kw_cuadrilla = max(0.0, min(100.0, min_kw_cuadrilla))

    log.append(f"  • Similitud mínima (Suministro): {min_sim_suministro:.2f}")
    log.append(f"  • Similitud mínima (Tarea): {min_sim_tarea:.2f}")
    log.append(f"  • Match mínimo (Cuadrilla): {min_kw_cuadrilla:.0f}%")

    # ========== BÚSQUEDA #1: SUMINISTRO ==========
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #1: SUMINISTRO")
    log.append("=" * 70)

    supply_types = [TipoAPU.SUMINISTRO.value, TipoAPU.SUMINISTRO_PREFABRICADO.value]

    if "tipo_apu" in df_processed_apus.columns:
        df_suministro_pool = df_processed_apus[
            df_processed_apus["tipo_apu"].isin(supply_types)
        ].copy()
        log.append(f"  📦 Pool de suministros: {len(df_suministro_pool)} APUs")
    else:
        log.append("  ⚠️ Columna 'tipo_apu' no encontrada, usando pool completo")
        df_suministro_pool = df_processed_apus.copy()

    # Intentar búsqueda semántica primero
    apu_suministro, details_suministro = _find_best_semantic_match(
        df_pool=df_suministro_pool,
        query_text=material_mapped,
        search_artifacts=search_artifacts,
        log=log,
        min_similarity=min_sim_suministro,
    )

    # Fallback a keywords si semántica falla
    if apu_suministro is None:
        log.append("\n  🔄 Fallback: Búsqueda por palabras clave...")
        apu_suministro, details_suministro = _find_best_keyword_match(
            df_suministro_pool,
            material_keywords,
            log,
            strict=False,
            min_match_percentage=DEFAULT_MIN_MATCH_PERCENTAGE,
        )

    if details_suministro:
        derivation_details["suministro"] = details_suministro.__dict__

    # Extraer valores
    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    apu_suministro_codigo = "N/A"

    if apu_suministro is not None:
        valor_suministro = safe_float_conversion(
            apu_suministro.get("VALOR_SUMINISTRO_UN", 0.0), 0.0
        )
        apu_suministro_desc = get_safe_column_value(
            apu_suministro, "original_description", "Sin descripción"
        )
        apu_suministro_codigo = get_safe_column_value(apu_suministro, "CODIGO_APU", "N/A")
        log.append(f"\n  ✅ APU encontrado: {apu_suministro_codigo}")
        log.append(f"  💰 Valor Suministro: ${valor_suministro:,.2f}")
    else:
        log.append("\n  ❌ No se encontró APU de suministro")

    # ========== BÚSQUEDA #2: CUADRILLA ==========
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #2: CUADRILLA")
    log.append("=" * 70)

    costo_diario_cuadrilla = 0.0
    apu_cuadrilla_desc = "No encontrada"
    apu_cuadrilla_codigo = "N/A"

    # Validar si se especificó cuadrilla
    cuadrilla_num = safe_int_conversion(cuadrilla, 0)

    if cuadrilla_num > 0:
        log.append(f"  👷 Buscando cuadrilla #{cuadrilla}")

        if "UNIDAD" in df_processed_apus.columns:
            df_cuadrilla_pool = df_processed_apus[
                df_processed_apus["UNIDAD"].astype(str).str.upper().str.strip() == "DIA"
            ].copy()
            log.append(f"  📦 Pool de cuadrillas: {len(df_cuadrilla_pool)} APUs")
        else:
            log.append("  ⚠️ Columna 'UNIDAD' no encontrada")
            df_cuadrilla_pool = df_processed_apus.copy()

        # Búsqueda por keywords (más precisa para cuadrillas)
        search_term = f"cuadrilla {cuadrilla}"
        try:
            cuadrilla_keywords = normalize_text(search_term).split()
        except Exception as e:
            log.append(f"  ⚠️ Error al normalizar búsqueda: {e}")
            cuadrilla_keywords = search_term.lower().split()

        apu_cuadrilla, details_cuadrilla = _find_best_keyword_match(
            df_cuadrilla_pool,
            cuadrilla_keywords,
            log,
            strict=False,
            min_match_percentage=min_kw_cuadrilla,
        )

        if details_cuadrilla:
            derivation_details["cuadrilla"] = details_cuadrilla.__dict__

        if apu_cuadrilla is not None:
            costo_diario_cuadrilla = safe_float_conversion(
                apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0), 0.0
            )
            apu_cuadrilla_desc = get_safe_column_value(
                apu_cuadrilla, "original_description", "Sin descripción"
            )
            apu_cuadrilla_codigo = get_safe_column_value(apu_cuadrilla, "CODIGO_APU", "N/A")
            log.append(f"\n  ✅ APU encontrado: {apu_cuadrilla_codigo}")
            log.append(f"  💰 Costo Cuadrilla: ${costo_diario_cuadrilla:,.2f}/día")
        else:
            log.append("\n  ❌ No se encontró APU de cuadrilla")
    else:
        log.append("  ⏭️ Cuadrilla no especificada, omitiendo búsqueda")

    # ========== BÚSQUEDA #3: TAREA (RENDIMIENTO) ==========
    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #3: TAREA (RENDIMIENTO)")
    log.append("=" * 70)

    if "tipo_apu" in df_processed_apus.columns:
        df_tarea_pool = df_processed_apus[
            df_processed_apus["tipo_apu"] == TipoAPU.INSTALACION.value
        ].copy()
        log.append(f"  📦 Pool de tareas: {len(df_tarea_pool)} APUs")
    else:
        log.append("  ⚠️ Columna 'tipo_apu' no encontrada, usando pool completo")
        df_tarea_pool = df_processed_apus.copy()

    # Intentar búsqueda semántica primero
    apu_tarea, details_tarea = _find_best_semantic_match(
        df_pool=df_tarea_pool,
        query_text=material_mapped,
        search_artifacts=search_artifacts,
        log=log,
        min_similarity=min_sim_tarea,
    )

    # Fallback a keywords si semántica falla
    if apu_tarea is None:
        log.append("\n  🔄 Fallback: Búsqueda por palabras clave...")
        apu_tarea, details_tarea = _find_best_keyword_match(
            df_tarea_pool,
            material_keywords,
            log,
            strict=False,
            min_match_percentage=DEFAULT_MIN_MATCH_PERCENTAGE,
        )

    # Fallback #2: Promedio de Históricos (Si Tarea específica no encontrada)
    if apu_tarea is None:
        log.append("\n  ⚠️ Tarea específica no encontrada. Intentando promedio de históricos...")

        # Estrategia: Buscar cualquier APU (no solo instalación) que coincida con las keywords
        # y que tenga rendimiento calculado.

        # 1. Buscar coincidencia amplia en todo el pool procesado
        potential_matches = []
        try:
            # Usar una búsqueda simple de contains para filtrar candidatos rápidos
            keywords_regex = "|".join([k for k in material_keywords if len(k) > 3])
            if keywords_regex:
                mask_keywords = df_processed_apus["DESC_NORMALIZED"].str.contains(keywords_regex, case=False, regex=True)
                df_candidates = df_processed_apus[mask_keywords].copy()
            else:
                df_candidates = pd.DataFrame() # No keywords útiles

            # 2. Filtrar los que tengan RENDIMIENTO_DIA > 0
            if "RENDIMIENTO_DIA" in df_candidates.columns:
                df_candidates = df_candidates[df_candidates["RENDIMIENTO_DIA"] > 0]

            if not df_candidates.empty:
                 avg_rendimiento = df_candidates["RENDIMIENTO_DIA"].mean()
                 count_matches = len(df_candidates)

                 log.append(f"  📊 Encontrados {count_matches} items similares con rendimiento.")
                 log.append(f"  ⏱️ Rendimiento promedio estimado: {avg_rendimiento:.4f} un/día")

                 # Crear un 'apu_tarea' sintético con el rendimiento promedio
                 apu_tarea = pd.Series({
                     "CODIGO_APU": "EST-AVG",
                     "original_description": f"Estimación Promedio ({count_matches} items similares)",
                     "RENDIMIENTO_DIA": avg_rendimiento,
                     "EQUIPO": 0.0 # Asumimos 0 si es promedio genérico
                 })

                 details_tarea = DerivationDetails(
                    match_method="HISTORICAL_AVERAGE",
                    confidence_score=0.5, # Confianza media/baja
                    source="Promedio Histórico",
                    reasoning=f"Promedio de {count_matches} items similares encontrados en histórico."
                 )

                 # Forzamos que se use este rendimiento más abajo
                 rendimiento_dia = avg_rendimiento
            else:
                 log.append("  ❌ No se encontraron items similares con rendimiento para promediar.")

        except Exception as e:
            log.append(f"  ⚠️ Error en cálculo de promedio histórico: {e}")

    if details_tarea:
        derivation_details["tarea"] = details_tarea.__dict__

    # Extraer valores y calcular rendimiento
    rendimiento_dia = 0.0
    costo_equipo = 0.0
    apu_tarea_desc = "No encontrado"
    apu_tarea_codigo = "N/A"

    if apu_tarea is not None:
        apu_tarea_desc = get_safe_column_value(
            apu_tarea, "original_description", "Sin descripción"
        )
        apu_tarea_codigo = get_safe_column_value(apu_tarea, "CODIGO_APU", "N/A")

        costo_equipo = safe_float_conversion(apu_tarea.get("EQUIPO", 0.0), 0.0)

        log.append(f"\n  ✅ APU encontrado: {apu_tarea_codigo}")
        log.append(f"  🛠️ Costo Equipo: ${costo_equipo:,.2f}")

        # Si ya calculamos rendimiento promedio en el fallback, no necesitamos recalcular desde detalle
        if apu_tarea_codigo == "EST-AVG" and rendimiento_dia > 0:
             log.append(f"  ⚡ Usando rendimiento promedio pre-calculado: {rendimiento_dia:.4f}")
        else:
             # Calcular rendimiento desde detalle de APUs
             log.append("\n  📊 Calculando rendimiento desde detalle...")
             apus_detail_list = data_store.get("apus_detail", [])

             if apus_detail_list:
                try:
                    df_detail = pd.DataFrame(apus_detail_list)

                    if validate_dataframe_columns(
                        df_detail, REQUIRED_COLUMNS_DETAIL, "df_detail"
                    ):
                        # Filtrar mano de obra para este APU
                        mano_obra = df_detail[
                            (df_detail["CODIGO_APU"] == apu_tarea_codigo)
                            & (df_detail["TIPO_INSUMO"] == TipoInsumo.MANO_OBRA.value)
                        ]

                        if not mano_obra.empty:
                            tiempo_total = sum(
                                safe_float_conversion(cant, 0.0)
                                for cant in mano_obra["CANTIDAD_APU"]
                            )

                            if tiempo_total > 0:
                                rendimiento_dia = 1.0 / tiempo_total
                                log.append(f"  ⏱️ Rendimiento: {rendimiento_dia:.4f} un/día")
                                log.append(
                                    f"  ⏱️ Tiempo total mano de obra: {tiempo_total:.4f} días/un"
                                )
                            else:
                                log.append("  ⚠️ Tiempo total de mano de obra es cero")
                        else:
                            log.append("  ⚠️ No se encontró mano de obra para este APU")
                    else:
                        log.append("  ⚠️ Detalle de APUs no tiene columnas requeridas")

                except Exception as e:
                    log.append(f"  ❌ ERROR al calcular rendimiento: {e}")
                    logger.exception("Error en cálculo de rendimiento")
             else:
                log.append("  ⚠️ No hay datos de detalle de APUs disponibles")
    else:
        log.append("\n  ❌ No se encontró APU de tarea")

    # ========== CÁLCULO FINAL ==========
    log.append("\n" + "=" * 70)
    log.append("🧮 CÁLCULO FINAL DE COSTOS")
    log.append("=" * 70)

    # Obtener reglas de negocio
    rules = config.get("estimator_rules", {})

    # Factores de ajuste
    factor_zona = safe_float_conversion(rules.get("factores_zona", {}).get(zona, 1.0), 1.0)
    costo_adicional_izaje = safe_float_conversion(
        rules.get("costo_adicional_izaje", {}).get(izaje, 0.0), 0.0
    )
    factor_seguridad = safe_float_conversion(
        rules.get("factor_seguridad", {}).get(seguridad, 1.0), 1.0
    )

    log.append(f"  📍 Factor Zona ({zona}): {factor_zona:.2f}")
    log.append(f"  🏗️ Costo Adicional Izaje ({izaje}): ${costo_adicional_izaje:,.2f}")
    log.append(f"  🦺 Factor Seguridad ({seguridad}): {factor_seguridad:.2f}")

    # Cálculo de costo de mano de obra
    if rendimiento_dia > 0 and costo_diario_cuadrilla > 0:
        costo_mo_base = costo_diario_cuadrilla / rendimiento_dia
        log.append(f"\n  👷 Costo MO Base: ${costo_mo_base:,.2f}")
    else:
        costo_mo_base = 0.0
        if rendimiento_dia <= 0:
            log.append("  ⚠️ Rendimiento no disponible, costo MO = 0")
        if costo_diario_cuadrilla <= 0:
            log.append("  ⚠️ Costo cuadrilla no disponible, costo MO = 0")

    # Aplicar factor de seguridad
    costo_mo_ajustado = costo_mo_base * factor_seguridad
    if factor_seguridad != 1.0:
        log.append(f"  👷 Costo MO Ajustado (seguridad): ${costo_mo_ajustado:,.2f}")

    # Calcular valor de instalación
    valor_instalacion = (
        costo_mo_ajustado + costo_equipo
    ) * factor_zona + costo_adicional_izaje

    log.append(f"\n  🔧 Valor Instalación: ${valor_instalacion:,.2f}")
    log.append(f"     ├─ MO Ajustada: ${costo_mo_ajustado:,.2f}")
    log.append(f"     ├─ Equipo: ${costo_equipo:,.2f}")
    log.append(f"     ├─ Factor Zona: x{factor_zona:.2f}")
    log.append(f"     └─ Izaje: +${costo_adicional_izaje:,.2f}")

    # Calcular valor total
    valor_construccion = valor_suministro + valor_instalacion

    # Validar que no haya valores negativos
    if valor_construccion < 0:
        log.append("\n  ⚠️ ADVERTENCIA: Valor de construcción negativo, ajustando a 0")
        valor_construccion = 0.0

    log.append("\n" + "=" * 70)
    log.append(f"💰 VALOR TOTAL CONSTRUCCIÓN: ${valor_construccion:,.2f}")
    log.append(f"   ├─ Suministro: ${valor_suministro:,.2f}")
    log.append(f"   └─ Instalación: ${valor_instalacion:,.2f}")
    log.append("=" * 70)

    # ========== PREPARAR RESPUESTA ==========
    apu_encontrado_str = (
        f"Suministro: {apu_suministro_desc} ({apu_suministro_codigo}) | "
        f"Tarea: {apu_tarea_desc} ({apu_tarea_codigo}) | "
        f"Cuadrilla: {apu_cuadrilla_desc} ({apu_cuadrilla_codigo})"
    )

    return {
        "valor_suministro": round(valor_suministro, 2),
        "valor_instalacion": round(valor_instalacion, 2),
        "valor_construccion": round(valor_construccion, 2),
        "rendimiento_m2_por_dia": round(rendimiento_dia, 4),
        "costo_equipo": round(costo_equipo, 2),
        "costo_mano_obra": round(costo_mo_ajustado, 2),
        "apu_suministro_codigo": apu_suministro_codigo,
        "apu_tarea_codigo": apu_tarea_codigo,
        "apu_cuadrilla_codigo": apu_cuadrilla_codigo,
        "apu_encontrado": apu_encontrado_str,
        "factores_aplicados": {
            "zona": factor_zona,
            "seguridad": factor_seguridad,
            "izaje": costo_adicional_izaje,
        },
        "derivation_details": derivation_details,
        "log": "\n".join(log),
    }
