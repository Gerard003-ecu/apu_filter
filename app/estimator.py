import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .financial_engine import (
    CapitalAssetPricing,
    FinancialConfig,
    RealOptionsAnalyzer,
    RiskQuantifier,
)
from .utils import normalize_text
from models.probability_models import run_monte_carlo_simulation

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


# ============================================================================
# CONSTANTES Y CONFIGURACIÓN ROBUSTECIDAS
# ============================================================================

# Límites de recursos para prevenir problemas de rendimiento
MAX_POOL_SIZE_FOR_ITERATION = 50000  # Máximo de APUs a iterar directamente
MAX_CANDIDATES_TO_TRACK = 1000  # Máximo de candidatos a mantener en memoria
MAX_FAISS_TOP_K = 500  # Límite superior para búsqueda FAISS
MIN_DESCRIPTION_LENGTH = 2  # Longitud mínima de descripción válida
MAX_KEYWORDS_COUNT = 50  # Máximo de keywords a procesar

# Umbrales de calidad de datos
MIN_DATA_QUALITY_THRESHOLD = 0.3  # Mínimo 30% de datos válidos para proceder
MIN_VALID_APUS_FOR_ESTIMATION = 10  # Mínimo de APUs válidos requeridos

# Columnas requeridas con prioridad
REQUIRED_COLUMNS_APU_CRITICAL = ["CODIGO_APU", "DESC_NORMALIZED"]
REQUIRED_COLUMNS_APU_OPTIONAL = ["original_description", "tipo_apu", "UNIDAD"]

REQUIRED_COLUMNS_DETAIL_CRITICAL = ["CODIGO_APU", "TIPO_INSUMO"]
REQUIRED_COLUMNS_DETAIL_OPTIONAL = ["CANTIDAD_APU"]

# Valores por defecto (mantenidos por compatibilidad)
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


@dataclass
class DataQualityMetrics:
    """Métricas de calidad de datos para diagnóstico."""

    total_records: int = 0
    valid_records: int = 0
    missing_descriptions: int = 0
    missing_codes: int = 0
    empty_numeric_fields: int = 0
    quality_score: float = 0.0

    def is_acceptable(self, threshold: float = MIN_DATA_QUALITY_THRESHOLD) -> bool:
        """Verifica si la calidad de datos es aceptable."""
        return self.quality_score >= threshold


@dataclass
class SearchResult:
    """Resultado estructurado de una búsqueda."""

    success: bool
    apu: Optional[pd.Series] = None
    details: Optional[DerivationDetails] = None
    candidates_evaluated: int = 0
    search_method: str = ""
    fallback_used: bool = False
    error_message: str = ""


# ============================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN Y CONVERSIÓN
# ============================================================================


def validate_search_artifacts(
    search_artifacts: Optional[SearchArtifacts], log: List[str], require_all: bool = True
) -> Tuple[bool, str]:
    """
    Valida exhaustivamente los artefactos de búsqueda semántica.

    ROBUSTECIDO:
    - Verificación de cada componente individualmente
    - Validación de dimensionalidad del modelo
    - Verificación del estado del índice FAISS
    - Validación del mapa de IDs

    Args:
        search_artifacts: Artefactos a validar.
        log: Lista de mensajes de log.
        require_all: Si True, todos los componentes son requeridos.

    Returns:
        Tuple[bool, str]: (es_válido, mensaje_de_error)
    """
    if search_artifacts is None:
        msg = "SearchArtifacts es None"
        log.append(f"  ❌ {msg}")
        return (False, msg)

    # Validar modelo
    if search_artifacts.model is None:
        msg = "Modelo de embeddings no disponible"
        log.append(f"  ❌ {msg}")
        if require_all:
            return (False, msg)
    else:
        try:
            # Verificar que el modelo puede generar embeddings
            model_dim = search_artifacts.model.get_sentence_embedding_dimension()
            if model_dim <= 0:
                msg = f"Dimensión del modelo inválida: {model_dim}"
                log.append(f"  ❌ {msg}")
                return (False, msg)
            log.append(f"  ✓ Modelo válido (dim={model_dim})")
        except Exception as e:
            msg = f"Error verificando modelo: {e}"
            log.append(f"  ⚠️ {msg}")
            if require_all:
                return (False, msg)

    # Validar índice FAISS
    if search_artifacts.faiss_index is None:
        msg = "Índice FAISS no disponible"
        log.append(f"  ❌ {msg}")
        if require_all:
            return (False, msg)
    else:
        try:
            index_size = search_artifacts.faiss_index.ntotal
            if index_size <= 0:
                msg = "Índice FAISS vacío"
                log.append(f"  ⚠️ {msg}")
                if require_all:
                    return (False, msg)
            else:
                log.append(f"  ✓ Índice FAISS válido ({index_size} vectores)")
        except Exception as e:
            msg = f"Error verificando índice FAISS: {e}"
            log.append(f"  ❌ {msg}")
            return (False, msg)

    # Validar mapa de IDs
    if search_artifacts.id_map is None:
        msg = "Mapa de IDs no disponible"
        log.append(f"  ❌ {msg}")
        if require_all:
            return (False, msg)
    elif not isinstance(search_artifacts.id_map, dict):
        msg = f"Mapa de IDs no es dict: {type(search_artifacts.id_map).__name__}"
        log.append(f"  ❌ {msg}")
        return (False, msg)
    elif len(search_artifacts.id_map) == 0:
        msg = "Mapa de IDs vacío"
        log.append(f"  ⚠️ {msg}")
        if require_all:
            return (False, msg)
    else:
        log.append(f"  ✓ Mapa de IDs válido ({len(search_artifacts.id_map)} entradas)")

    return (True, "")


def assess_data_quality(
    df: pd.DataFrame,
    critical_columns: List[str],
    optional_columns: List[str],
    log: List[str],
) -> DataQualityMetrics:
    """
    Evalúa la calidad de los datos de un DataFrame.

    ROBUSTECIDO:
    - Análisis detallado de completitud de datos
    - Métricas de calidad cuantificables
    - Identificación de problemas específicos

    Args:
        df: DataFrame a evaluar.
        critical_columns: Columnas críticas que deben existir.
        optional_columns: Columnas opcionales.
        log: Lista de mensajes de log.

    Returns:
        DataQualityMetrics con resultados del análisis.
    """
    metrics = DataQualityMetrics()

    if not isinstance(df, pd.DataFrame):
        log.append("  ❌ El objeto no es un DataFrame válido")
        return metrics

    metrics.total_records = len(df)

    if metrics.total_records == 0:
        log.append("  ⚠️ DataFrame vacío")
        return metrics

    # Verificar columnas críticas
    missing_critical = set(critical_columns) - set(df.columns)
    if missing_critical:
        log.append(f"  ❌ Columnas críticas faltantes: {missing_critical}")
        return metrics

    # Verificar columnas opcionales
    missing_optional = set(optional_columns) - set(df.columns)
    if missing_optional:
        log.append(f"  ⚠️ Columnas opcionales faltantes: {missing_optional}")

    # Analizar calidad de datos
    valid_count = 0

    for idx, row in df.iterrows():
        is_valid = True

        # Verificar código APU
        codigo = row.get("CODIGO_APU", "")
        if not codigo or (isinstance(codigo, str) and not codigo.strip()):
            metrics.missing_codes += 1
            is_valid = False

        # Verificar descripción
        desc = row.get("DESC_NORMALIZED", "")
        if not desc or (
            isinstance(desc, str) and len(desc.strip()) < MIN_DESCRIPTION_LENGTH
        ):
            metrics.missing_descriptions += 1
            is_valid = False

        if is_valid:
            valid_count += 1

    metrics.valid_records = valid_count
    metrics.quality_score = (
        valid_count / metrics.total_records if metrics.total_records > 0 else 0.0
    )

    # Log de resultados
    log.append("  📊 Calidad de datos:")
    log.append(f"     ├─ Total registros: {metrics.total_records}")
    log.append(
        f"     ├─ Registros válidos: {metrics.valid_records} ({metrics.quality_score * 100:.1f}%)"
    )
    log.append(f"     ├─ Códigos faltantes: {metrics.missing_codes}")
    log.append(f"     └─ Descripciones faltantes: {metrics.missing_descriptions}")

    return metrics


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str = "DataFrame",
    strict: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Valida que un DataFrame contenga las columnas requeridas.

    ROBUSTECIDO:
    - Retorna información detallada de columnas presentes/faltantes
    - Modo estricto vs permisivo
    - Validación de tipo del DataFrame

    Args:
        df: DataFrame a validar.
        required_columns: Lista de columnas requeridas.
        df_name: Nombre del DataFrame para mensajes.
        strict: Si True, falla si faltan columnas.

    Returns:
        Tuple[bool, List[str], List[str]]: (válido, columnas_presentes, columnas_faltantes)
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{df_name} no es un DataFrame válido (tipo: {type(df).__name__})")
        return (False, [], list(required_columns))

    if df.empty:
        logger.warning(f"{df_name} está vacío")
        # Aún puede tener las columnas aunque esté vacío

    present_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        log_func = logger.error if strict else logger.warning
        log_func(
            f"{df_name}: columnas faltantes {missing_columns}. "
            f"Disponibles: {list(df.columns)[:10]}..."
        )

    is_valid = len(missing_columns) == 0 if strict else True

    return (is_valid, present_columns, missing_columns)


def safe_float_conversion(
    value: Any,
    default: float = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    Convierte un valor a float de forma segura.

    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.
        min_value: Valor mínimo permitido (opcional).
        max_value: Valor máximo permitido (opcional).

    Returns:
        float: Valor convertido, limitado al rango si se especifica.
    """
    if value is None:
        return default

    # Manejar tipos especiales de pandas/numpy
    if pd.isna(value):
        return default

    try:
        # Manejar strings con formatos numéricos especiales
        if isinstance(value, str):
            value_clean = value.strip().replace(",", "").replace(" ", "")
            if not value_clean or value_clean in ("-", "N/A", "NA", "null", "None"):
                return default
            result = float(value_clean)
        elif isinstance(value, (np.integer, np.floating)):
            result = float(value)
        else:
            result = float(value)

        # Validar que no sea NaN o Inf
        if np.isnan(result) or np.isinf(result):
            logger.debug(f"Valor numérico inválido (nan/inf): {value}")
            return default

        # Aplicar límites de rango si se especifican
        if min_value is not None and result < min_value:
            logger.debug(f"Valor {result} menor que mínimo {min_value}, ajustando")
            result = min_value
        if max_value is not None and result > max_value:
            logger.debug(f"Valor {result} mayor que máximo {max_value}, ajustando")
            result = max_value

        return result

    except (ValueError, TypeError) as e:
        logger.debug(f"No se pudo convertir '{value}' ({type(value).__name__}) a float: {e}")
        return default


def safe_int_conversion(
    value: Any,
    default: int = 0,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """
    Convierte un valor a int de forma segura con validación de rango.

    ROBUSTECIDO:
    - Validación de rango opcional
    - Manejo de tipos especiales
    - Conversión desde float con truncamiento explícito

    Args:
        value: El valor a convertir.
        default: Valor por defecto en caso de fallo.
        min_value: Valor mínimo aceptable.
        max_value: Valor máximo aceptable.

    Returns:
        El entero convertido o el valor por defecto.
    """
    if value is None or pd.isna(value):
        return default

    try:
        # Primero convertir a float para manejar strings como "3.0"
        float_val = safe_float_conversion(value, float(default))
        result = int(float_val)

        # Aplicar límites de rango
        if min_value is not None and result < min_value:
            result = min_value
        if max_value is not None and result > max_value:
            result = max_value

        return result

    except (ValueError, TypeError, OverflowError) as e:
        logger.debug(f"No se pudo convertir '{value}' a int: {e}")
        return default


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
    max_iterations: int = MAX_POOL_SIZE_FOR_ITERATION,
) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
    """
    Encuentra la mejor coincidencia de APU para una lista de palabras clave.

    ROBUSTECIDO:
    - Límite de iteraciones para pools grandes
    - Early exit cuando se encuentra match perfecto
    - Validación exhaustiva de entradas
    - Manejo de memoria para candidatos
    - Logging detallado para diagnóstico
    - Coherente con validaciones de apu_processor

    Args:
        df_pool: DataFrame con APUs procesados.
        keywords: Palabras clave a buscar.
        log: Lista de mensajes de log (mutable).
        strict: Si True, requiere 100% de coincidencia.
        min_match_percentage: Umbral mínimo de coincidencia.
        match_mode: 'words' o 'substring'.
        max_iterations: Límite máximo de filas a procesar.

    Returns:
        Tuple[Optional[pd.Series], Optional[DerivationDetails]]
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDACIÓN DE ENTRADAS
    # ═══════════════════════════════════════════════════════════════════════════

    # Validar DataFrame
    if not isinstance(df_pool, pd.DataFrame):
        log.append(f"  ❌ ERROR: df_pool no es DataFrame (tipo: {type(df_pool).__name__})")
        return None, None

    if df_pool.empty:
        log.append("  ⚠️ Pool vacío, no hay datos para buscar.")
        return None, None

    # Verificar columna DESC_NORMALIZED
    if "DESC_NORMALIZED" not in df_pool.columns:
        log.append("  ❌ ERROR: Columna 'DESC_NORMALIZED' no encontrada")
        # Intentar usar original_description como fallback
        if "original_description" in df_pool.columns:
            log.append("  🔄 Usando 'original_description' como fallback")
            df_pool = df_pool.copy()
            df_pool["DESC_NORMALIZED"] = df_pool["original_description"].apply(
                lambda x: normalize_text(str(x)) if pd.notna(x) else ""
            )
        else:
            return None, None

    # Validar keywords
    if not keywords:
        log.append("  ⚠️ Keywords vacías.")
        return None, None

    if not isinstance(keywords, (list, tuple)):
        log.append(
            f"  ⚠️ Keywords no es lista (tipo: {type(keywords).__name__}), convirtiendo"
        )
        keywords = [str(keywords)]

    # Validar match_mode
    try:
        mode = MatchMode(match_mode) if isinstance(match_mode, str) else match_mode
    except ValueError:
        log.append(f"  ⚠️ Modo '{match_mode}' no válido. Usando '{MatchMode.WORDS}'")
        mode = MatchMode.WORDS

    # Validar y ajustar min_match_percentage
    min_match_percentage = safe_float_conversion(
        min_match_percentage, DEFAULT_MIN_MATCH_PERCENTAGE, min_value=0.0, max_value=100.0
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PREPARACIÓN DE KEYWORDS
    # ═══════════════════════════════════════════════════════════════════════════

    keywords_clean = []
    for k in keywords:
        if isinstance(k, str) and k.strip():
            cleaned = k.strip().lower()
            if len(cleaned) >= 2:  # Ignorar keywords muy cortas
                keywords_clean.append(cleaned)

    # Limitar número de keywords
    if len(keywords_clean) > MAX_KEYWORDS_COUNT:
        log.append(
            f"  ⚠️ Demasiadas keywords ({len(keywords_clean)}), usando primeras {MAX_KEYWORDS_COUNT}"
        )
        keywords_clean = keywords_clean[:MAX_KEYWORDS_COUNT]

    if not keywords_clean:
        log.append("  ⚠️ Después de limpieza, no hay keywords válidas.")
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # LOG INICIAL
    # ═══════════════════════════════════════════════════════════════════════════

    pool_size = len(df_pool)
    log.append(f"  🔍 Buscando: '{' '.join(keywords_clean)}'")
    log.append(f"  📊 Pool size: {pool_size} APUs")

    # Advertir si el pool es muy grande
    if pool_size > max_iterations:
        log.append(f"  ⚠️ Pool grande, procesando solo primeros {max_iterations} registros")

    modo_str = "ESTRICTO (100%)" if strict else f"FLEXIBLE (≥{min_match_percentage:.0f}%)"
    log.append(f"  ⚙️ Modo: {modo_str} | Estrategia: {mode.value}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PROCESAMIENTO DE CANDIDATOS
    # ═══════════════════════════════════════════════════════════════════════════

    best_match: Optional[pd.Series] = None
    best_percentage: float = -1.0
    best_details: Optional[DerivationDetails] = None
    candidates: List[MatchCandidate] = []

    processed_count = 0
    skipped_invalid = 0

    # Iterar sobre el pool con límite
    for idx, apu in df_pool.iterrows():
        if processed_count >= max_iterations:
            log.append(f"  ⚠️ Límite de iteraciones alcanzado ({max_iterations})")
            break

        processed_count += 1

        # Obtener descripción normalizada de forma segura
        desc_normalized = apu.get("DESC_NORMALIZED", "")

        if not desc_normalized or not isinstance(desc_normalized, str):
            skipped_invalid += 1
            continue

        desc_normalized = desc_normalized.strip().lower()

        if len(desc_normalized) < MIN_DESCRIPTION_LENGTH:
            skipped_invalid += 1
            continue

        # Calcular matches según el modo
        matches = 0
        percentage = 0.0
        method_name = "KEYWORD"

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
                matches = sum(1 for kw in keywords_clean if kw in desc_normalized)
                percentage = (
                    (matches / len(keywords_clean) * 100.0) if keywords_clean else 0.0
                )
                method_name = "PARTIAL_SUBSTRING"

        if matches == 0:
            continue

        # EARLY EXIT: Si encontramos match perfecto en modo estricto
        if strict and percentage == 100.0:
            candidate = _create_match_candidate(
                apu, matches, percentage, len(keywords_clean), method=method_name
            )
            if candidate:
                log.append("  ⚡ Early exit: Match perfecto encontrado")
                return candidate.apu, candidate.details

        # Crear candidato
        candidate = _create_match_candidate(
            apu, matches, percentage, len(keywords_clean), method=method_name
        )

        if candidate:
            # Limitar número de candidatos en memoria
            if len(candidates) < MAX_CANDIDATES_TO_TRACK:
                candidates.append(candidate)
            elif percentage > candidates[-1].percentage:
                # Reemplazar el peor candidato
                candidates[-1] = candidate
                candidates.sort(key=lambda x: (x.percentage, x.matches), reverse=True)

            # Actualizar mejor coincidencia
            if percentage > best_percentage:
                best_match = apu
                best_percentage = percentage
                best_details = candidate.details

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTADOS
    # ═══════════════════════════════════════════════════════════════════════════

    # Log de estadísticas de procesamiento
    if skipped_invalid > 0:
        log.append(f"  ⚠️ Registros omitidos por datos inválidos: {skipped_invalid}")

    # Ordenar y mostrar candidatos
    candidates.sort(key=lambda x: (x.percentage, x.matches), reverse=True)
    _log_top_candidates(candidates, log, top_n=3, keywords_count=len(keywords_clean))

    # ═══════════════════════════════════════════════════════════════════════════
    # DECISIÓN FINAL
    # ═══════════════════════════════════════════════════════════════════════════

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

    ROBUSTECIDO:
    - Validación exhaustiva de artefactos de búsqueda
    - Verificación de dimensionalidad de embeddings
    - Manejo específico de errores FAISS
    - Límites de recursos
    - Validación de coherencia entre índice y mapa de IDs
    - Fallback graceful cuando hay problemas

    Args:
        df_pool: DataFrame con APUs procesados a considerar.
        query_text: Texto de consulta.
        search_artifacts: Artefactos de búsqueda (modelo, índice, mapa).
        log: Lista de mensajes de log (mutable).
        min_similarity: Umbral mínimo de similitud [0.0-1.0].
        top_k: Número de vecinos a buscar.

    Returns:
        Tuple[Optional[pd.Series], Optional[DerivationDetails]]
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDACIÓN DE ARTEFACTOS DE BÚSQUEDA
    # ═══════════════════════════════════════════════════════════════════════════

    artifacts_valid, error_msg = validate_search_artifacts(search_artifacts, log)

    if not artifacts_valid:
        log.append(f"  ❌ Búsqueda semántica deshabilitada: {error_msg}")
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDACIÓN DE ENTRADAS
    # ═══════════════════════════════════════════════════════════════════════════

    # Validar DataFrame
    if not isinstance(df_pool, pd.DataFrame):
        log.append(f"  ❌ df_pool no es DataFrame (tipo: {type(df_pool).__name__})")
        return None, None

    if df_pool.empty:
        log.append("  ⚠️ Pool de APUs vacío para búsqueda semántica.")
        return None, None

    # Validar columna CODIGO_APU
    if "CODIGO_APU" not in df_pool.columns:
        log.append("  ❌ ERROR: Columna 'CODIGO_APU' no encontrada en df_pool.")
        return None, None

    # Validar texto de consulta
    if not query_text or not isinstance(query_text, str):
        log.append("  ⚠️ Texto de consulta vacío o inválido.")
        return None, None

    query_clean = query_text.strip()
    if len(query_clean) < MIN_DESCRIPTION_LENGTH:
        log.append(f"  ⚠️ Texto de consulta muy corto: '{query_clean}'")
        return None, None

    # Validar y ajustar parámetros numéricos
    min_similarity = safe_float_conversion(
        min_similarity, DEFAULT_MIN_SIMILARITY, min_value=0.0, max_value=1.0
    )

    top_k = safe_int_conversion(top_k, DEFAULT_TOP_K, min_value=1, max_value=MAX_FAISS_TOP_K)

    # ═══════════════════════════════════════════════════════════════════════════
    # LOG INICIAL
    # ═══════════════════════════════════════════════════════════════════════════

    log.append(
        f"  🧠 Búsqueda Semántica: '{query_clean[:50]}{'...' if len(query_clean) > 50 else ''}'"
    )
    log.append(f"  📊 Pool size: {len(df_pool)} APUs")
    log.append(f"  ⚙️ Umbral similitud: {min_similarity:.2f} | Top-K: {top_k}")

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERAR EMBEDDING
    # ═══════════════════════════════════════════════════════════════════════════

    try:
        query_embedding = search_artifacts.model.encode(
            [query_clean],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Validar embedding generado
        if query_embedding is None:
            log.append("  ❌ ERROR: Embedding generado es None")
            return None, None

        if not isinstance(query_embedding, np.ndarray):
            log.append(
                f"  ❌ ERROR: Embedding no es ndarray (tipo: {type(query_embedding).__name__})"
            )
            return None, None

        if query_embedding.size == 0:
            log.append("  ❌ ERROR: Embedding generado está vacío")
            return None, None

        # Verificar dimensionalidad
        expected_dim = search_artifacts.model.get_sentence_embedding_dimension()
        actual_dim = query_embedding.shape[-1]

        if actual_dim != expected_dim:
            log.append(
                f"  ❌ ERROR: Dimensionalidad incorrecta. "
                f"Esperado: {expected_dim}, Obtenido: {actual_dim}"
            )
            return None, None

        # Verificar valores válidos
        if np.any(np.isnan(query_embedding)) or np.any(np.isinf(query_embedding)):
            log.append("  ❌ ERROR: Embedding contiene valores NaN o Inf")
            return None, None

        log.append(f"  ✓ Embedding generado (dim={actual_dim})")

    except Exception as e:
        log.append(f"  ❌ ERROR al generar embedding: {type(e).__name__}: {str(e)}")
        logger.exception("Error en generación de embedding")
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # BÚSQUEDA EN FAISS
    # ═══════════════════════════════════════════════════════════════════════════

    try:
        # Asegurar tipo correcto para FAISS
        query_vector = query_embedding.astype(np.float32)

        # Ajustar top_k si es mayor que el tamaño del índice
        index_size = search_artifacts.faiss_index.ntotal

        if index_size == 0:
            log.append("  ❌ ERROR: Índice FAISS vacío")
            return None, None

        actual_k = min(top_k, index_size)

        if actual_k != top_k:
            log.append(f"  ⚠️ top_k ajustado de {top_k} a {actual_k} (tamaño del índice)")

        # Ejecutar búsqueda
        distances, indices = search_artifacts.faiss_index.search(query_vector, k=actual_k)

        # Validar resultados
        if distances is None or indices is None:
            log.append("  ❌ ERROR: FAISS retornó resultados nulos")
            return None, None

        if distances.size == 0 or indices.size == 0:
            log.append("  ⚠️ FAISS no retornó resultados")
            return None, None

        log.append(f"  ✓ FAISS retornó {len(indices[0])} resultados")

    except Exception as e:
        error_type = type(e).__name__
        log.append(f"  ❌ ERROR en búsqueda FAISS ({error_type}): {str(e)}")
        logger.exception("Error durante búsqueda en FAISS")
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # PROCESAMIENTO DE RESULTADOS
    # ═══════════════════════════════════════════════════════════════════════════

    candidates: List[MatchCandidate] = []

    # Crear conjunto de códigos APU del pool para búsqueda O(1)
    try:
        pool_apu_codes = set(df_pool["CODIGO_APU"].astype(str).str.strip())
    except Exception as e:
        log.append(f"  ⚠️ Error creando índice de códigos APU: {e}")
        pool_apu_codes = set()

    invalid_indices = 0
    not_in_pool = 0
    invalid_similarity = 0

    for i in range(len(indices[0])):
        try:
            faiss_idx = int(indices[0][i])
            similarity = float(distances[0][i])
        except (ValueError, TypeError, IndexError):
            invalid_indices += 1
            continue

        # Validar similitud
        if not (0.0 <= similarity <= 1.0):
            # Para índices de producto interno, la similitud puede ser > 1
            # Normalizar si es necesario
            if similarity > 1.0:
                similarity = min(similarity, 1.0)
            elif similarity < 0.0:
                invalid_similarity += 1
                continue

        # Obtener código APU desde el mapa
        apu_code = search_artifacts.id_map.get(str(faiss_idx))

        if apu_code is None:
            logger.debug(f"Índice FAISS {faiss_idx} no encontrado en id_map")
            continue

        # Limpiar código APU
        apu_code = str(apu_code).strip()

        # Verificar si el APU está en el pool actual
        if apu_code not in pool_apu_codes:
            not_in_pool += 1
            continue

        # Buscar el APU en el DataFrame
        try:
            apu_matches = df_pool[df_pool["CODIGO_APU"].astype(str).str.strip() == apu_code]
        except Exception as e:
            logger.debug(f"Error buscando APU {apu_code}: {e}")
            continue

        if apu_matches.empty:
            continue

        apu = apu_matches.iloc[0]
        original_desc = get_safe_column_value(apu, "original_description", "Sin descripción")

        # Crear detalles de derivación
        details = DerivationDetails(
            match_method="SEMANTIC",
            confidence_score=similarity,
            source="Vector Database (FAISS)",
            reasoning=f"Similitud semántica: {similarity:.3f} con '{original_desc[:50]}...'",
        )

        candidate = MatchCandidate(
            apu=apu,
            description=original_desc,
            matches=0,
            percentage=0.0,
            similarity=similarity,
            details=details,
        )
        candidates.append(candidate)

    # Log de estadísticas de procesamiento
    if invalid_indices > 0:
        log.append(f"  ⚠️ Índices inválidos: {invalid_indices}")
    if not_in_pool > 0:
        log.append(f"  ⚠️ APUs no en pool filtrado: {not_in_pool}")
    if invalid_similarity > 0:
        log.append(f"  ⚠️ Similitudes inválidas: {invalid_similarity}")

    if not candidates:
        log.append("  ⚠️ Ningún resultado de FAISS coincide con el pool filtrado")
        return None, None

    # ═══════════════════════════════════════════════════════════════════════════
    # ORDENAR Y SELECCIONAR
    # ═══════════════════════════════════════════════════════════════════════════

    candidates.sort(
        key=lambda x: x.similarity if x.similarity is not None else 0.0, reverse=True
    )
    _log_top_candidates(candidates, log, top_n=3)

    best_candidate = candidates[0]
    best_sim = best_candidate.similarity if best_candidate.similarity is not None else 0.0

    if best_sim >= min_similarity:
        log.append(
            f"  ✅ Coincidencia semántica encontrada: {best_sim:.3f} ≥ {min_similarity:.2f}"
        )
        return best_candidate.apu, best_candidate.details
    else:
        log.append(
            f"  ❌ Sin coincidencia válida. Mejor similitud: "
            f"{best_sim:.3f} < {min_similarity:.2f}"
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

    ROBUSTECIDO:
    - Validación exhaustiva de todos los parámetros de entrada
    - Evaluación de calidad de datos antes de procesar
    - Manejo defensivo de datos faltantes o corruptos
    - Fallbacks múltiples con logging detallado
    - Métricas de confianza en los resultados
    - Coherente con validaciones de apu_processor y report_parser_crudo

    Args:
        params: Diccionario con parámetros de entrada.
        data_store: Diccionario con datos procesados.
        config: Diccionario con configuración.
        search_artifacts: Artefactos de búsqueda semántica.

    Returns:
        Dict con resultados de estimación.
    """
    log: List[str] = ["🕵️ ESTIMADOR HÍBRIDO INICIADO"]
    log.append("=" * 70)

    derivation_details = {"suministro": None, "tarea": None, "cuadrilla": None}

    # Estructura de respuesta de error estándar
    def error_response(msg: str) -> Dict[str, Any]:
        log.append(f"  ❌ ERROR: {msg}")
        return {
            "error": msg,
            "log": "\n".join(log),
            "derivation_details": derivation_details,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDACIÓN DE PARÁMETROS DE ENTRADA
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n🔒 VALIDACIÓN DE ENTRADAS")
    log.append("-" * 70)

    # Validar params
    if not isinstance(params, dict):
        return error_response(f"params no es dict (tipo: {type(params).__name__})")

    # Validar data_store
    if not isinstance(data_store, dict):
        return error_response(f"data_store no es dict (tipo: {type(data_store).__name__})")

    # Validar config
    if not isinstance(config, dict):
        log.append("  ⚠️ config no es dict, usando configuración vacía")
        config = {}

    # Validar search_artifacts (no bloqueante, pero registrar)
    artifacts_valid, artifacts_msg = validate_search_artifacts(
        search_artifacts, log, require_all=False
    )
    if not artifacts_valid:
        log.append(f"  ⚠️ Artefactos de búsqueda no disponibles: {artifacts_msg}")
        log.append("  📝 La búsqueda semántica estará deshabilitada")

    log.append("  ✓ Validación de entradas completada")

    # ═══════════════════════════════════════════════════════════════════════════
    # CARGA Y VALIDACIÓN DE DATOS
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n📦 CARGA Y VALIDACIÓN DE DATOS")
    log.append("-" * 70)

    # Cargar APUs procesados
    processed_apus_raw = data_store.get("processed_apus", [])

    if not processed_apus_raw:
        return error_response("No hay datos de APU procesados (processed_apus vacío)")

    if not isinstance(processed_apus_raw, (list, pd.DataFrame)):
        return error_response(
            f"processed_apus tiene tipo inválido: {type(processed_apus_raw).__name__}"
        )

    # Convertir a DataFrame si es lista
    try:
        if isinstance(processed_apus_raw, list):
            if len(processed_apus_raw) == 0:
                return error_response("Lista de APUs procesados está vacía")
            df_processed_apus = pd.DataFrame(processed_apus_raw)
        else:
            df_processed_apus = processed_apus_raw.copy()
    except Exception as e:
        return error_response(f"Error convirtiendo APUs a DataFrame: {e}")

    log.append(f"  ✓ APUs cargados: {len(df_processed_apus)} registros")

    # Evaluar calidad de datos
    quality_metrics = assess_data_quality(
        df_processed_apus, REQUIRED_COLUMNS_APU_CRITICAL, REQUIRED_COLUMNS_APU_OPTIONAL, log
    )

    if not quality_metrics.is_acceptable():
        log.append(f"  ⚠️ Calidad de datos baja ({quality_metrics.quality_score * 100:.1f}%)")
        if quality_metrics.valid_records < MIN_VALID_APUS_FOR_ESTIMATION:
            return error_response(
                f"Insuficientes APUs válidos: {quality_metrics.valid_records} < {MIN_VALID_APUS_FOR_ESTIMATION}"
            )

    # Validar columnas críticas
    cols_valid, present_cols, missing_cols = validate_dataframe_columns(
        df_processed_apus, REQUIRED_COLUMNS_APU_CRITICAL, "df_processed_apus", strict=True
    )

    if not cols_valid:
        return error_response(f"Columnas críticas faltantes: {missing_cols}")

    # ═══════════════════════════════════════════════════════════════════════════
    # EXTRACCIÓN Y VALIDACIÓN DE PARÁMETROS
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n🎛️ EXTRACCIÓN DE PARÁMETROS")
    log.append("-" * 70)

    # Extraer parámetros con valores por defecto seguros
    material = ""
    try:
        material_raw = params.get("material", "")
        if material_raw and isinstance(material_raw, str):
            material = material_raw.strip().upper()
    except Exception as e:
        log.append(f"  ⚠️ Error extrayendo material: {e}")

    if not material:
        return error_response(
            "El parámetro 'material' es obligatorio y no puede estar vacío"
        )

    # Extraer otros parámetros con manejo defensivo
    def safe_param(key: str, default: str, uppercase: bool = True) -> str:
        try:
            val = params.get(key, default) or default
            if isinstance(val, str):
                return val.strip().upper() if uppercase else val.strip()
            return str(val).strip().upper() if uppercase else str(val).strip()
        except Exception:
            return default

    cuadrilla = safe_param("cuadrilla", "0", uppercase=False)
    zona = safe_param("zona", DEFAULT_ZONA)
    izaje = safe_param("izaje", DEFAULT_IZAJE)
    seguridad = safe_param("seguridad", DEFAULT_SEGURIDAD)

    log.append(f"  • Material: '{material}'")
    log.append(f"  • Cuadrilla: '{cuadrilla}'")
    log.append(f"  • Zona: '{zona}'")
    log.append(f"  • Izaje: '{izaje}'")
    log.append(f"  • Seguridad: '{seguridad}'")

    # Obtener mapeo de parámetros
    param_map = config.get("param_map", {})
    if not isinstance(param_map, dict):
        log.append("  ⚠️ param_map no es dict, usando valores sin mapear")
        param_map = {}

    material_mapped = param_map.get("material", {}).get(material, material) or material
    log.append(f"  • Material mapeado: '{material_mapped}'")

    # Generar keywords normalizadas con manejo de errores
    try:
        material_keywords = normalize_text(material_mapped).split()
        if not material_keywords:
            material_keywords = material_mapped.lower().split()
    except Exception as e:
        log.append(f"  ⚠️ Error al normalizar material: {e}")
        material_keywords = material_mapped.lower().split()

    # Filtrar keywords válidas
    material_keywords = [kw for kw in material_keywords if kw and len(kw) >= 2]

    if not material_keywords:
        return error_response(f"No se pudieron extraer keywords válidas de '{material}'")

    log.append(
        f"  • Keywords: {material_keywords[:10]}{'...' if len(material_keywords) > 10 else ''}"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # EXTRACCIÓN DE UMBRALES
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n⚙️ CONFIGURACIÓN DE UMBRALES")
    log.append("-" * 70)

    thresholds = config.get("estimator_thresholds", {})
    if not isinstance(thresholds, dict):
        log.append("  ⚠️ estimator_thresholds no es dict, usando valores por defecto")
        thresholds = {}

    min_sim_suministro = safe_float_conversion(
        thresholds.get("min_semantic_similarity_suministro", 0.30),
        0.30,
        min_value=0.0,
        max_value=1.0,
    )
    min_sim_tarea = safe_float_conversion(
        thresholds.get("min_semantic_similarity_tarea", 0.40),
        0.40,
        min_value=0.0,
        max_value=1.0,
    )
    min_kw_cuadrilla = safe_float_conversion(
        thresholds.get("min_keyword_match_percentage_cuadrilla", 50.0),
        50.0,
        min_value=0.0,
        max_value=100.0,
    )

    log.append(f"  • Similitud mínima (Suministro): {min_sim_suministro:.2f}")
    log.append(f"  • Similitud mínima (Tarea): {min_sim_tarea:.2f}")
    log.append(f"  • Match mínimo (Cuadrilla): {min_kw_cuadrilla:.0f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # BÚSQUEDA #1: SUMINISTRO
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #1: SUMINISTRO")
    log.append("=" * 70)

    apu_suministro = None
    details_suministro = None
    valor_suministro = 0.0
    apu_suministro_desc = "No encontrado"
    apu_suministro_codigo = "N/A"

    # Filtrar pool de suministros
    supply_types = [TipoAPU.SUMINISTRO.value, TipoAPU.SUMINISTRO_PREFABRICADO.value]

    if "tipo_apu" in df_processed_apus.columns:
        try:
            df_suministro_pool = df_processed_apus[
                df_processed_apus["tipo_apu"].isin(supply_types)
            ].copy()
            log.append(f"  📦 Pool de suministros: {len(df_suministro_pool)} APUs")
        except Exception as e:
            log.append(f"  ⚠️ Error filtrando suministros: {e}")
            df_suministro_pool = df_processed_apus.copy()
    else:
        log.append("  ⚠️ Columna 'tipo_apu' no encontrada, usando pool completo")
        df_suministro_pool = df_processed_apus.copy()

    if not df_suministro_pool.empty:
        # Intentar búsqueda semántica primero
        if artifacts_valid:
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
    else:
        log.append("  ⚠️ Pool de suministros vacío")

    if details_suministro:
        derivation_details["suministro"] = details_suministro.__dict__

    # Extraer valores de suministro
    if apu_suministro is not None:
        valor_suministro = safe_float_conversion(
            apu_suministro.get("VALOR_SUMINISTRO_UN", 0.0), 0.0, min_value=0.0
        )
        apu_suministro_desc = get_safe_column_value(
            apu_suministro, "original_description", "Sin descripción"
        )
        apu_suministro_codigo = get_safe_column_value(apu_suministro, "CODIGO_APU", "N/A")
        log.append(f"\n  ✅ APU encontrado: {apu_suministro_codigo}")
        log.append(f"  💰 Valor Suministro: ${valor_suministro:,.2f}")
    else:
        log.append("\n  ❌ No se encontró APU de suministro")

    # ═══════════════════════════════════════════════════════════════════════════
    # BÚSQUEDA #2: CUADRILLA
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #2: CUADRILLA")
    log.append("=" * 70)

    costo_diario_cuadrilla = 0.0
    apu_cuadrilla_desc = "No encontrada"
    apu_cuadrilla_codigo = "N/A"

    cuadrilla_num = safe_int_conversion(cuadrilla, 0, min_value=0)

    if cuadrilla_num > 0:
        log.append(f"  👷 Buscando cuadrilla #{cuadrilla_num}")

        # Filtrar pool de cuadrillas
        if "UNIDAD" in df_processed_apus.columns:
            try:
                df_cuadrilla_pool = df_processed_apus[
                    df_processed_apus["UNIDAD"].astype(str).str.upper().str.strip() == "DIA"
                ].copy()
                log.append(f"  📦 Pool de cuadrillas: {len(df_cuadrilla_pool)} APUs")
            except Exception as e:
                log.append(f"  ⚠️ Error filtrando cuadrillas: {e}")
                df_cuadrilla_pool = df_processed_apus.copy()
        else:
            log.append("  ⚠️ Columna 'UNIDAD' no encontrada")
            df_cuadrilla_pool = df_processed_apus.copy()

        if not df_cuadrilla_pool.empty:
            # Búsqueda por keywords
            search_term = f"cuadrilla {cuadrilla_num}"
            try:
                cuadrilla_keywords = normalize_text(search_term).split()
            except Exception:
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
                    apu_cuadrilla.get("VALOR_CONSTRUCCION_UN", 0.0), 0.0, min_value=0.0
                )
                apu_cuadrilla_desc = get_safe_column_value(
                    apu_cuadrilla, "original_description", "Sin descripción"
                )
                apu_cuadrilla_codigo = get_safe_column_value(
                    apu_cuadrilla, "CODIGO_APU", "N/A"
                )
                log.append(f"\n  ✅ APU encontrado: {apu_cuadrilla_codigo}")
                log.append(f"  💰 Costo Cuadrilla: ${costo_diario_cuadrilla:,.2f}/día")
            else:
                log.append("\n  ❌ No se encontró APU de cuadrilla")
        else:
            log.append("  ⚠️ Pool de cuadrillas vacío")
    else:
        log.append("  ⏭️ Cuadrilla no especificada, omitiendo búsqueda")

    # ═══════════════════════════════════════════════════════════════════════════
    # BÚSQUEDA #3: TAREA (RENDIMIENTO)
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n" + "=" * 70)
    log.append("🎯 BÚSQUEDA #3: TAREA (RENDIMIENTO)")
    log.append("=" * 70)

    apu_tarea = None
    details_tarea = None
    rendimiento_dia = 0.0
    costo_equipo = 0.0
    apu_tarea_desc = "No encontrado"
    apu_tarea_codigo = "N/A"

    # Filtrar pool de tareas
    if "tipo_apu" in df_processed_apus.columns:
        try:
            df_tarea_pool = df_processed_apus[
                df_processed_apus["tipo_apu"] == TipoAPU.INSTALACION.value
            ].copy()
            log.append(f"  📦 Pool de tareas: {len(df_tarea_pool)} APUs")
        except Exception as e:
            log.append(f"  ⚠️ Error filtrando tareas: {e}")
            df_tarea_pool = df_processed_apus.copy()
    else:
        log.append("  ⚠️ Columna 'tipo_apu' no encontrada, usando pool completo")
        df_tarea_pool = df_processed_apus.copy()

    if not df_tarea_pool.empty:
        # Intentar búsqueda semántica primero
        if artifacts_valid:
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
    else:
        log.append("  ⚠️ Pool de tareas vacío")

    # Fallback #2: Promedio de Históricos
    if apu_tarea is None:
        log.append(
            "\n  ⚠️ Tarea específica no encontrada. Intentando promedio de históricos..."
        )

        apu_tarea, details_tarea, rendimiento_dia = _calculate_historical_average(
            df_processed_apus, material_keywords, log
        )

    if details_tarea:
        derivation_details["tarea"] = details_tarea.__dict__

    # Extraer valores de tarea
    if apu_tarea is not None:
        apu_tarea_desc = get_safe_column_value(
            apu_tarea, "original_description", "Sin descripción"
        )
        apu_tarea_codigo = get_safe_column_value(apu_tarea, "CODIGO_APU", "N/A")
        costo_equipo = safe_float_conversion(
            apu_tarea.get("EQUIPO", 0.0), 0.0, min_value=0.0
        )

        log.append(f"\n  ✅ APU encontrado: {apu_tarea_codigo}")
        log.append(f"  🛠️ Costo Equipo: ${costo_equipo:,.2f}")

        # Calcular rendimiento si no viene del promedio histórico
        if apu_tarea_codigo != "EST-AVG" or rendimiento_dia <= 0:
            rendimiento_dia = _calculate_rendimiento_from_detail(
                apu_tarea_codigo, data_store, log
            )
    else:
        log.append("\n  ❌ No se encontró APU de tarea")

    # ═══════════════════════════════════════════════════════════════════════════
    # CÁLCULO FINAL
    # ═══════════════════════════════════════════════════════════════════════════

    log.append("\n" + "=" * 70)
    log.append("🧮 CÁLCULO FINAL DE COSTOS")
    log.append("=" * 70)

    # Obtener reglas de negocio con validación
    rules = config.get("estimator_rules", {})
    if not isinstance(rules, dict):
        rules = {}

    # Factores de ajuste con validación
    factor_zona = safe_float_conversion(
        rules.get("factores_zona", {}).get(zona, 1.0), 1.0, min_value=0.1, max_value=10.0
    )
    costo_adicional_izaje = safe_float_conversion(
        rules.get("costo_adicional_izaje", {}).get(izaje, 0.0), 0.0, min_value=0.0
    )
    factor_seguridad = safe_float_conversion(
        rules.get("factor_seguridad", {}).get(seguridad, 1.0),
        1.0,
        min_value=0.1,
        max_value=10.0,
    )

    log.append(f"  📍 Factor Zona ({zona}): {factor_zona:.2f}")
    log.append(f"  🏗️ Costo Adicional Izaje ({izaje}): ${costo_adicional_izaje:,.2f}")
    log.append(f"  🦺 Factor Seguridad ({seguridad}): {factor_seguridad:.2f}")

    # Cálculo de costo de mano de obra
    costo_mo_base = 0.0
    if rendimiento_dia > 0 and costo_diario_cuadrilla > 0:
        costo_mo_base = costo_diario_cuadrilla / rendimiento_dia
        log.append(f"\n  👷 Costo MO Base: ${costo_mo_base:,.2f}")
    else:
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

    # Validar resultado final
    if valor_construccion < 0:
        log.append("\n  ⚠️ ADVERTENCIA: Valor de construcción negativo, ajustando a 0")
        valor_construccion = 0.0

    log.append("\n" + "=" * 70)
    log.append(f"💰 VALOR TOTAL CONSTRUCCIÓN: ${valor_construccion:,.2f}")
    log.append(f"   ├─ Suministro: ${valor_suministro:,.2f}")
    log.append(f"   └─ Instalación: ${valor_instalacion:,.2f}")
    log.append("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════════
    # PREPARAR RESPUESTA
    # ═══════════════════════════════════════════════════════════════════════════

    apu_encontrado_str = (
        f"Suministro: {apu_suministro_desc[:50]} ({apu_suministro_codigo}) | "
        f"Tarea: {apu_tarea_desc[:50]} ({apu_tarea_codigo}) | "
        f"Cuadrilla: {apu_cuadrilla_desc[:50]} ({apu_cuadrilla_codigo})"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # ANÁLISIS FINANCIERO ADICIONAL
    # ═══════════════════════════════════════════════════════════════════════════
    log.append("\n" + "=" * 70)
    log.append("📈 ANÁLISIS DE VIABILIDAD FINANCIERA")
    log.append("=" * 70)

    financial_analysis = {}

    # 1. Obtener insumos para Monte Carlo
    apu_codes_for_simulation = [
        c
        for c in [apu_suministro_codigo, apu_tarea_codigo]
        if c and c not in ["N/A", "EST-AVG"]
    ]

    if apu_codes_for_simulation:
        all_insumos = data_store.get("apus_detail", [])
        insumos_for_simulation = [
            item
            for item in all_insumos
            if item.get("CODIGO_APU") in apu_codes_for_simulation
        ]
        log.append(
            f"  🔍 Insumos para simulación: {len(insumos_for_simulation)} items de {len(apu_codes_for_simulation)} APUs"
        )

        if insumos_for_simulation:
            # 2. Ejecutar simulación de Monte Carlo
            mc_results = run_monte_carlo_simulation(insumos_for_simulation)
            mc_stats = mc_results.get("statistics", {})
            mean_cost = mc_stats.get("mean", valor_construccion)
            std_dev = mc_stats.get("std_dev", 0)
            log.append(
                f"  📊 Resultados Monte Carlo: media=${mean_cost:,.2f}, std_dev=${std_dev:,.2f}"
            )

            # 3. Configurar y ejecutar motor financiero
            try:
                # Extraer parámetros financieros opcionales del request
                config_params = {
                    "risk_free_rate": safe_float_conversion(
                        params.get("risk_free_rate"), 0.04
                    ),
                    "market_premium": safe_float_conversion(
                        params.get("market_premium"), 0.06
                    ),
                    "beta": safe_float_conversion(params.get("beta"), 1.2),
                    "tax_rate": safe_float_conversion(params.get("tax_rate"), 0.30),
                    "cost_of_debt": safe_float_conversion(params.get("cost_of_debt"), 0.08),
                    "debt_to_equity_ratio": safe_float_conversion(
                        params.get("debt_to_equity_ratio"), 0.6
                    ),
                }
                fin_config = FinancialConfig(**config_params)

                # Instanciar motores
                capm_engine = CapitalAssetPricing(fin_config)
                risk_engine = RiskQuantifier()
                options_engine = RealOptionsAnalyzer()

                # Calcular métricas
                wacc = capm_engine.calculate_wacc()
                var_95 = risk_engine.calculate_var(mean_cost, std_dev, confidence_level=0.95)
                contingency = risk_engine.suggest_contingency(std_dev)

                # Para la opción de esperar, se asumen algunos valores
                # (ej. valor del proyecto es un % sobre el costo, volatilidad derivada de std_dev)
                project_value = mean_cost * 1.15  # Asumiendo un 15% de margen
                volatility = (std_dev / mean_cost) if mean_cost > 0 else 0

                option_value = options_engine.value_option_to_wait(
                    project_value=project_value,
                    investment_cost=mean_cost,
                    risk_free_rate=fin_config.risk_free_rate,
                    time_to_expire_years=1.0,  # Asumimos 1 año para decidir
                    volatility=volatility,
                )

                financial_analysis = {
                    "wacc": round(wacc, 4),
                    "var_95": round(var_95, 2),
                    "suggested_contingency": round(contingency, 2),
                    "option_to_wait_value": round(option_value, 2)
                    if option_value is not None
                    else None,
                    "monte_carlo_mean": round(mean_cost, 2),
                    "monte_carlo_std_dev": round(std_dev, 2),
                }
                log.append(
                    f"  ✅ Análisis financiero completado. WACC: {wacc:.2%}, VaR: ${var_95:,.2f}"
                )

            except Exception as e:
                log.append(f"  ❌ Error en el motor financiero: {e}")
                logger.error(f"Error en FinancialEngine: {e}", exc_info=True)
        else:
            log.append(
                "  ⚠️ No se encontraron insumos detallados para la simulación financiera."
            )
    else:
        log.append(
            "  ⚠️ No se encontraron APUs válidos para ejecutar el análisis financiero."
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
        "financial_analysis": financial_analysis,
        "data_quality": {
            "total_apus": quality_metrics.total_records,
            "valid_apus": quality_metrics.valid_records,
            "quality_score": round(quality_metrics.quality_score, 3),
        },
        "derivation_details": derivation_details,
        "log": "\n".join(log),
    }


def _calculate_historical_average(
    df_processed_apus: pd.DataFrame, material_keywords: List[str], log: List[str]
) -> Tuple[Optional[pd.Series], Optional[DerivationDetails], float]:
    """
    Calcula el rendimiento promedio de items históricos similares.

    ROBUSTECIDO:
    - Manejo de excepciones en cada paso
    - Validación de resultados
    - Límites en la búsqueda

    Returns:
        Tuple[apu_sintetico, detalles, rendimiento_promedio]
    """
    try:
        # Construir regex de búsqueda (solo keywords significativas)
        significant_keywords = [k for k in material_keywords if len(k) > 3]

        if not significant_keywords:
            log.append("  ⚠️ No hay keywords significativas para promedio histórico")
            return None, None, 0.0

        keywords_regex = "|".join(significant_keywords)

        # Verificar que la columna existe
        if "DESC_NORMALIZED" not in df_processed_apus.columns:
            log.append("  ⚠️ Columna DESC_NORMALIZED no disponible")
            return None, None, 0.0

        # Buscar coincidencias
        try:
            mask_keywords = df_processed_apus["DESC_NORMALIZED"].str.contains(
                keywords_regex, case=False, regex=True, na=False
            )
            df_candidates = df_processed_apus[mask_keywords].copy()
        except Exception as e:
            log.append(f"  ⚠️ Error en búsqueda regex: {e}")
            return None, None, 0.0

        # Filtrar por rendimiento válido
        if "RENDIMIENTO_DIA" in df_candidates.columns:
            df_candidates = df_candidates[
                df_candidates["RENDIMIENTO_DIA"].apply(
                    lambda x: safe_float_conversion(x, 0.0) > 0
                )
            ]
        else:
            log.append("  ⚠️ Columna RENDIMIENTO_DIA no disponible")
            return None, None, 0.0

        if df_candidates.empty:
            log.append(
                "  ❌ No se encontraron items similares con rendimiento para promediar"
            )
            return None, None, 0.0

        # Calcular promedio
        rendimientos = df_candidates["RENDIMIENTO_DIA"].apply(
            lambda x: safe_float_conversion(x, 0.0)
        )
        avg_rendimiento = rendimientos.mean()
        count_matches = len(df_candidates)

        log.append(f"  📊 Encontrados {count_matches} items similares con rendimiento")
        log.append(f"  ⏱️ Rendimiento promedio estimado: {avg_rendimiento:.4f} un/día")

        # Crear APU sintético
        apu_sintetico = pd.Series(
            {
                "CODIGO_APU": "EST-AVG",
                "original_description": f"Estimación Promedio ({count_matches} items similares)",
                "RENDIMIENTO_DIA": avg_rendimiento,
                "EQUIPO": 0.0,
            }
        )

        details = DerivationDetails(
            match_method="HISTORICAL_AVERAGE",
            confidence_score=min(0.5, count_matches / 10),  # Más matches = más confianza
            source="Promedio Histórico",
            reasoning=f"Promedio de {count_matches} items similares encontrados en histórico",
        )

        return apu_sintetico, details, avg_rendimiento

    except Exception as e:
        log.append(f"  ⚠️ Error en cálculo de promedio histórico: {e}")
        logger.exception("Error en _calculate_historical_average")
        return None, None, 0.0


def _calculate_rendimiento_from_detail(
    apu_codigo: str, data_store: Dict[str, Any], log: List[str]
) -> float:
    """
    Calcula el rendimiento a partir del detalle de APUs.

    ROBUSTECIDO:
    - Validación de estructura de datos
    - Manejo de excepciones
    - Valores por defecto seguros

    Returns:
        Rendimiento calculado (0.0 si no se puede calcular)
    """
    log.append("\n  📊 Calculando rendimiento desde detalle...")

    apus_detail_list = data_store.get("apus_detail", [])

    if not apus_detail_list:
        log.append("  ⚠️ No hay datos de detalle de APUs disponibles")
        return 0.0

    if not isinstance(apus_detail_list, list):
        log.append(f"  ⚠️ apus_detail no es lista (tipo: {type(apus_detail_list).__name__})")
        return 0.0

    try:
        df_detail = pd.DataFrame(apus_detail_list)
    except Exception as e:
        log.append(f"  ⚠️ Error convirtiendo detalle a DataFrame: {e}")
        return 0.0

    # Validar columnas
    cols_valid, _, missing = validate_dataframe_columns(
        df_detail, REQUIRED_COLUMNS_DETAIL_CRITICAL, "df_detail", strict=False
    )

    if missing:
        log.append(f"  ⚠️ Columnas faltantes en detalle: {missing}")

    # Verificar columna CANTIDAD_APU
    if "CANTIDAD_APU" not in df_detail.columns:
        log.append("  ⚠️ Columna CANTIDAD_APU no disponible")
        return 0.0

    try:
        # Filtrar mano de obra para este APU
        mano_obra = df_detail[
            (df_detail["CODIGO_APU"].astype(str).str.strip() == str(apu_codigo).strip())
            & (df_detail["TIPO_INSUMO"] == TipoInsumo.MANO_OBRA.value)
        ]

        if mano_obra.empty:
            log.append("  ⚠️ No se encontró mano de obra para este APU")
            return 0.0

        # Sumar tiempo de mano de obra
        tiempo_total = sum(
            safe_float_conversion(cant, 0.0, min_value=0.0)
            for cant in mano_obra["CANTIDAD_APU"]
        )

        if tiempo_total > 0:
            rendimiento_dia = 1.0 / tiempo_total
            log.append(f"  ⏱️ Rendimiento: {rendimiento_dia:.4f} un/día")
            log.append(f"  ⏱️ Tiempo total mano de obra: {tiempo_total:.4f} días/un")
            return rendimiento_dia
        else:
            log.append("  ⚠️ Tiempo total de mano de obra es cero")
            return 0.0

    except Exception as e:
        log.append(f"  ❌ ERROR al calcular rendimiento: {e}")
        logger.exception("Error en cálculo de rendimiento")
        return 0.0
