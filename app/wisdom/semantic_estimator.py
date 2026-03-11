"""
Microservicio: Semantic Estimator (Asesor Táctico)
Estrato DIKW: TACTICS (Nivel 2)

Responsabilidad
---------------
Aloja el espacio vectorial continuo (FAISS) y resuelve ambigüedades
semánticas entre descripciones humanas e ítems formales del presupuesto.

Actúa como motor de inferencia desacoplado del Plano de Control,
exponiendo sus capacidades a través de la MIC (Matriz de Interacción Central).

Arquitectura interna
--------------------
- **SearchEngine**: Encapsula FAISS + SentenceTransformer + keyword fallback.
- **CostCalculator**: Ensambla los componentes de costo (suministro, mano de
  obra, equipo) aplicando factores de zona, izaje y seguridad.
- **SemanticEstimatorService**: Fachada que orquesta ambos y se registra en
  la MIC.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.core.schemas import Stratum
from app.core.telemetry import TelemetryContext
from app.core.utils import normalize_text
from app.core.probability_models import run_monte_carlo_simulation
from app.adapters.tools_interface import MICRegistry


logger = logging.getLogger("SemanticEstimator")


# ============================================================================
# CONSTANTES
# ============================================================================

# --- Límites de búsqueda ---
MAX_POOL_SIZE_FOR_ITERATION: Final[int] = 50_000
MAX_CANDIDATES_TO_TRACK: Final[int] = 1_000
MAX_FAISS_TOP_K: Final[int] = 500
MIN_DESCRIPTION_LENGTH: Final[int] = 2
MAX_KEYWORDS_COUNT: Final[int] = 50

# --- Calidad de datos ---
MIN_DATA_QUALITY_THRESHOLD: Final[float] = 0.3
MIN_VALID_APUS_FOR_ESTIMATION: Final[int] = 10

# --- Columnas esperadas ---
REQUIRED_COLUMNS_APU_CRITICAL: Final[Tuple[str, ...]] = (
    "CODIGO_APU",
    "DESC_NORMALIZED",
)
REQUIRED_COLUMNS_APU_OPTIONAL: Final[Tuple[str, ...]] = (
    "original_description",
    "tipo_apu",
    "UNIDAD",
)
REQUIRED_COLUMNS_DETAIL_CRITICAL: Final[Tuple[str, ...]] = (
    "CODIGO_APU",
    "TIPO_INSUMO",
)
REQUIRED_COLUMNS_DETAIL_OPTIONAL: Final[Tuple[str, ...]] = ("CANTIDAD_APU",)

# --- Valores por defecto de búsqueda ---
DEFAULT_MIN_SIMILARITY: Final[float] = 0.5
DEFAULT_MIN_MATCH_PERCENTAGE: Final[float] = 30.0
DEFAULT_TOP_K: Final[int] = 5

# --- Valores por defecto de estimación ---
DEFAULT_ZONA: Final[str] = "ZONA 0"
DEFAULT_IZAJE: Final[str] = "MANUAL"
DEFAULT_SEGURIDAD: Final[str] = "NORMAL"

# --- Protección numérica ---
MIN_RENDIMIENTO_SAFE: Final[float] = 1e-6
MAX_COSTO_UNITARIO: Final[float] = 1e12


# ============================================================================
# ENUMERACIONES LOCALES
# ============================================================================

class MatchMode(str, Enum):
    """Estrategia de comparación textual."""
    WORDS = "words"
    SUBSTRING = "substring"


class TipoAPU(str, Enum):
    """Clasificación funcional de un APU."""
    SUMINISTRO = "Suministro"
    SUMINISTRO_PREFABRICADO = "Suministro (Pre-fabricado)"
    INSTALACION = "Instalación"


class TipoRecurso(str, Enum):
    """
    Tipos de recurso usados en el detalle de APU.

    .. note::
        Distinto de ``app.core.schemas.TipoInsumo`` que modela la ontología
        del insumo individual. Este enum clasifica *roles* dentro del
        cálculo de estimación.
    """
    MANO_OBRA = "MANO DE OBRA"
    EQUIPO = "EQUIPO"
    MATERIAL = "MATERIAL"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass(frozen=True, slots=True)
class DerivationDetails:
    """Trazabilidad de cómo se derivó un match."""
    match_method: str
    confidence_score: float
    source: str
    reasoning: str


@dataclass(slots=True)
class MatchCandidate:
    """Candidato evaluado durante una búsqueda."""
    apu: pd.Series
    description: str
    matches: int
    percentage: float
    similarity: Optional[float] = None
    details: Optional[DerivationDetails] = None

    @property
    def rank_key(self) -> Tuple[float, float, int]:
        """Clave de ordenamiento: (similitud, porcentaje, matches)."""
        sim = self.similarity if self.similarity is not None else 0.0
        return (sim, self.percentage, self.matches)


@dataclass(frozen=True, slots=True)
class SearchArtifacts:
    """Artefactos inmutables para búsqueda vectorial."""
    model: SentenceTransformer
    faiss_index: Any
    id_map: Dict[str, str]

    def is_complete(self) -> bool:
        """Verifica que todos los componentes estén presentes."""
        return (
            self.model is not None
            and self.faiss_index is not None
            and self.id_map is not None
            and len(self.id_map) > 0
        )


@dataclass(slots=True)
class DataQualityMetrics:
    """Métricas de calidad de un DataFrame de APUs."""
    total_records: int = 0
    valid_records: int = 0
    missing_descriptions: int = 0
    missing_codes: int = 0
    quality_score: float = 0.0

    def is_acceptable(
        self, threshold: float = MIN_DATA_QUALITY_THRESHOLD
    ) -> bool:
        return self.quality_score >= threshold


@dataclass(frozen=True, slots=True)
class EstimationFactors:
    """Factores de ajuste extraídos de la configuración."""
    factor_zona: float = 1.0
    costo_izaje: float = 0.0
    factor_seguridad: float = 1.0


@dataclass(slots=True)
class EstimationComponents:
    """Componentes intermedios del cálculo de estimación."""
    valor_suministro: float = 0.0
    costo_mano_obra: float = 0.0
    costo_equipo: float = 0.0
    rendimiento: float = 0.0
    codigo_suministro: str = "N/A"
    codigo_tarea: str = "N/A"
    codigo_cuadrilla: str = "N/A"


# ============================================================================
# CONVERSIÓN SEGURA DE TIPOS
# ============================================================================

class SafeConvert:
    """
    Conversiones numéricas robustas con clamp y manejo de valores faltantes.

    Todos los métodos son estáticos y libres de efectos colaterales.
    """

    _EMPTY_STRINGS: ClassVar[FrozenSet[str]] = frozenset(
        {"", "-", "N/A", "NA", "null", "None", "none", "n/a"}
    )

    @staticmethod
    def to_float(
        value: Any,
        default: float = 0.0,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Convierte *value* a float de forma segura.

        Maneja ``None``, ``NaN``, ``Inf``, strings con comas, y valores
        centinela como ``"N/A"``.
        """
        if value is None:
            return default

        # Fast path para tipos nativos
        if isinstance(value, (int, float)):
            fval = float(value)
        elif isinstance(value, str):
            cleaned = value.strip().replace(",", "").replace(" ", "")
            if cleaned in SafeConvert._EMPTY_STRINGS:
                return default
            try:
                fval = float(cleaned)
            except (ValueError, TypeError):
                return default
        else:
            try:
                fval = float(value)
            except (ValueError, TypeError):
                return default

        # Pandas NaN check
        try:
            if pd.isna(fval):
                return default
        except (ValueError, TypeError):
            pass

        if not math.isfinite(fval):
            return default

        if min_value is not None and fval < min_value:
            fval = min_value
        if max_value is not None and fval > max_value:
            fval = max_value

        return fval

    @staticmethod
    def to_int(
        value: Any,
        default: int = 0,
        *,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """Convierte *value* a int via float intermedio."""
        fval = SafeConvert.to_float(value, float(default))
        result = int(fval)
        if min_value is not None and result < min_value:
            result = min_value
        if max_value is not None and result > max_value:
            result = max_value
        return result

    @staticmethod
    def column_value(
        row: pd.Series,
        column: str,
        default: Any = "",
        expected_type: type = str,
    ) -> Any:
        """
        Extrae un valor de una fila de DataFrame con tipo esperado.

        Retorna *default* si el valor es ``NaN``, vacío o de tipo incorrecto.
        """
        value = row.get(column, default)

        try:
            if pd.isna(value):
                return default
        except (ValueError, TypeError):
            pass

        if expected_type is str:
            s = str(value).strip()
            return s if s else default

        if expected_type is float:
            return SafeConvert.to_float(value, default)

        if expected_type is int:
            return SafeConvert.to_int(value, default)

        return value if isinstance(value, expected_type) else default


# ============================================================================
# ACUMULADOR DE LOG ESTRUCTURADO
# ============================================================================

class EstimationLog:
    """
    Acumulador de trazas de estimación.

    Encapsula la lista mutable de líneas de log, evitando que se pase
    como parámetro mutable a través de toda la cadena de llamadas.
    """

    __slots__ = ("_lines",)

    def __init__(self, header: str = "") -> None:
        self._lines: List[str] = []
        if header:
            self._lines.append(header)

    def info(self, msg: str) -> None:
        self._lines.append(msg)

    def warn(self, msg: str) -> None:
        self._lines.append(f"  ⚠️ {msg}")

    def error(self, msg: str) -> None:
        self._lines.append(f"  ❌ {msg}")

    def success(self, msg: str) -> None:
        self._lines.append(f"  ✅ {msg}")

    def section(self, title: str) -> None:
        self._lines.append("")
        self._lines.append("=" * 70)
        self._lines.append(f"🎯 {title}")
        self._lines.append("=" * 70)

    def subsection(self, msg: str) -> None:
        self._lines.append(f"\n  {msg}")

    def render(self) -> str:
        return "\n".join(self._lines)

    def log_top_candidates(
        self,
        candidates: Sequence[MatchCandidate],
        top_n: int = 3,
        keywords_count: int = 0,
    ) -> None:
        """Formatea los mejores candidatos para el log."""
        if not candidates:
            self.info("  📋 No se encontraron candidatos.")
            return

        n = min(top_n, len(candidates))
        self.info(f"  📋 Top {n} candidatos:")

        for i, cand in enumerate(candidates[:n], 1):
            desc = (
                f"{cand.description[:60]}..."
                if len(cand.description) > 60
                else cand.description
            )
            code = SafeConvert.column_value(cand.apu, "CODIGO_APU", "N/A")

            if cand.similarity is not None:
                self.info(
                    f"    {i}. Sim: {cand.similarity:.3f} "
                    f"| Código: {code} | Desc: {desc}"
                )
            else:
                self.info(
                    f"    {i}. [{cand.matches}/{keywords_count}] "
                    f"({cand.percentage:.0f}%) - {desc}"
                )


# ============================================================================
# VALIDACIÓN DE DATOS
# ============================================================================

class DataValidator:
    """Validaciones sobre DataFrames de APUs."""

    @staticmethod
    def validate_search_artifacts(
        artifacts: Optional[SearchArtifacts], log: EstimationLog
    ) -> bool:
        """Verifica que los artefactos de búsqueda estén completos."""
        if artifacts is None:
            log.error("SearchArtifacts es None")
            return False

        if not artifacts.is_complete():
            log.error("SearchArtifacts incompleto (modelo, índice o mapa faltante)")
            return False

        return True

    @staticmethod
    def validate_dataframe(
        df: Any,
        required_columns: Sequence[str],
        name: str,
        log: EstimationLog,
    ) -> Optional[pd.DataFrame]:
        """
        Valida que *df* sea un DataFrame no vacío con las columnas requeridas.

        Returns
        -------
        pd.DataFrame | None
            El DataFrame validado, o ``None`` si falla.
        """
        if not isinstance(df, pd.DataFrame):
            log.error(f"{name} no es DataFrame (tipo: {type(df).__name__})")
            return None

        if df.empty:
            log.warn(f"{name} está vacío")
            return None

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            log.error(f"{name}: columnas faltantes {missing}")
            return None

        return df

    @staticmethod
    def assess_quality(
        df: pd.DataFrame,
        log: EstimationLog,
    ) -> DataQualityMetrics:
        """
        Evalúa la calidad de un DataFrame de APUs de forma vectorizada.

        Evita ``iterrows()`` para rendimiento sobre datasets grandes.
        """
        metrics = DataQualityMetrics()
        metrics.total_records = len(df)

        if metrics.total_records == 0:
            return metrics

        # Vectorizado: códigos faltantes
        if "CODIGO_APU" in df.columns:
            code_series = df["CODIGO_APU"].astype(str).str.strip()
            metrics.missing_codes = int((code_series == "").sum() + df["CODIGO_APU"].isna().sum())
        else:
            metrics.missing_codes = metrics.total_records

        # Vectorizado: descripciones faltantes o muy cortas
        if "DESC_NORMALIZED" in df.columns:
            desc_series = df["DESC_NORMALIZED"].astype(str).str.strip()
            short_mask = desc_series.str.len() < MIN_DESCRIPTION_LENGTH
            metrics.missing_descriptions = int(
                short_mask.sum() + df["DESC_NORMALIZED"].isna().sum()
            )
        else:
            metrics.missing_descriptions = metrics.total_records

        metrics.valid_records = metrics.total_records - max(
            metrics.missing_codes, metrics.missing_descriptions
        )
        metrics.valid_records = max(metrics.valid_records, 0)

        metrics.quality_score = (
            metrics.valid_records / metrics.total_records
            if metrics.total_records > 0
            else 0.0
        )

        return metrics

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        required: Sequence[str],
        *,
        strict: bool = False,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Verifica presencia de columnas.

        Returns
        -------
        tuple[bool, list[str], list[str]]
            ``(es_válido, columnas_presentes, columnas_faltantes)``
        """
        if not isinstance(df, pd.DataFrame):
            return False, [], list(required)

        present = [c for c in required if c in df.columns]
        missing = [c for c in required if c not in df.columns]
        ok = (len(missing) == 0) if strict else True
        return ok, present, missing


# ============================================================================
# MOTOR DE BÚSQUEDA
# ============================================================================

class SearchEngine:
    """
    Encapsula las estrategias de búsqueda: keyword y semántica (FAISS).

    Separa la lógica de búsqueda del cálculo de estimación para
    permitir testing independiente y composición.
    """

    def __init__(
        self, artifacts: Optional[SearchArtifacts] = None
    ) -> None:
        self._artifacts = artifacts

    @property
    def has_semantic_capability(self) -> bool:
        return self._artifacts is not None and self._artifacts.is_complete()

    # ------------------------------------------------------------------
    # Keyword Match
    # ------------------------------------------------------------------

    @staticmethod
    def _score_keywords(
        desc_words: Set[str], keywords: List[str]
    ) -> Tuple[int, float]:
        """Cuenta coincidencias de keywords en un conjunto de palabras."""
        if not desc_words or not keywords:
            return 0, 0.0
        matches = sum(1 for k in keywords if k in desc_words)
        pct = (matches / len(keywords)) * 100.0
        return matches, pct

    def find_keyword_match(
        self,
        df_pool: pd.DataFrame,
        keywords: List[str],
        log: EstimationLog,
        *,
        strict: bool = False,
        min_match_percentage: float = DEFAULT_MIN_MATCH_PERCENTAGE,
        match_mode: MatchMode = MatchMode.WORDS,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
        """
        Busca el mejor APU por coincidencia de palabras clave.

        Parameters
        ----------
        df_pool : pd.DataFrame
            Pool filtrado de APUs candidatos.
        keywords : list[str]
            Palabras clave normalizadas.
        log : EstimationLog
        strict : bool
            Si ``True``, solo acepta match 100%.
        min_match_percentage : float
            Porcentaje mínimo para match flexible.
        match_mode : MatchMode
            Estrategia de comparación.

        Returns
        -------
        tuple[pd.Series | None, DerivationDetails | None]
        """
        if df_pool.empty:
            log.warn("Pool vacío para búsqueda por keywords.")
            return None, None

        # Preparar keywords
        kw_clean = [
            k.strip().lower()
            for k in keywords
            if isinstance(k, str) and len(k.strip()) >= 2
        ]
        if len(kw_clean) > MAX_KEYWORDS_COUNT:
            log.warn(
                f"Demasiadas keywords ({len(kw_clean)}), "
                f"truncando a {MAX_KEYWORDS_COUNT}"
            )
            kw_clean = kw_clean[:MAX_KEYWORDS_COUNT]

        if not kw_clean:
            log.warn("Sin keywords válidas tras limpieza.")
            return None, None

        # Asegurar columna DESC_NORMALIZED
        if "DESC_NORMALIZED" not in df_pool.columns:
            if "original_description" in df_pool.columns:
                df_pool = df_pool.copy()
                df_pool["DESC_NORMALIZED"] = (
                    df_pool["original_description"]
                    .fillna("")
                    .apply(lambda x: normalize_text(str(x)))
                )
            else:
                log.error("Columna 'DESC_NORMALIZED' no encontrada")
                return None, None

        # Resolver modo
        if isinstance(match_mode, str):
            try:
                match_mode = MatchMode(match_mode)
            except ValueError:
                log.warn(
                    f"Modo '{match_mode}' no válido, usando WORDS"
                )
                match_mode = MatchMode.WORDS

        log.info(f"  🔍 Keywords: '{' '.join(kw_clean)}'")
        log.info(f"  📊 Pool: {len(df_pool)} APUs | Modo: {match_mode.value}")

        candidates: List[MatchCandidate] = []

        for _, apu_row in df_pool.iterrows():
            desc = str(apu_row.get("DESC_NORMALIZED", "")).strip().lower()
            if len(desc) < MIN_DESCRIPTION_LENGTH:
                continue

            matches, pct, method = 0, 0.0, "KEYWORD"

            if match_mode == MatchMode.WORDS:
                desc_words = set(desc.split())
                matches, pct = self._score_keywords(desc_words, kw_clean)

            elif match_mode == MatchMode.SUBSTRING:
                full_query = " ".join(kw_clean)
                if full_query in desc:
                    matches = len(kw_clean)
                    pct = 100.0
                    method = "EXACT_SUBSTRING"
                else:
                    matches = sum(1 for kw in kw_clean if kw in desc)
                    pct = (matches / len(kw_clean)) * 100.0 if kw_clean else 0.0
                    method = "PARTIAL_SUBSTRING"

            if matches == 0:
                continue

            orig_desc = SafeConvert.column_value(
                apu_row, "original_description", "Sin descripción"
            )
            details = DerivationDetails(
                match_method=method,
                confidence_score=pct / 100.0,
                source="Histórico",
                reasoning=f"Match: {pct:.0f}% ({matches} palabras)",
            )
            candidates.append(
                MatchCandidate(apu_row, orig_desc, matches, pct, details=details)
            )

            # Early exit en modo estricto
            if strict and pct == 100.0:
                log.info("  ⚡ Early exit: Match perfecto")
                log.log_top_candidates([candidates[-1]], 1, len(kw_clean))
                return apu_row, details

            # Limitar candidatos para evitar explosión de memoria
            if len(candidates) >= MAX_CANDIDATES_TO_TRACK:
                break

        # Ordenar por (porcentaje, matches) descendente
        candidates.sort(key=lambda c: (c.percentage, c.matches), reverse=True)
        log.log_top_candidates(candidates, 3, len(kw_clean))

        if not candidates:
            log.info("  ❌ Sin candidatos encontrados")
            return None, None

        best = candidates[0]

        if strict:
            log.info(
                f"  ❌ Sin match estricto. Mejor: {best.percentage:.0f}%"
            )
            return None, None

        if best.percentage >= min_match_percentage:
            log.success(
                f"Match flexible: {best.percentage:.0f}% "
                f"≥ {min_match_percentage:.0f}%"
            )
            return best.apu, best.details

        log.info(
            f"  ❌ Mejor: {best.percentage:.0f}% "
            f"< umbral {min_match_percentage:.0f}%"
        )
        return None, None

    # ------------------------------------------------------------------
    # Semantic Match (FAISS)
    # ------------------------------------------------------------------

    def find_semantic_match(
        self,
        df_pool: pd.DataFrame,
        query_text: str,
        log: EstimationLog,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
        """
        Busca el mejor APU por similitud semántica (embeddings + FAISS).

        Parameters
        ----------
        df_pool : pd.DataFrame
            Pool filtrado con columna ``CODIGO_APU``.
        query_text : str
            Descripción de consulta en lenguaje natural.
        log : EstimationLog
        min_similarity : float
            Umbral mínimo de similitud coseno.
        top_k : int
            Número de vecinos a consultar en FAISS.

        Returns
        -------
        tuple[pd.Series | None, DerivationDetails | None]
        """
        if not self.has_semantic_capability:
            log.error("Búsqueda semántica no disponible")
            return None, None

        artifacts = self._artifacts
        assert artifacts is not None  # Satisfecho por has_semantic_capability

        validated = DataValidator.validate_dataframe(
            df_pool, ["CODIGO_APU"], "df_pool (semántico)", log
        )
        if validated is None:
            return None, None

        query_clean = (query_text or "").strip()
        if len(query_clean) < MIN_DESCRIPTION_LENGTH:
            log.warn(f"Consulta muy corta: '{query_clean}'")
            return None, None

        log.info(
            f"  🧠 Búsqueda Semántica: '{query_clean[:50]}...'"
        )
        log.info(f"  📊 Pool: {len(validated)} APUs")
        log.info(
            f"  ⚙️ Umbral: {min_similarity:.2f} | Top-K: {top_k}"
        )

        try:
            embedding = artifacts.model.encode(
                [query_clean],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if embedding is None or embedding.size == 0:
                log.error("Embedding vacío generado")
                return None, None

            effective_k = min(top_k, artifacts.faiss_index.ntotal, MAX_FAISS_TOP_K)
            distances, indices = artifacts.faiss_index.search(
                embedding.astype(np.float32), k=effective_k
            )

            # Construir set de códigos del pool para filtrado O(1)
            pool_codes: Set[str] = set(
                validated["CODIGO_APU"].astype(str).str.strip()
            )

            candidates: List[MatchCandidate] = []

            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                raw_sim = float(distances[0][i])

                # FAISS con IP normalizado devuelve coseno ∈ [-1, 1].
                # Clamp y log si está fuera de [0, 1].
                if raw_sim < 0.0 or raw_sim > 1.0:
                    logger.debug(
                        "Similitud FAISS fuera de [0,1]: %.4f (idx=%d). "
                        "Verifique que el índice use Inner Product con "
                        "embeddings normalizados.",
                        raw_sim,
                        idx,
                    )
                    sim = max(0.0, min(raw_sim, 1.0))
                else:
                    sim = raw_sim

                code = artifacts.id_map.get(str(idx))
                if not code:
                    continue
                code = str(code).strip()

                if code not in pool_codes:
                    continue

                matches_df = validated[
                    validated["CODIGO_APU"].astype(str).str.strip() == code
                ]
                if matches_df.empty:
                    continue

                apu_row = matches_df.iloc[0]
                orig_desc = SafeConvert.column_value(
                    apu_row, "original_description", "Sin descripción"
                )

                details = DerivationDetails(
                    match_method="SEMANTIC",
                    confidence_score=sim,
                    source="Vector Database",
                    reasoning=f"Similitud semántica: {sim:.3f}",
                )
                candidates.append(
                    MatchCandidate(
                        apu_row, orig_desc, 0, 0.0,
                        similarity=sim, details=details,
                    )
                )

            if not candidates:
                log.warn("Ningún resultado FAISS coincide con el pool filtrado")
                return None, None

            candidates.sort(
                key=lambda c: c.similarity if c.similarity else 0.0,
                reverse=True,
            )
            log.log_top_candidates(candidates, 3)

            best = candidates[0]
            if best.similarity is not None and best.similarity >= min_similarity:
                log.success(
                    f"Coincidencia semántica: {best.similarity:.3f}"
                )
                return best.apu, best.details

            log.info(
                f"  ❌ Mejor similitud: "
                f"{best.similarity:.3f} < {min_similarity:.2f}"
            )

        except Exception as exc:
            logger.error("Error en búsqueda semántica: %s", exc, exc_info=True)
            log.error(f"Error en búsqueda semántica: {type(exc).__name__}: {exc}")

        return None, None

    # ------------------------------------------------------------------
    # Búsqueda con fallback
    # ------------------------------------------------------------------

    def find_best_match(
        self,
        df_pool: pd.DataFrame,
        query_text: str,
        keywords: List[str],
        log: EstimationLog,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_match_pct: float = DEFAULT_MIN_MATCH_PERCENTAGE,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
        """
        Búsqueda con cascada: semántica → keyword.

        Returns
        -------
        tuple[pd.Series | None, DerivationDetails | None]
        """
        # Intento 1: Semántica
        apu, details = self.find_semantic_match(
            df_pool, query_text, log, min_similarity=min_similarity
        )
        if apu is not None:
            return apu, details

        # Intento 2: Keywords
        log.subsection("🔄 Fallback: Búsqueda por palabras clave...")
        return self.find_keyword_match(
            df_pool, keywords, log, min_match_percentage=min_match_pct
        )


# ============================================================================
# CALCULADOR DE COSTOS
# ============================================================================

class CostCalculator:
    """
    Ensambla los componentes de costo de una estimación.

    Responsabilidades:
    - Extraer factores de configuración.
    - Buscar suministro, cuadrilla y tarea.
    - Calcular rendimiento desde detalle.
    - Aplicar factores de zona, izaje y seguridad.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        search_engine: SearchEngine,
    ) -> None:
        self._config = config
        self._search = search_engine

    # ------------------------------------------------------------------
    # Extracción de configuración
    # ------------------------------------------------------------------

    def _extract_factors(self, params: Dict[str, Any]) -> EstimationFactors:
        """Extrae factores de ajuste del config y parámetros."""
        rules = self._config.get("estimator_rules", {})
        if not isinstance(rules, dict):
            rules = {}

        zona = str(params.get("zona", DEFAULT_ZONA)).strip()
        izaje = str(params.get("izaje", DEFAULT_IZAJE)).strip()
        seguridad = str(params.get("seguridad", DEFAULT_SEGURIDAD)).strip()

        return EstimationFactors(
            factor_zona=SafeConvert.to_float(
                rules.get("factores_zona", {}).get(zona, 1.0), 1.0
            ),
            costo_izaje=SafeConvert.to_float(
                rules.get("costo_adicional_izaje", {}).get(izaje, 0.0), 0.0
            ),
            factor_seguridad=SafeConvert.to_float(
                rules.get("factor_seguridad", {}).get(seguridad, 1.0), 1.0
            ),
        )

    def _extract_thresholds(self) -> Dict[str, float]:
        """Extrae umbrales de búsqueda del config."""
        raw = self._config.get("estimator_thresholds", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "min_sim_suministro": SafeConvert.to_float(
                raw.get("min_semantic_similarity_suministro", 0.3), 0.3
            ),
            "min_sim_tarea": SafeConvert.to_float(
                raw.get("min_semantic_similarity_tarea", 0.4), 0.4
            ),
            "min_kw_cuadrilla": SafeConvert.to_float(
                raw.get("min_keyword_match_percentage_cuadrilla", 50.0), 50.0
            ),
        }

    # ------------------------------------------------------------------
    # Filtrado de pools
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_pool(
        df: pd.DataFrame,
        column: str,
        values: Sequence[str],
        log: EstimationLog,
        label: str,
    ) -> pd.DataFrame:
        """Filtra un DataFrame por valores en una columna."""
        if column not in df.columns:
            log.warn(f"Columna '{column}' no encontrada para filtrar {label}")
            return df.copy()
        mask = df[column].isin(values)
        filtered = df[mask]
        if filtered.empty:
            log.warn(f"Pool {label} vacío tras filtrar por {column}={values}")
            return df.copy()
        return filtered.copy()

    @staticmethod
    def _filter_by_unit(
        df: pd.DataFrame,
        unit: str,
        log: EstimationLog,
        label: str,
    ) -> pd.DataFrame:
        """Filtra por unidad de medida."""
        if "UNIDAD" not in df.columns:
            return df.copy()
        mask = df["UNIDAD"].astype(str).str.upper().str.strip() == unit.upper()
        filtered = df[mask]
        if filtered.empty:
            log.warn(f"Pool {label} vacío tras filtrar UNIDAD={unit}")
            return df.copy()
        return filtered.copy()

    # ------------------------------------------------------------------
    # Rendimiento desde detalle
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_rendimiento_from_detail(
        apu_code: str,
        data_store: Dict[str, Any],
        log: EstimationLog,
    ) -> float:
        """
        Calcula rendimiento como inverso del tiempo total de mano de obra.

        .. math::
            \\text{rendimiento} = \\frac{1}{\\sum t_i}

        donde :math:`t_i` son las cantidades de mano de obra del APU.
        """
        log.subsection("📊 Calculando rendimiento desde detalle...")

        details_raw = data_store.get("apus_detail", [])
        if not details_raw:
            log.warn("No hay datos de detalle disponibles")
            return 0.0

        try:
            df = pd.DataFrame(details_raw)

            required = ["CANTIDAD_APU", "TIPO_INSUMO", "CODIGO_APU"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                log.warn(f"Columnas faltantes en detalle: {missing}")
                return 0.0

            mask = (
                (df["CODIGO_APU"].astype(str).str.strip() == str(apu_code).strip())
                & (df["TIPO_INSUMO"] == TipoRecurso.MANO_OBRA.value)
            )
            mo_rows = df[mask]

            if mo_rows.empty:
                log.warn("No se encontró mano de obra para este APU")
                return 0.0

            total_time = mo_rows["CANTIDAD_APU"].apply(
                lambda x: SafeConvert.to_float(x, 0.0, min_value=0.0)
            ).sum()

            if total_time < MIN_RENDIMIENTO_SAFE:
                log.warn(
                    f"Tiempo total de MO ≈ 0 ({total_time:.6f})"
                )
                return 0.0

            rend = 1.0 / total_time

            # Proteger contra rendimientos absurdos
            if rend > 1e6:
                log.warn(f"Rendimiento anormalmente alto: {rend:.2f}")
                return 0.0

            log.info(f"  ⏱️ Rendimiento: {rend:.4f} un/día")
            return rend

        except Exception as exc:
            log.error(f"Error calculando rendimiento: {exc}")
            logger.error("Error rendimiento: %s", exc, exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Promedio histórico
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_historical_average(
        df_apus: pd.DataFrame,
        keywords: List[str],
        log: EstimationLog,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails], float]:
        """
        Fallback: calcula rendimiento promedio de APUs similares.

        Filtra por keywords significativas (len > 3) y promedia
        ``RENDIMIENTO_DIA``.
        """
        try:
            sig_kw = [k for k in keywords if len(k) > 3]
            if not sig_kw:
                return None, None, 0.0

            if "DESC_NORMALIZED" not in df_apus.columns:
                return None, None, 0.0

            regex = "|".join(sig_kw)
            mask = df_apus["DESC_NORMALIZED"].str.contains(
                regex, case=False, regex=True, na=False
            )
            matched = df_apus[mask]

            if "RENDIMIENTO_DIA" not in matched.columns:
                return None, None, 0.0

            # Filtrar rendimientos válidos (vectorizado)
            rend_series = matched["RENDIMIENTO_DIA"].apply(
                lambda x: SafeConvert.to_float(x, 0.0)
            )
            valid_mask = rend_series > 0
            valid_rend = rend_series[valid_mask]

            if valid_rend.empty:
                return None, None, 0.0

            avg_rend = float(valid_rend.mean())

            apu_synth = pd.Series({
                "CODIGO_APU": "EST-AVG",
                "original_description": f"Promedio ({len(valid_rend)} items)",
                "RENDIMIENTO_DIA": avg_rend,
                "EQUIPO": 0.0,
            })

            confidence = min(0.5, len(valid_rend) / 10.0)
            details = DerivationDetails(
                match_method="HISTORICAL_AVERAGE",
                confidence_score=confidence,
                source="Promedio Histórico",
                reasoning=f"Promedio de {len(valid_rend)} items con rend > 0",
            )

            log.info(
                f"  📈 Promedio histórico: {avg_rend:.4f} "
                f"({len(valid_rend)} muestras)"
            )
            return apu_synth, details, avg_rend

        except Exception as exc:
            logger.error("Error histórico: %s", exc, exc_info=True)
            return None, None, 0.0

    # ------------------------------------------------------------------
    # Cálculo principal
    # ------------------------------------------------------------------

    def calculate(
        self,
        params: Dict[str, Any],
        data_store: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ejecuta la estimación completa de costos.

        Pipeline:
        1. Validar datos de entrada.
        2. Preparar parámetros y keywords.
        3. Buscar suministro (semántico → keyword).
        4. Buscar cuadrilla (keyword).
        5. Buscar tarea / rendimiento (semántico → keyword → promedio).
        6. Ensamblar costos con factores de ajuste.
        7. Análisis financiero opcional (Monte Carlo).

        Returns
        -------
        dict[str, Any]
            Resultado con costos, códigos, factores y log.
        """
        log = EstimationLog("🕵️ ESTIMADOR HÍBRIDO INICIADO (TACTICS STRATUM)")
        derivation: Dict[str, Optional[Dict[str, Any]]] = {
            "suministro": None,
            "tarea": None,
            "cuadrilla": None,
        }
        components = EstimationComponents()

        # ---- 1. Validar datos de APU ----
        processed_apus = data_store.get("processed_apus", [])
        if not processed_apus:
            return self._error_result(
                "No hay datos de APU procesados (processed_apus vacío)", log
            )

        try:
            df_apus = (
                pd.DataFrame(processed_apus)
                if isinstance(processed_apus, list)
                else processed_apus.copy()
            )
        except Exception as exc:
            return self._error_result(f"Datos de APU inválidos: {exc}", log)

        # ---- 2. Parámetros ----
        material = str(params.get("material", "")).strip().upper()
        if not material:
            return self._error_result(
                "El parámetro 'material' es obligatorio", log
            )

        cuadrilla_str = str(params.get("cuadrilla", "0")).strip()

        # Mapeo y keywords
        param_map = self._config.get("param_map", {})
        if not isinstance(param_map, dict):
            param_map = {}
        material_mapped = param_map.get("material", {}).get(material, material)
        keywords = [
            k for k in normalize_text(material_mapped).split() if len(k) >= 2
        ]

        thresholds = self._extract_thresholds()
        factors = self._extract_factors(params)

        # ---- 3. Búsqueda Suministro ----
        log.section("Búsqueda Suministro")
        apu_sum, det_sum = self._search_suministro(
            df_apus, material_mapped, keywords, log,
            min_sim=thresholds["min_sim_suministro"],
        )
        if apu_sum is not None:
            components.valor_suministro = SafeConvert.to_float(
                apu_sum.get("VALOR_SUMINISTRO_UN", 0.0)
            )
            components.codigo_suministro = str(apu_sum.get("CODIGO_APU", "N/A"))
        if det_sum:
            derivation["suministro"] = self._details_to_dict(det_sum)

        # ---- 4. Búsqueda Cuadrilla ----
        log.section("Búsqueda Cuadrilla")
        costo_dia_cuadrilla = 0.0
        det_cuad: Optional[DerivationDetails] = None

        if SafeConvert.to_int(cuadrilla_str) > 0:
            apu_cuad, det_cuad = self._search_cuadrilla(
                df_apus, cuadrilla_str, log,
                min_kw_pct=thresholds["min_kw_cuadrilla"],
            )
            if apu_cuad is not None:
                costo_dia_cuadrilla = SafeConvert.to_float(
                    apu_cuad.get("VALOR_CONSTRUCCION_UN", 0.0)
                )
                components.codigo_cuadrilla = str(
                    apu_cuad.get("CODIGO_APU", "N/A")
                )
        if det_cuad:
            derivation["cuadrilla"] = self._details_to_dict(det_cuad)

        # ---- 5. Búsqueda Tarea (Rendimiento) ----
        log.section("Búsqueda Tarea")
        apu_tarea, det_tarea, rendimiento = self._search_tarea(
            df_apus, material_mapped, keywords, data_store, log,
            min_sim=thresholds["min_sim_tarea"],
        )
        if apu_tarea is not None:
            components.costo_equipo = SafeConvert.to_float(
                apu_tarea.get("EQUIPO", 0.0)
            )
            components.codigo_tarea = str(apu_tarea.get("CODIGO_APU", "N/A"))
        if det_tarea:
            derivation["tarea"] = self._details_to_dict(det_tarea)
        components.rendimiento = rendimiento

        # ---- 6. Ensamblaje de costos ----
        components.costo_mano_obra = self._calculate_labor_cost(
            costo_dia_cuadrilla, rendimiento, factors, log
        )

        valor_instalacion = (
            (components.costo_mano_obra + components.costo_equipo)
            * factors.factor_zona
            + factors.costo_izaje
        )
        valor_construccion = components.valor_suministro + valor_instalacion

        # ---- 7. Análisis financiero opcional ----
        financial = self._run_financial_analysis(
            components, data_store, valor_construccion, log
        )

        return {
            "valor_suministro": round(components.valor_suministro, 2),
            "valor_instalacion": round(valor_instalacion, 2),
            "valor_construccion": round(valor_construccion, 2),
            "rendimiento_m2_por_dia": round(rendimiento, 4),
            "costo_equipo": round(components.costo_equipo, 2),
            "costo_mano_obra": round(components.costo_mano_obra, 2),
            "apu_suministro_codigo": components.codigo_suministro,
            "apu_tarea_codigo": components.codigo_tarea,
            "apu_cuadrilla_codigo": components.codigo_cuadrilla,
            "factores_aplicados": {
                "zona": factors.factor_zona,
                "seguridad": factors.factor_seguridad,
                "izaje": factors.costo_izaje,
            },
            "financial_analysis": financial,
            "derivation_details": derivation,
            "log": log.render(),
        }

    # ------------------------------------------------------------------
    # Sub-búsquedas
    # ------------------------------------------------------------------

    def _search_suministro(
        self,
        df_apus: pd.DataFrame,
        material_mapped: str,
        keywords: List[str],
        log: EstimationLog,
        *,
        min_sim: float,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
        """Busca APU de suministro con cascada semántica → keyword."""
        supply_types = [
            TipoAPU.SUMINISTRO.value,
            TipoAPU.SUMINISTRO_PREFABRICADO.value,
        ]
        pool = self._filter_pool(
            df_apus, "tipo_apu", supply_types, log, "suministro"
        )
        if pool.empty:
            return None, None

        return self._search.find_best_match(
            pool, material_mapped, keywords, log, min_similarity=min_sim
        )

    def _search_cuadrilla(
        self,
        df_apus: pd.DataFrame,
        cuadrilla_id: str,
        log: EstimationLog,
        *,
        min_kw_pct: float,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails]]:
        """Busca APU de cuadrilla por keyword."""
        pool = self._filter_by_unit(df_apus, "DIA", log, "cuadrilla")
        if pool.empty:
            return None, None

        kw = normalize_text(f"cuadrilla {cuadrilla_id}").split()
        return self._search.find_keyword_match(
            pool, kw, log, min_match_percentage=min_kw_pct
        )

    def _search_tarea(
        self,
        df_apus: pd.DataFrame,
        material_mapped: str,
        keywords: List[str],
        data_store: Dict[str, Any],
        log: EstimationLog,
        *,
        min_sim: float,
    ) -> Tuple[Optional[pd.Series], Optional[DerivationDetails], float]:
        """
        Busca APU de tarea (rendimiento) con cascada:
        semántica → keyword → promedio histórico.

        Returns
        -------
        tuple[pd.Series | None, DerivationDetails | None, float]
            ``(apu, details, rendimiento)``
        """
        pool = self._filter_pool(
            df_apus, "tipo_apu", [TipoAPU.INSTALACION.value], log, "tarea"
        )

        apu, details = None, None
        rendimiento = 0.0

        if not pool.empty:
            apu, details = self._search.find_best_match(
                pool, material_mapped, keywords, log, min_similarity=min_sim
            )

        # Fallback: promedio histórico
        if apu is None:
            log.subsection(
                "⚠️ Tarea no encontrada. Intentando promedio histórico..."
            )
            apu, details, rendimiento = self._calculate_historical_average(
                df_apus, keywords, log
            )

        # Extraer rendimiento del detalle si no se obtuvo
        if apu is not None and rendimiento <= 0:
            code = str(apu.get("CODIGO_APU", ""))
            if code and code != "EST-AVG":
                rendimiento = self._calculate_rendimiento_from_detail(
                    code, data_store, log
                )

        return apu, details, rendimiento

    # ------------------------------------------------------------------
    # Cálculos internos
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_labor_cost(
        costo_dia: float,
        rendimiento: float,
        factors: EstimationFactors,
        log: EstimationLog,
    ) -> float:
        """
        Calcula costo de mano de obra ajustado.

        .. math::
            C_{MO} = \\frac{C_{día}}{R} \\times F_{seguridad}

        Con protección contra rendimiento ≈ 0 y costos excesivos.
        """
        if rendimiento < MIN_RENDIMIENTO_SAFE:
            if costo_dia > 0:
                log.warn(
                    f"Rendimiento ≈ 0 ({rendimiento:.6f}) con "
                    f"costo_dia={costo_dia:.2f}. MO = 0"
                )
            return 0.0

        costo_base = costo_dia / rendimiento

        if costo_base > MAX_COSTO_UNITARIO:
            log.warn(
                f"Costo MO base excesivo: {costo_base:.2f}. "
                f"Clamped a {MAX_COSTO_UNITARIO:.2f}"
            )
            costo_base = MAX_COSTO_UNITARIO

        return costo_base * factors.factor_seguridad

    @staticmethod
    def _run_financial_analysis(
        components: EstimationComponents,
        data_store: Dict[str, Any],
        valor_construccion: float,
        log: EstimationLog,
    ) -> Dict[str, Any]:
        """Ejecuta Monte Carlo si hay insumos disponibles."""
        result: Dict[str, Any] = {}

        codes = [
            c for c in [components.codigo_suministro, components.codigo_tarea]
            if c not in ("N/A", "EST-AVG", "")
        ]
        if not codes:
            return result

        try:
            all_insumos = data_store.get("apus_detail", [])
            insumos = [
                i for i in all_insumos
                if str(i.get("CODIGO_APU", "")) in codes
            ]
            if not insumos:
                return result

            mc = run_monte_carlo_simulation(insumos)
            stats = mc.get("statistics", {})
            result["monte_carlo_mean"] = stats.get("mean", valor_construccion)
            result["monte_carlo_std"] = stats.get("std_dev", 0.0)

        except Exception as exc:
            logger.error("Error en análisis financiero: %s", exc, exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(message: str, log: EstimationLog) -> Dict[str, Any]:
        """Genera un resultado de error estandarizado."""
        log.error(message)
        return {"error": message, "log": log.render()}

    @staticmethod
    def _details_to_dict(
        details: Optional[DerivationDetails],
    ) -> Optional[Dict[str, Any]]:
        """Convierte DerivationDetails a dict si no es None."""
        if details is None:
            return None
        return {
            "match_method": details.match_method,
            "confidence_score": details.confidence_score,
            "source": details.source,
            "reasoning": details.reasoning,
        }


# ============================================================================
# SERVICIO PRINCIPAL
# ============================================================================

class SemanticEstimatorService:
    """
    Fachada que orquesta búsqueda vectorial y estimación de costos.

    Carga los modelos pesados en un hilo background con sincronización
    por ``threading.Event`` para evitar race conditions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._artifacts: Optional[SearchArtifacts] = None
        self._ready_event = threading.Event()
        self._load_error: Optional[str] = None

        # Carga perezosa en hilo daemon
        self._loader = threading.Thread(
            target=self._load_tensor_space,
            name="VectorSpaceLoader",
            daemon=True,
        )
        self._loader.start()

    @property
    def is_ready(self) -> bool:
        """Indica si los artefactos están cargados y disponibles."""
        return self._ready_event.is_set() and self._artifacts is not None

    def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """
        Bloquea hasta que los artefactos estén cargados o expire el timeout.

        Returns
        -------
        bool
            ``True`` si se cargó exitosamente.
        """
        return self._ready_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Carga de modelos
    # ------------------------------------------------------------------

    def _load_tensor_space(self) -> None:
        """
        Carga modelos en memoria aislada (hilo background).

        Usa ``threading.Event`` para señalizar disponibilidad de forma
        thread-safe.
        """
        logger.info(
            "Iniciando carga del Espacio Vectorial (FAISS) en background..."
        )
        try:
            emb_meta = self._config.get("embedding_metadata", {})
            if not isinstance(emb_meta, dict):
                emb_meta = {}
            model_name = emb_meta.get("model_name", "all-MiniLM-L6-v2")

            embeddings_dir = Path(__file__).parent / "embeddings"
            index_path = embeddings_dir / "faiss.index"
            map_path = embeddings_dir / "id_map.json"

            if not index_path.exists() or not map_path.exists():
                self._load_error = "Artefactos FAISS no encontrados"
                logger.warning(
                    "%s. Búsqueda semántica deshabilitada.", self._load_error
                )
                self._ready_event.set()
                return

            model = SentenceTransformer(model_name)
            index = faiss.read_index(str(index_path))

            with open(map_path, "r", encoding="utf-8") as f:
                id_map = json.load(f)

            self._artifacts = SearchArtifacts(
                model=model, faiss_index=index, id_map=id_map
            )
            logger.info(
                "✅ Espacio Vectorial cargado. Dimensión: %d, Vectores: %d",
                index.d,
                index.ntotal,
            )

        except Exception as exc:
            self._load_error = str(exc)
            logger.critical(
                "❌ Fallo al cargar espacio vectorial: %s", exc, exc_info=True
            )
        finally:
            self._ready_event.set()

    # ------------------------------------------------------------------
    # Vectores MIC
    # ------------------------------------------------------------------

    def project_semantic_match(
        self,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        [Vector MIC] Proyecta una descripción humana al insumo formal
        más cercano en el espacio vectorial.

        Parameters
        ----------
        payload : dict
            ``{"query_text": str, "df_pool": list | DataFrame}``
        context : dict
            Puede contener ``"telemetry_context"``.

        Returns
        -------
        dict[str, Any]
        """
        telemetry: Optional[TelemetryContext] = context.get("telemetry_context")
        step_name = "semantic_projection"

        if telemetry:
            telemetry.start_step(step_name)

        if not self.is_ready or self._artifacts is None:
            error_msg = self._load_error or "Asesor Semántico no inicializado"
            return {"success": False, "error": error_msg}

        query_text = payload.get("query_text", "")
        df_pool_data = payload.get("df_pool")

        try:
            df_pool = (
                pd.DataFrame(df_pool_data)
                if isinstance(df_pool_data, list)
                else df_pool_data
            )
            if df_pool is None or (isinstance(df_pool, pd.DataFrame) and df_pool.empty):
                return {"success": False, "error": "Pool vacío"}

            engine = SearchEngine(self._artifacts)
            log = EstimationLog()

            apu, details = engine.find_semantic_match(
                df_pool, query_text, log
            )

            if telemetry:
                telemetry.end_step(step_name, "success")

            if apu is not None:
                return {
                    "success": True,
                    "matched_id": str(apu.get("CODIGO_APU")),
                    "confidence": details.confidence_score if details else 0.0,
                    "details": CostCalculator._details_to_dict(details),
                    "stratum": Stratum.TACTICS.name,
                }

            return {"success": False, "error": "No match found"}

        except Exception as exc:
            if telemetry:
                telemetry.record_error(step_name, str(exc))
            logger.error(
                "Error en proyección semántica: %s", exc, exc_info=True
            )
            return {"success": False, "error": str(exc)}

    def calculate_dynamic_estimate(
        self,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        [Vector MIC] Ejecuta estimación de costos delegada.

        Parameters
        ----------
        payload : dict
            ``{"params": dict, "data_store": dict}``
        context : dict
            Puede contener ``"telemetry_context"``.

        Returns
        -------
        dict[str, Any]
        """
        telemetry: Optional[TelemetryContext] = context.get("telemetry_context")
        step_name = "tactical_estimate"

        try:
            params = payload.get("params", {})
            data_store = payload.get("data_store", {})

            engine = SearchEngine(self._artifacts)
            calculator = CostCalculator(self._config, engine)

            result = calculator.calculate(params=params, data_store=data_store)

            return {
                "success": True,
                "estimate": result,
                "stratum": Stratum.TACTICS.name,
            }

        except Exception as exc:
            logger.error(
                "Fallo en cálculo táctico: %s", exc, exc_info=True
            )
            if telemetry:
                telemetry.record_error(step_name, str(exc))
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Registro MIC
    # ------------------------------------------------------------------

    def register_in_mic(self, mic: MICRegistry) -> None:
        """Registra las capacidades del agente en la MIC."""
        mic.register_vector(
            service_name="semantic_match",
            stratum=Stratum.TACTICS,
            handler=self.project_semantic_match,
        )
        mic.register_vector(
            service_name="tactical_estimate",
            stratum=Stratum.TACTICS,
            handler=self.calculate_dynamic_estimate,
        )
        logger.info("✅ Vectores Tácticos Semánticos registrados en la MIC.")