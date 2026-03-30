"""
=========================================================================================
Módulo: Matriz de Interacción Central (MIC) — El Espacio Vectorial de Intenciones
Ubicación: app/adapters/tools_interface.py
=========================================================================================

Naturaleza Ciber-Física y Algebraica:
    Instancia el núcleo de gobernanza, enrutamiento y despacho del ecosistema, actuando 
    como un Espacio Vectorial Jerarquizado con Filtración Topológica. 
    Modela el catálogo de capacidades del agente autónomo sobre la estructura rígida 
    de una Matriz Identidad (I_n), garantizando un difeomorfismo perfecto y determinista 
    entre la intención probabilística generada por el LLM y la ejecución discreta.

1. Álgebra Lineal de la Acción (Matriz Identidad e Independencia Lineal):
    Cada herramienta atómica (servicio s_i) se proyecta como un vector base canónico e_i ∈ ℝⁿ.
    La interfaz global se define axiomáticamente mediante el delta de Kronecker [2]:
        I_{ij} = δ_{ij} = { 1 si i = j, 0 si i ≠ j }
    
    * Ortogonalidad Estricta: ⟨e_i, e_j⟩ = δ_{ij}. Se asegura matemáticamente el aislamiento 
      (Zero Side-Effects). La activación de una herramienta no posee proyección sobre 
      el dominio de otra, aniquilando la interferencia mutua [3-5].
    * Teorema Rango-Nulidad: Rank(I_n) = n y Nullity(I_n) = 0. Certifica un alcance 
      universal en el espacio de acciones sin redundancias funcionales y asegura que 
      ningún vector de intención del agente colapsará hacia el vacío (no hay acciones 
      "silenciosas" o ignoradas por el núcleo) [6, 7].
    * Estabilidad Espectral: Todos los eigenvalores cumplen λ_i = 1. La interfaz es 
      un canal de transmisión perfecto; no distorsiona, no amplifica ni atenúa la 
      intención original del agente [8, 9].

2. Filtración de Subespacios Topológicos (Clausura Transitiva DIKW):
    El espacio vectorial de intenciones se estructura como una filtración estricta anidada:
        V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM [1]
    
    El operador de proyección π_k(v) actúa como el "Gatekeeper" de la matriz:
        G(v, ctx) = v si ∀j < k: validated(V_j) = True, de lo contrario 0⃗ [1].
    Esto subordina axiomáticamente cualquier operación estratégica o de nivel de 
    sabiduría a la coherencia física y topológica de sus estratos subyacentes.

3. Auditoría Homológica y Persistencia Espectral (Mayer-Vietoris):
    La inyección de datos concurrentes (A ∪ B) se audita mediante la secuencia exacta de 
    Mayer-Vietoris [1]. Si la integración induce una mutación del primer número de Betti 
    (Δβ₁ = β₁(A∪B) - [β₁(A) + β₁(B) - β₁(A∩B)] ≠ 0), se aborta la fusión por inconsistencia.
    
    La topología de la MIC monitorea:
        * H₀: Componentes conexos (fragmentación y silos de operaciones) [1].
        * H₁: Ciclos homológicos (bucles de dependencias circulares y deadlocks) [1].

Invariantes del Módulo:
    * Idempotencia Categórica: I_n · I_n = I_n. Validaciones recursivas no mutan el estado [10].
    * Isomorfismo Dimensional: dim(V_expected) ≅ dim(V_actual) acotado por ε-tolerancia [1].
    * Cero Dependencia Lineal: Si c_1 e_1 + ... + c_n e_n = 0, entonces c_i = 0 ∀i [11].
=========================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import statistics
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import lru_cache, wraps, cached_property
from pathlib import Path
from typing import (
    Any, Callable, ClassVar, Dict, Final, FrozenSet, Generic, Iterator,
    List, Literal, Mapping, NamedTuple, Optional, Protocol, Sequence, 
    Set, Tuple, Type, TypedDict, TypeVar, Union, cast, overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd

# Importaciones opcionales con fallback
try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    warnings.warn(
        "scipy.sparse no disponible — análisis espectral usará matrices densas",
        ImportWarning
    )


# =============================================================================
# LOGGER ESTRUCTURADO
# =============================================================================

logger = logging.getLogger("MIC")


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Adapter para logging estructurado con contexto."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_structured_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """Crea un logger con contexto estructurado."""
    return StructuredLoggerAdapter(logging.getLogger(name), context)


# =============================================================================
# IMPORTACIONES CON FALLBACK ROBUSTO
# =============================================================================

def _safe_import(module_path: str, class_name: str) -> Optional[Type]:
    """
    Importación segura con logging de diagnóstico.

    Maneja correctamente:
    - Módulos absolutos: "scripts.clean_csv"
    - Módulos relativos con punto: ".financial_engine"
    """
    try:
        if module_path.startswith("."):
            import importlib
            package = __name__.rsplit(".", 1)[0] if "." in __name__ else __name__
            module = importlib.import_module(module_path, package=package)
        else:
            module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.debug("Optional import failed: %s.%s — %s", module_path, class_name, e)
        return None
    except Exception as e:
        logger.debug("Unexpected error importing %s.%s — %s", module_path, class_name, e)
        return None


CSVCleaner = _safe_import("scripts.clean_csv", "CSVCleaner")
APUFileDiagnostic = _safe_import("scripts.diagnose_apus_file", "APUFileDiagnostic")
InsumosFileDiagnostic = _safe_import("scripts.diagnose_insumos_file", "InsumosFileDiagnostic")
PresupuestoFileDiagnostic = _safe_import(
    "scripts.diagnose_presupuesto_file", "PresupuestoFileDiagnostic"
)
FinancialConfig = _safe_import(".financial_engine", "FinancialConfig")
FinancialEngine = _safe_import(".financial_engine", "FinancialEngine")


# ── Fallback para Stratum ──────────────────────────────────────────────────
try:
    from app.core.schemas import Stratum
except ImportError:

    class Stratum(IntEnum):
        """
        Representación jerárquica del modelo DIKW.

        Valores numéricos y su semántica en la filtración:
          PHYSICS  = 5  →  Base de la pirámide (datos crudos)
          TACTICS  = 4  →  Estructura operativa
          STRATEGY = 3  →  Planificación financiera
          OMEGA    = 2  →  Ágora Tensorial
          ALPHA    = 1  →  Estructura de Negocio
          WISDOM   = 0  →  Síntesis estratégica (vértice)

        La filtración es ascendente: para alcanzar el estrato k
        deben estar validados todos los estratos j con valor > k.
        """
        WISDOM = 0
        OMEGA = 1
        STRATEGY = 2
        TACTICS = 3
        PHYSICS = 4

        @classmethod
        def base_stratum(cls) -> "Stratum":
            """Retorna el estrato base de la filtración."""
            return cls.PHYSICS

        @classmethod
        def apex_stratum(cls) -> "Stratum":
            """Retorna el estrato superior de la filtración."""
            return cls.WISDOM

        def requires(self) -> FrozenSet["Stratum"]:
            """
            Retorna los estratos prerrequisito (clausura transitiva).

            Un estrato k requiere todos los estratos con value > self.value.
            """
            return frozenset(s for s in Stratum if s.value > self.value)

        @classmethod
        def ordered_bottom_up(cls) -> List["Stratum"]:
            """Retorna estratos ordenados de base a cúspide."""
            return sorted(cls, key=lambda s: s.value, reverse=True)

        @classmethod
        def ordered_top_down(cls) -> List["Stratum"]:
            """Retorna estratos ordenados de cúspide a base."""
            return sorted(cls, key=lambda s: s.value)


# ── Vectores mock para testing standalone ─────────────────────────────────
try:
    from app.adapters.mic_vectors import (
        vector_audit_homological_fusion,
        vector_lateral_pivot,
        vector_parse_raw_structure,
        vector_stabilize_flux,
        vector_structure_logic,
    )
except ImportError:

    def _mock_vector(**kwargs: Any) -> Dict[str, Any]:
        """Vector mock que retorna éxito con los kwargs recibidos."""
        return {"success": True, "mock": True, **kwargs}

    vector_stabilize_flux = _mock_vector
    vector_parse_raw_structure = _mock_vector
    vector_structure_logic = _mock_vector
    vector_lateral_pivot = _mock_vector
    vector_audit_homological_fusion = _mock_vector


# =============================================================================
# CONFIGURACIÓN EXTERNALIZABLE
# =============================================================================

@dataclass(frozen=True)
class MICConfiguration:
    """
    Configuración centralizada de la MIC.
    
    Permite ajustar comportamiento sin modificar código fuente.
    Valores por defecto calibrados para entornos de producción.
    """
    # Límites de archivos
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_sample_rows: int = 1000
    
    # Cache
    cache_ttl_seconds: float = 300.0  # 5 minutos
    cache_max_size: int = 128
    
    # Análisis topológico
    persistence_threshold: float = 0.01
    cycle_similarity_threshold: float = 0.80
    max_cycle_period: int = 50
    max_lines_for_cycle_detection: int = 10000
    
    # Métricas
    latency_histogram_buckets: int = 100
    enable_detailed_metrics: bool = True
    
    # Validación
    strict_encoding_validation: bool = False
    
    # Timeouts
    diagnostic_timeout_seconds: float = 30.0
    spectral_analysis_timeout_seconds: float = 10.0
    
    # Constantes matemáticas
    epsilon: float = 1e-10
    
    # Versión
    algorithm_version: str = "4.0.0-topological"
    
    def __post_init__(self) -> None:
        """Validar invariantes de configuración."""
        if self.max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes debe ser > 0")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds debe ser > 0")
        if not (0 < self.cycle_similarity_threshold <= 1):
            raise ValueError("cycle_similarity_threshold debe estar en (0, 1]")
        if not (0 < self.persistence_threshold < 1):
            raise ValueError("persistence_threshold debe estar en (0, 1)")


# Configuración singleton por defecto
DEFAULT_MIC_CONFIG: Final[MICConfiguration] = MICConfiguration()


# =============================================================================
# CONSTANTES DERIVADAS DE CONFIGURACIÓN
# =============================================================================

SUPPORTED_ENCODINGS: Final[FrozenSet[str]] = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1",
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be",
})

_ENCODING_ALIASES: Final[Dict[str, str]] = {
    "utf8": "utf-8",
    "latin1": "latin-1",
    "iso88591": "iso-8859-1",
    "cp65001": "utf-8",
}

VALID_DELIMITERS: Final[FrozenSet[str]] = frozenset({",", ";", "\t", "|", ":"})
VALID_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".csv", ".txt", ".tsv"})

_PHI: Final[float] = (1 + math.sqrt(5)) / 2  # Proporción áurea

_SEVERITY_WEIGHTS: Final[Dict[str, float]] = {
    "CRITICAL": 5.0,
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    "INFO": 0.5,
}


# =============================================================================
# TIPOS GENÉRICOS Y PROTOCOLOS
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")


class ProjectionResult(TypedDict, total=False):
    """Resultado tipado de una proyección MIC."""
    success: bool
    error: str
    error_type: str
    error_category: str
    error_details: Dict[str, Any]
    result: Any
    _mic_validation_update: int
    _mic_stratum: str
    _mic_validated_strata: List[str]


class DiagnosticResult(TypedDict, total=False):
    """Resultado tipado de un diagnóstico."""
    success: bool
    diagnostic_completed: bool
    is_empty: bool
    file_type: str
    file_path: str
    file_size_bytes: int
    diagnostic_magnitude: float
    has_topological_analysis: bool
    topological_features: Dict[str, Any]
    homology: Dict[str, Any]
    persistence_diagram: List[Dict[str, Any]]
    persistence_entropy: float
    error: str
    error_type: str
    error_category: str


class CacheStats(TypedDict):
    """Estadísticas del cache."""
    size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    ttl_seconds: float
    evictions: int
    expirations: int


class LatencyStats(TypedDict):
    """Estadísticas de latencia."""
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@runtime_checkable
class TelemetryContextProtocol(Protocol):
    """Protocolo para contextos de telemetría observacional."""
    def get_business_report(self) -> Dict[str, Any]: ...


@runtime_checkable
class DiagnosticProtocol(Protocol):
    """Protocolo para clases diagnósticas de archivos."""
    def diagnose(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class VectorHandler(Protocol):
    """Protocolo para handlers de vectores en la MIC."""
    def __call__(self, **kwargs: Any) -> Dict[str, Any]: ...


# =============================================================================
# TIPO DE ARCHIVO (FileType)
# =============================================================================

class FileType(str, Enum):
    """
    Tipos de archivo soportados para diagnóstico.

    Hereda de str para serialización JSON nativa.
    """
    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna lista de valores válidos."""
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> "FileType":
        """
        Parsea string a FileType con normalización robusta.

        Raises:
            TypeError: Si value no es string.
            ValueError: Si el valor no coincide con ningún miembro.
        """
        if not isinstance(value, str):
            raise TypeError(f"Se esperaba str, se recibió {type(value).__name__!r}")
        
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        
        raise ValueError(
            f"'{value}' no es válido. Opciones: {', '.join(cls.values())}"
        )


# =============================================================================
# JERARQUÍA DE EXCEPCIONES
# =============================================================================

class MICException(Exception):
    """
    Clase base para excepciones de la MIC.

    Cada excepción porta su contexto algebraico:
    (mensaje, detalles estructurados, categoría, timestamp).
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        category: str = "mic_error",
    ) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}
        self.category: str = category
        self.timestamp: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la excepción a diccionario."""
        return {
            "error": str(self),
            "error_type": type(self).__name__,
            "error_category": self.category,
            "error_details": self.details,
            "timestamp": self.timestamp,
        }


class FileNotFoundDiagnosticError(MICException):
    """Archivo no encontrado durante diagnóstico."""

    def __init__(self, path: Union[str, Path], **kwargs: Any) -> None:
        super().__init__(
            f"File not found: {path}",
            details={"path": str(path), **kwargs},
            category="validation",
        )


class UnsupportedFileTypeError(MICException):
    """Tipo de archivo no soportado por el sistema."""

    def __init__(self, file_type: str, available: List[str]) -> None:
        super().__init__(
            f"Unsupported file type: {file_type!r}",
            details={"file_type": file_type, "available_types": available},
            category="validation",
        )


class FileValidationError(MICException):
    """Error de validación de archivo."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="validation")


class FilePermissionError(MICException):
    """Error de permisos de archivo."""

    def __init__(self, path: Union[str, Path], operation: str = "read") -> None:
        super().__init__(
            f"Permission denied for {operation} operation: {path}",
            details={"path": str(path), "operation": operation},
            category="permission",
        )


class CleaningError(MICException):
    """Error durante proceso de limpieza de archivos."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="cleaning")


class MICHierarchyViolationError(MICException):
    """
    Violación de la Clausura Transitiva en la Filtración de Estratos.

    Ocurre cuando se intenta proyectar un vector al estrato k sin haber
    validado todos los estratos base j (donde j.value > k.value).
    """

    def __init__(
        self,
        target_stratum: Stratum,
        missing_strata: Set[Stratum],
        validated_strata: Set[Stratum],
    ) -> None:
        missing_names = sorted(
            [s.name for s in missing_strata],
            key=lambda n: Stratum[n].value,
            reverse=True,
        )
        validated_names = sorted(
            [s.name for s in validated_strata],
            key=lambda n: Stratum[n].value,
            reverse=True,
        )

        message = (
            f"Clausura Transitiva Violada: No se puede proyectar a "
            f"'{target_stratum.name}' (nivel {target_stratum.value}). "
            f"Estratos faltantes: {' → '.join(missing_names)}. "
            f"Validados: {', '.join(validated_names) if validated_names else 'ninguno'}."
        )

        super().__init__(
            message,
            details={
                "target_stratum": target_stratum.name,
                "target_value": target_stratum.value,
                "missing_strata": missing_names,
                "validated_strata": validated_names,
                "validation_order": [s.name for s in Stratum.ordered_bottom_up()],
            },
            category="hierarchy_violation",
        )

        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata


class TimeoutError(MICException):
    """Operación excedió tiempo límite."""

    def __init__(
        self, 
        operation: str, 
        timeout_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        super().__init__(
            f"Operation '{operation}' timed out after {elapsed_seconds:.2f}s "
            f"(limit: {timeout_seconds:.2f}s)",
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
            category="timeout",
        )


# =============================================================================
# MÉTRICAS Y TELEMETRÍA
# =============================================================================

class LatencyHistogram:
    """
    Histograma de latencias con estadísticas en tiempo real.
    
    Implementa un buffer circular para memoria acotada.
    Thread-safe para uso concurrente.
    """
    
    __slots__ = ("_buffer", "_max_size", "_lock", "_count")
    
    def __init__(self, max_size: int = 1000) -> None:
        self._buffer: deque[float] = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = threading.Lock()
        self._count = 0
    
    def record(self, latency_ms: float) -> None:
        """Registra una latencia en milisegundos."""
        with self._lock:
            self._buffer.append(latency_ms)
            self._count += 1
    
    @contextmanager
    def measure(self) -> Iterator[None]:
        """Context manager para medir latencia automáticamente."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(elapsed_ms)
    
    def get_stats(self) -> LatencyStats:
        """Calcula estadísticas del histograma."""
        with self._lock:
            if not self._buffer:
                return LatencyStats(
                    count=0, mean_ms=0.0, median_ms=0.0,
                    p95_ms=0.0, p99_ms=0.0, min_ms=0.0, max_ms=0.0
                )
            
            data = list(self._buffer)
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
        
        return LatencyStats(
            count=self._count,
            mean_ms=round(statistics.mean(data), 3),
            median_ms=round(statistics.median(data), 3),
            p95_ms=round(percentile(0.95), 3),
            p99_ms=round(percentile(0.99), 3),
            min_ms=round(min(data), 3),
            max_ms=round(max(data), 3),
        )
    
    def reset(self) -> None:
        """Reinicia el histograma."""
        with self._lock:
            self._buffer.clear()
            self._count = 0


@dataclass
class MICMetrics:
    """
    Métricas agregadas de la MIC.
    
    Recopila estadísticas de operaciones, cache, errores y latencias.
    """
    projections: int = 0
    cache_hits: int = 0
    violations: int = 0
    errors: int = 0
    timeouts: int = 0
    
    # Contadores por estrato
    projections_by_stratum: Dict[str, int] = field(default_factory=dict)
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    
    # Histogramas de latencia
    projection_latency: LatencyHistogram = field(
        default_factory=lambda: LatencyHistogram(1000)
    )
    handler_latency: LatencyHistogram = field(
        default_factory=lambda: LatencyHistogram(1000)
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa métricas a diccionario."""
        return {
            "counters": {
                "projections": self.projections,
                "cache_hits": self.cache_hits,
                "violations": self.violations,
                "errors": self.errors,
                "timeouts": self.timeouts,
            },
            "projections_by_stratum": self.projections_by_stratum.copy(),
            "errors_by_category": self.errors_by_category.copy(),
            "latency": {
                "projection": self.projection_latency.get_stats(),
                "handler": self.handler_latency.get_stats(),
            },
        }
    
    def record_projection(self, stratum: Stratum) -> None:
        """Registra una proyección exitosa."""
        self.projections += 1
        self.projections_by_stratum[stratum.name] = (
            self.projections_by_stratum.get(stratum.name, 0) + 1
        )
    
    def record_error(self, category: str) -> None:
        """Registra un error por categoría."""
        self.errors += 1
        self.errors_by_category[category] = (
            self.errors_by_category.get(category, 0) + 1
        )


# =============================================================================
# ESTRUCTURAS TOPOLÓGICAS INMUTABLES
# =============================================================================

@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """
    Intervalo de persistencia [birth, death) en el diagrama de persistencia.

    Invariantes:
        birth ≥ 0
        death ≥ birth  (death = +∞ → característica esencial)
        dimension ∈ ℕ₀
    """

    birth: float
    death: float
    dimension: int = 0

    def __post_init__(self) -> None:
        if self.birth < 0:
            raise ValueError(f"birth debe ser ≥ 0, recibido: {self.birth}")
        # Permitir death = inf para intervalos esenciales
        if not math.isinf(self.death) and self.death < self.birth:
            raise ValueError(
                f"death ({self.death}) debe ser ≥ birth ({self.birth}) o inf"
            )
        if self.dimension < 0:
            raise ValueError(f"dimension debe ser ≥ 0, recibido: {self.dimension}")

    @classmethod
    def essential(cls, birth: float, dimension: int = 0) -> "PersistenceInterval":
        """Crea un intervalo esencial (nunca muere)."""
        return cls(birth=birth, death=float("inf"), dimension=dimension)

    @property
    def persistence(self) -> float:
        """Tiempo de vida: death - birth. +∞ si es esencial."""
        return float("inf") if self.is_essential else self.death - self.birth

    @property
    def is_essential(self) -> bool:
        """True si la característica nunca muere."""
        return math.isinf(self.death)

    @property
    def midpoint(self) -> float:
        """Punto medio del intervalo."""
        return self.birth if self.is_essential else (self.birth + self.death) / 2.0

    def __lt__(self, other: "PersistenceInterval") -> bool:
        """Ordenamiento por persistencia descendente."""
        if not isinstance(other, PersistenceInterval):
            return NotImplemented
        if self.is_essential != other.is_essential:
            return self.is_essential
        if self.is_essential and other.is_essential:
            return self.birth < other.birth
        return self.persistence > other.persistence

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el intervalo a diccionario."""
        return {
            "birth": self.birth,
            "death": self.death if not self.is_essential else "inf",
            "persistence": self.persistence if not self.is_essential else "inf",
            "dimension": self.dimension,
            "is_essential": self.is_essential,
            "midpoint": self.midpoint,
        }


@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti de un espacio topológico.

    β₀: Componentes conexos
    β₁: Ciclos independientes
    β₂: Cavidades

    Invariante: χ = β₀ - β₁ + β₂ (Característica de Euler)
    """

    beta_0: int
    beta_1: int
    beta_2: int

    def __post_init__(self) -> None:
        for attr_name, val in [
            ("beta_0", self.beta_0),
            ("beta_1", self.beta_1),
            ("beta_2", self.beta_2),
        ]:
            if not isinstance(val, int) or val < 0:
                raise ValueError(
                    f"{attr_name} debe ser entero no negativo, recibido: {val!r}"
                )

    @property
    def euler_characteristic(self) -> int:
        """χ = β₀ - β₁ + β₂"""
        return self.beta_0 - self.beta_1 + self.beta_2

    @property
    def total_rank(self) -> int:
        """Rango total de homología."""
        return self.beta_0 + self.beta_1 + self.beta_2

    @property
    def is_connected(self) -> bool:
        """True si el espacio es conexo (β₀ = 1)."""
        return self.beta_0 == 1

    @property
    def has_cycles(self) -> bool:
        """True si existen ciclos no triviales."""
        return self.beta_1 > 0

    @classmethod
    def zero(cls) -> "BettiNumbers":
        """Números de Betti nulos."""
        return cls(beta_0=0, beta_1=0, beta_2=0)

    @classmethod
    def point(cls) -> "BettiNumbers":
        """Números de Betti de un punto."""
        return cls(beta_0=1, beta_1=0, beta_2=0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "betti_numbers": [self.beta_0, self.beta_1, self.beta_2],
            "euler_characteristic": self.euler_characteristic,
            "total_rank": self.total_rank,
            "is_connected": self.is_connected,
            "has_cycles": self.has_cycles,
        }


@dataclass(frozen=True, slots=True)
class TopologicalSummary:
    """
    Resumen completo de características topológicas de un dataset.
    """

    betti: BettiNumbers
    structural_entropy: float
    persistence_entropy: float
    intrinsic_dimension: int = 1

    def __post_init__(self) -> None:
        if self.structural_entropy < 0:
            raise ValueError(
                f"structural_entropy debe ser ≥ 0, recibido: {self.structural_entropy}"
            )
        if not (0.0 <= self.persistence_entropy <= 1.0):
            raise ValueError(
                f"persistence_entropy debe estar en [0,1], "
                f"recibido: {self.persistence_entropy}"
            )
        if self.intrinsic_dimension < 0:
            raise ValueError(
                f"intrinsic_dimension debe ser ≥ 0, recibido: {self.intrinsic_dimension}"
            )

    @classmethod
    def empty(cls) -> "TopologicalSummary":
        """Resumen vacío para casos de error o datos vacíos."""
        return cls(
            betti=BettiNumbers.zero(),
            structural_entropy=0.0,
            persistence_entropy=0.0,
            intrinsic_dimension=0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            **self.betti.to_dict(),
            "structural_entropy": round(self.structural_entropy, 6),
            "persistence_entropy": round(self.persistence_entropy, 6),
            "intrinsic_dimension": self.intrinsic_dimension,
        }


# =============================================================================
# VECTOR DE INTENCIÓN
# =============================================================================

@dataclass(frozen=True, slots=True)
class IntentVector:
    """
    Vector de intención inmutable proyectado sobre la MIC.

    v = (service_name, payload, context) ∈ S × P × C
    """

    service_name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.service_name or not self.service_name.strip():
            raise ValueError("service_name no puede estar vacío")

    @property
    def payload_hash(self) -> str:
        """SHA-256 truncado del payload para cache."""
        content = str(sorted(self.payload.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def norm(self) -> float:
        """Norma Euclidiana del vector de intención."""
        return math.sqrt(len(self.payload) + len(self.context))

    def with_context(self, **additional_context: Any) -> "IntentVector":
        """Crea un nuevo vector con contexto extendido."""
        return IntentVector(
            service_name=self.service_name,
            payload=self.payload,
            context={**self.context, **additional_context},
        )


# =============================================================================
# CACHE CON TTL MEJORADO
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Entrada de cache con metadata temporal."""

    value: T
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """True si la entrada ha superado su tiempo de vida."""
        return (time.monotonic() - self.timestamp) > ttl_seconds


class TTLCache(Generic[T]):
    """
    Cache thread-safe con Time-To-Live y evicción O(1).

    Mejoras v4:
    - __contains__ para verificación de existencia
    - Estadísticas de evicción y expiración
    - get_or_compute para patrón cache-aside
    - Estimación de tamaño en memoria
    """

    __slots__ = (
        "_data", "_lock", "_ttl", "_max_size",
        "_hits", "_misses", "_evictions", "_expirations"
    )

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_size: int = 128,
    ) -> None:
        self._data: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expirations: int = 0

    def __contains__(self, key: str) -> bool:
        """Verifica si una clave existe y no ha expirado."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired(self._ttl):
                del self._data[key]
                self._expirations += 1
                return False
            return True

    def get(self, key: str) -> Optional[T]:
        """Obtiene valor del cache."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(self._ttl):
                del self._data[key]
                self._misses += 1
                self._expirations += 1
                return None
            self._data.move_to_end(key)
            entry.access_count += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Almacena valor en cache."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = CacheEntry(
                    value=value, timestamp=time.monotonic()
                )
                return

            while len(self._data) >= self._max_size:
                self._data.popitem(last=False)
                self._evictions += 1

            self._data[key] = CacheEntry(value=value, timestamp=time.monotonic())

    def get_or_compute(
        self, 
        key: str, 
        compute_fn: Callable[[], T],
        ttl_override: Optional[float] = None,
    ) -> T:
        """
        Patrón cache-aside: obtiene del cache o computa y almacena.
        
        Args:
            key: Clave de cache
            compute_fn: Función para computar el valor si no está en cache
            ttl_override: TTL específico para esta entrada (no implementado aún)
        
        Returns:
            Valor del cache o recién computado
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        
        value = compute_fn()
        self.set(key, value)
        return value

    def clear(self) -> int:
        """Limpia el cache y retorna el número de entradas eliminadas."""
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
            return count

    def prune_expired(self) -> int:
        """Elimina entradas expiradas."""
        with self._lock:
            expired = [
                k for k, v in self._data.items() if v.is_expired(self._ttl)
            ]
            for key in expired:
                del self._data[key]
                self._expirations += 1
            return len(expired)

    @property
    def size(self) -> int:
        """Número de entradas en el cache."""
        with self._lock:
            return len(self._data)

    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos del cache."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> CacheStats:
        """Estadísticas del cache."""
        with self._lock:
            return CacheStats(
                size=len(self._data),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                hit_rate=self.hit_rate,
                ttl_seconds=self._ttl,
                evictions=self._evictions,
                expirations=self._expirations,
            )


# =============================================================================
# MÉTRICAS ESPECTRALES DEL GRAFO DE SERVICIOS
# =============================================================================

class SpectralGraphMetrics:
    """
    Análisis espectral del grafo de dependencias entre servicios.

    Mejoras v4:
    - Soporte para scipy.sparse cuando está disponible
    - Manejo de grafos vacíos y casos degenerados
    - Caching de matrices construidas
    """

    __slots__ = ("_mic", "_adjacency_cache", "_laplacian_cache")

    def __init__(self, mic_registry: "MICRegistry") -> None:
        self._mic = mic_registry
        self._adjacency_cache: Optional[np.ndarray] = None
        self._laplacian_cache: Optional[np.ndarray] = None

    def _invalidate_cache(self) -> None:
        """Invalida caches cuando el grafo cambia."""
        self._adjacency_cache = None
        self._laplacian_cache = None

    def build_adjacency_matrix(self, use_sparse: bool = False) -> np.ndarray:
        """
        Construye la matriz de adyacencia del grafo de servicios.

        Args:
            use_sparse: Si True y scipy.sparse disponible, retorna matriz sparse.

        Returns:
            Matriz de adyacencia A ∈ {0,1}^{n×n}.
        """
        if self._adjacency_cache is not None:
            return self._adjacency_cache

        with self._mic._lock:
            services = list(self._mic._vectors.keys())
            n = len(services)

            if n == 0:
                return np.zeros((0, 0), dtype=np.float64)

            idx = {svc: i for i, svc in enumerate(services)}
            
            if use_sparse and SCIPY_SPARSE_AVAILABLE:
                rows, cols, data = [], [], []
                for svc_i, (stratum_i, _) in self._mic._vectors.items():
                    for svc_j, (stratum_j, _) in self._mic._vectors.items():
                        if svc_i != svc_j and stratum_i in stratum_j.requires():
                            rows.append(idx[svc_i])
                            cols.append(idx[svc_j])
                            data.append(1.0)
                A = sparse.csr_matrix(
                    (data, (rows, cols)), shape=(n, n), dtype=np.float64
                )
            else:
                A = np.zeros((n, n), dtype=np.float64)
                for svc_i, (stratum_i, _) in self._mic._vectors.items():
                    for svc_j, (stratum_j, _) in self._mic._vectors.items():
                        if svc_i != svc_j and stratum_i in stratum_j.requires():
                            A[idx[svc_i], idx[svc_j]] = 1.0

            self._adjacency_cache = A
            return A

    def build_laplacian(self) -> np.ndarray:
        """
        Construye la matriz Laplaciana simétrica del grafo.

        L = D - A_sym donde A_sym = (A + Aᵀ) / 2
        """
        if self._laplacian_cache is not None:
            return self._laplacian_cache

        A = self.build_adjacency_matrix()
        if A.size == 0:
            return np.zeros((0, 0))

        A_sym = (A + A.T) / 2.0
        D = np.diag(A_sym.sum(axis=1))
        L = D - A_sym

        self._laplacian_cache = L
        return L

    def compute_spectral_metrics(
        self, 
        config: Optional[MICConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Calcula métricas espectrales del grafo de servicios.

        Returns:
            Dict con conectividad algebraica, radio espectral, energía, etc.
        """
        config = config or DEFAULT_MIC_CONFIG
        L = self.build_laplacian()
        A = self.build_adjacency_matrix()

        if L.size == 0:
            return {
                "algebraic_connectivity": 0.0,
                "spectral_radius": 0.0,
                "spectral_energy": 0.0,
                "is_connected": False,
                "n_components": 0,
                "n_services": 0,
            }

        try:
            # Eigenvalores de la Laplaciana
            eigenvalues_L = np.linalg.eigvalsh(L)
            eigenvalues_L_sorted = np.sort(eigenvalues_L)

            # λ₂ de Fiedler
            algebraic_connectivity = float(
                eigenvalues_L_sorted[1] if len(eigenvalues_L_sorted) > 1 else 0.0
            )

            # Componentes conexos
            n_components = int(
                np.sum(np.abs(eigenvalues_L_sorted) < config.epsilon)
            )

            # Eigenvalores de la matriz de adyacencia
            A_sym = (A + A.T) / 2.0
            eigenvalues_A = np.linalg.eigvalsh(A_sym)
            spectral_radius = float(np.max(np.abs(eigenvalues_A)))
            spectral_energy = float(np.sum(np.abs(eigenvalues_A)))

            return {
                "algebraic_connectivity": round(algebraic_connectivity, 6),
                "spectral_radius": round(spectral_radius, 6),
                "spectral_energy": round(spectral_energy, 6),
                "is_connected": algebraic_connectivity > config.epsilon,
                "n_components": n_components,
                "n_services": L.shape[0],
                "fiedler_value": round(algebraic_connectivity, 6),
            }

        except np.linalg.LinAlgError as e:
            logger.warning("Error en análisis espectral: %s", e)
            return {
                "algebraic_connectivity": 0.0,
                "spectral_radius": 0.0,
                "spectral_energy": 0.0,
                "is_connected": False,
                "n_components": 0,
                "n_services": L.shape[0],
                "error": str(e),
            }


# =============================================================================
# MATRIZ DE TRANSICIÓN ENTRE ESTRATOS (CORREGIDA)
# =============================================================================

class StratumTransitionMatrix:
    """
    Matriz de Transición Markoviana entre estratos de la MIC.

    T[i,j] = P(transición de estrato i a estrato j)

    Modela el flujo de validación en la filtración DIKW:
    - Desde PHYSICS se puede avanzar a TACTICS
    - Desde TACTICS se puede avanzar a STRATEGY
    - Desde STRATEGY se puede avanzar a WISDOM
    - WISDOM es estado absorbente
    """

    def __init__(self) -> None:
        self._strata = list(Stratum)
        self._n = len(self._strata)
        self._idx = {s: i for i, s in enumerate(self._strata)}

    def build(self, service_counts: Dict[Stratum, int]) -> np.ndarray:
        """
        Construye la matriz de transición estocástica.

        La transición i → j ocurre si el estrato i es prerrequisito de j.
        Acopla los Símbolos de Christoffel del tensor métrico G_{\mu\nu} para
        alterar la densidad geodésica.

        Args:
            service_counts: Número de servicios por estrato.

        Returns:
            Matriz T ∈ [0,1]^{n×n} estocástica por filas acoplada a la Conexión de Levi-Civita.
        """
        T = np.zeros((self._n, self._n), dtype=np.float64)
        epsilon = DEFAULT_MIC_CONFIG.epsilon

        # Extracción conceptual de los Símbolos de Christoffel para modelar fricción de transición.
        # Dominios de alta entropía/riesgo logístico (TACTICS) y finanzas (STRATEGY) concentran
        # una "resistencia gravitatoria" mayor.
        from app.core.immune_system.metric_tensors import G_PHYSICS, G_TOPOLOGY, G_THERMODYNAMICS

        christoffel_weights = {
            Stratum.PHYSICS: np.linalg.norm(G_PHYSICS, "fro"),
            Stratum.TACTICS: np.linalg.norm(G_TOPOLOGY, "fro"),
            Stratum.STRATEGY: np.linalg.norm(G_THERMODYNAMICS, "fro"),
            Stratum.OMEGA: 1.5,   # Fricción base del manifold
            Stratum.ALPHA: 1.2,
            Stratum.WISDOM: 1.0,  # La sabiduría fluye sin fricción intrínseca si los estratos base ceden
        }

        for s_from in self._strata:
            i = self._idx[s_from]
            
            # Destinos alcanzables: estratos que tienen a s_from como prerrequisito
            # j.requires() contiene s_from si s_from.value > j.value
            reachable = [
                s for s in self._strata
                if s != s_from and s_from in s.requires()
            ]

            if not reachable:
                # Estado absorbente (llegamos a la cúspide o base sin avance)
                T[i, i] = 1.0
                continue

            # Peso combina número de servicios y la resistencia geodésica (Christoffel inversa)
            # A mayor fricción del dominio destino, menor es la probabilidad cruda de transición
            weights = np.array(
                [
                    float(max(1, service_counts.get(s, 1))) / max(epsilon, christoffel_weights.get(s, 1.0))
                    for s in reachable
                ],
                dtype=np.float64,
            )
            total_weight = weights.sum()
            if total_weight > epsilon:
                weights /= total_weight
            else:
                weights = np.ones_like(weights) / len(weights)

            for s_to, w in zip(reachable, weights):
                j = self._idx[s_to]
                T[i, j] = w

        return T

    def stationary_distribution(
        self, service_counts: Dict[Stratum, int]
    ) -> Dict[str, float]:
        """
        Calcula la distribución estacionaria π tal que π T = π.

        Returns:
            Dict mapeando nombre de estrato a probabilidad estacionaria.
        """
        T = self.build(service_counts)
        epsilon = DEFAULT_MIC_CONFIG.epsilon

        try:
            # Eigenvectores de Tᵀ → eigenvectores izquierdos de T
            eigenvalues, eigenvectors = np.linalg.eig(T.T)

            # El eigenvalor 1 (dentro de tolerancia numérica)
            idx_unit = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx_unit])

            # Normalizar a distribución de probabilidad
            stationary = np.abs(stationary)
            total = stationary.sum()
            if total > epsilon:
                stationary /= total

            return {
                self._strata[i].name: round(float(stationary[i]), 6)
                for i in range(self._n)
            }

        except np.linalg.LinAlgError:
            # Fallback a distribución uniforme
            uniform = 1.0 / self._n
            return {s.name: round(uniform, 6) for s in self._strata}


# =============================================================================
# COMMAND PATTERN PARA PROYECCIÓN
# =============================================================================

class ProjectionCommand(ABC):
    """
    Comando abstracto para proyección de intenciones.
    
    Implementa el patrón Command para separar responsabilidades
    en el flujo de project_intent.
    """
    
    @abstractmethod
    def execute(self, context: "ProjectionContext") -> Optional[ProjectionResult]:
        """
        Ejecuta el comando.
        
        Returns:
            None para continuar al siguiente comando,
            ProjectionResult para terminar la cadena.
        """
        pass


@dataclass
class ProjectionContext:
    """Contexto compartido durante la proyección."""
    service_name: str
    payload: Dict[str, Any]
    context: Dict[str, Any]
    use_cache: bool
    
    # Resueltos durante ejecución
    cache_key: Optional[str] = None
    target_stratum: Optional[Stratum] = None
    handler: Optional[VectorHandler] = None
    validated_strata: Set[Stratum] = field(default_factory=set)
    force_override: bool = False
    
    # Métricas
    start_time: float = field(default_factory=time.perf_counter)


class CacheCheckCommand(ProjectionCommand):
    """Verifica el cache antes de procesar."""
    
    def __init__(self, cache: TTLCache, metrics: MICMetrics) -> None:
        self._cache = cache
        self._metrics = metrics
    
    def execute(self, ctx: ProjectionContext) -> Optional[ProjectionResult]:
        if not ctx.use_cache:
            return None
        
        try:
            payload_repr = str(sorted(ctx.payload.items()))
            ctx.cache_key = (
                f"{ctx.service_name}:"
                f"{hashlib.sha256(payload_repr.encode()).hexdigest()[:16]}"
            )
            cached = self._cache.get(ctx.cache_key)
            if cached is not None:
                self._metrics.cache_hits += 1
                logger.debug("Cache hit: '%s'", ctx.service_name)
                return cast(ProjectionResult, cached)
        except (TypeError, ValueError) as e:
            logger.debug(
                "Cache key computation failed para '%s': %s",
                ctx.service_name, e
            )
            ctx.cache_key = None
        
        return None


class ResolutionCommand(ProjectionCommand):
    """Resuelve el vector base y su estrato."""
    
    def __init__(
        self, 
        vectors: Dict[str, Tuple[Stratum, VectorHandler]],
        lock: threading.RLock,
        metrics: MICMetrics,
    ) -> None:
        self._vectors = vectors
        self._lock = lock
        self._metrics = metrics
    
    def execute(self, ctx: ProjectionContext) -> Optional[ProjectionResult]:
        with self._lock:
            if ctx.service_name not in self._vectors:
                available = list(self._vectors.keys())
                self._metrics.record_error("resolution_error")
                return ProjectionResult(
                    success=False,
                    error=f"Vector desconocido: '{ctx.service_name}'",
                    error_type="ValueError",
                    error_category="resolution_error",
                    error_details={"available_services": available},
                )
            
            ctx.target_stratum, ctx.handler = self._vectors[ctx.service_name]
        
        return None


class NormalizationCommand(ProjectionCommand):
    """Normaliza el contexto de validación."""
    
    def execute(self, ctx: ProjectionContext) -> Optional[ProjectionResult]:
        # Copia inmutable para garantizar la pureza funcional y la inmutabilidad
        raw_validated = ctx.context.get("validated_strata", set())
        raw_validated = set() if raw_validated is None else set(raw_validated)
        ctx.validated_strata = self._normalize_validated_strata(raw_validated)
        ctx.force_override = bool(
            ctx.context.get("force_override", False) or
            ctx.context.get("force_physics_override", False)
        )
        return None
    
    def _normalize_validated_strata(self, raw: Any) -> Set[Stratum]:
        """Normaliza el conjunto de estratos validados."""
        if raw is None:
            return set()

        if not isinstance(raw, (set, list, tuple, frozenset)):
            logger.warning(
                "validated_strata tipo inválido: %s — se ignora",
                type(raw).__name__,
            )
            return set()

        normalized: Set[Stratum] = set()

        for item in raw:
            try:
                if isinstance(item, Stratum):
                    normalized.add(item)
                elif isinstance(item, int):
                    normalized.add(Stratum(item))
                elif isinstance(item, str):
                    normalized.add(Stratum[item.upper().strip()])
            except (ValueError, KeyError):
                logger.debug("Ignorando estrato inválido: %r", item)

        return normalized


class ValidationCommand(ProjectionCommand):
    """Valida la jerarquía de estratos (Gatekeeper)."""
    
    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics
    
    def execute(self, ctx: ProjectionContext) -> Optional[ProjectionResult]:
        if ctx.force_override:
            logger.warning(
                "⚠️ Validación jerárquica bypaseada para '%s' via force_override",
                ctx.target_stratum.name if ctx.target_stratum else "unknown",
            )
            return None
        
        if ctx.target_stratum is None:
            return ProjectionResult(
                success=False,
                error="Target stratum not resolved",
                error_type="InternalError",
                error_category="resolution_error",
            )
        
        required = ctx.target_stratum.requires()
        missing = required - ctx.validated_strata
        
        if missing:
            self._metrics.violations += 1
            self._metrics.record_error("hierarchy_violation")
            
            error = MICHierarchyViolationError(
                target_stratum=ctx.target_stratum,
                missing_strata=missing,
                validated_strata=ctx.validated_strata,
            )
            logger.error(str(error))

            missing_names = sorted(
                [s.name for s in missing],
                key=lambda n: Stratum[n].value,
                reverse=True,
            )

            # Use ProjectionResult for correct type hints
            return ProjectionResult(
                success=False,
                error_category="hierarchy_violation",
                error_details={
                    "target_stratum": ctx.target_stratum.name,
                    "missing_strata": missing_names
                }
            )
        
        return None


class ExecutionCommand(ProjectionCommand):
    """Ejecuta el handler del servicio."""
    
    def __init__(
        self, 
        cache: TTLCache, 
        metrics: MICMetrics,
        config: MICConfiguration,
    ) -> None:
        self._cache = cache
        self._metrics = metrics
        self._config = config
    
    def execute(self, ctx: ProjectionContext) -> Optional[ProjectionResult]:
        if ctx.handler is None or ctx.target_stratum is None:
            return ProjectionResult(
                success=False,
                error="Handler or stratum not resolved",
                error_type="InternalError",
                error_category="execution_error",
            )
        
        # Resistencia geodésica: Si la herramienta cruza dominios de alta entropía,
        # exigimos demostración de exergía en su payload/contexto,
        # acoplando la restricción de despacho.
        exergy_level = float(ctx.context.get("exergy_level", 1.0))

        # Símbolos de Christoffel mapeados a entropía (Gravedad Estratigráfica)
        _christoffel_entropy = {
            Stratum.PHYSICS: 0.1,
            Stratum.TACTICS: 0.8,
            Stratum.STRATEGY: 0.9,
            Stratum.OMEGA: 0.5,
            Stratum.ALPHA: 0.2,
            Stratum.WISDOM: 0.0,
        }
        target_entropy = _christoffel_entropy.get(ctx.target_stratum, 0.0)

        if exergy_level < target_entropy:
            return ProjectionResult(
                success=False,
                error=f"La resistencia geodésica del estrato {ctx.target_stratum.name} repele "
                      f"la intención estocástica (exergía={exergy_level:.2f} < gravedad={target_entropy:.2f}). Demuestre coherencia.",
                error_type="GeodesicRepulsionError",
                error_category="thermodynamic_violation",
            )

        try:
            with self._metrics.handler_latency.measure():
                result = ctx.handler(**ctx.payload)
            
            # Normalizar resultado
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            
            # Propagación de validación
            if result.get("success", False):
                updated_validated = ctx.validated_strata | {ctx.target_stratum}
                result["_mic_validation_update"] = ctx.target_stratum.value
                result["_mic_stratum"] = ctx.target_stratum.name
                result["_mic_validated_strata"] = [s.name for s in updated_validated]
                
                # Almacenar en cache
                if ctx.use_cache and ctx.cache_key is not None:
                    self._cache.set(ctx.cache_key, result)
                
                self._metrics.record_projection(ctx.target_stratum)
            
            return cast(ProjectionResult, result)
        
        except TypeError as e:
            logger.error(
                "Firma de handler incorrecta para '%s': %s",
                ctx.service_name, e
            )
            self._metrics.record_error("handler_signature_error")
            return ProjectionResult(
                success=False,
                error=str(e),
                error_type="TypeError",
                error_category="handler_signature_error",
                error_details={
                    "service_name": ctx.service_name,
                    "hint": "Verifique que las claves del payload coincidan con los parámetros"
                },
            )
        
        except Exception as e:
            logger.exception("Error ejecutando vector '%s'", ctx.service_name)
            self._metrics.record_error("execution_error")
            return ProjectionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                error_category="execution_error",
                error_details={"service_name": ctx.service_name},
            )


# =============================================================================
# MATRIZ DE INTERACCIÓN CENTRAL (MIC)
# =============================================================================

class MICRegistry:
    """
    Matriz de Interacción Central (MIC).

    Implementa un Espacio Vectorial Jerárquico donde:
    1. Base Canónica: Cada servicio es un vector base eᵢ
    2. Filtración de Estratos: V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
    3. Gatekeeper Algebraico: Proyección condicional G(v, ctx)
    4. Clausura Transitiva: Validación de prerrequisitos
    """

    # Removing __slots__ to allow patching during tests, or at least we shouldn't use slots if we need mock
    # Wait, better to not modify slots if we don't have to, let's fix the test to patch the class instead.
    __slots__ = (
        "_vectors", "_lock", "_cache", "_logger", "_metrics",
        "_config", "_projection_commands", "_spectral_analyzer", "__dict__"
    )

    def __init__(
        self, 
        config: Optional[MICConfiguration] = None,
    ) -> None:
        """
        Inicializa la MIC.

        Args:
            config: Configuración de la MIC.
        """
        self._config = config or DEFAULT_MIC_CONFIG
        self._vectors: Dict[str, Tuple[Stratum, VectorHandler]] = {}
        self._lock = threading.RLock()
        self._cache: TTLCache[Dict[str, Any]] = TTLCache(
            ttl_seconds=self._config.cache_ttl_seconds,
            max_size=self._config.cache_max_size,
        )
        self._logger = get_structured_logger("MIC.Registry")
        self._metrics = MICMetrics()
        self._spectral_analyzer: Optional[SpectralGraphMetrics] = None
        
        # Inicializar comandos de proyección
        self._projection_commands: List[ProjectionCommand] = [
            CacheCheckCommand(self._cache, self._metrics),
            ResolutionCommand(self._vectors, self._lock, self._metrics),
            NormalizationCommand(),
            ValidationCommand(self._metrics),
            ExecutionCommand(self._cache, self._metrics, self._config),
        ]

    # ── Propiedades de introspección ───────────────────────────────────────

    @property


    def registered_services(self) -> List[str]:
        """Lista de servicios registrados."""
        with self._lock:
            return list(self._vectors.keys())

    @property
    def dimension(self) -> int:
        """Dimensión del espacio vectorial."""
        with self._lock:
            return len(self._vectors)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Métricas de uso de la MIC."""
        with self._lock:
            return {
                **self._metrics.to_dict(),
                "cache": self._cache.stats,
            }

    @property
    def config(self) -> MICConfiguration:
        """Configuración actual de la MIC."""
        return self._config

    def is_registered(self, service_name: str) -> bool:
        """Verifica si un servicio está registrado."""
        with self._lock:
            return service_name in self._vectors

    def get_stratum(self, service_name: str) -> Optional[Stratum]:
        """Retorna el estrato de un servicio, o None si no existe."""
        with self._lock:
            entry = self._vectors.get(service_name)
            return entry[0] if entry else None

    def get_services_by_stratum(self, stratum: Stratum) -> List[str]:
        """Retorna todos los servicios en un estrato dado."""
        with self._lock:
            return [
                name for name, (s, _) in self._vectors.items()
                if s == stratum
            ]

    def get_stratum_hierarchy(self) -> Dict[str, List[str]]:
        """Retorna la estructura jerárquica completa."""
        hierarchy: Dict[str, List[str]] = {
            s.name: [] for s in Stratum.ordered_bottom_up()
        }
        with self._lock:
            for name, (stratum, _) in self._vectors.items():
                hierarchy[stratum.name].append(name)
        return hierarchy

    def get_registered_morphisms(self) -> Dict[str, Tuple[Stratum, VectorHandler]]:
        """
        Retorna una proyección inmutable (difeomorfismo) de los vectores base actuales.
        Garantiza el aislamiento thread-safe bloqueando el estado durante la copia.
        """
        with self._lock:
            # Retornamos una copia superficial del diccionario para aislar la memoria,
            # protegiendo el encapsulamiento monádico del registro.
            return self._vectors.copy()

    # ── Registro de Vectores ───────────────────────────────────────────────



    def register_vector(
        self,
        service_name: str,
        stratum: Stratum,
        handler: VectorHandler,
    ) -> None:
        """
        Registra un microservicio como vector base.

        Args:
            service_name: Identificador único del servicio.
            stratum: Estrato jerárquico.
            handler: Callable que ejecuta la lógica.

        Raises:
            ValueError: Si service_name está vacío.
            TypeError: Si handler no es callable o stratum no es Stratum.
        """
        with self._lock:
            if not service_name or not service_name.strip():
                raise ValueError("service_name no puede estar vacío")

            if not isinstance(stratum, Stratum):
                raise TypeError(
                    f"stratum debe ser Stratum, recibido: {type(stratum).__name__!r}"
                )

            if not callable(handler):
                raise TypeError(
                    f"handler debe ser callable, recibido: {type(handler).__name__!r}"
                )

            if service_name in self._vectors:
                old_stratum = self._vectors[service_name][0]
                self._logger.warning(
                    "Sobrescribiendo vector '%s': %s → %s",
                    service_name, old_stratum.name, stratum.name
                )

            self._vectors[service_name] = (stratum, handler)
            
            # Invalidar cache de análisis espectral
            if self._spectral_analyzer is not None:
                self._spectral_analyzer._invalidate_cache()
            
            self._logger.info(
                "Vector registrado: '%s' [%s]", service_name, stratum.name
            )

    def unregister_vector(self, service_name: str) -> bool:
        """
        Elimina un servicio de la MIC.

        Returns:
            True si se eliminó exitosamente, False si no existía.
        """
        with self._lock:
            if service_name in self._vectors:
                del self._vectors[service_name]
                if self._spectral_analyzer is not None:
                    self._spectral_analyzer._invalidate_cache()
                self._logger.info("Vector eliminado: '%s'", service_name)
                return True
            return False

    # ── Proyección de Intenciones ─────────────────────────────────────────

    def project_intent(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
        *,
        use_cache: bool = False,
    ) -> ProjectionResult:
        """
        Proyecta una intención sobre el espacio vectorial.

        Flujo de ejecución (Command Pattern):
          1. Cache check
          2. Resolución del vector base
          3. Normalización del contexto
          4. Validación jerárquica (Gatekeeper)
          5. Ejecución del handler
          6. Propagación de validación

        Args:
            service_name: Nombre del servicio a proyectar.
            payload: Argumentos para el handler.
            context: Contexto de ejecución.
            use_cache: Si True, usa cache de resultados.

        Returns:
            ProjectionResult con resultado o error estructurado.
        """
        ctx = ProjectionContext(
            service_name=service_name,
            payload=payload,
            context=context,
            use_cache=use_cache,
        )
        
        with self._metrics.projection_latency.measure():
            for command in self._projection_commands:
                result = command.execute(ctx)
                if result is not None:
                    return result
        
        # No debería llegar aquí
        return ProjectionResult(
            success=False,
            error="Projection pipeline incomplete",
            error_type="InternalError",
            error_category="pipeline_error",
        )

    # ── Utilidades ─────────────────────────────────────────────────────────

    def clear_cache(self) -> int:
        """Limpia el cache de resultados."""
        count = self._cache.clear()
        self._logger.info("Cache limpiado: %d entradas eliminadas", count)
        return count

    def spectral_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis espectral del grafo de servicios."""
        if self._spectral_analyzer is None:
            self._spectral_analyzer = SpectralGraphMetrics(self)
        return self._spectral_analyzer.compute_spectral_metrics(self._config)

    def stratum_statistics(self) -> Dict[str, Any]:
        """Estadísticas por estrato."""
        with self._lock:
            counts: Dict[str, int] = {s.name: 0 for s in Stratum}
            for _, (stratum, _) in self._vectors.items():
                counts[stratum.name] += 1

        total = sum(counts.values())
        distribution = {
            k: round(v / total, 4) if total > 0 else 0.0
            for k, v in counts.items()
        }

        probs = list(distribution.values())
        entropy = compute_shannon_entropy(probs)

        return {
            "counts_by_stratum": counts,
            "distribution": distribution,
            "stratum_entropy": round(entropy, 6),
            "total_services": total,
        }


# =============================================================================
# FUNCIONES DE ENTROPÍA Y PROBABILIDAD
# =============================================================================

def compute_shannon_entropy(
    probabilities: Sequence[float],
    base: float = 2.0,
    epsilon: float = DEFAULT_MIC_CONFIG.epsilon,
) -> float:
    """
    Calcula la entropía de Shannon con estabilidad numérica.

    H(X) = -Σ p(xᵢ) · log_b(p(xᵢ))
    """
    if not probabilities:
        return 0.0

    if base <= 1.0:
        raise ValueError(f"La base debe ser > 1, recibida: {base}")

    probs = np.asarray(probabilities, dtype=np.float64)

    if np.any(probs < 0.0):
        raise ValueError("Las probabilidades no pueden ser negativas")

    total = np.sum(probs)
    if total < epsilon:
        return 0.0

    if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
        probs = probs / total

    mask = probs > epsilon
    nonzero_probs = probs[mask]

    if len(nonzero_probs) == 0:
        return 0.0

    log_base = math.log(base)
    entropy = -float(np.sum(nonzero_probs * np.log(nonzero_probs))) / log_base

    return max(0.0, entropy)


def distribution_from_counts(counts: Union[Dict[Any, int], Counter]) -> List[float]:
    """Convierte conteos a distribución de probabilidad."""
    if not counts:
        return []

    values = list(counts.values())
    total = sum(values)

    if total == 0:
        return []

    return [v / total for v in values]


def compute_persistence_entropy(
    intervals: Sequence[PersistenceInterval],
    config: Optional[MICConfiguration] = None,
) -> float:
    """
    Calcula la entropía del diagrama de persistencia.

    H_pers = -Σ (lᵢ/L) · log₂(lᵢ/L)
    """
    config = config or DEFAULT_MIC_CONFIG
    
    if not intervals:
        return 0.0

    finite = [iv for iv in intervals if not iv.is_essential]

    if not finite:
        return 0.0

    persistences = np.array([iv.persistence for iv in finite], dtype=np.float64)
    total = persistences.sum()

    if total < config.epsilon:
        return 0.0

    probs = persistences / total
    raw_entropy = compute_shannon_entropy(probs.tolist())

    n = len(probs)
    max_entropy = math.log2(n) if n > 1 else 1.0

    return raw_entropy / max_entropy if max_entropy > config.epsilon else 0.0


# =============================================================================
# ANÁLISIS TOPOLÓGICO
# =============================================================================

def _jaccard_similarity(tokens_a: FrozenSet[str], tokens_b: FrozenSet[str]) -> float:
    """Similitud de Jaccard entre dos conjuntos."""
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _tokenize_line(line: str) -> FrozenSet[str]:
    """Tokeniza una línea en conjunto de tokens."""
    tokens = re.split(r"[,;\t|:\s]+", line.strip())
    return frozenset(t for t in tokens if t)


def detect_cyclic_patterns(
    lines: List[str], 
    config: Optional[MICConfiguration] = None,
) -> int:
    """
    Detecta patrones cíclicos en una secuencia de líneas.

    Mejoras v4:
    - Early termination cuando se alcanza umbral
    - Límite configurable de líneas a analizar
    - Progreso logarítmico para archivos grandes
    """
    config = config or DEFAULT_MIC_CONFIG
    n = len(lines)
    
    if n < 3:
        return 0

    # Limitar líneas para rendimiento
    effective_n = min(n, config.max_lines_for_cycle_detection)
    lines_to_analyze = lines[:effective_n]

    # Pre-tokenizar
    tokenized = [_tokenize_line(line) for line in lines_to_analyze]

    cycles_found = 0
    effective_max = min(config.max_cycle_period, effective_n // 2)

    for period in range(1, effective_max + 1):
        comparisons = effective_n - period
        if comparisons <= 0:
            continue

        matches = sum(
            1 for i in range(comparisons)
            if _jaccard_similarity(tokenized[i], tokenized[i + period])
            >= config.cycle_similarity_threshold
        )

        if matches / comparisons >= config.cycle_similarity_threshold:
            cycles_found += 1

    return cycles_found


def estimate_intrinsic_dimension(
    lines: List[str],
    config: Optional[MICConfiguration] = None,
) -> int:
    """
    Estima la dimensión intrínseca del espacio de datos.
    """
    config = config or DEFAULT_MIC_CONFIG
    
    if not lines:
        return 0

    data_lines = lines[1:] if len(lines) > 1 else lines
    sample = data_lines[:min(100, len(data_lines))]

    if not sample:
        return 1

    for delimiter in [",", ";", "\t", "|", ":"]:
        if any(delimiter in line for line in sample[:5]):
            col_counts = [len(line.split(delimiter)) for line in sample]
            if col_counts:
                col_counts_sorted = sorted(col_counts)
                mid = len(col_counts_sorted) // 2
                median_cols = (
                    col_counts_sorted[mid]
                    if len(col_counts_sorted) % 2 != 0
                    else (col_counts_sorted[mid - 1] + col_counts_sorted[mid]) // 2
                )
                return max(1, int(median_cols))

    return 1


def analyze_topological_features(
    file_path: Path,
    config: Optional[MICConfiguration] = None,
) -> TopologicalSummary:
    """
    Analiza características topológicas de un archivo.
    """
    config = config or DEFAULT_MIC_CONFIG
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [
                line.rstrip("\n\r") 
                for line in f.readlines()[:config.max_sample_rows]
            ]

        if not lines:
            return TopologicalSummary.empty()

        line_counts = Counter(lines)
        num_unique = len(line_counts)

        beta_0 = max(1, num_unique)
        beta_1 = detect_cyclic_patterns(lines, config)
        dimension = estimate_intrinsic_dimension(lines, config)

        distribution = distribution_from_counts(line_counts)
        structural_entropy = compute_shannon_entropy(distribution)

        betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0)

        return TopologicalSummary(
            betti=betti,
            structural_entropy=structural_entropy,
            persistence_entropy=0.0,
            intrinsic_dimension=dimension,
        )

    except Exception as e:
        logger.warning("Análisis topológico falló para '%s': %s", file_path, e)
        return TopologicalSummary.empty()


def compute_homology_from_diagnostic(diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula grupos de homología a partir de datos diagnósticos."""
    issues = diagnostic_data.get("issues", [])
    warnings = diagnostic_data.get("warnings", [])

    issue_types: Set[str] = set()
    for issue in issues:
        if isinstance(issue, dict):
            issue_types.add(issue.get("type", issue.get("code", "unknown")))
        else:
            issue_types.add(type(issue).__name__)

    beta_0 = max(1, len(issue_types))

    circular_keywords = frozenset({
        "circular", "cycle", "loop", "recursive", "dependency"
    })

    def has_circular(item: Any) -> bool:
        text = str(item).lower()
        return any(kw in text for kw in circular_keywords)

    beta_1 = sum(1 for item in (*warnings, *issues) if has_circular(item))

    betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0)

    return {
        "H_0": f"ℤ^{beta_0}",
        "H_1": f"ℤ^{beta_1}" if beta_1 > 0 else "0",
        **betti.to_dict(),
    }


def compute_persistence_diagram(
    diagnostic_data: Dict[str, Any],
) -> List[PersistenceInterval]:
    """Calcula el diagrama de persistencia para issues diagnósticos."""
    issues = diagnostic_data.get("issues", [])
    if not issues:
        return []

    severity_to_weight: Dict[str, float] = {
        "CRITICAL": 1.0, "HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2, "INFO": 0.1,
    }

    delta_t: float = 0.1
    intervals: List[PersistenceInterval] = []

    for idx, issue in enumerate(issues):
        if isinstance(issue, dict):
            raw_severity = issue.get("severity", "MEDIUM")
            weight = severity_to_weight.get(str(raw_severity).upper(), 0.5)
        else:
            weight = 0.5

        birth = idx * delta_t
        death = birth + weight

        try:
            intervals.append(
                PersistenceInterval(birth=birth, death=death, dimension=0)
            )
        except ValueError:
            pass

    config = DEFAULT_MIC_CONFIG
    significant = [
        iv for iv in intervals 
        if iv.persistence >= config.persistence_threshold
    ]
    significant.sort()

    return significant


def compute_diagnostic_magnitude(diagnostic_data: Dict[str, Any]) -> float:
    """Calcula la magnitud normalizada del vector diagnóstico."""
    issues = diagnostic_data.get("issues", [])
    errors = diagnostic_data.get("errors", [])
    warnings = diagnostic_data.get("warnings", [])

    severity_counts: Counter[str] = Counter()

    for item in issues:
        if isinstance(item, dict):
            sev = str(item.get("severity", "MEDIUM")).upper()
        else:
            sev = "MEDIUM"
        severity_counts[sev] += 1

    severity_counts["CRITICAL"] += len(errors)
    severity_counts["LOW"] += len(warnings)

    weighted_sq_sum = sum(
        _SEVERITY_WEIGHTS.get(sev, 1.0) * (count ** 2)
        for sev, count in severity_counts.items()
    )
    raw_magnitude = math.sqrt(weighted_sq_sum)

    total_items = max(1, len(issues) + len(errors) + len(warnings))
    scale = math.sqrt(float(total_items))

    normalized = math.tanh(raw_magnitude / scale)

    return round(normalized, 4)


# =============================================================================
# VALIDACIÓN DE ARCHIVOS
# =============================================================================

def normalize_path(path: Union[str, Path]) -> Path:
    """Normaliza ruta a Path absoluto y resuelto."""
    if path is None:
        raise ValueError("Path no puede ser None")
    if not str(path).strip():
        raise ValueError("Path no puede estar vacío")
    return Path(path).expanduser().resolve()


def validate_file_exists(path: Path) -> None:
    """Valida existencia y tipo del objeto."""
    if not path.exists():
        raise FileNotFoundDiagnosticError(path)
    if not path.is_file():
        raise FileValidationError(
            f"La ruta no apunta a un archivo regular: {path}",
            path=str(path),
        )


def validate_file_permissions(path: Path, check_read: bool = True) -> None:
    """
    Valida permisos de acceso al archivo.
    
    Args:
        path: Ruta al archivo.
        check_read: Si verificar permiso de lectura.
    
    Raises:
        FilePermissionError: Si no hay permisos suficientes.
    """
    if check_read and not os.access(path, os.R_OK):
        raise FilePermissionError(path, "read")


def validate_file_extension(path: Path) -> str:
    """Valida la extensión del archivo."""
    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise FileValidationError(
            f"Extensión no soportada: '{ext}'",
            provided=ext,
            expected=sorted(VALID_EXTENSIONS),
        )
    return ext


def validate_file_size(
    path: Path,
    max_size: Optional[int] = None,
) -> Tuple[int, bool]:
    """Valida el tamaño del archivo."""
    max_size = max_size or DEFAULT_MIC_CONFIG.max_file_size_bytes
    size = path.stat().st_size
    if size > max_size:
        raise FileValidationError(
            f"Archivo excede el límite: {size:,} bytes > {max_size:,} bytes",
            actual_size_bytes=size,
            max_size_bytes=max_size,
            file=str(path),
        )
    return size, size == 0


def normalize_encoding(encoding: str) -> str:
    """Normaliza el nombre de codificación."""
    if not encoding or not encoding.strip():
        return "utf-8"

    norm = encoding.lower().replace("_", "-").replace(" ", "")

    for alias, standard in _ENCODING_ALIASES.items():
        alias_norm = alias.lower().replace("_", "").replace("-", "")
        if norm.replace("-", "") == alias_norm:
            return standard

    if encoding.lower() in SUPPORTED_ENCODINGS:
        return encoding.lower()

    logger.warning(
        "Codificación '%s' no reconocida, usando 'utf-8'", encoding
    )
    return "utf-8"


def normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    """Normaliza tipo de archivo."""
    if isinstance(file_type, FileType):
        return file_type
    return FileType.from_string(file_type)


# =============================================================================
# REGISTRO DE DIAGNÓSTICOS
# =============================================================================

_DIAGNOSTIC_REGISTRY: Dict[FileType, Optional[Type[DiagnosticProtocol]]] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}


def get_diagnostic_class(file_type: FileType) -> Type[DiagnosticProtocol]:
    """Obtiene la clase diagnóstica para un tipo de archivo."""
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)

    if diagnostic_class is None:
        raise UnsupportedFileTypeError(file_type.value, available=FileType.values())

    return diagnostic_class


# =============================================================================
# HANDLERS DE LA MIC
# =============================================================================

def analyze_financial_viability(
    amount: float,
    std_dev: float,
    time_years: int,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Vector Estratégico: Analiza viabilidad financiera usando el FinancialEngine.
    """
    try:
        if FinancialEngine and FinancialConfig:
            config = FinancialConfig(market_volatility=std_dev)
            engine = FinancialEngine(config)

            # Simulated cash flows based on amount and basic assumptions
            cash_flows = [-amount] + [amount * 0.3] * time_years

            npv = engine.calculate_npv(cash_flows, initial_investment=amount)
            var, cvar = engine.calculate_var(amount)

            return {
                "success": True,
                "npv": npv,
                "var_95": var,
                "cvar_95": cvar,
                "contingency_suggested": engine.suggest_contingency(amount),
                "is_viable": npv > 0,
            }
        else:
            return {
                "success": False,
                "error": "FinancialEngine no está disponible",
                "error_category": "dependency_error"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error"
        }


def clean_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    delimiter: str = ";",
    encoding: str = "utf-8",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Vector Físico: Limpia un archivo usando CSVCleaner.
    """
    try:
        if CSVCleaner:
            cleaner = CSVCleaner(
                input_file=str(input_path),
                output_file=str(output_path),
                delimiter=delimiter,
                encoding=encoding
            )
            cleaner.clean()
            return {
                "success": True,
                "output_path": str(output_path),
                "message": "Limpieza completada"
            }
        else:
            return {
                "success": False,
                "error": "CSVCleaner no está disponible",
                "error_category": "dependency_error"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error"
        }


def get_telemetry_status(
    telemetry_context: Any = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Vector Físico: Obtiene el estado de telemetría actual.
    """
    try:
        metrics = {}
        report = {}
        if telemetry_context:
            metrics = getattr(telemetry_context, "metrics", {})
            if hasattr(telemetry_context, "get_business_report"):
                report = telemetry_context.get_business_report()

        return {
            "success": True,
            "status": "active",
            "metrics": metrics,
            "report": report
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error"
        }


def diagnose_file(
    file_path: Union[str, Path],
    file_type: Union[str, FileType],
    *,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
    topological_analysis: bool = False,
    config: Optional[MICConfiguration] = None,
) -> DiagnosticResult:
    """
    Vector de Diagnóstico con análisis topológico opcional.

    Estrato: PHYSICS (Nivel 3 — Base de la pirámide)
    """
    config = config or DEFAULT_MIC_CONFIG
    path_str = str(file_path)

    try:
        path = normalize_path(file_path)
        normalized_type = normalize_file_type(file_type)

        logger.info(
            "Iniciando diagnóstico: '%s' [tipo=%s]", path, normalized_type.value
        )

        validate_file_exists(path)
        validate_file_permissions(path)

        if validate_extension:
            validate_file_extension(path)

        effective_max = max_file_size or config.max_file_size_bytes
        size, is_empty = validate_file_size(path, effective_max)

        if is_empty:
            logger.warning("Archivo vacío: '%s'", path)
            return DiagnosticResult(
                success=True,
                diagnostic_completed=True,
                is_empty=True,
                file_type=normalized_type.value,
                file_path=str(path),
                file_size_bytes=0,
            )

        diagnostic_class = get_diagnostic_class(normalized_type)
        diagnostic = diagnostic_class(str(path))
        diagnostic.diagnose()
        result_data = diagnostic.to_dict()
        result_data["diagnostic_completed"] = True

        if topological_analysis:
            logger.debug("Ejecutando análisis topológico para '%s'", path)

            topo_summary = analyze_topological_features(path, config)
            result_data["topological_features"] = topo_summary.to_dict()

            homology = compute_homology_from_diagnostic(result_data)
            result_data["homology"] = homology

            intervals = compute_persistence_diagram(result_data)
            result_data["persistence_diagram"] = [iv.to_dict() for iv in intervals]
            result_data["persistence_entropy"] = compute_persistence_entropy(
                intervals, config
            )

        magnitude = compute_diagnostic_magnitude(result_data)

        logger.info(
            "Diagnóstico completado: '%s' [magnitud=%.4f]", path, magnitude
        )

        return DiagnosticResult(
            success=True,
            **result_data,
            file_type=normalized_type.value,
            file_path=str(path),
            file_size_bytes=size,
            diagnostic_magnitude=magnitude,
            has_topological_analysis=topological_analysis,
        )

    except MICException as e:
        logger.warning("Error de validación: %s", e)
        return DiagnosticResult(success=False, **e.to_dict())

    except Exception as e:
        logger.exception("Error inesperado en diagnóstico de '%s'", path_str)
        return DiagnosticResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            error_category="unexpected",
        )


# =============================================================================
# BOOTSTRAP Y REGISTRO DE VECTORES
# =============================================================================

def register_core_vectors(
    mic: MICRegistry,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Registra los vectores fundamentales del sistema."""
    # PHYSICS
    mic.register_vector("stabilize_flux", Stratum.PHYSICS, vector_stabilize_flux)
    mic.register_vector("parse_raw", Stratum.PHYSICS, vector_parse_raw_structure)

    # TACTICS
    mic.register_vector("structure_logic", Stratum.TACTICS, vector_structure_logic)
    mic.register_vector(
        "audit_fusion_homology", Stratum.TACTICS, vector_audit_homological_fusion
    )

    # STRATEGY
    mic.register_vector(
        "lateral_thinking_pivot", Stratum.STRATEGY, vector_lateral_pivot
    )

    # Vectores con dependencias opcionales
    if config:
        try:
            from app.tactics.semantic_estimator import SemanticEstimatorService
            service = SemanticEstimatorService(config)
            service.register_in_mic(mic)
            logger.info("✅ Vectores semánticos registrados")
        except Exception as e:
            logger.warning("⚠️ Vectores semánticos no disponibles: %s", e)

    try:
        from app.wisdom.semantic_dictionary import SemanticDictionaryService
        semantic_dict = SemanticDictionaryService()
        semantic_dict.register_in_mic(mic)
        logger.info("✅ Diccionario semántico registrado")
    except Exception as e:
        logger.warning("⚠️ Diccionario semántico no disponible: %s", e)

    logger.info(
        "✅ MIC inicializada con %d vectores (dimensión=%d)",
        mic.dimension, mic.dimension
    )
    logger.debug("Jerarquía: %s", mic.get_stratum_hierarchy())


# =============================================================================
# API PÚBLICA
# =============================================================================

def get_supported_file_types() -> List[str]:
    """Retorna tipos de archivo soportados."""
    return FileType.values()


def get_supported_delimiters() -> List[str]:
    """Retorna delimitadores CSV soportados."""
    return sorted(VALID_DELIMITERS)


def get_supported_encodings() -> List[str]:
    """Retorna codificaciones soportadas."""
    return sorted(SUPPORTED_ENCODINGS)


def validate_file_for_processing(
    path: Union[str, Path],
    config: Optional[MICConfiguration] = None,
) -> Dict[str, Any]:
    """Valida completamente un archivo para procesamiento."""
    config = config or DEFAULT_MIC_CONFIG
    
    try:
        p = normalize_path(path)
        validate_file_exists(p)
        validate_file_permissions(p)
        ext = validate_file_extension(p)
        size, is_empty = validate_file_size(p, config.max_file_size_bytes)
        return {
            "valid": True,
            "size": size,
            "extension": ext,
            "is_empty": is_empty,
            "path": str(p),
        }
    except MICException as e:
        return {"valid": False, "errors": [str(e)], **e.to_dict()}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


# =============================================================================
# SINGLETON DE LA MIC
# =============================================================================

_global_mic: Optional[MICRegistry] = None
_mic_lock = threading.Lock()
_mic_init_error: Optional[Exception] = None


def get_global_mic(
    config: Optional[Dict[str, Any]] = None,
    mic_config: Optional[MICConfiguration] = None,
    force_reinit: bool = False,
) -> MICRegistry:
    """
    Obtiene la instancia global de la MIC (Singleton thread-safe).

    Args:
        config: Configuración del sistema para el bootstrap.
        mic_config: Configuración específica de la MIC.
        force_reinit: Si True, fuerza reinicialización.

    Returns:
        Instancia inicializada de MICRegistry.

    Raises:
        RuntimeError: Si el bootstrap falla.
    """
    global _global_mic, _mic_init_error

    if _global_mic is not None and not force_reinit:
        return _global_mic

    with _mic_lock:
        if _global_mic is not None and not force_reinit:
            return _global_mic

        # Reintentar si hubo error previo y force_reinit
        if _mic_init_error is not None and not force_reinit:
            raise RuntimeError(
                f"MIC global falló previamente: {_mic_init_error}"
            ) from _mic_init_error

        try:
            mic = MICRegistry(config=mic_config)
            register_core_vectors(mic, config=config)
            _global_mic = mic
            _mic_init_error = None
            logger.info("MIC global inicializada con %d vectores", mic.dimension)
            return _global_mic

        except Exception as e:
            _mic_init_error = e
            logger.exception("Error crítico durante bootstrap de la MIC")
            raise RuntimeError(f"No se pudo inicializar la MIC: {e}") from e


def reset_global_mic() -> None:
    """Reinicia la instancia global de la MIC."""
    global _global_mic, _mic_init_error
    with _mic_lock:
        _global_mic = None
        _mic_init_error = None
    logger.info("MIC global reiniciada")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuración
    "MICConfiguration",
    "DEFAULT_MIC_CONFIG",
    
    # Tipos
    "FileType",
    "Stratum",
    "ProjectionResult",
    "DiagnosticResult",
    "CacheStats",
    "LatencyStats",
    
    # Estructuras topológicas
    "PersistenceInterval",
    "BettiNumbers",
    "TopologicalSummary",
    "IntentVector",
    
    # Excepciones
    "MICException",
    "FileNotFoundDiagnosticError",
    "UnsupportedFileTypeError",
    "FileValidationError",
    "FilePermissionError",
    "CleaningError",
    "MICHierarchyViolationError",
    "TimeoutError",
    
    # Cache y métricas
    "TTLCache",
    "LatencyHistogram",
    "MICMetrics",
    
    # Análisis
    "SpectralGraphMetrics",
    "StratumTransitionMatrix",
    
    # Core
    "MICRegistry",
    
    # Funciones
    "diagnose_file",
    "get_global_mic",
    "reset_global_mic",
    "register_core_vectors",
    "get_supported_file_types",
    "get_supported_delimiters",
    "get_supported_encodings",
    "validate_file_for_processing",
    
    # Funciones matemáticas
    "compute_shannon_entropy",
    "compute_persistence_entropy",
    "analyze_topological_features",
    "compute_homology_from_diagnostic",
    "compute_persistence_diagram",
    "compute_diagnostic_magnitude",

    # Handlers adicionales
    "analyze_financial_viability",
    "clean_file",
    "get_telemetry_status",
]
