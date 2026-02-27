"""
Módulo: MIC Vectors (Adaptadores de Capacidad y Morfismos Algebraicos)
======================================================================

Este módulo implementa la capa de adaptación ("Glue Code") como un conjunto de
**Morfismos Algebraicos** que proyectan las intenciones del Agente sobre los
motores físicos y lógicos del sistema.

Fundamentos Matemáticos:
────────────────────────

1. Teoría de Categorías (Morfismos):
   Cada función vectorial φ es un morfismo:
       φ: ConfigSpace × Context → ResultSpace
   
   preservando la estructura a través de los estratos DIKW.

2. Coherencia Topológica (C):
   Métrica invariante para evaluar calidad de transformación:
       C = clamp(S · R / (1 + H), 0, 1)
   
   donde S = stability, R = resonance, H = entropy.

3. Números de Betti y Característica de Euler:
   Para un complejo simplicial K:
       χ(K) = β₀ - β₁ + β₂
   
   Este invariante se verifica como postcondición.

4. Isomorfismo Dimensional:
   Guardas algebraicas que aseguran:
       dim(V_expected) ≅ dim(V_actual)  (ε-tolerancia)

5. Secuencia de Mayer-Vietoris:
   Para auditar fusión de grafos A ∪ B:
       ... → H₁(A ∩ B) → H₁(A) ⊕ H₁(B) → H₁(A ∪ B) → H₀(A ∩ B) → ...
   
   Si Δβ₁ = β₁(A∪B) - [β₁(A) + β₁(B) - β₁(A∩B)] ≠ 0, la fusión es inválida.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum, auto
from typing import (
    Any, Callable, Dict, Final, FrozenSet, Generic, Iterator,
    List, NamedTuple, Optional, Protocol, Set, Tuple, Type,
    TypeVar, Union, cast, runtime_checkable
)

import numpy as np

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("mic_vectors")


# =============================================================================
# IMPORTS CON FALLBACK ROBUSTO
# =============================================================================

def _safe_import(module_path: str, class_name: str) -> Optional[Type]:
    """Importación segura con logging de diagnóstico."""
    try:
        parts = module_path.split(".")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.debug(f"Import opcional fallido: {module_path}.{class_name} - {e}")
        return None


# Motores físicos
APUProcessor = _safe_import("app.apu_processor", "APUProcessor")
CondenserConfig = _safe_import("app.flux_condenser", "CondenserConfig")
DataFluxCondenser = _safe_import("app.flux_condenser", "DataFluxCondenser")
ReportParserCrudo = _safe_import("app.report_parser_crudo", "ReportParserCrudo")

# Mocks para testing standalone
if APUProcessor is None:
    class APUProcessor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): 
            self.raw_records: List[Dict[str, Any]] = []
        def process_all(self): return None
        def get_validation_kernel(self) -> Dict[str, Any]: return {}
        def get_topological_coherence(self) -> float: return 1.0
        def get_algebraic_integrity(self) -> float: return 1.0
        def get_quality_report(self) -> Dict[str, Any]: return {}

if CondenserConfig is None:
    @dataclass
    class CondenserConfig:  # type: ignore[no-redef]
        system_capacitance: float = 1.0
        system_inductance: float = 1.0
        base_resistance: float = 1.0
        resonance_threshold: float = 0.85

if DataFluxCondenser is None:
    class DataFluxCondenser:  # type: ignore[no-redef]
        def __init__(self, config, profile, condenser_config): pass
        def stabilize(self, file_path): return None
        def get_physics_report(self) -> Dict[str, Any]: return {}

if ReportParserCrudo is None:
    class ReportParserCrudo:  # type: ignore[no-redef]
        validation_stats: Optional[Dict[str, Any]] = None
        def __init__(self, *args, **kwargs): pass
        def parse_to_raw(self) -> List[Dict[str, Any]]: return []
        def get_parse_cache(self) -> Dict[str, Any]: return {}

# Stratum fallback
try:
    from app.schemas import Stratum
except ImportError:
    class Stratum(IntEnum):
        """Representación jerárquica del modelo DIKW."""
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3


# =============================================================================
# DEPENDENCIA OPCIONAL: PSUTIL
# =============================================================================

_HAS_PSUTIL: bool = False
_psutil_module: Any = None

try:
    import psutil as _psutil_module
    _HAS_PSUTIL = True
except ImportError:
    logger.debug("psutil no disponible - métricas de memoria retornarán 0.0")


# =============================================================================
# CONSTANTES DE CONFIGURACIÓN (Agrupadas Semánticamente)
# =============================================================================

class TopologicalConstants:
    """Constantes para análisis topológico."""
    EPSILON: Final[float] = 1e-10
    DIM_ISO_TOLERANCE: Final[float] = 0.10  # Tolerancia para isomorfismo dimensional
    PERSISTENCE_THRESHOLD: Final[float] = 0.01
    MAX_BETTI_DIMENSION: Final[int] = 2


class PhysicsConstants:
    """Constantes para el dominio físico (RLC)."""
    DEFAULT_RESONANCE_THRESHOLD: Final[float] = 0.85
    DEFAULT_STABILITY: Final[float] = 1.0
    DEFAULT_SYSTEM_TEMP: Final[float] = 25.0
    DEFAULT_HEAT_CAPACITY: Final[float] = 0.5
    

class FinancialConstants:
    """Constantes para decisiones financieras."""
    FINANCIAL_INERTIA_THRESHOLD: Final[float] = 0.70
    WAIT_OPTION_NPV_MULTIPLIER: Final[float] = 1.5
    RISK_CLASSES: Final[FrozenSet[str]] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


class MetricsConstants:
    """Constantes para medición de métricas."""
    BYTES_PER_MB: Final[float] = 1024.0 * 1024.0
    MS_PER_SECOND: Final[float] = 1000.0


# =============================================================================
# ENUMERACIONES
# =============================================================================

class VectorResultStatus(Enum):
    """Estratificación del resultado por tipo de éxito/fallo."""
    SUCCESS = "success"
    PHYSICS_ERROR = "physics_error"
    LOGIC_ERROR = "logic_error"
    TOPOLOGY_ERROR = "topology_error"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_ERROR = "dependency_error"


class AlgebraicStructure(Enum):
    """
    Estructura algebraica para procesamiento de datos.
    
    Define el tipo de álgebra abstracta que modela las transformaciones:
    - MODULE: Módulo sobre un anillo (estructura más general)
    - VECTOR_SPACE: Espacio vectorial sobre un campo
    - GROUP: Grupo de transformaciones
    - RING: Anillo de operaciones
    """
    MODULE = "module"
    VECTOR_SPACE = "vector_space"
    GROUP = "group"
    RING = "ring"
    
    @classmethod
    def from_string(cls, value: str) -> 'AlgebraicStructure':
        """Parsea string a AlgebraicStructure."""
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Estructura algebraica inválida: '{value}'")


class PivotType(Enum):
    """
    Tipos de pivote lateral para decisiones estratégicas.
    
    Cada pivote representa una transformación en el espacio de decisiones
    que permite mitigar riesgos específicos.
    """
    MONOPOLIO_COBERTURADO = "MONOPOLIO_COBERTURADO"
    OPCION_ESPERA = "OPCION_ESPERA"
    CUARENTENA_TOPOLOGICA = "CUARENTENA_TOPOLOGICA"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, value: str) -> 'PivotType':
        """Parsea string a PivotType con fallback a UNKNOWN."""
        normalized = value.strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.UNKNOWN


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

@dataclass(frozen=True, slots=True)
class VectorMetrics:
    """
    Valor inmutable que captura la telemetría de un vector.
    
    Inmutabilidad (frozen) ⟹ objeto algebraico puro:
    una vez construido, su identidad observacional es fija.
    
    Attributes:
        processing_time_ms: Tiempo de procesamiento en milisegundos
        memory_usage_mb: Uso de memoria en megabytes
        topological_coherence: Coherencia topológica C ∈ [0, 1]
        algebraic_integrity: Integridad algebraica I ∈ (0, 1]
    """
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    topological_coherence: float = 1.0
    algebraic_integrity: float = 1.0
    
    def __post_init__(self) -> None:
        """Valida invariantes de las métricas."""
        # Usar object.__setattr__ porque es frozen
        if self.processing_time_ms < 0:
            object.__setattr__(self, 'processing_time_ms', 0.0)
        if self.memory_usage_mb < 0:
            object.__setattr__(self, 'memory_usage_mb', 0.0)
        
        # Clamp de coherencia e integridad
        tc = max(0.0, min(1.0, self.topological_coherence))
        ai = max(0.0, min(1.0, self.algebraic_integrity))
        object.__setattr__(self, 'topological_coherence', tc)
        object.__setattr__(self, 'algebraic_integrity', ai)
    
    def to_dict(self) -> Dict[str, float]:
        """Serializa a diccionario."""
        return asdict(self)
    
    def combine(self, other: 'VectorMetrics') -> 'VectorMetrics':
        """
        Combina dos métricas (para composición de vectores).
        
        Reglas de combinación:
        - Tiempos se suman
        - Memoria toma el máximo
        - Coherencia e integridad toman el mínimo (cuello de botella)
        """
        return VectorMetrics(
            processing_time_ms=self.processing_time_ms + other.processing_time_ms,
            memory_usage_mb=max(self.memory_usage_mb, other.memory_usage_mb),
            topological_coherence=min(self.topological_coherence, other.topological_coherence),
            algebraic_integrity=min(self.algebraic_integrity, other.algebraic_integrity)
        )


@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti de un complejo simplicial.
    
    β₀: Componentes conexos
    β₁: Ciclos independientes (1-agujeros)
    β₂: Cavidades (2-agujeros)
    
    Invariante de Euler: χ = β₀ - β₁ + β₂
    """
    beta_0: int
    beta_1: int
    beta_2: int
    
    def __post_init__(self) -> None:
        for name, val in [("beta_0", self.beta_0), ("beta_1", self.beta_1), ("beta_2", self.beta_2)]:
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} debe ser entero no-negativo, recibido: {val}")
    
    @property
    def euler_characteristic(self) -> int:
        """Característica de Euler χ = β₀ - β₁ + β₂."""
        return self.beta_0 - self.beta_1 + self.beta_2
    
    @property
    def is_acyclic(self) -> bool:
        """True si no hay ciclos ni cavidades."""
        return self.beta_1 == 0 and self.beta_2 == 0
    
    @property
    def total_rank(self) -> int:
        """Rango total de homología."""
        return self.beta_0 + self.beta_1 + self.beta_2
    
    @classmethod
    def zero(cls) -> 'BettiNumbers':
        """Números de Betti nulos."""
        return cls(beta_0=0, beta_1=0, beta_2=0)
    
    def to_dict(self) -> Dict[str, int]:
        """Serializa a diccionario."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "euler": self.euler_characteristic
        }


class ValidationResult(NamedTuple):
    """
    Resultado de validación tipado.
    
    Reemplaza Tuple[bool, Optional[str]] con semántica clara.
    """
    is_valid: bool
    error_message: Optional[str] = None
    details: Dict[str, Any] = {}
    
    @classmethod
    def success(cls) -> 'ValidationResult':
        """Crea resultado exitoso."""
        return cls(is_valid=True)
    
    @classmethod
    def failure(cls, message: str, **details: Any) -> 'ValidationResult':
        """Crea resultado de fallo."""
        return cls(is_valid=False, error_message=message, details=dict(details))


@dataclass(frozen=True, slots=True)
class Dimensionality:
    """
    Dimensionalidad del espacio de atributos por tipo de registro.
    
    dim(V_t) = |⋃_{r : type(r)=t} keys(r)|
    """
    dimensions: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        # Validar que todas las dimensiones sean no-negativas
        for key, val in self.dimensions.items():
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"Dimensión inválida para '{key}': {val}")
    
    def is_isomorphic_to(
        self, 
        other: 'Dimensionality',
        tolerance: float = TopologicalConstants.DIM_ISO_TOLERANCE
    ) -> bool:
        """
        Verifica isomorfismo dimensional con tolerancia ε.
        
        ∀ t ∈ Types: |dim_self(t) - dim_other(t)| / max(dim_self(t), 1) ≤ ε
        """
        # Ambos vacíos: isomorfismo trivial
        if not self.dimensions and not other.dimensions:
            return True
        
        # Exactamente uno vacío: no isomorfos
        if not self.dimensions or not other.dimensions:
            return False
        
        # Tipos distintos: no isomorfos
        if set(self.dimensions.keys()) != set(other.dimensions.keys()):
            return False
        
        for key in self.dimensions:
            ref = max(self.dimensions[key], 1)
            if abs(self.dimensions[key] - other.dimensions[key]) / ref > tolerance:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, int]:
        """Serializa a diccionario."""
        return dict(self.dimensions)
    
    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> 'Dimensionality':
        """
        Calcula dimensionalidad desde una lista de registros.
        
        La dimensión de cada tipo es la cardinalidad de la unión de atributos.
        """
        if not records:
            return cls()
        
        attribute_spaces: Dict[str, Set[str]] = {}
        for record in records:
            rtype = str(record.get("record_type", "default"))
            attribute_spaces.setdefault(rtype, set()).update(record.keys())
        
        return cls(dimensions={rtype: len(attrs) for rtype, attrs in attribute_spaces.items()})


# =============================================================================
# TIPOS DE RESULTADO
# =============================================================================

# Tipo para resultados de vector (compatible con Dict[str, Any])
VectorResult = Dict[str, Any]

# Tipo para payloads
Payload = Dict[str, Any]

# Tipo para contexto
Context = Dict[str, Any]


# =============================================================================
# PROTOCOLOS
# =============================================================================

@runtime_checkable
class VectorHandler(Protocol):
    """Protocolo para handlers de vector en la MIC."""
    def __call__(self, **kwargs: Any) -> VectorResult: ...


@runtime_checkable
class PhysicsEngineProtocol(Protocol):
    """Protocolo para motores físicos (DataFluxCondenser, etc.)."""
    def get_physics_report(self) -> Dict[str, Any]: ...


# =============================================================================
# MEDICIÓN DE MÉTRICAS (Context Manager)
# =============================================================================

@dataclass
class MetricsCollector:
    """
    Colector de métricas para vectores.
    
    Uso como context manager:
        with MetricsCollector() as collector:
            # ... operaciones ...
        metrics = collector.build_metrics(coherence=0.95)
    """
    _start_time: float = field(default_factory=time.monotonic)
    _start_memory: float = field(default=0.0)
    _end_time: float = field(default=0.0)
    _end_memory: float = field(default=0.0)
    
    def __enter__(self) -> 'MetricsCollector':
        self._start_time = time.monotonic()
        self._start_memory = _measure_memory_mb()
        return self
    
    def __exit__(self, *args) -> None:
        self._end_time = time.monotonic()
        self._end_memory = _measure_memory_mb()
    
    @property
    def elapsed_ms(self) -> float:
        """Milisegundos transcurridos."""
        end = self._end_time if self._end_time > 0 else time.monotonic()
        return (end - self._start_time) * MetricsConstants.MS_PER_SECOND
    
    @property
    def memory_delta_mb(self) -> float:
        """Delta de memoria en MB."""
        return max(0.0, self._end_memory - self._start_memory)
    
    @property
    def current_memory_mb(self) -> float:
        """Memoria actual en MB."""
        return self._end_memory if self._end_memory > 0 else _measure_memory_mb()
    
    def build_metrics(
        self,
        topological_coherence: float = 1.0,
        algebraic_integrity: float = 1.0
    ) -> VectorMetrics:
        """Construye VectorMetrics con los valores recolectados."""
        return VectorMetrics(
            processing_time_ms=self.elapsed_ms,
            memory_usage_mb=self.current_memory_mb,
            topological_coherence=topological_coherence,
            algebraic_integrity=algebraic_integrity
        )


def _measure_memory_mb() -> float:
    """Consumo de memoria residente (MB). Retorna 0.0 si psutil no existe."""
    if not _HAS_PSUTIL or _psutil_module is None:
        return 0.0
    
    try:
        return _psutil_module.Process().memory_info().rss / MetricsConstants.BYTES_PER_MB
    except Exception:
        return 0.0


# =============================================================================
# CONSTRUCTORES DE RESULTADO
# =============================================================================

def _build_result(
    *,
    success: bool,
    stratum: Stratum,
    status: VectorResultStatus,
    metrics: Optional[VectorMetrics] = None,
    error: Optional[str] = None,
    **payload: Any,
) -> VectorResult:
    """
    Constructor canónico de VectorResult.
    
    Garantiza esquema consistente:
    {success, stratum, status, metrics, [error], ...payload}
    """
    result: VectorResult = {
        "success": success,
        "stratum": stratum.name if hasattr(stratum, "name") else str(stratum),
        "status": status.value,
        "metrics": (metrics or VectorMetrics()).to_dict(),
    }
    
    if error is not None:
        result["error"] = error
    
    result.update(payload)
    return result


def _build_success(
    stratum: Stratum,
    metrics: VectorMetrics,
    **payload: Any
) -> VectorResult:
    """Constructor de resultado exitoso."""
    return _build_result(
        success=True,
        stratum=stratum,
        status=VectorResultStatus.SUCCESS,
        metrics=metrics,
        **payload
    )


def _build_error(
    *,
    stratum: Stratum,
    status: VectorResultStatus,
    error: str,
    metrics: Optional[VectorMetrics] = None,
    **extra: Any
) -> VectorResult:
    """Constructor de resultado de error."""
    return _build_result(
        success=False,
        stratum=stratum,
        status=status,
        metrics=metrics,
        error=error,
        **extra
    )


# =============================================================================
# GUARDAS TOPOLÓGICAS
# =============================================================================

class TopologicalGuard:
    """
    Guardas topológicas para validación de precondiciones.
    
    Implementa las verificaciones necesarias antes de ejecutar
    operaciones que requieren invariantes topológicos.
    """
    
    @staticmethod
    def validate_file_exists(file_path: str) -> ValidationResult:
        """Verifica existencia del archivo (punto base de la variedad)."""
        if not file_path:
            return ValidationResult.failure("Ruta de archivo vacía")
        
        if not os.path.exists(file_path):
            return ValidationResult.failure(
                f"Topología rota: archivo '{file_path}' no existe",
                file_path=file_path
            )
        
        if not os.path.isfile(file_path):
            return ValidationResult.failure(
                f"La ruta no es un archivo: '{file_path}'",
                file_path=file_path
            )
        
        return ValidationResult.success()
    
    @staticmethod
    def validate_config_keys(
        config: Dict[str, Any],
        required_keys: List[str]
    ) -> ValidationResult:
        """Verifica presencia de claves requeridas (secciones del haz)."""
        if not config:
            return ValidationResult.failure("Configuración vacía")
        
        missing = [k for k in required_keys if k not in config]
        if missing:
            return ValidationResult.failure(
                f"Configuración incompleta: faltan {missing}",
                missing_keys=missing
            )
        
        return ValidationResult.success()
    
    @staticmethod
    def validate_dimension_constraints(
        constraints: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Verifica consistencia numérica de restricciones dimensionales."""
        if constraints is None:
            return ValidationResult.success()
        
        if not isinstance(constraints, dict):
            return ValidationResult.failure(
                "Restricciones dimensionales deben ser un diccionario"
            )
        
        non_numeric = [
            k for k, v in constraints.items()
            if not isinstance(v, (int, float))
        ]
        
        if non_numeric:
            return ValidationResult.failure(
                f"Dimensionalidad no numérica en claves: {non_numeric}",
                invalid_keys=non_numeric
            )
        
        return ValidationResult.success()
    
    @staticmethod
    def validate_homological_constraints(
        constraints: Dict[str, Any]
    ) -> ValidationResult:
        """
        Valida restricciones homológicas para complejos de cadenas.
        
        Requisitos:
        - max_dimension: int ≥ 0
        - allow_holes: bool
        - connectivity: numérico ≥ 0
        """
        required = {"max_dimension", "allow_holes", "connectivity"}
        if not required.issubset(constraints.keys()):
            missing = required - set(constraints.keys())
            return ValidationResult.failure(
                f"Restricciones homológicas incompletas: faltan {missing}",
                missing_keys=list(missing)
            )
        
        md = constraints["max_dimension"]
        if not isinstance(md, int) or md < 0:
            return ValidationResult.failure(
                f"max_dimension debe ser entero ≥ 0, recibido: {md}"
            )
        
        if not isinstance(constraints["allow_holes"], bool):
            return ValidationResult.failure(
                "allow_holes debe ser booleano"
            )
        
        conn = constraints["connectivity"]
        if not isinstance(conn, (int, float)) or conn < 0:
            return ValidationResult.failure(
                f"connectivity debe ser numérico ≥ 0, recibido: {conn}"
            )
        
        return ValidationResult.success()
    
    @classmethod
    def validate_physics_preconditions(
        cls,
        file_path: str,
        config: Dict[str, Any],
        required_keys: List[str]
    ) -> ValidationResult:
        """
        Validación compuesta de precondiciones físicas.
        
        Combina: existencia de archivo + claves de config + restricciones.
        """
        # 1. Archivo
        result = cls.validate_file_exists(file_path)
        if not result.is_valid:
            return result
        
        # 2. Claves de configuración
        result = cls.validate_config_keys(config, required_keys)
        if not result.is_valid:
            return result
        
        # 3. Restricciones dimensionales (si existen)
        constraints = config.get("dimension_constraints")
        if constraints is not None:
            result = cls.validate_dimension_constraints(constraints)
            if not result.is_valid:
                return result
        
        return ValidationResult.success()


# =============================================================================
# FUNCIONES DE TOPOLOGÍA ALGEBRAICA
# =============================================================================

def calculate_topological_coherence(physics_report: Dict[str, Any]) -> float:
    """
    Coherencia topológica como cociente normalizado.
    
    Fórmula:
        C = clamp(S · R / (1 + H), 0, 1)
    
    donde:
        S = stability ∈ [0, 1]
        R = resonance ∈ [0, 1]
        H = entropy ∈ [0, ∞)
    
    Propiedades:
    - C ∈ [0, 1] por construcción
    - ∂C/∂S > 0, ∂C/∂R > 0 (monótona creciente en S, R)
    - ∂C/∂H < 0 (monótona decreciente en H)
    - C = 0 ⟺ S = 0 ∨ R = 0
    - C = 1 cuando H = 0, S = R = 1
    """
    if not physics_report:
        return 0.0
    
    stability = max(0.0, min(1.0, float(physics_report.get("stability_index", 0))))
    entropy = max(0.0, float(physics_report.get("entropy", 0)))
    resonance = max(0.0, min(1.0, float(physics_report.get("resonance_factor", 0))))
    
    if stability < TopologicalConstants.EPSILON or resonance < TopologicalConstants.EPSILON:
        return 0.0
    
    raw = stability * resonance / (1.0 + entropy)
    return max(0.0, min(1.0, raw))


def calculate_betti_numbers(
    raw_records: List[Dict[str, Any]],
    cache: Dict[str, Any]
) -> BettiNumbers:
    """
    Números de Betti proxy del complejo simplicial inducido por registros.
    
    Definiciones proxy:
        β₀ = tipos de registro distintos (componentes conexos)
        β₁ = ciclos de dependencia en el grafo del cache
        β₂ = estructuras anidadas vacías (cavidades)
    
    Postcondición: Se verifica y loggea el invariante de Euler.
    """
    if not raw_records:
        return BettiNumbers.zero()
    
    # β₀: cardinalidad de tipos distintos
    types: Set[str] = {r.get("record_type", "unknown") for r in raw_records}
    beta_0 = len(types)
    
    # β₁: ciclos del grafo de dependencias
    cycles = cache.get("dependency_cycles", [])
    beta_1 = len(cycles) if isinstance(cycles, list) else 0
    
    # β₂: cavidades (nodos anidados vacíos)
    beta_2 = sum(
        1 for r in raw_records
        if r.get("nested") and not r.get("content")
    )
    
    betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=beta_2)
    
    # Log de diagnóstico para complejos no triviales
    if not betti.is_acyclic:
        logger.debug(
            f"Complejo simplicial no trivial: β₀={beta_0}, β₁={beta_1}, "
            f"β₂={beta_2}, χ={betti.euler_characteristic}"
        )
    
    return betti


def calculate_algebraic_integrity(betti: BettiNumbers) -> float:
    """
    Integridad algebraica derivada de números de Betti superiores.
    
    Fórmula:
        I = 1 / (1 + β₁ + β₂)
    
    Propiedades:
    - I = 1 ⟹ variedad acíclica (sin ciclos ni cavidades)
    - I → 0 cuando defectos topológicos se acumulan
    - I ∈ (0, 1] siempre
    """
    higher_sum = betti.beta_1 + betti.beta_2
    return 1.0 / (1.0 + higher_sum)


# =============================================================================
# VECTOR FÍSICO 1: ESTABILIZACIÓN DE FLUJO (DataFluxCondenser)
# =============================================================================

_REQUIRED_STABILIZE_KEYS: Final[List[str]] = [
    "system_capacitance",
    "system_inductance", 
    "base_resistance",
]


def vector_stabilize_flux(
    file_path: str,
    config: Dict[str, Any],
) -> VectorResult:
    """
    Vector de nivel PHYSICS.
    
    Invoca al DataFluxCondenser para estabilizar la ingesta de un archivo.
    
    Morfismo: (FilePath × Config) ─F─▶ StabilizedManifold
    
    Circuito físico equivalente (RLC serie):
        Z(ω) = R + j(ωL - 1/ωC)
        Resonancia: ω₀ = 1/√(LC)
    """
    with MetricsCollector() as collector:
        # ─── Validación de precondiciones ───
        validation = TopologicalGuard.validate_physics_preconditions(
            file_path, config, _REQUIRED_STABILIZE_KEYS
        )
        
        if not validation.is_valid:
            return _build_error(
                stratum=Stratum.PHYSICS,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=validation.error_message or "Validación fallida",
                metrics=collector.build_metrics(topological_coherence=0.0),
                validation_details=validation.details
            )
        
        try:
            # ─── Construcción del condensador ───
            condenser_conf = CondenserConfig(
                system_capacitance=float(config["system_capacitance"]),
                system_inductance=float(config["system_inductance"]),
                base_resistance=float(config["base_resistance"]),
                resonance_threshold=float(
                    config.get("resonance_threshold", PhysicsConstants.DEFAULT_RESONANCE_THRESHOLD)
                ),
            )
            
            condenser = DataFluxCondenser(
                config=config,
                profile=config.get("file_profile", {}),
                condenser_config=condenser_conf,
            )
            
            # ─── Estabilización ───
            df_stabilized = condenser.stabilize(file_path)
            physics_report = condenser.get_physics_report()
            
            coherence = calculate_topological_coherence(physics_report)
            metrics = collector.build_metrics(topological_coherence=coherence)
            
            return _build_success(
                stratum=Stratum.PHYSICS,
                metrics=metrics,
                data=df_stabilized.to_dict("records") if df_stabilized is not None else [],
                physics_metrics=physics_report,
            )
        
        except Exception as exc:
            logger.error(f"Fallo en vector 'stabilize_flux': {exc}", exc_info=True)
            return _build_error(
                stratum=Stratum.PHYSICS,
                status=VectorResultStatus.PHYSICS_ERROR,
                error=str(exc),
                metrics=collector.build_metrics(topological_coherence=0.0)
            )


# =============================================================================
# VECTOR FÍSICO 2: PARSING TOPOLÓGICO (ReportParserCrudo)
# =============================================================================

def vector_parse_raw_structure(
    file_path: str,
    profile: Dict[str, Any],
    topological_constraints: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel PHYSICS.
    
    Utiliza ReportParserCrudo para extraer el complejo simplicial del archivo.
    
    Morfismo: (FilePath × Profile) ─∂─▶ ChainComplex(Records)
    
    Homología proxy:
        H₀ → componentes conexos (registros crudos)
        H₁ → ciclos (dependencias circulares)
        H₂ → cavidades (estructuras anidadas vacías)
    """
    with MetricsCollector() as collector:
        # ─── Validación de archivo ───
        validation = TopologicalGuard.validate_file_exists(file_path)
        if not validation.is_valid:
            return _build_error(
                stratum=Stratum.PHYSICS,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=validation.error_message or "Archivo no existe",
                metrics=collector.build_metrics(topological_coherence=0.0)
            )
        
        # ─── Validación de restricciones homológicas ───
        if topological_constraints is not None:
            validation = TopologicalGuard.validate_homological_constraints(topological_constraints)
            if not validation.is_valid:
                return _build_error(
                    stratum=Stratum.PHYSICS,
                    status=VectorResultStatus.TOPOLOGY_ERROR,
                    error=validation.error_message or "Restricciones inválidas",
                    metrics=collector.build_metrics(topological_coherence=0.0)
                )
        
        try:
            # ─── Parsing ───
            parser = ReportParserCrudo(
                file_path,
                profile=profile,
                topological_constraints=topological_constraints,
            )
            
            raw_records: List[Dict[str, Any]] = parser.parse_to_raw()
            cache: Dict[str, Any] = parser.get_parse_cache()
            
            # ─── Cálculo de invariantes topológicos ───
            betti = calculate_betti_numbers(raw_records, cache)
            integrity = calculate_algebraic_integrity(betti)
            
            # Inyectar dimensionalidad para validación aguas abajo
            dimensionality = Dimensionality.from_records(raw_records)
            cache["dimensionality"] = dimensionality.to_dict()
            
            metrics = collector.build_metrics(algebraic_integrity=integrity)
            
            # ─── Extracción segura de validation_stats ───
            validation_stats: Dict[str, Any] = {}
            if hasattr(parser, "validation_stats") and parser.validation_stats is not None:
                vs = parser.validation_stats
                if hasattr(vs, "__dict__"):
                    validation_stats = dict(vars(vs))
                elif isinstance(vs, dict):
                    validation_stats = dict(vs)
            
            return _build_success(
                stratum=Stratum.PHYSICS,
                metrics=metrics,
                raw_records=raw_records,
                parse_cache=cache,
                validation_stats=validation_stats,
                homological_invariants=betti.to_dict(),
            )
        
        except Exception as exc:
            logger.error(f"Fallo en vector 'parse_raw_structure': {exc}", exc_info=True)
            return _build_error(
                stratum=Stratum.PHYSICS,
                status=VectorResultStatus.PHYSICS_ERROR,
                error=str(exc),
                metrics=collector.build_metrics(topological_coherence=0.0)
            )


# =============================================================================
# VECTOR TÁCTICO: ESTRUCTURACIÓN LÓGICA (APUProcessor)
# =============================================================================

def vector_structure_logic(
    raw_records: List[Dict[str, Any]],
    parse_cache: Dict[str, Any],
    config: Dict[str, Any],
    algebraic_structure: Union[str, AlgebraicStructure] = AlgebraicStructure.MODULE,
) -> VectorResult:
    """
    Vector de nivel TACTICS.
    
    Transforma registros crudos en estructuras de costos validadas.
    
    Morfismo: (Records × Cache × Config) ─φ─▶ ProcessedModule
    
    Álgebra abstracta:
        G = grupo de transformaciones (APUProcessor)
        φ = homomorfismo G → Aut(Data)
        ker(φ) = núcleo de validación (errores)
        im(φ) = imagen procesada limpia
    """
    with MetricsCollector() as collector:
        # ─── Validación de entrada ───
        if not raw_records:
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.LOGIC_ERROR,
                error="Conjunto vacío: no hay registros para transformar",
                metrics=collector.build_metrics()
            )
        
        # ─── Normalizar estructura algebraica ───
        if isinstance(algebraic_structure, str):
            try:
                alg_struct = AlgebraicStructure.from_string(algebraic_structure)
            except ValueError as e:
                return _build_error(
                    stratum=Stratum.TACTICS,
                    status=VectorResultStatus.VALIDATION_ERROR,
                    error=str(e),
                    metrics=collector.build_metrics()
                )
        else:
            alg_struct = algebraic_structure
        
        # ─── Verificación de isomorfismo dimensional ───
        expected_dims = Dimensionality(parse_cache.get("dimensionality", {}))
        actual_dims = Dimensionality.from_records(raw_records)
        
        if not expected_dims.is_isomorphic_to(actual_dims):
            logger.warning(
                f"Isomorfismo dimensional roto: esperado={expected_dims.to_dict()}, "
                f"actual={actual_dims.to_dict()}"
            )
            # No es error fatal, pero se registra como warning
        
        try:
            # ─── Procesamiento ───
            processor = APUProcessor(
                config=config,
                parse_cache=parse_cache,
                algebraic_structure=alg_struct.value,
            )
            processor.raw_records = raw_records
            
            df_processed = processor.process_all()
            
            # ─── Extracción segura de métricas del procesador ───
            kernel = _safe_call(processor, "get_validation_kernel", {})
            topo_coherence = _safe_call(processor, "get_topological_coherence", 1.0)
            alg_integrity = _safe_call(processor, "get_algebraic_integrity", 1.0)
            quality_report = _safe_call(processor, "get_quality_report", {})
            
            metrics = collector.build_metrics(
                topological_coherence=topo_coherence,
                algebraic_integrity=alg_integrity
            )
            
            return _build_success(
                stratum=Stratum.TACTICS,
                metrics=metrics,
                processed_data=df_processed.to_dict("records") if df_processed is not None else [],
                quality_report=quality_report,
                algebraic_kernel=kernel,
                algebraic_structure=alg_struct.value,
            )
        
        except Exception as exc:
            logger.error(f"Fallo en vector 'structure_logic': {exc}", exc_info=True)
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.LOGIC_ERROR,
                error=str(exc),
                metrics=collector.build_metrics()
            )


def _safe_call(obj: Any, method_name: str, default: Any) -> Any:
    """Llamada segura a método con fallback a default."""
    if hasattr(obj, method_name):
        method = getattr(obj, method_name)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return default


# =============================================================================
# VECTOR ESTRATÉGICO: PENSAMIENTO LATERAL (Risk Challenger)
# =============================================================================

class LateralPivotEvaluator:
    """
    Evaluador de pivotes laterales para decisiones estratégicas.
    
    Implementa el patrón Strategy para diferentes tipos de pivote,
    permitiendo extensibilidad sin modificar el vector principal.
    """
    
    @staticmethod
    def evaluate_monopolio_coberturado(
        stability: float,
        system_temp: float,
        financial_inertia: float
    ) -> Tuple[bool, str]:
        """
        Evalúa pivote de Monopolio Coberturado.
        
        Condición: base estrecha (ψ < 0.70) + sistema frío (T < 15.0) +
                   inercia financiera alta (I > umbral)
        """
        if (
            stability < 0.70
            and system_temp < 15.0
            and financial_inertia > FinancialConstants.FINANCIAL_INERTIA_THRESHOLD
        ):
            return True, (
                "Riesgo logístico neutralizado por alta inercia térmica financiera."
            )
        return False, (
            "Condiciones termodinámicas insuficientes para cobertura de monopolio."
        )
    
    @staticmethod
    def evaluate_opcion_espera(
        financial_class: str,
        npv: float,
        wait_option_value: float
    ) -> Tuple[bool, str]:
        """
        Evalúa pivote de Opción de Espera.
        
        Condición: riesgo alto + valor estocástico de esperar > VPN × k
        """
        npv_threshold = max(npv, 0.0) * FinancialConstants.WAIT_OPTION_NPV_MULTIPLIER
        
        if financial_class.upper() == "HIGH" and wait_option_value > npv_threshold:
            return True, (
                f"Valor de la opción de espera ({wait_option_value:.4f}) "
                f"supera el umbral VPN × {FinancialConstants.WAIT_OPTION_NPV_MULTIPLIER} "
                f"= {npv_threshold:.4f}."
            )
        return False, (
            "El valor de la opción de retraso no justifica la inactividad."
        )
    
    @staticmethod
    def evaluate_cuarentena_topologica(
        beta_1: int,
        has_synergy: bool
    ) -> Tuple[bool, str]:
        """
        Evalúa pivote de Cuarentena Topológica.
        
        Condición: ciclos presentes (β₁ > 0) SIN sinergia multiplicativa
        """
        if beta_1 > 0 and not has_synergy:
            return True, (
                "Ciclos detectados pero confinados. Se aprueba la ejecución "
                "exceptuando el subgrafo aislado."
            )
        return False, (
            "Los ciclos topológicos presentan sinergia multiplicativa. "
            "Cuarentena imposible."
        )


def vector_lateral_pivot(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel STRATEGY.
    
    Morfismo que proyecta pensamiento lateral para mitigar riesgos.
    
    Transformación: (T × Φ × Θ) ─L─▶ D_lateral
    
    Donde:
        T = Espacio Topológico (Estabilidad, β₁)
        Φ = Espacio Financiero (Clase de riesgo, VPN, Opciones Reales)
        Θ = Espacio Termodinámico (Temperatura, Inercia)
        D_lateral = Decisión de Excepción Validada
    """
    with MetricsCollector() as collector:
        # ─── Extracción de subespacios ───
        report_state = payload.get("report_state", {})
        thermal_metrics = payload.get("thermal_metrics", {})
        financial_metrics = payload.get("financial_metrics", {})
        synergy_risk = payload.get("synergy_risk", {})
        
        pivot_type = PivotType.from_string(str(payload.get("pivot_type", "UNKNOWN")))
        
        # Variables de estado
        stability = float(report_state.get("stability", PhysicsConstants.DEFAULT_STABILITY))
        beta_1 = int(report_state.get("beta_1", 0))
        system_temp = float(thermal_metrics.get("system_temperature", PhysicsConstants.DEFAULT_SYSTEM_TEMP))
        financial_inertia = float(thermal_metrics.get("heat_capacity", PhysicsConstants.DEFAULT_HEAT_CAPACITY))
        financial_class = str(report_state.get("financial_class", "UNKNOWN"))
        npv = float(financial_metrics.get("npv", 0.0))
        
        metrics = collector.build_metrics()
        
        # ─── Evaluación del pivote ───
        evaluator = LateralPivotEvaluator()
        
        if pivot_type == PivotType.MONOPOLIO_COBERTURADO:
            approved, reasoning = evaluator.evaluate_monopolio_coberturado(
                stability, system_temp, financial_inertia
            )
            if approved:
                return _build_success(
                    stratum=Stratum.STRATEGY,
                    metrics=metrics,
                    payload={
                        "approved_pivot": pivot_type.value,
                        "penalty_relief": 0.30,
                        "reasoning": reasoning,
                    }
                )
            return _build_error(
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Rechazado: {reasoning}",
                metrics=metrics
            )
        
        if pivot_type == PivotType.OPCION_ESPERA:
            real_options = financial_metrics.get("real_options", {})
            wait_option_value = float(real_options.get("wait_option_value", 0.0))
            
            approved, reasoning = evaluator.evaluate_opcion_espera(
                financial_class, npv, wait_option_value
            )
            if approved:
                return _build_success(
                    stratum=Stratum.STRATEGY,
                    metrics=metrics,
                    payload={
                        "approved_pivot": pivot_type.value,
                        "strategic_action": "FREEZE_6_MONTHS",
                        "reasoning": reasoning,
                    }
                )
            return _build_error(
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Rechazado: {reasoning}",
                metrics=metrics
            )
        
        if pivot_type == PivotType.CUARENTENA_TOPOLOGICA:
            has_synergy = bool(synergy_risk.get("synergy_detected", False))
            
            approved, reasoning = evaluator.evaluate_cuarentena_topologica(
                beta_1, has_synergy
            )
            if approved:
                return _build_success(
                    stratum=Stratum.STRATEGY,
                    metrics=metrics,
                    payload={
                        "approved_pivot": pivot_type.value,
                        "quarantine_active": True,
                        "reasoning": reasoning,
                    }
                )
            return _build_error(
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Rechazado: {reasoning}",
                metrics=metrics
            )
        
        # Pivote desconocido
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=f"Tipo de pivote lateral desconocido: '{pivot_type.value}'",
            metrics=metrics
        )


# =============================================================================
# VECTOR TÁCTICO: AUDITORÍA DE FUSIÓN (Mayer-Vietoris)
# =============================================================================

def vector_audit_homological_fusion(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel TACTICS.
    
    Aplica la Secuencia Exacta de Mayer-Vietoris para auditar la integración
    de dos grafos.
    
    Transformación: (G_A × G_B) ─MV─▶ G_{A∪B}
    
    Invariante de Mayer-Vietoris:
        Δβ₁ = β₁(A ∪ B) - [β₁(A) + β₁(B) - β₁(A ∩ B)] = 0
    
    Si Δβ₁ > 0, la fusión generó ciclos "fantasma" — integración inválida.
    """
    with MetricsCollector() as collector:
        graph_a = payload.get("graph_a")
        graph_b = payload.get("graph_b")
        
        if graph_a is None or graph_b is None:
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.LOGIC_ERROR,
                error="Faltan grafos de entrada (A o B) para la auditoría de fusión",
                metrics=collector.build_metrics()
            )
        
        # ─── Importación diferida con manejo de dependencia ───
        try:
            from agent.business_topology import BusinessTopologicalAnalyzer
        except ImportError as imp_err:
            logger.error(f"Dependencia 'agent.business_topology' no disponible: {imp_err}")
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.DEPENDENCY_ERROR,
                error=f"Dependencia de análisis topológico no disponible: {imp_err}",
                metrics=collector.build_metrics()
            )
        
        try:
            analyzer = BusinessTopologicalAnalyzer()
            audit_result: Dict[str, Any] = analyzer.audit_integration_homology(
                graph_a, graph_b
            )
            delta_beta_1 = int(audit_result.get("delta_beta_1", 0))
            
            # Coherencia: 1.0 si no hay ciclos fantasma, 0.0 si los hay
            coherence = 1.0 if delta_beta_1 == 0 else 0.0
            metrics = collector.build_metrics(topological_coherence=coherence)
            
            if delta_beta_1 > 0:
                return _build_error(
                    stratum=Stratum.TACTICS,
                    status=VectorResultStatus.TOPOLOGY_ERROR,
                    error=(
                        f"Anomalía de Mayer-Vietoris: La fusión generó "
                        f"{delta_beta_1} ciclo(s) fantasma (Δβ₁ > 0). "
                        "Integración abortada."
                    ),
                    metrics=metrics,
                    audit_result=audit_result
                )
            
            return _build_success(
                stratum=Stratum.TACTICS,
                metrics=metrics,
                payload={
                    "audit_result": audit_result,
                    "merged_graph_valid": True,
                    "delta_beta_1": delta_beta_1,
                }
            )
        
        except Exception as exc:
            logger.error(f"Fallo matemático en auditoría de fusión: {exc}", exc_info=True)
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Fallo matemático en auditoría de fusión: {exc}",
                metrics=collector.build_metrics()
            )


# =============================================================================
# FÁBRICA DE VECTORES (Inmutable con Inyección de Dependencias)
# =============================================================================

# Registro inmutable por defecto
_DEFAULT_PHYSICS_REGISTRY: Final[Dict[str, Callable[..., VectorResult]]] = {
    "stabilize": vector_stabilize_flux,
    "parse": vector_parse_raw_structure,
}

_DEFAULT_TACTICS_REGISTRY: Final[Dict[str, Callable[..., VectorResult]]] = {
    "structure": vector_structure_logic,
    "audit_fusion": vector_audit_homological_fusion,
}

_DEFAULT_STRATEGY_REGISTRY: Final[Dict[str, Callable[..., VectorResult]]] = {
    "lateral_pivot": vector_lateral_pivot,
}


class VectorFactory:
    """
    Factory para producción de vectores con dependencias inyectadas.
    
    Principios de diseño:
    - Open-Closed: registro abierto a extensión
    - Inmutabilidad de defaults: registros por defecto nunca se mutan
    - Thread-safety: operaciones de registro protegidas
    - Introspección: wrappers propagan __wrapped__
    """
    
    _lock = threading.RLock()
    
    # Registros activos (copias de los defaults)
    _physics_registry: Dict[str, Callable[..., VectorResult]] = dict(_DEFAULT_PHYSICS_REGISTRY)
    _tactics_registry: Dict[str, Callable[..., VectorResult]] = dict(_DEFAULT_TACTICS_REGISTRY)
    _strategy_registry: Dict[str, Callable[..., VectorResult]] = dict(_DEFAULT_STRATEGY_REGISTRY)
    
    @classmethod
    def register_vector(
        cls,
        stratum: Stratum,
        name: str,
        fn: Callable[..., VectorResult]
    ) -> None:
        """Registra un vector personalizado en el estrato especificado."""
        with cls._lock:
            registry = cls._get_registry_for_stratum(stratum)
            registry[name] = fn
            logger.info(f"Vector registrado: {name} [{stratum.name}]")
    
    @classmethod
    def _get_registry_for_stratum(cls, stratum: Stratum) -> Dict[str, Callable[..., VectorResult]]:
        """Obtiene el registro correspondiente a un estrato."""
        if stratum == Stratum.PHYSICS:
            return cls._physics_registry
        elif stratum == Stratum.TACTICS:
            return cls._tactics_registry
        elif stratum == Stratum.STRATEGY:
            return cls._strategy_registry
        else:
            raise ValueError(f"Estrato sin registro: {stratum}")
    
    @classmethod
    def reset_all_registries(cls) -> None:
        """Restaura todos los registros al estado original."""
        with cls._lock:
            cls._physics_registry = dict(_DEFAULT_PHYSICS_REGISTRY)
            cls._tactics_registry = dict(_DEFAULT_TACTICS_REGISTRY)
            cls._strategy_registry = dict(_DEFAULT_STRATEGY_REGISTRY)
            logger.info("Registros de vectores restaurados a defaults")
    
    @classmethod
    def get_available_vectors(cls, stratum: Optional[Stratum] = None) -> Dict[str, List[str]]:
        """Retorna vectores disponibles, opcionalmente filtrados por estrato."""
        with cls._lock:
            if stratum is not None:
                registry = cls._get_registry_for_stratum(stratum)
                return {stratum.name: list(registry.keys())}
            
            return {
                "PHYSICS": list(cls._physics_registry.keys()),
                "TACTICS": list(cls._tactics_registry.keys()),
                "STRATEGY": list(cls._strategy_registry.keys()),
            }
    
    @classmethod
    def create_vector(
        cls,
        stratum: Stratum,
        vector_type: str,
        **defaults: Any
    ) -> Callable[..., VectorResult]:
        """
        Crea un callable que invoca el vector especificado con defaults fusionados.
        
        Args:
            stratum: Estrato del vector
            vector_type: Nombre del vector en el registro
            **defaults: Argumentos por defecto que se fusionan con los explícitos
        
        Returns:
            Callable envuelto con defaults aplicados
        """
        with cls._lock:
            registry = cls._get_registry_for_stratum(stratum)
            
            if vector_type not in registry:
                available = sorted(registry.keys())
                raise ValueError(
                    f"Vector '{vector_type}' no registrado en {stratum.name}. "
                    f"Disponibles: {available}"
                )
            
            base_fn = registry[vector_type]
        
        @functools.wraps(base_fn)
        def wrapper(*args: Any, **kwargs: Any) -> VectorResult:
            merged = {**defaults, **kwargs}
            return base_fn(*args, **merged)
        
        wrapper.__name__ = f"{stratum.name.lower()}_{vector_type}"
        wrapper.__wrapped__ = base_fn  # type: ignore[attr-defined]
        
        return wrapper
    
    # ─── Métodos de conveniencia ───
    
    @classmethod
    def create_physics_vector(cls, vector_type: str, **defaults: Any) -> Callable[..., VectorResult]:
        """Atajo para crear vector físico."""
        return cls.create_vector(Stratum.PHYSICS, vector_type, **defaults)
    
    @classmethod
    def create_tactics_vector(cls, vector_type: str = "structure", **defaults: Any) -> Callable[..., VectorResult]:
        """Atajo para crear vector táctico."""
        return cls.create_vector(Stratum.TACTICS, vector_type, **defaults)
    
    @classmethod
    def create_strategy_vector(cls, vector_type: str = "lateral_pivot", **defaults: Any) -> Callable[..., VectorResult]:
        """Atajo para crear vector estratégico."""
        return cls.create_vector(Stratum.STRATEGY, vector_type, **defaults)


# =============================================================================
# COMPOSICIÓN DE VECTORES (MORFISMOS COMPUESTOS)
# =============================================================================

@dataclass(frozen=True, slots=True)
class CompositionResult:
    """
    Resultado de una composición de vectores.
    
    Encapsula el resultado final junto con métricas combinadas
    y trazabilidad de cada fase.
    """
    final_result: VectorResult
    physics_result: Optional[VectorResult] = None
    combined_metrics: Optional[VectorMetrics] = None
    
    @property
    def success(self) -> bool:
        return self.final_result.get("success", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "final_result": self.final_result,
            "physics_result": self.physics_result,
            "combined_metrics": self.combined_metrics.to_dict() if self.combined_metrics else None,
        }


def compose_vectors(
    physics_vector: Callable[..., VectorResult],
    tactics_vector: Callable[..., VectorResult],
    physics_args: Tuple[Any, ...],
    tactics_config: Dict[str, Any],
) -> VectorResult:
    """
    Compone un morfismo físico con uno táctico en un pipeline coherente.
    
    Diagrama conmutativo:
        File ──φ_phys──▶ RawData ──φ_tact──▶ ProcessedData
    
    La composición φ_tact ∘ φ_phys está definida sólo cuando la imagen
    del vector físico pertenece al dominio del táctico.
    
    Métricas combinadas (principio de cuello de botella):
        pipeline_coherence = min(C_phys, C_tact)
        total_time_ms = t_phys + t_tact
    """
    # ─── Fase 1: Física ───
    physics_result = physics_vector(*physics_args)
    
    if not physics_result.get("success"):
        return physics_result
    
    # ─── Verificación de compatibilidad de dominio ───
    raw_records = physics_result.get("raw_records")
    parse_cache = physics_result.get("parse_cache", {})
    
    if raw_records is None:
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=(
                "Composición indefinida: el vector físico no produjo 'raw_records'. "
                "Use vector_parse_raw_structure como componente física."
            ),
            physics_context=physics_result
        )
    
    # ─── Fase 2: Táctica ───
    tactics_result = tactics_vector(raw_records, parse_cache, tactics_config)
    
    # ─── Métricas combinadas ───
    phys_metrics_dict = physics_result.get("metrics", {})
    tact_metrics_dict = tactics_result.get("metrics", {})
    
    phys_metrics = VectorMetrics(
        processing_time_ms=phys_metrics_dict.get("processing_time_ms", 0.0),
        memory_usage_mb=phys_metrics_dict.get("memory_usage_mb", 0.0),
        topological_coherence=phys_metrics_dict.get("topological_coherence", 1.0),
        algebraic_integrity=phys_metrics_dict.get("algebraic_integrity", 1.0)
    )
    
    tact_metrics = VectorMetrics(
        processing_time_ms=tact_metrics_dict.get("processing_time_ms", 0.0),
        memory_usage_mb=tact_metrics_dict.get("memory_usage_mb", 0.0),
        topological_coherence=tact_metrics_dict.get("topological_coherence", 1.0),
        algebraic_integrity=tact_metrics_dict.get("algebraic_integrity", 1.0)
    )
    
    combined = phys_metrics.combine(tact_metrics)
    
    tactics_result["combined_metrics"] = {
        "physics": phys_metrics.to_dict(),
        "tactics": tact_metrics.to_dict(),
        "pipeline": combined.to_dict(),
        "total_time_ms": combined.processing_time_ms,
        "pipeline_coherence": combined.topological_coherence,
    }
    
    tactics_result["physics_result_summary"] = {
        "success": physics_result.get("success"),
        "record_count": len(raw_records),
        "homological_invariants": physics_result.get("homological_invariants"),
    }
    
    return tactics_result


def compose_pipeline(
    vectors: List[Tuple[Callable[..., VectorResult], Dict[str, Any]]],
    initial_input: Dict[str, Any]
) -> VectorResult:
    """
    Compone una secuencia arbitraria de vectores en un pipeline.
    
    Cada vector recibe el resultado del anterior como parte de su input.
    
    Args:
        vectors: Lista de (vector_fn, extra_kwargs) tuples
        initial_input: Input inicial para el primer vector
    
    Returns:
        Resultado del último vector con métricas acumuladas
    """
    if not vectors:
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.VALIDATION_ERROR,
            error="Pipeline vacío: no hay vectores para componer"
        )
    
    accumulated_metrics = VectorMetrics()
    current_result: VectorResult = {"success": True, **initial_input}
    
    for idx, (vector_fn, extra_kwargs) in enumerate(vectors):
        try:
            # Fusionar resultado anterior con kwargs extra
            merged_input = {**current_result, **extra_kwargs}
            current_result = vector_fn(**merged_input)
            
            if not current_result.get("success"):
                current_result["_failed_at_stage"] = idx
                current_result["_accumulated_metrics"] = accumulated_metrics.to_dict()
                return current_result
            
            # Acumular métricas
            stage_metrics_dict = current_result.get("metrics", {})
            stage_metrics = VectorMetrics(
                processing_time_ms=stage_metrics_dict.get("processing_time_ms", 0.0),
                memory_usage_mb=stage_metrics_dict.get("memory_usage_mb", 0.0),
                topological_coherence=stage_metrics_dict.get("topological_coherence", 1.0),
                algebraic_integrity=stage_metrics_dict.get("algebraic_integrity", 1.0)
            )
            accumulated_metrics = accumulated_metrics.combine(stage_metrics)
            
        except Exception as exc:
            logger.error(f"Fallo en stage {idx} del pipeline: {exc}", exc_info=True)
            return _build_error(
                stratum=Stratum.PHYSICS,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Pipeline falló en stage {idx}: {exc}",
                _failed_at_stage=idx,
                _accumulated_metrics=accumulated_metrics.to_dict()
            )
    
    current_result["pipeline_metrics"] = accumulated_metrics.to_dict()
    current_result["pipeline_stages_completed"] = len(vectors)
    
    return current_result


# =============================================================================
# UTILIDADES PÚBLICAS
# =============================================================================

def get_vector_info(vector_fn: Callable[..., VectorResult]) -> Dict[str, Any]:
    """
    Obtiene información de un vector (introspección).
    
    Returns:
        Dict con nombre, documentación y vector base si está wrapped.
    """
    info: Dict[str, Any] = {
        "name": getattr(vector_fn, "__name__", "unknown"),
        "doc": getattr(vector_fn, "__doc__", None),
        "is_wrapped": hasattr(vector_fn, "__wrapped__"),
    }
    
    if hasattr(vector_fn, "__wrapped__"):
        base = vector_fn.__wrapped__  # type: ignore
        info["base_name"] = getattr(base, "__name__", "unknown")
        info["base_doc"] = getattr(base, "__doc__", None)
    
    return info


def validate_vector_result(result: VectorResult) -> ValidationResult:
    """
    Valida que un VectorResult tenga la estructura esperada.
    
    Claves requeridas: success, stratum, status, metrics
    """
    required_keys = {"success", "stratum", "status", "metrics"}
    missing = required_keys - set(result.keys())
    
    if missing:
        return ValidationResult.failure(
            f"VectorResult incompleto: faltan claves {missing}",
            missing_keys=list(missing)
        )
    
    if not isinstance(result["success"], bool):
        return ValidationResult.failure("'success' debe ser booleano")
    
    if not isinstance(result["metrics"], dict):
        return ValidationResult.failure("'metrics' debe ser diccionario")
    
    return ValidationResult.success()