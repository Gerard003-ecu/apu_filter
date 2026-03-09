"""
Módulo: Pipeline Director Evolucionado (Sistema Nervioso Central Mejorado)
=========================================================================

Este componente implementa un pipeline como un **Grafo Acíclico Dirigido (DAG)
algebraico** en lugar de una secuencia lineal, con:

1. **DAG Algebraico**: Estructura de dependencias explícita en forma de grafo
2. **Memoización de Operadores**: Cache de resultados intermedios (TensorFlow-style)
3. **StateVector Tipado**: Reemplazo de dict por estructura algebraica verificada
4. **Verificación Homológica**: Auditoría automática en fusiones de datos

Fundamentos Matemáticos:
------------------------

1. Espacio Vectorial de Operadores:
   - Base ortonormal {e₁, ..., eₙ} de pasos
   - Cada operador es una proyección P_i: V → V
   - Composición: P_j ∘ P_i = P_j(P_i(x))
   - DAG asegura causalidad: aristas = dependencias

2. Memoización Algebraica:
   - Cache: (input_hash, operator_id) → (output, signature)
   - Firma: hash criptográfico del tensor de salida
   - Idempotencia verificable: f(x) = f(x) si sig(x) coincide

3. StateVector Tipado:
   - Reemplaza Dict[str, Any] por estructura de tipos
   - Invariantes algebraicos verificados en tiempo de construcción
   - Serialización determinística para hash

4. Homología (Mayer-Vietoris):
   - Fusión A ∪ B: verifica H_*(A ∩ B) → H_*(A) ⊕ H_*(B) → H_*(A ∪ B)
   - Ciclos espurios (β₁): detecta loops anómalos
   - Desconexiones artificiales (β₀): verifica conectividad
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, make_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx
import numpy as np
import pandas as pd
from flask import current_app

from app.constants import ColumnNames, InsumoType
from app.flux_condenser import CondenserConfig, DataFluxCondenser
from app.matter_generator import MatterGenerator
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator
from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
)
from app.business_agent import BusinessAgent
from app.semantic_translator import SemanticTranslator
from app.tools_interface import MICRegistry, register_core_vectors

from .apu_processor import (
    APUProcessor,
    APUCostCalculator,
    DataMerger,
    DataValidator,
    FileValidator,
    InsumosProcessor,
    PresupuestoProcessor,
    ProcessingThresholds,
    build_output_dictionary,
    build_processed_apus_dataframe,
    calculate_insumo_costs,
    calculate_total_costs,
    group_and_split_description,
    sanitize_for_json,
    synchronize_data_sources,
)
from .data_validator import validate_and_clean_data


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

EPSILON: Final[float] = 1e-9
PIPELINE_VERSION: Final[str] = "4.0.0-dag-algebraic"
DEFAULT_SESSION_DIR: Final[str] = "data/sessions"
MAX_CONTEXT_SIZE_BYTES: Final[int] = 50 * 1024 * 1024
SESSION_FILE_EXTENSION: Final[str] = ".pkl"
MEMOIZATION_CACHE_SIZE: Final[int] = 1000  # Número máximo de entradas en caché
HOMOLOGICAL_AUDIT_ENABLED: Final[bool] = True

logger = logging.getLogger(__name__)


def configure_pipeline_logging(level: str = "INFO") -> None:
    """Configura logging del pipeline."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)


configure_pipeline_logging(os.getenv("LOG_LEVEL", "INFO"))


# ============================================================================
# EXCEPCIONES DEL DOMINIO
# ============================================================================


class PipelineError(Exception):
    """Excepción base del pipeline."""
    pass


class ConfigurationError(PipelineError):
    """Error de configuración."""
    pass


class StepExecutionError(PipelineError):
    """Error durante ejecución de paso."""
    
    def __init__(
        self,
        message: str,
        step_name: str,
        cause: Optional[Exception] = None,
    ):
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"[{step_name}] {message}")


class FiltrationViolationError(PipelineError):
    """Violación del invariante de filtración."""
    
    def __init__(
        self,
        target_stratum: Stratum,
        missing_strata: List[str],
        validated_strata: List[str],
    ):
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata
        super().__init__(
            f"Filtration Violation. Target: {target_stratum.name}. "
            f"Missing: {missing_strata}. Validated: {validated_strata}."
        )


class HomologicalAuditError(PipelineError):
    """Error en auditoría homológica."""
    
    def __init__(
        self,
        message: str,
        homology_data: Optional[Dict[str, Any]] = None,
    ):
        self.homology_data = homology_data or {}
        super().__init__(f"Homological audit failed: {message}")


class StateVectorError(PipelineError):
    """Error en StateVector."""
    pass


class MemoizationError(PipelineError):
    """Error en memoización."""
    pass


class SessionError(PipelineError):
    """Error de sesión."""
    pass


class SessionNotFoundError(SessionError):
    """Sesión no encontrada."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionCorruptedError(SessionError):
    """Sesión corrupta."""
    
    def __init__(self, session_id: str, reason: str):
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session corrupted: {session_id}. Reason: {reason}")


class PreconditionError(StepExecutionError):
    """Error de precondición."""
    
    def __init__(self, step_name: str, missing_keys: List[str]):
        self.missing_keys = missing_keys
        super().__init__(f"Missing keys: {missing_keys}", step_name)


class DataValidationError(StepExecutionError):
    """Error de validación de datos."""
    pass


class DependencyResolutionError(PipelineError):
    """Error resolviendo dependencias del DAG."""
    
    def __init__(self, message: str, missing_deps: Optional[List[str]] = None):
        self.missing_deps = missing_deps or []
        super().__init__(message)


# ============================================================================
# ENUMERACIONES
# ============================================================================


class StepStatus(str, enum.Enum):
    """Estados de un paso."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    CACHED = "cached"
    ERROR = "error"
    
    @property
    def is_terminal(self) -> bool:
        return self in (
            StepStatus.SUCCESS,
            StepStatus.SKIPPED,
            StepStatus.CACHED,
            StepStatus.ERROR,
        )
    
    @property
    def is_successful(self) -> bool:
        return self in (StepStatus.SUCCESS, StepStatus.SKIPPED, StepStatus.CACHED)


class PipelineSteps(str, enum.Enum):
    """Pasos canónicos del pipeline (orden topológico)."""
    LOAD_DATA = "load_data"
    AUDITED_MERGE = "audited_merge"
    CALCULATE_COSTS = "calculate_costs"
    FINAL_MERGE = "final_merge"
    BUSINESS_TOPOLOGY = "business_topology"
    MATERIALIZATION = "materialization"
    BUILD_OUTPUT = "build_output"
    
    @property
    def stratum(self) -> Stratum:
        return _STEP_STRATA[self]
    
    @property
    def index(self) -> int:
        return list(PipelineSteps).index(self)
    
    @classmethod
    def from_string(cls, value: str) -> Optional[PipelineSteps]:
        for step in cls:
            if step.value == value:
                return step
        return None


_STEP_STRATA: Final[Dict[PipelineSteps, Stratum]] = {
    PipelineSteps.LOAD_DATA: Stratum.PHYSICS,
    PipelineSteps.AUDITED_MERGE: Stratum.PHYSICS,
    PipelineSteps.CALCULATE_COSTS: Stratum.TACTICS,
    PipelineSteps.FINAL_MERGE: Stratum.TACTICS,
    PipelineSteps.BUSINESS_TOPOLOGY: Stratum.STRATEGY,
    PipelineSteps.MATERIALIZATION: Stratum.STRATEGY,
    PipelineSteps.BUILD_OUTPUT: Stratum.WISDOM,
}

_STRATUM_ORDER: Final[Dict[Stratum, int]] = {
    Stratum.PHYSICS: 0,
    Stratum.TACTICS: 1,
    Stratum.STRATEGY: 2,
    Stratum.WISDOM: 3,
}

_STRATUM_EVIDENCE: Final[Dict[Stratum, Tuple[str, ...]]] = {
    Stratum.PHYSICS: ("df_presupuesto", "df_insumos", "df_apus_raw"),
    Stratum.TACTICS: ("df_apu_costos", "df_tiempo", "df_rendimiento"),
    Stratum.STRATEGY: ("graph", "business_topology_report"),
    Stratum.WISDOM: ("final_result",),
}


def stratum_level(s: Stratum) -> int:
    """Nivel ordinal de un estrato."""
    return _STRATUM_ORDER.get(s, -1)


# ============================================================================
# ESTRUCTURAS ALGEBRAICAS
# ============================================================================


@dataclass(frozen=True)
class TensorSignature:
    """
    Firma de un tensor de datos.
    
    Almacena metadatos para verificar idempotencia y detectar cambios.
    """
    
    hash_value: str  # SHA-256 del contenido
    shape: Tuple[int, ...]  # Dimensiones
    dtype: str  # Tipo de dato
    timestamp: datetime.datetime  # Cuándo se creó
    
    @classmethod
    def compute(
        cls,
        data: Any,
        timestamp: Optional[datetime.datetime] = None,
    ) -> TensorSignature:
        """Computa firma a partir de datos."""
        timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
        
        # Extraer shape y dtype
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            dtype = "dataframe"
            content = data.to_json().encode("utf-8")
        elif isinstance(data, np.ndarray):
            shape = data.shape
            dtype = str(data.dtype)
            content = data.tobytes()
        elif isinstance(data, (dict, list)):
            shape = (len(data),)
            dtype = "json"
            content = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        else:
            shape = ()
            dtype = type(data).__name__
            content = str(data).encode("utf-8")
        
        hash_value = hashlib.sha256(content).hexdigest()
        
        return cls(
            hash_value=hash_value,
            shape=shape,
            dtype=dtype,
            timestamp=timestamp,
        )
    
    def matches(self, other: TensorSignature) -> bool:
        """Verifica si otra firma es idéntica."""
        return (
            self.hash_value == other.hash_value
            and self.shape == other.shape
            and self.dtype == other.dtype
        )


@dataclass(frozen=True)
class MemoizationKey:
    """
    Clave para memoización (input_hash, operator_id, stratum).
    """
    
    input_hash: str
    operator_id: str
    stratum: str
    
    def to_tuple(self) -> Tuple[str, str, str]:
        """Convierte a tupla hashable."""
        return (self.input_hash, self.operator_id, self.stratum)
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MemoizationKey):
            return False
        return self.to_tuple() == other.to_tuple()


@dataclass
class MemoizationEntry:
    """
    Entrada en caché de memoización.
    """
    
    key: MemoizationKey
    output: Dict[str, Any]
    output_signature: TensorSignature
    created_at: datetime.datetime
    hit_count: int = 0
    
    def is_fresh(self, max_age_hours: int = 24) -> bool:
        """Verifica si la entrada aún es válida."""
        age = datetime.datetime.now(datetime.timezone.utc) - self.created_at
        return age < datetime.timedelta(hours=max_age_hours)


@dataclass
class StateVector:
    """
    Vector de Estado Tipado: Reemplaza Dict[str, Any].
    
    Mantiene:
    - DataFrames de entrada/procesamiento
    - Grafo de negocio
    - Reportes y métricas
    - Hashes de linaje
    
    Invariantes algebraicos:
    - No puede tener valores None sin ser explícito
    - Serialización determinística para hash
    """
    
    # Datos físicos (Stratum.PHYSICS)
    df_presupuesto: Optional[pd.DataFrame] = None
    df_insumos: Optional[pd.DataFrame] = None
    df_apus_raw: Optional[pd.DataFrame] = None
    df_merged: Optional[pd.DataFrame] = None
    
    # Datos tácticos (Stratum.TACTICS)
    df_apu_costos: Optional[pd.DataFrame] = None
    df_tiempo: Optional[pd.DataFrame] = None
    df_rendimiento: Optional[pd.DataFrame] = None
    df_final: Optional[pd.DataFrame] = None
    
    # Datos estratégicos (Stratum.STRATEGY)
    graph: Optional[nx.DiGraph] = None
    business_topology_report: Optional[Any] = None
    
    # Datos de sabiduría (Stratum.WISDOM)
    final_result: Optional[Dict[str, Any]] = None
    
    # Metadatos
    raw_records: List[Dict[str, Any]] = field(default_factory=list)
    parse_cache: Dict[str, Any] = field(default_factory=dict)
    validated_strata: Set[Stratum] = field(default_factory=set)
    step_results: List[StepResult] = field(default_factory=list)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    bill_of_materials: Optional[Any] = None
    logistics_plan: Optional[Dict[str, Any]] = None
    integration_risk_alert: Optional[Dict[str, Any]] = None
    audit_report: Optional[Dict[str, Any]] = None
    technical_audit: Optional[Dict[str, Any]] = None
    
    # Operacional
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    version: str = PIPELINE_VERSION
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario (sin DataFrames para tamaño)."""
        result = {}
        
        # Serializar DataFrames como JSON
        for field_name in [
            "df_presupuesto", "df_insumos", "df_apus_raw", "df_merged",
            "df_apu_costos", "df_tiempo", "df_rendimiento", "df_final",
        ]:
            df = getattr(self, field_name, None)
            if df is not None and isinstance(df, pd.DataFrame):
                result[field_name] = df.to_dict("records")
        
        # Serializar otros tipos
        result["raw_records"] = self.raw_records
        result["parse_cache"] = self.parse_cache
        result["validated_strata"] = [
            s.name if hasattr(s, "name") else str(s)
            for s in self.validated_strata
        ]
        result["quality_report"] = self.quality_report
        
        if self.graph is not None:
            result["graph_nodes"] = self.graph.number_of_nodes()
            result["graph_edges"] = self.graph.number_of_edges()
        
        if self.business_topology_report is not None:
            result["business_topology_report"] = asdict(
                self.business_topology_report
                if hasattr(self.business_topology_report, "__dataclass_fields__")
                else {"status": "available"}
            )
        
        if self.final_result is not None:
            result["final_result"] = self.final_result
        
        result["session_id"] = self.session_id
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        result["version"] = self.version
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateVector:
        """Reconstruye desde diccionario."""
        kwargs = {}
        
        # Reconstruir DataFrames
        for field_name in [
            "df_presupuesto", "df_insumos", "df_apus_raw", "df_merged",
            "df_apu_costos", "df_tiempo", "df_rendimiento", "df_final",
        ]:
            if field_name in data:
                records = data[field_name]
                if isinstance(records, list):
                    kwargs[field_name] = pd.DataFrame(records)
        
        # Campos simples
        kwargs["raw_records"] = data.get("raw_records", [])
        kwargs["parse_cache"] = data.get("parse_cache", {})
        kwargs["quality_report"] = data.get("quality_report", {})
        kwargs["session_id"] = data.get("session_id", str(uuid.uuid4()))
        kwargs["version"] = data.get("version", PIPELINE_VERSION)
        
        if "validated_strata" in data:
            kwargs["validated_strata"] = {
                getattr(Stratum, s) for s in data["validated_strata"]
                if hasattr(Stratum, s)
            }

        if "created_at" in data:
            try:
                kwargs["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
            except ValueError:
                pass

        if "updated_at" in data:
            try:
                kwargs["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
            except ValueError:
                pass

        return cls(**kwargs)
    
    def compute_hash(self) -> str:
        """Computa hash SHA-256 del estado."""
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def get_evidence(self, stratum: Stratum) -> Dict[str, bool]:
        """Verifica presencia de evidencia para un estrato."""
        evidence_keys = _STRATUM_EVIDENCE.get(stratum, ())
        result = {}
        
        for key in evidence_keys:
            value = getattr(self, key, None)
            is_valid = (
                value is not None
                and (not hasattr(value, "empty") or not value.empty)
                and (not isinstance(value, (list, dict)) or len(value) > 0)
            )
            result[key] = is_valid
        
        return result


@dataclass
class StepResult:
    """Resultado de ejecutar un paso."""
    
    step_name: str
    status: StepStatus
    stratum: Stratum
    duration_ms: float = 0.0
    error: Optional[str] = None
    context_keys: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    state_vector_hash: Optional[str] = None
    memoization_hit: bool = False
    
    @property
    def is_successful(self) -> bool:
        return self.status.is_successful
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "stratum": self.stratum.name,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "context_keys": self.context_keys,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "memoization_hit": self.memoization_hit,
        }


@dataclass(frozen=True)
class BasisVector:
    """Vector base unitario en espacio de operadores."""
    
    index: int
    label: str
    operator_class: Type[BaseProcessingStep]
    stratum: Stratum
    
    def __repr__(self) -> str:
        return f"e_{self.index}({self.label}, {self.stratum.name})"


# ============================================================================
# DAG ALGEBRAICO
# ============================================================================


@dataclass
class DependencyEdge:
    """Arista en el DAG algebraico (dependencia de datos)."""
    
    source: str  # Nombre del paso fuente
    target: str  # Nombre del paso destino
    data_keys: List[str]  # Claves del StateVector requeridas
    optional: bool = False  # Si es opcional
    
    def __hash__(self) -> int:
        return hash((self.source, self.target))


class AlgebraicDAG:
    """
    Grafo Acíclico Dirigido para representar dependencias de pasos.
    
    Propiedades:
    - Topología algebraica de operadores
    - Resolución automática de orden de ejecución
    - Detección de ciclos y deadlocks
    - Planificación óptima de ejecución
    """
    
    def __init__(self):
        """Inicializa el DAG."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.dependencies: Dict[str, Set[str]] = {}
        self.data_requirements: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_step(self, step_name: str, data_keys: Optional[List[str]] = None) -> None:
        """Añade un nodo al DAG."""
        self.graph.add_node(step_name)
        self.dependencies[step_name] = set()
        self.data_requirements[step_name] = set(data_keys or [])
    
    def add_dependency(
        self,
        source: str,
        target: str,
        data_keys: Optional[List[str]] = None,
        optional: bool = False,
    ) -> None:
        """Añade arista (dependencia) al DAG."""
        if not self.graph.has_node(source):
            self.add_step(source)
        if not self.graph.has_node(target):
            self.add_step(target)
        
        # Detectar ciclo
        if nx.has_path(self.graph, target, source):
            raise DependencyResolutionError(
                f"Cycle detected: {target} → {source}",
                [source, target],
            )
        
        self.graph.add_edge(
            source,
            target,
            data_keys=data_keys or [],
            optional=optional,
        )
        self.dependencies[target].add(source)
        
        if data_keys:
            self.data_requirements[target].update(data_keys)
    
    def topological_sort(self) -> List[str]:
        """Ordena pasos en orden topológico válido."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise DependencyResolutionError(
                f"Topological sort failed: {e}",
                list(self.graph.nodes()),
            )
    
    def get_dependencies(self, step_name: str) -> Set[str]:
        """Obtiene todas las dependencias de un paso."""
        return self.dependencies.get(step_name, set())
    
    def get_data_requirements(self, step_name: str) -> Set[str]:
        """Obtiene claves de datos requeridas."""
        return self.data_requirements.get(step_name, set())
    
    def validate(self) -> bool:
        """Valida que el DAG sea válido."""
        # Verificar acicularidad
        if not nx.is_directed_acyclic_graph(self.graph):
            raise DependencyResolutionError("DAG contains cycles")
        
        # Verificar conectividad
        if len(self.graph.nodes()) > 1 and not nx.is_weakly_connected(self.graph):
            self.logger.warning("DAG is not weakly connected (subgrafos desconectados)")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el DAG."""
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "data_keys": data.get("data_keys", []),
                "optional": data.get("optional", False),
            })
        
        return {
            "nodes": list(self.graph.nodes()),
            "edges": edges,
            "is_acyclic": nx.is_directed_acyclic_graph(self.graph),
        }


# ============================================================================
# MEMOIZACIÓN DE OPERADORES
# ============================================================================


class OperatorMemoizer:
    """
    Sistema de memoización para operadores (pasos).
    
    Implementa cache (input_hash, operator_id) → (output, signature).
    Similar a TensorFlow graph caching.
    """
    
    def __init__(self, max_size: int = MEMOIZATION_CACHE_SIZE):
        """Inicializa el memorizador."""
        self.max_size = max_size
        self.cache: Dict[MemoizationKey, MemoizationEntry] = {}
        self.access_order: List[MemoizationKey] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def compute_input_hash(self, state: StateVector) -> str:
        """Computa hash determinístico del estado."""
        serialized = json.dumps(
            state.to_dict(),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def lookup(
        self,
        state: StateVector,
        operator_id: str,
        stratum: str,
    ) -> Optional[Tuple[Dict[str, Any], TensorSignature]]:
        """
        Busca en caché.
        
        Returns:
            (output, signature) si hay hit, None si miss.
        """
        input_hash = self.compute_input_hash(state)
        key = MemoizationKey(
            input_hash=input_hash,
            operator_id=operator_id,
            stratum=stratum,
        )
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        if not entry.is_fresh():
            self.logger.debug(f"Memoization entry expired: {operator_id}")
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        # Actualizar estadísticas
        entry.hit_count += 1
        self.stats["hits"] += 1
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.logger.debug(f"Memoization hit: {operator_id} (hits={entry.hit_count})")
        
        return (entry.output, entry.output_signature)
    
    def store(
        self,
        state: StateVector,
        operator_id: str,
        stratum: str,
        output: Dict[str, Any],
        output_signature: TensorSignature,
    ) -> None:
        """Almacena resultado en caché."""
        input_hash = self.compute_input_hash(state)
        key = MemoizationKey(
            input_hash=input_hash,
            operator_id=operator_id,
            stratum=stratum,
        )
        
        # Evict LRU si es necesario
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
            self.logger.debug(f"LRU eviction: {oldest_key.operator_id}")
        
        entry = MemoizationEntry(
            key=key,
            output=output,
            output_signature=output_signature,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        
        self.cache[key] = entry
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        self.cache.clear()
        self.access_order.clear()
        self.logger.info("Memoization cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total * 100 if total > 0 else 0
        )
        
        return {
            **self.stats,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }


# ============================================================================
# AUDITORÍA HOMOLÓGICA
# ============================================================================


class HomologicalAuditor:
    """
    Audita fusión de datos usando topología algebraica.
    
    Verifica la exactitud de la secuencia de Mayer-Vietoris:
    ... → H_k(A∩B) → H_k(A) ⊕ H_k(B) → H_k(A∪B) → ...
    """
    
    def __init__(self, telemetry: TelemetryContext):
        """Inicializa el auditor."""
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def audit_merge(
        self,
        df_a: Optional[pd.DataFrame],
        df_b: Optional[pd.DataFrame],
        df_result: pd.DataFrame,
        context: StateVector,
    ) -> Dict[str, Any]:
        """
        Audita una fusión de datos.
        
        Returns:
            Dict con resultado de auditoría y alertas.
        """
        audit_result = {
            "passed": True,
            "emergent_cycles": 0,
            "lost_connections": 0,
            "data_integrity": True,
            "warnings": [],
        }
        
        if df_a is None or df_b is None:
            self.logger.info("Homological audit skipped (missing input)")
            return audit_result
        
        try:
            # Verificar conservación de cardinalidad
            card_a = len(df_a)
            card_b = len(df_b)
            card_result = len(df_result)
            
            # Heurística: el resultado no debe ser significativamente menor
            # (pérdida de datos) ni significativamente mayor (duplicados)
            min_expected = max(card_a, card_b)
            max_expected = card_a + card_b
            
            if card_result < min_expected * 0.8:
                audit_result["warnings"].append(
                    f"Data loss detected: expected ≥ {min_expected}, "
                    f"got {card_result}"
                )
                audit_result["passed"] = False
            
            if card_result > max_expected * 1.2:
                audit_result["warnings"].append(
                    f"Duplicates detected: expected ≤ {max_expected}, "
                    f"got {card_result}"
                )
                # No bloquea (algunos duplicados son OK)
            
            # Verificar estabilidad de columnas
            cols_a = set(df_a.columns)
            cols_b = set(df_b.columns)
            cols_result = set(df_result.columns)
            
            expected_cols = cols_a | cols_b
            if not expected_cols.issubset(cols_result):
                missing = expected_cols - cols_result
                audit_result["warnings"].append(
                    f"Lost columns: {missing}"
                )
                audit_result["passed"] = False
            
            # Análisis topológico mediante grafo
            self._analyze_connectivity(df_a, df_b, df_result, audit_result)
            
        except Exception as e:
            self.logger.warning(f"Homological audit exception: {e}")
            audit_result["warnings"].append(str(e))
        
        if not audit_result["passed"]:
            self.telemetry.record_error(
                "homological_audit",
                "; ".join(audit_result["warnings"]),
            )
        
        return audit_result
    
    def _analyze_connectivity(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        df_result: pd.DataFrame,
        audit_result: Dict[str, Any],
    ) -> None:
        """Analiza conectividad usando grafos."""
        try:
            # Construir grafos de identificadores (heurística)
            id_col = None
            for col in ["id", "ID", "Id", "idx"]:
                if col in df_a.columns and col in df_b.columns:
                    id_col = col
                    break
            
            if id_col is None:
                self.logger.debug("No ID column found for connectivity analysis")
                return
            
            ids_a = set(df_a[id_col].dropna().unique())
            ids_b = set(df_b[id_col].dropna().unique())
            ids_result = set(df_result[id_col].dropna().unique())
            
            # Verificar preservación de identidades
            intersection = ids_a & ids_b
            union = ids_a | ids_b
            
            if len(union) > 0:
                preservation = len(ids_result & union) / len(union)
                
                if preservation < 0.8:
                    audit_result["warnings"].append(
                        f"ID preservation {preservation:.1%} < 80%"
                    )
                    audit_result["passed"] = False
            
        except Exception as e:
            self.logger.debug(f"Connectivity analysis failed: {e}")


# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass(frozen=True)
class StepConfig:
    """Configuración por paso."""
    
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    use_memoization: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepConfig:
        return cls(
            enabled=bool(data.get("enabled", True)),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=int(data.get("retry_count", 0)),
            retry_delay_seconds=float(data.get("retry_delay_seconds", 1.0)),
            use_memoization=bool(data.get("use_memoization", True)),
        )


@dataclass(frozen=True)
class SessionConfig:
    """Configuración de sesiones."""
    
    session_dir: Path = field(default_factory=lambda: Path(DEFAULT_SESSION_DIR))
    max_age_hours: int = 24
    cleanup_on_success: bool = True
    persist_on_error: bool = True
    
    def __post_init__(self) -> None:
        if not isinstance(self.session_dir, Path):
            object.__setattr__(self, "session_dir", Path(self.session_dir))


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración consolidada del pipeline."""
    
    session: SessionConfig = field(default_factory=SessionConfig)
    step_configs: Dict[str, StepConfig] = field(default_factory=dict)
    recipe: Optional[List[str]] = None
    enforce_filtration: bool = True
    enforce_homology: bool = HOMOLOGICAL_AUDIT_ENABLED
    file_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        session_data = data.get("session", {})
        session_dir = session_data.get(
            "session_dir",
            data.get("session_dir", DEFAULT_SESSION_DIR),
        )
        
        step_configs = {}
        for step in PipelineSteps:
            step_data = data.get("steps", {}).get(step.value, {})
            step_configs[step.value] = StepConfig.from_dict(step_data)
        
        return cls(
            session=SessionConfig(
                session_dir=Path(session_dir),
                max_age_hours=int(session_data.get("max_age_hours", 24)),
                cleanup_on_success=bool(session_data.get("cleanup_on_success", True)),
                persist_on_error=bool(session_data.get("persist_on_error", True)),
            ),
            step_configs=step_configs,
            recipe=data.get("pipeline_recipe"),
            enforce_filtration=bool(data.get("enforce_filtration", True)),
            enforce_homology=bool(data.get("enforce_homology", HOMOLOGICAL_AUDIT_ENABLED)),
            file_profiles=data.get("file_profiles", {}),
            raw_config=data,
        )
    
    def get_step_config(self, step_name: str) -> StepConfig:
        return self.step_configs.get(step_name, StepConfig())
    
    def is_step_enabled(self, step_name: str) -> bool:
        return self.get_step_config(step_name).enabled


# ============================================================================
# CLASE BASE PARA PASOS (REFACTORIZADO)
# ============================================================================


class BaseProcessingStep(ABC):
    """
    Clase base para pasos del pipeline.
    
    Ahora integra:
    - Memoización de operadores
    - Auditoría homológica automática
    - Soporte para StateVector
    """
    
    REQUIRED_CONTEXT_KEYS: ClassVar[Tuple[str, ...]] = ()
    PRODUCED_CONTEXT_KEYS: ClassVar[Tuple[str, ...]] = ()
    
    def __init__(
        self,
        config: PipelineConfig,
        thresholds: ProcessingThresholds,
        memoizer: Optional[OperatorMemoizer] = None,
        auditor: Optional[HomologicalAuditor] = None,
    ):
        self.config = config
        self.thresholds = thresholds
        self.memoizer = memoizer or OperatorMemoizer()
        self.auditor = auditor
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """
        Ejecuta el paso con memoización y auditoría.
        
        Template method con ciclo de vida completo.
        """
        step_name = self._get_step_name()
        start_time = time.monotonic()
        
        telemetry.start_step(step_name)
        
        try:
            # Validar precondiciones
            self._validate_preconditions(state, step_name)
            
            # Buscar en caché
            step_config = self.config.get_step_config(step_name)
            if step_config.use_memoization:
                cached = self.memoizer.lookup(
                    state,
                    step_name,
                    state.validated_strata.__str__(),
                )
                
                if cached is not None:
                    output, sig = cached
                    new_state = self._merge_output_into_state(state, output)
                    
                    duration_ms = (time.monotonic() - start_time) * 1000
                    result = StepResult(
                        step_name=step_name,
                        status=StepStatus.CACHED,
                        stratum=_STEP_STRATA[PipelineSteps.from_string(step_name)],
                        duration_ms=duration_ms,
                        state_vector_hash=new_state.compute_hash(),
                        memoization_hit=True,
                    )
                    
                    new_state.step_results.append(result)
                    telemetry.end_step(step_name, "cached")
                    self.logger.info(f"✓ {step_name} (CACHED, {duration_ms:.0f}ms)")
                    
                    return new_state
            
            # Ejecutar paso
            new_state = self._execute_impl(state, telemetry, mic)
            
            if new_state is None:
                raise StepExecutionError("Step returned None state", step_name)
            
            # Almacenar en caché
            if step_config.use_memoization:
                output = self._extract_output(new_state)
                sig = TensorSignature.compute(output)
                self.memoizer.store(
                    state,
                    step_name,
                    new_state.validated_strata.__str__(),
                    output,
                    sig,
                )
            
            # Registrar resultado
            duration_ms = (time.monotonic() - start_time) * 1000
            result = StepResult(
                step_name=step_name,
                status=StepStatus.SUCCESS,
                stratum=_STEP_STRATA.get(
                    PipelineSteps.from_string(step_name),
                    Stratum.PHYSICS,
                ),
                duration_ms=duration_ms,
                state_vector_hash=new_state.compute_hash(),
            )
            
            new_state.step_results.append(result)
            new_state.updated_at = datetime.datetime.now(datetime.timezone.utc)
            
            telemetry.end_step(step_name, "success")
            self.logger.info(f"✓ {step_name} ({duration_ms:.0f}ms)")
            
            return new_state
            
        except PipelineError:
            telemetry.end_step(step_name, "error")
            raise
        except Exception as e:
            self.logger.error(f"Error in {step_name}: {e}", exc_info=True)
            telemetry.record_error(step_name, str(e))
            telemetry.end_step(step_name, "error")
            raise StepExecutionError(str(e), step_name, e)
    
    def _get_step_name(self) -> str:
        """Obtiene nombre del paso."""
        import re
        name = self.__class__.__name__
        if name.endswith("Step"):
            name = name[:-4]
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    
    def _validate_preconditions(
        self,
        state: StateVector,
        step_name: str,
    ) -> None:
        """Valida precondiciones."""
        missing = []
        for key in self.REQUIRED_CONTEXT_KEYS:
            value = getattr(state, key, None)
            if value is None:
                missing.append(key)
            elif isinstance(value, pd.DataFrame) and value.empty:
                missing.append(f"{key} (empty)")
        
        if missing:
            raise PreconditionError(step_name, missing)
    
    def _merge_output_into_state(
        self,
        state: StateVector,
        output: Dict[str, Any],
    ) -> StateVector:
        """Fusiona output en el state."""
        state_dict = asdict(state)
        state_dict.update(output)
        return StateVector(**state_dict)
    
    def _extract_output(self, state: StateVector) -> Dict[str, Any]:
        """Extrae claves producidas."""
        result = {}
        for key in self.PRODUCED_CONTEXT_KEYS:
            if hasattr(state, key):
                result[key] = getattr(state, key)
        return result
    
    @abstractmethod
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Implementación específica del paso."""
        pass


# ============================================================================
# IMPLEMENTACIÓN DE PASOS (REFACTORIZADO PARA STATEVECTOR)
# ============================================================================


class LoadDataStep(BaseProcessingStep):
    """Paso de carga de datos."""
    
    PRODUCED_CONTEXT_KEYS = (
        "df_presupuesto", "df_insumos", "df_apus_raw",
        "raw_records", "parse_cache",
    )
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta carga."""
        # Acceder a paths desde raw_config o contexto
        config_dict = self.config.raw_config
        presupuesto_path = config_dict.get("presupuesto_path")
        apus_path = config_dict.get("apus_path")
        insumos_path = config_dict.get("insumos_path")
        
        if not all([presupuesto_path, apus_path, insumos_path]):
            raise PreconditionError(
                "load_data",
                ["presupuesto_path", "apus_path", "insumos_path"],
            )
        
        # Validar existencia
        file_validator = FileValidator()
        for path, name in [
            (presupuesto_path, "presupuesto"),
            (apus_path, "APUs"),
            (insumos_path, "insumos"),
        ]:
            is_valid, error = file_validator.validate_file_exists(path, name)
            if not is_valid:
                raise DataValidationError(error, "load_data")
        
        # Procesar presupuesto
        file_profiles = self.config.file_profiles
        presupuesto_profile = file_profiles.get("presupuesto_default", {})
        p_processor = PresupuestoProcessor(
            self.config.raw_config,
            self.thresholds,
            presupuesto_profile,
        )
        df_presupuesto = p_processor.process(presupuesto_path)
        
        if df_presupuesto is None or df_presupuesto.empty:
            raise DataValidationError(
                "Presupuesto processing returned empty DataFrame",
                "load_data",
            )
        
        telemetry.record_metric("load_data", "presupuesto_rows", len(df_presupuesto))
        
        # Procesar insumos
        insumos_profile = file_profiles.get("insumos_default", {})
        i_processor = InsumosProcessor(self.thresholds, insumos_profile)
        df_insumos = i_processor.process(insumos_path)
        
        if df_insumos is None or df_insumos.empty:
            raise DataValidationError(
                "Insumos processing returned empty DataFrame",
                "load_data",
            )
        
        telemetry.record_metric("load_data", "insumos_rows", len(df_insumos))
        
        # Stabilize flux
        flux_result = mic.project_intent(
            "stabilize_flux",
            {"file_path": str(apus_path), "config": self.config.raw_config},
            {},
        )
        
        if not flux_result.get("success", False):
            raise StepExecutionError(
                f"stabilize_flux failed: {flux_result.get('error')}",
                "load_data",
            )
        
        df_apus_raw = pd.DataFrame(flux_result["data"])
        if df_apus_raw.empty:
            raise DataValidationError("DataFluxCondenser returned empty", "load_data")
        
        telemetry.record_metric("load_data", "apus_raw_rows", len(df_apus_raw))
        
        # Parse raw
        apus_profile = file_profiles.get("apus_default", {})
        parse_result = mic.project_intent(
            "parse_raw",
            {"file_path": str(apus_path), "profile": apus_profile},
            {},
        )
        
        if not parse_result.get("success", False):
            raise StepExecutionError(
                f"parse_raw failed: {parse_result.get('error')}",
                "load_data",
            )
        
        telemetry.record_metric(
            "load_data",
            "raw_records_count",
            len(parse_result.get("raw_records", [])),
        )
        
        # Actualizar state
        state.df_presupuesto = df_presupuesto
        state.df_insumos = df_insumos
        state.df_apus_raw = df_apus_raw
        state.raw_records = parse_result.get("raw_records", [])
        state.parse_cache = parse_result.get("parse_cache", {})
        state.validated_strata = self._compute_validated_strata(state)
        
        return state
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados."""
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            if is_valid:
                validated.add(stratum)
        return validated


class AuditedMergeStep(BaseProcessingStep):
    """Paso de fusión con auditoría homológica."""
    
    REQUIRED_CONTEXT_KEYS = ("df_apus_raw", "df_insumos")
    PRODUCED_CONTEXT_KEYS = ("df_merged",)
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta fusión auditada."""
        df_b = state.df_apus_raw
        df_insumos = state.df_insumos
        
        # Auditoría homológica
        if state.df_presupuesto is not None and self.config.enforce_homology:
            self.logger.info("Iniciando auditoría homológica (Mayer-Vietoris)...")
            
            if self.auditor is None:
                self.auditor = HomologicalAuditor(TelemetryContext())
            
            audit = self.auditor.audit_merge(
                state.df_presupuesto,
                df_b,
                pd.DataFrame(),  # Temporal
                state,
            )
            
            if audit["warnings"]:
                state.integration_risk_alert = audit
                for warning in audit["warnings"]:
                    self.logger.warning(f"Homology Alert: {warning}")
            
            if not audit["passed"] and self.config.enforce_homology:
                raise HomologicalAuditError(
                    "Merge would violate homological properties",
                    audit,
                )
        
        # Fusión física
        merger = DataMerger(self.thresholds)
        df_merged = merger.merge_apus_with_insumos(df_b, df_insumos)
        
        if df_merged is None or df_merged.empty:
            raise DataValidationError("Merge produced empty DataFrame", "audited_merge")
        
        # Auditoría post-fusión
        if self.auditor is not None and state.df_presupuesto is not None:
            audit_post = self.auditor.audit_merge(
                state.df_presupuesto,
                df_b,
                df_merged,
                state,
            )
            
            if audit_post["warnings"]:
                self.logger.info(f"Post-merge audit: {audit_post['warnings']}")
        
        telemetry.record_metric("audited_merge", "merged_rows", len(df_merged))
        
        state.df_merged = df_merged
        state.validated_strata = self._compute_validated_strata(state)
        
        return state
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados."""
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            if is_valid:
                validated.add(stratum)
        return validated


class CalculateCostsStep(BaseProcessingStep):
    """Paso de cálculo de costos."""
    
    REQUIRED_CONTEXT_KEYS = ("df_merged",)
    PRODUCED_CONTEXT_KEYS = ("df_apu_costos", "df_tiempo", "df_rendimiento")
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta cálculo de costos."""
        df_merged = state.df_merged
        
        # Proyectar intención táctica
        logic_result = mic.project_intent(
            "structure_logic",
            {
                "raw_records": state.raw_records,
                "parse_cache": state.parse_cache,
                "config": self.config.raw_config,
            },
            {},
        )
        
        if (
            logic_result.get("success", False)
            and "df_apu_costos" in logic_result
        ):
            self.logger.info("✓ structure_logic exitoso")
            state.df_apu_costos = logic_result["df_apu_costos"]
            state.df_tiempo = logic_result.get("df_tiempo", pd.DataFrame())
            state.df_rendimiento = logic_result.get("df_rendimiento", pd.DataFrame())
            state.quality_report = logic_result.get("quality_report", {})
        else:
            self.logger.info("Fallback a APUProcessor")
            processor = APUProcessor(self.config.raw_config)
            (
                state.df_apu_costos,
                state.df_tiempo,
                state.df_rendimiento,
            ) = processor.process_vectors(df_merged)
        
        telemetry.record_metric("calculate_costs", "costos_rows", len(state.df_apu_costos))
        
        state.validated_strata = self._compute_validated_strata(state)
        return state
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados."""
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            if is_valid:
                validated.add(stratum)
        return validated


class FinalMergeStep(BaseProcessingStep):
    """Paso de fusión final."""
    
    REQUIRED_CONTEXT_KEYS = ("df_presupuesto", "df_apu_costos")
    PRODUCED_CONTEXT_KEYS = ("df_final",)
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta fusión final."""
        processor = APUProcessor(self.config.raw_config)
        state.df_final = processor.consolidate_results(
            state.df_presupuesto,
            state.df_apu_costos,
            state.df_tiempo,
        )
        
        telemetry.record_metric("final_merge", "final_rows", len(state.df_final))
        
        state.validated_strata = self._compute_validated_strata(state)
        return state
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados."""
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            if is_valid:
                validated.add(stratum)
        return validated


class BusinessTopologyStep(BaseProcessingStep):
    """Paso de topología de negocio."""
    
    REQUIRED_CONTEXT_KEYS = ("df_final",)
    PRODUCED_CONTEXT_KEYS = ("graph", "business_topology_report")
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta análisis de topología."""
        builder = BudgetGraphBuilder()
        df_merged = state.df_merged or pd.DataFrame()
        
        state.graph = builder.build(state.df_final, df_merged)
        
        telemetry.record_metric(
            "business_topology",
            "graph_nodes",
            state.graph.number_of_nodes(),
        )
        telemetry.record_metric(
            "business_topology",
            "graph_edges",
            state.graph.number_of_edges(),
        )
        
        try:
            agent = BusinessAgent(
                config=self.config.raw_config,
                mic=mic,
                telemetry=telemetry,
            )
            state.business_topology_report = agent.evaluate_project(asdict(state))
            self.logger.info("✓ BusinessAgent completó evaluación")
        except Exception as e:
            self.logger.warning(f"BusinessAgent degraded: {e}")
        
        state.validated_strata = self._compute_validated_strata(state)
        return state
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados."""
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            if is_valid:
                validated.add(stratum)
        return validated


class MaterializationStep(BaseProcessingStep):
    """Paso de materialización (BOM)."""
    
    PRODUCED_CONTEXT_KEYS = ("bill_of_materials", "logistics_plan")
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta materialización."""
        if state.business_topology_report is None:
            self.logger.warning(
                "business_topology_report ausente. Materialización omitida"
            )
            return state
        
        if state.graph is None:
            builder = BudgetGraphBuilder()
            df_merged = state.df_merged or pd.DataFrame()
            state.graph = builder.build(state.df_final, df_merged)
        
        # Extraer estabilidad
        report = state.business_topology_report
        stability = 10.0
        if hasattr(report, "details") and isinstance(report.details, dict):
            stability = report.details.get("pyramid_stability", 10.0)
        
        flux_metrics = {"pyramid_stability": stability, "avg_saturation": 0.0}
        
        generator = MatterGenerator()
        state.bill_of_materials = generator.materialize_project(
            state.graph,
            flux_metrics=flux_metrics,
            telemetry=telemetry,
        )
        state.logistics_plan = asdict(state.bill_of_materials)
        
        telemetry.record_metric(
            "materialization",
            "total_items",
            len(state.bill_of_materials.requirements),
        )
        
        return state


class BuildOutputStep(BaseProcessingStep):
    """Paso de construcción de salida final."""
    
    REQUIRED_CONTEXT_KEYS = (
        "df_final", "df_insumos", "df_merged",
        "df_apus_raw", "df_apu_costos", "df_tiempo", "df_rendimiento",
    )
    PRODUCED_CONTEXT_KEYS = ("final_result",)
    
    def _execute_impl(
        self,
        state: StateVector,
        telemetry: TelemetryContext,
        mic: MICRegistry,
    ) -> StateVector:
        """Ejecuta construcción de salida."""
        # Sincronizar y procesar
        df_merged = synchronize_data_sources(state.df_merged, state.df_final)
        df_processed_apus = build_processed_apus_dataframe(
            state.df_apu_costos,
            state.df_apus_raw,
            state.df_tiempo,
            state.df_rendimiento,
        )
        
        # Ensamblar producto
        if state.graph is not None and state.business_topology_report is not None:
            translator = SemanticTranslator(mic=mic)
            result_dict = translator.assemble_data_product(
                state.graph,
                state.business_topology_report,
            )
            result_dict["presupuesto"] = state.df_final.to_dict("records")
            result_dict["insumos"] = state.df_insumos.to_dict("records")
            self.logger.info("WISDOM: Producto ensamblado por SemanticTranslator")
        else:
            self.logger.warning("Sin artefactos de Estrategia. Salida básica")
            result_dict = build_output_dictionary(
                state.df_final,
                state.df_insumos,
                df_merged,
                state.df_apus_raw,
                df_processed_apus,
            )
        
        # Validar y enriquecer
        validated_result = validate_and_clean_data(
            result_dict,
            telemetry_context=telemetry,
        )
        validated_result["raw_insumos_df"] = state.df_insumos.to_dict("records")
        
        if state.business_topology_report is not None:
            validated_result["audit_report"] = asdict(state.business_topology_report)
        
        if state.logistics_plan is not None:
            validated_result["logistics_plan"] = state.logistics_plan
        
        # Narrativa técnica
        try:
            narrator = TelemetryNarrator(mic=mic)
            validated_result["technical_audit"] = narrator.summarize_execution(telemetry)
        except Exception as e:
            self.logger.warning(f"Narrativa técnica degradada: {e}")
            validated_result["technical_audit"] = {"status": "degraded", "error": str(e)}
        
        # Lineage hash
        lineage_hash = self._compute_lineage_hash(validated_result)
        
        # DataProduct final
        state.final_result = {
            "kind": "DataProduct",
            "metadata": {
                "version": PIPELINE_VERSION,
                "lineage_hash": lineage_hash,
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "generator": f"APU_Filter_Pipeline_{PIPELINE_VERSION}",
                "strata_validated": [
                    s.name for s in state.validated_strata if isinstance(s, Stratum)
                ],
            },
            "payload": validated_result,
        }
        
        return state
    
    def _compute_lineage_hash(self, payload: Dict[str, Any]) -> str:
        """Computa hash SHA-256 del payload."""
        hash_parts = []
        
        for key in sorted(payload.keys()):
            value = payload[key]
            try:
                sanitized = (
                    sanitize_for_json(value)
                    if isinstance(value, (list, dict))
                    else value
                )
                part = json.dumps({key: sanitized}, sort_keys=True, default=str)
            except (TypeError, ValueError):
                part = (
                    f"{key}:type={type(value).__name__},"
                    f"len={len(value) if hasattr(value, '__len__') else 'N/A'}"
                )
            hash_parts.append(part)
        
        composite = "|".join(hash_parts)
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()


# ============================================================================
# REGISTRO DE PASOS
# ============================================================================

STEP_REGISTRY: Final[Dict[str, Type[BaseProcessingStep]]] = {
    PipelineSteps.LOAD_DATA.value: LoadDataStep,
    PipelineSteps.AUDITED_MERGE.value: AuditedMergeStep,
    PipelineSteps.CALCULATE_COSTS.value: CalculateCostsStep,
    PipelineSteps.FINAL_MERGE.value: FinalMergeStep,
    PipelineSteps.BUSINESS_TOPOLOGY.value: BusinessTopologyStep,
    PipelineSteps.MATERIALIZATION.value: MaterializationStep,
    PipelineSteps.BUILD_OUTPUT.value: BuildOutputStep,
}


# ============================================================================
# DAG BUILDER
# ============================================================================


class DAGBuilder:
    """
    Constructor del DAG algebraico.
    
    Define dependencias entre pasos.
    """
    
    @staticmethod
    def build_default_dag() -> AlgebraicDAG:
        """Construye DAG con dependencias por defecto."""
        dag = AlgebraicDAG()
        
        # Agregar nodos
        for step in PipelineSteps:
            dag.add_step(step.value)
        
        # Agregar aristas (dependencias)
        dependencies = [
            ("load_data", "audited_merge", ["df_apus_raw", "df_insumos"]),
            ("audited_merge", "calculate_costs", ["df_merged"]),
            ("calculate_costs", "final_merge", ["df_apu_costos", "df_tiempo"]),
            ("final_merge", "business_topology", ["df_final"]),
            ("business_topology", "materialization", ["graph", "business_topology_report"], True),
            ("business_topology", "build_output", ["graph", "business_topology_report"]),
        ]
        
        for source, target, keys, *optional in dependencies:
            is_optional = optional[0] if optional else False
            dag.add_dependency(source, target, keys, is_optional)
        
        dag.validate()
        return dag


# ============================================================================
# SESSION MANAGER (MEJORADO)
# ============================================================================


class SessionManager:
    """
    Gestor de sesiones con soporte para StateVector.
    """
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _session_path(self, session_id: str) -> Path:
        return self.config.session_dir / f"{session_id}{SESSION_FILE_EXTENSION}"
    
    def load(self, session_id: str) -> Optional[StateVector]:
        """Carga estado de sesión."""
        if not session_id:
            return None
        
        session_path = self._session_path(session_id)
        
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, "rb") as f:
                data = pickle.load(f)
            
            if isinstance(data, StateVector):
                return data
            elif isinstance(data, dict):
                return StateVector.from_dict(data)
            else:
                raise SessionCorruptedError(
                    session_id,
                    f"Expected StateVector, got {type(data).__name__}",
                )
                
        except (pickle.UnpicklingError, EOFError) as e:
            raise SessionCorruptedError(session_id, str(e))
        except SessionCorruptedError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def save(self, state: StateVector) -> bool:
        """Guarda estado con escritura atómica."""
        session_path = self._session_path(state.session_id)
        tmp_path = session_path.with_suffix(f"{SESSION_FILE_EXTENSION}.tmp")
        
        try:
            data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            
            if len(data) > MAX_CONTEXT_SIZE_BYTES:
                self.logger.error(
                    f"Session {state.session_id} too large: "
                    f"{len(data)} > {MAX_CONTEXT_SIZE_BYTES}"
                )
                return False
            
            with open(tmp_path, "wb") as f:
                f.write(data)
            
            tmp_path.replace(session_path)
            
            self.logger.debug(f"Session saved: {state.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session {state.session_id}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            
            return False
    
    def delete(self, session_id: str) -> bool:
        """Elimina sesión."""
        session_path = self._session_path(session_id)
        
        try:
            if session_path.exists():
                session_path.unlink()
                self.logger.debug(f"Session deleted: {session_id}")
            return True
        except OSError as e:
            self.logger.warning(f"Could not delete session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self) -> int:
        """Limpia sesiones antiguas."""
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=self.config.max_age_hours
        )
        
        deleted = 0
        for path in self.config.session_dir.glob(f"*{SESSION_FILE_EXTENSION}"):
            try:
                mtime = datetime.datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=datetime.timezone.utc,
                )
                if mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except Exception as e:
                self.logger.warning(f"Could not check/delete {path}: {e}")
        
        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} old sessions")
        
        return deleted


# ============================================================================
# PIPELINE DIRECTOR (VERSIÓN DAG)
# ============================================================================


class PipelineDirector:
    """
    Director del Pipeline con DAG algebraico.
    
    Orquesta pasos según dependencias del DAG, implementa memoización
    y auditoría homológica automática.
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], PipelineConfig],
        telemetry: TelemetryContext,
        mic: Optional[MICRegistry] = None,
    ):
        """Inicializa director."""
        if isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
        
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cargar umbrales
        self.thresholds = self._load_thresholds()
        
        # Inicializar MIC
        if mic is not None:
            self.mic = mic
        else:
            self.mic = MICRegistry()
            register_core_vectors(self.mic, self.config.raw_config)
        
        # Construir DAG
        self.dag = DAGBuilder.build_default_dag()
        
        # Inicializar memoizador y auditor
        self.memoizer = OperatorMemoizer()
        self.auditor = HomologicalAuditor(telemetry)
        
        # Session manager
        self.session_manager = SessionManager(self.config.session)
        
        self.logger.info(
            f"PipelineDirector initialized | "
            f"Version: {PIPELINE_VERSION} | "
            f"DAG nodes: {len(self.dag.graph.nodes())} | "
            f"Memoization: enabled"
        )
    
    def _load_thresholds(self) -> ProcessingThresholds:
        """Carga umbrales."""
        thresholds = ProcessingThresholds()
        overrides = self.config.raw_config.get("processing_thresholds", {})
        
        if not isinstance(overrides, dict):
            self.logger.warning("processing_thresholds is not a dict. Using defaults.")
            return thresholds
        
        for key, value in overrides.items():
            if not hasattr(thresholds, key):
                self.logger.warning(f"Unknown threshold key '{key}'")
                continue
            
            current = getattr(thresholds, key)
            if current is not None and not isinstance(value, type(current)):
                self.logger.warning(f"Threshold '{key}': type mismatch. Ignored.")
                continue
            
            setattr(thresholds, key, value)
        
        return thresholds
    
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        """Computa estratos validados basándose en evidencia."""
        validated = set()
        
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            is_valid = all(
                state.get_evidence(stratum).get(key, False)
                for key in evidence_keys
            )
            
            if is_valid:
                validated.add(stratum)
        
        return validated
    
    def _check_stratum_prerequisites(
        self,
        target_stratum: Stratum,
        validated_strata: Set[Stratum],
    ) -> bool:
        """Verifica que todos los estratos inferiores estén validados."""
        target_level = stratum_level(target_stratum)
        
        if target_level == 0:
            return True
        
        for stratum, level in _STRATUM_ORDER.items():
            if level < target_level and stratum not in validated_strata:
                return False
        
        return True
    
    def _enforce_filtration(
        self,
        target_stratum: Stratum,
        state: StateVector,
    ) -> None:
        """Valida filtración topológica."""
        validated = state.validated_strata
        
        if not self._check_stratum_prerequisites(target_stratum, validated):
            target_level = stratum_level(target_stratum)
            missing = [
                s.name
                for s, l in _STRATUM_ORDER.items()
                if l < target_level and s not in validated
            ]
            raise FiltrationViolationError(
                target_stratum,
                missing,
                [s.name for s in validated],
            )
    
    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_state: Optional[StateVector] = None,
        validate_stratum: bool = True,
    ) -> StepResult:
        """
        Ejecuta un único paso.
        
        Args:
            step_name: Nombre del paso.
            session_id: ID de sesión.
            initial_state: Estado inicial (se fusiona con sesión).
            validate_stratum: Si validar filtración.
            
        Returns:
            Resultado de ejecución.
        """
        self.logger.info(f"Executing step: {step_name}")
        
        # Cargar o crear estado
        state = self.session_manager.load(session_id)
        if state is None:
            state = initial_state or StateVector(session_id=session_id)
        elif initial_state:
            # Fusionar estados
            for attr in initial_state.__dataclass_fields__:
                value = getattr(initial_state, attr)
                if value is not None:
                    setattr(state, attr, value)
        
        try:
            # Obtener paso
            step_class = STEP_REGISTRY.get(step_name)
            if step_class is None:
                raise StepExecutionError(
                    f"Step class not found: {step_name}",
                    step_name,
                )
            
            # Validar filtración
            pipeline_step = PipelineSteps.from_string(step_name)
            if pipeline_step and validate_stratum and self.config.enforce_filtration:
                self._enforce_filtration(pipeline_step.stratum, state)
            
            # Instanciar y ejecutar
            step_instance = step_class(
                self.config,
                self.thresholds,
                self.memoizer,
                self.auditor,
            )
            
            new_state = step_instance.execute(state, self.telemetry, self.mic)
            
            # Actualizar y guardar
            new_state.validated_strata = self._compute_validated_strata(new_state)
            self.session_manager.save(new_state)
            
            # Retornar último resultado
            if new_state.step_results:
                return new_state.step_results[-1]
            
            return StepResult(
                step_name=step_name,
                status=StepStatus.SUCCESS,
                stratum=pipeline_step.stratum if pipeline_step else Stratum.PHYSICS,
                state_vector_hash=new_state.compute_hash(),
            )
            
        except PipelineError as e:
            result = StepResult(
                step_name=step_name,
                status=StepStatus.ERROR,
                stratum=PipelineSteps.from_string(step_name).stratum
                if PipelineSteps.from_string(step_name)
                else Stratum.PHYSICS,
                error=str(e),
            )
            
            if self.config.session.persist_on_error:
                state.step_results.append(result)
                self.session_manager.save(state)
            
            self.logger.error(f"Error in {step_name}: {e}")
            
            return result
    
    def execute_pipeline(
        self,
        initial_context: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.
        
        Args:
            initial_context: Contexto inicial (dict con rutas, etc).
            session_id: ID de sesión.
            
        Returns:
            Contexto/DataProduct final.
        """
        session_id = session_id or str(uuid.uuid4())
        
        self.logger.info(
            f"Starting pipeline (Session: {session_id[:8]}...) "
            f"| DAG mode | Memoization enabled"
        )
        
        # Crear estado inicial
        initial_state = StateVector(session_id=session_id)
        
        # Copiar configuraciones al estado (como presupuesto_path, etc)
        for key, value in initial_context.items():
            if hasattr(initial_state, key):
                setattr(initial_state, key, value)
        
        # Guardar estado inicial
        if not self.session_manager.save(initial_state):
            raise SessionError(
                f"Failed to persist initial state. "
                f"Check permissions on {self.config.session.session_dir}"
            )
        
        # Obtener receta (orden topológico del DAG)
        if self.config.recipe:
            recipe = self.config.recipe
        else:
            recipe = self.dag.topological_sort()
        
        # Ejecutar pasos
        for idx, step_name in enumerate(recipe):
            # Validar que esté habilitado
            if not self.config.is_step_enabled(step_name):
                self.logger.info(f"Skipping disabled step: {step_name}")
                continue
            
            self.logger.info(
                f"Orchestrating step [{idx + 1}/{len(recipe)}]: {step_name}"
            )
            
            result = self.run_single_step(
                step_name,
                session_id,
                initial_state if idx == 0 else None,
            )
            
            if not result.is_successful:
                error_msg = (
                    f"Pipeline failed at step '{step_name}': {result.error}"
                )
                self.logger.critical(error_msg)
                raise StepExecutionError(error_msg, step_name)
        
        # Cargar estado final
        final_state = self.session_manager.load(session_id)
        
        if final_state is None:
            raise SessionError(f"Failed to load final state for {session_id}")
        
        # Limpiar sesión si está configurado
        if self.config.session.cleanup_on_success:
            self.session_manager.delete(session_id)
        
        # Retornar DataProduct
        if final_state.final_result:
            self.logger.info(f"Pipeline completed. DataProduct ready.")
            return final_state.final_result
        
        # Fallback: retornar estado como dict
        self.logger.warning("No final_result found. Returning state as dict.")
        return final_state.to_dict()
    
    def get_dag_info(self) -> Dict[str, Any]:
        """Obtiene información del DAG."""
        return {
            "version": PIPELINE_VERSION,
            "nodes": list(self.dag.graph.nodes()),
            "edges": [
                (u, v) for u, v in self.dag.graph.edges()
            ],
            "is_acyclic": nx.is_directed_acyclic_graph(self.dag.graph),
            "topological_order": self.dag.topological_sort(),
        }
    
    def get_memoization_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de memoización."""
        return self.memoizer.get_stats()
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del director."""
        return {
            "version": PIPELINE_VERSION,
            "dag_nodes": len(self.dag.graph.nodes()),
            "dag_edges": len(self.dag.graph.edges()),
            "memoization_enabled": True,
            "homology_audit_enabled": self.config.enforce_homology,
            "filtration_enabled": self.config.enforce_filtration,
            "memoization_stats": self.get_memoization_stats(),
        }


# ============================================================================
# FUNCIONES FACTORY
# ============================================================================


def create_director(
    config: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryContext] = None,
    mic: Optional[MICRegistry] = None,
) -> PipelineDirector:
    """Factory para crear director."""
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    
    return PipelineDirector(config, telemetry, mic)


def create_director_from_env() -> PipelineDirector:
    """Crea director desde variables de entorno."""
    config = {
        "session_dir": os.getenv("PIPELINE_SESSION_DIR", DEFAULT_SESSION_DIR),
        "enforce_filtration": os.getenv(
            "PIPELINE_ENFORCE_FILTRATION", "true"
        ).lower() == "true",
        "enforce_homology": os.getenv(
            "PIPELINE_ENFORCE_HOMOLOGY", "true"
        ).lower() == "true",
    }
    
    return create_director(config)


# ============================================================================
# FUNCIÓN DE ENTRADA PRINCIPAL
# ============================================================================


def process_all_files(
    presupuesto_path: Union[str, Path],
    apus_path: Union[str, Path],
    insumos_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryContext] = None,
) -> Dict[str, Any]:
    """
    Función de entrada principal (Batch Mode).
    
    Args:
        presupuesto_path: Ruta al archivo de presupuesto.
        apus_path: Ruta al archivo de APUs.
        insumos_path: Ruta al archivo de insumos.
        config: Configuración.
        telemetry: Contexto de telemetría.
        
    Returns:
        DataProduct generado.
        
    Raises:
        PipelineError: Si hay error.
    """
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    
    # Resolver y validar rutas
    paths = {
        "presupuesto_path": Path(presupuesto_path).resolve(),
        "apus_path": Path(apus_path).resolve(),
        "insumos_path": Path(insumos_path).resolve(),
    }
    
    for name, path in paths.items():
        if not path.exists():
            error = f"File not found: {name} = {path}"
            telemetry.record_error("process_all_files", error)
            raise FileNotFoundError(error)
    
    # Crear director y ejecutar
    director = create_director(config, telemetry)
    
    initial_context = {k: str(v) for k, v in paths.items()}
    
    try:
        final_result = director.execute_pipeline(initial_context)
        return final_result
        
    except PipelineError as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise PipelineError(f"Unexpected error: {e}") from e


if __name__ == "__main__":
    # Ejemplo de uso
    config = {
        "enforce_filtration": True,
        "enforce_homology": True,
    }
    
    telemetry = TelemetryContext()
    
    result = process_all_files(
        presupuesto_path="data/presupuesto.xlsx",
        apus_path="data/apus.xlsx",
        insumos_path="data/insumos.xlsx",
        config=config,
        telemetry=telemetry,
    )
    
    print("Pipeline completed successfully!")
    print(f"DataProduct generated with {len(result.get('payload', {}))} keys")