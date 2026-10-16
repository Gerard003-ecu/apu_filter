# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pipeline Director v5.1.0 — Poincaré–Melnikov–Recursive-N3           ║
║ Ruta   : app/tactics/pipeline_director.py                                    ║
║ Versión: 5.1.0-DAG-Poincare-Melnikov-RecursiveN3-Strict                      ║
║ Fase   : 1 / 3  (OBSERVE) — este archivo se concatena con Fases 2 y 3        ║
╚══════════════════════════════════════════════════════════════════════════════╝

FILOSOFÍA —──────────────────────────────────────────────────────────────────────
Sea \(M = T^*Q\) el fibrado cotangente del espacio de configuraciones del
pipeline (variedad simpléctica de dimensión \(2n\), forma canónica
\(\omega = \mathrm{d}p\wedge\mathrm{d}q\)) y sea
\(H\colon M\to\mathbb{R}\) la función Hamiltoniana de costo computacional.
Cada ejecución es una trayectoria discreta \(\gamma\colon\mathbb{Z}\to M\)
que interpolamos a un flujo \(\varphi_t^{X_H}\) del campo hamiltoniano
\(X_H = \omega^\sharp(\mathrm{d}H)\).

Este módulo introduce TRES capas anidadas. La Fase 1 OBSERVA; no decide.

  (A) MECÁNICA CELESTE DE POINCARÉ
      · Sección transversal \(\Sigma = \{p\in M : h(p)=0\}\) con
        \(\mathrm{d}h(X_H)\neq 0\) (transversalidad).                     [Poincaré]
      · Codificación hipercompleja DIKW ≅ \(\mathbb{H}\):
            \(1\mapsto\mathrm{PHYSICS},\; i\mapsto\mathrm{TACTICS},
             j\mapsto\mathrm{STRATEGY},\; k\mapsto\mathrm{WISDOM}\).
        La función de sección es \(h(q)=\|{\mathrm{Im}\,q}\| - {\mathrm{Re}\,q}\).
      · Aplicación de primer retorno \(P\colon\Sigma\to\Sigma\),
        jacobiano variacional \(DP\) y espectro de Lyapunov normal.     [First-return]
      · Función de Melnikov (forma de Poisson):
            \(M(t_0)=\displaystyle\int_{\mathbb{R}}\{H_0,H_1\}(\gamma_0(t),t+t_0)\,\mathrm{d}t\).
        Cero simple \(\Leftrightarrow M(t_*)=0\) y \(\partial M/\partial t_0|_{t_*}\neq 0\)
        \(\Rightarrow W^s\pitchfork W^u\) (caos homoclínico).            [Melnikov]
      · Criterio KAM diofántico:
            \(|\langle k,\omega\rangle|\ge\gamma/|k|^\tau\), \(\varepsilon<\varepsilon_{\mathrm{KAM}}\).
                                                                        [KAM]
      · Difusión de Arnold + tiempo de Nekhoroshev
            \(T_{\mathrm{Nek}}\sim\exp\bigl((1/\varepsilon)^{1/(2n)}\bigr)\).  [Arnold]

  (B) AUTOMEJORA RECURSIVA DE NIVEL 3  (preparación observacional)
      Fase 1 NO ejecuta N1/N2/N3. Emite el vector de rasgos
            \(\theta_{\mathrm{seed}} = (\lambda_2,\nu,\varepsilon_{\mathrm{KAM}},|M|_\infty,D_{\mathrm{Arnold}},\lambda_{\mathrm{Lyap}})\)
      que el kernel N3 (Fase 2) ingiere como condición inicial.
      Contrato de frontera: Protocolo ``Phase2OrientKernel``.

  (C) 3 FASES ANIDADAS
      Fase 1 (OBSERVE)  ─►  espectro DAG + Σ + P + M + KAM + Arnold
                            ⟹  emite ObserverPacket
                            ⟹  project_to_phase_2  ≡  inicio formal de Fase 2
      Fase 2 (ORIENT)   ─►  clausura transitiva + kernel N3
                            ⟹  emite OrientationSeed
      Fase 3 (DECIDE)   ─►  Mayer-Vietoris + veredicto Ω₃
                            ⟹  DataProduct terminal
"""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import logging
import math
import os
import pickle
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, make_dataclass, replace
from pathlib import Path
from typing import (
    Any, Callable, ClassVar, Dict, Final, FrozenSet, Generic, Iterator,
    List, Mapping, Optional, Protocol, Sequence, Set, Tuple, Type, TypeVar,
    Union, runtime_checkable,
)

import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from flask import current_app  # noqa: F401
except Exception:                # pragma: no cover
    current_app = None

from app.core.constants import ColumnNames, InsumoType
from app.physics.flux_condenser import CondenserConfig, DataFluxCondenser
from app.physics.matter_generator import MatterGenerator
from app.core.schemas import Stratum
from app.core.telemetry import TelemetryContext
from app.core.telemetry_narrative import TelemetryNarrator
from app.tactics.business_topology import (
    BudgetGraphBuilder, BusinessTopologicalAnalyzer,
)
from app.strategy.business_agent import BusinessAgent
from app.wisdom.semantic_translator import SemanticTranslator
from app.adapters.tools_interface import MICRegistry, register_core_vectors

from app.tactics.apu_processor import (
    APUProcessor, APUCostCalculator, DataMerger, DataValidator, FileValidator,
    InsumosProcessor, PresupuestoProcessor, ProcessingThresholds,
    build_output_dictionary, build_processed_apus_dataframe,
    calculate_insumo_costs, calculate_total_costs,
    group_and_split_description, sanitize_for_json,
    synchronize_data_sources,
)
from app.tactics.data_validator import validate_and_clean_data


# ============================================================================
# CONSTANTES GLOBALES TRANSVERSALES
# ============================================================================

EPSILON: Final[float]              = 1e-9
PIPELINE_VERSION: Final[str]       = "5.1.0-dag-poincare-melnikov-n3"
DEFAULT_SESSION_DIR: Final[str]    = "data/sessions"
MAX_CONTEXT_SIZE_BYTES: Final[int] = 50 * 1024 * 1024
SESSION_FILE_EXTENSION: Final[str] = ".pkl"
MEMOIZATION_CACHE_SIZE: Final[int] = 1000
HOMOLOGICAL_AUDIT_ENABLED: Final[bool] = True

_WILKINSON_LIMIT: Final[float]     = 1.0e-12
_SPECTRAL_TOL: Final[float]        = 1.0e-9

# — Constantes de mecánica celeste ————————————————————————————————
_MELNIKOV_TOL: Final[float]        = 1.0e-8      # Cota superior de |M(t₀)| trivial
_MELNIKOV_DMDT_TOL: Final[float]   = 1.0e-10     # |∂M/∂t₀| mínimo para cero simple
_KAM_PERSISTENCE_TOL: Final[float] = 1.0e-6      # Tolerancia ε canónica
_KAM_DIOPH_GAMMA: Final[float]     = 1.0e-3      # γ de la condición diofántica
_KAM_DIOPH_TAU: Final[float]       = 2.5         # τ > n-1  (n efetivo = 4 estratos)
_ARNOLD_DIFFUSION_TOL: Final[float]= 1.0e-4      # Leakage admisible inter-estrato
_POINCARE_SECTION_TOL: Final[float]= 1.0e-7      # |h(p)| < tol para p ∈ Σ
_SYMPLECTIC_DEFECT_TOL: Final[float] = 1.0e-10   # ||MᵀΩM − Ω||_F admisible

# — Constantes de automejora recursiva (semilla observacional) ——————
_N3_MAX_LEVEL: Final[int]          = 3
_N3_IDEMPOTENCE_TOL: Final[float]  = 1.0e-7
_N3_ALPHA_FLOOR: Final[float]      = 1.0e-6
_N3_ALPHA_CEIL: Final[float]       = 1.0e+2
_N3_FEATURE_DIM: Final[int]        = 6           # dim(θ_seed) emitida por Fase 1

# — Constantes hipercomplejas (DIKW ≅ ℍ) ———————————————————————————
_QUAT_BASIS: Final[Dict[str, Tuple[float, float, float, float]]] = {
    "PHYSICS":  (1.0, 0.0, 0.0, 0.0),   # 1
    "TACTICS":  (0.0, 1.0, 0.0, 0.0),   # i
    "STRATEGY": (0.0, 0.0, 1.0, 0.0),   # j
    "WISDOM":   (0.0, 0.0, 0.0, 1.0),   # k
}

logger = logging.getLogger(__name__)


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► OBSERVE : ESPECTRO DAG Y SECCIÓN DE POINCARÉ                █
# █   (Acyclicity audit  +  Poincaré transversal section  +  KAM/Melnikov)   █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 1.0 EXCEPCIONES DEL DOMINIO ──────────────────────────────────────────────

class PipelineError(Exception):
    """Excepción base del pipeline (monoide de errores con unidad absorbente)."""
    pass


class ConfigurationError(PipelineError):
    """Error de configuración."""
    pass


class StepExecutionError(PipelineError):
    """Error durante ejecución de paso."""

    def __init__(self, message: str, step_name: str,
                 cause: Optional[Exception] = None):
        self.step_name = step_name
        self.cause = cause
        super().__init__(f"[{step_name}] {message}")


class FiltrationViolationError(PipelineError):
    """Violación del invariante de filtración \(V_k\subset V_{k+1}\)."""

    def __init__(self, target_stratum: Stratum,
                 missing_strata: List[str], validated_strata: List[str]):
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata
        super().__init__(
            f"Filtration Violation. Target: {target_stratum.name}. "
            f"Missing: {missing_strata}. Validated: {validated_strata}."
        )


class FiltrationAbortSignal(FiltrationViolationError):
    r"""
    Señal monádica absorbente (elemento \(\varnothing\) del monoide).
    Semántica:  \(\varnothing\circ f=\varnothing\)  \(\forall f\)
    — ningún estado corrupto se persiste.
    """

    def __init__(self, target_stratum: Stratum, missing_strata: List[str],
                 validated_strata: List[str], step_name: str = ""):
        self.step_name = step_name
        super().__init__(target_stratum, missing_strata, validated_strata)


class HomologicalAuditError(PipelineError):
    def __init__(self, message: str,
                 homology_data: Optional[Dict[str, Any]] = None):
        self.homology_data = homology_data or {}
        super().__init__(f"Homological audit failed: {message}")


class StateVectorError(PipelineError):        pass
class MemoizationError(PipelineError):         pass
class SessionError(PipelineError):             pass


class SessionNotFoundError(SessionError):
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionCorruptedError(SessionError):
    def __init__(self, session_id: str, reason: str):
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session corrupted: {session_id}. Reason: {reason}")


class PreconditionError(StepExecutionError):
    def __init__(self, step_name: str, missing_keys: List[str]):
        self.missing_keys = missing_keys
        super().__init__(f"Missing keys: {missing_keys}", step_name)


class DataValidationError(StepExecutionError): pass


class DependencyResolutionError(PipelineError):
    def __init__(self, message: str, missing_deps: Optional[List[str]] = None):
        self.missing_deps = missing_deps or []
        super().__init__(message)


class PoincareSectionError(PipelineError):
    r"""Falla en la captura/validación de la sección transversal \(\Sigma\)."""
    pass


class MelnikovTangleError(PipelineError):
    r"""Tangencia homoclínica no admisible (\(M(t_0)\not\le\varepsilon\))."""
    pass


class KAMBreakdownError(PipelineError):
    """Colapso de toro KAM (invariante de estrato perdido)."""
    pass


class RecursiveSelfImprovementError(PipelineError):
    """Falla en el kernel N3."""
    pass


class SymplecticConsistencyError(PipelineError):
    r"""El jacobiano no yace en \(\mathrm{Sp}(2n)\): \(M^\top\Omega M\neq\Omega\)."""
    pass


# ── 1.1 ENUMERACIONES ────────────────────────────────────────────────────────

class StepStatus(str, enum.Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    SKIPPED  = "skipped"
    CACHED   = "cached"
    ERROR    = "error"

    @property
    def is_terminal(self) -> bool:
        return self in (StepStatus.SUCCESS, StepStatus.SKIPPED,
                        StepStatus.CACHED, StepStatus.ERROR)

    @property
    def is_successful(self) -> bool:
        return self in (StepStatus.SUCCESS, StepStatus.SKIPPED,
                        StepStatus.CACHED)


class PipelineSteps(str, enum.Enum):
    """Pasos canónicos del pipeline (orden topológico pre-fijado)."""
    LITHOLOGICAL_BASELINE = "lithological_baseline"
    LOAD_DATA             = "load_data"
    AUDITED_MERGE         = "audited_merge"
    CALCULATE_COSTS       = "calculate_costs"
    FINAL_MERGE           = "final_merge"
    BUSINESS_TOPOLOGY     = "business_topology"
    MATERIALIZATION       = "materialization"
    BUILD_OUTPUT          = "build_output"

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


class ImprovementLevel(enum.IntEnum):
    r"""Niveles de automejora recursiva (contrato para Fase 2).

    N1 → θ (parámetros del pipeline)
    N2 → O (topología del DAG / acoplamientos)
    N3 → α (la propia regla de mejora) — punto fijo \(I_3\circ I_3=I_3\)
    """
    N1_PARAMETRIC = 1
    N2_TOPOLOGIC  = 2
    N3_META_META  = 3


class PoincareSectionType(str, enum.Enum):
    r"""Tipos de hipersuperficie transversal \(h^{-1}(0)\subset M\)."""
    STRATUM_TRANSITION = "stratum_transition"   # cruce entre estratos DIKW
    NILPOTENCE_EVENT   = "nilpotence_event"     # cambio de índice ν
    MEMO_HIT           = "memo_hit"             # evento de caché
    HOMOLOGICAL_ZERO   = "homological_zero"     # Δβ₁ = 0
    ENERGY_SURFACE     = "energy_surface"       # H(p) − E₀ = 0
    QUATERNION_BALANCE = "quaternion_balance"   # ‖Im q‖ − Re q = 0
    CUSTOM             = "custom"


_STEP_STRATA: Final[Dict[PipelineSteps, Stratum]] = {
    PipelineSteps.LITHOLOGICAL_BASELINE: Stratum.PHYSICS,
    PipelineSteps.LOAD_DATA:             Stratum.PHYSICS,
    PipelineSteps.AUDITED_MERGE:         Stratum.PHYSICS,
    PipelineSteps.CALCULATE_COSTS:       Stratum.TACTICS,
    PipelineSteps.FINAL_MERGE:           Stratum.TACTICS,
    PipelineSteps.BUSINESS_TOPOLOGY:     Stratum.STRATEGY,
    PipelineSteps.MATERIALIZATION:       Stratum.STRATEGY,
    PipelineSteps.BUILD_OUTPUT:          Stratum.WISDOM,
}

_STRATUM_ORDER: Final[Dict[Stratum, int]] = {
    Stratum.PHYSICS: 0, Stratum.TACTICS: 1,
    Stratum.STRATEGY: 2, Stratum.WISDOM: 3,
}

_STRATUM_EVIDENCE: Final[Dict[Stratum, Tuple[str, ...]]] = {
    Stratum.PHYSICS:  ("df_presupuesto", "df_insumos", "df_apus_raw"),
    Stratum.TACTICS:  ("df_apu_costos",),
    Stratum.STRATEGY: ("graph", "business_topology_report"),
    Stratum.WISDOM:   ("final_result",),
}


def stratum_level(s: Stratum) -> int:
    return _STRATUM_ORDER.get(s, -1)


# ── 1.2 ESTRUCTURAS ALGEBRAICAS BASE ────────────────────────────────────────

@dataclass(frozen=True)
class TensorSignature:
    """Firma criptográfica de un tensor de datos (SHA-256 + shape + dtype)."""
    hash_value: str
    shape: Tuple[int, ...]
    dtype: str
    timestamp: datetime.datetime

    @classmethod
    def compute(cls, data: Any,
                timestamp: Optional[datetime.datetime] = None) -> TensorSignature:
        timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
        if isinstance(data, pd.DataFrame):
            shape, dtype = data.shape, "dataframe"
            content = data.to_json().encode("utf-8")
        elif isinstance(data, np.ndarray):
            shape, dtype = data.shape, str(data.dtype)
            content = data.tobytes()
        elif isinstance(data, (dict, list)):
            shape, dtype = (len(data),), "json"
            content = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        else:
            shape, dtype = (), type(data).__name__
            content = str(data).encode("utf-8")
        return cls(hashlib.sha256(content).hexdigest(), shape, dtype, timestamp)

    def matches(self, other: "TensorSignature") -> bool:
        return (self.hash_value == other.hash_value
                and self.shape == other.shape
                and self.dtype == other.dtype)


@dataclass(frozen=True)
class MemoizationKey:
    input_hash:  str
    operator_id: str
    stratum:     str

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.input_hash, self.operator_id, self.stratum)

    def __hash__(self) -> int: return hash(self.to_tuple())
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MemoizationKey) and self.to_tuple() == other.to_tuple()


@dataclass
class MemoizationEntry:
    key:              MemoizationKey
    output:           Dict[str, Any]
    output_signature: TensorSignature
    created_at:       datetime.datetime
    hit_count:        int = 0

    def is_fresh(self, max_age_hours: int = 24) -> bool:
        age = datetime.datetime.now(datetime.timezone.utc) - self.created_at
        return age < datetime.timedelta(hours=max_age_hours)


@dataclass(frozen=True, slots=True)
class QuaternionStratumState:
    r"""
    Estado DIKW como cuaternión \(q=w+xi+yj+zk\in\mathbb{H}\).

    El producto de Hamilton \(q_{t}\cdot q_{t+1}\) compone transiciones
    de estrato; la norma \(\|q\|\) es el cardinal euclídeo de evidencia.
    """
    w: float  # PHYSICS
    x: float  # TACTICS
    y: float  # STRATEGY
    z: float  # WISDOM

    @classmethod
    def from_strata(cls, strata: Set[Stratum]) -> "QuaternionStratumState":
        acc = np.zeros(4, dtype=np.float64)
        for s in strata:
            key = s.name if hasattr(s, "name") else str(s)
            basis = _QUAT_BASIS.get(key)
            if basis is not None:
                acc += np.asarray(basis, dtype=np.float64)
        return cls(w=float(acc[0]), x=float(acc[1]),
                   y=float(acc[2]), z=float(acc[3]))

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)

    def hamilton_norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))

    def real_part(self) -> float:
        return self.w

    def imag_norm(self) -> float:
        return float(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def section_function(self) -> float:
        r"""\(h(q)=\|\mathrm{Im}\,q\|-\mathrm{Re}\,q\). Cero ⇔ balance DIKW."""
        return self.imag_norm() - self.real_part()

    def hamilton_product(self, other: "QuaternionStratumState") -> "QuaternionStratumState":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return QuaternionStratumState(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )


# ── 1.3 STATE VECTOR Y StepResult ────────────────────────────────────────────

@dataclass
class StepResult:
    step_name:         str
    status:            StepStatus
    stratum:           Stratum
    duration_ms:       float = 0.0
    error:             Optional[str] = None
    context_keys:      List[str] = field(default_factory=list)
    metrics:           Dict[str, Any] = field(default_factory=dict)
    timestamp:         datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    state_vector_hash: Optional[str] = None
    memoization_hit:   bool = False

    @property
    def is_successful(self) -> bool: return self.status.is_successful

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name, "status": self.status.value,
            "stratum": self.stratum.name,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error, "context_keys": self.context_keys,
            "metrics": self.metrics, "timestamp": self.timestamp.isoformat(),
            "memoization_hit": self.memoization_hit,
        }


@dataclass
class StateVector:
    """Vector de estado tipado — elemento genérico del espacio de fase \(M=T^*Q\)."""
    # PHYSICS
    presupuesto_path: Optional[str] = None
    apus_path:        Optional[str] = None
    insumos_path:     Optional[str] = None
    df_presupuesto:   Optional[pd.DataFrame] = None
    df_insumos:       Optional[pd.DataFrame] = None
    df_apus_raw:      Optional[pd.DataFrame] = None
    df_merged:        Optional[pd.DataFrame] = None
    # TACTICS
    df_apu_costos:    Optional[pd.DataFrame] = None
    df_tiempo:        Optional[pd.DataFrame] = None
    df_rendimiento:   Optional[pd.DataFrame] = None
    df_final:         Optional[pd.DataFrame] = None
    # STRATEGY
    graph:                    Optional[nx.DiGraph] = None
    business_topology_report: Optional[Any] = None
    # WISDOM
    final_result:     Optional[Dict[str, Any]] = None
    # Metadatos
    raw_records:      List[Dict[str, Any]] = field(default_factory=list)
    parse_cache:      Dict[str, Any]       = field(default_factory=dict)
    validated_strata: Set[Stratum]         = field(default_factory=set)
    step_results:     List[StepResult]     = field(default_factory=list)
    quality_report:   Dict[str, Any]       = field(default_factory=dict)
    bill_of_materials:    Optional[Any]                 = None
    logistics_plan:       Optional[Dict[str, Any]]      = None
    integration_risk_alert: Optional[Dict[str, Any]]    = None
    audit_report:         Optional[Dict[str, Any]]      = None
    technical_audit:      Optional[Dict[str, Any]]      = None
    # Fase 1 → 2:  huella de Poincaré (si existe)
    poincare_trace: Optional[List[Dict[str, Any]]] = None
    # Operacional
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    version:    str = PIPELINE_VERSION

    # Geomecánica basal inyectada por LithologicalBaselineStep
    lithological_config: Optional[Dict[str, Any]] = None
    system_capacitance:  Optional[float] = None
    system_inductance:   Optional[float] = None
    base_resistance:     Optional[float] = None

    def quaternion_state(self) -> QuaternionStratumState:
        """Proyección hipercompleja del conjunto de estratos validados."""
        return QuaternionStratumState.from_strata(self.validated_strata)

    def computational_hamiltonian(self) -> float:
        r"""
        Hamiltoniano de costo (unidades adimensionales):

            \(H = T + V\),
            \(T = \tfrac12\sum_k(\mathrm{duration}_k/1000)^2\)  (cinética de pasos),
            \(V = \tfrac12\|q_{\mathbb{H}}\|^2\)                 (potencial DIKW).

        Conservación aproximada de \(H\) ⇔ flujo casi-adiabático.
        """
        kinetic = 0.5 * sum(
            (sr.duration_ms / 1000.0) ** 2 for sr in self.step_results)
        potential = 0.5 * (self.quaternion_state().hamilton_norm() ** 2)
        return float(kinetic + potential)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for fn in ["df_presupuesto", "df_insumos", "df_apus_raw", "df_merged",
                   "df_apu_costos", "df_tiempo", "df_rendimiento", "df_final"]:
            df = getattr(self, fn, None)
            if isinstance(df, pd.DataFrame):
                out[fn] = df.to_dict("records")
        out["raw_records"]      = self.raw_records
        out["parse_cache"]      = self.parse_cache
        out["validated_strata"] = [s.name if hasattr(s, "name") else str(s)
                                   for s in self.validated_strata]
        out["quality_report"]   = self.quality_report
        if self.graph is not None:
            out["graph_nodes"] = self.graph.number_of_nodes()
            out["graph_edges"] = self.graph.number_of_edges()
        if self.business_topology_report is not None:
            out["business_topology_report"] = asdict(
                self.business_topology_report
                if hasattr(self.business_topology_report, "__dataclass_fields__")
                else {"status": "available"})
        if self.final_result is not None:
            out["final_result"] = self.final_result
        out.update(session_id=self.session_id,
                   created_at=self.created_at.isoformat(),
                   updated_at=self.updated_at.isoformat(),
                   version=self.version)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateVector":
        kwargs: Dict[str, Any] = {}
        for fn in ["df_presupuesto", "df_insumos", "df_apus_raw", "df_merged",
                   "df_apu_costos", "df_tiempo", "df_rendimiento", "df_final"]:
            if fn in data and isinstance(data[fn], list):
                kwargs[fn] = pd.DataFrame(data[fn])
        kwargs["raw_records"]    = data.get("raw_records", [])
        kwargs["parse_cache"]    = data.get("parse_cache", {})
        kwargs["quality_report"] = data.get("quality_report", {})
        kwargs["session_id"]     = data.get("session_id", str(uuid.uuid4()))
        kwargs["version"]        = data.get("version", PIPELINE_VERSION)
        if "validated_strata" in data:
            kwargs["validated_strata"] = {getattr(Stratum, s)
                                          for s in data["validated_strata"]
                                          if hasattr(Stratum, s)}
        for k in ("created_at", "updated_at"):
            if k in data:
                try:    kwargs[k] = datetime.datetime.fromisoformat(data[k])
                except ValueError: pass
        return cls(**kwargs)

    def compute_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def get_evidence(self, stratum: Stratum) -> Dict[str, bool]:
        keys = _STRATUM_EVIDENCE.get(stratum, ())
        result: Dict[str, bool] = {}
        for k in keys:
            v = getattr(self, k, None)
            is_valid = (v is not None
                        and (not hasattr(v, "empty") or not v.empty)
                        and (not isinstance(v, (list, dict)) or len(v) > 0))
            result[k] = is_valid
        return result


# ── 1.4 ESTRUCTURAS DE MECÁNICA CELESTE ─────────────────────────────────────

def canonical_symplectic_form(n: int) -> NDArray[np.float64]:
    r"""
    Forma simpléctica canónica \(\Omega_{2n}=\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix}\).
    Convierte \(T^*\mathbb{R}^n\simeq\mathbb{R}^{2n}\) en variedad simpléctica.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.block([
        [np.zeros((n, n), dtype=np.float64),  np.eye(n, dtype=np.float64)],
        [-np.eye(n, dtype=np.float64),        np.zeros((n, n), dtype=np.float64)],
    ])


def cotangent_lift(J: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Levantamiento cotangente de \(J\in\mathrm{GL}(n)\):

        \(\widehat{J}=\begin{pmatrix}J&0\\0&J^{-\top}\end{pmatrix}\in\mathrm{Sp}(2n)\).

    Es la única extensión lineal que preserva \(\omega\) si \(J\) es invertible.
    Si \(J\) es singular se sustituye \(J^{-1}\) por la pseudoinversa de Moore-Penrose
    (la elevación deja de ser estrictamente simpléctica; el defecto se mide aparte).
    """
    n = int(J.shape[0])
    try:
        j_inv_t = la.inv(J).T
    except la.LinAlgError:
        j_inv_t = la.pinv(J).T
    zeros = np.zeros((n, n), dtype=np.float64)
    return np.block([[J, zeros], [zeros, j_inv_t]])


def poisson_bracket_r2n(
    grad_f: NDArray[np.float64],
    grad_g: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> float:
    r"""
    Corchete de Poisson \(\{F,G\}=\omega(X_F,X_G)=\langle\nabla F,\,\Omega^{-1}\nabla G\rangle\).
    En coordenadas canónicas \(\Omega^{-1}=-\Omega\), luego
    \(\{F,G\}=\nabla F^\top\Omega^\top\nabla G\).
    """
    if grad_f.size != grad_g.size or omega.shape[0] != grad_f.size:
        # Proyección a la subvariedad común más pequeña.
        m = min(grad_f.size, grad_g.size, omega.shape[0])
        if m == 0:
            return 0.0
        grad_f = grad_f[:m]
        grad_g = grad_g[:m]
        omega = omega[:m, :m]
    return float(grad_f.T @ omega.T @ grad_g)


@dataclass(frozen=True, slots=True)
class PipelineSpectraCertificate:
    """Certificado espectral inmutable del grafo causal DAG."""
    fiedler_value:           float
    nilpotency_index:        int
    symplectic_volume_drift: float
    is_dag_acyclic:          bool
    symplectic_defect:       float = 0.0
    spectral_gap:            float = 0.0
    algebraic_connectivity_directed: float = 0.0
    is_strictly_symplectic:  bool = False


@dataclass(frozen=True)
class PoincareSection:
    r"""
    Punto de cruce \(p\in\Sigma=\{x\in M:h(x)=0\}\) con \(\mathrm{d}h(X_H)\neq 0\).

    ``tangent`` vive en \(T_p\Sigma=\ker\mathrm{d}h_p\). El campo \(X_H\) es
    transversal ssi \(\lvert\langle\nabla h,\,X_H\rangle\rvert>\texttt{_POINCARE_SECTION_TOL}\).
    """
    section_type: PoincareSectionType
    state_hash:   str
    t_crossing:   float
    h_value:      float
    tangent:      NDArray[np.float64]
    quaternion:   Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    energy:       float = 0.0
    is_transverse: bool = True
    metadata:     Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReturnMapReport:
    r"""
    Análisis de \(P\colon\Sigma\to\Sigma\).

    El exponente de Lyapunov normal \(\lambda=\lim_{N\to\infty}N^{-1}\sum_j\log\|DP_{p_j}\|\)
    clasifica el punto fijo: hiperbólico ssi \(\mathrm{Re}(\lambda)\neq 0\).
    """
    n_crossings:           int
    fixed_point_estimate:  Optional[NDArray[np.float64]]
    stability_exponent:    float
    is_hyperbolic:         bool
    is_periodic_orbit:     bool
    jacobian_eigenvalues:  Optional[Tuple[complex, ...]] = None
    poincare_period:       Optional[float] = None
    lyapunov_spectrum:     Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class MelnikovCertificate:
    r"""
    Certificado de la función de Melnikov.

        \(M(t_0)=\displaystyle\int_{-\infty}^{\infty}\{H_0,H_1\}(\gamma_0(t),t+t_0)\,\mathrm{d}t\)

    Cero simple \(\Leftrightarrow M(t_*)=0\) y \(\partial_t M(t_*)\neq 0\)
    \(\Rightarrow W^s\pitchfork W^u\) \(\Rightarrow\) caos homoclínico.
    """
    melnikov_values:              NDArray[np.float64]
    simple_zeros:                 int
    mean_magnitude:               float
    max_magnitude:                float
    tangle_detected:              bool
    simple_zero_locations:        Tuple[float, ...] = ()
    melnikov_derivative_at_zeros: Tuple[float, ...] = ()
    poisson_mean:                 float = 0.0


@dataclass(frozen=True)
class KAMCertificate:
    r"""
    Certificado de persistencia KAM.

        \(\varepsilon=\|H_1\|_\infty/\omega_{\min}\),
        \(|\langle k,\omega\rangle|\ge\gamma/|k|^\tau\)  (diofántica).

    Si \(\varepsilon<\varepsilon_{\mathrm{KAM}}\) y \(\omega\) es diofántica,
    los toros \(T^k\subset M\) (estratos DIKW) sobreviven.
    """
    epsilon_perturbation:  float
    epsilon_KAM_threshold: float
    tori_persisting:       int
    is_KAM_stable:         bool
    diophantine_held:      bool = True
    frequency_vector:      Tuple[float, ...] = ()
    greene_residue:        float = 0.0


@dataclass(frozen=True)
class ArnoldDiffusionMetric:
    r"""
    Difusión de Arnold — fuga entre resonancias adyacentes del poset DIKW.

        \(D_{\mathrm{Arnold}}^{(s)}=\sum_t\lvert\Delta\beta_1^{(s)}(t)\rvert^2/\omega_s^2\)

    Tiempo de Nekhoroshev: \(T_{\mathrm{Nek}}\sim\exp\bigl((1/\varepsilon)^{1/(2n)}\bigr)\).
    """
    per_stratum_leakage: Dict[str, float]
    total_diffusion:     float
    dominant_channel:    Optional[str]
    is_adiabatic:        bool
    nekhoroshev_time:    float = 0.0
    n_resonant_channels: int = 0


@dataclass(frozen=True)
class ObserverPacket:
    r"""
    ← ÚLTIMO ARTEFACTO DE LA FASE 1.  OBJETO DE FRONTERA FASE 1 → FASE 2.

    Transporta TODA la información observacional (espectro, Σ, P, M, KAM,
    Arnold, vector de rasgos N3) hacia la FASE 2, donde
    ``Phase2OrientKernel.ingest_observer_packet`` lo consume como semilla.
    """
    spectra:      PipelineSpectraCertificate
    sections:     Tuple[PoincareSection, ...]
    return_map:   ReturnMapReport
    melnikov:     MelnikovCertificate
    kam:          KAMCertificate
    arnold:       ArnoldDiffusionMetric
    timestamp:    datetime.datetime
    state_hash:   str
    n3_feature_vector: Tuple[float, ...] = ()
    hamiltonian_mean:  float = 0.0
    hamiltonian_drift: float = 0.0

    def to_n3_seed_vector(self) -> NDArray[np.float64]:
        r"""
        Vector de rasgos \(\theta_{\mathrm{seed}}\in\mathbb{R}^6\) — continuación
        formal hacia el kernel N3 (Fase 2, método ``ingest_observer_packet``):

            \((\lambda_2,\;\nu,\;\varepsilon_{\mathrm{KAM}},\;|M|_\infty,\;D_{\mathrm{Arnold}},\;\lambda_{\mathrm{Lyap}})\).
        """
        if self.n3_feature_vector:
            return np.asarray(self.n3_feature_vector, dtype=np.float64)
        return np.array([
            float(self.spectra.fiedler_value),
            float(self.spectra.nilpotency_index),
            float(self.kam.epsilon_perturbation),
            float(self.melnikov.max_magnitude),
            float(self.arnold.total_diffusion),
            float(self.return_map.stability_exponent),
        ], dtype=np.float64)


# ── 1.5 MOTOR ESPECTRAL + POINCARÉ ──────────────────────────────────────────

class PipelineDirectorEngine:
    r"""
    Motor Espectral Ciego de Causalidad Algebraica + Analizador Poincaré.

    Ejecuta el transporte simpléctico de fase, calcula la conectividad de
    Fiedler \(\lambda_2\) sobre \(L=D-A\), captura secciones transversales
    \(\Sigma\) (incl. balance cuaterniónico DIKW ≅ ℍ), evalúa la función de
    Melnikov vía corchete de Poisson, certifica persistencia KAM diofántica
    y mide difusión de Arnold con cota de Nekhoroshev.

    El método terminal ``project_to_phase_2`` es el funtor de frontera
    \(\mathbf{Obs}\to\mathbf{Orient}\): emite un ``ObserverPacket`` que
    Fase 2 ingiere.
    """

    def __init__(self, wilkinson_tol: float = _WILKINSON_LIMIT) -> None:
        self._wilkinson_tol = wilkinson_tol

    # ── 1.5.0 Utilidades simplécticas / hipercomplejas ──────────────────
    @staticmethod
    def _hash_to_phase_coords(state_hash: str, dim: int = 4) -> NDArray[np.float64]:
        """Inmersión estable hash → coordenadas de fase en \([-1,1]^{\mathrm{dim}}\)."""
        raw = hashlib.sha256(state_hash.encode("utf-8")).digest()
        coords = np.frombuffer(raw[: dim * 4], dtype=np.uint32).astype(np.float64)
        if coords.size < dim:
            coords = np.pad(coords, (0, dim - coords.size))
        return (coords[:dim] / float(np.iinfo(np.uint32).max)) * 2.0 - 1.0

    @staticmethod
    def _section_h(
        state: StateVector,
        section_type: PoincareSectionType,
        energy_ref: float = 0.0,
    ) -> float:
        r"""
        Función de sección \(h\colon M\to\mathbb{R}\).  Ceros = \(\Sigma\).

        Corrección respecto a v5.0.0: el cardinal \(\lvert\mathrm{validated}\rvert\)
        es siempre \(\ge 0\), luego \(h_{\mathrm{old}}\cdot h_{\mathrm{new}}<0\)
        era inalcanzable. Aquí \(h\) SÍ cambia de signo.
        """
        q = state.quaternion_state()
        if section_type is PoincareSectionType.QUATERNION_BALANCE:
            return q.section_function()
        if section_type is PoincareSectionType.ENERGY_SURFACE:
            return state.computational_hamiltonian() - energy_ref
        if section_type is PoincareSectionType.STRATUM_TRANSITION:
            # Centrado en el punto medio del poset DIKW: h ∈ [-1, +1].
            n_max = max(len(_STRATUM_ORDER), 1)
            return (2.0 * float(len(state.validated_strata)) / float(n_max)) - 1.0
        if section_type is PoincareSectionType.NILPOTENCE_EVENT:
            return q.hamilton_norm() - 1.0
        if section_type is PoincareSectionType.MEMO_HIT:
            hits = sum(1.0 for sr in state.step_results if sr.memoization_hit)
            total = max(len(state.step_results), 1)
            return (hits / total) - 0.5
        if section_type is PoincareSectionType.HOMOLOGICAL_ZERO:
            return q.imag_norm() - 0.5
        return q.section_function()

    # ── 1.5.1 Auditoría espectral clásica (ahora usa Ω) ─────────────────
    def evaluate_dag_spectral_invariants(
        self,
        adjacency_A: NDArray[np.float64],
        symplectic_jacobian_M: NDArray[np.float64],
        canonical_omega: NDArray[np.float64],
    ) -> PipelineSpectraCertificate:
        r"""
        Invariantes espectrales del DAG y consistencia simpléctica.

        1. Nilpotencia de \(A\) (caracterización de aciclicidad: \(A^n=0\)).
        2. Fiedler \(\lambda_2(L_{\mathrm{sym}})\) y gap \(\lambda_2-\lambda_1\).
        3. Conectividad algebraica dirigida \(\mathrm{Re}\,\lambda_2(L_{\mathrm{dir}})\).
        4. Defecto simpléctico \(\|M^\top\Omega M-\Omega\|_F\) tras
           levantamiento cotangente si \(\dim M\neq\dim\Omega\).
        5. Deriva de Liouville \(\lvert\det M-1\rvert\).
        """
        n = int(adjacency_A.shape[0])

        A_power = np.copy(adjacency_A)
        nilpotency_index = n + 1
        is_nilpotent = False
        for k in range(1, n + 1):
            if float(la.norm(A_power, ord="fro")) <= self._wilkinson_tol:
                is_nilpotent = True
                nilpotency_index = k
                break
            A_power = A_power @ adjacency_A

        degree_vec = np.sum(np.abs(adjacency_A), axis=1)
        L_laplacian = np.diag(degree_vec) - adjacency_A
        L_sym = 0.5 * (L_laplacian + L_laplacian.T)
        eigenvalues = np.sort(np.real(la.eigvalsh(L_sym)))
        fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        spectral_gap = (
            float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0)

        # Laplaciano dirigido (out-degree): L_dir = D_out − A
        d_out = np.sum(adjacency_A, axis=1)
        L_dir = np.diag(d_out) - adjacency_A
        try:
            dir_eigs = np.sort(np.real(la.eigvals(L_dir)))
            alg_conn_dir = float(dir_eigs[1]) if len(dir_eigs) > 1 else 0.0
        except la.LinAlgError:
            alg_conn_dir = 0.0

        # Consistencia dimensional: elevar J_n a Sp(2n) si hace falta.
        J = np.asarray(symplectic_jacobian_M, dtype=np.float64)
        Omega = np.asarray(canonical_omega, dtype=np.float64)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            J = np.eye(max(n, 1), dtype=np.float64)
        if Omega.shape != (2 * J.shape[0], 2 * J.shape[0]):
            Omega = canonical_symplectic_form(J.shape[0])
            J_hat = cotangent_lift(J)
        elif J.shape[0] == Omega.shape[0]:
            J_hat = J
        else:
            J_hat = cotangent_lift(J)
            Omega = canonical_symplectic_form(J.shape[0])

        residual = J_hat.T @ Omega @ J_hat - Omega
        symplectic_defect = float(la.norm(residual, ord="fro"))
        try:
            det_M = float(la.det(J_hat))
        except la.LinAlgError:
            det_M = 0.0
        volume_drift = abs(det_M - 1.0)
        is_strictly_symplectic = symplectic_defect <= _SYMPLECTIC_DEFECT_TOL

        is_dag_valid = (is_nilpotent
                        and fiedler_value > _SPECTRAL_TOL
                        and volume_drift <= max(self._wilkinson_tol, 1e-8))

        return PipelineSpectraCertificate(
            fiedler_value=fiedler_value,
            nilpotency_index=nilpotency_index,
            symplectic_volume_drift=volume_drift,
            is_dag_acyclic=is_dag_valid,
            symplectic_defect=symplectic_defect,
            spectral_gap=spectral_gap,
            algebraic_connectivity_directed=alg_conn_dir,
            is_strictly_symplectic=is_strictly_symplectic,
        )

    # ── 1.5.2 Flujo de Ricci discreto (preservado) ──────────────────────
    def apply_discrete_ricci_flow(
        self,
        G_k: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        delta_tau: float = 0.01,
    ) -> NDArray[np.float64]:
        r"""
        Flujo de Ricci discreto \(\partial_\tau g=-2\mathrm{Ric}+\frac{2\bar R}{n}g\).
        Preserva el volumen conforme a Hamilton–Chow (versión combinatoria).
        """
        n = G_k.shape[0]
        r_avg = float(np.trace(ricci_tensor)) / float(max(n, 1))
        return G_k - 2.0 * delta_tau * (ricci_tensor - (r_avg / float(n)) * G_k)

    # ── 1.5.3 Flujo de Brockett-Shahshahani (preservado) ────────────────
    def compute_brockett_shahshahani_flow(
        self,
        p_vec: NDArray[np.float64],
        rho_mat: NDArray[np.float64],
        K_kernel: NDArray[np.float64],
        dt: float = 0.001,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Flujo replicador de Shahshahani \(\dot p_i=p_i\bigl((Kp)_i-p^\top Kp\bigr)\)
        acoplado al conmutador de Brockett sobre densidades \(\rho\).
        Conserva \(\sum p_i=1\) y \(\mathrm{tr}\,\rho=1\).
        """
        Kp = K_kernel @ p_vec
        p_Kp = float(p_vec @ Kp)
        dp_dt = p_vec * (Kp - p_Kp)
        p_next = np.clip(p_vec + dt * dp_dt, 1e-12, None)
        p_next = p_next / np.sum(p_next)

        diag_p = np.diag(p_next)
        inner_comm = rho_mat @ diag_p - diag_p @ rho_mat
        drho_dt = rho_mat @ inner_comm - inner_comm @ rho_mat
        rho_next = rho_mat + dt * drho_dt
        trace_rho = float(np.trace(rho_next))
        if abs(trace_rho) > 1e-12:
            rho_next = rho_next / trace_rho
        return p_next, rho_next

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.4  SECCIÓN DE POINCARÉ — captura de la aplicación de retorno
    # ═══════════════════════════════════════════════════════════════════
    def capture_poincare_section(
        self,
        trajectory: Sequence[StateVector],
        section_type: PoincareSectionType = PoincareSectionType.STRATUM_TRANSITION,
        energy_ref: Optional[float] = None,
    ) -> List[PoincareSection]:
        r"""
        Calcula \(\Sigma=h^{-1}(0)\) y los cruces de \(\gamma(t)\) con \(\Sigma\).

        Detección por cambio de signo de \(h(\gamma_i)\cdot h(\gamma_{i+1})<0\)
        (ahora alcanzable: \(h\) está centrada y toma valores en \(\mathbb{R}\))
        más interpolación lineal de Newton de un paso:

            \(t_*=t_i-h_i/(h_{i+1}-h_i)\).

        Transversalidad: \(\lvert h_{i+1}-h_i\rvert>\texttt{_POINCARE_SECTION_TOL}\).
        El tangente se embebe en \(\mathbb{H}\simeq\mathbb{R}^4\) vía el producto
        de Hamilton \(q_i^{-1}q_{i+1}\) (incremento hipercomplejo de estrato).
        """
        if len(trajectory) < 2:
            return []

        if energy_ref is None:
            energies = [s.computational_hamiltonian() for s in trajectory]
            energy_ref = float(np.mean(energies)) if energies else 0.0

        sections: List[PoincareSection] = []
        prev_state = trajectory[0]
        prev_h = self._section_h(prev_state, section_type, energy_ref)

        for i, state in enumerate(trajectory[1:], start=1):
            curr_h = self._section_h(state, section_type, energy_ref)
            crossed = (prev_h * curr_h < 0.0) or (
                abs(prev_h) <= _POINCARE_SECTION_TOL < abs(curr_h)
            )
            if not crossed:
                prev_h = curr_h
                prev_state = state
                continue

            denom = (curr_h - prev_h) if abs(curr_h - prev_h) > EPSILON else 1.0
            t_star = float(i - 1) + float(-prev_h / denom)
            h_star = 0.0  # por construcción de la interpolación lineal

            q_prev = prev_state.quaternion_state()
            q_curr = state.quaternion_state()
            # Tangente hipercomplejo: incremento de Hamilton (q_prev̄ * q_curr).
            q_prev_conj = QuaternionStratumState(
                w=q_prev.w, x=-q_prev.x, y=-q_prev.y, z=-q_prev.z)
            dq = q_prev_conj.hamilton_product(q_curr)
            tangent = dq.as_array()
            dh_dt = abs(curr_h - prev_h)
            is_transverse = dh_dt > _POINCARE_SECTION_TOL

            if not is_transverse:
                # Cruce tangente: se registra pero se marca no-transversal.
                logger.debug(
                    "[Σ] cruce no transversal en t*=%.4f  dh/dt=%.3e",
                    t_star, dh_dt)

            sections.append(PoincareSection(
                section_type=section_type,
                state_hash=state.compute_hash(),
                t_crossing=t_star,
                h_value=h_star,
                tangent=tangent,
                quaternion=tuple(q_curr.as_array().tolist()),  # type: ignore[arg-type]
                energy=state.computational_hamiltonian(),
                is_transverse=is_transverse,
                metadata={
                    "h_before": prev_h,
                    "h_after":  curr_h,
                    "validated_before": float(len(prev_state.validated_strata)),
                    "validated_after":  float(len(state.validated_strata)),
                    "step": (state.step_results[-1].step_name
                             if state.step_results else "unknown"),
                },
            ))
            prev_h = curr_h
            prev_state = state

        return sections

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.5  APLICACIÓN DE RETORNO P: Σ → Σ  y análisis de Lyapunov
    # ═══════════════════════════════════════════════════════════════════
    def analyze_return_map(
        self, sections: Sequence[PoincareSection],
    ) -> ReturnMapReport:
        r"""
        Dada la sucesión de cruces \(p_0,p_1,\dots,p_{N-1}\in\Sigma\), estima:

          · Punto fijo \(p^*\)  ←  baricentro de los \(p_j\) más próximos.
          · Jacobiano discreto \(DP_{p_j}\approx(\tau_{j+1}-\tau_j)/(\tau_j-\tau_{j-1})\)
            sobre el tangente hipercomplejo.
          · Lyapunov normal \(\lambda=N^{-1}\sum\log\|DP_{p_j}\|\).
          · Espectro de \(DP\) (autovalores complejos del mapa 4-dimensional).
          · Periodicidad: \(\mathrm{std}(\Delta t)<\texttt{_POINCARE_SECTION_TOL}\).
        """
        if len(sections) < 2:
            return ReturnMapReport(
                n_crossings=len(sections),
                fixed_point_estimate=None,
                stability_exponent=0.0,
                is_hyperbolic=False,
                is_periodic_orbit=False,
            )

        tangents = np.stack(
            [np.asarray(s.tangent, dtype=np.float64).reshape(-1) for s in sections])
        # Igualar dimensiones (padding).
        max_d = max(int(t.size) for t in tangents)
        padded = np.zeros((len(sections), max_d), dtype=np.float64)
        for i, t in enumerate(tangents):
            padded[i, : t.size] = t

        ratios: List[float] = []
        dt_list: List[float] = []
        for a, b in zip(sections[:-1], sections[1:]):
            dt = abs(b.t_crossing - a.t_crossing)
            dt_list.append(dt)
            dh = float(np.linalg.norm(np.asarray(b.tangent) - np.asarray(a.tangent)))
            if dt > EPSILON:
                ratios.append(dh / dt + EPSILON)
            else:
                ratios.append(EPSILON)

        ratios_arr = np.asarray(ratios if ratios else [1.0], dtype=np.float64)
        lam = float(np.mean(np.log(np.clip(ratios_arr, EPSILON, None))))
        is_hyp = abs(lam) > _POINCARE_SECTION_TOL
        dt_arr = np.asarray(dt_list if dt_list else [0.0], dtype=np.float64)
        is_periodic = (len(sections) >= 3
                       and float(np.std(dt_arr)) < max(_POINCARE_SECTION_TOL, 1e-4))
        period = float(np.mean(dt_arr)) if is_periodic and dt_arr.size else None

        # Jacobiano empirico: mapa lineal de tangentes consecutivos (lstsq).
        jac_eigs: Optional[Tuple[complex, ...]] = None
        lyap_spec: Optional[Tuple[float, ...]] = None
        if len(sections) >= 3 and padded.shape[1] >= 1:
            X, Y = padded[:-1], padded[1:]
            try:
                # DP ≈ argmin ||Y − X·DP||  (pseudoinversa a derecha)
                dp, *_ = np.linalg.lstsq(X, Y, rcond=None)
                eigs = la.eigvals(dp) if dp.shape[0] == dp.shape[1] else np.array([])
                if eigs.size:
                    jac_eigs = tuple(complex(z) for z in eigs)
                    lyap_spec = tuple(float(np.log(abs(z) + EPSILON)) for z in eigs)
            except (la.LinAlgError, ValueError):
                jac_eigs = None

        fixed_point = None
        if is_periodic or len(sections) >= 2:
            fixed_point = np.mean(
                np.array([[s.t_crossing] for s in sections], dtype=np.float64),
                axis=0)

        return ReturnMapReport(
            n_crossings=len(sections),
            fixed_point_estimate=fixed_point,
            stability_exponent=lam,
            is_hyperbolic=is_hyp,
            is_periodic_orbit=is_periodic,
            jacobian_eigenvalues=jac_eigs,
            poincare_period=period,
            lyapunov_spectrum=lyap_spec,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.6  FUNCIÓN DE MELNIKOV — detección de caos homoclínico
    # ═══════════════════════════════════════════════════════════════════
    def evaluate_melnikov_function(
        self,
        trajectory: Sequence[StateVector],
        omega_perturbation: float = 1.0,
    ) -> MelnikovCertificate:
        r"""
        Discretización del corchete de Poisson a lo largo de \(\gamma_0\):

            \(M(t_0)\approx\sum_k\{H_0,H_1\}(\gamma_k,\,t_k+t_0)\,\Delta t\).

        Campo no perturbado \(X_{H_0}\): gradiente de \(H\) en coordenadas
        de fase extraídas del hash (inmersión estable).
        Perturbación \(H_1(\,\cdot\,,t)=\cos(\omega t)\,h_{\mathbb{H}}(q)\):
        forzamiento periódico del balance cuaterniónico.

        Ceros simples: cambio de signo + \(\lvert\partial_t M\rvert>\texttt{_MELNIKOV_DMDT_TOL}\).
        """
        if len(trajectory) < 3:
            return MelnikovCertificate(
                melnikov_values=np.zeros(1),
                simple_zeros=0, mean_magnitude=0.0,
                max_magnitude=0.0, tangle_detected=False,
            )

        n_seg = len(trajectory) - 1
        dim = 4
        Omega = canonical_symplectic_form(dim // 2)  # Sp(4) sobre ℍ ≃ ℝ⁴
        # Coordenadas de fase por estado (hash ↦ ℝ⁴) + hamiltoniano.
        phase = np.stack([
            self._hash_to_phase_coords(s.compute_hash(), dim=dim)
            for s in trajectory
        ])  # (N, 4)
        H0 = np.array([s.computational_hamiltonian() for s in trajectory],
                      dtype=np.float64)
        h_quat = np.array(
            [s.quaternion_state().section_function() for s in trajectory],
            dtype=np.float64)

        # ∇H₀ ≈ diferencias finitas en fase (proyección a T*Q).
        d_phase = np.diff(phase, axis=0)                      # (N-1, 4)
        d_H0 = np.diff(H0)                                    # (N-1,)
        # Gradiente direccional: dH ≈ ⟨∇H, Δx⟩ ⇒ ∇H ~ (dH / ||Δx||²) Δx
        norms2 = np.sum(d_phase * d_phase, axis=1, keepdims=True) + EPSILON
        grad_H0 = (d_H0[:, None] / norms2) * d_phase          # (N-1, 4)

        t_axis = np.arange(n_seg, dtype=np.float64)
        periods = max(8, int(round(2.0 * np.pi / max(omega_perturbation, EPSILON))))
        periods = min(periods, 64)
        melnikov_vals = np.zeros(periods, dtype=np.float64)
        poisson_acc = 0.0
        dt = 1.0

        for j in range(periods):
            t0 = j * (2.0 * np.pi / (omega_perturbation * periods))
            # ∇H₁: perturbación periódica del balance cuaterniónico.
            forcing = np.cos(omega_perturbation * t_axis + t0)
            d_h = np.diff(h_quat)
            grad_H1 = ((d_h[:, None] / norms2) * d_phase) * forcing[:, None]
            # {H₀, H₁} = ∇H₀ᵀ Ωᵀ ∇H₁   (Ω es 4×4)
            pb = np.einsum("ni,ij,nj->n", grad_H0, Omega.T, grad_H1)
            melnikov_vals[j] = float(np.sum(pb) * dt)
            poisson_acc += float(np.mean(pb))

        poisson_mean = poisson_acc / float(periods)

        # Ceros simples: interpolación + test de derivada.
        dM = np.gradient(melnikov_vals)
        zero_locs: List[float] = []
        dM_at: List[float] = []
        for i in range(len(melnikov_vals) - 1):
            a, b = melnikov_vals[i], melnikov_vals[i + 1]
            if a == 0.0:
                dM_star = float(dM[i])
                if abs(dM_star) > _MELNIKOV_DMDT_TOL:
                    zero_locs.append(float(i))
                    dM_at.append(dM_star)
                continue
            if a * b < 0.0:
                frac = float(-a / (b - a)) if abs(b - a) > EPSILON else 0.5
                dM_star = float((1.0 - frac) * dM[i] + frac * dM[i + 1])
                if abs(dM_star) > _MELNIKOV_DMDT_TOL:
                    zero_locs.append(float(i + frac))
                    dM_at.append(dM_star)

        simple_zeros = len(zero_locs)
        mean_mag = float(np.mean(np.abs(melnikov_vals)))
        max_mag = float(np.max(np.abs(melnikov_vals)))
        tangle = simple_zeros > 0 and mean_mag > _MELNIKOV_TOL

        return MelnikovCertificate(
            melnikov_values=melnikov_vals,
            simple_zeros=simple_zeros,
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            tangle_detected=tangle,
            simple_zero_locations=tuple(zero_locs),
            melnikov_derivative_at_zeros=tuple(dM_at),
            poisson_mean=poisson_mean,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.7  CERTIFICADO KAM — persistencia de toros invariantes
    # ═══════════════════════════════════════════════════════════════════
    def certify_KAM_persistence(
        self,
        spectra: PipelineSpectraCertificate,
        kam_threshold: float = 1.0e-3,
        frequency_vector: Optional[NDArray[np.float64]] = None,
    ) -> KAMCertificate:
        r"""
        \(\varepsilon=\|H_1\|_\infty/\omega_{\min}\)  ← razón de perturbación.

        Aquí:
            \(\|H_1\|=\max(\mathrm{drift},\,\mathrm{defect})\)  (perturbación del flujo)
            \(\omega_{\min}=\lambda_2\)                         (frecuencia Fiedler)

        Condición diofántica de Kolmogorov (truncada a \(|k|_\infty\le 4\)):

            \(|\langle k,\omega\rangle|\ge\gamma/|k|_1^\tau\)  \(\forall k\in\mathbb{Z}^n\setminus\{0\}\).

        Residuo de Greene (proxy): \(R\approx\varepsilon/\mathrm{gap}\).
        Si \(\varepsilon<\varepsilon_{\mathrm{KAM}}\) y \(\omega\) es diofántica,
        los estratos DIKW sobreviven como toros KAM.
        """
        omega_min = max(spectra.fiedler_value, _SPECTRAL_TOL)
        eps_pert = max(spectra.symplectic_volume_drift,
                       spectra.symplectic_defect) / omega_min
        eps_KAM = max(kam_threshold, _KAM_PERSISTENCE_TOL)

        if frequency_vector is None:
            # Frecuencias naturales del poset DIKW, perturbadas por el gap.
            gap = max(spectra.spectral_gap, _SPECTRAL_TOL)
            frequency_vector = np.array(
                [omega_min * (1.0 + k) + 0.1 * gap * k for k in range(4)],
                dtype=np.float64)
        omega = np.asarray(frequency_vector, dtype=np.float64)
        n_freq = int(omega.size)

        diophantine_held = True
        gamma, tau = _KAM_DIOPH_GAMMA, _KAM_DIOPH_TAU
        # Barrido acotado del retículo: k ∈ {−2,…,2}^n \ {0}, n≤4.
        from itertools import product as _product
        bound = 2
        for k_tuple in _product(range(-bound, bound + 1), repeat=n_freq):
            if all(ki == 0 for ki in k_tuple):
                continue
            k_vec = np.asarray(k_tuple, dtype=np.float64)
            lhs = abs(float(k_vec @ omega))
            rhs = gamma / (float(np.sum(np.abs(k_vec))) ** tau + EPSILON)
            if lhs < rhs:
                diophantine_held = False
                break

        greene_residue = float(eps_pert / max(spectra.spectral_gap, _SPECTRAL_TOL))
        n_tori = max(0, int(round(4.0 * max(0.0, 1.0 - eps_pert / eps_KAM))))
        is_stable = bool(eps_pert < eps_KAM and diophantine_held)

        return KAMCertificate(
            epsilon_perturbation=float(eps_pert),
            epsilon_KAM_threshold=float(eps_KAM),
            tori_persisting=n_tori,
            is_KAM_stable=is_stable,
            diophantine_held=diophantine_held,
            frequency_vector=tuple(float(x) for x in omega.tolist()),
            greene_residue=greene_residue,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.8  DIFUSIÓN DE ARNOLD — fuga inter-estrato
    # ═══════════════════════════════════════════════════════════════════
    def measure_arnold_diffusion(
        self,
        trajectory: Sequence[StateVector],
        spectra: PipelineSpectraCertificate,
    ) -> ArnoldDiffusionMetric:
        r"""
        \(D_{\mathrm{Arnold}}^{(s)}=\sum_t\lvert\Delta\beta_1^{(s)}(t)\rvert^2/\omega_s^2\).

        \(\Delta\beta_1^{(s)}\) es la variación de pertenencia del estrato \(s\)
        entre pasos consecutivos; \(\omega_s=\lambda_2\cdot(1+\mathrm{ord}(s))\).

        Tiempo de Nekhoroshev (estabilidad efectiva):

            \(T_{\mathrm{Nek}}=\exp\bigl((1/\varepsilon)^{1/(2n)}\bigr)\),
            \(\varepsilon=\max(D_{\mathrm{Arnold}},\texttt{EPSILON})\), \(n=4\).
        """
        per_stratum: Dict[str, float] = {}
        omega_base = max(spectra.fiedler_value, _SPECTRAL_TOL)
        n_resonant = 0

        for stratum in Stratum:
            omega_s = omega_base * (1.0 + _STRATUM_ORDER.get(stratum, 0))
            accum = 0.0
            for a, b in zip(trajectory[:-1], trajectory[1:]):
                in_a = 1.0 if stratum in a.validated_strata else 0.0
                in_b = 1.0 if stratum in b.validated_strata else 0.0
                delta = in_b - in_a
                accum += (delta * delta) / (omega_s * omega_s + EPSILON)
            per_stratum[stratum.name] = float(accum)
            if accum > _ARNOLD_DIFFUSION_TOL:
                n_resonant += 1

        total = float(sum(per_stratum.values()))
        dominant = (max(per_stratum.items(), key=lambda kv: kv[1])[0]
                    if per_stratum else None)
        is_adiabatic = total < _ARNOLD_DIFFUSION_TOL

        # Nekhoroshev: n_eff = 4 (DIKW); recorte numérico para evitar overflow.
        n_eff = 4.0
        eps_ad = max(total, EPSILON)
        exponent = (1.0 / eps_ad) ** (1.0 / (2.0 * n_eff))
        exponent = min(exponent, 80.0)  # exp(80) ~ 5e34, cota segura
        t_nek = float(math.exp(exponent)) if is_adiabatic else float(exponent)

        return ArnoldDiffusionMetric(
            per_stratum_leakage=per_stratum,
            total_diffusion=total,
            dominant_channel=dominant,
            is_adiabatic=is_adiabatic,
            nekhoroshev_time=t_nek,
            n_resonant_channels=n_resonant,
        )

    # ═══════════════════════════════════════════════════════════════════
    # 1.5.9  project_to_phase_2  ←  CIERRE DE FASE 1 / INICIO DE FASE 2
    # ═══════════════════════════════════════════════════════════════════
    def project_to_phase_2(
        self,
        trajectory: Sequence[StateVector],
        adjacency_A: Optional[NDArray[np.float64]] = None,
        symplectic_jacobian_M: Optional[NDArray[np.float64]] = None,
        canonical_omega: Optional[NDArray[np.float64]] = None,
        section_type: PoincareSectionType = PoincareSectionType.STRATUM_TRANSITION,
    ) -> ObserverPacket:
        r"""
        ← ÚLTIMO MÉTODO DE FASE 1.  FUNCTOR DE FRONTERA  Obs ⟶ Orient.

        Sintetiza TODAS las observaciones (espectro, \(\Sigma\), \(P\), \(M\),
        KAM, Arnold, \(H\), \(\theta_{\mathrm{seed}}\)) en un ``ObserverPacket``
        inmutable. La FASE 2 arranca formalmente al consumir este paquete
        mediante ``Phase2OrientKernel.ingest_observer_packet`` (contrato
        declarado inmediatamente debajo).

        Diagrama de continuación:

            project_to_phase_2(γ, A, M, Ω)  ──ObserverPacket──►
                Phase2OrientKernel.ingest_observer_packet
                    └──► N1/N2/N3  └──► OrientationSeed
                                          └──► Fase 3
        """
        n = len(trajectory)
        dim = max(n, 1)
        if adjacency_A is None:
            adjacency_A = np.zeros((dim, dim), dtype=np.float64)
        k = int(adjacency_A.shape[0])
        if symplectic_jacobian_M is None:
            symplectic_jacobian_M = np.eye(k, dtype=np.float64)
        if canonical_omega is None:
            canonical_omega = canonical_symplectic_form(k)

        spectra = self.evaluate_dag_spectral_invariants(
            adjacency_A, symplectic_jacobian_M, canonical_omega)
        sections = self.capture_poincare_section(trajectory, section_type)
        return_map = self.analyze_return_map(sections)
        melnikov = self.evaluate_melnikov_function(trajectory)
        kam = self.certify_KAM_persistence(spectra)
        arnold = self.measure_arnold_diffusion(trajectory, spectra)

        energies = [s.computational_hamiltonian() for s in trajectory] or [0.0]
        H_mean = float(np.mean(energies))
        H_drift = float(np.max(energies) - np.min(energies)) if energies else 0.0

        n3_vec = (
            float(spectra.fiedler_value),
            float(spectra.nilpotency_index),
            float(kam.epsilon_perturbation),
            float(melnikov.max_magnitude),
            float(arnold.total_diffusion),
            float(return_map.stability_exponent),
        )

        packet = ObserverPacket(
            spectra=spectra,
            sections=tuple(sections),
            return_map=return_map,
            melnikov=melnikov,
            kam=kam,
            arnold=arnold,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            state_hash=trajectory[-1].compute_hash() if trajectory else "",
            n3_feature_vector=n3_vec,
            hamiltonian_mean=H_mean,
            hamiltonian_drift=H_drift,
        )

        if not spectra.is_dag_acyclic:
            logger.warning(
                "[Fase 1::OBSERVE] DAG no acíclico (ν=%d, λ₂=%.3e, drift=%.3e)",
                spectra.nilpotency_index, spectra.fiedler_value,
                spectra.symplectic_volume_drift)
        if melnikov.tangle_detected:
            logger.warning(
                "[Fase 1::OBSERVE] Tangencia homoclínica: %d ceros simples, |M|_∞=%.3e",
                melnikov.simple_zeros, melnikov.max_magnitude)
        if not kam.is_KAM_stable:
            logger.warning(
                "[Fase 1::OBSERVE] Quiebre KAM: ε=%.3e ≰ ε_KAM=%.3e  diofántica=%s",
                kam.epsilon_perturbation, kam.epsilon_KAM_threshold,
                kam.diophantine_held)

        return packet


# ── 1.6 LOGGING ─────────────────────────────────────────────────────────────

def configure_pipeline_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)


configure_pipeline_logging(os.getenv("LOG_LEVEL", "INFO"))


# ── 1.7 CONTRATO DE CONTINUACIÓN  FASE 1 → FASE 2 ───────────────────────────

@runtime_checkable
class Phase2OrientKernel(Protocol):
    r"""
    ════════════════════════════════════════════════════════════════════════
    CONTRATO FORMAL DE FRONTERA  —  último objeto de la Fase 1
                                    ≡ primer objeto de la Fase 2
    ════════════════════════════════════════════════════════════════════════

    El funtor ``project_to_phase_2`` produce un ``ObserverPacket``.  Toda
    implementación de la Fase 2 (ORIENT) DEBE satisfacer este protocolo:

        ingest_observer_packet(packet)  :  traduce el paquete en (θ, O, α)
        project_to_phase_3(...)         :  cierra N3 y emite OrientationSeed

    Ecuación funcional que el kernel N3 (Fase 2) debe respetar:

        \(I_3\circ I_3=I_3\)     (idempotencia asintótica, tol = _N3_IDEMPOTENCE_TOL)
        \(\partial I_3/\partial\alpha\neq 0\)   (no-trivialidad)

    La definición de ``RecursiveSelfImprovementKernel`` — primer método
    concreto: ``ingest_observer_packet`` — CONTINÚA EN LA FASE 2.
    """

    def ingest_observer_packet(self, packet: ObserverPacket) -> None:
        """Traduce ObserverPacket → estado interno (θ, O, α) del kernel N3."""
        ...

    def project_to_phase_3(
        self,
        observer_packet: ObserverPacket,
        n_cycles: int = 3,
    ) -> Any:
        """Cierra N1→N2→N3 y emite OrientationSeed (frontera Fase 2 → Fase 3)."""
        ...


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FIN DE FASE 1 (OBSERVE)                                                █
# █   Continúa en FASE 2 (ORIENT):                                           █
# █     AlgebraicDAG · OperatorMemoizer · HomologicalAuditor                 █
# █     RecursiveSelfImprovementKernel.ingest_observer_packet  ← 1.er método █
# █     RecursiveSelfImprovementKernel.project_to_phase_3      ← frontera F3 █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████
# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► ORIENT : CLAUSURA TRANSITIVA + AUTOMEJORA RECURSIVA N3      █
# █   (Warshall  +  Poset DIKW  +  N1/N2/N3 bilevel  +  Poincaré–N3)         █
# █                                                                          █
# █   Contrato de frontera Fase 1 → Fase 2:                                  █
# █     Phase2OrientKernel.ingest_observer_packet  ← PRIMER MÉTODO           █
# █     Phase2OrientKernel.project_to_phase_3      ← ÚLTIMO MÉTODO           █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 2.0 ESTADO INTERNO Y CERTIFICADOS N3 ─────────────────────────────────────

@dataclass
class ImprovementState:
    r"""
    Estado interno del kernel N3 — punto de \(T^*\Theta\times\mathfrak{n}_n\times A\).

    \(\theta\in\mathbb{R}^d\)     : parámetros del pipeline (umbrales, ganancias)
    \(O\in\mathfrak{n}_n\subset M_n(\mathbb{R})\) : adyacencia estrictamente
                                    triangular superior (ideal nilpotente)
    \(\alpha\in\mathbb{R}^3_{>0}\) : meta-parámetros \((\alpha_1,\alpha_2,\alpha_3)\)
                                    que gobiernan \(\nabla\theta\), \(\nabla O\) y \(\nabla\alpha\)

    El álgebra de Banach \(\mathfrak{B}=(M_n(\mathbb{R}),\|\cdot\|_2)\) restringida
    a \(\mathfrak{n}_n=\{X:X_{ij}=0\text{ si }i\ge j\}\) satisface \(X^n=0\)
    (índice de nilpotencia \(\le n\)), de modo que todo \(O\) es un DAG.
    """
    theta: NDArray[np.float64]
    O:     NDArray[np.float64]
    alpha: NDArray[np.float64]
    loss_history:    List[float] = field(default_factory=list)
    level_history:   List[int]   = field(default_factory=list)
    theta_history:   List[NDArray[np.float64]] = field(default_factory=list)
    alpha_history:   List[NDArray[np.float64]] = field(default_factory=list)
    O_norm_history:  List[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class N3IdempotenceCertificate:
    r"""
    Certificado de idempotencia asintótica \(I_3\circ I_3=I_3\).

    Residuo:
        \(r=\bigl\|I_3(I_3(z))-I_3(z)\bigr\|_{\theta,O,\alpha}\)
        con \(\|z\|^2=\|\theta\|_2^2+\|O\|_F^2+\|\alpha\|_2^2\).

    \(r\le\texttt{_N3_IDEMPOTENCE_TOL}\) \(\Rightarrow\) \(I_3\) es una
    proyección (aprox.) sobre la variedad de puntos críticos locales.
    Espectro de un idempotente en álgebra de Banach: \(\sigma(I_3)\subseteq\{0,1\}\).
    """
    residual:          float
    residual_theta:    float
    residual_O:        float
    residual_alpha:    float
    is_idempotent:     bool
    n_extra_cycles:    int = 1


@dataclass(frozen=True, slots=True)
class N3NontrivialityCertificate:
    r"""
    Certificado de no-trivialidad \(\partial I_3/\partial\alpha\neq 0\).

    Si \(I_3\) no dependiera de \(\alpha\), el nivel 3 sería vacuamente N2.
    Se estima el jacobiano \(\mathrm{D}_\alpha I_3=(\partial\theta^*/\partial\alpha,\,
    \partial O^*/\partial\alpha)\) por diferencias centrales y se exige
    \(\|\mathrm{D}_\alpha I_3\|_F>\texttt{EPSILON}\).
    """
    jacobian_frobenius: float
    dtheta_dalpha_norm: float
    dO_dalpha_norm:     float
    is_nontrivial:      bool


@dataclass(frozen=True, slots=True)
class N3PoincareSection:
    r"""Cruce del flujo N3 con \(\{z:h_{\mathrm{N3}}(z)=0\}\), \(h=L-L_{\mathrm{ref}}\)."""
    t_index:     int
    loss_value:  float
    alpha_at:    Tuple[float, ...]
    turning:     str            # "min" | "max"


@dataclass(frozen=True, slots=True)
class N3LyapunovCertificate:
    r"""
    Función de Lyapunov discreta del bucle N3:

        \(V_t=L(\theta_t,O_t,\alpha_t)+\tfrac12\kappa\|\alpha_t-\alpha_{t-1}\|_2^2\).

    Convergencia monótona ssi \(V_{t+1}\le V_t-\eta\|\nabla L\|^2\) (descenso suficiente).
    """
    V_final:           float
    V_initial:         float
    is_monotone:       bool
    mean_descent:      float
    n_violations:      int


# ── 2.1 POSET DIKW Y ÁLGEBRA DE INCIDENCIA ───────────────────────────────────

class DIKWPoset:
    r"""
    Poset lineal de filtración

        \(\mathrm{PHYSICS}\;\le\;\mathrm{TACTICS}\;\le\;\mathrm{STRATEGY}\;\le\;\mathrm{WISDOM}\).

    Álgebra de incidencia \(I(P)=\{\,f:P\times P\to\mathbb{R}:f(x,y)=0\text{ si }x\nleq y\,\}\)
    con convolución \((f*g)(x,y)=\sum_{x\le z\le y}f(x,z)g(z,y)\).

    Función zeta \(\zeta(x,y)=1_{x\le y}\); Möbius \(\mu=\zeta^{-1}\).
    En una cadena, \(\mu(x,x)=1\), \(\mu(x,s(x))=-1\), \(\mu=0\) en el resto.
    """

    def __init__(self) -> None:
        self._order: List[Stratum] = sorted(
            _STRATUM_ORDER.keys(), key=lambda s: _STRATUM_ORDER[s])
        self._idx: Dict[Stratum, int] = {s: i for i, s in enumerate(self._order)}
        self._n = len(self._order)

    def __len__(self) -> int:
        return self._n

    def leq(self, a: Stratum, b: Stratum) -> bool:
        return stratum_level(a) <= stratum_level(b)

    def covers(self) -> List[Tuple[Stratum, Stratum]]:
        """Relaciones de cubrimiento \(x\prec y\) (Hasse)."""
        return list(zip(self._order[:-1], self._order[1:]))

    def height(self, s: Stratum) -> int:
        return stratum_level(s)

    def rank(self) -> int:
        return self._n - 1

    def zeta_matrix(self) -> NDArray[np.float64]:
        r"""\(\zeta_{ij}=1\) ssi \(i\le j\) en el orden lineal."""
        Z = np.triu(np.ones((self._n, self._n), dtype=np.float64), k=0)
        return Z

    def mobius_matrix(self) -> NDArray[np.float64]:
        r"""
        \(\mu=\zeta^{-1}\) sobre \(\mathbb{R}\).  Para la cadena de longitud \(n\):

            \(\mu=I-N\)  con \(N\) el shift nilpotente (\(N_{i,i+1}=1\)).
        """
        try:
            return np.linalg.inv(self.zeta_matrix())
        except np.linalg.LinAlgError:
            mu = np.eye(self._n, dtype=np.float64)
            for i in range(self._n - 1):
                mu[i, i + 1] = -1.0
            return mu

    def incidence_convolve(
        self,
        f: NDArray[np.float64],
        g: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Convolución de incidencia \((f*g)(x,y)=\sum_{x\le z\le y}f(x,z)g(z,y)\)."""
        n = self._n
        out = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                out[i, j] = float(np.dot(f[i, i:j + 1], g[i:j + 1, j]))
        return out

    def filtration_ok(self, validated: Set[Stratum]) -> Tuple[bool, List[str]]:
        r"""
        Axioma de filtración: si \(s\in V\) entonces \(\forall s'<s\), \(s'\in V\).
        Equivale a \(\mu*1_V\) soportado solo en maximales (inversión de Möbius).
        """
        missing: List[str] = []
        for s in list(validated):
            for t in self._order:
                if self.leq(t, s) and t not in validated and t is not s:
                    missing.append(t.name)
        missing = sorted(set(missing), key=lambda n: _STRATUM_ORDER.get(
            getattr(Stratum, n, s), 99))
        return (len(missing) == 0, missing)


# ── 2.2 DAG ALGEBRAICO (categoría Rel, semianillos Bool y tropical) ──────────

@dataclass
class DependencyEdge:
    source:    str
    target:    str
    data_keys: List[str]
    optional:  bool = False

    def __hash__(self) -> int:
        return hash((self.source, self.target))


class AlgebraicDAG:
    r"""
    Categoría libre generada por un grafo acíclico (quiver acíclico).

    Objetos = pasos; morfismos = caminos.  La clausura transitiva es el
    semigrupo de caminos:

        \(A^*=\bigvee_{k=0}^{n-1}A^k\)   en el semianillo \(\mathbb{B}=(\{0,1\},\lor,\land)\).

    En el semianillo tropical \((\mathbb{R}\cup\{\infty\},\min,+)\) la misma
    recurrencia de Floyd–Warshall produce distancias geodésicas.

    La reducción transitiva es el diagrama de Hasse del poset de alcanzabilidad.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.dependencies:      Dict[str, Set[str]] = {}
        self.data_requirements: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_step(self, step_name: str,
                 data_keys: Optional[List[str]] = None) -> None:
        self.graph.add_node(step_name)
        self.dependencies.setdefault(step_name, set())
        self.data_requirements.setdefault(step_name, set(data_keys or []))

    def add_dependency(self, source: str, target: str,
                       data_keys: Optional[List[str]] = None,
                       optional: bool = False) -> None:
        if not self.graph.has_node(source):
            self.add_step(source)
        if not self.graph.has_node(target):
            self.add_step(target)
        if nx.has_path(self.graph, target, source):
            raise DependencyResolutionError(
                f"Cycle detected: {target} → {source}", [source, target])
        self.graph.add_edge(source, target,
                            data_keys=data_keys or [], optional=optional)
        self.dependencies[target].add(source)
        if data_keys:
            self.data_requirements[target].update(data_keys)

    def topological_sort(self) -> List[str]:
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise DependencyResolutionError(
                f"Topological sort failed: {e}", list(self.graph.nodes()))

    def get_dependencies(self, step_name: str) -> Set[str]:
        return self.dependencies.get(step_name, set())

    def get_data_requirements(self, step_name: str) -> Set[str]:
        return self.data_requirements.get(step_name, set())

    def validate(self) -> bool:
        if not nx.is_directed_acyclic_graph(self.graph):
            raise DependencyResolutionError("DAG contains cycles")
        if (len(self.graph.nodes()) > 1
                and not nx.is_weakly_connected(self.graph)):
            self.logger.warning("DAG is not weakly connected")
        return True

    def nodes_in_order(self) -> List[str]:
        try:
            return self.topological_sort()
        except DependencyResolutionError:
            return list(self.graph.nodes())

    def adjacency_matrix(self) -> Tuple[NDArray[np.float64], List[str]]:
        nodes = self.nodes_in_order()
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in self.graph.edges():
            A[idx[u], idx[v]] = 1.0
        return A, nodes

    @classmethod
    def from_adjacency(
        cls,
        A: NDArray[np.float64],
        nodes: Optional[Sequence[str]] = None,
        weight_tol: float = EPSILON,
    ) -> "AlgebraicDAG":
        dag = cls()
        n = int(A.shape[0])
        labels = list(nodes) if nodes is not None else [f"n{i}" for i in range(n)]
        for lab in labels:
            dag.add_step(lab)
        for i in range(n):
            for j in range(n):
                if float(A[i, j]) > weight_tol:
                    dag.add_dependency(labels[i], labels[j])
        return dag

    def warshall_transitive_closure(self) -> NDArray[np.bool_]:
        r"""
        Clausura transitiva de Warshall en \(\mathbb{B}\).

            \(W\leftarrow A\)
            \(\forall k:\; W_{ij}\leftarrow W_{ij}\lor(W_{ik}\land W_{kj})\).

        Vectorizado: \(W\leftarrow W\lor (W_{\cdot k}\,W_{k\cdot})\).
        \(W_{ii}=\mathrm{True}\) ssi hay un ciclo que pasa por \(i\).
        """
        nodes = self.nodes_in_order()
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        W = np.zeros((n, n), dtype=bool)
        for u, v in self.graph.edges():
            W[idx[u], idx[v]] = True
        for k in range(n):
            W |= np.outer(W[:, k], W[k, :])
        return W

    def floyd_warshall_tropical(self) -> NDArray[np.float64]:
        r"""
        Distancias geodésicas en el semianillo tropical \((\mathbb{R}\cup\{\infty\},\min,+)\).

            \(D_{ij}\leftarrow\min(D_{ij},\,D_{ik}+D_{kj})\).
        """
        nodes = self.nodes_in_order()
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        inf = np.float64(1.0e18)
        D = np.full((n, n), inf, dtype=np.float64)
        np.fill_diagonal(D, 0.0)
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            D[idx[u], idx[v]] = min(D[idx[u], idx[v]], w)
        for k in range(n):
            D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
        return D

    def transitive_reduction(self) -> NDArray[np.bool_]:
        r"""
        Diagrama de Hasse: \(A_{\mathrm{red}}=A\land\lnot(A\cdot A^+)\),
        con \(A^+\) la clausura transitiva estricta (sin diagonal).
        """
        A, _ = self.adjacency_matrix()
        A_bool = A > EPSILON
        W = self.warshall_transitive_closure()
        n = W.shape[0]
        W_strict = W.copy()
        np.fill_diagonal(W_strict, False)
        # caminos compuestos de longitud ≥ 2
        composed = A_bool @ W_strict
        reduced = A_bool & ~composed
        np.fill_diagonal(reduced, False)
        return reduced

    def poset_height(self) -> int:
        r"""Altura = longitud máxima de cadena = \(\max_i(\mathrm{nivel}_i)\)."""
        try:
            levels = nx.dag_longest_path_length(self.graph)
            return int(levels)
        except (nx.NetworkXError, ValueError):
            return 0

    def zeta_of_reachability(self) -> NDArray[np.float64]:
        r"""Zeta del poset de alcanzabilidad: \(\zeta=I\lor W\)."""
        W = self.warshall_transitive_closure().astype(np.float64)
        return np.eye(W.shape[0], dtype=np.float64) + W

    def mobius_of_reachability(self) -> NDArray[np.float64]:
        r"""Möbius \(\mu=\zeta^{-1}\) del poset de alcanzabilidad (inversión)."""
        Z = self.zeta_of_reachability()
        try:
            return np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            return la.pinv(Z)

    def has_directed_cycle(self) -> bool:
        W = self.warshall_transitive_closure()
        return bool(np.any(np.diag(W)))

    def betti_undirected(self) -> Tuple[int, int]:
        r"""
        Números de Betti del grafo subyacente no dirigido:
            \(\beta_0=c\) (componentes), \(\beta_1=|E|-|V|+c\) (ciclos independientes).
        Un DAG puede tener \(\beta_1>0\) (diamantes); \(\beta_1^{\mathrm{dir}}=0\).
        """
        UG = self.graph.to_undirected()
        n = UG.number_of_nodes()
        m = UG.number_of_edges()
        c = nx.number_connected_components(UG) if n else 0
        beta0 = int(c)
        beta1 = int(m - n + c) if n else 0
        return beta0, max(beta1, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {"source": u, "target": v,
                 "data_keys": d.get("data_keys", []),
                 "optional": d.get("optional", False)}
                for u, v, d in self.graph.edges(data=True)
            ],
            "is_acyclic": nx.is_directed_acyclic_graph(self.graph),
            "height": self.poset_height(),
            "betti": {"beta0": self.betti_undirected()[0],
                      "beta1": self.betti_undirected()[1]},
        }


# ── 2.3 MEMOIZACIÓN (TTL acoplado al periodo de Poincaré) ────────────────────

class OperatorMemoizer:
    r"""
    Caché LRU \((\mathrm{input\_hash},\,\mathrm{operator\_id},\,\mathrm{stratum})
    \mapsto(\mathrm{output},\,\mathrm{signature})\).

    Si se conoce el periodo de Poincaré \(T_P\) de la sección \(\Sigma\),
    la frescura se acota por \(T_P\) (un retorno = una generación de caché).
    """

    def __init__(self, max_size: int = MEMOIZATION_CACHE_SIZE):
        self.max_size = max_size
        self.cache:        Dict[MemoizationKey, MemoizationEntry] = {}
        self.access_order: List[MemoizationKey] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._poincare_period: Optional[float] = None

    def set_poincare_period(self, period: Optional[float]) -> None:
        self._poincare_period = period if period is not None and period > 0 else None

    def compute_input_hash(self, state: StateVector) -> str:
        return hashlib.sha256(
            json.dumps(state.to_dict(), sort_keys=True, default=str)
                .encode("utf-8")).hexdigest()

    def _max_age_hours(self) -> float:
        if self._poincare_period is None:
            return 24.0
        # Un periodo de Poincaré (en índices de paso) ~ minutos operativos.
        return float(max(1.0 / 60.0, min(24.0, self._poincare_period / 60.0)))

    def lookup(self, state: StateVector, operator_id: str, stratum: str
               ) -> Optional[Tuple[Dict[str, Any], TensorSignature]]:
        input_hash = self.compute_input_hash(state)
        key = MemoizationKey(input_hash, operator_id, stratum)
        entry = self.cache.get(key)
        max_age = int(math.ceil(self._max_age_hours()))
        if entry is None or not entry.is_fresh(max_age_hours=max_age):
            self.stats["misses"] += 1
            return None
        entry.hit_count += 1
        self.stats["hits"] += 1
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        return (entry.output, entry.output_signature)

    def store(self, state: StateVector, operator_id: str, stratum: str,
              output: Dict[str, Any], output_signature: TensorSignature) -> None:
        input_hash = self.compute_input_hash(state)
        key = MemoizationKey(input_hash, operator_id, stratum)
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            self.stats["evictions"] += 1
        self.cache[key] = MemoizationEntry(
            key=key, output=output, output_signature=output_signature,
            created_at=datetime.datetime.now(datetime.timezone.utc))
        self.access_order.append(key)

    def clear(self) -> None:
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total else 0.0
        return {**self.stats, "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "cache_size": len(self.cache), "max_size": self.max_size,
                "poincare_period": self._poincare_period}


# ── 2.4 AUDITORÍA HOMOLÓGICA DE ORIENTACIÓN ──────────────────────────────────

class HomologicalAuditor:
    r"""
    Exactitud de Mayer–Vietoris a nivel de fusión de datos y de DAG:

        \(\cdots\to H_n(A\cap B)\to H_n(A)\oplus H_n(B)\to H_n(A\cup B)
          \xrightarrow{\partial} H_{n-1}(A\cap B)\to\cdots\)

    Invariante operativo: \(\Delta\beta_1\equiv 0\) (no emergen ciclos
    bajo la unión).  En el DAG, \(\beta_1^{\mathrm{dir}}=0\) es aciclicidad.
    """

    def __init__(self, telemetry: TelemetryContext):
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)

    def audit_merge(
        self,
        df_a: Optional[pd.DataFrame],
        df_b: Optional[pd.DataFrame],
        df_result: pd.DataFrame,
        context: StateVector,
    ) -> Dict[str, Any]:
        audit_result: Dict[str, Any] = {
            "passed": True, "emergent_cycles": 0, "lost_connections": 0,
            "data_integrity": True, "warnings": [],
            "betti_delta": 0,
        }
        if df_a is None or df_b is None:
            self.logger.info("Homological audit skipped (missing input)")
            return audit_result
        try:
            card_a, card_b = len(df_a), len(df_b)
            card_result = len(df_result)
            min_expected = max(card_a, card_b)
            max_expected = card_a + card_b
            if card_result < min_expected * 0.8:
                audit_result["warnings"].append(
                    f"Data loss: expected ≥ {min_expected}, got {card_result}")
                audit_result["passed"] = False
            if card_result > max_expected * 1.2:
                audit_result["warnings"].append(
                    f"Duplicates: expected ≤ {max_expected}, got {card_result}")
                audit_result["emergent_cycles"] = int(
                    card_result - max_expected)
            cols_a, cols_b = set(df_a.columns), set(df_b.columns)
            cols_result = set(df_result.columns)
            expected_cols = cols_a | cols_b
            if not expected_cols.issubset(cols_result):
                audit_result["warnings"].append(
                    f"Lost columns: {expected_cols - cols_result}")
                audit_result["passed"] = False
                audit_result["lost_connections"] = len(expected_cols - cols_result)
            self._analyze_connectivity(df_a, df_b, df_result, audit_result)
        except Exception as e:
            self.logger.warning(f"Homological audit exception: {e}")
            audit_result["warnings"].append(str(e))
        if not audit_result["passed"]:
            self.telemetry.record_error(
                "homological_audit", "; ".join(audit_result["warnings"]))
        return audit_result

    def _analyze_connectivity(self, df_a, df_b, df_result, audit_result):
        try:
            id_col = next((c for c in ["id", "ID", "Id", "idx"]
                           if c in df_a.columns and c in df_b.columns), None)
            if id_col is None:
                return
            ids_a = set(df_a[id_col].dropna().unique())
            ids_b = set(df_b[id_col].dropna().unique())
            ids_result = set(df_result[id_col].dropna().unique())
            union = ids_a | ids_b
            inter = ids_a & ids_b
            # Mayer–Vietoris cardinal: |A∪B| = |A|+|B|−|A∩B|
            mv_expected = len(ids_a) + len(ids_b) - len(inter)
            if union and abs(len(ids_result) - mv_expected) > 0.2 * max(mv_expected, 1):
                audit_result["betti_delta"] = int(len(ids_result) - mv_expected)
                audit_result["warnings"].append(
                    f"Mayer–Vietoris card Δ={audit_result['betti_delta']}")
            if len(union) > 0:
                preservation = len(ids_result & union) / len(union)
                if preservation < 0.8:
                    audit_result["warnings"].append(
                        f"ID preservation {preservation:.1%} < 80%")
                    audit_result["passed"] = False
        except Exception as e:
            self.logger.debug(f"Connectivity analysis failed: {e}")

    def audit_dag(self, dag: AlgebraicDAG) -> Dict[str, Any]:
        r"""Orientación homológica del DAG: \(\beta_1^{\mathrm{dir}}=0\), \(W_{ii}=0\)."""
        beta0, beta1 = dag.betti_undirected()
        has_cycle = dag.has_directed_cycle()
        report = {
            "passed": not has_cycle,
            "beta0": beta0,
            "beta1_undirected": beta1,
            "directed_cycle": has_cycle,
            "height": dag.poset_height(),
        }
        if has_cycle:
            self.telemetry.record_error(
                "homological_dag", "Directed cycle in algebraic DAG")
        return report


# ── 2.5 KERNEL DE AUTOMEJORA RECURSIVA N3 ────────────────────────────────────

class RecursiveSelfImprovementKernel:
    r"""
    Kernel N3 de automejora recursiva — implementación de ``Phase2OrientKernel``.

    Niveles (fibración \(\Theta\times\mathfrak{n}_n\times A\)):

      N1  \(\theta_{t+1}=\theta_t-\alpha[0]\,\nabla_\theta L\)          [fibra]
      N2  \(O_{t+1}=\Pi_{\mathfrak{n}_n}(O_t-\alpha[1]\,\nabla_O L)\)  [base DAG]
      N3  \(\alpha_{t+1}=\mathrm{clip}(\alpha_t-\alpha[2]\,\nabla_\alpha\Phi)\)
          con \(\Phi(\alpha)=L\bigl(I(\alpha)(\theta,O),\,\alpha\bigr)\)
          (pérdida *unrolled* un paso: meta-gradiente tipo MAML).     [meta-regla]

    Ecuación funcional exigida:

        \(I_3\circ I_3=I_3\)     (idempotencia asintótica)
        \(\partial I_3/\partial\alpha\neq 0\)   (no-trivialidad)

    Acoplamiento celeste (paquete de Fase 1):

      · Tangencia homoclínica \(\Rightarrow\) control de Pyragas sobre \(O\)
        (\(O\leftarrow O-\eta(O-O_{\mathrm{ref}})\), feedback retardado).
      · Quiebre KAM \(\Rightarrow\) ganancia \(\alpha[0]\) reducida (adiabático).
      · \(T_{\mathrm{Nek}}\) grande \(\Rightarrow\) pasos más largos (estabilidad).
      · El flujo N3 admite su propia sección de Poincaré (puntos de viraje de \(L\)).
    """

    def __init__(
        self,
        n_params: int = _N3_FEATURE_DIM,
        adjacency_shape: int = 8,
        alpha_init: Optional[NDArray[np.float64]] = None,
        learning_rate: float = 1e-2,
        weights: Optional[Tuple[float, ...]] = None,
        dag: Optional[AlgebraicDAG] = None,
    ) -> None:
        rng = np.random.default_rng(seed=0xBEEF)
        self._n_params = int(n_params)
        self._adj_shape = int(adjacency_shape)

        theta = rng.uniform(0.1, 1.0, size=self._n_params)
        O = rng.uniform(0.0, 0.05, size=(self._adj_shape, self._adj_shape))
        O = np.triu(O, k=1)

        if alpha_init is None:
            alpha_init = np.array([1e-2, 1e-3, 1e-4], dtype=np.float64)

        self.state = ImprovementState(
            theta=np.asarray(theta, dtype=np.float64),
            O=np.asarray(O, dtype=np.float64),
            alpha=np.asarray(alpha_init, dtype=np.float64).copy())

        self.lr = float(learning_rate)
        # (ν, λ₂, Arnold, Melnikov, KAM, Lyapunov, ρ(O), Tikhonov α)
        self.weights: Tuple[float, ...] = weights or (
            1.0, 1.0, 0.5, 0.5, 0.75, 0.5, 0.25, 0.15)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._packet: Optional[ObserverPacket] = None
        self._theta_seed: Optional[NDArray[np.float64]] = None
        self._O_ref: NDArray[np.float64] = O.copy()
        self._dag: Optional[AlgebraicDAG] = dag
        self._dikw = DIKWPoset()
        self._pyraga_eta: float = 0.0
        self._kam_gain: float = 1.0
        self._nek_scale: float = 1.0
        self._n3_sections: List[N3PoincareSection] = []

    # ═══════════════════════════════════════════════════════════════════
    # 2.5.0  ingest_observer_packet  ←  PRIMER MÉTODO DE FASE 2
    #         continuación literal de Phase2OrientKernel (Fase 1, §1.7)
    # ═══════════════════════════════════════════════════════════════════
    def ingest_observer_packet(self, packet: ObserverPacket) -> None:
        r"""
        ← PRIMER MÉTODO DE FASE 2.  FUNCTOR DE FRONTERA  Orient ⟵ Obs.

        Traduce el ``ObserverPacket`` (emisión de ``project_to_phase_2``) en
        condiciones iniciales concretas del fibrado N3:

          \(\theta\leftarrow\theta_{\mathrm{seed}}/ \|\theta_{\mathrm{seed}}\|\)
            con \(\theta_{\mathrm{seed}}=(\lambda_2,\nu,\varepsilon_{\mathrm{KAM}},
            |M|_\infty,D_{\mathrm{Arnold}},\lambda_{\mathrm{Lyap}})\).

          \(O\leftarrow\Pi_{\mathfrak{n}_n}(A_{\mathrm{path}}+\delta A_{\mathrm{Arnold}}
            +\delta A_{\mathrm{hyp}})\).

          \(\alpha\leftarrow\alpha(\mathrm{KAM},\,M,\,T_{\mathrm{Nek}},\,\lambda_{\mathrm{Lyap}})\).

        Efectos colaterales de mecánica celeste:

          · si \(M\) detecta tangencia, se arma el feedback de Pyragas;
          · si el toro KAM colapsa, se reduce \(\alpha[0]\) (N1 cauteloso);
          · \(T_{\mathrm{Nek}}\) reescala el learning-rate efectivo.
        """
        if not isinstance(packet, ObserverPacket):
            raise RecursiveSelfImprovementError(
                f"ingest_observer_packet espera ObserverPacket, recibió {type(packet)!r}")
        self._packet = packet

        theta_seed = self._seed_theta(packet)
        self._theta_seed = theta_seed.copy()
        self.state.theta = theta_seed

        O_seed = self._seed_adjacency(packet)
        self.state.O = O_seed
        self._O_ref = O_seed.copy()

        alpha_seed = self._seed_alpha(packet)
        self.state.alpha = alpha_seed

        self._pyraga_eta = self._chaos_control_pyraga_gain(packet)
        self._kam_gain = self._kam_restoration_gain(packet)
        self._nek_scale = self._nekhoroshev_step_scale(packet)

        self.state.loss_history.clear()
        self.state.level_history.clear()
        self.state.theta_history.clear()
        self.state.alpha_history.clear()
        self.state.O_norm_history.clear()
        self._n3_sections.clear()

        self.logger.info(
            "[N3::ingest] θ=%s | α=%s | η_Pyragas=%.3e | KAM_gain=%.3f | "
            "Nek_scale=%.3f | tangle=%s | KAM=%s",
            np.round(self.state.theta, 4).tolist(),
            np.round(self.state.alpha, 6).tolist(),
            self._pyraga_eta, self._kam_gain, self._nek_scale,
            packet.melnikov.tangle_detected, packet.kam.is_KAM_stable,
        )

    # ── 2.5.1 Semillas granulares (descomposición de ingest) ────────────
    def _feature_vector_from_packet(
        self, packet: ObserverPacket,
    ) -> NDArray[np.float64]:
        vec = packet.to_n3_seed_vector()
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        if vec.size < self._n_params:
            vec = np.pad(vec, (0, self._n_params - vec.size))
        return vec[: self._n_params]

    @staticmethod
    def _normalize_feature(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        nrm = float(np.linalg.norm(vec))
        if nrm < EPSILON:
            return np.full_like(vec, 1.0 / max(vec.size, 1))
        return vec / nrm

    def _seed_theta(self, packet: ObserverPacket) -> NDArray[np.float64]:
        r"""
        \(\theta_{\mathrm{seed}}\) normalizado. Coordenadas:

            0: \(\lambda_2\)   (Fiedler — queremos ↑)
            1: \(\nu\)         (nilpotencia — queremos ↓ hacia 1+)
            2: \(\varepsilon_{\mathrm{KAM}}\)
            3: \(|M|_\infty\)
            4: \(D_{\mathrm{Arnold}}\)
            5: \(\lambda_{\mathrm{Lyap}}\)
        """
        raw = self._feature_vector_from_packet(packet)
        # Estabilizar signos: métricas "malas" se mantienen positivas.
        raw = np.abs(raw)
        return self._normalize_feature(raw)

    def _seed_adjacency(self, packet: ObserverPacket) -> NDArray[np.float64]:
        r"""
        \(O\) inicial: camino hamiltoniano del DAG canónico (superdiagonal)
        perturbado por fuga de Arnold por estrato y por hiperbolicidad de \(P\).
        Proyección \(\Pi_{\mathfrak{n}_n}\) por construcción.
        """
        n = self._adj_shape
        O = np.zeros((n, n), dtype=np.float64)
        steps = list(PipelineSteps)
        leakage = packet.arnold.per_stratum_leakage
        hyp = 1.0 if packet.return_map.is_hyperbolic else 0.0
        periodic = 1.0 if packet.return_map.is_periodic_orbit else 0.0

        for i in range(n - 1):
            stratum_name = (
                steps[i].stratum.name if i < len(steps) else "PHYSICS")
            leak = float(leakage.get(stratum_name, 0.0))
            # Camino base + fuga + mezcla hiperbólica − concentración periódica.
            O[i, i + 1] = 0.05 + 0.10 * leak + 0.02 * hyp - 0.01 * periodic
            if i + 2 < n and hyp > 0.0:
                O[i, i + 2] = 0.01 * hyp  # atajo: mezcla no-local

        O = np.clip(np.triu(O, k=1), 0.0, 1.0)
        return O

    def _seed_alpha(self, packet: ObserverPacket) -> NDArray[np.float64]:
        r"""
        Meta-ganancias \((\alpha_1,\alpha_2,\alpha_3)\) moduladas por KAM / Melnikov /
        Lyapunov / Nekhoroshev.

          · KAM estable \(\Rightarrow\) N1 más agresivo.
          · Caos homoclínico \(\Rightarrow\) N2 (topología) más agresivo, N3 cauto
            pero no nulo (explorar la regla).
          · \(|\lambda_{\mathrm{Lyap}}|\) grande \(\Rightarrow\) amortiguar N1.
        """
        chaos = float(min(1.0, packet.melnikov.max_magnitude * 1.0e3))
        kam_metric = 1.0 if packet.kam.is_KAM_stable else 0.1
        lyap = abs(float(packet.return_map.stability_exponent))
        lyap_damp = 1.0 / (1.0 + lyap)

        a1 = 1.0e-2 * kam_metric * lyap_damp
        a2 = 1.0e-3 * (1.0 + chaos)
        a3 = 1.0e-4 * (1.0 + 2.0 * chaos) * kam_metric
        alpha = np.array([a1, a2, a3], dtype=np.float64)
        return np.clip(alpha, _N3_ALPHA_FLOOR, _N3_ALPHA_CEIL)

    def _chaos_control_pyraga_gain(self, packet: ObserverPacket) -> float:
        r"""
        Ganancia de Pyragas \(\eta\) para \(u(t)=-\eta\bigl(O(t)-O(t-T_P)\bigr)\).

        Se activa ssi hay tangencia homoclínica (ceros simples de Melnikov).
        \(\eta\propto\tanh(|M|_\infty/\texttt{_MELNIKOV_TOL})\).
        """
        if not packet.melnikov.tangle_detected:
            return 0.0
        mag = float(packet.melnikov.max_magnitude)
        return float(0.25 * math.tanh(mag / max(_MELNIKOV_TOL, EPSILON)))

    def _kam_restoration_gain(self, packet: ObserverPacket) -> float:
        """Factor \(\in(0,1]\) que deprime N1 si el toro no persiste."""
        if packet.kam.is_KAM_stable:
            return 1.0
        # Greene residue alto → más cautela.
        residue = max(float(packet.kam.greene_residue), EPSILON)
        return float(np.clip(1.0 / (1.0 + residue), 0.05, 1.0))

    def _nekhoroshev_step_scale(self, packet: ObserverPacket) -> float:
        r"""
        Reescala global del paso: \(T_{\mathrm{Nek}}\) grande \(\Rightarrow\)
        el sistema es efectivamente integrable, se permiten pasos mayores.
        """
        t_nek = float(getattr(packet.arnold, "nekhoroshev_time", 1.0) or 1.0)
        t_nek = max(t_nek, 1.0)
        # log10(T) / 8  recortado a [0.25, 2]
        return float(np.clip(math.log10(t_nek) / 8.0, 0.25, 2.0))

    def bind_algebraic_dag(self, dag: AlgebraicDAG) -> None:
        """Inyecta el DAG algebraico (N2 puede proyectar sobre su soporte)."""
        self._dag = dag
        A, _ = dag.adjacency_matrix()
        if A.shape[0] == self._adj_shape:
            self._O_ref = np.triu(A, k=1)

    # ── 2.5.2 Invariantes estructurales de \(O\) ────────────────────────
    @staticmethod
    def _nilpotency_index(O: NDArray[np.float64], tol: float = _WILKINSON_LIMIT) -> int:
        n = int(O.shape[0])
        P = np.eye(n, dtype=np.float64)
        for k in range(1, n + 1):
            P = P @ O
            if float(la.norm(P, "fro")) < tol:
                return k
        return n + 1

    @staticmethod
    def _fiedler_of(O: NDArray[np.float64]) -> Tuple[float, float]:
        """Retorna \((\lambda_2,\,\lambda_{\max})\) de \(L_{\mathrm{sym}}=D-|O|\)."""
        absO = np.abs(O)
        deg = np.sum(absO, axis=1)
        L = np.diag(deg) - absO
        L_sym = 0.5 * (L + L.T)
        try:
            evals = np.sort(np.real(la.eigvalsh(L_sym)))
        except la.LinAlgError:
            return 0.0, 1.0
        lam2 = float(evals[1]) if evals.size > 1 else 0.0
        lam_max = float(evals[-1]) if evals.size else 1.0
        return lam2, max(lam_max, EPSILON)

    @staticmethod
    def _spectral_radius_of(O: NDArray[np.float64]) -> float:
        try:
            eigs = la.eigvals(O)
            return float(np.max(np.abs(eigs)))
        except la.LinAlgError:
            return float(la.norm(O, 2))

    # ── 2.5.3 Pérdida compuesta (granular) ──────────────────────────────
    def _loss_structural(self, O: NDArray[np.float64]) -> float:
        w_nu, w_fiedler = self.weights[0], self.weights[1]
        w_rho = self.weights[6] if len(self.weights) > 6 else 0.25
        n = max(int(O.shape[0]), 1)
        nu = self._nilpotency_index(O)
        nu_term = 1.0 - (nu / n)          # 0 si nilpotente perfecto
        # Penalizar ν = n+1 (NO nilpotente = ciclo).
        if nu > n:
            nu_term += 5.0
        lam2, lam_max = self._fiedler_of(O)
        fiedler_term = -(lam2 / lam_max)  # queremos λ₂ grande
        rho = self._spectral_radius_of(O)
        rho_term = max(0.0, rho - 1.0)    # contracción: ρ(O)<1
        sparsity = float(np.sum(np.abs(O))) / float(n * n)
        return (w_nu * nu_term + w_fiedler * fiedler_term
                + w_rho * rho_term + 0.05 * sparsity)

    def _loss_parametric(self, theta: NDArray[np.float64]) -> float:
        r"""
        Empuja \(\theta\) hacia un *target mejorado* derivado de la semilla:
        ↑ Fiedler, ↓ (ν, ε, |M|, D, |λ|).  Distancia cuadrática + barrera.
        """
        if self._theta_seed is None:
            return float(np.linalg.norm(theta) ** 2) * 0.01
        seed = self._theta_seed
        target = seed.copy()
        # coordenadas: 0 Fiedler ↑ ; 1..5 métricas de daño ↓
        if target.size >= 1:
            target[0] = min(1.0, seed[0] * 1.15 + 0.05)
        for i in range(1, target.size):
            target[i] = max(0.0, seed[i] * 0.85)
        target = self._normalize_feature(np.abs(target))
        return float(np.sum((theta - target) ** 2))

    def _loss_meta(self, alpha: NDArray[np.float64]) -> float:
        w_tik = self.weights[7] if len(self.weights) > 7 else 0.15
        # Barrera logarítmica cerca de los bordes del intervalo [floor, ceil].
        barrier = 0.0
        for a in alpha:
            barrier += -math.log(max(a - _N3_ALPHA_FLOOR, EPSILON))
            barrier += -math.log(max(_N3_ALPHA_CEIL - a, EPSILON))
        return w_tik * float(np.sum(alpha ** 2)) + 1.0e-4 * barrier

    def _loss_celestial(
        self,
        theta: NDArray[np.float64],
        O: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> float:
        r"""
        Términos condicionados por el ObserverPacket (constantes de contexto):

          \(w_M\cdot 1_{\mathrm{tangle}}\cdot\|O-O_{\mathrm{ref}}\|_F^{-1}\eta\)
          \(w_K\cdot 1_{\lnot\mathrm{KAM}}\cdot\|\alpha\|_2\)
          \(w_A\cdot D_{\mathrm{Arnold}}\cdot\|O\|_F\)
          \(w_\lambda\cdot|\lambda_{\mathrm{Lyap}}|\cdot|\alpha[0]|\)
        """
        P = self._packet
        if P is None:
            return 0.0
        w_A = self.weights[2]
        w_M = self.weights[3]
        w_K = self.weights[4] if len(self.weights) > 4 else 0.75
        w_L = self.weights[5] if len(self.weights) > 5 else 0.5

        arnold_term = w_A * float(P.arnold.total_diffusion) * float(la.norm(O, "fro"))
        melnikov_term = 0.0
        if P.melnikov.tangle_detected:
            # Pyragas: penalizar alejarse de O_ref *y* recompensar amortiguación.
            drift = float(la.norm(O - self._O_ref, "fro"))
            melnikov_term = w_M * float(P.melnikov.max_magnitude) * (1.0 + drift)
        kam_term = 0.0
        if not P.kam.is_KAM_stable:
            kam_term = w_K * float(np.linalg.norm(alpha)) * (
                1.0 + float(P.kam.greene_residue))
        lyap_term = w_L * abs(float(P.return_map.stability_exponent)) * abs(float(alpha[0]))
        H_term = 0.05 * abs(float(P.hamiltonian_drift)) * float(np.linalg.norm(theta))
        return arnold_term + melnikov_term + kam_term + lyap_term + H_term

    def _loss(
        self,
        theta: NDArray[np.float64],
        O: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> float:
        return (
            self._loss_structural(O)
            + self._loss_parametric(theta)
            + self._loss_meta(alpha)
            + self._loss_celestial(theta, O, alpha)
        )

    # ── 2.5.4 Gradientes y búsqueda lineal ──────────────────────────────
    @staticmethod
    def _grad_central(
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        h: float = 1e-6,
        richardson: bool = True,
    ) -> NDArray[np.float64]:
        r"""
        Diferencias centrales.  Con Richardson (\(h\) y \(h/2\)):

            \(g=(4g_{h/2}-g_h)/3+\mathcal{O}(h^4)\).
        """
        x = np.asarray(x, dtype=np.float64)

        def _once(hh: float) -> NDArray[np.float64]:
            g = np.zeros_like(x, dtype=np.float64)
            it = np.nditer(x, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                xp = x.copy(); xm = x.copy()
                xp[idx] += hh
                xm[idx] -= hh
                g[idx] = (f(xp) - f(xm)) / (2.0 * hh)
                it.iternext()
            return g

        g_h = _once(h)
        if not richardson:
            return g_h
        g_h2 = _once(0.5 * h)
        return (4.0 * g_h2 - g_h) / 3.0

    @staticmethod
    def _armijo(
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        alpha0: float,
        c: float = 1.0e-4,
        beta: float = 0.5,
        max_backtrack: int = 12,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""Búsqueda de Armijo–Goldstein: \(f(x+\alpha d)\le f(x)+c\alpha\langle\nabla f,d\rangle\)."""
        f0 = f(x)
        slope = float(np.sum((-direction) * direction))  # d = −g  ⇒ ⟨∇f,d⟩=−||g||²
        # direction se pasa ya como −g, slope = ⟨∇f, d⟩ ≈ −||g||²
        g_norm2 = float(np.sum(direction * direction))
        slope = -g_norm2
        a = float(alpha0)
        for _ in range(max_backtrack):
            x_new = x + a * direction
            try:
                f_new = f(x_new)
            except Exception:
                a *= beta
                continue
            if f_new <= f0 + c * a * slope:
                return x_new, a
            a *= beta
        return x + a * direction, a

    # ── 2.5.5 Nivel 1: optimización de θ (fibra) ─────────────────────────
    def step_level_1(
        self, max_iter: int = 10, use_shahshahani: bool = True,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        N1: \(\theta_{t+1}=\theta_t-\alpha[0]\,\kappa_{\mathrm{KAM}}\,\kappa_{\mathrm{Nek}}\,\nabla_\theta L\).

        Si \(\theta>0\) y ``use_shahshahani``, el descenso es el flujo replicador
        (métrica de Shahshahani \(\langle u,v\rangle_\theta=\sum u_iv_i/\theta_i\)),
        coherente con el flujo Brockett–Shahshahani de la Fase 1.
        """
        theta = self.state.theta.copy()
        scale = float(self.state.alpha[0]) * self._kam_gain * self._nek_scale

        def f_th(th: NDArray[np.float64]) -> float:
            return self._loss(th, self.state.O, self.state.alpha)

        loss_prev = f_th(theta)
        for _ in range(max_iter):
            grad = self._grad_central(f_th, theta)
            if use_shahshahani and np.all(theta > 0):
                # gradiente natural: θ ⊙ ∇L  (identificación Shahshahani)
                nat = theta * grad
                direction = -nat
            else:
                direction = -grad
            theta_new, _ = self._armijo(f_th, theta, direction, scale)
            # mantener θ en el simplex no-negativo (rasgos como ganancias)
            theta_new = np.clip(theta_new, 0.0, None)
            nrm = float(np.linalg.norm(theta_new))
            if nrm > EPSILON:
                theta_new = theta_new / nrm
            loss_new = f_th(theta_new)
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                theta = theta_new
                loss_prev = loss_new
                break
            theta, loss_prev = theta_new, loss_new

        self.state.theta = theta
        self.state.loss_history.append(loss_prev)
        self.state.level_history.append(int(ImprovementLevel.N1_PARAMETRIC))
        self.state.theta_history.append(theta.copy())
        return theta, loss_prev

    # ── 2.5.6 Nivel 2: optimización de O (topología DAG) ────────────────
    @staticmethod
    def _project_strictly_upper(X: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""\(\Pi_{\mathfrak{n}_n}:M_n\to\mathfrak{n}_n\), \(X\mapsto\mathrm{triu}(X,1)\)."""
        return np.clip(np.triu(X, k=1), 0.0, 1.0)

    def _project_spectral_radius(
        self, O: NDArray[np.float64], rho_max: float = 0.99,
    ) -> NDArray[np.float64]:
        r"""Homotecia si \(\rho(O)>\rho_{\max}\): \(O\leftarrow(\rho_{\max}/\rho)O\)."""
        rho = self._spectral_radius_of(O)
        if rho > rho_max and rho > EPSILON:
            O = O * (rho_max / rho)
        return self._project_strictly_upper(O)

    def _apply_pyraga(self, O: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Feedback retardado de Pyragas: \(O\leftarrow O-\eta(O-O_{\mathrm{ref}})\)."""
        if self._pyraga_eta <= 0.0:
            return O
        return O - self._pyraga_eta * (O - self._O_ref)

    def step_level_2(
        self, max_iter: int = 5, sparsity_penalty: float = 0.05,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        N2: \(O_{t+1}=\Pi_{\rho}\circ\Pi_{\mathfrak{n}_n}\bigl(
             O_t-\alpha[1]\,\nabla_O(L+\lambda\|O\|_1)\bigr)\),
        seguido de Pyragas si hay tangencia homoclínica.
        """
        O = self.state.O.copy()
        scale = float(self.state.alpha[1]) * self._nek_scale

        def f_O(o: NDArray[np.float64]) -> float:
            return (self._loss(self.state.theta, o, self.state.alpha)
                    + sparsity_penalty * float(np.sum(np.abs(o))))

        loss_prev = f_O(O)
        for _ in range(max_iter):
            grad = self._grad_central(f_O, O)
            O_new = O - scale * grad
            O_new = self._apply_pyraga(O_new)
            O_new = self._project_spectral_radius(
                self._project_strictly_upper(O_new))
            loss_new = f_O(O_new)
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                O = O_new
                loss_prev = loss_new
                break
            O, loss_prev = O_new, loss_new

        self.state.O = O
        self.state.loss_history.append(loss_prev)
        self.state.level_history.append(int(ImprovementLevel.N2_TOPOLOGIC))
        self.state.O_norm_history.append(float(la.norm(O, "fro")))
        return O, loss_prev

    # ── 2.5.7 Nivel 3: meta-regla (unroll MAML = verdadero ∇_α Φ) ───────
    def _maml_unroll(
        self, alpha: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Un paso de \(I(\alpha)\): \((\theta,O)\mapsto(\theta',O')\) con el \(\alpha\) dado.

        \(\theta'=\Pi_{\Delta}(\theta-\alpha[0]\nabla_\theta L)\),
        \(O'=\Pi_{\mathfrak{n}_n}(O-\alpha[1]\nabla_O L)\).

        \(\Phi(\alpha)=L(\theta',O',\alpha)\) es la meta-pérdida (Danskin / MAML).
        """
        theta = self.state.theta
        O = self.state.O

        def f_th(th: NDArray[np.float64]) -> float:
            return self._loss(th, O, alpha)

        def f_O(o: NDArray[np.float64]) -> float:
            return self._loss(theta, o, alpha)

        g_th = self._grad_central(f_th, theta, richardson=False)
        g_O = self._grad_central(f_O, O, richardson=False)
        theta_p = theta - float(alpha[0]) * self._kam_gain * g_th
        theta_p = np.clip(theta_p, 0.0, None)
        nrm = float(np.linalg.norm(theta_p))
        if nrm > EPSILON:
            theta_p = theta_p / nrm
        O_p = self._project_strictly_upper(
            O - float(alpha[1]) * g_O)
        O_p = self._apply_pyraga(O_p)
        return theta_p, O_p

    def _meta_loss(self, alpha: NDArray[np.float64]) -> float:
        theta_p, O_p = self._maml_unroll(alpha)
        return self._loss(theta_p, O_p, alpha)

    def step_level_3(
        self, max_iter: int = 3,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        N3: \(\alpha_{t+1}=\mathrm{clip}\bigl(\alpha_t-\alpha[2]\,\nabla_\alpha\Phi(\alpha_t)\bigr)\).

        Semánticamente: la regla de actualización se actualiza a sí misma.
        El jacobiano \(\nabla_\alpha\Phi\) incluye (por diferencias) los términos
        de segundo orden \(\mathrm{D}_\alpha I(\alpha)\), que es lo que distingue
        N3 de un N2 disfrazado.
        """
        alpha = self.state.alpha.copy()
        loss_prev = self._meta_loss(alpha)
        for _ in range(max_iter):
            grad = self._grad_central(self._meta_loss, alpha, richardson=False)
            step = float(alpha[2]) * self._nek_scale
            direction = -grad
            alpha_new, _ = self._armijo(self._meta_loss, alpha, direction, step)
            alpha_new = np.clip(alpha_new, _N3_ALPHA_FLOOR, _N3_ALPHA_CEIL)
            loss_new = self._meta_loss(alpha_new)
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                alpha = alpha_new
                loss_prev = loss_new
                break
            alpha, loss_prev = alpha_new, loss_new

        self.state.alpha = alpha
        self.state.loss_history.append(loss_prev)
        self.state.level_history.append(int(ImprovementLevel.N3_META_META))
        self.state.alpha_history.append(alpha.copy())
        return alpha, loss_prev

    # ── 2.5.8 Certificados N3 (idempotencia, no-trivialidad, Lyapunov) ──
    def verify_idempotence(
        self, n_extra: int = 1,
    ) -> N3IdempotenceCertificate:
        r"""
        Ejecuta \(n_{\mathrm{extra}}\) ciclos N1→N2→N3 adicionales y mide
        \(\|z'-z\|\).  No muta el estado observable: restaura al finalizar.
        """
        snap_th = self.state.theta.copy()
        snap_O = self.state.O.copy()
        snap_a = self.state.alpha.copy()
        snap_loss = list(self.state.loss_history)
        snap_lvl = list(self.state.level_history)
        snap_th_h = list(self.state.theta_history)
        snap_a_h = list(self.state.alpha_history)
        snap_On = list(self.state.O_norm_history)

        for _ in range(max(1, n_extra)):
            self.step_level_1(max_iter=4)
            self.step_level_2(max_iter=3)
            self.step_level_3(max_iter=2)

        r_th = float(np.linalg.norm(self.state.theta - snap_th))
        r_O = float(la.norm(self.state.O - snap_O, "fro"))
        r_a = float(np.linalg.norm(self.state.alpha - snap_a))
        residual = math.sqrt(r_th * r_th + r_O * r_O + r_a * r_a)

        # Restaurar
        self.state.theta = snap_th
        self.state.O = snap_O
        self.state.alpha = snap_a
        self.state.loss_history = snap_loss
        self.state.level_history = snap_lvl
        self.state.theta_history = snap_th_h
        self.state.alpha_history = snap_a_h
        self.state.O_norm_history = snap_On

        return N3IdempotenceCertificate(
            residual=residual,
            residual_theta=r_th,
            residual_O=r_O,
            residual_alpha=r_a,
            is_idempotent=residual <= _N3_IDEMPOTENCE_TOL * (1.0 + residual),
            n_extra_cycles=n_extra,
        )

    def verify_nontriviality(
        self, h: float = 1.0e-4,
    ) -> N3NontrivialityCertificate:
        r"""
        Jacobiano \(\mathrm{D}_\alpha(\theta^*,O^*)\) por diferencias centrales
        de un unroll.  No-trivial ssi \(\|\mathrm{D}_\alpha I\|_F>\varepsilon\).
        """
        alpha0 = self.state.alpha.copy()
        d_th: List[NDArray[np.float64]] = []
        d_O: List[NDArray[np.float64]] = []
        for i in range(alpha0.size):
            ap = alpha0.copy(); am = alpha0.copy()
            ap[i] += h
            am[i] -= h
            th_p, O_p = self._maml_unroll(ap)
            th_m, O_m = self._maml_unroll(am)
            d_th.append((th_p - th_m) / (2.0 * h))
            d_O.append((O_p - O_m) / (2.0 * h))
        J_th = np.stack(d_th, axis=-1)  # (d, 3)
        J_O = np.stack(d_O, axis=-1)    # (n, n, 3)
        n_th = float(la.norm(J_th))
        n_O = float(la.norm(J_O))
        fro = math.sqrt(n_th * n_th + n_O * n_O)
        return N3NontrivialityCertificate(
            jacobian_frobenius=fro,
            dtheta_dalpha_norm=n_th,
            dO_dalpha_norm=n_O,
            is_nontrivial=fro > EPSILON,
        )

    def n3_lyapunov(self, kappa: float = 0.1) -> N3LyapunovCertificate:
        r"""
        \(V_t=L_t+\tfrac12\kappa\|\alpha_t-\alpha_{t-1}\|^2\) sobre el historial.
        """
        Lh = self.state.loss_history
        Ah = self.state.alpha_history
        if not Lh:
            return N3LyapunovCertificate(
                V_final=0.0, V_initial=0.0, is_monotone=True,
                mean_descent=0.0, n_violations=0)
        V: List[float] = []
        for i, Lt in enumerate(Lh):
            damp = 0.0
            if Ah and i > 0:
                j = min(i, len(Ah) - 1)
                j_prev = max(j - 1, 0)
                damp = 0.5 * kappa * float(
                    np.sum((Ah[j] - Ah[j_prev]) ** 2))
            V.append(float(Lt) + damp)
        diffs = [V[i + 1] - V[i] for i in range(len(V) - 1)]
        n_viol = sum(1 for d in diffs if d > _N3_IDEMPOTENCE_TOL)
        mean_desc = float(np.mean(diffs)) if diffs else 0.0
        return N3LyapunovCertificate(
            V_final=V[-1],
            V_initial=V[0],
            is_monotone=n_viol == 0,
            mean_descent=mean_desc,
            n_violations=n_viol,
        )

    def capture_n3_poincare_section(self) -> List[N3PoincareSection]:
        r"""
        Sección de Poincaré *del propio flujo N3*: puntos de viraje de \(L(t)\),
        i.e. ceros de \(\Delta L_t\cdot\Delta L_{t+1}<0\) (Poincaré–Bendixson 1-D).
        """
        Lh = self.state.loss_history
        Ah = self.state.alpha_history
        sections: List[N3PoincareSection] = []
        if len(Lh) < 3:
            self._n3_sections = sections
            return sections
        for i in range(1, len(Lh) - 1):
            v_prev = Lh[i] - Lh[i - 1]
            v_next = Lh[i + 1] - Lh[i]
            if v_prev * v_next < 0.0:
                turning = "max" if v_prev > 0.0 else "min"
                a_at = (tuple(float(x) for x in Ah[min(i, len(Ah) - 1)].tolist())
                        if Ah else tuple())
                sections.append(N3PoincareSection(
                    t_index=i, loss_value=float(Lh[i]),
                    alpha_at=a_at, turning=turning))
        self._n3_sections = sections
        return sections

    # ── 2.5.9 Bucle N3 completo ─────────────────────────────────────────
    def optimize_recursive_n3(
        self, n_cycles: int = 3,
    ) -> Dict[str, Any]:
        r"""
        Bucle anidado:

            for \(c=1\dots N\):
                N1: \(\theta^*\leftarrow\mathrm{argmin}\,L(\theta,O,\alpha)\)
                N2: \(O^*\leftarrow\mathrm{argmin}\,L(\theta^*,O,\alpha)\)
                N3: \(\alpha^*\leftarrow\mathrm{argmin}\,\Phi(\alpha)\)
        """
        if self._packet is None:
            raise RecursiveSelfImprovementError(
                "optimize_recursive_n3 requiere ingest_observer_packet previo")
        for c in range(n_cycles):
            self.step_level_1()
            self.step_level_2()
            self.step_level_3()
            self.logger.info(
                "[N3::cycle %d/%d] loss=%.6e  α=%s  ||O||_F=%.4e",
                c + 1, n_cycles,
                self.state.loss_history[-1],
                np.round(self.state.alpha, 6).tolist(),
                float(la.norm(self.state.O, "fro")))
        sections = self.capture_n3_poincare_section()
        return {
            "theta": self.state.theta.tolist(),
            "alpha": self.state.alpha.tolist(),
            "O": self.state.O.tolist(),
            "O_norm": float(la.norm(self.state.O, "fro")),
            "final_loss": (self.state.loss_history[-1]
                           if self.state.loss_history else 0.0),
            "loss_history": list(self.state.loss_history),
            "level_history": list(self.state.level_history),
            "n3_poincare_sections": [
                {"t": s.t_index, "L": s.loss_value, "turning": s.turning}
                for s in sections
            ],
        }

    def dag_from_O(self, weight_tol: float = 1.0e-3) -> AlgebraicDAG:
        """Materializa el DAG algebraico inducido por \(O^*\) (N2)."""
        labels = [s.value for s in PipelineSteps]
        if len(labels) < self._adj_shape:
            labels = labels + [f"aux_{i}" for i in range(len(labels), self._adj_shape)]
        labels = labels[: self._adj_shape]
        return AlgebraicDAG.from_adjacency(self.state.O, labels, weight_tol)

    # ═══════════════════════════════════════════════════════════════════
    # 2.5.10 project_to_phase_3  ←  CIERRE DE FASE 2 / INICIO DE FASE 3
    # ═══════════════════════════════════════════════════════════════════
    def project_to_phase_3(
        self,
        observer_packet: ObserverPacket,
        n_cycles: int = 3,
    ) -> "OrientationSeed":
        r"""
        ← ÚLTIMO MÉTODO DE FASE 2.  FUNCTOR DE FRONTERA  Orient ⟶ Decide.

        Cierra el kernel N3 (ciclos N1→N2→N3), certifica idempotencia y
        no-trivialidad, captura la sección de Poincaré del flujo N3 y emite
        un ``OrientationSeed`` inmutable.  La FASE 3 arranca formalmente al
        consumir este semilla mediante
        ``Phase3DecideDirector.ingest_orientation_seed``.

        Diagrama de continuación:

            ingest_observer_packet(packet)          ← frontera F1
                └──► optimize_recursive_n3
                        └──► verify_idempotence
                        └──► verify_nontriviality
                        └──► n3_lyapunov
            project_to_phase_3  ──OrientationSeed──►
                Phase3DecideDirector.ingest_orientation_seed
                    └──► reconfiguración de política
                    └──► DataProduct / veredicto Ω₃
        """
        if self._packet is None or observer_packet is not self._packet:
            self.ingest_observer_packet(observer_packet)

        report = self.optimize_recursive_n3(n_cycles=n_cycles)
        idem = self.verify_idempotence(n_extra=1)
        nontr = self.verify_nontriviality()
        lyap = self.n3_lyapunov()
        sections = tuple(self._n3_sections)

        if not idem.is_idempotent:
            self.logger.warning(
                "[N3::I₃∘I₃] residuo=%.3e ≰ tol=%.1e (no idempotente aún)",
                idem.residual, _N3_IDEMPOTENCE_TOL)
        if not nontr.is_nontrivial:
            raise RecursiveSelfImprovementError(
                "∂I₃/∂α ≈ 0: el nivel 3 es vacuamente N2 (trivialidad prohibida)")
        if not lyap.is_monotone:
            self.logger.warning(
                "[N3::Lyapunov] %d violaciones de descenso (V₀=%.3e → V∞=%.3e)",
                lyap.n_violations, lyap.V_initial, lyap.V_final)

        O_star = np.asarray(report["O"], dtype=np.float64)
        height = 0
        try:
            height = self.dag_from_O().poset_height()
        except Exception as exc:
            self.logger.debug("poset_height from O* failed: %s", exc)

        seed = OrientationSeed(
            theta_final=np.asarray(report["theta"], dtype=np.float64),
            alpha_final=np.asarray(report["alpha"], dtype=np.float64),
            O_final=O_star,
            O_norm=float(report["O_norm"]),
            final_loss=float(report["final_loss"]),
            observer_packet=observer_packet,
            kam_preserved=observer_packet.kam.is_KAM_stable,
            melnikov_chaotic=observer_packet.melnikov.tangle_detected,
            idempotence=idem,
            nontriviality=nontr,
            lyapunov=lyap,
            n3_sections=sections,
            poset_height=int(height),
            nekhoroshev_scale=float(self._nek_scale),
            pyraga_gain=float(self._pyraga_eta),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        self.logger.info(
            "[Fase 2::ORIENT] seed emitido | loss=%.6e | I₃∘I₃=%s | ∂I₃/∂α=%.3e | "
            "V monotone=%s | Σ_N3=%d",
            seed.final_loss, idem.is_idempotent, nontr.jacobian_frobenius,
            lyap.is_monotone, len(sections))
        return seed


@dataclass(frozen=True)
class OrientationSeed:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 2.  OBJETO DE FRONTERA FASE 2 → FASE 3.

    Transporta \((\theta^*,\alpha^*,O^*)\), los certificados N3 (idempotencia,
    no-trivialidad, Lyapunov), la sección de Poincaré del flujo N3 y las
    banderas celestes (KAM, Melnikov) hacia la FASE 3.
    """
    theta_final:      NDArray[np.float64]
    alpha_final:      NDArray[np.float64]
    O_final:          NDArray[np.float64]
    O_norm:           float
    final_loss:       float
    observer_packet:  ObserverPacket
    kam_preserved:    bool
    melnikov_chaotic: bool
    idempotence:      N3IdempotenceCertificate
    nontriviality:    N3NontrivialityCertificate
    lyapunov:         N3LyapunovCertificate
    n3_sections:      Tuple[N3PoincareSection, ...]
    poset_height:     int
    nekhoroshev_scale: float
    pyraga_gain:      float
    timestamp:        datetime.datetime

    def to_policy_dict(self) -> Dict[str, Any]:
        """Proyección serializable para ingestión de política en Fase 3."""
        return {
            "theta_final": self.theta_final.tolist(),
            "alpha_final": self.alpha_final.tolist(),
            "O_norm": float(self.O_norm),
            "final_loss": float(self.final_loss),
            "kam_preserved": bool(self.kam_preserved),
            "melnikov_chaotic": bool(self.melnikov_chaotic),
            "idempotent": bool(self.idempotence.is_idempotent),
            "idempotence_residual": float(self.idempotence.residual),
            "nontrivial": bool(self.nontriviality.is_nontrivial),
            "dI3_dalpha": float(self.nontriviality.jacobian_frobenius),
            "lyapunov_monotone": bool(self.lyapunov.is_monotone),
            "poset_height": int(self.poset_height),
            "nekhoroshev_scale": float(self.nekhoroshev_scale),
            "pyraga_gain": float(self.pyraga_gain),
            "n3_turning_points": len(self.n3_sections),
        }


# ── 2.6 CONTRATO DE CONTINUACIÓN  FASE 2 → FASE 3 ───────────────────────────

@runtime_checkable
class Phase3DecideDirector(Protocol):
    r"""
    ════════════════════════════════════════════════════════════════════════
    CONTRATO FORMAL DE FRONTERA  —  último objeto de la Fase 2
                                    ≡ primer objeto de la Fase 3
    ════════════════════════════════════════════════════════════════════════

    El funtor ``project_to_phase_3`` produce un ``OrientationSeed``.  Toda
    implementación de la Fase 3 (DECIDE) DEBE satisfacer este protocolo:

        ingest_orientation_seed(seed)  :  traduce (θ*, α*, O*, banderas KAM/
                                          Melnikov) en política operativa
                                          (umbrales, enforce_filtration,
                                          enforce_homology)
        execute_pipeline(...)          :  corre el DAG, Mayer–Vietoris, Ω₃

    Axioma de filtración que Fase 3 debe preservar:

        \(V_{\mathrm{PHYSICS}}\subset V_{\mathrm{TACTICS}}
          \subset V_{\mathrm{STRATEGY}}\subset V_{\mathrm{WISDOM}}\)

    La definición de ``PipelineDirector.ingest_orientation_seed`` — primer
    método concreto de Fase 3 — CONTINÚA EN LA FASE 3.
    """

    def ingest_orientation_seed(self, seed: OrientationSeed) -> None:
        """Traduce OrientationSeed → umbrales, filtración y homologías operativas."""
        ...


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FIN DE FASE 2 (ORIENT)                                                 █
# █   Continúa en FASE 3 (DECIDE & ACT):                                     █
# █     PipelineConfig · BaseProcessingStep · pasos concretos                █
# █     PipelineDirector.ingest_orientation_seed  ← 1.er método              █
# █     execute_pipeline / _run_nested_phases     ← veredicto Ω₃             █
# █     DataProduct terminal                                                 █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████
# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► DECIDE & ACT : MAYER–VIETORIS + VEREDICTO TERMINAL Ω₃       █
# █   (Pasos concretos  +  PipelineDirector  +  DataProduct / Ω₃)            █
# █                                                                          █
# █   Contrato de frontera Fase 2 → Fase 3:                                  █
# █     Phase3DecideDirector.ingest_orientation_seed  ← PRIMER MÉTODO        █
# █     PipelineDirector.emit_omega3_verdict          ← ÚLTIMO MÉTODO        █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 3.0 VEREDICTO Ω₃ Y CERTIFICADOS DE DECISIÓN ──────────────────────────────

class Omega3Status(str, enum.Enum):
    r"""
    Retículo de decisión del funtor terminal \(\Omega_3\).

    Orden (de más permisivo a absorbente):

        ACCEPT ≽ ACCEPT_WITH_RESIDUE ≽ RECONFIGURE
              ≽ ABORT_HOMOLOGY ≽ ABORT_KAM ≽ ABORT_CHAOS
              ≽ ABORT_FILTRATION ≽ ABORT_SYMPLECTIC ≽ ERROR.

    ABORT_FILTRATION es absorbente respecto de la filtración DIKW
    (coincide con ``FiltrationAbortSignal`` = \(\varnothing\) del monoide).
    """
    ACCEPT              = "Ω₃-ACCEPT"
    ACCEPT_WITH_RESIDUE = "Ω₃-ACCEPT-RESIDUE"
    RECONFIGURE         = "Ω₃-RECONFIGURE"
    ABORT_HOMOLOGY      = "Ω₃-ABORT-HOMOLOGY"
    ABORT_KAM           = "Ω₃-ABORT-KAM"
    ABORT_CHAOS         = "Ω₃-ABORT-CHAOS"
    ABORT_FILTRATION    = "Ω₃-ABORT-FILTRATION"
    ABORT_SYMPLECTIC    = "Ω₃-ABORT-SYMPLECTIC"
    ERROR               = "Ω₃-ERROR"

    @property
    def is_accepting(self) -> bool:
        return self in (Omega3Status.ACCEPT, Omega3Status.ACCEPT_WITH_RESIDUE)

    @property
    def is_absorbing(self) -> bool:
        return self in (
            Omega3Status.ABORT_FILTRATION, Omega3Status.ABORT_SYMPLECTIC,
            Omega3Status.ERROR,
        )


@dataclass(frozen=True, slots=True)
class MayerVietorisCertificate:
    r"""
    Exactitud cardinal de Mayer–Vietoris en \(H_0\):

        \(|A\cup B| = |A| + |B| - |A\cap B|\).

    El defecto \(\delta = |U| - (|A|+|B|-|I|)\) es un proxy de
    \(\Delta\beta_0\); \(\Delta\beta_1\) se estima por duplicados emergentes.
    Exactitud \(\Leftrightarrow \delta=0\) y \(\Delta\beta_1=0\).
    """
    card_A:            int
    card_B:            int
    card_intersection: int
    card_union:        int
    mv_expected:       int
    mv_defect:         int
    delta_beta1:       int
    exact_at_H0:       bool
    passed:            bool
    notes:             Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FiltrationCertificate:
    r"""
    Certificado de la cadena de filtración

        \(V_{\mathrm{PHYSICS}}\subset V_{\mathrm{TACTICS}}
          \subset V_{\mathrm{STRATEGY}}\subset V_{\mathrm{WISDOM}}\).

    Equivale a \(\mathrm{supp}(\mu * 1_V)\) concentrado en maximales
    (inversión de Möbius sobre el poset DIKW de Fase 2).
    """
    validated:     Tuple[str, ...]
    missing:       Tuple[str, ...]
    is_a_filter:   bool
    max_stratum:   Optional[str]
    mobius_support: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Omega3Verdict:
    r"""
    ← ARTEFACTO TERMINAL DE LA FASE 3.

    Morfismo \(\Omega_3\colon \mathbf{Decide}\to\mathbf{DataProduct}\)
    en la categoría de ejecuciones.  El score \(s\in[0,1]\) es la media
    de indicadores booleanos (KAM, Melnikov, N3, filtración, homología,
    simplecticidad, Lyapunov, Arnold).  Los axiomas listan las leyes
    verificadas o violadas.
    """
    status:       Omega3Status
    score:        float
    axioms:       Tuple[str, ...]
    filtration:   FiltrationCertificate
    mayer_vietoris: Optional[MayerVietorisCertificate]
    n3_idempotent: bool
    n3_nontrivial: bool
    kam_stable:   bool
    tangle_free:  bool
    symplectic_ok: bool
    session_id:   str
    lineage_hash: str
    timestamp:    datetime.datetime
    details:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": round(float(self.score), 6),
            "axioms": list(self.axioms),
            "filtration": {
                "validated": list(self.filtration.validated),
                "missing": list(self.filtration.missing),
                "is_a_filter": self.filtration.is_a_filter,
                "max_stratum": self.filtration.max_stratum,
            },
            "mayer_vietoris": (asdict(self.mayer_vietoris)
                               if self.mayer_vietoris is not None else None),
            "n3_idempotent": self.n3_idempotent,
            "n3_nontrivial": self.n3_nontrivial,
            "kam_stable": self.kam_stable,
            "tangle_free": self.tangle_free,
            "symplectic_ok": self.symplectic_ok,
            "session_id": self.session_id,
            "lineage_hash": self.lineage_hash,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class OperationalPolicy:
    r"""
    Recubrimiento *mutable* de política.  No se muta ``PipelineConfig``
    (frozen): la política inducida por el ``OrientationSeed`` vive aquí
    y tiene precedencia sobre la configuración estática.
    """
    enforce_filtration: bool = True
    enforce_homology:   bool = True
    kam_restored:       bool = True
    chaos_damped:       bool = True
    alpha_tolerance:    float = 0.1
    beta_tolerance:     float = 0.1
    n3_theta_star:      Tuple[float, ...] = ()
    n3_alpha_star:      Tuple[float, ...] = ()
    n3_final_loss:      float = 0.0
    pyraga_gain:        float = 0.0
    nekhoroshev_scale:  float = 1.0
    dag_weight_overlay: Optional[NDArray[np.float64]] = None
    poincare_period:    Optional[float] = None
    seed_ingested:      bool = False


def _to_plain(obj: Any) -> Any:
    """Proyección JSON-segura (ndarray / dataclass / datetime / set)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Omega3Status):
        return obj.value
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if issubclass(type(obj), tuple) and hasattr(obj, "_asdict"):
        return _to_plain(obj._asdict())
    try:
        if hasattr(obj, "__dataclass_fields__"):
            return _to_plain(asdict(obj))
    except Exception:
        pass
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    return str(obj)


# ── 3.1 CONFIGURACIÓN ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepConfig:
    enabled:               bool = True
    timeout_seconds:       Optional[int] = None
    retry_count:           int = 0
    retry_delay_seconds:   float = 1.0
    use_memoization:       bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=int(data.get("retry_count", 0)),
            retry_delay_seconds=float(data.get("retry_delay_seconds", 1.0)),
            use_memoization=bool(data.get("use_memoization", True)),
        )


@dataclass(frozen=True)
class SessionConfig:
    session_dir:        Path = field(default_factory=lambda: Path(DEFAULT_SESSION_DIR))
    max_age_hours:      int = 24
    cleanup_on_success: bool = True
    persist_on_error:   bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.session_dir, Path):
            object.__setattr__(self, "session_dir", Path(self.session_dir))


@dataclass(frozen=True)
class PipelineConfig:
    session:            SessionConfig = field(default_factory=SessionConfig)
    step_configs:       Dict[str, StepConfig] = field(default_factory=dict)
    recipe:             Optional[List[str]] = None
    enforce_filtration: bool = True
    enforce_homology:   bool = HOMOLOGICAL_AUDIT_ENABLED
    file_profiles:      Dict[str, Dict[str, Any]] = field(default_factory=dict)
    raw_config:         Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        session_data = data.get("session", {})
        session_dir = session_data.get(
            "session_dir", data.get("session_dir", DEFAULT_SESSION_DIR))
        step_configs: Dict[str, StepConfig] = {}
        for step in PipelineSteps:
            step_data = data.get("steps", {}).get(step.value, {})
            step_configs[step.value] = StepConfig.from_dict(step_data)
        raw_recipe = data.get("pipeline_recipe")
        parsed_recipe = None
        if raw_recipe is not None:
            parsed_recipe = []
            for item in raw_recipe:
                if isinstance(item, dict) and "step" in item:
                    sn = item["step"]
                    parsed_recipe.append(sn)
                    old = step_configs.get(sn, StepConfig())
                    step_configs[sn] = StepConfig(
                        enabled=item.get("enabled", old.enabled),
                        timeout_seconds=old.timeout_seconds,
                        retry_count=old.retry_count,
                        retry_delay_seconds=old.retry_delay_seconds,
                        use_memoization=old.use_memoization)
                elif isinstance(item, str):
                    parsed_recipe.append(item)
        return cls(
            session=SessionConfig(
                session_dir=Path(session_dir),
                max_age_hours=int(session_data.get("max_age_hours", 24)),
                cleanup_on_success=bool(session_data.get("cleanup_on_success", True)),
                persist_on_error=bool(session_data.get("persist_on_error", True))),
            step_configs=step_configs, recipe=parsed_recipe,
            enforce_filtration=bool(data.get("enforce_filtration", True)),
            enforce_homology=bool(data.get("enforce_homology", HOMOLOGICAL_AUDIT_ENABLED)),
            file_profiles=data.get("file_profiles", {}), raw_config=data)

    def get_step_config(self, step_name: str) -> StepConfig:
        return self.step_configs.get(step_name, StepConfig())

    def is_step_enabled(self, step_name: str) -> bool:
        return self.get_step_config(step_name).enabled


# ── 3.2 ENSAMBLADOR MAYER–VIETORIS ───────────────────────────────────────────

class MayerVietorisAssembler:
    r"""
    Ensambla \(A\cup B\) y certifica la sucesión exacta en \(H_0\).

    Usado por ``AuditedMergeStep`` (unión de dataframes) y por el
    veredicto \(\Omega_3\) (unión de evidencias de estrato).
    """

    def __init__(self, telemetry: Optional[TelemetryContext] = None) -> None:
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _id_set(df: pd.DataFrame) -> Optional[Set[Any]]:
        col = next((c for c in ("id", "ID", "Id", "idx", "APU", "CODIGO")
                    if c in df.columns), None)
        if col is None:
            return None
        return set(df[col].dropna().unique())

    def certify_union(
        self,
        df_a: Optional[pd.DataFrame],
        df_b: Optional[pd.DataFrame],
        df_union: Optional[pd.DataFrame],
    ) -> MayerVietorisCertificate:
        if df_a is None or df_b is None or df_union is None:
            return MayerVietorisCertificate(
                card_A=0, card_B=0, card_intersection=0, card_union=0,
                mv_expected=0, mv_defect=0, delta_beta1=0,
                exact_at_H0=True, passed=True,
                notes=("skipped: missing operand",),
            )
        ids_a = self._id_set(df_a)
        ids_b = self._id_set(df_b)
        ids_u = self._id_set(df_union)
        notes: List[str] = []
        if ids_a is None or ids_b is None or ids_u is None:
            # Fallback cardinal bruto (filas).
            a, b, u = len(df_a), len(df_b), len(df_union)
            expected = max(a, b)
            defect = u - expected
            notes.append("no-id-column: using raw cardinality")
            exact = abs(defect) <= max(1, int(0.05 * expected))
            return MayerVietorisCertificate(
                card_A=a, card_B=b, card_intersection=0, card_union=u,
                mv_expected=expected, mv_defect=int(defect),
                delta_beta1=max(0, u - (a + b)),
                exact_at_H0=exact, passed=exact, notes=tuple(notes),
            )
        inter = ids_a & ids_b
        expected = len(ids_a) + len(ids_b) - len(inter)
        defect = len(ids_u) - expected
        extras = ids_u - (ids_a | ids_b)
        missing = (ids_a | ids_b) - ids_u
        if extras:
            notes.append(f"emergent ids: {len(extras)}")
        if missing:
            notes.append(f"lost ids: {len(missing)}")
        delta_b1 = len(extras)
        exact = defect == 0 and delta_b1 == 0
        passed = abs(defect) <= max(1, int(0.05 * max(expected, 1))) and delta_b1 == 0
        return MayerVietorisCertificate(
            card_A=len(ids_a), card_B=len(ids_b),
            card_intersection=len(inter), card_union=len(ids_u),
            mv_expected=expected, mv_defect=int(defect),
            delta_beta1=int(delta_b1),
            exact_at_H0=exact, passed=passed, notes=tuple(notes),
        )

    def certify_stratum_cover(self, state: StateVector) -> MayerVietorisCertificate:
        r"""Mayer–Vietoris sobre evidencias PHYSICS ∪ TACTICS (merge operativo)."""
        return self.certify_union(state.df_presupuesto, state.df_apus_raw,
                                  state.df_merged if state.df_merged is not None
                                  else state.df_final)


# ── 3.3 CLASE BASE DE PASOS ──────────────────────────────────────────────────

class BaseProcessingStep(ABC):
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
        step_name = self._get_step_name()
        start_time = time.monotonic()
        telemetry.start_step(step_name)
        try:
            self._validate_preconditions(state, step_name)
            step_config = self.config.get_step_config(step_name)
            if step_config.use_memoization:
                cached = self.memoizer.lookup(
                    state, step_name, str(state.validated_strata))
                if cached is not None:
                    output, _ = cached
                    new_state = self._merge_output_into_state(state, output)
                    duration_ms = (time.monotonic() - start_time) * 1000
                    ps = PipelineSteps.from_string(step_name)
                    result = StepResult(
                        step_name=step_name, status=StepStatus.CACHED,
                        stratum=_STEP_STRATA.get(ps, Stratum.PHYSICS),
                        duration_ms=duration_ms,
                        state_vector_hash=new_state.compute_hash(),
                        memoization_hit=True)
                    new_state.step_results.append(result)
                    telemetry.end_step(step_name, "cached")
                    self.logger.info("✓ %s (CACHED, %.0fms)", step_name, duration_ms)
                    return new_state

            new_state = self._execute_impl(state, telemetry, mic)
            if new_state is None:
                raise StepExecutionError("Step returned None state", step_name)

            if step_config.use_memoization:
                output = self._extract_output(new_state)
                sig = TensorSignature.compute(output)
                self.memoizer.store(
                    state, step_name, str(new_state.validated_strata),
                    output, sig)

            duration_ms = (time.monotonic() - start_time) * 1000
            ps = PipelineSteps.from_string(step_name)
            result = StepResult(
                step_name=step_name, status=StepStatus.SUCCESS,
                stratum=_STEP_STRATA.get(ps, Stratum.PHYSICS),
                duration_ms=duration_ms,
                state_vector_hash=new_state.compute_hash())
            new_state.step_results.append(result)
            new_state.updated_at = datetime.datetime.now(datetime.timezone.utc)
            telemetry.end_step(step_name, "success")
            self.logger.info("✓ %s (%.0fms)", step_name, duration_ms)
            return new_state
        except PipelineError:
            telemetry.end_step(step_name, "error")
            raise
        except Exception as e:
            self.logger.error("Error in %s: %s", step_name, e, exc_info=True)
            telemetry.record_error(step_name, str(e))
            telemetry.end_step(step_name, "error")
            raise StepExecutionError(str(e), step_name, e)

    def _get_step_name(self) -> str:
        import re
        name = self.__class__.__name__
        if name.endswith("Step"):
            name = name[:-4]
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def _validate_preconditions(self, state: StateVector, step_name: str) -> None:
        missing: List[str] = []
        for key in self.REQUIRED_CONTEXT_KEYS:
            value = getattr(state, key, None)
            if value is None:
                missing.append(key)
            elif isinstance(value, pd.DataFrame) and value.empty:
                missing.append(f"{key} (empty)")
        if missing:
            raise PreconditionError(step_name, missing)

    def _merge_output_into_state(self, state: StateVector,
                                 output: Dict[str, Any]) -> StateVector:
        state_dict = asdict(state)
        state_dict.update(output)
        return StateVector(**state_dict)

    def _extract_output(self, state: StateVector) -> Dict[str, Any]:
        return {k: getattr(state, k) for k in self.PRODUCED_CONTEXT_KEYS
                if hasattr(state, k)}

    @abstractmethod
    def _execute_impl(self, state: StateVector,
                      telemetry: TelemetryContext,
                      mic: MICRegistry) -> StateVector:
        ...


# ── 3.4 PASOS CONCRETOS ──────────────────────────────────────────────────────

from app.physics.lithological_manifold import (
    LithologicalManifold, LithologicalSingularityError,
)


class LithologicalBaselineStep(BaseProcessingStep):
    PRODUCED_CONTEXT_KEYS = (
        "lithological_config", "system_capacitance",
        "system_inductance", "base_resistance")

    def _execute_impl(self, state, telemetry, mic):
        lithological_context = self.config.raw_config.get("lithological_context")
        if not lithological_context:
            raise StepExecutionError(
                "Missing 'lithological_context' payload.",
                "lithological_baseline")

        class _ConcreteLithologicalManifold(LithologicalManifold):
            @property
            def domain(self): return None
            @property
            def codomain(self): return None

        manifold = _ConcreteLithologicalManifold()
        try:
            categorical_state = manifold(lithological_context, {})
        except LithologicalSingularityError as e:
            raise StepExecutionError(
                f"Lithological Singularity detected: {e}",
                "lithological_baseline") from e
        except Exception as e:
            raise StepExecutionError(
                f"Failed to evaluate lithological manifold: {e}",
                "lithological_baseline") from e

        report = categorical_state.payload.get("lithological_report", {})
        sys_c = report.get("swelling_potential_index", 0.0) * 1000.0 + 5000.0
        sys_l = report.get("yielding_susceptibility_index", 0.0) * 5.0 + 2.0
        base_r = report.get("dynamic_rigidity_modulus_pa", 10.0) / 1e6
        if base_r < 0.1:
            base_r = 10.0

        baseline_vars = {
            "system_capacitance": sys_c,
            "system_inductance": sys_l,
            "base_resistance": base_r,
            "lithological_baseline_initialized": True,
            "lithological_report": report,
        }
        state.lithological_config = baseline_vars
        state.system_capacitance = sys_c
        state.system_inductance = sys_l
        state.base_resistance = base_r
        self.config.raw_config.update(baseline_vars)
        return state


class LoadDataStep(BaseProcessingStep):
    PRODUCED_CONTEXT_KEYS = (
        "df_presupuesto", "df_insumos", "df_apus_raw",
        "raw_records", "parse_cache")

    def _execute_impl(self, state, telemetry, mic):
        config_dict = self.config.raw_config
        presupuesto_path = state.presupuesto_path or config_dict.get("presupuesto_path")
        apus_path = state.apus_path or config_dict.get("apus_path")
        insumos_path = state.insumos_path or config_dict.get("insumos_path")
        if not all([presupuesto_path, apus_path, insumos_path]):
            raise PreconditionError(
                "load_data",
                ["presupuesto_path", "apus_path", "insumos_path"])

        fv = FileValidator()
        for path, name in [(presupuesto_path, "presupuesto"),
                           (apus_path, "APUs"),
                           (insumos_path, "insumos")]:
            ok, err = fv.validate_file_exists(path, name)
            if not ok:
                raise DataValidationError(err, "load_data")

        file_profiles = self.config.file_profiles
        p_cfg = {
            "presupuesto_column_map": config_dict.get("presupuesto_column_map", {}),
            "clean_apu_code_params":  config_dict.get("clean_apu_code_params", {})}
        p_processor = PresupuestoProcessor(
            p_cfg, self.thresholds,
            file_profiles.get("presupuesto_default", {}))
        df_presupuesto = p_processor.process(presupuesto_path)
        if df_presupuesto is None or df_presupuesto.empty:
            raise DataValidationError(
                "Presupuesto processing returned empty DataFrame", "load_data")
        telemetry.record_metric("load_data", "presupuesto_rows", len(df_presupuesto))

        i_processor = InsumosProcessor(
            self.thresholds, file_profiles.get("insumos_default", {}))
        df_insumos = i_processor.process(insumos_path)
        if df_insumos is None or df_insumos.empty:
            raise DataValidationError(
                "Insumos processing returned empty DataFrame", "load_data")
        telemetry.record_metric("load_data", "insumos_rows", len(df_insumos))

        config_to_pass = config_dict.copy()
        config_to_pass.setdefault("system_capacitance", 5000.0)
        config_to_pass.setdefault("system_inductance", 2.0)
        config_to_pass.setdefault("base_resistance", 10.0)

        flux_result = mic.project_intent(
            "stabilize_flux",
            {"file_path": str(apus_path), "config": config_to_pass}, {})
        if not flux_result.get("success", False):
            raise StepExecutionError(
                f"stabilize_flux failed: {flux_result.get('error')}",
                "load_data")
        df_apus_raw = pd.DataFrame(flux_result["data"])
        if df_apus_raw.empty:
            raise DataValidationError("DataFluxCondenser returned empty", "load_data")

        parse_result = mic.project_intent(
            "parse_raw",
            {"file_path": str(apus_path),
             "profile": file_profiles.get("apus_default", {})}, {})
        if not parse_result.get("success", False):
            raise StepExecutionError(
                f"parse_raw failed: {parse_result.get('error')}", "load_data")

        state.df_presupuesto = df_presupuesto
        state.df_insumos = df_insumos
        state.df_apus_raw = df_apus_raw
        state.raw_records = parse_result.get("raw_records", [])
        state.parse_cache = parse_result.get("parse_cache", {})
        # PHYSICS queda certificado por evidencia; TACTICS se gana más adelante.
        state.validated_strata = set(state.validated_strata)
        state.validated_strata.add(Stratum.PHYSICS)
        return state


class AuditedMergeStep(BaseProcessingStep):
    REQUIRED_CONTEXT_KEYS = ("df_apus_raw", "df_insumos")
    PRODUCED_CONTEXT_KEYS = ("df_merged",)

    def _execute_impl(self, state, telemetry, mic):
        df_b = state.df_apus_raw
        df_insumos = state.df_insumos
        merger = DataMerger(self.thresholds)
        df_merged = merger.merge_apus_with_insumos(df_b, df_insumos)

        if (state.df_presupuesto is not None
                and "CANTIDAD_PRESUPUESTO" not in df_merged.columns
                and "CANTIDAD_PRESUPUESTO" in state.df_presupuesto.columns):
            df_merged["CANTIDAD_PRESUPUESTO"] = (
                state.df_presupuesto["CANTIDAD_PRESUPUESTO"])

        if state.df_presupuesto is not None and self.config.enforce_homology:
            self.logger.info("Iniciando auditoría homológica (Mayer-Vietoris)...")
            if self.auditor is None:
                self.auditor = HomologicalAuditor(telemetry=TelemetryContext())
            audit = self.auditor.audit_merge(
                state.df_presupuesto, df_b, df_merged, state)
            mv = MayerVietorisAssembler(telemetry).certify_union(
                state.df_presupuesto, df_b, df_merged)
            audit["mayer_vietoris"] = asdict(mv)
            if audit["warnings"] or not mv.passed:
                state.integration_risk_alert = audit
                for w in audit["warnings"]:
                    self.logger.warning("Homology Alert: %s", w)
                for n in mv.notes:
                    self.logger.warning("MV: %s", n)
            if ((not audit["passed"] or not mv.passed)
                    and self.config.enforce_homology):
                raise HomologicalAuditError(
                    "Merge would violate homological / Mayer–Vietoris properties",
                    audit)

        if df_merged is None or df_merged.empty:
            raise DataValidationError("Merge produced empty DataFrame", "audited_merge")

        telemetry.record_metric("audited_merge", "merged_rows", len(df_merged))
        state.df_merged = df_merged
        return state


class CalculateCostsStep(BaseProcessingStep):
    REQUIRED_CONTEXT_KEYS = ("df_merged",)
    PRODUCED_CONTEXT_KEYS = ("df_apu_costos", "df_tiempo", "df_rendimiento")

    def _execute_impl(self, state, telemetry, mic):
        logic_result = mic.project_intent(
            "structure_logic",
            {"raw_records": state.raw_records, "parse_cache": state.parse_cache,
             "config": self.config.raw_config}, {})
        if (logic_result.get("success", False)
                and "df_apu_costos" in logic_result):
            state.df_apu_costos = logic_result["df_apu_costos"]
            state.df_tiempo = logic_result.get("df_tiempo", pd.DataFrame())
            state.df_rendimiento = logic_result.get("df_rendimiento", pd.DataFrame())
            state.quality_report = logic_result.get("quality_report", {})
        else:
            processor = APUProcessor(self.config.raw_config)
            thresholds = ProcessingThresholds(
                self.config.raw_config.get("validation_thresholds", {}))
            state.df_merged = calculate_insumo_costs(state.df_merged, thresholds)
            (state.df_apu_costos, state.df_tiempo,
             state.df_rendimiento) = processor.process_vectors(state.df_merged)
        telemetry.record_metric("calculate_costs", "costos_rows",
                                len(state.df_apu_costos)
                                if state.df_apu_costos is not None else 0)
        return state


class FinalMergeStep(BaseProcessingStep):
    REQUIRED_CONTEXT_KEYS = ("df_presupuesto", "df_apu_costos")
    PRODUCED_CONTEXT_KEYS = ("df_final",)

    def _execute_impl(self, state, telemetry, mic):
        processor = APUProcessor(self.config.raw_config)
        state.df_final = processor.consolidate_results(
            state.df_presupuesto, state.df_apu_costos, state.df_tiempo)
        telemetry.record_metric("final_merge", "final_rows",
                                len(state.df_final) if state.df_final is not None else 0)
        return state


class BusinessTopologyStep(BaseProcessingStep):
    REQUIRED_CONTEXT_KEYS = ("df_final",)
    PRODUCED_CONTEXT_KEYS = ("graph", "business_topology_report")

    def _execute_impl(self, state, telemetry, mic):
        builder = BudgetGraphBuilder()
        df_merged = state.df_merged if state.df_merged is not None else pd.DataFrame()
        state.graph = builder.build(state.df_final, df_merged)
        telemetry.record_metric("business_topology", "graph_nodes",
                                state.graph.number_of_nodes())
        telemetry.record_metric("business_topology", "graph_edges",
                                state.graph.number_of_edges())
        try:
            agent = BusinessAgent(config=self.config.raw_config,
                                  mic=mic, telemetry=telemetry)
            state.business_topology_report = agent.evaluate_project(asdict(state))
        except Exception as e:
            self.logger.warning("BusinessAgent degraded: %s", e)
        return state


class MaterializationStep(BaseProcessingStep):
    PRODUCED_CONTEXT_KEYS = ("bill_of_materials", "logistics_plan")

    def _execute_impl(self, state, telemetry, mic):
        if state.business_topology_report is None:
            self.logger.warning("business_topology_report ausente. Omitida")
            return state
        if state.graph is None:
            builder = BudgetGraphBuilder()
            df_merged = state.df_merged if state.df_merged is not None else pd.DataFrame()
            state.graph = builder.build(state.df_final, df_merged)
        report = state.business_topology_report
        stability = 10.0
        if hasattr(report, "details") and isinstance(report.details, dict):
            stability = report.details.get("pyramid_stability", 10.0)
        generator = MatterGenerator()
        state.bill_of_materials = generator.materialize_project(
            state.graph, flux_metrics={"pyramid_stability": stability,
                                       "avg_saturation": 0.0},
            telemetry=telemetry)
        state.logistics_plan = asdict(state.bill_of_materials)
        telemetry.record_metric("materialization", "total_items",
                                len(state.bill_of_materials.requirements))
        return state


class BuildOutputStep(BaseProcessingStep):
    REQUIRED_CONTEXT_KEYS = ("df_final", "df_insumos", "df_merged",
                             "df_apus_raw", "df_apu_costos")
    PRODUCED_CONTEXT_KEYS = ("final_result",)

    def _execute_impl(self, state, telemetry, mic):
        df_merged = synchronize_data_sources(state.df_merged, state.df_final)
        df_processed_apus = build_processed_apus_dataframe(
            state.df_apu_costos, state.df_apus_raw,
            state.df_tiempo, state.df_rendimiento)
        if (state.graph is not None
                and state.business_topology_report is not None):
            translator = SemanticTranslator(mic=mic)
            result_dict = translator.assemble_data_product(
                state.graph, state.business_topology_report)
            result_dict["presupuesto"] = state.df_final.to_dict("records")
            result_dict["insumos"] = state.df_insumos.to_dict("records")
        else:
            result_dict = build_output_dictionary(
                state.df_final, state.df_insumos, df_merged,
                state.df_apus_raw, df_processed_apus)
        validated_result = validate_and_clean_data(
            result_dict, telemetry_context=telemetry)
        validated_result["raw_insumos_df"] = state.df_insumos.to_dict("records")
        if state.business_topology_report is not None:
            try:
                validated_result["audit_report"] = asdict(state.business_topology_report)
            except Exception:
                validated_result["audit_report"] = {"status": "available"}
        if state.logistics_plan is not None:
            validated_result["logistics_plan"] = state.logistics_plan
        try:
            narrator = TelemetryNarrator(mic=mic)
            validated_result["technical_audit"] = narrator.summarize_execution(telemetry)
        except Exception as e:
            validated_result["technical_audit"] = {
                "status": "degraded", "error": str(e)}
        lineage_hash = self._compute_lineage_hash(validated_result)
        state.final_result = {
            "kind": "DataProduct",
            "metadata": {
                "version": PIPELINE_VERSION,
                "lineage_hash": lineage_hash,
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "generator": f"APU_Filter_Pipeline_{PIPELINE_VERSION}",
                "strata_validated": [s.name for s in state.validated_strata
                                     if isinstance(s, Stratum)],
            },
            "payload": validated_result,
        }
        return state

    def _compute_lineage_hash(self, payload: Dict[str, Any]) -> str:
        hash_parts = []
        for key in sorted(payload.keys()):
            value = payload[key]
            try:
                sanitized = (sanitize_for_json(value)
                             if isinstance(value, (list, dict)) else value)
                part = json.dumps({key: sanitized}, sort_keys=True, default=str)
            except (TypeError, ValueError):
                part = (f"{key}:type={type(value).__name__},"
                        f"len={len(value) if hasattr(value, '__len__') else 'N/A'}")
            hash_parts.append(part)
        return hashlib.sha256("|".join(hash_parts).encode("utf-8")).hexdigest()


# ── 3.5 REGISTRO Y DAG BUILDER ───────────────────────────────────────────────

STEP_REGISTRY: Final[Dict[str, Type[BaseProcessingStep]]] = {
    PipelineSteps.LITHOLOGICAL_BASELINE.value: LithologicalBaselineStep,
    PipelineSteps.LOAD_DATA.value:             LoadDataStep,
    PipelineSteps.AUDITED_MERGE.value:         AuditedMergeStep,
    PipelineSteps.CALCULATE_COSTS.value:       CalculateCostsStep,
    PipelineSteps.FINAL_MERGE.value:           FinalMergeStep,
    PipelineSteps.BUSINESS_TOPOLOGY.value:     BusinessTopologyStep,
    PipelineSteps.MATERIALIZATION.value:       MaterializationStep,
    PipelineSteps.BUILD_OUTPUT.value:          BuildOutputStep,
}


class DAGBuilder:
    @staticmethod
    def build_default_dag() -> AlgebraicDAG:
        dag = AlgebraicDAG()
        for step in PipelineSteps:
            dag.add_step(step.value)
        dependencies = [
            ("lithological_baseline", "load_data",
             ["lithological_config"]),
            ("load_data", "audited_merge", ["df_apus_raw", "df_insumos"]),
            ("audited_merge", "calculate_costs", ["df_merged"]),
            ("calculate_costs", "final_merge", ["df_apu_costos", "df_tiempo"]),
            ("final_merge", "business_topology", ["df_final"]),
            ("business_topology", "materialization",
             ["graph", "business_topology_report"], True),
            ("business_topology", "build_output",
             ["graph", "business_topology_report"]),
        ]
        for source, target, keys, *optional in dependencies:
            is_optional = optional[0] if optional else False
            dag.add_dependency(source, target, keys, is_optional)
        dag.validate()
        return dag


# ── 3.6 SESSION MANAGER ──────────────────────────────────────────────────────

class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config.session_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.config.session_dir / f"{session_id}{SESSION_FILE_EXTENSION}"

    def load(self, session_id: str) -> Optional[StateVector]:
        if not session_id:
            return None
        p = self._session_path(session_id)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, StateVector):
                return data
            if isinstance(data, dict):
                return StateVector.from_dict(data)
            raise SessionCorruptedError(
                session_id, f"Expected StateVector, got {type(data).__name__}")
        except (pickle.UnpicklingError, EOFError) as e:
            raise SessionCorruptedError(session_id, str(e))
        except SessionCorruptedError:
            raise
        except Exception as e:
            self.logger.error("Failed to load session %s: %s", session_id, e)
            return None

    def save(self, state: StateVector) -> bool:
        p = self._session_path(state.session_id)
        tmp = p.with_suffix(f"{SESSION_FILE_EXTENSION}.tmp")
        try:
            data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            if len(data) > MAX_CONTEXT_SIZE_BYTES:
                self.logger.error("Session too large: %s", len(data))
                return False
            with open(tmp, "wb") as f:
                f.write(data)
            tmp.replace(p)
            return True
        except Exception as e:
            self.logger.error("Failed to save session: %s", e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

    def delete(self, session_id: str) -> bool:
        p = self._session_path(session_id)
        try:
            if p.exists():
                p.unlink()
            return True
        except OSError as e:
            self.logger.warning("Could not delete session %s: %s", session_id, e)
            return False

    def cleanup_old_sessions(self) -> int:
        cutoff = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(hours=self.config.max_age_hours))
        deleted = 0
        for p in self.config.session_dir.glob(f"*{SESSION_FILE_EXTENSION}"):
            try:
                mtime = datetime.datetime.fromtimestamp(
                    p.stat().st_mtime, tz=datetime.timezone.utc)
                if mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except Exception as e:
                self.logger.warning("Could not check/delete %s: %s", p, e)
        if deleted:
            self.logger.info("Cleaned up %d old sessions", deleted)
        return deleted


# ── 3.7 PIPELINE DIRECTOR ────────────────────────────────────────────────────

class PipelineDirector:
    r"""
    Director del Pipeline — Fase 3 (DECIDE).

    Implementa ``Phase3DecideDirector``.  Ingiere el ``OrientationSeed``
    de la Fase 2 como condición de frontera, ejecuta pasos según el DAG,
    certifica Mayer–Vietoris y filtración, y emite el DataProduct con
    veredicto \(\Omega_3\).
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], PipelineConfig],
        telemetry: TelemetryContext,
        mic: Optional[MICRegistry] = None,
    ):
        self.config = (PipelineConfig.from_dict(config)
                       if isinstance(config, dict) else config)
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thresholds = self._load_thresholds()

        if mic is not None:
            self.mic = mic
        else:
            self.mic = MICRegistry()
            register_core_vectors(self.mic, self.config.raw_config)

        self.dag = DAGBuilder.build_default_dag()
        self.memoizer = OperatorMemoizer()
        self.auditor = HomologicalAuditor(telemetry)
        self.mv_assembler = MayerVietorisAssembler(telemetry)
        self.session_manager = SessionManager(self.config.session)
        self._dikw = DIKWPoset()

        self.engine = PipelineDirectorEngine()
        self.n3_kernel = RecursiveSelfImprovementKernel(
            n_params=_N3_FEATURE_DIM,
            adjacency_shape=len(list(PipelineSteps)),
            weights=(1.0, 1.0, 0.5, 0.5, 0.75, 0.5, 0.25, 0.15),
            dag=self.dag,
        )
        self.n3_kernel.bind_algebraic_dag(self.dag)

        self._policy = OperationalPolicy(
            enforce_filtration=self.config.enforce_filtration,
            enforce_homology=self.config.enforce_homology,
        )
        self._orientation_seed: Optional[OrientationSeed] = None
        self._last_observer: Optional[ObserverPacket] = None
        self._last_omega3: Optional[Omega3Verdict] = None
        self._last_mv: Optional[MayerVietorisCertificate] = None

        self.logger.info(
            "PipelineDirector v%s initialized | DAG nodes: %d | N3 kernel armed",
            PIPELINE_VERSION, len(self.dag.graph.nodes()))

    def _load_thresholds(self) -> ProcessingThresholds:
        t = ProcessingThresholds()
        overrides = self.config.raw_config.get("processing_thresholds", {})
        if not isinstance(overrides, dict):
            return t
        for k, v in overrides.items():
            if not hasattr(t, k):
                self.logger.warning("Unknown threshold key '%s'", k)
                continue
            cur = getattr(t, k)
            if cur is not None and not isinstance(v, type(cur)):
                self.logger.warning("Threshold '%s': type mismatch", k)
                continue
            setattr(t, k, v)
        return t

    def _filtration_enabled(self) -> bool:
        return bool(self._policy.enforce_filtration)

    def _homology_enabled(self) -> bool:
        return bool(self._policy.enforce_homology)

    # ═══════════════════════════════════════════════════════════════════
    # 3.7.1  ingest_orientation_seed  ←  PRIMER MÉTODO EFECTIVO DE FASE 3
    #         continuación literal de Phase3DecideDirector (Fase 2, §2.6)
    # ═══════════════════════════════════════════════════════════════════
    def ingest_orientation_seed(self, seed: OrientationSeed) -> OperationalPolicy:
        r"""
        ← PRIMER MÉTODO EFECTIVO DE FASE 3.  FUNCTOR DE FRONTERA  Decide ⟵ Orient.

        Traduce el ``OrientationSeed`` (emisión de ``project_to_phase_3``) en
        política operativa, *sin mutar* el ``PipelineConfig`` frozen:

          · \(\theta^*=(\lambda_2,\nu,\varepsilon,|M|_\infty,D,\lambda)\)
            \(\mapsto\) umbrales \(\alpha,\beta\) y holguras.
          · \(\mathrm{kam\_preserved}=\bot\) \(\Rightarrow\) filtración forzosa.
          · \(\mathrm{melnikov\_chaotic}=\top\) \(\Rightarrow\) homología forzosa
            + TTL de memoización acoplado a \(T_P\).
          · \(O^*\) se guarda como overlay de pesos (no reescribe dependencias
            de negocio: el DAG canónico es un invariante de filtración).
          · \(\kappa_{\mathrm{Nek}}\), \(\eta_{\mathrm{Pyragas}}\) escalan
            reintentos y cautela.

        Devuelve la ``OperationalPolicy`` resultante (también queda en
        ``self._policy``).  Idempotente a nivel de asignación: re-ingerir
        un seed sustituye la política, no la compone (la composición vive
        en N3).
        """
        if not isinstance(seed, OrientationSeed):
            raise RecursiveSelfImprovementError(
                f"ingest_orientation_seed espera OrientationSeed, recibió {type(seed)!r}")
        if not seed.nontriviality.is_nontrivial:
            raise RecursiveSelfImprovementError(
                "Seed N3 trivial (∂I₃/∂α≈0): rehusado por el director")

        self._orientation_seed = seed
        policy = self._policy

        self._apply_seed_to_thresholds(seed, policy)
        self._apply_seed_to_enforcement(seed, policy)
        self._apply_seed_to_memoizer(seed, policy)
        self._apply_seed_to_dag_weights(seed, policy)
        self._inject_seed_into_raw_config(seed)

        policy.n3_theta_star = tuple(float(x) for x in np.asarray(seed.theta_final).ravel())
        policy.n3_alpha_star = tuple(float(x) for x in np.asarray(seed.alpha_final).ravel())
        policy.n3_final_loss = float(seed.final_loss)
        policy.pyraga_gain = float(seed.pyraga_gain)
        policy.nekhoroshev_scale = float(seed.nekhoroshev_scale)
        policy.seed_ingested = True

        self.logger.info(
            "[ORIENT→DECIDE] filtración=%s homología=%s KAM=%s chaos=%s "
            "I₃∘I₃=%s ∂I₃/∂α=%.3e loss=%.6e η=%.3e κ_Nek=%.3f",
            policy.enforce_filtration, policy.enforce_homology,
            seed.kam_preserved, seed.melnikov_chaotic,
            seed.idempotence.is_idempotent,
            seed.nontriviality.jacobian_frobenius,
            seed.final_loss, seed.pyraga_gain, seed.nekhoroshev_scale,
        )
        return policy

    # ── 3.7.2 Proyectores granulares del seed ────────────────────────────
    def _apply_seed_to_thresholds(
        self, seed: OrientationSeed, policy: OperationalPolicy,
    ) -> None:
        r"""
        \(\theta[0]=\lambda_2\) (conectividad) \(\mapsto\) tolerancia \(\alpha\).
        \(\theta[1]=\nu\)        (nilpotencia)  \(\mapsto\) tolerancia \(\beta\).
        Residuo N3 alto \(\Rightarrow\) holguras más estrictas.
        """
        theta = np.asarray(seed.theta_final, dtype=np.float64).ravel()
        a = float(np.clip(theta[0] if theta.size > 0 else 0.1, 0.01, 0.5))
        b = float(np.clip(theta[1] if theta.size > 1 else 0.1, 0.01, 0.5))
        if not seed.idempotence.is_idempotent:
            a = min(a, 0.08)
            b = min(b, 0.08)
        policy.alpha_tolerance = a
        policy.beta_tolerance = b
        thr = self.thresholds
        if hasattr(thr, "alpha_tolerance"):
            setattr(thr, "alpha_tolerance", a)
        if hasattr(thr, "beta_tolerance"):
            setattr(thr, "beta_tolerance", b)

    def _apply_seed_to_enforcement(
        self, seed: OrientationSeed, policy: OperationalPolicy,
    ) -> None:
        r"""
        Leyes de forzado:

          \(\lnot\mathrm{KAM}\) \(\Rightarrow\) ``enforce_filtration``
          (los toros DIKW no persisten: hay que reconstruir la cadena).

          \(\mathrm{tangle}\) \(\Rightarrow\) ``enforce_homology``
          (caos homoclínico: Mayer–Vietoris deja de ser opcional).

        Nunca se *relaja* una imposición estática True→False: el seed
        solo puede endurecer la política (monoide \((\mathbb{B},\lor)\)).
        """
        pol_filt = bool(self.config.enforce_filtration)
        pol_hom = bool(self.config.enforce_homology)
        if not seed.kam_preserved:
            self.logger.warning(
                "[ORIENT] KAM no preservado ⇒ forzando enforce_filtration=True")
            pol_filt = True
            policy.kam_restored = False
        else:
            policy.kam_restored = True
        if seed.melnikov_chaotic:
            self.logger.warning(
                "[ORIENT] Caos homoclínico detectado ⇒ auditoría reforzada")
            pol_hom = True
            policy.chaos_damped = False
        else:
            policy.chaos_damped = True
        if not seed.lyapunov.is_monotone:
            pol_hom = True
        policy.enforce_filtration = pol_filt
        policy.enforce_homology = pol_hom

    def _apply_seed_to_memoizer(
        self, seed: OrientationSeed, policy: OperationalPolicy,
    ) -> None:
        """TTL de caché = periodo de Poincaré del paquete observacional."""
        period = seed.observer_packet.return_map.poincare_period
        policy.poincare_period = period
        self.memoizer.set_poincare_period(period)

    def _apply_seed_to_dag_weights(
        self, seed: OrientationSeed, policy: OperationalPolicy,
    ) -> None:
        r"""
        Overlay \(O^*\): *no* se reescriben aristas de negocio.  Se almacena
        como prioridad de scheduling / telemetría.  Un rewire real rompería
        el axioma de filtración (los pasos TACTICS no pueden preceder a
        PHYSICS aunque N2 lo sugiriera).
        """
        O = np.asarray(seed.O_final, dtype=np.float64)
        policy.dag_weight_overlay = O
        # Diagnóstico: si O* introduce un ciclo, se registra y se ignora.
        try:
            probe = AlgebraicDAG.from_adjacency(
                O, [s.value for s in PipelineSteps][: O.shape[0]])
            if probe.has_directed_cycle():
                self.logger.warning(
                    "[ORIENT] O* no es DAG (ciclo dirigido). Overlay retenido "
                    "solo como telemetría; el DAG canónico permanece.")
        except Exception as exc:
            self.logger.debug("O* probe skipped: %s", exc)

    def _inject_seed_into_raw_config(self, seed: OrientationSeed) -> None:
        raw = self.config.raw_config
        raw["n3_theta_star"] = np.asarray(seed.theta_final).tolist()
        raw["n3_alpha_star"] = np.asarray(seed.alpha_final).tolist()
        raw["n3_final_loss"] = float(seed.final_loss)
        raw["n3_policy"] = seed.to_policy_dict()

    # ── 3.7.3 Filtración DIKW (Möbius) ───────────────────────────────────
    def _compute_validated_strata(self, state: StateVector) -> Set[Stratum]:
        validated: Set[Stratum] = set()
        for stratum, keys in _STRATUM_EVIDENCE.items():
            if all(state.get_evidence(stratum).get(k, False) for k in keys):
                validated.add(stratum)
        return validated

    def certify_filtration(self, state: StateVector) -> FiltrationCertificate:
        validated = state.validated_strata
        ok, missing = self._dikw.filtration_ok(validated)
        ranked = sorted(validated, key=stratum_level)
        max_s = ranked[-1].name if ranked else None
        # Soporte de μ * 1_V : elementos donde la inversión es no nula.
        mu = self._dikw.mobius_matrix()
        chi = np.array([1.0 if s in validated else 0.0
                        for s in sorted(_STRATUM_ORDER, key=stratum_level)])
        # (μ * χ)(x) = Σ_{y≥x} μ(x,y) χ(y)  — en cadena, μ es bidiagonal.
        support: List[str] = []
        names = [s.name for s in sorted(_STRATUM_ORDER, key=stratum_level)]
        conv = mu @ chi  # aproximación matricial (orden lineal)
        for i, name in enumerate(names):
            if abs(float(conv[i])) > EPSILON:
                support.append(name)
        return FiltrationCertificate(
            validated=tuple(s.name for s in ranked),
            missing=tuple(missing),
            is_a_filter=ok,
            max_stratum=max_s,
            mobius_support=tuple(support),
        )

    def _check_stratum_prerequisites(self, target: Stratum,
                                     validated: Set[Stratum]) -> bool:
        tl = stratum_level(target)
        if tl == 0:
            return True
        for s, l in _STRATUM_ORDER.items():
            if l < tl and s not in validated:
                return False
        return True

    def _enforce_filtration(self, target: Stratum, state: StateVector) -> None:
        validated = state.validated_strata
        if not self._check_stratum_prerequisites(target, validated):
            tl = stratum_level(target)
            missing = [s.name for s, l in _STRATUM_ORDER.items()
                       if l < tl and s not in validated]
            raise FiltrationViolationError(
                target, missing, [s.name for s in validated])

    def _equalizer_seed(
        self, fab: FiltrationAbortSignal, session_id: str,
    ) -> Dict[str, Any]:
        r"""
        Equalizador categórico vacío: el par de morfismos
        \((\iota_{\mathrm{missing}},\mathrm{id})\) no tiene vértice común
        no nulo.  Semántica: \(\varnothing\circ f=\varnothing\).
        """
        return {
            "kind": "CategoricalEqualizerSeed",
            "status": "ERROR",
            "error_class": "FiltrationViolation",
            "omega3": Omega3Status.ABORT_FILTRATION.value,
            "session_id": session_id,
            "step": fab.step_name or "unknown",
            "target_stratum": fab.target_stratum.name,
            "missing_strata": fab.missing_strata,
            "validated_strata": fab.validated_strata,
            "message": str(fab),
            "axiom": (
                "V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM: "
                "Ley de Clausura Transitiva violada."),
        }

    # ── 3.7.4 Ejecución de un paso ───────────────────────────────────────
    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_state: Optional[StateVector] = None,
        validate_stratum: bool = True,
    ) -> StepResult:
        state = self.session_manager.load(session_id)
        if state is None:
            state = initial_state or StateVector(session_id=session_id)
        elif initial_state:
            for attr in initial_state.__dataclass_fields__:
                v = getattr(initial_state, attr)
                if v is not None:
                    setattr(state, attr, v)

        try:
            step_class = STEP_REGISTRY.get(step_name)
            if step_class is None:
                raise StepExecutionError(f"Step class not found: {step_name}",
                                         step_name)
            ps = PipelineSteps.from_string(step_name)
            if ps and validate_stratum and self._filtration_enabled():
                try:
                    self._enforce_filtration(ps.stratum, state)
                except FiltrationViolationError as fve:
                    self.logger.critical(
                        "[FILTRATION ABORT] Estrato=%s | Faltantes=%s",
                        fve.target_stratum.name, fve.missing_strata)
                    raise FiltrationAbortSignal(
                        fve.target_stratum, fve.missing_strata,
                        fve.validated_strata, step_name=step_name) from fve

            step_instance = step_class(
                self.config, self.thresholds, self.memoizer, self.auditor)
            new_state = step_instance.execute(state, self.telemetry, self.mic)
            new_state.validated_strata = self._compute_validated_strata(new_state)
            self.session_manager.save(new_state)
            if new_state.step_results:
                return new_state.step_results[-1]
            return StepResult(
                step_name=step_name, status=StepStatus.SUCCESS,
                stratum=ps.stratum if ps else Stratum.PHYSICS,
                state_vector_hash=new_state.compute_hash())
        except FiltrationAbortSignal:
            raise
        except PipelineError as e:
            ps = PipelineSteps.from_string(step_name)
            result = StepResult(
                step_name=step_name, status=StepStatus.ERROR,
                stratum=ps.stratum if ps else Stratum.PHYSICS,
                error=str(e))
            if self.config.session.persist_on_error:
                state.step_results.append(result)
                self.session_manager.save(state)
            self.logger.error("Error in %s: %s", step_name, e)
            return result

    # ── 3.7.5 Orquestador completo ───────────────────────────────────────
    def execute_pipeline(
        self,
        initial_context: Dict[str, Any],
        session_id: Optional[str] = None,
        run_poincare_audit: bool = True,
        orientation_seed: Optional[OrientationSeed] = None,
    ) -> Dict[str, Any]:
        r"""
        Bucle orquestador (Fases 1 → 2 → 3 anidadas).

        Estructura:

          (0) Si se provee un ``OrientationSeed`` *a priori*, se ingiere
              como condición de frontera (política endurecida).
          (a) Ejecución secuencial de pasos (cuerpo Decide).
          (b) Fase 1 (OBSERVE): trayectoria \(\mapsto\) ObserverPacket.
          (c) Fase 2 (ORIENT):  ObserverPacket \(\mapsto\) N3 \(\mapsto\) seed.
          (d) Fase 3 (DECIDE):  seed \(\mapsto\) ingest \(\mapsto\) \(\Omega_3\).
        """
        if orientation_seed is not None:
            self.ingest_orientation_seed(orientation_seed)

        session_id = session_id or str(uuid.uuid4())
        self.logger.info(
            "Starting pipeline (Session: %s...) | DAG mode | N3 armed",
            session_id[:8])

        initial_state = StateVector(session_id=session_id)
        for k, v in initial_context.items():
            if hasattr(initial_state, k):
                setattr(initial_state, k, v)
        if not self.session_manager.save(initial_state):
            raise SessionError(
                f"Failed to persist initial state at "
                f"{self.config.session.session_dir}")

        recipe = self.config.recipe or self.dag.topological_sort()

        try:
            for idx, step_name in enumerate(recipe):
                if not self.config.is_step_enabled(step_name):
                    self.logger.info("Skipping disabled step: %s", step_name)
                    continue
                self.logger.info(
                    "Orchestrating [%d/%d]: %s", idx + 1, len(recipe), step_name)
                result = self.run_single_step(
                    step_name, session_id,
                    initial_state if idx == 0 else None)
                if not result.is_successful:
                    raise StepExecutionError(
                        f"Pipeline failed at '{step_name}': {result.error}",
                        step_name)
        except FiltrationAbortSignal as fab:
            self.logger.critical(
                "[PIPELINE ABORT] FiltrationAbortSignal en '%s'.",
                fab.step_name or "unknown")
            abort = self._equalizer_seed(fab, session_id)
            verdict = self.emit_omega3_verdict(
                state=None, session_id=session_id,
                forced_status=Omega3Status.ABORT_FILTRATION,
                extra={"equalizer": abort})
            abort["omega3_verdict"] = verdict.to_dict()
            return abort

        final_state = self.session_manager.load(session_id)
        if final_state is None:
            raise SessionError(f"Failed to load final state {session_id}")

        if run_poincare_audit:
            self._run_nested_phases(final_state)

        verdict = self.emit_omega3_verdict(final_state, session_id=session_id)
        if final_state.final_result is None:
            final_state.final_result = {
                "kind": "DataProduct", "metadata": {}, "payload": {}}
        final_state.final_result.setdefault("metadata", {})
        final_state.final_result["metadata"]["omega3"] = verdict.to_dict()
        final_state.final_result.setdefault("payload", {})
        final_state.final_result["payload"]["omega3_verdict"] = verdict.to_dict()
        self.session_manager.save(final_state)

        if self.config.session.cleanup_on_success and verdict.status.is_accepting:
            self.session_manager.delete(session_id)

        if final_state.final_result:
            self.logger.info(
                "Pipeline completed. DataProduct ready. Ω₃=%s score=%.3f",
                verdict.status.value, verdict.score)
            return final_state.final_result
        self.logger.warning("No final_result. Returning state as dict.")
        out = final_state.to_dict()
        out["omega3_verdict"] = verdict.to_dict()
        return out

    # ── 3.7.6 Bucle anidado Fase 1 → 2 → 3 ──────────────────────────────
    def _run_nested_phases(self, final_state: StateVector) -> OrientationSeed:
        r"""
        Compone los tres funtores sobre la trayectoria registrada:

          \(\mathbf{Obs}(\gamma)\xrightarrow{P_2}
            \mathbf{Orient}(\mathrm{packet})\xrightarrow{P_3}
            \mathbf{Decide}(\mathrm{seed})\).
        """
        packet, seed = self.compose_nested_functors(final_state)
        self.ingest_orientation_seed(seed)
        self._attach_audit_payload(final_state, packet, seed)
        self.session_manager.save(final_state)
        return seed

    def compose_nested_functors(
        self, final_state: StateVector,
    ) -> Tuple[ObserverPacket, OrientationSeed]:
        r"""
        \(P_3\circ P_2\circ\mathrm{Obs}\).  Extrae \(\gamma\), construye \(A\),
        llama a ``engine.project_to_phase_2`` y a ``n3_kernel.project_to_phase_3``.
        """
        trajectory = self._extract_trajectory(final_state)
        if len(trajectory) < 2:
            self.logger.info("[NestedPhases] Trayectoria insuficiente. Semilla degenerada.")
            trajectory = [final_state, final_state]

        adjacency = self._build_adjacency(trajectory)
        k = int(adjacency.shape[0])
        jacobian = np.eye(k, dtype=np.float64)
        omega = canonical_symplectic_form(k)

        observer_packet = self.engine.project_to_phase_2(
            trajectory=trajectory,
            adjacency_A=adjacency,
            symplectic_jacobian_M=jacobian,
            canonical_omega=omega)
        self._last_observer = observer_packet

        self.logger.info(
            "[Fase 1::OBSERVE] Σ crosssections=%d | tangle=%s | KAM=%s "
            "| D_Arnold=%.4e | H_drift=%.4e",
            observer_packet.return_map.n_crossings,
            observer_packet.melnikov.tangle_detected,
            observer_packet.kam.is_KAM_stable,
            observer_packet.arnold.total_diffusion,
            observer_packet.hamiltonian_drift)

        self.n3_kernel.ingest_observer_packet(observer_packet)
        orientation_seed = self.n3_kernel.project_to_phase_3(
            observer_packet=observer_packet, n_cycles=3)
        self._orientation_seed = orientation_seed

        self.logger.info(
            "[Fase 2::ORIENT] θ*=%.4f | α*=%.6f | loss=%.6e | chaotic=%s | "
            "I₃∘I₃=%s",
            float(np.linalg.norm(orientation_seed.theta_final)),
            float(np.linalg.norm(orientation_seed.alpha_final)),
            orientation_seed.final_loss,
            orientation_seed.melnikov_chaotic,
            orientation_seed.idempotence.is_idempotent)
        return observer_packet, orientation_seed

    def _attach_audit_payload(
        self,
        final_state: StateVector,
        observer_packet: ObserverPacket,
        orientation_seed: OrientationSeed,
    ) -> None:
        if final_state.final_result is None:
            final_state.final_result = {
                "kind": "DataProduct", "metadata": {}, "payload": {}}
        payload = final_state.final_result.setdefault("payload", {})
        mv = self.mv_assembler.certify_stratum_cover(final_state)
        self._last_mv = mv
        filt = self.certify_filtration(final_state)
        payload["poincare_audit"] = _to_plain({
            "spectra": asdict(observer_packet.spectra),
            "return_map": {
                "n_crossings": observer_packet.return_map.n_crossings,
                "stability_exponent": observer_packet.return_map.stability_exponent,
                "is_hyperbolic": observer_packet.return_map.is_hyperbolic,
                "is_periodic_orbit": observer_packet.return_map.is_periodic_orbit,
                "poincare_period": observer_packet.return_map.poincare_period,
                "lyapunov_spectrum": observer_packet.return_map.lyapunov_spectrum,
            },
            "melnikov": {
                "simple_zeros": observer_packet.melnikov.simple_zeros,
                "max_magnitude": observer_packet.melnikov.max_magnitude,
                "tangle_detected": observer_packet.melnikov.tangle_detected,
                "simple_zero_locations": observer_packet.melnikov.simple_zero_locations,
                "poisson_mean": observer_packet.melnikov.poisson_mean,
            },
            "kam": asdict(observer_packet.kam),
            "arnold": {
                "total_diffusion": observer_packet.arnold.total_diffusion,
                "dominant_channel": observer_packet.arnold.dominant_channel,
                "is_adiabatic": observer_packet.arnold.is_adiabatic,
                "nekhoroshev_time": observer_packet.arnold.nekhoroshev_time,
                "n_resonant_channels": observer_packet.arnold.n_resonant_channels,
            },
            "hamiltonian": {
                "mean": observer_packet.hamiltonian_mean,
                "drift": observer_packet.hamiltonian_drift,
            },
            "n3_orientation_seed": orientation_seed.to_policy_dict(),
            "mayer_vietoris": asdict(mv),
            "filtration": asdict(filt),
        })
        final_state.poincare_trace = [
            {"t": s.t_crossing, "type": s.section_type.value,
             "hash": s.state_hash, "transverse": s.is_transverse}
            for s in observer_packet.sections
        ]

    def _extract_trajectory(self, final_state: StateVector) -> List[StateVector]:
        """
        Pseudo-trayectoria \(\gamma(t_k)\) a partir de ``step_results``.
        Cada fotograma acumula estratos validados (filtración creciente).
        """
        if not final_state.step_results:
            return [final_state]
        trajectory: List[StateVector] = []
        cumulative_strata: Set[Stratum] = set()
        for sr in final_state.step_results:
            cumulative_strata.add(sr.stratum)
            snapshot = StateVector(
                session_id=final_state.session_id,
                validated_strata=set(cumulative_strata))
            snapshot.step_results = [sr]
            trajectory.append(snapshot)
        trajectory.append(final_state)
        return trajectory

    def _build_adjacency(self, trajectory: List[StateVector]
                         ) -> NDArray[np.float64]:
        """Adyacencia del DAG de pasos restringida a la traza ejecutada."""
        n = len(list(PipelineSteps))
        steps = list(PipelineSteps)
        idx = {s.value: i for i, s in enumerate(steps)}
        A = np.zeros((n, n), dtype=np.float64)
        executed_sequence: List[str] = []
        for snapshot in trajectory:
            for sr in snapshot.step_results:
                executed_sequence.append(sr.step_name)
        for src, dst in zip(executed_sequence[:-1], executed_sequence[1:]):
            if src in idx and dst in idx:
                A[idx[src], idx[dst]] = 1.0
        # Overlay N3: refuerza (no inventa) aristas canónicas.
        overlay = self._policy.dag_weight_overlay
        if overlay is not None and overlay.shape == A.shape:
            A = np.where(A > 0.0, np.maximum(A, overlay), A)
            A = np.triu(A, k=1)
        return A

    # ═══════════════════════════════════════════════════════════════════
    # 3.7.7  emit_omega3_verdict  ←  ÚLTIMO MÉTODO DEL DIRECTOR / FASE 3
    # ═══════════════════════════════════════════════════════════════════
    def emit_omega3_verdict(
        self,
        state: Optional[StateVector],
        session_id: str = "",
        forced_status: Optional[Omega3Status] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Omega3Verdict:
        r"""
        ← ÚLTIMO MÉTODO DE FASE 3.  FUNCTOR TERMINAL  Decide ⟶ DataProduct.

        Agrega los certificados de las tres fases en un único veredicto
        \(\Omega_3\).  Score \(s=\frac1n\sum_i 1_{P_i}\) sobre los predicados:

            \(P_{\mathrm{DAG}},\; P_{\mathrm{Sp}},\; P_{\mathrm{KAM}},\;
              P_{\lnot M},\; P_{\mathrm{Arnold}},\; P_{I_3},\;
              P_{\partial\alpha},\; P_V,\; P_{\mathrm{filt}},\; P_{\mathrm{MV}}\).

        Precedencia absorbente:

            filtración \(\prec\) simplecticidad \(\prec\) homología
            \(\prec\) KAM/caos \(\prec\) residuo N3 \(\prec\) ACCEPT.

        Diagrama cerrado:

            ingest_orientation_seed(seed)          ← frontera F2
                └──► execute_pipeline
                        └──► compose_nested_functors   (P₃ ∘ P₂ ∘ Obs)
                        └──► ingest_orientation_seed   (reconfiguración)
            emit_omega3_verdict  ──Ω₃──► DataProduct.metadata.omega3
        """
        sid = session_id or (state.session_id if state is not None else "")
        now = datetime.datetime.now(datetime.timezone.utc)
        extra = extra or {}

        filt = (self.certify_filtration(state)
                if state is not None
                else FiltrationCertificate((), (), False, None))
        mv = self._last_mv
        if mv is None and state is not None:
            mv = self.mv_assembler.certify_stratum_cover(state)
            self._last_mv = mv

        seed = self._orientation_seed
        packet = self._last_observer or (seed.observer_packet if seed else None)

        n3_idem = bool(seed.idempotence.is_idempotent) if seed else False
        n3_nontr = bool(seed.nontriviality.is_nontrivial) if seed else False
        kam_ok = bool(packet.kam.is_KAM_stable) if packet else False
        tangle = bool(packet.melnikov.tangle_detected) if packet else False
        tangle_free = not tangle
        dag_ok = bool(packet.spectra.is_dag_acyclic) if packet else False
        sp_ok = bool(packet.spectra.is_strictly_symplectic
                     or packet.spectra.symplectic_defect <= _SYMPLECTIC_DEFECT_TOL
                     ) if packet else False
        arnold_ok = bool(packet.arnold.is_adiabatic) if packet else False
        lyap_ok = bool(seed.lyapunov.is_monotone) if seed else False
        filt_ok = bool(filt.is_a_filter)
        mv_ok = bool(mv.passed) if mv is not None else True

        indicators = (
            dag_ok, sp_ok, kam_ok, tangle_free, arnold_ok,
            n3_idem, n3_nontr, lyap_ok, filt_ok, mv_ok,
        )
        score = float(sum(1.0 for p in indicators if p) / max(len(indicators), 1))

        axioms: List[str] = []
        if filt_ok:
            axioms.append("FILTRATION: V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM")
        else:
            axioms.append("FILTRATION VIOLATED: " + ",".join(filt.missing))
        if mv_ok:
            axioms.append("MAYER–VIETORIS: Δβ₀=0, Δβ₁=0")
        else:
            axioms.append(
                f"MAYER–VIETORIS DEFECT: δ={mv.mv_defect if mv else '?'}"
            )
        if kam_ok:
            axioms.append("KAM: tori persisting")
        else:
            axioms.append("KAM BREAKDOWN")
        if tangle_free:
            axioms.append("MELNIKOV: W^s ∩ W^u empty or non-transverse")
        else:
            axioms.append("MELNIKOV TANGLE: homoclinic chaos")
        if n3_idem:
            axioms.append("N3: I₃∘I₃=I₃")
        if n3_nontr:
            axioms.append("N3: ∂I₃/∂α ≠ 0")
        if sp_ok:
            axioms.append("SP(2n): MᵀΩM=Ω")
        if dag_ok:
            axioms.append("DAG: A^n=0")

        if forced_status is not None:
            status = forced_status
        elif not filt_ok and self._filtration_enabled():
            status = Omega3Status.ABORT_FILTRATION
        elif not sp_ok and packet is not None and packet.spectra.symplectic_defect > 1e-3:
            status = Omega3Status.ABORT_SYMPLECTIC
        elif not mv_ok and self._homology_enabled():
            status = Omega3Status.ABORT_HOMOLOGY
        elif not kam_ok and packet is not None and packet.kam.greene_residue > 10.0:
            status = Omega3Status.ABORT_KAM
        elif tangle and seed is not None and seed.pyraga_gain <= 0.0:
            status = Omega3Status.ABORT_CHAOS
        elif (not kam_ok) or tangle or (not n3_idem) or (not lyap_ok):
            status = Omega3Status.RECONFIGURE
        elif score < 1.0:
            status = Omega3Status.ACCEPT_WITH_RESIDUE
        else:
            status = Omega3Status.ACCEPT

        lineage = ""
        if state is not None and isinstance(state.final_result, dict):
            lineage = str(state.final_result.get("metadata", {}).get("lineage_hash", ""))
        if not lineage and state is not None:
            lineage = state.compute_hash()

        verdict = Omega3Verdict(
            status=status,
            score=score,
            axioms=tuple(axioms),
            filtration=filt,
            mayer_vietoris=mv,
            n3_idempotent=n3_idem,
            n3_nontrivial=n3_nontr,
            kam_stable=kam_ok,
            tangle_free=tangle_free,
            symplectic_ok=sp_ok,
            session_id=sid,
            lineage_hash=lineage,
            timestamp=now,
            details=_to_plain({
                "indicators": {
                    "dag": dag_ok, "symplectic": sp_ok, "kam": kam_ok,
                    "tangle_free": tangle_free, "arnold": arnold_ok,
                    "n3_idempotent": n3_idem, "n3_nontrivial": n3_nontr,
                    "lyapunov": lyap_ok, "filtration": filt_ok, "mv": mv_ok,
                },
                "policy": {
                    "enforce_filtration": self._policy.enforce_filtration,
                    "enforce_homology": self._policy.enforce_homology,
                    "pyraga_gain": self._policy.pyraga_gain,
                    "nekhoroshev_scale": self._policy.nekhoroshev_scale,
                    "seed_ingested": self._policy.seed_ingested,
                },
                **extra,
            }),
        )
        self._last_omega3 = verdict
        self.logger.info(
            "[Fase 3::DECIDE] Ω₃=%s score=%.3f axioms=%d",
            verdict.status.value, verdict.score, len(verdict.axioms))
        return verdict

    # ── 3.7.8 Introspección ──────────────────────────────────────────────
    def get_dag_info(self) -> Dict[str, Any]:
        return {
            "version": PIPELINE_VERSION,
            "nodes": list(self.dag.graph.nodes()),
            "edges": list(self.dag.graph.edges()),
            "is_acyclic": nx.is_directed_acyclic_graph(self.dag.graph),
            "topological_order": self.dag.topological_sort(),
            "height": self.dag.poset_height(),
            "betti": {"beta0": self.dag.betti_undirected()[0],
                      "beta1": self.dag.betti_undirected()[1]},
        }

    def get_memoization_stats(self) -> Dict[str, Any]:
        return self.memoizer.get_stats()

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": PIPELINE_VERSION,
            "dag_nodes": len(self.dag.graph.nodes()),
            "dag_edges": len(self.dag.graph.edges()),
            "memoization_enabled": True,
            "homology_audit_enabled": self._homology_enabled(),
            "filtration_enabled": self._filtration_enabled(),
            "memoization_stats": self.get_memoization_stats(),
            "orientation_seed_ingested": self._orientation_seed is not None,
            "omega3": (self._last_omega3.to_dict()
                       if self._last_omega3 is not None else None),
            "policy": {
                "enforce_filtration": self._policy.enforce_filtration,
                "enforce_homology": self._policy.enforce_homology,
                "kam_restored": self._policy.kam_restored,
                "chaos_damped": self._policy.chaos_damped,
                "seed_ingested": self._policy.seed_ingested,
            },
        }


# ── 3.8 FACTORIES Y ENTRY POINT ──────────────────────────────────────────────

def create_director(
    config: Optional[Dict[str, Any]] = None,
    telemetry: Optional[TelemetryContext] = None,
    mic: Optional[MICRegistry] = None,
) -> PipelineDirector:
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    return PipelineDirector(config, telemetry, mic)


def create_director_from_env() -> PipelineDirector:
    config = {
        "session_dir": os.getenv("PIPELINE_SESSION_DIR", DEFAULT_SESSION_DIR),
        "enforce_filtration": os.getenv(
            "PIPELINE_ENFORCE_FILTRATION", "true").lower() == "true",
        "enforce_homology": os.getenv(
            "PIPELINE_ENFORCE_HOMOLOGY", "true").lower() == "true",
    }
    return create_director(config)


def process_all_files(
    presupuesto_path: Union[str, Path],
    apus_path:        Union[str, Path],
    insumos_path:     Union[str, Path],
    config:           Optional[Dict[str, Any]] = None,
    telemetry:        Optional[TelemetryContext] = None,
) -> Dict[str, Any]:
    """Entry point batch-mode (Fases 1→2→3 anidadas, veredicto Ω₃)."""
    config = config or {}
    telemetry = telemetry or TelemetryContext()
    paths = {
        "presupuesto_path": Path(presupuesto_path).resolve(),
        "apus_path":        Path(apus_path).resolve(),
        "insumos_path":     Path(insumos_path).resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            error = f"File not found: {name} = {path}"
            telemetry.record_error("process_all_files", error)
            raise FileNotFoundError(error)

    director = create_director(config, telemetry)
    try:
        return director.execute_pipeline({k: str(v) for k, v in paths.items()})
    except PipelineError as e:
        logger.critical("Pipeline failed: %s", e, exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error: %s", e, exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise PipelineError(f"Unexpected error: {e}") from e


if __name__ == "__main__":
    config = {"enforce_filtration": True, "enforce_homology": True}
    telemetry = TelemetryContext()
    result = process_all_files(
        presupuesto_path="data/presupuesto.xlsx",
        apus_path="data/apus.xlsx",
        insumos_path="data/insumos.xlsx",
        config=config, telemetry=telemetry)
    print("Pipeline completed successfully!")
    print(f"DataProduct with {len(result.get('payload', {}))} keys")
    omega = (result.get("metadata") or {}).get("omega3") or result.get("omega3_verdict")
    if omega:
        print(f"Ω₃ = {omega.get('status')}  score={omega.get('score')}")


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FIN DE FASE 3 (DECIDE & ACT)  —  CIERRE DEL MÓDULO v5.1.0              █
# █                                                                          █
# █   Funtor compuesto:                                                      █
# █     Obs  --project_to_phase_2-->  Orient                                 █
# █          --project_to_phase_3-->  Decide                                 █
# █          --emit_omega3_verdict--> DataProduct / Ω₃                       █
# █                                                                          █
# █   Concatenación: Fase 1 + Fase 2 + Fase 3 = pipeline_director.py         █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████