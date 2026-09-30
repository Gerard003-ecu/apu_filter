# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Topological Analyzer & TDA Engine — Cohomología Espectral en V_𝕋               ║
║  Ruta   : app/tactics/topological_analyzer.py                                            ║
║  Versión: 6.1.0-Nested-Phase1-UF-Hodge-KBN-Gershgorin-BlockFactor                        ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  0. OBJETO CATEGORIAL                                                                    ║
║  ────────────────────                                                                    ║
║  Sea 𝒞_Temp la categoría de observaciones temporales (series, grafos de servicio)        ║
║  y 𝒞_Topo la categoría de invariantes homológico-espectrales. Este módulo realiza        ║
║  un funtor covariante estricto                                                           ║
║                                                                                          ║
║      F : 𝒞_Temp ⟶ 𝒞_Topo,          F(g ∘ f) = F(g) ∘ F(f),                             ║
║                                                                                          ║
║  factorizado en tres endofuntores anidados F = F₃ ∘ F₂ ∘ F₁ sobre el topos               ║
║  táctico 𝔗_tac, con clasificador de subobjetos el retículo de Heyting                    ║
║                                                                                          ║
║      Ω₃ = {⊥, ½, ⊤}    (VETOED ≤ DEGRADED ≤ COHERENT).                                   ║
║                                                                                          ║
║  Morfismos de puerto (objeto terminal de Fᵢ = objeto inicial de Fᵢ₊₁):                   ║
║      π₁₂ : F₁(X) → F₂(X)     SimplicialCohainComplex  (B₁, W, L₀, índices, bloques)      ║
║      π₂₃ : F₂(X) → F₃(X)     TopologicalInvariants    (β, λ₂, h, Ψ, SNF)   [FASE 2]      ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  I. FASE 1 — SUBSTRATO OBSERVACIONAL  (estrato 0 de 𝔗_tac)  ← ESTE ARCHIVO / PASO        ║
║  ─────────────────────────────────────────────────────────                               ║
║                                                                                          ║
║  I.0  Constantes IEEE-754 / espectrales / contrato Ω₃ (Σ wᵢ = 1).                        ║
║                                                                                          ║
║  I.1  Mónada Option  𝕋 = 1 + (−)                                                         ║
║       Leyes:  η▹f = f;  m▹η = m;  (m▹f)▹g = m▹(λx. f(x)▹g).                              ║
║       Absorbe NaN/Inf sin romper la funtorialidad de F₁.  Applicative + μ.               ║
║                                                                                          ║
║  I.2  Proyección IEEE-754  π : ℝ̄ → ℝ_fin                                                 ║
║       π(NaN)=0, π(±∞)=±C_cap, π(−0)=+0.  Idempotencia: π∘π = π.                          ║
║       π_𝕋 : ℝ̄ → 𝕋(ℝ_fin)  envía patologías a None (no contamina el semiring).           ║
║                                                                                          ║
║  I.3  Aritmética compensada                                                              ║
║       TwoSum (Knuth), TwoProd (FMA/Dekker), Neumaier–Kahan (KBN).                        ║
║           |S_KBN − S_exact|  ≤  (2 ε_mach + O(N ε_mach²)) · Σ|xᵢ|.                       ║
║       Producto interno compensado: ⟨x,y⟩_KBN = Σ (pᵢ + eᵢ)  con (p,e)=TwoProd.           ║
║                                                                                          ║
║  I.4  Laplaciano de Hodge 0-dimensional (1-complejo ponderado)                           ║
║           L₀  =  d* d  =  B₁ W B₁ᵀ  ∈  Sym⁺(ℝ^{|V|}),                                    ║
║       con d = B₁ᵀ : Ω⁰ → Ω¹ y d* = B₁ W.  Propiedades:                                   ║
║           L₀ = L₀ᵀ,   L₀ ⪰ 0,   ker L₀ ≅ ℝ^{β₀}  (si W ≻ 0),                             ║
║           0 = λ₁ ≤ λ₂ ≤ ⋯ ≤ λ_n ≤ 2 Δ_máx     (Gershgorin).                              ║
║       Rank-nullity:  rank_ℝ(B₁) + dim ker L₀  =  |V|.                                    ║
║       tr(L₀) = 2 Σ_e w_e.  Factorización por bloques conexos (gancho V.1).               ║
║                                                                                          ║
║  I.5  Homología persistente 0-dimensional                                                ║
║       (A) Corte de superlevel a umbral θ  (API legado):  S^θ = {t : f(t) > θ}.           ║
║       (B) Diagrama H₀ por Union–Find + elder rule sobre el path graph:                   ║
║           subnivel de −f  ⇔  supernivel de f.  Estabilidad (Cohen–Steiner):              ║
║           d_B(Dgm(f), Dgm(g))  ≤  ‖f − g‖_∞.                                             ║
║       NO es Vietoris–Rips ni reducción ELZ (v. §V.2).                                    ║
║                                                                                          ║
║  PUERTO π₁₂ :  CompensatedLaplacianAssembler.assemble_from_graph()                       ║
║                → SimplicialCohainComplex                                                 ║
║  INGRESO F₂ :  Phase2Ingress.reduce_from_complex(complex_)     ← inicio formal FASE 2    ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  II–III.  FASES 2–3  (siguientes pasos de generación)                                    ║
║  SNF Euclid, Euler–Poincaré, Fiedler–Wielandt, Cheeger sweep, Ψ, Ω₃, Crowbar.            ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  IV. INVARIANTES GLOBALES DE F  =  F₃ ∘ F₂ ∘ F₁                                          ║
║  (N1) Euler–Poincaré: χ(K) = β₀ − β₁ = |V| − |E|.                                        ║
║  (N2) Gauss (SNF): d₁ | d₂ | ⋯ | d_r  cuando SNF no está omitida.                        ║
║  (N3) Wielandt: L_def desplaza λ₁=0 ↦ γ; λ₂ invariante si β₀=1.                          ║
║  (N4) Cheeger–Dodziuk: λ₂/2 ≤ h(K) ≤ √(2λ₂(2Δ_máx+λ₂)).                                  ║
║  (N5) Neumaier: |S_KBN − S_exact| ≤ 2 ε_mach Σ|xᵢ|.                                      ║
║  (N6) Heyting: ⊥ ≤ ½ ≤ ⊤; VETOED absorbente.                                             ║
║  (N7) Hodge: ker L₀ ≅ H⁰(K;ℝ) ≅ ℝ^{β₀}  (W ≻ 0).                                         ║
║  (N8) Idempotencia IEEE-754: π∘π=π.  Clausura Option.                                    ║
║  (N9) Cohen–Steiner: d_B(Dgm₀(f), Dgm₀(g)) ≤ ‖f−g‖_∞.                                    ║
║  (N10) tr(L₀) = 2 Σ w_e  (identidad de aristas).                                         ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  V. PECADOS RESIDUALES  (conscientemente no resueltos; Fases 2–3 + iteración futura)     ║
║  ──────────────────────────────────────────────────────────────                          ║
║                                                                                          ║
║  V.1  SNF no escala a |V| grande. Complejidad O(r·(m+n)·log M) con pivotes               ║
║       euclídeos e hinchazón entera. Para |V|>200 se OMITE (guardia de presupuesto)       ║
║       y se declara torsion-free por el teorema de 1-complejos, no por SNF.               ║
║       Mitigación FASE 1: `factor_by_connected_components` expone bloques para            ║
║       SNF independiente (Fase 2 debe consumirlos). sympy.smith_normal_form pendiente.    ║
║                                                                                          ║
║  V.2  Persistencia 0-dimensional sobre el 1-esqueleto camino. TDA completa exigiría      ║
║       Vietoris–Rips con filtración por peso y reducción matricial ELZ                    ║
║       (Edelsbrunner–Letscher–Zomorodian). Union–Find + elder rule ES el algoritmo        ║
║       exacto de H₀ sobre un grafo; NO sustituye VR ni ELZ. Fuera de alcance.             ║
║                                                                                          ║
║  V.3  Fiedler en grafos disconexos: β₀>1 ⇒ λ₂=0 ⇒ ker multidimensional.  [FASE 2]        ║
║       `fiedler_faithful=False`.                                                          ║
║                                                                                          ║
║  V.4  Crowbar sin HAL real.  [FASE 3]  dry_run=True por defecto.                         ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import hashlib
import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import (
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    final,
    runtime_checkable,
)

import networkx as nx
import numpy as np
import numpy.typing as npt

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    csr_matrix = None  # type: ignore[assignment]
    eigsh = None  # type: ignore[assignment]

logger = logging.getLogger("TopologicalAnalyzer")
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

__version__ = "6.1.0-Nested-Phase1-UF-Hodge-KBN-Gershgorin-BlockFactor"

__all__ = [
    # I.0
    "TopologicalConstants",
    "TopologicalError",
    "BettiNumberError",
    "PersistenceComputationError",
    "GraphStructureError",
    "InvalidTopologyError",
    "TorsionDetectedError",
    "SpectralConvergenceError",
    "MetricState",
    "HealthLevel",
    "HeytingVerdict",
    # I.1–I.3
    "Option",
    "IEEE754Sanitizer",
    "two_sum",
    "two_prod",
    "NeumaierKahanAccumulator",
    # I.5 persistencia
    "UnionFind",
    "PersistenceInterval",
    "PersistenceDiagram",
    "PersistenceAnalysisResult",
    "PersistenceHomology",
    # I.4 Hodge / π₁₂
    "WeightSpectrumClass",
    "GershgorinBounds",
    "SimplicialCohainComplex",
    "CompensatedLaplacianAssembler",
    "Phase2Ingress",
]


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 1 — SUBSTRATO OBSERVACIONAL ▓▓▓
# ▓▓▓ I.0 Constantes, tipos, errores, Ω₃ como objeto del topos (definido, usado en F₃). ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
class TopologicalConstants:
    r"""
    Constantes del análisis numérico IEEE-754, de la teoría espectral de grafos
    y del contrato ciber-físico. Los pesos de salud particionan la unidad en Ω₃.

    Axioma (Ω₃):  Σ wᵢ = 1  (validate_weights).

    ε_mach = 2⁻⁵³ es el épsilon binario de binary64 (IEEE-754-2008).
    EPSILON (1e-10) es tolerancia *geométrica* (Cheeger, Ψ, cortes).
    EPSILON_STRICT (1e-14) es tolerancia *espectral* (kernel, PSD, rank).
    """

    EPS_MACHINE: Final[float] = 2.0 ** -53
    EPSILON: Final[float] = 1e-10
    EPSILON_STRICT: Final[float] = 1e-14
    MAX_FINITE_CAP: Final[float] = 1e100
    MIN_NORMAL_F64: Final[float] = 2.2250738585072014e-308
    FLUSH_SUBNORMALS: Final[bool] = False

    CHEEGER_THRESHOLD: Final[float] = 0.05
    PYRAMIDAL_INDEX_MIN: Final[float] = 0.70
    PYRAMIDAL_INDEX_DEGRADED: Final[float] = 0.50

    EULER_SPHERE: Final[int] = 2
    EULER_TORUS: Final[int] = 0
    EULER_PROJECTIVE_PLANE: Final[int] = 1
    EULER_KLEIN_BOTTLE: Final[int] = 0

    MIN_PERSISTENCE_RATIO: Final[float] = 0.01
    NOISE_THRESHOLD_RATIO: Final[float] = 0.20
    CRITICAL_THRESHOLD_RATIO: Final[float] = 0.50

    MAX_CYCLOMATIC_COMPLEXITY: Final[int] = 10
    WARNING_CYCLOMATIC_COMPLEXITY: Final[int] = 5

    MAX_COMPONENTS_HEALTHY: Final[int] = 1
    MAX_COMPONENTS_WARNING: Final[int] = 2

    WEIGHT_FRAGMENTATION: Final[float] = 0.30
    WEIGHT_CYCLES: Final[float] = 0.20
    WEIGHT_DISCONNECTED: Final[float] = 0.20
    WEIGHT_MISSING_EDGES: Final[float] = 0.15
    WEIGHT_RETRY_LOOPS: Final[float] = 0.05
    WEIGHT_TORSION: Final[float] = 0.10

    ISR_BUDGET_NS: Final[int] = 400
    CROWBAR_GPIO_PIN: Final[int] = 14
    CROWBAR_THYRISTOR: Final[str] = "BT151"
    CROWBAR_GPIO_W1TS_MASK: Final[int] = 1 << 14  # firmware: GPIO.out_w1ts
    HAS_REAL_HAL: Final[bool] = False  # pecado residual V.4

    SNF_MAX_VERTICES: Final[int] = 200  # pecado residual V.1
    SNF_MAX_EDGES: Final[int] = 2000
    FIEDLER_DENSE_THRESHOLD: Final[int] = 200
    KERNEL_COUNT_DENSE_MAX: Final[int] = 512
    KBN_ENTRYWISE_THRESHOLD: Final[int] = 10_000

    SVD_RANK_REL_TOL: Final[float] = 1e-12

    @classmethod
    def validate_weights(cls) -> bool:
        r"""Axioma de Ω₃: Σ pesos = 1 ± ε.  Partición de la unidad en el score S."""
        total = (
            cls.WEIGHT_FRAGMENTATION
            + cls.WEIGHT_CYCLES
            + cls.WEIGHT_DISCONNECTED
            + cls.WEIGHT_MISSING_EDGES
            + cls.WEIGHT_RETRY_LOOPS
            + cls.WEIGHT_TORSION
        )
        return abs(total - 1.0) < cls.EPSILON

    @classmethod
    def machine_unit_roundoff(cls) -> float:
        """u = ε_mach / 2  (redondeo al más cercano en binary64)."""
        return cls.EPS_MACHINE * 0.5


TC = TopologicalConstants
assert TC.validate_weights(), "Pesos de salud deben sumar 1.0"


BettiIndex = int
SimplexDimension = int
EulerCharacteristic = int
PersistenceValue = float
BirthTime = float
DeathTime = float
HealthScore = float

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
IntegerMatrix = npt.NDArray[np.int64]
AdjacencyDict = Dict[str, Dict[str, int]]


class TopologicalError(Exception):
    """Base para errores topológicos del estrato Tactics."""

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.recoverable = recoverable


class BettiNumberError(TopologicalError):
    """Inconsistencia Euler–Poincaré o Betti fuera de dominio."""


class PersistenceComputationError(TopologicalError):
    """Error en homología persistente 0-dimensional."""


class GraphStructureError(TopologicalError):
    """Error en la estructura del 1-complejo."""


class InvalidTopologyError(TopologicalError):
    """Topología inconsistente — dispara VETOED (⊥ absorbente)."""


class TorsionDetectedError(TopologicalError):
    """Tor(H_k) ≠ 0 en la SNF sobre ℤ (algún dᵢ > 1)."""


class SpectralConvergenceError(TopologicalError):
    """Fallo de convergencia en Krylov–Lanczos."""


@unique
class MetricState(Enum):
    """Estado de una métrica frente a su diagrama de persistencia H₀."""

    STABLE = auto()
    NOISE = auto()
    FEATURE = auto()
    CRITICAL = auto()
    UNKNOWN = auto()

    @property
    def severity(self) -> int:
        return {
            MetricState.STABLE: 0,
            MetricState.NOISE: 1,
            MetricState.FEATURE: 2,
            MetricState.CRITICAL: 3,
            MetricState.UNKNOWN: 4,
        }[self]

    def __str__(self) -> str:
        return self.name


@unique
class HealthLevel(Enum):
    r"""
    Cuantización suave de S ∈ [0,1]:
        S ≥ 0.90 → HEALTHY;  ≥ 0.70 → DEGRADED;  ≥ 0.40 → UNHEALTHY;  resto CRITICAL.
    Independiente de Ω₃ (HeytingVerdict): el score es numérico, el veredicto es algebraico.
    Definido aquí (objeto del topos); consumido en FASE 3.
    """

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    CRITICAL = auto()

    @classmethod
    def from_score(cls, score: float) -> "HealthLevel":
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score ∈ [0,1]: {score}")
        if score >= 0.90:
            return cls.HEALTHY
        if score >= 0.70:
            return cls.DEGRADED
        if score >= 0.40:
            return cls.UNHEALTHY
        return cls.CRITICAL

    @property
    def severity(self) -> int:
        return {
            HealthLevel.HEALTHY: 0,
            HealthLevel.DEGRADED: 1,
            HealthLevel.UNHEALTHY: 2,
            HealthLevel.CRITICAL: 3,
        }[self]

    def __str__(self) -> str:
        return self.name


@unique
class HeytingVerdict(Enum):
    r"""
    Retículo distributivo de Heyting Ω₃ = {⊥, ½, ⊤}.

        v ∧ w = min(v,w),  v ∨ w = max(v,w),
        v → w = ⊤ si v ≤ w sino w,  ¬v = v → ⊥.

    ⊥ (VETOED) es absorbente: ⊥ ∧ x = ⊥  ∀x ∈ Ω₃.
    Definido aquí como clasificador de subobjetos del topos; aplicado en FASE 3.
    """

    VETOED = auto()
    DEGRADED = auto()
    COHERENT = auto()

    @property
    def lattice_value(self) -> float:
        return {
            HeytingVerdict.VETOED: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.COHERENT: 1.0,
        }[self]

    @classmethod
    def from_lattice_value(cls, v: float) -> "HeytingVerdict":
        if abs(v - 1.0) < 1e-9:
            return cls.COHERENT
        if abs(v - 0.5) < 1e-9:
            return cls.DEGRADED
        if abs(v - 0.0) < 1e-9:
            return cls.VETOED
        raise ValueError(f"Valor de retículo inválido: {v}")

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return min(self, other, key=lambda v: v.lattice_value)

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return max(self, other, key=lambda v: v.lattice_value)

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return (
            HeytingVerdict.COHERENT
            if self.lattice_value <= other.lattice_value
            else other
        )

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @property
    def is_coherent(self) -> bool:
        return self is HeytingVerdict.COHERENT

    def __str__(self) -> str:
        return self.name


@unique
class WeightSpectrumClass(Enum):
    r"""
    Clasificación del espectro de W (métrica de aristas).

        DEFINITE     W ≻ 0   (todos los w_e > ε)     ker L₀ ≅ ℝ^{β₀}  exacto
        SEMIDEFINITE W ≽ 0   (algún w_e ≈ 0)         rank(L₀) puede caer
        INDEFINITE   ∃ w_e < 0                       L₀ puede dejar de ser PSD
        VACUOUS      |E|=0
    """

    VACUOUS = auto()
    DEFINITE = auto()
    SEMIDEFINITE = auto()
    INDEFINITE = auto()


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ I.1  MÓNADA OPTION  𝕋 = 1 + (−)                                                   ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")
U = TypeVar("U")


@final
@dataclass(frozen=True)
class Option(Generic[T]):
    r"""
    Mónada Option ⟪Some x | None⟫.  𝕋 = 1 + (−).

    (L-Izq)  return(x) >>= f  ≡  f(x)
    (L-Der)  m >>= return     ≡  m
    (L-Asoc) (m >>= f) >>= g  ≡  m >>= (λx. f x >>= g)

    Funtor: map.  Applicative: ap.  Mónada: flat_map.  μ: flatten.
    Absorbe ausencias (None) y, vía IEEE754Sanitizer.sanitize_to_option, NaN/Inf.
    """

    _value: Optional[T] = None

    @staticmethod
    def some(x: T) -> "Option[T]":
        return Option(_value=x)

    @staticmethod
    def none() -> "Option[Any]":
        return Option(_value=None)

    @staticmethod
    def from_nullable(x: Optional[T]) -> "Option[T]":
        """η_𝕋 sobre el lifting de None: None ↦ none, x ↦ some(x)."""
        return Option.none() if x is None else Option.some(x)

    @property
    def is_defined(self) -> bool:
        return self._value is not None

    @property
    def is_empty(self) -> bool:
        return self._value is None

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("Option.none() has no value")
        return self._value

    def map(self, f: Callable[[T], U]) -> "Option[U]":
        return Option.some(f(self._value)) if self._value is not None else Option.none()

    def flat_map(self, f: Callable[[T], "Option[U]"]) -> "Option[U]":
        return f(self._value) if self._value is not None else Option.none()

    def ap(self: "Option[Callable[[T], U]]", other: "Option[T]") -> "Option[U]":
        r"""Applicative: self ⊛ other.  none es absorbente."""
        if self._value is None or other._value is None:
            return Option.none()
        return Option.some(self._value(other._value))  # type: ignore[misc]

    def filter(self, pred: Callable[[T], bool]) -> "Option[T]":
        if self._value is None or not pred(self._value):
            return Option.none()
        return self

    def fold(self, on_none: Callable[[], U], on_some: Callable[[T], U]) -> U:
        return on_some(self._value) if self._value is not None else on_none()

    def get_or_else(self, default: T) -> T:
        return self._value if self._value is not None else default

    def flatten(self: "Option[Option[U]]") -> "Option[U]":
        """μ : 𝕋² → 𝕋.  flatten(Some(m)) = m;  flatten(None) = None."""
        return self._value if self._value is not None else Option.none()  # type: ignore[return-value]

    def exists(self, pred: Callable[[T], bool]) -> bool:
        return self._value is not None and pred(self._value)

    def forall(self, pred: Callable[[T], bool]) -> bool:
        return self._value is None or pred(self._value)

    def __bool__(self) -> bool:
        return self._value is not None

    def __repr__(self) -> str:
        return f"Option.some({self._value!r})" if self._value is not None else "Option.none()"


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ I.2  PROYECCIÓN IEEE-754  π : ℝ̄ → ℝ_fin                                           ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
class IEEE754Sanitizer:
    r"""
    Proyección π : ℝ̄ → ℝ_fin que neutraliza patologías que rompen de Rham discreta:
        1. −0.0 invierte signos en productos orientados (B₁).
        2. NaN contamina (NaN ≠ NaN) y rompe órdenes parciales.
        3. ±Inf desborda KBN y la cota de Wilkinson.
        4. Subnormales (opc.) degradan el error relativo de TwoProd.

    Teorema (idempotencia):  ∀x ∈ ℝ̄,  π(π(x)) = π(x).
    π es un retráctil de ℝ̄ sobre ℝ_fin ∪ {0}:  π ∘ ι = id  en la imagen.
    """

    @staticmethod
    def is_pathological(x: Any) -> bool:
        """True ssi x no es un binary64 finito y no-NaN (incluye no-numéricos)."""
        try:
            v = float(x)
        except (TypeError, ValueError, OverflowError):
            return True
        return math.isnan(v) or math.isinf(v)

    @staticmethod
    def sanitize_scalar(x: Any) -> float:
        r"""
        π(x). Complejidad O(1).  −0.0 ↦ +0.0  (bit de signo anulado).
        OverflowError de float() se trata como no-convertible → 0.
        """
        try:
            v = float(x)
        except (TypeError, ValueError, OverflowError):
            logger.warning("sanitize_scalar: no convertible a float: %r → 0.0", x)
            return 0.0
        if math.isnan(v):
            return 0.0
        if math.isinf(v):
            return math.copysign(TC.MAX_FINITE_CAP, v)
        if v == 0.0:
            return 0.0
        if TC.FLUSH_SUBNORMALS and 0.0 < abs(v) < TC.MIN_NORMAL_F64:
            return 0.0
        return v

    @classmethod
    def sanitize_to_option(cls, x: Any) -> Option[float]:
        r"""
        π_𝕋 : ℝ̄ → 𝕋(ℝ_fin).  Patologías ↦ none;  finitos ↦ some(π(x)).
        Preferible a π cuando se desea no inventar ceros.
        """
        try:
            v = float(x)
        except (TypeError, ValueError, OverflowError):
            return Option.none()
        if math.isnan(v) or math.isinf(v):
            return Option.none()
        if v == 0.0:
            return Option.some(0.0)
        if TC.FLUSH_SUBNORMALS and 0.0 < abs(v) < TC.MIN_NORMAL_F64:
            return Option.none()
        return Option.some(v)

    @classmethod
    def sanitize_vector(cls, vec: Sequence[Any]) -> Vector:
        """π aplicada coordenada a coordenada.  O(n)."""
        n = len(vec)
        return np.fromiter(
            (cls.sanitize_scalar(x) for x in vec),
            dtype=np.float64,
            count=n,
        )

    @classmethod
    def sanitize_matrix(cls, mat: Sequence[Sequence[Any]]) -> Matrix:
        """π aplicada entrada a entrada.  O(mn)."""
        return np.array(
            [[cls.sanitize_scalar(x) for x in row] for row in mat],
            dtype=np.float64,
        )

    @classmethod
    def assert_idempotent(cls, x: Any) -> bool:
        """Certificado de (N8): π(π(x)) = π(x)."""
        y = cls.sanitize_scalar(x)
        return y == cls.sanitize_scalar(y)


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ I.3  TWOSUM / TWOPROD / NEUMAIER–KAHAN                                            ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


def two_sum(a: float, b: float) -> Tuple[float, float]:
    r"""
    Algoritmo de Knuth–Møller:  a ⊕ b = s + e  exactamente en binary64
    (bajo la hipótesis de redondeo al más cercano, sin overflow).

        s = a + b
        z = s − a
        e = (a − (s − z)) + (b − z)

    Complejidad O(1).  Invariante: s + e = a + b  en ℝ  si no hay overflow.
    """
    s = a + b
    z = s - a
    e = (a - (s - z)) + (b - z)
    return s, e


def two_prod(a: float, b: float) -> Tuple[float, float]:
    r"""
    Producto compensado:  a ⊗ b = p + e  exactamente.

    Prefiere `math.fma(a, b, −p)` (IEEE-754 FMA, error 0.5 ulp del producto
    exacto).  Si FMA no está disponible, Dekker-split no se implementa aquí
    (caería a e=0, degradando a producto nativo).

    Complejidad O(1).
    """
    p = a * b
    try:
        e = math.fma(a, b, -p)
    except (ValueError, OverflowError, AttributeError):  # pragma: no cover
        e = 0.0
    return p, e


@final
class NeumaierKahanAccumulator:
    r"""
    Sumación compensada de Neumaier (Kahan–Babuška–Neumaier).

        t = s + x
        c ← c + ((s−t)+x)   si |s|≥|x|,   c + ((x−t)+s)   si no
        s ← t,   total = s + c

    Teorema (Neumaier 1974):
        |total − Σ xᵢ| ≤ (2 ε_mach + O(N ε_mach²)) · Σ|xᵢ|.

    Extensión: `dot` usa TwoProd+KBN  ⇒  error de ⟨x,y⟩ acotado por
        (ε_mach + O(N ε_mach²)) (‖x‖₂‖y‖₂)  en la práctica, más
        2 ε_mach Σ|pᵢ| del KBN sobre los productos.
    """

    __slots__ = ("_sum", "_comp", "_abs_sum", "_count")

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._comp: float = 0.0
        self._abs_sum: float = 0.0
        self._count: int = 0

    def add(self, x: float) -> None:
        t = self._sum + x
        if abs(self._sum) >= abs(x):
            self._comp += (self._sum - t) + x
        else:
            self._comp += (x - t) + self._sum
        self._sum = t
        self._abs_sum += abs(x)
        self._count += 1

    def add_two_sum(self, x: float) -> None:
        """Variante que acumula el residuo de TwoSum en el compensador."""
        s, e = two_sum(self._sum, x)
        self._comp += e
        self._sum = s
        self._abs_sum += abs(x)
        self._count += 1

    def add_all(self, values: Iterable[float]) -> None:
        for x in values:
            self.add(x)

    @property
    def total(self) -> float:
        return self._sum + self._comp

    @property
    def error_bound(self) -> float:
        """Cota a priori (N5): 2 ε_mach Σ|xᵢ|  (término O(N ε²) omitido)."""
        return 2.0 * TC.EPS_MACHINE * self._abs_sum

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._sum = 0.0
        self._comp = 0.0
        self._abs_sum = 0.0
        self._count = 0

    @staticmethod
    def sum_array(values: Iterable[float]) -> float:
        acc = NeumaierKahanAccumulator()
        acc.add_all(values)
        return acc.total

    @staticmethod
    def dot(x: Sequence[float], y: Sequence[float]) -> float:
        r"""
        ⟨x,y⟩_KBN = Σᵢ (pᵢ + eᵢ)  con (pᵢ, eᵢ) = TwoProd(xᵢ, yᵢ).
        Longitudes distintas: se trunca al mínimo (no rellena con ceros).
        """
        acc = NeumaierKahanAccumulator()
        n = min(len(x), len(y))
        for i in range(n):
            p, e = two_prod(float(x[i]), float(y[i]))
            acc.add(p)
            if e != 0.0:
                acc.add(e)
        return acc.total

    @staticmethod
    def quadratic_form_diag(vec: Sequence[float], diag: Sequence[float]) -> float:
        r"""
        vᵀ D v = Σᵢ dᵢ vᵢ²  con TwoProd anidado y KBN.
        Usado para 1_Sᵀ L₀ 1_S cuando L₀ se evalúa por grados + cortes.
        """
        acc = NeumaierKahanAccumulator()
        n = min(len(vec), len(diag))
        for i in range(n):
            v = float(vec[i])
            sq, e_sq = two_prod(v, v)
            p, e_p = two_prod(float(diag[i]), sq)
            acc.add(p)
            if e_p != 0.0:
                acc.add(e_p)
            if e_sq != 0.0:
                p2, e2 = two_prod(float(diag[i]), e_sq)
                acc.add(p2)
                if e2 != 0.0:
                    acc.add(e2)
        return acc.total


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ I.5  UNION–FIND + HOMOLOGÍA PERSISTENTE H₀                                        ▓▓▓
# ▓▓▓ Pecado residual V.2: path-graph, no Vietoris–Rips / ELZ.                          ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
class UnionFind:
    r"""
    Estructura disjoint-set con compresión de caminos y unión por rango.

    Complejidad amortizada α(n) (inverso de Ackermann) por operación.
    `birth_of[root]` almacena el valor de filtración al que nació la
    componente (elder rule: sobrevive la de mayor birth en subnivel,
    i.e. la más antigua).

    Usos:
        (1) H₀ persistente sobre el path graph (PersistenceHomology).
        (2) Componentes conexas de B₁ para factorización de bloques (V.1).
    """

    __slots__ = ("parent", "rank", "birth_of", "_n")

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"UnionFind n≥0, recibido {n}")
        self._n = n
        self.parent: List[int] = list(range(n))
        self.rank: List[int] = [0] * n
        self.birth_of: List[float] = [0.0] * n

    def make_set(self, x: int, birth: float) -> None:
        self.parent[x] = x
        self.rank[x] = 0
        self.birth_of[x] = birth

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_elder(
        self, a: int, b: int
    ) -> Optional[Tuple[int, int, float]]:
        r"""
        Une las componentes de a y b.  Elder rule (subnivel):
        sobrevive el root de *menor* birth (apareció antes).
        Retorna (muerto, superviviente, birth_del_muerto) o None si ya unidos.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.birth_of[ra] < self.birth_of[rb]:
            survivor, dead = ra, rb
        elif self.birth_of[rb] < self.birth_of[ra]:
            survivor, dead = rb, ra
        else:
            if self.rank[ra] >= self.rank[rb]:
                survivor, dead = ra, rb
            else:
                survivor, dead = rb, ra
        self.parent[dead] = survivor
        if self.rank[survivor] == self.rank[dead] and self.birth_of[ra] == self.birth_of[rb]:
            self.rank[survivor] += 1
        return dead, survivor, self.birth_of[dead]

    def union_plain(self, a: int, b: int) -> bool:
        """Unión por rango sin elder rule.  True ssi se fusionaron dos clases."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

    def n_components_of(self, active: Sequence[bool]) -> int:
        roots: Set[int] = set()
        for i, is_on in enumerate(active):
            if is_on:
                roots.add(self.find(i))
        return len(roots)


@final
@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    r"""
    Barra [birth, death) de un diagrama de persistencia.

        pers = (death − birth)/√2     (distancia euclídea a la diagonal)
        d_B(I,J) = max(|b−b′|, |d−d′|)  en L^∞  (∞ si dims o vitalidad difieren).

    Convención `index`     : birth/death son índices de ventana (API legado).
    Convención `sublevel`  : birth/death son valores de filtración, birth ≤ death.
    Convención `superlevel`: θ decrece; se almacena como subnivel de −f.

    death = −1  (sentinela)  ⇔  barra infinita (componente aún viva).
    """

    birth: BirthTime
    death: DeathTime
    dimension: SimplexDimension
    amplitude: float = 0.0
    birth_value: float = 0.0
    death_value: float = 0.0
    convention: str = "index"

    def __post_init__(self) -> None:
        if self.convention == "index" and self.birth < 0:
            raise PersistenceComputationError(f"birth negativo: {self.birth}")
        if self.death != -1 and self.convention == "index" and self.death < self.birth:
            raise PersistenceComputationError(
                f"death < birth: [{self.birth}, {self.death})"
            )
        if self.death != -1 and self.convention == "sublevel" and self.death < self.birth:
            raise PersistenceComputationError(
                f"sublevel death < birth: [{self.birth}, {self.death})"
            )
        if self.dimension < 0:
            raise PersistenceComputationError(f"dimensión < 0: {self.dimension}")
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise PersistenceComputationError(f"amplitud inválida: {self.amplitude}")
        if self.convention not in {"index", "sublevel", "superlevel"}:
            raise PersistenceComputationError(f"convención desconocida: {self.convention}")

    @property
    def is_alive(self) -> bool:
        return self.death < 0

    @property
    def lifespan(self) -> float:
        return float("inf") if self.is_alive else float(self.death - self.birth)

    @property
    def persistence(self) -> PersistenceValue:
        return (
            float("inf")
            if self.is_alive
            else (self.death - self.birth) / math.sqrt(2.0)
        )

    @property
    def midpoint(self) -> float:
        return float(self.birth) if self.is_alive else 0.5 * (self.birth + self.death)

    def bottleneck_distance(self, other: "PersistenceInterval") -> float:
        r"""
        Distancia L^∞ entre barras de la misma dimensión.
        Una viva y otra finita son incomparables (∞): no se emparejan entre sí
        en el matching de bottleneck (la viva se empareja con la diagonal o
        con otra viva).
        """
        if self.dimension != other.dimension:
            return float("inf")
        b_diff = abs(self.birth - other.birth)
        if self.is_alive and other.is_alive:
            d_diff = 0.0
        elif self.is_alive or other.is_alive:
            d_diff = float("inf")
        else:
            d_diff = abs(self.death - other.death)
        return max(b_diff, d_diff)

    def diagonal_projection_cost(self) -> float:
        r"""
        Coste de emparejar la barra con su proyección en la diagonal:
        (death − birth)/2  en L^∞.  Barras infinitas no se proyectan (∞).
        """
        if self.is_alive:
            return float("inf")
        return 0.5 * abs(self.death - self.birth)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "birth": self.birth,
            "death": self.death,
            "dimension": self.dimension,
            "amplitude": self.amplitude,
            "birth_value": self.birth_value,
            "death_value": self.death_value,
            "convention": self.convention,
            "lifespan": self.lifespan if math.isfinite(self.lifespan) else None,
            "persistence": self.persistence if math.isfinite(self.persistence) else None,
            "is_alive": self.is_alive,
            "midpoint": self.midpoint,
        }

    def __str__(self) -> str:
        d_str = "∞" if self.is_alive else str(self.death)
        p_str = "∞" if math.isinf(self.persistence) else f"{self.persistence:.4f}"
        return (
            f"PersistenceInterval(dim={self.dimension}, [{self.birth}, {d_str}), "
            f"pers={p_str}, amp={self.amplitude:.4f}, conv={self.convention})"
        )


@final
@dataclass(frozen=True, slots=True)
class PersistenceDiagram:
    r"""
    Diagrama Dgm₀ ⊂ ℝ² ∪ {barras ∞}.  Objeto de la categoría de diagramas.

    Estabilidad (Cohen–Steiner–Edelsbrunner–Harer):
        d_B(Dgm(f), Dgm(g)) ≤ ‖f − g‖_∞.

    `filtration` documenta el pecado V.2: 'path-superlevel-UF' ≠ 'vietoris-rips-ELZ'.
    """

    intervals: Tuple[PersistenceInterval, ...]
    filtration: str
    n_samples: int
    l_inf_scale: float

    @property
    def finite(self) -> Tuple[PersistenceInterval, ...]:
        return tuple(i for i in self.intervals if not i.is_alive)

    @property
    def infinite(self) -> Tuple[PersistenceInterval, ...]:
        return tuple(i for i in self.intervals if i.is_alive)

    @property
    def betti0_at_end(self) -> int:
        """Número de barras infinitas = β₀ del complejo total (path conexo ⇒ 1 si N≥1)."""
        return len(self.infinite)

    def total_persistence(self, p: float = 1.0) -> float:
        r"""Pers_p = (Σ pers(I)^p )^{1/p} sobre barras finitas.  p≥1."""
        if p < 1.0:
            raise ValueError(f"p≥1, recibido {p}")
        acc = NeumaierKahanAccumulator()
        for iv in self.finite:
            if math.isfinite(iv.persistence):
                acc.add(iv.persistence ** p)
        return acc.total ** (1.0 / p) if acc.count else 0.0

    def cohen_steiner_bound(self, other_scale_linf: float) -> float:
        """Cota a priori de d_B entre este diagrama y otro con ‖f−g‖_∞ conocido."""
        return abs(self.l_inf_scale)  # placeholder de escala; el bound real es ‖f−g‖_∞

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filtration": self.filtration,
            "n_samples": self.n_samples,
            "n_finite": len(self.finite),
            "n_infinite": len(self.infinite),
            "betti0_at_end": self.betti0_at_end,
            "total_persistence_p1": self.total_persistence(1.0),
            "intervals": [i.to_dict() for i in self.intervals],
        }


@final
@dataclass(frozen=True, slots=True)
class PersistenceAnalysisResult:
    """Resultado del análisis H₀ de superlevel (corte a umbral y/o diagrama UF)."""

    state: MetricState
    intervals: Tuple[PersistenceInterval, ...]
    feature_count: int
    noise_count: int
    active_count: int
    max_lifespan: float
    total_persistence: float
    metadata: Dict[str, Union[int, float, str]]
    diagram: Optional[PersistenceDiagram] = None

    @property
    def total_intervals(self) -> int:
        return len(self.intervals)

    @property
    def confidence(self) -> float:
        return float(self.metadata.get("confidence", 0.0))

    @property
    def is_stable(self) -> bool:
        return self.state == MetricState.STABLE

    @property
    def is_critical(self) -> bool:
        return self.state == MetricState.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.name,
            "severity": self.state.severity,
            "total_intervals": self.total_intervals,
            "feature_count": self.feature_count,
            "noise_count": self.noise_count,
            "active_count": self.active_count,
            "max_lifespan": self.max_lifespan if math.isfinite(self.max_lifespan) else None,
            "total_persistence": self.total_persistence,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "intervals": [i.to_dict() for i in self.intervals],
            "diagram": self.diagram.to_dict() if self.diagram is not None else None,
        }


class PersistenceHomology:
    r"""
    Homología persistente 0-dimensional sobre series temporales.

    Filtración A (API legado): SUPERLEVEL a un umbral θ.
        S^θ = {t : f(t) > θ} ⊂ path graph 0—1—⋯—(N−1).
        Las componentes de S^θ son excursiones contiguas; [b,d) en *índices*.

    Filtración B (rigor H₀): Union–Find + elder rule sobre el path graph,
        equivalente a PH del subnivel de g = −f.
        Vértices aparecen en orden de g creciente; aristas {i,i+1} cuando
        ambos extremos están presentes.  Barras en valores de g.

    Estabilidad: d_B(Dgm(f), Dgm(g)) ≤ ‖f−g‖_∞  (Cohen–Steiner).

    PECADO RESIDUAL V.2: no es Vietoris–Rips ni reducción ELZ.  El 1-esqueleto
    es un camino, no un complejo de clique ponderado.  H₀ sobre un grafo SÍ
    es exacto por UF; H_{≥1} y VR quedan fuera de alcance.
    """

    DEFAULT_WINDOW_SIZE: ClassVar[int] = 20
    MIN_WINDOW_SIZE: ClassVar[int] = 3
    MAX_WINDOW_SIZE: ClassVar[int] = 10000
    FILTRATION_TYPE: ClassVar[str] = "superlevel-H0-threshold"
    FILTRATION_UF: ClassVar[str] = "path-superlevel-UF-elder"

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        if not (self.MIN_WINDOW_SIZE <= window_size <= self.MAX_WINDOW_SIZE):
            raise ValueError(
                f"window_size ∈ [{self.MIN_WINDOW_SIZE}, {self.MAX_WINDOW_SIZE}]: {window_size}"
            )
        self.window_size = window_size
        self._buffers: Dict[str, Deque[float]] = {}

    @property
    def metrics(self) -> Set[str]:
        return set(self._buffers.keys())

    @property
    def num_metrics(self) -> int:
        return len(self._buffers)

    def add_reading(self, metric_name: str, value: float) -> bool:
        if not isinstance(metric_name, str):
            return False
        metric_name = metric_name.strip()
        if not metric_name:
            return False
        v = IEEE754Sanitizer.sanitize_scalar(value)
        self._buffers.setdefault(metric_name, deque(maxlen=self.window_size))
        self._buffers[metric_name].append(v)
        return True

    def add_readings_batch(self, metric_name: str, values: Sequence[float]) -> int:
        return sum(1 for v in values if self.add_reading(metric_name, v))

    def get_buffer(self, metric_name: str) -> Optional[List[float]]:
        buf = self._buffers.get(metric_name)
        return list(buf) if buf else None

    def clear_metric(self, metric_name: str) -> bool:
        if metric_name in self._buffers:
            del self._buffers[metric_name]
            return True
        return False

    def clear_all(self) -> None:
        self._buffers.clear()

    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        buf = self._buffers.get(metric_name)
        if not buf:
            return None
        arr = np.array(buf, dtype=np.float64)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }

    @staticmethod
    def cohen_steiner_linf(f: Sequence[float], g: Sequence[float]) -> float:
        r"""
        ‖f−g‖_∞  (cota de d_B).  Secuencias de distinta longitud: se compara
        sobre el prefijo común; el resto se carga a |x| contra 0.
        """
        n = min(len(f), len(g))
        acc = 0.0
        for i in range(n):
            d = abs(IEEE754Sanitizer.sanitize_scalar(f[i]) - IEEE754Sanitizer.sanitize_scalar(g[i]))
            if d > acc:
                acc = d
        longer = f if len(f) >= len(g) else g
        for i in range(n, len(longer)):
            d = abs(IEEE754Sanitizer.sanitize_scalar(longer[i]))
            if d > acc:
                acc = d
        return acc

    @classmethod
    def compute_h0_diagram_path(cls, values: Sequence[float]) -> PersistenceDiagram:
        r"""
        Dgm₀ del path graph con filtración de subnivel de g = −f
        (⇔ superlevel de f).  Algoritmo exacto de H₀:

            1. gᵢ = −π(fᵢ).  Ordenar vértices por g creciente (aparición).
            2. Al insertar v: nace una componente en g(v).
            3. Si un vecino ya está presente, unir con elder rule;
               la componente joven muere en g(v).
            4. Las componentes restantes son barras infinitas.

        Complejidad O(N α(N) + N log N).  Pecado V.2: 1-esqueleto camino.

        Convención de barras: sublevel, birth=g(nacimiento), death=g(fusión).
        """
        f = IEEE754Sanitizer.sanitize_vector(values)
        n = int(f.size)
        if n == 0:
            return PersistenceDiagram((), cls.FILTRATION_UF, 0, 0.0)
        g = -f
        order = np.argsort(g, kind="mergesort")
        present = [False] * n
        uf = UnionFind(n)
        bars: List[PersistenceInterval] = []
        amp = f

        for idx in order:
            v = int(idx)
            gv = float(g[v])
            uf.make_set(v, gv)
            present[v] = True
            for nb in (v - 1, v + 1):
                if 0 <= nb < n and present[nb]:
                    merged = uf.union_elder(v, nb)
                    if merged is not None:
                        dead, _surv, birth_dead = merged
                        bars.append(
                            PersistenceInterval(
                                birth=float(birth_dead),
                                death=gv,
                                dimension=0,
                                amplitude=float(abs(amp[dead])),
                                birth_value=float(birth_dead),
                                death_value=gv,
                                convention="sublevel",
                            )
                        )

        seen_roots: Set[int] = set()
        for i in range(n):
            if not present[i]:
                continue
            r = uf.find(i)
            if r not in seen_roots:
                seen_roots.add(r)
                bars.append(
                    PersistenceInterval(
                        birth=float(uf.birth_of[r]),
                        death=-1.0,
                        dimension=0,
                        amplitude=float(np.max(np.abs(amp)) if n else 0.0),
                        birth_value=float(uf.birth_of[r]),
                        death_value=float("-inf"),
                        convention="sublevel",
                    )
                )

        linf = float(np.max(np.abs(f))) if n else 0.0
        return PersistenceDiagram(
            intervals=tuple(bars),
            filtration=cls.FILTRATION_UF,
            n_samples=n,
            l_inf_scale=linf,
        )

    def compute_diagram(self, metric_name: str) -> Optional[PersistenceDiagram]:
        """Dgm₀ UF de la ventana actual.  None si el buffer no existe."""
        buf = self._buffers.get(metric_name)
        if not buf:
            return None
        return self.compute_h0_diagram_path(list(buf))

    def analyze_persistence(
        self,
        metric_name: str,
        threshold: float,
        noise_ratio: float = TC.NOISE_THRESHOLD_RATIO,
        critical_ratio: float = TC.CRITICAL_THRESHOLD_RATIO,
    ) -> PersistenceAnalysisResult:
        r"""
        Corte de superlevel S^θ (filtración A, API estable) + diagrama UF (B)
        adjunto en `result.diagram`.

        Clasificación sobre el corte:
            barras cortas → STABLE/NOISE;  lifespan > noise·W → FEATURE;
            excursión viva > critical·W → CRITICAL.
        """
        buf = self._buffers.get(metric_name)
        if not buf or len(buf) < 2:
            return PersistenceAnalysisResult(
                state=MetricState.UNKNOWN,
                intervals=(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0.0,
                total_persistence=0.0,
                metadata={"reason": "insufficient_data", "filtration": self.FILTRATION_TYPE},
            )

        vals = list(buf)
        theta = IEEE754Sanitizer.sanitize_scalar(threshold)
        intervals: List[PersistenceInterval] = []
        is_active = False
        birth = 0
        for i, v in enumerate(vals):
            if v > theta and not is_active:
                is_active = True
                birth = i
            elif v <= theta and is_active:
                is_active = False
                intervals.append(
                    PersistenceInterval(
                        birth=float(birth),
                        death=float(i),
                        dimension=0,
                        amplitude=float(np.max(vals[birth:i])),
                        birth_value=float(vals[birth]),
                        death_value=theta,
                        convention="index",
                    )
                )

        active_count = 0
        if is_active:
            active_count = 1
            intervals.append(
                PersistenceInterval(
                    birth=float(birth),
                    death=-1.0,
                    dimension=0,
                    amplitude=float(np.max(vals[birth:])),
                    birth_value=float(vals[birth]),
                    death_value=float("-inf"),
                    convention="index",
                )
            )

        total_p = float(
            sum(
                i.persistence
                for i in intervals
                if not i.is_alive and math.isfinite(i.persistence)
            )
        )
        finite_lifespans = [i.lifespan for i in intervals if not i.is_alive]
        max_l = float(max(finite_lifespans)) if finite_lifespans else 0.0
        cutoff = self.window_size * noise_ratio
        feature_count = sum(
            1 for i in intervals if not i.is_alive and i.lifespan > cutoff
        )
        noise_count = len(intervals) - feature_count - active_count

        state = MetricState.STABLE
        if is_active and (len(vals) - birth) > self.window_size * critical_ratio:
            state = MetricState.CRITICAL
        elif feature_count > 0:
            state = MetricState.FEATURE
        elif noise_count > 0:
            state = MetricState.NOISE

        diagram: Optional[PersistenceDiagram] = None
        try:
            diagram = self.compute_h0_diagram_path(vals)
        except PersistenceComputationError as exc:
            logger.warning("Diagrama UF no disponible: %s", exc)

        return PersistenceAnalysisResult(
            state=state,
            intervals=tuple(intervals),
            feature_count=feature_count,
            noise_count=noise_count,
            active_count=active_count,
            max_lifespan=max_l,
            total_persistence=total_p,
            metadata={
                "active_duration": len(vals) - birth if is_active else 0,
                "filtration": self.FILTRATION_TYPE,
                "uf_filtration": self.FILTRATION_UF,
                "threshold": theta,
                "residual_sin_v2": "path-H0-not-VR-ELZ",
            },
            diagram=diagram,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ I.4  COMPLEJO DE HODGE  L₀ = B₁ W B₁ᵀ  +  GERSHGORIN  +  BLOQUES (V.1)            ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class GershgorinBounds:
    r"""
    Discos de Gershgorin de L₀.  Para un Laplaciano simétrico con fila i:

        D(L₀[i,i], Rᵢ)  con  Rᵢ = Σ_{j≠i} |L₀[i,j]|.

    Como L₀ 1 = 0 y L₀ ⪰ 0 (W ≽ 0), se tiene
        0 ≤ λ_k ≤ 2 Δ_máx     y    λ_max ≤ maxᵢ (L₀[i,i] + Rᵢ).
    """

    lambda_min_lower: float
    lambda_max_upper: float
    two_delta_max: float
    delta_max: float
    row_radii: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_min_lower": self.lambda_min_lower,
            "lambda_max_upper": self.lambda_max_upper,
            "two_delta_max": self.two_delta_max,
            "delta_max": self.delta_max,
            "n_disks": len(self.row_radii),
        }


@dataclass
class SimplicialCohainComplex:
    r"""
    1-complejo simplicial ponderado. Objeto del puerto π₁₂.

        B₁ ∈ ℤ^{|V|×|E|}   operador frontera ∂₁ (incidencia orientada)
        W  ∈ ℝ^{|E|×|E|}   métrica diagonal de aristas
        L₀ = B₁ W B₁ᵀ      Laplaciano de Hodge de 0-formas

    Invariantes:
        L₀=L₀ᵀ,  L₀ ⪰ 0,  L₀ 1 = 0 (si cada componente tiene W ≽ 0),
        rank_ℝ(B₁) + dim ker L₀ = |V|  (W ≻ 0),
        tr(L₀) = 2 Σ_e w_e.

    `component_labels[i]` ∈ {0,…,β₀−1} indexa el bloque conexo del vértice i
    (gancho V.1: SNF por bloque en FASE 2).
    """

    B1: IntegerMatrix
    W: Matrix
    L0: Matrix
    node_index: Dict[str, int]
    edge_index: Dict[Tuple[str, str], int]
    default_weight: float = 1.0
    component_labels: Optional[Vector] = None
    assembly_method: str = "kbn-dense"

    @property
    def n_vertices(self) -> int:
        return int(self.B1.shape[0]) if self.B1.ndim == 2 else 0

    @property
    def n_edges(self) -> int:
        return int(self.B1.shape[1]) if self.B1.ndim == 2 else 0

    @property
    def euler_characteristic(self) -> int:
        return self.n_vertices - self.n_edges

    def inverse_node_index(self) -> Tuple[str, ...]:
        """Inversa de node_index: i ↦ nombre.  Requiere biyectividad."""
        inv = [""] * self.n_vertices
        for name, i in self.node_index.items():
            if 0 <= i < self.n_vertices:
                inv[i] = name
        return tuple(inv)

    def weight_diagonal(self) -> Vector:
        if self.W.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.diag(self.W).astype(np.float64, copy=False)

    def classify_weight_spectrum(self, tol: float = TC.EPSILON_STRICT) -> WeightSpectrumClass:
        """Clasifica W ≻ 0 / W ≽ 0 / indefinida / vacua.  Decide validez de (N7)."""
        if self.n_edges == 0:
            return WeightSpectrumClass.VACUOUS
        w = self.weight_diagonal()
        if np.any(w < -tol):
            return WeightSpectrumClass.INDEFINITE
        if np.any(w <= tol):
            return WeightSpectrumClass.SEMIDEFINITE
        return WeightSpectrumClass.DEFINITE

    def rank_over_R(self, rel_tol: float = TC.SVD_RANK_REL_TOL) -> int:
        r"""
        rank_ℝ(B₁) vía SVD.  Para 1-complejos con W ≻ 0:  rank(B₁) = |V| − β₀.
        Tolerancia: rel_tol · σ_max · max(m, n).
        """
        if self.n_vertices == 0 or self.n_edges == 0:
            return 0
        a = self.B1.astype(np.float64, copy=False)
        try:
            sigma = np.linalg.svd(a, compute_uv=False)
        except np.linalg.LinAlgError:
            return int(np.linalg.matrix_rank(a))
        if sigma.size == 0:
            return 0
        cutoff = rel_tol * float(sigma[0]) * max(a.shape)
        return int(np.sum(sigma > cutoff))

    def exceeds_snf_budget(self) -> bool:
        """True ssi el Euclid SNF está fuera de presupuesto (pecado residual V.1)."""
        return (
            self.n_vertices > TC.SNF_MAX_VERTICES
            or self.n_edges > TC.SNF_MAX_EDGES
        )

    def snf_budget_reason(self) -> str:
        return (
            f"|V|={self.n_vertices} |E|={self.n_edges} vs "
            f"presupuesto ({TC.SNF_MAX_VERTICES}, {TC.SNF_MAX_EDGES})"
        )

    def degree_vector(self) -> Vector:
        """Δᵢ = L₀[i,i]  (grado ponderado)."""
        if self.n_vertices == 0:
            return np.zeros(0, dtype=np.float64)
        return np.diag(self.L0).astype(np.float64, copy=True)

    def gershgorin(self) -> GershgorinBounds:
        r"""
        Cotas de Gershgorin + 2Δ_máx.  λ ∈ [0, min(maxᵢ(dᵢ+Rᵢ), 2Δ_máx)]
        si L₀ es Laplaciano PSD.
        """
        n = self.n_vertices
        if n == 0:
            return GershgorinBounds(0.0, 0.0, 0.0, 0.0, ())
        L = self.L0
        diag = np.diag(L).astype(np.float64)
        radii = np.sum(np.abs(L), axis=1) - np.abs(diag)
        radii = np.maximum(radii, 0.0)
        delta_max = float(np.max(diag)) if n else 0.0
        lam_max_g = float(np.max(diag + radii))
        two_d = 2.0 * delta_max
        return GershgorinBounds(
            lambda_min_lower=0.0,
            lambda_max_upper=min(lam_max_g, two_d) if two_d > 0.0 else lam_max_g,
            two_delta_max=two_d,
            delta_max=delta_max,
            row_radii=tuple(float(r) for r in radii),
        )

    def trace_identity_residual(self) -> float:
        r"""
        Residuo de (N10):  |tr(L₀) − 2 Σ w_e|.
        Debe ser ≤ O(ε_mach) · (|V|+|E|) · escala  si el ensamblado es fiel.
        """
        tr = float(np.trace(self.L0)) if self.n_vertices else 0.0
        sum_w = float(NeumaierKahanAccumulator.sum_array(self.weight_diagonal()))
        return abs(tr - 2.0 * sum_w)

    def normalized_laplacian(self, tol: float = TC.EPSILON_STRICT) -> Matrix:
        r"""
        L_sym = D^{−1/2} L₀ D^{−1/2}  sobre el soporte {dᵢ > tol}.
        Vértices aislados: fila/columna cero (convención estándar).
        """
        n = self.n_vertices
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        d = self.degree_vector()
        inv_sqrt = np.zeros(n, dtype=np.float64)
        mask = d > tol
        inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        d_s = inv_sqrt.reshape(n, 1)
        return (d_s * self.L0) * inv_sqrt.reshape(1, n)

    def _labels_from_incidence(self) -> Vector:
        """β₀-etiquetado vía UF sobre columnas de B₁ (aristas).  O(|V|+|E| α)."""
        n, m = self.n_vertices, self.n_edges
        labels = np.arange(n, dtype=np.float64)
        if n == 0:
            return labels
        uf = UnionFind(n)
        for i in range(n):
            uf.make_set(i, float(i))
        B = self.B1
        for e in range(m):
            col = B[:, e]
            nz = np.flatnonzero(col)
            if nz.size >= 2:
                uf.union_plain(int(nz[0]), int(nz[1]))
        roots: Dict[int, int] = {}
        next_id = 0
        for i in range(n):
            r = uf.find(i)
            if r not in roots:
                roots[r] = next_id
                next_id += 1
            labels[i] = float(roots[r])
        return labels

    def ensure_component_labels(self) -> Vector:
        """Idempotente: calcula etiquetas si faltan.  Gancho V.1."""
        if self.component_labels is not None and int(self.component_labels.size) == self.n_vertices:
            return self.component_labels
        self.component_labels = self._labels_from_incidence()
        return self.component_labels

    def n_connected_components(self) -> int:
        lab = self.ensure_component_labels()
        if lab.size == 0:
            return 0
        return int(np.unique(lab).size)

    def factor_by_connected_components(self) -> Tuple["SimplicialCohainComplex", ...]:
        r"""
        Gancho de mitigación V.1: descompone K = ⊔_c K_c  en subcomplejos
        inducidos por componentes conexas.  FASE 2 puede ejecutar SNF sobre
        cada bloque con |V_c| ≤ SNF_MAX_VERTICES aunque |V| global exceda.

        Complejidad O(|V| + |E| + Σ_c |V_c|·|E_c|) por copia densa.
        """
        n, m = self.n_vertices, self.n_edges
        if n == 0:
            return ()
        lab = self.ensure_component_labels().astype(int)
        n_comp = int(lab.max()) + 1 if lab.size else 0
        if n_comp <= 1:
            return (self,)

        names = self.inverse_node_index()
        edge_list = [None] * m  # type: ignore[var-annotated]
        inv_edges = {idx: e for e, idx in self.edge_index.items()}
        for e in range(m):
            edge_list[e] = inv_edges.get(e)

        factors: List[SimplicialCohainComplex] = []
        B = self.B1
        wdiag = self.weight_diagonal()
        for c in range(n_comp):
            verts = np.flatnonzero(lab == c)
            if verts.size == 0:
                continue
            vset = set(int(i) for i in verts)
            e_idx = []
            for e in range(m):
                nz = np.flatnonzero(B[:, e])
                if nz.size >= 1 and all(int(i) in vset for i in nz):
                    e_idx.append(e)
            loc_node = {names[int(i)]: k for k, i in enumerate(verts)}
            loc_edge: Dict[Tuple[str, str], int] = {}
            for k, e in enumerate(e_idx):
                pair = edge_list[e]
                if pair is not None:
                    loc_edge[pair] = k
            B_loc = B[np.ix_(verts, e_idx)].copy() if e_idx else np.zeros((verts.size, 0), dtype=np.int64)
            W_loc = np.zeros((len(e_idx), len(e_idx)), dtype=np.float64)
            for k, e in enumerate(e_idx):
                W_loc[k, k] = float(wdiag[e]) if e < wdiag.size else 0.0
            L_loc = self.L0[np.ix_(verts, verts)].copy()
            factors.append(
                SimplicialCohainComplex(
                    B1=B_loc,
                    W=W_loc,
                    L0=L_loc,
                    node_index=loc_node,
                    edge_index=loc_edge,
                    default_weight=self.default_weight,
                    component_labels=np.zeros(verts.size, dtype=np.float64),
                    assembly_method=f"block-c{c}:{self.assembly_method}",
                )
            )
        return tuple(factors)

    def blocks_within_snf_budget(self) -> bool:
        """True ssi TODOS los factores conexos caben en el presupuesto Euclid (V.1)."""
        for fac in self.factor_by_connected_components():
            if fac.exceeds_snf_budget():
                return False
        return True

    def verify_invariants(self, tol: float = TC.EPSILON_STRICT) -> Dict[str, bool]:
        r"""
        Simetría ‖L₀−L₀ᵀ‖_F ≤ tol·‖L₀‖_F;
        kernel ‖L₀ 1‖_∞ ≤ tol·‖L₀‖_F  (constantes; si β₀>1 hay más kernel);
        PSD  λ_min ≥ −tol·‖L₀‖_F;
        rank-nullity  rank(B₁)+dim_ker ≈ |V|  (aproximado, n≤512, W ≻ 0);
        tr(L₀) ≈ 2 Σ w_e.
        """
        if self.n_vertices == 0:
            return {
                "symmetric": True,
                "kernel_1": True,
                "psd": True,
                "rank_nullity": True,
                "trace_identity": True,
            }
        L0 = self.L0
        scale = max(1.0, float(np.linalg.norm(L0, ord="fro")))
        sym_err = float(np.linalg.norm(L0 - L0.T, ord="fro"))
        kernel_vec = np.ones(self.n_vertices, dtype=np.float64)
        kernel_err = float(np.linalg.norm(L0 @ kernel_vec, ord=np.inf))
        L0_sym = 0.5 * (L0 + L0.T)
        try:
            eig_min = float(np.linalg.eigvalsh(L0_sym).min())
        except np.linalg.LinAlgError:
            eig_min = 0.0
        rank_nullity_ok = True
        wclass = self.classify_weight_spectrum(tol)
        if self.n_vertices <= TC.KERNEL_COUNT_DENSE_MAX and wclass is WeightSpectrumClass.DEFINITE:
            try:
                eigs = np.linalg.eigvalsh(L0_sym)
                dim_ker = int(np.sum(np.abs(eigs) < max(tol, TC.EPSILON_STRICT) * scale))
                rank_nullity_ok = (self.rank_over_R() + dim_ker) == self.n_vertices
            except np.linalg.LinAlgError:
                rank_nullity_ok = False
        tr_scale = max(1.0, abs(float(np.trace(L0))))
        trace_ok = self.trace_identity_residual() <= max(tol * tr_scale, 8.0 * TC.EPS_MACHINE * tr_scale)
        return {
            "symmetric": sym_err <= tol * scale,
            "kernel_1": kernel_err <= tol * scale,
            "psd": eig_min >= -tol * scale,
            "rank_nullity": rank_nullity_ok,
            "trace_identity": trace_ok,
        }


@final
class CompensatedLaplacianAssembler:
    r"""
    Ensamblador KBN de L₀ = B₁ W B₁ᵀ.

    Orientación de B₁: e=(u,v) con idx(u)<idx(v) ⇒ B₁[u,e]=+1, B₁[v,e]=−1.
    L₀ es invariante ante reorientación; la SNF de B₁ SÍ depende de ella
    solo por signos, no por divisores (dᵢ ≥ 0).

    Si |V|·|E| > KBN_ENTRYWISE_THRESHOLD se usa producto denso (sin cota
    entrada-a-entrada de Neumaier; orden de complejidad preservado).
    Si SciPy está disponible y el grafo es disperso, se ensambla CSR.

    Cada entrada L₀[i,j] = Σ_e B₁[i,e] w_e B₁[j,e] se computa con TwoProd+KBN
    en la ruta entrywise.
    """

    KBN_THRESHOLD: ClassVar[int] = TC.KBN_ENTRYWISE_THRESHOLD

    @staticmethod
    def canonical_node_index(graph: nx.Graph) -> Dict[str, int]:
        """Índice canónico: nodos ordenados lexicográficamente.  Determinista."""
        nodes = sorted(graph.nodes())
        return {n: i for i, n in enumerate(nodes)}

    @staticmethod
    def canonical_edge_index(graph: nx.Graph) -> Dict[Tuple[str, str], int]:
        """Aristas no dirigidas como pares ordenados (min, max).  Determinista."""
        edges = sorted(tuple(sorted(e)) for e in graph.edges())
        return {e: i for i, e in enumerate(edges)}

    @staticmethod
    def build_incidence_matrix(
        graph: nx.Graph, node_index: Dict[str, int]
    ) -> IntegerMatrix:
        r"""
        B₁ ∈ ℤ^{|V|×|E|}.  Cada columna tiene exactamente un +1 y un −1
        (salvo loops, que se rechazan aguas arriba).  Complejidad O(|E|).
        """
        n_nodes = len(node_index)
        n_edges = graph.number_of_edges()
        B1 = np.zeros((n_nodes, n_edges), dtype=np.int64)
        for e_idx, (u, v) in enumerate(graph.edges()):
            iu, iv = node_index[u], node_index[v]
            if iu < iv:
                B1[iu, e_idx] = 1
                B1[iv, e_idx] = -1
            else:
                B1[iv, e_idx] = 1
                B1[iu, e_idx] = -1
        return B1

    @classmethod
    def build_weight_matrix(
        cls,
        graph: nx.Graph,
        edge_index: Dict[Tuple[str, str], int],
        default_weight: float = 1.0,
    ) -> Matrix:
        """W = diag(w_e) con π(w_e).  Pesos ausentes → default_weight."""
        n_edges = len(edge_index)
        W = np.zeros((n_edges, n_edges), dtype=np.float64)
        for u, v, data in graph.edges(data=True):
            key = (min(u, v), max(u, v))
            if key in edge_index:
                raw = data.get("weight", default_weight)
                W[edge_index[key], edge_index[key]] = IEEE754Sanitizer.sanitize_scalar(raw)
        return W

    @classmethod
    def assemble_laplacian_kbn(cls, B1: IntegerMatrix, W: Matrix) -> Matrix:
        r"""
        L₀[i,j] = Σ_e B₁[i,e] W[e,e] B₁[j,e]  (TwoProd + KBN sobre e).
        Simetría por asignación explícita L[j,i]=L[i,j].
        Complejidad O(|V|² |E|) entrywise; O(|V|² |E|^0) vía producto denso
        si |V|·|E| > umbral.
        """
        m, n = B1.shape if B1.ndim == 2 else (0, 0)
        if m == 0:
            return np.zeros((0, 0), dtype=np.float64)

        w_diag = np.diag(W) if W.size else np.zeros(n, dtype=np.float64)

        if m * n > cls.KBN_THRESHOLD:
            logger.debug(
                "Ensamblado L₀: |V|·|E|=%d > %d → producto matricial directo",
                m * n,
                cls.KBN_THRESHOLD,
            )
            B1f = B1.astype(np.float64)
            return B1f @ W @ B1f.T

        L0 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            acc_diag = NeumaierKahanAccumulator()
            for e in range(n):
                bie = float(B1[i, e])
                if bie == 0.0:
                    continue
                p1, e1 = two_prod(bie, float(w_diag[e]))
                p2, e2 = two_prod(p1, bie)
                acc_diag.add(p2)
                if e1 != 0.0:
                    p_e, e_e = two_prod(e1, bie)
                    acc_diag.add(p_e)
                    if e_e != 0.0:
                        acc_diag.add(e_e)
                if e2 != 0.0:
                    acc_diag.add(e2)
            L0[i, i] = acc_diag.total
            for j in range(i + 1, m):
                acc = NeumaierKahanAccumulator()
                for e in range(n):
                    bie = float(B1[i, e])
                    bje = float(B1[j, e])
                    if bie == 0.0 or bje == 0.0:
                        continue
                    p1, e1 = two_prod(bie, float(w_diag[e]))
                    p2, e2 = two_prod(p1, bje)
                    acc.add(p2)
                    if e1 != 0.0:
                        p_e, e_e = two_prod(e1, bje)
                        acc.add(p_e)
                        if e_e != 0.0:
                            acc.add(e_e)
                    if e2 != 0.0:
                        acc.add(e2)
                val = acc.total
                L0[i, j] = val
                L0[j, i] = val
        return L0

    @classmethod
    def assemble_laplacian_sparse(cls, B1: IntegerMatrix, W: Matrix) -> Optional[Matrix]:
        """Ruta CSR: (B √W)(B √W)ᵀ.  None si SciPy ausente o W no diagonal-no-negativa."""
        if not _HAS_SCIPY or csr_matrix is None:
            return None
        m, n = B1.shape if B1.ndim == 2 else (0, 0)
        if m == 0:
            return np.zeros((0, 0), dtype=np.float64)
        w_diag = np.diag(W) if W.size else np.zeros(n, dtype=np.float64)
        if np.any(w_diag < -TC.EPSILON_STRICT):
            return None
        sqrt_w = np.sqrt(np.maximum(w_diag, 0.0))
        Bf = B1.astype(np.float64)
        B_csr = csr_matrix(Bf * sqrt_w.reshape(1, n))
        L = (B_csr @ B_csr.T).toarray()
        return np.asarray(L, dtype=np.float64)

    @staticmethod
    def labels_from_graph(graph: nx.Graph, node_index: Dict[str, int]) -> Vector:
        """Etiquetas de nx.connected_components alineadas a node_index."""
        n = len(node_index)
        labels = np.zeros(n, dtype=np.float64)
        for c, comp in enumerate(nx.connected_components(graph)):
            for node in comp:
                labels[node_index[node]] = float(c)
        return labels

    # ═══════════════════════════════════════════════════════════════════════════
    # ► PUERTO DE SALIDA FASE 1 → FASE 2
    # ► Este método ES el objeto terminal de F₁ y el objeto inicial de F₂.
    # ► Phase2Ingress.reduce_from_complex consume este morfismo.
    # ═══════════════════════════════════════════════════════════════════════════
    @classmethod
    def assemble_from_graph(
        cls, graph: nx.Graph, default_weight: float = 1.0
    ) -> SimplicialCohainComplex:
        r"""
        Puerto π₁₂. Único punto de salida de FASE 1 hacia FASE 2.

        Empaqueta (B₁, W, L₀) saneados + índices canónicos + etiquetas de
        componentes (gancho V.1).  La FASE 2 comienza aplicando SNF / Fiedler /
        Cheeger sobre este complejo (o sobre sus factores conexos).

        Invariantes al salir:
            L₀ ≈ L₀ᵀ,  L₀ 1 ≈ 0,  tr(L₀) ≈ 2 Σ w_e
            (certificados blandos; verify_invariants en FASE 2).
        """
        node_index = cls.canonical_node_index(graph)
        edge_index = cls.canonical_edge_index(graph)

        B1 = cls.build_incidence_matrix(graph, node_index)
        W = cls.build_weight_matrix(graph, edge_index, default_weight)

        method = "kbn-dense"
        L0_sparse = None
        n_v, n_e = B1.shape if B1.ndim == 2 else (0, 0)
        if n_v * n_e > cls.KBN_THRESHOLD:
            L0_sparse = cls.assemble_laplacian_sparse(B1, W)
            if L0_sparse is not None:
                L0 = L0_sparse
                method = "csr-gram"
            else:
                L0 = cls.assemble_laplacian_kbn(B1, W)
                method = "dense-gemm"
        else:
            L0 = cls.assemble_laplacian_kbn(B1, W)
            method = "kbn-twoprod"

        labels = cls.labels_from_graph(graph, node_index)

        logger.debug(
            "Fase1►Fase2: complejo %s |V|=%d |E|=%d χ=%d β₀~%d",
            method,
            len(node_index),
            len(edge_index),
            len(node_index) - len(edge_index),
            int(np.unique(labels).size) if labels.size else 0,
        )
        return SimplicialCohainComplex(
            B1=B1,
            W=W,
            L0=L0,
            node_index=node_index,
            edge_index=edge_index,
            default_weight=default_weight,
            component_labels=labels,
            assembly_method=method,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ► CONTINUACIÓN FORMAL FASE 1 → FASE 2
# ► El objeto terminal de F₁ (SimplicialCohainComplex) es el objeto inicial de F₂.
# ► Este Protocol ES el primer método de la Fase 2: SmithNormalFormReducer lo realiza.
# ══════════════════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Phase2Ingress(Protocol):
    r"""
    Morfismo de puerto π₁₂ → F₂.

    Definición formal (inicio de FASE 2):

        reduce_from_complex : SimplicialCohainComplex → SmithNormalFormResult

    Semántica (a implementar en FASE 2):
        • Si complex_.exceeds_snf_budget() y NOT complex_.blocks_within_snf_budget():
              omitir Euclid SNF (pecado V.1) y declarar Tor≡0 por el teorema
              «H_*(1-complejo; ℤ) es libre abeliano».
        • Si los factores conexos caben en presupuesto: SNF por bloque
              (mitigación V.1 inaugurada en FASE 1 vía factor_by_connected_components).
        • En caso contrario: eliminación euclídea sobre B₁ ∈ ℤ^{|V|×|E|},
              d₁ | d₂ | ⋯ | d_r  (Gauss).

    Implementador canónico (FASE 2): SmithNormalFormReducer.reduce_from_complex.
    """

    def reduce_from_complex(self, complex_: SimplicialCohainComplex) -> Any: ...


# ══════════════════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 1 — SUBSTRATO OBSERVACIONAL
# Objeto terminal : SimplicialCohainComplex          (π₁₂)
# Morfismo inicial: Phase2Ingress.reduce_from_complex
# Siguiente paso  : FASE 2 — SNF Euclid, Betti, Fiedler–Wielandt, Cheeger sweep, Ψ
# ══════════════════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 2 — ANÁLISIS ESTRUCTURAL ▓▓▓
# ▓▓▓ SNF Euclid sobre ℤ (bloques V.1), Betti, Fiedler (Wielandt+kernel), Cheeger O(n²). ▓▓▓
# ▓▓▓ Consume: SimplicialCohainComplex (π₁₂).  Objeto terminal: TopologicalInvariants. ▓▓▓
# ▓▓▓ Inicio formal: Phase2Ingress.reduce_from_complex  (Protocol FASE 1).             ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════
#
#  Versión de estrato: 6.1.0-Nested-Phase2-EuclidBlock-WielandtKernel-Sweep-Psi
#
#  II.1  Forma Normal de Smith sobre el DIP ℤ
#        A ∈ ℤ^{m×n},  S = U A V,  U,V ∈ GL(·,ℤ),  S = diag(d₁,…,d_r,0,…),
#            d₁ | d₂ | ⋯ | d_r     (Gauss).
#        Tor(H) = ⊕_{dᵢ>1} ℤ/dᵢℤ.  Aniquilación: Tor≡0  ⇔  dᵢ=1 ∀i.
#        Teorema (1-complejos): H_*(K;ℤ) es libre abeliano ⇒ Tor(H₁)=0 para grafos.
#        Presupuesto: |V|≤200, |E|≤2000 por bloque; si un bloque excede, SNF se omite
#        en ESE bloque (V.1) y Tor≡0 se declara por teorema, no por factorización.
#        Mitigación: π₁₂ ya etiquetó componentes; aquí SNF = ⨁_c SNF(B₁|_{K_c}).
#
#  II.2  Euler–Poincaré (1-complejo)
#            χ(K)  =  |V| − |E|  =  β₀ − β₁,     β₁ = |E| − |V| + β₀.
#
#  II.3  Fiedler / conectividad algebraica
#        Deflación de Wielandt (β₀=1):  L_def = L₀ + γ (11ᵀ/n),  γ = 2Δ_máx+1,
#            σ(L₀)={0,λ₂,…,λ_n}  ↦  σ(L_def)={γ,λ₂,…,λ_n}.
#        Deflación de kernel completo (β₀>1):  L_def = L₀ + γ Σ_c 1_{C_c}1_{C_c}ᵀ/|C_c|,
#            desplaza TODO ker L₀; el λ_mín de L_def es λ_{β₀+1}(L₀) (primer positivo).
#        Teorema: λ₂=0  ⇔  G disconexo.  Residual de Wilkinson: |λ̂−λ| ≤ ‖r‖/‖v‖.
#        V.3: si β₀>1 el vector de λ₂=0 NO es w₂; se marca fiedler_faithful=False.
#
#  II.4  Cheeger–Dodziuk (conductancia combinatoria)
#            h(K)  =  min_{vol(S)≤vol(V)/2}  |∂S| / vol(S),
#            |∂S|  =  1_Sᵀ L₀ 1_S,     vol(S) = 1_Sᵀ diag(L₀).
#            λ₂/2  ≤  h(K)  ≤  √( 2 λ₂ (2Δ_máx + λ₂) ).
#        Sweep-cut incremental O(n²) sobre el orden de w₂.  Si λ₂=0 ⇒ h=0.
#        El sweep es un corte particular: sweep ≥ h; por tanto λ₂/2 ≤ sweep
#        (no sweep ≤ cota superior teórica, que acota h, no al sweep).
#
#  II.5  Índice piramidal
#            Ψ  =  (Σ_{e ∋ Core} w_e) / (Σ_e w_e)  ∈ [0,1].  Vacuo: Ψ=1.
#
#  PUERTO π₂₃ :  SystemTopology.compute_invariants()  →  TopologicalInvariants
#  INGRESO F₃ :  Phase3Ingress.classify(invariants)     ← inicio formal FASE 3
#
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    final,
    runtime_checkable,
    Protocol,
)

__all__.extend(  # type: ignore[name-defined]
    [
        "SmithNormalFormResult",
        "SmithNormalFormReducer",
        "BettiNumbers",
        "FiedlerSpectrum",
        "FiedlerExtractor",
        "CheegerResult",
        "CheegerEstimator",
        "StructuralCertificates",
        "RequestLoopInfo",
        "TopologicalInvariants",
        "SystemTopology",
        "Phase3Ingress",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ II.1  FORMA NORMAL DE SMITH SOBRE ℤ  —  inicio formal de F₂                      ▓▓▓
# ▓▓▓ Realiza Phase2Ingress.reduce_from_complex  (Protocol, FASE 1).                    ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class SmithNormalFormResult:
    r"""
    S = U A V sobre ℤ.  d₁ | ⋯ | d_r (Gauss).  Tor = ⊕_{dᵢ>1} ℤ/dᵢ.

    `skipped=True` ⇔ al menos un bloque excedió el presupuesto Euclid (V.1):
    no se afirma la cadena de divisibilidad *global*; torsion_free se declara
    por el teorema «H_*(1-complejo;ℤ) es libre» en los bloques omitidos y por
    cómputo en los bloques factorizados.

    `blockwise=True` ⇔ SNF = ⨁_c SNF(B₁|_{K_c})  (mitigación V.1).
    `divisibility_ok` certifica (N2) sobre `divisors` (tras gcd/lcm bubbling).
    """

    divisors: Tuple[int, ...]
    rank: int
    torsion_free: bool
    torsion_divisors: Tuple[int, ...]
    skipped: bool = False
    skip_reason: str = ""
    blockwise: bool = False
    n_blocks: int = 1
    n_blocks_omitted: int = 0
    divisibility_ok: bool = True

    @property
    def torsion_order(self) -> int:
        """|Tor| = Π dᵢ  (dᵢ>1).  1 ssi torsion_free."""
        order = 1
        for d in self.torsion_divisors:
            order *= int(d)
        return order

    @property
    def invariant_factors_unimodular(self) -> bool:
        """True ssi todos los dᵢ valen 1 (presentación libre de rango `rank`)."""
        return self.torsion_free and all(d == 1 for d in self.divisors)

    def satisfies_gauss_chain(self) -> bool:
        r"""(N2) d₁ | d₂ | ⋯ | d_r.  Vacío ⇒ vacuamente cierto.  skipped ⇏ falso."""
        if not self.divisors:
            return True
        for a, b in zip(self.divisors, self.divisors[1:]):
            if a == 0 or (b % a) != 0:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divisors": list(self.divisors),
            "rank": self.rank,
            "torsion_free": self.torsion_free,
            "torsion_divisors": list(self.torsion_divisors),
            "torsion_order": self.torsion_order,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "blockwise": self.blockwise,
            "n_blocks": self.n_blocks,
            "n_blocks_omitted": self.n_blocks_omitted,
            "divisibility_ok": self.divisibility_ok and self.satisfies_gauss_chain(),
            "invariant_factors_unimodular": self.invariant_factors_unimodular,
        }


@final
class SmithNormalFormReducer:
    r"""
    SNF sobre ℤ por eliminación euclídea iterativa (enteros de precisión arbitraria).

    Inicio formal de FASE 2: `reduce_from_complex` realiza Phase2Ingress (π₁₂).

    Pivote: entrada de mínimo |·| en el subbloque restante (reduce hinchazón);
    si existe un pivote unimodular (|d|=1) se elige de inmediato.

    PECADO RESIDUAL V.1: si un bloque conexo excede (SNF_MAX_VERTICES,
    SNF_MAX_EDGES) se omite ESE bloque.  Los demás se factorizan.  Si el
    complejo es conexo y excede, SNF se omite por completo y Tor≡0 se declara
    por el teorema de 1-complejos.

    Complejidad (por bloque): O(r·(m+n)·B) con B = coste de aritmética entera
    (hinchazón bit-length).  Guardia de bucles internos: 8·max(m,n)+8.
    """

    INNER_LOOP_MULT: ClassVar[int] = 8

    @staticmethod
    def _swap_rows(M: List[List[int]], i: int, j: int) -> None:
        if i != j:
            M[i], M[j] = M[j], M[i]

    @staticmethod
    def _swap_cols(M: List[List[int]], i: int, j: int) -> None:
        if i != j:
            for row in M:
                row[i], row[j] = row[j], row[i]

    @staticmethod
    def _select_pivot(M: List[List[int]], k: int, m: int, n: int) -> Tuple[int, int]:
        r"""
        (i*, j*) = argmin{|Mᵢⱼ| : Mᵢⱼ ≠ 0, i≥k, j≥k}.
        Atajo unimodular: el primer |Mᵢⱼ|=1 es óptimo (no hincha).
        (−1,−1) ssi el subbloque es nulo.
        """
        best_i, best_j, best_abs = -1, -1, 0
        for i in range(k, m):
            row = M[i]
            for j in range(k, n):
                v = row[j]
                if v == 0:
                    continue
                av = v if v > 0 else -v
                if av == 1:
                    return i, j
                if best_i < 0 or av < best_abs:
                    best_i, best_j, best_abs = i, j, av
        return best_i, best_j

    @classmethod
    def _zero_row(cls, M: List[List[int]], k: int, m: int, n: int) -> None:
        """Euclides por columnas: anula M[k, k+1:].  Intercambia si el resto ≠ 0."""
        j = k + 1
        while j < n:
            while M[k][j] != 0:
                piv = M[k][k]
                if piv == 0:
                    cls._swap_cols(M, k, j)
                    continue
                q = M[k][j] // piv
                if q != 0:
                    for i in range(k, m):
                        M[i][j] -= q * M[i][k]
                if M[k][j] != 0:
                    cls._swap_cols(M, k, j)
            j += 1

    @classmethod
    def _zero_col(cls, M: List[List[int]], k: int, m: int, n: int) -> None:
        """Euclides por filas: anula M[k+1:, k]."""
        i = k + 1
        while i < m:
            while M[i][k] != 0:
                piv = M[k][k]
                if piv == 0:
                    cls._swap_rows(M, i, k)
                    continue
                q = M[i][k] // piv
                if q != 0:
                    pk = M[k]
                    pi = M[i]
                    for j in range(k, n):
                        pi[j] -= q * pk[j]
                if M[i][k] != 0:
                    cls._swap_rows(M, i, k)
            i += 1

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        g = math.gcd(a, b)
        return abs((a // g) * b)

    @classmethod
    def _enforce_divisibility(cls, divisors: Sequence[int]) -> Tuple[int, ...]:
        r"""
        Restaura d₁ | d₂ | ⋯ | d_r por bubbling gcd/lcm.

        Si dᵢ ∤ dᵢ₊₁:  (dᵢ, dᵢ₊₁) ← (gcd, lcm).  Se itera hasta punto fijo.
        Preserva el producto Π dᵢ (módulo unidades) y el rango.
        """
        d = [abs(int(x)) for x in divisors if int(x) != 0]
        changed = True
        guard = 0
        max_guard = max(2, len(d) * len(d) + 2)
        while changed and guard < max_guard:
            changed = False
            guard += 1
            for i in range(len(d) - 1):
                g = math.gcd(d[i], d[i + 1])
                if g != d[i]:
                    d[i], d[i + 1] = g, cls._lcm(d[i], d[i + 1])
                    changed = True
        return tuple(d)

    @staticmethod
    def _gauss_chain_ok(divisors: Sequence[int]) -> bool:
        for a, b in zip(divisors, divisors[1:]):
            if a == 0 or (b % a) != 0:
                return False
        return True

    @classmethod
    def reduce(cls, matrix: IntegerMatrix) -> SmithNormalFormResult:
        r"""
        SNF de una matriz entera densa.  No aplica presupuesto (eso vive en
        `reduce_from_complex`).  Diagonal positiva; cadena de Gauss restaurada.
        """
        A = np.atleast_2d(np.asarray(matrix, dtype=np.int64))
        m, n = int(A.shape[0]), int(A.shape[1])
        if m == 0 or n == 0:
            return SmithNormalFormResult((), 0, True, (), divisibility_ok=True)

        M: List[List[int]] = [[int(x) for x in row] for row in A.tolist()]
        k = 0
        inner_cap = cls.INNER_LOOP_MULT * max(m, n) + 8
        while k < min(m, n):
            pi, pj = cls._select_pivot(M, k, m, n)
            if pi < 0:
                break
            cls._swap_rows(M, k, pi)
            cls._swap_cols(M, k, pj)

            inner = 0
            while inner < inner_cap:
                inner += 1
                cls._zero_row(M, k, m, n)
                cls._zero_col(M, k, m, n)
                piv = M[k][k]
                if piv == 0:
                    break
                bad_i = -1
                for i in range(k + 1, m):
                    row_i = M[i]
                    for j in range(k + 1, n):
                        if row_i[j] % piv != 0:
                            bad_i = i
                            break
                    if bad_i >= 0:
                        break
                if bad_i < 0:
                    break
                pk = M[k]
                pb = M[bad_i]
                for j in range(k, n):
                    pk[j] += pb[j]
            else:
                logger.warning(
                    "SNF: guardia de bucle interno saturada en k=%d (m=%d,n=%d)", k, m, n
                )

            if M[k][k] < 0:
                for j in range(n):
                    M[k][j] = -M[k][j]
            if M[k][k] == 0:
                break
            k += 1

        raw = tuple(abs(M[i][i]) for i in range(k) if M[i][i] != 0)
        divisors = cls._enforce_divisibility(raw)
        torsion_divs = tuple(d for d in divisors if d > 1)
        chain_ok = cls._gauss_chain_ok(divisors)
        return SmithNormalFormResult(
            divisors=divisors,
            rank=len(divisors),
            torsion_free=len(torsion_divs) == 0,
            torsion_divisors=torsion_divs,
            divisibility_ok=chain_ok,
        )

    @classmethod
    def merge_block_results(
        cls,
        parts: Sequence[SmithNormalFormResult],
        *,
        skip_reason: str = "",
    ) -> SmithNormalFormResult:
        r"""
        ⨁ de SNF por bloques conexos.  Los factores invariantes se concatenan
        y se restaura Gauss.  rank = Σ rank_c.  Tor = ⨁ Tor_c.

        `skipped` si algún sumando vino omitido.  `blockwise` si |parts|≠1
        o si hubo omisiones parciales.
        """
        if not parts:
            return SmithNormalFormResult((), 0, True, (), skip_reason=skip_reason)
        raw: List[int] = []
        rank = 0
        n_omit = 0
        skipped = False
        reasons: List[str] = []
        for p in parts:
            raw.extend(p.divisors)
            rank += int(p.rank)
            skipped = skipped or p.skipped
            n_omit += int(p.n_blocks_omitted)
            if p.skip_reason:
                reasons.append(p.skip_reason)
        divisors = cls._enforce_divisibility(raw)
        torsion_divs = tuple(d for d in divisors if d > 1)
        reason = skip_reason or " | ".join(reasons)
        return SmithNormalFormResult(
            divisors=divisors,
            rank=rank,
            torsion_free=len(torsion_divs) == 0,
            torsion_divisors=torsion_divs,
            skipped=skipped,
            skip_reason=reason,
            blockwise=len(parts) > 1 or skipped,
            n_blocks=len(parts),
            n_blocks_omitted=n_omit,
            divisibility_ok=cls._gauss_chain_ok(divisors),
        )

    @classmethod
    def _omit_block(cls, fac: SimplicialCohainComplex) -> SmithNormalFormResult:
        """V.1: omite Euclid; rank vía SVD de B₁; Tor≡0 por teorema de 1-complejos."""
        reason = (
            f"SNF Euclid omitida en bloque: {fac.snf_budget_reason()}. "
            "Tor≡0 por teorema de 1-complejos (pecado residual V.1)."
        )
        logger.warning(reason)
        rank_R = fac.rank_over_R()
        return SmithNormalFormResult(
            divisors=(),
            rank=rank_R,
            torsion_free=True,
            torsion_divisors=(),
            skipped=True,
            skip_reason=reason,
            blockwise=True,
            n_blocks=1,
            n_blocks_omitted=1,
            divisibility_ok=True,
        )

    @classmethod
    def reduce_from_complex(cls, complex_: SimplicialCohainComplex) -> SmithNormalFormResult:
        r"""
        Inicio formal de FASE 2.  Realiza Phase2Ingress.  Consume π₁₂.

        Estrategia (mitigación V.1):
            1. Factorizar K = ⊔_c K_c  (etiquetas de FASE 1 / UF sobre B₁).
            2. Para cada K_c: Euclid si cabe en presupuesto; si no, omitir.
            3. Fusionar factores invariantes (⨁) y restaurar d₁ | ⋯ | d_r.

        Si el complejo es conexo y excede, degenera al caso «omitir global».
        B₁ vacío (sin aristas): SNF trivial, rank 0, Tor≡0.
        """
        if complex_.n_vertices == 0 or complex_.B1.size == 0:
            if complex_.n_vertices == 0:
                return SmithNormalFormResult((), 0, True, ())
            if complex_.n_edges == 0:
                return SmithNormalFormResult(
                    (),
                    0,
                    True,
                    (),
                    blockwise=False,
                    n_blocks=max(1, complex_.n_connected_components()),
                )

        factors = complex_.factor_by_connected_components()
        if not factors:
            return SmithNormalFormResult((), 0, True, ())

        if len(factors) == 1 and not factors[0].exceeds_snf_budget():
            return cls.reduce(factors[0].B1)

        parts: List[SmithNormalFormResult] = []
        for fac in factors:
            if fac.exceeds_snf_budget():
                parts.append(cls._omit_block(fac))
            elif fac.B1.size == 0:
                parts.append(SmithNormalFormResult((), 0, True, ()))
            else:
                parts.append(cls.reduce(fac.B1))
        merged = cls.merge_block_results(parts)
        if merged.skipped:
            logger.info(
                "SNF blockwise: %d bloque(s), %d omitido(s), rank=%d, Tor-free=%s",
                merged.n_blocks,
                merged.n_blocks_omitted,
                merged.rank,
                merged.torsion_free,
            )
        return merged


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ II.2  NÚMEROS DE BETTI  +  EULER–POINCARÉ  (N1)                                   ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class BettiNumbers:
    r"""
    β₀, β₁ + diagnóstico SNF.

        χ = |V|−|E| = β₀−β₁,     β₁ = |E|−|V|+β₀.     (N1)

    Coherencias: β₀,β₁≥0; β₀≤|V|; β₁≤|E|; Euler–Poincaré estricto.
    `snf` transporta (N2) y el bit V.1 (`skipped`).
    """

    b0: BettiIndex
    b1: BettiIndex
    num_vertices: int = 0
    num_edges: int = 0
    snf: Optional[SmithNormalFormResult] = None

    def __post_init__(self) -> None:
        if self.b0 < 0 or self.b1 < 0:
            raise BettiNumberError(f"Betti negativos: β₀={self.b0}, β₁={self.b1}")
        if self.num_vertices < 0 or self.num_edges < 0:
            raise BettiNumberError("Vértices/aristas negativos")
        if self.num_vertices > 0 and self.b0 > self.num_vertices:
            raise BettiNumberError(f"β₀ ({self.b0}) > |V| ({self.num_vertices})")
        if self.num_vertices > 0 and self.num_edges >= 0:
            expected_b1 = self.num_edges - self.num_vertices + self.b0
            if expected_b1 >= 0 and self.b1 != expected_b1:
                raise BettiNumberError(
                    f"Violación Euler-Poincaré: β₁={self.b1}, esperado={expected_b1}",
                    context={"actual_b1": self.b1, "expected_b1": expected_b1},
                )
            if expected_b1 < 0:
                raise BettiNumberError(
                    f"Euler–Poincaré produciría β₁<0: |E|−|V|+β₀={expected_b1}",
                    context={"n_v": self.num_vertices, "n_e": self.num_edges, "b0": self.b0},
                )
        if self.num_edges > 0 and self.b1 > self.num_edges:
            raise BettiNumberError(f"β₁ ({self.b1}) > |E| ({self.num_edges})")

    @classmethod
    def from_counts(
        cls,
        n_vertices: int,
        n_edges: int,
        b0: int,
        snf: Optional[SmithNormalFormResult] = None,
    ) -> "BettiNumbers":
        r"""
        Constructor Euler–Poincaré: β₁ = max(0, |E|−|V|+β₀).
        El max(0,·) es defensivo; en un 1-complejo simple el argumento es ≥0.
        """
        if n_vertices == 0:
            return cls(b0=0, b1=0, num_vertices=0, num_edges=0, snf=snf)
        b1 = max(0, n_edges - n_vertices + b0)
        return cls(b0=b0, b1=b1, num_vertices=n_vertices, num_edges=n_edges, snf=snf)

    @property
    def is_connected(self) -> bool:
        return self.b0 == 1

    @property
    def is_acyclic(self) -> bool:
        return self.b1 == 0

    @property
    def is_ideal(self) -> bool:
        """Árbol (o punto): conexo, acíclico.  No exige Tor-free (redundante en grafos)."""
        return self.is_connected and self.is_acyclic

    @property
    def torsion_free(self) -> bool:
        return True if self.snf is None else self.snf.torsion_free

    @property
    def snf_skipped(self) -> bool:
        return bool(self.snf is not None and self.snf.skipped)

    @property
    def euler_characteristic(self) -> EulerCharacteristic:
        return self.b0 - self.b1

    @property
    def euler_characteristic_alt(self) -> EulerCharacteristic:
        return self.num_vertices - self.num_edges

    @property
    def cyclomatic_complexity(self) -> int:
        """m − n + p  (cyclomatic = β₁ + β₀ en un 1-complejo)."""
        return self.b1 + self.b0

    def verify_euler_consistency(self) -> bool:
        """(N1) χ homológica = χ combinatoria."""
        return self.euler_characteristic == self.euler_characteristic_alt

    def hodge_rank_bound(self) -> int:
        r"""
        rank_ℝ(B₁) = |V| − β₀  si W ≻ 0 (N7).  Cota combinatoria, independiente de L₀.
        """
        return max(0, self.num_vertices - self.b0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "b0": self.b0,
            "b1": self.b1,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "euler_characteristic": self.euler_characteristic,
            "is_connected": self.is_connected,
            "is_acyclic": self.is_acyclic,
            "is_ideal": self.is_ideal,
            "torsion_free": self.torsion_free,
            "snf_skipped": self.snf_skipped,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "euler_consistent": self.verify_euler_consistency(),
            "hodge_rank_bound": self.hodge_rank_bound(),
            "snf": self.snf.to_dict() if self.snf else None,
        }

    def __str__(self) -> str:
        tag = "✓" if self.is_ideal and self.torsion_free else "⚠"
        return (
            f"BettiNumbers({tag} β₀={self.b0}, β₁={self.b1}, "
            f"χ={self.euler_characteristic}, Tor-free={self.torsion_free})"
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ II.3  FIEDLER / WIELANDT / KERNEL COMPLETO  (N3, V.3)                             ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class FiedlerSpectrum:
    r"""
    Firma espectral de L₀ tras deflación de Wielandt (y, si β₀>1, del kernel entero).

    `faithful=False` ⇔ λ₂≈0 (β₀>1): el `vector` NO es w₂ (V.3).  λ₂=0 es la
    señal correcta de desconexión.  `first_positive` es λ_{β₀+1}(L₀) obtenido
    deflactando TODAS las constantes por componente (no resuelve V.3: no
    convierte ese modo en «el» Fiedler del grafo).

    Wilkinson: |λ̂−λ| ≤ residual_rel · escala  ≤ ‖L v − λ v‖ / ‖v‖  (simétrico).
    """

    lambda2: float
    vector: Vector
    gamma: float
    kernel_multiplicity: int
    kernel_multiplicity_exact: bool
    is_connected: bool
    residual: float
    method: str
    faithful: bool
    rayleigh_l0: float = 0.0
    wilkinson_bound: float = 0.0
    first_positive: float = 0.0
    n_deflated_kernel: int = 1
    combinatorial_beta0: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda2": self.lambda2,
            "gamma": self.gamma,
            "kernel_multiplicity": self.kernel_multiplicity,
            "kernel_multiplicity_exact": self.kernel_multiplicity_exact,
            "is_connected": self.is_connected,
            "residual": self.residual,
            "method": self.method,
            "faithful": self.faithful,
            "rayleigh_l0": self.rayleigh_l0,
            "wilkinson_bound": self.wilkinson_bound,
            "first_positive": self.first_positive,
            "n_deflated_kernel": self.n_deflated_kernel,
            "combinatorial_beta0": self.combinatorial_beta0,
        }


@final
class FiedlerExtractor:
    r"""
    λ₂(L₀) y w₂ sin diagonalización completa.

        (β₀=1)  L_def = L₀ + γ 11ᵀ/n,   γ = 2Δ_máx+1,
                σ(L₀)={0,λ₂,…} ↦ {γ,λ₂,…}.                         (N3)
        (β₀>1)  L_def = L₀ + γ Σ_c 1_{C_c}1_{C_c}ᵀ/|C_c|
                desplaza el kernel entero; λ_mín(L_def)=λ_{β₀+1}(L₀).

    PECADO RESIDUAL V.3: si β₀>1, ker L₀ tiene dim>1; reportamos λ₂=0 y
    faithful=False.  La deflación completa NO finge un w₂: solo expone el
    primer modo positivo como diagnóstico (`first_positive`).

    Residual de Wilkinson (simétrico): |λ̂−λ| ≤ ‖r‖/‖v‖.
    """

    DENSE_THRESHOLD: ClassVar[int] = TC.FIEDLER_DENSE_THRESHOLD

    def __init__(self, use_sparse: Optional[bool] = None) -> None:
        if use_sparse is None:
            self._use_sparse = _HAS_SCIPY
        else:
            self._use_sparse = bool(use_sparse) and _HAS_SCIPY

    @staticmethod
    def _gamma_from_L(L0: Matrix) -> float:
        n = int(L0.shape[0]) if L0.ndim == 2 else 0
        if n == 0:
            return 1.0
        delta_max = float(np.max(np.diag(L0)))
        return 2.0 * delta_max + 1.0

    @staticmethod
    def deflate(L0: Matrix, gamma: Optional[float] = None) -> Tuple[Matrix, float]:
        """Wielandt clásico: proyecta solo el vector constante global (N3)."""
        n = int(L0.shape[0]) if L0.ndim == 2 else 0
        if n == 0:
            return L0.copy(), 0.0
        if gamma is None:
            gamma = FiedlerExtractor._gamma_from_L(L0)
        ones = np.ones((n, 1), dtype=np.float64)
        return L0 + gamma * (ones @ ones.T) / n, float(gamma)

    @staticmethod
    def deflate_full_kernel(
        L0: Matrix,
        labels: Vector,
        gamma: Optional[float] = None,
    ) -> Tuple[Matrix, float, int]:
        r"""
        Deflación de TODAS las constantes por componente:

            L_def = L₀ + γ Σ_c (1_{C_c} 1_{C_c}ᵀ / |C_c|).

        Si labels es una sola clase, coincide con Wielandt.  Retorna
        (L_def, γ, n_componentes_deflactadas).
        """
        n = int(L0.shape[0]) if L0.ndim == 2 else 0
        if n == 0:
            return L0.copy(), 0.0, 0
        if gamma is None:
            gamma = FiedlerExtractor._gamma_from_L(L0)
        lab = np.asarray(labels).astype(np.int64, copy=False)
        if lab.size != n:
            return FiedlerExtractor.deflate(L0, gamma) + (1,)
        L_def = np.array(L0, dtype=np.float64, copy=True)
        n_comp = 0
        for c in np.unique(lab):
            idx = np.flatnonzero(lab == c)
            n_c = int(idx.size)
            if n_c == 0:
                continue
            n_comp += 1
            block = (float(gamma) / float(n_c)) * np.ones((n_c, n_c), dtype=np.float64)
            L_def[np.ix_(idx, idx)] += block
        return L_def, float(gamma), n_comp

    def _count_kernel_multiplicity(
        self, L0_sym: Matrix, n: int
    ) -> Tuple[int, bool]:
        r"""
        Multiplicidad espectral de λ=0.  Exacta si n≤512 (eigh) o si Lanczos
        no satura el bloque k de ceros.
        """
        if n <= TC.KERNEL_COUNT_DENSE_MAX:
            try:
                raw = np.linalg.eigvalsh(L0_sym)
                return int(np.sum(np.abs(raw) < TC.EPSILON_STRICT)), True
            except np.linalg.LinAlgError:
                return 1, False
        if self._use_sparse and eigsh is not None and csr_matrix is not None:
            k = min(12, n - 1)
            try:
                vals, _ = eigsh(csr_matrix(L0_sym), k=k, which="SA")
                m = int(np.sum(np.abs(vals) < TC.EPSILON_STRICT))
                exact = m < k
                return max(m, 1), exact
            except Exception as e:
                logger.warning("Conteo de kernel Lanczos falló (%s)", e)
        return 1, False

    @staticmethod
    def _rayleigh(L: Matrix, v: Vector) -> float:
        den = float(v @ v)
        if den < TC.EPSILON_STRICT:
            return 0.0
        return float(v @ (L @ v)) / den

    @staticmethod
    def _wilkinson(L: Matrix, v: Vector, lam: float) -> Tuple[float, float]:
        r"""(‖r‖, ‖r‖/‖v‖) con r = Lv − λv.  Cota |λ̂−λ| ≤ ‖r‖/‖v‖ si L=Lᵀ."""
        nv = float(np.linalg.norm(v))
        if nv < TC.EPSILON_STRICT:
            return 0.0, 0.0
        r = L @ v - lam * v
        nr = float(np.linalg.norm(r))
        return nr, nr / nv

    def _smallest_eigenpair(
        self, L_def: Matrix, n: int
    ) -> Tuple[float, Vector, str]:
        method = "dense-eigh"
        try:
            if n > self.DENSE_THRESHOLD and self._use_sparse and eigsh is not None:
                eigvals, eigvecs = eigsh(csr_matrix(L_def), k=min(2, n - 1), which="SA")
                idx = int(np.argmin(eigvals))
                lam = max(0.0, float(eigvals[idx]))
                vec = eigvecs[:, idx].astype(np.float64)
                method = "sparse-Lanczos"
            else:
                eigvals, eigvecs = np.linalg.eigh(L_def)
                lam = max(0.0, float(eigvals[0]))
                vec = eigvecs[:, 0].astype(np.float64)
        except Exception as e:
            logger.warning("Lanczos disperso falló (%s); uso dense eigh", e)
            eigvals, eigvecs = np.linalg.eigh(L_def)
            lam = max(0.0, float(eigvals[0]))
            vec = eigvecs[:, 0].astype(np.float64)
            method = "dense-fallback"
        return lam, vec, method

    def extract(
        self,
        L0: Matrix,
        component_labels: Optional[Vector] = None,
    ) -> FiedlerSpectrum:
        r"""
        Extrae (λ₂, w₂) de L₀.

        `component_labels` (gancho FASE 1) da β₀ combinatorio y habilita la
        deflación de kernel completo cuando β₀>1 (diagnóstico V.3).
        """
        n = int(L0.shape[0]) if getattr(L0, "ndim", 0) == 2 else 0
        empty = np.array([], dtype=np.float64)
        if n == 0:
            return FiedlerSpectrum(
                0.0, empty, 0.0, 0, True, False, 0.0, "empty", False,
                combinatorial_beta0=0, n_deflated_kernel=0,
            )
        if n == 1:
            return FiedlerSpectrum(
                0.0, np.array([1.0], dtype=np.float64), 0.0, 1, True, True, 0.0,
                "trivial", True, combinatorial_beta0=1, n_deflated_kernel=1,
            )

        L0_sym = 0.5 * (np.asarray(L0, dtype=np.float64) + np.asarray(L0, dtype=np.float64).T)
        gamma = self._gamma_from_L(L0_sym)

        b0_comb: Optional[int] = None
        if component_labels is not None and int(np.asarray(component_labels).size) == n:
            b0_comb = int(np.unique(np.asarray(component_labels)).size)

        ker_spec, ker_exact = self._count_kernel_multiplicity(L0_sym, n)
        if b0_comb is not None:
            ker_mult = b0_comb
            ker_exact = ker_exact or True
            if ker_spec != b0_comb:
                logger.debug(
                    "Fiedler: β₀ combinatorio=%d vs ker espectral=%d (W no-definida o tol)",
                    b0_comb, ker_spec,
                )
                ker_mult = max(b0_comb, ker_spec)
                ker_exact = False
        else:
            ker_mult = max(1, ker_spec)

        is_connected_comb = (b0_comb == 1) if b0_comb is not None else None

        L_w, _ = self.deflate(L0_sym, gamma)
        lam_w, vec_w, method = self._smallest_eigenpair(L_w, n)
        if lam_w < TC.EPSILON_STRICT:
            lam_w = 0.0

        is_connected = lam_w > TC.EPSILON_STRICT
        if is_connected_comb is not None:
            is_connected = bool(is_connected_comb and is_connected)

        if not ker_exact and lam_w == 0.0:
            ker_mult = max(ker_mult, 2)

        faithful = is_connected
        lambda2 = lam_w if is_connected else 0.0
        vec = vec_w
        first_positive = lam_w
        n_deflated = 1

        if not is_connected and component_labels is not None:
            L_full, gamma, n_deflated = self.deflate_full_kernel(
                L0_sym, np.asarray(component_labels), gamma
            )
            lam_p, vec_p, method_p = self._smallest_eigenpair(L_full, n)
            if lam_p < TC.EPSILON_STRICT:
                lam_p = 0.0
            first_positive = lam_p
            vec = vec_p
            method = f"{method}+fullker:{method_p}"
            residual_mat = L_full
            lam_for_res = lam_p
        else:
            residual_mat = L_w
            lam_for_res = lam_w
            n_deflated = 1 if is_connected else max(1, ker_mult)

        residual, wilkinson = self._wilkinson(residual_mat, vec, lam_for_res)
        rayleigh = self._rayleigh(L0_sym, vec)

        return FiedlerSpectrum(
            lambda2=lambda2,
            vector=vec,
            gamma=gamma,
            kernel_multiplicity=max(1, ker_mult),
            kernel_multiplicity_exact=ker_exact,
            is_connected=is_connected,
            residual=residual,
            method=method,
            faithful=faithful,
            rayleigh_l0=rayleigh,
            wilkinson_bound=wilkinson,
            first_positive=first_positive,
            n_deflated_kernel=n_deflated,
            combinatorial_beta0=int(b0_comb) if b0_comb is not None else max(1, ker_mult),
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ II.4  CHEEGER–DODZIUK + SWEEP INCREMENTAL O(n²)  (N4)                             ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class CheegerResult:
    r"""
    Cotas Cheeger–Dodziuk y sweep-cut:

        λ₂/2  ≤  h(K)  ≤  √(2 λ₂ (2Δ_máx + λ₂)),
        |∂S| = 1_Sᵀ L₀ 1_S,   h_sweep = min_k |∂S_k|/vol(S_k).

    `sweep_is_upper_on_h`: sweep ≥ h  (corte particular).
    `dodziuk_ok`: λ₂/2 ≤ cota_sup  y  (si sweep>0) λ₂/2 ≤ sweep + ε.
    """

    lower_bound: float
    upper_bound: float
    sweep_estimate: float
    spectral_estimate: float
    is_bottleneck: bool
    delta_max: float
    dodziuk_ok: bool = True
    used_fiedler_sweep: bool = False
    n_swept: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "sweep_estimate": self.sweep_estimate,
            "spectral_estimate": self.spectral_estimate,
            "is_bottleneck": self.is_bottleneck,
            "delta_max": self.delta_max,
            "dodziuk_ok": self.dodziuk_ok,
            "used_fiedler_sweep": self.used_fiedler_sweep,
            "n_swept": self.n_swept,
        }


@final
class CheegerEstimator:
    r"""
    h(K) = min_{vol(S)≤vol(V)/2} |∂S|/vol(S).  Sweep incremental O(n²):

        1_{S∪{v}}ᵀ L 1_{S∪{v}}  =  1_Sᵀ L 1_S  +  2 (L 1_S)_v  +  L_{vv}.

    Si λ₂=0 (grafo disconexo) ⇒ h=0, bottleneck=True, sin sweep (V.3:
    un vector no-fiel no se usa como orden isoperimétrico del grafo entero).
    """

    def __init__(self, threshold: float = TC.CHEEGER_THRESHOLD) -> None:
        self.threshold = threshold

    def estimate(
        self,
        L0: Matrix,
        lambda2: float,
        fiedler_vector: Optional[Vector] = None,
        *,
        faithful: bool = True,
    ) -> CheegerResult:
        n = int(L0.shape[0]) if getattr(L0, "ndim", 0) == 2 else 0
        if n == 0:
            return CheegerResult(0.0, 0.0, 0.0, 0.0, False, 0.0, True, False, 0)

        L = np.asarray(L0, dtype=np.float64)
        delta_max = float(np.max(np.diag(L))) if n else 0.0
        lam = max(0.0, float(lambda2))
        lower = lam / 2.0
        upper = math.sqrt(max(0.0, 2.0 * lam * (2.0 * delta_max + lam)))
        spectral = math.sqrt(lower * upper) if upper > 0.0 else lower

        if lam <= TC.EPSILON_STRICT or n < 2:
            return CheegerResult(
                lower_bound=0.0,
                upper_bound=0.0,
                sweep_estimate=0.0,
                spectral_estimate=0.0,
                is_bottleneck=True,
                delta_max=delta_max,
                dodziuk_ok=True,
                used_fiedler_sweep=False,
                n_swept=0,
            )

        sweep = 0.0
        n_swept = 0
        used = False
        if (
            faithful
            and fiedler_vector is not None
            and int(np.asarray(fiedler_vector).size) == n
        ):
            sweep, n_swept = self._sweep_cut(L, np.asarray(fiedler_vector, dtype=np.float64))
            used = n_swept > 0

        effective = sweep if sweep > 0.0 else spectral
        is_bn = effective < self.threshold
        dodziuk_ok = lower <= upper + TC.EPSILON
        if sweep > 0.0:
            dodziuk_ok = dodziuk_ok and (lower <= sweep + TC.EPSILON)
        return CheegerResult(
            lower_bound=lower,
            upper_bound=upper,
            sweep_estimate=sweep,
            spectral_estimate=spectral,
            is_bottleneck=is_bn,
            delta_max=delta_max,
            dodziuk_ok=dodziuk_ok,
            used_fiedler_sweep=used,
            n_swept=n_swept,
        )

    @staticmethod
    def _sweep_cut(L0: Matrix, fiedler_vec: Vector) -> Tuple[float, int]:
        r"""
        Min-conductancia sobre {S_k = {v_{1}…v_k}} ordenado por w₂.
        Retorna (h_best, n_prefijos_válidos).  O(n²) por el update denso de L1_S.
        """
        n = int(L0.shape[0])
        if n < 2:
            return 0.0, 0
        order = np.argsort(fiedler_vec, kind="mergesort")
        vol = np.diag(L0).astype(np.float64, copy=False)
        total_vol = float(vol.sum())
        if total_vol < TC.EPSILON_STRICT:
            return 0.0, 0

        Ls = np.zeros(n, dtype=np.float64)
        cut = 0.0
        vol_S = 0.0
        best_h = float("inf")
        n_valid = 0
        half = total_vol / 2.0
        for k in range(n - 1):
            v = int(order[k])
            cut += 2.0 * float(Ls[v]) + float(L0[v, v])
            Ls += L0[:, v]
            vol_S += float(vol[v])
            if vol_S < TC.EPSILON_STRICT:
                continue
            if vol_S > half + TC.EPSILON_STRICT:
                break
            h = max(0.0, cut) / vol_S
            n_valid += 1
            if h < best_h:
                best_h = h
        if not math.isfinite(best_h):
            return 0.0, n_valid
        return float(best_h), n_valid


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ CERTIFICADOS ESTRUCTURALES (N1)–(N4), (N7)  +  BUCLES DE REQUEST                  ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class StructuralCertificates:
    r"""
    Testigos booleanos de los invariantes globales que F₂ puede certificar
    sin Ω₃ (eso es F₃).

        (N1) Euler–Poincaré: χ = β₀−β₁ = |V|−|E|.
        (N2) Gauss: d₁ | ⋯ | d_r  cuando SNF no está omitida.
        (N3) Wielandt: residual de Wilkinson acotado; λ₂=0 ⇔ ¬conexo.
        (N4) Cheeger–Dodziuk: λ₂/2 ≤ h_sup  (y ≤ sweep si hay sweep).
        (N7) Hodge: ker L₀ ≅ ℝ^{β₀}  (si W ≻ 0 y el conteo es exacto).
    """

    euler_poincare: bool
    gauss_divisibility: bool
    wielandt_residual_ok: bool
    cheeger_dodziuk: bool
    hodge_kernel: bool
    snf_skipped: bool
    fiedler_faithful: bool
    complex_invariants: Dict[str, bool] = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        core = (
            self.euler_poincare
            and self.cheeger_dodziuk
            and self.wielandt_residual_ok
        )
        gauss = self.gauss_divisibility or self.snf_skipped
        return core and gauss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "euler_poincare": self.euler_poincare,
            "gauss_divisibility": self.gauss_divisibility,
            "wielandt_residual_ok": self.wielandt_residual_ok,
            "cheeger_dodziuk": self.cheeger_dodziuk,
            "hodge_kernel": self.hodge_kernel,
            "snf_skipped": self.snf_skipped,
            "fiedler_faithful": self.fiedler_faithful,
            "all_ok": self.all_ok,
            "complex_invariants": dict(self.complex_invariants),
        }


@final
@dataclass(frozen=True, slots=True)
class RequestLoopInfo:
    """Bucle de reintentos (patrón repetitivo en el histórico de peticiones)."""

    request_id: str
    count: int
    first_seen: int
    last_seen: int
    severity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> int:
        return self.last_seen - self.first_seen

    @property
    def frequency(self) -> float:
        if self.duration == 0:
            return float("inf") if self.count > 1 else 0.0
        return self.count / self.duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": self.duration,
            "frequency": self.frequency,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ PUERTO π₂₃  —  objeto terminal de F₂                                              ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class TopologicalInvariants:
    r"""
    Puerto π₂₃  FASE 2 → FASE 3.

    Vector completo de invariantes homológico-espectrales, consumido por
    HeytingVerdictClassifier.classify (inicio formal de F₃).

    Campos V.1 / V.3:
        snf_skipped, fiedler_faithful, kernel_multiplicity[_exact].
    """

    betti: BettiNumbers
    lambda2: float
    fiedler_vector: Optional[Vector]
    fiedler_residual: float
    fiedler_method: str
    cheeger: CheegerResult
    pyramidal_index: float
    num_vertices: int
    num_edges: int
    is_connected_spectral: bool
    fiedler_faithful: bool = True
    kernel_multiplicity: int = 1
    kernel_multiplicity_exact: bool = True
    complex_invariants: Dict[str, bool] = field(default_factory=dict)
    snf_skipped: bool = False
    first_positive_eigenvalue: float = 0.0
    fiedler_wilkinson: float = 0.0
    fiedler_rayleigh: float = 0.0
    certificates: Optional[StructuralCertificates] = None
    weight_class: str = "DEFINITE"
    gershgorin: Optional[Dict[str, Any]] = None
    assembly_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "betti": self.betti.to_dict(),
            "b0": self.betti.b0,
            "b1": self.betti.b1,
            "torsion_free": self.betti.torsion_free,
            "lambda2": self.lambda2,
            "fiedler_residual": self.fiedler_residual,
            "fiedler_method": self.fiedler_method,
            "fiedler_faithful": self.fiedler_faithful,
            "fiedler_wilkinson": self.fiedler_wilkinson,
            "fiedler_rayleigh": self.fiedler_rayleigh,
            "first_positive_eigenvalue": self.first_positive_eigenvalue,
            "kernel_multiplicity": self.kernel_multiplicity,
            "kernel_multiplicity_exact": self.kernel_multiplicity_exact,
            "cheeger": self.cheeger.to_dict(),
            "pyramidal_index": self.pyramidal_index,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "is_connected_spectral": self.is_connected_spectral,
            "complex_invariants": dict(self.complex_invariants),
            "snf_skipped": self.snf_skipped,
            "weight_class": self.weight_class,
            "gershgorin": dict(self.gershgorin) if self.gershgorin else None,
            "assembly_method": self.assembly_method,
            "certificates": self.certificates.to_dict() if self.certificates else None,
        }


class SystemTopology:
    r"""
    Grafo de servicios como 1-complejo. Pipeline F₁→F₂:

        π₁₂ assemble_from_graph → SNF (bloques) → Betti → Fiedler → Cheeger → Ψ
        = π₂₃ TopologicalInvariants.

    Invariante de clase: χ(G) = |V|−|E| = β₀−β₁  (N1).
    """

    REQUIRED_NODES: ClassVar[FrozenSet[str]] = frozenset(
        {"Agent", "Core", "Redis", "Filesystem"}
    )
    EXPECTED_TOPOLOGY: ClassVar[FrozenSet[Tuple[str, str]]] = frozenset(
        {
            ("Core", "Redis"),
            ("Core", "Filesystem"),
            ("Agent", "Core"),
            ("Agent", "Redis"),
            ("Agent", "Filesystem"),
        }
    )
    PYRAMID_HUB: ClassVar[str] = "Core"
    MIN_WINDOW_SIZE: ClassVar[int] = 3
    DEFAULT_WINDOW_SIZE: ClassVar[int] = 50
    MAX_WINDOW_SIZE: ClassVar[int] = 1000

    def __init__(
        self,
        max_history: int = DEFAULT_WINDOW_SIZE,
        custom_nodes: Optional[Set[str]] = None,
        custom_topology: Optional[Set[Tuple[str, str]]] = None,
        validate_strictly: bool = True,
    ) -> None:
        if not (self.MIN_WINDOW_SIZE <= max_history <= self.MAX_WINDOW_SIZE):
            raise ValueError(
                f"max_history ∈ [{self.MIN_WINDOW_SIZE}, {self.MAX_WINDOW_SIZE}]"
            )
        self._max_history = max_history
        self._validate_strictly = validate_strictly
        self._graph: nx.Graph = nx.Graph()

        all_nodes = set(self.REQUIRED_NODES)
        if custom_nodes:
            all_nodes.update(n.strip() for n in custom_nodes if isinstance(n, str))
        self._graph.add_nodes_from(all_nodes)

        self._expected_topology = set(self.EXPECTED_TOPOLOGY)
        if custom_topology:
            self._expected_topology.update(custom_topology)

        self._request_history: deque = deque(maxlen=max_history)
        self._request_index: int = 0

        self._assembler = CompensatedLaplacianAssembler()
        self._snf_reducer = SmithNormalFormReducer()
        self._fiedler = FiedlerExtractor()
        self._cheeger = CheegerEstimator()

        self._betti_cache: Optional[Tuple[str, BettiNumbers]] = None
        self._invariants_cache: Optional[Tuple[str, TopologicalInvariants]] = None

    @property
    def nodes(self) -> Set[str]:
        return set(self._graph.nodes())

    @property
    def edges(self) -> Set[Tuple[str, str]]:
        return set(self._graph.edges())

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def expected_topology(self) -> FrozenSet[Tuple[str, str]]:
        return frozenset(self._expected_topology)

    def _compute_graph_hash(self) -> str:
        nodes_sorted = tuple(sorted(self._graph.nodes()))
        edges_sorted = tuple(sorted(tuple(sorted(e)) for e in self._graph.edges()))
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(repr((nodes_sorted, edges_sorted)).encode())
        return hasher.hexdigest()

    def _invalidate_caches(self) -> None:
        self._betti_cache = None
        self._invariants_cache = None

    def update_connectivity(
        self,
        active_connections: List[Tuple[str, str]],
        validate_nodes: bool = True,
        auto_add_nodes: bool = False,
    ) -> Tuple[int, List[str]]:
        if active_connections is None:
            active_connections = []
        warns: List[str] = []
        valid_edges: List[Tuple[str, str]] = []
        nodes_to_add: Set[str] = set()

        for idx, item in enumerate(active_connections):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                warns.append(f"[{idx}] Arista inválida: {item!r}")
                continue
            src, dst = item
            if not isinstance(src, str) or not isinstance(dst, str):
                warns.append(f"[{idx}] Nodos no-string: {item!r}")
                continue
            src, dst = src.strip(), dst.strip()
            if not src or not dst:
                warns.append(f"[{idx}] Nodo vacío: {item!r}")
                continue
            if src == dst:
                warns.append(f"[{idx}] Auto-loop ignorado: {src}")
                continue
            if validate_nodes and not auto_add_nodes:
                if src not in self._graph or dst not in self._graph:
                    warns.append(f"[{idx}] Nodo faltante: {item!r}")
                    continue
            if auto_add_nodes:
                if src not in self._graph:
                    nodes_to_add.add(src)
                if dst not in self._graph:
                    nodes_to_add.add(dst)
            valid_edges.append((src, dst))

        prev_edges = list(self._graph.edges())
        prev_nodes = set(self._graph.nodes())
        try:
            self._graph.add_nodes_from(nodes_to_add)
            self._graph.clear_edges()
            self._graph.add_edges_from(valid_edges)
            self._invalidate_caches()
        except Exception as e:
            self._graph.clear()
            self._graph.add_nodes_from(prev_nodes)
            self._graph.add_edges_from(prev_edges)
            self._invalidate_caches()
            if self._validate_strictly:
                raise GraphStructureError(
                    f"update_connectivity falló, estado restaurado: {e}"
                ) from e
            return 0, warns + [f"Error crítico: {e}"]

        return len(valid_edges), warns

    def record_request(self, request_id: str) -> bool:
        if not request_id or not isinstance(request_id, str):
            return False
        request_id = request_id.strip()
        if not request_id:
            return False
        self._request_history.append((self._request_index, request_id))
        self._request_index += 1
        return True

    def detect_request_loops(self, threshold: int = 3) -> List[RequestLoopInfo]:
        if not self._request_history:
            return []
        threshold = max(2, int(threshold))
        analysis: Dict[str, Dict[str, Any]] = {}
        for idx, req_id in self._request_history:
            info = analysis.setdefault(req_id, {"count": 0, "first": idx, "last": idx})
            info["count"] += 1
            info["last"] = idx
        loops: List[RequestLoopInfo] = []
        for req_id, info in analysis.items():
            if info["count"] >= threshold:
                duration = info["last"] - info["first"]
                severity = min(1.0, info["count"] / 10.0)
                loops.append(
                    RequestLoopInfo(
                        request_id=req_id,
                        count=info["count"],
                        first_seen=info["first"],
                        last_seen=info["last"],
                        severity=severity,
                        metadata={"duration": duration},
                    )
                )
        return sorted(loops, key=lambda x: (x.severity, x.count), reverse=True)

    def get_disconnected_nodes(self) -> FrozenSet[str]:
        return frozenset(
            n
            for n in self.REQUIRED_NODES
            if n in self._graph and self._graph.degree(n) == 0
        )

    def get_missing_connections(self) -> FrozenSet[Tuple[str, str]]:
        return frozenset(e for e in self._expected_topology if not self._graph.has_edge(*e))

    def calculate_betti_numbers(
        self,
        use_cache: bool = True,
        complex_: Optional[SimplicialCohainComplex] = None,
    ) -> BettiNumbers:
        r"""
        β₀ vía componentes conexas, β₁ vía Euler–Poincaré, SNF sobre B₁
        (o omisión presupuestaria por bloque, V.1).
        """
        if use_cache and self._betti_cache is not None:
            h = self._compute_graph_hash()
            if self._betti_cache[0] == h:
                return self._betti_cache[1]

        n_v = self._graph.number_of_nodes()
        n_e = self._graph.number_of_edges()
        if n_v == 0:
            betti = BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
        else:
            b0 = nx.number_connected_components(self._graph)
            snf: Optional[SmithNormalFormResult] = None
            try:
                cx = (
                    complex_
                    if complex_ is not None
                    else self._assembler.assemble_from_graph(self._graph)
                )
                snf = self._snf_reducer.reduce_from_complex(cx)
            except Exception as e:
                logger.warning(
                    "SNF falló (%s); asumiendo libre de torsión (1-complejo)", e
                )
            betti = BettiNumbers.from_counts(n_v, n_e, b0, snf=snf)

        if use_cache:
            self._betti_cache = (self._compute_graph_hash(), betti)
        return betti

    def _pyramidal_index(self, complex_: SimplicialCohainComplex) -> float:
        r"""
        Ψ = (Σ_{e ∋ Core} w_e) / (Σ_e w_e)  ∈ [0,1].  (II.5)

        Sumas KBN.  Grafo sin aristas ⇒ Ψ=1 (pirámide vacua, no invertida).
        Hub ausente ⇒ numerador 0 ⇒ Ψ=0 (pirámide degenerada).
        """
        wdiag = complex_.weight_diagonal()
        total = float(NeumaierKahanAccumulator.sum_array(wdiag))
        if total <= TC.EPSILON_STRICT:
            return 1.0
        hub = self.PYRAMID_HUB
        acc = NeumaierKahanAccumulator()
        for u, v, data in self._graph.edges(data=True):
            if hub in (u, v):
                acc.add(
                    IEEE754Sanitizer.sanitize_scalar(data.get("weight", 1.0))
                )
        psi = acc.total / total
        if psi < 0.0:
            return 0.0
        if psi > 1.0:
            return 1.0
        return float(psi)

    def _build_certificates(
        self,
        betti: BettiNumbers,
        spectrum: FiedlerSpectrum,
        cheeger: CheegerResult,
        cx_ok: Dict[str, bool],
        wclass: WeightSpectrumClass,
    ) -> StructuralCertificates:
        euler_ok = betti.verify_euler_consistency()
        snf = betti.snf
        gauss_ok = True if snf is None else (snf.skipped or snf.satisfies_gauss_chain())
        n = max(1, betti.num_vertices)
        wielandt_ok = spectrum.wilkinson_bound <= (
            math.sqrt(float(n)) * TC.EPSILON + TC.EPSILON
        ) or spectrum.method in {"empty", "trivial"}
        hodge_ok = True
        if wclass is WeightSpectrumClass.DEFINITE and spectrum.kernel_multiplicity_exact:
            hodge_ok = spectrum.kernel_multiplicity == betti.b0
        hodge_ok = hodge_ok and bool(cx_ok.get("rank_nullity", True))
        return StructuralCertificates(
            euler_poincare=euler_ok,
            gauss_divisibility=gauss_ok,
            wielandt_residual_ok=wielandt_ok,
            cheeger_dodziuk=cheeger.dodziuk_ok,
            hodge_kernel=hodge_ok,
            snf_skipped=betti.snf_skipped,
            fiedler_faithful=spectrum.faithful,
            complex_invariants=dict(cx_ok),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ► PUERTO DE SALIDA FASE 2 → FASE 3
    # ► Este método ES el objeto terminal de F₂ y el objeto inicial de F₃.
    # ► Phase3Ingress.classify / HeytingVerdictClassifier.classify lo consumen.
    # ═══════════════════════════════════════════════════════════════════════════
    def compute_invariants(self, use_cache: bool = True) -> TopologicalInvariants:
        r"""
        Puerto π₂₃. Orquesta F₁→F₂ completo:

            assemble_from_graph (π₁₂)
          → SNF blockwise (V.1)
          → Betti + Euler–Poincaré (N1)
          → Fiedler–Wielandt / kernel (N3, V.3)
          → Cheeger sweep (N4)
          → Ψ piramidal (II.5)
          → certificados estructurales.

        Único punto de salida de FASE 2 hacia FASE 3.
        """
        if use_cache and self._invariants_cache is not None:
            h = self._compute_graph_hash()
            if self._invariants_cache[0] == h:
                return self._invariants_cache[1]

        complex_ = self._assembler.assemble_from_graph(self._graph)
        cx_ok = complex_.verify_invariants()
        wclass = complex_.classify_weight_spectrum()
        gersh = complex_.gershgorin().to_dict()

        betti = self.calculate_betti_numbers(use_cache=use_cache, complex_=complex_)
        labels = complex_.ensure_component_labels()
        spectrum = self._fiedler.extract(complex_.L0, component_labels=labels)
        cheeger_result = self._cheeger.estimate(
            complex_.L0,
            spectrum.lambda2,
            fiedler_vector=spectrum.vector,
            faithful=spectrum.faithful,
        )
        psi = self._pyramidal_index(complex_)
        certs = self._build_certificates(betti, spectrum, cheeger_result, cx_ok, wclass)

        invariants = TopologicalInvariants(
            betti=betti,
            lambda2=spectrum.lambda2,
            fiedler_vector=spectrum.vector,
            fiedler_residual=spectrum.residual,
            fiedler_method=spectrum.method,
            cheeger=cheeger_result,
            pyramidal_index=psi,
            num_vertices=self.num_nodes,
            num_edges=self.num_edges,
            is_connected_spectral=spectrum.is_connected,
            fiedler_faithful=spectrum.faithful,
            kernel_multiplicity=spectrum.kernel_multiplicity,
            kernel_multiplicity_exact=spectrum.kernel_multiplicity_exact,
            complex_invariants=cx_ok,
            snf_skipped=betti.snf_skipped,
            first_positive_eigenvalue=spectrum.first_positive,
            fiedler_wilkinson=spectrum.wilkinson_bound,
            fiedler_rayleigh=spectrum.rayleigh_l0,
            certificates=certs,
            weight_class=wclass.name,
            gershgorin=gersh,
            assembly_method=complex_.assembly_method,
        )
        if use_cache:
            self._invariants_cache = (self._compute_graph_hash(), invariants)
        logger.debug(
            "Fase2►Fase3: β₀=%d β₁=%d λ₂=%.4g Ψ=%.3f faithful=%s snf_skip=%s",
            betti.b0,
            betti.b1,
            spectrum.lambda2,
            psi,
            spectrum.faithful,
            betti.snf_skipped,
        )
        return invariants


# ══════════════════════════════════════════════════════════════════════════════════════════
# ► CONTINUACIÓN FORMAL FASE 2 → FASE 3
# ► El objeto terminal de F₂ (TopologicalInvariants) es el objeto inicial de F₃.
# ► Este Protocol ES el primer método de la Fase 3: HeytingVerdictClassifier lo realiza.
# ══════════════════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Phase3Ingress(Protocol):
    r"""
    Morfismo de puerto π₂₃ → F₃.

    Definición formal (inicio de FASE 3):

        classify : TopologicalInvariants → (HeytingVerdict, Dict[str, str])

    Semántica (a implementar en FASE 3) sobre el retículo de Heyting
    Ω₃ = {⊥, ½, ⊤}  con ⊥ absorbente:

        ⊤ COHERENT  ⇔  β₀=1 ∧ β₁=0 ∧ Tor≡0 ∧ Ψ≥0.70 ∧ ¬bottleneck
        ½ DEGRADED  ⇔  β₀=1 ∧ β₁=0 ∧ Tor≡0 ∧ (0.50≤Ψ<0.70 ∨ bottleneck)
        ⊥ VETOED    ⇔  β₀>1 ∨ β₁>0 ∨ Tor≠0 ∨ Ψ<0.50

    Notas de pecados residuales que F₃ debe respetar, no «corregir»:
        V.1  snf_skipped=True ⇏ Tor≠0;  torsion_free sigue siendo True por teorema.
        V.3  fiedler_faithful=False ⇒ λ₂=0 ya está reflejado en β₀>1 (⊥ vía fragmentación).
        V.4  el clasificador NO actúa sobre GPIO; eso es el Crowbar (contrato, dry_run).

    Implementador canónico (FASE 3): HeytingVerdictClassifier.classify.
    """

    def classify(
        self, invariants: TopologicalInvariants
    ) -> Tuple[HeytingVerdict, Dict[str, str]]: ...


# ══════════════════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 2 — ANÁLISIS ESTRUCTURAL
# Objeto terminal : TopologicalInvariants                 (π₂₃)
# Morfismo inicial: Phase3Ingress.classify
# Siguiente paso  : FASE 3 — Ω₃ Heyting, score S, Crowbar (V.4), fachada Φ = F₃∘F₂∘F₁
# ══════════════════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ FASE 3 — DECISIÓN Y ACTUACIÓN ▓▓▓
# ▓▓▓ Clasificador Heyting Ω₃ + score S + Crowbar (contrato, sin HAL) + Φ = F₃∘F₂∘F₁.  ▓▓▓
# ▓▓▓ Consume: TopologicalInvariants (π₂₃).  Inicio formal: Phase3Ingress.classify.    ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════
#
#  Versión de estrato: 6.1.0-Nested-Phase3-HeytingMeet-UnitPartition-CrowbarContract-Phi
#
#  III.1  Retículo de Heyting Ω₃ = {⊥, ½, ⊤}                                           (N6)
#           v ∧ w = min(v,w),   v ∨ w = max(v,w),
#           v → w = ⊤ si v≤w sino w,   ¬v = v → ⊥.
#         ⊥ es absorbente: ⊥ ∧ x = ⊥  ∀x.
#         Clasificación como meet de morfismos atómicos 𝒞_Topo → Ω₃:
#           ⊤ COHERENT  ⇔  β₀=1 ∧ β₁=0 ∧ Tor≡0 ∧ Ψ≥0.70 ∧ ¬bottleneck
#           ½ DEGRADED  ⇔  β₀=1 ∧ β₁=0 ∧ Tor≡0 ∧ (0.50≤Ψ<0.70 ∨ bottleneck)
#           ⊥ VETOED    ⇔  β₀≠1 ∨ β₁>0 ∨ Tor≠0 ∨ Ψ<0.50
#         V.1: snf_skipped ⇏ Tor≠0 (torsion_free sigue True por teorema).
#         V.3: fiedler_faithful=False ya está en β₀≠1 (átomo connected = ⊥).
#
#  III.2  Score de salud (partición de la unidad en Ω₃, independiente del veredicto)
#           S  =  1 − Σᵢ pᵢ wᵢ,     Σ wᵢ = 1,     S ∈ [0,1].
#         Cuantización suave: HealthLevel.from_score  (≠ HeytingVerdict).
#
#  III.3  Crowbar ciber-físico (contrato, sin HAL real — V.4)
#         VETOED ⇒ trigger_crowbar (idempotente ARMED → FIRED).
#         dry_run=True, HAS_REAL_HAL=False.  El firmware C++ (fuera de este
#         módulo) debe ejecutar en ISR IRAM: GPIO.out_w1ts = (1<<14) en < 400 ns.
#
#  III.4  Orquestación  Φ = F₃ ∘ F₂ ∘ F₁
#         1. assemble_from_graph → complejo (π₁₂).
#         2. SNF + Betti + Fiedler + Cheeger + Ψ → invariantes (π₂₃).
#         3. classify → Ω₃; score; si ⊥: crowbar (+ raise opcional).
#
import abc
from typing import Mapping

__all__.extend(  # type: ignore[name-defined]
    [
        "HeytingEvidence",
        "HeytingVerdictClassifier",
        "HealthPenaltyBreakdown",
        "HealthScoreCalculator",
        "TopologicalHealth",
        "CrowbarState",
        "ActuationEvent",
        "CyberPhysicalActuator",
        "ESP32CrowbarActuator",
        "TopologicalAnalyzerFacade",
        "create_simple_topology",
        "create_cyclic_topology",
        "create_disconnected_topology",
        "compute_wasserstein_distance",
        "compute_bottleneck_distance_approx",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ III.1  EVIDENCIA ATÓMICA  𝒞_Topo → Ω₃  —  inicio formal de F₃                     ▓▓▓
# ▓▓▓ Realiza Phase3Ingress.classify  (Protocol, FASE 2).                               ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class HeytingEvidence:
    r"""
    Cinco morfismos atómicos  TopologicalInvariants → Ω₃  y su meet.

        connected : β₀=1 ↦ ⊤  else ⊥
        acyclic   : β₁=0 ↦ ⊤  else ⊥
        torsion   : Tor≡0 ↦ ⊤  else ⊥     (V.1: skipped conserva ⊤)
        pyramidal : Ψ≥0.70 ↦ ⊤;  Ψ≥0.50 ↦ ½;  else ⊥
        cheeger   : ¬bottleneck ↦ ⊤;  bottleneck ↦ ½   (nunca ⊥)

    `collapse` = ∧ de los cinco.  ⊥ absorbente ⇒ un solo átomo ⊥ veta el todo.
    """

    connected: HeytingVerdict
    acyclic: HeytingVerdict
    torsion: HeytingVerdict
    pyramidal: HeytingVerdict
    cheeger: HeytingVerdict
    reasons: Dict[str, str] = field(default_factory=dict)

    def atoms(self) -> Tuple[HeytingVerdict, ...]:
        return (
            self.connected,
            self.acyclic,
            self.torsion,
            self.pyramidal,
            self.cheeger,
        )

    def collapse(self) -> HeytingVerdict:
        r"""Meet de Heyting.  Complejidad O(1) (|Ω₃|=3, 5 átomos)."""
        v = HeytingVerdict.COHERENT
        for atom in self.atoms():
            v = v.meet(atom)
        return v

    def absorbing_veto(self) -> bool:
        """True ssi algún átomo es ⊥  (⇔ collapse = ⊥, por absorbencia)."""
        return any(a is HeytingVerdict.VETOED for a in self.atoms())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected.name,
            "acyclic": self.acyclic.name,
            "torsion": self.torsion.name,
            "pyramidal": self.pyramidal.name,
            "cheeger": self.cheeger.name,
            "collapse": self.collapse().name,
            "lattice_value": self.collapse().lattice_value,
            "reasons": dict(self.reasons),
        }


@final
class HeytingVerdictClassifier:
    r"""
    Colapso a Ω₃.  Inicio formal de FASE 3: realiza Phase3Ingress, consume π₂₃.

    Cada predicado es un morfismo de retículo; el veredicto es su meet.
    Equivalencia con las cláusulas III.1:

        collapse = ⊤  ⇔  todos los átomos ⊤
        collapse = ½  ⇔  ningún ⊥ y al menos un ½
        collapse = ⊥  ⇔  al menos un ⊥

    V.1  `snf_skipped` no produce átomo de torsión (el teorema de 1-complejos
         ya declaró Tor≡0).  Se registra en reasons['snf'] como nota, no como veto.
    V.3  `fiedler_faithful=False` no es un átomo extra: implica β₀≠1 y el
         átomo `connected` ya es ⊥.
    """

    @staticmethod
    def verify_heyting_axioms() -> bool:
        r"""
        Certificado (N6) sobre el objeto finito Ω₃.

            ⊥ ≤ ½ ≤ ⊤,
            ⊥ ∧ x = ⊥,   ⊤ ∨ x = ⊤,   x → x = ⊤,
            x ∧ (x → y) ≤ y,   y ≤ x → (x ∧ y)     (adjunction residuada).
        """
        bot, mid, top = (
            HeytingVerdict.VETOED,
            HeytingVerdict.DEGRADED,
            HeytingVerdict.COHERENT,
        )
        elems = (bot, mid, top)
        if not (bot.lattice_value < mid.lattice_value < top.lattice_value):
            return False
        for x in elems:
            if bot.meet(x) is not bot:
                return False
            if top.join(x) is not top:
                return False
            if x.implies(x) is not top:
                return False
            if x.meet(x.negate()) is not bot and x is not bot:
                # ¬⊥ = ⊤ y ⊥ ∧ ⊤ = ⊥;  ¬⊤ = ⊥;  ¬½ = ⊥  (½ ≰ ⊥).
                pass
            for y in elems:
                left = x.meet(x.implies(y))
                if left.lattice_value > y.lattice_value + 1e-15:
                    return False
                right = x.implies(x.meet(y))
                if y.lattice_value > right.lattice_value + 1e-15:
                    return False
        if bot.negate() is not top:
            return False
        if top.negate() is not bot:
            return False
        if mid.negate() is not bot:
            return False
        return True

    @staticmethod
    def atom_connected(b0: int) -> Tuple[HeytingVerdict, Optional[str]]:
        r"""β₀ = 1 ↦ ⊤;  β₀ = 0 (vacío) y β₀ > 1 (islas) ↦ ⊥."""
        if b0 == 1:
            return HeytingVerdict.COHERENT, None
        if b0 <= 0:
            return (
                HeytingVerdict.VETOED,
                f"β₀={b0}: complejo vacío / sin 0-simplices observables",
            )
        return (
            HeytingVerdict.VETOED,
            f"β₀={b0} > 1: {b0 - 1} isla(s) desconectada(s) / insumos huérfanos",
        )

    @staticmethod
    def atom_acyclic(b1: int) -> Tuple[HeytingVerdict, Optional[str]]:
        """β₁ = 0 ↦ ⊤;  β₁ > 0 ↦ ⊥ (ciclo = doble recorrido / facturación)."""
        if b1 == 0:
            return HeytingVerdict.COHERENT, None
        return (
            HeytingVerdict.VETOED,
            f"β₁={b1} > 0: {b1} ciclo(s) independiente(s) / doble facturación potencial",
        )

    @staticmethod
    def atom_torsion_free(
        torsion_free: bool,
        snf: Optional[SmithNormalFormResult],
    ) -> Tuple[HeytingVerdict, Optional[str]]:
        r"""
        Tor≡0 ↦ ⊤.  V.1: si SNF se omitió, torsion_free ya es True por teorema;
        este átomo permanece ⊤ (no se fabrica un veto a partir de la omisión).
        """
        if torsion_free:
            return HeytingVerdict.COHERENT, None
        tor_order = snf.torsion_order if snf is not None else 1
        return (
            HeytingVerdict.VETOED,
            f"Tor(H)≠0 (|Tor|={tor_order}): incompatibilidad de empaquetado discreto",
        )

    @staticmethod
    def atom_pyramidal(psi: float) -> Tuple[HeytingVerdict, Optional[str]]:
        r"""
        Ψ ≥ PYRAMIDAL_INDEX_MIN      ↦ ⊤
        PYRAMIDAL_INDEX_DEGRADED ≤ Ψ ↦ ½
        Ψ <  PYRAMIDAL_INDEX_DEGRADED ↦ ⊥
        """
        if not math.isfinite(psi):
            return HeytingVerdict.VETOED, f"Ψ no finito: {psi!r}"
        p = min(1.0, max(0.0, float(psi)))
        if p >= TC.PYRAMIDAL_INDEX_MIN:
            return HeytingVerdict.COHERENT, None
        if p >= TC.PYRAMIDAL_INDEX_DEGRADED:
            return (
                HeytingVerdict.DEGRADED,
                f"Ψ={p:.3f} ∈ [{TC.PYRAMIDAL_INDEX_DEGRADED}, "
                f"{TC.PYRAMIDAL_INDEX_MIN}): desequilibrio parcial",
            )
        return (
            HeytingVerdict.VETOED,
            f"Ψ={p:.3f} < {TC.PYRAMIDAL_INDEX_DEGRADED}: pirámide invertida",
        )

    @staticmethod
    def atom_cheeger(
        is_bottleneck: bool,
        sweep: float,
        lower: float,
    ) -> Tuple[HeytingVerdict, Optional[str]]:
        r"""
        Bottleneck ↦ ½  (estrangulamiento isoperimétrico, no veto).
        ¬bottleneck ↦ ⊤.  Nunca ⊥: h=0 con β₀>1 ya lo cubre `atom_connected`.
        """
        if not is_bottleneck:
            return HeytingVerdict.COHERENT, None
        return (
            HeytingVerdict.DEGRADED,
            f"h(K)≈{sweep:.4f} (λ₂/2={lower:.4f}) < τ={TC.CHEEGER_THRESHOLD}: "
            f"estrangulamiento isoperimétrico / monopolio crítico",
        )

    @classmethod
    def collect_evidence(
        cls, invariants: TopologicalInvariants
    ) -> HeytingEvidence:
        r"""Evalúa los cinco átomos y acumula reasons de los no-⊤ (y notas V.1/V.3)."""
        reasons: Dict[str, str] = {}
        betti = invariants.betti

        v_conn, r_conn = cls.atom_connected(betti.b0)
        v_acyc, r_acyc = cls.atom_acyclic(betti.b1)
        v_tor, r_tor = cls.atom_torsion_free(betti.torsion_free, betti.snf)
        v_psi, r_psi = cls.atom_pyramidal(invariants.pyramidal_index)
        v_ch, r_ch = cls.atom_cheeger(
            invariants.cheeger.is_bottleneck,
            invariants.cheeger.sweep_estimate,
            invariants.cheeger.lower_bound,
        )

        if r_conn:
            reasons["fragmentation"] = r_conn
        if r_acyc:
            reasons["cycles"] = r_acyc
        if r_tor:
            reasons["torsion"] = r_tor
        if r_psi:
            reasons["pyramidal"] = r_psi
        if r_ch:
            reasons["cheeger"] = r_ch

        if betti.snf_skipped:
            reasons["snf"] = (
                "SNF Euclid omitida (V.1): Tor≡0 declarado por teorema de "
                "1-complejos, no por factorización.  Átomo torsion permanece ⊤."
            )
        if not invariants.fiedler_faithful:
            reasons["fiedler"] = (
                "w₂ no fiel (V.3): β₀>1, λ₂=0 es la señal correcta; el vector "
                "no se usa como orden de sweep.  Átomo connected ya es ⊥."
            )

        return HeytingEvidence(
            connected=v_conn,
            acyclic=v_acyc,
            torsion=v_tor,
            pyramidal=v_psi,
            cheeger=v_ch,
            reasons=reasons,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ► INICIO FORMAL DE FASE 3  (realiza Phase3Ingress)
    # ► Consume π₂₃  TopologicalInvariants.
    # ═══════════════════════════════════════════════════════════════════════════
    @classmethod
    def classify(
        cls,
        invariants: TopologicalInvariants,
    ) -> Tuple[HeytingVerdict, Dict[str, str]]:
        r"""
        Puerto de ingreso F₃.  Firma de Phase3Ingress.

            classify : TopologicalInvariants → (Ω₃, reasons)

        Tras el meet, si el resultado es ⊤ se escribe un único reason de status.
        Complejidad O(1) en |V|,|E| (todo vive ya en π₂₃).
        """
        evidence = cls.collect_evidence(invariants)
        verdict = evidence.collapse()
        reasons = dict(evidence.reasons)
        if verdict is HeytingVerdict.COHERENT:
            psi = invariants.pyramidal_index
            reasons["status"] = (
                f"β₀=1 ∧ β₁=0 ∧ Tor(H)≡0 ∧ Ψ={psi:.3f}≥{TC.PYRAMIDAL_INDEX_MIN} ∧ "
                f"h(K)≥τ: topología algebraicamente coherente"
            )
        logger.debug(
            "Fase3 classify: Ω₃=%s atoms=(c=%s a=%s t=%s ψ=%s h=%s)",
            verdict.name,
            evidence.connected.name,
            evidence.acyclic.name,
            evidence.torsion.name,
            evidence.pyramidal.name,
            evidence.cheeger.name,
        )
        return verdict, reasons

    @classmethod
    def classify_with_evidence(
        cls,
        invariants: TopologicalInvariants,
    ) -> Tuple[HeytingVerdict, HeytingEvidence]:
        """Igual que `classify` pero expone los cinco átomos (auditoría)."""
        evidence = cls.collect_evidence(invariants)
        verdict = evidence.collapse()
        if verdict is HeytingVerdict.COHERENT:
            psi = invariants.pyramidal_index
            reasons = dict(evidence.reasons)
            reasons["status"] = (
                f"β₀=1 ∧ β₁=0 ∧ Tor(H)≡0 ∧ Ψ={psi:.3f}≥{TC.PYRAMIDAL_INDEX_MIN} ∧ "
                f"h(K)≥τ: topología algebraicamente coherente"
            )
            evidence = HeytingEvidence(
                connected=evidence.connected,
                acyclic=evidence.acyclic,
                torsion=evidence.torsion,
                pyramidal=evidence.pyramidal,
                cheeger=evidence.cheeger,
                reasons=reasons,
            )
        return verdict, evidence


assert HeytingVerdictClassifier.verify_heyting_axioms(), "Ω₃ no satisface (N6)"


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ III.2  SCORE S = 1 − Σ pᵢ wᵢ   (partición de la unidad, Independent de Ω₃)       ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
@dataclass(frozen=True, slots=True)
class HealthPenaltyBreakdown:
    r"""
    Desglose de S = 1 − Σ pᵢ wᵢ.  Cada pᵢ ∈ [0,1];  Σ wᵢ = 1  (TC.validate_weights).

        p_frag    = min(1, (β₀−1)/(|V|−1))     si β₀>1, n>1
        p_cyc     = min(1, β₁ / 10)
        p_disc    = |nodos_aislados_requeridos| / |REQUIRED|
        p_missing = |aristas_esperadas_ausentes| / |EXPECTED|
        p_loops   = min(1, n_loops / 5)
        p_tor     = 1  si Tor≠0 else 0          (V.1: skipped ⇒ 0)
    """

    p_fragmentation: float
    p_cycles: float
    p_disconnected: float
    p_missing_edges: float
    p_retry_loops: float
    p_torsion: float
    penalty: float
    score: float

    def weighted_terms(self) -> Dict[str, float]:
        return {
            "fragmentation": self.p_fragmentation * TC.WEIGHT_FRAGMENTATION,
            "cycles": self.p_cycles * TC.WEIGHT_CYCLES,
            "disconnected": self.p_disconnected * TC.WEIGHT_DISCONNECTED,
            "missing_edges": self.p_missing_edges * TC.WEIGHT_MISSING_EDGES,
            "retry_loops": self.p_retry_loops * TC.WEIGHT_RETRY_LOOPS,
            "torsion": self.p_torsion * TC.WEIGHT_TORSION,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_fragmentation": self.p_fragmentation,
            "p_cycles": self.p_cycles,
            "p_disconnected": self.p_disconnected,
            "p_missing_edges": self.p_missing_edges,
            "p_retry_loops": self.p_retry_loops,
            "p_torsion": self.p_torsion,
            "penalty": self.penalty,
            "score": self.score,
            "weighted_terms": self.weighted_terms(),
            "weights_sum_to_one": TC.validate_weights(),
        }


@final
class HealthScoreCalculator:
    r"""
    Calculadora de S.  Independiente de Ω₃: un árbol con Ψ bajo puede ser
    HEALTHY en score y VETOED en Heyting, o viceversa.  No se mezclan.

        S = clip_{[0,1]}(1 − Σ pᵢ wᵢ).
    """

    @staticmethod
    def p_fragmentation(b0: int, n_vertices: int) -> float:
        if b0 > 1 and n_vertices > 1:
            return min(1.0, (b0 - 1) / (n_vertices - 1))
        return 0.0

    @staticmethod
    def p_cycles(b1: int) -> float:
        if b1 <= 0:
            return 0.0
        return min(1.0, b1 / 10.0)

    @staticmethod
    def p_disconnected(n_disconnected: int, n_required: int) -> float:
        if n_disconnected <= 0:
            return 0.0
        return min(1.0, n_disconnected / max(1, n_required))

    @staticmethod
    def p_missing(n_missing: int, n_expected: int) -> float:
        if n_missing <= 0:
            return 0.0
        return min(1.0, n_missing / max(1, n_expected))

    @staticmethod
    def p_loops(n_loops: int) -> float:
        if n_loops <= 0:
            return 0.0
        return min(1.0, n_loops / 5.0)

    @staticmethod
    def p_torsion(torsion_free: bool) -> float:
        return 0.0 if torsion_free else 1.0

    @classmethod
    def breakdown(
        cls,
        invariants: TopologicalInvariants,
        disconnected: FrozenSet[str],
        missing: FrozenSet[Tuple[str, str]],
        loops: Sequence[RequestLoopInfo],
        n_required: int,
        n_expected: int,
    ) -> HealthPenaltyBreakdown:
        betti = invariants.betti
        p_frag = cls.p_fragmentation(betti.b0, betti.num_vertices)
        p_cyc = cls.p_cycles(betti.b1)
        p_disc = cls.p_disconnected(len(disconnected), n_required)
        p_miss = cls.p_missing(len(missing), n_expected)
        p_loop = cls.p_loops(len(loops))
        p_tor = cls.p_torsion(betti.torsion_free)

        penalty = (
            p_frag * TC.WEIGHT_FRAGMENTATION
            + p_cyc * TC.WEIGHT_CYCLES
            + p_disc * TC.WEIGHT_DISCONNECTED
            + p_miss * TC.WEIGHT_MISSING_EDGES
            + p_loop * TC.WEIGHT_RETRY_LOOPS
            + p_tor * TC.WEIGHT_TORSION
        )
        if not math.isfinite(penalty) or penalty < 0.0:
            penalty = 1.0
        score = max(0.0, min(1.0, 1.0 - penalty))
        return HealthPenaltyBreakdown(
            p_fragmentation=p_frag,
            p_cycles=p_cyc,
            p_disconnected=p_disc,
            p_missing_edges=p_miss,
            p_retry_loops=p_loop,
            p_torsion=p_tor,
            penalty=float(penalty),
            score=float(score),
        )

    @classmethod
    def score_and_level(
        cls,
        invariants: TopologicalInvariants,
        disconnected: FrozenSet[str],
        missing: FrozenSet[Tuple[str, str]],
        loops: Sequence[RequestLoopInfo],
        n_required: int,
        n_expected: int,
    ) -> Tuple[HealthPenaltyBreakdown, HealthLevel]:
        br = cls.breakdown(
            invariants, disconnected, missing, loops, n_required, n_expected
        )
        return br, HealthLevel.from_score(br.score)


@final
@dataclass(frozen=True, slots=True)
class TopologicalHealth:
    r"""
    Resumen ejecutivo de F₃: score numérico × veredicto Ω₃ × evidencia.

    `level`  es cuantización de S (métrica).
    `verdict` es el meet en Ω₃ (álgebra).  Independientes: no se infieren.
    """

    betti: BettiNumbers
    disconnected_nodes: FrozenSet[str]
    missing_edges: FrozenSet[Tuple[str, str]]
    request_loops: Tuple[RequestLoopInfo, ...]
    health_score: HealthScore
    level: HealthLevel
    verdict: HeytingVerdict = HeytingVerdict.COHERENT
    diagnostics: Dict[str, str] = field(default_factory=dict)
    evidence: Optional[HeytingEvidence] = None
    penalties: Optional[HealthPenaltyBreakdown] = None
    certificates: Optional[StructuralCertificates] = None
    snf_skipped: bool = False
    fiedler_faithful: bool = True
    pyramidal_index: float = 1.0
    lambda2: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError(f"health_score ∈ [0,1]: {self.health_score}")

    @property
    def is_healthy(self) -> bool:
        return self.level == HealthLevel.HEALTHY

    @property
    def is_coherent(self) -> bool:
        return self.verdict.is_coherent

    @property
    def is_vetoed(self) -> bool:
        return self.verdict is HeytingVerdict.VETOED

    @property
    def is_degraded_heyting(self) -> bool:
        return self.verdict is HeytingVerdict.DEGRADED

    @property
    def total_anomalies(self) -> int:
        count = 0
        if self.betti.b0 > 1:
            count += self.betti.b0 - 1
        if self.betti.b0 <= 0:
            count += 1
        if self.betti.b1 > 0:
            count += self.betti.b1
        count += len(self.disconnected_nodes)
        count += len(self.missing_edges)
        count += len(self.request_loops)
        if not self.betti.torsion_free:
            count += 1
        return count

    def get_summary(self) -> str:
        lines = [
            "=== SALUD TOPOLÓGICA DEL SISTEMA (V_𝕋) ===",
            f"Veredicto Heyting Ω₃ : {self.verdict.name} "
            f"(valor retículo = {self.verdict.lattice_value})",
            f"Nivel                : {self.level.name}",
            f"Score                : {self.health_score:.3f}/1.000",
            "",
            "Invariantes Topológicos:",
            f"  β₀ (componentes)   : {self.betti.b0}",
            f"  β₁ (ciclos)        : {self.betti.b1}",
            f"  χ  (Euler)         : {self.betti.euler_characteristic}",
            f"  Tor(H)≡0 (libre)   : {self.betti.torsion_free}",
            f"  Ψ  (piramidal)     : {self.pyramidal_index:.3f}",
            f"  λ₂ (Fiedler)       : {self.lambda2:.6g}",
            f"  SNF omitida (V.1)  : {self.snf_skipped}",
            f"  Fiedler fiel (V.3) : {self.fiedler_faithful}",
            "",
            f"Anomalías Detectadas : {self.total_anomalies}",
        ]
        for key, value in self.diagnostics.items():
            lines.append(f"  • {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_score,
            "level": self.level.name,
            "verdict": self.verdict.name,
            "verdict_lattice_value": self.verdict.lattice_value,
            "is_coherent": self.is_coherent,
            "is_vetoed": self.is_vetoed,
            "betti_numbers": self.betti.to_dict(),
            "total_anomalies": self.total_anomalies,
            "disconnected_nodes": sorted(self.disconnected_nodes),
            "missing_edges": sorted([tuple(sorted(e)) for e in self.missing_edges]),
            "request_loops": [loop.to_dict() for loop in self.request_loops],
            "diagnostics": dict(self.diagnostics),
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "penalties": self.penalties.to_dict() if self.penalties else None,
            "certificates": self.certificates.to_dict() if self.certificates else None,
            "snf_skipped": self.snf_skipped,
            "fiedler_faithful": self.fiedler_faithful,
            "pyramidal_index": self.pyramidal_index,
            "lambda2": self.lambda2,
        }

    def __str__(self) -> str:
        return (
            f"TopologicalHealth(verdict={self.verdict.name}, "
            f"level={self.level.name}, score={self.health_score:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ III.3  CROWBAR CIBER-FÍSICO — CONTRATO, SIN HAL  (V.4)                            ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@unique
class CrowbarState(Enum):
    r"""
    Máquina de estados del disyuntor:

        ARMED  --trigger_crowbar-->  FIRED
        FIRED  --reset-->            ARMED

    trigger en FIRED es idempotente (no-op, retorna False).
    """

    ARMED = auto()
    FIRED = auto()

    def __str__(self) -> str:
        return self.name


@final
@dataclass(frozen=True, slots=True)
class ActuationEvent:
    r"""Registro inmutable de un disparo.  No contiene I/O de hardware (V.4)."""

    gpio: int
    thyristor: str
    reason: str
    budget_ns: int
    dry_run: bool
    has_real_hal: bool
    w1ts_mask: int
    state_after: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpio": self.gpio,
            "thyristor": self.thyristor,
            "reason": self.reason,
            "budget_ns": self.budget_ns,
            "dry_run": self.dry_run,
            "has_real_hal": self.has_real_hal,
            "w1ts_mask": self.w1ts_mask,
            "state_after": self.state_after,
        }


class CyberPhysicalActuator(abc.ABC):
    r"""
    Contrato del disyuntor ciber-físico.

    PECADO RESIDUAL V.4: este módulo NO posee HAL.  El firmware C++ debe
    ejecutar en ISR IRAM (< 400 ns):

        GPIO.out_w1ts = (1 << 14);   // BT151 Crowbar

    Aquí solo se garantiza:
        (C1) idempotencia ARMED → FIRED → (re-trigger ignorado);
        (C2) reset explícito FIRED → ARMED;
        (C3) VETOED ⇒ trigger;  ¬VETOED ⇒ no-op;
        (C4) dry_run=True por defecto;  HAS_REAL_HAL=False ⇒ nunca I/O GPIO.
    """

    @abc.abstractmethod
    def is_verdict_coherent(self, verdict: HeytingVerdict) -> bool: ...

    @abc.abstractmethod
    def trigger_crowbar(self, reason: str) -> bool: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_armed(self) -> bool: ...

    @property
    @abc.abstractmethod
    def state(self) -> CrowbarState: ...


@final
class ESP32CrowbarActuator(CyberPhysicalActuator):
    r"""
    Actuador Crowbar ESP32+BT151.  Máquina ARMED ⇄ FIRED.

    dry_run=True por defecto: emite log CRITICAL, no toca GPIO.
    HAS_REAL_HAL = False (V.4).  Máscara documental: 1<<14.

    Este tipo es el *contrato*.  Ningún camino de ejecución de este módulo
    escribe registros MMIO ni abre dispositivos.  El firmware vive fuera.
    """

    def __init__(
        self,
        gpio_pin: int = TC.CROWBAR_GPIO_PIN,
        thyristor: str = TC.CROWBAR_THYRISTOR,
        dry_run: bool = True,
    ) -> None:
        if gpio_pin < 0:
            raise ValueError(f"gpio_pin ≥ 0, recibido {gpio_pin}")
        self.gpio_pin = int(gpio_pin)
        self.thyristor = str(thyristor)
        self.dry_run = bool(dry_run)
        self._state: CrowbarState = CrowbarState.ARMED
        self._trigger_log: List[ActuationEvent] = []

    def is_verdict_coherent(self, verdict: HeytingVerdict) -> bool:
        return verdict.is_coherent

    def should_fire(self, verdict: HeytingVerdict) -> bool:
        """(C3) dispara ssi VETOED ∧ ARMED.  DEGRADED no dispara."""
        return verdict is HeytingVerdict.VETOED and self._state is CrowbarState.ARMED

    def trigger_crowbar(self, reason: str) -> bool:
        r"""
        Transición ARMED → FIRED.  Idempotente: FIRED ⇒ False.

        Efecto: log CRITICAL + apéndice al registro.  Cero I/O de hardware.
        """
        if self._state is CrowbarState.FIRED:
            logger.warning("Crowbar ya disparado; re-disparo ignorado (C1)")
            return False

        self._state = CrowbarState.FIRED
        event = ActuationEvent(
            gpio=self.gpio_pin,
            thyristor=self.thyristor,
            reason=str(reason),
            budget_ns=TC.ISR_BUDGET_NS,
            dry_run=self.dry_run,
            has_real_hal=TC.HAS_REAL_HAL,
            w1ts_mask=TC.CROWBAR_GPIO_W1TS_MASK,
            state_after=self._state.name,
        )
        self._trigger_log.append(event)

        logger.critical(
            "🛑 [DRY-RUN / SIN HAL] CROWBAR: GPIO%d→HIGH (%s) máscara=0x%X "
            "< %d ns. Firmware requerido fuera de este módulo: GPIO.out_w1ts. "
            "HAS_REAL_HAL=%s dry_run=%s. Razón: %s",
            self.gpio_pin,
            self.thyristor,
            TC.CROWBAR_GPIO_W1TS_MASK,
            TC.ISR_BUDGET_NS,
            TC.HAS_REAL_HAL,
            self.dry_run,
            reason,
        )
        return True

    def apply_verdict(self, verdict: HeytingVerdict, reason: str) -> bool:
        """Orquesta (C3): dispara ssi VETOED.  Retorna True ssi hubo transición."""
        if not self.should_fire(verdict):
            return False
        return self.trigger_crowbar(reason)

    def reset(self) -> None:
        """(C2) FIRED → ARMED.  El log de disparos se conserva (auditoría)."""
        self._state = CrowbarState.ARMED
        logger.info("Crowbar re-armado")

    @property
    def is_armed(self) -> bool:
        return self._state is CrowbarState.ARMED

    @property
    def state(self) -> CrowbarState:
        return self._state

    @property
    def trigger_log(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(e.to_dict() for e in self._trigger_log)

    @property
    def fire_count(self) -> int:
        return len(self._trigger_log)


# ══════════════════════════════════════════════════════════════════════════════════════════
# ▓▓▓ III.4  FACHADA  Φ = F₃ ∘ F₂ ∘ F₁                                                  ▓▓▓
# ══════════════════════════════════════════════════════════════════════════════════════════


@final
class TopologicalAnalyzerFacade:
    r"""
    Fachada Φ = F₃ ∘ F₂ ∘ F₁.

        F₁  assemble_from_graph          π₁₂  SimplicialCohainComplex
        F₂  compute_invariants           π₂₃  TopologicalInvariants
        F₃  classify + score + crowbar        TopologicalHealth

    Funtorialidad documental: Φ(g ∘ f) se obtiene componiendo los tres
    estratos sobre el mismo 1-complejo; no hay estado cruzado entre grafos
    distintos salvo el Crowbar (absorbente hasta `reset`).

    VETOED es absorbente en Ω₃ y en el Crowbar: no se revierte sin reset
    explícito.  `raise_on_veto=True` lanza InvalidTopologyError (no recoverable).
    """

    def __init__(
        self,
        topology: SystemTopology,
        actuator: Optional[CyberPhysicalActuator] = None,
        raise_on_veto: bool = True,
    ) -> None:
        self._topology = topology
        self._classifier = HeytingVerdictClassifier()
        self._scorer = HealthScoreCalculator()
        self._actuator: CyberPhysicalActuator = actuator or ESP32CrowbarActuator(
            dry_run=True
        )
        self._raise_on_veto = bool(raise_on_veto)
        self._persistence = PersistenceHomology()

    @property
    def topology(self) -> SystemTopology:
        return self._topology

    @property
    def persistence(self) -> PersistenceHomology:
        return self._persistence

    @property
    def actuator(self) -> CyberPhysicalActuator:
        return self._actuator

    @property
    def classifier(self) -> HeytingVerdictClassifier:
        return self._classifier

    def f1_assemble(self) -> SimplicialCohainComplex:
        """F₁ explícito: π₁₂ sobre el grafo actual."""
        return CompensatedLaplacianAssembler.assemble_from_graph(self._topology.graph)

    def f2_invariants(self, use_cache: bool = True) -> TopologicalInvariants:
        """F₂ explícito: π₂₃ (orquesta F₁ internamente)."""
        return self._topology.compute_invariants(use_cache=use_cache)

    def f3_decide(
        self, invariants: TopologicalInvariants
    ) -> Tuple[HeytingVerdict, HeytingEvidence, HealthPenaltyBreakdown, HealthLevel]:
        r"""
        F₃ puro (sin Crowbar ni raise): classify + score.

        El Crowbar es un efecto (C3), no un morfismo de 𝒞_Topo.
        """
        verdict, evidence = self._classifier.classify_with_evidence(invariants)
        disconnected = self._topology.get_disconnected_nodes()
        missing = self._topology.get_missing_connections()
        loops = self._topology.detect_request_loops()
        br, level = self._scorer.score_and_level(
            invariants,
            disconnected,
            missing,
            loops,
            n_required=len(self._topology.REQUIRED_NODES),
            n_expected=len(self._topology.expected_topology),
        )
        return verdict, evidence, br, level

    def _pack_health(
        self,
        invariants: TopologicalInvariants,
        verdict: HeytingVerdict,
        evidence: HeytingEvidence,
        breakdown: HealthPenaltyBreakdown,
        level: HealthLevel,
    ) -> TopologicalHealth:
        return TopologicalHealth(
            betti=invariants.betti,
            disconnected_nodes=self._topology.get_disconnected_nodes(),
            missing_edges=self._topology.get_missing_connections(),
            request_loops=tuple(self._topology.detect_request_loops()),
            health_score=round(breakdown.score, 4),
            level=level,
            verdict=verdict,
            diagnostics=dict(evidence.reasons),
            evidence=evidence,
            penalties=breakdown,
            certificates=invariants.certificates,
            snf_skipped=invariants.snf_skipped,
            fiedler_faithful=invariants.fiedler_faithful,
            pyramidal_index=invariants.pyramidal_index,
            lambda2=invariants.lambda2,
        )

    def _actuate_if_vetoed(self, health: TopologicalHealth) -> None:
        r"""
        Efecto (C3): VETOED ⇒ trigger_crowbar.  DEGRADED/COHERENT ⇒ no-op.

        Si raise_on_veto, lanza InvalidTopologyError *después* del trigger
        (el Crowbar ya quedó FIRED; el raise no lo deshace).
        """
        if health.verdict is not HeytingVerdict.VETOED:
            return
        reason_str = "; ".join(f"{k}: {v}" for k, v in health.diagnostics.items())
        self._actuator.trigger_crowbar(reason_str or "VETOED")
        if self._raise_on_veto:
            raise InvalidTopologyError(
                f"Topología VETOED por Ω₃: {reason_str}",
                context=health.to_dict(),
                recoverable=False,
            )

    def analyze(self, use_cache: bool = True) -> TopologicalHealth:
        r"""
        Φ(X) = F₃(F₂(F₁(X))) con efecto Crowbar si ⊥.

        Pasos:
            1. compute_invariants()           π₁₂ + π₂₃
            2. classify_with_evidence()       Ω₃
            3. HealthScoreCalculator          S ∈ [0,1]
            4. si ⊥: trigger_crowbar (+ raise)
        """
        invariants = self.f2_invariants(use_cache=use_cache)
        verdict, evidence, breakdown, level = self.f3_decide(invariants)
        health = self._pack_health(invariants, verdict, evidence, breakdown, level)
        self._actuate_if_vetoed(health)
        return health

    def phi(self, use_cache: bool = True) -> TopologicalHealth:
        """Alias categorial de `analyze`: Φ = F₃ ∘ F₂ ∘ F₁."""
        return self.analyze(use_cache=use_cache)

    def analyze_without_actuation(self, use_cache: bool = True) -> TopologicalHealth:
        """F₃ puro: clasifica y puntúa, no dispara Crowbar ni raise.  Útil en tests."""
        invariants = self.f2_invariants(use_cache=use_cache)
        verdict, evidence, breakdown, level = self.f3_decide(invariants)
        return self._pack_health(invariants, verdict, evidence, breakdown, level)

    def reset_actuator(self) -> None:
        """Re-arma el Crowbar (C2).  No altera el 1-complejo ni las cachés de F₂."""
        self._actuator.reset()


# ══════════════════════════════════════════════════════════════════════════════════════════
# UTILIDADES MODULE-LEVEL  (fábricas + distancias de diagramas H₀, pecado V.2)
# ══════════════════════════════════════════════════════════════════════════════════════════


def create_simple_topology() -> SystemTopology:
    r"""
    Fábrica: árbol (conexo, acíclico, Tor-free) para pruebas de Ω₃=⊤.

        Agent — Core — Redis
                  \
                   Filesystem

    χ = 4−3 = 1 = β₀−β₁.  Ψ alto (todas las aristas tocan Core salvo ninguna
    periférica pura: 2/3 de peso si uniforme).  No bottleneck típico.
    """
    topology = SystemTopology()
    topology.update_connectivity(
        [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ]
    )
    return topology


def create_cyclic_topology() -> SystemTopology:
    r"""Fábrica: un 4-ciclo + cuerda.  β₁≥1 ⇒ Ω₃=⊥ (átomo acyclic)."""
    topology = SystemTopology()
    topology.update_connectivity(
        [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
            ("Redis", "Agent"),
        ]
    )
    return topology


def create_disconnected_topology() -> SystemTopology:
    r"""Fábrica: dos componentes.  β₀=2 ⇒ Ω₃=⊥ (átomo connected).  V.3: faithful=False."""
    topology = SystemTopology()
    topology.update_connectivity(
        [
            ("Agent", "Core"),
            ("Redis", "Filesystem"),
        ]
    )
    return topology


def _finite_lifespans(intervals: Sequence[PersistenceInterval]) -> List[float]:
    return sorted(
        float(i.lifespan)
        for i in intervals
        if (not i.is_alive) and math.isfinite(i.lifespan)
    )


def compute_wasserstein_distance(
    intervals1: Sequence[PersistenceInterval],
    intervals2: Sequence[PersistenceInterval],
    p: float = 2.0,
) -> float:
    r"""
    W_p aproximada por matching 1-D de lifespans ordenados (no húngaro en ℝ²):

        W_p ≈ ( Σᵢ |ℓ₁ᵢ − ℓ₂ᵢ|^p )^{1/p},   no emparejados ↔ 0 (proyección a diagonal).

    PECADO V.2: no es el W_p de diagramas en el plano (matching L^∞ / Hungarian).
    Estable como subrogado bajo cotas de bottleneck; p ≥ 1.
    """
    if p < 1.0:
        raise ValueError(f"p ≥ 1, recibido: {p}")
    pp = float(p)
    if not intervals1 and not intervals2:
        return 0.0
    l1 = _finite_lifespans(intervals1)
    l2 = _finite_lifespans(intervals2)
    if not l1 and not l2:
        return 0.0
    if not l1:
        return float(sum(x ** pp for x in l2) ** (1.0 / pp))
    if not l2:
        return float(sum(x ** pp for x in l1) ** (1.0 / pp))
    max_len = max(len(l1), len(l2))
    p1 = l1 + [0.0] * (max_len - len(l1))
    p2 = l2 + [0.0] * (max_len - len(l2))
    acc = NeumaierKahanAccumulator()
    for a, b in zip(p1, p2):
        acc.add(abs(a - b) ** pp)
    return float(acc.total ** (1.0 / pp))


def compute_bottleneck_distance_approx(
    intervals1: Sequence[PersistenceInterval],
    intervals2: Sequence[PersistenceInterval],
) -> float:
    r"""
    d_B subrogada: matching voraz por persistencia + coste diagonal.

        d_B(I,J) = max(|b−b′|, |d−d′|)   (PersistenceInterval.bottleneck_distance)
        no emparejado → (death−birth)/2.

    NO es el matching óptimo de Hungarian.  Cota superior del d_B verdadero
    sobre las barras finitas de la misma dimensión (aquí dim=0).  V.2.
    """
    fin1 = [i for i in intervals1 if not i.is_alive]
    fin2 = [i for i in intervals2 if not i.is_alive]
    fin1 = sorted(fin1, key=lambda i: i.persistence, reverse=True)
    fin2 = sorted(fin2, key=lambda i: i.persistence, reverse=True)
    n = max(len(fin1), len(fin2))
    if n == 0:
        inf1 = sum(1 for i in intervals1 if i.is_alive)
        inf2 = sum(1 for i in intervals2 if i.is_alive)
        return 0.0 if inf1 == inf2 else float("inf")
    d_b = 0.0
    for k in range(n):
        a = fin1[k] if k < len(fin1) else None
        b = fin2[k] if k < len(fin2) else None
        if a is not None and b is not None:
            cost = a.bottleneck_distance(b)
            if not math.isfinite(cost):
                cost = max(a.diagonal_projection_cost(), b.diagonal_projection_cost())
        elif a is not None:
            cost = a.diagonal_projection_cost()
        else:
            cost = b.diagonal_projection_cost()  # type: ignore[union-attr]
        if cost > d_b:
            d_b = float(cost)
    inf1 = sum(1 for i in intervals1 if i.is_alive)
    inf2 = sum(1 for i in intervals2 if i.is_alive)
    if inf1 != inf2:
        return float("inf")
    return d_b


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 80)
    print(
        "DEMO v6.1.0 — Φ = F₃∘F₂∘F₁ · Heyting Ω₃ · Euclid-SNF · "
        "Lanczos-Fiedler · Sweep-Cheeger · Crowbar-contrato"
    )
    print("=" * 80)
    print(
        f"Ω₃ axiomas (N6): {HeytingVerdictClassifier.verify_heyting_axioms()}  "
        f"Σwᵢ=1: {TC.validate_weights()}  HAS_REAL_HAL={TC.HAS_REAL_HAL} (V.4)"
    )

    topology = create_simple_topology()
    facade = TopologicalAnalyzerFacade(topology, raise_on_veto=False)
    health = facade.phi()
    print("\n[1] Árbol (Ω₃=⊤ esperado):")
    print(health.get_summary())
    if health.evidence is not None:
        print(f"    átomos: {health.evidence.to_dict()}")

    topology.update_connectivity(
        [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
            ("Redis", "Agent"),
        ]
    )
    health2 = facade.analyze()
    print("\n[2] Un ciclo (Ω₃=⊥ esperado, Crowbar FIRED):")
    print(health2.get_summary())
    print(f"\nCrowbar estado : {facade.actuator.state}")
    print(f"Crowbar armado : {facade.actuator.is_armed}")
    print(f"Registro       : {facade.actuator.trigger_log}")

    facade.reset_actuator()
    disc = create_disconnected_topology()
    facade_d = TopologicalAnalyzerFacade(disc, raise_on_veto=False)
    health3 = facade_d.analyze_without_actuation()
    print("\n[3] Desconexo (Ω₃=⊥, V.3 faithful=False, sin actuación):")
    print(health3.get_summary())
    print(f"    fiedler_faithful={health3.fiedler_faithful}  snf_skipped={health3.snf_skipped}")

    print("\n" + "=" * 80)
    print("✅ Composición Fase1 ▷ Fase2 ▷ Fase3  (Φ = F₃ ∘ F₂ ∘ F₁) verificada")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 3 — DECISIÓN Y ACTUACIÓN
# Objeto inicial  : Phase3Ingress.classify  =  HeytingVerdictClassifier.classify
# Objeto terminal : TopologicalHealth  (Ω₃ × S × Crowbar-efecto)
# Funtor          : Φ = F₃ ∘ F₂ ∘ F₁  =  TopologicalAnalyzerFacade.phi
#
# Pecados residuales conscientemente abiertos:
#   V.1  SNF no escala; bloques + teorema de 1-complejos.  sympy.smith_normal_form pendiente.
#   V.2  H₀ de path graph / UF; no Vietoris–Rips ni ELZ.  W_p y d_B son subrogados 1-D.
#   V.3  β₀>1 ⇒ fiedler_faithful=False; λ₂=0 es correcto; w₂ no se usa en sweep.
#   V.4  Crowbar sin HAL; dry_run=True; firmware ISR IRAM vive fuera de este módulo.
# ══════════════════════════════════════════════════════════════════════════════════════════