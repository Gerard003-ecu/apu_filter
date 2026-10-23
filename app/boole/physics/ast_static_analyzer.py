# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : AST Static Analyzer (Analizador Simpléctico y Cohomología de Haz)   ║
║ Ruta   : app/boole/physics/ast_static_analyzer.py                            ║
║ Versión: 5.1.0-RSI3-Poincare-KAM-Heyting-OODA                                ║
║ Función: Gobernanza OODA de física sintáctica en variedad simpléctica.       ║
╚══════════════════════════════════════════════════════════════════════════════╝

MEJORAS v5.1.0 RESPECTO A v5.0.0 (retrocompatibles, granulares, rigurosas):
─────────────────────────────────────────────────────────────────────────
FASE 1 — Geometría estructural (0-jet) + semilla celeste:
  • Forma de interferencia antisimétrica I(A,B) materializada como 2-forma
    ω_I ∈ Ω²(T*Q) y verificada (axioma de antisimetría) en el handoff.
  • Coordenadas acción-ángulo semilla (ω_i) con twist de Kolmogorov.
  • Flujo Hamiltoniano linealizado A ∈ 𝔰𝔭(2n) a partir de (reads, writes).
  • Handoff formal a FASE 2: `Phase1Handoff` ahora transporta
    `frequencies`, `hamiltonian_generator`, `interference_frobenius`,
    `antisymmetry_ok` — dominio estricto de Φ₂.
FASE 2 — Homología de Poincaré + RSI³ Banach (continuación estricta):
  • `_ingest_phase1_handoff` consume la semilla; NUNCA matrices aleatorias.
  • `construct_flow_map_from_sp_generator` : M = exp(εA) ∈ Sp(2n).
  • `action_angle_from_orbit` + `kolmogorov_nondegeneracy` (det ∂²H₀/∂I²).
  • `poincare_return_map` sobre la sección Σ = {q = ⟨q⟩}.
  • Ecuación homológica de Poincaré-Lindstedt resuelta en el pipeline.
  • `RecursiveSelfImprovementL3`: contracción de Banach de nivel 3
    sobre el espacio de operadores de mejora (no un contador).
  • Haz celular con mapas de restricción ρ_{v→e} (no haz constante).
FASE 3 — KAM / Melnikov / Hodge-Dirac / certificado Ω₃:
  • Laplaciano de Hodge-Dirac sobre C⁰ ⊕ C¹ con métricas G₀, G₁
    construidas desde la coborde REAL del haz (no ruido gaussiano).
  • Remainder de Birkhoff + tiempo de recurrencia de Poincaré.
  • Certificado con trayectoria RSI³ y axiomas de retículo Heyting.

ESTRUCTURA (fases anidadas — morfismos estrictos):
  §0      — Common (constantes, excepciones, HeytingVerdict, handoffs).
  FASE 1  — Geometría estructural (DataFlow, Complexity, Visitor).
            Última def.: `_ASTPhase1Pipeline._handoff_to_phase2`.
  FASE 2  — Homología de Poincaré y RSI³.
            Primera def.: `_ASTPhase2Pipeline._ingest_phase1_handoff`.
            Última def.: `_ASTPhase2Pipeline._handoff_to_phase3`.
  FASE 3  — KAM, Melnikov, Hodge-Dirac y certificado Ω₃.
            Primera def.: `_ASTPhase3Pipeline._ingest_phase2_handoff`.
            Punto de entrada público: `ASTStaticAnalyzer`.
"""

from __future__ import annotations

import ast
import logging
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations, product
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.linalg as la

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Γ.Physics.AST.v51")

__version__: str = "5.1.0"

# Tipo de Hamiltoniano escalar H: T*Q → ℝ  (q, p) ↦ H(q, p)
HamiltonianFn = Callable[[float, float], float]


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  §0 — COMÚN: CONSTANTES, EXCEPCIONES, HEYTING Ω₃, HANDOFFS                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class AnalysisLimits:
    r"""
    Límites algorítmicos con justificación teórica.

    • `MAX_AST_DEPTH`: profundidad típica O(log n) bien estructurado; límite
      práctico contra stack overflow. Ref: Aho–Sethi–Ullman (1986).
    • `MAX_CYCLOMATIC_COMPLEXITY`: ISO 26262 (automotive). Ref: McCabe (1976).
    • `MAX_JSON_DEPTH`: prevención de ataques Billion Laughs.
    • `HAMILTONIAN_EPSILON`, `FLOAT_TOLERANCE`: regularizadores IEEE-754.
    • `MAX_PATH_ENERGY`, `ENERGY_DECAY_SCALE`: bound de disipación
      Hamiltoniana de path-energy.
    • [v5.1.0] `GOLDEN_RATIO`, `POINCARE_SECTION_ATOL`, `RSI3_LIPSCHITZ_TARGET`:
      constantes de mecánica celeste y contracción de Banach de nivel 3.
    """

    MAX_AST_DEPTH: int = 50
    MAX_AST_NODES: int = 10_000

    MAX_CYCLOMATIC_COMPLEXITY: int = 20
    WARN_CYCLOMATIC_COMPLEXITY: int = 10
    CRITICAL_CYCLOMATIC_COMPLEXITY: int = 50

    MAX_JSON_DEPTH: int = 10
    MAX_JSON_KEYS: int = 100
    MAX_JSON_ARRAY_SIZE: int = 1_000

    MAX_FLATTEN_DEPTH: int = 3
    MAX_STRING_REPR_LENGTH: int = 100

    HAMILTONIAN_EPSILON: float = 1e-9
    FLOAT_TOLERANCE: float = 1e-12

    MAX_INTERFERENCE_MATRIX_SIZE: int = 1_000

    MAX_PATH_ENERGY: float = 30.0
    ENERGY_DECAY_SCALE: float = 10.0

    KAM_GAMMA: float = 1.0e-3
    KAM_TAU: float = 2.0
    KAM_MAX_K_NORM: int = 4
    BANACH_TARGET: float = 0.95
    RSI3_MAX_ITER: int = 20
    RSI3_MIN_ITER: int = 3
    RSI3_CONV_EPS: float = 1.0e-9
    RSI3_LIPSCHITZ_TARGET: float = 0.85

    SYMPLECTIC_ATOL: float = 1.0e-9
    MELNIKOV_THRESHOLD: float = 1.0e-2
    HODGE_LAMBDA_MIN_THRESHOLD: float = 1.0e-9

    GOLDEN_RATIO: float = 1.618033988749895
    POINCARE_SECTION_ATOL: float = 1.0e-9
    FLOW_MAP_EPSILON: float = 0.1
    BIRKHOFF_ORDER: int = 4
    MAX_SYMPLECTIC_DIM: int = 4


class AnalysisException(Exception):
    """Clase base para todas las excepciones de análisis."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.mathematical_interpretation = ""

    def __str__(self) -> str:
        base = super().__str__()
        if self.mathematical_interpretation:
            return f"{base}\n  Mathematical: {self.mathematical_interpretation}"
        return base


class ThermodynamicSingularityError(AnalysisException):
    r"""
    Violación de la disipación port-Hamiltoniana.

    Condición formal (path-energy):
        E(n) = Σ_{a ∈ Path(n)} CC_incr(a) · exp(−depth(a)/D₀) > MAX_PATH_ENERGY.

    Interpretación: el código concentra demasiada complejidad decisional en
    nodos superficiales (estructura "top-heavy"), indicando alta densidad de
    defectos y baja mantenibilidad.
    """

    def __init__(self, path_energy: float, threshold: float, depth: int, cc: int):
        ctx = {
            "path_energy": path_energy,
            "threshold": threshold,
            "depth": depth,
            "cyclomatic_complexity": cc,
        }
        msg = (
            f"Thermodynamic singularity: E(path) = {path_energy:.6f} > "
            f"{threshold:.6f}\n  At depth={depth}, CC={cc}"
        )
        super().__init__(msg, ctx)
        self.mathematical_interpretation = (
            "La energía de trayectoria excede el bound disipativo: "
            "complejidad decisional top-heavy."
        )


class CohomologicalObstructionError(AnalysisException):
    r"""
    Obstrucción topológica global: H¹(T, F) ≠ 0. Ciclos de dependencias
    globalmente inconsistentes (variables no inicializadas, deadlocks).
    """

    def __init__(
        self,
        cohomology_dimension: int,
        problematic_variables: Optional[Set[str]] = None,
    ):
        ctx = {
            "dim_H1": cohomology_dimension,
            "problematic_vars": problematic_variables or set(),
        }
        msg = (
            f"Cohomological obstruction: dim H¹(T,F) = {cohomology_dimension}\n"
            f"  Implies {cohomology_dimension} independent inconsistency cycles"
        )
        if problematic_variables:
            msg += f"\n  Problematic variables: {sorted(problematic_variables)}"
        super().__init__(msg, ctx)
        self.mathematical_interpretation = (
            f"dim H¹ = {cohomology_dimension} > 0: ciclos globalmente inconsistentes."
        )


class ComplexityBoundsViolationError(AnalysisException):
    def __init__(self, metric: str, value: int, limit: int):
        super().__init__(
            f"{metric} = {value} exceeds limit {limit}",
            {"metric": metric, "value": value, "limit": limit},
        )


class StructuralValidationError(AnalysisException):
    pass


class BanachContractionFailure(AnalysisException):
    r"""RSI³ no contrajo: Lip(T) ≥ 1, el operador de mejora no es contractivo."""

    def __init__(self, lipschitz: float, iterations: int):
        super().__init__(
            f"RSI³ Banach failure: Lip(T) = {lipschitz:.6f} ≥ 1 after {iterations} iterates",
            {"lipschitz": lipschitz, "iterations": iterations},
        )
        self.mathematical_interpretation = (
            "El operador de automejora T: B → B no es una contracción; "
            "no existe punto fijo único por el teorema de Banach."
        )


class HeytingVerdict(Enum):
    r"""Clasificador en el álgebra de Heyting trivalente $\Omega_3$.

    Orden: COHERENT=⊥ < DEGRADED=½ < VETOED=⊤. Join = OR bitwise (SIMD).
    """

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.COHERENT: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.VETOED: 1.0,
        }[self]

    @property
    def mask(self) -> int:
        return {
            HeytingVerdict.COHERENT: 0x00,
            HeytingVerdict.DEGRADED: 0x01,
            HeytingVerdict.VETOED: 0x03,
        }[self]

    @classmethod
    def from_omega(cls, v: float) -> "HeytingVerdict":
        if v <= 0.0:
            return cls.COHERENT
        if v < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def from_mask(cls, m: int) -> "HeytingVerdict":
        if (m & 0x03) == 0x03:
            return cls.VETOED
        if (m & 0x01) == 0x01:
            return cls.DEGRADED
        return cls.COHERENT

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_mask(self.mask | other.mask)

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_omega(min(self.omega, other.omega))

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingVerdict"]) -> "HeytingVerdict":
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)


# ── Handoff dataclasses (v5.1.0: transporte geométrico real entre fases) ─────

@dataclass(frozen=True)
class Phase1Handoff:
    r"""
    Morfismo FASE 1 → FASE 2: 0-jet geométrico del AST + semilla celeste.

    [v5.1.0] Transporta las frecuencias de acción-ángulo, el generador
    $A \in \mathfrak{sp}(2n)$ y la norma de Frobenius de la 2-forma de
    interferencia, de modo que Φ₂ no sintetiza ruido: Φ₂ = Φ₂(Φ₁(source)).
    """

    source_code: str
    filename: str
    dataflow_reads: FrozenSet[str]
    dataflow_writes: FrozenSet[str]
    cyclomatic_complexity: int
    max_nesting_depth: int
    num_functions: int
    num_classes: int
    num_branches: int
    num_loops: int
    block_flows: Tuple[Tuple[FrozenSet[str], FrozenSet[str], int, str], ...]
    tree_graph: Tuple[Tuple[int, Tuple[int, ...]], ...]
    hamiltonian_ok: bool
    max_path_energy: float
    warnings: Tuple[str, ...]
    # v5.1.0 — semilla de mecánica celeste
    frequencies: Tuple[float, ...]
    hamiltonian_generator: Tuple[Tuple[float, ...], ...]
    interference_frobenius: float
    antisymmetry_ok: bool
    symplectic_dim: int


@dataclass(frozen=True)
class SymplecticVerification:
    """Verificación explícita del axioma A1 (invarianza simpléctica)."""

    delta_omega: float
    det_M: float
    is_symplectic: bool


@dataclass(frozen=True)
class KAMResonanceLattice:
    """Retículo de resonancias vía fracciones continuas + γ diofántico."""

    k_vectors: Tuple[Tuple[int, ...], ...]
    diophantine_gamma: float
    tau: float
    is_kam_stable: bool


@dataclass(frozen=True)
class MelnikovResult:
    """Integral de Melnikov de la separatriz homoclínica."""

    M_value: float
    theta_0: float
    has_transversal_intersection: bool


@dataclass(frozen=True)
class LyapunovSpectrum:
    """Espectro de Lyapunov del flujo AST."""

    lambdas: Tuple[float, ...]
    lambda_max: float
    is_chaotic: bool


@dataclass(frozen=True)
class HodgeDiracSpectrum:
    """Espectro del Laplaciano Pesado Δ_G y verificación de axioma A4."""

    lambda_min: float
    lambda_max: float
    ker_dim_estimate: int
    is_trivial_cohomology: bool


@dataclass(frozen=True)
class ActionAngleChart:
    r"""Carta acción-ángulo $(I, \theta)$ sobre el toro invariante.

    $I_i = \frac{1}{2\pi}\oint_{\gamma_i} p\,dq$,  $\omega_i = \partial H_0 / \partial I_i$.
    """

    actions: Tuple[float, ...]
    frequencies: Tuple[float, ...]
    energy: float


@dataclass(frozen=True)
class TwistCondition:
    r"""Condición de no-degeneración de Kolmogorov: $\det(\partial^2 H_0 / \partial I_i \partial I_j) \neq 0$."""

    hessian_det: float
    is_nondegenerate: bool


@dataclass(frozen=True)
class PoincareReturnMap:
    r"""Aplicación de retorno de Poincaré sobre $\Sigma = \{q = \langle q \rangle\}$."""

    section_value: float
    crossings: int
    p_returns: Tuple[float, ...]
    recurrence_time: float
    is_recurrent: bool


@dataclass(frozen=True)
class RSI3Report:
    r"""
    Informe de automejora recursiva de nivel 3.

    Nivel 1: $x \mapsto f(x)$            (evaluación del analizador).
    Nivel 2: $f \mapsto P(f)$            (mejora del analizador).
    Nivel 3: $P \mapsto Q(P)$            (mejora del mejorador).

    Contracción: $\|x_{n+1}-x_n\| \le L\,\|x_n-x_{n-1}\|$, $L < 1$.
    """

    iterations: int
    lipschitz: float
    residual: float
    converged: bool
    score_trajectory: Tuple[float, ...]
    param_gamma_final: float
    param_tau_final: float
    param_eps_final: float


@dataclass(frozen=True)
class BirkhoffRemainder:
    r"""Estimación del resto de la forma normal de Birkhoff de orden $N$."""

    order: int
    remainder_norm: float
    is_normalized: bool


@dataclass(frozen=True)
class Phase2Handoff:
    r"""
    Morfismo FASE 2 → FASE 3: homología + KAM + Melnikov + Lyapunov + RSI³.

    [v5.1.0] Transporta la coborde de haz (densa, compactada por SVD a rango
    bajo si es necesario), el chart acción-ángulo, el retorno de Poincaré,
    el twist de Kolmogorov y el informe RSI³ — dominio estricto de Φ₃.
    """

    phase1: Phase1Handoff
    symplectic: SymplecticVerification
    cohomology_dim_H1: int
    problematic_variables: FrozenSet[str]
    kam_lattice: KAMResonanceLattice
    melnikov: MelnikovResult
    lyapunov: LyapunovSpectrum
    rsi3_iterations: int
    rsi3_converged: bool
    banach_L: float
    warnings: Tuple[str, ...]
    # v5.1.0
    action_angle: ActionAngleChart
    twist: TwistCondition
    poincare_return: PoincareReturnMap
    rsi3: RSI3Report
    homological_W_norm: float
    coboundary_shape: Tuple[int, int]
    coboundary_frobenius: float
    coboundary_svd_s: Tuple[float, ...]


@dataclass(frozen=True)
class AnalysisCertificate:
    """Certificado final OODA (3-fases) con veredicto Ω₃."""

    phase2: Phase2Handoff
    hodge_dirac: HodgeDiracSpectrum
    heyting_verdict: HeytingVerdict
    heyting_axioms_verified: bool
    is_coherent: bool
    is_hardened: bool
    overall_score: float
    warnings: Tuple[str, ...]
    birkhoff: BirkhoffRemainder
    poincare_recurrence_time: float


# ── Tipos y protocolos auxiliares ───────────────────────────────────────────

T = TypeVar("T")


class Monoid(Protocol[T]):
    @staticmethod
    def identity() -> T: ...

    def combine(self, other: T) -> T: ...


class NodeCategory(Enum):
    """Categorización semántica de nodos AST."""

    DECISION = auto()
    EXCEPTION = auto()
    DEFINITION = auto()
    EXPRESSION = auto()
    STATEMENT = auto()
    CONTROL_FLOW = auto()
    ASYNC = auto()

    def is_complexity_node(self) -> bool:
        return self in {NodeCategory.DECISION, NodeCategory.EXCEPTION}

    def is_definition_node(self) -> bool:
        return self == NodeCategory.DEFINITION


def _tuple_matrix(M: np.ndarray) -> Tuple[Tuple[float, ...], ...]:
    """Inmersión pura $\mathbb{R}^{n\times m} \hookrightarrow \mathrm{Tuple}$ (hashable)."""
    arr = np.asarray(M, dtype=np.float64)
    return tuple(tuple(float(x) for x in row) for row in arr)


def _from_tuple_matrix(t: Tuple[Tuple[float, ...], ...]) -> np.ndarray:
    if not t:
        return np.zeros((0, 0), dtype=np.float64)
    return np.array(t, dtype=np.float64)


def _safe_trapezoid(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))  # noqa: NP100 (numpy<2)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — GEOMETRÍA ESTRUCTURAL DEL AST (0-JET)                            ║
# ║  Última definición: `_ASTPhase1Pipeline._handoff_to_phase2`                ║
# ║  que ES el dominio de `_ASTPhase2Pipeline._ingest_phase1_handoff`.         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, order=True)
class DataFlowCoordinates:
    r"""
    Coordenadas en el fibrado cotangente $T^*Q$.

    • `reads ⊆ Q`: posición (variables leídas).
    • `writes ⊆ Q`: momento (variables escritas).
    • Forma simpléctica: $\omega = \sum_i d(\mathrm{read}_i)\wedge d(\mathrm{write}_i)$.

    Monoide bajo composición secuencial. La puntuación de interferencia
    $I(A,B) = |A.W \cap B.R| - |B.W \cap A.R|$ es el 2-cociclo discreto
    que desciende a $\omega_I$ en el handoff de FASE 1.
    """

    reads: FrozenSet[str] = field(default_factory=frozenset)
    writes: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.reads, frozenset):
            object.__setattr__(self, "reads", frozenset(self.reads))
        if not isinstance(self.writes, frozenset):
            object.__setattr__(self, "writes", frozenset(self.writes))
        for var in self.reads | self.writes:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid variable name: {repr(var)}")

    def __repr__(self) -> str:
        r = sorted(self.reads) if self.reads else []
        w = sorted(self.writes) if self.writes else []
        return f"DataFlow(R={r}, W={w})"

    @property
    def all_variables(self) -> FrozenSet[str]:
        return self.reads | self.writes

    @property
    def modified_variables(self) -> FrozenSet[str]:
        return self.reads & self.writes

    @property
    def pure_reads(self) -> FrozenSet[str]:
        return self.reads - self.writes

    @property
    def pure_writes(self) -> FrozenSet[str]:
        return self.writes - self.reads

    def combine(self, other: "DataFlowCoordinates") -> "DataFlowCoordinates":
        new_reads = self.reads | (other.reads - self.writes)
        new_writes = self.writes | other.writes
        return DataFlowCoordinates(reads=new_reads, writes=new_writes)

    @staticmethod
    def identity() -> "DataFlowCoordinates":
        return DataFlowCoordinates()

    def has_interference_with(self, other: "DataFlowCoordinates") -> bool:
        return bool(
            (self.reads & other.writes)
            or (self.writes & other.reads)
            or (self.writes & other.writes)
        )

    @staticmethod
    def interference_score(A: "DataFlowCoordinates", B: "DataFlowCoordinates") -> int:
        raw = len(A.writes & B.reads)
        war = len(B.writes & A.reads)
        return raw - war

    @staticmethod
    def verify_antisymmetry(A: "DataFlowCoordinates", B: "DataFlowCoordinates") -> bool:
        r"""Verifica programáticamente $I(A,B) = -I(B,A)$."""
        return DataFlowCoordinates.interference_score(
            A, B
        ) == -DataFlowCoordinates.interference_score(B, A)

    def dominates(self, other: "DataFlowCoordinates") -> bool:
        return DataFlowCoordinates.interference_score(self, other) > 0

    def spectral_radius(self) -> float:
        return math.sqrt(len(self.reads) ** 2 + len(self.writes) ** 2)

    def entropy(self) -> float:
        n = len(self.all_variables)
        return math.log2(n) if n > 0 else 0.0

    @staticmethod
    def interference_matrix(
        blocks: Sequence["DataFlowCoordinates"],
    ) -> np.ndarray:
        r"""
        [v5.1.0] 2-forma discreta $\omega_I \in \mathfrak{so}(m)$:
            $(\omega_I)_{ij} = I(b_i, b_j)$.
        Es antisimétrica por construcción ssi `verify_antisymmetry` vale ∀ pares.
        """
        m = len(blocks)
        Omega = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                s = float(DataFlowCoordinates.interference_score(blocks[i], blocks[j]))
                Omega[i, j] = s
                Omega[j, i] = -s
        return Omega

    def restriction_to(self, other: "DataFlowCoordinates") -> FrozenSet[str]:
        r"""
        [v5.1.0] Mapa de restricción de haz $\rho_{u\to v}: F(u)\to F(v)$:
        proyección sobre las variables comunes (intersección de soportes).
        """
        return self.all_variables & other.all_variables


@dataclass(frozen=True)
class ComplexityProfile:
    """Perfil multidimensional de complejidad (McCabe, Halstead)."""

    cyclomatic_complexity: int
    max_nesting_depth: int
    num_functions: int
    num_classes: int
    num_branches: int = 0
    num_loops: int = 0

    def __post_init__(self):
        if self.cyclomatic_complexity < 1:
            raise ValueError("Cyclomatic complexity must be ≥ 1")
        if self.max_nesting_depth < 0:
            raise ValueError("Nesting depth cannot be negative")
        if self.num_functions < 0 or self.num_classes < 0:
            raise ValueError("Counts cannot be negative")

    @property
    def is_maintainable(self) -> bool:
        return (
            self.cyclomatic_complexity <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY
            and self.max_nesting_depth <= 5
        )

    @property
    def risk_level(self) -> str:
        cc = self.cyclomatic_complexity
        if cc <= AnalysisLimits.WARN_CYCLOMATIC_COMPLEXITY:
            return "LOW"
        if cc <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY:
            return "MODERATE"
        if cc <= AnalysisLimits.CRITICAL_CYCLOMATIC_COMPLEXITY:
            return "HIGH"
        return "CRITICAL"

    @property
    def complexity_density(self) -> float:
        return self.cyclomatic_complexity / max(self.num_functions, 1)

    @property
    def essential_complexity(self) -> float:
        return max(1, self.cyclomatic_complexity - self.num_branches - self.num_loops + 1)

    @property
    def halstead_volume(self) -> float:
        n = max(self.num_functions + self.num_classes, 2)
        return self.cyclomatic_complexity * math.log2(n)

    def celestial_frequencies(self, dim: int) -> Tuple[float, ...]:
        r"""
        [v5.1.0] Frecuencias de acción-ángulo incommensurables a la
        razón áurea (evita resonancias exactas $\omega\cdot k = 0$):

            $\omega_i = \mu_i \cdot \varphi^{-i}$,
            $\mu = (\mathrm{CC}, \mathrm{depth}+1, \mathrm{branches}+1, \mathrm{loops}+1)$.
        """
        phi = AnalysisLimits.GOLDEN_RATIO
        base = (
            float(self.cyclomatic_complexity),
            float(self.max_nesting_depth + 1),
            float(self.num_branches + 1),
            float(self.num_loops + 1),
        )
        dim = max(1, min(int(dim), AnalysisLimits.MAX_SYMPLECTIC_DIM))
        omegas = []
        for i in range(dim):
            omegas.append(base[i % len(base)] * (phi ** (-i)))
        return tuple(omegas)

    def __repr__(self) -> str:
        return (
            f"Complexity(CC={self.cyclomatic_complexity}, "
            f"depth={self.max_nesting_depth}, risk={self.risk_level}, "
            f"density={self.complexity_density:.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "num_functions": self.num_functions,
            "num_classes": self.num_classes,
            "num_branches": self.num_branches,
            "num_loops": self.num_loops,
            "risk_level": self.risk_level,
            "complexity_density": self.complexity_density,
            "essential_complexity": self.essential_complexity,
            "is_maintainable": self.is_maintainable,
        }


class _LocalFlowExtractor(ast.NodeVisitor):
    """Extractor ligero de flujo de datos para un subárbol aislado."""

    def __init__(self):
        self.local_reads: Set[str] = set()
        self.local_writes: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.local_reads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.local_writes.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.local_reads.add(node.id)
            self.local_writes.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self.local_reads.add(node.attr)
        elif isinstance(node.ctx, ast.Store):
            self.local_writes.add(node.attr)
        elif isinstance(node.ctx, ast.Del):
            self.local_reads.add(node.attr)
            self.local_writes.add(node.attr)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.local_writes.add(node.arg)


class DataFlowAnalyzer(ast.NodeVisitor):
    r"""
    Analizador de flujo de datos con disipación Hamiltoniana de PATH-ENERGY.

    ENERGÍA DE TRAYECTORIA:
    -----------------------
    Para cada nodo $n$ con ancestros $a_1,\ldots,a_k=n$:
        $E(n) = \sum_i \mathrm{CC}_{incr}(a_i)\cdot\exp(-\mathrm{depth}(a_i)/D_0)$

    INVARIANTE (axioma de disipación):
        $E(n) \le \mathrm{MAX\_PATH\_ENERGY} = 30$ en todo nodo $n$.

    [v5.1.0] El incremento $\mathrm{CC}_{incr}$ incluye BoolOp/Match/IfExp
    (antes solo DECISION_NODES), alineando path-energy con McCabe real.
    """

    DECISION_NODES: FrozenSet[type] = frozenset(
        {
            ast.If,
            ast.While,
            ast.For,
            ast.AsyncFor,
            ast.ExceptHandler,
            ast.With,
            ast.AsyncWith,
            ast.IfExp,
            ast.Try,
        }
    )

    def __init__(
        self,
        enable_hamiltonian_monitor: bool = True,
        enable_block_tracking: bool = True,
        strict_mode: bool = True,
        legacy_hamiltonian: bool = False,
    ):
        self._reads: Set[str] = set()
        self._writes: Set[str] = set()

        self._current_depth: int = 0
        self._max_depth: int = 0
        self._cyclomatic: int = 1

        self._num_functions: int = 0
        self._num_classes: int = 0
        self._num_branches: int = 0
        self._num_loops: int = 0
        self._num_nodes_visited: int = 0

        self._in_function_args: bool = False
        self._in_comprehension: bool = False

        self._enable_hamiltonian: bool = enable_hamiltonian_monitor
        self._strict_mode: bool = strict_mode
        self._legacy_hamiltonian: bool = legacy_hamiltonian

        self._path_energy: float = 0.0
        self._max_path_energy: float = 0.0

        self._hamiltonian_stack: List[float] = []

        self._enable_block_tracking: bool = enable_block_tracking
        self._block_flows: List[Tuple[DataFlowCoordinates, int, str]] = []
        self._node_children: Dict[int, List[int]] = defaultdict(list)

    def _local_cc_increment(self, node: ast.AST) -> int:
        """Incremento ciclomático local alineado con McCabe (v5.1.0)."""
        t = type(node)
        if t in self.DECISION_NODES:
            return 1
        if t is ast.BoolOp:
            return max(0, len(node.values) - 1)
        if t is ast.Match:
            n_cases = len(getattr(node, "cases", []) or [])
            return max(1, n_cases)
        if t is ast.comprehension:
            return 1 + len(node.ifs)
        return 0

    def _compute_energy_increment(self, depth: int, cc_incr: int) -> float:
        """ΔE = CC_incr · exp(−depth / D₀)."""
        if cc_incr <= 0:
            return 0.0
        return float(cc_incr) * math.exp(
            -float(depth) / AnalysisLimits.ENERGY_DECAY_SCALE
        )

    def _check_path_energy_bound(self) -> None:
        if self._path_energy > AnalysisLimits.MAX_PATH_ENERGY:
            msg = (
                f"[v5.1.0] Path-energy violation: E={self._path_energy:.4f} > "
                f"{AnalysisLimits.MAX_PATH_ENERGY:.2f} at depth={self._current_depth}, "
                f"CC={self._cyclomatic}"
            )
            logger.warning(msg)
            if self._strict_mode:
                raise ThermodynamicSingularityError(
                    self._path_energy,
                    AnalysisLimits.MAX_PATH_ENERGY,
                    self._current_depth,
                    self._cyclomatic,
                )

    @staticmethod
    def _compute_hamiltonian(depth: int, complexity: int) -> float:
        return complexity / (depth + AnalysisLimits.HAMILTONIAN_EPSILON)

    def _push_hamiltonian(self) -> None:
        H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        self._hamiltonian_stack.append(H)

    def _pop_hamiltonian(self) -> None:
        if self._hamiltonian_stack:
            self._hamiltonian_stack.pop()

    def _check_legacy_dissipation(self) -> None:
        if not self._hamiltonian_stack:
            return
        parent_H = self._hamiltonian_stack[-1]
        current_H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        if current_H - parent_H > AnalysisLimits.FLOAT_TOLERANCE:
            if self._strict_mode:
                raise ThermodynamicSingularityError(
                    current_H,
                    parent_H,
                    self._current_depth,
                    self._cyclomatic,
                )

    def visit(self, node: ast.AST) -> Any:
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self._num_nodes_visited += 1

        if self._current_depth > AnalysisLimits.MAX_AST_DEPTH:
            raise ComplexityBoundsViolationError(
                "AST depth",
                self._current_depth,
                AnalysisLimits.MAX_AST_DEPTH,
            )
        if self._num_nodes_visited > AnalysisLimits.MAX_AST_NODES:
            raise ComplexityBoundsViolationError(
                "AST nodes",
                self._num_nodes_visited,
                AnalysisLimits.MAX_AST_NODES,
            )

        cc_incr_here = self._local_cc_increment(node)
        increment = self._compute_energy_increment(self._current_depth, cc_incr_here)

        if self._enable_hamiltonian:
            self._path_energy += increment
            self._max_path_energy = max(self._max_path_energy, self._path_energy)
            self._check_path_energy_bound()
            if self._legacy_hamiltonian:
                self._check_legacy_dissipation()
                self._push_hamiltonian()

        if self._enable_block_tracking:
            self._record_block_flow(node)

        try:
            result = super().visit(node)
        finally:
            if self._enable_hamiltonian:
                self._path_energy -= increment
                if self._legacy_hamiltonian:
                    self._pop_hamiltonian()
            self._current_depth -= 1

        return result

    def _record_block_flow(self, node: ast.AST) -> None:
        extractor = _LocalFlowExtractor()
        extractor.visit(node)
        coords = DataFlowCoordinates(
            reads=frozenset(extractor.local_reads),
            writes=frozenset(extractor.local_writes),
        )
        node_id = id(node)
        node_type = type(node).__name__
        self._block_flows.append((coords, node_id, node_type))
        for child in ast.iter_child_nodes(node):
            self._node_children[node_id].append(id(child))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._reads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(node.id)
            self._writes.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self._reads.add(node.attr)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(node.attr)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(node.attr)
            self._writes.add(node.attr)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if not self._in_function_args:
            self._writes.add(node.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._num_functions += 1
        self._in_function_args = True
        for arg in (
            *node.args.args,
            *node.args.posonlyargs,
            *node.args.kwonlyargs,
        ):
            self._writes.add(arg.arg)
        if node.args.vararg:
            self._writes.add(node.args.vararg.arg)
        if node.args.kwarg:
            self._writes.add(node.args.kwarg.arg)
        self._in_function_args = False
        for stmt in node.body:
            self.visit(stmt)
        for dec in node.decorator_list:
            self.visit(dec)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._num_functions += 1
        for arg in node.args.args:
            self._writes.add(arg.arg)
        self.visit(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._num_classes += 1
        self.generic_visit(node)

    def _increment_complexity(self, amount: int = 1) -> None:
        self._cyclomatic += amount

    def visit_If(self, node: ast.If) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._increment_complexity()
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._increment_complexity()
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._increment_complexity()
        if hasattr(node, "cases"):
            num_cases = len(node.cases)
            if num_cases > 1:
                self._increment_complexity(num_cases - 1)
            self._num_branches += num_cases
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        num_operands = len(node.values)
        if num_operands > 1:
            self._increment_complexity(num_operands - 1)
            self._num_branches += num_operands - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._increment_complexity()
        self._increment_complexity(len(node.ifs))
        self._num_loops += 1
        self._num_branches += len(node.ifs)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._increment_complexity()
        self.generic_visit(node)

    def get_dataflow_coordinates(self) -> DataFlowCoordinates:
        return DataFlowCoordinates(
            reads=frozenset(self._reads), writes=frozenset(self._writes)
        )

    def get_complexity_profile(self) -> ComplexityProfile:
        return ComplexityProfile(
            cyclomatic_complexity=self._cyclomatic,
            max_nesting_depth=self._max_depth,
            num_functions=self._num_functions,
            num_classes=self._num_classes,
            num_branches=self._num_branches,
            num_loops=self._num_loops,
        )

    def get_block_dataflows(self) -> List[Tuple[DataFlowCoordinates, int, str]]:
        return self._block_flows

    def get_tree_graph(self) -> Dict[int, List[int]]:
        return dict(self._node_children)

    def get_max_path_energy(self) -> float:
        return float(self._max_path_energy)


class JSONStructureValidator:
    """Validador de JSON con prevención DoS (Billion Laughs, stack overflow, etc.)."""

    @staticmethod
    def validate_structure(
        obj: Union[Dict, List, Any],
        max_depth: int = AnalysisLimits.MAX_JSON_DEPTH,
        max_keys: int = AnalysisLimits.MAX_JSON_KEYS,
        max_array_size: int = AnalysisLimits.MAX_JSON_ARRAY_SIZE,
        current_depth: int = 0,
        path: str = "$",
    ) -> Tuple[bool, str]:
        if current_depth > max_depth:
            return False, f"Depth {current_depth} exceeds {max_depth} at {path}"
        if isinstance(obj, dict):
            if len(obj) > max_keys:
                return False, f"Dict at {path} has {len(obj)} keys (limit {max_keys})"
            for key, value in obj.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    return False, f"Invalid key type at {path}"
                ok, err = JSONStructureValidator.validate_structure(
                    value,
                    max_depth,
                    max_keys,
                    max_array_size,
                    current_depth + 1,
                    f"{path}.{key}",
                )
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            if len(obj) > max_array_size:
                return False, f"Array at {path} has {len(obj)} items (limit {max_array_size})"
            for idx, item in enumerate(obj):
                ok, err = JSONStructureValidator.validate_structure(
                    item,
                    max_depth,
                    max_keys,
                    max_array_size,
                    current_depth + 1,
                    f"{path}[{idx}]",
                )
                if not ok:
                    return False, err
        elif isinstance(obj, (str, int, float, bool, type(None))):
            pass
        else:
            return False, f"Unsupported type {type(obj).__name__} at {path}"
        return True, ""

    @staticmethod
    def compute_depth(obj: Union[Dict, List, Any]) -> int:
        if isinstance(obj, dict):
            return (
                0
                if not obj
                else 1
                + max(
                    (JSONStructureValidator.compute_depth(v) for v in obj.values()),
                    default=0,
                )
            )
        if isinstance(obj, list):
            return (
                0
                if not obj
                else 1
                + max(
                    (JSONStructureValidator.compute_depth(i) for i in obj),
                    default=0,
                )
            )
        return 0

    @staticmethod
    def estimate_memory_size(obj: Union[Dict, List, Any]) -> int:
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                total += sys.getsizeof(k) + JSONStructureValidator.estimate_memory_size(v)
        elif isinstance(obj, list):
            for item in obj:
                total += JSONStructureValidator.estimate_memory_size(item)
        return total


class TabularNormalizer:
    """Normalizador a representación tabular (Markdown, CSV, flattened)."""

    @staticmethod
    def _serialize_value(
        value: Any,
        depth: int,
        max_depth: int = AnalysisLimits.MAX_FLATTEN_DEPTH,
        max_length: int = AnalysisLimits.MAX_STRING_REPR_LENGTH,
    ) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                if math.isnan(value):
                    return "NaN"
                if math.isinf(value):
                    return "∞" if value > 0 else "-∞"
                return f"{value:.6g}"
            return str(value)
        if isinstance(value, str):
            esc = value.replace('"', '\\"').replace("\n", "\\n")
            if len(esc) > max_length:
                esc = esc[: max_length - 3] + "..."
            return f'"{esc}"'
        if isinstance(value, dict):
            if depth >= max_depth:
                return f"{{...{len(value)} keys}}"
            items = [
                f"{k}: {TabularNormalizer._serialize_value(v, depth + 1, max_depth, max_length)}"
                for k, v in list(value.items())[:5]
            ]
            suffix = ", ..." if len(value) > 5 else ""
            return "{" + ", ".join(items) + suffix + "}"
        if isinstance(value, list):
            if depth >= max_depth:
                return f"[...{len(value)} items]"
            items = [
                TabularNormalizer._serialize_value(i, depth + 1, max_depth, max_length)
                for i in value[:5]
            ]
            suffix = ", ..." if len(value) > 5 else ""
            return "[" + ", ".join(items) + suffix + "]"
        return f"<{type(value).__name__}>"

    @staticmethod
    def to_markdown_table(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if not data:
            return "| (empty) |\n|---------|"
        keys = sorted(data.keys())
        header = "| " + " | ".join(str(k) for k in keys) + " |"
        separator = "|" + "|".join(["---"] * len(keys)) + "|"
        values = [TabularNormalizer._serialize_value(data[k], 0) for k in keys]
        row = "| " + " | ".join(values) + " |"
        return f"{header}\n{separator}\n{row}"

    @staticmethod
    def to_csv_row(data: Dict[str, Any]) -> Tuple[str, str]:
        if not isinstance(data, dict) or not data:
            return "", ""
        keys = sorted(data.keys())
        header = ",".join(str(k) for k in keys)
        row = ",".join(TabularNormalizer._serialize_value(data[k], 0) for k in keys)
        return header, row

    @staticmethod
    def flatten_dict(
        data: Dict[str, Any], parent_key: str = "", separator: str = "."
    ) -> Dict[str, Any]:
        items: List[Tuple[str, Any]] = []
        for k, v in data.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    TabularNormalizer.flatten_dict(v, new_key, separator).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)


class _CelestialSeedBuilder:
    r"""
    [v5.1.0] Construye el generador Hamiltoniano $A\in\mathfrak{sp}(2n)$
    a partir del 0-jet (reads/writes + 2-forma de interferencia).

    Bloque canónico:
        $A = \begin{pmatrix} 0 & \beta \\ \gamma & 0 \end{pmatrix}$,
        $\beta = \mathrm{diag}(\#W_i)$ simétrica, $\gamma = -\mathrm{diag}(\#R_i)$
        simétrica $\implies A\in\mathfrak{sp}(2n)$ (axioma: $A^\top J + JA = 0$).
    """

    @staticmethod
    def symplectic_dim(n_reads: int, n_writes: int) -> int:
        return max(2, min(AnalysisLimits.MAX_SYMPLECTIC_DIM, n_reads + n_writes, 4))

    @staticmethod
    def hamiltonian_generator(
        reads: FrozenSet[str],
        writes: FrozenSet[str],
        dim: int,
    ) -> np.ndarray:
        n = int(dim)
        beta = np.zeros((n, n), dtype=np.float64)
        gamma = np.zeros((n, n), dtype=np.float64)
        w_list = sorted(writes) or ["∅"]
        r_list = sorted(reads) or ["∅"]
        for i in range(n):
            beta[i, i] = 1.0 + float(len(w_list[i % len(w_list)]))
            gamma[i, i] = -1.0 - float(len(r_list[i % len(r_list)]))
        A = np.block(
            [
                [np.zeros((n, n), dtype=np.float64), beta],
                [gamma, np.zeros((n, n), dtype=np.float64)],
            ]
        )
        return A

    @staticmethod
    def verify_sp_algebra(A: np.ndarray, J: np.ndarray, atol: float = 1e-9) -> bool:
        r"""Verifica $A^\top J + JA = 0$ (pertenencia a $\mathfrak{sp}(2n)$)."""
        residual = A.T @ J + J @ A
        return float(np.linalg.norm(residual, ord="fro")) < atol


# ─── FASE 1 → FASE 2 HANDOFF (última definición de FASE 1) ───────────────────

class _ASTPhase1Pipeline:
    r"""
    Orquesta la FASE 1 (geometría estructural del AST + semilla celeste).

    Produce un `Phase1Handoff` que es el **dominio estricto** de
    `_ASTPhase2Pipeline._ingest_phase1_handoff` (primera flecha de FASE 2):

        $\Phi_2 \circ \Phi_1(\mathrm{source})
            = \texttt{\_ingest\_phase1\_handoff}(
                \texttt{\_handoff\_to\_phase2}(\mathrm{source}))$.
    """

    @staticmethod
    def _handoff_to_phase2(
        source_code: str,
        filename: str = "<string>",
        enable_hamiltonian: bool = True,
        enable_cohomology: bool = True,
        strict_mode: bool = True,
    ) -> Phase1Handoff:
        r"""
        Pipeline FASE 1 completa: parse → visita → métricas → semilla
        acción-ángulo / $\mathfrak{sp}(2n)$ → handoff.

        Esta es la **última** definición de FASE 1 y el **inicio** formal
        de FASE 2: su valor de retorno es el único argumento legal de
        `_ASTPhase2Pipeline._ingest_phase1_handoff`.
        """
        if not source_code or not source_code.strip():
            raise ValueError("Source code is empty or whitespace-only")

        try:
            tree = ast.parse(source_code, filename=filename, mode="exec")
        except SyntaxError as exc:
            logger.error(f"Syntax error in {filename}: {exc}")
            raise ValueError(f"Invalid Python syntax: {exc}") from exc

        analyzer = DataFlowAnalyzer(
            enable_hamiltonian_monitor=enable_hamiltonian,
            enable_block_tracking=enable_cohomology,
            strict_mode=strict_mode,
        )

        hamiltonian_ok = True
        warnings: List[str] = []

        try:
            analyzer.visit(tree)
        except ThermodynamicSingularityError as exc:
            hamiltonian_ok = False
            logger.error(f"Hamiltonian violation in {filename}: {exc}")
            if strict_mode:
                raise
            warnings.append(str(exc))

        dataflow = analyzer.get_dataflow_coordinates()
        complexity = analyzer.get_complexity_profile()

        raw_blocks = analyzer.get_block_dataflows()
        block_flows_tuple = tuple(
            (c.reads, c.writes, node_id, node_type)
            for c, node_id, node_type in raw_blocks
        )

        raw_graph = analyzer.get_tree_graph()
        tree_graph_tuple = tuple(
            (parent_id, tuple(children)) for parent_id, children in raw_graph.items()
        )

        # ── Semilla celeste (v5.1.0): 2-forma + generador 𝔰𝔭(2n) ────────────
        coords_only = [c for c, _, _ in raw_blocks]
        antisymmetry_ok = True
        interference_f = 0.0
        if len(coords_only) >= 2:
            sample = coords_only[: min(32, len(coords_only))]
            for a, b in combinations(sample, 2):
                if not DataFlowCoordinates.verify_antisymmetry(a, b):
                    antisymmetry_ok = False
                    warnings.append("Interference form failed antisymmetry")
                    break
            Omega = DataFlowCoordinates.interference_matrix(sample)
            interference_f = float(np.linalg.norm(Omega, ord="fro"))

        n_dof = _CelestialSeedBuilder.symplectic_dim(
            len(dataflow.reads), len(dataflow.writes)
        )
        frequencies = complexity.celestial_frequencies(n_dof)
        A = _CelestialSeedBuilder.hamiltonian_generator(
            dataflow.reads, dataflow.writes, n_dof
        )
        J = np.block(
            [
                [np.zeros((n_dof, n_dof)), np.eye(n_dof)],
                [-np.eye(n_dof), np.zeros((n_dof, n_dof))],
            ]
        )
        if not _CelestialSeedBuilder.verify_sp_algebra(A, J):
            warnings.append("Hamiltonian generator not in sp(2n); projecting")
            # Proyección al álgebra: A ↦ (A + J^{-1} A^T J)/2 no; usamos
            # A_sp = ½(A − J A^T J) que es la proyección J-antisimétrica.
            A = 0.5 * (A - J @ A.T @ J)

        handoff = Phase1Handoff(
            source_code=source_code,
            filename=filename,
            dataflow_reads=dataflow.reads,
            dataflow_writes=dataflow.writes,
            cyclomatic_complexity=complexity.cyclomatic_complexity,
            max_nesting_depth=complexity.max_nesting_depth,
            num_functions=complexity.num_functions,
            num_classes=complexity.num_classes,
            num_branches=complexity.num_branches,
            num_loops=complexity.num_loops,
            block_flows=block_flows_tuple,
            tree_graph=tree_graph_tuple,
            hamiltonian_ok=hamiltonian_ok,
            max_path_energy=analyzer.get_max_path_energy(),
            warnings=tuple(warnings),
            frequencies=frequencies,
            hamiltonian_generator=_tuple_matrix(A),
            interference_frobenius=interference_f,
            antisymmetry_ok=antisymmetry_ok,
            symplectic_dim=n_dof,
        )
        logger.info(
            "[FASE1→FASE2] %s: CC=%d, depth=%d, E_path=%.3f, blocks=%d, "
            "H_ok=%s, ω=%s, ‖ω_I‖_F=%.3f, sp(%d), warnings=%d",
            filename,
            complexity.cyclomatic_complexity,
            complexity.max_nesting_depth,
            handoff.max_path_energy,
            len(block_flows_tuple),
            hamiltonian_ok,
            tuple(round(x, 3) for x in frequencies),
            interference_f,
            2 * n_dof,
            len(warnings),
        )
        return handoff


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — HOMOLOGÍA DE POINCARÉ Y RSI³                                     ║
# ║  Primera def.: `_ASTPhase2Pipeline._ingest_phase1_handoff`                 ║
# ║  (continuación estricta de `_ASTPhase1Pipeline._handoff_to_phase2`)        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class SymplecticPoincareEngine:
    r"""
    Motor de Mecánica Celeste de Poincaré para el AST.

    Implementa:
      • proyección de Cayley-Darboux sobre $Sp(2n,\mathbb{R})$,
      • flujo Hamiltoniano $M=\exp(\varepsilon A)$ con $A\in\mathfrak{sp}(2n)$
        heredado de FASE 1 (nunca aleatorio),
      • carta acción-ángulo y condición de twist de Kolmogorov,
      • aplicación de retorno de Poincaré,
      • ecuación homológica de Poincaré-Lindstedt,
      • retículo KAM por fracciones continuas + vecinos de Farey,
      • espectro de Lyapunov Benettin–Shimada QR,
      • integral de Melnikov con bracket de Poisson.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"Dimensión simpléctica positiva requerida: {dim}")
        self.dim = int(dim)
        self.J = np.block(
            [
                [
                    np.zeros((dim, dim), dtype=np.float64),
                    np.eye(dim, dtype=np.float64),
                ],
                [
                    -np.eye(dim, dtype=np.float64),
                    np.zeros((dim, dim), dtype=np.float64),
                ],
            ]
        )

    def cayley_darboux_projection(self, M: np.ndarray) -> np.ndarray:
        r"""
        Proyección de Cayley-Darboux: $M \mapsto \Pi_{\mathrm{Sp}}(M)\in Sp(2n)$.

        1. $K=(M-I)(M+I)^{-1}$ (Cayley).
        2. $K_{sp}=\tfrac12(K - J K^\top J)$ (proyección al álgebra).
        3. $M_{sympl}=(I+K_{sp})(I-K_{sp})^{-1}$ (Cayley inversa).
        """
        dim2 = 2 * self.dim
        M_arr = np.asarray(M, dtype=np.float64)
        if M_arr.shape != (dim2, dim2):
            raise ValueError(
                f"Dimension incompatible: esperaba ({dim2},{dim2}), recibió {M_arr.shape}"
            )

        I = np.eye(dim2, dtype=np.float64)
        try:
            K = (M_arr - I) @ la.inv(M_arr + I)
            K_sp = 0.5 * (K - self.J @ K.T @ self.J)
            return (I + K_sp) @ la.inv(I - K_sp)
        except la.LinAlgError as exc:
            logger.warning(f"[PoincareEngine] Singularidad en Cayley: {exc}")
            M_reg = M_arr + 1e-12 * I
            K = (M_reg - I) @ la.inv(M_reg + I)
            K_sp = 0.5 * (K - self.J @ K.T @ self.J)
            return (I + K_sp) @ la.inv(I - K_sp)

    def construct_flow_map_from_sp_generator(
        self,
        A: np.ndarray,
        epsilon: float = AnalysisLimits.FLOW_MAP_EPSILON,
    ) -> np.ndarray:
        r"""
        [v5.1.0] Flujo Hamiltoniano $M=\exp(\varepsilon A)$ con
        $A\in\mathfrak{sp}(2n)$ heredado de FASE 1, proyectado a $Sp(2n)$.
        """
        A_arr = np.asarray(A, dtype=np.float64)
        dim2 = 2 * self.dim
        if A_arr.shape != (dim2, dim2):
            raise ValueError(
                f"Generador incompatible: esperaba ({dim2},{dim2}), recibió {A_arr.shape}"
            )
        M = la.expm(float(epsilon) * A_arr)
        return self.cayley_darboux_projection(M)

    def verify_symplectic_invariance(self, M_sympl: np.ndarray) -> SymplecticVerification:
        r"""
        Axioma A1: $\delta\omega=\|M^\top\Omega M-\Omega\|_F < \mathtt{SYMPLECTIC\_ATOL}$.
        Liouville: $\det M = +1$.
        """
        M = np.asarray(M_sympl, dtype=np.float64)
        Omega = self.J
        delta_omega = float(np.linalg.norm(M.T @ Omega @ M - Omega, ord="fro"))
        det_M = float(np.real(la.det(M)))
        is_sympl = delta_omega < AnalysisLimits.SYMPLECTIC_ATOL
        if not is_sympl:
            logger.warning(
                f"[PoincareEngine] δω={delta_omega:.3e} excede "
                f"{AnalysisLimits.SYMPLECTIC_ATOL:.1e}"
            )
        return SymplecticVerification(
            delta_omega=delta_omega, det_M=det_M, is_symplectic=is_sympl
        )

    def action_angle_from_orbit(
        self,
        orbit_q: np.ndarray,
        orbit_p: np.ndarray,
        frequencies: np.ndarray,
    ) -> ActionAngleChart:
        r"""
        [v5.1.0] Acciones $I_i=\frac{1}{2\pi}\oint p\,dq$ por regla trapezoidal
        sobre la órbita, y energía $H_0=\sum \omega_i I_i$.
        """
        q = np.asarray(orbit_q, dtype=np.float64).ravel()
        p = np.asarray(orbit_p, dtype=np.float64).ravel()
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        if q.size < 2 or p.size != q.size:
            I = tuple(float(abs(w)) for w in omega) if omega.size else (0.0,)
            return ActionAngleChart(actions=I, frequencies=tuple(float(x) for x in omega), energy=0.0)
        circulation = _safe_trapezoid(p, x=q)
        I0 = abs(circulation) / (2.0 * math.pi)
        n = max(1, omega.size)
        actions = tuple(I0 / n for _ in range(n))
        energy = float(np.dot(omega[:n], np.asarray(actions)))
        return ActionAngleChart(
            actions=actions,
            frequencies=tuple(float(x) for x in omega),
            energy=energy,
        )

    def kolmogorov_nondegeneracy(
        self,
        frequencies: np.ndarray,
        actions: np.ndarray,
    ) -> TwistCondition:
        r"""
        [v5.1.0] Twist: $H_0(I)=\omega\cdot I + \tfrac12 I^\top A I$ con
        $A_{ij}=\delta_{ij}\omega_i/(1+|I_i|)$ (Hessiano diagonal dominante).
        No-degenerado ssi $|\det A| > \varepsilon$.
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        I = np.asarray(actions, dtype=np.float64).ravel()
        n = min(omega.size, I.size)
        if n == 0:
            return TwistCondition(hessian_det=0.0, is_nondegenerate=False)
        hess = np.diag(omega[:n] / (1.0 + np.abs(I[:n])))
        try:
            det_h = float(np.real(la.det(hess)))
        except la.LinAlgError:
            det_h = 0.0
        return TwistCondition(
            hessian_det=det_h,
            is_nondegenerate=bool(abs(det_h) > AnalysisLimits.FLOAT_TOLERANCE),
        )

    def poincare_return_map(
        self,
        orbit_q: np.ndarray,
        orbit_p: np.ndarray,
    ) -> PoincareReturnMap:
        r"""
        [v5.1.0] Sección $\Sigma=\{q=\langle q\rangle\}$. Un cruce es un
        cambio de signo de $q-\langle q\rangle$ con $\dot q>0$. El mapa
        $p_i\mapsto p_{i+1}$ es el retorno; $T_{\mathrm{rec}}$ es el
        periodo medio entre cruces (recurrencia de Poincaré).
        """
        q = np.asarray(orbit_q, dtype=np.float64).ravel()
        p = np.asarray(orbit_p, dtype=np.float64).ravel()
        if q.size < 3 or p.size != q.size:
            return PoincareReturnMap(
                section_value=0.0,
                crossings=0,
                p_returns=(),
                recurrence_time=float("inf"),
                is_recurrent=False,
            )
        q0 = float(np.mean(q))
        s = q - q0
        p_hits: List[float] = []
        hit_idx: List[int] = []
        for i in range(1, s.size):
            if s[i - 1] <= 0.0 < s[i]:
                p_hits.append(float(p[i]))
                hit_idx.append(i)
        crossings = len(p_hits)
        if crossings >= 2:
            rec = float(np.mean(np.diff(np.asarray(hit_idx, dtype=np.float64))))
        elif crossings == 1:
            rec = float(q.size)
        else:
            rec = float("inf")
        return PoincareReturnMap(
            section_value=q0,
            crossings=crossings,
            p_returns=tuple(p_hits),
            recurrence_time=rec,
            is_recurrent=bool(crossings >= 2),
        )

    def _verify_zero_mean_perturbation(self, H1_coeffs: np.ndarray) -> bool:
        r"""Condición de solubilidad: $\langle H_1\rangle_{\mathbb{T}^n}=0$."""
        H1 = np.asarray(H1_coeffs, dtype=np.float64)
        if H1.size == 0:
            return True
        mean_H1 = float(np.mean(H1))
        scale = max(float(np.max(np.abs(H1))), 1e-15)
        return abs(mean_H1) / scale <= 1e-6

    def _generate_resonance_lattice(
        self,
        omega: np.ndarray,
        K: int = AnalysisLimits.KAM_MAX_K_NORM,
        tau: float = AnalysisLimits.KAM_TAU,
        gamma: float = AnalysisLimits.KAM_GAMMA,
    ) -> KAMResonanceLattice:
        r"""
        Retículo $k\in\mathbb{Z}^n$, $\|k\|_1\le K$, más convergentes CF y
        vecinos de Farey. $\gamma^*=\inf_k |\omega\cdot k|\,\|k\|_1^\tau$.
        KAM-estable ssi $\gamma^*\ge\gamma>0$.
        """
        omega = np.asarray(omega, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            return KAMResonanceLattice(
                k_vectors=(),
                diophantine_gamma=float("inf"),
                tau=tau,
                is_kam_stable=True,
            )
        if n == 1:
            g = float(abs(omega[0]))
            return KAMResonanceLattice(
                k_vectors=((1,),),
                diophantine_gamma=g,
                tau=tau,
                is_kam_stable=(g >= gamma),
            )

        k_set: Set[Tuple[int, ...]] = set()

        if n <= 4 and K <= 4:
            for k in product(range(-K, K + 1), repeat=n):
                if all(ki == 0 for ki in k):
                    continue
                k_set.add(tuple(int(ki) for ki in k))
        else:
            for i, j in combinations(range(n), 2):
                for d in product(range(-K, K + 1), repeat=2):
                    if d[0] == 0 and d[1] == 0:
                        continue
                    k_arr = [0] * n
                    k_arr[i] = int(d[0])
                    k_arr[j] = int(d[1])
                    k_set.add(tuple(k_arr))

        for i, j in combinations(range(n), 2):
            if abs(omega[j]) < 1e-15:
                continue
            ratio = float(omega[i] / omega[j])
            a = ratio
            p_prev, p_curr = 1, 0
            q_prev, q_curr = 0, 1
            convergents: List[Tuple[int, int]] = []
            for _ in range(20):
                ai = int(math.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr == 0 or abs(q_curr) > K:
                    break
                convergents.append((p_curr, q_curr))
                k_arr = [0] * n
                k_arr[i] = q_curr
                k_arr[j] = -p_curr
                k_set.add(tuple(k_arr))
                frac = a - ai
                if abs(frac) < 1e-12:
                    break
                a = 1.0 / frac
            # Vecinos de Farey: |p q' − p' q| = 1 (resonancias más peligrosas)
            for (p1, q1), (p2, q2) in zip(convergents, convergents[1:]):
                p_f, q_f = p1 + p2, q1 + q2
                if 0 < abs(q_f) <= K:
                    k_arr = [0] * n
                    k_arr[i] = q_f
                    k_arr[j] = -p_f
                    k_set.add(tuple(k_arr))

        min_gamma = float("inf")
        for k in k_set:
            k_arr = np.asarray(k, dtype=np.float64)
            k_norm1 = float(np.linalg.norm(k_arr, 1))
            if k_norm1 < 1e-15 or k_norm1 > K:
                continue
            dot = abs(float(np.dot(omega, k_arr)))
            candidate = dot * (k_norm1 ** tau)
            if candidate < min_gamma:
                min_gamma = candidate

        return KAMResonanceLattice(
            k_vectors=tuple(sorted(k_set)),
            diophantine_gamma=float(min_gamma),
            tau=tau,
            is_kam_stable=bool(min_gamma >= gamma),
        )

    def solve_poincare_homological_equation(
        self,
        H0_grad: np.ndarray,
        H1_coeffs: np.ndarray,
        gamma: float = AnalysisLimits.KAM_GAMMA,
        tau: float = AnalysisLimits.KAM_TAU,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Resuelve $\{H_0, W\}=H_1-\langle H_1\rangle$ en Fourier:
            $W_k = H_{1,k}/(i\,\omega\cdot k)$ para $k\neq 0$, $W_0=0$.

        Devuelve $(W, \|W\|_2)$. Verifica media cero y KAM-estabilidad.
        """
        omega0 = np.asarray(H0_grad, dtype=np.float64).ravel()
        H1 = np.asarray(H1_coeffs, dtype=np.float64).ravel()
        if omega0.size < 2:
            omega0 = np.pad(omega0, (0, max(0, 2 - omega0.size)), constant_values=1.0)

        H1_zm = H1 - np.mean(H1) if H1.size else H1
        if not self._verify_zero_mean_perturbation(H1_zm):
            raise ValueError(
                "[PoincareEngine] H1 tiene media no nula tras centrado."
            )

        lattice = self._generate_resonance_lattice(omega0, tau=tau, gamma=gamma)
        if not lattice.is_kam_stable:
            raise ValueError(
                f"[PoincareEngine] Retículo KAM inestable: "
                f"γ*={lattice.diophantine_gamma:.3e} < {gamma:.3e}."
            )

        k_vecs = lattice.k_vectors[: max(1, min(len(lattice.k_vectors), H1_zm.size, 16))]
        if not k_vecs:
            return np.zeros_like(H1_zm), 0.0

        W = np.zeros(len(k_vecs), dtype=np.complex128)
        n_w = min(omega0.size, len(k_vecs[0]))
        for idx, k in enumerate(k_vecs):
            k_arr = np.asarray(k[:n_w], dtype=np.float64)
            divisor = 1j * float(np.dot(omega0[:n_w], k_arr)) + 1e-12
            coeff = H1_zm[idx] if idx < H1_zm.size else 0.0
            W[idx] = coeff / divisor
        W_real = np.real(W)
        return W_real, float(np.linalg.norm(W_real))

    def evaluate_melnikov_integral_proper(
        self,
        H0: HamiltonianFn,
        H1: HamiltonianFn,
        orbit_q: np.ndarray,
        orbit_p: np.ndarray,
    ) -> MelnikovResult:
        r"""
        Integral de Melnikov:
            $M(\theta_0)=\int_{-\infty}^{+\infty}\{H_0,H_1\}(q_0(t-\theta_0),p_0)\,dt$,
        con $\{H_0,H_1\}=\partial_q H_0\,\partial_p H_1-\partial_p H_0\,\partial_q H_1$
        por diferencias finitas centradas y cuadratura trapezoidal.
        """
        q = np.asarray(orbit_q, dtype=np.float64).ravel()
        p = np.asarray(orbit_p, dtype=np.float64).ravel()
        if q.size < 3 or p.size != q.size:
            return MelnikovResult(
                M_value=0.0, theta_0=0.0, has_transversal_intersection=False
            )

        eps = 1e-6
        integrand = np.empty(q.size, dtype=np.float64)
        for i in range(q.size):
            qi, pi = float(q[i]), float(p[i])
            dqH0 = (H0(qi + eps, pi) - H0(qi - eps, pi)) / (2 * eps)
            dpH0 = (H0(qi, pi + eps) - H0(qi, pi - eps)) / (2 * eps)
            dqH1 = (H1(qi + eps, pi) - H1(qi - eps, pi)) / (2 * eps)
            dpH1 = (H1(qi, pi + eps) - H1(qi, pi - eps)) / (2 * eps)
            integrand[i] = dqH0 * dpH1 - dpH0 * dqH1

        M = _safe_trapezoid(integrand)
        abs_M = abs(M)
        theta_0 = float(np.argmax(np.abs(integrand))) if integrand.size else 0.0
        has_tangle = abs_M > AnalysisLimits.MELNIKOV_THRESHOLD
        return MelnikovResult(
            M_value=abs_M, theta_0=theta_0, has_transversal_intersection=has_tangle
        )

    def _lyapunov_spectrum_ast(
        self,
        orbit_q: np.ndarray,
        orbit_p: np.ndarray,
        stride: int = 1,
    ) -> LyapunovSpectrum:
        r"""
        Espectro de Lyapunov vía Benettin–Shimada:
        Jacobiano del mapa de Poincaré discreto $(q_k,p_k)\mapsto(q_{k+1},p_{k+1})$
        por diferencias, re-ortonormalización QR,
        $\lambda_i=(1/K)\sum_k\log|R_k^{(ii)}|$.
        """
        q = np.asarray(orbit_q, dtype=np.float64).ravel()
        p = np.asarray(orbit_p, dtype=np.float64).ravel()
        n_steps = min(q.size, p.size) - 1
        if n_steps < 3:
            return LyapunovSpectrum(
                lambdas=(-float("inf"),),
                lambda_max=-float("inf"),
                is_chaotic=False,
            )

        n = 2
        Q_mat = np.eye(n)
        log_sums = np.zeros(n)
        count = 0

        for k in range(1, n_steps):
            if k % stride != 0:
                continue
            x_k = np.array([q[k], p[k]], dtype=np.float64)
            x_next = np.array([q[k + 1], p[k + 1]], dtype=np.float64)
            x_prev = np.array([q[k - 1], p[k - 1]], dtype=np.float64)
            delta_x = x_k - x_prev
            delta_y = x_next - x_k
            Jloc = np.diag(
                np.where(
                    np.abs(delta_x) > 1e-15,
                    delta_y / np.maximum(np.abs(delta_x), 1e-15) * np.sign(delta_x),
                    1.0,
                )
            )
            M_qr = Jloc @ Q_mat
            Q_new, R = np.linalg.qr(M_qr)
            diag_R = np.abs(np.diag(R))
            log_sums += np.log(np.maximum(diag_R, 1e-15))
            Q_mat = Q_new
            count += 1

        if count == 0:
            return LyapunovSpectrum(
                lambdas=(-float("inf"),),
                lambda_max=-float("inf"),
                is_chaotic=False,
            )

        lambdas = np.sort(log_sums / count)[::-1]
        lam_max = float(lambdas[0])
        return LyapunovSpectrum(
            lambdas=tuple(float(x) for x in lambdas),
            lambda_max=lam_max,
            is_chaotic=bool(lam_max > 0.0),
        )

    def birkhoff_remainder(
        self,
        W_norm: float,
        epsilon: float = AnalysisLimits.FLOW_MAP_EPSILON,
        order: int = AnalysisLimits.BIRKHOFF_ORDER,
    ) -> BirkhoffRemainder:
        r"""
        [v5.1.0] Resto de Birkhoff: $\|R_N\| \lesssim \varepsilon^{N+1}\|W\|$.
        Normalizado ssi el resto es $< 1$.
        """
        rem = (abs(epsilon) ** (order + 1)) * abs(W_norm)
        return BirkhoffRemainder(
            order=order, remainder_norm=float(rem), is_normalized=bool(rem < 1.0)
        )


class RecursiveSelfImprovementL3:
    r"""
    Automejora recursiva de nivel 3 como contracción de Banach.

    Espacio: $X=\mathbb{R}^3\ni x=(\gamma,\tau,\varepsilon)$ (parámetros KAM/flujo).
    Operador de nivel 1: $f(x)=$ residual espectral (δω, 1−γ*/γ, |M|, λ_max⁺).
    Operador de nivel 2: $P(f)$ adapta $x$ en dirección del gradiente discreto.
    Operador de nivel 3: $Q(P)$ adapta la tasa de aprendizaje $\eta$ para
    forzar $\mathrm{Lip}(T)<1$.

    Punto fijo $x^*=T(x^*)$ ssi el analizador es internamente coherente
    con sus propios umbrales.
    """

    def __init__(self, engine: SymplecticPoincareEngine):
        self.engine = engine

    def _residual(
        self,
        gamma: float,
        tau: float,
        eps: float,
        omega: np.ndarray,
        A: np.ndarray,
        q: np.ndarray,
        p: np.ndarray,
        H0: HamiltonianFn,
        H1: HamiltonianFn,
    ) -> float:
        try:
            M = self.engine.construct_flow_map_from_sp_generator(A, epsilon=eps)
            sv = self.engine.verify_symplectic_invariance(M)
            lattice = self.engine._generate_resonance_lattice(
                omega, tau=tau, gamma=gamma
            )
            mel = self.engine.evaluate_melnikov_integral_proper(H0, H1, q, p)
            lyap = self.engine._lyapunov_spectrum_ast(q, p)
        except Exception:
            return 1.0
        r_sym = min(1.0, sv.delta_omega / max(AnalysisLimits.SYMPLECTIC_ATOL, 1e-18))
        r_kam = 0.0 if lattice.is_kam_stable else 1.0
        r_mel = 0.0 if not mel.has_transversal_intersection else 1.0
        r_ly = 0.0 if not lyap.is_chaotic else min(1.0, abs(lyap.lambda_max))
        return float(np.mean([r_sym, r_kam, r_mel, r_ly]))

    def iterate(
        self,
        omega: np.ndarray,
        A: np.ndarray,
        q: np.ndarray,
        p: np.ndarray,
        H0: HamiltonianFn,
        H1: HamiltonianFn,
        max_iter: int = AnalysisLimits.RSI3_MAX_ITER,
        min_iter: int = AnalysisLimits.RSI3_MIN_ITER,
        eps_conv: float = AnalysisLimits.RSI3_CONV_EPS,
    ) -> RSI3Report:
        gamma = float(AnalysisLimits.KAM_GAMMA)
        tau = float(AnalysisLimits.KAM_TAU)
        eps = float(AnalysisLimits.FLOW_MAP_EPSILON)
        eta = 0.35  # tasa de nivel 3 (Q)
        scores: List[float] = []
        prev_x = np.array([gamma, tau, eps], dtype=np.float64)
        prev_score = self._residual(gamma, tau, eps, omega, A, q, p, H0, H1)
        scores.append(prev_score)
        lip = 1.0
        converged = False
        it = 0

        for it in range(1, max_iter + 1):
            # Nivel 1: evaluar
            score = self._residual(gamma, tau, eps, omega, A, q, p, H0, H1)
            scores.append(score)

            # Nivel 2: mejorar parámetros (descenso sobre residual)
            # ∂r/∂γ < 0 si aumentar γ endurece KAM; adaptamos γ ↓ si inestable.
            if score > prev_score:
                gamma = max(gamma * 0.7, 1e-6)
                tau = min(tau * 1.05, 6.0)
                eps = max(eps * 0.8, 1e-4)
            else:
                gamma = min(gamma * 1.05, 1e-1)
                eps = min(eps * 1.02, 0.5)

            # Nivel 3: adaptar η para forzar Lip < 1
            x = np.array([gamma, tau, eps], dtype=np.float64)
            dx = float(np.linalg.norm(x - prev_x))
            ds = abs(score - prev_score)
            lip_step = ds / max(dx, 1e-15)
            # Q: si Lip_step ≥ 1, contrae η; si Lip_step ≪ 1, expande ligeramente
            if lip_step >= 1.0:
                eta *= 0.5
            else:
                eta = min(0.8, eta * 1.05)
            x = prev_x + eta * (x - prev_x)
            gamma, tau, eps = float(x[0]), float(x[1]), float(x[2])

            lip = lip_step
            if (
                it >= min_iter
                and ds < eps_conv
                and lip < AnalysisLimits.RSI3_LIPSCHITZ_TARGET
            ):
                converged = True
                prev_x = x
                prev_score = score
                break
            prev_x = x
            prev_score = score

        return RSI3Report(
            iterations=it,
            lipschitz=float(lip),
            residual=float(scores[-1] if scores else 1.0),
            converged=bool(converged and lip < 1.0),
            score_trajectory=tuple(float(s) for s in scores),
            param_gamma_final=float(gamma),
            param_tau_final=float(tau),
            param_eps_final=float(eps),
        )


class HeavyHodgeDiracLaplacian:
    r"""
    Laplaciano de Hodge-Dirac pesado sobre $C^0\oplus C^1$:

        $D=\begin{pmatrix}0 & d^*\\ d & 0\end{pmatrix}$,
        $\Delta=D^2=\mathrm{diag}(d^*d,\,dd^*)$,
        $d^*=G_0^{-1}d^\top G_1$.

    Axioma A4: $\lambda_{\min}(\Delta|_{C^1})>0 \iff H^1=0$.
    """

    def compute_heavy_laplacian(
        self,
        coboundary_d: np.ndarray,
        G1: np.ndarray,
        G0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        d = np.asarray(coboundary_d, dtype=np.float64)  # (n1, n0)
        if d.ndim != 2 or d.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        n1, n0 = d.shape
        G1_arr = np.asarray(G1, dtype=np.float64)
        if G1_arr.ndim == 1:
            G1_arr = np.diag(G1_arr)
        if G1_arr.shape != (n1, n1):
            G1_arr = np.eye(n1, dtype=np.float64)
        if G0 is None:
            G0_arr = np.eye(n0, dtype=np.float64)
        else:
            G0_arr = np.asarray(G0, dtype=np.float64)
            if G0_arr.ndim == 1:
                G0_arr = np.diag(G0_arr)
            if G0_arr.shape != (n0, n0):
                G0_arr = np.eye(n0, dtype=np.float64)
        # SPD-ización numérica
        G0_arr = 0.5 * (G0_arr + G0_arr.T) + 1e-12 * np.eye(n0)
        G1_arr = 0.5 * (G1_arr + G1_arr.T) + 1e-12 * np.eye(n1)
        try:
            G0_inv = la.inv(G0_arr)
        except la.LinAlgError:
            G0_inv = np.eye(n0, dtype=np.float64)
        # Δ0 = G0^{-1} d^T G1 d     (n0 × n0)
        # Δ1 = d G0^{-1} d^T G1     (n1 × n1)
        Delta0 = G0_inv @ d.T @ G1_arr @ d
        Delta1 = d @ G0_inv @ d.T @ G1_arr
        return np.block(
            [
                [Delta0, np.zeros((n0, n1), dtype=np.float64)],
                [np.zeros((n1, n0), dtype=np.float64), Delta1],
            ]
        )

    def verify_hodge_dirichlet_triviality(self, delta_G: np.ndarray) -> HodgeDiracSpectrum:
        D = np.asarray(delta_G, dtype=np.float64)
        if D.size == 0:
            return HodgeDiracSpectrum(
                lambda_min=float("inf"),
                lambda_max=float("inf"),
                ker_dim_estimate=0,
                is_trivial_cohomology=True,
            )
        D_sym = 0.5 * (D + D.T)
        try:
            evals = np.linalg.eigvalsh(D_sym)
        except np.linalg.LinAlgError:
            return HodgeDiracSpectrum(
                lambda_min=float("inf"),
                lambda_max=float("inf"),
                ker_dim_estimate=0,
                is_trivial_cohomology=False,
            )
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        tol = max(AnalysisLimits.HODGE_LAMBDA_MIN_THRESHOLD, 1e-10 * abs(lam_max))
        ker_dim = int(np.sum(np.abs(evals) < tol))
        is_trivial = bool(abs(lam_min) > tol)
        return HodgeDiracSpectrum(
            lambda_min=lam_min,
            lambda_max=lam_max,
            ker_dim_estimate=ker_dim,
            is_trivial_cohomology=is_trivial,
        )


class CellularSheafCohomology:
    r"""
    Cohomología de haces celulares sobre el grafo del AST.

    [v5.1.0] Mapas de restricción $\rho_{u\to v}$ = proyección al soporte
    común (no haz constante). $\delta^0$ actúa como

        $(\delta^0 s)(e)= \rho_{t(e)}(s(t(e))) - \rho_{s(e)}(s(s(e)))$.

    $\dim H^1 = \dim C^1 - \mathrm{rank}_{SVD}(\delta^0)$ (sobreestimación
    controlada: coker $\delta^0$ = $H^1 \oplus \mathrm{im}\,\delta^1$).
    """

    def __init__(self):
        self.vertices: List[DataFlowCoordinates] = []
        self.edges: List[Tuple[int, int]] = []
        self.all_variables: List[str] = []
        self._var_to_index: Dict[str, int] = {}
        self._delta0: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.vertices = []
        self.edges = []
        self.all_variables = []
        self._var_to_index = {}
        self._delta0 = None

    def build_from_ast_blocks(
        self,
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]],
    ) -> None:
        self.reset()
        node_id_to_index: Dict[int, int] = {}
        for idx, (coords, node_id, _) in enumerate(block_flows):
            node_id_to_index[node_id] = idx
            self.vertices.append(coords)
        var_set: Set[str] = set()
        for v in self.vertices:
            var_set.update(v.all_variables)
        self.all_variables = sorted(var_set)
        self._var_to_index = {v: i for i, v in enumerate(self.all_variables)}
        for parent_id, children_ids in tree_graph.items():
            if parent_id not in node_id_to_index:
                continue
            parent_idx = node_id_to_index[parent_id]
            for child_id in children_ids:
                if child_id in node_id_to_index:
                    self.edges.append((parent_idx, node_id_to_index[child_id]))
        logger.info(
            f"Built cellular sheaf: {len(self.vertices)} vertices, "
            f"{len(self.edges)} edges, {len(self.all_variables)} variables"
        )

    @staticmethod
    def _rigorous_rank(A: np.ndarray) -> int:
        r"""$r=\#\{\sigma_i > \sigma_{\max}\cdot\max(m,n)\cdot\varepsilon\}$."""
        if A.size == 0:
            return 0
        try:
            sv = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0
        if sv.size == 0:
            return 0
        sigma_max = float(sv[0])
        if sigma_max < 1e-300:
            return 0
        tol = sigma_max * max(A.shape) * float(np.finfo(np.float64).eps)
        return int(np.sum(sv > tol))

    def coboundary_matrix(self) -> np.ndarray:
        r"""
        [v5.1.0] $\delta^0$ con restricciones: la fila $(e,v)$ es nula
        si $v\notin \mathrm{supp}(\rho_{s(e)}\cap\rho_{t(e)})$.
        """
        if self._delta0 is not None:
            return self._delta0
        if not self.vertices or not self.all_variables or not self.edges:
            self._delta0 = np.zeros((0, 0), dtype=np.float64)
            return self._delta0
        n_vars = len(self.all_variables)
        n_edges = len(self.edges)
        n_verts = len(self.vertices)
        rows = n_edges * n_vars
        cols = n_verts * n_vars
        delta0 = np.zeros((rows, cols), dtype=np.float64)
        for e_idx, (p_idx, c_idx) in enumerate(self.edges):
            common = self.vertices[p_idx].restriction_to(self.vertices[c_idx])
            for var in common:
                v_idx = self._var_to_index[var]
                row = e_idx * n_vars + v_idx
                delta0[row, p_idx * n_vars + v_idx] = -1.0
                delta0[row, c_idx * n_vars + v_idx] = 1.0
        self._delta0 = delta0
        return delta0

    def compute_first_cohomology_dimension(self) -> int:
        if not self.vertices or not self.all_variables or not self.edges:
            return 0
        try:
            delta0 = self.coboundary_matrix()
            rank_delta0 = self._rigorous_rank(delta0)
            dim_H1 = int(delta0.shape[0] - rank_delta0)
            logger.info(
                f"Cohomology: dim C¹={delta0.shape[0]}, rank δ⁰={rank_delta0}, "
                f"dim H¹={dim_H1}"
            )
            return max(0, dim_H1)
        except Exception:
            return self._estimate_cohomology_without_numpy()

    def _estimate_cohomology_without_numpy(self) -> int:
        n_verts = len(self.vertices)
        n_edges = len(self.edges)
        n_vars = len(self.all_variables)
        beta1 = max(0, n_edges - n_verts + 1)
        return beta1 * n_vars

    def find_problematic_variables(self) -> Set[str]:
        all_reads: Set[str] = set()
        all_writes: Set[str] = set()
        for coords in self.vertices:
            all_reads.update(coords.reads)
            all_writes.update(coords.writes)
        return all_reads - all_writes

    def coboundary_spectrum(self, max_sv: int = 8) -> Tuple[Tuple[int, int], float, Tuple[float, ...]]:
        d = self.coboundary_matrix()
        if d.size == 0:
            return (0, 0), 0.0, ()
        frob = float(np.linalg.norm(d, ord="fro"))
        try:
            sv = np.linalg.svd(d, compute_uv=False)
            sv_t = tuple(float(x) for x in sv[:max_sv])
        except np.linalg.LinAlgError:
            sv_t = ()
        return (int(d.shape[0]), int(d.shape[1])), frob, sv_t

    @staticmethod
    def check_global_consistency(
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]],
        strict: bool = True,
    ) -> Tuple[int, Set[str], "CellularSheafCohomology"]:
        sheaf = CellularSheafCohomology()
        sheaf.build_from_ast_blocks(block_flows, tree_graph)
        dim_H1 = sheaf.compute_first_cohomology_dimension()
        problematic = sheaf.find_problematic_variables()
        if dim_H1 > 0 and strict:
            raise CohomologicalObstructionError(dim_H1, problematic)
        return dim_H1, problematic, sheaf


# ─── FASE 2: primera y última definición (continuación estricta) ────────────

class _ASTPhase2Pipeline:
    r"""
    Orquesta la FASE 2 (homología de Poincaré y RSI³).

    Continuación estricta:
        `_ingest_phase1_handoff(h1) → run → _handoff_to_phase3() → Phase2Handoff`.
    """

    def __init__(self):
        self._h1: Optional[Phase1Handoff] = None
        self._symplectic: Optional[SymplecticVerification] = None
        self._cohomology_dim: int = 0
        self._problematic_vars: FrozenSet[str] = frozenset()
        self._kam_lattice: Optional[KAMResonanceLattice] = None
        self._melnikov: Optional[MelnikovResult] = None
        self._lyapunov: Optional[LyapunovSpectrum] = None
        self._rsi3_iterations: int = 0
        self._rsi3_converged: bool = False
        self._banach_L: float = 1.0
        self._warnings: List[str] = []
        self._action_angle: Optional[ActionAngleChart] = None
        self._twist: Optional[TwistCondition] = None
        self._poincare_return: Optional[PoincareReturnMap] = None
        self._rsi3: Optional[RSI3Report] = None
        self._W_norm: float = 0.0
        self._coboundary_shape: Tuple[int, int] = (0, 0)
        self._coboundary_frob: float = 0.0
        self._coboundary_sv: Tuple[float, ...] = ()

    # ─── PRIMERA DEFINICIÓN DE FASE 2 ───────────────────────────────────────
    @classmethod
    def _ingest_phase1_handoff(cls, h1: Phase1Handoff) -> "_ASTPhase2Pipeline":
        r"""
        Consume el `Phase1Handoff` producido por
        `_ASTPhase1Pipeline._handoff_to_phase2`.

        Esta es la **primera** definición de FASE 2 y la **continuación
        estricta** de la última definición de FASE 1:

            $\mathrm{dom}(\texttt{\_ingest\_phase1\_handoff})
                = \mathrm{im}(\texttt{\_handoff\_to\_phase2})$.
        """
        pipeline = cls()
        pipeline._h1 = h1
        return pipeline

    def run(self) -> "_ASTPhase2Pipeline":
        """Cuerpo de FASE 2: Poincaré + sheaf + Lyapunov + RSI³ Banach."""
        if self._h1 is None:
            raise RuntimeError("_ingest_phase1_handoff no invocado.")

        block_flows: List[Tuple[DataFlowCoordinates, int, str]] = [
            (DataFlowCoordinates(reads=r, writes=w), nid, ntype)
            for (r, w, nid, ntype) in self._h1.block_flows
        ]
        tree_graph: Dict[int, List[int]] = {
            pid: list(children) for pid, children in self._h1.tree_graph
        }

        # ── Sheaf cohomology (restricciones reales)
        sheaf: Optional[CellularSheafCohomology] = None
        try:
            dim_H1, problematic, sheaf = CellularSheafCohomology.check_global_consistency(
                block_flows, tree_graph, strict=False
            )
            self._cohomology_dim = dim_H1
            self._problematic_vars = frozenset(problematic)
            shape, frob, sv = sheaf.coboundary_spectrum()
            self._coboundary_shape = shape
            self._coboundary_frob = frob
            self._coboundary_sv = sv
            if dim_H1 > 0:
                self._warnings.append(f"dim H¹ = {dim_H1}")
        except Exception as exc:
            self._warnings.append(f"Cohomology failed: {exc}")

        n_dof = int(self._h1.symplectic_dim)
        engine = SymplecticPoincareEngine(dim=n_dof)
        A = _from_tuple_matrix(self._h1.hamiltonian_generator)
        omega = np.asarray(self._h1.frequencies, dtype=np.float64)

        # ── Flujo Hamiltoniano DESDE la semilla de FASE 1 (no aleatorio)
        try:
            M_sympl = engine.construct_flow_map_from_sp_generator(A)
            self._symplectic = engine.verify_symplectic_invariance(M_sympl)
        except Exception as exc:
            self._warnings.append(f"Hamiltonian flow map failed: {exc}")
            self._symplectic = SymplecticVerification(
                delta_omega=float("inf"), det_M=0.0, is_symplectic=False
            )

        # ── Órbita de curvatura construida con las frecuencias de FASE 1
        cc = float(self._h1.cyclomatic_complexity)
        depth = float(self._h1.max_nesting_depth + 1)
        t = np.linspace(0.0, 1.0, max(16, self._h1.num_branches + 8))
        w0 = float(omega[0]) if omega.size else cc
        w1 = float(omega[1]) if omega.size > 1 else (w0 / AnalysisLimits.GOLDEN_RATIO)
        q_orbit = depth * np.cos(2 * np.pi * w0 * t) + 0.1 * np.sin(4 * np.pi * w1 * t)
        p_orbit = -depth * np.sin(2 * np.pi * w0 * t) + 0.1 * np.cos(4 * np.pi * w1 * t)

        self._action_angle = engine.action_angle_from_orbit(q_orbit, p_orbit, omega)
        self._twist = engine.kolmogorov_nondegeneracy(
            omega, np.asarray(self._action_angle.actions)
        )
        self._poincare_return = engine.poincare_return_map(q_orbit, p_orbit)
        self._lyapunov = engine._lyapunov_spectrum_ast(q_orbit, p_orbit)

        H0: HamiltonianFn = lambda qq, pp: 0.5 * pp * pp + 0.5 * (w0 ** 2) * qq * qq
        H1: HamiltonianFn = lambda qq, pp: 0.1 * qq * qq * qq
        self._melnikov = engine.evaluate_melnikov_integral_proper(
            H0, H1, q_orbit, p_orbit
        )

        # ── KAM + ecuación homológica
        self._kam_lattice = engine._generate_resonance_lattice(omega)
        H1_coeffs = 0.1 * np.cos(np.linspace(0.0, 2 * np.pi, max(4, omega.size * 2)))
        H1_coeffs = H1_coeffs - np.mean(H1_coeffs)
        try:
            _, self._W_norm = engine.solve_poincare_homological_equation(
                omega, H1_coeffs
            )
        except Exception as exc:
            self._warnings.append(f"Homological equation: {exc}")
            self._W_norm = float("inf")

        # ── RSI³ Banach (contracción real, no un contador)
        rsi = RecursiveSelfImprovementL3(engine)
        self._rsi3 = rsi.iterate(omega, A, q_orbit, p_orbit, H0, H1)
        self._rsi3_iterations = self._rsi3.iterations
        self._rsi3_converged = self._rsi3.converged
        self._banach_L = float(self._rsi3.lipschitz)
        if not self._rsi3.converged:
            self._warnings.append(
                f"RSI³ did not contract: Lip={self._rsi3.lipschitz:.3f}"
            )

        return self

    # ─── ÚLTIMA DEFINICIÓN DE FASE 2 ────────────────────────────────────────
    def _handoff_to_phase3(self) -> Phase2Handoff:
        r"""
        Empaqueta `Phase2Handoff`, dominio estricto de
        `_ASTPhase3Pipeline._ingest_phase2_handoff`.

        $\Phi_3\circ\Phi_2(h_1)=\texttt{\_ingest\_phase2\_handoff}(
            \texttt{\_handoff\_to\_phase3}(\mathrm{RSI}^3(h_1)))$.

        Esta es la **última** definición de FASE 2 y el **inicio** formal
        de FASE 3.
        """
        if (
            self._h1 is None
            or self._symplectic is None
            or self._kam_lattice is None
            or self._melnikov is None
            or self._lyapunov is None
            or self._action_angle is None
            or self._twist is None
            or self._poincare_return is None
            or self._rsi3 is None
        ):
            raise RuntimeError("FASE 2 incompleta: run() no invocado.")

        handoff = Phase2Handoff(
            phase1=self._h1,
            symplectic=self._symplectic,
            cohomology_dim_H1=self._cohomology_dim,
            problematic_variables=self._problematic_vars,
            kam_lattice=self._kam_lattice,
            melnikov=self._melnikov,
            lyapunov=self._lyapunov,
            rsi3_iterations=self._rsi3_iterations,
            rsi3_converged=self._rsi3_converged,
            banach_L=self._banach_L,
            warnings=tuple(self._warnings),
            action_angle=self._action_angle,
            twist=self._twist,
            poincare_return=self._poincare_return,
            rsi3=self._rsi3,
            homological_W_norm=self._W_norm,
            coboundary_shape=self._coboundary_shape,
            coboundary_frobenius=self._coboundary_frob,
            coboundary_svd_s=self._coboundary_sv,
        )
        logger.info(
            "[FASE2→FASE3] %s: δω=%.2e, dim_H¹=%d, γ_KAM=%.3e, |M|=%.3e, "
            "λ_max=%.3e, L_Φ=%.3f, conv=%s, twist=%s, Poincaré_rec=%s, ‖W‖=%.3e",
            self._h1.filename,
            self._symplectic.delta_omega,
            self._cohomology_dim,
            self._kam_lattice.diophantine_gamma,
            self._melnikov.M_value,
            self._lyapunov.lambda_max,
            self._banach_L,
            self._rsi3_converged,
            self._twist.is_nondegenerate,
            self._poincare_return.is_recurrent,
            self._W_norm,
        )
        return handoff


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — KAM, MELNIKOV, HEYTING Ω₃ Y CERTIFICADO OODA                    ║
# ║  Primera def.: `_ASTPhase3Pipeline._ingest_phase2_handoff`                 ║
# ║  (continuación estricta de `_ASTPhase2Pipeline._handoff_to_phase3`)        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _heyting_combine(
    hamiltonian_ok: bool,
    symplectic_ok: bool,
    cohomology_ok: bool,
    kam_stable: bool,
    melnikov_ok: bool,
    banach_ok: bool,
    hodge_trivial: bool,
    twist_ok: bool,
    rsi3_ok: bool,
) -> HeytingVerdict:
    """Join de átomos en Ω₃ (v5.1.0: incluye twist y RSI³)."""
    v_h = HeytingVerdict.COHERENT if hamiltonian_ok else HeytingVerdict.DEGRADED
    v_s = HeytingVerdict.COHERENT if symplectic_ok else HeytingVerdict.DEGRADED
    v_c = HeytingVerdict.COHERENT if cohomology_ok else HeytingVerdict.DEGRADED
    v_k = HeytingVerdict.COHERENT if kam_stable else HeytingVerdict.DEGRADED
    v_m = HeytingVerdict.VETOED if not melnikov_ok else HeytingVerdict.COHERENT
    v_b = HeytingVerdict.COHERENT if banach_ok else HeytingVerdict.DEGRADED
    v_hd = HeytingVerdict.COHERENT if hodge_trivial else HeytingVerdict.DEGRADED
    v_t = HeytingVerdict.COHERENT if twist_ok else HeytingVerdict.DEGRADED
    v_r = HeytingVerdict.COHERENT if rsi3_ok else HeytingVerdict.DEGRADED
    return HeytingVerdict.fast_join_all(
        [v_h, v_s, v_c, v_k, v_m, v_b, v_hd, v_t, v_r]
    )


def _heyting_lattice_axioms_check() -> bool:
    r"""
    Verifica empíricamente los axiomas de retículo Heyting en Ω₃:
    idempotencia, conmutatividad, asociatividad, absorción, distributividad,
    monotonicidad de ω.
    """
    elems = list(HeytingVerdict)
    ok = True
    for a in elems:
        if a.join(a) is not a or a.meet(a) is not a:
            ok = False
        for b in elems:
            if a.join(b) is not b.join(a) or a.meet(b) is not b.meet(a):
                ok = False
            for c in elems:
                if a.join(b).join(c) is not a.join(b.join(c)):
                    ok = False
                if a.meet(b).meet(c) is not a.meet(b.meet(c)):
                    ok = False
                if a.join(a.meet(b)) is not a:
                    ok = False
                if a.meet(a.join(b)) is not a:
                    ok = False
                if a.meet(b.join(c)) is not a.meet(b).join(a.meet(c)):
                    ok = False
    if not (
        HeytingVerdict.COHERENT.omega
        <= HeytingVerdict.DEGRADED.omega
        <= HeytingVerdict.VETOED.omega
    ):
        ok = False
    return ok


class _ASTPhase3Pipeline:
    r"""
    Orquesta la FASE 3 (KAM, Melnikov, Hodge-Dirac, Heyting Ω₃, certificado).

    Continuación estricta:
        `_ingest_phase2_handoff(h2) → run → finalize() → AnalysisCertificate`.
    """

    def __init__(self):
        self._h2: Optional[Phase2Handoff] = None
        self._hodge: Optional[HodgeDiracSpectrum] = None
        self._birkhoff: Optional[BirkhoffRemainder] = None
        self._warnings: List[str] = []

    # ─── PRIMERA DEFINICIÓN DE FASE 3 ───────────────────────────────────────
    @classmethod
    def _ingest_phase2_handoff(cls, h2: Phase2Handoff) -> "_ASTPhase3Pipeline":
        r"""
        Consume el `Phase2Handoff` de `_ASTPhase2Pipeline._handoff_to_phase3`.

        Esta es la **primera** definición de FASE 3 y la **continuación
        estricta** de la última definición de FASE 2:

            $\mathrm{dom}(\texttt{\_ingest\_phase2\_handoff})
                = \mathrm{im}(\texttt{\_handoff\_to\_phase3})$.
        """
        pipeline = cls()
        pipeline._h2 = h2
        return pipeline

    def run(self) -> "_ASTPhase3Pipeline":
        """Laplaciano de Hodge-Dirac sobre la coborde REAL del haz + Birkhoff."""
        if self._h2 is None:
            raise RuntimeError("_ingest_phase2_handoff no invocado.")

        # Reconstrucción de δ⁰ desde el handoff de FASE 2 (misma receta,
        # datos de FASE 1): NO se usa ruido gaussiano.
        block_flows: List[Tuple[DataFlowCoordinates, int, str]] = [
            (DataFlowCoordinates(reads=r, writes=w), nid, ntype)
            for (r, w, nid, ntype) in self._h2.phase1.block_flows
        ]
        tree_graph: Dict[int, List[int]] = {
            pid: list(children) for pid, children in self._h2.phase1.tree_graph
        }

        sheaf = CellularSheafCohomology()
        sheaf.build_from_ast_blocks(block_flows, tree_graph)
        d = sheaf.coboundary_matrix()

        if d.size == 0:
            self._hodge = HodgeDiracSpectrum(
                lambda_min=float("inf"),
                lambda_max=float("inf"),
                ker_dim_estimate=0,
                is_trivial_cohomology=True,
            )
        else:
            n1, n0 = d.shape
            # Métrica G₁: identidad + peso por valores singulares transportados
            G1 = np.eye(n1, dtype=np.float64)
            sv = np.asarray(self._h2.coboundary_svd_s, dtype=np.float64)
            if sv.size:
                w = np.ones(n1, dtype=np.float64)
                w[: min(sv.size, n1)] = 1.0 + sv[: min(sv.size, n1)]
                G1 = np.diag(w)
            hodge_op = HeavyHodgeDiracLaplacian()
            try:
                delta_G = hodge_op.compute_heavy_laplacian(d, G1)
                self._hodge = hodge_op.verify_hodge_dirichlet_triviality(delta_G)
            except Exception as exc:
                self._warnings.append(f"Hodge-Dirac failed: {exc}")
                self._hodge = HodgeDiracSpectrum(
                    lambda_min=float("inf"),
                    lambda_max=float("inf"),
                    ker_dim_estimate=0,
                    is_trivial_cohomology=False,
                )

        engine_dim = max(1, self._h2.phase1.symplectic_dim)
        engine = SymplecticPoincareEngine(dim=engine_dim)
        self._birkhoff = engine.birkhoff_remainder(self._h2.homological_W_norm)
        return self

    # ─── ÚLTIMA DEFINICIÓN DE FASE 3 ────────────────────────────────────────
    def finalize(self) -> AnalysisCertificate:
        r"""
        Emite el `AnalysisCertificate` final con veredicto Ω₃.

        Criterios (v5.1.0):
          • `hamiltonian_ok`  : Φ¹ no violó path-energy.
          • `symplectic_ok`   : δω < 1e-9.
          • `cohomology_ok`   : dim H¹ = 0.
          • `kam_stable`      : γ* ≥ 1e-3.
          • `melnikov_ok`     : |M| < 1e-2 (ausencia de tangle).
          • `banach_ok`       : Lip(T) < 0.95.
          • `hodge_trivial`   : λ_min(Δ) > 0.
          • `twist_ok`        : det(∂²H₀/∂I²) ≠ 0.
          • `rsi3_ok`         : contracción de Banach de nivel 3.
        """
        if self._h2 is None or self._hodge is None or self._birkhoff is None:
            raise RuntimeError("FASE 3 incompleta: run() no invocado.")

        h1 = self._h2.phase1
        symplectic_ok = self._h2.symplectic.is_symplectic
        cohomology_ok = self._h2.cohomology_dim_H1 == 0
        kam_stable = self._h2.kam_lattice.is_kam_stable
        melnikov_ok = not self._h2.melnikov.has_transversal_intersection
        banach_ok = self._h2.banach_L < AnalysisLimits.BANACH_TARGET
        hodge_trivial = self._hodge.is_trivial_cohomology
        twist_ok = self._h2.twist.is_nondegenerate
        rsi3_ok = self._h2.rsi3.converged

        verdict = _heyting_combine(
            hamiltonian_ok=h1.hamiltonian_ok,
            symplectic_ok=symplectic_ok,
            cohomology_ok=cohomology_ok,
            kam_stable=kam_stable,
            melnikov_ok=melnikov_ok,
            banach_ok=banach_ok,
            hodge_trivial=hodge_trivial,
            twist_ok=twist_ok,
            rsi3_ok=rsi3_ok,
        )

        heyting_ok = _heyting_lattice_axioms_check()
        if not heyting_ok:
            logger.error("[HEYTING] ¡Axiomas de retículo violados — bug crítico!")

        is_coherent = (
            h1.hamiltonian_ok
            and symplectic_ok
            and cohomology_ok
            and kam_stable
            and melnikov_ok
            and banach_ok
            and hodge_trivial
            and twist_ok
            and rsi3_ok
        )
        is_hardened = is_coherent and verdict is not HeytingVerdict.VETOED

        atoms = [
            1.0 if h1.hamiltonian_ok else 0.5,
            1.0 if symplectic_ok else 0.5,
            1.0 if cohomology_ok else 0.5,
            1.0 if kam_stable else 0.5,
            1.0 if melnikov_ok else 0.0,
            1.0 if banach_ok else 0.5,
            1.0 if hodge_trivial else 0.5,
            1.0 if twist_ok else 0.5,
            1.0 if rsi3_ok else 0.5,
            1.0 if self._birkhoff.is_normalized else 0.5,
        ]
        overall_score = float(np.mean(atoms))

        all_warnings = list(h1.warnings) + list(self._h2.warnings) + list(self._warnings)

        cert = AnalysisCertificate(
            phase2=self._h2,
            hodge_dirac=self._hodge,
            heyting_verdict=verdict,
            heyting_axioms_verified=heyting_ok,
            is_coherent=is_coherent,
            is_hardened=is_hardened,
            overall_score=overall_score,
            warnings=tuple(all_warnings),
            birkhoff=self._birkhoff,
            poincare_recurrence_time=self._h2.poincare_return.recurrence_time,
        )
        logger.info(
            "[OODA] %s: verdict=%s, score=%.3f, coherent=%s, hardened=%s, "
            "axiomas=%s, Birkhoff_R=%.3e, T_rec=%.3g, warnings=%d",
            h1.filename,
            verdict.value,
            overall_score,
            is_coherent,
            is_hardened,
            heyting_ok,
            self._birkhoff.remainder_norm,
            self._h2.poincare_return.recurrence_time,
            len(all_warnings),
        )
        return cert


# ─── PUNTO DE ENTRADA PÚBLICO ───────────────────────────────────────────────

class ASTStaticAnalyzer:
    r"""
    Analizador estático de código Python con verificación topológica OODA.

    Pipeline (3 fases anidadas estrictamente continuadas):

        FASE 1: `_ASTPhase1Pipeline._handoff_to_phase2(source)`
        FASE 2: `_ASTPhase2Pipeline._ingest_phase1_handoff(h1).run()._handoff_to_phase3()`
        FASE 3: `_ASTPhase3Pipeline._ingest_phase2_handoff(h2).run().finalize()`

    Compatible con la API v4.1.0 / v5.0.0 (`analyze_code` retorna dict).
    """

    @staticmethod
    def audit_three_phase(
        source_code: str,
        filename: str = "<string>",
        enable_hamiltonian: bool = True,
        enable_cohomology: bool = True,
        strict_mode: bool = True,
    ) -> AnalysisCertificate:
        r"""
        Ciclo OODA completo (3 fases anidadas) → `AnalysisCertificate`.
        """
        h1 = _ASTPhase1Pipeline._handoff_to_phase2(
            source_code=source_code,
            filename=filename,
            enable_hamiltonian=enable_hamiltonian,
            enable_cohomology=enable_cohomology,
            strict_mode=strict_mode,
        )
        h2 = (
            _ASTPhase2Pipeline._ingest_phase1_handoff(h1)
            .run()
            ._handoff_to_phase3()
        )
        cert = (
            _ASTPhase3Pipeline._ingest_phase2_handoff(h2).run().finalize()
        )
        return cert

    @staticmethod
    def analyze_code(
        source_code: str,
        filename: str = "<string>",
        enable_hamiltonian: bool = True,
        enable_cohomology: bool = True,
        strict_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        API compatible v4.1.0/v5.0.0: retorna dict con `dataflow`, `complexity`, etc.
        Internamente ejecuta el pipeline OODA v5.1.0 (3 fases anidadas).
        """
        cert = ASTStaticAnalyzer.audit_three_phase(
            source_code=source_code,
            filename=filename,
            enable_hamiltonian=enable_hamiltonian,
            enable_cohomology=enable_cohomology,
            strict_mode=strict_mode,
        )
        h1 = cert.phase2.phase1
        dataflow = DataFlowCoordinates(
            reads=h1.dataflow_reads, writes=h1.dataflow_writes
        )
        complexity = ComplexityProfile(
            cyclomatic_complexity=h1.cyclomatic_complexity,
            max_nesting_depth=h1.max_nesting_depth,
            num_functions=h1.num_functions,
            num_classes=h1.num_classes,
            num_branches=h1.num_branches,
            num_loops=h1.num_loops,
        )
        return {
            "dataflow": dataflow,
            "complexity": complexity,
            "hamiltonian_ok": h1.hamiltonian_ok,
            "cohomology_ok": cert.phase2.cohomology_dim_H1 == 0,
            "cohomology_dimension": cert.phase2.cohomology_dim_H1,
            "problematic_variables": set(cert.phase2.problematic_variables),
            "warnings": list(cert.warnings),
            "symplectic_verification": cert.phase2.symplectic,
            "kam_lattice": cert.phase2.kam_lattice,
            "melnikov_result": cert.phase2.melnikov,
            "lyapunov_spectrum": cert.phase2.lyapunov,
            "hodge_dirac_spectrum": cert.hodge_dirac,
            "heyting_verdict": cert.heyting_verdict.value,
            "heyting_axioms_verified": cert.heyting_axioms_verified,
            "is_coherent": cert.is_coherent,
            "is_hardened": cert.is_hardened,
            "overall_score": cert.overall_score,
            "certificate": cert,
            # v5.1.0
            "action_angle": cert.phase2.action_angle,
            "twist_condition": cert.phase2.twist,
            "poincare_return": cert.phase2.poincare_return,
            "rsi3": cert.phase2.rsi3,
            "birkhoff": cert.birkhoff,
            "poincare_recurrence_time": cert.poincare_recurrence_time,
            "frequencies": h1.frequencies,
            "antisymmetry_ok": h1.antisymmetry_ok,
        }

    @staticmethod
    def analyze_file(filepath: str, **kwargs) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
        return ASTStaticAnalyzer.analyze_code(
            source_code, filename=filepath, **kwargs
        )

    @staticmethod
    def validate_json_contract(
        schema: Union[Dict, List, Any],
        max_depth: Optional[int] = None,
        max_keys: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        max_depth = max_depth or AnalysisLimits.MAX_JSON_DEPTH
        max_keys = max_keys or AnalysisLimits.MAX_JSON_KEYS
        is_valid, error = JSONStructureValidator.validate_structure(
            schema, max_depth, max_keys
        )
        if not is_valid:
            logger.error(f"JSON validation failed: {error}")
            return False, error, None
        markdown = None
        if isinstance(schema, dict):
            try:
                markdown = TabularNormalizer.to_markdown_table(schema)
            except Exception as exc:
                markdown = f"(Normalization error: {exc})"
        return True, None, markdown

    @staticmethod
    def compute_interference_matrix(
        tools: Dict[str, DataFlowCoordinates],
    ) -> Dict[Tuple[str, str], int]:
        if len(tools) > AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE:
            raise ComplexityBoundsViolationError(
                "Interference matrix size",
                len(tools),
                AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE,
            )
        matrix: Dict[Tuple[str, str], int] = {}
        names = sorted(tools.keys())
        for a in names:
            for b in names:
                if a == b:
                    matrix[(a, b)] = 0
                else:
                    matrix[(a, b)] = DataFlowCoordinates.interference_score(
                        tools[a], tools[b]
                    )
        return matrix

    @staticmethod
    def topological_sort_by_interference(
        tools: Dict[str, DataFlowCoordinates],
    ) -> List[str]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)
        names = set(tools.keys())
        for n in names:
            in_degree[n] = 0
        for a in names:
            for b in names:
                if a != b and DataFlowCoordinates.interference_score(
                    tools[a], tools[b]
                ) > 0:
                    graph[a].add(b)
                    in_degree[b] += 1
        queue: deque = deque([n for n in names if in_degree[n] == 0])
        result: List[str] = []
        while queue:
            cur = queue.popleft()
            result.append(cur)
            for nb in graph[cur]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)
        if len(result) != len(names):
            raise ValueError(
                f"Dependency cycle detected among: {sorted(names - set(result))}"
            )
        return result


class ASTSymplecticParser:
    """API legacy (deprecated; usar `ASTStaticAnalyzer`)."""

    @staticmethod
    def parse_tool_dynamics(source_code: str) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        logger.warning(
            "ASTSymplecticParser.parse_tool_dynamics deprecated; usar ASTStaticAnalyzer."
        )
        result = ASTStaticAnalyzer.analyze_code(
            source_code,
            enable_hamiltonian=False,
            enable_cohomology=False,
            strict_mode=False,
        )
        return result["dataflow"], result["complexity"]

    @staticmethod
    def process_data_contract(json_schema: Dict[str, Any]) -> Optional[str]:
        logger.warning("ASTSymplecticParser.process_data_contract deprecated.")
        _, _, md = ASTStaticAnalyzer.validate_json_contract(json_schema)
        return md


PhaseSpaceCoordinates = DataFlowCoordinates
ThermodynamicProfile = ComplexityProfile


# ─── Utilidades de reporte ──────────────────────────────────────────────────

def print_complexity_report(
    complexity: ComplexityProfile, name: str = "Code", file=None
) -> None:
    output = file or sys.stdout
    width = 70
    print(f"\n{'=' * width}", file=output)
    print(f"COMPLEXITY ANALYSIS REPORT: {name}", file=output)
    print(f"{'=' * width}", file=output)
    print("\n📊 METRICS:", file=output)
    print(f"  • Cyclomatic Complexity:  {complexity.cyclomatic_complexity}", file=output)
    print(f"  • Max Nesting Depth:      {complexity.max_nesting_depth}", file=output)
    print(f"  • Functions:              {complexity.num_functions}", file=output)
    print(f"  • Classes:                {complexity.num_classes}", file=output)
    print(f"  • Branches:               {complexity.num_branches}", file=output)
    print(f"  • Loops:                  {complexity.num_loops}", file=output)
    print("\n📈 DERIVED:", file=output)
    print(f"  • Complexity Density:     {complexity.complexity_density:.2f}", file=output)
    print(f"  • Essential Complexity:   {complexity.essential_complexity:.1f}", file=output)
    print(f"  • Halstead Volume:        {complexity.halstead_volume:.1f} bits", file=output)
    print("\n🎯 ASSESSMENT:", file=output)
    print(f"  • Risk Level:             {complexity.risk_level}", file=output)
    print(
        f"  • Maintainable:           {'✓' if complexity.is_maintainable else '✗'}",
        file=output,
    )
    print(f"\n{'=' * width}\n", file=output)


def print_dataflow_report(
    dataflow: DataFlowCoordinates, name: str = "Module", file=None
) -> None:
    output = file or sys.stdout
    width = 70
    print(f"\n{'=' * width}", file=output)
    print(f"DATA FLOW ANALYSIS: {name}", file=output)
    print(f"{'=' * width}", file=output)
    print("\n📖 READS:", file=output)
    for v in sorted(dataflow.pure_reads) or ["(none)"]:
        print(f"  • {v}", file=output)
    print("\n✏️  WRITES:", file=output)
    for v in sorted(dataflow.pure_writes) or ["(none)"]:
        print(f"  • {v}", file=output)
    print("\n🔄 MODIFIED:", file=output)
    for v in sorted(dataflow.modified_variables) or ["(none)"]:
        print(f"  • {v}", file=output)
    print("\n📊 STATISTICS:", file=output)
    print(f"  • Total Variables:        {len(dataflow.all_variables)}", file=output)
    print(f"  • Spectral Radius:        {dataflow.spectral_radius():.2f}", file=output)
    print(f"  • Shannon Entropy:        {dataflow.entropy():.2f} bits", file=output)
    print(f"\n{'=' * width}\n", file=output)


__all__: List[str] = [
    "SymplecticPoincareEngine",
    "HeavyHodgeDiracLaplacian",
    "RecursiveSelfImprovementL3",
    "DataFlowCoordinates",
    "ComplexityProfile",
    "NodeCategory",
    "SymplecticVerification",
    "KAMResonanceLattice",
    "MelnikovResult",
    "LyapunovSpectrum",
    "HodgeDiracSpectrum",
    "ActionAngleChart",
    "TwistCondition",
    "PoincareReturnMap",
    "RSI3Report",
    "BirkhoffRemainder",
    "Phase1Handoff",
    "Phase2Handoff",
    "AnalysisCertificate",
    "DataFlowAnalyzer",
    "CellularSheafCohomology",
    "JSONStructureValidator",
    "TabularNormalizer",
    "ASTStaticAnalyzer",
    "ASTSymplecticParser",
    "AnalysisException",
    "ThermodynamicSingularityError",
    "CohomologicalObstructionError",
    "ComplexityBoundsViolationError",
    "StructuralValidationError",
    "BanachContractionFailure",
    "HeytingVerdict",
    "AnalysisLimits",
    "print_complexity_report",
    "print_dataflow_report",
]


if __name__ == "__main__":
    pass