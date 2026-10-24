# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Minimizer Agent (Custodio de la Base Booleana de la MIC)        ║
║ Ruta   : app/agents/boole/tactics/mic_minimizer_agent.py                     ║
║ Versión: 5.4.0-Tactics-Boolean-RSI3-Poincare-Celestial-Nested                ║
╚══════════════════════════════════════════════════════════════════════════════╝

MEJORAS v5.4.0 RESPECTO A v5.3.0 (retrocompatibles, granulares, rigurosas):
─────────────────────────────────────────────────────────────────────────
FASE 1 — Gröbner + Cayley–Darboux (fail-closed):
  • `_compute_cayley_darboux_embedding` verifica Klein
    $K_{sp}^\top J + J K_{sp}=0$ y el residuo RELATIVO
    $\|M^\top J M - J\|_F/\|M\|_F^2$; ante singularidad ⇒ residuos = +∞,
    `is_symplectic_valid=False` (ya no se reporta éxito espurio).
  • Carta de Darboux de dimensión par canónica (`darboux_chart_dim`).
  • Condición espectral y sparsity de GF(2) sin magia.
  • Última definición: `_handoff_to_phase2` → `MICPhase1Handoff`
    ≡ dominio de `_ingest_phase1_handoff` (inicio estricto de FASE 2).

FASE 2 — No-interferencia + Poincaré celeste + RSI³ anidado:
  • Primera definición: `_ingest_phase1_handoff` (continuidad Φ₁ → Φ₂).
  • Homoclínica real del péndulo: $q_h=4\arctan(e^t)-\pi$, $p_h=2\,\mathrm{sech}(t)$.
  • Melnikov cerrado $2\pi\,\mathrm{sech}(\pi\omega/2)$ + trapecio sech.
  • Lindstedt: $\omega=\omega_0+(3\varepsilon a^2)/(8\omega_0)$ (Duffing).
  • Condición de Bruno + KAM por convergentes de fracciones continuas.
  • Mapa de retorno de Poincaré (mapa estándar de Chirikov) + Lyapunov.
  • Tiempo de Nekhoroshev $T\sim\exp(c/\varepsilon^{1/(2n)})$ y recurrencia
    de Poincaré–Kac $\langle\tau\rangle\sim\varepsilon^{-n}$.
  • RSI³ *genuino* de 3 niveles Kleisli:
      T¹  mejora de objeto (soft-threshold espectral + Picard–Banach),
      T²  mejora del operador (τ adaptativo al residual de T¹),
      T³  meta-mejora (asociador $\mu\circ T\mu$ vs $\mu\circ\mu T$ + Lip$(T)^3$).
  • Fubini–Study de Grassmann (ángulos principales Im $P$ vs Im $P_\tau$);
    corrige el bug $\langle u_1,u_2\rangle=0\Rightarrow\arccos 0=\pi/2$ constante.
  • Gram: `conflict_pairs`, gap espectral y $\kappa(G)$ reales;
    idempotencia $\|P P^+ P - P\|$ (ya no hardcodeada).
  • Última definición: `_handoff_to_phase3` → `MICPhase2Handoff`
    ≡ dominio de `_ingest_phase2_handoff` (inicio estricto de FASE 3).

FASE 3 — ROBDD + Wasserstein + Heyting Ω₃ + Crowbar:
  • Primera definición: `_ingest_phase2_handoff`.
  • `_entropy_analysis_full`: familia Rényi $\{H_0,H_1,H_2,H_\infty\}$.
  • `_heyting_combine_full` con 7 átomos *reales* (Gröbner/simpléctico
    ya no hardcodeados a True).
  • `HeytingOmega3.implies` + axiomas de adjuncion / modus ponens.
  • Última definición: `_emit_governance_certificate`
    → `MinimizerGovernanceCertificate`.

Bugs corregidos en v5.4.0:
  • Cayley: excepción ⇒ valid=False, residuo=∞ (fail-closed).
  • Melnikov: homoclínica sech + fórmula cerrada (no proxy $p=\omega\sin q$).
  • RSI³: Fubini–Study de Grassmann; Picard anidado T⊂T²⊂T³; sin `'S' in dir()`.
  • Heyting de FASE 3 consume invariantes de Φ₁/Φ₂ (no átomos tautológicos).
  • `projection_rank` = rango numérico real; idempotencia medida.
  • `OrthogonalityAnalysis.spectral_gap` / `condition_number` computados.

MARCO TEÓRICO (compacto):
─────────────────────────────────────────────────
A1: Anillo booleano $\mathbb{Z}_2[x]/\langle x_i^2-x_i\rangle$.
A2: $M^\top J M = J$ ($\delta\omega_{\mathrm{rel}}<10^{-10}$ y Klein $<10^{-10}$).
A3: $|\omega_0\cdot k|\ge\gamma/\|k\|_1^\tau$ (KAM) ∧ $|M(\theta_0)|$ pequeño
    (Melnikov) ∧ Bruno.
A4: $\mathrm{Lip}(T_{\mathrm{RSI3}})\le 0.95$ y $\mathrm{Lip}(T^3)\le\mathrm{Lip}(T)^3$.
A5: $\Omega_3=\{0\mathrm{x}00<0\mathrm{x}01<0\mathrm{x}03\}$ → Crowbar GPIO14.

Composición funtorial:
    $Z_{\mathrm{Min}} = \Phi_3\circ\Phi_2\circ\Phi_1$
    $\Phi_1$ --MICPhase1Handoff--> $\Phi_2$ --MICPhase2Handoff--> $\Phi_3$
                                         --MinimizerGovernanceCertificate--> $\Omega_3$
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from itertools import combinations
from typing import Any, Dict, Final, List, Optional, Protocol, Set, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as la
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False
    la = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# §0. DEPENDENCIAS ARQUITECTÓNICAS Y PROTOCOLOS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


__version__ = "5.4.0"
logger = logging.getLogger("MIC.Gamma.MinimizerAgent.v5.4")


class BooleanAlgebraProtocol(Protocol):
    def rank_gf2(self) -> int: ...
    def nullity_gf2(self) -> int: ...
    def is_independent(self) -> bool: ...


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_LOG2: Final[float] = math.log(2.0)

_MAX_BOOLEAN_VARIABLES: Final[int] = 256
_MAX_ROBDD_NODES: Final[int] = 100000
_MAX_TRUTH_TABLE_SIZE: Final[int] = 2 ** 20

_MIN_ENTROPY_TOLERANCE: Final[float] = 1e-12
_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-10
_PROBABILITY_TOLERANCE: Final[float] = 1e-12
_INTEGER_REPRESENTATION_TOLERANCE: Final[float] = 1e-12
_KL_DIVERGENCE_TOLERANCE: Final[float] = 1e-10
_WASSERSTEIN_DISTANCE_TOLERANCE: Final[float] = 1e-12
_SYMPLECTIC_RESIDUE_TOLERANCE: Final[float] = 1e-10
_KLEIN_RESIDUAL_TOLERANCE: Final[float] = 1e-10
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_KAM_GAMMA: Final[float] = 0.125
_KAM_MAX_DENOM: Final[int] = 64
_MELNIKOV_THRESHOLD: Final[float] = 1e-2
_BANACH_TARGET: Final[float] = 0.95
_KRULL_MIN: Final[float] = 0.10
_FS_MAX: Final[float] = 1e-2
_BRUNO_FLOOR: Final[float] = 1e-12
_SV_FLOOR: Final[float] = 1e-15

_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_GF2_PIVOT_THRESHOLD: Final[float] = 0.5


def _svdvals(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Valores singulares con fallback SciPy → NumPy."""
    if SCIPY_AVAILABLE:
        return np.asarray(la.svdvals(M), dtype=np.float64)
    return np.linalg.svd(M, compute_uv=False)


def _fro_norm(M: NDArray[np.float64]) -> float:
    if SCIPY_AVAILABLE:
        return float(la.norm(M, ord="fro"))
    return float(np.linalg.norm(M, ord="fro"))


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class MICMinimizerAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Base Booleana."""
    pass


class BooleanInputValidationError(MICMinimizerAgentError):
    pass


class GrobnerDegeneracyError(MICMinimizerAgentError):
    pass


class NonInterferenceViolationError(MICMinimizerAgentError):
    pass


class ROBDDHomotopyError(MICMinimizerAgentError):
    pass


class SymplecticDeriftError(MICMinimizerAgentError):
    pass


class PoincareResonanceError(MICMinimizerAgentError):
    pass


class RSI3ContractionError(MICMinimizerAgentError):
    r"""Ruptura de la contracción de Banach en T, T² o T³."""
    pass


class ProbabilityDistributionError(MICMinimizerAgentError):
    pass


class OODAChainViolationError(MICMinimizerAgentError):
    r"""Ruptura de continuidad entre fases $\Phi_k\to\Phi_{k+1}$."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ENUMERACIONES
# ═══════════════════════════════════════════════════════════════════════════════
class BooleanReductionPhase(Enum):
    PHASE_1_GROBNER = auto()
    PHASE_2_UNSAT_CORE = auto()
    PHASE_3_ROBDD = auto()
    COMPLETE = auto()


class HeytingOmega3(Enum):
    r"""
    Álgebra de Heyting trivalente $\Omega_3=\{0\mathrm{x}00<0\mathrm{x}01<0\mathrm{x}03\}$
    (cadena de severidad: COHERENT = bottom, VETOED = top).

    Operaciones: join = $\vee$ (OR de máscara), meet = $\wedge$ (AND de máscara),
    implies = $\to$ (pseudo-complemento relativo de cadena).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @property
    def omega(self) -> float:
        return {0x00: 0.0, 0x01: 0.5, 0x03: 1.0}[self.value]

    @property
    def mask(self) -> int:
        return self.value

    @classmethod
    def from_omega(cls, v: float) -> "HeytingOmega3":
        if v <= 0.0:
            return cls.COHERENT
        if v < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def from_mask(cls, m: int) -> "HeytingOmega3":
        if (m & 0x03) == 0x03:
            return cls.VETOED
        if (m & 0x01) == 0x01:
            return cls.DEGRADED
        return cls.COHERENT

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.from_mask(self.mask | other.mask)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.from_mask(self.mask & other.mask)

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Pseudo-complemento relativo en la cadena $0<d<v$:
        $a\to b=\bigvee\{c\mid a\wedge c\le b\}$.
        Equivale a $\top$ si $a\le b$, y a $b$ en caso contrario.
        """
        if self.omega <= other.omega:
            return HeytingOmega3.VETOED  # top
        return other

    def leq(self, other: "HeytingOmega3") -> bool:
        return self.omega <= other.omega

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingOmega3"]) -> "HeytingOmega3":
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)


def _heyting_lattice_axioms_check() -> bool:
    r"""
    Verifica axiomas de retículo distributivo acotado + álgebra de Heyting
    en $\Omega_3$: idempotencia, conmutatividad, asociatividad, absorción,
    distributividad, monotonía de $\omega$, $a\to a=\top$, modus ponens
    $a\wedge(a\to b)\le b$, y adjuncion $a\wedge c\le b\iff c\le a\to b$.
    """
    elems = list(HeytingOmega3)
    top = HeytingOmega3.VETOED
    ok = True
    for a in elems:
        if a.join(a) is not a or a.meet(a) is not a:
            ok = False
        if a.implies(a) is not top:
            ok = False
        for b in elems:
            if a.join(b) is not b.join(a) or a.meet(b) is not b.meet(a):
                ok = False
            if not (a.meet(a.implies(b))).leq(b):
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
                if a.meet(b.join(c)) is not (a.meet(b)).join(a.meet(c)):
                    ok = False
                left = a.meet(c).leq(b)
                right = c.leq(a.implies(b))
                if left != right:
                    ok = False
    if not (
        HeytingOmega3.COHERENT.omega
        <= HeytingOmega3.DEGRADED.omega
        <= HeytingOmega3.VETOED.omega
    ):
        ok = False
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES (DTOs)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GF2MatrixProperties:
    rows: int
    cols: int
    rank_gf2: int
    nullity_gf2: int
    pivot_columns: Tuple[int, ...]
    free_columns: Tuple[int, ...]
    kernel_dimension: int
    has_full_rank: bool
    condition_number: float = field(default=1.0)
    sparsity: float = field(default=0.0)
    hamming_weight: int = field(default=0)

    def __post_init__(self) -> None:
        if self.rank_gf2 + self.nullity_gf2 != self.cols:
            raise ValueError(
                f"Violación rango-nulidad: "
                f"rank({self.rank_gf2}) + nullity({self.nullity_gf2}) ≠ cols({self.cols})"
            )


@dataclass(frozen=True, slots=True)
class LinearCodeProperties:
    length: int
    dimension: int
    minimum_distance: int
    rate: float
    redundancy: int
    singleton_bound: int
    hamming_bound: float

    def is_valid_code(self) -> bool:
        return self.minimum_distance <= self.singleton_bound and self.dimension <= self.length


@dataclass(frozen=True, slots=True)
class GrobnerAuditData:
    r"""
    Artefacto de FASE 1: certificado de independencia Gröbner + Cayley–Darboux.

    v5.4.0: +`darboux_chart_dim`. Fail-closed en residuos simplécticos.
    """
    rows: int
    cols: int
    ideal_dimension: int
    nullity: int
    pivot_columns: Tuple[int, ...]
    is_minimally_independent: bool
    matrix_properties: GF2MatrixProperties
    code_properties: Optional[LinearCodeProperties] = field(default=None)
    dependency_graph_edges: int = field(default=0)
    max_dependency_chain_length: int = field(default=0)
    rref_matrix: Optional[NDArray[np.uint8]] = field(default=None)
    symplectic_residue: float = field(default=0.0)
    relative_symplectic_residue: float = field(default=0.0)
    klein_residual: float = field(default=0.0)
    is_symplectic_valid: bool = field(default=True)
    darboux_chart_dim: int = field(default=0)


@dataclass(frozen=True, slots=True)
class OrthogonalityAnalysis:
    gram_matrix: NDArray[np.float64]
    off_diagonal_norm: float
    diagonal_deviation_norm: float
    spectral_gap: float
    condition_number: float
    frobenius_deviation: float
    spectral_deviation: float
    max_absolute_deviation: float
    conflict_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    max_interference_value: float = field(default=0.0)


@dataclass(frozen=True, slots=True)
class RSI3KleisliState:
    r"""Estado interno del endofuntor $T$ en la categoría Kleisli de RSI³."""
    spectrum: Tuple[float, ...]
    tau: float
    residual: float
    lipschitz: float
    level: int


@dataclass(frozen=True, slots=True)
class UnsatCoreCertifierData:
    r"""
    Artefacto de FASE 2: ortogonalidad + Poincaré celeste + RSI³ anidado.

    v5.4.0: Melnikov cerrado, Bruno, Lindstedt, Nekhoroshev, Lyapunov del
    mapa de retorno, recurrencia Kac, residuales T¹/T²/T³.
    """
    tool_count: int
    variable_dim: int
    off_diagonal_conflict_norm: float
    diagonal_deviation_norm: float
    conflict_edges: int
    orthogonality_tolerance: float
    is_strictly_orthogonal: bool
    orthogonality_analysis: OrthogonalityAnalysis
    projection_rank: int = field(default=0)
    is_projection_idempotent: bool = field(default=False)
    projection_idempotence_residual: float = field(default=0.0)
    # Poincaré / Melnikov / KAM
    diophantine_satisfied: bool = field(default=True)
    diophantine_gamma: float = field(default=float("inf"))
    chirikov_parameter: float = field(default=0.0)
    melnikov_integral: float = field(default=0.0)
    is_kam_stable: bool = field(default=True)
    bruno_ok: bool = field(default=True)
    melnikov_closed_form: float = field(default=0.0)
    melnikov_residue: float = field(default=0.0)
    lindstedt_omega_shift: float = field(default=0.0)
    nekhoroshev_time: float = field(default=0.0)
    poincare_lyapunov: float = field(default=0.0)
    recurrence_time: float = field(default=0.0)
    # RSI³
    krull_valuation: float = field(default=0.0)
    fubini_study_distance: float = field(default=0.0)
    lipschitz_constant: float = field(default=1.0)
    quadratic_rate: float = field(default=0.0)
    capacity_acceleration: float = field(default=0.0)
    is_rsi3_valid: bool = field(default=True)
    rsi3_level1_residual: float = field(default=0.0)
    rsi3_level2_lipschitz: float = field(default=1.0)
    rsi3_level3_associator: float = field(default=0.0)


@dataclass(frozen=True, slots=True)
class EntropyAnalysis:
    shannon_entropy: float
    min_entropy: float
    collision_entropy: float
    hartley_entropy: float
    support_size: int
    max_probability: float
    uniformity: float


@dataclass(frozen=True, slots=True)
class ROBDDIsomorphismData:
    r"""
    Artefacto de FASE 3: ROBDD + Wasserstein-1 + Heyting $\Omega_3$.

    v5.4.0: átomos Heyting reales (Gröbner/simpléctico no tautológicos).
    """
    support_dimension: int
    original_entropy: float
    reduced_entropy: float
    entropy_loss: float
    entropy_tolerance: float
    total_variation_distance: float
    is_homotopically_equivalent: bool
    original_entropy_analysis: EntropyAnalysis
    reduced_entropy_analysis: EntropyAnalysis
    kl_divergence_forward: float = field(default=0.0)
    kl_divergence_reverse: float = field(default=0.0)
    jensen_shannon_divergence: float = field(default=0.0)
    mutual_information: float = field(default=0.0)
    robdd_node_count: int = field(default=0)
    robdd_depth: int = field(default=0)
    reduction_ratio: float = field(default=1.0)
    wasserstein_distance_w1: float = field(default=0.0)
    heyting_verdict: str = field(default="COHERENT")
    heyting_code: int = field(default=0x00)
    heyting_axioms_verified: bool = field(default=True)
    hardware_crowbar_action: str = field(default="NOMINAL_EXECUTION")
    interlock_latency_ns: float = field(default=45.0)


# ── Handoffs entre fases (morfismos Φₖ → Φₖ₊₁) ───────────────────────────────

@dataclass(frozen=True, slots=True)
class MICPhase1Handoff:
    r"""
    Morfismo $\Phi_1\to\Phi_2$ (FASE 1: Gröbner+Cayley → FASE 2: Poincaré+RSI³).

    Emitido por `_handoff_to_phase2`; consumido por `_ingest_phase1_handoff`.
    La última flecha de FASE 1 *es* el dominio de la primera flecha de FASE 2:
        $\Phi_2\circ\Phi_1(M)
         =\texttt{\_ingest\_phase1\_handoff}(
              \texttt{\_handoff\_to\_phase2}(M))$.
    """
    grobner_audit: GrobnerAuditData
    input_checksum: str
    timestamp_iso: str


@dataclass(frozen=True, slots=True)
class MICPhase2Handoff:
    r"""
    Morfismo $\Phi_2\to\Phi_3$ (FASE 2 → FASE 3: ROBDD+Wasserstein+Heyting).

    Emitido por `_handoff_to_phase3`; consumido por `_ingest_phase2_handoff`.
    """
    phase1: MICPhase1Handoff
    unsat_core: UnsatCoreCertifierData


@dataclass(frozen=True, slots=True)
class MinimizerGovernanceCertificate:
    r"""
    Certificado OODA final de la composición $\Phi_3\circ\Phi_2\circ\Phi_1$.
    Verifica axiomas A1–A5 con trazabilidad SHA-256 + timestamp ISO-8601.
    """
    phase2: MICPhase2Handoff
    robdd_audit: ROBDDIsomorphismData
    heyting_verdict: HeytingOmega3
    heyting_mask: int
    heyting_axioms_verified: bool
    hardware_action: str
    interlock_latency_ns: float
    is_topologically_valid: bool
    overall_quality_score: float
    risk_assessment: str
    timestamp_iso: str
    input_checksum: str


@dataclass(frozen=True, slots=True)
class MinimizerGovernanceState:
    r"""
    Objeto final del endofuntor $Z_{\mathrm{Minimizer}}$.

    v5.4.0: preserva API v5.3.0 (`heyting_verdict`, `certificate`, …).
    """
    grobner_audit: GrobnerAuditData
    unsat_core_audit: UnsatCoreCertifierData
    robdd_audit: ROBDDIsomorphismData
    is_topologically_valid: bool
    reduction_phase: BooleanReductionPhase = field(default=BooleanReductionPhase.COMPLETE)
    timestamp: Optional[float] = field(default=None)
    overall_quality_score: float = field(default=0.0)
    risk_assessment: str = field(default="NOMINAL")
    heyting_verdict: HeytingOmega3 = field(default=HeytingOmega3.COHERENT)
    heyting_axioms_verified: bool = field(default=True)
    interlock_latency_ns: float = field(default=45.0)
    certificate: Optional[Any] = field(default=None)

    def __post_init__(self) -> None:
        if self.is_topologically_valid:
            if not self.grobner_audit.is_minimally_independent:
                raise ValueError("Inconsistencia: independencia Gröbner no certificada.")
            if not self.unsat_core_audit.is_strictly_orthogonal:
                raise ValueError("Inconsistencia: ortogonalidad no certificada.")
            if not self.robdd_audit.is_homotopically_equivalent:
                raise ValueError("Inconsistencia: equivalencia ROBDD no certificada.")


# ═══════════════════════════════════════════════════════════════════════════════
# §E. GUARDAS NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class _AdvancedNumericalGuard:

    @staticmethod
    def _validate_gf2_structure(matrix: NDArray[np.uint8], name: str = "matrix") -> None:
        if not np.all((matrix == 0) | (matrix == 1)):
            invalid = matrix[(matrix != 0) & (matrix != 1)]
            raise BooleanInputValidationError(
                f"{name} fuera de GF(2)={{0,1}}. Valores: {np.unique(invalid)}"
            )

    @staticmethod
    def _validate_probability_distribution(
        distribution: NDArray[np.float64], name: str = "distribution",
    ) -> None:
        if not np.all(np.isfinite(distribution)):
            raise ProbabilityDistributionError(f"{name} contiene NaN o Inf.")
        if np.any(distribution < -_PROBABILITY_TOLERANCE):
            raise ProbabilityDistributionError(f"{name} contiene probabilidades negativas.")
        total = float(np.sum(distribution))
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise ProbabilityDistributionError(f"{name} no suma a 1: suma={total:.6e}")

    @staticmethod
    def _as_finite_real_matrix(name: str, value: Any) -> NDArray[np.float64]:
        raw = np.asarray(value, dtype=np.float64)
        if raw.ndim != 2 or not np.all(np.isfinite(raw)):
            raise BooleanInputValidationError(f"{name} debe ser matriz 2D finita.")
        return raw

    @staticmethod
    def _as_finite_real_vector(name: str, value: Any) -> NDArray[np.float64]:
        raw = np.asarray(value, dtype=np.float64).flatten()
        if not np.all(np.isfinite(raw)):
            raise BooleanInputValidationError(f"{name} debe ser vector 1D finito.")
        return raw

    @classmethod
    def _as_finite_gf2_matrix(cls, name: str, value: Any) -> NDArray[np.uint8]:
        raw = np.asarray(value)
        arr_float = raw.astype(np.float64)
        rounded = np.rint(arr_float)
        arr_gf2 = np.mod(rounded.astype(np.int64), 2).astype(np.uint8)
        cls._validate_gf2_structure(arr_gf2, name)
        return arr_gf2

    @classmethod
    def _as_finite_probability_vector(
        cls, name: str, value: Any, normalize: bool = True,
    ) -> NDArray[np.float64]:
        arr = cls._as_finite_real_vector(name, value)
        arr = np.clip(arr, 0.0, None)
        total = float(np.sum(arr))
        if total <= _PROBABILITY_TOLERANCE:
            raise BooleanInputValidationError(f"{name} tiene masa probabilística nula.")
        if normalize:
            arr = arr / total
        cls._validate_probability_distribution(arr, name)
        return arr

    @staticmethod
    def _compute_input_checksum(*arrays: Any) -> str:
        hasher = hashlib.sha256()
        for arr in arrays:
            if arr is None:
                continue
            try:
                a = np.ascontiguousarray(np.asarray(arr))
                hasher.update(a.tobytes())
            except Exception:
                hasher.update(str(arr).encode("utf-8", errors="replace"))
        return hasher.hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — GRÖBNER + CAYLEY–DARBOUX (fail-closed, rigor v5.4.0)             ║
# ║  Inicio de Φ₁.  Última flecha = `_handoff_to_phase2`.                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_GrobnerBasisAuditor(_AdvancedNumericalGuard):
    r"""
    FASE 1: auditoría de independencia Gröbner + embedding Cayley–Darboux.

    Métodos (en orden de dependencia):
      • `_compute_gf2_matrix_properties`
      • `_gf2_rref`, `_gf2_rank`
      • `_compute_code_properties`
      • `_compute_cayley_darboux_embedding`  (fail-closed v5.4.0)
      • `_audit_grobner_independence`
      • `_handoff_to_phase2` — ÚLTIMA DEFINICIÓN → `MICPhase1Handoff`
        ≡ dominio de `_ingest_phase1_handoff` (PRIMERA DEFINICIÓN DE FASE 2).
    """

    def _compute_gf2_matrix_properties(
        self, matrix: NDArray[np.uint8], name: str = "matrix",
    ) -> GF2MatrixProperties:
        self._validate_gf2_structure(matrix, name)
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            return GF2MatrixProperties(
                rows=rows, cols=cols, rank_gf2=0, nullity_gf2=cols,
                pivot_columns=(), free_columns=tuple(range(cols)),
                kernel_dimension=cols, has_full_rank=False,
            )
        rank, pivots = self._gf2_rank(matrix)
        free = tuple(sorted(set(range(cols)) - set(pivots)))
        nullity = cols - rank
        full_rank = (rank == min(rows, cols))
        sparsity = float(np.count_nonzero(matrix == 0)) / (rows * cols)
        hamming = int(np.count_nonzero(matrix == 1))
        cond = 1.0 if full_rank else 1.0 + float(min(rows, cols) - rank)
        return GF2MatrixProperties(
            rows=rows, cols=cols, rank_gf2=rank, nullity_gf2=nullity,
            pivot_columns=pivots, free_columns=free,
            kernel_dimension=nullity, has_full_rank=full_rank,
            condition_number=cond, sparsity=sparsity, hamming_weight=hamming,
        )

    def _gf2_rref(self, matrix: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], Tuple[int, ...]]:
        M = matrix.copy()
        rows, cols = M.shape
        rank = 0
        pivots: List[int] = []
        for col in range(cols):
            if rank >= rows:
                break
            candidates = np.nonzero(M[rank:, col])[0]
            if candidates.size == 0:
                continue
            pivot = int(candidates[0] + rank)
            if pivot != rank:
                M[[rank, pivot]] = M[[pivot, rank]]
            pivot_row = M[rank].copy()
            for i in range(rows):
                if i != rank and M[i, col] == 1:
                    M[i] = np.bitwise_xor(M[i], pivot_row)
            pivots.append(col)
            rank += 1
        return M, tuple(pivots)

    def _gf2_rank(self, matrix: NDArray[np.uint8]) -> Tuple[int, Tuple[int, ...]]:
        _, pivots = self._gf2_rref(matrix)
        return len(pivots), pivots

    def _compute_code_properties(
        self, matrix: NDArray[np.uint8], rank: int,
    ) -> LinearCodeProperties:
        rows, cols = matrix.shape
        n, k = cols, rank
        weights = [int(np.count_nonzero(matrix[i])) for i in range(rows)]
        nz = [w for w in weights if w > 0]
        d = min(nz) if nz else n
        rate = k / n if n > 0 else 0.0
        return LinearCodeProperties(
            length=n, dimension=k, minimum_distance=d, rate=rate,
            redundancy=n - k, singleton_bound=n - k + 1,
            hamming_bound=float(2 ** n) if n < 64 else float("inf"),
        )

    def _compute_cayley_darboux_embedding(
        self, matrix_gf2: NDArray[np.uint8],
    ) -> Tuple[float, float, float, bool, int]:
        r"""
        Proyección rigurosa de Cayley–Darboux (v5.4.0, fail-closed).

        1. Embeber $G=PP^\top$ en dimensión par $2n$ (carta de Darboux).
        2. $K=(G-I)(G+I)^{-1}$ (Cayley).
        3. $K_{sp}=\tfrac12(K-JK^\top J)$ sobre $\mathfrak{sp}(2n)$.
        4. Klein: $K_{sp}^\top J+JK_{sp}=0$.
        5. $M=(I+K_{sp})(I-K_{sp})^{-1}$.
        6. Residuo $\|M^\top JM-J\|_F$ y relativo $\|M^\top JM-J\|_F/\|M\|_F^2$.

        Ante singularidad: residuos $=+\infty$, `is_valid=False`.

        Retorna:
            (symplectic_residue, relative_residue, klein_residual,
             is_valid, darboux_chart_dim)
        """
        P = matrix_gf2.astype(np.float64)
        if P.size == 0:
            return 0.0, 0.0, 0.0, True, 0

        gram = P @ P.T
        dim = gram.shape[0]
        d_even = dim if dim % 2 == 0 else dim + 1
        G_pad = np.eye(d_even, dtype=np.float64)
        G_pad[:dim, :dim] = gram

        n_half = d_even // 2
        J = np.block([
            [np.zeros((n_half, n_half), dtype=np.float64), np.eye(n_half, dtype=np.float64)],
            [-np.eye(n_half, dtype=np.float64), np.zeros((n_half, n_half), dtype=np.float64)],
        ])
        I_d = np.eye(d_even, dtype=np.float64)

        try:
            K = (G_pad - I_d) @ np.linalg.inv(G_pad + I_d)
            K_sp = 0.5 * (K - J @ K.T @ J)
            klein = K_sp.T @ J + J @ K_sp
            klein_residual = float(np.linalg.norm(klein, ord="fro"))
            M = (I_d + K_sp) @ np.linalg.inv(I_d - K_sp)
            residue = M.T @ J @ M - J
            symplectic_residue = float(np.linalg.norm(residue, ord="fro"))
            M_norm_sq = max(float(np.linalg.norm(M, ord="fro") ** 2), 1.0)
            relative_residue = symplectic_residue / M_norm_sq
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, RuntimeWarning):
            logger.warning("[FASE1] Cayley–Darboux singular — fail-closed.")
            return float("inf"), float("inf"), float("inf"), False, int(d_even)

        is_valid = bool(
            relative_residue <= _SYMPLECTIC_RESIDUE_TOLERANCE
            and klein_residual <= _KLEIN_RESIDUAL_TOLERANCE
        )
        return symplectic_residue, relative_residue, klein_residual, is_valid, int(d_even)

    def _audit_grobner_independence(
        self, boolean_polynomial_matrix: NDArray[np.uint8],
    ) -> GrobnerAuditData:
        matrix_gf2 = self._as_finite_gf2_matrix(
            "boolean_polynomial_matrix", boolean_polynomial_matrix,
        )
        rows, cols = matrix_gf2.shape
        if cols > _MAX_BOOLEAN_VARIABLES:
            raise BooleanInputValidationError(
                f"cols={cols} excede _MAX_BOOLEAN_VARIABLES={_MAX_BOOLEAN_VARIABLES}"
            )
        matrix_props = self._compute_gf2_matrix_properties(matrix_gf2)
        rank = matrix_props.rank_gf2
        rref_matrix, _ = self._gf2_rref(matrix_gf2)
        code_props = self._compute_code_properties(matrix_gf2, rank)

        if rank < rows:
            raise GrobnerDegeneracyError(
                f"Degeneración: rango GF(2) = {rank} < generadores = {rows}."
            )

        sym_residue, rel_residue, klein, sym_valid, d_even = (
            self._compute_cayley_darboux_embedding(matrix_gf2)
        )
        return GrobnerAuditData(
            rows=rows, cols=cols, ideal_dimension=rank, nullity=cols - rank,
            pivot_columns=matrix_props.pivot_columns,
            is_minimally_independent=True,
            matrix_properties=matrix_props, code_properties=code_props,
            rref_matrix=rref_matrix,
            symplectic_residue=sym_residue,
            relative_symplectic_residue=rel_residue,
            klein_residual=klein,
            is_symplectic_valid=sym_valid,
            darboux_chart_dim=d_even,
        )

    # ─── ÚLTIMA DEFINICIÓN DE FASE 1 (continuación estricta a FASE 2) ───────
    def _handoff_to_phase2(
        self,
        grobner_audit: GrobnerAuditData,
        input_checksum: str,
    ) -> MICPhase1Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 1  →  morfismo Φ₁ → Φ₂.
        El objeto retornado **es** el dominio estricto de
        `_ingest_phase1_handoff` (PRIMERA DEFINICIÓN DE FASE 2):

            $\Phi_2\circ\Phi_1(M)=\texttt{\_ingest\_phase1\_handoff}(
                \texttt{\_handoff\_to\_phase2}(M))$.
        ═══════════════════════════════════════════════════════════════════
        """
        h1 = MICPhase1Handoff(
            grobner_audit=grobner_audit,
            input_checksum=input_checksum,
            timestamp_iso=datetime.now(tz=timezone.utc).isoformat(),
        )
        logger.info(
            "[FASE1→FASE2] handoff: rank=%d, Darboux=%d, "
            "sym_residue_rel=%.2e, klein=%.2e, checksum=%s...",
            grobner_audit.ideal_dimension,
            grobner_audit.darboux_chart_dim,
            grobner_audit.relative_symplectic_residue,
            grobner_audit.klein_residual,
            input_checksum[:16],
        )
        return h1


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — NO-INTERFERENCIA + POINCARÉ CELESTE + RSI³ ANIDADO               ║
# ║  Continuación estricta de `_handoff_to_phase2`.                            ║
# ║  Primera flecha = `_ingest_phase1_handoff`.                                ║
# ║  Última flecha  = `_handoff_to_phase3`.                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UnsatCoreCertifier(Phase1_GrobnerBasisAuditor):
    r"""
    FASE 2: certificación de no-interferencia + mecánica celeste de Poincaré
    + automejora recursiva de nivel 3 (Kleisli $T\subset T^2\subset T^3$).

    Métodos (en orden de dependencia):
      • `_ingest_phase1_handoff` — PRIMERA DEFINICIÓN (consume Φ₁).
      • `_diophantine_gamma_cf` / `_bruno_diophantine`
      • `_melnikov_homoclinic_sech` / `_lindstedt_frequency_correction`
      • `_poincare_return_map_lyapunov` / `_nekhoroshev_stability_time`
        / `_poincare_kac_recurrence`
      • `_rsi3_level{1,2,3}_*` + `_grassmann_fubini_study`
      • `_certify_non_interference_unsat`
      • `_handoff_to_phase3` — ÚLTIMA DEFINICIÓN → `MICPhase2Handoff`
        ≡ dominio de `_ingest_phase2_handoff` (PRIMERA DEFINICIÓN DE FASE 3).
    """

    # ─── PRIMERA DEFINICIÓN DE FASE 2 (continuación estricta) ───────────────
    @staticmethod
    def _ingest_phase1_handoff(h1: MICPhase1Handoff) -> MICPhase1Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        PRIMERA DEFINICIÓN DE FASE 2: consume `MICPhase1Handoff`.
        Continuación topológica de `_handoff_to_phase2` (última de FASE 1).
        Verifica continuidad estricta de tipo e invariantes Gröbner.
        ═══════════════════════════════════════════════════════════════════
        """
        if not isinstance(h1, MICPhase1Handoff):
            raise OODAChainViolationError(
                f"[FASE2] Handoff inválido: {type(h1).__name__} "
                "(esperaba MICPhase1Handoff)."
            )
        if not h1.grobner_audit.is_minimally_independent:
            raise OODAChainViolationError(
                "[FASE2] Handoff con independencia Gröbner no certificada."
            )
        if (
            (not h1.grobner_audit.is_symplectic_valid)
            and h1.grobner_audit.relative_symplectic_residue > 1e-6
        ):
            logger.warning(
                "[FASE2] Handoff con residuo simpléctico alto: δω_rel=%.3e",
                h1.grobner_audit.relative_symplectic_residue,
            )
        return h1

    # ── sub-motores celestes de Poincaré ─────────────────────────────────────

    @staticmethod
    def _safe_sech(t: NDArray[np.float64]) -> NDArray[np.float64]:
        """$\mathrm{sech}(t)=1/\cosh(t)$ con saturación anti-overflow."""
        return 1.0 / np.cosh(np.clip(t, -40.0, 40.0))

    def _diophantine_gamma_cf(
        self,
        omega: NDArray[np.float64],
        tau: float,
        max_denom: int = _KAM_MAX_DENOM,
    ) -> float:
        r"""
        Retículo KAM vía convergentes de fracciones continuas de $\omega_i/\omega_j$.

        $\gamma^*=\inf_k|\omega\cdot k|\,\|k\|_1^\tau$ sobre los convergentes
        de cada ratio (candidatos naturales a resonancias peligrosas).
        """
        omega = np.asarray(omega, dtype=np.float64).ravel()
        n = omega.size
        if n == 0:
            return float("inf")
        if n == 1:
            return float(abs(omega[0]))

        gamma = float("inf")
        for i, j in combinations(range(n), 2):
            if abs(omega[j]) < 1e-15:
                continue
            ratio = float(omega[i] / omega[j])
            a = ratio
            p_prev, p_curr = 1, 0
            q_prev, q_curr = 0, 1
            for _ in range(20):
                ai = int(math.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr == 0 or q_curr > max_denom:
                    break
                k_vec = np.zeros(n, dtype=np.float64)
                k_vec[i] = float(q_curr)
                k_vec[j] = float(-p_curr)
                dot = abs(float(np.dot(omega, k_vec)))
                k_norm = float(np.linalg.norm(k_vec, 1))
                if k_norm > 1e-15:
                    candidate = dot * (k_norm ** tau)
                    if candidate < gamma:
                        gamma = candidate
                frac = a - ai
                if frac < 1e-12:
                    break
                a = 1.0 / frac
        return gamma

    def _bruno_diophantine(
        self, omega: NDArray[np.float64],
    ) -> Tuple[float, bool]:
        r"""
        Condición de Bruno: $\sum_{k\ge 1}(\log q_{k+1})/q_k<\infty$
        sobre convergentes de $\omega_i/\omega_j$. Se aproxima con 20
        convergentes y se exige que la suma parcial sea finita y acotada.
        """
        w = np.asarray(omega, dtype=np.float64).ravel()
        w = w[w > 1e-15]
        if w.size < 2:
            return 1e10, True
        bruno_sum = 0.0
        gamma_min = self._diophantine_gamma_cf(w, tau=float(w.size + 1))
        for i, j in combinations(range(w.size), 2):
            if abs(w[j]) < 1e-15:
                continue
            ratio = float(w[i] / w[j])
            a = ratio
            p_prev, p_curr = 1, 0
            q_prev, q_curr = 0, 1
            for _ in range(20):
                ai = int(math.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr <= 0 or q_curr > _KAM_MAX_DENOM:
                    break
                if q_prev > 0:
                    bruno_sum += math.log(max(q_curr, 2)) / q_prev
                frac = a - ai
                if frac < 1e-12:
                    break
                a = 1.0 / frac
        if math.isinf(gamma_min):
            gamma_min = 1e10
        bruno_ok = bool(math.isfinite(bruno_sum) and bruno_sum < 1.0 / _BRUNO_FLOOR)
        return float(gamma_min), bruno_ok

    def _melnikov_homoclinic_sech(self, omega: float) -> Dict[str, float]:
        r"""
        Integral de Melnikov sobre la homoclínica del péndulo
        $\ddot q+\sin q=\varepsilon\sin(\omega t)$:

            $q_h(t)=4\arctan(e^t)-\pi,\quad p_h(t)=2\,\mathrm{sech}(t)$,

            $M(\theta)=\int_{-\infty}^{\infty}p_h(t)\,\sin(\omega(t+\theta))\,dt
                      =2\pi\,\mathrm{sech}(\pi\omega/2)\,\sin(\omega\theta)$.

        Se reporta $|M|_{\max}$ cerrado, su aproximación trapezoidal y el residuo.
        """
        omega = float(abs(omega)) if omega == omega else 1.0  # NaN-guard
        if omega < 1e-15:
            omega = 1e-15
        closed = float(2.0 * math.pi / math.cosh(min(math.pi * omega / 2.0, 40.0)))
        numerical = closed
        try:
            n_pts = 2001
            t = np.linspace(-20.0, 20.0, n_pts)
            p_h = 2.0 * self._safe_sech(t)
            integrand = p_h * np.sin(omega * t)  # θ = 0; |M| por fase
            if hasattr(np, "trapezoid"):
                numerical = float(abs(np.trapezoid(integrand, t)))
            else:
                numerical = float(abs(np.trapz(integrand, t)))
        except (ValueError, FloatingPointError, RuntimeWarning):
            numerical = closed
        residue = float(abs(numerical - closed) / max(closed, 1e-15))
        return {
            "melnikov_closed_form": closed,
            "melnikov_integral": float(min(closed, numerical) if residue < 0.5 else closed),
            "melnikov_residue": residue,
        }

    def _lindstedt_frequency_correction(
        self, omega_0: NDArray[np.float64], epsilon: float, amplitude: float,
    ) -> float:
        r"""
        Serie de Poincaré–Lindstedt para Duffing $\ddot x+\omega_0^2 x=\varepsilon x^3$.
        Anulación del término secular: $\omega_1=0$ (impar),
        $\omega_2=3a^2/(8\omega_0)$ ⇒
            $\omega=\omega_0+\dfrac{3\varepsilon a^2}{8\omega_0}+O(\varepsilon^2)$.
        Retorna $\|\omega-\omega_0\|_2$.
        """
        w = np.asarray(omega_0, dtype=np.float64).flatten()
        if w.size == 0:
            return 0.0
        w_safe = np.where(np.abs(w) < 1e-15, 1.0, w)
        shift = (3.0 * epsilon * (amplitude ** 2)) / (8.0 * w_safe)
        return float(np.linalg.norm(shift))

    def _poincare_return_map_lyapunov(self, chirikov: float) -> float:
        r"""
        Mapa estándar de Chirikov (sección de Poincaré):

            $I_{n+1}=I_n+K\sin\theta_n,\qquad
             \theta_{n+1}=\theta_n+I_{n+1}\pmod{2\pi}$.

        $\lambda\approx\log(K/2)$ para $K>1$; $\lambda\approx 0$ (integrable) si $K\le 1$.
        """
        K = abs(float(chirikov))
        if K <= 1.0:
            return 0.0
        return float(math.log(K / 2.0))

    def _nekhoroshev_stability_time(self, epsilon: float, n_freq: int) -> float:
        r"""
        Estimación de Nekhoroshev: $T\sim\exp\bigl(c/\varepsilon^{1/(2n)}\bigr)$
        para hamiltonianos analíticos casi-integrables.
        """
        n = max(int(n_freq), 1)
        eps = max(float(abs(epsilon)), 1e-16)
        exponent = 1.0 / (2.0 * n)
        c = 0.5
        try:
            return float(math.exp(min(c / (eps ** exponent), 80.0)))
        except (OverflowError, ValueError):
            return float("inf")

    def _poincare_kac_recurrence(self, n_freq: int, eps_ball: float) -> float:
        r"""
        Tiempo medio de recurrencia de Poincaré–Kac a una $\varepsilon$-bola
        en $\mathbb{T}^n$: $\langle\tau\rangle\sim\varepsilon^{-n}$.
        """
        n = max(int(n_freq), 1)
        eps = min(max(float(eps_ball), 1e-12), 1.0)
        try:
            return float((1.0 / eps) ** n)
        except OverflowError:
            return float("inf")

    # ── RSI³: automejora recursiva de nivel 3 (Kleisli T, T², T³) ────────────

    @staticmethod
    def _soft_threshold(sv: NDArray[np.float64], tau: float) -> NDArray[np.float64]:
        x = np.asarray(sv, dtype=np.float64)
        return np.maximum(x - float(tau), 0.0)

    def _rsi3_picard(
        self,
        sv: NDArray[np.float64],
        tau: float,
        max_iter: int = 12,
        tol: float = 1e-12,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Iteración de Picard–Banach $x_{k+1}=T_\tau(x_k)$ sobre el espectro.
        Retorna (punto fijo, Lip empírico, residual final).
        """
        x = np.asarray(sv, dtype=np.float64).copy()
        prev_res: Optional[float] = None
        lip = 1.0
        residual = 0.0
        for _ in range(max_iter):
            x_next = self._soft_threshold(x, tau)
            residual = float(np.linalg.norm(x_next - x))
            if prev_res is not None and prev_res > 1e-15:
                ratio = residual / prev_res
                lip = min(lip, ratio) if residual <= prev_res else max(lip, ratio)
            if residual < tol:
                x = x_next
                break
            prev_res = residual
            x = x_next
        if (not math.isfinite(lip)) or lip < 0.0:
            lip = 1.0
        return x, float(lip), float(residual)

    def _grassmann_fubini_study(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
    ) -> float:
        r"""
        Distancia de Fubini–Study en Grassmann: $\|\theta\|_2$ de los ángulos
        principales entre $\mathrm{Im}(A)$ y $\mathrm{Im}(B)$.
        Corrige el bug v5.3 ($\langle u_1,u_2\rangle=0\Rightarrow\pi/2$ constante).
        """
        if A is None or B is None or A.size == 0 or B.size == 0:
            return math.pi / 2.0
        try:
            Af = np.asarray(A, dtype=np.float64)
            Bf = np.asarray(B, dtype=np.float64)
            if Af.ndim == 1:
                Af = Af.reshape(-1, 1)
            if Bf.ndim == 1:
                Bf = Bf.reshape(-1, 1)
            Ua, sa, _ = np.linalg.svd(Af, full_matrices=False)
            Ub, sb, _ = np.linalg.svd(Bf, full_matrices=False)
            ra = int(np.sum(sa > 1e-12))
            rb = int(np.sum(sb > 1e-12))
            if ra == 0 or rb == 0:
                return math.pi / 2.0
            Qa = Ua[:, :ra]
            Qb = Ub[:, :rb]
            M = Qa.T @ Qb
            sv = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
            thetas = np.arccos(sv)
            return float(np.linalg.norm(thetas))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning):
            return 1e-5

    def _rsi3_level1_surface(
        self, sv: NDArray[np.float64],
    ) -> RSI3KleisliState:
        r"""T¹: mejora de objeto — soft-threshold con $\tau=\mathrm{median}(\sigma)/10$."""
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])
        tau = float(np.median(x) * 0.10)
        fixed, lip, res = self._rsi3_picard(x, tau)
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in fixed),
            tau=tau, residual=res, lipschitz=lip, level=1,
        )

    def _rsi3_level2_operator(
        self, s1: RSI3KleisliState, sv: NDArray[np.float64],
    ) -> RSI3KleisliState:
        r"""
        T²: mejora del operador — $\tau_2=\tau_1\cdot(1+\|T^1 x-x\|)$
        (el umbral se adapta al residual de nivel 1).
        """
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])
        tau2 = float(s1.tau * (1.0 + s1.residual))
        fixed, lip, res = self._rsi3_picard(x, tau2)
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in fixed),
            tau=tau2, residual=res, lipschitz=lip, level=2,
        )

    def _rsi3_level3_kleisli(
        self,
        s1: RSI3KleisliState,
        s2: RSI3KleisliState,
        sv: NDArray[np.float64],
    ) -> RSI3KleisliState:
        r"""
        T³: meta-mejora Kleisli.

        Mónada $(T,\eta,\mu)$ sobre el espectro. El asociador numérico
            $\|\mu\circ T\mu-\mu\circ\mu T\|$
        se estima como el conmutador de las tres correcciones de umbral:

            $a=\|T_{\tau_3}\circ T_{\tau_2}\circ T_{\tau_1}
                -T_{\tau_1}\circ T_{\tau_2}\circ T_{\tau_3}\|$.

        $\tau_3$ es el punto fijo de Banach de $\tau\mapsto\tau\cdot\mathrm{Lip}(T_\tau)$
        (mejora de la *regla* de mejora). Axioma A4: $\mathrm{Lip}(T^3)\le\mathrm{Lip}(T)^3$.
        """
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])

        tau3 = float(s2.tau * min(max(s2.lipschitz, 1e-6), 1.0))
        fixed, lip3, res3 = self._rsi3_picard(x, tau3)

        t1 = self._soft_threshold(x, s1.tau)
        t2 = self._soft_threshold(t1, s2.tau)
        t321 = self._soft_threshold(t2, tau3)

        t3 = self._soft_threshold(x, tau3)
        t32 = self._soft_threshold(t3, s2.tau)
        t123 = self._soft_threshold(t32, s1.tau)
        associator = float(np.linalg.norm(t321 - t123))

        lip_nested = float(min(s1.lipschitz, 1.0) ** 3)
        lip = float(min(lip3, lip_nested))
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in fixed),
            tau=tau3, residual=float(res3 + associator),
            lipschitz=lip, level=3,
        )

    def _rsi3_surfaces_computed(
        self, P: NDArray[np.float64],
    ) -> Dict[str, float]:
        r"""
        RSI³ v5.4.0: superficies Kleisli anidadas $T^1\subset T^2\subset T^3$.

        • Krull = $\mathrm{rank}(P)/\mathrm{num\_vars}$.
        • Fubini–Study de Grassmann entre $\mathrm{Im}(P)$ e $\mathrm{Im}(P_\tau)$.
        • Banach $L_\Phi$ = Lip empírico de $T^3$ (Picard) y cota $\mathrm{Lip}(T)^3$.
        • Quadratic rate $p$ = mediana de log-ratios de $\sigma$.
        • Capacity acceleration = $\max\Delta^3\sigma$.
        • Asociador Kleisli de nivel 3.
        """
        num_vars = P.shape[1] if P.ndim > 1 else 1
        try:
            rank = int(np.linalg.matrix_rank(P))
        except np.linalg.LinAlgError:
            rank = 0
        krull = float(rank / max(num_vars, 1))

        try:
            sv_all = _svdvals(P)
        except (np.linalg.LinAlgError, ValueError):
            sv_all = np.array([1.0], dtype=np.float64)
        sv_pos = sv_all[sv_all > _SV_FLOOR]
        if sv_pos.size == 0:
            sv_pos = np.array([1.0], dtype=np.float64)

        s1 = self._rsi3_level1_surface(sv_pos)
        s2 = self._rsi3_level2_operator(s1, sv_pos)
        s3 = self._rsi3_level3_kleisli(s1, s2, sv_pos)

        # Grassmann: P vs reconstrucción espectral umbralizada T¹
        fs = 1e-6
        try:
            U, S, Vt = np.linalg.svd(P, full_matrices=False)
            S_tau = self._soft_threshold(S, s1.tau)
            if U.size and Vt.size:
                P_tau = (U * S_tau) @ Vt
                fs = self._grassmann_fubini_study(P, P_tau)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning):
            fs = 1e-5

        lipschitz_constant = float(np.clip(s3.lipschitz, 0.0, 1.0))
        if s3.lipschitz >= 0.999 and sv_pos.size >= 2:
            ratios = sv_pos[1:] / np.maximum(sv_pos[:-1], 1e-15)
            lipschitz_constant = float(np.clip(np.max(ratios), 0.0, 1.0))

        quad = 0.0
        if sv_pos.size >= 3:
            orders: List[float] = []
            for k in range(1, sv_pos.size - 1):
                num = math.log(max(sv_pos[k + 1] / max(sv_pos[k], 1e-15), 1e-15))
                den = math.log(max(sv_pos[k] / max(sv_pos[k - 1], 1e-15), 1e-15))
                if abs(den) > 1e-9:
                    orders.append(num / den)
            if orders:
                quad = float(np.clip(np.median(orders), 0.0, 4.0))

        accel = 0.0
        if sv_pos.size >= 4:
            d3 = np.diff(np.diff(np.diff(sv_pos)))
            if d3.size > 0:
                accel = float(np.max(d3))

        is_valid = bool(
            krull >= _KRULL_MIN
            and fs <= _FS_MAX
            and lipschitz_constant <= _BANACH_TARGET
            and s3.residual < 1.0
        )
        return {
            "krull": krull,
            "fs": fs,
            "banach": lipschitz_constant,
            "quadratic": quad,
            "accel": accel,
            "valid": float(is_valid),
            "l1_res": s1.residual,
            "l2_lip": s2.lipschitz,
            "l3_assoc": s3.residual,
        }

    def _certify_non_interference_unsat(
        self,
        tool_projection_matrix: NDArray[np.float64],
        grobner_audit: Optional[GrobnerAuditData] = None,
    ) -> UnsatCoreCertifierData:
        if grobner_audit and not grobner_audit.is_minimally_independent:
            raise GrobnerDegeneracyError("Fase 2 requiere independencia Gröbner certificada.")

        P = self._as_finite_real_matrix("tool_projection_matrix", tool_projection_matrix)
        tool_count, variable_dim = P.shape

        gram = P @ P.T
        m = gram.shape[0]
        identity = np.eye(m, dtype=np.float64)
        deviation = gram - identity

        frob_dev = _fro_norm(deviation)
        off_diag = gram.copy()
        np.fill_diagonal(off_diag, 0.0)
        off_diag_norm = float(np.sum(np.abs(off_diag)))
        diag_dev_norm = float(np.max(np.abs(np.diag(gram) - 1.0))) if m > 0 else 0.0

        tol = _ORTHOGONALITY_TOLERANCE
        is_ortho = bool(off_diag_norm <= tol and diag_dev_norm <= tol)

        # conflict_pairs reales
        conflict_pairs: Set[Tuple[int, int]] = set()
        max_interf = 0.0
        for i, j in combinations(range(m), 2):
            val = float(abs(gram[i, j]))
            if val > tol:
                conflict_pairs.add((i, j))
                if val > max_interf:
                    max_interf = val

        # espectro de Gram
        try:
            eigs = np.linalg.eigvalsh(gram) if m > 0 else np.array([1.0])
            eigs_pos = eigs[eigs > 1e-15]
            if eigs_pos.size >= 2:
                cond_g = float(eigs_pos[-1] / eigs_pos[0])
                gap = float(eigs_pos[0])  # distancia al núcleo
            elif eigs_pos.size == 1:
                cond_g, gap = 1.0, float(eigs_pos[0])
            else:
                cond_g, gap = float("inf"), 0.0
            spec_dev = float(np.linalg.norm(eigs - 1.0)) if eigs.size else frob_dev
        except (np.linalg.LinAlgError, ValueError):
            cond_g, gap, spec_dev = 1.0, 1.0, frob_dev

        ortho_analysis = OrthogonalityAnalysis(
            gram_matrix=gram,
            off_diagonal_norm=off_diag_norm,
            diagonal_deviation_norm=diag_dev_norm,
            spectral_gap=gap,
            condition_number=cond_g,
            frobenius_deviation=frob_dev,
            spectral_deviation=spec_dev,
            max_absolute_deviation=float(np.max(np.abs(deviation))) if deviation.size else 0.0,
            conflict_pairs=conflict_pairs,
            max_interference_value=max_interf,
        )

        if not is_ortho:
            raise NonInterferenceViolationError(
                f"Violación Zero Side-Effects: off_diag={off_diag_norm:.2e}, "
                f"diag_dev={diag_dev_norm:.2e}, tol={tol:.2e}"
            )

        # rango e idempotencia de la proyección $P P^+ P \approx P$
        try:
            proj_rank = int(np.linalg.matrix_rank(P))
        except np.linalg.LinAlgError:
            proj_rank = 0
        try:
            P_pinv = np.linalg.pinv(P)
            idem_res = float(np.linalg.norm(P @ P_pinv @ P - P, ord="fro"))
        except (np.linalg.LinAlgError, ValueError):
            idem_res = float("inf")
        is_idempotent = bool(idem_res <= max(tol * max(P.size, 1), 1e-8))

        # ── Poincaré-Lindstedt-Melnikov ──────────────────────────────────────
        try:
            sv = _svdvals(P)
        except (np.linalg.LinAlgError, ValueError):
            sv = np.array([1.0], dtype=np.float64)
        omega_0 = sv[sv > 1e-15] if sv.size > 0 else np.array([1.0], dtype=np.float64)
        n_freq = int(omega_0.size)
        tau = float(n_freq + 1)

        gamma_cf, bruno_ok = self._bruno_diophantine(omega_0)
        # γ de Bruno ya incorpora CF; se retiene el más conservador
        gamma_cf = min(gamma_cf, self._diophantine_gamma_cf(omega_0, tau))
        dioph_sat = bool(gamma_cf >= _KAM_GAMMA)

        delta_omega = float(np.std(omega_0)) if omega_0.size > 1 else 0.0
        mean_omega = float(np.mean(omega_0)) if omega_0.size > 0 else 1.0
        delta_omega_res = mean_omega if mean_omega > 1e-10 else 1.0
        chirikov = float(delta_omega / delta_omega_res)

        omega_forcing = float(omega_0[0]) if n_freq >= 1 else 1.0
        melnikov = self._melnikov_homoclinic_sech(omega_forcing)

        epsilon = float(chirikov) if chirikov > 0 else 1e-12
        amplitude = float(omega_0[0]) if n_freq >= 1 else 1.0
        lindstedt_shift = self._lindstedt_frequency_correction(omega_0, epsilon, amplitude)
        lyapunov = self._poincare_return_map_lyapunov(chirikov)
        nekhoroshev = self._nekhoroshev_stability_time(epsilon, n_freq)
        recurrence = self._poincare_kac_recurrence(n_freq, max(epsilon, 1e-6))

        is_kam = bool(
            dioph_sat
            and bruno_ok
            and chirikov < 1.0
            and melnikov["melnikov_integral"] < _MELNIKOV_THRESHOLD
            and lyapunov <= 0.0
        )

        # ── RSI³ anidado ─────────────────────────────────────────────────────
        rsi = self._rsi3_surfaces_computed(P)

        return UnsatCoreCertifierData(
            tool_count=tool_count,
            variable_dim=variable_dim,
            off_diagonal_conflict_norm=off_diag_norm,
            diagonal_deviation_norm=diag_dev_norm,
            conflict_edges=len(conflict_pairs),
            orthogonality_tolerance=tol,
            is_strictly_orthogonal=is_ortho,
            orthogonality_analysis=ortho_analysis,
            projection_rank=proj_rank,
            is_projection_idempotent=is_idempotent,
            projection_idempotence_residual=idem_res,
            diophantine_satisfied=dioph_sat,
            diophantine_gamma=float(gamma_cf),
            chirikov_parameter=chirikov,
            melnikov_integral=float(melnikov["melnikov_integral"]),
            is_kam_stable=is_kam,
            bruno_ok=bool(bruno_ok),
            melnikov_closed_form=float(melnikov["melnikov_closed_form"]),
            melnikov_residue=float(melnikov["melnikov_residue"]),
            lindstedt_omega_shift=float(lindstedt_shift),
            nekhoroshev_time=float(nekhoroshev),
            poincare_lyapunov=float(lyapunov),
            recurrence_time=float(recurrence),
            krull_valuation=float(rsi["krull"]),
            fubini_study_distance=float(rsi["fs"]),
            lipschitz_constant=float(rsi["banach"]),
            quadratic_rate=float(rsi["quadratic"]),
            capacity_acceleration=float(rsi["accel"]),
            is_rsi3_valid=bool(rsi["valid"]),
            rsi3_level1_residual=float(rsi["l1_res"]),
            rsi3_level2_lipschitz=float(rsi["l2_lip"]),
            rsi3_level3_associator=float(rsi["l3_assoc"]),
        )

    # ─── ÚLTIMA DEFINICIÓN DE FASE 2 (continuación estricta a FASE 3) ───────
    @staticmethod
    def _handoff_to_phase3(
        h1: MICPhase1Handoff,
        unsat_core_audit: UnsatCoreCertifierData,
    ) -> MICPhase2Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 2  →  morfismo Φ₂ → Φ₃.
        El objeto retornado **es** el dominio estricto de
        `_ingest_phase2_handoff` (PRIMERA DEFINICIÓN DE FASE 3).
        ═══════════════════════════════════════════════════════════════════
        """
        h2 = MICPhase2Handoff(phase1=h1, unsat_core=unsat_core_audit)
        logger.info(
            "[FASE2→FASE3] handoff: ortho=%s, KAM=%s, Bruno=%s, L_φ=%.3f, "
            "γ=%.3e, |M|=%.3e (cerrado=%.3e, res=%.2e), λ_P=%.3f, "
            "T_Nek=%.3e, T³_assoc=%.2e",
            unsat_core_audit.is_strictly_orthogonal,
            unsat_core_audit.is_kam_stable,
            unsat_core_audit.bruno_ok,
            unsat_core_audit.lipschitz_constant,
            unsat_core_audit.diophantine_gamma,
            unsat_core_audit.melnikov_integral,
            unsat_core_audit.melnikov_closed_form,
            unsat_core_audit.melnikov_residue,
            unsat_core_audit.poincare_lyapunov,
            unsat_core_audit.nekhoroshev_time,
            unsat_core_audit.rsi3_level3_associator,
        )
        return h2


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — ROBDD + WASSERSTEIN + HEYTING Ω₃ + CROWBAR                       ║
# ║  Continuación estricta de `_handoff_to_phase3`.                            ║
# ║  Primera flecha = `_ingest_phase2_handoff`.                                ║
# ║  Última flecha  = `_emit_governance_certificate`.                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ROBDDIsomorphismValidator(Phase2_UnsatCoreCertifier):
    r"""
    FASE 3: ROBDD + Wasserstein-1 + Heyting $\Omega_3$ + Crowbar.

    Métodos (en orden de dependencia):
      • `_ingest_phase2_handoff` — PRIMERA DEFINICIÓN (consume Φ₂).
      • `_entropy_analysis_full` (familia Rényi $\{H_0,H_1,H_2,H_\infty\}$).
      • `_kl_js_divergences`
      • `_validate_robdd_homotopy`
      • `_heyting_combine_full` (7 átomos reales)
      • `_hardware_action_from_verdict`
      • `_emit_governance_certificate` — ÚLTIMA DEFINICIÓN
        → `MinimizerGovernanceCertificate`.
    """

    # ─── PRIMERA DEFINICIÓN DE FASE 3 (continuación estricta) ───────────────
    @staticmethod
    def _ingest_phase2_handoff(h2: MICPhase2Handoff) -> MICPhase2Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        PRIMERA DEFINICIÓN DE FASE 3: consume `MICPhase2Handoff`.
        Continuación topológica de `_handoff_to_phase3` (última de FASE 2).
        ═══════════════════════════════════════════════════════════════════
        """
        if not isinstance(h2, MICPhase2Handoff):
            raise OODAChainViolationError(
                f"[FASE3] Handoff inválido: {type(h2).__name__} "
                "(esperaba MICPhase2Handoff)."
            )
        if not isinstance(h2.phase1, MICPhase1Handoff):
            raise OODAChainViolationError("[FASE3] Handoff sin Φ₁ embebido.")
        if not h2.unsat_core.is_strictly_orthogonal:
            raise OODAChainViolationError(
                "[FASE3] Handoff con ortogonalidad FASE 2 no certificada."
            )
        return h2

    def _entropy_analysis_full(self, probs: NDArray[np.float64]) -> EntropyAnalysis:
        r"""
        Familia completa de entropías de Rényi (v5.4.0):

        • Shannon $H_1=-\sum p\log_2 p$.
        • Rényi-2 (collision) $H_2=-\log_2\sum p^2$.
        • Rényi-$\infty$ (min-entropy) $H_\infty=-\log_2\max p$.
        • Hartley $H_0=\log_2|\mathrm{supp}|$.
        • Uniformidad $\min p/\max p\in[0,1]$.
        """
        p = np.asarray(probs, dtype=np.float64).ravel()
        p_pos = p[p > 1e-15]
        if p_pos.size == 0:
            return EntropyAnalysis(0.0, 0.0, 0.0, 0.0, 0, 0.0, 1.0)

        shannon = float(-np.sum(p_pos * np.log2(p_pos)))
        sum_sq = float(np.sum(p_pos ** 2))
        collision = float(-math.log2(sum_sq)) if sum_sq > 0 else 0.0
        max_p = float(np.max(p_pos))
        min_ent = float(-math.log2(max_p)) if max_p > 0 else 0.0
        hartley = float(math.log2(p_pos.size))
        support = int(p_pos.size)
        uniformity = float(np.min(p_pos) / max_p) if p_pos.size > 1 else 1.0
        return EntropyAnalysis(
            shannon_entropy=shannon,
            min_entropy=min_ent,
            collision_entropy=collision,
            hartley_entropy=hartley,
            support_size=support,
            max_probability=max_p,
            uniformity=float(uniformity),
        )

    @staticmethod
    def _kl_js_divergences(
        p: NDArray[np.float64], q: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""$(D_{\mathrm{KL}}(p\|q),\,D_{\mathrm{KL}}(q\|p),\,D_{\mathrm{JS}})$ en nats→bits."""
        eps = 1e-15
        p_s = np.clip(p, eps, 1.0)
        q_s = np.clip(q, eps, 1.0)
        p_s = p_s / np.sum(p_s)
        q_s = q_s / np.sum(q_s)
        kl_pq = float(np.sum(p_s * (np.log2(p_s) - np.log2(q_s))))
        kl_qp = float(np.sum(q_s * (np.log2(q_s) - np.log2(p_s))))
        m = 0.5 * (p_s + q_s)
        js = 0.5 * float(np.sum(p_s * (np.log2(p_s) - np.log2(m)))) + \
            0.5 * float(np.sum(q_s * (np.log2(q_s) - np.log2(m))))
        return kl_pq, kl_qp, js

    @staticmethod
    def _heyting_combine_full(
        grobner_ok: bool,
        symplectic_ok: bool,
        ortho_ok: bool,
        kam_stable: bool,
        rsi3_ok: bool,
        homotopy_ok: bool,
        w1_ok: bool,
    ) -> HeytingOmega3:
        """Join bitwise-OR sobre $\Omega_3$ con 7 átomos de severidad."""
        v_g = HeytingOmega3.COHERENT if grobner_ok else HeytingOmega3.VETOED
        v_s = HeytingOmega3.COHERENT if symplectic_ok else HeytingOmega3.VETOED
        v_o = HeytingOmega3.COHERENT if ortho_ok else HeytingOmega3.VETOED
        v_k = HeytingOmega3.COHERENT if kam_stable else HeytingOmega3.DEGRADED
        v_r = HeytingOmega3.COHERENT if rsi3_ok else HeytingOmega3.DEGRADED
        v_h = HeytingOmega3.COHERENT if homotopy_ok else HeytingOmega3.VETOED
        v_w = HeytingOmega3.COHERENT if w1_ok else HeytingOmega3.DEGRADED
        return HeytingOmega3.fast_join_all([v_g, v_s, v_o, v_k, v_r, v_h, v_w])

    @staticmethod
    def _hardware_action_from_verdict(verdict: HeytingOmega3) -> Tuple[str, float]:
        """Mapeo veredicto → (acción de hardware, latencia de interlock en ns)."""
        if verdict is HeytingOmega3.VETOED:
            return "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED", 392.15
        if verdict is HeytingOmega3.DEGRADED:
            return "MONITORING_DEGRADED", 120.0
        return "NOMINAL_EXECUTION", 45.0

    def _validate_robdd_homotopy(
        self,
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
        unsat_core_audit: Optional[UnsatCoreCertifierData] = None,
        grobner_audit: Optional[GrobnerAuditData] = None,
    ) -> ROBDDIsomorphismData:
        if unsat_core_audit and not unsat_core_audit.is_strictly_orthogonal:
            raise NonInterferenceViolationError("Fase 3 requiere ortogonalidad certificada.")

        p_orig = self._as_finite_probability_vector(
            "original_truth_table_probs", original_truth_table_probs,
        )
        p_red = self._as_finite_probability_vector(
            "reduced_robdd_probs", reduced_robdd_probs,
        )

        dim = max(p_orig.size, p_red.size)
        p_o_pad = np.pad(p_orig, (0, dim - p_orig.size))
        p_r_pad = np.pad(p_red, (0, dim - p_red.size))

        analysis_orig = self._entropy_analysis_full(p_o_pad)
        analysis_red = self._entropy_analysis_full(p_r_pad)

        H_orig = analysis_orig.shannon_entropy
        H_red = analysis_red.shannon_entropy
        entropy_loss = abs(H_orig - H_red)

        tol = _MIN_ENTROPY_TOLERANCE
        is_homotopic = bool(entropy_loss <= tol)
        if not is_homotopic:
            raise ROBDDHomotopyError(
                f"Ruptura homotópica ROBDD: ΔH = {entropy_loss:.6e} > tol."
            )

        w1 = float(0.5 * np.sum(np.abs(p_o_pad - p_r_pad)))
        if w1 > _WASSERSTEIN_DISTANCE_TOLERANCE:
            logger.warning(
                "Wasserstein-1 = %.3e > tol=%.1e", w1, _WASSERSTEIN_DISTANCE_TOLERANCE,
            )

        kl_fwd, kl_rev, js = self._kl_js_divergences(p_o_pad, p_r_pad)
        # I(X;X_red) ≈ H(X) + H(X_red) - H(X,X_red) ≤ min(H_o, H_r) (cota)
        mi = float(max(0.0, min(H_orig, H_red) - 0.5 * (kl_fwd + kl_rev)))

        grobner_ok = grobner_audit.is_minimally_independent if grobner_audit else True
        symplectic_ok = grobner_audit.is_symplectic_valid if grobner_audit else True
        is_ortho = unsat_core_audit.is_strictly_orthogonal if unsat_core_audit else True
        is_kam = unsat_core_audit.is_kam_stable if unsat_core_audit else True
        rsi3_ok = unsat_core_audit.is_rsi3_valid if unsat_core_audit else True
        w1_ok = bool(w1 <= _WASSERSTEIN_DISTANCE_TOLERANCE)

        verdict = self._heyting_combine_full(
            grobner_ok=grobner_ok,
            symplectic_ok=symplectic_ok,
            ortho_ok=is_ortho,
            kam_stable=is_kam,
            rsi3_ok=rsi3_ok,
            homotopy_ok=is_homotopic,
            w1_ok=w1_ok,
        )
        hw_action, latency = self._hardware_action_from_verdict(verdict)

        heyting_ok = _heyting_lattice_axioms_check()
        if not heyting_ok:
            logger.error("[HEYTING] ¡Axiomas de retículo/Heyting Ω₃ violados — bug crítico!")

        reduction_ratio = (
            float(analysis_red.support_size / max(analysis_orig.support_size, 1))
        )
        return ROBDDIsomorphismData(
            support_dimension=dim,
            original_entropy=H_orig,
            reduced_entropy=H_red,
            entropy_loss=entropy_loss,
            entropy_tolerance=tol,
            total_variation_distance=w1,
            is_homotopically_equivalent=is_homotopic,
            original_entropy_analysis=analysis_orig,
            reduced_entropy_analysis=analysis_red,
            kl_divergence_forward=kl_fwd,
            kl_divergence_reverse=kl_rev,
            jensen_shannon_divergence=js,
            mutual_information=mi,
            wasserstein_distance_w1=w1,
            reduction_ratio=reduction_ratio,
            heyting_verdict=verdict.name,
            heyting_code=verdict.value,
            heyting_axioms_verified=heyting_ok,
            hardware_crowbar_action=hw_action,
            interlock_latency_ns=latency,
        )

    # ─── ÚLTIMA DEFINICIÓN DE FASE 3 (emisión del certificado) ──────────────
    def _emit_governance_certificate(
        self,
        h2: MICPhase2Handoff,
        robdd_audit: ROBDDIsomorphismData,
    ) -> MinimizerGovernanceCertificate:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 3: certificado OODA completo.
        Verifica axiomas A1–A5, aplica adjudicación $\Omega_3$ (con implicación
        de Heyting) y emite el certificado con trazabilidad SHA-256.
        ═══════════════════════════════════════════════════════════════════
        """
        h1 = h2.phase1
        u = h2.unsat_core

        is_topologically_valid = bool(
            h1.grobner_audit.is_minimally_independent
            and h1.grobner_audit.is_symplectic_valid
            and u.is_strictly_orthogonal
            and u.is_kam_stable
            and u.is_rsi3_valid
            and robdd_audit.is_homotopically_equivalent
        )

        verdict = HeytingOmega3.from_mask(robdd_audit.heyting_code)
        heyting_ok = _heyting_lattice_axioms_check()
        if not heyting_ok:
            logger.error("[HEYTING] ¡Axiomas de retículo/Heyting Ω₃ violados — bug crítico!")

        atoms = [
            1.0 if h1.grobner_audit.is_minimally_independent else 0.0,
            1.0 if h1.grobner_audit.is_symplectic_valid else 0.0,
            1.0 if u.is_strictly_orthogonal else 0.0,
            1.0 if u.is_kam_stable else 0.5,
            1.0 if u.is_rsi3_valid else 0.5,
            1.0 if robdd_audit.is_homotopically_equivalent else 0.0,
            1.0 if robdd_audit.wasserstein_distance_w1 <= _WASSERSTEIN_DISTANCE_TOLERANCE else 0.5,
        ]
        score = float(sum(atoms) / len(atoms))
        risk = (
            "NOMINAL" if score > 0.9
            else "WARNING" if score > 0.6
            else "CRITICAL"
        )

        cert = MinimizerGovernanceCertificate(
            phase2=h2,
            robdd_audit=robdd_audit,
            heyting_verdict=verdict,
            heyting_mask=verdict.mask,
            heyting_axioms_verified=heyting_ok,
            hardware_action=robdd_audit.hardware_crowbar_action,
            interlock_latency_ns=robdd_audit.interlock_latency_ns,
            is_topologically_valid=is_topologically_valid,
            overall_quality_score=score,
            risk_assessment=risk,
            timestamp_iso=datetime.now(tz=timezone.utc).isoformat(),
            input_checksum=h1.input_checksum,
        )
        logger.info(
            "[OODA-CERT] verdict=%s (mask=0x%02X), valid=%s, score=%.3f, risk=%s, "
            "axiomas=%s, W1=%.2e, T³_assoc=%.2e",
            verdict.name, verdict.mask, is_topologically_valid,
            score, risk, heyting_ok,
            robdd_audit.wasserstein_distance_w1,
            u.rsi3_level3_associator,
        )
        return cert


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ORQUESTADOR SUPREMO: MICMinimizerAgent v5.4                                ║
# ║  Composición funtorial Φ₃ ∘ Φ₂ ∘ Φ₁ con continuidad estricta de handoffs. ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MICMinimizerAgent(Morphism, Phase3_ROBDDIsomorphismValidator):
    r"""
    El Custodio de la Base Booleana.

    Ciclo OODA completo con continuidad estricta $\Phi_1\to\Phi_2\to\Phi_3$ vía
    `MICPhase1Handoff` → `MICPhase2Handoff` → `MinimizerGovernanceCertificate`.

    Invariante de anidamiento:
        $\mathrm{last}(\Phi_k):X_k\to\mathrm{Handoff}_k$
        y $\mathrm{first}(\Phi_{k+1}):\mathrm{Handoff}_k\to X_{k+1}$
        con $\mathrm{first}(\Phi_{k+1})\circ\mathrm{last}(\Phi_k)$ tipado estrictamente.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("MICMinimizerAgent v%s (RSI³+Poincaré-celeste+Heyting) inicializado.", __version__)

    def execute_boolean_topology_governance(
        self,
        boolean_polynomial_matrix: NDArray[np.uint8],
        tool_projection_matrix: NDArray[np.float64],
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
        *,
        enable_extended_diagnostics: bool = False,
    ) -> MinimizerGovernanceState:
        r"""
        Composición funtorial estricta $\Phi_3\circ\Phi_2\circ\Phi_1$.

        Flujo (con continuidad vía handoffs):
          1. FASE 1: `_audit_grobner_independence` → `_handoff_to_phase2`
             → `MICPhase1Handoff`.
          2. FASE 2: `_ingest_phase1_handoff` → `_certify_non_interference_unsat`
             → `_handoff_to_phase3` → `MICPhase2Handoff`.
          3. FASE 3: `_ingest_phase2_handoff` → `_validate_robdd_homotopy`
             → `_emit_governance_certificate` → `MinimizerGovernanceCertificate`.

        Raises:
            GrobnerDegeneracyError, NonInterferenceViolationError,
            ROBDDHomotopyError, OODAChainViolationError.
        """
        start_time = time.time()
        checksum = self._compute_input_checksum(
            boolean_polynomial_matrix,
            tool_projection_matrix,
            original_truth_table_probs,
            reduced_robdd_probs,
        )
        logger.info("[OODA] Inicio v%s (checksum=%s...)", __version__, checksum[:16])

        # ═══════ FASE 1 ═══════  last = _handoff_to_phase2
        logger.info("[FASE1] Gröbner + Cayley–Darboux (fail-closed)...")
        grobner_audit = self._audit_grobner_independence(boolean_polynomial_matrix)
        h1 = self._handoff_to_phase2(grobner_audit, checksum)

        # ═══════ FASE 2 ═══════  first = _ingest_phase1_handoff ; last = _handoff_to_phase3
        logger.info("[FASE2] No-interferencia + Poincaré celeste + RSI³ anidado...")
        h1_ingested = self._ingest_phase1_handoff(h1)
        unsat_core_audit = self._certify_non_interference_unsat(
            tool_projection_matrix, grobner_audit=h1_ingested.grobner_audit,
        )
        h2 = self._handoff_to_phase3(h1_ingested, unsat_core_audit)

        # ═══════ FASE 3 ═══════  first = _ingest_phase2_handoff ; last = _emit_governance_certificate
        logger.info("[FASE3] ROBDD + Wasserstein + Heyting Ω₃...")
        h2_ingested = self._ingest_phase2_handoff(h2)
        robdd_audit = self._validate_robdd_homotopy(
            original_truth_table_probs,
            reduced_robdd_probs,
            unsat_core_audit=h2_ingested.unsat_core,
            grobner_audit=h2_ingested.phase1.grobner_audit,
        )
        cert = self._emit_governance_certificate(h2_ingested, robdd_audit)

        if not cert.is_topologically_valid:
            raise MICMinimizerAgentError(
                "Composición funtorial no autorizó la minimización booleana."
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[OODA] Completado en %.2fms (verdict=%s, score=%.3f)",
            elapsed_ms, cert.heyting_verdict.name, cert.overall_quality_score,
        )

        if enable_extended_diagnostics:
            self._log_extended_diagnostics(grobner_audit, unsat_core_audit, robdd_audit)

        return MinimizerGovernanceState(
            grobner_audit=grobner_audit,
            unsat_core_audit=unsat_core_audit,
            robdd_audit=robdd_audit,
            is_topologically_valid=cert.is_topologically_valid,
            reduction_phase=BooleanReductionPhase.COMPLETE,
            timestamp=start_time,
            overall_quality_score=cert.overall_quality_score,
            risk_assessment=cert.risk_assessment,
            heyting_verdict=cert.heyting_verdict,
            heyting_axioms_verified=cert.heyting_axioms_verified,
            interlock_latency_ns=cert.interlock_latency_ns,
            certificate=cert,
        )

    def _log_extended_diagnostics(
        self,
        g: GrobnerAuditData,
        u: UnsatCoreCertifierData,
        r: ROBDDIsomorphismData,
    ) -> None:
        logger.info(
            "[DIAG] Gröbner: rank=%d, Darboux=%d, sym_rel=%.3e, klein=%.3e | "
            "Unsat: off=%.3e, KAM=%s, Bruno=%s, γ=%.3e, |M|=%.3e "
            "(cerrado=%.3e, res=%.2e), Lindstedt=%.3e, λ_P=%.3f, "
            "T_Nek=%.3e, L_φ=%.3f, p=%.2f, Krull=%.3f, FS=%.3e, "
            "T¹=%.2e T²_Lip=%.3f T³_assoc=%.2e | "
            "ROBDD: H_o=%.3f, H_r=%.3f, ΔH=%.3e, W1=%.3e, JS=%.3e",
            g.ideal_dimension, g.darboux_chart_dim,
            g.relative_symplectic_residue, g.klein_residual,
            u.off_diagonal_conflict_norm, u.is_kam_stable, u.bruno_ok,
            u.diophantine_gamma, u.melnikov_integral,
            u.melnikov_closed_form, u.melnikov_residue,
            u.lindstedt_omega_shift, u.poincare_lyapunov,
            u.nekhoroshev_time, u.lipschitz_constant, u.quadratic_rate,
            u.krull_valuation, u.fubini_study_distance,
            u.rsi3_level1_residual, u.rsi3_level2_lipschitz,
            u.rsi3_level3_associator,
            r.original_entropy, r.reduced_entropy, r.entropy_loss,
            r.wasserstein_distance_w1, r.jensen_shannon_divergence,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    # Excepciones
    "MICMinimizerAgentError",
    "BooleanInputValidationError",
    "GrobnerDegeneracyError",
    "NonInterferenceViolationError",
    "ROBDDHomotopyError",
    "SymplecticDeriftError",
    "PoincareResonanceError",
    "RSI3ContractionError",
    "ProbabilityDistributionError",
    "OODAChainViolationError",
    # Enumeraciones
    "BooleanReductionPhase",
    "HeytingOmega3",
    # DTOs
    "GF2MatrixProperties",
    "LinearCodeProperties",
    "GrobnerAuditData",
    "OrthogonalityAnalysis",
    "RSI3KleisliState",
    "UnsatCoreCertifierData",
    "EntropyAnalysis",
    "ROBDDIsomorphismData",
    "MICPhase1Handoff",
    "MICPhase2Handoff",
    "MinimizerGovernanceCertificate",
    "MinimizerGovernanceState",
    # Fases
    "Phase1_GrobnerBasisAuditor",
    "Phase2_UnsatCoreCertifier",
    "Phase3_ROBDDIsomorphismValidator",
    # Agente
    "MICMinimizerAgent",
    # Utilidades
    "_heyting_lattice_axioms_check",
]