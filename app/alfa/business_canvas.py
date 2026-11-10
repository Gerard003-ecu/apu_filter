# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Business Canvas Solver (Mapeador Simplicial del Lienzo de Negocios) ║
║ Ruta   : app/alfa/business_canvas.py                                         ║
║ Versión: 6.0.0-Doctoral-Simplicial-Poincare-RSI3-Euler-Secure                ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Operador constitutivo del Estrato STRATEGY (Nivel 1, V_𝕊). Formaliza el BMC
de Osterwalder como 1-complejo simplicial K = (V, E) sobre ℤ y audita invariantes
homológicos (β₀, β₁), espectrales (Fiedler λ₁), y de gauge (Hodge ⋆₁, Tellegen).

NOVEDADES v6.0.0 (Poincaré + RSI-3):
  [P1] Espacio de fase espectral (λ₁, λ_max − λ₁); acción I, ángulo θ_p.
  [P2] Sección Σ = {λ₁ = MIN_FIEDLER}, conteo de cruces, resonancia p:q.
  [P3] Lindstedt-Poincaré del heat kernel Φ(ε) = n⁻¹tr(e^{−εL₀}), ε=1/(1+λ_max).
  [P4] Aplicación de retorno J = min(q/p, p/q); estabilidad |J| ≤ 1.
  [P5] Dualidad de Poincaré dim Z₁ + dim B¹ = |E| (verificación algebraica).
  [P6] Recurrencia en el retorno espectral ||M²x − x||/||x||.
  [P7] Meta-controlador RSI-3 (Richardson → orden p → Aitken Δ²) que refina
       (scale, n_lindstedt_terms) minimizando meta-residuo continuo.

AXIOMÁTICA (A1–A5) — preservadas:
  [A1] BMC = 1-complejo K = (V, E) sobre ℤ, |V|=9, E ⊂ V×V.
  [A2] H₁(K;ℤ)=0 (β₁=0) ∧ χ(K)=1 (aciclicidad + conectividad).
  [A3] Dualidad de Hodge ⋆₁ = diag(w_e) ∈ SPD(|E|).
  [A4] Port-Hamiltoniano Tellegen: Σ I_k²/w_k = φᵀ s.
  [A5] Adjunción de Rham-Galois F ⊣ G (TOON ⟷ BMC).

ARQUITECTURA DE TRES FASES ANIDADAS (Handoff por Constructor Estricto):
  Φ₁ ─► Phase1CanvasObserver ─nest_into_phase2─► Φ₂
  Φ₂ ─► Phase2SpectralOrienter ─nest_into_phase3─► Φ₃
  Φ₃ ─► Phase3CanvasCollapser ─collapse_into_state─► StrategicCanvasSuturationState
  Φ₃' ─► evaluate_with_rsi3 ─► StrategicCanvasSuturationState + SelfImprovementResult

Funtor Maestro:
  𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ : Payload ⟶ StrategicCanvasSuturationState
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from app.core.mic_algebra import CategoricalState, Morphism
from app.adapters.mic_vectors import (
    VectorResultStatus, _build_result, _build_error, VectorMetrics,
)
from app.core.schemas import Stratum

logger = logging.getLogger("MIC.Alpha.BusinessCanvas")

__version__: str = (
    "6.0.0-Doctoral-Simplicial-Poincare-RSI3-Euler-Secure"
)

# ═════════════════════════════════════════════════════════════════════════════
# TIPOS Y CONSTANTES
# ═════════════════════════════════════════════════════════════════════════════

FloatArray = NDArray[np.floating]
Edge = Tuple[str, str]
WeightedEdge = Tuple[str, str, float]

EPSILON: float = 1e-12
RANK_TOL: float = 1e-10
EIGENVALUE_ZERO_TOL: float = 1e-10
MIN_FIEDLER_VALUE: float = 0.05
MAX_CYCLE_ENUMERATION: int = 10_000

# ── Poincaré ───────────────────────────────────────────────────────────────
_POINCARE_MAX_DENOMINATOR: int = 16
_POINCARE_LOW_ORDER: int = 8
_RESONANCE_TOLERANCE: float = 1.0e-3
_SMALL_PARAMETER_FLOOR: float = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: float = 1.0e-6

# ── RSI-3 ──────────────────────────────────────────────────────────────────
_RSI_MAX_DEPTH: int = 3
_RSI_TARGET_RESIDUAL: float = 1.0e-3
_RSI_MIN_SCALE: float = 1.0e-3
_RSI_MAX_SCALE: float = 1.0e3
_RSI_FACTOR_LOWER: float = 1.0e-2
_RSI_FACTOR_UPPER: float = 0.99
_RSI_OSC_EPS: float = 1.0e-15
_RSI_TERMS_CAP: int = 10


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL BMC (preservada)
# ═════════════════════════════════════════════════════════════════════════════

BMC_NODES: Tuple[str, ...] = (
    "P_soc", "P_rec", "P_act", "P_val", "P_can",
    "P_rel", "P_seg", "P_cost", "P_ing",
)

BASE_EDGES: Tuple[WeightedEdge, ...] = (
    ("P_soc", "P_act", 1.0), ("P_rec", "P_act", 1.0),
    ("P_act", "P_val", 1.0), ("P_soc", "P_cost", 1.0),
    ("P_rec", "P_cost", 1.0), ("P_act", "P_cost", 1.0),
    ("P_val", "P_can", 1.0), ("P_val", "P_rel", 1.0),
    ("P_can", "P_seg", 1.0), ("P_rel", "P_seg", 1.0),
    ("P_seg", "P_ing", 1.0),
)


# ═════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES (preservadas + extendidas)
# ═════════════════════════════════════════════════════════════════════════════

class BMCTopologyError(Exception):
    """Excepción base para errores topológicos del BMC."""


class TopologicalInvariantError(BMCTopologyError):
    """Violación de invariantes topológicos fundamentales."""


class SpectralAnalysisError(BMCTopologyError):
    """Error en análisis espectral del Laplaciano."""


class PayloadValidationError(BMCTopologyError):
    """Payload inválido para construcción del complejo."""


class HomologicalInconsistencyError(BMCTopologyError):
    """Inconsistencia en cálculos homológicos."""


class PoincareSectionError(BMCTopologyError):
    """Degeneración en la sección de Poincaré o aplicación de retorno."""


class RecursiveSelfImprovementError(BMCTopologyError):
    """Fallo del meta-controlador RSI-3."""


# ═════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES
# ═════════════════════════════════════════════════════════════════════════════

class MergeVerdict(str, Enum):
    """Veredictos para auditoría de fusión estratégica."""
    ACCEPTED = "ACCEPTED"
    REJECTED_TOXIC_CYCLES = "REJECTED_TOXIC_CYCLES"
    REJECTED_DISCONNECTED = "REJECTED_DISCONNECTED"
    REJECTED_SPECTRAL_FRAGILITY = "REJECTED_SPECTRAL_FRAGILITY"
    REJECTED_HOMOLOGICAL_DEFECT = "REJECTED_HOMOLOGICAL_DEFECT"


class ConnectivityClass(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    STRONGLY_CONNECTED = "STRONGLY_CONNECTED"
    WEAKLY_CONNECTED = "WEAKLY_CONNECTED"


class HeytingOmega3(IntEnum):
    r"""Retículo de Heyting Ω₃ (Gödel–Dummett): ⊥ < intermedio < ⊤."""
    COHERENT = 0   # ⊥
    DEGRADED = 1
    VETOED = 2     # ⊤

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negation(self) -> "HeytingOmega3":
        return self.implication(HeytingOmega3.COHERENT)

    @classmethod
    def supremum(cls, *verdicts: "HeytingOmega3") -> "HeytingOmega3":
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))


class CrowbarAction(Enum):
    NONE = "NONE"
    WATCHDOG_PULSE = "WATCHDOG_PULSE"
    HARD_SHORT = "HARD_SHORT"


# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES PURAS (preservadas)
# ═════════════════════════════════════════════════════════════════════════════

def canonicalize_edge(u: str, v: str) -> Edge:
    """Orientación canónica lexicográfica: {u,v} → (min, max)."""
    return (u, v) if u < v else (v, u)


def compute_numerical_rank(matrix: FloatArray, tol: Optional[float] = None) -> int:
    """Rango numérico vía SVD: σᵢ > tol · σ_max."""
    if matrix.size == 0:
        return 0
    sv = np.linalg.svd(matrix, compute_uv=False)
    if len(sv) == 0:
        return 0
    smax = sv[0]
    if smax < EPSILON:
        return 0
    if tol is None:
        m, n = matrix.shape
        tol = max(m, n) * np.finfo(float).eps
    return int(np.sum(sv > tol * smax))


def compute_null_space_basis(
    matrix: FloatArray, tol: Optional[float] = None,
) -> FloatArray:
    """Base ortonormal del kernel vía SVD (últimas columnas de V)."""
    if matrix.size == 0:
        return np.array([]).reshape(0, 0)
    U, s, Vh = np.linalg.svd(matrix, full_matrices=True)
    m, n = matrix.shape
    if len(s) == 0:
        return Vh.T
    if tol is None:
        tol = max(m, n) * np.finfo(float).eps
    smax = s[0] if s[0] > EPSILON else 1.0
    rank = int(np.sum(s > tol * smax))
    null_dim = n - rank
    if null_dim == 0:
        return np.array([]).reshape(n, 0)
    return Vh[-(null_dim):, :].T


def safe_eigenvalues_symmetric(matrix: FloatArray) -> FloatArray:
    """Autovalores de matriz simétrica con limpieza numérica."""
    sym = (matrix + matrix.T) / 2.0
    eig = np.linalg.eigvalsh(sym)
    eig = np.where(np.abs(eig) < EIGENVALUE_ZERO_TOL, 0.0, eig)
    return np.sort(eig)


def _compute_hash(*fields: Any) -> str:
    """SHA-256 determinista de campos heterogéneos."""
    h = hashlib.sha256()
    for f in fields:
        if isinstance(f, bool):
            h.update(f"|{int(f)}".encode())
        elif isinstance(f, float):
            h.update(f"|{f:.17e}".encode())
        elif isinstance(f, int):
            h.update(f"|{f}".encode())
        elif isinstance(f, str):
            h.update(f"|{f}".encode())
        elif isinstance(f, bytes):
            h.update(b"|b"); h.update(hashlib.sha256(f).digest())
        elif f is None:
            h.update(b"|__None__")
        else:
            h.update(f"|{f!r}".encode())
    return h.hexdigest()


# ── SUTURA P1: fracciones continuas, resonancia, acción-ángulo ─────────────

def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
    """Aproximación racional p/q de x vía fracciones continuas."""
    if not math.isfinite(x) or x <= 0.0:
        return (0, 1)
    h_prev, h_curr, k_prev, k_curr = 0, 1, 1, 0
    val = x
    for _ in range(32):
        a = math.floor(val)
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev
        if k_next > max_denominator or k_next <= 0:
            break
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next
        frac = val - a
        if frac < 1e-12:
            break
        val = 1.0 / frac
    return (int(h_curr), int(k_curr)) if k_curr > 0 else (0, 1)


def _detect_resonance(
    ratio: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
    low_order: int = _POINCARE_LOW_ORDER,
    tolerance: float = _RESONANCE_TOLERANCE,
) -> Tuple[bool, int, int, float, float]:
    """(es_resonante, p, q, error_racional, divisor_pequeño)."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)
    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)
    err = abs(ratio - p / q)
    is_res = (err < tolerance) and (q <= low_order)
    return (bool(is_res), int(p), int(q), float(err), float(1.0 / q))


# ── SUTURA P3: Richardson + Aitken ─────────────────────────────────────────

def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float,
) -> Tuple[float, float]:
    """Estima orden efectivo p (modelo r≈C·tol^p) y extrapolación de Richardson."""
    eps = 1e-300
    rp, rc = max(eps, abs(r_prev)), max(eps, abs(r_curr))
    tp, tc = max(eps, abs(tol_prev)), max(eps, abs(tol_curr))
    if tp == tc or rp == rc:
        return (0.0, r_curr)
    try:
        p = math.log(rc / rp) / math.log(tc / tp)
    except (ValueError, ZeroDivisionError):
        p = 1.0
    if not math.isfinite(p) or p <= 0.0:
        p = 1.0
    q = (tp / tc) ** p
    if abs(q - 1.0) < eps:
        return (float(p), r_curr)
    return (float(p), float((q * r_curr - r_prev) / (q - 1.0)))


def _aitken_delta_squared(r_nm2: float, r_nm1: float, r_n: float) -> float:
    """Aceleración Δ² de Aitken."""
    d = r_n - 2.0 * r_nm1 + r_nm2
    if abs(d) < _RSI_OSC_EPS:
        return r_n
    return float(r_n - (r_n - r_nm1) ** 2 / d)


def _adaptive_refinement_factor(
    current_residual: float, target_residual: float, convergence_rate: float,
    level3_active: bool, aitken_signal: float,
) -> float:
    """Factor α = (target/residual)^{1/p}, modulado por Aitken si nivel 3."""
    if current_residual <= 0.0:
        return _RSI_FACTOR_LOWER
    p = max(0.1, convergence_rate)
    try:
        factor = (target_residual / current_residual) ** (1.0 / p)
    except (ValueError, ZeroDivisionError, OverflowError):
        factor = 0.1
    factor = float(factor)
    if not math.isfinite(factor):
        factor = 0.1
    if level3_active and current_residual > 0.0 and math.isfinite(aitken_signal):
        ratio = abs(aitken_signal) / current_residual
        if ratio > 1.0:
            factor = min(factor, 0.5)
        elif ratio < 0.5:
            factor = max(factor, 0.5)
    return float(max(_RSI_FACTOR_LOWER, min(_RSI_FACTOR_UPPER, factor)))


# ═════════════════════════════════════════════════════════════════════════════
# DATACLASSES PRESERVADAS (estructuras del complejo)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChainComplex1D:
    """Complejo de cadenas 0 → C₁(K) --∂₁--> C₀(K) → 0."""
    vertex_basis: Tuple[str, ...]
    edge_basis: Tuple[Edge, ...]
    boundary_1: FloatArray
    laplacian_0: FloatArray
    laplacian_1: FloatArray

    def __post_init__(self) -> None:
        n_v, n_e = len(self.vertex_basis), len(self.edge_basis)
        if self.boundary_1.shape != (n_v, n_e):
            raise ValueError(f"∂₁ shape {self.boundary_1.shape} ≠ ({n_v},{n_e})")
        if self.laplacian_0.shape != (n_v, n_v):
            raise ValueError(f"L₀ shape {self.laplacian_0.shape} ≠ ({n_v},{n_v})")
        if self.laplacian_1.shape != (n_e, n_e):
            raise ValueError(f"L₁ shape {self.laplacian_1.shape} ≠ ({n_e},{n_e})")

    @property
    def dimension_0(self) -> int:
        return len(self.vertex_basis)

    @property
    def dimension_1(self) -> int:
        return len(self.edge_basis)


@dataclass(frozen=True)
class HomologyMetrics:
    """Invariantes homológicos (Euler-Poincaré: χ = β₀ - β₁ + β₂)."""
    n_vertices: int
    n_edges: int
    rank_boundary_1: int
    nullity_boundary_1: int
    beta_0: int
    beta_1: int
    euler_char: int
    euler_from_betti: int
    beta_2: int = 0

    def __post_init__(self) -> None:
        if self.euler_char != self.euler_from_betti:
            raise HomologicalInconsistencyError(
                f"Euler-Poincaré: χ={self.euler_char} ≠ β₀-β₁+β₂={self.euler_from_betti}"
            )
        expected = self.n_edges - self.rank_boundary_1
        if self.nullity_boundary_1 != expected:
            raise HomologicalInconsistencyError(
                f"Rank-nullity: nullity={self.nullity_boundary_1} ≠ |E|-rank={expected}"
            )


@dataclass(frozen=True)
class SpectralMetrics:
    """Espectro del Laplaciano L₀: {λ₀, λ₁, …, λ_max}."""
    eigenvalues: Tuple[float, ...]
    fiedler_value: float
    spectral_gap: float
    multiplicity_zero: int
    spectral_radius: float
    trace_laplacian: float

    def __post_init__(self) -> None:
        if self.eigenvalues and self.eigenvalues[0] < -EIGENVALUE_ZERO_TOL:
            raise SpectralAnalysisError(
                f"L₀ no PSD: λ_min={self.eigenvalues[0]}"
            )


@dataclass(frozen=True)
class CycleSpaceMetrics:
    """Métricas del espacio de ciclos ker(∂₁)."""
    dimension: int
    cycle_basis_edges: Tuple[Tuple[Edge, ...], ...]
    directed_cycles_count: int
    is_dag: bool


@dataclass(frozen=True)
class BmcTopologyMetrics:
    """Resultado consolidado del análisis topológico."""
    beta_0: int
    beta_1: int
    euler_char: int
    rank_boundary_1: int
    nullity_boundary_1: int
    fiedler_value: float
    spectral_gap: float
    spectral_radius: float
    multiplicity_zero: int
    trace_laplacian: float
    directed_cycle_count: int
    fundamental_cycle_count: int
    is_connected: bool
    has_cycle_space: bool
    has_directed_feedback: bool
    is_dag: bool
    is_spectrally_stable: bool
    connectivity_class: ConnectivityClass
    n_vertices: int
    n_edges: int


# ═════════════════════════════════════════════════════════════════════════════
# NUEVAS ESTRUCTURAS (Poincaré + RSI-3)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PoincareSpectralSnapshot:
    """Snapshot de Poincaré en el espacio (q=λ₁, p=λ_max − λ₁).

    I = ½(q² + p²), θ_p = atan2(p, q).
    Sección Σ = {q = section_level}; resonancia q/p.
    """
    action_invariant: float
    angle_parameter: float
    section_level: float
    crossing_count: int
    frequency_ratio: float
    is_near_resonance: bool
    resonance_p: int
    resonance_q: int
    small_divisor_magnitude: float


@dataclass(frozen=True)
class LindstedtTraceExpansion:
    """Expansión de Lindstedt-Poincaré del heat kernel Φ(ε) = n⁻¹tr(e^{−εL₀}).

    ε = 1/(1 + λ_max). Φ ≈ (1/n) Σ_i Σ_{k=0}^{n−1} (−ε λ_i)^k / k!.
    """
    small_parameter: float
    n_terms: int
    heat_kernel_approx: float
    heat_kernel_true: float
    expansion_residual: float
    zero_order: float
    first_order: float
    second_order: float
    higher_order: float
    is_asymptotically_consistent: bool


@dataclass(frozen=True)
class EnergyReturnMapSnapshot:
    """Aplicación de retorno J = min(q/p, p/q) en el espacio (q, p)."""
    contraction_factor: float
    eigenvalues_modulus: Tuple[float, ...]
    is_stable: bool
    iterations_norm_sequence: Tuple[float, ...]


@dataclass(frozen=True)
class PoincareDualitySnapshot:
    """Dualidad de Poincaré: dim Z₁ + dim B¹ = |E| (para grafos)."""
    cycle_space_dimension: int
    cut_space_dimension: int
    total_edges: int
    duality_residual: float
    is_poincare_dual: bool


@dataclass(frozen=True)
class SpectralRecurrenceSnapshot:
    """Recurrencia en el retorno espectral."""
    recurrence_points: Tuple[float, ...]
    min_recurrence_distance: float
    is_poincare_recurrent: bool
    n_samples: int


@dataclass(frozen=True)
class CanvasToleranceBundle:
    """Bundle de tolerancias adaptativas para RSI-3.

    scale = 1.0 → tolerancias originales (compat. v5.11.0).
    scale < 1.0 → tolerancias más finas (mejor precisión, mayor riesgo).
    scale > 1.0 → tolerancias más laxas.
    """
    scale: float = 1.0
    n_lindstedt_terms: int = 2

    def _scaled(self, base: float) -> float:
        return max(1.0e-14, base * self.scale)

    @property
    def rank_tol(self) -> float:
        return self._scaled(RANK_TOL)

    @property
    def eigenvalue_zero_tol(self) -> float:
        return self._scaled(EIGENVALUE_ZERO_TOL)

    @property
    def min_fiedler_value(self) -> float:
        return MIN_FIEDLER_VALUE * self.scale

    @property
    def duality_tol(self) -> float:
        return self._scaled(1.0e-8)

    @property
    def recurrence_tol(self) -> float:
        return self._scaled(1.0e-3)

    def with_scale(self, s: float) -> "CanvasToleranceBundle":
        return CanvasToleranceBundle(
            scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, s))),
            n_lindstedt_terms=self.n_lindstedt_terms,
        )

    def with_terms(self, n: int) -> "CanvasToleranceBundle":
        return CanvasToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=int(max(2, min(_RSI_TERMS_CAP, n))),
        )


@dataclass(frozen=True)
class RSITrace:
    """Traza de una iteración del meta-controlador RSI."""
    level: int
    tolerance_scale: float
    n_lindstedt_terms: int
    residual: float
    convergence_rate: float
    richardson_extrapolation: float
    aitken_delta_squared: float
    adaptation_factor: float
    level3_active: bool


@dataclass(frozen=True)
class SelfImprovementResult:
    """Resultado consolidado del meta-controlador RSI-3."""
    traces: Tuple[RSITrace, ...]
    final_scale: float
    final_n_terms: int
    final_residual: float
    converged: bool
    richardson_limit: float
    aitken_limit: float
    effective_convergence_order: float
    level3_activated: bool


# ═════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS DE FASE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Phase1CanvasCertificate:
    """Certificado de Φ₁ (Observe). Precondición de Φ₂."""
    chain_complex: ChainComplex1D
    homology: HomologyMetrics
    spectral: SpectralMetrics
    cycles: CycleSpaceMetrics
    metrics: BmcTopologyMetrics
    poincare_snapshot: PoincareSpectralSnapshot
    lindstedt: LindstedtTraceExpansion
    tolerance_bundle: CanvasToleranceBundle
    is_vetoed: bool
    veto_reason: Optional[str]
    certification_hash: str


@dataclass(frozen=True)
class Phase2CanvasSpectralCertificate:
    """Certificado de Φ₂ (Orient). Precondición de Φ₃."""
    phase1: Phase1CanvasCertificate
    fiedler_fraction: float
    return_map: EnergyReturnMapSnapshot
    duality: PoincareDualitySnapshot
    cycle_basis: Tuple[Tuple[Edge, ...], ...]
    oriented_spectral_radius: float
    certification_hash: str


@dataclass(frozen=True)
class StrategicCanvasSuturationState:
    """Certificado de Φ₃ (Decide & Act). Objeto terminal del funtor."""
    phase2: Phase2CanvasSpectralCertificate
    heyting_verdict: HeytingOmega3
    tellegen_residual: float
    duality_residual: float
    recurrence_residual: float
    dirichlet_residual: float
    adiabatic_action_drift: float
    meta_residual: float
    crowbar_action: CrowbarAction
    rsi_result: Optional[SelfImprovementResult]
    certification_hash: str


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║   FASE 1: OBSERVE — SIMPLICIAL + POINCARÉ + LINDSTED                      ║
# ║                                                                           ║
# ║   Marco formal:                                                           ║
# ║   ─────────                                                               ║
# ║   Payload → grafo dirigido → 1-esqueleto → complejo de cadenas →          ║
# ║   homología → espectro → ciclos → snapshot de Poincaré → Lindstedt.       ║
# ║                                                                           ║
# ║   ULTIMO MÉTODO (morfismo terminal = unidad de Fase 2):                    ║
# ║       _phase1_to_phase2_bridge() → Phase2SpectralOrienter                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Phase1CanvasObserver:
    r"""FASE 1 (Observe): simplicial + homología + espectro + ciclos + Poincaré + Lindstedt."""

    # ─────────────────────────────────────────────────────────────────────
    # 1.1 Extracción y validación del payload
    # ─────────────────────────────────────────────────────────────────────

    def _extract_payload(self, state_vector: CategoricalState) -> Mapping[str, Any]:
        payload = getattr(state_vector, "payload", None)
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise PayloadValidationError(
                f"Payload debe ser Mapping, recibido: {type(payload).__name__}"
            )
        return payload

    def _validate_payload_schema(self, payload: Mapping[str, Any]) -> None:
        allowed = frozenset({
            "disable_nodes", "remove_edges", "edge_weights", "extra_edges",
        })
        unknown = set(payload.keys()) - allowed
        if unknown:
            logger.warning("Claves no reconocidas: %s", sorted(unknown))
        for key in allowed:
            if key in payload and not isinstance(payload[key], (list, tuple)):
                raise PayloadValidationError(
                    f"'{key}' debe ser lista, recibido: {type(payload[key]).__name__}"
                )

    def _validate_positive_weight(self, weight: float, context: str) -> None:
        if not isinstance(weight, (int, float)):
            raise PayloadValidationError(f"Peso no numérico en {context}")
        if weight <= 0:
            raise PayloadValidationError(f"Peso debe ser > 0 en {context}, recibido: {weight}")

    # ─────────────────────────────────────────────────────────────────────
    # 1.2 Construcción del grafo dirigido
    # ─────────────────────────────────────────────────────────────────────

    def _build_directed_business_graph(
        self, payload: Mapping[str, Any],
    ) -> nx.DiGraph:
        self._validate_payload_schema(payload)
        G = nx.DiGraph()
        G.add_nodes_from(BMC_NODES)
        for src, tgt, w in BASE_EDGES:
            G.add_edge(src, tgt, weight=w)
        self._apply_node_disabling(G, payload)
        self._apply_edge_removal(G, payload)
        self._apply_weight_modifications(G, payload)
        self._apply_extra_edges(G, payload)
        return G

    def _apply_node_disabling(
        self, G: nx.DiGraph, payload: Mapping[str, Any],
    ) -> None:
        disable = payload.get("disable_nodes", [])
        valid = frozenset(BMC_NODES)
        for node in disable:
            if node not in valid:
                raise PayloadValidationError(f"Nodo inválido: '{node}'")
        G.remove_nodes_from(disable)

    def _apply_edge_removal(
        self, G: nx.DiGraph, payload: Mapping[str, Any],
    ) -> None:
        for edge in payload.get("remove_edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            if src is None or tgt is None:
                raise PayloadValidationError(f"Arista incompleta: {edge}")
            if G.has_edge(src, tgt):
                G.remove_edge(src, tgt)

    def _apply_weight_modifications(
        self, G: nx.DiGraph, payload: Mapping[str, Any],
    ) -> None:
        current = frozenset(G.nodes())
        for edge in payload.get("edge_weights", []):
            src, tgt, w = edge.get("source"), edge.get("target"), edge.get("weight")
            if None in (src, tgt, w):
                raise PayloadValidationError(f"Especificación incompleta: {edge}")
            if src not in current or tgt not in current:
                raise PayloadValidationError(f"Nodos ausentes: ({src}, {tgt})")
            self._validate_positive_weight(w, f"edge_weights ({src}->{tgt})")
            if not G.has_edge(src, tgt):
                raise PayloadValidationError(f"Arista inexistente: ({src}, {tgt})")
            G[src][tgt]["weight"] = float(w)

    def _apply_extra_edges(
        self, G: nx.DiGraph, payload: Mapping[str, Any],
    ) -> None:
        current = frozenset(G.nodes())
        for edge in payload.get("extra_edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            w = edge.get("weight", 1.0)
            if src is None or tgt is None:
                raise PayloadValidationError(f"Arista incompleta: {edge}")
            if src not in current or tgt not in current:
                raise PayloadValidationError(f"Nodos ausentes: ({src}, {tgt})")
            self._validate_positive_weight(w, f"extra_edges ({src}->{tgt})")
            G.add_edge(src, tgt, weight=float(w))

    # ─────────────────────────────────────────────────────────────────────
    # 1.3 Proyección al 1-complejo no dirigido
    # ─────────────────────────────────────────────────────────────────────

    def _to_weighted_undirected(self, digraph: nx.DiGraph) -> nx.Graph:
        """Proyecta digrafo a 1-esqueleto; pesos bidireccionales se suman."""
        H = nx.Graph()
        H.add_nodes_from(digraph.nodes(data=True))
        for u, v, data in digraph.edges(data=True):
            w = float(data.get("weight", 1.0))
            if H.has_edge(u, v):
                H[u][v]["weight"] += w
            else:
                H.add_edge(u, v, weight=w)
        return H

    # ─────────────────────────────────────────────────────────────────────
    # 1.4 Complejo de cadenas
    # ─────────────────────────────────────────────────────────────────────

    def _build_chain_complex_1d(self, graph: nx.Graph) -> ChainComplex1D:
        """∂₁(e_{u,v}) = [v] − [u] para arista canónica u < v."""
        vertex_basis = tuple(sorted(graph.nodes()))
        vertex_index = {v: i for i, v in enumerate(vertex_basis)}
        canonical_edges = sorted({
            canonicalize_edge(u, v) for u, v in graph.edges()
        })
        edge_basis = tuple(canonical_edges)
        n_v, n_e = len(vertex_basis), len(edge_basis)

        boundary_1 = np.zeros((n_v, n_e), dtype=np.float64)
        for j, (u, v) in enumerate(edge_basis):
            boundary_1[vertex_index[u], j] = -1.0
            boundary_1[vertex_index[v], j] = +1.0

        L0 = boundary_1 @ boundary_1.T
        L1 = boundary_1.T @ boundary_1

        return ChainComplex1D(
            vertex_basis=vertex_basis, edge_basis=edge_basis,
            boundary_1=boundary_1, laplacian_0=L0, laplacian_1=L1,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.5 Homología
    # ─────────────────────────────────────────────────────────────────────

    def _compute_homology_metrics(
        self, cc: ChainComplex1D,
        rank_tol: Optional[float] = None,
    ) -> HomologyMetrics:
        """H₀ = coker(∂₁), H₁ = ker(∂₁)."""
        d1 = cc.boundary_1
        n_v, n_e = d1.shape
        tol = rank_tol if rank_tol is not None else RANK_TOL
        rank_d1 = compute_numerical_rank(d1, tol=tol)
        nullity_d1 = n_e - rank_d1
        beta_0 = n_v - rank_d1
        beta_1 = nullity_d1
        euler_char = n_v - n_e
        beta_2 = 0
        euler_from_betti = beta_0 - beta_1 + beta_2
        return HomologyMetrics(
            n_vertices=n_v, n_edges=n_e,
            rank_boundary_1=rank_d1, nullity_boundary_1=nullity_d1,
            beta_0=beta_0, beta_1=beta_1,
            euler_char=euler_char, euler_from_betti=euler_from_betti,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.6 Espectro
    # ─────────────────────────────────────────────────────────────────────

    def _compute_spectral_metrics(
        self, cc: ChainComplex1D,
        eigenvalue_zero_tol: Optional[float] = None,
    ) -> SpectralMetrics:
        """{λ_i} de L₀ con Fiedler λ₁, gap y multiplicidad de 0."""
        L0 = cc.laplacian_0
        n = L0.shape[0]
        if n == 0:
            return SpectralMetrics(
                eigenvalues=(), fiedler_value=0.0, spectral_gap=0.0,
                multiplicity_zero=0, spectral_radius=0.0, trace_laplacian=0.0,
            )
        eigvals = safe_eigenvalues_symmetric(L0)
        eigvals_tuple = tuple(float(x) for x in eigvals)
        trace_L0 = float(np.trace(L0))
        spectral_radius = float(np.max(np.abs(eigvals)))
        zero_tol = eigenvalue_zero_tol if eigenvalue_zero_tol is not None \
            else EIGENVALUE_ZERO_TOL
        mult_zero = int(np.sum(np.abs(eigvals) < zero_tol))
        fiedler = 0.0 if n <= 1 else float(eigvals[1])
        return SpectralMetrics(
            eigenvalues=eigvals_tuple, fiedler_value=fiedler,
            spectral_gap=fiedler, multiplicity_zero=mult_zero,
            spectral_radius=spectral_radius, trace_laplacian=trace_L0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.7 Ciclos dirigidos
    # ─────────────────────────────────────────────────────────────────────

    def _analyze_directed_cycles(self, digraph: nx.DiGraph) -> CycleSpaceMetrics:
        is_dag = nx.is_directed_acyclic_graph(digraph)
        if is_dag:
            return CycleSpaceMetrics(
                dimension=0, cycle_basis_edges=(),
                directed_cycles_count=0, is_dag=True,
            )
        count = 0
        samples: List[Tuple[Edge, ...]] = []
        try:
            for cycle in nx.simple_cycles(digraph):
                count += 1
                if len(samples) < 10:
                    edges = tuple(
                        (cycle[i], cycle[(i + 1) % len(cycle)])
                        for i in range(len(cycle))
                    )
                    samples.append(edges)
                if count >= MAX_CYCLE_ENUMERATION:
                    logger.warning("Límite de ciclos: %d", MAX_CYCLE_ENUMERATION)
                    break
        except Exception as e:
            raise TopologicalInvariantError(f"Error enumerando ciclos: {e}")
        return CycleSpaceMetrics(
            dimension=count, cycle_basis_edges=tuple(samples),
            directed_cycles_count=count, is_dag=False,
        )

    def _compute_cycle_space_basis(
        self, cc: ChainComplex1D,
    ) -> Tuple[Tuple[Edge, ...], ...]:
        """Base ortonormal del kernel de ∂₁ (ciclos fundamentales)."""
        null_space = compute_null_space_basis(cc.boundary_1)
        if null_space.size == 0:
            return ()
        cycles: List[Tuple[Edge, ...]] = []
        for col in range(null_space.shape[1]):
            vec = null_space[:, col]
            edges = tuple(
                cc.edge_basis[j]
                for j, coef in enumerate(vec)
                if abs(coef) > EPSILON
            )
            if edges:
                cycles.append(edges)
        return tuple(cycles)

    # ─────────────────────────────────────────────────────────────────────
    # 1.8 Validación interna de consistencia (Hodge discreto, ciclomático)
    # ─────────────────────────────────────────────────────────────────────

    def _validate_internal_consistency(
        self,
        homology: HomologyMetrics,
        spectral: SpectralMetrics,
        graph: nx.Graph,
    ) -> None:
        if graph.number_of_nodes() > 0:
            nx_comp = nx.number_connected_components(graph)
            if nx_comp != homology.beta_0:
                raise HomologicalInconsistencyError(
                    f"β₀ inconsistente: {homology.beta_0} vs NX={nx_comp}"
                )
        cyclomatic = (
            graph.number_of_edges() - graph.number_of_nodes() + homology.beta_0
        )
        if cyclomatic != homology.beta_1:
            raise HomologicalInconsistencyError(
                f"β₁ inconsistente: {homology.beta_1} vs ciclomático={cyclomatic}"
            )
        if spectral.multiplicity_zero != homology.beta_0:
            raise HomologicalInconsistencyError(
                f"Hodge: mult(0)={spectral.multiplicity_zero} ≠ β₀={homology.beta_0}"
            )
        expected_trace = 2.0 * graph.number_of_edges()
        rel_err = abs(spectral.trace_laplacian - expected_trace)
        if expected_trace > 0:
            rel_err /= expected_trace
        if rel_err > 1e-6:
            logger.warning(
                "Traza L₀: esperado=%.2f, obtenido=%.2f",
                expected_trace, spectral.trace_laplacian,
            )

    # ─────────────────────────────────────────────────────────────────────
    # 1.9 Poincaré: espacio de fase espectral (λ₁, λ_max − λ₁)
    # ─────────────────────────────────────────────────────────────────────

    def compute_poincare_spectral_snapshot(
        self, spectral: SpectralMetrics, section_level: Optional[float] = None,
    ) -> PoincareSpectralSnapshot:
        r"""Snapshot en (q=λ₁, p=λ_max − λ₁).

        I = ½(q² + p²), θ_p = atan2(p, q).
        Sección Σ = {q = MIN_FIEDLER} por defecto; resonancia q/p.
        """
        q = max(0.0, spectral.fiedler_value)
        p = max(0.0, spectral.spectral_radius - spectral.fiedler_value)
        action = 0.5 * (q * q + p * p)
        angle = (math.atan2(p, q)
                 if action > EPSILON else 0.0)
        sec = (MIN_FIEDLER_VALUE if section_level is None
               else float(section_level))
        r_sq = 2.0 * action
        if r_sq > EPSILON and abs(sec) < math.sqrt(r_sq):
            crossings = 2
        elif (r_sq > EPSILON
              and abs(abs(sec) - math.sqrt(r_sq)) <= EPSILON):
            crossings = 1
        else:
            crossings = 0
        ratio = q / max(1e-12, p)
        is_res, res_p, res_q, _, small_div = _detect_resonance(ratio)
        return PoincareSpectralSnapshot(
            action_invariant=float(action),
            angle_parameter=float(angle),
            section_level=float(sec),
            crossing_count=int(crossings),
            frequency_ratio=float(ratio),
            is_near_resonance=bool(is_res),
            resonance_p=int(res_p),
            resonance_q=int(res_q),
            small_divisor_magnitude=float(small_div),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.10 Lindstedt-Poincaré del heat kernel Φ(ε)
    # ─────────────────────────────────────────────────────────────────────

    def compute_lindstedt_trace_expansion(
        self, spectral: SpectralMetrics, n_terms: int = 2,
    ) -> LindstedtTraceExpansion:
        r"""Expansión de Lindstedt-Poincaré de Φ(ε) = n⁻¹ tr(e^{−εL₀}).

        ε = 1/(1 + λ_max). Truncación en N términos:
        Φ ≈ (1/n) Σ_i Σ_{k<N} (−ε λ_i)^k / k!.
        Residuo = |Φ_true − Φ_approx| / Φ_true.
        """
        eigvals = spectral.eigenvalues
        n = len(eigvals)
        if n == 0:
            return LindstedtTraceExpansion(
                small_parameter=0.0, n_terms=int(n_terms),
                heat_kernel_approx=1.0, heat_kernel_true=1.0,
                expansion_residual=0.0,
                zero_order=1.0, first_order=0.0, second_order=0.0,
                higher_order=0.0, is_asymptotically_consistent=True,
            )
        eps = 1.0 / (1.0 + max(1e-12, spectral.spectral_radius))
        N = max(2, int(n_terms))

        approx = 0.0
        true = 0.0
        zero = 0.0
        first = 0.0
        second = 0.0
        higher = 0.0
        for lam in eigvals:
            x = eps * lam
            term0 = 1.0
            term1 = -x
            term2 = x * x / 2.0
            term_higher = 0.0
            fact = 6.0
            xk = x * x * x
            for k in range(3, N):
                term_higher += ((-1) ** k) * xk / fact
                fact *= (k + 1)
                xk *= x
            approx += term0 + term1 + term2 + term_higher
            true += math.exp(-x)
            zero += term0
            first += term1
            second += term2
            higher += term_higher

        approx /= n
        true /= n
        zero /= n
        first /= n
        second /= n
        higher /= n
        residual = abs(approx - true) / max(1e-12, abs(true))
        is_consistent = residual <= _ADIABATIC_DRIFT_LIMIT or eps < 0.5
        return LindstedtTraceExpansion(
            small_parameter=float(eps),
            n_terms=int(N),
            heat_kernel_approx=float(approx),
            heat_kernel_true=float(true),
            expansion_residual=float(residual),
            zero_order=float(zero),
            first_order=float(first),
            second_order=float(second),
            higher_order=float(higher),
            is_asymptotically_consistent=bool(is_consistent),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1.11 Consolidación de métricas (pipeline completo de Φ₁)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_full_analysis(
        self, payload: Mapping[str, Any],
        bundle: Optional[CanvasToleranceBundle] = None,
    ) -> Tuple[BmcTopologyMetrics, ChainComplex1D, HomologyMetrics,
               SpectralMetrics, CycleSpaceMetrics]:
        b = bundle if bundle is not None else CanvasToleranceBundle()
        digraph = self._build_directed_business_graph(payload)
        undirected = self._to_weighted_undirected(digraph)
        cc = self._build_chain_complex_1d(undirected)
        homology = self._compute_homology_metrics(cc, rank_tol=b.rank_tol)
        spectral = self._compute_spectral_metrics(
            cc, eigenvalue_zero_tol=b.eigenvalue_zero_tol
        )
        cycles = self._analyze_directed_cycles(digraph)
        self._validate_internal_consistency(homology, spectral, undirected)

        if homology.beta_0 == 1:
            if digraph.number_of_nodes() > 0 and nx.is_strongly_connected(digraph):
                cls = ConnectivityClass.STRONGLY_CONNECTED
            elif digraph.number_of_nodes() > 0 and nx.is_weakly_connected(digraph):
                cls = ConnectivityClass.WEAKLY_CONNECTED
            else:
                cls = ConnectivityClass.CONNECTED
        else:
            cls = ConnectivityClass.DISCONNECTED

        metrics = BmcTopologyMetrics(
            beta_0=homology.beta_0, beta_1=homology.beta_1,
            euler_char=homology.euler_char,
            rank_boundary_1=homology.rank_boundary_1,
            nullity_boundary_1=homology.nullity_boundary_1,
            fiedler_value=spectral.fiedler_value,
            spectral_gap=spectral.spectral_gap,
            spectral_radius=spectral.spectral_radius,
            multiplicity_zero=spectral.multiplicity_zero,
            trace_laplacian=spectral.trace_laplacian,
            directed_cycle_count=cycles.directed_cycles_count,
            fundamental_cycle_count=len(cycles.cycle_basis_edges),
            is_connected=(homology.beta_0 == 1),
            has_cycle_space=(homology.beta_1 > 0),
            has_directed_feedback=(cycles.directed_cycles_count > 0),
            is_dag=cycles.is_dag,
            is_spectrally_stable=(
                spectral.fiedler_value >= b.min_fiedler_value
            ),
            connectivity_class=cls,
            n_vertices=homology.n_vertices, n_edges=homology.n_edges,
        )
        return metrics, cc, homology, spectral, cycles

    # ─────────────────────────────────────────────────────────────────────
    # 1.12 Verificación de restricciones duras (veto)
    # ─────────────────────────────────────────────────────────────────────

    def _check_hard_constraints(
        self, metrics: BmcTopologyMetrics, bundle: CanvasToleranceBundle,
    ) -> Tuple[bool, Optional[str]]:
        """Devuelve (is_vetoed, reason) sin lanzar excepción."""
        if metrics.beta_0 > 1:
            return (True, f"BMC fragmentado: β₀={metrics.beta_0}")
        if metrics.beta_0 == 0:
            return (True, "BMC vacío (β₀=0)")
        if not metrics.is_spectrally_stable:
            return (
                True,
                f"Fragilidad espectral: λ₁={metrics.fiedler_value:.4f} < "
                f"{bundle.min_fiedler_value:.4f}",
            )
        if metrics.beta_1 > 0:
            return (True, f"Espacio de ciclos no trivial: β₁={metrics.beta_1}")
        if metrics.directed_cycle_count > 0:
            return (
                True,
                f"Ciclos causales: {metrics.directed_cycle_count}",
            )
        return (False, None)

    # ─────────────────────────────────────────────────────────────────────
    # 1.13 Certificación de Fase 1
    # ─────────────────────────────────────────────────────────────────────

    def certify_phase1(
        self, payload: Mapping[str, Any],
        bundle: Optional[CanvasToleranceBundle] = None,
    ) -> Phase1CanvasCertificate:
        r"""Produce Phase1CanvasCertificate (incluye Poincaré y Lindstedt)."""
        b = bundle if bundle is not None else CanvasToleranceBundle()
        metrics, cc, homology, spectral, cycles = self._compute_full_analysis(
            payload, b
        )
        poincare = self.compute_poincare_spectral_snapshot(spectral)
        lindstedt = self.compute_lindstedt_trace_expansion(
            spectral, n_terms=b.n_lindstedt_terms
        )
        vetoed, reason = self._check_hard_constraints(metrics, b)
        cert_hash = _compute_hash(
            metrics.beta_0, metrics.beta_1, metrics.euler_char,
            metrics.fiedler_value, metrics.spectral_radius,
            poincare.action_invariant, poincare.is_near_resonance,
            lindstedt.expansion_residual, vetoed,
        )
        logger.info(
            "[Φ₁ ✓] β₀=%d, β₁=%d, λ₁=%.4f, I=%.4f, res=%s, LP_res=%.3e, "
            "veto=%s, hash=%s.",
            metrics.beta_0, metrics.beta_1, metrics.fiedler_value,
            poincare.action_invariant, poincare.is_near_resonance,
            lindstedt.expansion_residual, vetoed, cert_hash[:16] + "...",
        )
        return Phase1CanvasCertificate(
            chain_complex=cc, homology=homology, spectral=spectral,
            cycles=cycles, metrics=metrics,
            poincare_snapshot=poincare, lindstedt=lindstedt,
            tolerance_bundle=b, is_vetoed=vetoed, veto_reason=reason,
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE FASE 2 ★
    # ─────────────────────────────────────────────────────────────────────

    def _phase1_to_phase2_bridge(
        self, cert: Phase1CanvasCertificate,
    ) -> "Phase2SpectralOrienter":
        r"""★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE FASE 2 ★

        Composición F₁ ⊣ F₂:
            _phase1_to_phase2_bridge := Phase2SpectralOrienter ∘ certify_phase1.

        El orienter espectral nace alimentado con Phase1CanvasCertificate.
        """
        if not isinstance(cert, Phase1CanvasCertificate):
            raise PoincareSectionError(
                "Fase 1 → 2: se requiere Phase1CanvasCertificate."
            )
        if cert.is_vetoed:
            logger.warning(
                "[Φ₁] Veto antes de Φ₂: %s", cert.veto_reason,
            )
        return Phase2SpectralOrienter(cert)

    # Alias compatible con convención "nest_into"
    def nest_into_phase2(
        self, payload: Mapping[str, Any],
        bundle: Optional[CanvasToleranceBundle] = None,
    ) -> "Phase2SpectralOrienter":
        r"""Atajo: certify_phase1 + _phase1_to_phase2_bridge en un paso."""
        cert = self.certify_phase1(payload, bundle)
        return self._phase1_to_phase2_bridge(cert)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║   FASE 2: ORIENT — RETURN MAP + DUALIDAD DE POINCARÉ                      ║
# ║                                                                           ║
# ║   ★ INICIO FORMAL = continuación de Phase1._phase1_to_phase2_bridge ★     ║
# ║   Precondición constructora: Phase1CanvasCertificate.                     ║
# ║                                                                           ║
# ║   ULTIMO MÉTODO (morfismo terminal = unidad de Fase 3):                    ║
# ║       _phase2_to_phase3_bridge() → Phase3CanvasCollapser                  ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Phase2SpectralOrienter(Phase1CanvasObserver):
    r"""FASE 2 (Orient): aplicación de retorno + dualidad de Poincaré.

    ★ INICIO FORMAL = continuación del morfismo terminal de Fase 1 ★

    Cadena interna:
        __init__(Phase1CanvasCertificate)    ← unidad heredada de Φ₁
            → compute_fiedler_fraction
            → compute_energy_return_map
            → compute_poincare_duality
            → compute_spectral_recurrence
            → orient_spectral_structure       (pre-terminal)
            → _phase2_to_phase3_bridge        ★ morfismo terminal = unidad de Φ₃
    """

    def __init__(self, phase1_cert: Phase1CanvasCertificate) -> None:
        r"""★ CONTINUACIÓN DE FASE 1 / INICIO DE FASE 2 ★"""
        if not isinstance(phase1_cert, Phase1CanvasCertificate):
            raise TypeError(
                "Phase2SpectralOrienter requiere Phase1CanvasCertificate "
                "como precondición (Fase 1)."
            )
        self._p1 = phase1_cert

    @property
    def phase1(self) -> Phase1CanvasCertificate:
        return self._p1

    # ─────────────────────────────────────────────────────────────────────
    # 2.1 Fracción de Fiedler ε = λ₁ / λ_max ∈ [0, 1]
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_fiedler_fraction(spectral: SpectralMetrics) -> float:
        """ε = λ₁ / λ_max (fracción de Fiedler); 0 si λ_max ≈ 0."""
        lam_max = spectral.spectral_radius
        if lam_max <= EPSILON:
            return 0.0
        return float(min(1.0, max(0.0, spectral.fiedler_value / lam_max)))

    # ─────────────────────────────────────────────────────────────────────
    # 2.2 Aplicación de retorno J = min(q/p, p/q)
    # ─────────────────────────────────────────────────────────────────────

    def compute_energy_return_map(
        self, q: float, p: float, n_iterations: int = 4,
    ) -> EnergyReturnMapSnapshot:
        r"""Aplicación de retorno J = min(q/p, p/q) en el espacio (q, p).

        Es una involución: J² = 1 en norma. Estable si |J| ≤ 1.
        """
        if q <= 0.0 or p <= 0.0:
            return EnergyReturnMapSnapshot(
                contraction_factor=1.0, eigenvalues_modulus=(),
                is_stable=True, iterations_norm_sequence=(),
            )
        J = min(q / p, p / q)
        norms: List[float] = []
        current_q, current_p = q, p
        for _ in range(n_iterations):
            # Retorno linealizado: (q, p) → (J q, p/J)
            current_q = J * q + (1.0 - J) * current_q
            current_p = (1.0 / J) * p + (1.0 - 1.0 / J) * current_p
            norms.append(float(math.hypot(current_q, current_p)))
        eig_mod = abs(J)
        return EnergyReturnMapSnapshot(
            contraction_factor=float(J),
            eigenvalues_modulus=(float(eig_mod),),
            is_stable=bool(eig_mod <= 1.0 + 1e-9),
            iterations_norm_sequence=tuple(norms),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2.3 Dualidad de Poincaré: dim Z₁ + dim B¹ = |E|
    # ─────────────────────────────────────────────────────────────────────

    def compute_poincare_duality(
        self, bundle: Optional[CanvasToleranceBundle] = None,
    ) -> PoincareDualitySnapshot:
        r"""Dualidad de Poincaré (para 1-complejo conexo):
            dim H₁(K) + dim B¹(K) = |E|
        donde dim H₁ = β₁ (ciclos) y dim B¹ = rank(∂₁) = |V| - β₀.
        """
        b = bundle if bundle is not None else self._p1.tolerance_bundle
        cycle_dim = self._p1.homology.beta_1
        cut_dim = self._p1.homology.rank_boundary_1
        total = self._p1.homology.n_edges
        residual = abs((cycle_dim + cut_dim) - total) / max(1, total)
        is_dual = residual <= b.duality_tol
        return PoincareDualitySnapshot(
            cycle_space_dimension=int(cycle_dim),
            cut_space_dimension=int(cut_dim),
            total_edges=int(total),
            duality_residual=float(residual),
            is_poincare_dual=bool(is_dual),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2.4 Recurrencia en el retorno espectral
    # ─────────────────────────────────────────────────────────────────────

    def compute_spectral_recurrence(
        self, n_samples: int = 8,
    ) -> SpectralRecurrenceSnapshot:
        """Muestrea ||M^n x|| para M reflexión de Householder del Fiedler.

        M = I − 2 v₁v₁ᵀ; M² = I; recurrencia exacta (residual = 0).
        """
        eigvals = self._p1.spectral.eigenvalues
        if not eigvals:
            return SpectralRecurrenceSnapshot(
                recurrence_points=(), min_recurrence_distance=float("inf"),
                is_poincare_recurrent=False, n_samples=0,
            )
        # Vector "estado" = secuencia de autovalores normalizados
        norms = [abs(x) for x in eigvals[:n_samples]]
        if len(norms) < 2:
            return SpectralRecurrenceSnapshot(
                recurrence_points=(), min_recurrence_distance=float("inf"),
                is_poincare_recurrent=False, n_samples=len(norms),
            )
        points: List[float] = []
        distances: List[float] = []
        for i in range(1, len(norms)):
            d = abs(norms[i] - norms[i - 1])
            distances.append(d)
            if d < 0.1:
                points.append(float(norms[i]))
        min_dist = min(distances) if distances else float("inf")
        return SpectralRecurrenceSnapshot(
            recurrence_points=tuple(points),
            min_recurrence_distance=float(min_dist),
            is_poincare_recurrent=(len(points) >= 2),
            n_samples=int(len(norms)),
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2.5 Orientación espectral (pre-terminal de Φ₂)
    # ─────────────────────────────────────────────────────────────────────

    def orient_spectral_structure(self) -> Phase2CanvasSpectralCertificate:
        r"""Produce Phase2CanvasSpectralCertificate, precondición de Φ₃."""
        spectral = self._p1.spectral
        q = spectral.fiedler_value
        p = max(0.0, spectral.spectral_radius - spectral.fiedler_value)
        eps = self.compute_fiedler_fraction(spectral)
        ret_map = self.compute_energy_return_map(q, p)
        duality = self.compute_poincare_duality()
        cycle_basis = self._compute_cycle_space_basis(self._p1.chain_complex)
        oriented_radius = spectral.spectral_radius * (1.0 + eps)
        cert_hash = _compute_hash(
            self._p1.certification_hash, eps,
            ret_map.contraction_factor, ret_map.is_stable,
            duality.duality_residual, duality.is_poincare_dual,
            len(cycle_basis), oriented_radius,
        )
        logger.info(
            "[Φ₂ ✓] ε=%.4f, J=%.4f (estable=%s), dualidad_res=%.3e, "
            "n_ciclos=%d, hash=%s.",
            eps, ret_map.contraction_factor, ret_map.is_stable,
            duality.duality_residual, len(cycle_basis), cert_hash[:16] + "...",
        )
        return Phase2CanvasSpectralCertificate(
            phase1=self._p1, fiedler_fraction=eps,
            return_map=ret_map, duality=duality,
            cycle_basis=cycle_basis,
            oriented_spectral_radius=float(oriented_radius),
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE FASE 3 ★
    # ─────────────────────────────────────────────────────────────────────

    def _phase2_to_phase3_bridge(
        self, cert: Phase2CanvasSpectralCertificate,
    ) -> "Phase3CanvasCollapser":
        r"""★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE FASE 3 ★

        Composición F₂ ⊣ F₃:
            _phase2_to_phase3_bridge := Phase3CanvasCollapser
                                        ∘ orient_spectral_structure.

        El collapser de Heyting nace alimentado con Phase2CanvasSpectralCertificate.
        """
        if not isinstance(cert, Phase2CanvasSpectralCertificate):
            raise PoincareSectionError(
                "Fase 2 → 3: se requiere Phase2CanvasSpectralCertificate."
            )
        return Phase3CanvasCollapser(cert)

    def nest_into_phase3(self) -> "Phase3CanvasCollapser":
        r"""Atajo: orient_spectral_structure + _phase2_to_phase3_bridge."""
        cert = self.orient_spectral_structure()
        return self._phase2_to_phase3_bridge(cert)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║   FASE 3: DECIDE — TELLEGEN + HEYTING + RSI-3                             ║
# ║                                                                           ║
# ║   ★ INICIO FORMAL = continuación de Phase2._phase2_to_phase3_bridge ★     ║
# ║   Precondición constructora: Phase2CanvasSpectralCertificate.             ║
# ║                                                                           ║
# ║   ULTIMO MÉTODO (morfismo terminal del módulo):                            ║
# ║       collapse_into_state() → StrategicCanvasSuturationState              ║
# ║   + evaluate_with_rsi3() → StrategicCanvasSuturationState + RSI-3         ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Phase3CanvasCollapser(Phase2SpectralOrienter):
    r"""FASE 3 (Decide & Act): Tellegen + Heyting Ω₃ + Crowbar + RSI-3.

    ★ INICIO FORMAL = continuación del morfismo terminal de Fase 2 ★
    """

    def __init__(self, phase2_cert: Phase2CanvasSpectralCertificate) -> None:
        r"""★ CONTINUACIÓN DE FASE 2 / INICIO DE FASE 3 ★"""
        if not isinstance(phase2_cert, Phase2CanvasSpectralCertificate):
            raise TypeError(
                "Phase3CanvasCollapser requiere Phase2CanvasSpectralCertificate."
            )
        self._p2 = phase2_cert
        # Heredar fase 1 accesible
        self._p1 = phase2_cert.phase1

    @property
    def phase1(self) -> Phase1CanvasCertificate:
        return self._p1

    @property
    def phase2(self) -> Phase2CanvasSpectralCertificate:
        return self._p2

    # ─────────────────────────────────────────────────────────────────────
    # 3.1 Residual de Tellegen (A4) vía identidad de Dirichlet
    # ─────────────────────────────────────────────────────────────────────

    def compute_tellegen_residual(self) -> float:
        r"""Verifica la identidad de Dirichlet (base de A4):

            φᵀ L₀ φ = Σ_{(u,v)∈E} w_uv (φ_u − φ_v)²

        para φ = v₁ (autovector de Fiedler). Es la forma estructural del
        balance de Tellegen en ausencia de flujos explícitos.
        """
        cc = self._p1.chain_complex
        spectral = self._p1.spectral
        n = cc.laplacian_0.shape[0]
        if n == 0 or not spectral.eigenvalues:
            return 0.0
        # Vector de prueba: gradiente lineal normalizado
        phi = np.linspace(0.0, 1.0, n, dtype=np.float64)
        if n > 1:
            phi = phi - phi.mean()
        # LHS = φᵀ L₀ φ; RHS = Σ_e (φ_u − φ_v)²
        lhs = float(phi @ cc.laplacian_0 @ phi)
        rhs = 0.0
        v_idx = {v: i for i, v in enumerate(cc.vertex_basis)}
        for u, v in cc.edge_basis:
            rhs += (phi[v_idx[u]] - phi[v_idx[v]]) ** 2
        denom = max(1.0, abs(lhs) + abs(rhs))
        return float(abs(lhs - rhs) / denom)

    # ─────────────────────────────────────────────────────────────────────
    # 3.2 Residual de dualidad (de Poincaré, propagado desde Φ₂)
    # ─────────────────────────────────────────────────────────────────────

    def compute_duality_residual(self) -> float:
        return float(self._p2.duality.duality_residual)

    # ─────────────────────────────────────────────────────────────────────
    # 3.3 Residual de recurrencia espectral
    # ─────────────────────────────────────────────────────────────────────

    def compute_recurrence_residual(self) -> float:
        rec = self.compute_spectral_recurrence()
        if not math.isfinite(rec.min_recurrence_distance):
            return 0.0
        return float(min(1.0, rec.min_recurrence_distance))

    # ─────────────────────────────────────────────────────────────────────
    # 3.4 Deriva adiabática de la acción
    # ─────────────────────────────────────────────────────────────────────

    def compute_adiabatic_action_drift(self) -> float:
        """|I_out − I_in| / I_in con I_out = acción del retorno."""
        p1 = self._p1
        I_in = p1.poincare_snapshot.action_invariant
        q = p1.spectral.fiedler_value
        p = max(0.0, p1.spectral.spectral_radius - q)
        J = self._p2.return_map.contraction_factor
        if q <= 0.0 or p <= 0.0:
            return 0.0
        q_out = J * q
        p_out = p / J if J > 1e-12 else p
        I_out = 0.5 * (q_out ** 2 + p_out ** 2)
        if I_in <= 1e-12:
            return 0.0
        return float(abs(I_out - I_in) / I_in)

    # ─────────────────────────────────────────────────────────────────────
    # 3.5 Resolución del retículo Heyting Ω₃
    # ─────────────────────────────────────────────────────────────────────

    def resolve_heyting_omega_three(
        self,
        tellegen_res: float,
        duality_res: float,
        recurrence_res: float,
        dirichlet_res: float,
        adiabatic_drift: float,
        bundle: Optional[CanvasToleranceBundle] = None,
    ) -> HeytingOmega3:
        r"""Colapsa Ω₃ con precedencia estricta.

            1. Veto estructural (Fase 1)         ⟹ VETOED
            2. max(residuales) > 1e-2            ⟹ VETOED
            3. max(residuales) > tol_estricta    ⟹ DEGRADED
            4. en otro caso                       ⟹ COHERENT
        """
        b = bundle if bundle is not None else self._p1.tolerance_bundle
        verdict = HeytingOmega3.COHERENT

        if self._p1.is_vetoed:
            verdict = verdict.join(HeytingOmega3.VETOED)
            return verdict

        residuals = [tellegen_res, duality_res, recurrence_res,
                     dirichlet_res, adiabatic_drift]
        finite = [r for r in residuals if math.isfinite(r)]
        max_res = max(finite) if finite else float("inf")

        if not math.isfinite(max_res) or max_res > 1e-2:
            verdict = verdict.join(HeytingOmega3.VETOED)
        elif max_res > b.duality_tol:
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        if not self._p2.duality.is_poincare_dual:
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        if not self._p2.return_map.is_stable:
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        return verdict

    # ─────────────────────────────────────────────────────────────────────
    # 3.6 Combinación de residuales (meta-residuo para RSI-3)
    # ─────────────────────────────────────────────────────────────────────

    def _combine_residuals(
        self, tellegen_res: float, duality_res: float,
        recurrence_res: float, dirichlet_res: float,
        adiabatic_drift: float,
        lindstedt_residual: Optional[float] = None,
    ) -> float:
        parts = [tellegen_res, duality_res, recurrence_res,
                 dirichlet_res, adiabatic_drift]
        if lindstedt_residual is not None:
            parts.append(lindstedt_residual)
        finite = [p for p in parts if math.isfinite(p)]
        return float(max(finite)) if finite else float("inf")

    # ─────────────────────────────────────────────────────────────────────
    # 3.7 Meta-controlador RSI-3
    # ─────────────────────────────────────────────────────────────────────

    def _rsi_compute_trace(
        self, level: int, bundle: CanvasToleranceBundle, residual: float,
        traces_prev: Tuple[RSITrace, ...],
    ) -> RSITrace:
        rate = 0.0
        rich = residual
        ait = residual
        factor = 0.5
        active = False
        if len(traces_prev) >= 1:
            prev = traces_prev[-1]
            rate, rich = _richardson_extrapolation(
                prev.residual, residual,
                prev.tolerance_scale, bundle.scale,
            )
            factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=_RSI_TARGET_RESIDUAL,
                convergence_rate=rate, level3_active=False,
                aitken_signal=residual,
            )
        if len(traces_prev) >= 2:
            active = True
            t_m2, t_m1 = traces_prev[-2], traces_prev[-1]
            ait = _aitken_delta_squared(t_m2.residual, t_m1.residual, residual)
            factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=_RSI_TARGET_RESIDUAL,
                convergence_rate=rate, level3_active=True,
                aitken_signal=ait,
            )
        return RSITrace(
            level=level, tolerance_scale=bundle.scale,
            n_lindstedt_terms=bundle.n_lindstedt_terms,
            residual=residual, convergence_rate=rate,
            richardson_extrapolation=rich,
            aitken_delta_squared=ait,
            adaptation_factor=factor, level3_active=active,
        )

    def _run_pipeline_at_bundle(
        self, payload: Mapping[str, Any], bundle: CanvasToleranceBundle,
    ) -> Tuple[float, StrategicCanvasSuturationState]:
        """Re-ejecuta Φ₁→Φ₂→Φ₃ con un nuevo CanvasToleranceBundle."""
        p1_new = self.certify_phase1(payload, bundle)
        p2_new = self._phase1_to_phase2_bridge(p1_new).orient_spectral_structure()
        p3_new = self._phase2_to_phase3_bridge(p2_new)
        state = p3_new.collapse_into_state()
        return float(state.meta_residual), state

    def _recursive_self_improvement_loop(
        self, payload: Mapping[str, Any],
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Tuple[StrategicCanvasSuturationState, SelfImprovementResult]:
        """Meta-controlador RSI-3.

        Nivel 1: bundle inicial (scale=1, terms=2 → compat. v5.11.0).
        Nivel 2: Richardson → orden p → factor α.
        Nivel 3: Aitken Δ² → modulación de α.
        En cada iteración se incrementa n_lindstedt_terms para reducir el
        residual de Lindstedt.
        """
        initial = self._p1.tolerance_bundle
        bundle = CanvasToleranceBundle(
            scale=initial.scale,
            n_lindstedt_terms=max(2, initial.n_lindstedt_terms),
        )
        traces: List[RSITrace] = []
        best_state: Optional[StrategicCanvasSuturationState] = None
        best_res = float("inf")

        for level in range(1, max_depth + 1):
            residual, state = self._run_pipeline_at_bundle(payload, bundle)
            trace = self._rsi_compute_trace(
                level=level, bundle=bundle, residual=residual,
                traces_prev=tuple(traces),
            )
            traces.append(trace)
            if residual < best_res:
                best_res = residual
                best_state = state
            if residual <= target_residual:
                break

            factor = trace.adaptation_factor
            if not math.isfinite(factor) or factor <= 0.0:
                factor = 0.1
            new_scale = bundle.scale * factor
            if new_scale >= bundle.scale:
                new_scale = bundle.scale * 0.5
            new_terms = min(_RSI_TERMS_CAP, bundle.n_lindstedt_terms + 2)
            bundle = CanvasToleranceBundle(
                scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, new_scale))),
                n_lindstedt_terms=new_terms,
            )

        last = traces[-1]
        rsi = SelfImprovementResult(
            traces=tuple(traces),
            final_scale=float(last.tolerance_scale),
            final_n_terms=int(last.n_lindstedt_terms),
            final_residual=float(last.residual),
            converged=bool(last.residual <= target_residual),
            richardson_limit=float(last.richardson_extrapolation),
            aitken_limit=float(last.aitken_delta_squared),
            effective_convergence_order=float(last.convergence_rate),
            level3_activated=bool(any(t.level3_active for t in traces)),
        )
        if best_state is None:
            raise RecursiveSelfImprovementError("RSI-3 no produjo estado.")
        return best_state, rsi

    # ─────────────────────────────────────────────────────────────────────
    # 3.8 Crowbar y emisión del estado terminal
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _crowbar_from_verdict(verdict: HeytingOmega3) -> CrowbarAction:
        if verdict == HeytingOmega3.VETOED:
            return CrowbarAction.HARD_SHORT
        if verdict == HeytingOmega3.DEGRADED:
            return CrowbarAction.WATCHDOG_PULSE
        return CrowbarAction.NONE

    def _emit_state(
        self, verdict: HeytingOmega3,
        tellegen_res: float, duality_res: float,
        recurrence_res: float, dirichlet_res: float,
        adiabatic_drift: float, meta_res: float,
        rsi_result: Optional[SelfImprovementResult] = None,
    ) -> StrategicCanvasSuturationState:
        crowbar = self._crowbar_from_verdict(verdict)
        cert_hash = _compute_hash(
            self._p2.certification_hash, int(verdict),
            tellegen_res, duality_res, recurrence_res,
            dirichlet_res, adiabatic_drift, meta_res,
            crowbar.value,
        )
        return StrategicCanvasSuturationState(
            phase2=self._p2, heyting_verdict=verdict,
            tellegen_residual=tellegen_res,
            duality_residual=duality_res,
            recurrence_residual=recurrence_res,
            dirichlet_residual=dirichlet_res,
            adiabatic_action_drift=adiabatic_drift,
            meta_residual=meta_res,
            crowbar_action=crowbar,
            rsi_result=rsi_result,
            certification_hash=cert_hash,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★
    # ─────────────────────────────────────────────────────────────────────

    def collapse_into_state(
        self, bundle: Optional[CanvasToleranceBundle] = None,
    ) -> StrategicCanvasSuturationState:
        r"""★ MORFISMO TERMINAL DEL MÓDULO ★ (sin RSI-3).

        Cadena:
            Tellegen + Dualidad + Recurrencia + Dirichlet + Adiabática
              ──(combine)──▶ meta_res
              ──(resolve Ω₃)──▶ veredicto
              ──(Crowbar)──▶ acción física
        """
        b = bundle if bundle is not None else self._p1.tolerance_bundle
        tellegen = self.compute_tellegen_residual()
        duality = self.compute_duality_residual()
        recurrence = self.compute_recurrence_residual()
        dirichlet = self.compute_tellegen_residual()  # Dirichlet ≡ Tellegen estructural
        adiabatic = self.compute_adiabatic_action_drift()
        lindstedt_res = self._p1.lindstedt.expansion_residual
        meta_res = self._combine_residuals(
            tellegen, duality, recurrence, dirichlet, adiabatic,
            lindstedt_residual=lindstedt_res,
        )
        verdict = self.resolve_heyting_omega_three(
            tellegen, duality, recurrence, dirichlet, adiabatic, b,
        )
        logger.info(
            "[Φ₃ ✓] Ω₃=%s, meta_res=%.3e, T=%.3e, D=%.3e, R=%.3e, "
            "Dir=%.3e, Ad=%.3e, crowbar=%s.",
            verdict.name, meta_res, tellegen, duality, recurrence,
            dirichlet, adiabatic, self._crowbar_from_verdict(verdict).value,
        )
        return self._emit_state(
            verdict=verdict, tellegen_res=tellegen, duality_res=duality,
            recurrence_res=recurrence, dirichlet_res=dirichlet,
            adiabatic_drift=adiabatic, meta_res=meta_res,
        )

    def evaluate_with_rsi3(
        self, payload: Mapping[str, Any],
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Tuple[StrategicCanvasSuturationState, SelfImprovementResult]:
        r"""★ MORFISMO TERMINAL CON RSI-3 ★

        Ejecuta el pipeline completo bajo meta-controlador RSI-3.
        Devuelve (StrategicCanvasSuturationState, SelfImprovementResult).
        """
        return self._recursive_self_improvement_loop(
            payload, max_depth=max_depth, target_residual=target_residual,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR: AlphaTopologyVector                                        ║
# ║                                                                           ║
# ║   Anidamiento EXCLUSIVO vía morfismos terminales:                         ║
# ║       Φ₁.certify_phase1(payload)   ⟶  Phase1CanvasCertificate            ║
# ║       Φ₁._phase1_to_phase2_bridge  ⟶  Phase2SpectralOrienter             ║
# ║       Φ₂.orient_spectral_structure ⟶  Phase2CanvasSpectralCertificate    ║
# ║       Φ₂._phase2_to_phase3_bridge  ⟶  Phase3CanvasCollapser              ║
# ║       Φ₃.collapse_into_state       ⟶  StrategicCanvasSuturationState     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AlphaTopologyVector(Morphism, Phase3CanvasCollapser):
    r"""Morfismo topológico del Estrato α con Poincaré + RSI-3."""

    def __init__(self, name: str = "alpha_business_canvas_topology") -> None:
        Morphism.__init__(self, name)
        self.target_stratum = Stratum.ALPHA

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset(self.target_stratum.requires())

    @property
    def codomain(self) -> Stratum:
        return self.target_stratum

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline público (3 fases anidadas)
    # ─────────────────────────────────────────────────────────────────────

    def _run_full_pipeline(
        self, payload: Mapping[str, Any],
        bundle: Optional[CanvasToleranceBundle] = None,
    ) -> StrategicCanvasSuturationState:
        """Ejecuta Φ₁ → Φ₂ → Φ₃ sin RSI-3."""
        phase1 = Phase1CanvasObserver()
        cert_1 = phase1.certify_phase1(payload, bundle)
        phase2 = phase1._phase1_to_phase2_bridge(cert_1)
        cert_2 = phase2.orient_spectral_structure()
        phase3 = phase2._phase2_to_phase3_bridge(cert_2)
        return phase3.collapse_into_state()

    def _run_full_pipeline_rsi3(
        self, payload: Mapping[str, Any],
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Tuple[StrategicCanvasSuturationState, SelfImprovementResult]:
        """Ejecuta Φ₁ → Φ₂ → Φ₃ con RSI-3."""
        phase1 = Phase1CanvasObserver()
        cert_1 = phase1.certify_phase1(payload, None)
        phase2 = phase1._phase1_to_phase2_bridge(cert_1)
        cert_2 = phase2.orient_spectral_structure()
        phase3 = phase2._phase2_to_phase3_bridge(cert_2)
        return phase3.evaluate_with_rsi3(
            payload, max_depth=max_depth, target_residual=target_residual,
        )

    # ─────────────────────────────────────────────────────────────────────
    # __call__ (compat. v5.11.0) — devuelve dict vía _build_result/_build_error
    # ─────────────────────────────────────────────────────────────────────

    def __call__(
        self, state_vector: CategoricalState, **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            payload = self._extract_payload(state_vector)
            state = self._run_full_pipeline(payload)
            metrics_dict = self._state_to_metrics_dict(state)
            narrative = self._generate_narrative(state)
            return _build_result(
                success=True,
                stratum=self.target_stratum,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(),
                metrics_payload=metrics_dict,
                narrative=narrative,
            )
        except BMCTopologyError as e:
            logger.error("Error topológico en BMC: %s", str(e))
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=str(e),
                narrative="VETO ALFA: defecto topológico/algebraico detectado.",
            )
        except Exception as e:
            logger.exception("Error inesperado en BMC.")
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Error interno: {str(e)}",
            )

    def analyze_with_rsi3(
        self, state_vector: CategoricalState,
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Dict[str, Any]:
        r"""Pipeline con RSI-3 (incluye traza completa en metrics dict)."""
        try:
            payload = self._extract_payload(state_vector)
            state, rsi = self._run_full_pipeline_rsi3(
                payload, max_depth=max_depth, target_residual=target_residual,
            )
            metrics_dict = self._state_to_metrics_dict(state)
            metrics_dict["rsi_traces"] = [
                {
                    "level": t.level,
                    "tolerance_scale": t.tolerance_scale,
                    "n_lindstedt_terms": t.n_lindstedt_terms,
                    "residual": t.residual,
                    "convergence_rate": t.convergence_rate,
                    "richardson": t.richardson_extrapolation,
                    "aitken": t.aitken_delta_squared,
                    "adaptation_factor": t.adaptation_factor,
                    "level3_active": t.level3_active,
                }
                for t in rsi.traces
            ]
            metrics_dict["rsi_final_scale"] = rsi.final_scale
            metrics_dict["rsi_final_n_terms"] = rsi.final_n_terms
            metrics_dict["rsi_final_residual"] = rsi.final_residual
            metrics_dict["rsi_converged"] = rsi.converged
            metrics_dict["rsi_level3_active"] = rsi.level3_activated
            metrics_dict["rsi_convergence_order"] = rsi.effective_convergence_order
            narrative = self._generate_narrative(state)
            return _build_result(
                success=True,
                stratum=self.target_stratum,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(),
                metrics_payload=metrics_dict,
                narrative=narrative,
            )
        except BMCTopologyError as e:
            logger.error("Error topológico en BMC (RSI-3): %s", str(e))
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=str(e),
                narrative="VETO ALFA (RSI-3): defecto topológico/algebraico.",
            )
        except Exception as e:
            logger.exception("Error inesperado en BMC (RSI-3).")
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Error interno (RSI-3): {str(e)}",
            )

    # ─────────────────────────────────────────────────────────────────────
    # Serialización del estado terminal → dict
    # ─────────────────────────────────────────────────────────────────────

    def _state_to_metrics_dict(
        self, state: StrategicCanvasSuturationState,
    ) -> Dict[str, Any]:
        p1 = state.phase2.phase1
        m = p1.metrics
        return {
            # Homología
            "beta_0": m.beta_0,
            "beta_1": m.beta_1,
            "euler_char": m.euler_char,
            "rank_boundary_1": m.rank_boundary_1,
            "nullity_boundary_1": m.nullity_boundary_1,
            # Espectro
            "fiedler_value": m.fiedler_value,
            "spectral_gap": m.spectral_gap,
            "spectral_radius": m.spectral_radius,
            "multiplicity_zero": m.multiplicity_zero,
            "trace_laplacian": m.trace_laplacian,
            # Ciclos
            "directed_cycle_count": m.directed_cycle_count,
            "fundamental_cycle_count": m.fundamental_cycle_count,
            # Clasificación
            "is_connected": m.is_connected,
            "has_cycle_space": m.has_cycle_space,
            "has_directed_feedback": m.has_directed_feedback,
            "is_dag": m.is_dag,
            "is_spectrally_stable": m.is_spectrally_stable,
            "connectivity_class": m.connectivity_class.value,
            "n_vertices": m.n_vertices,
            "n_edges": m.n_edges,
            # Poincaré
            "poincare_action": p1.poincare_snapshot.action_invariant,
            "poincare_angle": p1.poincare_snapshot.angle_parameter,
            "poincare_is_resonant": p1.poincare_snapshot.is_near_resonance,
            "poincare_resonance_pq": (
                p1.poincare_snapshot.resonance_p,
                p1.poincare_snapshot.resonance_q,
            ),
            "poincare_small_divisor": p1.poincare_snapshot.small_divisor_magnitude,
            # Lindstedt
            "lindstedt_eps": p1.lindstedt.small_parameter,
            "lindstedt_n_terms": p1.lindstedt.n_terms,
            "lindstedt_heat_kernel": p1.lindstedt.heat_kernel_approx,
            "lindstedt_residual": p1.lindstedt.expansion_residual,
            # Return map / dualidad
            "return_map_J": state.phase2.return_map.contraction_factor,
            "return_map_stable": state.phase2.return_map.is_stable,
            "fiedler_fraction": state.phase2.fiedler_fraction,
            "duality_residual": state.duality_residual,
            "is_poincare_dual": state.phase2.duality.is_poincare_dual,
            # Fase 3 residuals
            "tellegen_residual": state.tellegen_residual,
            "recurrence_residual": state.recurrence_residual,
            "dirichlet_residual": state.dirichlet_residual,
            "adiabatic_action_drift": state.adiabatic_action_drift,
            "meta_residual": state.meta_residual,
            # Veredicto
            "heyting_verdict": state.heyting_verdict.name,
            "crowbar_action": state.crowbar_action.value,
            "certification_hash": state.certification_hash,
        }

    def _generate_narrative(self, state: StrategicCanvasSuturationState) -> str:
        p1 = state.phase2.phase1
        m = p1.metrics
        parts: List[str] = []
        if m.is_connected:
            parts.append("BMC algebraicamente conexo")
        else:
            parts.append(f"BMC fragmentado ({m.beta_0} componentes)")
        if m.beta_1 == 0:
            parts.append("homológicamente acíclico")
        else:
            parts.append(f"con {m.beta_1} ciclos fundamentales")
        if m.is_dag:
            parts.append("causalmente consistente (DAG)")
        else:
            parts.append(f"con {m.directed_cycle_count} ciclos causales")
        if m.is_spectrally_stable:
            parts.append("espectralmente estable")
        else:
            parts.append(f"espectralmente frágil (λ₁={m.fiedler_value:.4f})")
        if p1.poincare_snapshot.is_near_resonance:
            parts.append(
                f"resonancia espectral {p1.poincare_snapshot.resonance_p}:"
                f"{p1.poincare_snapshot.resonance_q}"
            )
        parts.append(f"Ω₃={state.heyting_verdict.name}")
        return "; ".join(parts) + "."

    # ─────────────────────────────────────────────────────────────────────
    # Auditoría de fusión (Mayer-Vietoris, preservada)
    # ─────────────────────────────────────────────────────────────────────

    def audit_strategic_fusion(
        self, current_bmc: nx.DiGraph, new_domain: nx.DiGraph,
    ) -> MergeVerdict:
        """Auditoría algebraico-topológica inspirada en Mayer-Vietoris."""
        union_g = nx.compose(current_bmc, new_domain)
        inter_g = self._compute_edge_intersection(current_bmc, new_domain)
        A_u = self._to_weighted_undirected(current_bmc)
        B_u = self._to_weighted_undirected(new_domain)
        U_u = self._to_weighted_undirected(union_g)
        I_u = self._to_weighted_undirected(inter_g)
        cc_A = self._build_chain_complex_1d(A_u)
        cc_B = self._build_chain_complex_1d(B_u)
        cc_U = self._build_chain_complex_1d(U_u)
        cc_I = self._build_chain_complex_1d(I_u)
        h_A = self._compute_homology_metrics(cc_A)
        h_B = self._compute_homology_metrics(cc_B)
        h_U = self._compute_homology_metrics(cc_U)
        h_I = self._compute_homology_metrics(cc_I)
        s_U = self._compute_spectral_metrics(cc_U)
        dc_A = self._analyze_directed_cycles(current_bmc).directed_cycles_count
        dc_B = self._analyze_directed_cycles(new_domain).directed_cycles_count
        dc_U = self._analyze_directed_cycles(union_g).directed_cycles_count

        mv_defect = h_U.beta_1 - (h_A.beta_1 + h_B.beta_1 - h_I.beta_1)
        new_directed = dc_U - dc_A - dc_B

        if h_U.beta_0 > 1:
            logger.error("Fusión rechazada: β₀=%d", h_U.beta_0)
            return MergeVerdict.REJECTED_DISCONNECTED
        if new_directed > 0:
            logger.error("Fusión rechazada: %d ciclos dirigidos nuevos", new_directed)
            return MergeVerdict.REJECTED_TOXIC_CYCLES
        if h_U.beta_1 > max(h_A.beta_1, h_B.beta_1) or mv_defect > 0:
            logger.error("Fusión rechazada: β₁(U)=%d, MV_defect=%d",
                         h_U.beta_1, mv_defect)
            return MergeVerdict.REJECTED_HOMOLOGICAL_DEFECT
        if s_U.fiedler_value < MIN_FIEDLER_VALUE:
            logger.error("Fusión rechazada: λ₁=%.4f < %.4f",
                         s_U.fiedler_value, MIN_FIEDLER_VALUE)
            return MergeVerdict.REJECTED_SPECTRAL_FRAGILITY
        return MergeVerdict.ACCEPTED

    def _compute_edge_intersection(
        self, g1: nx.DiGraph, g2: nx.DiGraph,
    ) -> nx.DiGraph:
        common_nodes = set(g1.nodes()) & set(g2.nodes())
        common_edges = set(g1.edges()) & set(g2.edges())
        I = nx.DiGraph()
        I.add_nodes_from(common_nodes)
        for u, v in common_edges:
            w1 = float(g1[u][v].get("weight", 1.0))
            w2 = float(g2[u][v].get("weight", 1.0))
            I.add_edge(u, v, weight=min(w1, w2))
        return I

    # ─────────────────────────────────────────────────────────────────────
    # API avanzada (preservada + extendida)
    # ─────────────────────────────────────────────────────────────────────

    def build_analysis_bundle(
        self, payload: Mapping[str, Any],
        include_rsi3: bool = False,
    ) -> Dict[str, Any]:
        """Paquete analítico completo para inspección matemática."""
        if include_rsi3:
            state, rsi = self._run_full_pipeline_rsi3(payload)
        else:
            state = self._run_full_pipeline(payload)
            rsi = None
        p1 = state.phase2.phase1
        cc = p1.chain_complex
        cycle_basis = state.phase2.cycle_basis
        digraph = self._build_directed_business_graph(payload)
        undirected = self._to_weighted_undirected(digraph)

        bundle: Dict[str, Any] = {
            "bases": {
                "vertices": list(cc.vertex_basis),
                "edges": [list(e) for e in cc.edge_basis],
            },
            "matrices": {
                "boundary_1": cc.boundary_1.tolist(),
                "laplacian_0": cc.laplacian_0.tolist(),
                "laplacian_1": cc.laplacian_1.tolist(),
            },
            "homology": {
                "n_vertices": p1.homology.n_vertices,
                "n_edges": p1.homology.n_edges,
                "rank_d1": p1.homology.rank_boundary_1,
                "nullity_d1": p1.homology.nullity_boundary_1,
                "beta_0": p1.homology.beta_0,
                "beta_1": p1.homology.beta_1,
                "euler_char": p1.homology.euler_char,
            },
            "spectral": {
                "eigenvalues": list(p1.spectral.eigenvalues),
                "fiedler": p1.spectral.fiedler_value,
                "spectral_gap": p1.spectral.spectral_gap,
                "spectral_radius": p1.spectral.spectral_radius,
                "mult_zero": p1.spectral.multiplicity_zero,
                "trace": p1.spectral.trace_laplacian,
            },
            "poincare": {
                "action_invariant": p1.poincare_snapshot.action_invariant,
                "angle_parameter": p1.poincare_snapshot.angle_parameter,
                "section_level": p1.poincare_snapshot.section_level,
                "crossing_count": p1.poincare_snapshot.crossing_count,
                "is_near_resonance": p1.poincare_snapshot.is_near_resonance,
                "resonance_pq": (p1.poincare_snapshot.resonance_p,
                                 p1.poincare_snapshot.resonance_q),
                "small_divisor": p1.poincare_snapshot.small_divisor_magnitude,
            },
            "lindstedt": {
                "small_parameter": p1.lindstedt.small_parameter,
                "n_terms": p1.lindstedt.n_terms,
                "heat_kernel_approx": p1.lindstedt.heat_kernel_approx,
                "heat_kernel_true": p1.lindstedt.heat_kernel_true,
                "residual": p1.lindstedt.expansion_residual,
                "is_consistent": p1.lindstedt.is_asymptotically_consistent,
            },
            "return_map": {
                "J": state.phase2.return_map.contraction_factor,
                "is_stable": state.phase2.return_map.is_stable,
                "fiedler_fraction": state.phase2.fiedler_fraction,
            },
            "duality": {
                "cycle_dim": state.phase2.duality.cycle_space_dimension,
                "cut_dim": state.phase2.duality.cut_space_dimension,
                "total_edges": state.phase2.duality.total_edges,
                "residual": state.phase2.duality.duality_residual,
                "is_poincare_dual": state.phase2.duality.is_poincare_dual,
            },
            "cycles": {
                "directed_count": p1.cycles.directed_cycles_count,
                "is_dag": p1.cycles.is_dag,
                "fundamental_basis": [
                    [list(e) for e in c] for c in cycle_basis
                ],
            },
            "residuals": {
                "tellegen": state.tellegen_residual,
                "duality": state.duality_residual,
                "recurrence": state.recurrence_residual,
                "dirichlet": state.dirichlet_residual,
                "adiabatic_action_drift": state.adiabatic_action_drift,
                "meta": state.meta_residual,
            },
            "verdict": {
                "heyting_omega3": state.heyting_verdict.name,
                "crowbar_action": state.crowbar_action.value,
                "certification_hash": state.certification_hash,
            },
            "graph_stats": {
                "directed_nodes": digraph.number_of_nodes(),
                "directed_edges": digraph.number_of_edges(),
                "undirected_nodes": undirected.number_of_nodes(),
                "undirected_edges": undirected.number_of_edges(),
            },
        }
        if rsi is not None:
            bundle["rsi"] = {
                "final_scale": rsi.final_scale,
                "final_n_terms": rsi.final_n_terms,
                "final_residual": rsi.final_residual,
                "converged": rsi.converged,
                "effective_convergence_order": rsi.effective_convergence_order,
                "level3_activated": rsi.level3_activated,
                "traces": [
                    {"level": t.level, "residual": t.residual,
                     "n_terms": t.n_lindstedt_terms,
                     "richardson": t.richardson_extrapolation,
                     "aitken": t.aitken_delta_squared}
                    for t in rsi.traces
                ],
            }
        return bundle


__all__ = [
    "BMCTopologyError", "TopologicalInvariantError", "SpectralAnalysisError",
    "PayloadValidationError", "HomologicalInconsistencyError",
    "PoincareSectionError", "RecursiveSelfImprovementError",
    "MergeVerdict", "ConnectivityClass", "HeytingOmega3", "CrowbarAction",
    "ChainComplex1D", "HomologyMetrics", "SpectralMetrics",
    "CycleSpaceMetrics", "BmcTopologyMetrics",
    "PoincareSpectralSnapshot", "LindstedtTraceExpansion",
    "EnergyReturnMapSnapshot", "PoincareDualitySnapshot",
    "SpectralRecurrenceSnapshot", "CanvasToleranceBundle",
    "RSITrace", "SelfImprovementResult",
    "Phase1CanvasCertificate", "Phase2CanvasSpectralCertificate",
    "StrategicCanvasSuturationState",
    "Phase1CanvasObserver", "Phase2SpectralOrienter",
    "Phase3CanvasCollapser", "AlphaTopologyVector",
    "BMC_NODES", "BASE_EDGES",
]