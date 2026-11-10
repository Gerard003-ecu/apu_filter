# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Alpha Boundary Agent (Orquestador de Haces Celulares)               ║
║ Ruta   : app/agents/alfa/alpha_agent.py                                      ║
║ Versión: 5.0.0-Rigorous-Sheaf-Cohomology-Poincare-RSI3-Consensus-NoHardware  ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Orquestador macroscópico del Estrato α. Instancia un Haz Celular F sobre el
1-complejo del BMC, ensambla el Laplaciano L_F = δᵀG⁻¹δ, verifica la identidad
de Euler-Poincaré χ(K)=β₀-β₁ y audita la conectividad de Fiedler λ₂(L₀).

NOVEDADES v5.0.0 (Poincaré + RSI-3):
  [P1] Snapshot de Poincaré en el espacio espectral (q=λ₂, p=λ_max−λ₂).
  [P2] Sección Σ = {q = MIN_FIEDLER}; resonancia λ₂/(λ_max−λ₂) vía fracciones
       continuas; divisor pequeño 1/q.
  [P3] Expansión de Lindstedt-Poincaré del heat kernel Φ(ε) = n⁻¹tr(e^{−εL₀})
       con ε = 1/(1+λ_max) y truncación en n_lindstedt_terms.
  [P4] Aplicación de retorno J = min(q/p, p/q); estabilidad |J| ≤ 1.
  [P5] Dualidad de Poincaré dim Z₁ + dim B¹ = |E|.
  [P6] Recurrencia espectral (distancias entre autovalores normalizados).
  [P7] Verificación de Tellegen vía identidad de Dirichlet
       φᵀL₀φ = Σ_{(u,v)∈E} w_uv(φᵤ−φᵥ)².
  [P8] Meta-controlador RSI-3 (Richardson → orden p → Aitken Δ²).
  Compatibilidad: con bundle default, los 3 vetos disparan idéntico a v4.0.0.

INVARIANTES (I1–I5) — preservados:
  [I1] L_F = δᵀG⁻¹δ ⪰ 0.
  [I2] ⟨s_val, ψ_ker⟩ = 0 ∀ψ_ker ∈ ker(L_F).
  [I3] χ(K) = β₀ − β₁ = |V| − |E| (verificación cruzada).
  [I4] λ₂(L₀) ≥ MIN_FIEDLER > 0 ⟺ β₀ = 1.
  [I5] Hom_D(F(X), Y) ≅ Hom_C(X, G(Y)).

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial):
  Fase 1 ─► _Phase1_CanvasSimplicialFibrator (Observe + Poincaré + Lindstedt)
            Última: _phase1_to_phase2_bridge → Phase2Certificate
  Fase 2 ─► _Phase2_HomologicalBettiAuditor (Orient + Return Map + Dualidad)
            Última: _phase2_to_phase3_bridge → Phase3Certificate
  Fase 3 ─► _Phase3_SpectralFiedlerAuditor (Decide + Tellegen + Heyting + RSI-3)
            Última: collapse_to_verdict → AlphaBoundaryVerdict

Funtor Maestro:
  𝓕 = Φ₃ ∘ Φ₂ ∘ Φ₁ : (nodes, flows) ⟶ AlphaBoundaryVerdict
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    Any, Dict, FrozenSet, List, NamedTuple, Optional, Protocol, Set, Tuple,
    Type, runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import (
        Morphism, CategoricalState, TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        pass

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class CategoricalState:  # type: ignore[no-redef]
        payload: Dict[str, Any]
        metadata: Dict[str, Any]
        stratum: Any

        def __init__(
            self, payload: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None,
            stratum: Any = None,
        ) -> None:
            self.payload = payload
            self.metadata = metadata or {}
            self.stratum = stratum

    class Stratum(Enum):  # type: ignore[no-redef]
        ALPHA = auto()


logger = logging.getLogger("MIC.Alpha.BoundaryAgent")

__version__: str = (
    "5.0.0-Rigorous-Sheaf-Cohomology-Poincare-RSI3-Consensus-NoHardware"
)

# ── Poincaré ──────────────────────────────────────────────────────────────
_POINCARE_MAX_DENOMINATOR: int = 16
_POINCARE_LOW_ORDER: int = 8
_RESONANCE_TOLERANCE: float = 1.0e-3
_SMALL_PARAMETER_FLOOR: float = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: float = 1.0e-6

# ── RSI-3 ─────────────────────────────────────────────────────────────────
_RSI_MAX_DEPTH: int = 3
_RSI_TARGET_RESIDUAL: float = 1.0e-3
_RSI_MIN_SCALE: float = 1.0e-3
_RSI_MAX_SCALE: float = 1.0e3
_RSI_FACTOR_LOWER: float = 1.0e-2
_RSI_FACTOR_UPPER: float = 0.99
_RSI_OSC_EPS: float = 1.0e-15
_RSI_TERMS_CAP: int = 10


# ═════════════════════════════════════════════════════════════════════════════
# A. CONSTANTES
# ═════════════════════════════════════════════════════════════════════════════

class AlphaConstants:
    """Constantes topológicas y tolerancias numéricas del Estrato α."""

    RANK_TOLERANCE: float = 1e-10
    MIN_FIEDLER_VALUE: float = 0.05
    MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

    @staticmethod
    def svd_rank_threshold(matrix: NDArray[np.float64]) -> float:
        if matrix.size == 0:
            return AlphaConstants.RANK_TOLERANCE
        m, n = matrix.shape if matrix.ndim == 2 else (matrix.size, 1)
        sigma_max = float(np.linalg.norm(matrix, ord=2))
        if sigma_max < AlphaConstants.RANK_TOLERANCE:
            return AlphaConstants.RANK_TOLERANCE
        tol_rel = max(m, n) * AlphaConstants.MACHINE_EPSILON * sigma_max
        return max(tol_rel, AlphaConstants.RANK_TOLERANCE)

    @staticmethod
    def laplacian_zero_tolerance(n: int, frobenius_norm: float) -> float:
        base = frobenius_norm if frobenius_norm > 0.0 else 1.0
        return n * AlphaConstants.MACHINE_EPSILON * base


# ═════════════════════════════════════════════════════════════════════════════
# B. EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════

class AlphaBoundaryError(TopologicalInvariantError):
    pass


class PreconditionError(AlphaBoundaryError):
    pass


class ToxicCycleVetoError(AlphaBoundaryError):
    pass


class EulerPoincareDegeneracyError(AlphaBoundaryError):
    pass


class SpectralFragilityError(AlphaBoundaryError):
    pass


class PoincareSectionError(AlphaBoundaryError):
    pass


class RecursiveSelfImprovementError(AlphaBoundaryError):
    pass


# ═════════════════════════════════════════════════════════════════════════════
# C. RETÍCULO HEYTING Ω₃ Y CROWBAR
# ═════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""Retículo de Heyting Ω₃ (Gödel-Dummett): ⊥ < intermedio < ⊤."""
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

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
# D. FUNCIONES AUXILIARES (fracciones continuas, Richardson, Aitken)
# ═════════════════════════════════════════════════════════════════════════════

def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
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
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)
    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)
    err = abs(ratio - p / q)
    is_res = (err < tolerance) and (q <= low_order)
    return (bool(is_res), int(p), int(q), float(err), float(1.0 / q))


def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float,
) -> Tuple[float, float]:
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
    d = r_n - 2.0 * r_nm1 + r_nm2
    if abs(d) < _RSI_OSC_EPS:
        return r_n
    return float(r_n - (r_n - r_nm1) ** 2 / d)


def _adaptive_refinement_factor(
    current_residual: float, target_residual: float, convergence_rate: float,
    level3_active: bool, aitken_signal: float,
) -> float:
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


def _compute_hash(*fields: Any) -> str:
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
        elif f is None:
            h.update(b"|__None__")
        else:
            h.update(f"|{f!r}".encode())
    return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# E. NUEVOS DTOs (Poincaré + Lindstedt + RSI-3)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareSpectralSnapshot:
    """Snapshot de Poincaré en el espacio (q=λ₂, p=λ_max−λ₂)."""
    action_invariant: float
    angle_parameter: float
    section_level: float
    crossing_count: int
    frequency_ratio: float
    is_near_resonance: bool
    resonance_p: int
    resonance_q: int
    small_divisor_magnitude: float


@dataclass(frozen=True, slots=True)
class LindstedtTraceExpansion:
    """Expansión de Lindstedt-Poincaré del heat kernel Φ(ε) = n⁻¹tr(e^{−εL₀})."""
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


@dataclass(frozen=True, slots=True)
class EnergyReturnMapSnapshot:
    """Aplicación de retorno J = min(q/p, p/q) en el espacio (q, p)."""
    contraction_factor: float
    eigenvalues_modulus: Tuple[float, ...]
    is_stable: bool
    iterations_norm_sequence: Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PoincareDualitySnapshot:
    """Dualidad de Poincaré: dim Z₁ + dim B¹ = |E|."""
    cycle_space_dimension: int
    cut_space_dimension: int
    total_edges: int
    duality_residual: float
    is_poincare_dual: bool


@dataclass(frozen=True, slots=True)
class SpectralRecurrenceSnapshot:
    """Recurrencia en el espectro del Laplaciano."""
    recurrence_points: Tuple[float, ...]
    min_recurrence_distance: float
    is_poincare_recurrent: bool
    n_samples: int


@dataclass(frozen=True, slots=True)
class AlphaToleranceBundle:
    """Bundle de tolerancias adaptativas para RSI-3.

    scale = 1.0 → tolerancias v4.0.0 (compatibilidad exacta).
    """
    scale: float = 1.0
    n_lindstedt_terms: int = 2

    def _scaled(self, base: float) -> float:
        return max(1.0e-14, base * self.scale)

    @property
    def rank_tolerance(self) -> float:
        return self._scaled(AlphaConstants.RANK_TOLERANCE)

    @property
    def min_fiedler_value(self) -> float:
        return AlphaConstants.MIN_FIEDLER_VALUE * self.scale

    @property
    def duality_tol(self) -> float:
        return self._scaled(1.0e-8)

    @property
    def tellegen_tol(self) -> float:
        return self._scaled(1.0e-8)

    def with_scale(self, s: float) -> "AlphaToleranceBundle":
        return AlphaToleranceBundle(
            scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, s))),
            n_lindstedt_terms=self.n_lindstedt_terms,
        )

    def with_terms(self, n: int) -> "AlphaToleranceBundle":
        return AlphaToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=int(max(2, min(_RSI_TERMS_CAP, n))),
        )


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
# F. DTOs EXISTENTES (extendidos + compatibles)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SimplicialComplexData:
    """1-esqueleto ponderado del BMC (sin cambios estructurales)."""
    vertices: Tuple[str, ...]
    edges: Tuple[Tuple[str, str, float], ...]
    boundary_operator: NDArray[np.float64]
    n_vertices: int
    n_edges: int

    def __post_init__(self) -> None:
        expected_shape = (self.n_vertices, self.n_edges)
        if self.boundary_operator.shape != expected_shape:
            raise PreconditionError(
                f"∂₁.shape={self.boundary_operator.shape} ≠ "
                f"({self.n_vertices},{self.n_edges})."
            )
        if not np.all(np.isfinite(self.boundary_operator)):
            raise PreconditionError(
                "∂₁ contiene valores no finitos (NaN/Inf)."
            )
        for (u, v, w) in self.edges:
            if w <= 0.0:
                raise PreconditionError(
                    f"Arista ({u},{v}) tiene peso w={w} ≤ 0."
                )


@dataclass(frozen=True, slots=True)
class HomologicalInvariants:
    """Invariantes algebraicos (sin cambios estructurales)."""
    rank: int
    beta_0: int
    beta_1: int
    euler_characteristic: int
    svd_singular_values: NDArray[np.float64]
    rank_threshold: float

    def __post_init__(self) -> None:
        computed = self.beta_0 - self.beta_1
        if computed != self.euler_characteristic:
            raise TopologicalInvariantError(
                f"Euler-Poincaré: β₀-β₁={computed} ≠ χ={self.euler_characteristic}."
            )


@dataclass(frozen=True, slots=True)
class SpectralFiedlerData:
    """Espectro del Laplaciano (sin cambios estructurales)."""
    laplacian: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    fiedler_value: float
    zero_tolerance: float
    is_connected: bool
    spectral_gap: float

    def __post_init__(self) -> None:
        if self.eigenvalues.size > 0:
            if float(self.eigenvalues[0]) < -self.zero_tolerance:
                raise TopologicalInvariantError(
                    f"L₀ no PSD: λ_min={float(self.eigenvalues[0]):.4e}."
                )


@dataclass(frozen=True, slots=True)
class Phase1Certificate:
    """Certificado enriquecido de Φ₁ (Observe + Poincaré + Lindstedt)."""
    complex_data: SimplicialComplexData
    poincare_snapshot: Optional[PoincareSpectralSnapshot]
    lindstedt_expansion: Optional[LindstedtTraceExpansion]
    tolerance_bundle: AlphaToleranceBundle
    certification_hash: str


@dataclass(frozen=True, slots=True)
class Phase2Certificate:
    """Certificado enriquecido de Φ₂ (Orient + Return Map + Dualidad)."""
    phase1: Phase1Certificate
    homology: HomologicalInvariants
    return_map: EnergyReturnMapSnapshot
    duality: PoincareDualitySnapshot
    spectral_recurrence: SpectralRecurrenceSnapshot
    fiedler_fraction: float
    certification_hash: str


@dataclass(frozen=True, slots=True)
class Phase3Certificate:
    """Certificado enriquecido de Φ₃ (Decide + Tellegen + Heyting)."""
    phase2: Phase2Certificate
    spectral: SpectralFiedlerData
    tellegen_residual: float
    duality_residual: float
    recurrence_residual: float
    dirichlet_residual: float
    adiabatic_action_drift: float
    meta_residual: float
    heyting_verdict: HeytingOmega3
    crowbar_action: CrowbarAction
    certification_hash: str


@dataclass(frozen=True, slots=True)
class AlphaBoundaryVerdict:
    """Veredicto inmutable — EXTENDIDO con campos opcionales (backward compat).

    Todos los campos nuevos tienen default None / 0.0 → instancias previas
    construidas con solo los 5 campos originales siguen siendo válidas.
    """
    is_viable: bool
    homology: HomologicalInvariants
    spectral: Optional[SpectralFiedlerData]
    veto_reason: Optional[str]
    veto_class: Optional[Type[AlphaBoundaryError]]
    # ── Añadidos v5.0.0 (todos opcionales con default) ──────────────────
    poincare_snapshot: Optional[PoincareSpectralSnapshot] = None
    lindstedt_expansion: Optional[LindstedtTraceExpansion] = None
    energy_return_map: Optional[EnergyReturnMapSnapshot] = None
    poincare_duality: Optional[PoincareDualitySnapshot] = None
    spectral_recurrence: Optional[SpectralRecurrenceSnapshot] = None
    heyting_verdict: Optional[HeytingOmega3] = None
    crowbar_action: Optional[CrowbarAction] = None
    tellegen_residual: float = 0.0
    duality_residual: float = 0.0
    recurrence_residual: float = 0.0
    dirichlet_residual: float = 0.0
    adiabatic_action_drift: float = 0.0
    meta_residual: float = 0.0
    rsi_result: Optional[SelfImprovementResult] = None


# ═════════════════════════════════════════════════════════════════════════════
# G. PROTOCOLOS FUNTORIALES (sin cambios)
# ═════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class CanvasFibratorPort(Protocol):
    def project_canvas_to_simplicial_complex(
        self, nodes: List[str], flows: List[Tuple[str, str, float]]
    ) -> SimplicialComplexData: ...


@runtime_checkable
class HomologicalAuditorPort(Protocol):
    def audit_betti_invariants(
        self, complex_data: SimplicialComplexData
    ) -> HomologicalInvariants: ...


@runtime_checkable
class SpectralAuditorPort(Protocol):
    def audit_fiedler_connectivity(
        self, complex_data: SimplicialComplexData
    ) -> SpectralFiedlerData: ...


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║   ORQUESTADOR SUPREMO: AlphaBoundaryAgent v5.0.0                           ║
# ║                                                                            ║
# ║   Las tres fases anidadas están encadenadas por sus DTOs inmutables.       ║
# ║   Añadido v5.0.0: Poincaré + Lindstedt + RSI-3 sin romper la API.          ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class AlphaBoundaryAgent(Morphism):
    r"""Orquestador macroscópico del Estrato α con Poincaré + RSI-3.

    Preserva 100% de la API v4.0.0 y añade:
      • Fase 1: snapshot de Poincaré + expansión de Lindstedt-Poincaré.
      • Fase 2: aplicación de retorno + dualidad de Poincaré + recurrencia.
      • Fase 3: Tellegen-Dirichlet + Heyting Ω₃ + Crowbar + RSI-3.
    """

    # ╔═════════════════════════════════════════════════════════════════════╗
    # ║                                                                     ║
    # ║   FASE 1: FIBRACIÓN SIMPLICIAL + POINCARÉ + LINDSTED                ║
    # ║                                                                     ║
    # ║   Métodos originales (preservados íntegros):                        ║
    # ║     · _validate_canvas_input                                        ║
    # ║     · _build_boundary_operator                                      ║
    # ║     · project_canvas_to_simplicial_complex                          ║
    # ║                                                                     ║
    # ║   Nuevos métodos (v5.0.0):                                          ║
    # ║     · _compute_poincare_spectral_snapshot                           ║
    # ║     · _compute_lindstedt_trace_expansion                            ║
    # ║     · certify_phase1_composite                                      ║
    # ║                                                                     ║
    # ║   ULTIMO MÉTODO (morfismo terminal = unidad de Fase 2):              ║
    # ║     · _phase1_to_phase2_bridge → Phase2Certificate                  ║
    # ║                                                                     ║
    # ╚═════════════════════════════════════════════════════════════════════╝

    class _Phase1_CanvasSimplicialFibrator:
        """FASE 1 — Fibración simplicial + Poincaré + Lindstedt."""

        # ─────────────────────────────────────────────────────────────
        # 1.1 Validación estructural (PRESERVADA ÍNTEGRAMENTE)
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _validate_canvas_input(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> None:
            if not nodes:
                raise PreconditionError(
                    "El colector del BMC no puede ser el conjunto vacío ∅."
                )
            for i, node in enumerate(nodes):
                if not isinstance(node, str):
                    raise PreconditionError(
                        f"nodes[{i}]={node!r} debe ser str, "
                        f"recibido {type(node).__name__}."
                    )
            if len(set(nodes)) != len(nodes):
                duplicates = [n for n in nodes if nodes.count(n) > 1]
                unique_dups = list(dict.fromkeys(duplicates))
                raise PreconditionError(
                    f"nodes contiene duplicados: {unique_dups}."
                )
            node_set: FrozenSet[str] = frozenset(nodes)
            seen_edges: Set[FrozenSet[str]] = set()
            for k, flow in enumerate(flows):
                if (not isinstance(flow, tuple) or len(flow) != 3
                        or not isinstance(flow[0], str)
                        or not isinstance(flow[1], str)
                        or not isinstance(flow[2], (int, float))):
                    raise PreconditionError(
                        f"flows[{k}]={flow!r} debe ser Tuple[str,str,float]."
                    )
                u, v, w = flow[0], flow[1], float(flow[2])
                if u == v:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}) es auto-bucle."
                    )
                edge_key: FrozenSet[str] = frozenset({u, v})
                if edge_key in seen_edges:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}) es multi-arista."
                    )
                seen_edges.add(edge_key)
                if not np.isfinite(w):
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}): peso no finito."
                    )
                if w <= 0.0:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}): w={w} ≤ 0."
                    )
                if u not in node_set:
                    raise PreconditionError(
                        f"flows[{k}]: origen '{u}' ∉ nodes."
                    )
                if v not in node_set:
                    raise PreconditionError(
                        f"flows[{k}]: destino '{v}' ∉ nodes."
                    )

        # ─────────────────────────────────────────────────────────────
        # 1.2 Operador frontera (PRESERVADO ÍNTEGRAMENTE)
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _build_boundary_operator(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> NDArray[np.float64]:
            n_v = len(nodes)
            n_e = len(flows)
            node_idx = {node: idx for idx, node in enumerate(nodes)}
            B = np.zeros((n_v, n_e), dtype=np.float64)
            for k, (u, v, w) in enumerate(flows):
                sqrt_w = np.sqrt(float(w))
                B[node_idx[u], k] = -sqrt_w
                B[node_idx[v], k] = +sqrt_w
            return B

        # ─────────────────────────────────────────────────────────────
        # 1.3 project_canvas_to_simplicial_complex (PRESERVADO)
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def project_canvas_to_simplicial_complex(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> SimplicialComplexData:
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator._validate_canvas_input(
                nodes, flows
            )
            boundary_op = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                ._build_boundary_operator(nodes, flows)
            )
            complex_data = SimplicialComplexData(
                vertices=tuple(nodes),
                edges=tuple((u, v, float(w)) for (u, v, w) in flows),
                boundary_operator=boundary_op,
                n_vertices=len(nodes),
                n_edges=len(flows),
            )
            logger.debug(
                "Fase 1: K=(%d vértices, %d aristas). ‖∂₁‖_F=%.4f.",
                complex_data.n_vertices, complex_data.n_edges,
                float(np.linalg.norm(boundary_op, 'fro')),
            )
            return complex_data

        # ─────────────────────────────────────────────────────────────
        # 1.4 NUEVO: Snapshot de Poincaré en el espacio espectral
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_poincare_spectral_snapshot(
            complex_data: SimplicialComplexData,
            section_level: Optional[float] = None,
        ) -> Optional[PoincareSpectralSnapshot]:
            r"""Snapshot en (q=λ₂, p=λ_max−λ₂).

            I = ½(q²+p²), θ_p = atan2(p, q).
            Sección Σ={q=MIN_FIEDLER}; resonancia λ₂/(λ_max−λ₂).
            Retorna None si |V| < 2 (Fiedler no definido).
            """
            if complex_data.n_vertices < 2:
                return None
            B = complex_data.boundary_operator
            L0 = B @ B.T
            L0 = 0.5 * (L0 + L0.T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return None
            if eig.size < 2:
                return None

            zero_tol = AlphaConstants.laplacian_zero_tolerance(
                complex_data.n_vertices,
                float(np.linalg.norm(L0, 'fro')),
            )
            positive = eig[eig > zero_tol]
            q = float(np.min(positive)) if positive.size > 0 else 0.0
            lam_max = float(eig[-1])
            p = max(0.0, lam_max - q)

            action = 0.5 * (q * q + p * p)
            angle = math.atan2(p, q) if action > AlphaConstants.MACHINE_EPSILON else 0.0

            sec = (AlphaConstants.MIN_FIEDLER_VALUE if section_level is None
                   else float(section_level))
            r_sq = 2.0 * action
            if r_sq > AlphaConstants.MACHINE_EPSILON and abs(sec) < math.sqrt(r_sq):
                crossings = 2
            elif (r_sq > AlphaConstants.MACHINE_EPSILON
                  and abs(abs(sec) - math.sqrt(r_sq)) <= AlphaConstants.MACHINE_EPSILON):
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

        # ─────────────────────────────────────────────────────────────
        # 1.5 NUEVO: Expansión de Lindstedt-Poincaré del heat kernel
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_lindstedt_trace_expansion(
            complex_data: SimplicialComplexData,
            n_terms: int = 2,
        ) -> Optional[LindstedtTraceExpansion]:
            r"""Expansión de Φ(ε) = n⁻¹tr(e^{−εL₀}) con ε = 1/(1+λ_max).

            Truncación: Φ ≈ (1/n) Σ_i Σ_{k<N} (−ε·λ_i)^k / k!.
            Residual = |Φ_true − Φ_approx| / Φ_true.
            """
            if complex_data.n_vertices == 0:
                return None
            B = complex_data.boundary_operator
            L0 = B @ B.T
            L0 = 0.5 * (L0 + L0.T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return None
            n = eig.size
            if n == 0:
                return None

            lam_max = float(np.max(np.abs(eig)))
            eps = 1.0 / (1.0 + max(1e-12, lam_max))
            N = max(2, int(n_terms))

            approx = 0.0
            true = 0.0
            zero = 0.0
            first = 0.0
            second = 0.0
            higher = 0.0
            for lam in eig:
                x = eps * float(lam)
                t0 = 1.0
                t1 = -x
                t2 = x * x / 2.0
                th = 0.0
                fact = 6.0
                xk = x * x * x
                for k in range(3, N):
                    th += ((-1.0) ** k) * xk / fact
                    fact *= (k + 1)
                    xk *= x
                approx += t0 + t1 + t2 + th
                try:
                    true += math.exp(-x)
                except OverflowError:
                    true += 0.0
                zero += t0
                first += t1
                second += t2
                higher += th

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

        # ─────────────────────────────────────────────────────────────
        # 1.6 NUEVO: Certificación completa de Φ₁ (composite)
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def certify_phase1_composite(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
            bundle: Optional[AlphaToleranceBundle] = None,
        ) -> Phase1Certificate:
            r"""Produce Phase1Certificate (SimplicialComplexData + Poincaré + Lindstedt)."""
            b = bundle if bundle is not None else AlphaToleranceBundle()
            complex_data = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                .project_canvas_to_simplicial_complex(nodes, flows)
            )
            poincare = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                ._compute_poincare_spectral_snapshot(complex_data)
            )
            lindstedt = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                ._compute_lindstedt_trace_expansion(
                    complex_data, n_terms=b.n_lindstedt_terms
                )
            )
            cert_hash = _compute_hash(
                complex_data.n_vertices, complex_data.n_edges,
                float(np.linalg.norm(complex_data.boundary_operator, 'fro')),
                (poincare.action_invariant if poincare else 0.0),
                (poincare.is_near_resonance if poincare else False),
                (lindstedt.expansion_residual if lindstedt else 0.0),
            )
            logger.info(
                "[Φ₁ ✓] V=%d, E=%d, I=%.4e, res=%s, LP_res=%.3e, hash=%s.",
                complex_data.n_vertices, complex_data.n_edges,
                (poincare.action_invariant if poincare else 0.0),
                (poincare.is_near_resonance if poincare else False),
                (lindstedt.expansion_residual if lindstedt else 0.0),
                cert_hash[:16] + "...",
            )
            return Phase1Certificate(
                complex_data=complex_data,
                poincare_snapshot=poincare,
                lindstedt_expansion=lindstedt,
                tolerance_bundle=b,
                certification_hash=cert_hash,
            )

        # ─────────────────────────────────────────────────────────────
        # ★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE FASE 2 ★
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _phase1_to_phase2_bridge(
            cert1: Phase1Certificate,
        ) -> Phase2Certificate:
            r"""★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE FASE 2 ★

            Composición F₁ ⊣ F₂:
                Phase1Certificate ──▶ Phase2Certificate
                (incluye homología + return map + dualidad + recurrencia)
            """
            if not isinstance(cert1, Phase1Certificate):
                raise PoincareSectionError(
                    "Fase 1 → 2: se requiere Phase1Certificate."
                )
            complex_data = cert1.complex_data
            bundle = cert1.tolerance_bundle

            # Delegar a Fase 2: homología + return map + dualidad + recurrencia.
            homology = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                .audit_betti_invariants(complex_data)
            )
            return_map = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_return_map(complex_data)
            )
            duality = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_poincare_duality(complex_data, homology, bundle)
            )
            recurrence = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_spectral_recurrence(complex_data)
            )
            eps = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_fiedler_fraction(complex_data)
            )
            cert_hash = _compute_hash(
                cert1.certification_hash,
                homology.beta_0, homology.beta_1,
                return_map.contraction_factor, return_map.is_stable,
                duality.duality_residual, duality.is_poincare_dual,
                recurrence.min_recurrence_distance, eps,
            )
            logger.info(
                "[Φ₂ ✓] β₀=%d, β₁=%d, J=%.4f, dual_res=%.3e, "
                "rec_res=%.3e, ε=%.4f, hash=%s.",
                homology.beta_0, homology.beta_1,
                return_map.contraction_factor, duality.duality_residual,
                recurrence.min_recurrence_distance, eps,
                cert_hash[:16] + "...",
            )
            return Phase2Certificate(
                phase1=cert1,
                homology=homology,
                return_map=return_map,
                duality=duality,
                spectral_recurrence=recurrence,
                fiedler_fraction=eps,
                certification_hash=cert_hash,
            )

    # ╔═════════════════════════════════════════════════════════════════════╗
    # ║                                                                     ║
    # ║   FASE 2: AUDITORÍA HOMOLÓGICA + RETURN MAP + DUALIDAD              ║
    # ║                                                                     ║
    # ║   Métodos originales (preservados íntegros):                        ║
    # ║     · _compute_numerical_rank                                       ║
    # ║     · _compute_betti_numbers                                        ║
    # ║     · _verify_euler_identity                                        ║
    # ║     · audit_betti_invariants                                        ║
    # ║                                                                     ║
    # ║   Nuevos métodos (v5.0.0):                                          ║
    # ║     · _compute_return_map                                           ║
    # ║     · _compute_poincare_duality                                     ║
    # ║     · _compute_spectral_recurrence                                  ║
    # ║     · _compute_fiedler_fraction                                     ║
    # ║                                                                     ║
    # ║   ULTIMO MÉTODO (morfismo terminal = unidad de Fase 3):              ║
    # ║     · _phase2_to_phase3_bridge → Phase3Certificate                  ║
    # ║                                                                     ║
    # ╚═════════════════════════════════════════════════════════════════════╝

    class _Phase2_HomologicalBettiAuditor:
        """FASE 2 — Homología + Return Map + Dualidad + Recurrencia."""

        @staticmethod
        def _compute_numerical_rank(
            boundary_operator: NDArray[np.float64],
        ) -> Tuple[int, NDArray[np.float64], float]:
            if boundary_operator.size == 0:
                return 0, np.array([], dtype=np.float64), AlphaConstants.RANK_TOLERANCE
            singular_values: NDArray[np.float64] = la.svd(
                boundary_operator, compute_uv=False
            )
            threshold = AlphaConstants.svd_rank_threshold(boundary_operator)
            rank = int(np.sum(singular_values > threshold))
            logger.debug(
                "SVD: σ_max=%.4e, σ_min=%.4e, τ=%.4e, rank=%d.",
                float(singular_values[0]) if singular_values.size > 0 else 0.0,
                float(singular_values[-1]) if singular_values.size > 0 else 0.0,
                threshold, rank,
            )
            return rank, singular_values, threshold

        @staticmethod
        def _compute_betti_numbers(
            n_vertices: int, n_edges: int, rank: int,
        ) -> Tuple[int, int, int]:
            beta_0 = n_vertices - rank
            beta_1 = n_edges - rank
            euler_from_betti = beta_0 - beta_1
            euler_direct = n_vertices - n_edges
            if euler_from_betti != euler_direct:
                raise TopologicalInvariantError(
                    f"Euler-Poincaré: β₀-β₁={euler_from_betti} ≠ "
                    f"|V|-|E|={euler_direct}."
                )
            return beta_0, beta_1, euler_direct

        @staticmethod
        def _verify_euler_identity(
            homology: HomologicalInvariants,
            n_vertices: int, n_edges: int,
        ) -> None:
            expected_chi = n_vertices - n_edges
            if homology.euler_characteristic != expected_chi:
                raise TopologicalInvariantError(
                    f"Post-verificación Euler-Poincaré: χ={homology.euler_characteristic} "
                    f"≠ |V|-|E|={expected_chi}."
                )

        @staticmethod
        def audit_betti_invariants(
            complex_data: SimplicialComplexData,
        ) -> HomologicalInvariants:
            rank, sigma_values, threshold = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_numerical_rank(complex_data.boundary_operator)
            )
            beta_0, beta_1, euler_chi = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_betti_numbers(
                    complex_data.n_vertices, complex_data.n_edges, rank,
                )
            )
            homology = HomologicalInvariants(
                rank=rank, beta_0=beta_0, beta_1=beta_1,
                euler_characteristic=euler_chi,
                svd_singular_values=sigma_values,
                rank_threshold=threshold,
            )
            AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor._verify_euler_identity(
                homology, complex_data.n_vertices, complex_data.n_edges
            )
            logger.info(
                "Homología: |V|=%d, |E|=%d, rank=%d ⟹ β₀=%d, β₁=%d, χ=%d.",
                complex_data.n_vertices, complex_data.n_edges, rank,
                beta_0, beta_1, euler_chi,
            )
            return homology

        # ─────────────────────────────────────────────────────────────
        # 2.5 NUEVO: Aplicación de retorno J = min(q/p, p/q)
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_return_map(
            complex_data: SimplicialComplexData,
            n_iterations: int = 4,
        ) -> EnergyReturnMapSnapshot:
            r"""Aplicación de retorno J = min(q/p, p/q).

            q = λ₂, p = λ_max − λ₂. Estable si |J| ≤ 1.
            Para grafos sin λ₂ definido (|V|<2 o L₀=0): J=1 (neutro).
            """
            if complex_data.n_vertices < 2:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                )
            B = complex_data.boundary_operator
            L0 = 0.5 * ((B @ B.T) + (B @ B.T).T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                )
            if eig.size < 2:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                )
            zero_tol = AlphaConstants.laplacian_zero_tolerance(
                complex_data.n_vertices,
                float(np.linalg.norm(L0, 'fro')),
            )
            positive = eig[eig > zero_tol]
            if positive.size == 0:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                )
            q = float(np.min(positive))
            lam_max = float(eig[-1])
            p = max(0.0, lam_max - q)
            if q <= 0.0 or p <= 0.0:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                )
            J = min(q / p, p / q)
            norms: List[float] = []
            cur_q, cur_p = q, p
            for _ in range(n_iterations):
                cur_q = J * q + (1.0 - J) * cur_q
                cur_p = (1.0 / J) * p + (1.0 - 1.0 / J) * cur_p
                norms.append(float(math.hypot(cur_q, cur_p)))
            return EnergyReturnMapSnapshot(
                contraction_factor=float(J),
                eigenvalues_modulus=(float(abs(J)),),
                is_stable=bool(abs(J) <= 1.0 + 1e-9),
                iterations_norm_sequence=tuple(norms),
            )

        # ─────────────────────────────────────────────────────────────
        # 2.6 NUEVO: Dualidad de Poincaré dim Z₁ + dim B¹ = |E|
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_poincare_duality(
            complex_data: SimplicialComplexData,
            homology: HomologicalInvariants,
            bundle: AlphaToleranceBundle,
        ) -> PoincareDualitySnapshot:
            r"""Dualidad de Poincaré: dim Z₁ + dim B¹ = |E|.

            dim Z₁ = β₁ (ciclos), dim B¹ = rank(∂₁) (cobordes).
            Para 1-complejo: β₁ + rank = |E| − rank + rank = |E| ✓.
            """
            cycle_dim = homology.beta_1
            cut_dim = homology.rank
            total = complex_data.n_edges
            residual = abs((cycle_dim + cut_dim) - total) / max(1, total)
            is_dual = residual <= bundle.duality_tol
            return PoincareDualitySnapshot(
                cycle_space_dimension=int(cycle_dim),
                cut_space_dimension=int(cut_dim),
                total_edges=int(total),
                duality_residual=float(residual),
                is_poincare_dual=bool(is_dual),
            )

        # ─────────────────────────────────────────────────────────────
        # 2.7 NUEVO: Recurrencia espectral
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_spectral_recurrence(
            complex_data: SimplicialComplexData,
            n_samples: int = 8,
        ) -> SpectralRecurrenceSnapshot:
            r"""Recurrencia en la secuencia de autovalores normalizados.

            Distancia entre λ_i consecutivos normalizados; recurrencia si
            dos muestras consecutivas distan < 0.1.
            """
            B = complex_data.boundary_operator
            L0 = 0.5 * ((B @ B.T) + (B @ B.T).T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return SpectralRecurrenceSnapshot(
                    recurrence_points=(), min_recurrence_distance=float("inf"),
                    is_poincare_recurrent=False, n_samples=0,
                )
            if eig.size < 2:
                return SpectralRecurrenceSnapshot(
                    recurrence_points=(), min_recurrence_distance=float("inf"),
                    is_poincare_recurrent=False, n_samples=int(eig.size),
                )
            seq = np.abs(eig[:n_samples])
            seq = seq / max(1e-12, float(np.max(seq)))
            distances: List[float] = []
            points: List[float] = []
            for i in range(1, len(seq)):
                d = float(abs(seq[i] - seq[i - 1]))
                distances.append(d)
                if d < 0.1:
                    points.append(float(seq[i]))
            min_dist = min(distances) if distances else float("inf")
            return SpectralRecurrenceSnapshot(
                recurrence_points=tuple(points),
                min_recurrence_distance=float(min_dist),
                is_poincare_recurrent=(len(points) >= 2),
                n_samples=int(len(seq)),
            )

        # ─────────────────────────────────────────────────────────────
        # 2.8 NUEVO: Fracción de Fiedler ε = λ₂/λ_max
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_fiedler_fraction(
            complex_data: SimplicialComplexData,
        ) -> float:
            """ε = λ₂ / λ_max ∈ [0, 1]; 0 si λ_max ≈ 0."""
            B = complex_data.boundary_operator
            L0 = 0.5 * ((B @ B.T) + (B @ B.T).T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return 0.0
            if eig.size < 2:
                return 0.0
            zero_tol = AlphaConstants.laplacian_zero_tolerance(
                complex_data.n_vertices,
                float(np.linalg.norm(L0, 'fro')),
            )
            positive = eig[eig > zero_tol]
            if positive.size == 0:
                return 0.0
            q = float(np.min(positive))
            lam_max = float(eig[-1])
            if lam_max <= AlphaConstants.MACHINE_EPSILON:
                return 0.0
            return float(min(1.0, max(0.0, q / lam_max)))

        # ─────────────────────────────────────────────────────────────
        # ★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE FASE 3 ★
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _phase2_to_phase3_bridge(
            cert2: Phase2Certificate,
        ) -> Phase3Certificate:
            r"""★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE FASE 3 ★

            Composición F₂ ⊣ F₃:
                Phase2Certificate ──▶ Phase3Certificate
                (incluye espectro + Tellegen-Dirichlet + Heyting + Crowbar)
            """
            if not isinstance(cert2, Phase2Certificate):
                raise PoincareSectionError(
                    "Fase 2 → 3: se requiere Phase2Certificate."
                )
            complex_data = cert2.phase1.complex_data
            bundle = cert2.phase1.tolerance_bundle

            # Delegar a Fase 3: espectro + residuales + Heyting.
            spectral = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                .audit_fiedler_connectivity(complex_data)
            )
            tellegen = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._compute_tellegen_dirichlet_residual(complex_data)
            )
            duality_res = float(cert2.duality.duality_residual)
            recurrence_res = (
                float(min(1.0, cert2.spectral_recurrence.min_recurrence_distance))
                if math.isfinite(cert2.spectral_recurrence.min_recurrence_distance)
                else 0.0
            )
            adiabatic = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._compute_adiabatic_action_drift(cert2)
            )
            meta_res = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._combine_residuals(
                    tellegen, duality_res, recurrence_res, tellegen,
                    adiabatic,
                    lindstedt_residual=(
                        cert2.phase1.lindstedt_expansion.expansion_residual
                        if cert2.phase1.lindstedt_expansion else None
                    ),
                )
            )
            verdict = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._resolve_heyting(
                    cert2, spectral, tellegen, duality_res,
                    recurrence_res, adiabatic, bundle,
                )
            )
            crowbar = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._crowbar_from_verdict(verdict)
            )
            cert_hash = _compute_hash(
                cert2.certification_hash,
                int(verdict), tellegen, duality_res, recurrence_res,
                adiabatic, meta_res, crowbar.value,
            )
            logger.info(
                "[Φ₃ ✓] Ω₃=%s, T=%.3e, D=%.3e, R=%.3e, Ad=%.3e, "
                "meta=%.3e, crowbar=%s, hash=%s.",
                verdict.name, tellegen, duality_res, recurrence_res,
                adiabatic, meta_res, crowbar.value, cert_hash[:16] + "...",
            )
            return Phase3Certificate(
                phase2=cert2, spectral=spectral,
                tellegen_residual=float(tellegen),
                duality_residual=float(duality_res),
                recurrence_residual=float(recurrence_res),
                dirichlet_residual=float(tellegen),
                adiabatic_action_drift=float(adiabatic),
                meta_residual=float(meta_res),
                heyting_verdict=verdict,
                crowbar_action=crowbar,
                certification_hash=cert_hash,
            )

    # ╔═════════════════════════════════════════════════════════════════════╗
    # ║                                                                     ║
    # ║   FASE 3: AUDITOR ESPECTRAL + TELLEGEN + HEYTING + RSI-3            ║
    # ║                                                                     ║
    # ║   Métodos originales (preservados íntegros):                        ║
    # ║     · _build_laplacian                                              ║
    # ║     · _compute_zero_tolerance                                       ║
    # ║     · _extract_fiedler_value                                        ║
    # ║     · audit_fiedler_connectivity                                    ║
    # ║                                                                     ║
    # ║   Nuevos métodos (v5.0.0):                                          ║
    # ║     · _compute_tellegen_dirichlet_residual                          ║
    # ║     · _compute_adiabatic_action_drift                               ║
    # ║     · _combine_residuals                                            ║
    # ║     · _resolve_heyting                                              ║
    # ║     · _crowbar_from_verdict                                         ║
    # ║     · _rsi_compute_trace                                            ║
    # ║                                                                     ║
    # ║   ULTIMO MÉTODO (morfismo terminal del módulo):                     ║
    # ║     · collapse_to_verdict → AlphaBoundaryVerdict                    ║
    # ║                                                                     ║
    # ╚═════════════════════════════════════════════════════════════════════╝

    class _Phase3_SpectralFiedlerAuditor:
        """FASE 3 — Espectro + Tellegen + Heyting + RSI-3."""

        @staticmethod
        def _build_laplacian(
            boundary_operator: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            L = boundary_operator @ boundary_operator.T
            L_sym = (L + L.T) * 0.5
            asym = float(np.max(np.abs(L - L.T)))
            fro = float(np.linalg.norm(L, 'fro'))
            if fro > 0 and asym > 1e-10 * fro:
                logger.warning(
                    "L₀ con asimetría numérica: %.2e. Simetrización aplicada.",
                    asym,
                )
            return L_sym

        @staticmethod
        def _compute_zero_tolerance(laplacian: NDArray[np.float64]) -> float:
            n = laplacian.shape[0]
            frobenius = float(np.linalg.norm(laplacian, 'fro'))
            return AlphaConstants.laplacian_zero_tolerance(n, frobenius)

        @staticmethod
        def _extract_fiedler_value(
            eigenvalues: NDArray[np.float64],
            zero_tolerance: float,
            n_vertices: int,
        ) -> Tuple[float, bool, float]:
            if n_vertices < 2:
                logger.warning(
                    "n_vertices=%d < 2: Fiedler no definido.", n_vertices,
                )
                return 0.0, False, 0.0
            positive_mask = eigenvalues > zero_tolerance
            if not np.any(positive_mask):
                logger.warning(
                    "Todos los autovalores ~= 0: grafo disconexo."
                )
                return 0.0, False, 0.0
            fiedler = float(np.min(eigenvalues[positive_mask]))
            lam_max = float(eigenvalues[-1])
            gap = max(0.0, lam_max - fiedler)
            is_connected = fiedler > zero_tolerance
            return fiedler, is_connected, gap

        @staticmethod
        def audit_fiedler_connectivity(
            complex_data: SimplicialComplexData,
        ) -> SpectralFiedlerData:
            L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor._build_laplacian(
                complex_data.boundary_operator
            )
            eigenvalues = la.eigvalsh(L0)
            zero_tol = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._compute_zero_tolerance(L0)
            )
            fiedler, is_connected, gap = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._extract_fiedler_value(eigenvalues, zero_tol, complex_data.n_vertices)
            )
            spectral_data = SpectralFiedlerData(
                laplacian=L0, eigenvalues=eigenvalues,
                fiedler_value=fiedler, zero_tolerance=zero_tol,
                is_connected=is_connected, spectral_gap=gap,
            )
            logger.info(
                "Espectro: λ₀=%.4e, λ₂=%.6f, λ_max=%.4f, gap=%.4f, conexo=%s.",
                float(eigenvalues[0]) if eigenvalues.size > 0 else 0.0,
                fiedler,
                float(eigenvalues[-1]) if eigenvalues.size > 0 else 0.0,
                gap, is_connected,
            )
            return spectral_data

        # ─────────────────────────────────────────────────────────────
        # 3.5 NUEVO: Tellegen vía identidad de Dirichlet
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_tellegen_dirichlet_residual(
            complex_data: SimplicialComplexData,
        ) -> float:
            r"""Verificación de la identidad de Dirichlet:

                φᵀ L₀ φ = Σ_{(u,v)∈E} w_uv (φᵤ − φᵥ)²

            con φ = gradiente lineal normalizado. Es la forma estructural del
            balance de Tellegen en ausencia de flujos explícitos.
            """
            n_v = complex_data.n_vertices
            if n_v == 0 or complex_data.n_edges == 0:
                return 0.0
            B = complex_data.boundary_operator
            L0 = B @ B.T
            phi = np.linspace(0.0, 1.0, n_v, dtype=np.float64)
            if n_v > 1:
                phi = phi - phi.mean()
            lhs = float(phi @ L0 @ phi)
            rhs = 0.0
            verts = complex_data.vertices
            v_idx = {v: i for i, v in enumerate(verts)}
            for (u, v, w) in complex_data.edges:
                rhs += float(w) * (phi[v_idx[u]] - phi[v_idx[v]]) ** 2
            denom = max(1.0, abs(lhs) + abs(rhs))
            return float(abs(lhs - rhs) / denom)

        # ─────────────────────────────────────────────────────────────
        # 3.6 NUEVO: Deriva adiabática de la acción
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _compute_adiabatic_action_drift(
            cert2: Phase2Certificate,
        ) -> float:
            """|I_out − I_in| / I_in bajo la aplicación de retorno J."""
            if cert2.phase1.poincare_snapshot is None:
                return 0.0
            I_in = cert2.phase1.poincare_snapshot.action_invariant
            if I_in <= 1e-12:
                return 0.0
            # Reconstruir q, p desde el snapshot
            q = cert2.phase1.poincare_snapshot.frequency_ratio
            # Recalcular p directamente para evitar ambigüedades
            B = cert2.phase1.complex_data.boundary_operator
            L0 = 0.5 * ((B @ B.T) + (B @ B.T).T)
            try:
                eig = la.eigvalsh(L0)
            except Exception:
                return 0.0
            zero_tol = AlphaConstants.laplacian_zero_tolerance(
                cert2.phase1.complex_data.n_vertices,
                float(np.linalg.norm(L0, 'fro')),
            )
            positive = eig[eig > zero_tol]
            if positive.size == 0:
                return 0.0
            q_real = float(np.min(positive))
            lam_max = float(eig[-1])
            p_real = max(0.0, lam_max - q_real)
            J = cert2.return_map.contraction_factor
            if q_real <= 0.0 or p_real <= 0.0 or J <= 1e-12:
                return 0.0
            q_out = J * q_real
            p_out = p_real / J
            I_out = 0.5 * (q_out ** 2 + p_out ** 2)
            return float(abs(I_out - I_in) / I_in)

        # ─────────────────────────────────────────────────────────────
        # 3.7 NUEVO: Combinación de residuales
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _combine_residuals(
            tellegen_res: float, duality_res: float,
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

        # ─────────────────────────────────────────────────────────────
        # 3.8 NUEVO: Resolución del retículo Heyting Ω₃
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _resolve_heyting(
            cert2: Phase2Certificate,
            spectral: SpectralFiedlerData,
            tellegen_res: float, duality_res: float,
            recurrence_res: float, adiabatic_drift: float,
            bundle: AlphaToleranceBundle,
        ) -> HeytingOmega3:
            r"""Colapsa Ω₃ con precedencia estricta.

                1. Homología defectuosa (β₁>0, χ≤0)  ⟹ VETOED
                2. λ₂ < threshold                     ⟹ VETOED
                3. max residual > 1e-2                ⟹ VETOED
                4. max residual > tolerance           ⟹ DEGRADED
                5. dualidad/estabilidad fallan        ⟹ DEGRADED
                6. en otro caso                       ⟹ COHERENT
            """
            homology = cert2.homology
            verdict = HeytingOmega3.COHERENT
            if homology.beta_1 > 0:
                return verdict.join(HeytingOmega3.VETOED)
            if homology.euler_characteristic <= 0:
                return verdict.join(HeytingOmega3.VETOED)
            if (cert2.phase1.complex_data.n_vertices >= 2
                    and spectral.fiedler_value < bundle.min_fiedler_value):
                return verdict.join(HeytingOmega3.VETOED)
            residuals = [tellegen_res, duality_res, recurrence_res,
                         adiabatic_drift]
            finite = [r for r in residuals if math.isfinite(r)]
            max_res = max(finite) if finite else float("inf")
            if not math.isfinite(max_res) or max_res > 1e-2:
                verdict = verdict.join(HeytingOmega3.VETOED)
            elif max_res > bundle.duality_tol:
                verdict = verdict.join(HeytingOmega3.DEGRADED)
            if not cert2.duality.is_poincare_dual:
                verdict = verdict.join(HeytingOmega3.DEGRADED)
            if not cert2.return_map.is_stable:
                verdict = verdict.join(HeytingOmega3.DEGRADED)
            return verdict

        # ─────────────────────────────────────────────────────────────
        # 3.9 NUEVO: Crowbar desde veredicto Heyting
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _crowbar_from_verdict(verdict: HeytingOmega3) -> CrowbarAction:
            if verdict == HeytingOmega3.VETOED:
                return CrowbarAction.HARD_SHORT
            if verdict == HeytingOmega3.DEGRADED:
                return CrowbarAction.WATCHDOG_PULSE
            return CrowbarAction.NONE

        # ─────────────────────────────────────────────────────────────
        # 3.10 NUEVO: Traza RSI
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def _rsi_compute_trace(
            level: int, bundle: AlphaToleranceBundle, residual: float,
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
                    convergence_rate=rate,
                    level3_active=False,
                    aitken_signal=residual,
                )
            if len(traces_prev) >= 2:
                active = True
                t_m2, t_m1 = traces_prev[-2], traces_prev[-1]
                ait = _aitken_delta_squared(t_m2.residual, t_m1.residual, residual)
                factor = _adaptive_refinement_factor(
                    current_residual=residual,
                    target_residual=_RSI_TARGET_RESIDUAL,
                    convergence_rate=rate,
                    level3_active=True,
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

        # ─────────────────────────────────────────────────────────────
        # ★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★
        # ─────────────────────────────────────────────────────────────

        @staticmethod
        def collapse_to_verdict(
            cert3: Phase3Certificate,
            vetoed: bool = False,
            veto_reason: Optional[str] = None,
            veto_class: Optional[Type[AlphaBoundaryError]] = None,
        ) -> AlphaBoundaryVerdict:
            r"""★ MORFISMO TERMINAL DEL MÓDULO ★

            Cadena: Phase3Certificate ──▶ AlphaBoundaryVerdict
            Preserva 3 vetos duros (is_viable=False + raise) y añade
            agregación Heyting Ω₃ como señal continua.
            """
            if not isinstance(cert3, Phase3Certificate):
                raise PoincareSectionError(
                    "Fase 3 → verdict: se requiere Phase3Certificate."
                )
            p1 = cert3.phase2.phase1
            homology = cert3.phase2.homology
            spectral = cert3.spectral
            is_viable = not vetoed
            return AlphaBoundaryVerdict(
                is_viable=is_viable,
                homology=homology,
                spectral=spectral,
                veto_reason=veto_reason,
                veto_class=veto_class,
                poincare_snapshot=p1.poincare_snapshot,
                lindstedt_expansion=p1.lindstedt_expansion,
                energy_return_map=cert3.phase2.return_map,
                poincare_duality=cert3.phase2.duality,
                spectral_recurrence=cert3.phase2.spectral_recurrence,
                heyting_verdict=cert3.heyting_verdict,
                crowbar_action=cert3.crowbar_action,
                tellegen_residual=cert3.tellegen_residual,
                duality_residual=cert3.duality_residual,
                recurrence_residual=cert3.recurrence_residual,
                dirichlet_residual=cert3.dirichlet_residual,
                adiabatic_action_drift=cert3.adiabatic_action_drift,
                meta_residual=cert3.meta_residual,
                rsi_result=None,
            )

    # ╔═════════════════════════════════════════════════════════════════════╗
    # ║   CONSTRUCTOR Y MÉTODOS PÚBLICOS DEL ORQUESTADOR                    ║
    # ╚═════════════════════════════════════════════════════════════════════╝

    def __init__(
        self,
        fiedler_threshold: Optional[float] = None,
    ) -> None:
        if fiedler_threshold is not None:
            if not isinstance(fiedler_threshold, (int, float)):
                raise PreconditionError(
                    f"fiedler_threshold debe ser numérico, "
                    f"recibido {type(fiedler_threshold).__name__}."
                )
            if float(fiedler_threshold) <= 0.0:
                raise PreconditionError(
                    f"fiedler_threshold debe ser > 0, "
                    f"recibido {fiedler_threshold}."
                )
            self._fiedler_threshold = float(fiedler_threshold)
        else:
            self._fiedler_threshold = AlphaConstants.MIN_FIEDLER_VALUE
        self._last_verdict: Optional[AlphaBoundaryVerdict] = None
        logger.info(
            "AlphaBoundaryAgent v%s inicializado. Umbral de Fiedler: %.4f.",
            __version__, self._fiedler_threshold,
        )

    @staticmethod
    def _validate_orchestrator_input(nodes: Any, flows: Any) -> None:
        if not isinstance(nodes, list):
            raise PreconditionError(
                f"nodes debe ser list, recibido {type(nodes).__name__}."
            )
        if not isinstance(flows, list):
            raise PreconditionError(
                f"flows debe ser list, recibido {type(flows).__name__}."
            )

    # ─────────────────────────────────────────────────────────────────────
    # 4.2 evaluate_business_canvas (PRESERVADO ÍNTEGRAMENTE)
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_business_canvas(
        self,
        nodes: List[str],
        flows: List[Tuple[str, str, float]],
    ) -> AlphaBoundaryVerdict:
        r"""Ejecuta el colapso topológico determinista (API v4.0.0 preservada).

        Los 3 vetos (β₁>0, χ≤0, λ₂<threshold) disparan idéntico a v4.0.0.
        El veredicto retornado es el mismo `AlphaBoundaryVerdict` con campos
        v5.0.0 poblados si es viable (Poincaré, Lindstedt, RSI omitidos).
        """
        self._validate_orchestrator_input(nodes, flows)
        homology: Optional[HomologicalInvariants] = None
        spectral: Optional[SpectralFiedlerData] = None

        try:
            complex_data = (
                self._Phase1_CanvasSimplicialFibrator
                .project_canvas_to_simplicial_complex(nodes, flows)
            )
            homology = (
                self._Phase2_HomologicalBettiAuditor
                .audit_betti_invariants(complex_data)
            )
            if homology.beta_1 > 0:
                veto_msg = (
                    f"Veto de Canibalización [β₁={homology.beta_1}]: "
                    f"{homology.beta_1} ciclo(s) independiente(s)."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable=False, homology=homology, spectral=None,
                    veto_reason=veto_msg, veto_class=ToxicCycleVetoError,
                )
                logger.critical("COLAPSO ESTRATO α: %s", veto_msg)
                raise ToxicCycleVetoError(veto_msg)
            if homology.euler_characteristic <= 0:
                veto_msg = (
                    f"Veto de Degeneración [χ={homology.euler_characteristic}]: "
                    f"χ≤0 con |V|={complex_data.n_vertices}, "
                    f"|E|={complex_data.n_edges}."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable=False, homology=homology, spectral=None,
                    veto_reason=veto_msg,
                    veto_class=EulerPoincareDegeneracyError,
                )
                logger.critical("COLAPSO ESTRATO α: %s", veto_msg)
                raise EulerPoincareDegeneracyError(veto_msg)
            spectral = (
                self._Phase3_SpectralFiedlerAuditor
                .audit_fiedler_connectivity(complex_data)
            )
            if (complex_data.n_vertices >= 2
                    and spectral.fiedler_value < self._fiedler_threshold):
                veto_msg = (
                    f"Veto de Fragilidad Espectral "
                    f"[λ₂={spectral.fiedler_value:.6f} < "
                    f"θ={self._fiedler_threshold}]: posible SPOF."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable=False, homology=homology, spectral=spectral,
                    veto_reason=veto_msg, veto_class=SpectralFragilityError,
                )
                logger.critical("COLAPSO ESTRATO α: %s", veto_msg)
                raise SpectralFragilityError(veto_msg)
            logger.info(
                "BMC superó auditoría: β₀=%d, β₁=%d, χ=%d, λ₂=%.6f.",
                homology.beta_0, homology.beta_1,
                homology.euler_characteristic,
                spectral.fiedler_value if spectral else 0.0,
            )
            verdict = AlphaBoundaryVerdict(
                is_viable=True, homology=homology, spectral=spectral,
                veto_reason=None, veto_class=None,
            )
            self._last_verdict = verdict
            return verdict
        except AlphaBoundaryError:
            raise

    # ─────────────────────────────────────────────────────────────────────
    # 4.3 NUEVO: evaluate_with_rsi3 (pipeline enriquecido + meta-controlador)
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_with_rsi3(
        self,
        nodes: List[str],
        flows: List[Tuple[str, str, float]],
        max_depth: int = _RSI_MAX_DEPTH,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> AlphaBoundaryVerdict:
        r"""Pipeline enriquecido con Poincaré + Lindstedt + RSI-3.

        Los 3 vetos duros disparan idéntico a `evaluate_business_canvas`.
        Si sobreviven los 3 vetos, se devuelve un veredicto con:
          • poincare_snapshot, lindstedt_expansion, return_map, dualidad,
          • residuals (tellegen, duality, recurrence, dirichlet, adiabatic),
          • heyting_verdict, crowbar_action,
          • rsi_result con la traza completa de RSI-3.
        """
        self._validate_orchestrator_input(nodes, flows)

        def run_pipeline(bundle: AlphaToleranceBundle) -> Tuple[
            Phase3Certificate, AlphaBoundaryVerdict
        ]:
            # Φ₁ → Phase1Certificate
            cert1 = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                .certify_phase1_composite(nodes, flows, bundle)
            )
            # Φ₂ (vía bridge) → Phase2Certificate
            cert2 = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator
                ._phase1_to_phase2_bridge(cert1)
            )
            # Verificación de vetos duros
            if cert2.homology.beta_1 > 0:
                raise ToxicCycleVetoError(
                    f"Veto β₁>0: {cert2.homology.beta_1}."
                )
            if cert2.homology.euler_characteristic <= 0:
                raise EulerPoincareDegeneracyError(
                    f"Veto χ≤0: {cert2.homology.euler_characteristic}."
                )
            # Φ₃ (vía bridge) → Phase3Certificate
            cert3 = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._phase2_to_phase3_bridge(cert2)
            )
            # Veto 3 (espectral)
            if (cert2.phase1.complex_data.n_vertices >= 2
                    and cert3.spectral.fiedler_value < self._fiedler_threshold):
                raise SpectralFragilityError(
                    f"Veto λ₂<θ: {cert3.spectral.fiedler_value:.6f}."
                )
            # Emitir veredicto
            verdict = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                .collapse_to_verdict(cert3)
            )
            return cert3, verdict

        try:
            # Nivel 1: bundle default (compat. v4.0.0)
            initial_bundle = AlphaToleranceBundle(
                scale=1.0, n_lindstedt_terms=2,
            )
            traces: List[RSITrace] = []
            best_verdict: Optional[AlphaBoundaryVerdict] = None
            best_meta: float = float("inf")
            bundle = initial_bundle

            for level in range(1, max_depth + 1):
                cert3, verdict = run_pipeline(bundle)
                meta_res = float(cert3.meta_residual)
                trace = (
                    AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                    ._rsi_compute_trace(
                        level=level, bundle=bundle, residual=meta_res,
                        traces_prev=tuple(traces),
                    )
                )
                traces.append(trace)
                if meta_res < best_meta:
                    best_meta = meta_res
                    best_verdict = verdict
                if meta_res <= target_residual:
                    break
                factor = trace.adaptation_factor
                if not math.isfinite(factor) or factor <= 0.0:
                    factor = 0.1
                new_scale = bundle.scale * factor
                if new_scale >= bundle.scale:
                    new_scale = bundle.scale * 0.5
                new_terms = min(_RSI_TERMS_CAP, bundle.n_lindstedt_terms + 2)
                bundle = AlphaToleranceBundle(
                    scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, new_scale))),
                    n_lindstedt_terms=new_terms,
                )

            last = traces[-1]
            rsi_result = SelfImprovementResult(
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
            if best_verdict is None:
                raise RecursiveSelfImprovementError("RSI-3 sin veredicto.")
            # Adjuntar rsi_result al veredicto (frozen: reconstruct)
            final_verdict = AlphaBoundaryVerdict(
                is_viable=best_verdict.is_viable,
                homology=best_verdict.homology,
                spectral=best_verdict.spectral,
                veto_reason=best_verdict.veto_reason,
                veto_class=best_verdict.veto_class,
                poincare_snapshot=best_verdict.poincare_snapshot,
                lindstedt_expansion=best_verdict.lindstedt_expansion,
                energy_return_map=best_verdict.energy_return_map,
                poincare_duality=best_verdict.poincare_duality,
                spectral_recurrence=best_verdict.spectral_recurrence,
                heyting_verdict=best_verdict.heyting_verdict,
                crowbar_action=best_verdict.crowbar_action,
                tellegen_residual=best_verdict.tellegen_residual,
                duality_residual=best_verdict.duality_residual,
                recurrence_residual=best_verdict.recurrence_residual,
                dirichlet_residual=best_verdict.dirichlet_residual,
                adiabatic_action_drift=best_verdict.adiabatic_action_drift,
                meta_residual=best_verdict.meta_residual,
                rsi_result=rsi_result,
            )
            self._last_verdict = final_verdict
            logger.info(
                "[RSI-3 ✓] converged=%s, final_res=%.3e, "
                "final_scale=%.4f, n_terms=%d, level3=%s.",
                rsi_result.converged, rsi_result.final_residual,
                rsi_result.final_scale, rsi_result.final_n_terms,
                rsi_result.level3_activated,
            )
            return final_verdict

        except (ToxicCycleVetoError, EulerPoincareDegeneracyError,
                SpectralFragilityError) as exc:
            veto_class = type(exc)
            # Construir veredicto de fallo mínimo (compat. con _last_verdict)
            homology_dummy = HomologicalInvariants(
                rank=0, beta_0=1, beta_1=0, euler_characteristic=1,
                svd_singular_values=np.array([], dtype=np.float64),
                rank_threshold=AlphaConstants.RANK_TOLERANCE,
            )
            self._last_verdict = AlphaBoundaryVerdict(
                is_viable=False, homology=homology_dummy, spectral=None,
                veto_reason=str(exc), veto_class=veto_class,
            )
            raise

    # ─────────────────────────────────────────────────────────────────────
    # 4.4 __call__ (protocolo MIC, PRESERVADO)
    # ─────────────────────────────────────────────────────────────────────

    def __call__(self, state: CategoricalState) -> CategoricalState:
        payload = state.payload
        nodes = payload.get("bmc_nodes", [])
        flows = payload.get("bmc_flows", [])
        if not nodes or not flows:
            raise PreconditionError(
                "El CategoricalState no contiene una topología BMC válida. "
                "Se requieren 'bmc_nodes' y 'bmc_flows' en state.payload."
            )
        verdict = self.evaluate_business_canvas(nodes, flows)
        new_payload = dict(payload)
        new_payload["alpha_verdict"] = verdict
        return CategoricalState(
            payload=new_payload,
            metadata=state.metadata,
            stratum=Stratum.ALPHA,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 4.5 Propiedades de solo lectura
    # ─────────────────────────────────────────────────────────────────────

    @property
    def last_verdict(self) -> Optional[AlphaBoundaryVerdict]:
        """Último veredicto emitido (incluye vetos, util para post-mortem)."""
        return self._last_verdict

    @property
    def fiedler_threshold(self) -> float:
        """Umbral de conectividad algebraica en uso."""
        return self._fiedler_threshold


# ═════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constantes
    "AlphaConstants",
    # Excepciones
    "AlphaBoundaryError", "PreconditionError", "ToxicCycleVetoError",
    "EulerPoincareDegeneracyError", "SpectralFragilityError",
    "PoincareSectionError", "RecursiveSelfImprovementError",
    # Retículo
    "HeytingOmega3", "CrowbarAction",
    # DTOs topológicos (preservados)
    "SimplicialComplexData", "HomologicalInvariants", "SpectralFiedlerData",
    "AlphaBoundaryVerdict",
    # DTOs nuevos (Poincaré + RSI-3)
    "PoincareSpectralSnapshot", "LindstedtTraceExpansion",
    "EnergyReturnMapSnapshot", "PoincareDualitySnapshot",
    "SpectralRecurrenceSnapshot", "AlphaToleranceBundle",
    "RSITrace", "SelfImprovementResult",
    # Certificados compuestos
    "Phase1Certificate", "Phase2Certificate", "Phase3Certificate",
    # Agente principal
    "AlphaBoundaryAgent",
]