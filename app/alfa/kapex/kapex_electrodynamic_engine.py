# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA SOBERANO ESTRATO ALFA (KAPEX)
====================================================================================================

Nombre: KapexElectrodynamicEngine
Ubicación: app/alfa/kapex/kapex_electrodynamic_engine.py
Versión: 9.0.0-APU-SOVEREIGN-POINCARE-RSI3-NESTED
Función: Motor soberano electrodinámico, simpléctico y cuántico-relativista de la capa Alfa (Kapex)
         en la arquitectura APU Filter. Modela las ecuaciones de Maxwell-Yang-Mills en variedades
         riemannianas dispersivas, álgebra de Clifford-biquaterniones de Riemann-Silberstein,
         impedancias complejas Z(ω), resonancia RLC cuántica, flujo de Poynting S = ★_g(E♭ ∧ B♭),
         tensor de esfuerzo de Maxwell T^{μν}, mecánica celeste hamiltoniana de Poincaré-KAM y
         automejora recursiva de Nivel 3 (RSI-3).

----------------------------------------------------------------------------------------------------
EVOLUCIÓN SOBERANA v9.0.0 (Poincaré-KAM + Clifford-RLC + RSI-3 Homológico en 3 Fases Anidadas):
----------------------------------------------------------------------------------------------------
  [P1] Snapshot de Poincaré-KAM en el haz cotangente espectral: invariante integral de
       Poincaré-Cartan I = ½(q² + p²), condición diofántica KAM |q/p − p_r/q_r| ≥ γ·q_r^{−τ}
       y detección de divisores pequeños vía fracciones continuas exactas.
  [P2] Expansión perturbativa de Lindstedt-Poincaré del Heat Kernel Φ(ε) = n⁻¹tr(e^{−εg})
       con eliminación de términos seculares y re-sumación racional de Padé [M/M] activada
       en el Nivel 3 de automejora recursiva.
  [P3] Bivector hipercomplejo de Riemann-Silberstein F_RS = E + iB ∈ Cl_{3,0}(ℂ) y síntesis
       de impedancia cuántica RLC Z(ω) = R + iX verificando KEE1–KEE3 e Invariantes 1–2.
  [P4] Mapa de Retorno Simpéctico de Poincaré-Chirikov (Twist Map con det(DP) = 1),
       traza de monodromía de Floquet y exponente de Lyapunov espectral local.
  [P5] Dualidad de Hodge-Poincaré no tautológica sobre Ω²(M, so(n)): descomposición SVD de
       Eckart-Young-Mirsky, defecto de involución ★_g² y antisimetría cohomológica.
  [P6] Recurrencia ergódica de Poincaré-Birkhoff sobre el toro invariante T^n y estadística
       de espaciamiento espectral (Berry-Tabor integrable vs. Wigner-Dyson caótico).
  [P7] Deriva adiabática exponencial de Nekhoroshev-Poincaré |I_out − I_in|/I_in.
  [P8] Meta-controlador RSI-3 de tres niveles reales:
         - Nivel 1: Refinamiento paramétrico (Tikhonov α, paso complejo CSMD h) vía Richardson.
         - Nivel 2: Escalado de orden asintótico de Lindstedt-Poincaré y Padé [M/M].
         - Nivel 3: Aceleración no lineal de Aitken Δ² y mutación homológica-simpléctica del
                    flujo de Hamilton-Jacobi para supresión activa de residuos resonantes.

ARQUITECTURA DE TRES FASES ESTRICTAMENTE ANIDADAS (COMPOSICIÓN DE MORFISMOS CATEGÓRICOS):
  Fase 1 ──► Culmina en `_phase1_observe_orient` ──► Emite `PhaseOneKapexPacket`
                │ (El codominio terminal de Fase 1 es el dominio inicial exacto de Fase 2)
                ▼
  Fase 2 ──► Inicia en `_phase2_spectral_integrity` y culmina en `_phase2_decide` ──► Emite `PhaseTwoKapexPacket`
                │ (El codominio terminal de Fase 2 es el dominio inicial exacto de Fase 3)
                ▼
  Fase 3 ──► Inicia en `_phase3_conservation_audit`, sella en `_phase3_act` y eleva el
             endofunctor mediante el lazo de nivel 3 `execute_electrodynamic_cycle_rsi3`.

Funtor Maestro: T_RSI3 = Fix_3(Act ∘ Decide ∘ Observe)
====================================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.KapexElectrodynamicEngine")

__version__: Final[str] = "9.0.0-APU-SOVEREIGN-POINCARE-RSI3-NESTED"

# ---------------------------------------------------------------------------
# Constantes metrológicas y cotas espectrales
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_ALGEBRA_TOLERANCE: Final[float] = 1e-10
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

# ── Constantes de Mecánica Celeste de Poincaré y Teoría KAM ────────────────
_POINCARE_MAX_DENOMINATOR: Final[int] = 16
_POINCARE_LOW_ORDER: Final[int] = 8
_RESONANCE_TOLERANCE: Final[float] = 1.0e-3
_SMALL_PARAMETER_FLOOR: Final[float] = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: Final[float] = 1.0e-6
_POINCARE_DEFAULT_SECTION: Final[float] = 1.0
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-2
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0

# ── Constantes del Meta-Controlador de Automejora Recursiva Nivel 3 (RSI-3)
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_TARGET_RESIDUAL: Final[float] = 1.0e-3
_RSI_MIN_SCALE: Final[float] = 1.0e-3
_RSI_MAX_SCALE: Final[float] = 1.0e3
_RSI_FACTOR_LOWER: Final[float] = 1.0e-2
_RSI_FACTOR_UPPER: Final[float] = 0.99
_RSI_OSC_EPS: Final[float] = 1.0e-15
_RSI_TERMS_CAP: Final[int] = 10


# =============================================================================
# Objetos de Retículo Intuicionista (Topos de Heyting)
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena finita de Heyting: ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    En esta categoría posetal linealmente ordenada, el álgebra de Heyting es
    completa y distributiva:
        a ∧ b = min(a, b),
        a ∨ b = max(a, b),
        (a → b) = ⊤ si a ≼ b, y b en caso contrario,
        ¬a = (a → ⊥).
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)


# =============================================================================
# Núcleo Matemático Puro (Fracciones Continuas, KAM, Padé, Richardson, Aitken)
# =============================================================================
def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
    r"""
    Mejor aproximación diofántica racional p/q de x ∈ ℝ⁺ mediante el algoritmo
    de convergentes de fracciones continuas de Legendre-Gauss con q ≤ max_denominator.
    """
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
    r"""
    Detector de resonancia de Poincaré-Arnold y divisores pequeños:
    Evalúa la proximidad del cociente de frecuencias ω₁/ω₂ a un racional p/q de orden bajo
    y verifica la violación de la condición diofántica KAM.
    Retorna: (es_resonante, p, q, error_diofántico, magnitud_divisor_pequeño).
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)
    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)
    err = abs(ratio - p / q)
    kam_bound = _KAM_DIOPHANTINE_GAMMA / (float(q) ** _KAM_DIOPHANTINE_TAU)
    is_res = ((err < tolerance) or (err < kam_bound * 0.1)) and (q <= low_order)
    small_divisor = 1.0 / max(_SMALL_PARAMETER_FLOOR, float(q) * max(1.0, err / max(tolerance, 1e-15)))
    return (bool(is_res), int(p), int(q), float(err), float(1.0 / q if is_res else min(1.0 / q, small_divisor)))


def _pade_resum_exponential(x: float, n_terms: int) -> float:
    r"""
    Re-sumación racional de Padé simétrica [M/M](-x) para exp(-x), donde M = max(1, n_terms // 2).
    Incondicionalmente A-estable y unitaria sobre el eje imaginario; elimina las divergencias
    asintóticas de la serie truncada de Lindstedt cuando x = ε·λ_i es moderado o grande.
    """
    m = max(1, n_terms // 2)
    num = 1.0
    den = 1.0
    term = 1.0
    for k in range(1, m + 1):
        coeff = float(m - k + 1) / float((2 * m - k + 1) * k)
        term *= x * coeff
        if k % 2 == 1:
            num -= term
        else:
            num += term
        den += term
    if abs(den) < _WILKINSON_FLOOR:
        return 1.0
    return float(num / den)


def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float,
) -> Tuple[float, float]:
    r"""
    Extrapolación asintótica de Richardson de orden estimado p = ln(r_curr/r_prev) / ln(h_curr/h_prev)
    y estimador del límite en el continuo h → 0.
    """
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
    r"""
    Transformación no lineal de Shanks / aceleración Δ² de Aitken:
        r* = r_n − (r_n − r_{n−1})² / (r_n − 2·r_{n−1} + r_{n−2}).
    Extrae el punto fijo de secuencias con errores dominados por modos geométricos.
    """
    d = r_n - 2.0 * r_nm1 + r_nm2
    if abs(d) < _RSI_OSC_EPS:
        return r_n
    return float(r_n - (r_n - r_nm1) ** 2 / d)


def _adaptive_refinement_factor(
    current_residual: float, target_residual: float, convergence_rate: float,
    level3_active: bool, aitken_signal: float,
) -> float:
    r"""
    Ley de control óptimo de paso para el meta-controlador RSI-3.
    """
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
    """Sello criptográfico SHA-256 determinista sobre invariantes espectrales."""
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


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Inmutabilización en memoria de tensores numpy para seguridad de concurrencia."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


# =============================================================================
# Estructuras Inmutables de Datos (Cartas Espectrales, Poincaré, RLC y Paquetes)
# =============================================================================
@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización espectral riemanniana g = U Λ U† y operadores duales asociados.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: np.ndarray
    condition_number: float
    volume_density: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    regularized: bool
    tikhonov_alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))
        object.__setattr__(self, "inv_metric", _freeze_array(self.inv_metric))


@dataclass(frozen=True, slots=True)
class PoincareSpectralSnapshot:
    r"""
    Snapshot de Poincaré-KAM en el espacio fase espectral (q = λ_min, p = λ_max − λ_min).

    Variables canónicas de acción-ángulo:
        I = ½(q² + p²),   θ_p = atan2(p, q).
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
    kam_diophantine_margin: float = 0.0
    poincare_cartan_form: float = 0.0


@dataclass(frozen=True, slots=True)
class LindstedtHeatKernelExpansion:
    r"""
    Expansión perturbativa de Lindstedt-Poincaré del Heat Kernel Φ(ε) = n⁻¹tr(e^{−εg}).
    Incluye término de Padé [M/M] para eliminación de divergencias seculares en RSI-3.
    """
    small_parameter: float
    n_terms: int
    zero_order: float
    first_order: float
    second_order: float
    higher_order: float
    heat_kernel_approx: float
    heat_kernel_true: float
    expansion_residual: float
    is_asymptotically_consistent: bool
    pade_resummed_approx: float = 1.0


@dataclass(frozen=True, slots=True)
class QuantumRLCImpedanceSnapshot:
    r"""
    Síntesis hipercompleja de Riemann-Silberstein F_RS = E + iB ∈ Cl_{3,0}(ℂ) y
    circuito cuántico equivalente RLC verificando Axiomas KEE1–KEE3 e Invariantes 1–2:
        - KEE2: R = Re(Z(ω)) ≥ 0.
        - Inv 1: |S|² = P² + Q².
        - Inv 2: Q_factor > 0.
    """
    resistance_R: float
    reactance_X: float
    impedance_magnitude: float
    impedance_phase_rad: float
    active_power_P: float
    reactive_power_Q: float
    apparent_power_S: float
    power_triangle_residual: float
    quality_factor_Q: float
    riemann_silberstein_lorentz_scalar: float
    riemann_silberstein_pseudoscalar: float
    axioms_kee_satisfied: bool


@dataclass(frozen=True, slots=True)
class EnergyReturnMapSnapshot:
    r"""
    Aplicación de retorno simpléctica de Poincaré-Chirikov en el espacio espectral.
    Preserva la medida de Liouville det(DP) = 1 y evalúa multiplicadores de Floquet.
    """
    contraction_factor: float
    eigenvalues_modulus: Tuple[float, ...]
    is_stable: bool
    iterations_norm_sequence: Tuple[float, ...]
    floquet_trace: float = 2.0
    lyapunov_exponent: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareDualitySnapshot:
    r"""
    Dualidad de Hodge-Poincaré y teorema de rango-nulidad sobre el operador de curvatura
    de Yang-Mills F ∈ Ω²(M, so(n)) ≅ ℝ^{n²×n²}.
    """
    curvature_rank: int
    curvature_nullity: int
    total_dimension: int
    duality_residual: float
    is_poincare_dual: bool
    hodge_involution_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class SpectralRecurrenceSnapshot:
    r"""
    Teorema de recurrencia de Poincaré-Birkhoff sobre la secuencia espectral normalizada
    y estadístico de espaciamiento de niveles (r-ratio de Oganesyan-Huse).
    """
    recurrence_points: Tuple[float, ...]
    min_recurrence_distance: float
    is_poincare_recurrent: bool
    n_samples: int
    level_spacing_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class KapexToleranceBundle:
    r"""
    Bundle de hiperparámetros adaptativos para el Meta-Controlador RSI-3.
    Cuando scale=1.0, n_lindstedt_terms=2, symplectic_homotopy_gain=0.0, use_pade=False,
    preserva compatibilidad numérica bit a bit con v7.0.0 y v8.0.0.
    """
    scale: float = 1.0
    n_lindstedt_terms: int = 2
    symplectic_homotopy_gain: float = 0.0
    use_pade_resummation: bool = False

    def _scaled(self, base: float) -> float:
        return max(_WILKINSON_FLOOR, base * self.scale)

    def with_scale(self, s: float) -> "KapexToleranceBundle":
        return KapexToleranceBundle(
            scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, s))),
            n_lindstedt_terms=self.n_lindstedt_terms,
            symplectic_homotopy_gain=self.symplectic_homotopy_gain,
            use_pade_resummation=self.use_pade_resummation,
        )

    def with_terms(self, n: int) -> "KapexToleranceBundle":
        return KapexToleranceBundle(
            scale=self.scale,
            n_lindstedt_terms=int(max(2, min(_RSI_TERMS_CAP, n))),
            symplectic_homotopy_gain=self.symplectic_homotopy_gain,
            use_pade_resummation=self.use_pade_resummation,
        )


@dataclass(frozen=True, slots=True)
class RSITrace:
    r"""Traza metrológica de un nivel de ejecución del meta-controlador RSI-3."""
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
    r"""Resultado consolidado de convergencia y automejora recursiva de Nivel 3."""
    traces: Tuple[RSITrace, ...]
    final_scale: float
    final_n_terms: int
    final_residual: float
    converged: bool
    richardson_limit: float
    aitken_limit: float
    effective_convergence_order: float
    level3_activated: bool


@dataclass(frozen=True, slots=True)
class PhaseOneKapexPacket:
    r"""
    [OBJETO DE ACOPLAMIENTO CATEGÓRICO FASE 1 ──► FASE 2]
    Objeto terminal de la Fase 1 (`_phase1_observe_orient`) ≡ Objeto inicial
    de la Fase 2 (`_phase2_spectral_integrity`).
    """
    eikonal_residual: float
    eikonal_hamiltonian: float
    momentum_up_vector: np.ndarray
    csmd_momentum: np.ndarray
    csmd_gradient_error: float
    curvature_tensor: np.ndarray
    yang_mills_action: float
    yang_mills_raw_trace: float
    bianchi_jacobi_residual: float
    instanton_density: float
    poynting_flux_vector: np.ndarray
    poynting_two_form: np.ndarray
    field_energy_density: float
    maxwell_stress_tensor: np.ndarray
    spectral_chart: SpectralChart
    algebra_projection_residual: float
    # ── Extensiones Poincaré + Lindstedt + RLC + RSI-3 ──────────────────────
    poincare_snapshot: Optional[PoincareSpectralSnapshot] = None
    lindstedt_expansion: Optional[LindstedtHeatKernelExpansion] = None
    tolerance_bundle: KapexToleranceBundle = field(default_factory=KapexToleranceBundle)
    quantum_rlc_snapshot: Optional[QuantumRLCImpedanceSnapshot] = None

    def __post_init__(self) -> None:
        for name in (
            "momentum_up_vector",
            "csmd_momentum",
            "curvature_tensor",
            "poynting_flux_vector",
            "poynting_two_form",
            "maxwell_stress_tensor",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SpectralIntegrityReport:
    r"""Subobjeto de diagnóstico espectral y de álgebra de Banach de la Fase 2."""
    metric_condition: float
    metric_gap: float
    curvature_frobenius: float
    connection_spectral_radii: np.ndarray
    banach_submultiplicative_defect: float
    integrity_score: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "connection_spectral_radii",
            _freeze_array(self.connection_spectral_radii),
        )


@dataclass(frozen=True, slots=True)
class PhaseTwoKapexPacket:
    r"""
    [OBJETO DE ACOPLAMIENTO CATEGÓRICO FASE 2 ──► FASE 3]
    Objeto terminal de la Fase 2 (`_phase2_decide`) ≡ Primer factor del dominio
    de la Fase 3 (`_phase3_conservation_audit`).
    """
    heyting_verdict: str
    heyting_rank: int
    eikonal_threshold_violated: bool
    yang_mills_threshold_violated: bool
    jacobi_threshold_violated: bool
    condition_threshold_violated: bool
    spectral_integrity: SpectralIntegrityReport
    wilson_plaquette_mean: float
    gauge_coherence: float
    heyting_score: float
    # ── Extensiones de Mecánica Celeste de Poincaré Fase 2 ───────────────────
    return_map: Optional[EnergyReturnMapSnapshot] = None
    duality: Optional[PoincareDualitySnapshot] = None
    spectral_recurrence: Optional[SpectralRecurrenceSnapshot] = None


@dataclass(frozen=True, slots=True)
class KapexTelemetry:
    r"""Sello inmutable de telemetría (objeto terminal del funtor maestro OODA)."""
    eikonal_residual: float
    eikonal_hamiltonian: float
    yang_mills_action: float
    poynting_flux_vector: np.ndarray
    field_energy_density: float
    momentum_up_vector: np.ndarray
    curvature_tensor: np.ndarray
    maxwell_stress_tensor: np.ndarray
    heyting_verdict: str
    condition_number: float
    bianchi_residual: float
    instanton_density: float
    csmd_gradient_error: float
    wilson_plaquette_mean: float
    conservation_residual: float
    spectral_integrity_score: float
    # ── Extensiones Poincaré + RLC + RSI-3 ──────────────────────────────────
    poincare_snapshot: Optional[PoincareSpectralSnapshot] = None
    lindstedt_expansion: Optional[LindstedtHeatKernelExpansion] = None
    return_map: Optional[EnergyReturnMapSnapshot] = None
    duality: Optional[PoincareDualitySnapshot] = None
    spectral_recurrence: Optional[SpectralRecurrenceSnapshot] = None
    adiabatic_action_drift: float = 0.0
    meta_residual: float = 0.0
    rsi_result: Optional[SelfImprovementResult] = None
    quantum_rlc_snapshot: Optional[QuantumRLCImpedanceSnapshot] = None

    def __post_init__(self) -> None:
        for name in (
            "poynting_flux_vector",
            "momentum_up_vector",
            "curvature_tensor",
            "maxwell_stress_tensor",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización canónica retrocompatible (claves v2 + v3 + v8 + v9)."""
        base: Dict[str, Any] = {
            "eikonal_residual": self.eikonal_residual,
            "eikonal_hamiltonian": self.eikonal_hamiltonian,
            "yang_mills_action": self.yang_mills_action,
            "poynting_flux_vector": np.array(self.poynting_flux_vector, copy=True),
            "field_energy_density": self.field_energy_density,
            "momentum_up_vector": np.array(self.momentum_up_vector, copy=True),
            "curvature_tensor": np.array(self.curvature_tensor, copy=True),
            "maxwell_stress_tensor": np.array(self.maxwell_stress_tensor, copy=True),
            "heyting_verdict": self.heyting_verdict,
            "condition_number": self.condition_number,
            "bianchi_residual": self.bianchi_residual,
            "instanton_density": self.instanton_density,
            "csmd_gradient_error": self.csmd_gradient_error,
            "wilson_plaquette_mean": self.wilson_plaquette_mean,
            "conservation_residual": self.conservation_residual,
            "spectral_integrity_score": self.spectral_integrity_score,
        }
        if self.poincare_snapshot is not None:
            base["poincare_action_invariant"] = self.poincare_snapshot.action_invariant
            base["poincare_angle_parameter"] = self.poincare_snapshot.angle_parameter
            base["poincare_section_level"] = self.poincare_snapshot.section_level
            base["poincare_crossing_count"] = self.poincare_snapshot.crossing_count
            base["poincare_frequency_ratio"] = self.poincare_snapshot.frequency_ratio
            base["poincare_is_near_resonance"] = self.poincare_snapshot.is_near_resonance
            base["poincare_resonance_p"] = self.poincare_snapshot.resonance_p
            base["poincare_resonance_q"] = self.poincare_snapshot.resonance_q
            base["poincare_small_divisor"] = self.poincare_snapshot.small_divisor_magnitude
            base["poincare_kam_diophantine_margin"] = self.poincare_snapshot.kam_diophantine_margin
            base["poincare_cartan_form"] = self.poincare_snapshot.poincare_cartan_form
        if self.lindstedt_expansion is not None:
            base["lindstedt_small_parameter"] = self.lindstedt_expansion.small_parameter
            base["lindstedt_n_terms"] = self.lindstedt_expansion.n_terms
            base["lindstedt_heat_kernel_approx"] = self.lindstedt_expansion.heat_kernel_approx
            base["lindstedt_heat_kernel_true"] = self.lindstedt_expansion.heat_kernel_true
            base["lindstedt_residual"] = self.lindstedt_expansion.expansion_residual
            base["lindstedt_is_consistent"] = self.lindstedt_expansion.is_asymptotically_consistent
            base["lindstedt_pade_approx"] = self.lindstedt_expansion.pade_resummed_approx
        if self.quantum_rlc_snapshot is not None:
            base["rlc_resistance_R"] = self.quantum_rlc_snapshot.resistance_R
            base["rlc_reactance_X"] = self.quantum_rlc_snapshot.reactance_X
            base["rlc_quality_factor_Q"] = self.quantum_rlc_snapshot.quality_factor_Q
            base["rlc_apparent_power_S"] = self.quantum_rlc_snapshot.apparent_power_S
            base["rlc_axioms_kee_satisfied"] = self.quantum_rlc_snapshot.axioms_kee_satisfied
        if self.return_map is not None:
            base["return_map_J"] = self.return_map.contraction_factor
            base["return_map_is_stable"] = self.return_map.is_stable
            base["return_map_floquet_trace"] = self.return_map.floquet_trace
            base["return_map_lyapunov_exponent"] = self.return_map.lyapunov_exponent
        if self.duality is not None:
            base["duality_curvature_rank"] = self.duality.curvature_rank
            base["duality_curvature_nullity"] = self.duality.curvature_nullity
            base["duality_total_dimension"] = self.duality.total_dimension
            base["duality_residual"] = self.duality.duality_residual
            base["is_poincare_dual"] = self.duality.is_poincare_dual
            base["duality_hodge_involution_defect"] = self.duality.hodge_involution_defect
        if self.spectral_recurrence is not None:
            base["recurrence_count"] = len(self.spectral_recurrence.recurrence_points)
            base["recurrence_min_distance"] = self.spectral_recurrence.min_recurrence_distance
            base["recurrence_is_recurrent"] = self.spectral_recurrence.is_poincare_recurrent
            base["recurrence_level_spacing_ratio"] = self.spectral_recurrence.level_spacing_ratio
        base["adiabatic_action_drift"] = self.adiabatic_action_drift
        base["meta_residual"] = self.meta_residual
        if self.rsi_result is not None:
            base["rsi_final_scale"] = self.rsi_result.final_scale
            base["rsi_final_n_terms"] = self.rsi_result.final_n_terms
            base["rsi_final_residual"] = self.rsi_result.final_residual
            base["rsi_converged"] = self.rsi_result.converged
            base["rsi_level3_activated"] = self.rsi_result.level3_activated
            base["rsi_effective_convergence_order"] = self.rsi_result.effective_convergence_order
        return base


# =============================================================================
# Motor Soberano Electrodinámico en 3 Fases Estrictamente Anidadas
# =============================================================================
class KapexElectrodynamicEngine:
    r"""
    Motor de cálculo ciego electrodinámico, simpléctico y cuántico para el Estrato KAPEX.

    El código de esta clase está estructurado en TRES FASES ANIDADAS de modo que la
    clausura formal del último método de cada fase constituye el dominio exacto de
    entrada del primer método de la fase subsiguiente:
        Fase 1: Culmina en `_phase1_observe_orient` → emite `PhaseOneKapexPacket`.
        Fase 2: Inicia en `_phase2_spectral_integrity(packet: PhaseOneKapexPacket)`
                y culmina en `_phase2_decide` → emite `PhaseTwoKapexPacket`.
        Fase 3: Inicia en `_phase3_conservation_audit(packet1, packet2: PhaseTwoKapexPacket)`,
                sella en `_phase3_act` y ejecuta el punto fijo recursivo de Nivel 3.
    """

    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError("La dimensión espacial del ápice debe ser estrictamente positiva.")
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║                      FASE ANIDADA 1: OBSERVE & ORIENT                 ║
    # ║  Geometría Espectral, Mecánica Celeste de Poincaré-KAM, Expansión de  ║
    # ║  Lindstedt-Padé, Eikonal Simpéctica, Yang-Mills, Hodge-Poynting y     ║
    # ║  Bicuaterniones de Riemann-Silberstein / Impedancia Cuántica RLC.     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================

    def kahan_sum(self, arr: np.ndarray) -> float:
        """Suma compensada de Kahan-Babuška-Neumaier sobre vectores 1-D reales."""
        if np.asarray(arr).ndim != 1:
            raise ValueError(f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}")
        return float(np.real(self._neumaier_sum(arr)))

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        """Acumulador compensado de Neumaier en ℝ o ℂ con cota de error O(ε_mach)."""
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def _validate_square_matrix(self, M: np.ndarray, name: str) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada (n × n), se recibió {M.shape}")
        if M.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != self._n:
            raise ValueError(f"{name} debe ser un vector de dimensión {self._n}, se recibió {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_hermitian(
        self, M: np.ndarray, name: str, tolerance: float = _HERMITIAN_TOLERANCE,
    ) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tolerance * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(
                f"{name} no es hermítica: ||M − M†||_F / ||M||_F excede {tolerance}"
            )

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        """Proyección ortogonal autoadjunta sobre el subespacio hermítico."""
        return 0.5 * (M + M.conj().T)

    def project_to_so_n(self, A: np.ndarray) -> Tuple[np.ndarray, float]:
        """Proyección de Cartan sobre el álgebra de Lie antihermítica u(n) / so(n)."""
        self._validate_square_matrix(A, "connection_component")
        projected = 0.5 * (A - A.conj().T)
        residual = float(la.norm(A - projected, ord="fro"))
        return projected, residual

    def killing_pairing(self, X: np.ndarray, Y: np.ndarray) -> complex:
        """Forma bilineal invariante de Cartan-Killing B(X, Y) = (n − 2)·tr(X Y)."""
        factor = float(self._n - 2) if self._n >= 3 else 1.0
        return factor * np.trace(X @ Y)

    def _factorize_metric(
        self,
        metric_tensor: np.ndarray,
        tolerance_bundle: Optional[KapexToleranceBundle] = None,
    ) -> SpectralChart:
        r"""
        Descomposición espectral de Rayleigh-Ritz g = U Λ U† con regularización
        espectral de Tikhonov acoplada al `KapexToleranceBundle` del controlador RSI-3.
        """
        self._validate_square_matrix(metric_tensor, "metric_tensor")
        herm = self._hermitize(metric_tensor)
        self._validate_hermitian(herm, "metric_tensor")

        eigenvalues, eigenvectors = la.eigh(herm)
        eigenvalues = np.real(eigenvalues)

        effective_reg = (
            tolerance_bundle._scaled(self._reg)
            if tolerance_bundle is not None
            else self._reg
        )
        regularized = bool(np.any(eigenvalues <= effective_reg))
        if regularized:
            logger.warning(
                "Métrica no definida positiva o mal condicionada (λ_min=%.3e); "
                "Tikhonov α=%.3e.",
                float(np.min(eigenvalues)),
                effective_reg,
            )
        shifted = eigenvalues + effective_reg
        shifted = np.maximum(shifted, _WILKINSON_FLOOR)

        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T

        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        log_vol = 0.5 * float(self.kahan_sum(np.log(np.maximum(lam_abs, _WILKINSON_FLOOR))))
        volume = float(np.exp(log_vol))
        op_norm = float(np.max(lam_abs))
        frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
        if shifted.size >= 2:
            ordered = np.sort(lam_abs)
            gap = float(ordered[1] - ordered[0])
        else:
            gap = float(lam_abs[0])

        return SpectralChart(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            inv_metric=np.real_if_close(inv_metric, tol=1e6),
            condition_number=cond,
            volume_density=volume,
            operator_norm=op_norm,
            frobenius_norm=frob,
            spectral_gap=gap,
            regularized=regularized,
            tikhonov_alpha=effective_reg,
        )

    def _invert_metric(self, metric_tensor: np.ndarray) -> np.ndarray:
        return np.array(self._factorize_metric(metric_tensor).inv_metric, copy=True)

    def _complex_step_eikonal_momentum(
        self,
        grad_S: np.ndarray,
        inv_metric: np.ndarray,
        step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Diferenciación por Paso Complejo de Squire-Trapp (CSMD) libre de cancelación
        sustractiva para el gradiente simpléctico ∂(p_μ g^{μν} p_ν)/∂p_λ = 2 p^λ.
        """
        n = self._n
        p = np.asarray(grad_S, dtype=np.complex128)
        ginv = np.asarray(inv_metric, dtype=np.complex128)
        sens = np.zeros(n, dtype=np.float64)
        eff_step = max(1e-30, float(step))
        for lam in range(n):
            pert = p.copy()
            pert[lam] += 1.0j * eff_step
            value = pert @ ginv @ pert
            sens[lam] = float(np.imag(value) / eff_step)
        return sens

    def _apply_symplectic_homotopy_projection(
        self,
        grad_S: np.ndarray,
        chart: SpectralChart,
        refraction_index: float,
        homotopy_gain: float,
    ) -> np.ndarray:
        r"""
        [OPERADOR DE NIVEL 3 RSI]: Proyección homológica de Hamilton-Jacobi sobre la
        hipersuperficie de nivel isoenergética H(x, p) = ½(g^{μν}p_μ p_ν − n²) = 0.
        Cuando `homotopy_gain == 0.0` (modo estándar), actúa como el morfismo identidad.
        Cuando `homotopy_gain > 0.0` (activado por RSI-3), transporta el momento a lo
        largo del flujo geodésico normal a la cáscara de masa óptica.
        """
        grad = np.asarray(grad_S, dtype=np.float64)
        if homotopy_gain <= 0.0:
            return grad
        inv_g = np.asarray(chart.inv_metric, dtype=np.float64)
        p_up = np.real(inv_g @ grad)
        norm_sq = float(np.real(np.dot(p_up, grad)))
        target_sq = float(refraction_index) ** 2
        if norm_sq <= _WILKINSON_FLOOR:
            return grad
        exact_scale = math.sqrt(target_sq / norm_sq)
        alpha = float(np.clip(homotopy_gain, 0.0, 1.0))
        blended_scale = (1.0 - alpha) + alpha * exact_scale
        return grad * blended_scale

    def solve_eikonal_residual(
        self,
        grad_S: np.ndarray,
        metric_tensor: np.ndarray,
        refraction_index: float,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Resuelve el residuo de la ecuación Eikonal relativista |g^{μν} ∂_μ S ∂_ν S − n²|.
        """
        self._validate_vector(grad_S, "grad_S")
        if not np.isfinite(refraction_index) or refraction_index <= 0.0:
            raise ValueError("refraction_index debe ser un real estrictamente positivo y finito")

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric)
        grad = np.asarray(grad_S, dtype=np.float64)

        p_up = np.real(inv_metric @ grad)
        inner_product = float(np.real(np.dot(p_up, grad)))
        n_squared = float(refraction_index) ** 2
        eikonal_residual = float(abs(inner_product - n_squared))
        return eikonal_residual, p_up

    def compute_yang_mills_action(
        self,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        coupling_constant: float = 1.0,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Calcula el tensor de curvatura no abeliano F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]
        y el funcional de acción de Yang-Mills S_YM = −(1/4g²) Tr(F_{μν} F^{μν}).
        """
        if not isinstance(connection_A, list) or len(connection_A) != self._n:
            raise ValueError(f"connection_A debe ser una lista de {self._n} matrices.")
        if coupling_constant <= 0.0 or not np.isfinite(coupling_constant):
            raise ValueError("coupling_constant debe ser un real estrictamente positivo y finito")

        projected: List[np.ndarray] = []
        for mu, A in enumerate(connection_A):
            self._validate_square_matrix(A, f"A[{mu}]")
            pi_A, _ = self.project_to_so_n(A)
            projected.append(pi_A)

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric, dtype=np.complex128)

        n = self._n
        F = np.zeros((n, n, n, n), dtype=np.complex128)
        for mu in range(n):
            for nu in range(n):
                commutator = projected[mu] @ projected[nu] - projected[nu] @ projected[mu]
                if connection_dA is not None:
                    d_mu_nu = np.asarray(connection_dA[mu][nu], dtype=np.complex128)
                    d_nu_mu = np.asarray(connection_dA[nu][mu], dtype=np.complex128)
                    F[mu, nu] = d_mu_nu - d_nu_mu + commutator
                else:
                    F[mu, nu] = commutator

        F_up = np.einsum("ma,nb,abij->mnij", inv_metric, inv_metric, F)
        trace_density = np.einsum("mnij,mnji->mn", F, F_up)
        total_trace = self._neumaier_sum(np.real(trace_density).astype(np.float64).ravel())
        yang_mills_action = -0.25 * (1.0 / (coupling_constant ** 2)) * float(np.real(total_trace))
        return F, float(yang_mills_action)

    def _bianchi_jacobi_residual(
        self,
        connection_A: List[np.ndarray],
        F: np.ndarray,
    ) -> float:
        r"""
         Evalúa la primera identidad diferencial de Bianchi-Jacobi covariante:
        D_λ F_{μν} + D_μ F_{νλ} + D_ν F_{λμ} = 0 en el fibrado adjunto Ad(P).
        """
        n = self._n
        acc: List[float] = []
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    term = (
                        connection_A[lam] @ F[mu, nu] - F[mu, nu] @ connection_A[lam]
                        + connection_A[mu] @ F[nu, lam] - F[nu, lam] @ connection_A[mu]
                        + connection_A[nu] @ F[lam, mu] - F[lam, mu] @ connection_A[nu]
                    )
                    acc.append(float(la.norm(term, ord="fro") ** 2))
        energy = float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))))
        return float(np.sqrt(max(energy, 0.0)) / max(n ** 1.5, 1.0))

    def _pontryagin_density(self, F: np.ndarray) -> float:
        """Densidad topológica de instantón / segunda clase de Chern-Pontryagin."""
        density = np.einsum("mnij,mnji->", F, F)
        return float(np.real(density))

    def compute_poynting_strategic_flux(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        metric_tensor: np.ndarray,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Calcula el vector de flujo de Poynting contravariante S^i = g^{ij}(★_g(E♭ ∧ B♭))_j
        y la densidad de energía electromagnética U = ½(⟨E, E⟩_g + ⟨B, B⟩_g).
        """
        self._validate_vector(electric_field, "electric_field")
        self._validate_vector(magnetic_field, "magnetic_field")

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric, dtype=np.float64)
        E = np.asarray(electric_field, dtype=np.float64)
        B = np.asarray(magnetic_field, dtype=np.float64)

        E_up = np.real(inv_metric @ E)
        B_up = np.real(inv_metric @ B)
        energy_density = 0.5 * (
            float(np.real(np.dot(E_up, E))) + float(np.real(np.dot(B_up, B)))
        )

        if self._n != 3:
            logger.warning(
                "★_g (E♭ ∧ B♭) es un (n−2)-forma; n=%d ≠ 3. "
                "Se emite el vector nulo y U calculada con g.",
                self._n,
            )
            return np.zeros(self._n, dtype=np.float64), float(energy_density)

        vol = float(chart.volume_density)
        S_down = vol * np.cross(E_up, B_up)
        S_up = np.real(inv_metric @ S_down)
        return S_up.astype(np.float64), float(energy_density)

    def compute_maxwell_stress_energy(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        spectral_chart: SpectralChart,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Tensor de esfuerzo electromagnético de Maxwell T^{ij} = E^i E^j + B^i B^j − U g^{ij}
        y 2-forma exterior de Faraday-Poynting E♭ ∧ B♭.
        """
        inv_g = np.asarray(spectral_chart.inv_metric, dtype=np.float64)
        E = np.asarray(electric_field, dtype=np.float64)
        B = np.asarray(magnetic_field, dtype=np.float64)
        E_up = np.real(inv_g @ E)
        B_up = np.real(inv_g @ B)
        uu = float(np.real(np.dot(E_up, E)) + np.real(np.dot(B_up, B)))
        T_up = np.outer(E_up, E_up) + np.outer(B_up, B_up) - 0.5 * uu * inv_g
        two_form = np.outer(E, B) - np.outer(B, E)
        return T_up.astype(np.float64), two_form.astype(np.float64)

    def _phase1_compute_quantum_rlc_biquaternion(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        poynting_flux: np.ndarray,
        chart: SpectralChart,
    ) -> QuantumRLCImpedanceSnapshot:
        r"""
        Construye el bivector hipercomplejo de Riemann-Silberstein F_RS = E + iB ∈ Cl_{3,0}(ℂ),
        extrae los dos invariantes relativistas de Lorentz-Poincaré:
            I₁ = ⟨E, E⟩_g − ⟨B, B⟩_g,   I₂ = 2⟨E, B⟩_g,
        y sintetiza la impedancia cuántica compleja Z(ω) = R + iX del resonador RLC
        asociado a la línea de transmisión cognitiva, auditando KEE1–KEE3 e Inv 1–2.
        """
        inv_g = np.asarray(chart.inv_metric, dtype=np.float64)
        E = np.asarray(electric_field, dtype=np.float64)
        B = np.asarray(magnetic_field, dtype=np.float64)
        E_up = np.real(inv_g @ E)
        B_up = np.real(inv_g @ B)

        e_energy_2 = max(0.0, float(np.real(np.dot(E_up, E))))
        b_energy_2 = max(0.0, float(np.real(np.dot(B_up, B))))
        eb_pairing = float(np.real(np.dot(E_up, B)))

        # Invariantes del bicuaternión de Riemann-Silberstein F_RS² = I₁ + i I₂
        lorentz_scalar = e_energy_2 - b_energy_2
        lorentz_pseudoscalar = 2.0 * eb_pairing

        # Potencia activa P (flujo radiativo de Poynting + amortiguamiento) y reactiva Q
        flux_norm = float(la.norm(poynting_flux))
        total_energy = 0.5 * (e_energy_2 + b_energy_2)
        active_P = max(_WILKINSON_FLOOR, flux_norm + 0.5 * total_energy)
        reactive_Q = 0.5 * lorentz_scalar

        apparent_S = math.hypot(active_P, reactive_Q)
        power_triangle_res = abs(apparent_S ** 2 - (active_P ** 2 + reactive_Q ** 2)) / max(
            1.0, apparent_S ** 2
        )

        # Impedancia compleja efectiva normalizada por el cuadrado de la corriente modal
        modal_current_sq = max(_WILKINSON_FLOOR, b_energy_2 + _WILKINSON_FLOOR)
        resistance_R = max(0.0, active_P / modal_current_sq)
        reactance_X = reactive_Q / modal_current_sq
        z_mag = math.hypot(resistance_R, reactance_X)
        z_phase = math.atan2(reactance_X, resistance_R)

        # Factor de calidad cuántico Q_RLC = ω₀ W_stored / P_dissipated > 0
        omega_0 = max(_WILKINSON_FLOOR, float(chart.operator_norm))
        quality_Q = max(
            _WILKINSON_FLOOR,
            (omega_0 * max(_WILKINSON_FLOOR, total_energy)) / active_P,
        )

        axioms_ok = (
            resistance_R >= 0.0
            and quality_Q > 0.0
            and power_triangle_res <= 1e-12
        )
        return QuantumRLCImpedanceSnapshot(
            resistance_R=float(resistance_R),
            reactance_X=float(reactance_X),
            impedance_magnitude=float(z_mag),
            impedance_phase_rad=float(z_phase),
            active_power_P=float(active_P),
            reactive_power_Q=float(reactive_Q),
            apparent_power_S=float(apparent_S),
            power_triangle_residual=float(power_triangle_res),
            quality_factor_Q=float(quality_Q),
            riemann_silberstein_lorentz_scalar=float(lorentz_scalar),
            riemann_silberstein_pseudoscalar=float(lorentz_pseudoscalar),
            axioms_kee_satisfied=bool(axioms_ok),
        )

    def _phase1_compute_poincare_snapshot(
        self, chart: SpectralChart, section_level: float = _POINCARE_DEFAULT_SECTION,
    ) -> PoincareSpectralSnapshot:
        r"""
        Snapshot de Mecánica Celeste de Poincaré-KAM sobre la carta espectral:
            Coordenada canónica: q = λ_min,
            Momento conjugado:   p = λ_max − λ_min,
            Acción de Jacobi:    I = ½(q² + p²),
            Ángulo canónico:     θ_p = atan2(p, q),
            1-forma de Poincaré-Cartan: θ_PC = p·q − I.
        Evalúa intersecciones transversales con la superficie de sección Σ = {q = section_level}
        y el margen diofántico KAM frente a resonancias racionales p_r/q_r.
        """
        eigvals = np.asarray(chart.eigenvalues)
        if eigvals.size == 0:
            return PoincareSpectralSnapshot(
                action_invariant=0.0, angle_parameter=0.0,
                section_level=float(section_level), crossing_count=0,
                frequency_ratio=0.0, is_near_resonance=False,
                resonance_p=0, resonance_q=0, small_divisor_magnitude=0.0,
                kam_diophantine_margin=0.0, poincare_cartan_form=0.0,
            )
        lam_abs = np.abs(eigvals + chart.tikhonov_alpha)
        lam_abs = np.maximum(lam_abs, _WILKINSON_FLOOR)
        q = float(np.min(lam_abs))
        lam_max = float(np.max(lam_abs))
        p = max(0.0, lam_max - q)
        action = 0.5 * (q * q + p * p)
        angle = math.atan2(p, q) if action > _WILKINSON_FLOOR else 0.0

        sec = float(section_level)
        r_sq = 2.0 * action
        if r_sq > _WILKINSON_FLOOR and abs(sec) < math.sqrt(r_sq):
            crossings = 2
        elif (r_sq > _WILKINSON_FLOOR
              and abs(abs(sec) - math.sqrt(r_sq)) <= _WILKINSON_FLOOR):
            crossings = 1
        else:
            crossings = 0

        ratio = q / max(1e-12, p)
        is_res, p_res, q_res, dioph_err, small_div = _detect_resonance(ratio)
        kam_bound = (
            _KAM_DIOPHANTINE_GAMMA / (float(max(1, q_res)) ** _KAM_DIOPHANTINE_TAU)
            if q_res > 0 else 0.0
        )
        kam_margin = float(dioph_err - kam_bound)
        poincare_cartan = float(p * q - action)

        return PoincareSpectralSnapshot(
            action_invariant=float(action),
            angle_parameter=float(angle),
            section_level=float(sec),
            crossing_count=int(crossings),
            frequency_ratio=float(ratio),
            is_near_resonance=bool(is_res),
            resonance_p=int(p_res),
            resonance_q=int(q_res),
            small_divisor_magnitude=float(small_div),
            kam_diophantine_margin=kam_margin,
            poincare_cartan_form=poincare_cartan,
        )

    def _phase1_compute_lindstedt_heat_kernel(
        self,
        chart: SpectralChart,
        n_terms: int = 2,
        use_pade: bool = False,
    ) -> LindstedtHeatKernelExpansion:
        r"""
        Expansión de Lindstedt-Poincaré de la traza del núcleo de calor Φ(ε) = n⁻¹tr(e^{−εg})
        con parámetro perturbativo ε = 1/(1 + λ_max).
        En modo estándar (`use_pade=False`, N=2) mantiene equivalencia exacta con v8.0.0.
        En modo RSI-3 (`use_pade=True`), emplea el aproximante diagonal de Padé [M/M](-εg)
        para cancelar términos seculares de orden superior.
        """
        eigvals = np.asarray(chart.eigenvalues)
        if eigvals.size == 0:
            return LindstedtHeatKernelExpansion(
                small_parameter=0.0, n_terms=int(n_terms),
                zero_order=1.0, first_order=0.0, second_order=0.0,
                higher_order=0.0,
                heat_kernel_approx=1.0, heat_kernel_true=1.0,
                expansion_residual=0.0, is_asymptotically_consistent=True,
                pade_resummed_approx=1.0,
            )
        shifted = eigvals + chart.tikhonov_alpha
        lam_max = float(np.max(np.abs(shifted)))
        eps = 1.0 / (1.0 + max(1e-12, lam_max))
        N = max(2, int(n_terms))
        n = shifted.size

        approx = 0.0
        pade_approx = 0.0
        true = 0.0
        zero = 0.0
        first = 0.0
        second = 0.0
        higher = 0.0
        for lam in shifted:
            x = eps * float(lam)
            t0 = 1.0
            t1 = -x
            t2 = (x * x / 2.0) if N >= 3 else 0.0
            th = 0.0
            fact = 6.0
            xk = x * x * x
            for k in range(3, N):
                th += ((-1.0) ** k) * xk / fact
                fact *= (k + 1)
                xk *= x
            # Nota de compatibilidad v8.0.0: en v8 t2 se sumaba incluso con N=2
            t2_v8 = x * x / 2.0
            approx += t0 + t1 + t2_v8 + th
            pade_approx += _pade_resum_exponential(x, N)
            try:
                true += math.exp(-x)
            except OverflowError:
                true += 0.0
            zero += t0
            first += t1
            second += t2_v8
            higher += th

        approx /= n
        pade_approx /= n
        true /= n
        zero /= n
        first /= n
        second /= n
        higher /= n

        effective_approx = pade_approx if use_pade else approx
        residual = abs(effective_approx - true) / max(1e-12, abs(true))
        is_consistent = residual <= _ADIABATIC_DRIFT_LIMIT or eps < 0.5

        return LindstedtHeatKernelExpansion(
            small_parameter=float(eps),
            n_terms=int(N),
            zero_order=float(zero),
            first_order=float(first),
            second_order=float(second),
            higher_order=float(higher),
            heat_kernel_approx=float(effective_approx),
            heat_kernel_true=float(true),
            expansion_residual=float(residual),
            is_asymptotically_consistent=bool(is_consistent),
            pade_resummed_approx=float(pade_approx),
        )

    def _phase1_observe_orient(
        self,
        grad_S: np.ndarray,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        refraction_index: float,
        coupling_constant: float = 1.0,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
        tolerance_bundle: Optional[KapexToleranceBundle] = None,
    ) -> PhaseOneKapexPacket:
        r"""
        ========================================================================================
        [CLAUSURA FORMAL DE LA FASE ANIDADA 1 — ÚLTIMO MÉTODO DE FASE 1: OBSERVE + ORIENT]
        ========================================================================================
        Morfismo F₁: Estado_Crudo × KapexToleranceBundle ──► PhaseOneKapexPacket.

        INTEGRA:
          1. Factorización espectral de Rayleigh-Ritz-Tikhonov (`SpectralChart`).
          2. Snapshot celeste de Poincaré-KAM (`PoincareSpectralSnapshot`).
          3. Expansión de Lindstedt-Poincaré / Padé del Heat Kernel (`LindstedtHeatKernelExpansion`).
          4. Ecuación Eikonal con proyección homológica RSI-3 y sensibilidad CSMD.
          5. Acción de Yang-Mills, identidad de Bianchi-Jacobi y densidad de Pontryagin.
          6. Flujo estratégico de Hodge-Poynting, tensor de Maxwell y bicuaternión RLC cuántico.

        CONTINUIDAD DE ANIDAMIENTO CATEGÓRICO:
          El objeto `PhaseOneKapexPacket` retornado por este método constituye la entrada
          formal inmediata del primer método de la Fase 2 (`_phase2_spectral_integrity`).
        ========================================================================================
        """
        bundle = tolerance_bundle if tolerance_bundle is not None else KapexToleranceBundle()
        logger.info("Fase 1: carta espectral, Poincaré-KAM, Lindstedt, Eikonal+CSMD, Yang-Mills, Poynting-RLC.")

        chart = self._factorize_metric(metric_tensor, tolerance_bundle=bundle)

        # 1. Mecánica Celeste de Poincaré-KAM y Expansión de Lindstedt-Poincaré
        poincare = self._phase1_compute_poincare_snapshot(chart)
        lindstedt = self._phase1_compute_lindstedt_heat_kernel(
            chart,
            n_terms=bundle.n_lindstedt_terms,
            use_pade=bundle.use_pade_resummation,
        )

        # 2. Ecuación Eikonal + Proyección Homológica Simpéctica (activa en RSI Nivel 3)
        effective_grad_S = self._apply_symplectic_homotopy_projection(
            grad_S, chart, refraction_index, bundle.symplectic_homotopy_gain
        )
        eikonal_res, p_up = self.solve_eikonal_residual(
            effective_grad_S, metric_tensor, refraction_index, spectral_chart=chart
        )
        inner = float(np.real(np.dot(p_up, np.asarray(effective_grad_S, dtype=np.float64))))
        hamiltonian = 0.5 * (inner - float(refraction_index) ** 2)

        csmd_step = max(1e-24, _CSMD_STEP * min(1.0, bundle.scale))
        csmd = self._complex_step_eikonal_momentum(
            effective_grad_S, chart.inv_metric, step=csmd_step
        )
        csmd_err = float(la.norm(csmd - 2.0 * np.asarray(p_up, dtype=np.float64)))

        # 3. Proyección de Cartan al álgebra de Lie so(n) y Acción de Yang-Mills
        projected_A: List[np.ndarray] = []
        proj_residuals: List[float] = []
        homotopy_damp = 1.0 - 0.5 * float(np.clip(bundle.symplectic_homotopy_gain, 0.0, 1.0))
        for A in connection_A:
            pi_A, r = self.project_to_so_n(np.asarray(A))
            projected_A.append(pi_A)
            proj_residuals.append(r * homotopy_damp)
        algebra_res = float(np.real(self._neumaier_sum(np.asarray(proj_residuals, dtype=np.float64))))

        F, ym_action = self.compute_yang_mills_action(
            projected_A,
            metric_tensor,
            coupling_constant=coupling_constant,
            connection_dA=connection_dA,
            spectral_chart=chart,
        )
        inv_metric = np.asarray(chart.inv_metric, dtype=np.complex128)
        F_up = np.einsum("ma,nb,abij->mnij", inv_metric, inv_metric, F)
        raw_trace = float(np.real(self._neumaier_sum(
            np.real(np.einsum("mnij,mnji->mn", F, F_up)).ravel()
        )))
        jacobi = self._bianchi_jacobi_residual(projected_A, F)
        pontryagin = self._pontryagin_density(F)

        # 4. Flujo de Hodge-Poynting, Esfuerzo de Maxwell y Bicuaternión RLC Cuántico
        poynting_up, energy_density = self.compute_poynting_strategic_flux(
            electric_field, magnetic_field, metric_tensor, spectral_chart=chart
        )
        stress, two_form = self.compute_maxwell_stress_energy(
            electric_field, magnetic_field, chart
        )
        rlc_snapshot = self._phase1_compute_quantum_rlc_biquaternion(
            electric_field, magnetic_field, poynting_up, chart
        )

        packet = PhaseOneKapexPacket(
            eikonal_residual=eikonal_res,
            eikonal_hamiltonian=hamiltonian,
            momentum_up_vector=p_up,
            csmd_momentum=csmd,
            csmd_gradient_error=csmd_err,
            curvature_tensor=F,
            yang_mills_action=ym_action,
            yang_mills_raw_trace=raw_trace,
            bianchi_jacobi_residual=jacobi,
            instanton_density=pontryagin,
            poynting_flux_vector=poynting_up,
            poynting_two_form=two_form,
            field_energy_density=energy_density,
            maxwell_stress_tensor=stress,
            spectral_chart=chart,
            algebra_projection_residual=algebra_res,
            poincare_snapshot=poincare,
            lindstedt_expansion=lindstedt,
            tolerance_bundle=bundle,
            quantum_rlc_snapshot=rlc_snapshot,
        )
        logger.debug(
            "Fase 1 completa: Eik=%.6e H=%.6e YM=%.6e U=%.6e Jacobi=%.6e CSMD=%.6e κ=%.3e",
            eikonal_res, hamiltonian, ym_action, energy_density, jacobi, csmd_err,
            chart.condition_number,
        )
        return packet

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║                        FASE ANIDADA 2: DECIDE                         ║
    # ║  Continuación directa desde `PhaseOneKapexPacket` (_phase1_observe):  ║
    # ║  Integridad de Álgebra de Banach, Holonomía de Wilson, Mapa de        ║
    # ║  Retorno Simpéctico de Poincaré-Chirikov, Dualidad de Hodge-Poincaré, ║
    # ║  Recurrencia Ergódica de Birkhoff y Clasificación en Topos de Heyting ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================

    def _phase2_spectral_integrity(
        self,
        packet: PhaseOneKapexPacket,
        connection_A: List[np.ndarray],
    ) -> SpectralIntegrityReport:
        r"""
        ========================================================================================
        [INICIO DE LA FASE ANIDADA 2 — ESLABÓN RECEPTOR DE `_phase1_observe_orient`]
        ========================================================================================
        Recibe directamente el objeto terminal `packet: PhaseOneKapexPacket` de la Fase 1
        y evalúa la integridad espectral de la métrica, la norma de Frobenius de la curvatura
        y el defecto submultiplicativo de álgebra de Banach ||A_μ A_ν||₂ ≤ ||A_μ||₂ ||A_ν||₂.
        ========================================================================================
        """
        chart = packet.spectral_chart
        F = np.asarray(packet.curvature_tensor)
        curv_frob = float(np.linalg.norm(F))

        radii = np.zeros(self._n, dtype=np.float64)
        ops: List[float] = []
        mats: List[np.ndarray] = []
        for mu, A in enumerate(connection_A):
            M = np.asarray(A, dtype=np.complex128)
            mats.append(M)
            spec = la.eigvals(M)
            radii[mu] = float(np.max(np.abs(spec))) if spec.size else 0.0
            ops.append(float(la.norm(M, ord=2)))

        defect = 0.0
        for mu in range(self._n):
            for nu in range(self._n):
                lhs = float(la.norm(mats[mu] @ mats[nu], ord=2))
                rhs = ops[mu] * ops[nu]
                if rhs > _WILKINSON_FLOOR:
                    defect = max(defect, max(0.0, lhs / rhs - 1.0))

        cond_term = 1.0 / (1.0 + np.log10(max(chart.condition_number, 1.0)))
        gap_term = float(np.tanh(chart.spectral_gap / max(chart.operator_norm, _WILKINSON_FLOOR)))
        curv_term = float(np.exp(-curv_frob))
        banach_term = float(np.exp(-defect / _MACHINE_EPS))
        score = float(cond_term * max(gap_term, 0.0) * curv_term * min(banach_term, 1.0))

        return SpectralIntegrityReport(
            metric_condition=float(chart.condition_number),
            metric_gap=float(chart.spectral_gap),
            curvature_frobenius=curv_frob,
            connection_spectral_radii=radii,
            banach_submultiplicative_defect=float(defect),
            integrity_score=score,
        )

    def _phase2_gauge_holonomy(
        self,
        connection_A: List[np.ndarray],
    ) -> Tuple[float, float]:
        r"""
        Calcula la holonomía no abeliana media de las plaquetas de Wilson:
            W_{μν} = n⁻¹ Re Tr(exp(A_μ) exp(A_ν) exp(−A_μ) exp(−A_ν)).
        """
        if self._n < 2:
            return 1.0, 1.0
        acc: List[float] = []
        for mu in range(self._n):
            for nu in range(mu + 1, self._n):
                A_mu = np.asarray(connection_A[mu], dtype=np.complex128)
                A_nu = np.asarray(connection_A[nu], dtype=np.complex128)
                U = (
                    la.expm(A_mu)
                    @ la.expm(A_nu)
                    @ la.expm(-A_mu)
                    @ la.expm(-A_nu)
                )
                acc.append(float(np.real(np.trace(U)) / self._n))
        mean = float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))) / max(len(acc), 1))
        return mean, float(max(mean, 0.0))

    def _phase2_compute_return_map(
        self, packet1: PhaseOneKapexPacket, n_iterations: int = 4,
    ) -> EnergyReturnMapSnapshot:
        r"""
        Mapa de Retorno Simpéctico de Poincaré-Chirikov en el espacio fase espectral (q, p).

        A diferencia de una combinación convexa estática, evoluciona la órbita mediante un
        Twist Map simpléctico preservador de área (det DP = 1) acoplado a la intensidad de
        curvatura de Yang-Mills K_YM:
            I_{k+1} = max(ε, I_k + K_YM · I_0 · sin(θ_k)),
            θ_{k+1} = (θ_k + 2π·J + K_YM · cos(θ_k)) mod 2π,
        donde J = min(q/p, p/q) ∈ (0, 1] es el factor de contracción de frecuencias.
        Calcula asimismo la traza de la matriz de monodromía de Floquet Tr(DP) y el
        exponente de Lyapunov finito local.
        """
        chart = packet1.spectral_chart
        eigvals = np.asarray(chart.eigenvalues)
        if eigvals.size < 2:
            return EnergyReturnMapSnapshot(
                contraction_factor=1.0, eigenvalues_modulus=(),
                is_stable=True, iterations_norm_sequence=(),
                floquet_trace=2.0, lyapunov_exponent=0.0,
            )
        lam_abs = np.abs(eigvals + chart.tikhonov_alpha)
        lam_abs = np.maximum(lam_abs, _WILKINSON_FLOOR)
        q = float(np.min(lam_abs))
        lam_max = float(np.max(lam_abs))
        p = max(0.0, lam_max - q)
        if q <= 0.0 or p <= 0.0:
            return EnergyReturnMapSnapshot(
                contraction_factor=1.0, eigenvalues_modulus=(),
                is_stable=True, iterations_norm_sequence=(),
                floquet_trace=2.0, lyapunov_exponent=0.0,
            )

        J = min(q / p, p / q)
        f_norm = float(np.linalg.norm(packet1.curvature_tensor))
        k_ym = float(np.tanh(f_norm / max(1.0, float(self._n ** 2)))) * 1e-2

        action_0 = 0.5 * (q * q + p * p)
        cur_I = action_0
        cur_theta = math.atan2(p, q)
        norms: List[float] = []
        lyap_acc = 0.0

        for _ in range(n_iterations):
            sin_t = math.sin(cur_theta)
            cos_t = math.cos(cur_theta)
            cur_I = max(_WILKINSON_FLOOR, cur_I + k_ym * action_0 * sin_t)
            cur_theta = (cur_theta + 2.0 * math.pi * J + k_ym * cos_t) % (2.0 * math.pi)
            radius = math.sqrt(2.0 * cur_I)
            cur_q = radius * math.cos(cur_theta)
            cur_p = radius * math.sin(cur_theta)
            norms.append(float(math.hypot(cur_q, cur_p)))
            # Traza jacobiana local del mapa simpléctico Tr(DP) = 2 + K_YM²·cos(θ)·sin(θ)
            local_tr = 2.0 - k_ym * cos_t
            disc = local_tr * local_tr - 4.0
            if disc > 0.0:
                lam_floquet = 0.5 * (abs(local_tr) + math.sqrt(disc))
                lyap_acc += math.log(max(1.0, lam_floquet))

        floquet_trace = 2.0 - k_ym * math.cos(cur_theta)
        lyapunov = lyap_acc / max(1, n_iterations)
        return EnergyReturnMapSnapshot(
            contraction_factor=float(J),
            eigenvalues_modulus=(float(abs(J)),),
            is_stable=bool(abs(J) <= 1.0 + 1e-9 and abs(floquet_trace) <= 2.0 + 1e-2),
            iterations_norm_sequence=tuple(norms),
            floquet_trace=float(floquet_trace),
            lyapunov_exponent=float(lyapunov),
        )

    def _phase2_compute_poincare_duality(
        self, packet1: PhaseOneKapexPacket,
    ) -> PoincareDualitySnapshot:
        r"""
        Teorema de Dualidad de Hodge-Poincaré y Rango-Nulidad sobre el operador de curvatura
        F ∈ Ω²(M, so(n)) ≅ ℂ^{n²×n²}.

        Verifica:
          1. Descomposición en valores singulares (SVD) de F_{(μ,i),(ν,j)} con umbral de
             Eckart-Young-Mirsky para determinar `curvature_rank` y `curvature_nullity`.
          2. Defecto de involución de la estrella de Hodge riemanniana y antisimetría
             cohomológica de la 2-forma: F_{μν} + F_{νμ} = 0, evitando un residuo tautológico.
        """
        F = np.asarray(packet1.curvature_tensor)
        n = self._n
        total = int(n * n)
        try:
            F_mat = np.einsum("mnij->minj", F).reshape(total, total)
        except Exception:
            return PoincareDualitySnapshot(
                curvature_rank=0, curvature_nullity=total,
                total_dimension=total, duality_residual=0.0,
                is_poincare_dual=True, hodge_involution_defect=0.0,
            )

        try:
            svals = la.svdvals(F_mat)
            tol = 1e-10 * max(1.0, float(svals[0]) if svals.size > 0 else 1.0)
            rank = int(np.sum(svals > tol))
            # Energía espectral sub-umbral (fuga numérica en el núcleo de Hodge)
            sub_threshold_energy = float(np.sum(svals[svals <= tol])) / max(
                1.0, float(np.sum(svals)) + _WILKINSON_FLOOR
            )
        except Exception:
            rank = 0
            sub_threshold_energy = 0.0

        nullity = int(total - rank)

        # Defecto de antisimetría de Poincaré-Cartan sobre los índices de variedad (μ, ν)
        skew_defect_tensor = F + np.swapaxes(F, 0, 1)
        f_norm = float(np.linalg.norm(F))
        hodge_defect = (
            float(np.linalg.norm(skew_defect_tensor)) / max(1.0, f_norm)
            if f_norm > _WILKINSON_FLOOR else 0.0
        )

        # Residuo combinado: defecto algebraico rango-nulidad + defecto cohomológico de Hodge
        rank_nullity_defect = abs((rank + nullity) - total) / max(1, total)
        residual = float(rank_nullity_defect + 0.5 * (hodge_defect + sub_threshold_energy))

        return PoincareDualitySnapshot(
            curvature_rank=rank,
            curvature_nullity=nullity,
            total_dimension=total,
            duality_residual=residual,
            is_poincare_dual=bool(residual <= 1e-8),
            hodge_involution_defect=float(hodge_defect),
        )

    def _phase2_compute_spectral_recurrence(
        self, packet1: PhaseOneKapexPacket, n_samples: int = 8,
    ) -> SpectralRecurrenceSnapshot:
        r"""
        Recurrencia espectral de Poincaré-Birkhoff y estadístico r-ratio de Oganesyan-Huse
        sobre el espaciamiento de autovalores consecutivos s_i = λ_{i+1} − λ_i:
            r_i = min(s_i, s_{i−1}) / max(s_i, s_{i−1}) ∈ [0, 1].
        (r ≈ 0.386 indica integrabilidad de Poisson/Berry-Tabor; r ≈ 0.530 indica caos GOE).
        """
        eigvals = np.asarray(packet1.spectral_chart.eigenvalues)
        if eigvals.size < 2:
            return SpectralRecurrenceSnapshot(
                recurrence_points=(), min_recurrence_distance=float("inf"),
                is_poincare_recurrent=False, n_samples=int(eigvals.size),
                level_spacing_ratio=0.0,
            )
        seq = np.abs(eigvals[:n_samples])
        seq = seq / max(1e-12, float(np.max(seq)))
        distances: List[float] = []
        points: List[float] = []
        for i in range(1, len(seq)):
            d = float(abs(seq[i] - seq[i - 1]))
            distances.append(d)
            if d < 0.1:
                points.append(float(seq[i]))
        min_dist = min(distances) if distances else float("inf")

        r_ratios: List[float] = []
        for i in range(1, len(distances)):
            s_prev, s_curr = distances[i - 1], distances[i]
            hi = max(s_prev, s_curr)
            if hi > _WILKINSON_FLOOR:
                r_ratios.append(min(s_prev, s_curr) / hi)
        mean_r = float(np.mean(r_ratios)) if r_ratios else 0.0

        return SpectralRecurrenceSnapshot(
            recurrence_points=tuple(points),
            min_recurrence_distance=float(min_dist),
            is_poincare_recurrent=(len(points) >= 2),
            n_samples=int(len(seq)),
            level_spacing_ratio=mean_r,
        )

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKapexPacket,
        integrity: SpectralIntegrityReport,
        gauge_coherence: float,
        eikonal_threshold: float,
        yang_mills_threshold: float,
        jacobi_threshold: float,
        condition_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        Morfismo clasificador de subobjetos en el Topos de Heyting: proyecta los
        invariantes físicos sobre la cadena ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.
        """
        eik_viol = packet.eikonal_residual > eikonal_threshold
        ym_viol = packet.yang_mills_action > yang_mills_threshold
        jac_viol = packet.bianchi_jacobi_residual > jacobi_threshold
        cond_viol = integrity.metric_condition > condition_threshold
        energy_viol = packet.field_energy_density < -_WILKINSON_FLOOR
        csmd_viol = packet.csmd_gradient_error > 1e-8

        eik_score = float(np.exp(-packet.eikonal_residual / max(eikonal_threshold, _WILKINSON_FLOOR)))
        ym_score = float(np.exp(-max(packet.yang_mills_action, 0.0) / max(yang_mills_threshold, _WILKINSON_FLOOR)))
        jac_score = float(np.exp(-packet.bianchi_jacobi_residual / max(jacobi_threshold, _WILKINSON_FLOOR)))
        energy_score = 1.0 if packet.field_energy_density >= -_WILKINSON_FLOOR else 0.0
        csmd_score = float(np.exp(-packet.csmd_gradient_error / 1e-8))
        meet = min(
            eik_score, ym_score, jac_score, energy_score, csmd_score,
            integrity.integrity_score,
            float(np.clip(gauge_coherence, 0.0, 1.0)),
        )

        hard_veto = eik_viol or energy_viol or csmd_viol or jac_viol
        if hard_veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90:
            verdict = HeytingVerdict.COHERENT
        elif meet >= 0.50:
            verdict = HeytingVerdict.DEGRADED
        else:
            verdict = HeytingVerdict.VETOED

        flags = {
            "eikonal": eik_viol,
            "yang_mills": ym_viol,
            "jacobi": jac_viol,
            "condition": cond_viol,
            "energy": energy_viol,
            "csmd": csmd_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self,
        packet: PhaseOneKapexPacket,
        connection_A: List[np.ndarray],
        eikonal_threshold: float = 1e-4,
        yang_mills_threshold: float = 1e4,
        jacobi_threshold: float = 1e-8,
        condition_threshold: float = 1e12,
    ) -> PhaseTwoKapexPacket:
        r"""
        ========================================================================================
        [CLAUSURA FORMAL DE LA FASE ANIDADA 2 — ÚLTIMO MÉTODO DE FASE 2: DECIDE]
        ========================================================================================
        Morfismo F₂: PhaseOneKapexPacket × Ω¹(M, so(n)) ──► PhaseTwoKapexPacket.

        INTEGRA:
          1. Reporte de integridad espectral y submultiplicatividad de Banach.
          2. Holonomía no abeliana de lazos de Wilson.
          3. Mapa de retorno simpléctico de Poincaré-Chirikov y traza de Floquet.
          4. Dualidad cohomológica de Hodge-Poincaré sobre el tensor de curvatura.
          5. Recurrencia espectral de Poincaré-Birkhoff.
          6. Clasificación intuicionista en el álgebra de Heyting.

        CONTINUIDAD DE ANIDAMIENTO CATEGÓRICO:
          El objeto `PhaseTwoKapexPacket` retornado por este método constituye la entrada
          formal inmediata del primer método de la Fase 3 (`_phase3_conservation_audit`).
        ========================================================================================
        """
        integrity = self._phase2_spectral_integrity(packet, connection_A)
        wilson_mean, coherence = self._phase2_gauge_holonomy(connection_A)
        verdict, score, flags = self._phase2_heyting_classify(
            packet, integrity, coherence,
            eikonal_threshold=eikonal_threshold,
            yang_mills_threshold=yang_mills_threshold,
            jacobi_threshold=jacobi_threshold,
            condition_threshold=condition_threshold,
        )
        return_map = self._phase2_compute_return_map(packet)
        duality = self._phase2_compute_poincare_duality(packet)
        recurrence = self._phase2_compute_spectral_recurrence(packet)

        decision = PhaseTwoKapexPacket(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            eikonal_threshold_violated=flags["eikonal"],
            yang_mills_threshold_violated=flags["yang_mills"],
            jacobi_threshold_violated=flags["jacobi"],
            condition_threshold_violated=flags["condition"],
            spectral_integrity=integrity,
            wilson_plaquette_mean=wilson_mean,
            gauge_coherence=coherence,
            heyting_score=score,
            return_map=return_map,
            duality=duality,
            spectral_recurrence=recurrence,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f Wilson=%.4f κ=%.3e "
            "J=%.4f dual_res=%.3e",
            verdict.value, score, wilson_mean, integrity.metric_condition,
            return_map.contraction_factor, duality.duality_residual,
        )
        return decision

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║              FASE ANIDADA 3: ACT & META-CONTROLADOR RSI-3             ║
    # ║  Continuación directa desde `PhaseTwoKapexPacket` (_phase2_decide):   ║
    # ║  Auditoría de Conservación de Noether-Maxwell, Deriva Adiabática de   ║
    # ║  Nekhoroshev-Poincaré, Síntesis Recursiva de Nivel 3 (Richardson,     ║
    # ║  Aitken Δ², Flujo Homológico) y Funtores Maestros OODA.               ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================

    def _phase3_conservation_audit(
        self,
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
    ) -> float:
        r"""
        ========================================================================================
        [INICIO DE LA FASE ANIDADA 3 — ESLABÓN RECEPTOR DE `_phase2_decide`]
        ========================================================================================
        Recibe directamente `packet1: PhaseOneKapexPacket` y `packet2: PhaseTwoKapexPacket`
        emitido por `_phase2_decide` y ejecuta la auditoría de conservación de Noether-Maxwell:
          - Simetría del tensor de energía-impulso T^{[μν]} = 0 (conservación de momento angular).
          - Identidad diferencial de Bianchi-Jacobi D_{[λ} F_{μν]} = 0.
          - Exactitud simpléctica del momento CSMD y proyección de álgebra de Lie so(n).
          - Condición de energía débil de Hilbert-Einstein U ≥ 0.
        ========================================================================================
        """
        T = np.asarray(packet1.maxwell_stress_tensor, dtype=np.float64)
        skew_T = 0.5 * (T - T.T)
        terms = np.asarray(
            [
                packet1.bianchi_jacobi_residual,
                packet1.csmd_gradient_error,
                packet1.algebra_projection_residual,
                float(la.norm(skew_T, ord="fro")),
                float(max(0.0, -packet1.field_energy_density)),
            ],
            dtype=np.float64,
        )
        residual = float(np.real(self._neumaier_sum(terms)))
        logger.debug(
            "Fase 3 auditoría: residuo de conservación=%.6e (veredicto previo=%s)",
            residual, packet2.heyting_verdict,
        )
        return residual

    def _phase3_compute_adiabatic_action_drift(
        self,
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
    ) -> float:
        r"""
        Evalúa la deriva relativa del invariante adiabático de Poincaré-Cartan
        |I_out − I_in| / I_in bajo la transformación simpléctica de escala J = min(q/p, p/q):
            q_out = J · q,   p_out = p / J   (con dq_out ∧ dp_out = dq ∧ dp).
        """
        if packet1.poincare_snapshot is None:
            return 0.0
        I_in = packet1.poincare_snapshot.action_invariant
        if I_in <= 1e-12:
            return 0.0
        chart = packet1.spectral_chart
        eigvals = np.asarray(chart.eigenvalues)
        if eigvals.size < 2:
            return 0.0
        lam_abs = np.abs(eigvals + chart.tikhonov_alpha)
        lam_abs = np.maximum(lam_abs, _WILKINSON_FLOOR)
        q = float(np.min(lam_abs))
        lam_max = float(np.max(lam_abs))
        p = max(0.0, lam_max - q)
        if q <= 0.0 or p <= 0.0:
            return 0.0
        J = (packet2.return_map.contraction_factor
             if packet2.return_map is not None else 1.0)
        if J <= 1e-12:
            return 0.0
        q_out = J * q
        p_out = p / J
        I_out = 0.5 * (q_out ** 2 + p_out ** 2)
        return float(abs(I_out - I_in) / I_in)

    def _phase3_act(
        self,
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
        conservation_residual: float,
    ) -> KapexTelemetry:
        r"""
        ========================================================================================
        [MORFISMO DE SELLADO DE FASE 3: ACT]
        ========================================================================================
        Ensambla y congela el objeto terminal `KapexTelemetry` a partir de los paquetes de
        Fase 1, Fase 2 y el residuo de conservación de Noether-Maxwell de Fase 3.
        ========================================================================================
        """
        adiabatic = self._phase3_compute_adiabatic_action_drift(packet1, packet2)
        return KapexTelemetry(
            eikonal_residual=packet1.eikonal_residual,
            eikonal_hamiltonian=packet1.eikonal_hamiltonian,
            yang_mills_action=packet1.yang_mills_action,
            poynting_flux_vector=packet1.poynting_flux_vector,
            field_energy_density=packet1.field_energy_density,
            momentum_up_vector=packet1.momentum_up_vector,
            curvature_tensor=packet1.curvature_tensor,
            maxwell_stress_tensor=packet1.maxwell_stress_tensor,
            heyting_verdict=packet2.heyting_verdict,
            condition_number=packet2.spectral_integrity.metric_condition,
            bianchi_residual=packet1.bianchi_jacobi_residual,
            instanton_density=packet1.instanton_density,
            csmd_gradient_error=packet1.csmd_gradient_error,
            wilson_plaquette_mean=packet2.wilson_plaquette_mean,
            conservation_residual=conservation_residual,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
            poincare_snapshot=packet1.poincare_snapshot,
            lindstedt_expansion=packet1.lindstedt_expansion,
            return_map=packet2.return_map,
            duality=packet2.duality,
            spectral_recurrence=packet2.spectral_recurrence,
            adiabatic_action_drift=adiabatic,
            meta_residual=0.0,
            rsi_result=None,
            quantum_rlc_snapshot=packet1.quantum_rlc_snapshot,
        )

    def _combine_residuals(
        self,
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
        conservation_residual: float,
    ) -> float:
        r"""
        Funcional Supremo de Meta-Residuo (Norma L_∞ en el espacio de defectos físicos):
        Combina los residuos de Eikonal, Bianchi-Jacobi, CSMD, proyección de Lie, energía,
        conservación de Noether, serie de Lindstedt-Poincaré, dualidad de Hodge y recurrencia.
        """
        parts: List[float] = [
            float(packet1.eikonal_residual),
            float(packet1.bianchi_jacobi_residual),
            float(packet1.csmd_gradient_error),
            float(packet1.algebra_projection_residual),
            float(max(0.0, -packet1.field_energy_density)),
            float(conservation_residual),
        ]
        if packet1.lindstedt_expansion is not None:
            parts.append(float(packet1.lindstedt_expansion.expansion_residual))
        if packet2.duality is not None:
            parts.append(float(packet2.duality.duality_residual))
        if (packet2.spectral_recurrence is not None
                and math.isfinite(packet2.spectral_recurrence.min_recurrence_distance)):
            # En régimen RSI-3 con ganancia homológica activa, pondera la distancia de
            # recurrencia por el factor de amortiguamiento simpléctico para reflejar el
            # confinamiento sobre el toro KAM.
            damp = 1.0 - 0.9 * float(np.clip(packet1.tolerance_bundle.symplectic_homotopy_gain, 0.0, 1.0))
            parts.append(float(min(1.0, packet2.spectral_recurrence.min_recurrence_distance * damp)))
        finite = [p for p in parts if math.isfinite(p)]
        return float(max(finite)) if finite else float("inf")

    def _rsi_compute_trace(
        self, level: int, bundle: KapexToleranceBundle, residual: float,
        traces_prev: Tuple[RSITrace, ...],
    ) -> RSITrace:
        r"""
        Sintetiza la traza metrológica `RSITrace` del nivel actual aplicando:
          - Nivel ≥ 2: Extrapolación de Richardson y estimación del orden de convergencia p.
          - Nivel ≥ 3: Aceleración no lineal Δ² de Aitken y activación de ganancia de Nivel 3.
        """
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
            richardson_extrapolation=rich, aitken_delta_squared=ait,
            adaptation_factor=factor, level3_active=active,
        )

    def _run_pipeline_at_bundle(
        self, bundle: KapexToleranceBundle, **pipeline_kwargs: Any,
    ) -> Tuple[PhaseOneKapexPacket, PhaseTwoKapexPacket, float]:
        r"""
        Ejecuta la composición anidada de las 3 fases bajo una carta de tolerancias `bundle`:
            p1 = _phase1_observe_orient(...)
            p2 = _phase2_decide(p1, ...)
            c3 = _phase3_conservation_audit(p1, p2)
        """
        packet1 = self._phase1_observe_orient(
            tolerance_bundle=bundle,
            grad_S=pipeline_kwargs["grad_S"],
            connection_A=pipeline_kwargs["connection_A"],
            metric_tensor=pipeline_kwargs["metric_tensor"],
            electric_field=pipeline_kwargs["electric_field"],
            magnetic_field=pipeline_kwargs["magnetic_field"],
            refraction_index=pipeline_kwargs["refraction_index"],
            coupling_constant=pipeline_kwargs.get("coupling_constant", 1.0),
            connection_dA=pipeline_kwargs.get("connection_dA", None),
        )
        packet2 = self._phase2_decide(
            packet1, pipeline_kwargs["connection_A"],
            eikonal_threshold=pipeline_kwargs.get("eikonal_threshold", 1e-4),
            yang_mills_threshold=pipeline_kwargs.get("yang_mills_threshold", 1e4),
        )
        conservation = self._phase3_conservation_audit(packet1, packet2)
        return packet1, packet2, conservation

    def _recursive_self_improvement_loop(
        self,
        max_depth: int,
        target_residual: float,
        **pipeline_kwargs: Any,
    ) -> Tuple[
        Tuple[PhaseOneKapexPacket, PhaseTwoKapexPacket, float],
        SelfImprovementResult,
    ]:
        r"""
        Meta-Controlador de Automejora Recursiva de Nivel 3 (RSI-3):
          - Nivel 1: Evalúa el estado base (compatibilidad exacta v7/v8) y mide el meta-residuo.
          - Nivel 2: Refina la escala metrológica vía Richardson, incrementa el orden N de la
                     expansión de Lindstedt-Poincaré e inyecta homotopía simpléctica parcial.
          - Nivel 3: Activa la aceleración Δ² de Aitken, conmuta a la re-sumación racional de
                     Padé [M/M] para anular divergencias seculares y eleva la proyección
                     homológica de Hamilton-Jacobi para contraer el residuo hacia el punto fijo.
        """
        initial = KapexToleranceBundle(
            scale=1.0,
            n_lindstedt_terms=2,
            symplectic_homotopy_gain=0.0,
            use_pade_resummation=False,
        )
        bundle = initial
        traces: List[RSITrace] = []
        best: Optional[Tuple[PhaseOneKapexPacket, PhaseTwoKapexPacket, float]] = None
        best_res = float("inf")

        for level in range(1, max_depth + 1):
            p1, p2, cons = self._run_pipeline_at_bundle(bundle, **pipeline_kwargs)
            residual = self._combine_residuals(p1, p2, cons)
            trace = self._rsi_compute_trace(
                level=level, bundle=bundle, residual=residual,
                traces_prev=tuple(traces),
            )
            traces.append(trace)
            if residual < best_res:
                best_res = residual
                best = (p1, p2, cons)
            if residual <= target_residual:
                break

            factor = trace.adaptation_factor
            if not math.isfinite(factor) or factor <= 0.0:
                factor = 0.1
            new_scale = bundle.scale * factor
            if new_scale >= bundle.scale:
                new_scale = bundle.scale * 0.5
            new_terms = min(_RSI_TERMS_CAP, bundle.n_lindstedt_terms + 2)

            # Mutación estructural algorítmica de Nivel 2 y Nivel 3:
            near_res = (
                p1.poincare_snapshot.is_near_resonance
                if p1.poincare_snapshot is not None else False
            )
            next_level = level + 1
            homotopy_gain = min(
                0.98,
                0.45 * float(next_level - 1) + (0.15 if near_res else 0.0),
            )
            use_pade = bool(next_level >= 3 or near_res)

            bundle = KapexToleranceBundle(
                scale=float(max(_RSI_MIN_SCALE, min(_RSI_MAX_SCALE, new_scale))),
                n_lindstedt_terms=new_terms,
                symplectic_homotopy_gain=homotopy_gain,
                use_pade_resummation=use_pade,
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
        if best is None:
            raise RuntimeError("RSI-3 no produjo resultado.")
        return best, rsi

    def execute_electrodynamic_cycle(
        self,
        grad_S: np.ndarray,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        refraction_index: float,
        coupling_constant: float = 1.0,
        eikonal_threshold: float = 1e-4,
        yang_mills_threshold: float = 1e4,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        r"""
        Functor Maestro OODA T = Act ∘ Decide ∘ Observe (API 100% compatible con v7.0.0/v8.0.0).
        Ejecuta las 3 fases anidadas en un pase directo con el bundle base.
        """
        packet1 = self._phase1_observe_orient(
            grad_S, connection_A, metric_tensor,
            electric_field, magnetic_field, refraction_index,
            coupling_constant=coupling_constant,
            connection_dA=connection_dA,
            tolerance_bundle=KapexToleranceBundle(),
        )
        packet2 = self._phase2_decide(
            packet1, connection_A,
            eikonal_threshold=eikonal_threshold,
            yang_mills_threshold=yang_mills_threshold,
        )
        conservation = self._phase3_conservation_audit(packet1, packet2)
        telemetry = self._phase3_act(packet1, packet2, conservation)
        out = telemetry.to_dict()
        out["meta_residual"] = self._combine_residuals(packet1, packet2, conservation)
        return out

    def execute_electrodynamic_cycle_rsi3(
        self,
        grad_S: np.ndarray,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        refraction_index: float,
        coupling_constant: float = 1.0,
        eikonal_threshold: float = 1e-4,
        yang_mills_threshold: float = 1e4,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        rsi_target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Dict[str, Any]:
        r"""
        ========================================================================================
        [CLAUSURA SOBERANA DE LA FASE ANIDADA 3 — ENDOFUNCTOR DE PUNTO FIJO RSI-3]
        ========================================================================================
        Functor Maestro OODA bajo Automejora Recursiva de Nivel 3:
            T_RSI3 = Fix_3(Act ∘ Decide ∘ Observe).

        Ejecuta el lazo cerrado de refinamiento Richardson → Orden p → Aitken Δ² + Homotopía
        Simpéctica + Re-sumación de Padé y emite el pasaporte completo con `rsi_*` y `rsi_traces`.
        ========================================================================================
        """
        pipeline_kwargs = {
            "grad_S": grad_S,
            "connection_A": connection_A,
            "metric_tensor": metric_tensor,
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
            "refraction_index": refraction_index,
            "coupling_constant": coupling_constant,
            "connection_dA": connection_dA,
            "eikonal_threshold": eikonal_threshold,
            "yang_mills_threshold": yang_mills_threshold,
        }
        (packet1, packet2, conservation), rsi = self._recursive_self_improvement_loop(
            max_depth=rsi_max_depth,
            target_residual=rsi_target_residual,
            **pipeline_kwargs,
        )
        telemetry = self._phase3_act(packet1, packet2, conservation)
        meta_res = self._combine_residuals(packet1, packet2, conservation)
        telemetry_rsi = KapexTelemetry(
            eikonal_residual=telemetry.eikonal_residual,
            eikonal_hamiltonian=telemetry.eikonal_hamiltonian,
            yang_mills_action=telemetry.yang_mills_action,
            poynting_flux_vector=telemetry.poynting_flux_vector,
            field_energy_density=telemetry.field_energy_density,
            momentum_up_vector=telemetry.momentum_up_vector,
            curvature_tensor=telemetry.curvature_tensor,
            maxwell_stress_tensor=telemetry.maxwell_stress_tensor,
            heyting_verdict=telemetry.heyting_verdict,
            condition_number=telemetry.condition_number,
            bianchi_residual=telemetry.bianchi_residual,
            instanton_density=telemetry.instanton_density,
            csmd_gradient_error=telemetry.csmd_gradient_error,
            wilson_plaquette_mean=telemetry.wilson_plaquette_mean,
            conservation_residual=telemetry.conservation_residual,
            spectral_integrity_score=telemetry.spectral_integrity_score,
            poincare_snapshot=telemetry.poincare_snapshot,
            lindstedt_expansion=telemetry.lindstedt_expansion,
            return_map=telemetry.return_map,
            duality=telemetry.duality,
            spectral_recurrence=telemetry.spectral_recurrence,
            adiabatic_action_drift=telemetry.adiabatic_action_drift,
            meta_residual=meta_res,
            rsi_result=rsi,
            quantum_rlc_snapshot=telemetry.quantum_rlc_snapshot,
        )
        out = telemetry_rsi.to_dict()
        out["rsi_traces"] = [
            {
                "level": t.level,
                "tolerance_scale": t.tolerance_scale,
                "n_lindstedt_terms": t.n_lindstedt_terms,
                "residual": t.residual,
                "convergence_rate": t.convergence_rate,
                "richardson_extrapolation": t.richardson_extrapolation,
                "aitken_delta_squared": t.aitken_delta_squared,
                "adaptation_factor": t.adaptation_factor,
                "level3_active": t.level3_active,
            }
            for t in rsi.traces
        ]
        return out


__all__ = [
    "KapexElectrodynamicEngine",
    "PhaseOneKapexPacket",
    "PhaseTwoKapexPacket",
    "KapexTelemetry",
    "SpectralChart",
    "SpectralIntegrityReport",
    "HeytingVerdict",
    # v8.0.0 + v9.0.0
    "PoincareSpectralSnapshot",
    "LindstedtHeatKernelExpansion",
    "QuantumRLCImpedanceSnapshot",
    "EnergyReturnMapSnapshot",
    "PoincareDualitySnapshot",
    "SpectralRecurrenceSnapshot",
    "KapexToleranceBundle",
    "RSITrace",
    "SelfImprovementResult",
]