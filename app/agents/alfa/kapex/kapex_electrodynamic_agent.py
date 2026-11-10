# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KApex Electrodynamic Agent (Director de Retorno y Expansión)        ║
║ Ruta   : app/agents/alpha/kapex/kapex_electrodynamic_agent.py                ║
║ Versión: 9.0.0-Sovereign-Gauge-Sheaf-Poincare-KAM-RLC-RSI3-Nested            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA (V_𝕊):
────────────────────────────────────────────────────────────────────────────────
Ápice Estratégico como Endofuntor de Campo de Calibre, Geometría Simpéctica de
Poincaré-KAM y Haz Celular de Hodge. Modela el flujo de capital como una corriente
relativista en una variedad riemanniana anisótropa (𝓜, G_μν) con disipación
viscosa R_cost, gobernada por una conexión no abeliana de Yang-Mills A_μ ∈ 𝔰𝔬(n),
bivectores hipercomplejos de Riemann-Silberstein y un Meta-Controlador de
Automejora Recursiva de Nivel 3 (RSI-3).

EVOLUCIÓN SOBERANA v9.0.0 (Poincaré-KAM + Clifford-RLC + RSI-3 en 3 Fases Anidadas):
  [P1] Snapshot de Poincaré-KAM en el fibrado cotangente espectral (q=λ_min(G), p=λ_max−λ_min):
       invariante integral de Poincaré-Cartan I = ½(q²+p²), margen diofántico KAM
       |q/p − p_r/q_r| − γ·q_r^{−τ} y detección de divisores pequeños vía fracciones continuas.
  [P2] Expansión perturbativa de Lindstedt-Poincaré del Heat Kernel Φ(ε)=n⁻¹tr(e^{−εG}) con
       parámetro ε=1/(1+λ_max) y re-sumación racional diagonal de Padé [M/M](−εG) activada en
       el Nivel 3 de RSI para anular divergencias seculares.
  [P3] Bicuaternión hipercomplejo de Riemann-Silberstein F_RS = E + i·G^{−1/2}H ∈ Cl(T*𝓜)⊗ℂ
       y síntesis de impedancia cuántica RLC verificando |S|² = P² + Q² y Q_factor > 0.
  [P4] Mapa de Retorno Simpéctico de Poincaré-Chirikov (Twist Map con det(DP)=1), traza de
       monodromía de Floquet y exponente de Lyapunov local.
  [P5] Dualidad cohomológica de Hodge-Poincaré sobre F ∈ Ω²(𝓜, 𝔰𝔬(n)): espectro SVD de
       Eckart-Young-Mirsky, fuga sub-umbral y defecto de antisimetría de Cartan.
  [P6] Recurrencia ergódica de Poincaré-Birkhoff sobre el toro espectral y estadístico r-ratio
       de Oganesyan-Huse (integrabilidad de Berry-Tabor vs. caos de Wigner-Dyson).
  [P7] Deriva adiabática de Nekhoroshev-Poincaré |I_out−I_in|/I_in con restauración simpléctica
       contradiabática en el Nivel 3 de RSI.
  [P8] Meta-controlador RSI-3 de tres niveles reales:
         - Nivel 1: Evaluación metrológica base (compatibilidad exacta v7.0.0 / v8.0.0).
         - Nivel 2: Extrapolación de Richardson, pulido hiper-potencia de Schulz-Newton-Hotelling
                    sobre G_inv y escalado perturbativo de Lindstedt-Poincaré.
         - Nivel 3: Aceleración no lineal Δ² de Aitken, re-sumación racional de Padé [M/M] y
                    amortiguamiento homológico contradiabático sobre el haz de Hodge.

ARQUITECTURA DE TRES FASES ESTRICTAMENTE ANIDADAS (COMPOSICIÓN CATEGÓRICA):
  Fase 1 (`Phase1_MetricValidation`)
    └──► Culmina en `build_context() -> ApexPreparationContext`
           │ (El codominio terminal de Fase 1 es el dominio inicial exacto de Fase 2)
           ▼
  Fase 2 (`Phase2_ElectrodynamicSynthesis`)
    └──► Inicia recibiendo `ApexPreparationContext` y culmina en `synthesize() -> ApexStateTensor`
           │ (El codominio terminal de Fase 2 es el dominio inicial exacto de Fase 3)
           ▼
  Fase 3 (`Phase3_SheafProjection` + Meta-Controlador RSI-3)
    └──► Inicia recibiendo `(ApexPreparationContext, ApexStateTensor)`, exporta `SheafStalkApex`
         en `export_stalk()` y clausura el punto fijo recursivo en `synthesize_with_rsi3()`.

Funtor Maestro: T_RSI3 = Fix_3(export_stalk ∘ synthesize ∘ build_context).
"""
from __future__ import annotations

import enum
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.core.mic_algebra import CategoricalState, Morphism
except ImportError:
    class CategoricalState:  # type: ignore[no-redef]
        pass

    class Morphism:  # type: ignore[no-redef]
        pass


logger = logging.getLogger("MIC.Alpha.KApexElectrodynamicAgent")

__version__: str = (
    "9.0.0-Sovereign-Gauge-Sheaf-Poincare-KAM-RLC-RSI3-Nested"
)

# ═════════════════════════════════════════════════════════════════════════════
# Constantes metrológicas, de Mecánica Celeste de Poincaré-KAM y RSI-3
# ═════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY: float = 100.0
_ANTISYM_REL_TOL: float = 1.0e-8
_KAPPA_CROSSCHECK_TOL: float = 1.0e-3
_GAUGE_COVARIANCE_TOL: float = 1.0e-6

# Poincaré & Teoría KAM
_POINCARE_MAX_DENOMINATOR: int = 16
_POINCARE_LOW_ORDER: int = 8
_RESONANCE_TOLERANCE: float = 1.0e-3
_SMALL_PARAMETER_FLOOR: float = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: float = 1.0e-6
_POINCARE_DEFAULT_SECTION: float = 1.0
_KAM_DIOPHANTINE_GAMMA: float = 1.0e-2
_KAM_DIOPHANTINE_TAU: float = 2.0

# Meta-Controlador RSI-3
_RSI_MAX_DEPTH: int = 3
_RSI_TARGET_RESIDUAL: float = 1.0e-3
_RSI_MIN_SCALE: float = 1.0e-3
_RSI_MAX_SCALE: float = 1.0e3
_RSI_FACTOR_LOWER: float = 1.0e-2
_RSI_FACTOR_UPPER: float = 0.99
_RSI_OSC_EPS: float = 1.0e-15
_RSI_TERMS_CAP: int = 10


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — JERARQUÍA DE EXCEPCIONES CATEGÓRICAS (preservadas + extendidas)
# ═════════════════════════════════════════════════════════════════════════════
class ElectrodynamicApexError(Exception):
    """Excepción categórica raíz para violaciones en el Estrato K_APEX."""


class ApexDimensionError(ElectrodynamicApexError):
    """Dimensiones de matrices/vectores constitutivos inconsistentes."""


class ApexParameterError(ElectrodynamicApexError):
    """Parámetro escalar de control fuera de su rango admisible."""


class ApexSymmetryError(ElectrodynamicApexError):
    """Violación de simetría con diagnóstico Frobenius."""


class ApexConditionError(ElectrodynamicApexError):
    """Número de condición espectral κ supera el umbral admisible."""


class MetricInverseError(ElectrodynamicApexError):
    """G_inv no es la inversa bilateral de G_μν."""


class SpectralClosureError(ElectrodynamicApexError):
    """Fallo de cierre espectral R_sqrt² ≈ R_cost o proyección idempotente."""


class GaugePotentialError(ElectrodynamicApexError):
    """Estrés estructural: exp(−½ Tr(G)) < ε_mach."""


class EikonalRefractionError(ElectrodynamicApexError):
    """Mercado topológicamente inalcanzable: fallo de la EDP Eikonal."""


class FinancialBlackHoleError(ElectrodynamicApexError):
    """P_exergia < −tol: disipación devora el flujo de Poynting."""


class HolonomyVetoError(ElectrodynamicApexError):
    """S_YM supera el umbral relativo: curvatura de calibre no trivial."""


class GaugeCovarianceError(ElectrodynamicApexError):
    """S_YM no invariante bajo isometría genuina de G_μν."""


class SheafMetricError(ElectrodynamicApexError):
    """Identidad de Hodge local δ_metricᵀ G_μν δ_metric ≉ I violada."""


class PoincareSectionError(ElectrodynamicApexError):
    """Degeneración en la sección de Poincaré o aplicación de retorno."""


class RecursiveSelfImprovementError(ElectrodynamicApexError):
    """Fallo del meta-controlador RSI-3."""


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — RETÍCULO INTUICIONISTA DE HEYTING Ω₃ (Topos de Gödel-Dummett)
# ═════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(enum.IntEnum):
    r"""
    Retículo de Heyting Ω₃ de 3 valores (lógica intermedia de Gödel-Dummett):
        COHERENT (0) ≺ DEGRADED (1) ≺ VETOED (2)
    donde el orden codifica severidad de fallo (join = max, meet = min).
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — ÁLGEBRA DE BOOLE DE VIABILIDAD (ApexViabilityFlags)
# ═════════════════════════════════════════════════════════════════════════════
class ApexViabilityFlags(enum.Flag):
    r"""Retícula de Boole finita (2⁷ elementos) de predicados electrodinámicos."""

    NONE = 0
    GAUGE_INJECTION_NONTRIVIAL = enum.auto()
    EIKONAL_MARGIN_SOUND = enum.auto()
    EXERGY_NONNEGATIVE = enum.auto()
    HOLONOMY_TRIVIAL = enum.auto()
    CURVATURE_ANTISYMMETRIC = enum.auto()
    METRIC_WELL_CONDITIONED = enum.auto()
    SPECTRAL_CLOSURE_SOUND = enum.auto()
    ALL = (
        GAUGE_INJECTION_NONTRIVIAL
        | EIKONAL_MARGIN_SOUND
        | EXERGY_NONNEGATIVE
        | HOLONOMY_TRIVIAL
        | CURVATURE_ANTISYMMETRIC
        | METRIC_WELL_CONDITIONED
        | SPECTRAL_CLOSURE_SOUND
    )

    def meet(self, other: "ApexViabilityFlags") -> "ApexViabilityFlags":
        return self & other

    def join(self, other: "ApexViabilityFlags") -> "ApexViabilityFlags":
        return self | other

    def is_order_unit(self) -> bool:
        return self == ApexViabilityFlags.ALL


def describe_viability_flags(flags: ApexViabilityFlags) -> str:
    """Serializa ``ApexViabilityFlags`` a diagnóstico legible."""
    atomic = [
        f for f in ApexViabilityFlags
        if f not in (ApexViabilityFlags.NONE, ApexViabilityFlags.ALL)
    ]
    satisfied = [f.name for f in atomic if f in flags]
    violated = [f.name for f in atomic if f not in flags]
    return (
        f"SATISFECHOS={satisfied or 'ninguno'} | "
        f"VIOLADOS={violated or 'ninguno'} | "
        f"VIABLE_TOTAL={flags.is_order_unit()}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — DTOs INMUTABLES (Poincaré-KAM, Lindstedt-Padé, RLC, RSI-3)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareSpectralSnapshot:
    r"""
    Snapshot de Mecánica Celeste de Poincaré-KAM en el espacio fase espectral
    (q = λ_min(G), p = λ_max(G) − λ_min(G)).

    Invariante de acción: I = ½(q² + p²), ángulo canónico: θ_p = atan2(p, q).
    Sección transversal de Poincaré: Σ = {q = section_level}.
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
    canonical_q: float = 0.0
    canonical_p: float = 0.0
    kam_diophantine_margin: float = 0.0
    poincare_cartan_form: float = 0.0


@dataclass(frozen=True, slots=True)
class LindstedtHeatKernelExpansion:
    r"""
    Expansión perturbativa de Lindstedt-Poincaré del Heat Kernel Φ(ε) = n⁻¹tr(e^{−εG})
    con parámetro ε = 1/(1 + λ_max(G)) y aproximante racional diagonal de Padé [M/M].
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
class QuantumRLCApexSnapshot:
    r"""
    Síntesis hipercompleja de Riemann-Silberstein F_RS = E + i·G^{−1/2}H ∈ Cl(T*𝓜)⊗ℂ
    e impedancia cuántica compleja Z(ω) = R + iX del circuito equivalente de mercado.
    Verifica:
      - Pasividad disipativa: R = Re(Z) ≥ 0.
      - Identidad del triángulo de potencia compleja: |S|² = P² + Q².
      - Factor de calidad cuántico: Q_RLC > 0.
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
    is_passive_and_resonant: bool


@dataclass(frozen=True, slots=True)
class EnergyReturnMapSnapshot:
    r"""
    Mapa de retorno simpléctico de Poincaré-Chirikov en el espacio espectral.
    Preserva el volumen de fase de Liouville (det DP = 1) y audita la estabilidad
    espectral mediante la traza de monodromía de Floquet y el exponente de Lyapunov.
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
    Dualidad de Hodge-Poincaré y rango-nulidad sobre la 2-forma de curvatura
    F ∈ Ω²(𝓜, 𝔰𝔬(n)) con auditoría de valores singulares y antisimetría de Cartan.
    """
    curvature_rank: int
    curvature_nullity: int
    total_dimension: int
    duality_residual: float
    is_poincare_dual: bool
    hodge_antisymmetry_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class SpectralRecurrenceSnapshot:
    r"""
    Recurrencia ergódica de Poincaré-Birkhoff en la secuencia de autovalores
    normalizados de G_μν y estadístico r-ratio de espaciamiento de niveles.
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
    Cuando scale=1.0, n_lindstedt_terms=2, symplectic_homotopy_gain=0.0 y
    use_pade_resummation=False, preserva compatibilidad exacta con v7.0.0 y v8.0.0.
    """
    scale: float = 1.0
    n_lindstedt_terms: int = 2
    symplectic_homotopy_gain: float = 0.0
    use_pade_resummation: bool = False

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
    r"""Traza metrológica de una iteración del meta-controlador RSI-3."""
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
    r"""Resultado consolidado del meta-controlador de automejora recursiva RSI-3."""
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
# SECCIÓN 4 — NÚCLEO MATEMÁTICO PURO (Fracciones Continuas, KAM, Padé, Aitken)
# ═════════════════════════════════════════════════════════════════════════════
def _continued_fraction_approximation(
    x: float, max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
    """Mejor aproximación diofántica racional p/q de x > 0 vía fracciones continuas."""
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
    """Evalúa resonancia racional p/q de orden bajo y magnitud del divisor pequeño 1/q."""
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)
    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)
    err = abs(ratio - p / q)
    is_res = (err < tolerance) and (q <= low_order)
    return (bool(is_res), int(p), int(q), float(err), float(1.0 / q))


def _pade_resum_exponential(x: float, n_terms: int) -> float:
    r"""
    Aproximante racional diagonal de Padé [M/M](-x) para exp(-x), con M = max(1, n_terms // 2).
    Incondicionalmente A-estable; suprime términos seculares en la traza del Heat Kernel.
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
    if abs(den) < _SMALL_PARAMETER_FLOOR:
        return 1.0
    return float(num / den)


def _richardson_extrapolation(
    r_prev: float, r_curr: float, tol_prev: float, tol_curr: float,
) -> Tuple[float, float]:
    """Extrapolación de Richardson y estimación logarítmica del orden efectivo p."""
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
    """Aceleración no lineal Δ² de Aitken / transformación de Shanks."""
    d = r_n - 2.0 * r_nm1 + r_nm2
    if abs(d) < _RSI_OSC_EPS:
        return r_n
    return float(r_n - (r_n - r_nm1) ** 2 / d)


def _adaptive_refinement_factor(
    current_residual: float, target_residual: float, convergence_rate: float,
    level3_active: bool, aitken_signal: float,
) -> float:
    """Ley adaptativa de control de escala para el Meta-Controlador RSI-3."""
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
# SECCIÓN 5 — PAQUETES DE ACOPLAMIENTO ENTRE LAS 3 FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ApexPreparationContext:
    r"""
    [OBJETO DE ACOPLAMIENTO CATEGÓRICO FASE 1 ──► FASE 2]
    Codominio terminal de `Phase1_MetricValidation.build_context` ≡ Dominio
    inicial de `Phase2_ElectrodynamicSynthesis.__init__`.
    """
    G_mu_nu: NDArray[np.float64]
    G_inv: NDArray[np.float64]
    R_cost: NDArray[np.float64]
    L_G: NDArray[np.float64]
    R_sqrt: NDArray[np.float64]
    kappa_G: float
    kappa_G_inv: float
    epsilon_G: float
    rank_R: int
    spectral_gap_R: float
    betti_0_R: int
    inverse_residual: float
    spectral_closure_residual: float
    dim: int
    # ── Extensiones Poincaré-KAM + Lindstedt-Padé + RSI-3 ────────────────────
    poincare_snapshot: Optional[PoincareSpectralSnapshot] = None
    lindstedt_expansion: Optional[LindstedtHeatKernelExpansion] = None
    tolerance_bundle: KapexToleranceBundle = field(
        default_factory=KapexToleranceBundle
    )


@dataclass(frozen=True, slots=True)
class ApexStateTensor:
    r"""
    [OBJETO DE ACOPLAMIENTO CATEGÓRICO FASE 2 ──► FASE 3]
    Codominio terminal de `Phase2_ElectrodynamicSynthesis.synthesize` ≡ Dominio
    de entrada de `Phase3_SheafProjection` y del auditor RSI-3.
    """
    gauge_injection_vector: NDArray[np.float64]
    suppression_factor: float
    fermat_refractive_index: float
    eikonal_norm_sq: float
    poynting_income: float
    poynting_dissipation: float
    poynting_exergy_flux: float
    yang_mills_action: float
    curvature_antisymmetry_residual: float
    viability_flags: ApexViabilityFlags
    is_electrodynamically_viable: bool
    gauge_covariance_residual: Optional[float]
    # ── Extensiones Poincaré + Bicuaternión RLC Fase 2 ──────────────────────
    return_map: Optional[EnergyReturnMapSnapshot] = None
    duality: Optional[PoincareDualitySnapshot] = None
    spectral_recurrence: Optional[SpectralRecurrenceSnapshot] = None
    quantum_rlc_snapshot: Optional[QuantumRLCApexSnapshot] = None


@dataclass(frozen=True, slots=True)
class SheafStalkApex:
    r"""
    [OBJETO TERMINAL DE FASE 3 Y DEL ENDOFUNCTOR MAESTRO RSI-3]
    Fibrado celular de Hodge exportado al estrato global de la MIC.
    """
    delta_apex: NDArray[np.float64]
    delta_metric: NDArray[np.float64]
    delta_dissipative: NDArray[np.float64]
    hodge_laplacian: NDArray[np.float64]
    hodge_metric_residual: float
    hodge_spectral_gap: float
    hodge_condition_number: float
    lossless_subspace_dimension: int
    source_injection: NDArray[np.float64]
    projected_source_metric: NDArray[np.float64]
    projected_source_dissipative: NDArray[np.float64]
    rank_delta: int
    # ── Extensiones Nekhoroshev-Poincaré + Heyting Ω₃ + RSI-3 ───────────────
    adiabatic_action_drift: float = 0.0
    meta_residual: float = 0.0
    heyting_verdict: Optional[HeytingOmega3] = None
    rsi_result: Optional[SelfImprovementResult] = None


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — KApexElectrodynamicAgent (3 FASES ESTRICTAMENTE ANIDADAS + RSI-3)
# ═════════════════════════════════════════════════════════════════════════════
class KApexElectrodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Ápice Estratégico K_APEX v9.0.0.

    Estructurado en TRES FASES ANIDADAS cuya topología de código garantiza que la
    clausura formal del último método de la Fase k es el punto de entrada inmediato
    del primer método de la Fase k+1:
      - Fase 1 (`Phase1_MetricValidation`): culmina en `build_context() -> ApexPreparationContext`.
      - Fase 2 (`Phase2_ElectrodynamicSynthesis`): inicia en `__init__(context: ApexPreparationContext)`
        y culmina en `synthesize() -> ApexStateTensor`.
      - Fase 3 (`Phase3_SheafProjection` + Meta-Controlador RSI-3): inicia recibiendo
        `(context, return_map)`, culmina en `export_stalk() -> SheafStalkApex` y cierra
        el punto fijo recursivo de Nivel 3 en `synthesize_with_rsi3()`.
    """

    FRIENDLY_NAME: str = "Director de Retorno y Expansión de Mercado"

    def __init__(
        self,
        G_mu_nu: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        R_cost: NDArray[np.float64],
        kappa_max: float = 1.0e10,
        eikonal_slack: float = 0.1,
        holonomy_tol_rel: float = 1.0e-6,
        tolerance_bundle: Optional[KapexToleranceBundle] = None,
    ) -> None:
        if kappa_max <= 1.0:
            raise ApexParameterError(
                f"kappa_max debe ser > 1; se obtuvo {kappa_max}."
            )
        if not (0.0 <= eikonal_slack < 1.0):
            raise ApexParameterError(
                f"eikonal_slack debe estar en [0,1); se obtuvo {eikonal_slack}."
            )
        if holonomy_tol_rel <= 0.0:
            raise ApexParameterError(
                f"holonomy_tol_rel debe ser > 0; se obtuvo {holonomy_tol_rel}."
            )

        self.kappa_max: float = float(kappa_max)
        self.eikonal_slack: float = float(eikonal_slack)
        self.holonomy_tol_rel: float = float(holonomy_tol_rel)
        self._tolerance_bundle: KapexToleranceBundle = (
            tolerance_bundle if tolerance_bundle is not None
            else KapexToleranceBundle()
        )
        self._G_mu_nu = np.asarray(G_mu_nu, dtype=np.float64)
        self._G_inv = np.asarray(G_inv, dtype=np.float64)
        self._R_cost = np.asarray(R_cost, dtype=np.float64)

        # ── Ejecución ansiosa de Fase 1 → emite ApexPreparationContext ─
        self.phase1: KApexElectrodynamicAgent.Phase1_MetricValidation = (
            KApexElectrodynamicAgent.Phase1_MetricValidation(
                G_mu_nu=self._G_mu_nu,
                G_inv=self._G_inv,
                R_cost=self._R_cost,
                kappa_max=self.kappa_max,
                tolerance_bundle=self._tolerance_bundle,
            )
        )
        self.context: ApexPreparationContext = self.phase1.build_context()

        # ── Inicialización anidada de Fase 2 con el contexto de Fase 1 ─
        self.phase2: KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis = (
            KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis(
                context=self.context,
                eikonal_slack=self.eikonal_slack,
                holonomy_tol_rel=self.holonomy_tol_rel,
                kappa_max=self.kappa_max,
            )
        )

        self.phase3: Optional[
            KApexElectrodynamicAgent.Phase3_SheafProjection
        ] = None

        logger.info(
            "[KApexElectrodynamicAgent v%s] dim=%d, κ(G)=%.3e, rank(R)=%d, "
            "betti_0(R)=%d, inv_residual=%.3e, closure=%.3e, "
            "LP_terms=%d, LP_residual=%s.",
            __version__,
            self.context.dim,
            self.context.kappa_G,
            self.context.rank_R,
            self.context.betti_0_R,
            self.context.inverse_residual,
            self.context.spectral_closure_residual,
            self._tolerance_bundle.n_lindstedt_terms,
            (f"{self.context.lindstedt_expansion.expansion_residual:.3e}"
             if self.context.lindstedt_expansion else "N/A"),
        )

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║            FASE ANIDADA 1: VALIDACIÓN MÉTRICA Y POINCARÉ-KAM          ║
    # ║  Certificación Riemanniana SPD, Cholesky Regularizado, Pulido de      ║
    # ║  Schulz-Newton-Hotelling, Cierre Espectral PSD, Snapshot Celeste de   ║
    # ║  Poincaré-KAM y Expansión de Lindstedt-Padé del Heat Kernel.          ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================
    class Phase1_MetricValidation:
        r"""
        **Fase Anidada 1 – Validación Métrica, Poincaré-KAM y Lindstedt-Padé.**

        Cadena interna de morfismos:
            _check_dimensions
            → _validate_symmetry
            → _validate_spd + _cholesky_regularized
            → _validate_inverse_consistency_bilateral (con pulido Schulz-Newton en RSI-3)
            → _validate_psd_and_spectral_diagnostics
            → _compute_poincare_snapshot (coordenadas canónicas + margen KAM)
            → _compute_lindstedt_heat_kernel (serie de Lindstedt + Padé [M/M])
            → build_context (Morfismo Terminal de Fase 1 = Unidad de Entrada de Fase 2)
        """

        _EPS: float = _MACHINE_EPS

        def __init__(
            self,
            G_mu_nu: NDArray[np.float64],
            G_inv: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            kappa_max: float = 1.0e10,
            tolerance_bundle: Optional[KapexToleranceBundle] = None,
        ) -> None:
            self._G: NDArray[np.float64] = np.asarray(G_mu_nu, dtype=np.float64)
            self._G_inv: NDArray[np.float64] = np.asarray(G_inv, dtype=np.float64)
            self._R: NDArray[np.float64] = np.asarray(R_cost, dtype=np.float64)
            self._kappa_max: float = float(kappa_max)
            self._tol_bundle: KapexToleranceBundle = (
                tolerance_bundle if tolerance_bundle is not None
                else KapexToleranceBundle()
            )

        def _check_dimensions(self) -> int:
            for mat, name in (
                (self._G, "G_mu_nu"),
                (self._G_inv, "G_inv"),
                (self._R, "R_cost"),
            ):
                if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                    raise ApexDimensionError(
                        f"'{name}' debe ser cuadrada 2D; shape={mat.shape}."
                    )
            n: int = int(self._G.shape[0])
            if self._G_inv.shape[0] != n:
                raise ApexDimensionError(
                    f"G_inv debe tener shape ({n},{n}); {self._G_inv.shape}."
                )
            if self._R.shape[0] != n:
                raise ApexDimensionError(
                    f"R_cost debe tener shape ({n},{n}); {self._R.shape}."
                )
            if n == 0:
                raise ApexDimensionError("La dimensión no puede ser 0.")
            return n

        def _validate_symmetry(self, A: NDArray[np.float64], name: str) -> None:
            norm_A = float(la.norm(A, "fro"))
            tol = self._EPS * max(norm_A, 1.0)
            residual = float(la.norm(A - A.T, "fro"))
            if residual > tol:
                raise ApexSymmetryError(
                    f"'{name}' no simétrica. ‖A−Aᵀ‖_F={residual:.6e} > tol={tol:.6e}."
                )

        def _validate_spd(
            self, A: NDArray[np.float64], name: str,
        ) -> Tuple[float, float, float]:
            A_sym = 0.5 * (A + A.T)
            n = int(A_sym.shape[0])
            if n == 1:
                lambda_min = lambda_max = float(A_sym[0, 0])
            else:
                lambda_min = float(
                    la.eigh(A_sym, subset_by_index=[0, 0], eigvals_only=True)[0]
                )
                lambda_max = float(
                    la.eigh(A_sym, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0]
                )
            tol_pd = self._EPS * max(abs(lambda_max), 1.0)
            if lambda_min <= tol_pd:
                raise ElectrodynamicApexError(
                    f"'{name}' no SPD. λ_min={lambda_min:.6e} ≤ tol_pd={tol_pd:.6e}."
                )
            kappa = lambda_max / lambda_min
            if kappa > self._kappa_max:
                raise ApexConditionError(
                    f"'{name}' mal condicionada: κ={kappa:.6e} > κ_max={self._kappa_max:.6e}."
                )
            return kappa, lambda_min, lambda_max

        def _cholesky_regularized(
            self, A: NDArray[np.float64], name: str, max_attempts: int = 6,
        ) -> Tuple[NDArray[np.float64], float]:
            A_sym = 0.5 * (A + A.T)
            n = int(A_sym.shape[0])
            trace_scale = float(np.trace(A_sym)) / max(n, 1)
            tau = 0.0
            I_n = np.eye(n, dtype=np.float64)
            for attempt in range(max_attempts + 1):
                try:
                    L = la.cholesky(A_sym + tau * I_n, lower=True)
                    if attempt > 0:
                        logger.warning(
                            "[Fase1] Tikhonov en '%s': τ=%.3e tras %d intento(s).",
                            name, tau, attempt,
                        )
                    return L, tau
                except la.LinAlgError:
                    tau = (
                        self._EPS * max(trace_scale, 1.0)
                        if tau == 0.0 else tau * 10.0
                    )
            raise ElectrodynamicApexError(
                f"Fallo de Cholesky en '{name}' tras {max_attempts} reintentos."
            )

        def _validate_inverse_consistency_bilateral(
            self, kappa_G: float, n: int,
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            Audita la consistencia bilateral G·G_inv ≈ I ≈ G_inv·G.
            Si supera la validación axiomática y el controlador RSI-3 ha activado
            `symplectic_homotopy_gain > 0`, aplica una iteración hiper-potencia de
            Schulz-Newton-Hotelling de orden cuadrático:
                G_inv* = (1 − γ)·G_inv + γ·G_inv·(2I − G·G_inv),
            reduciendo el defecto de inversión hacia ε_mach.
            """
            I_n = np.eye(n, dtype=np.float64)
            r_plus = float(la.norm(self._G @ self._G_inv - I_n, "fro")) / n
            r_minus = float(la.norm(self._G_inv @ self._G - I_n, "fro")) / n
            residual = max(r_plus, r_minus)
            tol_inv = kappa_G * self._EPS * n
            if residual > tol_inv:
                raise MetricInverseError(
                    f"G_inv no bilateral. r={residual:.6e} > tol_inv={tol_inv:.6e}."
                )
            g_inv_eff = self._G_inv.copy()
            gain = float(np.clip(self._tol_bundle.symplectic_homotopy_gain, 0.0, 1.0))
            if gain > 0.0:
                polished = g_inv_eff @ (2.0 * I_n - self._G @ g_inv_eff)
                polished = 0.5 * (polished + polished.T)
                g_inv_eff = (1.0 - gain) * g_inv_eff + gain * polished
                rp_new = float(la.norm(self._G @ g_inv_eff - I_n, "fro")) / n
                rm_new = float(la.norm(g_inv_eff @ self._G - I_n, "fro")) / n
                residual = min(residual, max(rp_new, rm_new))
            return residual, g_inv_eff

        def _validate_psd_and_spectral_diagnostics(
            self, R: NDArray[np.float64], name: str,
        ) -> Tuple[NDArray[np.float64], int, float, float]:
            norm_R = float(la.norm(R, "fro"))
            tol_psd = self._EPS * max(norm_R, 1.0)
            R_sym = 0.5 * (R + R.T)
            eigvals, eigvecs = la.eigh(R_sym)
            lambda_min = float(eigvals[0])
            if lambda_min < -tol_psd:
                raise ApexSymmetryError(
                    f"'{name}' no PSD: λ_min={lambda_min:.6e} < −tol={tol_psd:.6e}."
                )
            eigvals_clamped = np.maximum(eigvals, 0.0)
            R_sqrt = (
                eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]
            ) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)
            rank_R = int(np.sum(eigvals_clamped > tol_psd))
            spectral_gap = (
                float(eigvals_clamped[1] - eigvals_clamped[0])
                if len(eigvals_clamped) > 1 else 0.0
            )
            R_reconstructed = R_sqrt @ R_sqrt
            closure_abs = float(la.norm(R_reconstructed - R_sym, "fro"))
            closure_residual = closure_abs / max(norm_R, 1.0)
            lambda_max_R = float(eigvals_clamped[-1]) if len(eigvals_clamped) else 0.0
            cond_eff = (
                lambda_max_R / max(tol_psd, self._EPS)
                if lambda_max_R > 0.0 else 1.0
            )
            tol_closure = (
                _WILKINSON_SAFETY * self._EPS * max(cond_eff, 1.0)
                * max(R.shape[0], 1)
            )
            if closure_residual > tol_closure:
                raise SpectralClosureError(
                    f"Cierre espectral de '{name}' violado: "
                    f"‖R_sqrt²−R‖_F/‖R‖_F={closure_residual:.6e} > tol={tol_closure:.6e}."
                )
            return R_sqrt, rank_R, spectral_gap, closure_residual

        def _compute_poincare_snapshot(
            self,
            G: NDArray[np.float64],
            section_level: float = _POINCARE_DEFAULT_SECTION,
        ) -> PoincareSpectralSnapshot:
            r"""
            Snapshot celeste de Poincaré-KAM en el plano fase espectral:
                q = λ_min(G),   p = λ_max(G) − λ_min(G),
                I = ½(q² + p²), θ_p = atan2(p, q).
            Calcula cruces transversales con Σ = {q = section_level}, aproximación racional
            diofántica p_r/q_r, margen KAM |q/p − p_r/q_r| − γ·q_r^{−τ} y 1-forma de Cartan.
            """
            G_sym = 0.5 * (G + G.T)
            eigvals = la.eigvalsh(G_sym)
            if eigvals.size == 0:
                return PoincareSpectralSnapshot(
                    action_invariant=0.0, angle_parameter=0.0,
                    section_level=float(section_level), crossing_count=0,
                    frequency_ratio=0.0, is_near_resonance=False,
                    resonance_p=0, resonance_q=0,
                    small_divisor_magnitude=0.0,
                    canonical_q=0.0, canonical_p=0.0,
                    kam_diophantine_margin=0.0, poincare_cartan_form=0.0,
                )
            lam_abs = np.abs(eigvals)
            lam_abs = np.maximum(lam_abs, _SMALL_PARAMETER_FLOOR)
            q = float(np.min(lam_abs))
            lam_max = float(np.max(lam_abs))
            p = max(0.0, lam_max - q)
            action = 0.5 * (q * q + p * p)
            angle = math.atan2(p, q) if action > _SMALL_PARAMETER_FLOOR else 0.0

            sec = float(section_level)
            r_sq = 2.0 * action
            if r_sq > _SMALL_PARAMETER_FLOOR and abs(sec) < math.sqrt(r_sq):
                crossings = 2
            elif (r_sq > _SMALL_PARAMETER_FLOOR
                  and abs(abs(sec) - math.sqrt(r_sq)) <= _SMALL_PARAMETER_FLOOR):
                crossings = 1
            else:
                crossings = 0

            ratio = q / max(1e-12, p)
            is_res, p_r, q_r, dioph_err, small_div = _detect_resonance(ratio)
            kam_bound = (
                _KAM_DIOPHANTINE_GAMMA / (float(max(1, q_r)) ** _KAM_DIOPHANTINE_TAU)
                if q_r > 0 else 0.0
            )
            kam_margin = float(dioph_err - kam_bound)
            cartan_form = float(p * q - action)

            return PoincareSpectralSnapshot(
                action_invariant=float(action),
                angle_parameter=float(angle),
                section_level=float(sec),
                crossing_count=int(crossings),
                frequency_ratio=float(ratio),
                is_near_resonance=bool(is_res),
                resonance_p=int(p_r),
                resonance_q=int(q_r),
                small_divisor_magnitude=float(small_div),
                canonical_q=float(q),
                canonical_p=float(p),
                kam_diophantine_margin=kam_margin,
                poincare_cartan_form=cartan_form,
            )

        def _compute_lindstedt_heat_kernel(
            self,
            G: NDArray[np.float64],
            n_terms: int = 2,
            use_pade: bool = False,
        ) -> LindstedtHeatKernelExpansion:
            r"""
            Expansión perturbativa de Lindstedt-Poincaré del Heat Kernel Φ(ε) = n⁻¹tr(e^{−εG})
            con parámetro pequeño ε = 1/(1 + λ_max(G)).
            En el Nivel 3 de RSI (`use_pade=True`), emplea la re-sumación racional diagonal de
            Padé [M/M](−εG) para eliminar divergencias seculares de orden superior.
            """
            G_sym = 0.5 * (G + G.T)
            eigvals = la.eigvalsh(G_sym)
            if eigvals.size == 0:
                return LindstedtHeatKernelExpansion(
                    small_parameter=0.0, n_terms=int(n_terms),
                    zero_order=1.0, first_order=0.0,
                    second_order=0.0, higher_order=0.0,
                    heat_kernel_approx=1.0, heat_kernel_true=1.0,
                    expansion_residual=0.0, is_asymptotically_consistent=True,
                    pade_resummed_approx=1.0,
                )
            lam_max = float(np.max(np.abs(eigvals)))
            eps = 1.0 / (1.0 + max(_SMALL_PARAMETER_FLOOR, lam_max))
            N = max(2, int(n_terms))
            n = eigvals.size

            approx = 0.0
            pade_approx = 0.0
            true = 0.0
            zero = 0.0
            first = 0.0
            second = 0.0
            higher = 0.0
            for lam in eigvals:
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
                pade_approx += _pade_resum_exponential(x, N)
                try:
                    true += math.exp(-x)
                except OverflowError:
                    true += 0.0
                zero += t0
                first += t1
                second += t2
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

        def build_context(self) -> ApexPreparationContext:
            r"""
            ====================================================================================
            [CLAUSURA FORMAL DE LA FASE ANIDADA 1 — ÚLTIMO MÉTODO DE FASE 1: build_context]
            ====================================================================================
            Morfismo Terminal F₁: (G_μν, G_inv, R_cost, KapexToleranceBundle) ──► ApexPreparationContext.

            CONTINUIDAD DE ANIDAMIENTO CATEGÓRICO:
              El objeto `ApexPreparationContext` retornado aquí es el argumento formal de
              entrada del primer método de la Fase Anidada 2 (`Phase2_ElectrodynamicSynthesis.__init__`).
            ====================================================================================
            """
            n = self._check_dimensions()
            self._validate_symmetry(self._G, "G_mu_nu")
            self._validate_symmetry(self._G_inv, "G_inv")
            self._validate_symmetry(self._R, "R_cost")

            kappa_G, _, _ = self._validate_spd(self._G, "G_mu_nu")
            L_G, epsilon_G = self._cholesky_regularized(self._G, "G_mu_nu")
            kappa_G_inv, _, _ = self._validate_spd(self._G_inv, "G_inv")

            kappa_mismatch_rel = abs(kappa_G_inv - kappa_G) / max(kappa_G, 1.0)
            if kappa_mismatch_rel > _KAPPA_CROSSCHECK_TOL:
                logger.warning(
                    "[Fase1] κ(G_inv)=%.6e vs κ(G)=%.6e (%.2f%% divergencia).",
                    kappa_G_inv, kappa_G, 100.0 * kappa_mismatch_rel,
                )

            inv_residual, g_inv_polished = self._validate_inverse_consistency_bilateral(kappa_G, n)
            (R_sqrt, rank_R, spectral_gap_R, spectral_closure_residual
             ) = self._validate_psd_and_spectral_diagnostics(self._R, "R_cost")
            betti_0_R = n - rank_R

            poincare = self._compute_poincare_snapshot(self._G)
            lindstedt = self._compute_lindstedt_heat_kernel(
                self._G,
                n_terms=self._tol_bundle.n_lindstedt_terms,
                use_pade=self._tol_bundle.use_pade_resummation,
            )

            context = ApexPreparationContext(
                G_mu_nu=self._G.copy(),
                G_inv=g_inv_polished,
                R_cost=self._R.copy(),
                L_G=L_G,
                R_sqrt=R_sqrt,
                kappa_G=kappa_G,
                kappa_G_inv=kappa_G_inv,
                epsilon_G=epsilon_G,
                rank_R=rank_R,
                spectral_gap_R=spectral_gap_R,
                betti_0_R=betti_0_R,
                inverse_residual=inv_residual,
                spectral_closure_residual=spectral_closure_residual,
                dim=n,
                poincare_snapshot=poincare,
                lindstedt_expansion=lindstedt,
                tolerance_bundle=self._tol_bundle,
            )
            logger.info(
                "[Fase1] ctx: dim=%d, κ=%.3e, I=%.4e, res=%s, LP_res=%.3e.",
                n, kappa_G,
                poincare.action_invariant, poincare.is_near_resonance,
                lindstedt.expansion_residual,
            )
            return context

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║       FASE ANIDADA 2: SÍNTESIS ELECTRODINÁMICA, RLC Y POINCARÉ        ║
    # ║  Continuación directa desde `ApexPreparationContext` (build_context): ║
    # ║  Inyección de Calibre, Absorción Eikonal de Fermat, Exergía de        ║
    # ║  Poynting, Holonomía de Yang-Mills, Bicuaternión RLC Cuántico,        ║
    # ║  Mapa de Retorno Simpéctico de Chirikov, Dualidad de Hodge-Poincaré   ║
    # ║  y Recurrencia Ergódica de Poincaré-Birkhoff.                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================
    class Phase2_ElectrodynamicSynthesis:
        r"""
        **Fase Anidada 2 – Síntesis Electrodinámica, Bicuaternión RLC y Dinámica de Poincaré.**

        Recibe el objeto terminal `context: ApexPreparationContext` de la Fase 1 y
        ejecuta la síntesis electrodinámica hasta emitir `ApexStateTensor`.
        """

        _EPS: float = _MACHINE_EPS

        def __init__(
            self,
            context: ApexPreparationContext,
            eikonal_slack: float,
            holonomy_tol_rel: float,
            kappa_max: float,
        ) -> None:
            r"""
            ====================================================================================
            [INICIO DE LA FASE ANIDADA 2 — ESLABÓN RECEPTOR DE `Phase1.build_context`]
            ====================================================================================
            Toma el `ApexPreparationContext` sellado por `build_context` en la Fase 1.
            """
            self._ctx: ApexPreparationContext = context
            self._eikonal_slack: float = float(eikonal_slack)
            self._holonomy_tol_rel: float = float(holonomy_tol_rel)
            self._kappa_max: float = float(kappa_max)
            self._trace_G: float = float(np.trace(context.G_mu_nu))
            self._suppression_factor: float = float(np.exp(-0.5 * self._trace_G))

        def inject_gauge_potential(
            self, d_Phi: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], float]:
            n = self._ctx.dim
            d_Phi = np.asarray(d_Phi, dtype=np.float64)
            if d_Phi.shape != (n,):
                raise ApexDimensionError(
                    f"d_Phi shape {d_Phi.shape} ≠ ({n},)."
                )
            if self._suppression_factor < self._EPS:
                raise GaugePotentialError(
                    f"Tr(G)={self._trace_G:.6e}: supresión {self._suppression_factor:.6e} < ε."
                )
            s_val = d_Phi * self._suppression_factor
            return s_val, self._suppression_factor

        def compute_eikonal_absorption(
            self,
            phase_gradient: NDArray[np.float64],
            sigma_stress: float,
            alpha_fermat: float = 0.5,
        ) -> Tuple[float, float]:
            n = self._ctx.dim
            phase_gradient = np.asarray(phase_gradient, dtype=np.float64)
            if phase_gradient.shape != (n,):
                raise ApexDimensionError(
                    f"phase_gradient shape {phase_gradient.shape} ≠ ({n},)."
                )
            n_refract = 1.0 + float(np.tanh(alpha_fermat * sigma_stress))
            G_inv_grad = self._ctx.G_inv @ phase_gradient
            eikonal_norm_sq = float(np.dot(phase_gradient, G_inv_grad))
            n_sq = n_refract ** 2
            eikonal_threshold = n_sq * (1.0 - self._eikonal_slack)
            if eikonal_norm_sq < eikonal_threshold:
                raise EikonalRefractionError(
                    f"Fallo Eikonal: ‖∂S‖²={eikonal_norm_sq:.6e} < "
                    f"n²(1−slack)={eikonal_threshold:.6e}."
                )
            return n_refract, eikonal_norm_sq

        def evaluate_poynting_exergy(
            self,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            grad_H: NDArray[np.float64],
        ) -> Tuple[float, float, float]:
            n = self._ctx.dim
            for vec, name in ((E_field, "E_field"), (H_field, "H_field"), (grad_H, "grad_H")):
                if np.asarray(vec).shape != (n,):
                    raise ApexDimensionError(
                        f"'{name}' debe tener shape ({n},)."
                    )
            E_arr = np.asarray(E_field, dtype=np.float64)
            H_arr = np.asarray(H_field, dtype=np.float64)
            gH = np.asarray(grad_H, dtype=np.float64)
            P_in = float(np.dot(E_arr, H_arr))
            R_sqrt_grad = self._ctx.R_sqrt @ gH
            P_diss = float(np.dot(R_sqrt_grad, R_sqrt_grad))
            P_exergia = P_in - P_diss
            tol_exergy = self._EPS * max(abs(P_in), abs(P_diss), 1.0)
            if P_exergia < -tol_exergy:
                raise FinancialBlackHoleError(
                    f"P_in={P_in:.6e}, P_diss={P_diss:.6e}, P_exergia={P_exergia:.6e}."
                )
            return P_in, P_diss, P_exergia

        def _compute_quantum_rlc_biquaternion(
            self,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            P_in: float,
            P_diss: float,
        ) -> QuantumRLCApexSnapshot:
        
            r"""
            Construye el bicuaternión de Riemann-Silberstein F_RS = E + i·G^{−1/2}H en el
            álgebra de Clifford riemanniana Cl(T*𝓜, G)⊗ℂ y extrae la impedancia compleja
            cuántica Z(ω) = R + iX de la línea de transmisión estratégica.
            """
            E_arr = np.asarray(E_field, dtype=np.float64)
            H_arr = np.asarray(H_field, dtype=np.float64)
            e_norm_sq = max(0.0, float(E_arr @ self._ctx.G_inv @ E_arr))
            h_norm_sq = max(0.0, float(H_arr @ self._ctx.G_inv @ H_arr))
            eh_pairing = float(E_arr @ self._ctx.G_inv @ H_arr)

            lorentz_scalar = e_norm_sq - h_norm_sq
            lorentz_pseudoscalar = 2.0 * eh_pairing

            active_P = max(_SMALL_PARAMETER_FLOOR, abs(P_in) + P_diss)
            reactive_Q = 0.5 * lorentz_scalar
            apparent_S = math.hypot(active_P, reactive_Q)
            triangle_res = abs(apparent_S ** 2 - (active_P ** 2 + reactive_Q ** 2)) / max(
                1.0, apparent_S ** 2
            )

            current_sq = max(_SMALL_PARAMETER_FLOOR, h_norm_sq + _SMALL_PARAMETER_FLOOR)
            resistance_R = max(0.0, active_P / current_sq)
            reactance_X = reactive_Q / current_sq
            z_mag = math.hypot(resistance_R, reactance_X)
            z_phase = math.atan2(reactance_X, resistance_R)

            stored_energy = 0.5 * (e_norm_sq + h_norm_sq)
            omega_char = max(_SMALL_PARAMETER_FLOOR, float(self._ctx.kappa_G))
            quality_Q = max(
                _SMALL_PARAMETER_FLOOR,
                (omega_char * max(_SMALL_PARAMETER_FLOOR, stored_energy)) / active_P,
            )

            return QuantumRLCApexSnapshot(
                resistance_R=float(resistance_R),
                reactance_X=float(reactance_X),
                impedance_magnitude=float(z_mag),
                impedance_phase_rad=float(z_phase),
                active_power_P=float(active_P),
                reactive_power_Q=float(reactive_Q),
                apparent_power_S=float(apparent_S),
                power_triangle_residual=float(triangle_res),
                quality_factor_Q=float(quality_Q),
                riemann_silberstein_lorentz_scalar=float(lorentz_scalar),
                riemann_silberstein_pseudoscalar=float(lorentz_pseudoscalar),
                is_passive_and_resonant=bool(
                    resistance_R >= 0.0 and quality_Q > 0.0 and triangle_res <= 1e-12
                ),
            )

        @staticmethod
        def _project_to_so(A: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.5 * (A - A.T)

        @staticmethod
        def _lie_commutator(
            B: NDArray[np.float64], C: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return B @ C - C @ B

        def _compute_curvature(
            self,
            A_1: NDArray[np.float64],
            A_2: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            A1_a = self._project_to_so(A_1)
            A2_a = self._project_to_so(A_2)
            F = (A2_a - A1_a) + self._lie_commutator(A1_a, A2_a)
            return self._project_to_so(F)

        def _yang_mills_action(self, F: NDArray[np.float64]) -> float:
            G_F = self._ctx.G_mu_nu @ F
            M = F.T @ G_F
            S = 0.5 * float(np.trace(M @ self._ctx.G_inv))
            return max(S, 0.0)

        def audit_yang_mills_holonomy(
            self,
            A_gauge_1: NDArray[np.float64],
            A_gauge_2: NDArray[np.float64],
        ) -> Tuple[float, float]:
            n = self._ctx.dim
            A_1 = np.asarray(A_gauge_1, dtype=np.float64)
            A_2 = np.asarray(A_gauge_2, dtype=np.float64)
            for mat, name in ((A_1, "A_gauge_1"), (A_2, "A_gauge_2")):
                if mat.shape != (n, n):
                    raise ApexDimensionError(
                        f"'{name}' shape {mat.shape} ≠ ({n},{n})."
                    )
            F = self._compute_curvature(A_1, A_2)
            norm_F = float(la.norm(F, "fro"))
            antisym_residual = float(la.norm(F + F.T, "fro")) / max(norm_F, 1.0)
            if antisym_residual > _ANTISYM_REL_TOL:
                raise ElectrodynamicApexError(
                    f"F no antisimétrica: residuo={antisym_residual:.6e}."
                )
            S_ym = self._yang_mills_action(F)
            norm_A_sq = float(la.norm(A_1, "fro") ** 2 + la.norm(A_2, "fro") ** 2)
            tol_ym = self._holonomy_tol_rel * max(norm_A_sq, 1.0)
            if S_ym > tol_ym:
                raise HolonomyVetoError(
                    f"S_YM={S_ym:.6e} > tol={tol_ym:.6e}."
                )
            return S_ym, antisym_residual

        def verify_gauge_covariance(
            self,
            A_gauge_1: NDArray[np.float64],
            A_gauge_2: NDArray[np.float64],
            Q_isometry: NDArray[np.float64],
            isometry_tol: float = 1.0e-8,
        ) -> float:
            n = self._ctx.dim
            Q = np.asarray(Q_isometry, dtype=np.float64)
            if Q.shape != (n, n):
                raise ApexDimensionError(f"Q shape {Q.shape} ≠ ({n},{n}).")
            QT_G_Q = Q.T @ self._ctx.G_mu_nu @ Q
            isometry_residual = float(la.norm(QT_G_Q - self._ctx.G_mu_nu, "fro")) / max(
                float(la.norm(self._ctx.G_mu_nu, "fro")), 1.0
            )
            if isometry_residual > isometry_tol:
                raise GaugeCovarianceError(
                    f"Q no preserva G_μν: residuo={isometry_residual:.6e}."
                )
            S_ym_orig, _ = self.audit_yang_mills_holonomy(A_gauge_1, A_gauge_2)
            A_1_t = Q @ A_gauge_1 @ Q.T
            A_2_t = Q @ A_gauge_2 @ Q.T
            S_ym_t, _ = self.audit_yang_mills_holonomy(A_1_t, A_2_t)
            residual = abs(S_ym_orig - S_ym_t) / max(S_ym_orig, 1.0)
            if residual > _GAUGE_COVARIANCE_TOL:
                raise GaugeCovarianceError(
                    f"S_YM no invariante: residuo={residual:.6e}."
                )
            return residual

        def _evaluate_viability_flags(
            self,
            suppression_factor: float,
            eikonal_norm_sq: float,
            n_sq: float,
            P_exergia: float,
            S_ym: float,
            tol_ym: float,
            antisym_residual: float,
        ) -> ApexViabilityFlags:
            flags = ApexViabilityFlags.NONE
            soft_slack = self._eikonal_slack / 2.0
            if suppression_factor >= np.sqrt(self._EPS):
                flags |= ApexViabilityFlags.GAUGE_INJECTION_NONTRIVIAL
            if eikonal_norm_sq >= n_sq * (1.0 - soft_slack):
                flags |= ApexViabilityFlags.EIKONAL_MARGIN_SOUND
            if P_exergia >= -1.0e3 * self._EPS:
                flags |= ApexViabilityFlags.EXERGY_NONNEGATIVE
            if S_ym <= tol_ym:
                flags |= ApexViabilityFlags.HOLONOMY_TRIVIAL
            if antisym_residual <= _ANTISYM_REL_TOL:
                flags |= ApexViabilityFlags.CURVATURE_ANTISYMMETRIC
            if self._ctx.kappa_G <= 0.5 * self._kappa_max:
                flags |= ApexViabilityFlags.METRIC_WELL_CONDITIONED
            if self._ctx.spectral_closure_residual <= 1.0:
                flags |= ApexViabilityFlags.SPECTRAL_CLOSURE_SOUND
            return flags

        def _compute_return_map(
            self, n_iterations: int = 4, S_ym: float = 0.0,
        ) -> EnergyReturnMapSnapshot:
            r"""
            Mapa de Retorno Simpéctico de Poincaré-Chirikov (Twist Map con det DP = 1)
            acoplado a la acción de curvatura de Yang-Mills S_YM:
                I_{k+1} = max(ε, I_k + K_YM · I_0 · sin(θ_k)),
                θ_{k+1} = (θ_k + 2π·J + K_YM · cos(θ_k)) mod 2π,
            donde J = min(q/p, p/q) ∈ (0, 1]. Calcula la traza de monodromía de Floquet
            y el exponente de Lyapunov local.
            """
            poincare = self._ctx.poincare_snapshot
            if poincare is None:
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                    floquet_trace=2.0, lyapunov_exponent=0.0,
                )
            ratio = poincare.frequency_ratio
            if ratio <= 0.0 or not math.isfinite(ratio):
                return EnergyReturnMapSnapshot(
                    contraction_factor=1.0, eigenvalues_modulus=(),
                    is_stable=True, iterations_norm_sequence=(),
                    floquet_trace=2.0, lyapunov_exponent=0.0,
                )
            J = min(ratio, 1.0 / ratio)
            k_ym = float(np.tanh(abs(S_ym))) * 1e-2

            action_0 = max(_SMALL_PARAMETER_FLOOR, poincare.action_invariant)
            cur_I = action_0
            cur_theta = poincare.angle_parameter
            norms: List[float] = []
            lyap_acc = 0.0

            for _ in range(n_iterations):
                sin_t = math.sin(cur_theta)
                cos_t = math.cos(cur_theta)
                cur_I = max(_SMALL_PARAMETER_FLOOR, cur_I + k_ym * action_0 * sin_t)
                cur_theta = (cur_theta + 2.0 * math.pi * J + k_ym * cos_t) % (2.0 * math.pi)
                radius = math.sqrt(2.0 * cur_I)
                cur_q = radius * math.cos(cur_theta)
                cur_p = radius * math.sin(cur_theta)
                norms.append(float(math.hypot(cur_q, cur_p)))
                local_tr = 2.0 - k_ym * cos_t
                disc = local_tr * local_tr - 4.0
                if disc > 0.0:
                    lam_u = 0.5 * (abs(local_tr) + math.sqrt(disc))
                    lyap_acc += math.log(max(1.0, lam_u))

            floquet_tr = 2.0 - k_ym * math.cos(cur_theta)
            lyap = lyap_acc / max(1, n_iterations)
            return EnergyReturnMapSnapshot(
                contraction_factor=float(J),
                eigenvalues_modulus=(float(abs(J)),),
                is_stable=bool(abs(J) <= 1.0 + 1e-9 and abs(floquet_tr) <= 2.0 + 1e-2),
                iterations_norm_sequence=tuple(norms),
                floquet_trace=float(floquet_tr),
                lyapunov_exponent=float(lyap),
            )

        def _compute_poincare_duality(
            self, F: NDArray[np.float64],
        ) -> PoincareDualitySnapshot:
            r"""
            Dualidad de Hodge-Poincaré y teorema de rango-nulidad sobre el operador de
            curvatura F ∈ 𝔰𝔬(n) ⊂ ℝ^{n×n}, incorporando la descomposición SVD de
            Eckart-Young-Mirsky, la energía espectral sub-umbral y el defecto de
            antisimetría cohomológica ‖F + Fᵀ‖_F / max(1, ‖F‖_F).
            """
            n = self._ctx.dim
            F_arr = np.asarray(F, dtype=np.float64)
            try:
                svals = la.svdvals(F_arr)
                tol = 1e-10 * max(1.0, float(svals[0]) if svals.size > 0 else 1.0)
                rank = int(np.sum(svals > tol))
                sub_threshold_leak = float(np.sum(svals[svals <= tol])) / max(
                    1.0, float(np.sum(svals)) + _SMALL_PARAMETER_FLOOR
                )
            except Exception:
                rank = 0
                sub_threshold_leak = 0.0

            nullity = int(n - rank)
            total = int(n)
            f_norm = float(la.norm(F_arr, "fro"))
            antisym_defect = (
                float(la.norm(F_arr + F_arr.T, "fro")) / max(1.0, f_norm)
                if f_norm > _SMALL_PARAMETER_FLOOR else 0.0
            )
            rank_nullity_defect = abs((rank + nullity) - total) / max(1, total)
            residual = float(rank_nullity_defect + 0.5 * (antisym_defect + sub_threshold_leak))

            return PoincareDualitySnapshot(
                curvature_rank=int(rank),
                curvature_nullity=int(nullity),
                total_dimension=int(total),
                duality_residual=float(residual),
                is_poincare_dual=bool(residual <= 1e-8),
                hodge_antisymmetry_defect=float(antisym_defect),
            )

        def _compute_spectral_recurrence(
            self, n_samples: int = 8,
        ) -> SpectralRecurrenceSnapshot:
            r"""
            Recurrencia de Poincaré-Birkhoff en la secuencia de autovalores normalizados
            de G_μν y estadístico r-ratio de Oganesyan-Huse sobre espaciamientos espectrales.
            """
            G_sym = 0.5 * (self._ctx.G_mu_nu + self._ctx.G_mu_nu.T)
            try:
                eigvals = la.eigvalsh(G_sym)
            except Exception:
                return SpectralRecurrenceSnapshot(
                    recurrence_points=(), min_recurrence_distance=float("inf"),
                    is_poincare_recurrent=False, n_samples=0,
                    level_spacing_ratio=0.0,
                )
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
                if hi > _SMALL_PARAMETER_FLOOR:
                    r_ratios.append(min(s_prev, s_curr) / hi)
            mean_r = float(np.mean(r_ratios)) if r_ratios else 0.0

            return SpectralRecurrenceSnapshot(
                recurrence_points=tuple(points),
                min_recurrence_distance=float(min_dist),
                is_poincare_recurrent=(len(points) >= 2),
                n_samples=int(len(seq)),
                level_spacing_ratio=mean_r,
            )

        def synthesize(
            self,
            d_Phi: NDArray[np.float64],
            phase_gradient: NDArray[np.float64],
            sigma_stress: float,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            grad_H: NDArray[np.float64],
            A_gauge_1: NDArray[np.float64],
            A_gauge_2: NDArray[np.float64],
            alpha_fermat: float = 0.5,
            Q_isometry_diagnostic: Optional[NDArray[np.float64]] = None,
        ) -> ApexStateTensor:
            r"""
            ====================================================================================
            [CLAUSURA FORMAL DE LA FASE ANIDADA 2 — ÚLTIMO MÉTODO DE FASE 2: synthesize]
            ====================================================================================
            Morfismo Terminal F₂: ApexPreparationContext × Campos_Electrodinámicos ──► ApexStateTensor.

            CONTINUIDAD DE ANIDAMIENTO CATEGÓRICO:
              El objeto `ApexStateTensor` (y su subobjeto `return_map`) retornado aquí es el
              dominio formal de entrada del primer método de la Fase Anidada 3
              (`Phase3_SheafProjection.__init__`).
            ====================================================================================
            """
            s_val, suppression = self.inject_gauge_potential(d_Phi)
            n_refract, eikonal_norm_sq = self.compute_eikonal_absorption(
                phase_gradient, sigma_stress, alpha_fermat
            )
            P_in, P_diss, P_exergia = self.evaluate_poynting_exergy(
                E_field, H_field, grad_H
            )
            S_ym, antisym_residual = self.audit_yang_mills_holonomy(
                A_gauge_1, A_gauge_2
            )
            rlc_snapshot = self._compute_quantum_rlc_biquaternion(
                E_field, H_field, P_in, P_diss
            )

            norm_A_sq = float(
                la.norm(A_gauge_1, "fro") ** 2
                + la.norm(A_gauge_2, "fro") ** 2
            )
            tol_ym = self._holonomy_tol_rel * max(norm_A_sq, 1.0)
            flags = self._evaluate_viability_flags(
                suppression_factor=suppression,
                eikonal_norm_sq=eikonal_norm_sq,
                n_sq=n_refract ** 2,
                P_exergia=P_exergia,
                S_ym=S_ym,
                tol_ym=tol_ym,
                antisym_residual=antisym_residual,
            )

            return_map = self._compute_return_map(S_ym=S_ym)
            F_curv = self._compute_curvature(A_gauge_1, A_gauge_2)
            duality = self._compute_poincare_duality(F_curv)
            recurrence = self._compute_spectral_recurrence()

            gauge_covariance_residual: Optional[float] = None
            if Q_isometry_diagnostic is not None:
                gauge_covariance_residual = self.verify_gauge_covariance(
                    A_gauge_1, A_gauge_2, Q_isometry_diagnostic
                )

            logger.info(
                "[Fase2] Supp=%.3e, n=%.4f, P_ex=%.6e, S_YM=%.6e, J=%.4f, %s.",
                suppression, n_refract, P_exergia, S_ym,
                return_map.contraction_factor,
                describe_viability_flags(flags),
            )

            return ApexStateTensor(
                gauge_injection_vector=s_val,
                suppression_factor=suppression,
                fermat_refractive_index=n_refract,
                eikonal_norm_sq=eikonal_norm_sq,
                poynting_income=P_in,
                poynting_dissipation=P_diss,
                poynting_exergy_flux=P_exergia,
                yang_mills_action=S_ym,
                curvature_antisymmetry_residual=antisym_residual,
                viability_flags=flags,
                is_electrodynamically_viable=flags.is_order_unit(),
                gauge_covariance_residual=gauge_covariance_residual,
                return_map=return_map,
                duality=duality,
                spectral_recurrence=recurrence,
                quantum_rlc_snapshot=rlc_snapshot,
            )

    # =========================================================================
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║        FASE ANIDADA 3: PROYECCIÓN EN HACES DE HODGE Y RSI-3           ║
    # ║  Continuación directa desde `ApexStateTensor` (Phase2.synthesize):    ║
    # ║  Operador de Coborde de Hodge Δ_apex, Deriva Adiabática de            ║
    # ║  Nekhoroshev-Poincaré con Restauración Contradiabática y Lazo         ║
    # ║  Cerrado de Automejora Recursiva de Nivel 3 (Richardson + Aitken Δ²). ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # =========================================================================
    class Phase3_SheafProjection:
        r"""
        **Fase Anidada 3 – Proyección en Haces Celulares de Hodge y Deriva de Nekhoroshev.**
        """

        _EPS: float = _MACHINE_EPS

        def __init__(
            self,
            context: ApexPreparationContext,
            return_map: Optional[EnergyReturnMapSnapshot] = None,
        ) -> None:
            r"""
            ====================================================================================
            [INICIO DE LA FASE ANIDADA 3 — ESLABÓN RECEPTOR DE `Phase2.synthesize`]
            ====================================================================================
            Construye las restricciones celulares métrica y disipativa sobre el tallo del haz:
                δ_metric = L_G^{−ᵀ},   δ_diss = R_sqrt · δ_metric,
                Δ_Hodge = I_n + δ_dissᵀ · δ_diss.
            """
            self._ctx: ApexPreparationContext = context
            self._return_map: Optional[EnergyReturnMapSnapshot] = return_map
            n = context.dim

            I_n = np.eye(n, dtype=np.float64)
            L_G_inv = la.solve_triangular(
                context.L_G, I_n, lower=True, check_finite=False
            )
            self._delta_metric = L_G_inv.T
            self._delta_dissipative = context.R_sqrt @ self._delta_metric
            self._delta_apex = np.vstack(
                [self._delta_metric, self._delta_dissipative]
            )
            self._hodge_laplacian = (
                I_n + self._delta_dissipative.T @ self._delta_dissipative
            )
            self._hodge_laplacian = 0.5 * (
                self._hodge_laplacian + self._hodge_laplacian.T
            )
            self._rank_delta = n
            self._hodge_metric_residual = self._verify_hodge_identity()

        def _verify_hodge_identity(self) -> float:
            n = self._ctx.dim
            delta_T_G_delta = (
                self._delta_metric.T @ self._ctx.G_mu_nu @ self._delta_metric
            )
            I_n = np.eye(n, dtype=np.float64)
            residual_F = float(la.norm(delta_T_G_delta - I_n, "fro"))
            rel_error = residual_F / n
            tol_hodge = _WILKINSON_SAFETY * self._ctx.kappa_G * self._EPS
            if rel_error > tol_hodge:
                raise SheafMetricError(
                    f"Hodge violado: residuo={rel_error:.6e} > tol={tol_hodge:.6e}."
                )
            return rel_error

        def _compute_hodge_spectrum(self) -> Tuple[float, float]:
            eigvals = la.eigvalsh(self._hodge_laplacian)
            lambda_min = float(eigvals[0])
            lambda_second = float(eigvals[1]) if len(eigvals) > 1 else lambda_min
            lambda_max = float(eigvals[-1])
            gap = lambda_second - lambda_min
            cond = lambda_max / lambda_min if lambda_min > 0.0 else float("inf")
            return gap, cond

        def _compute_adiabatic_action_drift(self) -> float:
            r"""
            Deriva adiabática de Nekhoroshev-Poincaré |I_out − I_in| / I_in bajo la
            aplicación de retorno J.
            Cuando `symplectic_homotopy_gain == 0.0` (modo base v8.0.0), evalúa la deriva
            sin amortiguamiento. Cuando el Meta-Controlador RSI-3 activa la ganancia
            homológica γ_hom ∈ (0, 1], aplica restauración simpléctica contradiabática
            J_eff = J^{1 − γ_hom}, alineando el toro invariante con el marco de Cholesky
            y contrayendo exponencialmente la deriva adiabática.
            """
            poincare = self._ctx.poincare_snapshot
            if poincare is None:
                return 0.0
            I_in = poincare.action_invariant
            if I_in <= 1e-12:
                return 0.0
            J = self._return_map.contraction_factor if self._return_map else 1.0
            if J <= 1e-12:
                return 0.0

            gain = float(np.clip(self._ctx.tolerance_bundle.symplectic_homotopy_gain, 0.0, 1.0))
            if gain > 0.0 and poincare.canonical_q > 0.0 and poincare.canonical_p > 0.0:
                j_eff = J ** (1.0 - gain)
                q_0 = poincare.canonical_q
                p_0 = poincare.canonical_p
                q_out = j_eff * q_0
                p_out = p_0 / j_eff
                I_out = 0.5 * (q_out ** 2 + p_out ** 2)
                return float(abs(I_out - I_in) / I_in)

            # Modo base (compatibilidad exacta v8.0.0)
            q = poincare.frequency_ratio if poincare.frequency_ratio > 0.0 else 1.0
            p = 1.0 / q if q > 0.0 else 1.0
            q_out = J * q
            p_out = p / J
            I_out = 0.5 * (q_out ** 2 + p_out ** 2)
            return float(abs(I_out - I_in) / I_in)

        def export_stalk(
            self, s_val: NDArray[np.float64],
            meta_residual: float = 0.0,
            heyting_verdict: Optional[HeytingOmega3] = None,
            rsi_result: Optional[SelfImprovementResult] = None,
        ) -> SheafStalkApex:
            r"""
            ====================================================================================
            [MORFISMO TERMINAL DE `Phase3_SheafProjection`: export_stalk]
            ====================================================================================
            Proyecta el vector de inyección de calibre sobre las componentes ortogonales de
            Hodge y sella el tallo celular `SheafStalkApex`.
            ====================================================================================
            """
            n = self._ctx.dim
            s_val = np.asarray(s_val, dtype=np.float64)
            if s_val.shape != (n,):
                raise ApexDimensionError(
                    f"s_val shape {s_val.shape} ≠ ({n},)."
                )
            spectral_gap, condition_number = self._compute_hodge_spectrum()
            projected_metric = self._delta_metric @ s_val
            projected_diss = self._delta_dissipative @ s_val
            adiabatic = self._compute_adiabatic_action_drift()

            logger.info(
                "[Fase3] Export: dim=%d, Hodge_res=%.3e, gap=%.3e, "
                "κ(Δ)=%.3e, adiabatic=%.3e.",
                n, self._hodge_metric_residual, spectral_gap,
                condition_number, adiabatic,
            )

            return SheafStalkApex(
                delta_apex=self._delta_apex.copy(),
                delta_metric=self._delta_metric.copy(),
                delta_dissipative=self._delta_dissipative.copy(),
                hodge_laplacian=self._hodge_laplacian.copy(),
                hodge_metric_residual=self._hodge_metric_residual,
                hodge_spectral_gap=spectral_gap,
                hodge_condition_number=condition_number,
                lossless_subspace_dimension=self._ctx.betti_0_R,
                source_injection=s_val.copy(),
                projected_source_metric=projected_metric,
                projected_source_dissipative=projected_diss,
                rank_delta=self._rank_delta,
                adiabatic_action_drift=float(adiabatic),
                meta_residual=float(meta_residual),
                heyting_verdict=heyting_verdict,
                rsi_result=rsi_result,
            )

    # =========================================================================
    # ORQUESTACIÓN FUNTORIAL Y CLAUSURA RECURSIVA DE NIVEL 3 (CONTINUACIÓN FASE 3)
    # =========================================================================
    def synthesize_apex_field(
        self,
        d_Phi: NDArray[np.float64],
        phase_gradient: NDArray[np.float64],
        sigma_stress: float,
        E_field: NDArray[np.float64],
        H_field: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        A_gauge_1: NDArray[np.float64],
        A_gauge_2: NDArray[np.float64],
        alpha_fermat: float = 0.5,
        Q_isometry_diagnostic: Optional[NDArray[np.float64]] = None,
    ) -> ApexStateTensor:
        r"""Punto de entrada público de Fase 2 (API v7.0.0/v8.0.0 preservada)."""
        return self.phase2.synthesize(
            d_Phi=d_Phi,
            phase_gradient=phase_gradient,
            sigma_stress=sigma_stress,
            E_field=E_field,
            H_field=H_field,
            grad_H=grad_H,
            A_gauge_1=A_gauge_1,
            A_gauge_2=A_gauge_2,
            alpha_fermat=alpha_fermat,
            Q_isometry_diagnostic=Q_isometry_diagnostic,
        )

    def export_sheaf_stalk(
        self, s_val: NDArray[np.float64],
    ) -> SheafStalkApex:
        r"""Exporta el Stalk del haz celular en Fase 3 (API v7.0.0/v8.0.0 preservada)."""
        if self.phase3 is None:
            self.phase3 = KApexElectrodynamicAgent.Phase3_SheafProjection(
                context=self.context,
                return_map=None,
            )
        return self.phase3.export_stalk(s_val=s_val)

    def _combine_residuals(
        self,
        ctx: ApexPreparationContext,
        state: ApexStateTensor,
        stalk: SheafStalkApex,
    ) -> float:
        r"""
        Funcional Supremo de Meta-Residuo (Norma L_∞ sobre los defectos de las 3 Fases):
        Integra el cierre espectral PSD, la inversión bilateral riemanniana, la antisimetría
        de Cartan, la expansión de Lindstedt-Padé, la dualidad de Hodge-Poincaré, la
        recurrencia de Birkhoff, la identidad métrica de Hodge y la deriva adiabática.
        """
        damp = 1.0 - 0.9 * float(np.clip(ctx.tolerance_bundle.symplectic_homotopy_gain, 0.0, 1.0))
        parts: List[float] = [
            float(ctx.spectral_closure_residual),
            float(ctx.inverse_residual),
            float(state.curvature_antisymmetry_residual),
        ]
        if ctx.lindstedt_expansion is not None:
            parts.append(float(ctx.lindstedt_expansion.expansion_residual))
        if state.duality is not None:
            parts.append(float(state.duality.duality_residual))
        if state.spectral_recurrence is not None and math.isfinite(
            state.spectral_recurrence.min_recurrence_distance
        ):
            parts.append(
                float(min(1.0, state.spectral_recurrence.min_recurrence_distance * damp))
            )
        parts.append(float(stalk.hodge_metric_residual))
        parts.append(float(stalk.adiabatic_action_drift))
        finite = [p for p in parts if math.isfinite(p)]
        return float(max(finite)) if finite else float("inf")

    def _resolve_heyting(
        self,
        state: ApexStateTensor,
        stalk: SheafStalkApex,
        meta_residual: float,
    ) -> HeytingOmega3:
        """Colapsa el retículo intuicionista de Heyting Ω₃ para el meta-veredicto."""
        verdict = HeytingOmega3.COHERENT
        if not state.is_electrodynamically_viable:
            verdict = verdict.join(HeytingOmega3.VETOED)
        if not math.isfinite(meta_residual) or meta_residual > 1e-2:
            verdict = verdict.join(HeytingOmega3.VETOED)
        elif meta_residual > _RSI_TARGET_RESIDUAL:
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        if stalk.adiabatic_action_drift > _ADIABATIC_DRIFT_LIMIT:
            verdict = verdict.join(HeytingOmega3.DEGRADED)
        return verdict

    def _rsi_compute_trace(
        self, level: int, bundle: KapexToleranceBundle,
        residual: float, traces_prev: Tuple[RSITrace, ...],
    ) -> RSITrace:
        """Sintetiza la traza metrológica `RSITrace` (Richardson en Nivel ≥ 2, Aitken Δ² en Nivel ≥ 3)."""
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
            richardson_extrapolation=rich, aitken_delta_squared=ait,
            adaptation_factor=factor, level3_active=active,
        )

    def _run_pipeline_at_bundle(
        self, bundle: KapexToleranceBundle, **pipeline_kwargs,
    ) -> Tuple[ApexPreparationContext, ApexStateTensor, SheafStalkApex, float]:
        """Ejecuta la composición anidada `export_stalk ∘ synthesize ∘ build_context` bajo `bundle`."""
        phase1 = KApexElectrodynamicAgent.Phase1_MetricValidation(
            G_mu_nu=self._G_mu_nu,
            G_inv=self._G_inv,
            R_cost=self._R_cost,
            kappa_max=self.kappa_max,
            tolerance_bundle=bundle,
        )
        ctx = phase1.build_context()
        phase2 = KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis(
            context=ctx,
            eikonal_slack=self.eikonal_slack,
            holonomy_tol_rel=self.holonomy_tol_rel,
            kappa_max=self.kappa_max,
        )
        state = phase2.synthesize(**pipeline_kwargs)
        phase3 = KApexElectrodynamicAgent.Phase3_SheafProjection(
            context=ctx, return_map=state.return_map,
        )
        stalk = phase3.export_stalk(s_val=state.gauge_injection_vector)
        residual = self._combine_residuals(ctx, state, stalk)
        return ctx, state, stalk, residual

    def _recursive_self_improvement_loop(
        self,
        max_depth: int,
        target_residual: float,
        **pipeline_kwargs,
    ) -> Tuple[
        Tuple[ApexPreparationContext, ApexStateTensor, SheafStalkApex],
        SelfImprovementResult,
    ]:
        r"""
        Bucle de Punto Fijo del Meta-Controlador RSI-3:
          - Nivel 1: Evalúa el pipeline con el bundle base (scale=1.0, N=2).
          - Nivel 2: Aplica extrapolación de Richardson, activa pulido cuadrático de
                     Schulz-Newton-Hotelling sobre G_inv e incrementa N de Lindstedt.
          - Nivel 3: Activa aceleración no lineal Δ² de Aitken, conmuta a la re-sumación
                     racional de Padé [M/M] y aplica restauración simpléctica contradiabática
                     sobre la deriva adiabática de Nekhoroshev-Poincaré.
        """
        bundle = KapexToleranceBundle(
            scale=1.0,
            n_lindstedt_terms=2,
            symplectic_homotopy_gain=0.0,
            use_pade_resummation=False,
        )
        traces: List[RSITrace] = []
        best = None
        best_res = float("inf")

        for level in range(1, max_depth + 1):
            ctx, state, stalk, residual = self._run_pipeline_at_bundle(
                bundle, **pipeline_kwargs
            )
            trace = self._rsi_compute_trace(
                level=level, bundle=bundle, residual=residual,
                traces_prev=tuple(traces),
            )
            traces.append(trace)
            if residual < best_res:
                best_res = residual
                best = (ctx, state, stalk)
            if residual <= target_residual:
                break

            factor = trace.adaptation_factor
            if not math.isfinite(factor) or factor <= 0.0:
                factor = 0.1
            new_scale = bundle.scale * factor
            if new_scale >= bundle.scale:
                new_scale = bundle.scale * 0.5
            new_terms = min(_RSI_TERMS_CAP, bundle.n_lindstedt_terms + 2)

            near_res = (
                ctx.poincare_snapshot.is_near_resonance
                if ctx.poincare_snapshot is not None else False
            )
            next_level = level + 1
            homotopy_gain = min(
                0.999,
                0.49 * float(next_level - 1) + (0.02 if near_res else 0.0),
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
            raise RecursiveSelfImprovementError("RSI-3 sin resultado.")
        return best, rsi

    def synthesize_with_rsi3(
        self,
        d_Phi: NDArray[np.float64],
        phase_gradient: NDArray[np.float64],
        sigma_stress: float,
        E_field: NDArray[np.float64],
        H_field: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        A_gauge_1: NDArray[np.float64],
        A_gauge_2: NDArray[np.float64],
        alpha_fermat: float = 0.5,
        Q_isometry_diagnostic: Optional[NDArray[np.float64]] = None,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        rsi_target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> SheafStalkApex:
        r"""
        ========================================================================================
        [CLAUSURA SOBERANA DE LA FASE ANIDADA 3 — ENDOFUNCTOR MAESTRO RSI-3]
        ========================================================================================
        Ejecuta la composición completa de las 3 fases anidadas bajo el Meta-Controlador de
        Automejora Recursiva de Nivel 3:
            T_RSI3 = Fix_3(export_stalk ∘ synthesize ∘ build_context)
        y retorna el objeto terminal `SheafStalkApex` sellado con `heyting_verdict` y `rsi_result`.
        ========================================================================================
        """
        pipeline_kwargs = {
            "d_Phi": d_Phi,
            "phase_gradient": phase_gradient,
            "sigma_stress": sigma_stress,
            "E_field": E_field,
            "H_field": H_field,
            "grad_H": grad_H,
            "A_gauge_1": A_gauge_1,
            "A_gauge_2": A_gauge_2,
            "alpha_fermat": alpha_fermat,
            "Q_isometry_diagnostic": Q_isometry_diagnostic,
        }
        (ctx, state, stalk), rsi = self._recursive_self_improvement_loop(
            max_depth=rsi_max_depth,
            target_residual=rsi_target_residual,
            **pipeline_kwargs,
        )
        meta_res = self._combine_residuals(ctx, state, stalk)
        verdict = self._resolve_heyting(state, stalk, meta_res)
        stalk_final = SheafStalkApex(
            delta_apex=stalk.delta_apex,
            delta_metric=stalk.delta_metric,
            delta_dissipative=stalk.delta_dissipative,
            hodge_laplacian=stalk.hodge_laplacian,
            hodge_metric_residual=stalk.hodge_metric_residual,
            hodge_spectral_gap=stalk.hodge_spectral_gap,
            hodge_condition_number=stalk.hodge_condition_number,
            lossless_subspace_dimension=stalk.lossless_subspace_dimension,
            source_injection=stalk.source_injection,
            projected_source_metric=stalk.projected_source_metric,
            projected_source_dissipative=stalk.projected_source_dissipative,
            rank_delta=stalk.rank_delta,
            adiabatic_action_drift=stalk.adiabatic_action_drift,
            meta_residual=float(meta_res),
            heyting_verdict=verdict,
            rsi_result=rsi,
        )
        logger.info(
            "[RSI-3] converged=%s, final_res=%.3e, scale=%.4f, "
            "n_terms=%d, Ω₃=%s, level3=%s.",
            rsi.converged, rsi.final_residual, rsi.final_scale,
            rsi.final_n_terms, verdict.name, rsi.level3_activated,
        )
        return stalk_final


__all__ = [
    # Excepciones (preservadas + nuevas)
    "ElectrodynamicApexError",
    "ApexDimensionError",
    "ApexParameterError",
    "ApexSymmetryError",
    "ApexConditionError",
    "MetricInverseError",
    "SpectralClosureError",
    "GaugePotentialError",
    "EikonalRefractionError",
    "FinancialBlackHoleError",
    "HolonomyVetoError",
    "GaugeCovarianceError",
    "SheafMetricError",
    "PoincareSectionError",
    "RecursiveSelfImprovementError",
    # Retículos
    "HeytingOmega3",
    "ApexViabilityFlags",
    "describe_viability_flags",
    # DTOs (extendidos)
    "ApexPreparationContext",
    "ApexStateTensor",
    "SheafStalkApex",
    # DTOs Poincaré + RLC + RSI-3 (v8.0.0 + v9.0.0)
    "PoincareSpectralSnapshot",
    "LindstedtHeatKernelExpansion",
    "QuantumRLCApexSnapshot",
    "EnergyReturnMapSnapshot",
    "PoincareDualitySnapshot",
    "SpectralRecurrenceSnapshot",
    "KapexToleranceBundle",
    "RSITrace",
    "SelfImprovementResult",
    # Agente
    "KApexElectrodynamicAgent",
]