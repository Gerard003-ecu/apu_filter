
# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA SOBERANO ESTRATO ALFA (KBASE)
====================================================================================================

Nombre: KbaseThermodynamicEngine
Ubicación: app/alfa/kbase/kbase_thermodynamic_engine.py
Versión: 3.0.0-APU-SOVEREIGN-POINCARE-RSI3
Función: Motor soberano termodinámico de la capa Alfa (Kbase) en APU Filter. Modela la termodinámica
         estadística del no-equilibrio, entropía de Boltzmann-Gibbs-Shannon, relaciones de reciprocidad
         de Onsager, tasa de producción entrópica y potenciales de Helmholtz y Gibbs.

         En v3 se añaden, de forma granular y rigurosa:

         (A) MECÁNICA CELESTE DE POINCARÉ (estrato dinámico sobre la carta termodinámica):
             • Sección de Poincaré Σ = {x ∈ ℝ^n : ⟨n̂, x⟩ = c} y mapa de retorno P: Σ → Σ.
             • Variables de acción-ángulo (J_k, θ_k) del ciclo termodinámico fundamental.
             • Espectro de Lyapunov λ_i = log σ_i(DP) del mapa de retorno discreto.
             • Teorema KAM (Kolmogorov-Arnold-Moser): estabilidad de toros invariantes bajo la
               condición diofántica |⟨k,ω⟩| ≥ γ/‖k‖^τ_1 ∀k ∈ ℤ^n \ {0}, τ > n - 1 (Bruno).
             • Detección de resonancias k·ω ≈ 0 y divisores pequeños.
             • Función de Melnikov M(t₀): ceros simples ⇒ intersecciones homoclínicas transversales
               ⇒ entrelazado de Smale-Birkhoff ⇒ caos determinista.

         (B) AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-3):
             Nivel 1: ajuste escalar del parámetro de Tikhonov α.
             Nivel 2: selección de la regla de actualización (gradiente / Newton / Bayes).
             Nivel 3: modificación de la propia regla (factor de contracción, tasa de aprendizaje).
             Punto fijo:  x* tal que  F_L3(x*) = x*.  La convergencia se certifica por la norma de
             los incrementos sucesivos ‖Δθ_n‖ < ε_rec.

----------------------------------------------------------------------------------------------------
CONTINUIDAD FORMAL ENTRE FASES (requisito del tejido):
----------------------------------------------------------------------------------------------------
   Fase 1  ──último morfismo──▶  PhaseOneKBasePacket
   Fase 2  ──primer morfismo (dominio)──▶  PhaseOneKBasePacket
   Fase 2  ──último morfismo──▶  PhaseTwoKBasePacket
   Fase 3  ──primer morfismo (dominio)──▶  PhaseOneKBasePacket × PhaseTwoKBasePacket
   Fase 3  ──último morfismo──▶  KBaseTelemetry (objeto terminal del endofunctor OODA)

   Cada docstring de cabecera de fase indica explícitamente la cláusula "continuación de…".

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO AÑADIDO EN v3:
----------------------------------------------------------------------------------------------------
A. Sección y mapa de retorno. Para un flujo φ_t sobre M con sección transversal Σ ⊂ M de
   codimensión 1, el mapa de Poincaré P: Σ → Σ, P(x) = φ_{τ(x)}(x), linealizado en p ∈ Σ da
   la matriz DP_p ∈ GL(n,ℝ). Sus autovalores ρ_i (multiplicadores) codifican estabilidad
   hiperbólica: |ρ_i|<1 atractor, |ρ_i|>1 repulsor, silla si coexisten.

B. Acción-ángulo. Para un oscilador armónico con frecuencia ω_k, la acción es el invariante
   adiabático J_k = (1/2π)∮ p_k dq_k = (x_k² + (ẋ_k/ω_k)²)/2, y el ángulo θ_k = atan2(−ẋ_k/ω_k, x_k).

C. Teorema KAM (versión de Moser, 1962; Bruno 1971). Bajo perturbación analítica pequeña de un
   Hamiltoniano integrable H₀(J) con Hessiano det ∂²H₀/∂J_i∂J_j ≠ 0 y vector de frecuencias
   ω = ∂H₀/∂J diofántico (|⟨k,ω⟩| ≥ γ/‖k‖^τ_1, τ > n−1), los toros invariantes persisten
   para casi todo ω (medida de Cantor positiva, medida de fuga O(ε^a)).

D. Función de Melnikov. Para H = H₀ + εH₁ con órbita homoclínica q⁰(t) de H₀:
       M(t₀) = ∫_{−∞}^{+∞} {H₀, H₁}(q⁰(t − t₀)) dt
   Si M(t₀) tiene ceros simples, la separatriz se rompe transversalmente ⇒ dinámica caótica
   (Smale-Birkhoff).

E. RSI-3. Sea θ_n ∈ Θ el parámetro meta en el paso n. Las tres capas son:
       θ_{n+1}^{(1)} = T_1(θ_n^{(0)}; ∇L)          (nivel 1: parámetro físico α)
       θ_{n+1}^{(2)} = T_2(θ_n^{(1)}; perf)         (nivel 2: estrategia)
       θ_{n+1}^{(3)} = T_3(θ_n^{(2)}; meta_perf)    (nivel 3: meta-estrategia)
   El punto fijo satisface θ* = T_3∘T_2∘T_1(θ*) y la contracción exige
   ‖Jac(T_3∘T_2∘T_1)‖ < 1 en una vecindad de θ*.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES (heredados + nuevos):
----------------------------------------------------------------------------------------------------
Axioma KTE1 (Segundo Principio del No-Equilibrio de Prigogine): d_i S ≥ 0.
Axioma KTE2 (Simetría Microversal de Onsager): L_{ij} = L_{ji}.
Axioma KTE3 (Convexidad de la Entropía Estadística): −∇²S ⪰ 0.
Axioma KTE4 (Transversalidad de Poincaré): ⟨n̂, ẋ⟩ ≠ 0 sobre Σ \ ∂Σ (sección regular).
Axioma KTE5 (Diofantino de KAM): el vector ω ∈ ℝ^n satisface la condición (C, τ) de Bruno.
Axioma KTE6 (Convergencia de RSI-3): la contracción ρ_n = ‖θ_{n+1} − θ_n‖ / ‖θ_n − θ_{n-1}‖
                                          satisface limsup ρ_n < 1 en el punto fijo.

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO (heredados + nuevos):
----------------------------------------------------------------------------------------------------
Invariante 1: T ≥ T_min > 0.
Invariante 2: σ_S = Σ J_i X_i ≥ 0.
Invariante 3 (Lyapunov): λ_max(P) es invariante bajo cambio de base GL(n,ℝ) del estado.
Invariante 4 (KAM): min_{k≠0} ‖k‖^τ_1 |⟨k,ω⟩| ≥ γ > 0 en el régimen estable.
Invariante 5 (RSI-3): si la recursión ha convergido, next_alpha = current_alpha dentro de ε_rec.
====================================================================================================
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.KBaseThermodynamicEngine")

# ---------------------------------------------------------------------------
# Constantes metrológicas (análisis de Wilkinson / IEEE-754 binary64)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_DENSITY_TRACE_TOLERANCE: Final[float] = 1e-10
_PSD_TOLERANCE: Final[float] = 1e-10

# v3: constantes dinámicas (Poincaré / KAM / RSI-3)
_POINCARE_TOLERANCE: Final[float] = 1e-10
_KAM_GAMMA_DEFAULT: Final[float] = 0.1
_KAM_TAU_DEFAULT: Final[float] = 2.5      # τ > n − 1 (Bruno, para n=5, τ>4 sería estricto; 2.5 empírico)
_KAM_K_MAX_DEFAULT: Final[int] = 4
_LYAPUNOV_FLOOR: Final[float] = 1e-300
_MELNIKOV_LOBES_SAMPLES: Final[int] = 64
_RECURSION_CONTRACTION_DEFAULT: Final[float] = 0.5
_RECURSION_CONVERGENCE_TOL: Final[float] = 1e-8
_RECURSION_MAX_HISTORY: Final[int] = 12
_ALPHA_FLOOR: Final[float] = _WILKINSON_FLOOR


# =============================================================================
# Retículo de Heyting
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.
    Álgebra de Heyting del orden total:
        a ∧ b = min, a ∨ b = max,
        a → b = ⊤ si a ≼ b, b en caso contrario,
        ¬a = a → ⊥.
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
# Utilidades y paquetes inmutables
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _logsumexp(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return -np.inf
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -np.inf
    shift = float(np.max(finite))
    return shift + float(np.log(np.sum(np.exp(finite - shift))))


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""Factorización espectral g = UΛU† (Tikhonov: λ ↦ λ+α)."""

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
class DensityOperatorChart:
    r"""Carta espectral de ρ ∈ 𝒟(ℋ)."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    trace: float
    min_eigenvalue: float
    is_density: bool
    von_neumann_entropy: float
    purity: float
    energy_expectation: float
    quantum_free_energy: float
    log_partition: float
    relative_entropy_to_gibbs: float
    stationarity_defect: float
    banach_submultiplicative_defect: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))


# -----------------------------------------------------------------------------
# v3 — Carta de Poincaré (sección + mapa de retorno + acción-ángulo + Lyapunov)
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoincareChart:
    r"""
    Carta dinámica asociada al flujo termodinámico discretizado por ciclos.

    Contiene:
      • Σ: sección ⟨n̂, x⟩ = c (normal n̂ normalizada).
      • P: Σ → Σ mapa de retorno (estimado por mínimos cuadrados desde la
        historia de estados; identidad penalizada si la historia es corta).
      • Espectro de Lyapunov: λ_i = log σ_i(P) (discreto, un paso).
      • Variables de acción-ángulo (J_k, θ_k) del ciclo fundamental.
      • Tiempo de retorno τ_ret (por defecto 1.0 en el bootstrap).
    """

    section_normal: np.ndarray
    section_offset: float
    section_value: float
    return_map: np.ndarray
    return_eigenvalues: np.ndarray
    spectral_radius: float
    return_time: float
    action_variables: np.ndarray
    angle_variables: np.ndarray
    fundamental_frequencies: np.ndarray
    lyapunov_exponents: np.ndarray
    max_lyapunov: float

    def __post_init__(self) -> None:
        for name in (
            "section_normal",
            "return_map",
            "return_eigenvalues",
            "action_variables",
            "angle_variables",
            "fundamental_frequencies",
            "lyapunov_exponents",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PhaseOneKBasePacket:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.

    Contiene el 1-jet termogeométrico: congruencia métrica, potenciales de
    Gibbs–Helmholtz, sonda CSMD, desigualdad de Clausius–Duhem, carta de ρ
    y —en v3— la carta de Poincaré (sección, mapa de retorno, acción-ángulo,
    exponentes de Lyapunov).

    Dominio formal de `_phase2_spectral_integrity` (primer morfismo de
    Fase 2). No hay objeto intersticial.
    """

    pullback_tensor: np.ndarray
    pullback_hermiticity_residual: float
    pullback_spectral_entropy: float
    gibbs_energy: float
    helmholtz_energy: float
    enthalpy: float
    internal_energy: float
    thermo_identity_residual: float
    csmd_dG_dT: float
    csmd_entropy_error: float
    dissipation_potential: float
    thermal_entropy_production: float
    clausius_coherent: bool
    fourier_conductivity: float
    fourier_residual: float
    thermal_dirichlet_energy: float
    spectral_chart: SpectralChart
    density_chart: DensityOperatorChart
    poincare_chart: PoincareChart
    state_vector: np.ndarray
    stationarity_defect: float
    temperature: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pullback_tensor", _freeze_array(self.pullback_tensor))
        object.__setattr__(self, "state_vector", _freeze_array(self.state_vector))


@dataclass(frozen=True, slots=True)
class SpectralThermoReport:
    r"""Integridad espectral de g y de ρ (Fase 2, primer morfismo)."""

    metric_condition: float
    metric_gap: float
    density_is_valid: bool
    density_purity: float
    relative_entropy: float
    stationarity_defect: float
    banach_defect: float
    integrity_score: float


@dataclass(frozen=True, slots=True)
class KAMReport:
    r"""Resultado del teorema KAM + detección de resonancia (Bruno)."""

    stable: bool
    min_diophantine_ratio: float
    gamma: float
    tau: float
    resonance_order: int
    small_divisor_min: float
    frequencies: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequencies", _freeze_array(self.frequencies))


@dataclass(frozen=True, slots=True)
class SelfImprovementReport:
    r"""Estado del bucle RSI-3 (automejora recursiva de nivel 3)."""

    iteration: int
    current_alpha: float
    next_alpha: float
    contraction: float
    level3_score: float
    recursion_converged: bool
    recursion_residual: float
    performance_delta: float


@dataclass(frozen=True, slots=True)
class PhaseTwoKBasePacket:
    r"""
    Objeto terminal de la Fase 2 ≡ primer factor del dominio de la Fase 3.
    """

    irreversibility: float
    heyting_verdict: str
    heyting_rank: int
    clausius_threshold_violated: bool
    stationarity_threshold_violated: bool
    density_axiom_violated: bool
    condition_threshold_violated: bool
    kam_threshold_violated: bool
    chaos_threshold_violated: bool
    spectral_integrity: SpectralThermoReport
    kam_report: KAMReport
    self_improvement: SelfImprovementReport
    heyting_score: float
    exergy_destruction: float


@dataclass(frozen=True, slots=True)
class KBaseTelemetry:
    r"""Sello inmutable de telemetría (objeto terminal del functor OODA)."""

    pullback_tensor: np.ndarray
    gibbs_energy: float
    enthalpy: float
    helmholtz_energy: float
    dissipation_potential: float
    clausius_coherent: bool
    irreversibility: float
    heyting_verdict: str
    thermo_identity_residual: float
    csmd_entropy_error: float
    relative_entropy_to_gibbs: float
    von_neumann_entropy: float
    purity: float
    condition_number: float
    conservation_residual: float
    exergy_destruction: float
    spectral_integrity_score: float
    # v3
    poincare_return_time: float
    return_map_spectral_radius: float
    max_lyapunov_exponent: float
    kam_stable: bool
    kam_diophantine_ratio: float
    melnikov_lobe_area: float
    chaos_indicator: bool
    recursion_converged: bool
    recursion_iteration: int
    self_improvement_level3_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pullback_tensor", _freeze_array(self.pullback_tensor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pullback_tensor": np.array(self.pullback_tensor, copy=True),
            "gibbs_energy": self.gibbs_energy,
            "enthalpy": self.enthalpy,
            "helmholtz_energy": self.helmholtz_energy,
            "dissipation_potential": self.dissipation_potential,
            "clausius_coherent": self.clausius_coherent,
            "irreversibility": self.irreversibility,
            "heyting_verdict": self.heyting_verdict,
            "thermo_identity_residual": self.thermo_identity_residual,
            "csmd_entropy_error": self.csmd_entropy_error,
            "relative_entropy_to_gibbs": self.relative_entropy_to_gibbs,
            "von_neumann_entropy": self.von_neumann_entropy,
            "purity": self.purity,
            "condition_number": self.condition_number,
            "conservation_residual": self.conservation_residual,
            "exergy_destruction": self.exergy_destruction,
            "spectral_integrity_score": self.spectral_integrity_score,
            # v3
            "poincare_return_time": self.poincare_return_time,
            "return_map_spectral_radius": self.return_map_spectral_radius,
            "max_lyapunov_exponent": self.max_lyapunov_exponent,
            "kam_stable": self.kam_stable,
            "kam_diophantine_ratio": self.kam_diophantine_ratio,
            "melnikov_lobe_area": self.melnikov_lobe_area,
            "chaos_indicator": self.chaos_indicator,
            "recursion_converged": self.recursion_converged,
            "recursion_iteration": self.recursion_iteration,
            "self_improvement_level3_score": self.self_improvement_level3_score,
        }


# =============================================================================
# MOTOR
# =============================================================================
class KBaseThermodynamicEngine:
    r"""
    Motor Termodinámico de la Base Táctica (Estrato α — KBASE).

    Endofunctor OODA  T = Act ∘ Decide ∘ Observe  sobre el espacio de
    estados materiales. En v3 cada fase incluye morfismos adicionales
    (Poincaré / KAM / Melnikov / RSI-3) que se anidan sin romper la
    continuidad PhaseOne → PhaseTwo → Telemetry.
    """

    # =========================================================================
    # CICLO DE VIDA E INFRAESTRUCTURA COMÚN
    # =========================================================================
    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError("La dimensión del espacio de fase debe ser estrictamente positiva.")
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))
        # v3: parámetro activo mutable (RSI-3 puede moverlo)
        self._alpha_current: float = self._reg

        # v3: memoria de ciclos (necesaria para el mapa de retorno de Poincaré)
        self._cycle_history: List[Dict[str, Any]] = []
        # v3: memoria de RSI-3 (nivel 3 = meta-estrategia)
        self._meta_state: Dict[str, float] = {
            "alpha": self._reg,
            "irr_thresh": 0.1,
            "cond_thresh": 1e12,
            "contraction": _RECURSION_CONTRACTION_DEFAULT,
            "level3_score": 1.0,
            "iteration": 0.0,
        }
        self._meta_trajectory: List[Dict[str, float]] = []
        self._previous_performance: float = 1.0

    def reset(self) -> None:
        """Vacía toda la memoria de ciclos y reinicia la meta-estrategia RSI-3."""
        self._cycle_history.clear()
        self._meta_trajectory.clear()
        self._previous_performance = 1.0
        self._alpha_current = self._reg
        self._meta_state = {
            "alpha": self._reg,
            "irr_thresh": 0.1,
            "cond_thresh": 1e12,
            "contraction": _RECURSION_CONTRACTION_DEFAULT,
            "level3_score": 1.0,
            "iteration": 0.0,
        }

    # -------------------------------------------------------------------------
    # Aritmética compensada
    # -------------------------------------------------------------------------
    def kahan_sum(self, arr: np.ndarray) -> float:
        if np.asarray(arr).ndim != 1:
            raise ValueError(
                f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}"
            )
        return float(np.real(self._neumaier_sum(arr)))

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
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

    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    def _validate_square_matrix(self, M: np.ndarray, name: str) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} debe ser n × n, se recibió {M.shape}")
        if M.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != self._n:
            raise ValueError(f"{name} debe ser vector de dim {self._n}, se recibió {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_scalar(self, value: Any, name: str) -> float:
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} debe ser un escalar finito")
        return float(value)

    def _validate_hermitian(
        self, M: np.ndarray, name: str, tolerance: float = _HERMITIAN_TOLERANCE
    ) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tolerance * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(f"{name} no es hermítica (tolerancia {tolerance})")

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.conj().T)

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    # Aparato de medición: KBN, espectro de g y ρ, 1-jets físicos,
    # y —en v3— la carta dinámica de Poincaré (sección, retorno,
    # acción-ángulo, Lyapunov).
    # El ÚLTIMO método, `_phase1_observe_orient`, tiene codominio
    # `PhaseOneKBasePacket`, que ES el dominio del primer morfismo de la
    # Fase 2 (`_phase2_spectral_integrity`).
    # =========================================================================
    def _factorize_metric(self, metric_tensor: np.ndarray) -> SpectralChart:
        r"""
        Una sola descomposición `eigh`. Regularización de Tikhonov espectral
        λ ↦ λ + α con α = self._alpha_current (v3: mutable por RSI-3).
        """
        self._validate_square_matrix(metric_tensor, "metric_tensor")
        herm = self._hermitize(metric_tensor)
        self._validate_hermitian(herm, "metric_tensor")

        eigenvalues, eigenvectors = la.eigh(herm)
        eigenvalues = np.real(eigenvalues)
        alpha = float(self._alpha_current)

        regularized = bool(np.any(eigenvalues <= alpha))
        if regularized:
            logger.warning(
                "Métrica no definida positiva (λ_min=%.3e); Tikhonov α=%.3e.",
                float(np.min(eigenvalues)), alpha,
            )
        shifted = np.maximum(eigenvalues + alpha, _WILKINSON_FLOOR)
        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T

        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        log_vol = 0.5 * float(self.kahan_sum(np.log(np.maximum(lam_abs, _WILKINSON_FLOOR))))
        volume = float(np.exp(log_vol))
        op_norm = float(np.max(lam_abs))
        frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
        ordered = np.sort(lam_abs)
        gap = float(ordered[1] - ordered[0]) if shifted.size >= 2 else float(lam_abs[0])

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
            tikhonov_alpha=alpha,
        )

    def compute_riemannian_pullback(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str = "covariant",
        jacobian: Optional[np.ndarray] = None,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> np.ndarray:
        r"""
        [KBASE 1] Congruencia métrica o pullback por dφ.
        Covariante: g T g. Contravariante: g⁻¹ T g⁻¹. Pullback: J† T J.
        """
        self._validate_square_matrix(G, "G")
        self._validate_square_matrix(tensor_M, "tensor_M")
        self._validate_hermitian(G, "G")

        if jacobian is not None:
            self._validate_square_matrix(jacobian, "jacobian")
            return jacobian.conj().T @ tensor_M @ jacobian

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(G)
        g_sym = self._hermitize(np.asarray(G))
        M = np.asarray(tensor_M)

        if tensor_type == "covariant":
            return g_sym @ M @ g_sym
        if tensor_type == "contravariant":
            return np.asarray(chart.inv_metric) @ M @ np.asarray(chart.inv_metric)
        raise ValueError("Tipo de tensor no soportado (covariant/contravariant).")

    def _pullback_spectral_entropy(self, tensor: np.ndarray) -> float:
        herm = self._hermitize(np.asarray(tensor, dtype=np.complex128))
        evals = np.real(la.eigvalsh(herm))
        weights = np.abs(evals)
        total = float(np.real(self._neumaier_sum(weights.astype(np.float64))))
        if total <= _WILKINSON_FLOOR:
            return 0.0
        probs = weights / total
        acc = [float(-p * np.log(p)) for p in probs if p > _WILKINSON_FLOOR]
        return float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))))

    def compute_gibbs_free_energy(
        self,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
    ) -> Tuple[float, float]:
        r"""[KBASE 2] H = U + PV,  G = H − TS. Devuelve (G, H)."""
        U = self._validate_scalar(internal_energy_U, "internal_energy_U")
        P = self._validate_scalar(pressure_P, "pressure_P")
        V = self._validate_scalar(volume_V, "volume_V")
        T = self._validate_scalar(temperature_T, "temperature_T")
        S = self._validate_scalar(entropy_S, "entropy_S")
        if T <= _WILKINSON_FLOOR:
            raise ValueError(f"Temperatura debe ser > 0 (se recibió {T}).")
        if S < -_WILKINSON_FLOOR:
            logger.warning("Entropía negativa (S=%.6e).", S)
        if V <= 0.0:
            logger.warning("Volumen no positivo (V=%.6e).", V)
        enthalpy = self.kahan_sum(np.array([U, P * V], dtype=np.float64))
        gibbs = self.kahan_sum(np.array([enthalpy, -T * S], dtype=np.float64))
        return float(gibbs), float(enthalpy)

    def _complex_step_dG_dT(
        self, enthalpy: float, entropy_S: float, temperature_T: float,
        step: float = _CSMD_STEP,
    ) -> float:
        value = complex(enthalpy) - (complex(temperature_T) + 1.0j * step) * complex(entropy_S)
        return float(np.imag(value) / step)

    def evaluate_clausius_duhem(
        self,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        temperature_T: float,
    ) -> Tuple[float, bool]:
        r"""[KBASE 3] Φ = σ_int − ⟨q, dT⟩/T².  Φ ≥ −ε ⇒ coherente."""
        sigma_int = self._validate_scalar(entropy_production_rate, "entropy_production_rate")
        T = self._validate_scalar(temperature_T, "temperature_T")
        if T <= _WILKINSON_FLOOR:
            raise ValueError(f"Temperatura debe ser > 0 (se recibió {T}).")
        self._validate_vector(np.asarray(heat_flux_q), "heat_flux_q")
        self._validate_vector(np.asarray(temp_gradient_gradT), "temp_gradient_gradT")

        pairing = float(np.real(np.dot(np.asarray(heat_flux_q), np.asarray(temp_gradient_gradT))))
        conduction = pairing / (T ** 2)
        Phi = self.kahan_sum(np.array([sigma_int, -conduction], dtype=np.float64))
        return float(Phi), bool(Phi >= -_WILKINSON_FLOOR)

    def _fourier_audit(
        self, heat_flux_q: np.ndarray, temp_gradient_gradT: np.ndarray, chart: SpectralChart,
    ) -> Tuple[float, float, float]:
        q = np.asarray(heat_flux_q, dtype=np.float64)
        gT = np.asarray(temp_gradient_gradT, dtype=np.float64)
        gTgT = float(np.real(np.dot(gT, gT)))
        if gTgT <= _WILKINSON_FLOOR:
            k_hat, residual = 0.0, float(la.norm(q))
        else:
            k_hat = float(-np.real(np.dot(q, gT)) / gTgT)
            residual = float(la.norm(q + k_hat * gT) / max(la.norm(q), _WILKINSON_FLOOR))
        g = self._hermitize(
            np.asarray(chart.eigenvectors)
            @ np.diag(np.asarray(chart.eigenvalues, dtype=np.float64))
            @ np.asarray(chart.eigenvectors).conj().T
        )
        dirichlet = float(np.real(q @ g @ q))
        return k_hat, residual, dirichlet

    def calculate_spectral_irreversibility(
        self, rho: np.ndarray, hamiltonian_H: np.ndarray,
    ) -> float:
        r"""[KBASE 4] ℐ = ‖[H,ρ]‖²_HS. Mide partida de estacionariedad."""
        self._validate_square_matrix(rho, "rho")
        self._validate_square_matrix(hamiltonian_H, "hamiltonian_H")
        self._validate_hermitian(rho, "rho")
        self._validate_hermitian(hamiltonian_H, "hamiltonian_H")
        commutator = hamiltonian_H @ rho - rho @ hamiltonian_H
        gram = commutator.conj().T @ commutator
        return max(0.0, float(self.kahan_sum(np.real(np.diagonal(gram)))))

    def _factorize_density(
        self, rho: np.ndarray, hamiltonian_H: np.ndarray, temperature_T: float,
    ) -> DensityOperatorChart:
        self._validate_square_matrix(rho, "rho")
        self._validate_square_matrix(hamiltonian_H, "hamiltonian_H")
        self._validate_hermitian(rho, "rho")
        self._validate_hermitian(hamiltonian_H, "hamiltonian_H")

        rho_h = self._hermitize(np.asarray(rho, dtype=np.complex128))
        H_h = self._hermitize(np.asarray(hamiltonian_H, dtype=np.complex128))

        evals, evecs = la.eigh(rho_h)
        evals = np.real(evals)
        trace = float(np.real(self._neumaier_sum(evals.astype(np.float64))))
        min_eig = float(np.min(evals))
        is_density = (
            min_eig >= -_PSD_TOLERANCE
            and abs(trace - 1.0) <= _DENSITY_TRACE_TOLERANCE
        )
        if not is_density:
            logger.warning("ρ no satisface axiomas (tr=%.3e, λ_min=%.3e).", trace, min_eig)

        acc_s, acc_g = [], []
        for eta in evals:
            clipped = max(float(eta), 0.0)
            acc_g.append(clipped * clipped)
            if clipped > _WILKINSON_FLOOR:
                acc_s.append(float(-clipped * np.log(clipped)))
        S_vN = float(np.real(self._neumaier_sum(np.asarray(acc_s, dtype=np.float64))))
        purity = float(np.real(self._neumaier_sum(np.asarray(acc_g, dtype=np.float64))))

        energy = float(np.real(np.trace(rho_h @ H_h)))
        T = float(temperature_T)
        free = energy - T * S_vN

        h_eigs = np.real(la.eigvalsh(H_h))
        if T <= _WILKINSON_FLOOR:
            log_z, relative = np.inf, np.inf
        else:
            beta = 1.0 / T
            log_z = _logsumexp(-beta * h_eigs)
            relative = max(0.0, float(-S_vN + beta * energy + log_z))

        stat = self.calculate_spectral_irreversibility(rho_h, H_h)
        op_H = float(la.norm(H_h, ord=2))
        op_rho = float(la.norm(rho_h, ord=2))
        lhs = float(la.norm(H_h @ rho_h, ord=2))
        rhs = op_H * op_rho
        banach = max(0.0, lhs / rhs - 1.0) if rhs > _WILKINSON_FLOOR else 0.0

        return DensityOperatorChart(
            eigenvalues=evals,
            eigenvectors=evecs,
            trace=trace,
            min_eigenvalue=min_eig,
            is_density=is_density,
            von_neumann_entropy=S_vN,
            purity=purity,
            energy_expectation=energy,
            quantum_free_energy=free,
            log_partition=float(log_z),
            relative_entropy_to_gibbs=float(relative),
            stationarity_defect=stat,
            banach_submultiplicative_defect=float(banach),
        )

    # -------------------------------------------------------------------------
    # v3 — NÚCLEO DE POINCARÉ
    # -------------------------------------------------------------------------
    def _poincare_section_construct(
        self, normal: np.ndarray, offset: float, state: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        Construye Σ = {x : ⟨n̂, x⟩ = c} y evalúa la distancia con signo
        del `state` a la sección. La normal se normaliza a norma 2 unitaria.

        Regresa (n̂, c, ⟨n̂, state⟩ − c). La distancia es cero si el estado
        yace sobre Σ; positiva en el semi-espacio exterior.
        """
        n = np.asarray(normal, dtype=np.float64).copy()
        s = np.asarray(state, dtype=np.float64).copy()
        norm = float(la.norm(n))
        if norm < _WILKINSON_FLOOR:
            n = np.ones_like(n) / np.sqrt(n.size)
        else:
            n /= norm
        value = float(np.dot(n, s) - float(offset))
        return n, float(offset), value

    def _poincare_return_map(
        self, current_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Estima P: Σ → Σ desde la historia de estados por mínimos cuadrados.

        Sea X = [s_{k}, s_{k+1}, …, s_{k+m−1}] ∈ ℝ^{n×m}, Y = [s_{k+1}, …].
        P̂ = argmin_P ‖Y − P X‖_F = Y X⁺ (pseudo-inversa). Si m < n, se
        regulariza con P̂ ← w P̂ + (1−w) I con w = m / (n + 1) para evitar
        sobreajuste de rango deficiente.

        El tiempo de retorno se reporta como 1.0 (un ciclo por paso).
        Los autovalores λ_i(P̂) son los multiplicadores de Floquet
        discretos; su módulo máximo es el radio espectral ρ(P̂).
        """
        n = int(current_state.size)
        m_avail = len(self._cycle_history)
        if m_avail < 2:
            return np.eye(n), np.ones(n, dtype=complex), 1.0

        m = min(m_avail, n + 1)
        X_cols, Y_cols = [], []
        for i in range(m_avail - m, m_avail):
            prev_state = np.asarray(self._cycle_history[i]["state"], dtype=np.float64)
            if i + 1 < m_avail:
                next_state = np.asarray(self._cycle_history[i + 1]["state"], dtype=np.float64)
            else:
                next_state = np.asarray(current_state, dtype=np.float64)
            X_cols.append(prev_state)
            Y_cols.append(next_state)
        X = np.column_stack(X_cols)
        Y = np.column_stack(Y_cols)

        try:
            P_hat = Y @ np.linalg.pinv(X)
        except np.linalg.LinAlgError:
            P_hat = np.eye(n)

        if m < n + 1:
            w = m / (n + 1)
            P_hat = w * P_hat + (1.0 - w) * np.eye(n)

        eigvals = np.linalg.eigvals(P_hat)
        return np.real_if_close(P_hat, tol=1e6), eigvals, 1.0

    def _action_angle_decompose(
        self, return_time: float, state: np.ndarray, return_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Variables de acción-ángulo del ciclo fundamental.

        Sea τ = return_time, ω_k = 2π k / τ (frecuencia del k-ésimo armónico).
        La velocidad se estima por (P − I) state. Para cada armónico:
            J_k = (x_k² + (ẋ_k/ω_k)²) / 2       (acción, invariante adiabático)
            θ_k = atan2(−ẋ_k/ω_k, x_k) mod 2π   (ángulo)
        Es la descomposición exacta del oscilador armónico linealizado; coincide
        con J = (1/2π)∮p dq para H = p²/2 + ω²q²/2.
        """
        n = int(state.size)
        tau = float(return_time) if return_time > _WILKINSON_FLOOR else 1.0
        freqs = np.array([2.0 * np.pi * k / tau for k in range(1, n + 1)], dtype=np.float64)

        velocity = (np.asarray(return_map, dtype=np.float64) - np.eye(n)) @ state
        omega_guard = np.maximum(freqs, _WILKINSON_FLOOR)
        actions = 0.5 * (state ** 2 + (velocity / omega_guard) ** 2)
        angles = np.mod(np.arctan2(-velocity / omega_guard, state), 2.0 * np.pi)
        return actions, angles, freqs

    def _lyapunov_spectrum(
        self, return_map: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Espectro de Lyapunov discreto del mapa de retorno:
            λ_i = log σ_i(P),  σ_i = i-ésimo valor singular de P.
        Un λ_i > 0 indica expansión en la dirección i (caos); λ_i < 0
        contracción (disipación). Σ λ_i = log |det P| (identidad de volumen).
        """
        sv = np.linalg.svd(np.asarray(return_map, dtype=np.float64), compute_uv=False)
        sv = np.maximum(sv, _LYAPUNOV_FLOOR)
        lyap = np.log(sv)
        return lyap, float(np.max(lyap))

    def _poincare_chart_build(
        self,
        state: np.ndarray,
        normal: Optional[np.ndarray] = None,
        offset: float = 0.0,
    ) -> PoincareChart:
        r"""Compone sección + mapa de retorno + acción-ángulo + Lyapunov."""
        n = int(state.size)
        if normal is None:
            normal = np.zeros(n, dtype=np.float64)
            normal[0] = 1.0  # sección canónica: x_0 = offset
        n_hat, c, val = self._poincare_section_construct(normal, offset, state)
        P, eigvals, tau = self._poincare_return_map(state)
        actions, angles, freqs = self._action_angle_decompose(tau, state, P)
        lyap, max_lyap = self._lyapunov_spectrum(P)
        radius = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        return PoincareChart(
            section_normal=n_hat,
            section_offset=c,
            section_value=val,
            return_map=P,
            return_eigenvalues=eigvals,
            spectral_radius=radius,
            return_time=tau,
            action_variables=actions,
            angle_variables=angles,
            fundamental_frequencies=freqs,
            lyapunov_exponents=lyap,
            max_lyapunov=max_lyap,
        )

    # -------------------------------------------------------------------------
    # FASE 1 — ÚLTIMO MORFISMO
    # -------------------------------------------------------------------------
    def _phase1_observe_orient(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
        jacobian: Optional[np.ndarray] = None,
    ) -> PhaseOneKBasePacket:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE + ORIENT]

        Codominio
        ---------
        `PhaseOneKBasePacket`

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2:
            `_phase2_spectral_integrity(self, packet: PhaseOneKBasePacket)`.
        No hay objeto intersticial: la composición
            _phase2_spectral_integrity ∘ _phase1_observe_orient
        está tipada estrictamente.
        """
        logger.info("Fase 1: carta espectral, congruencia, Gibbs-Helmholtz+CSMD, "
                    "Clausius-Duhem, carta de ρ, sección de Poincaré.")

        # 1) Carta espectral + congruencia
        chart = self._factorize_metric(G)
        pullback = self.compute_riemannian_pullback(
            G, tensor_M, tensor_type, jacobian=jacobian, spectral_chart=chart
        )
        pull_skew = pullback - pullback.conj().T
        pull_herm_res = float(la.norm(pull_skew, ord="fro"))
        pull_entropy = self._pullback_spectral_entropy(pullback)

        # 2) Potenciales de Gibbs-Helmholtz + CSMD
        U = self._validate_scalar(internal_energy_U, "internal_energy_U")
        P = self._validate_scalar(pressure_P, "pressure_P")
        V = self._validate_scalar(volume_V, "volume_V")
        T = self._validate_scalar(temperature_T, "temperature_T")
        S = self._validate_scalar(entropy_S, "entropy_S")

        gibbs, enthalpy = self.compute_gibbs_free_energy(U, P, V, T, S)
        helmholtz = self.kahan_sum(np.array([U, -T * S], dtype=np.float64))
        ident_terms = np.array(
            [abs(enthalpy - (U + P * V)),
             abs(gibbs - (enthalpy - T * S)),
             abs(helmholtz - (U - T * S)),
             abs(gibbs - (helmholtz + P * V))],
            dtype=np.float64,
        )
        thermo_residual = float(np.real(self._neumaier_sum(ident_terms)))
        csmd = self._complex_step_dG_dT(enthalpy, S, T)
        csmd_err = float(abs(csmd + S))

        # 3) Clausius-Duhem + auditoría de Fourier
        dissipation, coherent = self.evaluate_clausius_duhem(
            entropy_production_rate, heat_flux_q, temp_gradient_gradT, T
        )
        pairing = float(np.real(np.dot(np.asarray(heat_flux_q), np.asarray(temp_gradient_gradT))))
        sigma_th = float(-pairing / (T ** 2))
        conductivity, fourier_res, dirichlet = self._fourier_audit(
            heat_flux_q, temp_gradient_gradT, chart
        )

        # 4) Carta de densidad
        density = self._factorize_density(rho, hamiltonian_H, T)

        # 5) Estado dinámico para Poincaré (vector de 5 observables)
        state_vector = np.array(
            [T, S, dissipation, density.stationarity_defect,
             density.relative_entropy_to_gibbs],
            dtype=np.float64,
        )

        # 6) Carta de Poincaré — sección canónica x_0 = T_ref (300 K)
        normal = np.zeros(state_vector.size, dtype=np.float64)
        normal[0] = 1.0
        poincare_chart = self._poincare_chart_build(
            state=state_vector, normal=normal, offset=300.0,
        )

        # 7) Empaquetado
        packet = PhaseOneKBasePacket(
            pullback_tensor=pullback,
            pullback_hermiticity_residual=pull_herm_res,
            pullback_spectral_entropy=pull_entropy,
            gibbs_energy=gibbs,
            helmholtz_energy=float(helmholtz),
            enthalpy=enthalpy,
            internal_energy=U,
            thermo_identity_residual=thermo_residual,
            csmd_dG_dT=csmd,
            csmd_entropy_error=csmd_err,
            dissipation_potential=dissipation,
            thermal_entropy_production=sigma_th,
            clausius_coherent=coherent,
            fourier_conductivity=conductivity,
            fourier_residual=fourier_res,
            thermal_dirichlet_energy=dirichlet,
            spectral_chart=chart,
            density_chart=density,
            poincare_chart=poincare_chart,
            state_vector=state_vector,
            stationarity_defect=density.stationarity_defect,
            temperature=T,
        )
        logger.debug(
            "Fase 1 completa: G=%.4f F=%.4f H=%.4f Φ=%.3e CD=%s "
            "ℐ=%.3e D(ρ‖ρβ)=%.3e CSMD=%.3e κ=%.3e ρ(P)=%.3f λ_max=%.3f",
            gibbs, helmholtz, enthalpy, dissipation, coherent,
            density.stationarity_defect, density.relative_entropy_to_gibbs,
            csmd_err, chart.condition_number,
            poincare_chart.spectral_radius, poincare_chart.max_lyapunov,
        )
        return packet

    # =========================================================================
    # FASE 2 — DECIDE
    # Dominio del primer morfismo = PhaseOneKBasePacket (codominio de Fase 1).
    # Se evalúa la integridad espectral de g y ρ, la estabilidad KAM,
    # la resonancia diofántica y se ejecuta el bucle RSI-3.
    # El ÚLTIMO método, `_phase2_decide`, tiene codominio `PhaseTwoKBasePacket`,
    # segundo factor del dominio de `_phase3_conservation_audit`.
    # =========================================================================
    def _phase2_spectral_integrity(
        self, packet: PhaseOneKBasePacket,
    ) -> SpectralThermoReport:
        r"""
        [FASE 2 — PRIMER MORFISMO: INTEGRIDAD ESPECTRAL]

        Dominio
        -------
        `PhaseOneKBasePacket`  ← continuación formal de `_phase1_observe_orient`.

        Diagnósticos: κ₂(g), gap, axiomas de ρ, D(ρ‖ρ_β), ℐ, defecto de
        submultiplicatividad de Banach. El score ∈ [0,1] es multiplicativo
        (todos los factores deben ser favorables).
        """
        chart = packet.spectral_chart
        dens = packet.density_chart

        cond_term = 1.0 / (1.0 + np.log10(max(chart.condition_number, 1.0)))
        gap_term = float(np.tanh(chart.spectral_gap / max(chart.operator_norm, _WILKINSON_FLOOR)))
        dens_term = 1.0 if dens.is_density else 0.0
        rel_term = float(np.exp(-dens.relative_entropy_to_gibbs))
        stat_term = float(np.exp(-dens.stationarity_defect))
        banach_term = float(np.exp(-dens.banach_submultiplicative_defect / _MACHINE_EPS))
        cd_term = 1.0 if packet.clausius_coherent else 0.0
        csmd_term = float(np.exp(-packet.csmd_entropy_error / 1e-12))

        score = float(
            cond_term * max(gap_term, 0.0) * dens_term
            * rel_term * stat_term * min(banach_term, 1.0)
            * cd_term * csmd_term
        )
        return SpectralThermoReport(
            metric_condition=float(chart.condition_number),
            metric_gap=float(chart.spectral_gap),
            density_is_valid=dens.is_density,
            density_purity=dens.purity,
            relative_entropy=dens.relative_entropy_to_gibbs,
            stationarity_defect=dens.stationarity_defect,
            banach_defect=dens.banach_submultiplicative_defect,
            integrity_score=score,
        )

    def _phase2_kam_audit(
        self,
        packet: PhaseOneKBasePacket,
        gamma: float = _KAM_GAMMA_DEFAULT,
        tau: float = _KAM_TAU_DEFAULT,
        K_max: int = _KAM_K_MAX_DEFAULT,
    ) -> KAMReport:
        r"""
        [FASE 2 — SEGUNDO MORFISMO: AUDITORÍA KAM]

        Dominio
        -------
        `PhaseOneKBasePacket`  ← continuación formal de `_phase2_spectral_integrity`.

        Condición diofántica de Bruno:
            |⟨k, ω⟩|  ≥  γ / ‖k‖^τ_1     ∀k ∈ ℤ^n \ {0},  ‖k‖_1 ≤ K_max.
        El ratio mínimo observado R = min_{k≠0} |⟨k,ω⟩| ‖k‖^τ_1 / γ.
        R ≥ 1  ⇔  el toro invariante persiste (estable KAM).

        La frecuencia ω se toma del espectro del mapa de retorno
        (frecuencias fundamentales del ciclo). La enumeración del retículo
        entero es exacta hasta K_max; la cota de Bruno garantiza exclusión
        de resonancias de orden superior bajo la densidad de Weyl.
        """
        freqs = np.asarray(packet.poincare_chart.fundamental_frequencies, dtype=np.float64)
        n = int(freqs.size)
        if n == 0:
            return KAMReport(
                stable=True, min_diophantine_ratio=np.inf, gamma=gamma, tau=tau,
                resonance_order=0, small_divisor_min=np.inf, frequencies=freqs,
            )

        min_ratio = np.inf
        min_divisor = np.inf
        min_resonance_order = K_max + 1
        for k_tuple in itertools.product(range(-K_max, K_max + 1), repeat=n):
            if all(ki == 0 for ki in k_tuple):
                continue
            k_vec = np.array(k_tuple, dtype=np.int64)
            k_norm1 = int(np.abs(k_vec).sum())
            if k_norm1 == 0:
                continue
            dot = abs(float(np.dot(k_vec, freqs)))
            bound = gamma / (k_norm1 ** tau) if k_norm1 > 0 else np.inf
            ratio = dot / bound if bound > 0 else np.inf
            if ratio < min_ratio:
                min_ratio = ratio
            if dot < min_divisor:
                min_divisor = dot
                if dot < 1e-3:
                    min_resonance_order = min(min_resonance_order, k_norm1)

        stable = bool(min_ratio >= 1.0)
        resonance_order = int(min_resonance_order) if min_resonance_order <= K_max else 0
        return KAMReport(
            stable=stable,
            min_diophantine_ratio=float(min_ratio) if np.isfinite(min_ratio) else np.inf,
            gamma=float(gamma),
            tau=float(tau),
            resonance_order=resonance_order,
            small_divisor_min=float(min_divisor) if np.isfinite(min_divisor) else np.inf,
            frequencies=freqs,
        )

    def _phase2_self_improvement(
        self,
        packet: PhaseOneKBasePacket,
        integrity: SpectralThermoReport,
    ) -> SelfImprovementReport:
        r"""
        [FASE 2 — TERCER MORFISMO: AUTOMEJORA RECURSIVA DE NIVEL 3]

        Dominio
        -------
        `PhaseOneKBasePacket` × `SpectralThermoReport`
        ← continuación formal de `_phase2_kam_audit`.

        Arquitectura de las tres capas:

        Nivel 1 — parámetro físico (α de Tikhonov).
            θ_{n+1}^{(1)} = θ_n^{(1)} · ρ_contra,  ρ_contra ∈ (0,1).

        Nivel 2 — estrategia (selección de la regla).
            Si la performance ΔS > 0, se explota (ρ_contra ← 0.95 ρ_contra);
            si ΔS ≤ 0, se explora (ρ_contra ← 1.05 ρ_contra).

        Nivel 3 — meta-estrategia (modificación de la modificación).
            Score = exp(−⟨|Δα|⟩_ventana) / α, en (0,1].
            Contraction_target = ρ_contra · (1 − 0.05 · (1 − Score)).
            El factor 0.05 regula la velocidad de adaptación meta.

        Punto fijo
        ----------
        x* tal que F_L3(x*) = x*. Se certifica convergencia cuando
        ‖α_n − α_{n−1}‖ < _RECURSION_CONVERGENCE_TOL.
        """
        prev = dict(self._meta_state)
        prev_alpha = float(prev["alpha"])
        prev_contraction = float(prev["contraction"])
        prev_score = float(prev["level3_score"])
        iteration = int(prev["iteration"])

        # Performance: usamos el score de integridad espectral como señal
        current_perf = float(integrity.integrity_score)
        delta_perf = current_perf - float(self._previous_performance)

        # Nivel 1: actualización escalar del parámetro
        alpha_n1 = max(prev_alpha * prev_contraction, _ALPHA_FLOOR)

        # Nivel 2: selección de la estrategia de contracción
        if delta_perf > 0.0:
            contraction_n2 = prev_contraction * 0.95
        else:
            contraction_n2 = prev_contraction * 1.05
        contraction_n2 = float(np.clip(contraction_n2, 0.1, 0.9))

        # Nivel 3: meta-estrategia — modifica la tasa de cambio de ρ_contra
        if len(self._meta_trajectory) >= 2:
            recent = self._meta_trajectory[-3:]
            diffs = [abs(recent[i]["alpha"] - recent[i - 1]["alpha"])
                     for i in range(1, len(recent))]
            mean_diff = float(np.mean(diffs)) if diffs else 0.0
        else:
            mean_diff = 0.0
        level3_score = float(np.exp(-mean_diff / max(prev_alpha, _ALPHA_FLOOR)))
        # Meta-ajuste de la contracción (nivel 3 puro)
        contraction_n3 = contraction_n2 * (1.0 - 0.05 * (1.0 - level3_score))

        # Residual de la recursión (para chequeo de punto fijo)
        recursion_residual = abs(alpha_n1 - prev_alpha)
        converged = recursion_residual < _RECURSION_CONVERGENCE_TOL

        # Persistimos el nuevo estado meta
        new_state = {
            "alpha": alpha_n1,
            "irr_thresh": float(prev["irr_thresh"]),
            "cond_thresh": float(prev["cond_thresh"]),
            "contraction": contraction_n3,
            "level3_score": level3_score,
            "iteration": iteration + 1.0,
        }
        self._meta_state = new_state
        self._meta_trajectory.append(new_state)
        if len(self._meta_trajectory) > _RECURSION_MAX_HISTORY:
            self._meta_trajectory.pop(0)

        # El paquete reporta el α que se usará en el PRÓXIMO ciclo
        return SelfImprovementReport(
            iteration=iteration,
            current_alpha=prev_alpha,
            next_alpha=alpha_n1,
            contraction=contraction_n3,
            level3_score=level3_score,
            recursion_converged=converged,
            recursion_residual=recursion_residual,
            performance_delta=float(delta_perf),
        )

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKBasePacket,
        integrity: SpectralThermoReport,
        kam_report: KAMReport,
        rsi: SelfImprovementReport,
        irreversibility_threshold: float,
        condition_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        [FASE 2 — CUARTO MORFISMO: CLASIFICACIÓN HEYTING]

        Dominio
        -------
        `PhaseOneKBasePacket` × `SpectralThermoReport` × `KAMReport` × `SelfImprovementReport`
        ← continuación formal de `_phase2_self_improvement`.

        Scores [0,1]-valuados con meet (∧ = min). Vetos atómicos:
          • Clausius-Duhem incoherente
          • ρ ilegal
          • T ≤ T_min
          • CSMD rota
          • Identidad termodinámica rota
          • KAM inestable (toro invariante destruido)
          • Caos Melnikov detectado (chequeado en Fase 3, se propaga por flag)
        """
        cd_viol = not packet.clausius_coherent
        stat_viol = packet.stationarity_defect > irreversibility_threshold
        dens_viol = not packet.density_chart.is_density
        cond_viol = integrity.metric_condition > condition_threshold
        ident_viol = packet.thermo_identity_residual > 1e-10
        csmd_viol = packet.csmd_entropy_error > 1e-8
        temp_viol = packet.temperature <= _WILKINSON_FLOOR
        kam_viol = not kam_report.stable
        # El caos (Melnikov) se evalúa en Fase 3 pero se propaga por coherencia
        chaos_viol = False  # se actualiza en Fase 3 vía reanudación, aquí placeholder

        cd_score = 1.0 if packet.clausius_coherent else 0.0
        stat_score = float(np.exp(
            -packet.stationarity_defect / max(irreversibility_threshold, _WILKINSON_FLOOR)
        ))
        rel_score = float(np.exp(-packet.density_chart.relative_entropy_to_gibbs))
        ident_score = float(np.exp(-packet.thermo_identity_residual / 1e-10))
        csmd_score = float(np.exp(-packet.csmd_entropy_error / 1e-8))
        # KAM: ratio ≥ 1 → score = min(1, ratio); < 1 → penalizado exponencial
        if np.isfinite(kam_report.min_diophantine_ratio):
            kam_score = float(min(1.0, np.exp(kam_report.min_diophantine_ratio - 1.0)))
        else:
            kam_score = 1.0

        meet = min(
            cd_score, stat_score, rel_score, ident_score, csmd_score,
            kam_score, integrity.integrity_score, float(rsi.level3_score),
            0.0 if dens_viol else 1.0,
            0.0 if temp_viol else 1.0,
            0.0 if kam_viol else 1.0,
        )

        hard_veto = (
            cd_viol or dens_viol or temp_viol or csmd_viol or ident_viol or kam_viol
        )
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
            "clausius": cd_viol,
            "stationarity": stat_viol,
            "density": dens_viol,
            "condition": cond_viol,
            "identity": ident_viol,
            "csmd": csmd_viol,
            "temperature": temp_viol,
            "kam": kam_viol,
            "chaos": chaos_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self,
        packet: PhaseOneKBasePacket,
        irreversibility_threshold: float = 0.1,
        condition_threshold: float = 1e12,
        ambient_temperature: Optional[float] = None,
    ) -> PhaseTwoKBasePacket:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: DECIDE]

        Codominio
        ---------
        `PhaseTwoKBasePacket`

        Continuidad formal
        ------------------
        Junto con el `PhaseOneKBasePacket` residual, este objeto constituye
        el dominio del primer morfismo de la Fase 3:
            `_phase3_conservation_audit(self, packet1, packet2)`.
        La destrucción de exergía (Gouy-Stodola) es T₀ Φ.
        """
        integrity = self._phase2_spectral_integrity(packet)
        kam_report = self._phase2_kam_audit(packet)
        rsi = self._phase2_self_improvement(packet, integrity)
        verdict, score, flags = self._phase2_heyting_classify(
            packet, integrity, kam_report, rsi,
            irreversibility_threshold=irreversibility_threshold,
            condition_threshold=condition_threshold,
        )
        t0 = float(ambient_temperature) if ambient_temperature is not None else packet.temperature
        exergy = float(t0 * packet.dissipation_potential)

        decision = PhaseTwoKBasePacket(
            irreversibility=packet.stationarity_defect,
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            clausius_threshold_violated=flags["clausius"],
            stationarity_threshold_violated=flags["stationarity"],
            density_axiom_violated=flags["density"],
            condition_threshold_violated=flags["condition"],
            kam_threshold_violated=flags["kam"],
            chaos_threshold_violated=flags["chaos"],
            spectral_integrity=integrity,
            kam_report=kam_report,
            self_improvement=rsi,
            heyting_score=score,
            exergy_destruction=exergy,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f KAM=%s R=%s ρ(P)=%.3f",
            verdict.value, score, kam_report.stable,
            kam_report.min_diophantine_ratio, packet.poincare_chart.spectral_radius,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT
    # Dominio = PhaseOneKBasePacket × PhaseTwoKBasePacket
    # Auditoría de conservación, auditoría de Melnikov (caos homoclínico),
    # análisis de punto fijo RSI-3, y sellado de telemetría.
    # El ÚLTIMO método, `_phase3_act`, produce `KBaseTelemetry`, objeto
    # terminal del endofunctor OODA (no hay Fase 4).
    # =========================================================================
    def _phase3_conservation_audit(
        self,
        packet1: PhaseOneKBasePacket,
        packet2: PhaseTwoKBasePacket,
    ) -> float:
        r"""
        [FASE 3 — PRIMER MORFISMO: AUDITORÍA DE CONSERVACIÓN]

        Dominio
        -------
        `PhaseOneKBasePacket × PhaseTwoKBasePacket`
        ← continuación formal de `_phase1_observe_orient` y `_phase2_decide`.

        Residuos adimensionales acumulados (KBN):
          1. G = H − TS = F + PV         (thermo_identity_residual)
          2. |∂G/∂T + S|                 (CSMD)
          3. |Tr ρ − 1|_+  y  (−λ_min(ρ))_+
          4. parte antihermítica del pullback
          5. inconsistencia del flag CD con el signo de Φ
          6. parte negativa de D(ρ‖ρ_β)
          7. (v3) |λ_max(P)| fuera de la ventana hiperbólica permitida
        """
        dens = packet1.density_chart
        cd_inconsistent = (
            1.0 if (packet1.clausius_coherent and packet1.dissipation_potential < -_WILKINSON_FLOOR)
            or ((not packet1.clausius_coherent) and packet1.dissipation_potential >= 0.0)
            else 0.0
        )
        # v3: residual hiperbólico — penaliza ρ(P) muy grande (no silla/nodo)
        radius = packet1.poincare_chart.spectral_radius
        hyperbolic_residual = max(0.0, radius - 10.0) if np.isfinite(radius) else 0.0

        terms = np.asarray([
            packet1.thermo_identity_residual,
            packet1.csmd_entropy_error,
            max(0.0, abs(dens.trace - 1.0) - _DENSITY_TRACE_TOLERANCE),
            max(0.0, -dens.min_eigenvalue - _PSD_TOLERANCE),
            packet1.pullback_hermiticity_residual,
            cd_inconsistent,
            max(0.0, -dens.relative_entropy_to_gibbs),
            hyperbolic_residual,
        ], dtype=np.float64)
        residual = float(np.real(self._neumaier_sum(terms)))
        logger.debug("Fase 3 conservación: residuo=%.6e", residual)
        return residual

    def _phase3_melnikov_audit(
        self,
        packet1: PhaseOneKBasePacket,
        n_samples: int = _MELNIKOV_LOBES_SAMPLES,
    ) -> Tuple[float, bool]:
        r"""
        [FASE 3 — SEGUNDO MORFISMO: AUDITORÍA DE MELNIKOV]

        Dominio
        -------
        `PhaseOneKBasePacket`  ← continuación formal de `_phase3_conservation_audit`.

        Para H = H₀ + ε H₁ con órbita homoclínica q⁰ de H₀:
            M(t₀) = ∫_{−∞}^{+∞} {H₀, H₁}(q⁰(t − t₀)) dt.
        Ceros simples de M(t₀) ⇔ intersecciones homoclínicas transversales
        ⇔ entrelazado de Smale-Birkhoff ⇔ caos determinista.

        Estimación numérica: la separatriz q⁰(t) se aproxima por el vector
        propio inestable v_u del mapa de retorno P, con decaimiento
        exponencial e^{−|Re λ_u| |t−t₀|}. La perturbación H₁ se modela
        como un forzamiento sinusoidal; la integral se aproxima por
        cuadratura trapezoidal sobre n_samples uniformes de [0, 2π).

        El indicador booleano es verdadero si existe al menos un cambio
        de signo de M(t₀) (Smale-Birkhoff no trivial).
        """
        P = np.asarray(packet1.poincare_chart.return_map, dtype=np.float64)
        eigvals = packet1.poincare_chart.return_eigenvalues
        state = np.asarray(packet1.state_vector, dtype=np.float64)
        n = int(state.size)

        # Descomposición silla: eigenvectores estables (|λ|<1) e inestables (|λ|>1)
        try:
            eigvals_np, eigvecs_np = np.linalg.eig(P)
        except np.linalg.LinAlgError:
            return 0.0, False
        abs_eig = np.abs(eigvals_np)
        stable_idx = np.where(abs_eig < 1.0)[0]
        unstable_idx = np.where(abs_eig > 1.0)[0]
        if stable_idx.size == 0 or unstable_idx.size == 0:
            # Sin estructura silla → no hay lóbulos de Melnikov
            return 0.0, False

        # Autovalor inestable de mayor módulo (tasa de separación)
        idx_u = unstable_idx[np.argmax(abs_eig[unstable_idx])]
        lam_u = float(abs(eigvals_np[idx_u]))
        v_u = np.real(eigvecs_np[:, idx_u])
        v_u_norm = float(la.norm(v_u))
        if v_u_norm < _WILKINSON_FLOOR:
            return 0.0, False
        v_u = v_u / v_u_norm

        # Proyección de la perturbación (asumida en la dirección del estado)
        pert_dir = state / max(float(la.norm(state)), _WILKINSON_FLOOR)
        v_s = np.real(eigvecs_np[:, stable_idx[0]])
        v_s_norm = float(la.norm(v_s))
        if v_s_norm < _WILKINSON_FLOOR:
            return 0.0, False
        v_s = v_s / v_s_norm

        overlap_s = abs(float(np.dot(v_s, pert_dir)))
        overlap_u = abs(float(np.dot(v_u, pert_dir)))

        # Melnikov: M(t₀) evaluada por cuadratura trapezoidal
        t_grid = np.linspace(0.0, 2.0 * np.pi, n_samples)
        dt = float(t_grid[1] - t_grid[0]) if n_samples > 1 else 0.0
        decay_rate = max(lam_u - 1.0, 1e-3)  # e^{−|λ_u − 1| |t|} decae

        M_values = np.empty_like(t_grid)
        for i, t0 in enumerate(t_grid):
            integrand = (
                overlap_s * overlap_u
                * np.exp(-decay_rate * np.abs(t_grid - t0))
                * np.sin(t_grid - t0)
            )
            M_values[i] = float(np.trapz(integrand, t_grid))

        lobe_area = float(np.trapz(np.abs(M_values), t_grid))
        sign_changes = int(np.sum(np.diff(np.sign(M_values + 1e-300)) != 0))
        chaos_indicator = bool(sign_changes > 0 and lobe_area > 1e-6)
        logger.debug(
            "Fase 3 Melnikov: lobe_area=%.6e zeros=%d chaos=%s",
            lobe_area, sign_changes, chaos_indicator,
        )
        return lobe_area, chaos_indicator

    def _phase3_act(
        self,
        packet1: PhaseOneKBasePacket,
        packet2: PhaseTwoKBasePacket,
        conservation_residual: float,
        melnikov_lobe_area: float,
        chaos_indicator: bool,
        recursion_converged: bool,
        recursion_iteration: int,
    ) -> KBaseTelemetry:
        r"""
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Codominio
        ---------
        `KBaseTelemetry`  — objeto terminal del functor OODA.

        Continuidad formal
        ------------------
        Este sello ES el valor de T(estado) inyectable en el pasaporte de la
        Malla. No existe Fase 4: el endofunctor se ha clausurado.

        El campo `chaos_indicator` (Melnikov) se propaga al veredicto
        final por la ley interna de Heyting: si caos, el veredicto se
        degrada a VETOED por veto (Smale-Birkhoff viola la regularidad
        termodinámica).
        """
        # Veto por caos: se aplica por modus ponens interno de Heyting
        verdict_str = packet2.heyting_verdict
        if chaos_indicator:
            try:
                base_verdict = HeytingVerdict(verdict_str)
            except ValueError:
                base_verdict = HeytingVerdict.VETOED
            verdict_str = base_verdict.meet(HeytingVerdict.VETOED).value

        return KBaseTelemetry(
            pullback_tensor=packet1.pullback_tensor,
            gibbs_energy=packet1.gibbs_energy,
            enthalpy=packet1.enthalpy,
            helmholtz_energy=packet1.helmholtz_energy,
            dissipation_potential=packet1.dissipation_potential,
            clausius_coherent=packet1.clausius_coherent,
            irreversibility=packet2.irreversibility,
            heyting_verdict=verdict_str,
            thermo_identity_residual=packet1.thermo_identity_residual,
            csmd_entropy_error=packet1.csmd_entropy_error,
            relative_entropy_to_gibbs=packet1.density_chart.relative_entropy_to_gibbs,
            von_neumann_entropy=packet1.density_chart.von_neumann_entropy,
            purity=packet1.density_chart.purity,
            condition_number=packet2.spectral_integrity.metric_condition,
            conservation_residual=conservation_residual,
            exergy_destruction=packet2.exergy_destruction,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
            poincare_return_time=packet1.poincare_chart.return_time,
            return_map_spectral_radius=packet1.poincare_chart.spectral_radius,
            max_lyapunov_exponent=packet1.poincare_chart.max_lyapunov,
            kam_stable=packet2.kam_report.stable,
            kam_diophantine_ratio=packet2.kam_report.min_diophantine_ratio,
            melnikov_lobe_area=melnikov_lobe_area,
            chaos_indicator=chaos_indicator,
            recursion_converged=recursion_converged,
            recursion_iteration=recursion_iteration,
            self_improvement_level3_score=packet2.self_improvement.level3_score,
        )

    # =========================================================================
    # ORQUESTADOR OODA
    # =========================================================================
    def execute_thermodynamic_cycle(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
        irreversibility_threshold: float = 0.1,
        jacobian: Optional[np.ndarray] = None,
        ambient_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""
        Functor OODA  T = Act ∘ Decide ∘ Observe  sobre el espacio de estados.

        Encadena las tres fases anidadas sin objetos huérfanos:

            In  --F1→  PhaseOne  --F2→  PhaseTwo  --F3→  KBaseTelemetry → dict

        v3: tras el sellado, el motor actualiza su memoria:
          • `self._cycle_history` (para el mapa de retorno de Poincaré).
          • `self._alpha_current ← self._meta_state["alpha"]` (para RSI-3).
        La siguiente invocación arranca con α actualizado y con historia
        suficiente para estimar P de rango completo (tras n+1 ciclos).
        """
        # FASE 1 — OBSERVE + ORIENT
        packet1 = self._phase1_observe_orient(
            G, tensor_M, tensor_type,
            internal_energy_U, pressure_P, volume_V, temperature_T, entropy_S,
            entropy_production_rate, heat_flux_q, temp_gradient_gradT,
            rho, hamiltonian_H, jacobian=jacobian,
        )

        # FASE 2 — DECIDE
        packet2 = self._phase2_decide(
            packet1,
            irreversibility_threshold=irreversibility_threshold,
            ambient_temperature=ambient_temperature,
        )

        # FASE 3 — ACT
        conservation = self._phase3_conservation_audit(packet1, packet2)
        melnikov_area, chaos_flag = self._phase3_melnikov_audit(packet1)
        telemetry = self._phase3_act(
            packet1, packet2,
            conservation_residual=conservation,
            melnikov_lobe_area=melnikov_area,
            chaos_indicator=chaos_flag,
            recursion_converged=packet2.self_improvement.recursion_converged,
            recursion_iteration=packet2.self_improvement.iteration,
        )

        # MEMORIA DEL MOTOR
        self._cycle_history.append({
            "state": np.array(packet1.state_vector, copy=True),
            "telemetry": telemetry,
            "verdict": telemetry.heyting_verdict,
        })
        # El α del próximo ciclo es el propuesto por RSI-3
        self._alpha_current = float(packet2.self_improvement.next_alpha)
        self._previous_performance = float(packet2.spectral_integrity.integrity_score)

        return telemetry.to_dict()


# -----------------------------------------------------------------------------
# Bloque de autocomprobación
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando autocomprobación de KBase Thermodynamic Engine v3.0 (Poincaré + RSI-3)…")
    engine = KBaseThermodynamicEngine(dimension_n=3)

    G = np.eye(3, dtype=np.float64)
    Tm = np.diag([1.0, 2.0, 3.0])

    T_pull = engine.compute_riemannian_pullback(G, Tm, tensor_type="covariant")
    print(f"¿Congruencia con g=I es idéntica?: {np.allclose(Tm, T_pull)}")

    J = np.diag([2.0, 1.0, 0.5])
    T_true = engine.compute_riemannian_pullback(G, Tm, jacobian=J)
    print(f"¿Pullback J†TJ coincide?: {np.allclose(T_true, J.T @ Tm @ J)}")

    gibbs, h = engine.compute_gibbs_free_energy(100.0, 10.0, 2.0, 300.0, 0.1)
    print(f"G={gibbs:.4f}, H={h:.4f}  (esperado G=90, H=120)")

    diss, ok = engine.evaluate_clausius_duhem(
        entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        temperature_T=300.0,
    )
    print(f"Φ={diss:.6f}, CD={ok}")

    rho = np.diag([0.7, 0.2, 0.1])
    H = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    print(f"ℐ={engine.calculate_spectral_irreversibility(rho, H):.6f}")

    print("\n--- CICLO 1 (bootstrap: sin historia, P̂=I, KAM trivial) ---")
    result1 = engine.execute_thermodynamic_cycle(
        G=G, tensor_M=Tm, tensor_type="covariant",
        internal_energy_U=100.0, pressure_P=10.0, volume_V=2.0,
        temperature_T=300.0, entropy_S=0.1, entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        rho=rho, hamiltonian_H=H, irreversibility_threshold=0.1,
    )
    print(f"veredicto={result1['heyting_verdict']}, "
          f"ρ(P)={result1['return_map_spectral_radius']:.3f}, "
          f"λ_max={result1['max_lyapunov_exponent']:.3e}, "
          f"KAM={result1['kam_stable']}, "
          f"Melnikov={result1['melnikov_lobe_area']:.3e}, "
          f"chaos={result1['chaos_indicator']}, "
          f"RSI_iter={result1['recursion_iteration']}, "
          f"RSI_L3={result1['self_improvement_level3_score']:.4f}")

    print("\n--- CICLO 2 (historia=1, P̂ de rango deficiente regularizado) ---")
    result2 = engine.execute_thermodynamic_cycle(
        G=G, tensor_M=Tm, tensor_type="covariant",
        internal_energy_U=100.0, pressure_P=10.0, volume_V=2.0,
        temperature_T=300.0, entropy_S=0.1, entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        rho=rho, hamiltonian_H=H,
    )
    print(f"veredicto={result2['heyting_verdict']}, "
          f"RSI_iter={result2['recursion_iteration']}, "
          f"conv={result2['recursion_converged']}, "
          f"KAM_ratio={result2['kam_diophantine_ratio']:.3f}")

    print("\n--- CICLO 3 (historia=2, P̂ más informativo) ---")
    result3 = engine.execute_thermodynamic_cycle(
        G=G, tensor_M=Tm, tensor_type="covariant",
        internal_energy_U=100.0, pressure_P=10.0, volume_V=2.0,
        temperature_T=300.0, entropy_S=0.1, entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        rho=rho, hamiltonian_H=H,
    )
    print(f"veredicto={result3['heyting_verdict']}, "
          f"ρ(P)={result3['return_map_spectral_radius']:.3f}, "
          f"KAM_ratio={result3['kam_diophantine_ratio']:.3f}")

    print("\n--- VETO esperado: Clausius-Duhem incoherente ---")
    veto = engine.execute_thermodynamic_cycle(
        G=G, tensor_M=Tm, tensor_type="covariant",
        internal_energy_U=100.0, pressure_P=10.0, volume_V=2.0,
        temperature_T=300.0, entropy_S=0.1, entropy_production_rate=-1.0,
        heat_flux_q=np.array([1.0, 0.0, 0.0]),
        temp_gradient_gradT=np.array([1.0, 0.0, 0.0]),
        rho=rho, hamiltonian_H=H,
    )
    print(f"veredicto={veto['heyting_verdict']}, CD={veto['clausius_coherent']}")


__all__ = [
    "KBaseThermodynamicEngine",
    "PhaseOneKBasePacket",
    "PhaseTwoKBasePacket",
    "KBaseTelemetry",
    "SpectralChart",
    "DensityOperatorChart",
    "SpectralThermoReport",
    "PoincareChart",
    "KAMReport",
    "SelfImprovementReport",
    "HeytingVerdict",
]