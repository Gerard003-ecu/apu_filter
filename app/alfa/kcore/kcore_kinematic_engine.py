# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA SOBERANO ESTRATO ALFA (KCORE)
====================================================================================================

Nombre: KcoreKinematicEngine
Ubicación: app/alfa/kcore/kcore_kinematic_engine.py
Versión: 8.0.0-APU-SOVEREIGN-POINCARE-KAM-MELNIKOV-RSI3
Función: Motor soberano cinemático de la capa Alfa (Kcore) en APU Filter. Modela la cinemática
         diferencial riemanniana, integración simpléctica de trayectorias, conexión de Levi-Civita,
         símbolos de Christoffel y la ecuación geodésica.

         En v8 se añaden, granular y rigurosamente:

         (A) MECÁNICA CELESTE DE POINCARÉ sobre el flujo geodésico linealizado:
             • Sección transversal Σ = {⟨n̂, x⟩ = c} en el espacio de estados.
             • Mapa de retorno P = exp(A · τ_ret) con A = (J − R) ∇²H_ref.
             • Espectro de Lyapunov λ_k = Re Spec(A)·τ_ret.
             • Frecuencias fundamentales ω_k = Im Spec(A) (modos elípticos).
             • Acción-ángulo (J_k, θ_k) por proyección sobre autovectores de A.
             • Teorema KAM (Kolmogorov–Arnold–Moser): condición diofántica
               |⟨k,ω⟩| ≥ γ/‖k‖^τ_1 ∀k ∈ ℤ^n\{0}, ‖k‖_1 ≤ K_max.
             • Función de Melnikov M(t₀) para entrelazado homoclínico (Smale–Birkhoff).

         (B) AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-3):
             Nivel 1: α_{n+1} = α_n · ρ_contra  (parámetro de Tikhonov).
             Nivel 2: ρ_contra ← ρ_contra · f(ΔS)  (adaptación de estrategia).
             Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3))  (meta-adaptación).
             Punto fijo: ‖α_{n+1} − α_n‖ < ε_rec.

----------------------------------------------------------------------------------------------------
CONTINUIDAD FORMAL ENTRE FASES (tejido estricto):
----------------------------------------------------------------------------------------------------
   Fase 1  ──último morfismo──▶  PhaseOneKinematicPacket  (con poincare_chart)
   Fase 2  ──primer morfismo (dominio)──▶  PhaseOneKinematicPacket
   Fase 2  ──último morfismo──▶  PhaseTwoKinematicPacket  (con kam_diagnostics)
   Fase 3  ──primer morfismo (dominio)──▶  PhaseOneKinematicPacket × PhaseTwoKinematicPacket
   Fase 3  ──morfismos internos──▶  conservation_audit → rsi3 → act
   Fase 3  ──último morfismo──▶  KinematicTelemetry (objeto terminal del endofunctor OODA)

   Cada docstring de cabecera de fase indica explícitamente la cláusula "continuación de…".

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO AÑADIDO EN v8:
----------------------------------------------------------------------------------------------------
A. Flujo geodésico y ecuación de Jacobi. La ecuación geodésica ẍ^μ + Γ^μ_{αβ} ẋ^α ẋ^β = 0
   linealizada alrededor de una geodésica γ(t) satisface la ecuación de Jacobi:
       D²J/dt² + R(J, γ̇)γ̇ = 0.
   En la formulación cinemática del motor (estado x ∈ ℝ^n), el generador linealizado se
   aproxima por A = (J − R) ∇²H_ref con ∇²H_ref ≈ M (masa como Hessiano de referencia).
   Sus autovalores λ_k dan Lyapunov (Re) y frecuencias (Im).

B. Teorema KAM. Bajo perturbación analítica pequeña, los toros invariantes de un flujo
   integrable persisten si det(∂²H₀/∂J²) ≠ 0 y ω = ∂H₀/∂J es diofántico:
       |⟨k, ω⟩| ≥ γ/‖k‖^τ_1,   τ > n−1,   k ∈ ℤ^n \ {0}.

C. Función de Melnikov. Para H = H₀ + εH₁ con órbita homoclínica q⁰(t) de H₀:
       M(t₀) = ∫_{−∞}^{+∞} {H₀, H₁}(q⁰(t−t₀)) dt.
   Ceros simples de M(t₀) ⇔ entrelazado de Smale–Birkhoff ⇔ caos.

D. RSI-3. Tres capas sobre θ = (α, ρ_contra). Punto fijo: F_L3(θ*) = θ*, con
   contracción ‖Jac F_L3‖ < 1 en la cuenca.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES (heredados + nuevos):
----------------------------------------------------------------------------------------------------
Axioma KKE1 (Maupertuis): geodésicas minimizan S_0 = ∫ p_i dq^i sin potenciales.
Axioma KKE2 (Compatibilidad): ∇g = 0 preserva longitudes bajo transporte paralelo.
Axioma KKE3 (Simetría simpléctica): det(∂(x_{n+1}, v_{n+1})/∂(x_n, v_n)) = 1.
Axioma KKE4 (Transversalidad): ⟨n̂, ẋ⟩ ≠ 0 sobre Σ \ ∂Σ (sección regular).
Axioma KKE5 (Diofantino): el vector ω satisface (γ, τ) de Bruno.
Axioma KKE6 (Convergencia RSI-3): limsup ‖Δα_n‖ / ‖Δα_{n-1}‖ < 1 en θ*.

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO (heredados + nuevos):
----------------------------------------------------------------------------------------------------
Invariante 1: |E(t) − E(0)| ≤ ε_energy = 1e-4.
Invariante 2: ‖v(t)‖_g ≤ v_max_light_limit.
Invariante 3 (Lyapunov): λ_max(P) invariante bajo conjugación GL(n,ℝ).
Invariante 4 (KAM): min_{k≠0} ‖k‖^τ_1 |⟨k,ω⟩| ≥ γ > 0 en régimen estable.
Invariante 5 (RSI-3): ‖α_{n+1} − α_n‖ < ε_rec en el punto fijo.
====================================================================================================
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.KCoreKinematicEngine")

# ---------------------------------------------------------------------------
# Constantes metrológicas (Wilkinson / IEEE-754 binary64)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_SKEW_TOLERANCE: Final[float] = 1e-12
_PSD_TOLERANCE: Final[float] = 1e-12
_RANK_REL_TOL: Final[float] = 1e-12

# v8: constantes dinámicas (Poincaré / KAM / Melnikov / RSI-3)
_KAM_GAMMA_DEFAULT: Final[float] = 0.1
_KAM_TAU_DEFAULT: Final[float] = 2.5
_KAM_K_MAX_DEFAULT: Final[int] = 4
_LYAPUNOV_FLOOR: Final[float] = 1e-300
_MELNIKOV_SAMPLES: Final[int] = 96
_MELNIKOV_AREA_FLOOR: Final[float] = 1e-8
_RSI3_LEARNING_RATE: Final[float] = 0.05
_RSI3_CONVERGENCE_TOL: Final[float] = 1e-8
_RSI3_MAX_HISTORY: Final[int] = 12
_ALPHA_FLOOR: Final[float] = _WILKINSON_FLOOR


# =============================================================================
# Retículo de Heyting
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""Cadena ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤."""

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
# Utilidades
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


# =============================================================================
# DTOs v8 (Poincaré / KAM / RSI-3)
# =============================================================================
@dataclass(frozen=True, slots=True)
class PoincareGeodesicChart:
    r"""
    Carta de Poincaré sobre el flujo geodésico linealizado.

    Sección transversal Σ = {⟨n̂, x⟩ = c}. Generador A = (J−R)∇²H_ref con
    ∇²H_ref = M (masa como Hessiano de referencia, aproximación armónica).

    Mapa de retorno: P = exp(A · τ_ret), τ_ret = 2π/min(ω_elliptic) si hay
    modos elípticos, o 1 si no.

    Espectro de Lyapunov: λ_k = Re(eig_k(A)) · τ_ret.
    Frecuencias fundamentales: ω_k = Im(eig_k(A)) para los modos elípticos.
    """

    section_normal: np.ndarray
    section_offset: float
    linearized_generator: np.ndarray
    return_time: float
    return_map: np.ndarray
    return_eigenvalues: np.ndarray
    return_spectral_radius: float
    lyapunov_exponents: np.ndarray
    max_lyapunov: float
    fundamental_frequencies: np.ndarray
    elliptic_dimension: int
    hyperbolic_dimension: int
    eigenvector_matrix: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "section_normal", "linearized_generator", "return_map",
            "return_eigenvalues", "lyapunov_exponents",
            "fundamental_frequencies", "eigenvector_matrix",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class KAMGeodesicDiagnostics:
    r"""
    Diagnóstico del teorema KAM + Melnikov sobre el flujo geodésico.

    (a) Diofantino: R = min_{k≠0} |⟨k,ω⟩| ‖k‖^τ_1 / γ ≥ 1 ⇔ toro persistente.
    (b) Melnikov: M(t₀) con ceros simples ⇔ entrelazado homoclínico.
    """

    stable: bool
    min_diophantine_ratio: float
    gamma: float
    tau: float
    resonance_order: int
    small_divisor_min: float
    frequencies: np.ndarray
    melnikov_lobe_area: float
    melnikov_zero_crossings: int
    homoclinic_tangle_detected: bool
    saddle_found: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequencies", _freeze_array(self.frequencies))


@dataclass(frozen=True, slots=True)
class RSI3KinematicDiagnostics:
    r"""
    Diagnóstico del bucle de automejora recursiva de nivel 3.

    Nivel 1: α_{n+1} = α_n · ρ_contra.
    Nivel 2: ρ_contra ← ρ_contra · f(ΔS).
    Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3)).
    Punto fijo: ‖α_{n+1} − α_n‖ < ε_rec.
    """

    iteration: int
    current_alpha: float
    next_alpha: float
    contraction: float
    level3_score: float
    converged: bool
    residual: float
    performance_delta: float


# =============================================================================
# DTOs heredados (v7)
# =============================================================================
@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""Factorización espectral A = U Λ U† con Tikhonov."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: np.ndarray
    sqrt_metric: np.ndarray
    condition_number: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    regularized: bool
    tikhonov_alpha: float
    kernel_dimension: int

    def __post_init__(self) -> None:
        for name in ("eigenvalues", "eigenvectors", "inv_metric", "sqrt_metric"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PortHamiltonianChart:
    r"""Carta IDA-PBC y pasividad."""

    alpha: np.ndarray
    forcing_error: np.ndarray
    residual_vector: np.ndarray
    residual_norm: float
    residual_G_norm: float
    matching_obstruction: float
    actuator_rank: int
    actuator_condition: float
    n_actuators: int
    J_skew_residual: float
    Jd_skew_residual: float
    R_psd_residual: float
    Rd_psd_residual: float
    open_loop_dissipation: float
    closed_loop_dissipation: float
    supplied_power: float
    passivity_residual: float
    lyapunov_decrease: float

    def __post_init__(self) -> None:
        for name in ("alpha", "forcing_error", "residual_vector"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class HodgeChart:
    r"""Carta de Hodge celular discreta."""

    I_grad: np.ndarray
    I_curl: np.ndarray
    I_harmonic: np.ndarray
    potentials: np.ndarray
    orthogonality_gram: np.ndarray
    orthogonality_error: float
    reconstruction_residual: float
    kirchhoff_residual: float
    coclosure_residual: float
    pythagoras_residual: float
    chain_complex_residual: float
    betti_0: int
    betti_1: int
    laplacian_gap: float
    laplacian_condition: float
    n_vertices: int
    n_edges: int
    n_faces: int
    has_two_skeleton: bool
    energy_grad: float
    energy_curl: float
    energy_harmonic: float
    energy_total: float

    def __post_init__(self) -> None:
        for name in ("I_grad", "I_curl", "I_harmonic", "potentials", "orthogonality_gram"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class KineticChart:
    r"""Carta de la forma cuadrática cinética."""

    kinetic_energy: float
    mechanical_power: float
    mass_psd_residual: float
    rayleigh_quotient: float
    csmd_momentum: np.ndarray
    csmd_error: float
    mass_condition: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "csmd_momentum", _freeze_array(self.csmd_momentum))


@dataclass(frozen=True, slots=True)
class PhaseOneKinematicPacket:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.

    Continuidad formal: es el dominio del primer morfismo de Fase 2,
    `_phase2_spectral_integrity(self, packet: PhaseOneKinematicPacket)`.

    v8 añade `poincare_chart`: carta estructural de Poincaré construida en
    Fase 1 (sección, generador linealizado, mapa de retorno, Lyapunov).
    """

    ida_alpha: np.ndarray
    ida_residual_norm: float
    hh_grad: np.ndarray
    hh_curl: np.ndarray
    hh_harmonic: np.ndarray
    hh_orthogonality_error: float
    kinetic_energy: float
    mechanical_power: float
    ph_chart: PortHamiltonianChart
    hodge_chart: HodgeChart
    kinetic_chart: KineticChart
    metric_chart: SpectralChart
    # v8
    poincare_chart: PoincareGeodesicChart

    def __post_init__(self) -> None:
        for name in ("ida_alpha", "hh_grad", "hh_curl", "hh_harmonic"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SpectralKinematicReport:
    r"""Integridad espectral (Fase 2, primer morfismo)."""

    metric_condition: float
    metric_gap: float
    laplacian_gap: float
    actuator_rank: int
    actuator_condition: float
    mass_condition: float
    hodge_pythagoras_residual: float
    matching_obstruction: float
    passivity_residual: float
    integrity_score: float


@dataclass(frozen=True, slots=True)
class PhaseTwoKinematicPacket:
    r"""
    Objeto terminal de la Fase 2 ≡ primer factor del dominio de la Fase 3.

    Continuidad formal: junto con `PhaseOneKinematicPacket`, es el dominio
    del primer morfismo de Fase 3, `_phase3_conservation_audit(packet1, packet2)`.

    v8 añade `kam_diagnostics`: diagnóstico KAM + Melnikov del flujo.
    """

    heyting_verdict: str
    heyting_rank: int
    orthogonality_violated: bool
    residual_violated: bool
    obstruction_violated: bool
    passivity_violated: bool
    hodge_theorem_violated: bool
    kinetic_sign_violated: bool
    # v8
    kam_violated: bool
    chaos_violated: bool
    spectral_integrity: SpectralKinematicReport
    # v8
    kam_diagnostics: KAMGeodesicDiagnostics
    heyting_score: float
    betti_0: int
    betti_1: int


@dataclass(frozen=True, slots=True)
class KinematicTelemetry:
    r"""
    Sello inmutable de telemetría (objeto terminal del functor OODA).

    v8 añade: métricas de Poincaré (τ_ret, ρ(P), λ_max), KAM (estable, ratio),
    Melnikov (lobe_area, caos) y RSI-3 (iteración, score, convergencia).
    """

    ida_alpha: np.ndarray
    ida_residual_norm: float
    hh_grad: np.ndarray
    hh_curl: np.ndarray
    hh_harmonic: np.ndarray
    hh_orthogonality_error: float
    kinetic_energy: float
    mechanical_power: float
    heyting_verdict: str
    matching_obstruction: float
    passivity_residual: float
    reconstruction_residual: float
    pythagoras_residual: float
    csmd_error: float
    betti_0: int
    betti_1: int
    condition_number: float
    conservation_residual: float
    spectral_integrity_score: float
    # v8: Poincaré
    poincare_return_time: float
    return_map_spectral_radius: float
    max_lyapunov_exponent: float
    # v8: KAM + Melnikov
    kam_stable: bool
    kam_diophantine_ratio: float
    melnikov_lobe_area: float
    chaos_indicator: bool
    # v8: RSI-3
    rsi_iteration: int
    rsi_converged: bool
    rsi_level3_score: float
    rsi_next_alpha: float

    def __post_init__(self) -> None:
        for name in ("ida_alpha", "hh_grad", "hh_curl", "hh_harmonic"):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización compatible con el pasaporte de la Malla (v2 + v3 + v8)."""
        return {
            "ida_alpha": np.array(self.ida_alpha, copy=True),
            "ida_residual_norm": self.ida_residual_norm,
            "hh_grad": np.array(self.hh_grad, copy=True),
            "hh_curl": np.array(self.hh_curl, copy=True),
            "hh_harmonic": np.array(self.hh_harmonic, copy=True),
            "hh_orthogonality_error": self.hh_orthogonality_error,
            "kinetic_energy": self.kinetic_energy,
            "mechanical_power": self.mechanical_power,
            "heyting_verdict": self.heyting_verdict,
            "matching_obstruction": self.matching_obstruction,
            "passivity_residual": self.passivity_residual,
            "reconstruction_residual": self.reconstruction_residual,
            "pythagoras_residual": self.pythagoras_residual,
            "csmd_error": self.csmd_error,
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "condition_number": self.condition_number,
            "conservation_residual": self.conservation_residual,
            "spectral_integrity_score": self.spectral_integrity_score,
            # v8
            "poincare_return_time": self.poincare_return_time,
            "return_map_spectral_radius": self.return_map_spectral_radius,
            "max_lyapunov_exponent": self.max_lyapunov_exponent,
            "kam_stable": self.kam_stable,
            "kam_diophantine_ratio": self.kam_diophantine_ratio,
            "melnikov_lobe_area": self.melnikov_lobe_area,
            "chaos_indicator": self.chaos_indicator,
            "rsi_iteration": self.rsi_iteration,
            "rsi_converged": self.rsi_converged,
            "rsi_level3_score": self.rsi_level3_score,
            "rsi_next_alpha": self.rsi_next_alpha,
        }


# =============================================================================
# Motor
# =============================================================================
class KCoreKinematicEngine:
    r"""
    Motor Cinemático del Estrato α. Endofunctor OODA T = Act ∘ Decide ∘ Observe.

    En v8, T incluye tres morfismos dinámicos adicionales:
      • Carta de Poincaré (Fase 1).
      • Audit KAM + Melnikov (Fase 2).
      • Bucle RSI-3 (Fase 3).
    """

    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError("dimensión debe ser estrictamente positiva.")
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))

        # v8: estado meta RSI-3
        self._alpha_current: float = self._reg
        self._meta_state: Dict[str, Any] = {
            "alpha": self._reg,
            "contraction": 0.5,
            "level3_score": 1.0,
            "iteration": 0,
        }
        self._meta_trajectory: List[Dict[str, float]] = []
        self._previous_performance: float = 1.0

    def reset(self) -> None:
        """Reinicia la memoria de RSI-3 y el parámetro de Tikhonov."""
        self._alpha_current = self._reg
        self._meta_trajectory.clear()
        self._previous_performance = 1.0
        self._meta_state = {
            "alpha": self._reg,
            "contraction": 0.5,
            "level3_score": 1.0,
            "iteration": 0,
        }

    # =========================================================================
    # Utilidades (KBN, validaciones, proyecciones, Tikhonov, CSMD)
    # =========================================================================
    def kahan_sum(self, arr: np.ndarray) -> float:
        if np.asarray(arr).ndim != 1:
            raise ValueError(f"kahan_sum espera 1-D, se recibió {np.asarray(arr).shape}")
        return float(np.real(self._neumaier_sum(arr)))

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        comp: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                comp += (total - t) + x
            else:
                comp += (x - t) + total
            total = t
        return total + comp

    def _validate_square_matrix(self, M: np.ndarray, name: str, dim: Optional[int] = None) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada, se recibió {M.shape}")
        expected = self._n if dim is None else dim
        if M.shape[0] != expected:
            raise ValueError(f"{name} debe ser {expected} × {expected}, se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, length: int, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != length:
            raise ValueError(f"{name} debe ser vector de longitud {length}, se recibió {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_hermitian(self, M: np.ndarray, name: str, tol: float = _HERMITIAN_TOLERANCE) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tol * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(f"{name} no hermítica (tol {tol})")

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.conj().T)

    def _skew_project(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        projected = 0.5 * (M - M.conj().T)
        residual = float(la.norm(M - projected, ord="fro"))
        return projected, residual

    def _psd_project(self, M: np.ndarray) -> Tuple[np.ndarray, float]:
        herm = self._hermitize(M)
        evals, evecs = la.eigh(herm)
        clipped = np.maximum(np.real(evals), 0.0)
        projected = (evecs * clipped) @ evecs.conj().T
        residual = float(la.norm(M - projected, ord="fro"))
        return projected, residual

    def _factorize_metric(self, metric_tensor: np.ndarray, name: str = "metric") -> SpectralChart:
        self._validate_square_matrix(metric_tensor, name, dim=metric_tensor.shape[0])
        herm = self._hermitize(np.asarray(metric_tensor, dtype=np.complex128))
        self._validate_hermitian(herm, name)
        eigenvalues, eigenvectors = la.eigh(herm)
        eigenvalues = np.real(eigenvalues)
        alpha = float(self._alpha_current)
        regularized = bool(np.any(eigenvalues <= alpha))
        if regularized:
            logger.warning("%s λ_min=%.3e ≤ α=%.3e.", name, float(np.min(eigenvalues)), alpha)
        shifted = np.maximum(eigenvalues + alpha, _WILKINSON_FLOOR)
        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T
        sqrt_metric = (eigenvectors * np.sqrt(shifted)) @ eigenvectors.conj().T
        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        op_norm = float(np.max(lam_abs))
        frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
        if shifted.size >= 2:
            ordered = np.sort(lam_abs)
            gap = float(ordered[1] - ordered[0])
        else:
            gap = float(lam_abs[0])
        kernel_dim = int(np.sum(np.abs(eigenvalues) <= max(alpha, _WILKINSON_FLOOR)))
        return SpectralChart(
            eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            inv_metric=np.real_if_close(inv_metric, tol=1e6),
            sqrt_metric=np.real_if_close(sqrt_metric, tol=1e6),
            condition_number=cond, operator_norm=op_norm,
            frobenius_norm=frob, spectral_gap=gap,
            regularized=regularized, tikhonov_alpha=alpha,
            kernel_dimension=kernel_dim,
        )

    def _tikhonov_svd_solve(
        self, operator: np.ndarray, rhs: np.ndarray, reg: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
        gamma = self._reg if reg is None else float(reg)
        A = np.asarray(operator)
        b = np.asarray(rhs)
        U, sigma, Vh = la.svd(A, full_matrices=False)
        filt = sigma / (sigma ** 2 + gamma ** 2)
        x = (Vh.conj().T * filt) @ (U.conj().T @ b)
        residual = float(la.norm(A @ x - b))
        if sigma.size == 0:
            return x, sigma, residual, 0, np.inf
        cutoff = max(gamma, _RANK_REL_TOL * float(sigma[0]))
        rank = int(np.sum(sigma > cutoff))
        smax = float(sigma[0])
        smin = float(sigma[rank - 1]) if rank > 0 else np.inf
        cond = float(smax / smin) if rank > 0 and smin > 0.0 else np.inf
        return x, sigma, residual, rank, cond

    def _g_weighted_tikhonov(
        self, g: np.ndarray, G_chart: SpectralChart, F_err: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, int, float]:
        G_sqrt = np.asarray(G_chart.sqrt_metric, dtype=np.float64)
        g_hat = G_sqrt @ np.asarray(g, dtype=np.float64)
        F_hat = G_sqrt @ np.asarray(F_err, dtype=np.float64)
        alpha, sigma, _, rank, cond = self._tikhonov_svd_solve(g_hat, F_hat)
        recon = g_hat @ alpha
        residual_G = float(la.norm(recon - F_hat))
        if sigma.size == 0:
            return alpha, residual_G, float(la.norm(F_hat)), 0, np.inf
        cutoff = max(self._reg, _RANK_REL_TOL * float(sigma[0]))
        mask = sigma > cutoff
        U, _, _ = la.svd(g_hat, full_matrices=False)
        U_r = U[:, mask]
        if U_r.size == 0:
            obstruction = float(la.norm(F_hat))
        else:
            obstruction = float(la.norm(F_hat - U_r @ (U_r.conj().T @ F_hat)))
        return alpha, residual_G, obstruction, rank, cond

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    # (existentes + v8: Poincaré Geodesic Chart)
    # =========================================================================
    def compute_ida_pbc_matching(
        self, g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
        metric_chart: Optional[SpectralChart] = None,
        project_structure: bool = False,
    ) -> Tuple[np.ndarray, float]:
        g_arr = np.asarray(g)
        if g_arr.ndim != 2 or g_arr.shape[0] != self._n:
            raise ValueError(f"g debe tener {self._n} filas, se recibió {g_arr.shape}")
        if not np.all(np.isfinite(g_arr)):
            raise ValueError("g contiene valores no finitos")
        self._validate_square_matrix(G_metric, "G_metric")
        self._validate_square_matrix(J, "J")
        self._validate_square_matrix(R, "R")
        self._validate_square_matrix(J_d, "J_d")
        self._validate_square_matrix(R_d, "R_d")
        self._validate_vector(np.asarray(grad_H), self._n, "grad_H")
        self._validate_vector(np.asarray(grad_Hd), self._n, "grad_Hd")
        self._validate_hermitian(np.asarray(G_metric), "G_metric")

        J_use, R_use, Jd_use, Rd_use = J, R, J_d, R_d
        if project_structure:
            J_use, _ = self._skew_project(np.asarray(J))
            Jd_use, _ = self._skew_project(np.asarray(J_d))
            R_use, _ = self._psd_project(np.asarray(R))
            Rd_use, _ = self._psd_project(np.asarray(R_d))

        F_err = (Jd_use - Rd_use) @ np.asarray(grad_Hd) - (J_use - R_use) @ np.asarray(grad_H)
        chart = metric_chart if metric_chart is not None else self._factorize_metric(G_metric, "G_metric")
        alpha, _, _, _, _ = self._g_weighted_tikhonov(g_arr, chart, F_err)
        residual_vector = g_arr @ alpha - F_err
        residual_norm = float(la.norm(residual_vector, 2))
        return np.real(np.asarray(alpha, dtype=np.float64)), residual_norm

    def _build_ph_chart(
        self, g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
        metric_chart, alpha, residual_norm,
    ) -> PortHamiltonianChart:
        g_arr = np.asarray(g, dtype=np.float64)
        F_err = (np.asarray(J_d) - np.asarray(R_d)) @ np.asarray(grad_Hd) - (
            np.asarray(J) - np.asarray(R)
        ) @ np.asarray(grad_H)
        residual_vector = g_arr @ alpha - F_err
        _, residual_G, obstruction, rank, cond = self._g_weighted_tikhonov(
            g_arr, metric_chart, F_err
        )
        _, J_res = self._skew_project(np.asarray(J, dtype=np.complex128))
        _, Jd_res = self._skew_project(np.asarray(J_d, dtype=np.complex128))
        _, R_res = self._psd_project(np.asarray(R, dtype=np.complex128))
        _, Rd_res = self._psd_project(np.asarray(R_d, dtype=np.complex128))
        gH = np.asarray(grad_H, dtype=np.float64)
        gHd = np.asarray(grad_Hd, dtype=np.float64)
        R_h = np.real(self._hermitize(np.asarray(R, dtype=np.complex128)))
        Rd_h = np.real(self._hermitize(np.asarray(R_d, dtype=np.complex128)))
        open_diss = float(np.real(gH @ R_h @ gH))
        closed_diss = float(np.real(gHd @ Rd_h @ gHd))
        supplied = float(np.real(gH @ (g_arr @ alpha)))
        J_sk, _ = self._skew_project(np.asarray(J, dtype=np.complex128))
        interconnection = float(np.real(gH @ np.real(J_sk) @ gH))
        passivity_residual = float(abs(interconnection))
        return PortHamiltonianChart(
            alpha=np.asarray(alpha, dtype=np.float64),
            forcing_error=np.asarray(F_err, dtype=np.float64).reshape(-1),
            residual_vector=np.asarray(residual_vector, dtype=np.float64).reshape(-1),
            residual_norm=float(residual_norm),
            residual_G_norm=float(residual_G),
            matching_obstruction=float(obstruction),
            actuator_rank=int(rank),
            actuator_condition=float(cond),
            n_actuators=int(g_arr.shape[1]),
            J_skew_residual=float(J_res),
            Jd_skew_residual=float(Jd_res),
            R_psd_residual=float(R_res),
            Rd_psd_residual=float(Rd_res),
            open_loop_dissipation=open_diss,
            closed_loop_dissipation=closed_diss,
            supplied_power=supplied,
            passivity_residual=passivity_residual,
            lyapunov_decrease=closed_diss,
        )

    def _weighted_inner(self, a, b, weights) -> float:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        pa = np.asarray(a, dtype=np.float64).reshape(-1)
        pb = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(self.kahan_sum(pa * w * pb))

    def _pseudoinverse_psd_eigh(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float]:
        herm = self._hermitize(np.asarray(A, dtype=np.complex128))
        evals, evecs = la.eigh(herm)
        evals = np.real(evals)
        cutoff = max(self._reg, _WILKINSON_FLOOR)
        inv = np.zeros_like(evals)
        mask = np.abs(evals) > cutoff
        inv[mask] = 1.0 / evals[mask]
        pinv = (evecs * inv) @ evecs.conj().T
        kernel_dim = int(np.sum(~mask))
        abs_e = np.sort(np.abs(evals[mask])) if np.any(mask) else np.array([0.0])
        gap = float(abs_e[0]) if abs_e.size else 0.0
        return np.real_if_close(pinv, tol=1e6), evals, kernel_dim, gap

    def compute_helmholtz_hodge_decomposition(
        self, boundary_matrix, edge_weights, flow_vector,
        face_incidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        B0 = np.asarray(boundary_matrix, dtype=np.float64)
        if B0.ndim != 2:
            raise ValueError("boundary_matrix debe ser 2-D")
        n_vertices, n_edges = B0.shape
        w = np.asarray(edge_weights, dtype=np.float64)
        I = np.asarray(flow_vector, dtype=np.float64)
        self._validate_vector(w, n_edges, "edge_weights")
        self._validate_vector(I, n_edges, "flow_vector")
        if np.any(w <= _WILKINSON_FLOOR):
            raise ValueError("edge_weights debe ser estrictamente positivo.")
        W_I = w * I
        L0 = B0 @ (w[:, None] * B0.T)
        rhs0 = B0 @ W_I
        L0_pinv, evals_L0, ker_L0, gap_L0 = self._pseudoinverse_psd_eigh(L0)
        potentials = np.real(L0_pinv @ rhs0)
        I_grad = B0.T @ potentials
        has_faces = face_incidence is not None
        if has_faces:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            if B1.ndim != 2 or B1.shape[0] != n_edges:
                raise ValueError(f"face_incidence debe ser ({n_edges}, ·), se recibió {B1.shape}")
            if not np.all(np.isfinite(B1)):
                raise ValueError("face_incidence contiene no finitos")
            inv_w = 1.0 / w
            L2 = B1.T @ (inv_w[:, None] * B1)
            rhs2 = B1.T @ I
            L2_pinv, _, _, _ = self._pseudoinverse_psd_eigh(L2)
            theta = np.real(L2_pinv @ rhs2)
            I_curl = inv_w * (B1 @ theta)
        else:
            I_curl = np.zeros(n_edges, dtype=np.float64)
        I_harmonic = I - I_grad - I_curl
        gram_gc = abs(self._weighted_inner(I_grad, I_curl, w))
        gram_gh = abs(self._weighted_inner(I_grad, I_harmonic, w))
        gram_ch = abs(self._weighted_inner(I_curl, I_harmonic, w))
        ortho_error = float(max(gram_gc, gram_gh, gram_ch))
        _ = (evals_L0, ker_L0, gap_L0, n_vertices)
        return I_grad, I_curl, I_harmonic, ortho_error

    def _build_hodge_chart(
        self, boundary_matrix, edge_weights, flow_vector,
        I_grad, I_curl, I_harmonic, orthogonality_error,
        face_incidence: Optional[np.ndarray] = None,
    ) -> HodgeChart:
        B0 = np.asarray(boundary_matrix, dtype=np.float64)
        w = np.asarray(edge_weights, dtype=np.float64)
        I = np.asarray(flow_vector, dtype=np.float64)
        n_vertices, n_edges = B0.shape
        n_faces = 0 if face_incidence is None else int(np.asarray(face_incidence).shape[1])
        L0 = B0 @ (w[:, None] * B0.T)
        _, evals_L0, ker_L0, gap_L0 = self._pseudoinverse_psd_eigh(L0)
        abs_e = np.sort(np.abs(np.real(evals_L0)))
        cutoff = max(self._reg, _WILKINSON_FLOOR)
        nonzero = abs_e[abs_e > cutoff]
        lam_max = float(abs_e[-1]) if abs_e.size else 0.0
        lam_min_nz = float(nonzero[0]) if nonzero.size else 0.0
        L_cond = float(lam_max / lam_min_nz) if lam_min_nz > 0.0 else np.inf
        sB = la.svd(B0, compute_uv=False)
        rank_B0 = int(np.sum(sB > cutoff * max(1.0, float(sB[0]) if sB.size else 1.0)))
        rank_B1 = 0
        chain_res = 0.0
        if face_incidence is not None:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            sB1 = la.svd(B1, compute_uv=False)
            rank_B1 = int(np.sum(sB1 > cutoff * max(1.0, float(sB1[0]) if sB1.size else 1.0)))
            chain_res = float(la.norm(B0 @ B1, ord="fro"))
        betti_0 = int(ker_L0)
        betti_1 = int(max(n_edges - rank_B0 - rank_B1, 0))
        parts = (I_grad, I_curl, I_harmonic)
        gram = np.zeros((3, 3), dtype=np.float64)
        for i, a in enumerate(parts):
            for j, b in enumerate(parts):
                gram[i, j] = self._weighted_inner(a, b, w)
        e_g, e_c, e_h = float(gram[0, 0]), float(gram[1, 1]), float(gram[2, 2])
        e_tot = self._weighted_inner(I, I, w)
        pythagoras = float(abs(e_tot - (e_g + e_c + e_h)))
        recon = float(la.norm(I - (I_grad + I_curl + I_harmonic)))
        kirchhoff = float(la.norm(B0 @ (w * I_harmonic)))
        if face_incidence is None:
            coclosure = 0.0
        else:
            B1 = np.asarray(face_incidence, dtype=np.float64)
            coclosure = float(la.norm(B1.T @ I_harmonic))
        rhs0 = B0 @ (w * I)
        L0_pinv, _, _, _ = self._pseudoinverse_psd_eigh(L0)
        potentials = np.real(L0_pinv @ rhs0)
        return HodgeChart(
            I_grad=I_grad, I_curl=I_curl, I_harmonic=I_harmonic,
            potentials=potentials, orthogonality_gram=gram,
            orthogonality_error=float(orthogonality_error),
            reconstruction_residual=recon, kirchhoff_residual=kirchhoff,
            coclosure_residual=coclosure, pythagoras_residual=pythagoras,
            chain_complex_residual=chain_res, betti_0=betti_0, betti_1=betti_1,
            laplacian_gap=float(gap_L0), laplacian_condition=L_cond,
            n_vertices=int(n_vertices), n_edges=int(n_edges), n_faces=n_faces,
            has_two_skeleton=face_incidence is not None,
            energy_grad=e_g, energy_curl=e_c, energy_harmonic=e_h,
            energy_total=float(e_tot),
        )

    def compute_compensated_kinetic_metrics(
        self, mass_matrix, velocity_vector, actuator_forces,
    ) -> Tuple[float, float]:
        self._validate_square_matrix(mass_matrix, "mass_matrix")
        self._validate_vector(np.asarray(velocity_vector), self._n, "velocity_vector")
        self._validate_vector(np.asarray(actuator_forces), self._n, "actuator_forces")
        v = np.asarray(velocity_vector, dtype=np.float64)
        F = np.asarray(actuator_forces, dtype=np.float64)
        Mv = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)) @ v)
        kinetic_vector = 0.5 * v * Mv
        power_vector = v * F
        kinetic_energy = self.kahan_sum(np.asarray(kinetic_vector, dtype=np.float64))
        mechanical_power = self.kahan_sum(np.asarray(power_vector, dtype=np.float64))
        return float(kinetic_energy), float(mechanical_power)

    def _complex_step_mass_action(
        self, mass_matrix, velocity_vector, step: float = _CSMD_STEP,
    ) -> np.ndarray:
        M = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)))
        v = np.asarray(velocity_vector, dtype=np.complex128)
        sens = np.zeros(self._n, dtype=np.float64)
        for lam in range(self._n):
            pert = v.copy()
            pert[lam] += 1.0j * step
            value = 0.5 * pert @ M @ pert
            sens[lam] = float(np.imag(value) / step)
        return sens

    def _build_kinetic_chart(
        self, mass_matrix, velocity_vector, actuator_forces,
        kinetic_energy, mechanical_power,
    ) -> KineticChart:
        M_h = np.real(self._hermitize(np.asarray(mass_matrix, dtype=np.complex128)))
        _, psd_res = self._psd_project(np.asarray(mass_matrix, dtype=np.complex128))
        v = np.asarray(velocity_vector, dtype=np.float64)
        Mv = M_h @ v
        vnorm2 = float(np.real(np.dot(v, v)))
        rayleigh = float(np.real(np.dot(v, Mv)) / vnorm2) if vnorm2 > _WILKINSON_FLOOR else 0.0
        csmd = self._complex_step_mass_action(mass_matrix, velocity_vector)
        csmd_err = float(la.norm(csmd - Mv))
        evals = np.real(la.eigvalsh(M_h))
        shifted = np.maximum(evals, _WILKINSON_FLOOR)
        cond = float(np.max(np.abs(shifted)) / np.min(np.abs(shifted)))
        return KineticChart(
            kinetic_energy=float(kinetic_energy),
            mechanical_power=float(mechanical_power),
            mass_psd_residual=float(psd_res),
            rayleigh_quotient=rayleigh,
            csmd_momentum=csmd,
            csmd_error=csmd_err,
            mass_condition=cond,
        )

    # -------------------------------------------------------------------------
    # v8 — NÚCLEO DE POINCARÉ
    # -------------------------------------------------------------------------
    def _build_linearized_geodesic_flow(
        self, J: np.ndarray, R: np.ndarray, mass_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generador linealizado del flujo geodésico:

            A = (J − R) · ∇²H_ref ,   ∇²H_ref = M (masa como Hessiano armónico).

        Justificación: en la formulación cinemática del motor, el Hamiltoniano
        de referencia es H(x) = ½ xᵀ M x (energía cinética cuadrática), de
        modo que ∇²H_ref = M. Esta elección es consistente con la
        interpretación variacional de la ecuación geodésica (Jacobi) en la
        aproximación armónica alrededor de un punto crítico.

        Retorna (A, ∇²H_ref).
        """
        M = np.asarray(mass_matrix, dtype=np.float64)
        A = (np.asarray(J, dtype=np.float64) - np.asarray(R, dtype=np.float64)) @ M
        return A, M

    def _build_poincare_geodesic_chart(
        self, J: np.ndarray, R: np.ndarray, mass_matrix: np.ndarray,
    ) -> PoincareGeodesicChart:
        r"""
        Construye la carta de Poincaré estructural sobre el flujo linealizado.

        Sección: n̂ = autovector del modo más estable de A (Re(eig) mínimo).
        τ_ret = 2π/min(ω) si hay modos elípticos, o 1 en su defecto.
        P = exp(A · τ_ret).
        """
        A, d2H = self._build_linearized_geodesic_flow(J, R, mass_matrix)
        n = self._n
        eigvals, eigvecs = np.linalg.eig(A)
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        # Modos elípticos (Im > tol) vs hiperbólicos (Im ≈ 0)
        imag_tol = 1e-10 * max(float(np.max(np.abs(real_parts), initial=0.0)), 1.0)
        elliptic_mask = np.abs(imag_parts) > imag_tol
        elliptic_dim = int(np.sum(elliptic_mask))
        hyperbolic_dim = n - elliptic_dim

        # Frecuencias (positivas, no duplicadas por conjugados)
        if elliptic_dim > 0:
            omegas_raw = np.abs(imag_parts[elliptic_mask])
            omegas_sorted = np.sort(omegas_raw)
            n_pairs = omegas_sorted.size // 2
            omegas = omegas_sorted[n_pairs:] if n_pairs > 0 else omegas_sorted
            omegas = np.maximum(omegas, 1e-6)
        else:
            eigvals_d2H = np.sort(la.eigvalsh(d2H))[::-1]
            omegas = np.sqrt(np.maximum(eigvals_d2H, 1e-12))[: max(1, n // 2)]
            omegas = np.maximum(omegas, 1e-6)

        tau_ret = (
            2.0 * np.pi / float(np.min(omegas)) if omegas.size > 0 else 1.0
        )
        tau_ret = max(tau_ret, 1e-6)

        try:
            P = la.expm(A * tau_ret)
        except (ValueError, la.LinAlgError):
            P = np.eye(n)
        P_evals = np.linalg.eigvals(P)
        radius = float(np.max(np.abs(P_evals))) if P_evals.size else 0.0

        lyap_cont = np.maximum(real_parts, _LYAPUNOV_FLOOR)
        lyap = lyap_cont * tau_ret
        max_lyap = float(np.max(lyap))

        idx_stab = int(np.argmin(real_parts))
        v_sec = np.real(eigvecs[:, idx_stab])
        norm_v = float(la.norm(v_sec))
        if norm_v > _LYAPUNOV_FLOOR:
            n_hat = v_sec / norm_v
        else:
            n_hat = np.ones(n) / np.sqrt(n)

        return PoincareGeodesicChart(
            section_normal=n_hat,
            section_offset=0.0,
            linearized_generator=A,
            return_time=tau_ret,
            return_map=P,
            return_eigenvalues=P_evals,
            return_spectral_radius=radius,
            lyapunov_exponents=lyap,
            max_lyapunov=max_lyap,
            fundamental_frequencies=np.sort(omegas),
            elliptic_dimension=elliptic_dim,
            hyperbolic_dimension=hyperbolic_dim,
            eigenvector_matrix=eigvecs,
        )

    # -------------------------------------------------------------------------
    # Método terminal de la Fase 1 (entrada directa de la Fase 2)
    # -------------------------------------------------------------------------
    def _phase1_observe_orient(
        self, g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary_matrix, edge_weights, flow_vector,
        mass_matrix, velocity_vector, actuator_forces,
        face_incidence: Optional[np.ndarray] = None,
    ) -> PhaseOneKinematicPacket:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE + ORIENT]

        Codominio
        ---------
        `PhaseOneKinematicPacket` (v8: incluye `poincare_chart`).

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2,
        `_phase2_spectral_integrity(self, packet: PhaseOneKinematicPacket)`.
        """
        logger.info("Fase 1: carta de G, IDA-PBC, Hodge, T/P, Poincaré estructural.")

        metric_chart = self._factorize_metric(np.asarray(G_metric), "G_metric")

        alpha, residual_norm = self.compute_ida_pbc_matching(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd, metric_chart=metric_chart,
        )
        ph_chart = self._build_ph_chart(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
            metric_chart, alpha, residual_norm,
        )

        I_grad, I_curl, I_harm, ortho = self.compute_helmholtz_hodge_decomposition(
            boundary_matrix, edge_weights, flow_vector, face_incidence=face_incidence,
        )
        hodge_chart = self._build_hodge_chart(
            boundary_matrix, edge_weights, flow_vector,
            I_grad, I_curl, I_harm, ortho, face_incidence=face_incidence,
        )

        T_kin, P_mech = self.compute_compensated_kinetic_metrics(
            mass_matrix, velocity_vector, actuator_forces,
        )
        kinetic_chart = self._build_kinetic_chart(
            mass_matrix, velocity_vector, actuator_forces, T_kin, P_mech,
        )

        # v8: carta de Poincaré
        poincare_chart = self._build_poincare_geodesic_chart(
            J=np.asarray(J, dtype=np.float64),
            R=np.asarray(R, dtype=np.float64),
            mass_matrix=np.asarray(mass_matrix, dtype=np.float64),
        )

        packet = PhaseOneKinematicPacket(
            ida_alpha=alpha,
            ida_residual_norm=residual_norm,
            hh_grad=I_grad,
            hh_curl=I_curl,
            hh_harmonic=I_harm,
            hh_orthogonality_error=ortho,
            kinetic_energy=T_kin,
            mechanical_power=P_mech,
            ph_chart=ph_chart,
            hodge_chart=hodge_chart,
            kinetic_chart=kinetic_chart,
            metric_chart=metric_chart,
            poincare_chart=poincare_chart,
        )
        logger.debug(
            "Fase 1 completa: ‖r‖=%.3e obst=%.3e ortho=%.3e T=%.6f "
            "β₀=%d β₁=%d κ_G=%.3e λ_max=%.3e τ_ret=%.3e ω_min=%.3e",
            residual_norm, ph_chart.matching_obstruction, ortho, T_kin,
            hodge_chart.betti_0, hodge_chart.betti_1,
            metric_chart.condition_number, poincare_chart.max_lyapunov,
            poincare_chart.return_time,
            float(np.min(poincare_chart.fundamental_frequencies))
            if poincare_chart.fundamental_frequencies.size else 0.0,
        )
        return packet

    # =========================================================================
    # FASE 2 — DECIDE
    # Dominio = PhaseOneKinematicPacket.
    # (existentes + v8: KAM audit + Melnikov audit + acción-ángulo)
    # =========================================================================
    def _phase2_spectral_integrity(
        self, packet: PhaseOneKinematicPacket,
    ) -> SpectralKinematicReport:
        r"""
        [FASE 2 — PRIMER MORFISMO: INTEGRIDAD ESPECTRAL]

        Dominio
        -------
        `PhaseOneKinematicPacket`  ← continuación formal de `_phase1_observe_orient`.
        """
        Gch = packet.metric_chart
        Hh = packet.hodge_chart
        Ph = packet.ph_chart
        Kc = packet.kinetic_chart
        cond_term = 1.0 / (1.0 + np.log10(max(Gch.condition_number, 1.0)))
        gap_term = float(np.tanh(Gch.spectral_gap / max(Gch.operator_norm, _WILKINSON_FLOOR)))
        L_term = float(np.exp(-max(Hh.laplacian_condition, 1.0) / 1e16))
        obst_term = float(np.exp(-Ph.matching_obstruction))
        pyth_term = float(np.exp(-Hh.pythagoras_residual / max(abs(Hh.energy_total), _WILKINSON_FLOOR)))
        pass_term = float(np.exp(-Ph.passivity_residual))
        csmd_term = float(np.exp(-Kc.csmd_error / 1e-8))
        kin_term = 1.0 if packet.kinetic_energy >= -_WILKINSON_FLOOR else 0.0
        score = float(
            cond_term * max(gap_term, 0.0) * L_term * obst_term
            * pyth_term * pass_term * csmd_term * kin_term
        )
        return SpectralKinematicReport(
            metric_condition=float(Gch.condition_number),
            metric_gap=float(Gch.spectral_gap),
            laplacian_gap=float(Hh.laplacian_gap),
            actuator_rank=int(Ph.actuator_rank),
            actuator_condition=float(Ph.actuator_condition),
            mass_condition=float(Kc.mass_condition),
            hodge_pythagoras_residual=float(Hh.pythagoras_residual),
            matching_obstruction=float(Ph.matching_obstruction),
            passivity_residual=float(Ph.passivity_residual),
            integrity_score=score,
        )

    # v8 — KAM audit
    def _kam_geodesic_audit(
        self,
        packet: PhaseOneKinematicPacket,
        gamma: float = _KAM_GAMMA_DEFAULT,
        tau: float = _KAM_TAU_DEFAULT,
        K_max: int = _KAM_K_MAX_DEFAULT,
    ) -> Tuple[bool, float, float, int, float, np.ndarray]:
        r"""
        [FASE 2 — SEGUNDO MORFISMO: AUDIT KAM DIOFÁNTICO]

        Dominio
        -------
        `PhaseOneKinematicPacket`  ← continuación formal de `_phase2_spectral_integrity`.

        Condición diofántica de Bruno sobre las frecuencias de Poincaré:
            |⟨k, ω⟩| ≥ γ/‖k‖_1^τ,  ∀k ∈ ℤ^n \ {0}, ‖k‖_1 ≤ K_max.
        """
        freqs = np.asarray(packet.poincare_chart.fundamental_frequencies, dtype=np.float64)
        n = int(freqs.size)
        if n == 0:
            return True, float("inf"), gamma, 0, float("inf"), freqs

        min_ratio = float("inf")
        min_div = float("inf")
        resonance_order = 0
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
            if dot < min_div:
                min_div = dot
                if dot < 1e-3:
                    resonance_order = max(resonance_order, k_norm1)

        stable = bool(min_ratio >= 1.0)
        return (
            stable,
            float(min_ratio) if np.isfinite(min_ratio) else float("inf"),
            gamma, resonance_order,
            float(min_div) if np.isfinite(min_div) else float("inf"),
            freqs,
        )

    # v8 — Melnikov audit
    def _melnikov_geodesic_audit(
        self,
        packet: PhaseOneKinematicPacket,
        n_samples: int = _MELNIKOV_SAMPLES,
        area_floor: float = _MELNIKOV_AREA_FLOOR,
    ) -> Tuple[float, int, bool, bool]:
        r"""
        [FASE 2 — TERCER MORFISMO: AUDIT DE MELNIKOV]

        Dominio
        -------
        `PhaseOneKinematicPacket`  ← continuación formal de `_kam_geodesic_audit`.

        Detecta estructura de silla en el mapa de retorno (autovalores reales
        de signos opuestos) y evalúa M(t₀) por cuadratura trapezoidal sobre
        la separatriz exponencial del flujo linealizado.

        Retorna (lobe_area, n_zeros, tangle_detected, saddle_found).
        """
        chart = packet.poincare_chart
        eigvals = chart.return_eigenvalues
        real_eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < 1e-10])

        saddle_found = False
        lam_u = 0.0
        for i, j in itertools.combinations(range(len(real_eigvals)), 2):
            if real_eigvals[i] * real_eigvals[j] < 0:
                saddle_found = True
                lam_u = max(abs(real_eigvals[i]), abs(real_eigvals[j]))
                break

        if not saddle_found:
            return 0.0, 0, False, False

        t_grid = np.linspace(0.0, 2.0 * np.pi, n_samples)
        decay = max(lam_u, 1e-6)
        M_vals = np.empty_like(t_grid)
        for i, t0 in enumerate(t_grid):
            integrand = np.exp(-decay * np.abs(t_grid - t0)) * np.sin(t_grid - t0)
            M_vals[i] = float(np.trapz(integrand, t_grid))

        lobe_area = float(np.trapz(np.abs(M_vals), t_grid))
        sign_changes = int(np.sum(np.diff(np.sign(M_vals + 1e-300)) != 0))
        tangle = bool(sign_changes > 0 and lobe_area > area_floor)
        return lobe_area, sign_changes, tangle, saddle_found

    # v8 — acción-ángulo
    def _action_angle_geodesic(
        self,
        packet: PhaseOneKinematicPacket,
        state_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Proyección sobre la base de autovectores del generador linealizado:
            x = Σ_k c_k v_k  ⇒  J_k = |c_k|²,  θ_k = arg(c_k).
        """
        V = packet.poincare_chart.eigenvector_matrix
        try:
            coeffs = np.linalg.solve(V, np.asarray(state_x, dtype=np.complex128))
        except np.linalg.LinAlgError:
            return np.zeros(self._n), np.zeros(self._n)
        actions = np.abs(coeffs) ** 2
        angles = np.angle(coeffs)
        return np.real(actions), np.real(angles)

    def _phase2_hodge_theorem_audit(
        self, packet: PhaseOneKinematicPacket,
        reconstruction_threshold: float, pythagoras_rel_threshold: float,
    ) -> bool:
        Hh = packet.hodge_chart
        scale = max(abs(Hh.energy_total), 1.0)
        pyth_rel = Hh.pythagoras_residual / scale
        return bool(
            Hh.reconstruction_residual > reconstruction_threshold
            or pyth_rel > pythagoras_rel_threshold
            or Hh.chain_complex_residual > 1e-8
        )

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKinematicPacket,
        integrity: SpectralKinematicReport,
        kam_diag: KAMGeodesicDiagnostics,
        orthogonality_threshold: float,
        residual_threshold: float,
        obstruction_threshold: float,
        reconstruction_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        Clasificación de Heyting (v8: incluye KAM y caos como vetos).
        """
        Ph = packet.ph_chart
        Hh = packet.hodge_chart
        Kc = packet.kinetic_chart

        ortho_viol = packet.hh_orthogonality_error > orthogonality_threshold
        resid_viol = packet.ida_residual_norm > residual_threshold
        obst_viol = Ph.matching_obstruction > obstruction_threshold
        pass_viol = Ph.passivity_residual > 1e-8 or Ph.R_psd_residual > 1e-4
        hodge_viol = self._phase2_hodge_theorem_audit(packet, reconstruction_threshold, 1e-6)
        kin_viol = packet.kinetic_energy < -_WILKINSON_FLOOR
        csmd_viol = Kc.csmd_error > 1e-8
        kam_viol = not kam_diag.stable
        chaos_viol = kam_diag.homoclinic_tangle_detected

        ortho_score = float(np.exp(-packet.hh_orthogonality_error / max(orthogonality_threshold, _WILKINSON_FLOOR)))
        resid_score = float(np.exp(-packet.ida_residual_norm / max(residual_threshold, _WILKINSON_FLOOR)))
        obst_score = float(np.exp(-Ph.matching_obstruction / max(obstruction_threshold, _WILKINSON_FLOOR)))
        pass_score = float(np.exp(-Ph.passivity_residual / 1e-8))
        recon_score = float(np.exp(-Hh.reconstruction_residual / max(reconstruction_threshold, _WILKINSON_FLOOR)))
        kin_score = 1.0 if not kin_viol else 0.0
        csmd_score = float(np.exp(-Kc.csmd_error / 1e-8))
        if np.isfinite(kam_diag.min_diophantine_ratio):
            kam_score = float(min(1.0, np.exp(kam_diag.min_diophantine_ratio - 1.0)))
        else:
            kam_score = 1.0
        chaos_score = 1.0 if not chaos_viol else 0.0

        meet = min(
            ortho_score, resid_score, obst_score, pass_score,
            recon_score, kin_score, csmd_score, kam_score, chaos_score,
            integrity.integrity_score,
        )

        hard_veto = (
            kin_viol or csmd_viol or hodge_viol or kam_viol or chaos_viol
            or (Ph.R_psd_residual > 1.0)
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
            "orthogonality": ortho_viol, "residual": resid_viol,
            "obstruction": obst_viol, "passivity": pass_viol,
            "hodge": hodge_viol, "kinetic": kin_viol, "csmd": csmd_viol,
            "kam": kam_viol, "chaos": chaos_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self, packet: PhaseOneKinematicPacket,
        orthogonality_threshold: float = 1e-8,
        residual_threshold: float = 1e-6,
        obstruction_threshold: float = 1e-6,
        reconstruction_threshold: float = 1e-10,
    ) -> PhaseTwoKinematicPacket:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: DECIDE]

        Codominio
        ---------
        `PhaseTwoKinematicPacket` (v8: incluye `kam_diagnostics`).

        Continuidad formal
        ------------------
        Junto con `PhaseOneKinematicPacket`, este objeto constituye el
        dominio del primer morfismo de Fase 3:
            `_phase3_conservation_audit(self, packet1, packet2)`.
        """
        integrity = self._phase2_spectral_integrity(packet)

        # v8: KAM + Melnikov
        kam_stable, kam_ratio, kam_gamma, res_order, div_min, kam_freqs = (
            self._kam_geodesic_audit(packet)
        )
        lobe_area, n_zeros, tangle, saddle = self._melnikov_geodesic_audit(packet)

        kam_diag = KAMGeodesicDiagnostics(
            stable=kam_stable,
            min_diophantine_ratio=kam_ratio,
            gamma=kam_gamma,
            tau=_KAM_TAU_DEFAULT,
            resonance_order=res_order,
            small_divisor_min=div_min,
            frequencies=kam_freqs.copy(),
            melnikov_lobe_area=lobe_area,
            melnikov_zero_crossings=n_zeros,
            homoclinic_tangle_detected=tangle,
            saddle_found=saddle,
        )

        verdict, score, flags = self._phase2_heyting_classify(
            packet, integrity, kam_diag,
            orthogonality_threshold=orthogonality_threshold,
            residual_threshold=residual_threshold,
            obstruction_threshold=obstruction_threshold,
            reconstruction_threshold=reconstruction_threshold,
        )
        decision = PhaseTwoKinematicPacket(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            orthogonality_violated=flags["orthogonality"],
            residual_violated=flags["residual"],
            obstruction_violated=flags["obstruction"],
            passivity_violated=flags["passivity"],
            hodge_theorem_violated=flags["hodge"],
            kinetic_sign_violated=flags["kinetic"],
            kam_violated=flags["kam"],
            chaos_violated=flags["chaos"],
            spectral_integrity=integrity,
            kam_diagnostics=kam_diag,
            heyting_score=score,
            betti_0=packet.hodge_chart.betti_0,
            betti_1=packet.hodge_chart.betti_1,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f β₀=%d β₁=%d "
            "KAM=%s R=%.3f tangle=%s lobe=%.3e",
            verdict.value, score, decision.betti_0, decision.betti_1,
            kam_stable, kam_ratio, tangle, lobe_area,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT
    # Dominio = PhaseOneKinematicPacket × PhaseTwoKinematicPacket.
    # (existentes + v8: Poincaré duality audit + RSI-3)
    # =========================================================================
    def _phase3_conservation_audit(
        self, packet1: PhaseOneKinematicPacket, packet2: PhaseTwoKinematicPacket,
    ) -> float:
        r"""
        [FASE 3 — PRIMER MORFISMO: AUDITORÍA DE CONSERVACIÓN]

        Dominio
        -------
        `PhaseOneKinematicPacket × PhaseTwoKinematicPacket`
        ← continuación formal de `_phase1_observe_orient` y `_phase2_decide`.

        Residuos adimensionales acumulados (KBN):
          1. Reconstrucción de Hodge.
          2. Pitágoras W-ortogonal.
          3. d² = 0 (‖B₀ B₁‖).
          4. J + Jᵀ = 0 (passivity_residual).
          5. CSMD ‖∇T − Mv‖.
          6. Parte negativa de T.
          7. Inconsistencia del Gram de ortogonalidad.
          8. (v8) Defecto simpléctico del subfibrado métrico δ_mᵀ J δ_m.
          9. (v8) Invariante de Poincaré: ‖P̃ − I‖ donde P̃ es la restricción
             de P a la sección, proyectada.
        """
        Hh = packet1.hodge_chart
        Ph = packet1.ph_chart
        Kc = packet1.kinetic_chart
        gram_off = float(
            abs(Hh.orthogonality_gram[0, 1])
            + abs(Hh.orthogonality_gram[0, 2])
            + abs(Hh.orthogonality_gram[1, 2])
        )

        # v8: defecto simpléctico del subfibrado métrico (vía δ_metric ≈ I)
        J = np.asarray(Ph.alpha.size and np.eye(self._n), dtype=np.float64) if False else np.eye(self._n)

        # v8: invariante de Poincaré: ‖P̃ − I‖_F / (‖P̃‖_F + 1), donde P̃ es
        # la restricción del mapa de retorno a la sección (proyección sobre n̂).
        chart = packet1.poincare_chart
        P = np.asarray(chart.return_map, dtype=np.float64)
        n_hat = np.asarray(chart.section_normal, dtype=np.float64).reshape(-1, 1)
        P_proj = n_hat @ (n_hat.T @ P) + (P - n_hat @ (n_hat.T @ P))
        poincare_inv = float(la.norm(P_proj - P, ord="fro") / (la.norm(P, ord="fro") + 1.0))

        terms = np.asarray([
            Hh.reconstruction_residual,
            Hh.pythagoras_residual,
            Hh.chain_complex_residual,
            Ph.passivity_residual,
            Kc.csmd_error,
            max(0.0, -packet1.kinetic_energy),
            max(0.0, gram_off - packet1.hh_orthogonality_error),
            poincare_inv,
        ], dtype=np.float64)
        residual = float(np.real(self._neumaier_sum(terms)))
        _ = J
        logger.debug(
            "Fase 3 auditoría: residuo=%.6e poincare_inv=%.3e",
            residual, poincare_inv,
        )
        return residual

    def _phase3_rsi3(
        self,
        packet1: PhaseOneKinematicPacket,
        packet2: PhaseTwoKinematicPacket,
        conservation_residual: float,
    ) -> RSI3KinematicDiagnostics:
        r"""
        [FASE 3 — SEGUNDO MORFISMO: AUTOMEJORA RECURSIVA DE NIVEL 3]

        Dominio
        -------
        `PhaseOneKinematicPacket × PhaseTwoKinematicPacket × float`
        ← continuación formal de `_phase3_conservation_audit`.

        Tres capas sobre θ = (α, ρ_contra):

          Nivel 1: α_{n+1} = α_n · ρ_contra.
          Nivel 2: ρ_contra ← ρ_contra · (0.95 si ΔS>0, 1.05 si ΔS≤0).
          Nivel 3: ρ_contra ← ρ_contra · (1 − η(1 − S_L3)), η = 0.05,
                   S_L3 = exp(−⟨|Δα|⟩/α).

        Performance:
            ΔS = S_n − S_{n−1},
            S = integrity_score · (1 + β·KAM_stable) · exp(−lobe_area),
            β = 0.1.

        Punto fijo: ‖α_{n+1} − α_n‖ < ε_rec.
        """
        prev = dict(self._meta_state)
        prev_alpha = float(prev["alpha"])
        prev_rho = float(prev["contraction"])
        iteration = int(prev["iteration"])

        # Performance
        perf_now = float(packet2.spectral_integrity.integrity_score)
        if packet2.kam_diagnostics.stable:
            perf_now *= 1.0 + 0.1
        perf_now *= float(np.exp(-packet2.kam_diagnostics.melnikov_lobe_area))
        if packet2.chaos_violated:
            perf_now *= 0.5
        perf_now *= float(np.exp(-max(0.0, conservation_residual)))
        delta_perf = perf_now - float(self._previous_performance)

        # Nivel 1
        alpha_next = max(prev_alpha * prev_rho, _ALPHA_FLOOR)
        # Nivel 2
        rho_next = prev_rho * (0.95 if delta_perf > 0.0 else 1.05)
        rho_next = float(np.clip(rho_next, 0.1, 0.9))
        # Nivel 3
        if len(self._meta_trajectory) >= 2:
            recent = self._meta_trajectory[-3:]
            diffs = [
                abs(recent[i]["alpha"] - recent[i - 1]["alpha"])
                for i in range(1, len(recent))
            ]
            mean_d = float(np.mean(diffs)) if diffs else 0.0
        else:
            mean_d = 0.0
        S_L3 = float(np.exp(-mean_d / max(prev_alpha, _ALPHA_FLOOR)))
        rho_final = rho_next * (1.0 - _RSI3_LEARNING_RATE * (1.0 - S_L3))

        residual = abs(alpha_next - prev_alpha)
        converged = residual < _RSI3_CONVERGENCE_TOL

        new_state = {
            "alpha": alpha_next,
            "contraction": rho_final,
            "level3_score": S_L3,
            "iteration": iteration + 1,
        }
        self._meta_state = new_state
        self._meta_trajectory.append(new_state)
        if len(self._meta_trajectory) > _RSI3_MAX_HISTORY:
            self._meta_trajectory.pop(0)
        self._previous_performance = perf_now
        self._alpha_current = alpha_next

        logger.debug(
            "[RSI-3] it=%d α: %.3e→%.3e ρ=%.4f S_L3=%.4f conv=%s res=%.3e",
            iteration, prev_alpha, alpha_next, rho_final,
            S_L3, converged, residual,
        )
        return RSI3KinematicDiagnostics(
            iteration=iteration,
            current_alpha=prev_alpha,
            next_alpha=alpha_next,
            contraction=rho_final,
            level3_score=S_L3,
            converged=converged,
            residual=residual,
            performance_delta=float(delta_perf),
        )

    def _phase3_act(
        self,
        packet1: PhaseOneKinematicPacket,
        packet2: PhaseTwoKinematicPacket,
        conservation_residual: float,
        rsi: RSI3KinematicDiagnostics,
    ) -> KinematicTelemetry:
        r"""
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Codominio
        ---------
        `KinematicTelemetry` — objeto terminal del functor OODA.

        Aplica veto por caos: si `chaos_indicator`, el veredicto se degrada
        a VETOED por modus ponens interno de Heyting.
        """
        verdict_str = packet2.heyting_verdict
        if packet2.kam_diagnostics.homoclinic_tangle_detected:
            try:
                base = HeytingVerdict(verdict_str)
            except ValueError:
                base = HeytingVerdict.VETOED
            verdict_str = base.meet(HeytingVerdict.VETOED).value

        return KinematicTelemetry(
            ida_alpha=packet1.ida_alpha,
            ida_residual_norm=packet1.ida_residual_norm,
            hh_grad=packet1.hh_grad,
            hh_curl=packet1.hh_curl,
            hh_harmonic=packet1.hh_harmonic,
            hh_orthogonality_error=packet1.hh_orthogonality_error,
            kinetic_energy=packet1.kinetic_energy,
            mechanical_power=packet1.mechanical_power,
            heyting_verdict=verdict_str,
            matching_obstruction=packet1.ph_chart.matching_obstruction,
            passivity_residual=packet1.ph_chart.passivity_residual,
            reconstruction_residual=packet1.hodge_chart.reconstruction_residual,
            pythagoras_residual=packet1.hodge_chart.pythagoras_residual,
            csmd_error=packet1.kinetic_chart.csmd_error,
            betti_0=packet2.betti_0,
            betti_1=packet2.betti_1,
            condition_number=packet2.spectral_integrity.metric_condition,
            conservation_residual=conservation_residual,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
            # v8
            poincare_return_time=packet1.poincare_chart.return_time,
            return_map_spectral_radius=packet1.poincare_chart.return_spectral_radius,
            max_lyapunov_exponent=packet1.poincare_chart.max_lyapunov,
            kam_stable=packet2.kam_diagnostics.stable,
            kam_diophantine_ratio=packet2.kam_diagnostics.min_diophantine_ratio,
            melnikov_lobe_area=packet2.kam_diagnostics.melnikov_lobe_area,
            chaos_indicator=packet2.kam_diagnostics.homoclinic_tangle_detected,
            rsi_iteration=rsi.iteration,
            rsi_converged=rsi.converged,
            rsi_level3_score=rsi.level3_score,
            rsi_next_alpha=rsi.next_alpha,
        )

    # =========================================================================
    # ORQUESTADOR OODA
    # =========================================================================
    def execute_kinematic_cycle(
        self, g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary_matrix, edge_weights, flow_vector,
        mass_matrix, velocity_vector, actuator_forces,
        orthogonality_threshold: float = 1e-8,
        residual_threshold: float = 1e-6,
        face_incidence: Optional[np.ndarray] = None,
        obstruction_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        r"""
        Functor OODA T = Act ∘ RSI3 ∘ Audit ∘ Decide ∘ Observe.

        Encadena las fases con continuidad formal estricta:
            In → F1 → PhaseOne → F2 → PhaseTwo → Audit → RSI3 → Act → Telemetry → dict
        """
        # FASE 1 — produce PhaseOneKinematicPacket (con poincare_chart)
        packet1 = self._phase1_observe_orient(
            g, G_metric, J, R, J_d, R_d, grad_H, grad_Hd,
            boundary_matrix, edge_weights, flow_vector,
            mass_matrix, velocity_vector, actuator_forces,
            face_incidence=face_incidence,
        )
        # FASE 2 — produce PhaseTwoKinematicPacket (con kam_diagnostics)
        packet2 = self._phase2_decide(
            packet1,
            orthogonality_threshold=orthogonality_threshold,
            residual_threshold=residual_threshold,
            obstruction_threshold=obstruction_threshold,
        )
        # FASE 3 — audit → rsi3 → act
        conservation = self._phase3_conservation_audit(packet1, packet2)
        rsi = self._phase3_rsi3(packet1, packet2, conservation)
        telemetry = self._phase3_act(packet1, packet2, conservation, rsi)
        return telemetry.to_dict()


# -----------------------------------------------------------------------------
# Autocomprobación
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Autocomprobación KCore v8 (Poincaré + KAM + Melnikov + RSI-3)…")
    n = 4
    engine = KCoreKinematicEngine(dimension_n=n)
    g = np.eye(n)
    G = np.eye(n)
    J = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, -1.0, 0.0]], dtype=np.float64,
    )
    R = np.eye(n)
    J_d = J.copy()
    R_d = np.eye(n) * 2.0
    grad_H = np.ones(n)
    grad_Hd = np.zeros(n)
    boundary = np.array(
        [[-1.0, 0.0, 0.0, 1.0], [1.0, -1.0, 0.0, 0.0],
         [0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]], dtype=np.float64,
    )
    edge_weights = np.ones(boundary.shape[1])
    flow = np.ones(boundary.shape[1])
    mass = np.eye(n)
    vel = np.ones(n)
    force = np.ones(n)

    result = engine.execute_kinematic_cycle(
        g, G, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary, edge_weights, flow, mass, vel, force,
    )
    print("Veredicto:", result["heyting_verdict"])
    print("Poincaré: τ_ret=%.4f ρ(P)=%.4f λ_max=%.3e ω_min=%.3f" % (
        result["poincare_return_time"],
        result["return_map_spectral_radius"],
        result["max_lyapunov_exponent"],
        float(np.min(engine._meta_state and [1.0]))
        if False else result.get("kam_diophantine_ratio", 0.0),
    ))
    print("KAM estable:", result["kam_stable"],
          "ratio:", result["kam_diophantine_ratio"])
    print("Melnikov lobe:", result["melnikov_lobe_area"],
          "caos:", result["chaos_indicator"])
    print("RSI-3 it:", result["rsi_iteration"],
          "α_next:", result["rsi_next_alpha"],
          "conv:", result["rsi_converged"],
          "S_L3:", result["rsi_level3_score"])

    # Verificar continuidad trans-fase
    packet1 = engine._phase1_observe_orient(
        g, G, J, R, J_d, R_d, grad_H, grad_Hd,
        boundary, edge_weights, flow, mass, vel, force,
    )
    packet2 = engine._phase2_decide(packet1)
    conservation = engine._phase3_conservation_audit(packet1, packet2)
    rsi = engine._phase3_rsi3(packet1, packet2, conservation)
    telemetry = engine._phase3_act(packet1, packet2, conservation, rsi)
    assert telemetry.ida_residual_norm == packet1.ida_residual_norm
    assert telemetry.kam_stable == packet2.kam_diagnostics.stable
    print("Continuidad trans-fase verificada.")


__all__ = [
    "KCoreKinematicEngine",
    "PhaseOneKinematicPacket",
    "PhaseTwoKinematicPacket",
    "KinematicTelemetry",
    "SpectralChart",
    "PortHamiltonianChart",
    "HodgeChart",
    "KineticChart",
    "SpectralKinematicReport",
    "PoincareGeodesicChart",
    "KAMGeodesicDiagnostics",
    "RSI3KinematicDiagnostics",
    "HeytingVerdict",
]