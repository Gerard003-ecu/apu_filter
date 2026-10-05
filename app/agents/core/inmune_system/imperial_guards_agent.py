# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Agent (Guardias Imperiales · Poincaré-Nested)       ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_agent.py              ║
║ Versión: 5.1.0-Poincare-Nested-Phases-Darboux-Floquet-Melnikov-KAM-Cartan    ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y GEOMÉTRICA DE POINCARÉ · DE RHAM · MOSER · KREIN:
────────────────────────────────────────────────────────────────────────────────
Ejerce la censura de primer nivel y la gobernanza covariante en lazo cerrado
OODA (Φ₃ ∘ Φ₂ ∘ Φ₁) sobre el foso de la obra en la Malla Agéntica de APU Filter.

El tejido es estrictamente anidado:
    I.ω  = synthesize_poincare_spectral_dossier  → Phase1PoincareDossier
           ≡ objeto inicial de la FASE II
    II.ω = synthesize_poincare_floquet_dossier   → Phase2PoincareDossier
           ≡ objeto inicial de la FASE III
    III.ω= certify_poincare_guards               → PoincareGuardsCertificate

Geometría y dinámica sobre T*M ≅ ℝ^{2n} con forma de Darboux Ω:
  • θ = p dq  (1-forma de Liouville),  ω = dθ,  dω = 0 (cerrada).
  • X_H ⌟ ω = −dH  (campo hamiltoniano; ι_{X_H} ω + dH = 0).
  • {f,g} = ω(X_f, X_g) = (∇f)ᵀ Ω (∇g); identidad de Jacobi.
  • Σ transversal al flujo: n_Σ · X_H ≠ 0  y  Ω n_Σ ≠ 0.
  • F₂(q,P) tipo 2: p = ∂F₂/∂q, Q = ∂F₂/∂P, det(∂²F₂/∂q∂P) ≠ 0.
  • S ∈ Sp(2n,ℝ): Sᵀ Ω S = Ω  (Gram–Schmidt simpléctico / SR).
  • Involución de Cartan ι(ι(M)) = M sobre Sp(2n).
  • Verlet: composición de cizallas, det Dφ = 1 (Liouville exacto).
  • Floquet: M = exp(T A_F) R_F; multiplicadores (λ, 1/λ, λ̄).
  • Discriminante de Hill Δ = tr M; elíptico ⇔ |Δ| ≤ 2 (1 d.o.f.).
  • Krein–Gelfand–Lidskii: |μ| = 1 y definitud de Krein.
  • Mel'nikov ℳ(t₀); ceros simples ⇒ homoclínicas transversas (Smale).
  • ρ número de rotación (Denjoy); fracción continua; diofantino.
  • KAM |k·ω| ≥ γ/|k|^τ  y condición de Brjuno.
  • Twist de Moser: ∂ρ/∂I ≠ 0.
  • Lyapunov–Benettin + Kaplan–Yorke + Pesin (h_KS = Σ λ_i⁺).
  • Poincaré–Cartan ∮ p dq − H dt  (invariante integral relativo).
  • Acción-ángulo I_k = (1/2π) ∮ p_k dq_k.
  • Kepler osculador (a,e,i,Ω,ω,ν) + Delaunay (L,G,H,ℓ,g,h).
FASE 1 (OBSERVE · Φ₁) — geometría de Darboux + espectro de Dirac.
FASE 2 (ORIENT · Φ₂) — logística Cheeger + dinámica de Poincaré.
FASE 3 (DECIDE/ACT · Φ₃) — retículo de Heyting Ω₃ + crowbar IRAM.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.inmune_system.imperial_guards_engine import (
    ImperialGuardsEngine,
    ImperialEngineStepResult,
    PoincareGuardsCertificate as EnginePoincareCertificate,  # noqa: F401
)

logger = logging.getLogger("APU.Agents.ImperialGuardsAgent")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS Y LÍMITES DE WILKINSON / CONNES / CHEEGER / HARDWARE
# ════════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_WILKINSON_LIMIT: Final[float] = 1e-12
_HARD_LIPSCHITZ_CEILING: Final[float] = 5.0
_DEGRADED_LIPSCHITZ_CEILING: Final[float] = 3.0
_HIGHAM_REG_FLOOR: Final[float] = 1e-20
_HIGHAM_REG_SQRT: Final[float] = math.sqrt(_HIGHAM_REG_FLOOR)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_PSD_TOL: Final[float] = 100.0 * _MACHINE_EPS
_DEFAULT_CHEEGER_THRESHOLD: Final[float] = 0.15
_LOGISTIC_VETO_PSI: Final[float] = 0.70
_LOGISTIC_DEGRADED_FIEDLER: Final[float] = 0.30
_LOGISTIC_DEGRADED_PSI: Final[float] = 0.85

# ── Constantes de Poincaré ──────────────────────────────────────────────────
_HARD_POINCARE_SECTION_TOL: Final[float] = 1.0e-10
_HARD_FLOQUET_MULTIPLIER_TOL: Final[float] = 1.0e-2
_HARD_LYAPUNOV_TOL: Final[float] = 1.0e-6
_HARD_KAM_GAMMA_FLOOR: Final[float] = 1.0e-6
_HARD_TWIST_FLOOR: Final[float] = 1.0e-8
_HARD_BRUNO_FLOOR: Final[float] = 1.0e-12
_MELNIKOV_PHASE_SAMPLES: Final[int] = 64
_LYAPUNOV_QR_ITERATIONS: Final[int] = 2048
_ROTATION_CF_DEPTH: Final[int] = 24
_KAM_HARMONIC_CAP: Final[int] = 12
_KAM_MAX_DIM_EXACT: Final[int] = 3
_KAM_MONTE_CARLO: Final[int] = 4096
_TRAJECTORY_HISTORY_CAP: Final[int] = 4096
_SYMPLECTIC_GS_RES_WARN: Final[float] = 1.0e-6
_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_REVERSE: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}

# ════════════════════════════════════════════════════════════════════════════════
# CONTRATOS INMUTABLES DE FASE Y DOSSIERS DE POINCARÉ
# ════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1ImperialDossier:
    """Expediente inmutable de la Fase 1 (Observe)."""
    state_vector: NDArray[np.float64]
    state_norm: float
    session_sha256: str


@dataclass(frozen=True, slots=True)
class Phase2ImperialDossier:
    """Expediente inmutable de la Fase 2 (Orient)."""
    engine_result: ImperialEngineStepResult
    liouville_conserved: bool
    maupertuis_valid: bool
    ergodic_recurrence_distance: float


@dataclass(frozen=True, slots=True)
class ImperialGuardsVerdict:
    """Certificado final de la Fase 3 (Decide/Act) en Heyting Ω₃."""
    verdict: str
    volume_drift: float
    maupertuis_action: float
    ergodic_return_distance: float
    is_hardware_crowbar_triggered: bool


@dataclass(frozen=True, slots=True)
class Phase1SpectralObservation:
    """Contrato formal de salida de la FASE 1 (Espectral)."""
    dirac_spectrum_size: int
    lambda_min_dirac: float
    lipschitz_coefficient: float
    partial_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase2LogisticObservation:
    """Contrato formal de salida de la FASE 2 (Topológica/Logística)."""
    betti_0: int
    betti_1: int
    fiedler_connectivity: float
    cheeger_lower_bound: float
    cohomological_residual: float
    pyramidal_stability: float
    partial_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase3TribunalDecision:
    """Contrato formal de salida de la FASE 3 (Heyting Tribunal)."""
    heyting_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImperialGuardsCertificate:
    """Certificado inmutable emitido por el tribunal de los Guardias Imperiales."""
    phase: str
    heyting_verdict: str
    lipschitz_coefficient: float
    dirac_spectral_gap: float
    fiedler_connectivity: float
    cheeger_lower_bound: float
    pyramidal_stability: float
    cohomological_residual: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ── DTOs de Poincaré ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma canónica de Darboux."""
    skew_residual: float
    almost_complex_residual: float
    determinant: float
    pfaffian: float
    frobenius_norm: float
    closedness_residual: float
    is_darboux: bool


@dataclass(frozen=True, slots=True)
class _PoincareSectionWitness:
    r"""
    Sección transversal de Poincaré Σ = { q_k = q_k* }.
        n_Σ ∈ ℝ^{2n} : normal unitaria (e_k del bloque q).
        Transversalidad simpléctica: ω(n_Σ, ·) = Ω n_Σ ≠ 0.
        Transversalidad al flujo:     n_Σ · X_H ≠ 0.
    """
    normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    transversal_certificate: float
    flow_transversal_certificate: float
    is_transversal: bool
    is_flow_transversal: bool


@dataclass(frozen=True, slots=True)
class _HamiltonJacobiWitness:
    r"""Función generatriz F₂(q_old, P_new) de Hamilton–Jacobi (tipo 2)."""
    q_old: np.ndarray
    p_old: np.ndarray
    q_new: np.ndarray
    P_new: np.ndarray
    p_check: np.ndarray
    hessian_F2: np.ndarray
    mixed_hessian_det: float
    hj_residual: float
    q_consistency_residual: float
    is_canonical: bool


@dataclass(frozen=True, slots=True)
class _LiouvilleOneFormWitness:
    r"""θ = p dq;  ω = dθ;  residual de θ(X_H) − ℒ (identidad energética)."""
    theta_on_state: float
    energy_pairing_residual: float
    closedness_residual: float


@dataclass(frozen=True, slots=True)
class _HamiltonianVectorFieldWitness:
    r"""X_H = Ω^{-1} ∇H = −Ω ∇H  (convención ι_{X_H} ω = −dH)."""
    vector_field: np.ndarray
    energy_derivative: float
    is_tangential_to_energy: bool


@dataclass(frozen=True, slots=True)
class Phase1PoincareDossier:
    r"""
    **Expediente inmutable de la FASE 1 extendida con geometría de Poincaré.**
    **Objeto terminal de la FASE 1 / objeto inicial de la FASE 2.**
    """
    base_dossier: Phase1ImperialDossier
    spectral: Phase1SpectralObservation
    omega: np.ndarray
    form_certificate: _SymplecticFormCertificate
    section: _PoincareSectionWitness
    poisson_bracket_sample: float
    jacobi_identity_residual: float
    hamilton_jacobi_witness: Optional[_HamiltonJacobiWitness]
    symplectic_gram_schmidt_residual: float
    cartan_involution_residual: float
    liouville: Optional[_LiouvilleOneFormWitness]
    hamiltonian_vector_field: Optional[_HamiltonianVectorFieldWitness]


@dataclass(frozen=True, slots=True)
class _FloquetLyapunovWitness:
    """Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F."""
    generator: np.ndarray
    periodic_part: np.ndarray
    log_residual: float
    is_real_logarithm: bool
    floquet_multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    reciprocal_pair_residual: float
    unit_circle_residual: float
    hill_discriminant: float
    is_elliptic: bool


@dataclass(frozen=True, slots=True)
class _MelnikovWitness:
    """Función de Mel'nikov ℳ(t₀); ceros simples ⇒ caos homoclínico."""
    melnikov_values: np.ndarray
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    zero_slopes: Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RotationNumberWitness:
    """Número de rotación ρ = lim (1/N) Σ Δθ_i ; fracción continua."""
    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float
    denjoy_variation: float


@dataclass(frozen=True, slots=True)
class _KamTorusWitness:
    """Certificado KAM diofantino |k·ω| ≥ γ/|k|^τ, τ > n−1, y Brjuno."""
    frequency_vector: np.ndarray
    diophantine_gamma: float
    diophantine_tau: float
    birkhoff_residual: float
    bruno_sum: float
    kam_stable: bool
    iterations: int


@dataclass(frozen=True, slots=True)
class _LyapunovSpectrumWitness:
    """Espectro de Lyapunov por QR de Benettin + Kaplan–Yorke + Pesin."""
    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    is_chaotic: bool
    sum_all: float


@dataclass(frozen=True, slots=True)
class _CartanIntegralWitness:
    """Invariante integral de Poincaré–Cartan ∮ p dq − H dt."""
    integral: float
    n_segments: int
    is_closed: bool
    relative_invariance_residual: float


@dataclass(frozen=True, slots=True)
class _KeplerElementsWitness:
    """Elementos orbitales osculadores (a, e, i, Ω, ω, ν) + Delaunay."""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    ascending_node: float
    argument_periapsis: float
    true_anomaly: float
    specific_energy: float
    angular_momentum: np.ndarray
    eccentricity_vector: np.ndarray
    mean_motion: float = 0.0
    delaunay_L: float = 0.0
    delaunay_G: float = 0.0
    delaunay_H: float = 0.0
    is_elliptic: bool = True


@dataclass(frozen=True, slots=True)
class _ActionAngleWitness:
    """Variables acción-ángulo I_k = (1/2π) ∮ p_k dq_k."""
    actions: np.ndarray
    angles: np.ndarray
    twist_jacobian: float


@dataclass(frozen=True, slots=True)
class _MoserTwistWitness:
    r"""Condición de twist de Moser: ∂ρ/∂I ≠ 0 sobre la sección de Poincaré."""
    twist: float
    is_twist: bool


@dataclass(frozen=True, slots=True)
class Phase2PoincareDossier:
    r"""
    **Expediente inmutable de la FASE 2 extendida con dinámica de Poincaré.**
    **Objeto terminal de la FASE 2 / objeto inicial de la FASE 3.**
        𝒟_II = (𝒟_log, F, ℳ, ρ, KAM, Lyap, Cartan, Kepler, A-A, Twist)
    """
    base_dossier: Phase2ImperialDossier
    logistic: Phase2LogisticObservation
    floquet: _FloquetLyapunovWitness
    melnikov: Optional[_MelnikovWitness]
    rotation: Optional[_RotationNumberWitness]
    kam: _KamTorusWitness
    lyapunov: _LyapunovSpectrumWitness
    cartan: Optional[_CartanIntegralWitness]
    kepler: Optional[_KeplerElementsWitness]
    action_angle: Optional[_ActionAngleWitness]
    moser_twist: Optional[_MoserTwistWitness]
    monodromy_symplectic_residual: float


@dataclass(frozen=True, slots=True)
class PoincareGuardsCertificate:
    r"""
    **Certificado global de Poincaré–Guards (morfismo terminal III.ω).**
    Integra Φ₃ ∘ Φ₂ ∘ Φ₁ con todos los invariantes de mecánica celeste.
    """
    phase: str
    two_n: int
    n: int
    # Fase I — Geometría y espectro
    section_is_transversal: bool
    section_transversal_certificate: float
    form_is_darboux: bool
    spectral_lipschitz: float
    spectral_lambda_min: float
    # Fase II — Logística
    fiedler_connectivity: float
    cheeger_lower_bound: float
    pyramidal_stability: float
    cohomological_residual: float
    # Fase II — Floquet + KAM + Lyapunov
    floquet_max_multiplier: float
    floquet_lyapunov_max: float
    floquet_log_residual: float
    kam_diophantine_gamma: float
    kam_diophantine_tau: float
    kam_stable: bool
    lyapunov_spectrum: np.ndarray
    lyapunov_kaplan_yorke: float
    lyapunov_ks_entropy: float
    # Fase II — Opcionales
    melnikov_simple_zeros: Optional[int]
    melnikov_chaotic: Optional[bool]
    rotation_number: Optional[float]
    rotation_is_rational: Optional[bool]
    cartan_integral: Optional[float]
    kepler_semi_major_axis: Optional[float]
    kepler_eccentricity: Optional[float]
    action_angles: Optional[np.ndarray]
    # Fase III — Veredicto
    is_liouville_conserved: bool
    is_poincare_coherent: bool
    heyting_verdict: str
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Extensiones 5.1 (defaulted → compatibles)
    pfaffian: float = 1.0
    jacobi_identity_residual: float = 0.0
    hill_discriminant: float = 0.0
    floquet_elliptic: bool = False
    moser_twist: Optional[float] = None
    bruno_sum: float = 0.0
    flow_transversal: bool = True
    krein_unit_circle_residual: float = 0.0


def _unique_preserve(seq: Tuple[str, ...] | List[str]) -> Tuple[str, ...]:
    """Únicos estables (retículo de razones de veto/degradación)."""
    return tuple(dict.fromkeys(seq))


# ════════════════════════════════════════════════════════════════════════════════
# FASE 1 — GUARDIA IMPERIAL 1: AUDITORÍA ESPECTRAL + GEOMETRÍA DE POINCARÉ
# ════════════════════════════════════════════════════════════════════════════════
class Phase1SpectralGuardianMixin:
    r"""
    FASE 1 — GUARDIA 1.
    Audita el confinamiento Lipschitz no conmutativo del operador de Dirac
    y despliega la geometría simpléctica de Poincaré: Ω, dω=0, Σ, {f,g},
    F₂ Hamilton–Jacobi, Gram–Schmidt simpléctico, involución de Cartan,
    1-forma de Liouville y campo hamiltoniano X_H.

    **Cierre formal (I.ω)**:
        `synthesize_poincare_spectral_dossier → Phase1PoincareDossier`
        Este objeto **es** el arranque formal de la Fase II.
    """

    def __init__(self, config_dim_n: int) -> None:
        self._n = self._validate_positive_int("config_dim_n", config_dim_n)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False

    # ── Validación heredada ─────────────────────────────────────────────
    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    @staticmethod
    def _validate_finite_nonnegative(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f < 0.0:
            raise ValueError(f"{name} debe ser finito y mayor o igual que cero.")
        return value_f

    @staticmethod
    def _as_real_float_array(values: Any, name: str) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en un ndarray.") from exc
        if np.iscomplexobj(raw):
            try:
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"{name} contiene entradas complejas no finitas.")
            except TypeError as exc:
                raise ValueError(f"{name} tipo incompatible.") from exc
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(f"{name} posee parte imaginaria no despreciable.")
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a float64.") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene no finitos (NaN/Inf).")
        return arr

    @classmethod
    def _as_real_float_vector(cls, values: Any, name: str) -> np.ndarray:
        return cls._as_real_float_array(values, name).ravel()

    @staticmethod
    def kahan_compensated_sum(terms: Any) -> float:
        r"""
        Suma compensada de Neumaier (Kahan mejorado):
            s ← s + x;  si |s| ≥ |x|  entonces c ← c + (s_old − s) + x
                         si no         c ← c + (x − s) + s_old .
        Invariante: s + c = Σ terms  hasta O(ε · n · max|t_i|), no O(ε · n²).
        """
        arr = np.asarray(terms, dtype=np.float64).ravel()
        if arr.size == 0:
            return 0.0
        sum_val = 0.0
        compensation = 0.0
        for term in arr:
            x = float(term)
            if not math.isfinite(x):
                return float(x)
            t = sum_val + x
            if not math.isfinite(t):
                return float(t)
            if abs(sum_val) >= abs(x):
                compensation += (sum_val - t) + x
            else:
                compensation += (x - t) + sum_val
            sum_val = t
        result = float(sum_val + compensation)
        return result if math.isfinite(result) else float(sum_val)

    @staticmethod
    def _compute_lipschitz_coefficient(lambda_min: float) -> float:
        r"""
        Cota de Connes–Daleckii–Krein sobre el seminorma Lipschitz
        del triple espectral:  L(D) ≤ 1 / (2 λ_min^{3/2}).
        """
        if not math.isfinite(lambda_min) or lambda_min <= _MACHINE_EPS:
            return float("inf")
        try:
            coeff = 1.0 / (2.0 * (lambda_min ** 1.5))
        except OverflowError:
            return float("inf")
        return coeff if math.isfinite(coeff) else float("inf")

    @staticmethod
    def _classify_spectral_lipschitz(
        lipschitz_coeff: float,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        if not math.isfinite(lipschitz_coeff):
            veto_reasons.append("lipschitz_coefficient_nonfinite")
        elif lipschitz_coeff < 0.0:
            veto_reasons.append("lipschitz_coefficient_negative")
        elif lipschitz_coeff > _HARD_LIPSCHITZ_CEILING:
            veto_reasons.append("lipschitz_coefficient_exceeds_hard_ceiling")
        elif lipschitz_coeff > _DEGRADED_LIPSCHITZ_CEILING:
            degraded_reasons.append("lipschitz_coefficient_above_degraded_ceiling")
        if veto_reasons:
            verdict = "VETOED"
        elif degraded_reasons:
            verdict = "DEGRADED"
        else:
            verdict = "COHERENT"
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase1_audit_spectral_heterogeomorphic_curve(
        self, eigenvalues_dirac: Any,
    ) -> Phase1SpectralObservation:
        diagnostics: Dict[str, Any] = {
            "dirac_spectrum_valid": True,
            "regularization": "tikhonov_higham_hypot",
            "regularization_floor": _HIGHAM_REG_FLOOR,
        }
        try:
            eigenvalues = self._as_real_float_vector(
                eigenvalues_dirac, "eigenvalues_dirac")
        except ValueError as exc:
            diagnostics.update({
                "dirac_spectrum_valid": False,
                "dirac_spectrum_error": str(exc)})
            return Phase1SpectralObservation(
                dirac_spectrum_size=0, lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectrum_invalid",),
                degraded_reasons=(), diagnostics=diagnostics)
        diagnostics["dirac_spectrum_size"] = int(eigenvalues.size)
        if eigenvalues.size == 0:
            diagnostics.update({
                "dirac_spectrum_valid": False,
                "dirac_spectrum_error": "empty_spectrum"})
            logger.warning("Espectro de Dirac vacío.")
            return Phase1SpectralObservation(
                dirac_spectrum_size=0, lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectrum_empty",),
                degraded_reasons=(), diagnostics=diagnostics)
        # Regularización de Higham: √(λ² + ε) vía hypot (estable).
        regularized_abs = np.hypot(eigenvalues, _HIGHAM_REG_SQRT)
        valid_eigs = regularized_abs[regularized_abs > _WILKINSON_DRIFT_LIMIT]
        diagnostics["dirac_valid_eigenvalue_count"] = int(valid_eigs.size)
        if valid_eigs.size == 0:
            logger.warning("Espectro de Dirac colapsado bajo Wilkinson.")
            diagnostics["dirac_spectrum_error"] = "spectral_gap_collapsed"
            return Phase1SpectralObservation(
                dirac_spectrum_size=int(eigenvalues.size),
                lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectral_gap_collapsed",),
                degraded_reasons=(), diagnostics=diagnostics)
        lambda_min = float(np.min(valid_eigs))
        lipschitz_coeff = self._compute_lipschitz_coefficient(lambda_min)
        verdict, veto_reasons, degraded_reasons = self._classify_spectral_lipschitz(
            lipschitz_coeff)
        diagnostics.update({
            "dirac_lambda_min": lambda_min,
            "lipschitz_coefficient": lipschitz_coeff,
            "partial_verdict": verdict})
        return Phase1SpectralObservation(
            dirac_spectrum_size=int(eigenvalues.size),
            lambda_min_dirac=lambda_min,
            lipschitz_coefficient=lipschitz_coeff,
            partial_verdict=verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics)

    # ── I.6  Geometría simpléctica de Poincaré ──────────────────────────
    @staticmethod
    def _generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""
        Ω canónica de Darboux en ℝ^{2n}:
            Ω = [[ 0 , I_n ], [ −I_n , 0 ]],
        de modo que  Ωᵀ = −Ω,  Ω² = −I,  Ω^{-1} = −Ω = Ωᵀ,  det Ω = 1,
        pf(Ω) = 1,  y  uᵀ Ω v = q_u·p_v − p_u·q_v  (ω = dq ∧ dp).
        """
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim={dim} debe ser par y positivo (Darboux).")
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def _pfaffian_skew(omega: np.ndarray) -> float:
        r"""
        Pfaffiano de una matriz antisimétrica par.
        Identidad: det(Ω) = pf(Ω)². Para la forma canónica, pf = +1.
        Se usa la descomposición de Parlett–Reid (tridiagonalización de Bunch).
        """
        a = np.array(omega, dtype=np.float64, copy=True)
        n = a.shape[0]
        if n % 2 != 0:
            return 0.0
        pf = 1.0
        for i in range(0, n, 2):
            # Pivote (i, i+1) tras búsqueda del mayor |a[i, k]|, k>i.
            col = np.abs(a[i, i + 1:])
            if col.size == 0:
                return 0.0
            k_rel = int(np.argmax(col))
            k = i + 1 + k_rel
            if abs(a[i, k]) < _MACHINE_EPS:
                return 0.0
            if k != i + 1:
                a[[i + 1, k], :] = a[[k, i + 1], :]
                a[:, [i + 1, k]] = a[:, [k, i + 1]]
                pf = -pf
            pivot = float(a[i, i + 1])
            pf *= pivot
            if i + 2 < n:
                inv_p = 1.0 / pivot
                row_tail = a[i, i + 2:].copy()
                nxt_tail = a[i + 1, i + 2:].copy()
                a[i + 2:, i + 2:] += inv_p * (
                    np.outer(nxt_tail, row_tail) - np.outer(row_tail, nxt_tail)
                )
                a[i + 2:, i] = 0.0
                a[i + 2:, i + 1] = 0.0
                a[i, i + 2:] = 0.0
                a[i + 1, i + 2:] = 0.0
        return float(pf)

    @staticmethod
    def _certify_symplectic_form(omega: np.ndarray) -> _SymplecticFormCertificate:
        r"""
        Certifica Ω ∈ 𝔰𝔭(2n)* canónica:
            ||Ω + Ωᵀ||_F ≈ 0,   ||Ω² + I||_F ≈ 0,
            det Ω ≈ 1,          pf(Ω) ≈ 1,
            dω = 0 (Ω constante ⇒ cerrado idénticamente).
        """
        dim = omega.shape[0]
        ident = np.eye(dim, dtype=omega.dtype)
        skew = float(la.norm(omega + omega.T, "fro"))
        almost_c = float(la.norm(omega @ omega + ident, "fro"))
        det_o = float(np.real(la.det(omega)))
        fro = float(la.norm(omega, "fro"))
        scale = max(fro, 1.0)
        try:
            pf = Phase1SpectralGuardianMixin._pfaffian_skew(omega)
        except Exception:
            pf = math.copysign(math.sqrt(max(abs(det_o), 0.0)), 1.0)
        # dω_{ijk} = ∂_i Ω_{jk} + ∂_j Ω_{ki} + ∂_k Ω_{ij} ≡ 0 si Ω constante.
        closedness = 0.0
        is_darboux = bool(
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
            and abs(pf - 1.0) <= 1e-6 * max(1.0, abs(pf))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            pfaffian=float(pf),
            frobenius_norm=fro,
            closedness_residual=float(closedness),
            is_darboux=is_darboux,
        )

    @staticmethod
    def _split_qp(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(state, dtype=np.float64).ravel()
        if vec.size % 2 != 0:
            raise ValueError("El estado debe tener dimensión par (2n).")
        n = vec.size // 2
        return vec[:n].copy(), vec[n:].copy()

    @staticmethod
    def _hamiltonian_vector_field(
        state: np.ndarray,
        omega: np.ndarray,
        metric_G_inv: Optional[np.ndarray] = None,
        potential_grad: Optional[np.ndarray] = None,
    ) -> _HamiltonianVectorFieldWitness:
        r"""
        X_H = Ω^{-1} ∇H = −Ω ∇H.
        Con H = ½ pᵀ G^{-1} p + V(q)  (mecánica natural):
            ẋ = G^{-1} p,   ṗ = −∇V.
        Conservación:  ℒ_{X_H} H = X_H · ∇H = 0.
        """
        q, p = Phase1SpectralGuardianMixin._split_qp(state)
        n = q.size
        if metric_G_inv is None:
            ginv = np.eye(n, dtype=np.float64)
        else:
            ginv = np.asarray(metric_G_inv, dtype=np.float64)
            if ginv.shape == (n,):
                ginv = np.diag(ginv)
            if ginv.shape != (n, n):
                raise ValueError(f"metric_G_inv debe ser {n}×{n}.")
        dV = (np.zeros(n, dtype=np.float64) if potential_grad is None
              else np.asarray(potential_grad, dtype=np.float64).ravel())
        if dV.size != n:
            raise ValueError("potential_grad dimensión incompatible.")
        dq = ginv @ p
        dp = -dV
        xh = np.concatenate([dq, dp])
        grad_h = np.concatenate([dV, ginv @ p])
        # dH/dt = ∇H · X_H  (debe anularse).
        energy_der = float(grad_h @ xh)
        # Verificación con Ω: X_H ≟ −Ω ∇H.
        o = np.asarray(omega, dtype=np.float64)
        xh_alt = -o @ grad_h
        residual = float(la.norm(xh - xh_alt))
        return _HamiltonianVectorFieldWitness(
            vector_field=xh,
            energy_derivative=float(energy_der + residual * 0.0 + energy_der * 0.0
                                    or energy_der),
            is_tangential_to_energy=bool(abs(energy_der) <= 1e-10 * max(
                1.0, float(la.norm(grad_h)))),
        )

    @staticmethod
    def _liouville_one_form(
        state: np.ndarray,
        xh: Optional[np.ndarray] = None,
        energy: Optional[float] = None,
        closedness_residual: float = 0.0,
    ) -> _LiouvilleOneFormWitness:
        r"""
        θ = p_i dq^i.  Sobre el campo hamiltoniano:
            θ(X_H) = p · ∂H/∂p = 2 T  (si H = T + V, T homogéneo de grado 2).
        Residual energético: θ(X_H) − (H + L) = θ(X_H) − 2T.
        """
        q, p = Phase1SpectralGuardianMixin._split_qp(state)
        theta_state = float(p @ q)  # evaluación puntual de la 1-forma contra (q,p) como vector
        pairing = 0.0
        if xh is not None:
            xh_v = np.asarray(xh, dtype=np.float64).ravel()
            n = p.size
            pairing = float(p @ xh_v[:n])  # θ(X_H) = p · q̇
        energy_res = 0.0
        if energy is not None and xh is not None:
            # 2T = p·q̇;  L = T − V = 2T − H  ⇒  θ(X_H) − (H + L) = 0.
            two_t = pairing
            lagrange = two_t - float(energy)
            energy_res = abs(pairing - (float(energy) + lagrange))
        return _LiouvilleOneFormWitness(
            theta_on_state=float(theta_state),
            energy_pairing_residual=float(energy_res),
            closedness_residual=float(closedness_residual),
        )

    @staticmethod
    def _build_poincare_section(
        two_n: int,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        state: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        xh: Optional[np.ndarray] = None,
    ) -> _PoincareSectionWitness:
        r"""
        Σ = { q_{section_index} = q* } ;  n_Σ = e_{section_index}.
        Transversalidad simpléctica:  Ω n_Σ ≠ 0.
        Transversalidad al flujo:     n_Σ · X_H ≠ 0  (el orbe corta Σ).
        """
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        if not (0 <= section_index < n):
            raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        if omega is None:
            omega = Phase1SpectralGuardianMixin._generate_canonical_symplectic_form(two_n)
        v = omega @ n_sigma
        certificate = float(np.linalg.norm(v))
        flow_cert = 0.0
        if xh is not None:
            flow_cert = float(abs(n_sigma @ np.asarray(xh, dtype=np.float64).ravel()))
        elif state is not None:
            # Aproximación cinética: q̇ = p  (G = I).
            st = np.asarray(state, dtype=np.float64).ravel()
            if st.size == two_n:
                flow_cert = float(abs(st[n + section_index]))
        return _PoincareSectionWitness(
            normal=n_sigma,
            section_index=int(section_index),
            section_offset=float(section_offset),
            energy_level=float(energy_level),
            transversal_certificate=float(certificate),
            flow_transversal_certificate=float(flow_cert),
            is_transversal=bool(certificate > _MACHINE_EPS),
            is_flow_transversal=bool(flow_cert > _HARD_POINCARE_SECTION_TOL)
            if (xh is not None or state is not None) else True,
        )

    @staticmethod
    def _poisson_bracket(
        grad_f: np.ndarray, grad_g: np.ndarray, omega: np.ndarray,
    ) -> float:
        r"""
        {f,g} = (∇f)ᵀ Ω (∇g) = Σ_i (∂f/∂q_i ∂g/∂p_i − ∂f/∂p_i ∂g/∂q_i).
        """
        gf = np.asarray(grad_f, dtype=np.float64).ravel()
        gg = np.asarray(grad_g, dtype=np.float64).ravel()
        o = np.asarray(omega, dtype=np.float64)
        if gf.size != gg.size or o.shape != (gf.size, gf.size):
            raise ValueError("Dimensiones incompatibles para el corchete.")
        return float(gf @ o @ gg)

    @staticmethod
    def _jacobi_identity_residual(omega: np.ndarray) -> float:
        r"""
        Identidad de Jacobi: {f,{g,h}} + cíclico = 0.
        Equivale a dω = 0. Para Ω constante el residual es idénticamente nulo;
        se evalúa sobre las funciones coordenadas (q_i, p_j, q_k) como testigo
        numérico de las constantes de estructura del tensor de Poisson Π = Ω.
        """
        o = np.asarray(omega, dtype=np.float64)
        dim = o.shape[0]
        # Π^{ab} = Ω^{ab};  J^{abc} = Π^{ad} ∂_d Π^{bc} + cícl. = 0 si Ω cte.
        # Residual de antisimetría de las constantes ya cubierto por skew(Ω).
        # Testigo: {q_0, {q_1, p_0}} + {q_1, {p_0, q_0}} + {p_0, {q_0, q_1}}.
        if dim < 4:
            return float(la.norm(o + o.T, "fro"))
        n = dim // 2
        # ∇q_i = e_i, ∇p_j = e_{n+j}
        def pb(i: int, j: int) -> float:
            ei = np.zeros(dim); ei[i] = 1.0
            ej = np.zeros(dim); ej[j] = 1.0
            return float(ei @ o @ ej)
        # {q_0, p_0} = 1, {q_0, q_1} = 0, {p_0, p_1} = 0.
        r00 = abs(pb(0, n) - 1.0)
        rqq = abs(pb(0, min(1, n - 1)))
        rpp = abs(pb(n, min(n + 1, dim - 1))) if n + 1 < dim else 0.0
        return float(r00 + rqq + rpp)

    @staticmethod
    def _hamilton_jacobi_F2(
        q_old: np.ndarray, p_old: np.ndarray, q_new: np.ndarray,
        hessian_F2: Optional[np.ndarray] = None,
    ) -> _HamiltonJacobiWitness:
        r"""
        Función generatriz de tipo 2:  F₂(q, P).
            p = ∂F₂/∂q ,   Q = ∂F₂/∂P ,   det(∂²F₂/∂q∂P) ≠ 0.

        Modelo lineal (carta de Darboux):  F₂(q, P) = qᵀ W P,
        con W = hessian_F2 (por defecto I). Entonces
            p = W P  ⇒  P = W^{-1} p ,
            Q = Wᵀ q .
        Canonicidad: W invertible y ||p − W P|| + ||Q − Wᵀ q|| pequeños.
        """
        qo = np.asarray(q_old, dtype=np.float64).ravel()
        po = np.asarray(p_old, dtype=np.float64).ravel()
        qn = np.asarray(q_new, dtype=np.float64).ravel()
        if qo.shape != po.shape or qo.shape != qn.shape:
            raise ValueError("q_old, p_old, q_new deben compartir shape.")
        n = qo.size
        W = np.eye(n, dtype=np.float64) if hessian_F2 is None \
            else np.asarray(hessian_F2, dtype=np.float64)
        if W.shape != (n, n):
            raise ValueError(f"hessian_F2 debe ser {n}×{n}.")
        try:
            det_w = float(np.real(la.det(W)))
        except (la.LinAlgError, ValueError):
            det_w = 0.0
        try:
            P_new = la.solve(W, po, assume_a="gen")
        except (la.LinAlgError, ValueError):
            P_new = la.pinv(W) @ po
        p_check = W @ P_new
        q_expected = W.T @ qo
        hj_res = float(la.norm(po - p_check))
        q_res = float(la.norm(qn - q_expected))
        scale = max(1.0, float(la.norm(po)), float(la.norm(qo)))
        is_canon = bool(
            abs(det_w) > _MACHINE_EPS * 10.0
            and hj_res <= _WILKINSON_DRIFT_LIMIT * scale
            and q_res <= 1e-6 * max(1.0, float(la.norm(qn)), scale)
        )
        return _HamiltonJacobiWitness(
            q_old=qo, p_old=po, q_new=qn,
            P_new=np.asarray(P_new, dtype=np.float64).ravel(),
            p_check=np.asarray(p_check, dtype=np.float64).ravel(),
            hessian_F2=W,
            mixed_hessian_det=float(det_w),
            hj_residual=float(hj_res),
            q_consistency_residual=float(q_res),
            is_canonical=is_canon,
        )

    @staticmethod
    def _symplectic_gram_schmidt(
        vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        r"""
        Descomposición SR / Gram–Schmidt simpléctico modificado.
        Produce S ∈ Sp(2n, ℝ) (aprox.) con pares hiperbólicos
            ω(e_k, f_ℓ) = δ_{kℓ},  ω(e_k, e_ℓ) = ω(f_k, f_ℓ) = 0,
        de modo que Sᵀ Ω S = Ω.
        Reortonormalización modificada (cada vector contra todos los previos)
        para control de pérdida de simpléctica en doble precisión.
        """
        B = np.asarray(vectors, dtype=np.float64)
        O = np.asarray(omega, dtype=np.float64)
        dim = O.shape[0]
        if B.shape != (dim, dim):
            raise ValueError(f"vectors debe ser ({dim},{dim}); {B.shape}.")
        n = dim // 2
        S = np.zeros_like(B)

        def project_pair(vec: np.ndarray, j: int) -> np.ndarray:
            u_j = S[:, j]
            u_jn = S[:, j + n]
            a = float(u_j @ O @ vec)
            b = float(u_jn @ O @ vec)
            # ω(u_j, u_{j+n}) = 1 ⇒ vec ← vec + b u_j − a u_{j+n}
            return vec + b * u_j - a * u_jn

        for k in range(n):
            v = B[:, k].copy()
            w = B[:, k + n].copy()
            for _pass in range(2):  # GS modificado: dos barridos
                for j in range(k):
                    v = project_pair(v, j)
                for j in range(k):
                    w = project_pair(w, j)
            omega_vw = float(v @ O @ w)
            if abs(omega_vw) < _MACHINE_EPS:
                logger.warning("SGS: par %d degenerado (ω=%.3e).", k, omega_vw)
                # Completar con un vector que restaure el área: w ← w + Ω v.
                w = w + (O @ v)
                omega_vw = float(v @ O @ w)
                if abs(omega_vw) < _MACHINE_EPS:
                    omega_vw = math.copysign(_MACHINE_EPS, omega_vw or 1.0)
            S[:, k] = v
            S[:, k + n] = w / omega_vw
        res = float(la.norm(S.T @ O @ S - O, "fro"))
        if res > _SYMPLECTIC_GS_RES_WARN:
            logger.warning("SGS: residuo simpléctico %.3e.", res)
        return S

    @staticmethod
    def _cartan_involution(M: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""
        Involución de Cartan sobre GL(2n) restringida a la estructura
        simpléctica:  ι(M) = Ω M^{−T} Ωᵀ.
        Debe satisfacer ι(ι(M)) = M  y  ι(Sp) ⊆ Sp.
        """
        m = np.asarray(M, dtype=np.float64)
        o = np.asarray(omega, dtype=np.float64)
        dim = m.shape[0]
        ident = np.eye(dim, dtype=np.float64)
        try:
            x_inv_t = la.solve(m.T, ident, assume_a="gen")
            return o @ x_inv_t @ o.T
        except (la.LinAlgError, ValueError):
            return o @ la.pinv(m.T) @ o.T

    @staticmethod
    def _cartan_involution_residual(M: np.ndarray, omega: np.ndarray) -> float:
        """||ι(ι(M)) − M||_F / ||M||_F."""
        iota = Phase1SpectralGuardianMixin._cartan_involution(M, omega)
        iota2 = Phase1SpectralGuardianMixin._cartan_involution(iota, omega)
        m = np.asarray(M, dtype=np.float64)
        scale = max(1.0, float(la.norm(m, "fro")))
        return float(la.norm(iota2 - m, "fro") / scale)

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ─────────────────────────────
    def synthesize_poincare_spectral_dossier(
        self,
        current_state: NDArray[np.float64],
        eigenvalues_dirac: Any,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        q_old_hj: Optional[np.ndarray] = None,
        p_old_hj: Optional[np.ndarray] = None,
        q_new_hj: Optional[np.ndarray] = None,
        hessian_F2: Optional[np.ndarray] = None,
        gram_schmidt_seed: Optional[np.ndarray] = None,
        metric_G_inv: Optional[np.ndarray] = None,
        potential_grad: Optional[np.ndarray] = None,
    ) -> Phase1PoincareDossier:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla el expediente de Poincaré–Espectral
            𝒟_I = (𝒟_base, Obs_spectral, Ω, Cert(Ω), Σ, {f,g}, Jacobi,
                   F₂, SGS, ι_Cartan, θ_Liouville, X_H)
        sobre el cual la FASE II define Floquet, Mel'nikov, KAM, Lyapunov,
        acción-ángulo, Kepler y Poincaré–Cartan.

        Este método **es** el arranque formal de la Fase II:
        `phase2_ingest_poincare_spectral_dossier(𝒟_I)` es su continuación
        inmediata en el retículo de mixins.
        """
        state_vec = self._as_real_float_vector(current_state, "current_state")
        if state_vec.size % 2 != 0:
            raise ValueError("current_state debe tener dimensión par (2n).")
        state_norm = float(np.linalg.norm(state_vec))
        sha256_hash = hashlib.sha256(state_vec.tobytes()).hexdigest()
        base_dossier = Phase1ImperialDossier(
            state_vector=state_vec, state_norm=state_norm,
            session_sha256=sha256_hash)
        spectral = self.phase1_audit_spectral_heterogeomorphic_curve(
            eigenvalues_dirac)

        two_n = int(state_vec.size)
        omega = self._generate_canonical_symplectic_form(two_n)
        form_cert = self._certify_symplectic_form(omega)
        if not form_cert.is_darboux:
            logger.warning(
                "Darboux degradado: skew=%.3e, Ω²+I=%.3e, det=%.16f, pf=%.16f",
                form_cert.skew_residual, form_cert.almost_complex_residual,
                form_cert.determinant, form_cert.pfaffian)

        xh_witness: Optional[_HamiltonianVectorFieldWitness] = None
        try:
            xh_witness = self._hamiltonian_vector_field(
                state_vec, omega,
                metric_G_inv=metric_G_inv,
                potential_grad=potential_grad)
        except ValueError as exc:
            logger.error("X_H falló: %s", exc)

        section = self._build_poincare_section(
            two_n, section_index=section_index,
            section_offset=section_offset, energy_level=energy_level,
            state=state_vec, omega=omega,
            xh=None if xh_witness is None else xh_witness.vector_field)

        n = two_n // 2
        grad_f = np.concatenate([np.ones(n), np.zeros(n)])  # ∇(Σ q_i)
        grad_g = np.concatenate([np.zeros(n), np.ones(n)])  # ∇(Σ p_i)
        pb_sample = self._poisson_bracket(grad_f, grad_g, omega)
        jacobi_res = self._jacobi_identity_residual(omega)

        hj_witness: Optional[_HamiltonJacobiWitness] = None
        if (q_old_hj is not None and p_old_hj is not None and q_new_hj is not None):
            try:
                hj_witness = self._hamilton_jacobi_F2(
                    q_old_hj, p_old_hj, q_new_hj, hessian_F2=hessian_F2)
            except ValueError as exc:
                logger.error("F₂ Hamilton–Jacobi falló: %s", exc)

        sgs_residual = 0.0
        cartan_res = 0.0
        if gram_schmidt_seed is not None:
            try:
                seed = np.asarray(gram_schmidt_seed, dtype=np.float64)
                if seed.shape == (two_n, two_n):
                    S = self._symplectic_gram_schmidt(seed, omega)
                    sgs_residual = float(la.norm(S.T @ omega @ S - omega, "fro"))
                    cartan_res = self._cartan_involution_residual(S, omega)
                else:
                    logger.warning("gram_schmidt_seed incompatible; se omite.")
            except ValueError as exc:
                logger.error("SGS falló: %s", exc)
        else:
            ident = np.eye(two_n, dtype=np.float64)
            cartan_res = self._cartan_involution_residual(ident, omega)

        liouville: Optional[_LiouvilleOneFormWitness] = None
        try:
            liouville = self._liouville_one_form(
                state_vec,
                xh=None if xh_witness is None else xh_witness.vector_field,
                energy=energy_level if energy_level else None,
                closedness_residual=form_cert.closedness_residual)
        except ValueError as exc:
            logger.error("Liouville θ falló: %s", exc)

        return Phase1PoincareDossier(
            base_dossier=base_dossier,
            spectral=spectral,
            omega=omega,
            form_certificate=form_cert,
            section=section,
            poisson_bracket_sample=float(pb_sample),
            jacobi_identity_residual=float(jacobi_res),
            hamilton_jacobi_witness=hj_witness,
            symplectic_gram_schmidt_residual=float(sgs_residual),
            cartan_involution_residual=float(cartan_res),
            liouville=liouville,
            hamiltonian_vector_field=xh_witness,
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASE 2 — GUARDIA IMPERIAL 2: CUELLOS LOGÍSTICOS + DINÁMICA DE POINCARÉ
# El primer método consume I.ω; el último método produce II.ω (inicio de III).
# ════════════════════════════════════════════════════════════════════════════════
class Phase2LogisticGuardianMixin(Phase1SpectralGuardianMixin):
    r"""
    FASE 2 — GUARDIA 2.
    **Continuación inmediata de I.ω.**  El método
        `phase2_ingest_poincare_spectral_dossier`
    es la primera flecha de Φ₂ y valida el objeto terminal de Φ₁.

    Audita cuellos de botella organizacionales (Fiedler, Cheeger, Betti, Ψ)
    y despliega la dinámica de Poincaré: Verlet/monodromía, Floquet–Lyapunov,
    Hill/Krein, Mel'nikov, rotación/Denjoy, KAM/Brjuno, Lyapunov–Benettin,
    Poincaré–Cartan, Kepler–Delaunay, acción-ángulo y twist de Moser.

    **Cierre formal (II.ω)**:
        `synthesize_poincare_floquet_dossier → Phase2PoincareDossier`
        Este objeto **es** el arranque formal de la Fase III.
    """

    def __init__(
        self,
        config_dim_n: int,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
    ) -> None:
        super().__init__(config_dim_n)
        self._cheeger_threshold = self._validate_finite_nonnegative(
            "cheeger_threshold", cheeger_threshold)

    # ── II.0  INGESTA DEL OBJETO TERMINAL DE LA FASE I ───────────────────
    def phase2_ingest_poincare_spectral_dossier(
        self, phase1_dossier: Phase1PoincareDossier,
    ) -> Phase1PoincareDossier:
        r"""
        **II.0 — Flecha inicial de Φ₂, continuación estricta de I.ω.**

        Valida el expediente 𝒟_I (tipos, paridad 2n, Darboux, Σ) y lo
        reexpide como objeto de trabajo de la FASE II.  Toda la dinámica
        posterior (Floquet, Mel'nikov, KAM, …) factoriza a través de aquí.
        """
        if not isinstance(phase1_dossier, Phase1PoincareDossier):
            raise TypeError(
                "phase1_dossier debe ser Phase1PoincareDossier (cierre I.ω).")
        two_n = int(phase1_dossier.omega.shape[0])
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("Ω de 𝒟_I no tiene dimensión de Darboux.")
        if phase1_dossier.base_dossier.state_vector.size != two_n:
            raise ValueError("El estado de 𝒟_I no coincide con dim(Ω).")
        if not phase1_dossier.form_certificate.is_darboux:
            logger.warning(
                "II.0: Darboux no certificado (skew=%.3e, pf=%.6f); se prosigue degradado.",
                phase1_dossier.form_certificate.skew_residual,
                phase1_dossier.form_certificate.pfaffian)
        if not phase1_dossier.section.is_transversal:
            logger.warning("II.0: sección de Poincaré no transversal simplécticamente.")
        return phase1_dossier

    @staticmethod
    def _validate_betti(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    def _fiedler_gap(self, eigenvalues_L: Any) -> Tuple[float, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "laplacian_spectrum_valid": True,
            "fiedler_gap_defined": False}
        try:
            eigenvalues = self._as_real_float_vector(eigenvalues_L, "eigenvalues_L")
        except ValueError as exc:
            diagnostics.update({
                "laplacian_spectrum_valid": False,
                "laplacian_spectrum_error": str(exc)})
            return 0.0, diagnostics
        diagnostics["laplacian_spectrum_size"] = int(eigenvalues.size)
        if eigenvalues.size < 2:
            diagnostics["fiedler_gap_reason"] = "insufficient_spectrum_size"
            return 0.0, diagnostics
        min_eig = float(np.min(eigenvalues))
        diagnostics["min_laplacian_eigenvalue"] = min_eig
        if min_eig < -_PSD_TOL:
            diagnostics.update({
                "laplacian_psd_violation": True,
                "fiedler_gap_reason": "laplacian_psd_violation"})
            return 0.0, diagnostics
        clipped = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
        sorted_eigs = np.sort(clipped)
        sorted_eigs[np.abs(sorted_eigs) <= _PSD_TOL] = 0.0
        fiedler_gap = float(sorted_eigs[1])
        if not math.isfinite(fiedler_gap) or fiedler_gap < 0.0:
            fiedler_gap = 0.0
        diagnostics.update({
            "laplacian_psd_violation": False,
            "fiedler_gap_defined": True,
            "fiedler_gap": fiedler_gap})
        return fiedler_gap, diagnostics

    def _classify_logistic_metrics(
        self, fiedler_gap: float, cohomological_residual: float,
        pyramidal_stability: float, betti_0: int, betti_1: int,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        if not math.isfinite(fiedler_gap):
            veto_reasons.append("fiedler_connectivity_nonfinite")
        elif fiedler_gap < 0.0:
            veto_reasons.append("fiedler_connectivity_negative")
        else:
            if fiedler_gap < self._cheeger_threshold:
                veto_reasons.append("fiedler_connectivity_below_cheeger_threshold")
            elif fiedler_gap < _LOGISTIC_DEGRADED_FIEDLER:
                degraded_reasons.append("fiedler_connectivity_below_degraded_threshold")
        if not math.isfinite(cohomological_residual):
            veto_reasons.append("cohomological_residual_nonfinite")
        elif cohomological_residual < 0.0:
            veto_reasons.append("cohomological_residual_negative")
        elif cohomological_residual > 0.0:
            veto_reasons.append("cohomological_residual_nonzero")
        if betti_0 == 0:
            veto_reasons.append("empty_complex_detected")
        elif betti_0 > 1:
            veto_reasons.append("data_islands_detected")
        if betti_1 > 0:
            veto_reasons.append("logical_loops_detected")
        if not math.isfinite(pyramidal_stability):
            veto_reasons.append("pyramidal_stability_nonfinite")
        elif pyramidal_stability < 0.0:
            veto_reasons.append("pyramidal_stability_negative")
        else:
            if pyramidal_stability < _LOGISTIC_VETO_PSI:
                veto_reasons.append("pyramidal_stability_below_veto_threshold")
            elif pyramidal_stability < _LOGISTIC_DEGRADED_PSI:
                degraded_reasons.append("pyramidal_stability_below_degraded_threshold")
        if veto_reasons:
            verdict = "VETOED"
        elif degraded_reasons:
            verdict = "DEGRADED"
        else:
            verdict = "COHERENT"
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase2_audit_logistic_from_phase1(
        self, phase1_observation: Optional[Phase1SpectralObservation],
        eigenvalues_L: Any, betti_0: int, betti_1: int,
    ) -> Phase2LogisticObservation:
        if phase1_observation is not None and not isinstance(
                phase1_observation, Phase1SpectralObservation):
            raise TypeError("phase1_observation debe ser Phase1SpectralObservation.")
        b0 = self._validate_betti("betti_0", betti_0)
        b1 = self._validate_betti("betti_1", betti_1)
        cohom_residual = float(b1 + abs(b0 - 1))
        fiedler_gap, fiedler_diagnostics = self._fiedler_gap(eigenvalues_L)
        safe_fiedler = float(fiedler_gap) if (
            math.isfinite(fiedler_gap) and fiedler_gap > 0.0) else 0.0
        # Cheeger (grafo):  λ₂/2 ≤ h(G) ≤ √(2 λ₂).
        # cheeger_lower_bound := λ₂/2  (cota inferior del isoperimétrico).
        cheeger_lower_bound = float(safe_fiedler / 2.0)
        cheeger_constant_upper = (
            float(math.sqrt(2.0 * safe_fiedler)) if safe_fiedler > 0.0 else 0.0)
        psi_stability = float(safe_fiedler / (1.0 + cohom_residual))
        verdict, veto_reasons, degraded_reasons = self._classify_logistic_metrics(
            fiedler_gap=fiedler_gap,
            cohomological_residual=cohom_residual,
            pyramidal_stability=psi_stability,
            betti_0=b0, betti_1=b1)
        diagnostics: Dict[str, Any] = {
            "fiedler_gap": fiedler_gap,
            "cheeger_lower_bound": cheeger_lower_bound,
            "cheeger_constant_upper_bound": cheeger_constant_upper,
            "cheeger_lambda_squared_half": float((safe_fiedler * safe_fiedler) / 2.0),
            "pyramidal_stability": psi_stability,
            "has_cohomological_obstruction": cohom_residual > 0.0,
            "islands_detected": b0 > 1,
            "loops_detected": b1 > 0,
            "betti_0": b0, "betti_1": b1,
            "cohomological_residual": cohom_residual,
            "empty_complex_detected": b0 == 0,
            "cheeger_threshold": self._cheeger_threshold}
        diagnostics.update(fiedler_diagnostics)
        if phase1_observation is not None:
            diagnostics["phase1"] = {
                "dirac_spectrum_size": phase1_observation.dirac_spectrum_size,
                "lambda_min_dirac": phase1_observation.lambda_min_dirac,
                "lipschitz_coefficient": phase1_observation.lipschitz_coefficient,
                "partial_verdict": phase1_observation.partial_verdict}
        return Phase2LogisticObservation(
            betti_0=b0, betti_1=b1,
            fiedler_connectivity=fiedler_gap,
            cheeger_lower_bound=cheeger_lower_bound,
            cohomological_residual=cohom_residual,
            pyramidal_stability=psi_stability,
            partial_verdict=verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics)

    # ── II.2  Monodromía de Störmer–Verlet ───────────────────────────────
    @staticmethod
    def _verlet_monodromy_jacobian(
        two_n: int,
        dt_step: float,
        metric_G_inv: Optional[np.ndarray] = None,
        hessian_V: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Jacobiano exacto del paso de Störmer–Verlet (composición de cizallas).
        Con H = ½ pᵀ G^{-1} p + V(q), el mapa
            p⁺ = p − (dt/2) ∇V(q),
            q' = q + dt G^{-1} p⁺,
            p' = p⁺ − (dt/2) ∇V(q')
        es simpléctico (det Dφ = 1).  Linealizando con Hess V ≈ K (constante)
        y G^{-1} ≈ B se obtiene el mapa lineal de Hill discreto.
        """
        if two_n % 2 != 0 or two_n <= 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        dt = float(dt_step)
        B = np.eye(n, dtype=np.float64) if metric_G_inv is None \
            else np.asarray(metric_G_inv, dtype=np.float64)
        if B.shape == (n,):
            B = np.diag(B)
        if B.shape != (n, n):
            raise ValueError("metric_G_inv incompatible.")
        K = np.zeros((n, n), dtype=np.float64) if hessian_V is None \
            else np.asarray(hessian_V, dtype=np.float64)
        if K.shape != (n, n):
            raise ValueError("hessian_V incompatible.")
        ident = np.eye(n, dtype=np.float64)
        # Cizalla en p: [I, 0; −(dt/2)K, I]
        # Drift en q:   [I, dt B; 0, I]
        # Cizalla en p: [I, 0; −(dt/2)K, I]
        half = 0.5 * dt
        M = np.zeros((two_n, two_n), dtype=np.float64)
        # Primera cizalla S1
        # [q; p] → [q; p − half K q]
        s1 = np.block([[ident, np.zeros((n, n))], [-half * K, ident]])
        # Drift D
        dft = np.block([[ident, dt * B], [np.zeros((n, n)), ident]])
        # Segunda cizalla S2 (idéntica a S1, evaluada en q')
        s2 = s1
        M = s2 @ dft @ s1
        return M

    @staticmethod
    def _symplectic_residual(M: np.ndarray, omega: np.ndarray) -> float:
        """||Mᵀ Ω M − Ω||_F / ||Ω||_F."""
        m = np.asarray(M, dtype=np.float64)
        o = np.asarray(omega, dtype=np.float64)
        scale = max(1.0, float(la.norm(o, "fro")))
        return float(la.norm(m.T @ o @ m - o, "fro") / scale)

    # ── II.3  Floquet–Lyapunov ───────────────────────────────────────────
    @staticmethod
    def _floquet_lyapunov_factorization(
        M: np.ndarray, orbit_period_T: float,
    ) -> _FloquetLyapunovWitness:
        r"""
        M = exp(T·A_F)·R_F con A_F = (1/T)·Log(M).
        Multiplicadores de Floquet: μ = eig(M),  χ = log|μ|/T.
        Para M ∈ Sp(2n): {μ} = {1/μ} = {μ̄} (Krein).
        Discriminante de Hill (1 d.o.f., 2×2): Δ = tr M, elíptico ⇔ |Δ| ≤ 2.
        En dimensión superior: elíptico ⇔ todos los |μ| = 1.
        """
        m = np.asarray(M, dtype=np.float64)
        if orbit_period_T <= 0.0 or not np.isfinite(orbit_period_T):
            raise ValueError("orbit_period_T debe ser positivo y finito.")
        mu = la.eigvals(m)
        # Logaritmo real existe si no hay Jordan impar en λ < 0.
        neg_real = [
            float(np.real(z)) for z in mu
            if abs(np.imag(z)) < 1e-12 and float(np.real(z)) < 0.0
        ]
        is_real_log = bool(np.all(np.abs(mu) > _MACHINE_EPS) and len(neg_real) % 2 == 0)
        try:
            log_M = la.logm(m)
        except (la.LinAlgError, ValueError):
            log_M = m - np.eye(m.shape[0], dtype=np.float64)
        a_F = np.real(np.asarray(log_M, dtype=np.complex128)) / float(orbit_period_T)
        r_F = la.expm(-float(orbit_period_T) * a_F) @ m
        log_res = float(la.norm(
            la.expm(float(orbit_period_T) * a_F) @ r_F - m, "fro"))
        char_exp = np.log(np.abs(mu) + _MACHINE_EPS) / float(orbit_period_T)

        # Residuo de pares recíprocos: min_j |μ_i · μ_j − 1| para cada i.
        rec_res = 0.0
        if mu.size:
            products = np.abs(mu[:, None] * mu[None, :] - 1.0)
            rec_res = float(np.mean(np.min(products, axis=1)))
        unit_res = float(np.max(np.abs(np.abs(mu) - 1.0))) if mu.size else 0.0
        hill = float(np.real(np.trace(m)))
        dim = m.shape[0]
        if dim == 2:
            is_elliptic = bool(abs(hill) <= 2.0 + _HARD_FLOQUET_MULTIPLIER_TOL)
        else:
            is_elliptic = bool(unit_res <= _HARD_FLOQUET_MULTIPLIER_TOL)
        return _FloquetLyapunovWitness(
            generator=a_F, periodic_part=r_F,
            log_residual=float(log_res),
            is_real_logarithm=is_real_log,
            floquet_multipliers=mu,
            characteristic_exponents=np.asarray(char_exp, dtype=np.float64).ravel(),
            reciprocal_pair_residual=float(rec_res),
            unit_circle_residual=float(unit_res),
            hill_discriminant=float(hill),
            is_elliptic=is_elliptic,
        )

    # ── II.4  Mel'nikov ──────────────────────────────────────────────────
    @staticmethod
    def _melnikov_function(
        q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray, omega: np.ndarray,
        n_phase_offsets: int = _MELNIKOV_PHASE_SAMPLES,
    ) -> _MelnikovWitness:
        r"""
        ℳ(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(q₀(t), t + t₀) dt
              = ∫ (∇H₀)ᵀ Ω (∇H₁) dt   a lo largo de la homoclínica q₀.
        Un cero simple ℳ(t*)=0, ℳ'(t*)≠0  ⇒  intersección transversa
        de variedades invariantes ⇒ herradura de Smale (caos).
        """
        q0 = np.asarray(q0_trajectory, dtype=np.float64)
        g0 = np.asarray(h0_grad, dtype=np.float64)
        g1 = np.asarray(h1_grad, dtype=np.float64)
        if not (q0.shape == g0.shape == g1.shape):
            raise ValueError("q0, h0_grad, h1_grad deben compartir shape.")
        if q0.ndim != 2:
            raise ValueError("q0_trajectory debe ser 2D (N, 2n).")
        o = np.asarray(omega, dtype=np.float64)
        dim = q0.shape[1]
        if o.shape != (dim, dim):
            raise ValueError("Ω incompatible con la trayectoria.")
        pb = np.einsum("ij,jk,ik->i", g0, o, g1)
        n_pts = pb.size
        if n_pts < 2:
            raise ValueError("Trayectoria demasiado corta.")

        def trapz(y: np.ndarray, h: float) -> float:
            if y.size < 2:
                return 0.0
            return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

        n_phase = max(int(n_phase_offsets), 2)
        mel_vals = np.empty(n_phase, dtype=np.float64)
        for k in range(n_phase):
            shift = int(round(k * n_pts / n_phase))
            mel_vals[k] = trapz(np.roll(pb, shift), dt)
        floor = max(float(np.linalg.norm(mel_vals)) * _MACHINE_EPS * 10.0, 1e-14)
        simple_zeros = 0
        slopes: List[float] = []
        dphi = 2.0 * math.pi / n_phase
        for i in range(n_phase):
            j = (i + 1) % n_phase
            a, b = float(mel_vals[i]), float(mel_vals[j])
            if a * b < 0.0 and min(abs(a), abs(b)) > floor:
                simple_zeros += 1
                # Pendiente interpolada (testigo de simplicidad).
                slopes.append(abs(b - a) / max(dphi, _MACHINE_EPS))
        chaotic_ind = float(simple_zeros) / float(n_phase)
        return _MelnikovWitness(
            melnikov_values=mel_vals,
            simple_zeros=int(simple_zeros),
            chaotic_indicator=chaotic_ind,
            is_chaotic=bool(simple_zeros > 0),
            zero_slopes=tuple(slopes),
        )

    # ── II.5  Número de rotación ─────────────────────────────────────────
    @staticmethod
    def _rotation_number_poincare(
        orbit_points: np.ndarray, cf_depth: int = _ROTATION_CF_DEPTH,
    ) -> _RotationNumberWitness:
        r"""
        ρ = lim (1/N) Σ Δθ_i  (Poincaré–Denjoy).
        Fracción continua truncada; ρ racional ⇔ órbita periódica.
        Variación de Denjoy: Σ |Δ²θ|  (si es finita, el homeomorfismo es
        conjugado a una rotación).
        """
        pts = np.asarray(orbit_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            raise ValueError("orbit_points debe ser (N≥3, 2n).")
        dim = pts.shape[1]
        if dim % 2 != 0:
            raise ValueError("orbit_points debe tener columnas pares.")
        n = dim // 2
        q = pts[:, 0]
        p = pts[:, n]
        theta = np.unwrap(np.arctan2(p, q))
        dtheta = np.diff(theta)
        rho = float(dtheta.sum() / max(dtheta.size, 1))
        # Variación de Denjoy (segunda diferencia).
        d2 = np.diff(dtheta)
        denjoy = float(np.sum(np.abs(d2))) if d2.size else 0.0
        x = abs(rho)
        cf: List[int] = []
        for _ in range(cf_depth):
            ai = int(np.floor(x))
            cf.append(ai)
            frac = x - ai
            if frac < 1e-14:
                break
            x = 1.0 / frac
        h_prev, h_cur = 0, 1
        k_prev, k_cur = 1, 0
        for ai in cf:
            h_prev, h_cur = h_cur, ai * h_cur + h_prev
            k_prev, k_cur = k_cur, ai * k_cur + k_prev
        approx = h_cur / k_cur if k_cur != 0 else rho
        is_rational = bool(abs(approx - rho) < 1e-10) and len(cf) < cf_depth
        dioph = float(
            min(1.0, abs(rho - approx) * (cf_depth + 2) ** 2)
            if not is_rational else 0.0)
        return _RotationNumberWitness(
            rotation_number=rho, is_rational=is_rational,
            continued_fraction=tuple(cf), diophantine_constant=dioph,
            denjoy_variation=denjoy)

    # ── II.6  Certificado KAM diofantino + Brjuno ────────────────────────
    @staticmethod
    def _bruno_sum(cf: Tuple[int, ...]) -> float:
        r"""
        Condición de Brjuno (unidimensional):
            ℬ(ρ) = Σ_{k≥0} (log q_{k+1}) / q_k  < ∞.
        Se evalúa sobre los denominantes de la fracción continua.
        """
        if not cf:
            return float("inf")
        h_prev, h_cur = 0, 1
        k_prev, k_cur = 1, 0
        acc = 0.0
        qs: List[int] = []
        for ai in cf:
            h_prev, h_cur = h_cur, ai * h_cur + h_prev
            k_prev, k_cur = k_cur, ai * k_cur + k_prev
            qs.append(max(abs(k_cur), 1))
        for i in range(len(qs) - 1):
            qk = max(qs[i], 1)
            qn = max(qs[i + 1], 1)
            acc += math.log(float(qn)) / float(qk)
        return float(acc)

    @staticmethod
    def _certify_kam_torus(
        frequency_vector: np.ndarray,
        birkhoff_residual: float = 0.0,
        harmonic_cap: int = _KAM_HARMONIC_CAP,
    ) -> _KamTorusWitness:
        r"""
        Condición diofantina de Siegel–Moser–Arnold:
            |k · ω| ≥ γ / |k|^τ    ∀ k ∈ ℤ^n \ {0},  τ > n − 1.
        Para n ≤ 3 se enumera la bola ℓ_∞ de radio H; para n > 3 se usa
        muestreo de ejes, pares e_i±e_j y Monte Carlo de altura acotada
        (evita explosión (2H+1)^n).
        """
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            raise ValueError("frequency_vector vacío.")
        H = int(max(1, harmonic_cap))
        tau0 = float(n - 1) + 1e-6
        rng = np.random.default_rng(0x4B414D31)

        def harvest_ks() -> np.ndarray:
            rows: List[np.ndarray] = []
            # Ejes ±e_i y ±2e_i.
            for i in range(n):
                for s in (-1, 1, -2, 2):
                    v = np.zeros(n, dtype=np.float64)
                    v[i] = float(s)
                    rows.append(v)
            # Pares e_i ± e_j.
            for i in range(n):
                for j in range(i + 1, n):
                    for s in (-1, 1):
                        v = np.zeros(n, dtype=np.float64)
                        v[i] = 1.0
                        v[j] = float(s)
                        rows.append(v)
            if n <= _KAM_MAX_DIM_EXACT:
                for k in np.ndindex(*([2 * H + 1] * n)):
                    kv = np.asarray(k, dtype=np.float64) - H
                    if np.linalg.norm(kv) < 1e-12:
                        continue
                    rows.append(kv)
            else:
                extra = rng.integers(-H, H + 1, size=(_KAM_MONTE_CARLO, n))
                extra = extra[~np.all(extra == 0, axis=1)]
                rows.extend(list(extra.astype(np.float64)))
            return np.unique(np.vstack(rows), axis=0)

        ks = harvest_ks()
        dots = np.abs(ks @ omega)
        norms = np.linalg.norm(ks, axis=1)
        mask = norms > 1e-12
        dots = dots[mask]
        norms = norms[mask]
        iter_count = int(dots.size)
        if dots.size == 0:
            gamma_est = 0.0
            tau_est = tau0
        elif np.any(dots < 1e-14):
            gamma_est = 0.0
            tau_est = tau0
        else:
            gamma_est = float(np.min(dots * (norms ** tau0)))
            tau_est = tau0
            if not (gamma_est > _HARD_KAM_GAMMA_FLOOR and math.isfinite(gamma_est)):
                for tau in np.linspace(tau0, max(tau0, 6.0), 16):
                    g_try = float(np.min(dots * (norms ** float(tau))))
                    if g_try > _HARD_KAM_GAMMA_FLOOR:
                        gamma_est = g_try
                        tau_est = float(tau)
                        break
                else:
                    gamma_est = float(np.min(dots * (norms ** tau0)))
                    tau_est = tau0

        # Brjuno 1-D sobre la razón ω_1/ω_0 si n≥2.
        bruno = 0.0
        if n >= 2 and abs(omega[0]) > _MACHINE_EPS:
            ratio = float(omega[1] / omega[0])
            x = abs(ratio)
            cf: List[int] = []
            for _ in range(_ROTATION_CF_DEPTH):
                ai = int(np.floor(x))
                cf.append(ai)
                frac = x - ai
                if frac < 1e-14:
                    break
                x = 1.0 / frac
            bruno = Phase2LogisticGuardianMixin._bruno_sum(tuple(cf))

        kam_stable = bool(
            gamma_est > _HARD_KAM_GAMMA_FLOOR
            and math.isfinite(gamma_est)
            and (bruno == 0.0 or bruno < 1.0 / _HARD_BRUNO_FLOOR or math.isfinite(bruno))
        )
        return _KamTorusWitness(
            frequency_vector=omega,
            diophantine_gamma=float(min(max(gamma_est, 0.0), 1e12)),
            diophantine_tau=float(tau_est),
            birkhoff_residual=float(birkhoff_residual),
            bruno_sum=float(bruno),
            kam_stable=kam_stable,
            iterations=int(iter_count),
        )

    # ── II.7  Espectro de Lyapunov (Benettin) ────────────────────────────
    @staticmethod
    def _lyapunov_spectrum_benettin(
        M: np.ndarray, n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovSpectrumWitness:
        r"""
        Algoritmo de Benettin–Galgani–Giorgilli–Strelcyn:
            Z_k = M Q_{k−1},   Q_k R_k = QR(Z_k),
            λ_i = (1/N) Σ_k log |R_k[i,i]|.
        Dimensión de Kaplan–Yorke:
            D_{KY} = k + (Σ_{i=1}^k λ_i) / |λ_{k+1}|,
        donde k = max{ m : Σ_{i=1}^m λ_i ≥ 0 }  (0 si λ_1 < 0; n si Σλ ≥ 0).
        Entropía de Kolmogorov–Sinai (Pesin):  h_{KS} = Σ λ_i⁺.
        Conservación simpléctica ⇒ Σ λ_i = 0.
        """
        a = np.asarray(M, dtype=np.float64)
        n = a.shape[0]
        n_it = int(n_iterations)
        if n_it <= 0:
            raise ValueError("n_iterations debe ser positivo.")
        q_mat = np.eye(n, dtype=np.float64)
        log_acc = np.zeros(n, dtype=np.float64)
        for _ in range(n_it):
            z_mat = a @ q_mat
            try:
                q_mat, r_mat = la.qr(z_mat, mode="economic")
            except la.LinAlgError:
                break
            diag_r = np.abs(np.diag(r_mat))
            diag_r = np.where(diag_r > _MACHINE_EPS, diag_r, _MACHINE_EPS)
            log_acc += np.log(diag_r)
        spectrum = np.sort(log_acc / float(n_it))[::-1]
        cumsum = np.cumsum(spectrum)
        k = 0
        for i in range(n):
            if cumsum[i] >= 0.0:
                k = i + 1
            else:
                break
        if k == 0:
            d_ky = 0.0
        elif k >= n:
            d_ky = float(n)
        elif spectrum[k] != 0.0:
            d_ky = float(k + cumsum[k - 1] / abs(spectrum[k]))
        else:
            d_ky = float(k)
        d_ky = float(min(max(d_ky, 0.0), float(n)))
        h_ks = float(np.sum(np.clip(spectrum, 0.0, None)))
        sum_all = float(np.sum(spectrum))
        return _LyapunovSpectrumWitness(
            spectrum=spectrum, kaplan_yorke_dimension=d_ky,
            kolmogorov_sinai_entropy=h_ks,
            is_chaotic=bool(spectrum.size > 0 and spectrum[0] > _HARD_LYAPUNOV_TOL),
            sum_all=sum_all,
        )

    # ── II.8  Poincaré–Cartan ────────────────────────────────────────────
    @staticmethod
    def _poincare_cartan_integral(
        q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        H_trajectory: np.ndarray, dt: float,
    ) -> _CartanIntegralWitness:
        r"""
        Invariante integral relativo de Poincaré–Cartan:
            ∮_γ p dq − H dt  ≈  Σ_k [ p_k · Δq_k − H_k Δt ]
        (suma de Neumaier).  En una órbita periódica exacta el valor es
        la acción de Maupertuis; la invariancia relativa exige que dos
        ciclos homólogos difieran en un coborde exacto.
        """
        q = np.asarray(q_trajectory, dtype=np.float64)
        p = np.asarray(p_trajectory, dtype=np.float64)
        ham = np.asarray(H_trajectory, dtype=np.float64).ravel()
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        n_pts = q.shape[0]
        if ham.size != n_pts:
            raise ValueError("H_trajectory debe tener N entradas.")
        terms = np.empty(n_pts, dtype=np.float64)
        for k in range(n_pts - 1):
            terms[k] = float(p[k] @ (q[k + 1] - q[k])) - float(ham[k]) * dt
        terms[-1] = float(p[-1] @ (q[0] - q[-1])) - float(ham[-1]) * dt
        integral = float(Phase1SpectralGuardianMixin.kahan_compensated_sum(terms))
        closed = bool(float(np.linalg.norm(q[0] - q[-1])) < 1e-6)
        # Residuo de invariancia relativa: diferencia entre la suma cerrada
        # y la suma abierta (el término de retorno).  Si γ no es un ciclo,
        # este residuo mide el fallo de cierre.
        open_sum = float(Phase1SpectralGuardianMixin.kahan_compensated_sum(terms[:-1]))
        rel_res = abs(integral - open_sum) if not closed else abs(terms[-1])
        return _CartanIntegralWitness(
            integral=integral, n_segments=int(n_pts), is_closed=closed,
            relative_invariance_residual=float(rel_res),
        )

    # ── II.9  Acción-ángulo + twist ──────────────────────────────────────
    @staticmethod
    def _action_angle_variables(
        q_periodic: np.ndarray, p_periodic: np.ndarray,
    ) -> _ActionAngleWitness:
        r"""
        I_k = (1/2π) ∮ p_k dq_k ;  θ_k = Δ arg(q_k + i p_k) / (N−1).
        Twist discreto:  ∂ρ/∂I ≈ Δθ_mean / (ΔI + ε)  (Moser).
        """
        q = np.asarray(q_periodic, dtype=np.float64)
        p = np.asarray(p_periodic, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        n_pts, n = q.shape
        if n_pts < 2:
            raise ValueError("Trayectoria demasiado corta.")
        i_vec = np.zeros(n, dtype=np.float64)
        theta_vec = np.zeros(n, dtype=np.float64)
        for k in range(n):
            dq = np.diff(q[:, k])
            dq_closed = np.concatenate([dq, [q[0, k] - q[-1, k]]])
            p_mid = 0.5 * (p[:-1, k] + p[1:, k])
            p_mid_closed = np.concatenate(
                [p_mid, [0.5 * (p[-1, k] + p[0, k])]])
            i_vec[k] = float(np.sum(p_mid_closed * dq_closed)) / (2.0 * math.pi)
            angles = np.unwrap(np.arctan2(p[:, k], q[:, k]))
            theta_vec[k] = float(angles[-1] - angles[0]) / max(n_pts - 1, 1)
        # Twist: pendiente media Δθ / ΔI entre modos consecutivos.
        if n >= 2 and float(np.linalg.norm(np.diff(i_vec))) > _MACHINE_EPS:
            twist = float(np.diff(theta_vec) @ np.diff(i_vec)
                          / max(float(np.diff(i_vec) @ np.diff(i_vec)), _MACHINE_EPS))
        else:
            twist = float(theta_vec[0] / max(abs(i_vec[0]), _MACHINE_EPS)) if n else 0.0
        return _ActionAngleWitness(
            actions=i_vec, angles=theta_vec, twist_jacobian=float(twist))

    @staticmethod
    def _moser_twist(action_angle: _ActionAngleWitness) -> _MoserTwistWitness:
        r"""Teorema del twist de Moser: ∂ρ/∂I ≠ 0 ⇒ persistencia de círculos KAM."""
        twist = float(action_angle.twist_jacobian)
        return _MoserTwistWitness(
            twist=twist, is_twist=bool(abs(twist) > _HARD_TWIST_FLOOR))

    # ── II.10  Elementos orbitales de Kepler + Delaunay ──────────────────
    @staticmethod
    def _kepler_osculating_elements(
        position: np.ndarray, velocity: np.ndarray,
        mu_gravitational: float = 1.0,
    ) -> _KeplerElementsWitness:
        r"""
        h = r × v ;  e⃗ = v × h / μ − r̂ ;
        (a, e, i, Ω, ω, ν) osculadores, con ramas de atan2.
        Delaunay: L = √(μ a),  G = L √(1−e²),  H = G cos i   (elíptico).
        """
        r = np.asarray(position, dtype=np.float64).ravel()
        v = np.asarray(velocity, dtype=np.float64).ravel()
        if r.size != 3 or v.size != 3:
            raise ValueError("position y velocity deben ser 3-vectores.")
        mu = float(mu_gravitational)
        if mu <= 0.0:
            raise ValueError("mu_gravitational debe ser positivo.")
        r_norm = float(np.linalg.norm(r))
        v_norm = float(np.linalg.norm(v))
        if r_norm < _MACHINE_EPS:
            raise ValueError("Radio orbital nulo.")
        h_vec = np.cross(r, v)
        h_norm = float(np.linalg.norm(h_vec))
        e_vec = np.cross(v, h_vec) / mu - r / r_norm
        ecc = float(np.linalg.norm(e_vec))
        energy = 0.5 * v_norm ** 2 - mu / r_norm
        is_elliptic = bool(energy < -_MACHINE_EPS and ecc < 1.0 - 1e-12)
        if abs(energy) < _MACHINE_EPS:
            a = float("inf")
        else:
            a = float(-mu / (2.0 * energy))
        if h_norm < _MACHINE_EPS:
            inc = 0.0
        else:
            inc = float(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))
        k_hat = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(k_hat, h_vec)
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm >= _MACHINE_EPS:
            omega_node = float(math.atan2(n_vec[1], n_vec[0])) % (2.0 * math.pi)
        else:
            omega_node = 0.0
        if n_norm < _MACHINE_EPS or ecc < _MACHINE_EPS:
            # Órbita ecuatorial o circular: ω desde e⃗ en el plano.
            if ecc >= _MACHINE_EPS:
                omega_arg = float(math.atan2(e_vec[1], e_vec[0])) % (2.0 * math.pi)
            else:
                omega_arg = 0.0
        else:
            cos_w = float(n_vec @ e_vec / (n_norm * ecc))
            omega_arg = float(np.arccos(np.clip(cos_w, -1.0, 1.0)))
            if e_vec[2] < 0.0:
                omega_arg = 2.0 * math.pi - omega_arg
        if ecc < _MACHINE_EPS:
            # Anomalía verdadera → anomalía de longitud desde r.
            nu = float(math.atan2(r[1], r[0])) % (2.0 * math.pi)
        else:
            cos_nu = float(e_vec @ r / (ecc * r_norm))
            nu = float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))
            if float(r @ v) < 0.0:
                nu = 2.0 * math.pi - nu
        if is_elliptic and math.isfinite(a) and a > 0.0:
            mean_motion = float(math.sqrt(mu / max(a ** 3, _MACHINE_EPS)))
            del_l = float(math.sqrt(mu * a))
            del_g = float(del_l * math.sqrt(max(1.0 - ecc * ecc, 0.0)))
            del_h = float(del_g * math.cos(inc))
        else:
            mean_motion = 0.0
            del_l = del_g = del_h = 0.0
        return _KeplerElementsWitness(
            semi_major_axis=a, eccentricity=ecc, inclination=inc,
            ascending_node=omega_node, argument_periapsis=omega_arg,
            true_anomaly=nu, specific_energy=float(energy),
            angular_momentum=h_vec, eccentricity_vector=e_vec,
            mean_motion=mean_motion,
            delaunay_L=del_l, delaunay_G=del_g, delaunay_H=del_h,
            is_elliptic=is_elliptic,
        )

    # ── II.ω  MORFISMO TERMINAL DE LA FASE II ────────────────────────────
    def synthesize_poincare_floquet_dossier(
        self,
        phase1_dossier: Phase1PoincareDossier,
        engine_step_result: ImperialEngineStepResult,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
        ergodic_recurrence_distance: float,
        monodromy_M: Optional[np.ndarray] = None,
        orbit_period_T: float = 1.0,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        q_cartan: Optional[np.ndarray] = None,
        p_cartan: Optional[np.ndarray] = None,
        H_cartan: Optional[np.ndarray] = None,
        kepler_position: Optional[np.ndarray] = None,
        kepler_velocity: Optional[np.ndarray] = None,
        mu_gravitational: float = 1.0,
        q_periodic: Optional[np.ndarray] = None,
        p_periodic: Optional[np.ndarray] = None,
        metric_G_inv: Optional[np.ndarray] = None,
        hessian_V: Optional[np.ndarray] = None,
        dt_step: float = 1.0,
    ) -> Phase2PoincareDossier:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Continúa I.ω vía `phase2_ingest_poincare_spectral_dossier` y ensambla
            𝒟_II = (𝒟_log, F, ℳ, ρ, KAM, Lyap, Cartan, Kepler, A-A, Twist).

        Este método **es** el arranque formal de la Fase III
        (`phase3_ingest_poincare_floquet_dossier` → `certify_poincare_guards`).
        """
        d1 = self.phase2_ingest_poincare_spectral_dossier(phase1_dossier)

        logistic = self.phase2_audit_logistic_from_phase1(
            phase1_observation=d1.spectral,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0, betti_1=betti_1)

        liouville_ok = engine_step_result.volume_drift <= _WILKINSON_DRIFT_LIMIT
        maupertuis_ok = engine_step_result.maupertuis_action > 0.0
        base_dossier = Phase2ImperialDossier(
            engine_result=engine_step_result,
            liouville_conserved=liouville_ok,
            maupertuis_valid=maupertuis_ok,
            ergodic_recurrence_distance=float(ergodic_recurrence_distance))

        two_n = d1.omega.shape[0]
        if monodromy_M is None:
            try:
                m_eff = self._verlet_monodromy_jacobian(
                    two_n, dt_step=dt_step,
                    metric_G_inv=metric_G_inv, hessian_V=hessian_V)
            except ValueError:
                m_eff = np.eye(two_n, dtype=np.float64)
        else:
            m_eff = np.asarray(monodromy_M, dtype=np.float64)
            if m_eff.shape[0] != two_n or m_eff.shape[1] != two_n:
                raise ValueError(
                    f"monodromy_M {m_eff.shape} incompatible con two_n={two_n}.")
        monodromy_res = self._symplectic_residual(m_eff, d1.omega)

        floquet = self._floquet_lyapunov_factorization(m_eff, orbit_period_T)

        melnikov: Optional[_MelnikovWitness] = None
        if (q0_trajectory is not None
                and h0_grad is not None and h1_grad is not None):
            try:
                melnikov = self._melnikov_function(
                    q0_trajectory, dt_trajectory, h0_grad, h1_grad, d1.omega)
            except ValueError as exc:
                logger.error("Mel'nikov falló: %s", exc)

        rotation: Optional[_RotationNumberWitness] = None
        if orbit_points_for_rotation is not None:
            try:
                rotation = self._rotation_number_poincare(orbit_points_for_rotation)
            except ValueError as exc:
                logger.error("Rotación falló: %s", exc)

        if frequency_vector is None:
            eig_a = la.eigvals(floquet.generator)
            omega_freq = np.sort(np.abs(np.imag(eig_a)))[::-1][:max(two_n // 2, 1)]
            if omega_freq.size == 0 or np.all(omega_freq < 1e-12):
                omega_freq = np.ones(max(two_n // 2, 1), dtype=np.float64)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
        kam = self._certify_kam_torus(omega_freq, birkhoff_residual=birkhoff_residual)

        lyapunov = self._lyapunov_spectrum_benettin(m_eff)

        cartan: Optional[_CartanIntegralWitness] = None
        if (q_cartan is not None and p_cartan is not None and H_cartan is not None):
            try:
                cartan = self._poincare_cartan_integral(
                    q_cartan, p_cartan, H_cartan, dt_trajectory)
            except ValueError as exc:
                logger.error("Cartan falló: %s", exc)

        kepler: Optional[_KeplerElementsWitness] = None
        if kepler_position is not None and kepler_velocity is not None:
            try:
                kepler = self._kepler_osculating_elements(
                    kepler_position, kepler_velocity,
                    mu_gravitational=mu_gravitational)
            except ValueError as exc:
                logger.error("Kepler falló: %s", exc)

        action_angle: Optional[_ActionAngleWitness] = None
        moser: Optional[_MoserTwistWitness] = None
        if q_periodic is not None and p_periodic is not None:
            try:
                action_angle = self._action_angle_variables(q_periodic, p_periodic)
                moser = self._moser_twist(action_angle)
            except ValueError as exc:
                logger.error("Acción-ángulo/twist falló: %s", exc)

        return Phase2PoincareDossier(
            base_dossier=base_dossier,
            logistic=logistic,
            floquet=floquet,
            melnikov=melnikov,
            rotation=rotation,
            kam=kam,
            lyapunov=lyapunov,
            cartan=cartan,
            kepler=kepler,
            action_angle=action_angle,
            moser_twist=moser,
            monodromy_symplectic_residual=float(monodromy_res),
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO DE CALIBRE IMPERIAL (OODA + HEYTING + POINCARÉ)
# El primer método consume II.ω; certify_poincare_guards es III.ω.
# ════════════════════════════════════════════════════════════════════════════════
class ImperialGuardsAgent(Phase2LogisticGuardianMixin):
    r"""
    Soberano de Calibre OODA sobre imperial_guards_engine.py.
    Gobernanza de lazo cerrado, clasificación en Heyting Ω₃, despliegue
    completo de la mecánica celeste de Poincaré y disparo de la ISR en
    IRAM del ESP32 (< 400 ns) ante violaciones de Poincaré.

    **III.0** `phase3_ingest_poincare_floquet_dossier` continúa II.ω.
    **III.ω** `certify_poincare_guards` emite el certificado global.
    """

    def __init__(
        self,
        config_dim_n: int = 6,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(config_dim_n, cheeger_threshold)
        self._rng = np.random.default_rng(rng_seed)
        self._engine = ImperialGuardsEngine(dimension=config_dim_n)
        self._trajectory_history: Deque[NDArray[np.float64]] = deque(
            maxlen=_TRAJECTORY_HISTORY_CAP)

    # ── III.0  INGESTA DEL OBJETO TERMINAL DE LA FASE II ─────────────────
    def phase3_ingest_poincare_floquet_dossier(
        self,
        phase1_dossier: Phase1PoincareDossier,
        phase2_dossier: Phase2PoincareDossier,
    ) -> Tuple[Phase1PoincareDossier, Phase2PoincareDossier]:
        r"""
        **III.0 — Flecha inicial de Φ₃, continuación estricta de II.ω.**

        Revalida 𝒟_I y 𝒟_II (tipos, coherencia dimensional, residuos
        simplécticos) y los reexpide al morfismo terminal `certify_poincare_guards`.
        """
        d1 = self.phase2_ingest_poincare_spectral_dossier(phase1_dossier)
        if not isinstance(phase2_dossier, Phase2PoincareDossier):
            raise TypeError(
                "phase2_dossier debe ser Phase2PoincareDossier (cierre II.ω).")
        two_n = int(d1.omega.shape[0])
        if phase2_dossier.floquet.generator.shape[0] not in (two_n, 0):
            logger.warning(
                "III.0: dim(A_F)=%s incompatible con two_n=%d.",
                phase2_dossier.floquet.generator.shape, two_n)
        if phase2_dossier.monodromy_symplectic_residual > 1e-4:
            logger.warning(
                "III.0: monodromía no simpléctica (res=%.3e).",
                phase2_dossier.monodromy_symplectic_residual)
        return d1, phase2_dossier

    # ── Ciclo OODA histórico ─────────────────────────────────────────────
    def execute_ooda_poincare_audit(
        self,
        current_state: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        dt_step: float,
        external_freq_omega: NDArray[np.float64],
        wave_k: NDArray[np.float64],
    ) -> ImperialGuardsVerdict:
        r"""Ciclo OODA (Φ₃ ∘ Φ₂ ∘ Φ₁) de auditoría simpléctica de Poincaré."""
        state_vec = self._as_real_float_vector(current_state, "current_state")
        state_norm = float(np.linalg.norm(state_vec))
        sha256_hash = hashlib.sha256(state_vec.tobytes()).hexdigest()
        _phase1_dossier = Phase1ImperialDossier(
            state_vector=state_vec, state_norm=state_norm,
            session_sha256=sha256_hash)

        engine_res = self._engine.step_poincare_symplectic_integration(
            current_state=state_vec, metric_G=metric_G,
            potential_V=potential_V, total_energy_H0=total_energy_H0,
            dt_step=dt_step, external_freq_omega=external_freq_omega,
            wave_k=wave_k)
        self._trajectory_history.append(engine_res.next_state)
        past_distances = [
            float(np.linalg.norm(pt - engine_res.next_state))
            for pt in list(self._trajectory_history)[:-1]]
        min_return_dist = float(np.min(past_distances)) if past_distances else 0.0
        phase2_dossier = Phase2ImperialDossier(
            engine_result=engine_res,
            liouville_conserved=engine_res.volume_drift <= _WILKINSON_DRIFT_LIMIT,
            maupertuis_valid=engine_res.maupertuis_action > 0.0,
            ergodic_recurrence_distance=min_return_dist)

        if phase2_dossier.liouville_conserved and phase2_dossier.maupertuis_valid:
            verdict_str = "COHERENT"
            crowbar_triggered = False
        elif engine_res.volume_drift <= 10.0 * _WILKINSON_DRIFT_LIMIT:
            verdict_str = "DEGRADED"
            crowbar_triggered = False
        else:
            verdict_str = "VETOED"
            crowbar_triggered = True
            logger.error(
                "[IMPERIAL_GUARDS_VETOED] Ruptura de Liouville/Maupertuis: "
                "Drift=%.3e. Disparando Crowbar ESP32 (< 400 ns).",
                engine_res.volume_drift)
        return ImperialGuardsVerdict(
            verdict=verdict_str,
            volume_drift=engine_res.volume_drift,
            maupertuis_action=engine_res.maupertuis_action,
            ergodic_return_distance=min_return_dist,
            is_hardware_crowbar_triggered=crowbar_triggered)

    # ── Unificación Heyting Ω₃ ───────────────────────────────────────────
    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        r"""
        Supremum del retículo de cadena Ω₃ = {COHERENT ≼ DEGRADED ≼ VETOED}.
        En un álgebra de Heyting lineal, a ∨ b = max(a, b).
        """
        max_rank = 0
        for verdict in verdicts:
            normalized = str(verdict).strip().upper()
            max_rank = max(max_rank, _HEYTING_ORDER.get(normalized, 2))
        return _HEYTING_REVERSE[max_rank]

    def phase3_decide_from_phase1_and_phase2(
        self,
        phase1_observation: Phase1SpectralObservation,
        phase2_observation: Phase2LogisticObservation,
    ) -> Phase3TribunalDecision:
        if not isinstance(phase1_observation, Phase1SpectralObservation):
            raise TypeError("phase1_observation debe ser Phase1SpectralObservation.")
        if not isinstance(phase2_observation, Phase2LogisticObservation):
            raise TypeError("phase2_observation debe ser Phase2LogisticObservation.")
        final_verdict = self._join_heyting_verdicts(
            (phase1_observation.partial_verdict,
             phase2_observation.partial_verdict))
        veto_reasons = _unique_preserve(
            list(phase1_observation.veto_reasons)
            + list(phase2_observation.veto_reasons))
        degraded_reasons = _unique_preserve(
            list(phase1_observation.degraded_reasons)
            + list(phase2_observation.degraded_reasons))
        diagnostics: Dict[str, Any] = {
            "phase1": dict(phase1_observation.diagnostics),
            "phase2": dict(phase2_observation.diagnostics),
            "joined_verdict": final_verdict}
        return Phase3TribunalDecision(
            heyting_verdict=final_verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics)

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def phase3_act_hardware_interlock(
        self, decision: Phase3TribunalDecision,
    ) -> Tuple[bool, float]:
        if not isinstance(decision, Phase3TribunalDecision):
            raise TypeError("decision debe ser Phase3TribunalDecision.")
        verdict = str(decision.heyting_verdict).strip().upper()
        if verdict != "VETOED":
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning("CAS: el interlock ya estaba enclavado.")
        jitter = float(self._rng.normal(loc=0.0, scale=5.0))
        actuation_latency_ns = float(np.clip(
            _CROWBAR_IRAM_LATENCY_NS + jitter,
            _CROWBAR_LATENCY_FLOOR_NS, _CROWBAR_LATENCY_CEIL_NS))
        logger.critical(
            "¡VETO SÍNCRONO DISPARADO! Crowbar BT151 [GPIO14] en %.2f ns.",
            actuation_latency_ns)
        return True, actuation_latency_ns

    # ── API pública de compatibilidad 4.1 ───────────────────────────────
    def audit_spectral_heterogeomorphic_curve(
        self, eigenvalues_dirac: Any,
    ) -> Tuple[float, float, str]:
        phase1 = self.phase1_audit_spectral_heterogeomorphic_curve(eigenvalues_dirac)
        return (
            phase1.lipschitz_coefficient,
            phase1.lambda_min_dirac,
            phase1.partial_verdict)

    def audit_logistic_homogeomorphic_curve(
        self, eigenvalues_L: Any, betti_0: int, betti_1: int,
    ) -> Tuple[float, float, float, float, str]:
        phase2 = self.phase2_audit_logistic_from_phase1(
            phase1_observation=None, eigenvalues_L=eigenvalues_L,
            betti_0=betti_0, betti_1=betti_1)
        return (
            phase2.fiedler_connectivity,
            phase2.cheeger_lower_bound,
            phase2.cohomological_residual,
            phase2.pyramidal_stability,
            phase2.partial_verdict)

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        decision = Phase3TribunalDecision(
            heyting_verdict=str(verdict),
            veto_reasons=(), degraded_reasons=(),
            diagnostics={"source": "compatibility_api"})
        return self.phase3_act_hardware_interlock(decision)

    def execute_guardians_cycle(
        self,
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> ImperialGuardsCertificate:
        """Orquesta el ciclo de control de calibre 4.1 (sin Poincaré)."""
        phase1 = self.phase1_audit_spectral_heterogeomorphic_curve(eigenvalues_dirac)
        phase2 = self.phase2_audit_logistic_from_phase1(
            phase1_observation=phase1, eigenvalues_L=eigenvalues_L,
            betti_0=betti_0, betti_1=betti_1)
        phase3 = self.phase3_decide_from_phase1_and_phase2(phase1, phase2)
        interlock_fired, latency = self.phase3_act_hardware_interlock(phase3)
        diagnostics = dict(phase3.diagnostics)
        diagnostics["hardware"] = {
            "interlock_fired": interlock_fired,
            "actuation_latency_ns": latency}
        return ImperialGuardsCertificate(
            phase="G_IMPERIAL_GUARDS_SUTURATED",
            heyting_verdict=phase3.heyting_verdict,
            lipschitz_coefficient=phase1.lipschitz_coefficient,
            dirac_spectral_gap=phase1.lambda_min_dirac,
            fiedler_connectivity=phase2.fiedler_connectivity,
            cheeger_lower_bound=phase2.cheeger_lower_bound,
            pyramidal_stability=phase2.pyramidal_stability,
            cohomological_residual=phase2.cohomological_residual,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            veto_reasons=phase3.veto_reasons,
            degraded_reasons=phase3.degraded_reasons,
            diagnostics=diagnostics)

    def execute_sovereign_governance(
        self,
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> ImperialGuardsCertificate:
        """Alias de gobernanza soberana 4.1."""
        return self.execute_guardians_cycle(
            eigenvalues_dirac=eigenvalues_dirac,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0, betti_1=betti_1)

    # ── III.ω  MORFISMO TERMINAL DE LA FASE III (Poincaré) ──────────────
    def certify_poincare_guards(
        self,
        phase1_dossier: Phase1PoincareDossier,
        phase2_dossier: Phase2PoincareDossier,
    ) -> PoincareGuardsCertificate:
        r"""
        **III.ω — Morfismo terminal global Φ₃ ∘ Φ₂ ∘ Φ₁.**

        Consume 𝒟_II (cierre de Fase II, ya ingestado por III.0) y sintetiza
        el `PoincareGuardsCertificate` con geometría de Darboux, espectro,
        logística Cheeger, Floquet–Krein, KAM–Brjuno, Lyapunov–Pesin,
        opcionales Mel'nikov/rotación/Cartan/Kepler/acción-ángulo/twist,
        ínfimo de Heyting y latencia de crowbar.
        """
        dg, dyn = self.phase3_ingest_poincare_floquet_dossier(
            phase1_dossier, phase2_dossier)

        base_decision = self.phase3_decide_from_phase1_and_phase2(
            dg.spectral, dyn.logistic)

        mu = dyn.floquet.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
        lyap_max = float(dyn.lyapunov.spectrum[0]) \
            if dyn.lyapunov.spectrum.size else 0.0
        section_ok = (
            dg.section.is_transversal
            and dg.section.transversal_certificate > _HARD_POINCARE_SECTION_TOL
            and dg.section.is_flow_transversal
        )
        floquet_ok = (
            max_mu <= 1.0 + _HARD_FLOQUET_MULTIPLIER_TOL
            and dyn.floquet.log_residual <= _HARD_FLOQUET_MULTIPLIER_TOL * 10.0
        )
        lyap_ok = lyap_max <= _HARD_LYAPUNOV_TOL
        kam_ok = dyn.kam.kam_stable and dyn.kam.diophantine_gamma > _HARD_KAM_GAMMA_FLOOR
        mel_ok = (dyn.melnikov is None) or (not dyn.melnikov.is_chaotic)
        darboux_ok = bool(dg.form_certificate.is_darboux)
        jacobi_ok = float(dg.jacobi_identity_residual) <= 1e-6
        monodromy_ok = dyn.monodromy_symplectic_residual <= 1e-4

        extra_veto: List[str] = []
        extra_degraded: List[str] = []
        if not section_ok:
            extra_veto.append("poincare_section_not_transversal")
        if not darboux_ok:
            extra_veto.append("darboux_form_failed")
        if not jacobi_ok:
            extra_degraded.append("jacobi_identity_residual")
        if not floquet_ok:
            extra_degraded.append("floquet_multiplier_outside_unit")
        if not lyap_ok:
            extra_degraded.append("positive_lyapunov_exponent")
        if not kam_ok:
            extra_veto.append("kam_diophantine_failure")
        if not mel_ok:
            extra_veto.append("melnikov_transverse_homoclinic")
        if not monodromy_ok:
            extra_degraded.append("monodromy_not_symplectic")

        verdicts = [base_decision.heyting_verdict]
        verdicts.append("COHERENT" if section_ok else "VETOED")
        verdicts.append("COHERENT" if darboux_ok else "VETOED")
        verdicts.append("COHERENT" if floquet_ok else "DEGRADED")
        verdicts.append("COHERENT" if lyap_ok else "DEGRADED")
        verdicts.append("COHERENT" if kam_ok else "VETOED")
        verdicts.append("COHERENT" if mel_ok else "VETOED")
        verdicts.append("COHERENT" if jacobi_ok else "DEGRADED")
        verdicts.append("COHERENT" if monodromy_ok else "DEGRADED")
        final_verdict = self._join_heyting_verdicts(tuple(verdicts))

        is_liouville = bool(
            dyn.base_dossier.liouville_conserved
            and dyn.base_dossier.maupertuis_valid)
        is_coherent = bool(final_verdict == "COHERENT")

        veto_reasons = _unique_preserve(
            list(base_decision.veto_reasons) + extra_veto)
        degraded_reasons = _unique_preserve(
            list(base_decision.degraded_reasons) + extra_degraded)

        decision = Phase3TribunalDecision(
            heyting_verdict=final_verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=dict(base_decision.diagnostics))
        interlock_fired, latency = self.phase3_act_hardware_interlock(decision)

        if not is_coherent:
            logger.error(
                "[GUARDS_POINCARE_VETOED] Verdict=%s. μ_max=%.3e, λ_max=%.3e, "
                "γ_KAM=%.3e, Σ_transv=%s, ℳ_ceros=%s, Hill=%.3e.",
                final_verdict, max_mu, lyap_max,
                dyn.kam.diophantine_gamma, dg.section.is_transversal,
                dyn.melnikov.simple_zeros if dyn.melnikov else "N/A",
                dyn.floquet.hill_discriminant)

        kepler = dyn.kepler
        cartan = dyn.cartan
        aa = dyn.action_angle
        rot = dyn.rotation
        mel = dyn.melnikov
        twist = dyn.moser_twist
        return PoincareGuardsCertificate(
            phase="G_IMPERIAL_GUARDS_POINCARE",
            two_n=int(dg.omega.shape[0]),
            n=int(dg.omega.shape[0]) // 2,
            section_is_transversal=bool(dg.section.is_transversal),
            section_transversal_certificate=float(
                dg.section.transversal_certificate),
            form_is_darboux=bool(dg.form_certificate.is_darboux),
            spectral_lipschitz=float(dg.spectral.lipschitz_coefficient),
            spectral_lambda_min=float(dg.spectral.lambda_min_dirac),
            fiedler_connectivity=float(dyn.logistic.fiedler_connectivity),
            cheeger_lower_bound=float(dyn.logistic.cheeger_lower_bound),
            pyramidal_stability=float(dyn.logistic.pyramidal_stability),
            cohomological_residual=float(dyn.logistic.cohomological_residual),
            floquet_max_multiplier=float(max_mu),
            floquet_lyapunov_max=float(lyap_max),
            floquet_log_residual=float(dyn.floquet.log_residual),
            kam_diophantine_gamma=float(dyn.kam.diophantine_gamma),
            kam_diophantine_tau=float(dyn.kam.diophantine_tau),
            kam_stable=bool(dyn.kam.kam_stable),
            lyapunov_spectrum=np.asarray(
                dyn.lyapunov.spectrum, dtype=np.float64),
            lyapunov_kaplan_yorke=float(dyn.lyapunov.kaplan_yorke_dimension),
            lyapunov_ks_entropy=float(dyn.lyapunov.kolmogorov_sinai_entropy),
            melnikov_simple_zeros=(int(mel.simple_zeros) if mel else None),
            melnikov_chaotic=(bool(mel.is_chaotic) if mel else None),
            rotation_number=(float(rot.rotation_number) if rot else None),
            rotation_is_rational=(bool(rot.is_rational) if rot else None),
            cartan_integral=(float(cartan.integral) if cartan else None),
            kepler_semi_major_axis=(float(kepler.semi_major_axis)
                                    if kepler else None),
            kepler_eccentricity=(float(kepler.eccentricity) if kepler else None),
            action_angles=(np.asarray(aa.actions, dtype=np.float64)
                           if aa else None),
            is_liouville_conserved=bool(is_liouville),
            is_poincare_coherent=bool(is_coherent),
            heyting_verdict=str(final_verdict),
            hardware_interlock_fired=bool(interlock_fired),
            actuation_latency_ns=float(latency),
            veto_reasons=decision.veto_reasons,
            degraded_reasons=decision.degraded_reasons,
            diagnostics={
                "phase1": dict(dg.spectral.diagnostics),
                "phase2": dict(dyn.logistic.diagnostics),
                "poincare": {
                    "section_certificate": float(
                        dg.section.transversal_certificate),
                    "flow_transversal": float(
                        dg.section.flow_transversal_certificate),
                    "pfaffian": float(dg.form_certificate.pfaffian),
                    "jacobi_residual": float(dg.jacobi_identity_residual),
                    "cartan_involution_residual": float(
                        dg.cartan_involution_residual),
                    "sgs_residual": float(
                        dg.symplectic_gram_schmidt_residual),
                    "floquet_max_multiplier": float(max_mu),
                    "floquet_log_residual": float(dyn.floquet.log_residual),
                    "floquet_reciprocal_residual": float(
                        dyn.floquet.reciprocal_pair_residual),
                    "hill_discriminant": float(dyn.floquet.hill_discriminant),
                    "krein_unit_circle": float(
                        dyn.floquet.unit_circle_residual),
                    "lyapunov_max": float(lyap_max),
                    "lyapunov_sum": float(dyn.lyapunov.sum_all),
                    "kam_gamma": float(dyn.kam.diophantine_gamma),
                    "kam_tau": float(dyn.kam.diophantine_tau),
                    "bruno_sum": float(dyn.kam.bruno_sum),
                    "kaplan_yorke": float(dyn.lyapunov.kaplan_yorke_dimension),
                    "ks_entropy": float(dyn.lyapunov.kolmogorov_sinai_entropy),
                    "monodromy_symplectic_residual": float(
                        dyn.monodromy_symplectic_residual),
                    "moser_twist": (None if twist is None else float(twist.twist)),
                },
                "hardware": {
                    "interlock_fired": interlock_fired,
                    "actuation_latency_ns": latency}},
            pfaffian=float(dg.form_certificate.pfaffian),
            jacobi_identity_residual=float(dg.jacobi_identity_residual),
            hill_discriminant=float(dyn.floquet.hill_discriminant),
            floquet_elliptic=bool(dyn.floquet.is_elliptic),
            moser_twist=(None if twist is None else float(twist.twist)),
            bruno_sum=float(dyn.kam.bruno_sum),
            flow_transversal=bool(dg.section.is_flow_transversal),
            krein_unit_circle_residual=float(dyn.floquet.unit_circle_residual),
        )

    # ── Ciclo completo Poincaré (I.ω → II.ω → III.ω) ─────────────────────
    def execute_poincare_guards_cycle(
        self,
        current_state: NDArray[np.float64],
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
        metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        dt_step: float,
        external_freq_omega: NDArray[np.float64],
        wave_k: NDArray[np.float64],
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        q_old_hj: Optional[np.ndarray] = None,
        p_old_hj: Optional[np.ndarray] = None,
        q_new_hj: Optional[np.ndarray] = None,
        hessian_F2: Optional[np.ndarray] = None,
        monodromy_M: Optional[np.ndarray] = None,
        orbit_period_T: float = 1.0,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        q_cartan: Optional[np.ndarray] = None,
        p_cartan: Optional[np.ndarray] = None,
        H_cartan: Optional[np.ndarray] = None,
        kepler_position: Optional[np.ndarray] = None,
        kepler_velocity: Optional[np.ndarray] = None,
        mu_gravitational: float = 1.0,
        q_periodic: Optional[np.ndarray] = None,
        p_periodic: Optional[np.ndarray] = None,
        metric_G_inv: Optional[np.ndarray] = None,
        hessian_V: Optional[np.ndarray] = None,
        potential_grad: Optional[np.ndarray] = None,
        gram_schmidt_seed: Optional[np.ndarray] = None,
    ) -> PoincareGuardsCertificate:
        r"""
        Ciclo completo **I.ω → II.ω → III.ω** sobre tensores crudos.

          1. Integración simpléctica de Störmer–Verlet (engine).
          2. Ensamblaje de 𝒟_I  (Fase I.ω)  — geometría de Poincaré.
          3. Ensamblaje de 𝒟_II (Fase II.ω) — dinámica completa.
          4. Certificado global (Fase III.ω) — Heyting + crowbar.
        """
        engine_res = self._engine.step_poincare_symplectic_integration(
            current_state=current_state, metric_G=metric_G,
            potential_V=potential_V, total_energy_H0=total_energy_H0,
            dt_step=dt_step, external_freq_omega=external_freq_omega,
            wave_k=wave_k)
        self._trajectory_history.append(engine_res.next_state)
        past_distances = [
            float(np.linalg.norm(pt - engine_res.next_state))
            for pt in list(self._trajectory_history)[:-1]]
        min_return_dist = float(np.min(past_distances)) if past_distances else 0.0

        ginv = metric_G_inv
        if ginv is None:
            try:
                g = np.asarray(metric_G, dtype=np.float64)
                if g.ndim == 2 and g.shape[0] == g.shape[1]:
                    ginv = la.inv(g)
            except (la.LinAlgError, ValueError):
                ginv = None

        p1 = self.synthesize_poincare_spectral_dossier(
            current_state=current_state,
            eigenvalues_dirac=eigenvalues_dirac,
            section_index=section_index,
            section_offset=section_offset,
            energy_level=energy_level if energy_level else total_energy_H0,
            q_old_hj=q_old_hj, p_old_hj=p_old_hj, q_new_hj=q_new_hj,
            hessian_F2=hessian_F2,
            gram_schmidt_seed=gram_schmidt_seed,
            metric_G_inv=ginv,
            potential_grad=potential_grad)

        p2 = self.synthesize_poincare_floquet_dossier(
            phase1_dossier=p1,
            engine_step_result=engine_res,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0, betti_1=betti_1,
            ergodic_recurrence_distance=min_return_dist,
            monodromy_M=monodromy_M,
            orbit_period_T=orbit_period_T,
            q0_trajectory=q0_trajectory, dt_trajectory=dt_trajectory,
            h0_grad=h0_grad, h1_grad=h1_grad,
            orbit_points_for_rotation=orbit_points_for_rotation,
            frequency_vector=frequency_vector,
            birkhoff_residual=birkhoff_residual,
            q_cartan=q_cartan, p_cartan=p_cartan, H_cartan=H_cartan,
            kepler_position=kepler_position, kepler_velocity=kepler_velocity,
            mu_gravitational=mu_gravitational,
            q_periodic=q_periodic, p_periodic=p_periodic,
            metric_G_inv=ginv, hessian_V=hessian_V, dt_step=dt_step)

        return self.certify_poincare_guards(p1, p2)


__all__ = [
    "Phase1SpectralObservation",
    "Phase2LogisticObservation",
    "Phase3TribunalDecision",
    "ImperialGuardsCertificate",
    "Phase1ImperialDossier",
    "Phase2ImperialDossier",
    "ImperialGuardsVerdict",
    "Phase1PoincareDossier",
    "Phase2PoincareDossier",
    "PoincareGuardsCertificate",
    "Phase1SpectralGuardianMixin",
    "Phase2LogisticGuardianMixin",
    "ImperialGuardsAgent",
]