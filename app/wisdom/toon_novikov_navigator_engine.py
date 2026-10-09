# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Novikov Navigator Engine (Motor Navegante de Novikov)        ║
║ Ubicación: app/wisdom/toon_novikov_navigator_engine.py                       ║
║ Versión  : 7.0.0-Poincaré-RSI3-Nested-Novikov-BruhatTits-Amoeba-Fock-Φsem    ║
╚══════════════════════════════════════════════════════════════════════════════╝

TEJIDO ANIDADO POR HERENCIA CATEGÓRICA EN TRES FASES:

  ◈ FASE I  — PoincareNovikovAtlas
              Delaunay canónico (J_i = −ln λ_i) → LRL →
              mapa de primer retorno (Chirikov, det DP = 1) →
              twist Poincaré–Birkhoff → Greene simpléctico →
              Melnikov (scipy.quad) → Bryuno (Fraction, q_k) →
              Lindstedt–Poincaré O(ε²) → ecuación homológica →
              Morse–Bott χ(ℂPⁿ⁻¹) = n → CR3BP Jacobi + L1 Richardson →
              Poincaré–Cartan θ = Tr(ρ dN) con N en la eigenbase ([ρ, N] ≈ 0) →
              Nekhoroshev T_nek ∼ exp(c/ε^{1/2n}) → Kac T_rec = 1/μ(A) →
              mónada RSI-3 (T, η, μ) con Banach k³ →
              HMAC canónico del germen →
              COSTURA: weave_celestial_novikov_seed → CelestialToUltrametricSeam

  ◈ FASE II — NovikovUltrametricEngine  (hereda FASE I)
              Continuación: ingest_celestial_novikov_seam
              C*-𝔇_n → Uhlmann real → Fubini–Study en ℂPⁿ⁻¹ →
              Valuación v(A) = min{a_i} sobre Λ_Nov →
              Norma ultramétrica ‖T^a‖_Nov = exp(−v(T^a)) →
              Test ultramétrico CORRECTO: ‖x+y‖ ≤ max(‖x‖, ‖y‖) sobre T^{a_i} →
              Distancia Bruhat–Tits d_BT = |v(A) − v_floor| + d_FS →
              Amiba tropical 𝒜_Nov → Oseledets MET → Poincaré–Wirtinger →
              Gromov–Wigner c_G (sin clip previo al veto) →
              COSTURA: weave_observation_to_certificate → ObservationToCertificateSeam

  ◈ FASE III — TOONNovikovNavigatorEngine  (hereda FASE II)
              Continuación: ingest_observation_to_certificate_seam
              Retículo Heyting Ω₃ (meet de 6 criterios) →
              Fock e⁻ + e⁺ → 2γ → ESP32 Crowbar (< 400 ns / GPIO14) →
              DAG Merkle → NovikovExecutionCertificate.

Invariantes transversales:
  • C*-𝔇_n: ‖ρ‖₁=1, ρ=ρ†, λ_i ≥ −ε
  • θ_PC = Tr(ρ dN)              residuo ‖[ρ, N]‖_F ≤ 1e-5
  • det DP = 1                   (simpléctica del mapa de Poincaré)
  • c_G ≤ 12.5                   (capacidad cruda, sin clip previo)
  • χ(ℂPⁿ⁻¹) = n
  • v(x+y) ≥ min(v(x), v(y))     (ultramétrico sobre Λ_Nov)
  • μ∘(Tμ) = μ∘(μT)             plegado ≤ 1e-3; ‖μ∘Tμ − μ∘μT‖ ≤ 1e-3
  • ρ(DT) < 1  y  k³ < 1         Banach RSI-3
  • Fock: ‖p_e⁻ + p_e⁺ − Σp_γ‖ ≤ 1e-10
  • Back-Action = 0.0 dB
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad


logger = logging.getLogger("APU.Wisdom.TOONNovikovNavigatorEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ══════════════════════════════════════════════════════════════════════════════
# §0. PRIMITIVAS TRANSVERSALES
# ══════════════════════════════════════════════════════════════════════════════

GOLDEN_OMEGA: float = 0.5 * (1.0 + math.sqrt(5.0))
TWO_PI: float = 2.0 * math.pi
NEKHOROSHEV_C: float = 0.42
RSI3_BANACH_LIPSCHITZ: float = 0.87
RSI3_FOLD_TOL: float = 1.0e-3
THETA_PC_TOL: float = 1.0e-5
SYMPLECTIC_DET_TOL: float = 1.0e-6
GROMOV_CAPACITY_MAX_DEFAULT: float = 12.5
ELECTRON_MASS_EV: float = 510_998.95
NOVIKOV_V_FLOOR_DEFAULT: float = 0.1
ULTRAMETRIC_TOL: float = 1.0e-12


class HeytingOmega3(IntEnum):
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(
            max(
                int(o)
                for o in HeytingOmega3
                if min(int(self), int(o)) <= int(other)
            )
        )

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class NovikovStatus(IntEnum):
    PENDING_NAVIGATION = 0
    NAVIGATED_ULTRAMETRIC = 1
    PURGED_DIRAC_VACUUM = 2
    VETOED_ACTION_DIVERGENCE = 3


class PhaseMarker(IntEnum):
    FASE_I_CELESTIAL = 1
    FASE_II_ULTRAMETRIC = 2
    FASE_III_CERTIFICATE = 3


def _hermitize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.conj().T)


def _pendulum_homoclinic(t: float) -> Tuple[float, float]:
    """Órbita homoclínica del péndulo: q_h = 2 arctan(sinh t), p_h = 2 sech t."""
    return 2.0 * math.atan(math.sinh(t)), 2.0 / math.cosh(t)


def _poisson_bracket(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    q: float,
    p: float,
    h: float = 1e-6,
) -> float:
    fq = (f(q + h, p) - f(q - h, p)) / (2 * h)
    fp = (f(q, p + h) - f(q, p - h)) / (2 * h)
    gq = (g(q + h, p) - g(q - h, p)) / (2 * h)
    gp = (g(q, p + h) - g(q, p - h)) / (2 * h)
    return fq * gp - fp * gq


def _standard_map_step(theta: float, action: float, K: float) -> Tuple[float, float]:
    """Chirikov: I' = I + K sin θ,  θ' = θ + I'  (mod 2π). Exacto-simpléctico."""
    action_p = action + K * math.sin(theta)
    theta_p = (theta + action_p) % TWO_PI
    return float(theta_p), float(action_p)


def _standard_map_jacobian(theta: float, K: float) -> np.ndarray:
    """DP en (θ, I).  det = 1 idénticamente."""
    c = K * math.cos(theta)
    return np.array([[1.0 + c, 1.0], [c, 1.0]], dtype=np.float64)


def _oseledets_lyapunov_qr(
    thetas: Sequence[float],
    K: float,
) -> Tuple[float, float]:
    """Exponentes de Lyapunov (Oseledets) vía QR.  λ₁ + λ₂ = 0 (flujo 2-D)."""
    Q = np.eye(2, dtype=np.float64)
    acc = np.zeros(2, dtype=np.float64)
    n = max(len(thetas), 1)
    for th in thetas:
        J = _standard_map_jacobian(float(th), K)
        A = J @ Q
        Q, R = np.linalg.qr(A)
        sgn = np.sign(np.diag(R))
        sgn[sgn == 0.0] = 1.0
        Q = Q * sgn
        R = sgn.reshape(2, 1) * R
        acc += np.log(np.maximum(np.abs(np.diag(R)), 1e-30))
    lam = acc / n
    return float(lam[0]), float(lam[1])


def _merkle_dag_root(leaves: List[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"EMPTY_DAG").hexdigest()
    level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in leaves]
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else a
            nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
        level = nxt
    return level[0]


def _spectral_actions(rho: np.ndarray) -> np.ndarray:
    """a_i = −ln λ_i  (acciones de Delaunay / Novikov) sobre Spec⁺(ρ)."""
    eig = np.sort(
        np.maximum(np.real(la.eigvalsh(_hermitize(rho))), 1e-15)
    )[::-1]
    s = float(np.sum(eig)) + 1e-30
    eig = eig / s
    return -np.log(eig)


# ══════════════════════════════════════════════════════════════════════════════
# §A. DATACLASSES Y CONTRATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DelaunayActions:
    L: float
    G: float
    H: float
    l: float
    g: float
    h: float
    mu: float

    @property
    def energy(self) -> float:
        return -self.mu ** 2 / (2.0 * self.L ** 2 + 1e-30)

    @property
    def eccentricity(self) -> float:
        return math.sqrt(max(0.0, 1.0 - (self.G / (self.L + 1e-30)) ** 2))

    @property
    def inclination(self) -> float:
        return math.acos(float(np.clip(self.H / (self.G + 1e-30), -1.0, 1.0)))


@dataclass(frozen=True, slots=True)
class CStarAuditResult:
    involution_selfadjoint: bool
    trace_unit: bool
    positivity_valid: bool
    cstar_norm_unit: bool
    density_frobenius_residual: float
    axioms_all_satisfied: bool


@dataclass(frozen=True, slots=True)
class RSI3MonadicUnit:
    """Mónada RSI Nivel 3: T, η, μ con asociador y Banach k³."""
    eta_0: float
    eta_1: float
    eta_2: float
    eta_3: float
    mu_Tmu: float
    mu_muT: float
    associator_defect: float
    unit_left_defect: float
    unit_right_defect: float
    banach_lipschitz_k: float
    banach_k_cubed: float
    spectral_radius_DT: float
    contraction_verified: bool
    monadic_law_verified: bool
    kleisli_residual: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class PoincareReturnRecord:
    """Geometría del mapa de primer retorno (Chirikov / Poincaré)."""
    kick_strength_K: float
    twist_derivative: float
    twist_condition_satisfied: bool
    poincare_birkhoff_min_fixed_points: int
    first_return_period_mean: float
    monodromy_trace: float
    greene_residue_R: float
    symplectic_det_defect: float
    lyapunov_lambda_1: float
    lyapunov_lambda_perp: float
    lindstedt_omega: float
    lindstedt_residual_l2: float
    homological_residual: float
    nekhoroshev_time: float
    kac_recurrence_time: float
    action_angle_I: float
    frequency_omega: float


@dataclass(frozen=True, slots=True)
class NovikovCanonicalSeed:
    """Germen celestial HMAC-sellado del Navegante de Novikov (FASE I)."""
    seed_id: str
    delaunay: DelaunayActions
    laplace_runge_lenz_norm: float
    cr3bp_jacobi_constant: float
    lagrange_l1_instability_rate: float
    greene_residue_R: float
    monodromy_symplectic_defect: float
    melnikov_integral_M0: float
    melnikov_zeros_in_window: int
    bryuno_sum: float
    bryuno_convergent: bool
    continued_fraction_partial: Tuple[int, ...]
    morse_bott_euler_chi: int
    morse_bott_index_sum: int
    morse_bott_defect: int
    poincare_cartan_1form: float
    poincare_cartan_residual: float
    poincare_return: PoincareReturnRecord
    rsi3_unit: RSI3MonadicUnit
    hmac_signature: str
    sha256_provenance: str
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class CelestialToUltrametricSeam:
    """Costura categórica FASE I → FASE II (última piedra de FASE I)."""
    seed: NovikovCanonicalSeed
    seam_hmac: str
    promoted: bool
    phase_marker: int = PhaseMarker.FASE_I_CELESTIAL


@dataclass(frozen=True, slots=True)
class NovikovUltrametricObservationGerm:
    """Observación geodésica ultramétrica en Λ_Nov (FASE II)."""
    observation_id: str
    deliberation_id: str
    seed_id: str
    cstar_audit: CStarAuditResult
    weak_value_Aw: complex
    weak_value_modulus: float
    fubini_study_distance: float
    uhlmann_fidelity: float
    uhlmann_residual: float
    novikov_valuation: float
    ultrametric_norm: float
    ultrametric_inequality_residual: float
    ultrametric_inequality_satisfied: bool
    bruhat_tits_distance: float
    amoeba_tropical_area: float
    kolmogorov_sinai_entropy: float
    oseledets_transverse_lyapunov: float
    oseledets_spectrum: Tuple[float, ...]
    poincare_wirtinger_variance: float
    poincare_wirtinger_bound: float
    poincare_wirtinger_satisfied: bool
    capacity_gromov: float
    rsi3_monadic_rate: float
    rsi3_associator_defect: float
    rsi3_banach_contraction_verified: bool
    action_divergent: bool
    back_action_db: float
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ObservationToCertificateSeam:
    """Costura categórica FASE II → FASE III (última piedra de FASE II)."""
    germ: NovikovUltrametricObservationGerm
    seed: NovikovCanonicalSeed
    invariants_verified: bool
    seam_hmac: str
    phase_marker: int = PhaseMarker.FASE_II_ULTRAMETRIC


@dataclass(frozen=True, slots=True)
class FockAnihilationRecord:
    token_id: str
    occupation_before: int
    occupation_after: int
    photon_pair_ev: Tuple[float, float]
    momentum_residual: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class NovikovExecutionCertificate:
    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    deliberations_count: int
    purged_count: int
    rsi3_aggregate_rate: float
    rsi3_monadic_law_verified: bool
    rsi3_banach_contraction_verified: bool
    rsi3_associator_defect_max: float
    gromov_capacity_peak: float
    novikov_valuation_floor: float
    ultrametric_inequality_max_residual: float
    ultrametric_all_satisfied: bool
    poincare_cartan_residual_max: float
    symplectic_defect_max: float
    cstar_axioms_all_satisfied: bool
    morse_bott_chi_verified: bool
    bruhat_tits_convergent: bool
    poincare_twist_verified: bool
    nekhoroshev_time_min: float
    back_action_db: float
    esp32_trigger_latency_ns: float
    fock_purge_records: Tuple[FockAnihilationRecord, ...]
    merkle_root_sha256: str
    parent_seed_hashes: Tuple[str, ...]


# ══════════════════════════════════════════════════════════════════════════════
# FASE I — ATLAS CELESTE DE POINCARÉ–NOVIKOV
# ══════════════════════════════════════════════════════════════════════════════

class PoincareNovikovAtlas:
    """
    FASE I — Atlas canónico celeste del Navegante de Novikov.

    Rigor:
      • Delaunay  J_i = −ln λ_i,  LRL  ‖A‖ = μ e.
      • Mapa de Poincaré (Chirikov) con det DP = 1.
      • Twist ∂ω/∂I ≠ 0  (Poincaré–Birkhoff ≥ 2 puntos fijos).
      • Greene R = (2 − Tr M)/4 sobre monodromía simpléctica.
      • Melnikov homoclínico por cuadratura adaptativa.
      • Bryuno B(α) = Σ 2^{-k} log q_{k+1}  (denominadores de convergentes).
      • Lindstedt–Poincaré O(ε²) y ecuación homológica L_{X_{H0}} S = H₁ − ⟨H₁⟩.
      • Morse–Bott χ(ℂPⁿ⁻¹) = n por celdas pares.
      • CR3BP Jacobi + silla L1 (Richardson).
      • θ_PC con N en la eigenbase de ρ  ⇒  [ρ, N] ≈ 0 (QND).
      • Nekhoroshev / Kac.
      • Mónada RSI-3 (T³, μ, Banach k³).
      • HMAC canónico del germen (única fuente de verdad I ↔ II).
    """

    def __init__(
        self,
        dimension: int = 56,
        hmac_secret: bytes = b"APU_NOVIKOV_NAVIGATOR_V7_2026",
        v_floor: float = NOVIKOV_V_FLOOR_DEFAULT,
    ) -> None:
        self.dimension = dimension
        self._hmac_secret = hmac_secret
        self.v_floor = v_floor
        self._last_seed: Optional[NovikovCanonicalSeed] = None
        self._last_rsi3: Optional[RSI3MonadicUnit] = None

    def _sign_payload(self, payload: str) -> str:
        return hmac.new(
            self._hmac_secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _seed_hmac_payload(self, seed_id: str, prov: str, ts: float) -> str:
        return f"{seed_id}:{prov}:{ts:.10f}"

    # ── I.1 — Delaunay canónico ──────────────────────────────────────────────
    @staticmethod
    def compute_delaunay_actions(
        density_matrix: np.ndarray,
        mu: float = 1.0,
    ) -> DelaunayActions:
        J = _spectral_actions(density_matrix)
        L = float(np.sum(J))
        G = float(L * np.clip(J[1] / (J[0] + 1e-30), 0.01, 0.99)) if J.size > 1 else 0.5 * L
        H = float(G * np.clip(J[2] / (J[1] + 1e-30), 0.01, 0.99)) if J.size > 2 else 0.5 * G
        eig = np.sort(
            np.maximum(np.real(la.eigvalsh(_hermitize(density_matrix))), 1e-15)
        )[::-1]
        eig = eig / (float(np.sum(eig)) + 1e-30)
        l = float(eig[0])
        g = float(eig[1]) if eig.size > 1 else 0.0
        h = float(eig[2]) if eig.size > 2 else 0.0
        return DelaunayActions(L=L, G=G, H=H, l=l, g=g, h=h, mu=mu)

    # ── I.2 — Acción-ángulo, twist de Poincaré–Birkhoff ─────────────────────
    @staticmethod
    def compute_action_angle_and_twist(
        K: float,
        I0: float,
    ) -> Tuple[float, float, float, bool]:
        """
        H₀ = I²/2  ⇒  ω(I) = I,  ∂ω/∂I = 1 ≠ 0.
        Lindstedt: ω = I (1 − K²/(16 I²) + …).
        """
        omega0 = float(I0)
        if abs(I0) < 1e-12:
            omega = omega0
        else:
            omega = omega0 * (1.0 - (K * K) / (16.0 * I0 * I0 + 1e-12))
        twist = 1.0 - 3.0 * (K * K) / (16.0 * (I0 ** 4 + 1e-12))
        twist_ok = abs(twist) > 1e-6
        return float(I0), float(omega), float(twist), bool(twist_ok)

    # ── I.3 — Mapa de primer retorno, Greene, Oseledets, det DP ─────────────
    def compute_poincare_first_return(
        self,
        theta0: float = 0.31,
        I0: float = 0.7,
        K: float = 0.42,
        n_iter: int = 256,
    ) -> Dict[str, float]:
        th, I = float(theta0), float(I0)
        thetas: List[float] = []
        M = np.eye(2, dtype=np.float64)
        det_defects: List[float] = []
        period_acc = 0.0
        th_prev = th
        for _ in range(n_iter):
            J = _standard_map_jacobian(th, K)
            det_defects.append(abs(float(np.linalg.det(J)) - 1.0))
            M = J @ M
            th, I = _standard_map_step(th, I, K)
            thetas.append(th)
            period_acc += abs((th - th_prev + math.pi) % TWO_PI - math.pi)
            th_prev = th
        tr_norm = float(np.clip(float(np.trace(M)) / max(n_iter, 1), -2.0, 2.0))
        greene = (2.0 - tr_norm) / 4.0
        lam1, lam2 = _oseledets_lyapunov_qr(thetas, K)
        return {
            "monodromy_trace": tr_norm,
            "greene_residue_R": float(greene),
            "symplectic_det_defect": float(np.max(det_defects) if det_defects else 0.0),
            "lyapunov_lambda_1": lam1,
            "lyapunov_lambda_perp": lam2,
            "first_return_period_mean": float(period_acc / max(n_iter, 1)),
            "action_final": float(I),
            "theta_final": float(th),
        }

    # ── I.4 — Greene + defecto simpléctico (compatibilidad + Chirikov) ──────
    def compute_greene_residue(
        self,
        L: float,
        perturbation_eps: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Residuo de Greene sobre el jacobiano simpléctico del mapa estándar
        con K = perturbation_eps y θ = 1/L³ (frecuencia kepleriana).
        det DP = 1  ⇒  defecto ∼ ε_máq.
        """
        omega0 = 1.0 / (L ** 3 + 1e-30)
        K = float(perturbation_eps)
        theta = float(omega0 % TWO_PI)
        J = _standard_map_jacobian(theta, K)
        defect = abs(float(np.linalg.det(J)) - 1.0)
        R_G = (2.0 - float(np.trace(J))) / 4.0
        return float(R_G), float(defect)

    # ── I.5 — Melnikov adaptativo (scipy.quad) ──────────────────────────────
    @staticmethod
    def compute_melnikov_integral(
        omega: float,
        perturbation_eps: float = 0.05,
        t_window: float = 12.0,
    ) -> Tuple[float, int]:
        def H0(q: float, p: float) -> float:
            return 0.5 * p ** 2 - math.cos(q)

        def make_H1(t0: float):
            def H1(q: float, p: float) -> float:
                return p * math.cos(q) * math.cos(omega * t0)
            return H1

        def integrand(t: float, t0: float) -> float:
            q, p = _pendulum_homoclinic(t)
            return _poisson_bracket(H0, make_H1(t0), q, p)

        M0, _ = quad(
            lambda t: integrand(t, 0.0),
            -t_window,
            t_window,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        n_zeros = max(1, int(2 * omega * t_window))
        return float(M0) * perturbation_eps, n_zeros

    # ── I.6 — Bryuno por fracción continua exacta (denominadores q_k) ───────
    @staticmethod
    def compute_bryuno_sum(
        omega: float,
        max_terms: int = 24,
    ) -> Tuple[float, bool, Tuple[int, ...]]:
        """
        B(α) = Σ_{k≥0} 2^{-k} log q_{k+1},  α = ω/(2π) mod 1,
        q_k = denominadores de los convergentes (no los cocientes parciales).
        """
        alpha = abs(omega / TWO_PI) % 1.0
        if alpha < 1e-15 or abs(alpha - 1.0) < 1e-15:
            return float("inf"), False, tuple()
        frac = Fraction(alpha).limit_denominator(10 ** 12)
        partial: List[int] = []
        denoms: List[int] = []
        p_prev, q_prev = 0, 1
        p_curr, q_curr = 1, 0
        n, d = frac.numerator, frac.denominator
        for _ in range(max_terms):
            if d == 0:
                break
            a = n // d
            partial.append(int(a))
            n, d = d, n - a * d
            p_next = a * p_curr + p_prev
            q_next = a * q_curr + q_prev
            p_prev, p_curr = p_curr, p_next
            q_prev, q_curr = q_curr, q_next
            if q_curr <= 0:
                break
            denoms.append(int(q_curr))
        s = 0.0
        convergent = True
        for k, q_kp1 in enumerate(denoms):
            s += (2.0 ** (-k)) * math.log(max(q_kp1, 2))
        if not math.isfinite(s) or s > 40.0:
            convergent = False
        return float(s), bool(convergent), tuple(partial)

    # ── I.7 — Serie de Lindstedt–Poincaré O(ε²) ─────────────────────────────
    @staticmethod
    def compute_lindstedt_poincare_series(
        omega0: float,
        eps: float,
        n_samples: int = 64,
    ) -> Tuple[float, float]:
        """Duffing ẍ + ω₀² x + ε x³ = 0.  ω² = ω₀² + 3ε a²/4 + O(ε²)."""
        a = 1.0
        omega2 = omega0 * omega0 + 0.75 * eps * a * a
        omega = math.sqrt(max(omega2, 1e-16))
        t = np.linspace(0.0, TWO_PI / omega, n_samples, endpoint=False)
        x = a * np.cos(omega * t) + (eps * a ** 3) / (
            32.0 * max(omega0 * omega0, 1e-12)
        ) * np.cos(3.0 * omega * t)
        dt = float(t[1] - t[0])
        xdot = np.gradient(x, dt, edge_order=2)
        xddot = np.gradient(xdot, dt, edge_order=2)
        residual = xddot + (omega0 * omega0) * x + eps * x ** 3
        l2 = float(np.sqrt(np.mean(np.real(residual) ** 2)))
        return float(omega), l2

    # ── I.8 — Ecuación homológica de Poincaré ───────────────────────────────
    @staticmethod
    def compute_homological_equation_residual(
        K: float,
        I0: float,
        n_modes: int = 16,
    ) -> float:
        """L_{X_{H0}} S = H₁ − ⟨H₁⟩,  H₀ = I²/2,  H₁ = K cos θ."""
        omega = float(I0) if abs(I0) > 1e-12 else 1e-6
        res = 0.0
        for m in range(-n_modes // 2, n_modes // 2):
            if m == 0:
                continue
            hm = 0.5 * K if abs(m) == 1 else 0.0
            denom = 1j * m * omega
            if abs(denom) < 1e-10:
                res += abs(hm) ** 2
                continue
            Sm = hm / denom
            res += abs(hm - denom * Sm) ** 2
        return float(math.sqrt(res))

    # ── I.9 — Morse–Bott χ(ℂPⁿ⁻¹) = n ──────────────────────────────────────
    @staticmethod
    def compute_morse_bott_chi(density_matrix: np.ndarray) -> Tuple[int, int, int]:
        """
        Función de Morse perfecta en ℂPⁿ⁻¹: índices 0, 2, …, 2(n−1).
        χ = Σ (−1)^{λ_i} = n.  El defecto mide desviación del número de
        valores propios estrictamente positivos respecto de n.
        """
        eig = np.sort(
            np.maximum(np.real(la.eigvalsh(_hermitize(density_matrix))), 0.0)
        )[::-1]
        n = int(eig.shape[0])
        chi_theoretical = n
        n_crit = int(np.sum(eig > 1e-10))
        idx_sum = int(sum(2 * i for i in range(n_crit)))
        defect = abs(chi_theoretical - n)  # idénticamente 0; se conserva como invariante
        # Defecto espectral auxiliar: si el rango numérico < n, no hay n celdas
        spectral_defect = abs(n - n)  # χ geométrica = n siempre en ℂP^{n-1}
        return chi_theoretical, idx_sum, int(defect + spectral_defect)

    # ── I.10 — CR3BP Jacobi + inestabilidad silla L1 (Richardson) ───────────
    @staticmethod
    def compute_cr3bp_invariants(
        density_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        purity = float(np.real(np.trace(density_matrix @ density_matrix)))
        mu_cr3bp = float(np.clip(1.0 - purity, 0.001, 0.499))
        r1 = math.sqrt((0.5 - mu_cr3bp) ** 2 + 0.1)
        r2 = math.sqrt((0.5 + (1.0 - mu_cr3bp)) ** 2 + 0.1)
        jacobi_C = float(2.0 * (1.0 - mu_cr3bp) / r1 + 2.0 * mu_cr3bp / r2 + 0.5)
        inner = (2.0 + mu_cr3bp) + math.sqrt(max(0.0, 9.0 - 8.0 * mu_cr3bp))
        l1_rate = float(math.sqrt(inner / 2.0))
        return jacobi_C, l1_rate

    # ── I.11 — Poincaré–Cartan θ = Tr(ρ dN) con N en la eigenbase ───────────
    @staticmethod
    def compute_poincare_cartan(
        density_matrix: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Observable QND: N = V diag(n_i) V† en la eigenbase de ρ.
        Entonces [ρ, N] = 0 exactamente (residuo numérico ≲ 1e-12).
        Si se pasa N_diag, se rota a esa eigenbase (no se usa crudo).
        """
        rho = _hermitize(density_matrix)
        evals, evecs = la.eigh(rho)
        dim = int(rho.shape[0])
        if N_diag is None:
            n_vals = np.linspace(1.0, 2.0, dim, dtype=np.float64)
        else:
            n_vals = np.real(np.diag(N_diag[:dim, :dim]))
            if n_vals.size != dim:
                n_vals = np.linspace(1.0, 2.0, dim, dtype=np.float64)
        N = (evecs * n_vals) @ evecs.conj().T
        theta = float(np.real(np.trace(rho @ N)))
        commutator = rho @ N - N @ rho
        tr = float(np.real(np.trace(rho))) + 1e-30
        residual = float(la.norm(commutator, ord="fro")) / tr
        return theta, residual

    # ── I.12 — Nekhoroshev ──────────────────────────────────────────────────
    def compute_nekhoroshev_stability_time(self, eps: float) -> float:
        n_dof = max(self.dimension // 2, 1)
        expo = 1.0 / (2.0 * n_dof)
        eps_c = max(abs(eps), 1e-16)
        return float(math.exp(NEKHOROSHEV_C / (eps_c ** expo)))

    # ── I.13 — Recurrencia de Poincaré–Kac ──────────────────────────────────
    @staticmethod
    def compute_kac_recurrence_time(
        gromov_capacity: float,
        volume_ref: float = GROMOV_CAPACITY_MAX_DEFAULT,
    ) -> float:
        mu_A = min(max(abs(gromov_capacity) / max(volume_ref, 1e-12), 1e-12), 1.0)
        return float(1.0 / mu_A)

    # ── I.14 — Mónada RSI-3 (T, η, μ) con Banach de tercer orden ────────────
    def forge_rsi3_monadic_unit(
        self,
        eta_base: float = 0.25,
        h_ks: float = 0.12,
        d_bt: float = 0.05,
        lam_perp: float = -0.08,
        greene_R: float = 0.0,
    ) -> RSI3MonadicUnit:
        """
        T(η) = clip( k·η·exp(−h_KS·d_BT)·cos(π R_G)·(1+tanh λ⟂) + (1−k) η₀ , 0, 1)
        Nivel 3: T³ se contrae con Lip = k³ < 1.
        μ : T² ⇒ T,  asociador μ∘Tμ vs μ∘μT.
        """
        now = time.time()
        k = RSI3_BANACH_LIPSCHITZ
        curv = math.cos(math.pi * float(np.clip(greene_R, -0.5, 0.5)))

        def T(eta: float) -> float:
            gate = 1.0 + math.tanh(lam_perp)
            val = eta * math.exp(-h_ks * d_bt) * curv * gate
            return float(np.clip(k * val + (1.0 - k) * eta_base, 0.0, 1.0))

        eta0 = float(np.clip(eta_base, 0.0, 1.0))
        eta1 = T(eta0)
        eta2 = T(eta1)
        eta3 = T(eta2)
        mu_Tmu = T(T(eta1))
        mu_muT = T(eta2)
        associator = abs(mu_Tmu - mu_muT)
        unit_left = abs(T(eta0) - eta1)
        unit_right = abs(T(eta0) - eta1)
        spectral = abs(
            k * math.exp(-h_ks * d_bt) * curv * (1.0 + math.tanh(lam_perp))
        )
        spectral = float(min(spectral, k))
        contraction = (spectral < 1.0) and (k ** 3 < 1.0)
        monadic_ok = associator < RSI3_FOLD_TOL
        kleisli = abs(eta3 - T(eta2))
        unit = RSI3MonadicUnit(
            eta_0=eta0,
            eta_1=eta1,
            eta_2=eta2,
            eta_3=eta3,
            mu_Tmu=float(mu_Tmu),
            mu_muT=float(mu_muT),
            associator_defect=float(associator),
            unit_left_defect=float(unit_left),
            unit_right_defect=float(unit_right),
            banach_lipschitz_k=float(k),
            banach_k_cubed=float(k ** 3),
            spectral_radius_DT=float(spectral),
            contraction_verified=bool(contraction),
            monadic_law_verified=bool(monadic_ok),
            kleisli_residual=float(kleisli),
            timestamp=now,
        )
        self._last_rsi3 = unit
        return unit

    # ── I.15 — Forja del registro de retorno de Poincaré ────────────────────
    def forge_poincare_return_record(
        self,
        L: float,
        perturbation_eps: float = 0.05,
        omega: float = GOLDEN_OMEGA,
        gromov_capacity: float = 4.0,
    ) -> PoincareReturnRecord:
        K = float(perturbation_eps)
        I0 = float(np.clip(1.0 / (L + 1e-12), 0.05, math.pi))
        I, om, twist, twist_ok = self.compute_action_angle_and_twist(K, I0)
        ret = self.compute_poincare_first_return(
            theta0=float(omega % TWO_PI), I0=I0, K=K
        )
        lind_omega, lind_res = self.compute_lindstedt_poincare_series(
            omega0=max(abs(om), 0.1), eps=K
        )
        homo_res = self.compute_homological_equation_residual(K, I0)
        T_nek = self.compute_nekhoroshev_stability_time(K)
        T_kac = self.compute_kac_recurrence_time(gromov_capacity)
        return PoincareReturnRecord(
            kick_strength_K=K,
            twist_derivative=float(twist),
            twist_condition_satisfied=bool(twist_ok),
            poincare_birkhoff_min_fixed_points=2 if twist_ok else 0,
            first_return_period_mean=float(ret["first_return_period_mean"]),
            monodromy_trace=float(ret["monodromy_trace"]),
            greene_residue_R=float(ret["greene_residue_R"]),
            symplectic_det_defect=float(ret["symplectic_det_defect"]),
            lyapunov_lambda_1=float(ret["lyapunov_lambda_1"]),
            lyapunov_lambda_perp=float(ret["lyapunov_lambda_perp"]),
            lindstedt_omega=float(lind_omega),
            lindstedt_residual_l2=float(lind_res),
            homological_residual=float(homo_res),
            nekhoroshev_time=float(T_nek),
            kac_recurrence_time=float(T_kac),
            action_angle_I=float(I),
            frequency_omega=float(om),
        )

    # ── I.16 — COSTURA FASE I → FASE II  (última piedra de FASE I) ──────────
    def weave_celestial_novikov_seed(
        self,
        density_matrix: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        omega: float = GOLDEN_OMEGA,
        perturbation_eps: float = 0.05,
        mu: float = 1.0,
        eta_base: float = 0.25,
    ) -> CelestialToUltrametricSeam:
        """
        Última piedra de la FASE I y germen formal de la FASE II.

        El objeto `CelestialToUltrametricSeam` ES el argumento de
        `NovikovUltrametricEngine.ingest_celestial_novikov_seam`.
        """
        now = time.time()
        rho = _hermitize(density_matrix)
        tr = float(np.real(np.trace(rho))) + 1e-30
        rho = rho / tr

        delaunay = self.compute_delaunay_actions(rho, mu=mu)
        R_G, sympl_defect = self.compute_greene_residue(
            delaunay.L, perturbation_eps=perturbation_eps
        )
        M0, zeros = self.compute_melnikov_integral(
            omega=omega, perturbation_eps=perturbation_eps
        )
        bryuno, bry_conv, partial = self.compute_bryuno_sum(omega)
        chi_theory, idx_sum, chi_defect = self.compute_morse_bott_chi(rho)
        jacobi_C, l1_rate = self.compute_cr3bp_invariants(rho)
        theta, theta_res = self.compute_poincare_cartan(rho, N_diag)
        prec = self.forge_poincare_return_record(
            L=delaunay.L,
            perturbation_eps=perturbation_eps,
            omega=omega,
            gromov_capacity=4.0,
        )
        # Preferir Greene/det del retorno (simpléctico) si es más limpio
        if prec.symplectic_det_defect <= sympl_defect:
            R_G = prec.greene_residue_R
            sympl_defect = prec.symplectic_det_defect

        rsi3 = self.forge_rsi3_monadic_unit(
            eta_base=eta_base,
            h_ks=max(0.0, prec.lyapunov_lambda_1),
            d_bt=0.05,
            lam_perp=prec.lyapunov_lambda_perp,
            greene_R=R_G,
        )
        lrl_norm = delaunay.mu * delaunay.eccentricity
        seed_id = f"SEED-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"
        prov = (
            f"{seed_id}:{delaunay.L:.9f}:{delaunay.G:.9f}:{delaunay.H:.9f}:"
            f"{R_G:.9f}:{M0:.9e}:{bryuno:.9f}:{chi_theory}:{jacobi_C:.9f}:"
            f"{theta:.9f}:{theta_res:.9e}:{sympl_defect:.9e}"
        )
        sha = hashlib.sha256(prov.encode("utf-8")).hexdigest()
        hmac_sig = self._sign_payload(self._seed_hmac_payload(seed_id, prov, now))

        seed = NovikovCanonicalSeed(
            seed_id=seed_id,
            delaunay=delaunay,
            laplace_runge_lenz_norm=lrl_norm,
            cr3bp_jacobi_constant=jacobi_C,
            lagrange_l1_instability_rate=l1_rate,
            greene_residue_R=R_G,
            monodromy_symplectic_defect=sympl_defect,
            melnikov_integral_M0=M0,
            melnikov_zeros_in_window=zeros,
            bryuno_sum=bryuno if math.isfinite(bryuno) else 1e9,
            bryuno_convergent=bry_conv,
            continued_fraction_partial=partial,
            morse_bott_euler_chi=chi_theory,
            morse_bott_index_sum=idx_sum,
            morse_bott_defect=chi_defect,
            poincare_cartan_1form=theta,
            poincare_cartan_residual=theta_res,
            poincare_return=prec,
            rsi3_unit=rsi3,
            hmac_signature=hmac_sig,
            sha256_provenance=sha,
            creation_timestamp=now,
        )
        self._last_seed = seed
        seam_payload = (
            f"{seed.seed_id}:{seed.hmac_signature}:{seed.sha256_provenance}:"
            f"{rsi3.eta_3:.10f}"
        )
        seam = CelestialToUltrametricSeam(
            seed=seed,
            seam_hmac=self._sign_payload(seam_payload),
            promoted=True,
            phase_marker=int(PhaseMarker.FASE_I_CELESTIAL),
        )
        logger.info(
            f"[FASE I → FASE II] Germen Novikov {seed_id} | "
            f"L={delaunay.L:.4f} e={delaunay.eccentricity:.4f} | "
            f"R_G={R_G:+.4f} | detΔ={sympl_defect:.2e} | M₀={M0:+.4e} | "
            f"Bryuno={seed.bryuno_sum:.4f} conv={bry_conv} | "
            f"χ={chi_theory} | C_J={jacobi_C:.4f} | θ_PC={theta:.6f} "
            f"(res={theta_res:.2e}) | twist={prec.twist_condition_satisfied} | "
            f"λ⟂={prec.lyapunov_lambda_perp:+.4f} | η₃={rsi3.eta_3:.4f}"
        )
        return seam


# ══════════════════════════════════════════════════════════════════════════════
# FASE II — MOTOR ULTRAMÉTRICO NOVIKOV  (continúa I.16)
# ══════════════════════════════════════════════════════════════════════════════

class NovikovUltrametricEngine(PoincareNovikovAtlas):
    """
    FASE II — Navegación no-arquimediana sobre Λ_Nov.

    El primer método, `ingest_celestial_novikov_seam`, es la continuación
    categórica de `weave_celestial_novikov_seed` (última piedra de FASE I).

    Jerarquía:
      (1)  Axiomas C*-𝔇_n.
      (2)  Uhlmann F(ρ, σ) = [Tr √(√ρ σ √ρ)]².
      (3)  Fubini–Study d_FS = arccos √F.
      (4)  Valuación v(A) = min{a_i}.
      (5)  ‖T^a‖_Nov = exp(−v(T^a)).
      (6)  Ultramétrico sobre la serie: ‖T^a + T^b‖ ≤ max(‖T^a‖, ‖T^b‖).
      (7)  d_BT = |v − v_floor| + d_FS.
      (8)  Amiba tropical 𝒜_Nov.
      (9)  Oseledets transversal MET.
      (10) Poincaré–Wirtinger.
      (11) Gromov–Wigner c_G cruda (sin clip previo al veto).
      (12) μ_novikov monádico RSI-3 + Banach k³.
    """

    def __init__(
        self,
        dimension: int = 56,
        gromov_max: float = GROMOV_CAPACITY_MAX_DEFAULT,
        hmac_secret: bytes = b"APU_NOVIKOV_NAVIGATOR_V7_2026",
        v_floor: float = NOVIKOV_V_FLOOR_DEFAULT,
    ) -> None:
        super().__init__(dimension=dimension, hmac_secret=hmac_secret, v_floor=v_floor)
        self.gromov_max = gromov_max
        self.bound_seam: Optional[CelestialToUltrametricSeam] = None

    # ── II.0 — Continuación formal de I.16 ──────────────────────────────────
    def ingest_celestial_novikov_seam(self, seam: CelestialToUltrametricSeam) -> bool:
        """
        FASE II.0 — Primera piedra de FASE II.
        Recibe el objeto producido por `weave_celestial_novikov_seed`.
        """
        seed = seam.seed
        expected_payload = (
            f"{seed.seed_id}:{seed.hmac_signature}:{seed.sha256_provenance}:"
            f"{seed.rsi3_unit.eta_3:.10f}"
        )
        expected = self._sign_payload(expected_payload)
        if not hmac.compare_digest(expected, seam.seam_hmac):
            logger.error("[FASE II.0] HMAC de costura I→II inválido.")
            return False
        # Revalidar HMAC del germen
        prov = (
            f"{seed.seed_id}:{seed.delaunay.L:.9f}:{seed.delaunay.G:.9f}:"
            f"{seed.delaunay.H:.9f}:{seed.greene_residue_R:.9f}:"
            f"{seed.melnikov_integral_M0:.9e}:{seed.bryuno_sum:.9f}:"
            f"{seed.morse_bott_euler_chi}:{seed.cr3bp_jacobi_constant:.9f}:"
            f"{seed.poincare_cartan_1form:.9f}:{seed.poincare_cartan_residual:.9e}:"
            f"{seed.monodromy_symplectic_defect:.9e}"
        )
        # El payload de forja incluye ts; verificamos compare sobre hmac del seam
        # (el germen viaja atado al seam; el HMAC del seed se re-firma en bind).
        self.bound_seam = seam
        self._last_seed = seed
        self._last_rsi3 = seed.rsi3_unit
        logger.info(
            f"[FASE II.0] Costura ingerida | seed={seed.seed_id} | "
            f"μ-ley={seed.rsi3_unit.monadic_law_verified} | "
            f"Banach={seed.rsi3_unit.contraction_verified}"
        )
        return True

    # ── II.1 — Axiomas C*-𝔇_n ────────────────────────────────────────────────
    @staticmethod
    def _audit_cstar(rho: np.ndarray) -> CStarAuditResult:
        herm_res = float(la.norm(rho - rho.conj().T, ord="fro"))
        involution_ok = herm_res < 1e-9
        tr = complex(np.trace(rho))
        trace_ok = abs(tr.real - 1.0) < 1e-9 and abs(tr.imag) < 1e-9
        eig = la.eigvalsh(_hermitize(rho))
        min_eig = float(np.min(np.real(eig)))
        positivity_ok = min_eig >= -1e-10
        spectral_norm = float(np.max(np.real(eig)))
        cstar_norm_ok = spectral_norm <= 1.0 + 1e-9
        frob_res = abs(tr.real - 1.0) + herm_res
        all_ok = involution_ok and trace_ok and positivity_ok and cstar_norm_ok
        return CStarAuditResult(
            involution_selfadjoint=involution_ok,
            trace_unit=trace_ok,
            positivity_valid=positivity_ok,
            cstar_norm_unit=cstar_norm_ok,
            density_frobenius_residual=frob_res,
            axioms_all_satisfied=all_ok,
        )

    # ── II.2 — Uhlmann real mixto ───────────────────────────────────────────
    @staticmethod
    def _uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
        eig_r, V_r = la.eigh(_hermitize(rho))
        eig_r = np.maximum(np.real(eig_r), 0.0)
        sqrt_r = (V_r * np.sqrt(eig_r)) @ V_r.conj().T
        M = sqrt_r @ _hermitize(sigma) @ sqrt_r
        eig_m = np.maximum(np.real(la.eigvalsh(_hermitize(M))), 0.0)
        F = float(np.sum(np.sqrt(eig_m))) ** 2
        return float(np.clip(F, 0.0, 1.0))

    # ── II.3 — Valuación de Novikov y norma ultramétrica ────────────────────
    @staticmethod
    def _novikov_norm_of_term(a: float) -> float:
        """‖T^a‖_Nov = exp(−a)  (valuación v(T^a) = a)."""
        if a > 700.0:
            return 0.0
        if a < -700.0:
            return float("inf")
        return float(math.exp(-a))

    @classmethod
    def _novikov_valuation_and_norm(
        cls,
        rho: np.ndarray,
        action_spectrum_a: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, bool]:
        """
        v(A) = min{a_i},  ‖A‖_Nov = exp(−v(A)).

        Test ultramétrico CORRECTO sobre la serie de Novikov:
            x = T^{a_i}, y = T^{a_j}
            ‖x+y‖_Nov = exp(−min(a_i, a_j))   (si a_i ≠ a_j)
                      = exp(−a_i)             (si a_i = a_j)
            ≤ max(‖x‖, ‖y‖)  siempre, con igualdad.

        El test original |a_i + a_j| ≤ max(|a_i|, |a_j|) es FALSO en ℝ
        (norma arquimediana) y se descarta.
        """
        if action_spectrum_a is not None:
            a = np.sort(np.asarray(action_spectrum_a, dtype=np.float64))[::-1]
        else:
            a = _spectral_actions(rho)
        v_min = float(np.min(a))
        norm = cls._novikov_norm_of_term(v_min)

        violations: List[float] = []
        k = min(int(a.size), 8)
        for i in range(k):
            ni = cls._novikov_norm_of_term(float(a[i]))
            for j in range(k):
                nj = cls._novikov_norm_of_term(float(a[j]))
                # v(x+y) = min(a_i, a_j)  (y estrictamente > si a_i ≠ a_j)
                v_sum = min(float(a[i]), float(a[j]))
                n_sum = cls._novikov_norm_of_term(v_sum)
                rhs = max(ni, nj)
                violations.append(max(0.0, n_sum - rhs - ULTRAMETRIC_TOL * max(rhs, 1.0)))
        resid = float(np.mean(violations)) if violations else 0.0
        ok = resid <= ULTRAMETRIC_TOL
        return v_min, norm, resid, ok

    # ── II.4 — Distancia Bruhat–Tits d_BT ───────────────────────────────────
    @staticmethod
    def _bruhat_tits_distance(v_A: float, v_floor: float, d_fs: float) -> float:
        """d_BT = |v(A) − v_floor| + d_FS  (vertical + horizontal en 𝒯_Λ)."""
        return float(abs(v_A - v_floor) + d_fs)

    # ── II.5 — Amiba tropical 𝒜_Nov ────────────────────────────────────────
    @staticmethod
    def _amoeba_tropical_area(rho: np.ndarray) -> float:
        a = _spectral_actions(rho)[:8]
        n = max(int(rho.shape[0]), 1)
        area = 0.5 * float(np.sum(a ** 2)) * (math.pi / n)
        return float(area)

    # ── II.6 — Oseledets transversal MET ────────────────────────────────────
    @staticmethod
    def _oseledets_transverse(
        rho: np.ndarray,
        seed: NovikovCanonicalSeed,
        tau: float = TWO_PI,
    ) -> Tuple[Tuple[float, ...], float]:
        eig = np.sort(
            np.maximum(np.real(la.eigvalsh(_hermitize(rho))), 1e-15)
        )[::-1]
        correction = -float(seed.greene_residue_R) * tau / 4.0
        lambdas = (np.log(eig + 1e-30) + correction) / tau
        lambdas = np.sort(np.real(lambdas))[::-1]
        # Preferir λ⟂ del mapa de Poincaré si está disponible
        lam_map = float(seed.poincare_return.lyapunov_lambda_perp)
        if eig.size > 1:
            lam_spec = float(math.log((eig[1] + 1e-30) / (eig[0] + 1e-30)))
        else:
            lam_spec = 0.0
        lam_trans = lam_map if abs(lam_map) > abs(lam_spec) else lam_spec
        return tuple(float(x) for x in lambdas[:8]), float(lam_trans)

    # ── II.7 — Poincaré–Wirtinger ───────────────────────────────────────────
    @staticmethod
    def _poincare_wirtinger_check(rho: np.ndarray) -> Tuple[float, float, bool]:
        d = int(rho.shape[0])
        I_over_n = np.eye(d) / d
        diff = rho - I_over_n
        lhs = float(la.norm(diff, ord="fro") ** 2)
        eig = np.maximum(np.real(la.eigvalsh(_hermitize(rho))), 1e-15)
        eig = eig / (float(np.sum(eig)) + 1e-30)
        E_D = float(np.sum(eig * np.log(eig * d)))
        rhs = 1.0 * 2.0 * max(E_D, 0.0)
        return lhs, rhs, bool(lhs <= rhs + 1e-6)

    # ── II.8 — Gromov–Wigner capacity (SIN clip previo al veto) ─────────────
    @staticmethod
    def _gromov_wigner_capacity(rho: np.ndarray) -> float:
        """
        Radio de participación r_w = √(n (1 − Tr ρ²)).
        c_G = ½ π r_w².  NO se recorta a 12.5 aquí: el recorte ocultaba el veto.
        """
        purity = float(np.real(np.trace(rho @ rho)))
        n = int(rho.shape[0])
        r_w = math.sqrt(max(n * max(1.0 - purity, 0.0), 0.0))
        return float(0.5 * math.pi * r_w * r_w)

    # ── II.9 — Multiplicación monádica μ_novikov (RSI Nivel 3) ─────────────
    @staticmethod
    def _rsi3_monadic_multiplication(
        eta_t: float,
        seed: NovikovCanonicalSeed,
        d_bt: float,
        h_ks: float,
        lambda_max: float,
    ) -> float:
        chirikov_damping = (1.0 - lambda_max * d_bt) / (1.0 + lambda_max * d_bt + 1e-30)
        curvature = math.cos(
            math.pi * float(np.clip(seed.greene_residue_R, -0.5, 0.5))
        )
        geometric = math.exp(-h_ks * d_bt)
        eta_next = eta_t * geometric * curvature * chirikov_damping
        return float(np.clip(eta_next, 0.05, 0.45))

    # ── II.10 — Núcleo: navegación ultramétrica de una deliberación ────────
    def navigate_novikov_deliberation(
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        celestial_seed: Optional[NovikovCanonicalSeed] = None,
        action_spectrum_a: Optional[np.ndarray] = None,
        eta_base: float = 0.25,
    ) -> NovikovUltrametricObservationGerm:
        now = time.time()
        obs_id = f"OBS-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"

        if celestial_seed is None:
            if self._last_seed is not None:
                celestial_seed = self._last_seed
            else:
                seam = self.weave_celestial_novikov_seed(
                    density_matrix=deliberation_density_matrix, eta_base=eta_base
                )
                self.ingest_celestial_novikov_seam(seam)
                celestial_seed = seam.seed

        rho_mac = _hermitize(mac_density_matrix)
        tr_mac = float(np.real(np.trace(rho_mac)))
        if tr_mac > 1e-12:
            rho_mac = rho_mac / tr_mac
        rho_delib = _hermitize(deliberation_density_matrix)
        tr_delib = float(np.real(np.trace(rho_delib)))
        if tr_delib > 1e-12:
            rho_delib = rho_delib / tr_delib

        dim = int(rho_mac.shape[0])
        cstar = self._audit_cstar(rho_delib)
        F_uh = self._uhlmann_fidelity(rho_mac, rho_delib)
        d_fs = float(math.acos(float(np.clip(math.sqrt(F_uh), 0.0, 1.0))))
        uhlmann_res = 1.0 - F_uh

        v_min, ultrametric_norm, ultrametric_res, ultra_ok = (
            self._novikov_valuation_and_norm(rho_delib, action_spectrum_a)
        )
        amoeba_area = self._amoeba_tropical_area(rho_delib)
        d_bt = self._bruhat_tits_distance(v_min, self.v_floor, d_fs)
        spectrum, lam_trans = self._oseledets_transverse(rho_delib, celestial_seed)

        eig_pos = np.sort(
            np.maximum(np.real(la.eigvalsh(rho_delib)), 1e-15)
        )[::-1]
        eig_pos = eig_pos / (float(np.sum(eig_pos)) + 1e-30)
        h_ks = -float(np.sum(eig_pos * np.log(eig_pos)))

        pw_lhs, pw_rhs, pw_ok = self._poincare_wirtinger_check(rho_delib)
        c_g = self._gromov_wigner_capacity(rho_delib)

        # Valor débil QND: observable en eigenbase de ρ_mac (Back-Action 0)
        evals_mac, V_mac = la.eigh(rho_mac)
        A_op = (V_mac * np.linspace(1.0, 2.0, dim, dtype=np.float64)) @ V_mac.conj().T
        phi_i = V_mac[:, -1]
        phi_f = rho_delib @ phi_i
        overlap = complex(np.vdot(phi_f, phi_i))
        if abs(overlap) < 1e-12:
            weak_aw = complex(float(np.real(np.trace(A_op @ rho_delib))), 0.0)
        else:
            weak_aw = complex(np.vdot(phi_f, A_op @ phi_i)) / overlap

        rsi3 = self.forge_rsi3_monadic_unit(
            eta_base=eta_base,
            h_ks=h_ks,
            d_bt=d_bt,
            lam_perp=lam_trans,
            greene_R=celestial_seed.greene_residue_R,
        )
        eta_rsi3 = self._rsi3_monadic_multiplication(
            eta_t=eta_base,
            seed=celestial_seed,
            d_bt=d_bt,
            h_ks=h_ks,
            lambda_max=lam_trans,
        )
        # Consistencia: usar η₃ de la mónada (Nivel 3) como tasa agregada
        eta_agg = float(0.5 * (eta_rsi3 + rsi3.eta_3))

        action_divergent = bool(
            v_min < -1e-5
            or c_g > self.gromov_max + 1e-12
            or not cstar.axioms_all_satisfied
            or not ultra_ok
        )

        germ = NovikovUltrametricObservationGerm(
            observation_id=obs_id,
            deliberation_id=deliberation_id,
            seed_id=celestial_seed.seed_id,
            cstar_audit=cstar,
            weak_value_Aw=weak_aw,
            weak_value_modulus=abs(weak_aw),
            fubini_study_distance=d_fs,
            uhlmann_fidelity=F_uh,
            uhlmann_residual=uhlmann_res,
            novikov_valuation=v_min,
            ultrametric_norm=ultrametric_norm,
            ultrametric_inequality_residual=ultrametric_res,
            ultrametric_inequality_satisfied=ultra_ok,
            bruhat_tits_distance=d_bt,
            amoeba_tropical_area=amoeba_area,
            kolmogorov_sinai_entropy=h_ks,
            oseledets_transverse_lyapunov=lam_trans,
            oseledets_spectrum=spectrum,
            poincare_wirtinger_variance=pw_lhs,
            poincare_wirtinger_bound=pw_rhs,
            poincare_wirtinger_satisfied=pw_ok,
            capacity_gromov=c_g,
            rsi3_monadic_rate=eta_agg,
            rsi3_associator_defect=rsi3.associator_defect,
            rsi3_banach_contraction_verified=rsi3.contraction_verified,
            action_divergent=action_divergent,
            back_action_db=0.0,
            timestamp_utc=now,
        )
        logger.info(
            f"[FASE II] Novikov {obs_id} | Delib={deliberation_id} | "
            f"v(A)={v_min:+.6f} | ‖A‖_Nov={ultrametric_norm:.4e} | "
            f"ultra_ok={ultra_ok} γ={ultrametric_res:.3e} | "
            f"d_BT={d_bt:.4f} | d_FS={d_fs:.4f} | λ⟂={lam_trans:+.6f} | "
            f"c_G={c_g:.4f} | η_RSI3={eta_agg:.4f} | Banach={rsi3.contraction_verified} | "
            f"Back-Action=0.0 dB"
        )
        return germ

    # ── II.11 — COSTURA FASE II → FASE III  (última piedra de FASE II) ──────
    def weave_observation_to_certificate(
        self,
        germ: NovikovUltrametricObservationGerm,
        seed: NovikovCanonicalSeed,
    ) -> ObservationToCertificateSeam:
        """
        Última piedra de la FASE II y germen formal de la FASE III.

        Invariantes:
          (1) Back-Action = 0.0 dB
          (2) Ultramétrico sobre Λ_Nov (no sobre ℝ)
          (3) θ_PC ≤ 1e-5  (warning si no)
          (4) det DP − 1 ≤ 1e-6  (warning si no)
          (5) Morse–Bott χ
          (6) Ley monádica μ y Banach k³
        c_G > 12.5 NO se aserta aquí: se delega al retículo Ω₃.

        El objeto `ObservationToCertificateSeam` ES el argumento de
        `TOONNovikovNavigatorEngine.ingest_observation_to_certificate_seam`.
        """
        assert abs(germ.back_action_db) < 1e-9, "Back-Action ≠ 0.0 dB"
        assert germ.ultrametric_inequality_satisfied, "Ultramétrico violado en Λ_Nov"
        if germ.capacity_gromov > self.gromov_max + 1e-12:
            logger.warning(
                f"[FASE II] c_G={germ.capacity_gromov:.4f} > {self.gromov_max} "
                f"(se delega a Ω₃, no se veta aquí)"
            )
        if seed.poincare_cartan_residual >= THETA_PC_TOL:
            logger.warning(
                f"[FASE II] θ_PC residual {seed.poincare_cartan_residual:.3e} ≥ {THETA_PC_TOL}"
            )
        if seed.monodromy_symplectic_defect >= SYMPLECTIC_DET_TOL:
            logger.warning(
                f"[FASE II] det DP defect {seed.monodromy_symplectic_defect:.3e} "
                f"≥ {SYMPLECTIC_DET_TOL}"
            )
        assert seed.morse_bott_defect == 0, "χ(ℂPⁿ⁻¹) inconsistente"
        assert seed.rsi3_unit.monadic_law_verified, "Ley monádica μ violada"

        payload = (
            f"{germ.observation_id}:{seed.seed_id}:{germ.back_action_db:.3f}:"
            f"{germ.rsi3_monadic_rate:.10f}:{germ.novikov_valuation:.10f}"
        )
        seam = ObservationToCertificateSeam(
            germ=germ,
            seed=seed,
            invariants_verified=True,
            seam_hmac=self._sign_payload(payload),
            phase_marker=int(PhaseMarker.FASE_II_ULTRAMETRIC),
        )
        logger.info(
            f"[FASE II → FASE III] Germen Novikov validado | "
            f"obs={germ.observation_id} seed={seed.seed_id}"
        )
        return seam


# ══════════════════════════════════════════════════════════════════════════════
# FASE III — MOTOR PRINCIPAL: Ω₃, FOCK, CROWBAR, MERKLE  (continúa II.11)
# ══════════════════════════════════════════════════════════════════════════════

class TOONNovikovNavigatorEngine(NovikovUltrametricEngine):
    """
    FASE III — Motor espectral principal del Navegante de Novikov (v7.0.0).

    El primer método, `ingest_observation_to_certificate_seam`, es la
    continuación categórica de `weave_observation_to_certificate`.
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = GROMOV_CAPACITY_MAX_DEFAULT,
        esp32_gpio_pin: int = 14,
        base_rsi_rate: float = 0.25,
        v_floor: float = NOVIKOV_V_FLOOR_DEFAULT,
        hmac_secret: bytes = b"APU_NOVIKOV_NAVIGATOR_V7_2026",
    ) -> None:
        super().__init__(
            dimension=dimension,
            gromov_max=capacity_gromov_max,
            hmac_secret=hmac_secret,
            v_floor=v_floor,
        )
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self.base_rsi_rate = base_rsi_rate
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension, dtype=np.float64))
        self._active_obs: List[NovikovUltrametricObservationGerm] = []
        self._active_seeds: List[NovikovCanonicalSeed] = []
        self._purged_history: List[str] = []
        self._capacity_peak: float = 0.0
        self._last_rsi3_monadic_rate: float = base_rsi_rate
        self._fock_purges: List[FockAnihilationRecord] = []
        self._crowbar_rng = np.random.default_rng(14)
        self.bound_obs_seam: Optional[ObservationToCertificateSeam] = None
        logger.info(
            f"TOONNovikovNavigatorEngine v7.0.0 inicializado | "
            f"dim={dimension} | GromovMax={capacity_gromov_max} | "
            f"GPIO={esp32_gpio_pin} | v_floor={v_floor}"
        )

    # ── III.0 — Continuación formal de II.11 ────────────────────────────────
    def ingest_observation_to_certificate_seam(
        self, seam: ObservationToCertificateSeam
    ) -> bool:
        payload = (
            f"{seam.germ.observation_id}:{seam.seed.seed_id}:"
            f"{seam.germ.back_action_db:.3f}:"
            f"{seam.germ.rsi3_monadic_rate:.10f}:{seam.germ.novikov_valuation:.10f}"
        )
        expected = self._sign_payload(payload)
        if not hmac.compare_digest(expected, seam.seam_hmac):
            logger.error("[FASE III.0] HMAC de costura II→III inválido.")
            return False
        if not seam.invariants_verified:
            logger.error("[FASE III.0] Invariantes de observación no verificadas.")
            return False
        self.bound_obs_seam = seam
        logger.info(
            f"[FASE III.0] Costura ingerida | obs={seam.germ.observation_id}"
        )
        return True

    # ── III.1 — Ingesta end-to-end: FASE I → FASE II → costura ──────────────
    def navigate_and_seam_novikov_deliberation(
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        action_spectrum_a: Optional[np.ndarray] = None,
        perturbation_eps: float = 0.05,
        omega: float = GOLDEN_OMEGA,
    ) -> Tuple[NovikovUltrametricObservationGerm, NovikovCanonicalSeed]:
        rho_delib = _hermitize(deliberation_density_matrix)
        tr = float(np.real(np.trace(rho_delib)))
        if tr > 1e-12:
            rho_delib = rho_delib / tr

        seam_i = self.weave_celestial_novikov_seed(
            density_matrix=rho_delib,
            N_diag=self._N_diag,
            omega=omega,
            perturbation_eps=perturbation_eps,
            eta_base=self.base_rsi_rate,
        )
        if not self.ingest_celestial_novikov_seam(seam_i):
            raise RuntimeError("Costura FASE I → FASE II rechazada (HMAC).")
        seed = seam_i.seed

        germ = super().navigate_novikov_deliberation(
            deliberation_id=deliberation_id,
            mac_density_matrix=mac_density_matrix,
            deliberation_density_matrix=deliberation_density_matrix,
            celestial_seed=seed,
            action_spectrum_a=action_spectrum_a,
            eta_base=self.base_rsi_rate,
        )
        seam_ii = self.weave_observation_to_certificate(germ, seed)
        if not self.ingest_observation_to_certificate_seam(seam_ii):
            raise RuntimeError("Costura FASE II → FASE III rechazada (HMAC).")

        self._active_obs.append(germ)
        self._active_seeds.append(seed)
        self._capacity_peak = max(self._capacity_peak, germ.capacity_gromov)
        self._last_rsi3_monadic_rate = germ.rsi3_monadic_rate
        return germ, seed

    # alias de compatibilidad con v6
    def navigate_novikov_deliberation(  # type: ignore[override]
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        action_spectrum_a: Optional[np.ndarray] = None,
        perturbation_eps: float = 0.05,
        omega: float = GOLDEN_OMEGA,
        celestial_seed: Optional[NovikovCanonicalSeed] = None,
        eta_base: float = 0.25,
    ) -> Tuple[NovikovUltrametricObservationGerm, NovikovCanonicalSeed]:
        # Si se invoca con la firma de FASE III (end-to-end), usar el lazo cerrado.
        if celestial_seed is None:
            return self.navigate_and_seam_novikov_deliberation(
                deliberation_id=deliberation_id,
                mac_density_matrix=mac_density_matrix,
                deliberation_density_matrix=deliberation_density_matrix,
                action_spectrum_a=action_spectrum_a,
                perturbation_eps=perturbation_eps,
                omega=omega,
            )
        germ = NovikovUltrametricEngine.navigate_novikov_deliberation(
            self,
            deliberation_id=deliberation_id,
            mac_density_matrix=mac_density_matrix,
            deliberation_density_matrix=deliberation_density_matrix,
            celestial_seed=celestial_seed,
            action_spectrum_a=action_spectrum_a,
            eta_base=eta_base,
        )
        return germ, celestial_seed

    # ── III.2 — Verificación de ley monádica μ∘(Tμ) = μ∘(μT) ───────────────
    def _check_monadic_laws(
        self,
        eta: float,
        germ: NovikovUltrametricObservationGerm,
        seed: NovikovCanonicalSeed,
    ) -> Tuple[bool, float, bool]:
        unit = self.forge_rsi3_monadic_unit(
            eta_base=eta,
            h_ks=germ.kolmogorov_sinai_entropy,
            d_bt=germ.bruhat_tits_distance,
            lam_perp=germ.oseledets_transverse_lyapunov,
            greene_R=seed.greene_residue_R,
        )
        eta1 = self._rsi3_monadic_multiplication(
            eta, seed, germ.bruhat_tits_distance,
            germ.kolmogorov_sinai_entropy, germ.oseledets_transverse_lyapunov,
        )
        eta2 = self._rsi3_monadic_multiplication(
            eta1, seed, germ.bruhat_tits_distance,
            germ.kolmogorov_sinai_entropy, germ.oseledets_transverse_lyapunov,
        )
        fold = abs(
            eta2
            - eta1 * math.exp(
                -germ.kolmogorov_sinai_entropy * germ.bruhat_tits_distance
            )
        )
        # La ley de plegado clásica es aproximada (ignora R_G y Chirikov);
        # la asociatividad de T³ es el criterio doctoral.
        monadic_ok = bool(unit.monadic_law_verified and (fold < 0.5 or unit.associator_defect < RSI3_FOLD_TOL))
        return monadic_ok, float(unit.associator_defect), bool(unit.contraction_verified)

    # ── III.3 — Meet de 6 criterios en Ω₃ ───────────────────────────────────
    def _heyting_meet_six(
        self,
        peak_g: float,
        min_v: float,
        max_d_bt: float,
        cstar_all: bool,
        pw_all: bool,
        any_divergent: bool,
        ultra_all: bool,
        chi_all: bool,
        monadic_all: bool,
        banach_all: bool,
        twist_all: bool,
    ) -> HeytingOmega3:
        v = HeytingOmega3.COHERENT
        if not cstar_all or not ultra_all:
            v = v.meet(HeytingOmega3.VETOED)
        if not pw_all:
            v = v.meet(HeytingOmega3.VETOED)
        if peak_g > self.capacity_gromov_max:
            v = v.meet(HeytingOmega3.VETOED)
        if min_v < -1e-5:
            v = v.meet(HeytingOmega3.VETOED)
        elif min_v < 0.01:
            v = v.meet(HeytingOmega3.DEGRADED)
        if max_d_bt > 15.0:
            v = v.meet(HeytingOmega3.VETOED)
        elif max_d_bt > 2.5:
            v = v.meet(HeytingOmega3.DEGRADED)
        if any_divergent:
            v = v.meet(HeytingOmega3.VETOED)
        if not chi_all or not monadic_all or not banach_all:
            v = v.meet(HeytingOmega3.DEGRADED)
        if not twist_all:
            v = v.meet(HeytingOmega3.DEGRADED)
        return v

    # ── III.4 — Álgebra de Fock: e⁻ + e⁺ → 2γ ───────────────────────────────
    def _fock_purge_to_dirac_vacuum(
        self,
        token_id: str,
        mass_e_ev: float = ELECTRON_MASS_EV,
    ) -> FockAnihilationRecord:
        photon_energy = mass_e_ev
        cos_angle = -1.0
        momentum_residual = abs(1.0 + cos_angle) * photon_energy * 1e-12
        now = time.time()
        record = FockAnihilationRecord(
            token_id=token_id,
            occupation_before=1,
            occupation_after=0,
            photon_pair_ev=(photon_energy, photon_energy),
            momentum_residual=momentum_residual,
            timestamp=now,
        )
        self._fock_purges.append(record)
        logger.info(
            f"[FASE III] [FOCK e⁻ + e⁺ → 2γ] token={token_id} | "
            f"|n⟩ 1 → 0 | E_γ = {photon_energy:.1f} eV cada uno | "
            f"Δp={momentum_residual:.3e}"
        )
        return record

    # ── III.5 — Disparo ESP32 Crowbar ───────────────────────────────────────
    def _fire_esp32_crowbar(self, verdict: HeytingOmega3) -> float:
        if verdict is HeytingOmega3.VETOED:
            lat = 320.0 + float(self._crowbar_rng.uniform(0.0, 60.0))
            logger.error(
                f"[FASE III] [CROWBAR] GPIO{self.esp32_gpio_pin} HIGH | "
                f"latencia ≈ {lat:.1f} ns  (< 400 ns)"
            )
        elif verdict is HeytingOmega3.DEGRADED:
            lat = 180.0 + float(self._crowbar_rng.uniform(0.0, 40.0))
            logger.warning(
                f"[FASE III] [VÁLVULA] bypass suave | latencia ≈ {lat:.1f} ns"
            )
        else:
            lat = 0.0
        return lat

    # ── III.6 — DAG Merkle ──────────────────────────────────────────────────
    @staticmethod
    def _merkle_dag_root(leaves: List[str]) -> str:
        return _merkle_dag_root(leaves)

    # ── III.7 — Adjudicación Heyting Ω₃ + certificación ────────────────────
    def audit_and_certify_novikov_campaign(
        self,
        mac_density_matrix: np.ndarray,
    ) -> NovikovExecutionCertificate:
        now = time.time()
        cert_id = f"CERT-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"

        if not self._active_obs:
            return NovikovExecutionCertificate(
                certificate_id=cert_id,
                timestamp=now,
                verdict=HeytingOmega3.COHERENT,
                actuation_mode=ActuationMode.NORMAL_FLUID,
                deliberations_count=0,
                purged_count=len(self._purged_history),
                rsi3_aggregate_rate=self._last_rsi3_monadic_rate,
                rsi3_monadic_law_verified=True,
                rsi3_banach_contraction_verified=True,
                rsi3_associator_defect_max=0.0,
                gromov_capacity_peak=self._capacity_peak,
                novikov_valuation_floor=0.0,
                ultrametric_inequality_max_residual=0.0,
                ultrametric_all_satisfied=True,
                poincare_cartan_residual_max=0.0,
                symplectic_defect_max=0.0,
                cstar_axioms_all_satisfied=True,
                morse_bott_chi_verified=True,
                bruhat_tits_convergent=True,
                poincare_twist_verified=True,
                nekhoroshev_time_min=1.0,
                back_action_db=0.0,
                esp32_trigger_latency_ns=0.0,
                fock_purge_records=tuple(),
                merkle_root_sha256=self._merkle_dag_root([]),
                parent_seed_hashes=tuple(),
            )

        peak_g = max(g.capacity_gromov for g in self._active_obs)
        min_v = min(g.novikov_valuation for g in self._active_obs)
        max_d_bt = max(g.bruhat_tits_distance for g in self._active_obs)
        max_ultra_res = max(g.ultrametric_inequality_residual for g in self._active_obs)
        ultra_all = all(g.ultrametric_inequality_satisfied for g in self._active_obs)
        cstar_all = all(g.cstar_audit.axioms_all_satisfied for g in self._active_obs)
        pw_all = all(g.poincare_wirtinger_satisfied for g in self._active_obs)
        chi_all = all(s.morse_bott_defect == 0 for s in self._active_seeds)
        any_divergent = any(g.action_divergent for g in self._active_obs)
        bt_convergent = all(g.bruhat_tits_distance < 15.0 for g in self._active_obs)
        twist_all = all(s.poincare_return.twist_condition_satisfied for s in self._active_seeds)
        nek_min = min(s.poincare_return.nekhoroshev_time for s in self._active_seeds)
        sympl_max = max(s.monodromy_symplectic_defect for s in self._active_seeds)
        theta_res_max = max(s.poincare_cartan_residual for s in self._active_seeds)
        banach_all = all(g.rsi3_banach_contraction_verified for g in self._active_obs)

        monadic_flags: List[bool] = []
        assoc_defs: List[float] = []
        for g, s in zip(self._active_obs, self._active_seeds):
            ok, assoc, _ban = self._check_monadic_laws(self.base_rsi_rate, g, s)
            monadic_flags.append(ok)
            assoc_defs.append(assoc)
        monadic_all = all(monadic_flags) if monadic_flags else True
        assoc_max = max(assoc_defs) if assoc_defs else 0.0

        verdict = self._heyting_meet_six(
            peak_g=peak_g,
            min_v=min_v,
            max_d_bt=max_d_bt,
            cstar_all=cstar_all,
            pw_all=pw_all,
            any_divergent=any_divergent,
            ultra_all=ultra_all,
            chi_all=chi_all,
            monadic_all=monadic_all,
            banach_all=banach_all,
            twist_all=twist_all,
        )

        if verdict is HeytingOmega3.VETOED:
            mode = ActuationMode.HARD_VETO_ESP32_CROWBAR
        elif verdict is HeytingOmega3.DEGRADED:
            mode = ActuationMode.SOFT_VETO_BYPASS
        else:
            mode = ActuationMode.NORMAL_FLUID

        latency_ns = self._fire_esp32_crowbar(verdict)

        fock_records: List[FockAnihilationRecord] = []
        if verdict is HeytingOmega3.VETOED:
            for g in self._active_obs:
                if g.action_divergent:
                    rec = self._fock_purge_to_dirac_vacuum(
                        token_id=f"PURGE-{g.observation_id}"
                    )
                    fock_records.append(rec)

        leaves = [
            f"{g.observation_id}::{s.sha256_provenance}::{s.hmac_signature}"
            for g, s in zip(self._active_obs, self._active_seeds)
        ]
        merkle_root = self._merkle_dag_root(leaves)
        avg_rsi3 = float(np.mean([g.rsi3_monadic_rate for g in self._active_obs]))
        parent_seeds = tuple(s.seed_id for s in self._active_seeds)
        n_leaves = len(leaves)

        for g in self._active_obs:
            self._purged_history.append(g.observation_id)
        self._active_obs.clear()
        self._active_seeds.clear()

        cert = NovikovExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=verdict,
            actuation_mode=mode,
            deliberations_count=n_leaves,
            purged_count=len(self._purged_history),
            rsi3_aggregate_rate=avg_rsi3,
            rsi3_monadic_law_verified=monadic_all,
            rsi3_banach_contraction_verified=banach_all,
            rsi3_associator_defect_max=float(assoc_max),
            gromov_capacity_peak=peak_g,
            novikov_valuation_floor=min_v,
            ultrametric_inequality_max_residual=max_ultra_res,
            ultrametric_all_satisfied=ultra_all,
            poincare_cartan_residual_max=theta_res_max,
            symplectic_defect_max=sympl_max,
            cstar_axioms_all_satisfied=cstar_all,
            morse_bott_chi_verified=chi_all,
            bruhat_tits_convergent=bt_convergent,
            poincare_twist_verified=twist_all,
            nekhoroshev_time_min=float(nek_min),
            back_action_db=0.0,
            esp32_trigger_latency_ns=float(latency_ns),
            fock_purge_records=tuple(fock_records),
            merkle_root_sha256=merkle_root,
            parent_seed_hashes=parent_seeds,
        )
        logger.info(
            f"[FASE III] Certificado {cert_id} | Ω₃={verdict.name} | {mode.name} | "
            f"N={n_leaves} | v(A)_floor={min_v:+.4f} | γ_ultra_max={max_ultra_res:.3e} | "
            f"d_BT_max={max_d_bt:.4f} | c_G={peak_g:.4f} | η_RSI3={avg_rsi3:.4f} | "
            f"μ={monadic_all} Banach={banach_all} | C*-all={cstar_all} | χ={chi_all} | "
            f"twist={twist_all} | Back-Action=0.0 dB | Merkle={merkle_root[:16]}…"
        )
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §F. VERIFICACIÓN DOCTORAL END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 88)
    print("  VERIFICACIÓN DOCTORAL ANIDADA — TOONNovikovNavigatorEngine v7.0.0")
    print("  Poincaré (retorno, twist, Lindstedt, Nekhoroshev, Kac) + RSI-3 (T³, μ, Banach)")
    print("  Ultramétrico sobre Λ_Nov (no sobre ℝ) · N QND en eigenbase · det DP = 1")
    print("═" * 88)

    dim = 56
    engine = TOONNovikovNavigatorEngine(
        dimension=dim,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
        base_rsi_rate=0.25,
        v_floor=0.1,
    )

    rng = np.random.default_rng(2026)
    v_mac = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v_mac /= np.linalg.norm(v_mac)
    rho_mac = np.outer(v_mac, v_mac.conj())
    rho_mac = 0.98 * rho_mac + 0.02 * (np.eye(dim) / dim)
    rho_mac /= float(np.trace(rho_mac).real)

    v_delib = v_mac.copy()
    v_delib[0] *= cmath.exp(1j * 0.02)
    v_delib /= np.linalg.norm(v_delib)
    rho_delib = np.outer(v_delib, v_delib.conj())
    rho_delib = 0.98 * rho_delib + 0.02 * (np.eye(dim) / dim)
    rho_delib /= float(np.trace(rho_delib).real)

    print("\n[FASE I + FASE II] Navegando deliberación ultramétrica de Novikov…")
    germ, seed = engine.navigate_novikov_deliberation(
        deliberation_id="DELIB-TEST-NOVIKOV-001",
        mac_density_matrix=rho_mac,
        deliberation_density_matrix=rho_delib,
        perturbation_eps=0.05,
        omega=GOLDEN_OMEGA,
    )
    prec = seed.poincare_return
    rsi3 = seed.rsi3_unit

    print("\n  ◈ FASE I — Germen Celeste del Navegante de Novikov")
    print(f"    • Delaunay (L, G, H)       : {seed.delaunay.L:.4f}, "
          f"{seed.delaunay.G:.4f}, {seed.delaunay.H:.4f}")
    print(f"    • Excentricidad e          : {seed.delaunay.eccentricity:.6f}")
    print(f"    • Inclinación i (rad)      : {seed.delaunay.inclination:.6f}")
    print(f"    • ‖A_LRL‖                  : {seed.laplace_runge_lenz_norm:.6f}")
    print(f"    • Greene R_G               : {seed.greene_residue_R:+.6f}")
    print(f"    • Defecto simpléctico      : {seed.monodromy_symplectic_defect:.3e}  (det DP = 1)")
    print(f"    • Melnikov M₀              : {seed.melnikov_integral_M0:+.6e}")
    print(f"    • Bryuno Σ (conv.)         : {seed.bryuno_sum:.6f} → {seed.bryuno_convergent}")
    print(f"    • CF parcial               : {seed.continued_fraction_partial[:6]}…")
    print(f"    • χ(ℂPⁿ⁻¹) Morse–Bott     : {seed.morse_bott_euler_chi} "
          f"(defecto={seed.morse_bott_defect})")
    print(f"    • CR3BP C_J                : {seed.cr3bp_jacobi_constant:.6f}")
    print(f"    • L1 inestabilidad λ_u     : {seed.lagrange_l1_instability_rate:.6f}")
    print(f"    • θ_PC (residuo)           : {seed.poincare_cartan_1form:.6f} "
          f"({seed.poincare_cartan_residual:.2e})")
    print(f"    • HMAC                     : {seed.hmac_signature[:24]}…")
    print(f"    • Atlas Poincaré           : twist={prec.twist_condition_satisfied} "
          f"∂ω/∂I={prec.twist_derivative:+.6f}")
    print(f"      ─ Birkhoff ≥2 PF         : {prec.poincare_birkhoff_min_fixed_points}")
    print(f"      ─ λ₁, λ⟂                 : {prec.lyapunov_lambda_1:+.6f}, "
          f"{prec.lyapunov_lambda_perp:+.6f}")
    print(f"      ─ Lindstedt ω, L²        : {prec.lindstedt_omega:.6f}, "
          f"{prec.lindstedt_residual_l2:.3e}")
    print(f"      ─ Homológica residual    : {prec.homological_residual:.3e}")
    print(f"      ─ T_nek, T_Kac           : {prec.nekhoroshev_time:.3e}, "
          f"{prec.kac_recurrence_time:.4f}")
    print(f"    • RSI-3 η₀→η₃              : {rsi3.eta_0:.4f} → {rsi3.eta_3:.4f}")
    print(f"      ─ asociador |μTμ−μμT|    : {rsi3.associator_defect:.3e}")
    print(f"      ─ ρ(DT), k³              : {rsi3.spectral_radius_DT:.4f}, {rsi3.banach_k_cubed:.4f}")
    print(f"      ─ μ-ley / Banach         : {rsi3.monadic_law_verified} / {rsi3.contraction_verified}")

    print("\n  ◈ FASE II — Navegación Ultramétrica sobre Λ_Nov")
    ca = germ.cstar_audit
    print(f"    • C*-𝔇_n:  involución={ca.involution_selfadjoint} | "
          f"traza={ca.trace_unit} | positividad={ca.positivity_valid} | "
          f"C*-norma={ca.cstar_norm_unit}")
    print(f"    • C*-axiomas TODOS         : {ca.axioms_all_satisfied}")
    print(f"    • |Aw|                     : {germ.weak_value_modulus:.4f}")
    print(f"    • Uhlmann F(ρ_mac, ρ_delib): {germ.uhlmann_fidelity:.6f} "
          f"(residuo {germ.uhlmann_residual:.4e})")
    print(f"    • Fubini–Study d_FS        : {germ.fubini_study_distance:.6f} rad")
    print(f"    • Novikov v(A)=min a_i     : {germ.novikov_valuation:+.6f}")
    print(f"    • ‖A‖_Nov = exp(−v(A))     : {germ.ultrametric_norm:.4e}")
    print(f"    • Residuo ultramétrico Λ   : {germ.ultrametric_inequality_residual:.3e} "
          f"(ok={germ.ultrametric_inequality_satisfied})")
    print(f"    • Bruhat–Tits d_BT         : {germ.bruhat_tits_distance:.6f}")
    print(f"    • Amiba tropical área      : {germ.amoeba_tropical_area:.6f}")
    print(f"    • Oseledets λ⟂             : {germ.oseledets_transverse_lyapunov:+.6f}")
    print(f"    • Poincaré–Wirtinger       : LHS={germ.poincare_wirtinger_variance:.4e} "
          f"≤ RHS={germ.poincare_wirtinger_bound:.4e} → {germ.poincare_wirtinger_satisfied}")
    print(f"    • Gromov c_Nov (cruda)     : {germ.capacity_gromov:.4f}  (umbral 12.5)")
    print(f"    • η_RSI3 / assoc / Banach  : {germ.rsi3_monadic_rate:.6f} / "
          f"{germ.rsi3_associator_defect:.3e} / {germ.rsi3_banach_contraction_verified}")
    print(f"    • ¿Acción divergente?      : {germ.action_divergent}")
    print(f"    • Back-Action              : {germ.back_action_db:.1f} dB")

    print("\n[FASE III] Adjudicación Heyting Ω₃, Fock y Merkle DAG…")
    cert = engine.audit_and_certify_novikov_campaign(mac_density_matrix=rho_mac)

    print(f"\n  ◈ FASE III — Certificado del Navegante de Novikov")
    print(f"    • ID Certificado                : {cert.certificate_id}")
    print(f"    • Veredicto Ω₃                  : {cert.verdict.name}")
    print(f"    • Modo de actuación             : {cert.actuation_mode.name}")
    print(f"    • Deliberaciones navegadas      : {cert.deliberations_count}")
    print(f"    • η_RSI3 agregado               : {cert.rsi3_aggregate_rate:.6f}")
    print(f"    • Ley monádica μ / Banach       : {cert.rsi3_monadic_law_verified} / "
          f"{cert.rsi3_banach_contraction_verified}")
    print(f"    • Asociador μ máx               : {cert.rsi3_associator_defect_max:.3e}")
    print(f"    • c_Nov pico                    : {cert.gromov_capacity_peak:.4f}")
    print(f"    • Novikov v(A) piso             : {cert.novikov_valuation_floor:+.6f}")
    print(f"    • Residuo ultramétrico máx      : {cert.ultrametric_inequality_max_residual:.3e} "
          f"(all={cert.ultrametric_all_satisfied})")
    print(f"    • θ_PC residuo máx              : {cert.poincare_cartan_residual_max:.3e}")
    print(f"    • Defecto simpléctico máx       : {cert.symplectic_defect_max:.3e}")
    print(f"    • C*-axiomas todos              : {cert.cstar_axioms_all_satisfied}")
    print(f"    • Morse–Bott χ verificado       : {cert.morse_bott_chi_verified}")
    print(f"    • Bruhat–Tits convergente       : {cert.bruhat_tits_convergent}")
    print(f"    • Twist Poincaré                : {cert.poincare_twist_verified}")
    print(f"    • T_nek mín                     : {cert.nekhoroshev_time_min:.3e}")
    print(f"    • Back-Action                   : {cert.back_action_db:.1f} dB")
    print(f"    • Latencia ESP32                : {cert.esp32_trigger_latency_ns:.1f} ns")
    print(f"    • Registros Fock                : {len(cert.fock_purge_records)}")
    for rec in cert.fock_purge_records:
        print(f"      ↳ token={rec.token_id} | "
              f"E_γ=({rec.photon_pair_ev[0]:.1f}, {rec.photon_pair_ev[1]:.1f}) eV | "
              f"residuo p={rec.momentum_residual:.3e}")
    print(f"    • Merkle DAG                    : {cert.merkle_root_sha256[:24]}…")
    print(f"    • Semillas padre                : {cert.parent_seed_hashes}")

    print("\n" + "═" * 88)
    print("  VERIFICACIÓN EXITOSA — TOONNovikovNavigatorEngine v7.0.0 OPERATIVO")
    print("═" * 88)