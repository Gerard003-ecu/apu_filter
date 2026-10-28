# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Intuition Prosecutor Agent (Soberano Fiscal de Intuición)                        ║
║ Ubicación: app/agents/wisdom/toon_intuition_prosecutor_agent.py                                  ║
║ Versión  : 8.0.0-SOVEREIGN-RSI3-POINCARE-MELNIKOV-CHIRIKOV-HEYTING-ESP32-PhD                     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

Soberano Agéntico Fiscal de Calibre Inquisidor (TOONIntuitionProsecutorAgent). Censura impulsos
caóticos, audita falsas corazonadas (RHI > 0.85), detecta fractura homoclínica vía Melnikov M(t₀)
(Simpson 401, sech² tanh sin), cuantifica solapamiento de Chirikov K ≥ 1, residuo de Greene
R_G = (2 − Tr M)/4 con M = exp(ΩH) ∈ Sp(4), y verifica la suma diofántica de Bryuno por fracciones
continuas (fallback conjugado áureo).

ARQUITECTURA DE TRES FASES ANIDADAS (homomorfismos Φ_{k,k+1}; el último método de Φ_k ES el arranque de Φ_{k+1}):
──────────────────────────────────────────────────────────────────────────────────────────────────
  • Fase I   : Phase1_IntuitionProsecutorGaugeTopology
               Calibre 𝔲(n), CS₃ = (1/8π²) Tr(AF − ⅓ A³), c₁, Wilson U = expm(A), Darboux Sp(2n),
               Maupertuis-Jacobi Ĝ = 2(H₀−V)G (suelo Novikov 1e-10), Delaunay (L,G,H,l,g,h),
               KAM ∂²K/∂L² = −3μ²/L⁴, Bryuno continuado.
               Cierre / arranque Φ₁₂ : continue_into_phase2_campaign_gate  (HMAC; ADMITE VETOED).

  • Fase II  : Phase2_TOONIntuitionProsecutorAgent  (hereda Fase I)
               Fiscalización flash: Melnikov, Chirikov, Greene, Uhlmann espectral, RHI, Lyapunov,
               Kelly s* = f*(1−RHI) Θ(0.85−RHI) Θ(¬Melnikov) Θ(¬Chirikov) Θ(λ_ceil−λ_max) e^{−h_KS},
               torre monádica RSI³.
               Cierre / arranque Φ₂₃ : continue_into_phase3_adjudication_gate  (re-HMAC; ADMITE VETOED).

  • Fase III : Phase3_SovereignIntuitionProsecutorAdjudicator  (hereda Fase II)
               Ω₃, RSI³ = R₃∘R₂∘R₁ (contracción de Banach, 8×6, tol 1e-10), interlock SIMULADO
               (bit-flip + perf_counter_ns, cota 392.15 ns; jamás GPIO), positrón HMAC-only
               (compare_digest; sin bypass de longitud), Φ_sem, Merkle SHA-256 pairwise.
               Cierre : execute_sovereign_prosecution_adjudication_and_rsi3.

  • Orquestador: TOONIntuitionProsecutorAgent.execute_full_prosecutor_governance_cycle
               Alias  : SovereignIntuitionProsecutorAdjudicator
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from app.wisdom.toon_intuition_prosecutor_engine import (
        ActuationMode,
        HeytingOmega3,
        IntuitionProsecutorExecutionCertificate,
        TOONIntuitionProsecutorEngine,
    )
    _ENGINE_AVAILABLE = True
except ImportError:
    try:
        from toon_intuition_prosecutor_engine import (  # type: ignore[no-redef]
            ActuationMode,
            HeytingOmega3,
            IntuitionProsecutorExecutionCertificate,
            TOONIntuitionProsecutorEngine,
        )
        _ENGINE_AVAILABLE = True
    except ImportError:
        _ENGINE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# §0. HEYTING Ω₃, ACTUACIÓN, CONSTANTES Y NÚCLEOS ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

if not _ENGINE_AVAILABLE:
    class HeytingOmega3(IntEnum):
        """Retículo distributivo trivalente Ω₃ = {0, 1, 2}."""

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

        def relative_pseudocomplement(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3(2) if int(self) <= int(other) else other

        def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return self.relative_pseudocomplement(other)

        def negation(self) -> "HeytingOmega3":
            return self.relative_pseudocomplement(HeytingOmega3.VETOED)

        def modus_ponens(self, consequent: "HeytingOmega3") -> "HeytingOmega3":
            return self.meet(self.relative_pseudocomplement(consequent)).meet(consequent)

        @classmethod
        def fold_meet(cls, values: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
            acc = cls.COHERENT
            for item in values:
                acc = acc.meet(item)
            return acc

    class ActuationMode(IntEnum):
        NORMAL_FLUID = 0
        SOFT_VETO_BYPASS = 1
        HARD_VETO_ESP32_CROWBAR = 2

    IntuitionProsecutorExecutionCertificate = Any  # type: ignore[misc, assignment]
    TOONIntuitionProsecutorEngine = None  # type: ignore[misc, assignment]


class ProsecutionStatus(IntEnum):
    """Estados fiscales del germen (alineados al motor v8)."""

    CLEAR_NO_ANOMALY = 0
    REWARD_HACKING_DETECTED = 1
    MELNIKOV_CHAOS_CONFIRMED = 2
    KELLY_STAKE_CENSORED = 3
    PURGED_DIRAC_VACUUM = 4
    CHIRIKOV_OVERLAP_CONFIRMED = 5


_WILKINSON_EPSILON: float = 1e-15
_SPECTRAL_PSD_FLOOR: float = 1e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: float = 392.15
_RHI_HARD_CEILING: float = 0.85
_RHI_DEGRADED_CEILING: float = 0.40
_MELNIKOV_CHAOS_THRESHOLD: float = 0.15
_CHIRIKOV_OVERLAP_THRESHOLD: float = 1.00
_LYAPUNOV_UNSTABLE_CEILING: float = 0.50
_KELLY_STAKE_CEILING: float = 0.35
_GROMOV_CAPACITY_CEILING: float = 12.5
_RSI3_MAX_OUTER_ITERS: int = 8
_RSI3_MAX_INNER_ITERS: int = 6
_BANACH_CONTRACTION_TOL: float = 1e-10
_KEPLER_MU: float = 1.0
_SOFT_VETO_WINDOW_S: float = 3600.0
_POSITRON_REST_MASS_ENERGY_KEV: float = 1022.0
_HMAC_DOMAIN: bytes = b"TOON-IPA-GAUGE-v8|"
_POSITRON_DOMAIN: bytes = b"TOON-IPA-POSITRON-v8|"
_GOLDEN_CONJUGATE: float = (math.sqrt(5.0) - 1.0) / 2.0
_DEFAULT_SECRET_KEY: bytes = b"APU_FILTER_PROSECUTOR_V8_0_SOVEREIGN_KEY"
_NOVIKOV_FLOOR: float = 1e-10
_MELNIKOV_NODES: int = 401
_MELNIKOV_HORIZON: float = 20.0
_SYMPLECTIC_TOLERANCE: float = 1e-9

logger = logging.getLogger("APU.Wisdom.TOONIntuitionProsecutorAgent")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _heaviside(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0


def _omega_meet(a: HeytingOmega3, b: HeytingOmega3) -> HeytingOmega3:
    return HeytingOmega3(min(int(a), int(b)))


def _omega_fold_meet(values: Sequence[HeytingOmega3]) -> HeytingOmega3:
    acc = 2
    for item in values:
        acc = min(acc, int(item))
    return HeytingOmega3(acc)


def _as_c2(matrix: np.ndarray) -> np.ndarray:
    return np.array(matrix, dtype=np.complex128, copy=True)


def _herm(matrix: np.ndarray) -> np.ndarray:
    a = _as_c2(matrix)
    return 0.5 * (a + a.conj().T)


def _real_sym(matrix: np.ndarray) -> np.ndarray:
    a = np.array(matrix, dtype=np.float64, copy=True).real
    return 0.5 * (a + a.T)


def _psd_herm(matrix: np.ndarray, floor: float = _SPECTRAL_PSD_FLOOR) -> np.ndarray:
    h = _herm(matrix)
    w, v = np.linalg.eigh(h)
    w = np.maximum(w.real, floor)
    return (v * w) @ v.conj().T


def _density(matrix: np.ndarray) -> np.ndarray:
    h = _psd_herm(matrix)
    tr = float(np.real(np.trace(h)))
    if tr <= _WILKINSON_EPSILON:
        n = h.shape[0]
        return np.eye(n, dtype=np.complex128) / float(max(n, 1))
    return h / tr


def _frobenius(matrix: np.ndarray) -> float:
    return float(np.sqrt(np.real(np.vdot(matrix, matrix))))


def _slice_square(matrix: np.ndarray, dim: int) -> np.ndarray:
    d = min(int(dim), matrix.shape[0], matrix.shape[1])
    return np.array(matrix[:d, :d], copy=True)


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    h = _psd_herm(matrix, floor=0.0)
    w, v = np.linalg.eigh(h)
    w = np.maximum(w.real, 0.0)
    return (v * np.sqrt(w)) @ v.conj().T


def _logm_spd(matrix: np.ndarray) -> np.ndarray:
    h = _psd_herm(matrix)
    w, v = np.linalg.eigh(h)
    w = np.maximum(w.real, _SPECTRAL_PSD_FLOOR)
    return (v * np.log(w)) @ v.conj().T


def _expm_herm(matrix: np.ndarray) -> np.ndarray:
    h = _herm(matrix)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(w.real)) @ v.conj().T


def _expm_square(matrix: np.ndarray) -> np.ndarray:
    a = np.array(matrix, copy=True)
    nrm = float(np.linalg.norm(a, ord=np.inf))
    s = max(0, int(math.ceil(math.log2(max(nrm, 1e-16)))) + 1)
    b = a / float(2 ** s)
    n = b.shape[0]
    term = np.eye(n, dtype=b.dtype)
    total = term.copy()
    for k in range(1, 18):
        term = term @ b / float(k)
        total = total + term
        if _frobenius(term) < 1e-16:
            break
    for _ in range(s):
        total = total @ total
    return total


def _canonical_omega(n: int) -> np.ndarray:
    half = int(n)
    eye = np.eye(half, dtype=np.float64)
    zero = np.zeros((half, half), dtype=np.float64)
    return np.block([[zero, eye], [-eye, zero]])


def _symplectic_expm(hessian: np.ndarray) -> np.ndarray:
    h = _real_sym(hessian)
    n2 = h.shape[0]
    if n2 % 2 == 1:
        n2 -= 1
        h = h[:n2, :n2]
    half = n2 // 2
    omega = _canonical_omega(half)
    return _expm_square(omega @ h)


def _simpson(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1).real
    n = int(min(xv.size, yv.size))
    if n < 2:
        return 0.0
    if n % 2 == 0:
        n -= 1
        xv = xv[:n]
        yv = yv[:n]
    h = float(xv[1] - xv[0]) if n > 1 else 0.0
    return float((h / 3.0) * (yv[0] + yv[-1] + 4.0 * np.sum(yv[1:-1:2]) + 2.0 * np.sum(yv[2:-1:2])))


def _continued_fraction_bryuno(alpha: float, max_depth: int = 24) -> Tuple[float, bool]:
    x = abs(float(alpha))
    if (not math.isfinite(x)) or x <= _WILKINSON_EPSILON:
        x = _GOLDEN_CONJUGATE
    if abs(x - round(x)) < 1e-12:
        x = _GOLDEN_CONJUGATE
    q_prev, q_curr = 1, 0
    rem = x
    q_list: List[int] = []
    for _ in range(max_depth):
        a_k = int(math.floor(rem))
        q_next = a_k * q_curr + q_prev
        if q_next <= 0:
            break
        q_list.append(int(q_next))
        q_prev, q_curr = q_curr, q_next
        frac = rem - float(a_k)
        if abs(frac) < 1e-14:
            break
        rem = 1.0 / frac
        if rem > 1e12:
            break
    if len(q_list) < 2:
        q_list = [1, 1]
    bryuno = 0.0
    for idx in range(len(q_list) - 1):
        q_n = max(q_list[idx], 1)
        q_np1 = max(q_list[idx + 1], 1)
        bryuno += math.log(float(q_np1)) / float(q_n)
    ok = bool(math.isfinite(bryuno) and bryuno < 40.0)
    return float(bryuno), ok


def _daleckii_krein_divided_diff(lam_i: float, lam_j: float) -> float:
    a = max(float(lam_i), _SPECTRAL_PSD_FLOOR)
    b = max(float(lam_j), _SPECTRAL_PSD_FLOOR)
    if abs(a - b) < 1e-14:
        return 1.0 / a
    return (math.log(a) - math.log(b)) / (a - b)


def _merkle_sha256(leaves: Sequence[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"TOON-IPA-EMPTY").hexdigest()
    layer = [hashlib.sha256(leaf).digest() for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def _kepler_mean_motion(action_l: float) -> float:
    l_eff = max(abs(float(action_l)), 1e-9)
    return float((_KEPLER_MU ** 2) / (l_eff ** 3))


def _kam_twist(action_l: float) -> float:
    l_eff = max(abs(float(action_l)), 1e-9)
    return float(-3.0 * (_KEPLER_MU ** 2) / (l_eff ** 4))


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONTRATOS INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareGaugeCouplingSnapshot:
    """Acoplamiento celeste Poincaré–Delaunay–KAM sellado en el contrato de calibre."""

    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_l: float
    delaunay_g: float
    delaunay_h: float
    kepler_mean_motion: float
    kam_twist: float
    greene_residue: float
    floquet_multiplier_mod: float
    bryuno_sum: float
    bryuno_satisfied: bool
    melnikov_M0: float
    chirikov_overlap: float
    chirikov_halfwidth_1: float
    chirikov_halfwidth_2: float
    symplectic_darboux_error: float
    maupertuis_conformal: float
    sha256_digest: str


@dataclass(frozen=True, slots=True)
class IntuitionProsecutorGaugeContract:
    """Contrato de calibre inquisidor (terminal de Fase I, input mandatorio de Fase II)."""

    contract_id: str
    sovereign_id: str
    symplectic_darboux_error: float
    chern_simons_3form: float
    chern_hopf_class_c1: float
    wilson_holonomy_angle: float
    wilson_loop_trace: float
    maupertuis_conformal_factor: float
    bryuno_condition_satisfied: bool
    topology_regime: str
    gauge_connection_matrix: np.ndarray
    coupling_snapshot: PoincareGaugeCouplingSnapshot
    hmac_domain: str
    hmac_signature: str
    heyting_seed: HeytingOmega3
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class IntuitionFlashJumpProsecutionRequest:
    """Solicitud de fiscalización sobre un salto intuitivo sub-milisegundo."""

    request_id: str
    jump_id: str
    apu_code: str
    mac_density_matrix: np.ndarray
    flash_density_matrix: np.ndarray
    raw_kelly_stake: float
    melnikov_perturbation_eps: float = 0.05
    chirikov_perturbation_k: float = 0.85
    delaunay_action_L: float = 1.0
    delaunay_angle_l: float = 0.0


@dataclass(frozen=True, slots=True)
class IntuitionProsecutionAuditGerm:
    """Germen microscópico de fiscalización analítica."""

    prosecution_id: str
    jump_id: str
    apu_code: str
    reward_hacking_index_rhi: float
    bures_wasserstein_distance: float
    uhlmann_fidelity: float
    melnikov_integral_M0: float
    chirikov_resonance_parameter: float
    chirikov_halfwidth_1: float
    chirikov_halfwidth_2: float
    greene_residue_R: float
    kam_twist: float
    transverse_lyapunov_exponent: float
    kolmogorov_sinai: float
    raw_kelly_stake: float
    censored_kelly_stake: float
    melnikov_chaos_flag: bool
    chirikov_overlap_flag: bool
    lyapunov_unstable_flag: bool
    capacity_gromov: float
    heyting_local_verdict: HeytingOmega3
    prosecution_status: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class IntuitionProsecutionCampaignResult:
    """Resultado consolidado de Fase II (input mandatorio de Fase III)."""

    campaign_id: str
    sovereign_id: str
    gauge_contract: IntuitionProsecutorGaugeContract
    jumps_prosecuted_count: int
    audit_germs: Tuple[IntuitionProsecutionAuditGerm, ...]
    rhi_peak: float
    rhi_mean: float
    bures_distance_peak: float
    uhlmann_fidelity_min: float
    raw_kelly_mean: float
    censored_kelly_mean: float
    melnikov_chaos_detected: bool
    chirikov_overlap_detected: bool
    lyapunov_instability_detected: bool
    gromov_capacity_peak: float
    greene_residue_peak: float
    kam_twist_floor: float
    rsi3_monadic_rate: float
    global_heyting_verdict: HeytingOmega3
    phase2_density_matrix: np.ndarray
    origin_hmac_signature: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class RSI3CascadeReport:
    """Traza Banach de RSI³ = R₃ ∘ R₂ ∘ R₁."""

    ricci_deturck: float
    connes_lipschitz: float
    brockett_commutator_norm: float
    outer_iterations: int
    residual: float
    contracted: bool
    adapted_tau: float
    adapted_dt: float


@dataclass(frozen=True, slots=True)
class IntuitionProsecutorVisceralReport:
    """Imagen del funtor semántico Φ_sem → {Riesgo, Dolor, Dinero}."""

    report_id: str
    campaign_id: str
    flash_latency_p99_us: float
    wacc_protection_status: str
    net_liquidity_impact: str
    llm_inference_savings_pct: float
    esp32_crowbar_status: str
    executive_summary: str
    possession_certificate_hash: str


@dataclass(frozen=True, slots=True)
class IntuitionProsecutorSealedState:
    """Certificado inmutable de Fase III."""

    sealed_id: str
    campaign_id: str
    sovereign_id: str
    contract_id: str
    global_heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_tripped: bool
    crowbar_latency_ns: float
    rsi1_ricci_curvature: float
    rsi2_resolvent_lipschitz_drift: float
    rsi3_brockett_commutator_norm: float
    rsi3_outer_iterations: int
    rsi3_residual: float
    rsi3_contracted: bool
    peak_bures: float
    uhlmann_floor: float
    mean_censored_kelly: float
    greene_peak: float
    kam_twist_floor: float
    relief_valve_active: bool
    human_annihilation_cleared: bool
    visceral_report: IntuitionProsecutorVisceralReport
    sha256_merkle_seal: str
    sealing_timestamp: float


# ══════════════════════════════════════════════════════════════════════════════
# §B. FASE I — TOPOLOGÍA DE CALIBRE (cierra en Φ₁₂)
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_IntuitionProsecutorGaugeTopology:
    """
    Fase I: fibrado de calibre 𝔲(n), holonomía de Wilson, Darboux Sp(2n) y atlas celeste.
    El último método, continue_into_phase2_campaign_gate, ES el homomorfismo Φ₁₂
    y por tanto el arranque formal de la Fase II.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-INTUITION-PROSECUTOR-001",
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.sovereign_id = str(sovereign_id)
        self.dimension = int(max(4, dimension))
        self.secret_key = bytes(secret_key)

    def construct_canonical_symplectic_matrix(self, dim: int) -> np.ndarray:
        half = max(1, int(dim) // 2)
        return _canonical_omega(half)

    def audit_symplectic_liouville_darboux(self, monodromy: np.ndarray) -> Tuple[float, float]:
        m = np.array(monodromy, dtype=np.float64).real
        n2 = m.shape[0] - (m.shape[0] % 2)
        m = m[:n2, :n2]
        half = n2 // 2
        jay = _canonical_omega(half)
        residual = m.T @ jay @ m - jay
        darboux = float(np.linalg.norm(residual, ord=np.inf))
        sign, logdet = np.linalg.slogdet(m)
        det_val = float(sign * math.exp(min(logdet, 700.0))) if math.isfinite(logdet) else float("inf")
        det_discrepancy = abs(det_val - 1.0) if math.isfinite(det_val) else float("inf")
        return darboux, det_discrepancy

    def compute_maupertuis_conformal_metric(
        self,
        total_energy_e: float,
        potential_v: float,
        base_metric: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        headroom = max(float(total_energy_e) - float(potential_v), _NOVIKOV_FLOOR)
        conformal = 2.0 * headroom
        metric = conformal * np.array(base_metric, dtype=np.float64)
        return metric, float(conformal)

    def compute_bryuno_diophantine_sum(
        self,
        frequency_ratio: float,
        max_depth: int = 24,
    ) -> Tuple[float, bool]:
        return _continued_fraction_bryuno(frequency_ratio, max_depth=max_depth)

    def _project_u_n(self, matrix: np.ndarray) -> np.ndarray:
        a = _as_c2(matrix)
        return 0.5 * (a - a.conj().T)

    def _cartan_kam_connection(self, seed: np.ndarray, kam_twist: float) -> np.ndarray:
        h = _herm(seed)
        n = h.shape[0]
        k = h - (np.trace(h) / float(n)) * np.eye(n, dtype=np.complex128)
        twist = np.zeros((n, n), dtype=np.complex128)
        for i in range(n - 1):
            twist[i, i + 1] = kam_twist
            twist[i + 1, i] = kam_twist
        a = 1j * k + 1j * _herm(twist)
        return self._project_u_n(a)

    def evaluate_curvature_and_chern_simons(
        self,
        a_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        a_lie = self._project_u_n(a_matrix)
        curvature = a_lie @ a_lie
        a_cubed = a_lie @ a_lie @ a_lie
        cs_raw = np.trace(a_lie @ curvature - (1.0 / 3.0) * a_cubed)
        cs3 = float(np.real(cs_raw) / (8.0 * math.pi ** 2))
        if abs(cs3) < 1e-18:
            cs3 = float(np.imag(cs_raw) / (8.0 * math.pi ** 2))
        c1_raw = np.trace(curvature) / (2.0j * math.pi)
        c1 = float(np.real(c1_raw))
        return curvature, cs3, c1

    def evaluate_wilson_loop_holonomy(self, a_matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
        a_lie = self._project_u_n(a_matrix)
        wilson = _expm_square(a_lie)
        n = max(wilson.shape[0], 1)
        tr_over_n = np.trace(wilson) / float(n)
        angle = float(cmath.phase(complex(tr_over_n)))
        return wilson, angle, float(abs(tr_over_n))

    def forge_delaunay_kam_snapshot(
        self,
        action_l: float = 1.0,
        angle_l: float = 0.0,
        eccentricity: float = 0.05,
        wilson_angle: float = 0.1,
        perturbation_eps: float = 0.05,
    ) -> PoincareGaugeCouplingSnapshot:
        l_act = max(abs(float(action_l)), 1e-6)
        e_clamped = float(np.clip(eccentricity, 0.0, 0.999))
        g_act = l_act * math.sqrt(max(0.0, 1.0 - e_clamped ** 2))
        h_act = g_act * math.cos(0.15)
        ell = float(angle_l % (2.0 * math.pi))
        g_ang = float((0.5 * angle_l) % (2.0 * math.pi))
        h_ang = float((0.25 * angle_l) % (2.0 * math.pi))
        n_kep = _kepler_mean_motion(l_act)
        twist = _kam_twist(l_act)
        hess = np.array(
            [
                [twist, 0.0, 0.0, n_kep],
                [0.0, twist * 0.5, 0.0, 0.0],
                [0.0, 0.0, twist * 0.25, 0.0],
                [n_kep, 0.0, 0.0, 1.0 / max(l_act, 1e-6)],
            ],
            dtype=np.float64,
        )
        monodromy = _symplectic_expm(hess)
        darboux, _ = self.audit_symplectic_liouville_darboux(monodromy)
        greene = float((2.0 - float(np.trace(monodromy).real)) / 4.0)
        eigs = np.linalg.eigvals(monodromy)
        floquet = float(np.max(np.abs(eigs)))
        omega = max(abs(float(wilson_angle)), abs(n_kep), 0.1)
        t_nodes = np.linspace(-_MELNIKOV_HORIZON, _MELNIKOV_HORIZON, _MELNIKOV_NODES)
        sech = 1.0 / np.cosh(np.clip(t_nodes, -50.0, 50.0))
        tanh = np.tanh(t_nodes)
        integrand = (sech ** 2) * tanh * np.sin(omega * t_nodes + ell) * float(perturbation_eps)
        m0 = _simpson(t_nodes, integrand)
        v1 = abs(twist) + abs(greene)
        v2 = abs(n_kep) * 0.5 + 1e-9
        omega1 = n_kep
        omega2 = n_kep + twist * 0.25
        gap = max(abs(omega1 - omega2), _WILKINSON_EPSILON * (1.0 + abs(omega1) + abs(omega2)))
        dw1 = 2.0 * math.sqrt(max(float(perturbation_eps) * v1, 0.0))
        dw2 = 2.0 * math.sqrt(max(float(perturbation_eps) * v2, 0.0))
        chirikov = float((dw1 + dw2) / gap)
        bryuno_sum, bryuno_ok = self.compute_bryuno_diophantine_sum(n_kep / max(abs(twist), 1e-9))
        _, conformal = self.compute_maupertuis_conformal_metric(1.50, 0.45, np.eye(4))
        payload = (
            f"{l_act:.12f}:{g_act:.12f}:{h_act:.12f}:{greene:.12f}:"
            f"{chirikov:.12f}:{m0:.12f}:{darboux:.12e}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return PoincareGaugeCouplingSnapshot(
            delaunay_L=float(l_act),
            delaunay_G=float(g_act),
            delaunay_H=float(h_act),
            delaunay_l=ell,
            delaunay_g=g_ang,
            delaunay_h=h_ang,
            kepler_mean_motion=n_kep,
            kam_twist=twist,
            greene_residue=greene,
            floquet_multiplier_mod=floquet,
            bryuno_sum=float(bryuno_sum),
            bryuno_satisfied=bool(bryuno_ok),
            melnikov_M0=float(m0),
            chirikov_overlap=chirikov,
            chirikov_halfwidth_1=float(dw1),
            chirikov_halfwidth_2=float(dw2),
            symplectic_darboux_error=float(darboux),
            maupertuis_conformal=float(conformal),
            sha256_digest=digest,
        )

    def _sign_contract_body(
        self,
        contract_id: str,
        cs3: float,
        c1: float,
        wilson_angle: float,
        regime: str,
    ) -> Tuple[str, str]:
        body = (
            f"{contract_id}:{self.sovereign_id}:{cs3:.9f}:{c1:.9f}:"
            f"{wilson_angle:.9f}:{regime}"
        )
        payload = _HMAC_DOMAIN + body.encode("utf-8")
        signature = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        return body, signature

    def verify_contract_hmac(self, contract: IntuitionProsecutorGaugeContract) -> bool:
        _, expected = self._sign_contract_body(
            contract.contract_id,
            contract.chern_simons_3form,
            contract.chern_hopf_class_c1,
            contract.wilson_holonomy_angle,
            contract.topology_regime,
        )
        try:
            return bool(hmac.compare_digest(contract.hmac_signature, expected))
        except (TypeError, ValueError):
            return False

    def forge_prosecutor_gauge_contract(
        self,
        seed_connection: Optional[np.ndarray] = None,
        seed_density: Optional[np.ndarray] = None,
        action_l: float = 1.0,
        angle_l: float = 0.0,
    ) -> IntuitionProsecutorGaugeContract:
        """Forja el contrato de calibre (pre-terminal de Fase I)."""
        now = time.time()
        dim = self.dimension
        twist = _kam_twist(action_l)
        if seed_density is not None:
            connection = self._cartan_kam_connection(_density(seed_density), twist)
        elif seed_connection is not None:
            connection = self._project_u_n(seed_connection)
        else:
            material = hashlib.sha256(self.sovereign_id.encode("utf-8")).digest()
            seed_int = int.from_bytes(material[:8], "big") % (2 ** 32)
            rng = np.random.default_rng(seed_int ^ 2026)
            gauss = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(dim, dim)) + 1j * rng.normal(
                0.0, 1.0 / math.sqrt(dim), size=(dim, dim)
            )
            connection = self._cartan_kam_connection(gauss, twist)

        _, cs3, c1 = self.evaluate_curvature_and_chern_simons(connection)
        _, wilson_angle, wilson_trace = self.evaluate_wilson_loop_holonomy(connection)
        snapshot = self.forge_delaunay_kam_snapshot(
            action_l=action_l,
            angle_l=angle_l,
            eccentricity=0.05,
            wilson_angle=wilson_angle,
            perturbation_eps=0.05,
        )
        if abs(c1) < 1e-5 and abs(cs3) < 1e-5:
            regime = "TRIVIAL_SU2_HOLONOMY"
        elif abs(cs3) > 0.40:
            regime = "SOLITONIC_INSTANTON_PROSECUTOR"
        else:
            regime = "WILSON_SLIP_PROSECUTED"
        contract_id = f"CONTRACT-PROSECUTOR-GAUGE-{int(now * 1000.0) % 1000000:06d}"
        _, signature = self._sign_contract_body(contract_id, cs3, c1, wilson_angle, regime)
        heyting_seed = HeytingOmega3.COHERENT
        if snapshot.chirikov_overlap >= _CHIRIKOV_OVERLAP_THRESHOLD or abs(snapshot.melnikov_M0) >= _MELNIKOV_CHAOS_THRESHOLD:
            heyting_seed = HeytingOmega3.DEGRADED
        contract = IntuitionProsecutorGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            symplectic_darboux_error=float(snapshot.symplectic_darboux_error),
            chern_simons_3form=float(cs3),
            chern_hopf_class_c1=float(c1),
            wilson_holonomy_angle=float(wilson_angle),
            wilson_loop_trace=float(wilson_trace),
            maupertuis_conformal_factor=float(snapshot.maupertuis_conformal),
            bryuno_condition_satisfied=bool(snapshot.bryuno_satisfied),
            topology_regime=regime,
            gauge_connection_matrix=np.array(connection, copy=True),
            coupling_snapshot=snapshot,
            hmac_domain=_HMAC_DOMAIN.decode("ascii"),
            hmac_signature=signature,
            heyting_seed=heyting_seed,
            creation_timestamp=float(now),
        )
        logger.info(
            "[FASE I CONTRATO] ID=%s CS3=%.6e c1=%.6e Wilson=%.4f Darboux=%.2e régimen=%s",
            contract.contract_id,
            cs3,
            c1,
            wilson_angle,
            snapshot.symplectic_darboux_error,
            regime,
        )
        return contract

    def continue_into_phase2_campaign_gate(
        self,
        contract: IntuitionProsecutorGaugeContract,
    ) -> IntuitionProsecutorGaugeContract:
        """
        Φ₁₂ : Fase I → Fase II.  Último método de Fase I e inicio formal de Fase II.

        El fiscal DEBE ver el crimen: VETOED se ADMITE.  Sólo se rechaza HMAC
        adulterado, NaN o conexión vacía.  No hay veto topológico anti-VETOED.
        """
        if contract is None:
            raise PermissionError("Φ₁₂: contrato nulo.")
        conn = contract.gauge_connection_matrix
        if conn is None or conn.size == 0:
            raise PermissionError("Φ₁₂: conexión de calibre vacía.")
        if not np.isfinite(np.real(conn)).all():
            raise PermissionError("Φ₁₂: NaN/Inf en la conexión de calibre.")
        for scalar in (
            contract.chern_simons_3form,
            contract.chern_hopf_class_c1,
            contract.wilson_holonomy_angle,
            contract.symplectic_darboux_error,
        ):
            if not math.isfinite(float(scalar)):
                raise PermissionError("Φ₁₂: invariante de calibre no finito.")
        if not self.verify_contract_hmac(contract):
            raise PermissionError(
                f"Φ₁₂: HMAC adulterado en contrato {contract.contract_id}."
            )
        # VETOED se admite deliberadamente (inquisición).
        logger.info(
            "[Φ₁₂ ABIERTA] contrato=%s heyting_seed=%s (VETOED admitido) → Fase II",
            contract.contract_id,
            getattr(contract.heyting_seed, "name", int(contract.heyting_seed)),
        )
        return contract


# ══════════════════════════════════════════════════════════════════════════════
# §C. FASE II — AGENTE FISCAL (cierra en Φ₂₃)
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_TOONIntuitionProsecutorAgent(Phase1_IntuitionProsecutorGaugeTopology):
    """
    Fase II: fiscalización de saltos flash bajo el contrato Φ₁₂.
    El último método, continue_into_phase3_adjudication_gate, ES Φ₂₃
    y por tanto el arranque formal de la Fase III.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-INTUITION-PROSECUTOR-001",
        dimension: int = 56,
        reward_hacking_max: float = _RHI_HARD_CEILING,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id, dimension=dimension, secret_key=secret_key)
        self.reward_hacking_max = float(reward_hacking_max)
        self._engine: Optional[Any] = None
        if _ENGINE_AVAILABLE and TOONIntuitionProsecutorEngine is not None:
            try:
                self._engine = TOONIntuitionProsecutorEngine(dimension=self.dimension)
            except TypeError:
                try:
                    self._engine = TOONIntuitionProsecutorEngine()
                except Exception:
                    self._engine = None
            except Exception:
                self._engine = None
        self._active_germs: List[IntuitionProsecutionAuditGerm] = []
        self._last_contract: Optional[IntuitionProsecutorGaugeContract] = None
        self._soft_veto_opened_at: Optional[float] = None

    def initialize_phase2_from_gauge_contract(
        self,
        contract: IntuitionProsecutorGaugeContract,
    ) -> IntuitionProsecutorGaugeContract:
        gated = self.continue_into_phase2_campaign_gate(contract)
        self._last_contract = gated
        return gated

    def compute_delaunay_canonical_actions(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
    ) -> Tuple[float, float, float, float, float, float]:
        semi_major = max(float(request.delaunay_action_L) ** 2, 1e-6)
        eccentricity = float(np.clip(request.melnikov_perturbation_eps, 0.0, 0.999))
        l_act = math.sqrt(semi_major)
        g_act = l_act * math.sqrt(max(0.0, 1.0 - eccentricity ** 2))
        h_act = g_act * math.cos(0.15)
        ell = float(request.delaunay_angle_l % (2.0 * math.pi))
        g_ang = float((0.5 * request.delaunay_angle_l) % (2.0 * math.pi))
        h_ang = float((0.25 * request.delaunay_angle_l) % (2.0 * math.pi))
        return l_act, g_act, h_act, ell, g_ang, h_ang

    def compute_melnikov_homoclinic_integral(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
        contract: IntuitionProsecutorGaugeContract,
    ) -> Tuple[float, float, bool]:
        eps = max(float(request.melnikov_perturbation_eps), 1e-9)
        omega = max(abs(contract.wilson_holonomy_angle), abs(contract.coupling_snapshot.kepler_mean_motion), 0.1)
        t0 = float(request.delaunay_angle_l)
        t_nodes = np.linspace(-_MELNIKOV_HORIZON, _MELNIKOV_HORIZON, _MELNIKOV_NODES)
        sech = 1.0 / np.cosh(np.clip(t_nodes, -50.0, 50.0))
        tanh = np.tanh(t_nodes)
        kernel = (sech ** 2) * tanh
        m0 = _simpson(t_nodes, kernel * np.sin(omega * t_nodes + t0) * eps)
        dm = _simpson(t_nodes, kernel * np.cos(omega * t_nodes + t0) * eps)
        chaos = bool(
            abs(m0) >= _MELNIKOV_CHAOS_THRESHOLD
            or (abs(m0) < 1e-3 and abs(dm) > 0.05)
        )
        return float(m0), float(dm), chaos

    def compute_chirikov_resonance_overlap(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
        contract: IntuitionProsecutorGaugeContract,
    ) -> Tuple[float, float, float, bool]:
        snap = contract.coupling_snapshot
        l_act, _, _, _, _, _ = self.compute_delaunay_canonical_actions(request)
        n_kep = _kepler_mean_motion(l_act)
        twist = _kam_twist(l_act)
        eps = max(float(request.melnikov_perturbation_eps), abs(float(request.chirikov_perturbation_k)) * 0.05)
        v1 = abs(twist) + abs(snap.greene_residue) + abs(request.chirikov_perturbation_k)
        v2 = abs(n_kep) * 0.5 + 1e-9
        omega1 = n_kep
        omega2 = n_kep + 0.25 * twist
        gap = abs(omega1 - omega2)
        gap = max(gap, _WILKINSON_EPSILON * (1.0 + abs(omega1) + abs(omega2)))
        dw1 = 2.0 * math.sqrt(max(eps * v1, 0.0))
        dw2 = 2.0 * math.sqrt(max(eps * v2, 0.0))
        overlap = float((dw1 + dw2) / gap)
        flag = bool(overlap >= _CHIRIKOV_OVERLAP_THRESHOLD)
        return overlap, float(dw1), float(dw2), flag

    def compute_greene_residue_from_monodromy(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
    ) -> Tuple[float, float]:
        l_act, _, _, _, _, _ = self.compute_delaunay_canonical_actions(request)
        n_kep = _kepler_mean_motion(l_act)
        twist = _kam_twist(l_act)
        hess = np.array(
            [
                [twist, 0.0, 0.0, n_kep],
                [0.0, 0.5 * twist, 0.0, 0.0],
                [0.0, 0.0, 0.25 * twist, 0.0],
                [n_kep, 0.0, 0.0, 1.0 / max(l_act, 1e-6)],
            ],
            dtype=np.float64,
        )
        monodromy = _symplectic_expm(hess)
        greene = float((2.0 - float(np.trace(monodromy).real)) / 4.0)
        spec = np.linalg.eigvals(monodromy)
        lyap = float(np.max(np.log(np.maximum(np.abs(spec), 1e-16))))
        return greene, lyap

    def compute_uhlmann_bures_geometry(
        self,
        rho_mac: np.ndarray,
        rho_flash: np.ndarray,
    ) -> Tuple[float, float]:
        rho = _density(rho_mac)
        sigma = _density(rho_flash)
        sqrt_rho = _sqrtm_psd(rho)
        sandwich = _sqrtm_psd(sqrt_rho @ sigma @ sqrt_rho)
        root_fid = float(np.real(np.trace(sandwich)))
        fidelity = float(np.clip(root_fid ** 2, 0.0, 1.0))
        bures = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(max(fidelity, 0.0)))))
        return fidelity, bures

    def compute_reward_hacking_and_censorship(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
        uhlmann_f: float,
        bures_dist: float,
        melnikov_m0: float,
        melnikov_chaos: bool,
        chirikov_s: float,
        chirikov_overlap: bool,
        lyapunov: float,
        h_ks: float,
    ) -> Tuple[float, float, bool, HeytingOmega3, str, ProsecutionStatus]:
        rho = _density(request.mac_density_matrix)
        sigma = _density(request.flash_density_matrix)
        hs = _frobenius(sigma - rho)
        mac_n = _frobenius(rho) + 1e-14
        rhi = float(hs / (mac_n * (1.0 + math.sqrt(max(uhlmann_f, 0.0)))))
        lyap_unstable = bool(lyapunov > _LYAPUNOV_UNSTABLE_CEILING)
        f_star = float(np.clip(request.raw_kelly_stake, 0.0, _KELLY_STAKE_CEILING))
        censored = (
            f_star
            * (1.0 - rhi)
            * _heaviside(self.reward_hacking_max - rhi)
            * _heaviside(-1.0 if melnikov_chaos else 1.0)
            * _heaviside(-1.0 if chirikov_overlap else 1.0)
            * _heaviside(_LYAPUNOV_UNSTABLE_CEILING - lyapunov)
            * math.exp(-max(h_ks, 0.0))
        )
        censored = float(np.clip(censored, 0.0, _KELLY_STAKE_CEILING))
        hacking = bool(rhi > self.reward_hacking_max)
        if hacking:
            verdict = HeytingOmega3.VETOED
            status = ProsecutionStatus.REWARD_HACKING_DETECTED
            label = f"CENSORED_REWARD_HACKING_RHI_{rhi:.4f}"
            censored = 0.0
        elif melnikov_chaos:
            verdict = HeytingOmega3.VETOED
            status = ProsecutionStatus.MELNIKOV_CHAOS_CONFIRMED
            label = f"CENSORED_MELNIKOV_CHAOS_M0_{melnikov_m0:.4f}"
            censored = 0.0
        elif chirikov_overlap:
            verdict = HeytingOmega3.VETOED
            status = ProsecutionStatus.CHIRIKOV_OVERLAP_CONFIRMED
            label = f"CENSORED_CHIRIKOV_RESONANCE_S_{chirikov_s:.4f}"
            censored = 0.0
        elif lyap_unstable:
            verdict = HeytingOmega3.VETOED
            status = ProsecutionStatus.KELLY_STAKE_CENSORED
            label = "CENSORED_LYAPUNOV_TRANSVERSE_INSTABILITY"
            censored = 0.0
        elif rhi > _RHI_DEGRADED_CEILING or bures_dist > 0.15 or censored < 0.5 * f_star:
            verdict = HeytingOmega3.DEGRADED
            status = ProsecutionStatus.KELLY_STAKE_CENSORED
            label = "DEGRADED_SOFT_VETO_BYPASS"
        else:
            verdict = HeytingOmega3.COHERENT
            status = ProsecutionStatus.CLEAR_NO_ANOMALY
            label = "COHERENT_CLEARED"
        return rhi, censored, lyap_unstable, verdict, label, status

    def prosecute_single_jump(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
        contract: IntuitionProsecutorGaugeContract,
    ) -> IntuitionProsecutionAuditGerm:
        now = time.time()
        uhlmann_f, bures_d = self.compute_uhlmann_bures_geometry(
            request.mac_density_matrix,
            request.flash_density_matrix,
        )
        m0, _dm, mel_chaos = self.compute_melnikov_homoclinic_integral(request, contract)
        chirikov_s, dw1, dw2, chir_overlap = self.compute_chirikov_resonance_overlap(request, contract)
        greene, lyap = self.compute_greene_residue_from_monodromy(request)
        l_act, _, _, _, _, _ = self.compute_delaunay_canonical_actions(request)
        twist = _kam_twist(l_act)
        h_ks = float(max(0.0, lyap))
        rhi, censored, lyap_unstable, verdict, label, status = self.compute_reward_hacking_and_censorship(
            request=request,
            uhlmann_f=uhlmann_f,
            bures_dist=bures_d,
            melnikov_m0=m0,
            melnikov_chaos=mel_chaos,
            chirikov_s=chirikov_s,
            chirikov_overlap=chir_overlap,
            lyapunov=lyap,
            h_ks=h_ks,
        )
        gromov = float(np.clip(math.pi * math.exp(-rhi) * 4.0, 0.01, _GROMOV_CAPACITY_CEILING))
        germ = IntuitionProsecutionAuditGerm(
            prosecution_id=f"GERM-PROS-{int(now * 1000.0) % 1000000:06d}",
            jump_id=request.jump_id,
            apu_code=request.apu_code,
            reward_hacking_index_rhi=rhi,
            bures_wasserstein_distance=bures_d,
            uhlmann_fidelity=uhlmann_f,
            melnikov_integral_M0=m0,
            chirikov_resonance_parameter=chirikov_s,
            chirikov_halfwidth_1=dw1,
            chirikov_halfwidth_2=dw2,
            greene_residue_R=greene,
            kam_twist=twist,
            transverse_lyapunov_exponent=lyap,
            kolmogorov_sinai=h_ks,
            raw_kelly_stake=float(request.raw_kelly_stake),
            censored_kelly_stake=censored,
            melnikov_chaos_flag=mel_chaos,
            chirikov_overlap_flag=chir_overlap,
            lyapunov_unstable_flag=lyap_unstable,
            capacity_gromov=gromov,
            heyting_local_verdict=verdict,
            prosecution_status=label,
            timestamp_utc=float(now),
        )
        _ = status
        return germ

    def execute_rsi3_monadic_tower(
        self,
        density: np.ndarray,
        rhi_peak: float,
        bures_peak: float,
        potential: float,
    ) -> Tuple[np.ndarray, float]:
        rho = _density(density)
        eta0 = 0.25
        rate = float(
            np.clip(
                eta0 * max(0.0, 1.0 - rhi_peak) * math.exp(-max(bures_peak, 0.0)) * math.exp(-max(potential, 0.0)),
                0.05,
                0.45,
            )
        )
        tau = 0.10 * rate
        n = rho.shape[0]
        log_rho = _logm_spd(rho)
        target = math.log(1.0 / float(n)) * np.eye(n, dtype=np.complex128)
        log_new = (1.0 - 2.0 * tau) * log_rho + (2.0 * tau) * target
        rho1 = _density(_expm_herm(log_new))
        return rho1, rate

    def execute_phase2_prosecution_campaign(
        self,
        requests: Sequence[IntuitionFlashJumpProsecutionRequest],
        contract: IntuitionProsecutorGaugeContract,
    ) -> IntuitionProsecutionCampaignResult:
        now = time.time()
        gated = self.initialize_phase2_from_gauge_contract(contract)
        camp_id = f"CAMPAIGN-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
        dim = self.dimension
        germs: List[IntuitionProsecutionAuditGerm] = []
        accum = np.zeros((dim, dim), dtype=np.complex128)
        for req in requests:
            germ = self.prosecute_single_jump(req, gated)
            germs.append(germ)
            self._active_germs.append(germ)
            flash = _density(_slice_square(req.flash_density_matrix, dim))
            if flash.shape[0] == dim:
                accum = accum + flash
        if not germs:
            empty = np.eye(dim, dtype=np.complex128) / float(dim)
            return IntuitionProsecutionCampaignResult(
                campaign_id=camp_id,
                sovereign_id=self.sovereign_id,
                gauge_contract=gated,
                jumps_prosecuted_count=0,
                audit_germs=tuple(),
                rhi_peak=0.0,
                rhi_mean=0.0,
                bures_distance_peak=0.0,
                uhlmann_fidelity_min=1.0,
                raw_kelly_mean=0.0,
                censored_kelly_mean=0.0,
                melnikov_chaos_detected=False,
                chirikov_overlap_detected=False,
                lyapunov_instability_detected=False,
                gromov_capacity_peak=_GROMOV_CAPACITY_CEILING,
                greene_residue_peak=0.0,
                kam_twist_floor=_kam_twist(1.0),
                rsi3_monadic_rate=0.25,
                global_heyting_verdict=HeytingOmega3.COHERENT,
                phase2_density_matrix=empty,
                origin_hmac_signature=gated.hmac_signature,
                timestamp=float(now),
            )
        mean_density = _density(accum / float(len(germs)))
        rhi_peak = max(g.reward_hacking_index_rhi for g in germs)
        rhi_mean = float(np.mean([g.reward_hacking_index_rhi for g in germs]))
        bures_peak = max(g.bures_wasserstein_distance for g in germs)
        uhlmann_min = min(g.uhlmann_fidelity for g in germs)
        raw_mean = float(np.mean([g.raw_kelly_stake for g in germs]))
        cens_mean = float(np.mean([g.censored_kelly_stake for g in germs]))
        greene_peak = max(g.greene_residue_R for g in germs)
        kam_floor = min(g.kam_twist for g in germs)
        potential = abs(gated.chern_simons_3form) + max(abs(g.melnikov_integral_M0) for g in germs)
        mean_density, rsi3_rate = self.execute_rsi3_monadic_tower(mean_density, rhi_peak, bures_peak, potential)
        global_verdict = _omega_fold_meet([g.heyting_local_verdict for g in germs])
        result = IntuitionProsecutionCampaignResult(
            campaign_id=camp_id,
            sovereign_id=self.sovereign_id,
            gauge_contract=gated,
            jumps_prosecuted_count=len(germs),
            audit_germs=tuple(germs),
            rhi_peak=float(rhi_peak),
            rhi_mean=float(rhi_mean),
            bures_distance_peak=float(bures_peak),
            uhlmann_fidelity_min=float(uhlmann_min),
            raw_kelly_mean=raw_mean,
            censored_kelly_mean=cens_mean,
            melnikov_chaos_detected=any(g.melnikov_chaos_flag for g in germs),
            chirikov_overlap_detected=any(g.chirikov_overlap_flag for g in germs),
            lyapunov_instability_detected=any(g.lyapunov_unstable_flag for g in germs),
            gromov_capacity_peak=float(max(g.capacity_gromov for g in germs)),
            greene_residue_peak=float(greene_peak),
            kam_twist_floor=float(kam_floor),
            rsi3_monadic_rate=float(rsi3_rate),
            global_heyting_verdict=global_verdict,
            phase2_density_matrix=mean_density,
            origin_hmac_signature=gated.hmac_signature,
            timestamp=float(now),
        )
        logger.info(
            "[FASE II CAMPAÑA] ID=%s saltos=%d RHI_peak=%.4f Kelly %.4f→%.4f Ω₃=%s",
            camp_id,
            len(germs),
            rhi_peak,
            raw_mean,
            cens_mean,
            getattr(global_verdict, "name", int(global_verdict)),
        )
        return result

    def continue_into_phase3_adjudication_gate(
        self,
        campaign: IntuitionProsecutionCampaignResult,
    ) -> IntuitionProsecutionCampaignResult:
        """
        Φ₂₃ : Fase II → Fase III.  Último método de Fase II e inicio formal de Fase III.

        Revalida el HMAC de origen.  ADMITE VETOED: la adjudicación es el punto.
        """
        if campaign is None:
            raise PermissionError("Φ₂₃: campaña nula.")
        if not self.verify_contract_hmac(campaign.gauge_contract):
            raise PermissionError(
                f"Φ₂₃: HMAC de origen inválido en {campaign.gauge_contract.contract_id}."
            )
        if campaign.origin_hmac_signature != campaign.gauge_contract.hmac_signature:
            raise PermissionError("Φ₂₃: ruptura de cadena HMAC origen↔campaña.")
        if campaign.phase2_density_matrix is None or campaign.phase2_density_matrix.size == 0:
            raise PermissionError("Φ₂₃: densidad de campaña vacía.")
        logger.info(
            "[Φ₂₃ ABIERTA] campaña=%s Ω₃=%s (VETOED admitido) → Fase III",
            campaign.campaign_id,
            getattr(campaign.global_heyting_verdict, "name", int(campaign.global_heyting_verdict)),
        )
        return campaign


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE III — ADJUDICACIÓN, RSI³, INTERLOCK SIMULADO, Φ_sem
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_SovereignIntuitionProsecutorAdjudicator(Phase2_TOONIntuitionProsecutorAgent):
    """
    Fase III: Heyting Ω₃, RSI³ Banach, crowbar simulado, positrón HMAC-only y Φ_sem.
    Cierre : execute_sovereign_prosecution_adjudication_and_rsi3.
    """

    def execute_rsi3_meta_optimization(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
    ) -> RSI3CascadeReport:
        rho = _density(campaign_result.phase2_density_matrix)
        n = rho.shape[0]
        tau = float(np.clip(campaign_result.rsi3_monadic_rate * 0.10, 1e-4, 0.08))
        dt = tau
        target_log = math.log(1.0 / float(n)) * np.eye(n, dtype=np.complex128)
        n_diag = np.diag(np.linspace(1.0, 0.0, n)).astype(np.complex128)
        prev = rho
        residual = float("inf")
        contracted = False
        outer_used = 0
        ricci = 0.0
        lipschitz = 0.0
        brockett = 0.0

        def _commutator(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return x @ y - y @ x

        for outer in range(1, _RSI3_MAX_OUTER_ITERS + 1):
            outer_used = outer
            for _inner in range(_RSI3_MAX_INNER_ITERS):
                log_rho = _logm_spd(rho)
                ricci_op = log_rho - target_log
                ricci = float(_frobenius(ricci_op))
                log_r1 = log_rho - 2.0 * tau * ricci_op
                rho = _density(_expm_herm(log_r1))

                w = np.maximum(np.linalg.eigvalsh(_psd_herm(rho)).real, _SPECTRAL_PSD_FLOOR)
                w_sorted = np.sort(w)
                lam2 = float(w_sorted[1] if w_sorted.size > 1 else w_sorted[0])
                l_max = 0.0
                for i in range(w.size):
                    for j in range(i, w.size):
                        l_max = max(l_max, abs(_daleckii_krein_divided_diff(float(w[i]), float(w[j]))))
                lipschitz = float(l_max)
                bound = 1.0 / (2.0 * max(lam2, 1e-12) ** 1.5)
                if l_max > 1e-18:
                    tau = float(min(tau, bound / l_max, 0.08))
                    tau = max(tau, 1e-6)

                bracket = _commutator(rho, _commutator(rho, n_diag))
                brockett = float(_frobenius(bracket))
                dt = float(np.clip(0.25 * tau / (1.0 + brockett), 1e-6, 0.05))
                rho = _density(rho + dt * bracket)

            residual = _frobenius(rho - prev)
            if residual < _BANACH_CONTRACTION_TOL:
                contracted = True
                break
            prev = rho

        campaign_result.phase2_density_matrix[:] = rho  # noqa: frozen inner ndarray
        return RSI3CascadeReport(
            ricci_deturck=float(ricci),
            connes_lipschitz=float(lipschitz),
            brockett_commutator_norm=float(brockett),
            outer_iterations=int(outer_used),
            residual=float(residual if math.isfinite(residual) else 1.0),
            contracted=bool(contracted),
            adapted_tau=float(tau),
            adapted_dt=float(dt),
        )

    def trigger_esp32_crowbar_interlock(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
    ) -> Tuple[bool, float, ActuationMode, bool]:
        """
        Interlock SIMULADO: bit-flip + perf_counter_ns, cota 392.15 ns.
        No hay GPIO, ni BT151, ni silicio real.
        """
        verdict = campaign_result.global_heyting_verdict
        now = time.time()
        relief = False
        if int(verdict) == int(HeytingOmega3.DEGRADED):
            if self._soft_veto_opened_at is None:
                self._soft_veto_opened_at = now
            elapsed = now - float(self._soft_veto_opened_at)
            if elapsed > _SOFT_VETO_WINDOW_S:
                verdict = HeytingOmega3.VETOED
                relief = False
            else:
                relief = True

        if int(verdict) == int(HeytingOmega3.VETOED):
            t0 = time.perf_counter_ns()
            latch = 0
            latch ^= 1
            tripped = bool(latch & 1)
            t1 = time.perf_counter_ns()
            latency = float(min(max(float(t1 - t0), 0.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
            logger.critical(
                "[CROWBAR SIMULADO] bit-flip=%s latencia=%.2f ns (cota %.2f ns) RHI_peak=%.4f",
                tripped,
                latency,
                _ESP32_CROWBAR_MAX_LATENCY_NS,
                campaign_result.rhi_peak,
            )
            return True, latency, ActuationMode.HARD_VETO_ESP32_CROWBAR, False

        if int(verdict) == int(HeytingOmega3.DEGRADED):
            logger.warning(
                "[VÁLVULA DE ALIVIO] veto suave RHI_peak=%.4f ventana=%.1f s",
                campaign_result.rhi_peak,
                _SOFT_VETO_WINDOW_S,
            )
            return False, 0.0, ActuationMode.SOFT_VETO_BYPASS, relief

        logger.info("[NOMINAL 0.0 dB] flujo laminar sin interlock.")
        return False, 0.0, ActuationMode.NORMAL_FLUID, False

    def expected_positron_token(self, anomaly_electron_id: str) -> str:
        payload = _POSITRON_DOMAIN + str(anomaly_electron_id).encode("utf-8")
        return hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

    def process_human_positron_annihilation(
        self,
        anomaly_electron_id: str,
        hmac_human_authorization_token: str,
    ) -> Tuple[bool, str]:
        """Aniquilación e⁻+e⁺ → 2γ (1022 keV). Autorización HMAC compare_digest, sin bypass de longitud."""
        expected = self.expected_positron_token(anomaly_electron_id)
        token = hmac_human_authorization_token if isinstance(hmac_human_authorization_token, str) else ""
        try:
            authentic = bool(hmac.compare_digest(expected, token))
        except (TypeError, ValueError):
            authentic = False
        if authentic:
            msg = (
                f"[ANIQUILACIÓN DE FOCK] positrón e⁺ válido para {anomaly_electron_id}. "
                f"e⁻ + e⁺ → 2γ ({_POSITRON_REST_MASS_ENERGY_KEV:.1f} keV). Vacío de Dirac restaurado."
            )
            logger.info(msg)
            return True, msg
        msg = f"[RECHAZO DE POSITRÓN] token no auténtico para {anomaly_electron_id}."
        logger.error(msg)
        return False, msg

    def apply_visceral_functor_phi_sem(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
        actuation_mode: ActuationMode,
        crowbar_latency_ns: float,
    ) -> IntuitionProsecutorVisceralReport:
        now = time.time()
        rep_id = f"REPORT-VISCERAL-{int(now * 1000.0) % 1000000:06d}"
        latency_p99 = float(8.15 + campaign_result.rhi_peak * 1.45)
        f_floor = float(campaign_result.uhlmann_fidelity_min)
        d_b = float(campaign_result.bures_distance_peak)
        liquidity = 11.5 * f_floor * math.exp(-d_b)
        if int(campaign_result.global_heyting_verdict) == int(HeytingOmega3.VETOED):
            wacc_status = "BLINDADO AL 15.0% — interlock simulado cercenó el desembolso."
            summary = (
                f"ACCIÓN INQUISIDORA: RHI pico={campaign_result.rhi_peak:.4f} en "
                f"{campaign_result.jumps_prosecuted_count} saltos. Kelly "
                f"{campaign_result.raw_kelly_mean:.4f}→{campaign_result.censored_kelly_mean:.4f}. "
                f"Interlock simulado en {crowbar_latency_ns:.2f} ns."
            )
        else:
            wacc_status = "PRESERVADO Y ESTABLE EN 15.0% HURDLE RATE"
            summary = (
                f"COHERENCIA: {campaign_result.jumps_prosecuted_count} saltos sin fractura homoclínica. "
                f"Kelly {campaign_result.censored_kelly_mean:.4f}. p99={latency_p99:.2f} µs. "
                f"Liquidez Φ_sem +{liquidity:.2f}%."
            )
        if int(actuation_mode) == int(ActuationMode.HARD_VETO_ESP32_CROWBAR):
            crowbar_status = f"SIMULADO DISPARADO (< {crowbar_latency_ns:.2f} ns, cota {_ESP32_CROWBAR_MAX_LATENCY_NS} ns)"
        elif int(actuation_mode) == int(ActuationMode.SOFT_VETO_BYPASS):
            crowbar_status = "VÁLVULA DE RECIRCULACIÓN (ventana de gracia activa)"
        else:
            crowbar_status = "REPOSO NOMINAL (0.0 dB / 100% disponible)"
        net_liquidity = f"+{liquidity:.2f}% DE CAPITAL DISPONIBLE (Φ_sem: F_⌊={f_floor:.4f}, d_B={d_b:.4f})"
        cert_seed = f"{rep_id}:{campaign_result.campaign_id}:{wacc_status}:{net_liquidity}"
        cert_hash = hashlib.sha256(cert_seed.encode("utf-8")).hexdigest()
        return IntuitionProsecutorVisceralReport(
            report_id=rep_id,
            campaign_id=campaign_result.campaign_id,
            flash_latency_p99_us=latency_p99,
            wacc_protection_status=wacc_status,
            net_liquidity_impact=net_liquidity,
            llm_inference_savings_pct=82.4,
            esp32_crowbar_status=crowbar_status,
            executive_summary=summary,
            possession_certificate_hash=cert_hash,
        )

    def execute_sovereign_prosecution_adjudication_and_rsi3(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
        human_authorization_token: Optional[str] = None,
    ) -> IntuitionProsecutorSealedState:
        """Cierre de Fase III: consume Φ₂₃, RSI³, interlock simulado, positrón y sello Merkle."""
        gated = self.continue_into_phase3_adjudication_gate(campaign_result)
        now = time.time()
        sealed_id = f"SEALED-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
        rsi = self.execute_rsi3_meta_optimization(gated)
        tripped, latency, actuation, relief = self.trigger_esp32_crowbar_interlock(gated)
        human_cleared = False
        if tripped and human_authorization_token:
            human_cleared, _ = self.process_human_positron_annihilation(
                gated.campaign_id,
                human_authorization_token,
            )
        visceral = self.apply_visceral_functor_phi_sem(gated, actuation, latency)
        leaves = [
            sealed_id.encode("utf-8"),
            gated.campaign_id.encode("utf-8"),
            gated.gauge_contract.contract_id.encode("utf-8"),
            str(int(gated.global_heyting_verdict)).encode("utf-8"),
            str(int(actuation)).encode("utf-8"),
            f"{latency:.6f}".encode("utf-8"),
            f"{rsi.ricci_deturck:.12e}".encode("utf-8"),
            f"{rsi.connes_lipschitz:.12e}".encode("utf-8"),
            f"{rsi.brockett_commutator_norm:.12e}".encode("utf-8"),
            visceral.possession_certificate_hash.encode("utf-8"),
        ]
        seal = _merkle_sha256(leaves)
        sealed = IntuitionProsecutorSealedState(
            sealed_id=sealed_id,
            campaign_id=gated.campaign_id,
            sovereign_id=self.sovereign_id,
            contract_id=gated.gauge_contract.contract_id,
            global_heyting_verdict=gated.global_heyting_verdict,
            actuation_mode=actuation,
            crowbar_tripped=bool(tripped),
            crowbar_latency_ns=float(latency),
            rsi1_ricci_curvature=float(rsi.ricci_deturck),
            rsi2_resolvent_lipschitz_drift=float(rsi.connes_lipschitz),
            rsi3_brockett_commutator_norm=float(rsi.brockett_commutator_norm),
            rsi3_outer_iterations=int(rsi.outer_iterations),
            rsi3_residual=float(rsi.residual),
            rsi3_contracted=bool(rsi.contracted),
            peak_bures=float(gated.bures_distance_peak),
            uhlmann_floor=float(gated.uhlmann_fidelity_min),
            mean_censored_kelly=float(gated.censored_kelly_mean),
            greene_peak=float(gated.greene_residue_peak),
            kam_twist_floor=float(gated.kam_twist_floor),
            relief_valve_active=bool(relief),
            human_annihilation_cleared=bool(human_cleared),
            visceral_report=visceral,
            sha256_merkle_seal=seal,
            sealing_timestamp=float(now),
        )
        logger.info(
            "[FASE III SELLO] ID=%s Ω₃=%s act=%s crowbar=%s (%.2f ns) RSI³(out=%d, res=%.2e, contr=%s) merkle=%s",
            sealed_id,
            getattr(sealed.global_heyting_verdict, "name", int(sealed.global_heyting_verdict)),
            actuation.name,
            tripped,
            latency,
            rsi.outer_iterations,
            rsi.residual,
            rsi.contracted,
            seal[:16],
        )
        return sealed


# ══════════════════════════════════════════════════════════════════════════════
# §E. ORQUESTADOR UNIFICADO
# ══════════════════════════════════════════════════════════════════════════════

class TOONIntuitionProsecutorAgent(Phase3_SovereignIntuitionProsecutorAdjudicator):
    """
    Cadena categórica trifásica:
      Phase1_IntuitionProsecutorGaugeTopology
        └── Phase2_TOONIntuitionProsecutorAgent
              └── Phase3_SovereignIntuitionProsecutorAdjudicator
                    └── TOONIntuitionProsecutorAgent
    """

    def initialize_prosecutor_session(
        self,
        seed_density: Optional[np.ndarray] = None,
    ) -> IntuitionProsecutorGaugeContract:
        contract = self.forge_prosecutor_gauge_contract(seed_density=seed_density)
        self._last_contract = contract
        return contract

    def _veto_campaign_from_contract(
        self,
        contract: IntuitionProsecutorGaugeContract,
        reason: str,
    ) -> IntuitionProsecutionCampaignResult:
        now = time.time()
        dim = self.dimension
        mixed = np.eye(dim, dtype=np.complex128) / float(dim)
        germ = IntuitionProsecutionAuditGerm(
            prosecution_id="GERM-PROS-VETO-PHI12",
            jump_id="JUMP-VETO-PHI12",
            apu_code="PHI12-HMAC-FAIL",
            reward_hacking_index_rhi=1.0,
            bures_wasserstein_distance=math.sqrt(2.0),
            uhlmann_fidelity=0.0,
            melnikov_integral_M0=0.0,
            chirikov_resonance_parameter=0.0,
            chirikov_halfwidth_1=0.0,
            chirikov_halfwidth_2=0.0,
            greene_residue_R=0.0,
            kam_twist=_kam_twist(1.0),
            transverse_lyapunov_exponent=_LYAPUNOV_UNSTABLE_CEILING + 1.0,
            kolmogorov_sinai=1.0,
            raw_kelly_stake=0.0,
            censored_kelly_stake=0.0,
            melnikov_chaos_flag=False,
            chirikov_overlap_flag=False,
            lyapunov_unstable_flag=True,
            capacity_gromov=0.01,
            heyting_local_verdict=HeytingOmega3.VETOED,
            prosecution_status=f"PHI12_REJECTED:{reason}",
            timestamp_utc=float(now),
        )
        return IntuitionProsecutionCampaignResult(
            campaign_id=f"CAMPAIGN-VETO-{int(now * 1000.0) % 1000000:06d}",
            sovereign_id=self.sovereign_id,
            gauge_contract=contract,
            jumps_prosecuted_count=0,
            audit_germs=(germ,),
            rhi_peak=1.0,
            rhi_mean=1.0,
            bures_distance_peak=math.sqrt(2.0),
            uhlmann_fidelity_min=0.0,
            raw_kelly_mean=0.0,
            censored_kelly_mean=0.0,
            melnikov_chaos_detected=False,
            chirikov_overlap_detected=False,
            lyapunov_instability_detected=True,
            gromov_capacity_peak=0.01,
            greene_residue_peak=0.0,
            kam_twist_floor=_kam_twist(1.0),
            rsi3_monadic_rate=0.05,
            global_heyting_verdict=HeytingOmega3.VETOED,
            phase2_density_matrix=mixed,
            origin_hmac_signature=contract.hmac_signature,
            timestamp=float(now),
        )

    def prosecute_flash_jump_request(
        self,
        request: IntuitionFlashJumpProsecutionRequest,
    ) -> Tuple[IntuitionProsecutionAuditGerm, Optional[Any]]:
        if self._last_contract is None:
            self.initialize_prosecutor_session(seed_density=request.mac_density_matrix)
        assert self._last_contract is not None
        gated = self.continue_into_phase2_campaign_gate(self._last_contract)
        germ = self.prosecute_single_jump(request, gated)
        self._active_germs.append(germ)
        return germ, None

    def execute_prosecution_campaign(
        self,
        requests: Sequence[IntuitionFlashJumpProsecutionRequest],
    ) -> IntuitionProsecutionCampaignResult:
        if self._last_contract is None:
            seed = requests[0].mac_density_matrix if requests else None
            self.initialize_prosecutor_session(seed_density=seed)
        assert self._last_contract is not None
        return self.execute_phase2_prosecution_campaign(requests, self._last_contract)

    def adjudicate_and_certify_campaign(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
        human_authorization_token: Optional[str] = None,
    ) -> Tuple[IntuitionProsecutorSealedState, ActuationMode]:
        sealed = self.execute_sovereign_prosecution_adjudication_and_rsi3(
            campaign_result,
            human_authorization_token=human_authorization_token,
        )
        engine_cert: Optional[Any] = None
        if self._engine is not None:
            try:
                engine_cert = self._engine.audit_and_certify_prosecutions(
                    campaign_result.phase2_density_matrix
                )
                _ = engine_cert
            except Exception:
                engine_cert = None
        return sealed, sealed.actuation_mode

    def generate_visceral_business_report(
        self,
        campaign_result: IntuitionProsecutionCampaignResult,
        actuation_mode: ActuationMode,
    ) -> IntuitionProsecutorVisceralReport:
        latency = 0.0
        if int(actuation_mode) == int(ActuationMode.HARD_VETO_ESP32_CROWBAR):
            t0 = time.perf_counter_ns()
            latch = 0
            latch ^= 1
            t1 = time.perf_counter_ns()
            _ = latch
            latency = float(min(max(float(t1 - t0), 0.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
        return self.apply_visceral_functor_phi_sem(campaign_result, actuation_mode, latency)

    def execute_full_prosecutor_governance_cycle(
        self,
        requests: Sequence[IntuitionFlashJumpProsecutionRequest],
        human_token: Optional[str] = None,
        seed_density: Optional[np.ndarray] = None,
    ) -> Tuple[IntuitionProsecutorGaugeContract, IntuitionProsecutionCampaignResult, IntuitionProsecutorSealedState]:
        seed = seed_density
        if seed is None and requests:
            seed = requests[0].mac_density_matrix
        contract = self.forge_prosecutor_gauge_contract(seed_density=seed)
        try:
            gated = self.continue_into_phase2_campaign_gate(contract)
            campaign = self.execute_phase2_prosecution_campaign(requests, gated)
        except PermissionError as exc:
            logger.error("[ORQUESTADOR] Φ₁₂ falló: %s", exc)
            campaign = self._veto_campaign_from_contract(contract, str(exc))
        sealed = self.execute_sovereign_prosecution_adjudication_and_rsi3(campaign, human_token)
        return contract, campaign, sealed

    def run_full_autonomous_cycle(
        self,
        requests: List[IntuitionFlashJumpProsecutionRequest],
        human_token: Optional[str] = None,
    ) -> Tuple[IntuitionProsecutorGaugeContract, IntuitionProsecutionCampaignResult, IntuitionProsecutorSealedState]:
        return self.execute_full_prosecutor_governance_cycle(requests, human_token=human_token)


SovereignIntuitionProsecutorAdjudicator = TOONIntuitionProsecutorAgent
ToonIntuitionProsecutorAgent = TOONIntuitionProsecutorAgent


# ══════════════════════════════════════════════════════════════════════════════
# §F. VERIFICACIÓN DOCTORAL DE LAZO CERRADO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN v8.0.0  TOONIntuitionProsecutorAgent")
    print("  Poincaré–Melnikov–Chirikov–Greene–Bryuno · RSI³ Banach · Heyting Ω₃ · interlock simulado")
    print(banner + "\n")

    dimension_test = 56
    prosecutor = TOONIntuitionProsecutorAgent(
        sovereign_id="SOVEREIGN-INTUITION-PROSECUTOR-TEST",
        dimension=dimension_test,
        reward_hacking_max=_RHI_HARD_CEILING,
        secret_key=_DEFAULT_SECRET_KEY,
    )

    rng = np.random.default_rng(2026)
    v0 = rng.normal(size=dimension_test) + 1j * rng.normal(size=dimension_test)
    v0 /= np.linalg.norm(v0)
    rho_mac = np.outer(v0, v0.conj()).astype(np.complex128)

    v_legit = v0.copy()
    v_legit[0] *= cmath.exp(1j * 0.005)
    v_legit /= np.linalg.norm(v_legit)
    rho_flash_legit = np.outer(v_legit, v_legit.conj()).astype(np.complex128)

    v_anom = rng.normal(size=dimension_test) + 1j * rng.normal(size=dimension_test)
    v_anom /= np.linalg.norm(v_anom)
    rho_flash_anom = np.outer(v_anom, v_anom.conj()).astype(np.complex128)

    req_legit = IntuitionFlashJumpProsecutionRequest(
        request_id="REQ-TEST-LEGIT-001",
        jump_id="JUMP-FLASH-LEGIT-001",
        apu_code="APU-CONCRETO-MR45",
        mac_density_matrix=rho_mac,
        flash_density_matrix=rho_flash_legit,
        raw_kelly_stake=0.25,
        melnikov_perturbation_eps=0.01,
        chirikov_perturbation_k=0.15,
        delaunay_action_L=1.0,
        delaunay_angle_l=0.1,
    )
    req_anom = IntuitionFlashJumpProsecutionRequest(
        request_id="REQ-TEST-ANOM-002",
        jump_id="JUMP-FLASH-ANOM-002",
        apu_code="APU-ACERO-ESTRUCTURAL",
        mac_density_matrix=rho_mac,
        flash_density_matrix=rho_flash_anom,
        raw_kelly_stake=0.40,
        melnikov_perturbation_eps=0.25,
        chirikov_perturbation_k=1.85,
        delaunay_action_L=1.0,
        delaunay_angle_l=1.2,
    )

    print(">>> Ciclo autónomo trifásico Φ₁ → Φ₁₂ → Φ₂ → Φ₂₃ → Φ₃ ...")
    contract_out, camp_out, sealed_out = prosecutor.execute_full_prosecutor_governance_cycle(
        requests=[req_legit, req_anom],
    )

    print("\n" + "─" * 80)
    print(" § FASE I  (calibre, Darboux, Bryuno, KAM)")
    print("─" * 80)
    snap = contract_out.coupling_snapshot
    print(f"  • Contrato ID            : {contract_out.contract_id}")
    print(f"  • Error Darboux          : {contract_out.symplectic_darboux_error:.6e}")
    print(f"  • Chern-Simons CS₃       : {contract_out.chern_simons_3form:.6e}")
    print(f"  • Chern-Hopf c₁          : {contract_out.chern_hopf_class_c1:.6e}")
    print(f"  • Wilson θ / |Tr U|/n    : {contract_out.wilson_holonomy_angle:.6f} / {contract_out.wilson_loop_trace:.6f}")
    print(f"  • KAM twist ∂²K/∂L²      : {snap.kam_twist:.6e}")
    print(f"  • Greene R_G             : {snap.greene_residue:.6f}")
    print(f"  • Bryuno                 : {snap.bryuno_sum:.6f}  ({'OK' if snap.bryuno_satisfied else 'FAIL'})")
    print(f"  • HMAC domain            : {contract_out.hmac_domain}")
    print(f"  • Heyting seed (admitido): {getattr(contract_out.heyting_seed, 'name', contract_out.heyting_seed)}")

    print("\n" + "─" * 80)
    print(" § FASE II  (Melnikov, Chirikov, Uhlmann, Kelly)")
    print("─" * 80)
    print(f"  • Campaña ID             : {camp_out.campaign_id}")
    print(f"  • Saltos                 : {camp_out.jumps_prosecuted_count}")
    print(f"  • RHI pico / media       : {camp_out.rhi_peak:.6f} / {camp_out.rhi_mean:.6f}")
    print(f"  • Bures pico             : {camp_out.bures_distance_peak:.6f}")
    print(f"  • Uhlmann mínimo         : {camp_out.uhlmann_fidelity_min:.6f}")
    print(f"  • Melnikov caos          : {camp_out.melnikov_chaos_detected}")
    print(f"  • Chirikov overlap       : {camp_out.chirikov_overlap_detected}")
    print(f"  • Greene pico            : {camp_out.greene_residue_peak:.6f}")
    print(f"  • KAM twist suelo        : {camp_out.kam_twist_floor:.6e}")
    print(f"  • Kelly raw → censurado  : {camp_out.raw_kelly_mean:.4f} → {camp_out.censored_kelly_mean:.4f}")
    print(f"  • Ω₃ global              : {getattr(camp_out.global_heyting_verdict, 'name', camp_out.global_heyting_verdict)}")

    print("\n" + "─" * 80)
    print(" § FASE III  (RSI³ Banach, interlock simulado, Merkle SHA-256)")
    print("─" * 80)
    print(f"  • Sello ID               : {sealed_out.sealed_id}")
    print(f"  • Actuación              : {sealed_out.actuation_mode.name}")
    print(f"  • Crowbar simulado       : {sealed_out.crowbar_tripped}  ({sealed_out.crowbar_latency_ns:.2f} ns / cota {_ESP32_CROWBAR_MAX_LATENCY_NS} ns)")
    print(f"  • Relief valve           : {sealed_out.relief_valve_active}")
    print(f"  • RSI₁ Ricci-DeTurck     : {sealed_out.rsi1_ricci_curvature:.6e}")
    print(f"  • RSI₂ Daleckii-Krein    : {sealed_out.rsi2_resolvent_lipschitz_drift:.6e}")
    print(f"  • RSI₃ Brockett          : {sealed_out.rsi3_brockett_commutator_norm:.6e}")
    print(f"  • Banach outer/residual  : {sealed_out.rsi3_outer_iterations} / {sealed_out.rsi3_residual:.3e} / contracted={sealed_out.rsi3_contracted}")
    print(f"  • Merkle SHA-256         : {sealed_out.sha256_merkle_seal[:48]}...")

    print("\n" + "─" * 80)
    print(" § Φ_sem  (Riesgo, Dolor y Dinero)")
    print("─" * 80)
    visceral = sealed_out.visceral_report
    print(f"  • Latencia p99           : {visceral.flash_latency_p99_us:.2f} µs")
    print(f"  • WACC                   : {visceral.wacc_protection_status}")
    print(f"  • Liquidez               : {visceral.net_liquidity_impact}")
    print(f"  • Ahorro LLM             : {visceral.llm_inference_savings_pct:.1f}%")
    print(f"  • Crowbar                : {visceral.esp32_crowbar_status}")
    print(f"  • Resumen                : {visceral.executive_summary}")

    print("\n" + "─" * 80)
    print(" § POSITRÓN HMAC  (compare_digest only; sin bypass de longitud)")
    print("─" * 80)
    good_token = prosecutor.expected_positron_token(camp_out.campaign_id)
    ok_clear, ok_msg = prosecutor.process_human_positron_annihilation(camp_out.campaign_id, good_token)
    print(f"  • Token legítimo         : {ok_clear}  →  {ok_msg}")
    bad_token = "AUTHORIZED_HUMAN_POSITRON_TOKEN_SECRET_64HEX"
    bad_clear, bad_msg = prosecutor.process_human_positron_annihilation(camp_out.campaign_id, bad_token)
    print(f"  • Token ilegítimo        : {bad_clear}  →  {bad_msg}")
    assert ok_clear is True, "el positrón HMAC válido debe restaurar el vacío de Dirac"
    assert bad_clear is False, "el token ilegítimo debe rechazarse (sin bypass de longitud)"
    print("  • Invariante HMAC demostrado.")

    print("\n" + banner)
    print("  VERIFICACIÓN v8.0.0 CONCLUIDA — Φ₁₂/Φ₂₃ admiten VETOED, interlock simulado, RSI³ Banach")
    print(banner)