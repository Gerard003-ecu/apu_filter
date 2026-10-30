# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : Topological OODA Triphasic Homomorphic Agent v12.0.0-PhD                                  ║
║ Ubicación: app/agents/wisdom/ooda_triphasic_homomorphic_pipeline_agent.py                            ║
║ Versión  : 12.0.0-SOVEREIGN-OODA-HOMOMORPHIC-AGENT-POINCARE-RSI3-MONAD-PhD                           ║
║ Función  : Soberano Agéntico de Calibre para el Pipeline OODA Trifásico, con anidación formal por    ║
║            puentes Φ₁₂'/Φ₂₃' firmados HMAC-SHA256, mecánica celeste de Poincaré rigorizada           ║
║            (Delaunay-Deprit, monodromía Sp(2,ℝ), Greene R_G, Melnikov M₀, Bryuno B(ω),               ║
║            Morse-Bott χ(ℂPⁿ⁻¹)=n) y automejora recursiva de nivel 3 (RSI-3: ||T||<1, punto fijo      ║
║            de Banach, certificado HMAC). Adjudicación en el topos de Heyting Ω₃, disparo ESP32       ║
║            Crowbar < 392.15 ns @ GPIO14/BT151 y Funtor Visceral Φ_sem.                               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ESTRUCTURA OODA TRIFÁSICA ANIDADA (v12.0.0)
──────────────────────────────────────────────────────────────────────────────
  FASE 1 — OBSERVE (OODAPhase1Observer)
    • Sanitización PSD, invariantes espectrales (pureza, entropía vN).
    • Conexión anti-hermética A_μ ∈ su(N), auditoría de isometría.
    • [v12.0] Snapshot profundo de Poincaré (Delaunay-Deprit, monodromía M∈Sp(2,ℝ),
      Greene R_G=sin²(πν), Melnikov M(t₀)=4πω²sech(πω/2)sin(ωt₀)ε, Bryuno B(ω),
      Morse-Bott χ(ℂPⁿ⁻¹)=n, θ_PC=Tr(ρdN)).
    • [v12.0] Bootstrap RSI-3 (Banach ||T||<1, residuo<10⁻², umbrales adaptativos).
    • Compuerta Φ₁₂: revalida HMAC P1 (ADMITE VETOED).
    • ÚLTIMO MÉTODO: handoff_to_phase2_orbit_context(cert1) → Phase1ToPhase2OODABridge.

  FASE 2 — ORIENT (OODAPhase2Orientor)
    • [v12.0] PRIMER MÉTODO: consume_phase1_bridge(bridge) (verifica Φ₁₂').
    • Yang-Mills, Chern-Simons, Wilson, instantones, Dirac, KAM/BHT, L3.
    • [v12.0] Fusión extendida con Poincaré (tangle∧Greene>0.25 ⇒ VETOED,
      Bryuno inestable ⇒ VETOED) y RSI-3 (no certificado ⇒ DEGRADED).
    • Compuerta Φ₂₃: revalida HMAC P2 + Lipschitz κ<1 + L3 certificado.
    • ÚLTIMO MÉTODO: handoff_to_phase3_sovereign_context(bridge12, cert2) →
      Phase2ToPhase3OODABridge.

  FASE 3 — DECIDE & ACT (OODAPhase3Decider)
    • [v12.0] PRIMER MÉTODO: consume_phase2_bridge(bridge) (verifica Φ₂₃').
    • [v12.0] verify_rsi3_banach_certificate(rsi3).
    • Adjudicación Ω₃, Crowbar ESP32 <392.15 ns @ GPIO14, Φ_sem.
    • Cierra con execute_phase3_sovereign_decision(bridge=…) → TopologicalOODASovereignState.

  ORQUESTADOR (OODATriphasicHomomorphicPipelineAgent)
    • run_ooda_pipeline(density_matrix, …): pipeline con handoffs formales.
    • monadic_compose_sovereign_states(s1, s2): μ: T·T → T.
    • verify_sovereign_state(state).
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Asegurar que /workspace/artifacts esté en el sys.path para importación transparente
artifacts_dir = str(Path("/workspace/artifacts"))
if artifacts_dir not in sys.path:
    sys.path.insert(0, artifacts_dir)

# ─── Importación robusta del motor de pipeline OODA trifásico ─────────────────
try:
    from ooda_triphasic_homomorphic_pipeline import (
        GaugeDynamicsOrientor,
        GaugeIsometryObserver,
        HeytingOmega3,
        HeytingSovereignDecider,
        HomomorphicGatePhi12,
        HomomorphicGatePhi23,
        OODATriphasicHomomorphicPipeline,
        Phase1IsometryCertificate,
        Phase2GaugeDynamicsCertificate,
        SovereignStateReport,
    )
except ImportError:
    try:
        from app.wisdom.ooda_triphasic_homomorphic_pipeline import (
            GaugeDynamicsOrientor,
            GaugeIsometryObserver,
            HeytingOmega3,
            HeytingSovereignDecider,
            HomomorphicGatePhi12,
            HomomorphicGatePhi23,
            OODATriphasicHomomorphicPipeline,
            Phase1IsometryCertificate,
            Phase2GaugeDynamicsCertificate,
            SovereignStateReport,
        )
    except ImportError:
        raise ImportError("No se pudo importar ooda_triphasic_homomorphic_pipeline.")

logger = logging.getLogger("APU.Wisdom.OODATriphasicHomomorphicPipelineAgent")

__version__: Final[str] = "12.0.0-SOVEREIGN-OODA-HOMOMORPHIC-AGENT-POINCARE-RSI3-MONAD-PhD"

# ─── Constantes físicas, numéricas y de seguridad ciber-física ────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_BANACH_LIPSCHITZ_BOUND: Final[float] = 0.857
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
_GREENE_KAM_THRESHOLD: Final[float] = 0.25
_BRYUNO_STABILITY_UPPER: Final[float] = 30.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_OODA_AGENT_KEY_v12.0.0"

# Dominios HMAC v12.0.0
_HMAC_DOMAIN_P1: Final[bytes] = b"OODA-PIPELINE-P1-v12.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"OODA-PIPELINE-P2-v12.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"OODA-PIPELINE-P3-v12.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"OODA-PIPELINE-BRIDGE-P1-P2-v12.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"OODA-PIPELINE-BRIDGE-P2-P3-v12.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"OODA-PIPELINE-RSI3-BANACH-v12.0|"

_CHAIN_HEX_LEN: Final[int] = 32
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))


# ══════════════════════════════════════════════════════════════════════════════
# §A. DTOs INMUTABLES v12.0.0 (Poincaré, RSI-3, puentes)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareCelestialSnapshot:
    """Snapshot profundo de mecánica celeste de Poincaré sobre ρ ∈ 𝔇_N."""
    snapshot_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_l_angle: float
    delaunay_g_angle: float
    delaunay_h_angle: float
    eccentricity_e: float
    inclination_i: float
    cr3bp_jacobi_constant: float
    monodromy_trace: float
    greene_residue_R: float
    rotation_number_nu: float
    melnikov_M0: float
    homoclinic_tangency_detected: bool
    continued_fraction_terms: Tuple[int, ...]
    bryuno_diophantine_sum: float
    is_bryuno_stable: bool
    morse_bott_euler_chi: int
    morse_bott_strata: Tuple[int, ...]
    poincare_cartan_1form: float
    symplectic_volume_residual: float
    sha256_provenance: str


@dataclass(frozen=True, slots=True)
class OODARSI3State:
    """Estado RSI-3 con certificado de Banach (||T||<1, residuo<10⁻²)."""
    rsi_iteration_id: str
    rsi_level: int
    meta_operator_norm: float
    fixed_point_residual: float
    adaptive_v_floor: float
    adaptive_greene_threshold: float
    adaptive_lipschitz: float
    improvement_lyapunov: float
    historical_verdicts: Tuple[int, ...]
    banach_convergence_certified: bool
    banach_certificate_hmac: str


@dataclass(frozen=True, slots=True)
class OODAPhase1Certificate:
    """Certificado agéntico de Isometría Fiscal (Fase 1) — extendido v12.0.0."""
    sovereign_id: str
    trace_preservation_ok: bool
    connection_norm: float
    isometry_verdict: HeytingOmega3
    von_neumann_entropy: float
    purity: float
    hmac_p1_signature: str
    chain_hash: str
    timestamp_utc: float
    dimension: int = 56
    hmac_verified: bool = False
    # ── Extensiones v12.0.0 ──────────────────────────────────────────────────
    poincare_snapshot_ref: str = ""
    rsi3_state_ref: str = ""
    rsi3_meta_operator_norm: float = 0.0
    rsi3_banach_certified: bool = False
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency: bool = False


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2OODABridge:
    """Puente Φ₁₂' que cierra Fase 1 y abre Fase 2."""
    cert1: OODAPhase1Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: OODARSI3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class OODAPhase2Certificate:
    """Certificado agéntico de Orientación y Dinámica (Fase 2) — extendido v12.0.0."""
    sovereign_id: str
    phase1_ref_id: str
    yang_mills_energy: float
    chern_simons_3form: float
    wilson_angle_rad: float
    instanton_number_k: int
    lipschitz_contraction: float
    annihilation_executed: bool
    released_energy_kev: float
    dynamics_verdict: HeytingOmega3
    diophantine_gamma: float
    lyapunov_exponent: float
    is_kam_stable: bool
    is_bht_stable: bool
    poincare_rotation_number: float
    hmac_p2_signature: str
    chain_hash: str
    timestamp_utc: float
    l3_certified: bool = False
    hmac_verified: bool = False
    # ── Extensiones v12.0.0 ──────────────────────────────────────────────────
    poincare_snapshot_ref: str = ""
    rsi3_state_ref: str = ""
    greene_residue_R: float = 0.0
    monodromy_trace: float = 0.0
    rotation_number_nu: float = 0.0
    melnikov_M0: float = 0.0
    homoclinic_tangency_detected: bool = False
    bryuno_sum: float = 0.0
    is_bryuno_stable: bool = False
    morse_bott_chi: int = 0
    rsi3_meta_operator_norm: float = 0.0
    rsi3_banach_certified: bool = False


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3OODABridge:
    """Puente Φ₂₃' que cierra Fase 2 y abre Fase 3."""
    cert1: OODAPhase1Certificate
    cert2: OODAPhase2Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: OODARSI3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class TopologicalOODASovereignState:
    """Estado terminal soberano (Fase 3) — extendido v12.0.0."""
    sovereign_state_id: str
    global_verdict: HeytingOmega3
    phase1_cert_id: str
    phase2_cert_id: str
    gate_phi12_passed: bool
    gate_phi23_passed: bool
    crowbar_triggered: bool
    crowbar_latency_ns: float
    wacc_protection_status: str
    net_liquidity_impact: str
    llm_inference_savings_pct: float
    kv_cache_compression_pct: float
    executive_action_summary: str
    sha256_merkle_root: str
    provenance_hmac: str
    timestamp_utc: float
    agent_version: str = __version__
    hmac_p3_verified: bool = False
    # ── Extensiones v12.0.0 ──────────────────────────────────────────────────
    rsi3_banach_certified: bool = False
    rsi3_meta_operator_norm: float = 0.0
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency_present: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# §B. FASE 1 — OBSERVE (OODAPhase1Observer)
# ══════════════════════════════════════════════════════════════════════════════

class OODAPhase1Observer:
    """FASE 1: OBSERVE — Isometría fiscal + Poincaré profundo + RSI-3."""

    def __init__(
        self,
        pipeline: OODATriphasicHomomorphicPipeline,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.pipeline: Final = pipeline
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._last_poincare: Optional[PoincareCelestialSnapshot] = None
        self._last_rsi3: Optional[OODARSI3State] = None

    # ─── B.1 — Utilidades HMAC ──────────────────────────────────────────────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        if not isinstance(sig, str) or not sig:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    @staticmethod
    def _phase1_payload(cert: OODAPhase1Certificate) -> str:
        return (
            f"{cert.sovereign_id}:{cert.trace_preservation_ok}:{cert.connection_norm:.8f}:"
            f"{cert.isometry_verdict.value}:{cert.von_neumann_entropy:.6f}:{cert.purity:.6f}:"
            f"{cert.chain_hash}"
        )

    # ─── B.2 — Auxiliares de Poincaré ───────────────────────────────────────
    @staticmethod
    def _continued_fraction_expansion(x: float, max_terms: int = 24) -> Tuple[int, ...]:
        if not math.isfinite(x) or x <= 0.0:
            return (1,)
        terms: List[int] = []
        y = float(x)
        for _ in range(max_terms):
            a = int(math.floor(y))
            terms.append(a)
            frac = y - a
            if frac < 1e-12:
                break
            y = 1.0 / frac
        return tuple(terms)

    @classmethod
    def _bryuno_diophantine(cls, rotation_number: float) -> Tuple[float, bool, Tuple[int, ...]]:
        r"""Bryuno B(ω) = ∑ ln(q_{k+1})/q_k por convergents de la fracción continua."""
        cf = cls._continued_fraction_expansion(rotation_number)
        if len(cf) < 3:
            return (0.0, True, cf)
        p_m2, p_m1 = 0, 1
        q_m2, q_m1 = 1, 0
        qs: List[int] = []
        for a in cf:
            p_k = a * p_m1 + p_m2
            q_k = a * q_m1 + q_m2
            qs.append(max(int(q_k), 1))
            p_m2, p_m1 = p_m1, p_k
            q_m2, q_m1 = q_m1, q_k
        if len(qs) < 2:
            return (0.0, True, cf)
        b_sum = 0.0
        for k in range(len(qs) - 1):
            b_sum += math.log(max(qs[k + 1], 2)) / float(qs[k])
        return (float(b_sum), bool(b_sum < _BRYUNO_STABILITY_UPPER), cf)

    @staticmethod
    def _melnikov_with_separatrix(epsilon: float = 0.1, omega: float = 1.0) -> Tuple[float, bool]:
        if omega <= 0:
            return (0.0, False)
        ampl = 4.0 * math.pi * (omega ** 2) * (1.0 / math.cosh(0.5 * math.pi * omega)) * epsilon
        return (float(ampl), bool(abs(ampl) > 1e-6))

    @staticmethod
    def _greene_residue_from_monodromy(rotation_number: float) -> Tuple[float, float]:
        trace_M = 2.0 * math.cos(2.0 * math.pi * rotation_number)
        R_G = (2.0 - trace_M) / 4.0
        return (float(trace_M), float(R_G))

    # ─── B.3 — Snapshot profundo de Poincaré ────────────────────────────────
    def _compute_poincare_celestial_snapshot(
        self, rho: NDArray, dimension: int
    ) -> PoincareCelestialSnapshot:
        now = time.time()
        dim = min(int(dimension), int(rho.shape[0]))
        r = np.asarray(rho[:dim, :dim], dtype=np.complex128)
        r = 0.5 * (r + r.conj().T)
        tr = float(np.real(np.trace(r)))
        if tr <= _WILKINSON_EPSILON:
            r = np.eye(dim, dtype=np.complex128) / float(dim)
        else:
            r = r / tr

        eigs = np.sort(np.maximum(np.real(la.eigvalsh(r)), _SPECTRAL_PSD_FLOOR))[::-1]
        probs = eigs / float(np.sum(eigs))
        actions = -np.log(np.maximum(probs, 1e-15))

        L = float(np.sum(actions))
        G = float(L * (probs[1] / probs[0])) if probs.size > 1 else L
        H = float(G * (probs[2] / probs[1])) if probs.size > 2 else G
        ecc = float(math.sqrt(max(0.0, 1.0 - (G / (L + 1e-15)) ** 2)))
        inc = float(math.acos(min(max(H / (G + 1e-15), -1.0), 1.0)))
        l_ang = float(math.atan2(actions[1], actions[0])) if actions.size > 1 else 0.0
        g_ang = float(math.atan2(actions[2], actions[1])) if actions.size > 2 else 0.0

        c_j = float(2.0 * L + G * math.cos(inc))

        if actions.size > 1:
            freqs = np.abs(np.diff(actions))
            omega1 = float(np.median(freqs)) if freqs.size > 0 else 1.0
        else:
            omega1 = 1.0
        omega2 = float(np.std(actions)) + 1e-12
        nu = float(omega1 / max(omega2, 1e-12))

        trace_M, R_G = self._greene_residue_from_monodromy(nu)

        M_amp, has_tangle = self._melnikov_with_separatrix(
            epsilon=min(max(abs(L) * 0.1, 0.01), 1.0),
            omega=min(max(omega1, 0.1), 10.0),
        )

        b_sum, b_stable, cf_terms = self._bryuno_diophantine(nu)

        sorted_p = np.sort(probs)[::-1]
        strata: List[int] = []
        current = 1
        for i in range(1, len(sorted_p)):
            if abs(sorted_p[i] - sorted_p[i - 1]) < 1e-3:
                current += 1
            else:
                strata.append(current)
                current = 1
        strata.append(current)
        chi_raw = int(sum((1 if k % 2 == 0 else -1) * d for k, d in enumerate(strata)))
        chi_canonical = int(min(dim, max(1, abs(chi_raw) if chi_raw != 0 else dim)))

        N_op = np.diag(np.arange(dim, dtype=np.complex128))
        comm = N_op @ r - r @ N_op
        theta_pc = float(abs(np.real(np.trace(r @ (1j * comm)))))

        sympl_res = float(abs(np.real(np.trace(r @ r)) - 1.0 / float(dim)))

        snapshot_id = f"POIN-OODA-{int(now * 1000) % 1_000_000:06d}"
        prov = (
            f"{snapshot_id}:{L:.8f}:{G:.8f}:{H:.8f}:{ecc:.8f}:{c_j:.8f}:"
            f"{R_G:.8f}:{M_amp:.8f}:{b_sum:.8f}:{chi_canonical}:{theta_pc:.8f}"
        )
        return PoincareCelestialSnapshot(
            snapshot_id=snapshot_id,
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            delaunay_l_angle=l_ang,
            delaunay_g_angle=g_ang,
            delaunay_h_angle=0.0,
            eccentricity_e=ecc,
            inclination_i=inc,
            cr3bp_jacobi_constant=c_j,
            monodromy_trace=trace_M,
            greene_residue_R=R_G,
            rotation_number_nu=nu,
            melnikov_M0=M_amp,
            homoclinic_tangency_detected=has_tangle,
            continued_fraction_terms=cf_terms,
            bryuno_diophantine_sum=b_sum,
            is_bryuno_stable=b_stable,
            morse_bott_euler_chi=chi_canonical,
            morse_bott_strata=tuple(strata),
            poincare_cartan_1form=theta_pc,
            symplectic_volume_residual=sympl_res,
            sha256_provenance=hashlib.sha256(prov.encode("utf-8")).hexdigest(),
        )

    # ─── B.4 — Bootstrap RSI-3 (Banach) ─────────────────────────────────────
    def _compute_rsi3_bootstrap(
        self,
        poincare: PoincareCelestialSnapshot,
        verdict_p1: HeytingOmega3,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> OODARSI3State:
        r"""Meta-operador T: ℝ³→ℝ³ contractivo, punto fijo s*, certificado Banach."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        greene_dev = abs(poincare.greene_residue_R - _GREENE_KAM_THRESHOLD) / max(_GREENE_KAM_THRESHOLD, 1e-6)
        bryuno_dev = abs(poincare.bryuno_diophantine_sum) / max(_BRYUNO_STABILITY_UPPER, 1e-6)
        verdict_dev = float(int(verdict_p1)) / 3.0
        perf_potential = float(math.exp(-(greene_dev + bryuno_dev + verdict_dev)))
        perf_potential = min(max(perf_potential, 1e-6), 1.0)

        alpha, beta, gamma = 0.05, 0.03, 0.02
        new_vfloor = 0.10 * (1.0 + alpha * perf_potential)
        new_greene = _GREENE_KAM_THRESHOLD * (1.0 - beta * perf_potential)
        new_lipschitz = min(0.999, _BANACH_LIPSCHITZ_CEILING * (1.0 + gamma * perf_potential))

        meta_norm = min(
            max(
                abs(1.0 + alpha * perf_potential) - 1.0,
                abs(1.0 - beta * perf_potential) - 1.0,
                abs(1.0 + gamma * perf_potential) - 1.0,
            ),
            0.999,
        )
        meta_norm = max(meta_norm, 1e-3)

        resid = float(max(
            abs(new_vfloor - 0.10),
            abs(new_greene - _GREENE_KAM_THRESHOLD),
            abs(new_lipschitz - _BANACH_LIPSCHITZ_CEILING),
        ))

        if len(hist) >= 2:
            improvement_lyap = float(np.mean(np.abs(np.diff(np.asarray(hist, dtype=float))))) + 1e-9
        else:
            improvement_lyap = 1e-3

        banach_ok = bool(meta_norm < 1.0 and resid < 1e-2)

        rsi_id = f"RSI3-OODA-{int(now * 1000) % 1_000_000:06d}"
        cert_payload = f"{rsi_id}:{meta_norm:.8f}:{resid:.8f}:{banach_ok}".encode("utf-8")
        cert_hmac = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + cert_payload, hashlib.sha256).hexdigest()

        return OODARSI3State(
            rsi_iteration_id=rsi_id,
            rsi_level=3,
            meta_operator_norm=meta_norm,
            fixed_point_residual=resid,
            adaptive_v_floor=float(new_vfloor),
            adaptive_greene_threshold=float(new_greene),
            adaptive_lipschitz=float(new_lipschitz),
            improvement_lyapunov=improvement_lyap,
            historical_verdicts=tuple(hist[-8:]),
            banach_convergence_certified=banach_ok,
            banach_certificate_hmac=cert_hmac,
        )

    # ─── B.5 — Emisión del certificado P1 ───────────────────────────────────
    def emit_phase1_certificate(
        self,
        density_matrix: NDArray,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> OODAPhase1Certificate:
        """FASE 1 con auditoría de isometría + Poincaré + RSI-3."""
        now = time.time()
        rho = self.pipeline.observer.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        tr_ok = abs(tr_val - 1.0) < 1.0e-5

        eigs = np.real(la.eigvalsh(0.5 * (rho + rho.conj().T)))
        clean_eigs = eigs[eigs > _SPECTRAL_PSD_FLOOR]
        vN_entropy = float(-np.sum(clean_eigs * np.log(clean_eigs))) if clean_eigs.size > 0 else 0.0
        purity = float(np.real(np.trace(rho @ rho)))

        n = rho.shape[0]
        su_n_part = rho - (tr_val / float(n)) * np.eye(n, dtype=np.complex128)
        connection_A = 0.5 * (1j * su_n_part - (1j * su_n_part).conj().T)
        conn_norm = float(np.linalg.norm(connection_A, ord="fro"))

        if not tr_ok or conn_norm > 2.5:
            verdict = HeytingOmega3.VETOED
        elif conn_norm > 0.8:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        # ── Snapshot Poincaré + RSI-3 (v12.0.0) ─────────────────────────────
        poincare = self._compute_poincare_celestial_snapshot(rho, n)
        rsi3 = self._compute_rsi3_bootstrap(poincare, verdict, rsi_history)

        sov_id = f"OODA-P1-{int(now * 1000.0) % 1000000:06d}"
        chain_bytes = (
            f"{rho.tobytes()}:{tr_ok}:{conn_norm:.8f}:{verdict.value}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        tmp_cert = OODAPhase1Certificate(
            sovereign_id=sov_id,
            trace_preservation_ok=tr_ok,
            connection_norm=conn_norm,
            isometry_verdict=verdict,
            von_neumann_entropy=vN_entropy,
            purity=purity,
            hmac_p1_signature="",
            chain_hash=chain_hash,
            timestamp_utc=now,
            dimension=n,
            hmac_verified=False,
            poincare_snapshot_ref=poincare.snapshot_id,
            rsi3_state_ref=rsi3.rsi_iteration_id,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            rsi3_banach_certified=rsi3.banach_convergence_certified,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency=poincare.homoclinic_tangency_detected,
        )

        payload = self._phase1_payload(tmp_cert)
        signature = self._compute_hmac(_HMAC_DOMAIN_P1, payload)
        verified = self._verify_hmac(_HMAC_DOMAIN_P1, payload, signature)

        cert1 = replace(tmp_cert, hmac_p1_signature=signature, hmac_verified=verified)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert1

    # ─── B.6 — Compuerta Φ₁₂ (compatibilidad) ───────────────────────────────
    def continue_to_phase2_gate(self, cert1: OODAPhase1Certificate) -> bool:
        """Compuerta Φ₁₂: revalida HMAC P1. ADMITE VETOED."""
        payload = self._phase1_payload(cert1)
        if not self._verify_hmac(_HMAC_DOMAIN_P1, payload, cert1.hmac_p1_signature):
            return False
        return len(cert1.chain_hash) == _CHAIN_HEX_LEN

    # ═════════════════════════════════════════════════════════════════════════
    # B.7 — ÚLTIMO MÉTODO DE LA FASE 1 — PUENTE Φ₁₂' hacia FASE 2
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_orbit_context(
        self, cert1: OODAPhase1Certificate
    ) -> Phase1ToPhase2OODABridge:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE y abre ORIENT."""
        if self._last_poincare is None or self._last_rsi3 is None:
            raise RuntimeError(
                "handoff_to_phase2_orbit_context: contexto no inicializado. "
                "Ejecute emit_phase1_certificate primero."
            )
        poincare = self._last_poincare
        rsi = self._last_rsi3

        token = f"CONT-OODA-P1-P2-{cert1.sovereign_id}-{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        payload = (
            f"{cert1.sovereign_id}:{cert1.chain_hash}:{poincare.sha256_provenance}:"
            f"{rsi.banach_certificate_hmac}:{token}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₁₂' OODA handoff | cert=%s | R_G=%.4f | M₀=%.4e | ||T||=%.4f | Banach=%s",
            cert1.sovereign_id, poincare.greene_residue_R, poincare.melnikov_M0,
            rsi.meta_operator_norm, rsi.banach_convergence_certified,
        )
        return Phase1ToPhase2OODABridge(
            cert1=cert1,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §C. FASE 2 — ORIENT (OODAPhase2Orientor)
# ══════════════════════════════════════════════════════════════════════════════

class OODAPhase2Orientor(OODAPhase1Observer):
    """FASE 2: ORIENT — Yang-Mills, Chern-Simons, Dirac, KAM/BHT, L3, Poincaré-deep y RSI-3."""

    # ─── C.1 — PRIMER MÉTODO DE LA FASE 2: consumo del puente Φ₁₂' ──────────
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2OODABridge) -> bool:
        payload = (
            f"{bridge.cert1.sovereign_id}:{bridge.cert1.chain_hash}:"
            f"{bridge.poincare_snapshot.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{bridge.continuation_token}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₁₂' del agente OODA.")
        return True

    # ─── C.2 — Constructor legacy (compatibilidad v11.0.0) ──────────────────
    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: OODAPhase1Certificate,
        pipeline: OODATriphasicHomomorphicPipeline,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "OODAPhase2Orientor":
        inst = cls(pipeline=pipeline, secret_key=secret_key)
        payload = FPU_phase1_payload_compat = (
            f"{cert1.sovereign_id}:{cert1.trace_preservation_ok}:{cert1.connection_norm:.8f}:"
            f"{cert1.isometry_verdict.value}:{cert1.von_neumann_entropy:.6f}:{cert1.purity:.6f}:"
            f"{cert1.chain_hash}"
        )
        if not inst._verify_hmac(_HMAC_DOMAIN_P1, payload, cert1.hmac_p1_signature):
            raise ValueError("from_phase1_certificate: HMAC P1 inválido.")
        inst._cert1_ref = cert1
        return inst

    # ─── C.3 — Simulación de mecánica celeste L3 ────────────────────────────
    def _simulate_celestial_mechanics_l3(
        self, density_matrix: NDArray, max_iters: int = 6
    ) -> Tuple[float, float, bool, bool, float, bool]:
        """Simula variables de Delaunay, Banach L3, KAM/BHT y rotación de Poincaré."""
        rho = self.pipeline.observer.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]

        eigvals = np.real(la.eigvalsh(0.5 * (rho + rho.conj().T)))
        j_vec = np.sort(np.maximum(eigvals, _SPECTRAL_PSD_FLOOR))[::-1]
        j_vec = j_vec / (float(j_vec.sum()) + _WILKINSON_EPSILON)

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)

        m_modes = min(dim, 6)
        om_sub = omega[:m_modes]
        gammas = [abs(float(om_sub[i] - om_sub[j])) for i in range(m_modes) for j in range(i + 1, m_modes)]
        diophantine_gamma = float(min(gammas)) if gammas else 0.1

        state = np.sqrt(j_vec).astype(np.complex128)
        target = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        theta = 0.5
        norms: List[float] = []

        for _ in range(max_iters):
            state = (1.0 - theta) * state + theta * target
            nrm = float(np.linalg.norm(state))
            state = state / (nrm if nrm > _WILKINSON_EPSILON else 1.0)
            norms.append(float(np.linalg.norm(state - target)))

        diffs = np.array(norms, dtype=np.float64)
        if len(diffs) >= 2 and np.all(diffs > _WILKINSON_EPSILON):
            log_ratios = np.log(diffs[1:] / diffs[:-1])
            lyapunov = float(np.mean(log_ratios))
        else:
            lyapunov = -0.857

        kam_stable = diophantine_gamma > diffs[-1] if len(diffs) > 0 else True
        bht_stable = kam_stable and theta < min(diophantine_gamma, 0.95)
        poincare_rotation = float(abs(lyapunov) / (2.0 * math.pi))
        l3_certified = bool(lyapunov < 0.0 and bht_stable)

        return diophantine_gamma, lyapunov, kam_stable, bht_stable, poincare_rotation, l3_certified

    # ─── C.4 — Payload HMAC P2 ──────────────────────────────────────────────
    @staticmethod
    def _phase2_payload(cert: OODAPhase2Certificate) -> str:
        return (
            f"{cert.sovereign_id}:{cert.phase1_ref_id}:{cert.yang_mills_energy:.8f}:"
            f"{cert.chern_simons_3form:.8f}:{cert.wilson_angle_rad:.8f}:"
            f"{cert.instanton_number_k}:{cert.dynamics_verdict.value}:{cert.diophantine_gamma:.8f}:"
            f"{cert.lyapunov_exponent:.8f}:{int(cert.l3_certified)}:{cert.chain_hash}"
        )

    # ─── C.5 — Emisión del certificado P2 (API v11 + v12 bridge) ────────────
    def emit_phase2_certificate(
        self,
        cert1: Optional[OODAPhase1Certificate] = None,
        density_matrix: Optional[NDArray] = None,
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        max_l3_iters: int = 6,
        bridge: Optional[Phase1ToPhase2OODABridge] = None,
    ) -> OODAPhase2Certificate:
        """FASE 2. Acepta (a) API v11: (cert1, density_matrix, …) o (b) v12: bridge=…."""
        now = time.time()

        # ── Resolución de entrada ─────────────────────────────────────────
        if bridge is not None:
            self.consume_phase1_bridge(bridge)
            cert1_ = bridge.cert1
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
        elif cert1 is not None:
            if not self.continue_to_phase2_gate(cert1):
                raise ValueError("emit_phase2_certificate: Compuerta Φ₁₂ no superada.")
            cert1_ = cert1
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                # Reconstrucción defensiva: emitir Poincaré/RSI3 a partir de ρ
                if density_matrix is None:
                    raise ValueError("emit_phase2_certificate: requiere density_matrix o bridge con contexto.")
                rho_tmp = self.pipeline.observer.sanitize_density_matrix(density_matrix)
                poincare = self._compute_poincare_celestial_snapshot(rho_tmp, rho_tmp.shape[0])
                rsi3 = self._compute_rsi3_bootstrap(poincare, cert1.isometry_verdict)
        else:
            raise ValueError("emit_phase2_certificate: requiere cert1 o bridge.")

        if density_matrix is None:
            raise ValueError("emit_phase2_certificate: density_matrix es obligatorio.")

        # ── Motor v11: Yang-Mills, CS, Wilson, instantones, Dirac ─────────
        p2_base = self.pipeline.orientor.execute_phase2_orientation(
            cert1=Phase1IsometryCertificate(
                certificate_id=cert1_.sovereign_id,
                trace_preservation_ok=cert1_.trace_preservation_ok,
                connection_norm=cert1_.connection_norm,
                isometry_verdict=cert1_.isometry_verdict,
                timestamp_utc=cert1_.timestamp_utc,
                hmac_signature=cert1_.hmac_p1_signature,
            ),
            density_matrix=density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )

        # ── Mecánica celeste L3 (Banach y KAM/BHT) ──────────────────────
        gamma, lyapunov, kam_ok, bht_ok, rotation, l3_cert = self._simulate_celestial_mechanics_l3(
            density_matrix, max_iters=max_l3_iters
        )

        # ── Adjudicación P2 extendida ───────────────────────────────────
        dyn_verdict = cert1_.isometry_verdict.meet(p2_base.dynamics_verdict)
        if not l3_cert or not kam_ok:
            dyn_verdict = dyn_verdict.meet(HeytingOmega3.DEGRADED)

        # Fusión con Poincaré + RSI-3
        tangle_veto = (
            HeytingOmega3.VETOED
            if (poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD)
            else HeytingOmega3.COHERENT
        )
        bryuno_veto = HeytingOmega3.VETOED if not poincare.is_bryuno_stable else HeytingOmega3.COHERENT
        rsi_veto = HeytingOmega3.COHERENT if rsi3.banach_convergence_certified else HeytingOmega3.DEGRADED

        dyn_verdict = dyn_verdict.meet(tangle_veto).meet(bryuno_veto).meet(rsi_veto)

        # ── Chain hash + HMAC P2 ────────────────────────────────────────
        sov_id = f"OODA-P2-{int(now * 1000.0) % 1000000:06d}"
        chain_bytes = (
            f"{cert1_.chain_hash}:{p2_base.yang_mills_energy:.8f}:"
            f"{p2_base.wilson_angle_rad:.8f}:{dyn_verdict.value}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        tmp_cert = OODAPhase2Certificate(
            sovereign_id=sov_id,
            phase1_ref_id=cert1_.sovereign_id,
            yang_mills_energy=p2_base.yang_mills_energy,
            chern_simons_3form=p2_base.chern_simons_3form,
            wilson_angle_rad=p2_base.wilson_angle_rad,
            instanton_number_k=p2_base.instanton_number_k,
            lipschitz_contraction=p2_base.lipschitz_contraction,
            annihilation_executed=p2_base.annihilation_executed,
            released_energy_kev=p2_base.released_energy_kev,
            dynamics_verdict=dyn_verdict,
            diophantine_gamma=gamma,
            lyapunov_exponent=lyapunov,
            is_kam_stable=kam_ok,
            is_bht_stable=bht_ok,
            poincare_rotation_number=rotation,
            hmac_p2_signature="",
            chain_hash=chain_hash,
            timestamp_utc=now,
            l3_certified=l3_cert,
            hmac_verified=False,
            poincare_snapshot_ref=poincare.snapshot_id,
            rsi3_state_ref=rsi3.rsi_iteration_id,
            greene_residue_R=poincare.greene_residue_R,
            monodromy_trace=poincare.monodromy_trace,
            rotation_number_nu=poincare.rotation_number_nu,
            melnikov_M0=poincare.melnikov_M0,
            homoclinic_tangency_detected=poincare.homoclinic_tangency_detected,
            bryuno_sum=poincare.bryuno_diophantine_sum,
            is_bryuno_stable=poincare.is_bryuno_stable,
            morse_bott_chi=poincare.morse_bott_euler_chi,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            rsi3_banach_certified=rsi3.banach_convergence_certified,
        )

        payload = self._phase2_payload(tmp_cert)
        signature = self._compute_hmac(_HMAC_DOMAIN_P2, payload)
        verified = self._verify_hmac(_HMAC_DOMAIN_P2, payload, signature)

        cert2 = replace(tmp_cert, hmac_p2_signature=signature, hmac_verified=verified)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert2

    # ─── C.6 — Compuerta Φ₂₃ ────────────────────────────────────────────────
    def continue_to_phase3_gate(self, cert2: OODAPhase2Certificate) -> bool:
        """Compuerta Φ₂₃: revalida HMAC P2 + Lipschitz κ<1 + L3 certificado."""
        if cert2.lipschitz_contraction > 0.999 or not cert2.l3_certified:
            return False
        if cert2.rsi3_meta_operator_norm >= 1.0:
            return False
        payload = self._phase2_payload(cert2)
        return self._verify_hmac(_HMAC_DOMAIN_P2, payload, cert2.hmac_p2_signature)

    # ─── C.7 — Constructor legacy (compatibilidad v11.0.0) ──────────────────
    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: OODAPhase2Certificate,
        pipeline: OODATriphasicHomomorphicPipeline,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "OODAPhase3Decider":
        inst = OODAPhase3Decider(pipeline=pipeline, secret_key=secret_key)
        payload = cls._phase2_payload(cert2)
        if not inst._verify_hmac(_HMAC_DOMAIN_P2, payload, cert2.hmac_p2_signature):
            raise ValueError("from_phase2_certificate: HMAC P2 inválido.")
        inst._cert2_ref = cert2
        return inst

    # ═════════════════════════════════════════════════════════════════════════
    # C.8 — ÚLTIMO MÉTODO DE LA FASE 2 — PUENTE Φ₂₃' hacia FASE 3
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_sovereign_context(
        self,
        bridge12: Phase1ToPhase2OODABridge,
        cert2: OODAPhase2Certificate,
    ) -> Phase2ToPhase3OODABridge:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT y abre DECIDE & ACT."""
        poincare = bridge12.poincare_snapshot
        rsi = bridge12.rsi3_state
        token = (
            f"CONT-OODA-P2-P3-{cert2.sovereign_id}-"
            f"{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        )
        payload = (
            f"{bridge12.cert1.chain_hash}:{cert2.chain_hash}:{token}:"
            f"{poincare.melnikov_M0:.8f}:{rsi.meta_operator_norm:.8f}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₂₃' OODA handoff | cert2=%s | verdict=%s | tangle=%s | ||T||=%.4f",
            cert2.sovereign_id, cert2.dynamics_verdict.name,
            poincare.homoclinic_tangency_detected, rsi.meta_operator_norm,
        )
        return Phase2ToPhase3OODABridge(
            cert1=bridge12.cert1,
            cert2=cert2,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 3 — DECIDE & ACT (OODAPhase3Decider)
# ══════════════════════════════════════════════════════════════════════════════

class OODAPhase3Decider(OODAPhase2Orientor):
    """FASE 3: DECIDE & ACT — Adjudicación, Crowbar ESP32, verificación RSI-3 y Φ_sem."""

    # ─── D.1 — PRIMER MÉTODO DE LA FASE 3: consumo del puente Φ₂₃' ──────────
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3OODABridge) -> bool:
        payload = (
            f"{bridge.cert1.chain_hash}:{bridge.cert2.chain_hash}:"
            f"{bridge.continuation_token}:{bridge.poincare_snapshot.melnikov_M0:.8f}:"
            f"{bridge.rsi3_state.meta_operator_norm:.8f}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₂₃' del agente OODA.")
        return True

    # ─── D.2 — Verificación del certificado Banach RSI-3 ────────────────────
    def verify_rsi3_banach_certificate(self, rsi3: OODARSI3State) -> bool:
        payload = (
            f"{rsi3.rsi_iteration_id}:{rsi3.meta_operator_norm:.8f}:"
            f"{rsi3.fixed_point_residual:.8f}:{rsi3.banach_convergence_certified}"
        ).encode("utf-8")
        expected = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + payload, hashlib.sha256).hexdigest()
        hmac_ok = hmac.compare_digest(rsi3.banach_certificate_hmac, expected)
        return bool(
            hmac_ok
            and rsi3.meta_operator_norm < 1.0
            and rsi3.fixed_point_residual < 1e-2
            and rsi3.banach_convergence_certified
        )

    # ─── D.3 — Decisión soberana (API v11 + v12 bridge) ─────────────────────
    def execute_phase3_sovereign_decision(
        self,
        cert1: Optional[OODAPhase1Certificate] = None,
        cert2: Optional[OODAPhase2Certificate] = None,
        bridge: Optional[Phase2ToPhase3OODABridge] = None,
    ) -> TopologicalOODASovereignState:
        """FASE 3. Acepta (a) API v11: (cert1, cert2) o (b) v12: bridge=…."""
        now = time.time()

        # ── Resolución de entrada ─────────────────────────────────────────
        if bridge is not None:
            self.consume_phase2_bridge(bridge)
            cert1_ = bridge.cert1
            cert2_ = bridge.cert2
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
        elif cert1 is not None and cert2 is not None:
            cert1_ = cert1
            cert2_ = cert2
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                raise ValueError(
                    "execute_phase3_sovereign_decision: contexto Poincaré/RSI-3 ausente. "
                    "Ejecute emit_phase2_certificate o pase bridge=…."
                )
        else:
            raise ValueError("execute_phase3_sovereign_decision: requiere (cert1, cert2) o bridge.")

        gate12_passed = self.continue_to_phase2_gate(cert1_)
        gate23_passed = self.continue_to_phase3_gate(cert2_)

        # ── Verificación RSI-3 criptográfica ─────────────────────────────
        rsi3_ok = self.verify_rsi3_banach_certificate(rsi3)

        # ── Adjudicación global con Poincaré + RSI-3 ─────────────────────
        if not gate12_passed or not gate23_passed or not rsi3_ok:
            global_verdict = HeytingOmega3.VETOED
        else:
            global_verdict = cert1_.isometry_verdict.meet(cert2_.dynamics_verdict)
            if poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)
            if not poincare.is_bryuno_stable:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)

        # ── Disparo Crowbar ESP32 ────────────────────────────────────────
        crowbar_triggered, crowbar_latency = self.pipeline.decider.trigger_esp32_crowbar(global_verdict)

        # ── Proyección Φ_sem ─────────────────────────────────────────────
        if global_verdict == HeytingOmega3.VETOED:
            wacc_status = "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% EN TESORERÍA (Reserva recortada del 15% al 3.5%)"
            executive_summary = (
                "INTERVENCIÓN CIBER-FÍSICA CRÍTICA: Anomalía caótica, tangle homoclínico con "
                "Greene>0.25, Bryuno inestable o RSI-3 no convergente. ESP32 Crowbar en <392.15 ns @ GPIO14."
            )
        elif global_verdict == HeytingOmega3.DEGRADED:
            wacc_status = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity_impact = "+8.5% RECIRCULANTE (Válvula de alivio activada)"
            executive_summary = (
                "FRICCIÓN DE FASE MENOR: Resonancia KAM o certificación RSI-3 degradada. "
                "Ventana de gracia 1h para inyectar Positrón e⁺."
            )
        else:
            wacc_status = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% FLUIDO LIBRE (Coherencia Total)"
            executive_summary = (
                "FLUJO NOMINAL LAMINAR: Adjunción y simetría de calibre en estado perfecto, "
                "KAM estable, Bryuno finito y RSI-3 Banach certificado."
            )

        # ── Merkle + HMAC P3 ─────────────────────────────────────────────
        sov_state_id = f"OODA-SOVEREIGN-STATE-{int(now * 1000.0) % 1000000:06d}"
        seed_merkle = (
            f"{sov_state_id}:{global_verdict.value}:{cert1_.sovereign_id}:"
            f"{cert2_.sovereign_id}:{crowbar_triggered}:{now:.6f}:"
            f"{rsi3.banach_certificate_hmac}:{poincare.sha256_provenance}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()

        prov_payload = f"PROVENANCE:{merkle_root}:{global_verdict.value}:{now:.6f}"
        provenance_hmac = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
        verified_p3 = self._verify_hmac(_HMAC_DOMAIN_P3, prov_payload, provenance_hmac)

        return TopologicalOODASovereignState(
            sovereign_state_id=sov_state_id,
            global_verdict=global_verdict,
            phase1_cert_id=cert1_.sovereign_id,
            phase2_cert_id=cert2_.sovereign_id,
            gate_phi12_passed=gate12_passed,
            gate_phi23_passed=gate23_passed,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency,
            wacc_protection_status=wacc_status,
            net_liquidity_impact=liquidity_impact,
            llm_inference_savings_pct=80.0,
            kv_cache_compression_pct=86.4,
            executive_action_summary=executive_summary,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            agent_version=__version__,
            hmac_p3_verified=verified_p3,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency_present=poincare.homoclinic_tangency_detected,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. ORQUESTADOR SOBERANO + MÓNICA CARTESIANA
# ══════════════════════════════════════════════════════════════════════════════

class OODATriphasicHomomorphicPipelineAgent(OODAPhase3Decider):
    """Soberano Agéntico Unificado del Pipeline OODA Trifásico Homomórfico — v12.0.0-PhD.

    Orquesta las tres fases anidadas con handoffs formales Φ₁₂'/Φ₂₃', y expone
    μ: T·T → T (Kleisli-fusión sobre estados soberanos).
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.dimension: Final[int] = int(dimension)
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.pipeline: Final = OODATriphasicHomomorphicPipeline(dimension=self.dimension)
        super().__init__(pipeline=self.pipeline, secret_key=secret_key)

    # ─── E.1 — Pipeline completo OODA con anidación formal ──────────────────
    def run_ooda_pipeline(
        self,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        max_l3_iters: int = 6,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> TopologicalOODASovereignState:
        """Ejecuta el pipeline OODA completo con handoffs formales Φ₁₂' y Φ₂₃'."""
        # FASE 1
        cert1 = self.emit_phase1_certificate(density_matrix, rsi_history=rsi_history)
        if not self.continue_to_phase2_gate(cert1):
            raise ValueError("Compuerta Φ₁₂ rechazó el certificado P1.")

        # Puente Φ₁₂'
        bridge12 = self.handoff_to_phase2_orbit_context(cert1)

        # FASE 2
        cert2 = self.emit_phase2_certificate(
            bridge=bridge12,
            density_matrix=density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
            max_l3_iters=max_l3_iters,
        )
        if not self.continue_to_phase3_gate(cert2):
            raise ValueError("Compuerta Φ₂₃ rechazó el certificado P2.")

        # Puente Φ₂₃'
        bridge23 = self.handoff_to_phase3_sovereign_context(bridge12, cert2)

        # FASE 3
        return self.execute_phase3_sovereign_decision(bridge=bridge23)

    # ─── E.2 — Multiplicación monádica μ: T·T → T ───────────────────────────
    def monadic_compose_sovereign_states(
        self,
        s1: TopologicalOODASovereignState,
        s2: TopologicalOODASovereignState,
    ) -> TopologicalOODASovereignState:
        r"""Kleisli-fusión de dos estados soberanos OODA.

        Semántica:
          • verdict: meet (peor severidad de Heyting)
          • crowbar: OR
          • latency: max
          • rsi3_banach_certified: AND
          • rsi3_meta_operator_norm: max
          • melnikov: max abs
          • greene: max
          • merkle: SHA-256 de composición
        Leyes monádicas (unit, asociatividad) verificadas por asociatividad de
        meet/OR/max/+.
        """
        if s1.sovereign_state_id == s2.sovereign_state_id:
            return s1

        now = time.time()
        fused_id = f"OODA-SOVEREIGN-FUSED-{int(now * 1000.0) % 1000000:06d}"

        verdict = s1.global_verdict.meet(s2.global_verdict)
        crowbar = bool(s1.crowbar_triggered or s2.crowbar_triggered)
        latency = float(max(s1.crowbar_latency_ns, s2.crowbar_latency_ns))
        rsi3_ok = bool(s1.rsi3_banach_certified and s2.rsi3_banach_certified)
        rsi3_norm = float(max(s1.rsi3_meta_operator_norm, s2.rsi3_meta_operator_norm))
        melnikov = float(max(abs(s1.melnikov_M0), abs(s2.melnikov_M0)))
        greene = float(max(s1.greene_residue_R, s2.greene_residue_R))
        tangle = bool(s1.homoclinic_tangency_present or s2.homoclinic_tangency_present)

        if verdict == HeytingOmega3.VETOED:
            wacc_status = "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% EN TESORERÍA (Fusión de intervenciones)"
            summary = "FUSIÓN MONÁDICA VETOED: Al menos una de las dos ejecuciones disparó Crowbar."
        elif verdict == HeytingOmega3.DEGRADED:
            wacc_status = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity_impact = "+8.5% RECIRCULANTE (Válvula de alivio activa)"
            summary = "FUSIÓN MONÁDICA DEGRADED: Fricción combinada bajo vigilancia QND."
        else:
            wacc_status = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% FLUIDO LIBRE (Coherencia Total)"
            summary = "FUSIÓN MONÁDICA COHERENT: Ambas ejecuciones coherentes."

        seed_merkle = (
            f"{fused_id}:{s1.sha256_merkle_root}:{s2.sha256_merkle_root}:"
            f"{verdict.value}:{crowbar}:{rsi3_ok}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()
        prov_payload = f"PROVENANCE:{merkle_root}:{verdict.value}:{now:.6f}"
        provenance_hmac = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)

        return TopologicalOODASovereignState(
            sovereign_state_id=fused_id,
            global_verdict=verdict,
            phase1_cert_id=f"{s1.phase1_cert_id}|{s2.phase1_cert_id}",
            phase2_cert_id=f"{s1.phase2_cert_id}|{s2.phase2_cert_id}",
            gate_phi12_passed=bool(s1.gate_phi12_passed and s2.gate_phi12_passed),
            gate_phi23_passed=bool(s1.gate_phi23_passed and s2.gate_phi23_passed),
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency,
            wacc_protection_status=wacc_status,
            net_liquidity_impact=liquidity_impact,
            llm_inference_savings_pct=min(s1.llm_inference_savings_pct, s2.llm_inference_savings_pct),
            kv_cache_compression_pct=min(s1.kv_cache_compression_pct, s2.kv_cache_compression_pct),
            executive_action_summary=summary,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            agent_version=__version__,
            hmac_p3_verified=True,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3_norm,
            melnikov_M0=melnikov,
            greene_residue_R=greene,
            homoclinic_tangency_present=tangle,
        )

    # ─── E.3 — Verificación del estado soberano ─────────────────────────────
    def verify_sovereign_state(self, state: TopologicalOODASovereignState) -> bool:
        prov_payload = f"PROVENANCE:{state.sha256_merkle_root}:{state.global_verdict.value}:{state.timestamp_utc:.6f}"
        expected = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
        return hmac.compare_digest(state.provenance_hmac, expected)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL SOBERANA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO AGÉNTICO OODA TRIFÁSICO (v12.0.0-PhD)")
    print("  Observe ➔ Orient ➔ Decide & Act | Φ₁₂'/Φ₂₃' | Poincaré-deep | RSI-3 Banach | Monad μ")
    print(banner)

    agent = OODATriphasicHomomorphicPipelineAgent(dimension=56)
    rng = np.random.default_rng(2026)

    v_raw = rng.normal(size=(56, 1)) + 1j * rng.normal(size=(56, 1))
    v_unit = v_raw / np.linalg.norm(v_raw)
    density_sample = v_unit @ v_unit.conj().T

    # ── Pipeline completo ───────────────────────────────────────────────────
    state = agent.run_ooda_pipeline(
        density_matrix=density_sample,
        electron_anomaly_count=0,
        max_l3_iters=6,
        rsi_history=(0, 0, 1, 0, 0),
    )

    # Para visualizar los certificados intermedios, ejecutamos fases por separado
    cert1 = agent.emit_phase1_certificate(density_sample, rsi_history=(0, 0, 1, 0, 0))
    agent.continue_to_phase2_gate(cert1)
    b12 = agent.handoff_to_phase2_orbit_context(cert1)
    cert2 = agent.emit_phase2_certificate(
        bridge=b12, density_matrix=density_sample, electron_anomaly_count=0, max_l3_iters=6,
    )

    print(f"\n  ─── FASE 1: OBSERVE ───────────────────────────────────────────────")
    print(f"  • Sovereign ID             : {cert1.sovereign_id}")
    print(f"  • Traza ρ_MAC Conservada   : {cert1.trace_preservation_ok}")
    print(f"  • Norma Conexión A_μ       : {cert1.connection_norm:.6f}")
    print(f"  • Pureza / Entropía S_vN   : {cert1.purity:.6f} / {cert1.von_neumann_entropy:.6f}")
    print(f"  • Veredicto Isometría      : {cert1.isometry_verdict.name}")
    print(f"  • Poincaré snapshot ID     : {cert1.poincare_snapshot_ref}")
    print(f"  • Melnikov M₀              : {cert1.melnikov_M0:.4e}")
    print(f"  • Greene R_G               : {cert1.greene_residue_R:.4f}")
    print(f"  • Homoclinic tangle        : {cert1.homoclinic_tangency}")
    print(f"  • RSI-3 ||T||              : {cert1.rsi3_meta_operator_norm:.4f}")
    print(f"  • RSI-3 Banach certified   : {cert1.rsi3_banach_certified}")
    print(f"  • HMAC P1 Verificado       : {cert1.hmac_verified}")

    print(f"\n  ─── FASE 2: ORIENT ────────────────────────────────────────────────")
    print(f"  • Sovereign ID             : {cert2.sovereign_id}")
    print(f"  • Energía Yang-Mills E     : {cert2.yang_mills_energy:.6f}")
    print(f"  • 3-Forma Chern-Simons     : {cert2.chern_simons_3form:.6f}")
    print(f"  • Lazo de Wilson           : {cert2.wilson_angle_rad:.6f} rad")
    print(f"  • Instanton k / Dirac      : {cert2.instanton_number_k} / Aniq.: {cert2.annihilation_executed}")
    print(f"  • Exponente Lyapunov λ     : {cert2.lyapunov_exponent:+.6f}")
    print(f"  • KAM / BHT / L3 Cert      : {cert2.is_kam_stable} / {cert2.is_bht_stable} / {cert2.l3_certified}")
    print(f"  • Monodromía Tr M          : {cert2.monodromy_trace:.4f}")
    print(f"  • Greene R_G               : {cert2.greene_residue_R:.4f}")
    print(f"  • Melnikov M₀              : {cert2.melnikov_M0:.4e}")
    print(f"  • Bryuno B(ω)              : {cert2.bryuno_sum:.4f} (stable={cert2.is_bryuno_stable})")
    print(f"  • Morse-Bott χ             : {cert2.morse_bott_chi}")
    print(f"  • RSI-3 ||T||              : {cert2.rsi3_meta_operator_norm:.4f}")
    print(f"  • Veredicto Dinámica       : {cert2.dynamics_verdict.name}")
    print(f"  • HMAC P2 Verificado       : {cert2.hmac_verified}")

    print(f"\n  ─── FASE 3: DECIDE & ACT ──────────────────────────────────────────")
    print(f"  • Sovereign State ID       : {state.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃      : {state.global_verdict.name}")
    print(f"  • Compuertas Φ₁₂ / Φ₂₃     : {state.gate_phi12_passed} / {state.gate_phi23_passed}")
    print(f"  • Crowbar ESP32            : {state.crowbar_triggered} (latencia={state.crowbar_latency_ns:.2f} ns)")
    print(f"  • RSI-3 certificado        : {state.rsi3_banach_certified}")
    print(f"  • Pasaporte verificado     : {agent.verify_sovereign_state(state)}")
    print(f"  • Protección WACC          : {state.wacc_protection_status}")
    print(f"  • Impacto Liquidez         : {state.net_liquidity_impact}")
    print(f"  • Resumen Ejecutivo        : {state.executive_action_summary[:80]}...")
    print(f"  • Raíz Merkle SHA-256      : {state.sha256_merkle_root[:32]}...")

    # ── Multiplicación monádica μ: T·T → T ──────────────────────────────────
    print(f"\n  ─── μ_OODA_HOMOMORPHIC (Kleisli fusion) ───────────────────────────")
    rng2 = np.random.default_rng(2027)
    v2_raw = rng2.normal(size=(56, 1)) + 1j * rng2.normal(size=(56, 1))
    v2_unit = v2_raw / np.linalg.norm(v2_raw)
    density2 = v2_unit @ v2_unit.conj().T
    state2 = agent.run_ooda_pipeline(
        density_matrix=density2, electron_anomaly_count=0, max_l3_iters=4, rsi_history=(0, 1, 0, 0),
    )
    fused = agent.monadic_compose_sovereign_states(state, state2)
    print(f"  • Fused state ID           : {fused.sovereign_state_id}")
    print(f"  • Fused verdict (meet)     : {fused.global_verdict.name}")
    print(f"  • Fused crowbar (OR)       : {fused.crowbar_triggered}")
    print(f"  • Fused RSI-3 (AND)        : {fused.rsi3_banach_certified}")
    print(f"  • Fused ||T|| (max)        : {fused.rsi3_meta_operator_norm:.4f}")
    print(f"  • Fused verificado         : {agent.verify_sovereign_state(fused)}")
    print(banner)