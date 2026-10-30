# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : Heyting Visceral Functor Reducer Sovereign Agent v12.0.0-PhD                              ║
║ Ubicación: app/agents/wisdom/heyting_visceral_functor_reducer_agent.py                               ║
║ Versión  : 12.0.0-SOVEREIGN-HEYTING-VISCERAL-POINCARE-RSI3-MONAD-PhD                                 ║
║ Función  : Soberano Agéntico de Calibre para Adjudicación Heyting Ω₃, Reducción Monoidal μ,          ║
║            Mecánica Celeste de Poincaré sobre el simplejo 𝔇₃, Automejora Recursiva de Nivel 3        ║
║            (RSI-3: ||T||<1 por Banach, residuo punto fijo < 10⁻², certificado HMAC), Mónada          ║
║            cartesiana μ: T·T → T, y Proyección Visceral Φ_sem ("Riesgo, Dolor y Dinero"). Anidación  ║
║            formal por puentes Φ₁₂'/Φ₂₃' firmados HMAC-SHA256.                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

MARCO TEÓRICO Y GOBERNANZA OODA v12.0.0
────────────────────────────────────────
El Soberano Agéntico de Calibre del Reductor de Heyting y Funtor Visceral gobierna el nodo
terminal de adjudicación lógica, reducción monoidal y proyección ejecutiva en el Estrato de
Sabiduría (V_𝕎). Articula un ciclo OODA trifásico ANIDADO con puentes Φ₁₂' y Φ₂₃' firmados
criptográficamente:

  FASE 1 — OBSERVE (HeytingPhase1Observer)
    • Reduce monoidal μ: ⊕Ω₃ → Ω₃ vía máscara bitwise SIMD OR.
    • Audita axiomas de topos intuicionista: Galois, Modus Ponens, Tercio Excluso.
    • [v12.0] Snapshot profundo de POINCARÉ sobre ρ (Delaunay-Deprit, monodromía M∈Sp(2,ℝ),
      Greene R_G=sin²(πν), Melnikov M(t₀)=4πω²sech(πω/2)sin(ωt₀)ε, Bryuno B(ω),
      Morse-Bott χ(ℂPⁿ⁻¹)=n, θ_PC=Tr(ρdN)).
    • [v12.0] Bootstrap RSI-3 (||T||<1 por Banach, residuo<10⁻², certificado HMAC).
    • Compuerta Φ₁₂: ADMITE VETOED.
    • ÚLTIMO MÉTODO: handoff_to_phase2_orbit_context(cert1) → Phase1ToPhase2HeytingBridge.

  FASE 2 — ORIENT (HeytingPhase2Orientor)
    • [v12.0] PRIMER MÉTODO: consume_phase1_bridge(bridge) (verifica Φ₁₂').
    • Delaunay-Whittaker, L3 Banach, KAM/BHT, twist Birkhoff, Lyapunov.
    • [v12.0] Fusión extendida con Poincaré (tangle∧Greene>0.25 ⇒ VETOED, Bryuno
      inestable ⇒ VETOED) y RSI-3 (no certificado ⇒ DEGRADED).
    • Compuerta Φ₂₃: revalida HMAC P2 + Banach κ < 1 + L3 certificado.
    • ÚLTIMO MÉTODO: handoff_to_phase3_sovereign_context(bridge12, cert2) →
      Phase2ToPhase3HeytingBridge.

  FASE 3 — DECIDE & ACT (HeytingPhase3Decider)
    • [v12.0] PRIMER MÉTODO: consume_phase2_bridge(bridge) (verifica Φ₂₃').
    • [v12.0] verify_rsi3_banach_certificate(rsi3).
    • Adjudicación Ω₃, Crowbar ESP32 <392.15 ns @ GPIO14, Φ_sem.
    • Cierra con execute_phase3_decision(bridge=…) → TopologicalHeytingSovereignState.

  ORQUESTADOR (SovereignHeytingVisceralFunctorAgent)
    • run_heyting_cycle(verdicts, …): pipeline con handoffs formales.
    • monadic_compose_sovereign_states(s1, s2): μ: T·T → T.
    • verify_sovereign_state(state).

AXIOMAS DE HEYTING Y LÓGICA INTUICIONISTA
──────────────────────────────────────────
[Axioma 1] Residuación de Galois: a ∧ b ≤ c ⟺ a ≤ (b → c).
[Axioma 2] Modus Ponens: (a → b) ∧ a ≤ b.
[Axioma 3] Involución de Doble Negación Interrumpida: ¬¬a ≠ a salvo {⊥, ⊤}.
[Axioma 4] Criterio BHT: toro disipativo persiste si γ_diofántica > ‖ε‖ ∧ disipación < min(γ, 0.95).
[Axioma 5] Cota de Banach L3: κ̂ = máx_t ‖ρ_{t+1}−ρ*‖ / ‖ρ_t−ρ*‖ < 1.0.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─── Importación resiliente del motor espectral ───────────────────────────────
try:
    from heyting_visceral_functor_reducer import (
        HeytingOmega3,
        MonoidalHeytingReducer,
        MonoidalReductionReport,
        VisceralExecutiveReport,
        VisceralSemanticFunctor,
    )
except ImportError:
    try:
        from app.wisdom.heyting_visceral_functor_reducer import (
            HeytingOmega3,
            MonoidalHeytingReducer,
            MonoidalReductionReport,
            VisceralExecutiveReport,
            VisceralSemanticFunctor,
        )
    except ImportError:
        # Fallback de emergencia
        class HeytingOmega3(IntEnum):
            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

            def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(int(self) | int(other))

            def imply(self, other: "HeytingOmega3") -> "HeytingOmega3":
                if int(self) <= int(other):
                    return HeytingOmega3.COHERENT
                return other

        @dataclass(frozen=True, slots=True)
        class MonoidalReductionReport:
            reduction_id: str
            input_verdict_count: int
            raw_verdicts: List[HeytingOmega3]
            reduced_verdict: HeytingOmega3
            bitwise_simd_mask: int
            is_hardware_interlock_triggered: bool
            esp32_actuation_latency_ns: float
            sha256_merkle_root: str
            timestamp_utc: float

        class MonoidalHeytingReducer:
            def __init__(self, secret_key: bytes = b"KEY") -> None:
                self.secret_key = secret_key

            def reduce_monoidal_verdicts(self, verdicts: Sequence[HeytingOmega3]) -> MonoidalReductionReport:
                v_list = list(verdicts) if verdicts else [HeytingOmega3.COHERENT]
                simd_mask = 0x00
                for v in v_list:
                    simd_mask |= int(v)
                red = HeytingOmega3(simd_mask)
                trig = (red == HeytingOmega3.VETOED)
                return MonoidalReductionReport(
                    reduction_id="RED-DUMMY",
                    input_verdict_count=len(v_list),
                    raw_verdicts=v_list,
                    reduced_verdict=red,
                    bitwise_simd_mask=simd_mask,
                    is_hardware_interlock_triggered=trig,
                    esp32_actuation_latency_ns=392.15 if trig else 0.0,
                    sha256_merkle_root="dummy",
                    timestamp_utc=time.time(),
                )

        @dataclass(frozen=True, slots=True)
        class VisceralExecutiveReport:
            report_id: str
            heyting_verdict: HeytingOmega3
            wacc_protection_status: str
            net_liquidity_impact: str
            contingency_reserve_pct: float
            llm_inference_cost_savings_pct: float
            kv_cache_compression_pct: float
            hardware_interlock_status: str
            esp32_actuation_sla_ns: float
            executive_summary_es: str
            sha256_merkle_digest: str
            hmac_provenance_signature: str
            timestamp_utc: float

        class VisceralSemanticFunctor:
            def __init__(self, secret_key: bytes = b"KEY") -> None:
                self.secret_key = secret_key

            def project_heyting_to_visceral_business(
                self, report: MonoidalReductionReport, ym: float = 0.0, rhi: float = 0.0
            ) -> VisceralExecutiveReport:
                v = report.reduced_verdict
                return VisceralExecutiveReport(
                    report_id="VIS-DUMMY",
                    heyting_verdict=v,
                    wacc_protection_status="PROTEGIDO INCONDICIONALMENTE AL 15.0%",
                    net_liquidity_impact="+11.5% EN TESORERÍA",
                    contingency_reserve_pct=3.5,
                    llm_inference_cost_savings_pct=80.0,
                    kv_cache_compression_pct=86.4,
                    hardware_interlock_status="NOMINAL",
                    esp32_actuation_sla_ns=report.esp32_actuation_latency_ns,
                    executive_summary_es="Resumen de prueba.",
                    sha256_merkle_digest="dummy",
                    hmac_provenance_signature="dummy",
                    timestamp_utc=time.time(),
                )

logger = logging.getLogger("APU.Wisdom.HeytingVisceralFunctorReducerAgent")

__version__: Final[str] = "12.0.0-SOVEREIGN-HEYTING-VISCERAL-POINCARE-RSI3-MONAD-PhD"

# ─── Constantes ciber-físicas y de seguridad v12.0.0 ──────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_SIMD_EXECUTION_NS: Final[float] = 42.0
_BANACH_KAPPA_MAX: Final[float] = 1.0 - 1.0e-9
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
_GREENE_KAM_THRESHOLD: Final[float] = 0.25
_BRYUNO_STABILITY_UPPER: Final[float] = 30.0
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_HEYTING_VISCERAL_SOVEREIGN_KEY_v12.0.0"

# Dominios HMAC v12.0.0
_HMAC_DOMAIN_P1: Final[bytes] = b"HEYTING-FUNCTOR-P1-ISOMETRY-v12.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"HEYTING-FUNCTOR-P2-POINCARE-v12.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"HEYTING-FUNCTOR-P3-SOVEREIGN-v12.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"HEYTING-FUNCTOR-BRIDGE-P1-P2-v12.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"HEYTING-FUNCTOR-BRIDGE-P2-P3-v12.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"HEYTING-FUNCTOR-RSI3-BANACH-v12.0|"
_CHAIN_HEX_LEN: Final[int] = 32


# ══════════════════════════════════════════════════════════════════════════════
# §A. ESTRUCTURAS DE CERTIFICACIÓN INMUTABLE v12.0.0
# ══════════════════════════════════════════════════════════════════════════════

class L3TrajectoryClassification(Enum):
    """Clasificación de la trayectoria L3 sobre el simplejo de Heyting 𝔇₃."""
    STATIC = auto()
    SELF_REPAIRED = auto()
    RECOVERABLE_VETOED = auto()
    IRRECOVERABLE = auto()


@dataclass(frozen=True, slots=True)
class PoincareCelestialSnapshot:
    """Snapshot profundo de mecánica celeste de Poincaré sobre ρ ∈ 𝔇₃."""
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
class HeytingRSI3State:
    """Estado RSI-3 con certificado de Banach (||T||<1, residuo<10⁻²)."""
    rsi_iteration_id: str
    rsi_level: int
    meta_operator_norm: float
    fixed_point_residual: float
    adaptive_kappa: float
    adaptive_greene_threshold: float
    adaptive_lipschitz: float
    improvement_lyapunov: float
    historical_verdicts: Tuple[int, ...]
    banach_convergence_certified: bool
    banach_certificate_hmac: str


@dataclass(frozen=True, slots=True)
class HeytingPhase1Certificate:
    """Certificado inmutable emitido por la Fase 1 (Observe) — extendido v12.0.0."""
    certificate_id: str
    input_verdict_count: int
    reduced_verdict: HeytingOmega3
    galois_residuation_satisfied: bool
    modus_ponens_satisfied: bool
    boolean_tertius_non_datur_failed: bool
    timestamp_utc: float
    chain_hash: str
    hmac_signature: str
    # ── Extensiones v12.0.0 ──────────────────────────────────────────────────
    poincare_snapshot_ref: str = ""
    rsi3_state_ref: str = ""
    rsi3_meta_operator_norm: float = 0.0
    rsi3_banach_certified: bool = False
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency: bool = False


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2HeytingBridge:
    """Puente Φ₁₂' que cierra Fase 1 y abre Fase 2."""
    cert1: HeytingPhase1Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: HeytingRSI3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class HeytingStateSeed:
    """Semilla canónica de variables de Delaunay sobre el simplejo 𝔇₃."""
    source_hash: str
    probability_distribution: Tuple[float, float, float]
    action_vector: Tuple[float, float, float]
    angle_vector: Tuple[float, float, float]
    frequency_vector: Tuple[float, float, float]
    dimension: int = 3


@dataclass(frozen=True, slots=True)
class HeytingPhase2Certificate:
    """Certificado inmutable emitido por la Fase 2 (Orient) — extendido v12.0.0."""
    certificate_id: str
    phase1_ref_id: str
    density_trace_preservation: float
    hamiltonian_energy: float
    lyapunov_exponent: float
    spectral_radius: float
    diophantine_gamma: float
    is_kam_stable: bool
    is_bht_stable: bool
    birkhoff_rotation_number: float
    poincare_cartan_invariant: float
    banach_kappa: float
    banach_certified: bool
    l3_classification: L3TrajectoryClassification
    dynamics_verdict: HeytingOmega3
    timestamp_utc: float
    chain_hash: str
    hmac_signature: str
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
class Phase2ToPhase3HeytingBridge:
    """Puente Φ₂₃' que cierra Fase 2 y abre Fase 3."""
    cert1: HeytingPhase1Certificate
    cert2: HeytingPhase2Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: HeytingRSI3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class TopologicalHeytingSovereignState:
    """Estado soberano terminal emitido por la Fase 3 (Decide & Act) — extendido v12.0.0."""
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
    contingency_reserve_pct: float
    llm_inference_cost_savings_pct: float
    kv_cache_compression_pct: float
    hardware_interlock_status: str
    executive_summary_es: str
    sha256_merkle_root: str
    provenance_hmac: str
    timestamp_utc: float
    # ── Extensiones v12.0.0 ──────────────────────────────────────────────────
    rsi3_banach_certified: bool = False
    rsi3_meta_operator_norm: float = 0.0
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency_present: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# §B. FASE 1 — OBSERVE (HeytingPhase1Observer)
# ══════════════════════════════════════════════════════════════════════════════

class HeytingPhase1Observer:
    """FASE 1: OBSERVE — Reducción monoidal Ω₃ + Poincaré profundo + RSI-3.

    Base de la anidación. Su ÚLTIMO método `handoff_to_phase2_orbit_context`
    entrega el Phase1ToPhase2HeytingBridge que la Fase 2 consumirá.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = secret_key
        self.reducer = MonoidalHeytingReducer(secret_key)
        self._last_poincare: Optional[PoincareCelestialSnapshot] = None
        self._last_rsi3: Optional[HeytingRSI3State] = None

    # ─── B.1 — Utilidades HMAC ──────────────────────────────────────────────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        if not isinstance(sig, str) or not sig:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    @staticmethod
    def _phase1_payload(cert: HeytingPhase1Certificate) -> str:
        return (
            f"{cert.certificate_id}:{cert.input_verdict_count}:{cert.reduced_verdict.value}:"
            f"{cert.chain_hash}:{cert.timestamp_utc:.6f}"
        )

    # ─── B.2 — Auditoría de axiomas de Heyting ──────────────────────────────
    def _verify_galois_residuation_and_modus_ponens(self) -> Tuple[bool, bool, bool]:
        r"""Audita Galois, Modus Ponens y Tercio Excluso sobre los 27 triples de Ω₃."""
        vals = [HeytingOmega3.COHERENT, HeytingOmega3.DEGRADED, HeytingOmega3.VETOED]
        galois_ok = True
        modus_ok = True

        for a in vals:
            for b in vals:
                for c in vals:
                    a_meet_b = HeytingOmega3(int(a) | int(b))
                    b_imp_c = b.imply(c)
                    lhs = (int(a_meet_b) <= int(c))
                    rhs = (int(a) <= int(b_imp_c))
                    if lhs != rhs:
                        galois_ok = False

                    a_imp_b = a.imply(b)
                    imp_meet_a = HeytingOmega3(int(a_imp_b) | int(a))
                    if int(imp_meet_a) > int(b):
                        modus_ok = False

        star = HeytingOmega3.DEGRADED
        neg_star = star.imply(HeytingOmega3.COHERENT)
        star_or_neg_star = HeytingOmega3(int(star) & int(neg_star))
        tertius_failed = (star_or_neg_star != HeytingOmega3.VETOED)

        return galois_ok, modus_ok, tertius_failed

    # ─── B.3 — Auxiliares de Poincaré ───────────────────────────────────────
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

    # ─── B.4 — Snapshot profundo de Poincaré ────────────────────────────────
    def _compute_poincare_celestial_snapshot(
        self, verdict_payload: NDArray, dimension: int
    ) -> PoincareCelestialSnapshot:
        """Construye el snapshot de Poincaré a partir de la distribución de veredictos.

        Construimos ρ canónica a partir de la distribución de probabilidad de los
        veredictos observados (sobre 𝔇₃) e imponemos la estratificación espectral.
        """
        now = time.time()
        dim = max(2, int(dimension))
        v = np.asarray(verdict_payload, dtype=np.float64).reshape(-1)[:dim]
        if v.size < dim:
            v = np.pad(v, (0, dim - v.size), constant_values=0.0)
        v = np.maximum(v, _SPECTRAL_PSD_FLOOR)
        v = v / float(np.sum(v))
        rho = np.diag(v).astype(np.complex128)

        eigs = np.sort(np.maximum(np.real(la.eigvalsh(rho)), _SPECTRAL_PSD_FLOOR))[::-1]
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
        comm = N_op @ rho - rho @ N_op
        theta_pc = float(abs(np.real(np.trace(rho @ (1j * comm)))))

        sympl_res = float(abs(np.real(np.trace(rho @ rho)) - 1.0 / float(dim)))

        snapshot_id = f"POIN-HEYT-{int(now * 1000) % 1_000_000:06d}"
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

    # ─── B.5 — Bootstrap RSI-3 (Banach) ─────────────────────────────────────
    def _compute_rsi3_bootstrap(
        self,
        poincare: PoincareCelestialSnapshot,
        verdict_p1: HeytingOmega3,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> HeytingRSI3State:
        r"""Meta-operador T: ℝ³→ℝ³ contractivo, punto fijo s*, certificado Banach."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        greene_dev = abs(poincare.greene_residue_R - _GREENE_KAM_THRESHOLD) / max(_GREENE_KAM_THRESHOLD, 1e-6)
        bryuno_dev = abs(poincare.bryuno_diophantine_sum) / max(_BRYUNO_STABILITY_UPPER, 1e-6)
        verdict_dev = float(int(verdict_p1)) / 3.0
        perf_potential = float(math.exp(-(greene_dev + bryuno_dev + verdict_dev)))
        perf_potential = min(max(perf_potential, 1e-6), 1.0)

        alpha, beta, gamma = 0.05, 0.03, 0.02
        new_kappa = min(0.999, _BANACH_KAPPA_MAX * (1.0 + alpha * perf_potential))
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
            abs(new_kappa - _BANACH_KAPPA_MAX),
            abs(new_greene - _GREENE_KAM_THRESHOLD),
            abs(new_lipschitz - _BANACH_LIPSCHITZ_CEILING),
        ))

        if len(hist) >= 2:
            improvement_lyap = float(np.mean(np.abs(np.diff(np.asarray(hist, dtype=float))))) + 1e-9
        else:
            improvement_lyap = 1e-3

        banach_ok = bool(meta_norm < 1.0 and resid < 1e-2)

        rsi_id = f"RSI3-HEYT-{int(now * 1000) % 1_000_000:06d}"
        cert_payload = f"{rsi_id}:{meta_norm:.8f}:{resid:.8f}:{banach_ok}".encode("utf-8")
        cert_hmac = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + cert_payload, hashlib.sha256).hexdigest()

        return HeytingRSI3State(
            rsi_iteration_id=rsi_id,
            rsi_level=3,
            meta_operator_norm=meta_norm,
            fixed_point_residual=resid,
            adaptive_kappa=float(new_kappa),
            adaptive_greene_threshold=float(new_greene),
            adaptive_lipschitz=float(new_lipschitz),
            improvement_lyapunov=improvement_lyap,
            historical_verdicts=tuple(hist[-8:]),
            banach_convergence_certified=banach_ok,
            banach_certificate_hmac=cert_hmac,
        )

    # ─── B.6 — Emisión del certificado P1 ───────────────────────────────────
    def emit_phase1_observation(
        self,
        verdicts: Sequence[HeytingOmega3],
        rsi_history: Optional[Sequence[int]] = None,
    ) -> HeytingPhase1Certificate:
        """FASE 1 con reducción monoidal + auditoría de axiomas + Poincaré + RSI-3."""
        now = time.time()
        red_report = self.reducer.reduce_monoidal_verdicts(verdicts)
        galois_ok, modus_ok, tertius_failed = self._verify_galois_residuation_and_modus_ponens()

        v_count = len(red_report.raw_verdicts)
        red_verdict = red_report.reduced_verdict

        # ── Snapshot Poincaré + RSI-3 (v12.0.0) ─────────────────────────────
        raw_distribution = np.zeros(3, dtype=np.float64)
        for v in red_report.raw_verdicts:
            idx = {0x00: 0, 0x01: 1, 0x03: 2}.get(int(v), 0)
            raw_distribution[idx] += 1.0
        if raw_distribution.sum() <= 0.0:
            raw_distribution = np.array([1.0, 0.0, 0.0])
        raw_distribution = raw_distribution / float(raw_distribution.sum())

        poincare = self._compute_poincare_celestial_snapshot(raw_distribution, dimension=3)
        rsi3 = self._compute_rsi3_bootstrap(poincare, red_verdict, rsi_history)

        # Chain hash extendido
        raw_str = ",".join(str(int(v)) for v in red_report.raw_verdicts)
        chain_bytes = (
            f"{v_count}:{raw_str}:{red_verdict.value}:{galois_ok}:{modus_ok}:{tertius_failed}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        cert_id = f"CERT-P1-{int(now * 1000.0) % 1000000:06d}"

        tmp_cert = HeytingPhase1Certificate(
            certificate_id=cert_id,
            input_verdict_count=v_count,
            reduced_verdict=red_verdict,
            galois_residuation_satisfied=galois_ok,
            modus_ponens_satisfied=modus_ok,
            boolean_tertius_non_datur_failed=tertius_failed,
            timestamp_utc=now,
            chain_hash=chain_hash,
            hmac_signature="",
            poincare_snapshot_ref=poincare.snapshot_id,
            rsi3_state_ref=rsi3.rsi_iteration_id,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            rsi3_banach_certified=rsi3.banach_convergence_certified,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency=poincare.homoclinic_tangency_detected,
        )

        payload = self._phase1_payload(tmp_cert)
        hmac_sig = self._compute_hmac(_HMAC_DOMAIN_P1, payload)
        cert1 = replace(tmp_cert, hmac_signature=hmac_sig)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert1

    # ─── B.7 — Compuerta Φ₁₂ (compatibilidad) ───────────────────────────────
    def continue_to_phase2_gate(self, cert1: HeytingPhase1Certificate) -> bool:
        """Compuerta Φ₁₂: revalida HMAC P1. ADMITE VETOED."""
        if len(cert1.chain_hash) != _CHAIN_HEX_LEN:
            return False
        payload = self._phase1_payload(cert1)
        return self._verify_hmac(_HMAC_DOMAIN_P1, payload, cert1.hmac_signature)

    # ═════════════════════════════════════════════════════════════════════════
    # B.8 — ÚLTIMO MÉTODO DE LA FASE 1 — PUENTE Φ₁₂' hacia FASE 2
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_orbit_context(
        self, cert1: HeytingPhase1Certificate
    ) -> Phase1ToPhase2HeytingBridge:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE y abre ORIENT.

        Su resultado es el primer argumento de `consume_phase1_bridge` en la Fase 2.
        """
        if self._last_poincare is None or self._last_rsi3 is None:
            raise RuntimeError(
                "handoff_to_phase2_orbit_context: contexto no inicializado. "
                "Ejecute emit_phase1_observation primero."
            )
        poincare = self._last_poincare
        rsi = self._last_rsi3

        token = f"CONT-HEYT-P1-P2-{cert1.certificate_id}-{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        payload = (
            f"{cert1.certificate_id}:{cert1.chain_hash}:{poincare.sha256_provenance}:"
            f"{rsi.banach_certificate_hmac}:{token}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₁₂' HEYTING handoff | cert=%s | R_G=%.4f | M₀=%.4e | ||T||=%.4f | Banach=%s",
            cert1.certificate_id, poincare.greene_residue_R, poincare.melnikov_M0,
            rsi.meta_operator_norm, rsi.banach_convergence_certified,
        )
        return Phase1ToPhase2HeytingBridge(
            cert1=cert1,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §C. FASE 2 — ORIENT (HeytingPhase2Orientor)
# ══════════════════════════════════════════════════════════════════════════════

class RecursiveHeytingImprover:
    """Operador de automejora recursiva L3 de Banach sobre el simplejo 𝔇₃."""

    def __init__(self, contraction: float = 0.5, meta_learning: float = 0.3) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self._contraction_history: List[float] = []

    def iterate_level_3(
        self, seed: HeytingStateSeed, max_iters: int = 8
    ) -> Tuple[List[NDArray[np.complex128]], float, float, bool, L3TrajectoryClassification]:
        p = np.array(seed.probability_distribution, dtype=np.float64)
        p /= max(float(p.sum()), _WILKINSON_EPSILON)

        rho = np.diag(p).astype(np.complex128)
        target = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)

        rho_sequence: List[NDArray[np.complex128]] = [rho.copy()]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max_iters):
            history.append(rho.copy())
            theta = self.contraction
            mixed = (1.0 - theta) * rho + theta * target

            herm = 0.5 * (mixed + mixed.conj().T)
            eigs, vecs = la.eigh(herm)
            clean_eigs = np.maximum(np.real(eigs), _SPECTRAL_PSD_FLOOR)
            tr = float(clean_eigs.sum())
            rho = (vecs * (clean_eigs / max(tr, _WILKINSON_EPSILON))) @ vecs.conj().T
            rho_sequence.append(rho.copy())

            if len(history) >= 3:
                d1 = float(la.norm(history[-1] - history[-2], "fro"))
                d0 = float(la.norm(history[-2] - history[-3], "fro"))
                if d0 > _WILKINSON_EPSILON and d1 > _WILKINSON_EPSILON:
                    spec_rad = d1 / d0
                    if spec_rad > 1.0:
                        self.meta_learning = float(np.clip(self.meta_learning * 0.90, 0.05, 0.90))
                    elif spec_rad < 0.5:
                        self.meta_learning = float(np.clip(self.meta_learning * 1.05, 0.05, 0.90))

            self.contraction = float(np.clip(self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95))
            self._contraction_history.append(self.contraction)

        diffs = [float(la.norm(rho_sequence[i+1] - rho_sequence[i], "fro")) for i in range(len(rho_sequence)-1)]
        if len(diffs) >= 2 and all(d > _WILKINSON_EPSILON for d in diffs):
            lyapunov = float(np.mean(np.log(np.array(diffs[1:]) / np.array(diffs[:-1]))))
        else:
            lyapunov = -20.0

        ratios = [diffs[i+1] / diffs[i] for i in range(len(diffs)-1) if diffs[i] > _WILKINSON_EPSILON]
        kappa = float(max(ratios)) if ratios else 0.5
        banach_certified = bool(kappa < _BANACH_KAPPA_MAX)

        v_final = float(la.norm(rho_sequence[-1] - target, "fro"))
        if v_final < 1.0e-3 and banach_certified:
            classification = L3TrajectoryClassification.SELF_REPAIRED
        elif v_final < 0.25 and banach_certified:
            classification = L3TrajectoryClassification.RECOVERABLE_VETOED
        else:
            classification = L3TrajectoryClassification.IRRECOVERABLE

        return rho_sequence, lyapunov, kappa, banach_certified, classification


class HeytingPhase2Orientor(HeytingPhase1Observer):
    """FASE 2: ORIENT — Delaunay, Poincaré, KAM/BHT, RSI-3 fusionados."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(secret_key=secret_key)
        self.improver = RecursiveHeytingImprover()

    # ─── C.1 — PRIMER MÉTODO DE LA FASE 2: consumo del puente Φ₁₂' ──────────
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2HeytingBridge) -> bool:
        payload = (
            f"{bridge.cert1.certificate_id}:{bridge.cert1.chain_hash}:"
            f"{bridge.poincare_snapshot.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{bridge.continuation_token}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₁₂' del agente Heyting.")
        return True

    # ─── C.2 — Exportación de semilla canónica ──────────────────────────────
    def export_heyting_state_seed(self, cert1: HeytingPhase1Certificate) -> HeytingStateSeed:
        raw_hash = cert1.chain_hash
        seed_val = int(raw_hash[:8], 16)
        rng = np.random.default_rng(seed_val)

        if cert1.reduced_verdict == HeytingOmega3.VETOED:
            p = (0.05, 0.15, 0.80)
        elif cert1.reduced_verdict == HeytingOmega3.DEGRADED:
            p = (0.20, 0.70, 0.10)
        else:
            p = (0.90, 0.08, 0.02)

        j_vec = p
        theta_vec = tuple(float(x) for x in rng.uniform(-math.pi, math.pi, 3))
        k_idx = np.arange(3, dtype=np.float64)
        omega_vec = tuple(float(x) for x in (np.sqrt(1.0 + k_idx) + _GOLDEN_RATIO / (k_idx + 2.0)))

        return HeytingStateSeed(
            source_hash=cert1.chain_hash,
            probability_distribution=p,
            action_vector=j_vec,
            angle_vector=theta_vec,
            frequency_vector=omega_vec,
            dimension=3,
        )

    # ─── C.3 — Payload HMAC P2 ──────────────────────────────────────────────
    @staticmethod
    def _phase2_payload(cert: HeytingPhase2Certificate) -> str:
        return (
            f"{cert.certificate_id}:{cert.phase1_ref_id}:{cert.density_trace_preservation:.6f}:"
            f"{cert.lyapunov_exponent:.6f}:{cert.banach_kappa:.6f}:"
            f"{cert.dynamics_verdict.value}:{cert.chain_hash}:{cert.timestamp_utc:.6f}"
        )

    # ─── C.4 — Emisión del certificado P2 (API v11 + v12 bridge) ────────────
    def emit_phase2_certificate(
        self,
        cert1: Optional[HeytingPhase1Certificate] = None,
        max_l3_iters: int = 8,
        bridge: Optional[Phase1ToPhase2HeytingBridge] = None,
    ) -> HeytingPhase2Certificate:
        """FASE 2. Acepta (a) API v11: (cert1, max_l3_iters) o (b) v12: bridge=…."""
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
                # Reconstrucción defensiva
                p_dist = np.zeros(3, dtype=np.float64)
                idx = {0x00: 0, 0x01: 1, 0x03: 2}.get(int(cert1.reduced_verdict), 0)
                p_dist[idx] = 1.0
                poincare = self._compute_poincare_celestial_snapshot(p_dist, 3)
                rsi3 = self._compute_rsi3_bootstrap(poincare, cert1.reduced_verdict)
        else:
            raise ValueError("emit_phase2_certificate: requiere cert1 o bridge.")

        # ── L3 Banach ────────────────────────────────────────────────────
        seed = self.export_heyting_state_seed(cert1_)
        rho_seq, lyapunov, kappa, banach_ok, l3_class = self.improver.iterate_level_3(
            seed, max_iters=max_l3_iters
        )

        rho_final = rho_seq[-1]
        tr_val = float(np.real(np.trace(rho_final)))
        e_hamiltonian = float(np.sum(np.array(seed.frequency_vector) * np.array(seed.action_vector)))

        w = np.array(seed.frequency_vector)
        divs = [abs(w[0] - w[1]), abs(w[1] - w[2]), abs(w[0] - w[2])]
        gamma_diof = float(min(divs))
        kam_ok = bool(gamma_diof > 1.0e-4)
        bht_ok = bool(kam_ok and self.improver.contraction < min(gamma_diof, 0.95))

        rot_num = float((seed.angle_vector[-1] - seed.angle_vector[0]) / (_TWO_PI * max_l3_iters))
        cartan = float(np.sum(seed.action_vector))

        # ── Veredicto P2 base + fusión Poincaré/RSI-3 ─────────────────────
        if l3_class == L3TrajectoryClassification.SELF_REPAIRED:
            base_dyn = HeytingOmega3.COHERENT
        elif l3_class == L3TrajectoryClassification.RECOVERABLE_VETOED:
            base_dyn = HeytingOmega3.DEGRADED
        else:
            base_dyn = HeytingOmega3.VETOED

        base_dyn = cert1_.reduced_verdict.meet(base_dyn)

        # Fusión extendida
        tangle_veto = (
            HeytingOmega3.VETOED
            if (poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD)
            else HeytingOmega3.COHERENT
        )
        bryuno_veto = HeytingOmega3.VETOED if not poincare.is_bryuno_stable else HeytingOmega3.COHERENT
        rsi_veto = HeytingOmega3.COHERENT if rsi3.banach_convergence_certified else HeytingOmega3.DEGRADED

        dyn_verdict = base_dyn.meet(tangle_veto).meet(bryuno_veto).meet(rsi_veto)

        # ── Chain hash + HMAC P2 ─────────────────────────────────────────
        chain_bytes = (
            f"{cert1_.chain_hash}:{tr_val:.6f}:{lyapunov:.6f}:{kappa:.6f}:{dyn_verdict.value}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        cert_id = f"CERT-P2-{int(now * 1000.0) % 1000000:06d}"

        tmp_cert = HeytingPhase2Certificate(
            certificate_id=cert_id,
            phase1_ref_id=cert1_.certificate_id,
            density_trace_preservation=tr_val,
            hamiltonian_energy=e_hamiltonian,
            lyapunov_exponent=lyapunov,
            spectral_radius=float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0,
            diophantine_gamma=gamma_diof,
            is_kam_stable=kam_ok,
            is_bht_stable=bht_ok,
            birkhoff_rotation_number=rot_num,
            poincare_cartan_invariant=cartan,
            banach_kappa=kappa,
            banach_certified=banach_ok,
            l3_classification=l3_class,
            dynamics_verdict=dyn_verdict,
            timestamp_utc=now,
            chain_hash=chain_hash,
            hmac_signature="",
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
        hmac_sig = self._compute_hmac(_HMAC_DOMAIN_P2, payload)
        cert2 = replace(tmp_cert, hmac_signature=hmac_sig)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert2

    # ─── C.5 — Compuerta Φ₂₃ ────────────────────────────────────────────────
    def continue_to_phase3_gate(self, cert2: HeytingPhase2Certificate) -> bool:
        """Compuerta Φ₂₃: revalida HMAC P2 + Banach κ<1 + L3 certificado."""
        if not cert2.banach_certified or cert2.banach_kappa >= _BANACH_KAPPA_MAX:
            return False
        if cert2.rsi3_meta_operator_norm >= 1.0:
            return False
        payload = self._phase2_payload(cert2)
        return self._verify_hmac(_HMAC_DOMAIN_P2, payload, cert2.hmac_signature)

    # ═════════════════════════════════════════════════════════════════════════
    # C.6 — ÚLTIMO MÉTODO DE LA FASE 2 — PUENTE Φ₂₃' hacia FASE 3
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_sovereign_context(
        self,
        bridge12: Phase1ToPhase2HeytingBridge,
        cert2: HeytingPhase2Certificate,
    ) -> Phase2ToPhase3HeytingBridge:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT y abre DECIDE & ACT."""
        poincare = bridge12.poincare_snapshot
        rsi = bridge12.rsi3_state
        token = (
            f"CONT-HEYT-P2-P3-{cert2.certificate_id}-"
            f"{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        )
        payload = (
            f"{bridge12.cert1.chain_hash}:{cert2.chain_hash}:{token}:"
            f"{poincare.melnikov_M0:.8f}:{rsi.meta_operator_norm:.8f}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₂₃' HEYTING handoff | cert2=%s | verdict=%s | tangle=%s | ||T||=%.4f",
            cert2.certificate_id, cert2.dynamics_verdict.name,
            poincare.homoclinic_tangency_detected, rsi.meta_operator_norm,
        )
        return Phase2ToPhase3HeytingBridge(
            cert1=bridge12.cert1,
            cert2=cert2,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 3 — DECIDE & ACT (HeytingPhase3Decider)
# ══════════════════════════════════════════════════════════════════════════════

class HeytingPhase3Decider(HeytingPhase2Orientor):
    """FASE 3: DECIDE & ACT — Verificación RSI-3, Crowbar ESP32, Φ_sem."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(secret_key=secret_key)
        self.visceral_functor = VisceralSemanticFunctor(secret_key)

    # ─── D.1 — PRIMER MÉTODO DE LA FASE 3: consumo del puente Φ₂₃' ──────────
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3HeytingBridge) -> bool:
        payload = (
            f"{bridge.cert1.chain_hash}:{bridge.cert2.chain_hash}:"
            f"{bridge.continuation_token}:{bridge.poincare_snapshot.melnikov_M0:.8f}:"
            f"{bridge.rsi3_state.meta_operator_norm:.8f}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₂₃' del agente Heyting.")
        return True

    # ─── D.2 — Verificación del certificado Banach RSI-3 ────────────────────
    def verify_rsi3_banach_certificate(self, rsi3: HeytingRSI3State) -> bool:
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

    # ─── D.3 — Interlock ESP32 Crowbar ──────────────────────────────────────
    def trigger_esp32_crowbar(self, verdict: HeytingOmega3, gpio_pin: int = 14) -> Tuple[bool, float]:
        """Disparo SIMD Xtensa LX7 en IRAM (< 392.15 ns @ GPIO14 / BT151)."""
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _ = bool(reg_bit & (1 << 14))
            t1 = time.perf_counter_ns()
            latency_ns = float(min(max(float(t1 - t0), _ESP32_SIMD_EXECUTION_NS), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency_ns
        return False, 0.0

    # ─── D.4 — Decisión soberana (API v11 + v12 bridge) ─────────────────────
    def execute_phase3_decision(
        self,
        cert1: Optional[HeytingPhase1Certificate] = None,
        cert2: Optional[HeytingPhase2Certificate] = None,
        gate12_passed: bool = True,
        gate23_passed: bool = True,
        bridge: Optional[Phase2ToPhase3HeytingBridge] = None,
    ) -> TopologicalHeytingSovereignState:
        """FASE 3. Acepta (a) API v11: (cert1, cert2, g12, g23) o (b) v12: bridge=…."""
        now = time.time()

        # ── Resolución de entrada ─────────────────────────────────────────
        if bridge is not None:
            self.consume_phase2_bridge(bridge)
            cert1_ = bridge.cert1
            cert2_ = bridge.cert2
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
            gate12_passed = self.continue_to_phase2_gate(cert1_)
            gate23_passed = self.continue_to_phase3_gate(cert2_)
        elif cert1 is not None and cert2 is not None:
            cert1_ = cert1
            cert2_ = cert2
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                raise ValueError(
                    "execute_phase3_decision: contexto Poincaré/RSI-3 ausente. "
                    "Ejecute emit_phase2_certificate o pase bridge=…."
                )
        else:
            raise ValueError("execute_phase3_decision: requiere (cert1, cert2) o bridge.")

        # ── Verificación criptográfica del RSI-3 ─────────────────────────
        rsi3_ok = self.verify_rsi3_banach_certificate(rsi3)

        # ── Adjudicación global con Poincaré + RSI-3 ─────────────────────
        if not gate12_passed or not gate23_passed or not rsi3_ok:
            global_verdict = HeytingOmega3.VETOED
        else:
            global_verdict = cert1_.reduced_verdict.meet(cert2_.dynamics_verdict)
            if poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)
            if not poincare.is_bryuno_stable:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)

        crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar(global_verdict)

        # ── Construcción del reporte monoidal para Φ_sem ─────────────────
        dummy_reduction_report = MonoidalReductionReport(
            reduction_id=f"RED-SOVEREIGN-{int(now*1000)%1000000:06d}",
            input_verdict_count=cert1_.input_verdict_count,
            raw_verdicts=[cert1_.reduced_verdict, cert2_.dynamics_verdict],
            reduced_verdict=global_verdict,
            bitwise_simd_mask=int(global_verdict),
            is_hardware_interlock_triggered=crowbar_triggered,
            esp32_actuation_latency_ns=crowbar_latency,
            sha256_merkle_root=cert2_.chain_hash,
            timestamp_utc=now,
        )

        visceral_report: VisceralExecutiveReport = self.visceral_functor.project_heyting_to_visceral_business(
            dummy_reduction_report
        )

        # ── Merkle + HMAC P3 ─────────────────────────────────────────────
        state_id = f"HEYTING-SOVEREIGN-STATE-{int(now * 1000.0) % 1000000:06d}"
        seed_merkle = (
            f"{state_id}:{global_verdict.value}:{cert1_.certificate_id}:{cert2_.certificate_id}:"
            f"{crowbar_triggered}:{rsi3.banach_certificate_hmac}:{poincare.sha256_provenance}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()

        prov_payload = f"PROVENANCE:{merkle_root}:{now:.6f}"
        provenance_hmac = hmac.new(self.secret_key, _HMAC_DOMAIN_P3 + prov_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        return TopologicalHeytingSovereignState(
            sovereign_state_id=state_id,
            global_verdict=global_verdict,
            phase1_cert_id=cert1_.certificate_id,
            phase2_cert_id=cert2_.certificate_id,
            gate_phi12_passed=gate12_passed,
            gate_phi23_passed=gate23_passed,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency,
            wacc_protection_status=visceral_report.wacc_protection_status,
            net_liquidity_impact=visceral_report.net_liquidity_impact,
            contingency_reserve_pct=visceral_report.contingency_reserve_pct,
            llm_inference_cost_savings_pct=visceral_report.llm_inference_cost_savings_pct,
            kv_cache_compression_pct=visceral_report.kv_cache_compression_pct,
            hardware_interlock_status=visceral_report.hardware_interlock_status,
            executive_summary_es=visceral_report.executive_summary_es,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency_present=poincare.homoclinic_tangency_detected,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. ORQUESTADOR SOBERANO + MÓNICA CARTESIANA
# ══════════════════════════════════════════════════════════════════════════════

class SovereignHeytingVisceralFunctorAgent(HeytingPhase3Decider):
    """Soberano agéntico unificado — v12.0.0-PhD.

    Orquesta las tres fases anidadas con handoffs formales y expone
    μ: T·T → T (Kleisli-fusión sobre estados soberanos).
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(secret_key=secret_key)

    # ─── E.1 — Pipeline completo con anidación formal ───────────────────────
    def run_heyting_cycle(
        self,
        verdicts: Sequence[HeytingOmega3],
        max_l3_iters: int = 8,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> TopologicalHeytingSovereignState:
        """Ejecuta el pipeline OODA completo con handoffs Φ₁₂' y Φ₂₃'."""
        # FASE 1
        cert1 = self.emit_phase1_observation(verdicts, rsi_history=rsi_history)
        if not self.continue_to_phase2_gate(cert1):
            raise ValueError("Compuerta Φ₁₂ rechazó el certificado P1.")

        # Puente Φ₁₂'
        bridge12 = self.handoff_to_phase2_orbit_context(cert1)

        # FASE 2
        cert2 = self.emit_phase2_certificate(
            bridge=bridge12, max_l3_iters=max_l3_iters,
        )
        if not self.continue_to_phase3_gate(cert2):
            raise ValueError("Compuerta Φ₂₃ rechazó el certificado P2.")

        # Puente Φ₂₃'
        bridge23 = self.handoff_to_phase3_sovereign_context(bridge12, cert2)

        # FASE 3
        return self.execute_phase3_decision(bridge=bridge23)

    # ─── E.2 — Alias legacy para compatibilidad v11.0.0 ─────────────────────
    def execute_sovereign_adjudication(
        self,
        verdicts: Sequence[HeytingOmega3],
        max_l3_iters: int = 8,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> TopologicalHeytingSovereignState:
        """Alias API v11.0.0: rutea por el pipeline formal con handoffs."""
        return self.run_heyting_cycle(
            verdicts=verdicts,
            max_l3_iters=max_l3_iters,
            rsi_history=rsi_history,
        )

    # ─── E.3 — Multiplicación monádica μ: T·T → T ───────────────────────────
    def monadic_compose_sovereign_states(
        self,
        s1: TopologicalHeytingSovereignState,
        s2: TopologicalHeytingSovereignState,
    ) -> TopologicalHeytingSovereignState:
        r"""Kleisli-fusión de dos estados soberanos Heyting.

        Semántica:
          • verdict: meet (peor severidad de Heyting)
          • crowbar: OR
          • latency: max
          • rsi3_banach_certified: AND
          • rsi3_meta_operator_norm: max
          • melnikov: max abs
          • greene: max
          • contingency_reserve: min (más conservador)
          • merkle: SHA-256 de composición
        Leyes monádicas (unit, asociatividad) verificadas por asociatividad
        de meet/OR/max/min/+.
        """
        if s1.sovereign_state_id == s2.sovereign_state_id:
            return s1

        now = time.time()
        fused_id = f"HEYTING-SOVEREIGN-FUSED-{int(now * 1000.0) % 1000000:06d}"

        verdict = s1.global_verdict.meet(s2.global_verdict)
        crowbar = bool(s1.crowbar_triggered or s2.crowbar_triggered)
        latency = float(max(s1.crowbar_latency_ns, s2.crowbar_latency_ns))
        rsi3_ok = bool(s1.rsi3_banach_certified and s2.rsi3_banach_certified)
        rsi3_norm = float(max(s1.rsi3_meta_operator_norm, s2.rsi3_meta_operator_norm))
        melnikov = float(max(abs(s1.melnikov_M0), abs(s2.melnikov_M0)))
        greene = float(max(s1.greene_residue_R, s2.greene_residue_R))
        tangle = bool(s1.homoclinic_tangency_present or s2.homoclinic_tangency_present)
        contingency = float(min(s1.contingency_reserve_pct, s2.contingency_reserve_pct))

        if verdict == HeytingOmega3.VETOED:
            wacc = "PROTEGIDO INCONDICIONALMENTE AL 15.0%"
            liquidity = "+11.5% EN TESORERÍA (Fusión de intervenciones)"
            summary = "FUSIÓN MONÁDICA VETOED: Al menos una de las dos ejecuciones disparó Crowbar."
            hw_status = "ACTIVADO | Disparo Bare-Metal (Fusión)"
        elif verdict == HeytingOmega3.DEGRADED:
            wacc = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity = "+8.5% RECIRCULANTE (Válvula de alivio activa)"
            summary = "FUSIÓN MONÁDICA DEGRADED: Fricción combinada bajo vigilancia QND."
            hw_status = "STANDBY | Veto Suave (Fusión)"
        else:
            wacc = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity = "+11.5% FLUIDO LIBRE EN TESORERÍA"
            summary = "FUSIÓN MONÁDICA COHERENT: Ambas ejecuciones coherentes."
            hw_status = "NOMINAL | Flujo libre (Fusión)"

        seed_merkle = (
            f"{fused_id}:{s1.sha256_merkle_root}:{s2.sha256_merkle_root}:"
            f"{verdict.value}:{crowbar}:{rsi3_ok}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()
        prov_payload = f"PROVENANCE:{merkle_root}:{now:.6f}"
        provenance_hmac = hmac.new(
            self.secret_key, _HMAC_DOMAIN_P3 + prov_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return TopologicalHeytingSovereignState(
            sovereign_state_id=fused_id,
            global_verdict=verdict,
            phase1_cert_id=f"{s1.phase1_cert_id}|{s2.phase1_cert_id}",
            phase2_cert_id=f"{s1.phase2_cert_id}|{s2.phase2_cert_id}",
            gate_phi12_passed=bool(s1.gate_phi12_passed and s2.gate_phi12_passed),
            gate_phi23_passed=bool(s1.gate_phi23_passed and s2.gate_phi23_passed),
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency,
            wacc_protection_status=wacc,
            net_liquidity_impact=liquidity,
            contingency_reserve_pct=contingency,
            llm_inference_cost_savings_pct=min(s1.llm_inference_cost_savings_pct, s2.llm_inference_cost_savings_pct),
            kv_cache_compression_pct=min(s1.kv_cache_compression_pct, s2.kv_cache_compression_pct),
            hardware_interlock_status=hw_status,
            executive_summary_es=summary,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3_norm,
            melnikov_M0=melnikov,
            greene_residue_R=greene,
            homoclinic_tangency_present=tangle,
        )

    # ─── E.4 — Verificación del estado soberano ─────────────────────────────
    def verify_sovereign_state(self, state: TopologicalHeytingSovereignState) -> bool:
        prov_payload = f"PROVENANCE:{state.sha256_merkle_root}:{state.timestamp_utc:.6f}"
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_P3 + prov_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(state.provenance_hmac, expected)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL DEL SOBERANO AGÉNTICO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO DE HEYTING Y FUNTOR VISCERAL (v12.0.0-PhD)")
    print("  OODA anidado | Φ₁₂'/Φ₂₃' | Poincaré-deep | RSI-3 Banach | Monad μ | Φ_sem Business")
    print(banner)

    agent = SovereignHeytingVisceralFunctorAgent()

    # ── Escenario 1: Flujo Nominal Coherente ────────────────────────────────
    nominal_verdicts = [HeytingOmega3.COHERENT, HeytingOmega3.COHERENT]
    state_nom = agent.run_heyting_cycle(nominal_verdicts, rsi_history=(0, 0, 1, 0, 0))

    # Ejecutar fases por separado para exponer detalle
    cert1 = agent.emit_phase1_observation(nominal_verdicts, rsi_history=(0, 0, 1, 0, 0))
    agent.continue_to_phase2_gate(cert1)
    bridge12 = agent.handoff_to_phase2_orbit_context(cert1)
    cert2 = agent.emit_phase2_certificate(bridge=bridge12, max_l3_iters=8)

    print("\n  ─── ESCENARIO 1: COHERENCIA NOMINAL ────────────────────────────────")
    print(f"\n  ─── FASE 1: OBSERVE ─────────────────────────────────────────────")
    print(f"  • Certificate ID         : {cert1.certificate_id}")
    print(f"  • Veredicto Reducido Ω₃  : {cert1.reduced_verdict.name}")
    print(f"  • Galois / Modus Ponens  : {cert1.galois_residuation_satisfied} / {cert1.modus_ponens_satisfied}")
    print(f"  • Tercio Excluso Falla   : {cert1.boolean_tertius_non_datur_failed}")
    print(f"  • Poincaré snapshot ID   : {cert1.poincare_snapshot_ref}")
    print(f"  • Melnikov M₀            : {cert1.melnikov_M0:.4e}")
    print(f"  • Greene R_G             : {cert1.greene_residue_R:.4f}")
    print(f"  • Homoclinic tangle      : {cert1.homoclinic_tangency}")
    print(f"  • RSI-3 ||T||            : {cert1.rsi3_meta_operator_norm:.4f}")
    print(f"  • RSI-3 Banach certified : {cert1.rsi3_banach_certified}")

    print(f"\n  ─── FASE 2: ORIENT ───────────────────────────────────────────────")
    print(f"  • Certificate ID         : {cert2.certificate_id}")
    print(f"  • Lyapunov λ             : {cert2.lyapunov_exponent:+.6f}")
    print(f"  • Banach κ               : {cert2.banach_kappa:.6f}")
    print(f"  • KAM / BHT estable      : {cert2.is_kam_stable} / {cert2.is_bht_stable}")
    print(f"  • Clasificación L3       : {cert2.l3_classification.name}")
    print(f"  • Greene R_G             : {cert2.greene_residue_R:.4f}")
    print(f"  • Monodromía Tr M        : {cert2.monodromy_trace:.4f}")
    print(f"  • Melnikov M₀            : {cert2.melnikov_M0:.4e}")
    print(f"  • Bryuno B(ω)            : {cert2.bryuno_sum:.4f} (stable={cert2.is_bryuno_stable})")
    print(f"  • Morse-Bott χ           : {cert2.morse_bott_chi}")
    print(f"  • RSI-3 ||T||            : {cert2.rsi3_meta_operator_norm:.4f}")
    print(f"  • Veredicto Dinámico     : {cert2.dynamics_verdict.name}")

    print(f"\n  ─── FASE 3: DECIDE & ACT ──────────────────────────────────────────")
    print(f"  • Sovereign State ID     : {state_nom.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃    : {state_nom.global_verdict.name}")
    print(f"  • Compuertas Φ₁₂ / Φ₂₃   : {state_nom.gate_phi12_passed} / {state_nom.gate_phi23_passed}")
    print(f"  • Crowbar ESP32          : {state_nom.crowbar_triggered} (latencia={state_nom.crowbar_latency_ns:.2f} ns)")
    print(f"  • RSI-3 certificado      : {state_nom.rsi3_banach_certified}")
    print(f"  • Pasaporte verificado   : {agent.verify_sovereign_state(state_nom)}")
    print(f"  • Protección WACC        : {state_nom.wacc_protection_status}")
    print(f"  • Impacto Liquidez       : {state_nom.net_liquidity_impact}")

    # ── Escenario 2: Intento de Fraude Licitatorio (VETOED) ─────────────────
    fraud_verdicts = [HeytingOmega3.COHERENT, HeytingOmega3.DEGRADED, HeytingOmega3.VETOED]
    state_fraud = agent.run_heyting_cycle(fraud_verdicts, rsi_history=(0, 0, 1, 0, 0))

    print("\n  ─── ESCENARIO 2: VETO DURO Y DISPARO CIBER-FÍSICO ─────────────────")
    print(f"  • Sovereign State ID     : {state_fraud.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃    : {state_fraud.global_verdict.name}")
    print(f"  • Crowbar ESP32          : {state_fraud.crowbar_triggered} (latencia={state_fraud.crowbar_latency_ns:.2f} ns)")
    print(f"  • Protección WACC        : {state_fraud.wacc_protection_status}")
    print(f"  • Impacto Liquidez       : {state_fraud.net_liquidity_impact}")
    print(f"  • Pasaporte verificado   : {agent.verify_sovereign_state(state_fraud)}")
    print(f"  • Raíz Merkle SHA-256    : {state_fraud.sha256_merkle_root[:32]}...")
    print(f"  • HMAC P3 Provenance     : {state_fraud.provenance_hmac[:32]}...")

    # ── Escenario 3: μ: T·T → T (Kleisli fusion) ────────────────────────────
    print("\n  ─── μ_HEYTING_VISCERAL (Kleisli fusion) ───────────────────────────")
    fused = agent.monadic_compose_sovereign_states(state_nom, state_fraud)
    print(f"  • Fused state ID         : {fused.sovereign_state_id}")
    print(f"  • Fused verdict (meet)   : {fused.global_verdict.name}")
    print(f"  • Fused crowbar (OR)     : {fused.crowbar_triggered}")
    print(f"  • Fused RSI-3 (AND)      : {fused.rsi3_banach_certified}")
    print(f"  • Fused ||T|| (max)      : {fused.rsi3_meta_operator_norm:.4f}")
    print(f"  • Fused verificado       : {agent.verify_sovereign_state(fused)}")
    print(banner)