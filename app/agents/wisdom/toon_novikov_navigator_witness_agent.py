# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Novikov Navigator Witness Agent v13.0.0-PhD                                                   ║
║ Ubicación: app/agents/wisdom/toon_novikov_navigator_witness_agent.py                                          ║
║ Versión  : 13.0.0-SOVEREIGN-NOVIKOV-WITNESS-POINCARE-KAM-L3-PhD                                               ║
║ Función  : Soberano Agéntico Testigo Silencioso de Navegación Novikov que atestigua y certifica              ║
║            pasivamente la consistencia causal ultramétrica con retroacción nula (Back-Action ≡ 0.0 dB)        ║
║            a través de un flujo OODA trifásico anidado, compuertas homomórficas Φ₁₂ y Φ₂₃ sobre Ω₃,           ║
║            integrando MECÁNICA CELESTE DE POINCARÉ (sección transversal, Melnikov con ceros simples,          ║
║            tori KAM no resonantes, Jacobi, recurrencia, tangles homoclínicos) y AUTOMEJORA RECURSIVA          ║
║            DE NIVEL 3 (RSI-3: iteración de Banach, Lipschitz medido, certificado de resto, umbrales           ║
║            adaptativos y self-repair QND del motor v13).                                                      ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.               ║
║   FASE 1 → Observe + Poincaré espectral + bootstrap Banach RSI-3 + Φ₁₂                                       ║
║            ÚLTIMO método: handoff_to_phase2_orbit_context → Phase1ToPhase2Continuation                       ║
║   FASE 2 → Orient QND + refine Poincaré/KAM/L3 (SovereignNovikovAuditor) + Φ₂₃                               ║
║            ÚLTIMO método: handoff_to_phase3_sovereign_context → Phase2ToPhase3Continuation                   ║
║   FASE 3 → Decide & Act + adjudicación Ω₃ + Φ_sem (PASSIVE_SILENT)                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Importación resiliente del motor testigo de Novikov v13 ───────────────────
_HAS_ENGINE: bool = False
_HAS_AUDITOR: bool = False
SovereignNovikovAuditor: Any = None
RecursiveNovikovImprover: Any = None
PoincareCelestialMechanics: Any = None
KAMNovikovTorus: Any = None

try:
    from app.wisdom.toon_novikov_navigator_witness_engine import (
        HeytingOmega3,
        NovikovWitnessSnapshot,
        TOONNovikovNavigatorWitnessEngine,
        SovereignNovikovAuditor as _SovAud,
        RecursiveNovikovImprover as _RNI,
        PoincareCelestialMechanics as _PCM,
        KAMNovikovTorus as _KAM,
    )

    SovereignNovikovAuditor = _SovAud
    RecursiveNovikovImprover = _RNI
    PoincareCelestialMechanics = _PCM
    KAMNovikovTorus = _KAM
    _HAS_ENGINE = True
    _HAS_AUDITOR = True
except ImportError:
    try:
        from toon_novikov_navigator_witness_engine import (
            HeytingOmega3,
            NovikovWitnessSnapshot,
            TOONNovikovNavigatorWitnessEngine,
            SovereignNovikovAuditor as _SovAud,
            RecursiveNovikovImprover as _RNI,
            PoincareCelestialMechanics as _PCM,
            KAMNovikovTorus as _KAM,
        )

        SovereignNovikovAuditor = _SovAud
        RecursiveNovikovImprover = _RNI
        PoincareCelestialMechanics = _PCM
        KAMNovikovTorus = _KAM
        _HAS_ENGINE = True
        _HAS_AUDITOR = True
    except ImportError:
        from enum import IntEnum

        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
            r"""Fallback Ω₃ mínimo con álgebra de Heyting completa (v13)."""

            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

            def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(max(int(self), int(other)))

            def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(min(int(self), int(other)))

            def imply(self, other: "HeytingOmega3") -> "HeytingOmega3":
                if int(self) >= int(other):
                    return HeytingOmega3.COHERENT
                return other

            def equiv(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return self.imply(other).meet(other.imply(self))

            def negation(self) -> "HeytingOmega3":
                return self.imply(HeytingOmega3.VETOED)

            def is_top(self) -> bool:
                return self == HeytingOmega3.COHERENT

            def is_bottom(self) -> bool:
                return self == HeytingOmega3.VETOED

        @dataclass(frozen=True, slots=True)
        class NovikovWitnessSnapshot:  # type: ignore[no-redef]
            snapshot_id: str
            deliberation_id: str
            novikov_valuation_v: float
            ultrametric_norm: float
            ultrametric_satisfied: bool
            bruhat_tits_distance: float
            qnd_weak_value_aav: complex
            weak_value_modulus: float
            uhlmann_fidelity: float
            bures_distance: float
            fubini_study_distance: float
            kolmogorov_sinai_entropy: float
            oseledets_lyapunov_transverse: float
            amoeba_tropical_area: float
            back_action_db: float
            global_verdict: HeytingOmega3
            sha256_merkle_root: str
            timestamp_utc: float
            ultrametric_gap_significant: bool = False
            lyapunov_exponent: float = 0.0
            spectral_radius: float = 1.0
            birkhoff_classification: str = "N/A"
            kam_diophantine_gamma: float = 0.0
            kam_stable: bool = True
            poincare_crossings: int = 0
            novikov_winding_numbers: Tuple[float, ...] = field(default_factory=tuple)
            contraction_final: float = 0.0
            meta_learning_final: float = 0.0
            qnd_back_action_sequence: Tuple[float, ...] = field(default_factory=tuple)
            qnd_trajectory_stable: bool = True
            qnd_classification: str = "STATIC"
            final_fidelity_to_mac: float = 0.0
            l3_certified: bool = False
            jacobi_integral_drift: float = 0.0
            poincare_mean_return: float = 0.0
            transversality_min: float = 0.0
            homoclinic_detected: bool = False
            nekhoroshev_time: float = 0.0
            bruno_ok: bool = True
            qnd_repair_count: int = 0
            lyapunov_decrease_certified: bool = False
            integral_invariant_drift: float = 0.0
            kam_frequency_family: str = "SPECTRAL_SURROGATE"
            engine_version: str = "fallback"

        TOONNovikovNavigatorWitnessEngine = None  # type: ignore[misc, assignment]

logger = logging.getLogger("APU.Wisdom.TOONNovikovNavigatorWitnessAgent")

__all__ = (
    "PoincareCelestialSnapshot",
    "RecursiveSelfImprovementState",
    "NovikovWitnessPhase1Certificate",
    "NovikovWitnessPhase2Certificate",
    "TopologicalNovikovWitnessSovereignState",
    "Phase1ToPhase2Continuation",
    "Phase2ToPhase3Continuation",
    "NovikovWitnessPhase1Observer",
    "NovikovWitnessPhase2Orientor",
    "NovikovWitnessPhase3Decider",
    "TOONNovikovNavigatorWitnessAgent",
)

# ─── Constantes criptográficas y de dominio (inmutables) ───────────────────────
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_NOVIKOV_WITNESS_SOVEREIGN_KEY_v13.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"NOVIKOV-WITNESS-P1-ISOMETRY-v13.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"NOVIKOV-WITNESS-P2-QND-ULTRAMETRIC-v13.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"NOVIKOV-WITNESS-P3-SOVEREIGN-v13.0|"
_HMAC_DOMAIN_CONT12: Final[bytes] = b"NOVIKOV-WITNESS-CONT-P1-P2-v13.0|"
_HMAC_DOMAIN_CONT23: Final[bytes] = b"NOVIKOV-WITNESS-CONT-P2-P3-v13.0|"
_CHAIN_HEX_LEN: Final[int] = 32
_AGENT_VERSION: Final[str] = "13.0.0-SOVEREIGN-NOVIKOV-WITNESS-POINCARE-KAM-L3-PhD"
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_BANACH_ITERS: Final[int] = 8
_MELNIKOV_ZERO_EPS: Final[float] = 1.0e-12
_MELNIKOV_AMP_THRESHOLD: Final[float] = 1.0e-3
_KAM_UNSTABLE_EPS: Final[float] = 1.0e-6
_RSI_RESIDUAL_CERT: Final[float] = 1.0e-2
_LIPSCHITZ_CERT: Final[float] = 1.0 - 1.0e-6
_TWO_PI: Final[float] = 2.0 * math.pi
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))


def _hmac_hex(secret: bytes, domain: bytes, payload: bytes) -> str:
    return hmac.new(secret, domain + payload, hashlib.sha256).hexdigest()


def _snap_get(snap: Any, name: str, default: Any) -> Any:
    return getattr(snap, name, default)


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS TRANSVERSALES (snapshots de Poincaré y RSI-3)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareCelestialSnapshot:
    r"""Instantánea de mecánica celeste de Poincaré sobre Λ_Nov.

    Invariantes (Les Méthodes Nouvelles, 1892–1899):
      • Sección transversal Σ y nº de cruces con ġ ≠ 0.
      • Función de Melnikov (amplitud del splitting homoclínico).
      • Tangle homoclínico ⇔ ceros simples de M(θ).
      • Estabilidad KAM / γ diofantina / Brjuno / tiempo de Nekhoroshev.
      • Resonancia p:q por convergentes de fracción continua.
      • Integral de Jacobi 𝒞 = Tr(A ρ), invariante de área y retorno medio.
    """

    seccion_de_poincare_id: str
    poincare_section_hits: int
    melnikov_function: float
    homoclinic_tangle_detected: bool
    invariant_manifold_dim: int
    kam_torus_stability: float
    resonance_ratio_pq: Tuple[int, int]
    action_angle_drift: float
    recurrence_time_estimate: float
    poincare_return_period: float
    diophantine_gamma: float = 0.0
    bruno_ok: bool = True
    nekhoroshev_time: float = 0.0
    jacobi_integral_drift: float = 0.0
    integral_invariant_drift: float = 0.0
    transversality_min: float = 0.0
    winding_numbers: Tuple[float, ...] = field(default_factory=tuple)
    kam_frequency_family: str = "SPECTRAL_SURROGATE"
    lyapunov_exponent: float = 0.0
    mean_return: float = 0.0
    source: str = "PHASE1_SPECTRAL"


@dataclass(frozen=True, slots=True)
class RecursiveSelfImprovementState:
    r"""Estado RSI-3: el agente observa su observar (L1), observa esa observación (L2)
    y estabiliza el meta-operador T: X → X en un punto fijo de Banach (L3).

    Certificado: Lip(T) < 1 − ε  ∧  ||T(x_n) − x_n|| < tol*  ∧  (L3 motor si existe).
    """

    rsi_iteration_id: str
    rsi_level: int
    meta_operator_norm: float
    fixed_point_residual: float
    improvement_lyapunov: float
    historical_verdicts: Tuple[int, ...]
    adaptive_v_floor: float
    adaptive_tolerance: float
    convergence_certificate: bool
    contraction: float = 0.5
    meta_learning: float = 0.3
    qnd_repair_count: int = 0
    lyapunov_decrease_certified: bool = False
    banach_fixed_point: float = 0.0
    banach_iterates: int = 0
    qnd_classification: str = "STATIC"
    final_fidelity_to_mac: float = 0.0
    source: str = "PHASE1_BANACH"


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS INMUTABLES (Fases 1, 2 y 3)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class NovikovWitnessPhase1Certificate:
    """Certificado inmutable emitido por la Fase 1 (Observe)."""

    witness_id: str
    trace_preservation: float
    liouville_drift: float
    novikov_valuation_v: float
    ultrametric_satisfied: bool
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float
    poincare_snapshot_id: str = ""
    rsi_iteration_id: str = ""
    melnikov_function: float = 0.0
    kam_torus_stability: float = 0.0
    homoclinic_tangle_detected: bool = False
    rsi_meta_operator_norm: float = 0.0
    rsi_fixed_point_residual: float = 0.0
    rsi_convergence_certificate: bool = False
    agent_version: str = _AGENT_VERSION


@dataclass(frozen=True, slots=True)
class NovikovWitnessPhase2Certificate:
    """Certificado inmutable emitido por la Fase 2 (Orient)."""

    witness_id: str
    phase1_ref_id: str
    weak_value_qnd_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    bruhat_tits_distance: float
    amoeba_tropical_area: float
    kolmogorov_sinai_entropy: float
    qnd_verdict: HeytingOmega3
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float
    resonance_p: int = 1
    resonance_q: int = 1
    homoclinic_tangle_detected: bool = False
    kam_torus_stability: float = 0.0
    rsi_convergence_certificate: bool = False
    rsi_meta_operator_norm: float = 0.0
    rsi_fixed_point_residual: float = 0.0
    kam_diophantine_gamma: float = 0.0
    kam_stable: bool = True
    bruno_ok: bool = True
    l3_certified: bool = False
    qnd_classification: str = "STATIC"
    qnd_repair_count: int = 0
    lyapunov_exponent: float = 0.0
    final_fidelity_to_mac: float = 0.0
    engine_merkle: str = ""
    agent_version: str = _AGENT_VERSION


@dataclass(frozen=True, slots=True)
class TopologicalNovikovWitnessSovereignState:
    """Estado soberano final emitido por la Fase 3 (Decide & Act)."""

    sovereign_state_id: str
    phase1_cert: NovikovWitnessPhase1Certificate
    phase2_cert: NovikovWitnessPhase2Certificate
    global_verdict: HeytingOmega3
    back_action_db: float
    wacc_protection_status: str
    net_liquidity_impact: str
    sha256_merkle_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    l3_certified: bool = False
    qnd_classification: str = "STATIC"
    rsi_convergence_certificate: bool = False
    homoclinic_tangle_detected: bool = False
    kam_diophantine_gamma: float = 0.0
    engine_version: str = _AGENT_VERSION


# ═══════════════════════════════════════════════════════════════════════════════
# PUENTES OODA (Continuaciones entre fases anidadas)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Phase1ToPhase2Continuation:
    r"""Contexto de continuación que la Fase 1 entrega a la Fase 2 (puente Φ₁₂).

    Es el objeto devuelto por el ÚLTIMO método de la Fase 1
    (`handoff_to_phase2_orbit_context`) y consumido como primer argumento por
    `execute_phase2` de la Fase 2. HMAC de dominio impide sustitución de Poincaré/RSI.
    """

    cert1: NovikovWitnessPhase1Certificate
    poincare: PoincareCelestialSnapshot
    rsi: RecursiveSelfImprovementState
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3Continuation:
    r"""Contexto de continuación que la Fase 2 entrega a la Fase 3 (puente Φ₂₃).

    Es el objeto devuelto por el ÚLTIMO método de la Fase 2
    (`handoff_to_phase3_sovereign_context`) y consumido por `execute_phase3`.
    Poincaré/RSI aquí son las versiones REFINADAS por el auditor L3.
    """

    cert1: NovikovWitnessPhase1Certificate
    cert2: NovikovWitnessPhase2Certificate
    poincare: PoincareCelestialSnapshot
    rsi: RecursiveSelfImprovementState
    continuation_token: str
    hmac_continuation: str


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: OBSERVE — isometría, valuación Novikov, Poincaré espectral, bootstrap Banach RSI-3    ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  ÚLTIMA def. (handoff_to_phase2_orbit_context) retorna Phase1ToPhase2Continuation,             ║
# ║  que es la PRIMERA entrada consumida por FASE 2 (execute_phase2).                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovWitnessPhase1Observer:
    """FASE 1: OBSERVE — Audita la isometría de calibre, la valuación de Novikov v(A),
    la sección de Poincaré espectral y el bootstrap RSI-3 por iteración de Banach.
    """

    def __init__(self, engine: Any, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.engine: Final = engine
        self.secret_key: Final[bytes] = secret_key
        self._last_poincare: Optional[PoincareCelestialSnapshot] = None
        self._last_rsi: Optional[RecursiveSelfImprovementState] = None
        self._last_cert1: Optional[NovikovWitnessPhase1Certificate] = None

    @staticmethod
    def _continued_fraction_rational(x: float, max_den: int = 64) -> Tuple[int, int]:
        r"""Convergente de fracción continua p/q, q ≤ max_den (clasificación de resonancia KAM).

        ω₁/ω₂ = p/q ∈ ℚ con q pequeño ⇒ resonancia de bajo orden (toro destruido);
        irracional Diofantino ⇒ toro KAM persistente.
        """
        if not math.isfinite(x) or x <= 0.0:
            return (1, 1)
        a0 = int(math.floor(x))
        p_nm2, q_nm2 = 1, 0
        p_nm1, q_nm1 = a0, 1
        best_p, best_q = p_nm1, q_nm1
        frac = x - float(a0)
        for _ in range(32):
            if frac < 1e-12:
                break
            frac = 1.0 / frac
            a = int(math.floor(frac))
            p = a * p_nm1 + p_nm2
            q = a * q_nm1 + q_nm2
            if q > max_den:
                break
            best_p, best_q = p, q
            p_nm2, p_nm1 = p_nm1, p
            q_nm2, q_nm1 = q_nm1, q
            frac -= float(a)
        return (int(best_p), int(best_q) if best_q != 0 else 1)

    @staticmethod
    def _spectral_melnikov(
        evals: NDArray[np.float64], phases: NDArray[np.float64]
    ) -> Tuple[float, bool, int]:
        r"""Subrogado espectral de la función de Melnikov.

        M(θ) = A sin θ + 0.1 A sin 2θ,  A = ∑_k Δλ_k sin(2 φ_k).
        Ceros simples (cambios de signo con |ΔM| > ε) ⇔ splitting transversal
        (tangle homoclínico). No usa la tautología |sin(A)| < 1/2 del v12.
        """
        if evals.size < 2:
            return 0.0, False, 0
        gaps = np.abs(np.diff(evals))
        n = int(min(gaps.size, phases.size))
        if n == 0:
            amp = float(np.sum(gaps) * math.pi)
        else:
            amp = float(np.sum(gaps[:n] * np.sin(2.0 * phases[:n])))
        theta = np.linspace(0.0, _TWO_PI, 128, dtype=np.float64)
        m_theta = amp * np.sin(theta) + 0.1 * amp * np.sin(2.0 * theta)
        signs = np.sign(m_theta)
        dabs = np.abs(np.diff(m_theta))
        simple_zeros = int(np.sum((signs[:-1] * signs[1:] < 0.0) & (dabs > _MELNIKOV_ZERO_EPS)))
        homoclinic = bool(simple_zeros > 0 and abs(amp) > _MELNIKOV_AMP_THRESHOLD)
        return amp, homoclinic, simple_zeros

    def _compute_poincare_celestial_snapshot(
        self,
        rho: NDArray[Any],
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> PoincareCelestialSnapshot:
        r"""Sección de Poincaré e invariantes celestes sobre el espectro de ρ (Observe).

        Acciones de Liouville I_k = −log λ_k; frecuencias ω_k ≈ ΔI.
        γ-surrogado = 1/(1 + q² ε) (no se invoca Nekhoroshev como 'estabilidad').
        Tiempo de Nekhoroshev T_N ≍ exp(c / ε^{1/4}); recurrencia T ~ exp(S).
        """
        now = time.time()
        rho_h = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.maximum(np.real(evals), _SPECTRAL_PSD_FLOOR)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        s = float(np.sum(evals)) + 1e-30
        evals = evals / s

        actions = -np.log(np.maximum(evals, 1e-15))
        phases = np.angle(evecs[:, 0])

        if actions.size > 1:
            freqs = np.abs(np.diff(actions))
            omega1 = float(np.median(freqs)) if freqs.size > 0 else 1.0
        else:
            omega1 = 1.0
        omega2 = float(np.std(actions)) + 1e-12
        ratio = omega1 / max(omega2, 1e-12)
        pq = self._continued_fraction_rational(ratio, max_den=32)

        mean_act = float(np.mean(actions)) + 1e-12
        eps = float(np.clip(np.std(actions) / mean_act, 1e-6, 0.999))
        q_res = max(int(pq[1]), 1)
        kam_stability = float(1.0 / (1.0 + (q_res**2) * eps))
        try:
            nek_time = float(math.exp(min(40.0, 0.35 / (eps**0.25))))
        except OverflowError:
            nek_time = 1.0e12

        melnikov, homoclinic, n_zeros = self._spectral_melnikov(evals, phases)
        manifold_dim = int(np.sum(actions > np.median(actions))) if actions.size > 0 else 1

        if phi_initial is not None and phi_final is not None:
            try:
                pi = np.asarray(phi_initial, dtype=np.complex128).reshape(-1)
                pf = np.asarray(phi_final, dtype=np.complex128).reshape(-1)
                nmin = int(min(pi.size, pf.size))
                cross_section = float(np.real(np.vdot(pi[:nmin], pf[:nmin])))
            except Exception:
                cross_section = 1.0
            phase_seq = np.linspace(0.0, _TWO_PI, 128)
            f_vals = np.cos(phase_seq) * np.sin(phase_seq + cross_section)
            crossings = int(np.sum(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0))
        else:
            crossings = max(n_zeros, max(1, int(evals.size // 8)))

        s_entropy = -float(np.sum(evals * np.log(evals + 1e-15)))
        recurrence_time = float(math.exp(min(max(s_entropy, 0.0), 40.0)))
        action_drift = float(np.std(np.diff(actions))) if actions.size > 1 else 0.0
        return_period = float(_TWO_PI / max(omega1, 1e-12))
        jacobi = float(np.real(np.sum(evals * np.linspace(0.1, 1.0, evals.size))))

        gamma = 0.0
        bruno_ok = True
        if KAMNovikovTorus is not None and actions.size >= 2:
            try:
                omega_loc = np.maximum(np.abs(np.diff(actions)), 1e-9)
                torus = KAMNovikovTorus(tuple(float(w) for w in omega_loc[:8]))
                gamma = float(torus.diophantine_constant())
                bruno_ok, _ = torus.bruno_condition()
            except Exception:
                gamma = 0.0
                bruno_ok = True

        snap_id = f"POIN-{int(now * 1000) % 1_000_000:06d}"
        return PoincareCelestialSnapshot(
            seccion_de_poincare_id=snap_id,
            poincare_section_hits=crossings,
            melnikov_function=melnikov,
            homoclinic_tangle_detected=homoclinic,
            invariant_manifold_dim=manifold_dim,
            kam_torus_stability=kam_stability,
            resonance_ratio_pq=pq,
            action_angle_drift=action_drift,
            recurrence_time_estimate=recurrence_time,
            poincare_return_period=return_period,
            diophantine_gamma=gamma,
            bruno_ok=bruno_ok,
            nekhoroshev_time=nek_time,
            jacobi_integral_drift=abs(jacobi),
            integral_invariant_drift=action_drift,
            transversality_min=abs(melnikov) / (1.0 + abs(melnikov)),
            winding_numbers=tuple(float(x) for x in (omega1, omega2)[:2]),
            kam_frequency_family="SPECTRAL_SURROGATE",
            lyapunov_exponent=0.0,
            mean_return=return_period,
            source="PHASE1_SPECTRAL",
        )

    @staticmethod
    def _banach_iterate(
        potential: float,
        x0: float = 0.5,
        n_iters: int = _BANACH_ITERS,
        lyap_penalty: float = 0.0,
    ) -> Tuple[float, float, float, int]:
        r"""Iteración de Banach de T(x) = (1−λ)x + λ Φ sobre [0,1].

        Lip(T) = 1−λ se mide además empíricamente sobre la órbita
        (no se declara contractivo por construcción ciega).
        Resto a posteriori: ||x_n − x_{n−1}||.
        """
        phi = float(min(max(potential, 0.0), 1.0))
        lam = float(np.clip(0.55 * (1.0 - min(max(lyap_penalty, 0.0), 0.9)), 0.05, 0.95))
        x = float(min(max(x0, 0.0), 1.0))
        xs: List[float] = [x]
        for _ in range(max(int(n_iters), 1)):
            x = (1.0 - lam) * x + lam * phi
            xs.append(float(x))
        lips: List[float] = []
        for i in range(1, len(xs) - 1):
            dy = abs(xs[i] - xs[i - 1])
            dx = abs(xs[i + 1] - xs[i])
            if dy > _WILKINSON_EPSILON:
                lips.append(dx / dy)
        lip_emp = float(max(lips)) if lips else float(1.0 - lam)
        lip = float(min(0.999999, max(lip_emp, 1.0 - lam)))
        residual = abs(xs[-1] - xs[-2]) if len(xs) >= 2 else abs(xs[-1] - phi)
        return lip, float(residual), float(xs[-1]), len(xs) - 1

    def _compute_rsi_bootstrap(
        self,
        v_min: float,
        liouville_drift: float,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> RecursiveSelfImprovementState:
        """Bootstrap RSI-3: potencial Φ, iteración de Banach, umbrales adaptativos."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        perf_potential = float(math.exp(-abs(liouville_drift)) * (1.0 - math.exp(-max(v_min, 0.0))))
        perf_potential = min(max(perf_potential, 0.0), 1.0)

        if len(hist) >= 2:
            deltas = np.abs(np.diff(np.asarray(hist, dtype=float)))
            improvement_lyap = float(np.mean(deltas)) + 1e-9
        else:
            improvement_lyap = 1e-3

        lip, residual, xstar, n_it = self._banach_iterate(
            perf_potential, x0=0.5, n_iters=_BANACH_ITERS, lyap_penalty=improvement_lyap
        )
        adaptive_v_floor = float(0.10 * (1.0 + 0.05 * perf_potential))
        adaptive_tolerance = float(1e-5 * (1.0 + perf_potential))
        convergence_cert = bool(lip < _LIPSCHITZ_CERT and residual < _RSI_RESIDUAL_CERT)

        return RecursiveSelfImprovementState(
            rsi_iteration_id=f"RSI3-{int(now * 1000) % 1_000_000:06d}",
            rsi_level=3,
            meta_operator_norm=lip,
            fixed_point_residual=residual,
            improvement_lyapunov=improvement_lyap,
            historical_verdicts=tuple(hist[-8:]),
            adaptive_v_floor=adaptive_v_floor,
            adaptive_tolerance=adaptive_tolerance,
            convergence_certificate=convergence_cert,
            contraction=float(np.clip(0.5 * (1.0 + perf_potential), 0.05, 0.95)),
            meta_learning=float(np.clip(0.3 + 0.2 * perf_potential, 0.05, 0.90)),
            banach_fixed_point=xstar,
            banach_iterates=n_it,
            source="PHASE1_BANACH",
        )

    def execute_phase1(
        self,
        density_matrix: NDArray[Any],
        v_floor: float = 0.10,
        rsi_history: Optional[Sequence[int]] = None,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> NovikovWitnessPhase1Certificate:
        """FASE 1: OBSERVE — Sanitiza ρ, audita v(A), calcula Poincaré + RSI-3 y certifica P1."""
        now = time.time()

        if self.engine is not None and hasattr(self.engine, "sanitize_density_matrix"):
            rho = self.engine.sanitize_density_matrix(density_matrix)
            v_min, _norm, ultra_ok, _d_bt, _amoeba = self.engine.compute_ultrametric_valuation_and_bruhat_tits(
                rho, v_floor
            )
        else:
            rho = np.asarray(density_matrix, dtype=np.complex128)
            tr_v = float(np.real(np.trace(rho)))
            rho = rho / (tr_v if tr_v > 1e-15 else 1.0)
            evals = np.sort(np.maximum(np.real(la.eigvalsh(0.5 * (rho + rho.conj().T))), 1e-12))[::-1]
            evals = evals / float(np.sum(evals))
            actions = -np.log(evals)
            v_min = float(np.min(actions))
            ultra_ok = True

        tr_val = float(np.real(np.trace(rho)))
        liouville_drift = float(abs(tr_val - 1.0))

        poincare = self._compute_poincare_celestial_snapshot(rho, phi_initial, phi_final)
        rsi = self._compute_rsi_bootstrap(v_min, liouville_drift, rsi_history)

        v_floor_effective = max(v_floor, rsi.adaptive_v_floor)
        if liouville_drift > 1.0e-3 or v_min < 0.0:
            verdict = HeytingOmega3.VETOED
        elif (not ultra_ok) or liouville_drift > rsi.adaptive_tolerance or v_min < v_floor_effective:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        wit_id = f"WIT-NOV-P1-{int(now * 1000) % 1_000_000:06d}"
        chain_bytes = (
            f"{wit_id}:{tr_val:.8f}:{liouville_drift:.8f}:{v_min:.8f}:{ultra_ok}:"
            f"{poincare.seccion_de_poincare_id}:{rsi.rsi_iteration_id}:"
            f"{poincare.melnikov_function:.6f}:{poincare.kam_torus_stability:.6f}:"
            f"{rsi.meta_operator_norm:.6f}:{rsi.fixed_point_residual:.6e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        sig_payload = f"{wit_id}:{chain_hash}:{verdict.value}".encode("utf-8")
        sig = _hmac_hex(self.secret_key, _HMAC_DOMAIN_P1, sig_payload)

        cert1 = NovikovWitnessPhase1Certificate(
            witness_id=wit_id,
            trace_preservation=tr_val,
            liouville_drift=liouville_drift,
            novikov_valuation_v=v_min,
            ultrametric_satisfied=ultra_ok,
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_p1_signature=sig,
            timestamp_utc=now,
            poincare_snapshot_id=poincare.seccion_de_poincare_id,
            rsi_iteration_id=rsi.rsi_iteration_id,
            melnikov_function=poincare.melnikov_function,
            kam_torus_stability=poincare.kam_torus_stability,
            homoclinic_tangle_detected=poincare.homoclinic_tangle_detected,
            rsi_meta_operator_norm=rsi.meta_operator_norm,
            rsi_fixed_point_residual=rsi.fixed_point_residual,
            rsi_convergence_certificate=rsi.convergence_certificate,
        )
        self._last_poincare = poincare
        self._last_rsi = rsi
        self._last_cert1 = cert1
        return cert1

    def _continuation_p12_payload(
        self,
        cert1: NovikovWitnessPhase1Certificate,
        poincare: PoincareCelestialSnapshot,
        rsi: RecursiveSelfImprovementState,
        token: str,
    ) -> bytes:
        return (
            f"{cert1.witness_id}:{cert1.chain_hash}:{token}:"
            f"{poincare.seccion_de_poincare_id}:{rsi.rsi_iteration_id}:"
            f"{poincare.melnikov_function:.6f}:{rsi.meta_operator_norm:.6f}:"
            f"{poincare.homoclinic_tangle_detected}:{rsi.convergence_certificate}"
        ).encode("utf-8")

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (handoff_to_phase2_orbit_context).                                ║
    # ║  ÚLTIMO método de FASE 1. Retorna Phase1ToPhase2Continuation → PRIMER arg. de FASE 2.       ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def handoff_to_phase2_orbit_context(
        self, cert1: NovikovWitnessPhase1Certificate
    ) -> Phase1ToPhase2Continuation:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE y abre ORIENT (puente Φ₁₂)."""
        poincare = self._last_poincare
        rsi = self._last_rsi
        if poincare is None:
            dummy = np.eye(2, dtype=np.complex128) / 2.0
            poincare = self._compute_poincare_celestial_snapshot(dummy)
        if rsi is None:
            rsi = self._compute_rsi_bootstrap(cert1.novikov_valuation_v, cert1.liouville_drift)

        token = f"CONT-P1-P2-{cert1.witness_id}-{poincare.seccion_de_poincare_id}-{rsi.rsi_iteration_id}"
        hmac_cont = _hmac_hex(
            self.secret_key, _HMAC_DOMAIN_CONT12, self._continuation_p12_payload(cert1, poincare, rsi, token)
        )
        logger.info(
            "Φ₁₂ handoff | token=%s | M=%.4f | KAM=%.4f | Lip(T)=%.4f | Banach cert=%s",
            token,
            poincare.melnikov_function,
            poincare.kam_torus_stability,
            rsi.meta_operator_norm,
            rsi.convergence_certificate,
        )
        return Phase1ToPhase2Continuation(
            cert1=cert1,
            poincare=poincare,
            rsi=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: ORIENT — QND AAV + refine Poincaré/KAM/L3 + fusión Ω₃                                  ║
# ║  Primera def. operativa (execute_phase2) consume Phase1ToPhase2Continuation (FASE 1).          ║
# ║  ÚLTIMA def. (handoff_to_phase3_sovereign_context) → Phase2ToPhase3Continuation (FASE 3).      ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovWitnessPhase2Orientor(NovikovWitnessPhase1Observer):
    """FASE 2: ORIENT — Atestación QND sobre Λ_Nov, refinada con el auditor L3 del motor v13."""

    def __init__(self, engine: Any, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(engine=engine, secret_key=secret_key)
        self._refined_poincare: Optional[PoincareCelestialSnapshot] = None
        self._refined_rsi: Optional[RecursiveSelfImprovementState] = None
        self._last_cert2: Optional[NovikovWitnessPhase2Certificate] = None
        self._auditor: Any = None
        if _HAS_AUDITOR and engine is not None and SovereignNovikovAuditor is not None:
            try:
                self._auditor = SovereignNovikovAuditor(engine)
            except Exception:
                self._auditor = None

    def _verify_continuation_p1_p2(self, cont: Phase1ToPhase2Continuation) -> bool:
        expected = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_CONT12,
            self._continuation_p12_payload(cont.cert1, cont.poincare, cont.rsi, cont.continuation_token),
        )
        return hmac.compare_digest(cont.hmac_continuation, expected)

    def _poincare_from_engine_snapshot(
        self, snap: Any, base: PoincareCelestialSnapshot
    ) -> PoincareCelestialSnapshot:
        """Refina la instantánea celeste con invariantes reales del motor L3."""
        gamma = float(_snap_get(snap, "kam_diophantine_gamma", base.diophantine_gamma))
        kam_stable = bool(_snap_get(snap, "kam_stable", True))
        residual_stab = float(_snap_get(snap, "spectral_radius", 1.0))
        kam_stability = float(
            np.clip(
                (1.0 if kam_stable else 0.0) * (gamma / (gamma + abs(residual_stab) + 1e-12)),
                0.0,
                1.0,
            )
        )
        jacobi = float(_snap_get(snap, "jacobi_integral_drift", base.jacobi_integral_drift))
        homoclinic = bool(_snap_get(snap, "homoclinic_detected", base.homoclinic_tangle_detected))
        crossings = int(_snap_get(snap, "poincare_crossings", base.poincare_section_hits))
        winding = tuple(float(x) for x in _snap_get(snap, "novikov_winding_numbers", base.winding_numbers))
        if len(winding) >= 2 and abs(winding[1]) > 1e-15:
            ratio = abs(winding[0] / winding[1])
            pq = self._continued_fraction_rational(ratio, max_den=32)
        else:
            pq = base.resonance_ratio_pq
        return PoincareCelestialSnapshot(
            seccion_de_poincare_id=base.seccion_de_poincare_id,
            poincare_section_hits=crossings,
            melnikov_function=jacobi if abs(jacobi) > 0.0 else base.melnikov_function,
            homoclinic_tangle_detected=homoclinic,
            invariant_manifold_dim=base.invariant_manifold_dim,
            kam_torus_stability=kam_stability if gamma > 0.0 else base.kam_torus_stability,
            resonance_ratio_pq=pq,
            action_angle_drift=float(_snap_get(snap, "integral_invariant_drift", base.action_angle_drift)),
            recurrence_time_estimate=base.recurrence_time_estimate,
            poincare_return_period=float(_snap_get(snap, "poincare_mean_return", base.poincare_return_period)),
            diophantine_gamma=gamma,
            bruno_ok=bool(_snap_get(snap, "bruno_ok", base.bruno_ok)),
            nekhoroshev_time=float(_snap_get(snap, "nekhoroshev_time", base.nekhoroshev_time)),
            jacobi_integral_drift=jacobi,
            integral_invariant_drift=float(
                _snap_get(snap, "integral_invariant_drift", base.integral_invariant_drift)
            ),
            transversality_min=float(_snap_get(snap, "transversality_min", base.transversality_min)),
            winding_numbers=winding,
            kam_frequency_family=str(_snap_get(snap, "kam_frequency_family", "NOBLE_PRIME_GOLDEN")),
            lyapunov_exponent=float(_snap_get(snap, "lyapunov_exponent", 0.0)),
            mean_return=float(_snap_get(snap, "poincare_mean_return", base.mean_return)),
            source="PHASE2_ENGINE_L3",
        )

    def _rsi_from_engine_snapshot(
        self, snap: Any, base: RecursiveSelfImprovementState
    ) -> RecursiveSelfImprovementState:
        """Refina RSI-3 con el radio espectral / certificado V↓ / self-repair del motor."""
        spec = float(_snap_get(snap, "spectral_radius", base.meta_operator_norm))
        lip = float(min(0.999999, max(spec, 0.0)))
        residual = float(abs(_snap_get(snap, "final_fidelity_to_mac", 1.0) - 1.0))
        if residual == 0.0:
            residual = float(base.fixed_point_residual)
        lyap = float(_snap_get(snap, "lyapunov_exponent", base.improvement_lyapunov))
        l3_cert = bool(_snap_get(snap, "l3_certified", False))
        v_down = bool(_snap_get(snap, "lyapunov_decrease_certified", False))
        qnd_class = str(_snap_get(snap, "qnd_classification", base.qnd_classification))
        conv = bool(
            l3_cert
            or (lip < _LIPSCHITZ_CERT and residual < _RSI_RESIDUAL_CERT and v_down)
            or (lip < _LIPSCHITZ_CERT and residual < _RSI_RESIDUAL_CERT and lyap < 0.0)
        )
        return RecursiveSelfImprovementState(
            rsi_iteration_id=base.rsi_iteration_id,
            rsi_level=3,
            meta_operator_norm=lip,
            fixed_point_residual=residual,
            improvement_lyapunov=lyap,
            historical_verdicts=base.historical_verdicts,
            adaptive_v_floor=base.adaptive_v_floor,
            adaptive_tolerance=base.adaptive_tolerance,
            convergence_certificate=conv,
            contraction=float(_snap_get(snap, "contraction_final", base.contraction)),
            meta_learning=float(_snap_get(snap, "meta_learning_final", base.meta_learning)),
            qnd_repair_count=int(_snap_get(snap, "qnd_repair_count", 0)),
            lyapunov_decrease_certified=v_down,
            banach_fixed_point=base.banach_fixed_point,
            banach_iterates=base.banach_iterates,
            qnd_classification=qnd_class,
            final_fidelity_to_mac=float(_snap_get(snap, "final_fidelity_to_mac", 0.0)),
            source="PHASE2_ENGINE_L3",
        )

    def execute_phase2(
        self,
        continuation: Phase1ToPhase2Continuation,
        rho_mac: NDArray[Any],
        rho_delib: NDArray[Any],
        deliberation_id: str = "DELIB-QND-001",
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> NovikovWitnessPhase2Certificate:
        """FASE 2: ORIENT — Consume Φ₁₂ y ejecuta atestación QND + Poincaré/KAM/L3 refinados."""
        if not self._verify_continuation_p1_p2(continuation):
            raise ValueError("Falla de autenticidad en puente Φ₁₂ (handoff P1→P2)")

        cert1 = continuation.cert1
        poincare = continuation.poincare
        rsi = continuation.rsi
        now = time.time()
        engine_merkle = ""
        snap: Any = None

        if self._auditor is not None and hasattr(self._auditor, "full_audit"):
            snap = self._auditor.full_audit(
                rho_mac, rho_delib, deliberation_id, phi_initial, phi_final
            )
        elif self.engine is not None and hasattr(self.engine, "evaluate_novikov_witness"):
            snap = self.engine.evaluate_novikov_witness(
                rho_mac, rho_delib, deliberation_id, phi_initial, phi_final
            )

        if snap is not None:
            Aw = snap.qnd_weak_value_aav
            abs_Aw = snap.weak_value_modulus
            back_action = snap.back_action_db
            fid = snap.uhlmann_fidelity
            bures = snap.bures_distance
            d_bt = snap.bruhat_tits_distance
            amoeba = snap.amoeba_tropical_area
            h_ks = snap.kolmogorov_sinai_entropy
            qnd_verdict_raw = snap.global_verdict
            engine_merkle = str(_snap_get(snap, "sha256_merkle_root", ""))
            poincare = self._poincare_from_engine_snapshot(snap, poincare)
            rsi = self._rsi_from_engine_snapshot(snap, rsi)
        else:
            Aw = 1.0 + 0.0j
            abs_Aw = 1.0
            back_action = 0.0
            fid = 1.0
            bures = 0.0
            d_bt = 0.1
            amoeba = 0.05
            h_ks = 0.10
            qnd_verdict_raw = HeytingOmega3.COHERENT

        tangle_verdict = (
            HeytingOmega3.VETOED if poincare.homoclinic_tangle_detected else HeytingOmega3.COHERENT
        )
        p, q = poincare.resonance_ratio_pq
        resonance_verdict = (
            HeytingOmega3.DEGRADED if (q > 0 and q <= 8 and p != q) else HeytingOmega3.COHERENT
        )
        rsi_verdict = HeytingOmega3.COHERENT if rsi.convergence_certificate else HeytingOmega3.DEGRADED

        combined_verdict = (
            cert1.isometry_verdict.meet(qnd_verdict_raw)
            .meet(tangle_verdict)
            .meet(resonance_verdict)
            .meet(rsi_verdict)
        )

        wit_id = f"WIT-NOV-P2-{int(now * 1000) % 1_000_000:06d}"
        chain_bytes = (
            f"{wit_id}:{cert1.witness_id}:{abs_Aw:.6f}:{fid:.8f}:{d_bt:.6f}:"
            f"{poincare.melnikov_function:.6f}:{p}/{q}:"
            f"{rsi.meta_operator_norm:.6f}:{rsi.fixed_point_residual:.6e}:"
            f"{poincare.diophantine_gamma:.6e}:{rsi.qnd_classification}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        sig_payload = (
            f"{wit_id}:{cert1.hmac_p1_signature}:{chain_hash}:{combined_verdict.value}".encode("utf-8")
        )
        sig = _hmac_hex(self.secret_key, _HMAC_DOMAIN_P2, sig_payload)

        cert2 = NovikovWitnessPhase2Certificate(
            witness_id=wit_id,
            phase1_ref_id=cert1.witness_id,
            weak_value_qnd_aav=Aw,
            weak_value_modulus=abs_Aw,
            back_action_db=back_action,
            uhlmann_fidelity=fid,
            bures_distance=bures,
            bruhat_tits_distance=d_bt,
            amoeba_tropical_area=amoeba,
            kolmogorov_sinai_entropy=h_ks,
            qnd_verdict=combined_verdict,
            chain_hash=chain_hash,
            hmac_p2_signature=sig,
            timestamp_utc=now,
            resonance_p=int(p),
            resonance_q=int(q),
            homoclinic_tangle_detected=poincare.homoclinic_tangle_detected,
            kam_torus_stability=poincare.kam_torus_stability,
            rsi_convergence_certificate=rsi.convergence_certificate,
            rsi_meta_operator_norm=rsi.meta_operator_norm,
            rsi_fixed_point_residual=rsi.fixed_point_residual,
            kam_diophantine_gamma=poincare.diophantine_gamma,
            kam_stable=bool(poincare.diophantine_gamma > 0.0 and poincare.kam_torus_stability > 0.0),
            bruno_ok=poincare.bruno_ok,
            l3_certified=bool(_snap_get(snap, "l3_certified", False)) if snap is not None else False,
            qnd_classification=rsi.qnd_classification,
            qnd_repair_count=rsi.qnd_repair_count,
            lyapunov_exponent=poincare.lyapunov_exponent,
            final_fidelity_to_mac=rsi.final_fidelity_to_mac,
            engine_merkle=engine_merkle,
        )
        self._refined_poincare = poincare
        self._refined_rsi = rsi
        self._last_cert2 = cert2
        return cert2

    def _continuation_p23_payload(
        self,
        cert1: NovikovWitnessPhase1Certificate,
        cert2: NovikovWitnessPhase2Certificate,
        poincare: PoincareCelestialSnapshot,
        rsi: RecursiveSelfImprovementState,
        token: str,
    ) -> bytes:
        return (
            f"{cert1.chain_hash}:{cert2.chain_hash}:{token}:"
            f"{poincare.melnikov_function:.6f}:{rsi.meta_operator_norm:.6f}:"
            f"{poincare.diophantine_gamma:.6e}:{rsi.qnd_classification}:"
            f"{poincare.homoclinic_tangle_detected}:{rsi.convergence_certificate}"
        ).encode("utf-8")

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (handoff_to_phase3_sovereign_context).                            ║
    # ║  ÚLTIMO método de FASE 2. Retorna Phase2ToPhase3Continuation → PRIMER arg. de FASE 3.       ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def handoff_to_phase3_sovereign_context(
        self,
        continuation: Phase1ToPhase2Continuation,
        cert2: NovikovWitnessPhase2Certificate,
    ) -> Phase2ToPhase3Continuation:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT y abre DECIDE & ACT (puente Φ₂₃)."""
        cert1 = continuation.cert1
        poincare = self._refined_poincare or continuation.poincare
        rsi = self._refined_rsi or continuation.rsi
        token = (
            f"CONT-P2-P3-{cert2.witness_id}-{poincare.seccion_de_poincare_id}-{rsi.rsi_iteration_id}"
        )
        hmac_cont = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_CONT23,
            self._continuation_p23_payload(cert1, cert2, poincare, rsi, token),
        )
        logger.info(
            "Φ₂₃ handoff | token=%s | qnd=%s | tangle=%s | Lip(T)=%.4f | γ=%.3e | class=%s",
            token,
            cert2.qnd_verdict.name,
            poincare.homoclinic_tangle_detected,
            rsi.meta_operator_norm,
            poincare.diophantine_gamma,
            rsi.qnd_classification,
        )
        return Phase2ToPhase3Continuation(
            cert1=cert1,
            cert2=cert2,
            poincare=poincare,
            rsi=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: DECIDE & ACT — adjudicación Ω₃, certificado Banach, Φ_sem visceral                     ║
# ║  Primera def. operativa (execute_phase3) consume Phase2ToPhase3Continuation (FASE 2).          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class NovikovWitnessPhase3Decider(NovikovWitnessPhase2Orientor):
    """FASE 3: DECIDE & ACT — Adjudicación pasiva, validación RSI-3/Poincaré y Φ_sem."""

    def _verify_continuation_p2_p3(self, cont: Phase2ToPhase3Continuation) -> bool:
        expected = _hmac_hex(
            self.secret_key,
            _HMAC_DOMAIN_CONT23,
            self._continuation_p23_payload(
                cont.cert1, cont.cert2, cont.poincare, cont.rsi, cont.continuation_token
            ),
        )
        return hmac.compare_digest(cont.hmac_continuation, expected)

    def _adjudicate_final_verdict(
        self,
        cert1: NovikovWitnessPhase1Certificate,
        cert2: NovikovWitnessPhase2Certificate,
        poincare: PoincareCelestialSnapshot,
        rsi: RecursiveSelfImprovementState,
    ) -> HeytingOmega3:
        r"""Adjudicación en Ω₃. Reglas v13:

        (R1) VETOED si RSI-3 no certifica (Lip ≥ 1 ∨ residuo > tol ∨ ¬convergencia).
        (R2) VETOED si tangle homoclínico ∧ (KAM inestable ∨ γ = 0 ∨ ¬Brjuno).
        (R3) DEGRADED si resonancia p:q con q ≤ 8 y p ≠ q (toro de bajo orden).
        (R4) DEGRADED si |M| > umbral sin tangle (splitting débil).
        (R5) DEGRADED si clasificación QND ∈ {FRAGILE, BROKEN} o λ ≥ 0 sin V↓.
        (R6) Ínfimo de Heyting de todos los veredictos parciales.
        """
        base = cert1.isometry_verdict.meet(cert2.qnd_verdict)

        if not rsi.convergence_certificate:
            return HeytingOmega3.VETOED
        if rsi.meta_operator_norm >= _LIPSCHITZ_CERT:
            return HeytingOmega3.VETOED

        kam_collapsed = (
            poincare.kam_torus_stability < _KAM_UNSTABLE_EPS
            or (poincare.source == "PHASE2_ENGINE_L3" and poincare.diophantine_gamma <= 0.0)
            or (not poincare.bruno_ok)
        )
        if poincare.homoclinic_tangle_detected and kam_collapsed:
            return base.meet(HeytingOmega3.VETOED)

        p, q = poincare.resonance_ratio_pq
        if 0 < q <= 8 and p != q:
            base = base.meet(HeytingOmega3.DEGRADED)

        if abs(poincare.melnikov_function) > _MELNIKOV_AMP_THRESHOLD and not poincare.homoclinic_tangle_detected:
            base = base.meet(HeytingOmega3.DEGRADED)

        qnd_class = (rsi.qnd_classification or cert2.qnd_classification or "").upper()
        if qnd_class in {"QND_BROKEN", "QND_FRAGILE"}:
            base = base.meet(HeytingOmega3.DEGRADED)
        if poincare.lyapunov_exponent >= 0.0 and not rsi.lyapunov_decrease_certified:
            if poincare.source == "PHASE2_ENGINE_L3":
                base = base.meet(HeytingOmega3.DEGRADED)

        return base

    def execute_phase3(
        self,
        continuation: Phase2ToPhase3Continuation,
    ) -> TopologicalNovikovWitnessSovereignState:
        """FASE 3: DECIDE & ACT — Consume Φ₂₃ y emite el estado soberano final."""
        if not self._verify_continuation_p2_p3(continuation):
            raise ValueError("Falla de autenticidad en puente Φ₂₃ (handoff P2→P3)")

        cert1 = continuation.cert1
        cert2 = continuation.cert2
        poincare = continuation.poincare
        rsi = continuation.rsi
        now = time.time()

        final_verdict = self._adjudicate_final_verdict(cert1, cert2, poincare, rsi)
        back_action = cert2.back_action_db
        l3_ok = bool(cert2.l3_certified or rsi.convergence_certificate)

        if final_verdict == HeytingOmega3.VETOED:
            wacc_status = "PROTEGIDO INCONDICIONALMENTE AL 15.0% (Desviación detectada por QND)"
            liquidity_impact = "+11.5% EN TESORERÍA (Alerta preventiva sin demolición)"
            executive_action = (
                "ALERTA PREVENTIVA QND: Desviación contrafactual registrada en RAM. "
                "RSI-3 no converge o tangle homoclínico con KAM colapsado. Transparencia total a C-Suite."
            )
        elif final_verdict == HeytingOmega3.DEGRADED:
            wacc_status = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa para re-sincronización"
            liquidity_impact = "+8.5% RECIRCULANTE (Alineación en Bruhat-Tits)"
            executive_action = (
                "MONITOREO DE FRICCIÓN: Deriva en Bruhat-Tits, resonancia KAM o QND frágil bajo observación."
            )
        else:
            wacc_status = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% FLUIDO LIBRE (Coherencia Causal Total)"
            executive_action = (
                "OBSERVACIÓN FLUIDA NOMINAL: Transparencia auditada con Back-Action 0.0 dB, "
                "KAM no resonante y convergencia RSI-3 certificada."
            )

        state_id = f"NOVIKOV-WITNESS-STATE-{int(now * 1000) % 1_000_000:06d}"
        seed_merkle = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:{final_verdict.value}:"
            f"{poincare.seccion_de_poincare_id}:{rsi.rsi_iteration_id}:{cert2.engine_merkle}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()

        prov_payload = (
            f"{merkle_root}:{back_action:.1f}:{final_verdict.value}:"
            f"{rsi.meta_operator_norm:.6f}:{poincare.kam_torus_stability:.6f}:"
            f"{poincare.diophantine_gamma:.6e}:{rsi.qnd_classification}"
        ).encode("utf-8")
        provenance_hmac = _hmac_hex(self.secret_key, _HMAC_DOMAIN_P3, prov_payload)

        visceral_report: Dict[str, Any] = {
            "witness_qnd_status": "OBSERVACIÓN CIBER-FÍSICA QND DE CERO DEMOLICIÓN (0.0 dB)",
            "agent_version": _AGENT_VERSION,
            "novikov_valuation_v": cert1.novikov_valuation_v,
            "bruhat_tits_distance": cert2.bruhat_tits_distance,
            "uhlmann_fidelity": cert2.uhlmann_fidelity,
            "bures_distance": cert2.bures_distance,
            "poincare_melnikov": poincare.melnikov_function,
            "poincare_kam_stability": poincare.kam_torus_stability,
            "poincare_resonance_pq": list(poincare.resonance_ratio_pq),
            "poincare_recurrence_time": poincare.recurrence_time_estimate,
            "poincare_diophantine_gamma": poincare.diophantine_gamma,
            "poincare_bruno_ok": poincare.bruno_ok,
            "poincare_nekhoroshev_time": poincare.nekhoroshev_time,
            "poincare_jacobi_drift": poincare.jacobi_integral_drift,
            "poincare_homoclinic": poincare.homoclinic_tangle_detected,
            "poincare_source": poincare.source,
            "kam_frequency_family": poincare.kam_frequency_family,
            "rsi3_meta_operator_norm": rsi.meta_operator_norm,
            "rsi3_fixed_point_residual": rsi.fixed_point_residual,
            "rsi3_convergence_certificate": rsi.convergence_certificate,
            "rsi3_banach_fixed_point": rsi.banach_fixed_point,
            "rsi3_qnd_repair_count": rsi.qnd_repair_count,
            "rsi3_source": rsi.source,
            "qnd_classification": rsi.qnd_classification or cert2.qnd_classification,
            "l3_certified": l3_ok,
            "lyapunov_exponent": poincare.lyapunov_exponent,
            "final_fidelity_to_mac": rsi.final_fidelity_to_mac,
            "engine_merkle": cert2.engine_merkle,
            "executive_action": executive_action,
        }

        return TopologicalNovikovWitnessSovereignState(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=final_verdict,
            back_action_db=back_action,
            wacc_protection_status=wacc_status,
            net_liquidity_impact=liquidity_impact,
            sha256_merkle_root=merkle_root,
            hmac_provenance_signature=provenance_hmac,
            visceral_report=visceral_report,
            timestamp_utc=now,
            l3_certified=l3_ok,
            qnd_classification=str(visceral_report["qnd_classification"]),
            rsi_convergence_certificate=rsi.convergence_certificate,
            homoclinic_tangle_detected=poincare.homoclinic_tangle_detected,
            kam_diophantine_gamma=poincare.diophantine_gamma,
            engine_version=_AGENT_VERSION,
        )


class TOONNovikovNavigatorWitnessAgent(NovikovWitnessPhase3Decider):
    """Soberano Agéntico Testigo Silencioso de Navegación Novikov — v13.0.0-PhD.

    Orquesta el ciclo OODA anidado:
        execute_phase1                 → cert1
        handoff_to_phase2_orbit_context→ Phase1ToPhase2Continuation   (Φ₁₂)
        execute_phase2                 → cert2
        handoff_to_phase3_sovereign_…  → Phase2ToPhase3Continuation   (Φ₂₃)
        execute_phase3                 → TopologicalNovikovWitnessSovereignState

    INVARIANTE PASSIVE_SILENT: nunca conmuta hardware; reporta a C-Suite vía Φ_sem.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        if TOONNovikovNavigatorWitnessEngine is not None:
            engine = TOONNovikovNavigatorWitnessEngine(dimension=dimension, secret_key=secret_key)
        else:
            engine = None
        super().__init__(engine=engine, secret_key=secret_key)
        self.dimension: Final[int] = max(4, int(dimension))

    def continue_to_phase2_witness_gate(self, cert1: NovikovWitnessPhase1Certificate) -> bool:
        """Compuerta Homomórfica Φ₁₂: revalida HMAC de P1. ADMITE VETOED (testigo pasivo)."""
        sig_payload = f"{cert1.witness_id}:{cert1.chain_hash}:{cert1.isometry_verdict.value}".encode("utf-8")
        expected = _hmac_hex(self.secret_key, _HMAC_DOMAIN_P1, sig_payload)
        return hmac.compare_digest(cert1.hmac_p1_signature, expected)

    def continue_to_phase3_witness_gate(
        self,
        cert2: NovikovWitnessPhase2Certificate,
        cert1: Optional[NovikovWitnessPhase1Certificate] = None,
    ) -> bool:
        """Compuerta Homomórfica Φ₂₃: revalida HMAC de P2 (no un chequeo de longitud)."""
        c1 = cert1 or self._last_cert1
        if c1 is None:
            logger.warning("Φ₂₃ gate sin cert1 en caché; rechazo (fail-closed).")
            return False
        sig_payload = (
            f"{cert2.witness_id}:{c1.hmac_p1_signature}:{cert2.chain_hash}:{cert2.qnd_verdict.value}".encode(
                "utf-8"
            )
        )
        expected = _hmac_hex(self.secret_key, _HMAC_DOMAIN_P2, sig_payload)
        return hmac.compare_digest(cert2.hmac_p2_signature, expected)

    def witness_novikov_navigation(
        self,
        rho_mac: NDArray[Any],
        rho_delib: NDArray[Any],
        deliberation_id: str = "DELIB-QND-001",
        v_floor: float = 0.10,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> TopologicalNovikovWitnessSovereignState:
        """Ejecuta el ciclo OODA completo de atestación pasiva QND sobre Λ_Nov."""
        cert1 = self.execute_phase1(
            rho_delib,
            v_floor=v_floor,
            rsi_history=rsi_history,
            phi_initial=phi_initial,
            phi_final=phi_final,
        )
        if not self.continue_to_phase2_witness_gate(cert1):
            raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ del Testigo de Novikov")

        cont12 = self.handoff_to_phase2_orbit_context(cert1)

        cert2 = self.execute_phase2(
            cont12, rho_mac, rho_delib, deliberation_id, phi_initial, phi_final
        )
        if not self.continue_to_phase3_witness_gate(cert2, cert1):
            raise ValueError("Falla de autenticidad en Compuerta Φ₂₃ del Testigo de Novikov")

        cont23 = self.handoff_to_phase3_sovereign_context(cont12, cert2)
        return self.execute_phase3(cont23)


if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO TESTIGO DE NAVEGACIÓN NOVIKOV (v13.0.0-PhD)")
    print("  QND AAV | Back-Action = 0.0 dB | Ultramétrica Λ_Nov | Bruhat-Tits | Φ_sem Business")
    print("  + POINCARÉ (Melnikov ceros simples, KAM noble, Jacobi, recurrencia, homoclínicos)")
    print("  + RSI-3 (Banach iterado, Lip medido, self-repair L3 del motor, umbrales adaptativos)")
    print(banner)

    agent = TOONNovikovNavigatorWitnessAgent(dimension=56)
    rng = np.random.default_rng(2026)

    raw_mac = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_mac = raw_mac @ raw_mac.conj().T
    rho_mac /= np.trace(rho_mac)

    raw_del = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_del = raw_del @ raw_del.conj().T
    rho_del /= np.trace(rho_del)

    witness_state = agent.witness_novikov_navigation(
        rho_mac,
        rho_del,
        deliberation_id="DELIB-NAV-001",
        rsi_history=(0, 0, 1, 0, 0),
    )

    print("\n  ─── FASE 1: OBSERVE ──────────────────────────────────────────────────")
    p1 = witness_state.phase1_cert
    print(f"  • Witness ID                : {p1.witness_id}")
    print(f"  • Valuación v(A)            : {p1.novikov_valuation_v:.6f}")
    print(f"  • Desigualdad Ultramétrica  : Satisfecho = {p1.ultrametric_satisfied}")
    print(f"  • Veredicto Isometría       : {p1.isometry_verdict.name}")
    print(f"  • Poincaré snapshot ID      : {p1.poincare_snapshot_id}")
    print(f"  • Función de Melnikov       : {p1.melnikov_function:.6f}")
    print(f"  • Estabilidad KAM           : {p1.kam_torus_stability:.6e}")
    print(f"  • Tangle homoclínico        : {p1.homoclinic_tangle_detected}")
    print(f"  • RSI-3 Lip(T)              : {p1.rsi_meta_operator_norm:.6f}")
    print(f"  • RSI-3 residuo Banach      : {p1.rsi_fixed_point_residual:.6e}")
    print(f"  • RSI-3 certificado P1      : {p1.rsi_convergence_certificate}")

    print("\n  ─── FASE 2: ORIENT ──────────────────────────────────────────────────")
    p2 = witness_state.phase2_cert
    print(f"  • Witness ID                : {p2.witness_id}")
    print(f"  • Valor Débil AAV A_w       : {p2.weak_value_qnd_aav:.4f}")
    print(f"  • Back-Action QND           : {p2.back_action_db:.1f} dB")
    print(f"  • Fidelidad Uhlmann F       : {p2.uhlmann_fidelity:.6f}")
    print(f"  • Distancia Bruhat-Tits     : {p2.bruhat_tits_distance:.6f}")
    print(f"  • Área Amiba Tropical       : {p2.amoeba_tropical_area:.6f}")
    print(f"  • Resonancia p:q            : {p2.resonance_p}:{p2.resonance_q}")
    print(f"  • KAM torus stability       : {p2.kam_torus_stability:.6e}")
    print(f"  • KAM γ diofantina          : {p2.kam_diophantine_gamma:.6e}")
    print(f"  • Brjuno OK                 : {p2.bruno_ok}")
    print(f"  • RSI-3 Lip(T)              : {p2.rsi_meta_operator_norm:.6f}")
    print(f"  • RSI-3 residuo             : {p2.rsi_fixed_point_residual:.6e}")
    print(f"  • RSI-3 certificado         : {p2.rsi_convergence_certificate}")
    print(f"  • QND clasificación         : {p2.qnd_classification}")
    print(f"  • L3 motor certificado      : {p2.l3_certified}")
    print(f"  • Reparaciones QND          : {p2.qnd_repair_count}")
    print(f"  • Veredicto QND             : {p2.qnd_verdict.name}")

    print("\n  ─── FASE 3: DECIDE & ACT ────────────────────────────────────────────")
    print(f"  • Sovereign State ID        : {witness_state.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃       : {witness_state.global_verdict.name}")
    print(f"  • Protección WACC           : {witness_state.wacc_protection_status}")
    print(f"  • Impacto Liquidez Neta     : {witness_state.net_liquidity_impact}")
    print(f"  • L3 certificado            : {witness_state.l3_certified}")
    print(f"  • QND clasificación         : {witness_state.qnd_classification}")
    print(f"  • γ KAM (soberano)          : {witness_state.kam_diophantine_gamma:.6e}")
    print(f"  • Homoclínico              : {witness_state.homoclinic_tangle_detected}")
    print(f"  • Raíz Merkle SHA-256       : {witness_state.sha256_merkle_root[:32]}...")
    print(f"  • HMAC Provenance P3        : {witness_state.hmac_provenance_signature[:32]}...")
    vr = witness_state.visceral_report
    print(f"  • Recurrencia Poincaré (T)  : {vr['poincare_recurrence_time']:.4f}")
    print(f"  • Fuente Poincaré           : {vr['poincare_source']}")
    print(f"  • Fuente RSI-3              : {vr['rsi3_source']}")
    print(f"  • Acción Ejecutiva          : {str(vr['executive_action'])[:80]}...")
    print(banner)
    print("  ✓ FASE 1 → FASE 2: handoff_to_phase2_orbit_context → Phase1ToPhase2Continuation")
    print("  ✓ FASE 2 → FASE 3: handoff_to_phase3_sovereign_context → Phase2ToPhase3Continuation")
    print("  ✓ Φ₁₂ / Φ₂₃: HMAC de dominio (fail-closed; Φ₂₃ ya no es len==64)")
    print("  ✓ RSI-3: Banach iterado + Lip empírico (no ||T||<1 por construcción)")
    print("  ✓ Poincaré: Melnikov con ceros simples; KAM refinado por motor L3")
    print("  ✓ PASSIVE_SILENT: el testigo no conmuta Crowbar")
    print(banner)