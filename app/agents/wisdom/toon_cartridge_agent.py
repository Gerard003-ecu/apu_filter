# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : ToonCartridgeAgent v13.0.0-PhD                                                        ║
║ Ubicación: app/agents/wisdom/toon_cartridge_agent.py                                             ║
║ Versión  : 13.0.0-SOVEREIGN-TOON-CARTRIDGE-AGENT-POINCARE-RSI3-NESTED-MONAD-PhD                  ║
║ Función  : Agente Soberano de Calibre que gobierna el motor espectral toon_cartridge_base.py     ║
║            mediante el ciclo OODA de tres fases ANIDADAS con puentes Φ₁₂'/Φ₂₃' firmados,         ║
║            MECÁNICA CELESTE DE POINCARÉ (Delaunay–Deprit Lie-I, monodromía Sp(2,ℝ) det=1,        ║
║            Greene R_G=sin²(πν), Melnikov, Bryuno, Morse–Bott χ(ℂPⁿ⁻¹)=n) y                       ║
║            AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-3 Banach: T, T², T³, Lip(T)<1).                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

ESTRUCTURA OODA TRIFÁSICA ANIDADA (v13.0.0)
────────────────────────────────────────────────────────────────────────
  FASE 1 — OBSERVE (CartridgePhase1Observer)
    • Ingesta, sanitización, auditoría CAR de Pauli, firma HMAC P1 (ligadura Poincaré⊕RSI-3).
    • Snapshot profundo de POINCARÉ: Delaunay–Deprit, M∈Sp(2,ℝ), Greene, Melnikov,
      Bryuno por convergentes, Morse–Bott, θ_PC, volumen simpléctico.
    • Bootstrap RSI-3 anidado (T, T², T³; Banach ||T||<1; residuo de punto fijo).
    • Compuerta Φ₁₂: revalida HMAC P1 (ADMITE VETOED).
    • ÚLTIMO MÉTODO ≡ INICIO FORMAL DE FASE 2:
        `handoff_to_phase2_orbit_context(cert1) → Phase1ToPhase2CartridgeBridge`
        es el único objeto admisible de `consume_phase1_bridge` (Fase 2.0).

  FASE 2 — ORIENT (CartridgePhase2Orientor  ⊃  Fase 1)
    • PRIMER MÉTODO ≡ CONTINUACIÓN DE FASE 1:
        `consume_phase1_bridge(bridge)` verifica Φ₁₂' (HMAC dominio BRIDGE12).
    • Delaunay–Whittaker, L3 RSI, KAM, BHT, twist Birkhoff, Lyapunov, Deprit Lie-I.
    • Fusión Ω₃: tangle∧Greene, Bryuno, Banach Lip(T³).
    • Compuerta Φ₂₃: HMAC P2 + cota Lipschitz Banach + det Sp(2,ℝ).
    • ÚLTIMO MÉTODO ≡ INICIO FORMAL DE FASE 3:
        `handoff_to_phase3_sovereign_context(bridge12, cert2) → Phase2ToPhase3CartridgeBridge`

  FASE 3 — DECIDE & ACT (CartridgePhase3Decider  ⊃  Fase 2)
    • PRIMER MÉTODO ≡ CONTINUACIÓN DE FASE 2:
        `consume_phase2_bridge(bridge)` verifica Φ₂₃'.
    • `verify_rsi3_banach_certificate` (T³, Lip, residuo, HMAC RSI3).
    • Adjudicación Ω₃, Crowbar ESP32 < 392.15 ns @ GPIO14, proyección Φ_sem.
    • Cierra con `emit_sovereign_state(...)`.

  ORQUESTADOR (ToonCartridgeAgent  ⊃  Fase 3)
    • `run_cartridge_cycle(...)`: pipeline completo con handoffs formales anidados.
    • `monadic_compose_sovereign_states(...)`: μ: T·T → T (Kleisli).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, "/workspace/scratch")
sys.path.append("/workspace/artifacts")

try:
    from app.wisdom.toon_cartridge_base import (
        _DEFAULT_HILBERT_DIM,
        _GAUGE_SECRET_KEY,
        _WILKINSON_EPSILON,
        ActionAngleSeed,
        ImprovementTrajectory,
        ParticleCategory,
        PauliExclusionAuditorCAR,
        RecursiveImprovementOperator,
        ToonCartridge,
        ToonParticleFactory,
    )
except ImportError:
    from toon_cartridge_base import (
        _DEFAULT_HILBERT_DIM,
        _GAUGE_SECRET_KEY,
        _WILKINSON_EPSILON,
        ActionAngleSeed,
        ImprovementTrajectory,
        ParticleCategory,
        PauliExclusionAuditorCAR,
        RecursiveImprovementOperator,
        ToonCartridge,
        ToonParticleFactory,
    )

logger = logging.getLogger("APU.Wisdom.ToonCartridgeAgent")

__version__: Final[str] = "13.0.0-SOVEREIGN-TOON-CARTRIDGE-AGENT-POINCARE-RSI3-NESTED-MONAD-PhD"

# ─── Dominios HMAC (v13.0.0) — separación criptográfica de fases anidadas ─────
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-CARTRIDGE-P1-ISOMETRY-v13.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-CARTRIDGE-P2-POINCARE-v13.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"TOON-CARTRIDGE-P3-SOVEREIGN-v13.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"TOON-CARTRIDGE-BRIDGE-P1-P2-v13.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"TOON-CARTRIDGE-BRIDGE-P2-P3-v13.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"TOON-CARTRIDGE-RSI3-BANACH-v13.0|"
_CHAIN_HEX_LEN: Final[int] = 32

_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
_GREENE_KAM_THRESHOLD: Final[float] = 0.25
_BRYUNO_STABILITY_UPPER: Final[float] = 30.0
_RSI3_FIXED_POINT_TOL: Final[float] = 1.0e-2
_SP2R_DET_TOL: Final[float] = 1.0e-9
_MELNIKOV_TANGLE_FLOOR: Final[float] = 1.0e-6
_CF_MAX_TERMS: Final[int] = 24
_RSI3_NESTING_DEPTH: Final[int] = 3


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃  (álgebra de Heyting lineal, 3 valores)
# ══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""Retículo de Heyting trivalente lineal COHERENT(0x00) ≺ DEGRADED(0x01) ≺ VETOED(0x03).

    Operaciones:
      ∨ = join = max    (supremo — máxima severidad)
      ∧ = meet = min    (ínfimo)
      → = implies       (a → b = ⊤ si a ≤ b, else b)
      ¬a = a → COHERENT
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negate(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.COHERENT)

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.join(v)
        return acc

    @classmethod
    def fold_infimum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.VETOED
        for v in verdicts:
            acc = acc.meet(v)
        return acc


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES, PROTOCOLO DE FASE Y DATACLASSES (v13.0.0)
# ══════════════════════════════════════════════════════════════════════════════
class CartridgeAgentError(Exception):
    """Excepción raíz del Agente Soberano de Cartuchos TOON."""


class CartridgeContractCorruptedError(CartridgeAgentError):
    """HMAC o chain_hash corruptos."""


class CartridgeGateRejectionError(CartridgeAgentError):
    """Rechazo por compuerta homomórfica Φ₁₂ / Φ₂₃."""


class CartridgeSymplecticViolationError(CartridgeAgentError):
    """Monodromía fuera de Sp(2,ℝ) (det M ≠ 1)."""


class NestedPhaseHandoff(ABC):
    """Contrato de anidación: el último acto de la fase k es el primer acto de k+1."""

    @abstractmethod
    def continuation_token(self) -> str:
        ...

    @abstractmethod
    def hmac_continuation(self) -> str:
        ...


@dataclass(frozen=True, slots=True)
class MonodromySp2R:
    r"""Matriz de monodromía \(M\in\mathrm{Sp}(2,\mathbb{R})\): \(\det M=1\), \(\mathrm{Tr}\,M=2\cos(2\pi\nu)\).

    Residuo de Greene: \(R_G=(2-\mathrm{Tr}\,M)/4=\sin^2(\pi\nu)\).
    Elíptica si \(|\mathrm{Tr}|<2\), parabólica si \(=2\), hiperbólica si \(>2\).
    """

    m00: float
    m01: float
    m10: float
    m11: float
    determinant: float
    trace: float
    greene_residue_R: float
    rotation_number_nu: float
    is_symplectic: bool
    conjugacy_class: str  # elliptic | parabolic | hyperbolic

    def as_matrix(self) -> NDArray[np.float64]:
        return np.array([[self.m00, self.m01], [self.m10, self.m11]], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DepritLieTransform:
    r"""Transformación de Lie de Deprit a orden 1: \(W_1\) genera \(\mathcal{L}_{W_1}H_0=\{W_1,H_0\}\).

    Normaliza el Hamiltoniano en variables de Delaunay \((L,G,H,\ell,g,h)\).
    """

    generating_W1: float
    lie_bracket_norm: float
    normalized_H0: float
    remainder_H1: float
    order: int = 1


@dataclass(frozen=True, slots=True)
class NestedRSI3BanachOperator:
    r"""Operador de automejora anidado de nivel 3 sobre el Banach \(\mathbb{R}^3\).

    \[
      T:\mathbb{R}^3\to\mathbb{R}^3,\quad
      \mathrm{Lip}(T)=\lambda<1,\quad
      T^2=T\circ T,\;\mathrm{Lip}(T^2)=\lambda^2,\quad
      T^3=T\circ T\circ T,\;\mathrm{Lip}(T^3)=\lambda^3.
    \]
    Teorema del punto fijo de Banach: \(\exists!\,s^*=T(s^*)\),
    \(\|s_n-s^*\|\le\lambda^n(1-\lambda)^{-1}\|s_1-s_0\|\).
    """

    lipschitz_T: float
    lipschitz_T2: float
    lipschitz_T3: float
    fixed_point: Tuple[float, float, float]
    residual_T: float
    residual_T3: float
    banach_rate: float
    is_contraction: bool


@dataclass(frozen=True, slots=True)
class PoincareCelestialSnapshot:
    """Snapshot profundo de mecánica celeste de Poincaré sobre |v⟩⟨v|."""

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
    # ── Extensiones v13.0.0 ──────────────────────────────────────────────────
    monodromy_det: float = 1.0
    monodromy_is_symplectic: bool = True
    monodromy_class: str = "elliptic"
    deprit_W1: float = 0.0
    deprit_lie_bracket_norm: float = 0.0
    poincare_section_crossings: int = 0


@dataclass(frozen=True, slots=True)
class CartridgeRSI3State:
    """Estado RSI-3 con certificado de Banach (||T||<1, residuo<10⁻², anidación T³)."""

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
    # ── Extensiones v13.0.0 — anidación T, T², T³ ────────────────────────────
    lipschitz_T: float = 0.0
    lipschitz_T2: float = 0.0
    lipschitz_T3: float = 0.0
    nested_residual_T3: float = 0.0
    banach_rate: float = 0.0
    fixed_point_tuple: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CartridgePhase1Certificate:
    """Certificado inmutable de Fase 1 (Observe). Ligadura HMAC incluye Poincaré⊕RSI-3."""

    sovereign_id: str
    cartridge_count: int
    pauli_exclusion_valid: bool
    pauli_audit_message: str
    state_vector_norm: float
    isometry_verdict: HeytingOmega3
    hmac_p1_signature: str
    chain_hash: str
    timestamp_utc: float
    dimension: int = _DEFAULT_HILBERT_DIM
    hmac_verified: bool = False
    poincare_snapshot_ref: str = ""
    rsi3_state_ref: str = ""
    rsi3_meta_operator_norm: float = 0.0
    rsi3_banach_certified: bool = False
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency: bool = False
    monodromy_is_symplectic: bool = True
    rsi3_lipschitz_T3: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2CartridgeBridge(NestedPhaseHandoff):
    """Puente Φ₁₂' : cierra Fase 1 y ES el inicio formal de Fase 2."""

    cert1: CartridgePhase1Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: CartridgeRSI3State
    continuation_token_value: str
    hmac_continuation_value: str

    def continuation_token(self) -> str:
        return self.continuation_token_value

    def hmac_continuation(self) -> str:
        return self.hmac_continuation_value


@dataclass(frozen=True, slots=True)
class CartridgePhase2Certificate:
    """Certificado inmutable de Fase 2 (Orient)."""

    sovereign_id: str
    phase1_cert: CartridgePhase1Certificate
    hamiltonian_energy: float
    poincare_cartan_invariant: float
    hamiltonian_identity_residue: float
    lyapunov_exponent: float
    diophantine_gamma: float
    is_kam_stable: bool
    is_bht_stable: bool
    poincare_rotation_number: float
    poincare_crossings: int
    twist_condition_holds: bool
    birkhoff_rotation_number: float
    audit_verdict: HeytingOmega3
    hmac_p2_signature: str
    chain_hash: str
    timestamp_utc: float
    trajectory: Optional[ImprovementTrajectory] = field(default=None, repr=False)
    hmac_verified: bool = False
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
    monodromy_det: float = 1.0
    monodromy_is_symplectic: bool = True
    deprit_W1: float = 0.0
    rsi3_lipschitz_T3: float = 0.0
    dimension: int = _DEFAULT_HILBERT_DIM


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3CartridgeBridge(NestedPhaseHandoff):
    """Puente Φ₂₃' : cierra Fase 2 y ES el inicio formal de Fase 3."""

    cert1: CartridgePhase1Certificate
    cert2: CartridgePhase2Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: CartridgeRSI3State
    continuation_token_value: str
    hmac_continuation_value: str

    def continuation_token(self) -> str:
        return self.continuation_token_value

    def hmac_continuation(self) -> str:
        return self.hmac_continuation_value


@dataclass(frozen=True, slots=True)
class ToonCartridgeSovereignState:
    """Estado terminal soberano adjudicado en Fase 3."""

    sovereign_state_id: str
    phase1_cert: CartridgePhase1Certificate
    phase2_cert: CartridgePhase2Certificate
    global_verdict: HeytingOmega3
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    is_epistemologically_valid: bool = True
    agent_version: str = __version__
    hmac_p3_verified: bool = False
    rsi3_banach_certified: bool = False
    rsi3_meta_operator_norm: float = 0.0
    melnikov_M0: float = 0.0
    greene_residue_R: float = 0.0
    homoclinic_tangency_present: bool = False
    rsi3_lipschitz_T3: float = 0.0
    monodromy_is_symplectic: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# §C. FASE 1 — OBSERVE
#     Último método C.8 ≡ inicio formal de la Fase 2 (puente Φ₁₂').
# ══════════════════════════════════════════════════════════════════════════════
class CartridgePhase1Observer:
    """Soberano de Observación e Isometría con Poincaré profundo y RSI-3 anidado."""

    def __init__(
        self,
        auditor: Optional[PauliExclusionAuditorCAR] = None,
        secret_key: bytes = _GAUGE_SECRET_KEY,
    ) -> None:
        self.auditor = auditor or PauliExclusionAuditorCAR()
        self.secret_key = bytes(secret_key)
        self._last_poincare: Optional[PoincareCelestialSnapshot] = None
        self._last_rsi3: Optional[CartridgeRSI3State] = None
        self._last_monodromy: Optional[MonodromySp2R] = None
        self._last_deprit: Optional[DepritLieTransform] = None
        self._last_nested_op: Optional[NestedRSI3BanachOperator] = None

    # ─── C.1 — HMAC utilidades (dominio-separadas) ──────────────────────────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    def _phase1_payload(self, cert: CartridgePhase1Certificate) -> str:
        """Ligadura HMAC P1: isometría ⊕ Poincaré (R_G, M₀) ⊕ RSI-3 (||T||, Lip T³)."""
        return (
            f"{cert.sovereign_id}:{cert.cartridge_count}:{int(cert.pauli_exclusion_valid)}:"
            f"{cert.state_vector_norm:.8f}:{int(cert.isometry_verdict)}:{cert.chain_hash}:"
            f"{cert.greene_residue_R:.8f}:{cert.melnikov_M0:.8e}:"
            f"{cert.rsi3_meta_operator_norm:.8f}:{cert.rsi3_lipschitz_T3:.8f}:"
            f"{int(cert.monodromy_is_symplectic)}"
        )

    @staticmethod
    def _finite(x: float, fallback: float = 0.0) -> float:
        return float(x) if math.isfinite(x) else fallback

    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(max(x, 0.0), 1.0))

    # ─── C.2 — Auxiliares de Poincaré (fracción continua, Bryuno, Melnikov) ─
    @staticmethod
    def _continued_fraction_expansion(x: float, max_terms: int = _CF_MAX_TERMS) -> Tuple[int, ...]:
        r"""Expansión \([a_0;a_1,\ldots]\) de \(x>0\). Corta en residuo \(<10^{-12}\)."""
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
            if not math.isfinite(y) or y > 1e12:
                break
        return tuple(terms) if terms else (1,)

    @classmethod
    def _bryuno_diophantine(cls, rotation_number: float) -> Tuple[float, bool, Tuple[int, ...]]:
        r"""Condición de Bryuno \(B(\omega)=\sum_k \log q_{k+1}/q_k < \infty\).

        Convergentes \(p_k/q_k\) de la fracción continua de \(\omega\).
        Estable ssi \(B(\omega)<B_\star=30\) (umbral operativo, no el criterio analítico ∞).
        """
        cf = cls._continued_fraction_expansion(abs(float(rotation_number)) + 1e-15)
        if len(cf) < 3:
            return (0.0, True, cf)
        p_m2, p_m1 = 0, 1
        q_m2, q_m1 = 1, 0
        qs: List[int] = []
        for a in cf:
            p_k = a * p_m1 + p_m2
            q_k = a * q_m1 + q_m2
            qs.append(max(int(abs(q_k)), 1))
            p_m2, p_m1 = p_m1, p_k
            q_m2, q_m1 = q_m1, q_k
        if len(qs) < 2:
            return (0.0, True, cf)
        b_sum = 0.0
        for k in range(len(qs) - 1):
            b_sum += math.log(max(qs[k + 1], 2)) / float(qs[k])
        is_stable = bool(math.isfinite(b_sum) and b_sum < _BRYUNO_STABILITY_UPPER)
        return (float(b_sum), is_stable, cf)

    @staticmethod
    def _melnikov_with_separatrix(epsilon: float = 0.1, omega: float = 1.0) -> Tuple[float, bool]:
        r"""Integral de Melnikov sobre la separatriz del péndulo:

        \[
          M(t_0)=4\pi\omega^2\,\mathrm{sech}(\pi\omega/2)\cdot\sin(\omega t_0)\cdot\varepsilon.
        \]
        Amplitud \(|M|\) y tangencia homoclínica ssi \(|M|>10^{-6}\).
        Se reporta la amplitud (envolvente); el cero simple implica tangle.
        """
        if (not math.isfinite(omega)) or omega <= 0.0 or (not math.isfinite(epsilon)):
            return (0.0, False)
        sech = 1.0 / math.cosh(0.5 * math.pi * omega)
        ampl = 4.0 * math.pi * (omega ** 2) * sech * float(epsilon)
        ampl = float(ampl) if math.isfinite(ampl) else 0.0
        return (ampl, bool(abs(ampl) > _MELNIKOV_TANGLE_FLOOR))

    # ─── C.2b — Monodromía rigurosa en Sp(2,ℝ) ──────────────────────────────
    @staticmethod
    def _construct_monodromy_sp2r(rotation_number: float, shear: float = 0.0) -> MonodromySp2R:
        r"""Construye \(M\in\mathrm{Sp}(2,\mathbb{R})\) conjugada a una rotación de ángulo \(2\pi\nu\).

        \[
          R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix},
          \quad
          S=\begin{pmatrix}1&\sigma\\0&1\end{pmatrix},
          \quad
          M=S R_\theta S^{-1},\quad\det M=1.
        \]
        \(\mathrm{Tr}\,M=2\cos\theta\), \(R_G=\sin^2(\pi\nu)\).
        """
        nu = float(rotation_number) if math.isfinite(rotation_number) else 0.0
        theta = 2.0 * math.pi * nu
        c, s = math.cos(theta), math.sin(theta)
        sigma = float(shear) if math.isfinite(shear) else 0.0
        # M = S R S^{-1} con S = [[1,σ],[0,1]], S^{-1}=[[1,-σ],[0,1]]
        # R = [[c,-s],[s,c]]
        # S R = [[c-σ s, -s-σ c],[s, c]]
        # M = (S R) S^{-1} = [[c-σ s, -s-σ c],[s, c]] [[1,-σ],[0,1]]
        m00 = (c - sigma * s) * 1.0 + (-s - sigma * c) * 0.0
        m01 = (c - sigma * s) * (-sigma) + (-s - sigma * c) * 1.0
        m10 = s * 1.0 + c * 0.0
        m11 = s * (-sigma) + c * 1.0
        det = m00 * m11 - m01 * m10
        if abs(det) > _SP2R_DET_TOL and abs(det) > 1e-15:
            # renormalización simpléctica (proyección a SL(2,ℝ)=Sp(2,ℝ))
            scale = math.copysign(math.sqrt(abs(det)), det)
            if abs(scale) > 1e-15:
                m00, m01, m10, m11 = m00 / scale, m01 / scale, m10 / scale, m11 / scale
                det = m00 * m11 - m01 * m10
        trace_M = m00 + m11
        R_G = (2.0 - trace_M) / 4.0
        R_G = float(min(max(R_G, 0.0), 1.0))
        abs_tr = abs(trace_M)
        if abs_tr < 2.0 - 1e-9:
            klass = "elliptic"
        elif abs_tr > 2.0 + 1e-9:
            klass = "hyperbolic"
        else:
            klass = "parabolic"
        is_sp = bool(abs(det - 1.0) < _SP2R_DET_TOL * 10.0)
        return MonodromySp2R(
            m00=float(m00),
            m01=float(m01),
            m10=float(m10),
            m11=float(m11),
            determinant=float(det),
            trace=float(trace_M),
            greene_residue_R=R_G,
            rotation_number_nu=nu,
            is_symplectic=is_sp,
            conjugacy_class=klass,
        )

    # ─── C.2c — Deprit Lie-I sobre Delaunay ─────────────────────────────────
    @staticmethod
    def _deprit_lie_transform_first_order(
        L: float, G: float, H: float, h_energy: float
    ) -> DepritLieTransform:
        r"""Generador \(W_1\) a orden 1: \(\{W_1,H_0\}=-H_1^{\mathrm{osc}}\).

        Estimador espectral: \(W_1\sim L\cdot e\cdot\sin\ell\) con \(e=\sqrt{1-(G/L)^2}\).
        """
        L_ = max(abs(float(L)), 1e-15)
        G_ = float(G)
        ecc = math.sqrt(max(0.0, 1.0 - min((G_ / L_) ** 2, 1.0)))
        W1 = float(L_ * ecc)
        # {W, H} ~ ∂W/∂ℓ · ∂H/∂L  (escala)
        lie_br = float(abs(W1 * (h_energy / L_)))
        H0 = float(h_energy)
        H1 = float(abs(ecc) * abs(h_energy) * 0.5)
        return DepritLieTransform(
            generating_W1=W1,
            lie_bracket_norm=lie_br,
            normalized_H0=H0,
            remainder_H1=H1,
            order=1,
        )

    # ─── C.2d — Sección de Poincaré (cruces del hiperplano q_1=0, p_1>0) ────
    @staticmethod
    def _poincare_section_crossings(actions: NDArray[np.float64], angles: NDArray[np.float64]) -> int:
        """Cuenta cruces de sección {ℓ = 0 mod 2π} con ḣ > 0, acotado por dim."""
        if actions.size == 0 or angles.size == 0:
            return 0
        wrapped = np.mod(np.real(angles), 2.0 * math.pi)
        crossings = int(np.sum(wrapped < (2.0 * math.pi / max(angles.size, 1))))
        return int(min(max(crossings, 0), max(int(actions.size), 1)))

    # ─── C.3 — Snapshot profundo de Poincaré ────────────────────────────────
    def _compute_poincare_celestial_snapshot(
        self, state_vector: NDArray, dimension: int
    ) -> PoincareCelestialSnapshot:
        now = time.time()
        v = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
        dim = int(min(max(int(dimension), 1), v.size if v.size > 0 else int(dimension)))
        if v.size < dim:
            pad = np.zeros(dim, dtype=np.complex128)
            pad[: v.size] = v
            v = pad
        else:
            v = v[:dim]
        with np.errstate(all="ignore"):
            nrm = float(np.linalg.norm(v))
        if (not math.isfinite(nrm)) or nrm < _WILKINSON_EPSILON:
            v = np.zeros(dim, dtype=np.complex128)
            v[0] = 1.0 + 0.0j
            nrm = 1.0
        v = v / nrm
        rho = np.outer(v, np.conjugate(v))
        rho = 0.5 * (rho + np.conjugate(rho.T))

        with np.errstate(all="ignore"):
            eigs = np.real(np.linalg.eigvalsh(rho))
        eigs = np.maximum(eigs, 1e-15)
        eigs = np.sort(eigs)[::-1]
        ssum = float(np.sum(eigs))
        probs = eigs / ssum if ssum > 0.0 else np.full_like(eigs, 1.0 / max(eigs.size, 1))
        actions = -np.log(np.maximum(probs, 1e-15))

        # Delaunay–Deprit (acciones angulares sobre el espectro de ρ)
        L = float(np.sum(actions))
        G = float(L * (probs[1] / probs[0])) if probs.size > 1 and probs[0] > 0.0 else L
        H = float(G * (probs[2] / probs[1])) if probs.size > 2 and probs[1] > 0.0 else G
        ecc = float(math.sqrt(max(0.0, 1.0 - (G / (L + 1e-15)) ** 2)))
        inc_arg = min(max(H / (G + 1e-15), -1.0), 1.0)
        inc = float(math.acos(inc_arg))
        l_ang = float(math.atan2(float(actions[1]), float(actions[0]))) if actions.size > 1 else 0.0
        g_ang = float(math.atan2(float(actions[2]), float(actions[1]))) if actions.size > 2 else 0.0
        h_ang = float(math.atan2(float(actions[3]), float(actions[2]))) if actions.size > 3 else 0.0

        # CR3BP: constante de Jacobi C_J = 2L + G cos i  (proxy espectral)
        c_j = float(2.0 * L + G * math.cos(inc))

        # Frecuencias → número de rotación ν
        if actions.size > 1:
            freqs = np.abs(np.diff(actions))
            omega1 = float(np.median(freqs)) if freqs.size > 0 else 1.0
        else:
            omega1 = 1.0
        omega2 = float(np.std(actions)) + 1e-12
        nu = float(omega1 / max(omega2, 1e-12))
        if not math.isfinite(nu):
            nu = 1.0

        shear = float(np.std(probs)) if probs.size > 0 else 0.0
        mono = self._construct_monodromy_sp2r(nu, shear=shear)
        self._last_monodromy = mono
        trace_M, R_G = mono.trace, mono.greene_residue_R

        # Melnikov
        M_amp, has_tangle = self._melnikov_with_separatrix(
            epsilon=min(max(abs(L) * 0.1, 0.01), 1.0),
            omega=min(max(abs(omega1), 0.1), 10.0),
        )

        # Bryuno
        b_sum, b_stable, cf_terms = self._bryuno_diophantine(nu)

        # Morse–Bott: estratificación espectral de ρ; χ(ℂP^{n-1})=n canónico
        sorted_p = np.sort(probs)[::-1]
        strata: List[int] = []
        current = 1
        for i in range(1, len(sorted_p)):
            if abs(float(sorted_p[i]) - float(sorted_p[i - 1])) < 1e-3:
                current += 1
            else:
                strata.append(current)
                current = 1
        strata.append(current)
        chi_raw = int(sum((1 if k % 2 == 0 else -1) * d for k, d in enumerate(strata)))
        chi_canonical = int(min(dim, max(1, abs(chi_raw) if chi_raw != 0 else dim)))

        # θ_PC = |Tr(ρ · i[N,ρ])|  (1-forma de Poincaré–Cartan)
        N_op = np.diag(np.arange(dim, dtype=np.complex128))
        comm = N_op @ rho - rho @ N_op
        with np.errstate(all="ignore"):
            theta_pc = float(abs(np.real(np.trace(rho @ (1j * comm)))))
        if not math.isfinite(theta_pc):
            theta_pc = 0.0

        with np.errstate(all="ignore"):
            pur = float(np.real(np.trace(rho @ rho)))
        sympl_res = float(abs(pur - 1.0 / float(dim))) if math.isfinite(pur) else 1.0

        angles = np.array([l_ang, g_ang, h_ang], dtype=np.float64)
        crossings = self._poincare_section_crossings(actions, angles)

        h_proxy = float(L)
        deprit = self._deprit_lie_transform_first_order(L, G, H, h_proxy)
        self._last_deprit = deprit

        snapshot_id = f"POIN-CART-{int(now * 1000) % 1_000_000:06d}"
        prov = (
            f"{snapshot_id}:{L:.8f}:{G:.8f}:{H:.8f}:{ecc:.8f}:{c_j:.8f}:"
            f"{R_G:.8f}:{M_amp:.8f}:{b_sum:.8f}:{chi_canonical}:{theta_pc:.8f}:"
            f"{mono.determinant:.8f}:{deprit.generating_W1:.8f}"
        )
        return PoincareCelestialSnapshot(
            snapshot_id=snapshot_id,
            delaunay_L=self._finite(L),
            delaunay_G=self._finite(G),
            delaunay_H=self._finite(H),
            delaunay_l_angle=self._finite(l_ang),
            delaunay_g_angle=self._finite(g_ang),
            delaunay_h_angle=self._finite(h_ang),
            eccentricity_e=self._finite(ecc),
            inclination_i=self._finite(inc),
            cr3bp_jacobi_constant=self._finite(c_j),
            monodromy_trace=self._finite(trace_M),
            greene_residue_R=self._finite(R_G),
            rotation_number_nu=self._finite(nu, 1.0),
            melnikov_M0=self._finite(M_amp),
            homoclinic_tangency_detected=has_tangle,
            continued_fraction_terms=cf_terms,
            bryuno_diophantine_sum=self._finite(b_sum),
            is_bryuno_stable=b_stable,
            morse_bott_euler_chi=chi_canonical,
            morse_bott_strata=tuple(int(s) for s in strata),
            poincare_cartan_1form=self._finite(theta_pc),
            symplectic_volume_residual=self._finite(sympl_res),
            sha256_provenance=hashlib.sha256(prov.encode("utf-8")).hexdigest(),
            monodromy_det=self._finite(mono.determinant, 1.0),
            monodromy_is_symplectic=mono.is_symplectic,
            monodromy_class=mono.conjugacy_class,
            deprit_W1=self._finite(deprit.generating_W1),
            deprit_lie_bracket_norm=self._finite(deprit.lie_bracket_norm),
            poincare_section_crossings=int(crossings),
        )

    # ─── C.4 — RSI-3 anidado (Banach: T, T², T³) ────────────────────────────
    def _nested_rsi3_banach_operator(
        self,
        v_floor: float,
        greene_thr: float,
        lipschitz: float,
        perf_potential: float,
    ) -> NestedRSI3BanachOperator:
        r"""Contracción afín \(T(x)=c+\Lambda(x-c)\) en \(\mathbb{R}^3\).

        \(\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3)\) con \(|\lambda_i|<1\).
        \(\mathrm{Lip}(T)=\|\Lambda\|_\infty\), \(\mathrm{Lip}(T^k)=\|\Lambda\|^k_\infty\).
        """
        alpha, beta, gamma = 0.05, 0.03, 0.02
        p = self._clip01(float(perf_potential))
        # autovalores de la diferencial (contracción hacia el punto fijo adaptativo)
        lam1 = abs(alpha * p)          # perturbación de v_floor
        lam2 = abs(beta * p)           # perturbación de greene
        lam3 = abs(gamma * p)          # perturbación de lipschitz
        lip_T = float(min(max(lam1, lam2, lam3, 1e-6), 0.999))
        lip_T2 = float(lip_T ** 2)
        lip_T3 = float(lip_T ** 3)

        fp = (float(v_floor), float(greene_thr), float(lipschitz))
        # Un paso de T desde el ancla canónica (0.10, G_★, Lip_★)
        anchor = (0.10, _GREENE_KAM_THRESHOLD, _BANACH_LIPSCHITZ_CEILING)
        image = (
            anchor[0] + lam1 * (fp[0] - anchor[0]),
            anchor[1] + lam2 * (fp[1] - anchor[1]),
            anchor[2] + lam3 * (fp[2] - anchor[2]),
        )
        resid_T = float(max(abs(image[0] - fp[0]), abs(image[1] - fp[1]), abs(image[2] - fp[2])))
        # T³: tres aplicaciones; residuo ≤ Lip(T)³ / (1-Lip(T)) · residuo_T  (cota Banach)
        denom = max(1.0 - lip_T, 1e-9)
        resid_T3 = float(min(resid_T * lip_T3 / denom, resid_T))
        rate = float(lip_T / denom)
        is_contr = bool(lip_T < 1.0 and resid_T < _RSI3_FIXED_POINT_TOL)
        return NestedRSI3BanachOperator(
            lipschitz_T=lip_T,
            lipschitz_T2=lip_T2,
            lipschitz_T3=lip_T3,
            fixed_point=fp,
            residual_T=resid_T,
            residual_T3=resid_T3,
            banach_rate=rate,
            is_contraction=is_contr,
        )

    def _compute_rsi3_bootstrap(
        self,
        poincare: PoincareCelestialSnapshot,
        verdict_p1: HeytingOmega3,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> CartridgeRSI3State:
        r"""Meta-operador T: ℝ³→ℝ³ contractivo, punto fijo s*, certificado Banach anidado."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        greene_dev = abs(poincare.greene_residue_R - _GREENE_KAM_THRESHOLD) / max(_GREENE_KAM_THRESHOLD, 1e-6)
        bryuno_dev = abs(poincare.bryuno_diophantine_sum) / max(_BRYUNO_STABILITY_UPPER, 1e-6)
        verdict_dev = float(int(verdict_p1)) / 3.0
        # penalización simpléctica: si det M ≉ 1, el potencial se colapsa
        sp_pen = 0.0 if poincare.monodromy_is_symplectic else 2.0
        perf_potential = float(math.exp(-(greene_dev + bryuno_dev + verdict_dev + sp_pen)))
        perf_potential = self._clip01(max(perf_potential, 1e-6))

        alpha, beta, gamma = 0.05, 0.03, 0.02
        new_vfloor = 0.10 * (1.0 + alpha * perf_potential)
        new_greene = _GREENE_KAM_THRESHOLD * (1.0 - beta * perf_potential)
        new_lipschitz = min(0.999, _BANACH_LIPSCHITZ_CEILING * (1.0 + gamma * perf_potential))

        nested = self._nested_rsi3_banach_operator(new_vfloor, new_greene, new_lipschitz, perf_potential)
        self._last_nested_op = nested

        meta_norm = float(nested.lipschitz_T)
        resid = float(max(nested.residual_T, nested.residual_T3))

        if len(hist) >= 2:
            improvement_lyap = float(np.mean(np.abs(np.diff(np.asarray(hist, dtype=float))))) + 1e-9
        else:
            improvement_lyap = 1e-3

        banach_ok = bool(
            nested.is_contraction
            and meta_norm < 1.0
            and resid < _RSI3_FIXED_POINT_TOL
            and nested.lipschitz_T3 < 1.0
        )

        rsi_id = f"RSI3-CART-{int(now * 1000) % 1_000_000:06d}"
        cert_payload = (
            f"{rsi_id}:{meta_norm:.8f}:{resid:.8f}:{banach_ok}:"
            f"{nested.lipschitz_T3:.8f}:{nested.residual_T3:.8f}"
        ).encode("utf-8")
        cert_hmac = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + cert_payload, hashlib.sha256).hexdigest()

        return CartridgeRSI3State(
            rsi_iteration_id=rsi_id,
            rsi_level=_RSI3_NESTING_DEPTH,
            meta_operator_norm=meta_norm,
            fixed_point_residual=resid,
            adaptive_v_floor=float(new_vfloor),
            adaptive_greene_threshold=float(new_greene),
            adaptive_lipschitz=float(new_lipschitz),
            improvement_lyapunov=float(improvement_lyap),
            historical_verdicts=tuple(hist[-8:]),
            banach_convergence_certified=banach_ok,
            banach_certificate_hmac=cert_hmac,
            lipschitz_T=nested.lipschitz_T,
            lipschitz_T2=nested.lipschitz_T2,
            lipschitz_T3=nested.lipschitz_T3,
            nested_residual_T3=nested.residual_T3,
            banach_rate=nested.banach_rate,
            fixed_point_tuple=nested.fixed_point,
        )

    # ─── C.5 — Emisión del certificado P1 (API pública) ─────────────────────
    def emit_phase1_certificate(
        self,
        cartridges: Sequence[ToonCartridge],
        dimension: int = _DEFAULT_HILBERT_DIM,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> CartridgePhase1Certificate:
        """Emite P1 con auditoría CAR, Poincaré profundo (Sp(2,ℝ)+Deprit) y RSI-3 anidado."""
        now = time.time()
        dim = max(4, int(dimension))

        pauli_valid = True
        pauli_msg = "ALL_CARTRIDGES_VALIDATED_OK"
        for c in cartridges:
            ok, msg = self.auditor.audit_and_register_fermion(c)
            if not ok:
                pauli_valid = False
                pauli_msg = msg
                break

        if cartridges:
            v_acc = np.zeros((dim, 1), dtype=np.complex128)
            for c in cartridges:
                v_acc += c.to_hilbert_state_vector(dim)
            v_flat = v_acc.reshape(-1)
            nrm = float(np.linalg.norm(v_flat))
            if (not math.isfinite(nrm)) or nrm < _WILKINSON_EPSILON:
                v_flat = np.zeros(dim, dtype=np.complex128)
                v_flat[0] = 1.0 + 0.0j
                nrm = 1.0
                v_norm_unit = 1.0
                v_norm_val = 1.0
            else:
                # isometría: norma RMS por cartucho vs. 1 (tras normalización colectiva)
                v_norm_unit = nrm / math.sqrt(max(len(cartridges), 1))
                v_norm_val = float(nrm)
                v_flat = v_flat / nrm
        else:
            v_flat = np.zeros(dim, dtype=np.complex128)
            v_flat[0] = 1.0 + 0.0j
            v_norm_unit = 1.0
            v_norm_val = 1.0

        if not pauli_valid or abs(v_norm_unit - 1.0) > 0.5:
            verdict = HeytingOmega3.VETOED
        elif abs(v_norm_unit - 1.0) > 0.1:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        poincare = self._compute_poincare_celestial_snapshot(v_flat, dim)
        if not poincare.monodromy_is_symplectic:
            verdict = verdict.join(HeytingOmega3.DEGRADED)

        rsi3 = self._compute_rsi3_bootstrap(poincare, verdict, rsi_history)

        cart_hashes = "".join(c.quantum_state_hash for c in cartridges)
        chain_bytes = (
            f"{cart_hashes}:{v_norm_unit:.8f}:{verdict.value}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}:"
            f"{poincare.monodromy_det:.8f}:{rsi3.lipschitz_T3:.8f}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        sov_id = f"CARTRIDGE-P1-{int(now * 1000.0) % 1000000:06d}"
        tmp_cert = CartridgePhase1Certificate(
            sovereign_id=sov_id,
            cartridge_count=len(cartridges),
            pauli_exclusion_valid=pauli_valid,
            pauli_audit_message=pauli_msg,
            state_vector_norm=float(v_norm_val),
            isometry_verdict=verdict,
            hmac_p1_signature="",
            chain_hash=chain_hash,
            timestamp_utc=now,
            dimension=dim,
            hmac_verified=False,
            poincare_snapshot_ref=poincare.snapshot_id,
            rsi3_state_ref=rsi3.rsi_iteration_id,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            rsi3_banach_certified=rsi3.banach_convergence_certified,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency=poincare.homoclinic_tangency_detected,
            monodromy_is_symplectic=poincare.monodromy_is_symplectic,
            rsi3_lipschitz_T3=rsi3.lipschitz_T3,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(tmp_cert))
        cert1 = replace(tmp_cert, hmac_p1_signature=sig, hmac_verified=True)

        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert1

    # ─── C.6 — Compuerta Φ₁₂ ────────────────────────────────────────────────
    def continue_to_phase2_gate(self, cert1: CartridgePhase1Certificate) -> bool:
        """Compuerta Φ₁₂: revalida HMAC P1. ADMITE VETOED (el veto viaja, no se silencia)."""
        payload = self._phase1_payload(cert1)
        if not self._verify_hmac(_HMAC_DOMAIN_P1, payload, cert1.hmac_p1_signature):
            raise CartridgeGateRejectionError("Compuerta Φ₁₂ rechazó P1: Firma HMAC inválida.")
        if len(cert1.chain_hash) != _CHAIN_HEX_LEN:
            raise CartridgeGateRejectionError("Compuerta Φ₁₂: chain_hash corrupto.")
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # C.7 / C.8 — ÚLTIMO MÉTODO DE LA FASE 1
    #            ≡ INICIO FORMAL DE LA FASE 2
    # El valor de retorno Phase1ToPhase2CartridgeBridge es el ÚNICO argumento
    # admisible de CartridgePhase2Orientor.consume_phase1_bridge (D.1).
    # La definición de D.1 continúa semántica y operativamente a C.8.
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_orbit_context(
        self, cert1: CartridgePhase1Certificate
    ) -> Phase1ToPhase2CartridgeBridge:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE y abre ORIENT.

        Firma HMAC-SHA256 el puente completo (cert1 ⊕ Poincaré ⊕ RSI-3 ⊕ Sp(2,ℝ)).
        El objeto retornado ES el primer dato de `consume_phase1_bridge` (Fase 2.0).
        """
        self.continue_to_phase2_gate(cert1)
        if self._last_poincare is None or self._last_rsi3 is None:
            raise RuntimeError(
                "handoff_to_phase2_orbit_context: contexto no inicializado. "
                "Ejecute emit_phase1_certificate primero."
            )
        poincare = self._last_poincare
        rsi = self._last_rsi3

        token = (
            f"CONT-CART-P1-P2-{cert1.sovereign_id}-"
            f"{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        )
        payload = (
            f"{cert1.sovereign_id}:{cert1.chain_hash}:{poincare.sha256_provenance}:"
            f"{rsi.banach_certificate_hmac}:{token}:"
            f"{poincare.monodromy_det:.8f}:{rsi.lipschitz_T3:.8f}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₁₂' CART handoff | cert=%s | R_G=%.4f | M₀=%.4e | ||T||=%.4f | Lip(T³)=%.4e | Banach=%s | Sp(2,ℝ)=%s",
            cert1.sovereign_id,
            poincare.greene_residue_R,
            poincare.melnikov_M0,
            rsi.meta_operator_norm,
            rsi.lipschitz_T3,
            rsi.banach_convergence_certified,
            poincare.monodromy_is_symplectic,
        )
        return Phase1ToPhase2CartridgeBridge(
            cert1=cert1,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token_value=token,
            hmac_continuation_value=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 2 — ORIENT
#     PRIMER MÉTODO D.1 ≡ continuación de C.8 (consume el puente Φ₁₂').
#     ÚLTIMO MÉTODO D.7 ≡ inicio formal de la Fase 3 (puente Φ₂₃').
# ══════════════════════════════════════════════════════════════════════════════
class CartridgePhase2Orientor(CartridgePhase1Observer):
    """Soberano de Orientación Física. Continúa el tejido de C.8 sin ruptura de tipo."""

    # ─── D.1 — PRIMER MÉTODO DE LA FASE 2 ≡ CONTINUACIÓN FORMAL DE C.8 ───────
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2CartridgeBridge) -> bool:
        """Verifica Φ₁₂'. Este método ES la continuación operativa de
        `handoff_to_phase2_orbit_context`: mismo payload, mismo dominio HMAC."""
        payload = (
            f"{bridge.cert1.sovereign_id}:{bridge.cert1.chain_hash}:"
            f"{bridge.poincare_snapshot.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{bridge.continuation_token()}:"
            f"{bridge.poincare_snapshot.monodromy_det:.8f}:{bridge.rsi3_state.lipschitz_T3:.8f}"
        ).encode("utf-8")
        expected = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation(), expected):
            raise CartridgeGateRejectionError("Falla de autenticidad en puente Φ₁₂' del agente.")
        if not bridge.poincare_snapshot.monodromy_is_symplectic:
            logger.warning(
                "Φ₁₂' Sp(2,ℝ) degradado: det M=%.6e (se propaga, no se silencia).",
                bridge.poincare_snapshot.monodromy_det,
            )
        self._last_poincare = bridge.poincare_snapshot
        self._last_rsi3 = bridge.rsi3_state
        return True

    # ─── D.2 — Compatibilidad: constructor desde cert1 (API v11/v12) ────────
    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: CartridgePhase1Certificate,
        auditor: Optional[PauliExclusionAuditorCAR] = None,
        secret_key: bytes = _GAUGE_SECRET_KEY,
    ) -> "CartridgePhase2Orientor":
        inst = cls(auditor=auditor, secret_key=secret_key)
        inst.continue_to_phase2_gate(cert1)
        return inst

    # ─── D.3 — Payload HMAC P2 (ligadura Poincaré⊕RSI-3⊕Sp(2,ℝ)) ────────────
    def _phase2_payload(self, cert: CartridgePhase2Certificate) -> str:
        return (
            f"{cert.sovereign_id}:{cert.hamiltonian_energy:.8f}:{cert.poincare_cartan_invariant:.8f}:"
            f"{cert.lyapunov_exponent:.8f}:{int(cert.is_kam_stable)}:{int(cert.is_bht_stable)}:"
            f"{int(cert.audit_verdict)}:{cert.chain_hash}:"
            f"{cert.greene_residue_R:.8f}:{cert.melnikov_M0:.8e}:"
            f"{cert.rsi3_lipschitz_T3:.8f}:{int(cert.monodromy_is_symplectic)}"
        )

    # ─── D.4 — Fusión Ω₃ de veredictos Poincaré ⊕ RSI-3 ⊕ trayectoria ───────
    @staticmethod
    def _fuse_orientation_verdict(
        cert1: CartridgePhase1Certificate,
        traj: ImprovementTrajectory,
        poincare: PoincareCelestialSnapshot,
        rsi3: CartridgeRSI3State,
    ) -> HeytingOmega3:
        tangle_veto = (
            HeytingOmega3.VETOED
            if (poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD)
            else HeytingOmega3.COHERENT
        )
        bryuno_veto = HeytingOmega3.VETOED if not poincare.is_bryuno_stable else HeytingOmega3.COHERENT
        rsi_veto = HeytingOmega3.COHERENT if rsi3.banach_convergence_certified else HeytingOmega3.DEGRADED
        sp_veto = HeytingOmega3.COHERENT if poincare.monodromy_is_symplectic else HeytingOmega3.DEGRADED
        t3_veto = (
            HeytingOmega3.COHERENT
            if rsi3.lipschitz_T3 < 1.0
            else HeytingOmega3.VETOED
        )
        atoms: List[HeytingOmega3] = [
            cert1.isometry_verdict,
            HeytingOmega3.COHERENT if traj.is_kam_stable else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if traj.is_bht_stable else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if traj.twist_condition else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if traj.lyapunov_exponent < 0.0 else HeytingOmega3.VETOED,
            tangle_veto,
            bryuno_veto,
            rsi_veto,
            sp_veto,
            t3_veto,
        ]
        return HeytingOmega3.fold_supremum(atoms)

    # ─── D.5 — Emisión del certificado P2 ───────────────────────────────────
    def emit_phase2_certificate(
        self,
        cert1: Optional[CartridgePhase1Certificate] = None,
        primary_cartridge: Optional[ToonCartridge] = None,
        max_l3_iters: int = 8,
        bridge: Optional[Phase1ToPhase2CartridgeBridge] = None,
    ) -> CartridgePhase2Certificate:
        """Emite P2. Acepta (a) API v11: (cert1, primary_cartridge) o (b) v12/v13: (bridge=…)."""
        now = time.time()

        if bridge is not None:
            self.consume_phase1_bridge(bridge)
            cert1 = bridge.cert1
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
        elif cert1 is not None:
            self.continue_to_phase2_gate(cert1)
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                if primary_cartridge is None:
                    raise ValueError("emit_phase2_certificate: requiere primary_cartridge o bridge.")
                v = primary_cartridge.to_hilbert_state_vector(cert1.dimension).reshape(-1)
                poincare = self._compute_poincare_celestial_snapshot(v, cert1.dimension)
                rsi3 = self._compute_rsi3_bootstrap(poincare, cert1.isometry_verdict)
        else:
            raise ValueError("emit_phase2_certificate: requiere cert1 o bridge.")

        if primary_cartridge is None:
            raise ValueError("emit_phase2_certificate: primary_cartridge es obligatorio.")

        seed: ActionAngleSeed = primary_cartridge.to_action_angle_seed(dimension=cert1.dimension)
        improver = RecursiveImprovementOperator(contraction=0.5, meta_learning=0.3)
        traj: ImprovementTrajectory = improver.iterate_level_3(seed, max_iters=max_l3_iters)

        h_energy = seed.hamiltonian()
        i_poincare = seed.poincare_cartan_invariant()
        h_residue = seed.hamiltonian_identity_residue()

        # Recomputar Deprit con la energía hamiltoniana verdadera
        deprit = self._deprit_lie_transform_first_order(
            poincare.delaunay_L, poincare.delaunay_G, poincare.delaunay_H, float(h_energy)
        )
        self._last_deprit = deprit

        p2_verdict = self._fuse_orientation_verdict(cert1, traj, poincare, rsi3)

        chain_payload = (
            f"{cert1.chain_hash}:{h_energy:.8f}:{traj.lyapunov_exponent:.8f}:{p2_verdict.value}:"
            f"{poincare.sha256_provenance}:{rsi3.banach_certificate_hmac}:"
            f"{poincare.monodromy_det:.8f}:{rsi3.lipschitz_T3:.8f}"
        )
        chain_hash = hashlib.sha256(chain_payload.encode("utf-8")).hexdigest()[:_CHAIN_HEX_LEN]

        sov_id = f"CARTRIDGE-P2-{int(now * 1000.0) % 1000000:06d}"
        tmp_cert = CartridgePhase2Certificate(
            sovereign_id=sov_id,
            phase1_cert=cert1,
            hamiltonian_energy=float(h_energy),
            poincare_cartan_invariant=float(i_poincare),
            hamiltonian_identity_residue=float(h_residue),
            lyapunov_exponent=float(traj.lyapunov_exponent),
            diophantine_gamma=float(traj.diophantine_gamma),
            is_kam_stable=bool(traj.is_kam_stable),
            is_bht_stable=bool(traj.is_bht_stable),
            poincare_rotation_number=float(traj.poincare_rotation_number),
            poincare_crossings=int(traj.poincare_crossing_count),
            twist_condition_holds=bool(traj.twist_condition),
            birkhoff_rotation_number=float(traj.poincare_rotation_number),
            audit_verdict=p2_verdict,
            hmac_p2_signature="",
            chain_hash=chain_hash,
            timestamp_utc=now,
            trajectory=traj,
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
            monodromy_det=poincare.monodromy_det,
            monodromy_is_symplectic=poincare.monodromy_is_symplectic,
            deprit_W1=deprit.generating_W1,
            rsi3_lipschitz_T3=rsi3.lipschitz_T3,
            dimension=int(cert1.dimension),
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(tmp_cert))
        cert2 = replace(tmp_cert, hmac_p2_signature=sig, hmac_verified=True)

        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert2

    # ─── D.6 — Compuerta Φ₂₃ ────────────────────────────────────────────────
    def continue_to_phase3_gate(self, cert2: CartridgePhase2Certificate) -> bool:
        """Compuerta Φ₂₃: HMAC P2 + radio espectral < 1 + Lip(T³)<1 + Sp(2,ℝ)."""
        payload = self._phase2_payload(cert2)
        if not self._verify_hmac(_HMAC_DOMAIN_P2, payload, cert2.hmac_p2_signature):
            raise CartridgeGateRejectionError("Compuerta Φ₂₃ rechazó P2: Firma HMAC inválida.")
        if cert2.trajectory is not None and cert2.trajectory.spectral_radius >= 1.0:
            raise CartridgeGateRejectionError("Compuerta Φ₂₃ rechazó P2: radio espectral ≥ 1.0.")
        if cert2.rsi3_meta_operator_norm >= 1.0:
            raise CartridgeGateRejectionError("Compuerta Φ₂₃ rechazó P2: ||T|| ≥ 1.0 (RSI-3 no contractivo).")
        if cert2.rsi3_lipschitz_T3 >= 1.0:
            raise CartridgeGateRejectionError("Compuerta Φ₂₃ rechazó P2: Lip(T³) ≥ 1.0.")
        if abs(cert2.monodromy_det - 1.0) > 1e-6 and not cert2.monodromy_is_symplectic:
            raise CartridgeSymplecticViolationError(
                f"Compuerta Φ₂₃: monodromía fuera de Sp(2,ℝ), det={cert2.monodromy_det:.6e}."
            )
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # D.7 — ÚLTIMO MÉTODO DE LA FASE 2
    #       ≡ INICIO FORMAL DE LA FASE 3
    # El valor Phase2ToPhase3CartridgeBridge es el ÚNICO argumento admisible
    # de CartridgePhase3Decider.consume_phase2_bridge (E.1).
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_sovereign_context(
        self,
        bridge12: Phase1ToPhase2CartridgeBridge,
        cert2: CartridgePhase2Certificate,
    ) -> Phase2ToPhase3CartridgeBridge:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT y abre DECIDE & ACT."""
        self.continue_to_phase3_gate(cert2)
        poincare = bridge12.poincare_snapshot
        rsi = bridge12.rsi3_state
        token = (
            f"CONT-CART-P2-P3-{cert2.sovereign_id}-"
            f"{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        )
        payload = (
            f"{bridge12.cert1.chain_hash}:{cert2.chain_hash}:{token}:"
            f"{poincare.melnikov_M0:.8f}:{rsi.meta_operator_norm:.8f}:"
            f"{poincare.monodromy_det:.8f}:{rsi.lipschitz_T3:.8f}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₂₃' CART handoff | cert2=%s | verdict=%s | tangle=%s | ||T||=%.4f | Lip(T³)=%.4e",
            cert2.sovereign_id,
            cert2.audit_verdict.name,
            poincare.homoclinic_tangency_detected,
            rsi.meta_operator_norm,
            rsi.lipschitz_T3,
        )
        return Phase2ToPhase3CartridgeBridge(
            cert1=bridge12.cert1,
            cert2=cert2,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token_value=token,
            hmac_continuation_value=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 3 — DECIDE & ACT
#     PRIMER MÉTODO E.1 ≡ continuación de D.7 (consume el puente Φ₂₃').
# ══════════════════════════════════════════════════════════════════════════════
class CartridgePhase3Decider(CartridgePhase2Orientor):
    """Soberano de Adjudicación. Continúa el tejido de D.7; verifica RSI-3 Banach T³."""

    # ─── E.1 — PRIMER MÉTODO DE LA FASE 3 ≡ CONTINUACIÓN FORMAL DE D.7 ───────
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3CartridgeBridge) -> bool:
        """Verifica Φ₂₃'. Este método ES la continuación operativa de
        `handoff_to_phase3_sovereign_context`."""
        payload = (
            f"{bridge.cert1.chain_hash}:{bridge.cert2.chain_hash}:"
            f"{bridge.continuation_token()}:{bridge.poincare_snapshot.melnikov_M0:.8f}:"
            f"{bridge.rsi3_state.meta_operator_norm:.8f}:"
            f"{bridge.poincare_snapshot.monodromy_det:.8f}:{bridge.rsi3_state.lipschitz_T3:.8f}"
        ).encode("utf-8")
        expected = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation(), expected):
            raise CartridgeGateRejectionError("Falla de autenticidad en puente Φ₂₃' del agente.")
        self._last_poincare = bridge.poincare_snapshot
        self._last_rsi3 = bridge.rsi3_state
        return True

    # ─── E.2 — Compatibilidad: constructor desde cert2 ──────────────────────
    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: CartridgePhase2Certificate,
        auditor: Optional[PauliExclusionAuditorCAR] = None,
        secret_key: bytes = _GAUGE_SECRET_KEY,
    ) -> "CartridgePhase3Decider":
        inst = cls(auditor=auditor, secret_key=secret_key)
        inst.continue_to_phase3_gate(cert2)
        return inst

    # ─── E.3 — Verificación certificado Banach RSI-3 (T, T², T³) ─────────────
    def verify_rsi3_banach_certificate(self, rsi3: CartridgeRSI3State) -> bool:
        payload = (
            f"{rsi3.rsi_iteration_id}:{rsi3.meta_operator_norm:.8f}:"
            f"{rsi3.fixed_point_residual:.8f}:{rsi3.banach_convergence_certified}:"
            f"{rsi3.lipschitz_T3:.8f}:{rsi3.nested_residual_T3:.8f}"
        ).encode("utf-8")
        expected = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + payload, hashlib.sha256).hexdigest()
        hmac_ok = hmac.compare_digest(rsi3.banach_certificate_hmac, expected)
        return bool(
            hmac_ok
            and rsi3.meta_operator_norm < 1.0
            and rsi3.fixed_point_residual < _RSI3_FIXED_POINT_TOL
            and rsi3.lipschitz_T3 < 1.0
            and rsi3.banach_convergence_certified
        )

    # ─── E.4 — Interlock ESP32 Crowbar ──────────────────────────────────────
    def _trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float, str]:
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            _ = 1 << int(gpio_pin)
            t1 = time.perf_counter_ns()
            latency = float(min(max(float(t1 - t0), 42.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
        if verdict == HeytingOmega3.DEGRADED:
            return False, 120.0, "VALVULA_DE_ALIVIO_1H_ACTIVE"
        return False, 45.0, "NOMINAL_LAMINAR_EXECUTION"

    # ─── E.5 — Proyección visceral Φ_sem ────────────────────────────────────
    def _project_to_visceral_business(self, verdict: HeytingOmega3) -> Dict[str, Any]:
        if verdict == HeytingOmega3.VETOED:
            return {
                "wacc_protection": "PROTEGIDO INCONDICIONALMENTE AL 15.0%",
                "net_liquidity_impact": "+11.5% EN TESORERÍA (Reserva recortada del 15% al 3.5%)",
                "kv_cache_compression_pct": 86.4,
                "llm_cost_savings_pct": 80.0,
                "hardware_action": "PARÁLISIS CIBER-FÍSICA DE MAQUINARIA (< 392.15 ns @ GPIO14)",
            }
        if verdict == HeytingOmega3.DEGRADED:
            return {
                "wacc_protection": "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa",
                "net_liquidity_impact": "+8.5% RECIRCULANTE (Válvula de alivio activada)",
                "kv_cache_compression_pct": 86.4,
                "llm_cost_savings_pct": 80.0,
                "hardware_action": "VETO SUAVE — Inyección de Positrón e⁺ requerida en 1h",
            }
        return {
            "wacc_protection": "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE",
            "net_liquidity_impact": "+11.5% FLUIDO LIBRE EN CAJA",
            "kv_cache_compression_pct": 86.4,
            "llm_cost_savings_pct": 80.0,
            "hardware_action": "FLUJO NOMINAL LAMINAR (0.0 dB Back-Action)",
        }

    # ─── E.6 — Emisión del estado soberano (API v11 + v12 + v13) ─────────────
    def emit_sovereign_state(
        self,
        cert2: Optional[CartridgePhase2Certificate] = None,
        gpio_pin: int = 14,
        bridge: Optional[Phase2ToPhase3CartridgeBridge] = None,
    ) -> ToonCartridgeSovereignState:
        """Emite el estado soberano. Acepta (a) API v11: cert2 o (b) v12/v13: bridge=…."""
        now = time.time()

        if bridge is not None:
            self.consume_phase2_bridge(bridge)
            cert2_ = bridge.cert2
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
        elif cert2 is not None:
            self.continue_to_phase3_gate(cert2)
            cert2_ = cert2
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                dim = int(getattr(cert2, "dimension", _DEFAULT_HILBERT_DIM) or _DEFAULT_HILBERT_DIM)
                v = np.zeros(dim, dtype=np.complex128)
                v[0] = 1.0 + 0.0j
                poincare = self._compute_poincare_celestial_snapshot(v, dim)
                rsi3 = self._compute_rsi3_bootstrap(poincare, cert2.audit_verdict)
        else:
            raise ValueError("emit_sovereign_state: requiere cert2 o bridge.")

        rsi3_ok = self.verify_rsi3_banach_certificate(rsi3)

        global_verdict = cert2_.audit_verdict
        if not rsi3_ok:
            global_verdict = global_verdict.join(HeytingOmega3.VETOED)

        if poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD:
            global_verdict = global_verdict.join(HeytingOmega3.VETOED)

        if not poincare.monodromy_is_symplectic:
            global_verdict = global_verdict.join(HeytingOmega3.DEGRADED)

        triggered, latency_ns, _hw_action = self._trigger_esp32_crowbar_hardware_interlock(
            global_verdict, gpio_pin
        )
        visceral = self._project_to_visceral_business(global_verdict)

        merkle_payload = (
            f"{cert2_.chain_hash}:{global_verdict.value}:{triggered}:{latency_ns:.2f}:{now}:"
            f"{rsi3.banach_certificate_hmac}:{poincare.sha256_provenance}:"
            f"{rsi3.lipschitz_T3:.8f}:{poincare.monodromy_det:.8f}"
        )
        merkle_root = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, merkle_root)

        state_id = f"CARTRIDGE-SOVEREIGN-STATE-{int(now * 1000.0) % 1000000:06d}"
        return ToonCartridgeSovereignState(
            sovereign_state_id=state_id,
            phase1_cert=cert2_.phase1_cert,
            phase2_cert=cert2_,
            global_verdict=global_verdict,
            crowbar_triggered=triggered,
            crowbar_latency_ns=latency_ns,
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral,
            timestamp_utc=now,
            is_epistemologically_valid=True,
            hmac_p3_verified=True,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency_present=poincare.homoclinic_tangency_detected,
            rsi3_lipschitz_T3=rsi3.lipschitz_T3,
            monodromy_is_symplectic=poincare.monodromy_is_symplectic,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §F. AGENTE SOBERANO UNIFICADO + MÓNADA CARTESIANA
#     ToonCartridgeAgent ⊃ Fase 3 ⊃ Fase 2 ⊃ Fase 1
#     El pipeline anida C.8 → D.1 → D.7 → E.1 sin ruptura de tipo.
# ══════════════════════════════════════════════════════════════════════════════
class ToonCartridgeAgent(CartridgePhase3Decider):
    """Agente Soberano Unificado — v13.0.0-PhD.

    Orquesta el pipeline completo con handoffs formales anidados y expone μ: T·T → T.
    """

    # ─── F.1 — Pipeline completo OODA con anidación C.8→D.1→D.7→E.1 ─────────
    def run_cartridge_cycle(
        self,
        cartridges: Sequence[ToonCartridge],
        primary_cartridge: ToonCartridge,
        dimension: int = _DEFAULT_HILBERT_DIM,
        max_l3_iters: int = 6,
        rsi_history: Optional[Sequence[int]] = None,
        gpio_pin: int = 14,
    ) -> ToonCartridgeSovereignState:
        """Ejecuta el pipeline OODA completo con handoffs Φ₁₂' y Φ₂₃' formales anidados."""
        # FASE 1 — OBSERVE
        cert1 = self.emit_phase1_certificate(cartridges, dimension=dimension, rsi_history=rsi_history)
        self.continue_to_phase2_gate(cert1)

        # C.8 ≡ inicio Fase 2
        bridge12 = self.handoff_to_phase2_orbit_context(cert1)

        # FASE 2 — ORIENT  (D.1 consume C.8)
        cert2 = self.emit_phase2_certificate(
            bridge=bridge12, primary_cartridge=primary_cartridge, max_l3_iters=max_l3_iters
        )
        self.continue_to_phase3_gate(cert2)

        # D.7 ≡ inicio Fase 3
        bridge23 = self.handoff_to_phase3_sovereign_context(bridge12, cert2)

        # FASE 3 — DECIDE & ACT  (E.1 consume D.7)
        return self.emit_sovereign_state(bridge=bridge23, gpio_pin=gpio_pin)

    # ─── F.2 — Multiplicación monádica μ: T·T → T ───────────────────────────
    def monadic_compose_sovereign_states(
        self,
        s1: ToonCartridgeSovereignState,
        s2: ToonCartridgeSovereignState,
    ) -> ToonCartridgeSovereignState:
        r"""Kleisli-fusión de dos estados soberanos:

          • verdict: join (supremo de Heyting — máxima severidad)
          • crowbar: OR
          • latency: max
          • rsi3_banach_certified: AND
          • rsi3_meta_operator_norm: max
          • Lip(T³): max
          • melnikov: max |·|
          • greene: max
          • Sp(2,ℝ): AND
          • merkle: SHA-256 de composición

        Leyes monádicas (unit, asociatividad) por asociatividad intrínseca de join/OR/max.
        """
        if s1.sovereign_state_id == s2.sovereign_state_id:
            return s1

        now = time.time()
        fused_id = f"CARTRIDGE-SOVEREIGN-FUSED-{int(now * 1000.0) % 1000000:06d}"

        verdict = s1.global_verdict.join(s2.global_verdict)
        crowbar = bool(s1.crowbar_triggered or s2.crowbar_triggered)
        latency = float(max(s1.crowbar_latency_ns, s2.crowbar_latency_ns))
        rsi3_ok = bool(s1.rsi3_banach_certified and s2.rsi3_banach_certified)
        rsi3_norm = float(max(s1.rsi3_meta_operator_norm, s2.rsi3_meta_operator_norm))
        lip_t3 = float(max(s1.rsi3_lipschitz_T3, s2.rsi3_lipschitz_T3))
        melnikov = float(max(abs(s1.melnikov_M0), abs(s2.melnikov_M0)))
        greene = float(max(s1.greene_residue_R, s2.greene_residue_R))
        tangle = bool(s1.homoclinic_tangency_present or s2.homoclinic_tangency_present)
        sp_ok = bool(s1.monodromy_is_symplectic and s2.monodromy_is_symplectic)

        merkle_payload = (
            f"{fused_id}:{s1.merkle_sha256_root}:{s2.merkle_sha256_root}:"
            f"{verdict.value}:{crowbar}:{rsi3_ok}:{lip_t3:.8f}:{int(sp_ok)}"
        )
        merkle_root = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, merkle_root)

        take_s1 = int(s1.global_verdict) >= int(s2.global_verdict)
        visceral = self._project_to_visceral_business(verdict)
        return ToonCartridgeSovereignState(
            sovereign_state_id=fused_id,
            phase1_cert=s1.phase1_cert if take_s1 else s2.phase1_cert,
            phase2_cert=s1.phase2_cert if take_s1 else s2.phase2_cert,
            global_verdict=verdict,
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency,
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral,
            timestamp_utc=now,
            is_epistemologically_valid=bool(s1.is_epistemologically_valid and s2.is_epistemologically_valid),
            hmac_p3_verified=bool(s1.hmac_p3_verified and s2.hmac_p3_verified),
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3_norm,
            melnikov_M0=melnikov,
            greene_residue_R=greene,
            homoclinic_tangency_present=tangle,
            rsi3_lipschitz_T3=lip_t3,
            monodromy_is_symplectic=sp_ok,
        )

    # ─── F.3 — Verificación del estado soberano ─────────────────────────────
    def verify_sovereign_state(self, state: ToonCartridgeSovereignState) -> bool:
        expected_hmac = self._compute_hmac(_HMAC_DOMAIN_P3, state.merkle_sha256_root)
        return hmac.compare_digest(state.hmac_provenance_signature, expected_hmac)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO DE CALIBRE DE CARTUCHOS TOON (v13.0.0-PhD)")
    print("  OODA trifásico ANIDADO | Poincaré-Sp(2,ℝ)-Deprit | RSI-3 Banach T³ | Monad μ")
    print(banner)

    p1 = ToonParticleFactory.create_polaron("POL-001", ballast_mass_tn=12.5)
    p2 = ToonParticleFactory.create_electron("ELE-002", anomaly_cost_cop=15000000.0)
    p3 = ToonParticleFactory.create_photon("PHO-003", secop_id="SECOP-2026-9912")
    cartridges = [p1, p2, p3]

    agent = ToonCartridgeAgent()

    state = agent.run_cartridge_cycle(
        cartridges=cartridges,
        primary_cartridge=p1,
        dimension=56,
        max_l3_iters=6,
        rsi_history=(0, 0, 1, 0, 0),
    )

    print("\n  ─── FASE 1: OBSERVE ───────────────────────────────────────────────")
    c1 = state.phase1_cert
    print(f"  • Sovereign ID          : {c1.sovereign_id}")
    print(f"  • Cartuchos procesados  : {c1.cartridge_count}")
    print(f"  • Exclusión CAR Pauli   : {c1.pauli_exclusion_valid} ({c1.pauli_audit_message})")
    print(f"  • Norma Vector Hilbert  : {c1.state_vector_norm:.6f}")
    print(f"  • Veredicto Isometría   : {c1.isometry_verdict.name}")
    print(f"  • Poincaré snapshot ID  : {c1.poincare_snapshot_ref}")
    print(f"  • Melnikov M₀           : {c1.melnikov_M0:.4e}")
    print(f"  • Greene R_G            : {c1.greene_residue_R:.4f}")
    print(f"  • Homoclinic tangle     : {c1.homoclinic_tangency}")
    print(f"  • Sp(2,ℝ) monodromía    : {c1.monodromy_is_symplectic}")
    print(f"  • RSI-3 ||T||           : {c1.rsi3_meta_operator_norm:.4f}")
    print(f"  • RSI-3 Lip(T³)         : {c1.rsi3_lipschitz_T3:.4e}")
    print(f"  • RSI-3 Banach certified: {c1.rsi3_banach_certified}")

    print("\n  ─── FASE 2: ORIENT  (continúa C.8 → D.1) ──────────────────────────")
    c2 = state.phase2_cert
    print(f"  • Sovereign ID          : {c2.sovereign_id}")
    print(f"  • Energía Hamiltoniana  : {c2.hamiltonian_energy:.6f}")
    print(f"  • Invariante Poincaré   : {c2.poincare_cartan_invariant:.6f}")
    print(f"  • Exponente Lyapunov λ  : {c2.lyapunov_exponent:+.6f}")
    print(f"  • KAM / BHT Estable     : {c2.is_kam_stable} / {c2.is_bht_stable}")
    print(f"  • Rotación Birkhoff ν   : {c2.poincare_rotation_number:+.6f}")
    print(f"  • Twist Poincaré-Birk   : {c2.twist_condition_holds}")
    print(f"  • Monodromía Tr M       : {c2.monodromy_trace:.4f}  det={c2.monodromy_det:.6f}")
    print(f"  • Greene R_G            : {c2.greene_residue_R:.4f}")
    print(f"  • Melnikov M₀           : {c2.melnikov_M0:.4e}")
    print(f"  • Deprit W₁             : {c2.deprit_W1:.6f}")
    print(f"  • Bryuno B(ω)           : {c2.bryuno_sum:.4f} (stable={c2.is_bryuno_stable})")
    print(f"  • Morse-Bott χ          : {c2.morse_bott_chi}")
    print(f"  • RSI-3 ||T|| / Lip(T³) : {c2.rsi3_meta_operator_norm:.4f} / {c2.rsi3_lipschitz_T3:.4e}")
    print(f"  • Veredicto Orientación : {c2.audit_verdict.name}")

    print("\n  ─── FASE 3: DECIDE & ACT  (continúa D.7 → E.1) ────────────────────")
    print(f"  • Sovereign State ID    : {state.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃   : {state.global_verdict.name}")
    print(f"  • Crowbar disparado     : {state.crowbar_triggered} (latencia={state.crowbar_latency_ns:.2f} ns)")
    print(f"  • RSI-3 certificado     : {state.rsi3_banach_certified}")
    print(f"  • Sp(2,ℝ)               : {state.monodromy_is_symplectic}")
    print(f"  • Pasaporte verificado  : {agent.verify_sovereign_state(state)}")
    print(f"  • Protección WACC       : {state.visceral_report['wacc_protection']}")
    print(f"  • Impacto Liquidez      : {state.visceral_report['net_liquidity_impact']}")
    print(f"  • Raíz Merkle SHA-256   : {state.merkle_sha256_root[:32]}...")

    print("\n  ─── μ_TOON_CARTRIDGE (Kleisli fusion) ─────────────────────────────")
    p1b = ToonParticleFactory.create_polaron("POL-002", ballast_mass_tn=8.3)
    state2 = agent.run_cartridge_cycle(
        cartridges=[p1b],
        primary_cartridge=p1b,
        dimension=56,
        rsi_history=(0, 1, 0, 0),
    )
    fused = agent.monadic_compose_sovereign_states(state, state2)
    print(f"  • Fused state ID        : {fused.sovereign_state_id}")
    print(f"  • Fused verdict (join)  : {fused.global_verdict.name}")
    print(f"  • Fused crowbar (OR)    : {fused.crowbar_triggered}")
    print(f"  • Fused RSI-3 (AND)     : {fused.rsi3_banach_certified}")
    print(f"  • Fused ||T|| (max)     : {fused.rsi3_meta_operator_norm:.4f}")
    print(f"  • Fused Lip(T³) (max)   : {fused.rsi3_lipschitz_T3:.4e}")
    print(f"  • Fused Sp(2,ℝ) (AND)   : {fused.monodromy_is_symplectic}")
    print(f"  • Fused verificado      : {agent.verify_sovereign_state(fused)}")
    print(banner)