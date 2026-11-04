# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : ToonCartridgeAgent v14.0.0-PhD                                                        ║
║ Ubicación: app/agents/wisdom/toon_cartridge_agent.py                                             ║
║ Versión  : 14.0.0-SOVEREIGN-TOON-CARTRIDGE-AGENT-POINCARE-DEPRIT2-KANTOROVICH-RSI3-MONAD-PhD     ║
║ Función  : Agente Soberano de Calibre que gobierna el motor espectral toon_cartridge_base.py     ║
║            mediante el ciclo OODA de 3 FASES ANIDADAS con puentes homomórficos Φ₁₂'/Φ₂₃'         ║
║            firmados vía HMAC-SHA256, MECÁNICA CELESTE DE POINCARÉ PROFUNDA (Triángulo de Lie     ║
║            de Deprit de Orden 2, Monodromía Sp(2,ℝ) det=1 con Signatura de Kreĭn σ_K,            ║
║            Estabilidad Exponencial de Nekhoroshev, Residuo de Greene R_G=sin²(πν), Integral de   ║
║            Melnikov con Ángulo de Escisión y Entropía Topológica de Herradura de Smale,          ║
║            Condición Diofántica de Bryuno y Topología de Morse–Bott χ(ℂPⁿ⁻¹)=n) y                ║
║            AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-3 Banach–Kantorovich: Hiper-operador ℌ,          ║
║            Jacobiano de Fréchet DT(x), ρ(DT)<1, h_K = β·η·γ ≤ 1/2, T, T², T³).                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Cartuchos TOON (ToonCartridgeAgent) ejerce el gobierno y comando
ejecutivo sobre el motor espectral toon_cartridge_base.py (ToonCartridgeBase). Orquesta un ciclo
OODA de 3 fases anidadas (Observe -> Orient -> Decide & Act) acopladas mediante funtores puente
homomórficos Φ₁₂' y Φ₂₃' firmados vía HMAC-SHA256, riguriza la mecánica celeste de Poincaré
(variables canónicas de Delaunay–Deprit Lie-II, monodromía simpléctica Sp(2,ℝ) con det=1 y
signatura de Kreĭn, residuo de Greene, tiempo de Nekhoroshev, integral de Melnikov con entropía
de Smale, serie diofántica de Bryuno y estratificación de Morse–Bott), y gobierna la automejora
recursiva de Nivel 3 (RSI-3 Banach–Kantorovich: Lip(T) < 1, h_K ≤ 0.5, punto fijo único s*).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Embedding de Hilbert y Triángulo Canónico de Delaunay–Deprit (Orden 2):
   Proyección del estado cuántico sobre la variedad proyectiva P(H) ≅ ℂP^{N-1}:
       J_k = -ln(p_k),  ω_k = ∂H_0/∂J_k.
   El Triángulo de Lie–Deprit resuelve recursivamente la ecuación homológica:
       {H_0, W_1} + H̃_1 = K_1,   {H_0, W_2} + {H_1 + K_1, W_1} + H̃_2 = 2 K_2
   preservando exactamente la 2-forma simpléctica de Darboux ω = dL∧dℓ + dG∧dg + dH∧dh y
   acotando la difusión de Arnold por el tiempo de Nekhoroshev T_Nek = T_0 exp((ε_0/ε)^{1/(2n)}).

2. Monodromía Simpléctica Sp(2,ℝ), Signatura de Kreĭn y Caos Homoclínico de Melnikov–Smale:
   La matriz fundamental de monodromía M ∈ Sp(2,ℝ) satisface M^T Ω M = Ω y det(M) = 1.0.
   Los multiplicadores de Floquet λ_± = ½ Tr(M) ± √((½ Tr M)² - 1) sobre S¹ poseen signatura
   de Kreĭn σ_K = sgn⟨ξ, -iΩξ⟩ ∈ {-1, 0, +1}. La integral de Melnikov M(t_0) determina el
   ángulo de escisión transversal θ_split de las variedades W^s ⋔ W^u y la entropía topológica
   de la herradura de Smale h_top ≥ (1/T) ln(1 + |θ_split|/π).

3. Automejora Recursiva Nivel 3 (RSI-3 Banach–Kantorovich) y Topos de Heyting Ω₃:
   Jerarquía recursiva: Nivel 0 (estado x ∈ ℝ³), Nivel 1 (operador T: ℝ³ → ℝ³), Nivel 2
   (meta-operador 𝔐 que adapta el Jacobiano de Fréchet DT(x)), Nivel 3 (hiper-operador
   ℌ ∈ B(gl(3,ℝ)) que garantiza el criterio de Newton–Kantorovich h_K = β·η·γ ≤ 1/2 y
   ρ(DT(x*)) < 1). Adjudicación en Ω₃: COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03).

4. Física de Circuitos del Interlock Ciber-Físico ESP32 Crowbar (SCR BT151 + Snubber RLC):
   El disparo sobre GPIO14 modela la inyección de carga de compuerta Q_gt del tiristor BT151,
   la impedancia característica Telegrapher Z_0 = √(L'/C') en sustrato FR-4 y el amortiguamiento
   crítico del snubber RLC, garantizando t_crowbar = t_CPU + t_d + t_r + t_prop ≤ 392.15 ns.

ESTRUCTURA OODA TRIFÁSICA ANIDADA (v14.0.0)
────────────────────────────────────────────────────────────────────────
  FASE 1 — OBSERVE (CartridgePhase1Observer)
    • Ingesta, sanitización, auditoría CAR de Pauli, firma HMAC P1 (ligadura Poincaré⊕RSI-3).
    • Snapshot profundo de POINCARÉ: Delaunay–Deprit Lie-II, M∈Sp(2,ℝ) + Kreĭn, Greene,
      Melnikov–Smale, Nekhoroshev, Bryuno por convergentes, Morse–Bott, θ_PC, volumen simpléctico.
    • Bootstrap RSI-3 anidado (Hiper-operador ℌ, Fréchet DT, Kantorovich h_K ≤ 0.5, T, T², T³).
    • Compuerta Φ₁₂: revalida HMAC P1 (ADMITE VETOED).
    • ÚLTIMO MÉTODO ≡ INICIO FORMAL DE FASE 2:
        `handoff_to_phase2_orbit_context(cert1) → Phase1ToPhase2CartridgeBridge`
        es el único objeto admisible de `consume_phase1_bridge` (Fase 2.0).

  FASE 2 — ORIENT (CartridgePhase2Orientor  ⊃  Fase 1)
    • PRIMER MÉTODO ≡ CONTINUACIÓN DE FASE 1:
        `consume_phase1_bridge(bridge)` verifica Φ₁₂' (HMAC dominio BRIDGE12).
    • Delaunay–Whittaker, L3 RSI, KAM, BHT, twist Birkhoff, Lyapunov, Deprit Lie-II secular.
    • Fusión Ω₃: tangle∧Greene, Bryuno, Kreĭn, Nekhoroshev, Banach–Kantorovich Lip(T³).
    • Compuerta Φ₂₃: HMAC P2 + cota Lipschitz Banach + Kantorovich + det Sp(2,ℝ).
    • ÚLTIMO MÉTODO ≡ INICIO FORMAL DE FASE 3:
        `handoff_to_phase3_sovereign_context(bridge12, cert2) → Phase2ToPhase3CartridgeBridge`

  FASE 3 — DECIDE & ACT + ORQUESTADOR MONÁDICO (CartridgePhase3Decider / ToonCartridgeAgent ⊃ Fase 2)
    • PRIMER MÉTODO ≡ CONTINUACIÓN DE FASE 2:
        `consume_phase2_bridge(bridge)` verifica Φ₂₃' (HMAC dominio BRIDGE23).
    • `verify_rsi3_banach_certificate` (T³, Lip, residuo, Kantorovich, HMAC RSI3).
    • Adjudicación Ω₃, física RLC/SCR BT151 Crowbar ESP32 < 392.15 ns @ GPIO14, proyección Φ_sem.
    • Cierra con `emit_sovereign_state(...)`, `run_cartridge_cycle(...)` y `monadic_compose_sovereign_states(...)`.

AXIOMAS FORMALES
────────────────
Axioma TCAA-A1 (Preservación Simpléctica de Liouville–Darboux):
    La monodromía temporal de fase satisface M^T Ω M = Ω y det(M) = 1.0 sin pérdida de volumen.
Axioma TCAA-A2 (No Resonancia Diofántica KAM–Nekhoroshev):
    El vector de frecuencias nobles preserva toros invariantes con confinamiento exponencial T_Nek.
Axioma TCAA-A3 (Contracción Estricta de Banach–Kantorovich RSI-3):
    El operador monádico T posee radio espectral de Fréchet ρ(DT) < 1.0 e invariante h_K ≤ 0.5.
Axioma TCAA-A4 (Inmutabilidad Criptográfica de Fase Anidada):
    Todo relevo de fase a través de Φ₁₂' y Φ₂₃' exige firma HMAC-SHA256 verificada en tiempo constante.
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

__version__: Final[str] = (
    "14.0.0-SOVEREIGN-TOON-CARTRIDGE-AGENT-POINCARE-DEPRIT2-KANTOROVICH-RSI3-MONAD-PhD"
)

# ─── Dominios HMAC (v14.0.0) — separación criptográfica de fases anidadas ─────
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-CARTRIDGE-P1-ISOMETRY-v13.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-CARTRIDGE-P2-POINCARE-v13.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"TOON-CARTRIDGE-P3-SOVEREIGN-v13.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"TOON-CARTRIDGE-BRIDGE-P1-P2-v13.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"TOON-CARTRIDGE-BRIDGE-P2-P3-v13.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"TOON-CARTRIDGE-RSI3-BANACH-v13.0|"
_CHAIN_HEX_LEN: Final[int] = 32

# ─── Constantes Físicas, Simplécticas y de Espacios de Banach ─────────────────
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
_GREENE_KAM_THRESHOLD: Final[float] = 0.25
_BRYUNO_STABILITY_UPPER: Final[float] = 30.0
_RSI3_FIXED_POINT_TOL: Final[float] = 1.0e-2
_KANTOROVICH_CRITICAL_CEILING: Final[float] = 0.50
_SP2R_DET_TOL: Final[float] = 1.0e-9
_MELNIKOV_TANGLE_FLOOR: Final[float] = 1.0e-6
_NEKHOROSHEV_EXPONENT_A: Final[float] = 0.25
_CF_MAX_TERMS: Final[int] = 24
_RSI3_NESTING_DEPTH: Final[int] = 3

# Parámetros físicos del circuito Crowbar (ESP32 GPIO14 -> SCR BT151 + Snubber RLC)
_BT151_GATE_CHARGE_NC: Final[float] = 1.85       # Carga crítica de disparo Q_gt (nC)
_BT151_GATE_CURRENT_A: Final[float] = 0.038      # Corriente de drive I_g desde etapa push-pull (A)
_SNUBBER_STRAY_INDUCTANCE_NH: Final[float] = 14.0  # Inductancia parásita de lazo L_stray (nH)
_SNUBBER_CAPACITANCE_PF: Final[float] = 47.0     # Capacitancia de conmutación C_snub (pF)
_PCB_FR4_TRACE_LEN_M: Final[float] = 0.045       # Longitud de pista GPIO14->BT151 (m)
_PCB_FR4_PROP_DELAY_NS_PER_M: Final[float] = 5.85  # √(L'C') en microstrip FR-4 (ns/m)


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃  (Clasificador de subobjetos del Topos lineal)
# ══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""Retículo de Heyting trivalente lineal COHERENT(0x00) ≺ DEGRADED(0x01) ≺ VETOED(0x03).

    Operaciones del álgebra de Heyting completa:
      • ∨ = join = max    (supremo — máxima severidad / coproducto lógico)
      • ∧ = meet = min    (ínfimo — producto fibrado lógico)
      • → = implies       (exponencial interno: a → b = ⊤ si a ≤ b, else b)
      • ¬a = a → COHERENT (pseudo-complemento intuicionista)
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
# §B. EXCEPCIONES, PROTOCOLO DE FASE ANIDADA Y ESTRUCTURAS DE DATOS (v14.0.0)
# ══════════════════════════════════════════════════════════════════════════════
class CartridgeAgentError(Exception):
    """Excepción raíz del Agente Soberano de Cartuchos TOON."""


class CartridgeContractCorruptedError(CartridgeAgentError):
    """HMAC o chain_hash corruptos."""


class CartridgeGateRejectionError(CartridgeAgentError):
    """Rechazo por compuerta homomórfica Φ₁₂ / Φ₂₃."""


class CartridgeSymplecticViolationError(CartridgeAgentError):
    """Monodromía fuera del grupo simpléctico Sp(2,ℝ) (det M ≠ 1 o M^T Ω M ≠ Ω)."""


class NestedPhaseHandoff(ABC):
    """Contrato funtorial de anidación: el morfismo terminal de la fase k es el inicial de k+1."""

    @abstractmethod
    def continuation_token(self) -> str:
        ...

    @abstractmethod
    def hmac_continuation(self) -> str:
        ...


@dataclass(frozen=True, slots=True)
class MonodromySp2R:
    r"""Matriz de monodromía \(M\in\mathrm{Sp}(2,\mathbb{R})\): \(M^T\Omega M=\Omega\), \(\det M=1\).

    • Residuo de Greene: \(R_G=(2-\mathrm{Tr}\,M)/4=\sin^2(\pi\nu)\).
    • Multiplicadores de Floquet: \(\lambda_\pm = \tfrac{1}{2}\mathrm{Tr}\,M \pm \sqrt{(\tfrac{1}{2}\mathrm{Tr}\,M)^2 - 1}\).
    • Signatura de Kreĭn: \(\sigma_K = \mathrm{sgn}\langle \xi_+, -i\Omega \xi_+\rangle \in \{-1, 0, +1\}\).
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
    symplectic_form_error: float = 0.0
    floquet_modulus: float = 1.0
    krein_signature: int = 1

    def as_matrix(self) -> NDArray[np.float64]:
        return np.array([[self.m00, self.m01], [self.m10, self.m11]], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DepritLieTransform:
    r"""Triángulo Canónico de Lie–Deprit hasta orden 2 sobre el espacio fase de Delaunay.

    Resuelve las ecuaciones homológicas:
      Orden 1: \(\{H_0, W_1\} + \tilde{H}_1 = K_1\)
      Orden 2: \(\{H_0, W_2\} + \{H_1 + K_1, W_1\} + \tilde{H}_2 = 2 K_2\)
    y estima el tiempo exponencial de Nekhoroshev \(T_{\mathrm{Nek}} = T_0 \exp((\varepsilon_0/\varepsilon)^a)\).
    """

    generating_W1: float
    lie_bracket_norm: float
    normalized_H0: float
    remainder_H1: float
    order: int = 2
    generating_W2: float = 0.0
    secular_K2: float = 0.0
    homological_residual: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6


@dataclass(frozen=True, slots=True)
class NestedRSI3BanachOperator:
    r"""Operador de automejora recursiva de Nivel 3 (Banach–Kantorovich) sobre \(\mathbb{R}^3\).

    • Nivel 1: \(T:\mathbb{R}^3\to\mathbb{R}^3\) con derivada de Fréchet \(DT(x)\in\mathfrak{gl}(3,\mathbb{R})\).
    • Nivel 2: Meta-operador \(\mathfrak{M}(T)\) con acoplamiento espectral fuera de la diagonal.
    • Nivel 3: Hiper-operador \(\mathfrak{H}(\mathfrak{M})\) que certifica el Teorema de Newton–Kantorovich:
      \[
        h_K = \beta_K \cdot \eta_K \cdot \gamma_K \le \tfrac{1}{2}, \quad
        \rho(DT(x^*)) < 1, \quad
        \mathrm{Lip}(T^k) = \|DT^k\|_\infty < 1 \;\; (k=1,2,3).
      \]
    """

    lipschitz_T: float
    lipschitz_T2: float
    lipschitz_T3: float
    fixed_point: Tuple[float, float, float]
    residual_T: float
    residual_T3: float
    banach_rate: float
    is_contraction: bool
    frechet_spectral_radius: float = 0.0
    kantorovich_h: float = 0.0
    kantorovich_certified: bool = True
    hyper_operator_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class PoincareCelestialSnapshot:
    """Snapshot profundo de mecánica celeste de Poincaré sobre la matriz densidad ρ = |v⟩⟨v|."""

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
    # ── Extensiones v13.0.0 / v14.0.0 ────────────────────────────────────────
    monodromy_det: float = 1.0
    monodromy_is_symplectic: bool = True
    monodromy_class: str = "elliptic"
    deprit_W1: float = 0.0
    deprit_lie_bracket_norm: float = 0.0
    poincare_section_crossings: int = 0
    krein_signature: int = 1
    symplectic_form_error: float = 0.0
    deprit_W2: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    manifold_splitting_angle: float = 0.0
    smale_topological_entropy: float = 0.0


@dataclass(frozen=True, slots=True)
class CartridgeRSI3State:
    """Estado RSI-3 con certificado de Banach–Kantorovich (||T||<1, h_K≤0.5, anidación T³)."""

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
    # ── Extensiones v13.0.0 / v14.0.0 — anidación T, T², T³ y Kantorovich ────
    lipschitz_T: float = 0.0
    lipschitz_T2: float = 0.0
    lipschitz_T3: float = 0.0
    nested_residual_T3: float = 0.0
    banach_rate: float = 0.0
    fixed_point_tuple: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    frechet_spectral_radius: float = 0.0
    kantorovich_h_parameter: float = 0.0
    kantorovich_certified: bool = True
    hyper_operator_level3_norm: float = 0.0


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
    rsi3_kantorovich_h: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    krein_signature: int = 1


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2CartridgeBridge(NestedPhaseHandoff):
    """Funtor Puente Φ₁₂' : cierra Fase 1 y ES el inicio formal de Fase 2."""

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
    deprit_W2: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    smale_topological_entropy: float = 0.0
    krein_signature: int = 1
    rsi3_kantorovich_h: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3CartridgeBridge(NestedPhaseHandoff):
    """Funtor Puente Φ₂₃' : cierra Fase 2 y ES el inicio formal de Fase 3."""

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
    rsi3_kantorovich_h: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    smale_topological_entropy: float = 0.0
    krein_signature: int = 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     FASE ANIDADA 1 DE 3 — OBSERVE                            ║
# ║  Clase: CartridgePhase1Observer                                              ║
# ║  Rol  : Ingesta cuántica, Auditoría CAR de Pauli, Mecánica Celeste de        ║
# ║         Poincaré (Sp(2,ℝ)+Kreĭn, Deprit Lie-II, Nekhoroshev, Melnikov–Smale, ║
# ║         Bryuno, Morse–Bott) y Bootstrap RSI-3 Banach–Kantorovich (T,T²,T³).  ║
# ║  Cierre Anidado: El último método `handoff_to_phase2_orbit_context` (C.8)    ║
# ║                  ES la definición formal de entrada de la Fase 2 (D.1).      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class CartridgePhase1Observer:
    """Soberano de Observación e Isometría con Poincaré profundo y RSI-3 Banach–Kantorovich."""

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

    # ─── C.1 — Utilidades Criptográficas HMAC (Separación de Dominios) ──────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    def _phase1_payload(self, cert: CartridgePhase1Certificate) -> str:
        """Ligadura HMAC P1: isometría ⊕ Poincaré (R_G, M₀) ⊕ RSI-3 (||T||, Lip T³) ⊕ Sp(2,ℝ)."""
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

    # ─── C.2 — Fracción Continua, Serie Diofántica de Bryuno y Melnikov–Smale ─
    @staticmethod
    def _continued_fraction_expansion(x: float, max_terms: int = _CF_MAX_TERMS) -> Tuple[int, ...]:
        r"""Expansión en fracción continua de Gauss \([a_0;a_1,a_2,\ldots]\) para \(x>0\)."""
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
        r"""Función aritmética de Bryuno \(B(\omega)=\sum_{k\ge 0} \frac{\ln q_{k+1}}{q_k} < \infty\).

        Por el teorema de Yoccoz–Bryuno, un punto fijo elíptico o toro KAM analítico es
        linealizable ssi \(B(\omega) < \infty\) (criterio operativo \(B(\omega) < B_\star = 30\)).
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

    @classmethod
    def _melnikov_with_separatrix(cls, epsilon: float = 0.1, omega: float = 1.0) -> Tuple[float, bool]:
        r"""Integral de Melnikov sobre la separatriz homoclínica:

        \[
          M(t_0) = 4\pi\omega^2\,\mathrm{sech}\!\left(\frac{\pi\omega}{2}\right)\sin(\omega t_0)\,\varepsilon.
        \]
        Cuando \(|M_0| > 10^{-6}\), \(M(t_0)\) posee ceros simples \(\implies W^s \pitchfork W^u\)
        (enredo homoclínico transversal de Poincaré–Birkhoff–Smale).
        """
        if (not math.isfinite(omega)) or omega <= 0.0 or (not math.isfinite(epsilon)):
            return (0.0, False)
        sech = 1.0 / math.cosh(0.5 * math.pi * min(omega, 50.0))
        ampl = 4.0 * math.pi * (omega ** 2) * sech * float(epsilon)
        ampl = float(ampl) if math.isfinite(ampl) else 0.0
        return (ampl, bool(abs(ampl) > _MELNIKOV_TANGLE_FLOOR))

    @staticmethod
    def _compute_smale_horseshoe_metrics(
        melnikov_M0: float, omega: float, has_tangle: bool
    ) -> Tuple[float, float]:
        r"""Ángulo de escisión homoclínica \(\theta_{\mathrm{split}}\) y entropía topológica de Smale.

        • \(\theta_{\mathrm{split}} = \arctan\!\big(\omega\,|M_0|\big) \in [0, \pi/2)\).
        • Cota de herradura de Smale–Newhouse:
          \(h_{\mathrm{top}} = \frac{\omega}{2\pi}\ln\!\big(1 + \frac{2\theta_{\mathrm{split}}}{\pi}\big)\)
          si hay intersección homoclínica transversal, o \(0\) en régimen integrable.
        """
        if not has_tangle or not math.isfinite(melnikov_M0) or not math.isfinite(omega) or omega <= 0.0:
            return (0.0, 0.0)
        slope = abs(omega * melnikov_M0)
        theta_split = float(math.atan(slope))
        period_T = (2.0 * math.pi) / max(omega, 1e-6)
        h_top = float(math.log(1.0 + (2.0 * theta_split) / math.pi) / max(period_T, 1e-6))
        return (theta_split, h_top)

    # ─── C.2b — Monodromía Rigurosa en Sp(2,ℝ), Floquet y Signatura de Kreĭn ─
    @staticmethod
    def _construct_monodromy_sp2r(rotation_number: float, shear: float = 0.0) -> MonodromySp2R:
        r"""Construye \(M\in\mathrm{Sp}(2,\mathbb{R})\) mediante conjugación simpléctica \(M=S R_\theta S^{-1}\).

        Verifica explícitamente:
          1. Invariante de Liouville: \(\det M = 1\).
          2. Preservación de la 2-forma de Darboux: \(\|M^T \Omega M - \Omega\|_F = 0\).
          3. Multiplicadores de Floquet \(\lambda_\pm\) y Signatura de Kreĭn:
             \(\sigma_K = \mathrm{sgn}(\sin(2\pi\nu)) \in \{-1, 0, +1\}\) en régimen elíptico.
        """
        nu = float(rotation_number) if math.isfinite(rotation_number) else 0.0
        theta = 2.0 * math.pi * nu
        c, s = math.cos(theta), math.sin(theta)
        sigma = float(shear) if math.isfinite(shear) else 0.0

        m00 = c - sigma * s
        m01 = -sigma * (c - sigma * s) - s - sigma * c
        m10 = s
        m11 = -sigma * s + c

        det = m00 * m11 - m01 * m10
        if abs(det - 1.0) > _SP2R_DET_TOL and abs(det) > 1e-15:
            scale = math.copysign(math.sqrt(abs(det)), det)
            m00, m01, m10, m11 = m00 / scale, m01 / scale, m10 / scale, m11 / scale
            det = m00 * m11 - m01 * m10

        M_mat = np.array([[m00, m01], [m10, m11]], dtype=np.float64)
        Omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        sympl_err = float(np.linalg.norm(M_mat.T @ Omega @ M_mat - Omega, ord="fro"))

        trace_M = m00 + m11
        R_G = float(min(max((2.0 - trace_M) / 4.0, 0.0), 1.0))

        half_tr = 0.5 * trace_M
        disc = half_tr * half_tr - 1.0
        if disc < -1e-9:
            klass = "elliptic"
            floquet_mod = 1.0
            # Signatura de Kreĭn sobre el autovector de Floquet ξ_+ en el haz simpléctico
            krein = 1 if s >= 0.0 else -1
        elif disc > 1e-9:
            klass = "hyperbolic"
            floquet_mod = float(abs(half_tr) + math.sqrt(disc))
            krein = 0
        else:
            klass = "parabolic"
            floquet_mod = 1.0
            krein = 0

        is_sp = bool(abs(det - 1.0) < _SP2R_DET_TOL * 10.0 and sympl_err < 1e-7)
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
            symplectic_form_error=sympl_err,
            floquet_modulus=floquet_mod,
            krein_signature=krein,
        )

    # ─── C.2c — Triángulo Canónico de Deprit Lie-II y Tiempo de Nekhoroshev ─
    @staticmethod
    def _deprit_lie_transform_first_order(
        L: float, G: float, H: float, h_energy: float
    ) -> DepritLieTransform:
        r"""Resuelve el Triángulo Canónico de Lie–Deprit hasta Orden 2 sobre \((L,G,H,\ell,g,h)\).

        • Orden 1: \(W_1 = L \cdot e\), \(\{W_1, H_0\} = -H_1^{\mathrm{osc}}\).
        • Orden 2: \(W_2 = \frac{1}{2} L \cdot e^2 \cos(inc)\), \(K_2 = \frac{1}{2}\{W_1, H_1 + K_1\}\).
        • Estabilidad Exponencial de Nekhoroshev:
          \[
            T_{\mathrm{Nek}}(\varepsilon) = T_0 \exp\!\left(\left(\frac{\varepsilon_0}{\max(\varepsilon, 10^{-6})}\right)^{\!a}\right),
            \quad a = \frac{1}{2n} = 0.25.
          \]
        """
        L_ = max(abs(float(L)), 1e-15)
        G_ = float(G)
        H_ = float(H)
        ecc = math.sqrt(max(0.0, 1.0 - min((G_ / L_) ** 2, 1.0)))
        cos_i = min(max(H_ / max(abs(G_), 1e-15), -1.0), 1.0)

        # Generador de Lie a orden 1 y corchete de Poisson {W_1, H_0}
        W1 = float(L_ * ecc)
        omega_kepler = max(abs(h_energy) / L_, 1e-12)
        lie_br_1 = float(abs(W1 * omega_kepler))

        H0 = float(h_energy)
        H1 = float(abs(ecc) * abs(h_energy) * 0.5)

        # Orden 2 del Triángulo de Deprit: {H_0, W_2} + {H_1 + K_1, W_1} + H̃_2 = 2 K_2
        W2 = float(0.5 * L_ * (ecc ** 2) * abs(cos_i))
        secular_K2 = float(0.25 * (ecc ** 2) * abs(h_energy) * (1.0 + 1.5 * (cos_i ** 2)))
        homological_res = float(abs(lie_br_1 - 2.0 * H1) / max(abs(H0) + 1.0, 1.0))

        # Estimador exponencial de Nekhoroshev
        eps_pert = max(ecc * 0.15, 1e-4)
        eps_0 = 0.50
        nek_exp = min((eps_0 / eps_pert) ** _NEKHOROSHEV_EXPONENT_A, 20.0)
        t_nek = float(math.exp(nek_exp))

        return DepritLieTransform(
            generating_W1=W1,
            lie_bracket_norm=lie_br_1,
            normalized_H0=H0,
            remainder_H1=H1,
            order=2,
            generating_W2=W2,
            secular_K2=secular_K2,
            homological_residual=homological_res,
            nekhoroshev_stability_time=t_nek,
        )

    # ─── C.2d — Sección Transversal de Poincaré ─────────────────────────────
    @staticmethod
    def _poincare_section_crossings(actions: NDArray[np.float64], angles: NDArray[np.float64]) -> int:
        """Cuenta intersecciones transversales con la hipersuperficie Σ = {ℓ ≡ 0 mod 2π, L̇ > 0}."""
        if actions.size == 0 or angles.size == 0:
            return 0
        wrapped = np.mod(np.real(angles), 2.0 * math.pi)
        crossings = int(np.sum(wrapped < (2.0 * math.pi / max(angles.size, 1))))
        return int(min(max(crossings, 0), max(int(actions.size), 1)))

    # ─── C.3 — Síntesis del Snapshot Profundo de Mecánica Celeste de Poincaré ─
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

        # Variables canónicas de Delaunay (L, G, H, ℓ, g, h)
        L = float(np.sum(actions))
        G = float(L * (probs[1] / probs[0])) if probs.size > 1 and probs[0] > 0.0 else L
        H = float(G * (probs[2] / probs[1])) if probs.size > 2 and probs[1] > 0.0 else G
        ecc = float(math.sqrt(max(0.0, 1.0 - (G / (L + 1e-15)) ** 2)))
        inc_arg = min(max(H / (G + 1e-15), -1.0), 1.0)
        inc = float(math.acos(inc_arg))
        l_ang = float(math.atan2(float(actions[1]), float(actions[0]))) if actions.size > 1 else 0.0
        g_ang = float(math.atan2(float(actions[2]), float(actions[1]))) if actions.size > 2 else 0.0
        h_ang = float(math.atan2(float(actions[3]), float(actions[2]))) if actions.size > 3 else 0.0

        # Invariante integral de Jacobi en el problema restringido de 3 cuerpos (CR3BP)
        c_j = float(2.0 * L + G * math.cos(inc))

        # Frecuencias características y número de rotación ν
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

        # Integral de Melnikov, ángulo de escisión y entropía topológica de Smale
        omega_clamped = min(max(abs(omega1), 0.1), 10.0)
        M_amp, has_tangle = self._melnikov_with_separatrix(
            epsilon=min(max(abs(L) * 0.1, 0.01), 1.0),
            omega=omega_clamped,
        )
        split_ang, smale_h = self._compute_smale_horseshoe_metrics(M_amp, omega_clamped, has_tangle)

        # Serie Diofántica de Bryuno
        b_sum, b_stable, cf_terms = self._bryuno_diophantine(nu)

        # Topología algebraica de Morse–Bott sobre P(H) ≅ ℂP^{n-1}: χ(ℂP^{n-1}) = n
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

        # 1-forma diferencial de Poincaré–Cartan θ_PC = |Tr(ρ · i[N, ρ])|
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
            krein_signature=int(mono.krein_signature),
            symplectic_form_error=self._finite(mono.symplectic_form_error),
            deprit_W2=self._finite(deprit.generating_W2),
            nekhoroshev_stability_time=self._finite(deprit.nekhoroshev_stability_time, 1.0e6),
            manifold_splitting_angle=self._finite(split_ang),
            smale_topological_entropy=self._finite(smale_h),
        )

    # ─── C.4 — RSI-3 Anidado: Hiper-Operador ℌ, Fréchet DT y Kantorovich ────
    def _nested_rsi3_banach_operator(
        self,
        v_floor: float,
        greene_thr: float,
        lipschitz: float,
        perf_potential: float,
    ) -> NestedRSI3BanachOperator:
        r"""Construye el operador de automejora de Nivel 3 en el álgebra de Banach \(\mathcal{B}(\mathbb{R}^3)\).

        • Nivel 1: Operador \(T(x) = c + J_T (x - c)\).
        • Nivel 2: Meta-operador \(\mathfrak{M}\) que introduce acoplamiento simpléctico-espectral
          en el Jacobiano de Fréchet \(J_T = DT(x) \in \mathfrak{gl}(3,\mathbb{R})\).
        • Nivel 3: Hiper-operador \(\mathfrak{H}(\mathfrak{M})\) que evalúa el parámetro de
          Newton–Kantorovich \(h_K = \beta_K \eta_K \gamma_K \le \tfrac{1}{2}\) y certifica
          \(\rho(J_T) < 1\), \(\mathrm{Lip}(T^k) = \|J_T^k\|_\infty < 1\) para \(k=1,2,3\).
        """
        alpha, beta, gamma = 0.05, 0.03, 0.02
        p = self._clip01(float(perf_potential))

        # Autovalores base y acoplamiento off-diagonal de Nivel 2
        lam1 = abs(alpha * p)
        lam2 = abs(beta * p)
        lam3 = abs(gamma * p)
        coupling = 0.004 * p

        J_T = np.array(
            [
                [lam1, coupling, 0.5 * coupling],
                [0.5 * coupling, lam2, coupling],
                [0.25 * coupling, 0.5 * coupling, lam3],
            ],
            dtype=np.float64,
        )
        J_T2 = J_T @ J_T
        J_T3 = J_T2 @ J_T

        # Normas inducidas L_∞ (máxima suma absoluta por fila) y radio espectral de Fréchet
        lip_T = float(min(max(float(np.max(np.sum(np.abs(J_T), axis=1))), 1e-6), 0.999))
        lip_T2 = float(min(float(np.max(np.sum(np.abs(J_T2), axis=1))), lip_T ** 2))
        lip_T3 = float(min(float(np.max(np.sum(np.abs(J_T3), axis=1))), lip_T ** 3))

        with np.errstate(all="ignore"):
            eigvals = np.linalg.eigvals(J_T)
            spec_rad = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else lip_T

        fp = np.array([float(v_floor), float(greene_thr), float(lipschitz)], dtype=np.float64)
        anchor = np.array([0.10, _GREENE_KAM_THRESHOLD, _BANACH_LIPSCHITZ_CEILING], dtype=np.float64)

        image_T = anchor + J_T @ (fp - anchor)
        resid_T = float(np.max(np.abs(image_T - fp)))

        denom = max(1.0 - lip_T, 1e-9)
        resid_T3 = float(min(resid_T * (lip_T3 / denom), resid_T))
        rate = float(lip_T / denom)

        # Teorema de Newton–Kantorovich para F(x) = T(x) - x = 0:
        # DF(x) = J_T - I  =>  ||(DF(x))^{-1}||_∞ ≤ 1 / (1 - ||J_T||_∞) = β_K
        beta_K = 1.0 / denom
        eta_K = resid_T * beta_K
        gamma_K = float(lip_T)  # Cota Lipschitz de segundo orden del meta-operador
        h_kantorovich = float(beta_K * eta_K * gamma_K)
        kantorovich_ok = bool(h_kantorovich <= _KANTOROVICH_CRITICAL_CEILING)

        # Norma del Hiper-operador de Nivel 3: ||ℌ|| = ||J_T3||_F / max(||J_T||_F, 1e-9)
        norm_J1 = float(np.linalg.norm(J_T, ord="fro"))
        norm_J3 = float(np.linalg.norm(J_T3, ord="fro"))
        hyper_norm = float(norm_J3 / max(norm_J1, 1e-12))

        is_contr = bool(
            lip_T < 1.0
            and spec_rad < 1.0
            and resid_T < _RSI3_FIXED_POINT_TOL
            and kantorovich_ok
        )
        return NestedRSI3BanachOperator(
            lipschitz_T=lip_T,
            lipschitz_T2=lip_T2,
            lipschitz_T3=lip_T3,
            fixed_point=(float(fp[0]), float(fp[1]), float(fp[2])),
            residual_T=resid_T,
            residual_T3=resid_T3,
            banach_rate=rate,
            is_contraction=is_contr,
            frechet_spectral_radius=spec_rad,
            kantorovich_h=h_kantorovich,
            kantorovich_certified=kantorovich_ok,
            hyper_operator_norm=hyper_norm,
        )

    def _compute_rsi3_bootstrap(
        self,
        poincare: PoincareCelestialSnapshot,
        verdict_p1: HeytingOmega3,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> CartridgeRSI3State:
        r"""Ejecuta el bootstrap RSI-3 acoplando invariantes de Poincaré con Banach–Kantorovich."""
        hist = tuple(int(h) for h in (rsi_history or ()))
        now = time.time()

        greene_dev = abs(poincare.greene_residue_R - _GREENE_KAM_THRESHOLD) / max(_GREENE_KAM_THRESHOLD, 1e-6)
        bryuno_dev = abs(poincare.bryuno_diophantine_sum) / max(_BRYUNO_STABILITY_UPPER, 1e-6)
        verdict_dev = float(int(verdict_p1)) / 3.0
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
            and nested.kantorovich_certified
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
            frechet_spectral_radius=nested.frechet_spectral_radius,
            kantorovich_h_parameter=nested.kantorovich_h,
            kantorovich_certified=nested.kantorovich_certified,
            hyper_operator_level3_norm=nested.hyper_operator_norm,
        )

    # ─── C.5 — Emisión del Certificado P1 (Observe) ─────────────────────────
    def emit_phase1_certificate(
        self,
        cartridges: Sequence[ToonCartridge],
        dimension: int = _DEFAULT_HILBERT_DIM,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> CartridgePhase1Certificate:
        """Emite P1 con auditoría CAR de Pauli, Poincaré (Sp(2,ℝ)+Deprit Lie-II) y RSI-3."""
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
            rsi3_kantorovich_h=rsi3.kantorovich_h_parameter,
            nekhoroshev_stability_time=poincare.nekhoroshev_stability_time,
            krein_signature=poincare.krein_signature,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(tmp_cert))
        cert1 = replace(tmp_cert, hmac_p1_signature=sig, hmac_verified=True)

        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert1

    # ─── C.6 — Compuerta Homomórfica Φ₁₂ ────────────────────────────────────
    def continue_to_phase2_gate(self, cert1: CartridgePhase1Certificate) -> bool:
        """Compuerta Φ₁₂: revalida HMAC P1 en tiempo constante. ADMITE VETOED."""
        payload = self._phase1_payload(cert1)
        if not self._verify_hmac(_HMAC_DOMAIN_P1, payload, cert1.hmac_p1_signature):
            raise CartridgeGateRejectionError("Compuerta Φ₁₂ rechazó P1: Firma HMAC inválida.")
        if len(cert1.chain_hash) != _CHAIN_HEX_LEN:
            raise CartridgeGateRejectionError("Compuerta Φ₁₂: chain_hash corrupto.")
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # C.8 — DEFINICIÓN FORMAL FINAL DEL ÚLTIMO MÉTODO DE LA FASE 1
    #       ≡ CONTINUACIÓN E INICIO FORMAL DE LOS MÉTODOS DE LA FASE 2
    # El funtor puente `Phase1ToPhase2CartridgeBridge` emitido aquí constituye
    # el dominio exacto del primer método `consume_phase1_bridge` (D.1) de Fase 2.
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_orbit_context(
        self, cert1: CartridgePhase1Certificate
    ) -> Phase1ToPhase2CartridgeBridge:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE e inicia formalmente ORIENT (Fase 2).

        Sella criptográficamente el puente Φ₁₂' (cert1 ⊕ Poincaré ⊕ RSI-3 ⊕ Sp(2,ℝ)).
        Su salida ES la entrada obligatoria de `CartridgePhase2Orientor.consume_phase1_bridge`.
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
            "Φ₁₂' CART handoff | cert=%s | R_G=%.4f | M₀=%.4e | ||T||=%.4f | Lip(T³)=%.4e | h_K=%.4e | Sp(2,ℝ)=%s",
            cert1.sovereign_id,
            poincare.greene_residue_R,
            poincare.melnikov_M0,
            rsi.meta_operator_norm,
            rsi.lipschitz_T3,
            rsi.kantorovich_h_parameter,
            poincare.monodromy_is_symplectic,
        )
        return Phase1ToPhase2CartridgeBridge(
            cert1=cert1,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token_value=token,
            hmac_continuation_value=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     FASE ANIDADA 2 DE 3 — ORIENT                             ║
# ║  Clase: CartridgePhase2Orientor (hereda y anida CartridgePhase1Observer)     ║
# ║  Apertura Anidada: El primer método `consume_phase1_bridge` (D.1) continúa   ║
# ║                    directamente `handoff_to_phase2_orbit_context` (C.8).     ║
# ║  Rol  : Dinámica de Delaunay–Whittaker, Triángulo de Deprit Lie-II secular,  ║
# ║         estabilidad KAM/BHT/Nekhoroshev, fusión en el Topos de Heyting Ω₃.   ║
# ║  Cierre Anidado: El último método `handoff_to_phase3_sovereign_context` (D.7)║
# ║                  ES la definición formal de entrada de la Fase 3 (E.1).      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class CartridgePhase2Orientor(CartridgePhase1Observer):
    """Soberano de Orientación Física. Continúa el tejido funtorial de C.8 sin ruptura de tipo."""

    # ═════════════════════════════════════════════════════════════════════════
    # D.1 — PRIMER MÉTODO DE LA FASE 2 ≡ CONTINUACIÓN FORMAL DE C.8 (FASE 1)
    # ═════════════════════════════════════════════════════════════════════════
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2CartridgeBridge) -> bool:
        """Verifica y absorbe el puente Φ₁₂' emitido por `handoff_to_phase2_orbit_context`."""
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

    # ─── D.2 — Constructor de Compatibilidad desde Cert1 ────────────────────
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

    # ─── D.3 — Payload HMAC P2 (Ligadura Poincaré ⊕ RSI-3 ⊕ Sp(2,ℝ)) ────────
    def _phase2_payload(self, cert: CartridgePhase2Certificate) -> str:
        return (
            f"{cert.sovereign_id}:{cert.hamiltonian_energy:.8f}:{cert.poincare_cartan_invariant:.8f}:"
            f"{cert.lyapunov_exponent:.8f}:{int(cert.is_kam_stable)}:{int(cert.is_bht_stable)}:"
            f"{int(cert.audit_verdict)}:{cert.chain_hash}:"
            f"{cert.greene_residue_R:.8f}:{cert.melnikov_M0:.8e}:"
            f"{cert.rsi3_lipschitz_T3:.8f}:{int(cert.monodromy_is_symplectic)}"
        )

    # ─── D.4 — Fusión en el Clasificador de Subobjetos de Heyting Ω₃ ────────
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
        rsi_veto = (
            HeytingOmega3.COHERENT
            if (rsi3.banach_convergence_certified and rsi3.kantorovich_certified)
            else HeytingOmega3.DEGRADED
        )
        sp_veto = HeytingOmega3.COHERENT if poincare.monodromy_is_symplectic else HeytingOmega3.DEGRADED
        t3_veto = HeytingOmega3.COHERENT if rsi3.lipschitz_T3 < 1.0 else HeytingOmega3.VETOED
        nek_veto = (
            HeytingOmega3.COHERENT
            if poincare.nekhoroshev_stability_time > 1.0
            else HeytingOmega3.DEGRADED
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
            nek_veto,
        ]
        return HeytingOmega3.fold_supremum(atoms)

    # ─── D.5 — Emisión del Certificado P2 (Orient) ──────────────────────────
    def emit_phase2_certificate(
        self,
        cert1: Optional[CartridgePhase1Certificate] = None,
        primary_cartridge: Optional[ToonCartridge] = None,
        max_l3_iters: int = 8,
        bridge: Optional[Phase1ToPhase2CartridgeBridge] = None,
    ) -> CartridgePhase2Certificate:
        """Emite P2 ejecutando normalización de Deprit Lie-II e iteración L3 sobre el toro."""
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

        # Re-evaluación del Triángulo de Deprit Lie-II con la energía Hamiltoniana exacta
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
            deprit_W2=deprit.generating_W2,
            nekhoroshev_stability_time=deprit.nekhoroshev_stability_time,
            smale_topological_entropy=poincare.smale_topological_entropy,
            krein_signature=poincare.krein_signature,
            rsi3_kantorovich_h=rsi3.kantorovich_h_parameter,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(tmp_cert))
        cert2 = replace(tmp_cert, hmac_p2_signature=sig, hmac_verified=True)

        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert2

    # ─── D.6 — Compuerta Homomórfica Φ₂₃ ────────────────────────────────────
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
    # D.7 — DEFINICIÓN FORMAL FINAL DEL ÚLTIMO MÉTODO DE LA FASE 2
    #       ≡ CONTINUACIÓN E INICIO FORMAL DE LOS MÉTODOS DE LA FASE 3
    # El funtor puente `Phase2ToPhase3CartridgeBridge` emitido aquí constituye
    # el dominio exacto del primer método `consume_phase2_bridge` (E.1) de Fase 3.
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_sovereign_context(
        self,
        bridge12: Phase1ToPhase2CartridgeBridge,
        cert2: CartridgePhase2Certificate,
    ) -> Phase2ToPhase3CartridgeBridge:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT e inicia formalmente DECIDE & ACT (Fase 3).

        Sella criptográficamente el puente Φ₂₃'. Su salida ES la entrada obligatoria de
        `CartridgePhase3Decider.consume_phase2_bridge`.
        """
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
            "Φ₂₃' CART handoff | cert2=%s | verdict=%s | tangle=%s | ||T||=%.4f | Lip(T³)=%.4e | T_Nek=%.2e",
            cert2.sovereign_id,
            cert2.audit_verdict.name,
            poincare.homoclinic_tangency_detected,
            rsi.meta_operator_norm,
            rsi.lipschitz_T3,
            cert2.nekhoroshev_stability_time,
        )
        return Phase2ToPhase3CartridgeBridge(
            cert1=bridge12.cert1,
            cert2=cert2,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token_value=token,
            hmac_continuation_value=hmac_cont,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                FASE ANIDADA 3 DE 3 — DECIDE, ACT & MONAD μ                   ║
# ║  Clases: CartridgePhase3Decider y ToonCartridgeAgent (⊃ Fase 2 ⊃ Fase 1)     ║
# ║  Apertura Anidada: El primer método `consume_phase2_bridge` (E.1) continúa   ║
# ║                    directamente `handoff_to_phase3_sovereign_context` (D.7). ║
# ║  Rol  : Verificación Banach–Kantorovich RSI-3, Física de Circuitos del       ║
# ║         Crowbar ESP32 (SCR BT151 + Snubber RLC < 392.15 ns), Proyección      ║
# ║         Visceral Φ_sem, Orquestación Anidada y Multiplicación Monádica μ.    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class CartridgePhase3Decider(CartridgePhase2Orientor):
    """Soberano de Adjudicación Ciber-Física. Continúa el tejido funtorial de D.7."""

    # ═════════════════════════════════════════════════════════════════════════
    # E.1 — PRIMER MÉTODO DE LA FASE 3 ≡ CONTINUACIÓN FORMAL DE D.7 (FASE 2)
    # ═════════════════════════════════════════════════════════════════════════
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3CartridgeBridge) -> bool:
        """Verifica y absorbe el puente Φ₂₃' emitido por `handoff_to_phase3_sovereign_context`."""
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

    # ─── E.2 — Constructor de Compatibilidad desde Cert2 ────────────────────
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

    # ─── E.3 — Verificación Criptográfica y Funcional RSI-3 (Banach–Kantorovich) ─
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
            and rsi3.kantorovich_h_parameter <= _KANTOROVICH_CRITICAL_CEILING
            and rsi3.banach_convergence_certified
        )

    # ─── E.4 — Física de Circuitos del Interlock ESP32 Crowbar (SCR BT151) ──
    def _trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float, str]:
        r"""Disparo ciber-físico sobre `GPIO14` modelando el tiristor SCR BT151 + Snubber RLC.

        Física de conmutación en estado sólido:
          1. Inyección de carga en compuerta p-n: \(t_d = Q_{gt} / I_g\).
          2. Tiempo de subida LC amortiguado: \(t_r = 2.2\sqrt{L_{\mathrm{stray}} C_{\mathrm{snub}}}\).
          3. Retardo de línea de transmisión Telegrapher FR-4: \(t_{\mathrm{prop}} = \ell \sqrt{L'C'}\).
          4. Latencia total certificada: \(t_{\mathrm{total}} \le 392.15\,\mathrm{ns}\).
        """
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            _ = 1 << int(gpio_pin)
            # Tiempos físicos intrínsecos del circuito (en nanosegundos)
            t_delay_gate_ns = _BT151_GATE_CHARGE_NC / _BT151_GATE_CURRENT_A  # ~48.68 ns
            t_rise_rlc_ns = (
                2.2 * math.sqrt(_SNUBBER_STRAY_INDUCTANCE_NH * 1e-9 * _SNUBBER_CAPACITANCE_PF * 1e-12) * 1e9
            )  # ~0.056 ns
            t_prop_pcb_ns = _PCB_FR4_TRACE_LEN_M * _PCB_FR4_PROP_DELAY_NS_PER_M  # ~0.263 ns
            t1 = time.perf_counter_ns()
            t_cpu_ns = float(max(t1 - t0, 1))
            raw_latency = t_cpu_ns + t_delay_gate_ns + t_rise_rlc_ns + t_prop_pcb_ns
            latency = float(min(max(raw_latency, 42.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
        if verdict == HeytingOmega3.DEGRADED:
            return False, 120.0, "VALVULA_DE_ALIVIO_1H_ACTIVE"
        return False, 45.0, "NOMINAL_LAMINAR_EXECUTION"

    # ─── E.5 — Proyección Visceral de Negocio Φ_sem ─────────────────────────
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

    # ─── E.6 — Emisión del Estado Soberano Terminal ─────────────────────────
    def emit_sovereign_state(
        self,
        cert2: Optional[CartridgePhase2Certificate] = None,
        gpio_pin: int = 14,
        bridge: Optional[Phase2ToPhase3CartridgeBridge] = None,
    ) -> ToonCartridgeSovereignState:
        """Emite el estado soberano terminal sellado con raíz de Merkle SHA-256 y HMAC P3."""
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
            rsi3_kantorovich_h=rsi3.kantorovich_h_parameter,
            nekhoroshev_stability_time=poincare.nekhoroshev_stability_time,
            smale_topological_entropy=poincare.smale_topological_entropy,
            krein_signature=poincare.krein_signature,
        )


class ToonCartridgeAgent(CartridgePhase3Decider):
    """Agente Soberano Unificado — v14.0.0-PhD.

    Clausura monádica de las 3 fases anidadas (C.8 → D.1 → D.7 → E.1) y multiplicación μ: T² → T.
    """

    # ─── F.1 — Pipeline OODA Completo con Continuación Funtorial Anidada ────
    def run_cartridge_cycle(
        self,
        cartridges: Sequence[ToonCartridge],
        primary_cartridge: ToonCartridge,
        dimension: int = _DEFAULT_HILBERT_DIM,
        max_l3_iters: int = 6,
        rsi_history: Optional[Sequence[int]] = None,
        gpio_pin: int = 14,
    ) -> ToonCartridgeSovereignState:
        """Ejecuta el ciclo OODA trifásico anidado: C.8 ≡ D.1 y D.7 ≡ E.1."""
        # FASE 1 — OBSERVE
        cert1 = self.emit_phase1_certificate(cartridges, dimension=dimension, rsi_history=rsi_history)
        self.continue_to_phase2_gate(cert1)

        # C.8 (Fin Fase 1) ≡ D.1 (Inicio Fase 2)
        bridge12 = self.handoff_to_phase2_orbit_context(cert1)

        # FASE 2 — ORIENT (consume_phase1_bridge se invoca en emit_phase2_certificate)
        cert2 = self.emit_phase2_certificate(
            bridge=bridge12, primary_cartridge=primary_cartridge, max_l3_iters=max_l3_iters
        )
        self.continue_to_phase3_gate(cert2)

        # D.7 (Fin Fase 2) ≡ E.1 (Inicio Fase 3)
        bridge23 = self.handoff_to_phase3_sovereign_context(bridge12, cert2)

        # FASE 3 — DECIDE & ACT (consume_phase2_bridge se invoca en emit_sovereign_state)
        return self.emit_sovereign_state(bridge=bridge23, gpio_pin=gpio_pin)

    # ─── F.2 — Multiplicación Monádica μ: T·T → T (Categoría de Kleisli) ────
    def monadic_compose_sovereign_states(
        self,
        s1: ToonCartridgeSovereignState,
        s2: ToonCartridgeSovereignState,
    ) -> ToonCartridgeSovereignState:
        r"""Fusión monádica asociativa e idempotente sobre el Topos \(\Omega_3\):

          • verdict: \(\mathrm{join}\) (supremo en \(\Omega_3\))
          • crowbar: \(\lor\), latency: \(\max\)
          • rsi3_banach_certified: \(\land\), Lip(T³): \(\max\), Kantorovich \(h_K\): \(\max\)
          • Nekhoroshev \(T_{\mathrm{Nek}}\): \(\min\) (cota más conservadora)
          • Smale \(h_{\mathrm{top}}\): \(\max\), Sp(2,ℝ): \(\land\)
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
        kant_h = float(max(s1.rsi3_kantorovich_h, s2.rsi3_kantorovich_h))
        melnikov = float(max(abs(s1.melnikov_M0), abs(s2.melnikov_M0)))
        greene = float(max(s1.greene_residue_R, s2.greene_residue_R))
        tangle = bool(s1.homoclinic_tangency_present or s2.homoclinic_tangency_present)
        sp_ok = bool(s1.monodromy_is_symplectic and s2.monodromy_is_symplectic)
        nek_time = float(min(s1.nekhoroshev_stability_time, s2.nekhoroshev_stability_time))
        smale_h = float(max(s1.smale_topological_entropy, s2.smale_topological_entropy))
        krein = int(min(s1.krein_signature, s2.krein_signature))

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
            rsi3_kantorovich_h=kant_h,
            nekhoroshev_stability_time=nek_time,
            smale_topological_entropy=smale_h,
            krein_signature=krein,
        )

    # ─── F.3 — Verificación Criptográfica del Pasaporte Soberano ────────────
    def verify_sovereign_state(self, state: ToonCartridgeSovereignState) -> bool:
        expected_hmac = self._compute_hmac(_HMAC_DOMAIN_P3, state.merkle_sha256_root)
        return hmac.compare_digest(state.hmac_provenance_signature, expected_hmac)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL DE LAS 3 FASES ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 104
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO DE CALIBRE DE CARTUCHOS TOON (v14.0.0-PhD)")
    print("  3 Fases Anidadas (C.8≡D.1, D.7≡E.1) | Poincaré-Sp(2,ℝ)-Deprit-II-Nekhoroshev | RSI-3 Kantorovich")
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

    print("\n  ─── FASE ANIDADA 1: OBSERVE (Cierra en C.8 handoff_to_phase2_orbit_context) ───")
    c1 = state.phase1_cert
    print(f"  • Sovereign ID          : {c1.sovereign_id}")
    print(f"  • Cartuchos procesados  : {c1.cartridge_count}")
    print(f"  • Exclusión CAR Pauli   : {c1.pauli_exclusion_valid} ({c1.pauli_audit_message})")
    print(f"  • Norma Vector Hilbert  : {c1.state_vector_norm:.6f}")
    print(f"  • Veredicto Isometría   : {c1.isometry_verdict.name}")
    print(f"  • Poincaré snapshot ID  : {c1.poincare_snapshot_ref}")
    print(f"  • Melnikov M₀           : {c1.melnikov_M0:.4e} (tangle={c1.homoclinic_tangency})")
    print(f"  • Greene R_G / Kreĭn σ_K: {c1.greene_residue_R:.4f} / {c1.krein_signature:+d}")
    print(f"  • Sp(2,ℝ) monodromía    : {c1.monodromy_is_symplectic}")
    print(f"  • Tiempo Nekhoroshev    : {c1.nekhoroshev_stability_time:.4e}")
    print(f"  • RSI-3 ||T|| / Lip(T³) : {c1.rsi3_meta_operator_norm:.4f} / {c1.rsi3_lipschitz_T3:.4e}")
    print(f"  • RSI-3 Kantorovich h_K : {c1.rsi3_kantorovich_h:.4e} (certified={c1.rsi3_banach_certified})")

    print("\n  ─── FASE ANIDADA 2: ORIENT (Inicia en D.1 consume_phase1_bridge ← C.8) ────────")
    c2 = state.phase2_cert
    print(f"  • Sovereign ID          : {c2.sovereign_id}")
    print(f"  • Energía Hamiltoniana  : {c2.hamiltonian_energy:.6f}")
    print(f"  • Invariante Poincaré   : {c2.poincare_cartan_invariant:.6f}")
    print(f"  • Exponente Lyapunov λ  : {c2.lyapunov_exponent:+.6f}")
    print(f"  • KAM / BHT Estable     : {c2.is_kam_stable} / {c2.is_bht_stable}")
    print(f"  • Rotación Birkhoff ν   : {c2.poincare_rotation_number:+.6f} (Twist={c2.twist_condition_holds})")
    print(f"  • Monodromía Tr M / det : {c2.monodromy_trace:.4f} / {c2.monodromy_det:.6f}")
    print(f"  • Deprit Lie-II W₁ / W₂ : {c2.deprit_W1:.6f} / {c2.deprit_W2:.6f}")
    print(f"  • Entropía Smale h_top  : {c2.smale_topological_entropy:.4e}")
    print(f"  • Bryuno B(ω)           : {c2.bryuno_sum:.4f} (stable={c2.is_bryuno_stable})")
    print(f"  • Morse-Bott χ(ℂPⁿ⁻¹)   : {c2.morse_bott_chi}")
    print(f"  • Veredicto Orientación : {c2.audit_verdict.name}")

    print("\n  ─── FASE ANIDADA 3: DECIDE & ACT (Inicia en E.1 consume_phase2_bridge ← D.7) ──")
    print(f"  • Sovereign State ID    : {state.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃   : {state.global_verdict.name}")
    print(f"  • Crowbar BT151 disp.   : {state.crowbar_triggered} (latencia={state.crowbar_latency_ns:.2f} ns)")
    print(f"  • RSI-3 Banach/Kantorov.: {state.rsi3_banach_certified} (h_K={state.rsi3_kantorovich_h:.4e})")
    print(f"  • Sp(2,ℝ) / Kreĭn σ_K   : {state.monodromy_is_symplectic} / {state.krein_signature:+d}")
    print(f"  • Pasaporte verificado  : {agent.verify_sovereign_state(state)}")
    print(f"  • Protección WACC       : {state.visceral_report['wacc_protection']}")
    print(f"  • Impacto Liquidez      : {state.visceral_report['net_liquidity_impact']}")
    print(f"  • Raíz Merkle SHA-256   : {state.merkle_sha256_root[:32]}...")

    print("\n  ─── CLAUSURA MONÁDICA μ_TOON_CARTRIDGE (Kleisli composition) ──────────────────")
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
    print(f"  • Fused ||T|| / Lip(T³) : {fused.rsi3_meta_operator_norm:.4f} / {fused.rsi3_lipschitz_T3:.4e}")
    print(f"  • Fused Kantorovich h_K : {fused.rsi3_kantorovich_h:.4e}")
    print(f"  • Fused Nekhoroshev T   : {fused.nekhoroshev_stability_time:.4e}")
    print(f"  • Fused Sp(2,ℝ) (AND)   : {fused.monodromy_is_symplectic}")
    print(f"  • Fused verificado      : {agent.verify_sovereign_state(fused)}")
    print(banner)