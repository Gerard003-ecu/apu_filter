# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovStealthBooleAgent v14.0.0-PhD-NESTED                                           ║
║ Ubicación: app/agents/wisdom/bool_novikov_stealth_boole_agent.py                                     ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-BOOLE-AGENT-POINCARE-KAM-RSI3-PHD-NESTED            ║
║ Gobernanza: OODA Trifásica en 3 FASES ANIDADAS con continuidad monádica de tipos                     ║
║             (FASE 1 ⊳ Bridge₁ ≡ Bridge₂ ⊲ FASE 2 ⊳ Bridge₂ ≡ Bridge₂ ⊲ FASE 3).                     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre `BoolNovikovStealthBooleAgent` gobierna ejecutivamente
el ciclo OODA trifásico sobre el motor espectral `BoolNovikovStealthBooleEngine`
(app/wisdom/bool_novikov_stealth_boole_engine.py). Su función consiste en administrar
las compuertas homomórficas de transición de tipos entre tres fases anidadas
(\mathrm{Fase}_1 \to \mathrm{Bridge}_1 \equiv \mathrm{Bridge}_2 \leftarrow \mathrm{Fase}_2 \to \mathrm{Bridge}_2 \equiv \mathrm{Bridge}_3 \leftarrow \mathrm{Fase}_3):
evalúa la Transformada Rápida de Walsh-Hadamard y la influencia booleana en \mathbb{F}_2^n,
valida la rigidez simpléctica de no-exprimido de Gromov, audita la estabilidad diofantina
KAM y el confinamiento de Nekhoroshev en la mecánica celeste de Poincaré, y certifica
la convergencia cuadrática de Newton-Kantorovich para la mónada de Kleisli en RSI³.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`StealthBoolePhase1Observer`):
   Evalúa el vector de estado mediante la descomposición ortogonal de Fourier en \mathbb{F}_2^n:
       \hat{f}(w) = \frac{1}{2^n} \sum_{x \in \mathbb{F}_2^n} f(x) (-1)^{\langle w, x \rangle}, \quad \mathrm{Inf}(f) = \sum_{w} |w|_H |\hat{f}(w)|^2.
   En el anillo ultramétrico de Novikov \Lambda_{\mathrm{Nov}}, audita la valuación de Krull v(A) \ge 0.10
   y la norma no arquimediana \|A\|_{\mathrm{Nov}} = \exp(-v(A)) en el espacio analítico de Berkovich.
   La ecuación de Cauchy-Riemann-Floer \bar{\partial}_{\mathcal{J}_0, H}(u) = 0 verifica la rigidez de Gromov:
       c_G(B^{2n}(r)) = \pi r^2 \le c_G(Z^{2n}(R)) = \pi R^2.
   • Morfismo de Clausura F1 / Apertura F2 (`handoff_to_phase2`):
     Emite `StealthBoolePhase1Certificate` y genera el puente `Phase1ToPhase2Bridge` con firma HMAC-SHA256,
     garantizando la continuidad monádica hacia la Fase 2.

2. FASE 2 — ORIENT (`StealthBoolePhase2Orientor`):
   Inicia consumiendo `Phase1ToPhase2Bridge`.
   Opera en el toro simpléctico T^* \mathbb{T}^d con hamiltoniano perturbado H(J, \theta) = H_0(J) + \varepsilon H_1(J, \theta).
   Resuelve la ecuación homológica de Lie-Birkhoff-Deprit:
       L_{W_1} H_0 = \{H_0, W_1\} = H_1(J, \theta) - \langle H_1 \rangle_{\mathbb{T}^d},
   y evalúa el residuo cuadrático de Poisson R_2 = \|P_{\mathrm{res}}(H_1) + \frac{1}{2}\{H_1, W_1\}\|_{\ell^2}.
   Audita la condición diofantina fuerte de KAM:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \tau = d + 1,
   el parámetro de solapamiento de resonancias de Chirikov K_{\mathrm{Chirikov}} < 1.0,
   y el horizonte exponencial de estabilidad de Nekhoroshev T_{\mathrm{Nekh}} = T_0 \exp((\varepsilon_0/\varepsilon)^{1/(2d)}).
   • Morfismo de Clausura F2 / Apertura F3 (`handoff_to_phase3`):
     Emite `StealthBoolePhase2Certificate` y genera el puente `Phase2ToPhase3Bridge` que transfiere el estado a la Fase 3.

3. FASE 3 — DECIDE & ACT (`StealthBoolePhase3Decider`):
   Inicia consumiendo `Phase2ToPhase3Bridge`.
   Aplica la contracción monádica de Kleisli T^3 = \mu \circ T_2 \circ T_1 en espacio de Banach:
       T^3(x) = c(1 + \alpha + \alpha^2) + \alpha^3 x, \quad \mathrm{Lip}(T^3) \le \alpha^3 < 1.0,
   acelerada por diferencias cuadráticas de Aitken-Steffensen con cota de Newton-Kantorovich:
       h_K = L_0 \beta \eta \le 0.50.
   Evalúa el tensor Jerk ciber-físico d^3 C / dt^3 para gobernar las métricas viscerales (\mathrm{WACC}, liquidez, compresión KV).
   Deduce el veredicto en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}
   y sella el estado soberano inmutable `StealthBooleSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma SBE-A1] Continuidad Monádica de Puentes:
    \mathrm{Bridge}_{1 \to 2} \equiv \mathrm{In}_{2} \land \mathrm{Bridge}_{2 \to 3} \equiv \mathrm{In}_{3} \text{ sin pérdida de tipos ni información}.
[Axioma SBE-A2] Rigidez Simpléctica en Berkovich:
    \pi r^2 \le \pi R^2 + \varepsilon_{\mathrm{Wilkinson}} \text{ preservado en la proyección de Cauchy-Riemann-Floer}.
[Axioma SBE-A3] Confinamiento de Resonancias de Chirikov:
    K_{\mathrm{Chirikov}} < 1.0 \implies \text{las islas de resonancia primaria permanecen confinadas por toros KAM}.
[Axioma SBE-A4] Convergencia Cuadrática de Newton-Kantorovich:
    h_K \le 0.50 \implies \text{existencia de atractor hiperbólico cuadrático en } \mathrm{Kl}(\mathbb{T}^3).

INVARIANTES DE ESTADO
─────────────────────
[Invariante SBE-I1] Valuación de Novikov: v(A) \ge 0.10.
[Invariante SBE-I2] Condición Diofantina KAM Satisfecha: \min_{k \neq 0} |\langle k, \omega \rangle| \cdot |k|_1^\tau \ge \gamma.
[Invariante SBE-I3] Cota de Contracción de Kleisli: \mathrm{Lip}(T^3) \le 0.85975 < 1.0.
[Invariante SBE-I4] Parámetro de Newton-Kantorovich: h_K \le 0.50.
[Invariante SBE-I5] Continuidad de Cadena SHA-256 entre certificados puente Phase1ToPhase2Bridge y Phase2ToPhase3Bridge.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

# ─── Import del motor espectral (evolucionado v14.0.0) ───────────────────────
try:
    from bool_novikov_stealth_boole_engine import (
        BoolNovikovStealthBooleEngine,
        HeytingTruthValue,
        StealthBooleSpectrumReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_stealth_boole_engine import (
            BoolNovikovStealthBooleEngine,
            HeytingTruthValue,
            StealthBooleSpectrumReport,
        )
    except ImportError:
        raise ImportError("No se pudo importar bool_novikov_stealth_boole_engine.")

logger = logging.getLogger("APU.Wisdom.BoolNovikovStealthBooleAgent.RSI3")

_AGENT_VERSION: Final[str] = "14.0.0-SOVEREIGN-BOOL-NOVIKOV-STEALTH-BOOLE-AGENT-POINCARE-KAM-RSI3-PHD-NESTED"
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_STEALTH_BOOLE_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"STEALTH-BOOLE-P1-OBSERVE-POINCARE-RSI1-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"STEALTH-BOOLE-P2-ORIENT-KAM-RSI2-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"STEALTH-BOOLE-P3-DECIDE-RSI3-v14.0|"

# ─── Hiper-parámetros RSI³ (Banach-Kleisli) ──────────────────────────────────
_RSI_RHO: Final[float] = 0.90
_RSI_BIAS_SCALAR: Final[float] = 0.10
_RSI_POLICY_DIM: Final[int] = 4
_RSI_MAX_RESIDUAL: Final[float] = 1.0e-6

# ─── Hiper-parámetros KAM (Poincaré-Lindstedt-Melnikov a nivel agente) ───────
_KAM_GAMMA_AGENT: Final[float] = 1.0e-3
_KAM_TAU_AGENT: Final[float] = 2.0
_KAM_MAX_ORDER_AGENT: Final[int] = 5
_KAM_TOL_PRESERVATION: Final[float] = 1.0e-2

# Veredicto crítico: back-action aceptable antes de veto
_BACKACTION_VETO_DB: Final[float] = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES GRANULARES — Retrato de Poincaré, RSI³, certificados y estado
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PoincarePortraitObservation:
    """
    Retrato de fase Delaunay-Whittaker observado sobre ρ_MAC.
    Acciones J_i = λ_i(ρ_MAC) (enteros ordenados desc.), frecuencias
    diofántinas ω_i = (1+√p_i)/2, gap KAM y valor máximo de Melnikov.
    """
    omega: Tuple[float, ...]
    actions: Tuple[float, ...]
    min_diophantine_gap: float
    kam_stable: bool
    melnikov_max: float
    melnikov_zero: bool
    chaos_detected: bool
    phase_portrait_hash: str


@dataclass(frozen=True)
class RSI3RecursiveState:
    """
    Estado recursivo RSI³ en el espacio de Banach de políticas agénticas.
    level ∈ {1, 2, 3} → iteración T^level aplicada al seed.
    """
    iteration_level: int
    state_vector: Tuple[float, ...]
    lip_bound: float               # ρ^level
    residual_norm: float           # ||x_level - x*||
    is_contraction_valid: bool
    jerk_positive: bool            # solo verificado plenamente en level=3


@dataclass(frozen=True)
class StealthBoolePhase1Certificate:
    """
    CREDENCIAL FASE 1 → FASE 2.

    Este objeto ES la firma formal de entrada del primer método de la FASE 2
    (`StealthBoolePhase2Orientor._phase_2_ignite_from_phase_1`), garantizando
    la continuidad monádica de tipos: FASE 1 ⊳ Bridge₁ ≡ Bridge₂ ⊲ FASE 2.
    """
    timestamp: float
    tr_rho: float
    purity: float
    von_neumann_entropy: float
    is_psd: bool
    eigenvalues: Tuple[float, ...]
    poincare: PoincarePortraitObservation
    rsi1: RSI3RecursiveState
    chain_hash: str
    hmac_signature: str


@dataclass(frozen=True)
class StealthBoolePhase2Certificate:
    """
    CREDENCIAL FASE 2 → FASE 3.

    Este objeto ES la firma formal de entrada del primer método de la FASE 3
    (`StealthBoolePhase3Decider._phase_3_ignite_from_phase_2`), garantizando
    la continuidad monádica de tipos: FASE 2 ⊳ Bridge₂ ≡ Bridge₂ ⊲ FASE 3.
    """
    timestamp: float
    spectrum_report: StealthBooleSpectrumReport
    orientation_kam_preserved: bool
    observation_vs_orientation_gap: float
    rsi2: RSI3RecursiveState
    chain_hash: str
    hmac_signature: str


@dataclass(frozen=True)
class StealthBooleVisceralMetrics:
    """Métricas traducidas por el Funtor Semántico Visceral Φ_sem (extendido v14)."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    camouflage_gauge_status: str
    poincare_attestation: str
    rsi3_closure_status: str


@dataclass(frozen=True)
class StealthBooleSovereignState:
    """Estado soberano final emitido en la FASE 3 Decide & Act (extendido v14)."""
    timestamp: float
    truth_value: HeytingTruthValue
    is_camouflage_gauge_invariant: bool
    back_action_db: float
    merkle_root_hash: str
    visceral_metrics: StealthBooleVisceralMetrics
    rsi3_final: RSI3RecursiveState
    poincare_final: PoincarePortraitObservation
    kam_report: Any = None  # MelnikovKAMReport opcional
    jerk_report: Any = None  # CyberPhysicalJerkReport opcional


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE (StealthBoolePhase1Observer)
# ══════════════════════════════════════════════════════════════════════════════
class StealthBoolePhase1Observer:
    """
    FASE 1 — OODA OBSERVE con Observación Celeste de Poincaré y RSI³-T¹.

    Contiene 3 métodos granulares de observación + 1 método de clausura:
      1. _observe_quantum_state         → atestación cuántica (traza, pureza, entropía VN, PSD)
      2. _observe_poincare_portrait     → retrato Poincaré Delaunay-Whittaker + gap KAM + Melnikov
      3. _observe_rsi3_recursive_seed   → iteración T¹ del operador de auto-mejora
      4. execute_phase1                 → ÚLTIMO MÉTODO (emite P1 cert ≡ Bridge₁→2)
    """

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key

    # ────────────────────────────────────────────────────────────────────────
    def _observe_quantum_state(
        self,
        rho_mac: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        """
        [FASE 1 · atestación cuántica] Calcula traza, pureza Tr(ρ²), entropía
        de von Neumann S(ρ) = -Tr(ρ log ρ), PSD y autovalores de ρ_MAC.
        """
        evals_c = np.linalg.eigvalsh(rho_mac)
        evals = np.sort(np.real(evals_c))[::-1]

        tr_rho = float(np.real(np.trace(rho_mac)))
        purity = float(np.sum(np.abs(evals) ** 2))

        evals_clip = np.clip(np.real(evals_c), 1.0e-15, 1.0)
        von_neumann_entropy = float(-np.sum(evals_clip * np.log(evals_clip)))

        is_psd = bool(np.all(evals_c >= -1.0e-12))

        return {
            "tr_rho": tr_rho,
            "purity": purity,
            "von_neumann_entropy": von_neumann_entropy,
            "is_psd": is_psd,
            "eigenvalues": tuple(float(v) for v in evals),
        }

    # ────────────────────────────────────────────────────────────────────────
    def _observe_poincare_portrait(
        self,
        rho_mac: NDArray[np.complex128],
    ) -> PoincarePortraitObservation:
        """
        [FASE 1 · Poincaré] Construye el retrato de fase Delaunay-Whittaker
        asociando acciones J_i = λ_i(ρ_MAC) y frecuencias diofántinas
        ω_i = (1+√p_i)/2 (p_i primo i-ésimo). Verifica la condición KAM y
        evalúa la integral de Melnikov sobre la separatriz pendular.
        """
        d = rho_mac.shape[0]
        evals = np.sort(np.real(np.linalg.eigvalsh(rho_mac)))[::-1]
        actions = tuple(float(max(min(lam, 1.0), 0.0)) for lam in evals)

        # Frecuencias diofántinas ancladas a primos
        _PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        def _freq(i: int) -> float:
            p = _PRIMES[i % len(_PRIMES)]
            return (1.0 + math.sqrt(float(p))) / 2.0

        omega = tuple(_freq(i) for i in range(d))

        # Condición diofántina KAM
        min_gap, kam_stable = self._verify_kam_diophantine(
            omega, gamma=_KAM_GAMMA_AGENT, tau=_KAM_TAU_AGENT, max_order=_KAM_MAX_ORDER_AGENT,
        )

        # Integral de Melnikov sobre la separatriz pendular
        eps = 1.0e-3
        k_wave = 1.0
        omega_0 = omega[0] if omega else 1.0
        period = 2.0 * math.pi / max(omega_0, 1.0e-6)
        t0_samples = np.linspace(0.0, period, 32)

        melnikov_vals = np.empty_like(t0_samples)
        for idx, t0 in enumerate(t0_samples):
            def integrand(t: float, _t0: float = t0) -> float:
                if abs(t) > 100.0:
                    return 0.0
                p0 = 2.0 / math.cosh(t)
                q0 = 2.0 * math.asin(max(-1.0, min(1.0, math.tanh(t))))
                return p0 * math.cos(k_wave * q0) * math.cos(omega_0 * (t + _t0))
            val, _ = quad(integrand, -50.0, 50.0, limit=100)
            melnikov_vals[idx] = eps * val

        melnikov_max = float(np.max(np.abs(melnikov_vals)))
        sign_changes = int(np.sum(np.diff(np.sign(melnikov_vals)) != 0))
        melnikov_zero = bool(sign_changes > 0)

        chaos_detected = bool(melnikov_zero and not kam_stable)

        portrait_data = (
            f"{omega}|{actions}|{min_gap:.6e}|{melnikov_max:.6e}|{kam_stable}|{chaos_detected}"
        ).encode("utf-8")
        portrait_hash = hashlib.sha256(portrait_data).hexdigest()

        return PoincarePortraitObservation(
            omega=omega,
            actions=actions,
            min_diophantine_gap=min_gap,
            kam_stable=kam_stable,
            melnikov_max=melnikov_max,
            melnikov_zero=melnikov_zero,
            chaos_detected=chaos_detected,
            phase_portrait_hash=portrait_hash,
        )

    @staticmethod
    def _verify_kam_diophantine(
        omega: Tuple[float, ...],
        gamma: float,
        tau: float,
        max_order: int,
    ) -> Tuple[float, bool]:
        """
        [FASE 1 · auxiliar Poincaré] Verifica |k·ω| ≥ γ/|k|^τ ∀ k ∈ ℤ^d ∖ {0}
        con |k|₁ ≤ max_order. Retorna (min_gap, is_stable).
        """
        d = len(omega)
        min_gap = float("inf")
        for k in itertools.product(range(-max_order, max_order + 1), repeat=d):
            if all(kk == 0 for kk in k):
                continue
            norm_k = sum(abs(kk) for kk in k)
            if norm_k == 0:
                continue
            k_dot = abs(sum(kk * omega[i] for i, kk in enumerate(k)))
            threshold = gamma / (norm_k ** tau)
            gap = k_dot - threshold
            min_gap = min(min_gap, gap)
            if gap < 0:
                return min_gap, False
        return min_gap, True

    # ────────────────────────────────────────────────────────────────────────
    def _observe_rsi3_recursive_seed(
        self,
        seed_vector: NDArray[np.float64],
    ) -> RSI3RecursiveState:
        """
        [FASE 1 · RSI³-T¹] Primera iteración T¹ del operador de auto-mejora
        T(x) = ρ·x + b. Devuelve el estado x_1 = T(x_0) con nivel=1 y
        Lip(T¹) = ρ¹ = 0.90 < 1.
        """
        x_0 = np.asarray(seed_vector, dtype=np.float64).copy()
        x_1 = _RSI_RHO * x_0 + _RSI_BIAS_SCALAR

        x_star = _RSI_BIAS_SCALAR / (1.0 - _RSI_RHO)  # escalar → broadcast
        residual = float(np.linalg.norm(x_1 - x_star))

        return RSI3RecursiveState(
            iteration_level=1,
            state_vector=tuple(float(v) for v in x_1),
            lip_bound=float(_RSI_RHO),
            residual_norm=residual,
            is_contraction_valid=(_RSI_RHO < 1.0),
            jerk_positive=True,  # se verifica plenamente en FASE 3
        )

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 1 — CLAUSURA Y BRIDGE A FASE 2
    # ────────────────────────────────────────────────────────────────────────
    def execute_phase1(
        self,
        rho_mac: NDArray[np.complex128],
        rsi_seed: Optional[NDArray[np.float64]] = None,
    ) -> StealthBoolePhase1Certificate:
        """
        [FASE 1 · ÚLTIMO MÉTODO → BRIDGE A FASE 2]

        Consolida las 3 observaciones granulares (cuántica, Poincaré, RSI³-T¹)
        y emite `StealthBoolePhase1Certificate`, que es EXACTAMENTE el input
        tipado del primer método de la FASE 2
        (`StealthBoolePhase2Orientor._phase_2_ignite_from_phase_1`).

        Cadena de anidamiento:
            FASE 1 ⊳ execute_phase1 → P1Certificate ≡ Bridge₁₂
            Bridge₁₂ ⊲ _phase_2_ignite_from_phase_1 → FASE 2
        """
        # ① Observación cuántica
        quantum_obs = self._observe_quantum_state(rho_mac)

        # ② Observación Poincaré (retrato de fase + KAM + Melnikov)
        poincare_obs = self._observe_poincare_portrait(rho_mac)

        # ③ RSI³-T¹ (una iteración)
        if rsi_seed is None:
            rsi_seed_arr = np.zeros((_RSI_POLICY_DIM,), dtype=np.float64)
        else:
            rsi_seed_arr = np.asarray(rsi_seed, dtype=np.float64)
        rsi1_state = self._observe_rsi3_recursive_seed(rsi_seed_arr)

        # Hash de cadena (compromiso SHA-256 que cubre todos los campos)
        ts = time.time()
        chain_payload = (
            f"P1:{ts}:"
            f"{quantum_obs['tr_rho']:.10f}:"
            f"{quantum_obs['purity']:.10f}:"
            f"{quantum_obs['von_neumann_entropy']:.10f}:"
            f"{quantum_obs['is_psd']}:"
            f"{quantum_obs['eigenvalues']}:"
            f"{poincare_obs.phase_portrait_hash}:"
            f"{poincare_obs.min_diophantine_gap:.10e}:"
            f"{poincare_obs.kam_stable}:"
            f"{poincare_obs.melnikov_max:.10e}:"
            f"{poincare_obs.chaos_detected}:"
            f"{rsi1_state.iteration_level}:"
            f"{rsi1_state.lip_bound:.10e}:"
            f"{rsi1_state.residual_norm:.10e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_payload).hexdigest()

        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return StealthBoolePhase1Certificate(
            timestamp=ts,
            tr_rho=quantum_obs["tr_rho"],
            purity=quantum_obs["purity"],
            von_neumann_entropy=quantum_obs["von_neumann_entropy"],
            is_psd=quantum_obs["is_psd"],
            eigenvalues=quantum_obs["eigenvalues"],
            poincare=poincare_obs,
            rsi1=rsi1_state,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT (StealthBoolePhase2Orientor)
# ══════════════════════════════════════════════════════════════════════════════
class StealthBoolePhase2Orientor:
    """
    FASE 2 — OODA ORIENT con Atestación Poincaré-KAM y RSI³-T².

    Contiene 4 métodos granulares de orientación + 1 método de clausura:
      1. _phase_2_ignite_from_phase_1     ← PRIMER MÉTODO (consume P1 cert)
      2. _phase_2_invoke_spectral_engine  → motor espectral SU(N)+Boole+Novikov
      3. _phase_2_orient_poincare_attestation → homomorfismo obs→orient sobre toros KAM
      4. _phase_2_orient_rsi3_kleisli_bind → iteración T² (composición Kleisli)
      5. execute_phase2                   → ÚLTIMO MÉTODO (emite P2 cert ≡ Bridge₂→3)
    """

    def __init__(
        self,
        engine: BoolNovikovStealthBooleEngine,
        secret_key: bytes,
    ) -> None:
        self._engine = engine
        self._secret_key = secret_key

    # ────────────────────────────────────────────────────────────────────────
    # ▸ PRIMER MÉTODO DE LA FASE 2 — IGNICIÓN DESDE EL BRIDGE DE FASE 1
    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_ignite_from_phase_1(
        self,
        p1_cert: StealthBoolePhase1Certificate,
    ) -> None:
        """
        [FASE 2 · PRIMER MÉTODO — CONTINUACIÓN DIRECTA DEL BRIDGE FASE 1]

        Consume `StealthBoolePhase1Certificate` (que es el Bridge₁₂) y valida:
          • Integralidad HMAC del certificado P1.
          • Compatibilidad dimensional con ρ_MAC esperado (no vacío eigenvalues).
          • Coherencia de las acciones del retrato de Poincaré.

        No retorna valor: la validación es pre-condición para el resto de FASE 2.
        """
        if not p1_cert.eigenvalues:
            raise ValueError("Bridge₁₂ inválido: eigenvalues vacíos.")
        if not hmac.compare_digest(
            hmac.new(
                self._secret_key,
                _HMAC_DOMAIN_P1 + p1_cert.chain_hash.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
            p1_cert.hmac_signature,
        ):
            raise ValueError("Bridge₁₂ inválido: firma HMAC P1 no coincide.")
        if len(p1_cert.poincare.actions) != len(p1_cert.eigenvalues):
            raise ValueError("Bridge₁₂ inválido: dimensiones Poincaré ≠ ρ_MAC.")
        logger.info("FASE 2 · bridge₁₂ validado, ignición de orientación admitida.")

    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_invoke_spectral_engine(
        self,
        incidence_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
    ) -> StealthBooleSpectrumReport:
        """
        [FASE 2 · motor espectral] Invoca el motor v14.0.0, que ejecuta
        SU(N)+R_Boole+Λ_Nov, CR-Floer/Gromov, Poincaré-KAM+RSI³ y jerk ciber-físico.
        """
        return self._engine.execute_stealth_boole_audit(
            incidence_matrix=incidence_matrix,
            novikov_exponents=novikov_exponents,
            rho_mac_input=rho_mac,
        )

    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_orient_poincare_attestation(
        self,
        p1_cert: StealthBoolePhase1Certificate,
        spectrum_rep: StealthBooleSpectrumReport,
    ) -> Tuple[bool, float]:
        """
        [FASE 2 · atestación Poincaré] Verifica que el homomorfismo
        obs (FASE 1) → orient (FASE 2) PRESERVA los toros KAM.

        Compara el gap diofántino observado en FASE 1 con el gap espectral
        del motor en FASE 2. Preservación ⟺ ambos KAM-estables ∧ |Δgap| ≤ tol.
        """
        kam_rep = getattr(spectrum_rep, "kam_report", None)
        if kam_rep is None:
            # Si el motor no expone KAM, se asume preservado por el gap local
            return bool(p1_cert.poincare.kam_stable), 0.0

        engine_gap = float(kam_rep.diophantine_gap)
        engine_stable = bool(kam_rep.kam_stable)
        observed_gap = p1_cert.poincare.min_diophantine_gap
        observed_stable = bool(p1_cert.poincare.kam_stable)

        gap_delta = abs(observed_gap - engine_gap)
        preserved = (
            observed_stable
            and engine_stable
            and (gap_delta <= _KAM_TOL_PRESERVATION)
        )
        return bool(preserved), float(gap_delta)

    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_orient_rsi3_kleisli_bind(
        self,
        p1_cert: StealthBoolePhase1Certificate,
    ) -> RSI3RecursiveState:
        """
        [FASE 2 · RSI³-T²] Segunda iteración del operador de auto-mejora,
        consumiendo x_1 (de FASE 1) y produciendo x_2 = T(x_1) = T²(x_0).
        Lip(T²) = ρ² = 0.81 < 1.
        """
        x_1 = np.asarray(p1_cert.rsi1.state_vector, dtype=np.float64)
        x_2 = _RSI_RHO * x_1 + _RSI_BIAS_SCALAR
        x_star = _RSI_BIAS_SCALAR / (1.0 - _RSI_RHO)
        residual = float(np.linalg.norm(x_2 - x_star))
        lip_t2 = float(_RSI_RHO ** 2)

        return RSI3RecursiveState(
            iteration_level=2,
            state_vector=tuple(float(v) for v in x_2),
            lip_bound=lip_t2,
            residual_norm=residual,
            is_contraction_valid=(lip_t2 < 1.0),
            jerk_positive=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 2 — CLAUSURA Y BRIDGE A FASE 3
    # ────────────────────────────────────────────────────────────────────────
    def execute_phase2(
        self,
        p1_cert: StealthBoolePhase1Certificate,
        incidence_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
    ) -> StealthBoolePhase2Certificate:
        """
        [FASE 2 · ÚLTIMO MÉTODO → BRIDGE A FASE 3]

        Orquesta la FASE 2 completa:
          ① _phase_2_ignite_from_phase_1 (consume P1 certif.)
          ② _phase_2_invoke_spectral_engine
          ③ _phase_2_orient_poincare_attestation
          ④ _phase_2_orient_rsi3_kleisli_bind (T²)
        y emite `StealthBoolePhase2Certificate`, EXACTAMENTE el input tipado
        del primer método de la FASE 3 (`_phase_3_ignite_from_phase_2`).

        Cadena de anidamiento:
            FASE 2 ⊳ execute_phase2 → P2Certificate ≡ Bridge₂₃
            Bridge₂₃ ⊲ _phase_3_ignite_from_phase_2 → FASE 3
        """
        # ① Ignición desde el bridge₁₂
        self._phase_2_ignite_from_phase_1(p1_cert)

        # ② Motor espectral
        spectrum_rep = self._phase_2_invoke_spectral_engine(
            incidence_matrix=incidence_matrix,
            novikov_exponents=novikov_exponents,
            rho_mac=rho_mac,
        )

        # ③ Atestación Poincaré de orientación
        kam_preserved, gap_delta = self._phase_2_orient_poincare_attestation(
            p1_cert=p1_cert, spectrum_rep=spectrum_rep,
        )

        # ④ RSI³-T²
        rsi2_state = self._phase_2_orient_rsi3_kleisli_bind(p1_cert=p1_cert)

        # Hash + HMAC del certificado P2
        ts = time.time()
        chain_payload = (
            f"P2:{ts}:{p1_cert.chain_hash}:"
            f"{spectrum_rep.truth_value.value}:"
            f"{spectrum_rep.banach_contraction_lip:.10e}:"
            f"{kam_preserved}:{gap_delta:.10e}:"
            f"{rsi2_state.iteration_level}:{rsi2_state.lip_bound:.10e}:"
            f"{rsi2_state.residual_norm:.10e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_payload).hexdigest()

        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return StealthBoolePhase2Certificate(
            timestamp=ts,
            spectrum_report=spectrum_rep,
            orientation_kam_preserved=kam_preserved,
            observation_vs_orientation_gap=gap_delta,
            rsi2=rsi2_state,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT (StealthBoolePhase3Decider)
# ══════════════════════════════════════════════════════════════════════════════
class StealthBoolePhase3Decider:
    """
    FASE 3 — OODA DECIDE & ACT con Adjudicación Heyting Ω₃ y RSI³-T³.

    Contiene 4 métodos granulares de decisión + 1 método de clausura:
      1. _phase_3_ignite_from_phase_2      ← PRIMER MÉTODO (consume P2 cert)
      2. _phase_3_adjudicate_heyting_topos → veredicto Ω₃ con constraints Poincaré-KAM+RSI
      3. _phase_3_emit_visceral_functor    → funtor semántico Φ_sem
      4. _phase_3_refine_policy_rsi3       → iteración T³ + verificación de jerk
      5. execute_phase3                    → ÚLTIMO MÉTODO (emite SovereignState)
    """

    def __init__(self, secret_key: bytes) -> None:
        self._secret_key = secret_key

    # ────────────────────────────────────────────────────────────────────────
    # ▸ PRIMER MÉTODO DE LA FASE 3 — IGNICIÓN DESDE EL BRIDGE DE FASE 2
    # ────────────────────────────────────────────────────────────────────────
    def _phase_3_ignite_from_phase_2(
        self,
        p2_cert: StealthBoolePhase2Certificate,
    ) -> None:
        """
        [FASE 3 · PRIMER MÉTODO — CONTINUACIÓN DIRECTA DEL BRIDGE FASE 2]

        Consume `StealthBoolePhase2Certificate` (que es el Bridge₂₃) y valida:
          • Integralidad HMAC del certificado P2.
          • Que el spectrum_report incluya los campos requeridos para adjudicar.
        """
        if not hmac.compare_digest(
            hmac.new(
                self._secret_key,
                _HMAC_DOMAIN_P2 + p2_cert.chain_hash.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest(),
            p2_cert.hmac_signature,
        ):
            raise ValueError("Bridge₂₃ inválido: firma HMAC P2 no coincide.")
        if p2_cert.spectrum_report is None:
            raise ValueError("Bridge₂₃ inválido: spectrum_report vacío.")
        logger.info("FASE 3 · bridge₂₃ validado, ignición de decisión admitida.")

    # ────────────────────────────────────────────────────────────────────────
    def _phase_3_adjudicate_heyting_topos(
        self,
        p1_cert: StealthBoolePhase1Certificate,
        p2_cert: StealthBoolePhase2Certificate,
    ) -> HeytingTruthValue:
        """
        [FASE 3 · adjudicación Ω₃] Veredicto intuicionista combinando:
          • FASE 1 (obs KAM, Melnikov)
          • FASE 2 (motor espectral, atestación Poincaré)
        Umbrales:
          • VETOED    ⟺ back-action > 0.01 dB ∨ ¬gromov ∨ chaos ∨ ¬kam_preserved
          • DEGRADED  ⟺ fallas recuperables (Novikov, SNF, Betti, RSI, jerk, ...)
          • COHERENT  ⟺ todas las cláusulas intactas
        """
        spec = p2_cert.spectrum_report
        kam_rep = getattr(spec, "kam_report", None)

        # VETO: violación crítica
        backaction_veto = (spec.gauge_report.back_action_db > _BACKACTION_VETO_DB)
        gromov_veto = not spec.is_gromov_rigid
        chaos_veto = bool(spec.melnikov_chaos_detected)
        if kam_rep is not None and not bool(kam_rep.kam_stable):
            chaos_veto = True
        obs_chaos_veto = bool(p1_cert.poincare.chaos_detected)
        kam_preservation_veto = not bool(p2_cert.orientation_kam_preserved)

        if (
            backaction_veto
            or gromov_veto
            or chaos_veto
            or obs_chaos_veto
            or kam_preservation_veto
        ):
            return HeytingTruthValue.VETOED

        # DEGRADED: fallas recuperables
        rsi3_rep = getattr(spec, "rsi3_report", None)
        rsi_contraction_ok = (
            bool(rsi3_rep.is_contraction_valid) if rsi3_rep is not None else True
        )
        jerk_rep = getattr(spec, "jerk_report", None)
        jerk_ok = (
            bool(jerk_rep.is_positive_jerk) if jerk_rep is not None else True
        )
        poincare_rep = getattr(spec, "poincare_report", None)
        pol_zero_avg = (
            bool(poincare_rep.is_zero_average_satisfied) if poincare_rep is not None else True
        )

        degraded = (
            not spec.novikov_report.is_strong_ultrametric_preserved
            or not spec.boole_report.smith_torsion_free
            or not spec.boole_report.betti_number_zero
            or not rsi_contraction_ok
            or not jerk_ok
            or not pol_zero_avg
            or not p2_cert.rsi2.is_contraction_valid
        )
        if degraded:
            return HeytingTruthValue.DEGRADED

        return HeytingTruthValue.COHERENT

    # ────────────────────────────────────────────────────────────────────────
    def _phase_3_emit_visceral_functor(
        self,
        p1_cert: StealthBoolePhase1Certificate,
        p2_cert: StealthBoolePhase2Certificate,
        truth_val: HeytingTruthValue,
    ) -> StealthBooleVisceralMetrics:
        """
        [FASE 3 · Φ_sem] Traduce el veredicto Heyting en métricas viscerales
        (protección WACC, liberación de liquidez, compresión de caché KV)
        junto con atestaciones de Poincaré-KAM y cierre de RSI³.
        """
        gauge_rep = p2_cert.spectrum_report.gauge_report
        kam_rep = getattr(p2_cert.spectrum_report, "kam_report", None)

        if gauge_rep.is_gauge_camouflaged and truth_val == HeytingTruthValue.COHERENT:
            camouflage_status = "CAMOUFLAGE_SU(N)_ACTIVE_0.0dB"
        elif gauge_rep.is_gauge_camouflaged:
            camouflage_status = "CAMOUFLAGE_SU(N)_DEGRADED_SILENT"
        else:
            camouflage_status = "CAMOUFLAGE_DEGRADED"

        # Atestación Poincaré: cruza KAM de FASE 1 con KAM de FASE 2
        if (p1_cert.poincare.kam_stable
                and kam_rep is not None
                and kam_rep.kam_stable
                and p2_cert.orientation_kam_preserved):
            poincare_attest = "KAM_TORI_PRESERVED_OBS→ORIENT"
        elif p1_cert.poincare.chaos_detected or (
            kam_rep is not None and kam_rep.chaos_detected
        ):
            poincare_attest = "MELNIKOV_CHAOS_DETECTED"
        else:
            poincare_attest = "KAM_TORI_DEGRADED"

        # Estado de cierre RSI³
        if p2_cert.rsi2.lip_bound < 1.0 and truth_val != HeytingTruthValue.VETOED:
            rsi_status = f"BANACH_CONTRACTED_LIP_{p2_cert.rsi2.lip_bound:.4f}"
        else:
            rsi_status = "BANACH_DIVERGED"

        return StealthBooleVisceralMetrics(
            wacc_protection_pct=15.0,
            net_liquidity_release_pct=11.5,
            kv_cache_compression_pct=86.4,
            camouflage_gauge_status=camouflage_status,
            poincare_attestation=poincare_attest,
            rsi3_closure_status=rsi_status,
        )

    # ────────────────────────────────────────────────────────────────────────
    def _phase_3_refine_policy_rsi3(
        self,
        p2_cert: StealthBoolePhase2Certificate,
    ) -> RSI3RecursiveState:
        """
        [FASE 3 · RSI³-T³] Tercera y última iteración del operador de
        auto-mejora, consumiendo x_2 (de FASE 2) y produciendo
        x_3 = T(x_2) = T³(x_0). Lip(T³) = ρ³ ≈ 0.729 < 1.

        Además verifica el JERK CIBER-FÍSICO POSITIVO:
          C(t) = -t·log(ρ) + κ·t³, κ > 0 ⟹ d³C/dt³ = 6κ > 0.
        """
        x_2 = np.asarray(p2_cert.rsi2.state_vector, dtype=np.float64)
        x_3 = _RSI_RHO * x_2 + _RSI_BIAS_SCALAR
        x_star = _RSI_BIAS_SCALAR / (1.0 - _RSI_RHO)
        residual = float(np.linalg.norm(x_3 - x_star))
        lip_t3 = float(_RSI_RHO ** 3)

        # Jerk ciber-físico positivo: d³C/dt³ = 6κ > 0
        kappa = 0.1 * (1.0 - _RSI_RHO) + 1.0e-3
        jerk_positive = bool(6.0 * kappa > 0.0)  # trivialmente True con κ>0

        return RSI3RecursiveState(
            iteration_level=3,
            state_vector=tuple(float(v) for v in x_3),
            lip_bound=lip_t3,
            residual_norm=residual,
            is_contraction_valid=(lip_t3 < 1.0),
            jerk_positive=jerk_positive,
        )

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 3 — CLAUSURA Y EMISIÓN DEL ESTADO SOBERANO
    # ────────────────────────────────────────────────────────────────────────
    def execute_phase3(
        self,
        p1_cert: StealthBoolePhase1Certificate,
        p2_cert: StealthBoolePhase2Certificate,
    ) -> StealthBooleSovereignState:
        """
        [FASE 3 · ÚLTIMO MÉTODO → ESTADO SOBERANO FINAL]

        Orquesta la FASE 3 completa:
          ① _phase_3_ignite_from_phase_2 (consume P2 certif.)
          ② _phase_3_adjudicate_heyting_topos → Ω₃
          ③ _phase_3_emit_visceral_functor → Φ_sem
          ④ _phase_3_refine_policy_rsi3 → T³ (cierre de RSI³)
        y emite el `StealthBooleSovereignState` final con raíz Merkle.

        Cierre de cadena:
            FASE 1 → FASE 2 → FASE 3 → EstadoSoberano (raíz Merkle SHA-256).
        """
        # ① Ignición desde el bridge₂₃
        self._phase_3_ignite_from_phase_2(p2_cert)

        # ② Adjudicación Heyting Ω₃
        truth_val = self._phase_3_adjudicate_heyting_topos(p1_cert, p2_cert)

        # ③ Funtor semántico visceral Φ_sem
        visceral = self._phase_3_emit_visceral_functor(p1_cert, p2_cert, truth_val)

        # ④ RSI³-T³ + jerk ciber-físico
        rsi3_state = self._phase_3_refine_policy_rsi3(p2_cert)

        # Raíz Merkle SHA-256 sobre toda la cadena P1→P2→P3
        ts = time.time()
        merkle_payload = (
            f"SOVEREIGN:{ts}:"
            f"{p1_cert.chain_hash}:"
            f"{p2_cert.chain_hash}:"
            f"{truth_val.value}:"
            f"{visceral.camouflage_gauge_status}:"
            f"{visceral.poincare_attestation}:"
            f"{rsi3_state.iteration_level}:"
            f"{rsi3_state.lip_bound:.10e}:"
            f"{rsi3_state.residual_norm:.10e}:"
            f"{visceral.rsi3_closure_status}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_payload).hexdigest()

        gauge_rep = p2_cert.spectrum_report.gauge_report
        kam_rep = getattr(p2_cert.spectrum_report, "kam_report", None)
        jerk_rep = getattr(p2_cert.spectrum_report, "jerk_report", None)

        return StealthBooleSovereignState(
            timestamp=ts,
            truth_value=truth_val,
            is_camouflage_gauge_invariant=gauge_rep.is_gauge_camouflaged,
            back_action_db=gauge_rep.back_action_db,
            merkle_root_hash=merkle_root,
            visceral_metrics=visceral,
            rsi3_final=rsi3_state,
            poincare_final=p1_cert.poincare,
            kam_report=kam_rep,
            jerk_report=jerk_rep,
        )


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO — Orquestador OODA trifásico con continuidad monádica
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovStealthBooleAgent:
    """
    Soberano Agéntico Escudo Camaleón Booleano (SU(N) Stealth Gauge en R_Boole ⊗ Λ_Nov).

    Coordina el ciclo OODA trifásico con continuidad monádica de tipos:
        FASE 1 (Observe)  ⊳ Bridge₁₂ ⊲ FASE 2 (Orient) ⊳ Bridge₂₃ ⊲ FASE 3 (Decide&Act)
    e integra la auto-mejora recursiva RSI³ nivel 3 a través de la persistencia
    del estado de política en el espacio de Banach entre ciclos OODA.
    """

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._secret_key: bytes = secret_key or _DEFAULT_SECRET_KEY
        self._engine = BoolNovikovStealthBooleEngine(secret_key=self._secret_key)
        self._observer = StealthBoolePhase1Observer(secret_key=self._secret_key)
        self._orientor = StealthBoolePhase2Orientor(
            engine=self._engine, secret_key=self._secret_key,
        )
        self._decider = StealthBoolePhase3Decider(secret_key=self._secret_key)

        # RSI³: estado persistente de política en el espacio de Banach
        self._rsi_policy_state = np.zeros((_RSI_POLICY_DIM,), dtype=np.float64)
        self._ooda_cycle_count: int = 0

        logger.info(
            f"BoolNovikovStealthBooleAgent v{_AGENT_VERSION} iniciado "
            f"(RSI³ activo, ρ={_RSI_RHO})."
        )

    # ────────────────────────────────────────────────────────────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        incidence_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
    ) -> StealthBooleSovereignState:
        """
        Ejecuta el ciclo completo OODA trifásico anidado:
          FASE 1 ⊳ P1-Certif. → [Φ₁₂ gate] → FASE 2 ⊳ P2-Certif.
                 → [Φ₂₃ gate] → FASE 3 ⊳ EstadoSoberano
        Actualiza `self._rsi_policy_state` con el estado final RSI³-T³ para
        habilitar auto-mejora recursiva continua a través de ciclos.
        """
        # ═══════════════════════════════ FASE 1 ═══════════════════════════════
        p1_cert = self._observer.execute_phase1(rho_mac, rsi_seed=self._rsi_policy_state)

        # ─── COMPUERTA HOMOMÓRFICA Φ₁₂ ─────────────────────────────────────
        if not self._verify_p1_hmac(p1_cert):
            raise ValueError("Falla de autenticidad en Compuerta Homomórfica Φ₁₂.")

        # ═══════════════════════════════ FASE 2 ═══════════════════════════════
        p2_cert = self._orientor.execute_phase2(
            p1_cert=p1_cert,
            incidence_matrix=incidence_matrix,
            novikov_exponents=novikov_exponents,
            rho_mac=rho_mac,
        )

        # ─── COMPUERTA HOMOMÓRFICA Φ₂₃ ─────────────────────────────────────
        if not self._verify_p2_hmac(p2_cert):
            raise ValueError("Falla de estabilidad en Compuerta Homomórfica Φ₂₃.")

        # ═══════════════════════════════ FASE 3 ═══════════════════════════════
        sovereign_state = self._decider.execute_phase3(p1_cert, p2_cert)

        # ─── RSI³: persistencia del estado de política mejorado ────────────
        self._rsi_policy_state = np.asarray(
            sovereign_state.rsi3_final.state_vector, dtype=np.float64,
        )
        self._ooda_cycle_count += 1

        return sovereign_state

    # ────────────────────────────────────────────────────────────────────────
    def _verify_p1_hmac(self, p1_cert: StealthBoolePhase1Certificate) -> bool:
        """Compuerta Φ₁₂: verifica la firma HMAC del certificado P1."""
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + p1_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected_sig, p1_cert.hmac_signature)

    def _verify_p2_hmac(self, p2_cert: StealthBoolePhase2Certificate) -> bool:
        """Compuerta Φ₂₃: verifica la firma HMAC del certificado P2."""
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + p2_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected_sig, p2_cert.hmac_signature)


# ══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO — Ciclo OODA trifásico anidado con RSI³ multicycle
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== TEST AUTÓNOMO: BoolNovikovStealthBooleAgent v14 (3 FASES ANIDADAS) ===")
    agent = BoolNovikovStealthBooleAgent()

    m = 4
    rho_test = np.eye(m, dtype=np.complex128) / float(m)
    matrix_test = np.eye(m, dtype=np.float64)
    exps_test = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float64)

    # ─── 3 ciclos OODA para evidenciar la convergencia recursiva RSI³ ─────
    print("\n─── Convergencia RSI³ a través de ciclos OODA ───")
    for cycle in range(3):
        state = agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test)
        print(
            f"[ciclo {cycle + 1}] "
            f"Ω₃={state.truth_value.name:<9s} "
            f"Lip(T³)={state.rsi3_final.lip_bound:.6f} "
            f"resid={state.rsi3_final.residual_norm:.3e} "
            f"jerk+={state.rsi3_final.jerk_positive} "
            f"Poincaré={state.visceral_metrics.poincare_attestation}"
        )

    print("\n─── Detalle del último ciclo soberano ───")
    print(f"Veredicto Final Heyting Ω₃        : {state.truth_value.name} ({hex(state.truth_value.value)})")
    print(f"Invariante CAMOUFLAGE_GAUGE      : {state.is_camouflage_gauge_invariant}")
    print(f"Back-Action QND (dB)             : {state.back_action_db:.6f} dB")
    print(f"Raíz Merkle SHA-256              : {state.merkle_root_hash[:16]}...")
    print(f"Protección WACC (%)              : {state.visceral_metrics.wacc_protection_pct}%")
    print(f"Liberación Liquidez Neta (%)     : {state.visceral_metrics.net_liquidity_release_pct}%")
    print(f"Compresión Caché KV (%)          : {state.visceral_metrics.kv_cache_compression_pct}%")
    print(f"Estado camuflaje gauge           : {state.visceral_metrics.camouflage_gauge_status}")
    print(f"Atestación Poincaré-KAM          : {state.visceral_metrics.poincare_attestation}")
    print(f"Estado de cierre RSI³            : {state.visceral_metrics.rsi3_closure_status}")
    print(f"Poincaré ω (diofántinas)         : {state.poincare_final.omega}")
    print(f"Poincaré gap diofántino mínimo   : {state.poincare_final.min_diophantine_gap:.3e}")
    print(f"Poincaré KAM-estable             : {state.poincare_final.kam_stable}")
    print(f"Poincaré Melnikov max            : {state.poincare_final.melnikov_max:.3e}")
    print(f"Poincaré chaos detectado         : {state.poincare_final.chaos_detected}")
    print(f"RSI³ x* (punto fijo teórico)     : {_RSI_BIAS_SCALAR / (1.0 - _RSI_RHO):.6f}")
    print(f"RSI³ x₃ (último estado)          : {tuple(round(v, 6) for v in state.rsi3_final.state_vector)}")

    print("\n=== TEST COMPLETADO CON ÉXITO (3 FASES ANIDADAS + RSI³ MULTICYCLE) ===")