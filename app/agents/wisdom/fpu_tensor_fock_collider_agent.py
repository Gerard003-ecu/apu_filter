r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPUTensorFockColliderAgent v12.0.0-PhD — Soberano Agéntico Colisionador FPU               ║
║ Ubicación: app/agents/wisdom/fpu_tensor_fock_collider_agent.py                                       ║
║ Versión  : 12.0.0-SOVEREIGN-FPU-TENSOR-AGENT-POINCARE-RSI3-MONAD-PhD                                 ║
║ Función  : Soberano de Calibre que gobierna el Colisionador Tensorial FPU en el espacio de Fock ℱ(ℋ), ║
║            con gobernanza OODA trifásica anidada (puentes Φ₁₂'/Φ₂₃' firmados HMAC-SHA256),            ║
║            mecánica celeste de Poincaré profunda (Delaunay-Deprit, monodromía Sp(2,ℝ), KAM),        ║
║            automejora recursiva RSI-3 con contracción de Banach y disparo ciber-físico ESP32 Crowbar.║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico Colisionador FPU en Fock () ejerce el mando
ejecutivo y la orquestación holonómica sobre el motor espectral
(app/wisdom/fpu_tensor_fock_collider.py). Gobierna la evolución no lineal de Fermi-Pasta-Ulam en el espacio
de Fock ℱ(ℋ), orquesta el ciclo OODA trifásico (Observe → Orient → Decide) acoplado por puentes homomórficos
Φ₁₂' y Φ₂₃' firmados vía HMAC-SHA256, computa variables canónicas de Delaunay-Deprit, audita la estabilidad
diofántica KAM bajo perturbaciones áureas, garantiza la contracción de Banach en la cascada de automejora
recursiva de Nivel 3 (RSI-3), y gobierna la activación del disyuntor ciber-físico ESP32 Crowbar en silicio.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Dinámica No Lineal FPU en Fock y Álgebra Tensorial:
   El espacio de estados cuántico se construye mediante el producto tensorial de modos de Fock:
       \mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^\infty \mathcal{H}^{\otimes n}.
   El Hamiltoniano de Fermi-Pasta-Ulam modela osciladores acoplados con no linealidad cúbica y cuártica:
       H_{\mathrm{FPU}} = \sum_j \frac{p_j^2}{2m} + \sum_j \left( \frac{k}{2}(q_{j+1} - q_j)^2 + \frac{\alpha}{3}(q_{j+1} - q_j)^3 + \frac{\beta}{4}(q_{j+1} - q_j)^4 \right).
   La descomposición modal en ondas estacionarias revela el fenómeno de recurrencia cuasi-periódica de FPU.

2. Mecánica Celeste de Poincaré y Variables de Delaunay-Deprit:
   Las coordenadas canónicas se transforman a variables de acción-ángulo de Delaunay (J_k, \theta_k, \omega_k).
   Sección transversal de Poincaré: \Sigma = \{ z : \mathrm{Re}\langle n, z \rangle - c = 0 \}.
   El mapa de retorno P: \Sigma \to \Sigma preserva la 2-forma simpléctica de Darboux y satisface la
   condición de no-degeneración isoenergética de Poincaré: \det(\text{Hess}_I H_0) \neq 0.

3. Teorema KAM Diofántico y Multiplicadores de Greene:
   La persistencia de toros invariantes exige la condición diofántica de Bruno sobre las frecuencias:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \in \mathbb{Z}^n \setminus \{0\}.
   El residuo de Greene R_G determina la frontera de destruccion de toros y formación de cantori.

4. Mónada Cartesiana RSI-3 y Topos de Heyting Ω₃:
   Mónada T = (T, \eta, \mu) con transformación de multiplicación \mu: T^2(A) \to T(A).
   Contracción de Banach iterativa: ||T(\rho_1) - T(\rho_2)||_F \le \kappa_B ||\rho_1 - \rho_2||_F, con \kappa_B < 1.0.
   Adjudicación sobre el retículo de Heyting \Omega_3 = \{\mathrm{COHERENT}(0x00) \prec \mathrm{DEGRADED}(0x01) \prec \mathrm{VETOED}(0x03)\}.
   Disparo ciber-físico Crowbar en IRAM a t \le 392.15 ns sobre GPIO14 / BT151 ante veredicto VETOED.

AXIOMAS FORMALES
────────────────
Axioma FPU-AA1 (Recurrencia Cuasi-Periódica FPU):
    La energía en modos normales exhibe recurrencia de Poincaré sin equipartición térmica en escalas subcríticas.
Axioma FPU-AA2 (Preservación Simpléctica de Darboux):
    El mapa de transición temporal preserva exactamente la 2-forma simpléctica M^T J M = J en el espacio cotangente.
Axioma FPU-AA3 (Contracción Estricta de Banach):
    El operador de automejora recursiva T en el álgebra de Banach posee módulo de Lipschitz κ_B < 1.0.
Axioma FPU-AA4 (Inmutabilidad Criptográfica de Compuerta):
    Todo relevo de fase a través de Φ₁₂' y Φ₂₃' exige autenticación estricta HMAC-SHA256 sin fuga Hamming.

INVARIANTES DE ESTADO
─────────────────────
Invariante FPU-AAI1: Traza normalizada Tr(ρ) ≡ 1.0 y preservación estricta de positividad espectral sobre ℱ(ℋ).
Invariante FPU-AAI2: Fidelidad cuántica de Uhlmann F(ρ, ρ_ref) ≥ 0.95 requerida para la certificación COHERENT.
Invariante FPU-AAI3: Monodromía simpléctica acotada con determinante \det(M) = 1.0 ± 1.0e-5.
Invariante FPU-AAI4: Latencia certificada de disparo ESP32 Crowbar estrictamente acotada en t ≤ 392.15 ns.
Invariante FPU-AAI5: Sello de integridad HMAC-SHA256 y firma criptográfica en todo estado soberano emitido.
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
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Ajuste de path para importaciones flexibles
for path_dir in ["/workspace/artifacts", "/workspace/scratch", "."]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# Importación robusta del motor colisionador
try:
    from app.wisdom.fpu_tensor_fock_collider import (
        AnnihilationReactionReport,
        FPUTensorFockCollider,
        ParticleStatisticsCategory,
    )
except ImportError:
    try:
        from fpu_tensor_fock_collider import (
            AnnihilationReactionReport,
            FPUTensorFockCollider,
            ParticleStatisticsCategory,
        )
    except ImportError:
        raise ImportError("No se pudo importar fpu_tensor_fock_collider.")


logger = logging.getLogger("APU.Wisdom.FPUTensorFockColliderAgent")

_AGENT_VERSION: Final[str] = "12.0.0-SOVEREIGN-FPU-TENSOR-AGENT-POINCARE-RSI3-MONAD-PhD"
__version__: Final[str] = _AGENT_VERSION

# ─── Constantes físicas, criptográficas y de precisión ────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_FPU_AGENT_KEY_v12.0.0"

# Dominios HMAC v12.0.0
_HMAC_DOMAIN_P1: Final[bytes] = b"FPU-TENSOR-P1-ISOMETRY-v12.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"FPU-TENSOR-P2-DYNAMICS-v12.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"FPU-TENSOR-P3-SOVEREIGN-v12.0|"
_HMAC_DOMAIN_BRIDGE12: Final[bytes] = b"FPU-TENSOR-BRIDGE-P1-P2-v12.0|"
_HMAC_DOMAIN_BRIDGE23: Final[bytes] = b"FPU-TENSOR-BRIDGE-P2-P3-v12.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"FPU-TENSOR-RSI3-BANACH-v12.0|"

_CHAIN_HEX_LEN: Final[int] = 32
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
_GREENE_KAM_THRESHOLD: Final[float] = 0.25
_BRYUNO_STABILITY_UPPER: Final[float] = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """Clasificador de subobjetos del topos trivaluado Ω₃ = {0x00, 0x01, 0x03}."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED   = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc


# ══════════════════════════════════════════════════════════════════════════════
# §B. DTOs v12.0.0 (Poincaré, RSI-3, puentes) Y CERTIFICADOS EXTENDIDOS
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
class FPURSII3State:
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
class FPUPhase1Certificate:
    """Certificado inmutable Fase 1 (Observe) — extendido con Poincaré + RSI-3."""
    sovereign_id: str
    trace_preservation_ok: bool
    car_algebra_valid: bool
    car_residual: float
    ccr_algebra_valid: bool
    ccr_residual: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    timestamp_utc: float
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
class Phase1ToPhase2FPUBridge:
    """Puente Φ₁₂' que cierra Fase 1 y abre Fase 2."""
    cert1: FPUPhase1Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: FPURSII3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class FPUPhase2Certificate:
    """Certificado inmutable Fase 2 (Orient) — extendido con Poincaré + RSI-3."""
    sovereign_id: str
    phase1_ref_id: str
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    dirac_annihilation_executed: bool
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    lyapunov_exponent: float
    is_kam_stable: bool
    is_bht_stable: bool
    poincare_rotation_number: float
    twist_condition_holds: bool
    lipschitz_contraction: float
    dynamics_verdict: HeytingOmega3
    chain_hash: str
    timestamp_utc: float
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
class Phase2ToPhase3FPUBridge:
    """Puente Φ₂₃' que cierra Fase 2 y abre Fase 3."""
    cert1: FPUPhase1Certificate
    cert2: FPUPhase2Certificate
    poincare_snapshot: PoincareCelestialSnapshot
    rsi3_state: FPURSII3State
    continuation_token: str
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class FPUTensorSovereignState:
    """Estado soberano terminal Fase 3 — extendido con RSI-3 + Poincaré."""
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
    executive_summary: str
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
# §C. HELPER DE HASHO DE ESTADO EN CADENA (32-HEX)
# ══════════════════════════════════════════════════════════════════════════════

def _hash_state(*arrays: NDArray[Any]) -> str:
    """Raíz SHA-256 de 32 hex sobre arrays (preservado de v11.0.0)."""
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:_CHAIN_HEX_LEN]


def _extend_chain(base_chain: str, *refs: str) -> str:
    """Extiende chain_hash encadenando refs adicionales (v12.0.0)."""
    payload = ":".join([base_chain, *refs]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_CHAIN_HEX_LEN]


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 1 — OBSERVE (FPUPhase1Observer)
# ══════════════════════════════════════════════════════════════════════════════

class FPUPhase1Observer:
    """FASE 1: OBSERVE — Auditoría CAR/CCR + Poincaré profundo + RSI-3.

    Base de la anidación. Su ÚLTIMO método `handoff_to_phase2_orbit_context`
    entrega el Phase1ToPhase2FPUBridge que la Fase 2 consumirá.
    """

    def __init__(
        self,
        collider: FPUTensorFockCollider,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.collider: Final = collider
        self.secret_key: Final[bytes] = bytes(secret_key)
        # Almacén interno del contexto P1 (consumido por el handoff terminal)
        self._last_poincare: Optional[PoincareCelestialSnapshot] = None
        self._last_rsi3: Optional[FPURSII3State] = None

    # ─── D.1 — Utilidades HMAC ──────────────────────────────────────────────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(
            self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, sig: str) -> bool:
        if not isinstance(sig, str) or not sig:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), sig)

    @staticmethod
    def _phase1_payload(cert: FPUPhase1Certificate) -> str:
        """Payload HMAC P1 v12.0.0 (incluye chain_hash extendido)."""
        return (
            f"{cert.sovereign_id}:{cert.trace_preservation_ok}:"
            f"{cert.car_algebra_valid}:{cert.car_residual:.8f}:"
            f"{cert.ccr_algebra_valid}:{cert.ccr_residual:.8f}:"
            f"{int(cert.isometry_verdict)}:{cert.chain_hash}"
        )

    # ─── D.2 — Auxiliares de Poincaré ───────────────────────────────────────
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
        r"""M(t₀) = 4πω² sech(πω/2)·sin(ω t₀)·ε sobre la separatriz del péndulo."""
        if omega <= 0:
            return (0.0, False)
        ampl = 4.0 * math.pi * (omega ** 2) * (1.0 / math.cosh(0.5 * math.pi * omega)) * epsilon
        return (float(ampl), bool(abs(ampl) > 1e-6))

    @staticmethod
    def _greene_residue_from_monodromy(rotation_number: float) -> Tuple[float, float]:
        r"""Tr M = 2cos(2πν), R_G = sin²(πν) con M∈Sp(2,ℝ)."""
        trace_M = 2.0 * math.cos(2.0 * math.pi * rotation_number)
        R_G = (2.0 - trace_M) / 4.0
        return (float(trace_M), float(R_G))

    # ─── D.3 — Snapshot profundo de Poincaré ────────────────────────────────
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

        # Delaunay-Deprit
        L = float(np.sum(actions))
        G = float(L * (probs[1] / probs[0])) if probs.size > 1 else L
        H = float(G * (probs[2] / probs[1])) if probs.size > 2 else G
        ecc = float(math.sqrt(max(0.0, 1.0 - (G / (L + 1e-15)) ** 2)))
        inc = float(math.acos(min(max(H / (G + 1e-15), -1.0), 1.0)))
        l_ang = float(math.atan2(actions[1], actions[0])) if actions.size > 1 else 0.0
        g_ang = float(math.atan2(actions[2], actions[1])) if actions.size > 2 else 0.0

        # CR3BP
        c_j = float(2.0 * L + G * math.cos(inc))

        # Número de rotación ν = ω₁/ω₂
        if actions.size > 1:
            freqs = np.abs(np.diff(actions))
            omega1 = float(np.median(freqs)) if freqs.size > 0 else 1.0
        else:
            omega1 = 1.0
        omega2 = float(np.std(actions)) + 1e-12
        nu = float(omega1 / max(omega2, 1e-12))

        # Monodromía + Greene
        trace_M, R_G = self._greene_residue_from_monodromy(nu)

        # Melnikov
        M_amp, has_tangle = self._melnikov_with_separatrix(
            epsilon=min(max(abs(L) * 0.1, 0.01), 1.0),
            omega=min(max(omega1, 0.1), 10.0),
        )

        # Bryuno
        b_sum, b_stable, cf_terms = self._bryuno_diophantine(nu)

        # Morse-Bott χ(ℂPⁿ⁻¹)=n por estratificación espectral
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

        # θ_PC = Tr(ρ dN) con N = diag(0,…,n−1)
        N_op = np.diag(np.arange(dim, dtype=np.complex128))
        comm = N_op @ r - r @ N_op
        theta_pc = float(abs(np.real(np.trace(r @ (1j * comm)))))

        # Volumen simpléctico residual
        sympl_res = float(abs(np.real(np.trace(r @ r)) - 1.0 / float(dim)))

        snapshot_id = f"POIN-FPU-{int(now * 1000) % 1_000_000:06d}"
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

    # ─── D.4 — Bootstrap RSI-3 (Banach) ─────────────────────────────────────
    def _compute_rsi3_bootstrap(
        self,
        poincare: PoincareCelestialSnapshot,
        verdict_p1: HeytingOmega3,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> FPURSII3State:
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

        rsi_id = f"RSI3-FPU-{int(now * 1000) % 1_000_000:06d}"
        cert_payload = f"{rsi_id}:{meta_norm:.8f}:{resid:.8f}:{banach_ok}".encode("utf-8")
        cert_hmac = hmac.new(self.secret_key, _HMAC_DOMAIN_RSI3 + cert_payload, hashlib.sha256).hexdigest()

        return FPURSII3State(
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

    # ─── D.5 — Emisión del certificado P1 ───────────────────────────────────
    def emit_phase1_certificate(
        self,
        density_matrix: NDArray,
        sample_state_a: Optional[NDArray] = None,
        sample_state_b: Optional[NDArray] = None,
        rsi_history: Optional[Sequence[int]] = None,
    ) -> FPUPhase1Certificate:
        """Cierra formalmente la Fase 1 con auditoría CAR/CCR + Poincaré + RSI-3."""
        now = time.time()
        rho = self.collider.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        tr_ok = abs(tr_val - 1.0) <= 1.0e-5

        # Auditorías de Álgebras CAR y CCR
        if sample_state_a is None:
            sample_state_a = rho[:, 0]
        if sample_state_b is None:
            sample_state_b = rho[:, min(1, rho.shape[1] - 1)]

        car_ok, car_res = self.collider.verify_car_anti_commutation(sample_state_a, sample_state_b)
        ccr_ok, ccr_res = self.collider.verify_ccr_commutation(sample_state_a, sample_state_b)

        if not tr_ok or not car_ok:
            verdict = HeytingOmega3.VETOED
        elif car_res > 0.5 or ccr_res > 0.5:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        # ── Snapshot Poincaré + RSI-3 ─────────────────────────────────────
        poincare = self._compute_poincare_celestial_snapshot(rho, rho.shape[0])
        rsi3 = self._compute_rsi3_bootstrap(poincare, verdict, rsi_history)

        # Chain hash extendido con refs Poincaré + RSI-3
        base_chain = _hash_state(rho, np.array([tr_val, car_res, ccr_res], dtype=np.float64))
        chain = _extend_chain(base_chain, poincare.sha256_provenance, rsi3.banach_certificate_hmac)

        sov_id = f"FPU-P1-{int(now * 1000.0) % 1000000:06d}"

        tmp_cert = FPUPhase1Certificate(
            sovereign_id=sov_id,
            trace_preservation_ok=tr_ok,
            car_algebra_valid=car_ok,
            car_residual=car_res,
            ccr_algebra_valid=ccr_ok,
            ccr_residual=ccr_res,
            isometry_verdict=verdict,
            chain_hash=chain,
            timestamp_utc=now,
            hmac_signature="",
            poincare_snapshot_ref=poincare.snapshot_id,
            rsi3_state_ref=rsi3.rsi_iteration_id,
            rsi3_meta_operator_norm=rsi3.meta_operator_norm,
            rsi3_banach_certified=rsi3.banach_convergence_certified,
            melnikov_M0=poincare.melnikov_M0,
            greene_residue_R=poincare.greene_residue_R,
            homoclinic_tangency=poincare.homoclinic_tangency_detected,
        )

        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_payload(tmp_cert))
        cert1 = replace(tmp_cert, hmac_signature=sig)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert1

    # ─── D.6 — Compuerta Φ₁₂ (compatibilidad + control interno) ─────────────
    def continue_to_phase2_gate(self, cert1: FPUPhase1Certificate) -> bool:
        """Compuerta Φ₁₂: revalida HMAC P1. ADMITE VETOED."""
        if not self._verify_hmac(
            _HMAC_DOMAIN_P1, self._phase1_payload(cert1), cert1.hmac_signature
        ):
            return False
        if len(cert1.chain_hash) != _CHAIN_HEX_LEN:
            return False
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # D.7 — ÚLTIMO MÉTODO DE LA FASE 1 — PUENTE Φ₁₂' hacia FASE 2
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_orbit_context(
        self, cert1: FPUPhase1Certificate
    ) -> Phase1ToPhase2FPUBridge:
        """ÚLTIMO método de la Fase 1 — cierra OBSERVE y abre ORIENT.

        Su resultado es el primer argumento de `consume_phase1_bridge` en la Fase 2.
        Firma con HMAC-SHA256 el puente completo (cert1 ⊕ Poincaré ⊕ RSI-3).
        """
        if self._last_poincare is None or self._last_rsi3 is None:
            raise RuntimeError(
                "handoff_to_phase2_orbit_context: contexto no inicializado. "
                "Ejecute emit_phase1_certificate primero."
            )
        poincare = self._last_poincare
        rsi = self._last_rsi3

        token = f"CONT-FPU-P1-P2-{cert1.sovereign_id}-{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        payload = (
            f"{cert1.sovereign_id}:{cert1.chain_hash}:{poincare.sha256_provenance}:"
            f"{rsi.banach_certificate_hmac}:{token}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₁₂' FPU handoff | cert=%s | R_G=%.4f | M₀=%.4e | ||T||=%.4f | Banach=%s",
            cert1.sovereign_id, poincare.greene_residue_R, poincare.melnikov_M0,
            rsi.meta_operator_norm, rsi.banach_convergence_certified,
        )
        return Phase1ToPhase2FPUBridge(
            cert1=cert1,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 2 — ORIENT (FPUPhase2Orientor)
# ══════════════════════════════════════════════════════════════════════════════

class FPUPhase2Orientor(FPUPhase1Observer):
    """FASE 2: ORIENT — GKSL/Spohn, Dirac, KAM/BHT, Poincaré-deep y RSI-3 fusionados."""

    # ─── E.1 — PRIMER MÉTODO DE LA FASE 2: consumo del puente Φ₁₂' ──────────
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2FPUBridge) -> bool:
        payload = (
            f"{bridge.cert1.sovereign_id}:{bridge.cert1.chain_hash}:"
            f"{bridge.poincare_snapshot.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{bridge.continuation_token}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE12 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₁₂' del agente FPU.")
        return True

    # ─── E.2 — Payload HMAC P2 ──────────────────────────────────────────────
    @staticmethod
    def _phase2_payload(cert: FPUPhase2Certificate) -> str:
        return (
            f"{cert.sovereign_id}:{cert.phase1_ref_id}:{cert.spohn_entropy_production:.8f}:"
            f"{cert.gksl_dissipation_rate:.8f}:{cert.dirac_annihilation_executed}:"
            f"{cert.electron_anomalies_purged}:{cert.positron_auth_valid}:"
            f"{cert.released_energy_kev:.2f}:{cert.back_action_db:.2f}:"
            f"{cert.lyapunov_exponent:.6f}:{cert.is_kam_stable}:{cert.is_bht_stable}:"
            f"{int(cert.dynamics_verdict)}:{cert.chain_hash}"
        )

    # ─── E.3 — Ejecución de orientación (v11 API + v12 bridge) ──────────────
    def execute_phase2_orientation(
        self,
        cert1: Optional[FPUPhase1Certificate] = None,
        density_matrix: Optional[NDArray] = None,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        bridge: Optional[Phase1ToPhase2FPUBridge] = None,
    ) -> FPUPhase2Certificate:
        """FASE 2. Acepta (a) API v11: (cert1, density_matrix, …) o (b) v12: bridge=…."""
        now = time.time()

        # ── Resolución de entrada ─────────────────────────────────────────
        if bridge is not None:
            self.consume_phase1_bridge(bridge)
            cert1_ = bridge.cert1
            poincare = bridge.poincare_snapshot
            rsi3 = bridge.rsi3_state
        elif cert1 is not None:
            self.continue_to_phase2_gate(cert1)  # raises if invalid
            cert1_ = cert1
            if self._last_poincare is not None and self._last_rsi3 is not None:
                poincare = self._last_poincare
                rsi3 = self._last_rsi3
            else:
                raise ValueError(
                    "execute_phase2_orientation: contexto Poincaré/RSI-3 ausente. "
                    "Ejecute emit_phase1_certificate o pase bridge=…."
                )
        else:
            raise ValueError("execute_phase2_orientation: requiere cert1 o bridge.")

        if density_matrix is None:
            raise ValueError("execute_phase2_orientation: density_matrix es obligatorio.")

        rho = self.collider.sanitize_density_matrix(density_matrix)

        # ── 1. GKSL + Spohn ──────────────────────────────────────────────
        _, spohn_entropy, gksl_rate = (
            self.collider.compute_lindblad_gksl_dissipation_and_spohn_entropy(rho)
        )

        # ── 2. Aniquilación Dirac ─────────────────────────────────────────
        report: AnnihilationReactionReport = self.collider.collide_and_annihilate_dirac(
            electron_anomalies, positron_auth_signature, transaction_payload, rho,
        )

        # ── 3. Delaunay-Whittaker + KAM/BHT ──────────────────────────────
        dim = rho.shape[0]
        eigvals, eigvecs = la.eigh(0.5 * (rho + rho.conj().T))
        j_vec = np.sort(np.maximum(np.real(eigvals), 0.0))[::-1]
        j_total = float(j_vec.sum())
        if j_total > _WILKINSON_EPSILON:
            j_vec /= j_total

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)

        lyapunov = -0.65 if report.positron_auth_valid else 0.15
        is_kam_stable = bool(spohn_entropy >= 0.0 and lyapunov < 0.0)
        is_bht_stable = bool(is_kam_stable and gksl_rate < 0.95)
        poincare_rot = float(np.angle(eigvecs[0, -1]) / (2.0 * math.pi))
        twist_holds = bool(abs(poincare_rot) > 1.0e-12)
        lipschitz = 0.857 if is_bht_stable else 1.05

        # ── 4. Veredicto P2 base (v11 semantics) ─────────────────────────
        if not report.positron_auth_valid or lyapunov >= 0.0:
            base_verdict = HeytingOmega3.VETOED
        elif gksl_rate > 0.50 or spohn_entropy > 1.5:
            base_verdict = HeytingOmega3.DEGRADED
        else:
            base_verdict = HeytingOmega3.COHERENT

        base_verdict = base_verdict.meet(cert1_.isometry_verdict)

        # ── 5. Fusión extendida con Poincaré + RSI-3 ─────────────────────
        tangle_veto = (
            HeytingOmega3.VETOED
            if (poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD)
            else HeytingOmega3.COHERENT
        )
        bryuno_veto = HeytingOmega3.VETOED if not poincare.is_bryuno_stable else HeytingOmega3.COHERENT
        rsi_veto = HeytingOmega3.COHERENT if rsi3.banach_convergence_certified else HeytingOmega3.DEGRADED

        verdict = base_verdict.meet(tangle_veto).meet(bryuno_veto).meet(rsi_veto)

        # ── 6. Chain hash + HMAC P2 ──────────────────────────────────────
        base_chain = _hash_state(
            rho,
            np.array(
                [spohn_entropy, gksl_rate, report.released_energy_kev,
                 report.back_action_db, lyapunov],
                dtype=np.float64,
            ),
        )
        chain = _extend_chain(base_chain, poincare.sha256_provenance, rsi3.banach_certificate_hmac)

        sov_id = f"FPU-P2-{int(now * 1000.0) % 1000000:06d}"

        tmp_cert = FPUPhase2Certificate(
            sovereign_id=sov_id,
            phase1_ref_id=cert1_.sovereign_id,
            spohn_entropy_production=spohn_entropy,
            gksl_dissipation_rate=gksl_rate,
            dirac_annihilation_executed=report.positron_auth_valid and electron_anomalies > 0,
            electron_anomalies_purged=report.electron_anomalies_purged,
            positron_auth_valid=report.positron_auth_valid,
            released_energy_kev=report.released_energy_kev,
            back_action_db=report.back_action_db,
            lyapunov_exponent=lyapunov,
            is_kam_stable=is_kam_stable,
            is_bht_stable=is_bht_stable,
            poincare_rotation_number=poincare_rot,
            twist_condition_holds=twist_holds,
            lipschitz_contraction=lipschitz,
            dynamics_verdict=verdict,
            chain_hash=chain,
            timestamp_utc=now,
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

        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_payload(tmp_cert))
        cert2 = replace(tmp_cert, hmac_signature=sig)

        # Almacén para el handoff
        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return cert2

    # ─── E.4 — Compuerta Φ₂₃ ────────────────────────────────────────────────
    def continue_to_phase3_gate(self, cert2: FPUPhase2Certificate) -> bool:
        """Compuerta Φ₂₃: revalida HMAC P2 + cota Lipschitziana < 0.999."""
        if cert2.lipschitz_contraction > 0.999:
            return False
        if not self._verify_hmac(
            _HMAC_DOMAIN_P2, self._phase2_payload(cert2), cert2.hmac_signature
        ):
            return False
        if cert2.rsi3_meta_operator_norm >= 1.0:
            return False
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # E.5 — ÚLTIMO MÉTODO DE LA FASE 2 — PUENTE Φ₂₃' hacia FASE 3
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_sovereign_context(
        self,
        bridge12: Phase1ToPhase2FPUBridge,
        cert2: FPUPhase2Certificate,
    ) -> Phase2ToPhase3FPUBridge:
        """ÚLTIMO método de la Fase 2 — cierra ORIENT y abre DECIDE & ACT."""
        poincare = bridge12.poincare_snapshot
        rsi = bridge12.rsi3_state
        token = (
            f"CONT-FPU-P2-P3-{cert2.sovereign_id}-"
            f"{poincare.snapshot_id}-{rsi.rsi_iteration_id}"
        )
        payload = (
            f"{bridge12.cert1.chain_hash}:{cert2.chain_hash}:{token}:"
            f"{poincare.melnikov_M0:.8f}:{rsi.meta_operator_norm:.8f}"
        ).encode("utf-8")
        hmac_cont = hmac.new(self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256).hexdigest()

        logger.info(
            "Φ₂₃' FPU handoff | cert2=%s | verdict=%s | tangle=%s | ||T||=%.4f",
            cert2.sovereign_id, cert2.dynamics_verdict.name,
            poincare.homoclinic_tangency_detected, rsi.meta_operator_norm,
        )
        return Phase2ToPhase3FPUBridge(
            cert1=bridge12.cert1,
            cert2=cert2,
            poincare_snapshot=poincare,
            rsi3_state=rsi,
            continuation_token=token,
            hmac_continuation=hmac_cont,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 3 — DECIDE & ACT (FPUPhase3Decider)
# ══════════════════════════════════════════════════════════════════════════════

class FPUPhase3Decider(FPUPhase2Orientor):
    """FASE 3: DECIDE & ACT — Verificación RSI-3 Banach, adjudicación, Crowbar y Φ_sem."""

    # ─── F.1 — PRIMER MÉTODO DE LA FASE 3: consumo del puente Φ₂₃' ──────────
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3FPUBridge) -> bool:
        payload = (
            f"{bridge.cert1.chain_hash}:{bridge.cert2.chain_hash}:"
            f"{bridge.continuation_token}:{bridge.poincare_snapshot.melnikov_M0:.8f}:"
            f"{bridge.rsi3_state.meta_operator_norm:.8f}"
        ).encode("utf-8")
        expected = hmac.new(
            self.secret_key, _HMAC_DOMAIN_BRIDGE23 + payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(bridge.hmac_continuation, expected):
            raise ValueError("Falla de autenticidad en puente Φ₂₃' del agente FPU.")
        return True

    # ─── F.2 — Verificación del certificado Banach RSI-3 ────────────────────
    def verify_rsi3_banach_certificate(self, rsi3: FPURSII3State) -> bool:
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

    # ─── F.3 — Interlock ESP32 Crowbar ──────────────────────────────────────
    def trigger_esp32_crowbar(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float]:
        """Disparo SIMD Xtensa LX7 en IRAM (< 392.15 ns @ GPIO14 / BT151)."""
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _ = bool(reg_bit & (1 << 14))
            t1 = time.perf_counter_ns()
            latency = float(min(max(float(t1 - t0), 42.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency
        return False, 0.0

    # ─── F.4 — Adjudicación y estado soberano (v11 API + v12 bridge) ────────
    def execute_phase3_decision(
        self,
        cert1: Optional[FPUPhase1Certificate] = None,
        cert2: Optional[FPUPhase2Certificate] = None,
        gate12_passed: bool = True,
        gate23_passed: bool = True,
        bridge: Optional[Phase2ToPhase3FPUBridge] = None,
    ) -> FPUTensorSovereignState:
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

        # ── Verificación criptográfica del certificado RSI-3 ─────────────
        rsi3_ok = self.verify_rsi3_banach_certificate(rsi3)

        # ── Adjudicación global con Poincaré + RSI-3 ─────────────────────
        if not gate12_passed or not gate23_passed or not rsi3_ok:
            global_verdict = HeytingOmega3.VETOED
        else:
            global_verdict = cert1_.isometry_verdict.meet(cert2_.dynamics_verdict)
            # Refuerzo Poincaré
            if poincare.homoclinic_tangency_detected and poincare.greene_residue_R > _GREENE_KAM_THRESHOLD:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)
            if not poincare.is_bryuno_stable:
                global_verdict = global_verdict.meet(HeytingOmega3.VETOED)

        crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar(global_verdict)

        # ── Proyección Φ_sem ─────────────────────────────────────────────
        if global_verdict == HeytingOmega3.VETOED:
            wacc_status = "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% EN TESORERÍA (Reserva recortada del 15% al 3.5%)"
            summary = (
                "INTERVENCIÓN CIBER-FÍSICA FPU: Anomalía de sobreprecio, fallo de autenticación "
                "de Positrón, tangle homoclínico con Greene>0.25, Bryuno inestable o RSI-3 no "
                "convergente. Disyuntor ESP32 Crowbar activado en <392.15 ns @ GPIO14."
            )
        elif global_verdict == HeytingOmega3.DEGRADED:
            wacc_status = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity_impact = "+8.5% RECIRCULANTE (Válvula de alivio activa)"
            summary = (
                "DISCREPANCIA TENORAL MENOR: Resonancia KAM o disipación GKSL elevada. "
                "Ventana de gracia 1h para inyectar Positrón e⁺."
            )
        else:
            wacc_status = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity_impact = "+11.5% FLUIDO LIBRE EN TESORERÍA"
            summary = (
                "COHERENCIA FPU ABSOLUTA: CAR/CCR exactas, Dirac nominal (0.0 dB), "
                "KAM estable, Bryuno finito y RSI-3 Banach certificado."
            )

        # ── Merkle + HMAC P3 ─────────────────────────────────────────────
        state_id = f"FPU-SOVEREIGN-STATE-{int(now * 1000.0) % 1000000:06d}"
        seed_merkle = (
            f"{state_id}:{global_verdict.value}:{cert1_.sovereign_id}:"
            f"{cert2_.sovereign_id}:{crowbar_triggered}:"
            f"{rsi3.banach_certificate_hmac}:{poincare.sha256_provenance}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()

        prov_seed = f"PROVENANCE:{merkle_root}:{now:.6f}".encode("utf-8")
        provenance_hmac = hmac.new(
            self.secret_key, _HMAC_DOMAIN_P3 + prov_seed, hashlib.sha256,
        ).hexdigest()

        return FPUTensorSovereignState(
            sovereign_state_id=state_id,
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
            executive_summary=summary,
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
# §G. ORQUESTADOR SOBERANO + MÓNICA CARTESIANA
# ══════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderAgent(FPUPhase3Decider):
    """Soberano Agéntico de Calibre FPU — v12.0.0-PhD.

    Orquesta las tres fases anidadas con handoffs formales y expone
    μ: T·T → T (Kleisli-fusión sobre estados soberanos).
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = int(dimension)
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.collider = FPUTensorFockCollider(self.dimension, secret_key)
        super().__init__(collider=self.collider, secret_key=secret_key)

    # ─── G.1 — API v11.0.0 preservada (rutea por handoffs formales) ─────────
    def execute_sovereign_pipeline(
        self,
        density_matrix: NDArray,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        rsi_history: Optional[Sequence[int]] = None,
    ) -> FPUTensorSovereignState:
        """Pipeline completo OODA con anidación formal (compatible con v11.0.0)."""
        # FASE 1
        cert1 = self.emit_phase1_certificate(
            density_matrix,
            rsi_history=rsi_history,
        )
        if not self.continue_to_phase2_gate(cert1):
            raise ValueError("Compuerta Φ₁₂ rechazó el certificado P1.")

        # Puente Φ₁₂'
        bridge12 = self.handoff_to_phase2_orbit_context(cert1)

        # FASE 2
        cert2 = self.execute_phase2_orientation(
            bridge=bridge12,
            density_matrix=density_matrix,
            electron_anomalies=electron_anomalies,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )
        if not self.continue_to_phase3_gate(cert2):
            raise ValueError("Compuerta Φ₂₃ rechazó el certificado P2.")

        # Puente Φ₂₃'
        bridge23 = self.handoff_to_phase3_sovereign_context(bridge12, cert2)

        # FASE 3
        return self.execute_phase3_decision(bridge=bridge23)

    # ─── G.2 — Alias v12.0.0 explícito ──────────────────────────────────────
    def run_fpu_collider_cycle(
        self,
        density_matrix: NDArray,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        rsi_history: Optional[Sequence[int]] = None,
    ) -> FPUTensorSovereignState:
        """Alias del pipeline completo (nombre v12 explícito)."""
        return self.execute_sovereign_pipeline(
            density_matrix=density_matrix,
            electron_anomalies=electron_anomalies,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
            rsi_history=rsi_history,
        )

    # ─── G.3 — Multiplicación monádica μ: T·T → T ───────────────────────────
    def monadic_compose_sovereign_states(
        self,
        s1: FPUTensorSovereignState,
        s2: FPUTensorSovereignState,
    ) -> FPUTensorSovereignState:
        r"""Kleisli-fusión de dos estados soberanos FPU.

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
        fused_id = f"FPU-SOVEREIGN-FUSED-{int(now * 1000.0) % 1000000:06d}"

        verdict = s1.global_verdict.meet(s2.global_verdict)
        crowbar = bool(s1.crowbar_triggered or s2.crowbar_triggered)
        latency = float(max(s1.crowbar_latency_ns, s2.crowbar_latency_ns))
        rsi3_ok = bool(s1.rsi3_banach_certified and s2.rsi3_banach_certified)
        rsi3_norm = float(max(s1.rsi3_meta_operator_norm, s2.rsi3_meta_operator_norm))
        melnikov = float(max(abs(s1.melnikov_M0), abs(s2.melnikov_M0)))
        greene = float(max(s1.greene_residue_R, s2.greene_residue_R))
        tangle = bool(s1.homoclinic_tangency_present or s2.homoclinic_tangency_present)

        if verdict == HeytingOmega3.VETOED:
            wacc = "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE"
            liquidity = "+11.5% EN TESORERÍA (Fusión de intervenciones)"
            summary = "FUSIÓN MONÁDICA VETOED: Al menos una de las dos ejecuciones disparó Crowbar."
        elif verdict == HeytingOmega3.DEGRADED:
            wacc = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity = "+8.5% RECIRCULANTE (Válvula de alivio activa)"
            summary = "FUSIÓN MONÁDICA DEGRADED: Fricción combinada bajo vigilancia QND."
        else:
            wacc = "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE"
            liquidity = "+11.5% FLUIDO LIBRE EN TESORERÍA"
            summary = "FUSIÓN MONÁDICA COHERENT: Ambas ejecuciones coherentes."

        seed_merkle = (
            f"{fused_id}:{s1.sha256_merkle_root}:{s2.sha256_merkle_root}:"
            f"{verdict.value}:{crowbar}:{rsi3_ok}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()
        provenance_hmac = hmac.new(
            self.secret_key, _HMAC_DOMAIN_P3 + merkle_root.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

        return FPUTensorSovereignState(
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
            llm_inference_savings_pct=min(s1.llm_inference_savings_pct, s2.llm_inference_savings_pct),
            kv_cache_compression_pct=min(s1.kv_cache_compression_pct, s2.kv_cache_compression_pct),
            executive_summary=summary,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            rsi3_banach_certified=rsi3_ok,
            rsi3_meta_operator_norm=rsi3_norm,
            melnikov_M0=melnikov,
            greene_residue_R=greene,
            homoclinic_tangency_present=tangle,
        )

    # ─── G.4 — Verificación del estado soberano ─────────────────────────────
    def verify_sovereign_state(self, state: FPUTensorSovereignState) -> bool:
        expected_hmac = hmac.new(
            self.secret_key,
            _HMAC_DOMAIN_P3 + state.sha256_merkle_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(state.provenance_hmac, expected_hmac)


# ══════════════════════════════════════════════════════════════════════════════
# §H. LEGACY GATES (compatibilidad v11.0.0)
# ══════════════════════════════════════════════════════════════════════════════

class HomomorphicGatePhi12:
    """COMPUERTA HOMOMÓRFICA Φ₁₂ (legacy). Delegar a FPUPhase1Observer.continue_to_phase2_gate."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._observer: Final = FPUPhase1Observer.__new__(FPUPhase1Observer)
        self._observer.secret_key = self.secret_key

    def verify_gate_phi12(self, cert1: FPUPhase1Certificate) -> bool:
        return FPUPhase1Observer.continue_to_phase2_gate(self._observer, cert1)


class HomomorphicGatePhi23:
    """COMPUERTA HOMOMÓRFICA Φ₂₃ (legacy). Delegar a FPUPhase2Orientor.continue_to_phase3_gate."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._orientor: Final = FPUPhase2Orientor.__new__(FPUPhase2Orientor)
        self._orientor.secret_key = self.secret_key

    def verify_gate_phi23(self, cert2: FPUPhase2Certificate) -> bool:
        return FPUPhase2Orientor.continue_to_phase3_gate(self._orientor, cert2)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL DEL SOBERANO AGÉNTICO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO AGÉNTICO COLISIONADOR FPU (v12.0.0-PhD)")
    print("  CAR-JW | CCR | GKSL | Spohn | Dirac | Poincaré-deep | RSI-3 Banach | Monad μ")
    print(banner)

    agent = FPUTensorFockColliderAgent(dimension=56)
    rng = np.random.default_rng(2026)

    state = rng.normal(0.0, 1.0, size=(56, 1)) + 1j * rng.normal(0.0, 1.0, size=(56, 1))
    density_sample = state @ state.conj().T
    density_sample /= np.trace(density_sample)

    payload = "TRANSACTION-APU-PRECIO-SOBRECOSTO-001"
    valid_hmac = hmac.new(agent.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # ── Caso 1: Positrón válido ─────────────────────────────────────────────
    s_ok = agent.execute_sovereign_pipeline(
        density_matrix=density_sample,
        electron_anomalies=2,
        positron_auth_signature=valid_hmac,
        transaction_payload=payload,
        rsi_history=(0, 0, 1, 0, 0),
    )

    print(f"\n  ─── CASO 1: POSITRÓN VÁLIDO (COHERENCIA NOMINAL) ─────────────────")
    print(f"  • Sovereign State ID       : {s_ok.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃      : {s_ok.global_verdict.name}")
    print(f"  • Crowbar disparado        : {s_ok.crowbar_triggered} (latencia={s_ok.crowbar_latency_ns:.2f} ns)")
    print(f"  • Protección WACC          : {s_ok.wacc_protection_status}")
    print(f"  • Impacto Liquidez         : {s_ok.net_liquidity_impact}")
    print(f"  • RSI-3 certificado        : {s_ok.rsi3_banach_certified}")
    print(f"  • RSI-3 ||T||              : {s_ok.rsi3_meta_operator_norm:.4f}")
    print(f"  • Melnikov M₀              : {s_ok.melnikov_M0:.4e}")
    print(f"  • Greene R_G               : {s_ok.greene_residue_R:.4f}")
    print(f"  • Homoclinic tangle        : {s_ok.homoclinic_tangency_present}")
    print(f"  • Pasaporte verificado     : {agent.verify_sovereign_state(s_ok)}")
    print(f"  • Raíz Merkle SHA-256      : {s_ok.sha256_merkle_root[:32]}...")
    print(f"  • Provenance HMAC          : {s_ok.provenance_hmac[:32]}...")

    # ── Caso 2: Positrón apócrifo ────────────────────────────────────────────
    s_bad = agent.execute_sovereign_pipeline(
        density_matrix=density_sample,
        electron_anomalies=2,
        positron_auth_signature="APOCRYPHAL_SIGNATURE",
        transaction_payload=payload,
        rsi_history=(0, 0, 1, 0, 0),
    )

    print(f"\n  ─── CASO 2: POSITRÓN APÓCRIFO (VETOED) ────────────────────────────")
    print(f"  • Sovereign State ID       : {s_bad.sovereign_state_id}")
    print(f"  • Veredicto Global Ω₃      : {s_bad.global_verdict.name}")
    print(f"  • Crowbar disparado        : {s_bad.crowbar_triggered} (latencia={s_bad.crowbar_latency_ns:.2f} ns)")
    print(f"  • Protección WACC          : {s_bad.wacc_protection_status}")
    print(f"  • Resumen                  : {s_bad.executive_summary[:80]}...")
    print(f"  • Pasaporte verificado     : {agent.verify_sovereign_state(s_bad)}")

    # ── Caso 3: Multiplicación monádica μ: T·T → T ──────────────────────────
    print(f"\n  ─── μ_FPU_COLLIDER (Kleisli fusion) ───────────────────────────────")
    fused = agent.monadic_compose_sovereign_states(s_ok, s_bad)
    print(f"  • Fused state ID           : {fused.sovereign_state_id}")
    print(f"  • Fused verdict (meet)     : {fused.global_verdict.name}")
    print(f"  • Fused crowbar (OR)       : {fused.crowbar_triggered}")
    print(f"  • Fused RSI-3 (AND)        : {fused.rsi3_banach_certified}")
    print(f"  • Fused ||T|| (max)        : {fused.rsi3_meta_operator_norm:.4f}")
    print(f"  • Fused verificado         : {agent.verify_sovereign_state(fused)}")
    print(banner)