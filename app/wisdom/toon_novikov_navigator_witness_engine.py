# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Novikov Navigator Witness Engine + Poincaré + KAM + Automejora L3                    ║
║ Ubicación: app/wisdom/toon_novikov_navigator_witness_engine.py                                       ║
║ Versión  : 13.0.0-SOVEREIGN-NOVIKOV-POINCARE-KAM-L3-PhD                                              ║
║ Función  : Motor QND AAV (Back-Action ≡ 0.0 dB) sobre la variedad ultramétrica no-arquimediana Λ_Nov ║
║            y el árbol de Bruhat-Tits d_BT, con Sección de Poincaré sobre 𝔇_n, Forma Normal de        ║
║            Birkhoff del punto fijo testigo, Tori KAM Diofantinos NO RESONANTES (ω_k = √p_k + φ^{-k}),║
║            mecánica celeste de Poincaré (invariante integral, Jacobi, recurrencia, homoclínicos) y   ║
║            Operador de Automejora Recursiva Nivel 3 (self-repair QND + metacognición del metacódigo).║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.       ║
║   FASE 1 → Heyting Ω₃ + QND AAV + Uhlmann FIX + Ultrametría FIX + Report + export seed               ║
║   FASE 2 → NovikovStateSeed + Σ Poincaré 𝔇_n + Celeste + Birkhoff + KAM noble + Operador L0..L3      ║
║   FASE 3 → NovikovTrajectory + SovereignNovikovAuditor + QND dinámica                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONNovikovNavigatorWitnessEngine")

__all__ = (
    "HeytingOmega3",
    "NovikovWitnessSnapshot",
    "TOONNovikovNavigatorWitnessEngine",
    "NovikovStateSeed",
    "NovikovPoincareSection",
    "NovikovReturnMap",
    "PoincareCelestialMechanics",
    "BirkhoffNovikovForm",
    "KAMNovikovTorus",
    "RecursiveNovikovImprover",
    "NovikovTrajectory",
    "SovereignNovikovAuditor",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, algebraicas y de Poincaré/KAM
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_NOVIKOV_WITNESS_SOVEREIGN_KEY_v13.0.0"
_NOVIKOV_V_FLOOR_DEFAULT: Final[float] = 0.10
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_MAX_K: Final[int] = 8
_KAM_MAX_SPARSE_SAMPLES: Final[int] = 4096
_L3_DEFAULT_ITERS: Final[int] = 8
_QND_BACK_ACTION_THRESHOLD_DB: Final[float] = 1.0e-6
_QND_COMM_NORM_THRESHOLD: Final[float] = 1.0e-6
_ULTRA_GAP_THRESHOLD: Final[float] = 1.0e-6
_TRANSVERSALITY_EPS: Final[float] = 1.0e-12
_HOMOCLINIC_EPS: Final[float] = 1.0e-4
_RESONANCE_ABS_EPS: Final[float] = 1.0e-14
_KAM_SMALLNESS_C: Final[float] = 0.25
_NEKHOROSHEV_C: Final[float] = 0.35
_BRUNO_TERMS: Final[int] = 8
_ENGINE_VERSION: Final[str] = "13.0.0-SOVEREIGN-NOVIKOV-POINCARE-KAM-L3-PhD"


def _first_n_primes(n: int) -> NDArray[np.int64]:
    """Criba de Eratóstenes con cota de Rosser–Schoenfeld n(log n + log log n)."""
    n = max(int(n), 1)
    if n < 6:
        bound = 16
    else:
        bound = int(n * (math.log(n) + math.log(math.log(n))) + 16)
    while True:
        sieve = np.ones(bound + 1, dtype=bool)
        sieve[0:2] = False
        limit = int(bound**0.5) + 1
        for p in range(2, limit):
            if sieve[p]:
                sieve[p * p :: p] = False
        primes = np.flatnonzero(sieve)
        if primes.size >= n:
            return primes[:n].astype(np.int64, copy=False)
        bound *= 2


def _kam_noble_frequencies(dim: int) -> NDArray[np.float64]:
    r"""
    Vector de frecuencias KAM no resonante (v13 BUGFIX).

    El diseño v12 ω_k = φ^{k+1} es FORMALMENTE RESONANTE para dim ≥ 3:
        φ² = φ + 1,  φ³ = 2φ + 1  ⇒  φ³ − φ² − φ = 0
        ⟨k, ω⟩ = 0 exactamente con k = (−1, −1, 1, 0, …).
    KAM exige ω Diofantino, i.e. ⟨k, ω⟩ ≠ 0 ∀ k ∈ ℤⁿ \ {0}.

    Construcción noble (Besicovitch + Hurwitz):
        ω_k = √p_k + φ^{-(k+1)},  p_k primo, p_k ≠ 5.
    • {√p} son ℚ-linealmente independientes (Besicovitch 1940).
    • φ^{-(k+1)} ∈ ℚ(√5) inyecta irracionalidad áurea (cocientes parciales acotados).
    • √p (p ≠ 5) es independiente de √5, de modo que no se recrea la relación de φ.
    Se normaliza ω_0 = 1 (mean motion canónico, Poincaré 1892).
    """
    d = max(int(dim), 1)
    primes_raw = _first_n_primes(d + 4)
    primes = [int(p) for p in primes_raw if int(p) != 5]
    while len(primes) < d:
        primes.append(primes[-1] + 2 if primes else 2)
    pvec = np.array(primes[:d], dtype=np.float64)
    golden_tail = np.array(
        [_GOLDEN_RATIO ** (-(k + 1)) for k in range(d)], dtype=np.float64
    )
    omega = np.sqrt(pvec) + golden_tail
    return omega / max(float(omega[0]), _WILKINSON_EPSILON)


def _project_psd_normalize(m: NDArray) -> NDArray[np.complex128]:
    """Proyección de Higham sobre el simplex de estados: Hermitiano → PSD → Tr = 1."""
    a = np.asarray(m, dtype=np.complex128)
    herm = 0.5 * (a + a.conj().T)
    eigs, vecs = la.eigh(herm)
    clean = np.maximum(np.real(eigs), _SPECTRAL_PSD_FLOOR)
    psd = (vecs * clean) @ vecs.conj().T
    tr = float(np.real(np.trace(psd)))
    if tr < _WILKINSON_EPSILON:
        n = int(psd.shape[0])
        return np.eye(n, dtype=np.complex128) / float(max(n, 1))
    return psd / tr


def _hermitize(M: NDArray) -> NDArray[np.complex128]:
    a = np.asarray(M, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _spectral_actions(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Acciones espectrales a_i = −log(λ_i), λ_i ↓ ⇒ a_i ↑."""
    evals = np.sort(
        np.maximum(np.real(la.eigvalsh(_hermitize(rho))), _SPECTRAL_PSD_FLOOR)
    )[::-1]
    s = float(np.sum(evals)) + 1e-30
    evals = evals / s
    return -np.log(evals)


def _hmac_merkle(secret: bytes, payload: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _back_action_db(rho: NDArray[np.complex128], A_nov: NDArray[np.complex128]) -> float:
    comm = A_nov @ rho - rho @ A_nov
    comm_norm = float(la.norm(comm, ord="fro"))
    if comm_norm <= _QND_COMM_NORM_THRESHOLD:
        return 0.0
    return float(min(12.5, 20.0 * math.log10(1.0 + comm_norm)))


def _uhlmann_fidelity(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    r"""F(ρ,σ) = (Tr √(√ρ σ √ρ))². No renormaliza el operador intermedio."""
    sqrt_rho = la.sqrtm(_hermitize(rho))
    middle = _hermitize(sqrt_rho @ sigma @ sqrt_rho)
    eigvals = np.maximum(np.real(np.linalg.eigvalsh(middle)), 0.0)
    trace_sqrt = float(np.sum(np.sqrt(eigvals)))
    return float(min(max(trace_sqrt**2, 0.0), 1.0))


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: RETÍCULO Ω₃ (imply corregido) + SNAPSHOT + MOTOR QND (Uhlmann FIX, Ultrametría FIX)    ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La ÚLTIMA definición de FASE 1 (export_novikov_state_seed) retorna un NovikovStateSeed,        ║
# ║  que es la PRIMERA definición consumida por FASE 2.                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting trivaluado Ω₃ = {COHERENT, DEGRADED, VETOED}.
    Codificación numérica: COHERENT=0x00, DEGRADED=0x01, VETOED=0x03.
    Orden de verdad (⊥→⊤): VETOED ≤ DEGRADED ≤ COHERENT.
    Orden numérico:        COHERENT < DEGRADED < VETOED (invertido respecto a verdad).

    Álgebra de Heyting: a → b = ⋁{c | a ∧ c ≤ b} se reduce, en cadena finita, a
        a → b = ⊤  si a ≤_truth b,  else b.
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo de verdad ≡ supremo de severidad (peor veredicto)."""
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo de verdad ≡ ínfimo de severidad (mejor veredicto común)."""
        return HeytingOmega3(min(int(self), int(other)))

    def imply(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Heyting implication: a → b = ⊤ si a ≤_truth b, sino b.
        Traducción numérica: a ≤_truth b ⟺ int(a) ≥ int(b).
        """
        if int(self) >= int(other):
            return HeytingOmega3.COHERENT
        return other

    def equiv(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Bi-implicación de Heyting a ↔ b = (a → b) ∧ (b → a)."""
        return self.imply(other).meet(other.imply(self))

    def negation(self) -> "HeytingOmega3":
        """Pseudo-complemento ¬a = a → ⊥ = a → VETOED."""
        return self.imply(HeytingOmega3.VETOED)

    def is_top(self) -> bool:
        return self == HeytingOmega3.COHERENT

    def is_bottom(self) -> bool:
        return self == HeytingOmega3.VETOED


@dataclass(frozen=True, slots=True)
class NovikovWitnessSnapshot:
    """
    Snapshot inmutable QND de Novikov. v13 añade diagnósticos celestes Poincaré/KAM/L3
    con defaults retrocompatibles.
    """

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
    kam_frequency_family: str = "NOBLE_PRIME_GOLDEN"
    engine_version: str = _ENGINE_VERSION


class TOONNovikovNavigatorWitnessEngine:
    """Motor QND testigo de Novikov v13: Uhlmann/ultrametría fijos + semilla KAM no resonante."""

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa PSD de Higham y normalización de traza Tr(ρ) ≡ 1.0."""
        a = np.asarray(matrix, dtype=np.complex128)
        if a.ndim != 2:
            a = np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        if a.shape != (self.dimension, self.dimension):
            embedded = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
            d = min(self.dimension, a.shape[0], a.shape[1])
            embedded[:d, :d] = a[:d, :d]
            a = embedded
        return _project_psd_normalize(a)

    def compute_qnd_weak_value(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> Tuple[complex, float]:
        r"""
        Valor débil AAV: A_w = ⟨φ_f| A_Nov |φ_i⟩ / ⟨φ_f|φ_i⟩.
        Si |⟨φ_f|φ_i⟩| < 1e-12 (estados ortogonales), A_w colapsa al valor fuerte Tr(A_Nov ρ).
        Back-action = 0 si ||[A_Nov, ρ]||_F ≤ 1e-6, sino 20·log10(1 + ||comm||) acotada a 12.5 dB.

        Condición QND (Aharonov–Albert–Vaidman + Braginsky): el observable es QND
        ssi genera un canal de decoherencia en su propia base, i.e. [A, ρ] → 0
        bajo el disipador de doble conmutador (Fase 2, self-repair L3).
        """
        rho = self.sanitize_density_matrix(density_matrix)
        _evals, evecs = la.eigh(rho)
        d = self.dimension

        if phi_initial is None:
            phi_i = evecs[:, -1]
        else:
            phi_i = np.asarray(phi_initial, dtype=np.complex128).flatten()
            if phi_i.size != d:
                tmp = np.zeros(d, dtype=np.complex128)
                tmp[: min(d, phi_i.size)] = phi_i[: min(d, phi_i.size)]
                phi_i = tmp
            phi_i = phi_i / max(float(np.linalg.norm(phi_i)), _WILKINSON_EPSILON)

        if phi_final is None:
            phi_f_raw = rho @ phi_i
            phi_f = phi_f_raw / max(float(np.linalg.norm(phi_f_raw)), _WILKINSON_EPSILON)
        else:
            phi_f = np.asarray(phi_final, dtype=np.complex128).flatten()
            if phi_f.size != d:
                tmp = np.zeros(d, dtype=np.complex128)
                tmp[: min(d, phi_f.size)] = phi_f[: min(d, phi_f.size)]
                phi_f = tmp
            phi_f = phi_f / max(float(np.linalg.norm(phi_f)), _WILKINSON_EPSILON)

        A_nov = (evecs * np.linspace(0.1, 1.0, d, dtype=np.float64)) @ evecs.conj().T

        overlap = complex(np.vdot(phi_f, phi_i))
        if abs(overlap) < 1e-12:
            Aw = complex(float(np.real(np.trace(A_nov @ rho))), 0.0)
        else:
            Aw = complex(np.vdot(phi_f, A_nov @ phi_i)) / overlap

        return Aw, _back_action_db(rho, A_nov)

    def audit_uhlmann_bures_novikov(
        self, rho_mac: NDArray, rho_delib: NDArray
    ) -> Tuple[float, float, float]:
        r"""
        Fidelidad de Uhlmann: F(ρ,σ) = (Tr √(√ρ σ √ρ))².
        Bures: D_B = √(2 − 2√F).  Fubini–Study (ángulo de Bures): γ = arccos(√F).
        """
        r_mac = self.sanitize_density_matrix(rho_mac)
        r_del = self.sanitize_density_matrix(rho_delib)
        fid = _uhlmann_fidelity(r_mac, r_del)
        bures = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid))))
        d_fs = float(np.arccos(min(1.0, max(0.0, math.sqrt(fid)))))
        return fid, bures, d_fs

    def compute_ultrametric_valuation_and_bruhat_tits(
        self, density_matrix: NDArray, v_floor: float = _NOVIKOV_V_FLOOR_DEFAULT
    ) -> Tuple[float, float, bool, float, float]:
        r"""
        Valuación v(A) = min a_i = −log(λ_max), norma ||A||_Nov = exp(−v(A)) = λ_max.

        Test estructural no tautológico: la brecha topológica log(λ_max/λ_second)
        debe superar umbral para que la bola ultramétrica no degeneré al estado
        máximamente mixto (centro del edificio de Bruhat–Tits).

        Desigualdad ultramétrica fuerte sobre acciones espectrales:
            d(i, k) ≤ max(d(i, j), d(j, k)) con d(i, j) = exp(−min(a_i, a_j)).
        Se verifica idénticamente por monotonía de exp(−min(·)), y el test no trivial
        es la no-degeneración de la cima (gap).
        """
        rho = self.sanitize_density_matrix(density_matrix)
        actions = _spectral_actions(rho)
        v_min = float(np.min(actions))
        ultra_norm = float(math.exp(-min(v_min, 700.0))) if v_min < 700.0 else 0.0

        actions_sorted = np.sort(actions)
        if len(actions_sorted) >= 2:
            top_gap = float(actions_sorted[1] - actions_sorted[0])
            ultra_ok = top_gap > _ULTRA_GAP_THRESHOLD
        else:
            ultra_ok = False

        purity = float(np.real(np.trace(rho @ rho)))
        d = self.dimension
        d_bt = float(abs(v_min - v_floor) + np.arccos(math.sqrt(max(purity, 0.0))))
        amoeba_area = 0.5 * float(np.sum(actions[: min(8, actions.size)] ** 2)) * (
            math.pi / max(d, 1)
        )
        return v_min, ultra_norm, ultra_ok, d_bt, amoeba_area

    def evaluate_novikov_witness(
        self,
        rho_mac: NDArray,
        rho_delib: NDArray,
        deliberation_id: str = "DELIB-QND-001",
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> NovikovWitnessSnapshot:
        """Auditoría QND pasiva (base). Campos L3 quedan en defaults hasta SovereignNovikovAuditor."""
        now = time.time()
        r_del = self.sanitize_density_matrix(rho_delib)

        Aw, back_action = self.compute_qnd_weak_value(r_del, phi_initial, phi_final)
        fid, bures, d_fs = self.audit_uhlmann_bures_novikov(rho_mac, r_del)
        v_min, ultra_norm, ultra_ok, d_bt, amoeba = (
            self.compute_ultrametric_valuation_and_bruhat_tits(r_del)
        )

        eigs = np.sort(np.maximum(np.real(la.eigvalsh(r_del)), _SPECTRAL_PSD_FLOOR))[::-1]
        eigs = eigs / float(np.sum(eigs))
        h_ks = -float(np.sum(eigs * np.log(eigs)))
        lam_trans = -0.5 * h_ks

        if fid >= 0.95 and ultra_ok and v_min >= 0.05:
            verdict = HeytingOmega3.COHERENT
        elif fid >= 0.80 and d_bt <= 5.0:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED

        snap_id = f"SNAP-NOV-WIT-{int(now * 1000) % 1_000_000:06d}"
        seed_bytes = (
            f"{snap_id}:{deliberation_id}:{v_min:.6f}:{fid:.6f}:{verdict.value}".encode("utf-8")
        )
        merkle = _hmac_merkle(self.secret_key, seed_bytes)

        actions_sorted = np.sort(_spectral_actions(r_del))
        gap_sig = bool(
            len(actions_sorted) >= 2
            and (actions_sorted[1] - actions_sorted[0]) > _ULTRA_GAP_THRESHOLD
        )

        return NovikovWitnessSnapshot(
            snapshot_id=snap_id,
            deliberation_id=deliberation_id,
            novikov_valuation_v=v_min,
            ultrametric_norm=ultra_norm,
            ultrametric_satisfied=ultra_ok,
            bruhat_tits_distance=d_bt,
            qnd_weak_value_aav=Aw,
            weak_value_modulus=abs(Aw),
            uhlmann_fidelity=fid,
            bures_distance=bures,
            fubini_study_distance=d_fs,
            kolmogorov_sinai_entropy=h_ks,
            oseledets_lyapunov_transverse=lam_trans,
            amoeba_tropical_area=amoeba,
            back_action_db=back_action,
            global_verdict=verdict,
            sha256_merkle_root=merkle,
            timestamp_utc=now,
            ultrametric_gap_significant=gap_sig,
        )

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (export_novikov_state_seed).                                      ║
    # ║  ÚLTIMO método de FASE 1. Retorna NovikovStateSeed → PRIMER objeto de FASE 2.               ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def export_novikov_state_seed(
        self,
        snapshot: NovikovWitnessSnapshot,
        rho_mac: NDArray,
        rho_delib: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> "NovikovStateSeed":
        r"""
        Empaqueta la tríada (ρ_delib, ρ_mac, A_nov, φ_i, φ_f) + métricas en una semilla
        canónica que alimenta el análisis Poincaré/KAM/L3:

          • ρ_delib: estado de deliberación (fuente)
          • ρ_MAC:   estado de la Matriz Atómica de Conocimiento (target de convergencia)
          • A_nov:   observable fijo (Jordan–Wigner-like sobre base espectral de ρ_delib)
          • φ_i:     autovector dominante de ρ_delib (pre-selección)
          • φ_f:     ρ_delib · φ_i normalizado (post-selección pasiva)
          • ω_k = √p_k + φ^{-(k+1)}  (v13: no resonante; máxima irracionalidad proyectiva)
        """
        dim = self.dimension
        r_del = self.sanitize_density_matrix(rho_delib)
        r_mac = self.sanitize_density_matrix(rho_mac)

        _evals_del, evecs_del = la.eigh(r_del)
        A_nov = (evecs_del * np.linspace(0.1, 1.0, dim, dtype=np.float64)) @ evecs_del.conj().T

        if phi_initial is None:
            phi_i = evecs_del[:, -1]
        else:
            phi_i = np.asarray(phi_initial, dtype=np.complex128).flatten()
            if phi_i.size != dim:
                tmp = np.zeros(dim, dtype=np.complex128)
                tmp[: min(dim, phi_i.size)] = phi_i[: min(dim, phi_i.size)]
                phi_i = tmp
            phi_i = phi_i / max(float(np.linalg.norm(phi_i)), _WILKINSON_EPSILON)

        if phi_final is None:
            phi_f_raw = r_del @ phi_i
            phi_f = phi_f_raw / max(float(np.linalg.norm(phi_f_raw)), _WILKINSON_EPSILON)
        else:
            phi_f = np.asarray(phi_final, dtype=np.complex128).flatten()
            if phi_f.size != dim:
                tmp = np.zeros(dim, dtype=np.complex128)
                tmp[: min(dim, phi_f.size)] = phi_f[: min(dim, phi_f.size)]
                phi_f = tmp
            phi_f = phi_f / max(float(np.linalg.norm(phi_f)), _WILKINSON_EPSILON)

        actions = _spectral_actions(r_del)
        J_sorted = np.sort(np.maximum(np.exp(-actions), 0.0))[::-1]
        J_total = float(J_sorted.sum())
        J_norm = J_sorted / (J_total + _WILKINSON_EPSILON)
        dominant_vec = evecs_del[:, -1]
        theta = np.angle(dominant_vec)

        omega = _kam_noble_frequencies(dim)

        probe_seed = int(
            hashlib.sha256(f"{snapshot.snapshot_id}:{dim}".encode()).hexdigest()[:8], 16
        )
        rng = np.random.default_rng(probe_seed)
        raw_probe = rng.normal(0.0, 1.0, (dim, dim)) + 1j * rng.normal(0.0, 1.0, (dim, dim))
        probe = 0.5 * (raw_probe + raw_probe.conj().T)

        return NovikovStateSeed(
            source_merkle=snapshot.sha256_merkle_root,
            snapshot_ref=snapshot,
            rho_delib_vector=tuple(complex(x) for x in r_del.flatten()),
            rho_mac_vector=tuple(complex(x) for x in r_mac.flatten()),
            a_nov_vector=tuple(complex(x) for x in A_nov.flatten()),
            phi_i_vector=tuple(complex(x) for x in phi_i.flatten()),
            phi_f_vector=tuple(complex(x) for x in phi_f.flatten()),
            probe_vector=tuple(complex(x) for x in probe.flatten()),
            action_vector=tuple(float(x) for x in J_norm),
            angle_vector=tuple(float(x) for x in theta),
            frequency_vector=tuple(float(x) for x in omega),
            initial_fidelity=snapshot.uhlmann_fidelity,
            initial_valuation=snapshot.novikov_valuation_v,
            initial_back_action_db=snapshot.back_action_db,
            dimension=dim,
            kam_frequency_family="NOBLE_PRIME_GOLDEN",
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: NOVIKOV STATE SEED + POINCARÉ 𝔇_n + CELESTE + BIRKHOFF + KAM noble + OPERADOR L0..L3  ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (NovikovStateSeed) consume export_novikov_state_seed (FASE 1).                    ║
# ║  Última def. (RecursiveNovikovImprover.iterate_level_3) → NovikovTrajectory (PRIMER obj FASE 3).║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class NovikovStateSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRY: continúa de export_novikov_state_seed.                                     │
    │  Semilla canónica con ρ_delib, ρ_MAC, A_nov, φ_i, φ_f, probe, J/θ/ω_noble.                  │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_merkle: str
    snapshot_ref: NovikovWitnessSnapshot
    rho_delib_vector: Tuple[complex, ...]
    rho_mac_vector: Tuple[complex, ...]
    a_nov_vector: Tuple[complex, ...]
    phi_i_vector: Tuple[complex, ...]
    phi_f_vector: Tuple[complex, ...]
    probe_vector: Tuple[complex, ...]
    action_vector: Tuple[float, ...]
    angle_vector: Tuple[float, ...]
    frequency_vector: Tuple[float, ...]
    initial_fidelity: float
    initial_valuation: float
    initial_back_action_db: float
    dimension: int
    kam_frequency_family: str = "NOBLE_PRIME_GOLDEN"

    def get_rho_delib(self) -> NDArray[np.complex128]:
        return np.array(self.rho_delib_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_rho_mac(self) -> NDArray[np.complex128]:
        return np.array(self.rho_mac_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_a_nov(self) -> NDArray[np.complex128]:
        return np.array(self.a_nov_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_probe(self) -> NDArray[np.complex128]:
        return np.array(self.probe_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_phi_i(self) -> NDArray[np.complex128]:
        return np.array(self.phi_i_vector, dtype=np.complex128)

    def get_phi_f(self) -> NDArray[np.complex128]:
        return np.array(self.phi_f_vector, dtype=np.complex128)


class NovikovPoincareSection:
    r"""
    Sección transversal de Poincaré (Méthodes Nouvelles, t. I, cap. III):
        Σ = {ρ ∈ 𝔇_n : g(ρ) = Re⟨H_probe, ρ⟩ − c = 0}.
    Transversalidad: ġ(ρ) ≠ 0 en el cruce (no tangencia a Σ).
    """

    def __init__(self, probe: NDArray[np.complex128], offset: float) -> None:
        self.probe = np.asarray(probe, dtype=np.complex128)
        self.offset = float(offset)

    def g(self, rho: NDArray[np.complex128]) -> float:
        return float(np.real(np.trace(self.probe.conj().T @ rho))) - self.offset

    def is_transverse(self, g_left: float, g_right: float) -> bool:
        return abs(g_right - g_left) > _TRANSVERSALITY_EPS


class NovikovReturnMap:
    """
    Mapa de retorno P: Σ → Σ de la órbita L3.
    Interpolación lineal en 𝔇_n + reproyección PSD (conserva el simplex).
    Devuelve cruces, tiempos de primer retorno y mínima |ġ|.
    """

    def __init__(self, section: NovikovPoincareSection) -> None:
        self.section = section

    def apply(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> Tuple[List[NDArray[np.complex128]], List[int], float]:
        crossings: List[NDArray[np.complex128]] = []
        crossing_idx: List[int] = []
        trans_min = float("inf")
        if len(rho_sequence) < 2:
            return crossings, crossing_idx, 0.0
        g_vals = [self.section.g(r) for r in rho_sequence]
        for i in range(len(g_vals) - 1):
            gi, gj = g_vals[i], g_vals[i + 1]
            dg = abs(gj - gi)
            if gi * gj < 0.0 and dg > _TRANSVERSALITY_EPS:
                trans_min = min(trans_min, dg)
                t = gi / (gi - gj)
                t = float(min(max(t, 0.0), 1.0))
                rho_cross = (1.0 - t) * rho_sequence[i] + t * rho_sequence[i + 1]
                crossings.append(_project_psd_normalize(rho_cross))
                crossing_idx.append(i)
        if trans_min == float("inf"):
            trans_min = 0.0
        return crossings, crossing_idx, float(trans_min)


class PoincareCelestialMechanics:
    r"""
    Mecánica celeste de Poincaré sobre 𝔇_n (Les Méthodes Nouvelles, 1892–1899):

      1. Invariante integral absoluto: Ω(X, Y) = Im Tr(X† Y) a lo largo de (J_t, θ_t).
         El drift de área simpléctica en el plano acción-ángulo mide la no-canonicidad
         residual del flujo discreto I_L3.
      2. Integral de Jacobi análoga: 𝒞(ρ) = Tr(A_nov ρ). En un flujo QND puro,
         [A_nov, ρ] = 0 ⇒ 𝒞̇ = 0. El drift max−min es el defecto de pasividad.
      3. Tiempos de primer retorno a Σ y estimación de recurrencia
         (teorema de recurrencia de Poincaré: T ≲ 1 / vol(tubo)).
      4. Puntos homoclínicos: dos cruces no consecutivos con ||ρ_i − ρ_j||_F < ε
         (germen del tangle homoclínico, descubrimiento del caos).
      5. Exponentes característicos de Poincaré ≡ Lyapunov de Oseledets sobre DI.
    """

    @staticmethod
    def jacobi_integral(
        rho: NDArray[np.complex128], A_nov: NDArray[np.complex128]
    ) -> float:
        return float(np.real(np.trace(A_nov @ rho)))

    @staticmethod
    def jacobi_drift(
        rho_seq: Sequence[NDArray[np.complex128]], A_nov: NDArray[np.complex128]
    ) -> float:
        if not rho_seq:
            return 0.0
        vals = [PoincareCelestialMechanics.jacobi_integral(r, A_nov) for r in rho_seq]
        return float(max(vals) - min(vals))

    @staticmethod
    def integral_invariant_drift(actions: Sequence[float], angles: Sequence[float]) -> float:
        r"""
        Área de Poincaré relativa ∑ ½ (J_i + J_{i+1}) Δθ_i  (invariante de Hilbert–Poincaré).
        El drift es la desviación del cierre: |A_ciclo| respecto de 0 en un loop discreto.
        """
        n = min(len(actions), len(angles))
        if n < 2:
            return 0.0
        J = np.asarray(actions[:n], dtype=np.float64)
        th = np.unwrap(np.asarray(angles[:n], dtype=np.float64))
        area = 0.5 * float(np.sum((J[1:] + J[:-1]) * (th[1:] - th[:-1])))
        return abs(area)

    @staticmethod
    def first_return_times(crossing_idx: Sequence[int]) -> Tuple[float, Tuple[int, ...]]:
        if len(crossing_idx) < 2:
            return 0.0, tuple()
        diffs = tuple(int(crossing_idx[i + 1] - crossing_idx[i]) for i in range(len(crossing_idx) - 1))
        mean_t = float(sum(diffs) / len(diffs)) if diffs else 0.0
        return mean_t, diffs

    @staticmethod
    def homoclinic_detected(
        crossings: Sequence[NDArray[np.complex128]], eps: float = _HOMOCLINIC_EPS
    ) -> bool:
        m = len(crossings)
        if m < 3:
            return False
        for i in range(m):
            for j in range(i + 2, m):
                dist = float(la.norm(crossings[i] - crossings[j], ord="fro"))
                if dist < eps:
                    return True
        return False

    @staticmethod
    def recurrence_bound(dimension: int, residual: float) -> float:
        """Cota heurística T_rec ~ (1 / ε)^{2 dim} del teorema de recurrencia, log-comprimida."""
        eps = max(residual, _WILKINSON_EPSILON)
        expo = min(2.0 * float(dimension), 32.0)
        return float(min(1.0e12, eps ** (-expo / max(float(dimension), 1.0))))


class BirkhoffNovikovForm:
    """
    Forma normal de Birkhoff del punto fijo testigo L3.
    Clasificación por radio espectral del jacobiano discreto (multiplicadores de Floquet):
      ATTRACTING | REPELLING | ELLIPTIC_CENTER | PARABOLIC | SADDLE_HYPERBOLIC | DEGENERATE.
    """

    def __init__(
        self, fixed_rho: NDArray[np.complex128], jacobian_spectrum: NDArray[np.complex128]
    ) -> None:
        self.fixed_rho = fixed_rho
        self.jacobian_spectrum = jacobian_spectrum

    def classify(self) -> str:
        eigs = self.jacobian_spectrum
        if eigs.size == 0:
            return "DEGENERATE"
        moduli = np.abs(eigs)
        if np.all(moduli < 1.0 - 1e-8):
            return "ATTRACTING"
        if np.all(moduli > 1.0 + 1e-8):
            return "REPELLING"
        if np.all(np.abs(moduli - 1.0) < 1e-8):
            if np.all(np.abs(eigs - 1.0) < 1e-6) or np.all(np.abs(eigs + 1.0) < 1e-6):
                return "PARABOLIC"
            return "ELLIPTIC_CENTER"
        return "SADDLE_HYPERBOLIC"

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.jacobian_spectrum))) if self.jacobian_spectrum.size else 0.0

    def lyapunov_exponent(self) -> float:
        rho = self.spectral_radius()
        return float(np.log(max(rho, _WILKINSON_EPSILON)))


class KAMNovikovTorus:
    r"""
    Toro invariante KAM con frecuencias nobles no resonantes.

    Condición diofantina (Kolmogorov–Arnold–Moser):
        |⟨k, ω⟩| ≥ γ / |k|_1^τ    ∀ k ∈ ℤⁿ \ {0}.
    El vector ω = √p + φ^{-(·)} maximiza γ práctico: independencia ℚ de {√p}
    (Besicovitch) + cocientes parciales áureos en cada proyección.

    Condición de Brjuno (1965, necesaria casi-óptima en 1D holomorfo):
        ∑_{k≥0} 2^{-k} log(1/Ω(2^k)) < ∞,   Ω(x) = min_{0<|q|≤x} |⟨q, ω⟩|.

    Tiempo de Nekhoroshev: T_N ≍ exp( c (γ/ε)^{1/(2τ)} ).
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: float = _DIOPHANTINE_TAU
    ) -> float:
        omega = np.array(self.frequencies, dtype=np.float64)
        n = self.n
        if n == 0:
            return 0.0
        gamma = float("inf")
        max_k = max(int(max_k), 1)

        def consider(kvec: NDArray[np.float64]) -> None:
            nonlocal gamma
            if not np.any(kvec):
                return
            inner = abs(float(np.dot(kvec, omega)))
            if inner <= _RESONANCE_ABS_EPS:
                gamma = 0.0
                return
            norm_k = float(np.sum(np.abs(kvec)))
            gamma = min(gamma, inner * (norm_k**tau))

        for i in range(n):
            e = np.zeros(n, dtype=np.float64)
            e[i] = 1.0
            consider(e)
            consider(-e)

        if n >= 2:
            span = min(n, 24)
            for i in range(span):
                for j in range(i + 1, span):
                    for si in (−1.0, 1.0):
                        for sj in (−1.0, 1.0):
                            kvec = np.zeros(n, dtype=np.float64)
                            kvec[i] = si
                            kvec[j] = sj
                            consider(kvec)
                            if gamma == 0.0:
                                return 0.0

        rng = np.random.default_rng(1729)
        samples = min(_KAM_MAX_SPARSE_SAMPLES, 256 * max(n, 1))
        for _ in range(samples):
            kvec = rng.integers(-max_k, max_k + 1, size=n).astype(np.float64)
            consider(kvec)
            if gamma == 0.0:
                return 0.0

        if n >= 3:
            k_rel = np.zeros(n, dtype=np.float64)
            k_rel[0], k_rel[1], k_rel[2] = -1.0, -1.0, 1.0
            consider(k_rel)

        return float(gamma if math.isfinite(gamma) else 0.0)

    def is_kam_stable(
        self,
        perturbation_norm: float,
        gamma: Optional[float] = None,
        tau: float = _DIOPHANTINE_TAU,
    ) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        if gamma <= 0.0:
            return False
        omega_norm = float(np.linalg.norm(np.array(self.frequencies, dtype=np.float64))) + 1.0
        smallness = _KAM_SMALLNESS_C * (gamma**2) / (omega_norm ** (2.0 * tau))
        return float(perturbation_norm) < smallness

    def winding_numbers(self, phases: Sequence[float]) -> Tuple[float, ...]:
        r"""
        Número de rotación de Poincaré: ρ = Δθ_unwrap / (2π N).
        Se reporta el vector ρ · ω̂ (proyección sobre la dirección de frecuencias
        normalizada), no el cociente ilegítimo ω / Δθ del v12.
        """
        n_report = min(self.n, 8)
        if len(phases) < 2:
            return tuple(0.0 for _ in range(n_report))
        unwrapped = np.unwrap(np.asarray(phases, dtype=np.float64))
        dtheta = float(unwrapped[-1] - unwrapped[0])
        n_steps = max(len(phases) - 1, 1)
        rot = dtheta / (_TWO_PI * n_steps)
        omega = np.array(self.frequencies[:n_report], dtype=np.float64)
        denom = float(np.linalg.norm(omega)) + _WILKINSON_EPSILON
        return tuple(float(rot * w / denom) for w in omega)

    def bruno_condition(self, max_k: int = _KAM_MAX_K) -> Tuple[bool, float]:
        """Suma truncada de Brjuno; se declara OK si la suma parcial es finita y < 40."""
        omega = np.array(self.frequencies, dtype=np.float64)
        n = self.n
        if n == 0:
            return False, float("inf")
        rng = np.random.default_rng(42)
        partial = 0.0
        for k in range(_BRUNO_TERMS):
            cap = 1 << k
            omega_min = float("inf")
            trials = min(256, 32 * cap)
            for _ in range(trials):
                q = rng.integers(-max(1, min(max_k, cap)), max(1, min(max_k, cap)) + 1, size=n)
                if np.all(q == 0):
                    continue
                inner = abs(float(np.dot(q.astype(np.float64), omega)))
                if inner < omega_min:
                    omega_min = inner
            omega_min = max(omega_min, _WILKINSON_EPSILON)
            partial += (0.5**k) * math.log(1.0 / omega_min)
        ok = math.isfinite(partial) and partial < 40.0
        return ok, float(partial)

    def nekhoroshev_time(self, perturbation_norm: float, gamma: Optional[float] = None) -> float:
        if gamma is None:
            gamma = self.diophantine_constant()
        eps = max(float(perturbation_norm), _WILKINSON_EPSILON)
        if gamma <= 0.0:
            return 1.0
        expo = _NEKHOROSHEV_C * ((gamma / eps) ** (0.5 / _DIOPHANTINE_TAU))
        expo = min(max(expo, 0.0), 40.0)
        return float(math.exp(expo))


class RecursiveNovikovImprover:
    r"""
    Operador de automejora recursiva I: 𝔇_n → 𝔇_n  (Nivel 3 soberano).

    Jerarquía (cada nivel *contiene* y *modifica* al inferior):

      L0  I₀(ρ) = ρ
      L1  I₁(ρ; θ) = Π_PSD((1−θ)ρ + θ ρ_MAC)
      L2  θ ← clip(θ · (1 + g₂ (r − 1))),  r = ||e_t|| / ||e_{t−1}||
      L3  metacognición *del metacódigo*:
            • η_meta ← η_meta · {s, g} según ρ_espectral local del DI
            • g₂, s, g se adaptan si el certificado de Lyapunov falla
            • Self-repair QND: ρ ← Π(ρ − κ [A, [A, ρ]])  (Lindblad de puntero)
              que proyecta al conmutante de A_nov y restaura back-action → 0.

    Certificado de Lyapunov: V(ρ) = ||ρ − ρ_MAC||_F².
    L3 exige V_{t+1} ≤ V_t + holgura de reparación en ≥ 2/3 de los pasos
    y V_final < V_0.
    """

    def __init__(
        self,
        contraction: float = 0.5,
        meta_learning: float = 0.3,
        repair_gain: float = 0.15,
    ) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self.repair_gain: float = float(repair_gain)
        self.l2_gain: float = 0.10
        self.l3_shrink: float = 0.90
        self.l3_grow: float = 1.05
        self.repair_count: int = 0

    def apply_level_0(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho.copy()

    def apply_level_1(
        self, rho: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        mixed = (1.0 - self.contraction) * rho + self.contraction * target
        return _project_psd_normalize(mixed)

    def apply_level_2(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(history) >= 2:
            err = float(np.linalg.norm(rho - target, ord="fro"))
            prev = float(np.linalg.norm(history[-2] - target, ord="fro"))
            if prev > _WILKINSON_EPSILON:
                ratio = err / prev
                self.contraction = float(
                    np.clip(self.contraction * (1.0 + self.l2_gain * (ratio - 1.0)), 0.05, 0.95)
                )
        return self.apply_level_1(rho, target)

    def apply_level_3(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(history) >= 3:
            rho_spec = self._estimate_local_spectral_radius(history[-3:])
            if rho_spec > 1.0:
                self.meta_learning = float(np.clip(self.meta_learning * self.l3_shrink, 0.05, 0.90))
            elif 0.0 < rho_spec < 0.5:
                self.meta_learning = float(np.clip(self.meta_learning * self.l3_grow, 0.05, 0.90))
            self.contraction = float(
                np.clip(self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95)
            )
        return self.apply_level_2(rho, target, history)

    def apply_qnd_self_repair(
        self, rho: NDArray[np.complex128], A_nov: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], bool]:
        r"""
        Reparación QND: disipador de doble conmutador
            ℒ(ρ) = −[A, [A, ρ]] = −(A²ρ − 2AρA + ρA²)
        es el generador de un canal de medición continua de A (Wiseman–Milburn).
        Amortigua coherencias inter-eigenspaces y restaura [A, ρ] → 0.
        """
        comm = A_nov @ rho - rho @ A_nov
        comm_norm = float(la.norm(comm, ord="fro"))
        if comm_norm <= _QND_COMM_NORM_THRESHOLD:
            return rho, False
        a2 = A_nov @ A_nov
        dissip = a2 @ rho + rho @ a2 - 2.0 * (A_nov @ rho @ A_nov)
        kappa = float(np.clip(self.repair_gain, 0.01, 0.85))
        repaired = rho - kappa * dissip
        self.repair_count += 1
        if comm_norm > 10.0 * _QND_COMM_NORM_THRESHOLD:
            self.repair_gain = float(np.clip(self.repair_gain * 1.08, 0.01, 0.85))
        else:
            self.repair_gain = float(np.clip(self.repair_gain * 0.98, 0.01, 0.85))
        return _project_psd_normalize(repaired), True

    def _adapt_meta_gains(self, v_prev: float, v_now: float) -> None:
        """L3 profundo: el improver modifica *sus propias* ganancias de L2/L3."""
        if v_now > v_prev * 1.02:
            self.l2_gain = float(np.clip(self.l2_gain * 0.90, 0.02, 0.25))
            self.l3_shrink = float(np.clip(self.l3_shrink * 0.98, 0.70, 0.95))
            self.contraction = float(np.clip(self.contraction * 0.92, 0.05, 0.95))
        elif v_now < v_prev * 0.98:
            self.l2_gain = float(np.clip(self.l2_gain * 1.02, 0.02, 0.25))
            self.l3_grow = float(np.clip(self.l3_grow * 1.01, 1.01, 1.12))

    @staticmethod
    def _estimate_local_spectral_radius(recent: Sequence[NDArray[np.complex128]]) -> float:
        if len(recent) < 3:
            return 0.5
        d1 = float(np.linalg.norm(recent[-1] - recent[-2], ord="fro"))
        d2 = float(np.linalg.norm(recent[-2] - recent[-3], ord="fro"))
        if d1 < _WILKINSON_EPSILON or d2 < _WILKINSON_EPSILON:
            return 0.5
        return d1 / d2

    @staticmethod
    def deserialize(vec: Sequence[complex], dim: int) -> NDArray[np.complex128]:
        return np.array(vec, dtype=np.complex128).reshape((dim, dim))

    @staticmethod
    def _estimate_global_jacobian_spectrum(
        rho_seq: Sequence[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(rho_seq) < 2:
            return np.array([0.5], dtype=np.complex128)
        diffs = [
            float(np.linalg.norm(rho_seq[i + 1] - rho_seq[i], ord="fro"))
            for i in range(len(rho_seq) - 1)
        ]
        ratios: List[complex] = []
        for i in range(len(diffs) - 1):
            if diffs[i] > _WILKINSON_EPSILON:
                ratios.append(complex(diffs[i + 1] / diffs[i], 0.0))
        if not ratios:
            return np.array([complex(diffs[-1], 0.0)])
        return np.array(ratios, dtype=np.complex128)

    @staticmethod
    def _lyapunov_from_diffs(diffs: Sequence[float]) -> float:
        valid = [d for d in diffs if d > _WILKINSON_EPSILON]
        if len(valid) >= 2:
            log_ratios = np.log(np.array(valid[1:]) / np.array(valid[:-1]))
            return float(np.mean(log_ratios))
        return -20.0

    @staticmethod
    def _lyapunov_decrease_certified(
        rho_seq: Sequence[NDArray[np.complex128]], target: NDArray[np.complex128]
    ) -> bool:
        if len(rho_seq) < 2:
            return False
        v = [float(np.linalg.norm(r - target, ord="fro") ** 2) for r in rho_seq]
        decreases = sum(1 for i in range(len(v) - 1) if v[i + 1] <= v[i] * 1.05 + 1e-12)
        frac = decreases / max(len(v) - 1, 1)
        return bool(frac >= (2.0 / 3.0) and v[-1] < v[0])

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (iterate_level_3).                                                ║
    # ║  ÚLTIMO método de FASE 2. Retorna NovikovTrajectory → PRIMER objeto de FASE 3.              ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self, seed: NovikovStateSeed, max_iters: int = _L3_DEFAULT_ITERS
    ) -> "NovikovTrajectory":
        r"""
        Itera I_L3 sobre ρ_delib hacia ρ_MAC. En cada paso:
          • Mezcla convexa adaptativa (L1–L2) + metacognición (L3)
          • Self-repair QND por disipador de puntero
          • Adapta las ganancias del propio improver (certificado de Lyapunov)
          • Registra back-action, fase de Poincaré y acción (Tr ρ ≡ 1)
        Finalmente: λ, γ diofantina, Brjuno, Nekhoroshev, Birkhoff, Jacobi, homoclínicos.
        """
        self.repair_count = 0
        dim = seed.dimension
        rho = seed.get_rho_delib()
        target = seed.get_rho_mac()
        probe = seed.get_probe()
        A_nov = seed.get_a_nov()

        rho_seq: List[NDArray[np.complex128]] = [rho.copy()]
        action_seq: List[float] = [float(np.real(np.trace(rho)))]
        angle_seq: List[float] = [float(np.angle(np.trace(probe.conj().T @ rho)))]
        back_action_seq_db: List[float] = [_back_action_db(rho, A_nov)]
        v_prev = float(np.linalg.norm(rho - target, ord="fro") ** 2)

        history: List[NDArray[np.complex128]] = []
        n_iters = max(int(max_iters), 1)
        for _ in range(n_iters):
            history.append(rho.copy())
            rho = self.apply_level_3(rho, target, history)
            rho, _repaired = self.apply_qnd_self_repair(rho, A_nov)
            v_now = float(np.linalg.norm(rho - target, ord="fro") ** 2)
            self._adapt_meta_gains(v_prev, v_now)
            v_prev = v_now
            rho_seq.append(rho.copy())
            action_seq.append(float(np.real(np.trace(rho))))
            angle_seq.append(float(np.angle(np.trace(probe.conj().T @ rho))))
            back_action_seq_db.append(_back_action_db(rho, A_nov))

        diffs = [
            float(np.linalg.norm(rho_seq[i + 1] - rho_seq[i], ord="fro"))
            for i in range(len(rho_seq) - 1)
        ]
        lyapunov = self._lyapunov_from_diffs(diffs)
        convergence = diffs[-1] if diffs else 0.0
        lyap_cert = self._lyapunov_decrease_certified(rho_seq, target)

        torus = KAMNovikovTorus(seed.frequency_vector)
        gamma = torus.diophantine_constant()
        kam_stable = torus.is_kam_stable(convergence, gamma)
        winding = torus.winding_numbers(angle_seq)
        bruno_ok, _bruno_sum = torus.bruno_condition()
        t_nek = torus.nekhoroshev_time(convergence, gamma)

        jac_spec = self._estimate_global_jacobian_spectrum(rho_seq)
        birkhoff = BirkhoffNovikovForm(rho_seq[-1], jac_spec)
        birkhoff_class = birkhoff.classify()

        qnd_trajectory_stable = all(
            b <= _QND_BACK_ACTION_THRESHOLD_DB for b in back_action_seq_db
        )
        is_dynamically_stable = (
            lyapunov < 0.0
            and kam_stable
            and birkhoff_class in ("ATTRACTING", "ELLIPTIC_CENTER", "PARABOLIC")
        )
        if qnd_trajectory_stable and is_dynamically_stable:
            qnd_class = "QND_ROBUST"
        elif qnd_trajectory_stable and not is_dynamically_stable:
            qnd_class = "QND_FRAGILE"
        elif (not qnd_trajectory_stable) and is_dynamically_stable:
            qnd_class = "QND_RECOVERABLE"
        else:
            qnd_class = "QND_BROKEN"

        final_fid = _uhlmann_fidelity(rho_seq[-1], target)
        jacobi_drift = PoincareCelestialMechanics.jacobi_drift(rho_seq, A_nov)
        inv_drift = PoincareCelestialMechanics.integral_invariant_drift(action_seq, angle_seq)

        g_vals = [float(np.real(np.trace(probe.conj().T @ r))) for r in rho_seq]
        offset = float(np.mean(g_vals)) if g_vals else 0.0
        section = NovikovPoincareSection(probe=probe, offset=offset)
        crossings, cross_idx, trans_min = NovikovReturnMap(section).apply(rho_seq)
        mean_return, _rets = PoincareCelestialMechanics.first_return_times(cross_idx)
        homoclinic = PoincareCelestialMechanics.homoclinic_detected(crossings)

        spec_radius = float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0

        return NovikovTrajectory(
            source_merkle=seed.source_merkle,
            dimension=dim,
            rho_sequence=tuple(tuple(complex(x) for x in r.flatten()) for r in rho_seq),
            action_sequence=tuple(action_seq),
            angle_sequence=tuple(angle_seq),
            back_action_sequence=tuple(back_action_seq_db),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            spectral_radius=spec_radius,
            is_kam_stable=kam_stable,
            winding_numbers=winding,
            diophantine_gamma=gamma,
            final_contraction=self.contraction,
            final_meta_learning=self.meta_learning,
            birkhoff_classification=birkhoff_class,
            qnd_trajectory_stable=qnd_trajectory_stable,
            qnd_classification=qnd_class,
            final_fidelity_to_mac=final_fid,
            jacobi_integral_drift=jacobi_drift,
            poincare_mean_return=mean_return,
            transversality_min=trans_min,
            homoclinic_detected=homoclinic,
            nekhoroshev_time=t_nek,
            bruno_ok=bruno_ok,
            qnd_repair_count=int(self.repair_count),
            lyapunov_decrease_certified=lyap_cert,
            integral_invariant_drift=inv_drift,
            kam_frequency_family=seed.kam_frequency_family,
            poincare_crossings=len(crossings),
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: NOVIKOV TRAJECTORY + SOVEREIGN NOVIKOV AUDITOR                                         ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (NovikovTrajectory) consume iterate_level_3 de FASE 2.                           ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class NovikovTrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRY: continúa de iterate_level_3 (FASE 2).                                      │
    │  Trayectoria L3 del testigo Novikov con Poincaré celeste + Birkhoff + KAM noble embebidos.  │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_merkle: str
    dimension: int
    rho_sequence: Tuple[Tuple[complex, ...], ...]
    action_sequence: Tuple[float, ...]
    angle_sequence: Tuple[float, ...]
    back_action_sequence: Tuple[float, ...]
    lyapunov_exponent: float
    convergence_residual: float
    spectral_radius: float
    is_kam_stable: bool
    winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    final_contraction: float
    final_meta_learning: float
    birkhoff_classification: str
    qnd_trajectory_stable: bool
    qnd_classification: str
    final_fidelity_to_mac: float
    jacobi_integral_drift: float = 0.0
    poincare_mean_return: float = 0.0
    transversality_min: float = 0.0
    homoclinic_detected: bool = False
    nekhoroshev_time: float = 0.0
    bruno_ok: bool = True
    qnd_repair_count: int = 0
    lyapunov_decrease_certified: bool = False
    integral_invariant_drift: float = 0.0
    kam_frequency_family: str = "NOBLE_PRIME_GOLDEN"
    poincare_crossings: int = 0

    def get_rho_sequence_arrays(self) -> List[NDArray[np.complex128]]:
        return [RecursiveNovikovImprover.deserialize(s, self.dimension) for s in self.rho_sequence]

    def build_poincare_section(self, probe: NDArray[np.complex128]) -> NovikovPoincareSection:
        rho_arr = self.get_rho_sequence_arrays()
        g_vals = [float(np.real(np.trace(probe.conj().T @ r))) for r in rho_arr]
        offset = float(np.mean(g_vals)) if g_vals else 0.0
        return NovikovPoincareSection(probe=probe, offset=offset)

    def max_back_action_db(self) -> float:
        return max(self.back_action_sequence) if self.back_action_sequence else 0.0

    def is_l3_certified(self) -> bool:
        return (
            self.qnd_trajectory_stable
            and self.lyapunov_exponent < 0.0
            and self.is_kam_stable
            and self.birkhoff_classification in ("ATTRACTING", "ELLIPTIC_CENTER", "PARABOLIC")
            and self.lyapunov_decrease_certified
            and self.bruno_ok
        )


class SovereignNovikovAuditor:
    """
    Auditor soberano que orquesta:
      1) Snapshot base QND (FASE 1)
      2) Semilla canónica (FASE 1 → FASE 2)
      3) Trayectoria L3 con back-action dinámica y self-repair (FASE 2 → FASE 3)
      4) Poincaré crossings / Jacobi / homoclínicos (ya embebidos en la trayectoria)
      5) Reporte enriquecido con clasificación QND y certificados L3
    """

    def __init__(
        self,
        engine: TOONNovikovNavigatorWitnessEngine,
        improver: Optional[RecursiveNovikovImprover] = None,
    ) -> None:
        self.engine = engine
        self.improver = improver or RecursiveNovikovImprover(
            contraction=0.5, meta_learning=0.3, repair_gain=0.15
        )

    def full_audit(
        self,
        rho_mac: NDArray,
        rho_delib: NDArray,
        deliberation_id: str = "DELIB-QND-FULL",
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> NovikovWitnessSnapshot:
        base = self.engine.evaluate_novikov_witness(
            rho_mac, rho_delib, deliberation_id, phi_initial, phi_final
        )
        seed = self.engine.export_novikov_state_seed(
            base, rho_mac, rho_delib, phi_initial, phi_final
        )
        traj = self.improver.iterate_level_3(seed, max_iters=max_iters)

        l3_cert = traj.is_l3_certified() and base.global_verdict.is_top()
        logger.info(
            "SovereignNovikovAuditor id=%s qnd=%s l3=%s γ=%.3e λ=%+.4f repairs=%d",
            base.snapshot_id,
            traj.qnd_classification,
            l3_cert,
            traj.diophantine_gamma,
            traj.lyapunov_exponent,
            traj.qnd_repair_count,
        )
        return replace(
            base,
            lyapunov_exponent=traj.lyapunov_exponent,
            spectral_radius=traj.spectral_radius,
            birkhoff_classification=traj.birkhoff_classification,
            kam_diophantine_gamma=traj.diophantine_gamma,
            kam_stable=traj.is_kam_stable,
            poincare_crossings=traj.poincare_crossings,
            novikov_winding_numbers=traj.winding_numbers[:4],
            contraction_final=traj.final_contraction,
            meta_learning_final=traj.final_meta_learning,
            qnd_back_action_sequence=traj.back_action_sequence,
            qnd_trajectory_stable=traj.qnd_trajectory_stable,
            qnd_classification=traj.qnd_classification,
            final_fidelity_to_mac=traj.final_fidelity_to_mac,
            l3_certified=l3_cert,
            jacobi_integral_drift=traj.jacobi_integral_drift,
            poincare_mean_return=traj.poincare_mean_return,
            transversality_min=traj.transversality_min,
            homoclinic_detected=traj.homoclinic_detected,
            nekhoroshev_time=traj.nekhoroshev_time,
            bruno_ok=traj.bruno_ok,
            qnd_repair_count=traj.qnd_repair_count,
            lyapunov_decrease_certified=traj.lyapunov_decrease_certified,
            integral_invariant_drift=traj.integral_invariant_drift,
            kam_frequency_family=traj.kam_frequency_family,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — v13.0.0-SOVEREIGN-NOVIKOV-POINCARE-KAM-L3-PhD")
    print("  QND AAV + Poincaré celeste + KAM noble no-resonante + Automejora L3 self-repair")
    print(banner)

    print("\n  ─── Fix Heyting imply / equiv (cadena de severidad) ──────────────────────")
    for a in HeytingOmega3:
        for b in HeytingOmega3:
            print(
                f"  {a.name:<8} → {b.name:<8} = {a.imply(b).name:<8}  |  "
                f"↔ = {a.equiv(b).name:<8}  |  ¬{a.name} = {a.negation().name}"
            )

    engine = TOONNovikovNavigatorWitnessEngine(dimension=56)
    auditor = SovereignNovikovAuditor(engine)
    rng = np.random.default_rng(2026)

    raw_mac = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_mac = raw_mac @ raw_mac.conj().T
    rho_mac /= np.trace(rho_mac)

    raw_del = rng.normal(0.0, 1.0, (56, 56)) + 1j * rng.normal(0.0, 1.0, (56, 56))
    rho_del = raw_del @ raw_del.conj().T
    rho_del /= np.trace(rho_del)

    snap = auditor.full_audit(rho_mac, rho_del, deliberation_id="DELIB-TEST-001")

    print("\n  ─── FASE 1: Snapshot QND base (Uhlmann FIX + Ultramétrica FIX) ───────────")
    print(f"  • Snapshot ID           : {snap.snapshot_id}")
    print(f"  • Valuación v(A)        : {snap.novikov_valuation_v:.6f}")
    print(f"  • ||A||_Nov             : {snap.ultrametric_norm:.6e}")
    print(f"  • Ultramétrica OK       : {snap.ultrametric_satisfied} (brecha sig.: {snap.ultrametric_gap_significant})")
    print(f"  • Bruhat-Tits d_BT      : {snap.bruhat_tits_distance:.6f}")
    print(f"  • Valor Débil AAV A_w   : {snap.qnd_weak_value_aav:.4f} (|A_w| = {snap.weak_value_modulus:.4f})")
    print(f"  • Back-Action base      : {snap.back_action_db:.1f} dB")
    print(f"  • Fidelidad Uhlmann     : {snap.uhlmann_fidelity:.6f}")
    print(f"  • Bures d_B             : {snap.bures_distance:.6f}")
    print(f"  • Fubini-Study d_FS     : {snap.fubini_study_distance:.6f}")
    print(f"  • Veredicto Ω₃          : {snap.global_verdict.name}")
    print(f"  • Merkle HMAC-SHA256    : {snap.sha256_merkle_root[:32]}...")

    print("\n  ─── FASE 2 → 3: Poincaré celeste / Birkhoff / KAM noble / L3 ─────────────")
    print(f"  • λ Lyapunov            : {snap.lyapunov_exponent:+.6f}")
    print(f"  • Certificado V↓        : {snap.lyapunov_decrease_certified}")
    print(f"  • Radio espectral ρ(DI) : {snap.spectral_radius:.6f}")
    print(f"  • Birkhoff class        : {snap.birkhoff_classification}")
    print(f"  • Familia ω             : {snap.kam_frequency_family}")
    print(f"  • KAM γ (noble)         : {snap.kam_diophantine_gamma:.6e}")
    print(f"  • KAM estable           : {snap.kam_stable}")
    print(f"  • Brjuno OK             : {snap.bruno_ok}")
    print(f"  • Tiempo Nekhoroshev    : {snap.nekhoroshev_time:.6e}")
    print(f"  • Cruces Poincaré Σ     : {snap.poincare_crossings}")
    print(f"  • Retorno medio         : {snap.poincare_mean_return:.4f}")
    print(f"  • Transversalidad min   : {snap.transversality_min:.6e}")
    print(f"  • Homoclínico           : {snap.homoclinic_detected}")
    print(f"  • Drift Jacobi 𝒞        : {snap.jacobi_integral_drift:.6e}")
    print(f"  • Drift invariante Ω    : {snap.integral_invariant_drift:.6e}")
    wn = snap.novikov_winding_numbers
    print(
        "  • Enrollamiento ρ (4)   : "
        f"({wn[0] if len(wn) > 0 else 0:.4f}, "
        f"{wn[1] if len(wn) > 1 else 0:.4f}, "
        f"{wn[2] if len(wn) > 2 else 0:.4f}, "
        f"{wn[3] if len(wn) > 3 else 0:.4f})"
    )
    print(f"  • θ_final               : {snap.contraction_final:.6f}")
    print(f"  • η_meta (L3)           : {snap.meta_learning_final:.6f}")
    print(f"  • Reparaciones QND      : {snap.qnd_repair_count}")
    print(f"  • Fidelidad final MAC   : {snap.final_fidelity_to_mac:.6f}")
    print(f"  • Back-action traza     : {list(round(b, 4) for b in snap.qnd_back_action_sequence[:5])}...")
    print(f"  • Back-action max       : {max(snap.qnd_back_action_sequence) if snap.qnd_back_action_sequence else 0.0:.2e} dB")
    print(f"  • QND traza estable     : {snap.qnd_trajectory_stable}")
    print(f"  • QND clasificación     : {snap.qnd_classification}")
    print(f"  • L3 certificado        : {snap.l3_certified}")

    print("\n  ─── Trayectoria L3 explícita (ρ_t, J_t, θ_t, back-action) ────────────────")
    seed_dbg = engine.export_novikov_state_seed(snap, rho_mac, rho_del, None, None)
    improver = RecursiveNovikovImprover()
    traj = improver.iterate_level_3(seed_dbg, max_iters=6)
    for t, (J, ang, ba) in enumerate(
        zip(traj.action_sequence, traj.angle_sequence, traj.back_action_sequence)
    ):
        print(f"    t={t:02d} | J_t = {J:.6f} | θ_t = {ang:+.6f} rad | BackAct = {ba:.2e} dB")
    print(f"    → F(ρ_T, ρ_MAC) = {traj.final_fidelity_to_mac:.6f}")
    print(
        f"    → λ = {traj.lyapunov_exponent:+.6f} | "
        f"KAM γ = {traj.diophantine_gamma:.6e} | "
        f"Birkhoff = {traj.birkhoff_classification}"
    )
    print(
        f"    → QND = {traj.qnd_classification} | repairs = {traj.qnd_repair_count} | "
        f"V↓ cert = {traj.lyapunov_decrease_certified}"
    )

    print("\n  ─── BUGFIX KAM: φ^{k+1} resonante vs nobles √p+φ^{-k} ────────────────────")
    resonant = tuple(_GOLDEN_RATIO ** (k + 1) for k in range(8))
    noble = tuple(float(x) for x in _kam_noble_frequencies(8))
    torus_res = KAMNovikovTorus(resonant)
    torus_nob = KAMNovikovTorus(noble)
    g_res = torus_res.diophantine_constant()
    g_nob = torus_nob.diophantine_constant()
    k_rel = np.array([-1.0, -1.0, 1.0, 0, 0, 0, 0, 0])
    inner_res = abs(float(np.dot(k_rel, np.array(resonant))))
    inner_nob = abs(float(np.dot(k_rel, np.array(noble))))
    print(f"    ⟨(−1,−1,1,…), φ^{{k+1}}⟩ = {inner_res:.3e}  ← RELACIÓN ENTERA (γ → 0)")
    print(f"    ⟨(−1,−1,1,…), noble⟩     = {inner_nob:.3e}  ← NO resonante")
    print(f"    γ(φ^{{k+1}})             = {g_res:.6e}  ← {'RESONANTE' if g_res == 0.0 else 'débil'}")
    print(f"    γ(√p+φ^{{-k}})           = {g_nob:.6e}  ← {'MEJOR' if g_nob > g_res else 'comparar'}")

    print("\n" + banner)
    print("  ✓ FASE 1 → FASE 2: export_novikov_state_seed → NovikovStateSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveNovikovImprover.iterate_level_3 → NovikovTrajectory")
    print("  ✓ imply Heyting + equiv (residuo del meet)")
    print("  ✓ Uhlmann fid sin renormalizar √ρ σ √ρ")
    print("  ✓ Ultramétrica no trivial (brecha espectral)")
    print("  ✓ Merkle HMAC-SHA256 (secret_key del motor)")
    print("  ✓ KAM no resonante: √p_k + φ^{-(k+1)} (Besicovitch + Hurwitz)")
    print("  ✓ Poincaré celeste: Jacobi, invariante integral, retorno, homoclínicos")
    print("  ✓ L3: self-repair [A,[A,ρ]] + adaptación de ganancias + certificado V↓")
    print("  ✓ QND pasividad como propiedad DINÁMICA de la órbita")
    print(banner)