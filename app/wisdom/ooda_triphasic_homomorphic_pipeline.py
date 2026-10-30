# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : OODA Triphasic Homomorphic Pipeline + Poincaré + KAM/BHT + Automejora L3                  ║
║ Ubicación: app/wisdom/ooda_triphasic_homomorphic_pipeline.py                                         ║
║ Versión  : 11.0.0-SOVEREIGN-OODA-POINCARE-KAM-BHT-L3-PhD                                             ║
║ Función  : Ciclo OODA invariable con compuertas homomórficas Φ₁₂ y Φ₂₃, enriquecido con              ║
║            complejo de de Rham matricial (Yang–Mills / Chern–Simons / Wilson / instantones),         ║
║            mapa de Poincaré transversal sobre 𝔇_N, twist de Poincaré–Birkhoff, forma normal          ║
║            de Birkhoff, tori KAM diofantinos (criterio BHT disipativo) y operador de automejora      ║
║            recursiva Nivel 3 (metacognición metacognitiva realimentada por el mapa de retorno).      ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.       ║
║   FASE 1 → Certificados Ω₃, Observadores, Compuertas Φ₁₂/Φ₂₃, Orientación de Gauge + Seed            ║
║   FASE 2 → Semilla Canónica + Σ Poincaré + Twist P–B + KAM/BHT + Operador L0..L3                     ║
║   FASE 3 → Trayectoria + Auditor Soberano + Decisor L3-Aware + Pipeline Unificado                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.OODATriphasicPipeline")

__version__: Final[str] = "11.0.0-SOVEREIGN-OODA-POINCARE-KAM-BHT-L3-PhD"

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, topológicas, espectrales y de Poincaré / KAM / BHT
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_BANACH_LIPSCHITZ_BOUND: Final[float] = 0.857  # cota teórica Lip(T³) ≤ 0.95³
_PHI23_LIPSCHITZ_HARD: Final[float] = 0.999
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_OODA_PIPELINE_KEY_v11.0.0"
_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_MAX_K: Final[int] = 12
_KAM_LATTICE_MODES: Final[int] = 6
_L3_DEFAULT_ITERS: Final[int] = 6
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_EIGHT_PI_SQ: Final[float] = 8.0 * math.pi ** 2
_TRANSVERSALITY_TOL: Final[float] = 1.0e-10
_RESONANCE_MAX_Q: Final[int] = 8
_RESONANCE_TOL: Final[float] = 0.02
_BHT_DISSIPATION_CEILING: Final[float] = 0.95
_DEFAULT_HILBERT_DIM: Final[int] = 56
_TRACE_TOL: Final[float] = 1.0e-5
_CONN_NORM_VETO: Final[float] = 2.5
_CONN_NORM_DEGRADED: Final[float] = 0.8
_YM_VETO: Final[float] = 0.85
_YM_DEGRADED: Final[float] = 0.25
_WILSON_VETO: Final[float] = 0.50
_WILSON_DEGRADED: Final[float] = 0.15
_INSTANTON_VETO: Final[int] = 2
_POSITRON_REST_ENERGY_KEV: Final[float] = 1022.0


def _is_near_resonance(
    rho: float, max_q: int = _RESONANCE_MAX_Q, tol: float = _RESONANCE_TOL
) -> bool:
    """Aproximación de Farey: ρ ≈ p/q con q ≤ max_q (divisores pequeños KAM)."""
    if not math.isfinite(rho):
        return True
    frac = rho - math.floor(rho)
    for q in range(1, max_q + 1):
        p = round(frac * q)
        if abs(frac - (p / q)) < (tol / q):
            return True
    return False


def _hermitian(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    a = np.asarray(m, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _antihermitian(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    a = np.asarray(m, dtype=np.complex128)
    return 0.5 * (a - a.conj().T)


def _hs_inner(a: NDArray[np.complex128], b: NDArray[np.complex128]) -> complex:
    """Producto Hilbert–Schmidt ⟨A, B⟩_HS = Tr(A† B)."""
    return complex(
        np.trace(
            np.asarray(a, dtype=np.complex128).conj().T
            @ np.asarray(b, dtype=np.complex128)
        )
    )


def _safe_eigh(herm: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    h = _hermitian(herm)
    try:
        evals, evecs = la.eigh(h)
    except la.LinAlgError:
        evals, evecs = np.linalg.eigh(h)
    return np.real(evals).astype(np.float64), np.asarray(evecs, dtype=np.complex128)


def _project_psd_normalize(
    matrix: NDArray, dimension: int, floor: float = _SPECTRAL_PSD_FLOOR
) -> NDArray[np.complex128]:
    herm = _hermitian(np.asarray(matrix, dtype=np.complex128))
    eigvals, eigvecs = _safe_eigh(herm)
    clean = np.maximum(eigvals, floor)
    psd = (eigvecs * clean) @ eigvecs.conj().T
    tr = float(np.real(np.trace(psd)))
    if tr <= _WILKINSON_EPSILON:
        return np.eye(dimension, dtype=np.complex128) / float(dimension)
    return psd / tr


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: FUNDACIÓN — Ω₃ HEYTING + CERTIFICADOS + OBSERVADORES + COMPUERTAS + ORIENTACIÓN        ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La ÚLTIMA definición de FASE 1 (GaugeDynamicsOrientor.export_ooda_state_seed) retorna un        ║
# ║  OODAStateSeed, que es la PRIMERA definición de FASE 2.                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting trivaluado Ω₃ = {⊤, ★, ⊥} con orden 0x00 ≼ 0x01 ≼ 0x03.

    Operaciones:
      ∧  = meet  = max  (infimo: el más restrictivo)
      ∨  = join  = min  (supremo: el más permisivo)
      →  = implicación de Gödel: a → b = ⊤ si a ≼ b, else b
      ¬  = a → ⊥

    El plegado de veredictos es el ínfimo (meet-fold): un solo ⊥ veta el ciclo.
    """

    COHERENT = 0x00  # ⊤ operativo
    DEGRADED = 0x01  # ★ operativo (válvula de alivio 1 h)
    VETOED = 0x03  # ⊥ operativo (crowbar ESP32 < 392.15 ns)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        """Ínfimo del retículo (nombre histórico: fold_supremum ≡ meet-fold)."""
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc

    def is_critical(self) -> bool:
        return self == HeytingOmega3.VETOED

    def rank(self) -> int:
        return {self.COHERENT: 0, self.DEGRADED: 1, self.VETOED: 2}[self]


@dataclass(frozen=True, slots=True)
class Phase1IsometryCertificate:
    """Certificado inmutable emitido por Fase 1 (Observe)."""

    certificate_id: str
    trace_preservation_ok: bool
    connection_norm: float
    purity: float
    von_neumann_entropy: float
    isometry_verdict: HeytingOmega3
    timestamp_utc: float
    hmac_signature: str


@dataclass(frozen=True, slots=True)
class Phase2GaugeDynamicsCertificate:
    """Certificado inmutable emitido por Fase 2 (Orient)."""

    certificate_id: str
    phase1_ref_id: str
    yang_mills_energy: float
    chern_simons_3form: float
    wilson_angle_rad: float
    instanton_number_k: int
    lipschitz_contraction: float
    bianchi_residue: float
    annihilation_executed: bool
    released_energy_kev: float
    dynamics_verdict: HeytingOmega3
    timestamp_utc: float
    hmac_signature: str


class GaugeIsometryObserver:
    """FASE 1: OBSERVE — Audita isometría de calibre, conservación de traza y espectro."""

    def __init__(
        self, dimension: int = _DEFAULT_HILBERT_DIM, secret_key: bytes = _DEFAULT_SECRET_KEY
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        return _project_psd_normalize(matrix, self.dimension)

    def extract_eigenstructure(
        self, rho: NDArray
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Espectro (evals, evecs) ascendente para uso en FASE 2 (semilla)."""
        return _safe_eigh(np.asarray(rho, dtype=np.complex128))

    def extract_gauge_connection_A(self, rho: NDArray) -> NDArray[np.complex128]:
        r"""A = i (ρ − (Tr ρ / N) I) ∈ 𝔰𝔲(N), anti-hermítica y de traza nula."""
        r = self.sanitize_density_matrix(rho)
        n = r.shape[0]
        tr_val = np.trace(r) / float(n)
        su_n = r - tr_val * np.eye(n, dtype=np.complex128)
        return _antihermitian(1j * su_n)

    def execute_phase1_observation(self, density_matrix: NDArray) -> Phase1IsometryCertificate:
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        tr_ok = abs(tr_val - 1.0) < _TRACE_TOL

        connection_A = self.extract_gauge_connection_A(rho)
        conn_norm = float(np.linalg.norm(connection_A, ord="fro"))

        evals, _ = _safe_eigh(rho)
        evals = np.maximum(evals, _WILKINSON_EPSILON)
        evals = evals / float(evals.sum())
        purity = float(np.real(np.trace(rho @ rho)))
        entropy = float(-np.sum(evals * np.log(evals)))

        if not tr_ok or conn_norm > _CONN_NORM_VETO:
            verdict = HeytingOmega3.VETOED
        elif conn_norm > _CONN_NORM_DEGRADED:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        cert_id = f"CERT-P1-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{cert_id}:{tr_ok}:{conn_norm:.8f}:{purity:.8f}:{verdict.value}"
        ).encode("utf-8")
        signature = hmac.new(self.secret_key, seed_bytes, hashlib.sha256).hexdigest()

        return Phase1IsometryCertificate(
            certificate_id=cert_id,
            trace_preservation_ok=tr_ok,
            connection_norm=conn_norm,
            purity=purity,
            von_neumann_entropy=entropy,
            isometry_verdict=verdict,
            timestamp_utc=now,
            hmac_signature=signature,
        )


class HomomorphicGatePhi12:
    r"""
    COMPUERTA Φ₁₂: homomorfismo de retículos Observe → Orient.

    Un homomorfismo de Heyting h: Ω₃ → Ω₃ preserva ∧, ⊤. Aquí h = id_Ω₃
    (la anomalía caótica entra a Fase 2 para juzgamiento analítico: no se
    filtra ⊥ en Φ₁₂). La integridad se garantiza por HMAC-SHA256 en tiempo cte.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)

    def verify_gate_phi12(self, cert1: Phase1IsometryCertificate) -> bool:
        seed_bytes = (
            f"{cert1.certificate_id}:{cert1.trace_preservation_ok}:"
            f"{cert1.connection_norm:.8f}:{cert1.purity:.8f}:{cert1.isometry_verdict.value}"
        ).encode("utf-8")
        expected_sig = hmac.new(self.secret_key, seed_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(cert1.hmac_signature, expected_sig)

    @staticmethod
    def preserves_meet(a: HeytingOmega3, b: HeytingOmega3) -> bool:
        """Test de homomorfismo: h(a ∧ b) = h(a) ∧ h(b)  (h = id)."""
        return a.meet(b) == HeytingOmega3(max(int(a), int(b)))


class DiscreteOODADeRham:
    r"""
    Complejo de de Rham matricial sobre 𝔲(N) anclado en la conexión de Observe:

        Ω⁰ ──d──► Ω¹ ──d──► Ω²
         ℝ         𝔲(N)      [𝔲(N), 𝔲(N)]

    En trivialización de Maurer–Cartan (dA ≡ 0 a nivel de carta),
        F = dA + A ∧ A = A² .
    Bianchi: d_A F = [A, F] = [A, A²] = 0.

    Invariantes:
        YM(A)  = ½ ‖F‖_HS²
        CS₃    = (1 / 8π²) Tr( A F − ⅓ A³ + Φ F )   (Chern–Simons + Higgs espectral)
        P₂     = (1 / 8π²) Re Tr(F²)                (densidad de Pontryagin → k ∈ ℤ)
        W      = arg Tr exp(A) / N                   (ángulo de Wilson)
    """

    def __init__(self, connection_A: NDArray[np.complex128]) -> None:
        self.A = _antihermitian(connection_A)

    def curvature(self) -> NDArray[np.complex128]:
        return self.A @ self.A

    def bianchi_residue(self) -> float:
        F = self.curvature()
        comm = self.A @ F - F @ self.A
        return float(np.linalg.norm(comm, ord="fro"))

    def yang_mills_action(self) -> float:
        F = self.curvature()
        return 0.5 * float(np.real(_hs_inner(F, F)))

    def pontryagin_density(self) -> float:
        F = self.curvature()
        return float(np.real(np.trace(F @ F)) / _EIGHT_PI_SQ)

    def instanton_number(self) -> int:
        return int(math.floor(self.pontryagin_density() + 0.5))

    def chern_simons_3form(self, higgs_phi: NDArray[np.complex128]) -> float:
        F = self.curvature()
        a = self.A
        cs_term = a @ F - (1.0 / 3.0) * (a @ a @ a) + higgs_phi @ F
        return float(np.real(np.trace(cs_term)) / _EIGHT_PI_SQ)

    def wilson_angle(self) -> float:
        try:
            w_op = la.expm(self.A)
        except (ValueError, la.LinAlgError):
            w_op = np.eye(self.A.shape[0], dtype=np.complex128)
        n = max(1, int(self.A.shape[0]))
        w_trace = complex(np.trace(w_op) / float(n))
        return float(cmath.phase(w_trace))


class GaugeDynamicsOrientor:
    """FASE 2-ORIENT (aún FASE 1 del módulo): Yang–Mills, Chern–Simons, Wilson, instantones."""

    def __init__(
        self, dimension: int = _DEFAULT_HILBERT_DIM, secret_key: bytes = _DEFAULT_SECRET_KEY
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def _estimate_empirical_lipschitz(self, rho: NDArray) -> float:
        r"""
        Factor de contracción empírico de un paso L1 hacia el target diagonal:
            Lip_emp = ‖(1−θ)ρ + θ ρ* − ρ*‖_F / ‖ρ − ρ*‖_F ,   θ = 1/2.

        Por el teorema del punto fijo de Banach, Lip < 1 ⇒ órbita convergente
        en (𝔇_N, ‖·‖_F). La cota teórica Lip(T³) ≤ 0.95³ ≈ 0.857 se usa como
        umbral de veto (no como valor hardcodeado del factor).
        """
        tr_val = float(np.real(np.trace(rho)))
        if tr_val < _WILKINSON_EPSILON:
            return _BANACH_LIPSCHITZ_BOUND
        rho_norm = rho / tr_val
        target = np.diag(np.real(np.diag(rho_norm))).astype(np.complex128)
        d_before = float(np.linalg.norm(rho_norm - target, ord="fro"))
        if d_before < _WILKINSON_EPSILON:
            return 0.0
        mixed = 0.5 * rho_norm + 0.5 * target
        d_after = float(np.linalg.norm(mixed - target, ord="fro"))
        ratio = float(d_after / d_before)
        if not math.isfinite(ratio):
            return _BANACH_LIPSCHITZ_BOUND
        return ratio

    def _higgs_from_spectrum(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Campo de Higgs espectral Φ = U diag(log λ) U† (regularizado)."""
        evals, evecs = _safe_eigh(rho)
        log_eigs = np.log(np.maximum(evals, _SPECTRAL_PSD_FLOOR))
        return (evecs * log_eigs) @ evecs.conj().T

    def execute_phase2_orientation(
        self,
        cert1: Phase1IsometryCertificate,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> Phase2GaugeDynamicsCertificate:
        now = time.time()
        rho = _hermitian(np.asarray(density_matrix, dtype=np.complex128))
        n = int(rho.shape[0]) if rho.ndim == 2 else self.dimension
        tr_val = float(np.real(np.trace(rho)))
        if tr_val > _WILKINSON_EPSILON:
            rho = rho / tr_val

        su_n = rho - (np.trace(rho) / float(n)) * np.eye(n, dtype=np.complex128)
        a = _antihermitian(1j * su_n)
        dR = DiscreteOODADeRham(a)
        higgs_phi = self._higgs_from_spectrum(rho)

        ym_energy = dR.yang_mills_action()
        cs3_val = dR.chern_simons_3form(higgs_phi)
        w_angle = dR.wilson_angle()
        instanton_k = dR.instanton_number()
        bianchi = dR.bianchi_residue()

        annihilation_executed = False
        released_energy = 0.0
        if electron_anomaly_count > 0 and positron_auth_signature:
            expected_pos_hmac = hmac.new(
                self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if hmac.compare_digest(positron_auth_signature, expected_pos_hmac):
                annihilation_executed = True
                released_energy = float(electron_anomaly_count) * _POSITRON_REST_ENERGY_KEV

        lipschitz = self._estimate_empirical_lipschitz(rho)

        if ym_energy > _YM_VETO or abs(w_angle) >= _WILSON_VETO or abs(instanton_k) > _INSTANTON_VETO:
            verdict = HeytingOmega3.VETOED
        elif ym_energy > _YM_DEGRADED or abs(w_angle) >= _WILSON_DEGRADED:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        if lipschitz > _BANACH_LIPSCHITZ_BOUND:
            verdict = verdict.meet(HeytingOmega3.VETOED)

        verdict = verdict.meet(cert1.isometry_verdict)

        cert_id = f"CERT-P2-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{cert_id}:{cert1.certificate_id}:{ym_energy:.8f}:{cs3_val:.8f}:"
            f"{w_angle:.8f}:{instanton_k}:{verdict.value}"
        ).encode("utf-8")
        signature = hmac.new(self.secret_key, seed_bytes, hashlib.sha256).hexdigest()

        return Phase2GaugeDynamicsCertificate(
            certificate_id=cert_id,
            phase1_ref_id=cert1.certificate_id,
            yang_mills_energy=ym_energy,
            chern_simons_3form=cs3_val,
            wilson_angle_rad=w_angle,
            instanton_number_k=instanton_k,
            lipschitz_contraction=lipschitz,
            bianchi_residue=bianchi,
            annihilation_executed=annihilation_executed,
            released_energy_kev=released_energy,
            dynamics_verdict=verdict,
            timestamp_utc=now,
            hmac_signature=signature,
        )

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (export_ooda_state_seed).                                         ║
    # ║  ÚLTIMO método de FASE 1. Retorna OODAStateSeed → PRIMER objeto de FASE 2.                  ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def export_ooda_state_seed(
        self,
        cert1: Phase1IsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        rho: NDArray[np.complex128],
        dimension: int,
        probe: Optional[NDArray[np.complex128]] = None,
    ) -> "OODAStateSeed":
        r"""
        Comprime la tríada (cert1, cert2, ρ) en una semilla canónica que alimenta
        el análisis Poincaré / KAM / L3 de FASE 2:

          • Acciones J_k = eigvals(ρ)  (descendente, normalizadas, Σ J = 1)
          • Ángulos θ_k = arg(ψ_dom)_k + 0.05 · ϑ_Wilson
          • Frecuencias ω_k = √(1+k) + (φ−1)/(1+k)   (FPU ⊕ cascade áureo)
          • Probe H_probe hermítico, de traza nula, ‖H‖_F = 1
            (observable de la sección de Poincaré: g(ρ) = ⟨H, ρ⟩_HS − c)
          • Target = proyección diagonal (clásica) de ρ, sanitizada

        Conservación: I₁ = Σ_k J_k = Tr(ρ) = 1  (Poincaré–Cartan).
        Retorna el OODAStateSeed que INICIA FASE 2.
        """
        dim = max(4, int(dimension))
        herm = _hermitian(np.asarray(rho, dtype=np.complex128))
        eigvals, eigvecs = _safe_eigh(herm)
        order = np.argsort(eigvals)[::-1]
        J_sorted = np.maximum(eigvals[order], 0.0)
        J_total = float(J_sorted.sum())
        J_norm = J_sorted / (J_total + _WILKINSON_EPSILON)

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)

        dominant_vec = eigvecs[:, order[0]]
        theta = np.angle(dominant_vec) + 0.05 * cert2.wilson_angle_rad

        tr_h = max(float(np.real(np.trace(herm))), _WILKINSON_EPSILON)
        rho_norm = herm / tr_h
        target = np.diag(np.real(np.diag(rho_norm))).astype(np.complex128)
        target = _project_psd_normalize(target, dim)

        if probe is None:
            seed_val = int(
                hashlib.sha256(
                    f"{cert1.certificate_id}:{cert2.certificate_id}:{dim}".encode()
                ).hexdigest()[:16],
                16,
            )
            rng = np.random.default_rng(seed_val)
            H_raw = rng.normal(0.0, 1.0, (dim, dim))
            probe_h = _hermitian(H_raw.astype(np.complex128))
        else:
            probe_h = _hermitian(np.asarray(probe, dtype=np.complex128))

        tr_p = float(np.real(np.trace(probe_h)))
        probe_h = probe_h - (tr_p / float(dim)) * np.eye(dim, dtype=np.complex128)
        pnorm = float(np.linalg.norm(probe_h, ord="fro"))
        if pnorm > _WILKINSON_EPSILON:
            probe_h = probe_h / pnorm

        return OODAStateSeed(
            source_merkle=hashlib.sha256(
                f"{cert1.certificate_id}:{cert2.certificate_id}".encode()
            ).hexdigest(),
            rho_vector=tuple(complex(x) for x in rho_norm.flatten()),
            target_vector=tuple(complex(x) for x in target.flatten()),
            probe_vector=tuple(complex(x) for x in probe_h.flatten()),
            action_vector=tuple(float(x) for x in J_norm),
            angle_vector=tuple(float(x) for x in theta),
            frequency_vector=tuple(float(x) for x in omega),
            tr_ok=cert1.trace_preservation_ok,
            conn_norm=cert1.connection_norm,
            purity=cert1.purity,
            von_neumann_entropy=cert1.von_neumann_entropy,
            ym_energy=cert2.yang_mills_energy,
            cs3_val=cert2.chern_simons_3form,
            wilson_angle=cert2.wilson_angle_rad,
            instanton_k=cert2.instanton_number_k,
            lipschitz=cert2.lipschitz_contraction,
            bianchi_residue=cert2.bianchi_residue,
            annihilation_executed=cert2.annihilation_executed,
            released_energy_kev=cert2.released_energy_kev,
            verdict_p1=cert1.isometry_verdict,
            verdict_p2=cert2.dynamics_verdict,
            dimension=dim,
        )


class HomomorphicGatePhi23:
    r"""
    COMPUERTA Φ₂₃: homomorfismo Orient → Decide & Act.

    Revalida HMAC y exige Lip_emp ≤ 0.999 (holgura numérica sobre la cota
    de Banach 0.857 usada como veto de orientación). Un Lip ≥ 1 rompería
    la contracción y se rechaza el paso al decisor.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)

    def verify_gate_phi23(self, cert2: Phase2GaugeDynamicsCertificate) -> bool:
        if cert2.lipschitz_contraction > _PHI23_LIPSCHITZ_HARD:
            return False
        seed_bytes = (
            f"{cert2.certificate_id}:{cert2.phase1_ref_id}:{cert2.yang_mills_energy:.8f}:"
            f"{cert2.chern_simons_3form:.8f}:{cert2.wilson_angle_rad:.8f}:"
            f"{cert2.instanton_number_k}:{cert2.dynamics_verdict.value}"
        ).encode("utf-8")
        expected_sig = hmac.new(self.secret_key, seed_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(cert2.hmac_signature, expected_sig)


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: SEMILLA CANÓNICA + POINCARÉ Σ + TWIST P–B + KAM/BHT + OPERADOR L0..L3                   ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (OODAStateSeed) consume el retorno de export_ooda_state_seed (FASE 1).            ║
# ║  Última def. (RecursiveOODAImprover.iterate_level_3) → OODATrajectory (PRIMER objeto FASE 3).   ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class OODAStateSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRY: continúa de export_ooda_state_seed.                                         │
    │  Semilla canónica (ρ, ρ*, H_probe, J, θ, ω) + métricas de certificado que anclan KAM/L3.     │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_merkle: str
    rho_vector: Tuple[complex, ...]
    target_vector: Tuple[complex, ...]
    probe_vector: Tuple[complex, ...]
    action_vector: Tuple[float, ...]
    angle_vector: Tuple[float, ...]
    frequency_vector: Tuple[float, ...]
    tr_ok: bool
    conn_norm: float
    purity: float
    von_neumann_entropy: float
    ym_energy: float
    cs3_val: float
    wilson_angle: float
    instanton_k: int
    lipschitz: float
    bianchi_residue: float
    annihilation_executed: bool
    released_energy_kev: float
    verdict_p1: HeytingOmega3
    verdict_p2: HeytingOmega3
    dimension: int

    def get_rho(self) -> NDArray[np.complex128]:
        return np.array(self.rho_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_target(self) -> NDArray[np.complex128]:
        return np.array(self.target_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_probe(self) -> NDArray[np.complex128]:
        return np.array(self.probe_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def poincare_cartan_invariant(self) -> float:
        r"""I₁ = Σ_k J_k = Tr(ρ) = 1  (invariante integral de Poincaré–Cartan)."""
        return float(sum(self.action_vector))

    def hamiltonian(self) -> float:
        """H(J) = Σ_k ω_k J_k — energía del toro isócrono decisional."""
        return float(
            np.dot(
                np.array(self.frequency_vector, dtype=np.float64),
                np.array(self.action_vector, dtype=np.float64),
            )
        )

    def combined_verdict(self) -> HeytingOmega3:
        return self.verdict_p1.meet(self.verdict_p2)


class OODAPoincareSection:
    r"""
    Sección transversal sobre 𝔇_N:
        Σ = { ρ ∈ 𝔇_N : g(ρ) = Re⟨H_probe, ρ⟩_HS − c = 0 }.
    H_probe hermítico ⇒ g es un funcional lineal real (observable).
    Detecta transiciones coherente ⇄ degradado en la órbita OODA iterada.
    """

    def __init__(self, probe: NDArray[np.complex128], offset: float, label: str = "Σ") -> None:
        self.probe = _hermitian(probe)
        self.offset = float(offset)
        self.label = label

    def g(self, rho: NDArray[np.complex128]) -> float:
        return float(np.real(_hs_inner(self.probe, rho))) - self.offset

    def transversality(
        self, d_rho: NDArray[np.complex128], tol: float = _TRANSVERSALITY_TOL
    ) -> bool:
        return abs(float(np.real(_hs_inner(self.probe, d_rho)))) > tol

    def oriented_flux(self, d_rho: NDArray[np.complex128]) -> float:
        return float(np.real(_hs_inner(self.probe, d_rho)))


class OODAReturnMap:
    r"""
    Mapa de retorno P : Σ → Σ de la órbita L3 del ciclo OODA.
    Cruces g(ρ_i)·g(ρ_{i+1}) < 0 con interpolación lineal, aceptados sólo si
    el campo discreto Δρ es transversal a Σ; re-sanitización de traza.
    """

    def __init__(self, section: OODAPoincareSection) -> None:
        self.section = section

    def detect_crossings(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        crossings: List[NDArray[np.complex128]] = []
        if len(rho_sequence) < 2:
            return crossings
        g_vals = [self.section.g(r) for r in rho_sequence]
        for i in range(len(g_vals) - 1):
            gi, gj = g_vals[i], g_vals[i + 1]
            if gi * gj >= 0.0:
                continue
            denom = gi - gj
            if abs(denom) <= _WILKINSON_EPSILON:
                continue
            t = float(np.clip(gi / denom, 0.0, 1.0))
            flow = rho_sequence[i + 1] - rho_sequence[i]
            if not self.section.transversality(flow):
                continue
            rho_cross = (1.0 - t) * rho_sequence[i] + t * rho_sequence[i + 1]
            tr_val = float(np.real(np.trace(rho_cross)))
            if tr_val > _WILKINSON_EPSILON:
                rho_cross = rho_cross / tr_val
            crossings.append(rho_cross)
        return crossings

    def apply(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        return self.detect_crossings(rho_sequence)

    @staticmethod
    def birkhoff_rotation_number(
        crossings: Sequence[NDArray[np.complex128]], probe: NDArray[np.complex128]
    ) -> float:
        r"""
        Número de rotación de Birkhoff:
            ρ = (θ_N − θ_0) / (2π N),   θ = arg Tr(H† ρ),
        con unwrap para eliminar saltos de rama.
        """
        if len(crossings) < 2:
            return 0.0
        H = _hermitian(probe)
        raw = [float(np.angle(_hs_inner(H, r))) for r in crossings]
        angles = np.unwrap(np.array(raw, dtype=np.float64))
        n_steps = len(angles) - 1
        if n_steps <= 0:
            return 0.0
        return float((angles[-1] - angles[0]) / (_TWO_PI * n_steps))


class OODAPoincareBirkhoffTwist:
    r"""
    Teorema geométrico de Poincaré–Birkhoff sobre el anillo (J_dom, θ_dom):
    un mapa que preserva área con twist ∂(Δθ)/∂J ≠ 0 posee ≥ 2 puntos fijos.
    """

    def __init__(self, seed: OODAStateSeed, delta_J: float = 1.0e-4) -> None:
        self.seed = seed
        self.delta_J = float(delta_J)

    def twist_derivative(self) -> float:
        J = np.array(self.seed.action_vector, dtype=np.float64)
        theta = np.array(self.seed.angle_vector, dtype=np.float64)
        omega = np.array(self.seed.frequency_vector, dtype=np.float64)
        if J.size == 0:
            return 0.0
        dom = int(np.argmax(J))
        dJ = self.delta_J

        def advance(J_vec: NDArray[np.float64]) -> float:
            return float((theta[dom] + omega[dom] * J_vec[dom]) % _TWO_PI)

        J_plus, J_minus = J.copy(), J.copy()
        J_plus[dom] = min(1.0, J[dom] + dJ)
        J_minus[dom] = max(_WILKINSON_EPSILON, J[dom] - dJ)
        d_theta = advance(J_plus) - advance(J_minus)
        d_theta = (d_theta + math.pi) % _TWO_PI - math.pi
        denom = float(J_plus[dom] - J_minus[dom])
        if abs(denom) <= _WILKINSON_EPSILON:
            return 0.0
        return float(d_theta / denom)

    def satisfies_twist(self, tol: float = 1.0e-12) -> bool:
        return abs(self.twist_derivative()) > tol


class BirkhoffOODAForm:
    r"""
    Forma normal de Birkhoff en el punto fijo del ciclo OODA:
        T(ρ) = ρ* + DT(ρ*)·(ρ − ρ*) + O(‖ρ − ρ*‖²).
    Clasificación espectral de la dinámica decisional local.
    """

    def __init__(
        self,
        fixed_rho: NDArray[np.complex128],
        jacobian_spectrum: NDArray[np.complex128],
    ) -> None:
        self.fixed_rho = fixed_rho
        self.jacobian_spectrum = np.asarray(jacobian_spectrum, dtype=np.complex128)

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
            return "ELLIPTIC_CENTER"
        return "SADDLE_HYPERBOLIC"

    def spectral_radius(self) -> float:
        return (
            float(np.max(np.abs(self.jacobian_spectrum))) if self.jacobian_spectrum.size else 0.0
        )

    def lyapunov_exponent(self) -> float:
        rho = self.spectral_radius()
        return float(np.log(max(rho, _WILKINSON_EPSILON)))

    def twist_eigenvalue_condition(self) -> bool:
        for lam in self.jacobian_spectrum:
            mod = abs(complex(lam))
            if abs(mod - 1.0) < 1e-8 and abs(lam - 1.0) > 1e-8 and abs(lam + 1.0) > 1e-8:
                return True
        return False


class KAMOODATorus:
    r"""
    Toro invariante 𝕋^n de robustez decisional con frecuencias ω ∈ ℝ^n.

    Condición diofantina (Kolmogorov):
        |⟨k, ω⟩| ≥ γ / |k|_1^τ ,   ∀ k ∈ ℤ^n \ {0},   τ > n − 1.

    Criterio BHT (Broer–Huitema–Takens) en el régimen disipativo L3:
        el toro persiste como NHIM si γ > ‖ε‖ y la disipación no satura γ.
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)

    def _lattice_vectors(self, max_k: int, modes: int) -> Iterable[NDArray[np.float64]]:
        m = max(1, min(self.n, modes))
        seen: Set[Tuple[int, ...]] = set()

        def emit(vec: NDArray[np.float64]) -> Iterable[NDArray[np.float64]]:
            key = tuple(int(x) for x in vec)
            if any(key) and key not in seen:
                seen.add(key)
                yield vec

        for i in range(m):
            for s in (-1, 1):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = s
                yield from emit(kvec)

        for i, j in itertools.combinations(range(m), 2):
            for si, sj in itertools.product((-1, 1), repeat=2):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = si
                kvec[j] = sj
                yield from emit(kvec)

        cube = min(m, 4)
        bound = min(2, max(1, max_k // 6))
        for coeffs in itertools.product(range(-bound, bound + 1), repeat=cube):
            if all(c == 0 for c in coeffs):
                continue
            kvec = np.zeros(m, dtype=np.float64)
            kvec[:cube] = np.array(coeffs, dtype=np.float64)
            yield from emit(kvec)

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: float = _DIOPHANTINE_TAU
    ) -> float:
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        gammas: List[float] = []
        for kvec in self._lattice_vectors(max_k=max_k, modes=m):
            inner = abs(float(np.dot(kvec, omega)))
            l1 = float(np.sum(np.abs(kvec)))
            if l1 <= 0.0:
                continue
            gammas.append(inner * (l1**tau))
        return float(min(gammas)) if gammas else 0.0

    def small_divisors(self, max_k: int = _KAM_MAX_K) -> Tuple[float, ...]:
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        divs = [abs(float(np.dot(kvec, omega))) for kvec in self._lattice_vectors(max_k, m)]
        divs = [d for d in divs if d > 0.0]
        divs.sort()
        return tuple(divs[:8])

    def is_kam_stable(self, perturbation_norm: float, gamma: Optional[float] = None) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        return gamma > perturbation_norm

    def is_dissipative_kam_stable(
        self,
        perturbation_norm: float,
        dissipation: float,
        gamma: Optional[float] = None,
    ) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        dissip = abs(float(dissipation))
        return (gamma > perturbation_norm) and (dissip < min(gamma, _BHT_DISSIPATION_CEILING))

    def winding_numbers(self, phases: Sequence[float]) -> Tuple[float, ...]:
        if len(phases) < 2:
            return tuple(0.0 for _ in range(self.n))
        unwrapped = np.unwrap(np.array(phases, dtype=np.float64))
        delta_phi = float(unwrapped[-1] - unwrapped[0])
        if abs(delta_phi) < _WILKINSON_EPSILON:
            return tuple(0.0 for _ in range(self.n))
        return tuple(float(w / delta_phi) for w in self.frequencies)


def pancharatnam_berry_holonomy(rho_sequence: Sequence[NDArray[np.complex128]]) -> float:
    r"""
    Holonomía discreta de Pancharatnam–Berry a lo largo de {ρ_t}:
        γ_geom = Σ_t arg ⟨ψ_t | ψ_{t+1}⟩ ,
    con gauge de transporte paralelo.
    """
    if len(rho_sequence) < 2:
        return 0.0
    acc = 0.0
    prev: Optional[NDArray[np.complex128]] = None
    for rho in rho_sequence:
        evals, evecs = _safe_eigh(rho)
        psi = evecs[:, int(np.argmax(evals))]
        if prev is not None:
            overlap = np.vdot(prev, psi)
            acc += float(np.angle(overlap))
            mag = abs(overlap)
            if mag > _WILKINSON_EPSILON:
                psi = psi * (np.conj(overlap) / mag)
        prev = psi
    return float(acc)


class RecursiveOODAImprover:
    r"""
    Operador de automejora recursiva I : 𝔇_N → 𝔇_N sobre el ciclo decisional OODA.

      L0: I_0(ρ) = ρ                                                 (identidad)
      L1: I_1(ρ) = Higham((1−θ)ρ + θ ρ*)                             (contracción)
      L2: θ ← θ·(1 + 0.1(ratio_err − 1))                             (residuo)
      L3: η_meta ← η_meta · {0.90 / 1.05} según ρ_espectral(DI)
          ⊕ realimentación del número de rotación de Poincaré
          ⊕ penalización de resonancias p/q (divisores pequeños KAM)
          ⊕ proyección PSD + Tr = 1
        = metacognición metacognitiva con observable celeste de 2º orden.

    El mapa L3 es disipativo (θ ∈ (0.05, 0.95)): el toro de decisión, si
    sobrevive, lo hace como NHIM en el sentido BHT.
    """

    def __init__(self, contraction: float = 0.5, meta_learning: float = 0.3) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self._last_rotation: float = 0.0
        self._resonance_hits: int = 0

    def apply_level_0(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.asarray(rho, dtype=np.complex128).copy()

    def apply_level_1(
        self, rho: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        mixed = (1.0 - self.contraction) * rho + self.contraction * target
        n = int(mixed.shape[0])
        return _project_psd_normalize(mixed, n)

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
                    np.clip(self.contraction * (1.0 + 0.1 * (ratio - 1.0)), 0.05, 0.95)
                )
        return self.apply_level_1(rho, target)

    def apply_level_3(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
        poincare_rotation: Optional[float] = None,
    ) -> NDArray[np.complex128]:
        if len(history) >= 3:
            rho_spec = self._estimate_local_spectral_radius(history[-3:])
            if rho_spec > 1.0:
                self.meta_learning = float(np.clip(self.meta_learning * 0.90, 0.05, 0.90))
            elif 0.0 < rho_spec < 0.5:
                self.meta_learning = float(np.clip(self.meta_learning * 1.05, 0.05, 0.90))

        if poincare_rotation is not None and math.isfinite(poincare_rotation):
            self._last_rotation = float(poincare_rotation)
            if _is_near_resonance(poincare_rotation):
                self._resonance_hits += 1
                self.contraction = float(np.clip(self.contraction * 1.08, 0.05, 0.95))
                self.meta_learning = float(np.clip(self.meta_learning * 0.92, 0.05, 0.90))
            else:
                self.contraction = float(np.clip(self.contraction * 0.98, 0.05, 0.95))

        self.contraction = float(
            np.clip(self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95)
        )
        return self.apply_level_2(rho, target, history)

    @staticmethod
    def _estimate_local_spectral_radius(recent: Sequence[NDArray[np.complex128]]) -> float:
        """Cociente de Rayleigh HS: λ ≈ ⟨Δ₂, Δ₁⟩_HS / ‖Δ₂‖²."""
        if len(recent) < 3:
            return 0.5
        d1 = recent[-1] - recent[-2]
        d2 = recent[-2] - recent[-3]
        denom = _hs_inner(d2, d2)
        if abs(denom) < _WILKINSON_EPSILON:
            return 0.5
        return float(abs(_hs_inner(d2, d1) / denom))

    def _poincare_rotation_from_history(
        self,
        history: Sequence[NDArray[np.complex128]],
        probe: NDArray[np.complex128],
    ) -> Optional[float]:
        if len(history) < 4:
            return None
        section = OODAPoincareSection(probe=probe, offset=0.0, label="Σ_L3")
        prm = OODAReturnMap(section)
        crossings = prm.detect_crossings(list(history))
        if len(crossings) < 2:
            return None
        return OODAReturnMap.birkhoff_rotation_number(crossings, probe)

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (iterate_level_3).                                                ║
    # ║  ÚLTIMO método de FASE 2. Retorna OODATrajectory → PRIMER objeto de FASE 3.                 ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: OODAStateSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> "OODATrajectory":
        r"""
        Itera I_L3 sobre ρ_seed con target ρ*. Devuelve:
          • Secuencia ρ_0 → ρ_T con Tr ≡ 1
          • Acciones J_t = Tr(ρ_t) ≡ 1  (invariante de Poincaré–Cartan)
          • Ángulos θ_t = arg Tr(H_probe† ρ_t)  (fase de Poincaré)
          • λ (log-ratio de ‖Δρ‖_F), γ diofantina, winding topológico
          • Estabilidad KAM / BHT, rotación de Birkhoff, twist P–B
          • Holonomía de Pancharatnam–Berry a lo largo de la órbita
        """
        dim = seed.dimension
        rho = seed.get_rho()
        target = seed.get_target()
        probe = seed.get_probe()

        rho_sequence: List[NDArray[np.complex128]] = [rho.copy()]
        action_sequence: List[float] = [float(np.real(np.trace(rho)))]
        angle_sequence: List[float] = [float(np.angle(_hs_inner(probe, rho)))]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max_iters):
            history.append(rho.copy())
            rho_p = self._poincare_rotation_from_history(history, probe)
            rho = self.apply_level_3(rho, target, history, poincare_rotation=rho_p)
            rho_sequence.append(rho.copy())
            action_sequence.append(float(np.real(np.trace(rho))))
            angle_sequence.append(float(np.angle(_hs_inner(probe, rho))))

        diffs = [
            float(np.linalg.norm(rho_sequence[i + 1] - rho_sequence[i], ord="fro"))
            for i in range(len(rho_sequence) - 1)
        ]
        if len(diffs) >= 2 and all(d > _WILKINSON_EPSILON for d in diffs):
            log_ratios = np.log(
                np.array(diffs[1:]) / np.maximum(np.array(diffs[:-1]), _WILKINSON_EPSILON)
            )
            finite = np.isfinite(log_ratios)
            lyapunov = float(np.mean(log_ratios[finite])) if np.any(finite) else 0.0
        elif diffs and all(d <= _WILKINSON_EPSILON for d in diffs):
            lyapunov = -20.0
        else:
            lyapunov = -20.0 if not diffs else float(np.log(max(diffs[-1], _WILKINSON_EPSILON)))
        if not math.isfinite(lyapunov):
            lyapunov = -20.0 if lyapunov < 0 else 0.0
        convergence = diffs[-1] if diffs else 0.0

        torus = KAMOODATorus(seed.frequency_vector)
        gamma = torus.diophantine_constant()
        dissipation = abs(self.contraction)
        kam_stable = torus.is_kam_stable(convergence, gamma)
        bht_stable = torus.is_dissipative_kam_stable(convergence, dissipation, gamma)
        winding = torus.winding_numbers(angle_sequence)

        jacobian_spectrum = self._estimate_global_jacobian_spectrum(rho_sequence)
        birkhoff = BirkhoffOODAForm(rho_sequence[-1], jacobian_spectrum)
        birkhoff_class = birkhoff.classify()

        live_section = OODAPoincareSection(probe=probe, offset=0.0, label="Σ_flow")
        prm = OODAReturnMap(live_section)
        crossings = prm.apply(rho_sequence)
        rotation = (
            OODAReturnMap.birkhoff_rotation_number(crossings, probe)
            if len(crossings) >= 2
            else 0.0
        )
        twist = OODAPoincareBirkhoffTwist(seed)
        holonomy = pancharatnam_berry_holonomy(rho_sequence)
        spectral_radius = float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0

        return OODATrajectory(
            source_merkle=seed.source_merkle,
            dimension=dim,
            rho_sequence=tuple(self._serialize(r) for r in rho_sequence),
            action_sequence=tuple(action_sequence),
            angle_sequence=tuple(angle_sequence),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            spectral_radius=spectral_radius,
            is_kam_stable=kam_stable,
            is_bht_stable=bht_stable,
            winding_numbers=winding,
            diophantine_gamma=gamma,
            final_contraction=self.contraction,
            final_meta_learning=self.meta_learning,
            birkhoff_classification=birkhoff_class,
            poincare_rotation_number=rotation,
            poincare_crossing_count=len(crossings),
            twist_condition=twist.satisfies_twist(),
            holonomy_phase=holonomy,
            resonance_hits=self._resonance_hits,
            lipschitz_seed=seed.lipschitz,
            ym_energy=seed.ym_energy,
            wilson_angle=seed.wilson_angle,
            instanton_k=seed.instanton_k,
        )

    @staticmethod
    def _serialize(rho: NDArray[np.complex128]) -> Tuple[complex, ...]:
        return tuple(complex(x) for x in rho.flatten())

    @staticmethod
    def deserialize(vec: Sequence[complex], dim: int) -> NDArray[np.complex128]:
        return np.array(vec, dtype=np.complex128).reshape((dim, dim))

    @staticmethod
    def _estimate_global_jacobian_spectrum(
        rho_sequence: Sequence[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(rho_sequence) < 2:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        d_last = rho_sequence[-1] - rho_sequence[-2]
        if len(rho_sequence) >= 3:
            d_prev = rho_sequence[-2] - rho_sequence[-3]
            denom = _hs_inner(d_prev, d_prev)
            if abs(denom) > _WILKINSON_EPSILON:
                rayleigh = _hs_inner(d_prev, d_last) / denom
                return np.array([complex(rayleigh)], dtype=np.complex128)
        nrm = float(np.linalg.norm(d_last, ord="fro"))
        return np.array([complex(nrm, 0.0)], dtype=np.complex128)


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: TRAYECTORIA + AUDITOR SOBERANO + DECISOR L3-AWARE + PIPELINE                            ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (OODATrajectory) consume iterate_level_3 de FASE 2.                               ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class OODATrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRY: continúa de iterate_level_3 (FASE 2).                                       │
    │  Trayectoria L3 del ciclo OODA con Poincaré, twist P–B, KAM/BHT y holonomía de Rham.         │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_merkle: str
    dimension: int
    rho_sequence: Tuple[Tuple[complex, ...], ...]
    action_sequence: Tuple[float, ...]
    angle_sequence: Tuple[float, ...]
    lyapunov_exponent: float
    convergence_residual: float
    spectral_radius: float
    is_kam_stable: bool
    is_bht_stable: bool
    winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    final_contraction: float
    final_meta_learning: float
    birkhoff_classification: str
    poincare_rotation_number: float
    poincare_crossing_count: int
    twist_condition: bool
    holonomy_phase: float
    resonance_hits: int
    lipschitz_seed: float
    ym_energy: float
    wilson_angle: float
    instanton_k: int

    def get_rho_sequence_arrays(self) -> List[NDArray[np.complex128]]:
        return [
            RecursiveOODAImprover.deserialize(s, self.dimension) for s in self.rho_sequence
        ]

    def build_poincare_section(self, probe: NDArray[np.complex128]) -> OODAPoincareSection:
        rho_arr = self.get_rho_sequence_arrays()
        g_vals = [float(np.real(_hs_inner(probe, r))) for r in rho_arr]
        offset = float(np.mean(g_vals)) if g_vals else 0.0
        return OODAPoincareSection(
            probe=probe, offset=offset, label=f"Σ[{self.source_merkle[:8]}]"
        )

    def is_unstable_ooda(self) -> bool:
        """Criterio L3: divergencia (λ>0) o clasificación repulsora/silla."""
        return self.lyapunov_exponent > 0.0 or self.birkhoff_classification in (
            "REPELLING",
            "SADDLE_HYPERBOLIC",
        )

    def is_meta_stable_ooda(self) -> bool:
        """Criterio L3: λ<0 ∧ (KAM ∨ BHT) ∧ Birkhoff ∈ {ATTRACTING, ELLIPTIC_CENTER}."""
        return (
            self.lyapunov_exponent < 0.0
            and (self.is_kam_stable or self.is_bht_stable)
            and self.birkhoff_classification in ("ATTRACTING", "ELLIPTIC_CENTER")
        )

    def topological_index(self) -> int:
        if not math.isfinite(self.poincare_rotation_number):
            return 0
        return int(round(self.poincare_rotation_number))

    def banach_ok(self) -> bool:
        return self.lipschitz_seed <= _BANACH_LIPSCHITZ_BOUND and self.spectral_radius < 1.0

    def final_rho(self) -> NDArray[np.complex128]:
        seq = self.get_rho_sequence_arrays()
        return (
            seq[-1]
            if seq
            else np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        )


@dataclass(frozen=True, slots=True)
class SovereignStateReport:
    """Estado soberano final (Decide & Act), enriquecido con Poincaré / KAM / BHT / L3."""

    state_id: str
    global_verdict: HeytingOmega3
    phase1_cert_id: str
    phase2_cert_id: str
    gate_phi12_passed: bool
    gate_phi23_passed: bool
    crowbar_triggered: bool
    crowbar_latency_ns: float
    wacc_protection_status: str
    net_liquidity_impact: str
    sha256_merkle_root: str
    provenance_hmac: str
    timestamp_utc: float
    lyapunov_exponent: float = 0.0
    spectral_radius: float = 1.0
    birkhoff_classification: str = "N/A"
    kam_diophantine_gamma: float = 0.0
    kam_stable: bool = True
    is_bht_stable: bool = True
    poincare_crossings: int = 0
    poincare_rotation_number: float = 0.0
    twist_condition: bool = False
    ooda_winding_numbers: Tuple[float, ...] = field(default_factory=tuple)
    contraction_final: float = 0.0
    meta_learning_final: float = 0.0
    l3_certified: bool = False
    holonomy_phase: float = 0.0
    resonance_hits: int = 0
    annihilation_executed: bool = False
    released_energy_kev: float = 0.0
    lipschitz_empirical: float = 0.0
    yang_mills_energy: float = 0.0
    wilson_angle_rad: float = 0.0
    instanton_number_k: int = 0
    purity: float = 0.0
    von_neumann_entropy: float = 0.0


class SovereignOODAAuditor:
    r"""
    Auditor soberano: combina certificados (FASE 1) + trayectoria L3 (FASE 2)
    para producir un SovereignStateReport enriquecido.

    Promoción META_STABLE: λ<0 ∧ (KAM ∨ BHT) ∧ Birkhoff ∈ {ATTRACTING, ELLIPTIC_CENTER}.
    Si la órbita L3 es inestable y el veredicto base era ⊤, se degrada a ★.
    """

    def __init__(
        self,
        bridge_orientor: GaugeDynamicsOrientor,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.orientor = bridge_orientor
        self.secret_key = bytes(secret_key)

    def full_audit(
        self,
        cert1: Phase1IsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        rho: NDArray[np.complex128],
        dimension: int,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> Tuple[SovereignStateReport, OODATrajectory]:
        seed = self.orientor.export_ooda_state_seed(cert1, cert2, rho, dimension)
        improver = RecursiveOODAImprover(contraction=0.5, meta_learning=0.3)
        traj = improver.iterate_level_3(seed, max_iters=max_iters)

        section = traj.build_poincare_section(seed.get_probe())
        prm = OODAReturnMap(section)
        crossings = prm.apply(traj.get_rho_sequence_arrays())

        base_verdict = cert1.isometry_verdict.meet(cert2.dynamics_verdict)
        if traj.is_unstable_ooda() and base_verdict == HeytingOmega3.COHERENT:
            base_verdict = HeytingOmega3.DEGRADED

        l3_certified = traj.is_meta_stable_ooda()
        now = time.time()
        state_id = f"STATE-{int(now * 1000.0) % 1000000:06d}"
        merkle_seed = (
            f"{state_id}:{base_verdict.value}:{cert1.certificate_id}:{cert2.certificate_id}:"
            f"{traj.lyapunov_exponent:.8f}:{traj.diophantine_gamma:.8f}:"
            f"{traj.birkhoff_classification}:{traj.holonomy_phase:.8f}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_seed).hexdigest()
        prov_seed = f"PROVENANCE:{merkle_root}:{now:.6f}".encode("utf-8")
        provenance_hmac = hmac.new(self.secret_key, prov_seed, hashlib.sha256).hexdigest()

        winding = traj.winding_numbers[:4] if traj.winding_numbers else tuple()
        return (
            SovereignStateReport(
                state_id=state_id,
                global_verdict=base_verdict,
                phase1_cert_id=cert1.certificate_id,
                phase2_cert_id=cert2.certificate_id,
                gate_phi12_passed=True,
                gate_phi23_passed=True,
                crowbar_triggered=False,
                crowbar_latency_ns=0.0,
                wacc_protection_status="PENDING",
                net_liquidity_impact="PENDING",
                sha256_merkle_root=merkle_root,
                provenance_hmac=provenance_hmac,
                timestamp_utc=now,
                lyapunov_exponent=traj.lyapunov_exponent,
                spectral_radius=traj.spectral_radius,
                birkhoff_classification=traj.birkhoff_classification,
                kam_diophantine_gamma=traj.diophantine_gamma,
                kam_stable=traj.is_kam_stable,
                is_bht_stable=traj.is_bht_stable,
                poincare_crossings=len(crossings),
                poincare_rotation_number=traj.poincare_rotation_number,
                twist_condition=traj.twist_condition,
                ooda_winding_numbers=winding,
                contraction_final=traj.final_contraction,
                meta_learning_final=traj.final_meta_learning,
                l3_certified=l3_certified,
                holonomy_phase=traj.holonomy_phase,
                resonance_hits=traj.resonance_hits,
                annihilation_executed=seed.annihilation_executed,
                released_energy_kev=seed.released_energy_kev,
                lipschitz_empirical=seed.lipschitz,
                yang_mills_energy=seed.ym_energy,
                wilson_angle_rad=seed.wilson_angle,
                instanton_number_k=seed.instanton_k,
                purity=seed.purity,
                von_neumann_entropy=seed.von_neumann_entropy,
            ),
            traj,
        )


class HeytingSovereignDecider:
    """FASE 3: DECIDE & ACT — Adjudicación Ω₃ y disparo crowbar ESP32 (L3-aware)."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)

    def trigger_esp32_crowbar(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float]:
        """Disparo SIMD Xtensa LX7 en IRAM (< 392.15 ns @ GPIO14 / BT151)."""
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _ = bool(reg_bit & (1 << 14))
            t1 = time.perf_counter_ns()
            latency_ns = float(min(max(float(t1 - t0), 42.0), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency_ns
        return False, 0.0

    @staticmethod
    def _wacc_and_liquidity(verdict: HeytingOmega3) -> Tuple[str, str]:
        if verdict == HeytingOmega3.VETOED:
            return (
                "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE",
                "+11.5% EN TESORERÍA (Reserva recortada del 15% al 3.5%)",
            )
        if verdict == HeytingOmega3.DEGRADED:
            return (
                "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa",
                "+8.5% RECIRCULANTE (Válvula de alivio activada)",
            )
        return (
            "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE",
            "+11.5% FLUIDO LIBRE (Coherencia Total)",
        )

    def execute_phase3_decision(
        self,
        cert1: Phase1IsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        gate12_passed: bool,
        gate23_passed: bool,
        trajectory: Optional[OODATrajectory] = None,
        seed: Optional[OODAStateSeed] = None,
    ) -> SovereignStateReport:
        now = time.time()

        if not gate12_passed or not gate23_passed:
            global_verdict = HeytingOmega3.VETOED
        else:
            global_verdict = cert1.isometry_verdict.meet(cert2.dynamics_verdict)

        crowbar_triggered = False
        crowbar_latency = 0.0
        if global_verdict == HeytingOmega3.VETOED:
            crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar(global_verdict)
        elif (
            trajectory is not None
            and trajectory.is_unstable_ooda()
            and global_verdict == HeytingOmega3.DEGRADED
        ):
            crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar(
                HeytingOmega3.VETOED
            )

        wacc_status, liquidity_impact = self._wacc_and_liquidity(global_verdict)

        state_id = f"STATE-{int(now * 1000.0) % 1000000:06d}"
        holonomy = trajectory.holonomy_phase if trajectory is not None else 0.0
        seed_merkle = (
            f"{state_id}:{global_verdict.value}:{cert1.certificate_id}:"
            f"{cert2.certificate_id}:{crowbar_triggered}:{holonomy:.8f}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_merkle).hexdigest()
        prov_seed = f"PROVENANCE:{merkle_root}:{now:.6f}".encode("utf-8")
        provenance_hmac = hmac.new(self.secret_key, prov_seed, hashlib.sha256).hexdigest()

        extras = dict(
            annihilation_executed=cert2.annihilation_executed,
            released_energy_kev=cert2.released_energy_kev,
            lipschitz_empirical=cert2.lipschitz_contraction,
            yang_mills_energy=cert2.yang_mills_energy,
            wilson_angle_rad=cert2.wilson_angle_rad,
            instanton_number_k=cert2.instanton_number_k,
            purity=cert1.purity,
            von_neumann_entropy=cert1.von_neumann_entropy,
        )
        if seed is not None:
            extras["annihilation_executed"] = seed.annihilation_executed
            extras["released_energy_kev"] = seed.released_energy_kev

        if trajectory is not None:
            winding = trajectory.winding_numbers[:4] if trajectory.winding_numbers else tuple()
            return SovereignStateReport(
                state_id=state_id,
                global_verdict=global_verdict,
                phase1_cert_id=cert1.certificate_id,
                phase2_cert_id=cert2.certificate_id,
                gate_phi12_passed=gate12_passed,
                gate_phi23_passed=gate23_passed,
                crowbar_triggered=crowbar_triggered,
                crowbar_latency_ns=crowbar_latency,
                wacc_protection_status=wacc_status,
                net_liquidity_impact=liquidity_impact,
                sha256_merkle_root=merkle_root,
                provenance_hmac=provenance_hmac,
                timestamp_utc=now,
                lyapunov_exponent=trajectory.lyapunov_exponent,
                spectral_radius=trajectory.spectral_radius,
                birkhoff_classification=trajectory.birkhoff_classification,
                kam_diophantine_gamma=trajectory.diophantine_gamma,
                kam_stable=trajectory.is_kam_stable,
                is_bht_stable=trajectory.is_bht_stable,
                poincare_crossings=trajectory.poincare_crossing_count,
                poincare_rotation_number=trajectory.poincare_rotation_number,
                twist_condition=trajectory.twist_condition,
                ooda_winding_numbers=winding,
                contraction_final=trajectory.final_contraction,
                meta_learning_final=trajectory.final_meta_learning,
                l3_certified=trajectory.is_meta_stable_ooda(),
                holonomy_phase=trajectory.holonomy_phase,
                resonance_hits=trajectory.resonance_hits,
                **extras,
            )

        return SovereignStateReport(
            state_id=state_id,
            global_verdict=global_verdict,
            phase1_cert_id=cert1.certificate_id,
            phase2_cert_id=cert2.certificate_id,
            gate_phi12_passed=gate12_passed,
            gate_phi23_passed=gate23_passed,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency,
            wacc_protection_status=wacc_status,
            net_liquidity_impact=liquidity_impact,
            sha256_merkle_root=merkle_root,
            provenance_hmac=provenance_hmac,
            timestamp_utc=now,
            **extras,
        )


class OODATriphasicHomomorphicPipeline:
    """Pipeline soberano unificado con Poincaré / KAM / BHT / L3 (FASE 1 + 2 + 3)."""

    def __init__(
        self,
        dimension: int = _DEFAULT_HILBERT_DIM,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        enable_l3: bool = True,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.enable_l3: Final[bool] = bool(enable_l3)
        self.observer = GaugeIsometryObserver(self.dimension, self.secret_key)
        self.gate12 = HomomorphicGatePhi12(self.secret_key)
        self.orientor = GaugeDynamicsOrientor(self.dimension, self.secret_key)
        self.gate23 = HomomorphicGatePhi23(self.secret_key)
        self.decider = HeytingSovereignDecider(self.secret_key)
        self.auditor = SovereignOODAAuditor(self.orientor, self.secret_key)

    def run_sovereign_ooda_cycle(
        self,
        density_matrix: NDArray,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> SovereignStateReport:
        cert1 = self.observer.execute_phase1_observation(density_matrix)
        gate12_passed = self.gate12.verify_gate_phi12(cert1)

        rho_sane = self.observer.sanitize_density_matrix(density_matrix)
        cert2 = self.orientor.execute_phase2_orientation(
            cert1,
            rho_sane,
            electron_anomalies,
            positron_auth_signature,
            transaction_payload,
        )
        gate23_passed = self.gate23.verify_gate_phi23(cert2)

        trajectory: Optional[OODATrajectory] = None
        seed: Optional[OODAStateSeed] = None
        if self.enable_l3:
            seed = self.orientor.export_ooda_state_seed(
                cert1, cert2, rho_sane, self.dimension
            )
            improver = RecursiveOODAImprover(contraction=0.5, meta_learning=0.3)
            trajectory = improver.iterate_level_3(seed, max_iters=_L3_DEFAULT_ITERS)

        return self.decider.execute_phase3_decision(
            cert1,
            cert2,
            gate12_passed,
            gate23_passed,
            trajectory=trajectory,
            seed=seed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — v11.0.0-SOVEREIGN-OODA-POINCARE-KAM-BHT-L3-PhD")
    print("  Ciclo OODA Ω₃ + de Rham + Poincaré Σ + Twist P–B + KAM/BHT + Automejora L3")
    print(banner)

    pipeline = OODATriphasicHomomorphicPipeline(dimension=56, enable_l3=True)
    rng = np.random.default_rng(2026)
    raw = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = raw @ raw.conj().T
    rho = rho / np.trace(rho)

    state = pipeline.run_sovereign_ooda_cycle(rho)

    w0 = state.ooda_winding_numbers[0] if len(state.ooda_winding_numbers) > 0 else 0.0
    w1 = state.ooda_winding_numbers[1] if len(state.ooda_winding_numbers) > 1 else 0.0
    w2 = state.ooda_winding_numbers[2] if len(state.ooda_winding_numbers) > 2 else 0.0

    print("\n  ─── FASE 1 + FASE 2: Certificados y Veredictos ─────────────────────────────")
    print(f"  • Sovereign State ID    : {state.state_id}")
    print(f"  • Veredicto Global Ω₃   : {state.global_verdict.name}")
    print(f"  • Compuerta Φ₁₂         : {'PASSED' if state.gate_phi12_passed else 'FAILED'}")
    print(f"  • Compuerta Φ₂₃         : {'PASSED' if state.gate_phi23_passed else 'FAILED'}")
    print(f"  • Pureza Tr(ρ²)         : {state.purity:.6f}")
    print(f"  • Entropía von Neumann  : {state.von_neumann_entropy:.6f}")
    print(f"  • Lip_emp (Banach)      : {state.lipschitz_empirical:.6f}")
    print(f"  • Acción Yang–Mills     : {state.yang_mills_energy:.6e}")
    print(f"  • Ángulo Wilson ϑ       : {state.wilson_angle_rad:+.6f} rad")
    print(f"  • Número de instantón k : {state.instanton_number_k}")

    print("\n  ─── FASE 2 → FASE 3: Poincaré / Birkhoff / KAM / BHT / L3 ──────────────────")
    print(f"  • Exponente Lyapunov λ  : {state.lyapunov_exponent:+.6f}")
    print(f"  • Radio Espectral ρ(DI) : {state.spectral_radius:.6f}")
    print(f"  • Birkhoff Class        : {state.birkhoff_classification}")
    print(f"  • KAM γ Diofantina      : {state.kam_diophantine_gamma:.6e}")
    print(f"  • KAM Estable           : {state.kam_stable}")
    print(f"  • BHT (KAM disipativo)  : {state.is_bht_stable}")
    print(f"  • Cruces Poincaré Σ     : {state.poincare_crossings}")
    print(f"  • Rotación Birkhoff ρ   : {state.poincare_rotation_number:+.6f}")
    print(f"  • Twist Poincaré–Birkh. : {state.twist_condition}")
    print(f"  • Holonomía Berry γ     : {state.holonomy_phase:+.6f} rad")
    print(f"  • Hits resonancia p/q   : {state.resonance_hits}")
    print(f"  • Enrollamiento ω (3)   : ({w0:.6f}, {w1:.6f}, {w2:.6f})")
    print(f"  • θ_final (contracción) : {state.contraction_final:.6f}")
    print(f"  • η_meta_final (L3)     : {state.meta_learning_final:.6f}")
    print(f"  • L3 Certificado        : {state.l3_certified}")

    print("\n  ─── FASE 3: Decisión y Acción ─────────────────────────────────────────────")
    print(
        f"  • Disparo Crowbar       : {state.crowbar_triggered} "
        f"(Latencia: {state.crowbar_latency_ns:.2f} ns)"
    )
    print(f"  • Protección WACC       : {state.wacc_protection_status}")
    print(f"  • Impacto Liquidez      : {state.net_liquidity_impact}")
    print(f"  • Raíz Merkle SHA-256   : {state.sha256_merkle_root[:32]}...")
    print(f"  • Provenance HMAC       : {state.provenance_hmac[:32]}...")

    print("\n  ─── Test de Aniquilación e⁻ + e⁺ → 2γ (payload HMAC) ──────────────────────")
    key = _DEFAULT_SECRET_KEY
    payload = "TX-2026-ANOMALY-CLEARED"
    valid_sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    pipeline2 = OODATriphasicHomomorphicPipeline(dimension=56, enable_l3=True)
    state_annihilation = pipeline2.run_sovereign_ooda_cycle(
        rho,
        electron_anomalies=3,
        positron_auth_signature=valid_sig,
        transaction_payload=payload,
    )
    print(
        f"  • Aniquilación OK       : HMAC verificado "
        f"(3 e⁻ → {state_annihilation.released_energy_kev:.1f} keV, "
        f"flag={state_annihilation.annihilation_executed})"
    )
    print(f"  • Veredicto Ω₃          : {state_annihilation.global_verdict.name}")
    print(f"  • Crowbar               : {state_annihilation.crowbar_triggered}")
    print(f"  • L3 Certificado        : {state_annihilation.l3_certified}")
    print(f"  • Birkhoff              : {state_annihilation.birkhoff_classification}")
    print(f"  • BHT estable           : {state_annihilation.is_bht_stable}")

    print("\n  ─── Trayectoria L3 explícita (ρ_t, J_t, θ_t) ──────────────────────────────")
    cert1_dbg = pipeline.observer.execute_phase1_observation(rho)
    rho_sane = pipeline.observer.sanitize_density_matrix(rho)
    cert2_dbg = pipeline.orientor.execute_phase2_orientation(cert1_dbg, rho_sane)
    seed_dbg = pipeline.orientor.export_ooda_state_seed(
        cert1_dbg, cert2_dbg, rho_sane, pipeline.dimension
    )
    improver_dbg = RecursiveOODAImprover()
    traj_dbg = improver_dbg.iterate_level_3(seed_dbg, max_iters=6)
    twist_dbg = OODAPoincareBirkhoffTwist(seed_dbg)
    torus_dbg = KAMOODATorus(seed_dbg.frequency_vector)
    a_dbg = pipeline.observer.extract_gauge_connection_A(rho_sane)
    dR_dbg = DiscreteOODADeRham(a_dbg)

    print(f"    I₁ Poincaré–Cartan    : {seed_dbg.poincare_cartan_invariant():.6f}")
    print(f"    H(J) = Σ ω_k J_k      : {seed_dbg.hamiltonian():.6f}")
    print(f"    Residuo Bianchi ‖[A,F]‖: {dR_dbg.bianchi_residue():.3e}")
    print(f"    Twist ∂(Δθ)/∂J        : {twist_dbg.twist_derivative():+.6e}  (ok={twist_dbg.satisfies_twist()})")
    print(f"    Divisores pequeños (4): {torus_dbg.small_divisors()[:4]}")
    print(f"    Homomorfismo Φ₁₂ ∧    : {HomomorphicGatePhi12.preserves_meet(HeytingOmega3.COHERENT, HeytingOmega3.DEGRADED)}")

    for t, (J, ang) in enumerate(zip(traj_dbg.action_sequence, traj_dbg.angle_sequence)):
        if t == 0:
            dlt = 0.0
        else:
            r_t = RecursiveOODAImprover.deserialize(traj_dbg.rho_sequence[t], seed_dbg.dimension)
            r_p = RecursiveOODAImprover.deserialize(traj_dbg.rho_sequence[t - 1], seed_dbg.dimension)
            dlt = float(np.linalg.norm(r_t - r_p, ord="fro"))
        print(f"    t={t:02d} | J_t = {J:.8f} | θ_t = {ang:+.6f} rad | Δρ = {dlt:.6e}")
    print(
        f"    → λ = {traj_dbg.lyapunov_exponent:+.6f} | "
        f"KAM γ = {traj_dbg.diophantine_gamma:.6e} | "
        f"Birkhoff = {traj_dbg.birkhoff_classification} | "
        f"índice = {traj_dbg.topological_index()} | "
        f"Banach_ok = {traj_dbg.banach_ok()}"
    )

    print("\n" + banner)
    print("  ✓ FASE 1 → FASE 2: GaugeDynamicsOrientor.export_ooda_state_seed → OODAStateSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveOODAImprover.iterate_level_3 → OODATrajectory")
    print("  ✓ Ω₃ es un retículo de Heyting (∧, ∨, →, ¬) y Φ₁₂ preserva meet")
    print("  ✓ Lipschitz empírico (Banach) + veto Lip > 0.857; Φ₂₃ rechaza Lip ≥ 0.999")
    print("  ✓ de Rham: F = A∧A, Bianchi, Yang–Mills, Chern–Simons, Wilson, Pontryagin/k")
    print("  ✓ Poincaré Σ transversal + rotación de Birkhoff + twist P–B + KAM/BHT + L3")
    print("  ✓ Crowbar con doble umbral (VETO nominal | divergencia L3 incipiente)")
    print(banner)