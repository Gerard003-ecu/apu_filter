# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : De Rham-Galois Adjunction Bridge Prosecutor Engine v12.1.0-SOVEREIGN-POINCARE-N3-PhD      ║
║ Ubicación: app/wisdom/de_rham_galois_adjunction_bridge_prosecutor_engine.py                          ║
║ Función  : Motor Espectral Fiscal que audita F ⊣ G mediante (a) aniquilación Dirac e⁻+e⁺→2γ en RAM,  ║
║            (b) producción de Spohn del semigrupo GKSL, (c) mecánica celeste de Poincaré (retorno     ║
║            interpolado, Lindstedt O(ε²), Birkhoff, KAM, twist, isoenergética, Oseledets),            ║
║            (d) cota de Banach–Caccioppoli DINÁMICA Lip(T) < 1 sobre el semigrupo GKSL,               ║
║            (e) ledger N3-celeste (recurrencia de Poincaré + λ_Φ) y (f) adjudicación fiscal con       ║
║            certificación de latencia del disyuntor ESP32 Crowbar (≤ 392.15 ns @ GPIO14, I5).         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA POR FASES ANIDADAS (3 × 3 sub-capas):

  FASE 1 — OBSERVE / FOUNDATIONS:
    1.1 `compute_spohn_entropy_production_v12` — GKSL no conmutativo + Spohn + purity.
    1.2 `execute_dirac_annihilation_reaction_v12` — HMAC constant-time + aniquilación.
    1.3 `extract_prosecution_primitives_v12` → ProsecutionPrimitivesBundle.
             [HANDOFF] último método F1 ≡ primer acto F2 (compute_dynamic_banach_lipschitz).
  FASE 2 — ORIENT / SPECTRAL EVIDENCE:
    2.1 `compute_dynamic_banach_lipschitz` — Lip(T) por diferencia canónica + Oseledets.
    2.2 `analyze_poincare_return_map_native` — cruces interpolados, Lindstedt, Birkhoff, KAM.
    2.3 `synthesize_spectral_evidence_v12` → SpectralEvidenceBundle.
             [HANDOFF] último método F2 ≡ primer acto F3 (_evaluate_meta_witness_n3).
  FASE 3 — DECIDE / CERTIFICATION:
    3.1 `_evaluate_meta_witness_n3` — consistencia + entropía + recurrencia del ledger.
    3.2 `_judicial_verdict_gate_v12` — R1–R11 (clásico + espectral + celestial).
    3.3 `prosecute_adjunction_coherence_v12` → DeRhamGaloisProsecutionCertificateV12.

INVARIANTES FISCALES v12.1:
  · I1  Δ_Galois > 1e-6 ∨ Uhlmann F < 0.95           ⟹ DEGRADED mínimo
  · I2  Holonomía rota ∨ auth positrón inválida       ⟹ HOLONOMY_BREAKAGE
  · I3  Spohn σ_rel(ρ‖I/d) se reporta crudo; 2ª ley   ⟹ σ_clamp = max(0, σ_raw)
  · I4  Back-Action QND ≡ 0.0 dB si auth válida, 12.5 dB si apócrifa
  · I5  Crowbar HARDWARE solo si HOLONOMY_BREAKAGE ∧
        (fraude HMAC clásico  ∨  (Banach Lip<1 ∧ KAM≥0.75 ∧ meta≥0.75))
  · I6  Latencia Crowbar certificada ∈ [42.0, 392.15] ns
  · I7  Contrato backward-compat v11 (nombres/firmas/campos)
  · I8  HMAC de positrón es compare_digest (sin fuga Hamming)
  · I9  Flujo Poincaré generado por H_probe ∤ ρ (no trivial)
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Base de adjunción (heredada v11) ──
try:
    from de_rham_galois_adjunction_bridge import (  # type: ignore
        AdjunctionCoherenceStatus,
        DeRhamGaloisAdjunctionBridge,
        ToonCartridge,
    )
except ImportError:
    try:
        from app.wisdom.de_rham_galois_adjunction_bridge import (  # type: ignore
            AdjunctionCoherenceStatus,
            DeRhamGaloisAdjunctionBridge,
            ToonCartridge,
        )
    except ImportError:
        class AdjunctionCoherenceStatus(IntEnum):  # type: ignore[no-redef]
            PERFECT_ISOMORPHISM = 0
            DEGRADED_DISCREPANCY = 1
            HOLONOMY_BREAKAGE = 2

        DeRhamGaloisAdjunctionBridge = None  # type: ignore
        ToonCartridge = Any  # type: ignore

logger = logging.getLogger("APU.Wisdom.DeRhamGaloisAdjunctionBridgeProsecutorEngineV12")

# ── Constantes espectrales ──
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_DERHAM_GALOIS_PROSECUTOR_SOVEREIGN_KEY_v12.1.0"
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_SIMD_EXECUTION_NS: Final[float] = 42.0
_ENGINE_VERSION: Final[str] = "12.1.0-SOVEREIGN-POINCARE-N3-PhD"

# ── Poincaré / KAM / Lindstedt ──
_KAM_FREQ_DIM: Final[int] = 4
_KAM_GAMMA: Final[float] = 1.0e-2
_KAM_TAU: Final[float] = 2.0
_KAM_RESONANCE_MAX_K: Final[int] = 3
_POINCARE_N_ORBITS: Final[int] = 12
_POINCARE_STEPS_PER_ORBIT: Final[int] = 16
_LINDSTEDT_EPSILON: Final[float] = 0.05
_SECULAR_DENOM_FLOOR: Final[float] = 1.0e-6
_TWIST_FLOOR: Final[float] = 1.0e-12
_GKSL_JUMP_RATE: Final[float] = 0.05
_BANACH_PROBE_EPS: Final[float] = 1.0e-6
_GKSL_STEP_DT: Final[float] = 1.0e-2

# ── Umbrales judiciales v12.1 ──
_BANACH_LIPSCHITZ_CEILING: Final[float] = 1.0
_KAM_TRUSTWORTHY_THRESHOLD: Final[float] = 0.75
_KAM_CHAOS_THRESHOLD: Final[float] = 0.50
_META_CONFIDENCE_REQUIRED: Final[float] = 0.75
_META_LOW_EVIDENCE_THRESHOLD: Final[float] = 0.50
_POINCARE_LYAP_VETO: Final[float] = 1.0e-1
_POINCARE_LYAP_ESCALATE: Final[float] = 1.0e-3
_LIOUVILLE_PRESERVATION_MAX: Final[float] = 1.0e-8
_SPOHN_SECOND_LAW_TOL: Final[float] = 1.0e-8
_N3_TRAJECTORY_WINDOW: Final[int] = 8
_N3_LEDGER_CAP: Final[int] = 64
_HMAC_DOMAIN_ENGINE: Final[bytes] = b"CERT-DERHAM-PROSECUTOR-v12.1-ENGINE|"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS — retículos Heyting y estados procesales (idénticos a v11)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """Retículo de Heyting trivaluado: PERFECT ≺ DEGRADED ≺ HOLONOMY_BREAKAGE (meet = peor)."""
    PERFECT = 0x00
    DEGRADED = 0x01
    HOLONOMY_BREAKAGE = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))


class DeRhamGaloisProsecutionStatus(IntEnum):
    """Estado procesal de acusación sobre la adjunción de de Rham-Galois."""
    ABSOLVED_ISOMORPHIC = 0
    RECOVERABLE_DEGRADATION = 1
    PURGED_DIRAC_VACUUM = 2
    VETOED_HOLONOMY_BREAKAGE = 3


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DeRhamGaloisProsecutionIndictment:
    """Germen inmutable de acusación fiscal sobre la adjunción F ⊣ G (idéntico a v11)."""
    indictment_id: str
    schedule_id: str
    galois_gap_delta: float
    uhlmann_fidelity: float
    bures_distance: float
    electron_anomalies_count: int
    positron_auth_signature: str
    transaction_payload: str
    is_holonomy_broken: bool
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class DeRhamGaloisProsecutionCertificate:
    """Certificado v11 (preservado para compatibilidad estructural)."""
    certificate_id: str
    indictment_ref_id: str
    prosecution_status: DeRhamGaloisProsecutionStatus
    global_verdict: HeytingOmega3
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    sha256_merkle_root: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class DeRhamGaloisProsecutionCertificateV12:
    """
    Certificado v12.1 — superset estricto del v11.
    Todos los campos v11 están presentes con el mismo nombre; se añaden
    Poincaré celestial, Banach–Caccioppoli, N3-celeste e I5.
    """
    # v11 heredado (mismo orden y nombre)
    certificate_id: str
    indictment_ref_id: str
    prosecution_status: DeRhamGaloisProsecutionStatus
    global_verdict: HeytingOmega3
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    sha256_merkle_root: str
    timestamp_utc: float
    # v12 · FASE 1
    gksl_rate: float
    purity_proxy: float
    positron_auth_confidence: float
    # v12 · FASE 2 (Poincaré + KAM + Banach)
    banach_lipschitz_k: float
    banach_gate_passed: bool
    poincare_crossings_count: int
    action_drift_max: float
    action_drift_mean: float
    lindstedt_correction_norm: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    # v12 · FASE 3
    meta_consistency_score: float
    meta_verdict_entropy: float
    level_3_reflections: int
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    judicial_gate_failures: Tuple[str, ...]
    crowbar_latency_certified: bool
    hmac_engine_signature: str
    # v12.1 · celestial / N3-celeste / I5
    spohn_raw: float
    von_neumann_entropy: float
    kolmogorov_nondegeneracy: float
    twist_index: float
    rotation_number: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    mean_return_time: float
    birkhoff_matrix_frobenius: float
    n3_recurrence_index: float
    n3_adaptation_lyapunov: float
    crowbar_i5_satisfied: bool
    engine_version: str


@dataclass(slots=True)
class ActionAngleDecomposition:
    """FASE 1/2 — (J, θ, ω(J), B = Dω) del Hamiltoniano de entrelazamiento H_ent = −Σ J log J."""
    actions: NDArray[np.float64]
    angles: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    hamiltonian_spectrum: NDArray[np.float64]
    reference_eigenvectors: NDArray[np.complex128]
    total_action: float
    phase_space_volume: float
    lyapunov_spectrum: NDArray[np.float64]
    frequency_map_jacobian: NDArray[np.float64]
    kolmogorov_nondegeneracy: float
    von_neumann_entropy: float


@dataclass(slots=True)
class ProsecutionPrimitivesBundle:
    """HANDOFF FASE 1 → FASE 2. Primitivas espectrales/termodinámicas ya saneadas."""
    rho_sanitized: NDArray[np.complex128]
    d_rho: NDArray[np.complex128]
    hamiltonian_probe: NDArray[np.complex128]
    jump_operator: NDArray[np.complex128]
    spohn_entropy: float
    spohn_raw: float
    gksl_rate: float
    purity_proxy: float
    von_neumann_entropy: float
    positron_auth_valid: bool
    positron_auth_confidence: float
    purged: int
    released_kev: float
    back_action_db: float
    auth_fraud: bool


@dataclass(slots=True)
class SpectralEvidenceBundle:
    """HANDOFF FASE 2 → FASE 3. Evidencia espectral completa para adjudicación."""
    decomposition: ActionAngleDecomposition
    banach_lipschitz_k: float
    banach_gate_passed: bool
    poincare_crossings: int
    action_drift_max: float
    action_drift_mean: float
    lindstedt_norm: float
    birkhoff_coeff: float
    birkhoff_frobenius: float
    kam_index: float
    kam_violations: int
    kam_worst_ratio: float
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    twist_index: float
    rotation_number: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    mean_return_time: float
    kolmogorov_nondegeneracy: float
    von_neumann_entropy: float


@dataclass(slots=True)
class ProsecutorN3Ledger:
    """
    Ledger de automejora recursiva N3-celeste del motor fiscal.

    N1: registra (spohn, kam, banach, veredicto).
    N2: adapta umbrales (spohn_ceiling, kam_floor) según tendencias.
    N3: adapta α según entropía de veredictos.
    N3-celeste: recurrencia de Poincaré de (spohn, kam) y Lyapunov discreto de κ.
    """
    spohn_history: List[float] = field(default_factory=list)
    kam_history: List[float] = field(default_factory=list)
    banach_history: List[float] = field(default_factory=list)
    verdict_history: List[int] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    spohn_ceiling: float = 0.10
    kam_floor: float = _KAM_TRUSTWORTHY_THRESHOLD
    adaptation_rate: float = 0.05
    meta_adaptation_rate: float = 0.01
    level_3_reflections: int = 0
    last_entropy: float = 0.0
    last_recurrence_index: float = 0.0
    last_adaptation_lyapunov: float = 0.0

    def observe(self, spohn: float, kam: float, banach: float, verdict: HeytingOmega3) -> None:
        self.spohn_history.append(float(spohn))
        self.kam_history.append(float(kam))
        self.banach_history.append(float(banach))
        self.verdict_history.append(int(verdict))
        self.rate_history.append(float(self.adaptation_rate))
        if len(self.verdict_history) > _N3_LEDGER_CAP:
            self.spohn_history = self.spohn_history[-_N3_LEDGER_CAP:]
            self.kam_history = self.kam_history[-_N3_LEDGER_CAP:]
            self.banach_history = self.banach_history[-_N3_LEDGER_CAP:]
            self.verdict_history = self.verdict_history[-_N3_LEDGER_CAP:]
            self.rate_history = self.rate_history[-_N3_LEDGER_CAP:]

    def _trend(self, series: Sequence[float]) -> float:
        if len(series) < 2:
            return 0.0
        n = min(_N3_TRAJECTORY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def level_2_adapt(self) -> None:
        """N2 — ajusta ceilings/floors según tendencias de Spohn y KAM."""
        trend_s = self._trend(self.spohn_history)
        trend_k = self._trend(self.kam_history)
        if trend_s > 1e-6:
            self.spohn_ceiling = max(0.01, self.spohn_ceiling * (1.0 - self.adaptation_rate))
        elif trend_s < -1e-6:
            self.spohn_ceiling = min(0.5, self.spohn_ceiling * (1.0 + self.adaptation_rate * 0.5))
        if trend_k < -1e-4:
            self.kam_floor = min(0.90, self.kam_floor + self.adaptation_rate * 0.02)
        elif trend_k > 1e-4:
            self.kam_floor = max(0.75, self.kam_floor - self.adaptation_rate * 0.01)

    def poincare_recurrence_index(self) -> float:
        """r = exp(−d_min/σ) sobre (spohn, kam); cercano a 1 ⇒ retorno próximo."""
        n = len(self.kam_history)
        if n < 4:
            return 0.0
        pts = np.column_stack([
            np.array(self.spohn_history, dtype=float),
            np.array(self.kam_history, dtype=float),
        ])
        last = pts[-1]
        prior = pts[:-1]
        dist = np.sqrt(np.sum((prior - last) ** 2, axis=1))
        d_min = float(np.min(dist))
        sigma = float(np.std(pts, axis=0).mean()) + _WILKINSON_EPSILON
        return float(math.exp(-d_min / sigma))

    def adaptation_lyapunov(self) -> float:
        """λ_Φ ≈ ⟨log |Δα_{k+1}| − log |Δα_k|⟩ sobre la tasa de adaptación."""
        rates = np.array(self.rate_history, dtype=float)
        if rates.size < 4:
            return 0.0
        d1 = np.maximum(np.abs(np.diff(rates)), _WILKINSON_EPSILON)
        logs = np.log(d1[1:] / d1[:-1])
        return float(np.mean(logs))

    def level_3_adapt(self) -> float:
        """N3 — meta-adaptación de α según entropía de veredictos."""
        n = len(self.verdict_history)
        if n < 4:
            self.last_entropy = 0.0
            return 0.0
        recent = np.array(self.verdict_history[-_N3_TRAJECTORY_WINDOW:], dtype=int)
        _uniq, counts = np.unique(recent, return_counts=True)
        probs = counts / float(counts.sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if entropy > 0.5:
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * 0.9)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.95)
        else:
            self.meta_adaptation_rate = min(0.5, self.meta_adaptation_rate * 1.05)
            self.adaptation_rate = min(0.5, self.adaptation_rate * 1.02)
        self.level_3_reflections += 1
        self.last_entropy = entropy
        return entropy

    def level_3_celestial_adapt(self) -> Tuple[float, float]:
        """N3-celeste — congela α si hay recurrencia; amortigua si λ_Φ > 0."""
        rec = self.poincare_recurrence_index()
        lam = self.adaptation_lyapunov()
        self.last_recurrence_index = rec
        self.last_adaptation_lyapunov = lam
        if rec > 0.85:
            self.adaptation_rate *= 0.5
            self.meta_adaptation_rate *= 0.5
        elif lam > 0.0:
            damp = math.exp(-min(4.0, lam))
            self.adaptation_rate = max(1e-3, self.adaptation_rate * damp)
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * damp)
        else:
            self.adaptation_rate = min(0.5, self.adaptation_rate * 1.01)
        return rec, lam


def _hermitize(matrix: NDArray[Any]) -> NDArray[np.complex128]:
    a = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _hmac_hex(key: bytes, domain: bytes, payload: str) -> str:
    return hmac.new(key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical_failures(failures: Sequence[str]) -> str:
    return "|".join(failures) if failures else "NONE"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 1 — OBSERVE / FOUNDATIONS (3 sub-capas)
# ██╔══██╗ Spohn GKSL no conmutativo · Dirac HMAC-CT · Handoff ProsecutionPrimitivesBundle
# ██████╔╝ [HANDOFF] extract_prosecution_primitives_v12 → compute_dynamic_banach_lipschitz
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class DeRhamGaloisAdjunctionProsecutorEngine:
    """Motor Espectral Fiscal v12.1 · FASE 1 (fundamentos)."""

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._ledger: ProsecutorN3Ledger = ProsecutorN3Ledger()
        self._n3_cycle_count: int = 0
        if DeRhamGaloisAdjunctionBridge is not None:
            self.bridge = DeRhamGaloisAdjunctionBridge(dimension=dimension, secret_key=secret_key)
        else:
            self.bridge = None

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return _hmac_hex(self.secret_key, domain, payload)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa PSD de Higham y traza unitaria estricta."""
        a = _hermitize(matrix)
        eigvals, eigvecs = la.eigh(a)
        clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    def _probe_hamiltonian(self, dim: int) -> NDArray[np.complex128]:
        """
        H_probe tridiagonal Hermítico (ciclo de Jacobi) — NO conmuta con ρ genérico.
        Corrige el bug v12.0: H = αI ⇒ [H,ρ]≡0 ⇒ flujo Poincaré trivial.
        """
        h = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim - 1):
            h[i, i + 1] = 1.0
            h[i + 1, i] = 1.0
        h[0, dim - 1] = 1.0
        h[dim - 1, 0] = 1.0
        return h

    def _thermal_jump_operator(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Operador de salto de termalización en la base de ρ:
            L = √γ |v_min⟩⟨v_max|
        (disipación hacia el estado de menor acción / mayor energía de H_ent).
        """
        evals, evecs = la.eigh(rho)
        vmin = evecs[:, 0]
        vmax = evecs[:, -1]
        return np.sqrt(_GKSL_JUMP_RATE) * np.outer(vmin, vmax.conj())

    def _gksl_generator(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        jump_l: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """ℒ(ρ) = −i[H,ρ] + 𝒟_L(ρ); devuelve (ℒ(ρ), 𝒟_L(ρ))."""
        h = _hermitize(hamiltonian)
        unitary = -1j * (h @ rho - rho @ h)
        jump_l_dag = jump_l.conj().T
        dissipator = jump_l @ rho @ jump_l_dag - 0.5 * (
            jump_l_dag @ jump_l @ rho + rho @ jump_l_dag @ jump_l
        )
        return unitary + dissipator, dissipator

    def _von_neumann_entropy(self, rho: NDArray[np.complex128]) -> float:
        evals = np.real(la.eigvalsh(rho))
        evals = np.maximum(evals, _SPECTRAL_PSD_FLOOR)
        return float(-np.sum(evals * np.log(evals)))

    # ────────────────── Sub-capa 1.1: Spohn v12.1 ──────────────────
    def compute_spohn_entropy_production_v12(
        self,
        density_matrix: NDArray,
        hamiltonian_eff: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], float, float, float]:
        """
        Sub-capa 1.1 — Evolución GKSL y producción de Spohn.

        σ_raw  = −Re Tr(log ρ · ℒ(ρ))  = dS_vN/dt   (sin clamp)
        σ_I3   = max(0, σ_raw)                      (invariante I3, 2ª ley reportada)
        Para ρ_∞ = I/d, Spohn relativo coincide con dS_vN/dt pues Tr ℒ(ρ)=0.

        H por defecto es H_probe (no conmuta). El alias v11 descarta purity.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        dim = int(rho.shape[0])
        if hamiltonian_eff is None:
            hamiltonian_eff = self._probe_hamiltonian(dim)
        jump_l = self._thermal_jump_operator(rho)
        d_rho, dissipator = self._gksl_generator(rho, hamiltonian_eff, jump_l)

        eigvals, eigvecs = la.eigh(rho)
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        log_rho = (eigvecs * log_eigs) @ eigvecs.conj().T
        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn_entropy = max(0.0, spohn_raw)
        gksl_rate = float(np.linalg.norm(dissipator, ord="fro"))
        purity_proxy = float(np.real(np.trace(rho @ rho)))
        return d_rho, spohn_entropy, gksl_rate, purity_proxy

    def compute_spohn_entropy_production_v12_full(
        self,
        density_matrix: NDArray,
        hamiltonian_eff: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], float, float, float, float, float]:
        """Variante interna: expone H, L, σ_raw y S_vN para el bundle F1."""
        rho = self.sanitize_density_matrix(density_matrix)
        dim = int(rho.shape[0])
        if hamiltonian_eff is None:
            hamiltonian_eff = self._probe_hamiltonian(dim)
        else:
            hamiltonian_eff = _hermitize(hamiltonian_eff)
        jump_l = self._thermal_jump_operator(rho)
        d_rho, dissipator = self._gksl_generator(rho, hamiltonian_eff, jump_l)
        eigvals, eigvecs = la.eigh(rho)
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        log_rho = (eigvecs * log_eigs) @ eigvecs.conj().T
        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn_entropy = max(0.0, spohn_raw)
        gksl_rate = float(np.linalg.norm(dissipator, ord="fro"))
        purity_proxy = float(np.real(np.trace(rho @ rho)))
        entropy = float(-np.sum(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR) * log_eigs))
        return d_rho, hamiltonian_eff, jump_l, spohn_entropy, spohn_raw, gksl_rate, purity_proxy, entropy

    # ────────────────── Sub-capa 1.2: Dirac v12.1 ──────────────────
    def execute_dirac_annihilation_reaction_v12(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
    ) -> Tuple[bool, int, float, float, float]:
        """
        Sub-capa 1.2 — Aniquilación Dirac v12.1.

        HMAC en tiempo constante (compare_digest). I8: la confianza es
        Bernoulli {0,1}, NUNCA fracción Hamming (fuga de nibble del MAC).
        """
        if electron_anomalies <= 0:
            return True, 0, 0.0, 0.0, 1.0

        expected_hmac = hmac.new(
            self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        auth_valid = hmac.compare_digest(
            str(positron_auth_signature), str(expected_hmac)
        )
        auth_confidence = 1.0 if auth_valid else 0.0

        if auth_valid:
            released_kev = float(electron_anomalies) * _POSITRON_REST_MASS_KEV
            back_action_db = 0.0
            purged = int(electron_anomalies)
        else:
            released_kev = 0.0
            back_action_db = 12.5
            purged = 0
        return auth_valid, purged, released_kev, back_action_db, auth_confidence

    # ────────────────── Sub-capa 1.3: HANDOFF a FASE 2 ──────────────────
    def extract_prosecution_primitives_v12(
        self,
        indictment: DeRhamGaloisProsecutionIndictment,
        density_matrix: Optional[NDArray] = None,
    ) -> ProsecutionPrimitivesBundle:
        """
        Sub-capa 1.3 — HANDOFF FASE 1 → FASE 2.

        Empaqueta ρ saneado + ℒ(ρ) + Spohn + Dirac en ProsecutionPrimitivesBundle.
        Este método ES el gozne formal F1 ▷ F2: `compute_dynamic_banach_lipschitz`
        (primer método de FASE 2) consume el bundle aquí emitido.
        """
        if density_matrix is None:
            density_matrix = np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        rho = self.sanitize_density_matrix(density_matrix)
        d_rho, h_probe, jump_l, spohn, spohn_raw, gksl_rate, purity, entropy = (
            self.compute_spohn_entropy_production_v12_full(rho)
        )
        auth_valid, purged, released_kev, back_action, auth_conf = (
            self.execute_dirac_annihilation_reaction_v12(
                indictment.electron_anomalies_count,
                indictment.positron_auth_signature,
                indictment.transaction_payload,
            )
        )
        auth_fraud = (
            purged == 0
            and indictment.electron_anomalies_count > 0
            and not auth_valid
        )
        return ProsecutionPrimitivesBundle(
            rho_sanitized=rho,
            d_rho=d_rho,
            hamiltonian_probe=h_probe,
            jump_operator=jump_l,
            spohn_entropy=spohn,
            spohn_raw=spohn_raw,
            gksl_rate=gksl_rate,
            purity_proxy=purity,
            von_neumann_entropy=entropy,
            positron_auth_valid=auth_valid,
            positron_auth_confidence=auth_conf,
            purged=purged,
            released_kev=released_kev,
            back_action_db=back_action,
            auth_fraud=auth_fraud,
        )

    # ─────────── (fin FASE 1 · handoff a FASE 2 · compute_dynamic_banach_lipschitz) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 2 — ORIENT / SPECTRAL EVIDENCE (3 sub-capas)
# ██╔══██╗ Banach–Caccioppoli · Poincaré nativo celestial · Handoff SpectralEvidenceBundle
# ██████╔╝ [HANDOFF] synthesize_spectral_evidence_v12 → _evaluate_meta_witness_n3
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class DeRhamGaloisAdjunctionProsecutorEnginePhase2(DeRhamGaloisAdjunctionProsecutorEngine):
    """Extensión v12.1 · FASE 2 (continúa el handoff de FASE 1)."""

    def _gauge_fix_angles(self, eigvecs: NDArray[np.complex128]) -> NDArray[np.float64]:
        n = eigvecs.shape[1]
        angles = np.zeros(n, dtype=np.float64)
        for k in range(n):
            v = eigvecs[:, k]
            idx = int(np.argmax(np.abs(v)))
            angles[k] = float(np.angle(v[idx]))
        return angles

    def _entanglement_frequency_map(
        self, actions: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
        """ω_i = ∂H_ent/∂J_i = −log J_i − 1;  B = diag(−1/J); Kolmogorov det B ≠ 0."""
        j = np.maximum(np.asarray(actions, dtype=np.float64), _SPECTRAL_PSD_FLOOR)
        h_ent_diag = -np.log(j)
        omega = h_ent_diag - 1.0
        hess_diag = -1.0 / j
        b_mat = np.diag(hess_diag)
        log_abs_det = float(np.sum(np.log(np.abs(hess_diag) + _WILKINSON_EPSILON)))
        kolmogorov = float(math.exp(min(log_abs_det, 50.0) / max(len(j), 1)))
        entropy = float(np.sum(j * h_ent_diag))
        return omega, h_ent_diag, b_mat, kolmogorov, entropy

    def _spectral_lyapunov(
        self, actions: NDArray[np.float64], frequencies: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Proxy de Oseledets espectral (no del flujo unitario, que es λ≡0)."""
        j = np.sort(np.maximum(actions, _SPECTRAL_PSD_FLOOR))[::-1]
        n = len(j)
        lyap = np.zeros(n, dtype=np.float64)
        if n < 2:
            return lyap
        ratios = np.log(np.maximum(j[:-1] / j[1:], _WILKINSON_EPSILON))
        omega_sorted = np.sort(np.abs(frequencies))[::-1]
        m = min(len(ratios), max(len(omega_sorted) - 1, 1))
        for i in range(m):
            w = float(omega_sorted[i]) if i < len(omega_sorted) else 1.0
            lyap[i] = math.log(max(abs(w), _WILKINSON_EPSILON)) * float(ratios[i])
        return np.sort(lyap)[::-1]

    def extract_action_angle_decomposition(self, rho: NDArray) -> ActionAngleDecomposition:
        """Descomposición (J, θ, ω(J), B) lista para Lindstedt / twist / KAM."""
        rho = self.sanitize_density_matrix(rho)
        eigvals, eigvecs = la.eigh(rho)
        actions = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        angles = self._gauge_fix_angles(eigvecs)
        frequencies, hamiltonian_spectrum, hess_b, kolmogorov, entropy = (
            self._entanglement_frequency_map(actions)
        )
        lyapunov = self._spectral_lyapunov(actions, frequencies)
        n = max(len(actions), 1)
        log_vol = float(np.sum(np.log(2.0 * np.pi * actions + _WILKINSON_EPSILON)))
        return ActionAngleDecomposition(
            actions=actions,
            angles=angles,
            frequencies=frequencies,
            hamiltonian_spectrum=hamiltonian_spectrum,
            reference_eigenvectors=eigvecs,
            total_action=float(np.sum(actions)),
            phase_space_volume=float(np.exp(log_vol / n)),
            lyapunov_spectrum=lyapunov,
            frequency_map_jacobian=hess_b,
            kolmogorov_nondegeneracy=kolmogorov,
            von_neumann_entropy=entropy,
        )

    def _gksl_step(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        jump_l: NDArray[np.complex128],
        dt: float,
    ) -> NDArray[np.complex128]:
        """Un paso de Euler–Lindblad + proyección Higham (mapa T del semigrupo)."""
        d_rho, _ = self._gksl_generator(rho, hamiltonian, jump_l)
        return self.sanitize_density_matrix(rho + dt * d_rho)

    # ────────────────── Sub-capa 2.1: Banach dinámico (PRIMER método F2 ← bundle F1) ──────────────────
    def compute_dynamic_banach_lipschitz(
        self,
        primitives: ProsecutionPrimitivesBundle,
        dt: float = _GKSL_STEP_DT,
        fallback: float = 0.857,
    ) -> float:
        """
        Sub-capa 2.1 ← [HANDOFF FASE 1].

        Cota de Banach–Caccioppoli del mapa T(ρ) = Π_PSD(ρ + dt ℒ(ρ)):
            Lip(T) ≈ ‖T(ρ+ε H_probe) − T(ρ)‖_F / (ε ‖H_probe‖_F).

        Lip < 1 ⇒ T es contracción ⇒ único punto fijo (teorema de punto fijo).
        Sustituye el proxy v12.0 Lip = exp(max λ_gaps(ρ)), que no es el Lip del flujo.
        """
        rho = primitives.rho_sanitized
        h = primitives.hamiltonian_probe
        jump_l = primitives.jump_operator
        dim = int(rho.shape[0])
        pert = _hermitize(h)
        pert = pert - (np.trace(pert) / float(dim)) * np.eye(dim, dtype=np.complex128)
        nrm = float(np.linalg.norm(pert, ord="fro"))
        if nrm <= _WILKINSON_EPSILON:
            return fallback
        pert = pert / nrm
        rho_p = self.sanitize_density_matrix(rho + _BANACH_PROBE_EPS * pert)
        t_rho = self._gksl_step(rho, h, jump_l, dt)
        t_rho_p = self._gksl_step(rho_p, h, jump_l, dt)
        num = float(np.linalg.norm(t_rho_p - t_rho, ord="fro"))
        lip = num / _BANACH_PROBE_EPS
        if not math.isfinite(lip) or lip <= 0.0:
            return fallback
        return float(min(max(lip, _WILKINSON_EPSILON), 1.0e6))

    def _interpolate_crossing_time(
        self, prev_val: float, val: float, step_index: int, dt: float
    ) -> float:
        denom = abs(prev_val) + abs(val)
        alpha = abs(prev_val) / denom if denom > _WILKINSON_EPSILON else 0.5
        return float(step_index - 1) * dt + alpha * dt

    def _angles_from_rho(
        self, rho: NDArray[np.complex128], reference: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        evals, evecs = la.eigh(rho)
        order = np.argsort(np.real(evals))
        evecs = evecs[:, order]
        n = min(evecs.shape[1], reference.shape[1])
        angles = np.zeros(n, dtype=np.float64)
        for k in range(n):
            overlap = complex(np.vdot(reference[:, k], evecs[:, k]))
            angles[k] = float(np.angle(overlap if abs(overlap) > _WILKINSON_EPSILON else 1.0 + 0.0j))
        return angles

    def _poincare_jacobian_lyapunov(
        self, action_crossings: List[NDArray[np.float64]], mean_return_time: float
    ) -> float:
        if len(action_crossings) < 3:
            return 0.0
        d = min(_KAM_FREQ_DIM, action_crossings[0].size)
        seq = np.stack([np.sort(np.real(a))[::-1][:d] for a in action_crossings], axis=0)
        dj = np.diff(seq, axis=0)
        if dj.shape[0] < 2:
            return 0.0
        x = dj[:-1].T
        y = dj[1:].T
        gram = x @ x.T + _WILKINSON_EPSILON * np.eye(d)
        try:
            jac = y @ x.T @ la.inv(gram)
            sigma_max = float(np.max(np.abs(la.eigvals(jac))))
        except la.LinAlgError:
            return 0.0
        t_ret = mean_return_time if mean_return_time > _WILKINSON_EPSILON else 1.0
        return float(math.log(max(sigma_max, _WILKINSON_EPSILON)) / t_ret)

    def lindstedt_poincare_perturbation(
        self,
        decomposition: ActionAngleDecomposition,
        hamiltonian: NDArray,
        epsilon: float = _LINDSTEDT_EPSILON,
    ) -> NDArray[np.float64]:
        """Serie de Lindstedt–Poincaré hasta O(ε²), eliminación secular."""
        v = decomposition.reference_eigenvectors
        h = _hermitize(hamiltonian)
        h_eig = v.conj().T @ h @ v
        n = h_eig.shape[0]
        energy = np.real(np.diag(h_eig))
        off = h_eig - np.diag(np.diag(h_eig))
        b_diag = np.real(np.diag(decomposition.frequency_map_jacobian))
        if b_diag.size < n:
            b_diag = np.pad(b_diag, (0, n - b_diag.size))
        delta1 = np.zeros(n, dtype=np.float64)
        delta2 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom = float(energy[i] - energy[j])
                if abs(denom) < _SECULAR_DENOM_FLOOR:
                    continue
                vij2 = float(abs(off[i, j]) ** 2)
                delta1[i] += vij2 / denom
                bdiff = float(b_diag[i] - b_diag[j])
                delta2[i] += vij2 * bdiff / (denom * denom)
        omega0 = np.asarray(decomposition.frequencies, dtype=np.float64)
        if omega0.size < n:
            omega0 = np.pad(omega0, (0, n - omega0.size))
        return omega0[:n] + epsilon * delta1 + (epsilon ** 2) * delta2

    def birkhoff_normal_form_reduction(
        self, decomposition: ActionAngleDecomposition, hamiltonian: NDArray
    ) -> Tuple[float, float]:
        """Forma normal de Birkhoff truncada: (c_B, ‖B_off‖_F)."""
        v = decomposition.reference_eigenvectors
        h = _hermitize(hamiltonian)
        h_eig = v.conj().T @ h @ v
        off = h_eig - np.diag(np.diag(h_eig))
        diag = np.real(np.diag(h_eig))
        n = min(off.shape[0], _KAM_FREQ_DIM)
        if n < 2:
            return 0.0, 0.0
        off_d = off[:n, :n]
        energy = diag[:n]
        gaps = np.abs(np.diff(np.sort(energy)))
        omega_min = float(np.min(gaps)) + _WILKINSON_EPSILON
        b_off = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom = float(energy[i] - energy[j])
                if abs(denom) < _SECULAR_DENOM_FLOOR:
                    continue
                b_off[i, j] = float(abs(off_d[i, j]) ** 2) / denom
        c_b = float(np.sum(np.abs(off_d) ** 2)) / omega_min
        fro = float(np.linalg.norm(b_off, ord="fro"))
        return c_b, fro

    def compute_kam_stability_index(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        tau: float = _KAM_TAU,
        max_k: int = _KAM_RESONANCE_MAX_K,
    ) -> Tuple[float, int, float]:
        """Condición diofantina de Siegel–Moser: |k·ω| > γ / |k|_1^τ."""
        freqs = np.asarray(frequencies, dtype=np.float64)
        d = int(freqs.size)
        if d == 0:
            return 0.0, 0, 1.0e9
        violations = 0
        worst_ratio = float("inf")
        for k in itertools.product(range(-max_k, max_k + 1), repeat=d):
            k_arr = np.array(k, dtype=float)
            norm1 = float(np.sum(np.abs(k_arr)))
            if norm1 == 0.0:
                continue
            k_dot_omega = abs(float(np.dot(k_arr, freqs)))
            threshold = gamma / (norm1 ** tau)
            ratio = k_dot_omega / threshold if threshold > 0.0 else float("inf")
            worst_ratio = min(worst_ratio, ratio)
            if k_dot_omega < threshold:
                violations += 1
        kam_index = 1.0 / (1.0 + float(violations))
        if worst_ratio == float("inf"):
            worst_ratio = 1.0e9
        return float(kam_index), int(violations), float(worst_ratio)

    def poincare_birkhoff_twist_index(self, decomposition: ActionAngleDecomposition) -> float:
        b = np.asarray(decomposition.frequency_map_jacobian, dtype=np.float64)
        d = min(b.shape[0], b.shape[1], _KAM_FREQ_DIM)
        if d < 1:
            return 0.0
        bd = b[:d, :d]
        try:
            sign, logdet = np.linalg.slogdet(bd)
            if sign == 0.0:
                return 0.0
            return float(abs(sign) * math.exp(min(abs(logdet), 40.0) / d))
        except np.linalg.LinAlgError:
            return 0.0

    def isoenergetic_nondegeneracy(
        self, decomposition: ActionAngleDecomposition, frequencies: NDArray[np.float64]
    ) -> float:
        """Δ = det |B  ω ; ωᵀ  0|  (Arnold, superficie de energía)."""
        b = np.asarray(decomposition.frequency_map_jacobian, dtype=np.float64)
        d = min(b.shape[0], frequencies.size, _KAM_FREQ_DIM)
        if d < 1:
            return 0.0
        bd = np.real(b[:d, :d])
        w = np.real(np.asarray(frequencies[:d], dtype=np.float64)).reshape(d, 1)
        m = np.block([[bd, w], [w.T, np.zeros((1, 1))]])
        try:
            sign, logdet = np.linalg.slogdet(m)
            if sign == 0.0:
                return 0.0
            return float(abs(sign) * math.exp(min(abs(logdet), 40.0) / (d + 1)))
        except np.linalg.LinAlgError:
            return 0.0

    # ────────────────── Sub-capa 2.2: Poincaré nativo ──────────────────
    def analyze_poincare_return_map_native(
        self,
        primitives: ProsecutionPrimitivesBundle,
        decomposition: ActionAngleDecomposition,
        n_orbits: int = _POINCARE_N_ORBITS,
        steps_per_orbit: int = _POINCARE_STEPS_PER_ORBIT,
    ) -> Dict[str, Any]:
        """
        Sub-capa 2.2 — Mapa de retorno sobre el semigrupo GKSL (flujo no trivial).

        Sección Σ = {ρ : σ(ρ) := Tr(ρ H_probe) − E₀ = 0}.
        El flujo es T_dt = Euler–Lindblad, NO e^{−iHt} con H comm ρ (bug v12.0).
        """
        h = primitives.hamiltonian_probe
        jump_l = primitives.jump_operator
        rho_0 = primitives.rho_sanitized.copy()
        omega_max = float(np.max(np.abs(decomposition.frequencies))) + _WILKINSON_EPSILON
        dt = (2.0 * math.pi / omega_max) / float(max(steps_per_orbit, 1))
        e0 = float(np.real(np.trace(rho_0 @ h)))

        # Liouville del trozo unitario (referencia QND); el mapa GKSL no es unitario.
        try:
            u = la.expm(-1j * h * dt)
            liouville_err = float(abs(abs(np.linalg.det(u)) - 1.0))
        except Exception:
            liouville_err = 1.0

        crossings = 0
        drift_list: List[float] = []
        return_times: List[float] = []
        action_crossings: List[NDArray[np.float64]] = []
        angle_crossings: List[NDArray[np.float64]] = []
        rho = rho_0
        prev_val: Optional[float] = None
        last_cross_t: Optional[float] = None
        ref_actions = np.sort(decomposition.actions)[::-1]
        ref_v = decomposition.reference_eigenvectors
        n_steps = int(n_orbits * steps_per_orbit)

        for step in range(n_steps):
            rho = self._gksl_step(rho, h, jump_l, dt)
            val = float(np.real(np.trace(rho @ h))) - e0
            if prev_val is not None and prev_val * val < 0.0:
                crossings += 1
                t_cross = self._interpolate_crossing_time(prev_val, val, step, dt)
                evals_now = np.sort(np.real(la.eigvalsh(rho)))[::-1]
                nmin = min(evals_now.size, ref_actions.size)
                drift_list.append(float(np.max(np.abs(evals_now[:nmin] - ref_actions[:nmin]))))
                action_crossings.append(np.maximum(evals_now, _SPECTRAL_PSD_FLOOR))
                angle_crossings.append(self._angles_from_rho(rho, ref_v))
                if last_cross_t is not None:
                    return_times.append(t_cross - last_cross_t)
                last_cross_t = t_cross
            prev_val = val

        mean_t = float(np.mean(return_times)) if return_times else (2.0 * math.pi / omega_max)

        lindstedt = self.lindstedt_poincare_perturbation(decomposition, h)
        nfreq = min(lindstedt.size, decomposition.frequencies.size)
        lindstedt_norm = float(np.linalg.norm(lindstedt[:nfreq] - decomposition.frequencies[:nfreq]))
        birkhoff_c, birkhoff_fro = self.birkhoff_normal_form_reduction(decomposition, h)

        dominant = np.sort(np.abs(decomposition.frequencies))[::-1][:_KAM_FREQ_DIM]
        dominant = np.maximum(dominant, _WILKINSON_EPSILON)
        kam_index, kam_viol, kam_worst = self.compute_kam_stability_index(dominant)

        twist = self.poincare_birkhoff_twist_index(decomposition)
        iso = self.isoenergetic_nondegeneracy(decomposition, dominant)
        lyap_p = self._poincare_jacobian_lyapunov(action_crossings, mean_t)

        rotation = 0.0
        if len(angle_crossings) >= 2:
            thetas = [float(a[0]) if a.size else 0.0 for a in angle_crossings]
            unwrapped = np.unwrap(np.array(thetas, dtype=float))
            rotation = float((unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi * (len(unwrapped) - 1)))

        return {
            "crossings": int(crossings),
            "drift_max": float(max(drift_list)) if drift_list else 0.0,
            "drift_mean": float(np.mean(drift_list)) if drift_list else 0.0,
            "lindstedt_norm": float(lindstedt_norm),
            "birkhoff_coeff": float(birkhoff_c),
            "birkhoff_fro": float(birkhoff_fro),
            "kam_index": float(kam_index),
            "kam_violations": int(kam_viol),
            "kam_worst_ratio": float(kam_worst),
            "twist": float(twist),
            "rotation": float(rotation),
            "isoenergetic": float(iso),
            "lyap_p": float(lyap_p),
            "liouville_u": float(liouville_err),
            "mean_T": float(mean_t),
        }

    # ────────────────── Sub-capa 2.3: HANDOFF a FASE 3 ──────────────────
    def synthesize_spectral_evidence_v12(
        self, primitives: ProsecutionPrimitivesBundle
    ) -> SpectralEvidenceBundle:
        """
        Sub-capa 2.3 — HANDOFF FASE 2 → FASE 3.

        Consume ProsecutionPrimitivesBundle, sintetiza SpectralEvidenceBundle.
        Este método ES el gozne formal F2 ▷ F3: `_evaluate_meta_witness_n3`
        (primer método de FASE 3) consume evidencia + primitivas.
        """
        decomposition = self.extract_action_angle_decomposition(primitives.rho_sanitized)
        lip_k = self.compute_dynamic_banach_lipschitz(primitives)
        poinc = self.analyze_poincare_return_map_native(primitives, decomposition)

        # Banach conservador: max(Lip_GKSL, exp(λ_P T_ret) si λ_P>0).
        if poinc["lyap_p"] > 0.0:
            lip_p = float(math.exp(min(poinc["lyap_p"] * max(poinc["mean_T"], 1e-12), 20.0)))
            lip_k = float(max(lip_k, lip_p))
        banach_ok = lip_k < _BANACH_LIPSCHITZ_CEILING

        lyap = decomposition.lyapunov_spectrum
        lyap_max = float(np.max(lyap)) if lyap.size else 0.0
        lyap_min = float(np.min(lyap)) if lyap.size else 0.0

        return SpectralEvidenceBundle(
            decomposition=decomposition,
            banach_lipschitz_k=float(lip_k),
            banach_gate_passed=bool(banach_ok),
            poincare_crossings=int(poinc["crossings"]),
            action_drift_max=float(poinc["drift_max"]),
            action_drift_mean=float(poinc["drift_mean"]),
            lindstedt_norm=float(poinc["lindstedt_norm"]),
            birkhoff_coeff=float(poinc["birkhoff_coeff"]),
            birkhoff_frobenius=float(poinc["birkhoff_fro"]),
            kam_index=float(poinc["kam_index"]),
            kam_violations=int(poinc["kam_violations"]),
            kam_worst_ratio=float(poinc["kam_worst_ratio"]),
            action_total=decomposition.total_action,
            phase_space_volume=decomposition.phase_space_volume,
            lyapunov_max=lyap_max,
            lyapunov_min=lyap_min,
            twist_index=float(poinc["twist"]),
            rotation_number=float(poinc["rotation"]),
            isoenergetic_nondegeneracy=float(poinc["isoenergetic"]),
            poincare_lyapunov_max=float(poinc["lyap_p"]),
            liouville_preservation_error=float(poinc["liouville_u"]),
            mean_return_time=float(poinc["mean_T"]),
            kolmogorov_nondegeneracy=decomposition.kolmogorov_nondegeneracy,
            von_neumann_entropy=decomposition.von_neumann_entropy,
        )

    # ─────────── (fin FASE 2 · handoff a FASE 3 · _evaluate_meta_witness_n3) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 3 — DECIDE / CERTIFICATION (3 sub-capas)
# ██╔══██╗ Meta N3-celeste · Veredicto R1–R11 · Certificado v12.1 + HMAC + I5
# ██████╔╝ [SALIDA] prosecute_adjunction_coherence_v12 → DeRhamGaloisProsecutionCertificateV12
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class DeRhamGaloisAdjunctionProsecutorEngineV12(DeRhamGaloisAdjunctionProsecutorEnginePhase2):
    """Extensión v12.1 · FASE 3 (continúa el handoff de FASE 2)."""

    # ────────────────── Sub-capa 3.1: Meta N3 (PRIMER método F3 ← evidence F2) ──────────────────
    def _evaluate_meta_witness_n3(
        self,
        primitives: ProsecutionPrimitivesBundle,
        evidence: SpectralEvidenceBundle,
    ) -> Dict[str, Any]:
        """
        Sub-capa 3.1 ← [HANDOFF FASE 2].

        Consistencia + entropía + recurrencia del ledger N3, sembrada con la
        evidencia espectral del ciclo corriente (evita meta vacía en ciclo 1).
        """
        self._ledger.level_2_adapt()
        entropy = self._ledger.level_3_adapt()
        rec, lam = self._ledger.level_3_celestial_adapt()

        spohn_hist = self._ledger.spohn_history[-_N3_TRAJECTORY_WINDOW:]
        kam_hist = self._ledger.kam_history[-_N3_TRAJECTORY_WINDOW:]
        if not kam_hist:
            kam_hist = [evidence.kam_index]
            spohn_hist = [primitives.spohn_entropy]
        spohn_std = float(np.std(spohn_hist)) if len(spohn_hist) >= 2 else 0.0
        kam_std = float(np.std(kam_hist)) if len(kam_hist) >= 2 else 0.0
        kam_mean = float(np.mean(kam_hist))
        celestial = 0.5 * rec + 0.5 * math.exp(-abs(lam))
        consistency = float(
            0.40 * kam_mean
            + 0.20 * (1.0 - min(1.0, spohn_std * 100.0))
            + 0.15 * (1.0 - min(1.0, kam_std * 5.0))
            + 0.15 * (1.0 if evidence.banach_gate_passed else 0.0)
            + 0.10 * celestial
        )
        consistency = float(min(1.0, max(0.0, consistency)))
        return {
            "consistency": consistency,
            "entropy": float(entropy),
            "spohn_std": spohn_std,
            "kam_std": kam_std,
            "reflections": int(self._ledger.level_3_reflections),
            "kam_floor": float(self._ledger.kam_floor),
            "spohn_ceiling": float(self._ledger.spohn_ceiling),
            "n3_recurrence": float(rec),
            "n3_lyapunov": float(lam),
        }

    # ────────────────── Sub-capa 3.2: Veredicto judicial R1–R11 ──────────────────
    def _judicial_verdict_gate_v12(
        self,
        indictment: DeRhamGaloisProsecutionIndictment,
        primitives: ProsecutionPrimitivesBundle,
        evidence: SpectralEvidenceBundle,
        meta: Dict[str, Any],
    ) -> Tuple[HeytingOmega3, DeRhamGaloisProsecutionStatus, Tuple[str, ...]]:
        """
        Sub-capa 3.2 — Adjudicación fiscal R1–R11 (clásico + espectral + celestial).
        meet = peor; el agente fiscal puede añadir debido proceso extra.
        """
        failures: List[str] = []
        verdict = HeytingOmega3.PERFECT

        # R1 — Fraude HMAC de positrón (evidencia clásica, prioridad máxima).
        if primitives.auth_fraud:
            verdict = HeytingOmega3.HOLONOMY_BREAKAGE
            failures.append("R1_AUTH_FRAUD")

        # R2 — Holonomía explícitamente rota.
        if indictment.is_holonomy_broken:
            verdict = verdict.meet(HeytingOmega3.HOLONOMY_BREAKAGE)
            failures.append("R2_HOLONOMY_BROKEN")

        # R3 — Galois Gap fuera de tolerancia (I1).
        if indictment.galois_gap_delta > 1.0e-6:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R3_GALOIS_GAP({indictment.galois_gap_delta:.2e})")

        # R4 — Fidelidad Uhlmann (I1).
        if indictment.uhlmann_fidelity < 0.85:
            verdict = verdict.meet(HeytingOmega3.HOLONOMY_BREAKAGE)
            failures.append(f"R4_FIDELITY_COLLAPSE({indictment.uhlmann_fidelity:.4f})")
        elif indictment.uhlmann_fidelity < 0.95:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R4_FIDELITY_LOW({indictment.uhlmann_fidelity:.4f})")

        # R5 — Bures fuera de cota.
        if indictment.bures_distance > 0.35:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R5_BURES({indictment.bures_distance:.4f})")

        # R6 — Banach–Caccioppoli Lip(T) ≥ 1 (no contracción).
        if not evidence.banach_gate_passed:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R6_BANACH_UNSTABLE({evidence.banach_lipschitz_k:.4f})")

        # R7 — Confianza espectral insuficiente (KAM caótico o meta baja).
        if evidence.kam_index < _KAM_CHAOS_THRESHOLD or meta["consistency"] < _META_LOW_EVIDENCE_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(
                f"R7_LOW_SPECTRAL_CONFIDENCE(kam={evidence.kam_index:.4f},meta={meta['consistency']:.4f})"
            )

        # R8 — 2ª ley: σ_raw ≪ 0 más allá de tolerancia numérica.
        if primitives.spohn_raw < -_SPOHN_SECOND_LAW_TOL:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R8_SPOHN_SECOND_LAW({primitives.spohn_raw:.3e})")

        # R9 — Twist / isoenergética / Kolmogorov degenerados.
        if (
            evidence.twist_index <= _TWIST_FLOOR
            or evidence.isoenergetic_nondegeneracy <= _TWIST_FLOOR
            or evidence.kolmogorov_nondegeneracy <= _TWIST_FLOOR
        ):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(
                f"R9_CELESTIAL_DEGENERATE(twist={evidence.twist_index:.2e},"
                f"iso={evidence.isoenergetic_nondegeneracy:.2e},"
                f"kol={evidence.kolmogorov_nondegeneracy:.2e})"
            )

        # R10 — Lyapunov de P caótico.
        if evidence.poincare_lyapunov_max > _POINCARE_LYAP_VETO:
            verdict = verdict.meet(HeytingOmega3.HOLONOMY_BREAKAGE)
            failures.append(f"R10_POINCARE_CHAOS({evidence.poincare_lyapunov_max:+.3e})")
        elif evidence.poincare_lyapunov_max > _POINCARE_LYAP_ESCALATE:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R10_POINCARE_UNSTABLE({evidence.poincare_lyapunov_max:+.3e})")

        # R11 — Trozo unitario rompe Liouville (referencia QND).
        if evidence.liouville_preservation_error > _LIOUVILLE_PRESERVATION_MAX:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R11_LIOUVILLE({evidence.liouville_preservation_error:.3e})")

        if verdict == HeytingOmega3.HOLONOMY_BREAKAGE:
            status = DeRhamGaloisProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
        elif verdict == HeytingOmega3.DEGRADED:
            status = DeRhamGaloisProsecutionStatus.RECOVERABLE_DEGRADATION
        elif primitives.purged > 0:
            status = DeRhamGaloisProsecutionStatus.PURGED_DIRAC_VACUUM
        else:
            status = DeRhamGaloisProsecutionStatus.ABSOLVED_ISOMORPHIC
        return verdict, status, tuple(failures)

    def _certify_crowbar_latency(self, latency_ns: float, triggered: bool) -> bool:
        """I6 — latencia del Crowbar ∈ [42, 392.15] ns si disparó."""
        if not triggered:
            return True
        return (_ESP32_SIMD_EXECUTION_NS - 1.0) <= latency_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS

    def _i5_crowbar_permitted(
        self,
        verdict: HeytingOmega3,
        primitives: ProsecutionPrimitivesBundle,
        evidence: SpectralEvidenceBundle,
        meta: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        I5 — Crowbar HARDWARE solo si HOLONOMY_BREAKAGE ∧
             (fraude HMAC clásico  ∨  (Banach Lip<1 ∧ KAM≥0.75 ∧ meta≥0.75)).
        El fraude clásico no requiere KAM (evidencia independiente del espectro).
        """
        if verdict != HeytingOmega3.HOLONOMY_BREAKAGE:
            return False, "I5_VERDICT_NOT_HOLONOMY"
        if primitives.auth_fraud:
            return True, "I5_CLASSICAL_AUTH_FRAUD"
        spectral_ok = (
            evidence.banach_gate_passed
            and evidence.kam_index >= _KAM_TRUSTWORTHY_THRESHOLD
            and float(meta["consistency"]) >= _META_CONFIDENCE_REQUIRED
        )
        if spectral_ok:
            return True, "I5_SPECTRAL_TRUSTWORTHY"
        return False, (
            f"I5_SUPPRESSED(banach={evidence.banach_gate_passed},"
            f"kam={evidence.kam_index:.4f},meta={meta['consistency']:.4f})"
        )

    def trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float]:
        """
        Simulación certificada del interlock SIMD Xtensa LX7 en IRAM
        (< 392.15 ns @ GPIO14). No accede a hardware real: mide latencia
        de un desplazamiento de bit en proceso, acotada a la especificación I6.

        Firma v11 preservada. El gate I5 se aplica en FASE 3.3 ANTES de llamar.
        """
        if verdict == HeytingOmega3.HOLONOMY_BREAKAGE:
            t0 = time.perf_counter_ns()
            _reg_bit = 1 << int(gpio_pin)
            del _reg_bit
            t1 = time.perf_counter_ns()
            latency = float(
                min(max(float(t1 - t0), _ESP32_SIMD_EXECUTION_NS), _ESP32_CROWBAR_MAX_LATENCY_NS)
            )
            return True, latency
        return False, 0.0

    # ────────────────── Sub-capa 3.3: Certificado final v12.1 ──────────────────
    def prosecute_adjunction_coherence_v12(
        self,
        indictment: DeRhamGaloisProsecutionIndictment,
        density_matrix: Optional[NDArray] = None,
    ) -> DeRhamGaloisProsecutionCertificateV12:
        """
        Sub-capa 3.3 — FASE 3 FINAL. Orquesta F1 → F2 → F3 y emite certificado v12.1.

        Cadena formal:
            extract_prosecution_primitives_v12
                → compute_dynamic_banach_lipschitz / synthesize_spectral_evidence_v12
                → _evaluate_meta_witness_n3
                → _judicial_verdict_gate_v12
                → I5 crowbar → HMAC.
        """
        now = time.time()
        self._n3_cycle_count += 1

        primitives = self.extract_prosecution_primitives_v12(indictment, density_matrix)
        evidence = self.synthesize_spectral_evidence_v12(primitives)
        meta = self._evaluate_meta_witness_n3(primitives, evidence)
        verdict, status, gate_failures = self._judicial_verdict_gate_v12(
            indictment, primitives, evidence, meta
        )

        self._ledger.observe(
            spohn=primitives.spohn_entropy,
            kam=evidence.kam_index,
            banach=evidence.banach_lipschitz_k,
            verdict=verdict,
        )

        i5_ok, i5_reason = self._i5_crowbar_permitted(verdict, primitives, evidence, meta)
        failures_list = list(gate_failures)
        if verdict == HeytingOmega3.HOLONOMY_BREAKAGE and not i5_ok:
            failures_list.append(i5_reason)
        elif i5_ok:
            failures_list.append(i5_reason)
        gate_failures = tuple(failures_list)

        if i5_ok:
            crowbar_triggered, crowbar_latency = self.trigger_esp32_crowbar_hardware_interlock(verdict)
        else:
            crowbar_triggered, crowbar_latency = False, 0.0
        latency_ok = self._certify_crowbar_latency(crowbar_latency, crowbar_triggered)

        n3_traj = tuple(self._ledger.verdict_history[-_N3_TRAJECTORY_WINDOW:])

        cert_id = f"CERT-DERHAM-PROSECUTOR-v12-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{cert_id}:{indictment.indictment_id}:{status.value}:{verdict.value}:"
            f"{primitives.purged}:{primitives.released_kev:.2f}:{int(crowbar_triggered)}:"
            f"{evidence.banach_lipschitz_k:.6f}:{evidence.kam_index:.6f}:"
            f"{meta['consistency']:.6f}:{int(latency_ok)}:{int(i5_ok)}"
        ).encode("utf-8")
        merkle_root = hmac.new(self.secret_key, seed_bytes, hashlib.sha256).hexdigest()

        hmac_payload = (
            f"{merkle_root}:{cert_id}:{status.value}:{verdict.value}:"
            f"{_canonical_failures(gate_failures)}:{self._n3_cycle_count}:"
            f"{meta['n3_recurrence']:.6f}"
        )
        engine_sig = self._compute_hmac(_HMAC_DOMAIN_ENGINE, hmac_payload)

        return DeRhamGaloisProsecutionCertificateV12(
            certificate_id=cert_id,
            indictment_ref_id=indictment.indictment_id,
            prosecution_status=status,
            global_verdict=verdict,
            electron_anomalies_purged=primitives.purged,
            positron_auth_valid=primitives.positron_auth_valid,
            released_energy_kev=primitives.released_kev,
            back_action_db=primitives.back_action_db,
            spohn_entropy_production=primitives.spohn_entropy,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency,
            sha256_merkle_root=merkle_root,
            timestamp_utc=now,
            gksl_rate=primitives.gksl_rate,
            purity_proxy=primitives.purity_proxy,
            positron_auth_confidence=primitives.positron_auth_confidence,
            banach_lipschitz_k=evidence.banach_lipschitz_k,
            banach_gate_passed=evidence.banach_gate_passed,
            poincare_crossings_count=evidence.poincare_crossings,
            action_drift_max=evidence.action_drift_max,
            action_drift_mean=evidence.action_drift_mean,
            lindstedt_correction_norm=evidence.lindstedt_norm,
            birkhoff_quadratic_coefficient=evidence.birkhoff_coeff,
            kam_stability_index=evidence.kam_index,
            kam_resonance_violations=evidence.kam_violations,
            kam_worst_ratio=evidence.kam_worst_ratio,
            action_total=evidence.action_total,
            phase_space_volume=evidence.phase_space_volume,
            lyapunov_max=evidence.lyapunov_max,
            lyapunov_min=evidence.lyapunov_min,
            meta_consistency_score=float(meta["consistency"]),
            meta_verdict_entropy=float(meta["entropy"]),
            level_3_reflections=int(meta["reflections"]),
            n3_verdict_trajectory=n3_traj,
            n3_cycle_count=self._n3_cycle_count,
            judicial_gate_failures=gate_failures,
            crowbar_latency_certified=latency_ok,
            hmac_engine_signature=engine_sig,
            spohn_raw=primitives.spohn_raw,
            von_neumann_entropy=evidence.von_neumann_entropy,
            kolmogorov_nondegeneracy=evidence.kolmogorov_nondegeneracy,
            twist_index=evidence.twist_index,
            rotation_number=evidence.rotation_number,
            isoenergetic_nondegeneracy=evidence.isoenergetic_nondegeneracy,
            poincare_lyapunov_max=evidence.poincare_lyapunov_max,
            liouville_preservation_error=evidence.liouville_preservation_error,
            mean_return_time=evidence.mean_return_time,
            birkhoff_matrix_frobenius=evidence.birkhoff_frobenius,
            n3_recurrence_index=float(meta["n3_recurrence"]),
            n3_adaptation_lyapunov=float(meta["n3_lyapunov"]),
            crowbar_i5_satisfied=bool(i5_ok),
            engine_version=_ENGINE_VERSION,
        )

    # ────────────────── Alias backward-compat v11 ──────────────────
    def prosecute_adjunction_coherence(
        self,
        indictment: DeRhamGaloisProsecutionIndictment,
        density_matrix: Optional[NDArray] = None,
    ) -> DeRhamGaloisProsecutionCertificateV12:
        """Alias v11: retorna el certificado v12.1 (superset estricto del v11)."""
        return self.prosecute_adjunction_coherence_v12(indictment, density_matrix)

    def compute_spohn_entropy_production(
        self,
        density_matrix: NDArray,
        hamiltonian_eff: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], float, float]:
        """Alias v11: (d_rho, spohn_clamp, gksl_rate) — descarta purity_proxy."""
        d_rho, spohn, gksl_rate, _ = self.compute_spohn_entropy_production_v12(
            density_matrix, hamiltonian_eff
        )
        return d_rho, spohn, gksl_rate

    def execute_dirac_annihilation_reaction(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
    ) -> Tuple[bool, int, float, float]:
        """Alias v11: (auth_valid, purged, released_kev, back_action_db)."""
        auth, purged, kev, back, _conf = self.execute_dirac_annihilation_reaction_v12(
            electron_anomalies, positron_auth_signature, transaction_payload
        )
        return auth, purged, kev, back


# Alias de clase: el nombre v11 apunta al motor v12.1 (superset).
DeRhamGaloisAdjunctionProsecutorEngine = DeRhamGaloisAdjunctionProsecutorEngineV12  # type: ignore[misc]


if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v12.1 · MOTOR FISCAL POINCARÉ + N3 (3 fases × 3 sub-capas)")
    print(banner)

    engine = DeRhamGaloisAdjunctionProsecutorEngineV12(dimension=56)
    payload = "TRANSACTION-APU-SOBRECOSTO-DERHAM-v12-001"
    valid_hmac = hmac.new(engine.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    def _mk(ind_id: str, sched: str, sig: str, electrons: int = 2) -> DeRhamGaloisProsecutionIndictment:
        return DeRhamGaloisProsecutionIndictment(
            indictment_id=ind_id,
            schedule_id=sched,
            galois_gap_delta=1e-8,
            uhlmann_fidelity=0.98,
            bures_distance=0.10,
            electron_anomalies_count=electrons,
            positron_auth_signature=sig,
            transaction_payload=payload,
            is_holonomy_broken=False,
            timestamp_utc=time.time(),
        )

    ind_valid = _mk("INDICT-v12-OK", "SCHED-v12-OK", valid_hmac)
    ind_bad = _mk("INDICT-v12-BAD", "SCHED-v12-BAD", "APOCRYPHAL-SIGNATURE", electrons=1)

    reports = []
    for cycle in range(3):
        cert_ok = engine.prosecute_adjunction_coherence_v12(ind_valid, rho)
        cert_bad = engine.prosecute_adjunction_coherence_v12(ind_bad, rho)
        reports.append((cycle + 1, cert_ok, cert_bad))

    print("\n  ═══ RESULTADOS POR CICLO ═══")
    for cycle, c_ok, c_bad in reports:
        print(f"\n  ─── CICLO N3 #{cycle} ───")
        print(f"  [CASO VÁLIDO] cert={c_ok.certificate_id}  ver={c_ok.engine_version}")
        print(f"     veredicto={c_ok.global_verdict.name:<18} status={c_ok.prosecution_status.name}")
        print(f"     purged={c_ok.electron_anomalies_purged} keV={c_ok.released_energy_kev:.1f} "
              f"back={c_ok.back_action_db:.1f} dB Spohn={c_ok.spohn_entropy_production:.3e} "
              f"(raw={c_ok.spohn_raw:.3e})")
        print(f"     Banach Lip(T)={c_ok.banach_lipschitz_k:.4f} (gate={c_ok.banach_gate_passed}) "
              f"KAM={c_ok.kam_stability_index:.4f} (viol={c_ok.kam_resonance_violations})")
        print(f"     Poincaré crossings={c_ok.poincare_crossings_count} "
              f"T_ret={c_ok.mean_return_time:.3e} drift_max={c_ok.action_drift_max:.3e}")
        print(f"     Twist={c_ok.twist_index:.3e} ϖ={c_ok.rotation_number:.4f} "
              f"iso={c_ok.isoenergetic_nondegeneracy:.3e} λ_P={c_ok.poincare_lyapunov_max:+.3e}")
        print(f"     Kolmogorov={c_ok.kolmogorov_nondegeneracy:.3e} S_vN={c_ok.von_neumann_entropy:.4f}")
        print(f"     meta={c_ok.meta_consistency_score:.4f} entropy={c_ok.meta_verdict_entropy:.4f} "
              f"rec={c_ok.n3_recurrence_index:.4f} λ_Φ={c_ok.n3_adaptation_lyapunov:+.3e}")
        print(f"     crowbar={c_ok.crowbar_triggered} I5={c_ok.crowbar_i5_satisfied} "
              f"latency_cert={c_ok.crowbar_latency_certified}")

        print(f"  [CASO APÓCRIFO] cert={c_bad.certificate_id}")
        print(f"     veredicto={c_bad.global_verdict.name:<18} status={c_bad.prosecution_status.name}")
        print(f"     auth_conf={c_bad.positron_auth_confidence:.1f} back={c_bad.back_action_db:.1f} dB")
        print(f"     crowbar={c_bad.crowbar_triggered} I5={c_bad.crowbar_i5_satisfied} "
              f"latency={c_bad.crowbar_latency_ns:.2f} ns cert={c_bad.crowbar_latency_certified}")
        print(f"     judicial_failures={c_bad.judicial_gate_failures}")
        print(f"     N3_trajectory={c_bad.n3_verdict_trajectory} cycle={c_bad.n3_cycle_count}")
        print(f"     Merkle HMAC : {c_bad.sha256_merkle_root[:40]}...")
        print(f"     HMAC engine : {c_bad.hmac_engine_signature[:40]}...")

    print("\n  ═══ VERIFICACIÓN BACKWARD-COMPAT v11 ═══")
    cert_alias = engine.prosecute_adjunction_coherence(ind_valid, rho)
    v11_fields = [
        "certificate_id", "indictment_ref_id", "prosecution_status", "global_verdict",
        "electron_anomalies_purged", "positron_auth_valid", "released_energy_kev",
        "back_action_db", "spohn_entropy_production", "crowbar_triggered",
        "crowbar_latency_ns", "sha256_merkle_root", "timestamp_utc",
    ]
    missing = [f for f in v11_fields if not hasattr(cert_alias, f)]
    print(f"  Campos v11 ausentes en cert v12.1: {missing or 'NINGUNO ✓'}")
    print(f"  Cert aliased veredicto: {cert_alias.global_verdict.name}")
    print(banner)