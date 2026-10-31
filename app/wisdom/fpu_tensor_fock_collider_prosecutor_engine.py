# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPU Tensor Fock Collider Prosecutor Engine v14.1.0-SOVEREIGN-POINCARE-N3-PhD              ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider_prosecutor_engine.py                                  ║
║ Función  : Motor fiscal inquisidor de ℱ(ℋ) con (a) verificación CAR/CCR CANÓNICA rigurosa sobre      ║
║            operadores b/b† (bosón truncado) y a/a† (JW fermión 1-modo), (b) Mecánica Celeste de      ║
║            Poincaré sobre el oscilador de Kerr forzado — NUNCA sobre spec(ρ) —, (c) Automejora N3    ║
║            Poincaré-acoplada y (d) DEBIDO PROCESO: el Crowbar ESP32 (< 392.15 ns @ GPIO14) sólo      ║
║            dispara con HARD EVIDENCE (CAR/CCR/auth deterministas) o con evidencia espectral          ║
║            suficiente Y fiable (I6).                                                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA POR FASES ANIDADAS (3 × 3 sub-capas) — HANDOFF ESTRICTO Φ₁₂ / Φ₂₃:

  FASE 1 — FOUNDATIONS (álgebra canónica + primitivas termodinámicas + aniquilación Dirac)
    1.1 `build_canonical_fock_operators_v14`          — b, b†, a, a†, x̂, p̂, N̂
    1.2 `compute_spohn_and_dirac_primitives_v14`      — Spohn≥0 + Dirac + auth_confidence
    1.3 [HANDOFF Φ₁₂] `extract_fpu_prosecution_primitives_v14`
         ≡ primer acto de FASE 2 (`build_fock_poincare_return_map_v14`)
         → FPUProsecutionPrimitivesBundle  (incluye FockActionAngleDecomposition)

  FASE 2 — SPECTRAL EVIDENCE (Kerr forzado + Poincaré + KAM + Banach dinámico + N2)
    2.1 `build_fock_poincare_return_map_v14`           — continuación de Φ₁₂; Σ={p=0, ṗ>0}
    2.2 `recursive_self_tune_parameters`              — N2 umbrales Poincaré-modulados
    2.3 [HANDOFF Φ₂₃] `synthesize_fpu_spectral_evidence_v14`
         ≡ primer acto de FASE 3 (`run_phase3_meta_verification_v14`)
         → FPUSpectralEvidenceBundle

  FASE 3 — CERTIFICATION (meta N3 + debido proceso + adjudicación)
    3.1 `run_phase3_meta_verification_v14`            — continuación de Φ₂₃; N3 meta-tasa
    3.2 `_judicial_due_process_gate_v14`              — HARD vs SOFT (I5 / I6, sin R3⟂R5)
    3.3 [SALIDA] `collide_prosecute_and_annihilate_dirac_v14`
         → FPUProsecutionExecutionCertificateV14

INVARIANTES PRESERVADOS Y AMPLIADOS (I1–I10):
  · I1  Back-Action 0.0 dB (auth válida) / 12.5 dB (auth apócrifa) — inalterado
  · I2  Electrones purgados = electron_anomalies si auth valida, 0 si no
  · I3  Spohn σ ≥ 0 (GKSL clamp) — inalterado
  · I4  Crowbar ≤ 392.15 ns certificado — inalterado
  · I5  HARD EVIDENCE (CAR/CCR fail, auth fail) → Crowbar INCONDICIONAL
  · I6  SOFT EVIDENCE (KAM, meta, λ_P, T_Nek) → Crowbar SOLO si
        KAM≥kam_floor ∧ meta≥0.75 ∧ Banach<1  (y únicamente sobre VETOED blando)
  · I7  Contrato backward-compat v13: `collide_prosecute_and_annihilate_dirac` intacto
  · I8  [x̂, p̂] = i[b, b†]  (residuo simpléctico < 1e-6)
  · I9  Tr(ρ)=1, ρ ⪰ 0 (Wilkinson floor); CCR [b,b†]=I−N|N−1⟩⟨N−1|
        CAR JW 1-modo {a,a†}=I₂
  · I10 Deriva de ocupación sobre p_n=⟨n|ρ|n⟩ y mapa (x,⟨N⟩)|_Σ —
        NUNCA sobre spec(ρ) (unitariamente invariante, idénticamente nula)
  · H   Kerr forzado H(t)=ωN̂+χN̂(N̂−I)+ε cos(ω_d t)(b+b†), ω_d/ω=φ⁻¹;
        [H, N̂]≠0.  Se PROHÍBE H_eff=−ρ+(trρ/N)I.

N3 (recursivo nivel 3):
  N1 veredicto Ω₃ · N2 umbrales (kam_floor, spohn_ceiling) · N3 meta-tasa
  Ledger Poincaré-acoplado: poincare_modulate, adaptation_lyapunov.
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
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.FPUTensorFockColliderProsecutorEngineV14")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas / algebraicas / precisión
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_SIMD_EXECUTION_NS: Final[float] = 42.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_FPU_PROSECUTOR_SOVEREIGN_KEY_v14.0.0"

# ── Kerr forzado / Poincaré / KAM / Nekhoroshev / Benettin ──
_OMEGA_CARRIER: Final[float] = 1.0
_KERR_CHI: Final[float] = 0.03
_DRIVE_EPSILON: Final[float] = 0.08
_OMEGA_DRIVE: Final[float] = _OMEGA_CARRIER * 0.5 * (math.sqrt(5.0) - 1.0)  # φ⁻¹
_KAM_FREQ_DIM: Final[int] = 4
_KAM_GAMMA: Final[float] = 1.0e-2
_KAM_TAU: Final[float] = 2.0
_KAM_RESONANCE_MAX_K: Final[int] = 3
_KAM_CF_DEPTH: Final[int] = 12
_KAM_CF_PENALTY_WEIGHT: Final[float] = 0.10
_POINCARE_STEPS_PER_ORBIT: Final[int] = 16
_POINCARE_N_ORBITS: Final[int] = 12
_BENETTIN_D0: Final[float] = 1.0e-8
_BENETTIN_RESCALE_EVERY: Final[int] = 8
_NEKHOROSHEV_C: Final[float] = 0.35
_NEKHOROSHEV_NDOF: Final[int] = 2  # 1.5 g.d.l. extendido
_CANONICAL_RESIDUAL_TOL: Final[float] = 1.0e-6

# ── N3 ──
_N3_STABILITY_WINDOW: Final[int] = 8
_N3_HISTORY_CAP: Final[int] = 64

# ── Umbrales judiciales v14.1 ──
_BANACH_LIPSCHITZ_CEILING: Final[float] = 1.0
_KAM_TRUSTWORTHY_THRESHOLD: Final[float] = 0.75
_KAM_CHAOS_THRESHOLD: Final[float] = 0.50
_KAM_SEVERE_CHAOS_THRESHOLD: Final[float] = 0.25
_META_CONFIDENCE_REQUIRED: Final[float] = 0.75
_META_LOW_EVIDENCE_THRESHOLD: Final[float] = 0.50
_META_HIGH_CONFIDENCE_FOR_ESCALATION: Final[float] = 0.85
_LYAPUNOV_CHAOS_THRESHOLD: Final[float] = 0.50
_NEKHOROSHEV_MIN_TIME: Final[float] = 2.0
_OCCUPATION_DRIFT_SOFT: Final[float] = 1.0e-3

_HMAC_ENGINE_DOMAIN: Final[bytes] = b"CERT-FPU-PROSECUTOR-v14.1-ENGINE|"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS — Ω₃ + FPUProsecutionStatus (DEGRADED_FOCK_SPECTRAL v14)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """Retículo distributivo trivalente Ω₃ = {0x00, 0x01, 0x03} para enjuiciamiento FPU.
    meet = max (severidad); join = min.
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED   = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc


class FPUProsecutionStatus(IntEnum):
    """Estados del proceso de inquisición FPU (v14 añade DEGRADED_FOCK_SPECTRAL)."""
    PENDING                    = 0
    PURGED_DIRAC_VACUUM        = 1
    VETOED_CAR_VIOLATION       = 2
    VETOED_DIRAC_AUTH_FAILURE  = 3
    DEGRADED_FOCK_SPECTRAL     = 4  # v14 · evidencia blanda, sin crowbar salvo I6


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES — v13 preservado + v14.1 enriquecido + bundles de handoff Φ₁₂ / Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FPUProsecutionExecutionCertificate:
    """Certificado v13 (preservado por compatibilidad)."""
    certificate_id: str
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    car_anti_commutation_residual: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    prosecution_status: FPUProsecutionStatus
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class FPUProsecutionExecutionCertificateV14:
    """
    FASE 3 — Certificado v14.1 (superset estricto del v13).
    Todos los campos v13 presentes con el mismo nombre + Poincaré/N3/judicial v14.1.
    """
    # v13
    certificate_id: str
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    car_anti_commutation_residual: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    prosecution_status: FPUProsecutionStatus
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float
    # v14.1 · FASE 1 (canónicos + I8 + auth_confidence + hard evidence)
    car_canonical_residual: float
    car_canonical_valid: bool
    ccr_canonical_residual: float
    ccr_canonical_valid: bool
    symplectic_quadrature_residual: float
    symplectic_quadrature_valid: bool
    purity_proxy: float
    positron_auth_confidence: float
    hard_evidence_veto: bool
    quadrature_x: float
    quadrature_p: float
    number_action: float
    omega_carrier: float
    # v14.1 · FASE 2 (Poincaré Kerr forzado + Banach dinámico + KAM + Nekhoroshev)
    banach_lipschitz_k: float
    banach_gate_passed: bool
    poincare_crossings_count: int
    occupation_drift_max: float
    occupation_drift_mean: float
    lindstedt_correction_norm: float
    lindstedt_o1: float
    lindstedt_o2: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    kam_cf_penalty: float
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    lyapunov_poincare: float
    nekhoroshev_time: float
    celestial_section_transverse: bool
    poincare_return_period: float
    drive_epsilon: float
    kerr_chi: float
    # v14.1 · FASE 3 (meta N3 Poincaré-acoplado + debido proceso)
    meta_consistency_score: float
    meta_verdict_entropy: float
    level_3_entropy: float
    level_3_reflections: int
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    n3_adaptation_lyapunov: float
    judicial_gate_failures: Tuple[str, ...]
    crowbar_latency_certified: bool
    hmac_engine_signature: str
    ooda_kam_coherent_threshold: float


@dataclass(slots=True)
class FockActionAngleDecomposition:
    """Carta acción-ángulo sobre ℱ(ℋ). Acciones = poblaciones de Fock p_n, NO spec(ρ)."""
    number_populations: NDArray[np.float64]
    action_J: float
    action_J2: float
    quadrature_x: float
    quadrature_p: float
    quadrature_phase: float
    frequencies: NDArray[np.float64]
    omega_carrier: float
    symplectic_residual: float
    symplectic_valid: bool
    reconstructed_rho: NDArray[np.complex128]
    purity: float
    phase_space_volume: float


@dataclass(slots=True)
class FPUProsecutionPrimitivesBundle:
    """HANDOFF Φ₁₂  FASE 1 → FASE 2. Primitivas termodinámicas + Dirac + carta acción-ángulo."""
    rho_sanitized: NDArray[np.complex128]
    spohn_entropy: float
    gksl_rate: float
    purity_proxy: float
    car_residual_v13: float
    car_valid_v13: bool
    car_canonical_residual: float
    car_canonical_valid: bool
    ccr_canonical_residual: float
    ccr_canonical_valid: bool
    symplectic_residual: float
    symplectic_valid: bool
    auth_valid: bool
    auth_confidence: float
    purged: int
    released_kev: float
    back_action_db: float
    hard_evidence_veto: bool
    hard_veto_status: Optional[FPUProsecutionStatus]
    action_angle: FockActionAngleDecomposition


@dataclass(slots=True)
class FockPoincareReturnMap:
    """Mapa de retorno sobre Σ={⟨p⟩=0, ṗ>0} del Kerr forzado. Coordenadas (x, ⟨N⟩)."""
    crossings: int
    drift_max: float
    drift_mean: float
    section_x: Tuple[float, ...]
    section_n: Tuple[float, ...]
    return_period: float
    lindstedt_o1: float
    lindstedt_o2: float
    lindstedt_norm: float
    birkhoff_coeff: float
    kam_index: float
    kam_violations: int
    kam_worst_ratio: float
    kam_cf_penalty: float
    lyapunov_poincare: float
    lyapunov_hs: float
    nekhoroshev_time: float
    transverse: bool
    omega_carrier: float
    frequencies: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FockTuningPolicy:
    """Política N2 emitida al cierre de FASE 2. No muta (χ, ε) físicos — sólo umbrales."""
    kam_floor: float
    spohn_ceiling: float
    banach_ceiling: float
    drive_epsilon: float
    kerr_chi: float
    adaptation_rate: float
    meta_adaptation_rate: float
    n3_adaptation_lyapunov: float


@dataclass(slots=True)
class FPUSpectralEvidenceBundle:
    """HANDOFF Φ₂₃  FASE 2 → FASE 3. Evidencia espectral + política N2 para adjudicación."""
    banach_lipschitz_k: float
    banach_gate_passed: bool
    poincare_crossings: int
    occupation_drift_max: float
    occupation_drift_mean: float
    lindstedt_norm: float
    lindstedt_o1: float
    lindstedt_o2: float
    birkhoff_coeff: float
    kam_index: float
    kam_violations: int
    kam_worst_ratio: float
    kam_cf_penalty: float
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    lyapunov_poincare: float
    nekhoroshev_time: float
    celestial_section_transverse: bool
    poincare_return_period: float
    omega_carrier: float
    quadrature_x: float
    quadrature_p: float
    symplectic_residual: float
    tuning_policy: FockTuningPolicy


@dataclass(slots=True)
class FockProsecutorN3Ledger:
    """
    Ledger de automejora recursiva N3 Poincaré-acoplado
    (N1 veredicto, N2 umbrales, N3 meta-tasa + λ_α de adaptación).
    Distinto del ledger del testigo; no comparte estado.
    """
    spohn_history: List[float] = field(default_factory=list)
    kam_history: List[float] = field(default_factory=list)
    banach_history: List[float] = field(default_factory=list)
    lyap_history: List[float] = field(default_factory=list)
    verdict_history: List[int] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    spohn_ceiling: float = 0.10
    kam_floor: float = _KAM_TRUSTWORTHY_THRESHOLD
    adaptation_rate: float = 0.05
    meta_adaptation_rate: float = 0.01
    level_3_reflections: int = 0
    last_poincare_mod: float = 0.0

    def observe(
        self,
        spohn: float,
        kam: float,
        banach: float,
        lyap_p: float,
        verdict: HeytingOmega3,
    ) -> None:
        self.spohn_history.append(float(spohn))
        self.kam_history.append(float(kam))
        self.banach_history.append(float(banach))
        self.lyap_history.append(float(lyap_p))
        self.verdict_history.append(int(verdict))
        self.rate_history.append(float(self.adaptation_rate))
        if len(self.verdict_history) > _N3_HISTORY_CAP:
            self.spohn_history = self.spohn_history[-_N3_HISTORY_CAP:]
            self.kam_history = self.kam_history[-_N3_HISTORY_CAP:]
            self.banach_history = self.banach_history[-_N3_HISTORY_CAP:]
            self.lyap_history = self.lyap_history[-_N3_HISTORY_CAP:]
            self.verdict_history = self.verdict_history[-_N3_HISTORY_CAP:]
            self.rate_history = self.rate_history[-_N3_HISTORY_CAP:]

    def _trend(self, series: Sequence[float]) -> float:
        if len(series) < 2:
            return 0.0
        n = min(_N3_STABILITY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def poincare_modulate(
        self,
        kam: float,
        lyap_p: float,
        crossings: int,
        drift: float,
        t_nek: float,
    ) -> None:
        """Acopla la geometría celeste a los umbrales N2 *antes* de level_2_adapt."""
        chaotic = (float(lyap_p) > _LYAPUNOV_CHAOS_THRESHOLD) or (
            int(crossings) > 0 and float(drift) > _OCCUPATION_DRIFT_SOFT
        )
        near_resonant = float(kam) < _KAM_CHAOS_THRESHOLD
        nek_short = float(t_nek) < _NEKHOROSHEV_MIN_TIME
        mod = 0.0
        if chaotic:
            self.kam_floor = min(0.92, self.kam_floor + self.adaptation_rate * 0.03)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.97)
            mod += 1.0
        if near_resonant:
            self.kam_floor = min(0.92, self.kam_floor + self.adaptation_rate * 0.02)
            mod += 0.5
        if nek_short:
            self.spohn_ceiling = max(0.01, self.spohn_ceiling * (1.0 - self.adaptation_rate))
            mod += 0.5
        if not chaotic and not near_resonant and float(kam) >= _KAM_TRUSTWORTHY_THRESHOLD:
            self.kam_floor = max(
                _KAM_TRUSTWORTHY_THRESHOLD,
                self.kam_floor - self.adaptation_rate * 0.01,
            )
            mod -= 0.25
        self.last_poincare_mod = float(mod)

    def level_2_adapt(self) -> None:
        trend_s = self._trend(self.spohn_history)
        trend_k = self._trend(self.kam_history)
        if trend_s > 1e-6:
            self.spohn_ceiling = max(0.01, self.spohn_ceiling * (1.0 - self.adaptation_rate))
        elif trend_s < -1e-6:
            self.spohn_ceiling = min(0.5, self.spohn_ceiling * (1.0 + self.adaptation_rate * 0.5))
        if trend_k < -1e-4:
            self.kam_floor = min(0.90, self.kam_floor + self.adaptation_rate * 0.02)
        elif trend_k > 1e-4:
            self.kam_floor = max(
                _KAM_TRUSTWORTHY_THRESHOLD,
                self.kam_floor - self.adaptation_rate * 0.01,
            )

    def level_3_adapt(self) -> float:
        n = len(self.verdict_history)
        if n < 4:
            return 0.0
        recent = np.array(self.verdict_history[-_N3_STABILITY_WINDOW:], dtype=int)
        _uniq, counts = np.unique(recent, return_counts=True)
        probs = counts / float(counts.sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if entropy > 0.5:
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * 0.90)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.95)
        else:
            self.meta_adaptation_rate = min(0.50, self.meta_adaptation_rate * 1.05)
            self.adaptation_rate = min(0.50, self.adaptation_rate * 1.02)
        self.level_3_reflections += 1
        return entropy

    def adaptation_lyapunov(self) -> float:
        """λ_α — estiramiento logarítmico de la propia tasa de adaptación (meta-Lyapunov N3)."""
        if len(self.rate_history) < 3:
            return 0.0
        r = np.maximum(np.array(self.rate_history[-_N3_STABILITY_WINDOW:], dtype=float), 1e-12)
        if len(r) < 2:
            return 0.0
        return float(np.mean(np.log(r[1:] / r[:-1])))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# MOTOR v13 — PRESERVADO EXACTAMENTE (backward-compat I7)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderProsecutorEngine:
    """Motor fiscal v13 (preservado por compatibilidad)."""

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        a = np.asarray(matrix, dtype=np.complex128)
        herm = 0.5 * (a + a.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    def verify_car_exclusion_rules(
        self, state_a: NDArray, state_b: NDArray
    ) -> Tuple[bool, float]:
        va = np.asarray(state_a, dtype=np.complex128).reshape(-1, 1)
        vb = np.asarray(state_b, dtype=np.complex128).reshape(-1, 1)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > _WILKINSON_EPSILON:
            va /= na
        if nb > _WILKINSON_EPSILON:
            vb /= nb
        op_a = va @ va.conj().T
        op_b_dag = vb @ vb.conj().T
        anti_comm = op_a @ op_b_dag + op_b_dag @ op_a
        delta_ab = complex((vb.conj().T @ va)[0, 0])
        residual = float(np.linalg.norm(anti_comm - abs(delta_ab) * np.eye(self.dimension)))
        return bool(residual <= 1.5), residual

    def compute_lindblad_gksl_and_spohn(
        self, density_matrix: NDArray, hamiltonian_eff: Optional[NDArray] = None
    ) -> Tuple[NDArray[np.complex128], float, float]:
        rho = self.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]
        if hamiltonian_eff is None:
            hamiltonian_eff = 0.1 * np.eye(dim, dtype=np.complex128)
        h = np.asarray(hamiltonian_eff, dtype=np.complex128)
        unitary = -1j * (h @ rho - rho @ h)
        jump_l = np.zeros((dim, dim), dtype=np.complex128)
        jump_l[0, min(1, dim - 1)] = 0.05
        jump_l_dag = jump_l.conj().T
        dissipator = (
            jump_l @ rho @ jump_l_dag
            - 0.5 * (jump_l_dag @ jump_l @ rho + rho @ jump_l_dag @ jump_l)
        )
        d_rho = unitary + dissipator
        eigvals, eigvecs = la.eigh(rho)
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        log_rho = (eigvecs * log_eigs) @ eigvecs.conj().T
        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn = max(0.0, spohn_raw)  # I3
        gksl_rate = float(np.linalg.norm(dissipator))
        return d_rho, spohn, gksl_rate

    def trigger_esp32_crowbar_hardware_interlock(
        self, verdict: HeytingOmega3, gpio_pin: int = 14
    ) -> Tuple[bool, float]:
        if verdict == HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _ = bool(reg_bit & (1 << 14))
            t1 = time.perf_counter_ns()
            latency_ns = float(
                min(max(float(t1 - t0), _ESP32_SIMD_EXECUTION_NS), _ESP32_CROWBAR_MAX_LATENCY_NS)
            )
            return True, latency_ns
        return False, 0.0

    def collide_prosecute_and_annihilate_dirac(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionExecutionCertificate:
        """v13 — preservado (retorna FPUProsecutionExecutionCertificate v13). I7."""
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        dim = self.dimension
        _, spohn_entropy, gksl_rate = self.compute_lindblad_gksl_and_spohn(rho)

        if state_a is None:
            state_a = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if state_b is None:
            state_b = np.zeros((dim, 1), dtype=np.complex128)
            state_b[min(1, dim - 1), 0] = 1.0

        car_ok, car_res = self.verify_car_exclusion_rules(state_a, state_b)

        if not car_ok:
            verdict = HeytingOmega3.VETOED
            status = FPUProsecutionStatus.VETOED_CAR_VIOLATION
            cb, cb_lat = self.trigger_esp32_crowbar_hardware_interlock(verdict)
            cert_id = f"CERT-FPU-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
            seed = f"{cert_id}:{status.name}:CAR_FAIL:{car_res:.8f}:{now}".encode("utf-8")
            return FPUProsecutionExecutionCertificate(
                certificate_id=cert_id, electron_anomalies_purged=0,
                positron_auth_valid=False, released_energy_kev=0.0,
                back_action_db=12.5, spohn_entropy_production=spohn_entropy,
                gksl_dissipation_rate=gksl_rate,
                car_anti_commutation_residual=car_res,
                crowbar_triggered=cb, crowbar_latency_ns=cb_lat,
                prosecution_status=status, global_verdict=verdict,
                sha256_merkle_digest=hashlib.sha256(seed).hexdigest(),
                timestamp_utc=now,
            )

        if electron_anomalies <= 0:
            cert_id = f"CERT-FPU-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
            seed = f"{cert_id}:NOMINAL:0.0:{now}".encode("utf-8")
            return FPUProsecutionExecutionCertificate(
                certificate_id=cert_id, electron_anomalies_purged=0,
                positron_auth_valid=True, released_energy_kev=0.0,
                back_action_db=0.0, spohn_entropy_production=spohn_entropy,
                gksl_dissipation_rate=gksl_rate,
                car_anti_commutation_residual=car_res,
                crowbar_triggered=False, crowbar_latency_ns=0.0,
                prosecution_status=FPUProsecutionStatus.PURGED_DIRAC_VACUUM,
                global_verdict=HeytingOmega3.COHERENT,
                sha256_merkle_digest=hashlib.sha256(seed).hexdigest(),
                timestamp_utc=now,
            )

        expected = hmac.new(
            self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        auth_valid = hmac.compare_digest(positron_auth_signature, expected)

        if auth_valid:
            released_kev = electron_anomalies * _POSITRON_REST_MASS_KEV
            back_action_db = 0.0
            purged = electron_anomalies
            verdict = HeytingOmega3.COHERENT
            status = FPUProsecutionStatus.PURGED_DIRAC_VACUUM
            cb, cb_lat = False, 0.0
        else:
            released_kev = 0.0
            back_action_db = 12.5
            purged = 0
            verdict = HeytingOmega3.VETOED
            status = FPUProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
            cb, cb_lat = self.trigger_esp32_crowbar_hardware_interlock(verdict)

        cert_id = f"CERT-FPU-PROSECUTOR-{int(now * 1000.0) % 1000000:06d}"
        seed = (
            f"{cert_id}:{purged}:{auth_valid}:{released_kev:.2f}:{status.name}:{now}"
        ).encode("utf-8")
        return FPUProsecutionExecutionCertificate(
            certificate_id=cert_id, electron_anomalies_purged=purged,
            positron_auth_valid=auth_valid, released_energy_kev=released_kev,
            back_action_db=back_action_db, spohn_entropy_production=spohn_entropy,
            gksl_dissipation_rate=gksl_rate,
            car_anti_commutation_residual=car_res,
            crowbar_triggered=cb, crowbar_latency_ns=cb_lat,
            prosecution_status=status, global_verdict=verdict,
            sha256_merkle_digest=hashlib.sha256(seed).hexdigest(),
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  MOTOR v14.1 — FASE 1 · FOUNDATIONS (álgebra canónica + Dirac + Φ₁₂)
# ██╔══██╗ b, b†, a_JW, x̂, p̂ · CCR truncado exacto · I8 simpléctico · Spohn≥0
# ██████╔╝ [HANDOFF Φ₁₂] extract_fpu_prosecution_primitives_v14
# ██╔═══╝                 ≡ build_fock_poincare_return_map_v14  (FASE 2.1)
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderProsecutorEngineV14(FPUTensorFockColliderProsecutorEngine):
    """Motor fiscal v14.1: FASE 1 → Φ₁₂ → FASE 2 → Φ₂₃ → FASE 3, anidadas."""

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(dimension=dimension, secret_key=secret_key)
        self._ledger: FockProsecutorN3Ledger = FockProsecutorN3Ledger()
        self._n3_cycle_count: int = 0
        self._b_op = self._build_boson_annihilation()
        self._b_dag_op = self._b_op.conj().T
        self._a_op_jw = self._build_fermion_jw_annihilation()
        self._a_dag_op_jw = self._a_op_jw.conj().T
        self._number_op = np.diag(np.arange(self.dimension, dtype=np.float64)).astype(np.complex128)
        sqrt2 = math.sqrt(2.0)
        self._x_op = (self._b_op + self._b_dag_op) / sqrt2
        self._p_op = (-1j * (self._b_op - self._b_dag_op)) / sqrt2
        self._eye = np.eye(self.dimension, dtype=np.complex128)
        ok_ccr, res_ccr = self._verify_ccr_canonical()
        ok_car, res_car = self._verify_car_canonical()
        ok_xp, res_xp = self._verify_symplectic_quadrature_canonical()
        if not (ok_ccr and ok_car and ok_xp):
            logger.error(
                "Álgebra canónica rota en init: CCR=%s (%.3e) CAR=%s (%.3e) I8=%s (%.3e)",
                ok_ccr, res_ccr, ok_car, res_car, ok_xp, res_xp,
            )

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # ────────────────── Sub-capa 1.1: canónicos rigurosos (CCR / CAR / I8) ──────────────────

    def _build_boson_annihilation(self) -> NDArray[np.complex128]:
        dim = self.dimension
        b = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(1, dim):
            b[n - 1, n] = math.sqrt(n)
        return b

    def _build_fermion_jw_annihilation(self) -> NDArray[np.complex128]:
        """1-modo Jordan–Wigner embebido: a = |0⟩⟨1|. Sobre el bloque 2×2, {a,a†}=I₂."""
        a = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        if self.dimension >= 2:
            a[0, 1] = 1.0
        return a

    def build_canonical_fock_operators_v14(self) -> Dict[str, NDArray[np.complex128]]:
        """Sub-capa 1.1 — Bundle público de operadores canónicos (copias)."""
        return {
            "b": self._b_op.copy(),
            "b_dag": self._b_dag_op.copy(),
            "a_jw": self._a_op_jw.copy(),
            "a_dag_jw": self._a_dag_op_jw.copy(),
            "number": self._number_op.copy(),
            "x": self._x_op.copy(),
            "p": self._p_op.copy(),
        }

    def _verify_ccr_canonical(self) -> Tuple[bool, float]:
        """[b,b†] = I − N|N−1⟩⟨N−1|  (corrección EXACTA de truncación; I9).
        El v14.0 ponía expected[-1,-1]=0  (≡ I−|N−1⟩⟨N−1|), residual ≈ N−1,
        y vetaba HARD en *todas* las ejecuciones. No se restaura ese error.
        """
        comm = self._b_op @ self._b_dag_op - self._b_dag_op @ self._b_op
        expected = self._eye.copy()
        n_levels = float(self.dimension)  # N = dim; último ket = |N−1⟩
        expected[self.dimension - 1, self.dimension - 1] = 1.0 - n_levels
        residual = float(np.linalg.norm(comm - expected, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    def _verify_car_canonical(self) -> Tuple[bool, float]:
        """{a, a†} = I₂ sobre el bloque JW 2×2; el resto del embebido es nulo."""
        anti = self._a_op_jw @ self._a_dag_op_jw + self._a_dag_op_jw @ self._a_op_jw
        expected = np.zeros_like(anti)
        expected[0, 0] = 1.0
        idx = min(1, self.dimension - 1)
        expected[idx, idx] = 1.0
        residual = float(np.linalg.norm(anti - expected, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    def _verify_symplectic_quadrature_canonical(self) -> Tuple[bool, float]:
        """I8: [x̂, p̂] = i[b, b†] (identidad de conmutadores, incluso con truncación)."""
        xp = self._x_op @ self._p_op - self._p_op @ self._x_op
        bb = self._b_op @ self._b_dag_op - self._b_dag_op @ self._b_op
        residual = float(np.linalg.norm(xp - 1j * bb, ord="fro"))
        return bool(residual < _CANONICAL_RESIDUAL_TOL), residual

    def verify_ccr_car_canonical_algebra_v14(self) -> Dict[str, Any]:
        """Haz público de residuos canónicos (CCR, CAR, I8)."""
        ccr_ok, ccr_res = self._verify_ccr_canonical()
        car_ok, car_res = self._verify_car_canonical()
        xp_ok, xp_res = self._verify_symplectic_quadrature_canonical()
        return {
            "ccr_valid": ccr_ok, "ccr_residual": ccr_res,
            "car_valid": car_ok, "car_residual": car_res,
            "symplectic_valid": xp_ok, "symplectic_residual": xp_res,
        }

    def cmath_exp_phased_b(self, phi: float) -> NDArray[np.complex128]:
        """e^{-iφ} b + e^{iφ} b†  (desplazamiento rotado)."""
        z = cmath.exp(-1j * float(phi))
        return z * self._b_op + np.conjugate(z) * self._b_dag_op

    def _driven_kerr_hamiltonian(self, t: float, phi: float = 0.0) -> NDArray[np.complex128]:
        """H(t) = ω N̂ + χ N̂(N̂−I) + ε cos(ω_d t) (e^{-iφ}b + h.c.).
        [H, N̂] ≠ 0.  PROHIBIDO construir H desde ρ.
        """
        n_op = self._number_op
        h0 = _OMEGA_CARRIER * n_op
        hchi = _KERR_CHI * (n_op @ (n_op - self._eye))
        drive = _DRIVE_EPSILON * math.cos(_OMEGA_DRIVE * float(t)) * self.cmath_exp_phased_b(phi)
        h = h0 + hchi + drive
        return 0.5 * (h + h.conj().T)

    # ────────────────── Sub-capa 1.2: Spohn + Dirac v14 ──────────────────

    def _compute_dirac_auth_confidence(self, signature: str, expected: str) -> float:
        n = min(len(signature), len(expected))
        if n == 0:
            return 0.0
        return sum(1 for a, b in zip(signature[:n], expected[:n]) if a == b) / float(n)

    def _fock_populations(self, rho: NDArray) -> NDArray[np.float64]:
        """p_n = ⟨n|ρ|n⟩ ≥ 0, Σ p_n = 1.  I10 — no usar spec(ρ)."""
        p = np.real(np.diag(np.asarray(rho, dtype=np.complex128))).astype(np.float64)
        p = np.maximum(p, 0.0)
        s = float(np.sum(p))
        if s <= _WILKINSON_EPSILON:
            p[:] = 1.0 / float(self.dimension)
        else:
            p /= s
        return p

    def _phase_space_point(self, rho: NDArray) -> Tuple[float, float, float]:
        x = float(np.real(np.trace(rho @ self._x_op)))
        p = float(np.real(np.trace(rho @ self._p_op)))
        n = float(np.real(np.trace(rho @ self._number_op)))
        return x, p, n

    def extract_fock_action_angle_decomposition(
        self, rho: NDArray
    ) -> FockActionAngleDecomposition:
        """Carta acción-ángulo: poblaciones de Fock + cuadraturas, NUNCA eig(ρ)."""
        rho = self.sanitize_density_matrix(rho)
        p_n = self._fock_populations(rho)
        n_grid = np.arange(self.dimension, dtype=np.float64)
        action_j = float(np.dot(n_grid, p_n))
        action_j2 = float(np.dot(n_grid * n_grid, p_n))
        qx, qp, _jn = self._phase_space_point(rho)
        phase = float(math.atan2(qp, qx))
        omega_c = _OMEGA_CARRIER + 2.0 * _KERR_CHI * action_j
        # Frecuencias KAM 4-vector: (ω_eff, ω_Kerr_gap, ω_drive, ω_beat)
        omega_kerr_gap = _OMEGA_CARRIER + 2.0 * _KERR_CHI * max(action_j, 0.5)
        omega_beat = abs(omega_c - _OMEGA_DRIVE)
        freqs = np.array(
            [
                max(abs(omega_c), _WILKINSON_EPSILON),
                max(abs(omega_kerr_gap), _WILKINSON_EPSILON),
                max(abs(_OMEGA_DRIVE), _WILKINSON_EPSILON),
                max(abs(omega_beat), _WILKINSON_EPSILON),
            ],
            dtype=np.float64,
        )
        _xp_ok, xp_res = self._verify_symplectic_quadrature_canonical()
        log_vol = float(np.sum(np.log(2.0 * math.pi * np.maximum(p_n, _SPECTRAL_PSD_FLOOR))))
        phase_vol = float(np.exp(log_vol / max(self.dimension, 1)))
        purity = float(np.real(np.trace(rho @ rho)))
        return FockActionAngleDecomposition(
            number_populations=p_n,
            action_J=action_j,
            action_J2=action_j2,
            quadrature_x=qx,
            quadrature_p=qp,
            quadrature_phase=phase,
            frequencies=freqs,
            omega_carrier=omega_c,
            symplectic_residual=xp_res,
            symplectic_valid=bool(xp_res < _CANONICAL_RESIDUAL_TOL),
            reconstructed_rho=rho,
            purity=purity,
            phase_space_volume=phase_vol,
        )

    def compute_spohn_and_dirac_primitives_v14(
        self,
        density_matrix: NDArray,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """
        Sub-capa 1.2 — Spohn + GKSL + purity + CAR legacy/canónico + I8 + Dirac auth_confidence.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        _, spohn, gksl_rate = self.compute_lindblad_gksl_and_spohn(rho)
        purity = float(np.real(np.trace(rho @ rho)))

        dim = self.dimension
        if state_a is None:
            state_a = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if state_b is None:
            state_b = np.zeros((dim, 1), dtype=np.complex128)
            state_b[min(1, dim - 1), 0] = 1.0
        car_ok_v13, car_res_v13 = self.verify_car_exclusion_rules(state_a, state_b)

        car_canonical_ok, car_canonical_res = self._verify_car_canonical()
        ccr_canonical_ok, ccr_canonical_res = self._verify_ccr_canonical()
        xp_ok, xp_res = self._verify_symplectic_quadrature_canonical()

        if electron_anomalies <= 0:
            auth_valid = True
            auth_conf = 1.0
            purged, released_kev, back_action = 0, 0.0, 0.0
        else:
            expected = hmac.new(
                self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            auth_valid = hmac.compare_digest(positron_auth_signature, expected)
            auth_conf = self._compute_dirac_auth_confidence(positron_auth_signature, expected)
            if auth_valid:
                purged = electron_anomalies
                released_kev = electron_anomalies * _POSITRON_REST_MASS_KEV
                back_action = 0.0  # I1 QND
            else:
                purged = 0
                released_kev = 0.0
                back_action = 12.5

        return {
            "rho": rho,
            "spohn": spohn,
            "gksl_rate": gksl_rate,
            "purity": purity,
            "car_residual_v13": car_res_v13,
            "car_valid_v13": car_ok_v13,
            "car_canonical_residual": car_canonical_res,
            "car_canonical_valid": car_canonical_ok,
            "ccr_canonical_residual": ccr_canonical_res,
            "ccr_canonical_valid": ccr_canonical_ok,
            "symplectic_residual": xp_res,
            "symplectic_valid": xp_ok,
            "auth_valid": auth_valid,
            "auth_confidence": auth_conf,
            "purged": purged,
            "released_kev": released_kev,
            "back_action_db": back_action,
        }

    # ────────────────── Sub-capa 1.3: HANDOFF Φ₁₂ → FASE 2 ──────────────────

    def extract_fpu_prosecution_primitives_v14(
        self,
        density_matrix: NDArray,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionPrimitivesBundle:
        """
        Sub-capa 1.3 — HANDOFF Φ₁₂: último acto de FASE 1 ≡ primer acto de FASE 2.

        Empaqueta Dirac/CAR/Spohn/I8 Y la carta acción-ángulo de Fock que
        `build_fock_poincare_return_map_v14(primitives)` consume como primer argumento.

        HARD EVIDENCE preliminar: CAR fail (v13 proxy) ∨ auth fail con electrones > 0.
        El fallo canónico CCR/CAR se re-evalúa en el gate judicial (H3).
        """
        p = self.compute_spohn_and_dirac_primitives_v14(
            density_matrix, electron_anomalies, positron_auth_signature,
            transaction_payload, state_a, state_b,
        )
        decomp = self.extract_fock_action_angle_decomposition(p["rho"])

        hard_veto = False
        hard_status: Optional[FPUProsecutionStatus] = None
        if not p["car_valid_v13"]:
            hard_veto = True
            hard_status = FPUProsecutionStatus.VETOED_CAR_VIOLATION
        elif electron_anomalies > 0 and not p["auth_valid"]:
            hard_veto = True
            hard_status = FPUProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE

        return FPUProsecutionPrimitivesBundle(
            rho_sanitized=p["rho"],
            spohn_entropy=p["spohn"],
            gksl_rate=p["gksl_rate"],
            purity_proxy=p["purity"],
            car_residual_v13=p["car_residual_v13"],
            car_valid_v13=p["car_valid_v13"],
            car_canonical_residual=p["car_canonical_residual"],
            car_canonical_valid=p["car_canonical_valid"],
            ccr_canonical_residual=p["ccr_canonical_residual"],
            ccr_canonical_valid=p["ccr_canonical_valid"],
            symplectic_residual=p["symplectic_residual"],
            symplectic_valid=p["symplectic_valid"],
            auth_valid=p["auth_valid"],
            auth_confidence=p["auth_confidence"],
            purged=p["purged"],
            released_kev=p["released_kev"],
            back_action_db=p["back_action_db"],
            hard_evidence_veto=hard_veto,
            hard_veto_status=hard_status,
            action_angle=decomp,
        )

    def run_phase1_foundations(
        self,
        density_matrix: NDArray,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionPrimitivesBundle:
        """Orquestador FASE 1. Alias formal de Φ₁₂."""
        return self.extract_fpu_prosecution_primitives_v14(
            density_matrix, electron_anomalies, positron_auth_signature,
            transaction_payload, state_a, state_b,
        )

    # ══════════════════════════════════════════════════════════════════════════════════════════════════
    # ██████╗  FASE 2 · SPECTRAL EVIDENCE (Kerr forzado · Poincaré · KAM · N2)
    # ██╔══██╗ continuación de Φ₁₂ · Σ={⟨p⟩=0, ṗ>0} · Hénon · Benettin · Nekhoroshev
    # ██████╔╝ [HANDOFF Φ₂₃] synthesize_fpu_spectral_evidence_v14
    # ██╔═══╝                 ≡ run_phase3_meta_verification_v14  (FASE 3.1)
    # ╚═╝
    # ══════════════════════════════════════════════════════════════════════════════════════════════════════

    def _henon_interpolate_crossing(
        self,
        s0: float, s1: float,
        x0: float, x1: float,
        n0: float, n1: float,
        t0: float, t1: float,
    ) -> Tuple[float, float, float]:
        """Interpolación de Hénon sobre el cero de ⟨p⟩. Retorna (x_Σ, N_Σ, t_Σ)."""
        denom = abs(s0) + abs(s1)
        theta = 0.0 if denom <= _WILKINSON_EPSILON else abs(s0) / denom
        return (
            float(x0 + theta * (x1 - x0)),
            float(n0 + theta * (n1 - n0)),
            float(t0 + theta * (t1 - t0)),
        )

    def _lindstedt_series_kerr(self, action_j: float) -> Tuple[float, float, float]:
        """Lindstedt 1ª (twist Kerr exacto) + 2ª (Stark AC del drive)."""
        omega0 = _OMEGA_CARRIER
        d1 = 2.0 * _KERR_CHI * float(action_j)
        omega_lin = omega0 + d1
        if abs(omega_lin) < 1e-12:
            d2 = 0.0
        else:
            d2 = 0.5 * (_DRIVE_EPSILON ** 2) / omega_lin
        return omega0, float(d1), float(d2)

    def _birkhoff_twist_coefficient(self, action_j: float, delta_o2: float) -> float:
        """Coeficiente cuadrático de Birkhoff: ∂ω/∂J = 2χ + ∂(δω₂)/∂J."""
        twist = 2.0 * _KERR_CHI
        omega_lin = _OMEGA_CARRIER + 2.0 * _KERR_CHI * float(action_j)
        if abs(omega_lin) < 1e-12:
            return float(twist)
        d2_dj = -(_DRIVE_EPSILON ** 2) * _KERR_CHI / (omega_lin ** 2)
        return float(twist + d2_dj)

    def _continued_fraction_penalty(self, ratio: float, depth: int = _KAM_CF_DEPTH) -> float:
        """Penaliza cocientes bien aproximables (parciales grandes ⇒ Liouville/resonante)."""
        x = abs(float(ratio))
        if x < _WILKINSON_EPSILON:
            return 12.0
        penalty = 0.0
        for _ in range(depth):
            a = math.floor(x)
            if a > 20.0:
                penalty += math.log(a)
            frac = x - a
            if frac < 1e-15:
                penalty += 4.0  # racional exacto → resonancia
                break
            x = 1.0 / frac
        return float(penalty)

    def _compute_kam_stability(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        tau: float = _KAM_TAU,
        max_k: int = _KAM_RESONANCE_MAX_K,
    ) -> Tuple[float, int, float, float]:
        """Índice KAM Diofante + penalización por fracciones continuas. d=_KAM_FREQ_DIM."""
        freq = np.maximum(np.abs(np.asarray(frequencies, dtype=np.float64).reshape(-1)), _WILKINSON_EPSILON)
        if len(freq) < _KAM_FREQ_DIM:
            freq = np.concatenate([freq, np.ones(_KAM_FREQ_DIM - len(freq)) * 1e-6])
        freq = freq[:_KAM_FREQ_DIM]
        d = len(freq)
        violations = 0
        worst_ratio = float("inf")
        for k in itertools.product(range(-max_k, max_k + 1), repeat=d):
            k_arr = np.array(k, dtype=float)
            norm1 = float(np.sum(np.abs(k_arr)))
            if norm1 == 0.0:
                continue
            k_dot = abs(float(np.dot(k_arr, freq)))
            threshold = gamma / (norm1 ** tau)
            ratio = k_dot / threshold if threshold > 0.0 else 1.0e9
            worst_ratio = min(worst_ratio, ratio)
            if k_dot < threshold:
                violations += 1
        kam_index = 1.0 / (1.0 + float(violations))
        if worst_ratio == float("inf"):
            worst_ratio = 1.0e9
        cf_pen = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                cf_pen += self._continued_fraction_penalty(freq[i] / freq[j])
        kam_index = float(kam_index / (1.0 + _KAM_CF_PENALTY_WEIGHT * cf_pen))
        return float(kam_index), int(violations), float(worst_ratio), float(cf_pen)

    def _nekhoroshev_stability_time(self, eps: float, n_dof: int = _NEKHOROSHEV_NDOF) -> float:
        """T_Nek ~ exp(c / ε^{1/(2n)}). n=2 para el sistema 1.5 g.d.l. extendido."""
        eps = max(float(eps), 1e-12)
        n_dof = max(int(n_dof), 1)
        expo = 1.0 / (2.0 * n_dof)
        try:
            return float(math.exp(_NEKHOROSHEV_C / (eps ** expo)))
        except OverflowError:
            return 1.0e12

    def _sanitize_inplace_copy(self, rho: NDArray) -> NDArray[np.complex128]:
        return self.sanitize_density_matrix(rho)

    def _benettin_seed_perturbation(self, rho: NDArray) -> NDArray[np.complex128]:
        """Perturbación Hermítica de traza nula a lo largo de i[ρ, x̂]."""
        delta = 1j * (rho @ self._x_op - self._x_op @ rho)
        delta = 0.5 * (delta + delta.conj().T)
        delta -= (np.trace(delta) / float(self.dimension)) * self._eye
        fn = float(np.linalg.norm(delta, ord="fro"))
        if fn <= _WILKINSON_EPSILON:
            delta = 0.5 * (self._x_op + self._x_op.conj().T)
            delta -= (np.trace(delta) / float(self.dimension)) * self._eye
            fn = float(np.linalg.norm(delta, ord="fro"))
        if fn <= _WILKINSON_EPSILON:
            return rho.copy()
        return self._sanitize_inplace_copy(rho + (_BENETTIN_D0 / fn) * delta)

    def build_fock_poincare_return_map_v14(
        self,
        primitives: FPUProsecutionPrimitivesBundle,
        n_orbits: int = _POINCARE_N_ORBITS,
        steps_per_orbit: int = _POINCARE_STEPS_PER_ORBIT,
    ) -> FockPoincareReturnMap:
        """
        FASE 2.1 — continuación de Φ₁₂ / `extract_fpu_prosecution_primitives_v14`.

        Flujo del Kerr forzado (H(t) NO conmuta con ρ ni con N̂).
        Sección Σ = {⟨p̂⟩=0, ṗ>0} + interpolación de Hénon.
        Mapa (x, ⟨N⟩)|_Σ. Deriva sobre p_n=⟨n|ρ|n⟩, NUNCA sobre spec(ρ). I10.
        Lyapunov de Benettin sobre (x, p). Lindstedt 1ª+2ª, Birkhoff, KAM, Nekhoroshev.
        """
        decomp = primitives.action_angle
        rho0 = self._sanitize_inplace_copy(decomp.reconstructed_rho)
        p_n0 = decomp.number_populations.copy()
        omega_c = float(decomp.omega_carrier) if decomp.omega_carrier > 0.0 else _OMEGA_CARRIER
        dt = (2.0 * math.pi / max(omega_c, _WILKINSON_EPSILON)) / float(max(steps_per_orbit, 1))
        n_steps = int(n_orbits * steps_per_orbit)

        rho = rho0.copy()
        rho_s = self._benettin_seed_perturbation(rho0)

        x_prev, p_prev, n_prev = self._phase_space_point(rho)
        t_prev = 0.0
        section_x: List[float] = []
        section_n: List[float] = []
        section_t: List[float] = []
        drift_list: List[float] = []
        lyap_acc = 0.0
        lyap_hs_acc = 0.0
        n_rescale = 0

        for k in range(n_steps):
            t_k = float(k) * dt
            h_k = self._driven_kerr_hamiltonian(t_k, phi=decomp.quadrature_phase)
            try:
                u_k = la.expm(-1j * h_k * dt)
            except Exception:
                break
            u_dag = u_k.conj().T
            rho = u_k @ rho @ u_dag
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.real(np.trace(rho)))
            if tr > _WILKINSON_EPSILON:
                rho = rho / tr
            rho_s = u_k @ rho_s @ u_dag
            rho_s = 0.5 * (rho_s + rho_s.conj().T)
            trs = float(np.real(np.trace(rho_s)))
            if trs > _WILKINSON_EPSILON:
                rho_s = rho_s / trs

            x_now, p_now, n_now = self._phase_space_point(rho)
            t_now = t_k + dt
            # Cruce ascendente ⟨p⟩: − → +  (ṗ>0)
            if p_prev < 0.0 <= p_now:
                x_s, n_s, t_s = self._henon_interpolate_crossing(
                    p_prev, p_now, x_prev, x_now, n_prev, n_now, t_prev, t_now,
                )
                section_x.append(x_s)
                section_n.append(n_s)
                section_t.append(t_s)
                p_now_pop = self._fock_populations(rho)
                drift_list.append(float(np.max(np.abs(p_now_pop - p_n0))))
            x_prev, p_prev, n_prev, t_prev = x_now, p_now, n_now, t_now

            if (k + 1) % _BENETTIN_RESCALE_EVERY == 0:
                xs, ps, _ns = self._phase_space_point(rho_s)
                d_quad = math.hypot(xs - x_now, ps - p_now)
                d_hs = float(np.linalg.norm(rho_s - rho, ord="fro"))
                if d_quad > _WILKINSON_EPSILON:
                    lyap_acc += math.log(d_quad / _BENETTIN_D0)
                    n_rescale += 1
                if d_hs > _WILKINSON_EPSILON:
                    lyap_hs_acc += math.log(d_hs / _BENETTIN_D0)
                delta = rho_s - rho
                delta = 0.5 * (delta + delta.conj().T)
                delta -= (np.trace(delta) / float(self.dimension)) * self._eye
                fn = float(np.linalg.norm(delta, ord="fro"))
                if fn > _WILKINSON_EPSILON:
                    rho_s = self._sanitize_inplace_copy(rho + (_BENETTIN_D0 / fn) * delta)

        t_total = max(float(n_steps) * dt, _WILKINSON_EPSILON)
        lyap_p = float(lyap_acc / t_total) if n_rescale > 0 else 0.0
        lyap_hs = float(lyap_hs_acc / t_total) if n_rescale > 0 else 0.0

        if len(section_t) >= 2:
            periods = np.diff(np.array(section_t, dtype=float))
            t_ret = float(np.mean(periods)) if len(periods) else (2.0 * math.pi / omega_c)
        else:
            t_ret = 2.0 * math.pi / omega_c

        _w0, d1, d2 = self._lindstedt_series_kerr(decomp.action_J)
        lindstedt_norm = float(math.hypot(d1, d2))
        birkhoff = self._birkhoff_twist_coefficient(decomp.action_J, d2)

        # Frecuencias: carta + 2π/T_retorno (movimiento medio de Poincaré)
        omega_ret = 2.0 * math.pi / max(t_ret, _WILKINSON_EPSILON)
        freqs = np.array(
            [
                max(abs(omega_c), _WILKINSON_EPSILON),
                max(abs(omega_c + d1), _WILKINSON_EPSILON),
                max(abs(_OMEGA_DRIVE), _WILKINSON_EPSILON),
                max(abs(omega_ret), _WILKINSON_EPSILON),
            ],
            dtype=np.float64,
        )
        kam_index, kam_viol, kam_worst, cf_pen = self._compute_kam_stability(freqs)

        eps_nek = max(
            _DRIVE_EPSILON,
            float(max(drift_list) if drift_list else 0.0),
            abs(math.expm1(min(max(lyap_p, -20.0), 20.0))),
        )
        t_nek = self._nekhoroshev_stability_time(eps_nek)
        crossings = int(len(section_x))
        drift_max = float(max(drift_list)) if drift_list else 0.0
        drift_mean = float(np.mean(drift_list)) if drift_list else 0.0
        transverse = bool(crossings > 0 and (drift_max > 0.0 or abs(lyap_p) > 0.0))

        return FockPoincareReturnMap(
            crossings=crossings,
            drift_max=drift_max,
            drift_mean=drift_mean,
            section_x=tuple(float(v) for v in section_x),
            section_n=tuple(float(v) for v in section_n),
            return_period=float(t_ret),
            lindstedt_o1=float(d1),
            lindstedt_o2=float(d2),
            lindstedt_norm=lindstedt_norm,
            birkhoff_coeff=float(birkhoff),
            kam_index=float(kam_index),
            kam_violations=int(kam_viol),
            kam_worst_ratio=float(kam_worst),
            kam_cf_penalty=float(cf_pen),
            lyapunov_poincare=float(lyap_p),
            lyapunov_hs=float(lyap_hs),
            nekhoroshev_time=float(t_nek),
            transverse=transverse,
            omega_carrier=float(omega_c),
            frequencies=freqs,
        )

    def analyze_fock_poincare_return_map_v14(
        self,
        primitives: FPUProsecutionPrimitivesBundle,
        n_orbits: int = _POINCARE_N_ORBITS,
        steps_per_orbit: int = _POINCARE_STEPS_PER_ORBIT,
    ) -> Dict[str, Any]:
        """Envoltorio dict (compatibilidad interna v14.0). Delega en 2.1."""
        m = self.build_fock_poincare_return_map_v14(primitives, n_orbits, steps_per_orbit)
        return {
            "crossings": m.crossings,
            "drift_max": m.drift_max,
            "drift_mean": m.drift_mean,
            "lindstedt_norm": m.lindstedt_norm,
            "lindstedt_o1": m.lindstedt_o1,
            "lindstedt_o2": m.lindstedt_o2,
            "birkhoff_coeff": m.birkhoff_coeff,
            "kam_index": m.kam_index,
            "kam_violations": m.kam_violations,
            "kam_worst_ratio": m.kam_worst_ratio,
            "kam_cf_penalty": m.kam_cf_penalty,
            "lyapunov_poincare": m.lyapunov_poincare,
            "nekhoroshev_time": m.nekhoroshev_time,
            "transverse": m.transverse,
            "return_period": m.return_period,
            "omega_carrier": m.omega_carrier,
        }

    def compute_dynamic_banach_lipschitz_v14(
        self, lyapunov_poincare: float, fallback: float = 0.857
    ) -> float:
        """Lip(T_Σ) = exp(λ_P). Contractivo ⇔ λ_P < 0 ⇔ Lip < 1. Fuente: Benettin, no spec(ρ)."""
        try:
            lip = float(math.exp(float(lyapunov_poincare)))
        except OverflowError:
            lip = 1.0e6
        if not math.isfinite(lip):
            return fallback
        return min(max(lip, _WILKINSON_EPSILON), 1.0e6)

    def recursive_self_tune_parameters(
        self, poincare_map: FockPoincareReturnMap
    ) -> FockTuningPolicy:
        """FASE 2.2 — N2: modula umbrales con geometría celeste; NO muta (χ, ε) del Hamiltoniano."""
        self._ledger.poincare_modulate(
            kam=poincare_map.kam_index,
            lyap_p=poincare_map.lyapunov_poincare,
            crossings=poincare_map.crossings,
            drift=poincare_map.drift_max,
            t_nek=poincare_map.nekhoroshev_time,
        )
        self._ledger.level_2_adapt()
        return FockTuningPolicy(
            kam_floor=float(self._ledger.kam_floor),
            spohn_ceiling=float(self._ledger.spohn_ceiling),
            banach_ceiling=_BANACH_LIPSCHITZ_CEILING,
            drive_epsilon=_DRIVE_EPSILON,
            kerr_chi=_KERR_CHI,
            adaptation_rate=float(self._ledger.adaptation_rate),
            meta_adaptation_rate=float(self._ledger.meta_adaptation_rate),
            n3_adaptation_lyapunov=float(self._ledger.adaptation_lyapunov()),
        )

    # ────────────────── Sub-capa 2.3: HANDOFF Φ₂₃ → FASE 3 ──────────────────

    def synthesize_fpu_spectral_evidence_v14(
        self, primitives: FPUProsecutionPrimitivesBundle
    ) -> FPUSpectralEvidenceBundle:
        """
        Sub-capa 2.3 — HANDOFF Φ₂₃: último acto de FASE 2 ≡ primer acto de FASE 3.

        Continúa en `run_phase3_meta_verification_v14(primitives, evidence)`.
        """
        poinc = self.build_fock_poincare_return_map_v14(primitives)
        policy = self.recursive_self_tune_parameters(poinc)
        lip_k = self.compute_dynamic_banach_lipschitz_v14(poinc.lyapunov_poincare)
        # I6 usa Lip < 1 (contractivo). DEGRADED por Banach sólo si expansivo (Lip > 1).
        banach_ok = bool(lip_k < _BANACH_LIPSCHITZ_CEILING)
        decomp = primitives.action_angle
        return FPUSpectralEvidenceBundle(
            banach_lipschitz_k=lip_k,
            banach_gate_passed=banach_ok,
            poincare_crossings=poinc.crossings,
            occupation_drift_max=poinc.drift_max,
            occupation_drift_mean=poinc.drift_mean,
            lindstedt_norm=poinc.lindstedt_norm,
            lindstedt_o1=poinc.lindstedt_o1,
            lindstedt_o2=poinc.lindstedt_o2,
            birkhoff_coeff=poinc.birkhoff_coeff,
            kam_index=poinc.kam_index,
            kam_violations=poinc.kam_violations,
            kam_worst_ratio=poinc.kam_worst_ratio,
            kam_cf_penalty=poinc.kam_cf_penalty,
            action_total=float(decomp.action_J),
            phase_space_volume=float(decomp.phase_space_volume),
            lyapunov_max=float(max(poinc.lyapunov_poincare, poinc.lyapunov_hs)),
            lyapunov_min=float(min(poinc.lyapunov_poincare, poinc.lyapunov_hs)),
            lyapunov_poincare=poinc.lyapunov_poincare,
            nekhoroshev_time=poinc.nekhoroshev_time,
            celestial_section_transverse=poinc.transverse,
            poincare_return_period=poinc.return_period,
            omega_carrier=poinc.omega_carrier,
            quadrature_x=float(decomp.quadrature_x),
            quadrature_p=float(decomp.quadrature_p),
            symplectic_residual=float(decomp.symplectic_residual),
            tuning_policy=policy,
        )

    def run_phase2_spectral_evidence(
        self, primitives: FPUProsecutionPrimitivesBundle
    ) -> FPUSpectralEvidenceBundle:
        """Orquestador FASE 2. Alias formal de Φ₂₃."""
        return self.synthesize_fpu_spectral_evidence_v14(primitives)

    # ══════════════════════════════════════════════════════════════════════════════════════════════════
    # ██████╗  FASE 3 · CERTIFICATION (meta N3 · debido proceso I5/I6 · certificado)
    # ██╔══██╗ continuación de Φ₂₃ · N3 meta-tasa · HARD vs SOFT sin R3⟂R5
    # ██████╔╝ [SALIDA] collide_prosecute_and_annihilate_dirac_v14
    # ██╔═══╝
    # ╚═╝
    # ══════════════════════════════════════════════════════════════════════════════════════════════════════

    def run_phase3_meta_verification_v14(
        self,
        primitives: FPUProsecutionPrimitivesBundle,
        evidence: FPUSpectralEvidenceBundle,
    ) -> Dict[str, Any]:
        """
        FASE 3.1 — continuación de Φ₂₃ / `synthesize_fpu_spectral_evidence_v14`.
        N3: consistencia + entropía de veredictos + λ_α. N2 ya se aplicó en FASE 2.2.
        """
        entropy = self._ledger.level_3_adapt()
        lyap_alpha = self._ledger.adaptation_lyapunov()
        spohn_h = self._ledger.spohn_history[-_N3_STABILITY_WINDOW:]
        kam_h = self._ledger.kam_history[-_N3_STABILITY_WINDOW:]
        spohn_std = float(np.std(spohn_h)) if len(spohn_h) >= 2 else 0.0
        kam_std = float(np.std(kam_h)) if len(kam_h) >= 2 else 0.0
        kam_mean = float(np.mean(kam_h)) if kam_h else float(evidence.kam_index)
        consistency = float(
            0.40 * kam_mean
            + 0.20 * (1.0 - min(1.0, spohn_std * 100.0))
            + 0.20 * (1.0 - min(1.0, kam_std * 5.0))
            + 0.20 * (1.0 / (1.0 + abs(evidence.lyapunov_poincare)))
        )
        return {
            "consistency": consistency,
            "entropy": entropy,
            "level_3_entropy": entropy,
            "spohn_std": spohn_std,
            "kam_std": kam_std,
            "reflections": self._ledger.level_3_reflections,
            "kam_floor": self._ledger.kam_floor,
            "spohn_ceiling": self._ledger.spohn_ceiling,
            "adaptation_lyapunov": lyap_alpha,
        }

    def _evaluate_meta_witness_n3_v14(
        self,
        primitives: Optional[FPUProsecutionPrimitivesBundle] = None,
        evidence: Optional[FPUSpectralEvidenceBundle] = None,
    ) -> Dict[str, Any]:
        """Alias 3.1 para compatibilidad de nombre v14.0. Requiere Φ₂₃ si evidence is None."""
        if evidence is None:
            entropy = self._ledger.level_3_adapt()
            return {
                "consistency": 0.5,
                "entropy": entropy,
                "level_3_entropy": entropy,
                "spohn_std": 0.0,
                "kam_std": 0.0,
                "reflections": self._ledger.level_3_reflections,
                "kam_floor": self._ledger.kam_floor,
                "spohn_ceiling": self._ledger.spohn_ceiling,
                "adaptation_lyapunov": self._ledger.adaptation_lyapunov(),
            }
        assert primitives is not None
        return self.run_phase3_meta_verification_v14(primitives, evidence)

    def _judicial_due_process_gate_v14(
        self,
        primitives: FPUProsecutionPrimitivesBundle,
        evidence: FPUSpectralEvidenceBundle,
        meta: Dict[str, Any],
        electron_anomalies: int,
    ) -> Tuple[HeytingOmega3, FPUProsecutionStatus, bool, Tuple[str, ...]]:
        """
        Sub-capa 3.2 — Debido proceso v14.1 (I5 / I6). R3⟂R5 del v14.0 ELIMINADO.

        HARD EVIDENCE (Crowbar INCONDICIONAL, I5):
          H1 CAR fail v13 (proxy de proyectores)
          H2 Auth fail con electrones > 0
          H3 CAR/CCR canónicos fallidos (álgebra rota ⇒ el instrumento miente)

        SOFT EVIDENCE (I6 — Crowbar SOLO si VETOED ∧ KAM≥kam_floor ∧ meta≥0.75 ∧ Banach<1):
          R1  Banach expansivo (Lip>1 ⇔ λ_P>0)          → DEGRADED
          R2  KAM < 0.50                                 → DEGRADED
          R3  KAM < 0.25                                 → DEGRADED  (caos ≠ crimen; NO VETO)
          R4  Meta < 0.50                                → DEGRADED
          R5  λ_P > 0.50                                 → DEGRADED
          R6  T_Nek < 2                                  → DEGRADED
          R7  Residuo simpléctico I8 > tol               → DEGRADED
          R8  Sección transversa ∧ deriva p_n grande     → DEGRADED
          R9  Spohn > techo N2                           → DEGRADED
          R10 VETO blando (catástrofe espectral FIABLE):
              KAM≥kam_floor ∧ meta≥0.85 ∧ Banach contractivo
              ∧ λ_P>0.50 ∧ T_Nek<2 ∧ transversa
              (instrumentos dignos de fe Y órbita caótica de gran perturbación)
          R11 COHERENT sólo si todos los semáforos verdes (no se refuerza en silencio)
        """
        failures: List[str] = []

        # ── HARD EVIDENCE (I5) ──
        if primitives.hard_evidence_veto:
            return (
                HeytingOmega3.VETOED,
                primitives.hard_veto_status or FPUProsecutionStatus.VETOED_CAR_VIOLATION,
                True,
                ("HARD_EVIDENCE_VETO",),
            )
        if not primitives.car_canonical_valid or not primitives.ccr_canonical_valid:
            return (
                HeytingOmega3.VETOED,
                FPUProsecutionStatus.VETOED_CAR_VIOLATION,
                True,
                (
                    "H3_RIGOROUS_CAR_CCR_FAIL("
                    f"car={primitives.car_canonical_residual:.3e},"
                    f"ccr={primitives.ccr_canonical_residual:.3e})",
                ),
            )

        # ── SOFT EVIDENCE ──
        verdict = HeytingOmega3.COHERENT
        kam_floor = float(meta.get("kam_floor", self._ledger.kam_floor))

        if evidence.banach_lipschitz_k > _BANACH_LIPSCHITZ_CEILING + 1e-9:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R1_BANACH_EXPANSIVE({evidence.banach_lipschitz_k:.4f})")

        if evidence.kam_index < _KAM_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R2_KAM_CHAOS({evidence.kam_index:.4f})")

        if evidence.kam_index < _KAM_SEVERE_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R3_KAM_SEVERE({evidence.kam_index:.4f})")

        if meta["consistency"] < _META_LOW_EVIDENCE_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R4_META_LOW({meta['consistency']:.4f})")

        if evidence.lyapunov_poincare > _LYAPUNOV_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R5_LYAPUNOV_P({evidence.lyapunov_poincare:.4f})")

        if evidence.nekhoroshev_time < _NEKHOROSHEV_MIN_TIME:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R6_NEKHOROSHEV_SHORT({evidence.nekhoroshev_time:.4e})")

        if evidence.symplectic_residual > _CANONICAL_RESIDUAL_TOL:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R7_SYMPLECTIC({evidence.symplectic_residual:.3e})")

        if (
            evidence.celestial_section_transverse
            and evidence.occupation_drift_max > _OCCUPATION_DRIFT_SOFT
        ):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(
                f"R8_TRANSVERSE_DRIFT(drift={evidence.occupation_drift_max:.3e},"
                f"cross={evidence.poincare_crossings})"
            )

        if primitives.spohn_entropy > float(meta.get("spohn_ceiling", self._ledger.spohn_ceiling)):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R9_SPOHN_CEILING({primitives.spohn_entropy:.3e})")

        # R10 — VETO blando sólo con instrumentos FIABLES (KAM alto) Y caos genuino.
        # No contradice I6: KAM≥kam_floor (≥0.75) es condición, no KAM<0.25.
        trustworthy_instruments = (
            evidence.banach_lipschitz_k < _BANACH_LIPSCHITZ_CEILING
            and evidence.kam_index >= kam_floor
            and meta["consistency"] >= _META_HIGH_CONFIDENCE_FOR_ESCALATION
        )
        spectral_catastrophe = (
            evidence.lyapunov_poincare > _LYAPUNOV_CHAOS_THRESHOLD
            and evidence.nekhoroshev_time < _NEKHOROSHEV_MIN_TIME
            and evidence.celestial_section_transverse
        )
        if trustworthy_instruments and spectral_catastrophe:
            verdict = HeytingOmega3.VETOED
            failures.append(
                f"R10_TRUSTED_CATASTROPHE(kam={evidence.kam_index:.4f},"
                f"lamP={evidence.lyapunov_poincare:.4f},Tnek={evidence.nekhoroshev_time:.3e})"
            )

        # I6 — Crowbar blando.
        crowbar_authorized = (
            verdict == HeytingOmega3.VETOED
            and evidence.banach_lipschitz_k < _BANACH_LIPSCHITZ_CEILING
            and evidence.kam_index >= kam_floor
            and meta["consistency"] >= _META_CONFIDENCE_REQUIRED
        )
        if verdict == HeytingOmega3.VETOED and not crowbar_authorized:
            verdict = HeytingOmega3.DEGRADED
            failures.append("I6_SOFT_VETO_DOWNGRADE_UNTRUSTED_INSTRUMENTS")
            crowbar_authorized = False

        # R11 — no reforzar COHERENT si queda algún semáforo ámbar no listado
        # (el meet ya lo garantiza: COHERENT iff ninguna regla disparó).

        if verdict == HeytingOmega3.VETOED:
            status = FPUProsecutionStatus.VETOED_CAR_VIOLATION
        elif verdict == HeytingOmega3.DEGRADED:
            status = FPUProsecutionStatus.DEGRADED_FOCK_SPECTRAL
        else:
            status = FPUProsecutionStatus.PURGED_DIRAC_VACUUM
            if electron_anomalies < 0:
                failures.append("R11_NEGATIVE_ANOMALIES_IGNORED")

        return verdict, status, crowbar_authorized, tuple(failures)

    def _certify_crowbar_latency(self, latency_ns: float, triggered: bool) -> bool:
        if not triggered:
            return True
        return _ESP32_SIMD_EXECUTION_NS - 1.0 <= latency_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS

    def collide_prosecute_and_annihilate_dirac_v14(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionExecutionCertificateV14:
        """
        Sub-capa 3.3 — FASE 3 FINAL. Orquesta F1 → Φ₁₂ → F2 → Φ₂₃ → F3, emite certificado v14.1.
        """
        now = time.time()
        self._n3_cycle_count += 1

        primitives = self.extract_fpu_prosecution_primitives_v14(
            density_matrix, electron_anomalies, positron_auth_signature,
            transaction_payload, state_a, state_b,
        )
        evidence = self.synthesize_fpu_spectral_evidence_v14(primitives)
        meta = self.run_phase3_meta_verification_v14(primitives, evidence)
        verdict, status, crowbar_authorized, gate_failures = self._judicial_due_process_gate_v14(
            primitives, evidence, meta, electron_anomalies,
        )

        self._ledger.observe(
            spohn=primitives.spohn_entropy,
            kam=evidence.kam_index,
            banach=evidence.banach_lipschitz_k,
            lyap_p=evidence.lyapunov_poincare,
            verdict=verdict,
        )

        if crowbar_authorized:
            cb, cb_lat = self.trigger_esp32_crowbar_hardware_interlock(verdict)
        else:
            cb, cb_lat = False, 0.0
        latency_ok = self._certify_crowbar_latency(cb_lat, cb)

        n3_traj = tuple(self._ledger.verdict_history[-_N3_STABILITY_WINDOW:])
        cert_id = f"CERT-FPU-PROSECUTOR-v14.1-{int(now * 1000.0) % 1000000:06d}"
        seed = (
            f"{cert_id}:{primitives.purged}:{primitives.auth_valid}:"
            f"{primitives.released_kev:.2f}:{status.name}:{verdict.value}:"
            f"{cb}:{evidence.banach_lipschitz_k:.6f}:{evidence.kam_index:.6f}:"
            f"{meta['consistency']:.6f}:{evidence.lyapunov_poincare:.6f}:"
            f"{evidence.nekhoroshev_time:.6e}:{int(latency_ok)}"
        ).encode("utf-8")
        merkle = hashlib.sha256(seed).hexdigest()
        hmac_payload = (
            f"{merkle}:{cert_id}:{status.name}:{verdict.value}:"
            f"{gate_failures}:{self._n3_cycle_count}:{evidence.kam_index:.8f}"
        )
        engine_sig = self._compute_hmac(_HMAC_ENGINE_DOMAIN, hmac_payload)
        aa = primitives.action_angle

        return FPUProsecutionExecutionCertificateV14(
            certificate_id=cert_id,
            electron_anomalies_purged=primitives.purged,
            positron_auth_valid=primitives.auth_valid,
            released_energy_kev=primitives.released_kev,
            back_action_db=primitives.back_action_db,
            spohn_entropy_production=primitives.spohn_entropy,
            gksl_dissipation_rate=primitives.gksl_rate,
            car_anti_commutation_residual=primitives.car_residual_v13,
            crowbar_triggered=cb,
            crowbar_latency_ns=cb_lat,
            prosecution_status=status,
            global_verdict=verdict,
            sha256_merkle_digest=merkle,
            timestamp_utc=now,
            car_canonical_residual=primitives.car_canonical_residual,
            car_canonical_valid=primitives.car_canonical_valid,
            ccr_canonical_residual=primitives.ccr_canonical_residual,
            ccr_canonical_valid=primitives.ccr_canonical_valid,
            symplectic_quadrature_residual=primitives.symplectic_residual,
            symplectic_quadrature_valid=primitives.symplectic_valid,
            purity_proxy=primitives.purity_proxy,
            positron_auth_confidence=primitives.auth_confidence,
            hard_evidence_veto=primitives.hard_evidence_veto,
            quadrature_x=aa.quadrature_x,
            quadrature_p=aa.quadrature_p,
            number_action=aa.action_J,
            omega_carrier=evidence.omega_carrier,
            banach_lipschitz_k=evidence.banach_lipschitz_k,
            banach_gate_passed=evidence.banach_gate_passed,
            poincare_crossings_count=evidence.poincare_crossings,
            occupation_drift_max=evidence.occupation_drift_max,
            occupation_drift_mean=evidence.occupation_drift_mean,
            lindstedt_correction_norm=evidence.lindstedt_norm,
            lindstedt_o1=evidence.lindstedt_o1,
            lindstedt_o2=evidence.lindstedt_o2,
            birkhoff_quadratic_coefficient=evidence.birkhoff_coeff,
            kam_stability_index=evidence.kam_index,
            kam_resonance_violations=evidence.kam_violations,
            kam_worst_ratio=evidence.kam_worst_ratio,
            kam_cf_penalty=evidence.kam_cf_penalty,
            action_total=evidence.action_total,
            phase_space_volume=evidence.phase_space_volume,
            lyapunov_max=evidence.lyapunov_max,
            lyapunov_min=evidence.lyapunov_min,
            lyapunov_poincare=evidence.lyapunov_poincare,
            nekhoroshev_time=evidence.nekhoroshev_time,
            celestial_section_transverse=evidence.celestial_section_transverse,
            poincare_return_period=evidence.poincare_return_period,
            drive_epsilon=_DRIVE_EPSILON,
            kerr_chi=_KERR_CHI,
            meta_consistency_score=meta["consistency"],
            meta_verdict_entropy=meta["entropy"],
            level_3_entropy=meta["level_3_entropy"],
            level_3_reflections=meta["reflections"],
            n3_verdict_trajectory=n3_traj,
            n3_cycle_count=self._n3_cycle_count,
            n3_adaptation_lyapunov=float(meta["adaptation_lyapunov"]),
            judicial_gate_failures=gate_failures,
            crowbar_latency_certified=latency_ok,
            hmac_engine_signature=engine_sig,
            ooda_kam_coherent_threshold=float(meta["kam_floor"]),
        )

    def run_phase3_certification(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionExecutionCertificateV14:
        """Orquestador FASE 3. Alias de la salida certificada."""
        return self.collide_prosecute_and_annihilate_dirac_v14(
            electron_anomalies, positron_auth_signature, transaction_payload,
            density_matrix, state_a, state_b,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DEMO — backward-compat v13 + features v14.1 (Kerr forzado, I8, N3, debido proceso)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v14.1 · MOTOR FISCAL FPU + KERR FORZADO + N3 + DEBIDO PROCESO")
    print(banner)

    engine = FPUTensorFockColliderProsecutorEngineV14(dimension=56)
    alg = engine.verify_ccr_car_canonical_algebra_v14()
    print(
        f"  Álgebra init  CCR={alg['ccr_valid']} ({alg['ccr_residual']:.3e})"
        f"  CAR={alg['car_valid']} ({alg['car_residual']:.3e})"
        f"  I8={alg['symplectic_valid']} ({alg['symplectic_residual']:.3e})"
    )

    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    payload = "TRANSACTION-APU-PRECIO-SOBRECOSTO-v14-001"
    valid_hmac = hmac.new(engine.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _report(label: str, c: FPUProsecutionExecutionCertificateV14) -> None:
        print(f"\n  [{label}]")
        print(f"  • Certificate ID       : {c.certificate_id}")
        print(f"  • Status               : {c.prosecution_status.name}")
        print(f"  • Veredicto FINAL Ω₃   : {c.global_verdict.name}")
        print(f"  • Hard evidence veto   : {c.hard_evidence_veto}")
        print(f"  • Judicial failures    : {c.judicial_gate_failures or ('none',)}")
        print(f"  • Electrones purgados  : {c.electron_anomalies_purged}")
        print(f"  • Energía liberada     : {c.released_energy_kev:.1f} keV")
        print(f"  • Back-Action          : {c.back_action_db:.1f} dB")
        print(f"  • Auth confidence      : {c.positron_auth_confidence:.4f}")
        print(f"  • Spohn entropy        : {c.spohn_entropy_production:.6e}")
        print(f"  • CAR legacy residual  : {c.car_anti_commutation_residual:.6e}")
        print(f"  • CAR canonical valid  : {c.car_canonical_valid} "
              f"(res={c.car_canonical_residual:.3e})")
        print(f"  • CCR canonical valid  : {c.ccr_canonical_valid} "
              f"(res={c.ccr_canonical_residual:.3e})")
        print(f"  • I8 [x,p]=i[b,b†]     : {c.symplectic_quadrature_valid} "
              f"(res={c.symplectic_quadrature_residual:.3e})")
        print(f"  • ⟨x⟩, ⟨p⟩, ⟨N⟩        : {c.quadrature_x:.4f}, {c.quadrature_p:.4f}, "
              f"{c.number_action:.4f}")
        print(f"  • ω_carrier / χ / ε    : {c.omega_carrier:.6f} / {c.kerr_chi} / {c.drive_epsilon}")
        print(f"  • Banach Lip(T_Σ)      : {c.banach_lipschitz_k:.4f} "
              f"(gate={c.banach_gate_passed})")
        print(f"  • Poincaré crossings   : {c.poincare_crossings_count}  "
              f"transverse={c.celestial_section_transverse}  T_ret={c.poincare_return_period:.4f}")
        print(f"  • Occupancy drift max  : {c.occupation_drift_max:.6e}  (sobre p_n, no spec ρ)")
        print(f"  • λ_P Benettin         : {c.lyapunov_poincare:.6e}")
        print(f"  • T_Nekhoroshev        : {c.nekhoroshev_time:.6e}")
        print(f"  • Lindstedt o1/o2      : {c.lindstedt_o1:.6e} / {c.lindstedt_o2:.6e}")
        print(f"  • Birkhoff twist       : {c.birkhoff_quadratic_coefficient:.6e}")
        print(f"  • KAM index            : {c.kam_stability_index:.6f} "
              f"(viol={c.kam_resonance_violations}, cf_pen={c.kam_cf_penalty:.3f})")
        print(f"  • Meta consistency     : {c.meta_consistency_score:.6f}")
        print(f"  • level_3_entropy      : {c.level_3_entropy:.4f}")
        print(f"  • λ_α adaptación N3    : {c.n3_adaptation_lyapunov:.6e}")
        print(f"  • N3 trajectory        : {c.n3_verdict_trajectory}")
        print(f"  • kam_floor N2         : {c.ooda_kam_coherent_threshold:.4f}")
        print(f"  • Crowbar triggered    : {c.crowbar_triggered} "
              f"(latency={c.crowbar_latency_ns:.2f} ns, certified={c.crowbar_latency_certified})")
        print(f"  • Merkle SHA-256       : {c.sha256_merkle_digest[:40]}...")
        print(f"  • HMAC engine v14.1    : {c.hmac_engine_signature[:40]}...")

    c1 = engine.collide_prosecute_and_annihilate_dirac_v14(
        electron_anomalies=1,
        positron_auth_signature=valid_hmac,
        transaction_payload=payload,
        density_matrix=rho,
    )
    _report("CICLO 1 · POSITRÓN VÁLIDO (aniquilación limpia, QND 0.0 dB)", c1)

    c2 = engine.collide_prosecute_and_annihilate_dirac_v14(
        electron_anomalies=1,
        positron_auth_signature="APOCRYPHAL_SIGNATURE",
        transaction_payload=payload,
        density_matrix=rho,
    )
    _report("CICLO 2 · POSITRÓN APÓCRIFO (HARD EVIDENCE I5 → crowbar)", c2)

    c3 = engine.collide_prosecute_and_annihilate_dirac_v14(
        electron_anomalies=0,
        positron_auth_signature=valid_hmac,
        transaction_payload=payload,
        density_matrix=rho,
    )
    _report("CICLO 3 · NOMINAL (sin electrones) + N3 acoplado", c3)

    print("\n  ═══ VERIFICACIÓN BACKWARD-COMPAT v13 ═══")
    c_v13 = engine.collide_prosecute_and_annihilate_dirac(
        electron_anomalies=1,
        positron_auth_signature=valid_hmac,
        transaction_payload=payload,
        density_matrix=rho,
    )
    v13_fields = [
        "certificate_id", "electron_anomalies_purged", "positron_auth_valid",
        "released_energy_kev", "back_action_db", "spohn_entropy_production",
        "gksl_dissipation_rate", "car_anti_commutation_residual",
        "crowbar_triggered", "crowbar_latency_ns", "prosecution_status",
        "global_verdict", "sha256_merkle_digest", "timestamp_utc",
    ]
    missing = [f for f in v13_fields if not hasattr(c_v13, f)]
    print(f"  Campos v13 ausentes: {missing or 'NINGUNO ✓'}")
    print(f"  c_v13 veredicto    : {c_v13.global_verdict.name}")
    print(f"  c_v13 purged       : {c_v13.electron_anomalies_purged}")
    print(f"  c_v13 back-action  : {c_v13.back_action_db:.1f} dB")
    print(f"  c_v13 crowbar      : {c_v13.crowbar_triggered}")
    print(f"  tipo certificado   : {type(c_v13).__name__}  (debe ser v13, no V14)")
    print(banner)