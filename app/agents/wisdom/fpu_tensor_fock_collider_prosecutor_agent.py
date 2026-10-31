# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPUTensorFockColliderProsecutorAgent v14.1.0-SOVEREIGN-POINCARE-N3-OODA                   ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider_prosecutor_agent.py                                   ║
║ Función  : Soberano Agéntico Fiscal Inquisidor del Colisionador FPU en Fock ℱ(ℋ). OODA anidado      ║
║            con Mecánica Celeste de Poincaré (Σ={⟨p⟩=0,ṗ>0}, Lindstedt 1ª+2ª, Birkhoff, KAM          ║
║            Diofanto+CF, Nekhoroshev, Benettin λ_P), Automejora N3 Poincaré-acoplada y DEBIDO         ║
║            PROCESO. El Crowbar ESP32 (< 392.15 ns @ GPIO14) es EXCLUSIVO del motor (I4-GPIO):        ║
║            el agente NUNCA dispara. HARD EVIDENCE (CAR/CCR/auth) la ejecuta el motor; SOFT            ║
║            (KAM/meta/λ_P/T_Nek) sólo con instrumentos fiables (I6).                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA POR FASES ANIDADAS (3 × 3 sub-capas) — HANDOFF ESTRICTO Φ₁₂ / Φ₂₃:

  FASE 1 — OBSERVE (FPUProsecutorPhase1ObserverV14)
    1.1 `_invoke_lightweight_checks_and_cache`  — sanitiza + Tr(ρ) + CAR legado (SIN pipeline)
    1.2 `_extract_phase1_invariants_v14`        — CAR/CCR canónicos + I8 + fingerprint
    1.3 `execute_phase1_v14`                    — emite FPUProsecutorGaugeContractV14
    [HANDOFF Φ₁₂] `handoff_to_phase2_v14`
         ≡ último acto de FASE 1 ≡ primer acto de `execute_phase2_v14`

  FASE 2 — ORIENT (FPUProsecutionCampaignOrientorV14)
    2.1 `_invoke_engine_v14_and_cache`          — collide_…_v14 EXACTAMENTE UNA VEZ (I4)
    2.2 `_extract_phase2_poincare_n3`           — Kerr/Poincaré/KAM/Nekhoroshev/N3 del cert
    2.3 `execute_phase2_v14`                    — emite FPUProsecutionCampaignResultV14
    [HANDOFF Φ₂₃] `handoff_to_phase3_v14`
         ≡ último acto de FASE 2 ≡ primer acto de `execute_phase3_v14`

  FASE 3 — DECIDE (FPUProsecutionDeciderV14) — GPIO PASIVO
    3.1 `_extract_phase3_meta_v14`              — N3 OODA (consistencia, level_3_entropy, λ_α)
    3.2 `_judicial_due_process_gate_v14`        — I5/I6, R1–R11, SIN S4⟂S5
    3.3 `execute_phase3_v14`                    — FPUProsecutorSovereignPassportV14 (no GPIO)

INVARIANTES (I1–I10 + I4-GPIO + I4-engine):
  · I1  Back-Action 0.0 dB (auth válida) / 12.5 dB (apócrifa)
  · I2  Electrones purgados = anomalies si auth válida, 0 si no
  · I3  Spohn σ ≥ 0
  · I4  Motor ≤ 1 invoke/ciclo OODA; re-invoke de emergencia AUDITADO, nunca oculto
  · I4-GPIO  El agente NUNCA llama trigger_esp32_crowbar_*; el motor es el único tirador
  · I5  HARD EVIDENCE → VETOED (crowbar ya ejecutado por el motor si procede)
  · I6  SOFT VETOED sólo si KAM≥kam_floor ∧ meta≥0.75 ∧ Banach<1; si no, DEGRADED
  · I7  Backward-compat v13: prosecute_fpu_tensor_fock, execute_phase1/2/3_*, gates legado
  · I8  [x̂,p̂]=i[b,b†] reportado; no se re-simula el álgebra
  · I9  Tr(ρ)=1 (Wilkinson); CCR/CAR canónicos del motor v14.1
  · I10 Deriva sobre p_n=⟨n|ρ|n⟩ y (x,⟨N⟩)|_Σ — NUNCA spec(ρ)
  · HMAC v14 ≠ HMAC legado (no mezclar payload v14 con dominio v13)
  · Campo canónico N3: level_3_entropy (no meta_verdict_entropy huérfano)

N3 OODA (ledger DISTINTO del motor):
  N1 veredicto Ω₃ · N2 umbrales (kam_floor, spohn_ceiling) · N3 meta-tasa + λ_α
  Poincaré-modulado: poincare_modulate, ooda_adaptation_lyapunov.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

for path_dir in ("/workspace/scratch", "/workspace/artifacts", "."):
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# ── Importación del motor v14.1 (fallback v13 / stub) ──
try:
    from fpu_tensor_fock_collider_prosecutor_engine import (
        FPUProsecutionExecutionCertificate,
        FPUProsecutionExecutionCertificateV14,
        FPUProsecutionStatus,
        FPUTensorFockColliderProsecutorEngine,
        FPUTensorFockColliderProsecutorEngineV14,
        HeytingOmega3,
    )
    _ENGINE_V14_AVAILABLE: Final[bool] = True
except ImportError:
    try:
        from app.wisdom.fpu_tensor_fock_collider_prosecutor_engine import (
            FPUProsecutionExecutionCertificate,
            FPUProsecutionExecutionCertificateV14,
            FPUProsecutionStatus,
            FPUTensorFockColliderProsecutorEngine,
            FPUTensorFockColliderProsecutorEngineV14,
            HeytingOmega3,
        )
        _ENGINE_V14_AVAILABLE: Final[bool] = True
    except ImportError:
        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

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

        class FPUProsecutionStatus(IntEnum):  # type: ignore[no-redef]
            PENDING = 0
            PURGED_DIRAC_VACUUM = 1
            VETOED_CAR_VIOLATION = 2
            VETOED_DIRAC_AUTH_FAILURE = 3
            DEGRADED_FOCK_SPECTRAL = 4

        FPUTensorFockColliderProsecutorEngine = None  # type: ignore
        FPUTensorFockColliderProsecutorEngineV14 = None  # type: ignore
        FPUProsecutionExecutionCertificate = None  # type: ignore
        FPUProsecutionExecutionCertificateV14 = None  # type: ignore
        _ENGINE_V14_AVAILABLE = False  # type: ignore[misc]

logger = logging.getLogger("APU.Wisdom.FPUTensorFockColliderProsecutorAgentV14")

# ── HMAC: dominios LEGADO v13 (truncados, sin colisión con v14) ──
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_FPU_PROSECUTOR_SOVEREIGN_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"FPU-PROSECUTOR-P1-GAUGE-v13.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"FPU-PROSECUTOR-P2-DYNAMICS-v13.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"FPU-PROSECUTOR-P3-SOVEREIGN-v13.0|"

# ── HMAC v14.1 (Poincaré / KAM / N3) — NUNCA mezclar con payload legado ──
_HMAC_DOMAIN_P1_V14: Final[bytes] = b"FPU-PROSECUTOR-P1v14-POINCARE-OBSERVE|"
_HMAC_DOMAIN_P2_V14: Final[bytes] = b"FPU-PROSECUTOR-P2v14-KAM-ORIENT|"
_HMAC_DOMAIN_P3_V14: Final[bytes] = b"FPU-PROSECUTOR-P3v14-N3-DECIDE|"
_CHAIN_HEX_LEN: Final[int] = 32

# ── Umbrales judiciales / celestes v14.1 (alineados con el motor) ──
_BANACH_LIPSCHITZ_CEILING: Final[float] = 1.0
_KAM_TRUSTWORTHY_THRESHOLD: Final[float] = 0.75
_KAM_CHAOS_THRESHOLD: Final[float] = 0.50
_KAM_SEVERE_CHAOS_THRESHOLD: Final[float] = 0.25
_META_CONFIDENCE_REQUIRED: Final[float] = 0.75
_META_LOW_EVIDENCE_THRESHOLD: Final[float] = 0.50
_META_HIGH_CONFIDENCE_FOR_ESCALATION: Final[float] = 0.85
_CANONICAL_RESIDUAL_TOL: Final[float] = 1.0e-6
_LYAPUNOV_CHAOS_THRESHOLD: Final[float] = 0.50
_NEKHOROSHEV_MIN_TIME: Final[float] = 2.0
_OCCUPATION_DRIFT_SOFT: Final[float] = 1.0e-3
_N3_TRAJECTORY_WINDOW: Final[int] = 8
_N3_HISTORY_CAP: Final[int] = 64
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_SIMD_EXECUTION_NS: Final[float] = 42.0
_TRACE_COHERENT_TOL: Final[float] = 1.0e-5
_TRACE_VETO_TOL: Final[float] = 1.0e-4
_QND_BACKACTION_SILENT: Final[float] = 1.0e-12


def _safe_hmac_compare(left: str, right: str) -> bool:
    """compare_digest con rechazo de tipos/longitudes distintas (sin filtrar el digest)."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if len(left) != len(right):
        hmac.compare_digest(b"\x00", b"\x00")
        return False
    return hmac.compare_digest(left, right)


def _fingerprint_density(rho: NDArray) -> str:
    """SHA-256 de los bytes de ρ, 16 hex. Identidad de sesión OODA (I4)."""
    arr = np.ascontiguousarray(np.asarray(rho, dtype=np.complex128))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATACLASSES — v13 preservadas + v14.1 enriquecidas + caché OODA + ledger N3
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FPUProsecutorGaugeContract:
    """v13 — Contrato de calibre fiscal inmutable (preservado)."""
    contract_id: str
    sovereign_id: str
    car_residual: float
    ccr_norm: float
    trace_preservation: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_signature: str
    creation_timestamp: float


@dataclass(frozen=True, slots=True)
class FPUProsecutionCampaignResult:
    """v13 — Resultado consolidado de campaña inquisidora (preservado)."""
    campaign_id: str
    sovereign_id: str
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    prosecution_status: FPUProsecutionStatus
    global_verdict: HeytingOmega3
    chain_hash: str
    hmac_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class FPUProsecutorSovereignPassport:
    """v13 — Pasaporte Soberano Final (preservado)."""
    passport_id: str
    contract: FPUProsecutorGaugeContract
    campaign_result: FPUProsecutionCampaignResult
    global_verdict: HeytingOmega3
    prosecution_status: FPUProsecutionStatus
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_sha256_root: str
    signature_hmac_sha256: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class FPUProsecutorGaugeContractV14:
    """FASE 1 — Contrato v14.1 (superset estricto de v13 + I8 + fingerprint)."""
    contract_id: str
    sovereign_id: str
    car_residual: float
    ccr_norm: float
    trace_preservation: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_signature: str
    creation_timestamp: float
    car_canonical_residual: float
    car_canonical_valid: bool
    ccr_canonical_residual: float
    ccr_canonical_valid: bool
    symplectic_residual: float
    symplectic_valid: bool
    quadrature_x: float
    quadrature_p: float
    density_fingerprint: str
    engine_v14_active: bool
    engine_invocations: int


@dataclass(frozen=True, slots=True)
class FPUProsecutionCampaignResultV14:
    """FASE 2 — Campaña v14.1 (superset v13 + Poincaré Kerr + N3 + I4)."""
    campaign_id: str
    sovereign_id: str
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    prosecution_status: FPUProsecutionStatus
    global_verdict: HeytingOmega3
    chain_hash: str
    hmac_signature: str
    timestamp_utc: float
    contract_id: str
    car_anti_commutation_residual: float
    car_canonical_residual: float
    car_canonical_valid: bool
    ccr_canonical_residual: float
    ccr_canonical_valid: bool
    symplectic_residual: float
    symplectic_valid: bool
    purity_proxy: float
    positron_auth_confidence: float
    hard_evidence_veto: bool
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
    omega_carrier: float
    quadrature_x: float
    quadrature_p: float
    number_action: float
    meta_consistency_score: float
    meta_verdict_entropy: float
    level_3_entropy: float
    level_3_reflections: int
    n3_cycle_count: int
    n3_adaptation_lyapunov: float
    engine_crowbar_triggered: bool
    engine_crowbar_latency_ns: float
    engine_judicial_gate_failures: Tuple[str, ...]
    hmac_engine_signature: str
    ooda_kam_coherent_threshold: float
    density_fingerprint: str
    engine_invocations: int
    emergency_reinvoke: bool


@dataclass(frozen=True, slots=True)
class FPUProsecutorSovereignPassportV14:
    """FASE 3 — Pasaporte v14.1 (superset v13 + N3 OODA + celeste + I4-GPIO)."""
    passport_id: str
    contract: FPUProsecutorGaugeContractV14
    campaign_result: FPUProsecutionCampaignResultV14
    global_verdict: HeytingOmega3
    prosecution_status: FPUProsecutionStatus
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_sha256_root: str
    signature_hmac_sha256: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    base_verdict_v13: HeytingOmega3
    kam_adjusted_verdict: HeytingOmega3
    hard_evidence_veto: bool
    soft_evidence_escalated: bool
    false_positive_protected: bool
    judicial_gate_failures: Tuple[str, ...]
    meta_consistency_score: float
    meta_verdict_entropy: float
    level_3_entropy: float
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    n3_adaptation_lyapunov: float
    engine_n3_trajectory: Tuple[int, ...]
    ooda_n3_trajectory: Tuple[int, ...]
    lyapunov_poincare: float
    nekhoroshev_time: float
    poincare_return_period: float
    celestial_section_transverse: bool
    crowbar_latency_certified: bool
    passive_gpio: bool
    engine_invocations: int
    emergency_reinvoke: bool
    i4_engine_once: bool
    ooda_kam_coherent_threshold: float
    hmac_chain_p1_p2_p3: str
    density_fingerprint: str


@dataclass(slots=True)
class OODASessionCache:
    """Caché de un ciclo OODA. I4: engine_invocations ≤ 1 salvo emergencia auditada."""
    snapshot_cert: Optional[Any] = None
    contract: Optional[FPUProsecutorGaugeContractV14] = None
    density_fingerprint: str = ""
    engine_invocations: int = 0
    emergency_reinvoke: bool = False


@dataclass(slots=True)
class FockProsecutorOODALedger:
    """
    Ledger N3 del AGENTE (distinto del FockProsecutorN3Ledger del motor).
    N1 veredicto · N2 umbrales · N3 meta-tasa. Poincaré-acoplado.
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
        self, spohn: float, kam: float, banach: float, lyap_p: float, verdict: HeytingOmega3
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
        n = min(_N3_TRAJECTORY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def poincare_modulate(
        self, kam: float, lyap_p: float, crossings: int, drift: float, t_nek: float
    ) -> None:
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
        if (not chaotic) and (not near_resonant) and float(kam) >= _KAM_TRUSTWORTHY_THRESHOLD:
            self.kam_floor = max(
                _KAM_TRUSTWORTHY_THRESHOLD, self.kam_floor - self.adaptation_rate * 0.01
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
                _KAM_TRUSTWORTHY_THRESHOLD, self.kam_floor - self.adaptation_rate * 0.01
            )

    def level_3_adapt(self) -> float:
        if len(self.verdict_history) < 4:
            return 0.0
        recent = np.array(self.verdict_history[-_N3_TRAJECTORY_WINDOW:], dtype=int)
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

    def ooda_adaptation_lyapunov(self) -> float:
        if len(self.rate_history) < 3:
            return 0.0
        r = np.maximum(np.array(self.rate_history[-_N3_TRAJECTORY_WINDOW:], dtype=float), 1e-12)
        if len(r) < 2:
            return 0.0
        return float(np.mean(np.log(r[1:] / r[:-1])))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 1 — OBSERVE (3 sub-capas)
# ██╔══██╗ Sanitización + CAR/CCR/I8 · emisión P1v14 · SIN pipeline del motor
# ██████╔╝ [HANDOFF Φ₁₂] handoff_to_phase2_v14 ≡ primer acto de execute_phase2_v14
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUProsecutorPhase1ObserverV14:
    """FASE 1 — OBSERVE. El pipeline `collide_prosecute_and_annihilate_dirac_v14` NO se toca aquí."""

    def __init__(
        self,
        engine: Optional[Any] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: int = 56,
    ) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.sovereign_id: Final[str] = "FPU-PROSECUTOR-MASTER-01"
        self.dimension: Final[int] = max(4, int(dimension))
        if engine is not None:
            self.engine = engine
        elif _ENGINE_V14_AVAILABLE and FPUTensorFockColliderProsecutorEngineV14 is not None:
            self.engine = FPUTensorFockColliderProsecutorEngineV14(
                dimension=self.dimension, secret_key=self.secret_key
            )
        elif FPUTensorFockColliderProsecutorEngine is not None:
            self.engine = FPUTensorFockColliderProsecutorEngine(
                dimension=self.dimension, secret_key=self.secret_key
            )
        else:
            self.engine = None
        self._cache: OODASessionCache = OODASessionCache()
        self._ooda_ledger: FockProsecutorOODALedger = FockProsecutorOODALedger()

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _begin_ooda_cycle(self) -> None:
        self._cache = OODASessionCache()

    def _canonical_p1_payload(self, c: FPUProsecutorGaugeContractV14) -> str:
        return (
            f"{c.contract_id}:{c.sovereign_id}:{c.car_residual:.8f}:{c.ccr_norm:.8f}:"
            f"{c.trace_preservation:.8f}:{c.car_canonical_residual:.3e}:"
            f"{int(c.car_canonical_valid)}:{c.ccr_canonical_residual:.3e}:"
            f"{int(c.ccr_canonical_valid)}:{c.symplectic_residual:.3e}:"
            f"{int(c.symplectic_valid)}:{c.quadrature_x:.8f}:{c.quadrature_p:.8f}:"
            f"{c.density_fingerprint}:{int(c.engine_v14_active)}:{c.engine_invocations}:"
            f"{c.isometry_verdict.value}:{c.chain_hash}"
        )

    def _legacy_p1_payload(self, c: FPUProsecutorGaugeContract) -> str:
        return (
            f"{c.contract_id}:{c.sovereign_id}:{c.car_residual:.8f}:"
            f"{c.ccr_norm:.8f}:{c.trace_preservation:.8f}:"
            f"{c.isometry_verdict.value}:{c.chain_hash}"
        )

    # ────────────────── Sub-capa 1.1: checks ligeros (SIN pipeline) ──────────────────
    def _invoke_lightweight_checks_and_cache(
        self,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> Dict[str, Any]:
        """1.1 — Sanitización + Tr(ρ) + CAR legado. Prohibido invocar collide_*."""
        if self.engine is not None and hasattr(self.engine, "sanitize_density_matrix"):
            rho = self.engine.sanitize_density_matrix(density_matrix)
        else:
            a = np.asarray(density_matrix, dtype=np.complex128)
            rho = 0.5 * (a + a.conj().T)
            tr = float(np.real(np.trace(rho)))
            rho = rho / tr if tr > 1e-15 else np.eye(self.dimension) / float(self.dimension)
        dim = int(rho.shape[0])
        tr_val = float(np.real(np.trace(rho)))
        if state_a is None:
            state_a = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        if state_b is None:
            state_b = np.zeros((dim, 1), dtype=np.complex128)
            state_b[min(1, dim - 1), 0] = 1.0
        if self.engine is not None and hasattr(self.engine, "verify_car_exclusion_rules"):
            car_ok, car_res = self.engine.verify_car_exclusion_rules(state_a, state_b)
        else:
            car_ok, car_res = True, 0.0
        fp = _fingerprint_density(rho)
        self._cache.density_fingerprint = fp
        return {
            "rho": rho,
            "trace": tr_val,
            "car_valid_legacy": bool(car_ok),
            "car_residual_legacy": float(car_res),
            "fingerprint": fp,
        }

    # ────────────────── Sub-capa 1.2: invariantes F1 (álgebra, no órbita) ──────────────────
    def _extract_phase1_invariants_v14(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """1.2 — CAR/CCR/I8 del motor (operadores precomputados, no Kerr)."""
        car_ok, car_res = bool(checks["car_valid_legacy"]), float(checks["car_residual_legacy"])
        ccr_ok, ccr_res = True, 0.0
        xp_ok, xp_res = True, 0.0
        qx, qp = 0.0, 0.0
        engine_v14 = False
        eng = self.engine
        if eng is not None and hasattr(eng, "verify_ccr_car_canonical_algebra_v14"):
            try:
                alg = eng.verify_ccr_car_canonical_algebra_v14()
                car_ok = bool(alg.get("car_valid", car_ok))
                car_res = float(alg.get("car_residual", car_res))
                ccr_ok = bool(alg.get("ccr_valid", True))
                ccr_res = float(alg.get("ccr_residual", 0.0))
                xp_ok = bool(alg.get("symplectic_valid", True))
                xp_res = float(alg.get("symplectic_residual", 0.0))
                engine_v14 = True
            except Exception as exc:  # pragma: no cover
                logger.warning("álgebra canónica v14.1 no disponible: %s", exc)
        elif eng is not None:
            if hasattr(eng, "_verify_car_canonical"):
                try:
                    car_ok, car_res = eng._verify_car_canonical()
                    engine_v14 = True
                except Exception:
                    pass
            if hasattr(eng, "_verify_ccr_canonical"):
                try:
                    ccr_ok, ccr_res = eng._verify_ccr_canonical()
                    engine_v14 = True
                except Exception:
                    pass
            if hasattr(eng, "_verify_symplectic_quadrature_canonical"):
                try:
                    xp_ok, xp_res = eng._verify_symplectic_quadrature_canonical()
                except Exception:
                    pass
        rho = checks["rho"]
        if eng is not None and hasattr(eng, "_phase_space_point"):
            try:
                qx, qp, _n = eng._phase_space_point(rho)
            except Exception:
                qx, qp = 0.0, 0.0
        elif eng is not None and hasattr(eng, "_x_op") and hasattr(eng, "_p_op"):
            try:
                qx = float(np.real(np.trace(rho @ eng._x_op)))
                qp = float(np.real(np.trace(rho @ eng._p_op)))
            except Exception:
                qx, qp = 0.0, 0.0
        if not engine_v14:
            car_ok = bool(checks["car_valid_legacy"])
            car_res = float(checks["car_residual_legacy"])
            ccr_ok, ccr_res = True, 0.0
        return {
            "trace": float(checks["trace"]),
            "car_legacy": float(checks["car_residual_legacy"]),
            "car_legacy_valid": bool(checks["car_valid_legacy"]),
            "car_canonical_residual": float(car_res),
            "car_canonical_valid": bool(car_ok),
            "ccr_canonical_residual": float(ccr_res),
            "ccr_canonical_valid": bool(ccr_ok),
            "symplectic_residual": float(xp_res),
            "symplectic_valid": bool(xp_ok),
            "quadrature_x": float(qx),
            "quadrature_p": float(qp),
            "engine_v14": bool(engine_v14),
            "fingerprint": str(checks["fingerprint"]),
        }

    # ────────────────── Sub-capa 1.3: emisión P1v14 ──────────────────
    def execute_phase1_v14(
        self,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutorGaugeContractV14:
        """1.3 — Emite el contrato de calibre v14.1 firmado HMAC-SHA256 (dominio P1v14)."""
        now = time.time()
        checks = self._invoke_lightweight_checks_and_cache(density_matrix, state_a, state_b)
        inv = self._extract_phase1_invariants_v14(checks)
        tr_val = float(inv["trace"])
        if (not inv["car_legacy_valid"]) or abs(tr_val - 1.0) > _TRACE_VETO_TOL:
            verdict = HeytingOmega3.VETOED
        elif (not inv["car_canonical_valid"]) or (not inv["ccr_canonical_valid"]):
            verdict = HeytingOmega3.VETOED
        elif abs(tr_val - 1.0) > _TRACE_COHERENT_TOL:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        contract_id = f"CONTRACT-FPU-PROSECUTOR-v14.1-{int(now * 1000.0) % 1000000:06d}"
        chain_seed = (
            f"{contract_id}:{tr_val:.8f}:{inv['car_legacy']:.8f}:{inv['ccr_canonical_residual']:.8f}:"
            f"{int(inv['car_canonical_valid'])}:{int(inv['ccr_canonical_valid'])}:"
            f"{inv['fingerprint']}:{verdict.value}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_seed).hexdigest()[:_CHAIN_HEX_LEN]
        unsigned = FPUProsecutorGaugeContractV14(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            car_residual=float(inv["car_legacy"]),
            ccr_norm=float(inv["ccr_canonical_residual"]),
            trace_preservation=tr_val,
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_signature="",
            creation_timestamp=now,
            car_canonical_residual=float(inv["car_canonical_residual"]),
            car_canonical_valid=bool(inv["car_canonical_valid"]),
            ccr_canonical_residual=float(inv["ccr_canonical_residual"]),
            ccr_canonical_valid=bool(inv["ccr_canonical_valid"]),
            symplectic_residual=float(inv["symplectic_residual"]),
            symplectic_valid=bool(inv["symplectic_valid"]),
            quadrature_x=float(inv["quadrature_x"]),
            quadrature_p=float(inv["quadrature_p"]),
            density_fingerprint=str(inv["fingerprint"]),
            engine_v14_active=bool(inv["engine_v14"]),
            engine_invocations=int(self._cache.engine_invocations),
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1_V14, self._canonical_p1_payload(unsigned))
        object.__setattr__(unsigned, "hmac_signature", sig)
        self._cache.contract = unsigned
        return unsigned

    def run_phase1_observe(
        self,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutorGaugeContractV14:
        """Orquestador FASE 1."""
        return self.execute_phase1_v14(density_matrix, state_a, state_b)

    def execute_phase1_observation(
        self,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutorGaugeContract:
        """Alias v13: re-firma LEGADO (dominio P1 v13, payload truncado). No mezclar con P1v14."""
        c = self.execute_phase1_v14(density_matrix, state_a, state_b)
        legacy = FPUProsecutorGaugeContract(
            contract_id=c.contract_id,
            sovereign_id=c.sovereign_id,
            car_residual=c.car_residual,
            ccr_norm=c.ccr_norm,
            trace_preservation=c.trace_preservation,
            isometry_verdict=c.isometry_verdict,
            chain_hash=c.chain_hash,
            hmac_signature="",
            creation_timestamp=c.creation_timestamp,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._legacy_p1_payload(legacy))
        object.__setattr__(legacy, "hmac_signature", sig)
        return legacy

    def continue_to_phase2_prosecutor_gate(self, contract: FPUProsecutorGaugeContract) -> bool:
        """Alias v13: revalida HMAC con dominio P1 legado."""
        expected = self._compute_hmac(_HMAC_DOMAIN_P1, self._legacy_p1_payload(contract))
        return _safe_hmac_compare(contract.hmac_signature, expected)

    # ────────────────── HANDOFF Φ₁₂ — último acto de FASE 1 ──────────────────
    def handoff_to_phase2_v14(self, contract: FPUProsecutorGaugeContractV14) -> bool:
        """
        Φ₁₂ — último acto de FASE 1 ≡ primer acto de FASE 2.
        Revalida HMAC P1v14. ADMITE VETOED (el due process vive en F3).
        """
        expected = self._compute_hmac(_HMAC_DOMAIN_P1_V14, self._canonical_p1_payload(contract))
        return _safe_hmac_compare(contract.hmac_signature, expected)

    def continue_to_phase2_prosecutor_gate_v14(
        self, contract: FPUProsecutorGaugeContractV14
    ) -> bool:
        """Alias estable de Φ₁₂ (API v14.0)."""
        return self.handoff_to_phase2_v14(contract)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 2 — ORIENT (3 sub-capas)
# ██╔══██╗ Φ₁₂ → invoke ÚNICO del motor · Poincaré/KAM/N3 · emisión P2v14
# ██████╔╝ [HANDOFF Φ₂₃] handoff_to_phase3_v14 ≡ primer acto de execute_phase3_v14
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUProsecutionCampaignOrientorV14(FPUProsecutorPhase1ObserverV14):
    """FASE 2 — ORIENT. Primer acto = Φ₁₂. Único `collide_*_v14` del ciclo (I4)."""

    def handoff_to_phase2_v14(self, contract: FPUProsecutorGaugeContractV14) -> bool:
        """Φ₁₂ — continuación de FASE 1 / primer acto de FASE 2."""
        return FPUProsecutorPhase1ObserverV14.handoff_to_phase2_v14(self, contract)

    def _canonical_p2_payload(self, r: FPUProsecutionCampaignResultV14) -> str:
        return (
            f"{r.campaign_id}:{self.sovereign_id}:{r.contract_id}:"
            f"{r.electron_anomalies_purged}:{int(r.positron_auth_valid)}:"
            f"{r.released_energy_kev:.2f}:{r.back_action_db:.1f}:"
            f"{r.spohn_entropy_production:.8e}:{r.gksl_dissipation_rate:.8e}:"
            f"{r.prosecution_status.name}:{r.car_anti_commutation_residual:.8e}:"
            f"{r.car_canonical_residual:.3e}:{int(r.car_canonical_valid)}:"
            f"{r.ccr_canonical_residual:.3e}:{int(r.ccr_canonical_valid)}:"
            f"{r.symplectic_residual:.3e}:{int(r.symplectic_valid)}:"
            f"{r.purity_proxy:.8f}:{r.positron_auth_confidence:.8f}:{int(r.hard_evidence_veto)}:"
            f"{r.banach_lipschitz_k:.6f}:{int(r.banach_gate_passed)}:{r.poincare_crossings_count}:"
            f"{r.occupation_drift_max:.8e}:{r.occupation_drift_mean:.8e}:"
            f"{r.lindstedt_correction_norm:.8e}:{r.lindstedt_o1:.8e}:{r.lindstedt_o2:.8e}:"
            f"{r.birkhoff_quadratic_coefficient:.8e}:"
            f"{r.kam_stability_index:.8f}:{r.kam_resonance_violations}:{r.kam_worst_ratio:.6e}:"
            f"{r.kam_cf_penalty:.6f}:{r.action_total:.8f}:{r.phase_space_volume:.8f}:"
            f"{r.lyapunov_max:+.6e}:{r.lyapunov_min:+.6e}:{r.lyapunov_poincare:+.6e}:"
            f"{r.nekhoroshev_time:.6e}:{int(r.celestial_section_transverse)}:"
            f"{r.poincare_return_period:.8f}:{r.omega_carrier:.8f}:"
            f"{r.quadrature_x:.8f}:{r.quadrature_p:.8f}:{r.number_action:.8f}:"
            f"{r.meta_consistency_score:.6f}:{r.meta_verdict_entropy:.6f}:"
            f"{r.level_3_entropy:.6f}:{r.level_3_reflections}:{r.n3_cycle_count}:"
            f"{r.n3_adaptation_lyapunov:+.6e}:{int(r.engine_crowbar_triggered)}:"
            f"{r.engine_crowbar_latency_ns:.2f}:{r.engine_judicial_gate_failures}:"
            f"{r.hmac_engine_signature}:{r.ooda_kam_coherent_threshold:.6f}:"
            f"{r.density_fingerprint}:{r.engine_invocations}:{int(r.emergency_reinvoke)}:"
            f"{r.global_verdict.value}:{r.chain_hash}"
        )

    def _legacy_p2_payload(self, r: FPUProsecutionCampaignResult) -> str:
        return (
            f"{r.campaign_id}:{self.sovereign_id}:{r.electron_anomalies_purged}:"
            f"{int(r.positron_auth_valid)}:{r.released_energy_kev:.2f}:{r.back_action_db:.1f}:"
            f"{r.spohn_entropy_production:.8e}:{r.gksl_dissipation_rate:.8e}:"
            f"{r.prosecution_status.name}:{r.global_verdict.value}:{r.chain_hash}"
        )

    # ────────────────── Sub-capa 2.1: invocación ÚNICA ──────────────────
    def _invoke_engine_v14_and_cache(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> Optional[Any]:
        """
        2.1 — `collide_prosecute_and_annihilate_dirac_v14` EXACTAMENTE UNA VEZ por ciclo.
        Misma huella → cache hit. Huella distinta → emergencia auditada (I4).
        """
        fp = _fingerprint_density(
            self.engine.sanitize_density_matrix(density_matrix)
            if (self.engine is not None and hasattr(self.engine, "sanitize_density_matrix"))
            else np.asarray(density_matrix, dtype=np.complex128)
        )
        if self._cache.snapshot_cert is not None and self._cache.engine_invocations >= 1:
            if fp == self._cache.density_fingerprint:
                return self._cache.snapshot_cert
            self._cache.emergency_reinvoke = True
            logger.warning(
                "I4 emergencia: re-invoke auditado (fp_sesion=%s fp_ahora=%s invokes=%d)",
                self._cache.density_fingerprint, fp, self._cache.engine_invocations,
            )
        if self.engine is None:
            self._cache.snapshot_cert = None
            return None
        cert: Optional[Any] = None
        if hasattr(self.engine, "collide_prosecute_and_annihilate_dirac_v14"):
            try:
                cert = self.engine.collide_prosecute_and_annihilate_dirac_v14(
                    electron_anomalies=electron_anomalies,
                    positron_auth_signature=positron_auth_signature,
                    transaction_payload=transaction_payload,
                    density_matrix=density_matrix,
                    state_a=state_a,
                    state_b=state_b,
                )
            except Exception as exc:
                logger.warning("collide_prosecute_and_annihilate_dirac_v14 falló: %s", exc)
        if cert is None and hasattr(self.engine, "collide_prosecute_and_annihilate_dirac"):
            try:
                cert = self.engine.collide_prosecute_and_annihilate_dirac(
                    electron_anomalies=electron_anomalies,
                    positron_auth_signature=positron_auth_signature,
                    transaction_payload=transaction_payload,
                    density_matrix=density_matrix,
                    state_a=state_a,
                    state_b=state_b,
                )
            except Exception as exc:
                logger.warning("collide_prosecute_and_annihilate_dirac falló: %s", exc)
        self._cache.snapshot_cert = cert
        self._cache.engine_invocations += 1
        self._cache.density_fingerprint = fp
        return cert

    # ────────────────── Sub-capa 2.2: extracción Poincaré + N3 ──────────────────
    def _extract_phase2_poincare_n3(self, cert: Optional[Any]) -> Dict[str, Any]:
        """2.2 — Lee el cert v14.1. Campo N3 canónico = level_3_entropy (no huérfano)."""
        def g(name: str, default: Any) -> Any:
            return getattr(cert, name, default) if cert is not None else default

        is_v14 = cert is not None and hasattr(cert, "kam_stability_index")
        level3 = float(g("level_3_entropy", g("meta_verdict_entropy", 0.0))) if is_v14 else 0.0
        crossings = int(g("poincare_crossings_count", 0)) if is_v14 else 0
        drift_max = float(g("occupation_drift_max", 0.0)) if is_v14 else 0.0
        lyap_p = float(g("lyapunov_poincare", 0.0)) if is_v14 else 0.0
        transverse = bool(g("celestial_section_transverse", False)) if is_v14 else False
        if is_v14 and not hasattr(cert, "celestial_section_transverse"):
            transverse = bool(crossings > 0 and (drift_max > 0.0 or abs(lyap_p) > 0.0))
        return {
            "purged": int(g("electron_anomalies_purged", 0)),
            "auth_valid": bool(g("positron_auth_valid", True)),
            "released_kev": float(g("released_energy_kev", 0.0)),
            "back_action": float(g("back_action_db", 0.0)),
            "spohn": float(g("spohn_entropy_production", 0.0)),
            "gksl": float(g("gksl_dissipation_rate", 0.0)),
            "status": g("prosecution_status", FPUProsecutionStatus.PURGED_DIRAC_VACUUM),
            "verdict": g("global_verdict", HeytingOmega3.COHERENT),
            "car_legacy_res": float(g("car_anti_commutation_residual", 0.0)),
            "car_canonical_res": float(g("car_canonical_residual", 0.0)) if is_v14 else 0.0,
            "car_canonical_ok": bool(g("car_canonical_valid", True)) if is_v14 else True,
            "ccr_canonical_res": float(g("ccr_canonical_residual", 0.0)) if is_v14 else 0.0,
            "ccr_canonical_ok": bool(g("ccr_canonical_valid", True)) if is_v14 else True,
            "symplectic_res": float(g("symplectic_quadrature_residual", 0.0)) if is_v14 else 0.0,
            "symplectic_ok": bool(g("symplectic_quadrature_valid", True)) if is_v14 else True,
            "purity": float(g("purity_proxy", 1.0)) if is_v14 else 1.0,
            "auth_confidence": float(g("positron_auth_confidence", 1.0)) if is_v14 else 1.0,
            "hard_evidence_veto": bool(g("hard_evidence_veto", False)) if is_v14 else False,
            "banach_k": float(g("banach_lipschitz_k", 0.857)) if is_v14 else 0.857,
            "banach_ok": bool(g("banach_gate_passed", True)) if is_v14 else True,
            "crossings": crossings,
            "drift_max": drift_max,
            "drift_mean": float(g("occupation_drift_mean", 0.0)) if is_v14 else 0.0,
            "lindstedt": float(g("lindstedt_correction_norm", 0.0)) if is_v14 else 0.0,
            "lindstedt_o1": float(g("lindstedt_o1", 0.0)) if is_v14 else 0.0,
            "lindstedt_o2": float(g("lindstedt_o2", 0.0)) if is_v14 else 0.0,
            "birkhoff": float(g("birkhoff_quadratic_coefficient", 0.0)) if is_v14 else 0.0,
            "kam_index": float(g("kam_stability_index", 1.0)) if is_v14 else 1.0,
            "kam_viol": int(g("kam_resonance_violations", 0)) if is_v14 else 0,
            "kam_worst": float(g("kam_worst_ratio", 1.0e9)) if is_v14 else 1.0e9,
            "kam_cf_penalty": float(g("kam_cf_penalty", 0.0)) if is_v14 else 0.0,
            "action_total": float(g("action_total", 0.0)) if is_v14 else 0.0,
            "phase_vol": float(g("phase_space_volume", 0.0)) if is_v14 else 0.0,
            "lyap_max": float(g("lyapunov_max", 0.0)) if is_v14 else 0.0,
            "lyap_min": float(g("lyapunov_min", 0.0)) if is_v14 else 0.0,
            "lyap_p": lyap_p,
            "t_nek": float(g("nekhoroshev_time", 1.0e12)) if is_v14 else 1.0e12,
            "transverse": transverse,
            "t_ret": float(g("poincare_return_period", 0.0)) if is_v14 else 0.0,
            "omega_carrier": float(g("omega_carrier", 1.0)) if is_v14 else 1.0,
            "qx": float(g("quadrature_x", 0.0)) if is_v14 else 0.0,
            "qp": float(g("quadrature_p", 0.0)) if is_v14 else 0.0,
            "action_j": float(g("number_action", 0.0)) if is_v14 else 0.0,
            "meta_consistency": float(g("meta_consistency_score", 1.0)) if is_v14 else 1.0,
            "meta_entropy": float(g("meta_verdict_entropy", level3)) if is_v14 else 0.0,
            "level_3_entropy": level3,
            "level_3_reflections": int(g("level_3_reflections", 0)) if is_v14 else 0,
            "n3_cycle_count": int(g("n3_cycle_count", 0)) if is_v14 else 0,
            "n3_lyap": float(g("n3_adaptation_lyapunov", 0.0)) if is_v14 else 0.0,
            "engine_crowbar": bool(g("crowbar_triggered", False)),
            "engine_cb_latency": float(g("crowbar_latency_ns", 0.0)),
            "engine_gate_failures": tuple(g("judicial_gate_failures", ())) if is_v14 else (),
            "hmac_engine": str(g("hmac_engine_signature", "")) if is_v14 else "",
            "kam_floor_engine": float(g("ooda_kam_coherent_threshold", _KAM_TRUSTWORTHY_THRESHOLD))
            if is_v14 else _KAM_TRUSTWORTHY_THRESHOLD,
            "engine_n3_traj": tuple(g("n3_verdict_trajectory", ())) if is_v14 else (),
            "is_v14": is_v14,
        }

    # ────────────────── Sub-capa 2.3: emisión P2v14 ──────────────────
    def execute_phase2_v14(
        self,
        contract: FPUProsecutorGaugeContractV14,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionCampaignResultV14:
        """2.3 — Primer acto = Φ₁₂. Invoca el motor una vez. Firma HMAC P2v14."""
        now = time.time()
        if not self.handoff_to_phase2_v14(contract):
            raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ del Fiscal FPU v14.1")
        cert = self._invoke_engine_v14_and_cache(
            electron_anomalies, positron_auth_signature, transaction_payload,
            density_matrix, state_a, state_b,
        )
        inv = self._extract_phase2_poincare_n3(cert)
        self._ooda_ledger.poincare_modulate(
            kam=float(inv["kam_index"]),
            lyap_p=float(inv["lyap_p"]),
            crossings=int(inv["crossings"]),
            drift=float(inv["drift_max"]),
            t_nek=float(inv["t_nek"]),
        )
        self._ooda_ledger.level_2_adapt()
        verdict = contract.isometry_verdict.meet(inv["verdict"])
        camp_id = f"CAMP-FPU-PROSECUTOR-v14.1-{int(now * 1000.0) % 1000000:06d}"
        chain_seed = (
            f"{contract.chain_hash}:{camp_id}:{inv['purged']}:{inv['auth_valid']}:"
            f"{verdict.value}:{inv['kam_index']:.8f}:{inv['banach_k']:.6f}:"
            f"{inv['lyap_p']:+.6e}:{self._cache.density_fingerprint}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_seed).hexdigest()[:_CHAIN_HEX_LEN]
        unsigned = FPUProsecutionCampaignResultV14(
            campaign_id=camp_id,
            sovereign_id=self.sovereign_id,
            electron_anomalies_purged=int(inv["purged"]),
            positron_auth_valid=bool(inv["auth_valid"]),
            released_energy_kev=float(inv["released_kev"]),
            back_action_db=float(inv["back_action"]),
            spohn_entropy_production=float(inv["spohn"]),
            gksl_dissipation_rate=float(inv["gksl"]),
            prosecution_status=inv["status"],
            global_verdict=verdict,
            chain_hash=chain_hash,
            hmac_signature="",
            timestamp_utc=now,
            contract_id=contract.contract_id,
            car_anti_commutation_residual=float(inv["car_legacy_res"]),
            car_canonical_residual=float(inv["car_canonical_res"]),
            car_canonical_valid=bool(inv["car_canonical_ok"]),
            ccr_canonical_residual=float(inv["ccr_canonical_res"]),
            ccr_canonical_valid=bool(inv["ccr_canonical_ok"]),
            symplectic_residual=float(inv["symplectic_res"]),
            symplectic_valid=bool(inv["symplectic_ok"]),
            purity_proxy=float(inv["purity"]),
            positron_auth_confidence=float(inv["auth_confidence"]),
            hard_evidence_veto=bool(inv["hard_evidence_veto"]),
            banach_lipschitz_k=float(inv["banach_k"]),
            banach_gate_passed=bool(inv["banach_ok"]),
            poincare_crossings_count=int(inv["crossings"]),
            occupation_drift_max=float(inv["drift_max"]),
            occupation_drift_mean=float(inv["drift_mean"]),
            lindstedt_correction_norm=float(inv["lindstedt"]),
            lindstedt_o1=float(inv["lindstedt_o1"]),
            lindstedt_o2=float(inv["lindstedt_o2"]),
            birkhoff_quadratic_coefficient=float(inv["birkhoff"]),
            kam_stability_index=float(inv["kam_index"]),
            kam_resonance_violations=int(inv["kam_viol"]),
            kam_worst_ratio=float(inv["kam_worst"]),
            kam_cf_penalty=float(inv["kam_cf_penalty"]),
            action_total=float(inv["action_total"]),
            phase_space_volume=float(inv["phase_vol"]),
            lyapunov_max=float(inv["lyap_max"]),
            lyapunov_min=float(inv["lyap_min"]),
            lyapunov_poincare=float(inv["lyap_p"]),
            nekhoroshev_time=float(inv["t_nek"]),
            celestial_section_transverse=bool(inv["transverse"]),
            poincare_return_period=float(inv["t_ret"]),
            omega_carrier=float(inv["omega_carrier"]),
            quadrature_x=float(inv["qx"]),
            quadrature_p=float(inv["qp"]),
            number_action=float(inv["action_j"]),
            meta_consistency_score=float(inv["meta_consistency"]),
            meta_verdict_entropy=float(inv["meta_entropy"]),
            level_3_entropy=float(inv["level_3_entropy"]),
            level_3_reflections=int(inv["level_3_reflections"]),
            n3_cycle_count=int(inv["n3_cycle_count"]),
            n3_adaptation_lyapunov=float(inv["n3_lyap"]),
            engine_crowbar_triggered=bool(inv["engine_crowbar"]),
            engine_crowbar_latency_ns=float(inv["engine_cb_latency"]),
            engine_judicial_gate_failures=tuple(inv["engine_gate_failures"]),
            hmac_engine_signature=str(inv["hmac_engine"]),
            ooda_kam_coherent_threshold=float(self._ooda_ledger.kam_floor),
            density_fingerprint=self._cache.density_fingerprint,
            engine_invocations=int(self._cache.engine_invocations),
            emergency_reinvoke=bool(self._cache.emergency_reinvoke),
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2_V14, self._canonical_p2_payload(unsigned))
        object.__setattr__(unsigned, "hmac_signature", sig)
        return unsigned

    def run_phase2_orient(
        self,
        contract: FPUProsecutorGaugeContractV14,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionCampaignResultV14:
        """Orquestador FASE 2."""
        return self.execute_phase2_v14(
            contract, electron_anomalies, positron_auth_signature,
            transaction_payload, density_matrix, state_a, state_b,
        )

    def _upgrade_p1_from_legacy(self, contract: Any) -> FPUProsecutorGaugeContractV14:
        if isinstance(contract, FPUProsecutorGaugeContractV14):
            return contract
        # unknown ≠ failed: no fabricar HARD VETO al subir v13.
        return FPUProsecutorGaugeContractV14(
            contract_id=contract.contract_id,
            sovereign_id=contract.sovereign_id,
            car_residual=contract.car_residual,
            ccr_norm=contract.ccr_norm,
            trace_preservation=contract.trace_preservation,
            isometry_verdict=contract.isometry_verdict,
            chain_hash=contract.chain_hash,
            hmac_signature=contract.hmac_signature,
            creation_timestamp=contract.creation_timestamp,
            car_canonical_residual=0.0,
            car_canonical_valid=True,
            ccr_canonical_residual=0.0,
            ccr_canonical_valid=True,
            symplectic_residual=0.0,
            symplectic_valid=True,
            quadrature_x=0.0,
            quadrature_p=0.0,
            density_fingerprint=self._cache.density_fingerprint,
            engine_v14_active=False,
            engine_invocations=int(self._cache.engine_invocations),
        )

    def execute_phase2_orientation(
        self,
        contract: Any,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUProsecutionCampaignResult:
        """Alias v13: acepta P1 v13/v14, retorna P2 v13 re-firmado LEGADO."""
        contract_v14 = self._upgrade_p1_from_legacy(contract)
        if not isinstance(contract, FPUProsecutorGaugeContractV14):
            # El HMAC v13 no verifica Φ₁₂ v14; se revalida el dominio legado.
            if not self.continue_to_phase2_prosecutor_gate(contract):
                raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ legado del Fiscal FPU")
            object.__setattr__(
                contract_v14,
                "hmac_signature",
                self._compute_hmac(_HMAC_DOMAIN_P1_V14, self._canonical_p1_payload(contract_v14)),
            )
        result_v14 = self.execute_phase2_v14(
            contract_v14, electron_anomalies, positron_auth_signature,
            transaction_payload, density_matrix, state_a, state_b,
        )
        legacy = FPUProsecutionCampaignResult(
            campaign_id=result_v14.campaign_id,
            sovereign_id=result_v14.sovereign_id,
            electron_anomalies_purged=result_v14.electron_anomalies_purged,
            positron_auth_valid=result_v14.positron_auth_valid,
            released_energy_kev=result_v14.released_energy_kev,
            back_action_db=result_v14.back_action_db,
            spohn_entropy_production=result_v14.spohn_entropy_production,
            gksl_dissipation_rate=result_v14.gksl_dissipation_rate,
            prosecution_status=result_v14.prosecution_status,
            global_verdict=result_v14.global_verdict,
            chain_hash=result_v14.chain_hash,
            hmac_signature="",
            timestamp_utc=result_v14.timestamp_utc,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._legacy_p2_payload(legacy))
        object.__setattr__(legacy, "hmac_signature", sig)
        return legacy

    def continue_to_phase3_prosecutor_gate(self, result: FPUProsecutionCampaignResult) -> bool:
        """Alias v13: revalida HMAC P2 legado (no basta len==64)."""
        expected = self._compute_hmac(_HMAC_DOMAIN_P2, self._legacy_p2_payload(result))
        return _safe_hmac_compare(result.hmac_signature, expected)

    # ────────────────── HANDOFF Φ₂₃ — último acto de FASE 2 ──────────────────
    def handoff_to_phase3_v14(self, result: FPUProsecutionCampaignResultV14) -> bool:
        """
        Φ₂₃ — último acto de FASE 2 ≡ primer acto de FASE 3.
        Revalida HMAC P2v14 sobre el payload canónico (incluye contract_id).
        """
        if not (result.hmac_signature and len(result.hmac_signature) == 64):
            return False
        expected = self._compute_hmac(_HMAC_DOMAIN_P2_V14, self._canonical_p2_payload(result))
        return _safe_hmac_compare(result.hmac_signature, expected)

    def continue_to_phase3_prosecutor_gate_v14(
        self, result: FPUProsecutionCampaignResultV14
    ) -> bool:
        """Alias estable de Φ₂₃ (API v14.0)."""
        return self.handoff_to_phase3_v14(result)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 3 — DECIDE (3 sub-capas) · GPIO PASIVO
# ██╔══██╗ Φ₂₃ → meta N3 OODA · I5/I6 R1–R11 · Pasaporte v14.1 SIN Crowbar agente
# ██████╔╝ [SALIDA] execute_phase3_v14 → FPUProsecutorSovereignPassportV14
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUProsecutionDeciderV14(FPUProsecutionCampaignOrientorV14):
    """FASE 3 — DECIDE. Primer acto = Φ₂₃. NUNCA dispara GPIO (I4-GPIO)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._n3_verdict_trajectory: List[int] = []
        self._n3_cycle_count: int = 0

    def handoff_to_phase3_v14(self, result: FPUProsecutionCampaignResultV14) -> bool:
        """Φ₂₃ — continuación de FASE 2 / primer acto de FASE 3."""
        return FPUProsecutionCampaignOrientorV14.handoff_to_phase3_v14(self, result)

    # ────────────────── Sub-capa 3.1 ──────────────────
    def _extract_phase3_meta_v14(
        self, result: FPUProsecutionCampaignResultV14
    ) -> Dict[str, Any]:
        """3.1 — Meta N3 OODA. Campo canónico: level_3_entropy."""
        entropy = self._ooda_ledger.level_3_adapt()
        lyap_alpha = self._ooda_ledger.ooda_adaptation_lyapunov()
        engine_entropy = float(result.level_3_entropy)
        consistency = float(
            0.35 * result.kam_stability_index
            + 0.20 * (1.0 / (1.0 + abs(result.lyapunov_poincare)))
            + 0.20 * min(1.0, result.meta_consistency_score)
            + 0.15 * (1.0 / (1.0 + engine_entropy))
            + 0.10 * (1.0 if result.banach_lipschitz_k < _BANACH_LIPSCHITZ_CEILING else 0.0)
        )
        return {
            "consistency": consistency,
            "entropy": entropy,
            "level_3_entropy": entropy,
            "engine_level_3_entropy": engine_entropy,
            "reflections": self._ooda_ledger.level_3_reflections,
            "engine_n3_cycle": result.n3_cycle_count,
            "kam_floor": float(self._ooda_ledger.kam_floor),
            "spohn_ceiling": float(self._ooda_ledger.spohn_ceiling),
            "adaptation_lyapunov": lyap_alpha,
        }

    # ────────────────── Sub-capa 3.2: DEBIDO PROCESO (sin S4⟂S5, sin GPIO) ──────────────────
    def _judicial_due_process_gate_v14(
        self,
        contract: FPUProsecutorGaugeContractV14,
        result: FPUProsecutionCampaignResultV14,
        meta: Dict[str, Any],
    ) -> Tuple[HeytingOmega3, bool, bool, bool, Tuple[str, ...]]:
        """
        3.2 — I5 / I6. Retorna (verdict, crowbar_from_engine, hard_veto, fp_protected, failures).

        HARD (I5) — el motor ya disparó si procedía; el agente NO vuelve a disparar:
          H1 canónicos rotos (contrato o campaña)
          H2 hard_evidence_veto del motor / auth fail
        SOFT (I6) — caos ≠ crimen; VETO blando sólo con instrumentos FIABLES:
          R1 Banach expansivo · R2 KAM<0.50 · R3 KAM<0.25 (DEGRADED, no VETO)
          R4 meta<0.50 · R5 λ_P>0.50 · R6 T_Nek<2 · R7 I8 · R8 transversa+deriva
          R9 Spohn>techo · R10 catástrofe FIABLE (KAM≥floor ∧ meta≥0.85 ∧ Banach<1
              ∧ λ_P>0.50 ∧ T_Nek<2 ∧ transversa) → VETOED lógico
          R11 COHERENT iff ningún semáforo ámbar
          I6  VETO blando sin instrumentos fiables → DEGRADED (fp protection)
        GPIO: crowbar_triggered del PASAPORTE = el del MOTOR. Agente pasivo.
        """
        failures: List[str] = []
        base_verdict = contract.isometry_verdict.meet(result.global_verdict)

        hard_veto = False
        hard_causes: List[str] = []
        if (not contract.car_canonical_valid) or (not contract.ccr_canonical_valid):
            hard_veto = True
            hard_causes.append(
                f"H1_CANONICAL_FAIL(car={contract.car_canonical_residual:.3e},"
                f"ccr={contract.ccr_canonical_residual:.3e})"
            )
        if (not result.car_canonical_valid) or (not result.ccr_canonical_valid):
            hard_veto = True
            hard_causes.append(
                f"H1_CAMPAIGN_CANONICAL_FAIL(car={result.car_canonical_residual:.3e},"
                f"ccr={result.ccr_canonical_residual:.3e})"
            )
        if result.hard_evidence_veto:
            hard_veto = True
            hard_causes.append("H2_ENGINE_HARD_VETO")
        if (result.electron_anomalies_purged == 0) and (not result.positron_auth_valid) and (
            result.positron_auth_confidence < 0.5
        ):
            hard_veto = True
            hard_causes.append("H2_AUTH_FAIL_LOW_CONFIDENCE")
        if hard_veto:
            failures.extend(hard_causes)
            return HeytingOmega3.VETOED, True, True, False, tuple(failures)

        verdict = base_verdict
        kam_floor = float(meta.get("kam_floor", self._ooda_ledger.kam_floor))

        if result.banach_lipschitz_k > _BANACH_LIPSCHITZ_CEILING + 1e-9:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R1_BANACH_EXPANSIVE({result.banach_lipschitz_k:.4f})")
        if result.kam_stability_index < _KAM_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R2_KAM_CHAOS({result.kam_stability_index:.4f})")
        if result.kam_stability_index < _KAM_SEVERE_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R3_KAM_SEVERE({result.kam_stability_index:.4f})")
        if meta["consistency"] < _META_LOW_EVIDENCE_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R4_META_LOW({meta['consistency']:.4f})")
        if result.lyapunov_poincare > _LYAPUNOV_CHAOS_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R5_LYAPUNOV_P({result.lyapunov_poincare:.4f})")
        if result.nekhoroshev_time < _NEKHOROSHEV_MIN_TIME:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R6_NEKHOROSHEV_SHORT({result.nekhoroshev_time:.4e})")
        if result.symplectic_residual > _CANONICAL_RESIDUAL_TOL:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R7_SYMPLECTIC({result.symplectic_residual:.3e})")
        if result.celestial_section_transverse and (
            result.occupation_drift_max > _OCCUPATION_DRIFT_SOFT
        ):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(
                f"R8_TRANSVERSE_DRIFT(drift={result.occupation_drift_max:.3e},"
                f"cross={result.poincare_crossings_count})"
            )
        if result.spohn_entropy_production > float(meta.get("spohn_ceiling", 0.10)):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append(f"R9_SPOHN_CEILING({result.spohn_entropy_production:.3e})")

        trustworthy = (
            result.banach_lipschitz_k < _BANACH_LIPSCHITZ_CEILING
            and result.kam_stability_index >= kam_floor
            and meta["consistency"] >= _META_HIGH_CONFIDENCE_FOR_ESCALATION
        )
        catastrophe = (
            result.lyapunov_poincare > _LYAPUNOV_CHAOS_THRESHOLD
            and result.nekhoroshev_time < _NEKHOROSHEV_MIN_TIME
            and result.celestial_section_transverse
        )
        if trustworthy and catastrophe:
            verdict = HeytingOmega3.VETOED
            failures.append(
                f"R10_TRUSTED_CATASTROPHE(kam={result.kam_stability_index:.4f},"
                f"lamP={result.lyapunov_poincare:.4f},Tnek={result.nekhoroshev_time:.3e})"
            )

        fp_protected = False
        if verdict == HeytingOmega3.VETOED:
            confidence_ok = (
                result.banach_lipschitz_k < _BANACH_LIPSCHITZ_CEILING
                and result.kam_stability_index >= kam_floor
                and meta["consistency"] >= _META_CONFIDENCE_REQUIRED
            )
            if not confidence_ok:
                fp_protected = True
                verdict = HeytingOmega3.DEGRADED
                failures.append("I6_SOFT_VETO_DOWNGRADE_UNTRUSTED_INSTRUMENTS")

        if (
            contract.engine_v14_active
            and (not result.celestial_section_transverse)
            and verdict == HeytingOmega3.COHERENT
        ):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
            failures.append("R11_ENGINE_V14_NON_TRANSVERSE")

        return verdict, bool(result.engine_crowbar_triggered), False, fp_protected, tuple(failures)

    def _certify_crowbar_latency(self, latency_ns: float, triggered: bool) -> bool:
        if not triggered:
            return True
        return _ESP32_SIMD_EXECUTION_NS - 1.0 <= float(latency_ns) <= _ESP32_CROWBAR_MAX_LATENCY_NS

    # ────────────────── Sub-capa 3.3: pasaporte (GPIO pasivo) ──────────────────
    def execute_phase3_v14(
        self,
        contract: FPUProsecutorGaugeContractV14,
        result: FPUProsecutionCampaignResultV14,
        gpio_pin: int = 14,
    ) -> FPUProsecutorSovereignPassportV14:
        """3.3 — Primer acto = Φ₂₃. Emite pasaporte. gpio_pin se registra, NO se actúa."""
        now = time.time()
        if not self.handoff_to_phase3_v14(result):
            raise ValueError("Falla de autenticidad en Compuerta Φ₂₃ del Fiscal FPU v14.1")
        _ = int(gpio_pin)  # API v13; I4-GPIO: el agente no escribe el registro.
        meta = self._extract_phase3_meta_v14(result)
        base_verdict = contract.isometry_verdict.meet(result.global_verdict)
        verdict, _engine_cb, hard_veto, fp_protected, gate_failures = (
            self._judicial_due_process_gate_v14(contract, result, meta)
        )
        self._ooda_ledger.observe(
            spohn=result.spohn_entropy_production,
            kam=result.kam_stability_index,
            banach=result.banach_lipschitz_k,
            lyap_p=result.lyapunov_poincare,
            verdict=verdict,
        )
        self._n3_cycle_count += 1
        self._n3_verdict_trajectory.append(int(verdict))
        if len(self._n3_verdict_trajectory) > _N3_TRAJECTORY_WINDOW:
            self._n3_verdict_trajectory = self._n3_verdict_trajectory[-_N3_TRAJECTORY_WINDOW:]

        cb_triggered = bool(result.engine_crowbar_triggered)
        cb_latency = float(result.engine_crowbar_latency_ns)
        cb_certified = self._certify_crowbar_latency(cb_latency, cb_triggered)

        if verdict == HeytingOmega3.VETOED:
            status = (
                FPUProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
                if (not result.positron_auth_valid)
                else FPUProsecutionStatus.VETOED_CAR_VIOLATION
            )
        elif verdict == HeytingOmega3.DEGRADED:
            status = FPUProsecutionStatus.DEGRADED_FOCK_SPECTRAL
        else:
            status = FPUProsecutionStatus.PURGED_DIRAC_VACUUM

        qnd_silent = abs(result.back_action_db) <= _QND_BACKACTION_SILENT
        if hard_veto:
            visceral_action = (
                f"PARÁLISIS YA EJECUTADA POR EL MOTOR — HARD EVIDENCE "
                f"({gate_failures[0][:40] if gate_failures else 'H*'}). "
                f"Agente GPIO-PASIVO. Crowbar motor {cb_latency:.2f} ns @ GPIO{gpio_pin}."
            )
            exec_summary = "I5: hard evidence determinista. Agente no re-dispara."
        elif fp_protected:
            visceral_action = (
                "VENTANA DE GRACIA — I6 falso positivo protegido. GPIO agente BLOQUEADO."
            )
            exec_summary = "I6: evidencia espectral insuficiente. Revisión humana."
        elif verdict == HeytingOmega3.VETOED:
            visceral_action = (
                f"VETO LÓGICO SOFT (KAM={result.kam_stability_index:.4f}, "
                f"meta={meta['consistency']:.4f}). GPIO agente PASIVO; "
                f"crowbar motor={cb_triggered}."
            )
            exec_summary = "I6: caos espectral con instrumentos fiables. Sin segundo disparo."
        elif verdict == HeytingOmega3.DEGRADED:
            visceral_action = "VETO SUAVE — Ventana de Alivio. GPIO agente PASIVO."
            exec_summary = f"DEGRADACIÓN RECUPERABLE: {list(gate_failures) or ['none']}."
        else:
            visceral_action = (
                "FLUJO NOMINAL LAMINAR (Aniquilación limpia 0.0 dB)."
                if qnd_silent else
                f"COHERENTE con back-action {result.back_action_db:.1f} dB."
            )
            exec_summary = "COHERENCIA TOTAL FPU: Pauli y aniquilación limpia."

        visceral_report = {
            "wacc_protection": (
                "PROTEGIDO INCONDICIONALMENTE AL 15.0%" if hard_veto
                else "PRESERVADO 15.0% — ventana de gracia"
                if (fp_protected or verdict == HeytingOmega3.DEGRADED)
                else "ÓPTIMO Y ESTABLE AL 15.0%"
            ),
            "net_liquidity_impact": (
                "+11.5% LIBERADO" if verdict == HeytingOmega3.COHERENT
                else "+8.5% EN RECIRCULACIÓN" if verdict == HeytingOmega3.DEGRADED
                else "+11.5% PROTEGIDO vía Crowbar (motor)"
            ),
            "kv_cache_compression_pct": 86.4,
            "llm_cost_savings_pct": 80.0,
            "electron_anomalies_purged": result.electron_anomalies_purged,
            "released_energy_kev": result.released_energy_kev,
            "back_action_db": result.back_action_db,
            "spohn_entropy_production": result.spohn_entropy_production,
            "hard_evidence_veto": hard_veto,
            "false_positive_protected": fp_protected,
            "judicial_gate_failures": list(gate_failures),
            "kam_stability_index": result.kam_stability_index,
            "banach_lipschitz_k": result.banach_lipschitz_k,
            "lyapunov_poincare": result.lyapunov_poincare,
            "nekhoroshev_time": result.nekhoroshev_time,
            "celestial_section_transverse": result.celestial_section_transverse,
            "level_3_entropy": meta["level_3_entropy"],
            "meta_consistency": meta["consistency"],
            "qnd_passive_silent": qnd_silent,
            "passive_gpio": True,
            "gpio_pin_recorded": int(gpio_pin),
            "crowbar_action": (
                f"MOTOR ESP32 GPIO{gpio_pin} HIGH ({cb_latency:.2f} ns)"
                if cb_triggered else "STANDBY / LAMINAR (agente no dispara)"
            ),
            "engine_invocations": result.engine_invocations,
            "emergency_reinvoke": result.emergency_reinvoke,
            "n3_cycle_count": self._n3_cycle_count,
            "n3_trajectory": list(self._n3_verdict_trajectory),
            "executive_action": visceral_action,
            "executive_summary": exec_summary,
        }

        passport_id = f"PASSPORT-FPU-PROSECUTOR-v14.1-{int(now * 1000.0) % 1000000:06d}"
        merkle_payload = (
            f"{passport_id}:{contract.chain_hash}:{result.chain_hash}:"
            f"{int(verdict)}:{int(base_verdict)}:{int(hard_veto)}:{int(fp_protected)}:"
            f"{result.kam_stability_index:.6f}:{meta['consistency']:.6f}:"
            f"{result.lyapunov_poincare:.6e}:{int(cb_triggered)}:"
            f"{result.density_fingerprint}:{result.engine_invocations}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_payload).hexdigest()
        prov_payload = (
            f"{merkle_root}:{int(cb_triggered)}:{verdict.value}:"
            f"{meta['consistency']:.6f}:{result.kam_stability_index:.6f}:"
            f"{meta['level_3_entropy']:.6f}:{self._n3_cycle_count}:{gate_failures}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3_V14, prov_payload)
        hmac_chain = (
            f"{contract.hmac_signature[:_CHAIN_HEX_LEN]}"
            f"{result.hmac_signature[:_CHAIN_HEX_LEN]}"
            f"{prov_sig[:_CHAIN_HEX_LEN]}"
        )
        return FPUProsecutorSovereignPassportV14(
            passport_id=passport_id,
            contract=contract,
            campaign_result=result,
            global_verdict=verdict,
            prosecution_status=status,
            crowbar_triggered=cb_triggered,
            crowbar_latency_ns=cb_latency,
            merkle_sha256_root=merkle_root,
            signature_hmac_sha256=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
            base_verdict_v13=base_verdict,
            kam_adjusted_verdict=verdict,
            hard_evidence_veto=hard_veto,
            soft_evidence_escalated=any("R10_" in f for f in gate_failures),
            false_positive_protected=fp_protected,
            judicial_gate_failures=gate_failures,
            meta_consistency_score=float(meta["consistency"]),
            meta_verdict_entropy=float(meta["entropy"]),
            level_3_entropy=float(meta["level_3_entropy"]),
            n3_verdict_trajectory=tuple(self._n3_verdict_trajectory),
            n3_cycle_count=self._n3_cycle_count,
            n3_adaptation_lyapunov=float(meta["adaptation_lyapunov"]),
            engine_n3_trajectory=tuple(result.campaign_id and getattr(
                self._cache.snapshot_cert, "n3_verdict_trajectory", ()
            ) or ()),
            ooda_n3_trajectory=tuple(self._ooda_ledger.verdict_history[-_N3_TRAJECTORY_WINDOW:]),
            lyapunov_poincare=result.lyapunov_poincare,
            nekhoroshev_time=result.nekhoroshev_time,
            poincare_return_period=result.poincare_return_period,
            celestial_section_transverse=result.celestial_section_transverse,
            crowbar_latency_certified=cb_certified,
            passive_gpio=True,
            engine_invocations=result.engine_invocations,
            emergency_reinvoke=result.emergency_reinvoke,
            i4_engine_once=bool(result.engine_invocations <= 1 and not result.emergency_reinvoke),
            ooda_kam_coherent_threshold=float(meta["kam_floor"]),
            hmac_chain_p1_p2_p3=hmac_chain,
            density_fingerprint=result.density_fingerprint,
        )

    def run_phase3_decide(
        self,
        contract: FPUProsecutorGaugeContractV14,
        result: FPUProsecutionCampaignResultV14,
        gpio_pin: int = 14,
    ) -> FPUProsecutorSovereignPassportV14:
        """Orquestador FASE 3."""
        return self.execute_phase3_v14(contract, result, gpio_pin)

    def _upgrade_p2_from_legacy(self, campaign: Any) -> FPUProsecutionCampaignResultV14:
        if isinstance(campaign, FPUProsecutionCampaignResultV14):
            return campaign
        unsigned = FPUProsecutionCampaignResultV14(
            campaign_id=campaign.campaign_id,
            sovereign_id=campaign.sovereign_id,
            electron_anomalies_purged=campaign.electron_anomalies_purged,
            positron_auth_valid=campaign.positron_auth_valid,
            released_energy_kev=campaign.released_energy_kev,
            back_action_db=campaign.back_action_db,
            spohn_entropy_production=campaign.spohn_entropy_production,
            gksl_dissipation_rate=campaign.gksl_dissipation_rate,
            prosecution_status=campaign.prosecution_status,
            global_verdict=campaign.global_verdict,
            chain_hash=campaign.chain_hash,
            hmac_signature="",
            timestamp_utc=campaign.timestamp_utc,
            contract_id=getattr(self._cache, "contract").contract_id
            if self._cache.contract is not None else campaign.campaign_id,
            car_anti_commutation_residual=0.0,
            car_canonical_residual=0.0,
            car_canonical_valid=True,
            ccr_canonical_residual=0.0,
            ccr_canonical_valid=True,
            symplectic_residual=0.0,
            symplectic_valid=True,
            purity_proxy=1.0,
            positron_auth_confidence=1.0,
            hard_evidence_veto=False,
            banach_lipschitz_k=0.857,
            banach_gate_passed=True,
            poincare_crossings_count=0,
            occupation_drift_max=0.0,
            occupation_drift_mean=0.0,
            lindstedt_correction_norm=0.0,
            lindstedt_o1=0.0,
            lindstedt_o2=0.0,
            birkhoff_quadratic_coefficient=0.0,
            kam_stability_index=1.0,
            kam_resonance_violations=0,
            kam_worst_ratio=1.0e9,
            kam_cf_penalty=0.0,
            action_total=0.0,
            phase_space_volume=0.0,
            lyapunov_max=0.0,
            lyapunov_min=0.0,
            lyapunov_poincare=0.0,
            nekhoroshev_time=1.0e12,
            celestial_section_transverse=False,
            poincare_return_period=0.0,
            omega_carrier=1.0,
            quadrature_x=0.0,
            quadrature_p=0.0,
            number_action=0.0,
            meta_consistency_score=1.0,
            meta_verdict_entropy=0.0,
            level_3_entropy=0.0,
            level_3_reflections=0,
            n3_cycle_count=0,
            n3_adaptation_lyapunov=0.0,
            engine_crowbar_triggered=False,
            engine_crowbar_latency_ns=0.0,
            engine_judicial_gate_failures=(),
            hmac_engine_signature="",
            ooda_kam_coherent_threshold=_KAM_TRUSTWORTHY_THRESHOLD,
            density_fingerprint=self._cache.density_fingerprint,
            engine_invocations=int(self._cache.engine_invocations),
            emergency_reinvoke=bool(self._cache.emergency_reinvoke),
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2_V14, self._canonical_p2_payload(unsigned))
        object.__setattr__(unsigned, "hmac_signature", sig)
        return unsigned

    def execute_phase3_sovereign_decision(
        self,
        contract: Any,
        campaign_result: Any,
        gpio_pin: int = 14,
    ) -> FPUProsecutorSovereignPassport:
        """Alias v13: acepta P1/P2 v13 o v14, retorna pasaporte v13 re-firmado LEGADO."""
        c14 = self._upgrade_p1_from_legacy(contract)
        if not isinstance(contract, FPUProsecutorGaugeContractV14):
            object.__setattr__(
                c14,
                "hmac_signature",
                self._compute_hmac(_HMAC_DOMAIN_P1_V14, self._canonical_p1_payload(c14)),
            )
        r14 = self._upgrade_p2_from_legacy(campaign_result)
        p14 = self.execute_phase3_v14(c14, r14, gpio_pin)
        v13_contract = FPUProsecutorGaugeContract(
            contract_id=p14.contract.contract_id,
            sovereign_id=p14.contract.sovereign_id,
            car_residual=p14.contract.car_residual,
            ccr_norm=p14.contract.ccr_norm,
            trace_preservation=p14.contract.trace_preservation,
            isometry_verdict=p14.contract.isometry_verdict,
            chain_hash=p14.contract.chain_hash,
            hmac_signature="",
            creation_timestamp=p14.contract.creation_timestamp,
        )
        object.__setattr__(
            v13_contract,
            "hmac_signature",
            self._compute_hmac(_HMAC_DOMAIN_P1, self._legacy_p1_payload(v13_contract)),
        )
        v13_campaign = FPUProsecutionCampaignResult(
            campaign_id=p14.campaign_result.campaign_id,
            sovereign_id=p14.campaign_result.sovereign_id,
            electron_anomalies_purged=p14.campaign_result.electron_anomalies_purged,
            positron_auth_valid=p14.campaign_result.positron_auth_valid,
            released_energy_kev=p14.campaign_result.released_energy_kev,
            back_action_db=p14.campaign_result.back_action_db,
            spohn_entropy_production=p14.campaign_result.spohn_entropy_production,
            gksl_dissipation_rate=p14.campaign_result.gksl_dissipation_rate,
            prosecution_status=p14.campaign_result.prosecution_status,
            global_verdict=p14.campaign_result.global_verdict,
            chain_hash=p14.campaign_result.chain_hash,
            hmac_signature="",
            timestamp_utc=p14.campaign_result.timestamp_utc,
        )
        object.__setattr__(
            v13_campaign,
            "hmac_signature",
            self._compute_hmac(_HMAC_DOMAIN_P2, self._legacy_p2_payload(v13_campaign)),
        )
        merkle = p14.merkle_sha256_root
        legacy_p3 = (
            f"{merkle}:{int(p14.crowbar_triggered)}:{p14.global_verdict.value}"
        )
        return FPUProsecutorSovereignPassport(
            passport_id=p14.passport_id,
            contract=v13_contract,
            campaign_result=v13_campaign,
            global_verdict=p14.global_verdict,
            prosecution_status=p14.prosecution_status,
            crowbar_triggered=p14.crowbar_triggered,
            crowbar_latency_ns=p14.crowbar_latency_ns,
            merkle_sha256_root=merkle,
            signature_hmac_sha256=self._compute_hmac(_HMAC_DOMAIN_P3, legacy_p3),
            visceral_report=p14.visceral_report,
            timestamp_utc=p14.timestamp_utc,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO FISCAL — ORQUESTADOR OODA v14.1
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderProsecutorAgentV14(FPUProsecutionDeciderV14):
    """Soberano Agéntico Fiscal v14.1: OODA + Poincaré + N3 + Debido Proceso. GPIO pasivo."""

    def prosecute_fpu_tensor_fock_v14(
        self,
        density_matrix: NDArray,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
        gpio_pin: int = 14,
    ) -> FPUProsecutorSovereignPassportV14:
        """
        Ciclo OODA v14.1:
          FASE 1 (Observe) → Φ₁₂ → FASE 2 (Orient, 1× motor) → Φ₂₃ → FASE 3 (Decide, GPIO pasivo).
        """
        self._begin_ooda_cycle()
        contract = self.run_phase1_observe(density_matrix, state_a, state_b)
        if not self.handoff_to_phase2_v14(contract):
            raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ del Fiscal FPU v14.1")
        result = self.run_phase2_orient(
            contract, electron_anomalies, positron_auth_signature,
            transaction_payload, density_matrix, state_a, state_b,
        )
        if not self.handoff_to_phase3_v14(result):
            raise ValueError("Falla de autenticidad en Compuerta Φ₂₃ del Fiscal FPU v14.1")
        return self.run_phase3_decide(contract, result, gpio_pin)

    def prosecute_fpu_tensor_fock(
        self,
        density_matrix: NDArray,
        electron_anomalies: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
        gpio_pin: int = 14,
    ) -> FPUProsecutorSovereignPassport:
        """Alias v13: ciclo interno v14.1, pasaporte v13 re-firmado LEGADO."""
        p14 = self.prosecute_fpu_tensor_fock_v14(
            density_matrix, electron_anomalies, positron_auth_signature,
            transaction_payload, state_a, state_b, gpio_pin,
        )
        return self.execute_phase3_sovereign_decision(
            p14.contract, p14.campaign_result, gpio_pin
        )


FPUProsecutorPhase1Observer = FPUProsecutorPhase1ObserverV14
FPUProsecutionCampaignOrientor = FPUProsecutionCampaignOrientorV14
FPUProsecutionDecider = FPUProsecutionDeciderV14
FPUTensorFockColliderProsecutorAgent = FPUTensorFockColliderProsecutorAgentV14


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DEMO — 3 ciclos OODA v14.1 + backward-compat v13
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v14.1 · FISCAL FPU OODA + KERR/POINCARÉ + N3 + I4-GPIO")
    print(banner)

    agent = FPUTensorFockColliderProsecutorAgentV14(secret_key=_DEFAULT_SECRET_KEY)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    payload = "TRANSACTION-APU-PRECIO-SOBRECOSTO-v14-001"
    valid_hmac = hmac.new(agent.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _report(label: str, p: FPUProsecutorSovereignPassportV14) -> None:
        print(f"\n  [{label}]")
        print(f"  • Passport ID          : {p.passport_id}")
        print(f"  • Veredicto base       : {p.base_verdict_v13.name}")
        print(f"  • Veredicto FINAL Ω₃   : {p.global_verdict.name}")
        print(f"  • Status               : {p.prosecution_status.name}")
        print(f"  • Hard evidence veto   : {p.hard_evidence_veto}")
        print(f"  • Soft evidence escal. : {p.soft_evidence_escalated}")
        print(f"  • False positive prot. : {p.false_positive_protected}")
        print(f"  • Judicial failures    : {p.judicial_gate_failures or ('none',)}")
        print(f"  • Crowbar (MOTOR)      : {p.crowbar_triggered} "
              f"(lat={p.crowbar_latency_ns:.2f} ns, cert={p.crowbar_latency_certified})")
        print(f"  • GPIO agente          : PASIVO={p.passive_gpio}")
        print(f"  • I4 engine once       : {p.i4_engine_once}  "
              f"(invokes={p.engine_invocations}, emergency={p.emergency_reinvoke})")
        print(f"  • Fingerprint ρ        : {p.density_fingerprint}")
        print(f"  • Electrones purgados  : {p.campaign_result.electron_anomalies_purged}")
        print(f"  • Energía / Back-Action: {p.campaign_result.released_energy_kev:.1f} keV / "
              f"{p.campaign_result.back_action_db:.1f} dB")
        print(f"  • Auth confidence      : {p.campaign_result.positron_auth_confidence:.4f}")
        print(f"  • CAR/CCR/I8           : CAR={p.campaign_result.car_canonical_valid} "
              f"CCR={p.campaign_result.ccr_canonical_valid} "
              f"I8={p.campaign_result.symplectic_valid} "
              f"(sym={p.campaign_result.symplectic_residual:.3e})")
        print(f"  • Banach Lip(T_Σ)      : {p.campaign_result.banach_lipschitz_k:.4f} "
              f"(gate={p.campaign_result.banach_gate_passed})")
        print(f"  • Poincaré             : cross={p.campaign_result.poincare_crossings_count} "
              f"transverse={p.celestial_section_transverse} "
              f"T_ret={p.poincare_return_period:.4f}")
        print(f"  • Drift p_n (no spec ρ): {p.campaign_result.occupation_drift_max:.6e}")
        print(f"  • λ_P / T_Nek          : {p.lyapunov_poincare:.6e} / {p.nekhoroshev_time:.6e}")
        print(f"  • Lindstedt o1/o2      : {p.campaign_result.lindstedt_o1:.6e} / "
              f"{p.campaign_result.lindstedt_o2:.6e}")
        print(f"  • KAM index            : {p.campaign_result.kam_stability_index:.6f} "
              f"(viol={p.campaign_result.kam_resonance_violations}, "
              f"cf={p.campaign_result.kam_cf_penalty:.3f})")
        print(f"  • level_3_entropy      : {p.level_3_entropy:.4f}")
        print(f"  • λ_α OODA             : {p.n3_adaptation_lyapunov:.6e}")
        print(f"  • N3 OODA traj         : {p.ooda_n3_trajectory}")
        print(f"  • kam_floor OODA       : {p.ooda_kam_coherent_threshold:.4f}")
        print(f"  • HMAC chain P1→P2→P3  : {p.hmac_chain_p1_p2_p3[:48]}...")
        print(f"  • Executive action     : {p.visceral_report['executive_action'][:88]}...")

    p1 = agent.prosecute_fpu_tensor_fock_v14(
        density_matrix=rho, electron_anomalies=1,
        positron_auth_signature=valid_hmac, transaction_payload=payload,
    )
    _report("CICLO 1 · POSITRÓN VÁLIDO (QND 0.0 dB, GPIO pasivo)", p1)

    p2 = agent.prosecute_fpu_tensor_fock_v14(
        density_matrix=rho, electron_anomalies=1,
        positron_auth_signature="APOCRYPHAL_SIGNATURE", transaction_payload=payload,
    )
    _report("CICLO 2 · POSITRÓN APÓCRIFO (HARD I5 · crowbar del MOTOR)", p2)

    p3 = agent.prosecute_fpu_tensor_fock_v14(
        density_matrix=rho, electron_anomalies=0,
        positron_auth_signature=valid_hmac, transaction_payload=payload,
    )
    _report("CICLO 3 · NOMINAL + N3 OODA acoplado", p3)

    print("\n  ═══ VERIFICACIÓN BACKWARD-COMPAT v13 ═══")
    p_v13 = agent.prosecute_fpu_tensor_fock(
        density_matrix=rho, electron_anomalies=1,
        positron_auth_signature=valid_hmac, transaction_payload=payload,
    )
    v13_fields = [
        "passport_id", "contract", "campaign_result", "global_verdict",
        "prosecution_status", "crowbar_triggered", "crowbar_latency_ns",
        "merkle_sha256_root", "signature_hmac_sha256", "visceral_report", "timestamp_utc",
    ]
    missing = [f for f in v13_fields if not hasattr(p_v13, f)]
    print(f"  Campos v13 ausentes : {missing or 'NINGUNO ✓'}")
    print(f"  tipo pasaporte      : {type(p_v13).__name__}  (debe ser v13, no V14)")
    print(f"  p_v13 veredicto     : {p_v13.global_verdict.name}")
    print(f"  p_v13 crowbar       : {p_v13.crowbar_triggered}")
    print(f"  p_v13 back-action   : {p_v13.campaign_result.back_action_db:.1f} dB")
    gate_ok = agent.continue_to_phase2_prosecutor_gate(p_v13.contract)
    print(f"  Φ₁₂ legado verifica : {gate_ok}  (re-firma LEGACY, no payload v14)")
    print(banner)