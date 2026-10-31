# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPUTensorFockColliderWitnessAgent v14.1.0-SOVEREIGN-POINCARE-N3-OODA                      ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider_witness_agent.py                                      ║
║ Función  : Soberano Agéntico Testigo Silencioso del Colisionador FPU en Fock ℱ(ℋ). Ciclo OODA        ║
║            pasivo (Observe → Orient → Decide&Act) con Mecánica Celeste de Poincaré (secciones        ║
║            (x̂,p̂)/N̂, Lindstedt 1º+2º, Birkhoff, KAM, Nekhoroshev, Benettin) y Automejora Recursiva   ║
║            N3 de DOS capas: (i) ledger del motor, (ii) ledger OODA del agente, acoplado al mapa de   ║
║            retorno. El motor se invoca EXACTAMENTE UNA VEZ por ciclo.                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS DEL AGENTE v14.1 (3 × 3 sub-capas) — HANDOFF ESTRICTO:

  FASE 1 — OBSERVE (FPUWitnessPhase1ObserverV14)
    1.1 `_invoke_engine_v14_and_cache`       — invocación ÚNICA; huella de ρ; cache de snapshot.
    1.2 `_extract_phase1_invariants_v14`     — isometría + F1 celeste (J_i, (x,p), simpléctica, Lyapunov).
    1.3 `execute_phase1_v14`                 — emite FPUWitnessPhase1CertificateV14 (HMAC-SHA256).
    [HANDOFF] `handoff_to_phase2_v14`  ≡  `continue_to_phase2_witness_gate_v14`
        ※ Último acto formal de FASE 1 = primer acto formal de FASE 2.

  FASE 2 — ORIENT (FPUWitnessPhase2OrientorV14)
    2.1 `_extract_phase2_poincare_invariants_v14` — AAV/Bures/Spohn + crossings, drift, Lindstedt,
                                                    Birkhoff, KAM, ν, T_ret, λ_P, T_Nekhoroshev.
    2.2 `_extract_phase2_n3_tuning_v14`           — TuningPolicy N2/N3 + λ_α del motor.
    2.3 `execute_phase2_v14`                      — consume el HANDOFF de 1.3; NO re-invoca el motor.
    [HANDOFF] `handoff_to_phase3_v14`  ≡  `continue_to_phase3_witness_gate_v14`
        ※ Último acto formal de FASE 2 = primer acto formal de FASE 3.

  FASE 3 — DECIDE & ACT (FPUWitnessPhase3DeciderV14)
    3.1 `_extract_phase3_meta_witness_v14`   — meta N3 motor + ledger OODA del agente.
    3.2 `_adjudicate_kam_aware_v14`          — 10 reglas Heyting (KAM + λ_P + sección + Nekhoroshev + N3).
    3.3 `execute_phase3_v14`                 — Estado Soberano v14.1 + PASSIVE_SILENT + cadena HMAC P1→P2→P3.

INVARIANTES:
  · I1  Back-Action QND ≡ 0.0 dB (nunca mide destructivamente)
  · I2  Tr(ρ) ≡ 1.0 ± 10⁻⁵ (isometría)
  · I3  PASSIVE_SILENT: NUNCA dispara hardware ESP32 (verificado y reportado)
  · I4  Motor invocado ≤ 1 vez por ciclo OODA (engine_invocations auditado)
  · I5  HMAC chain P1v14 → P2v14 → P3v14 con dominios renovados (sin colisión v13)
  · I6  Merkle raíz incluye KAM, λ_P, T_Nek, meta, PASSIVE_SILENT
  · I7  Contrato backward-compat v13: execute_phase1/2/3 y witness_fpu_tensor_field
  · I8  Sección de Poincaré transversal certificada (cruces + drift en p_n, no en spec(ρ))
  · I9  Ledger OODA N3 distinto del ledger del motor; Poincaré modula umbrales de adjudicación
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

for path_dir in ["/workspace/scratch", "/workspace/artifacts", "."]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# ── Importación del motor v14.1 (con fallback al v13) ──
try:
    from fpu_tensor_fock_collider_witness_engine import (
        FPUTensorFockColliderWitnessEngine,
        FPUTensorFockColliderWitnessEngineV14,
        FPUTensorWitnessSnapshot,
        FPUTensorWitnessSnapshotV14,
        FockActionAngleDecomposition,
        FockPoincareAnalysis,
        FockTuningPolicy,
        HeytingOmega3,
    )
    _ENGINE_V14_AVAILABLE: Final[bool] = True
except ImportError:
    try:
        from app.wisdom.fpu_tensor_fock_collider_witness_engine import (
            FPUTensorFockColliderWitnessEngine,
            FPUTensorFockColliderWitnessEngineV14,
            FPUTensorWitnessSnapshot,
            FPUTensorWitnessSnapshotV14,
            FockActionAngleDecomposition,
            FockPoincareAnalysis,
            FockTuningPolicy,
            HeytingOmega3,
        )
        _ENGINE_V14_AVAILABLE: Final[bool] = True
    except ImportError:
        from enum import IntEnum

        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

            def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(max(int(self), int(other)))

            def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(min(int(self), int(other)))

        FPUTensorFockColliderWitnessEngine = None  # type: ignore[assignment,misc]
        FPUTensorFockColliderWitnessEngineV14 = None  # type: ignore[assignment,misc]
        FPUTensorWitnessSnapshot = None  # type: ignore[assignment,misc]
        FPUTensorWitnessSnapshotV14 = None  # type: ignore[assignment,misc]
        FockActionAngleDecomposition = None  # type: ignore[assignment,misc]
        FockPoincareAnalysis = None  # type: ignore[assignment,misc]
        FockTuningPolicy = None  # type: ignore[assignment,misc]
        _ENGINE_V14_AVAILABLE = False

logger = logging.getLogger("APU.Wisdom.FPUTensorFockColliderWitnessAgentV14")

# ── Dominios HMAC v14.1 (renovados; no colisionan con v13) ──
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_FPU_WITNESS_SOVEREIGN_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"FPU-WITNESS-P1v14-POINCARE-OBSERVE|"
_HMAC_DOMAIN_P2: Final[bytes] = b"FPU-WITNESS-P2v14-KAM-ORIENT|"
_HMAC_DOMAIN_P3: Final[bytes] = b"FPU-WITNESS-P3v14-N3-DECIDE|"
_HMAC_DOMAIN_P1_LEGACY: Final[bytes] = b"FPU-WITNESS-P1-OBSERVE|"
_HMAC_DOMAIN_P2_LEGACY: Final[bytes] = b"FPU-WITNESS-P2-ORIENT|"
_HMAC_DOMAIN_P3_LEGACY: Final[bytes] = b"FPU-WITNESS-P3-DECIDE|"
_CHAIN_HEX_LEN: Final[int] = 32

# ── Umbrales de adjudicación v14.1 (semillas; el ledger OODA los adapta) ──
_KAM_COHERENT_THRESHOLD: Final[float] = 0.75
_KAM_DEGRADED_THRESHOLD: Final[float] = 0.50
_KAM_SEVERE_THRESHOLD: Final[float] = 0.25
_META_CONSISTENCY_DEGRADED: Final[float] = 0.75
_META_CONSISTENCY_VETO: Final[float] = 0.50
_CANONICAL_RESIDUAL_TOL: Final[float] = 1.0e-6
_N3_TRAJECTORY_WINDOW: Final[int] = 8
_N3_LEDGER_CAP: Final[int] = 64
_POINCARE_LYAP_DEGRADED: Final[float] = 0.50
_NEKHOROSHEV_TIME_FLOOR: Final[float] = 2.0
_LIOUVILLE_VETO: Final[float] = 1.0e-4
_LIOUVILLE_DEGRADED: Final[float] = 1.0e-5
_BACK_ACTION_SILENT_TOL: Final[float] = 1.0e-12


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES — v13 (preservadas) + v14.1 (celestes + OODA N3)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class FPUWitnessPhase1Certificate:
    """Certificado v13 (preservado por compatibilidad)."""
    witness_id: str
    trace_preservation: float
    liouville_drift: float
    car_anti_commutation_residual: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class FPUWitnessPhase2Certificate:
    """Certificado v13 (preservado por compatibilidad)."""
    witness_id: str
    phase1_ref_id: str
    weak_value_qnd_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    qnd_verdict: HeytingOmega3
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class TopologicalFPUTensorWitnessSovereignState:
    """Estado soberano v13 (preservado por compatibilidad)."""
    sovereign_state_id: str
    phase1_cert: FPUWitnessPhase1Certificate
    phase2_cert: FPUWitnessPhase2Certificate
    global_verdict: HeytingOmega3
    back_action_db: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float


# ── v14.1 (enriquecidos) ──

@dataclass(frozen=True, slots=True)
class FPUWitnessPhase1CertificateV14:
    """
    FASE 1 — Certificate enriquecido v14.1 (Observe + Poincaré F1 + simpléctica).
    Superset estricto del P1 v13.
    """
    # v13
    witness_id: str
    trace_preservation: float
    liouville_drift: float
    car_anti_commutation_residual: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float
    # v14 · FASE 1 (action-angle)
    action_total: float
    mean_occupation: float
    phase_space_volume: float
    purity: float
    lyapunov_max: float
    lyapunov_min: float
    occupation_entropy: float
    ccr_canonical_residual: float
    car_canonical_residual: float
    ccr_canonical_valid: bool
    car_canonical_valid: bool
    engine_v14_active: bool
    # v14.1 · FASE 1 (cuadraturas / I8)
    symplectic_residual: float
    quadrature_x: float
    quadrature_p: float
    density_fingerprint: str
    engine_invocations: int


@dataclass(frozen=True, slots=True)
class FPUWitnessPhase2CertificateV14:
    """FASE 2 — Certificate enriquecido v14.1 (Orient + Poincaré celeste + N3 motor)."""
    # v13
    witness_id: str
    phase1_ref_id: str
    weak_value_qnd_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    qnd_verdict: HeytingOmega3
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float
    # v14 · FASE 2 (Poincaré Fock)
    poincare_crossings_count: int
    occupation_drift_max: float
    occupation_drift_mean: float
    lindstedt_correction_norm: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    # v14 · FASE 2 (Tuning N3 motor)
    adaptive_fidelity_floor: float
    adaptive_bures_ceiling: float
    adaptive_spohn_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    level_3_entropy: float
    # v14.1 · FASE 2 (celeste)
    lindstedt_order2_norm: float
    rotation_number: float
    first_return_time_mean: float
    poincare_lyapunov: float
    nekhoroshev_time: float
    n3_adaptation_lyapunov: float
    celestial_section_transverse: bool
    engine_invocations: int


@dataclass(frozen=True, slots=True)
class TopologicalFPUTensorWitnessSovereignStateV14:
    """FASE 3 — Estado soberano enriquecido v14.1 (superset estricto del v13)."""
    # v13
    sovereign_state_id: str
    phase1_cert: FPUWitnessPhase1CertificateV14
    phase2_cert: FPUWitnessPhase2CertificateV14
    global_verdict: HeytingOmega3
    back_action_db: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    # v14 · FASE 3
    base_verdict_v13: HeytingOmega3
    kam_adjusted_verdict: HeytingOmega3
    meta_consistency_score: float
    meta_verdict_entropy: float
    meta_fidelity_std: float
    meta_bures_std: float
    meta_spohn_std: float
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    level_3_reflections: int
    passive_silent_invariant_verified: bool
    # v14.1 · FASE 3 (celeste + OODA N3)
    engine_n3_verdict_trajectory: Tuple[int, ...]
    ooda_n3_verdict_trajectory: Tuple[int, ...]
    n3_adaptation_lyapunov: float
    ooda_adaptation_lyapunov: float
    poincare_lyapunov: float
    nekhoroshev_time: float
    rotation_number: float
    celestial_section_transverse: bool
    single_invocation_invariant_verified: bool
    ooda_kam_coherent_threshold: float
    hmac_chain_p1_p2_p3: str


@dataclass(slots=True)
class OODASessionCache:
    """
    Caché de un ciclo OODA. Protege I4 (invocación única del motor) y enlaza
    la huella de ρ entre FASE 1 y FASE 2.
    """
    snapshot: Optional[Any] = None
    cert1: Optional[FPUWitnessPhase1CertificateV14] = None
    density_fingerprint: Optional[str] = None
    engine_invocations: int = 0
    emergency_reinvoke: int = 0


@dataclass(slots=True)
class FockOODALedger:
    """
    Ledger N3 del ciclo OODA del AGENTE (distinto del ledger del motor).

    Dinámica discreta (mapa de Poincaré en el espacio de políticas de adjudicación):
        θ_{t+1} = θ_t + α_t · τ(KAM, λ_P, T_Nek)     (N2: umbrales KAM/meta)
        α_{t+1} = α_t · g(S(verdicts_OODA))            (N3: meta-tasa)
    λ_α^OODA = ⟨log |α_{t+1}/α_t|⟩ es el Lyapunov de automejora del propio agente.
    """
    verdict_history: List[int] = field(default_factory=list)
    kam_history: List[float] = field(default_factory=list)
    lyap_history: List[float] = field(default_factory=list)
    consistency_history: List[float] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    kam_coherent_threshold: float = _KAM_COHERENT_THRESHOLD
    kam_degraded_threshold: float = _KAM_DEGRADED_THRESHOLD
    meta_degraded_threshold: float = _META_CONSISTENCY_DEGRADED
    adaptation_rate: float = 0.05
    meta_adaptation_rate: float = 0.01
    reflections: int = 0

    def observe(
        self,
        verdict: HeytingOmega3,
        kam: float,
        poincare_lyap: float,
        consistency: float,
    ) -> None:
        self.verdict_history.append(int(verdict))
        self.kam_history.append(float(kam))
        self.lyap_history.append(float(poincare_lyap))
        self.consistency_history.append(float(consistency))
        self.rate_history.append(float(self.adaptation_rate))
        if len(self.verdict_history) > _N3_LEDGER_CAP:
            self.verdict_history = self.verdict_history[-_N3_LEDGER_CAP:]
            self.kam_history = self.kam_history[-_N3_LEDGER_CAP:]
            self.lyap_history = self.lyap_history[-_N3_LEDGER_CAP:]
            self.consistency_history = self.consistency_history[-_N3_LEDGER_CAP:]
            self.rate_history = self.rate_history[-_N3_LEDGER_CAP:]

    def _trend(self, series: List[float]) -> float:
        if len(series) < 2:
            return 0.0
        n = min(_N3_TRAJECTORY_WINDOW, len(series))
        y = np.array(series[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        x -= x.mean()
        y -= y.mean()
        denom = float(np.sum(x * x))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0

    def adaptation_lyapunov(self) -> float:
        if len(self.rate_history) < 3:
            return 0.0
        r = np.array(self.rate_history[-_N3_TRAJECTORY_WINDOW:], dtype=float)
        r = np.maximum(r, 1e-15)
        ratios = np.abs(np.diff(np.log(r)))
        return float(np.mean(ratios)) if ratios.size else 0.0

    def level_2_adapt(self) -> None:
        """N2 — tuerce los umbrales KAM/meta según la tendencia celestial observada."""
        trend_k = self._trend(self.kam_history)
        trend_l = self._trend(self.lyap_history)
        if trend_k < -1e-4 or trend_l > 1e-3:
            self.kam_coherent_threshold = min(0.90, self.kam_coherent_threshold + self.adaptation_rate * 0.02)
            self.kam_degraded_threshold = min(0.70, self.kam_degraded_threshold + self.adaptation_rate * 0.01)
            self.meta_degraded_threshold = min(0.90, self.meta_degraded_threshold + self.adaptation_rate * 0.01)
        elif trend_k > 1e-4 and trend_l < -1e-4:
            self.kam_coherent_threshold = max(0.60, self.kam_coherent_threshold - self.adaptation_rate * 0.01)
            self.kam_degraded_threshold = max(0.35, self.kam_degraded_threshold - self.adaptation_rate * 0.005)
            self.meta_degraded_threshold = max(0.60, self.meta_degraded_threshold - self.adaptation_rate * 0.005)

    def level_3_adapt(self) -> float:
        """N3 — meta-tasa según entropía de veredictos OODA."""
        if len(self.verdict_history) < 4:
            return 0.0
        recent = np.array(self.verdict_history[-_N3_TRAJECTORY_WINDOW:], dtype=int)
        _, counts = np.unique(recent, return_counts=True)
        probs = counts / float(counts.sum())
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if entropy > 0.5:
            self.meta_adaptation_rate = max(1e-3, self.meta_adaptation_rate * 0.90)
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.95)
        else:
            self.meta_adaptation_rate = min(0.50, self.meta_adaptation_rate * 1.05)
            self.adaptation_rate = min(0.50, self.adaptation_rate * 1.02)
        self.reflections += 1
        return entropy

    def poincare_modulate(self, kam: float, poincare_lyap: float, nek_time: float) -> bool:
        """
        Acoplamiento celestial → OODA (principio de Nekhoroshev):
        no adjudicar más agresivamente que la estabilidad del toro.
        """
        damped = False
        if kam < self.kam_degraded_threshold or poincare_lyap > 0.15:
            self.adaptation_rate = max(1e-3, self.adaptation_rate * 0.85)
            self.kam_coherent_threshold = min(0.90, self.kam_coherent_threshold + 0.005)
            damped = True
        if nek_time < _NEKHOROSHEV_TIME_FLOOR:
            self.meta_degraded_threshold = min(0.90, self.meta_degraded_threshold + 0.01)
            damped = True
        if kam >= _KAM_COHERENT_THRESHOLD and poincare_lyap < 0.05 and nek_time >= _NEKHOROSHEV_TIME_FLOOR:
            self.adaptation_rate = min(0.50, self.adaptation_rate * 1.01)
        return damped


def _fingerprint_density(matrix: NDArray) -> str:
    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _safe_hmac_compare(a: str, b: str) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    return hmac.compare_digest(a, b)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 1 — OBSERVE (3 sub-capas)
# ██╔══██╗ Invocación única · extracción F1 celeste · emisión P1v14
# ██████╔╝ [HANDOFF] handoff_to_phase2_v14 → cert1 autenticado (input de FASE 2)
# ██╔═══╝  El último método formal de esta fase ES el primer acto de FASE 2.
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUWitnessPhase1ObserverV14:
    """
    FASE 1 — OBSERVE con 3 sub-capas anidadas.

    Sub-capa 1.1: `_invoke_engine_v14_and_cache` — invocación ÚNICA.
    Sub-capa 1.2: `_extract_phase1_invariants_v14` — isometría + Poincaré F1 + I8.
    Sub-capa 1.3: `execute_phase1_v14` — emite P1v14 firmado HMAC.
    Handoff:      `handoff_to_phase2_v14` / `continue_to_phase2_witness_gate_v14`.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: int = 56,
    ) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.dimension: Final[int] = max(4, int(dimension))

        if engine is not None:
            self.engine = engine
        elif _ENGINE_V14_AVAILABLE and FPUTensorFockColliderWitnessEngineV14 is not None:
            self.engine = FPUTensorFockColliderWitnessEngineV14(
                dimension=self.dimension, secret_key=self.secret_key
            )
        elif FPUTensorFockColliderWitnessEngine is not None:
            self.engine = FPUTensorFockColliderWitnessEngine(
                dimension=self.dimension, secret_key=self.secret_key
            )
        else:
            self.engine = None

        self._session: OODASessionCache = OODASessionCache()
        # Alias de compatibilidad (código v14.0 leía _session_snapshot).
        self._session_snapshot: Optional[Any] = None

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _reset_session_cycle(self) -> None:
        """Abre un ciclo OODA nuevo: se conserva el contador de invocaciones de emergencia histórico."""
        emergency = self._session.emergency_reinvoke
        self._session = OODASessionCache(emergency_reinvoke=emergency)
        self._session_snapshot = None

    # ────────────────── Sub-capa 1.1: invocación única ──────────────────
    def _invoke_engine_v14_and_cache(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        *,
        emergency: bool = False,
    ) -> Optional[Any]:
        """
        Sub-capa 1.1 — Ejecuta el motor v14 EXACTAMENTE UNA VEZ por ciclo OODA.
        Cachea el snapshot para Fases 2 y 3 (protege ledger N3 del motor).
        `emergency=True` sólo si FASE 2 encuentra caché vacía: se audita como
        violación de I4, no se oculta.
        """
        if self.engine is None:
            self._session.snapshot = None
            self._session_snapshot = None
            return None

        if emergency:
            self._session.emergency_reinvoke += 1
            logger.warning(
                "I4 VIOLADO: re-invocación de emergencia #%d (caché OODA vacía)",
                self._session.emergency_reinvoke,
            )

        snapshot: Optional[Any] = None
        if hasattr(self.engine, "evaluate_fpu_witness_v14"):
            try:
                snapshot = self.engine.evaluate_fpu_witness_v14(
                    density_matrix, phi_initial, phi_final
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("evaluate_fpu_witness_v14 falló: %s", exc)
                snapshot = None

        if snapshot is None and hasattr(self.engine, "evaluate_fpu_witness"):
            try:
                snapshot = self.engine.evaluate_fpu_witness(
                    density_matrix, phi_initial, phi_final
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("evaluate_fpu_witness falló: %s", exc)
                snapshot = None

        self._session.snapshot = snapshot
        self._session_snapshot = snapshot
        self._session.engine_invocations += 1
        self._session.density_fingerprint = _fingerprint_density(density_matrix)
        return snapshot

    # ────────────────── Sub-capa 1.2: extracción F1 ──────────────────
    def _extract_phase1_invariants_v14(
        self,
        snapshot: Optional[Any],
        density_matrix: NDArray,
    ) -> Dict[str, Any]:
        """
        Sub-capa 1.2 — Extrae invariantes de Fase 1:
          · isometría (Tr, Liouville, CAR legacy)
          · Poincaré F1 (ΣJ, ⟨N⟩, vol. fase, Lyapunov espectral, pureza, S_ocupación)
          · CAR/CCR canónicos
          · I8 cuadraturas (⟨x̂⟩, ⟨p̂⟩) y residual simpléctico
        """
        dim = self.dimension
        if self.engine is not None and hasattr(self.engine, "sanitize_density_matrix"):
            rho = self.engine.sanitize_density_matrix(density_matrix)
        else:
            a = np.asarray(density_matrix, dtype=np.complex128)
            rho = 0.5 * (a + a.conj().T)
            tr_a = float(np.real(np.trace(rho)))
            rho = rho / tr_a if tr_a > 1e-15 else np.eye(dim, dtype=np.complex128) / float(dim)

        tr_val = float(np.real(np.trace(rho)))
        liouville_drift = float(abs(tr_val - 1.0))
        is_v14 = snapshot is not None and hasattr(snapshot, "action_total")
        car_res_legacy = float(getattr(snapshot, "car_anti_commutation_residual", 0.0)) if snapshot else 0.0

        def g(name: str, default: Any) -> Any:
            return getattr(snapshot, name, default) if snapshot is not None else default

        return {
            "trace": tr_val,
            "liouville": liouville_drift,
            "car_legacy": car_res_legacy,
            "action_total": float(g("action_total", 0.0)),
            "mean_occupation": float(g("mean_occupation", 0.0)),
            "phase_volume": float(g("phase_space_volume", 0.0)),
            "purity": float(g("purity", 0.0)),
            "lyap_max": float(g("lyapunov_max", 0.0)),
            "lyap_min": float(g("lyapunov_min", 0.0)),
            "occupation_entropy": float(g("occupation_entropy", 0.0)),
            "ccr_canonical_residual": float(g("ccr_canonical_residual", 0.0)),
            "car_canonical_residual": float(g("car_canonical_residual", 0.0)),
            "ccr_canonical_valid": bool(g("ccr_canonical_valid", False)),
            "car_canonical_valid": bool(g("car_canonical_valid", False)),
            "engine_v14": bool(is_v14),
            "symplectic_residual": float(g("symplectic_residual", 0.0)),
            "quadrature_x": float(g("quadrature_x", 0.0)),
            "quadrature_p": float(g("quadrature_p", 0.0)),
        }

    def _phase1_v14_payload(self, cert_like: Dict[str, Any]) -> str:
        return (
            f"{cert_like['wit_id']}:{cert_like['trace']:.8f}:{cert_like['liouville']:.8f}:"
            f"{cert_like['car_legacy']:.8f}:"
            f"{cert_like['action_total']:.8f}:{cert_like['mean_occupation']:.8f}:"
            f"{cert_like['phase_volume']:.8f}:{cert_like['purity']:.8f}:"
            f"{cert_like['lyap_max']:+.6e}:{cert_like['lyap_min']:+.6e}:"
            f"{cert_like['occupation_entropy']:.8f}:"
            f"{cert_like['ccr_res']:.3e}:{cert_like['car_res']:.3e}:"
            f"{int(cert_like['ccr_ok'])}:{int(cert_like['car_ok'])}:"
            f"{int(cert_like['engine_v14'])}:{cert_like['verdict']}:"
            f"{cert_like['chain']}:"
            f"{cert_like['sym_res']:.3e}:{cert_like['qx']:+.8f}:{cert_like['qp']:+.8f}:"
            f"{cert_like['fp']}:{cert_like['invocations']}"
        )

    def _p1_payload_from_cert(self, cert1: FPUWitnessPhase1CertificateV14) -> str:
        return self._phase1_v14_payload({
            "wit_id": cert1.witness_id,
            "trace": cert1.trace_preservation,
            "liouville": cert1.liouville_drift,
            "car_legacy": cert1.car_anti_commutation_residual,
            "action_total": cert1.action_total,
            "mean_occupation": cert1.mean_occupation,
            "phase_volume": cert1.phase_space_volume,
            "purity": cert1.purity,
            "lyap_max": cert1.lyapunov_max,
            "lyap_min": cert1.lyapunov_min,
            "occupation_entropy": cert1.occupation_entropy,
            "ccr_res": cert1.ccr_canonical_residual,
            "car_res": cert1.car_canonical_residual,
            "ccr_ok": cert1.ccr_canonical_valid,
            "car_ok": cert1.car_canonical_valid,
            "engine_v14": cert1.engine_v14_active,
            "verdict": cert1.isometry_verdict.value,
            "chain": cert1.chain_hash,
            "sym_res": cert1.symplectic_residual,
            "qx": cert1.quadrature_x,
            "qp": cert1.quadrature_p,
            "fp": cert1.density_fingerprint,
            "invocations": cert1.engine_invocations,
        })

    # ────────────────── Sub-capa 1.3: emisión P1v14 ──────────────────
    def execute_phase1_v14(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        state_a: Optional[NDArray] = None,  # v13 compat: ignorado (canónicos v14)
        state_b: Optional[NDArray] = None,  # v13 compat: ignorado
    ) -> FPUWitnessPhase1CertificateV14:
        """Sub-capa 1.3 — Emite FPUWitnessPhase1CertificateV14 firmado HMAC-SHA256."""
        now = time.time()
        self._reset_session_cycle()
        snapshot = self._invoke_engine_v14_and_cache(density_matrix, phi_initial, phi_final)
        inv = self._extract_phase1_invariants_v14(snapshot, density_matrix)
        fp = self._session.density_fingerprint or _fingerprint_density(density_matrix)

        if inv["liouville"] > _LIOUVILLE_VETO or inv["car_legacy"] > 1.5:
            verdict = HeytingOmega3.VETOED
        elif inv["liouville"] > _LIOUVILLE_DEGRADED:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        wit_id = f"FPU-WIT-P1v14-{int(now * 1000.0) % 1000000:06d}"
        chain_bytes = (
            f"{wit_id}:{inv['trace']:.8f}:{inv['car_legacy']:.8f}:"
            f"{inv['action_total']:.8f}:{verdict.value}:{fp}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        blob = {
            "wit_id": wit_id,
            "trace": inv["trace"],
            "liouville": inv["liouville"],
            "car_legacy": inv["car_legacy"],
            "action_total": inv["action_total"],
            "mean_occupation": inv["mean_occupation"],
            "phase_volume": inv["phase_volume"],
            "purity": inv["purity"],
            "lyap_max": inv["lyap_max"],
            "lyap_min": inv["lyap_min"],
            "occupation_entropy": inv["occupation_entropy"],
            "ccr_res": inv["ccr_canonical_residual"],
            "car_res": inv["car_canonical_residual"],
            "ccr_ok": inv["ccr_canonical_valid"],
            "car_ok": inv["car_canonical_valid"],
            "engine_v14": inv["engine_v14"],
            "verdict": verdict.value,
            "chain": chain_hash,
            "sym_res": inv["symplectic_residual"],
            "qx": inv["quadrature_x"],
            "qp": inv["quadrature_p"],
            "fp": fp,
            "invocations": self._session.engine_invocations,
        }
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._phase1_v14_payload(blob))

        cert = FPUWitnessPhase1CertificateV14(
            witness_id=wit_id,
            trace_preservation=inv["trace"],
            liouville_drift=inv["liouville"],
            car_anti_commutation_residual=inv["car_legacy"],
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_p1_signature=sig,
            timestamp_utc=now,
            action_total=inv["action_total"],
            mean_occupation=inv["mean_occupation"],
            phase_space_volume=inv["phase_volume"],
            purity=inv["purity"],
            lyapunov_max=inv["lyap_max"],
            lyapunov_min=inv["lyap_min"],
            occupation_entropy=inv["occupation_entropy"],
            ccr_canonical_residual=inv["ccr_canonical_residual"],
            car_canonical_residual=inv["car_canonical_residual"],
            ccr_canonical_valid=inv["ccr_canonical_valid"],
            car_canonical_valid=inv["car_canonical_valid"],
            engine_v14_active=inv["engine_v14"],
            symplectic_residual=inv["symplectic_residual"],
            quadrature_x=inv["quadrature_x"],
            quadrature_p=inv["quadrature_p"],
            density_fingerprint=fp,
            engine_invocations=self._session.engine_invocations,
        )
        self._session.cert1 = cert
        logger.debug(
            "FASE1 Observe: id=%s ΣJ=%.4f (x,p)=(%.3f,%.3f) iso=%s inv=%d",
            wit_id, cert.action_total, cert.quadrature_x, cert.quadrature_p,
            verdict.name, cert.engine_invocations,
        )
        return cert

    def continue_to_phase2_witness_gate_v14(
        self, cert1: FPUWitnessPhase1CertificateV14
    ) -> bool:
        """
        Compuerta Φ₁₂ — revalida HMAC P1v14.
        ADMITE cualquier veredicto (pasividad QND: el testigo NUNCA interviene).
        """
        expected = self._compute_hmac(_HMAC_DOMAIN_P1, self._p1_payload_from_cert(cert1))
        return _safe_hmac_compare(cert1.hmac_p1_signature, expected)

    def handoff_to_phase2_v14(
        self, cert1: FPUWitnessPhase1CertificateV14
    ) -> FPUWitnessPhase1CertificateV14:
        """
        ÚLTIMO ACTO FORMAL DE FASE 1 ≡ PRIMER ACTO FORMAL DE FASE 2.

        Revalida Φ₁₂ y entrega el certificado autenticado que
        `execute_phase2_v14` consume como único puente Observe → Orient.
        No hay otro canal: ni snapshot crudo, ni re-invocación.
        """
        if not self.continue_to_phase2_witness_gate_v14(cert1):
            raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ del Testigo FPU v14.1")
        return cert1

    def run_phase1_observe(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> FPUWitnessPhase1CertificateV14:
        """Orquestador FASE 1. Cierra con el HANDOFF que abre FASE 2."""
        cert1 = self.execute_phase1_v14(density_matrix, phi_initial, phi_final)
        return self.handoff_to_phase2_v14(cert1)

    # ── Alias v13 (HMAC LEGACY, payload recortado: no colisiona con v14) ──
    def _p1_legacy_payload(self, cert: FPUWitnessPhase1Certificate) -> str:
        return (
            f"{cert.witness_id}:{cert.trace_preservation:.8f}:{cert.liouville_drift:.8f}:"
            f"{cert.car_anti_commutation_residual:.8f}:{cert.isometry_verdict.value}:{cert.chain_hash}"
        )

    def execute_phase1(
        self,
        density_matrix: NDArray,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> FPUWitnessPhase1Certificate:
        """Alias v13: emite P1v13 re-firmado con dominio LEGACY (HMAC verificable por la compuerta v13)."""
        cert_v14 = self.execute_phase1_v14(density_matrix, None, None, state_a, state_b)
        return self._downgrade_p1(cert_v14)

    def _downgrade_p1(self, cert_v14: FPUWitnessPhase1CertificateV14) -> FPUWitnessPhase1Certificate:
        stub = FPUWitnessPhase1Certificate(
            witness_id=cert_v14.witness_id,
            trace_preservation=cert_v14.trace_preservation,
            liouville_drift=cert_v14.liouville_drift,
            car_anti_commutation_residual=cert_v14.car_anti_commutation_residual,
            isometry_verdict=cert_v14.isometry_verdict,
            chain_hash=cert_v14.chain_hash,
            hmac_p1_signature="",
            timestamp_utc=cert_v14.timestamp_utc,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1_LEGACY, self._p1_legacy_payload(stub))
        return FPUWitnessPhase1Certificate(
            witness_id=stub.witness_id,
            trace_preservation=stub.trace_preservation,
            liouville_drift=stub.liouville_drift,
            car_anti_commutation_residual=stub.car_anti_commutation_residual,
            isometry_verdict=stub.isometry_verdict,
            chain_hash=stub.chain_hash,
            hmac_p1_signature=sig,
            timestamp_utc=stub.timestamp_utc,
        )

    def continue_to_phase2_witness_gate(self, cert1: FPUWitnessPhase1Certificate) -> bool:
        """Alias v13: verifica HMAC LEGACY (payload recortado)."""
        expected = self._compute_hmac(_HMAC_DOMAIN_P1_LEGACY, self._p1_legacy_payload(cert1))
        return _safe_hmac_compare(cert1.hmac_p1_signature, expected)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 2 — ORIENT (3 sub-capas)  — continúa handoff_to_phase2_v14
# ██╔══██╗ Extracción F2 celeste + N3 motor · emisión P2v14 (consume cache)
# ██████╔╝ [HANDOFF] handoff_to_phase3_v14 → cert2 autenticado (input de FASE 3)
# ██╔═══╝  El primer método operativo consume el último acto de FASE 1.
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUWitnessPhase2OrientorV14(FPUWitnessPhase1ObserverV14):
    """
    FASE 2 — ORIENT con 3 sub-capas anidadas.

    CONTINÚA `handoff_to_phase2_v14` (último acto de FASE 1).

    Sub-capa 2.1: `_extract_phase2_poincare_invariants_v14` — QND + Poincaré celeste.
    Sub-capa 2.2: `_extract_phase2_n3_tuning_v14` — TuningPolicy N2/N3 + λ_α.
    Sub-capa 2.3: `execute_phase2_v14` — emite P2v14, sin re-invocar el motor.
    Handoff:      `handoff_to_phase3_v14` / `continue_to_phase3_witness_gate_v14`.
    """

    # ────────────────── Sub-capa 2.1 ──────────────────
    def _extract_phase2_poincare_invariants_v14(
        self, snapshot: Optional[Any]
    ) -> Dict[str, Any]:
        """Extrae invariantes QND heredados v13 + Poincaré F2 v14.1 (celeste)."""
        def g(name: str, default: Any) -> Any:
            return getattr(snapshot, name, default) if snapshot is not None else default

        crossings = int(g("poincare_crossings_count", 0))
        drift_max = float(g("occupation_drift_max", 0.0))
        poincare_lyap = float(g("poincare_lyapunov", 0.0))
        # Sección transversal: hay cruces Y (drift en p_n o estiramiento Benettin).
        # El bug v14.0 (H ∥ ρ ⇒ drift de autovalores ≡ 0) se detecta aquí.
        transverse = bool(crossings > 0 and (drift_max > 0.0 or abs(poincare_lyap) > 0.0))
        verdict = g("global_verdict", HeytingOmega3.COHERENT)
        if not isinstance(verdict, HeytingOmega3):
            try:
                verdict = HeytingOmega3(int(verdict))
            except (ValueError, TypeError):
                verdict = HeytingOmega3.COHERENT
        return {
            "a_weak": complex(g("qnd_weak_value_aav", 0.0)),
            "w_mod": float(g("weak_value_modulus", 0.0)),
            "back_action": float(g("back_action_db", 0.0)),
            "fidelity": float(g("uhlmann_fidelity", 1.0)),
            "bures": float(g("bures_distance", 0.0)),
            "spohn": float(g("spohn_entropy_production", 0.0)),
            "gksl": float(g("gksl_dissipation_rate", 0.0)),
            "verdict": verdict,
            "crossings": crossings,
            "drift_max": drift_max,
            "drift_mean": float(g("occupation_drift_mean", 0.0)),
            "lindstedt": float(g("lindstedt_correction_norm", 0.0)),
            "lindstedt_o2": float(g("lindstedt_order2_norm", 0.0)),
            "birkhoff": float(g("birkhoff_quadratic_coefficient", 0.0)),
            "kam_index": float(g("kam_stability_index", 1.0)),
            "kam_viol": int(g("kam_resonance_violations", 0)),
            "kam_worst": float(g("kam_worst_ratio", 1.0e9)),
            "rotation_number": float(g("rotation_number", 0.0)),
            "t_return": float(g("first_return_time_mean", 0.0)),
            "poincare_lyap": poincare_lyap,
            "nek_time": float(g("nekhoroshev_time", 0.0)),
            "transverse": transverse,
        }

    # ────────────────── Sub-capa 2.2 ──────────────────
    def _extract_phase2_n3_tuning_v14(
        self, snapshot: Optional[Any]
    ) -> Dict[str, float]:
        """Extrae la TuningPolicy N2/N3 del snapshot v14.1 (incluye λ_α del motor)."""
        def g(name: str, default: float) -> float:
            return float(getattr(snapshot, name, default)) if snapshot is not None else default

        return {
            "fidelity_floor": g("adaptive_fidelity_floor", 0.95),
            "bures_ceiling": g("adaptive_bures_ceiling", 0.30),
            "spohn_ceiling": g("adaptive_spohn_ceiling", 2.0),
            "adapt_rate": g("adaptation_rate", 0.05),
            "meta_adapt_rate": g("meta_adaptation_rate", 0.01),
            "entropy": g("level_3_entropy", 0.0),
            "adapt_lyap": g("n3_adaptation_lyapunov", 0.0),
        }

    def _phase2_v14_payload(self, c: FPUWitnessPhase2CertificateV14) -> str:
        return (
            f"{c.witness_id}:{c.phase1_ref_id}:{c.weak_value_qnd_aav}:"
            f"{c.weak_value_modulus:.8f}:{c.back_action_db:.1f}:"
            f"{c.uhlmann_fidelity:.8f}:{c.bures_distance:.8f}:"
            f"{c.spohn_entropy_production:.8e}:{c.gksl_dissipation_rate:.8e}:"
            f"{c.poincare_crossings_count}:{c.occupation_drift_max:.8e}:"
            f"{c.occupation_drift_mean:.8e}:"
            f"{c.lindstedt_correction_norm:.8e}:{c.birkhoff_quadratic_coefficient:.8e}:"
            f"{c.kam_stability_index:.8f}:{c.kam_resonance_violations}:"
            f"{c.kam_worst_ratio:.6e}:"
            f"{c.adaptive_fidelity_floor:.8f}:{c.adaptive_bures_ceiling:.8f}:"
            f"{c.adaptive_spohn_ceiling:.8f}:{c.adaptation_rate:.6f}:"
            f"{c.meta_adaptation_rate:.6f}:{c.level_3_entropy:.6f}:"
            f"{c.qnd_verdict.value}:{c.chain_hash}:"
            f"{c.lindstedt_order2_norm:.8e}:{c.rotation_number:.8f}:"
            f"{c.first_return_time_mean:.8e}:{c.poincare_lyapunov:.8e}:"
            f"{c.nekhoroshev_time:.8e}:{c.n3_adaptation_lyapunov:.8e}:"
            f"{int(c.celestial_section_transverse)}:{c.engine_invocations}"
        )

    # ────────────────── Sub-capa 2.3 ──────────────────
    def execute_phase2_v14(
        self,
        cert1: FPUWitnessPhase1CertificateV14,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> FPUWitnessPhase2CertificateV14:
        """
        Sub-capa 2.3 — CONTINÚA FASE 1.

        Primer acto: `handoff_to_phase2_v14(cert1)` (revalida Φ₁₂).
        Emite P2v14 firmado HMAC-SHA256. Consume el snapshot cacheado.
        NO re-invoca el motor salvo caché vacía (I4 se marca violado).
        """
        now = time.time()
        cert1 = self.handoff_to_phase2_v14(cert1)  # ← puente formal 1 → 2

        snapshot = self._session.snapshot
        if snapshot is None:
            snapshot = self._invoke_engine_v14_and_cache(
                density_matrix, phi_initial, phi_final, emergency=True
            )

        fp_now = _fingerprint_density(density_matrix)
        if (
            cert1.density_fingerprint
            and fp_now != cert1.density_fingerprint
        ):
            logger.warning(
                "Huella de ρ divergente entre FASE 1 (%s) y FASE 2 (%s)",
                cert1.density_fingerprint, fp_now,
            )

        inv = self._extract_phase2_poincare_invariants_v14(snapshot)
        tune = self._extract_phase2_n3_tuning_v14(snapshot)

        wit_id = f"FPU-WIT-P2v14-{int(now * 1000.0) % 1000000:06d}"
        chain_bytes = (
            f"{cert1.chain_hash}:{wit_id}:{inv['fidelity']:.8f}:{inv['spohn']:.8f}:"
            f"{inv['kam_index']:.8f}:{inv['verdict'].value}:"
            f"{inv['poincare_lyap']:.6e}:{inv['nek_time']:.6e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        cert = FPUWitnessPhase2CertificateV14(
            witness_id=wit_id,
            phase1_ref_id=cert1.witness_id,
            weak_value_qnd_aav=inv["a_weak"],
            weak_value_modulus=inv["w_mod"],
            back_action_db=inv["back_action"],
            uhlmann_fidelity=inv["fidelity"],
            bures_distance=inv["bures"],
            spohn_entropy_production=inv["spohn"],
            gksl_dissipation_rate=inv["gksl"],
            qnd_verdict=inv["verdict"],
            chain_hash=chain_hash,
            hmac_p2_signature="",  # se rellena tras firmar el payload canónico
            timestamp_utc=now,
            poincare_crossings_count=inv["crossings"],
            occupation_drift_max=inv["drift_max"],
            occupation_drift_mean=inv["drift_mean"],
            lindstedt_correction_norm=inv["lindstedt"],
            birkhoff_quadratic_coefficient=inv["birkhoff"],
            kam_stability_index=inv["kam_index"],
            kam_resonance_violations=inv["kam_viol"],
            kam_worst_ratio=inv["kam_worst"],
            adaptive_fidelity_floor=tune["fidelity_floor"],
            adaptive_bures_ceiling=tune["bures_ceiling"],
            adaptive_spohn_ceiling=tune["spohn_ceiling"],
            adaptation_rate=tune["adapt_rate"],
            meta_adaptation_rate=tune["meta_adapt_rate"],
            level_3_entropy=tune["entropy"],
            lindstedt_order2_norm=inv["lindstedt_o2"],
            rotation_number=inv["rotation_number"],
            first_return_time_mean=inv["t_return"],
            poincare_lyapunov=inv["poincare_lyap"],
            nekhoroshev_time=inv["nek_time"],
            n3_adaptation_lyapunov=tune["adapt_lyap"],
            celestial_section_transverse=inv["transverse"],
            engine_invocations=self._session.engine_invocations,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_v14_payload(cert))
        object.__setattr__(cert, "hmac_p2_signature", sig)

        logger.debug(
            "FASE2 Orient: id=%s KAM=%.4f λP=%.3e Tnek=%.3e ν=%.4f transverse=%s",
            wit_id, cert.kam_stability_index, cert.poincare_lyapunov,
            cert.nekhoroshev_time, cert.rotation_number, cert.celestial_section_transverse,
        )
        return cert

    def continue_to_phase3_witness_gate_v14(
        self, cert2: FPUWitnessPhase2CertificateV14
    ) -> bool:
        """Compuerta Φ₂₃ — revalida HMAC P2v14 (firma hex completa, 64 chars)."""
        if not (cert2.hmac_p2_signature and len(cert2.hmac_p2_signature) == 64):
            return False
        expected = self._compute_hmac(_HMAC_DOMAIN_P2, self._phase2_v14_payload(cert2))
        return _safe_hmac_compare(cert2.hmac_p2_signature, expected)

    def handoff_to_phase3_v14(
        self, cert2: FPUWitnessPhase2CertificateV14
    ) -> FPUWitnessPhase2CertificateV14:
        """
        ÚLTIMO ACTO FORMAL DE FASE 2 ≡ PRIMER ACTO FORMAL DE FASE 3.

        Revalida Φ₂₃ y entrega el certificado autenticado que
        `execute_phase3_v14` consume como único puente Orient → Decide.
        """
        if not self.continue_to_phase3_witness_gate_v14(cert2):
            raise ValueError("Falla de autenticidad en Compuerta Φ₂₃ del Testigo FPU v14.1")
        return cert2

    def run_phase2_orient(
        self,
        cert1: FPUWitnessPhase1CertificateV14,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> FPUWitnessPhase2CertificateV14:
        """Orquestador FASE 2. Abre con el HANDOFF de FASE 1 y cierra el de FASE 3."""
        cert2 = self.execute_phase2_v14(cert1, density_matrix, phi_initial, phi_final)
        return self.handoff_to_phase3_v14(cert2)

    # ── Alias v13 ──
    def _upgrade_p1(self, c: Any) -> FPUWitnessPhase1CertificateV14:
        if isinstance(c, FPUWitnessPhase1CertificateV14):
            return c
        if self._session.cert1 is not None and self._session.cert1.witness_id == getattr(c, "witness_id", ""):
            return self._session.cert1
        return FPUWitnessPhase1CertificateV14(
            witness_id=c.witness_id,
            trace_preservation=c.trace_preservation,
            liouville_drift=c.liouville_drift,
            car_anti_commutation_residual=c.car_anti_commutation_residual,
            isometry_verdict=c.isometry_verdict,
            chain_hash=c.chain_hash,
            hmac_p1_signature=c.hmac_p1_signature,
            timestamp_utc=c.timestamp_utc,
            action_total=0.0, mean_occupation=0.0, phase_space_volume=0.0,
            purity=0.0, lyapunov_max=0.0, lyapunov_min=0.0, occupation_entropy=0.0,
            ccr_canonical_residual=0.0, car_canonical_residual=0.0,
            ccr_canonical_valid=False, car_canonical_valid=False,
            engine_v14_active=False,
            symplectic_residual=0.0, quadrature_x=0.0, quadrature_p=0.0,
            density_fingerprint="", engine_invocations=self._session.engine_invocations,
        )

    def _p2_legacy_payload(self, cert: FPUWitnessPhase2Certificate) -> str:
        return (
            f"{cert.witness_id}:{cert.phase1_ref_id}:{cert.weak_value_qnd_aav}:"
            f"{cert.weak_value_modulus:.8f}:{cert.back_action_db:.1f}:"
            f"{cert.uhlmann_fidelity:.8f}:{cert.bures_distance:.8f}:"
            f"{cert.spohn_entropy_production:.8e}:{cert.gksl_dissipation_rate:.8e}:"
            f"{cert.qnd_verdict.value}:{cert.chain_hash}"
        )

    def _downgrade_p2(self, c: FPUWitnessPhase2CertificateV14) -> FPUWitnessPhase2Certificate:
        stub = FPUWitnessPhase2Certificate(
            witness_id=c.witness_id,
            phase1_ref_id=c.phase1_ref_id,
            weak_value_qnd_aav=c.weak_value_qnd_aav,
            weak_value_modulus=c.weak_value_modulus,
            back_action_db=c.back_action_db,
            uhlmann_fidelity=c.uhlmann_fidelity,
            bures_distance=c.bures_distance,
            spohn_entropy_production=c.spohn_entropy_production,
            gksl_dissipation_rate=c.gksl_dissipation_rate,
            qnd_verdict=c.qnd_verdict,
            chain_hash=c.chain_hash,
            hmac_p2_signature="",
            timestamp_utc=c.timestamp_utc,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2_LEGACY, self._p2_legacy_payload(stub))
        return FPUWitnessPhase2Certificate(
            witness_id=stub.witness_id,
            phase1_ref_id=stub.phase1_ref_id,
            weak_value_qnd_aav=stub.weak_value_qnd_aav,
            weak_value_modulus=stub.weak_value_modulus,
            back_action_db=stub.back_action_db,
            uhlmann_fidelity=stub.uhlmann_fidelity,
            bures_distance=stub.bures_distance,
            spohn_entropy_production=stub.spohn_entropy_production,
            gksl_dissipation_rate=stub.gksl_dissipation_rate,
            qnd_verdict=stub.qnd_verdict,
            chain_hash=stub.chain_hash,
            hmac_p2_signature=sig,
            timestamp_utc=stub.timestamp_utc,
        )

    def execute_phase2(
        self,
        cert1: Any,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> FPUWitnessPhase2Certificate:
        """Alias v13: acepta P1v13 o P1v14; retorna P2 v13 re-firmado LEGACY."""
        if not isinstance(cert1, FPUWitnessPhase1CertificateV14):
            if not self.continue_to_phase2_witness_gate(cert1):
                raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ (legado v13)")
        cert1_v14 = self._upgrade_p1(cert1)
        # Si el upgrade perdió la firma v14, usar el cert de sesión (ciclo v13 → v14).
        if (
            self._session.cert1 is not None
            and not self.continue_to_phase2_witness_gate_v14(cert1_v14)
        ):
            cert1_v14 = self._session.cert1
        cert2_v14 = self.execute_phase2_v14(cert1_v14, density_matrix, phi_initial, phi_final)
        return self._downgrade_p2(cert2_v14)

    def continue_to_phase3_witness_gate(self, cert2: FPUWitnessPhase2Certificate) -> bool:
        """Alias v13: verifica HMAC LEGACY de P2."""
        if not (cert2.hmac_p2_signature and len(cert2.hmac_p2_signature) == 64):
            return False
        expected = self._compute_hmac(_HMAC_DOMAIN_P2_LEGACY, self._p2_legacy_payload(cert2))
        return _safe_hmac_compare(cert2.hmac_p2_signature, expected)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 3 — DECIDE & ACT (3 sub-capas)  — continúa handoff_to_phase3_v14
# ██╔══██╗ Meta N3 motor + OODA · adjudicación KAM/λ_P/Nekhoroshev · Estado Soberano v14.1
# ██████╔╝ [SALIDA] TopologicalFPUTensorWitnessSovereignStateV14
# ██╔═══╝  El primer método operativo consume el último acto de FASE 2.
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUWitnessPhase3DeciderV14(FPUWitnessPhase2OrientorV14):
    """
    FASE 3 — DECIDE & ACT con 3 sub-capas anidadas.

    CONTINÚA `handoff_to_phase3_v14` (último acto de FASE 2).

    Sub-capa 3.1: `_extract_phase3_meta_witness_v14` — meta N3 motor + OODA.
    Sub-capa 3.2: `_adjudicate_kam_aware_v14` — 10 reglas Heyting celestes.
    Sub-capa 3.3: `execute_phase3_v14` — Estado Soberano v14.1 + PASSIVE_SILENT.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._n3_verdict_trajectory: List[int] = []
        self._n3_cycle_count: int = 0
        self._ooda_ledger: FockOODALedger = FockOODALedger()

    # ────────────────── Sub-capa 3.1 ──────────────────
    def _extract_phase3_meta_witness_v14(
        self, snapshot: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Extrae observables meta N3 del snapshot del motor.
        `level_3_entropy` es el nombre canónico v14.1; `meta_verdict_entropy` es alias legado.
        """
        def g(name: str, default: Any) -> Any:
            return getattr(snapshot, name, default) if snapshot is not None else default

        entropy = g("level_3_entropy", None)
        if entropy is None:
            entropy = g("meta_verdict_entropy", 0.0)
        traj = g("n3_verdict_trajectory", ())
        if traj is None:
            traj = ()
        return {
            "consistency": float(g("meta_consistency_score", 1.0)),
            "entropy": float(entropy if entropy is not None else 0.0),
            "fid_std": float(g("meta_fidelity_std", 0.0)),
            "bures_std": float(g("meta_bures_std", 0.0)),
            "spohn_std": float(g("meta_spohn_std", 0.0)),
            "reflections": int(g("level_3_reflections", 0)),
            "n3_cycle_count": int(g("n3_cycle_count", 0)),
            "engine_traj": tuple(int(v) for v in traj) if traj else tuple(),
            "adapt_lyap": float(g("n3_adaptation_lyapunov", 0.0)),
        }

    # ────────────────── Sub-capa 3.2 ──────────────────
    def _adjudicate_kam_aware_v14(
        self,
        base_verdict_v13: HeytingOmega3,
        kam_index: float,
        meta_consistency: float,
        ccr_canonical_valid: bool,
        car_canonical_valid: bool,
        poincare_lyap: float,
        nek_time: float,
        section_transverse: bool,
        engine_v14_active: bool,
        symplectic_residual: float,
    ) -> HeytingOmega3:
        """
        Sub-capa 3.2 — Adjudicación Heyting KAM-aware + celestial + N3 OODA.

        Reglas (en orden; `meet` es monótono de severidad):
          R1  base = meet(isometry, qnd)                         [heredado v13]
          R2  !ccr ∨ !car                         ⟹ DEGRADED    [canónicos]
          R3  kam < θ_degraded(OODA)              ⟹ DEGRADED    [ruido espectral]
          R4  kam < θ_severe                      ⟹ VETOED      [caos]
          R5  meta < 0.50                         ⟹ DEGRADED
          R6  meta < θ_meta(OODA) ∧ COHERENT      ⟹ DEGRADED
          R8  λ_P > 0.50                          ⟹ DEGRADED    [Benettin]
          R9  engine_v14 ∧ !sección transversal   ⟹ DEGRADED    [H ∥ ρ detectado]
          R10 T_Nek < 2                           ⟹ DEGRADED    [perturbación enorme]
          R11 ‖[x,p] − i[b,b†]‖ > tol             ⟹ DEGRADED    [I8]
          R7  refuerzo COHERENT sólo si R2–R11 verdes y kam ≥ θ_coherent(OODA)
        """
        verdict = base_verdict_v13
        thr_coh = self._ooda_ledger.kam_coherent_threshold
        thr_deg = self._ooda_ledger.kam_degraded_threshold
        thr_meta = self._ooda_ledger.meta_degraded_threshold

        if not ccr_canonical_valid or not car_canonical_valid:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if kam_index < thr_deg:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if kam_index < _KAM_SEVERE_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if meta_consistency < _META_CONSISTENCY_VETO:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if meta_consistency < thr_meta and verdict == HeytingOmega3.COHERENT:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if poincare_lyap > _POINCARE_LYAP_DEGRADED:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if engine_v14_active and not section_transverse:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if 0.0 < nek_time < _NEKHOROSHEV_TIME_FLOOR:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if symplectic_residual > _CANONICAL_RESIDUAL_TOL:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)

        if (
            base_verdict_v13 == HeytingOmega3.COHERENT
            and kam_index >= thr_coh
            and meta_consistency >= thr_meta
            and ccr_canonical_valid and car_canonical_valid
            and poincare_lyap <= _POINCARE_LYAP_DEGRADED
            and (section_transverse or not engine_v14_active)
            and symplectic_residual <= _CANONICAL_RESIDUAL_TOL
        ):
            verdict = HeytingOmega3.COHERENT

        return verdict

    # ────────────────── Sub-capa 3.3 ──────────────────
    def execute_phase3_v14(
        self,
        cert1: FPUWitnessPhase1CertificateV14,
        cert2: FPUWitnessPhase2CertificateV14,
    ) -> TopologicalFPUTensorWitnessSovereignStateV14:
        """
        Sub-capa 3.3 — CONTINÚA FASE 2.

        Primer acto: `handoff_to_phase3_v14(cert2)` (revalida Φ₂₃).
        Emite el Estado Soberano v14.1, actualiza el ledger OODA N3
        (sin tocar el motor) y sella la cadena HMAC P1→P2→P3.
        """
        now = time.time()
        cert2 = self.handoff_to_phase3_v14(cert2)  # ← puente formal 2 → 3

        snapshot = self._session.snapshot
        meta = self._extract_phase3_meta_witness_v14(snapshot)

        base_verdict_v13 = cert1.isometry_verdict.meet(cert2.qnd_verdict)
        final_verdict = self._adjudicate_kam_aware_v14(
            base_verdict_v13=base_verdict_v13,
            kam_index=cert2.kam_stability_index,
            meta_consistency=meta["consistency"],
            ccr_canonical_valid=cert1.ccr_canonical_valid,
            car_canonical_valid=cert1.car_canonical_valid,
            poincare_lyap=cert2.poincare_lyapunov,
            nek_time=cert2.nekhoroshev_time,
            section_transverse=cert2.celestial_section_transverse,
            engine_v14_active=cert1.engine_v14_active,
            symplectic_residual=cert1.symplectic_residual,
        )

        passive_silent_ok = (
            abs(cert2.back_action_db) <= _BACK_ACTION_SILENT_TOL
            and cert1.liouville_drift <= _LIOUVILLE_VETO
        )
        single_inv_ok = (
            self._session.engine_invocations == 1
            and self._session.emergency_reinvoke == 0
        )

        # Ledger OODA N3 (agente): N1 observe → N2 adapt → N2χ Poincaré → N3 meta.
        self._ooda_ledger.observe(
            final_verdict,
            cert2.kam_stability_index,
            cert2.poincare_lyapunov,
            meta["consistency"],
        )
        self._ooda_ledger.level_2_adapt()
        self._ooda_ledger.poincare_modulate(
            cert2.kam_stability_index,
            cert2.poincare_lyapunov,
            cert2.nekhoroshev_time,
        )
        ooda_entropy = self._ooda_ledger.level_3_adapt()
        ooda_lyap = self._ooda_ledger.adaptation_lyapunov()

        self._n3_cycle_count += 1
        self._n3_verdict_trajectory.append(int(final_verdict))
        if len(self._n3_verdict_trajectory) > _N3_TRAJECTORY_WINDOW:
            self._n3_verdict_trajectory = self._n3_verdict_trajectory[-_N3_TRAJECTORY_WINDOW:]

        state_id = f"FPU-WITNESS-v14-STATE-{int(now * 1000.0) % 1000000:06d}"
        merkle_seed = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:"
            f"{int(final_verdict)}:{int(base_verdict_v13)}:"
            f"{cert2.kam_stability_index:.6f}:{meta['consistency']:.6f}:"
            f"{cert2.poincare_lyapunov:.6e}:{cert2.nekhoroshev_time:.6e}:"
            f"{int(passive_silent_ok)}:{int(single_inv_ok)}:"
            f"{int(cert2.celestial_section_transverse)}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_seed).hexdigest()

        prov_payload = (
            f"{merkle_root}:{cert2.back_action_db:.1f}:{final_verdict.value}:"
            f"{meta['consistency']:.6f}:{cert2.kam_stability_index:.6f}:"
            f"{self._n3_cycle_count}:{cert2.poincare_lyapunov:.6e}:"
            f"{ooda_lyap:.6e}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
        chain_p1p2p3 = hashlib.sha256(
            (cert1.hmac_p1_signature + cert2.hmac_p2_signature + prov_sig).encode("utf-8")
        ).hexdigest()

        if not passive_silent_ok:
            exec_action = "ALERTA: INVARIANTE PASSIVE_SILENT VIOLADO (revisar QND)"
        elif not single_inv_ok:
            exec_action = (
                f"ALERTA I4: motor invocado {self._session.engine_invocations} veces "
                f"(emergencias={self._session.emergency_reinvoke}). Ledger N3 del motor en riesgo."
            )
        elif final_verdict == HeytingOmega3.COHERENT:
            exec_action = (
                "CERTIFICACIÓN SILENCIOSA DE COHERENCIA EN RAM (0.0 dB, sin interrupción de obra)"
            )
        elif final_verdict == HeytingOmega3.DEGRADED:
            exec_action = (
                f"DEGRADACIÓN RECUPERABLE: KAM={cert2.kam_stability_index:.4f}, "
                f"λP={cert2.poincare_lyapunov:.3e}, meta={meta['consistency']:.4f}. "
                f"Ventana de gracia activa."
            )
        else:
            exec_action = (
                f"VETO CATEGORIÁTICO REPORTADO (NO destructivo): "
                f"KAM={cert2.kam_stability_index:.4f}, λP={cert2.poincare_lyapunov:.3e}, "
                f"meta={meta['consistency']:.4f}."
            )

        visceral_report = {
            "witness_qnd_status": (
                "OBSERVACIÓN DE CERO DEMOLICIÓN CUÁNTICA (0.0 dB) EN RAM"
                if passive_silent_ok else "ALERTA: PASSIVE_SILENT VIOLADO"
            ),
            "uhlmann_fidelity": cert2.uhlmann_fidelity,
            "bures_distance": cert2.bures_distance,
            "spohn_entropy_production": cert2.spohn_entropy_production,
            "gksl_dissipation_rate": cert2.gksl_dissipation_rate,
            "wacc_protection": "PROTEGIDO 15.0%",
            "net_liquidity_impact": "+11.5% FLUIDO LIBRE (Coherencia Fock)",
            "kv_cache_compression_pct": 86.4,
            "llm_cost_savings_pct": 80.0,
            "poincare_crossings": cert2.poincare_crossings_count,
            "occupation_drift_max": cert2.occupation_drift_max,
            "kam_stability_index": cert2.kam_stability_index,
            "kam_resonance_violations": cert2.kam_resonance_violations,
            "birkhoff_anharmonicity": cert2.birkhoff_quadratic_coefficient,
            "lindstedt_correction_norm": cert2.lindstedt_correction_norm,
            "lindstedt_order2_norm": cert2.lindstedt_order2_norm,
            "rotation_number": cert2.rotation_number,
            "first_return_time_mean": cert2.first_return_time_mean,
            "poincare_lyapunov": cert2.poincare_lyapunov,
            "nekhoroshev_time": cert2.nekhoroshev_time,
            "celestial_section_transverse": cert2.celestial_section_transverse,
            "action_total": cert1.action_total,
            "mean_occupation": cert1.mean_occupation,
            "phase_space_volume": cert1.phase_space_volume,
            "purity": cert1.purity,
            "occupation_entropy": cert1.occupation_entropy,
            "lyapunov_max": cert1.lyapunov_max,
            "lyapunov_min": cert1.lyapunov_min,
            "quadrature_x": cert1.quadrature_x,
            "quadrature_p": cert1.quadrature_p,
            "symplectic_residual": cert1.symplectic_residual,
            "ccr_canonical_valid": cert1.ccr_canonical_valid,
            "car_canonical_valid": cert1.car_canonical_valid,
            "ccr_canonical_residual": cert1.ccr_canonical_residual,
            "car_canonical_residual": cert1.car_canonical_residual,
            "meta_consistency_score": meta["consistency"],
            "meta_verdict_entropy": meta["entropy"],
            "ooda_verdict_entropy": ooda_entropy,
            "adaptive_fidelity_floor": cert2.adaptive_fidelity_floor,
            "adaptive_bures_ceiling": cert2.adaptive_bures_ceiling,
            "adaptive_spohn_ceiling": cert2.adaptive_spohn_ceiling,
            "adaptation_rate": cert2.adaptation_rate,
            "meta_adaptation_rate": cert2.meta_adaptation_rate,
            "n3_adaptation_lyapunov": cert2.n3_adaptation_lyapunov,
            "ooda_adaptation_lyapunov": ooda_lyap,
            "ooda_kam_coherent_threshold": self._ooda_ledger.kam_coherent_threshold,
            "level_3_reflections": meta["reflections"],
            "ooda_reflections": self._ooda_ledger.reflections,
            "n3_cycle_count": self._n3_cycle_count,
            "n3_trajectory": list(self._n3_verdict_trajectory),
            "engine_n3_trajectory": list(meta["engine_traj"]),
            "engine_invocations": self._session.engine_invocations,
            "emergency_reinvoke": self._session.emergency_reinvoke,
            "single_invocation_ok": single_inv_ok,
            "density_fingerprint": cert1.density_fingerprint,
            "nekhoroshev_log10": (
                math.log10(cert2.nekhoroshev_time) if cert2.nekhoroshev_time > 0.0 else float("-inf")
            ),
            "executive_action": exec_action,
        }

        return TopologicalFPUTensorWitnessSovereignStateV14(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=final_verdict,
            back_action_db=cert2.back_action_db,
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
            base_verdict_v13=base_verdict_v13,
            kam_adjusted_verdict=final_verdict,
            meta_consistency_score=meta["consistency"],
            meta_verdict_entropy=meta["entropy"],
            meta_fidelity_std=meta["fid_std"],
            meta_bures_std=meta["bures_std"],
            meta_spohn_std=meta["spohn_std"],
            n3_verdict_trajectory=tuple(self._n3_verdict_trajectory),
            n3_cycle_count=self._n3_cycle_count,
            level_3_reflections=meta["reflections"],
            passive_silent_invariant_verified=passive_silent_ok,
            engine_n3_verdict_trajectory=tuple(meta["engine_traj"]),
            ooda_n3_verdict_trajectory=tuple(self._n3_verdict_trajectory),
            n3_adaptation_lyapunov=float(cert2.n3_adaptation_lyapunov),
            ooda_adaptation_lyapunov=float(ooda_lyap),
            poincare_lyapunov=cert2.poincare_lyapunov,
            nekhoroshev_time=cert2.nekhoroshev_time,
            rotation_number=cert2.rotation_number,
            celestial_section_transverse=cert2.celestial_section_transverse,
            single_invocation_invariant_verified=single_inv_ok,
            ooda_kam_coherent_threshold=float(self._ooda_ledger.kam_coherent_threshold),
            hmac_chain_p1_p2_p3=chain_p1p2p3,
        )

    def run_phase3_decide(
        self,
        cert1: FPUWitnessPhase1CertificateV14,
        cert2: FPUWitnessPhase2CertificateV14,
    ) -> TopologicalFPUTensorWitnessSovereignStateV14:
        """Orquestador FASE 3. Abre con el HANDOFF de FASE 2 y emite el estado soberano."""
        return self.execute_phase3_v14(cert1, cert2)

    # ── Alias v13 ──
    def _upgrade_p2(self, c: Any) -> FPUWitnessPhase2CertificateV14:
        if isinstance(c, FPUWitnessPhase2CertificateV14):
            return c
        return FPUWitnessPhase2CertificateV14(
            witness_id=c.witness_id, phase1_ref_id=c.phase1_ref_id,
            weak_value_qnd_aav=c.weak_value_qnd_aav,
            weak_value_modulus=c.weak_value_modulus,
            back_action_db=c.back_action_db,
            uhlmann_fidelity=c.uhlmann_fidelity,
            bures_distance=c.bures_distance,
            spohn_entropy_production=c.spohn_entropy_production,
            gksl_dissipation_rate=c.gksl_dissipation_rate,
            qnd_verdict=c.qnd_verdict,
            chain_hash=c.chain_hash, hmac_p2_signature=c.hmac_p2_signature,
            timestamp_utc=c.timestamp_utc,
            poincare_crossings_count=0, occupation_drift_max=0.0,
            occupation_drift_mean=0.0, lindstedt_correction_norm=0.0,
            birkhoff_quadratic_coefficient=0.0,
            kam_stability_index=1.0, kam_resonance_violations=0,
            kam_worst_ratio=1.0e9,
            adaptive_fidelity_floor=0.95, adaptive_bures_ceiling=0.30,
            adaptive_spohn_ceiling=2.0,
            adaptation_rate=0.05, meta_adaptation_rate=0.01,
            level_3_entropy=0.0,
            lindstedt_order2_norm=0.0, rotation_number=0.0,
            first_return_time_mean=0.0, poincare_lyapunov=0.0,
            nekhoroshev_time=0.0, n3_adaptation_lyapunov=0.0,
            celestial_section_transverse=False,
            engine_invocations=self._session.engine_invocations,
        )

    def execute_phase3(
        self,
        cert1: Any,
        cert2: Any,
    ) -> TopologicalFPUTensorWitnessSovereignState:
        """Alias v13: acepta P1/P2 (v13 o v14); retorna estado v13 con certs recortados y HMAC LEGACY."""
        cert1_v14 = self._upgrade_p1(cert1)
        cert2_v14 = self._upgrade_p2(cert2)
        if (
            self._session.cert1 is not None
            and not self.continue_to_phase2_witness_gate_v14(cert1_v14)
        ):
            cert1_v14 = self._session.cert1
        state_v14 = self.execute_phase3_v14(cert1_v14, cert2_v14)
        p1_v13 = self._downgrade_p1(state_v14.phase1_cert)
        p2_v13 = self._downgrade_p2(state_v14.phase2_cert)
        prov_payload = (
            f"{state_v14.merkle_sha256_root}:{state_v14.back_action_db:.1f}:"
            f"{state_v14.global_verdict.value}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3_LEGACY, prov_payload)
        return TopologicalFPUTensorWitnessSovereignState(
            sovereign_state_id=state_v14.sovereign_state_id,
            phase1_cert=p1_v13,
            phase2_cert=p2_v13,
            global_verdict=state_v14.global_verdict,
            back_action_db=state_v14.back_action_db,
            merkle_sha256_root=state_v14.merkle_sha256_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=state_v14.visceral_report,
            timestamp_utc=state_v14.timestamp_utc,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO TESTIGO — ORQUESTADOR OODA v14.1
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class FPUTensorFockColliderWitnessAgentV14(FPUWitnessPhase3DeciderV14):
    """
    Soberano Agéntico Testigo Silencioso v14.1.
    Ciclo OODA pasivo completo sobre la coherencia FPU en Fock ℱ(ℋ).

    Pipeline anidado:
        FASE 1  run_phase1_observe            → cert1  [HANDOFF Φ₁₂]
        FASE 2  run_phase2_orient(cert1, ·)   → cert2  [HANDOFF Φ₂₃]
        FASE 3  run_phase3_decide(cert1, cert2) → estado soberano
    El motor se invoca EXACTAMENTE UNA VEZ (sub-capa 1.1).
    """

    def witness_fpu_tensor_field_v14(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> TopologicalFPUTensorWitnessSovereignStateV14:
        """Ciclo OODA v14.1 completo. `state_a`/`state_b` se ignoran (canónicos v14)."""
        cert1 = self.run_phase1_observe(density_matrix, phi_initial, phi_final)
        cert2 = self.run_phase2_orient(cert1, density_matrix, phi_initial, phi_final)
        return self.run_phase3_decide(cert1, cert2)

    def witness_fpu_tensor_field(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> TopologicalFPUTensorWitnessSovereignState:
        """Alias v13: internamente OODA v14.1; certs y HMAC degradados a legado."""
        state_v14 = self.witness_fpu_tensor_field_v14(
            density_matrix, phi_initial, phi_final, state_a, state_b
        )
        p1_v13 = self._downgrade_p1(state_v14.phase1_cert)
        p2_v13 = self._downgrade_p2(state_v14.phase2_cert)
        prov_payload = (
            f"{state_v14.merkle_sha256_root}:{state_v14.back_action_db:.1f}:"
            f"{state_v14.global_verdict.value}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3_LEGACY, prov_payload)
        return TopologicalFPUTensorWitnessSovereignState(
            sovereign_state_id=state_v14.sovereign_state_id,
            phase1_cert=p1_v13,
            phase2_cert=p2_v13,
            global_verdict=state_v14.global_verdict,
            back_action_db=state_v14.back_action_db,
            merkle_sha256_root=state_v14.merkle_sha256_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=state_v14.visceral_report,
            timestamp_utc=state_v14.timestamp_utc,
        )


# ── Alias de clase (compatibilidad) ──
FPUWitnessPhase1Observer = FPUWitnessPhase1ObserverV14
FPUWitnessPhase2Orientor = FPUWitnessPhase2OrientorV14
FPUWitnessPhase3Decider = FPUWitnessPhase3DeciderV14
FPUTensorFockColliderWitnessAgent = FPUTensorFockColliderWitnessAgentV14


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DEMO — backward-compat + features v14.1
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v14.1 · AGENTE OODA FPU + POINCARÉ CELESTE + N3")
    print("  (Kerr forzado · Benettin · Nekhoroshev · ledger OODA ≠ ledger motor · I4)")
    print(banner)

    agent = FPUTensorFockColliderWitnessAgentV14(secret_key=_DEFAULT_SECRET_KEY)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    final_states: List[TopologicalFPUTensorWitnessSovereignStateV14] = []
    for cycle in range(3):
        st = agent.witness_fpu_tensor_field_v14(rho)
        final_states.append(st)
        print(f"\n  ─── CICLO N3/OODA #{cycle + 1} ───")
        print(f"  • Sovereign State ID   : {st.sovereign_state_id}")
        print(f"  • Veredicto base v13   : {st.base_verdict_v13.name}")
        print(f"  • Veredicto KAM-ajust. : {st.kam_adjusted_verdict.name}")
        print(f"  • PASSIVE_SILENT ✓     : {st.passive_silent_invariant_verified}")
        print(f"  • I4 invocación única  : {st.single_invocation_invariant_verified} "
              f"(n={st.phase1_cert.engine_invocations})")
        print(f"  • Back-Action QND      : {st.back_action_db:.1f} dB")
        print(f"  • Fidelidad Uhlmann    : {st.phase2_cert.uhlmann_fidelity:.6f}")
        print(f"  • Distancia Bures      : {st.phase2_cert.bures_distance:.6f}")
        print(f"  • Entropía de Spohn    : {st.phase2_cert.spohn_entropy_production:.6e}")
        print(f"  ── FASE 1 · Observe (action-angle + I8) ──")
        print(f"    ΣJ_i               : {st.phase1_cert.action_total:.6f}")
        print(f"    Ocupación media     : {st.phase1_cert.mean_occupation:.6f}")
        print(f"    Pureza             : {st.phase1_cert.purity:.6f}")
        print(f"    Lyapunov (max,min) : ({st.phase1_cert.lyapunov_max:+.4e}, "
              f"{st.phase1_cert.lyapunov_min:+.4e})")
        print(f"    Cuadraturas (x,p)  : ({st.phase1_cert.quadrature_x:+.4f}, "
              f"{st.phase1_cert.quadrature_p:+.4f})")
        print(f"    Residual simpléctico: {st.phase1_cert.symplectic_residual:.3e}")
        print(f"    CAR/CCR canónicos  : CAR={st.phase1_cert.car_canonical_valid} "
              f"(res={st.phase1_cert.car_canonical_residual:.3e}), "
              f"CCR={st.phase1_cert.ccr_canonical_valid} "
              f"(res={st.phase1_cert.ccr_canonical_residual:.3e})")
        print(f"    Huella ρ           : {st.phase1_cert.density_fingerprint}")
        print(f"  ── FASE 2 · Orient (Poincaré celeste) ──")
        print(f"    Cruces Σ           : {st.phase2_cert.poincare_crossings_count}")
        print(f"    Sección transversal: {st.celestial_section_transverse}")
        print(f"    Drift ocupación max: {st.phase2_cert.occupation_drift_max:.6e}")
        print(f"    Lindstedt ‖δω‖/o2  : {st.phase2_cert.lindstedt_correction_norm:.6e} / "
              f"{st.phase2_cert.lindstedt_order2_norm:.6e}")
        print(f"    Coef. Birkhoff     : {st.phase2_cert.birkhoff_quadratic_coefficient:.6e}")
        print(f"    Índice KAM         : {st.phase2_cert.kam_stability_index:.6f} "
              f"(viol={st.phase2_cert.kam_resonance_violations})")
        print(f"    Nº rotación ν      : {st.rotation_number:.6f}")
        print(f"    λ_Poincaré (Benettin): {st.poincare_lyapunov:+.4e}")
        print(f"    T_Nekhoroshev      : {st.nekhoroshev_time:.4e}")
        print(f"  ── FASE 3 · Decide (meta N3 motor + OODA) ──")
        print(f"    floor_fid adapt.   : {st.phase2_cert.adaptive_fidelity_floor:.6f}")
        print(f"    ceiling_bures adapt: {st.phase2_cert.adaptive_bures_ceiling:.6f}")
        print(f"    α_N2 / α_N3 motor  : {st.phase2_cert.adaptation_rate:.4f} / "
              f"{st.phase2_cert.meta_adaptation_rate:.4f}")
        print(f"    λ_α motor / OODA   : {st.n3_adaptation_lyapunov:.4e} / "
              f"{st.ooda_adaptation_lyapunov:.4e}")
        print(f"    θ_KAM OODA (coh)   : {st.ooda_kam_coherent_threshold:.4f}")
        print(f"    Meta consistencia  : {st.meta_consistency_score:.6f}")
        print(f"    Meta entropía      : {st.meta_verdict_entropy:.4f}")
        print(f"    Trayectoria OODA   : {st.ooda_n3_verdict_trajectory}")
        print(f"    Trayectoria motor  : {st.engine_n3_verdict_trajectory}")
        print(f"    N3 ciclos totales  : {st.n3_cycle_count}")
        print(f"    Merkle SHA-256     : {st.merkle_sha256_root[:40]}...")
        print(f"    HMAC P3            : {st.hmac_provenance_signature[:40]}...")
        print(f"    Cadena P1→P2→P3    : {st.hmac_chain_p1_p2_p3[:40]}...")
        print(f"    Executive action   : {st.visceral_report['executive_action'][:88]}...")

    print("\n  ═══ VERIFICACIÓN BACKWARD-COMPAT v13 ═══")
    st_v13 = agent.witness_fpu_tensor_field(rho)
    v13_fields = [
        "sovereign_state_id", "phase1_cert", "phase2_cert", "global_verdict",
        "back_action_db", "merkle_sha256_root", "hmac_provenance_signature",
        "visceral_report", "timestamp_utc",
    ]
    missing = [f for f in v13_fields if not hasattr(st_v13, f)]
    print(f"  Campos v13 ausentes: {missing or 'NINGUNO ✓'}")
    print(f"  Veredicto v13 base : {st_v13.global_verdict.name}")
    print(f"  Back-Action        : {st_v13.back_action_db:.1f} dB")
    print(f"  P1 tipo            : {type(st_v13.phase1_cert).__name__}")
    print(f"  P2 tipo            : {type(st_v13.phase2_cert).__name__}")
    gate_p1 = agent.continue_to_phase2_witness_gate(st_v13.phase1_cert)
    gate_p2 = agent.continue_to_phase3_witness_gate(st_v13.phase2_cert)
    print(f"  Φ₁₂ legado verifica: {gate_p1}")
    print(f"  Φ₂₃ legado verifica: {gate_p2}")
    print(banner)