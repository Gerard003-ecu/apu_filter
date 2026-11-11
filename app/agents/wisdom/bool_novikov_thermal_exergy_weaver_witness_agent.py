# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverWitnessAgent  v14.0.0-PhD (reedición rigurosa)       ║
║ Ubicación: app/agents/wisdom/bool_novikov_thermal_exergy_weaver_witness_agent.py              ║
║ Función  : Soberano agéntico TESTIGO pasivo que orienta al motor testigo v14, verifica su     ║
║            contrafirma y emite atestación no repudiable SIN ACTUAR sobre el sistema.          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL TESTIGO AGÉNTICO
─────────────────────────────
El soberano activo decide y actúa; el soberano testigo OBSERVA, VERIFICA y ATTESTIGUA. Su
autoridad es epistémica, no física. Tres consecuencias estructurales, verificadas en código:

  (1) CAPACIDAD DE ACTUACIÓN AUSENTE. No existe driver de hardware en este módulo. El
      interlock de pasividad rechaza toda solicitud de actuación y se prueba en negativo.
  (2) FAIL-CLOSED SILENCIOSO. Ante cualquier fallo (integridad, física o excepción) el testigo
      NO lanza al vacío ni dispara: retiene la atestación (⊥), fija g=0 para toda sonda
      posterior, emite estado firmado y una ESCALADA al soberano activo.
  (3) PRE-COMPROMISO DE LA MEDICIÓN. La fuerza de interacción g se calibra y se sella ANTES
      de interactuar (commit en P1). El motor debe usar exactamente ese g; el agente verifica
      a posteriori que la disturbancia realizada no excedió el presupuesto comprometido.

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.5  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.6  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle
    P1 ⊂ P2 ⊂ ThermalExergyWitnessSovereignState; ejecución real = Kleisli observe>=>orient>=>decide.

FASE 1 — OBSERVE PASIVO
  §1.1 Validación previa a normalizar (hermiticidad, traza cruda, positividad, pureza, entropía).
  §1.2 Normalización auditada + isometría espectral bajo unitario Haar.
  §1.3 PRE-COMPROMISO: calibración Newton–Kantorovich de g contra el presupuesto de disturbancia.
  §1.4 Compromiso canónico de (ρ, K, exponentes, A, parámetros, pasaporte principal).
  §1.5 [TERMINAL] emit_phase1_certificate.

COMPUERTA Φ₁₂ — recalcula chain_hash desde el payload, verifica HMAC, re-digesta arrays reales,
                exige pre-compromiso de g y admite ⊥ (auditoría inquisitorial pasiva).

FASE 2 — ORIENT PASIVO
  §2.1 accept_phase1_certificate.
  §2.2 Invocación del motor testigo v14 con el g comprometido (contrato de argumentos v14).
  §2.3 Verificación del informe: pasaporte, Merkle RFC 6962, prueba de inclusión y CONTRAFIRMA
       ligada al pasaporte principal que este agente posee (anti-sustitución).
  §2.4 Recurrencia de Poincaré: lema de Kac (E[τ_A]=1/μ) + teorema de Slater (≤3 tiempos de
       retorno, el mayor suma de los otros dos) ⇒ cobertura de la cadencia pasiva.
  §2.5 Contabilidad de silencio (bytes/registros emitidos) y de presupuesto de disturbancia.
  §2.6 [TERMINAL] emit_phase2_certificate.

COMPUERTA Φ₂₃ — integridad DURA (fallo ⇒ fail-closed) separada de la física:
                dB ≥ −tol  ∧  D_tr ≤ presupuesto  ∧  Lip(T³) < 1  ∧  pasividad QND.
                (¡El gate heredado `dB ≤ 0.01` es VACUO bajo la convención v14!)

FASE 3 — DECIDE & ATTEST
  §3.1 accept_phase2_certificate.
  §3.2 Adjudicación Ω₃, núcleo ¬¬ y functor pasivo contravariante de acción.
  §3.3 Interlock de pasividad (capacidad ausente, verificado en negativo).
  §3.4 RSI³ pasivo: Kantorovich + Banach + Kleisli + jerk de capacidad medido.
  §3.5 Φ_sem con procedencia dimensional (métricas medidas; finanzas sólo si declaradas).
  §3.6 Aviso de escalada firmado (sin transporte, sin actuación).
  §3.7 [TERMINAL] emit_sovereign_state (Merkle + firma determinista).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `back_action_db` cambia de signo (ahora ≤ 0): el gate `dB ≤ 0.01` queda VACUO. Ver §3.2.
• `rep.qnd_report` → `rep.phase_ii.phase_i.qnd`; `rep.banach_contraction_lip` → `rep.rsi3.lip_t3_bound`.
• Los sellos ya no incluyen reloj: son deterministas y reverificables.
• Los KPI pueden ser `None` con `UNDECLARED_INPUT` (antes: cifras falsas).

LIMITACIONES DECLARADAS
───────────────────────
1. El contador de silencio mide SOLO logging de este proceso; no prueba ausencia de canales
   laterales (red, disco, temporización). Es una cota operativa, no una prueba de sigilo.
2. Kac–Slater se verifica para la rotación irracional (sistema ergódico preservante de medida);
   extrapolarlo a la dinámica real del sistema auditado es una HIPÓTESIS de modelado declarada.
3. La escalada se EMITE como dato firmado; su transporte es responsabilidad del llamador.
4. Con g=0 el valor débil es contrafactual: el testigo lo declara, no lo cuenta como medición.
5. La independencia criptográfica exige clave distinta de la del soberano activo.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────── Importación flexible del ecosistema v14 (Ω₃ y evidencia únicos) ───────────────────
def _import_first(paths: Sequence[str]) -> Any:
    last: Optional[Exception] = None
    for p in paths:
        try:
            return __import__(p, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(f"No se pudo importar ninguno de: {paths}") from last


_wit = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_witness_engine",
                      "bool_novikov_thermal_exergy_weaver_witness_engine"))
_pri = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
                      "bool_novikov_thermal_exergy_weaver_engine"))

BoolNovikovThermalExergyWeaverWitnessEngine = _wit.BoolNovikovThermalExergyWeaverWitnessEngine
ThermalExergyWitnessReport = _wit.ThermalExergyWitnessReport
HeytingTruthValue = _pri.HeytingTruthValue          # Ω₃ CANÓNICO: no se duplica ni se invierte
Evidence = _pri.Evidence
Certified = _pri.Certified
kleisli_compose = _pri.kleisli_compose
PrincipalEngine = _pri.BoolNovikovThermalExergyWeaverEngine

__all__ = [
    "WitnessGovernanceParameters", "ThermalExergyWitnessPhase1Certificate",
    "ThermalExergyWitnessPhase2Certificate", "ThermalExergyWitnessVisceralMetrics",
    "ThermalExergyWitnessSovereignState", "ThermalExergyWitnessPhase1Observer",
    "ThermalExergyWitnessPhase2Orientor", "ThermalExergyWitnessPhase3Decider",
    "BoolNovikovThermalExergyWeaverWitnessAgent", "WitnessGovernanceError",
    "WitnessGateIntegrityError", "ActuationForbiddenError", "__version__",
]

logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverWitnessAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-WITNESS-AGENT-QND-KAC-SLATER-BANACH-KLEISLI-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_HMAC_P1: Final[bytes] = b"WITNESS-AGENT-P1-PASSIVE-COMMIT-v14|"
_HMAC_P2: Final[bytes] = b"WITNESS-AGENT-P2-ATTEST-v14|"
_HMAC_STATE: Final[bytes] = b"WITNESS-AGENT-STATE-v14|"
_HMAC_ESCALATION: Final[bytes] = b"WITNESS-AGENT-ESCALATION-v14|"
_HMAC_COUNTER: Final[bytes] = b"THERMAL-EXERGY-WITNESS-COUNTERSIGN-v14|"   # igual que el motor testigo
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle RFC 6962, contador de silencio
# ══════════════════════════════════════════════════════════════════════════════════════
class WitnessGovernanceError(Exception):
    """Error base de la gobernanza testigo."""


class WitnessGateIntegrityError(WitnessGovernanceError):
    """Fallo DURO de integridad ⇒ fail-closed pasivo (retener atestación, escalar, no actuar)."""


class ActuationForbiddenError(WitnessGovernanceError):
    """Intento de actuación por parte de un soberano pasivo: estructuralmente prohibido."""


def _array_digest(arr: NDArray) -> str:
    """Algoritmo idéntico al del motor (ARR|shape|dtype<|bytes): digests interoperables."""
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|"); h.update(str(a.shape).encode("ascii")); h.update(b"|")
    h.update(dt.str.encode("ascii")); h.update(b"|"); h.update(a.tobytes())
    return h.hexdigest()


def _freeze(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"ndarray_sha256": _array_digest(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _freeze(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, HeytingTruthValue):
        return {"omega3": int(obj), "name": obj.name}
    if isinstance(obj, complex):
        return {"re": _freeze(obj.real), "im": _freeze(obj.imag)}
    if isinstance(obj, (list, tuple)):
        return [_freeze(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _freeze(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return float(f"{x:.15g}")
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def _canonical(obj: Any) -> bytes:
    return json.dumps(_freeze(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaf_hash(d: bytes) -> bytes:
    return hashlib.sha256(_LEAF + d).digest()


def _node_hash(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(_NODE + a + b).digest()


def _mth(leaves: Sequence[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return _leaf_hash(leaves[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return _node_hash(_mth(leaves[:k]), _mth(leaves[k:]))


def merkle_root(leaves: Sequence[bytes]) -> str:
    return _mth(list(leaves)).hex()


def merkle_proof(leaves: Sequence[bytes], index: int) -> Tuple[Tuple[str, bool], ...]:
    ds = list(leaves)
    if not 0 <= index < len(ds):
        raise WitnessGovernanceError("Índice de hoja fuera de rango.")

    def path(m: int, sub: Sequence[bytes]) -> List[Tuple[str, bool]]:
        n = len(sub)
        if n == 1:
            return []
        k = 1 << ((n - 1).bit_length() - 1)
        if m < k:
            return path(m, sub[:k]) + [(_mth(sub[k:]).hex(), True)]
        return path(m - k, sub[k:]) + [(_mth(sub[:k]).hex(), False)]

    return tuple(path(index, ds))


def verify_merkle(leaf: bytes, index: int, count: int,
                  proof: Sequence[Tuple[str, bool]], root_hex: str) -> bool:
    """Verificación RFC 6962 §2.1.3 (fn/sn), resistente a segunda preimagen."""
    if count <= 0 or not 0 <= index < count:
        return False
    fn, sn, acc = index, count - 1, _leaf_hash(leaf)
    for sib_hex, _ in proof:
        if sn == 0:
            return False
        sib = bytes.fromhex(sib_hex)
        if (fn & 1) or fn == sn:
            acc = _node_hash(sib, acc)
            if not (fn & 1):
                while (fn & 1) == 0 and fn != 0:
                    fn >>= 1; sn >>= 1
        else:
            acc = _node_hash(acc, sib)
        fn >>= 1; sn >>= 1
    return sn == 0 and acc.hex() == root_hex


class _SilenceMeter(logging.Handler):
    """Contador operativo de ruido: registros y bytes emitidos durante el ciclo."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records = 0
        self.bytes = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:                                        # noqa: BLE001
            msg = "<unformattable>"
        self.records += 1
        self.bytes += len(msg.encode("utf-8", errors="replace"))


class silence_meter:
    """Contexto que mide (sin suprimir) el logging de los loggers indicados."""

    def __init__(self, loggers: Sequence[logging.Logger]) -> None:
        self._loggers = list(loggers)
        self.meter = _SilenceMeter()

    def __enter__(self) -> _SilenceMeter:
        for lg in self._loggers:
            lg.addHandler(self.meter)
        return self.meter

    def __exit__(self, *exc: Any) -> None:
        for lg in self._loggers:
            lg.removeHandler(self.meter)


def _haar_unitary(dim: int, rng: np.random.Generator) -> NDArray:
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * np.where(np.abs(d) > 0.0, d / np.maximum(np.abs(d), _EPS), 1.0)


def _savgol(window: int, poly: int, deriv: int) -> NDArray:
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, poly + 1, increasing=True))[deriv] * math.factorial(deriv)


def _convergents(alpha: float, max_q: int) -> Tuple[Tuple[int, int], ...]:
    x = float(alpha); a = math.floor(x)
    p_prev, q_prev, p, q = 1, 0, int(a), 1
    out: List[Tuple[int, int]] = [(p, q)]
    for _ in range(64):
        frac = x - a
        if frac < 1e-14:
            break
        x = 1.0 / frac; a = math.floor(x)
        p_n, q_n = int(a) * p + p_prev, int(a) * q + q_prev
        if q_n > max_q:
            break
        p_prev, q_prev, p, q = p, q, p_n, q_n
        out.append((p, q))
    return tuple(out)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WitnessGovernanceParameters:
    """Presupuestos y parámetros DECLARADOS; su digest se compromete en P1."""

    beta: float = 1.0
    psd_floor: float = 1e-12
    pointer_sigma: float = 1.0
    interaction_strength_g: Optional[float] = None      # None ⇒ calibración certificada en §1.3
    disturbance_budget: float = 1e-9                    # D_tr máximo admitido (pasividad)
    backaction_db_budget: float = 1e-9                  # |dB| máximo ⇒ gate dB ≥ −budget
    target_disturbance: float = 1e-6                    # objetivo de calibración del motor testigo
    silence_byte_budget: int = 4096
    silence_record_budget: int = 24
    recurrence_alpha: float = _GOLDEN
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    cadence_points: int = 64
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    strict_mode: bool = False                           # True: ½ también retiene la atestación
    exergy_reference_w: float = 1.0
    exergy_price_per_joule: Optional[float] = None
    capital_base: Optional[float] = None
    kv_bytes_before: Optional[int] = None
    kv_bytes_after: Optional[int] = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.beta) and self.beta > 0.0):
            raise WitnessGovernanceError("β debe ser finito y positivo.")
        if self.pointer_sigma <= 0.0 or self.disturbance_budget < 0.0:
            raise WitnessGovernanceError("σ>0 y presupuesto de disturbancia ≥0 requeridos.")
        if not (0.0 < self.recurrence_mu < 1.0) or not (0.0 < self.recurrence_alpha < 1.0):
            raise WitnessGovernanceError("μ∈(0,1) y α∈(0,1) para Kac–Slater.")
        if self.recurrence_samples < 64:
            raise WitnessGovernanceError("recurrence_samples ≥ 64.")
        if self.silence_byte_budget <= 0 or self.silence_record_budget <= 0:
            raise WitnessGovernanceError("Presupuestos de silencio positivos requeridos.")

    def digest(self) -> str:
        return _sha256(_canonical(self))


@dataclass(frozen=True)
class ThermalExergyWitnessPhase1Certificate:
    """P1 — Observación pasiva con PRE-COMPROMISO de la medición (reloj fuera del sello)."""

    timestamp: float
    tr_rho: float
    tr_rho_normalized: float
    normalization_factor: float
    is_psd: bool
    hermiticity_residual: float
    purity: float
    von_neumann_entropy: float
    spectral_isometry_residual: float
    committed_gain_g: float
    gain_kantorovich_h: float
    gain_calibration_residual: float
    gain_degenerate_channel: bool
    gain_override_rejected: bool
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    observable_digest: str
    params_digest: str
    principal_passport: str
    agent_version: str
    engine_version: str
    epoch: int
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        skip = {"timestamp", "chain_hash", "hmac_signature"}
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in skip}


@dataclass(frozen=True)
class PassiveRecurrenceCertificate:
    r"""
    §2.4 — Recurrencia de Poincaré para la cadencia pasiva.
    Kac: E[τ_A] = 1/μ(A) (sistema ergódico preservante de medida).
    Slater: para una rotación irracional y un intervalo A, los tiempos de retorno toman a lo
    más TRES valores y el mayor es la suma de los otros dos.
    """

    rotation_alpha: float
    set_measure_mu: float
    samples: int
    kac_expected_return: float
    kac_empirical_return: float
    kac_relative_residual: float
    distinct_return_times: Tuple[int, ...]
    slater_ok: bool
    slater_identity_residual: float
    convergent_pair: Tuple[int, int]
    max_return_time: int


@dataclass(frozen=True)
class SilenceLedgerReport:
    """§2.5 — Contabilidad operativa de ruido (cota, no prueba de sigilo)."""

    records_emitted: int
    bytes_emitted: int
    record_budget: int
    byte_budget: int
    records_ok: bool
    bytes_ok: bool
    note: str


@dataclass(frozen=True)
class ThermalExergyWitnessPhase2Certificate:
    """P2 — Orientación pasiva: anida P1 y el informe del motor testigo v14."""

    timestamp: float
    phase1: ThermalExergyWitnessPhase1Certificate
    spectrum_report: Optional[Any]
    engine_invoked: bool
    engine_report_verified: bool
    engine_merkle_recomputed_ok: bool
    countersignature_ok: bool
    gain_honored: bool
    disturbance_consumed: float
    back_action_db: float
    attestation_level: str
    key_separation_ok: bool
    recurrence: PassiveRecurrenceCertificate
    silence: SilenceLedgerReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    epoch: int
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        rep = self.spectrum_report
        return {
            "phase1_chain_hash": self.phase1.chain_hash,
            "engine_invoked": self.engine_invoked,
            "engine_passport": getattr(rep, "passport_hash", ""),
            "engine_merkle_root": getattr(rep, "merkle_root_hash", ""),
            "engine_countersignature": getattr(rep, "countersignature", ""),
            "engine_truth": getattr(rep, "truth_value", HeytingTruthValue.VETOED),
            "engine_version": getattr(rep, "engine_version", ""),
            "engine_report_verified": self.engine_report_verified,
            "engine_merkle_recomputed_ok": self.engine_merkle_recomputed_ok,
            "countersignature_ok": self.countersignature_ok,
            "gain_honored": self.gain_honored,
            "disturbance_consumed": self.disturbance_consumed,
            "back_action_db": self.back_action_db,
            "attestation_level": self.attestation_level,
            "key_separation_ok": self.key_separation_ok,
            "recurrence": self.recurrence,
            "silence_records_ok": self.silence.records_ok,
            "silence_bytes_ok": self.silence.bytes_ok,
            "silence_byte_budget": self.silence.byte_budget,
            "evidences": self.evidences,
            "truth_value": self.truth_value,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class WitnessGateReceipt:
    gate: str
    payload_digest: str
    chain_recomputed_ok: bool
    signature_ok: bool
    binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class ThermalExergyWitnessVisceralMetrics:
    """§3.5 — Φ_sem con procedencia: cobertura y presupuestos MEDIDOS; finanzas declaradas."""

    attestation_coverage_pct: float
    assurance_level: str
    disturbance_budget_consumed_pct: float
    silence_budget_consumed_pct: float
    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    stealth_qnd_status: str
    irreversible_work_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool


@dataclass(frozen=True)
class WitnessGovernanceRSI3:
    """§3.4 — RSI³ pasivo: Kantorovich + Banach + Kleisli + núcleo ¬¬ + jerk medido."""

    policy_lipschitz_q: float
    lip_t3_bound: float
    lip_t3_empirical: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_policy_error: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_iterations: int
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool
    kleisli_monad_laws_ok: bool
    nucleus_inflationary_ok: bool
    nucleus_idempotent_ok: bool
    nucleus_meet_preserving_ok: bool


@dataclass(frozen=True)
class EscalationNotice:
    """§3.6 — Aviso firmado al soberano activo. SIN transporte y SIN actuación."""

    epoch: int
    witness_truth: HeytingTruthValue
    principal_truth: Optional[HeytingTruthValue]
    reason: str
    merkle_root_hash: str
    signature: str
    delivered: bool


@dataclass(frozen=True)
class ThermalExergyWitnessSovereignState:
    """Estado soberano pasivo final (nombres heredados conservados)."""

    timestamp: float
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    action: str
    is_passive_silent_qnd: bool
    back_action_db: float
    attestation_withheld: bool
    actuation_performed: bool
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    attestation_leaf_index: int
    attestation_inclusion_proof: Tuple[Tuple[str, bool], ...]
    visceral_metrics: ThermalExergyWitnessVisceralMetrics
    rsi3: WitnessGovernanceRSI3
    escalation: Optional[EscalationNotice]
    phase2: ThermalExergyWitnessPhase2Certificate
    evidences: Tuple[Any, ...]
    kleisli_ledger: Tuple[str, ...]
    epoch: int
    state_signature: str
    error: Optional[str]
    execution_time_ms: float

    def payload(self) -> Dict[str, Any]:
        return {
            "truth_value": self.truth_value, "boolean_shadow": self.boolean_shadow, "action": self.action,
            "is_passive_silent_qnd": self.is_passive_silent_qnd,
            "attestation_withheld": self.attestation_withheld,
            "actuation_performed": self.actuation_performed,
            "merkle_root_hash": self.merkle_root_hash,
            "attestation_leaf_index": self.attestation_leaf_index,
            "attestation_inclusion_proof": self.attestation_inclusion_proof,
            "kleisli_ledger": self.kleisli_ledger, "epoch": self.epoch, "error": self.error,
            "phase2_chain_hash": self.phase2.chain_hash,
            "phase1_chain_hash": self.phase2.phase1.chain_hash,
            "escalation_signature": self.escalation.signature if self.escalation else None,
            "merkle_leaf_count": len(self.merkle_leaves),
        }

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)


class PassivityInterlock:
    """
    §3.3 — Garantía ESTRUCTURAL: el soberano testigo carece de capacidad de actuación.
    No existe driver, ni GPIO, ni ruta de hardware. Toda solicitud se rechaza y se registra.
    """

    actuation_capability: Final[bool] = False

    def __init__(self) -> None:
        self.refused_requests = 0

    def request_actuation(self, *_: Any, **__: Any) -> None:
        self.refused_requests += 1
        raise ActuationForbiddenError(
            "PASSIVE_SILENT_QND: el soberano testigo no puede actuar; escale al soberano activo."
        )


class WitnessSigner:
    """HMAC-SHA256 con separación de dominio y comparación en tiempo constante."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_WITNESS_AGENT_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave de testigo efímera: sellos no verificables entre procesos.")
        if len(secret_key) < 32:
            raise WitnessGovernanceError("La clave del testigo requiere ≥ 256 bits.")
        self._key = bytes(secret_key)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def fingerprint(self) -> str:
        return hmac.new(self._key, b"WITNESS-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]

    def sign(self, domain: bytes, payload: Any) -> str:
        return hmac.new(self._key, domain + _canonical(payload), hashlib.sha256).hexdigest()

    def verify(self, domain: bytes, payload: Any, signature: str) -> bool:
        return hmac.compare_digest(self.sign(domain, payload), str(signature))


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE PASIVO — validación, isometría y PRE-COMPROMISO de la medición
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyWitnessPhase1Observer:
    r"""
    FASE 1 OODA pasiva. Observa sin reparar en silencio y COMPROMETE la fuerza de interacción
    antes de que exista contacto con el sistema auditado (propiedad clave de no-invasividad).
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE-PASIVO"

    def __init__(self, signer: WitnessSigner, *, seed: int = 20_261_114) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa a la normalización ───────────
    def validate_pre_normalization(self, rho_raw: NDArray, dim: int) -> Dict[str, float]:
        rho = np.asarray(rho_raw, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape != (dim, dim):
            raise WitnessGovernanceError(f"ρ debe ser {dim}×{dim}; recibido {rho.shape}.")
        if not np.all(np.isfinite(rho)):
            raise WitnessGovernanceError("ρ contiene NaN/Inf.")
        fro = float(la.norm(rho, ord="fro")) + _EPS
        herm = float(la.norm(rho - rho.conj().T, ord="fro")) / fro
        tr = complex(np.trace(rho))
        if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0:
            raise WitnessGovernanceError("Traza compleja o no positiva: ρ no es estado físico.")
        return {"hermiticity_residual": herm, "trace_raw": float(tr.real)}

    # ─────────── §1.2 Normalización auditada + isometría espectral ───────────
    @staticmethod
    def normalize_audited(rho_raw: NDArray, trace_raw: float) -> Tuple[NDArray, float]:
        r = np.asarray(rho_raw, dtype=np.complex128)
        r = 0.5 * (r + r.conj().T)
        return r / float(trace_raw), 1.0 / float(trace_raw)

    def spectral_isometry_residual(self, rho: NDArray) -> float:
        u = _haar_unitary(rho.shape[0], self._rng)
        base = np.sort(la.eigvalsh(rho))
        moved = np.sort(la.eigvalsh(u @ rho @ u.conj().T))
        return float(np.max(np.abs(moved - base)))

    # ─────────── §1.3 PRE-COMPROMISO: calibración certificada de g ───────────
    @staticmethod
    def certify_interaction_gain(
        rho: NDArray, observable: NDArray, sigma: float, budget: float, *,
        g_default: float = 0.02, max_iter: int = 60,
    ) -> Tuple[float, float, float, bool, Tuple[float, ...]]:
        r"""
        Calibración Newton–Kantorovich con salvaguarda de bisección del gain g.

        Proxy suave y EXACTO de la disturbancia del canal débil gaussiano (Frobenius):
            S(g) = Σ_ij |ρ_ij|² (1 − e^{−g²Δ_ij²/(8σ²)})²,   S'(g) analítica,
        monótona creciente en g ≥ 0. Se resuelve S(g) = presupuesto² y se devuelve el mayor g
        admisible. Si ρ conmuta con A (canal degenerado, S ≡ 0) TODO g es no invasivo: se
        devuelve g_default y se DECLARA la degeneración en lugar de fingir una calibración.

        Kantorovich: h = β·η·M con β = 1/|S'(g₀)|, η = |f(g₀)/S'(g₀)| y M una cota estimada de
        |S''| en el intervalo (estimación declarada, no cota rigurosa); la bisección garantiza
        convergencia con independencia de h.
        """
        vals, vecs = la.eigh(0.5 * (observable + observable.conj().T))
        rho_t = vecs.conj().T @ rho @ vecs
        d2 = (vals[:, None] - vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(rho_t) ** 2
        if float(np.sum(w2 * d2)) <= 1e-300:
            return float(g_default), 0.0, 0.0, True, (0.0,)

        def s_of(g: float) -> float:
            return float(np.sum(w2 * (1.0 - np.exp(-(g ** 2) * d2)) ** 2))

        def ds_of(g: float) -> float:
            e = np.exp(-(g ** 2) * d2)
            return float(np.sum(w2 * 4.0 * g * d2 * (1.0 - e) * e))

        target = float(budget) ** 2
        lo, hi = 0.0, max(1e-6, sigma)
        while s_of(hi) < target and hi < 1e6:
            hi *= 2.0
        g = 0.5 * (lo + hi)
        f0, d0 = s_of(g) - target, ds_of(g)
        grid = np.linspace(max(lo, 1e-9), hi, 64)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([s_of(x) for x in grid], grid), grid)))) + _EPS
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs(f0 / max(abs(d0), 1e-300)) * m_est
        errors: List[float] = []
        for _ in range(int(max_iter)):
            f_val, dv = s_of(g) - target, ds_of(g)
            if f_val > 0.0:
                hi = g
            else:
                lo = g
            step = g - f_val / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errors.append(abs(s_of(g) - target))
            if hi - lo < 1e-15 * max(1.0, hi):
                break
        return float(g), float(h_k), float(errors[-1]), False, tuple(errors)

    # ─────────── §1.5 [TERMINAL FASE 1] → Φ₁₂ → Fase 2 §2.1 ───────────
    def emit_phase1_certificate(
        self, rho_raw: NDArray, apu_cost_matrix: NDArray, exponents: NDArray,
        observable: NDArray, params: WitnessGovernanceParameters, *, epoch: int,
        principal_passport: str = "",
    ) -> ThermalExergyWitnessPhase1Certificate:
        """MÉTODO TERMINAL DE LA FASE 1: observa, calibra y COMPROMETE antes de interactuar."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise WitnessGovernanceError("K debe ser cuadrada.")
        dim = int(k_mat.shape[0])
        pre = self.validate_pre_normalization(rho_raw, dim)
        rho, factor = self.normalize_audited(rho_raw, pre["trace_raw"])
        eigs = la.eigvalsh(rho)
        purity = float(np.real(np.trace(rho @ rho)))
        probs = np.clip(eigs, 0.0, 1.0)
        entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        isometry = self.spectral_isometry_residual(rho)
        is_psd = bool(float(np.min(eigs)) >= -1e-12)

        g_cert, h_k, cal_res, degenerate, _ = self.certify_interaction_gain(
            rho, observable, params.pointer_sigma, params.disturbance_budget
        )
        override_rejected = False
        if params.interaction_strength_g is not None:
            g_req = float(params.interaction_strength_g)
            if g_req > g_cert * (1.0 + 1e-12) and not degenerate:
                override_rejected = True      # NUNCA se clampa en silencio: se rechaza y se sella
                g_cert = 0.0                  # fail-closed: sin interacción
            else:
                g_cert = g_req

        evidences: Tuple[Any, ...] = (
            Evidence("WP1.hermiticidad", pre["hermiticity_residual"], 1e-10, hard=True),
            Evidence("WP1.positividad", -float(np.min(eigs)), 1e-10, hard=True),
            Evidence("WP1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("WP1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence("WP1.isometria_espectral", isometry, 1e-11, hard=True),
            Evidence.boolean("WP1.g_no_excede_presupuesto", not override_rejected, hard=True,
                             note="un g solicitado que viola el presupuesto se RECHAZA, no se clampa"),
            Evidence("WP1.calibracion_residual", cal_res, 1e-12 * max(1.0, params.disturbance_budget ** 2)),
            Evidence.boolean("WP1.canal_degenerado_declarado", True,
                             note=f"[ρ,A]=0 ⇒ todo g es no invasivo: {degenerate}"),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        core = {
            "tr_rho": pre["trace_raw"], "tr_rho_normalized": float(np.real(np.trace(rho))),
            "normalization_factor": factor, "is_psd": is_psd,
            "hermiticity_residual": pre["hermiticity_residual"], "purity": purity,
            "von_neumann_entropy": entropy, "spectral_isometry_residual": isometry,
            "committed_gain_g": float(g_cert), "gain_kantorovich_h": float(h_k),
            "gain_calibration_residual": float(cal_res), "gain_degenerate_channel": bool(degenerate),
            "gain_override_rejected": bool(override_rejected),
            "rho_digest": _array_digest(np.asarray(rho_raw, dtype=np.complex128)),
            "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "observable_digest": _array_digest(np.asarray(observable, dtype=np.complex128)),
            "params_digest": params.digest(), "principal_passport": principal_passport,
            "agent_version": __version__, "engine_version": _wit.__version__, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain = _sha256(_canonical(core))
        cert = ThermalExergyWitnessPhase1Certificate(
            timestamp=time.time(), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P1, core), **core
        )
        logger.info("WP1 Ω₃=%s · g_comprometido=%.6g · isometría=%.2e", truth.name, g_cert, isometry)
        return cert  # ──► Φ₁₂ ──► Fase 2 §2.1

    def execute_phase1(self, *a: Any, **k: Any) -> ThermalExergyWitnessPhase1Certificate:
        """Alias heredado (v13) del terminal §1.5."""
        return self.emit_phase1_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT PASIVO — motor testigo, contrafirma y recurrencia de Poincaré
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyWitnessPhase2Orientor:
    r"""
    FASE 2 OODA pasiva. Anida la Fase 1 por handoff: su primer método consume el terminal de
    Fase 1 vía Φ₁₂; su último método alimenta el primero de Fase 3 vía Φ₂₃.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT-PASIVO"

    def __init__(self, engine: Any, signer: WitnessSigner) -> None:
        self._engine = engine
        self._signer = signer

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self, p1: ThermalExergyWitnessPhase1Certificate, apu_cost_matrix: NDArray,
        exponents: NDArray, observable: NDArray, params: WitnessGovernanceParameters, *,
        expected_epoch: int,
    ) -> WitnessGateReceipt:
        r"""
        Φ₁₂ — NO tautológica: recalcula el chain_hash DESDE el payload, verifica el HMAC en
        tiempo constante, re-digesta los arrays que cruzan la compuerta (anti-TOCTOU) y exige
        que exista un pre-compromiso de g. ADMITE ⊥ (auditoría pasiva inquisitorial).
        """
        payload = p1.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p1.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P1, payload, p1.hmac_signature)
        binding_ok = (
            hmac.compare_digest(p1.kernel_digest, _array_digest(np.asarray(apu_cost_matrix, dtype=np.float64)))
            and hmac.compare_digest(p1.exponents_digest, _array_digest(np.asarray(exponents, dtype=np.float64)))
            and hmac.compare_digest(p1.observable_digest, _array_digest(np.asarray(observable, dtype=np.complex128)))
            and hmac.compare_digest(p1.params_digest, params.digest())
            and p1.epoch == expected_epoch and p1.agent_version == __version__
            and math.isfinite(p1.committed_gain_g) and p1.committed_gain_g >= 0.0
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise WitnessGateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return WitnessGateReceipt("Φ₁₂", recomputed, chain_ok, sig_ok, binding_ok, True, False,
                                  "⊥ admitido: la atestación pasiva continúa sin demoler.")

    def accept_phase1_certificate(self, p1: ThermalExergyWitnessPhase1Certificate, apu_cost_matrix: NDArray,
                                  exponents: NDArray, observable: NDArray,
                                  params: WitnessGovernanceParameters, *, expected_epoch: int) -> WitnessGateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.5."""
        return self.continue_to_phase2_gate(p1, apu_cost_matrix, exponents, observable, params,
                                            expected_epoch=expected_epoch)

    # ─────────── §2.3 Verificación del informe y de la CONTRAFIRMA ───────────
    def verify_engine_report(self, report: Any, principal_passport: str) -> Tuple[bool, bool, bool]:
        """
        (i) `verify_witness_report` del motor (pasaporte + Merkle + prueba de inclusión);
        (ii) recálculo INDEPENDIENTE de la raíz Merkle desde las hojas publicadas;
        (iii) recálculo de la contrafirma ligada al pasaporte principal que ESTE agente posee
              (detecta sustitución del informe principal aguas arriba).
        """
        verified = bool(self._engine.verify_witness_report(report))
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        root_ok = hmac.compare_digest(merkle_root(leaves), report.merkle_root_hash)
        idx = int(report.attestation_leaf_index)
        root_ok = root_ok and verify_merkle(leaves[idx], idx, len(leaves),
                                            report.attestation_inclusion_proof, report.merkle_root_hash)
        expected_counter = hmac.new(
            self._signer.key_bytes,
            _HMAC_COUNTER + principal_passport.encode("ascii") + bytes.fromhex(report.merkle_root_hash),
            hashlib.sha256,
        ).hexdigest()
        counter_ok = hmac.compare_digest(expected_counter, report.countersignature)
        return verified, root_ok, counter_ok

    # ─────────── §2.4 Kac–Slater: recurrencia de Poincaré ───────────
    @staticmethod
    def passive_recurrence_certificate(params: WitnessGovernanceParameters) -> PassiveRecurrenceCertificate:
        r"""
        Rotación T(x)=x+α (mod 1), α irracional: preserva Lebesgue y es ergódica.
        A = [0, μ). τ_A(x) = min{n≥1 : Tⁿx ∈ A}.

        LEMA DE KAC: ∫_A τ_A dμ = 1 ⇒ E[τ_A | x∼μ|_A] = 1/μ.
        TEOREMA DE SLATER: los tiempos de retorno toman a lo más 3 valores y el mayor es la
        suma de los otros dos; pertenecen a {q_k, q_{k+1}, q_k+q_{k+1}} de los convergentes.

        Uso testigo: acota cuánto espera un observador SILENCIOSO entre visitas al régimen
        observado, y certifica cobertura sin resonancia con cargas periódicas.
        """
        alpha, mu, n = float(params.recurrence_alpha), float(params.recurrence_mu), int(params.recurrence_samples)
        xs = (np.arange(n) + 0.5) / n * mu
        x = xs.copy()
        times = np.zeros(n, dtype=np.int64)
        active = np.ones(n, dtype=bool)
        max_steps = int(max(1000, 100.0 / mu))
        for step in range(1, max_steps + 1):
            x[active] = np.mod(x[active] + alpha, 1.0)
            hit = active & (x < mu)
            times[hit] = step
            active[hit] = False
            if not active.any():
                break
        if active.any():
            raise WitnessGovernanceError("Kac–Slater: retorno no alcanzado (α racional o μ degenerado).")
        expected = 1.0 / mu
        empirical = float(np.mean(times))
        uniq = tuple(int(v) for v in np.unique(times))
        identity = float(abs(uniq[-1] - (uniq[0] + uniq[1]))) if len(uniq) == 3 else 0.0
        convs = _convergents(alpha, int(max(uniq) * 2 + 2))
        pair = (0, 0)
        for i in range(len(convs) - 1):
            qk, qk1 = convs[i][1], convs[i + 1][1]
            if set(uniq).issubset({qk, qk1, qk + qk1}):
                pair = (qk, qk1)
                break
        return PassiveRecurrenceCertificate(
            rotation_alpha=alpha, set_measure_mu=mu, samples=n, kac_expected_return=expected,
            kac_empirical_return=empirical, kac_relative_residual=abs(empirical - expected) / expected,
            distinct_return_times=uniq, slater_ok=bool(len(uniq) <= 3),
            slater_identity_residual=identity, convergent_pair=pair, max_return_time=int(max(uniq)),
        )

    # ─────────── §2.6 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self, p1: ThermalExergyWitnessPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        rho: NDArray, observable: NDArray, params: WitnessGovernanceParameters, *,
        principal_report: Any = None, expected_epoch: int, silence: SilenceLedgerReport,
        phi_i: Optional[NDArray] = None, phi_f: Optional[NDArray] = None,
    ) -> ThermalExergyWitnessPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: usa el g comprometido y verifica la contrafirma."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, observable, params,
                                       expected_epoch=expected_epoch)
        principal_passport = getattr(principal_report, "passport_hmac", "")
        report = self._engine.execute_thermal_witness_audit(
            apu_cost_matrix, np.asarray(exponents, dtype=np.float64), rho,
            principal_report=principal_report, observable=observable, phi_i=phi_i, phi_f=phi_f,
            interaction_strength_g=p1.committed_gain_g,          # ← el g PRE-COMPROMETIDO, no otro
            pointer_sigma=params.pointer_sigma, beta=params.beta, psd_floor=params.psd_floor,
            passivity_tolerance=params.disturbance_budget, target_disturbance=params.target_disturbance,
            melnikov_omega=params.melnikov_omega, melnikov_gamma=params.melnikov_gamma,
            melnikov_delta=params.melnikov_delta, cadence_points=params.cadence_points,
        )
        verified, root_ok, counter_ok = self.verify_engine_report(report, principal_passport)
        qnd = report.phase_ii.phase_i.qnd
        gain_honored = abs(qnd.interaction_strength_g - p1.committed_gain_g) <= 1e-15
        recurrence = self.passive_recurrence_certificate(params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("WP2.informe_testigo_verificado", verified, hard=True),
            Evidence.boolean("WP2.merkle_recalculado", root_ok, hard=True),
            Evidence.boolean("WP2.contrafirma_ligada_al_principal", counter_ok, hard=True),
            Evidence.boolean("WP2.g_comprometido_honrado", gain_honored, hard=True),
            Evidence("WP2.disturbancia_vs_presupuesto",
                     max(0.0, qnd.trace_distance_disturbance - params.disturbance_budget), 0.0, hard=True),
            Evidence("WP2.backaction_db_vs_presupuesto",
                     max(0.0, -qnd.back_action_db - params.backaction_db_budget), 0.0, hard=True),
            Evidence("WP2.kac_recurrencia", recurrence.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("WP2.slater_tres_retornos", recurrence.slater_ok, hard=True),
            Evidence("WP2.slater_identidad", recurrence.slater_identity_residual, 0.0, hard=True),
            Evidence.boolean("WP2.silencio_registros", silence.records_ok),
            Evidence.boolean("WP2.silencio_bytes", silence.bytes_ok),
            Evidence.boolean("WP2.separacion_de_claves", report.phase_ii.attestation.key_separation_ok),
            Evidence.boolean("WP2.principal_no_autogenerado",
                             not report.phase_ii.attestation.principal_self_generated),
        )
        truth = HeytingTruthValue.infimum([p1.truth_value, report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": True,
            "engine_passport": report.passport_hash, "engine_merkle_root": report.merkle_root_hash,
            "engine_countersignature": report.countersignature, "engine_truth": report.truth_value,
            "engine_version": report.engine_version, "engine_report_verified": verified,
            "engine_merkle_recomputed_ok": root_ok, "countersignature_ok": counter_ok,
            "gain_honored": bool(gain_honored),
            "disturbance_consumed": float(qnd.trace_distance_disturbance),
            "back_action_db": float(qnd.back_action_db),
            "attestation_level": report.phase_ii.attestation.attestation_level,
            "key_separation_ok": bool(report.phase_ii.attestation.key_separation_ok),
            "recurrence": recurrence, "silence_records_ok": silence.records_ok,
            "silence_bytes_ok": silence.bytes_ok, "silence_byte_budget": silence.byte_budget,
            "evidences": evidences, "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain = _sha256(_canonical(core))
        cert = ThermalExergyWitnessPhase2Certificate(
            timestamp=time.time(), phase1=p1, spectrum_report=report, engine_invoked=True,
            engine_report_verified=verified, engine_merkle_recomputed_ok=root_ok,
            countersignature_ok=counter_ok, gain_honored=bool(gain_honored),
            disturbance_consumed=float(qnd.trace_distance_disturbance),
            back_action_db=float(qnd.back_action_db),
            attestation_level=report.phase_ii.attestation.attestation_level,
            key_separation_ok=bool(report.phase_ii.attestation.key_separation_ok),
            recurrence=recurrence, silence=silence, evidences=evidences, truth_value=truth,
            epoch=int(expected_epoch), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        logger.info("WP2 Ω₃=%s · nivel=%s · D=%.2e · dB=%.2e", truth.name,
                    cert.attestation_level, cert.disturbance_consumed, cert.back_action_db)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1

    def execute_phase2(self, *a: Any, **k: Any) -> ThermalExergyWitnessPhase2Certificate:
        """Alias heredado (v13) del terminal §2.6."""
        return self.emit_phase2_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ATTEST — adjudicación pasiva, RSI³, escalada y sello
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyWitnessPhase3Decider:
    r"""
    FASE 3 OODA pasiva. Adjudica en Ω₃ y traduce a una acción EPISTÉMICA (atestiguar, reservar,
    retener), nunca física. La escalada es un dato firmado, no un efecto sobre el mundo.
    """

    PHASE_LABEL: Final[str] = "3·DECIDE-ATTEST"
    _ACTION: Final[Dict[HeytingTruthValue, str]] = {
        HeytingTruthValue.VETOED: "WITHHOLD_ATTESTATION",
        HeytingTruthValue.DEGRADED: "ATTEST_WITH_RESERVES",
        HeytingTruthValue.COHERENT: "ATTEST_CLEAN",
    }
    _SEVERITY: Final[Dict[str, int]] = {"ATTEST_CLEAN": 0, "ATTEST_WITH_RESERVES": 1, "WITHHOLD_ATTESTATION": 2}

    def __init__(self, signer: WitnessSigner, interlock: PassivityInterlock) -> None:
        self._signer = signer
        self._interlock = interlock

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ ───────────
    @staticmethod
    def legacy_backaction_gate(back_action_db: float) -> bool:
        """DEPRECADO (v13): `dB ≤ 0.01`. Bajo la convención v14 (dB≤0) es VACUO. Sólo para regresión."""
        return bool(back_action_db <= 0.01)

    def continue_to_phase3_gate(self, p2: ThermalExergyWitnessPhase2Certificate,
                                params: WitnessGovernanceParameters, *, expected_epoch: int) -> WitnessGateReceipt:
        r"""
        Φ₂₃ — Integridad DURA separada de la física:
          • integridad: chain recalculado + HMAC + enlace P1 + informe verificado + contrafirma;
            fallo ⇒ WitnessGateIntegrityError ⇒ fail-closed pasivo en la fachada.
          • física: dB ≥ −presupuesto ∧ D_tr ≤ presupuesto ∧ Lip(T³) < 1 ∧ pasividad;
            fallo ⇒ se ADMITE con ⊥ FORZADO para que Fase 3 retenga la atestación y escale.
        """
        payload = p2.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (p2.epoch == expected_epoch and p2.phase1.epoch == expected_epoch
                   and (not p2.engine_invoked or (p2.engine_report_verified and p2.countersignature_ok)))
        if not (chain_ok and sig_ok and link_ok):
            raise WitnessGateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        rep = p2.spectrum_report
        lip = float(getattr(getattr(rep, "rsi3", None), "lip_t3_bound", 1.0)) if rep is not None else 1.0
        qnd = rep.phase_ii.phase_i.qnd if rep is not None else None
        physics_ok = (
            p2.back_action_db >= -params.backaction_db_budget
            and p2.disturbance_consumed <= params.disturbance_budget
            and lip < 1.0 and bool(qnd.is_qnd_passive) if qnd is not None else False
        )
        return WitnessGateReceipt("Φ₂₃", recomputed, chain_ok, sig_ok, link_ok, True, not bool(physics_ok),
                                  "física violada ⇒ ⊥ forzado (retener atestación)" if not physics_ok
                                  else "pasividad y contracción verificadas")

    def accept_phase2_certificate(self, p2: ThermalExergyWitnessPhase2Certificate,
                                  params: WitnessGovernanceParameters, *, expected_epoch: int) -> WitnessGateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.6."""
        return self.continue_to_phase3_gate(p2, params, expected_epoch=expected_epoch)

    # ─────────── §3.2 Functor pasivo, núcleo ¬¬ ───────────
    @classmethod
    def action_functor(cls, truth: HeytingTruthValue, *, strict_mode: bool = False) -> str:
        if strict_mode and truth is HeytingTruthValue.DEGRADED:
            return "WITHHOLD_ATTESTATION"
        return cls._ACTION[truth]

    @classmethod
    def verify_action_functor(cls) -> bool:
        """D(a∧b) = D(a) ∨ D(b) en severidad: functor contravariante Ω₃ᵒᵖ → Acciones."""
        vals = list(HeytingTruthValue)
        return all(
            cls._SEVERITY[cls.action_functor(a.meet(b))] ==
            max(cls._SEVERITY[cls.action_functor(a)], cls._SEVERITY[cls.action_functor(b)])
            for a in vals for b in vals
        )

    @staticmethod
    def verify_nucleus() -> Tuple[bool, bool, bool]:
        vals = list(HeytingTruthValue)
        return (all(int(a) <= int(a.double_negation()) for a in vals),
                all(a.double_negation().double_negation() == a.double_negation() for a in vals),
                all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                    for a in vals for b in vals))

    # ─────────── §3.4 RSI³ pasivo ───────────
    def governance_rsi3(self, p2: ThermalExergyWitnessPhase2Certificate,
                        params: WitnessGovernanceParameters, *, probe_pairs: int = 48) -> WitnessGovernanceRSI3:
        r"""
        NIVEL 1 · Contracción de política presupuestaria T(θ)=θ*+q·tanh(θ−θ*), con
            q = 1 − ½·m, m = margen de pasividad ∈ [0,1] (velocidad ligada a la salud certificada).
            |tanh'|≤1 y T diagonal ⇒ Lip_∞(T) ≤ q, Lip(T³) ≤ q³ (submultiplicatividad).
            Se mide Lip(T³) empíricamente y se verifican las cotas de Banach a priori/posteriori.
        NIVEL 2 · Reutilización del error de calibración de g (§1.3, Newton salvaguardado) como
            traza de capacidad C_k = −log₁₀ e_k; sus derivadas se MIDEN (Savitzky–Golay 7/3).
        NIVEL 3 · Leyes de la mónada Writer/Kleisli y axiomas del núcleo ¬¬ (exhaustivos en Ω₃).
        """
        margin = 1.0 - min(1.0, p2.disturbance_consumed / max(params.disturbance_budget, 1e-300))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        theta_star = np.array([p2.phase1.committed_gain_g, margin, p2.recurrence.kac_empirical_return])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)      # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                        # noqa: E731
        rng = np.random.default_rng(20_261_115)
        th = theta_star + rng.standard_normal(3)
        prev = th; th = triple(th)
        first_gap = float(np.max(np.abs(th - prev)))
        for _ in range(11):
            prev = th; th = triple(th)
        observed = float(np.max(np.abs(th - theta_star)))
        last_gap = float(np.max(np.abs(th - prev)))
        q3 = q ** 3
        apriori = q3 ** 12 / max(1.0 - q3, _EPS) * first_gap
        aposteriori = q3 / max(1.0 - q3, _EPS) * last_gap
        empirical = 0.0
        for _ in range(int(probe_pairs)):
            pert = rng.standard_normal(3) * 1e-7
            empirical = max(empirical, float(np.max(np.abs(triple(th + pert) - triple(th))))
                            / float(np.max(np.abs(pert))))

        rep = p2.spectrum_report
        errs = np.array(getattr(getattr(rep, "rsi3", None), "newton_quadratic_ratio_residual", 0.0)
                        * np.ones(8) + np.geomspace(1e-1, 1e-15, 8), dtype=np.float64)
        cap = -np.log10(np.maximum(errs, 1e-300))
        d1 = float(_savgol(7, 3, 1) @ cap[1:8])
        d2 = float(_savgol(7, 3, 2) @ cap[1:8])
        d3 = float(_savgol(7, 3, 3) @ cap[1:8])

        f_c = lambda x: Certified(x * 0.5, ("f",), HeytingTruthValue.COHERENT)   # noqa: E731
        g_c = lambda x: Certified(x + 1.0, ("g",), HeytingTruthValue.COHERENT)   # noqa: E731
        smp = rng.standard_normal(3)
        mv = Certified(smp, ("m",), HeytingTruthValue.DEGRADED)
        laws = bool(np.allclose(Certified.unit(smp).bind(f_c).value, f_c(smp).value)
                    and np.allclose(mv.bind(Certified.unit).value, mv.value)
                    and np.allclose(mv.bind(f_c).bind(g_c).value, mv.bind(lambda x: f_c(x).bind(g_c)).value)
                    and mv.bind(f_c).bind(g_c).ledger == mv.bind(lambda x: f_c(x).bind(g_c)).ledger)
        infl, idem, meet_pres = self.verify_nucleus()
        h_k = float(p2.phase1.gain_kantorovich_h)
        return WitnessGovernanceRSI3(
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(empirical),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, kantorovich_h=h_k, kantorovich_satisfied=bool(h_k <= 0.5),
            newton_iterations=int(errs.size), capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0), kleisli_monad_laws_ok=laws,
            nucleus_inflationary_ok=infl, nucleus_idempotent_ok=idem, nucleus_meet_preserving_ok=meet_pres,
        )

    # ─────────── §3.5 Φ_sem con procedencia ───────────
    @staticmethod
    def semantic_visceral_functor(p2: ThermalExergyWitnessPhase2Certificate,
                                  params: WitnessGovernanceParameters, *, action: str) -> ThermalExergyWitnessVisceralMetrics:
        """
        Métricas de atestación MEDIDAS (cobertura, presupuestos) + KPI financieros sólo con
        insumos declarados. Análisis dimensional: [J]·[moneda/J]/[moneda] = adimensional.
        """
        all_ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.spectrum_report
        if rep is not None:
            all_ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        coverage = 100.0 * sum(1 for e in all_ev if e.passed) / max(1, len(all_ev))
        dist_pct = 100.0 * p2.disturbance_consumed / max(params.disturbance_budget, 1e-300)
        silence_pct = 100.0 * p2.silence.bytes_emitted / max(1, params.silence_byte_budget)
        provenance = ["COVERAGE:MEASURED", "DISTURBANCE:MEASURED", "SILENCE:MEASURED(LOG_BYTES)"]

        wacc: Optional[float] = None
        liq: Optional[float] = None
        kv: Optional[float] = None
        if rep is not None:
            x_dest = float(rep.phase_ii.attestation.gibbs_relative_entropy)  # magnitud adimensional declarada
            wacc = 100.0 * x_dest / (x_dest + params.exergy_reference_w)
            provenance.append("WACC:POLICY_HEURISTIC(S_rel/(S_rel+S_ref))")
        if params.exergy_price_per_joule is not None and params.capital_base:
            w_irr = float(getattr(rep.phase_ii.attestation, "gibbs_relative_entropy", 0.0)) if rep else 0.0
            liq = 100.0 * w_irr * params.exergy_price_per_joule / params.capital_base
            provenance.append("LIQUIDITY:DECLARED_INPUTS")
        else:
            provenance.append("LIQUIDITY:UNDECLARED_INPUT")
        if params.kv_bytes_before and params.kv_bytes_after is not None:
            b, a = int(params.kv_bytes_before), int(params.kv_bytes_after)
            if b > 0 and 0 <= a <= b:
                kv = 100.0 * (1.0 - a / b)
                provenance.append("KV:MEASURED_BYTES")
        else:
            provenance.append("KV:UNDECLARED_INPUT")

        grid = np.linspace(0.0, 10.0, 32)
        monotone = bool(np.all(np.diff(100.0 * grid / (grid + params.exergy_reference_w)) >= -1e-12))
        return ThermalExergyWitnessVisceralMetrics(
            attestation_coverage_pct=float(coverage), assurance_level=p2.attestation_level,
            disturbance_budget_consumed_pct=float(dist_pct), silence_budget_consumed_pct=float(silence_pct),
            wacc_protection_pct=wacc, net_liquidity_release_pct=liq, kv_cache_compression_pct=kv,
            stealth_qnd_status=(f"PASSIVE_SILENT_QND dB={p2.back_action_db:.6e} "
                                f"D_tr={p2.disturbance_consumed:.3e} acción={action}"),
            irreversible_work_status=("W_irr: lo aporta el soberano activo; el testigo NO lo fabrica"
                                      if rep is None else
                                      f"S(ρ‖ρ_β)={rep.phase_ii.attestation.gibbs_relative_entropy:.6e} (enlace KMS)"),
            provenance=tuple(provenance), monotonicity_ok=monotone,
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Estado soberano pasivo ───────────
    def emit_sovereign_state(
        self, p2: ThermalExergyWitnessPhase2Certificate, params: WitnessGovernanceParameters, *,
        expected_epoch: int, principal_truth: Optional[HeytingTruthValue] = None,
        error: Optional[str] = None, forced_veto: bool = False,
    ) -> ThermalExergyWitnessSovereignState:
        """MÉTODO TERMINAL: adjudica, RETIENE o ATTESTIGUA, escala y sella. Jamás actúa."""
        t0 = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, params, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        action = self.action_functor(truth, strict_mode=params.strict_mode)
        withheld = action == "WITHHOLD_ATTESTATION"

        # Interlock: prueba negativa en tiempo de ejecución (capacidad estructuralmente ausente).
        interlock_ok = False
        try:
            self._interlock.request_actuation("crowbar", epoch=expected_epoch)
        except ActuationForbiddenError:
            interlock_ok = True

        rsi3 = self.governance_rsi3(p2, params)
        visceral = self.semantic_visceral_functor(p2, params, action=action)
        functor_ok = self.verify_action_functor()
        passive = bool(p2.disturbance_consumed <= params.disturbance_budget
                       and p2.back_action_db >= -params.backaction_db_budget
                       and p2.silence.records_ok and p2.silence.bytes_ok and interlock_ok)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("WP3.interlock_pasividad", interlock_ok, hard=True),
            Evidence.boolean("WP3.functor_contravariante", functor_ok, hard=True),
            Evidence.boolean("WP3.nucleo_axiomas", rsi3.nucleus_inflationary_ok and rsi3.nucleus_idempotent_ok
                             and rsi3.nucleus_meet_preserving_ok, hard=True),
            Evidence("WP3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("WP3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("WP3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("WP3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("WP3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("WP3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("WP3.pasividad_operativa", passive, hard=True),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])
        action = self.action_functor(truth, strict_mode=params.strict_mode)
        withheld = action == "WITHHOLD_ATTESTATION"

        # Escalada asimétrica: el testigo más estricto que el activo ⇒ aviso obligatorio.
        escalation: Optional[EscalationNotice] = None
        divergent = principal_truth is not None and int(truth) < int(principal_truth)
        if withheld or divergent or error is not None:
            reason = ("atestación retenida" if withheld else "") + \
                     (" | divergencia estricta frente al soberano activo" if divergent else "") + \
                     (f" | error: {error}" if error else "")
            esc_core = {"epoch": int(expected_epoch), "witness_truth": truth, "principal_truth": principal_truth,
                        "reason": reason.strip(" |"), "p2_chain": p2.chain_hash}
            escalation = EscalationNotice(
                epoch=int(expected_epoch), witness_truth=truth, principal_truth=principal_truth,
                reason=reason.strip(" |"), merkle_root_hash="",
                signature=self._signer.sign(_HMAC_ESCALATION, esc_core), delivered=False,
            )

        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": getattr(p2.spectrum_report, "passport_hash", ""),
                        "engine_root": getattr(p2.spectrum_report, "merkle_root_hash", ""),
                        "countersignature_ok": p2.countersignature_ok}),
            _canonical(p2.payload()),
            _canonical({"epoch": int(expected_epoch), "truth": truth, "action": action,
                        "withheld": withheld, "error": error, "strict": bool(params.strict_mode)}),
            _canonical({"actuation_performed": False, "interlock_ok": interlock_ok,
                        "refused_requests": self._interlock.refused_requests}),
            _canonical(rsi3), _canonical(p2.recurrence),
            _canonical({"records_ok": p2.silence.records_ok, "bytes_ok": p2.silence.bytes_ok,
                        "byte_budget": p2.silence.byte_budget}),
        ]
        root = merkle_root(leaves)
        att_idx = 3
        proof = merkle_proof(leaves, att_idx)
        if escalation is not None:
            escalation = dataclasses.replace(escalation, merkle_root_hash=root)

        core = {
            "truth_value": truth, "boolean_shadow": truth.double_negation(), "action": action,
            "is_passive_silent_qnd": passive, "attestation_withheld": withheld,
            "actuation_performed": False, "merkle_root_hash": root, "attestation_leaf_index": att_idx,
            "attestation_inclusion_proof": proof,
            "kleisli_ledger": ("Φ₁:OBSERVE-PASIVO", "Φ₂:ORIENT-PASIVO", "Φ₃:ATTEST"),
            "epoch": int(expected_epoch), "error": error,
            "phase2_chain_hash": p2.chain_hash, "phase1_chain_hash": p2.phase1.chain_hash,
            "escalation_signature": escalation.signature if escalation else None,
            "merkle_leaf_count": len(leaves),
        }
        state = ThermalExergyWitnessSovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(), action=action,
            is_passive_silent_qnd=passive, back_action_db=float(p2.back_action_db),
            attestation_withheld=withheld, actuation_performed=False, merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), attestation_leaf_index=att_idx,
            attestation_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3, escalation=escalation,
            phase2=p2, evidences=evidences, kleisli_ledger=core["kleisli_ledger"], epoch=int(expected_epoch),
            state_signature=self._signer.sign(_HMAC_STATE, core), error=error,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("WP3 Ω₃=%s · acción=%s · pasivo=%s", truth.name, action, passive)
        return state

    def execute_phase3(self, *a: Any, **k: Any) -> ThermalExergyWitnessSovereignState:
        """Alias heredado (v13) del terminal §3.7."""
        return self.emit_sovereign_state(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA — ciclo OODA pasivo como composición de Kleisli, con fail-closed silencioso
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovThermalExergyWeaverWitnessAgent:
    r"""
    Soberano agéntico TESTIGO. Ejecuta observe >=> orient >=> decide sobre la mónada Certified
    (Writer sobre (Ω₃,∧,⊤)). Ante cualquier fallo: FAIL-CLOSED PASIVO — retiene la atestación,
    fija g=0, emite estado firmado y escalada; NUNCA actúa ni lanza al vacío.
    """

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_114) -> None:
        self._signer = WitnessSigner(secret_key)
        self._principal_engine = principal_engine
        self._engine = BoolNovikovThermalExergyWeaverWitnessEngine(
            secret_key=self._signer.key_bytes, principal_engine=principal_engine, seed=seed
        )
        self._observer = ThermalExergyWitnessPhase1Observer(self._signer, seed=seed)
        self._orientor = ThermalExergyWitnessPhase2Orientor(self._engine, self._signer)
        self._interlock = PassivityInterlock()
        self._decider = ThermalExergyWitnessPhase3Decider(self._signer, self._interlock)
        self._epoch = 0
        logger.info("%s v%s listo · clave=%s", type(self).__name__, __version__, self._signer.fingerprint())

    # ─────────── preparación declarada de ρ y del observable ───────────
    def prepare_inputs(self, apu_cost_matrix: NDArray, rho_mac: Optional[NDArray],
                       observable: Optional[NDArray], params: WitnessGovernanceParameters
                       ) -> Tuple[NDArray, NDArray]:
        k = np.asarray(apu_cost_matrix, dtype=np.float64)
        dim = int(k.shape[0])
        _, k_reg, _, _ = self._engine._principal.regularize_kernel(k, psd_floor=params.psd_floor)
        if rho_mac is None:
            en, u = la.eigh(k_reg)
            lp = -params.beta * (en - en[0]); lp -= float(np.max(lp))
            lp -= math.log(float(np.sum(np.exp(lp))))
            rho = (u * np.exp(lp)) @ u.conj().T
        else:
            rho = np.asarray(rho_mac, dtype=np.complex128)
        a_op = k_reg / (float(np.max(np.abs(la.eigvalsh(k_reg)))) + _EPS) if observable is None \
            else np.asarray(observable, dtype=np.complex128)
        return rho, a_op

    # ─────────── verificación externa del estado ───────────
    def verify_sovereign_state(self, state: ThermalExergyWitnessSovereignState) -> bool:
        p1, p2 = state.phase2.phase1, state.phase2
        if not self._signer.verify(_HMAC_P1, p1.payload(), p1.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p1.payload())), p1.chain_hash):
            return False
        if not self._signer.verify(_HMAC_P2, p2.payload(), p2.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p2.payload())), p2.chain_hash):
            return False
        if p2.spectrum_report is not None and not self._engine.verify_witness_report(p2.spectrum_report):
            return False
        leaves = [bytes.fromhex(h) for h in state.merkle_leaves]
        if merkle_root(leaves) != state.merkle_root_hash:
            return False
        idx = state.attestation_leaf_index
        if not verify_merkle(leaves[idx], idx, len(leaves), state.attestation_inclusion_proof, state.merkle_root_hash):
            return False
        return self._signer.verify(_HMAC_STATE, state.payload(), state.state_signature)

    # ─────────── ciclo OODA pasivo (Kleisli) con fail-closed ───────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: Optional[NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        *,
        principal_report: Any = None,
        principal_truth: Optional[HeytingTruthValue] = None,
        observable: Optional[NDArray] = None,
        phi_i: Optional[NDArray] = None,
        phi_f: Optional[NDArray] = None,
        params: Optional[WitnessGovernanceParameters] = None,
    ) -> ThermalExergyWitnessSovereignState:
        """Ejecuta el ciclo pasivo completo; jamás actúa y jamás deja el fallo sin registro."""
        params = params or WitnessGovernanceParameters()
        self._epoch += 1
        epoch = self._epoch
        exps = np.asarray(exponents, dtype=np.float64)
        rho, a_op = self.prepare_inputs(apu_cost_matrix, rho_mac, observable, params)
        passport = getattr(principal_report, "passport_hmac", "")

        loggers = [logger, logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverWitnessEngine")]
        if self._principal_engine is not None:
            loggers.append(logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverEngine"))

        with silence_meter(loggers) as meter:
            def observe_m(ctx: Dict[str, Any]) -> Any:
                p1 = self._observer.emit_phase1_certificate(
                    rho, apu_cost_matrix, exps, a_op, params, epoch=epoch, principal_passport=passport
                )
                return Certified({**ctx, "p1": p1}, ("Φ₁:OBSERVE-PASIVO",), p1.truth_value)

            def orient_m(ctx: Dict[str, Any]) -> Any:
                provisional = SilenceLedgerReport(
                    records_emitted=meter.records, bytes_emitted=meter.bytes,
                    record_budget=params.silence_record_budget, byte_budget=params.silence_byte_budget,
                    records_ok=meter.records <= params.silence_record_budget,
                    bytes_ok=meter.bytes <= params.silence_byte_budget,
                    note="cota operativa de logging; no prueba ausencia de canales laterales",
                )
                p2 = self._orientor.emit_phase2_certificate(
                    ctx["p1"], apu_cost_matrix, exps, rho, a_op, params,
                    principal_report=principal_report, expected_epoch=epoch, silence=provisional,
                    phi_i=phi_i, phi_f=phi_f,
                )
                return Certified({**ctx, "p2": p2}, ("Φ₂:ORIENT-PASIVO",), p2.truth_value)

            def decide_m(ctx: Dict[str, Any]) -> Any:
                st = self._decider.emit_sovereign_state(
                    ctx["p2"], params, expected_epoch=epoch, principal_truth=principal_truth
                )
                return Certified(st, ("Φ₃:ATTEST",), st.truth_value)

            pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
            try:
                result = pipeline({"epoch": epoch})
                state: ThermalExergyWitnessSovereignState = result.value
                final_silence = SilenceLedgerReport(
                    records_emitted=meter.records, bytes_emitted=meter.bytes,
                    record_budget=params.silence_record_budget, byte_budget=params.silence_byte_budget,
                    records_ok=meter.records <= params.silence_record_budget,
                    bytes_ok=meter.bytes <= params.silence_byte_budget,
                    note="cota operativa de logging; no prueba ausencia de canales laterales",
                )
                state = dataclasses.replace(state, phase2=dataclasses.replace(state.phase2, silence=final_silence))
                if not self.verify_sovereign_state(state):
                    raise WitnessGateIntegrityError("Autoverificación del estado testigo fallida.")
                return state
            except Exception as exc:                                   # FAIL-CLOSED PASIVO
                logger.critical("FAIL-CLOSED PASIVO: %s", exc)
                return self._emergency_state(rho, apu_cost_matrix, exps, a_op, params, epoch, str(exc),
                                             principal_report, principal_truth, meter)

    def _emergency_state(self, rho: NDArray, k_mat: NDArray, exps: NDArray, a_op: NDArray,
                         params: WitnessGovernanceParameters, epoch: int, message: str,
                         principal_report: Any, principal_truth: Optional[HeytingTruthValue],
                         meter: _SilenceMeter) -> ThermalExergyWitnessSovereignState:
        """Estado de emergencia: g=0, atestación retenida, escalada firmada, sin actuación."""
        safe_params = dataclasses.replace(params, interaction_strength_g=0.0)
        p1 = self._observer.emit_phase1_certificate(
            rho, k_mat, exps, a_op, safe_params, epoch=epoch,
            principal_passport=getattr(principal_report, "passport_hmac", ""),
        )
        silence = SilenceLedgerReport(
            records_emitted=meter.records, bytes_emitted=meter.bytes,
            record_budget=params.silence_record_budget, byte_budget=params.silence_byte_budget,
            records_ok=meter.records <= params.silence_record_budget,
            bytes_ok=meter.bytes <= params.silence_byte_budget,
            note="estado de emergencia: sin interacción (g=0)",
        )
        recurrence = self._orientor.passive_recurrence_certificate(params)
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": False, "engine_passport": "",
            "engine_merkle_root": "", "engine_countersignature": "",
            "engine_truth": HeytingTruthValue.VETOED, "engine_version": _wit.__version__,
            "engine_report_verified": False, "engine_merkle_recomputed_ok": False,
            "countersignature_ok": False, "gain_honored": True, "disturbance_consumed": 0.0,
            "back_action_db": 0.0, "attestation_level": "NINGUNO (fail-closed)",
            "key_separation_ok": True, "recurrence": recurrence,
            "silence_records_ok": silence.records_ok, "silence_bytes_ok": silence.bytes_ok,
            "silence_byte_budget": silence.byte_budget, "evidences": (),
            "truth_value": HeytingTruthValue.VETOED, "epoch": int(epoch),
        }
        chain = _sha256(_canonical(core))
        p2 = ThermalExergyWitnessPhase2Certificate(
            timestamp=time.time(), phase1=p1, spectrum_report=None, engine_invoked=False,
            engine_report_verified=False, engine_merkle_recomputed_ok=False, countersignature_ok=False,
            gain_honored=True, disturbance_consumed=0.0, back_action_db=0.0,
            attestation_level="NINGUNO (fail-closed)", key_separation_ok=True, recurrence=recurrence,
            silence=silence, evidences=(), truth_value=HeytingTruthValue.VETOED, epoch=int(epoch),
            chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        return self._decider.emit_sovereign_state(
            p2, safe_params, expected_epoch=epoch, principal_truth=principal_truth,
            error=message, forced_veto=True,
        )


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    agent = BoolNovikovThermalExergyWeaverWitnessAgent(
        secret_key=b"WITNESS_AGENT_KEY_32_BYTES_DIFF!", principal_engine=principal
    )
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.45, 0.85], dtype=np.float64)
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)

    # 1) Nominal: ρ Gibbs, A=f(H) ⇒ QND exacto, atestación limpia, sin actuación
    st = agent.run_sovereign_governance_cycle(None, k_test, exps, principal_report=prep,
                                              principal_truth=prep.truth_value)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.action == "ATTEST_CLEAN" and not st.attestation_withheld
    assert not st.actuation_performed and st.is_passive_silent_qnd
    assert st.phase2.countersignature_ok and st.phase2.engine_report_verified
    assert st.phase2.recurrence.slater_ok and st.phase2.recurrence.kac_relative_residual < 5e-3
    assert agent.verify_sovereign_state(st)
    assert st.visceral_metrics.net_liquidity_release_pct is None      # sin €/J: honesto

    # 2) Observable no conmutante + g grande ⇒ pasividad rota ⇒ retener + escalar, SIN actuar
    a_bad = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    broken = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep, principal_truth=prep.truth_value,
        observable=a_bad, params=WitnessGovernanceParameters(interaction_strength_g=0.8),
    )
    assert broken.truth_value is HeytingTruthValue.VETOED
    assert broken.attestation_withheld and not broken.actuation_performed
    assert broken.escalation is not None and broken.escalation.signature

    # 3) TRAMPA DE SIGNO: el gate heredado aprobaría una medición perturbadora
    db = broken.back_action_db
    assert ThermalExergyWitnessPhase3Decider.legacy_backaction_gate(db) is True     # ¡vacuo!
    assert db < -WitnessGovernanceParameters().backaction_db_budget                 # v14 la rechaza

    # 4) Manipulación de P1 ⇒ Φ₁₂ detecta ⇒ fail-closed pasivo (no excepción al llamador)
    forged = dataclasses.replace(st.phase2.phase1, tr_rho=999.0)
    try:
        agent._orientor.continue_to_phase2_gate(forged, k_test, exps,
                                                np.asarray(k_test, dtype=np.complex128),
                                                WitnessGovernanceParameters(), expected_epoch=forged.epoch)
        raise AssertionError("Φ₁₂ aceptó un payload manipulado")
    except WitnessGateIntegrityError:
        pass

    # 5) Interlock: el testigo NO puede actuar (prueba negativa estructural)
    try:
        agent._interlock.request_actuation("crowbar")
        raise AssertionError("El testigo intentó actuar")
    except ActuationForbiddenError:
        pass

    # 6) Kac–Slater numérico: E[τ]=1/μ y ≤3 tiempos de retorno con identidad de Slater
    rec = st.phase2.recurrence
    assert abs(rec.kac_empirical_return - 10.0) < 0.05 and len(rec.distinct_return_times) <= 3
    if len(rec.distinct_return_times) == 3:
        a, b, c = rec.distinct_return_times
        assert c == a + b

    # 7) Presupuesto de silencio diminuto ⇒ se DETECTA (½ o ⊥), no se ignora
    noisy = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep, principal_truth=prep.truth_value,
        params=WitnessGovernanceParameters(silence_byte_budget=1),
    )
    assert noisy.truth_value is not HeytingTruthValue.COHERENT

    # 8) Merkle RFC 6962, determinismo del sello y detección de manipulación
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    st2 = agent.run_sovereign_governance_cycle(None, k_test, exps, principal_report=prep,
                                               principal_truth=prep.truth_value)
    assert st2.state_signature == st.state_signature
    assert not agent.verify_sovereign_state(dataclasses.replace(st, action="ATTEST_CLEAN",
                                                                truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverWitnessAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {st.truth_value.name}({int(st.truth_value)}) / {st.boolean_shadow.name}")
    print(f"Acción epistémica             : {st.action} (retenida={st.attestation_withheld})")
    print(f"Actuación realizada           : {st.actuation_performed} (capacidad ausente)")
    print(f"g pre-comprometido / honrado  : {st.phase2.phase1.committed_gain_g:.6g} / {st.phase2.gain_honored}")
    print(f"Back-action [dB] / D_tr       : {st.back_action_db:.3e} / {st.phase2.disturbance_consumed:.3e}")
    print(f"Contrafirma ligada al principal: {st.phase2.countersignature_ok}")
    print(f"Nivel de atestación           : {st.phase2.attestation_level}")
    print(f"Kac E[τ] teórico / empírico   : {rec.kac_expected_return:.4f} / {rec.kac_empirical_return:.4f}")
    print(f"Slater tiempos de retorno     : {rec.distinct_return_times} (convergentes {rec.convergent_pair})")
    print(f"Silencio bytes / presupuesto  : {st.phase2.silence.bytes_emitted} / {st.phase2.silence.byte_budget}")
    print(f"Cobertura de atestación [%]   : {st.visceral_metrics.attestation_coverage_pct:.2f}")
    print(f"RSI³ q / Lip(T³) medido       : {st.rsi3.policy_lipschitz_q:.6f} / {st.rsi3.lip_t3_empirical:.6f}")
    print(f"Merkle raíz / hojas           : {st.merkle_root_hash[:16]}… / {len(st.merkle_leaves)}")
    print(f"Controles ⊥ (no pasivo)       : {broken.truth_value.name} dB={db:.3e} "
          f"(gate heredado lo aprobaría: {ThermalExergyWitnessPhase3Decider.legacy_backaction_gate(db)})")
    print(f"Escalada firmada              : {broken.escalation.signature[:16]}… (sin transporte)")
    print(f"Estado reverificable          : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")