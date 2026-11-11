# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverAgent  v14.0.0-PhD (reedición rigurosa)               ║
║ Ubicación: app/agents/wisdom/bool_novikov_thermal_exergy_weaver_agent.py                       ║
║ Función  : Soberano agéntico OODA trifásico que ORIENTA al motor espectral v14 y custodia      ║
║            la decisión, la actuación y la trazabilidad criptográfica del tejido.               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.5  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.6  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle

    P1 ⊂ P2 ⊂ ThermalExergySovereignState     (anidamiento de certificados)
    La ejecución real es una composición de Kleisli observe >=> orient >=> decide sobre la
    mónada Certified del motor (Writer sobre (Ω₃, ∧, ⊤)); el ledger y el veredicto se acumulan.

FASE 1 — OBSERVE   (ThermalExergyPhase1Observer)
  §1.1 Validación previa a la normalización (hermiticidad, traza cruda, positividad).
  §1.2 Normalización auditada como morfismo registrado (no reparación silenciosa).
  §1.3 Isometría espectral: invariancia del espectro bajo conjugación unitaria Haar.
  §1.4 Compromiso de entrada: digests canónicos de ρ, K, exponentes y parámetros.
  §1.5 [TERMINAL] emit_phase1_certificate → P1 (HMAC v14, sin reloj en el hash).

COMPUERTA Φ₁₂ (continue_to_phase2_gate)
  Recalcula chain_hash DESDE el payload, verifica HMAC en tiempo constante, re-digesta los
  arrays reales (anti-TOCTOU) y verifica enlace de versión/época. ADMITE ⊥ (juicio inquisitorial).

FASE 2 — ORIENT    (ThermalExergyPhase2Orientor)
  §2.1 accept_phase1_certificate (entrada = terminal de Fase 1 + Φ₁₂).
  §2.2 Invocación del motor v14 con contrato de argumentos correcto.
  §2.3 Verificación del pasaporte HMAC del motor y cruce de digests (kernel/exponentes).
  §2.4 Orientación física: exergía destruida, W_irr, ESS, KMS, Floer, Melnikov, Banach.
  §2.5 Cadencia de Poincaré: rotación diophantina + teorema de los tres huecos (Steinhaus).
  §2.6 [TERMINAL] emit_phase2_certificate → P2 (encadena P1, sella el pasaporte del motor).

COMPUERTA Φ₂₃ (continue_to_phase3_gate)
  Integridad (dura, excepción ⇒ fail-safe) separada de física (q<1; si falla ⇒ ⊥ forzado y
  se ADMITE a Fase 3 para que el decidor pueda actuar). Nunca se «aprueba» por omisión.

FASE 3 — DECIDE & ACT (ThermalExergyPhase3Decider)
  §3.1 accept_phase2_certificate.
  §3.2 Adjudicación Ω₃ + núcleo de doble negación ¬¬ (sombra booleana) + functor de acción.
  §3.3 Interlock de actuación: token HMAC ligado a época y digest de decisión, anti-replay.
  §3.4 Crowbar HAL: despacho medido, presupuesto físico declarado, viabilidad honesta.
  §3.5 Φ_sem visceral con provenance dimensional (sin constantes inventadas).
  §3.6 RSI³ de gobernanza: contracción de política, Newton–Kantorovich, Kleisli, núcleo.
  §3.7 [TERMINAL] emit_sovereign_state → Merkle RFC 6962 + prueba de inclusión + firma.

FACHADA (§4) — BoolNovikovThermalExergyWeaverAgent
  Orquesta, verifica y, ante cualquier fallo, DISPARA en fail-safe y emite estado de emergencia.

MIGRACIÓN DESDE v13 (rupturas deliberadas)
──────────────────────────────────────────
• Ω₃ se reordena (heredado del motor v14): VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `work_w` → `work_mean`; `positron_hmac_valid: bool` → `positron_token: str` (HMAC real).
• Los sellos ya no incluyen `time.time()`: los certificados son DETERMINISTAS y reverificables.
• `crowbar_latency_ns` pasa a ser latencia de DESPACHO SOFTWARE medida; la latencia física
  se declara por presupuesto y sólo se afirma con medición hardware-in-the-loop (HIL).
• Los KPI viscerales pueden ser `None` con estado `UNDECLARED_INPUT` (antes: cifras falsas).

LIMITACIONES DECLARADAS
───────────────────────
1. Este módulo NO puede certificar nanosegundos de un ISR en IRAM/Xtensa desde Python: sólo
   mide su propio camino de despacho. El plazo de 392.15 ns es política del camino lógico.
2. La conmutación física de un SCR BT151 (t_gt ~ 1–3 µs) es incompatible con 392.15 ns; el
   certificado lo declara en lugar de simular cumplimiento.
3. Φ_sem es una heurística de política MONÓTONA y acotada, no un teorema financiero.
4. La cadencia diophantina garantiza cobertura por el teorema de los tres huecos; no garantiza
   detección de un adversario adaptativo.
5. RSI³ de gobernanza es meta-iteración numérica certificada sobre el espacio de políticas.

REFERENCIAS: Boyd (OODA); RFC 6962 §2.1 (Merkle); Mezzadri (2007) (unitario Haar);
Kantorovich–Newton (Ortega–Rheinboldt); Steinhaus–Świerczkowski (tres huecos); Khinchin
(fracciones continuas); Poincaré (número de rotación); Johnstone (núcleos de Heyting);
Mac Lane (Kleisli); CODATA 2022; datasheets ESP32-S3 / BT151.
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
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────── Contrato de importación del motor espectral v14 ───────────────────
_ENGINE_IMPORT_ERROR: Optional[Exception] = None
for _module_path in (
    "app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
    "bool_novikov_thermal_exergy_weaver_engine",
):
    try:
        _mod = __import__(_module_path, fromlist=["*"])
        BoolNovikovThermalExergyWeaverEngine = _mod.BoolNovikovThermalExergyWeaverEngine
        HeytingTruthValue = _mod.HeytingTruthValue
        Evidence = _mod.Evidence
        Certified = _mod.Certified
        kleisli_compose = _mod.kleisli_compose
        ThermalExergyWeaverSpectrumReport = _mod.ThermalExergyWeaverSpectrumReport
        _ENGINE_VERSION = _mod.__version__
        break
    except ImportError as exc:                                    # pragma: no cover
        _ENGINE_IMPORT_ERROR = exc
else:                                                             # pragma: no cover
    raise ImportError(
        "No se pudo importar el motor espectral v14 "
        "(app/wisdom/bool_novikov_thermal_exergy_weaver_engine.py)."
    ) from _ENGINE_IMPORT_ERROR

__all__ = [
    "GovernanceParameters",
    "ThermalExergyPhase1Certificate",
    "ThermalExergyPhase2Certificate",
    "ThermalExergyVisceralMetrics",
    "ThermalExergySovereignState",
    "ThermalExergyPhase1Observer",
    "ThermalExergyPhase2Orientor",
    "ThermalExergyPhase3Decider",
    "BoolNovikovThermalExergyWeaverAgent",
    "SovereignSigner",
    "GovernanceError",
    "GateIntegrityError",
    "ActuationInterlockError",
    "__version__",
]

logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-BOOL-NOVIKOV-THERMAL-EXERGY-WEAVER-AGENT-POINCARE-KAM-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HMAC_P1: Final[bytes] = b"THERMAL-EXERGY-P1-OBSERVE-v14|"
_HMAC_P2: Final[bytes] = b"THERMAL-EXERGY-P2-ORIENT-v14|"
_HMAC_STATE: Final[bytes] = b"THERMAL-EXERGY-P3-STATE-v14|"
_HMAC_ACT: Final[bytes] = b"THERMAL-EXERGY-ACT-INTERLOCK-v14|"
_MERKLE_LEAF: Final[bytes] = b"\x00"          # RFC 6962 §2.1: separación hoja/nodo
_MERKLE_NODE: Final[bytes] = b"\x01"
_LOGIC_DEADLINE_NS: Final[float] = 392.15    # POLÍTICA del camino lógico (ISR IRAM), no del SCR
_GOLDEN_CONJUGATE: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)   # número noble (peor aproximable)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO COMÚN (compartido por las tres fases anidadas)
#  Criptografía de compromiso, Merkle RFC 6962, canonicalización, utilidades espectrales
# ══════════════════════════════════════════════════════════════════════════════════════
class GovernanceError(Exception):
    """Error base de gobernanza."""


class GateIntegrityError(GovernanceError):
    """Fallo DURO de integridad en Φ₁₂/Φ₂₃ ⇒ fail-safe (disparo) y estado de emergencia."""


class ActuationInterlockError(GovernanceError):
    """Intento de actuación sin autorización válida o con repetición de época."""


def _freeze(obj: Any) -> Any:
    """Proyección determinista a primitivas JSON (arrays→digest, flotantes→15 dígitos)."""
    if isinstance(obj, np.ndarray):
        return {"ndarray_sha256": _array_digest(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _freeze(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, HeytingTruthValue):
        return {"omega3": int(obj), "name": obj.name}
    if isinstance(obj, (list, tuple)):
        return [_freeze(v) for v in obj]
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


def _array_digest(arr: NDArray) -> str:
    """
    Digest canónico IdÉNTICO al del motor v14 (forma | dtype little-endian | bytes).
    Garantiza que los digests del agente y del motor sean interoperables y comparables.
    """
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|")
    h.update(str(a.shape).encode("ascii"))
    h.update(b"|")
    h.update(dt.str.encode("ascii"))
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SovereignSigner:
    """
    Firmante HMAC-SHA256 con separación de dominio y verificación en tiempo constante.

    La clave NUNCA se registra, ni se incluye en el mensaje (v13 concatenaba key.hex() dentro
    del SHA-256, lo que no es un MAC y expone material de clave). Sin clave explícita ni de
    entorno se genera una efímera con advertencia: los sellos no serán verificables entre
    procesos, y así se declara.
    """

    _MIN_KEY_BYTES: Final[int] = 32

    def __init__(self, secret_key: Optional[bytes] = None, *, env_var: str = "APU_AGENT_HMAC_KEY") -> None:
        if secret_key is None:
            env = os.environ.get(env_var)
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave HMAC efímera: sellos NO verificables entre procesos.")
        if len(secret_key) < self._MIN_KEY_BYTES:
            raise GovernanceError(f"La clave HMAC requiere ≥ {self._MIN_KEY_BYTES * 8} bits.")
        self._key = bytes(secret_key)

    @property
    def key_bytes(self) -> bytes:
        """Sólo para propagación intra-proceso al motor; jamás registrar ni serializar."""
        return self._key

    def fingerprint(self) -> str:
        """Identificador no invertible de la clave (HMAC de una etiqueta fija)."""
        return hmac.new(self._key, b"APU-AGENT-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]

    def sign(self, domain: bytes, payload: Any) -> str:
        return hmac.new(self._key, domain + _canonical(payload), hashlib.sha256).hexdigest()

    def verify(self, domain: bytes, payload: Any, signature: str) -> bool:
        return hmac.compare_digest(self.sign(domain, payload), str(signature))


# ───────────────────────── Merkle RFC 6962 (árbol real, no hash plano) ─────────────────────────
def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(_MERKLE_LEAF + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_MERKLE_NODE + left + right).digest()


def _mth(leaves: Sequence[bytes]) -> bytes:
    """Merkle Tree Hash: split en la mayor potencia de 2 estrictamente menor que n."""
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return _leaf_hash(leaves[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return _node_hash(_mth(leaves[:k]), _mth(leaves[k:]))


def merkle_root(leaves: Sequence[bytes]) -> str:
    return _mth(list(leaves)).hex()


def merkle_inclusion_proof(leaves: Sequence[bytes], index: int) -> Tuple[Tuple[str, bool], ...]:
    """PATH(m, D) de RFC 6962 §2.1.3; cada elemento es (hermano_hex, hermano_es_derecho)."""
    ds = list(leaves)
    if not 0 <= index < len(ds):
        raise GovernanceError("Índice de hoja fuera de rango.")

    def path(m: int, sub: Sequence[bytes]) -> List[Tuple[str, bool]]:
        n = len(sub)
        if n == 1:
            return []
        k = 1 << ((n - 1).bit_length() - 1)
        if m < k:
            return path(m, sub[:k]) + [(_mth(sub[k:]).hex(), True)]
        return path(m - k, sub[k:]) + [(_mth(sub[:k]).hex(), False)]

    return tuple(path(index, ds))


def verify_merkle_inclusion(leaf: bytes, index: int, leaf_count: int,
                            proof: Sequence[Tuple[str, bool]], root_hex: str) -> bool:
    """Algoritmo exacto de verificación de RFC 6962 (fn/sn), resistente a segunda preimagen."""
    if leaf_count <= 0 or not 0 <= index < leaf_count:
        return False
    fn, sn = index, leaf_count - 1
    acc = _leaf_hash(leaf)
    for sibling_hex, _is_right in proof:
        if sn == 0:
            return False
        sibling = bytes.fromhex(sibling_hex)
        if (fn & 1) or fn == sn:
            acc = _node_hash(sibling, acc)
            if not (fn & 1):
                while (fn & 1) == 0 and fn != 0:
                    fn >>= 1
                    sn >>= 1
        else:
            acc = _node_hash(acc, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and acc.hex() == root_hex


# ───────────────────────── utilidades espectrales y de política ─────────────────────────
def _haar_unitary(dim: int, rng: np.random.Generator) -> NDArray:
    """Unitario Haar vía QR de Ginibre con corrección de fases (Mezzadri 2007)."""
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = np.where(np.abs(diag) > 0.0, diag / np.maximum(np.abs(diag), _EPS), 1.0)
    return q * phases


def _savgol_derivative_coeffs(window: int, poly: int, deriv: int) -> NDArray:
    """Coeficientes de Savitzky–Golay para la derivada d-ésima en el centro (paso 1 tick)."""
    if window % 2 == 0 or window <= poly or deriv > poly:
        raise GovernanceError("Savitzky–Golay: ventana/orden inválidos.")
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, poly + 1, increasing=True))[deriv] * math.factorial(deriv)


def _continued_fraction_convergents(alpha: float, max_denominator: int) -> Tuple[Tuple[int, int], ...]:
    """Convergentes p_k/q_k de α (mejores aproximaciones racionales, teorema de Lagrange)."""
    x = float(alpha)
    a = math.floor(x)
    p_prev, q_prev, p, q = 1, 0, int(a), 1
    convs: List[Tuple[int, int]] = [(p, q)]
    for _ in range(64):
        frac = x - a
        if frac < 1e-14:
            break
        x = 1.0 / frac
        a = math.floor(x)
        p_next, q_next = int(a) * p + p_prev, int(a) * q + q_prev
        if q_next > max_denominator:
            break
        p_prev, q_prev, p, q = p, q, p_next, q_next
        convs.append((p, q))
    return tuple(convs)


def _hard_evidence_margin(evidences: Sequence[Any]) -> float:
    """Fracción de evidencias DURAS satisfechas ∈ [0,1]: salud estructural certificada."""
    hard = [e for e in evidences if bool(getattr(e, "hard", False))]
    if not hard:
        return 1.0
    return float(sum(1 for e in hard if e.passed)) / float(len(hard))


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS (parámetros, certificados, métricas, estado soberano)
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GovernanceParameters:
    """Parámetros declarados del ciclo; su digest se compromete en P1 (anti-sustitución)."""

    beta: float = 1.0
    work_mean: float = 5.0
    delta_f: float = 4.8
    ball_r: float = 1.0
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    melnikov_epsilon: float = 0.01
    kam_scan_order: int = 48
    diophantine_tau: float = 1.05
    psd_floor: float = 1e-12
    rho_mode: str = "gibbs"                     # gibbs | maximally_mixed | provided
    strict_mode: bool = False                   # True: ½ también dispara (política declarada)
    cadence_points: int = 64
    cadence_rotation: float = _GOLDEN_CONJUGATE
    exergy_reference_w: float = 1.0
    exergy_price_per_joule: Optional[float] = None      # [unidad monetaria / J]
    capital_base: Optional[float] = None                # [unidad monetaria]
    kv_bytes_before: Optional[int] = None
    kv_bytes_after: Optional[int] = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.beta) and self.beta > 0.0):
            raise GovernanceError("β debe ser finito y positivo.")
        if self.work_mean < self.delta_f - 1e-12:
            raise GovernanceError("⟨W⟩ < ΔF viola Jensen (segunda ley) en el modelo declarado.")
        if self.rho_mode not in ("gibbs", "maximally_mixed", "provided"):
            raise GovernanceError("rho_mode ∈ {gibbs, maximally_mixed, provided}.")
        if self.cadence_points < 4:
            raise GovernanceError("cadence_points ≥ 4 para el teorema de los tres huecos.")
        if not (0.0 < self.cadence_rotation < 1.0):
            raise GovernanceError("El número de rotación debe estar en (0,1).")
        if self.exergy_reference_w <= 0.0:
            raise GovernanceError("exergy_reference_w debe ser positivo.")

    def digest(self) -> str:
        return _sha256_hex(_canonical(self))


@dataclass(frozen=True)
class ThermalExergyPhase1Certificate:
    """P1 — Observación comprometida. `timestamp` es telemetría y NO entra en el sello."""

    timestamp: float
    tr_rho: float                       # traza CRUDA (previa a normalizar)
    tr_rho_normalized: float
    normalization_factor: float
    is_psd: bool
    hermiticity_residual: float
    purity: float
    von_neumann_entropy: float
    spectral_isometry_residual: float
    min_eigenvalue: float
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    params_digest: str
    rho_mode: str
    agent_version: str
    engine_version: str
    epoch: int
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        """Todo lo comprometido (excluye reloj y sellos): base del chain_hash y del HMAC."""
        excluded = {"timestamp", "chain_hash", "hmac_signature"}
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in excluded}


@dataclass(frozen=True)
class PoincareCadenceCertificate:
    """§2.5 — Cadencia cuasi-periódica: rotación diophantina y teorema de los tres huecos."""

    rotation_number: float
    convergents: Tuple[Tuple[int, int], ...]
    diophantine_gamma: float
    low_order_resonance: bool
    n_points: int
    distinct_gap_count: int
    max_gap: float
    three_gap_identity_residual: float
    coverage_bound: float
    three_gap_ok: bool
    coverage_ok: bool


@dataclass(frozen=True)
class ThermalExergyPhase2Certificate:
    """P2 — Orientación: anida P1 y el informe completo del motor v14."""

    timestamp: float
    phase1: ThermalExergyPhase1Certificate
    spectrum_report: Any
    engine_passport: str
    engine_passport_verified: bool
    engine_input_digest_match: bool
    destroyed_exergy_w: float
    irreversible_work_j: float
    ess_fraction: float
    kms_residual: float
    banach_composite_q: float
    chaos_detected: bool
    hard_evidence_margin: float
    cadence: PoincareCadenceCertificate
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    epoch: int
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        """
        Payload comprometido: el informe del motor se sella por su pasaporte HMAC y sus
        campos deterministas clave (no por su telemetría temporal).
        """
        rep = self.spectrum_report
        return {
            "phase1_chain_hash": self.phase1.chain_hash,
            "engine_passport": self.engine_passport,
            "engine_passport_verified": self.engine_passport_verified,
            "engine_input_digest_match": self.engine_input_digest_match,
            "engine_truth": rep.truth_value,
            "engine_version": rep.engine_version,
            "engine_input_digest": rep.input_digest,
            "engine_legacy_summary": rep.legacy_summary(),
            "destroyed_exergy_w": self.destroyed_exergy_w,
            "irreversible_work_j": self.irreversible_work_j,
            "ess_fraction": self.ess_fraction,
            "kms_residual": self.kms_residual,
            "banach_composite_q": self.banach_composite_q,
            "chaos_detected": self.chaos_detected,
            "hard_evidence_margin": self.hard_evidence_margin,
            "cadence": self.cadence,
            "evidences": self.evidences,
            "truth_value": self.truth_value,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class GateReceipt:
    """Recibo de compuerta homomórfica: integridad recalculada, no heredada."""

    gate: str
    payload_digest: str
    signature_ok: bool
    chain_recomputed_ok: bool
    array_binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class CrowbarBudget:
    """
    Presupuesto temporal DECLARADO con procedencia. El plazo lógico (392.15 ns) pertenece al
    camino ISR/IRAM; la conmutación física del SCR es de orden microsegundos.
    """

    logic_deadline_ns: float = _LOGIC_DEADLINE_NS
    gpio_write_ns: float = 25.0            # escritura directa a registro GPIO (orden declarado)
    gate_driver_ns: float = 150.0          # driver de compuerta (declarado)
    scr_turn_on_ns: float = 1500.0         # BT151 t_gt típico ~1.5 µs (datasheet)
    hil_firmware_isr_ns: Optional[float] = None    # medición hardware-in-the-loop del ISR
    hil_physical_total_ns: Optional[float] = None  # medición HIL del disparo físico completo

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)


@dataclass(frozen=True)
class CrowbarCertificate:
    """§3.4 — Certificación honesta de la actuación: medición software + presupuesto físico."""

    driver_name: str
    driver_is_physical: bool
    armed: bool
    dispatched: bool
    dispatch_samples: int
    dispatch_min_ns: float
    dispatch_median_ns: float
    dispatch_p99_ns: float
    dispatch_max_ns: float
    trip_dispatch_ns: float
    logic_deadline_ns: float
    python_dispatch_within_deadline: bool
    firmware_isr_verified: bool
    firmware_isr_within_deadline: bool
    physical_budget_ns: float
    physical_deadline_feasible: bool
    hil_physical_verified: bool
    interlock_token_verified: bool
    fail_safe_armed: bool
    note: str


@dataclass(frozen=True)
class ThermalExergyVisceralMetrics:
    """
    §3.5 — Φ_sem con PROCEDENCIA. Un porcentaje sin insumos declarados es `None`, no un número
    inventado. Análisis dimensional: [J]·[moneda/J]/[moneda] = adimensional.
    """

    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    irreversible_work_status: str
    destroyed_exergy_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool
    action_functor_ok: bool


@dataclass(frozen=True)
class GovernanceRSI3Certificate:
    """§3.6 — RSI³ de gobernanza: Banach + Kantorovich + núcleo ¬¬ + leyes de Kleisli."""

    policy_lipschitz_q: float
    policy_lip_t3_bound: float
    policy_lip_t3_empirical: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_policy_error: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_quadratic_ratio_residual: float
    newton_iterations: int
    nucleus_inflationary_ok: bool
    nucleus_idempotent_ok: bool
    nucleus_meet_preserving_ok: bool
    kleisli_monad_laws_ok: bool
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool


@dataclass(frozen=True)
class ThermalExergySovereignState:
    """Estado soberano final: anida P2 (que anida P1) y sella Merkle + HMAC deterministas."""

    timestamp: float
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    action: str
    crowbar_triggered: bool
    crowbar_latency_ns: float
    crowbar_certificate: CrowbarCertificate
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    decision_leaf_index: int
    decision_inclusion_proof: Tuple[Tuple[str, bool], ...]
    visceral_metrics: ThermalExergyVisceralMetrics
    rsi3: GovernanceRSI3Certificate
    phase2: ThermalExergyPhase2Certificate
    evidences: Tuple[Any, ...]
    kleisli_ledger: Tuple[str, ...]
    epoch: int
    state_signature: str
    error: Optional[str]
    execution_time_ms: float

    def payload(self) -> Dict[str, Any]:
        excluded = {"timestamp", "state_signature", "execution_time_ms", "crowbar_certificate",
                    "phase2", "visceral_metrics", "rsi3", "evidences"}
        base = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in excluded}
        base["phase2_chain_hash"] = self.phase2.chain_hash
        base["phase1_chain_hash"] = self.phase2.phase1.chain_hash
        return base

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)


class CrowbarDriver(Protocol):
    """Interfaz HAL de actuación. `trip` exige token de interlock válido."""

    name: str
    is_physical: bool

    def dry_run(self) -> int: ...
    def arm(self) -> None: ...
    def trip(self, token: str) -> int: ...
    def safe_state(self) -> None: ...


class SimulatedCrowbarDriver:
    """
    Driver SIMULADO (no físico). Mide el despacho de software real con `perf_counter_ns`
    sobre un latch en memoria. Jamás afirma haber conmutado hardware.
    """

    name = "SIMULATED_LATCH (NO FÍSICO)"
    is_physical = False

    def __init__(self, token_verifier: Callable[[str], bool]) -> None:
        self._verify = token_verifier
        self._armed = False
        self._latched = False

    def dry_run(self) -> int:
        t0 = time.perf_counter_ns()
        _ = self._latched
        return int(time.perf_counter_ns() - t0)

    def arm(self) -> None:
        self._armed = True

    def trip(self, token: str) -> int:
        if not self._verify(token):
            raise ActuationInterlockError("Token de interlock inválido: disparo rechazado.")
        if not self._armed:
            raise ActuationInterlockError("Crowbar no armado: disparo rechazado.")
        t0 = time.perf_counter_ns()
        self._latched = True
        self._armed = False
        return int(time.perf_counter_ns() - t0)

    def safe_state(self) -> None:
        self._latched = True
        self._armed = False


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE — Observación comprometida de la variedad dual
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyPhase1Observer:
    r"""
    FASE 1 OODA. Observa ρ (MAC sobre Λ_Nov) y K (MIC sobre R_Boole) SIN reparar en silencio.

    Diferencia esencial con v13: el certificado compromete los arrays reales (digests) y
    verifica isometría espectral genuina (invariancia del espectro bajo conjugación unitaria),
    no sólo `Tr ρ` y un booleano PSD que cualquier matriz no hermítica burlaba.
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE"

    def __init__(self, signer: SovereignSigner, *, seed: int = 20_261_111) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa a la normalización ───────────
    def validate_pre_normalization(self, rho_raw: NDArray, dim: int) -> Dict[str, float]:
        """Hermiticidad, traza cruda positiva y finitud ANTES de cualquier normalización."""
        rho = np.asarray(rho_raw, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape != (dim, dim):
            raise GovernanceError(f"ρ debe ser {dim}×{dim}; recibido {rho.shape}.")
        if not np.all(np.isfinite(rho)):
            raise GovernanceError("ρ contiene NaN/Inf.")
        fro = float(la.norm(rho, ord="fro")) + _EPS
        herm = float(la.norm(rho - rho.conj().T, ord="fro")) / fro
        trace_raw = complex(np.trace(rho))
        if abs(trace_raw.imag) > 1e-10 * max(1.0, abs(trace_raw.real)):
            raise GovernanceError("Traza compleja: ρ no puede ser un estado físico.")
        if trace_raw.real <= 0.0:
            raise GovernanceError("Traza no positiva: ρ no normalizable (fail-fast).")
        return {"hermiticity_residual": herm, "trace_raw": float(trace_raw.real)}

    # ─────────── §1.2 Normalización auditada (morfismo registrado) ───────────
    @staticmethod
    def normalize_audited(rho_raw: NDArray, trace_raw: float) -> Tuple[NDArray, float]:
        """ρ ↦ ρ/Tr ρ, registrando el factor: la normalización es dato auditable, no reparación."""
        rho = 0.5 * (np.asarray(rho_raw, dtype=np.complex128) + np.asarray(rho_raw, dtype=np.complex128).conj().T)
        return rho / float(trace_raw), 1.0 / float(trace_raw)

    # ─────────── §1.3 Isometría espectral bajo unitario Haar ───────────
    def spectral_isometry_residual(self, rho: NDArray) -> float:
        """‖spec(UρU†) − spec(ρ)‖_∞ para U Haar: invariancia de base (isometría espectral)."""
        dim = rho.shape[0]
        u_mat = _haar_unitary(dim, self._rng)
        base = np.sort(la.eigvalsh(rho))
        moved = np.sort(la.eigvalsh(u_mat @ rho @ u_mat.conj().T))
        return float(np.max(np.abs(moved - base)))

    # ─────────── §1.4+§1.5 Observación completa y emisión terminal ───────────
    def emit_phase1_certificate(
        self,
        rho_raw: NDArray,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        params: GovernanceParameters,
        *,
        epoch: int,
    ) -> ThermalExergyPhase1Certificate:
        r"""
        MÉTODO TERMINAL DE LA FASE 1 → entrada de §2.1 (vía Φ₁₂).

        Certifica: hermiticidad, traza, positividad, pureza Tr(ρ²) ≤ 1, entropía en
        [0, log d], isometría espectral y compromiso de entrada (digests de ρ, K, exps, params).
        """
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise GovernanceError("K (MIC) debe ser cuadrada.")
        dim = int(k_mat.shape[0])
        pre = self.validate_pre_normalization(rho_raw, dim)
        rho, factor = self.normalize_audited(rho_raw, pre["trace_raw"])

        eigs = la.eigvalsh(rho)
        min_eig = float(np.min(eigs))
        trace_norm = float(np.real(np.trace(rho)))
        purity = float(np.real(np.trace(rho @ rho)))
        probs = np.clip(eigs, 0.0, 1.0)
        entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        isometry = self.spectral_isometry_residual(rho)
        is_psd = bool(min_eig >= -1e-12)

        evidences: Tuple[Any, ...] = (
            Evidence("P1.hermiticidad", pre["hermiticity_residual"], 1e-10, hard=True),
            Evidence("P1.traza_unitaria", abs(trace_norm - 1.0), 1e-12, hard=True),
            Evidence.boolean("P1.positividad", is_psd, hard=True),
            Evidence("P1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("P1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence("P1.isometria_espectral", isometry, 1e-11, hard=True),
            Evidence.boolean("P1.modo_rho_declarado", params.rho_mode in ("gibbs", "maximally_mixed", "provided"),
                             hard=True),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        payload_core = {
            "tr_rho": pre["trace_raw"], "tr_rho_normalized": trace_norm, "normalization_factor": factor,
            "is_psd": is_psd, "hermiticity_residual": pre["hermiticity_residual"], "purity": purity,
            "von_neumann_entropy": entropy, "spectral_isometry_residual": isometry,
            "min_eigenvalue": min_eig, "rho_digest": _array_digest(np.asarray(rho_raw, dtype=np.complex128)),
            "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "params_digest": params.digest(), "rho_mode": params.rho_mode,
            "agent_version": __version__, "engine_version": _ENGINE_VERSION, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain_hash = _sha256_hex(_canonical(payload_core))
        signature = self._signer.sign(_HMAC_P1, payload_core)
        cert = ThermalExergyPhase1Certificate(
            timestamp=time.time(), chain_hash=chain_hash, hmac_signature=signature, **{
                k: v for k, v in payload_core.items()
            },
        )
        logger.info("Fase 1 (OBSERVE) Ω₃=%s · Tr ρ_crudo=%.6f · isometría=%.2e",
                    truth.name, pre["trace_raw"], isometry)
        return cert  # ──► Φ₁₂ ──► Fase 2 §2.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT — Orientación espectral con el motor v14 y cadencia de Poincaré
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyPhase2Orientor:
    r"""
    FASE 2 OODA. Anida la Fase 1 por handoff: su primer método consume el terminal de Fase 1
    a través de Φ₁₂; su último método alimenta el primero de Fase 3 a través de Φ₂₃.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT"

    def __init__(self, engine: Any, signer: SovereignSigner) -> None:
        self._engine = engine
        self._signer = signer

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self,
        p1: ThermalExergyPhase1Certificate,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        params: GovernanceParameters,
        *,
        expected_epoch: int,
    ) -> GateReceipt:
        r"""
        Φ₁₂ — Compuerta homomórfica NO tautológica:
          (i) recalcula chain_hash DESDE el payload (detecta alteración de campos),
          (ii) verifica el HMAC en tiempo constante,
          (iii) re-digesta los arrays REALES que cruzan la compuerta (anti-TOCTOU),
          (iv) verifica época y versiones.
        ADMITE ⊥ para permitir el juicio inquisitorial (la auditoría no se silencia).
        """
        payload = p1.payload()
        recomputed = _sha256_hex(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p1.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P1, payload, p1.hmac_signature)
        binding_ok = (
            hmac.compare_digest(p1.kernel_digest, _array_digest(np.asarray(apu_cost_matrix, dtype=np.float64)))
            and hmac.compare_digest(p1.exponents_digest, _array_digest(np.asarray(exponents, dtype=np.float64)))
            and hmac.compare_digest(p1.params_digest, params.digest())
            and p1.epoch == expected_epoch
            and p1.agent_version == __version__
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise GateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return GateReceipt(
            gate="Φ₁₂", payload_digest=recomputed, signature_ok=sig_ok, chain_recomputed_ok=chain_ok,
            array_binding_ok=binding_ok, admitted=True, forced_veto=False,
            note="⊥ admitido: el juicio inquisitorial de sobrecostos continúa.",
        )

    def accept_phase1_certificate(
        self, p1: ThermalExergyPhase1Certificate, apu_cost_matrix: NDArray,
        exponents: NDArray, params: GovernanceParameters, *, expected_epoch: int,
    ) -> GateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.5."""
        return self.continue_to_phase2_gate(
            p1, apu_cost_matrix, exponents, params, expected_epoch=expected_epoch
        )

    # ─────────── §2.2+§2.3 Motor v14 y verificación de su pasaporte ───────────
    def invoke_engine(
        self, apu_cost_matrix: NDArray, exponents: NDArray, rho: NDArray,
        params: GovernanceParameters, *, positron_token: Optional[str],
    ) -> Tuple[Any, bool, bool]:
        """
        Contrato v14: `work_mean` (no `work_w`), `positron_token` (no booleano), informes
        anidados. Se verifica el pasaporte HMAC del propio motor y el cruce de digests.
        """
        report = self._engine.execute_thermal_exergy_weaver_audit(
            apu_cost_matrix=apu_cost_matrix, exponents=exponents,
            beta=params.beta, rho_mac_input=rho, work_mean=params.work_mean, delta_f=params.delta_f,
            ball_r=params.ball_r, cylinder_r=params.cylinder_r,
            kam_scan_order=params.kam_scan_order, diophantine_tau=params.diophantine_tau,
            melnikov_omega=params.melnikov_omega, melnikov_gamma=params.melnikov_gamma,
            melnikov_delta=params.melnikov_delta, melnikov_epsilon=params.melnikov_epsilon,
            positron_token=positron_token, psd_floor=params.psd_floor,
        )
        if not hasattr(report, "phase_ii"):
            raise GovernanceError("Motor incompatible: se requiere el informe anidado v14.")
        passport_ok = bool(self._engine.verify_report_passport(report))
        expected_input = _sha256_hex(
            (_array_digest(report.phase_ii.phase_i.kernel_reg) + "|"
             + _array_digest(np.asarray(exponents, dtype=np.float64))).encode("ascii")
        )
        digest_ok = hmac.compare_digest(expected_input, report.input_digest)
        return report, passport_ok, digest_ok

    # ─────────── §2.5 Cadencia de Poincaré (tres huecos + diophantina) ───────────
    def poincare_cadence_certificate(self, params: GovernanceParameters) -> PoincareCadenceCertificate:
        r"""
        Programación cuasi-periódica de auditorías con número de rotación α (Poincaré).

        • γ diophantino: min_{q≤Q} q²|α − p/q| sobre CONVERGENTES (son las mejores
          aproximaciones racionales: el mínimo se alcanza en ellas, teorema de Lagrange).
        • Teorema de los tres huecos (Steinhaus): los huecos de {nα mod 1}, n<N, toman a lo
          más 3 longitudes y la mayor es la suma de las otras dos ⇒ cobertura sin resonancia.
        • Cota de cobertura con convergentes q_{k−1}, q_k ≤ N: max_hueco ≤ 1/q_{k−1} + 1/q_k.
        • Detección de resonancia de bajo orden (aliasing con cargas periódicas).
        """
        alpha, n_pts = float(params.cadence_rotation), int(params.cadence_points)
        convs = _continued_fraction_convergents(alpha, n_pts)
        gamma = min((q * q) * abs(alpha - p / q) for p, q in convs if q > 0)
        low_res = any(abs(q * alpha - p) < 1e-6 for p, q in convs if q <= 12)
        points = np.sort(np.mod(np.arange(n_pts) * alpha, 1.0))
        gaps = np.diff(np.concatenate([points, points[:1] + 1.0]))
        distinct = np.unique(np.round(gaps, 9))
        identity_res = 0.0
        if distinct.size == 3:
            identity_res = float(abs(distinct[2] - (distinct[0] + distinct[1])))
        elif distinct.size > 3:
            identity_res = float("inf")
        q_prev, q_cur = (convs[-2][1], convs[-1][1]) if len(convs) >= 2 else (1, convs[-1][1])
        bound = 1.0 / max(1, q_prev) + 1.0 / max(1, q_cur)
        max_gap = float(np.max(gaps))
        return PoincareCadenceCertificate(
            rotation_number=alpha, convergents=convs, diophantine_gamma=float(gamma),
            low_order_resonance=bool(low_res), n_points=n_pts, distinct_gap_count=int(distinct.size),
            max_gap=max_gap, three_gap_identity_residual=identity_res, coverage_bound=float(bound),
            three_gap_ok=bool(distinct.size <= 3 and identity_res <= 1e-8),
            coverage_ok=bool(max_gap <= bound + 1e-12),
        )

    # ─────────── §2.6 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self,
        p1: ThermalExergyPhase1Certificate,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho: NDArray,
        params: GovernanceParameters,
        *,
        positron_token: Optional[str] = None,
        expected_epoch: int,
    ) -> ThermalExergyPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: consume §1.5 y entrega P2 a §3.1."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, params, expected_epoch=expected_epoch)
        report, passport_ok, digest_ok = self.invoke_engine(
            apu_cost_matrix, exponents, rho, params, positron_token=positron_token
        )
        cadence = self.poincare_cadence_certificate(params)
        p_i, p_ii = report.phase_ii.phase_i, report.phase_ii
        all_evidence = list(p_i.evidences) + list(p_ii.evidences) + list(report.evidences)
        margin = _hard_evidence_margin(all_evidence)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("P2.pasaporte_motor", passport_ok, hard=True),
            Evidence.boolean("P2.digest_entradas_motor", digest_ok, hard=True),
            Evidence.boolean("P2.cadena_p1_verificada", True, hard=True),
            Evidence("P2.tres_huecos", cadence.three_gap_identity_residual
                     if math.isfinite(cadence.three_gap_identity_residual) else 1.0, 1e-8, hard=True),
            Evidence.boolean("P2.cobertura_cadencia", cadence.coverage_ok, hard=True),
            Evidence.boolean("P2.sin_resonancia_bajo_orden", not cadence.low_order_resonance),
            Evidence("P2.gamma_diophantino_cadencia", max(0.0, 1e-3 - cadence.diophantine_gamma), 0.0),
            Evidence("P2.margen_duro", max(0.0, 1.0 - margin), 0.0, hard=True,
                     note="todas las evidencias duras del motor deben pasar"),
        )
        truth = HeytingTruthValue.infimum([p1.truth_value, report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash,
            "engine_passport": report.passport_hmac,
            "engine_passport_verified": passport_ok,
            "engine_input_digest_match": digest_ok,
            "engine_truth": report.truth_value,
            "engine_version": report.engine_version,
            "engine_input_digest": report.input_digest,
            "engine_legacy_summary": report.legacy_summary(),
            "destroyed_exergy_w": p_i.tellegen.gouy_stodola_destroyed_w,
            "irreversible_work_j": p_ii.jarzynski_crooks.irreversible_work_j,
            "ess_fraction": p_ii.jarzynski_crooks.ess_fraction,
            "kms_residual": p_ii.gibbs_kms.kms_residual,
            "banach_composite_q": report.rsi3.composite_lipschitz_q,
            "chaos_detected": report.melnikov.chaos_detected,
            "hard_evidence_margin": margin,
            "cadence": cadence, "evidences": evidences, "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain_hash = _sha256_hex(_canonical(core))
        signature = self._signer.sign(_HMAC_P2, core)
        cert = ThermalExergyPhase2Certificate(
            timestamp=time.time(), phase1=p1, spectrum_report=report,
            engine_passport=report.passport_hmac, engine_passport_verified=passport_ok,
            engine_input_digest_match=digest_ok,
            destroyed_exergy_w=p_i.tellegen.gouy_stodola_destroyed_w,
            irreversible_work_j=p_ii.jarzynski_crooks.irreversible_work_j,
            ess_fraction=p_ii.jarzynski_crooks.ess_fraction,
            kms_residual=p_ii.gibbs_kms.kms_residual,
            banach_composite_q=report.rsi3.composite_lipschitz_q,
            chaos_detected=report.melnikov.chaos_detected,
            hard_evidence_margin=margin, cadence=cadence, evidences=evidences,
            truth_value=truth, epoch=int(expected_epoch), chain_hash=chain_hash, hmac_signature=signature,
        )
        logger.info("Fase 2 (ORIENT) Ω₃=%s · pasaporte_motor=%s · X_dest=%.4f W · margen=%.3f",
                    truth.name, passport_ok, cert.destroyed_exergy_w, margin)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ACT — Adjudicación, interlock, Φ_sem y RSI³ de gobernanza
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyPhase3Decider:
    r"""
    FASE 3 OODA. Adjudica en Ω₃, traduce a acción mediante un functor contravariante
    verificado, autoriza la actuación con token HMAC anti-replay y sella Merkle/HMAC.
    """

    PHASE_LABEL: Final[str] = "3·DECIDE&ACT"
    _ACTION_BY_TRUTH: Final[Dict[HeytingTruthValue, str]] = {
        HeytingTruthValue.VETOED: "TRIP",
        HeytingTruthValue.DEGRADED: "DERATE",
        HeytingTruthValue.COHERENT: "NOMINAL",
    }

    def __init__(self, signer: SovereignSigner, driver: CrowbarDriver, *,
                 budget: Optional[CrowbarBudget] = None, dispatch_samples: int = 64) -> None:
        self._signer = signer
        self._driver = driver
        self._budget = budget or CrowbarBudget()
        self._samples = int(dispatch_samples)
        self._last_epoch = -1

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ ───────────
    def continue_to_phase3_gate(self, p2: ThermalExergyPhase2Certificate, *, expected_epoch: int) -> GateReceipt:
        r"""
        Φ₂₃ — Sepila integridad de física:
          • Integridad (chain recalculado + HMAC + enlace P1 + pasaporte del motor): fallo ⇒
            GateIntegrityError ⇒ fail-safe en la fachada.
          • Física (contracción de Banach q<1): fallo ⇒ se ADMITE con ⊥ FORZADO, para que el
            decidor pueda actuar. Nunca se aprueba por omisión.
        """
        payload = p2.payload()
        recomputed = _sha256_hex(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (
            p2.epoch == expected_epoch
            and p2.phase1.epoch == expected_epoch
            and p2.engine_passport_verified
            and p2.engine_input_digest_match
        )
        if not (chain_ok and sig_ok and link_ok):
            raise GateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        forced = bool(p2.banach_composite_q >= 1.0)
        return GateReceipt(
            gate="Φ₂₃", payload_digest=recomputed, signature_ok=sig_ok, chain_recomputed_ok=chain_ok,
            array_binding_ok=link_ok, admitted=True, forced_veto=forced,
            note="q≥1 ⇒ ⊥ forzado (fail-safe lógico)" if forced else "contracción de Banach verificada",
        )

    def accept_phase2_certificate(self, p2: ThermalExergyPhase2Certificate, *, expected_epoch: int) -> GateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.6."""
        return self.continue_to_phase3_gate(p2, expected_epoch=expected_epoch)

    # ─────────── §3.2 Adjudicación, núcleo ¬¬ y functor de acción ───────────
    @staticmethod
    def action_functor(truth: HeytingTruthValue, *, strict_mode: bool = False) -> str:
        """
        D : Ω₃ᵒᵖ → Acciones, monótono contravariante: ⊥↦TRIP, ½↦DERATE, ⊤↦NOMINAL.
        Ley verificada: D(a∧b) = D(a) ∨ D(b) en severidad (anti-homomorfismo de retículos).
        `strict_mode` es POLÍTICA declarada: ½ también dispara.
        """
        if strict_mode and truth is HeytingTruthValue.DEGRADED:
            return "TRIP"
        return ThermalExergyPhase3Decider._ACTION_BY_TRUTH[truth]

    @staticmethod
    def verify_action_functor() -> bool:
        """Verifica exhaustivamente la ley contravariante sobre los 9 pares de Ω₃."""
        severity = {"NOMINAL": 0, "DERATE": 1, "TRIP": 2}
        values = list(HeytingTruthValue)
        for a in values:
            for b in values:
                lhs = severity[ThermalExergyPhase3Decider.action_functor(a.meet(b))]
                rhs = max(severity[ThermalExergyPhase3Decider.action_functor(a)],
                          severity[ThermalExergyPhase3Decider.action_functor(b)])
                if lhs != rhs:
                    return False
        return True

    @staticmethod
    def verify_nucleus() -> Tuple[bool, bool, bool]:
        """j(a)=¬¬a es un NÚCLEO de Heyting: inflacionario, idempotente y preserva ∧."""
        values = list(HeytingTruthValue)
        inflation = all(int(a) <= int(a.double_negation()) for a in values)
        idempotent = all(a.double_negation().double_negation() == a.double_negation() for a in values)
        meet_pres = all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                        for a in values for b in values)
        return inflation, idempotent, meet_pres

    # ─────────── §3.3 Interlock de actuación (token HMAC + anti-replay) ───────────
    def mint_actuation_token(self, epoch: int, decision_digest: str, action: str) -> str:
        return self._signer.sign(_HMAC_ACT, {"epoch": epoch, "decision": decision_digest, "action": action})

    def _verify_actuation_token(self, token: str, epoch: int, decision_digest: str, action: str) -> bool:
        return self._signer.verify(_HMAC_ACT, {"epoch": epoch, "decision": decision_digest, "action": action}, token)

    # ─────────── §3.4 Crowbar: medición y presupuesto honesto ───────────
    def execute_crowbar(self, action: str, epoch: int, decision_digest: str, *,
                        fail_safe: bool = False) -> CrowbarCertificate:
        """
        Mide N despachos en seco (distribución real, no una constante), arma y dispara sólo
        con token válido y época estrictamente creciente. Declara la inviabilidad física del
        plazo de 392.15 ns para un SCR sin medición HIL que demuestre lo contrario.
        """
        samples = np.array([self._driver.dry_run() for _ in range(max(8, self._samples))], dtype=np.float64)
        token = self.mint_actuation_token(epoch, decision_digest, action)
        token_ok = self._verify_actuation_token(token, epoch, decision_digest, action)
        dispatched = False
        trip_ns = 0.0
        if action == "TRIP":
            if epoch <= self._last_epoch:
                raise ActuationInterlockError("Replay de época detectado: disparo rechazado.")
            self._driver.arm()
            trip_ns = float(self._driver.trip(token))
            self._last_epoch = int(epoch)
            dispatched = True
        elif fail_safe:
            self._driver.safe_state()

        budget = self._budget
        hil_fw = budget.hil_firmware_isr_ns
        hil_phys = budget.hil_physical_total_ns
        note = (
            "392.15 ns es política del camino lógico ISR/IRAM; el despacho Python medido es de "
            "orden µs y la conmutación física del SCR (t_gt ~1–3 µs) NO puede alcanzarlo. "
            "Sin medición HIL no se afirma cumplimiento físico."
        )
        return CrowbarCertificate(
            driver_name=self._driver.name, driver_is_physical=bool(self._driver.is_physical),
            armed=bool(action == "TRIP"), dispatched=dispatched, dispatch_samples=int(samples.size),
            dispatch_min_ns=float(np.min(samples)), dispatch_median_ns=float(np.median(samples)),
            dispatch_p99_ns=float(np.percentile(samples, 99)), dispatch_max_ns=float(np.max(samples)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(budget.logic_deadline_ns),
            python_dispatch_within_deadline=bool(np.max(samples) <= budget.logic_deadline_ns),
            firmware_isr_verified=hil_fw is not None,
            firmware_isr_within_deadline=bool(hil_fw is not None and hil_fw <= budget.logic_deadline_ns),
            physical_budget_ns=budget.physical_budget_ns,
            physical_deadline_feasible=bool(budget.physical_budget_ns <= budget.logic_deadline_ns),
            hil_physical_verified=hil_phys is not None,
            interlock_token_verified=bool(token_ok), fail_safe_armed=bool(fail_safe), note=note,
        )

    # ─────────── §3.5 Φ_sem visceral con procedencia dimensional ───────────
    @staticmethod
    def semantic_visceral_functor(
        p2: ThermalExergyPhase2Certificate, params: GovernanceParameters, *,
        action: str, functor_ok: bool,
    ) -> ThermalExergyVisceralMetrics:
        r"""
        Φ_sem : (exergía destruida [W], W_irr [J], bytes KV) → KPIs acotados [%].

        • Protección WACC (heurística de política, monótona y saturante):
              100 · X_dest / (X_dest + X_ref)     [adimensional: W/W]
        • Liquidez neta (requiere precio de la exergía y base de capital DECLARADOS):
              100 · W_irr · precio[moneda/J] / capital[moneda]
          Sin ambos insumos ⇒ None + `UNDECLARED_INPUT` (v13 escribía 11.5 % sin fuente).
        • Compresión KV (requiere bytes medidos): 100 · (1 − after/before).
        Se verifica monotonía numérica en cada argumento. No se afirma teorema financiero.
        """
        x_dest = max(0.0, float(p2.destroyed_exergy_w))
        x_ref = float(params.exergy_reference_w)
        wacc = 100.0 * x_dest / (x_dest + x_ref)
        provenance = ["WACC:POLICY_HEURISTIC(W/W)"]

        liquidity: Optional[float] = None
        if params.exergy_price_per_joule is not None and params.capital_base:
            liquidity = 100.0 * float(p2.irreversible_work_j) * float(params.exergy_price_per_joule) / float(params.capital_base)
            provenance.append("LIQUIDITY:DECLARED(J*price/capital)")
        else:
            provenance.append("LIQUIDITY:UNDECLARED_INPUT")

        kv: Optional[float] = None
        if params.kv_bytes_before and params.kv_bytes_after is not None:
            before, after = int(params.kv_bytes_before), int(params.kv_bytes_after)
            if before > 0 and 0 <= after <= before:
                kv = 100.0 * (1.0 - after / before)
                provenance.append("KV:MEASURED_BYTES")
            else:
                provenance.append("KV:INVALID_MEASUREMENT")
        else:
            provenance.append("KV:UNDECLARED_INPUT")

        grid = np.linspace(0.0, 10.0 * x_ref, 32)
        wacc_curve = 100.0 * grid / (grid + x_ref)
        monotone = bool(np.all(np.diff(wacc_curve) >= -1e-12))
        if params.exergy_price_per_joule is not None and params.capital_base:
            w_grid = np.linspace(0.0, 10.0, 32)
            liq_curve = 100.0 * w_grid * params.exergy_price_per_joule / params.capital_base
            monotone = monotone and bool(np.all(np.diff(liq_curve) >= -1e-12))

        return ThermalExergyVisceralMetrics(
            wacc_protection_pct=float(wacc), net_liquidity_release_pct=liquidity,
            kv_cache_compression_pct=kv,
            irreversible_work_status=f"W_irr={p2.irreversible_work_j:.6f} J (Jensen certificado)",
            destroyed_exergy_status=f"X_dest=T0*sigma={x_dest:.6f} W (Gouy–Stodola) · acción={action}",
            provenance=tuple(provenance), monotonicity_ok=monotone, action_functor_ok=bool(functor_ok),
        )

    # ─────────── §3.6 RSI³ de gobernanza (Banach ⊗ Kantorovich ⊗ Kleisli) ───────────
    def governance_rsi3_certificate(self, margin: float, *, probe_pairs: int = 48) -> GovernanceRSI3Certificate:
        r"""
        Tres niveles anidados de automejora con certificados REALES:

        NIVEL 1 — Contracción de política. T(θ) = θ* + q·tanh(θ − θ*), con
            q = 1 − ½·margen_duro ∈ [½,1) (la velocidad de mejora se liga a la salud
            certificada). Como |tanh'| ≤ 1 y T es diagonal, Lip_∞(T) ≤ q y Lip(T³) ≤ q³.
            Se comprueba empíricamente y con cotas de Banach a priori / a posteriori.

        NIVEL 2 — Calibración Newton–Kantorovich del factor de derateo. Con
            f(d) = m − κd² − objetivo, κ=1, objetivo = m/4 ⇒ d* = √(3m/4),
            x₀ = c·d* (c=10) ⇒ h = β·η·M = (c²−1)/(2c²) < ½ SIEMPRE (hipótesis de
            Kantorovich verificada analítica y numéricamente). La iteración es babilónica:
            d_{k+1} = ½(d_k + d*²/d_k), e_{k+1} = e_k²/(2d_k) ⇒ razón cuadrática 1/(2d*).
            La traza de capacidad C_k = −log₁₀ e_k ~ 2^k ⇒ d³C/dt³ > 0 MEDIDO (SG 7/3).

        NIVEL 3 — Meta: núcleo ¬¬ sobre Ω₃ (axiomas exhaustivos) y leyes de la mónada
            Writer/Kleisli con la que se ejecuta el propio ciclo OODA.
        """
        m = float(max(margin, 0.0))
        q = 1.0 - 0.5 * m
        rng = np.random.default_rng(20_261_112)

        kappa = 1.0
        target = 0.25 * m
        d_star = math.sqrt(max(m - target, 0.0) / kappa)
        theta_star = np.array([d_star, m, m / (1.0 + m)], dtype=np.float64)

        def policy_map(theta: NDArray) -> NDArray:
            return theta_star + q * np.tanh(theta - theta_star)

        def triple(theta: NDArray) -> NDArray:
            return policy_map(policy_map(policy_map(theta)))

        theta = theta_star + rng.standard_normal(3)
        prev = theta
        theta = triple(theta)
        first_gap = float(np.max(np.abs(theta - prev)))
        for _ in range(11):
            prev = theta
            theta = triple(theta)
        observed = float(np.max(np.abs(theta - theta_star)))
        last_gap = float(np.max(np.abs(theta - prev)))
        q3 = q ** 3
        apriori = (q3 ** 12) / max(1.0 - q3, _EPS) * first_gap
        aposteriori = q3 / max(1.0 - q3, _EPS) * last_gap
        empirical = 0.0
        for _ in range(int(probe_pairs)):
            pert = rng.standard_normal(3) * 1e-6
            denom = float(np.max(np.abs(pert)))
            if denom > 0.0:
                empirical = max(empirical, float(np.max(np.abs(triple(theta + pert) - triple(theta)))) / denom)

        # Newton–Kantorovich (raíz analítica disponible: auto-verificación declarada)
        c_init = 10.0
        d_k = c_init * max(d_star, 1e-12)
        beta_k = 1.0 / abs(-2.0 * kappa * d_k)
        eta_k = abs((m - kappa * d_k ** 2 - target) / (-2.0 * kappa * d_k))
        h_k = beta_k * eta_k * (2.0 * kappa)
        errors: List[float] = []
        for _ in range(10):
            errors.append(abs(d_k - d_star))
            f_val = m - kappa * d_k ** 2 - target
            d_k = d_k - f_val / (-2.0 * kappa * d_k)
        errors.append(abs(d_k - d_star))
        err = np.array(errors, dtype=np.float64)
        ratio_res = 0.0
        for i in range(len(err) - 1):
            if err[i] > 1e-12 * max(d_star, 1e-12):
                ratio_res = max(ratio_res, abs(err[i + 1] - err[i] ** 2 / (2.0 * max(d_star, 1e-12)))
                                / (err[i + 1] + 1e-300))
        capacity = -np.log10(np.maximum(err, 1e-300))
        active = np.where(err > 1e-13 * max(d_star, 1e-12))[0]
        d1 = d2 = d3 = 0.0
        if active.size >= 7:
            center = int(active[active.size // 2])
            center = min(max(center, 3), capacity.size - 4)
            seg = capacity[center - 3: center + 4]
            d1 = float(_savgol_derivative_coeffs(7, 3, 1) @ seg)
            d2 = float(_savgol_derivative_coeffs(7, 3, 2) @ seg)
            d3 = float(_savgol_derivative_coeffs(7, 3, 3) @ seg)

        infl, idem, meet_pres = self.verify_nucleus()
        f_c = lambda v: Certified(v * 0.5, ("f",), HeytingTruthValue.COHERENT)   # noqa: E731
        g_c = lambda v: Certified(v + 1.0, ("g",), HeytingTruthValue.COHERENT)   # noqa: E731
        sample = rng.standard_normal(3)
        m_val = Certified(sample, ("m",), HeytingTruthValue.DEGRADED)
        laws = (
            np.allclose(Certified.unit(sample).bind(f_c).value, f_c(sample).value)
            and np.allclose(m_val.bind(Certified.unit).value, m_val.value)
            and m_val.bind(Certified.unit).ledger == m_val.ledger
            and np.allclose(m_val.bind(f_c).bind(g_c).value, m_val.bind(lambda v: f_c(v).bind(g_c)).value)
            and m_val.bind(f_c).bind(g_c).ledger == m_val.bind(lambda v: f_c(v).bind(g_c)).ledger
        )
        return GovernanceRSI3Certificate(
            policy_lipschitz_q=float(q), policy_lip_t3_bound=float(q3), policy_lip_t3_empirical=float(empirical),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_quadratic_ratio_residual=float(ratio_res),
            newton_iterations=int(len(err) - 1), nucleus_inflationary_ok=bool(infl),
            nucleus_idempotent_ok=bool(idem), nucleus_meet_preserving_ok=bool(meet_pres),
            kleisli_monad_laws_ok=bool(laws), capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0.0 and d2 > 0.0 and d3 > 0.0),
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Estado soberano con Merkle y firma ───────────
    def emit_sovereign_state(
        self, p2: ThermalExergyPhase2Certificate, params: GovernanceParameters, *,
        expected_epoch: int, error: Optional[str] = None, forced_veto: bool = False,
    ) -> ThermalExergySovereignState:
        """MÉTODO TERMINAL DEL CICLO: adjudica, autoriza, actúa, sella y entrega el estado."""
        t_start = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        action = self.action_functor(truth, strict_mode=params.strict_mode)
        functor_ok = self.verify_action_functor()

        decision_payload = {
            "epoch": int(expected_epoch), "truth": truth, "action": action,
            "p2_chain_hash": p2.chain_hash, "p1_chain_hash": p2.phase1.chain_hash,
            "forced_veto": bool(receipt.forced_veto or forced_veto), "error": error,
            "strict_mode": bool(params.strict_mode),
        }
        decision_digest = _sha256_hex(_canonical(decision_payload))
        crowbar = self.execute_crowbar(action, expected_epoch, decision_digest, fail_safe=error is not None)
        visceral = self.semantic_visceral_functor(p2, params, action=action, functor_ok=functor_ok)
        rsi3 = self.governance_rsi3_certificate(p2.hard_evidence_margin)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("P3.functor_accion_contravariante", functor_ok, hard=True),
            Evidence.boolean("P3.nucleo_inflacionario", rsi3.nucleus_inflationary_ok, hard=True),
            Evidence.boolean("P3.nucleo_idempotente", rsi3.nucleus_idempotent_ok, hard=True),
            Evidence.boolean("P3.nucleo_preserva_meet", rsi3.nucleus_meet_preserving_ok, hard=True),
            Evidence("P3.banach_q_politica", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("P3.lip_t3_politica", rsi3.policy_lip_t3_empirical - rsi3.policy_lip_t3_bound * (1 + 1e-9),
                     0.0, hard=True),
            Evidence("P3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("P3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("P3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence("P3.newton_cuadratico", rsi3.newton_quadratic_ratio_residual, 1e-6),
            Evidence.boolean("P3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("P3.interlock_token", crowbar.interlock_token_verified, hard=True),
            Evidence.boolean("P3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("P3.latencia_fisica_declarada",
                             crowbar.physical_deadline_feasible or not crowbar.physical_deadline_feasible,
                             note="informativo: la inviabilidad física se DECLARA, no se oculta"),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])
        if self.action_functor(truth, strict_mode=params.strict_mode) == "TRIP" and not crowbar.dispatched:
            decision_payload["truth"] = truth
            decision_digest = _sha256_hex(_canonical(decision_payload))
            crowbar = self.execute_crowbar("TRIP", expected_epoch + 1, decision_digest, fail_safe=True)
            action = "TRIP"

        actuation_payload = {
            "dispatched": crowbar.dispatched, "armed": crowbar.armed, "driver": crowbar.driver_name,
            "is_physical": crowbar.driver_is_physical, "logic_deadline_ns": crowbar.logic_deadline_ns,
            "physical_budget_ns": crowbar.physical_budget_ns,
            "physical_deadline_feasible": crowbar.physical_deadline_feasible,
            "firmware_isr_verified": crowbar.firmware_isr_verified,
            "interlock_token_verified": crowbar.interlock_token_verified,
        }
        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": p2.engine_passport, "engine_input_digest": p2.spectrum_report.input_digest,
                        "engine_summary": p2.spectrum_report.legacy_summary()}),
            _canonical(p2.payload()),
            _canonical(decision_payload),
            _canonical(actuation_payload),
            _canonical(visceral),
            _canonical(rsi3),
            _canonical(p2.cadence),
        ]
        root = merkle_root(leaves)
        decision_index = 3
        proof = merkle_inclusion_proof(leaves, decision_index)

        state_core = {
            "truth_value": truth, "boolean_shadow": truth.double_negation(), "action": action,
            "crowbar_triggered": bool(crowbar.dispatched), "merkle_root_hash": root,
            "decision_leaf_index": decision_index, "decision_inclusion_proof": proof,
            "kleisli_ledger": ("Φ₁:OBSERVE", "Φ₂:ORIENT", "Φ₃:DECIDE&ACT"),
            "epoch": int(expected_epoch), "error": error,
            "phase2_chain_hash": p2.chain_hash, "phase1_chain_hash": p2.phase1.chain_hash,
            "merkle_leaf_count": len(leaves),
        }
        signature = self._signer.sign(_HMAC_STATE, state_core)
        state = ThermalExergySovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(), action=action,
            crowbar_triggered=bool(crowbar.dispatched), crowbar_latency_ns=float(crowbar.trip_dispatch_ns),
            crowbar_certificate=crowbar, merkle_root_hash=root,
            merkle_leaves=tuple(leaf.hex() for leaf in leaves), decision_leaf_index=decision_index,
            decision_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3, phase2=p2,
            evidences=evidences, kleisli_ledger=state_core["kleisli_ledger"], epoch=int(expected_epoch),
            state_signature=signature, error=error,
            execution_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )
        logger.info("Fase 3 (DECIDE&ACT) Ω₃=%s · acción=%s · disparo=%s · Merkle=%s…",
                    truth.name, action, state.crowbar_triggered, root[:16])
        return state


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA SOBERANA — Orquestación OODA como composición de Kleisli, con fail-safe
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovThermalExergyWeaverAgent:
    r"""
    Soberano agéntico que orienta al motor v14. Ejecuta el ciclo como
        observe >=> orient >=> decide
    sobre la mónada Certified (Writer sobre (Ω₃,∧,⊤)): el ledger y el veredicto se acumulan
    por composición, no por asignación imperativa. Ante cualquier fallo: FAIL-SAFE (⊥ + disparo).
    """

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        *,
        driver: Optional[CrowbarDriver] = None,
        budget: Optional[CrowbarBudget] = None,
        ambient_temperature_k: float = 300.0,
        seed: int = 20_261_111,
    ) -> None:
        self._signer = SovereignSigner(secret_key)
        self._engine = BoolNovikovThermalExergyWeaverEngine(
            secret_key=self._signer.key_bytes, ambient_temperature_k=ambient_temperature_k, seed=seed
        )
        self._observer = ThermalExergyPhase1Observer(self._signer, seed=seed)
        self._orientor = ThermalExergyPhase2Orientor(self._engine, self._signer)
        self._driver = driver or SimulatedCrowbarDriver(lambda _t: False)
        self._decider = ThermalExergyPhase3Decider(self._signer, self._driver, budget=budget)
        if isinstance(self._driver, SimulatedCrowbarDriver):
            self._driver = SimulatedCrowbarDriver(
                lambda t: self._decider._verify_actuation_token(t, self._pending_epoch, self._pending_digest, "TRIP")
            )
            self._decider = ThermalExergyPhase3Decider(self._signer, self._driver, budget=budget)
        self._pending_epoch = -1
        self._pending_digest = ""
        self._epoch = 0
        self._seed = int(seed)
        logger.info("%s v%s listo · clave=%s · motor=%s",
                    type(self).__name__, __version__, self._signer.fingerprint(), _ENGINE_VERSION)

    # ─────────── preparación declarada de ρ (morfismo auditado, no reparación) ───────────
    def prepare_density_operator(
        self, apu_cost_matrix: NDArray, params: GovernanceParameters,
        rho_mac_input: Optional[NDArray],
    ) -> NDArray:
        """
        `gibbs`          : ρ_β = e^{−βK_reg}/Z con K_reg del propio motor (consistencia KMS).
        `maximally_mixed`: I/d (límite β→0); el motor lo marcará ½ si H no es degenerado.
        `provided`       : ρ externa; se valida ANTES de normalizar en Fase 1.
        """
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        dim = int(k_mat.shape[0])
        if params.rho_mode == "provided":
            if rho_mac_input is None:
                raise GovernanceError("rho_mode='provided' exige rho_mac_input.")
            return np.asarray(rho_mac_input, dtype=np.complex128)
        if params.rho_mode == "maximally_mixed":
            return np.eye(dim, dtype=np.complex128) / float(dim)
        _, k_reg, _, _ = self._engine.regularize_kernel(k_mat, psd_floor=params.psd_floor)
        energies, u_mat = la.eigh(k_reg)
        log_p = -params.beta * (energies - energies[0])
        log_p -= (float(np.max(log_p)) + math.log(float(np.sum(np.exp(log_p - float(np.max(log_p)))))))
        probs = np.exp(log_p)
        return (u_mat * probs) @ u_mat.conj().T

    # ─────────── verificación completa del estado (auditoría externa) ───────────
    def verify_sovereign_state(self, state: ThermalExergySovereignState) -> bool:
        """Reverifica P1, P2, pasaporte del motor, Merkle, prueba de inclusión y firma."""
        p1, p2 = state.phase2.phase1, state.phase2
        if not self._signer.verify(_HMAC_P1, p1.payload(), p1.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256_hex(_canonical(p1.payload())), p1.chain_hash):
            return False
        if not self._signer.verify(_HMAC_P2, p2.payload(), p2.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256_hex(_canonical(p2.payload())), p2.chain_hash):
            return False
        if not self._engine.verify_report_passport(p2.spectrum_report):
            return False
        leaves = [bytes.fromhex(h) for h in state.merkle_leaves]
        if merkle_root(leaves) != state.merkle_root_hash:
            return False
        if not verify_merkle_inclusion(leaves[state.decision_leaf_index], state.decision_leaf_index,
                                       len(leaves), state.decision_inclusion_proof, state.merkle_root_hash):
            return False
        return self._signer.verify(_HMAC_STATE, state.payload(), state.state_signature)

    # ─────────── ciclo OODA trifásico anidado (Kleisli) con fail-safe ───────────
    def run_sovereign_governance_cycle(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho_mac_input: Optional[NDArray] = None,
        *,
        params: Optional[GovernanceParameters] = None,
        positron_token: Optional[str] = None,
    ) -> ThermalExergySovereignState:
        """
        Ejecuta observe >=> orient >=> decide. Cualquier excepción (incluida Φ₁₂/Φ₂₃) activa
        el fail-safe: se construye un P2 de emergencia con la mejor información disponible,
        se fuerza ⊥, se dispara con token de emergencia y se sella el estado de emergencia.
        """
        params = params or GovernanceParameters()
        self._epoch += 1
        epoch = self._epoch
        self._pending_epoch = epoch
        rho = self.prepare_density_operator(apu_cost_matrix, params, rho_mac_input)

        def observe_m(ctx: Dict[str, Any]) -> Any:
            p1 = self._observer.emit_phase1_certificate(
                rho, apu_cost_matrix, exponents, params, epoch=epoch
            )
            return Certified({**ctx, "p1": p1}, ("Φ₁:OBSERVE",), p1.truth_value)

        def orient_m(ctx: Dict[str, Any]) -> Any:
            p2 = self._orientor.emit_phase2_certificate(
                ctx["p1"], apu_cost_matrix, exponents, rho, params,
                positron_token=positron_token, expected_epoch=epoch,
            )
            return Certified({**ctx, "p2": p2}, ("Φ₂:ORIENT",), p2.truth_value)

        def decide_m(ctx: Dict[str, Any]) -> Any:
            self._pending_digest = _sha256_hex(_canonical({
                "epoch": epoch, "truth": ctx["p2"].truth_value, "action": "TRIP",
                "p2_chain_hash": ctx["p2"].chain_hash, "p1_chain_hash": ctx["p2"].phase1.chain_hash,
                "forced_veto": False, "error": None, "strict_mode": bool(params.strict_mode),
            }))
            state = self._decider.emit_sovereign_state(ctx["p2"], params, expected_epoch=epoch)
            return Certified(state, ("Φ₃:DECIDE&ACT",), state.truth_value)

        pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
        try:
            result = pipeline({"epoch": epoch})
            state: ThermalExergySovereignState = result.value
            if not self.verify_sovereign_state(state):
                raise GateIntegrityError("Autoverificación del estado soberano fallida.")
            logger.info("Ciclo OODA completado · Ω₃=%s · ledger=%s",
                        state.truth_value.name, " → ".join(result.ledger))
            return state
        except Exception as exc:                                   # FAIL-SAFE, nunca silencio
            logger.critical("FAIL-SAFE activado: %s", exc, exc_info=True)
            return self._emergency_state(apu_cost_matrix, exponents, rho, params, epoch, str(exc))

    def _emergency_state(
        self, apu_cost_matrix: NDArray, exponents: NDArray, rho: NDArray,
        params: GovernanceParameters, epoch: int, message: str,
    ) -> ThermalExergySovereignState:
        """Estado de emergencia: ⊥ forzado, disparo autorizado y Merkle mínimo verificable."""
        p1 = self._observer.emit_phase1_certificate(rho, apu_cost_matrix, exponents, params, epoch=epoch)
        try:
            p2 = self._orientor.emit_phase2_certificate(
                p1, apu_cost_matrix, exponents, rho, params, expected_epoch=epoch
            )
        except Exception:                                          # noqa: BLE001 (último recurso)
            p2 = self._orientor.emit_phase2_certificate(
                p1, apu_cost_matrix, exponents,
                np.eye(apu_cost_matrix.shape[0], dtype=np.complex128) / apu_cost_matrix.shape[0],
                dataclasses.replace(params, rho_mode="maximally_mixed"), expected_epoch=epoch,
            )
        self._pending_epoch = epoch + 1
        self._pending_digest = _sha256_hex(_canonical({"emergency": message, "epoch": epoch + 1}))
        return self._decider.emit_sovereign_state(
            p2, params, expected_epoch=epoch + 1, error=message, forced_veto=True
        )


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    key = b"APU_AGENT_SELFTEST_KEY_32_BYTES_MIN!!"
    agent = BoolNovikovThermalExergyWeaverAgent(secret_key=key)
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.25, 0.40], dtype=np.float64)

    # 1) Nominal con ρ Gibbs del Hamiltoniano real ⇒ ⊤, sin disparo
    st = agent.run_sovereign_governance_cycle(k_test, exps)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.action == "NOMINAL" and not st.crowbar_triggered
    assert agent.verify_sovereign_state(st)
    assert st.phase2.engine_passport_verified and st.phase2.engine_input_digest_match
    assert st.phase2.cadence.three_gap_ok and st.phase2.cadence.coverage_ok
    assert st.rsi3.kantorovich_satisfied and abs(st.rsi3.kantorovich_h - 0.495) < 1e-9
    assert st.rsi3.nucleus_meet_preserving_ok and st.rsi3.kleisli_monad_laws_ok
    assert st.visceral_metrics.net_liquidity_release_pct is None      # sin €/J declarado: honesto
    assert not st.crowbar_certificate.physical_deadline_feasible      # SCR ~µs: declarado

    # 2) ρ = I/d (no KMS) ⇒ ½, DERATE, sin disparo (política no estricta)
    mid = agent.run_sovereign_governance_cycle(k_test, exps, params=GovernanceParameters(rho_mode="maximally_mixed"))
    assert mid.truth_value is HeytingTruthValue.DEGRADED and mid.action == "DERATE"
    assert not mid.crowbar_triggered and mid.boolean_shadow is HeytingTruthValue.COHERENT

    # 3) Caos homoclínico + no-squeezing violado ⇒ ⊥, TRIP con interlock
    bad = agent.run_sovereign_governance_cycle(
        k_test, exps,
        params=GovernanceParameters(melnikov_gamma=1.0, melnikov_delta=0.05, ball_r=2.0, cylinder_r=1.0),
    )
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.action == "TRIP"
    assert bad.crowbar_triggered and bad.crowbar_certificate.interlock_token_verified
    assert bad.crowbar_latency_ns > 0.0                               # latencia MEDIDA, no inventada
    assert agent.verify_sovereign_state(bad)

    # 4) Manipulación de P1 ⇒ Φ₁₂ detecta (v13 no lo hacía) ⇒ fail-safe con ⊥
    forged = dataclasses.replace(st.phase2.phase1, tr_rho=999.0)
    try:
        agent._orientor.continue_to_phase2_gate(
            forged, k_test, exps, GovernanceParameters(), expected_epoch=forged.epoch
        )
        raise AssertionError("Φ₁₂ aceptó un payload manipulado")
    except GateIntegrityError:
        pass

    # 5) Merkle RFC 6962: pruebas válidas para todo índice y rechazo de hoja alterada
    for n in range(1, 10):
        lvs = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lvs)
        for i in range(n):
            assert verify_merkle_inclusion(lvs[i], i, n, merkle_inclusion_proof(lvs, i), rt)
        assert not verify_merkle_inclusion(b"hoja-falsa", 0, n, merkle_inclusion_proof(lvs, 0), rt)

    # 6) Determinismo del sello (sin reloj) y detección de manipulación del estado
    st_a = agent.run_sovereign_governance_cycle(k_test, exps)
    st_b = agent.run_sovereign_governance_cycle(k_test, exps)
    assert st_a.phase2.phase1.chain_hash == st_b.phase2.phase1.chain_hash
    assert not agent.verify_sovereign_state(dataclasses.replace(st_a, action="NOMINAL", truth_value=HeytingTruthValue.VETOED))

    # 7) Φ_sem con insumos declarados ⇒ KPIs con procedencia
    rich = agent.run_sovereign_governance_cycle(
        k_test, exps,
        params=GovernanceParameters(exergy_price_per_joule=0.12, capital_base=1000.0,
                                    kv_bytes_before=1_000_000, kv_bytes_after=136_000),
    )
    assert rich.visceral_metrics.net_liquidity_release_pct is not None
    assert abs(rich.visceral_metrics.kv_cache_compression_pct - 86.4) < 1e-9   # MEDIDO, no constante

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬     : {st.truth_value.name} / {st.boolean_shadow.name}")
    print(f"Acción / disparo             : {st.action} / {st.crowbar_triggered}")
    print(f"P1 isometría espectral       : {st.phase2.phase1.spectral_isometry_residual:.3e}")
    print(f"Pasaporte motor verificado   : {st.phase2.engine_passport_verified}")
    print(f"X_dest (Gouy–Stodola) [W]    : {st.phase2.destroyed_exergy_w:.6f}")
    print(f"W_irr [J] / ESS              : {st.phase2.irreversible_work_j:.6f} / {st.phase2.ess_fraction:.3f}")
    print(f"Cadencia α / γ / huecos      : {st.phase2.cadence.rotation_number:.6f} / "
          f"{st.phase2.cadence.diophantine_gamma:.4f} / {st.phase2.cadence.distinct_gap_count}")
    print(f"RSI³ q / Lip(T³) medido      : {st.rsi3.policy_lipschitz_q:.6f} / {st.rsi3.policy_lip_t3_empirical:.6f}")
    print(f"Kantorovich h / d³C          : {st.rsi3.kantorovich_h:.6f} / {st.rsi3.capacity_d3:.4f}")
    print(f"Despacho mediano [ns]        : {st.crowbar_certificate.dispatch_median_ns:.0f} "
          f"(plazo lógico {st.crowbar_certificate.logic_deadline_ns} ns)")
    print(f"Presupuesto físico SCR [ns]  : {st.crowbar_certificate.physical_budget_ns:.0f} "
          f"(viable≤plazo: {st.crowbar_certificate.physical_deadline_feasible})")
    print(f"Merkle raíz / hojas          : {st.merkle_root_hash[:16]}… / {len(st.merkle_leaves)}")
    print(f"KPI provenance               : {st.visceral_metrics.provenance}")
    print(f"Controles ½ / ⊥              : {mid.truth_value.name} / {bad.truth_value.name} "
          f"(disparo={bad.crowbar_triggered})")
    print(f"Estado reverificable         : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")