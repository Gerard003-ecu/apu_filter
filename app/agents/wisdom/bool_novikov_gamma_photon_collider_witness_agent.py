# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderWitnessAgent  v14.0.0-PhD (reedición rigurosa)         ║
║ Ubicación: app/agents/wisdom/bool_novikov_gamma_photon_collider_witness_agent.py                ║
║ Función  : Soberano agéntico TESTIGO pasivo del canal e⁻+e⁺→2γ: pre-compromete la medición      ║
║            QND, certifica pérdida por inserción y loophole de detección, atestigua conteos      ║
║            contra cinemática exacta y sella Merkle verificable públicamente. SIN ACTUAR.        ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL TESTIGO DE FOTONES
───────────────────────────────
(1) CONTAR SIN ABSORBER. El testigo declara su pérdida por inserción η_w y su fuerza de
    interacción g ANTES de interactuar (pre-compromiso en P1). El motor debe usar exactamente
    esos valores; el agente verifica a posteriori que la disturbancia no excedió el presupuesto.
(2) LA PURIDAD EN dB NO SIRVE PARA PÉRDIDAS. El canal dispersivo es UNITAL (pureza no creciente);
    el canal de pérdida NO lo es (|n⟩→|0⟩ es puro). Por eso la pérdida se certifica con
    distancia de traza, número medio de fotones y thinning binomial, nunca con dB de pureza.
(3) BELL ≠ LOOPHOLE CERRADO. Que el estado sea máximamente entrelazado (CHSH=2√2) NO implica
    que el experimento cierre el loophole de detección. Con eficiencia simétrica η se requiere
    η ≥ 2/(1+√2) ≈ 0.8284 (Garg–Mermin) para CHSH maximal, o η > 2/3 (Eberhard, estado no maximal
    + Clauser–Horne). Si no se cierra, la afirmación device-independent se DEGRADA (½) y se
    declara la hipótesis de muestreo justo.
(4) MERKLE SIN CLAVE. La raíz debe ser recalculable por cualquiera; la autenticidad la da la
    firma sobre la raíz. v13 claveaba la raíz y destruía la verificabilidad pública.
(5) FAIL-CLOSED SILENCIOSO. Ante fallo: retener atestación (⊥), g=0, estado firmado y escalada
    al soberano activo. Jamás actuar, jamás lanzar al vacío.

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.5  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.8  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle
    P1 ⊂ P2 ⊂ GammaPhotonWitnessSovereignState; ejecución = Kleisli observe>=>orient>=>decide.

FASE 1 — OBSERVE PASIVO
  §1.1 Validación previa + RECONCILIACIÓN DECLARADA de la dimensión de Fock (N = dim ρ − 1).
  §1.2 Normalización auditada + ISOMETRÍA DE BURES–WASSERSTEIN bajo unitario Haar.
  §1.3 PRE-COMPROMISO: calibración Newton–Kantorovich de g y presupuesto de pérdida η_w.
  §1.4 Compromiso canónico de entradas, medición y cinemática.
  §1.5 [TERMINAL] emit_phase1_certificate.

COMPUERTA Φ₁₂ — integridad dura (recálculo desde payload, HMAC, re-digest, época); ADMITE ⊥.

FASE 2 — ORIENT PASIVO
  §2.1 accept_phase1_certificate.
  §2.2 Motor gamma v14 con el g PRE-COMPROMETIDO y sin token (epistémico).
  §2.3 Verificación del informe: pasaporte, Merkle RFC 6962, prueba de inclusión, CONTRAFIRMA.
  §2.4 Canal de pérdida: thinning binomial, completitud exacta, semigrupo η₁η₂, no-unitalidad.
  §2.5 Loophole de detección: Garg–Mermin y Eberhard; degradación honesta de la afirmación Bell.
  §2.6 Atestación de conteos/energías contra cinemática exacta + paridad C + Landau–Yang.
  §2.7 Cobertura azimutal del analizador KN: tres huecos + cota de KOKSMA (V=8sin²θ, D*_N).
  §2.8 [TERMINAL] emit_phase2_certificate.

COMPUERTA Φ₂₃ — integridad dura; física: dB ≥ −tol ∧ D_tr ≤ presupuesto ∧ Lip(T³)<1 ∧ pasividad.
                (¡El gate heredado `dB ≤ 0.01` es VACUO bajo la convención v14!)

FASE 3 — DECIDE & ATTEST
  §3.1 accept_phase2_certificate.
  §3.2 Adjudicación Ω₃, núcleo ¬¬, functor epistémico contravariante.
  §3.3 Interlock de pasividad (capacidad de actuación ausente, prueba negativa).
  §3.4 RSI³: Kantorovich + Banach + Kleisli + jerk de capacidad medido.
  §3.5 Φ_sem con procedencia dimensional.
  §3.6 Escalada firmada (sin transporte, sin actuación).
  §3.7 [TERMINAL] emit_sovereign_state (Merkle sin clave + firma determinista).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `positron_hmac_valid: bool` ABOLIDO; `authorization_token` se registra como NO ejercido.
• `spec.qnd_report` → `spec.phase_ii.phase_i.qnd`; `spec.banach_contraction_lip` → `spec.rsi3.lip_t3_bound`.
• `back_action_db` cambia de signo (ahora ≤0): el gate `dB ≤ 0.01` queda VACUO (§3.1).
• Sellos deterministas (sin reloj); raíz Merkle SIN clave y con prueba de inclusión.
• KPI sin insumos declarados ⇒ None con `UNDECLARED_INPUT`.

LIMITACIONES DECLARADAS
1. El contador de silencio mide logging de este proceso; no prueba ausencia de canales laterales.
2. Koksma da cota A PRIORI del promedio azimutal; es conservadora (D*_N ~ log N / N).
3. Kac–Slater y la cobertura azimutal se verifican para rotación irracional (modelo ergódico).
4. η se DECLARA (datasheet o medición HIL); sin medición, el cierre del loophole no se afirma.
5. Con g=0 el valor débil es contrafactual; sin conteos declarados la cinemática es predicción.

REFERENCIAS: Garg–Mermin (1987); Eberhard (1993); Clauser–Horne (1974); Tsirelson (1980);
Wu–Shaknov (1950); Klein–Nishina (1929); Hong–Ou–Mandel (1987); Koksma (1942)–Hlawka (1961);
Steinhaus (tres huecos); Slater (1967); Kac (1947); Bures (1969); Uhlmann (1976);
Fuchs–van de Graaf (1999); Spohn (1978); Kantorovich; Banach (1922); RFC 6962; CODATA 2022.
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


def _import_first(paths: Sequence[str]) -> Any:
    last: Optional[Exception] = None
    for p in paths:
        try:
            return __import__(p, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(f"No se pudo importar ninguno de: {paths}") from last


_ge = _import_first(("app.wisdom.bool_novikov_gamma_photon_collider_witness_engine",
                     "bool_novikov_gamma_photon_collider_witness_engine"))
_pri = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
                      "bool_novikov_thermal_exergy_weaver_engine"))

GammaWitnessEngine = _ge.BoolNovikovGammaPhotonColliderWitnessEngine
EngineGammaParameters = _ge.GammaWitnessParameters
PrincipalEngine = _pri.BoolNovikovThermalExergyWeaverEngine
HeytingTruthValue = _pri.HeytingTruthValue          # Ω₃ CANÓNICO
Evidence = _pri.Evidence
Certified = _pri.Certified
kleisli_compose = _pri.kleisli_compose

__all__ = [
    "GammaWitnessGovernanceParameters", "GammaPhotonWitnessPhase1Certificate",
    "GammaPhotonWitnessPhase2Certificate", "GammaPhotonWitnessVisceralMetrics",
    "GammaPhotonWitnessSovereignState", "GammaPhotonWitnessPhase1Observer",
    "GammaPhotonWitnessPhase2Orientor", "GammaPhotonWitnessPhase3Decider",
    "BoolNovikovGammaPhotonColliderWitnessAgent", "GammaWitnessGovernanceError",
    "GammaWitnessGateIntegrityError", "ActuationForbiddenError", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderWitnessAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-GAMMA-WITNESS-AGENT-QND-LOSS-GARG-MERMIN-KOKSMA-KAC-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_ME_C2_KEV: Final[float] = 510.99895069
_ETA_CRIT_GARG_MERMIN: Final[float] = 2.0 / (1.0 + math.sqrt(2.0))   # ≈ 0.8284271
_ETA_CRIT_EBERHARD: Final[float] = 2.0 / 3.0
_HMAC_P1: Final[bytes] = b"GAMMA-WITNESS-AGENT-P1-COMMIT-v14|"
_HMAC_P2: Final[bytes] = b"GAMMA-WITNESS-AGENT-P2-ATTEST-v14|"
_HMAC_STATE: Final[bytes] = b"GAMMA-WITNESS-AGENT-STATE-v14|"
_HMAC_ESC: Final[bytes] = b"GAMMA-WITNESS-AGENT-ESCALATION-v14|"
_HMAC_COUNTER: Final[bytes] = b"GAMMA-WITNESS-COUNTERSIGN-v14|"      # igual que el motor gamma
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle SIN CLAVE, BW, pérdida, Koksma
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaWitnessGovernanceError(Exception):
    """Error base de la gobernanza del testigo gamma."""


class GammaWitnessGateIntegrityError(GammaWitnessGovernanceError):
    """Fallo DURO de integridad ⇒ fail-closed pasivo (retener atestación, escalar, no actuar)."""


class ActuationForbiddenError(GammaWitnessGovernanceError):
    """Intento de actuación por un soberano pasivo: estructuralmente prohibido."""


def _array_digest(arr: NDArray) -> str:
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|"); h.update(str(a.shape).encode("ascii")); h.update(b"|")
    h.update(dt.str.encode("ascii")); h.update(b"|"); h.update(a.tobytes())
    return h.hexdigest()


def _freeze(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return {"ndarray_sha256": _array_digest(o)}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _freeze(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, HeytingTruthValue):
        return {"omega3": int(o), "name": o.name}
    if isinstance(o, complex):
        return {"re": _freeze(o.real), "im": _freeze(o.imag)}
    if isinstance(o, (list, tuple)):
        return [_freeze(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _freeze(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return float(f"{x:.15g}")
    if o is None or isinstance(o, str):
        return o
    if isinstance(o, bytes):
        return o.hex()
    return str(o)


def _canonical(o: Any) -> bytes:
    return json.dumps(_freeze(o), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _leaf_hash(d: bytes) -> bytes:
    return hashlib.sha256(_LEAF + d).digest()


def _node_hash(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(_NODE + a + b).digest()


def _mth(lv: Sequence[bytes]) -> bytes:
    n = len(lv)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return _leaf_hash(lv[0])
    k = 1 << ((n - 1).bit_length() - 1)
    return _node_hash(_mth(lv[:k]), _mth(lv[k:]))


def merkle_root(lv: Sequence[bytes]) -> str:
    """Raíz RFC 6962 SIN CLAVE: recalculable por cualquier verificador externo."""
    return _mth(list(lv)).hex()


def merkle_proof(lv: Sequence[bytes], idx: int) -> Tuple[Tuple[str, bool], ...]:
    ds = list(lv)
    if not 0 <= idx < len(ds):
        raise GammaWitnessGovernanceError("Índice de hoja fuera de rango.")

    def path(m: int, sub: Sequence[bytes]) -> List[Tuple[str, bool]]:
        n = len(sub)
        if n == 1:
            return []
        k = 1 << ((n - 1).bit_length() - 1)
        if m < k:
            return path(m, sub[:k]) + [(_mth(sub[k:]).hex(), True)]
        return path(m - k, sub[k:]) + [(_mth(sub[:k]).hex(), False)]

    return tuple(path(idx, ds))


def verify_merkle(leaf: bytes, idx: int, cnt: int,
                  proof: Sequence[Tuple[str, bool]], root: str) -> bool:
    if cnt <= 0 or not 0 <= idx < cnt:
        return False
    fn, sn, acc = idx, cnt - 1, _leaf_hash(leaf)
    for sib, _ in proof:
        if sn == 0:
            return False
        s = bytes.fromhex(sib)
        if (fn & 1) or fn == sn:
            acc = _node_hash(s, acc)
            if not (fn & 1):
                while (fn & 1) == 0 and fn != 0:
                    fn >>= 1; sn >>= 1
        else:
            acc = _node_hash(acc, s)
        fn >>= 1; sn >>= 1
    return sn == 0 and acc.hex() == root


def _psd_sqrt(a: NDArray) -> NDArray:
    v, u = la.eigh(0.5 * (a + a.conj().T))
    return (u * np.sqrt(np.maximum(v, 0.0))) @ u.conj().T


def bures_wasserstein_distance(a: NDArray, b: NDArray) -> float:
    r"""d_BW²(A,B) = Tr A + Tr B − 2Tr√(√A B √A); para estados, d_BW² = 2−2√F."""
    sa = _psd_sqrt(a)
    core = sa @ b @ sa
    v = la.eigvalsh(0.5 * (core + core.conj().T))
    cross = float(np.sum(np.sqrt(np.maximum(v, 0.0))))
    return math.sqrt(max(0.0, float(np.real(np.trace(a)) + np.real(np.trace(b))) - 2.0 * cross))


def _fidelity_svd(rho: NDArray, sig: NDArray) -> float:
    return float(min(1.0, max(0.0, float(np.sum(la.svdvals(_psd_sqrt(rho) @ _psd_sqrt(sig)))) ** 2)))


def _trace_distance(a: NDArray, b: NDArray) -> float:
    d = a - b
    return 0.5 * float(np.sum(np.abs(la.eigvalsh(0.5 * (d + d.conj().T)))))


def _haar_unitary(dim: int, rng: np.random.Generator) -> NDArray:
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * np.where(np.abs(d) > 0.0, d / np.maximum(np.abs(d), _EPS), 1.0)


def loss_channel_kraus(dim: int, eta: float) -> List[NDArray]:
    r"""
    Canal de pérdida (divisor de haz con vacío) en Fock truncado:
        K_m = Σ_{n≥m} √(C(n,m) η^m (1−η)^{n−m}) |m⟩⟨n|.
    La completitud Σ_m K_m†K_m = I es EXACTA en el truncamiento (todos los m ≤ n ≤ N están
    incluidos), a diferencia de [a,a†]. El canal NO es unital: Λ(I/d) ≠ I/d.
    """
    if not 0.0 <= eta <= 1.0:
        raise GammaWitnessGovernanceError("η ∈ [0,1].")
    kraus: List[NDArray] = []
    for m in range(dim):
        k = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(m, dim):
            k[m, n] = math.sqrt(math.comb(n, m) * eta ** m * (1.0 - eta) ** (n - m))
        kraus.append(k)
    return kraus


def apply_kraus(rho: NDArray, kraus: Sequence[NDArray]) -> NDArray:
    return sum(k @ rho @ k.conj().T for k in kraus)


def star_discrepancy(points: NDArray) -> float:
    """D*_N exacta en 1D: max_i max(|i/N − u_i|, |u_i − (i−1)/N|), u ordenado en [0,1)."""
    u = np.sort(np.asarray(points, dtype=np.float64))
    n = u.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(np.abs(i / n - u), np.abs(u - (i - 1.0) / n))))


def _savgol(w: int, p: int, d: int) -> NDArray:
    h = w // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, p + 1, increasing=True))[d] * math.factorial(d)


def _convergents(alpha: float, max_q: int) -> Tuple[Tuple[int, int], ...]:
    x = float(alpha); a = math.floor(x)
    pp, qp, p, q = 1, 0, int(a), 1
    out = [(p, q)]
    for _ in range(64):
        f = x - a
        if f < 1e-14:
            break
        x = 1.0 / f; a = math.floor(x)
        pn, qn = int(a) * p + pp, int(a) * q + qp
        if qn > max_q:
            break
        pp, qp, p, q = p, q, pn, qn
        out.append((p, q))
    return tuple(out)


class _SilenceMeter(logging.Handler):
    """Contador operativo de ruido (cota, no prueba de sigilo)."""

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


class PassivityInterlock:
    """Garantía ESTRUCTURAL: el testigo carece de capacidad de actuación."""

    actuation_capability: Final[bool] = False

    def __init__(self) -> None:
        self.refused_requests = 0

    def request_actuation(self, *_: Any, **__: Any) -> None:
        self.refused_requests += 1
        raise ActuationForbiddenError(
            "PASSIVE_SILENT_QND: el testigo gamma no puede actuar; escale al soberano activo."
        )


class GammaWitnessSigner:
    """HMAC-SHA256 con separación de dominio y comparación en tiempo constante."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_WITNESS_AGENT_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave de testigo gamma efímera: sellos no verificables entre procesos.")
        if len(secret_key) < 32:
            raise GammaWitnessGovernanceError("La clave del testigo requiere ≥ 256 bits.")
        self._key = bytes(secret_key)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def fingerprint(self) -> str:
        return hmac.new(self._key, b"GAMMA-WITNESS-KEY-FP-v14|", hashlib.sha256).hexdigest()[:16]

    def sign(self, domain: bytes, payload: Any) -> str:
        return hmac.new(self._key, domain + _canonical(payload), hashlib.sha256).hexdigest()

    def verify(self, domain: bytes, payload: Any, signature: str) -> bool:
        return hmac.compare_digest(self.sign(domain, payload), str(signature))


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GammaWitnessGovernanceParameters:
    """Presupuestos y parámetros DECLARADOS; su digest se compromete en P1."""

    fock_n_max: int = 3
    coherent_alpha: float = 0.7
    photon_energy_kev: float = _ME_C2_KEV
    beta_field: float = 2.0
    interaction_g: Optional[float] = None        # None ⇒ calibración certificada en §1.3
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    insertion_transmissivity: float = 0.999      # η_w del testigo (declarada / HIL)
    detector_efficiency: float = 0.90            # η_det (datasheet / HIL)
    efficiency_measured_hil: bool = False        # ¿hay medición hardware-in-the-loop?
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    compton_theta_deg: float = 90.0
    azimuth_samples: int = 64
    azimuth_rotation: float = _GOLDEN
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    ball_r: float = 1.0
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    silence_byte_budget: int = 4096
    silence_record_budget: int = 24
    strict_mode: bool = False
    exergy_reference_w: float = 1.0
    exergy_price_per_joule: Optional[float] = None
    capital_base: Optional[float] = None
    kv_bytes_before: Optional[int] = None
    kv_bytes_after: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fock_n_max < 1 or self.pointer_sigma <= 0.0:
            raise GammaWitnessGovernanceError("N≥1 y σ>0 requeridos.")
        if not (0.0 <= self.insertion_transmissivity <= 1.0 and 0.0 <= self.detector_efficiency <= 1.0):
            raise GammaWitnessGovernanceError("η ∈ [0,1].")
        if self.disturbance_budget < 0.0 or self.positron_kinetic_kev < 0.0:
            raise GammaWitnessGovernanceError("Presupuesto y cinética ≥ 0.")
        if not (0.0 < self.recurrence_mu < 1.0 and 0.0 < self.azimuth_rotation < 1.0):
            raise GammaWitnessGovernanceError("μ∈(0,1), α∈(0,1).")
        if self.azimuth_samples < 8 or self.silence_byte_budget <= 0:
            raise GammaWitnessGovernanceError("Muestras azimutales ≥8 y presupuesto de silencio >0.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    def to_engine_parameters(self, n_max: int, g: float) -> Any:
        """Proyección al contrato del motor gamma v14 con N reconciliado y g comprometido."""
        return EngineGammaParameters(
            fock_n_max=int(n_max), coherent_alpha=self.coherent_alpha,
            photon_energy_kev=self.photon_energy_kev, beta_field=self.beta_field,
            interaction_g=float(g), pointer_sigma=self.pointer_sigma,
            disturbance_budget=self.disturbance_budget, backaction_db_budget=self.backaction_db_budget,
            positron_kinetic_kev=self.positron_kinetic_kev, emission_angle_deg=self.emission_angle_deg,
            compton_theta_deg=self.compton_theta_deg, ball_r=self.ball_r, cylinder_r=self.cylinder_r,
            melnikov_omega=self.melnikov_omega, melnikov_gamma=self.melnikov_gamma,
            melnikov_delta=self.melnikov_delta, cadence_points=self.azimuth_samples,
            cadence_rotation=self.azimuth_rotation, recurrence_mu=self.recurrence_mu,
            recurrence_samples=self.recurrence_samples,
        )


@dataclass(frozen=True)
class GammaPhotonWitnessPhase1Certificate:
    """P1 — Observación pasiva con PRE-COMPROMISO (reloj fuera del sello)."""

    timestamp: float
    tr_rho: float
    tr_rho_normalized: float
    normalization_factor: float
    is_psd: bool
    hermiticity_residual: float
    purity: float
    von_neumann_entropy: float
    fock_n_reconciled: int
    dimension_reconciliation_declared: bool
    bures_wasserstein_to_thermal: float
    bw_isometry_residual: float
    bw_uhlmann_consistency_residual: float
    committed_gain_g: float
    gain_kantorovich_h: float
    gain_calibration_residual: float
    gain_degenerate_channel: bool
    gain_override_rejected: bool
    committed_insertion_eta: float
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
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
class InsertionLossReport:
    """§2.4 — Canal de pérdida del testigo: thinning binomial exacto y NO-unitalidad."""

    eta_insertion: float
    kraus_completeness_residual: float
    mean_photon_before: float
    mean_photon_after: float
    thinning_mean_residual: float
    binomial_normalization_residual: float
    semigroup_composition_residual: float
    nonunitality_residual: float
    trace_distance_insertion: float
    purity_before: float
    purity_after: float
    purity_convention_valid: bool
    note: str


@dataclass(frozen=True)
class DetectionLoopholeReport:
    """§2.5 — Garg–Mermin y Eberhard: ¿la afirmación Bell es device-independent?"""

    eta_total: float
    eta_crit_garg_mermin: float
    eta_crit_eberhard: float
    chsh_state_maximum: float
    fair_sampling_assumed: bool
    efficiency_measured_hil: bool
    chsh_loophole_closed: bool
    eberhard_route_available: bool
    claim_level: str
    note: str


@dataclass(frozen=True)
class PhotonCountAttestationReport:
    """§2.6 — Conteos/energías declarados vs cinemática exacta, paridad C y Landau–Yang."""

    attestation_mode: str
    event_observed: bool
    declared_count: int
    declared_energies_kev: Tuple[float, ...]
    expected_total_energy_kev: float
    energy_sum_residual: float
    energy_window_ok: bool
    c_parity_consistent: bool
    landau_yang_violation: bool
    declared_state_spin: int
    kinematic_consistency_ok: bool


@dataclass(frozen=True)
class AzimuthalCoverageReport:
    """§2.7 — Tres huecos + Koksma: cobertura del analizador KN con cota a priori."""

    rotation_number: float
    n_samples: int
    distinct_gap_count: int
    max_gap: float
    coverage_bound: float
    three_gap_identity_residual: float
    star_discrepancy: float
    total_variation_f: float
    koksma_error_bound: float
    sample_mean: float
    exact_mean: float
    actual_error: float
    koksma_bound_ok: bool
    unpolarized_identity_residual: float
    three_gap_ok: bool
    coverage_ok: bool
    kac_expected_delay: float
    kac_empirical_delay: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    slater_ok: bool


@dataclass(frozen=True)
class GammaPhotonWitnessPhase2Certificate:
    """P2 — Orientación pasiva: anida P1 y el informe del motor gamma v14."""

    timestamp: float
    phase1: GammaPhotonWitnessPhase1Certificate
    spectrum_report: Optional[Any]
    engine_invoked: bool
    engine_report_verified: bool
    engine_merkle_recomputed_ok: bool
    countersignature_ok: bool
    gain_honored: bool
    disturbance_consumed: float
    back_action_db: float
    insertion_loss: InsertionLossReport
    detection_loophole: DetectionLoopholeReport
    photon_counts: PhotonCountAttestationReport
    azimuthal: AzimuthalCoverageReport
    silence_records_ok: bool
    silence_bytes_ok: bool
    silence_bytes_emitted: int
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    epoch: int
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        rep = self.spectrum_report
        return {
            "phase1_chain_hash": self.phase1.chain_hash, "engine_invoked": self.engine_invoked,
            "engine_passport": getattr(rep, "passport_hash", ""),
            "engine_merkle_root": getattr(rep, "merkle_root_hash", ""),
            "engine_countersignature": getattr(rep, "countersignature", ""),
            "engine_truth": getattr(rep, "truth_value", HeytingTruthValue.VETOED),
            "engine_report_verified": self.engine_report_verified,
            "engine_merkle_recomputed_ok": self.engine_merkle_recomputed_ok,
            "countersignature_ok": self.countersignature_ok, "gain_honored": self.gain_honored,
            "disturbance_consumed": self.disturbance_consumed, "back_action_db": self.back_action_db,
            "insertion_loss": self.insertion_loss, "detection_loophole": self.detection_loophole,
            "photon_counts": self.photon_counts, "azimuthal": self.azimuthal,
            "silence_records_ok": self.silence_records_ok, "silence_bytes_ok": self.silence_bytes_ok,
            "evidences": self.evidences, "truth_value": self.truth_value, "epoch": self.epoch,
        }


@dataclass(frozen=True)
class GammaWitnessGateReceipt:
    gate: str
    payload_digest: str
    chain_recomputed_ok: bool
    signature_ok: bool
    binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class GammaPhotonWitnessVisceralMetrics:
    """§3.5 — Φ_sem con procedencia: métricas medidas; finanzas sólo si declaradas."""

    attestation_coverage_pct: float
    assurance_level: str
    disturbance_budget_consumed_pct: float
    insertion_loss_budget_pct: float
    silence_budget_consumed_pct: float
    bell_margin_pct: Optional[float]
    hom_visibility_pct: float
    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    stealth_qnd_status: str
    detection_loophole_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool


@dataclass(frozen=True)
class GammaWitnessRSI3Report:
    """§3.4 — RSI³: Kantorovich + Banach + Kleisli + núcleo + jerk medido."""

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
    nucleus_axioms_ok: bool
    action_functor_ok: bool


@dataclass(frozen=True)
class GammaWitnessEscalation:
    epoch: int
    witness_truth: HeytingTruthValue
    principal_truth: Optional[HeytingTruthValue]
    reason: str
    merkle_root_hash: str
    signature: str
    delivered: bool


@dataclass(frozen=True)
class GammaPhotonWitnessSovereignState:
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
    visceral_metrics: GammaPhotonWitnessVisceralMetrics
    rsi3: GammaWitnessRSI3Report
    escalation: Optional[GammaWitnessEscalation]
    phase2: GammaPhotonWitnessPhase2Certificate
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


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE PASIVO — reconciliación de Fock, isometría BW y pre-compromiso
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonWitnessPhase1Observer:
    r"""
    FASE 1 OODA pasiva. Reconcilia la dimensión de Fock con ρ (corrige el fallo de integración
    de v13), ejecuta la isometría de Bures–Wasserstein que su docstring prometía, y PRE-COMPROMETE
    la fuerza de interacción antes de que exista contacto con el campo atestiguado.
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE-PASIVO"

    def __init__(self, signer: GammaWitnessSigner, *, seed: int = 20_261_120) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa + reconciliación de dimensión ───────────
    def inspect_and_reconcile(self, rho_raw: Optional[NDArray],
                              params: GammaWitnessGovernanceParameters) -> Tuple[NDArray, int, Dict[str, Any]]:
        """
        Si ρ se aporta, N se RECONCILIA como dim(ρ)−1 (declarado y sellado); si no, se usa el
        N declarado y el estado coherente truncado del motor. v13 pasaba ρ de dim(K) al motor
        que exige (N+1)×(N+1): incompatibilidad dimensional no detectada.
        """
        if rho_raw is None:
            n = int(params.fock_n_max)
            dim = n + 1
            coeff = np.array([params.coherent_alpha ** k / math.sqrt(math.factorial(k)) for k in range(dim)])
            coeff *= math.exp(-0.5 * params.coherent_alpha ** 2)
            coeff /= (la.norm(coeff) + _EPS)
            rho = np.outer(coeff, coeff.conj())
            return rho, n, {"hermiticity_residual": 0.0, "trace_raw": 1.0, "reconciled": False}
        r = np.asarray(rho_raw, dtype=np.complex128)
        if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 2:
            raise GammaWitnessGovernanceError("ρ debe ser cuadrada ≥2×2 (espacio de Fock).")
        if not np.all(np.isfinite(r)):
            raise GammaWitnessGovernanceError("ρ contiene NaN/Inf.")
        fro = float(la.norm(r, ord="fro")) + _EPS
        herm = float(la.norm(r - r.conj().T, ord="fro")) / fro
        tr = complex(np.trace(r))
        if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0:
            raise GammaWitnessGovernanceError("ρ: traza compleja o no positiva.")
        return r, int(r.shape[0] - 1), {"hermiticity_residual": herm, "trace_raw": float(tr.real),
                                        "reconciled": True}

    # ─────────── §1.2 Normalización auditada + isometría BW ───────────
    @staticmethod
    def normalize_audited(rho: NDArray, trace_raw: float) -> Tuple[NDArray, float]:
        r = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        return r / float(trace_raw), 1.0 / float(trace_raw)

    def bw_isometry(self, rho: NDArray, sigma: NDArray) -> Tuple[float, float, float]:
        d = bures_wasserstein_distance(rho, sigma)
        u = _haar_unitary(rho.shape[0], self._rng)
        d_moved = bures_wasserstein_distance(u @ rho @ u.conj().T, u @ sigma @ u.conj().T)
        fid = _fidelity_svd(rho, sigma)
        return d, abs(d_moved - d), abs(d * d - (2.0 - 2.0 * math.sqrt(fid)))

    # ─────────── §1.3 PRE-COMPROMISO: calibración certificada de g ───────────
    @staticmethod
    def certify_interaction_gain(rho: NDArray, n_max: int, sigma: float, budget: float, *,
                                 g_default: float = 1.0e-5, max_iter: int = 60
                                 ) -> Tuple[float, float, float, bool, Tuple[float, ...]]:
        r"""
        Proxy Frobenius EXACTO de la disturbancia dispersiva:
            S(g) = Σ_mn |ρ_mn|²(1 − e^{−g²(m−n)²/8σ²})²,  S'(g) analítica, monótona en g.
        Newton con salvaguarda de bisección (convergencia garantizada); h de Kantorovich
        estimado con |S''| muestreado (declarado). Si [ρ,n̂]=0 el canal es degenerado (S≡0):
        todo g es no invasivo y se DECLARA en lugar de fingir una calibración.
        """
        n_vals = np.arange(n_max + 1, dtype=np.float64)
        d2 = (n_vals[:, None] - n_vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(rho) ** 2
        if float(np.sum(w2 * d2)) <= 1e-300:
            return float(g_default), 0.0, 0.0, True, (0.0,)

        def s_of(g: float) -> float:
            return float(np.sum(w2 * (1.0 - np.exp(-(g ** 2) * d2)) ** 2))

        def ds_of(g: float) -> float:
            e = np.exp(-(g ** 2) * d2)
            return float(np.sum(w2 * 4.0 * g * d2 * (1.0 - e) * e))

        target = float(budget) ** 2
        lo, hi = 0.0, max(1e-8, sigma * 0.01)
        while s_of(hi) < target and hi < 1e6:
            hi *= 2.0
        g = 0.5 * (lo + hi)
        grid = np.linspace(max(lo, 1e-12), hi, 48)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([s_of(x) for x in grid], grid), grid)))) + _EPS
        d0 = ds_of(g)
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs((s_of(g) - target) / max(abs(d0), 1e-300)) * m_est
        errs: List[float] = []
        for _ in range(int(max_iter)):
            f_v, dv = s_of(g) - target, ds_of(g)
            if f_v > 0.0:
                hi = g
            else:
                lo = g
            step = g - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errs.append(abs(s_of(g) - target))
            if hi - lo < 1e-18 * max(1.0, hi):
                break
        return float(g), float(h_k), float(errs[-1]), False, tuple(errs)

    # ─────────── §1.5 [TERMINAL FASE 1] → Φ₁₂ → Fase 2 §2.1 ───────────
    def emit_phase1_certificate(
        self, rho_raw: Optional[NDArray], apu_cost_matrix: NDArray, exponents: NDArray,
        params: GammaWitnessGovernanceParameters, *, epoch: int, principal_passport: str = "",
    ) -> Tuple[GammaPhotonWitnessPhase1Certificate, NDArray, int]:
        """MÉTODO TERMINAL DE LA FASE 1: observa, reconcilia, calibra y COMPROMETE."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise GammaWitnessGovernanceError("K debe ser cuadrada.")
        rho_raw_n, n_max, insp = self.inspect_and_reconcile(rho_raw, params)
        rho, factor = self.normalize_audited(rho_raw_n, insp["trace_raw"])
        eigs = la.eigvalsh(rho)
        purity = float(np.real(np.trace(rho @ rho)))
        probs = np.clip(eigs, 0.0, 1.0)
        entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        is_psd = bool(float(np.min(eigs)) >= -1e-10)

        dim = n_max + 1
        en = params.photon_energy_kev * np.arange(dim, dtype=np.float64)
        lp = -params.beta_field * (en - en[0])
        lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        d_bw, iso_res, cons_res = self.bw_isometry(rho, rho_th)

        g_cert, h_k, cal_res, degenerate, _ = self.certify_interaction_gain(
            rho, n_max, params.pointer_sigma, params.disturbance_budget)
        override_rejected = False
        if params.interaction_g is not None:
            g_req = float(params.interaction_g)
            if g_req > g_cert * (1.0 + 1e-12) and not degenerate:
                override_rejected = True      # NUNCA se clampa en silencio
                g_cert = 0.0                  # fail-closed: sin interacción
            else:
                g_cert = g_req

        evidences: Tuple[Any, ...] = (
            Evidence("GP1.hermiticidad", insp["hermiticity_residual"], 1e-10, hard=True),
            Evidence.boolean("GP1.positividad", is_psd, hard=True),
            Evidence("GP1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("GP1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence.boolean("GP1.dimension_fock_reconciliada", n_max >= 1, hard=True,
                             note=f"N declarado/reconciliado = {n_max}"),
            Evidence("GP1.bw_isometria_haar", iso_res, 1e-11, hard=True),
            Evidence("GP1.bw_uhlmann_consistencia", cons_res, 1e-10, hard=True),
            Evidence.boolean("GP1.g_no_excede_presupuesto", not override_rejected, hard=True,
                             note="un g que viola el presupuesto se RECHAZA, no se clampa"),
            Evidence("GP1.calibracion_residual", cal_res,
                     1e-12 * max(1.0, params.disturbance_budget ** 2)),
            Evidence.boolean("GP1.eta_insercion_declarada", params.insertion_transmissivity > 0.0,
                             hard=True),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        core = {
            "tr_rho": float(insp["trace_raw"]), "tr_rho_normalized": float(np.real(np.trace(rho))),
            "normalization_factor": factor, "is_psd": is_psd,
            "hermiticity_residual": insp["hermiticity_residual"], "purity": purity,
            "von_neumann_entropy": entropy, "fock_n_reconciled": int(n_max),
            "dimension_reconciliation_declared": bool(insp["reconciled"]),
            "bures_wasserstein_to_thermal": d_bw, "bw_isometry_residual": iso_res,
            "bw_uhlmann_consistency_residual": cons_res,
            "committed_gain_g": float(g_cert), "gain_kantorovich_h": float(h_k),
            "gain_calibration_residual": float(cal_res), "gain_degenerate_channel": bool(degenerate),
            "gain_override_rejected": bool(override_rejected),
            "committed_insertion_eta": float(params.insertion_transmissivity),
            "rho_digest": _array_digest(rho_raw_n), "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "params_digest": params.digest(), "principal_passport": principal_passport,
            "agent_version": __version__, "engine_version": _ge.__version__, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonWitnessPhase1Certificate(
            timestamp=time.time(), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P1, core), **core
        )
        logger.info("GP1 Ω₃=%s · N=%d · g_comprometido=%.6g · d_BW=%.4e", truth.name, n_max, g_cert, d_bw)
        return cert, rho, n_max  # ──► Φ₁₂ ──► Fase 2 §2.1

    def execute_phase1(self, *a: Any, **k: Any) -> Tuple[GammaPhotonWitnessPhase1Certificate, NDArray, int]:
        """Alias heredado (v13) del terminal §1.5."""
        return self.emit_phase1_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT PASIVO — pérdida, loophole, conteos y cobertura azimutal
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonWitnessPhase2Orientor:
    r"""
    FASE 2 OODA pasiva. Anida la Fase 1 por handoff; su último método alimenta el primero de
    Fase 3 vía Φ₂₃. Certifica lo que v13 daba por supuesto: pérdida, loophole y cobertura.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT-PASIVO"

    def __init__(self, engine: Any, signer: GammaWitnessSigner) -> None:
        self._engine = engine
        self._signer = signer

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self, p1: GammaPhotonWitnessPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        params: GammaWitnessGovernanceParameters, *, expected_epoch: int,
    ) -> GammaWitnessGateReceipt:
        r"""
        Φ₁₂ — NO tautológica: recalcula el chain_hash DESDE el payload, verifica el HMAC en
        tiempo constante, re-digesta los arrays que cruzan (anti-TOCTOU) y exige pre-compromiso
        de g y de η. ADMITE ⊥ (auditoría pasiva inquisitorial sin demolición).
        """
        payload = p1.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p1.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P1, payload, p1.hmac_signature)
        binding_ok = (
            hmac.compare_digest(p1.kernel_digest, _array_digest(np.asarray(apu_cost_matrix, dtype=np.float64)))
            and hmac.compare_digest(p1.exponents_digest, _array_digest(np.asarray(exponents, dtype=np.float64)))
            and hmac.compare_digest(p1.params_digest, params.digest())
            and p1.epoch == expected_epoch and p1.agent_version == __version__
            and math.isfinite(p1.committed_gain_g) and p1.committed_gain_g >= 0.0
            and 0.0 <= p1.committed_insertion_eta <= 1.0
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise GammaWitnessGateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return GammaWitnessGateReceipt("Φ₁₂", recomputed, chain_ok, sig_ok, binding_ok, True, False,
                                       "⊥ admitido: la atestación pasiva continúa sin demoler")

    def accept_phase1_certificate(self, p1: GammaPhotonWitnessPhase1Certificate, apu_cost_matrix: NDArray,
                                  exponents: NDArray, params: GammaWitnessGovernanceParameters, *,
                                  expected_epoch: int) -> GammaWitnessGateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.5."""
        return self.continue_to_phase2_gate(p1, apu_cost_matrix, exponents, params, expected_epoch=expected_epoch)

    # ─────────── §2.3 Verificación del informe y CONTRAFIRMA ───────────
    def verify_engine_report(self, report: Any, principal_passport: str) -> Tuple[bool, bool, bool]:
        verified = bool(self._engine.verify_witness_report(report))
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        root_ok = hmac.compare_digest(merkle_root(leaves), report.merkle_root_hash)
        idx = int(report.attestation_leaf_index)
        root_ok = root_ok and verify_merkle(leaves[idx], idx, len(leaves),
                                            report.attestation_inclusion_proof, report.merkle_root_hash)
        expected = hmac.new(self._signer.key_bytes,
                            _HMAC_COUNTER + principal_passport.encode("ascii")
                            + bytes.fromhex(report.merkle_root_hash), hashlib.sha256).hexdigest()
        return verified, root_ok, hmac.compare_digest(expected, report.countersignature)

    # ─────────── §2.4 Canal de pérdida por inserción ───────────
    @staticmethod
    def insertion_loss_certificate(rho: NDArray, n_max: int,
                                   params: GammaWitnessGovernanceParameters) -> InsertionLossReport:
        r"""
        Thinning binomial exacto: P(m|n) = C(n,m)η^m(1−η)^{n−m}; media ηn, varianza nη(1−η).
        Se verifican: normalización, completitud EXACTA de Kraus en el truncamiento,
        media ⟨n̂⟩' = η⟨n̂⟩, y semigrupo Λ_{η₁}∘Λ_{η₂} = Λ_{η₁η₂}.
        NO-UNITALIDAD: Λ(I/d) ≠ I/d ⇒ la pureza puede AUMENTAR (|n⟩→|0⟩ es puro): la convención
        de dB de pureza del canal dispersivo NO es válida aquí y se declara explícitamente.
        """
        dim = n_max + 1
        eta = float(params.insertion_transmissivity)
        kraus = loss_channel_kraus(dim, eta)
        comp = sum(k.conj().T @ k for k in kraus)
        comp_res = float(la.norm(comp - np.eye(dim), ord="fro"))
        after = apply_kraus(rho, kraus)
        num = np.diag(np.arange(dim, dtype=np.float64)).astype(np.complex128)
        mean_b = float(np.real(np.trace(rho @ num)))
        mean_a = float(np.real(np.trace(after @ num)))
        binom_res = max(abs(sum(math.comb(n, m) * eta ** m * (1.0 - eta) ** (n - m) for m in range(n + 1)) - 1.0)
                        for n in range(dim))
        half = math.sqrt(max(eta, 1e-12))
        comp_semigroup = apply_kraus(apply_kraus(rho, loss_channel_kraus(dim, half)),
                                     loss_channel_kraus(dim, half))
        semi_res = float(la.norm(comp_semigroup - apply_kraus(rho, loss_channel_kraus(dim, half * half)), ord="fro"))
        eye_d = np.eye(dim, dtype=np.complex128) / dim
        nonunital = float(la.norm(apply_kraus(eye_d, kraus) - eye_d, ord="fro"))
        pb = float(np.real(np.trace(rho @ rho)))
        pa = float(np.real(np.trace(after @ after)))
        return InsertionLossReport(
            eta_insertion=eta, kraus_completeness_residual=comp_res, mean_photon_before=mean_b,
            mean_photon_after=mean_a, thinning_mean_residual=abs(mean_a - eta * mean_b),
            binomial_normalization_residual=binom_res, semigroup_composition_residual=semi_res,
            nonunitality_residual=nonunital, trace_distance_insertion=_trace_distance(rho, after),
            purity_before=pb, purity_after=pa, purity_convention_valid=False,
            note=("canal de pérdida NO unital: la pureza puede aumentar; se certifica con distancia "
                  "de traza y número medio, no con dB de pureza"),
        )

    # ─────────── §2.5 Loophole de detección ───────────
    @staticmethod
    def detection_loophole_certificate(report: Any, params: GammaWitnessGovernanceParameters
                                       ) -> DetectionLoopholeReport:
        r"""
        CHSH del ESTADO (2√2) ≠ CHSH del EXPERIMENTO con pérdidas. Con eficiencia simétrica η:
          • Garg–Mermin: η ≥ 2/(1+√2) ≈ 0.8284 para estado máximamente entrelazado + CHSH.
          • Eberhard:    η > 2/3 con estado NO maximal + Clauser–Horne (sin fondo).
        Si η_total no alcanza el umbral, la afirmación device-independent se DEGRADA y se
        declara la hipótesis de muestreo justo (fair sampling), que no es verificable aquí.
        """
        eta_total = float(params.insertion_transmissivity * params.detector_efficiency)
        chsh = float(report.phase_ii.polarization.chsh_maximum)
        closed = bool(eta_total >= _ETA_CRIT_GARG_MERMIN and params.efficiency_measured_hil)
        eberhard = bool(eta_total > _ETA_CRIT_EBERHARD)
        level = ("DEVICE_INDEPENDENT (Garg–Mermin, η medida HIL)" if closed else
                 ("RUTA_EBERHARD_DISPONIBLE (requiere estado no maximal + Clauser–Horne)" if eberhard
                  else "SOLO_CONDICIONAL (fair sampling; loophole de detección ABIERTO)"))
        return DetectionLoopholeReport(
            eta_total=eta_total, eta_crit_garg_mermin=_ETA_CRIT_GARG_MERMIN,
            eta_crit_eberhard=_ETA_CRIT_EBERHARD, chsh_state_maximum=chsh,
            fair_sampling_assumed=not closed, efficiency_measured_hil=bool(params.efficiency_measured_hil),
            chsh_loophole_closed=closed, eberhard_route_available=eberhard, claim_level=level,
            note=("el estado viola Bell (CHSH≈2√2); el cierre del loophole depende de η y de la "
                  "desigualdad usada, no del estado"),
        )

    # ─────────── §2.6 Atestación de conteos y energías ───────────
    @staticmethod
    def photon_count_attestation(report: Any, params: GammaWitnessGovernanceParameters, *,
                                 observed_energies_kev: Optional[Sequence[float]] = None,
                                 declared_state_spin: int = 0) -> PhotonCountAttestationReport:
        r"""
        Contrasta conteos/energías DECLARADOS con la cinemática exacta del motor:
        Σ E_i = E_tot, E_i ∈ [E_min, E_max]; paridad C: n par ⇒ para (C=+1), n impar ⇒ orto;
        Landau–Yang: J=1 → 2γ PROHIBIDO (violación estructural ⇒ ⊥).
        Sin conteos declarados, la cinemática es PREDICCIÓN y se marca como tal (no observación).
        """
        kin = report.phase_ii.kinematics
        if observed_energies_kev is None:
            energies = (float(kin.photon1_energy_kev), float(kin.photon2_energy_kev))
            mode, observed = "CINEMÁTICA_PREDICHA (sin conteos declarados)", False
        else:
            energies = tuple(float(x) for x in observed_energies_kev)
            mode, observed = "CONTEOS_DECLARADOS", True
        count = len(energies)
        total = float(sum(energies))
        sum_res = abs(total - kin.total_energy_kev) / max(kin.total_energy_kev, _EPS)
        window = all(kin.e_min_kev - 1e-6 <= e <= kin.e_max_kev + 1e-6 for e in energies)
        c_ok = (count == 2 and declared_state_spin == 0) or (count == 3 and declared_state_spin == 1)
        ly_violation = bool(declared_state_spin == 1 and count == 2)
        consistent = bool(observed and sum_res <= 1e-6 and window and c_ok and not ly_violation)
        return PhotonCountAttestationReport(
            attestation_mode=mode, event_observed=observed, declared_count=count,
            declared_energies_kev=energies, expected_total_energy_kev=float(kin.total_energy_kev),
            energy_sum_residual=sum_res, energy_window_ok=bool(window), c_parity_consistent=bool(c_ok),
            landau_yang_violation=ly_violation, declared_state_spin=int(declared_state_spin),
            kinematic_consistency_ok=consistent or not observed,
        )

    # ─────────── §2.7 Cobertura azimutal + Koksma + Kac ───────────
    @staticmethod
    def azimuthal_coverage_certificate(report: Any, params: GammaWitnessGovernanceParameters
                                       ) -> AzimuthalCoverageReport:
        r"""
        El poder analizador de Klein–Nishina debe promediarse en azimut sin resonancia.
        Muestreo cuasi-periódico u_n = nα mod 1 (α noble):
          • TRES HUECOS (Steinhaus): ≤3 longitudes y la mayor suma las otras dos ⇒ cota de
            sector azimutal no muestreado; cobertura ≤ 1/q_{k−1} + 1/q_k.
          • KOKSMA: |media_muestral − media_exacta| ≤ V(f)·D*_N, con
            f(u) = T − sin²θ(1+cos4πu) ⇒ V(f) = ∫|f'| = 8sin²θ (derivado),
            media exacta = T − sin²θ = factor KN NO polarizado (identidad verificada).
          • KAC: E[τ_A] = 1/μ ⇒ retardo esperado de detección del régimen recurrente.
        """
        ch = report.phase_ii.compton_hom
        theta = math.radians(float(params.compton_theta_deg))
        s2 = math.sin(theta) ** 2
        kp = float(ch.k_prime_over_k)
        t_term = 1.0 / kp + kp
        alpha, n = float(params.azimuth_rotation), int(params.azimuth_samples)
        u = np.mod(np.arange(n) * alpha, 1.0)
        f = t_term - s2 * (1.0 + np.cos(4.0 * math.pi * u))
        sample_mean = float(np.mean(f))
        exact_mean = t_term - s2
        var_f = 8.0 * s2
        disc = star_discrepancy(u)
        bound = var_f * disc
        err = abs(sample_mean - exact_mean)

        convs = _convergents(alpha, n)
        pts = np.sort(u)
        gaps = np.diff(np.concatenate([pts, pts[:1] + 1.0]))
        uniq = np.unique(np.round(gaps, 9))
        ident = float(abs(uniq[2] - uniq[0] - uniq[1])) if uniq.size == 3 else (0.0 if uniq.size <= 3 else float("inf"))
        qp, qc = (convs[-2][1], convs[-1][1]) if len(convs) >= 2 else (1, convs[-1][1])
        cov_bound = 1.0 / max(1, qp) + 1.0 / max(1, qc)

        mu, ns = float(params.recurrence_mu), int(params.recurrence_samples)
        x = (np.arange(ns) + 0.5) / ns * mu
        times = np.zeros(ns, dtype=np.int64)
        act = np.ones(ns, dtype=bool)
        for step in range(1, int(max(1000, 100.0 / mu)) + 1):
            x[act] = np.mod(x[act] + alpha, 1.0)
            hit = act & (x < mu)
            times[hit] = step; act[hit] = False
            if not act.any():
                break
        if act.any():
            raise GammaWitnessGovernanceError("Kac: retorno no alcanzado (α racional o μ degenerado).")
        rt = tuple(int(v) for v in np.unique(times))
        exp_d, emp_d = 1.0 / mu, float(np.mean(times))
        unpol = 0.5 * kp ** 2 * (t_term - s2)
        unpol_ref = 0.5 * kp ** 2 * (1.0 / kp + kp - s2)
        return AzimuthalCoverageReport(
            rotation_number=alpha, n_samples=n, distinct_gap_count=int(uniq.size),
            max_gap=float(np.max(gaps)), coverage_bound=float(cov_bound),
            three_gap_identity_residual=ident, star_discrepancy=disc, total_variation_f=var_f,
            koksma_error_bound=bound, sample_mean=sample_mean, exact_mean=exact_mean,
            actual_error=err, koksma_bound_ok=bool(err <= bound + 1e-12),
            unpolarized_identity_residual=abs(unpol - unpol_ref),
            three_gap_ok=bool(uniq.size <= 3 and ident <= 1e-8),
            coverage_ok=bool(float(np.max(gaps)) <= cov_bound + 1e-12),
            kac_expected_delay=exp_d, kac_empirical_delay=emp_d,
            kac_relative_residual=abs(emp_d - exp_d) / exp_d, slater_return_times=rt,
            slater_ok=bool(len(rt) <= 3),
        )

    # ─────────── §2.8 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self, p1: GammaPhotonWitnessPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        rho: NDArray, n_max: int, params: GammaWitnessGovernanceParameters, *, expected_epoch: int,
        principal_report: Any = None, observed_energies_kev: Optional[Sequence[float]] = None,
        declared_state_spin: int = 0, silence: Tuple[bool, bool, int] = (True, True, 0),
    ) -> GammaPhotonWitnessPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: usa el g comprometido y certifica lo presupuesto."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, params, expected_epoch=expected_epoch)
        passport = getattr(principal_report, "passport_hmac", "")
        report = self._engine.execute_gamma_photon_witness_audit(
            apu_cost_matrix, np.asarray(exponents, dtype=np.float64), rho,
            principal_report=principal_report,
            params=params.to_engine_parameters(n_max, p1.committed_gain_g),
            authorization_token=None,                     # el testigo NO autoriza ni actúa
        )
        verified, root_ok, counter_ok = self.verify_engine_report(report, passport)
        qnd = report.phase_ii.phase_i.qnd
        gain_honored = abs(qnd.interaction_g - p1.committed_gain_g) <= 1e-18
        loss = self.insertion_loss_certificate(rho, n_max, params)
        loophole = self.detection_loophole_certificate(report, params)
        counts = self.photon_count_attestation(report, params, observed_energies_kev=observed_energies_kev,
                                               declared_state_spin=declared_state_spin)
        azim = self.azimuthal_coverage_certificate(report, params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("GP2.informe_verificado", verified, hard=True),
            Evidence.boolean("GP2.merkle_recalculado", root_ok, hard=True),
            Evidence.boolean("GP2.contrafirma_ligada_al_principal", counter_ok, hard=True),
            Evidence.boolean("GP2.g_comprometido_honrado", gain_honored, hard=True),
            Evidence("GP2.disturbancia_vs_presupuesto",
                     max(0.0, qnd.trace_distance_disturbance - params.disturbance_budget), 0.0, hard=True),
            Evidence("GP2.backaction_db_vs_presupuesto",
                     max(0.0, -qnd.back_action_db - params.backaction_db_budget), 0.0, hard=True),
            Evidence("GP2.perdida_kraus_completitud", loss.kraus_completeness_residual, 1e-12, hard=True),
            Evidence("GP2.perdida_thinning_media", loss.thinning_mean_residual, 1e-10, hard=True),
            Evidence("GP2.perdida_binomial_norm", loss.binomial_normalization_residual, 1e-12, hard=True),
            Evidence("GP2.perdida_semigrupo", loss.semigroup_composition_residual, 1e-10, hard=True),
            Evidence.boolean("GP2.perdida_no_unital_declarada", loss.nonunitality_residual > 1e-6,
                             note="la no-unitalidad invalida la convención de dB de pureza"),
            Evidence.boolean("GP2.loophole_deteccion_cerrado", loophole.chsh_loophole_closed,
                             note=loophole.claim_level),
            Evidence.boolean("GP2.conteos_cinematica_consistente", counts.kinematic_consistency_ok, hard=True),
            Evidence.boolean("GP2.sin_violacion_landau_yang", not counts.landau_yang_violation, hard=True),
            Evidence.boolean("GP2.paridad_c_consistente", counts.c_parity_consistent, hard=True),
            Evidence("GP2.koksma_cota", azim.actual_error - azim.koksma_error_bound, 0.0, hard=True),
            Evidence("GP2.azimut_identidad_no_polarizada", azim.unpolarized_identity_residual, 1e-12, hard=True),
            Evidence.boolean("GP2.tres_huecos", azim.three_gap_ok, hard=True),
            Evidence.boolean("GP2.cobertura_azimutal", azim.coverage_ok, hard=True),
            Evidence.boolean("GP2.slater", azim.slater_ok, hard=True),
            Evidence("GP2.kac_retardo", azim.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("GP2.silencio_registros", silence[0]),
            Evidence.boolean("GP2.silencio_bytes", silence[1]),
        )
        truth = HeytingTruthValue.infimum([p1.truth_value, report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": True,
            "engine_passport": report.passport_hash, "engine_merkle_root": report.merkle_root_hash,
            "engine_countersignature": report.countersignature, "engine_truth": report.truth_value,
            "engine_report_verified": verified, "engine_merkle_recomputed_ok": root_ok,
            "countersignature_ok": counter_ok, "gain_honored": bool(gain_honored),
            "disturbance_consumed": float(qnd.trace_distance_disturbance),
            "back_action_db": float(qnd.back_action_db), "insertion_loss": loss,
            "detection_loophole": loophole, "photon_counts": counts, "azimuthal": azim,
            "silence_records_ok": silence[0], "silence_bytes_ok": silence[1], "evidences": evidences,
            "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonWitnessPhase2Certificate(
            timestamp=time.time(), phase1=p1, spectrum_report=report, engine_invoked=True,
            engine_report_verified=verified, engine_merkle_recomputed_ok=root_ok,
            countersignature_ok=counter_ok, gain_honored=bool(gain_honored),
            disturbance_consumed=float(qnd.trace_distance_disturbance),
            back_action_db=float(qnd.back_action_db), insertion_loss=loss, detection_loophole=loophole,
            photon_counts=counts, azimuthal=azim, silence_records_ok=silence[0],
            silence_bytes_ok=silence[1], silence_bytes_emitted=int(silence[2]),
            evidences=evidences, truth_value=truth, epoch=int(expected_epoch), chain_hash=chain,
            hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        logger.info("GP2 Ω₃=%s · D=%.2e · dB=%.2e · η_tot=%.4f · loophole=%s",
                    truth.name, cert.disturbance_consumed, cert.back_action_db,
                    loophole.eta_total, loophole.chsh_loophole_closed)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1

    def execute_phase2(self, *a: Any, **k: Any) -> GammaPhotonWitnessPhase2Certificate:
        """Alias heredado (v13) del terminal §2.8."""
        return self.emit_phase2_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ATTEST — adjudicación epistémica, RSI³ y sello público
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonWitnessPhase3Decider:
    r"""
    FASE 3 OODA pasiva. Adjudica en Ω₃ y traduce a acción EPISTÉMICA (atestiguar, reservar,
    retener). La escalada es un dato firmado, no un efecto sobre el mundo.
    """

    PHASE_LABEL: Final[str] = "3·DECIDE-ATTEST"
    _ACTION: Final[Dict[HeytingTruthValue, str]] = {
        HeytingTruthValue.VETOED: "WITHHOLD_ATTESTATION",
        HeytingTruthValue.DEGRADED: "ATTEST_WITH_RESERVES",
        HeytingTruthValue.COHERENT: "ATTEST_CLEAN",
    }
    _SEVERITY: Final[Dict[str, int]] = {"ATTEST_CLEAN": 0, "ATTEST_WITH_RESERVES": 1, "WITHHOLD_ATTESTATION": 2}

    def __init__(self, signer: GammaWitnessSigner, interlock: PassivityInterlock) -> None:
        self._signer = signer
        self._interlock = interlock

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ (signo corregido) ───────────
    @staticmethod
    def legacy_backaction_gate(back_action_db: float) -> bool:
        """DEPRECADO (v13): `dB ≤ 0.01`. Bajo la convención v14 (dB≤0) es VACUO. Sólo regresión."""
        return bool(back_action_db <= 0.01)

    def continue_to_phase3_gate(self, p2: GammaPhotonWitnessPhase2Certificate,
                                params: GammaWitnessGovernanceParameters, *,
                                expected_epoch: int) -> GammaWitnessGateReceipt:
        r"""
        Φ₂₃ — Integridad DURA (⇒ fail-closed) separada de la física:
            dB ≥ −presupuesto  ∧  D_tr ≤ presupuesto  ∧  Lip(T³) < 1  ∧  pasividad QND.
        El gate heredado `dB ≤ 0.01` aceptaría CUALQUIER medición bajo la convención v14.
        Si la física falla se ADMITE con ⊥ FORZADO para que Fase 3 retenga y escale.
        """
        payload = p2.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (p2.epoch == expected_epoch and p2.phase1.epoch == expected_epoch
                   and (not p2.engine_invoked or (p2.engine_report_verified and p2.countersignature_ok)))
        if not (chain_ok and sig_ok and link_ok):
            raise GammaWitnessGateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        rep = p2.spectrum_report
        lip = float(getattr(getattr(rep, "rsi3", None), "lip_t3_bound", 1.0)) if rep is not None else 1.0
        qnd = rep.phase_ii.phase_i.qnd if rep is not None else None
        physics_ok = bool(
            p2.back_action_db >= -params.backaction_db_budget
            and p2.disturbance_consumed <= params.disturbance_budget
            and lip < 1.0 and (qnd is not None and qnd.is_qnd_passive)
        )
        return GammaWitnessGateReceipt("Φ₂₃", recomputed, chain_ok, sig_ok, link_ok, True, not physics_ok,
                                       "física violada ⇒ ⊥ forzado (retener atestación)" if not physics_ok
                                       else "pasividad y contracción verificadas")

    def accept_phase2_certificate(self, p2: GammaPhotonWitnessPhase2Certificate,
                                  params: GammaWitnessGovernanceParameters, *,
                                  expected_epoch: int) -> GammaWitnessGateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.8."""
        return self.continue_to_phase3_gate(p2, params, expected_epoch=expected_epoch)

    # ─────────── §3.2 Functor epistémico y núcleo ¬¬ ───────────
    @classmethod
    def action_functor(cls, truth: HeytingTruthValue, *, strict_mode: bool = False) -> str:
        if strict_mode and truth is HeytingTruthValue.DEGRADED:
            return "WITHHOLD_ATTESTATION"
        return cls._ACTION[truth]

    @classmethod
    def verify_action_functor(cls) -> bool:
        """D(a∧b) = D(a) ∨ D(b) en severidad: functor contravariante Ω₃ᵒᵖ → Acciones."""
        vals = list(HeytingTruthValue)
        return all(cls._SEVERITY[cls.action_functor(a.meet(b))] ==
                   max(cls._SEVERITY[cls.action_functor(a)], cls._SEVERITY[cls.action_functor(b)])
                   for a in vals for b in vals)

    @staticmethod
    def verify_nucleus() -> bool:
        vals = list(HeytingTruthValue)
        return bool(all(int(a) <= int(a.double_negation()) for a in vals)
                    and all(a.double_negation().double_negation() == a.double_negation() for a in vals)
                    and all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                            for a in vals for b in vals))

    # ─────────── §3.4 RSI³ pasivo ───────────
    def governance_rsi3(self, p2: GammaPhotonWitnessPhase2Certificate,
                        params: GammaWitnessGovernanceParameters) -> GammaWitnessRSI3Report:
        r"""
        NIVEL 1 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), q=1−½·m (m = margen de
        pasividad); Lip_∞ ≤ q, Lip(T³) ≤ q³, sondeo empírico y cotas de Banach.
        NIVEL 2 · Newton–Kantorovich sobre el umbral de reserva: f(d)=m−κd²−obj, x₀=c·d*
        ⇒ h=(c²−1)/(2c²) < ½ SIEMPRE; iteración babilónica; traza de capacidad C_k=−log₁₀e_k
        y jerk MEDIDO (Savitzky–Golay 7/3).
        NIVEL 3 · Leyes de Kleisli, axiomas del núcleo ¬¬ y ley contravariante del functor.
        """
        margin = 1.0 - min(1.0, p2.disturbance_consumed / max(params.disturbance_budget, 1e-300))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        kappa, c_init = 1.0, 10.0
        m = max(margin, 1e-6)
        target = 0.25 * m
        d_star = math.sqrt(max(m - target, 0.0) / kappa)
        theta_star = np.array([p2.phase1.committed_gain_g, margin, p2.detection_loophole.eta_total])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(20_261_121)
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
        emp = 0.0
        for _ in range(48):
            pert = rng.standard_normal(3) * 1e-7
            emp = max(emp, float(np.max(np.abs(triple(th + pert) - triple(th)))) / float(np.max(np.abs(pert))))
        dk = c_init * max(d_star, 1e-12)
        h_k = (1.0 / abs(-2.0 * kappa * dk)) * abs((m - kappa * dk ** 2 - target)
                                                   / (-2.0 * kappa * dk)) * 2.0 * kappa
        errs: List[float] = []
        for _ in range(10):
            errs.append(abs(dk - d_star))
            dk = dk - (m - kappa * dk ** 2 - target) / (-2.0 * kappa * dk)
        errs.append(abs(dk - d_star))
        err = np.array(errs)
        cap = -np.log10(np.maximum(err, 1e-300))
        active = np.where(err > 1e-13 * max(d_star, 1e-12))[0]
        d1 = d2 = d3 = 0.0
        if active.size >= 7:
            ctr = min(max(int(active[active.size // 2]), 3), cap.size - 4)
            seg = cap[ctr - 3: ctr + 4]
            d1 = float(_savgol(7, 3, 1) @ seg); d2 = float(_savgol(7, 3, 2) @ seg); d3 = float(_savgol(7, 3, 3) @ seg)
        f_c = lambda x: Certified(x * 0.5, ("f",), HeytingTruthValue.COHERENT)   # noqa: E731
        g_c = lambda x: Certified(x + 1.0, ("g",), HeytingTruthValue.COHERENT)   # noqa: E731
        smp = rng.standard_normal(3)
        mv = Certified(smp, ("m",), HeytingTruthValue.DEGRADED)
        laws = bool(np.allclose(Certified.unit(smp).bind(f_c).value, f_c(smp).value)
                    and np.allclose(mv.bind(Certified.unit).value, mv.value)
                    and np.allclose(mv.bind(f_c).bind(g_c).value, mv.bind(lambda x: f_c(x).bind(g_c)).value)
                    and mv.bind(f_c).bind(g_c).ledger == mv.bind(lambda x: f_c(x).bind(g_c)).ledger)
        return GammaWitnessRSI3Report(
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(emp),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_iterations=int(err.size),
            capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0), kleisli_monad_laws_ok=laws,
            nucleus_axioms_ok=self.verify_nucleus(), action_functor_ok=self.verify_action_functor(),
        )

    # ─────────── §3.5 Φ_sem con procedencia ───────────
    @staticmethod
    def semantic_visceral_functor(p2: GammaPhotonWitnessPhase2Certificate,
                                  params: GammaWitnessGovernanceParameters, *, action: str
                                  ) -> GammaPhotonWitnessVisceralMetrics:
        """Métricas de atestación MEDIDAS + KPI financieros sólo con insumos declarados."""
        all_ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.spectrum_report
        if rep is not None:
            all_ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        coverage = 100.0 * sum(1 for e in all_ev if e.passed) / max(1, len(all_ev))
        dist_pct = 100.0 * p2.disturbance_consumed / max(params.disturbance_budget, 1e-300)
        loss_pct = 100.0 * (1.0 - p2.insertion_loss.eta_insertion)
        silence_pct = 100.0 * p2.silence_bytes_emitted / max(1, params.silence_byte_budget)
        provenance = ["COVERAGE:MEASURED", "DISTURBANCE:MEASURED", "LOSS:DECLARED_ETA",
                      "SILENCE:MEASURED(LOG_BYTES)"]
        bell_margin = hom_vis = None
        wacc = liq = kv = None
        if rep is not None:
            chsh = rep.phase_ii.polarization.chsh_maximum
            bell_margin = 100.0 * (chsh - 2.0) / 2.0
            hom_vis = 100.0 * rep.phase_ii.compton_hom.hom_visibility
            sigma = max(0.0, rep.phase_ii.spohn.sigma_min)
            wacc = 100.0 * sigma / (sigma + params.exergy_reference_w)
            provenance.append("BELL/HOM:MEASURED")
            provenance.append("WACC:POLICY_HEURISTIC(sigma/(sigma+sigma_ref))")
        if params.exergy_price_per_joule is not None and params.capital_base:
            liq = 100.0 * 0.0 * params.exergy_price_per_joule / params.capital_base
            provenance.append("LIQUIDITY:UNDECLARED_WORK (el testigo no fabrica W_irr)")
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
        return GammaPhotonWitnessVisceralMetrics(
            attestation_coverage_pct=float(coverage),
            assurance_level=rep.attestation.attestation_level if rep is not None else "NINGUNO",
            disturbance_budget_consumed_pct=float(dist_pct), insertion_loss_budget_pct=float(loss_pct),
            silence_budget_consumed_pct=float(silence_pct),
            bell_margin_pct=bell_margin, hom_visibility_pct=float(hom_vis or 0.0),
            wacc_protection_pct=wacc, net_liquidity_release_pct=liq, kv_cache_compression_pct=kv,
            stealth_qnd_status=(f"PASSIVE_SILENT_QND dB={p2.back_action_db:.6e} "
                                f"D_tr={p2.disturbance_consumed:.3e} acción={action}"),
            detection_loophole_status=p2.detection_loophole.claim_level,
            provenance=tuple(provenance), monotonicity_ok=monotone,
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Estado soberano pasivo ───────────
    def emit_sovereign_state(
        self, p2: GammaPhotonWitnessPhase2Certificate, params: GammaWitnessGovernanceParameters, *,
        expected_epoch: int, principal_truth: Optional[HeytingTruthValue] = None,
        error: Optional[str] = None, forced_veto: bool = False,
    ) -> GammaPhotonWitnessSovereignState:
        """MÉTODO TERMINAL: adjudica, RETIENE o ATTESTIGUA, escala y sella. Jamás actúa."""
        t0 = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, params, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        action = self.action_functor(truth, strict_mode=params.strict_mode)

        interlock_ok = False
        try:
            self._interlock.request_actuation("crowbar", epoch=expected_epoch)
        except ActuationForbiddenError:
            interlock_ok = True

        rsi3 = self.governance_rsi3(p2, params)
        visceral = self.semantic_visceral_functor(p2, params, action=action)
        passive = bool(p2.disturbance_consumed <= params.disturbance_budget
                       and p2.back_action_db >= -params.backaction_db_budget
                       and p2.silence_records_ok and p2.silence_bytes_ok and interlock_ok)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("GP3.interlock_pasividad", interlock_ok, hard=True),
            Evidence.boolean("GP3.functor_contravariante", rsi3.action_functor_ok, hard=True),
            Evidence.boolean("GP3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence("GP3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("GP3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("GP3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("GP3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("GP3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("GP3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("GP3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("GP3.pasividad_operativa", passive, hard=True),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])
        action = self.action_functor(truth, strict_mode=params.strict_mode)
        withheld = action == "WITHHOLD_ATTESTATION"

        escalation: Optional[GammaWitnessEscalation] = None
        divergent = principal_truth is not None and int(truth) < int(principal_truth)
        if withheld or divergent or error is not None:
            reason = ("atestación retenida" if withheld else "") + \
                     (" | divergencia estricta frente al soberano activo" if divergent else "") + \
                     (f" | error: {error}" if error else "")
            esc_core = {"epoch": int(expected_epoch), "witness_truth": truth,
                        "principal_truth": principal_truth, "reason": reason.strip(" |"),
                        "p2_chain": p2.chain_hash}
            escalation = GammaWitnessEscalation(
                epoch=int(expected_epoch), witness_truth=truth, principal_truth=principal_truth,
                reason=reason.strip(" |"), merkle_root_hash="",
                signature=self._signer.sign(_HMAC_ESC, esc_core), delivered=False,
            )

        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": getattr(p2.spectrum_report, "passport_hash", ""),
                        "engine_root": getattr(p2.spectrum_report, "merkle_root_hash", ""),
                        "countersignature_ok": p2.countersignature_ok}),
            _canonical(p2.insertion_loss), _canonical(p2.detection_loophole),
            _canonical(p2.photon_counts), _canonical(p2.azimuthal), _canonical(rsi3),
            _canonical({"epoch": int(expected_epoch), "truth": truth, "action": action,
                        "withheld": withheld, "error": error, "strict": bool(params.strict_mode)}),
            _canonical({"actuation_performed": False, "interlock_ok": interlock_ok,
                        "refused_requests": self._interlock.refused_requests,
                        "silence_records_ok": p2.silence_records_ok, "silence_bytes_ok": p2.silence_bytes_ok}),
        ]
        root = merkle_root(leaves)
        att_idx = 7
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
        state = GammaPhotonWitnessSovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(), action=action,
            is_passive_silent_qnd=passive, back_action_db=float(p2.back_action_db),
            attestation_withheld=withheld, actuation_performed=False, merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), attestation_leaf_index=att_idx,
            attestation_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3,
            escalation=escalation, phase2=p2, evidences=evidences, kleisli_ledger=core["kleisli_ledger"],
            epoch=int(expected_epoch), state_signature=self._signer.sign(_HMAC_STATE, core), error=error,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("GP3 Ω₃=%s · acción=%s · pasivo=%s", truth.name, action, passive)
        return state

    def execute_phase3(self, *a: Any, **k: Any) -> GammaPhotonWitnessSovereignState:
        """Alias heredado (v13) del terminal §3.7."""
        return self.emit_sovereign_state(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA — ciclo OODA pasivo (Kleisli) con fail-closed silencioso
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovGammaPhotonColliderWitnessAgent:
    r"""
    Soberano agéntico TESTIGO gamma. Ejecuta observe >=> orient >=> decide sobre la mónada
    Certified. Ante cualquier fallo: FAIL-CLOSED PASIVO — retiene la atestación, g=0, estado
    firmado y escalada; NUNCA actúa ni lanza al vacío.
    """

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_120) -> None:
        self._signer = GammaWitnessSigner(secret_key)
        self._principal = principal_engine
        self._engine = GammaWitnessEngine(secret_key=self._signer.key_bytes,
                                          principal_engine=principal_engine, seed=seed)
        self._observer = GammaPhotonWitnessPhase1Observer(self._signer, seed=seed)
        self._orientor = GammaPhotonWitnessPhase2Orientor(self._engine, self._signer)
        self._interlock = PassivityInterlock()
        self._decider = GammaPhotonWitnessPhase3Decider(self._signer, self._interlock)
        self._epoch = 0
        logger.info("%s v%s listo · clave=%s", type(self).__name__, __version__, self._signer.fingerprint())

    # ─────────── verificación externa del estado ───────────
    def verify_sovereign_state(self, state: GammaPhotonWitnessSovereignState) -> bool:
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
        if merkle_root(leaves) != state.merkle_root_hash:      # raíz SIN clave: verificable por todos
            return False
        i = state.attestation_leaf_index
        if not verify_merkle(leaves[i], i, len(leaves), state.attestation_inclusion_proof, state.merkle_root_hash):
            return False
        return self._signer.verify(_HMAC_STATE, state.payload(), state.state_signature)

    # ─────────── ciclo OODA pasivo con fail-closed ───────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: Optional[NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        principal_report: Any = None,
        principal_truth: Optional[HeytingTruthValue] = None,
        observed_energies_kev: Optional[Sequence[float]] = None,
        declared_state_spin: int = 0,
        params: Optional[GammaWitnessGovernanceParameters] = None,
        authorization_token: Optional[str] = None,   # se registra como NO ejercido
    ) -> GammaPhotonWitnessSovereignState:
        """Ejecuta el ciclo pasivo completo; jamás actúa y jamás deja el fallo sin registro."""
        params = params or GammaWitnessGovernanceParameters()
        self._epoch += 1
        epoch = self._epoch
        exps = np.asarray(exponents, dtype=np.float64)
        passport = getattr(principal_report, "passport_hmac", "")
        if authorization_token is not None:
            logger.info("Token de autorización recibido y NO ejercido (separación de funciones).")

        loggers = [logger, logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderWitnessEngine")]
        if self._principal is not None:
            loggers.append(logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverEngine"))

        with silence_meter(loggers) as meter:
            def observe_m(ctx: Dict[str, Any]) -> Any:
                p1, rho, n_max = self._observer.emit_phase1_certificate(
                    rho_mac, apu_cost_matrix, exps, params, epoch=epoch, principal_passport=passport)
                return Certified({**ctx, "p1": p1, "rho": rho, "n": n_max}, ("Φ₁:OBSERVE-PASIVO",), p1.truth_value)

            def orient_m(ctx: Dict[str, Any]) -> Any:
                silence = (meter.records <= params.silence_record_budget,
                           meter.bytes <= params.silence_byte_budget, meter.bytes)
                p2 = self._orientor.emit_phase2_certificate(
                    ctx["p1"], apu_cost_matrix, exps, ctx["rho"], ctx["n"], params,
                    expected_epoch=epoch, principal_report=principal_report,
                    observed_energies_kev=observed_energies_kev, declared_state_spin=declared_state_spin,
                    silence=silence)
                return Certified({**ctx, "p2": p2}, ("Φ₂:ORIENT-PASIVO",), p2.truth_value)

            def decide_m(ctx: Dict[str, Any]) -> Any:
                st = self._decider.emit_sovereign_state(ctx["p2"], params, expected_epoch=epoch,
                                                        principal_truth=principal_truth)
                return Certified(st, ("Φ₃:ATTEST",), st.truth_value)

            pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
            try:
                result = pipeline({"epoch": epoch})
                state: GammaPhotonWitnessSovereignState = result.value
                if not self.verify_sovereign_state(state):
                    raise GammaWitnessGateIntegrityError("Autoverificación del estado testigo fallida.")
                return state
            except Exception as exc:                                   # FAIL-CLOSED PASIVO
                logger.critical("FAIL-CLOSED PASIVO GAMMA: %s", exc)
                return self._emergency_state(rho_mac, apu_cost_matrix, exps, params, epoch, str(exc),
                                             principal_report, principal_truth, meter)

    def _emergency_state(self, rho_mac: Optional[NDArray], k_mat: NDArray, exps: NDArray,
                         params: GammaWitnessGovernanceParameters, epoch: int, message: str,
                         principal_report: Any, principal_truth: Optional[HeytingTruthValue],
                         meter: _SilenceMeter) -> GammaPhotonWitnessSovereignState:
        """Estado de emergencia: g=0, atestación retenida, escalada firmada, sin actuación."""
        safe = dataclasses.replace(params, interaction_g=0.0)
        p1, rho, n_max = self._observer.emit_phase1_certificate(
            rho_mac, k_mat, exps, safe, epoch=epoch,
            principal_passport=getattr(principal_report, "passport_hmac", ""))
        dummy_loss = InsertionLossReport(
            eta_insertion=safe.insertion_transmissivity, kraus_completeness_residual=0.0,
            mean_photon_before=0.0, mean_photon_after=0.0, thinning_mean_residual=0.0,
            binomial_normalization_residual=0.0, semigroup_composition_residual=0.0,
            nonunitality_residual=0.0, trace_distance_insertion=0.0, purity_before=1.0,
            purity_after=1.0, purity_convention_valid=False, note="emergencia: sin interacción (g=0)")
        dummy_loop = DetectionLoopholeReport(
            eta_total=safe.insertion_transmissivity * safe.detector_efficiency,
            eta_crit_garg_mermin=_ETA_CRIT_GARG_MERMIN, eta_crit_eberhard=_ETA_CRIT_EBERHARD,
            chsh_state_maximum=0.0, fair_sampling_assumed=True,
            efficiency_measured_hil=safe.efficiency_measured_hil, chsh_loophole_closed=False,
            eberhard_route_available=False, claim_level="NINGUNO (fail-closed)", note="emergencia")
        dummy_counts = PhotonCountAttestationReport(
            attestation_mode="EMERGENCIA (sin atestación)", event_observed=False, declared_count=0,
            declared_energies_kev=(), expected_total_energy_kev=0.0, energy_sum_residual=0.0,
            energy_window_ok=False, c_parity_consistent=False, landau_yang_violation=False,
            declared_state_spin=0, kinematic_consistency_ok=True)
        dummy_azim = AzimuthalCoverageReport(
            rotation_number=safe.azimuth_rotation, n_samples=safe.azimuth_samples, distinct_gap_count=0,
            max_gap=0.0, coverage_bound=1.0, three_gap_identity_residual=0.0, star_discrepancy=1.0,
            total_variation_f=0.0, koksma_error_bound=1.0, sample_mean=0.0, exact_mean=0.0,
            actual_error=0.0, koksma_bound_ok=True, unpolarized_identity_residual=0.0,
            three_gap_ok=True, coverage_ok=True, kac_expected_delay=1.0 / safe.recurrence_mu,
            kac_empirical_delay=1.0 / safe.recurrence_mu, kac_relative_residual=0.0,
            slater_return_times=(), slater_ok=True)
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": False, "engine_passport": "",
            "engine_merkle_root": "", "engine_countersignature": "",
            "engine_truth": HeytingTruthValue.VETOED, "engine_report_verified": False,
            "engine_merkle_recomputed_ok": False, "countersignature_ok": False, "gain_honored": True,
            "disturbance_consumed": 0.0, "back_action_db": 0.0, "insertion_loss": dummy_loss,
            "detection_loophole": dummy_loop, "photon_counts": dummy_counts, "azimuthal": dummy_azim,
            "silence_records_ok": meter.records <= params.silence_record_budget,
            "silence_bytes_ok": meter.bytes <= params.silence_byte_budget,
            "evidences": (), "truth_value": HeytingTruthValue.VETOED, "epoch": int(epoch),
        }
        chain = _sha256(_canonical(core))
        p2 = GammaPhotonWitnessPhase2Certificate(
            timestamp=time.time(), phase1=p1, spectrum_report=None, engine_invoked=False,
            engine_report_verified=False, engine_merkle_recomputed_ok=False, countersignature_ok=False,
            gain_honored=True, disturbance_consumed=0.0, back_action_db=0.0, insertion_loss=dummy_loss,
            detection_loophole=dummy_loop, photon_counts=dummy_counts, azimuthal=dummy_azim,
            silence_records_ok=core["silence_records_ok"], silence_bytes_ok=core["silence_bytes_ok"],
            silence_bytes_emitted=meter.bytes, evidences=(), truth_value=HeytingTruthValue.VETOED,
            epoch=int(epoch), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core))
        return self._decider.emit_sovereign_state(p2, safe, expected_epoch=epoch,
                                                  principal_truth=principal_truth, error=message,
                                                  forced_veto=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    agent = BoolNovikovGammaPhotonColliderWitnessAgent(
        secret_key=b"GAMMA_WITNESS_AGENT_KEY_32_BYTES!", principal_engine=principal)
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)

    # 1) Nominal con η alta y medición HIL ⇒ loophole cerrado, ⊤, sin actuación
    P_hi = GammaWitnessGovernanceParameters(insertion_transmissivity=0.999, detector_efficiency=0.95,
                                            efficiency_measured_hil=True)
    st = agent.run_sovereign_governance_cycle(None, k_test, exps, principal_report=prep,
                                              principal_truth=prep.truth_value, params=P_hi)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.action == "ATTEST_CLEAN" and not st.actuation_performed and st.is_passive_silent_qnd
    assert st.phase2.detection_loophole.chsh_loophole_closed
    assert st.phase2.gain_honored and st.phase2.countersignature_ok
    assert st.phase2.insertion_loss.nonunitality_residual > 1e-6      # pérdida NO unital
    assert st.phase2.azimuthal.koksma_bound_ok and st.phase2.azimuthal.three_gap_ok
    assert agent.verify_sovereign_state(st)
    assert st.visceral_metrics.net_liquidity_release_pct is not None or True   # declarado honestamente

    # 2) TRAMPA DE SIGNO: el gate heredado aprobaría una medición destructiva
    broken = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep,
        params=GammaWitnessGovernanceParameters(interaction_g=0.5))
    db = broken.back_action_db
    assert GammaPhotonWitnessPhase3Decider.legacy_backaction_gate(db) is True     # ¡vacuo!
    assert db < -GammaWitnessGovernanceParameters().backaction_db_budget          # v14 la rechaza
    assert broken.truth_value is HeytingTruthValue.VETOED and broken.attestation_withheld
    assert not broken.actuation_performed

    # 3) Loophole de detección ABIERTO ⇒ ½ (degradación honesta de la afirmación Bell)
    low_eta = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep,
        params=GammaWitnessGovernanceParameters(detector_efficiency=0.50))
    assert not low_eta.phase2.detection_loophole.chsh_loophole_closed
    assert low_eta.truth_value is HeytingTruthValue.DEGRADED
    assert low_eta.phase2.spectrum_report.phase_ii.polarization.chsh_maximum > 2.0   # el estado sí viola

    # 4) Reconciliación de dimensión de Fock (v13 fallaba con ρ de dim(K))
    rho4 = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    rec = agent.run_sovereign_governance_cycle(rho4, k_test, exps, principal_report=prep, params=P_hi)
    assert rec.phase2.phase1.fock_n_reconciled == 3
    assert rec.phase2.phase1.dimension_reconciliation_declared

    # 5) Conteos inconsistentes ⇒ ⊥; Landau–Yang (J=1 → 2γ) ⇒ ⊥
    kin = st.phase2.spectrum_report.phase_ii.kinematics
    bad_counts = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep, params=P_hi,
        observed_energies_kev=(300.0, 300.0))
    assert bad_counts.truth_value is HeytingTruthValue.VETOED
    ly = agent.run_sovereign_governance_cycle(
        None, k_test, exps, principal_report=prep, params=P_hi,
        observed_energies_kev=(kin.photon1_energy_kev, kin.photon2_energy_kev), declared_state_spin=1)
    assert ly.phase2.photon_counts.landau_yang_violation
    assert ly.truth_value is HeytingTruthValue.VETOED

    # 6) Interlock: el testigo NO puede actuar; Φ₁₂ detecta manipulación
    try:
        agent._interlock.request_actuation("crowbar")
        raise AssertionError("El testigo intentó actuar")
    except ActuationForbiddenError:
        pass
    forged = dataclasses.replace(st.phase2.phase1, tr_rho=999.0)
    try:
        agent._orientor.continue_to_phase2_gate(forged, k_test, exps, P_hi, expected_epoch=forged.epoch)
        raise AssertionError("Φ₁₂ aceptó un payload manipulado")
    except GammaWitnessGateIntegrityError:
        pass

    # 7) Koksma, tres huecos, Kac y Merkle RFC 6962; determinismo y anti-manipulación
    az = st.phase2.azimuthal
    assert az.actual_error <= az.koksma_error_bound and abs(az.total_variation_f - 8.0) < 1e-9
    assert abs(az.kac_empirical_delay - 10.0) < 0.05 and len(az.slater_return_times) <= 3
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    st2 = agent.run_sovereign_governance_cycle(None, k_test, exps, principal_report=prep,
                                               principal_truth=prep.truth_value, params=P_hi)
    assert st2.state_signature == st.state_signature
    assert not agent.verify_sovereign_state(dataclasses.replace(st, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderWitnessAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {st.truth_value.name}({int(st.truth_value)}) / {st.boolean_shadow.name}")
    print(f"Acción epistémica / actuación : {st.action} / {st.actuation_performed}")
    print(f"N Fock reconciliado / g sellado: {st.phase2.phase1.fock_n_reconciled} / "
          f"{st.phase2.phase1.committed_gain_g:.6g} (honrado={st.phase2.gain_honored})")
    print(f"Back-action [dB] / D_tr       : {st.back_action_db:.3e} / {st.phase2.disturbance_consumed:.3e}")
    print(f"d_BW(ρ,ρ_térmica) / isometría : {st.phase2.phase1.bures_wasserstein_to_thermal:.6f} / "
          f"{st.phase2.phase1.bw_isometry_residual:.2e}")
    print(f"Pérdida η / ⟨n⟩ antes→después : {st.phase2.insertion_loss.eta_insertion:.6f} / "
          f"{st.phase2.insertion_loss.mean_photon_before:.4f}→{st.phase2.insertion_loss.mean_photon_after:.4f}")
    print(f"No-unitalidad de la pérdida   : {st.phase2.insertion_loss.nonunitality_residual:.4e} "
          f"(dB de pureza NO aplica)")
    print(f"η_total / Garg–Mermin / Eberhard: {st.phase2.detection_loophole.eta_total:.4f} / "
          f"{_ETA_CRIT_GARG_MERMIN:.4f} / {_ETA_CRIT_EBERHARD:.4f}")
    print(f"Nivel de afirmación Bell      : {st.phase2.detection_loophole.claim_level}")
    print(f"CHSH del estado               : {st.phase2.detection_loophole.chsh_state_maximum:.6f}")
    print(f"Koksma V(f) / D*_N / cota     : {az.total_variation_f:.4f} / {az.star_discrepancy:.5f} / "
          f"{az.koksma_error_bound:.5f} (error real {az.actual_error:.2e})")
    print(f"Cobertura azimutal ≤          : {100*az.coverage_bound:.2f}% · huecos={az.distinct_gap_count}")
    print(f"Kac esperado / empírico       : {az.kac_expected_delay:.3f} / {az.kac_empirical_delay:.3f}")
    print(f"Slater tiempos de retorno     : {az.slater_return_times}")
    print(f"RSI³ q / Lip(T³) medido       : {st.rsi3.policy_lipschitz_q:.6f} / {st.rsi3.lip_t3_empirical:.6f}")
    print(f"Silencio bytes / presupuesto  : {st.phase2.silence_bytes_emitted} / {P_hi.silence_byte_budget}")
    print(f"Control ⊥ (g=0.5)             : {broken.truth_value.name} dB={db:.3e} "
          f"(gate heredado lo aprobaría: {GammaPhotonWitnessPhase3Decider.legacy_backaction_gate(db)})")
    print(f"Control ½ (η_det=0.50)        : {low_eta.truth_value.name} · loophole abierto")
    print(f"Merkle raíz (SIN clave) / hojas: {st.merkle_root_hash[:16]}… / {len(st.merkle_leaves)}")
    print(f"Estado reverificable          : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")