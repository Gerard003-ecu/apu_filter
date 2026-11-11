# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverProsecutorAgent  v14.0.0-PhD (reedición rigurosa)    ║
║ Ubicación: app/agents/wisdom/bool_novikov_thermal_exergy_weaver_prosecutor_agent.py           ║
║ Función  : Soberano agéntico FISCAL: instruye el proceso, controla la multiplicidad de cargos,║
║            sentencia con proporcionalidad y es la ÚNICA autoridad que autoriza y ejecuta.     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA PROCESAL DEL SOBERANO FISCAL
─────────────────────────────────────
(1) LA ANOMALÍA NO OTORGA INMUNIDAD. v13 exigía `is_psd` en Φ₁₂: un estado malformado quedaba
    FUERA del proceso. Aquí la integridad criptográfica es dura, pero la anomalía física se
    ADMITE, se sella el objeto crudo y se propaga ⊥. Corromper la evidencia agrava, no exime.
(2) ORIENT NO ACTÚA. El motor fiscal se ejecuta con un driver de SÓLO LECTURA y sin token:
    es puramente epistémico. La actuación es el ÚLTIMO commit de Fase 3, después de Φ₂₃.
    (v13 disparaba en Fase 2 y luego podía rechazar la compuerta: disparo sin certificado.)
(3) NADIE CONDENA POR ACUMULACIÓN. Cada canal de coste produce un p-value ANYTIME-VALID
    (martingala de razón de verosimilitud + desigualdad de Ville); la familia se resuelve con
    HOLM–BONFERRONI a α_family declarado, y el FWER se verifica por Monte Carlo.
(4) PROPORCIONALIDAD. Escalera WARN→DERATE→ISOLATE→TRIP. Una condena sólo estadística NUNCA
    alcanza el peldaño máximo: hace falta prueba estructural Y declaración de insuficiencia
    de medidas menores.
(5) AUTORIZACIÓN QUE HABILITA. El agente es la autoridad: emite/verifica el token HMAC ligado
    a {acta, época, peldaño, autorizador}, aplica ne bis in idem y anti-replay. Sin token:
    estado seguro + escalada, JAMÁS disparo. `force_veto` queda abolido.

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.4  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.6  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle
    P1 ⊂ P2 ⊂ ThermalProsecutorSovereignState; ejecución real = Kleisli observe>=>orient>=>decide.

FASE 1 — OBSERVE (instrucción preliminar)
  §1.1 Validación previa (hermiticidad, traza cruda, positividad, pureza, entropía) SIN excluir.
  §1.2 Normalización auditada + ISOMETRÍA DE BURES–WASSERSTEIN bajo unitario Haar.
  §1.3 Compromiso canónico de entradas y de la política de carga (α_family, β, escalera).
  §1.4 [TERMINAL] emit_phase1_certificate.

COMPUERTA Φ₁₂ — integridad dura (recálculo desde payload, HMAC, re-digest, época); ADMITE ⊥.

FASE 2 — ORIENT (instrucción probatoria, epistémica)
  §2.1 accept_phase1_certificate.
  §2.2 Motor fiscal v14 en modo epistémico (driver sólo lectura, sin token).
  §2.3 Verificación del acta del motor + contrafirma ligada al pasaporte principal + Merkle.
  §2.4 PLIEGO DE CARGOS: p-values de Ville + Holm–Bonferroni + verificación MC del FWER.
  §2.5 Cadencia de Poincaré ANTI-EVASIÓN: rotación diophantina, tres huecos (cota de ventana
       de evasión) y Kac (retardo esperado de detección).
  §2.6 [TERMINAL] emit_phase2_certificate.

COMPUERTA Φ₂₃ — integridad dura; física/epistémica ⇒ ⊥ forzado pero ADMITE a Fase 3.

FASE 3 — DECIDE & ACT (sentencia y ejecución autorizada)
  §3.1 accept_phase2_certificate.
  §3.2 Sentencia Ω₃: presunción de inocencia, Holm, núcleo ¬¬, functor contravariante.
  §3.3 Proporcionalidad + autoridad de autorización + ne bis in idem + anti-replay.
  §3.4 RSI³: Newton–Kantorovich, contracción de Banach, Kleisli, jerk medido.
  §3.5 Φ_sem con procedencia dimensional.
  §3.6 COMMIT de actuación (HAL medido, presupuesto físico declarado) — último paso.
  §3.7 [TERMINAL] emit_sovereign_state (Merkle + firma determinista + escalada).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `positron_hmac_valid: bool` → `authorization_token: str`; `force_veto` abolido (se rechaza).
• `rep.dirac_report` → `rep.dirac_crowbar`; `rep.banach_contraction_lip` → `rep.rsi3.lip_t3_bound`;
  `rep.jarzynski_kms_report` → `rep.phase_ii.jarzynski_kms`.
• Sellos deterministas (sin reloj); `crowbar_latency_ns` es despacho MEDIDO.
• KPI sin insumos declarados ⇒ `None` con `UNDECLARED_INPUT`.

LIMITACIONES DECLARADAS
───────────────────────
1. Los p-values de Ville son válidos bajo el modelo gaussiano declarado; colas pesadas o
   deriva invalidan la garantía y así se declara.
2. Holm controla FWER, no FDR; con muchos cargos correlacionados es conservador.
3. Kac–Slater se verifica para la rotación irracional (modelo ergódico); extrapolarlo a la
   dinámica del adversario es una HIPÓTESIS de modelado declarada.
4. 392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza.
5. La escalada se emite como dato firmado; el transporte es responsabilidad del llamador.

REFERENCIAS: Ville (1939); Wald (1945); Holm (1979); Benjamini–Hochberg (1995); Bures (1969);
Wasserstein; Uhlmann (1976); Spohn (1978); Kac (1947); Slater (1967); Steinhaus; Khinchin;
Kantorovich; Banach (1922); Mac Lane (Kleisli); Johnstone (núcleos); RFC 6962; CODATA 2022.
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


def _import_first(paths: Sequence[str]) -> Any:
    last: Optional[Exception] = None
    for p in paths:
        try:
            return __import__(p, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(f"No se pudo importar ninguno de: {paths}") from last


_pe = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_prosecutor_engine",
                     "bool_novikov_thermal_exergy_weaver_prosecutor_engine"))
_pri = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
                      "bool_novikov_thermal_exergy_weaver_engine"))

ProsecutorEngine = _pe.BoolNovikovThermalExergyWeaverProsecutorEngine
EngineProsecutionParameters = _pe.ProsecutionParameters
PrincipalEngine = _pri.BoolNovikovThermalExergyWeaverEngine
HeytingTruthValue = _pri.HeytingTruthValue          # Ω₃ CANÓNICO: no se duplica ni se invierte
Evidence = _pri.Evidence
Certified = _pri.Certified
kleisli_compose = _pri.kleisli_compose

__all__ = [
    "ProsecutorialGovernanceParameters", "ThermalProsecutorPhase1Certificate",
    "ThermalProsecutorPhase2Certificate", "ThermalProsecutorVisceralMetrics",
    "ThermalProsecutorSovereignState", "ThermalExergyProsecutorPhase1Observer",
    "ThermalExergyProsecutorPhase2Orientor", "ThermalExergyProsecutorPhase3Decider",
    "BoolNovikovThermalExergyWeaverProsecutorAgent", "ProsecutorialGovernanceError",
    "ProsecutorialGateIntegrityError", "AuthorizationError", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverProsecutorAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-PROSECUTOR-AGENT-VILLE-HOLM-BURES-WASSERSTEIN-KAC-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")
_HMAC_P1: Final[bytes] = b"PROSECUTOR-AGENT-P1-CUSTODY-v14|"
_HMAC_P2: Final[bytes] = b"PROSECUTOR-AGENT-P2-CHARGES-v14|"
_HMAC_STATE: Final[bytes] = b"PROSECUTOR-AGENT-SENTENCE-v14|"
_HMAC_AUTH: Final[bytes] = b"PROSECUTOR-AGENT-HUMAN-AUTHORIZATION-v14|"
_HMAC_ESC: Final[bytes] = b"PROSECUTOR-AGENT-ESCALATION-v14|"
_HMAC_COUNTER: Final[bytes] = b"PROSECUTOR-COUNTERSIGN-v14|"     # igual que el motor fiscal
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle, Bures–Wasserstein, Ville/Holm
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorialGovernanceError(Exception):
    """Error base de la gobernanza fiscal."""


class ProsecutorialGateIntegrityError(ProsecutorialGovernanceError):
    """Fallo DURO de integridad ⇒ fail-closed (sin actuación) y acta de emergencia."""


class AuthorizationError(ProsecutorialGovernanceError):
    """Actuación sin autorización válida, con replay de época o acta ya juzgada."""


def _array_digest(arr: NDArray) -> str:
    """Algoritmo idéntico al del motor: digests interoperables (ARR|shape|dtype<|bytes)."""
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
        raise ProsecutorialGovernanceError("Índice de hoja fuera de rango.")

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
    """Verificación RFC 6962 §2.1.3 (fn/sn)."""
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


def _psd_sqrt(a: NDArray) -> NDArray:
    vals, vecs = la.eigh(0.5 * (a + a.conj().T))
    return (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.conj().T


def bures_wasserstein_distance(a: NDArray, b: NDArray) -> float:
    r"""
    Distancia de Bures–Wasserstein sobre matrices PSD (no exige traza unitaria):
        d_BW²(A,B) = Tr A + Tr B − 2 Tr√(√A B √A).
    Para estados normalizados coincide con la de Bures: d_BW² = 2 − 2√F.
    Cálculo estable por eigh del núcleo hermítico √A B √A (sin `sqrtm`).
    """
    sa = _psd_sqrt(a)
    core = sa @ b @ sa
    vals = la.eigvalsh(0.5 * (core + core.conj().T))
    cross = float(np.sum(np.sqrt(np.maximum(vals, 0.0))))
    val = float(np.real(np.trace(a)) + np.real(np.trace(b))) - 2.0 * cross
    return math.sqrt(max(0.0, val))


def _haar_unitary(dim: int, rng: np.random.Generator) -> NDArray:
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * np.where(np.abs(d) > 0.0, d / np.maximum(np.abs(d), _EPS), 1.0)


def ville_anytime_pvalue(stream: NDArray, mu0: float, mu1: float, sigma: float) -> Tuple[float, float]:
    r"""
    p-value ANYTIME-VALID por la desigualdad de Ville.
    Bajo H₀, exp(Λ_n) es una martingala no negativa con E=1 (razón de verosimilitud gaussiana):
        P(∃n: exp(Λ_n) ≥ 1/α) ≤ α  ⇒  p = min(1, exp(−max_n Λ_n)) es válido en cualquier parada.
    Devuelve (p, max_n Λ_n). No requiere fijar el tamaño de muestra de antemano.
    """
    x = np.asarray(stream, dtype=np.float64).reshape(-1)
    z = (mu1 - mu0) / sigma ** 2 * (x - 0.5 * (mu0 + mu1))
    lam_max = float(np.max(np.cumsum(z))) if x.size else 0.0
    return float(min(1.0, math.exp(-max(lam_max, 0.0)))), lam_max


def holm_bonferroni(pvals: Sequence[float], alpha_family: float) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    r"""
    Procedimiento step-down de Holm: ordena p₍₀₎≤…≤p₍ₘ₋₁₎ y rechaza mientras
    p₍ₖ₎ ≤ α/(m−k). Controla el FWER ≤ α sin hipótesis de independencia.
    p ajustados: p̃₍ₖ₎ = max_{j≤k} min(1, (m−j)·p₍ⱼ₎) (monótonos).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return (), ()
    order = np.argsort(p, kind="stable")
    rejected: List[int] = []
    adj = np.ones(m, dtype=np.float64)
    running = 0.0
    stopped = False
    for k, idx in enumerate(order):
        running = max(running, min(1.0, (m - k) * float(p[idx])))
        adj[idx] = running
        if not stopped and p[idx] <= alpha_family / (m - k):
            rejected.append(int(idx))
        else:
            stopped = True
    return tuple(rejected), tuple(float(v) for v in adj)


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
        pn, qn = int(a) * p + p_prev, int(a) * q + q_prev
        if qn > max_q:
            break
        p_prev, q_prev, p, q = p, q, pn, qn
        out.append((p, q))
    return tuple(out)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProsecutorialGovernanceParameters:
    """Política de carga y presupuestos DECLARADOS; su digest se compromete en P1."""

    alpha_family: float = 0.01                 # FWER de la familia de cargos estadísticos
    beta_sprt: float = 0.10
    cost_mu0: float = 0.0
    cost_mu1: float = 0.5
    cost_sigma: float = 1.0
    mc_paths: int = 1500
    mc_max_len: int = 300
    psd_floor: float = 1e-12
    beta_thermo: float = 1.0
    work_mean: float = 5.0
    delta_f: float = 4.8
    ball_r: float = 1.0                        # v13 del motor condenaba por defecto (0.25>0.20)
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    melnikov_epsilon: float = 0.01
    require_human_authorization: bool = True
    lesser_measures_insufficient: bool = False
    cadence_points: int = 64
    cadence_rotation: float = _GOLDEN
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    logic_deadline_ns: float = 392.15
    gpio_write_ns: float = 25.0
    gate_driver_ns: float = 150.0
    scr_turn_on_ns: float = 1500.0
    hil_firmware_isr_ns: Optional[float] = None
    exergy_reference_w: float = 1.0
    exergy_price_per_joule: Optional[float] = None
    capital_base: Optional[float] = None
    kv_bytes_before: Optional[int] = None
    kv_bytes_after: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha_family < 0.5 and 0.0 < self.beta_sprt < 0.5):
            raise ProsecutorialGovernanceError("α_family, β ∈ (0,0.5).")
        if self.cost_sigma <= 0.0 or self.cost_mu1 <= self.cost_mu0:
            raise ProsecutorialGovernanceError("σ>0 y μ₁>μ₀ (H1 separable de H0).")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0:
            raise ProsecutorialGovernanceError("Radios positivos.")
        if not (0.0 < self.recurrence_mu < 1.0 and 0.0 < self.cadence_rotation < 1.0):
            raise ProsecutorialGovernanceError("μ∈(0,1), α∈(0,1) para Kac–Slater.")
        if self.mc_paths < 200 or self.mc_max_len < 50:
            raise ProsecutorialGovernanceError("Monte Carlo insuficiente para certificar tasas.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)

    def to_engine_parameters(self) -> Any:
        """Proyección al contrato del motor fiscal v14 (sin autorizar actuación)."""
        return EngineProsecutionParameters(
            alpha_false_conviction=min(self.alpha_family, 0.05), beta_false_acquittal=self.beta_sprt,
            cost_mu0=self.cost_mu0, cost_mu1=self.cost_mu1, cost_sigma=self.cost_sigma,
            psd_floor=self.psd_floor, beta_thermo=self.beta_thermo, work_mean=self.work_mean,
            delta_f=self.delta_f, ball_r=self.ball_r, cylinder_r=self.cylinder_r,
            melnikov_omega=self.melnikov_omega, melnikov_gamma=self.melnikov_gamma,
            melnikov_delta=self.melnikov_delta, melnikov_epsilon=self.melnikov_epsilon,
            require_human_authorization=True,          # el motor NUNCA autoemite token
            lesser_measures_insufficient=self.lesser_measures_insufficient,
            logic_deadline_ns=self.logic_deadline_ns, gpio_write_ns=self.gpio_write_ns,
            gate_driver_ns=self.gate_driver_ns, scr_turn_on_ns=self.scr_turn_on_ns,
            hil_firmware_isr_ns=self.hil_firmware_isr_ns,
        )


@dataclass(frozen=True)
class ThermalProsecutorPhase1Certificate:
    """P1 — Custodia preliminar. La anomalía se registra, NO se excluye (reloj fuera del sello)."""

    timestamp: float
    tr_rho: float
    tr_rho_normalized: float
    normalization_factor: float
    is_psd: bool
    malformation_detected: bool
    substitute_declared: bool
    hermiticity_residual: float
    purity: float
    von_neumann_entropy: float
    bures_wasserstein_to_gibbs: float
    bw_isometry_residual: float
    bw_uhlmann_consistency_residual: float
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    cost_streams_digest: str
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
class ChargeSheetReport:
    """§2.4 — Pliego de cargos con control de multiplicidad (Ville + Holm)."""

    n_charges: int
    pvalues_anytime: Tuple[float, ...]
    pvalues_adjusted: Tuple[float, ...]
    rejected_indices: Tuple[int, ...]
    alpha_family: float
    structural_charges: Tuple[str, ...]
    statistical_convictions: int
    mc_ville_validity_rate: float
    mc_fwer: float
    mc_rates_within_declared: bool
    max_log_likelihood: float
    model_note: str


@dataclass(frozen=True)
class AntiEvasionCadenceReport:
    """§2.5 — Cadencia no resonante: cota de ventana de evasión y retardo de detección."""

    rotation_number: float
    diophantine_gamma: float
    distinct_gap_count: int
    max_gap: float
    coverage_bound: float
    three_gap_identity_residual: float
    low_order_resonance: bool
    kac_expected_detection_delay: float
    kac_empirical_delay: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    three_gap_ok: bool
    coverage_ok: bool
    slater_ok: bool


@dataclass(frozen=True)
class ThermalProsecutorPhase2Certificate:
    """P2 — Instrucción probatoria epistémica: anida P1 y el acta del motor (sin actuación)."""

    timestamp: float
    phase1: ThermalProsecutorPhase1Certificate
    engine_report: Optional[Any]
    engine_invoked: bool
    engine_acta_verified: bool
    engine_merkle_recomputed_ok: bool
    engine_countersignature_ok: bool
    engine_actuation_dispatched: bool
    engine_truth: HeytingTruthValue
    charge_sheet: ChargeSheetReport
    cadence: AntiEvasionCadenceReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    epoch: int
    chain_hash: str
    hmac_signature: str

    def payload(self) -> Dict[str, Any]:
        rep = self.engine_report
        return {
            "phase1_chain_hash": self.phase1.chain_hash, "engine_invoked": self.engine_invoked,
            "engine_passport": getattr(rep, "passport_hash", ""),
            "engine_acta": getattr(getattr(rep, "sentence", None), "indictment_digest", ""),
            "engine_merkle_root": getattr(rep, "merkle_root_hash", ""),
            "engine_countersignature": getattr(rep, "countersignature", ""),
            "engine_acta_verified": self.engine_acta_verified,
            "engine_merkle_recomputed_ok": self.engine_merkle_recomputed_ok,
            "engine_countersignature_ok": self.engine_countersignature_ok,
            "engine_actuation_dispatched": self.engine_actuation_dispatched,
            "engine_truth": self.engine_truth,
            "charge_sheet": self.charge_sheet, "cadence": self.cadence,
            "evidences": self.evidences, "truth_value": self.truth_value, "epoch": self.epoch,
        }


@dataclass(frozen=True)
class ProsecutorialGateReceipt:
    gate: str
    payload_digest: str
    chain_recomputed_ok: bool
    signature_ok: bool
    binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class ThermalProsecutorVisceralMetrics:
    """§3.5 — Φ_sem con procedencia: cobertura y presupuestos medidos; finanzas declaradas."""

    charge_coverage_pct: float
    proven_charges: int
    family_wise_error_budget_pct: float
    evasion_window_bound_pct: float
    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    irreversible_work_status: str
    due_process_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool


@dataclass(frozen=True)
class ProsecutorialRSI3Report:
    """§3.4 — RSI³ del fiscal: Kantorovich + Banach + Kleisli + núcleo + jerk medido."""

    severity_threshold_calibrated: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_quadratic_ratio_residual: float
    policy_lipschitz_q: float
    lip_t3_bound: float
    lip_t3_empirical: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_policy_error: float
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool
    kleisli_monad_laws_ok: bool
    nucleus_axioms_ok: bool
    action_functor_ok: bool


@dataclass(frozen=True)
class ActuationAuthorizationReport:
    """§3.3+§3.6 — Autoridad de autorización y COMMIT de actuación."""

    ladder_rung: int
    action: str
    indictment_digest: str
    epoch: int
    authorization_token_verified: bool
    authorization_self_issued: bool
    ne_bis_in_idem_ok: bool
    anti_replay_ok: bool
    actuation_requested: bool
    actuation_authorized: bool
    actuation_dispatched: bool
    safe_state_applied: bool
    driver_name: str
    driver_is_physical: bool
    dispatch_median_ns: float
    dispatch_p99_ns: float
    trip_dispatch_ns: float
    logic_deadline_ns: float
    physical_budget_ns: float
    physical_deadline_feasible: bool
    firmware_isr_verified: bool
    note: str


@dataclass(frozen=True)
class ProsecutorialEscalation:
    epoch: int
    truth_value: HeytingTruthValue
    principal_truth: Optional[HeytingTruthValue]
    witness_truth: Optional[HeytingTruthValue]
    reason: str
    merkle_root_hash: str
    signature: str
    delivered: bool


@dataclass(frozen=True)
class ThermalProsecutorSovereignState:
    """Estado soberano fiscal final (nombres heredados conservados)."""

    timestamp: float
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_ns: float
    attestation_withheld: bool
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    sentence_leaf_index: int
    sentence_inclusion_proof: Tuple[Tuple[str, bool], ...]
    visceral_metrics: ThermalProsecutorVisceralMetrics
    rsi3: ProsecutorialRSI3Report
    authorization: ActuationAuthorizationReport
    escalation: Optional[ProsecutorialEscalation]
    phase2: ThermalProsecutorPhase2Certificate
    evidences: Tuple[Any, ...]
    kleisli_ledger: Tuple[str, ...]
    epoch: int
    state_signature: str
    error: Optional[str]
    execution_time_ms: float

    def payload(self) -> Dict[str, Any]:
        return {
            "truth_value": self.truth_value, "boolean_shadow": self.boolean_shadow,
            "crowbar_triggered": self.crowbar_triggered,
            "attestation_withheld": self.attestation_withheld,
            "merkle_root_hash": self.merkle_root_hash,
            "sentence_leaf_index": self.sentence_leaf_index,
            "sentence_inclusion_proof": self.sentence_inclusion_proof,
            "kleisli_ledger": self.kleisli_ledger, "epoch": self.epoch, "error": self.error,
            "phase2_chain_hash": self.phase2.chain_hash,
            "phase1_chain_hash": self.phase2.phase1.chain_hash,
            "authorization": {
                "rung": self.authorization.ladder_rung, "action": self.authorization.action,
                "acta": self.authorization.indictment_digest,
                "authorized": self.authorization.actuation_authorized,
                "dispatched": self.authorization.actuation_dispatched,
                "token_verified": self.authorization.authorization_token_verified,
                "ne_bis": self.authorization.ne_bis_in_idem_ok,
                "anti_replay": self.authorization.anti_replay_ok,
                "safe_state": self.authorization.safe_state_applied,
            },
            "escalation_signature": self.escalation.signature if self.escalation else None,
            "merkle_leaf_count": len(self.merkle_leaves),
        }

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)


class CrowbarDriver(Protocol):
    name: str
    is_physical: bool

    def dry_run(self) -> int: ...
    def arm(self) -> None: ...
    def trip(self, token: str) -> int: ...
    def safe_state(self) -> None: ...


class ReadOnlyCrowbarDriver:
    """
    Driver de SÓLO LECTURA inyectado al motor en Fase 2: garantiza estructuralmente que
    ORIENT no puede tocar hardware (corrige el disparo en Fase 2 de v13).
    """

    name = "READ_ONLY (ORIENT epistémico)"
    is_physical = False

    def __init__(self) -> None:
        self.refused = 0

    def dry_run(self) -> int:
        t0 = time.perf_counter_ns()
        _ = self.refused
        return int(time.perf_counter_ns() - t0)

    def arm(self) -> None:
        self.refused += 1
        raise AuthorizationError("ORIENT es epistémico: no puede armar el crowbar.")

    def trip(self, token: str) -> int:
        self.refused += 1
        raise AuthorizationError("ORIENT es epistémico: no puede disparar.")

    def safe_state(self) -> None:
        self.refused += 1   # registro; sin efecto físico


class SimulatedCrowbarDriver:
    """Driver SIMULADO del agente (no físico). Mide despacho real; exige token válido."""

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
            raise AuthorizationError("Token de autorización inválido: disparo rechazado.")
        if not self._armed:
            raise AuthorizationError("Crowbar no armado: disparo rechazado.")
        t0 = time.perf_counter_ns()
        self._latched = True
        self._armed = False
        return int(time.perf_counter_ns() - t0)

    def safe_state(self) -> None:
        self._latched = True
        self._armed = False


class ProsecutorialSigner:
    """HMAC-SHA256 con separación de dominio y comparación en tiempo constante."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_PROSECUTOR_AGENT_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave fiscal efímera: sentencias no verificables entre procesos.")
        if len(secret_key) < 32:
            raise ProsecutorialGovernanceError("La clave fiscal requiere ≥ 256 bits.")
        self._key = bytes(secret_key)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def fingerprint(self) -> str:
        return hmac.new(self._key, b"PROSECUTOR-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]

    def sign(self, domain: bytes, payload: Any) -> str:
        return hmac.new(self._key, domain + _canonical(payload), hashlib.sha256).hexdigest()

    def verify(self, domain: bytes, payload: Any, signature: str) -> bool:
        return hmac.compare_digest(self.sign(domain, payload), str(signature))


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE — custodia preliminar e isometría de Bures–Wasserstein
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyProsecutorPhase1Observer:
    r"""
    FASE 1 OODA fiscal. Observa sin excluir: la malformación es AGRAVANTE, no salvoconducto.
    Ejecuta la isometría de Bures–Wasserstein que el docstring de v13 prometía y no calculaba.
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE-CUSTODIA"

    def __init__(self, signer: ProsecutorialSigner, *, seed: int = 20_261_117) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa (registra, no excluye) ───────────
    def inspect_state(self, rho_raw: NDArray, dim: int) -> Dict[str, Any]:
        r = np.asarray(rho_raw, dtype=np.complex128)
        malformed = r.ndim != 2 or r.shape != (dim, dim) or not np.all(np.isfinite(r))
        if malformed:
            return {"malformed": True, "hermiticity_residual": float("inf"), "trace_raw": 0.0,
                    "substitute": np.eye(dim, dtype=np.complex128) / dim}
        fro = float(la.norm(r, ord="fro")) + _EPS
        herm = float(la.norm(r - r.conj().T, ord="fro")) / fro
        tr = complex(np.trace(r))
        bad_trace = abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0
        substitute = np.eye(dim, dtype=np.complex128) / dim if bad_trace else None
        return {"malformed": bool(bad_trace), "hermiticity_residual": herm,
                "trace_raw": float(tr.real), "substitute": substitute}

    # ─────────── §1.2 Normalización auditada + isometría BW ───────────
    @staticmethod
    def normalize_audited(rho_raw: NDArray, trace_raw: float) -> Tuple[NDArray, float]:
        r = np.asarray(rho_raw, dtype=np.complex128)
        r = 0.5 * (r + r.conj().T)
        return r / float(trace_raw), 1.0 / float(trace_raw)

    def bures_wasserstein_isometry(self, rho: NDArray, sigma: NDArray) -> Tuple[float, float, float]:
        """
        (d_BW(ρ,σ), residuo de invariancia bajo unitario Haar, consistencia con 2−2√F).
        La invariancia d_BW(UρU†,UσU†)=d_BW(ρ,σ) es la isometría que v13 sólo enunciaba.
        """
        d = bures_wasserstein_distance(rho, sigma)
        u = _haar_unitary(rho.shape[0], self._rng)
        d_moved = bures_wasserstein_distance(u @ rho @ u.conj().T, u @ sigma @ u.conj().T)
        sr, ss = _psd_sqrt(rho), _psd_sqrt(sigma)
        fid = float(np.sum(la.svdvals(sr @ ss))) ** 2
        consistency = abs(d * d - (2.0 - 2.0 * math.sqrt(min(1.0, max(0.0, fid)))))
        return d, abs(d_moved - d), consistency

    # ─────────── §1.4 [TERMINAL FASE 1] → Φ₁₂ → Fase 2 §2.1 ───────────
    def emit_phase1_certificate(
        self, rho_raw: NDArray, apu_cost_matrix: NDArray, exponents: NDArray,
        cost_streams: Sequence[NDArray], params: ProsecutorialGovernanceParameters, *,
        epoch: int, principal_passport: str = "", gibbs_reference: Optional[NDArray] = None,
    ) -> ThermalProsecutorPhase1Certificate:
        """MÉTODO TERMINAL DE LA FASE 1: custodia, isometría BW y compromiso de la política."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise ProsecutorialGovernanceError("K debe ser cuadrada.")
        dim = int(k_mat.shape[0])
        insp = self.inspect_state(rho_raw, dim)
        substitute = insp["substitute"] is not None
        if substitute:
            rho = np.asarray(insp["substitute"], dtype=np.complex128)
            factor, trace_norm = 0.0, 1.0
        else:
            rho, factor = self.normalize_audited(rho_raw, insp["trace_raw"])
            trace_norm = float(np.real(np.trace(rho)))
        eigs = la.eigvalsh(rho)
        purity = float(np.real(np.trace(rho @ rho)))
        probs = np.clip(eigs, 0.0, 1.0)
        entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        is_psd = bool(float(np.min(eigs)) >= -1e-10 and not substitute)
        sigma = gibbs_reference if gibbs_reference is not None else np.eye(dim, dtype=np.complex128) / dim
        d_bw, iso_res, cons_res = self.bures_wasserstein_isometry(rho, sigma)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("AP1.estado_no_malformado", not insp["malformed"], hard=True,
                             note="la malformación se ADMITE y agrava: no otorga inmunidad"),
            Evidence("AP1.hermiticidad", insp["hermiticity_residual"] if math.isfinite(
                insp["hermiticity_residual"]) else 1.0, 1e-10, hard=True),
            Evidence.boolean("AP1.positividad", is_psd, hard=True),
            Evidence("AP1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("AP1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence("AP1.bw_isometria_haar", iso_res, 1e-11, hard=True),
            Evidence("AP1.bw_uhlmann_consistencia", cons_res, 1e-10, hard=True),
            Evidence.boolean("AP1.sustituto_declarado", True,
                             note=f"sustituto declarado para poder instruir: {substitute}"),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        core = {
            "tr_rho": float(insp["trace_raw"]), "tr_rho_normalized": trace_norm,
            "normalization_factor": factor, "is_psd": is_psd,
            "malformation_detected": bool(insp["malformed"]), "substitute_declared": bool(substitute),
            "hermiticity_residual": float(insp["hermiticity_residual"]) if math.isfinite(
                insp["hermiticity_residual"]) else float("inf"),
            "purity": purity, "von_neumann_entropy": entropy,
            "bures_wasserstein_to_gibbs": d_bw, "bw_isometry_residual": iso_res,
            "bw_uhlmann_consistency_residual": cons_res,
            "rho_digest": _array_digest(np.asarray(rho_raw, dtype=np.complex128)),
            "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "cost_streams_digest": _sha256(b"|".join(
                _array_digest(np.asarray(s, dtype=np.float64)).encode("ascii") for s in cost_streams)),
            "params_digest": params.digest(), "principal_passport": principal_passport,
            "agent_version": __version__, "engine_version": _pe.__version__, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain = _sha256(_canonical(core))
        cert = ThermalProsecutorPhase1Certificate(
            timestamp=time.time(), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P1, core), **core
        )
        logger.info("AP1 Ω₃=%s · malformado=%s · d_BW=%.4e · isometría=%.2e",
                    truth.name, insp["malformed"], d_bw, iso_res)
        return cert  # ──► Φ₁₂ ──► Fase 2 §2.1

    def execute_phase1(self, *a: Any, **k: Any) -> ThermalProsecutorPhase1Certificate:
        """Alias heredado (v13) del terminal §1.4."""
        return self.emit_phase1_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT — instrucción probatoria EPISTÉMICA (Ville + Holm + Poincaré)
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyProsecutorPhase2Orientor:
    r"""
    FASE 2 OODA fiscal. Instruye el proceso SIN actuar: el motor recibe un driver de sólo
    lectura y ningún token. Su último método alimenta el primero de Fase 3 vía Φ₂₃.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT-INSTRUCCION"

    def __init__(self, engine: Any, signer: ProsecutorialSigner, *, seed: int = 20_261_117) -> None:
        self._engine = engine
        self._signer = signer
        self._seed = int(seed)

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self, p1: ThermalProsecutorPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        cost_streams: Sequence[NDArray], params: ProsecutorialGovernanceParameters, *, expected_epoch: int,
    ) -> ProsecutorialGateReceipt:
        r"""
        Φ₁₂ — Corrige la inversión de v13 (`HMAC ∧ is_psd`): la INTEGRIDAD es dura, la
        ANOMALÍA FÍSICA se ADMITE. Rechazar un estado no PSD equivalía a conceder inmunidad
        al corruptor. Aquí se recalcula el hash desde el payload, se verifica el HMAC y se
        re-digestan los arrays reales que cruzan la compuerta (anti-TOCTOU).
        """
        payload = p1.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p1.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P1, payload, p1.hmac_signature)
        streams_digest = _sha256(b"|".join(
            _array_digest(np.asarray(s, dtype=np.float64)).encode("ascii") for s in cost_streams))
        binding_ok = (
            hmac.compare_digest(p1.kernel_digest, _array_digest(np.asarray(apu_cost_matrix, dtype=np.float64)))
            and hmac.compare_digest(p1.exponents_digest, _array_digest(np.asarray(exponents, dtype=np.float64)))
            and hmac.compare_digest(p1.cost_streams_digest, streams_digest)
            and hmac.compare_digest(p1.params_digest, params.digest())
            and p1.epoch == expected_epoch and p1.agent_version == __version__
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise ProsecutorialGateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return ProsecutorialGateReceipt("Φ₁₂", recomputed, chain_ok, sig_ok, binding_ok, True,
                                        not p1.is_psd, "anomalía ADMITIDA: se instruye el proceso, no se exime")

    def accept_phase1_certificate(self, p1: ThermalProsecutorPhase1Certificate, apu_cost_matrix: NDArray,
                                  exponents: NDArray, cost_streams: Sequence[NDArray],
                                  params: ProsecutorialGovernanceParameters, *,
                                  expected_epoch: int) -> ProsecutorialGateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.4."""
        return self.continue_to_phase2_gate(p1, apu_cost_matrix, exponents, cost_streams, params,
                                            expected_epoch=expected_epoch)

    # ─────────── §2.3 Verificación del acta del motor y de su contrafirma ───────────
    def verify_engine_acta(self, report: Any, principal_passport: str) -> Tuple[bool, bool, bool]:
        verified = bool(self._engine.verify_prosecutor_report(report))
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        root_ok = hmac.compare_digest(merkle_root(leaves), report.merkle_root_hash)
        idx = int(report.acta_leaf_index)
        root_ok = root_ok and verify_merkle(leaves[idx], idx, len(leaves),
                                            report.acta_inclusion_proof, report.merkle_root_hash)
        expected = hmac.new(self._signer.key_bytes,
                            _HMAC_COUNTER + principal_passport.encode("ascii")
                            + bytes.fromhex(report.merkle_root_hash), hashlib.sha256).hexdigest()
        return verified, root_ok, hmac.compare_digest(expected, report.countersignature)

    # ─────────── §2.4 Pliego de cargos: Ville + Holm + verificación MC ───────────
    def charge_sheet(self, cost_streams: Sequence[NDArray], params: ProsecutorialGovernanceParameters,
                     structural_charges: Sequence[str]) -> ChargeSheetReport:
        r"""
        Un cargo estadístico por canal. Cada p-value es ANYTIME-VALID (Ville), de modo que
        mirar los datos repetidamente no infla el error. La familia se resuelve con HOLM a
        α_family, controlando FWER sin hipótesis de independencia. Los cargos ESTRUCTURALES
        (identidades fallidas) son pruebas, no hipótesis: no entran en la multiplicidad.

        Verificación MC: (i) validez de Ville bajo H₀: P(p ≤ α) ≤ α; (ii) FWER de Holm bajo
        la hipótesis nula global con m canales: P(algún rechazo) ≤ α_family.
        """
        pvals: List[float] = []
        lam_max = 0.0
        for s in cost_streams:
            p, lm = ville_anytime_pvalue(np.asarray(s, dtype=np.float64),
                                         params.cost_mu0, params.cost_mu1, params.cost_sigma)
            pvals.append(p); lam_max = max(lam_max, lm)
        rejected, adjusted = holm_bonferroni(pvals, params.alpha_family)

        rng = np.random.default_rng(self._seed + 91)
        m, paths, length = max(1, len(pvals)), int(params.mc_paths), int(params.mc_max_len)
        mu0, mu1, sg = params.cost_mu0, params.cost_mu1, params.cost_sigma

        def null_pvalues(n_ch: int) -> NDArray:
            out = np.ones((paths, n_ch), dtype=np.float64)
            for ch in range(n_ch):
                x = rng.normal(mu0, sg, size=(paths, length))
                z = (mu1 - mu0) / sg ** 2 * (x - 0.5 * (mu0 + mu1))
                out[:, ch] = np.minimum(1.0, np.exp(-np.maximum(np.max(np.cumsum(z, axis=1), axis=1), 0.0)))
            return out

        single = null_pvalues(1)[:, 0]
        ville_rate = float(np.mean(single <= params.alpha_family))
        family = null_pvalues(m)
        any_reject = np.zeros(paths, dtype=bool)
        for i in range(paths):
            rej, _ = holm_bonferroni(family[i], params.alpha_family)
            any_reject[i] = len(rej) > 0
        fwer = float(np.mean(any_reject))
        slack = 3.0 * math.sqrt(params.alpha_family * (1 - params.alpha_family) / paths)
        within = bool(ville_rate <= params.alpha_family + slack and fwer <= params.alpha_family + slack)
        return ChargeSheetReport(
            n_charges=len(pvals), pvalues_anytime=tuple(pvals), pvalues_adjusted=adjusted,
            rejected_indices=rejected, alpha_family=float(params.alpha_family),
            structural_charges=tuple(structural_charges), statistical_convictions=len(rejected),
            mc_ville_validity_rate=ville_rate, mc_fwer=fwer, mc_rates_within_declared=within,
            max_log_likelihood=float(lam_max),
            model_note="Ville/Holm válidos bajo el modelo gaussiano declarado; colas pesadas o deriva los invalidan.",
        )

    # ─────────── §2.5 Cadencia anti-evasión (Poincaré, tres huecos, Kac) ───────────
    @staticmethod
    def anti_evasion_cadence(params: ProsecutorialGovernanceParameters) -> AntiEvasionCadenceReport:
        r"""
        Un adversario racional sincroniza el fraude con el calendario de auditoría. La
        cadencia cuasi-periódica con número de rotación noble α elimina la resonancia:
          • TEOREMA DE LOS TRES HUECOS (Steinhaus): los huecos de {nα mod 1} toman ≤3 valores
            y el mayor es la suma de los otros dos ⇒ cota explícita de VENTANA DE EVASIÓN.
          • Cota de cobertura con convergentes q_{k−1},q_k ≤ N: max_hueco ≤ 1/q_{k−1}+1/q_k.
          • LEMA DE KAC: E[τ_A]=1/μ(A) ⇒ retardo esperado de detección de un régimen recurrente.
        """
        alpha, n_pts = float(params.cadence_rotation), int(params.cadence_points)
        convs = _convergents(alpha, n_pts)
        gamma = min(q * q * abs(alpha - p / q) for p, q in convs if q > 0)
        low_res = any(abs(q * alpha - p) < 1e-6 for p, q in convs if q <= 12)
        pts = np.sort(np.mod(np.arange(n_pts) * alpha, 1.0))
        gaps = np.diff(np.concatenate([pts, pts[:1] + 1.0]))
        distinct = np.unique(np.round(gaps, 9))
        ident = float(abs(distinct[2] - distinct[0] - distinct[1])) if distinct.size == 3 else (
            0.0 if distinct.size <= 3 else float("inf"))
        q_prev, q_cur = (convs[-2][1], convs[-1][1]) if len(convs) >= 2 else (1, convs[-1][1])
        bound = 1.0 / max(1, q_prev) + 1.0 / max(1, q_cur)

        mu, n_s = float(params.recurrence_mu), int(params.recurrence_samples)
        x = (np.arange(n_s) + 0.5) / n_s * mu
        times = np.zeros(n_s, dtype=np.int64)
        active = np.ones(n_s, dtype=bool)
        for step in range(1, int(max(1000, 100.0 / mu)) + 1):
            x[active] = np.mod(x[active] + alpha, 1.0)
            hit = active & (x < mu)
            times[hit] = step
            active[hit] = False
            if not active.any():
                break
        if active.any():
            raise ProsecutorialGovernanceError("Kac–Slater: retorno no alcanzado (α racional o μ degenerado).")
        uniq = tuple(int(v) for v in np.unique(times))
        expected = 1.0 / mu
        empirical = float(np.mean(times))
        return AntiEvasionCadenceReport(
            rotation_number=alpha, diophantine_gamma=float(gamma),
            distinct_gap_count=int(distinct.size), max_gap=float(np.max(gaps)), coverage_bound=float(bound),
            three_gap_identity_residual=ident, low_order_resonance=bool(low_res),
            kac_expected_detection_delay=expected, kac_empirical_delay=empirical,
            kac_relative_residual=abs(empirical - expected) / expected, slater_return_times=uniq,
            three_gap_ok=bool(distinct.size <= 3 and ident <= 1e-8),
            coverage_ok=bool(float(np.max(gaps)) <= bound + 1e-12), slater_ok=bool(len(uniq) <= 3),
        )

    # ─────────── §2.6 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self, p1: ThermalProsecutorPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        rho: NDArray, cost_streams: Sequence[NDArray], params: ProsecutorialGovernanceParameters, *,
        expected_epoch: int, principal_report: Any = None, witness_report: Any = None,
    ) -> ThermalProsecutorPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: instruye, verifica el acta y NO actúa."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, cost_streams, params,
                                       expected_epoch=expected_epoch)
        passport = getattr(principal_report, "passport_hmac", "")
        engine_report = self._engine.execute_thermal_prosecutor_audit(
            apu_cost_matrix, np.asarray(exponents, dtype=np.float64), rho,
            cost_stream=np.asarray(cost_streams[0], dtype=np.float64) if cost_streams else None,
            principal_report=principal_report, witness_report=witness_report,
            authorization_token=None,                       # ← Orient NUNCA autoriza
            params=params.to_engine_parameters(),
        )
        acta_ok, root_ok, counter_ok = self.verify_engine_acta(engine_report, passport)
        dispatched = bool(engine_report.dirac_crowbar.actuation_dispatched)

        structural: List[str] = []
        if p1.malformation_detected:
            structural.append("estado_malformado")
        if not p1.is_psd:
            structural.append("rho_no_PSD")
        if engine_report.phase_ii.geometry.gromov_obstruction:
            structural.append("obstruccion_de_gromov")
        if engine_report.phase_ii.geometry.melnikov_chaos:
            structural.append("caos_homoclinico")
        if any((not e.passed) and e.hard for e in
               list(engine_report.phase_ii.phase_i.evidences) + list(engine_report.phase_ii.evidences)):
            structural.append("identidad_estructural_fallida")
        sheet = self.charge_sheet(cost_streams, params, structural)
        cadence = self.anti_evasion_cadence(params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("AP2.acta_motor_verificada", acta_ok, hard=True),
            Evidence.boolean("AP2.merkle_motor_recalculado", root_ok, hard=True),
            Evidence.boolean("AP2.contrafirma_ligada_al_principal", counter_ok, hard=True),
            Evidence.boolean("AP2.orient_no_actuo", not dispatched, hard=True,
                             note="arquitectura: la actuación es el último commit de Fase 3"),
            Evidence.boolean("AP2.tasas_ville_holm", sheet.mc_rates_within_declared),
            Evidence("AP2.kac_retardo_deteccion", cadence.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("AP2.tres_huecos", cadence.three_gap_ok, hard=True),
            Evidence.boolean("AP2.cobertura_anti_evasion", cadence.coverage_ok, hard=True),
            Evidence.boolean("AP2.slater_tres_retornos", cadence.slater_ok, hard=True),
            Evidence.boolean("AP2.sin_resonancia_bajo_orden", not cadence.low_order_resonance),
        )
        truth = HeytingTruthValue.infimum(
            [p1.truth_value, engine_report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": True,
            "engine_passport": engine_report.passport_hash,
            "engine_acta": engine_report.sentence.indictment_digest,
            "engine_merkle_root": engine_report.merkle_root_hash,
            "engine_countersignature": engine_report.countersignature,
            "engine_acta_verified": acta_ok, "engine_merkle_recomputed_ok": root_ok,
            "engine_countersignature_ok": counter_ok, "engine_actuation_dispatched": dispatched,
            "engine_truth": engine_report.truth_value, "charge_sheet": sheet, "cadence": cadence,
            "evidences": evidences, "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain = _sha256(_canonical(core))
        cert = ThermalProsecutorPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=engine_report, engine_invoked=True,
            engine_acta_verified=acta_ok, engine_merkle_recomputed_ok=root_ok,
            engine_countersignature_ok=counter_ok, engine_actuation_dispatched=dispatched,
            engine_truth=engine_report.truth_value, charge_sheet=sheet, cadence=cadence,
            evidences=evidences, truth_value=truth, epoch=int(expected_epoch), chain_hash=chain,
            hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        logger.info("AP2 Ω₃=%s · cargos=%d (estad. probados=%d, estructurales=%d) · FWER=%.4f",
                    truth.name, sheet.n_charges, sheet.statistical_convictions,
                    len(structural), sheet.mc_fwer)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1

    def execute_phase2(self, *a: Any, **k: Any) -> ThermalProsecutorPhase2Certificate:
        """Alias heredado (v13) del terminal §2.6."""
        return self.emit_phase2_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ACT — sentencia, autorización y COMMIT de actuación
# ══════════════════════════════════════════════════════════════════════════════════════
class ThermalExergyProsecutorPhase3Decider:
    r"""
    FASE 3 OODA fiscal. Única autoridad que autoriza y ejecuta. La actuación es el último
    paso, después de todas las compuertas, con token ligado al acta y presupuesto declarado.
    """

    PHASE_LABEL: Final[str] = "3·SENTENCIA-COMMIT"
    _SEVERITY: Final[Dict[str, int]] = {"WARN": 0, "DERATE": 1, "ISOLATE": 2, "TRIP": 3}

    def __init__(self, signer: ProsecutorialSigner, driver: CrowbarDriver) -> None:
        self._signer = signer
        self._driver = driver
        self._judged_actas: set = set()
        self._last_epoch = -1
        self._pending: Dict[str, Any] = {}

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ ───────────
    def continue_to_phase3_gate(self, p2: ThermalProsecutorPhase2Certificate,
                                params: ProsecutorialGovernanceParameters, *,
                                expected_epoch: int) -> ProsecutorialGateReceipt:
        r"""
        Φ₂₃ — Integridad DURA (⇒ fail-closed) separada de la física/epistémica:
        Lip(T³)<1, acta verificada, contrafirma, y que ORIENT no haya actuado. Si la física
        falla se ADMITE con ⊥ FORZADO para que Fase 3 pueda sentenciar y aislar.
        """
        payload = p2.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (p2.epoch == expected_epoch and p2.phase1.epoch == expected_epoch
                   and (not p2.engine_invoked or (p2.engine_acta_verified and p2.engine_countersignature_ok))
                   and not p2.engine_actuation_dispatched)
        if not (chain_ok and sig_ok and link_ok):
            raise ProsecutorialGateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        rep = p2.engine_report
        lip = float(getattr(getattr(rep, "rsi3", None), "lip_t3_bound", 1.0)) if rep else 1.0
        physics_ok = lip < 1.0 and p2.truth_value is not HeytingTruthValue.VETOED
        return ProsecutorialGateReceipt("Φ₂₃", recomputed, chain_ok, sig_ok, link_ok, True,
                                        not bool(physics_ok),
                                        "⊥ forzado: se sentencia y aísla" if not physics_ok
                                        else "contracción y custodia verificadas")

    def accept_phase2_certificate(self, p2: ThermalProsecutorPhase2Certificate,
                                  params: ProsecutorialGovernanceParameters, *,
                                  expected_epoch: int) -> ProsecutorialGateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.6."""
        return self.continue_to_phase3_gate(p2, params, expected_epoch=expected_epoch)

    # ─────────── §3.2 Functor de severidad, núcleo ¬¬ ───────────
    @classmethod
    def action_functor(cls, truth: HeytingTruthValue, *, charge_proven: bool,
                       structural: bool, lesser_insufficient: bool) -> str:
        """
        D : Ω₃ᵒᵖ → Acciones (contravariante, verificado). Política DECLARADA:
        ⊤⇒WARN, ½⇒DERATE, ⊥⇒ISOLATE; TRIP exige prueba ESTRUCTURAL, cargo probado y
        declaración de insuficiencia de medidas menores. Una condena sólo estadística
        nunca alcanza el peldaño máximo.
        """
        if truth is HeytingTruthValue.COHERENT:
            return "WARN"
        if truth is HeytingTruthValue.DEGRADED:
            return "DERATE"
        return "TRIP" if (charge_proven and structural and lesser_insufficient) else "ISOLATE"

    @classmethod
    def verify_action_functor(cls) -> bool:
        vals = list(HeytingTruthValue)
        f = lambda t: cls._SEVERITY[cls.action_functor(t, charge_proven=True, structural=True,
                                                       lesser_insufficient=True)]   # noqa: E731
        return all(f(a.meet(b)) == max(f(a), f(b)) for a in vals for b in vals)

    @staticmethod
    def verify_nucleus() -> bool:
        vals = list(HeytingTruthValue)
        return bool(all(int(a) <= int(a.double_negation()) for a in vals)
                    and all(a.double_negation().double_negation() == a.double_negation() for a in vals)
                    and all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                            for a in vals for b in vals))

    # ─────────── §3.3 Autoridad de autorización ───────────
    def mint_authorization_token(self, acta: str, epoch: int, rung: int,
                                 authorizer_id: str = "HUMAN") -> str:
        return self._signer.sign(_HMAC_AUTH, {"acta": acta, "epoch": int(epoch),
                                              "rung": int(rung), "authorizer": authorizer_id})

    def verify_authorization_token(self, token: str, acta: str, epoch: int, rung: int,
                                   authorizer_id: str = "HUMAN") -> bool:
        return self._signer.verify(_HMAC_AUTH, {"acta": acta, "epoch": int(epoch),
                                                "rung": int(rung), "authorizer": authorizer_id}, token)

    def _verify_pending(self, token: str) -> bool:
        p = self._pending
        return bool(p) and self.verify_authorization_token(token, p["acta"], p["epoch"], p["rung"])

    # ─────────── §3.4 RSI³ del fiscal ───────────
    def prosecutorial_rsi3(self, p2: ThermalProsecutorPhase2Certificate,
                           params: ProsecutorialGovernanceParameters) -> ProsecutorialRSI3Report:
        r"""
        NIVEL 1 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), q=1−½·m con m = margen de
        evidencias duras; Lip_∞ ≤ q y Lip(T³) ≤ q³ (submultiplicatividad), sondeo empírico y
        cotas de Banach a priori/a posteriori.
        NIVEL 2 · Calibración del umbral de severidad por Newton–Kantorovich:
        f(d)=m−κd²−obj, x₀=c·d* ⇒ h=(c²−1)/(2c²) < ½ SIEMPRE; iteración babilónica,
        razón cuadrática medida; traza de capacidad C_k=−log₁₀ e_k y jerk MEDIDO (SG 7/3).
        NIVEL 3 · Leyes de Kleisli, axiomas del núcleo ¬¬ y ley contravariante del functor.
        """
        ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.engine_report
        if rep is not None:
            ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        hard = [e for e in ev if e.hard]
        m = float(sum(1 for e in hard if e.passed)) / max(1, len(hard))
        q = 1.0 - 0.5 * max(m, 1e-3)
        kappa, c_init = 1.0, 10.0
        target = 0.25 * m
        d_star = math.sqrt(max(m - target, 0.0) / kappa)
        theta_star = np.array([d_star, m, 1.0 - params.alpha_family])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(20_261_118)
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
        for _ in range(48):
            pert = rng.standard_normal(3) * 1e-7
            empirical = max(empirical, float(np.max(np.abs(triple(th + pert) - triple(th))))
                            / float(np.max(np.abs(pert))))
        dk = c_init * max(d_star, 1e-12)
        h_k = (1.0 / abs(-2.0 * kappa * dk)) * abs((m - kappa * dk ** 2 - target)
                                                   / (-2.0 * kappa * dk)) * 2.0 * kappa
        errs: List[float] = []
        for _ in range(10):
            errs.append(abs(dk - d_star))
            dk = dk - (m - kappa * dk ** 2 - target) / (-2.0 * kappa * dk)
        errs.append(abs(dk - d_star))
        err = np.array(errs)
        ratio = 0.0
        for i in range(len(err) - 1):
            if err[i] > 1e-12 * max(d_star, 1e-12):
                ratio = max(ratio, abs(err[i + 1] - err[i] ** 2 / (2.0 * max(d_star, 1e-12)))
                            / (err[i + 1] + 1e-300))
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
        return ProsecutorialRSI3Report(
            severity_threshold_calibrated=float(d_star), kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_quadratic_ratio_residual=float(ratio),
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(empirical),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0), kleisli_monad_laws_ok=laws,
            nucleus_axioms_ok=self.verify_nucleus(), action_functor_ok=self.verify_action_functor(),
        )

    # ─────────── §3.5 Φ_sem con procedencia ───────────
    @staticmethod
    def semantic_visceral_functor(p2: ThermalProsecutorPhase2Certificate,
                                  params: ProsecutorialGovernanceParameters, *, action: str,
                                  proven: int) -> ThermalProsecutorVisceralMetrics:
        """Métricas procesales MEDIDAS + KPI financieros sólo con insumos declarados ([J]·[€/J]/[€])."""
        all_ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.engine_report
        if rep is not None:
            all_ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        coverage = 100.0 * sum(1 for e in all_ev if e.passed) / max(1, len(all_ev))
        provenance = ["COVERAGE:MEASURED", "FWER:DECLARED_ALPHA", "EVASION:THREE_GAP_BOUND"]
        wacc = liq = kv = None
        w_irr = 0.0
        if rep is not None:
            w_irr = float(rep.phase_ii.jarzynski_kms.irreversible_work_j)
            x_dest = float(rep.phase_ii.spohn.sigma_min)
            wacc = 100.0 * max(0.0, x_dest) / (max(0.0, x_dest) + params.exergy_reference_w)
            provenance.append("WACC:POLICY_HEURISTIC(sigma/(sigma+sigma_ref))")
        if params.exergy_price_per_joule is not None and params.capital_base:
            liq = 100.0 * w_irr * params.exergy_price_per_joule / params.capital_base
            provenance.append("LIQUIDITY:DECLARED(J*price/capital)")
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
        return ThermalProsecutorVisceralMetrics(
            charge_coverage_pct=float(coverage), proven_charges=int(proven),
            family_wise_error_budget_pct=100.0 * params.alpha_family,
            evasion_window_bound_pct=100.0 * p2.cadence.coverage_bound,
            wacc_protection_pct=wacc, net_liquidity_release_pct=liq, kv_cache_compression_pct=kv,
            irreversible_work_status=f"W_irr={w_irr:.6f} J (Jensen certificado por el motor)",
            due_process_status=(f"Ville+Holm α_family={params.alpha_family:.4f} · acción={action} · "
                                f"cargos_estructurales={len(p2.charge_sheet.structural_charges)}"),
            provenance=tuple(provenance), monotonicity_ok=monotone,
        )

    # ─────────── §3.6 COMMIT de actuación ───────────
    def commit_actuation(self, action: str, acta: str, epoch: int, *, token: Optional[str],
                         params: ProsecutorialGovernanceParameters,
                         self_issued: bool) -> ActuationAuthorizationReport:
        """
        Último paso del ciclo. Mide N despachos en seco, verifica token, ne bis in idem y
        anti-replay; sólo entonces arma y dispara. Sin token: estado seguro + escalada.
        Declara la inviabilidad física del plazo lógico para un SCR sin medición HIL.
        """
        rung = self._SEVERITY[action]
        samples = np.array([self._driver.dry_run() for _ in range(64)], dtype=np.float64)
        ne_bis = acta not in self._judged_actas
        anti_replay = epoch > self._last_epoch
        token_ok = bool(token) and self.verify_authorization_token(str(token), acta, epoch, rung)
        requested = bool(rung == 3)
        authorized = bool(requested and token_ok and ne_bis and anti_replay)
        dispatched, trip_ns, safe = False, 0.0, False
        if authorized:
            self._pending = {"acta": acta, "epoch": epoch, "rung": rung}
            self._driver.arm()
            trip_ns = float(self._driver.trip(str(token)))
            dispatched = True
            self._judged_actas.add(acta)
            self._last_epoch = int(epoch)
        elif rung >= 2:
            self._driver.safe_state()
            safe = True
            self._judged_actas.add(acta)
            self._last_epoch = int(epoch)
        return ActuationAuthorizationReport(
            ladder_rung=rung, action=action, indictment_digest=acta, epoch=int(epoch),
            authorization_token_verified=token_ok, authorization_self_issued=bool(self_issued),
            ne_bis_in_idem_ok=bool(ne_bis), anti_replay_ok=bool(anti_replay),
            actuation_requested=requested, actuation_authorized=authorized,
            actuation_dispatched=dispatched, safe_state_applied=safe,
            driver_name=self._driver.name, driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            note=("392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo "
                  "alcanza. Sin token válido NO hay disparo: estado seguro + escalada."),
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Sentencia sellada ───────────
    def emit_sovereign_state(
        self, p2: ThermalProsecutorPhase2Certificate, params: ProsecutorialGovernanceParameters, *,
        expected_epoch: int, authorization_token: Optional[str] = None,
        principal_truth: Optional[HeytingTruthValue] = None,
        witness_truth: Optional[HeytingTruthValue] = None,
        error: Optional[str] = None, forced_veto: bool = False,
    ) -> ThermalProsecutorSovereignState:
        """MÉTODO TERMINAL: sentencia, autoriza, ejecuta el commit y sella el acta final."""
        t0 = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, params, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        sheet = p2.charge_sheet
        structural = len(sheet.structural_charges) > 0
        proven = sheet.statistical_convictions + len(sheet.structural_charges)
        charge_proven = proven > 0
        action = self.action_functor(truth, charge_proven=charge_proven, structural=structural,
                                     lesser_insufficient=params.lesser_measures_insufficient)
        rsi3 = self.prosecutorial_rsi3(p2, params)

        acta_payload = {
            "epoch": int(expected_epoch), "truth": truth, "action": action,
            "p2_chain": p2.chain_hash, "p1_chain": p2.phase1.chain_hash,
            "engine_acta": getattr(getattr(p2.engine_report, "sentence", None), "indictment_digest", ""),
            "structural": sheet.structural_charges, "statistical": sheet.rejected_indices,
            "error": error,
        }
        acta = _sha256(_canonical(acta_payload))
        self_issued = False
        token = authorization_token
        if token is None and action == "TRIP" and not params.require_human_authorization:
            token = self.mint_authorization_token(acta, expected_epoch, 3)
            self_issued = True
        auth = self.commit_actuation(action, acta, expected_epoch, token=token, params=params,
                                     self_issued=self_issued)
        visceral = self.semantic_visceral_functor(p2, params, action=action, proven=proven)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("AP3.functor_contravariante", rsi3.action_functor_ok, hard=True),
            Evidence.boolean("AP3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence("AP3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("AP3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("AP3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("AP3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("AP3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("AP3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("AP3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("AP3.autorizacion_habilita",
                             (not auth.actuation_dispatched) or auth.authorization_token_verified, hard=True),
            Evidence.boolean("AP3.ne_bis_in_idem", auth.ne_bis_in_idem_ok, hard=True),
            Evidence.boolean("AP3.anti_replay", auth.anti_replay_ok, hard=True),
            Evidence.boolean("AP3.proporcionalidad",
                             auth.ladder_rung < 3 or (structural and params.lesser_measures_insufficient),
                             hard=True),
            Evidence.boolean("AP3.presuncion_inocencia",
                             not (sheet.statistical_convictions > 0 and not structural
                                  and auth.ladder_rung == 3), hard=True),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])

        escalation: Optional[ProsecutorialEscalation] = None
        divergent = ((principal_truth is not None and int(truth) < int(principal_truth))
                     or (witness_truth is not None and int(truth) < int(witness_truth)))
        if auth.ladder_rung >= 2 or divergent or error is not None:
            reason = (f"peldaño={auth.action}" + (" | divergencia estricta" if divergent else "")
                      + (f" | error: {error}" if error else ""))
            esc_core = {"epoch": int(expected_epoch), "truth": truth, "principal": principal_truth,
                        "witness": witness_truth, "reason": reason, "acta": acta}
            escalation = ProsecutorialEscalation(
                epoch=int(expected_epoch), truth_value=truth, principal_truth=principal_truth,
                witness_truth=witness_truth, reason=reason, merkle_root_hash="",
                signature=self._signer.sign(_HMAC_ESC, esc_core), delivered=False,
            )

        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": getattr(p2.engine_report, "passport_hash", ""),
                        "engine_root": getattr(p2.engine_report, "merkle_root_hash", ""),
                        "countersignature_ok": p2.engine_countersignature_ok}),
            _canonical(sheet), _canonical(p2.cadence), _canonical(acta_payload),
            _canonical(rsi3),
            _canonical({k: v for k, v in _freeze(auth).items()
                        if k not in ("dispatch_median_ns", "dispatch_p99_ns", "trip_dispatch_ns")}),
            _canonical(p2.payload()),
        ]
        root = merkle_root(leaves)
        s_idx = 4
        proof = merkle_proof(leaves, s_idx)
        if escalation is not None:
            escalation = dataclasses.replace(escalation, merkle_root_hash=root)
        core = {
            "truth_value": truth, "boolean_shadow": truth.double_negation(),
            "crowbar_triggered": bool(auth.actuation_dispatched),
            "attestation_withheld": bool(auth.ladder_rung >= 2),
            "merkle_root_hash": root, "sentence_leaf_index": s_idx, "sentence_inclusion_proof": proof,
            "kleisli_ledger": ("Φ₁:CUSTODIA", "Φ₂:INSTRUCCION", "Φ₃:SENTENCIA-COMMIT"),
            "epoch": int(expected_epoch), "error": error,
            "phase2_chain_hash": p2.chain_hash, "phase1_chain_hash": p2.phase1.chain_hash,
            "authorization": {"rung": auth.ladder_rung, "action": auth.action, "acta": acta,
                              "authorized": auth.actuation_authorized,
                              "dispatched": auth.actuation_dispatched,
                              "token_verified": auth.authorization_token_verified,
                              "ne_bis": auth.ne_bis_in_idem_ok, "anti_replay": auth.anti_replay_ok,
                              "safe_state": auth.safe_state_applied},
            "escalation_signature": escalation.signature if escalation else None,
            "merkle_leaf_count": len(leaves),
        }
        state = ThermalProsecutorSovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(),
            crowbar_triggered=bool(auth.actuation_dispatched),
            crowbar_latency_ns=float(auth.trip_dispatch_ns),
            attestation_withheld=bool(auth.ladder_rung >= 2), merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), sentence_leaf_index=s_idx,
            sentence_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3, authorization=auth,
            escalation=escalation, phase2=p2, evidences=evidences, kleisli_ledger=core["kleisli_ledger"],
            epoch=int(expected_epoch), state_signature=self._signer.sign(_HMAC_STATE, core), error=error,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("AP3 Ω₃=%s · %s · autorizado=%s · disparo=%s",
                    truth.name, auth.action, auth.actuation_authorized, auth.actuation_dispatched)
        return state

    def execute_phase3(self, *a: Any, **k: Any) -> ThermalProsecutorSovereignState:
        """Alias heredado (v13) del terminal §3.7."""
        return self.emit_sovereign_state(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA — ciclo OODA fiscal como composición de Kleisli, con fail-closed
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovThermalExergyWeaverProsecutorAgent:
    r"""
    Soberano agéntico FISCAL. Ejecuta observe >=> orient >=> decide sobre la mónada Certified.
    El motor corre en modo epistémico (driver sólo lectura, sin token). Ante cualquier fallo:
    FAIL-CLOSED — ⊥, sin disparo, estado seguro y acta de emergencia firmada.
    """

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 driver: Optional[CrowbarDriver] = None, seed: int = 20_261_117) -> None:
        self._signer = ProsecutorialSigner(secret_key)
        self._principal = principal_engine
        self._readonly = ReadOnlyCrowbarDriver()
        self._engine = ProsecutorEngine(secret_key=self._signer.key_bytes,
                                        principal_engine=principal_engine, seed=seed,
                                        driver=self._readonly)
        self._observer = ThermalExergyProsecutorPhase1Observer(self._signer, seed=seed)
        self._orientor = ThermalExergyProsecutorPhase2Orientor(self._engine, self._signer, seed=seed)
        self._decider = ThermalExergyProsecutorPhase3Decider(self._signer, driver or self._make_driver())
        self._epoch = 0
        self._seed = int(seed)
        logger.info("%s v%s listo · clave=%s", type(self).__name__, __version__, self._signer.fingerprint())

    def _make_driver(self) -> CrowbarDriver:
        return SimulatedCrowbarDriver(
            lambda t: self._decider._verify_pending(t) if hasattr(self, "_decider") else False
        )

    # ─────────── preparación declarada de ρ y referencia de Gibbs ───────────
    def prepare_inputs(self, apu_cost_matrix: NDArray, rho_mac: Optional[NDArray],
                       params: ProsecutorialGovernanceParameters) -> Tuple[NDArray, NDArray]:
        k = np.asarray(apu_cost_matrix, dtype=np.float64)
        dim = int(k.shape[0])
        _, k_reg, _, _ = self._engine._principal.regularize_kernel(k, psd_floor=params.psd_floor)
        en, u = la.eigh(k_reg)
        lp = -params.beta_thermo * (en - en[0])
        lp -= float(np.max(lp)) + math.log(float(np.sum(np.exp(lp - float(np.max(lp))))))
        gibbs = (u * np.exp(lp)) @ u.conj().T
        rho = gibbs if rho_mac is None else np.asarray(rho_mac, dtype=np.complex128)
        return rho, gibbs

    # ─────────── verificación externa del estado ───────────
    def verify_sovereign_state(self, state: ThermalProsecutorSovereignState) -> bool:
        p1, p2 = state.phase2.phase1, state.phase2
        if not self._signer.verify(_HMAC_P1, p1.payload(), p1.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p1.payload())), p1.chain_hash):
            return False
        if not self._signer.verify(_HMAC_P2, p2.payload(), p2.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p2.payload())), p2.chain_hash):
            return False
        if p2.engine_report is not None and not self._engine.verify_prosecutor_report(p2.engine_report):
            return False
        leaves = [bytes.fromhex(h) for h in state.merkle_leaves]
        if merkle_root(leaves) != state.merkle_root_hash:
            return False
        i = state.sentence_leaf_index
        if not verify_merkle(leaves[i], i, len(leaves), state.sentence_inclusion_proof, state.merkle_root_hash):
            return False
        return self._signer.verify(_HMAC_STATE, state.payload(), state.state_signature)

    # ─────────── ciclo OODA fiscal (Kleisli) con fail-closed ───────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: Optional[NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        cost_streams: Optional[Sequence[NDArray[np.float64]]] = None,
        principal_report: Any = None,
        witness_report: Any = None,
        authorization_token: Optional[str] = None,
        params: Optional[ProsecutorialGovernanceParameters] = None,
        force_veto: bool = False,          # ABOLIDO: se rechaza, jamás causa actuación
    ) -> ThermalProsecutorSovereignState:
        """Ejecuta el proceso completo. La actuación es el último commit, sólo con token."""
        params = params or ProsecutorialGovernanceParameters()
        if force_veto:
            raise ProsecutorialGovernanceError(
                "force_veto está abolido: un booleano no autenticado no puede causar actuación física. "
                "Presente un cargo probado y un token de autorización firmado.")
        self._epoch += 1
        epoch = self._epoch
        exps = np.asarray(exponents, dtype=np.float64)
        if cost_streams is None:
            cost_streams = (np.random.default_rng(self._seed + 5).normal(
                params.cost_mu0, params.cost_sigma, 200),)
        streams = tuple(np.asarray(s, dtype=np.float64) for s in cost_streams)
        rho, gibbs = self.prepare_inputs(apu_cost_matrix, rho_mac, params)
        passport = getattr(principal_report, "passport_hmac", "")
        p_truth = getattr(principal_report, "truth_value", None)
        w_truth = getattr(witness_report, "truth_value", None)

        def observe_m(ctx: Dict[str, Any]) -> Any:
            p1 = self._observer.emit_phase1_certificate(
                rho, apu_cost_matrix, exps, streams, params, epoch=epoch,
                principal_passport=passport, gibbs_reference=gibbs)
            return Certified({**ctx, "p1": p1}, ("Φ₁:CUSTODIA",), p1.truth_value)

        def orient_m(ctx: Dict[str, Any]) -> Any:
            p2 = self._orientor.emit_phase2_certificate(
                ctx["p1"], apu_cost_matrix, exps, rho, streams, params, expected_epoch=epoch,
                principal_report=principal_report, witness_report=witness_report)
            return Certified({**ctx, "p2": p2}, ("Φ₂:INSTRUCCION",), p2.truth_value)

        def decide_m(ctx: Dict[str, Any]) -> Any:
            st = self._decider.emit_sovereign_state(
                ctx["p2"], params, expected_epoch=epoch, authorization_token=authorization_token,
                principal_truth=p_truth, witness_truth=w_truth)
            return Certified(st, ("Φ₃:SENTENCIA-COMMIT",), st.truth_value)

        pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
        try:
            result = pipeline({"epoch": epoch})
            state: ThermalProsecutorSovereignState = result.value
            if not self.verify_sovereign_state(state):
                raise ProsecutorialGateIntegrityError("Autoverificación de la sentencia fallida.")
            logger.info("Proceso OODA fiscal completado · Ω₃=%s · ledger=%s",
                        state.truth_value.name, " → ".join(result.ledger))
            return state
        except Exception as exc:                                   # FAIL-CLOSED, nunca silencio
            logger.critical("FAIL-CLOSED FISCAL: %s", exc)
            return self._emergency_state(rho, gibbs, apu_cost_matrix, exps, streams, params, epoch,
                                         str(exc), principal_report, p_truth, w_truth)

    def _emergency_state(self, rho: NDArray, gibbs: NDArray, k_mat: NDArray, exps: NDArray,
                         streams: Sequence[NDArray], params: ProsecutorialGovernanceParameters,
                         epoch: int, message: str, principal_report: Any,
                         p_truth: Optional[HeytingTruthValue],
                         w_truth: Optional[HeytingTruthValue]) -> ThermalProsecutorSovereignState:
        """Acta de emergencia: ⊥, sin disparo, estado seguro, cadena de custodia preservada."""
        p1 = self._observer.emit_phase1_certificate(
            rho, k_mat, exps, streams, params, epoch=epoch,
            principal_passport=getattr(principal_report, "passport_hmac", ""), gibbs_reference=gibbs)
        sheet = ChargeSheetReport(
            n_charges=len(streams), pvalues_anytime=tuple(1.0 for _ in streams),
            pvalues_adjusted=tuple(1.0 for _ in streams), rejected_indices=(),
            alpha_family=params.alpha_family, structural_charges=("fallo_de_integridad",),
            statistical_convictions=0, mc_ville_validity_rate=0.0, mc_fwer=0.0,
            mc_rates_within_declared=False, max_log_likelihood=0.0,
            model_note="acta de emergencia: instrucción no completada",
        )
        cadence = self._orientor.anti_evasion_cadence(params)
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": False, "engine_passport": "",
            "engine_acta": "", "engine_merkle_root": "", "engine_countersignature": "",
            "engine_acta_verified": False, "engine_merkle_recomputed_ok": False,
            "engine_countersignature_ok": False, "engine_actuation_dispatched": False,
            "engine_truth": HeytingTruthValue.VETOED, "charge_sheet": sheet, "cadence": cadence,
            "evidences": (), "truth_value": HeytingTruthValue.VETOED, "epoch": int(epoch),
        }
        chain = _sha256(_canonical(core))
        p2 = ThermalProsecutorPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=None, engine_invoked=False,
            engine_acta_verified=False, engine_merkle_recomputed_ok=False,
            engine_countersignature_ok=False, engine_actuation_dispatched=False,
            engine_truth=HeytingTruthValue.VETOED, charge_sheet=sheet, cadence=cadence,
            evidences=(), truth_value=HeytingTruthValue.VETOED, epoch=int(epoch), chain_hash=chain,
            hmac_signature=self._signer.sign(_HMAC_P2, core))
        return self._decider.emit_sovereign_state(
            p2, params, expected_epoch=epoch, authorization_token=None,
            principal_truth=p_truth, witness_truth=w_truth, error=message, forced_veto=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    agent = BoolNovikovThermalExergyWeaverProsecutorAgent(
        secret_key=b"PROSECUTOR_AGENT_KEY_32_BYTES!!", principal_engine=principal)
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.45, 0.85], dtype=np.float64)
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)
    rng = np.random.default_rng(9)
    P = ProsecutorialGovernanceParameters()
    innocent = tuple(rng.normal(P.cost_mu0, P.cost_sigma, 200) for _ in range(3))

    # 1) Nominal: H0 en todos los canales ⇒ ⊤, WARN, SIN disparo; Orient no actuó
    st = agent.run_sovereign_governance_cycle(None, k_test, exps, cost_streams=innocent,
                                              principal_report=prep)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.authorization.action == "WARN" and not st.crowbar_triggered
    assert not st.phase2.engine_actuation_dispatched            # ← Orient epistémico
    assert st.phase2.engine_acta_verified and st.phase2.engine_countersignature_ok
    assert st.phase2.charge_sheet.statistical_convictions == 0
    assert st.phase2.charge_sheet.mc_rates_within_declared
    assert st.phase2.cadence.three_gap_ok and st.phase2.cadence.slater_ok
    assert st.phase1_is_ok if False else st.phase2.phase1.bw_isometry_residual < 1e-11
    assert agent.verify_sovereign_state(st)
    assert st.visceral_metrics.net_liquidity_release_pct is None    # sin €/J: honesto

    # 2) LA MALFORMACIÓN NO OTORGA INMUNIDAD (v13 la eximía en Φ₁₂)
    bad_rho = np.array([[1.0, 0.0], [0.0, -0.2]], dtype=np.complex128)   # no PSD
    mal = agent.run_sovereign_governance_cycle(bad_rho, k_test, exps, cost_streams=innocent,
                                               principal_report=prep)
    assert not mal.phase2.phase1.is_psd and mal.phase2.phase1.truth_value is HeytingTruthValue.VETOED
    assert mal.truth_value is HeytingTruthValue.VETOED           # ← procesado, no eximido
    assert "rho_no_PSD" in mal.phase2.charge_sheet.structural_charges
    assert not mal.crowbar_triggered

    # 3) Culpabilidad estadística: Holm rechaza, pero NUNCA alcanza TRIP por sí sola
    guilty = tuple(rng.normal(P.cost_mu1, P.cost_sigma, 300) for _ in range(3))
    g = agent.run_sovereign_governance_cycle(None, k_test, exps, cost_streams=guilty,
                                             principal_report=prep)
    assert g.phase2.charge_sheet.statistical_convictions >= 1
    assert g.authorization.ladder_rung <= 2 and not g.crowbar_triggered

    # 4) Violación ESTRUCTURAL + insuficiencia declarada ⇒ TRIP solicitado; SIN token NO dispara
    hard_params = ProsecutorialGovernanceParameters(ball_r=2.0, cylinder_r=1.0,
                                                    lesser_measures_insufficient=True)
    bad = agent.run_sovereign_governance_cycle(None, k_test, exps, cost_streams=innocent,
                                               principal_report=prep, params=hard_params)
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.authorization.action == "TRIP"
    assert bad.authorization.actuation_requested and not bad.authorization.actuation_authorized
    assert not bad.crowbar_triggered and bad.authorization.safe_state_applied

    # 5) CON token ligado al acta ⇒ disparo autorizado y latencia MEDIDA
    tok = agent._decider.mint_authorization_token(bad.authorization.indictment_digest,
                                                  bad.epoch + 1, 3)
    armed = agent.run_sovereign_governance_cycle(None, k_test, exps, cost_streams=innocent,
                                                 principal_report=prep, params=hard_params,
                                                 authorization_token=tok)
    assert armed.authorization.actuation_authorized and armed.crowbar_triggered
    assert armed.crowbar_latency_ns > 0.0
    assert not armed.authorization.physical_deadline_feasible     # SCR ~µs: declarado

    # 6) force_veto abolido: un booleano no puede causar actuación
    try:
        agent.run_sovereign_governance_cycle(None, k_test, exps, force_veto=True)
        raise AssertionError("force_veto no fue rechazado")
    except ProsecutorialGovernanceError:
        pass

    # 7) Manipulación de P1 ⇒ Φ₁₂ detecta (v13 no lo hacía)
    forged = dataclasses.replace(st.phase2.phase1, tr_rho=999.0)
    try:
        agent._orientor.continue_to_phase2_gate(forged, k_test, exps, innocent, P,
                                                expected_epoch=forged.epoch)
        raise AssertionError("Φ₁₂ aceptó un payload manipulado")
    except ProsecutorialGateIntegrityError:
        pass

    # 8) Ville + Holm: validez anytime y FWER bajo la nula global
    p_v, _ = ville_anytime_pvalue(rng.normal(0.0, 1.0, 500), 0.0, 0.5, 1.0)
    assert 0.0 <= p_v <= 1.0
    rej, adj = holm_bonferroni([0.001, 0.02, 0.6], 0.05)
    assert rej == (0,) and adj[0] <= 0.003 + 1e-12
    assert st.phase2.charge_sheet.mc_fwer <= P.alpha_family + 0.01

    # 9) Kac–Slater y Merkle RFC 6962; determinismo y detección de manipulación
    rec = st.phase2.cadence
    assert abs(rec.kac_empirical_delay - 10.0) < 0.05 and len(rec.slater_return_times) <= 3
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    st2 = agent.run_sovereign_governance_cycle(None, k_test, exps, cost_streams=innocent,
                                               principal_report=prep)
    assert st2.phase2.phase1.chain_hash == st.phase2.phase1.chain_hash   # sellos sin reloj
    assert not agent.verify_sovereign_state(dataclasses.replace(st, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverProsecutorAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {st.truth_value.name}({int(st.truth_value)}) / {st.boolean_shadow.name}")
    print(f"Sentencia / disparo           : {st.authorization.action} / {st.crowbar_triggered}")
    print(f"Orient actuó (debe ser False) : {st.phase2.engine_actuation_dispatched}")
    print(f"d_BW(ρ,ρ_β) / isometría Haar  : {st.phase2.phase1.bures_wasserstein_to_gibbs:.6f} / "
          f"{st.phase2.phase1.bw_isometry_residual:.2e}")
    print(f"Cargos / rechazados (Holm)    : {st.phase2.charge_sheet.n_charges} / "
          f"{st.phase2.charge_sheet.statistical_convictions}")
    print(f"p-values anytime              : "
          + ", ".join(f"{v:.4f}" for v in st.phase2.charge_sheet.pvalues_anytime))
    print(f"Ville P(p≤α) / FWER MC        : {st.phase2.charge_sheet.mc_ville_validity_rate:.4f} / "
          f"{st.phase2.charge_sheet.mc_fwer:.4f} (α={P.alpha_family})")
    print(f"Cadencia α / γ / huecos       : {rec.rotation_number:.6f} / {rec.diophantine_gamma:.4f} / "
          f"{rec.distinct_gap_count}")
    print(f"Ventana de evasión ≤          : {100*rec.coverage_bound:.2f}% del ciclo")
    print(f"Kac retardo esperado/empírico : {rec.kac_expected_detection_delay:.3f} / {rec.kac_empirical_delay:.3f}")
    print(f"Slater tiempos de retorno     : {rec.slater_return_times}")
    print(f"RSI³ q / Lip(T³) medido       : {st.rsi3.policy_lipschitz_q:.6f} / {st.rsi3.lip_t3_empirical:.6f}")
    print(f"Kantorovich h / d³C           : {st.rsi3.kantorovich_h:.6f} / {st.rsi3.capacity_d3:.4f}")
    print(f"No-PSD: procesado (no eximido): {mal.truth_value.name} · cargos={mal.phase2.charge_sheet.structural_charges}")
    print(f"⊥ SIN token                   : {bad.authorization.action} autorizado={bad.authorization.actuation_authorized} "
          f"disparo={bad.crowbar_triggered} seguro={bad.authorization.safe_state_applied}")
    print(f"⊥ CON token                   : {armed.authorization.action} disparo={armed.crowbar_triggered} "
          f"latencia={armed.crowbar_latency_ns:.0f} ns")
    print(f"Merkle raíz / hojas           : {st.merkle_root_hash[:16]}… / {len(st.merkle_leaves)}")
    print(f"Sentencia reverificable       : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")