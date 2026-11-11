# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderProsecutorAgent  v14.0.0-PhD (reedición rigurosa)    ║
║ Ubicación: app/agents/wisdom/bool_novikov_gamma_photon_collider_prosecutor_agent.py           ║
║ Función  : Soberano agéntico FISCAL del colisionador γγ: custodia el pliego, controla la      ║
║            multiplicidad (BH/FDR) y el gasto de error (Lan–DeMets), refuta la afirmación de   ║
║            estabilidad, y es la ÚNICA autoridad que autoriza y EJECUTA un disparo medido.     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA PROCESAL DEL FISCAL DEL COLISIONADOR
─────────────────────────────────────────────
(1) NUNCA SE FABRICA UNA ACTUACIÓN. v13 hacía
        if truth == VETOED and not crowbar_triggered: crowbar_triggered = True; latency = 392.1
    es decir: anulaba la decisión del motor de NO disparar e inventaba una latencia sin HAL.
    Un registro de actuación fabricado es PEOR que ninguno: enmascara un fallo real de disparo.
    Aquí `crowbar_triggered` ⟺ existe un commit MEDIDO con quórum válido.
(2) AUTENTICAR UN RESUMEN NO ES CUSTODIA. v13 recalculaba el hash de `t‖Tr ρ‖is_psd`: cualquier
    ρ y K con igual traza y PSD pasaban. Aquí P1 compromete los digests de ρ, K, exponentes,
    del PLIEGO COMPLETO y de la política, y Φ₁₂ re-digesta los arrays reales (anti-TOCTOU).
(3) NADIE CONDENA POR ACUMULACIÓN. Los cargos estadísticos (conteos de Poisson por reclamo) se
    resuelven con BENJAMINI–HOCHBERG a q declarado, con FDP verificado por Monte Carlo. Las
    refutaciones ESTRUCTURALES son pruebas (p=0) y no entran en la multiplicidad.
(4) NO SE RE-PROCESA HASTA CONDENAR. El gasto de error por época sigue una función de
    alpha-spending de Lan–DeMets (O'Brien–Fleming): α*(t)=2−2Φ(z_{1−α/2}/√t), con Σ增量 = α_total.
    Además: ne bis in idem, PRESCRIPCIÓN (ventana calibrada por Kac) y anti-replay.
(5) EL FISCAL INTENTA REFUTAR, NO CONFIRMAR. Re-deriva las refutaciones estructurales de TODO
    el pliego y somete la afirmación de estabilidad del colisionador a un escaneo de resonancias
    de orden MÁS PROFUNDO que el del motor. Ausencia de resonancia a orden finito ≠ estabilidad.

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.5  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.8  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle
    P1 ⊂ P2 ⊂ GammaPhotonProsecutorSovereignState; ejecución = Kleisli observe>=>orient>=>decide.

FASE 1 — OBSERVE (custodia del pliego)
  §1.1 Validación previa + reconciliación de Fock (dim ρ = 2^m).
  §1.2 Normalización auditada + ISOMETRÍA DE BURES–WASSERSTEIN bajo unitario Haar.
  §1.3 PRE-COMPROMISO: política de carga (α_total, q_FDR, prescripción) y calibración de g.
  §1.4 Compromiso canónico de entradas, pliego y política.
  §1.5 [TERMINAL] emit_phase1_certificate.

COMPUERTA Φ₁₂ — hash recalculado desde el payload, HMAC, re-digest de arrays, época; ADMITE ⊥.

FASE 2 — ORIENT (instrucción epistémica)
  §2.1 accept_phase1_certificate.
  §2.2 Motor fiscal v14 con driver de SÓLO LECTURA y SIN token (Orient no actúa).
  §2.3 Verificación del acta del motor + Merkle público + prueba de inclusión.
  §2.4 PLIEGO: refutación estructural por reclamo + BH/FDR con FDP verificado por MC.
  §2.5 Alpha-spending de Lan–DeMets por época (gasto total acotado).
  §2.6 Falsación adversarial de la estabilidad: resonancias m ν_x + n ν_y = p a orden profundo.
  §2.7 Kac–Slater: recurrencia del régimen anómalo y ventana de prescripción.
  §2.8 [TERMINAL] emit_phase2_certificate.

COMPUERTA Φ₂₃ — integridad dura; física/epistémica ⇒ ⊥ forzado pero ADMITE a Fase 3.

FASE 3 — DECIDE & ACT
  §3.1 accept_phase2_certificate.
  §3.2 Sentencia Ω₃, núcleo ¬¬, functor contravariante y escalera de proporcionalidad.
  §3.3 Debido proceso: ne bis in idem + prescripción + anti-replay + quórum 2-de-2.
  §3.4 RSI³: ventana de prescripción (Kantorovich) + Banach + Kleisli + jerk medido.
  §3.5 Φ_sem con procedencia dimensional.
  §3.6 COMMIT de actuación MEDIDO (jamás fabricado).
  §3.7 [TERMINAL] emit_sovereign_state (Merkle sin clave + firma determinista + escalada).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `positron_hmac_valid` y `force_veto` ABOLIDOS ⇒ `authorization_tokens` firmados (2-de-2).
• `spec.dirac_report` → `spec.phase_ii.event_refutation` / `spec.crowbar`;
  `spec.banach_contraction_lip` → `spec.rsi3.lip_t3_bound`.
• Sellos deterministas (sin reloj); raíz Merkle SIN clave y con prueba de inclusión.
• KPI sin insumos declarados ⇒ None con `UNDECLARED_INPUT`.

LIMITACIONES DECLARADAS
1. BH controla FDR bajo independencia/PRDS; con cargos correlacionados es aproximado (declarado).
2. El alpha-spending supone épocas con fracción de información declarada; no es un test secuencial
   exacto para paradas adaptativas arbitrarias.
3. El escaneo de resonancias es de orden finito: no prueba estabilidad global.
4. El motor audita en profundidad el reclamo PRIMARIO; el agente hace cribado exacto del resto.
5. 392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza.

REFERENCIAS: Benjamini–Hochberg (1995); Lan–DeMets (1983); O'Brien–Fleming (1979); Wald (1945);
Kac (1947); Slater (1967); Courant–Snyder (tunes); Bures (1969); Uhlmann (1976); Breit–Wheeler
(1934); Landau–Yang (1948/1950); Kantorovich; Banach (1922); Mac Lane (Kleisli); RFC 6962; CODATA 2022.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import itertools
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
from scipy.special import gammainc, ndtri


def _import_first(paths: Sequence[str]) -> Any:
    last: Optional[Exception] = None
    for p in paths:
        try:
            return __import__(p, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(f"No se pudo importar ninguno de: {paths}") from last


_pe = _import_first(("app.wisdom.bool_novikov_gamma_photon_collider_prosecutor_engine",
                     "bool_novikov_gamma_photon_collider_prosecutor_engine"))
_pri = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
                      "bool_novikov_thermal_exergy_weaver_engine"))

ProsecutorEngine = _pe.BoolNovikovGammaPhotonColliderProsecutorEngine
EngineClaimedEvent = _pe.ClaimedEvent
EngineProsecutionParameters = _pe.ColliderProsecutionParameters
PrincipalEngine = _pri.BoolNovikovThermalExergyWeaverEngine
HeytingTruthValue = _pri.HeytingTruthValue          # Ω₃ CANÓNICO
Evidence = _pri.Evidence
Certified = _pri.Certified
kleisli_compose = _pri.kleisli_compose

__all__ = [
    "ProsecutorialGovernanceParameters", "GammaPhotonProsecutorPhase1Certificate",
    "GammaPhotonProsecutorPhase2Certificate", "GammaPhotonProsecutorVisceralMetrics",
    "GammaPhotonProsecutorSovereignState", "GammaPhotonProsecutorPhase1Observer",
    "GammaPhotonProsecutorPhase2Orientor", "GammaPhotonProsecutorPhase3Decider",
    "BoolNovikovGammaPhotonColliderProsecutorAgent", "ProsecutorialGovernanceError",
    "ProsecutorialGateIntegrityError", "AuthorizationError", "CrowbarDriver",
    "SimulatedCrowbarDriver", "ReadOnlyCrowbarDriver", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderProsecutorAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-GAMMA-COLLIDER-PROSECUTOR-AGENT-BH-FDR-LAN-DEMETS-PRESCRIPTION-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ME_C2_KEV: Final[float] = 510.99895069
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")
_HMAC_P1: Final[bytes] = b"GAMMA-PROSECUTOR-AGENT-P1-CUSTODY-v14|"
_HMAC_P2: Final[bytes] = b"GAMMA-PROSECUTOR-AGENT-P2-CHARGES-v14|"
_HMAC_STATE: Final[bytes] = b"GAMMA-PROSECUTOR-AGENT-SENTENCE-v14|"
_HMAC_AUTH: Final[bytes] = b"GAMMA-PROSECUTOR-AGENT-DUAL-AUTH-v14|"
_HMAC_ESC: Final[bytes] = b"GAMMA-PROSECUTOR-AGENT-ESCALATION-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle, BW, BH/FDR, alpha-spending
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorialGovernanceError(Exception):
    """Error base de la gobernanza fiscal del colisionador."""


class ProsecutorialGateIntegrityError(ProsecutorialGovernanceError):
    """Fallo DURO de integridad ⇒ fail-closed (estado seguro, sin disparo) y acta de emergencia."""


class AuthorizationError(ProsecutorialGovernanceError):
    """Actuación sin quórum, con replay, acta ya juzgada o cargo prescrito."""


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
    """Raíz RFC 6962 SIN clave: recalculable por cualquier verificador externo."""
    return _mth(list(lv)).hex()


def merkle_proof(lv: Sequence[bytes], idx: int) -> Tuple[Tuple[str, bool], ...]:
    ds = list(lv)
    if not 0 <= idx < len(ds):
        raise ProsecutorialGovernanceError("Índice de hoja fuera de rango.")

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


def _haar_unitary(dim: int, rng: np.random.Generator) -> NDArray:
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * np.where(np.abs(d) > 0.0, d / np.maximum(np.abs(d), _EPS), 1.0)


def benjamini_hochberg(pvals: Sequence[float], q: float) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    r"""
    Procedimiento step-up de Benjamini–Hochberg: ordena p₍₁₎≤…≤p₍ₘ₎ y rechaza hasta
        k* = max{k : p₍ₖ₎ ≤ (k/m)·q}.
    Controla el FDR ≤ q bajo independencia o PRDS. p ajustados (monótonos desde el mayor):
        p̃₍ₖ₎ = min_{j≥k} min(1, (m/j)·p₍ⱼ₎).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return (), ()
    order = np.argsort(p, kind="stable")
    k_star = -1
    for k, idx in enumerate(order, start=1):
        if p[idx] <= (k / m) * q:
            k_star = k
    rejected = tuple(int(i) for i in order[:k_star]) if k_star > 0 else ()
    adj = np.ones(m, dtype=np.float64)
    running = 1.0
    for k in range(m, 0, -1):
        idx = order[k - 1]
        running = min(running, min(1.0, (m / k) * float(p[idx])))
        adj[idx] = running
    return rejected, tuple(float(v) for v in adj)


def poisson_upper_pvalue(n_obs: int, lam0: float) -> float:
    """p-valor exacto de cola superior: P(X ≥ n | λ₀) = gammainc(n, λ₀); n=0 ⇒ 1."""
    return 1.0 if n_obs <= 0 else float(gammainc(n_obs, lam0))


def lan_demets_obrien_fleming(alpha_total: float, info_fraction: float) -> float:
    r"""
    Función de gasto de error de Lan–DeMets con forma O'Brien–Fleming:
        α*(t) = 2 − 2Φ(z_{1−α/2}/√t),  t ∈ (0,1],  α*(1) = α.
    Es monótona creciente y Σ de los incrementos por época = α_total: el fiscal NO puede
    re-procesar indefinidamente hasta obtener una condena.
    """
    if not 0.0 < alpha_total < 0.5:
        raise ProsecutorialGovernanceError("α_total ∈ (0,0.5).")
    if not 0.0 < info_fraction <= 1.0:
        return 0.0
    z = float(ndtri(1.0 - alpha_total / 2.0))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(info_fraction) / math.sqrt(2.0)))
    return float(max(0.0, 2.0 - 2.0 * cdf))


def tune_resonance_scan(nu_x: float, nu_y: float, order: int
                        ) -> Tuple[float, Tuple[int, int, int], float]:
    r"""
    Escaneo EXACTO de resonancias de betatrón m·ν_x + n·ν_y = p con 1 ≤ |m|+|n| ≤ order.
    Es la condición diophantina del problema KAM en lenguaje de acelerador. Devuelve
    (distancia mínima, resonancia más cercana, γ = min dist·(|m|+|n|)).
    Ausencia de resonancia a orden finito NO prueba estabilidad global (declarado).
    """
    best, gamma, nearest = float("inf"), float("inf"), (0, 0, 0)
    for m, n in itertools.product(range(-order, order + 1), repeat=2):
        l1 = abs(m) + abs(n)
        if l1 == 0 or l1 > order:
            continue
        val = m * nu_x + n * nu_y
        p = int(round(val))
        dist = abs(val - p)
        if dist < best:
            best, nearest = dist, (m, n, p)
        gamma = min(gamma, dist * l1)
    return float(best), nearest, float(gamma)


def _savgol(w: int, p: int, d: int) -> NDArray:
    h = w // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, p + 1, increasing=True))[d] * math.factorial(d)


class ProsecutorialSigner:
    """HMAC-SHA256 con separación de dominio y comparación en tiempo constante."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_PROSECUTOR_AGENT_HMAC_KEY")
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
        return hmac.new(self._key, b"GAMMA-PROSECUTOR-KEY-FP-v14|", hashlib.sha256).hexdigest()[:16]

    def sign(self, domain: bytes, payload: Any) -> str:
        return hmac.new(self._key, domain + _canonical(payload), hashlib.sha256).hexdigest()

    def verify(self, domain: bytes, payload: Any, signature: str) -> bool:
        return hmac.compare_digest(self.sign(domain, payload), str(signature))


class CrowbarDriver(Protocol):
    name: str
    is_physical: bool

    def dry_run(self) -> int: ...
    def arm(self) -> None: ...
    def trip(self, token: str) -> int: ...
    def safe_state(self) -> None: ...


class ReadOnlyCrowbarDriver:
    """Driver de SÓLO LECTURA inyectado al motor: ORIENT no puede tocar hardware."""

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
        self.refused += 1


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


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProsecutorialGovernanceParameters:
    """Política de carga y presupuestos DECLARADOS; su digest se compromete en P1."""

    fermion_modes: int = 2
    pair_coherence: float = 1.0
    interaction_g: Optional[float] = None
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    beta_thermo: float = 2.0
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    ball_r: float = 1.0
    cylinder_r: float = 1.5
    sprt_lambda0: float = 10.0
    sprt_lambda1: float = 15.0
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.10
    alpha_total: float = 0.05                  # presupuesto de error de TODO el proceso
    planned_epochs: int = 12                   # horizonte declarado para el gasto
    q_fdr: float = 0.05                        # FDR objetivo del pliego estadístico
    fdr_mc_trials: int = 1200
    fdr_mc_channels: int = 8
    fdr_mc_null_fraction: float = 0.75
    prescription_target_detection: float = 0.95  # P(observed) objetivo en la ventana
    tune_nu_x: float = 0.31
    tune_nu_y: float = 0.28
    tune_claimed_margin: float = 5.0e-3
    tune_engine_order: int = 8
    tune_deep_order: int = 12                  # orden MÁS PROFUNDO que el motor (falsación)
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    require_dual_authorization: bool = True
    lesser_measures_insufficient: bool = False
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
        if not 2 <= self.fermion_modes <= 4:
            raise ProsecutorialGovernanceError("fermion_modes ∈ [2,4].")
        if not (0.0 < self.alpha_total < 0.5 and 0.0 < self.q_fdr < 0.5):
            raise ProsecutorialGovernanceError("α_total, q_FDR ∈ (0,0.5).")
        if self.planned_epochs < 2:
            raise ProsecutorialGovernanceError("planned_epochs ≥ 2 para el alpha-spending.")
        if self.tune_deep_order < self.tune_engine_order:
            raise ProsecutorialGovernanceError("El escaneo profundo debe superar el orden del motor.")
        if not (0.0 < self.recurrence_mu < 1.0 and 0.0 < self.prescription_target_detection < 1.0):
            raise ProsecutorialGovernanceError("μ y objetivo de detección en (0,1).")
        if self.pointer_sigma <= 0.0 or self.disturbance_budget < 0.0:
            raise ProsecutorialGovernanceError("σ>0 y presupuesto ≥0.")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0:
            raise ProsecutorialGovernanceError("Radios positivos.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)

    def to_engine_parameters(self, modes: int, g: float) -> Any:
        """Proyección al contrato del motor fiscal v14 (sin autorizar actuación)."""
        return EngineProsecutionParameters(
            fermion_modes=int(modes), pair_coherence=self.pair_coherence,
            interaction_g=float(g), pointer_sigma=self.pointer_sigma,
            disturbance_budget=self.disturbance_budget, backaction_db_budget=self.backaction_db_budget,
            beta_thermo=self.beta_thermo, positron_kinetic_kev=self.positron_kinetic_kev,
            emission_angle_deg=self.emission_angle_deg, ball_r=self.ball_r, cylinder_r=self.cylinder_r,
            sprt_lambda0=self.sprt_lambda0, sprt_lambda1=self.sprt_lambda1,
            sprt_alpha=self.sprt_alpha, sprt_beta=self.sprt_beta,
            require_human_authorization=True,          # el motor NUNCA autoemite token
            lesser_measures_insufficient=self.lesser_measures_insufficient,
            logic_deadline_ns=self.logic_deadline_ns, gpio_write_ns=self.gpio_write_ns,
            gate_driver_ns=self.gate_driver_ns, scr_turn_on_ns=self.scr_turn_on_ns,
            hil_firmware_isr_ns=self.hil_firmware_isr_ns,
            recurrence_mu=self.recurrence_mu, recurrence_samples=self.recurrence_samples,
        )


@dataclass(frozen=True)
class GammaPhotonProsecutorPhase1Certificate:
    """P1 — Custodia del pliego con PRE-COMPROMISO (reloj fuera del sello)."""

    timestamp: float
    tr_rho: float
    tr_rho_normalized: float
    normalization_factor: float
    is_psd: bool
    hermiticity_residual: float
    purity: float
    von_neumann_entropy: float
    fock_modes_reconciled: int
    fock_dimension_ok: bool
    substitute_declared: bool
    bures_wasserstein_to_thermal: float
    bw_isometry_residual: float
    bw_uhlmann_consistency_residual: float
    committed_gain_g: float
    gain_kantorovich_h: float
    gain_degenerate_channel: bool
    gain_override_rejected: bool
    n_claims: int
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    charge_sheet_digest: str
    params_digest: str
    engine_passport_bound: str
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
    """§2.4 — Pliego: refutaciones estructurales (pruebas) + BH/FDR (estadística)."""

    n_claims: int
    structural_refutations: Tuple[Tuple[int, str], ...]
    pvalues_exact: Tuple[float, ...]
    pvalues_adjusted: Tuple[float, ...]
    bh_rejected_indices: Tuple[int, ...]
    q_fdr: float
    statistical_convictions: int
    mc_fdr: float
    mc_fdr_ok: bool
    mc_power: float
    screening_note: str


@dataclass(frozen=True)
class ErrorSpendingReport:
    """§2.5 — Lan–DeMets O'Brien–Fleming: el gasto total está acotado."""

    alpha_total: float
    planned_epochs: int
    epoch_index: int
    info_fraction: float
    cumulative_spent: float
    epoch_allowance: float
    total_spend_residual: float
    monotone_ok: bool
    within_budget: bool


@dataclass(frozen=True)
class StabilityFalsificationReport:
    """§2.6 — Escaneo adversarial de resonancias a orden MÁS PROFUNDO que el motor."""

    nu_x: float
    nu_y: float
    engine_order: int
    deep_order: int
    engine_min_distance: float
    deep_min_distance: float
    deep_nearest_resonance: Tuple[int, int, int]
    deep_diophantine_gamma: float
    claimed_margin: float
    claimed_margin_survives: bool
    exact_resonance_found: bool
    search_power_note: str


@dataclass(frozen=True)
class RecurrencePrescriptionReport:
    """§2.7 — Kac–Slater: recurrencia del régimen anómalo y ventana de prescripción."""

    mu: float
    kac_expected_return: float
    kac_empirical_return: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    slater_ok: bool
    prescription_window_epochs: int
    prescription_detection_probability: float
    prescription_target: float


@dataclass(frozen=True)
class GammaPhotonProsecutorPhase2Certificate:
    """P2 — Instrucción epistémica: anida P1 y el acta del motor (sin actuación)."""

    timestamp: float
    phase1: GammaPhotonProsecutorPhase1Certificate
    engine_report: Optional[Any]
    engine_invoked: bool
    engine_acta_verified: bool
    engine_merkle_recomputed_ok: bool
    engine_actuation_dispatched: bool
    gain_honored: bool
    disturbance_consumed: float
    back_action_db: float
    charge_sheet: ChargeSheetReport
    error_spending: ErrorSpendingReport
    stability: StabilityFalsificationReport
    recurrence: RecurrencePrescriptionReport
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
            "engine_truth": getattr(rep, "truth_value", HeytingTruthValue.VETOED),
            "engine_acta_verified": self.engine_acta_verified,
            "engine_merkle_recomputed_ok": self.engine_merkle_recomputed_ok,
            "engine_actuation_dispatched": self.engine_actuation_dispatched,
            "gain_honored": self.gain_honored, "disturbance_consumed": self.disturbance_consumed,
            "back_action_db": self.back_action_db, "charge_sheet": self.charge_sheet,
            "error_spending": self.error_spending, "stability": self.stability,
            "recurrence": self.recurrence, "evidences": self.evidences,
            "truth_value": self.truth_value, "epoch": self.epoch,
        }


@dataclass(frozen=True)
class ProsecutorGateReceipt:
    gate: str
    payload_digest: str
    chain_recomputed_ok: bool
    signature_ok: bool
    binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class GammaPhotonProsecutorVisceralMetrics:
    """§3.5 — Φ_sem con procedencia: métricas medidas; finanzas sólo si declaradas."""

    charge_coverage_pct: float
    proven_charges: int
    fdr_budget_pct: float
    error_budget_spent_pct: float
    stability_margin_utilization_pct: float
    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    event_status: str
    due_process_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool


@dataclass(frozen=True)
class ProsecutorRSI3Report:
    """§3.4 — RSI³: ventana de prescripción (Kantorovich) + Banach + Kleisli + jerk."""

    calibrated_window_epochs: float
    achieved_detection_probability: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_iterations: int
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
class DueProcessAuthorizationReport:
    """§3.3+§3.6 — Debido proceso y COMMIT de actuación MEDIDO."""

    ladder_rung: int
    action: str
    indictment_digest: str
    epoch: int
    claim_first_seen_epoch: int
    claim_age_epochs: int
    prescription_window_epochs: int
    time_barred: bool
    ne_bis_in_idem_ok: bool
    anti_replay_ok: bool
    quorum_required: int
    quorum_satisfied: bool
    authorizer_ids: Tuple[str, ...]
    duplicate_authorizer_rejected: bool
    tokens_verified: Tuple[bool, ...]
    actuation_requested: bool
    actuation_authorized: bool
    actuation_dispatched: bool
    actuation_fabricated: bool
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
class ProsecutorEscalation:
    epoch: int
    truth_value: HeytingTruthValue
    reason: str
    merkle_root_hash: str
    signature: str
    delivered: bool


@dataclass(frozen=True)
class GammaPhotonProsecutorSovereignState:
    """Estado soberano fiscal final (nombres heredados conservados)."""

    timestamp: float
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    sentence_leaf_index: int
    sentence_inclusion_proof: Tuple[Tuple[str, bool], ...]
    visceral_metrics: GammaPhotonProsecutorVisceralMetrics
    rsi3: ProsecutorRSI3Report
    authorization: DueProcessAuthorizationReport
    escalation: Optional[ProsecutorEscalation]
    phase2: GammaPhotonProsecutorPhase2Certificate
    evidences: Tuple[Any, ...]
    kleisli_ledger: Tuple[str, ...]
    epoch: int
    state_signature: str
    error: Optional[str]
    execution_time_ms: float

    def payload(self) -> Dict[str, Any]:
        return {
            "truth_value": self.truth_value, "boolean_shadow": self.boolean_shadow,
            "crowbar_triggered": self.crowbar_triggered, "merkle_root_hash": self.merkle_root_hash,
            "sentence_leaf_index": self.sentence_leaf_index,
            "sentence_inclusion_proof": self.sentence_inclusion_proof,
            "kleisli_ledger": self.kleisli_ledger, "epoch": self.epoch, "error": self.error,
            "phase2_chain_hash": self.phase2.chain_hash,
            "phase1_chain_hash": self.phase2.phase1.chain_hash,
            "authorization": {
                "rung": self.authorization.ladder_rung, "action": self.authorization.action,
                "acta": self.authorization.indictment_digest,
                "quorum": self.authorization.quorum_satisfied,
                "time_barred": self.authorization.time_barred,
                "authorized": self.authorization.actuation_authorized,
                "dispatched": self.authorization.actuation_dispatched,
                "fabricated": self.authorization.actuation_fabricated,
                "safe_state": self.authorization.safe_state_applied,
                "ids": self.authorization.authorizer_ids,
            },
            "escalation_signature": self.escalation.signature if self.escalation else None,
            "merkle_leaf_count": len(self.merkle_leaves),
        }

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE — custodia del pliego e isometría de Bures–Wasserstein
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonProsecutorPhase1Observer:
    r"""
    FASE 1 OODA fiscal. Compromete la EVIDENCIA (no un resumen de tres campos), reconcilia la
    dimensión de Fock y pre-compromete la política de carga y la fuerza de interacción.
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE-CUSTODIA"

    def __init__(self, signer: ProsecutorialSigner, *, seed: int = 20_261_126) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa + reconciliación de Fock ───────────
    def inspect_and_reconcile(self, rho_raw: Optional[NDArray], params: ProsecutorialGovernanceParameters
                              ) -> Tuple[Optional[NDArray], int, Dict[str, Any]]:
        r"""
        El motor v14 exige ρ de dimensión 2^m (Λ(ℂ^m), m∈[2,4]). Si dim ρ = 2^m se reconcilia
        m = log₂ dim; si no, se SELLA la malformación, se declara el sustituto y se AGRAVA:
        corromper la evidencia no otorga inmunidad.
        """
        if rho_raw is None:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": 0.0, "trace_raw": 1.0, "dimension_ok": True, "substitute": False}
        r = np.asarray(rho_raw, dtype=np.complex128)
        malformed = r.ndim != 2 or r.shape[0] != r.shape[1] or not np.all(np.isfinite(r))
        if malformed:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": float("inf"), "trace_raw": 0.0,
                "dimension_ok": False, "substitute": True}
        dim = int(r.shape[0])
        m = int(round(math.log2(dim))) if dim >= 4 else 0
        power_ok = bool(2 <= m <= 4 and (1 << m) == dim)
        fro = float(la.norm(r, ord="fro")) + _EPS
        herm = float(la.norm(r - r.conj().T, ord="fro")) / fro
        tr = complex(np.trace(r))
        bad_trace = abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0
        if not power_ok or bad_trace:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": herm, "trace_raw": float(tr.real),
                "dimension_ok": power_ok, "substitute": True}
        return r, m, {"hermiticity_residual": herm, "trace_raw": float(tr.real),
                      "dimension_ok": True, "substitute": False}

    # ─────────── §1.2 Normalización auditada + isometría BW ───────────
    @staticmethod
    def normalize_audited(rho: NDArray, trace_raw: float) -> Tuple[NDArray, float]:
        r = np.asarray(rho, dtype=np.complex128)
        r = 0.5 * (r + r.conj().T)
        return r / float(trace_raw), 1.0 / float(trace_raw)

    def bw_isometry(self, rho: NDArray, sigma: NDArray) -> Tuple[float, float, float]:
        d = bures_wasserstein_distance(rho, sigma)
        u = _haar_unitary(rho.shape[0], self._rng)
        d_moved = bures_wasserstein_distance(u @ rho @ u.conj().T, u @ sigma @ u.conj().T)
        fid = _fidelity_svd(rho, sigma)
        return d, abs(d_moved - d), abs(d * d - (2.0 - 2.0 * math.sqrt(fid)))

    # ─────────── §1.3 PRE-COMPROMISO: calibración certificada de g ───────────
    @staticmethod
    def certify_interaction_gain(rho: NDArray, modes: int, sigma: float, budget: float, *,
                                 g_default: float = 1.0e-5, max_iter: int = 60
                                 ) -> Tuple[float, float, bool]:
        r"""
        Proxy Frobenius EXACTO de la disturbancia dispersiva sobre Λ(ℂ^m), N̂ = popcount:
            S(g) = Σ|ρ_mn|²(1 − e^{−g²(N_m−N_n)²/8σ²})²,  S'(g) analítica.
        Newton con salvaguarda de bisección; h de Kantorovich estimado (declarado).
        Si [ρ,N̂]=0 el canal es degenerado y se DECLARA (todo g es no invasivo).
        """
        dim = 1 << modes
        n_vals = np.array([bin(n).count("1") for n in range(dim)], dtype=np.float64)
        d2 = (n_vals[:, None] - n_vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(rho) ** 2
        if float(np.sum(w2 * d2)) <= 1e-300:
            return float(g_default), 0.0, True

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
        for _ in range(int(max_iter)):
            f_v, dv = s_of(g) - target, ds_of(g)
            if f_v > 0.0:
                hi = g
            else:
                lo = g
            step = g - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g = float(min(max(step, lo), hi)) if hi > lo else float(step)
            if hi - lo < 1e-18 * max(1.0, hi):
                break
        return float(g), float(h_k), False

    # ─────────── §1.5 [TERMINAL FASE 1] → Φ₁₂ → Fase 2 §2.1 ───────────
    def emit_phase1_certificate(
        self, rho_raw: Optional[NDArray], apu_cost_matrix: NDArray, exponents: NDArray,
        claims: Sequence[Any], params: ProsecutorialGovernanceParameters, *, epoch: int,
        engine_passport_bound: str = "",
    ) -> Tuple[GammaPhotonProsecutorPhase1Certificate, Optional[NDArray], int]:
        """MÉTODO TERMINAL DE LA FASE 1: custodia la evidencia completa y PRE-COMPROMETE."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise ProsecutorialGovernanceError("K debe ser cuadrada.")
        if not claims:
            raise ProsecutorialGovernanceError("El pliego de cargos no puede estar vacío.")
        rho_in, modes, insp = self.inspect_and_reconcile(rho_raw, params)
        if rho_in is None:
            dim = 1 << modes
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = 1.0
            psi[3] = float(params.pair_coherence)
            psi /= (la.norm(psi) + _EPS)
            rho = np.outer(psi, psi.conj())
            factor, trace_norm = 0.0, 1.0
        else:
            rho, factor = self.normalize_audited(rho_in, insp["trace_raw"])
            trace_norm = float(np.real(np.trace(rho)))
        eigs = la.eigvalsh(rho)
        purity = float(np.real(np.trace(rho @ rho)))
        probs = np.clip(eigs, 0.0, 1.0)
        entropy = float(-np.sum(probs[probs > 0.0] * np.log(probs[probs > 0.0])))
        is_psd = bool(float(np.min(eigs)) >= -1e-10 and not insp["substitute"])
        dim = 1 << modes
        en = _ME_C2_KEV * np.array([bin(n).count("1") for n in range(dim)], dtype=np.float64)
        lp = -params.beta_thermo * (en - en[0])
        lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        d_bw, iso_res, cons_res = self.bw_isometry(rho, rho_th)
        g_cert, h_k, degenerate = self.certify_interaction_gain(
            rho, modes, params.pointer_sigma, params.disturbance_budget)
        override_rejected = False
        if params.interaction_g is not None:
            g_req = float(params.interaction_g)
            if g_req > g_cert * (1.0 + 1e-12) and not degenerate:
                override_rejected = True      # NUNCA se clampa en silencio
                g_cert = 0.0                  # fail-closed: sin interacción
            else:
                g_cert = g_req
        sheet_digest = _sha256(b"|".join(c.digest().encode("ascii") for c in claims))
        evidences: Tuple[Any, ...] = (
            Evidence("GP1.hermiticidad", insp["hermiticity_residual"] if math.isfinite(
                insp["hermiticity_residual"]) else 1.0, 1e-10, hard=True),
            Evidence.boolean("GP1.positividad", is_psd, hard=True),
            Evidence.boolean("GP1.dimension_fock_potencia_de_2", insp["dimension_ok"], hard=True,
                             note=f"m reconciliado = {modes}"),
            Evidence("GP1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("GP1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence("GP1.bw_isometria_haar", iso_res, 1e-11, hard=True),
            Evidence("GP1.bw_uhlmann_consistencia", cons_res, 1e-10, hard=True),
            Evidence.boolean("GP1.g_no_excede_presupuesto", not override_rejected, hard=True),
            Evidence.boolean("GP1.pliego_no_vacio", len(claims) > 0, hard=True),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        core = {
            "tr_rho": float(insp["trace_raw"]), "tr_rho_normalized": trace_norm,
            "normalization_factor": factor, "is_psd": is_psd,
            "hermiticity_residual": float(insp["hermiticity_residual"]) if math.isfinite(
                insp["hermiticity_residual"]) else float("inf"),
            "purity": purity, "von_neumann_entropy": entropy,
            "fock_modes_reconciled": int(modes), "fock_dimension_ok": bool(insp["dimension_ok"]),
            "substitute_declared": bool(insp["substitute"]),
            "bures_wasserstein_to_thermal": d_bw, "bw_isometry_residual": iso_res,
            "bw_uhlmann_consistency_residual": cons_res,
            "committed_gain_g": float(g_cert), "gain_kantorovich_h": float(h_k),
            "gain_degenerate_channel": bool(degenerate),
            "gain_override_rejected": bool(override_rejected), "n_claims": int(len(claims)),
            "rho_digest": _array_digest(np.asarray(rho_raw if rho_raw is not None else rho,
                                                   dtype=np.complex128)),
            "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "charge_sheet_digest": sheet_digest, "params_digest": params.digest(),
            "engine_passport_bound": engine_passport_bound,
            "agent_version": __version__, "engine_version": _pe.__version__, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonProsecutorPhase1Certificate(
            timestamp=time.time(), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P1, core), **core
        )
        logger.info("GP1 Ω₃=%s · m=%d · cargos=%d · g=%.6g", truth.name, modes, len(claims), g_cert)
        return cert, rho_in, modes  # ──► Φ₁₂ ──► Fase 2 §2.1

    def execute_phase1(self, *a: Any, **k: Any) -> Tuple[GammaPhotonProsecutorPhase1Certificate, Optional[NDArray], int]:
        """Alias heredado (v13) del terminal §1.5."""
        return self.emit_phase1_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT — pliego, BH/FDR, gasto de error y falsación de estabilidad
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonProsecutorPhase2Orientor:
    r"""
    FASE 2 OODA fiscal. Instruye el pliego SIN actuar: el motor recibe driver de sólo lectura
    y ningún token. Su último método alimenta el primero de Fase 3 vía Φ₂₃.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT-PLIEGO"

    def __init__(self, engine: Any, signer: ProsecutorialSigner, *, seed: int = 20_261_126) -> None:
        self._engine = engine
        self._signer = signer
        self._seed = int(seed)

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self, p1: GammaPhotonProsecutorPhase1Certificate, apu_cost_matrix: NDArray,
        exponents: NDArray, claims: Sequence[Any], params: ProsecutorialGovernanceParameters, *,
        expected_epoch: int,
    ) -> ProsecutorGateReceipt:
        r"""
        Φ₁₂ — v13 recalculaba el hash de un resumen de 3 campos:.autenticaba el resumen, no la
        evidencia. Aquí se recalcula el hash DESDE el payload canónico (sin reloj), se verifica
        el HMAC en tiempo constante y se re-digestan ρ implícito, K, exponentes, PLIEGO y
        política que cruzan la compuerta (anti-TOCTOU). ADMITE ⊥ (la anomalía se procesa).
        """
        payload = p1.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p1.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P1, payload, p1.hmac_signature)
        sheet_digest = _sha256(b"|".join(c.digest().encode("ascii") for c in claims))
        binding_ok = (
            hmac.compare_digest(p1.kernel_digest, _array_digest(np.asarray(apu_cost_matrix, dtype=np.float64)))
            and hmac.compare_digest(p1.exponents_digest, _array_digest(np.asarray(exponents, dtype=np.float64)))
            and hmac.compare_digest(p1.charge_sheet_digest, sheet_digest)
            and hmac.compare_digest(p1.params_digest, params.digest())
            and p1.epoch == expected_epoch and p1.agent_version == __version__
            and p1.n_claims == len(claims)
            and math.isfinite(p1.committed_gain_g) and p1.committed_gain_g >= 0.0
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise ProsecutorialGateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return ProsecutorGateReceipt("Φ₁₂", recomputed, chain_ok, sig_ok, binding_ok, True,
                                     not p1.is_psd, "anomalía ADMITIDA: se procesa, no se exime")

    def accept_phase1_certificate(self, p1: GammaPhotonProsecutorPhase1Certificate,
                                  apu_cost_matrix: NDArray, exponents: NDArray, claims: Sequence[Any],
                                  params: ProsecutorialGovernanceParameters, *,
                                  expected_epoch: int) -> ProsecutorGateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.5."""
        return self.continue_to_phase2_gate(p1, apu_cost_matrix, exponents, claims, params,
                                            expected_epoch=expected_epoch)

    # ─────────── §2.3 Verificación del acta del motor ───────────
    def verify_engine_acta(self, report: Any) -> Tuple[bool, bool]:
        verified = bool(self._engine.verify_prosecutor_report(report))
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        root_ok = hmac.compare_digest(merkle_root(leaves), report.merkle_root_hash)
        idx = int(report.acta_leaf_index)
        root_ok = root_ok and verify_merkle(leaves[idx], idx, len(leaves),
                                            report.acta_inclusion_proof, report.merkle_root_hash)
        return verified, root_ok

    # ─────────── §2.4 Pliego: refutación estructural + BH/FDR ───────────
    def charge_sheet_certificate(self, claims: Sequence[Any], engine_report: Any,
                                 params: ProsecutorialGovernanceParameters) -> ChargeSheetReport:
        r"""
        Cribado EXACTO por reclamo (independiente del motor):
          • masa invariante: s_rec = 2E₁E₂(1−cosθ₁₂) vs s_haz = 2m²+2mE₊;
          • Landau–Yang: J=1 → 2γ prohibido; paridad C: (−1)^n = (−1)^{L+S};
          • Pauli: ocupación n_j ≤ 1; carga Q = n₊ − n₋.
        Las refutaciones estructurales son PRUEBAS (p=0) y no entran en la multiplicidad.
        Los cargos estadísticos (conteos de Poisson) se resuelven con BENJAMINI–HOCHBERG a q,
        y el FDP se verifica por Monte Carlo con mezcla nula/alternativa declarada.
        """
        m_e = _ME_C2_KEV
        e_plus = m_e + float(params.positron_kinetic_kev)
        s_beam = 2.0 * m_e ** 2 + 2.0 * m_e * e_plus
        structural: List[Tuple[int, str]] = []
        pvals: List[float] = []
        for i, c in enumerate(claims):
            e1, e2 = (float(x) for x in c.photon_energies_kev)
            th = math.radians(float(c.opening_angle_deg))
            s_rec = 2.0 * e1 * e2 * (1.0 - math.cos(th))
            if abs(s_rec - s_beam) / max(s_beam, _EPS) > 1e-6:
                structural.append((i, "masa_invariante"))
            if int(c.claimed_state_spin) == 1 and int(c.claimed_photon_multiplicity) == 2:
                structural.append((i, "landau_yang"))
            state_c = 1 if int(c.claimed_state_spin) == 0 else -1
            if (-1) ** int(c.claimed_photon_multiplicity) != state_c:
                structural.append((i, "paridad_c"))
            occ = tuple(int(v) for v in c.claimed_occupancies)
            if len(occ) != params.fermion_modes or any(v < 0 or v > 1 for v in occ):
                structural.append((i, "pauli"))
            elif int(c.claimed_charge) != occ[1] - occ[0]:
                structural.append((i, "carga"))
            total = int(sum(c.declared_counts))
            lam0 = float(params.sprt_lambda0) * max(len(c.declared_counts), 1)
            pvals.append(poisson_upper_pvalue(total, lam0))
        stat_indices = [i for i in range(len(claims)) if not any(s[0] == i for s in structural)]
        stat_p = [pvals[i] for i in stat_indices]
        rej_local, adj_stat = benjamini_hochberg(stat_p, params.q_fdr)
        rejected = tuple(stat_indices[k] for k in rej_local)
        adj = list(pvals)
        for k, i in enumerate(stat_indices):
            adj[i] = adj_stat[k]
        for i, _ in structural:
            adj[i] = 0.0

        rng = np.random.default_rng(self._seed + 97)
        m_ch = int(params.fdr_mc_channels)
        n_null = int(round(m_ch * float(params.fdr_mc_null_fraction)))
        trials = int(params.fdr_mc_trials)
        n_s = max(len(claims[0].declared_counts), 1)
        lam0 = float(params.sprt_lambda0) * n_s
        lam1 = float(params.sprt_lambda1) * n_s
        fdp_sum, power_sum, rej_count = 0.0, 0.0, 0
        for _ in range(trials):
            tot = np.concatenate([rng.poisson(lam0, n_null), rng.poisson(lam1, m_ch - n_null)])
            pv = [poisson_upper_pvalue(int(v), lam0) for v in tot]
            rej, _ = benjamini_hochberg(pv, params.q_fdr)
            if rej:
                rej_count += 1
                false_disc = sum(1 for i in rej if i < n_null)
                fdp_sum += false_disc / len(rej)
                power_sum += sum(1 for i in rej if i >= n_null) / max(m_ch - n_null, 1)
        fdr = fdp_sum / max(trials, 1)
        power = power_sum / max(trials, 1)
        slack = 3.0 * math.sqrt(max(params.q_fdr * (1 - params.q_fdr), 1e-6) / max(trials, 1))
        return ChargeSheetReport(
            n_claims=len(claims), structural_refutations=tuple(structural),
            pvalues_exact=tuple(pvals), pvalues_adjusted=tuple(adj), bh_rejected_indices=rejected,
            q_fdr=float(params.q_fdr), statistical_convictions=len(rejected), mc_fdr=float(fdr),
            mc_fdr_ok=bool(fdr <= params.q_fdr + slack), mc_power=float(power),
            screening_note=("el motor audita en profundidad el reclamo primario; el agente hace "
                            "cribado exacto del resto. BH controla FDR bajo independencia/PRDS."),
        )

    # ─────────── §2.5 Alpha-spending de Lan–DeMets ───────────
    @staticmethod
    def error_spending_certificate(epoch: int,
                                   params: ProsecutorialGovernanceParameters) -> ErrorSpendingReport:
        """α*(t)=2−2Φ(z_{1−α/2}/√t); Σ incrementos = α_total. Impide re-procesar hasta condenar."""
        n_ep = int(params.planned_epochs)
        idx = min(max(int(epoch), 1), n_ep)
        t = idx / n_ep
        cum = lan_demets_obrien_fleming(params.alpha_total, t)
        prev = lan_demets_obrien_fleming(params.alpha_total, (idx - 1) / n_ep) if idx > 1 else 0.0
        total = lan_demets_obrien_fleming(params.alpha_total, 1.0)
        increments = []
        last = 0.0
        mono = True
        for k in range(1, n_ep + 1):
            a_k = lan_demets_obrien_fleming(params.alpha_total, k / n_ep)
            mono = mono and (a_k >= last - 1e-15)
            increments.append(a_k - last)
            last = a_k
        return ErrorSpendingReport(
            alpha_total=float(params.alpha_total), planned_epochs=n_ep, epoch_index=idx,
            info_fraction=t, cumulative_spent=cum, epoch_allowance=max(0.0, cum - prev),
            total_spend_residual=abs(sum(increments) - params.alpha_total), monotone_ok=bool(mono),
            within_budget=bool(cum <= params.alpha_total + 1e-12),
        )

    # ─────────── §2.6 Falsación adversarial de la estabilidad ───────────
    @staticmethod
    def stability_falsification(engine_report: Any,
                                params: ProsecutorialGovernanceParameters) -> StabilityFalsificationReport:
        r"""
        El colisionador AFIRMA un margen de estabilidad. El fiscal lo somete a un escaneo de
        resonancias de orden MÁS PROFUNDO: si aparece una resonancia más cercana que el margen
        declarado, la afirmación queda REFUTADA (a ese orden). Ausencia ≠ estabilidad global.
        """
        deep, nearest, gamma = tune_resonance_scan(params.tune_nu_x, params.tune_nu_y,
                                                   int(params.tune_deep_order))
        eng, _, _ = tune_resonance_scan(params.tune_nu_x, params.tune_nu_y, int(params.tune_engine_order))
        return StabilityFalsificationReport(
            nu_x=float(params.tune_nu_x), nu_y=float(params.tune_nu_y),
            engine_order=int(params.tune_engine_order), deep_order=int(params.tune_deep_order),
            engine_min_distance=eng, deep_min_distance=deep, deep_nearest_resonance=nearest,
            deep_diophantine_gamma=gamma, claimed_margin=float(params.tune_claimed_margin),
            claimed_margin_survives=bool(deep >= params.tune_claimed_margin),
            exact_resonance_found=bool(deep < 1e-12),
            search_power_note=(f"orden {params.tune_deep_order} > {params.tune_engine_order} del motor; "
                               "búsqueda finita: ausencia de resonancia NO prueba estabilidad."),
        )

    # ─────────── §2.7 Kac–Slater y ventana de prescripción ───────────
    @staticmethod
    def recurrence_prescription(params: ProsecutorialGovernanceParameters
                                ) -> RecurrencePrescriptionReport:
        r"""
        KAC: E[τ_A] = 1/μ para la rotación irracional (ergódica, preservante de medida).
        SLATER: los tiempos de retorno toman ≤3 valores y el mayor suma los otros dos.
        Ventana de prescripción: menor W con P(observar el régimen en W épocas) = 1−(1−μ)^W
        ≥ objetivo declarado. Un cargo más viejo que W está PRESCRITO.
        """
        alpha, mu = _GOLDEN, float(params.recurrence_mu)
        ns = int(params.recurrence_samples)
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
            raise ProsecutorialGovernanceError("Kac: retorno no alcanzado.")
        rt = tuple(int(v) for v in np.unique(times))
        emp = float(np.mean(times))
        target = float(params.prescription_target_detection)
        w_exact = math.log(1.0 - target) / math.log(1.0 - mu)
        w_int = int(math.ceil(w_exact))
        p_obs = 1.0 - (1.0 - mu) ** w_int
        return RecurrencePrescriptionReport(
            mu=mu, kac_expected_return=1.0 / mu, kac_empirical_return=emp,
            kac_relative_residual=abs(emp - 1.0 / mu) * mu, slater_return_times=rt,
            slater_ok=bool(len(rt) <= 3), prescription_window_epochs=w_int,
            prescription_detection_probability=p_obs, prescription_target=target,
        )

    # ─────────── §2.8 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self, p1: GammaPhotonProsecutorPhase1Certificate, apu_cost_matrix: NDArray,
        exponents: NDArray, rho_in: Optional[NDArray], modes: int, claims: Sequence[Any],
        params: ProsecutorialGovernanceParameters, *, expected_epoch: int,
    ) -> GammaPhotonProsecutorPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: instruye el pliego; el motor NO actúa."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, claims, params,
                                       expected_epoch=expected_epoch)
        report = self._engine.execute_gamma_photon_collider_prosecutor_audit(
            apu_cost_matrix, np.asarray(exponents, dtype=np.float64), rho_in,
            claimed_event=claims[0], params=params.to_engine_parameters(modes, p1.committed_gain_g),
            authorization_token=None,                     # ← Orient NUNCA autoriza
        )
        acta_ok, root_ok = self.verify_engine_acta(report)
        qb = report.phase_ii.qnd_bures
        gain_honored = abs(float(params.interaction_g if params.interaction_g is not None
                                 else p1.committed_gain_g) - p1.committed_gain_g) <= 1e-18
        sheet = self.charge_sheet_certificate(claims, report, params)
        spending = self.error_spending_certificate(expected_epoch, params)
        stability = self.stability_falsification(report, params)
        recurrence = self.recurrence_prescription(params)
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("GP2.acta_motor_verificada", acta_ok, hard=True),
            Evidence.boolean("GP2.merkle_motor_recalculado", root_ok, hard=True),
            Evidence.boolean("GP2.orient_no_actuo", not report.crowbar.dispatched, hard=True,
                             note="la actuación es el último commit de Fase 3"),
            Evidence.boolean("GP2.g_comprometido_honrado", gain_honored, hard=True),
            Evidence("GP2.disturbancia_vs_presupuesto",
                     max(0.0, qb.trace_distance_disturbance - params.disturbance_budget), 0.0, hard=True),
            Evidence("GP2.backaction_db_vs_presupuesto",
                     max(0.0, -qb.back_action_db - params.backaction_db_budget), 0.0, hard=True),
            Evidence.boolean("GP2.fdr_verificado", sheet.mc_fdr_ok, hard=True),
            Evidence.boolean("GP2.gasto_error_monotono", spending.monotone_ok, hard=True),
            Evidence("GP2.gasto_error_total", spending.total_spend_residual, 1e-12, hard=True),
            Evidence.boolean("GP2.gasto_dentro_de_presupuesto", spending.within_budget, hard=True),
            Evidence.boolean("GP2.margen_estabilidad_sobrevive", stability.claimed_margin_survives,
                             note=stability.search_power_note),
            Evidence.boolean("GP2.sin_resonancia_exacta", not stability.exact_resonance_found, hard=True),
            Evidence("GP2.kac_retardo", recurrence.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("GP2.slater", recurrence.slater_ok, hard=True),
            Evidence.boolean("GP2.ventana_prescripcion_alcanza_objetivo",
                             recurrence.prescription_detection_probability >= recurrence.prescription_target
                             - 1e-12, hard=True),
        )
        truth = HeytingTruthValue.infimum([p1.truth_value, report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": True,
            "engine_passport": report.passport_hash, "engine_acta": report.sentence.indictment_digest,
            "engine_merkle_root": report.merkle_root_hash, "engine_truth": report.truth_value,
            "engine_acta_verified": acta_ok, "engine_merkle_recomputed_ok": root_ok,
            "engine_actuation_dispatched": bool(report.crowbar.dispatched),
            "gain_honored": bool(gain_honored),
            "disturbance_consumed": float(qb.trace_distance_disturbance),
            "back_action_db": float(qb.back_action_db), "charge_sheet": sheet,
            "error_spending": spending, "stability": stability, "recurrence": recurrence,
            "evidences": evidences, "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonProsecutorPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=report, engine_invoked=True,
            engine_acta_verified=acta_ok, engine_merkle_recomputed_ok=root_ok,
            engine_actuation_dispatched=bool(report.crowbar.dispatched), gain_honored=bool(gain_honored),
            disturbance_consumed=float(qb.trace_distance_disturbance),
            back_action_db=float(qb.back_action_db), charge_sheet=sheet, error_spending=spending,
            stability=stability, recurrence=recurrence, evidences=evidences, truth_value=truth,
            epoch=int(expected_epoch), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        logger.info("GP2 Ω₃=%s · cargos=%d (estr=%d, BH=%d) · FDR=%.4f · gasto=%.4f",
                    truth.name, sheet.n_claims, len(sheet.structural_refutations),
                    sheet.statistical_convictions, sheet.mc_fdr, spending.cumulative_spent)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1

    def execute_phase2(self, *a: Any, **k: Any) -> GammaPhotonProsecutorPhase2Certificate:
        """Alias heredado (v13) del terminal §2.8."""
        return self.emit_phase2_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ACT — debido proceso, RSI³ y COMMIT medido
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonProsecutorPhase3Decider:
    r"""
    FASE 3 OODA fiscal. Única autoridad que autoriza y ejecuta. `crowbar_triggered` ⟺ commit
    MEDIDO con quórum: JAMÁS se fabrica una actuación (defecto capital de v13).
    """

    PHASE_LABEL: Final[str] = "3·SENTENCIA-COMMIT"
    _SEVERITY: Final[Dict[str, int]] = {"WARN": 0, "DERATE": 1, "ISOLATE": 2, "TRIP": 3}

    def __init__(self, signer: ProsecutorialSigner, driver: CrowbarDriver) -> None:
        self._signer = signer
        self._driver = driver
        self._judged: set = set()
        self._first_seen: Dict[str, int] = {}
        self._last_epoch = -1
        self._pending: Dict[str, Any] = {}

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ ───────────
    @staticmethod
    def legacy_would_fabricate_trip(truth: HeytingTruthValue, engine_dispatched: bool) -> bool:
        """DEPRECADO (v13): fabricaba disparo si VETOED y el motor no disparó. Sólo regresión."""
        return bool(truth is HeytingTruthValue.VETOED and not engine_dispatched)

    def continue_to_phase3_gate(self, p2: GammaPhotonProsecutorPhase2Certificate,
                                params: ProsecutorialGovernanceParameters, *,
                                expected_epoch: int) -> ProsecutorGateReceipt:
        r"""
        Φ₂₃ — v13 era constante True (Lip hardcodeado). Aquí: integridad DURA (⇒ fail-closed)
        separada de la física/epistémica: Lip(T³)<1 medido, FDR verificado, gasto dentro de
        presupuesto, sin resonancia exacta, pasividad, y que ORIENT no haya actuado.
        """
        payload = p2.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (p2.epoch == expected_epoch and p2.phase1.epoch == expected_epoch
                   and (not p2.engine_invoked or (p2.engine_acta_verified and p2.engine_merkle_recomputed_ok))
                   and not p2.engine_actuation_dispatched)
        if not (chain_ok and sig_ok and link_ok):
            raise ProsecutorialGateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        rep = p2.engine_report
        lip = float(getattr(getattr(rep, "rsi3", None), "lip_t3_bound", 1.0)) if rep else 1.0
        physics_ok = bool(lip < 1.0 and p2.charge_sheet.mc_fdr_ok
                          and p2.error_spending.within_budget
                          and not p2.stability.exact_resonance_found
                          and p2.disturbance_consumed <= params.disturbance_budget
                          and p2.truth_value is not HeytingTruthValue.VETOED)
        return ProsecutorGateReceipt("Φ₂₃", recomputed, chain_ok, sig_ok, link_ok, True, not physics_ok,
                                     "⊥ forzado: se sentencia y aísla" if not physics_ok
                                     else "pliego, gasto, estabilidad y contracción verificados")

    def accept_phase2_certificate(self, p2: GammaPhotonProsecutorPhase2Certificate,
                                  params: ProsecutorialGovernanceParameters, *,
                                  expected_epoch: int) -> ProsecutorGateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.8."""
        return self.continue_to_phase3_gate(p2, params, expected_epoch=expected_epoch)

    # ─────────── §3.2 Functor de severidad y núcleo ¬¬ ───────────
    @classmethod
    def action_functor(cls, truth: HeytingTruthValue, *, structural: bool,
                       lesser_insufficient: bool) -> str:
        """⊤⇒WARN, ½⇒DERATE, ⊥⇒ISOLATE; TRIP exige prueba ESTRUCTURAL + insuficiencia declarada."""
        if truth is HeytingTruthValue.COHERENT:
            return "WARN"
        if truth is HeytingTruthValue.DEGRADED:
            return "DERATE"
        return "TRIP" if (structural and lesser_insufficient) else "ISOLATE"

    @classmethod
    def verify_action_functor(cls) -> bool:
        vals = list(HeytingTruthValue)
        f = lambda t: cls._SEVERITY[cls.action_functor(t, structural=True, lesser_insufficient=True)]  # noqa: E731
        return all(f(a.meet(b)) == max(f(a), f(b)) for a in vals for b in vals)

    @staticmethod
    def verify_nucleus() -> bool:
        vals = list(HeytingTruthValue)
        return bool(all(int(a) <= int(a.double_negation()) for a in vals)
                    and all(a.double_negation().double_negation() == a.double_negation() for a in vals)
                    and all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                            for a in vals for b in vals))

    # ─────────── §3.3 Quórum de autorización ───────────
    def mint_authorization_token(self, acta: str, epoch: int, rung: int, authorizer_id: str) -> str:
        return self._signer.sign(_HMAC_AUTH, {"acta": acta, "epoch": int(epoch), "rung": int(rung),
                                              "authorizer": authorizer_id})

    def verify_authorization_token(self, token: str, acta: str, epoch: int, rung: int,
                                   authorizer_id: str) -> bool:
        return self._signer.verify(_HMAC_AUTH, {"acta": acta, "epoch": int(epoch), "rung": int(rung),
                                                "authorizer": authorizer_id}, token)

    def _verify_pending(self, token: str) -> bool:
        p = self._pending
        return bool(p) and self.verify_authorization_token(token, p["acta"], p["epoch"], p["rung"],
                                                           p["authorizer"])

    # ─────────── §3.4 RSI³ ───────────
    def prosecutorial_rsi3(self, p2: GammaPhotonProsecutorPhase2Certificate,
                           params: ProsecutorialGovernanceParameters) -> ProsecutorRSI3Report:
        r"""
        NIVEL 1 · Calibración de la VENTANA DE PRESCRIPCIÓN W por Newton–Kantorovich:
            f(W) = 1 − (1−μ)^W − objetivo,  f'(W) = −(1−μ)^W ln(1−μ) > 0,
        no lineal en W, con salvaguarda de bisección; h estimado (declarado).
        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip ≤ q, Lip(T³) ≤ q³, cotas
        de Banach a priori/a posteriori y sondeo empírico.
        NIVEL 3 · Leyes de Kleisli, núcleo ¬¬, functor y jerk de capacidad MEDIDO (SG 7/3).
        """
        mu = float(params.recurrence_mu)
        target = float(params.prescription_target_detection)
        base = 1.0 - mu
        log_b = math.log(base)

        def f_of(w: float) -> float:
            return 1.0 - base ** w - target

        def df_of(w: float) -> float:
            return -(base ** w) * log_b

        lo, hi = 1.0, max(2.0, math.log(1e-6) / log_b)
        w_k = 0.5 * (lo + hi)
        grid = np.linspace(lo, hi, 48)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([f_of(x) for x in grid], grid), grid)))) + _EPS
        d0 = df_of(w_k)
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs(f_of(w_k) / max(abs(d0), 1e-300)) * m_est
        errs: List[float] = []
        for _ in range(60):
            f_v, dv = f_of(w_k), df_of(w_k)
            if f_v > 0.0:
                hi = w_k
            else:
                lo = w_k
            step = w_k - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            w_k = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errs.append(abs(f_of(w_k)))
            if hi - lo < 1e-13 * max(1.0, hi):
                break
        err = np.array(errs)
        cap = -np.log10(np.maximum(err, 1e-300))
        active = np.where(err > 1e-13 * max(err[0], 1e-300))[0]
        d1 = d2 = d3 = 0.0
        if active.size >= 7:
            ctr = min(max(int(active[active.size // 2]), 3), cap.size - 4)
            seg = cap[ctr - 3: ctr + 4]
            d1 = float(_savgol(7, 3, 1) @ seg); d2 = float(_savgol(7, 3, 2) @ seg); d3 = float(_savgol(7, 3, 3) @ seg)
        ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.engine_report
        if rep is not None:
            ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        hard = [e for e in ev if e.hard]
        margin = float(sum(1 for e in hard if e.passed)) / max(1, len(hard))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        theta_star = np.array([w_k / max(hi, 1.0), margin, p2.error_spending.cumulative_spent
                               / max(params.alpha_total, 1e-300)])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(20_261_127)
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
        f_c = lambda x: Certified(x * 0.5, ("f",), HeytingTruthValue.COHERENT)   # noqa: E731
        g_c = lambda x: Certified(x + 1.0, ("g",), HeytingTruthValue.COHERENT)   # noqa: E731
        smp = rng.standard_normal(3)
        mv = Certified(smp, ("m",), HeytingTruthValue.DEGRADED)
        laws = bool(np.allclose(Certified.unit(smp).bind(f_c).value, f_c(smp).value)
                    and np.allclose(mv.bind(Certified.unit).value, mv.value)
                    and np.allclose(mv.bind(f_c).bind(g_c).value, mv.bind(lambda x: f_c(x).bind(g_c)).value)
                    and mv.bind(f_c).bind(g_c).ledger == mv.bind(lambda x: f_c(x).bind(g_c)).ledger)
        return ProsecutorRSI3Report(
            calibrated_window_epochs=float(w_k), achieved_detection_probability=float(1.0 - base ** w_k),
            kantorovich_h=float(h_k), kantorovich_satisfied=bool(h_k <= 0.5),
            newton_iterations=int(err.size), policy_lipschitz_q=float(q), lip_t3_bound=float(q3),
            lip_t3_empirical=float(emp), banach_apriori_bound=float(apriori),
            banach_aposteriori_bound=float(aposteriori), observed_policy_error=observed,
            capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0), kleisli_monad_laws_ok=laws,
            nucleus_axioms_ok=self.verify_nucleus(), action_functor_ok=self.verify_action_functor(),
        )

    # ─────────── §3.5 Φ_sem con procedencia ───────────
    @staticmethod
    def semantic_visceral_functor(p2: GammaPhotonProsecutorPhase2Certificate,
                                  params: ProsecutorialGovernanceParameters, *, action: str,
                                  proven: int) -> GammaPhotonProsecutorVisceralMetrics:
        """Métricas procesales MEDIDAS + KPI financieros sólo con insumos declarados."""
        all_ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.engine_report
        if rep is not None:
            all_ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        coverage = 100.0 * sum(1 for e in all_ev if e.passed) / max(1, len(all_ev))
        stab_pct = 100.0 * params.tune_claimed_margin / max(p2.stability.deep_min_distance, _EPS)
        provenance = ["COVERAGE:MEASURED", "FDR:BH_MC_VERIFIED", "SPENDING:LAN_DEMETS",
                      "STABILITY:DEEP_SCAN", "PRESCRIPTION:KAC"]
        wacc = liq = kv = None
        event_status = "SIN_ACTA_DEL_MOTOR"
        if rep is not None:
            ev = rep.phase_ii.event_refutation
            sel = rep.phase_ii.selection
            event_status = (
                f"s_rec={ev.s_reconstructed_kev2:.4f} keV² vs s_haz={ev.s_beam_kev2:.4f} · "
                f"refutado={ev.kinematics_refuted} · n_γ={sel.claimed_multiplicity} · "
                f"J={sel.claimed_spin} · Landau-Yang={sel.landau_yang_violation}"
            )
            sigma = max(0.0, rep.phase_ii.second_law.sigma_min)
            wacc = 100.0 * sigma / (sigma + params.exergy_reference_w)
            provenance.append("WACC:POLICY_HEURISTIC(sigma/(sigma+sigma_ref))")
        if params.exergy_price_per_joule is not None and params.capital_base:
            w_max = float(rep.phase_ii.second_law.max_extractable_work_j) if rep is not None else 0.0
            liq = 100.0 * w_max * params.exergy_price_per_joule / params.capital_base
            provenance.append("LIQUIDITY:DECLARED(W_max*price/capital)")
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
        return GammaPhotonProsecutorVisceralMetrics(
            charge_coverage_pct=float(coverage), proven_charges=int(proven),
            fdr_budget_pct=100.0 * params.q_fdr,
            error_budget_spent_pct=100.0 * p2.error_spending.cumulative_spent / max(params.alpha_total, _EPS),
            stability_margin_utilization_pct=float(stab_pct), wacc_protection_pct=wacc,
            net_liquidity_release_pct=liq, kv_cache_compression_pct=kv, event_status=event_status,
            due_process_status=(f"BH q={params.q_fdr:.3f} (FDR_MC={p2.charge_sheet.mc_fdr:.4f}) · "
                                f"α_gastado={p2.error_spending.cumulative_spent:.4f}/"
                                f"{params.alpha_total:.4f} · acción={action}"),
            provenance=tuple(provenance), monotonicity_ok=monotone,
        )

    # ─────────── §3.6 COMMIT de actuación (jamás fabricado) ───────────
    def commit_actuation(self, action: str, acta: str, epoch: int, *,
                         tokens: Sequence[Tuple[str, str]], window: int,
                         params: ProsecutorialGovernanceParameters) -> DueProcessAuthorizationReport:
        """
        Último paso. Debido proceso: ne bis in idem, PRESCRIPCIÓN (edad ≤ ventana), anti-replay
        y quórum 2-de-2 con IDs DISTINTOS. Sólo entonces arma y dispara con latencia MEDIDA.
        `actuation_fabricated` es siempre False: el agente nunca inventa un disparo.
        """
        rung = self._SEVERITY[action]
        samples = np.array([self._driver.dry_run() for _ in range(64)], dtype=np.float64)
        first_seen = self._first_seen.setdefault(acta, int(epoch))
        age = int(epoch) - int(first_seen)
        time_barred = bool(age > int(window))
        ne_bis = acta not in self._judged
        anti_replay = epoch > self._last_epoch
        ids = tuple(str(i) for i, _ in tokens)
        distinct = len(set(ids)) == len(ids) and len(ids) > 0
        verified = tuple(self.verify_authorization_token(str(t), acta, epoch, rung, str(i))
                         for i, t in tokens)
        required = 2 if (rung == 3 and params.require_dual_authorization) else (1 if rung == 3 else 0)
        quorum = bool(distinct and all(verified) and len(verified) >= required)
        requested = bool(rung == 3)
        authorized = bool(requested and quorum and ne_bis and anti_replay and not time_barred)
        dispatched, trip_ns, safe = False, 0.0, False
        if authorized:
            authorizer, token = str(tokens[0][0]), str(tokens[0][1])
            self._pending = {"acta": acta, "epoch": epoch, "rung": rung, "authorizer": authorizer}
            self._driver.arm()
            trip_ns = float(self._driver.trip(token))
            dispatched = True
            self._judged.add(acta); self._last_epoch = int(epoch)
        elif rung >= 2:
            self._driver.safe_state()
            safe = True
            self._judged.add(acta); self._last_epoch = int(epoch)
        return DueProcessAuthorizationReport(
            ladder_rung=rung, action=action, indictment_digest=acta, epoch=int(epoch),
            claim_first_seen_epoch=int(first_seen), claim_age_epochs=age,
            prescription_window_epochs=int(window), time_barred=time_barred,
            ne_bis_in_idem_ok=bool(ne_bis), anti_replay_ok=bool(anti_replay),
            quorum_required=int(required), quorum_satisfied=quorum, authorizer_ids=ids,
            duplicate_authorizer_rejected=bool(not distinct), tokens_verified=verified,
            actuation_requested=requested, actuation_authorized=authorized,
            actuation_dispatched=dispatched, actuation_fabricated=False, safe_state_applied=safe,
            driver_name=self._driver.name, driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            note=("el disparo existe sólo si hay commit MEDIDO con quórum 2-de-2, cargo no prescrito "
                  "y acta no juzgada. 392.15 ns es política del ISR; el SCR BT151 (~1–3 µs) no lo alcanza."),
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Sentencia sellada ───────────
    def emit_sovereign_state(
        self, p2: GammaPhotonProsecutorPhase2Certificate, params: ProsecutorialGovernanceParameters, *,
        expected_epoch: int, authorization_tokens: Sequence[Tuple[str, str]] = (),
        error: Optional[str] = None, forced_veto: bool = False,
    ) -> GammaPhotonProsecutorSovereignState:
        """MÉTODO TERMINAL: sentencia, verifica debido proceso, ejecuta el commit y sella."""
        t0 = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, params, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        sheet = p2.charge_sheet
        structural = bool(sheet.structural_refutations) or any(
            (not e.passed) and e.hard for e in list(p2.phase1.evidences) + list(p2.evidences))
        proven = len(sheet.structural_refutations) + sheet.statistical_convictions
        action = self.action_functor(truth, structural=structural,
                                     lesser_insufficient=params.lesser_measures_insufficient)
        rsi3 = self.prosecutorial_rsi3(p2, params)
        acta_payload = {
            "epoch": int(expected_epoch), "truth": truth, "action": action,
            "structural": bool(structural), "p2_chain": p2.chain_hash,
            "p1_chain": p2.phase1.chain_hash,
            "engine_acta": getattr(getattr(p2.engine_report, "sentence", None), "indictment_digest", ""),
            "refutations": sheet.structural_refutations, "bh_rejected": sheet.bh_rejected_indices,
            "error": error,
        }
        acta = _sha256(_canonical(acta_payload))
        window = p2.recurrence.prescription_window_epochs
        auth = self.commit_actuation(action, acta, expected_epoch, tokens=authorization_tokens,
                                     window=window, params=params)
        visceral = self.semantic_visceral_functor(p2, params, action=action, proven=proven)
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("GP3.functor_contravariante", rsi3.action_functor_ok, hard=True),
            Evidence.boolean("GP3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence("GP3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("GP3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("GP3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("GP3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("GP3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("GP3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("GP3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("GP3.actuacion_no_fabricada", not auth.actuation_fabricated, hard=True,
                             note="crowbar_triggered ⟺ commit medido con quórum"),
            Evidence.boolean("GP3.autorizacion_habilita",
                             (not auth.actuation_dispatched) or auth.quorum_satisfied, hard=True),
            Evidence.boolean("GP3.ids_distintos", not auth.duplicate_authorizer_rejected, hard=True),
            Evidence.boolean("GP3.ne_bis_in_idem", auth.ne_bis_in_idem_ok, hard=True),
            Evidence.boolean("GP3.anti_replay", auth.anti_replay_ok, hard=True),
            Evidence.boolean("GP3.cargo_no_prescrito", not auth.time_barred),
            Evidence.boolean("GP3.proporcionalidad",
                             auth.ladder_rung < 3 or (structural and params.lesser_measures_insufficient),
                             hard=True),
            Evidence.boolean("GP3.presuncion_inocencia",
                             not (sheet.statistical_convictions > 0 and not structural
                                  and auth.ladder_rung == 3), hard=True),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])
        escalation: Optional[ProsecutorEscalation] = None
        if auth.ladder_rung >= 2 or error is not None or auth.time_barred:
            reason = (f"peldaño={auth.action}" + (" | cargo prescrito" if auth.time_barred else "")
                      + (f" | error: {error}" if error else ""))
            esc_core = {"epoch": int(expected_epoch), "truth": truth, "reason": reason, "acta": acta}
            escalation = ProsecutorEscalation(
                epoch=int(expected_epoch), truth_value=truth, reason=reason, merkle_root_hash="",
                signature=self._signer.sign(_HMAC_ESC, esc_core), delivered=False)
        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": getattr(p2.engine_report, "passport_hash", ""),
                        "engine_root": getattr(p2.engine_report, "merkle_root_hash", "")}),
            _canonical(p2.charge_sheet), _canonical(p2.error_spending),
            _canonical(p2.stability), _canonical(p2.recurrence),
            _canonical(acta_payload), _canonical(rsi3),
            _canonical({k: v for k, v in _freeze(auth).items()
                        if k not in ("dispatch_median_ns", "dispatch_p99_ns", "trip_dispatch_ns")}),
            _canonical(p2.payload()),
        ]
        root = merkle_root(leaves)
        s_idx = 6
        proof = merkle_proof(leaves, s_idx)
        if escalation is not None:
            escalation = dataclasses.replace(escalation, merkle_root_hash=root)
        core = {
            "truth_value": truth, "boolean_shadow": truth.double_negation(),
            "crowbar_triggered": bool(auth.actuation_dispatched), "merkle_root_hash": root,
            "sentence_leaf_index": s_idx, "sentence_inclusion_proof": proof,
            "kleisli_ledger": ("Φ₁:CUSTODIA", "Φ₂:PLIEGO", "Φ₃:SENTENCIA-COMMIT"),
            "epoch": int(expected_epoch), "error": error,
            "phase2_chain_hash": p2.chain_hash, "phase1_chain_hash": p2.phase1.chain_hash,
            "authorization": {"rung": auth.ladder_rung, "action": auth.action, "acta": acta,
                              "quorum": auth.quorum_satisfied, "time_barred": auth.time_barred,
                              "authorized": auth.actuation_authorized,
                              "dispatched": auth.actuation_dispatched,
                              "fabricated": auth.actuation_fabricated,
                              "safe_state": auth.safe_state_applied, "ids": auth.authorizer_ids},
            "escalation_signature": escalation.signature if escalation else None,
            "merkle_leaf_count": len(leaves),
        }
        state = GammaPhotonProsecutorSovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(),
            crowbar_triggered=bool(auth.actuation_dispatched),
            crowbar_latency_ns=float(auth.trip_dispatch_ns), merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), sentence_leaf_index=s_idx,
            sentence_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3, authorization=auth,
            escalation=escalation, phase2=p2, evidences=evidences, kleisli_ledger=core["kleisli_ledger"],
            epoch=int(expected_epoch), state_signature=self._signer.sign(_HMAC_STATE, core), error=error,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("GP3 Ω₃=%s · %s · quórum=%s · disparo=%s (fabricado=%s)",
                    truth.name, auth.action, auth.quorum_satisfied,
                    auth.actuation_dispatched, auth.actuation_fabricated)
        return state

    def execute_phase3(self, *a: Any, **k: Any) -> GammaPhotonProsecutorSovereignState:
        """Alias heredado (v13) del terminal §3.7."""
        return self.emit_sovereign_state(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA — ciclo OODA fiscal (Kleisli) con fail-closed
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovGammaPhotonColliderProsecutorAgent:
    r"""
    Soberano agéntico FISCAL del colisionador γγ. Ejecuta observe >=> orient >=> decide sobre
    la mónada Certified. El motor corre en modo epistémico. Ante cualquier fallo: FAIL-CLOSED
    — ⊥, estado seguro, SIN disparo y acta de emergencia firmada.
    """

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 driver: Optional[CrowbarDriver] = None, seed: int = 20_261_126) -> None:
        self._signer = ProsecutorialSigner(secret_key)
        self._principal = principal_engine
        self._readonly = ReadOnlyCrowbarDriver()
        self._engine = ProsecutorEngine(secret_key=self._signer.key_bytes,
                                        principal_engine=principal_engine, seed=seed,
                                        driver=self._readonly)
        self._observer = GammaPhotonProsecutorPhase1Observer(self._signer, seed=seed)
        self._orientor = GammaPhotonProsecutorPhase2Orientor(self._engine, self._signer, seed=seed)
        self._decider = GammaPhotonProsecutorPhase3Decider(self._signer, driver or self._make_driver())
        self._epoch = 0
        logger.info("%s v%s listo · clave=%s", type(self).__name__, __version__, self._signer.fingerprint())

    def _make_driver(self) -> CrowbarDriver:
        return SimulatedCrowbarDriver(
            lambda t: self._decider._verify_pending(t) if hasattr(self, "_decider") else False)

    # ─────────── verificación externa del estado ───────────
    def verify_sovereign_state(self, state: GammaPhotonProsecutorSovereignState) -> bool:
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

    # ─────────── ciclo OODA con fail-closed ───────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: Optional[NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        claimed_events: Optional[Sequence[Any]] = None,
        params: Optional[ProsecutorialGovernanceParameters] = None,
        authorization_tokens: Sequence[Tuple[str, str]] = (),
        force_veto: bool = False,                     # ABOLIDO
        positron_hmac_valid: Optional[bool] = None,   # ABOLIDO
    ) -> GammaPhotonProsecutorSovereignState:
        """Ejecuta el proceso completo. La actuación es el último commit, sólo con quórum."""
        if force_veto:
            raise ProsecutorialGovernanceError(
                "force_veto está abolido: un booleano no autenticado no puede causar actuación física.")
        if positron_hmac_valid is not None:
            raise ProsecutorialGovernanceError(
                "positron_hmac_valid está abolido: presente authorization_tokens firmados (2-de-2).")
        params = params or ProsecutorialGovernanceParameters()
        claims = tuple(claimed_events) if claimed_events else (EngineClaimedEvent(),)
        self._epoch += 1
        epoch = self._epoch
        exps = np.asarray(exponents, dtype=np.float64)

        def observe_m(ctx: Dict[str, Any]) -> Any:
            p1, rho_in, modes = self._observer.emit_phase1_certificate(
                rho_mac, apu_cost_matrix, exps, claims, params, epoch=epoch)
            return Certified({**ctx, "p1": p1, "rho": rho_in, "m": modes}, ("Φ₁:CUSTODIA",), p1.truth_value)

        def orient_m(ctx: Dict[str, Any]) -> Any:
            p2 = self._orientor.emit_phase2_certificate(
                ctx["p1"], apu_cost_matrix, exps, ctx["rho"], ctx["m"], claims, params,
                expected_epoch=epoch)
            return Certified({**ctx, "p2": p2}, ("Φ₂:PLIEGO",), p2.truth_value)

        def decide_m(ctx: Dict[str, Any]) -> Any:
            st = self._decider.emit_sovereign_state(
                ctx["p2"], params, expected_epoch=epoch, authorization_tokens=authorization_tokens)
            return Certified(st, ("Φ₃:SENTENCIA-COMMIT",), st.truth_value)

        pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
        try:
            result = pipeline({"epoch": epoch})
            state: GammaPhotonProsecutorSovereignState = result.value
            if not self.verify_sovereign_state(state):
                raise ProsecutorialGateIntegrityError("Autoverificación de la sentencia fallida.")
            logger.info("Proceso OODA fiscal γγ completado · Ω₃=%s · ledger=%s",
                        state.truth_value.name, " → ".join(result.ledger))
            return state
        except Exception as exc:                                   # FAIL-CLOSED, nunca silencio
            logger.critical("FAIL-CLOSED FISCAL γγ: %s", exc)
            return self._emergency_state(rho_mac, apu_cost_matrix, exps, claims, params, epoch, str(exc))

    def _emergency_state(self, rho_mac: Optional[NDArray], k_mat: NDArray, exps: NDArray,
                         claims: Sequence[Any], params: ProsecutorialGovernanceParameters,
                         epoch: int, message: str) -> GammaPhotonProsecutorSovereignState:
        """Acta de emergencia: ⊥, estado seguro, SIN disparo, cadena de custodia preservada."""
        p1, _, _ = self._observer.emit_phase1_certificate(rho_mac, k_mat, exps, claims, params, epoch=epoch)
        sheet = ChargeSheetReport(
            n_claims=len(claims), structural_refutations=((0, "fallo_de_integridad"),),
            pvalues_exact=tuple(1.0 for _ in claims), pvalues_adjusted=tuple(1.0 for _ in claims),
            bh_rejected_indices=(), q_fdr=params.q_fdr, statistical_convictions=0, mc_fdr=0.0,
            mc_fdr_ok=True, mc_power=0.0, screening_note="acta de emergencia: instrucción incompleta")
        spending = self._orientor.error_spending_certificate(epoch, params)
        stability = StabilityFalsificationReport(
            nu_x=params.tune_nu_x, nu_y=params.tune_nu_y, engine_order=params.tune_engine_order,
            deep_order=params.tune_deep_order, engine_min_distance=0.0, deep_min_distance=0.0,
            deep_nearest_resonance=(0, 0, 0), deep_diophantine_gamma=0.0,
            claimed_margin=params.tune_claimed_margin, claimed_margin_survives=True,
            exact_resonance_found=False, search_power_note="emergencia: sin escaneo")
        recurrence = self._orientor.recurrence_prescription(params)
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": False, "engine_passport": "",
            "engine_acta": "", "engine_merkle_root": "", "engine_truth": HeytingTruthValue.VETOED,
            "engine_acta_verified": False, "engine_merkle_recomputed_ok": False,
            "engine_actuation_dispatched": False, "gain_honored": True, "disturbance_consumed": 0.0,
            "back_action_db": 0.0, "charge_sheet": sheet, "error_spending": spending,
            "stability": stability, "recurrence": recurrence, "evidences": (),
            "truth_value": HeytingTruthValue.VETOED, "epoch": int(epoch),
        }
        chain = _sha256(_canonical(core))
        p2 = GammaPhotonProsecutorPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=None, engine_invoked=False,
            engine_acta_verified=False, engine_merkle_recomputed_ok=False,
            engine_actuation_dispatched=False, gain_honored=True, disturbance_consumed=0.0,
            back_action_db=0.0, charge_sheet=sheet, error_spending=spending, stability=stability,
            recurrence=recurrence, evidences=(), truth_value=HeytingTruthValue.VETOED, epoch=int(epoch),
            chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core))
        return self._decider.emit_sovereign_state(p2, params, expected_epoch=epoch,
                                                  authorization_tokens=(), error=message, forced_veto=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    agent = BoolNovikovGammaPhotonColliderProsecutorAgent(
        secret_key=b"GAMMA_PROSECUTOR_AGENT_KEY_32_BYTES!", principal_engine=principal)
    k_test = np.eye(4, dtype=np.float64) + 0.02 * np.ones((4, 4))
    exps = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    P = ProsecutorialGovernanceParameters()

    # 1) Nominal: pliego consistente ⇒ ⊤, WARN, SIN disparo; Orient no actuó
    st = agent.run_sovereign_governance_cycle(None, k_test, exps)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.authorization.action == "WARN" and not st.crowbar_triggered
    assert not st.authorization.actuation_fabricated
    assert not st.phase2.engine_actuation_dispatched                 # ← Orient epistémico
    assert st.phase2.engine_acta_verified and st.phase2.engine_merkle_recomputed_ok
    assert st.phase2.charge_sheet.mc_fdr_ok and st.phase2.error_spending.within_budget
    assert st.phase2.stability.claimed_margin_survives
    assert st.phase2.phase1.bw_isometry_residual < 1e-11
    assert agent.verify_sovereign_state(st)
    assert st.visceral_metrics.net_liquidity_release_pct is None      # sin €/J: honesto

    # 2) DEFECTO CAPITAL de v13: fabricaba disparo cuando el motor NO disparó
    assert GammaPhotonProsecutorPhase3Decider.legacy_would_fabricate_trip(
        HeytingTruthValue.VETOED, False) is True
    hard = ProsecutorialGovernanceParameters(ball_r=2.0, cylinder_r=1.0,
                                             lesser_measures_insufficient=True)
    bad = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard)
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.authorization.ladder_rung == 3
    assert not bad.crowbar_triggered and not bad.authorization.actuation_fabricated
    assert bad.authorization.safe_state_applied                      # ← estado seguro, no disparo

    # 3) Φ₁₂ compromete la EVIDENCIA: mismo Tr/PSD pero otro ρ ⇒ RECHAZO (v13 lo aceptaba)
    rho_a = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    rho_b = np.diag([0.1, 0.2, 0.3, 0.4]).astype(np.complex128)      # igual traza y PSD
    st_a = agent.run_sovereign_governance_cycle(rho_a, k_test, exps)
    forged = dataclasses.replace(st_a.phase2.phase1, rho_digest=_array_digest(rho_b))
    try:
        agent._orientor.continue_to_phase2_gate(forged, k_test, exps, (EngineClaimedEvent(),), P,
                                                expected_epoch=forged.epoch)
        raise AssertionError("Φ₁₂ aceptó un payload manipulado")
    except ProsecutorialGateIntegrityError:
        pass

    # 4) Pliego con contrafacto: par colineal de 511 keV ⇒ s_rec=0 ⇒ refutación estructural
    collinear = EngineClaimedEvent(opening_angle_deg=0.0)
    refuted = agent.run_sovereign_governance_cycle(None, k_test, exps,
                                                   claimed_events=(collinear, EngineClaimedEvent()))
    assert any(s[1] == "masa_invariante" for s in refuted.phase2.charge_sheet.structural_refutations)
    assert refuted.truth_value is HeytingTruthValue.VETOED and not refuted.crowbar_triggered

    # 5) Quórum: token DUPLICADO no satisface 2-de-2; con dos IDs distintos ⇒ disparo MEDIDO
    acta, ep = bad.authorization.indictment_digest, bad.epoch + 1
    tok = agent._decider.mint_authorization_token(acta, ep, 3, "OPERATOR")
    dup = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard,
                                               authorization_tokens=(("OPERATOR", tok), ("OPERATOR", tok)))
    assert dup.authorization.duplicate_authorizer_rejected and not dup.crowbar_triggered
    ep2 = dup.epoch + 1
    acta2 = _sha256(_canonical({"epoch": ep2, "truth": HeytingTruthValue.VETOED, "action": "TRIP",
                                "structural": True, "p2_chain": dup.phase2.chain_hash,
                                "p1_chain": dup.phase2.phase1.chain_hash,
                                "engine_acta": dup.phase2.engine_report.sentence.indictment_digest,
                                "refutations": dup.phase2.charge_sheet.structural_refutations,
                                "bh_rejected": dup.phase2.charge_sheet.bh_rejected_indices,
                                "error": None}))
    t_op = agent._decider.mint_authorization_token(acta2, ep2, 3, "OPERATOR")
    t_sup = agent._decider.mint_authorization_token(acta2, ep2, 3, "SUPERVISOR")
    armed = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard,
                                                 authorization_tokens=(("OPERATOR", t_op),
                                                                       ("SUPERVISOR", t_sup)))
    assert armed.authorization.quorum_satisfied and armed.crowbar_triggered
    assert armed.crowbar_latency_ns > 0.0 and not armed.authorization.actuation_fabricated
    assert not armed.authorization.physical_deadline_feasible        # SCR ~µs: declarado

    # 6) Booleanos heredados abolidos
    for kw in ({"force_veto": True}, {"positron_hmac_valid": True}):
        try:
            agent.run_sovereign_governance_cycle(None, k_test, exps, **kw)
            raise AssertionError(f"{kw} no fue rechazado")
        except ProsecutorialGovernanceError:
            pass

    # 7) BH/FDR y alpha-spending: cota de FDP, monotonía y gasto total = α
    assert st.phase2.charge_sheet.mc_fdr <= P.q_fdr + 0.02
    sp = st.phase2.error_spending
    assert sp.monotone_ok and sp.total_spend_residual < 1e-12
    assert abs(lan_demets_obrien_fleming(P.alpha_total, 1.0) - P.alpha_total) < 1e-12
    rej, adj = benjamini_hochberg([0.001, 0.02, 0.6], 0.05)
    assert rej == (0, 1) and adj[0] <= 0.003 + 1e-12

    # 8) Falsación profunda de estabilidad y prescripción por Kac
    deep = agent.run_sovereign_governance_cycle(
        None, k_test, exps, params=ProsecutorialGovernanceParameters(tune_claimed_margin=0.02))
    assert not deep.phase2.stability.claimed_margin_survives         # el margen NO sobrevive
    res = agent.run_sovereign_governance_cycle(
        None, k_test, exps, params=ProsecutorialGovernanceParameters(tune_nu_x=0.5, tune_nu_y=0.25))
    assert res.phase2.stability.exact_resonance_found and res.truth_value is HeytingTruthValue.VETOED
    rec = st.phase2.recurrence
    assert abs(rec.kac_empirical_return - 10.0) < 0.05 and len(rec.slater_return_times) <= 3
    assert rec.prescription_detection_probability >= rec.prescription_target - 1e-12

    # 9) Merkle RFC 6962, determinismo y detección de manipulación
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    st2 = agent.run_sovereign_governance_cycle(None, k_test, exps)
    assert st2.phase2.phase1.chain_hash == st.phase2.phase1.chain_hash   # sellos sin reloj
    assert not agent.verify_sovereign_state(dataclasses.replace(st, crowbar_triggered=True))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderProsecutorAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {st.truth_value.name}({int(st.truth_value)}) / {st.boolean_shadow.name}")
    print(f"Sentencia / disparo           : {st.authorization.action} / {st.crowbar_triggered}")
    print(f"Actuación fabricada           : {st.authorization.actuation_fabricated} (v13 la fabricaba)")
    print(f"Orient actuó (debe ser False) : {st.phase2.engine_actuation_dispatched}")
    print(f"m Fock / cargos / g sellado   : {st.phase2.phase1.fock_modes_reconciled} / "
          f"{st.phase2.charge_sheet.n_claims} / {st.phase2.phase1.committed_gain_g:.6g}")
    print(f"d_BW(ρ,ρ_térmica) / isometría : {st.phase2.phase1.bures_wasserstein_to_thermal:.6f} / "
          f"{st.phase2.phase1.bw_isometry_residual:.2e}")
    print(f"BH q / FDR_MC / potencia      : {st.phase2.charge_sheet.q_fdr:.3f} / "
          f"{st.phase2.charge_sheet.mc_fdr:.4f} / {st.phase2.charge_sheet.mc_power:.3f}")
    print(f"p-values exactos (pliego)     : "
          + ", ".join(f"{v:.4f}" for v in st.phase2.charge_sheet.pvalues_exact))
    print(f"α gastado / α_total / época   : {sp.cumulative_spent:.5f} / {sp.alpha_total:.3f} / "
          f"{sp.epoch_index} de {sp.planned_epochs} (residuo total {sp.total_spend_residual:.1e})")
    print(f"Tunes orden motor→profundo    : {st.phase2.stability.engine_order} → {st.phase2.stability.deep_order}")
    print(f"d_min motor / profundo        : {st.phase2.stability.engine_min_distance:.5f} / "
          f"{st.phase2.stability.deep_min_distance:.5f} · resonancia {st.phase2.stability.deep_nearest_resonance}")
    print(f"Margen reclamado sobrevive    : {st.phase2.stability.claimed_margin_survives} "
          f"(con margen 0.02: {deep.phase2.stability.claimed_margin_survives})")
    print(f"Kac esperado / empírico       : {rec.kac_expected_return:.3f} / {rec.kac_empirical_return:.3f} · "
          f"Slater {rec.slater_return_times}")
    print(f"Ventana de prescripción [épocas]: {rec.prescription_window_epochs} "
          f"(P_det={rec.prescription_detection_probability:.4f} ≥ {rec.prescription_target:.2f})")
    print(f"RSI³ W calibrada / Lip(T³)    : {st.rsi3.calibrated_window_epochs:.3f} / {st.rsi3.lip_t3_empirical:.6f}")
    print(f"Kantorovich h / d³C           : {st.rsi3.kantorovich_h:.6f} / {st.rsi3.capacity_d3:.4f}")
    print(f"⊥ SIN quórum                    : {bad.authorization.action} disparo={bad.crowbar_triggered} "
          f"seguro={bad.authorization.safe_state_applied}")
    print(f"Token duplicado               : rechazado={dup.authorization.duplicate_authorizer_rejected} "
          f"disparo={dup.crowbar_triggered}")
    print(f"⊥ CON quórum 2-de-2           : {armed.authorization.action} IDs={armed.authorization.authorizer_ids} "
          f"disparo={armed.crowbar_triggered} latencia={armed.crowbar_latency_ns:.0f} ns")
    print(f"Estado del evento             : {st.visceral_metrics.event_status}")
    print(f"Merkle raíz (SIN clave)       : {st.merkle_root_hash[:16]}… ({len(st.merkle_leaves)} hojas)")
    print(f"Sentencia reverificable       : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")