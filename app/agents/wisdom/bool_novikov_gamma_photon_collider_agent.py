# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderAgent  v14.0.0-PhD (reedición rigurosa)                ║
║ Ubicación: app/agents/wisdom/bool_novikov_gamma_photon_collider_agent.py                        ║
║ Función  : Soberano agéntico ACTIVO del colisionador γγ: custodia criptográfica real,           ║
║            protección de máquina (MPS) con causa común, presupuesto térmico del dump,          ║
║            footprint de tunes, regla de dos hombres y COMMIT de actuación autorizada.           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL SOBERANO ACTIVO
────────────────────────────
(1) UNA COMPUERTA QUE MIDE LONGITUDES NO AUTENTICA. v13 aceptaba cualquier cadena de 64
    caracteres como firma HMAC. Aquí la compuerta RECALCULA el hash desde el payload y
    verifica el MAC en tiempo constante; además re-digesta los arrays que cruzan (anti-TOCTOU).
(2) ORIENT NO ACTÚA. El motor se ejecuta con driver de SÓLO LECTURA y sin token. La actuación
    es el ÚLTIMO commit de Fase 3, después de Φ₂₃. (v13 disparaba en Fase 2 y luego podía
    rechazar la compuerta: hardware actuado sin certificado final.)
(3) REGLA DE DOS HOMBRES. TRIP exige quórum 2-de-2 con autorizador IDs DISTINTOS, tokens
    ligados a {acta, época, peldaño, autorizador}, anti-replay y ne bis in idem. Un token
    duplicado no satisface el quórum. Sin quórum: ISOLATE + estado seguro, jamás disparo.
(4) LA REDUNDANCIA NO VENCE LA CAUSA COMÚN. Modelo β-factor: PFD = PFD_ind((1−β)λ) + βλT/2.
    Aumentar n reduce el término independiente pero NO el piso β·p. Se certifica el piso.
(5) LO QUE ES MODELO SE DECLARA. Las bandas SIL, los λ_du, las capacidades térmicas y los
    tunes son INSUMOS DECLARADOS (datasheet/HIL); el certificado es sobre el modelo, no una
    acreditación IEC 61508 ni una simulación de QED dinámica.

CONTRATO DE ANIDAMIENTO (tres fases OODA anidadas por datos y por continuación)
─────────────────────────────────────────────────────────────────────────────
    Fase 1 §1.5  emit_phase1_certificate  ──►  Fase 2 §2.1  accept_phase1_certificate  (Φ₁₂)
    Fase 2 §2.8  emit_phase2_certificate  ──►  Fase 3 §3.1  accept_phase2_certificate  (Φ₂₃)
    Fase 3 §3.7  emit_sovereign_state     ──►  Fachada §4    run_sovereign_governance_cycle
    P1 ⊂ P2 ⊂ GammaPhotonColliderSovereignState; ejecución = Kleisli observe>=>orient>=>decide.

FASE 1 — OBSERVE (custodia y pre-compromiso)
  §1.1 Validación previa + RECONCILIACIÓN de Fock (m = log₂ dim ρ; potencia de 2 exigida).
  §1.2 Normalización auditada + ISOMETRÍA DE BURES–WASSERSTEIN bajo unitario Haar.
  §1.3 PRE-COMPROMISO: calibración Newton–Kantorovich de g y del presupuesto de energía.
  §1.4 Compromiso canónico de entradas, medición, cinemática y luminosidad.
  §1.5 [TERMINAL] emit_phase1_certificate.

COMPUERTA Φ₁₂ — integridad dura (hash recalculado, HMAC, re-digest, época); ADMITE ⊥.

FASE 2 — ORIENT (epistémico)
  §2.1 accept_phase1_certificate.
  §2.2 Motor gamma v14 con g comprometido, driver sólo lectura y SIN token.
  §2.3 Verificación del acta del motor + Merkle público + prueba de inclusión.
  §2.4 MPS: binomial exacto k-de-n, β-factor (piso de causa común), PFD y banda SIL declarada.
  §2.5 Presupuesto térmico del dump: E = N·E_k·[J/keV], ΔT = E/(m·c_p), margen declarado.
  §2.6 Consistencia de conteos: Fano/dispersión con banda χ² exacta + cobertura Monte Carlo.
  §2.7 Footprint de tunes: resonancias m ν_x + n ν_y = p (escaneo exacto) + γ diophantino.
  §2.8 [TERMINAL] emit_phase2_certificate.

COMPUERTA Φ₂₃ — integridad dura; física (Lip<1, PFD≤budget, ΔT≤max, sin resonancia) ⇒ ⊥ forzado.

FASE 3 — DECIDE & ACT
  §3.1 accept_phase2_certificate.
  §3.2 Sentencia Ω₃, núcleo ¬¬, functor contravariante y escalera de proporcionalidad.
  §3.3 Regla de dos hombres: quórum, IDs distintos, anti-replay, ne bis in idem.
  §3.4 RSI³: calibración del intervalo de prueba (Kantorovich) + Banach + Kleisli + jerk.
  §3.5 Φ_sem con procedencia dimensional.
  §3.6 COMMIT de actuación (HAL medido, presupuesto físico declarado, estado seguro).
  §3.7 [TERMINAL] emit_sovereign_state (Merkle sin clave + firma determinista + escalada).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `positron_hmac_valid: bool` y `force_veto: bool` ABOLIDOS ⇒ `authorization_tokens` firmados.
• `rep.annihilation_report` → `rep.phase_ii.kinematics`; `rep.banach_contraction_lip` →
  `rep.rsi3.lip_t3_bound`; `rep.qnd_report` → `rep.phase_ii.phase_i.qnd`.
• Sellos deterministas (sin reloj); raíz Merkle SIN clave y con prueba de inclusión.
• KPI sin insumos declarados ⇒ None con `UNDECLARED_INPUT`.

LIMITACIONES DECLARADAS
1. MPS es un modelo de fiabilidad con insumos declarados; no sustituye una certificación SIL.
2. El modelo β-factor es una aproximación de causa común; no cubre fallos dependientes
   complejos (cascadas, errores de software comunes).
3. ΔT del dump es un modelo adiabático de depósito uniforme; ignora gradientes y choque térmico.
4. El escaneo de resonancias es de orden finito: ausencia de resonancia ≠ estabilidad global.
5. 392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza.

REFERENCIAS: IEC 61508 (PFD, SIL, β-factor); Lundteigen–Rausand (2008); Garwood (1936);
Fano (1947); Fisher (dispersión χ²); Courant–Snyder (tunes); Chirikov (solapamiento);
Khinchin (diophantina); Bures (1969); Uhlmann (1976); Kac (1947); Kantorovich; Banach (1922);
Mac Lane (Kleisli); RFC 6962; CODATA 2022.
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
from scipy.special import gammainc, gammaincinv


def _import_first(paths: Sequence[str]) -> Any:
    last: Optional[Exception] = None
    for p in paths:
        try:
            return __import__(p, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(f"No se pudo importar ninguno de: {paths}") from last


_ce = _import_first(("app.wisdom.bool_novikov_gamma_photon_collider_engine",
                     "bool_novikov_gamma_photon_collider_engine"))
_pri = _import_first(("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
                      "bool_novikov_thermal_exergy_weaver_engine"))

ColliderEngine = _ce.BoolNovikovGammaPhotonColliderEngine
EngineColliderParameters = _ce.ColliderParameters
PrincipalEngine = _pri.BoolNovikovThermalExergyWeaverEngine
HeytingTruthValue = _pri.HeytingTruthValue          # Ω₃ CANÓNICO
Evidence = _pri.Evidence
Certified = _pri.Certified
kleisli_compose = _pri.kleisli_compose

__all__ = [
    "ColliderGovernanceParameters", "GammaPhotonPhase1Certificate", "GammaPhotonPhase2Certificate",
    "GammaPhotonVisceralMetrics", "GammaPhotonColliderSovereignState",
    "GammaPhotonPhase1Observer", "GammaPhotonPhase2Orientor", "GammaPhotonPhase3Decider",
    "BoolNovikovGammaPhotonColliderAgent", "ColliderGovernanceError",
    "ColliderGateIntegrityError", "AuthorizationError", "CrowbarDriver",
    "SimulatedCrowbarDriver", "ReadOnlyCrowbarDriver", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderAgent")

__version__: Final[str] = "14.0.0-SOVEREIGN-GAMMA-COLLIDER-AGENT-MPS-BETA-FACTOR-TUNES-DUAL-AUTH-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_J_PER_KEV: Final[float] = 1.602176634e-16        # exacto (SI 2019): 1 keV en julios
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")
_SIL_BANDS: Final[Tuple[Tuple[float, float, str], ...]] = (
    (1e-5, 1e-4, "SIL4"), (1e-4, 1e-3, "SIL3"), (1e-3, 1e-2, "SIL2"), (1e-2, 1e-1, "SIL1"),
)
_HMAC_P1: Final[bytes] = b"GAMMA-COLLIDER-AGENT-P1-CUSTODY-v14|"
_HMAC_P2: Final[bytes] = b"GAMMA-COLLIDER-AGENT-P2-MPS-v14|"
_HMAC_STATE: Final[bytes] = b"GAMMA-COLLIDER-AGENT-SENTENCE-v14|"
_HMAC_AUTH: Final[bytes] = b"GAMMA-COLLIDER-AGENT-DUAL-AUTHORIZATION-v14|"
_HMAC_ESC: Final[bytes] = b"GAMMA-COLLIDER-AGENT-ESCALATION-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle, BW, fiabilidad MPS, dispersión
# ══════════════════════════════════════════════════════════════════════════════════════
class ColliderGovernanceError(Exception):
    """Error base de la gobernanza del colisionador."""


class ColliderGateIntegrityError(ColliderGovernanceError):
    """Fallo DURO de integridad ⇒ fail-safe (estado seguro, sin disparo) y acta de emergencia."""


class AuthorizationError(ColliderGovernanceError):
    """Actuación sin quórum válido, con replay de época o acta ya juzgada."""


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
        raise ColliderGovernanceError("Índice de hoja fuera de rango.")

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


def mps_failure_probability(n_channels: int, k_required: int, p_channel: float,
                            beta_common_cause: float) -> Tuple[float, float, float]:
    r"""
    Fiabilidad EXACTA de un sistema k-de-n con modelo β-factor (IEC 61508-6).

    El sistema falla en la demanda si fallan más de n−k canales:
        P_ind(p) = Σ_{j=n−k+1}^{n} C(n,j) p^j (1−p)^{n−j}.
    Causa común: p = p_ind + p_cc con p_cc = β·p ⇒
        P_total = P_ind((1−β)p) + β·p.
    CONSECUENCIA: por más que crezca n, P_total ≥ β·p (PISO DE CAUSA COMÚN).
    Devuelve (P_total, P_ind, piso_cc).
    """
    if not (1 <= k_required <= n_channels):
        raise ColliderGovernanceError("Se requiere 1 ≤ k ≤ n.")
    if not (0.0 <= p_channel <= 1.0 and 0.0 <= beta_common_cause <= 1.0):
        raise ColliderGovernanceError("p y β en [0,1].")
    p_cc = beta_common_cause * p_channel
    p_ind = (1.0 - beta_common_cause) * p_channel

    def ind(p: float) -> float:
        return float(sum(math.comb(n_channels, j) * p ** j * (1.0 - p) ** (n_channels - j)
                         for j in range(n_channels - k_required + 1, n_channels + 1)))

    p_ind_val = ind(p_ind)
    return min(1.0, p_ind_val + p_cc), p_ind_val, p_cc


def mps_pfd_avg(n_channels: int, k_required: int, lambda_du_per_h: float,
                proof_test_hours: float, beta_common_cause: float) -> float:
    r"""
    PFD medio (baja demanda) para k-de-n con prueba periódica T:
        PFD_ind = C(n, n−k+1)·(λ_du T)^{n−k+1}/(n−k+2),   PFD = PFD_ind((1−β)λ) + β λ T/2.
    Verificado contra los casos conocidos: 1oo2 ⇒ (λT)²/3; 2oo3 ⇒ (λT)².
    """
    m = n_channels - k_required + 1
    lam_ind = (1.0 - beta_common_cause) * lambda_du_per_h
    lt = lam_ind * proof_test_hours
    pfd_ind = math.comb(n_channels, m) * lt ** m / (m + 1)
    return float(pfd_ind + beta_common_cause * lambda_du_per_h * proof_test_hours / 2.0)


def sil_band(pfd: float) -> str:
    """Banda SIL declarada por el modelo (NO una acreditación IEC 61508)."""
    for lo, hi, name in _SIL_BANDS:
        if lo <= pfd < hi:
            return name
    return "SIN_BANDA_SIL (PFD fuera de 1e-5..1e-1)" if pfd >= 1e-1 else "MEJOR_QUE_SIL4"


def poisson_dispersion_test(counts: Sequence[int], alpha: float = 0.05
                            ) -> Tuple[float, float, float, float, float, bool]:
    r"""
    Prueba de dispersión de Poisson (Fano). Bajo H₀ (conteos Poisson iid):
        D = Σ(x_i − x̄)²/x̄  ~  χ²_{n−1}  (aproximación clásica de Fisher),
        F̂ = D/(n−1) ≈ 1.
    Banda exacta con χ²_p(k) = 2·gammaincinv(k/2, p); p-valor bilateral con gammainc.
    Devuelve (F̂, D, lo, hi, p_valor, acepta).
    """
    x = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 3:
        raise ColliderGovernanceError("Se requieren ≥3 conteos para la prueba de dispersión.")
    mean = float(np.mean(x))
    if mean <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 1.0, True
    d_stat = float(np.sum((x - mean) ** 2) / mean)
    df = n - 1
    lo = 2.0 * float(gammaincinv(df / 2.0, alpha / 2.0))
    hi = 2.0 * float(gammaincinv(df / 2.0, 1.0 - alpha / 2.0))
    cdf = float(gammainc(df / 2.0, d_stat / 2.0))
    p_val = 2.0 * min(cdf, 1.0 - cdf)
    return d_stat / df, d_stat, lo, hi, min(1.0, p_val), bool(lo <= d_stat <= hi)


def tune_resonance_scan(nu_x: float, nu_y: float, order: int, margin: float,
                        tau: float = 1.0) -> Tuple[float, float, Tuple[int, int, int], bool]:
    r"""
    Footprint de tunes (Courant–Snyder): una resonancia de orden |m|+|n| ocurre cuando
        m·ν_x + n·ν_y = p,  m,n,p ∈ ℤ.
    Es la condición diophantina del problema KAM en su forma de acelerador. Se barre todo
    (m,n) con 1 ≤ |m|+|n| ≤ order y p entero cercano, y se certifica
        min |m ν_x + n ν_y − p| ≥ margin,   γ = min (dist·(|m|+|n|)^τ).
    Ausencia de resonancia a orden finito NO prueba estabilidad global (declarado).
    """
    if order < 1 or margin < 0.0:
        raise ColliderGovernanceError("order ≥ 1 y margin ≥ 0.")
    best, gamma = float("inf"), float("inf")
    nearest = (0, 0, 0)
    p_max = int(math.ceil(order * max(abs(nu_x), abs(nu_y)))) + 2
    for m, n in itertools.product(range(-order, order + 1), repeat=2):
        l1 = abs(m) + abs(n)
        if l1 == 0 or l1 > order:
            continue
        val = m * nu_x + n * nu_y
        p = int(round(val))
        if abs(p) > p_max:
            continue
        dist = abs(val - p)
        if dist < best:
            best, nearest = dist, (m, n, p)
        gamma = min(gamma, dist * l1 ** tau)
    return float(best), float(gamma), nearest, bool(best >= margin)


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


class ColliderSigner:
    """HMAC-SHA256 con separación de dominio y comparación en tiempo constante."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_COLLIDER_AGENT_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave de colisionador efímera: sentencias no verificables entre procesos.")
        if len(secret_key) < 32:
            raise ColliderGovernanceError("La clave del colisionador requiere ≥ 256 bits.")
        self._key = bytes(secret_key)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def fingerprint(self) -> str:
        return hmac.new(self._key, b"GAMMA-COLLIDER-KEY-FP-v14|", hashlib.sha256).hexdigest()[:16]

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
class ColliderGovernanceParameters:
    """Insumos DECLARADOS (datasheet/HIL/política); su digest se compromete en P1."""

    fermion_modes: int = 2
    pair_coherence: float = 1.0
    interaction_g: Optional[float] = None        # None ⇒ calibración certificada en §1.3
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    beta_field: float = 2.0
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    photon_crossing_deg: float = 180.0
    luminosity_cm2_s: float = 1.0e32
    run_time_s: float = 1.0
    ball_r: float = 1.0
    cylinder_r: float = 1.5
    # --- MPS (protección de máquina) ---
    mps_channels: int = 3
    mps_required: int = 2
    mps_p_channel: float = 1.0e-3
    mps_beta_common_cause: float = 0.05
    mps_lambda_du_per_h: float = 1.0e-6
    mps_proof_test_hours: float = 8760.0
    mps_pfd_budget: float = 1.0e-3
    # --- Presupuesto térmico del dump ---
    beam_particles: float = 1.0e14
    beam_energy_per_particle_kev: float = 510.99895069
    dump_mass_kg: float = 0.010
    dump_specific_heat_j_kg_k: float = 710.0     # grafito (declarado)
    dump_delta_t_max_k: float = 1500.0
    dump_capacity_j: float = 1.0e5
    # --- Conteos y tunes ---
    observed_counts: Tuple[int, ...] = (98, 104, 95, 101, 107, 92, 100, 103, 99, 96, 105, 100)
    counts_alpha: float = 0.05
    counts_mc_draws: int = 4000
    tune_nu_x: float = 0.31
    tune_nu_y: float = 0.28
    tune_order: int = 8
    tune_margin: float = 5.0e-3
    # --- Autorización y hardware ---
    require_dual_authorization: bool = True
    lesser_measures_insufficient: bool = False
    logic_deadline_ns: float = 392.15
    gpio_write_ns: float = 25.0
    gate_driver_ns: float = 150.0
    scr_turn_on_ns: float = 1500.0
    hil_firmware_isr_ns: Optional[float] = None
    cadence_points: int = 64
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    # --- Φ_sem ---
    exergy_reference_w: float = 1.0
    exergy_price_per_joule: Optional[float] = None
    capital_base: Optional[float] = None
    kv_bytes_before: Optional[int] = None
    kv_bytes_after: Optional[int] = None

    def __post_init__(self) -> None:
        if not 2 <= self.fermion_modes <= 6:
            raise ColliderGovernanceError("fermion_modes ∈ [2,6].")
        if self.pointer_sigma <= 0.0 or self.disturbance_budget < 0.0:
            raise ColliderGovernanceError("σ>0 y presupuesto ≥0.")
        if not (1 <= self.mps_required <= self.mps_channels):
            raise ColliderGovernanceError("MPS: 1 ≤ k ≤ n.")
        if not (0.0 <= self.mps_beta_common_cause <= 1.0 and 0.0 < self.mps_p_channel <= 1.0):
            raise ColliderGovernanceError("MPS: β∈[0,1], p∈(0,1].")
        if self.mps_lambda_du_per_h <= 0.0 or self.mps_proof_test_hours <= 0.0:
            raise ColliderGovernanceError("MPS: λ_du>0 y T>0.")
        if self.dump_mass_kg <= 0.0 or self.dump_specific_heat_j_kg_k <= 0.0:
            raise ColliderGovernanceError("Dump: masa y c_p positivas.")
        if self.beam_particles <= 0.0 or self.beam_energy_per_particle_kev <= 0.0:
            raise ColliderGovernanceError("Haz: N y energía positivos.")
        if not (0.0 < self.recurrence_mu < 1.0):
            raise ColliderGovernanceError("μ∈(0,1).")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)

    def to_engine_parameters(self, modes: int, g: float) -> Any:
        """Proyección al contrato del motor v14 (sin autorizar actuación)."""
        return EngineColliderParameters(
            fermion_modes=int(modes), pair_coherence=self.pair_coherence,
            interaction_g=float(g), pointer_sigma=self.pointer_sigma,
            disturbance_budget=self.disturbance_budget, backaction_db_budget=self.backaction_db_budget,
            beta_field=self.beta_field, positron_kinetic_kev=self.positron_kinetic_kev,
            emission_angle_deg=self.emission_angle_deg, photon_crossing_deg=self.photon_crossing_deg,
            luminosity_cm2_s=self.luminosity_cm2_s, run_time_s=self.run_time_s,
            ball_r=self.ball_r, cylinder_r=self.cylinder_r,
            require_human_authorization=True,          # el motor NUNCA autoemite token
            lesser_measures_insufficient=self.lesser_measures_insufficient,
            logic_deadline_ns=self.logic_deadline_ns, gpio_write_ns=self.gpio_write_ns,
            gate_driver_ns=self.gate_driver_ns, scr_turn_on_ns=self.scr_turn_on_ns,
            hil_firmware_isr_ns=self.hil_firmware_isr_ns,
            cadence_points=self.cadence_points, recurrence_mu=self.recurrence_mu,
            recurrence_samples=self.recurrence_samples,
        )


@dataclass(frozen=True)
class GammaPhotonPhase1Certificate:
    """P1 — Custodia con PRE-COMPROMISO (reloj fuera del sello)."""

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
    gain_calibration_residual: float
    gain_degenerate_channel: bool
    gain_override_rejected: bool
    committed_beam_energy_j: float
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
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
class MachineProtectionReport:
    """§2.4 — MPS: binomial exacto, piso de causa común y PFD/banda SIL declarada."""

    n_channels: int
    k_required: int
    p_channel: float
    beta_common_cause: float
    p_fail_total: float
    p_fail_independent: float
    common_cause_floor: float
    p_fail_single_channel: float
    redundancy_improves: bool
    floor_dominates: bool
    lambda_du_per_h: float
    proof_test_hours: float
    pfd_avg: float
    pfd_budget: float
    pfd_within_budget: bool
    sil_band_claimed: str
    pfd_1oo2_identity_residual: float
    pfd_2oo3_identity_residual: float
    note: str


@dataclass(frozen=True)
class BeamDumpThermalReport:
    """§2.5 — Presupuesto térmico adiabático del blanco de absorción."""

    beam_particles: float
    energy_per_particle_kev: float
    stored_energy_j: float
    dump_capacity_j: float
    capacity_ok: bool
    dump_mass_kg: float
    specific_heat_j_kg_k: float
    delta_t_k: float
    delta_t_max_k: float
    thermal_margin_ok: bool
    joule_per_kev_used: float
    note: str


@dataclass(frozen=True)
class CountStatisticsReport:
    """§2.6 — Consistencia de conteos con Poisson (Fano/dispersión χ²)."""

    n_samples: int
    sample_mean: float
    expected_from_luminosity: float
    fano_factor: float
    dispersion_statistic: float
    chi2_lower: float
    chi2_upper: float
    p_value: float
    dispersion_accepts: bool
    mc_acceptance_rate: float
    mc_rate_ok: bool
    relative_rate_residual: float


@dataclass(frozen=True)
class TuneFootprintReport:
    """§2.7 — Resonancias de betatrón (KAM en lenguaje de acelerador)."""

    nu_x: float
    nu_y: float
    scan_order: int
    margin_required: float
    min_resonance_distance: float
    diophantine_gamma: float
    nearest_resonance: Tuple[int, int, int]
    exact_resonance: bool
    margin_ok: bool
    kac_expected_demand_delay: float
    kac_empirical_delay: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    slater_ok: bool


@dataclass(frozen=True)
class GammaPhotonPhase2Certificate:
    """P2 — Orientación epistémica: anida P1 y el acta del motor (sin actuación)."""

    timestamp: float
    phase1: GammaPhotonPhase1Certificate
    engine_report: Optional[Any]
    engine_invoked: bool
    engine_acta_verified: bool
    engine_merkle_recomputed_ok: bool
    engine_actuation_dispatched: bool
    gain_honored: bool
    disturbance_consumed: float
    back_action_db: float
    machine_protection: MachineProtectionReport
    beam_dump: BeamDumpThermalReport
    count_statistics: CountStatisticsReport
    tune_footprint: TuneFootprintReport
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
            "back_action_db": self.back_action_db, "machine_protection": self.machine_protection,
            "beam_dump": self.beam_dump, "count_statistics": self.count_statistics,
            "tune_footprint": self.tune_footprint, "evidences": self.evidences,
            "truth_value": self.truth_value, "epoch": self.epoch,
        }


@dataclass(frozen=True)
class ColliderGateReceipt:
    gate: str
    payload_digest: str
    chain_recomputed_ok: bool
    signature_ok: bool
    binding_ok: bool
    admitted: bool
    forced_veto: bool
    note: str


@dataclass(frozen=True)
class GammaPhotonVisceralMetrics:
    """§3.5 — Φ_sem con procedencia: métricas medidas; finanzas sólo si declaradas."""

    certificate_coverage_pct: float
    mps_risk_reduction_factor: float
    dump_thermal_utilization_pct: float
    fano_deviation_pct: float
    tune_margin_utilization_pct: float
    wacc_protection_pct: Optional[float]
    net_liquidity_release_pct: Optional[float]
    kv_cache_compression_pct: Optional[float]
    fock_annihilation_status: str
    machine_protection_status: str
    provenance: Tuple[str, ...]
    monotonicity_ok: bool


@dataclass(frozen=True)
class ColliderRSI3Report:
    """§3.4 — RSI³: intervalo de prueba (Kantorovich) + Banach + Kleisli + jerk."""

    calibrated_proof_test_hours: float
    target_pfd: float
    achieved_pfd: float
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
class DualAuthorizationReport:
    """§3.3+§3.6 — Regla de dos hombres y COMMIT de actuación."""

    ladder_rung: int
    action: str
    indictment_digest: str
    epoch: int
    quorum_required: int
    quorum_satisfied: bool
    authorizer_ids: Tuple[str, ...]
    duplicate_authorizer_rejected: bool
    tokens_verified: Tuple[bool, ...]
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
class ColliderEscalation:
    epoch: int
    truth_value: HeytingTruthValue
    reason: str
    merkle_root_hash: str
    signature: str
    delivered: bool


@dataclass(frozen=True)
class GammaPhotonColliderSovereignState:
    """Estado soberano final (nombres heredados conservados)."""

    timestamp: float
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    sentence_leaf_index: int
    sentence_inclusion_proof: Tuple[Tuple[str, bool], ...]
    visceral_metrics: GammaPhotonVisceralMetrics
    rsi3: ColliderRSI3Report
    authorization: DualAuthorizationReport
    escalation: Optional[ColliderEscalation]
    phase2: GammaPhotonPhase2Certificate
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
                "authorized": self.authorization.actuation_authorized,
                "dispatched": self.authorization.actuation_dispatched,
                "safe_state": self.authorization.safe_state_applied,
                "ids": self.authorization.authorizer_ids,
            },
            "escalation_signature": self.escalation.signature if self.escalation else None,
            "merkle_leaf_count": len(self.merkle_leaves),
        }

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 1 · OBSERVE — custodia, reconciliación de Fock y pre-compromiso
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonPhase1Observer:
    r"""
    FASE 1 OODA. Observa sin excluir (la malformación agrava), reconcilia la dimensión de
    Fock fermiónico y PRE-COMPROMETE la fuerza de interacción antes de cualquier contacto.
    """

    PHASE_LABEL: Final[str] = "1·OBSERVE-CUSTODIA"

    def __init__(self, signer: ColliderSigner, *, seed: int = 20_261_123) -> None:
        self._signer = signer
        self._rng = np.random.default_rng(seed)

    # ─────────── §1.1 Validación previa + reconciliación de Fock ───────────
    def inspect_and_reconcile(self, rho_raw: Optional[NDArray], params: ColliderGovernanceParameters
                              ) -> Tuple[Optional[NDArray], int, Dict[str, Any]]:
        r"""
        El motor v14 exige ρ de dimensión 2^m (Λ(ℂ^m)). v13 pasaba ρ de dim(K): con dim=3
        ni siquiera es potencia de 2. Aquí: si dim ρ = 2^m se reconcilia m = log₂ dim; si no,
        se SELLA la malformación, se declara el sustituto del motor y se agrava (no se exime).
        """
        if rho_raw is None:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": 0.0, "trace_raw": 1.0,
                "dimension_ok": True, "substitute": False}
        r = np.asarray(rho_raw, dtype=np.complex128)
        malformed = r.ndim != 2 or r.shape[0] != r.shape[1] or not np.all(np.isfinite(r))
        if malformed:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": float("inf"), "trace_raw": 0.0,
                "dimension_ok": False, "substitute": True}
        dim = int(r.shape[0])
        m = int(round(math.log2(dim))) if dim >= 2 else 0
        power_of_two = bool(m >= 1 and (1 << m) == dim and m <= 6)
        fro = float(la.norm(r, ord="fro")) + _EPS
        herm = float(la.norm(r - r.conj().T, ord="fro")) / fro
        tr = complex(np.trace(r))
        bad_trace = abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0.0
        if not power_of_two or bad_trace:
            return None, int(params.fermion_modes), {
                "hermiticity_residual": herm, "trace_raw": float(tr.real),
                "dimension_ok": power_of_two, "substitute": True}
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
                                 ) -> Tuple[float, float, float, bool]:
        r"""
        Proxy Frobenius EXACTO de la disturbancia dispersiva sobre Λ(ℂ^m), con N̂ = popcount:
            S(g) = Σ_mn |ρ_mn|²(1 − e^{−g²(N_m−N_n)²/8σ²})²,  S'(g) analítica.
        Newton con salvaguarda de bisección; h de Kantorovich estimado (declarado).
        Si [ρ,N̂]=0 el canal es degenerado: todo g es no invasivo y se DECLARA.
        """
        dim = 1 << modes
        n_vals = np.array([bin(n).count("1") for n in range(dim)], dtype=np.float64)
        d2 = (n_vals[:, None] - n_vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(rho) ** 2
        if float(np.sum(w2 * d2)) <= 1e-300:
            return float(g_default), 0.0, 0.0, True

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
        last = 0.0
        for _ in range(int(max_iter)):
            f_v, dv = s_of(g) - target, ds_of(g)
            if f_v > 0.0:
                hi = g
            else:
                lo = g
            step = g - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g = float(min(max(step, lo), hi)) if hi > lo else float(step)
            last = abs(s_of(g) - target)
            if hi - lo < 1e-18 * max(1.0, hi):
                break
        return float(g), float(h_k), float(last), False

    # ─────────── §1.5 [TERMINAL FASE 1] → Φ₁₂ → Fase 2 §2.1 ───────────
    def emit_phase1_certificate(
        self, rho_raw: Optional[NDArray], apu_cost_matrix: NDArray, exponents: NDArray,
        params: ColliderGovernanceParameters, *, epoch: int, engine_passport_bound: str = "",
    ) -> Tuple[GammaPhotonPhase1Certificate, Optional[NDArray], int]:
        """MÉTODO TERMINAL DE LA FASE 1: custodia, reconcilia, calibra y COMPROMETE."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise ColliderGovernanceError("K debe ser cuadrada.")
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
        en = 510.99895069 * np.array([bin(n).count("1") for n in range(dim)], dtype=np.float64)
        lp = -params.beta_field * (en - en[0])
        lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        d_bw, iso_res, cons_res = self.bw_isometry(rho, rho_th)

        g_cert, h_k, cal_res, degenerate = self.certify_interaction_gain(
            rho, modes, params.pointer_sigma, params.disturbance_budget)
        override_rejected = False
        if params.interaction_g is not None:
            g_req = float(params.interaction_g)
            if g_req > g_cert * (1.0 + 1e-12) and not degenerate:
                override_rejected = True      # NUNCA se clampa en silencio
                g_cert = 0.0                  # fail-safe: sin interacción
            else:
                g_cert = g_req
        beam_energy_j = float(params.beam_particles * params.beam_energy_per_particle_kev * _J_PER_KEV)

        evidences: Tuple[Any, ...] = (
            Evidence("CP1.hermiticidad", insp["hermiticity_residual"] if math.isfinite(
                insp["hermiticity_residual"]) else 1.0, 1e-10, hard=True),
            Evidence.boolean("CP1.positividad", is_psd, hard=True),
            Evidence.boolean("CP1.dimension_fock_potencia_de_2", insp["dimension_ok"], hard=True,
                             note=f"m reconciliado = {modes}"),
            Evidence("CP1.pureza<=1", max(0.0, purity - 1.0), 1e-10, hard=True),
            Evidence("CP1.entropia_rango", max(0.0, -entropy, entropy - math.log(dim)), 1e-10, hard=True),
            Evidence("CP1.bw_isometria_haar", iso_res, 1e-11, hard=True),
            Evidence("CP1.bw_uhlmann_consistencia", cons_res, 1e-10, hard=True),
            Evidence.boolean("CP1.g_no_excede_presupuesto", not override_rejected, hard=True,
                             note="un g que viola el presupuesto se RECHAZA, no se clampa"),
            Evidence("CP1.calibracion_residual", cal_res,
                     1e-12 * max(1.0, params.disturbance_budget ** 2)),
            Evidence.boolean("CP1.energia_haz_declarada", beam_energy_j > 0.0, hard=True),
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
            "gain_calibration_residual": float(cal_res), "gain_degenerate_channel": bool(degenerate),
            "gain_override_rejected": bool(override_rejected),
            "committed_beam_energy_j": beam_energy_j,
            "rho_digest": _array_digest(np.asarray(rho_raw if rho_raw is not None else rho,
                                                   dtype=np.complex128)),
            "kernel_digest": _array_digest(k_mat),
            "exponents_digest": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "params_digest": params.digest(), "engine_passport_bound": engine_passport_bound,
            "agent_version": __version__, "engine_version": _ce.__version__, "epoch": int(epoch),
            "evidences": evidences, "truth_value": truth,
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonPhase1Certificate(
            timestamp=time.time(), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P1, core), **core
        )
        logger.info("CP1 Ω₃=%s · m=%d · g_comprometido=%.6g · E_haz=%.4e J", truth.name, modes, g_cert, beam_energy_j)
        return cert, rho_in, modes  # ──► Φ₁₂ ──► Fase 2 §2.1

    def execute_phase1(self, *a: Any, **k: Any) -> Tuple[GammaPhotonPhase1Certificate, Optional[NDArray], int]:
        """Alias heredado (v13) del terminal §1.5."""
        return self.emit_phase1_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 2 · ORIENT — MPS, dump, conteos y footprint de tunes (EPISTÉMICO)
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonPhase2Orientor:
    r"""
    FASE 2 OODA. Instruye la protección de máquina SIN actuar: el motor recibe un driver de
    sólo lectura y ningún token. Su último método alimenta el primero de Fase 3 vía Φ₂₃.
    """

    PHASE_LABEL: Final[str] = "2·ORIENT-MPS"

    def __init__(self, engine: Any, signer: ColliderSigner, *, seed: int = 20_261_123) -> None:
        self._engine = engine
        self._signer = signer
        self._seed = int(seed)

    # ─────────── §2.1 ENTRADA + COMPUERTA Φ₁₂ ───────────
    def continue_to_phase2_gate(
        self, p1: GammaPhotonPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        params: ColliderGovernanceParameters, *, expected_epoch: int,
    ) -> ColliderGateReceipt:
        r"""
        Φ₁₂ — Corrige el defecto capital de v13 (`len(hmac_signature)==64`): recalcula el
        chain_hash DESDE el payload, verifica el HMAC en tiempo constante y re-digesta los
        arrays que cruzan la compuerta (anti-TOCTOU). ADMITE ⊥ (la anomalía se procesa).
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
        )
        if not (chain_ok and sig_ok and binding_ok):
            raise ColliderGateIntegrityError(
                f"Φ₁₂: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, binding={binding_ok})."
            )
        return ColliderGateReceipt("Φ₁₂", recomputed, chain_ok, sig_ok, binding_ok, True,
                                   not p1.is_psd, "anomalía ADMITIDA: se procesa, no se exime")

    def accept_phase1_certificate(self, p1: GammaPhotonPhase1Certificate, apu_cost_matrix: NDArray,
                                  exponents: NDArray, params: ColliderGovernanceParameters, *,
                                  expected_epoch: int) -> ColliderGateReceipt:
        """Entrada formal de la Fase 2: continuación exacta de §1.5."""
        return self.continue_to_phase2_gate(p1, apu_cost_matrix, exponents, params,
                                            expected_epoch=expected_epoch)

    # ─────────── §2.3 Verificación del acta del motor ───────────
    def verify_engine_acta(self, report: Any) -> Tuple[bool, bool]:
        verified = bool(self._engine.verify_collider_report(report))
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        root_ok = hmac.compare_digest(merkle_root(leaves), report.merkle_root_hash)
        idx = int(report.acta_leaf_index)
        root_ok = root_ok and verify_merkle(leaves[idx], idx, len(leaves),
                                            report.acta_inclusion_proof, report.merkle_root_hash)
        return verified, root_ok

    # ─────────── §2.4 MPS: redundancia exacta y causa común ───────────
    @staticmethod
    def machine_protection_certificate(params: ColliderGovernanceParameters) -> MachineProtectionReport:
        r"""
        Binomial exacto k-de-n + modelo β-factor. Se verifica ADEMÁS la identidad del PFD
        contra los casos conocidos 1oo2 ⇒ (λT)²/3 y 2oo3 ⇒ (λT)², y se certifica que la
        redundancia mejora pero NO baja del piso β·p (causa común).
        """
        n, k = int(params.mps_channels), int(params.mps_required)
        p, beta = float(params.mps_p_channel), float(params.mps_beta_common_cause)
        total, ind, floor = mps_failure_probability(n, k, p, beta)
        single, _, _ = mps_failure_probability(1, 1, p, beta)
        lam, t_h = float(params.mps_lambda_du_per_h), float(params.mps_proof_test_hours)
        pfd = mps_pfd_avg(n, k, lam, t_h, beta)
        lt = (1.0 - beta) * lam * t_h
        pfd_1oo2 = mps_pfd_avg(2, 1, lam, t_h, beta)
        pfd_2oo3 = mps_pfd_avg(3, 2, lam, t_h, beta)
        res_1oo2 = abs(pfd_1oo2 - (lt ** 2 / 3.0 + beta * lam * t_h / 2.0))
        res_2oo3 = abs(pfd_2oo3 - (lt ** 2 + beta * lam * t_h / 2.0))
        return MachineProtectionReport(
            n_channels=n, k_required=k, p_channel=p, beta_common_cause=beta,
            p_fail_total=total, p_fail_independent=ind, common_cause_floor=floor,
            p_fail_single_channel=single, redundancy_improves=bool(total < single),
            floor_dominates=bool(ind < floor), lambda_du_per_h=lam, proof_test_hours=t_h,
            pfd_avg=pfd, pfd_budget=float(params.mps_pfd_budget),
            pfd_within_budget=bool(pfd <= params.mps_pfd_budget), sil_band_claimed=sil_band(pfd),
            pfd_1oo2_identity_residual=res_1oo2, pfd_2oo3_identity_residual=res_2oo3,
            note=("modelo β-factor con insumos declarados; la banda SIL es una afirmación sobre "
                  "el modelo, NO una acreditación IEC 61508"),
        )

    # ─────────── §2.5 Presupuesto térmico del dump ───────────
    @staticmethod
    def beam_dump_certificate(params: ColliderGovernanceParameters) -> BeamDumpThermalReport:
        r"""
        Modelo adiabático de depósito uniforme: E = N·E_k·[J/keV], ΔT = E/(m·c_p).
        Se certifica E ≤ capacidad declarada y ΔT ≤ ΔT_max (margen de sublimación).
        Limitación: ignora gradientes espaciales y choque térmico (declarado).
        """
        e_j = float(params.beam_particles * params.beam_energy_per_particle_kev * _J_PER_KEV)
        dt = e_j / (float(params.dump_mass_kg) * float(params.dump_specific_heat_j_kg_k))
        return BeamDumpThermalReport(
            beam_particles=float(params.beam_particles),
            energy_per_particle_kev=float(params.beam_energy_per_particle_kev),
            stored_energy_j=e_j, dump_capacity_j=float(params.dump_capacity_j),
            capacity_ok=bool(e_j <= params.dump_capacity_j), dump_mass_kg=float(params.dump_mass_kg),
            specific_heat_j_kg_k=float(params.dump_specific_heat_j_kg_k), delta_t_k=dt,
            delta_t_max_k=float(params.dump_delta_t_max_k),
            thermal_margin_ok=bool(dt <= params.dump_delta_t_max_k), joule_per_kev_used=_J_PER_KEV,
            note="modelo adiabático uniforme; sin gradientes ni choque térmico",
        )

    # ─────────── §2.6 Consistencia de conteos ───────────
    def count_statistics_certificate(self, report: Any, params: ColliderGovernanceParameters
                                     ) -> CountStatisticsReport:
        """Fano/dispersión χ² exacta + verificación MC de la tasa de aceptación bajo H₀."""
        fano, d_stat, lo, hi, p_val, ok = poisson_dispersion_test(
            params.observed_counts, float(params.counts_alpha))
        expected = float(report.phase_ii.luminosity.expected_events)
        mean = float(np.mean(params.observed_counts))
        rng = np.random.default_rng(self._seed + 61)
        n_s = len(params.observed_counts)
        accepts = 0
        draws = int(params.counts_mc_draws)
        for _ in range(draws):
            sample = rng.poisson(max(mean, 1.0), n_s)
            _, _, _, _, _, acc = poisson_dispersion_test(sample, float(params.counts_alpha))
            accepts += int(acc)
        rate = accepts / float(draws)
        nominal = 1.0 - float(params.counts_alpha)
        slack = 3.0 * math.sqrt(nominal * (1.0 - nominal) / draws)
        rel_res = abs(mean - expected) / max(expected, 1.0) if expected > 0 else 0.0
        return CountStatisticsReport(
            n_samples=n_s, sample_mean=mean, expected_from_luminosity=expected, fano_factor=fano,
            dispersion_statistic=d_stat, chi2_lower=lo, chi2_upper=hi, p_value=p_val,
            dispersion_accepts=ok, mc_acceptance_rate=rate,
            mc_rate_ok=bool(rate >= nominal - slack), relative_rate_residual=rel_res,
        )

    # ─────────── §2.7 Footprint de tunes + Kac ───────────
    @staticmethod
    def tune_footprint_certificate(params: ColliderGovernanceParameters) -> TuneFootprintReport:
        r"""
        Escaneo EXACTO de resonancias de betatrón m ν_x + n ν_y = p (KAM en aceleradores) y
        γ diophantino. Kac: E[τ_A]=1/μ ⇒ retardo esperado entre demandas del MPS.
        """
        dist, gamma, nearest, ok = tune_resonance_scan(
            float(params.tune_nu_x), float(params.tune_nu_y),
            int(params.tune_order), float(params.tune_margin))
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
            raise ColliderGovernanceError("Kac: retorno no alcanzado.")
        rt = tuple(int(v) for v in np.unique(times))
        exp_d, emp_d = 1.0 / mu, float(np.mean(times))
        return TuneFootprintReport(
            nu_x=float(params.tune_nu_x), nu_y=float(params.tune_nu_y),
            scan_order=int(params.tune_order), margin_required=float(params.tune_margin),
            min_resonance_distance=dist, diophantine_gamma=gamma, nearest_resonance=nearest,
            exact_resonance=bool(dist < 1e-12), margin_ok=ok,
            kac_expected_demand_delay=exp_d, kac_empirical_delay=emp_d,
            kac_relative_residual=abs(emp_d - exp_d) / exp_d, slater_return_times=rt,
            slater_ok=bool(len(rt) <= 3),
        )

    # ─────────── §2.8 [TERMINAL FASE 2] → Φ₂₃ → Fase 3 §3.1 ───────────
    def emit_phase2_certificate(
        self, p1: GammaPhotonPhase1Certificate, apu_cost_matrix: NDArray, exponents: NDArray,
        rho_in: Optional[NDArray], modes: int, params: ColliderGovernanceParameters, *,
        expected_epoch: int,
    ) -> GammaPhotonPhase2Certificate:
        """MÉTODO TERMINAL DE LA FASE 2: usa el g comprometido; el motor NO actúa."""
        self.accept_phase1_certificate(p1, apu_cost_matrix, exponents, params,
                                       expected_epoch=expected_epoch)
        report = self._engine.execute_gamma_photon_collider_audit(
            apu_cost_matrix, np.asarray(exponents, dtype=np.float64), rho_in,
            params=params.to_engine_parameters(modes, p1.committed_gain_g),
            authorization_token=None,                     # ← Orient NUNCA autoriza
        )
        acta_ok, root_ok = self.verify_engine_acta(report)
        qnd = report.phase_ii.phase_i.qnd
        gain_honored = abs(qnd.interaction_g - p1.committed_gain_g) <= 1e-18
        mps = self.machine_protection_certificate(params)
        dump = self.beam_dump_certificate(params)
        counts = self.count_statistics_certificate(report, params)
        tunes = self.tune_footprint_certificate(params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("CP2.acta_motor_verificada", acta_ok, hard=True),
            Evidence.boolean("CP2.merkle_motor_recalculado", root_ok, hard=True),
            Evidence.boolean("CP2.orient_no_actuo", not report.crowbar.dispatched, hard=True,
                             note="la actuación es el último commit de Fase 3"),
            Evidence.boolean("CP2.g_comprometido_honrado", gain_honored, hard=True),
            Evidence("CP2.disturbancia_vs_presupuesto",
                     max(0.0, qnd.trace_distance_disturbance - params.disturbance_budget), 0.0, hard=True),
            Evidence("CP2.backaction_db_vs_presupuesto",
                     max(0.0, -qnd.back_action_db - params.backaction_db_budget), 0.0, hard=True),
            Evidence("CP2.mps_pfd_vs_presupuesto", max(0.0, mps.pfd_avg - mps.pfd_budget), 0.0, hard=True),
            Evidence.boolean("CP2.mps_redundancia_mejora", mps.redundancy_improves, hard=True),
            Evidence("CP2.mps_identidad_1oo2", mps.pfd_1oo2_identity_residual, 1e-15, hard=True),
            Evidence("CP2.mps_identidad_2oo3", mps.pfd_2oo3_identity_residual, 1e-15, hard=True),
            Evidence.boolean("CP2.mps_piso_causa_comun_declarado", True,
                             note=f"piso β·p = {mps.common_cause_floor:.3e} (la redundancia no lo vence)"),
            Evidence.boolean("CP2.dump_capacidad", dump.capacity_ok, hard=True),
            Evidence.boolean("CP2.dump_margen_termico", dump.thermal_margin_ok, hard=True),
            Evidence.boolean("CP2.conteos_dispersion_poisson", counts.dispersion_accepts),
            Evidence.boolean("CP2.conteos_cobertura_mc", counts.mc_rate_ok, hard=True),
            Evidence.boolean("CP2.tunes_sin_resonancia_exacta", not tunes.exact_resonance, hard=True),
            Evidence.boolean("CP2.tunes_margen", tunes.margin_ok, hard=True),
            Evidence("CP2.kac_retardo_demanda", tunes.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("CP2.slater", tunes.slater_ok, hard=True),
        )
        truth = HeytingTruthValue.infimum([p1.truth_value, report.truth_value] + [e.truth for e in evidences])
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": True,
            "engine_passport": report.passport_hash, "engine_acta": report.sentence.indictment_digest,
            "engine_merkle_root": report.merkle_root_hash, "engine_truth": report.truth_value,
            "engine_acta_verified": acta_ok, "engine_merkle_recomputed_ok": root_ok,
            "engine_actuation_dispatched": bool(report.crowbar.dispatched),
            "gain_honored": bool(gain_honored),
            "disturbance_consumed": float(qnd.trace_distance_disturbance),
            "back_action_db": float(qnd.back_action_db), "machine_protection": mps,
            "beam_dump": dump, "count_statistics": counts, "tune_footprint": tunes,
            "evidences": evidences, "truth_value": truth, "epoch": int(expected_epoch),
        }
        chain = _sha256(_canonical(core))
        cert = GammaPhotonPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=report, engine_invoked=True,
            engine_acta_verified=acta_ok, engine_merkle_recomputed_ok=root_ok,
            engine_actuation_dispatched=bool(report.crowbar.dispatched), gain_honored=bool(gain_honored),
            disturbance_consumed=float(qnd.trace_distance_disturbance),
            back_action_db=float(qnd.back_action_db), machine_protection=mps, beam_dump=dump,
            count_statistics=counts, tune_footprint=tunes, evidences=evidences, truth_value=truth,
            epoch=int(expected_epoch), chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core),
        )
        logger.info("CP2 Ω₃=%s · PFD=%.3e (%s) · ΔT=%.3f K · tunes d_min=%.4f",
                    truth.name, mps.pfd_avg, mps.sil_band_claimed, dump.delta_t_k,
                    tunes.min_resonance_distance)
        return cert  # ──► Φ₂₃ ──► Fase 3 §3.1

    def execute_phase2(self, *a: Any, **k: Any) -> GammaPhotonPhase2Certificate:
        """Alias heredado (v13) del terminal §2.8."""
        return self.emit_phase2_certificate(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE 3 · DECIDE & ACT — dos hombres, RSI³ y COMMIT de actuación
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhotonPhase3Decider:
    r"""
    FASE 3 OODA. Única autoridad que autoriza y ejecuta. TRIP exige quórum 2-de-2 con
    autorizador IDs distintos; sin quórum: ISOLATE + estado seguro. Jamás dispara por omisión.
    """

    PHASE_LABEL: Final[str] = "3·SENTENCIA-COMMIT"
    _SEVERITY: Final[Dict[str, int]] = {"WARN": 0, "DERATE": 1, "ISOLATE": 2, "TRIP": 3}

    def __init__(self, signer: ColliderSigner, driver: CrowbarDriver) -> None:
        self._signer = signer
        self._driver = driver
        self._judged: set = set()
        self._last_epoch = -1
        self._pending: Dict[str, Any] = {}

    # ─────────── §3.1 ENTRADA + COMPUERTA Φ₂₃ ───────────
    @staticmethod
    def legacy_gate_would_accept(signature: str, tr_rho: float) -> bool:
        """DEPRECADO (v13): `tr_rho>0 and len(sig)==64`. NO autentica. Sólo prueba de regresión."""
        return bool(tr_rho > 0.0 and len(signature) == 64)

    def continue_to_phase3_gate(self, p2: GammaPhotonPhase2Certificate,
                                params: ColliderGovernanceParameters, *,
                                expected_epoch: int) -> ColliderGateReceipt:
        r"""
        Φ₂₃ — v13 era constante True (Lip hardcodeado). Aquí: integridad DURA (⇒ fail-safe)
        separada de la física: Lip(T³)<1 medido, PFD≤presupuesto, ΔT≤máx, sin resonancia
        exacta, y que ORIENT no haya actuado. Si la física falla ⇒ ⊥ FORZADO pero se ADMITE.
        """
        payload = p2.payload()
        recomputed = _sha256(_canonical(payload))
        chain_ok = hmac.compare_digest(recomputed, p2.chain_hash)
        sig_ok = self._signer.verify(_HMAC_P2, payload, p2.hmac_signature)
        link_ok = (p2.epoch == expected_epoch and p2.phase1.epoch == expected_epoch
                   and (not p2.engine_invoked or (p2.engine_acta_verified and p2.engine_merkle_recomputed_ok))
                   and not p2.engine_actuation_dispatched)
        if not (chain_ok and sig_ok and link_ok):
            raise ColliderGateIntegrityError(
                f"Φ₂₃: integridad comprometida (chain={chain_ok}, hmac={sig_ok}, link={link_ok})."
            )
        rep = p2.engine_report
        lip = float(getattr(getattr(rep, "rsi3", None), "lip_t3_bound", 1.0)) if rep else 1.0
        physics_ok = bool(lip < 1.0 and p2.machine_protection.pfd_within_budget
                          and p2.beam_dump.thermal_margin_ok and p2.beam_dump.capacity_ok
                          and not p2.tune_footprint.exact_resonance
                          and p2.truth_value is not HeytingTruthValue.VETOED)
        return ColliderGateReceipt("Φ₂₃", recomputed, chain_ok, sig_ok, link_ok, True, not physics_ok,
                                   "⊥ forzado: se sentencia y aísla" if not physics_ok
                                   else "MPS, dump, tunes y contracción verificados")

    def accept_phase2_certificate(self, p2: GammaPhotonPhase2Certificate,
                                  params: ColliderGovernanceParameters, *,
                                  expected_epoch: int) -> ColliderGateReceipt:
        """Entrada formal de la Fase 3: continuación exacta de §2.8."""
        return self.continue_to_phase3_gate(p2, params, expected_epoch=expected_epoch)

    # ─────────── §3.2 Functor de severidad y núcleo ¬¬ ───────────
    @classmethod
    def action_functor(cls, truth: HeytingTruthValue, *, structural: bool,
                       lesser_insufficient: bool) -> str:
        """⊤⇒WARN, ½⇒DERATE, ⊥⇒ISOLATE; TRIP exige fallo estructural + insuficiencia declarada."""
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

    # ─────────── §3.3 Regla de dos hombres ───────────
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
    def collider_rsi3(self, p2: GammaPhotonPhase2Certificate,
                      params: ColliderGovernanceParameters) -> ColliderRSI3Report:
        r"""
        NIVEL 1 · Calibración del INTERVALO DE PRUEBA T para alcanzar el presupuesto de PFD:
        PFD(T) = C(n,m)(λT)^m/(m+1) + βλT/2 con m = n−k+1 ⇒ NO lineal en T; Newton con
        salvaguarda de bisección y derivada analítica; h de Kantorovich estimado (declarado).
        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip ≤ q, Lip(T³) ≤ q³,
        cotas de Banach a priori/a posteriori y sondeo empírico.
        NIVEL 3 · Leyes de Kleisli, núcleo ¬¬, functor y jerk de capacidad MEDIDO (SG 7/3).
        """
        n, k = int(params.mps_channels), int(params.mps_required)
        m = n - k + 1
        beta = float(params.mps_beta_common_cause)
        lam = (1.0 - beta) * float(params.mps_lambda_du_per_h)
        coeff = math.comb(n, m) / (m + 1)
        lin = beta * float(params.mps_lambda_du_per_h) / 2.0
        target = float(params.mps_pfd_budget)

        def pfd_of(t: float) -> float:
            return coeff * (lam * t) ** m + lin * t

        def dpfd_of(t: float) -> float:
            return coeff * m * lam ** m * t ** (m - 1) + lin

        lo, hi = 0.0, max(1.0, float(params.mps_proof_test_hours))
        while pfd_of(hi) < target and hi < 1e9:
            hi *= 2.0
        t_k = 0.5 * (lo + hi)
        grid = np.linspace(max(lo, 1e-6), hi, 48)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([pfd_of(x) for x in grid], grid), grid)))) + _EPS
        d0 = dpfd_of(t_k)
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs((pfd_of(t_k) - target) / max(abs(d0), 1e-300)) * m_est
        errs: List[float] = []
        for _ in range(60):
            f_v, dv = pfd_of(t_k) - target, dpfd_of(t_k)
            if f_v > 0.0:
                hi = t_k
            else:
                lo = t_k
            step = t_k - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            t_k = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errs.append(abs(pfd_of(t_k) - target))
            if hi - lo < 1e-12 * max(1.0, hi):
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
        theta_star = np.array([t_k / max(hi, 1.0), margin, p2.machine_protection.pfd_avg / max(target, 1e-300)])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(20_261_124)
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
        return ColliderRSI3Report(
            calibrated_proof_test_hours=float(t_k), target_pfd=target, achieved_pfd=float(pfd_of(t_k)),
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
    def semantic_visceral_functor(p2: GammaPhotonPhase2Certificate,
                                  params: ColliderGovernanceParameters, *, action: str
                                  ) -> GammaPhotonVisceralMetrics:
        """Métricas de protección MEDIDAS + KPI financieros sólo con insumos declarados."""
        all_ev = list(p2.phase1.evidences) + list(p2.evidences)
        rep = p2.engine_report
        if rep is not None:
            all_ev += list(rep.phase_ii.phase_i.evidences) + list(rep.phase_ii.evidences) + list(rep.evidences)
        coverage = 100.0 * sum(1 for e in all_ev if e.passed) / max(1, len(all_ev))
        mps = p2.machine_protection
        reduction = mps.p_fail_single_channel / max(mps.p_fail_total, 1e-300)
        dump_pct = 100.0 * p2.beam_dump.delta_t_k / max(params.dump_delta_t_max_k, _EPS)
        fano_pct = 100.0 * abs(p2.count_statistics.fano_factor - 1.0)
        tune_pct = 100.0 * params.tune_margin / max(p2.tune_footprint.min_resonance_distance, _EPS)
        provenance = ["COVERAGE:MEASURED", "MPS:BETA_FACTOR_MODEL", "DUMP:ADIABATIC_MODEL",
                      "FANO:CHI2_TEST", "TUNES:EXACT_SCAN"]
        wacc = liq = kv = None
        kin_status = "SIN_ACTA_DEL_MOTOR"
        if rep is not None:
            kin = rep.phase_ii.kinematics
            kin_status = (f"√s={kin.sqrt_s_kev:.6f} keV · E₁={kin.photon1_energy_kev:.4f} · "
                          f"E₂={kin.photon2_energy_kev:.4f} · umbral={'OK' if kin.threshold_satisfied else 'NO'}")
            sigma = max(0.0, rep.phase_ii.bures_spohn.spohn_sigma_min)
            wacc = 100.0 * sigma / (sigma + params.exergy_reference_w)
            provenance.append("WACC:POLICY_HEURISTIC(sigma/(sigma+sigma_ref))")
        if params.exergy_price_per_joule is not None and params.capital_base:
            liq = 100.0 * p2.beam_dump.stored_energy_j * params.exergy_price_per_joule / params.capital_base
            provenance.append("LIQUIDITY:DECLARED(J_beam*price/capital)")
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
        return GammaPhotonVisceralMetrics(
            certificate_coverage_pct=float(coverage), mps_risk_reduction_factor=float(reduction),
            dump_thermal_utilization_pct=float(dump_pct), fano_deviation_pct=float(fano_pct),
            tune_margin_utilization_pct=float(tune_pct), wacc_protection_pct=wacc,
            net_liquidity_release_pct=liq, kv_cache_compression_pct=kv,
            fock_annihilation_status=kin_status,
            machine_protection_status=(f"{mps.k_required}oo{mps.n_channels} PFD={mps.pfd_avg:.3e} "
                                       f"({mps.sil_band_claimed}) · piso_cc={mps.common_cause_floor:.3e} · "
                                       f"acción={action}"),
            provenance=tuple(provenance), monotonicity_ok=monotone,
        )

    # ─────────── §3.6 COMMIT de actuación con quórum ───────────
    def commit_actuation(self, action: str, acta: str, epoch: int, *,
                         tokens: Sequence[Tuple[str, str]], params: ColliderGovernanceParameters
                         ) -> DualAuthorizationReport:
        """
        Último paso del ciclo. Verifica quórum 2-de-2 con IDs DISTINTOS (un token duplicado
        no satisface el quórum), anti-replay y ne bis in idem; sólo entonces arma y dispara.
        Sin quórum: estado seguro + escalada. Latencia MEDIDA; presupuesto físico DECLARADO.
        """
        rung = self._SEVERITY[action]
        samples = np.array([self._driver.dry_run() for _ in range(64)], dtype=np.float64)
        ids = tuple(str(i) for i, _ in tokens)
        distinct = len(set(ids)) == len(ids) and len(ids) > 0
        verified = tuple(self.verify_authorization_token(str(t), acta, epoch, rung, str(i))
                         for i, t in tokens)
        required = 2 if (rung == 3 and params.require_dual_authorization) else (1 if rung == 3 else 0)
        quorum = bool(distinct and all(verified) and len(verified) >= required)
        ne_bis = acta not in self._judged
        anti_replay = epoch > self._last_epoch
        requested = bool(rung == 3)
        authorized = bool(requested and quorum and ne_bis and anti_replay)
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
        return DualAuthorizationReport(
            ladder_rung=rung, action=action, indictment_digest=acta, epoch=int(epoch),
            quorum_required=int(required), quorum_satisfied=quorum, authorizer_ids=ids,
            duplicate_authorizer_rejected=bool(not distinct), tokens_verified=verified,
            ne_bis_in_idem_ok=bool(ne_bis), anti_replay_ok=bool(anti_replay),
            actuation_requested=requested, actuation_authorized=authorized,
            actuation_dispatched=dispatched, safe_state_applied=safe,
            driver_name=self._driver.name, driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            note=("TRIP exige quórum 2-de-2 con IDs distintos. 392.15 ns es política del camino "
                  "lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza."),
        )

    # ─────────── §3.7 [TERMINAL FASE 3] Sentencia sellada ───────────
    def emit_sovereign_state(
        self, p2: GammaPhotonPhase2Certificate, params: ColliderGovernanceParameters, *,
        expected_epoch: int, authorization_tokens: Sequence[Tuple[str, str]] = (),
        error: Optional[str] = None, forced_veto: bool = False,
    ) -> GammaPhotonColliderSovereignState:
        """MÉTODO TERMINAL: sentencia, verifica quórum, ejecuta el commit y sella el acta."""
        t0 = time.perf_counter()
        receipt = self.accept_phase2_certificate(p2, params, expected_epoch=expected_epoch)
        truth = p2.truth_value
        if receipt.forced_veto or forced_veto or error is not None:
            truth = HeytingTruthValue.VETOED
        structural = any((not e.passed) and e.hard for e in
                         list(p2.phase1.evidences) + list(p2.evidences))
        action = self.action_functor(truth, structural=structural,
                                     lesser_insufficient=params.lesser_measures_insufficient)
        rsi3 = self.collider_rsi3(p2, params)

        acta_payload = {
            "epoch": int(expected_epoch), "truth": truth, "action": action,
            "structural": bool(structural), "p2_chain": p2.chain_hash,
            "p1_chain": p2.phase1.chain_hash,
            "engine_acta": getattr(getattr(p2.engine_report, "sentence", None), "indictment_digest", ""),
            "error": error,
        }
        acta = _sha256(_canonical(acta_payload))
        auth = self.commit_actuation(action, acta, expected_epoch, tokens=authorization_tokens,
                                     params=params)
        visceral = self.semantic_visceral_functor(p2, params, action=action)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("CP3.functor_contravariante", rsi3.action_functor_ok, hard=True),
            Evidence.boolean("CP3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence("CP3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("CP3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("CP3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("CP3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("CP3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("CP3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("CP3.phi_sem_monotono", visceral.monotonicity_ok, hard=True),
            Evidence.boolean("CP3.autorizacion_habilita",
                             (not auth.actuation_dispatched) or auth.quorum_satisfied, hard=True),
            Evidence.boolean("CP3.ids_distintos", not auth.duplicate_authorizer_rejected, hard=True),
            Evidence.boolean("CP3.ne_bis_in_idem", auth.ne_bis_in_idem_ok, hard=True),
            Evidence.boolean("CP3.anti_replay", auth.anti_replay_ok, hard=True),
            Evidence.boolean("CP3.proporcionalidad",
                             auth.ladder_rung < 3 or (structural and params.lesser_measures_insufficient),
                             hard=True),
        )
        truth = HeytingTruthValue.infimum([truth] + [e.truth for e in evidences])

        escalation: Optional[ColliderEscalation] = None
        if auth.ladder_rung >= 2 or error is not None:
            reason = f"peldaño={auth.action}" + (f" | error: {error}" if error else "")
            esc_core = {"epoch": int(expected_epoch), "truth": truth, "reason": reason, "acta": acta}
            escalation = ColliderEscalation(
                epoch=int(expected_epoch), truth_value=truth, reason=reason, merkle_root_hash="",
                signature=self._signer.sign(_HMAC_ESC, esc_core), delivered=False)

        leaves: List[bytes] = [
            _canonical(p2.phase1.payload()),
            _canonical({"engine_passport": getattr(p2.engine_report, "passport_hash", ""),
                        "engine_root": getattr(p2.engine_report, "merkle_root_hash", "")}),
            _canonical(p2.machine_protection), _canonical(p2.beam_dump),
            _canonical(p2.count_statistics), _canonical(p2.tune_footprint),
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
            "kleisli_ledger": ("Φ₁:CUSTODIA", "Φ₂:MPS", "Φ₃:SENTENCIA-COMMIT"),
            "epoch": int(expected_epoch), "error": error,
            "phase2_chain_hash": p2.chain_hash, "phase1_chain_hash": p2.phase1.chain_hash,
            "authorization": {"rung": auth.ladder_rung, "action": auth.action, "acta": acta,
                              "quorum": auth.quorum_satisfied, "authorized": auth.actuation_authorized,
                              "dispatched": auth.actuation_dispatched, "safe_state": auth.safe_state_applied,
                              "ids": auth.authorizer_ids},
            "escalation_signature": escalation.signature if escalation else None,
            "merkle_leaf_count": len(leaves),
        }
        state = GammaPhotonColliderSovereignState(
            timestamp=time.time(), truth_value=truth, boolean_shadow=truth.double_negation(),
            crowbar_triggered=bool(auth.actuation_dispatched),
            crowbar_latency_ns=float(auth.trip_dispatch_ns), merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), sentence_leaf_index=s_idx,
            sentence_inclusion_proof=proof, visceral_metrics=visceral, rsi3=rsi3, authorization=auth,
            escalation=escalation, phase2=p2, evidences=evidences, kleisli_ledger=core["kleisli_ledger"],
            epoch=int(expected_epoch), state_signature=self._signer.sign(_HMAC_STATE, core), error=error,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("CP3 Ω₃=%s · %s · quórum=%s · disparo=%s",
                    truth.name, auth.action, auth.quorum_satisfied, auth.actuation_dispatched)
        return state

    def execute_phase3(self, *a: Any, **k: Any) -> GammaPhotonColliderSovereignState:
        """Alias heredado (v13) del terminal §3.7."""
        return self.emit_sovereign_state(*a, **k)


# ══════════════════════════════════════════════════════════════════════════════════════
#  §4 · FACHADA — ciclo OODA (Kleisli) con fail-safe
# ══════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovGammaPhotonColliderAgent:
    r"""
    Soberano agéntico ACTIVO del colisionador γγ. Ejecuta observe >=> orient >=> decide sobre
    la mónada Certified. El motor corre en modo epistémico. Ante cualquier fallo: FAIL-SAFE
    — ⊥, estado seguro, sin disparo y acta de emergencia firmada.
    """

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 driver: Optional[CrowbarDriver] = None, seed: int = 20_261_123) -> None:
        self._signer = ColliderSigner(secret_key)
        self._principal = principal_engine
        self._readonly = ReadOnlyCrowbarDriver()
        self._engine = ColliderEngine(secret_key=self._signer.key_bytes, principal_engine=principal_engine,
                                      seed=seed, driver=self._readonly)
        self._observer = GammaPhotonPhase1Observer(self._signer, seed=seed)
        self._orientor = GammaPhotonPhase2Orientor(self._engine, self._signer, seed=seed)
        self._decider = GammaPhotonPhase3Decider(self._signer, driver or self._make_driver())
        self._epoch = 0
        logger.info("%s v%s listo · clave=%s", type(self).__name__, __version__, self._signer.fingerprint())

    def _make_driver(self) -> CrowbarDriver:
        return SimulatedCrowbarDriver(
            lambda t: self._decider._verify_pending(t) if hasattr(self, "_decider") else False)

    # ─────────── verificación externa del estado ───────────
    def verify_sovereign_state(self, state: GammaPhotonColliderSovereignState) -> bool:
        p1, p2 = state.phase2.phase1, state.phase2
        if not self._signer.verify(_HMAC_P1, p1.payload(), p1.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p1.payload())), p1.chain_hash):
            return False
        if not self._signer.verify(_HMAC_P2, p2.payload(), p2.hmac_signature):
            return False
        if not hmac.compare_digest(_sha256(_canonical(p2.payload())), p2.chain_hash):
            return False
        if p2.engine_report is not None and not self._engine.verify_collider_report(p2.engine_report):
            return False
        leaves = [bytes.fromhex(h) for h in state.merkle_leaves]
        if merkle_root(leaves) != state.merkle_root_hash:
            return False
        i = state.sentence_leaf_index
        if not verify_merkle(leaves[i], i, len(leaves), state.sentence_inclusion_proof, state.merkle_root_hash):
            return False
        return self._signer.verify(_HMAC_STATE, state.payload(), state.state_signature)

    # ─────────── ciclo OODA con fail-safe ───────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: Optional[NDArray[np.complex128]],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        params: Optional[ColliderGovernanceParameters] = None,
        authorization_tokens: Sequence[Tuple[str, str]] = (),
        force_veto: bool = False,          # ABOLIDO
        positron_hmac_valid: Optional[bool] = None,   # ABOLIDO (compatibilidad ruidosa)
    ) -> GammaPhotonColliderSovereignState:
        """Ejecuta el ciclo completo. La actuación es el último commit, sólo con quórum."""
        if force_veto:
            raise ColliderGovernanceError(
                "force_veto está abolido: un booleano no autenticado no puede causar actuación física.")
        if positron_hmac_valid is not None:
            raise ColliderGovernanceError(
                "positron_hmac_valid está abolido: presente authorization_tokens firmados "
                "(regla de dos hombres) en su lugar.")
        params = params or ColliderGovernanceParameters()
        self._epoch += 1
        epoch = self._epoch
        exps = np.asarray(exponents, dtype=np.float64)

        def observe_m(ctx: Dict[str, Any]) -> Any:
            p1, rho_in, modes = self._observer.emit_phase1_certificate(
                rho_mac, apu_cost_matrix, exps, params, epoch=epoch)
            return Certified({**ctx, "p1": p1, "rho": rho_in, "m": modes}, ("Φ₁:CUSTODIA",), p1.truth_value)

        def orient_m(ctx: Dict[str, Any]) -> Any:
            p2 = self._orientor.emit_phase2_certificate(
                ctx["p1"], apu_cost_matrix, exps, ctx["rho"], ctx["m"], params, expected_epoch=epoch)
            return Certified({**ctx, "p2": p2}, ("Φ₂:MPS",), p2.truth_value)

        def decide_m(ctx: Dict[str, Any]) -> Any:
            st = self._decider.emit_sovereign_state(
                ctx["p2"], params, expected_epoch=epoch, authorization_tokens=authorization_tokens)
            return Certified(st, ("Φ₃:SENTENCIA-COMMIT",), st.truth_value)

        pipeline = kleisli_compose(kleisli_compose(observe_m, orient_m), decide_m)
        try:
            result = pipeline({"epoch": epoch})
            state: GammaPhotonColliderSovereignState = result.value
            if not self.verify_sovereign_state(state):
                raise ColliderGateIntegrityError("Autoverificación de la sentencia fallida.")
            logger.info("Ciclo OODA collider completado · Ω₃=%s · ledger=%s",
                        state.truth_value.name, " → ".join(result.ledger))
            return state
        except Exception as exc:                                   # FAIL-SAFE, nunca silencio
            logger.critical("FAIL-SAFE COLLIDER: %s", exc)
            return self._emergency_state(rho_mac, apu_cost_matrix, exps, params, epoch, str(exc))

    def _emergency_state(self, rho_mac: Optional[NDArray], k_mat: NDArray, exps: NDArray,
                         params: ColliderGovernanceParameters, epoch: int, message: str
                         ) -> GammaPhotonColliderSovereignState:
        """Acta de emergencia: ⊥, estado seguro, sin disparo, cadena de custodia preservada."""
        p1, rho_in, modes = self._observer.emit_phase1_certificate(rho_mac, k_mat, exps, params, epoch=epoch)
        mps = self._orientor.machine_protection_certificate(params)
        dump = self._orientor.beam_dump_certificate(params)
        tunes = self._orientor.tune_footprint_certificate(params)
        counts = CountStatisticsReport(
            n_samples=len(params.observed_counts), sample_mean=float(np.mean(params.observed_counts)),
            expected_from_luminosity=0.0, fano_factor=1.0, dispersion_statistic=0.0, chi2_lower=0.0,
            chi2_upper=0.0, p_value=1.0, dispersion_accepts=True, mc_acceptance_rate=1.0,
            mc_rate_ok=True, relative_rate_residual=0.0)
        core = {
            "phase1_chain_hash": p1.chain_hash, "engine_invoked": False, "engine_passport": "",
            "engine_acta": "", "engine_merkle_root": "", "engine_truth": HeytingTruthValue.VETOED,
            "engine_acta_verified": False, "engine_merkle_recomputed_ok": False,
            "engine_actuation_dispatched": False, "gain_honored": True, "disturbance_consumed": 0.0,
            "back_action_db": 0.0, "machine_protection": mps, "beam_dump": dump,
            "count_statistics": counts, "tune_footprint": tunes, "evidences": (),
            "truth_value": HeytingTruthValue.VETOED, "epoch": int(epoch),
        }
        chain = _sha256(_canonical(core))
        p2 = GammaPhotonPhase2Certificate(
            timestamp=time.time(), phase1=p1, engine_report=None, engine_invoked=False,
            engine_acta_verified=False, engine_merkle_recomputed_ok=False,
            engine_actuation_dispatched=False, gain_honored=True, disturbance_consumed=0.0,
            back_action_db=0.0, machine_protection=mps, beam_dump=dump, count_statistics=counts,
            tune_footprint=tunes, evidences=(), truth_value=HeytingTruthValue.VETOED, epoch=int(epoch),
            chain_hash=chain, hmac_signature=self._signer.sign(_HMAC_P2, core))
        return self._decider.emit_sovereign_state(p2, params, expected_epoch=epoch,
                                                  authorization_tokens=(), error=message, forced_veto=True)


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    agent = BoolNovikovGammaPhotonColliderAgent(secret_key=b"GAMMA_COLLIDER_AGENT_KEY_32_BYTES!",
                                                principal_engine=principal)
    k_test = np.eye(4, dtype=np.float64) + 0.02 * np.ones((4, 4))
    exps = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    P = ColliderGovernanceParameters()

    # 1) Nominal: MPS/dump/tunes conformes ⇒ ⊤, WARN, SIN disparo; Orient no actuó
    st = agent.run_sovereign_governance_cycle(None, k_test, exps)
    assert st.truth_value is HeytingTruthValue.COHERENT, st.truth_value
    assert st.authorization.action == "WARN" and not st.crowbar_triggered
    assert not st.phase2.engine_actuation_dispatched                 # ← Orient epistémico
    assert st.phase2.engine_acta_verified and st.phase2.engine_merkle_recomputed_ok
    assert st.phase2.machine_protection.pfd_within_budget
    assert st.phase2.beam_dump.thermal_margin_ok
    assert st.phase2.tune_footprint.margin_ok and not st.phase2.tune_footprint.exact_resonance
    assert st.phase2.phase1.bw_isometry_residual < 1e-11
    assert agent.verify_sovereign_state(st)
    assert st.visceral_metrics.net_liquidity_release_pct is None      # sin €/J: honesto

    # 2) DEFECTO CAPITAL: una firma FALSA de 64 caracteres (v13 la aceptaba)
    forged = dataclasses.replace(st.phase2.phase1, hmac_signature="0" * 64)
    assert GammaPhotonPhase3Decider.legacy_gate_would_accept(forged.hmac_signature, forged.tr_rho)
    try:
        agent._orientor.continue_to_phase2_gate(forged, k_test, exps, P, expected_epoch=forged.epoch)
        raise AssertionError("Φ₁₂ aceptó una firma apócrifa de 64 caracteres")
    except ColliderGateIntegrityError:
        pass

    # 3) NO vacuidad del QND: g grande ⇒ pasividad rota ⇒ ⊥, ISOLATE, SIN disparo
    broken = agent.run_sovereign_governance_cycle(
        None, k_test, exps, params=ColliderGovernanceParameters(interaction_g=0.5))
    assert broken.truth_value is HeytingTruthValue.VETOED
    assert broken.authorization.action == "ISOLATE" and not broken.crowbar_triggered
    assert broken.authorization.safe_state_applied

    # 4) Violación estructural ⇒ TRIP solicitado; SIN quórum NO dispara
    hard = ColliderGovernanceParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True)
    bad = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard)
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.authorization.ladder_rung == 3
    assert bad.authorization.actuation_requested and not bad.authorization.actuation_authorized
    assert not bad.crowbar_triggered and bad.authorization.safe_state_applied

    # 5) Token DUPLICADO (mismo autorizador) no satisface el quórum 2-de-2
    acta, ep = bad.authorization.indictment_digest, bad.epoch + 1
    tok_a = agent._decider.mint_authorization_token(acta, ep, 3, "OPERATOR")
    dup = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard,
                                               authorization_tokens=(("OPERATOR", tok_a), ("OPERATOR", tok_a)))
    assert dup.authorization.duplicate_authorizer_rejected and not dup.crowbar_triggered

    # 6) QUÓRUM válido (dos IDs distintos) ⇒ disparo autorizado y latencia MEDIDA
    ep2 = dup.epoch + 1
    acta2 = _sha256(_canonical({"epoch": ep2, "truth": HeytingTruthValue.VETOED, "action": "TRIP",
                                "structural": True, "p2_chain": dup.phase2.chain_hash,
                                "p1_chain": dup.phase2.phase1.chain_hash,
                                "engine_acta": dup.phase2.engine_report.sentence.indictment_digest,
                                "error": None}))
    t_op = agent._decider.mint_authorization_token(acta2, ep2, 3, "OPERATOR")
    t_sup = agent._decider.mint_authorization_token(acta2, ep2, 3, "SUPERVISOR")
    armed = agent.run_sovereign_governance_cycle(None, k_test, exps, params=hard,
                                                 authorization_tokens=(("OPERATOR", t_op),
                                                                       ("SUPERVISOR", t_sup)))
    assert armed.authorization.quorum_satisfied and armed.authorization.actuation_authorized
    assert armed.crowbar_triggered and armed.crowbar_latency_ns > 0.0
    assert not armed.authorization.physical_deadline_feasible        # SCR ~µs: declarado

    # 7) force_veto y positron_hmac_valid abolidos
    for kwargs in ({"force_veto": True}, {"positron_hmac_valid": True}):
        try:
            agent.run_sovereign_governance_cycle(None, k_test, exps, **kwargs)
            raise AssertionError(f"{kwargs} no fue rechazado")
        except ColliderGovernanceError:
            pass

    # 8) MPS: el piso de causa común NO se vence con redundancia; identidades 1oo2/2oo3
    mps = st.phase2.machine_protection
    assert mps.redundancy_improves and mps.p_fail_total >= mps.common_cause_floor - 1e-15
    big = ColliderGovernanceParameters(mps_channels=6, mps_required=2)
    tot6, ind6, floor6 = mps_failure_probability(6, 2, big.mps_p_channel, big.mps_beta_common_cause)
    assert ind6 < mps.p_fail_independent and abs(tot6 - floor6) < floor6 * 0.05
    assert mps.pfd_1oo2_identity_residual < 1e-15 and mps.pfd_2oo3_identity_residual < 1e-15

    # 9) Dump térmico: violación detectada; tunes resonantes detectados
    hot = agent.run_sovereign_governance_cycle(
        None, k_test, exps, params=ColliderGovernanceParameters(beam_particles=1.0e20))
    assert not hot.phase2.beam_dump.thermal_margin_ok and hot.truth_value is HeytingTruthValue.VETOED
    res = agent.run_sovereign_governance_cycle(
        None, k_test, exps, params=ColliderGovernanceParameters(tune_nu_x=0.5, tune_nu_y=0.25))
    assert res.phase2.tune_footprint.exact_resonance and res.truth_value is HeytingTruthValue.VETOED

    # 10) Merkle RFC 6962, determinismo y detección de manipulación
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    st2 = agent.run_sovereign_governance_cycle(None, k_test, exps)
    assert st2.phase2.phase1.chain_hash == st.phase2.phase1.chain_hash   # sellos sin reloj
    assert not agent.verify_sovereign_state(dataclasses.replace(st, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderAgent v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {st.truth_value.name}({int(st.truth_value)}) / {st.boolean_shadow.name}")
    print(f"Sentencia / disparo           : {st.authorization.action} / {st.crowbar_triggered}")
    print(f"Orient actuó (debe ser False) : {st.phase2.engine_actuation_dispatched}")
    print(f"m Fock reconciliado / g sellado: {st.phase2.phase1.fock_modes_reconciled} / "
          f"{st.phase2.phase1.committed_gain_g:.6g} (honrado={st.phase2.gain_honored})")
    print(f"d_BW(ρ,ρ_térmica) / isometría : {st.phase2.phase1.bures_wasserstein_to_thermal:.6f} / "
          f"{st.phase2.phase1.bw_isometry_residual:.2e}")
    print(f"MPS k-de-n / PFD / banda      : {mps.k_required}oo{mps.n_channels} / {mps.pfd_avg:.4e} / "
          f"{mps.sil_band_claimed}")
    print(f"P_fail total / indep / piso cc: {mps.p_fail_total:.4e} / {mps.p_fail_independent:.4e} / "
          f"{mps.common_cause_floor:.4e}")
    print(f"Reducción de riesgo (×)       : {st.visceral_metrics.mps_risk_reduction_factor:.2f}")
    print(f"E_haz [J] / ΔT dump [K]       : {st.phase2.beam_dump.stored_energy_j:.4e} / "
          f"{st.phase2.beam_dump.delta_t_k:.4f} (máx {st.phase2.beam_dump.delta_t_max_k:.0f})")
    print(f"Fano / D / banda χ²           : {st.phase2.count_statistics.fano_factor:.4f} / "
          f"{st.phase2.count_statistics.dispersion_statistic:.3f} / "
          f"[{st.phase2.count_statistics.chi2_lower:.3f}, {st.phase2.count_statistics.chi2_upper:.3f}]")
    print(f"Cobertura MC de dispersión    : {st.phase2.count_statistics.mc_acceptance_rate:.4f}")
    print(f"Tunes (ν_x,ν_y) / d_min       : ({st.phase2.tune_footprint.nu_x}, {st.phase2.tune_footprint.nu_y}) / "
          f"{st.phase2.tune_footprint.min_resonance_distance:.5f}")
    print(f"Resonancia más cercana (m,n,p): {st.phase2.tune_footprint.nearest_resonance} · "
          f"γ={st.phase2.tune_footprint.diophantine_gamma:.5f}")
    print(f"Kac demanda esperada/empírica : {st.phase2.tune_footprint.kac_expected_demand_delay:.3f} / "
          f"{st.phase2.tune_footprint.kac_empirical_delay:.3f}")
    print(f"RSI³ T_prueba [h] / PFD logrado: {st.rsi3.calibrated_proof_test_hours:.2f} / "
          f"{st.rsi3.achieved_pfd:.4e}")
    print(f"RSI³ q / Lip(T³) medido       : {st.rsi3.policy_lipschitz_q:.6f} / {st.rsi3.lip_t3_empirical:.6f}")
    print(f"Firma apócrifa 64 chars       : v13 aceptaría="
          f"{GammaPhotonPhase3Decider.legacy_gate_would_accept('0'*64, 1.0)} · v14 RECHAZA")
    print(f"⊥ SIN quórum                    : {bad.authorization.action} autorizado="
          f"{bad.authorization.actuation_authorized} disparo={bad.crowbar_triggered}")
    print(f"Token duplicado               : rechazado={dup.authorization.duplicate_authorizer_rejected} "
          f"disparo={dup.crowbar_triggered}")
    print(f"⊥ CON quórum 2-de-2           : {armed.authorization.action} IDs={armed.authorization.authorizer_ids} "
          f"disparo={armed.crowbar_triggered} latencia={armed.crowbar_latency_ns:.0f} ns")
    print(f"Estado Fock / cinemática      : {st.visceral_metrics.fock_annihilation_status}")
    print(f"Merkle raíz (SIN clave)       : {st.merkle_root_hash[:16]}… ({len(st.merkle_leaves)} hojas)")
    print(f"Sentencia reverificable       : {agent.verify_sovereign_state(st)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")