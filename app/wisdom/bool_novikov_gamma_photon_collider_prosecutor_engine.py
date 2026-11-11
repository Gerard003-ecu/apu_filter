# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderProsecutorEngine  v14.0.0-PhD (reedición rigurosa)     ║
║ Ubicación: app/wisdom/bool_novikov_gamma_photon_collider_prosecutor_engine.py                  ║
║ Función  : Motor FISCAL del colisionador γγ: refuta afirmaciones físicas imposibles (masa       ║
║            invariante, cota de Breit–Wheeler, Landau–Yang, Pauli, segunda ley), cuantifica la    ║
║            carga de la prueba con SPRT de Poisson y sentencia con proporcionalidad; la          ║
║            actuación física exige autorización humana criptográfica.                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA FISCAL DEL COLISIONADOR
────────────────────────────────
Un fiscal que no puede refutar nada no acusa: sella. Este módulo se rige por:

 (1) REFUTACIÓN ESTRUCTURAL PRIMERO. Una afirmación que viola una identidad exacta es un
     CONTRAFACTO, no una hipótesis: se condena con prueba, no con estadística. Se refutan:
       • masa invariante:  s_rec = 2E₁E₂(1−cosθ₁₂) ≠ s_haz  (p.ej. par colineal de 511 keV ⇒ s_rec=0)
       • cota de sección eficaz: σ_reclamada > σ_max(Breit–Wheeler) ≈ 0.26 σ_T
       • Landau–Yang: J=1 → 2γ prohibido; paridad C: (−1)^n = (−1)^{L+S}
       • Pauli: ocupación reclamada n_j > 1 en Λ(ℂ^m); conservación de carga [Q, c₊c₋]=0
       • segunda ley: W_reclamado > S(ρ‖ρ_β)/β (trabajo máximo extraíble) o σ_Spohn < 0
 (2) CARGA DE LA PRUEBA CUANTIFICADA. El sobre-reclamo de conteos es una HIPÓTESIS: SPRT de
     Poisson con α (falsa condena) y β declarados, verificados por Monte Carlo, más p-valor
     exacto de cola de Poisson. Una condena sólo estadística NUNCA produce ⊥ por sí sola.
 (3) PRESUNCIÓN DE INOCENCIA Y PROPORCIONALIDAD. ⊤⇒WARN, ½⇒DERATE, ⊥⇒ISOLATE; TRIP exige
     prueba estructural Y declaración de insuficiencia de medidas menores.
 (4) LA AUTORIZACIÓN HABILITA. v13 disparaba cuando FALTABA la autorización. Aquí el token
     HMAC ligado a {acta, época, peldaño, autorizador} es NECESARIO; sin él: estado seguro.
 (5) UN SOLO ESTADÍSTICO DECLARADO. La «anomalía» es ‖K−I‖_F/√n (adimensional, normalizada);
     los umbrales mágicos divergentes de la familia (1.5 / 1.8 / 2.5) se reportan como
     diagnóstico de inconsistencia, no como criterio.

TRES FASES ANIDADAS
    Fase I  §I.5  build_prosecutor_phase_i_handoff  ──► Fase II §II.1 accept_prosecutor_phase_i
    Fase II §II.8 build_prosecutor_phase_ii_handoff ──► Fase III §III.1 accept_prosecutor_phase_ii
    Fase III §III.7 execute_gamma_photon_collider_prosecutor_audit (terminal)
    Handoffs: ProsecutorPhaseIHandoff ⊂ ProsecutorPhaseIIHandoff ⊂ informe final.

FASE I — INSTRUMENTO DE CARGA FERMIONICO
  §I.1 Validación, regularización PSD (mismo núcleo que el principal) y compromiso.
  §I.2 Λ(ℂ^m) con Jordan–Wigner: CAR, Pauli, paridad, carga Q, operador de par.
  §I.3 Falsación adversarial (PSD, Lip, ocupación, carga) con poder de búsqueda declarado.
  §I.4 SPRT de Poisson + p-valor exacto + verificación Monte Carlo de α̂ y potencia.
  §I.5 [TERMINAL] build_prosecutor_phase_i_handoff.

FASE II — INSTRUCCIÓN FÍSICA Y REFUTACIÓN
  §II.1 accept_prosecutor_phase_i.
  §II.2 Cinemática exacta e⁻e⁺→2γ y RECONSTRUCCIÓN de masa invariante del evento reclamado.
  §II.3 Breit–Wheeler: umbral, σ(β), σ(umbral)=0 y COTA de sección eficaz (refutación).
  §II.4 Reglas de selección: paridad C y Landau–Yang como refutación estructural.
  §II.5 Segunda ley: Davies/Spohn con propagador exacto; W ≤ S(ρ‖ρ_β)/β.
  §II.6 QND dispersivo medido + Bures–Uhlmann con referencia declarada.
  §II.7 Nivel C: SNF por menores, Novikov, Floer, Melnikov.
  §II.8 [TERMINAL] build_prosecutor_phase_ii_handoff.

FASE III — SENTENCIA Y EJECUCIÓN AUTORIZADA
  §III.1 accept_prosecutor_phase_ii.
  §III.2 RSI³: Newton–Kantorovich + Banach + Kleisli + jerk medido.
  §III.3 KAM diophantino, Lindstedt y cadencia (tres huecos + Kac).
  §III.4 Sentencia Ω₃: estructural vs estadística, presunción de inocencia, núcleo ¬¬.
  §III.5 Proporcionalidad + autorización humana + ne bis in idem + anti-replay.
  §III.6 Dirac (CODATA) y crowbar HAL: despacho medido, presupuesto físico declarado.
  §III.7 [TERMINAL] execute_gamma_photon_collider_prosecutor_audit (acta + Merkle + pasaporte).

LIMITACIONES DECLARADAS
1. Modelo cinemático y de árbol; no QED dinámica ni reciprocidad detallada certificada.
2. El SPRT de Poisson controla error bajo el modelo declarado; deriva o sobre-dispersión lo
   invalidan (se verifica dispersión y se declara).
3. Las sondas de falsación son finitas: ausencia de contraejemplo ≠ prueba; los chequeos
   exactos sí son pruebas.
4. 392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza.
5. La escalada se emite como dato firmado; el transporte es del llamador.

REFERENCIAS: CODATA 2022; Breit–Wheeler (1934); Landau (1948)–Yang (1950); Jordan–Wigner (1928);
Wald (1945); Garwood (1936); Spohn (1978); Davies (1974); Uhlmann (1976); Fuchs–van de Graaf
(1999); Aharonov–Albert–Vaidman (1988); Melnikov (1963); Gromov (1985); Ekeland–Hofer (1990);
Kac (1947); Slater (1967); Kantorovich; Banach (1922); RFC 6962.
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
from scipy.special import gammainc

_ERR: Optional[Exception] = None
for _p in ("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
           "bool_novikov_thermal_exergy_weaver_engine"):
    try:
        _m = __import__(_p, fromlist=["*"])
        PrincipalEngine = _m.BoolNovikovThermalExergyWeaverEngine
        HeytingTruthValue = _m.HeytingTruthValue        # Ω₃ CANÓNICO
        Evidence = _m.Evidence
        Certified = _m.Certified
        kleisli_compose = _m.kleisli_compose
        break
    except ImportError as exc:                                       # pragma: no cover
        _ERR = exc
else:                                                                # pragma: no cover
    raise ImportError("Se requiere el motor principal v14.") from _ERR

__all__ = ["BoolNovikovGammaPhotonColliderProsecutorEngine",
           "GammaPhotonColliderProsecutorSpectrumReport", "ColliderProsecutionParameters",
           "ClaimedEvent", "ProsecutorError", "AuthorizationError",
           "CrowbarDriver", "SimulatedCrowbarDriver", "__version__"]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderProsecutorEngine")

__version__: Final[str] = "14.0.0-ENGINE-GAMMA-COLLIDER-PROSECUTOR-INVARIANT-MASS-BW-BOUND-POISSON-SPRT-SPOHN-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ME_C2_KEV: Final[float] = 510.99895069            # CODATA 2022
_TWO_ME_C2_KEV: Final[float] = 2.0 * _ME_C2_KEV    # 1021.99790138 keV
_R_E_CM: Final[float] = 2.8179403262e-13
_SIGMA_T_RE2: Final[float] = 8.0 * math.pi / 3.0
_SIGMA_T_CM2: Final[float] = _SIGMA_T_RE2 * _R_E_CM ** 2
_KRULL_FLOOR: Final[float] = 0.10
_MINOR_CAP: Final[int] = 20_000
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")
_LEGACY_THRESHOLDS: Final[Tuple[float, ...]] = (1.5, 1.8, 2.5)   # diagnóstico de divergencia
_HMAC_ACTA: Final[bytes] = b"GAMMA-COLLIDER-PROSECUTOR-ACTA-v14|"
_HMAC_AUTH: Final[bytes] = b"GAMMA-COLLIDER-PROSECUTOR-HUMAN-AUTH-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: canonicalización, Merkle, álgebra estable, SNF exacta, utilidades
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorError(Exception):
    """Error base del fiscal del colisionador."""


class AuthorizationError(ProsecutorError):
    """Actuación sin autorización válida, con replay o acta ya juzgada."""


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


def _array_digest(arr: NDArray) -> str:
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|"); h.update(str(a.shape).encode("ascii")); h.update(b"|")
    h.update(dt.str.encode("ascii")); h.update(b"|"); h.update(a.tobytes())
    return h.hexdigest()


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
    """Raíz RFC 6962 SIN clave: verificable públicamente."""
    return _mth(list(lv)).hex()


def merkle_proof(lv: Sequence[bytes], idx: int) -> Tuple[Tuple[str, bool], ...]:
    ds = list(lv)

    def path(m: int, sub: Sequence[bytes]) -> List[Tuple[str, bool]]:
        n = len(sub)
        if n == 1:
            return []
        k = 1 << ((n - 1).bit_length() - 1)
        if m < k:
            return path(m, sub[:k]) + [(_mth(sub[k:]).hex(), True)]
        return path(m - k, sub[k:]) + [(_mth(sub[:k]).hex(), False)]

    return tuple(path(idx, ds))


def verify_merkle(leaf: bytes, idx: int, cnt: int, proof: Sequence[Tuple[str, bool]], root: str) -> bool:
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


def _trace_distance(a: NDArray, b: NDArray) -> float:
    d = a - b
    return 0.5 * float(np.sum(np.abs(la.eigvalsh(0.5 * (d + d.conj().T)))))


def _fidelity_svd(rho: NDArray, sig: NDArray) -> float:
    return float(min(1.0, max(0.0, float(np.sum(la.svdvals(_psd_sqrt(rho) @ _psd_sqrt(sig)))) ** 2)))


def _fidelity_sqrt(rho: NDArray, sig: NDArray) -> float:
    """Método ALTERNATIVO (independencia): (Tr√(√ρσ√ρ))² vía eigh."""
    sr = _psd_sqrt(rho)
    core = sr @ sig @ sr
    v = la.eigvalsh(0.5 * (core + core.conj().T))
    return float(np.sum(np.sqrt(np.maximum(v, 0.0)))) ** 2


def _entropy(rho: NDArray) -> float:
    v = np.clip(la.eigvalsh(0.5 * (rho + rho.conj().T)), 0.0, 1.0)
    v = v[v > 1e-15]
    return float(-np.sum(v * np.log(v)))


def _bareiss(mat: List[List[int]]) -> int:
    n = len(mat)
    if n == 0:
        return 1
    m = [r[:] for r in mat]
    sign, prev = 1, 1
    for k in range(n - 1):
        if m[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if sw is None:
                return 0
            m[k], m[sw] = m[sw], m[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // prev
            m[i][k] = 0
        prev = m[k][k]
    return sign * m[n - 1][n - 1]


def _smith_by_minors(matrix: NDArray, cap: int = _MINOR_CAP
                     ) -> Optional[Tuple[Tuple[int, ...], int]]:
    """SNF por divisores determinantes D_k = gcd de menores k×k; d_k = D_k/D_{k−1}."""
    mi = [[int(round(float(v))) for v in row] for row in np.asarray(matrix)]
    n = len(mi); m = len(mi[0]) if n else 0
    div, rank = [1], 0
    for k in range(1, min(n, m) + 1):
        if math.comb(n, k) * math.comb(m, k) > cap:
            return None
        g = 0
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.combinations(range(m), k):
                g = math.gcd(g, abs(_bareiss([[mi[i][j] for j in cols] for i in rows])))
                if g == 1:
                    break
            if g == 1:
                break
        if g == 0:
            break
        div.append(g); rank = k
    return tuple(div[i] // div[i - 1] for i in range(1, len(div))), rank


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


def _trapz(y: NDArray, x: NDArray) -> float:
    f = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(f(y, x))


def breit_wheeler_sigma_re2(beta: float) -> float:
    r"""
    σ_γγ→e⁺e⁻ (Breit–Wheeler 1934, orden árbol) en unidades de r_e²:
        σ = (π/2)(1−β²)[(3−β⁴)ln((1+β)/(1−β)) − 2β(2−β²)],  β = √(1−1/x), x = s_γγ/(4m²c⁴).
    σ(β=0)=0 (umbral, espacio de fase nulo); pico ≈ 0.2556 σ_T; σ ≤ σ_T.
    """
    if beta <= 0.0 or beta >= 1.0:
        return 0.0
    return (math.pi / 2.0) * (1.0 - beta ** 2) * (
        (3.0 - beta ** 4) * math.log((1.0 + beta) / (1.0 - beta)) - 2.0 * beta * (2.0 - beta ** 2))


def poisson_sprt(counts: Sequence[int], lam0: float, lam1: float,
                 alpha: float, beta_err: float) -> Tuple[str, int, float, float, float]:
    r"""
    SPRT de Wald para conteos de Poisson: H₀: λ=λ₀ vs H₁: λ=λ₁>λ₀.
        z_i = x_i·ln(λ₁/λ₀) − (λ₁−λ₀),  Λ_n = Σ z_i,
        umbrales A = ln((1−β)/α), B = ln(β/(1−α)).
    Devuelve (decisión, n_observado, Λ, A, B). UNDECIDED ⇒ ½, nunca ⊥.
    """
    if lam1 <= lam0 or lam0 <= 0.0:
        raise ProsecutorError("Se requiere λ₁ > λ₀ > 0.")
    x = np.asarray(counts, dtype=np.float64).reshape(-1)
    z = x * math.log(lam1 / lam0) - (lam1 - lam0)
    cum = np.cumsum(z)
    thr_c, thr_a = math.log((1.0 - beta_err) / alpha), math.log(beta_err / (1.0 - alpha))
    up = np.flatnonzero(cum >= thr_c)
    dn = np.flatnonzero(cum <= thr_a)
    i_up = int(up[0]) if up.size else 10 ** 9
    i_dn = int(dn[0]) if dn.size else 10 ** 9
    if i_up < i_dn:
        return "CONVICT", i_up + 1, float(cum[i_up]), thr_c, thr_a
    if i_dn < i_up:
        return "ACQUIT", i_dn + 1, float(cum[i_dn]), thr_c, thr_a
    return "UNDECIDED", int(x.size), float(cum[-1]) if x.size else 0.0, thr_c, thr_a


def poisson_upper_pvalue(n_obs: int, lam0: float) -> float:
    """p-valor exacto de cola superior: P(X ≥ n | λ₀) = gammainc(n, λ₀) (n≥1); n=0 ⇒ 1."""
    if n_obs <= 0:
        return 1.0
    return float(gammainc(n_obs, lam0))


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ClaimedEvent:
    """
    Evento RECLAMADO por el operador del colisionador. El fiscal intenta REFUTARLO.
    Todos los campos son afirmaciones del reclamante, no hechos del motor.
    """

    photon_energies_kev: Tuple[float, float] = (_ME_C2_KEV, _ME_C2_KEV)
    opening_angle_deg: float = 180.0
    claimed_state_spin: int = 0                 # 0 = para-positronio, 1 = orto
    claimed_photon_multiplicity: int = 2
    claimed_sigma_cm2: float = 0.0              # sección eficaz afirmada
    claimed_occupancies: Tuple[int, ...] = (1, 1)
    claimed_charge: int = 0
    claimed_work_j: float = 0.0                 # trabajo extraíble afirmado
    declared_counts: Tuple[int, ...] = (10, 9, 11, 10, 12, 8, 10, 11, 9, 10)

    def digest(self) -> str:
        return _sha256(_canonical(self))


@dataclass(frozen=True)
class ColliderProsecutionParameters:
    """Insumos DECLARADOS del proceso; su digest se compromete en el acta."""

    fermion_modes: int = 2
    pair_coherence: float = 1.0
    interaction_g: float = 1.0e-5
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    beta_thermo: float = 2.0
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    photon_crossing_deg: float = 180.0
    photon1_energy_kev: float = _ME_C2_KEV
    photon2_energy_kev: float = _ME_C2_KEV
    sprt_lambda0: float = 10.0
    sprt_lambda1: float = 15.0
    sprt_alpha: float = 0.01
    sprt_beta: float = 0.10
    sprt_mc_paths: int = 2000
    sprt_mc_max_len: int = 200
    anomaly_threshold_normalized: float = 0.50   # POLÍTICA declarada sobre ‖K−I‖_F/√n
    falsification_probes: int = 192
    psd_floor: float = 1e-12
    ball_r: float = 1.0                          # v13: 0.28 > 0.20 ⇒ condenaba por defecto
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    kam_tau: float = 1.05
    kam_scan_order: int = 32
    cadence_points: int = 64
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    spohn_gamma0: float = 1.0
    spohn_times: Tuple[float, ...] = (0.0, 0.05, 0.2, 0.6, 1.5)
    require_human_authorization: bool = True
    lesser_measures_insufficient: bool = False
    logic_deadline_ns: float = 392.15
    gpio_write_ns: float = 25.0
    gate_driver_ns: float = 150.0
    scr_turn_on_ns: float = 1500.0
    hil_firmware_isr_ns: Optional[float] = None

    def __post_init__(self) -> None:
        if not 2 <= self.fermion_modes <= 4:
            raise ProsecutorError("fermion_modes ∈ [2,4] (Liouvilliano 4^m acotado).")
        if self.pointer_sigma <= 0.0 or self.interaction_g < 0.0:
            raise ProsecutorError("σ>0 y g≥0.")
        if not (0.0 < self.sprt_alpha < 0.5 and 0.0 < self.sprt_beta < 0.5):
            raise ProsecutorError("α,β ∈ (0,0.5).")
        if self.sprt_lambda1 <= self.sprt_lambda0 <= 0.0:
            raise ProsecutorError("λ₁ > λ₀ > 0.")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0:
            raise ProsecutorError("Radios positivos.")
        if self.positron_kinetic_kev < 0.0 or self.disturbance_budget < 0.0:
            raise ProsecutorError("Cinética y presupuesto ≥ 0.")
        if tuple(sorted(self.spohn_times)) != tuple(self.spohn_times) or self.spohn_times[0] != 0.0:
            raise ProsecutorError("spohn_times creciente y empezando en 0.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)


@dataclass(frozen=True)
class FermionicFockReport:
    """§I.2 — Λ(ℂ^m) con Jordan–Wigner: identidades EXACTAS."""

    modes: int
    dimension: int
    car_residual: float
    pauli_residual: float
    parity_residual: float
    antisymmetry_residual: float
    pair_operator_residual: float
    charge_conservation_residual: float
    number_spectrum: Tuple[int, ...]


@dataclass(frozen=True)
class AdversarialFalsificationReport:
    """§I.3 — Búsqueda de contraejemplos con poder declarado."""

    probes: int
    anomaly_normalized: float
    anomaly_threshold: float
    legacy_threshold_divergence: Tuple[float, ...]
    psd_min_eigenvalue_exact: float
    psd_counterexamples: int
    lipschitz_claimed: float
    lipschitz_probed_max: float
    lipschitz_counterexamples: int
    occupancy_claim_ok: bool
    charge_claim_ok: bool
    search_power_note: str


@dataclass(frozen=True)
class PoissonBurdenReport:
    """§I.4 — SPRT de Poisson con α/β verificados por Monte Carlo."""

    lambda0: float
    lambda1: float
    alpha_declared: float
    beta_declared: float
    log_threshold_convict: float
    log_threshold_acquit: float
    decision: str
    samples_observed: int
    log_likelihood_ratio: float
    total_counts: int
    exact_upper_pvalue: float
    fano_factor: float
    mc_false_conviction: float
    mc_power: float
    mc_asn_h0: float
    mc_asn_h1: float
    mc_rates_ok: bool


@dataclass(frozen=True)
class EventRefutationReport:
    """§II.2 — Reconstrucción de masa invariante del evento RECLAMADO."""

    claimed_energies_kev: Tuple[float, float]
    claimed_opening_deg: float
    s_reconstructed_kev2: float
    s_beam_kev2: float
    invariant_mass_residual: float
    energy_closure_residual: float
    momentum_closure_residual: float
    sqrt_s_reconstructed_kev: float
    threshold_satisfied: bool
    kinematics_refuted: bool
    refutation_reason: str


@dataclass(frozen=True)
class BreitWheelerBoundReport:
    """§II.3 — Cota de sección eficaz: refutación de sobre-reclamos."""

    x_ratio: float
    beta_velocity: float
    sigma_model_cm2: float
    sigma_max_cm2: float
    sigma_max_over_sigma_t: float
    thomson_sigma_cm2: float
    sigma_at_threshold_cm2: float
    claimed_sigma_cm2: float
    claim_exceeds_bound: bool
    bound_ok: bool


@dataclass(frozen=True)
class SelectionRulesReport:
    """§II.4 — Paridad C y Landau–Yang como refutación estructural."""

    para_c_parity: int
    ortho_c_parity: int
    claimed_spin: int
    claimed_multiplicity: int
    c_parity_allowed: bool
    landau_yang_violation: bool
    multiplicity_allowed: bool
    selection_refuted: bool


@dataclass(frozen=True)
class SecondLawProsecutionReport:
    """§II.5 — Spohn (Davies, propagador exacto) y cota de trabajo máximo."""

    stationarity_residual: float
    trace_preservation_residual: float
    positivity_min_eigenvalue: float
    relative_entropy_trace: Tuple[float, ...]
    entropy_monotone: bool
    sigma_min: float
    sigma_nonnegative: bool
    detailed_balance_residual: float
    relative_entropy_athermality: float
    max_extractable_work_j: float
    claimed_work_j: float
    work_claim_exceeds_bound: bool
    perpetual_motion_refuted: bool


@dataclass(frozen=True)
class QNDBuresProsecutionReport:
    """§II.6 — QND dispersivo medido y Bures con referencia declarada."""

    qnd_dynamical_residual: float
    state_nondisturbance_residual: float
    kraus_completeness_residual: float
    trace_distance_disturbance: float
    disturbance_apriori_bound: float
    back_action_db: float
    purity_before: float
    purity_after: float
    repeatability_residual: float
    is_qnd_passive: bool
    fidelity: float
    fidelity_alternate: float
    method_agreement_residual: float
    bures_distance: float
    fvg_lower_ok: bool
    fvg_upper_ok: bool
    cptp_monotonicity_residual: float
    legacy_self_fidelity: float


@dataclass(frozen=True)
class AlgebraGeometryReport:
    """§II.7 — Nivel C: SNF por menores, Novikov, Floer, Melnikov."""

    smith_verified: bool
    smith_invariants: Tuple[int, ...]
    smith_rank: int
    betti_0: int
    betti_1: int
    scaled_invariants: Tuple[int, ...]
    scaling_fabricates_torsion: bool
    novikov_valuation: float
    novikov_norm: float
    novikov_ultrametric_residual: float
    novikov_multiplicativity_residual: float
    novikov_filtration_compliant: bool
    floer_energy_area_residual: float
    floer_cr_bound: float
    floer_control_residual: float
    legacy_cr_constant_one: float
    gromov_obstruction: bool
    eh_monotonicity_ok: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_critical_ratio: float
    melnikov_quadrature_residual: float
    melnikov_chaos: bool


@dataclass(frozen=True)
class ProsecutorRSI3Report:
    """§III.2 — RSI³ certificado."""

    calibrated_gain: float
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


@dataclass(frozen=True)
class KAMCadenceReport:
    """§III.3 — KAM diophantino, Lindstedt, tres huecos y Kac."""

    diophantine_gamma: float
    min_small_divisor: float
    resonance_detected: bool
    homological_residual: float
    twist_det: float
    cadence_max_gap: float
    cadence_coverage_bound: float
    cadence_distinct_gaps: int
    three_gap_ok: bool
    coverage_ok: bool
    kac_expected_delay: float
    kac_empirical_delay: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    slater_ok: bool


@dataclass(frozen=True)
class SentenceAuthorizationReport:
    """§III.4–§III.5 — Sentencia, proporcionalidad y autorización."""

    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    ladder_rung: int
    action: str
    structural_refutations: Tuple[str, ...]
    statistical_conviction: bool
    charge_proven: bool
    presumption_of_innocence_honored: bool
    indictment_digest: str
    epoch: int
    token_verified: bool
    token_self_issued: bool
    ne_bis_in_idem_ok: bool
    anti_replay_ok: bool
    actuation_requested: bool
    actuation_authorized: bool


@dataclass(frozen=True)
class CrowbarProsecutionReport:
    """§III.6 — Despacho MEDIDO y presupuesto físico DECLARADO."""

    dispatched: bool
    safe_state_applied: bool
    driver_name: str
    driver_is_physical: bool
    dispatch_median_ns: float
    dispatch_p99_ns: float
    trip_dispatch_ns: float
    logic_deadline_ns: float
    firmware_isr_verified: bool
    physical_budget_ns: float
    physical_deadline_feasible: bool
    sqrt_s_kev: float
    photon_energy_cm_kev: float
    note: str


@dataclass(frozen=True)
class ProsecutorPhaseIHandoff:
    """§I.5 [TERMINAL FASE I] — entrada de §II.1."""

    fock: FermionicFockReport
    falsification: AdversarialFalsificationReport
    burden: PoissonBurdenReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    commitment_digest: str
    kernel_digest: str
    exponents_digest: str
    claim_digest: str
    rho_before: NDArray = field(repr=False)
    number_operator: NDArray = field(repr=False)
    field_hamiltonian: NDArray = field(repr=False)
    kernel_reg: NDArray = field(repr=False)


@dataclass(frozen=True)
class ProsecutorPhaseIIHandoff:
    """§II.8 [TERMINAL FASE II] — anida Fase I; entrada de §III.1."""

    phase_i: ProsecutorPhaseIHandoff
    event_refutation: EventRefutationReport
    breit_wheeler: BreitWheelerBoundReport
    selection: SelectionRulesReport
    second_law: SecondLawProsecutionReport
    qnd_bures: QNDBuresProsecutionReport
    algebra: AlgebraGeometryReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue


@dataclass(frozen=True)
class GammaPhotonColliderProsecutorSpectrumReport:
    """§III.7 [TERMINAL] — acta anidada I ⊂ II ⊂ III con Merkle y pasaporte."""

    engine_version: str
    phase_ii: ProsecutorPhaseIIHandoff
    rsi3: ProsecutorRSI3Report
    kam_cadence: KAMCadenceReport
    sentence: SentenceAuthorizationReport
    crowbar: CrowbarProsecutionReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    acta_leaf_index: int
    acta_inclusion_proof: Tuple[Tuple[str, bool], ...]
    passport_hash: str
    execution_time_ms: float

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)

    def legacy_summary(self) -> Dict[str, Any]:
        p1, p2 = self.phase_ii.phase_i, self.phase_ii
        return {
            "truth_value": self.truth_value.name,
            "dirac_report": p2.event_refutation, "qnd_report": p2.qnd_bures,
            "rademacher_walsh_norm": None,
            "smith_torsion_free": all(d == 1 for d in p2.algebra.smith_invariants),
            "betti_number_zero": p2.algebra.betti_0,
            "krull_valuation": p2.algebra.novikov_valuation,
            "novikov_norm": p2.algebra.novikov_norm,
            "cauchy_riemann_residual": p2.algebra.floer_energy_area_residual,
            "is_gromov_violated": p2.algebra.gromov_obstruction,
            "melnikov_chaos_detected": p2.algebra.melnikov_chaos,
            "banach_contraction_lip": self.rsi3.lip_t3_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "crowbar_triggered": self.crowbar.dispatched,
            "crowbar_latency_ns": self.crowbar.trip_dispatch_ns,
            "passport_hash": self.passport_hash,
        }


class CrowbarDriver(Protocol):
    name: str
    is_physical: bool

    def dry_run(self) -> int: ...
    def arm(self) -> None: ...
    def trip(self, token: str) -> int: ...
    def safe_state(self) -> None: ...


class SimulatedCrowbarDriver:
    """Driver SIMULADO (no físico). Mide despacho real; exige token válido y armado."""

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
            raise AuthorizationError("Token inválido: disparo rechazado.")
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
#  FASE I · INSTRUMENTO DE CARGA FERMIONICO
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIndictment:
    r"""
    FASE I — El fiscal no confirma: intenta REFUTAR. Y no condena sin carga de la prueba.
    """

    PHASE_LABEL: Final[str] = "I·INSTRUMENTO-DE-CARGA"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_125, driver: Optional[CrowbarDriver] = None) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_PROSECUTOR_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave fiscal efímera: actas no verificables entre procesos.")
        if len(secret_key) < 32:
            raise ProsecutorError("La clave fiscal requiere ≥ 256 bits.")
        self._key = bytes(secret_key)
        self._seed = int(seed)
        self._principal = principal_engine or PrincipalEngine(secret_key=os.urandom(32), seed=seed)
        self._epoch = 0
        self._judged: set = set()
        self._last_epoch = -1
        self._pending: Dict[str, Any] = {}
        self._driver = driver or SimulatedCrowbarDriver(self._verify_pending)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    # ─────────── §I.2 Fock fermiónico: CAR, Pauli, paridad, carga ───────────
    @staticmethod
    def fermionic_fock(params: ColliderProsecutionParameters
                       ) -> Tuple[NDArray, NDArray, FermionicFockReport]:
        r"""
        Λ(ℂ^m), c_j|n⟩ = (−1)^{Σ_{k<j}b_k}|n⊕2^j⟩ si b_j=1. Identidades EXACTAS:
        {c_i,c_j†}=δ_ij I, (c_j†)²=0 (Pauli ⇒ anti-fraude de ocupación), P=(−1)^N̂ con
        P c_j P = −c_j, c₀c₁=−c₁c₀, A_par=c₊c₋ con A_par†A_par=n₊n₋ y [Q,A_par]=0.
        """
        m = int(params.fermion_modes)
        dim = 1 << m
        ops: List[NDArray] = []
        for j in range(m):
            c = np.zeros((dim, dim), dtype=np.complex128)
            for n in range(dim):
                if (n >> j) & 1:
                    sign = -1.0 if (bin(n & ((1 << j) - 1)).count("1") % 2) else 1.0
                    c[n ^ (1 << j), n] = sign
            ops.append(c)
        eye = np.eye(dim, dtype=np.complex128)
        nums = [c.conj().T @ c for c in ops]
        n_tot = sum(nums)
        parity = np.diag([(-1.0) ** bin(n).count("1") for n in range(dim)]).astype(np.complex128)
        charge = nums[1] - nums[0]
        a_pair = ops[1] @ ops[0]
        car = max(float(la.norm(ops[i] @ ops[j].conj().T + ops[j].conj().T @ ops[i]
                                - (eye if i == j else np.zeros_like(eye)), ord="fro"))
                  for i in range(m) for j in range(m))
        pauli = max(float(la.norm(c.conj().T @ c.conj().T, ord="fro")) for c in ops)
        par = max(float(la.norm(parity @ c @ parity + c, ord="fro")) for c in ops)
        antisym = float(la.norm(ops[0] @ ops[1] + ops[1] @ ops[0], ord="fro"))
        pair_id = float(la.norm(a_pair.conj().T @ a_pair - nums[1] @ nums[0], ord="fro"))
        charge_res = float(la.norm(charge @ a_pair - a_pair @ charge, ord="fro"))
        spectrum = tuple(int(round(float(np.real(n_tot[n, n])))) for n in range(dim))
        return n_tot, charge, FermionicFockReport(
            modes=m, dimension=dim, car_residual=car, pauli_residual=pauli, parity_residual=par,
            antisymmetry_residual=antisym, pair_operator_residual=pair_id,
            charge_conservation_residual=charge_res, number_spectrum=spectrum,
        )

    # ─────────── §I.3 Falsación adversarial ───────────
    def adversarial_falsification(self, k_reg: NDArray, eigvals: NDArray, eigvecs: NDArray,
                                  claim: ClaimedEvent, params: ColliderProsecutionParameters,
                                  *, lipschitz_claimed: float) -> AdversarialFalsificationReport:
        r"""
        Sondas dirigidas a los autovectores extremos (mejor respuesta adversarial) para PSD y
        cota de Lipschitz; y chequeos EXACTOS de las afirmaciones de ocupación y carga:
        n_j ≤ 1 (Pauli) y Q_reclamada = n₊ − n₋. Ausencia de contraejemplo ≠ prueba.
        """
        n = k_reg.shape[0]
        rng = np.random.default_rng(self._seed + 3)
        probes = int(params.falsification_probes)
        psd_ce = 0
        for _ in range(probes):
            v = rng.standard_normal(n); v /= (la.norm(v) + _EPS)
            if float(v @ (k_reg @ v)) < params.psd_floor - 1e-10:
                psd_ce += 1
        v_ext = eigvecs[:, 0]
        if float(v_ext @ (k_reg @ v_ext)) < params.psd_floor - 1e-10:
            psd_ce += 1
        h_opt = 2.0 / math.sqrt(float(eigvals[0]) * float(eigvals[-1]))
        a_mat = np.eye(n) + 0.5 * h_opt * k_reg
        cayley = la.solve(a_mat, np.eye(n) - 0.5 * h_opt * k_reg, assume_a="pos")
        triple = cayley @ cayley @ cayley
        lip_max = 0.0
        for _ in range(probes):
            d = rng.standard_normal(n) * 1e-6
            den = float(la.norm(d))
            if den > 0.0:
                lip_max = max(lip_max, float(la.norm(triple @ d)) / den)
        occ = tuple(int(v) for v in claim.claimed_occupancies)
        occ_ok = bool(len(occ) == params.fermion_modes and all(0 <= v <= 1 for v in occ))
        charge_ok = bool(occ_ok and int(claim.claimed_charge) == (occ[1] - occ[0]))
        anomaly = float(la.norm(k_reg - np.eye(n), ord="fro")) / math.sqrt(n)
        return AdversarialFalsificationReport(
            probes=probes + 1, anomaly_normalized=anomaly,
            anomaly_threshold=float(params.anomaly_threshold_normalized),
            legacy_threshold_divergence=_LEGACY_THRESHOLDS,
            psd_min_eigenvalue_exact=float(np.min(eigvals)), psd_counterexamples=psd_ce,
            lipschitz_claimed=float(lipschitz_claimed), lipschitz_probed_max=lip_max,
            lipschitz_counterexamples=int(lip_max > lipschitz_claimed * (1.0 + 1e-9)),
            occupancy_claim_ok=occ_ok, charge_claim_ok=charge_ok,
            search_power_note=(f"{probes} sondas + chequeos exactos (autovalor mínimo, Pauli, carga). "
                               "Ausencia de contraejemplo NO es prueba."),
        )

    # ─────────── §I.4 SPRT de Poisson ───────────
    def burden_of_proof(self, claim: ClaimedEvent,
                        params: ColliderProsecutionParameters) -> PoissonBurdenReport:
        """SPRT de Poisson + p-valor exacto de cola + verificación MC de α̂, potencia y ASN."""
        counts = tuple(int(v) for v in claim.declared_counts)
        decision, n_obs, llr, thr_c, thr_a = poisson_sprt(
            counts, params.sprt_lambda0, params.sprt_lambda1, params.sprt_alpha, params.sprt_beta)
        total = int(sum(counts))
        p_val = poisson_upper_pvalue(total, params.sprt_lambda0 * max(len(counts), 1))
        arr = np.asarray(counts, dtype=np.float64)
        mean = float(np.mean(arr))
        fano = float(np.var(arr, ddof=1) / mean) if mean > 0 else 0.0
        rng = np.random.default_rng(self._seed + 83)
        paths, length = int(params.sprt_mc_paths), int(params.sprt_mc_max_len)
        lr = math.log(params.sprt_lambda1 / params.sprt_lambda0)
        delta = params.sprt_lambda1 - params.sprt_lambda0

        def simulate(lam: float) -> Tuple[float, float]:
            x = rng.poisson(lam, size=(paths, length)).astype(np.float64)
            cc = np.cumsum(x * lr - delta, axis=1)
            hc, ha = cc >= thr_c, cc <= thr_a
            any_c, any_a = hc.any(axis=1), ha.any(axis=1)
            first_c = np.where(any_c, hc.argmax(axis=1), length + 1)
            first_a = np.where(any_a, ha.argmax(axis=1), length + 1)
            return float(np.mean(first_c < first_a)), float(np.mean(np.minimum(np.minimum(first_c, first_a), length)))

        alpha_hat, asn0 = simulate(params.sprt_lambda0)
        power_hat, asn1 = simulate(params.sprt_lambda1)
        a, b = params.sprt_alpha, params.sprt_beta
        slack_a = 3.0 * math.sqrt(a * (1 - a) / paths)
        slack_p = 3.0 * math.sqrt(b * (1 - b) / paths)
        ok = bool(alpha_hat <= a + slack_a and power_hat >= (1.0 - b) - slack_p)
        return PoissonBurdenReport(
            lambda0=float(params.sprt_lambda0), lambda1=float(params.sprt_lambda1),
            alpha_declared=float(a), beta_declared=float(b), log_threshold_convict=thr_c,
            log_threshold_acquit=thr_a, decision=decision, samples_observed=int(n_obs),
            log_likelihood_ratio=llr, total_counts=total, exact_upper_pvalue=p_val, fano_factor=fano,
            mc_false_conviction=alpha_hat, mc_power=power_hat, mc_asn_h0=asn0, mc_asn_h1=asn1,
            mc_rates_ok=ok,
        )

    # ─────────── §I.5 [TERMINAL FASE I] → Fase II §II.1 ───────────
    def build_prosecutor_phase_i_handoff(
        self, apu_cost_matrix: NDArray, exponents: NDArray, claim: ClaimedEvent,
        params: ColliderProsecutionParameters, *, rho_mac_input: Optional[NDArray] = None,
        lipschitz_claimed: float = 0.86,
    ) -> ProsecutorPhaseIHandoff:
        """MÉTODO TERMINAL DE LA FASE I: CAR exactas, falsación y carga de la prueba sellada."""
        k_raw = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_raw.ndim != 2 or k_raw.shape[0] != k_raw.shape[1]:
            raise ProsecutorError("K debe ser cuadrada.")
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if exps.size == 0:
            raise ProsecutorError("exponents vacío: v(A) no definida.")
        _, k_reg, eigvals, eigvecs = self._principal.regularize_kernel(k_raw, psd_floor=params.psd_floor)
        n_tot, _, fock = self.fermionic_fock(params)
        dim = fock.dimension
        if rho_mac_input is None:
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = 1.0
            psi[3] = float(params.pair_coherence)     # (|00⟩+c|11⟩)/norm: coherencia de PAR
            psi /= (la.norm(psi) + _EPS)
            rho = np.outer(psi, psi.conj())
        else:
            r = np.asarray(rho_mac_input, dtype=np.complex128)
            if r.shape != (dim, dim):
                raise ProsecutorError(f"ρ debe ser {dim}×{dim} (Λ(ℂ^{fock.modes})).")
            tr = complex(np.trace(r))
            if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0:
                raise ProsecutorError("ρ: traza compleja o no positiva.")
            rho = 0.5 * (r + r.conj().T) / tr.real
        h_field = _ME_C2_KEV * n_tot
        fals = self.adversarial_falsification(k_reg, eigvals, eigvecs, claim, params,
                                              lipschitz_claimed=lipschitz_claimed)
        burden = self.burden_of_proof(claim, params)
        evidences: Tuple[Any, ...] = (
            Evidence("P1.car", fock.car_residual, 1e-12, hard=True),
            Evidence("P1.pauli", fock.pauli_residual, 1e-12, hard=True),
            Evidence("P1.paridad", fock.parity_residual, 1e-12, hard=True),
            Evidence("P1.antisimetria", fock.antisymmetry_residual, 1e-12, hard=True),
            Evidence("P1.operador_de_par", fock.pair_operator_residual, 1e-12, hard=True),
            Evidence("P1.conservacion_carga", fock.charge_conservation_residual, 1e-12, hard=True),
            Evidence.boolean("P1.psd_sin_contraejemplo", fals.psd_counterexamples == 0, hard=True),
            Evidence.boolean("P1.lip_sin_contraejemplo", fals.lipschitz_counterexamples == 0, hard=True),
            Evidence.boolean("P1.ocupacion_reclamada_admisible", fals.occupancy_claim_ok, hard=True,
                             note="Pauli: n_j ≤ 1 en Λ(ℂ^m)"),
            Evidence.boolean("P1.carga_reclamada_consistente", fals.charge_claim_ok, hard=True),
            Evidence("P1.anomalia_normalizada", max(0.0, fals.anomaly_normalized - fals.anomaly_threshold), 0.0),
            Evidence.boolean("P1.tasas_sprt_verificadas", burden.mc_rates_ok),
            Evidence.boolean("P1.carga_estadistica_probada", burden.decision == "CONVICT",
                             note="acusación estadística: sólo condena si supera el umbral de Wald"),
        )
        hard_ok = all(e.passed for e in evidences if e.hard)
        truth = (HeytingTruthValue.COHERENT if hard_ok and burden.decision != "CONVICT"
                 else (HeytingTruthValue.VETOED if not hard_ok else HeytingTruthValue.DEGRADED))
        commitment = _sha256(_canonical({
            "K": _array_digest(k_reg), "exps": _array_digest(exps), "rho": _array_digest(rho),
            "params": params.digest(), "claim": claim.digest(), "version": __version__,
        }))
        handoff = ProsecutorPhaseIHandoff(
            fock=fock, falsification=fals, burden=burden, evidences=evidences, truth_value=truth,
            commitment_digest=commitment, kernel_digest=_array_digest(k_reg),
            exponents_digest=_array_digest(exps), claim_digest=claim.digest(),
            rho_before=rho, number_operator=n_tot, field_hamiltonian=h_field, kernel_reg=k_reg,
        )
        logger.info("Fase I fiscal Ω₃=%s · SPRT=%s (α̂=%.4f, pot=%.3f) · anomalía=%.4f",
                    truth.name, burden.decision, burden.mc_false_conviction, burden.mc_power,
                    fals.anomaly_normalized)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · INSTRUCCIÓN FÍSICA Y REFUTACIÓN  (anida Fase I)
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIRefutation(ProsecutorPhaseIIndictment):
    r"""FASE II — Refuta el evento reclamado con cinemática exacta, cotas y segunda ley."""

    PHASE_LABEL: Final[str] = "II·REFUTACION-FISICA"

    def accept_prosecutor_phase_i(self, h: ProsecutorPhaseIHandoff) -> ProsecutorPhaseIHandoff:
        if not isinstance(h, ProsecutorPhaseIHandoff):
            raise ProsecutorError("Fase II requiere el ProsecutorPhaseIHandoff de §I.5.")
        return h

    # ─────────── §II.2 Refutación por masa invariante ───────────
    @staticmethod
    def refute_event_kinematics(claim: ClaimedEvent,
                                params: ColliderProsecutionParameters) -> EventRefutationReport:
        r"""
        Dos fotones reclamados (E₁,E₂,θ₁₂). Reconstrucción EXACTA:
            s_rec = 2E₁E₂(1−cosθ₁₂),   |p⃗₁+p⃗₂|² = E₁²+E₂²+2E₁E₂cosθ₁₂.
        Debe coincidir con el haz: s_haz = 2m²+2mE₊, E_tot = m+E₊, p_tot = √(E₊²−m²).
        Un par COLINEAL de 511 keV da s_rec=0 ≠ 4m² ⇒ REFUTACIÓN ESTRUCTURAL (contrafacto).
        """
        m = _ME_C2_KEV
        e1, e2 = (float(x) for x in claim.photon_energies_kev)
        th = math.radians(float(claim.opening_angle_deg))
        s_rec = 2.0 * e1 * e2 * (1.0 - math.cos(th))
        e_plus = m + float(params.positron_kinetic_kev)
        p_plus = math.sqrt(max(0.0, e_plus ** 2 - m ** 2))
        s_beam = 2.0 * m ** 2 + 2.0 * m * e_plus
        e_tot = m + e_plus
        p_rec_sq = e1 ** 2 + e2 ** 2 + 2.0 * e1 * e2 * math.cos(th)
        inv_res = abs(s_rec - s_beam) / max(s_beam, _EPS)
        en_res = abs((e1 + e2) - e_tot) / max(e_tot, _EPS)
        mom_res = abs(math.sqrt(max(0.0, p_rec_sq)) - p_plus) / max(p_plus, m)
        refuted = bool(inv_res > 1e-6 or en_res > 1e-6 or mom_res > 1e-6)
        reason = ("cinemática consistente" if not refuted else
                  f"masa invariante/cierre refutados (inv={inv_res:.3e}, E={en_res:.3e}, p={mom_res:.3e})")
        return EventRefutationReport(
            claimed_energies_kev=(e1, e2), claimed_opening_deg=float(claim.opening_angle_deg),
            s_reconstructed_kev2=s_rec, s_beam_kev2=s_beam, invariant_mass_residual=inv_res,
            energy_closure_residual=en_res, momentum_closure_residual=mom_res,
            sqrt_s_reconstructed_kev=math.sqrt(max(0.0, s_rec)),
            threshold_satisfied=bool(s_rec >= 4.0 * m ** 2 - 1e-6),
            kinematics_refuted=refuted, refutation_reason=reason,
        )

    # ─────────── §II.3 Cota de Breit–Wheeler ───────────
    @staticmethod
    def breit_wheeler_bound(claim: ClaimedEvent,
                            params: ColliderProsecutionParameters) -> BreitWheelerBoundReport:
        r"""
        x = s_γγ/(4m²c⁴) con s_γγ = 2E₁E₂(1−cosθ); β=√(1−1/x).
        σ_max = máx_β σ(β) ≈ 0.2556 σ_T (cota de árbol). Una afirmación σ_reclamada > σ_max
        es ESTRUCTURALMENTE IMPOSIBLE ⇒ refutación (sobre-reclamo de rendimiento/luminosidad).
        """
        e1, e2 = float(params.photon1_energy_kev), float(params.photon2_energy_kev)
        th = math.radians(float(params.photon_crossing_deg))
        s_gg = 2.0 * e1 * e2 * (1.0 - math.cos(th))
        x = s_gg / (4.0 * _ME_C2_KEV ** 2)
        beta = math.sqrt(max(0.0, 1.0 - 1.0 / x)) if x >= 1.0 else 0.0
        sig = breit_wheeler_sigma_re2(beta) if x >= 1.0 else 0.0
        grid = np.linspace(1e-4, 1.0 - 1e-6, 4000)
        peak = max(breit_wheeler_sigma_re2(float(b)) for b in grid)
        sig_max_cm2 = peak * _R_E_CM ** 2
        claimed = float(claim.claimed_sigma_cm2)
        exceeds = bool(claimed > sig_max_cm2 * (1.0 + 1e-9))
        return BreitWheelerBoundReport(
            x_ratio=x, beta_velocity=beta, sigma_model_cm2=sig * _R_E_CM ** 2,
            sigma_max_cm2=sig_max_cm2, sigma_max_over_sigma_t=peak / _SIGMA_T_RE2,
            thomson_sigma_cm2=_SIGMA_T_CM2, sigma_at_threshold_cm2=0.0,
            claimed_sigma_cm2=claimed, claim_exceeds_bound=exceeds,
            bound_ok=bool(sig <= _SIGMA_T_RE2 + 1e-12),
        )

    # ─────────── §II.4 Reglas de selección ───────────
    @staticmethod
    def selection_rules(claim: ClaimedEvent) -> SelectionRulesReport:
        r"""
        C = (−1)^{L+S} del positronio vs C = (−1)^n de n fotones; Landau–Yang: J=1 ↛ 2γ.
        Para (L=0,S=0): C=+1 ⇒ n par. Orto (L=0,S=1): C=−1 ⇒ n impar (3γ mínimo).
        """
        spin = int(claim.claimed_state_spin)
        n_ph = int(claim.claimed_photon_multiplicity)
        para_c, ortho_c = 1, -1
        state_c = para_c if spin == 0 else ortho_c
        c_ok = bool((-1) ** n_ph == state_c)
        ly = bool(spin == 1 and n_ph == 2)
        mult_ok = bool(n_ph >= 2)
        return SelectionRulesReport(
            para_c_parity=para_c, ortho_c_parity=ortho_c, claimed_spin=spin,
            claimed_multiplicity=n_ph, c_parity_allowed=c_ok, landau_yang_violation=ly,
            multiplicity_allowed=mult_ok, selection_refuted=bool((not c_ok) or ly or (not mult_ok)),
        )

    # ─────────── §II.5 Segunda ley: Spohn + trabajo máximo ───────────
    def second_law_prosecution(self, h: ProsecutorPhaseIHandoff, claim: ClaimedEvent,
                               params: ColliderProsecutionParameters) -> SecondLawProsecutionReport:
        r"""
        Generador de Davies en la base propia de H con tasas γ_{i←j}=γ₀√(p_i/p_j) que cumplen
        balance detallado ⇒ ρ_β punto fijo. Propagador EXACTO vec(ρ_t)=e^{𝓛t}vec(ρ₀).
        SPOHN: S(ρ_t‖ρ_β) no creciente ⇒ σ=−dS/dt ≥ 0.
        TRABAJO MÁXIMO EXTRAÍBLE: W ≤ F(ρ)−F(ρ_β) = S(ρ‖ρ_β)/β (athermality). Un reclamo
        W > ese valor es un perpetuum mobile de segunda especie ⇒ REFUTACIÓN ESTRUCTURAL.
        (Unidades: k_B=1, β en 1/keV ⇒ S/β en keV; se convierte a J con 1.602176634e-16.)
        """
        k_reg = h.kernel_reg
        en, u = la.eigh(k_reg)
        d = int(en.size)
        beta = float(params.beta_thermo)
        logp = -beta * (en - en[0]); logp -= float(np.max(logp))
        logp -= math.log(float(np.sum(np.exp(logp))))
        p = np.exp(logp)
        eye = np.eye(d)
        jumps: List[NDArray] = []
        db_res = 0.0
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                rate = float(params.spohn_gamma0) * math.sqrt(p[i] / p[j])
                back = float(params.spohn_gamma0) * math.sqrt(p[j] / p[i])
                db_res = max(db_res, abs(rate * p[j] - back * p[i]))
                lop = np.zeros((d, d), dtype=np.complex128)
                lop[i, j] = math.sqrt(rate)
                jumps.append(lop)
        liouv = np.zeros((d * d, d * d), dtype=np.complex128)
        for lop in jumps:
            ld_l = lop.conj().T @ lop
            liouv += np.kron(lop.conj(), lop) - 0.5 * np.kron(eye, ld_l) - 0.5 * np.kron(ld_l.T, eye)
        rho_eig = u.conj().T @ h.rho_before @ u
        vec0 = rho_eig.reshape(-1, order="F")
        gibbs_vec = np.diag(p).astype(np.complex128).reshape(-1, order="F")

        def rel_ent(x: NDArray) -> float:
            v = np.clip(la.eigvalsh(0.5 * (x + x.conj().T)), 0.0, 1.0)
            v = v[v > 1e-15]
            return float(np.sum(v * np.log(v)) - np.sum(logp * np.real(np.diag(x))))

        trace: List[float] = []
        traces: List[float] = []
        min_eig = float("inf")
        for t in params.spohn_times:
            rt = (la.expm(liouv * float(t)) @ vec0).reshape((d, d), order="F")
            rt = 0.5 * (rt + rt.conj().T)
            trace.append(rel_ent(rt))
            traces.append(float(np.real(np.trace(rt))))
            min_eig = min(min_eig, float(np.min(la.eigvalsh(rt))))
        stat = float(la.norm(la.expm(liouv * float(params.spohn_times[-1])) @ gibbs_vec - gibbs_vec))
        monotone = all(trace[i + 1] <= trace[i] + 1e-10 for i in range(len(trace) - 1))
        sigmas = [-(trace[i + 1] - trace[i]) / (params.spohn_times[i + 1] - params.spohn_times[i])
                  for i in range(len(trace) - 1)]
        s_rel = rel_ent(rho_eig)
        w_max_kev = s_rel / beta
        w_max_j = w_max_kev * 1.602176634e-16
        claimed_w = float(claim.claimed_work_j)
        exceeds = bool(claimed_w > w_max_j * (1.0 + 1e-9) + 1e-30)
        return SecondLawProsecutionReport(
            stationarity_residual=stat,
            trace_preservation_residual=float(max(abs(t - 1.0) for t in traces)),
            positivity_min_eigenvalue=min_eig, relative_entropy_trace=tuple(trace),
            entropy_monotone=bool(monotone), sigma_min=float(min(sigmas)),
            sigma_nonnegative=bool(min(sigmas) >= -1e-10), detailed_balance_residual=db_res,
            relative_entropy_athermality=s_rel, max_extractable_work_j=w_max_j,
            claimed_work_j=claimed_w, work_claim_exceeds_bound=exceeds,
            perpetual_motion_refuted=exceeds,
        )

    # ─────────── §II.6 QND dispersivo + Bures ───────────
    def qnd_bures_certificate(self, h: ProsecutorPhaseIHandoff,
                              params: ColliderProsecutionParameters) -> QNDBuresProsecutionReport:
        r"""
        Canal dispersivo EXACTO ρ'_mn = ρ_mn exp(−g²(N_m−N_n)²/8σ²), UNITAL ⇒ dB ≤ 0.
        Bures con referencia DECLARADA (ρ pre y ρ térmica), F por SVD + método alternativo,
        Fuchs–van de Graaf y monotonía CPTP. Diagnóstico: Tr(√ρ√ρ)=Tr ρ ≡ 1 (v13).
        """
        rho = h.rho_before
        n_vals = np.real(np.diag(h.number_operator))
        g, sigma = float(params.interaction_g), float(params.pointer_sigma)
        delta = n_vals[:, None] - n_vals[None, :]
        factor = np.exp(-(g ** 2) * delta ** 2 / (8.0 * sigma ** 2))
        after = rho * factor
        y = np.linspace(g * float(n_vals.min()) - 10 * sigma, g * float(n_vals.max()) + 10 * sigma, 4001)
        m_k = (2 * math.pi * sigma ** 2) ** (-0.25) * np.exp(-(y[:, None] - g * n_vals[None, :]) ** 2 / (4 * sigma ** 2))
        integral = np.array([_trapz(m_k[:, j] ** 2, y) for j in range(n_vals.size)])
        completeness = float(np.max(np.abs(integral - 1.0)))
        d = rho.shape[0]
        bound = 0.5 * math.sqrt(d) * (g ** 2 * float(np.max(delta ** 2)) / (8 * sigma ** 2)) * float(la.norm(rho, ord="fro"))
        n_fro = float(la.norm(h.number_operator, ord="fro")) + _EPS
        h_fro = float(la.norm(h.field_hamiltonian, ord="fro")) + _EPS
        r_fro = float(la.norm(rho, ord="fro")) + _EPS
        dyn = float(la.norm(h.number_operator @ h.field_hamiltonian
                            - h.field_hamiltonian @ h.number_operator, ord="fro")) / (n_fro * h_fro)
        nondist = float(la.norm(rho @ h.number_operator - h.number_operator @ rho, ord="fro")) / (r_fro * n_fro)
        pb = float(np.real(np.trace(rho @ rho)))
        pa = float(np.real(np.trace(after @ after)))
        db = 10.0 * math.log10(max(pa, 1e-300) / max(pb, 1e-300))
        dist = _trace_distance(rho, after)
        repeat = float(np.max(np.abs(np.real(np.diag(after)) - np.real(np.diag(rho)))))
        passive = bool(dist <= params.disturbance_budget and db >= -params.backaction_db_budget)
        en = n_vals * _ME_C2_KEV
        lp = -params.beta_thermo * (en - en[0]); lp -= float(np.max(lp))
        lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        fid = _fidelity_svd(rho, rho_th)
        alt = _fidelity_sqrt(rho, rho_th)
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid)))
        f2 = _fidelity_svd(rho * factor, rho_th * factor)
        mono = max(0.0, math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f2))) - bures)
        dt_th = _trace_distance(rho, rho_th)
        legacy = float(np.real(np.trace(_psd_sqrt(rho) @ _psd_sqrt(rho))))
        return QNDBuresProsecutionReport(
            qnd_dynamical_residual=dyn, state_nondisturbance_residual=nondist,
            kraus_completeness_residual=completeness, trace_distance_disturbance=dist,
            disturbance_apriori_bound=bound, back_action_db=db, purity_before=pb, purity_after=pa,
            repeatability_residual=repeat, is_qnd_passive=passive, fidelity=fid,
            fidelity_alternate=alt, method_agreement_residual=abs(fid - alt), bures_distance=bures,
            fvg_lower_ok=bool(1.0 - math.sqrt(fid) <= dt_th + 1e-12),
            fvg_upper_ok=bool(dt_th <= math.sqrt(max(0.0, 1.0 - fid)) + 1e-12),
            cptp_monotonicity_residual=mono, legacy_self_fidelity=legacy,
        )

    # ─────────── §II.7 Álgebra y geometría (nivel C) ───────────
    def algebra_geometry(self, k_reg: NDArray, exponents: NDArray,
                         params: ColliderProsecutionParameters) -> AlgebraGeometryReport:
        """SNF por menores (independiente), Novikov con axiomas, Floer con cota, Melnikov."""
        n = k_reg.shape[0]
        rows, cols = np.triu_indices(n, 1)
        mask = np.abs(k_reg[rows, cols]) > 1e-12
        rows, cols = rows[mask], cols[mask]
        m_edges = int(rows.size)
        inc = np.zeros((n, m_edges))
        if m_edges:
            inc[rows, np.arange(m_edges)] = 1.0
            inc[cols, np.arange(m_edges)] = -1.0
        sm = _smith_by_minors(inc) if m_edges else ((), 0)
        verified = sm is not None
        inv, rank = sm if verified else ((), 0)
        scaled = _smith_by_minors(inc * 100.0) if m_edges else ((), 0)
        s_inv = scaled[0] if scaled is not None else ()
        fabrication = bool(verified and all(d == 1 for d in inv) and any(d > 1 for d in s_inv))
        lam = np.asarray(exponents, dtype=np.float64).reshape(-1)
        val = float(np.min(lam))
        vb = float(np.min(lam + 0.25))
        ultra = max(0.0, math.exp(-val) - max(math.exp(-val), math.exp(-vb))) / (math.exp(-val) + _EPS)
        mult = abs(float(np.min(lam[:, None] + lam[None, :])) - 2.0 * val) / (1.0 + 2.0 * val)

        nodes, w = np.polynomial.legendre.leggauss(48)
        s = 0.5 * nodes - 0.5; ws = 0.5 * w
        tau = np.arange(32) * (2 * math.pi / 32)
        S, T = np.meshgrid(s, tau, indexing="ij")
        c = 0.25
        u_s = np.zeros(S.shape + (2,)); u_t = np.zeros(S.shape + (2,)); fifth = 0.0
        for kk, aa, bb in ((1, 0.7, 0.0), (2, 0.3, -0.2)):
            lm = kk + c; env = np.exp(lm * S)
            px = aa * np.cos(kk * T) - bb * np.sin(kk * T)
            py = aa * np.sin(kk * T) + bb * np.cos(kk * T)
            u_s[..., 0] += lm * env * px; u_s[..., 1] += lm * env * py
            u_t[..., 0] += -kk * env * py; u_t[..., 1] += kk * env * px
            fifth += abs(lm) ** 5 * float(np.max(env)) * math.hypot(aa, bb)
        dtau = 2 * math.pi / 32
        e_v = float(np.sum(ws[:, None] * dtau * np.sum(u_s ** 2, axis=-1)))
        a_v = float(np.sum(ws[:, None] * dtau * (-u_s[..., 1] * u_t[..., 0] + u_s[..., 0] * u_t[..., 1])))
        ea = abs(e_v - a_v) / (abs(e_v) + abs(a_v) + _EPS)
        cr_bound = 4.0 * ((1.0 / 64.0) ** 4 / 30.0) * fifth
        ctrl = 0.5 * float(np.max(np.abs(u_s)))
        g4 = np.random.default_rng(7).standard_normal(4)
        j4 = np.block([[np.zeros((2, 2)), -np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
        legacy_one = float(la.norm(j4 @ g4) / la.norm(g4))     # ≡ 1: vacuidad de v13
        caps_e = sorted(math.pi * mm * a for a in (params.ball_r ** 2, (0.8 * params.ball_r) ** 2)
                        for mm in range(1, 5))[:4]
        caps_b = [math.pi * params.ball_r ** 2 * math.ceil((k + 1) / 2) for k in range(4)]
        eh_ok = bool(all(x <= y * (1 + 1e-12) for x, y in zip(caps_e, caps_b)))
        om, ga, de = params.melnikov_omega, params.melnikov_gamma, params.melnikov_delta
        amp = math.sqrt(2.0) * math.pi * ga * om / math.cosh(math.pi * om / 2.0)
        off = 4.0 * de / 3.0
        crit = 4.0 * math.cosh(math.pi * om / 2.0) / (3.0 * math.sqrt(2.0) * math.pi * om)
        t = np.linspace(-40.0, 40.0, 1 << 14)
        xd = -math.sqrt(2.0) / np.cosh(t) * np.tanh(t)
        qres = 0.0
        for tp in np.linspace(0.0, 2 * math.pi / om, 8, endpoint=False):
            num = _trapz(xd * (ga * np.cos(om * (t + tp)) - de * xd), t)
            qres = max(qres, abs(num - (amp * math.sin(om * tp) - off)) / (amp + off + 1e-12))
        return AlgebraGeometryReport(
            smith_verified=bool(verified), smith_invariants=inv, smith_rank=int(rank),
            betti_0=int(n - rank), betti_1=int(m_edges - rank), scaled_invariants=s_inv,
            scaling_fabricates_torsion=fabrication, novikov_valuation=val,
            novikov_norm=math.exp(-val), novikov_ultrametric_residual=ultra,
            novikov_multiplicativity_residual=mult,
            novikov_filtration_compliant=bool(val >= _KRULL_FLOOR),
            floer_energy_area_residual=ea, floer_cr_bound=cr_bound, floer_control_residual=ctrl,
            legacy_cr_constant_one=legacy_one, gromov_obstruction=bool(params.ball_r > params.cylinder_r),
            eh_monotonicity_ok=eh_ok, melnikov_amplitude=amp, melnikov_offset=off,
            melnikov_critical_ratio=crit, melnikov_quadrature_residual=qres,
            melnikov_chaos=bool(amp > off + 1e-12),
        )

    # ─────────── §II.8 [TERMINAL FASE II] → Fase III §III.1 ───────────
    def build_prosecutor_phase_ii_handoff(
        self, h: ProsecutorPhaseIHandoff, exponents: NDArray, claim: ClaimedEvent,
        params: ColliderProsecutionParameters,
    ) -> ProsecutorPhaseIIHandoff:
        """MÉTODO TERMINAL DE LA FASE II: refutación cinemática, cotas y segunda ley."""
        h = self.accept_prosecutor_phase_i(h)
        ev = self.refute_event_kinematics(claim, params)
        bw = self.breit_wheeler_bound(claim, params)
        sel = self.selection_rules(claim)
        sl = self.second_law_prosecution(h, claim, params)
        qb = self.qnd_bures_certificate(h, params)
        alg = self.algebra_geometry(h.kernel_reg, exponents, params)
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("P2.cinematica_no_refutada", not ev.kinematics_refuted, hard=True,
                             note=ev.refutation_reason),
            Evidence.boolean("P2.umbral_reconstruido", ev.threshold_satisfied, hard=True),
            Evidence.boolean("P2.sigma_reclamada_bajo_cota", not bw.claim_exceeds_bound, hard=True,
                             note="σ > máx Breit–Wheeler es estructuralmente imposible"),
            Evidence.boolean("P2.cota_thomson", bw.bound_ok, hard=True),
            Evidence("P2.sigma_umbral_cero", bw.sigma_at_threshold_cm2, 0.0, hard=True),
            Evidence.boolean("P2.paridad_c_permitida", sel.c_parity_allowed, hard=True),
            Evidence.boolean("P2.sin_violacion_landau_yang", not sel.landau_yang_violation, hard=True),
            Evidence.boolean("P2.multiplicidad_admisible", sel.multiplicity_allowed, hard=True),
            Evidence("P2.spohn_estacionariedad", sl.stationarity_residual, 1e-9, hard=True),
            Evidence("P2.spohn_traza", sl.trace_preservation_residual, 1e-10, hard=True),
            Evidence("P2.spohn_positividad", -sl.positivity_min_eigenvalue, 1e-10, hard=True),
            Evidence.boolean("P2.spohn_monotonia", sl.entropy_monotone, hard=True),
            Evidence.boolean("P2.spohn_sigma_no_negativa", sl.sigma_nonnegative, hard=True),
            Evidence("P2.balance_detallado", sl.detailed_balance_residual, 1e-12, hard=True),
            Evidence.boolean("P2.trabajo_reclamado_admisible", not sl.work_claim_exceeds_bound, hard=True,
                             note="W ≤ S(ρ‖ρ_β)/β (athermality): perpetuum mobile refutado"),
            Evidence("P2.kraus_completitud", qb.kraus_completeness_residual, 1e-9, hard=True),
            Evidence("P2.qnd_dinamico", qb.qnd_dynamical_residual, 1e-12, hard=True),
            Evidence("P2.qnd_repetibilidad", qb.repeatability_residual, 1e-14, hard=True),
            Evidence.boolean("P2.pasividad_presupuestada", qb.is_qnd_passive, hard=True),
            Evidence("P2.disturbancia_vs_cota", qb.trace_distance_disturbance - qb.disturbance_apriori_bound,
                     1e-16, hard=True),
            Evidence.boolean("P2.pureza_no_creciente", qb.purity_after <= qb.purity_before + 1e-12, hard=True),
            Evidence("P2.bures_metodos", qb.method_agreement_residual, 1e-9, hard=True),
            Evidence.boolean("P2.fvg_inferior", qb.fvg_lower_ok, hard=True),
            Evidence.boolean("P2.fvg_superior", qb.fvg_upper_ok, hard=True),
            Evidence("P2.bures_monotonia_cptp", qb.cptp_monotonicity_residual, 1e-12, hard=True),
            Evidence("P2.legacy_fidelidad_tautologica", abs(qb.legacy_self_fidelity - 1.0), 1e-12,
                     note="diagnóstico: Tr(√ρ√ρ)=Tr ρ ≡ 1 (v13)"),
            Evidence.boolean("P2.smith_verificado", alg.smith_verified, hard=True),
            Evidence.boolean("P2.smith_sin_torsion", all(d == 1 for d in alg.smith_invariants), hard=True),
            Evidence("P2.novikov_ultrametrica", alg.novikov_ultrametric_residual, 1e-12, hard=True),
            Evidence("P2.novikov_multiplicatividad", alg.novikov_multiplicativity_residual, 1e-12, hard=True),
            Evidence.boolean("P2.novikov_filtracion", alg.novikov_filtration_compliant),
            Evidence("P2.floer_energia_area", alg.floer_energy_area_residual, 1e-10, hard=True),
            Evidence.boolean("P2.floer_control", alg.floer_control_residual > 10 * alg.floer_cr_bound, hard=True),
            Evidence.boolean("P2.gromov_sin_obstruccion", not alg.gromov_obstruction, hard=True),
            Evidence.boolean("P2.eh_monotonia", alg.eh_monotonicity_ok),
            Evidence("P2.melnikov_cuadratura", alg.melnikov_quadrature_residual, 1e-8, hard=True),
            Evidence.boolean("P2.sin_caos_homoclinico", not alg.melnikov_chaos, hard=True),
            Evidence("P2.legacy_cr_constante_uno", abs(alg.legacy_cr_constant_one - 1.0), 1e-12,
                     note="diagnóstico: el residual CR de v13 era la constante 1"),
        )
        truth = HeytingTruthValue.infimum([h.truth_value] + [e.truth for e in evidences])
        out = ProsecutorPhaseIIHandoff(
            phase_i=h, event_refutation=ev, breit_wheeler=bw, selection=sel, second_law=sl,
            qnd_bures=qb, algebra=alg, evidences=evidences, truth_value=truth,
        )
        logger.info("Fase II fiscal Ω₃=%s · s_rec=%.4f keV² · σ_recl=%.3e · W_recl=%.3e J",
                    truth.name, ev.s_reconstructed_kev2, bw.claimed_sigma_cm2, sl.claimed_work_j)
        return out  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · SENTENCIA, PROPORCIONALIDAD Y EJECUCIÓN AUTORIZADA  (anida II y I)
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIISentence(ProsecutorPhaseIIRefutation):
    r"""FASE III — Sentencia proporcional; la actuación exige autorización humana."""

    PHASE_LABEL: Final[str] = "III·SENTENCIA-AUTORIZADA"

    def accept_prosecutor_phase_ii(self, h: ProsecutorPhaseIIHandoff) -> ProsecutorPhaseIIHandoff:
        if not isinstance(h, ProsecutorPhaseIIHandoff):
            raise ProsecutorError("Fase III requiere el ProsecutorPhaseIIHandoff de §II.8.")
        self.accept_prosecutor_phase_i(h.phase_i)
        return h

    # ─────────── §III.2 RSI³ ───────────
    def prosecutor_rsi3(self, h: ProsecutorPhaseIIHandoff,
                        params: ColliderProsecutionParameters) -> ProsecutorRSI3Report:
        r"""
        NIVEL 1 · Calibración de g por Newton con bisección sobre el proxy Frobenius exacto;
        h de Kantorovich estimado (declarado).
        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip ≤ q, Lip(T³) ≤ q³, cotas
        de Banach y sondeo empírico.
        NIVEL 3 · Leyes de Kleisli, núcleo ¬¬ y jerk de capacidad MEDIDO (SG 7/3).
        """
        p1 = h.phase_i
        n_vals = np.real(np.diag(p1.number_operator))
        sigma = float(params.pointer_sigma)
        d2 = (n_vals[:, None] - n_vals[None, :]) ** 2 / (8.0 * sigma ** 2)
        w2 = np.abs(p1.rho_before) ** 2

        def s_of(g: float) -> float:
            return float(np.sum(w2 * (1.0 - np.exp(-(g ** 2) * d2)) ** 2))

        def ds_of(g: float) -> float:
            e = np.exp(-(g ** 2) * d2)
            return float(np.sum(w2 * 4.0 * g * d2 * (1.0 - e) * e))

        target = float(params.disturbance_budget) ** 2
        lo, hi = 0.0, max(1e-8, sigma * 0.01)
        while s_of(hi) < target and hi < 1e6:
            hi *= 2.0
        g_k = 0.5 * (lo + hi)
        grid = np.linspace(max(lo, 1e-12), hi, 48)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([s_of(x) for x in grid], grid), grid)))) + _EPS
        d0 = ds_of(g_k)
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs((s_of(g_k) - target) / max(abs(d0), 1e-300)) * m_est
        errs: List[float] = []
        for _ in range(50):
            f_v, dv = s_of(g_k) - target, ds_of(g_k)
            if f_v > 0.0:
                hi = g_k
            else:
                lo = g_k
            step = g_k - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g_k = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errs.append(abs(s_of(g_k) - target))
            if hi - lo < 1e-18 * max(1.0, hi):
                break
        err = np.array(errs)
        cap = -np.log10(np.maximum(err, 1e-300))
        active = np.where(err > 1e-13 * max(err[0], 1e-300))[0]
        d1 = d2c = d3 = 0.0
        if active.size >= 7:
            ctr = min(max(int(active[active.size // 2]), 3), cap.size - 4)
            seg = cap[ctr - 3: ctr + 4]
            d1 = float(_savgol(7, 3, 1) @ seg); d2c = float(_savgol(7, 3, 2) @ seg); d3 = float(_savgol(7, 3, 3) @ seg)
        ev = list(p1.evidences) + list(h.evidences)
        hard = [e for e in ev if e.hard]
        margin = float(sum(1 for e in hard if e.passed)) / max(1, len(hard))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        theta_star = np.array([g_k, margin, h.breit_wheeler.x_ratio])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(self._seed + 23)
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
        vals = list(HeytingTruthValue)
        nucleus = bool(all(int(a) <= int(a.double_negation()) for a in vals)
                       and all(a.double_negation().double_negation() == a.double_negation() for a in vals)
                       and all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                               for a in vals for b in vals))
        return ProsecutorRSI3Report(
            calibrated_gain=float(g_k), kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_iterations=int(err.size),
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(emp),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, capacity_d1=d1, capacity_d2=d2c, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2c > 0 and d3 > 0),
            kleisli_monad_laws_ok=laws, nucleus_axioms_ok=nucleus,
        )

    # ─────────── §III.3 KAM + cadencia ───────────
    @staticmethod
    def kam_cadence(params: ColliderProsecutionParameters,
                    frequencies: Sequence[float] = (1.0, _GOLDEN),
                    n_theta: int = 32) -> KAMCadenceReport:
        """Diophantina por barrido entero, Lindstedt en Fourier, tres huecos y Kac."""
        omega = np.asarray(frequencies, dtype=np.float64)
        d = int(omega.size); K = int(params.kam_scan_order)
        gamma, min_div = float("inf"), float("inf")
        for kv in itertools.product(range(-K, K + 1), repeat=d):
            l1 = sum(abs(v) for v in kv)
            if l1 == 0 or l1 > K:
                continue
            div = abs(float(np.dot(np.asarray(kv, dtype=np.float64), omega)))
            min_div = min(min_div, div); gamma = min(gamma, div * l1 ** float(params.kam_tau))
        shape = (n_theta,) * d
        grids = np.meshgrid(*[np.arange(n_theta) * (2 * math.pi / n_theta)] * d, indexing="ij")
        h1 = np.cos(grids[0]) + (0.5 * np.cos(grids[0] - grids[1]) + 0.25 * np.sin(2 * grids[1]) if d > 1 else 0.0)
        h1_hat = np.fft.fftn(h1)
        km = np.meshgrid(*[np.fft.fftfreq(n_theta, d=1.0 / n_theta)] * d, indexing="ij")
        divisor = sum(omega[i] * km[i] for i in range(d))
        nyq = np.zeros(shape, dtype=bool)
        for i in range(d):
            nyq |= np.abs(km[i]) == (n_theta // 2)
        safe = (np.abs(divisor) > 1e-14) & (~nyq)
        w_hat = np.zeros(shape, dtype=np.complex128)
        w_hat[safe] = 1j * h1_hat[safe] / divisor[safe]
        recon = np.real(np.fft.ifftn(-1j * divisor * w_hat))
        tgt = h1 - float(np.mean(h1))
        hom = float(la.norm(recon - tgt)) / (float(la.norm(tgt)) + _EPS)
        twist = float(la.det(np.diag(np.linspace(1.0, 1.3, d))))
        alpha, npts = _GOLDEN, int(params.cadence_points)
        convs = _convergents(alpha, npts)
        pts = np.sort(np.mod(np.arange(npts) * alpha, 1.0))
        gaps = np.diff(np.concatenate([pts, pts[:1] + 1.0]))
        uniq = np.unique(np.round(gaps, 9))
        ident = float(abs(uniq[2] - uniq[0] - uniq[1])) if uniq.size == 3 else (0.0 if uniq.size <= 3 else float("inf"))
        qp, qc = (convs[-2][1], convs[-1][1]) if len(convs) >= 2 else (1, convs[-1][1])
        bound = 1.0 / max(1, qp) + 1.0 / max(1, qc)
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
            raise ProsecutorError("Kac: retorno no alcanzado.")
        rt = tuple(int(v) for v in np.unique(times))
        exp_d, emp_d = 1.0 / mu, float(np.mean(times))
        return KAMCadenceReport(
            diophantine_gamma=float(gamma), min_small_divisor=float(min_div),
            resonance_detected=bool(min_div < 1e-12), homological_residual=hom, twist_det=twist,
            cadence_max_gap=float(np.max(gaps)), cadence_coverage_bound=float(bound),
            cadence_distinct_gaps=int(uniq.size),
            three_gap_ok=bool(uniq.size <= 3 and ident <= 1e-8),
            coverage_ok=bool(float(np.max(gaps)) <= bound + 1e-12),
            kac_expected_delay=exp_d, kac_empirical_delay=emp_d,
            kac_relative_residual=abs(emp_d - exp_d) / exp_d, slater_return_times=rt,
            slater_ok=bool(len(rt) <= 3),
        )

    # ─────────── §III.5 Autorización humana ───────────
    def mint_authorization_token(self, acta: str, epoch: int, rung: int,
                                 authorizer: str = "HUMAN") -> str:
        return hmac.new(self.key_bytes, _HMAC_AUTH + _canonical(
            {"acta": acta, "epoch": int(epoch), "rung": int(rung), "authorizer": authorizer}
        ), hashlib.sha256).hexdigest()

    def verify_authorization_token(self, token: str, acta: str, epoch: int, rung: int,
                                   authorizer: str = "HUMAN") -> bool:
        return hmac.compare_digest(self.mint_authorization_token(acta, epoch, rung, authorizer), str(token))

    def _verify_pending(self, token: str) -> bool:
        p = self._pending
        return bool(p) and self.verify_authorization_token(token, p["acta"], p["epoch"], p["rung"])

    @staticmethod
    def proportionality_rung(truth: HeytingTruthValue, *, structural: bool,
                             lesser_insufficient: bool) -> int:
        """⊤⇒0, ½⇒1, ⊥⇒2; TRIP exige refutación ESTRUCTURAL + insuficiencia declarada."""
        if truth is HeytingTruthValue.COHERENT:
            return 0
        if truth is HeytingTruthValue.DEGRADED:
            return 1
        return 3 if (structural and lesser_insufficient) else 2

    # ─────────── §III.6 Crowbar medido ───────────
    def crowbar_certificate(self, sentence: SentenceAuthorizationReport, ev: EventRefutationReport,
                            params: ColliderProsecutionParameters) -> CrowbarProsecutionReport:
        """Mide N despachos; dispara SÓLO con token y peldaño 3; si no, estado seguro."""
        samples = np.array([self._driver.dry_run() for _ in range(64)], dtype=np.float64)
        dispatched, trip_ns, safe = False, 0.0, False
        if sentence.actuation_authorized and sentence.ladder_rung == 3:
            self._pending = {"acta": sentence.indictment_digest, "epoch": sentence.epoch, "rung": 3}
            token = self.mint_authorization_token(sentence.indictment_digest, sentence.epoch, 3)
            self._driver.arm()
            trip_ns = float(self._driver.trip(token))
            dispatched = True
        elif sentence.ladder_rung >= 2:
            self._driver.safe_state()
            safe = True
        sqrt_s = math.sqrt(max(0.0, ev.s_beam_kev2))
        return CrowbarProsecutionReport(
            dispatched=dispatched, safe_state_applied=safe, driver_name=self._driver.name,
            driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            sqrt_s_kev=sqrt_s, photon_energy_cm_kev=0.5 * sqrt_s,
            note=("392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) "
                  "no lo alcanza. Sin token válido NO hay disparo: estado seguro + escalada."),
        )

    # ─────────── §III.7 [TERMINAL] Orquestador ───────────
    def execute_gamma_photon_collider_prosecutor_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho_mac_input: Optional[NDArray] = None,
        *,
        claimed_event: Optional[ClaimedEvent] = None,
        params: Optional[ColliderProsecutionParameters] = None,
        authorization_token: Optional[str] = None,
        lipschitz_claimed: float = 0.86,
        force_veto: bool = False,          # ABOLIDO
        positron_hmac_valid: Optional[bool] = None,   # ABOLIDO
    ) -> GammaPhotonColliderProsecutorSpectrumReport:
        r"""
        MÉTODO TERMINAL. Ejecuta §I.5→§II.1→§II.8→§III.1→§III.2..§III.6, sentencia en Ω₃ con
        presunción de inocencia y sella el acta (Merkle SIN clave + pasaporte determinista).
        La actuación requiere token humano; los booleanos heredados se rechazan.
        """
        if force_veto:
            raise ProsecutorError("force_veto está abolido: un booleano no autenticado no puede actuar.")
        if positron_hmac_valid is not None:
            raise ProsecutorError("positron_hmac_valid está abolido: presente authorization_token firmado.")
        t0 = time.perf_counter()
        params = params or ColliderProsecutionParameters()
        claim = claimed_event or ClaimedEvent()
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)

        h1 = self.build_prosecutor_phase_i_handoff(
            k_mat, exps, claim, params, rho_mac_input=rho_mac_input, lipschitz_claimed=lipschitz_claimed)
        h2 = self.build_prosecutor_phase_ii_handoff(h1, exps, claim, params)
        h2 = self.accept_prosecutor_phase_ii(h2)
        rsi3 = self.prosecutor_rsi3(h2, params)
        kam = self.kam_cadence(params)
        evidences: Tuple[Any, ...] = (
            Evidence("P3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("P3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("P3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("P3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("P3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("P3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("P3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence.boolean("P3.sin_resonancia_kam", not kam.resonance_detected, hard=True),
            Evidence("P3.gamma_diophantino", max(0.0, 1e-3 - kam.diophantine_gamma), 0.0),
            Evidence("P3.ecuacion_homologica", kam.homological_residual, 1e-10, hard=True),
            Evidence.boolean("P3.twist_no_degenerado", abs(kam.twist_det) > 1e-12, hard=True),
            Evidence.boolean("P3.tres_huecos", kam.three_gap_ok, hard=True),
            Evidence.boolean("P3.cobertura_cadencia", kam.coverage_ok, hard=True),
            Evidence.boolean("P3.slater", kam.slater_ok, hard=True),
            Evidence("P3.kac_retardo", kam.kac_relative_residual, 5e-3, hard=True),
        )
        base_truth = HeytingTruthValue.infimum([h2.truth_value] + [e.truth for e in evidences])
        refutations: List[str] = []
        if h2.event_refutation.kinematics_refuted:
            refutations.append("masa_invariante/cierre_cinematico")
        if h2.breit_wheeler.claim_exceeds_bound:
            refutations.append("seccion_eficaz_sobre_cota_Breit-Wheeler")
        if h2.selection.selection_refuted:
            refutations.append("reglas_de_seleccion_C/Landau-Yang")
        if h2.second_law.perpetual_motion_refuted:
            refutations.append("trabajo_sobre_cota_athermality")
        if not h1.falsification.occupancy_claim_ok:
            refutations.append("ocupacion_viola_Pauli")
        if not h1.falsification.charge_claim_ok:
            refutations.append("carga_no_consistente")
        if h2.algebra.gromov_obstruction:
            refutations.append("obstruccion_de_Gromov")
        if not h2.qnd_bures.is_qnd_passive:
            refutations.append("pasividad_QND_violada")
        structural = bool(refutations) or any((not e.passed) and e.hard for e in
                                              list(h1.evidences) + list(h2.evidences) + list(evidences))
        statistical = h1.burden.decision == "CONVICT"
        if base_truth is HeytingTruthValue.COHERENT and statistical:
            base_truth = HeytingTruthValue.DEGRADED     # estadística sola nunca ⇒ ⊥
        truth = base_truth
        charge_proven = bool(structural or statistical)
        rung = self.proportionality_rung(truth, structural=structural,
                                         lesser_insufficient=params.lesser_measures_insufficient)
        self._epoch += 1
        epoch = self._epoch
        acta_payload = {
            "epoch": epoch, "truth": truth, "rung": rung, "structural": structural,
            "refutations": tuple(refutations), "statistical": statistical,
            "commitment": h1.commitment_digest, "claim": h1.claim_digest, "version": __version__,
        }
        acta = _sha256(_canonical(acta_payload))
        ne_bis = acta not in self._judged
        anti_replay = epoch > self._last_epoch
        self_issued = bool(authorization_token is None and rung == 3
                           and not params.require_human_authorization)
        token = authorization_token or (self.mint_authorization_token(acta, epoch, rung)
                                        if self_issued else "")
        token_ok = bool(token) and self.verify_authorization_token(str(token), acta, epoch, rung)
        requested = bool(rung == 3)
        authorized = bool(requested and token_ok and ne_bis and anti_replay)
        sentence = SentenceAuthorizationReport(
            truth_value=truth, boolean_shadow=truth.double_negation(), ladder_rung=rung,
            action=_LADDER[rung], structural_refutations=tuple(refutations),
            statistical_conviction=statistical, charge_proven=charge_proven,
            presumption_of_innocence_honored=bool(not (statistical and not structural
                                                       and truth is HeytingTruthValue.VETOED)),
            indictment_digest=acta, epoch=epoch, token_verified=token_ok,
            token_self_issued=self_issued, ne_bis_in_idem_ok=bool(ne_bis),
            anti_replay_ok=bool(anti_replay), actuation_requested=requested,
            actuation_authorized=authorized,
        )
        crowbar = self.crowbar_certificate(sentence, h2.event_refutation, params)
        self._judged.add(acta)
        self._last_epoch = epoch
        leaves: List[bytes] = [
            _canonical({"commitment": h1.commitment_digest, "K": h1.kernel_digest,
                        "exps": h1.exponents_digest, "claim": h1.claim_digest}),
            _canonical(h1.fock), _canonical(h1.falsification), _canonical(h1.burden),
            _canonical(h2.event_refutation), _canonical(h2.breit_wheeler), _canonical(h2.selection),
            _canonical(h2.second_law), _canonical(h2.qnd_bures), _canonical(h2.algebra),
            _canonical(rsi3), _canonical(kam), _canonical(acta_payload),
            _canonical({k: v for k, v in _freeze(crowbar).items()
                        if k not in ("dispatch_median_ns", "dispatch_p99_ns", "trip_dispatch_ns")}),
        ]
        root = merkle_root(leaves)
        acta_idx = 12
        proof = merkle_proof(leaves, acta_idx)
        payload = {"version": __version__, "truth": truth, "acta": acta, "merkle_root": root,
                   "rung": rung, "authorized": authorized, "commitment": h1.commitment_digest}
        passport = hmac.new(self.key_bytes, _HMAC_ACTA + _canonical(payload), hashlib.sha256).hexdigest()
        report = GammaPhotonColliderProsecutorSpectrumReport(
            engine_version=__version__, phase_ii=h2, rsi3=rsi3, kam_cadence=kam, sentence=sentence,
            crowbar=crowbar, evidences=evidences, truth_value=truth, merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), acta_leaf_index=acta_idx,
            acta_inclusion_proof=proof, passport_hash=passport,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("Fiscal γγ v14 Ω₃=%s · %s · refutaciones=%d · autorizado=%s · disparo=%s",
                    truth.name, sentence.action, len(refutations), authorized, crowbar.dispatched)
        return report

    def verify_prosecutor_report(self, r: GammaPhotonColliderProsecutorSpectrumReport) -> bool:
        """Reverifica Merkle (público), prueba de inclusión del acta y pasaporte HMAC."""
        lv = [bytes.fromhex(x) for x in r.merkle_leaves]
        if merkle_root(lv) != r.merkle_root_hash:
            return False
        i = r.acta_leaf_index
        if not verify_merkle(lv[i], i, len(lv), r.acta_inclusion_proof, r.merkle_root_hash):
            return False
        payload = {"version": r.engine_version, "truth": r.truth_value,
                   "acta": r.sentence.indictment_digest, "merkle_root": r.merkle_root_hash,
                   "rung": r.sentence.ladder_rung, "authorized": r.sentence.actuation_authorized,
                   "commitment": r.phase_ii.phase_i.commitment_digest}
        exp = hmac.new(self.key_bytes, _HMAC_ACTA + _canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(exp, r.passport_hash)


class BoolNovikovGammaPhotonColliderProsecutorEngine(ProsecutorPhaseIIISentence):
    """Fachada pública (nombre conservado). Anidamiento: Fase I ⊂ II ⊂ III ⊂ Fachada."""

    PHASE_LABEL: Final[str] = "FACHADA·FISCAL-GAMMA-PHOTON-COLLIDER"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    eng = BoolNovikovGammaPhotonColliderProsecutorEngine(secret_key=b"GAMMA_PROSECUTOR_KEY_32_BYTES_MIN!")
    k_test = np.eye(4, dtype=np.float64) + 0.02 * np.ones((4, 4))
    exps = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)

    # 1) Nominal: evento consistente, sin sobre-reclamos ⇒ ABSUELTO, ⊤, SIN disparo
    rep = eng.execute_gamma_photon_collider_prosecutor_audit(k_test, exps)
    p1, p2 = rep.phase_ii.phase_i, rep.phase_ii
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert p1.burden.decision == "ACQUIT" and rep.sentence.ladder_rung == 0
    assert not rep.crowbar.dispatched and not rep.sentence.structural_refutations
    assert p1.fock.car_residual < 1e-12 and p1.fock.charge_conservation_residual < 1e-12
    assert p2.qnd_bures.state_nondisturbance_residual > 1e-6        # ← NO vacuidad
    assert p2.qnd_bures.is_qnd_passive and p2.qnd_bures.back_action_db <= 0.0
    assert p2.event_refutation.invariant_mass_residual < 1e-12
    assert abs(p2.event_refutation.s_reconstructed_kev2 - 4 * _ME_C2_KEV ** 2) < 1e-6
    assert p2.second_law.sigma_nonnegative and p2.second_law.entropy_monotone
    assert abs(p2.qnd_bures.legacy_self_fidelity - 1.0) < 1e-12     # tautología de v13
    assert abs(p2.algebra.legacy_cr_constant_one - 1.0) < 1e-12     # vacuidad de v13
    assert p2.algebra.scaling_fabricates_torsion
    assert eng.verify_prosecutor_report(rep)

    # 2) REFUTACIÓN: par COLINEAL de 511 keV ⇒ s_rec = 0 ≠ 4m² ⇒ ⊥ estructural
    collinear = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(opening_angle_deg=0.0))
    assert collinear.phase_ii.event_refutation.s_reconstructed_kev2 == 0.0
    assert collinear.phase_ii.event_refutation.kinematics_refuted
    assert collinear.truth_value is HeytingTruthValue.VETOED
    assert "masa_invariante/cierre_cinematico" in collinear.sentence.structural_refutations

    # 3) REFUTACIÓN: σ reclamada sobre la cota de Breit–Wheeler
    over_sigma = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(claimed_sigma_cm2=1.0e-24))
    assert over_sigma.phase_ii.breit_wheeler.claim_exceeds_bound
    assert over_sigma.truth_value is HeytingTruthValue.VETOED

    # 4) REFUTACIÓN: Landau–Yang (J=1 → 2γ) y ocupación que viola Pauli
    ly = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(claimed_state_spin=1, claimed_photon_multiplicity=2))
    assert ly.phase_ii.selection.landau_yang_violation and ly.truth_value is HeytingTruthValue.VETOED
    pauli = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(claimed_occupancies=(2, 0)))
    assert not pauli.phase_ii.phase_i.falsification.occupancy_claim_ok
    assert pauli.truth_value is HeytingTruthValue.VETOED

    # 5) REFUTACIÓN: perpetuum mobile (W reclamado > S(ρ‖ρ_β)/β)
    perpetuum = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(claimed_work_j=1.0))
    assert perpetuum.phase_ii.second_law.work_claim_exceeds_bound
    assert perpetuum.truth_value is HeytingTruthValue.VETOED

    # 6) Carga estadística probada (conteos inflados) ⇒ condena SPRT pero NUNCA ⊥ por sí sola
    rng = np.random.default_rng(11)
    inflated = eng.execute_gamma_photon_collider_prosecutor_audit(
        k_test, exps, claimed_event=ClaimedEvent(declared_counts=tuple(int(v) for v in rng.poisson(15.0, 60))))
    assert inflated.phase_ii.phase_i.burden.decision == "CONVICT"
    assert inflated.truth_value is HeytingTruthValue.DEGRADED
    assert inflated.sentence.presumption_of_innocence_honored

    # 7) Violación estructural + insuficiencia ⇒ TRIP solicitado; SIN token NO dispara
    hard = ColliderProsecutionParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True)
    bad = eng.execute_gamma_photon_collider_prosecutor_audit(k_test, exps, params=hard)
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.sentence.ladder_rung == 3
    assert bad.sentence.actuation_requested and not bad.sentence.actuation_authorized
    assert not bad.crowbar.dispatched and bad.crowbar.safe_state_applied

    # 8) CON token ligado al acta ⇒ disparo autorizado y latencia MEDIDA
    tok = eng.mint_authorization_token(bad.sentence.indictment_digest, bad.sentence.epoch + 1, 3)
    armed = eng.execute_gamma_photon_collider_prosecutor_audit(k_test, exps, params=hard,
                                                               authorization_token=tok)
    assert armed.sentence.actuation_authorized and armed.crowbar.dispatched
    assert armed.crowbar.trip_dispatch_ns > 0.0 and not armed.crowbar.physical_deadline_feasible

    # 9) Booleanos heredados abolidos
    for kw in ({"force_veto": True}, {"positron_hmac_valid": True}):
        try:
            eng.execute_gamma_photon_collider_prosecutor_audit(k_test, exps, **kw)
            raise AssertionError(f"{kw} no fue rechazado")
        except ProsecutorError:
            pass

    # 10) SPRT: umbrales de Wald, tasas MC y p-valor exacto; BW: pico y umbral
    b = p1.burden
    assert abs(b.log_threshold_convict - math.log(0.9 / 0.01)) < 1e-12
    assert b.mc_false_conviction <= b.alpha_declared + 0.01 and b.mc_power >= 0.87
    assert abs(poisson_upper_pvalue(0, 10.0) - 1.0) < 1e-12
    bw = p2.breit_wheeler
    assert bw.sigma_at_threshold_cm2 == 0.0 and 0.20 < bw.sigma_max_over_sigma_t < 0.30
    assert abs(bw.thomson_sigma_cm2 - 6.6524587321e-25) / 6.6524587321e-25 < 1e-6

    # 11) SNF por menores, Merkle, determinismo y anti-manipulación
    tri = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=np.float64)
    assert _smith_by_minors(tri)[0] == (1, 1) and _smith_by_minors(tri * 100.0)[0] == (100, 100)
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    rep2 = eng.execute_gamma_photon_collider_prosecutor_audit(k_test, exps)
    assert rep2.passport_hash != rep.passport_hash                # época distinta ⇒ acta distinta
    assert not eng.verify_prosecutor_report(dataclasses.replace(rep, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderProsecutorEngine v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {rep.truth_value.name}({int(rep.truth_value)}) / "
          f"{rep.sentence.boolean_shadow.name}")
    print(f"Sentencia / disparo           : {rep.sentence.action} / {rep.crowbar.dispatched}")
    print(f"CAR / Pauli / carga           : {p1.fock.car_residual:.2e} / {p1.fock.pauli_residual:.2e} / "
          f"{p1.fock.charge_conservation_residual:.2e}")
    print(f"Anomalía normalizada / umbral : {p1.falsification.anomaly_normalized:.4f} / "
          f"{p1.falsification.anomaly_threshold:.2f} (umbrales heredados {p1.falsification.legacy_threshold_divergence})")
    print(f"SPRT Poisson λ₀→λ₁ / decisión : {b.lambda0:.1f}→{b.lambda1:.1f} / {b.decision}")
    print(f"α̂ / potencia / ASN(H₀,H₁)     : {b.mc_false_conviction:.4f} / {b.mc_power:.3f} / "
          f"({b.mc_asn_h0:.1f}, {b.mc_asn_h1:.1f})")
    print(f"p-valor exacto de cola        : {b.exact_upper_pvalue:.4f} · Fano={b.fano_factor:.3f}")
    print(f"s_rec / s_haz [keV²]          : {p2.event_refutation.s_reconstructed_kev2:.4f} / "
          f"{p2.event_refutation.s_beam_kev2:.4f} (residuo {p2.event_refutation.invariant_mass_residual:.2e})")
    print(f"σ BW modelo / máx / σ_T       : {bw.sigma_model_cm2:.3e} / {bw.sigma_max_cm2:.3e} / "
          f"{bw.thomson_sigma_cm2:.3e} cm²")
    print(f"Pico σ_γγ/σ_T                 : {bw.sigma_max_over_sigma_t:.4f}")
    print(f"Spohn σ_min / W_máx [J]       : {p2.second_law.sigma_min:.4e} / "
          f"{p2.second_law.max_extractable_work_j:.4e}")
    print(f"[ρ,N̂] / D_tr / dB             : {p2.qnd_bures.state_nondisturbance_residual:.3e} / "
          f"{p2.qnd_bures.trace_distance_disturbance:.3e} / {p2.qnd_bures.back_action_db:.3e}")
    print(f"SNF real / ×100               : {p2.algebra.smith_invariants} / {p2.algebra.scaled_invariants}")
    print(f"RSI³ g / Lip(T³) medido       : {rep.rsi3.calibrated_gain:.6e} / {rep.rsi3.lip_t3_empirical:.6f}")
    print(f"Kac esperado / empírico       : {rep.kam_cadence.kac_expected_delay:.3f} / "
          f"{rep.kam_cadence.kac_empirical_delay:.3f} · Slater {rep.kam_cadence.slater_return_times}")
    print(f"Refutación colineal           : {collinear.truth_value.name} · s_rec="
          f"{collinear.phase_ii.event_refutation.s_reconstructed_kev2:.1f} keV²")
    print(f"Refutaciones (σ, LY, Pauli, W): {over_sigma.truth_value.name}, {ly.truth_value.name}, "
          f"{pauli.truth_value.name}, {perpetuum.truth_value.name}")
    print(f"Sólo estadística (½)          : {inflated.truth_value.name} · inocencia="
          f"{inflated.sentence.presumption_of_innocence_honored}")
    print(f"⊥ SIN token / CON token       : disparo={bad.crowbar.dispatched} / {armed.crowbar.dispatched} "
          f"(latencia {armed.crowbar.trip_dispatch_ns:.0f} ns)")
    print(f"Merkle raíz (SIN clave)       : {rep.merkle_root_hash[:16]}… ({len(rep.merkle_leaves)} hojas)")
    print(f"Acta reverificable            : {eng.verify_prosecutor_report(rep)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")