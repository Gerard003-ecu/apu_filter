# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderEngine  v14.0.0-PhD (reedición rigurosa)               ║
║ Ubicación: app/wisdom/bool_novikov_gamma_photon_collider_engine.py                             ║
║ Función  : Motor ACTIVO del colisionador γγ: espacio de Fock fermiónico (CAR/Jordan–Wigner),     ║
║            cinemática exacta e⁻e⁺→2γ, sección eficaz de Breit–Wheeler γγ→e⁺e⁻, estadística de    ║
║            luminosidad, QND dispersivo medido y actuación SÓLO con autorización humana.         ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL MOTOR ACTIVO
─────────────────────────
(1) LA AUTORIZACIÓN HABILITA. v13 disparaba el crowbar cuando `positron_hmac_valid` era FALSO:
    la ausencia de autorización causaba actuación física. Aquí el token HMAC ligado a
    {acta, época, peldaño, autorizador} es condición NECESARIA; sin él: ISOLATE + estado seguro.
    `force_veto` queda abolido: un booleano no autenticado no puede tocar silicio.
(2) LA FÍSICA NO LA DECIDE UNA NORMA. v13 decidía si «ocurrió» la aniquilación con `‖K−I‖≤2.5`.
    Aquí la cinemática es relativista y exacta, con identidades verificables, y las reglas de
    selección (paridad C, Landau–Yang) determinan el canal permitido.
(3) LO QUE SE MIDE SE MIDE. La retroacción QND se calcula con el canal gaussiano cerrado; la
    latencia de despacho se mide; la sección eficaz se evalúa con Breit–Wheeler; el intervalo
    de conteos es el exacto de Garwood con cobertura verificada por Monte Carlo.
(4) LO QUE ES MODELO SE DECLARA. Este módulo NO es una simulación de QED dinámica: es un
    modelo cinemático + reglas de selección + secciones eficaces de árbol, con fermiones en
    espacio de Fock truncado. Las amplitudes dinámicas y la reciprocidad detallada entre
    aniquilación y creación de pares quedan fuera del certificado y se declaran.

TRES FASES ANIDADAS
    Fase I  §I.5  build_collider_phase_i_handoff  ──► Fase II §II.1 accept_collider_phase_i
    Fase II §II.7 build_collider_phase_ii_handoff ──► Fase III §III.1 accept_collider_phase_ii
    Fase III §III.7 execute_gamma_photon_collider_audit (terminal)
    Handoffs: ColliderPhaseIHandoff ⊂ ColliderPhaseIIHandoff ⊂ GammaPhotonColliderSpectrumReport

FASE I — SUSTRATO FERMIONICO Y QND
  §I.1 Validación, regularización PSD (mismo núcleo que el principal) y compromiso.
  §I.2 Fock fermiónico Λ(ℂ^m): CAR, Pauli, paridad, antisimetría, operador de par, carga Q.
  §I.3 QND dispersivo del número: canal gaussiano exacto, completitud, cota a priori.
  §I.4 AAV: valor débil, regla de suma, anomalía, límite débil del puntero (orden medido).
  §I.5 [TERMINAL] build_collider_phase_i_handoff.

FASE II — CINEMATICA, SECCIONES EFICACES Y ESTADISTICA
  §II.1 accept_collider_phase_i.
  §II.2 Cinemática e⁻e⁺→2γ exacta; prohibición de 1γ; reglas de selección C/Landau–Yang.
  §II.3 Breit–Wheeler γγ→e⁺e⁻: umbral, σ(β), σ(umbral)=0, pico≈0.26σ_T, saturación en reposo.
  §II.4 Polarización derivada de (ε₁×ε₂)·k̂: singlete, 1 bit, CHSH=2√2 (Tsirelson).
  §II.5 Luminosidad R=Lσ, Poisson, intervalo exacto de Garwood + cobertura Monte Carlo.
  §II.6 Bures–Uhlmann (referencias declaradas) + Spohn del semigrupo de dephasing.
  §II.7 Álgebra/geometría nivel C: SNF por menores, Novikov, Floer, Melnikov. [TERMINAL]

FASE III — SENTENCIA, AUTORIZACION Y ACTUACION
  §III.1 accept_collider_phase_ii.
  §III.2 RSI³: Newton–Kantorovich + Banach + Kleisli + jerk medido.
  §III.3 KAM diophantino, Lindstedt y cadencia (tres huecos + Kac).
  §III.4 Sentencia Ω₃ y escalera de proporcionalidad WARN→DERATE→ISOLATE→TRIP.
  §III.5 Autorización humana HMAC, ne bis in idem, anti-replay.
  §III.6 Crowbar HAL: despacho medido y presupuesto físico declarado.
  §III.7 [TERMINAL] execute_gamma_photon_collider_audit (acta + Merkle + pasaporte).

LIMITACIONES DECLARADAS
1. Modelo cinemático/de árbol, no QED dinámica; sin reciprocidad detallada certificada.
2. Fock truncado: el defecto de [a,a†] no aplica a fermiones (CAR es exacta en Λ(ℂ^m)), pero
   el truncamiento de modos sí limita el espacio de estados accesible.
3. σ de Breit–Wheeler en orden árbol; correcciones radiativas y haces reales no incluidas.
4. 392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) no lo alcanza.
5. Garwood es conservador (cobertura ≥ nominal); el MC verifica la cota, no la igualdad.

REFERENCIAS: CODATA 2022; Breit–Wheeler (1934); Klein–Nishina (1929); Landau (1948)–Yang (1950);
Wu–Shaknov (1950); Tsirelson (1980); Jordan–Wigner (1928); Garwood (1936); Aharonov–Albert–
Vaidman (1988); Spohn (1978); Uhlmann (1976); Fuchs–van de Graaf (1999); Melnikov (1963);
Gromov (1985); Ekeland–Hofer (1990); Kac (1947); Slater (1967); Kantorovich; Banach (1922); RFC 6962.
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
from scipy.special import gammaincinv

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

__all__ = ["BoolNovikovGammaPhotonColliderEngine", "GammaPhotonColliderSpectrumReport",
           "ColliderParameters", "ColliderError", "AuthorizationError",
           "CrowbarDriver", "SimulatedCrowbarDriver", "__version__"]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderEngine")

__version__: Final[str] = "14.0.0-ENGINE-GAMMA-PHOTON-COLLIDER-CAR-BREIT-WHEELER-GARWOOD-QND-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ME_C2_KEV: Final[float] = 510.99895069            # CODATA 2022
_TWO_ME_C2_KEV: Final[float] = 2.0 * _ME_C2_KEV    # 1021.99790138 keV
_R_E_CM: Final[float] = 2.8179403262e-13           # radio clásico del electrón [cm]
_SIGMA_T_CM2: Final[float] = 8.0 * math.pi / 3.0 * _R_E_CM ** 2   # ≈6.6524587e-25 cm²
_SIGMA_T_CODATA: Final[float] = 6.6524587321e-25
_KRULL_FLOOR: Final[float] = 0.10
_MINOR_CAP: Final[int] = 20_000
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_TSIRELSON: Final[float] = 2.0 * math.sqrt(2.0)
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")
_HMAC_ACTA: Final[bytes] = b"GAMMA-COLLIDER-ACTA-v14|"
_HMAC_AUTH: Final[bytes] = b"GAMMA-COLLIDER-HUMAN-AUTHORIZATION-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: canonicalización, Merkle, álgebra lineal estable, SNF exacta
# ══════════════════════════════════════════════════════════════════════════════════════
class ColliderError(Exception):
    """Error base del colisionador."""


class AuthorizationError(ColliderError):
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


def _psd_sqrt(a: NDArray) -> Tuple[NDArray, float]:
    v, u = la.eigh(0.5 * (a + a.conj().T))
    return (u * np.sqrt(np.maximum(v, 0.0))) @ u.conj().T, float(max(0.0, -np.min(v)))


def _trace_distance(a: NDArray, b: NDArray) -> float:
    d = a - b
    return 0.5 * float(np.sum(np.abs(la.eigvalsh(0.5 * (d + d.conj().T)))))


def _fidelity_svd(rho: NDArray, sig: NDArray) -> Tuple[float, float]:
    sr, cr = _psd_sqrt(rho)
    ss, cs = _psd_sqrt(sig)
    raw = float(np.sum(la.svdvals(sr @ ss))) ** 2
    clipped = min(1.0, max(0.0, raw))
    return clipped, abs(raw - clipped) + cr + cs


def _fidelity_sqrt(rho: NDArray, sig: NDArray) -> float:
    """Método ALTERNATIVO (independencia metodológica): (Tr√(√ρσ√ρ))² vía eigh."""
    sr, _ = _psd_sqrt(rho)
    core = sr @ sig @ sr
    v = la.eigvalsh(0.5 * (core + core.conj().T))
    return float(np.sum(np.sqrt(np.maximum(v, 0.0)))) ** 2


def _entropy(rho: NDArray, base: float = math.e) -> float:
    v = np.clip(la.eigvalsh(0.5 * (rho + rho.conj().T)), 0.0, 1.0)
    v = v[v > 1e-15]
    return float(-np.sum(v * np.log(v)) / math.log(base))


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


def garwood_interval(n_obs: int, alpha: float = 0.05) -> Tuple[float, float]:
    r"""
    Intervalo exacto de Garwood para la media de Poisson:
        λ_lo = ½χ²_{α/2}(2n),  λ_hi = ½χ²_{1−α/2}(2n+2),
    y con χ²_p(k) = 2·gammaincinv(k/2, p) ⇒ λ_lo = gammaincinv(n, α/2),
    λ_hi = gammaincinv(n+1, 1−α/2). Para n=0, λ_lo=0. Cobertura ≥ 1−α (conservador).
    """
    if n_obs < 0:
        raise ColliderError("n_obs ≥ 0 requerido.")
    lo = 0.0 if n_obs == 0 else float(gammaincinv(n_obs, alpha / 2.0))
    hi = float(gammaincinv(n_obs + 1, 1.0 - alpha / 2.0))
    return lo, hi


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ColliderParameters:
    """Parámetros DECLARADOS del colisionador; su digest se compromete en el acta."""

    fermion_modes: int = 2                  # (e⁻, e⁺); Fock dim = 2^m
    pair_coherence: float = 1.0             # peso de |11⟩ en (|00⟩+c|11⟩)/norm
    interaction_g: float = 1.0e-5           # QND dispersivo débil
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    beta_field: float = 2.0                 # β·ε para la referencia térmica
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    photon_crossing_deg: float = 180.0      # head-on por defecto
    photon1_energy_kev: float = _ME_C2_KEV
    photon2_energy_kev: float = _ME_C2_KEV
    luminosity_cm2_s: float = 1.0e32
    run_time_s: float = 1.0
    observed_events: int = 0
    poisson_alpha: float = 0.05
    poisson_mc_draws: int = 20000
    poisson_mc_true_mean: float = 5.0
    psd_floor: float = 1e-12
    ball_r: float = 1.0
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    kam_tau: float = 1.05
    kam_scan_order: int = 36
    cadence_points: int = 64
    cadence_rotation: float = _GOLDEN
    recurrence_mu: float = 0.10
    recurrence_samples: int = 4096
    require_human_authorization: bool = True
    lesser_measures_insufficient: bool = False
    logic_deadline_ns: float = 392.15
    gpio_write_ns: float = 25.0
    gate_driver_ns: float = 150.0
    scr_turn_on_ns: float = 1500.0          # BT151 t_gt típico (datasheet)
    hil_firmware_isr_ns: Optional[float] = None

    def __post_init__(self) -> None:
        if not 2 <= self.fermion_modes <= 6:
            raise ColliderError("fermion_modes ∈ [2,6] (Fock 2^m acotado).")
        if self.pointer_sigma <= 0.0 or self.interaction_g < 0.0:
            raise ColliderError("σ>0 y g≥0 requeridos.")
        if self.disturbance_budget < 0.0 or self.positron_kinetic_kev < 0.0:
            raise ColliderError("Presupuesto y cinética ≥ 0.")
        if self.photon1_energy_kev <= 0.0 or self.photon2_energy_kev <= 0.0:
            raise ColliderError("Energías de fotón positivas.")
        if self.luminosity_cm2_s <= 0.0 or self.run_time_s <= 0.0:
            raise ColliderError("Luminosidad y tiempo positivos.")
        if not (0.0 < self.poisson_alpha < 0.5) or self.observed_events < 0:
            raise ColliderError("α∈(0,0.5) y conteos ≥0.")
        if not (0.0 < self.recurrence_mu < 1.0 and 0.0 < self.cadence_rotation < 1.0):
            raise ColliderError("μ∈(0,1), α∈(0,1).")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0:
            raise ColliderError("Radios positivos.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)


@dataclass(frozen=True)
class FermionicFockReport:
    """§I.2 — Λ(ℂ^m) con signos de Jordan–Wigner: identidades EXACTAS."""

    modes: int
    dimension: int
    car_anticommutator_residual: float
    car_nilpotency_residual: float
    pauli_exclusion_residual: float
    parity_involution_residual: float
    parity_anticommutation_residual: float
    antisymmetry_residual: float
    pair_operator_identity_residual: float
    charge_conservation_residual: float
    pair_annihilation_amplitude: complex
    number_spectrum: Tuple[int, ...]


@dataclass(frozen=True)
class QNDColliderReport:
    """§I.3–§I.4 — QND dispersivo del número: retroacción MEDIDA."""

    qnd_dynamical_residual: float
    state_nondisturbance_residual: float
    kraus_completeness_residual: float
    trace_distance_disturbance: float
    disturbance_apriori_bound: float
    purity_before: float
    purity_after: float
    back_action_db: float
    qnd_repeatability_residual: float
    interaction_g: float
    pointer_sigma: float
    is_counterfactual: bool
    is_qnd_passive: bool
    weak_value: complex
    weak_value_anomalous: bool
    postselection_probability: float
    weak_sum_rule_residual: float
    pointer_weak_limit_residual: float
    pointer_convergence_order: float


@dataclass(frozen=True)
class AnnihilationKinematicsReport:
    """§II.2 — Cinemática relativista exacta (keV, c=1) y reglas de selección."""

    electron_rest_energy_kev: float
    threshold_kev: float
    positron_kinetic_kev: float
    mandelstam_s_kev2: float
    sqrt_s_kev: float
    threshold_satisfied: bool
    total_energy_kev: float
    photon1_energy_kev: float
    photon2_energy_kev: float
    e_max_kev: float
    e_min_kev: float
    sum_identity_residual: float
    product_identity_residual: float
    energy_conservation_residual: float
    momentum_conservation_residual: float
    single_photon_forbidden: bool
    para_two_gamma_allowed: bool
    ortho_two_gamma_allowed: bool
    ortho_three_gamma_allowed: bool
    landau_yang_forbids_j1: bool


@dataclass(frozen=True)
class BreitWheelerReport:
    """§II.3 — γγ→e⁺e⁻ (Breit–Wheeler 1934), orden árbol."""

    photon1_energy_kev: float
    photon2_energy_kev: float
    crossing_angle_deg: float
    s_gg_kev2: float
    x_ratio: float
    beta_velocity: float
    threshold_satisfied: bool
    sigma_re2: float
    sigma_cm2: float
    sigma_at_threshold_re2: float
    peak_sigma_over_sigma_t: float
    thomson_sigma_cm2: float
    thomson_codata_residual: float
    bound_sigma_le_thomson: bool
    rest_head_on_saturation_residual: float


@dataclass(frozen=True)
class PolarizationReport:
    """§II.4 — Singlete derivado de (ε₁×ε₂)·k̂; CHSH y Tsirelson."""

    amplitudes: Tuple[complex, ...]
    purity: float
    reduced_purity: float
    entanglement_entropy_bits: float
    concurrence: float
    parallel_probability: float
    perpendicular_probability: float
    correlation_formula_residual: float
    chsh_maximum: float
    tsirelson_residual: float
    bell_violation: bool


@dataclass(frozen=True)
class LuminosityPoissonReport:
    """§II.5 — R=Lσ, Poisson y intervalo exacto de Garwood con cobertura MC."""

    luminosity_cm2_s: float
    sigma_cm2: float
    run_time_s: float
    rate_hz: float
    expected_events: float
    probability_zero_events: float
    observed_events: int
    garwood_low: float
    garwood_high: float
    alpha: float
    mc_coverage: float
    mc_coverage_ok: bool
    interval_monotone: bool


@dataclass(frozen=True)
class BuresSpohnReport:
    """§II.6 — Bures–Uhlmann con referencia declarada + Spohn del dephasing."""

    reference_label: str
    fidelity: float
    fidelity_alternate: float
    method_agreement_residual: float
    bures_distance: float
    trace_distance: float
    fvg_lower_ok: bool
    fvg_upper_ok: bool
    cptp_monotonicity_residual: float
    spohn_times: Tuple[float, ...]
    spohn_relative_entropy: Tuple[float, ...]
    spohn_monotone: bool
    spohn_sigma_min: float
    spohn_sigma_nonnegative: bool
    coherence_identity_residual: float
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
class ColliderRSI3Report:
    """§III.2 — RSI³ certificado."""

    calibrated_gain: float
    target_disturbance: float
    achieved_proxy: float
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
    """§III.3 — KAM diophantino, Lindstedt y cadencia (tres huecos + Kac)."""

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
    structural_failure: bool
    indictment_digest: str
    epoch: int
    token_verified: bool
    token_self_issued: bool
    ne_bis_in_idem_ok: bool
    anti_replay_ok: bool
    actuation_requested: bool
    actuation_authorized: bool


@dataclass(frozen=True)
class CrowbarActuationReport:
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
    note: str


@dataclass(frozen=True)
class ColliderPhaseIHandoff:
    """§I.5 [TERMINAL FASE I] — entrada de §II.1."""

    fock: FermionicFockReport
    qnd: QNDColliderReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    commitment_digest: str
    rho_digest: str
    kernel_digest: str
    exponents_digest: str
    rho_before: NDArray = field(repr=False)
    rho_after: NDArray = field(repr=False)
    number_operator: NDArray = field(repr=False)
    field_hamiltonian: NDArray = field(repr=False)


@dataclass(frozen=True)
class ColliderPhaseIIHandoff:
    """§II.7 [TERMINAL FASE II] — anida Fase I; entrada de §III.1."""

    phase_i: ColliderPhaseIHandoff
    kinematics: AnnihilationKinematicsReport
    breit_wheeler: BreitWheelerReport
    polarization: PolarizationReport
    luminosity: LuminosityPoissonReport
    bures_spohn: BuresSpohnReport
    algebra: AlgebraGeometryReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue


@dataclass(frozen=True)
class GammaPhotonColliderSpectrumReport:
    """§III.7 [TERMINAL] — acta anidada I ⊂ II ⊂ III con Merkle y pasaporte."""

    engine_version: str
    phase_ii: ColliderPhaseIIHandoff
    rsi3: ColliderRSI3Report
    kam_cadence: KAMCadenceReport
    sentence: SentenceAuthorizationReport
    crowbar: CrowbarActuationReport
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
            "annihilation_report": p2.kinematics, "qnd_report": p1.qnd,
            "rademacher_walsh_norm": None,
            "smith_torsion_free": all(d == 1 for d in p2.algebra.smith_invariants),
            "betti_number_zero": p2.algebra.betti_0,
            "krull_valuation": p2.algebra.novikov_valuation,
            "novikov_norm": p2.algebra.novikov_norm,
            "cauchy_riemann_residual": p2.algebra.floer_energy_area_residual,
            "is_gromov_rigid": not p2.algebra.gromov_obstruction,
            "melnikov_chaos_detected": p2.algebra.melnikov_chaos,
            "banach_contraction_lip": self.rsi3.lip_t3_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "gamma_photons_emitted": 2 if p2.kinematics.threshold_satisfied else 0,
            "total_energy_released_kev": p2.kinematics.sqrt_s_kev,
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
#  FASE I · SUSTRATO FERMIONICO DE FOCK Y QND DISPERSIVO
# ══════════════════════════════════════════════════════════════════════════════════════
class ColliderPhaseIFermionicSubstrate:
    r"""
    FASE I — Construye Λ(ℂ^m) con signos de Jordan–Wigner y mide la retroacción real del
    canal dispersivo sobre un estado CON coherencias de par (v13 usaba I/d, que conmuta
    con todo y hacía el certificado QND vacuo).
    """

    PHASE_LABEL: Final[str] = "I·FOCK-FERMIONICO-QND"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_122) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_COLLIDER_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning("Clave de colisionador efímera: actas no verificables entre procesos.")
        if len(secret_key) < 32:
            raise ColliderError("La clave del colisionador requiere ≥ 256 bits.")
        self._key = bytes(secret_key)
        self._seed = int(seed)
        self._principal = principal_engine or PrincipalEngine(secret_key=os.urandom(32), seed=seed)
        self._epoch = 0
        self._judged: set = set()
        self._last_epoch = -1
        self._pending: Dict[str, Any] = {}

    @property
    def key_bytes(self) -> bytes:
        return self._key

    # ─────────── §I.2 Fock fermiónico: CAR, Pauli, paridad, carga ───────────
    @staticmethod
    def fermionic_fock(params: ColliderParameters) -> Tuple[List[NDArray], NDArray, NDArray,
                                                           NDArray, FermionicFockReport]:
        r"""
        Λ(ℂ^m), base de ocupaciones |b_{m−1}…b₀⟩. Jordan–Wigner:
            c_j|n⟩ = (−1)^{Σ_{k<j} b_k} |n ⊕ 2^j⟩  si b_j=1,  0 si b_j=0.
        Identidades EXACTAS verificadas:
            {c_i,c_j†}=δ_ij I,  {c_i,c_j}=0,  (c_j†)²=0 (Pauli),
            P=(−1)^N̂: P²=I y P c_j P = −c_j,  c₀c₁ = −c₁c₀ (antisimetría),
            A_par = c₊c₋:  A_par†A_par = n₊n₋ (proyector de doble ocupación),
            [Q, A_par] = 0 con Q = n₊ − n₋ (CONSERVACIÓN DE CARGA en la aniquilación).
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
        charge = nums[1] - nums[0]                       # Q = n₊ − n₋ (unidades de e)
        a_pair = ops[1] @ ops[0]

        car_res = 0.0
        nil_res = 0.0
        pauli_res = 0.0
        for i in range(m):
            for j in range(m):
                anti = ops[i] @ ops[j].conj().T + ops[j].conj().T @ ops[i]
                target = eye if i == j else np.zeros_like(eye)
                car_res = max(car_res, float(la.norm(anti - target, ord="fro")))
                nil_res = max(nil_res, float(la.norm(ops[i] @ ops[j] + ops[j] @ ops[i], ord="fro")))
            pauli_res = max(pauli_res, float(la.norm(ops[i].conj().T @ ops[i].conj().T, ord="fro")))
        par_inv = float(la.norm(parity @ parity - eye, ord="fro"))
        par_anti = max(float(la.norm(parity @ c @ parity + c, ord="fro")) for c in ops)
        antisym = float(la.norm(ops[0] @ ops[1] + ops[1] @ ops[0], ord="fro"))
        pair_id = float(la.norm(a_pair.conj().T @ a_pair - nums[1] @ nums[0], ord="fro"))
        charge_res = float(la.norm(charge @ a_pair - a_pair @ charge, ord="fro"))
        idx_pair = (1 << 0) | (1 << 1)
        v_pair = np.zeros(dim, dtype=np.complex128); v_pair[idx_pair] = 1.0
        amp = complex((a_pair @ v_pair)[0])
        spectrum = tuple(int(round(float(np.real(n_tot[n, n])))) for n in range(dim))
        return ops, n_tot, charge, a_pair, FermionicFockReport(
            modes=m, dimension=dim, car_anticommutator_residual=car_res,
            car_nilpotency_residual=nil_res, pauli_exclusion_residual=pauli_res,
            parity_involution_residual=par_inv, parity_anticommutation_residual=par_anti,
            antisymmetry_residual=antisym, pair_operator_identity_residual=pair_id,
            charge_conservation_residual=charge_res, pair_annihilation_amplitude=amp,
            number_spectrum=spectrum,
        )

    # ─────────── §I.3 Canal dispersivo gaussiano EXACTO ───────────
    @staticmethod
    def dispersive_channel(rho: NDArray, n_vals: NDArray, g: float, sigma: float
                           ) -> Tuple[NDArray, float, float]:
        r"""
        H_int = ħχ N̂ ⊗ P̂ conserva el número ⇒ QND. Kraus gaussianos M_y ∝ exp(−(y−gN̂)²/4σ²)
        dan la acción cerrada ρ'_mn = ρ_mn exp(−g²(N_m−N_n)²/(8σ²)), canal UNITAL ⇒
        Tr ρ'² ≤ Tr ρ² (Kadison–Schwarz) ⇒ dB ≤ 0. Cota: ‖ρ'−ρ‖₁/2 ≤ ½√d(g²Δ²_max/8σ²)‖ρ‖_F.
        """
        delta = n_vals[:, None] - n_vals[None, :]
        after = rho * np.exp(-(g ** 2) * delta ** 2 / (8.0 * sigma ** 2))
        y = np.linspace(g * float(n_vals.min()) - 10 * sigma, g * float(n_vals.max()) + 10 * sigma, 4001)
        m = (2 * math.pi * sigma ** 2) ** (-0.25) * np.exp(-(y[:, None] - g * n_vals[None, :]) ** 2 / (4 * sigma ** 2))
        integral = np.array([_trapz(m[:, j] ** 2, y) for j in range(n_vals.size)])
        completeness = float(np.max(np.abs(integral - 1.0)))
        d = rho.shape[0]
        bound = 0.5 * math.sqrt(d) * (g ** 2 * float(np.max(delta ** 2)) / (8 * sigma ** 2)) * float(la.norm(rho, ord="fro"))
        return after, completeness, bound

    # ─────────── §I.4 AAV: valor débil, regla de suma, límite del puntero ───────────
    @staticmethod
    def weak_value(a_op: NDArray, phi_i: NDArray, phi_f: NDArray) -> Tuple[complex, float]:
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
        f_v = np.asarray(phi_f, dtype=np.complex128).reshape(-1)
        i_v /= (la.norm(i_v) + _EPS); f_v /= (la.norm(f_v) + _EPS)
        ov = complex(np.vdot(f_v, i_v))
        if abs(ov) < 1e-12:
            raise ColliderError("Postselección casi ortogonal: A_w no certificable.")
        return complex(np.vdot(f_v, a_op @ i_v)) / ov, float(abs(ov) ** 2)

    def sum_rule_residual(self, a_op: NDArray, phi_i: NDArray, basis: NDArray) -> float:
        """Σ_f |⟨φ_f|φ_i⟩|² A_w(f) = ⟨φ_i|A|φ_i⟩ (resolución completa de la identidad)."""
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
        i_v /= (la.norm(i_v) + _EPS)
        tot = 0.0j
        for j in range(basis.shape[1]):
            f_v = basis[:, j]
            ov = complex(np.vdot(f_v, i_v))
            if abs(ov) > 1e-12:
                tot += abs(ov) ** 2 * (complex(np.vdot(f_v, a_op @ i_v)) / ov)
        exp = complex(np.vdot(i_v, a_op @ i_v))
        return float(abs(tot - exp)) / (abs(exp) + 1.0)

    def pointer_weak_limit(self, n_vals: NDArray, phi_i: NDArray, phi_f: NDArray,
                           sigma: float, gains: Sequence[float]) -> Tuple[float, float]:
        """ȳ/g → Re A_w; por paridad p_{−g}(−y)=p_g(y) el error es O(g²): orden MEDIDO."""
        a_op = np.diag(n_vals.astype(np.complex128))
        target = self.weak_value(a_op, phi_i, phi_f)[0].real
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1); i_v /= (la.norm(i_v) + _EPS)
        f_v = np.asarray(phi_f, dtype=np.complex128).reshape(-1); f_v /= (la.norm(f_v) + _EPS)
        errs: List[float] = []
        for g in gains:
            y = np.linspace(g * float(n_vals.min()) - 12 * sigma, g * float(n_vals.max()) + 12 * sigma, 8001)
            m = np.exp(-(y[:, None] - g * n_vals[None, :]) ** 2 / (4 * sigma ** 2))
            amp = np.sum(np.conj(f_v)[None, :] * m * i_v[None, :], axis=1)
            p = np.abs(amp) ** 2
            mean = _trapz(y * p, y) / (_trapz(p, y) + _EPS)
            errs.append(abs(mean / g - target))
        order = math.log2(errs[0] / max(errs[1], 1e-300)) if len(errs) >= 2 and errs[1] > 0 else 0.0
        return float(errs[-1]), float(order)

    # ─────────── §I.5 [TERMINAL FASE I] → Fase II §II.1 ───────────
    def build_collider_phase_i_handoff(
        self, apu_cost_matrix: NDArray, exponents: NDArray, params: ColliderParameters, *,
        rho_mac_input: Optional[NDArray] = None,
    ) -> ColliderPhaseIHandoff:
        """MÉTODO TERMINAL DE LA FASE I: CAR exactas, QND medido y compromiso de entradas."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise ColliderError("K debe ser cuadrada.")
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if exps.size == 0:
            raise ColliderError("exponents vacío: v(A) no definida.")
        _, n_tot, _, _, fock = self.fermionic_fock(params)
        dim = fock.dimension
        if rho_mac_input is None:
            c = float(params.pair_coherence)
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = 1.0
            psi[(1 << 0) | (1 << 1)] = c            # (|00⟩ + c|11⟩)/norm: coherencia de PAR
            psi /= (la.norm(psi) + _EPS)
            rho = np.outer(psi, psi.conj())
        else:
            r = np.asarray(rho_mac_input, dtype=np.complex128)
            if r.shape != (dim, dim):
                raise ColliderError(f"ρ debe ser {dim}×{dim} (Fock fermiónico m={fock.modes}).")
            tr = complex(np.trace(r))
            if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0:
                raise ColliderError("ρ: traza compleja o no positiva.")
            rho = 0.5 * (r + r.conj().T) / tr.real
        n_vals = np.real(np.diag(n_tot))
        h_field = _ME_C2_KEV * n_tot
        g, sigma = float(params.interaction_g), float(params.pointer_sigma)
        after, completeness, bound = self.dispersive_channel(rho, n_vals, g, sigma)

        n_fro = float(la.norm(n_tot, ord="fro")) + _EPS
        h_fro = float(la.norm(h_field, ord="fro")) + _EPS
        r_fro = float(la.norm(rho, ord="fro")) + _EPS
        dyn = float(la.norm(n_tot @ h_field - h_field @ n_tot, ord="fro")) / (n_fro * h_fro)
        nondist = float(la.norm(rho @ n_tot - n_tot @ rho, ord="fro")) / (r_fro * n_fro)
        pb = float(np.real(np.trace(rho @ rho)))
        pa = float(np.real(np.trace(after @ after)))
        db = 10.0 * math.log10(max(pa, 1e-300) / max(pb, 1e-300))
        dist = _trace_distance(rho, after)
        repeat = float(np.max(np.abs(np.real(np.diag(after)) - np.real(np.diag(rho)))))
        passive = bool(dist <= params.disturbance_budget and db >= -params.backaction_db_budget)

        rng = np.random.default_rng(self._seed)
        phi_i = rho_diag = np.sqrt(np.maximum(np.real(np.diag(rho)), 0.0)).astype(np.complex128)
        phi_i = np.zeros(dim, dtype=np.complex128); phi_i[0] = 1.0
        phi_i[(1 << 0) | (1 << 1)] = float(params.pair_coherence)
        phi_i /= (la.norm(phi_i) + _EPS)
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        phi_f = v / la.norm(v)
        a_w, p_f = self.weak_value(n_tot, phi_i, phi_f)
        spec = float(np.max(np.abs(n_vals)))
        anom = bool(a_w.real > spec + 1e-9 or a_w.real < -1e-9 or abs(a_w.imag) > 1e-9)
        haar = la.qr(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)))[0]
        sum_res = max(self.sum_rule_residual(n_tot, phi_i, np.eye(dim, dtype=np.complex128)),
                      self.sum_rule_residual(n_tot, phi_i, haar))
        ptr_res, ptr_order = self.pointer_weak_limit(
            n_vals, phi_i, phi_f, sigma, (0.1 * sigma, 0.05 * sigma, 0.025 * sigma))

        qnd = QNDColliderReport(
            qnd_dynamical_residual=dyn, state_nondisturbance_residual=nondist,
            kraus_completeness_residual=completeness, trace_distance_disturbance=dist,
            disturbance_apriori_bound=bound, purity_before=pb, purity_after=pa, back_action_db=db,
            qnd_repeatability_residual=repeat, interaction_g=g, pointer_sigma=sigma,
            is_counterfactual=bool(g == 0.0), is_qnd_passive=passive, weak_value=a_w,
            weak_value_anomalous=anom, postselection_probability=p_f, weak_sum_rule_residual=sum_res,
            pointer_weak_limit_residual=ptr_res, pointer_convergence_order=ptr_order,
        )
        evidences: Tuple[Any, ...] = (
            Evidence("C1.car_anticommutador", fock.car_anticommutator_residual, 1e-12, hard=True),
            Evidence("C1.car_nilpotencia", fock.car_nilpotency_residual, 1e-12, hard=True),
            Evidence("C1.pauli_exclusion", fock.pauli_exclusion_residual, 1e-12, hard=True),
            Evidence("C1.paridad_involucion", fock.parity_involution_residual, 1e-12, hard=True),
            Evidence("C1.paridad_anticommutacion", fock.parity_anticommutation_residual, 1e-12, hard=True),
            Evidence("C1.antisimetria_fermionica", fock.antisymmetry_residual, 1e-12, hard=True),
            Evidence("C1.operador_de_par", fock.pair_operator_identity_residual, 1e-12, hard=True),
            Evidence("C1.conservacion_de_carga", fock.charge_conservation_residual, 1e-12, hard=True),
            Evidence("C1.kraus_completitud", completeness, 1e-9, hard=True),
            Evidence("C1.qnd_dinamico_[N,H]", dyn, 1e-12, hard=True),
            Evidence("C1.qnd_repetibilidad", repeat, 1e-14, hard=True),
            Evidence.boolean("C1.pasividad_presupuestada", passive, hard=True),
            Evidence("C1.disturbancia_vs_cota", dist - bound, 1e-16, hard=True),
            Evidence.boolean("C1.pureza_no_creciente", pa <= pb + 1e-12, hard=True),
            Evidence("C1.aav_regla_de_suma", sum_res, 1e-10, hard=True),
            Evidence("C1.aav_limite_debil", ptr_res, 1e-3),
            Evidence("C1.aav_orden", abs(ptr_order - 2.0), 0.4),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        commitment = _sha256(_canonical({
            "rho": _array_digest(rho), "K": _array_digest(k_mat), "exps": _array_digest(exps),
            "params": params.digest(), "version": __version__,
        }))
        handoff = ColliderPhaseIHandoff(
            fock=fock, qnd=qnd, evidences=evidences, truth_value=truth,
            commitment_digest=commitment, rho_digest=_array_digest(rho),
            kernel_digest=_array_digest(k_mat), exponents_digest=_array_digest(exps),
            rho_before=rho, rho_after=after, number_operator=n_tot, field_hamiltonian=h_field,
        )
        logger.info("Fase I collider Ω₃=%s · CAR=%.2e · [ρ,N]=%.3e · D_tr=%.3e · dB=%.3e",
                    truth.name, fock.car_anticommutator_residual, nondist, dist, db)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · CINEMATICA, BREIT–WHEELER, POLARIZACION Y ESTADISTICA  (anida Fase I)
# ══════════════════════════════════════════════════════════════════════════════════════
class ColliderPhaseIICrossSections(ColliderPhaseIFermionicSubstrate):
    r"""FASE II — Cinemática exacta, sección eficaz de Breit–Wheeler y estadística de conteos."""

    PHASE_LABEL: Final[str] = "II·CINEMATICA-SECCIONES-EFICACES"

    def accept_collider_phase_i(self, h: ColliderPhaseIHandoff) -> ColliderPhaseIHandoff:
        if not isinstance(h, ColliderPhaseIHandoff):
            raise ColliderError("Fase II requiere el ColliderPhaseIHandoff de §I.5.")
        return h

    # ─────────── §II.2 Cinemática exacta y reglas de selección ───────────
    @staticmethod
    def annihilation_kinematics(params: ColliderParameters) -> AnnihilationKinematicsReport:
        r"""
        e⁻ en reposo, e⁺ con cinética K: E₊=m+K, p₊=√(E₊²−m²), s=2m²+2mE₊.
        Dos fotones: (P−k₁)²=0 ⇒ E₁(θ)=s/(2(E_tot−p_tot cosθ)).
        Identidades exactas: E_max+E_min=E_tot, E_max·E_min=s/4.
        1γ prohibido (k²=0 pero s>0). Selección: C=(−1)^{L+S} vs (−1)^n; Landau–Yang J=1↛2γ.
        """
        m = _ME_C2_KEV
        k_en = float(params.positron_kinetic_kev)
        e_plus = m + k_en
        p_plus = math.sqrt(max(0.0, e_plus ** 2 - m ** 2))
        s = 2.0 * m ** 2 + 2.0 * m * e_plus
        e_tot, p_tot = m + e_plus, p_plus
        th = math.radians(float(params.emission_angle_deg))
        e1 = s / (2.0 * (e_tot - p_tot * math.cos(th)))
        e2 = e_tot - e1
        cos2 = (p_tot - e1 * math.cos(th)) / (e2 + _EPS)
        e_max = s / (2.0 * (e_tot - p_tot))
        e_min = s / (2.0 * (e_tot + p_tot))
        mom = math.hypot(e1 * math.cos(th) + e2 * cos2 - p_tot,
                         e1 * math.sin(th) - e2 * math.sqrt(max(0.0, 1.0 - cos2 ** 2)))
        para_c, ortho_c = 1, -1
        return AnnihilationKinematicsReport(
            electron_rest_energy_kev=m, threshold_kev=_TWO_ME_C2_KEV, positron_kinetic_kev=k_en,
            mandelstam_s_kev2=s, sqrt_s_kev=math.sqrt(s),
            threshold_satisfied=bool(math.sqrt(s) >= _TWO_ME_C2_KEV - 1e-9),
            total_energy_kev=e_tot, photon1_energy_kev=e1, photon2_energy_kev=e2,
            e_max_kev=e_max, e_min_kev=e_min,
            sum_identity_residual=abs((e_max + e_min) - e_tot) / e_tot,
            product_identity_residual=abs(e_max * e_min - s / 4.0) / (s / 4.0),
            energy_conservation_residual=abs((e1 + e2) - e_tot) / e_tot,
            momentum_conservation_residual=mom / (p_tot + m),
            single_photon_forbidden=bool(s > 0.0),
            para_two_gamma_allowed=bool((-1) ** 2 == para_c),
            ortho_two_gamma_allowed=False, ortho_three_gamma_allowed=bool((-1) ** 3 == ortho_c),
            landau_yang_forbids_j1=True,
        )

    # ─────────── §II.3 Breit–Wheeler γγ→e⁺e⁻ ───────────
    @staticmethod
    def breit_wheeler(params: ColliderParameters) -> BreitWheelerReport:
        r"""
        s_γγ = 2E₁E₂(1−cosθ);  x = s_γγ/(4m²c⁴);  β = √(1−1/x)  (x≥1).
        σ_γγ = (π r_e²/2)(1−β²)[(3−β⁴)ln((1+β)/(1−β)) − 2β(2−β²)].
        Verificado: σ(x=1)=0 (umbral, β=0); pico ≈ 0.26 σ_T; σ ≤ σ_T;
        σ_T = (8π/3) r_e² coincide con CODATA; y los fotones de aniquilación en reposo
        head-on SATURAN el umbral exactamente (x=1 ⇒ espacio de fase nulo).
        """
        e1, e2 = float(params.photon1_energy_kev), float(params.photon2_energy_kev)
        th = math.radians(float(params.photon_crossing_deg))
        s_gg = 2.0 * e1 * e2 * (1.0 - math.cos(th))
        x = s_gg / (4.0 * _ME_C2_KEV ** 2)

        def sigma_of_beta(b: float) -> float:
            if b <= 0.0:
                return 0.0
            if b >= 1.0:
                return 0.0
            return (math.pi / 2.0) * (1.0 - b ** 2) * (
                (3.0 - b ** 4) * math.log((1.0 + b) / (1.0 - b)) - 2.0 * b * (2.0 - b ** 2))

        beta = math.sqrt(max(0.0, 1.0 - 1.0 / x)) if x >= 1.0 else 0.0
        sig = sigma_of_beta(beta) if x >= 1.0 else 0.0
        grid = np.linspace(1e-4, 1.0 - 1e-6, 4000)
        peak = float(np.max([sigma_of_beta(float(b)) for b in grid]))
        sigma_t_re2 = 8.0 * math.pi / 3.0
        rest_x = (2.0 * _ME_C2_KEV * _ME_C2_KEV * 2.0) / (4.0 * _ME_C2_KEV ** 2)   # =1 exacto
        return BreitWheelerReport(
            photon1_energy_kev=e1, photon2_energy_kev=e2, crossing_angle_deg=float(params.photon_crossing_deg),
            s_gg_kev2=s_gg, x_ratio=x, beta_velocity=beta, threshold_satisfied=bool(x >= 1.0 - 1e-12),
            sigma_re2=sig, sigma_cm2=sig * _R_E_CM ** 2, sigma_at_threshold_re2=sigma_of_beta(0.0),
            peak_sigma_over_sigma_t=peak / sigma_t_re2, thomson_sigma_cm2=sigma_t_re2 * _R_E_CM ** 2,
            thomson_codata_residual=abs(sigma_t_re2 * _R_E_CM ** 2 - _SIGMA_T_CODATA) / _SIGMA_T_CODATA,
            bound_sigma_le_thomson=bool(sig <= sigma_t_re2 + 1e-12),
            rest_head_on_saturation_residual=abs(rest_x - 1.0),
        )

    # ─────────── §II.4 Polarización derivada ───────────
    @staticmethod
    def polarization() -> PolarizationReport:
        r"""
        M(ε₁,ε₂) ∝ (ε₁×ε₂)·k̂ con k̂=ẑ y ejes de LAB ⇒ M=[[0,1],[−1,0]] ⇒
        |Ψ⟩=(|xy⟩−|yx⟩)/√2 (singlete). Se DERIVA. E(θ₁,θ₂)=−cos2(θ₁−θ₂);
        CHSH máx = 2√2 (Tsirelson) y >2 (violación de Bell); correlación perpendicular
        (Wu–Shaknov). El cierre del loophole de detección NO se afirma aquí (es del agente).
        """
        eps = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        k_hat = np.array([0.0, 0.0, 1.0])
        amp = np.zeros((2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                amp[i, j] = complex(np.dot(np.cross(eps[i], eps[j]), k_hat))
        psi = (amp / la.norm(amp)).reshape(4)
        rho = np.outer(psi, psi.conj())
        t4 = rho.reshape(2, 2, 2, 2)
        red = np.array([[np.sum(t4[i, :, j, :]) for j in range(2)] for i in range(2)])
        conc = 2.0 * abs(amp[0, 0] * amp[1, 1] - amp[0, 1] * amp[1, 0]) / (la.norm(amp) ** 2)

        def proj(t: float) -> NDArray:
            v = np.array([math.cos(t), math.sin(t)], dtype=np.complex128)
            return np.outer(v, v.conj())

        rng = np.random.default_rng(13)
        fres = 0.0
        for t1, t2 in rng.uniform(0, math.pi, (8, 2)):
            a = 2 * proj(t1) - np.eye(2)
            b = 2 * proj(t2) - np.eye(2)
            corr = float(np.real(np.trace(rho @ np.kron(a, b))))
            fres = max(fres, abs(corr + math.cos(2 * (t1 - t2))))
        g = np.linspace(0.0, math.pi, 16, endpoint=False)
        A, Ap, B, Bp = np.meshgrid(g, g, g, g, indexing="ij")
        s_mat = np.abs(-np.cos(2 * (A - B)) + np.cos(2 * (A - Bp))
                       - np.cos(2 * (Ap - B)) - np.cos(2 * (Ap - Bp)))
        best = float(np.max(s_mat))
        return PolarizationReport(
            amplitudes=tuple(complex(x) for x in psi), purity=float(np.real(np.trace(rho @ rho))),
            reduced_purity=float(np.real(np.trace(red @ red))),
            entanglement_entropy_bits=_entropy(red, base=2.0), concurrence=float(conc),
            parallel_probability=float(abs(psi[0]) ** 2 + abs(psi[3]) ** 2),
            perpendicular_probability=float(abs(psi[1]) ** 2 + abs(psi[2]) ** 2),
            correlation_formula_residual=fres, chsh_maximum=best,
            tsirelson_residual=abs(best - _TSIRELSON), bell_violation=bool(best > 2.0 + 1e-6),
        )

    # ─────────── §II.5 Luminosidad y estadística de Poisson ───────────
    @staticmethod
    def luminosity_poisson(bw: BreitWheelerReport, params: ColliderParameters,
                           seed: int) -> LuminosityPoissonReport:
        r"""
        R = L·σ [s⁻¹], N̄ = R·T. Conteos ~ Poisson(N̄); P(0)=e^{−N̄}.
        Intervalo EXACTO de Garwood (cobertura ≥ 1−α, conservador), verificado por Monte
        Carlo con semilla: la fracción de intervalos que cubren la media verdadera debe ser
        ≥ 1−α (menos holgura estadística). Monotonía del intervalo en n también se verifica.
        """
        rate = float(params.luminosity_cm2_s) * bw.sigma_cm2
        n_bar = rate * float(params.run_time_s)
        lo, hi = garwood_interval(int(params.observed_events), float(params.poisson_alpha))
        rng = np.random.default_rng(seed + 77)
        draws = rng.poisson(float(params.poisson_mc_true_mean), int(params.poisson_mc_draws))
        covered = 0
        for n in draws:
            l_n, h_n = garwood_interval(int(n), float(params.poisson_alpha))
            if l_n <= params.poisson_mc_true_mean <= h_n:
                covered += 1
        cov = covered / float(draws.size)
        nominal = 1.0 - float(params.poisson_alpha)
        slack = 3.0 * math.sqrt(nominal * (1.0 - nominal) / draws.size)
        prev_hi = -1.0
        mono = True
        for n in range(0, 12):
            l_n, h_n = garwood_interval(n, float(params.poisson_alpha))
            mono = mono and (h_n >= prev_hi) and (l_n <= h_n)
            prev_hi = h_n
        return LuminosityPoissonReport(
            luminosity_cm2_s=float(params.luminosity_cm2_s), sigma_cm2=bw.sigma_cm2,
            run_time_s=float(params.run_time_s), rate_hz=rate, expected_events=n_bar,
            probability_zero_events=math.exp(-n_bar) if n_bar < 700 else 0.0,
            observed_events=int(params.observed_events), garwood_low=lo, garwood_high=hi,
            alpha=float(params.poisson_alpha), mc_coverage=cov,
            mc_coverage_ok=bool(cov >= nominal - slack), interval_monotone=bool(mono),
        )

    # ─────────── §II.6 Bures–Uhlmann + Spohn ───────────
    def bures_spohn_certificate(self, h: ColliderPhaseIHandoff,
                                params: ColliderParameters) -> BuresSpohnReport:
        r"""
        Referencias DECLARADAS: ρ pre-colisión (pasividad) y ρ térmica del campo.
        F por norma traza (SVD) + contraste con (Tr√(√ρσ√ρ))²; Fuchs–van de Graaf;
        monotonía CPTP del canal dispersivo. Spohn: ρ_t = ρ∘exp(−tΓ), Γ_mn=(N_m−N_n)²/8,
        punto fijo diag ρ ⇒ S(ρ_t‖diag ρ) no creciente ⇒ σ=−dS/dt ≥ 0; además
        S(ρ‖diag ρ)=S(diag ρ)−S(ρ)=C_rel(ρ).
        Diagnóstico: la «fidelidad» de v13 era Tr(√ρ√ρ)=Tr ρ ≡ 1.
        """
        rho, after = h.rho_before, h.rho_after
        n_vals = np.real(np.diag(h.number_operator))
        fid, _ = _fidelity_svd(rho, after)
        alt = _fidelity_sqrt(rho, after)
        dist = _trace_distance(rho, after)
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid)))
        g, sigma = h.qnd.interaction_g, h.qnd.pointer_sigma
        chan = lambda r: self.dispersive_channel(r, n_vals, g, sigma)[0]      # noqa: E731
        en = n_vals * _ME_C2_KEV
        lp = -params.beta_field * (en - en[0])
        lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        f_th, _ = _fidelity_svd(chan(rho), chan(rho_th))
        mono_res = max(0.0, math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f_th))) -
                       math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(_fidelity_svd(rho, rho_th)[0]))))
        gamma = (n_vals[:, None] - n_vals[None, :]) ** 2 / 8.0
        diag_rho = np.diag(np.diag(rho))
        log_d = np.log(np.maximum(np.real(np.diag(rho)), 1e-300))

        def rel_ent(x: NDArray) -> float:
            v = np.clip(la.eigvalsh(0.5 * (x + x.conj().T)), 0.0, 1.0)
            v = v[v > 1e-15]
            return float(np.sum(v * np.log(v)) - np.sum(log_d * np.real(np.diag(x))))

        times = (0.0, 0.05, 0.2, 0.6, 1.5)
        trace = [rel_ent(rho * np.exp(-t * gamma)) for t in times]
        sigmas = [-(trace[i + 1] - trace[i]) / (times[i + 1] - times[i]) for i in range(len(times) - 1)]
        sr, _ = _psd_sqrt(rho)
        legacy = float(np.real(np.trace(sr @ sr)))     # ≡ Tr ρ = 1: tautología de v13
        return BuresSpohnReport(
            reference_label="ρ_pre_colisión (pasividad) y ρ_térmica (βε)", fidelity=fid,
            fidelity_alternate=alt, method_agreement_residual=abs(fid - alt), bures_distance=bures,
            trace_distance=dist, fvg_lower_ok=bool(1.0 - math.sqrt(fid) <= dist + 1e-12),
            fvg_upper_ok=bool(dist <= math.sqrt(max(0.0, 1.0 - fid)) + 1e-12),
            cptp_monotonicity_residual=mono_res, spohn_times=times,
            spohn_relative_entropy=tuple(trace),
            spohn_monotone=bool(all(trace[i + 1] <= trace[i] + 1e-10 for i in range(len(trace) - 1))),
            spohn_sigma_min=float(min(sigmas)), spohn_sigma_nonnegative=bool(min(sigmas) >= -1e-10),
            coherence_identity_residual=abs(trace[0] - (_entropy(diag_rho) - _entropy(rho))),
            legacy_self_fidelity=legacy,
        )

    # ─────────── §II.7 Álgebra y geometría (nivel C) ───────────
    def algebra_geometry(self, k_mat: NDArray, exponents: NDArray,
                         params: ColliderParameters) -> AlgebraGeometryReport:
        """SNF por menores (independiente), Novikov con axiomas, Floer con cota, Melnikov."""
        _, k_reg, _, _ = self._principal.regularize_kernel(k_mat, psd_floor=params.psd_floor)
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
        g4 = np.random.default_rng(5).standard_normal(4)
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

    # ─────────── §II.7b [TERMINAL FASE II] → Fase III §III.1 ───────────
    def build_collider_phase_ii_handoff(self, h: ColliderPhaseIHandoff, k_mat: NDArray,
                                        exponents: NDArray,
                                        params: ColliderParameters) -> ColliderPhaseIIHandoff:
        """MÉTODO TERMINAL DE LA FASE II: cinemática, Breit–Wheeler, estadística y geometría."""
        h = self.accept_collider_phase_i(h)
        kin = self.annihilation_kinematics(params)
        bw = self.breit_wheeler(params)
        pol = self.polarization()
        lum = self.luminosity_poisson(bw, params, self._seed)
        bs = self.bures_spohn_certificate(h, params)
        alg = self.algebra_geometry(k_mat, exponents, params)
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("C2.umbral_aniquilacion", kin.threshold_satisfied, hard=True),
            Evidence("C2.identidad_suma", kin.sum_identity_residual, 1e-12, hard=True),
            Evidence("C2.identidad_producto", kin.product_identity_residual, 1e-12, hard=True),
            Evidence("C2.conservacion_energia", kin.energy_conservation_residual, 1e-12, hard=True),
            Evidence("C2.conservacion_momento", kin.momentum_conservation_residual, 1e-10, hard=True),
            Evidence.boolean("C2.un_foton_prohibido", kin.single_photon_forbidden, hard=True),
            Evidence.boolean("C2.para_2gamma", kin.para_two_gamma_allowed, hard=True),
            Evidence.boolean("C2.orto_2gamma_prohibido", not kin.ortho_two_gamma_allowed, hard=True),
            Evidence.boolean("C2.landau_yang", kin.landau_yang_forbids_j1, hard=True),
            Evidence("C2.bw_sigma_umbral_cero", bw.sigma_at_threshold_re2, 0.0, hard=True),
            Evidence("C2.bw_saturacion_reposo", bw.rest_head_on_saturation_residual, 1e-12, hard=True),
            Evidence("C2.thomson_codata", bw.thomson_codata_residual, 1e-6, hard=True),
            Evidence.boolean("C2.bw_cota_thomson", bw.bound_sigma_le_thomson, hard=True),
            Evidence("C2.bw_pico_ventana", abs(bw.peak_sigma_over_sigma_t - 0.2556), 0.02),
            Evidence("C2.polarizacion_pureza", abs(pol.purity - 1.0), 1e-12, hard=True),
            Evidence("C2.reducida_maximixta", abs(pol.reduced_purity - 0.5), 1e-12, hard=True),
            Evidence("C2.entropia_1bit", abs(pol.entanglement_entropy_bits - 1.0), 1e-12, hard=True),
            Evidence("C2.concurrence", abs(pol.concurrence - 1.0), 1e-12, hard=True),
            Evidence("C2.correlacion_formula", pol.correlation_formula_residual, 1e-12, hard=True),
            Evidence("C2.tsirelson", pol.tsirelson_residual, 5e-3, hard=True),
            Evidence.boolean("C2.violacion_bell", pol.bell_violation, hard=True),
            Evidence.boolean("C2.garwood_cobertura", lum.mc_coverage_ok, hard=True),
            Evidence.boolean("C2.garwood_monotonia", lum.interval_monotone, hard=True),
            Evidence("C2.bures_metodos", bs.method_agreement_residual, 1e-9, hard=True),
            Evidence.boolean("C2.fvg_inferior", bs.fvg_lower_ok, hard=True),
            Evidence.boolean("C2.fvg_superior", bs.fvg_upper_ok, hard=True),
            Evidence("C2.bures_monotonia_cptp", bs.cptp_monotonicity_residual, 1e-12, hard=True),
            Evidence.boolean("C2.spohn_monotonia", bs.spohn_monotone, hard=True),
            Evidence.boolean("C2.spohn_sigma", bs.spohn_sigma_nonnegative, hard=True),
            Evidence("C2.spohn_coherencia", bs.coherence_identity_residual, 1e-10, hard=True),
            Evidence("C2.legacy_fidelidad_tautologica", abs(bs.legacy_self_fidelity - 1.0), 1e-12,
                     note="diagnóstico: Tr(√ρ√ρ)=Tr ρ ≡ 1 (v13)"),
            Evidence.boolean("C2.smith_verificado", alg.smith_verified, hard=True),
            Evidence.boolean("C2.smith_sin_torsion", all(d == 1 for d in alg.smith_invariants), hard=True),
            Evidence("C2.novikov_ultrametrica", alg.novikov_ultrametric_residual, 1e-12, hard=True),
            Evidence("C2.novikov_multiplicatividad", alg.novikov_multiplicativity_residual, 1e-12, hard=True),
            Evidence.boolean("C2.novikov_filtracion", alg.novikov_filtration_compliant),
            Evidence("C2.floer_energia_area", alg.floer_energy_area_residual, 1e-10, hard=True),
            Evidence.boolean("C2.floer_control", alg.floer_control_residual > 10 * alg.floer_cr_bound, hard=True),
            Evidence.boolean("C2.gromov_sin_obstruccion", not alg.gromov_obstruction, hard=True),
            Evidence.boolean("C2.eh_monotonia", alg.eh_monotonicity_ok),
            Evidence("C2.melnikov_cuadratura", alg.melnikov_quadrature_residual, 1e-8, hard=True),
            Evidence.boolean("C2.sin_caos_homoclinico", not alg.melnikov_chaos, hard=True),
            Evidence("C2.legacy_cr_constante_uno", abs(alg.legacy_cr_constant_one - 1.0), 1e-12,
                     note="diagnóstico: el residual CR de v13 era la constante 1"),
        )
        truth = HeytingTruthValue.infimum([h.truth_value] + [e.truth for e in evidences])
        out = ColliderPhaseIIHandoff(
            phase_i=h, kinematics=kin, breit_wheeler=bw, polarization=pol, luminosity=lum,
            bures_spohn=bs, algebra=alg, evidences=evidences, truth_value=truth,
        )
        logger.info("Fase II collider Ω₃=%s · √s=%.6f keV · x_γγ=%.6f · σ=%.4e cm² · CHSH=%.6f",
                    truth.name, kin.sqrt_s_kev, bw.x_ratio, bw.sigma_cm2, pol.chsh_maximum)
        return out  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · SENTENCIA, AUTORIZACION Y ACTUACION  (anida Fases II y I)
# ══════════════════════════════════════════════════════════════════════════════════════
class ColliderPhaseIIIActuation(ColliderPhaseIICrossSections):
    r"""
    FASE III — Sentencia proporcional y actuación SÓLO con autorización humana criptográfica.
    Corrige la inversión de v13: la ausencia de token ya NO dispara.
    """

    PHASE_LABEL: Final[str] = "III·SENTENCIA-AUTORIZACION"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_122, driver: Optional[CrowbarDriver] = None) -> None:
        super().__init__(secret_key, principal_engine=principal_engine, seed=seed)
        self._driver = driver or SimulatedCrowbarDriver(self._verify_pending)

    def accept_collider_phase_ii(self, h: ColliderPhaseIIHandoff) -> ColliderPhaseIIHandoff:
        if not isinstance(h, ColliderPhaseIIHandoff):
            raise ColliderError("Fase III requiere el ColliderPhaseIIHandoff de §II.7.")
        self.accept_collider_phase_i(h.phase_i)
        return h

    # ─────────── §III.2 RSI³ ───────────
    def collider_rsi3(self, h: ColliderPhaseIIHandoff, params: ColliderParameters) -> ColliderRSI3Report:
        r"""
        NIVEL 1 · Calibración de g por Newton con salvaguarda de bisección sobre el proxy
        Frobenius EXACTO S(g)=Σ|ρ_mn|²(1−e^{−g²Δ²/8σ²})²; h de Kantorovich estimado (declarado).
        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip ≤ q, Lip(T³) ≤ q³,
        cotas de Banach a priori/a posteriori y sondeo empírico.
        NIVEL 3 · Leyes de Kleisli, núcleo ¬¬ y jerk de capacidad MEDIDO (SG 7/3).
        """
        p1 = h.phase_i
        n_vals = np.real(np.diag(p1.number_operator))
        sigma = p1.qnd.pointer_sigma
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

        margin = 1.0 - min(1.0, p1.qnd.trace_distance_disturbance / max(params.disturbance_budget, 1e-300))
        q = 1.0 - 0.5 * max(margin, 1e-3)
        theta_star = np.array([g_k, margin, h.breit_wheeler.x_ratio])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(self._seed + 19)
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
        return ColliderRSI3Report(
            calibrated_gain=float(g_k), target_disturbance=float(params.disturbance_budget),
            achieved_proxy=math.sqrt(max(0.0, s_of(g_k))), kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_iterations=int(err.size),
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(emp),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, capacity_d1=d1, capacity_d2=d2c, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2c > 0 and d3 > 0),
            kleisli_monad_laws_ok=laws, nucleus_axioms_ok=nucleus,
        )

    # ─────────── §III.3 KAM + cadencia ───────────
    @staticmethod
    def kam_cadence(params: ColliderParameters, frequencies: Sequence[float] = (1.0, _GOLDEN),
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

        alpha, npts = float(params.cadence_rotation), int(params.cadence_points)
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
            raise ColliderError("Kac: retorno no alcanzado.")
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
        """⊤⇒0 WARN, ½⇒1 DERATE, ⊥⇒2 ISOLATE; TRIP exige fallo estructural + insuficiencia."""
        if truth is HeytingTruthValue.COHERENT:
            return 0
        if truth is HeytingTruthValue.DEGRADED:
            return 1
        return 3 if (structural and lesser_insufficient) else 2

    # ─────────── §III.6 Crowbar medido ───────────
    def crowbar_certificate(self, sentence: SentenceAuthorizationReport,
                            params: ColliderParameters) -> CrowbarActuationReport:
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
        return CrowbarActuationReport(
            dispatched=dispatched, safe_state_applied=safe, driver_name=self._driver.name,
            driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            note=("392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) "
                  "no lo alcanza. Sin token válido NO hay disparo: estado seguro + escalada."),
        )

    # ─────────── §III.7 [TERMINAL] Orquestador ───────────
    def execute_gamma_photon_collider_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho_mac_input: Optional[NDArray] = None,
        *,
        params: Optional[ColliderParameters] = None,
        authorization_token: Optional[str] = None,
        force_veto: bool = False,          # ABOLIDO: se rechaza explícitamente
    ) -> GammaPhotonColliderSpectrumReport:
        r"""
        MÉTODO TERMINAL. Ejecuta §I.5→§II.1→§II.7→§III.1→§III.2..§III.6, sentencia en Ω₃ y
        sella el acta (Merkle SIN clave + pasaporte HMAC determinista). La actuación requiere
        token humano ligado al acta; `force_veto` se rechaza por diseño.
        """
        if force_veto:
            raise ColliderError(
                "force_veto está abolido: un booleano no autenticado no puede causar actuación "
                "física. Presente una violación estructural y un token de autorización firmado.")
        t0 = time.perf_counter()
        params = params or ColliderParameters()
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)

        h1 = self.build_collider_phase_i_handoff(k_mat, exps, params, rho_mac_input=rho_mac_input)
        h2 = self.build_collider_phase_ii_handoff(h1, k_mat, exps, params)
        h2 = self.accept_collider_phase_ii(h2)
        rsi3 = self.collider_rsi3(h2, params)
        kam = self.kam_cadence(params)

        evidences: Tuple[Any, ...] = (
            Evidence("C3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("C3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("C3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("C3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("C3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("C3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("C3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence.boolean("C3.sin_resonancia_kam", not kam.resonance_detected, hard=True),
            Evidence("C3.gamma_diophantino", max(0.0, 1e-3 - kam.diophantine_gamma), 0.0),
            Evidence("C3.ecuacion_homologica", kam.homological_residual, 1e-10, hard=True),
            Evidence.boolean("C3.twist_no_degenerado", abs(kam.twist_det) > 1e-12, hard=True),
            Evidence.boolean("C3.tres_huecos", kam.three_gap_ok, hard=True),
            Evidence.boolean("C3.cobertura_cadencia", kam.coverage_ok, hard=True),
            Evidence.boolean("C3.slater", kam.slater_ok, hard=True),
            Evidence("C3.kac_retardo", kam.kac_relative_residual, 5e-3, hard=True),
        )
        truth = HeytingTruthValue.infimum([h2.truth_value] + [e.truth for e in evidences])
        structural = any((not e.passed) and e.hard for e in
                         list(h1.evidences) + list(h2.evidences) + list(evidences))
        rung = self.proportionality_rung(truth, structural=structural,
                                         lesser_insufficient=params.lesser_measures_insufficient)
        self._epoch += 1
        epoch = self._epoch
        acta_payload = {
            "epoch": epoch, "truth": truth, "rung": rung, "structural": structural,
            "commitment": h1.commitment_digest, "version": __version__,
            "sqrt_s_kev": h2.kinematics.sqrt_s_kev, "x_gg": h2.breit_wheeler.x_ratio,
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
            action=_LADDER[rung], structural_failure=bool(structural), indictment_digest=acta,
            epoch=epoch, token_verified=token_ok, token_self_issued=self_issued,
            ne_bis_in_idem_ok=bool(ne_bis), anti_replay_ok=bool(anti_replay),
            actuation_requested=requested, actuation_authorized=authorized,
        )
        crowbar = self.crowbar_certificate(sentence, params)
        self._judged.add(acta)
        self._last_epoch = epoch

        leaves: List[bytes] = [
            _canonical({"commitment": h1.commitment_digest, "rho": h1.rho_digest,
                        "K": h1.kernel_digest, "exps": h1.exponents_digest}),
            _canonical(h1.fock), _canonical(h1.qnd), _canonical(h2.kinematics),
            _canonical(h2.breit_wheeler), _canonical(h2.polarization), _canonical(h2.luminosity),
            _canonical(h2.bures_spohn), _canonical(h2.algebra), _canonical(rsi3), _canonical(kam),
            _canonical(acta_payload),
            _canonical({k: v for k, v in _freeze(crowbar).items()
                        if k not in ("dispatch_median_ns", "dispatch_p99_ns", "trip_dispatch_ns")}),
        ]
        root = merkle_root(leaves)
        acta_idx = 11
        proof = merkle_proof(leaves, acta_idx)
        payload = {"version": __version__, "truth": truth, "acta": acta, "merkle_root": root,
                   "rung": rung, "authorized": authorized, "commitment": h1.commitment_digest}
        passport = hmac.new(self.key_bytes, _HMAC_ACTA + _canonical(payload), hashlib.sha256).hexdigest()
        report = GammaPhotonColliderSpectrumReport(
            engine_version=__version__, phase_ii=h2, rsi3=rsi3, kam_cadence=kam, sentence=sentence,
            crowbar=crowbar, evidences=evidences, truth_value=truth, merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), acta_leaf_index=acta_idx,
            acta_inclusion_proof=proof, passport_hash=passport,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("Collider v14 Ω₃=%s · %s · autorizado=%s · disparo=%s",
                    truth.name, sentence.action, authorized, crowbar.dispatched)
        return report

    def verify_collider_report(self, r: GammaPhotonColliderSpectrumReport) -> bool:
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


class BoolNovikovGammaPhotonColliderEngine(ColliderPhaseIIIActuation):
    """Fachada pública (nombre conservado). Anidamiento: Fase I ⊂ II ⊂ III ⊂ Fachada."""

    PHASE_LABEL: Final[str] = "FACHADA·GAMMA-PHOTON-COLLIDER"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    eng = BoolNovikovGammaPhotonColliderEngine(secret_key=b"GAMMA_COLLIDER_KEY_32_BYTES_MIN!!")
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)

    # 1) Nominal: CAR exactas, QND pasivo real, cinemática y Breit–Wheeler coherentes, SIN disparo
    rep = eng.execute_gamma_photon_collider_audit(k_test, exps)
    p1, p2 = rep.phase_ii.phase_i, rep.phase_ii
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert p1.fock.car_anticommutator_residual < 1e-12
    assert p1.fock.charge_conservation_residual < 1e-12          # [Q, c₊c₋] = 0
    assert p1.qnd.state_nondisturbance_residual > 1e-6           # ← NO vacuidad (v13: ≡0)
    assert p1.qnd.is_qnd_passive and p1.qnd.back_action_db <= 0.0
    assert abs(p2.kinematics.sqrt_s_kev - 1021.99790138) < 1e-6
    assert p2.kinematics.product_identity_residual < 1e-12
    assert p2.breit_wheeler.sigma_at_threshold_re2 == 0.0        # σ(umbral)=0
    assert p2.breit_wheeler.rest_head_on_saturation_residual < 1e-12
    assert p2.breit_wheeler.thomson_codata_residual < 1e-6
    assert abs(p2.polarization.chsh_maximum - 2 * math.sqrt(2)) < 5e-3
    assert p2.luminosity.mc_coverage_ok and p2.luminosity.interval_monotone
    assert abs(p2.bures_spohn.legacy_self_fidelity - 1.0) < 1e-12   # tautología de v13
    assert rep.sentence.ladder_rung == 0 and not rep.crowbar.dispatched
    assert eng.verify_collider_report(rep)

    # 2) NO vacuidad del QND: g grande ⇒ pasividad rota ⇒ ⊥, ISOLATE, SIN disparo
    broken = eng.execute_gamma_photon_collider_audit(
        k_test, exps, params=ColliderParameters(interaction_g=0.5))
    assert broken.phase_ii.phase_i.qnd.trace_distance_disturbance > 1e-4
    assert not broken.phase_ii.phase_i.qnd.is_qnd_passive
    assert broken.truth_value is HeytingTruthValue.VETOED
    assert broken.sentence.action == "ISOLATE" and not broken.crowbar.dispatched
    assert broken.crowbar.safe_state_applied

    # 3) Violación estructural + insuficiencia declarada ⇒ TRIP solicitado; SIN token NO dispara
    hard = ColliderParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True)
    bad = eng.execute_gamma_photon_collider_audit(k_test, exps, params=hard)
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.sentence.ladder_rung == 3
    assert bad.sentence.actuation_requested and not bad.sentence.actuation_authorized
    assert not bad.crowbar.dispatched and bad.crowbar.safe_state_applied

    # 4) CON token ligado al acta ⇒ disparo autorizado y latencia MEDIDA
    tok = eng.mint_authorization_token(bad.sentence.indictment_digest, bad.sentence.epoch + 1, 3)
    armed = eng.execute_gamma_photon_collider_audit(k_test, exps, params=hard, authorization_token=tok)
    assert armed.sentence.actuation_authorized and armed.crowbar.dispatched
    assert armed.crowbar.trip_dispatch_ns > 0.0
    assert not armed.crowbar.physical_deadline_feasible          # SCR ~µs: declarado

    # 5) force_veto abolido: un booleano no autenticado no puede actuar
    try:
        eng.execute_gamma_photon_collider_audit(k_test, exps, force_veto=True)
        raise AssertionError("force_veto no fue rechazado")
    except ColliderError:
        pass

    # 6) Breit–Wheeler por encima del umbral: σ>0 y pico ≈0.26 σ_T
    above = eng.breit_wheeler(ColliderParameters(photon1_energy_kev=900.0, photon2_energy_kev=900.0))
    assert above.x_ratio > 1.0 and above.sigma_cm2 > 0.0
    assert 0.20 < above.peak_sigma_over_sigma_t < 0.30 and above.bound_sigma_le_thomson

    # 7) Garwood: cobertura ≥ nominal y valores conocidos; CAR/Pauli/paridad exactas
    lo0, hi0 = garwood_interval(0, 0.05)
    assert lo0 == 0.0 and abs(hi0 - 3.6888794541139363) < 1e-6      # ½χ²_{0.975}(2)=3.6889
    fock = p1.fock
    assert fock.pauli_exclusion_residual < 1e-12 and fock.antisymmetry_residual < 1e-12
    assert abs(fock.pair_annihilation_amplitude) == 1.0

    # 8) Merkle RFC 6962, determinismo del pasaporte y detección de manipulación
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    rep2 = eng.execute_gamma_photon_collider_audit(k_test, exps)
    assert rep2.passport_hash != rep.passport_hash                # época distinta ⇒ acta distinta
    assert not eng.verify_collider_report(dataclasses.replace(rep, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderEngine v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {rep.truth_value.name}({int(rep.truth_value)}) / "
          f"{rep.sentence.boolean_shadow.name}")
    print(f"Fock fermiónico m / dim       : {fock.modes} / {fock.dimension} · espectro N̂={fock.number_spectrum}")
    print(f"CAR / Pauli / antisimetría    : {fock.car_anticommutator_residual:.2e} / "
          f"{fock.pauli_exclusion_residual:.2e} / {fock.antisymmetry_residual:.2e}")
    print(f"[Q, c₊c₋] (carga conservada)  : {fock.charge_conservation_residual:.2e} · A_par|11⟩={fock.pair_annihilation_amplitude}")
    print(f"[ρ,N̂] (no vacuidad QND)       : {p1.qnd.state_nondisturbance_residual:.3e}")
    print(f"D_tr / cota / dB              : {p1.qnd.trace_distance_disturbance:.3e} / "
          f"{p1.qnd.disturbance_apriori_bound:.3e} / {p1.qnd.back_action_db:.3e}")
    print(f"A_w / anómalo / p_f           : {p1.qnd.weak_value:.4f} / {p1.qnd.weak_value_anomalous} / "
          f"{p1.qnd.postselection_probability:.4f}")
    print(f"√s / umbral [keV]             : {p2.kinematics.sqrt_s_kev:.6f} / {p2.kinematics.threshold_kev:.6f}")
    print(f"E₁(θ) / E₂ [keV]              : {p2.kinematics.photon1_energy_kev:.6f} / {p2.kinematics.photon2_energy_kev:.6f}")
    print(f"E_max·E_min = s/4 (residuo)   : {p2.kinematics.product_identity_residual:.2e}")
    print(f"Breit–Wheeler x / β           : {p2.breit_wheeler.x_ratio:.9f} / {p2.breit_wheeler.beta_velocity:.3e}")
    print(f"σ_γγ [cm²] / σ(umbral)        : {p2.breit_wheeler.sigma_cm2:.4e} / {p2.breit_wheeler.sigma_at_threshold_re2:.1f}")
    print(f"σ_T calculado / CODATA        : {p2.breit_wheeler.thomson_sigma_cm2:.6e} / {_SIGMA_T_CODATA:.6e}")
    print(f"Pico σ_γγ/σ_T                 : {p2.breit_wheeler.peak_sigma_over_sigma_t:.4f}")
    print(f"R=Lσ [Hz] / N̄ / P(0)          : {p2.luminosity.rate_hz:.4e} / {p2.luminosity.expected_events:.4e} / "
          f"{p2.luminosity.probability_zero_events:.4e}")
    print(f"Garwood [lo,hi] / cobertura MC: [{p2.luminosity.garwood_low:.4f}, {p2.luminosity.garwood_high:.4f}] / "
          f"{p2.luminosity.mc_coverage:.4f}")
    print(f"CHSH / Tsirelson / 1 bit      : {p2.polarization.chsh_maximum:.6f} / {2*math.sqrt(2):.6f} / "
          f"{p2.polarization.entanglement_entropy_bits:.6f}")
    print(f"Spohn σ_min / C_rel residuo   : {p2.bures_spohn.spohn_sigma_min:.4e} / "
          f"{p2.bures_spohn.coherence_identity_residual:.2e}")
    print(f"SNF real / ×100               : {p2.algebra.smith_invariants} / {p2.algebra.scaled_invariants} "
          f"(fabrica torsión={p2.algebra.scaling_fabricates_torsion})")
    print(f"RSI³ g calibrado / Lip(T³)    : {rep.rsi3.calibrated_gain:.6e} / {rep.rsi3.lip_t3_empirical:.6f}")
    print(f"Kac esperado / empírico       : {rep.kam_cadence.kac_expected_delay:.3f} / "
          f"{rep.kam_cadence.kac_empirical_delay:.3f} · Slater {rep.kam_cadence.slater_return_times}")
    print(f"Sentencia nominal             : {rep.sentence.action} (disparo={rep.crowbar.dispatched})")
    print(f"⊥ SIN token                   : {bad.sentence.action} autorizado={bad.sentence.actuation_authorized} "
          f"disparo={bad.crowbar.dispatched} seguro={bad.crowbar.safe_state_applied}")
    print(f"⊥ CON token                   : {armed.sentence.action} disparo={armed.crowbar.dispatched} "
          f"latencia={armed.crowbar.trip_dispatch_ns:.0f} ns")
    print(f"Merkle raíz (SIN clave)       : {rep.merkle_root_hash[:16]}… ({len(rep.merkle_leaves)} hojas)")
    print(f"Acta reverificable            : {eng.verify_collider_report(rep)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")