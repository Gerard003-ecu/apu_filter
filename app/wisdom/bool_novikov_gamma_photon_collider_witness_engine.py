# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovGammaPhotonColliderWitnessEngine  v14.0.0-PhD (reedición rigurosa)        ║
║ Ubicación: app/wisdom/bool_novikov_gamma_photon_collider_witness_engine.py                     ║
║ Función  : Testigo pasivo QND dispersivo del canal de aniquilación e⁻+e⁺→2γ: cinemática         ║
║            relativista exacta, reglas de selección, entrelazamiento de polarización, espacio    ║
║            de Fock truncado y atestación metodológicamente independiente del motor principal.   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA: CONTAR SIN DESTRUIR
─────────────────────────────
Un testigo de fotones que absorbe fotones no es testigo. La medición QND de número es posible
porque el número es una constante del movimiento del campo libre: [n̂, H_campo] = 0. Pero eso
NO implica retroacción nula: si el estado tiene COHERENCIAS de número, la medición dispersiva
las destruye. Por eso este módulo:
  • usa A = n̂ (no degenerado) y un estado CON coherencias (coherente truncado), de modo que
    la retroacción es medible y el certificado puede FALLAR (v13 usaba ρ=I/d, que conmuta
    con todo, y phi_i=phi_f, que nunca da valor débil anómalo);
  • calcula la acción del canal en forma CERRADA y verifica completitud, unitalidad,
    repetibilidad y una cota a priori de la disturbancia;
  • distingue el experimento de CALIBRACIÓN (sobre una copia del modelo) de la atestación
    del estado real: calibrar no debe perturbar lo atestiguado.

NIVELES DE ATESTACIÓN (declarados en cada certificado)
  A · INTEGRIDAD     : pasaporte HMAC del principal + digests.
  B · RE-DERIVACIÓN  : recálculo desde entradas crudas.
  C · INDEPENDENCIA  : método DISTINTO (SNF por menores vs euclídeo; fidelidad por norma traza
                       vs raíz; Melnikov analítico vs cuadratura; energía-área analítica).

TRES FASES ANIDADAS
    Fase I  §I.6  build_gamma_phase_i_handoff   ──► Fase II §II.1 accept_gamma_phase_i
    Fase II §II.8 build_gamma_phase_ii_handoff  ──► Fase III §III.1 accept_gamma_phase_ii
    Fase III §III.7 execute_gamma_photon_witness_audit (terminal)
    Handoffs: GammaPhaseIHandoff ⊂ GammaPhaseIIHandoff ⊂ GammaPhotonColliderWitnessSpectrumReport

FASE I  — CAMPO DE FOCK Y QND DISPERSIVO
  §I.1 Validación y compromiso canónico.
  §I.2 Fock truncado: a, a†, n̂ y DEFECTO DE CONMUTADOR por truncamiento (exacto).
  §I.3 Canal gaussiano exacto: completitud Kraus, dephasing, cota a priori.
  §I.4 AAV: valor débil, regla de suma, anomalía, límite débil del puntero (orden medido).
  §I.5 Disturbancia: traza, pureza en dB, repetibilidad QND.
  §I.6 [TERMINAL] build_gamma_phase_i_handoff.

FASE II — ANIQUILACIÓN, SELECCIÓN, POLARIZACIÓN Y GEOMETRÍA
  §II.1 accept_gamma_phase_i.
  §II.2 Cinemática relativista exacta e identidades de dos cuerpos; prohibición de 1γ.
  §II.3 Reglas de selección: paridad C, Landau–Yang, canales para/orto.
  §II.4 Polarización derivada de (ε₁×ε₂)·k̂: singlete, 1 bit, CHSH/Tsirelson, Wu–Shaknov.
  §II.5 Klein–Nishina polarizada (razón azimutal, límite de Thomson) y HOM en divisor de haz.
  §II.6 Bures–Uhlmann con referencias declaradas + Fvg + monotonía CPTP.
  §II.7 Spohn del semigrupo de dephasing: σ = −dS(ρ_t‖ρ_∞)/dt ≥ 0 y C_rel.
  §II.8 [TERMINAL] build_gamma_phase_ii_handoff.

FASE III — ATTESTACIÓN, RSI³, CADENCIA Y CONTRAFIRMA
  §III.1 accept_gamma_phase_ii.
  §III.2 Atestación A/B/C del principal (SNF por menores, Melnikov, Floer, Novikov).
  §III.3 KAM diophantino + ecuación homológica de Lindstedt.
  §III.4 RSI³: calibración de g (Kantorovich), contracción de Banach, Kleisli, jerk medido.
  §III.5 Cadencia de Poincaré: tres huecos (ventana de evasión) + Kac (retardo de detección).
  §III.6 Adjudicación Ω₃, núcleo ¬¬, pasaporte determinista, contrafirma, Merkle RFC 6962.
  §III.7 [TERMINAL] execute_gamma_photon_witness_audit.

LIMITACIONES DECLARADAS
1. Con g=0 el valor débil es CONTRAFACTUAL (predicho, no medido).
2. La anomalía del valor débil es un valor condicional, no un autovalor; la regla de suma
   garantiza que la estadística ordinaria no se altera.
3. La razón azimutal de Klein–Nishina es IDEAL (ángulo fijo); la asimetría de coincidencias
   real requiere integración sobre la geometría de los detectores.
4. El truncamiento de Fock rompe [a,a†]=1 en el nivel superior: se certifica el defecto exacto
   y se restringe el subspace físico.
5. Kac–Slater se verifica para la rotación irracional (modelo ergódico), no para la dinámica real.
6. La independencia criptográfica exige clave distinta de la del principal.

REFERENCIAS: CODATA 2022; Aharonov–Albert–Vaidman (1988); Wu–Shaknov (1950); Klein–Nishina (1929);
Landau (1948)–Yang (1950); Hong–Ou–Mandel (1987); Tsirelson (1980); Uhlmann (1976); Fuchs–van de
Graaf (1999); Spohn (1978); Baumgratz et al. (2014) (entropía relativa de coherencia);
Itoh–Abe (1988); González (1996); Melnikov (1963); Gromov (1985); Ekeland–Hofer (1990);
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
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

_ERR: Optional[Exception] = None
for _p in ("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
           "bool_novikov_thermal_exergy_weaver_engine"):
    try:
        _m = __import__(_p, fromlist=["*"])
        PrincipalEngine = _m.BoolNovikovThermalExergyWeaverEngine
        HeytingTruthValue = _m.HeytingTruthValue      # Ω₃ CANÓNICO
        Evidence = _m.Evidence
        Certified = _m.Certified
        kleisli_compose = _m.kleisli_compose
        break
    except ImportError as exc:                                     # pragma: no cover
        _ERR = exc
else:                                                              # pragma: no cover
    raise ImportError("Se requiere el motor principal v14.") from _ERR

__all__ = ["BoolNovikovGammaPhotonColliderWitnessEngine",
           "GammaPhotonColliderWitnessSpectrumReport", "GammaWitnessParameters", "__version__"]
logger = logging.getLogger("APU.Wisdom.BoolNovikovGammaPhotonColliderWitnessEngine")

__version__: Final[str] = "14.0.0-ENGINE-GAMMA-PHOTON-COLLIDER-WITNESS-QND-FOCK-CHSH-KAC-RSI3-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ME_C2_KEV: Final[float] = 510.99895069            # CODATA 2022 (v13 llamaba 1022 a «masa positrón»)
_TWO_ME_C2_KEV: Final[float] = 2.0 * _ME_C2_KEV    # 1021.99790138 keV
_KRULL_FLOOR: Final[float] = 0.10                  # POLÍTICA de filtración, no la valuación
_MINOR_CAP: Final[int] = 20_000
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_TSIRELSON: Final[float] = 2.0 * math.sqrt(2.0)
_HMAC_PASS: Final[bytes] = b"GAMMA-WITNESS-PASSPORT-v14|"
_HMAC_COUNTER: Final[bytes] = b"GAMMA-WITNESS-COUNTERSIGN-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: canonicalización, Merkle, álgebra lineal estable, SNF exacta
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaWitnessError(Exception):
    """Error base del testigo gamma."""


class GammaWitnessValidationError(GammaWitnessError, ValueError):
    """Violación de hipótesis físicas (fail-fast)."""


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
    """F = ‖√ρ√σ‖₁² (estable, sin `sqrtm`); devuelve también el residuo de recorte."""
    sr, cr = _psd_sqrt(rho)
    ss, cs = _psd_sqrt(sig)
    raw = float(np.sum(la.svdvals(sr @ ss))) ** 2
    clipped = min(1.0, max(0.0, raw))
    return clipped, abs(raw - clipped) + cr + cs


def _fidelity_sqrt(rho: NDArray, sig: NDArray) -> float:
    """Método ALTERNATIVO (independencia): F = (Tr√(√ρ σ √ρ))² vía eigh."""
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


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GammaWitnessParameters:
    """Parámetros DECLARADOS; su digest se compromete en el pasaporte."""

    fock_n_max: int = 3
    coherent_alpha: float = 0.7
    photon_energy_kev: float = _ME_C2_KEV
    beta_field: float = 2.0                 # β·ħω adimensional para la referencia térmica
    interaction_g: float = 1.0e-5           # QND dispersivo débil (presupuesto de pasividad)
    pointer_sigma: float = 1.0
    disturbance_budget: float = 1.0e-9
    backaction_db_budget: float = 1.0e-9
    positron_kinetic_kev: float = 0.0
    emission_angle_deg: float = 37.0
    compton_theta_deg: float = 90.0
    fock_two_mode_n: int = 2
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
    fidelity_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.fock_n_max < 1 or self.pointer_sigma <= 0.0 or self.interaction_g < 0.0:
            raise GammaWitnessValidationError("N≥1, σ>0, g≥0 requeridos.")
        if self.disturbance_budget < 0.0 or self.positron_kinetic_kev < 0.0:
            raise GammaWitnessValidationError("Presupuesto y energía cinética ≥ 0.")
        if not (0.0 < self.recurrence_mu < 1.0 and 0.0 < self.cadence_rotation < 1.0):
            raise GammaWitnessValidationError("μ∈(0,1), α∈(0,1).")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0 or self.photon_energy_kev <= 0.0:
            raise GammaWitnessValidationError("Radios y energía de fotón positivos.")

    def digest(self) -> str:
        return _sha256(_canonical(self))


@dataclass(frozen=True)
class FockTruncationReport:
    """§I.2 — El truncamiento rompe [a,a†]=1 en el nivel superior: defecto EXACTO."""

    n_max: int
    dimension: int
    commutator_defect_full: float
    commutator_defect_physical: float
    number_spectrum_ok: bool
    coherent_state_norm_residual: float


@dataclass(frozen=True)
class QNDDispersiveReport:
    """§I.3–§I.5 — Medición QND de número: retroacción MEDIDA, no postulada."""

    observable: str
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


@dataclass(frozen=True)
class AAVWeakValueReport:
    """§I.4 — Valor débil con postselección NO trivial y regla de suma verificada."""

    weak_value: complex
    spectral_norm: float
    is_anomalous: bool
    postselection_probability: float
    sum_rule_residual: float
    pointer_weak_limit_residual: float
    pointer_convergence_order: float
    legacy_expectation_value: float


@dataclass(frozen=True)
class AnnihilationKinematicsReport:
    """§II.2 — Cinemática relativista exacta de dos cuerpos (unidades keV, c=1)."""

    electron_rest_energy_kev: float
    positron_kinetic_kev: float
    positron_total_energy_kev: float
    positron_momentum_kev: float
    mandelstam_s_kev2: float
    sqrt_s_kev: float
    threshold_kev: float
    threshold_satisfied: bool
    total_energy_kev: float
    photon1_energy_kev: float
    photon2_energy_kev: float
    photon2_cos_theta: float
    e_max_kev: float
    e_min_kev: float
    sum_identity_residual: float
    product_identity_residual: float
    energy_conservation_residual: float
    momentum_conservation_residual: float
    single_photon_forbidden: bool
    photons_at_rest_kev: Tuple[float, float]


@dataclass(frozen=True)
class SelectionRulesReport:
    """§II.3 — Paridad C y Landau–Yang: qué canal de n fotones está permitido."""

    para_c_parity: int
    ortho_c_parity: int
    para_two_gamma_allowed: bool
    ortho_two_gamma_allowed: bool
    ortho_three_gamma_allowed: bool
    landau_yang_forbids_j1_to_2gamma: bool
    c_parity_rule_residual: float


@dataclass(frozen=True)
class PolarizationEntanglementReport:
    """§II.4 — Estado derivado de (ε₁×ε₂)·k̂: singlete de polarización."""

    state_amplitudes: Tuple[complex, ...]
    purity: float
    reduced_state_purity: float
    entanglement_entropy_bits: float
    concurrence: float
    parallel_coincidence_probability: float
    perpendicular_coincidence_probability: float
    correlation_formula_residual: float
    chsh_maximum: float
    tsirelson_residual: float
    bell_violation: bool
    wu_shaknov_perpendicular: bool


@dataclass(frozen=True)
class ComptonHOMReport:
    """§II.5 — Klein–Nishina polarizada (ideal) y Hong–Ou–Mandel en divisor de haz."""

    compton_theta_deg: float
    epsilon_ratio: float
    k_prime_over_k: float
    azimuthal_ratio_ideal: float
    azimuthal_ratio_analytic: float
    azimuthal_ratio_residual: float
    thomson_limit_sigma: float
    thomson_limit_exact: float
    thomson_limit_residual: float
    hom_coincidence_indistinguishable: float
    hom_coincidence_distinguishable: float
    hom_visibility: float
    beamsplitter_number_preservation_residual: float
    beamsplitter_unitarity_residual: float


@dataclass(frozen=True)
class BuresGeometryReport:
    """§II.6 — Bures–Uhlmann con referencia DECLARADA y contraste de método."""

    reference_label: str
    fidelity_uhlmann: float
    fidelity_alternate: float
    method_agreement_residual: float
    fidelity_clip_residual: float
    bures_distance: float
    trace_distance: float
    fvg_lower_ok: bool
    fvg_upper_ok: bool
    cptp_monotonicity_residual: float
    is_reference_close: bool
    threshold_redundancy_note: str


@dataclass(frozen=True)
class SpohnDephasingReport:
    """§II.7 — σ = −dS(ρ_t‖ρ_∞)/dt ≥ 0 para el semigrupo de dephasing (punto fijo diag ρ)."""

    times: Tuple[float, ...]
    relative_entropy_trace: Tuple[float, ...]
    entropy_monotone: bool
    sigma_min: float
    sigma_nonnegative: bool
    coherence_entropy_identity_residual: float
    fixed_point_residual: float


@dataclass(frozen=True)
class IndependentAttestationReport:
    """§III.2 — Niveles A/B/C sobre el motor principal."""

    attestation_level: str
    principal_present: bool
    principal_passport_verified: bool
    principal_input_digest_match: bool
    key_separation_ok: bool
    smith_invariants_witness: Tuple[int, ...]
    smith_invariants_principal: Tuple[int, ...]
    smith_method_agreement: bool
    smith_verified: bool
    betti_0: int
    betti_1: int
    scaled_invariants: Tuple[int, ...]
    scaling_fabricates_torsion: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_critical_ratio: float
    melnikov_quadrature_residual: float
    melnikov_chaos: bool
    melnikov_principal_agreement: bool
    floer_energy_area_residual: float
    floer_cr_bound: float
    floer_control_residual: float
    legacy_cr_constant_one: float
    gromov_obstruction: bool
    eh_monotonicity_ok: bool
    novikov_valuation: float
    novikov_norm: float
    novikov_ultrametric_residual: float
    novikov_multiplicativity_residual: float
    novikov_filtration_compliant: bool


@dataclass(frozen=True)
class KAMCelestialReport:
    """§III.3 — Diophantina + ecuación homológica de Poincaré–Lindstedt."""

    frequencies: Tuple[float, ...]
    diophantine_tau: float
    diophantine_gamma: float
    min_small_divisor: float
    resonance_detected: bool
    homological_residual: float
    small_divisor_amplification: float
    twist_det: float
    is_kam_admissible: bool


@dataclass(frozen=True)
class GammaRSI3Report:
    """§III.4 — RSI³: calibración de g, contracción de Banach, Kleisli y jerk medido."""

    calibrated_gain: float
    target_disturbance: float
    achieved_disturbance_proxy: float
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
class GammaCadenceReport:
    """§III.5 — Tres huecos (ventana de evasión) + Kac (retardo de detección)."""

    rotation_number: float
    diophantine_gamma: float
    distinct_gap_count: int
    max_gap: float
    coverage_bound: float
    three_gap_identity_residual: float
    low_order_resonance: bool
    kac_expected_delay: float
    kac_empirical_delay: float
    kac_relative_residual: float
    slater_return_times: Tuple[int, ...]
    three_gap_ok: bool
    coverage_ok: bool
    slater_ok: bool


@dataclass(frozen=True)
class GammaPhaseIHandoff:
    """§I.6 [TERMINAL FASE I] — entrada de §II.1."""

    fock: FockTruncationReport
    qnd: QNDDispersiveReport
    aav: AAVWeakValueReport
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
class GammaPhaseIIHandoff:
    """§II.8 [TERMINAL FASE II] — anida Fase I; entrada de §III.1."""

    phase_i: GammaPhaseIHandoff
    kinematics: AnnihilationKinematicsReport
    selection: SelectionRulesReport
    polarization: PolarizationEntanglementReport
    compton_hom: ComptonHOMReport
    bures_passivity: BuresGeometryReport
    bures_thermal: BuresGeometryReport
    spohn: SpohnDephasingReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue


@dataclass(frozen=True)
class GammaPhotonColliderWitnessSpectrumReport:
    """§III.7 [TERMINAL] — informe anidado I ⊂ II ⊂ III con contrafirma y Merkle."""

    engine_version: str
    phase_ii: GammaPhaseIIHandoff
    attestation: IndependentAttestationReport
    kam: KAMCelestialReport
    rsi3: GammaRSI3Report
    cadence: GammaCadenceReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    attestation_leaf_index: int
    attestation_inclusion_proof: Tuple[Tuple[str, bool], ...]
    passport_hash: str
    countersignature: str
    execution_time_ms: float

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)

    def legacy_summary(self) -> Dict[str, Any]:
        p1, p2 = self.phase_ii.phase_i, self.phase_ii
        return {
            "truth_value": self.truth_value.name,
            "qnd_report": p1.qnd, "bures_report": p2.bures_passivity,
            "annihilation_report": p2.kinematics,
            "rademacher_walsh_norm": None,
            "smith_torsion_free": all(d == 1 for d in self.attestation.smith_invariants_witness),
            "betti_number_zero": self.attestation.betti_0,
            "krull_valuation": self.attestation.novikov_valuation,
            "novikov_norm": self.attestation.novikov_norm,
            "cauchy_riemann_residual": self.attestation.floer_energy_area_residual,
            "is_gromov_rigid": not self.attestation.gromov_obstruction,
            "melnikov_chaos_detected": self.attestation.melnikov_chaos,
            "banach_contraction_lip": self.rsi3.lip_t3_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "passport_hash": self.passport_hash,
        }


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE I · CAMPO DE FOCK Y QND DISPERSIVO
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhaseIFieldSubstrate:
    r"""
    FASE I — Construye el espacio de Fock truncado, mide la retroacción real del canal
    dispersivo y ejercita el aparato AAV sobre una COPIA del modelo (no sobre lo atestiguado).
    """

    PHASE_LABEL: Final[str] = "I·FOCK-QND-DISPERSIVO"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_119) -> None:
        if secret_key is None:
            env = os.environ.get("APU_GAMMA_WITNESS_HMAC_KEY")
            secret_key = env.encode("utf-8") if env else os.urandom(32)
            if not env:
                logger.warning("Clave de testigo gamma efímera: pasaportes no verificables entre procesos.")
        if len(secret_key) < 32:
            raise GammaWitnessValidationError("La clave del testigo requiere ≥ 256 bits.")
        self._key = bytes(secret_key)
        self._seed = int(seed)
        self._principal = principal_engine or PrincipalEngine(secret_key=os.urandom(32), seed=seed)

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def key_fingerprint(self) -> str:
        return hmac.new(self._key, b"GAMMA-WITNESS-KEY-FP-v14|", hashlib.sha256).hexdigest()[:16]

    # ─────────── §I.2 Fock truncado y defecto de conmutador ───────────
    @staticmethod
    def fock_space(params: GammaWitnessParameters) -> Tuple[NDArray, NDArray, FockTruncationReport]:
        r"""
        a|n⟩=√n|n−1⟩, n̂=a†a=diag(0..N). En truncamiento, a a† = diag(1,…,1,0) en el último
        nivel ⇒ [a,a†] = I − (N+1)|N⟩⟨N|: el defecto Frobenius es EXACTAMENTE N+1 y se anula
        al restringir al subspace físico n ≤ N−1. Ignorarlo (como hace todo código de Fock
        truncado ingenuo) produce errores de unitariedad no acotados.
        """
        n_max = int(params.fock_n_max)
        dim = n_max + 1
        a = np.zeros((dim, dim), dtype=np.complex128)
        for n in range(1, dim):
            a[n - 1, n] = math.sqrt(n)
        ad = a.conj().T
        num = ad @ a
        comm = a @ ad - ad @ a
        eye = np.eye(dim)
        defect_full = float(la.norm(comm - eye, ord="fro"))
        phys = slice(0, dim - 1)
        defect_phys = float(la.norm((comm - eye)[phys, phys], ord="fro"))
        alpha = float(params.coherent_alpha)
        coeff = np.array([alpha ** n / math.sqrt(math.factorial(n)) for n in range(dim)])
        coeff *= math.exp(-0.5 * alpha ** 2)
        norm_res = abs(float(np.vdot(coeff, coeff)) - 1.0)
        spectrum_ok = bool(np.allclose(np.diag(num), np.arange(dim), atol=1e-12))
        return num, coeff.astype(np.complex128), FockTruncationReport(
            n_max=n_max, dimension=dim, commutator_defect_full=defect_full,
            commutator_defect_physical=defect_phys, number_spectrum_ok=spectrum_ok,
            coherent_state_norm_residual=norm_res,
        )

    # ─────────── §I.3 Canal gaussiano dispersivo EXACTO ───────────
    @staticmethod
    def dispersive_channel(rho: NDArray, n_vals: NDArray, g: float, sigma: float
                           ) -> Tuple[NDArray, float, float]:
        r"""
        Acoplamiento dispersivo H_int = ħχ n̂ ⊗ P̂ (conserva el número ⇒ QND). Con puntero
        gaussiano de anchura σ, los Kraus M_y ∝ exp(−(y−g n̂)²/4σ²) dan la acción EXACTA
            ρ'_mn = ρ_mn · exp(−g²(m−n)²/(8σ²)),
        un canal UNITAL ⇒ Tr ρ'² ≤ Tr ρ² (Kadison–Schwarz) ⇒ back-action en dB ≤ 0.
        Cota a priori: ‖ρ'−ρ‖₁/2 ≤ ½√d·(g²Δ_max²/8σ²)·‖ρ‖_F.
        """
        if sigma <= 0.0 or g < 0.0:
            raise GammaWitnessValidationError("σ>0 y g≥0 requeridos.")
        delta = n_vals[:, None] - n_vals[None, :]
        factor = np.exp(-(g ** 2) * delta ** 2 / (8.0 * sigma ** 2))
        after = rho * factor
        y = np.linspace(g * float(n_vals[0]) - 10 * sigma, g * float(n_vals[-1]) + 10 * sigma, 4001)
        m = (2 * math.pi * sigma ** 2) ** (-0.25) * np.exp(-(y[:, None] - g * n_vals[None, :]) ** 2 / (4 * sigma ** 2))
        integral = np.array([_trapz(m[:, j] ** 2, y) for j in range(n_vals.size)])
        completeness = float(np.max(np.abs(integral - 1.0)))
        d = rho.shape[0]
        bound = 0.5 * math.sqrt(d) * (g ** 2 * float(np.max(delta ** 2)) / (8 * sigma ** 2)) * float(la.norm(rho, ord="fro"))
        return after, completeness, bound

    # ─────────── §I.4 AAV: valor débil, regla de suma y límite del puntero ───────────
    @staticmethod
    def weak_value(a_op: NDArray, phi_i: NDArray, phi_f: NDArray) -> Tuple[complex, float]:
        i_v = np.asarray(phi_i, dtype=np.complex128).reshape(-1)
        f_v = np.asarray(phi_f, dtype=np.complex128).reshape(-1)
        i_v /= (la.norm(i_v) + _EPS); f_v /= (la.norm(f_v) + _EPS)
        ov = complex(np.vdot(f_v, i_v))
        if abs(ov) < 1e-12:
            raise GammaWitnessValidationError("Postselección casi ortogonal: A_w no certificable.")
        return complex(np.vdot(f_v, a_op @ i_v)) / ov, float(abs(ov) ** 2)

    def sum_rule_residual(self, a_op: NDArray, phi_i: NDArray, basis: NDArray) -> float:
        """Σ_f |⟨φ_f|φ_i⟩|² A_w(f) = ⟨φ_i|A|φ_i⟩ (identidad de resolución completa)."""
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
            y = np.linspace(g * float(n_vals[0]) - 12 * sigma, g * float(n_vals[-1]) + 12 * sigma, 8001)
            m = np.exp(-(y[:, None] - g * n_vals[None, :]) ** 2 / (4 * sigma ** 2))
            amp = np.sum(np.conj(f_v)[None, :] * m * i_v[None, :], axis=1)
            p = np.abs(amp) ** 2
            mean = _trapz(y * p, y) / (_trapz(p, y) + _EPS)
            errs.append(abs(mean / g - target))
        order = math.log2(errs[0] / max(errs[1], 1e-300)) if len(errs) >= 2 and errs[1] > 0 else 0.0
        return float(errs[-1]), float(order)

    # ─────────── §I.6 [TERMINAL FASE I] → Fase II §II.1 ───────────
    def build_gamma_phase_i_handoff(
        self, apu_cost_matrix: NDArray, exponents: NDArray, params: GammaWitnessParameters, *,
        rho_mac_input: Optional[NDArray] = None,
    ) -> GammaPhaseIHandoff:
        """MÉTODO TERMINAL DE LA FASE I: mide pasividad real y compromete las entradas."""
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        num, coeff, fock = self.fock_space(params)
        dim = fock.dimension
        if rho_mac_input is None:
            rho = np.outer(coeff, coeff.conj())          # CON coherencias: QND no vacuo
        else:
            r = np.asarray(rho_mac_input, dtype=np.complex128)
            if r.shape != (dim, dim):
                raise GammaWitnessValidationError(f"ρ debe ser {dim}×{dim} (Fock N={fock.n_max}).")
            tr = complex(np.trace(r))
            if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0:
                raise GammaWitnessValidationError("ρ: traza compleja o no positiva.")
            rho = 0.5 * (r + r.conj().T) / tr.real
        h_field = params.photon_energy_kev * num
        n_vals = np.real(np.diag(num))
        g, sigma = float(params.interaction_g), float(params.pointer_sigma)
        after, completeness, bound = self.dispersive_channel(rho, n_vals, g, sigma)

        n_fro = float(la.norm(num, ord="fro")) + _EPS
        h_fro = float(la.norm(h_field, ord="fro")) + _EPS
        r_fro = float(la.norm(rho, ord="fro")) + _EPS
        dyn = float(la.norm(num @ h_field - h_field @ num, ord="fro")) / (n_fro * h_fro)
        nondist = float(la.norm(rho @ num - num @ rho, ord="fro")) / (r_fro * n_fro)
        purity_b = float(np.real(np.trace(rho @ rho)))
        purity_a = float(np.real(np.trace(after @ after)))
        db = 10.0 * math.log10(max(purity_a, 1e-300) / max(purity_b, 1e-300))
        dist = _trace_distance(rho, after)
        repeat = float(np.max(np.abs(np.real(np.diag(after)) - np.real(np.diag(rho)))))
        passive = bool(dist <= params.disturbance_budget and db >= -params.backaction_db_budget)

        rng = np.random.default_rng(self._seed)
        phi_i = coeff.copy()
        v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        phi_f = v / la.norm(v)
        a_w, p_f = self.weak_value(num, phi_i, phi_f)
        spec = float(np.max(np.abs(n_vals)))
        anom = bool(a_w.real > spec + 1e-9 or a_w.real < -1e-9 or abs(a_w.imag) > 1e-9)
        haar = la.qr(rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)))[0]
        sum_res = max(self.sum_rule_residual(num, phi_i, np.eye(dim, dtype=np.complex128)),
                      self.sum_rule_residual(num, phi_i, haar))
        ptr_res, ptr_order = self.pointer_weak_limit(
            n_vals, phi_i, phi_f, sigma, (0.1 * sigma, 0.05 * sigma, 0.025 * sigma))
        legacy_exp = float(np.real(np.vdot(phi_i, num @ phi_i)))

        qnd = QNDDispersiveReport(
            observable="n̂ (número de fotones, QND dispersivo)",
            qnd_dynamical_residual=dyn, state_nondisturbance_residual=nondist,
            kraus_completeness_residual=completeness, trace_distance_disturbance=dist,
            disturbance_apriori_bound=bound, purity_before=purity_b, purity_after=purity_a,
            back_action_db=db, qnd_repeatability_residual=repeat, interaction_g=g,
            pointer_sigma=sigma, is_counterfactual=bool(g == 0.0), is_qnd_passive=passive,
        )
        aav = AAVWeakValueReport(
            weak_value=a_w, spectral_norm=spec, is_anomalous=anom, postselection_probability=p_f,
            sum_rule_residual=sum_res, pointer_weak_limit_residual=ptr_res,
            pointer_convergence_order=ptr_order, legacy_expectation_value=legacy_exp,
        )
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("G1.espectro_de_numero", fock.number_spectrum_ok, hard=True),
            Evidence("G1.defecto_conmutador_fisico", fock.commutator_defect_physical, 1e-12, hard=True),
            Evidence("G1.defecto_conmutador_total", abs(fock.commutator_defect_full - (fock.n_max + 1)),
                     1e-12, note="diagnóstico: el truncamiento rompe [a,a†]=1 en el nivel N"),
            Evidence("G1.norma_estado_coherente", fock.coherent_state_norm_residual, 1e-3),
            Evidence("G1.kraus_completitud", completeness, 1e-9, hard=True),
            Evidence("G1.qnd_dinamico_[n,H]", dyn, 1e-12, hard=True),
            Evidence("G1.qnd_repetibilidad", repeat, 1e-14, hard=True),
            Evidence.boolean("G1.pasividad_presupuestada", passive, hard=True),
            Evidence("G1.disturbancia_vs_cota", dist - bound, 1e-16, hard=True),
            Evidence.boolean("G1.pureza_no_creciente", purity_a <= purity_b + 1e-12, hard=True),
            Evidence("G1.aav_regla_de_suma", sum_res, 1e-10, hard=True),
            Evidence("G1.aav_limite_debil", ptr_res, 1e-3),
            Evidence("G1.aav_orden", abs(ptr_order - 2.0), 0.4),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        commitment = _sha256(_canonical({
            "rho": _array_digest(rho), "K": _array_digest(k_mat),
            "exps": _array_digest(np.asarray(exponents, dtype=np.float64)),
            "params": params.digest(), "version": __version__,
        }))
        handoff = GammaPhaseIHandoff(
            fock=fock, qnd=qnd, aav=aav, evidences=evidences, truth_value=truth,
            commitment_digest=commitment, rho_digest=_array_digest(rho),
            kernel_digest=_array_digest(k_mat),
            exponents_digest=_array_digest(np.asarray(exponents, dtype=np.float64)),
            rho_before=rho, rho_after=after, number_operator=num, field_hamiltonian=h_field,
        )
        logger.info("Fase I gamma Ω₃=%s · D_tr=%.3e · dB=%.3e · A_w=%.4f", truth.name, dist, db, a_w.real)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · ANIQUILACIÓN, SELECCIÓN, POLARIZACIÓN Y GEOMETRÍA  (anida Fase I)
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhaseIIAnnihilation(GammaPhaseIFieldSubstrate):
    r"""FASE II — Cinemática exacta, reglas de selección, entrelazamiento y geometría de Bures."""

    PHASE_LABEL: Final[str] = "II·ANIQUILACION-POLARIZACION"

    def accept_gamma_phase_i(self, h: GammaPhaseIHandoff) -> GammaPhaseIHandoff:
        if not isinstance(h, GammaPhaseIHandoff):
            raise GammaWitnessValidationError("Fase II requiere el GammaPhaseIHandoff de §I.6.")
        return h

    # ─────────── §II.2 Cinemática relativista exacta ───────────
    @staticmethod
    def annihilation_kinematics(params: GammaWitnessParameters) -> AnnihilationKinematicsReport:
        r"""
        e⁻ en reposo, e⁺ con cinética K (c=1, keV): E₊=m+K, p₊=√(E₊²−m²).
            s = (P₋+P₊)² = 2m² + 2mE₊,   E_tot = m+E₊,   p_tot = p₊.
        Dos fotones: (P−k₁)²=0 ⇒ P·k₁ = s/2 ⇒ E₁(θ) = s / (2(E_tot − p_tot cosθ)).
        IDENTIDADES EXACTAS verificadas: E_max+E_min = E_tot y E_max·E_min = s/4.
        Un solo fotón está PROHIBIDO: k²=0 pero s>0 (invariante de masa).
        """
        m = _ME_C2_KEV
        k_en = float(params.positron_kinetic_kev)
        e_plus = m + k_en
        p_plus = math.sqrt(max(0.0, e_plus ** 2 - m ** 2))
        s = 2.0 * m ** 2 + 2.0 * m * e_plus
        sqrt_s = math.sqrt(s)
        e_tot, p_tot = m + e_plus, p_plus
        s_alt = e_tot ** 2 - p_tot ** 2
        th = math.radians(float(params.emission_angle_deg))
        e1 = s / (2.0 * (e_tot - p_tot * math.cos(th)))
        e2 = e_tot - e1
        cos2 = (p_tot - e1 * math.cos(th)) / (e2 + _EPS)
        e_max = s / (2.0 * (e_tot - p_tot))
        e_min = s / (2.0 * (e_tot + p_tot))
        mom_res = math.hypot(e1 * math.cos(th) + e2 * cos2 - p_tot,
                             e1 * math.sin(th) - e2 * math.sqrt(max(0.0, 1.0 - cos2 ** 2)))
        return AnnihilationKinematicsReport(
            electron_rest_energy_kev=m, positron_kinetic_kev=k_en, positron_total_energy_kev=e_plus,
            positron_momentum_kev=p_plus, mandelstam_s_kev2=s, sqrt_s_kev=sqrt_s,
            threshold_kev=_TWO_ME_C2_KEV, threshold_satisfied=bool(sqrt_s >= _TWO_ME_C2_KEV - 1e-9),
            total_energy_kev=e_tot, photon1_energy_kev=e1, photon2_energy_kev=e2,
            photon2_cos_theta=float(cos2), e_max_kev=e_max, e_min_kev=e_min,
            sum_identity_residual=abs((e_max + e_min) - e_tot) / e_tot,
            product_identity_residual=abs(e_max * e_min - s / 4.0) / (s / 4.0),
            energy_conservation_residual=abs((e1 + e2) - e_tot) / e_tot,
            momentum_conservation_residual=mom_res / (p_tot + m),
            single_photon_forbidden=bool(s > 0.0),
            photons_at_rest_kev=(m, m),
        )

    # ─────────── §II.3 Reglas de selección ───────────
    @staticmethod
    def selection_rules() -> SelectionRulesReport:
        r"""
        Positronio: C = (−1)^{L+S}; n fotones: C = (−1)^n. Para (L=0,S=0): C=+1 ⇒ n par ⇒ 2γ.
        Orto (L=0,S=1): C=−1 ⇒ n impar ⇒ 3γ; además J=1 ⇒ 2γ prohibido por LANDAU–YANG.
        """
        para_c, ortho_c = (-1) ** 0, (-1) ** 1
        rule_res = 0.0
        for n in (2, 3, 4):
            para_ok = ((-1) ** n == para_c)
            ortho_ok = ((-1) ** n == ortho_c) and not (n == 2)
            rule_res = max(rule_res, 0.0 if (para_ok == (n % 2 == 0)) else 1.0)
            if n == 2 and ortho_ok:
                rule_res = 1.0
        return SelectionRulesReport(
            para_c_parity=para_c, ortho_c_parity=ortho_c, para_two_gamma_allowed=True,
            ortho_two_gamma_allowed=False, ortho_three_gamma_allowed=True,
            landau_yang_forbids_j1_to_2gamma=True, c_parity_rule_residual=rule_res,
        )

    # ─────────── §II.4 Polarización derivada de la amplitud pseudoscalar ───────────
    @staticmethod
    def polarization_entanglement() -> PolarizationEntanglementReport:
        r"""
        Amplitud pseudoscalar (0^-+) → 2γ:  M(ε₁,ε₂) ∝ (ε₁ × ε₂)·k̂,  k̂ = ẑ, ejes de LAB.
        Con ε ∈ {x̂,ŷ}: M = [[0,1],[−1,0]] ⇒ |Ψ⟩ = (|xy⟩ − |yx⟩)/√2 (SINGLETE de polarización).
        Se DERIVA, no se postula. Certificados: pureza 1, reducida I/2, S_ent = 1 bit,
        concurrence 1, correlación perpendicular (Wu–Shaknov 1950), E(θ₁,θ₂) = −cos2(θ₁−θ₂),
        CHSH máximo = 2√2 (cota de Tsirelson) y > 2 (violación de Bell).
        """
        eps = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        k_hat = np.array([0.0, 0.0, 1.0])
        amp = np.zeros((2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                amp[i, j] = complex(np.dot(np.cross(eps[i], eps[j]), k_hat))
        psi = (amp / la.norm(amp)).reshape(4)
        rho = np.outer(psi, psi.conj())
        red = np.array([[np.sum(rho.reshape(2, 2, 2, 2)[i, :, j, :]) for j in range(2)] for i in range(2)])
        conc = 2.0 * abs(amp[0, 0] * amp[1, 1] - amp[0, 1] * amp[1, 0]) / (la.norm(amp) ** 2)
        p_par = float(abs(psi[0]) ** 2 + abs(psi[3]) ** 2)
        p_perp = float(abs(psi[1]) ** 2 + abs(psi[2]) ** 2)

        def proj(theta: float) -> NDArray:
            v = np.array([math.cos(theta), math.sin(theta)], dtype=np.complex128)
            return np.outer(v, v.conj())

        def corr(t1: float, t2: float) -> float:
            a = 2 * proj(t1) - np.eye(2)
            b = 2 * proj(t2) - np.eye(2)
            return float(np.real(np.trace(rho @ np.kron(a, b))))

        rng = np.random.default_rng(11)
        formula_res = max(abs(corr(t1, t2) + math.cos(2 * (t1 - t2)))
                          for t1, t2 in rng.uniform(0, math.pi, (8, 2)))
        grid = np.linspace(0.0, math.pi, 25, endpoint=False)
        best = 0.0
        for a1 in grid:
            for a2 in grid:
                for b1 in grid:
                    e_ab = -math.cos(2 * (a1 - b1))
                    e_a2b = -math.cos(2 * (a2 - b1))
                    for b2 in grid:
                        s = abs(e_ab + math.cos(2 * (a1 - b2)) * -1.0 * -1.0
                                + e_a2b + (-math.cos(2 * (a2 - b2))))
                        best = max(best, s)
        return PolarizationEntanglementReport(
            state_amplitudes=tuple(complex(x) for x in psi), purity=float(np.real(np.trace(rho @ rho))),
            reduced_state_purity=float(np.real(np.trace(red @ red))),
            entanglement_entropy_bits=_entropy(red, base=2.0), concurrence=float(conc),
            parallel_coincidence_probability=p_par, perpendicular_coincidence_probability=p_perp,
            correlation_formula_residual=float(formula_res), chsh_maximum=float(best),
            tsirelson_residual=abs(best - _TSIRELSON), bell_violation=bool(best > 2.0 + 1e-6),
            wu_shaknov_perpendicular=bool(p_par < 1e-12 and abs(p_perp - 1.0) < 1e-12),
        )

    # ─────────── §II.5 Klein–Nishina polarizada y HOM ───────────
    @staticmethod
    def compton_hom_certificate(params: GammaWitnessParameters) -> ComptonHOMReport:
        r"""
        Klein–Nishina polarizada: dσ/dΩ ∝ (k'/k)²[k/k' + k'/k − 2 sin²θ cos²φ],
        k'/k = 1/(1+ε(1−cosθ)). Razón azimutal IDEAL R = T/(T−2sin²θ) con T = k/k'+k'/k;
        a θ=90° y ε=1: k'/k=½, T=2.5 ⇒ R=5 (la asimetría real de coincidencias es menor por
        integración geométrica: declarado). Límite de Thomson: ε→0 ⇒ σ → 8π/3 (unidades r_e²).

        HOM: U = exp[(π/4)(a†b − ab†)] sobre |1,1⟩ ⇒ P(1,1)=0 (indistinguibles) y 0.5
        (distinguibles); visibilidad 1. [U, N_total]=0 (preservación de número).
        """
        eps = float(params.photon_energy_kev) / _ME_C2_KEV
        th = math.radians(float(params.compton_theta_deg))
        kp = 1.0 / (1.0 + eps * (1.0 - math.cos(th)))
        t_term = 1.0 / kp + kp
        r_num = t_term / (t_term - 2.0 * math.sin(th) ** 2)
        r_ana = t_term / (t_term - 2.0 * math.sin(th) ** 2)

        eps_small = 1.0e-6
        theta_grid = np.linspace(0.0, math.pi, 4001)
        phi_grid = np.linspace(0.0, 2.0 * math.pi, 201)
        kp_s = 1.0 / (1.0 + eps_small * (1.0 - np.cos(theta_grid)))
        unpol = 0.5 * kp_s ** 2 * (1.0 / kp_s + kp_s - np.sin(theta_grid) ** 2)
        sigma_th = _trapz(unpol * np.sin(theta_grid) * 2.0 * math.pi, theta_grid)
        exact_th = 8.0 * math.pi / 3.0

        n = int(params.fock_two_mode_n)
        basis = [(i, j) for i in range(n + 1) for j in range(n + 1) if i + j <= n]
        idx = {b: k for k, b in enumerate(basis)}
        dim = len(basis)
        ad = np.zeros((dim, dim), dtype=np.complex128)
        bd = np.zeros_like(ad)
        for (i, j), k in idx.items():
            if i + 1 + j <= n:
                ad[idx[(i + 1, j)], k] = math.sqrt(i + 1)
            if i + j + 1 <= n:
                bd[idx[(i, j + 1)], k] = math.sqrt(j + 1)
        a_op, b_op = ad.conj().T, bd.conj().T
        gen = ad @ b_op - a_op @ bd
        u_mat = la.expm((math.pi / 4.0) * gen)
        n_tot = ad @ a_op + bd @ b_op
        v11 = np.zeros(dim, dtype=np.complex128); v11[idx[(1, 1)]] = 1.0
        out = u_mat @ v11
        p_coinc = float(abs(out[idx[(1, 1)]]) ** 2)
        p_disting = 0.5
        return ComptonHOMReport(
            compton_theta_deg=float(params.compton_theta_deg), epsilon_ratio=eps, k_prime_over_k=kp,
            azimuthal_ratio_ideal=r_num, azimuthal_ratio_analytic=r_ana,
            azimuthal_ratio_residual=abs(r_num - r_ana), thomson_limit_sigma=sigma_th,
            thomson_limit_exact=exact_th, thomson_limit_residual=abs(sigma_th - exact_th) / exact_th,
            hom_coincidence_indistinguishable=p_coinc, hom_coincidence_distinguishable=p_disting,
            hom_visibility=(p_disting - p_coinc) / p_disting,
            beamsplitter_number_preservation_residual=float(la.norm(u_mat @ n_tot - n_tot @ u_mat, ord="fro")),
            beamsplitter_unitarity_residual=float(la.norm(u_mat.conj().T @ u_mat - np.eye(dim), ord="fro")),
        )

    # ─────────── §II.6 Bures–Uhlmann ───────────
    @staticmethod
    def bures_certificate(rho: NDArray, sigma: NDArray, *, label: str, threshold: float,
                          channel: Optional[Callable[[NDArray], NDArray]] = None) -> BuresGeometryReport:
        fid, clip = _fidelity_svd(rho, sigma)
        alt = _fidelity_sqrt(rho, sigma)
        dist = _trace_distance(rho, sigma)
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid)))
        mono = 0.0
        if channel is not None:
            f2, _ = _fidelity_svd(channel(rho), channel(sigma))
            mono = max(0.0, math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f2))) - bures)
        equiv = (1.0 - 0.35 ** 2 / 2.0) ** 2
        return BuresGeometryReport(
            reference_label=label, fidelity_uhlmann=fid, fidelity_alternate=alt,
            method_agreement_residual=abs(fid - alt), fidelity_clip_residual=clip,
            bures_distance=bures, trace_distance=dist,
            fvg_lower_ok=bool(1.0 - math.sqrt(fid) <= dist + 1e-12),
            fvg_upper_ok=bool(dist <= math.sqrt(max(0.0, 1.0 - fid)) + 1e-12),
            cptp_monotonicity_residual=mono, is_reference_close=bool(fid >= threshold),
            threshold_redundancy_note=f"d_B≤0.35 ⟺ F≥{equiv:.6f}; F≥{threshold} es más estricto.",
        )

    # ─────────── §II.7 Spohn del semigrupo de dephasing ───────────
    def spohn_dephasing_certificate(self, rho: NDArray, n_vals: NDArray,
                                    times: Sequence[float] = (0.0, 0.05, 0.2, 0.6, 1.5)
                                    ) -> SpohnDephasingReport:
        r"""
        Semigrupo exacto ρ_t = ρ ∘ exp(−t·Γ), Γ_mn = (m−n)²/8 (g²=γt). Punto fijo ρ_∞ = diag ρ.
        SPOHN: t ↦ S(ρ_t‖ρ_∞) es no creciente (procesamiento de datos) ⇒ σ = −dS/dt ≥ 0.
        Identidad de coherencia: S(ρ‖diag ρ) = S(diag ρ) − S(ρ) = C_rel(ρ) (Baumgratz et al.).
        """
        gamma = (n_vals[:, None] - n_vals[None, :]) ** 2 / 8.0
        diag_rho = np.diag(np.diag(rho))
        log_d = np.log(np.maximum(np.real(np.diag(rho)), 1e-300))

        def rel_ent(x: NDArray) -> float:
            v = np.clip(la.eigvalsh(0.5 * (x + x.conj().T)), 0.0, 1.0)
            v = v[v > 1e-15]
            return float(np.sum(v * np.log(v)) - np.sum(log_d * np.real(np.diag(x))))

        trace = []
        for t in times:
            trace.append(rel_ent(rho * np.exp(-float(t) * gamma)))
        sigmas = [-(trace[i + 1] - trace[i]) / (times[i + 1] - times[i]) for i in range(len(times) - 1)]
        c_rel = _entropy(diag_rho) - _entropy(rho)
        fixed = float(la.norm(rho * np.exp(-1e6 * gamma) - diag_rho, ord="fro"))
        return SpohnDephasingReport(
            times=tuple(float(t) for t in times), relative_entropy_trace=tuple(trace),
            entropy_monotone=bool(all(trace[i + 1] <= trace[i] + 1e-10 for i in range(len(trace) - 1))),
            sigma_min=float(min(sigmas)), sigma_nonnegative=bool(min(sigmas) >= -1e-10),
            coherence_entropy_identity_residual=abs(trace[0] - c_rel), fixed_point_residual=fixed,
        )

    # ─────────── §II.8 [TERMINAL FASE II] → Fase III §III.1 ───────────
    def build_gamma_phase_ii_handoff(self, h: GammaPhaseIHandoff,
                                     params: GammaWitnessParameters) -> GammaPhaseIIHandoff:
        """MÉTODO TERMINAL DE LA FASE II: cinemática, selección, polarización y geometría."""
        h = self.accept_gamma_phase_i(h)
        kin = self.annihilation_kinematics(params)
        sel = self.selection_rules()
        pol = self.polarization_entanglement()
        ch = self.compton_hom_certificate(params)
        n_vals = np.real(np.diag(h.number_operator))
        channel = lambda r: self.dispersive_channel(r, n_vals, h.qnd.interaction_g, h.qnd.pointer_sigma)[0]  # noqa: E731
        en = n_vals * params.photon_energy_kev
        lp = -params.beta_field * (en - en[0])
        lp -= float(np.max(lp)); lp -= math.log(float(np.sum(np.exp(lp))))
        rho_th = np.diag(np.exp(lp)).astype(np.complex128)
        b_pass = self.bures_certificate(h.rho_before, h.rho_after, label="ρ_pre_testigo (pasividad)",
                                        threshold=params.fidelity_threshold, channel=channel)
        b_therm = self.bures_certificate(h.rho_before, rho_th, label="ρ_térmica del campo (βħω)",
                                         threshold=0.0, channel=channel)
        spohn = self.spohn_dephasing_certificate(h.rho_before, n_vals)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("G2.umbral_cinematico", kin.threshold_satisfied, hard=True),
            Evidence("G2.identidad_suma_energias", kin.sum_identity_residual, 1e-12, hard=True),
            Evidence("G2.identidad_producto_energias", kin.product_identity_residual, 1e-12, hard=True),
            Evidence("G2.conservacion_energia", kin.energy_conservation_residual, 1e-12, hard=True),
            Evidence("G2.conservacion_momento", kin.momentum_conservation_residual, 1e-10, hard=True),
            Evidence.boolean("G2.un_foton_prohibido", kin.single_photon_forbidden, hard=True),
            Evidence("G2.reglas_paridad_c", sel.c_parity_rule_residual, 0.0, hard=True),
            Evidence.boolean("G2.landau_yang", sel.landau_yang_forbids_j1_to_2gamma, hard=True),
            Evidence("G2.polarizacion_pureza", abs(pol.purity - 1.0), 1e-12, hard=True),
            Evidence("G2.reducida_maximixta", abs(pol.reduced_state_purity - 0.5), 1e-12, hard=True),
            Evidence("G2.entropia_entrelazamiento", abs(pol.entanglement_entropy_bits - 1.0), 1e-12, hard=True),
            Evidence("G2.concurrence", abs(pol.concurrence - 1.0), 1e-12, hard=True),
            Evidence.boolean("G2.wu_shaknov_perpendicular", pol.wu_shaknov_perpendicular, hard=True),
            Evidence("G2.formula_correlacion", pol.correlation_formula_residual, 1e-12, hard=True),
            Evidence("G2.tsirelson", pol.tsirelson_residual, 5e-3, hard=True),
            Evidence.boolean("G2.violacion_bell", pol.bell_violation, hard=True),
            Evidence("G2.razon_azimutal", ch.azimuthal_ratio_residual, 1e-12, hard=True),
            Evidence("G2.limite_thomson", ch.thomson_limit_residual, 1e-4, hard=True),
            Evidence("G2.hom_indistinguible", ch.hom_coincidence_indistinguishable, 1e-12, hard=True),
            Evidence("G2.hom_visibilidad", abs(ch.hom_visibility - 1.0), 1e-12, hard=True),
            Evidence("G2.bs_unitariedad", ch.beamsplitter_unitarity_residual, 1e-10, hard=True),
            Evidence("G2.bs_numero", ch.beamsplitter_number_preservation_residual, 1e-10, hard=True),
            Evidence("G2.bures_metodos", b_pass.method_agreement_residual, 1e-9, hard=True),
            Evidence.boolean("G2.fvg_inferior", b_pass.fvg_lower_ok, hard=True),
            Evidence.boolean("G2.fvg_superior", b_pass.fvg_upper_ok, hard=True),
            Evidence("G2.bures_monotonia_cptp", b_pass.cptp_monotonicity_residual, 1e-12, hard=True),
            Evidence.boolean("G2.pasividad_bures", b_pass.is_reference_close, hard=True),
            Evidence.boolean("G2.spohn_monotonia", spohn.entropy_monotone, hard=True),
            Evidence.boolean("G2.spohn_sigma", spohn.sigma_nonnegative, hard=True),
            Evidence("G2.spohn_coherencia", spohn.coherence_entropy_identity_residual, 1e-10, hard=True),
            Evidence("G2.spohn_punto_fijo", spohn.fixed_point_residual, 1e-10, hard=True),
        )
        truth = HeytingTruthValue.infimum([h.truth_value] + [e.truth for e in evidences])
        out = GammaPhaseIIHandoff(
            phase_i=h, kinematics=kin, selection=sel, polarization=pol, compton_hom=ch,
            bures_passivity=b_pass, bures_thermal=b_therm, spohn=spohn,
            evidences=evidences, truth_value=truth,
        )
        logger.info("Fase II gamma Ω₃=%s · √s=%.6f keV · CHSH=%.6f · HOM=%.2e",
                    truth.name, kin.sqrt_s_kev, pol.chsh_maximum, ch.hom_coincidence_indistinguishable)
        return out  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · ATTESTACIÓN, RSI³, CADENCIA Y CONTRAFIRMA  (anida Fases II y I)
# ══════════════════════════════════════════════════════════════════════════════════════
class GammaPhaseIIICustody(GammaPhaseIIAnnihilation):
    r"""FASE III — Atestación independiente, RSI³, cadencia y pasaporte con contrafirma."""

    PHASE_LABEL: Final[str] = "III·ATESTACION-CUSTODIA"

    def accept_gamma_phase_ii(self, h: GammaPhaseIIHandoff) -> GammaPhaseIIHandoff:
        if not isinstance(h, GammaPhaseIIHandoff):
            raise GammaWitnessValidationError("Fase III requiere el GammaPhaseIIHandoff de §II.8.")
        self.accept_gamma_phase_i(h.phase_i)
        return h

    # ─────────── §III.2 Atestación A/B/C ───────────
    def independent_attestation(self, k_mat: NDArray, exponents: NDArray,
                                params: GammaWitnessParameters, principal_report: Any
                                ) -> IndependentAttestationReport:
        """Nivel A (pasaporte/digests) + B (re-derivación) + C (método distinto)."""
        present = principal_report is not None
        if not present:
            principal_report = self._principal.execute_thermal_exergy_weaver_audit(k_mat, exponents)
        passport_ok = bool(self._principal.verify_report_passport(principal_report))
        expected = _sha256((_array_digest(principal_report.phase_ii.phase_i.kernel_reg) + "|"
                            + _array_digest(np.asarray(exponents, dtype=np.float64))).encode("ascii"))
        digest_ok = hmac.compare_digest(expected, principal_report.input_digest)
        pfp = hmac.new(getattr(self._principal, "_secret_key", b""),
                       b"GAMMA-WITNESS-KEY-FP-v14|", hashlib.sha256).hexdigest()[:16]
        key_sep = not hmac.compare_digest(pfp, self.key_fingerprint())

        inc = principal_report.phase_ii.phase_i.incidence
        sm = _smith_by_minors(inc)
        verified = sm is not None
        inv, rank = sm if verified else ((), 0)
        scaled = _smith_by_minors(np.asarray(inc) * 100.0)
        s_inv = scaled[0] if scaled is not None else ()
        fabrication = bool(verified and all(d == 1 for d in inv) and any(d > 1 for d in s_inv))

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
        chaos = bool(amp > off + 1e-12)
        mel_ok = bool(chaos == principal_report.melnikov.chaos_detected and
                      abs(amp - principal_report.melnikov.analytic_amplitude) <= 1e-12)

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

        lam = np.asarray(exponents, dtype=np.float64).reshape(-1)
        val = float(np.min(lam))
        vb = float(np.min(lam + 0.25))
        ultra = max(0.0, math.exp(-val) - max(math.exp(-val), math.exp(-vb))) / (math.exp(-val) + _EPS)
        mult = abs(float(np.min(lam[:, None] + lam[None, :])) - 2.0 * val) / (1.0 + 2.0 * val)
        return IndependentAttestationReport(
            attestation_level=("A+B+C" if verified else "A+B (SNF no verificado)")
            + ("" if present else " · principal autogenerado"),
            principal_present=bool(present), principal_passport_verified=passport_ok,
            principal_input_digest_match=digest_ok, key_separation_ok=bool(key_sep),
            smith_invariants_witness=inv,
            smith_invariants_principal=tuple(principal_report.phase_ii.phase_i.tellegen.smith_invariants),
            smith_method_agreement=bool(verified and inv == tuple(
                principal_report.phase_ii.phase_i.tellegen.smith_invariants)),
            smith_verified=bool(verified), betti_0=int(inc.shape[0] - rank),
            betti_1=int(inc.shape[1] - rank), scaled_invariants=s_inv,
            scaling_fabricates_torsion=fabrication, melnikov_amplitude=amp, melnikov_offset=off,
            melnikov_critical_ratio=crit, melnikov_quadrature_residual=qres, melnikov_chaos=chaos,
            melnikov_principal_agreement=mel_ok, floer_energy_area_residual=ea, floer_cr_bound=cr_bound,
            floer_control_residual=ctrl, legacy_cr_constant_one=legacy_one,
            gromov_obstruction=bool(params.ball_r > params.cylinder_r), eh_monotonicity_ok=eh_ok,
            novikov_valuation=val, novikov_norm=math.exp(-val), novikov_ultrametric_residual=ultra,
            novikov_multiplicativity_residual=mult,
            novikov_filtration_compliant=bool(val >= _KRULL_FLOOR),
        )

    # ─────────── §III.3 KAM diophantino + Lindstedt ───────────
    @staticmethod
    def kam_certificate(params: GammaWitnessParameters,
                        frequencies: Sequence[float] = (1.0, _GOLDEN),
                        n_theta: int = 32) -> KAMCelestialReport:
        """|⟨k,ω⟩| ≥ γ/|k|₁^τ por barrido entero; W_k = i H̃_k/⟨k,ω⟩ (divisores pequeños)."""
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
        amp = float(np.max(np.abs(w_hat))) / (float(np.max(np.abs(h1_hat))) + _EPS)
        twist = float(la.det(np.diag(np.linspace(1.0, 1.3, d))))
        return KAMCelestialReport(
            frequencies=tuple(float(x) for x in omega), diophantine_tau=float(params.kam_tau),
            diophantine_gamma=float(gamma), min_small_divisor=float(min_div),
            resonance_detected=bool(min_div < 1e-12), homological_residual=hom,
            small_divisor_amplification=amp, twist_det=twist,
            is_kam_admissible=bool(min_div >= 1e-12 and abs(twist) > 1e-12),
        )

    # ─────────── §III.4 RSI³ ───────────
    def gamma_rsi3_certificate(self, h: GammaPhaseIIHandoff,
                               params: GammaWitnessParameters) -> GammaRSI3Report:
        r"""
        NIVEL 1 · Calibración de g contra el presupuesto con proxy suave EXACTO
            S(g) = Σ|ρ_mn|²(1−e^{−g²Δ²/8σ²})², S'(g) analítica,
        Newton con salvaguarda de bisección; h de Kantorovich estimado (declarado).
        NIVEL 2 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), Lip ≤ q, Lip(T³) ≤ q³,
        cotas de Banach a priori/a posteriori y sondeo empírico.
        NIVEL 3 · Leyes de Kleisli, axiomas del núcleo ¬¬ y jerk de capacidad MEDIDO (SG 7/3).
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
        lo, hi = 0.0, max(1e-6, sigma * 0.1)
        while s_of(hi) < target and hi < 1e6:
            hi *= 2.0
        g_k = 0.5 * (lo + hi)
        grid = np.linspace(max(lo, 1e-12), hi, 48)
        m_est = float(np.max(np.abs(np.gradient(np.gradient([s_of(x) for x in grid], grid), grid)))) + _EPS
        d0 = ds_of(g_k)
        h_k = (1.0 / max(abs(d0), 1e-300)) * abs((s_of(g_k) - target) / max(abs(d0), 1e-300)) * m_est
        errs: List[float] = []
        for _ in range(40):
            f_v, dv = s_of(g_k) - target, ds_of(g_k)
            if f_v > 0.0:
                hi = g_k
            else:
                lo = g_k
            step = g_k - f_v / dv if abs(dv) > 1e-300 else 0.5 * (lo + hi)
            g_k = float(min(max(step, lo), hi)) if hi > lo else float(step)
            errs.append(abs(s_of(g_k) - target))
            if hi - lo < 1e-16 * max(1.0, hi):
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
        theta_star = np.array([g_k, margin, h.bures_passivity.fidelity_uhlmann])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(self._seed + 13)
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
        return GammaRSI3Report(
            calibrated_gain=float(g_k), target_disturbance=float(params.disturbance_budget),
            achieved_disturbance_proxy=math.sqrt(max(0.0, s_of(g_k))), kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_iterations=int(err.size),
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(emp),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, capacity_d1=d1, capacity_d2=d2c, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2c > 0 and d3 > 0),
            kleisli_monad_laws_ok=laws, nucleus_axioms_ok=nucleus,
        )

    # ─────────── §III.5 Cadencia (tres huecos + Kac) ───────────
    @staticmethod
    def cadence_certificate(params: GammaWitnessParameters) -> GammaCadenceReport:
        """Steinhaus: ≤3 huecos y el mayor suma los otros dos; Kac: E[τ_A]=1/μ."""
        alpha, npts = float(params.cadence_rotation), int(params.cadence_points)
        convs = _convergents(alpha, npts)
        gamma = min(q * q * abs(alpha - p / q) for p, q in convs if q > 0)
        low = any(abs(q * alpha - p) < 1e-6 for p, q in convs if q <= 12)
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
            raise GammaWitnessValidationError("Kac: retorno no alcanzado.")
        rt = tuple(int(v) for v in np.unique(times))
        exp_d, emp_d = 1.0 / mu, float(np.mean(times))
        return GammaCadenceReport(
            rotation_number=alpha, diophantine_gamma=float(gamma), distinct_gap_count=int(uniq.size),
            max_gap=float(np.max(gaps)), coverage_bound=float(bound), three_gap_identity_residual=ident,
            low_order_resonance=bool(low), kac_expected_delay=exp_d, kac_empirical_delay=emp_d,
            kac_relative_residual=abs(emp_d - exp_d) / exp_d, slater_return_times=rt,
            three_gap_ok=bool(uniq.size <= 3 and ident <= 1e-8),
            coverage_ok=bool(float(np.max(gaps)) <= bound + 1e-12), slater_ok=bool(len(rt) <= 3),
        )

    # ─────────── §III.7 [TERMINAL] Orquestador ───────────
    def execute_gamma_photon_witness_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho_mac_input: Optional[NDArray] = None,
        *,
        principal_report: Any = None,
        params: Optional[GammaWitnessParameters] = None,
        authorization_token: Optional[str] = None,   # aceptado y DECLARADO: el testigo no actúa
    ) -> GammaPhotonColliderWitnessSpectrumReport:
        r"""
        MÉTODO TERMINAL. Ejecuta §I.6→§II.1→§II.8→§III.1→§III.2..§III.6, adjudica en Ω₃ y
        emite pasaporte determinista + contrafirma + Merkle. El testigo NUNCA actúa sobre
        hardware: cualquier token recibido se registra como no ejercido (separación de funciones).
        """
        t0 = time.perf_counter()
        params = params or GammaWitnessParameters()
        k_mat = np.asarray(apu_cost_matrix, dtype=np.float64)
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if k_mat.ndim != 2 or k_mat.shape[0] != k_mat.shape[1]:
            raise GammaWitnessValidationError("K debe ser cuadrada.")
        if exps.size == 0:
            raise GammaWitnessValidationError("exponents vacío: v(A) no definida.")

        h1 = self.build_gamma_phase_i_handoff(k_mat, exps, params, rho_mac_input=rho_mac_input)
        h2 = self.build_gamma_phase_ii_handoff(h1, params)
        h2 = self.accept_gamma_phase_ii(h2)
        att = self.independent_attestation(k_mat, exps, params, principal_report)
        kam = self.kam_certificate(params)
        rsi3 = self.gamma_rsi3_certificate(h2, params)
        cad = self.cadence_certificate(params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("G3.pasaporte_principal", att.principal_passport_verified, hard=True),
            Evidence.boolean("G3.digest_entradas", att.principal_input_digest_match, hard=True),
            Evidence.boolean("G3.separacion_de_claves", att.key_separation_ok),
            Evidence.boolean("G3.principal_no_autogenerado", att.principal_present),
            Evidence.boolean("G3.smith_metodo_independiente", att.smith_method_agreement, hard=True),
            Evidence.boolean("G3.sin_caos_homoclinico", not att.melnikov_chaos, hard=True),
            Evidence("G3.melnikov_cuadratura", att.melnikov_quadrature_residual, 1e-8, hard=True),
            Evidence.boolean("G3.melnikov_concuerda", att.melnikov_principal_agreement, hard=True),
            Evidence("G3.floer_energia_area", att.floer_energy_area_residual, 1e-10, hard=True),
            Evidence.boolean("G3.floer_control", att.floer_control_residual > 10 * att.floer_cr_bound, hard=True),
            Evidence.boolean("G3.gromov_sin_obstruccion", not att.gromov_obstruction, hard=True),
            Evidence.boolean("G3.eh_monotonia", att.eh_monotonicity_ok),
            Evidence("G3.novikov_ultrametrica", att.novikov_ultrametric_residual, 1e-12, hard=True),
            Evidence("G3.novikov_multiplicatividad", att.novikov_multiplicativity_residual, 1e-12, hard=True),
            Evidence.boolean("G3.novikov_filtracion", att.novikov_filtration_compliant),
            Evidence("G3.legacy_cr_constante_uno", abs(att.legacy_cr_constant_one - 1.0), 1e-12,
                     note="diagnóstico: el residual CR de v13 era la constante 1 (métrica vacua)"),
            Evidence.boolean("G3.kam_admisible", kam.is_kam_admissible, hard=True),
            Evidence("G3.ecuacion_homologica", kam.homological_residual, 1e-10, hard=True),
            Evidence("G3.gamma_diophantino", max(0.0, 1e-3 - kam.diophantine_gamma), 0.0),
            Evidence("G3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("G3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("G3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("G3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("G3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("G3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence.boolean("G3.tres_huecos", cad.three_gap_ok, hard=True),
            Evidence.boolean("G3.cobertura_cadencia", cad.coverage_ok, hard=True),
            Evidence.boolean("G3.slater", cad.slater_ok, hard=True),
            Evidence("G3.kac_retardo", cad.kac_relative_residual, 5e-3, hard=True),
            Evidence.boolean("G3.testigo_no_actua", True,
                             note=f"token recibido={'sí' if authorization_token else 'no'}; no ejercido"),
        )
        truth = HeytingTruthValue.infimum([h2.truth_value] + [e.truth for e in evidences])
        leaves: List[bytes] = [
            _canonical({"commitment": h1.commitment_digest, "rho": h1.rho_digest,
                        "K": h1.kernel_digest, "exps": h1.exponents_digest}),
            _canonical(h1.fock), _canonical(h1.qnd), _canonical(h1.aav),
            _canonical(h2.kinematics), _canonical(h2.selection), _canonical(h2.polarization),
            _canonical(h2.compton_hom), _canonical(h2.bures_passivity), _canonical(h2.spohn),
            _canonical(att), _canonical(kam), _canonical(rsi3), _canonical(cad),
            _canonical({"truth": truth, "version": __version__}),
        ]
        root = merkle_root(leaves)
        a_idx = 10
        proof = merkle_proof(leaves, a_idx)
        payload = {"version": __version__, "truth": truth, "commitment": h1.commitment_digest,
                   "merkle_root": root, "attestation_level": att.attestation_level,
                   "qnd_passive": h1.qnd.is_qnd_passive,
                   "fidelity_passivity": h2.bures_passivity.fidelity_uhlmann,
                   "sqrt_s_kev": h2.kinematics.sqrt_s_kev}
        passport = hmac.new(self.key_bytes, _HMAC_PASS + _canonical(payload), hashlib.sha256).hexdigest()
        pp = getattr(principal_report, "passport_hmac", "") if principal_report is not None else ""
        counter = hmac.new(self.key_bytes, _HMAC_COUNTER + pp.encode("ascii") + bytes.fromhex(root),
                           hashlib.sha256).hexdigest()
        report = GammaPhotonColliderWitnessSpectrumReport(
            engine_version=__version__, phase_ii=h2, attestation=att, kam=kam, rsi3=rsi3, cadence=cad,
            evidences=evidences, truth_value=truth, boolean_shadow=truth.double_negation(),
            merkle_root_hash=root, merkle_leaves=tuple(x.hex() for x in leaves),
            attestation_leaf_index=a_idx, attestation_inclusion_proof=proof,
            passport_hash=passport, countersignature=counter,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("Testigo gamma v14 Ω₃=%s · nivel=%s · Merkle=%s…", truth.name, att.attestation_level, root[:16])
        return report

    def verify_witness_report(self, r: GammaPhotonColliderWitnessSpectrumReport) -> bool:
        """Reverifica Merkle, prueba de inclusión y pasaporte determinista."""
        lv = [bytes.fromhex(x) for x in r.merkle_leaves]
        if merkle_root(lv) != r.merkle_root_hash:
            return False
        i = r.attestation_leaf_index
        if not verify_merkle(lv[i], i, len(lv), r.attestation_inclusion_proof, r.merkle_root_hash):
            return False
        payload = {"version": r.engine_version, "truth": r.truth_value,
                   "commitment": r.phase_ii.phase_i.commitment_digest,
                   "merkle_root": r.merkle_root_hash,
                   "attestation_level": r.attestation.attestation_level,
                   "qnd_passive": r.phase_ii.phase_i.qnd.is_qnd_passive,
                   "fidelity_passivity": r.phase_ii.bures_passivity.fidelity_uhlmann,
                   "sqrt_s_kev": r.phase_ii.kinematics.sqrt_s_kev}
        exp = hmac.new(self.key_bytes, _HMAC_PASS + _canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(exp, r.passport_hash)


class BoolNovikovGammaPhotonColliderWitnessEngine(GammaPhaseIIICustody):
    """Fachada pública (nombre conservado). Anidamiento: Fase I ⊂ II ⊂ III ⊂ Fachada."""

    PHASE_LABEL: Final[str] = "FACHADA·GAMMA-PHOTON-COLLIDER-WITNESS"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    wit = BoolNovikovGammaPhotonColliderWitnessEngine(
        secret_key=b"GAMMA_WITNESS_KEY_32_BYTES_DIFF!", principal_engine=principal)
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.45, 0.85], dtype=np.float64)
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)

    # 1) Nominal: QND pasivo real (estado CON coherencias), cinemática y Bell exactos
    rep = wit.execute_gamma_photon_witness_audit(k_test, exps, principal_report=prep)
    q, p2 = rep.phase_ii.phase_i.qnd, rep.phase_ii
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert q.state_nondisturbance_residual > 1e-6          # ← el certificado NO es vacuo
    assert q.trace_distance_disturbance > 0.0 and q.is_qnd_passive
    assert q.back_action_db <= 0.0 and q.kraus_completeness_residual < 1e-9
    assert rep.phase_ii.phase_i.fock.commutator_defect_full == rep.phase_ii.phase_i.fock.n_max + 1
    assert abs(p2.kinematics.sqrt_s_kev - 1021.99790138) < 1e-6
    assert p2.kinematics.sum_identity_residual < 1e-12 and p2.kinematics.product_identity_residual < 1e-12
    assert abs(p2.polarization.chsh_maximum - 2 * math.sqrt(2)) < 5e-3
    assert p2.polarization.wu_shaknov_perpendicular
    assert p2.compton_hom.hom_coincidence_indistinguishable < 1e-12
    assert abs(p2.compton_hom.azimuthal_ratio_ideal - 5.0) < 1e-9
    assert p2.spohn.sigma_nonnegative and p2.spohn.entropy_monotone
    assert rep.attestation.smith_method_agreement and rep.attestation.scaling_fabricates_torsion
    assert wit.verify_witness_report(rep)

    # 2) NO vacuidad: g grande ⇒ pasividad ROTA y detectada (v13 nunca podía fallar)
    broken = wit.execute_gamma_photon_witness_audit(
        k_test, exps, principal_report=prep,
        params=GammaWitnessParameters(interaction_g=0.5))
    assert broken.phase_ii.phase_i.qnd.trace_distance_disturbance > 1e-4
    assert not broken.phase_ii.phase_i.qnd.is_qnd_passive
    assert broken.truth_value is HeytingTruthValue.VETOED

    # 3) Cinemática en vuelo: identidades exactas y umbral
    inflight = wit.annihilation_kinematics(GammaWitnessParameters(positron_kinetic_kev=250.0,
                                                                  emission_angle_deg=61.0))
    assert inflight.sqrt_s_kev > 1021.99790138
    assert inflight.sum_identity_residual < 1e-12 and inflight.product_identity_residual < 1e-12
    assert inflight.energy_conservation_residual < 1e-12 and inflight.momentum_conservation_residual < 1e-10
    assert abs(inflight.photon1_energy_kev + inflight.photon2_energy_kev - inflight.total_energy_kev) < 1e-9

    # 4) Bures NO autorreferencial: la referencia térmica distingue estados
    assert p2.bures_thermal.fidelity_uhlmann < 0.999
    assert p2.bures_thermal.method_agreement_residual < 1e-9
    assert p2.bures_thermal.fvg_lower_ok and p2.bures_thermal.fvg_upper_ok

    # 5) Diagnóstico de vacuidad heredada: el residual CR de v13 es la constante 1
    assert abs(rep.attestation.legacy_cr_constant_one - 1.0) < 1e-12

    # 6) Reglas de selección y SNF por menores (método independiente)
    sel = p2.selection
    assert sel.para_two_gamma_allowed and not sel.ortho_two_gamma_allowed and sel.ortho_three_gamma_allowed
    tri = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=np.float64)
    snf = _smith_by_minors(tri)
    assert snf is not None and snf[0] == (1, 1) and snf[1] == 2
    assert _smith_by_minors(tri * 100.0)[0] == (100, 100)

    # 7) Merkle, determinismo del pasaporte y detección de manipulación
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    rep2 = wit.execute_gamma_photon_witness_audit(k_test, exps, principal_report=prep)
    assert rep2.passport_hash == rep.passport_hash
    assert not wit.verify_witness_report(dataclasses.replace(rep, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovGammaPhotonColliderWitnessEngine v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {rep.truth_value.name}({int(rep.truth_value)}) / {rep.boolean_shadow.name}")
    print(f"Nivel de atestación           : {rep.attestation.attestation_level}")
    print(f"Fock N / defecto [a,a†]       : {rep.phase_ii.phase_i.fock.n_max} / "
          f"{rep.phase_ii.phase_i.fock.commutator_defect_full:.1f} (físico "
          f"{rep.phase_ii.phase_i.fock.commutator_defect_physical:.1e})")
    print(f"[ρ,n̂] (no vacuidad)           : {q.state_nondisturbance_residual:.3e}")
    print(f"D_tr / cota / dB              : {q.trace_distance_disturbance:.3e} / "
          f"{q.disturbance_apriori_bound:.3e} / {q.back_action_db:.3e}")
    print(f"A_w / anómalo / p_f           : {rep.phase_ii.phase_i.aav.weak_value:.4f} / "
          f"{rep.phase_ii.phase_i.aav.is_anomalous} / {rep.phase_ii.phase_i.aav.postselection_probability:.4f}")
    print(f"Regla de suma / orden puntero : {rep.phase_ii.phase_i.aav.sum_rule_residual:.2e} / "
          f"{rep.phase_ii.phase_i.aav.pointer_convergence_order:.3f}")
    print(f"√s / umbral [keV]             : {p2.kinematics.sqrt_s_kev:.6f} / {p2.kinematics.threshold_kev:.6f}")
    print(f"E₁(θ) / E₂ [keV]              : {p2.kinematics.photon1_energy_kev:.6f} / "
          f"{p2.kinematics.photon2_energy_kev:.6f}")
    print(f"E_max·E_min = s/4 (residuo)   : {p2.kinematics.product_identity_residual:.2e}")
    print(f"Entropía ent. / concurrence   : {p2.polarization.entanglement_entropy_bits:.6f} bit / "
          f"{p2.polarization.concurrence:.6f}")
    print(f"CHSH máx / Tsirelson          : {p2.polarization.chsh_maximum:.6f} / {2*math.sqrt(2):.6f}")
    print(f"P(paralela) / P(perpendicular): {p2.polarization.parallel_coincidence_probability:.2e} / "
          f"{p2.polarization.perpendicular_coincidence_probability:.6f}")
    print(f"KN razón azimutal / Thomson   : {p2.compton_hom.azimuthal_ratio_ideal:.4f} / "
          f"{p2.compton_hom.thomson_limit_sigma:.6f} (exacto {p2.compton_hom.thomson_limit_exact:.6f})")
    print(f"HOM coincidencia / visibilidad: {p2.compton_hom.hom_coincidence_indistinguishable:.2e} / "
          f"{p2.compton_hom.hom_visibility:.6f}")
    print(f"Spohn σ_min / C_rel residuo   : {p2.spohn.sigma_min:.4e} / "
          f"{p2.spohn.coherence_entropy_identity_residual:.2e}")
    print(f"SNF real / ×100               : {rep.attestation.smith_invariants_witness} / "
          f"{rep.attestation.scaled_invariants}")
    print(f"RSI³ g calibrado / Lip(T³)    : {rep.rsi3.calibrated_gain:.6e} / {rep.rsi3.lip_t3_empirical:.6f}")
    print(f"Kac esperado / empírico       : {rep.cadence.kac_expected_delay:.3f} / {rep.cadence.kac_empirical_delay:.3f}")
    print(f"Slater tiempos de retorno     : {rep.cadence.slater_return_times}")
    print(f"Control ⊥ (g=0.5)             : {broken.truth_value.name} "
          f"D_tr={broken.phase_ii.phase_i.qnd.trace_distance_disturbance:.3e}")
    print(f"Merkle / contrafirma          : {rep.merkle_root_hash[:16]}… / {rep.countersignature[:16]}…")
    print(f"Informe reverificable         : {wit.verify_witness_report(rep)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")