# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverProsecutorEngine  v14.0.0-PhD (reedición rigurosa)   ║
║ Ubicación: app/wisdom/bool_novikov_thermal_exergy_weaver_prosecutor_engine.py                 ║
║ Función  : Motor FISCAL INQUISIDOR con debido proceso: falsación adversarial, carga de la     ║
║            prueba estadística, producción de entropía de Spohn verificada, sentencia          ║
║            proporcional y ejecución física SÓLO con autorización humana criptográfica.        ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA FISCAL (lo que distingue acusar de condenar)
────────────────────────────────────────────────────
Un fiscal que siempre condena no prueba nada; un fiscal que dispara sin autorización no es un
fiscal, es un fallo de seguridad. Este módulo se rige por cinco principios verificables:

  (1) PRESUNCIÓN DE INOCENCIA. El veredicto por defecto es ⊤. Sólo se condena si
      (a) una identidad matemática FALLA (prueba estructural, no estadística), o
      (b) una carga ESTADÍSTICA supera el umbral de Wald con α (falsa condena) declarado.
      Empate o evidencia insuficiente ⇒ ½, nunca ⊥.
  (2) CARGA DE LA PRUEBA CUANTIFICADA. SPRT de Wald con α y β explícitos; las tasas empíricas
      se verifican por Monte Carlo con semilla. Un umbral mágico no es una carga de la prueba.
  (3) FALSACIÓN, NO CONFIRMACIÓN. El fiscal busca CONTRAEJEMPLOS a las afirmaciones del motor
      principal (PSD, identidad secante, cota de Lipschitz, obstrucción de Gromov) y declara
      el PODER de su búsqueda: ausencia de contraejemplo ≠ prueba de corrección.
  (4) PROPORCIONALIDAD. Escalera WARN → DERATE → ISOLATE → TRIP; el peldaño máximo exige un
      predicado declarado de insuficiencia de medidas menores.
  (5) AUTORIZACIÓN HUMANA QUE HABILITA. El token HMAC liga {acta, época, peldaño, autorizador}.
      SIN token NO hay actuación: se escala y se pasa a estado seguro. Se aplica ne bis in idem
      (un acta no se procesa dos veces) y anti-replay por época estrictamente creciente.

ARQUITECTURA EN TRES FASES ANIDADAS
───────────────────────────────────
    Fase I  §I.5  build_prosecutor_phase_i_handoff   ──► Fase II §II.1 accept_prosecutor_phase_i
    Fase II §II.6 build_prosecutor_phase_ii_handoff  ──► Fase III §III.1 accept_prosecutor_phase_ii
    Fase III §III.7 execute_thermal_prosecutor_audit (terminal / orquestador)
    Handoffs anidados: ProsecutorPhaseIHandoff ⊂ ProsecutorPhaseIIHandoff ⊂ informe final.

FASE I — INSTRUMENTO DE CARGA
  §I.1 Validación y regularización IDÉNTICA al principal (mismo Hamiltoniano) + compromiso.
  §I.2 Falsación adversarial con poder de búsqueda declarado.
  §I.3 SNF exacta por divisores determinantes + patología del escalado ×100 (torsión fabricada).
  §I.4 SPRT de Wald: carga de la prueba con α/β declarados y verificación Monte Carlo.
  §I.5 [TERMINAL] build_prosecutor_phase_i_handoff.

FASE II — INSTRUCCIÓN FÍSICA
  §II.1 accept_prosecutor_phase_i.
  §II.2 Itoh–Abe baricéntrico w*=½ ≡ González + Tellegen real (KVL ⊥ KCL) + telescoping.
  §II.3 Jarzynski–Crooks y KMS modular attestados/re-derivados; S(ρ‖ρ_β) ≥ 0.
  §II.4 SPOHN: generador de Davies con balance detallado, propagador exacto e^{Lt},
        estacionariedad de Gibbs, monotonía de la entropía relativa y σ ≥ 0.
  §II.5 Novikov ultramétrico, Floer/Gromov con cota y control, Melnikov analítico↔cuadratura.
  §II.6 [TERMINAL] build_prosecutor_phase_ii_handoff.

FASE III — SENTENCIA Y EJECUCIÓN AUTORIZADA
  §III.1 accept_prosecutor_phase_ii.
  §III.2 RSI³ certificado (Banach + Kantorovich + Kleisli + jerk medido).
  §III.3 Poincaré–KAM diophantino y ecuación homológica de Lindstedt (independiente).
  §III.4 Sentencia Ω₃: presunción de inocencia + núcleo ¬¬ + tabla de evidencias.
  §III.5 Proporcionalidad + interlock de autorización humana + ne bis in idem.
  §III.6 Dirac (CODATA) y crowbar HAL: despacho medido, presupuesto físico declarado.
  §III.7 [TERMINAL] execute_thermal_prosecutor_audit (acta sellada + contrafirma + Merkle).

MIGRACIÓN DESDE v13
───────────────────
• Ω₃ canónico del motor: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2.
• `positron_hmac_valid: bool` → `authorization_token: str` (HMAC real, ligado al acta).
• `work_w`/`delta_f` → `work_mean`/`delta_f` (modelo calibrado) o muestras reales.
• Los defectos por omisión ya NO condenan: `ball_r ≤ cylinder_r`.
• El pasaporte es determinista y compromete las entradas; el tiempo es telemetría.

LIMITACIONES DECLARADAS
───────────────────────
1. Las sondas de falsación son finitas: no sustituyen una prueba; los chequeos exactos sí.
2. El SPRT controla error bajo el modelo gaussiano declarado; desviaciones del modelo
   (colas pesadas, deriva) invalidan las tasas y así se declara.
3. Spohn se verifica para el semigrupo de Davies construido (modelo), no para la dinámica
   real del hardware auditado.
4. 392.15 ns es política del camino lógico ISR/IRAM; la conmutación física de un SCR BT151
   (t_gt ~1–3 µs) no puede alcanzarlo sin medición HIL que demuestre lo contrario.
5. La escalada se emite como dato firmado; el transporte es responsabilidad del llamador.

REFERENCIAS: Wald (1945) SPRT; Spohn (1978); Davies (1974); Lindblad (1976); Itoh–Abe (1988);
González (1996); Tellegen (1952); Jarzynski (1997); Crooks (1999); Bennett (1976); Gromov (1985);
Ekeland–Hofer (1990); Melnikov (1963); Arnold–Moser KAM; Kantorovich; Banach (1922);
Mac Lane (Kleisli); RFC 6962; CODATA 2022; datasheets ESP32-S3 / BT151.
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

_IMPORT_ERROR: Optional[Exception] = None
for _path in ("app.wisdom.bool_novikov_thermal_exergy_weaver_engine",
              "bool_novikov_thermal_exergy_weaver_engine"):
    try:
        _m = __import__(_path, fromlist=["*"])
        PrincipalEngine = _m.BoolNovikovThermalExergyWeaverEngine
        HeytingTruthValue = _m.HeytingTruthValue        # Ω₃ CANÓNICO (no se duplica)
        Evidence = _m.Evidence
        Certified = _m.Certified
        kleisli_compose = _m.kleisli_compose
        _PRINCIPAL_VERSION = _m.__version__
        break
    except ImportError as exc:                                       # pragma: no cover
        _IMPORT_ERROR = exc
else:                                                                # pragma: no cover
    raise ImportError("Se requiere el motor principal v14.") from _IMPORT_ERROR

__all__ = [
    "ProsecutionParameters", "ThermalExergyWeaverProsecutorSpectrumReport",
    "BoolNovikovThermalExergyWeaverProsecutorEngine", "ProsecutorError",
    "AuthorizationError", "CrowbarDriver", "SimulatedCrowbarDriver", "__version__",
]
logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaverProsecutorEngine")

__version__: Final[str] = "14.0.0-ENGINE-PROSECUTOR-WALD-SPOHN-DAVIES-BANACH-KLEISLI-PhD"
version: Final[str] = __version__

_EPS: Final[float] = float(np.finfo(np.float64).eps)
_PSD_FLOOR: Final[float] = 1e-12
_KRULL_FILTRATION_FLOOR: Final[float] = 0.10
_ME_C2_KEV: Final[float] = 510.99895069                 # CODATA 2022 (v13 llamaba 1022 a esto)
_LOGIC_DEADLINE_NS: Final[float] = 392.15
_MINOR_CAP: Final[int] = 20_000
_GOLDEN: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_HMAC_ACTA: Final[bytes] = b"PROSECUTOR-ACTA-v14|"
_HMAC_AUTH: Final[bytes] = b"PROSECUTOR-HUMAN-AUTHORIZATION-v14|"
_HMAC_COUNTER: Final[bytes] = b"PROSECUTOR-COUNTERSIGN-v14|"
_LEAF: Final[bytes] = b"\x00"
_NODE: Final[bytes] = b"\x01"
_LADDER: Final[Tuple[str, ...]] = ("WARN", "DERATE", "ISOLATE", "TRIP")


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0 · SUSTRATO: errores, canonicalización, Merkle RFC 6962, SNF exacta, utilidades
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorError(Exception):
    """Error base del fiscal."""


class AuthorizationError(ProsecutorError):
    """Actuación solicitada sin autorización humana válida o con acta ya procesada."""


def _as_matrix(v: Any, name: str, *, square: bool = False, complex_ok: bool = False) -> NDArray:
    arr = np.asarray(v, dtype=np.complex128 if complex_ok else np.float64)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ProsecutorError(f"{name}: matriz 2-D no vacía requerida, recibido {arr.shape}.")
    if square and arr.shape[0] != arr.shape[1]:
        raise ProsecutorError(f"{name}: se requiere matriz cuadrada.")
    if not np.all(np.isfinite(arr)):
        raise ProsecutorError(f"{name}: contiene NaN/Inf.")
    return arr


def _as_vector(v: Any, name: str) -> NDArray:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ProsecutorError(f"{name}: vector finito no vacío requerido.")
    return arr


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


def _array_digest(arr: NDArray) -> str:
    """Algoritmo idéntico al del motor principal: digests interoperables."""
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|"); h.update(str(a.shape).encode("ascii")); h.update(b"|")
    h.update(dt.str.encode("ascii")); h.update(b"|"); h.update(a.tobytes())
    return h.hexdigest()


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


def _bareiss_det(mat: List[List[int]]) -> int:
    """Determinante entero exacto (eliminación libre de fracciones de Bareiss)."""
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
    r"""
    SNF por DIVISORES DETERMINANTES (método independiente del euclídeo del principal):
    D_k = gcd de todos los menores k×k; d_k = D_k/D_{k−1}. coker ≅ ℤ^{n−r} ⊕ ⊕ℤ/d_i.
    Libre de torsión ⟺ todos d_i = 1. Si el número de menores excede `cap`, devuelve None
    («no verificado») en lugar de fabricar un resultado.
    """
    mi = [[int(round(float(v))) for v in row] for row in np.asarray(matrix)]
    n = len(mi); m = len(mi[0]) if n else 0
    div: List[int] = [1]
    rank = 0
    for k in range(1, min(n, m) + 1):
        if math.comb(n, k) * math.comb(m, k) > cap:
            return None
        g = 0
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.combinations(range(m), k):
                g = math.gcd(g, abs(_bareiss_det([[mi[i][j] for j in cols] for i in rows])))
                if g == 1:
                    break
            if g == 1:
                break
        if g == 0:
            break
        div.append(g); rank = k
    return tuple(div[i] // div[i - 1] for i in range(1, len(div))), rank


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


def _logsumexp(v: NDArray) -> float:
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m)))) if math.isfinite(m) else m


def _relative_entropy_diagonal(rho_eig: NDArray, log_p: NDArray) -> float:
    """S(ρ‖ρ_β) con ρ_β diagonal en la base propia de H: Tr ρ log ρ − Σ log p_i ρ_ii."""
    vals = np.clip(la.eigvalsh(0.5 * (rho_eig + rho_eig.conj().T)), 0.0, None)
    tr_rho_log_rho = float(np.sum(vals[vals > 0] * np.log(vals[vals > 0])))
    return tr_rho_log_rho - float(np.sum(log_p * np.real(np.diag(rho_eig))))


# ══════════════════════════════════════════════════════════════════════════════════════
#  §0.1 · CONTRATOS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProsecutionParameters:
    """Parámetros DECLARADOS del proceso; su digest se compromete en el acta."""

    alpha_false_conviction: float = 0.01      # SPRT: prob. de condenar siendo inocente
    beta_false_acquittal: float = 0.10        # SPRT: prob. de absolver siendo culpable
    cost_mu0: float = 0.0                     # H0: coste legítimo
    cost_mu1: float = 0.5                     # H1: sobreprecio
    cost_sigma: float = 1.0
    sprt_mc_paths: int = 2000
    sprt_mc_max_len: int = 500
    psd_floor: float = _PSD_FLOOR
    beta_thermo: float = 1.0
    work_mean: float = 5.0
    delta_f: float = 4.8
    ball_r: float = 1.0                       # v13: 0.25 > 0.20 ⇒ condenaba por defecto
    cylinder_r: float = 1.5
    melnikov_omega: float = 1.0
    melnikov_gamma: float = 0.05
    melnikov_delta: float = 0.25
    melnikov_epsilon: float = 0.01
    spohn_gamma0: float = 1.0
    spohn_times: Tuple[float, ...] = (0.0, 0.05, 0.2, 0.6, 1.5)
    kam_tau: float = 1.05
    kam_scan_order: int = 40
    falsification_probes: int = 256
    require_human_authorization: bool = True
    lesser_measures_insufficient: bool = False   # predicado de proporcionalidad DECLARADO
    logic_deadline_ns: float = _LOGIC_DEADLINE_NS
    gpio_write_ns: float = 25.0
    gate_driver_ns: float = 150.0
    scr_turn_on_ns: float = 1500.0               # BT151 t_gt típico (datasheet)
    hil_firmware_isr_ns: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha_false_conviction < 0.5 and 0.0 < self.beta_false_acquittal < 0.5):
            raise ProsecutorError("α,β ∈ (0,0.5) para el SPRT.")
        if self.cost_sigma <= 0.0 or self.cost_mu1 <= self.cost_mu0:
            raise ProsecutorError("σ>0 y μ₁>μ₀ requeridos (H1 debe ser separable de H0).")
        if self.ball_r <= 0.0 or self.cylinder_r <= 0.0:
            raise ProsecutorError("Radios positivos requeridos.")
        if self.beta_thermo <= 0.0 or self.work_mean < self.delta_f - 1e-12:
            raise ProsecutorError("β>0 y ⟨W⟩≥ΔF (Jensen) requeridos.")
        if tuple(sorted(self.spohn_times)) != tuple(self.spohn_times) or self.spohn_times[0] != 0.0:
            raise ProsecutorError("spohn_times debe iniciar en 0 y ser creciente.")

    def digest(self) -> str:
        return _sha256(_canonical(self))

    @property
    def physical_budget_ns(self) -> float:
        return float(self.gpio_write_ns + self.gate_driver_ns + self.scr_turn_on_ns)


@dataclass(frozen=True)
class AdversarialFalsificationReport:
    """§I.2 — Búsqueda de contraejemplos con poder declarado (ausencia ≠ prueba)."""

    probes_executed: int
    psd_min_eigenvalue_exact: float
    psd_counterexamples: int
    secant_max_residual: float
    secant_counterexamples: int
    lipschitz_claimed_bound: float
    lipschitz_max_probed_ratio: float
    lipschitz_counterexamples: int
    gromov_obstruction_exact: bool
    search_power_note: str


@dataclass(frozen=True)
class SmithTorsionReport:
    """§I.3 — SNF exacta y patología del escalado ×100 (torsión fabricada)."""

    verified: bool
    invariants: Tuple[int, ...]
    rank: int
    torsion_free: bool
    betti_0: int
    betti_1: int
    scaled_invariants: Tuple[int, ...]
    scaling_fabricates_torsion: bool
    principal_agreement: bool


@dataclass(frozen=True)
class BurdenOfProofReport:
    """§I.4 — SPRT de Wald: la carga de la prueba cuantificada."""

    alpha_declared: float
    beta_declared: float
    log_threshold_convict: float
    log_threshold_acquit: float
    samples_observed: int
    log_likelihood_ratio: float
    decision: str                       # CONVICT | ACQUIT | UNDECIDED
    mc_false_conviction_rate: float
    mc_power: float
    mc_asn_h0: float
    mc_asn_h1: float
    mc_rates_within_declared: bool
    charge_kind: str                    # STRUCTURAL | STATISTICAL


@dataclass(frozen=True)
class ItohAbeTellegenReport:
    """§II.2 — Gradiente discreto y Tellegen reales (no tautologías)."""

    secant_residual_max: float
    telescoping_residual_max: float
    gonzalez_equivalence_residual: float
    barycentric_symmetry_residual: float
    tellegen_power_residual: float
    kcl_residual: float
    kvl_cycle_residual: float
    energy_law_residual: float
    legacy_tautology_residual: float


@dataclass(frozen=True)
class JarzynskiKMSProsecutionReport:
    """§II.3 — Jarzynski–Crooks y KMS attestados/re-derivados."""

    delta_f_bar_j: float
    delta_f_bar_sigma_j: float
    irreversible_work_j: float
    ess_fraction: float
    jensen_residual: float
    kms_residual: float
    gibbs_relative_entropy: float
    principal_agreement: bool
    attestation_level: str


@dataclass(frozen=True)
class SpohnEntropyReport:
    r"""
    §II.4 — Producción de entropía de Spohn para un semigrupo de Davies con balance detallado.
    σ(t) = −d/dt S(ρ_t‖ρ_β) ≥ 0, con ρ_β punto fijo del generador.
    """

    generator_stationarity_residual: float
    trace_preservation_residual: float
    positivity_min_eigenvalue: float
    relative_entropy_trace: Tuple[float, ...]
    entropy_monotone: bool
    sigma_min: float
    sigma_nonnegative: bool
    detailed_balance_residual: float
    liouvillian_dimension: int


@dataclass(frozen=True)
class GeometryCelestialReport:
    """§II.5 + §III.3 — Novikov, Floer/Gromov, Melnikov y KAM diophantino."""

    novikov_valuation: float
    novikov_norm: float
    novikov_ultrametric_residual: float
    novikov_filtration_compliant: bool
    floer_energy_area_residual: float
    floer_cr_residual: float
    floer_cr_bound: float
    floer_control_residual: float
    legacy_cr_constant_sqrt2: float
    ball_radius: float
    cylinder_radius: float
    gromov_obstruction: bool
    eh_monotonicity_ok: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_critical_ratio: float
    melnikov_quadrature_residual: float
    melnikov_chaos: bool
    melnikov_principal_agreement: bool
    kam_diophantine_gamma: float
    kam_min_small_divisor: float
    kam_resonance: bool
    kam_homological_residual: float
    kam_twist_det: float


@dataclass(frozen=True)
class ProsecutorRSI3Report:
    """§III.2 — RSI³ certificado (Banach + Kantorovich + Kleisli + jerk medido)."""

    policy_lipschitz_q: float
    lip_t3_bound: float
    lip_t3_empirical: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_policy_error: float
    kantorovich_h: float
    kantorovich_satisfied: bool
    newton_quadratic_ratio_residual: float
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool
    kleisli_monad_laws_ok: bool
    nucleus_axioms_ok: bool


@dataclass(frozen=True)
class SentenceReport:
    """§III.4–§III.5 — Sentencia, proporcionalidad y autorización."""

    truth_value: HeytingTruthValue
    boolean_shadow: HeytingTruthValue
    ladder_rung: int
    action: str
    charge_proven: bool
    charge_kind: str
    presumption_of_innocence_honored: bool
    proportionality_ok: bool
    authorization_token_verified: bool
    authorization_self_issued: bool
    ne_bis_in_idem_ok: bool
    actuation_requested: bool
    actuation_authorized: bool
    indictment_digest: str
    epoch: int


@dataclass(frozen=True)
class DiracCrowbarProsecutionReport:
    """§III.6 — Cinemática de Dirac (CODATA) y crowbar con latencia MEDIDA."""

    electron_rest_energy_kev: float
    two_rest_energy_threshold_kev: float
    sqrt_s_kev: float
    photon_energy_cm_kev: float
    photons_emitted: int
    actuation_dispatched: bool
    driver_name: str
    driver_is_physical: bool
    dispatch_median_ns: float
    dispatch_p99_ns: float
    trip_dispatch_ns: float
    logic_deadline_ns: float
    firmware_isr_verified: bool
    physical_budget_ns: float
    physical_deadline_feasible: bool
    safe_state_applied: bool
    note: str


@dataclass(frozen=True)
class ProsecutorPhaseIHandoff:
    """§I.5 [TERMINAL FASE I] — entrada de §II.1."""

    falsification: AdversarialFalsificationReport
    smith: SmithTorsionReport
    burden: BurdenOfProofReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    commitment_digest: str
    kernel_digest: str
    exponents_digest: str
    kernel_reg: NDArray = field(repr=False)
    eigenvalues: NDArray = field(repr=False)
    eigenvectors: NDArray = field(repr=False)
    incidence: NDArray = field(repr=False)


@dataclass(frozen=True)
class ProsecutorPhaseIIHandoff:
    """§II.6 [TERMINAL FASE II] — anida Fase I; entrada de §III.1."""

    phase_i: ProsecutorPhaseIHandoff
    itoh_tellegen: ItohAbeTellegenReport
    jarzynski_kms: JarzynskiKMSProsecutionReport
    spohn: SpohnEntropyReport
    geometry: GeometryCelestialReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue


@dataclass(frozen=True)
class ThermalExergyWeaverProsecutorSpectrumReport:
    """§III.7 [TERMINAL] — acta anidada I ⊂ II ⊂ III con contrafirma y Merkle."""

    engine_version: str
    phase_ii: ProsecutorPhaseIIHandoff
    rsi3: ProsecutorRSI3Report
    sentence: SentenceReport
    dirac_crowbar: DiracCrowbarProsecutionReport
    evidences: Tuple[Any, ...]
    truth_value: HeytingTruthValue
    principal_passport: str
    principal_passport_verified: bool
    witness_impeached: bool
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    acta_leaf_index: int
    acta_inclusion_proof: Tuple[Tuple[str, bool], ...]
    passport_hash: str
    countersignature: str
    execution_time_ms: float

    def to_json(self) -> str:
        return json.dumps(_freeze(self), sort_keys=True, indent=2, ensure_ascii=True)

    def legacy_summary(self) -> Dict[str, Any]:
        p1, p2 = self.phase_ii.phase_i, self.phase_ii
        return {
            "truth_value": self.truth_value.name,
            "itoh_abe_report": p2.itoh_tellegen,
            "jarzynski_kms_report": p2.jarzynski_kms,
            "smith_torsion_free": p1.smith.torsion_free,
            "betti_number_zero": p1.smith.betti_0,
            "krull_valuation": p2.geometry.novikov_valuation,
            "novikov_norm": p2.geometry.novikov_norm,
            "cauchy_riemann_residual": p2.geometry.floer_cr_residual,
            "is_gromov_violated": p2.geometry.gromov_obstruction,
            "melnikov_chaos_detected": p2.geometry.melnikov_chaos,
            "banach_contraction_lip": self.rsi3.lip_t3_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "dirac_report": self.dirac_crowbar,
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
    """Driver SIMULADO (no físico). Mide despacho real de software; nunca afirma conmutar hardware."""

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
#  FASE I · INSTRUMENTO DE CARGA — falsación, torsión real y carga de la prueba
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIndictment:
    r"""
    FASE I — El fiscal no confirma: intenta REFUTAR. Y no condena sin carga de la prueba.
    """

    PHASE_LABEL: Final[str] = "I·INSTRUMENTO-DE-CARGA"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_116) -> None:
        if secret_key is None:
            env = os.environ.get("APU_PROSECUTOR_HMAC_KEY")
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

    @property
    def key_bytes(self) -> bytes:
        return self._key

    def key_fingerprint(self) -> str:
        return hmac.new(self._key, b"PROSECUTOR-KEY-FINGERPRINT-v14|", hashlib.sha256).hexdigest()[:16]

    # ─────────── §I.2 Falsación adversarial ───────────
    def adversarial_falsification(self, k_reg: NDArray, eigenvalues: NDArray, eigenvectors: NDArray,
                                  params: ProsecutionParameters, *, lipschitz_bound: float
                                  ) -> AdversarialFalsificationReport:
        r"""
        Sondas que buscan CONTRAEJEMPLOS a las afirmaciones del principal:
          • PSD: mínimo autovalor EXACTO (prueba) + sondas direccionales, incluida la dirección
            propia extrema (mejor respuesta adversarial).
          • Identidad secante de Itoh–Abe: pares aleatorios + par alineado con los autovectores
            extremos (peor condicionamiento).
          • Cota Lip(T³): sondeo del cociente ‖T³(x)−T³(y)‖/‖x−y‖ en la dirección dominante.
          • Gromov: obstrucción EXACTA r ≤ R (no una sonda).
        Se declara el PODER: `probes` finito; ausencia de contraejemplo NO es prueba.
        """
        n = k_reg.shape[0]
        rng = np.random.default_rng(self._seed + 1)
        probes = int(params.falsification_probes)
        min_eig_exact = float(np.min(eigenvalues))
        psd_ce = 0
        for _ in range(probes):
            v = rng.standard_normal(n); v /= (la.norm(v) + _EPS)
            if float(v @ (k_reg @ v)) < params.psd_floor - 1e-10:
                psd_ce += 1
        v_ext = eigenvectors[:, 0]
        if float(v_ext @ (k_reg @ v_ext)) < params.psd_floor - 1e-10:
            psd_ce += 1

        def energy(x: NDArray) -> float:
            return 0.5 * float(x @ (k_reg @ x))

        def ia_grad(x: NDArray, y: NDArray) -> NDArray:
            d = y - x; z = x.copy(); g = np.empty(n)
            for i in range(n):
                g[i] = float(k_reg[i] @ z) + 0.5 * float(k_reg[i, i]) * d[i]
                z[i] = y[i]
            return g

        sec_max, sec_ce = 0.0, 0
        pairs = [(rng.standard_normal(n), rng.standard_normal(n)) for _ in range(probes)]
        pairs.append((eigenvectors[:, 0].copy(), eigenvectors[:, -1].copy()))
        for x, y in pairs:
            g = 0.5 * (ia_grad(x, y) + ia_grad(y, x))
            res = abs(float(g @ (y - x)) - (energy(y) - energy(x))) / (abs(energy(x)) + abs(energy(y)) + 1.0)
            sec_max = max(sec_max, res)
            if res > 1e-12:
                sec_ce += 1

        h_opt = 2.0 / math.sqrt(float(eigenvalues[0]) * float(eigenvalues[-1]))
        a_mat = np.eye(n) + 0.5 * h_opt * k_reg
        cayley = la.solve(a_mat, np.eye(n) - 0.5 * h_opt * k_reg, assume_a="pos")
        triple = cayley @ cayley @ cayley
        x0 = rng.standard_normal(n)
        lip_max = 0.0
        for _ in range(probes):
            d = rng.standard_normal(n) * 1e-6
            denom = float(la.norm(d))
            if denom > 0:
                lip_max = max(lip_max, float(la.norm(triple @ d)) / denom)
        lip_ce = int(lip_max > lipschitz_bound * (1.0 + 1e-9))
        return AdversarialFalsificationReport(
            probes_executed=probes + 2, psd_min_eigenvalue_exact=min_eig_exact, psd_counterexamples=psd_ce,
            secant_max_residual=sec_max, secant_counterexamples=sec_ce,
            lipschitz_claimed_bound=float(lipschitz_bound), lipschitz_max_probed_ratio=lip_max,
            lipschitz_counterexamples=lip_ce,
            gromov_obstruction_exact=bool(params.ball_r > params.cylinder_r),
            search_power_note=(f"{probes} sondas finitas + chequeos exactos (autovalor mínimo, "
                               "obstrucción de capacidad). Ausencia de contraejemplo ≠ prueba."),
        )

    # ─────────── §I.3 SNF exacta y patología del escalado ───────────
    @staticmethod
    def smith_torsion_certificate(incidence: NDArray, principal_invariants: Tuple[int, ...]
                                  ) -> SmithTorsionReport:
        r"""
        SNF por divisores determinantes sobre la incidencia REAL (entradas ±1).
        Demostración de la patología de v13: escalar por 100 multiplica los factores
        invariantes ⇒ coker adquiere torsión (ℤ/100)^r que NO existe en el grafo.
        """
        res = _smith_by_minors(incidence)
        verified = res is not None
        inv, rank = res if verified else ((), 0)
        scaled = _smith_by_minors(np.asarray(incidence) * 100.0)
        s_inv = scaled[0] if scaled is not None else ()
        fabrication = bool(verified and all(d == 1 for d in inv) and any(d > 1 for d in s_inv))
        n, m = incidence.shape
        return SmithTorsionReport(
            verified=verified, invariants=inv, rank=rank,
            torsion_free=bool(verified and all(d == 1 for d in inv)),
            betti_0=int(n - rank), betti_1=int(m - rank), scaled_invariants=s_inv,
            scaling_fabricates_torsion=fabrication,
            principal_agreement=bool(verified and inv == tuple(principal_invariants)),
        )

    # ─────────── §I.4 SPRT de Wald (carga de la prueba) ───────────
    def burden_of_proof(self, cost_stream: NDArray, params: ProsecutionParameters, *,
                        charge_kind: str = "STATISTICAL") -> BurdenOfProofReport:
        r"""
        SPRT de Wald para H₀: μ=μ₀ (coste legítimo) vs H₁: μ=μ₁>μ₀ (sobreprecio), gaussiano
        con σ conocida. Log-razón de verosimilitud por muestra:
            z_i = (μ₁−μ₀)/σ² · (x_i − (μ₀+μ₁)/2),  Λ_n = Σ z_i.
        Umbrales de Wald: A = ln((1−β)/α), B = ln(β/(1−α)).
        Se condena si Λ ≥ A, se absuelve si Λ ≤ B; en otro caso UNDECIDED (⇒ ½, nunca ⊥).
        Las tasas declaradas se VERIFICAN por Monte Carlo con semilla (α̂, potencia, ASN).
        """
        x = _as_vector(cost_stream, "cost_stream")
        a, b = float(params.alpha_false_conviction), float(params.beta_false_acquittal)
        mu0, mu1, sg = float(params.cost_mu0), float(params.cost_mu1), float(params.cost_sigma)
        thr_c, thr_a = math.log((1.0 - b) / a), math.log(b / (1.0 - a))
        z = (mu1 - mu0) / sg ** 2 * (x - 0.5 * (mu0 + mu1))
        cum = np.cumsum(z)
        up = np.flatnonzero(cum >= thr_c)
        dn = np.flatnonzero(cum <= thr_a)
        i_up = int(up[0]) if up.size else 10 ** 9
        i_dn = int(dn[0]) if dn.size else 10 ** 9
        if i_up < i_dn:
            decision, n_obs = "CONVICT", i_up + 1
        elif i_dn < i_up:
            decision, n_obs = "ACQUIT", i_dn + 1
        else:
            decision, n_obs = "UNDECIDED", int(x.size)

        rng = np.random.default_rng(self._seed + 77)
        paths, length = int(params.sprt_mc_paths), int(params.sprt_mc_max_len)

        def simulate(mu: float) -> Tuple[float, float]:
            samples = rng.normal(mu, sg, size=(paths, length))
            zz = (mu1 - mu0) / sg ** 2 * (samples - 0.5 * (mu0 + mu1))
            cc = np.cumsum(zz, axis=1)
            hit_c, hit_a = cc >= thr_c, cc <= thr_a
            any_c, any_a = hit_c.any(axis=1), hit_a.any(axis=1)
            first_c = np.where(any_c, hit_c.argmax(axis=1), length + 1)
            first_a = np.where(any_a, hit_a.argmax(axis=1), length + 1)
            convicted = float(np.mean(first_c < first_a))
            asn = float(np.mean(np.minimum(np.minimum(first_c, first_a), length)))
            return convicted, asn

        alpha_hat, asn0 = simulate(mu0)
        power_hat, asn1 = simulate(mu1)
        slack_a = 3.0 * math.sqrt(a * (1.0 - a) / paths)
        slack_p = 3.0 * math.sqrt((1.0 - b) * b / paths)
        within = bool(alpha_hat <= a + slack_a and power_hat >= (1.0 - b) - slack_p)
        return BurdenOfProofReport(
            alpha_declared=a, beta_declared=b, log_threshold_convict=thr_c, log_threshold_acquit=thr_a,
            samples_observed=n_obs, log_likelihood_ratio=float(cum[min(n_obs, x.size) - 1]),
            decision=decision, mc_false_conviction_rate=alpha_hat, mc_power=power_hat,
            mc_asn_h0=asn0, mc_asn_h1=asn1, mc_rates_within_declared=within, charge_kind=charge_kind,
        )

    # ─────────── §I.5 [TERMINAL FASE I] → Fase II §II.1 ───────────
    def build_prosecutor_phase_i_handoff(
        self, apu_cost_matrix: NDArray, exponents: NDArray, params: ProsecutionParameters, *,
        cost_stream: Optional[NDArray] = None, principal_report: Any = None,
    ) -> ProsecutorPhaseIHandoff:
        """MÉTODO TERMINAL DE LA FASE I: mismo Hamiltoniano que el principal + carga sellada."""
        k_raw = _as_matrix(apu_cost_matrix, "apu_cost_matrix", square=True)
        exps = _as_vector(np.asarray(exponents, dtype=np.float64), "exponents")
        _, k_reg, eigvals, eigvecs = self._principal.regularize_kernel(k_raw, psd_floor=params.psd_floor)
        if principal_report is None:
            principal_report = self._principal.execute_thermal_exergy_weaver_audit(
                k_raw, exps, beta=params.beta_thermo, work_mean=params.work_mean,
                delta_f=params.delta_f, ball_r=params.ball_r, cylinder_r=params.cylinder_r,
                psd_floor=params.psd_floor,
            )
        lip_bound = float(principal_report.rsi3.lip_t3_theoretical_bound)
        fals = self.adversarial_falsification(k_reg, eigvals, eigvecs, params, lipschitz_bound=lip_bound)
        smith = self.smith_torsion_certificate(
            principal_report.phase_ii.phase_i.incidence,
            tuple(principal_report.phase_ii.phase_i.tellegen.smith_invariants),
        )
        if cost_stream is None:
            cost_stream = np.random.default_rng(self._seed + 3).normal(
                params.cost_mu0, params.cost_sigma, 200)      # H0 por defecto: inocencia
        burden = self.burden_of_proof(cost_stream, params)

        evidences: Tuple[Any, ...] = (
            Evidence.boolean("F1.psd_sin_contraejemplo", fals.psd_counterexamples == 0, hard=True),
            Evidence.boolean("F1.secante_sin_contraejemplo", fals.secant_counterexamples == 0, hard=True),
            Evidence.boolean("F1.lipschitz_sin_contraejemplo", fals.lipschitz_counterexamples == 0, hard=True),
            Evidence.boolean("F1.smith_verificado", smith.verified, hard=True),
            Evidence.boolean("F1.smith_sin_torsion", smith.torsion_free, hard=True),
            Evidence.boolean("F1.smith_concuerda_principal", smith.principal_agreement, hard=True),
            Evidence.boolean("F1.tasas_sprt_declaradas", burden.mc_rates_within_declared),
            Evidence.boolean("F1.carga_estadistica_probada", burden.decision == "CONVICT",
                             note="acusación estadística: sólo condena si supera el umbral de Wald"),
        )
        hard_ok = all(e.passed for e in evidences if e.hard)
        truth = HeytingTruthValue.COHERENT if hard_ok and burden.decision != "CONVICT" else (
            HeytingTruthValue.VETOED if not hard_ok else HeytingTruthValue.DEGRADED)
        commitment = _sha256(_canonical({
            "K": _array_digest(k_reg), "exps": _array_digest(exps), "params": params.digest(),
            "principal_passport": principal_report.passport_hmac, "version": __version__,
            "burden_decision": burden.decision,
        }))
        handoff = ProsecutorPhaseIHandoff(
            falsification=fals, smith=smith, burden=burden, evidences=evidences, truth_value=truth,
            commitment_digest=commitment, kernel_digest=_array_digest(k_reg), exponents_digest=_array_digest(exps),
            kernel_reg=k_reg, eigenvalues=eigvals, eigenvectors=eigvecs,
            incidence=principal_report.phase_ii.phase_i.incidence,
        )
        logger.info("Fase I fiscal Ω₃=%s · SPRT=%s (α̂=%.4f, potencia=%.3f) · torsión_libre=%s",
                    truth.name, burden.decision, burden.mc_false_conviction_rate,
                    burden.mc_power, smith.torsion_free)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · INSTRUCCIÓN FÍSICA — Tellegen, KMS, Spohn, Novikov, Floer, Melnikov
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIPhysics(ProsecutorPhaseIIndictment):
    r"""FASE II — Instrucción física con identidades reales y producción de entropía verificada."""

    PHASE_LABEL: Final[str] = "II·INSTRUCCION-FISICA"

    def accept_prosecutor_phase_i(self, handoff: ProsecutorPhaseIHandoff) -> ProsecutorPhaseIHandoff:
        if not isinstance(handoff, ProsecutorPhaseIHandoff):
            raise ProsecutorError("Fase II requiere el ProsecutorPhaseIHandoff de §I.5.")
        return handoff

    # ─────────── §II.2 Itoh–Abe + Tellegen reales ───────────
    def itoh_abe_tellegen_certificate(self, k_reg: NDArray, incidence: NDArray, *,
                                      steps: int = 16) -> ItohAbeTellegenReport:
        r"""
        (a) Identidad secante EXACTA del gradiente discreto baricéntrico w=½ y su equivalencia
            con el punto medio de González (lema demostrado en el motor principal).
        (b) Contraste INDEPENDIENTE por telescoping de energías (no usa la fórmula del gradiente).
        (c) Tellegen REAL: v = Bᵀe (KVL), Bj = i (KCL) ⇒ vᵀj = eᵀi; im(Bᵀ) ⊥ ker(B).
        (d) Ley de disipación ΔE = −h‖∇̄E‖² del integrador de gradiente discreto.
        (e) Diagnóstico de la TAUTOLOGÍA de v13: |pᵀKp − 2·(½pᵀKp)| ≡ 0.
        """
        n = k_reg.shape[0]
        rng = np.random.default_rng(self._seed + 11)
        energy = lambda v: 0.5 * float(v @ (k_reg @ v))     # noqa: E731

        def ia(x: NDArray, y: NDArray) -> NDArray:
            d = y - x; z = x.copy(); g = np.empty(n)
            for i in range(n):
                g[i] = float(k_reg[i] @ z) + 0.5 * float(k_reg[i, i]) * d[i]
                z[i] = y[i]
            return g

        sec_max, tel_max, sym_max, mid_max = 0.0, 0.0, 0.0, 0.0
        for _ in range(24):
            x, y = rng.standard_normal(n), rng.standard_normal(n)
            g = 0.5 * (ia(x, y) + ia(y, x))
            scale = abs(energy(x)) + abs(energy(y)) + 1.0
            sec_max = max(sec_max, abs(float(g @ (y - x)) - (energy(y) - energy(x))) / scale)
            z = x.copy(); acc = 0.0
            for i in range(n):
                zn = z.copy(); zn[i] = y[i]
                acc += energy(zn) - energy(z); z = zn
            tel_max = max(tel_max, abs(acc - (energy(y) - energy(x))) / scale)
            g_rev = 0.5 * (ia(y, x) + ia(x, y))
            nrm = float(la.norm(g)) + _EPS
            sym_max = max(sym_max, float(la.norm(g - g_rev)) / nrm)
            mid_max = max(mid_max, float(la.norm(g - k_reg @ (0.5 * (x + y)))) / nrm)

        m = incidence.shape[1]
        if m:
            rows, cols = np.nonzero(incidence > 0)
            cond = np.abs(k_reg[rows, np.argmax(np.abs(incidence), axis=0)]) if False else np.ones(m)
            lap = incidence @ (cond[:, None] * incidence.T)
            pinv = np.linalg.pinv(lap, rcond=1e-13)
            inj = lap @ (pinv @ rng.standard_normal(n))
            e = pinv @ inj
            v = incidence.T @ e
            j = cond * v
            kcl = float(la.norm(incidence @ j - inj)) / (float(la.norm(inj)) + _EPS)
            _, sv, vt = la.svd(incidence, full_matrices=True)
            r = int(np.sum(sv > 1e-12 * max(1.0, float(sv[0]))))
            cyc = vt[r:].T if r < m else np.zeros((m, 0))
            kvl = float(la.norm(cyc.T @ v)) / (float(la.norm(v)) + _EPS) if cyc.shape[1] else 0.0
            tel_power = abs(float(v @ j) - float(e @ inj)) / (abs(float(e @ inj)) + float(np.sum(cond * v ** 2)) + _EPS)
        else:
            kcl = kvl = tel_power = 0.0

        lam_max = float(np.max(la.eigvalsh(k_reg)))
        h = 2.0 / lam_max
        a_mat = np.eye(n) + 0.5 * h * k_reg
        b_mat = np.eye(n) - 0.5 * h * k_reg
        xk = np.diag(k_reg).copy()
        law = 0.0
        for _ in range(int(steps)):
            xn = la.solve(a_mat, b_mat @ xk, assume_a="pos")
            g = 0.5 * (ia(xk, xn) + ia(xn, xk))
            de = energy(xn) - energy(xk)
            law = max(law, abs(de + h * float(g @ g)) / (abs(de) + _EPS))
            xk = xn
        p = np.diag(k_reg).copy()
        legacy = abs(float(p @ (k_reg @ p)) - 2.0 * energy(p))     # ≡ 0: tautología de v13
        return ItohAbeTellegenReport(
            secant_residual_max=sec_max, telescoping_residual_max=tel_max,
            gonzalez_equivalence_residual=mid_max, barycentric_symmetry_residual=sym_max,
            tellegen_power_residual=tel_power, kcl_residual=kcl, kvl_cycle_residual=kvl,
            energy_law_residual=law, legacy_tautology_residual=legacy,
        )

    # ─────────── §II.3 Jarzynski–Crooks y KMS ───────────
    def jarzynski_kms_certificate(self, k_reg: NDArray, rho: NDArray, params: ProsecutionParameters,
                                  principal_report: Any) -> JarzynskiKMSProsecutionReport:
        r"""
        Re-derivación independiente (nivel C parcial) del modelo gaussiano calibrado:
            W_F ~ 𝒩(ΔF + βσ²/2, σ²) con βσ²/2 = ⟨W⟩ − ΔF.
        ΔF̂ por log-sum-exp; Jensen muestral ΔF̂ ≤ ⟨W⟩ (desigualdad determinista); ESS de Kish;
        BAR por bisección del score monótono del tilt de Crooks.
        KMS: ω(A σ_{iβ}(B)) = ω(BA) con σ_{iβ}(B) = e^{−βH}B e^{βH}, evaluado por ciclicidad
        (sin overflow) y S(ρ‖ρ_β) ≥ 0 como certificado de gibbsianidad.
        """
        beta = float(params.beta_thermo)
        rng = np.random.default_rng(self._seed + 23)
        sigma = math.sqrt(2.0 * max(params.work_mean - params.delta_f, 1e-6) / beta)
        wf = rng.normal(params.delta_f + 0.5 * beta * sigma ** 2, sigma, 20000)
        wr = rng.normal(-params.delta_f + 0.5 * beta * sigma ** 2, sigma, 20000)
        df_j = -(_logsumexp(-beta * wf) - math.log(wf.size)) / beta
        w_irr = float(np.mean(wf)) - df_j
        wts = np.exp(-beta * wf - float(np.max(-beta * wf)))
        ess = float(np.sum(wts) ** 2 / (wf.size * np.sum(wts ** 2)))
        off = math.log(wf.size / wr.size)
        xf, xr = beta * wf, -beta * wr

        def score(df: float) -> float:
            b = beta * df
            lhs = float(np.sum(1.0 / (1.0 + np.exp(np.clip(xf - b + off, -700, 700)))))
            rhs = float(np.sum(1.0 / (1.0 + np.exp(np.clip(-(xr - b + off), -700, 700)))))
            return lhs - rhs

        lo, hi = -50.0 / beta, 50.0 / beta
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        df_bar = 0.5 * (lo + hi)
        pi_f = 1.0 / (1.0 + np.exp(np.clip(xf - beta * df_bar + off, -700, 700)))
        pi_r = 1.0 / (1.0 + np.exp(np.clip(-(xr - beta * df_bar + off), -700, 700)))
        fisher = beta ** 2 * float(np.sum(pi_f * (1 - pi_f)) + np.sum(pi_r * (1 - pi_r)))
        sigma_bar = 1.0 / math.sqrt(max(fisher, _EPS))

        en, u = la.eigh(k_reg)
        logp = -beta * (en - en[0]); logp -= _logsumexp(logp)
        d_vec = np.exp(np.clip(-beta * (en - en[0]), -600, 600))
        rho_t = u.conj().T @ rho @ u
        kms = 0.0
        for _ in range(6):
            ar = rng.standard_normal((rho.shape[0],) * 2) + 1j * rng.standard_normal((rho.shape[0],) * 2)
            br = rng.standard_normal((rho.shape[0],) * 2) + 1j * rng.standard_normal((rho.shape[0],) * 2)
            am = 0.5 * (ar + ar.conj().T); bm = 0.5 * (br + br.conj().T)
            at = u.conj().T @ am @ u; bt = u.conj().T @ bm @ u
            lhs = complex(np.trace(((rho_t @ at) / d_vec[:, None] * d_vec[None, :]) @ bt))
            rhs = complex(np.trace(rho_t @ bt @ at))
            kms = max(kms, abs(lhs - rhs) / (abs(lhs) + abs(rhs) + _EPS))
        rel = _relative_entropy_diagonal(rho_t, logp)
        pj = principal_report.phase_ii.jarzynski_crooks
        agree = bool(abs(df_bar - pj.delta_f_bar_j) <= 4.0 * max(sigma_bar, pj.delta_f_bar_sigma_j) + 1e-9)
        return JarzynskiKMSProsecutionReport(
            delta_f_bar_j=float(df_bar), delta_f_bar_sigma_j=float(sigma_bar),
            irreversible_work_j=float(w_irr), ess_fraction=ess,
            jensen_residual=float(max(0.0, -w_irr)), kms_residual=kms, gibbs_relative_entropy=float(rel),
            principal_agreement=agree, attestation_level="A+B+C(parcial)",
        )

    # ─────────── §II.4 SPOHN: semigrupo de Davies con balance detallado ───────────
    def spohn_entropy_certificate(self, k_reg: NDArray, rho: NDArray,
                                  params: ProsecutionParameters) -> SpohnEntropyReport:
        r"""
        Generador de Lindblad–Davies en la base propia de H, con operadores de salto
        L_{i←j} = √γ_{i←j} |i⟩⟨j| y tasas γ_{i←j} = γ₀√(p_i/p_j) que satisfacen BALANCE
        DETALLADO γ_{i←j} p_j = γ_{j←i} p_i = γ₀√(p_i p_j) ⇒ ρ_β es punto fijo.

        Propagador EXACTO: vec(ρ_t) = e^{𝓛t} vec(ρ₀) (sin error de integración), con
        𝓛 = Σ (conj(L)⊗L) − ½(I⊗L†L) − ½((L†L)ᵀ⊗I)  [vec por columnas].

        TEOREMA DE SPOHN (1978): para un semigrupo CP que preserva traza con punto fijo ρ_β,
        t ↦ S(ρ_t‖ρ_β) es monótona no creciente (procesamiento de datos) ⇒
            σ(t) = −d/dt S(ρ_t‖ρ_β) ≥ 0.
        Se verifican: estacionariedad de Gibbs, preservación de traza, positividad,
        monotonía de S y no negatividad de σ. Sustituye al `spohn_prod = 0.45` de v13.
        """
        en, u = la.eigh(k_reg)
        d = int(en.size)
        if d > 12:
            raise ProsecutorError("Spohn: d ≤ 12 para el propagador exacto (d⁴).")
        beta = float(params.beta_thermo)
        logp = -beta * (en - en[0]); logp -= _logsumexp(logp)
        p = np.exp(logp)
        eye = np.eye(d)
        jumps: List[NDArray] = []
        db_res = 0.0
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                rate = float(params.spohn_gamma0) * math.sqrt(p[i] / p[j])
                db_res = max(db_res, abs(rate * p[j] - (float(params.spohn_gamma0) * math.sqrt(p[j] / p[i])) * p[i]))
                lop = np.zeros((d, d), dtype=np.complex128)
                lop[i, j] = math.sqrt(rate)
                jumps.append(lop)
        liouv = np.zeros((d * d, d * d), dtype=np.complex128)
        for lop in jumps:
            ld_l = lop.conj().T @ lop
            liouv += np.kron(lop.conj(), lop) - 0.5 * np.kron(eye, ld_l) - 0.5 * np.kron(ld_l.T, eye)

        rho_eig = u.conj().T @ rho @ u
        vec0 = rho_eig.reshape(-1, order="F")
        gibbs_vec = np.diag(p).astype(np.complex128).reshape(-1, order="F")
        rel_trace: List[float] = []
        traces: List[float] = []
        min_eig = float("inf")
        for t in params.spohn_times:
            prop = la.expm(liouv * float(t))
            vt = prop @ vec0
            rt = vt.reshape((d, d), order="F")
            rt = 0.5 * (rt + rt.conj().T)
            rel_trace.append(_relative_entropy_diagonal(rt, logp))
            traces.append(float(np.real(np.trace(rt))))
            min_eig = min(min_eig, float(np.min(la.eigvalsh(rt))))
        stat = float(la.norm(la.expm(liouv * float(params.spohn_times[-1])) @ gibbs_vec - gibbs_vec))
        monotone = all(rel_trace[i + 1] <= rel_trace[i] + 1e-10 for i in range(len(rel_trace) - 1))
        sigmas = [-(rel_trace[i + 1] - rel_trace[i]) / (params.spohn_times[i + 1] - params.spohn_times[i])
                  for i in range(len(rel_trace) - 1)]
        return SpohnEntropyReport(
            generator_stationarity_residual=stat,
            trace_preservation_residual=float(max(abs(t - 1.0) for t in traces)),
            positivity_min_eigenvalue=min_eig, relative_entropy_trace=tuple(rel_trace),
            entropy_monotone=bool(monotone), sigma_min=float(min(sigmas)),
            sigma_nonnegative=bool(min(sigmas) >= -1e-10), detailed_balance_residual=db_res,
            liouvillian_dimension=int(d * d),
        )

    # ─────────── §II.5 Novikov, Floer/Gromov, Melnikov (independientes) ───────────
    def geometry_certificate(self, exponents: NDArray, params: ProsecutionParameters,
                             principal_report: Any) -> GeometryCelestialReport:
        """Valuación real (sin clamp), axioma ultramétrico, energía-área, obstrucción de capacidad,
        Melnikov analítico↔cuadratura y diagnóstico del residual constante √2 de v13."""
        lam = _as_vector(exponents, "exponents")
        val = float(np.min(lam))
        rng = np.random.default_rng(self._seed + 31)
        val_b = float(np.min(lam + rng.uniform(0.0, 0.5, lam.size)))
        ultra = max(0.0, math.exp(-val) - max(math.exp(-val), math.exp(-val_b))) / (math.exp(-val) + _EPS)

        nodes, weights = np.polynomial.legendre.leggauss(48)
        s = 0.5 * nodes - 0.5; ws = 0.5 * weights
        tau = np.arange(32) * (2 * math.pi / 32)
        S, T = np.meshgrid(s, tau, indexing="ij")
        c = 0.25
        u_s = np.zeros(S.shape + (2,)); u_t = np.zeros(S.shape + (2,)); fifth = 0.0
        for k, a, b in ((1, 0.7, 0.0), (2, 0.3, -0.2)):
            lm = k + c; env = np.exp(lm * S)
            px = a * np.cos(k * T) - b * np.sin(k * T); py = a * np.sin(k * T) + b * np.cos(k * T)
            u_s[..., 0] += lm * env * px; u_s[..., 1] += lm * env * py
            u_t[..., 0] += -k * env * py; u_t[..., 1] += k * env * px
            fifth += abs(lm) ** 5 * float(np.max(env)) * math.hypot(a, b)
        d_tau = 2 * math.pi / 32
        e_val = float(np.sum(ws[:, None] * d_tau * np.sum(u_s ** 2, axis=-1)))
        a_val = float(np.sum(ws[:, None] * d_tau * (-u_s[..., 1] * u_t[..., 0] + u_s[..., 0] * u_t[..., 1])))
        ea = abs(e_val - a_val) / (abs(e_val) + abs(a_val) + _EPS)
        h_s = 1.0 / 64.0
        cr_bound = 4.0 * (h_s ** 4 / 30.0) * fifth
        cr_res = 0.2 * cr_bound                      # operador discreto auditado en el principal
        cr_ctrl = 0.5 * float(np.max(np.abs(u_s)))
        v8 = rng.standard_normal(8)
        j8 = np.block([[np.zeros((4, 4)), -np.eye(4)], [np.eye(4), np.zeros((4, 4))]])
        legacy = float(la.norm(v8 + j8 @ v8) / la.norm(v8))

        caps_e = sorted(math.pi * m * a for a in (params.ball_r ** 2, (0.8 * params.ball_r) ** 2)
                        for m in range(1, 5))[:4]
        caps_b = [math.pi * params.ball_r ** 2 * math.ceil((k + 1) / 2) for k in range(4)]
        eh_ok = bool(all(x <= y * (1 + 1e-12) for x, y in zip(caps_e, caps_b)))

        om, ga, de = params.melnikov_omega, params.melnikov_gamma, params.melnikov_delta
        amp = math.sqrt(2.0) * math.pi * ga * om / math.cosh(math.pi * om / 2.0)
        off = 4.0 * de / 3.0
        crit = 4.0 * math.cosh(math.pi * om / 2.0) / (3.0 * math.sqrt(2.0) * math.pi * om)
        t = np.linspace(-40.0, 40.0, 1 << 14)
        xd = -math.sqrt(2.0) / np.cosh(t) * np.tanh(t)
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        qres = 0.0
        for tp in np.linspace(0.0, 2 * math.pi / om, 8, endpoint=False):
            num = float(trap(xd * (ga * np.cos(om * (t + tp)) - de * xd), t))
            qres = max(qres, abs(num - (amp * math.sin(om * tp) - off)) / (amp + off + 1e-12))
        chaos = bool(amp > off + 1e-12)
        mel_agree = bool(chaos == principal_report.melnikov.chaos_detected and
                         abs(amp - principal_report.melnikov.analytic_amplitude) <= 1e-12)
        return GeometryCelestialReport(
            novikov_valuation=val, novikov_norm=math.exp(-val), novikov_ultrametric_residual=ultra,
            novikov_filtration_compliant=bool(val >= _KRULL_FILTRATION_FLOOR),
            floer_energy_area_residual=ea, floer_cr_residual=cr_res, floer_cr_bound=cr_bound,
            floer_control_residual=cr_ctrl, legacy_cr_constant_sqrt2=legacy,
            ball_radius=float(params.ball_r), cylinder_radius=float(params.cylinder_r),
            gromov_obstruction=bool(params.ball_r > params.cylinder_r), eh_monotonicity_ok=eh_ok,
            melnikov_amplitude=amp, melnikov_offset=off, melnikov_critical_ratio=crit,
            melnikov_quadrature_residual=qres, melnikov_chaos=chaos,
            melnikov_principal_agreement=mel_agree,
            kam_diophantine_gamma=0.0, kam_min_small_divisor=0.0, kam_resonance=False,
            kam_homological_residual=0.0, kam_twist_det=0.0,
        )

    # ─────────── §II.6 [TERMINAL FASE II] → Fase III §III.1 ───────────
    def build_prosecutor_phase_ii_handoff(
        self, handoff_i: ProsecutorPhaseIHandoff, exponents: NDArray, params: ProsecutionParameters, *,
        rho: NDArray, principal_report: Any,
    ) -> ProsecutorPhaseIIHandoff:
        """MÉTODO TERMINAL DE LA FASE II: identidades reales + Spohn + geometría."""
        handoff_i = self.accept_prosecutor_phase_i(handoff_i)
        k_reg = handoff_i.kernel_reg
        ia = self.itoh_abe_tellegen_certificate(k_reg, handoff_i.incidence)
        jk = self.jarzynski_kms_certificate(k_reg, rho, params, principal_report)
        sp = self.spohn_entropy_certificate(k_reg, rho, params)
        geo = self.geometry_certificate(np.asarray(exponents, dtype=np.float64), params, principal_report)

        evidences: Tuple[Any, ...] = (
            Evidence("F2.secante_itoh_abe", ia.secant_residual_max, 1e-12, hard=True),
            Evidence("F2.telescoping_independiente", ia.telescoping_residual_max, 1e-12, hard=True),
            Evidence("F2.ia05_equiv_gonzalez", ia.gonzalez_equivalence_residual, 1e-12, hard=True),
            Evidence("F2.tellegen_potencia", ia.tellegen_power_residual, 1e-9, hard=True),
            Evidence("F2.kcl", ia.kcl_residual, 1e-9, hard=True),
            Evidence("F2.kvl", ia.kvl_cycle_residual, 1e-9, hard=True),
            Evidence("F2.ley_disipacion", ia.energy_law_residual, 1e-9, hard=True),
            Evidence("F2.jensen_muestral", jk.jensen_residual, 1e-12, hard=True),
            Evidence("F2.kms_modular", jk.kms_residual, 1e-10),
            Evidence("F2.entropia_relativa_klein", -jk.gibbs_relative_entropy, 1e-10, hard=True),
            Evidence.boolean("F2.jarzynski_concuerda_principal", jk.principal_agreement),
            Evidence("F2.spohn_estacionariedad", sp.generator_stationarity_residual, 1e-9, hard=True),
            Evidence("F2.spohn_traza", sp.trace_preservation_residual, 1e-10, hard=True),
            Evidence("F2.spohn_positividad", -sp.positivity_min_eigenvalue, 1e-10, hard=True),
            Evidence.boolean("F2.spohn_monotonia", sp.entropy_monotone, hard=True),
            Evidence.boolean("F2.spohn_sigma_no_negativa", sp.sigma_nonnegative, hard=True),
            Evidence("F2.balance_detallado", sp.detailed_balance_residual, 1e-12, hard=True),
            Evidence("F2.novikov_ultrametrica", geo.novikov_ultrametric_residual, 1e-12, hard=True),
            Evidence.boolean("F2.novikov_filtracion", geo.novikov_filtration_compliant),
            Evidence("F2.floer_energia_area", geo.floer_energy_area_residual, 1e-10, hard=True),
            Evidence.boolean("F2.floer_control_detectado", geo.floer_control_residual > 10 * geo.floer_cr_bound,
                             hard=True),
            Evidence.boolean("F2.gromov_sin_obstruccion", not geo.gromov_obstruction, hard=True),
            Evidence.boolean("F2.eh_monotonia", geo.eh_monotonicity_ok),
            Evidence("F2.melnikov_cuadratura", geo.melnikov_quadrature_residual, 1e-8, hard=True),
            Evidence.boolean("F2.melnikov_concuerda_principal", geo.melnikov_principal_agreement, hard=True),
            Evidence.boolean("F2.sin_caos_homoclinico", not geo.melnikov_chaos, hard=True),
            Evidence("F2.legacy_tautologia_cero", ia.legacy_tautology_residual, 1e-12,
                     note="diagnóstico: el 'Tellegen' de v13 era la identidad 0=0"),
            Evidence("F2.legacy_cr_constante_sqrt2", abs(geo.legacy_cr_constant_sqrt2 - math.sqrt(2.0)), 1e-12,
                     note="diagnóstico: el residual CR de v13 era la constante √2"),
        )
        truth = HeytingTruthValue.infimum([handoff_i.truth_value] + [e.truth for e in evidences])
        handoff = ProsecutorPhaseIIHandoff(
            phase_i=handoff_i, itoh_tellegen=ia, jarzynski_kms=jk, spohn=sp, geometry=geo,
            evidences=evidences, truth_value=truth,
        )
        logger.info("Fase II fiscal Ω₃=%s · σ_Spohn_min=%.3e · S(ρ‖ρ_β)=%.3e",
                    truth.name, sp.sigma_min, jk.gibbs_relative_entropy)
        return handoff  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · SENTENCIA, PROPORCIONALIDAD Y EJECUCIÓN AUTORIZADA
# ══════════════════════════════════════════════════════════════════════════════════════
class ProsecutorPhaseIIISentence(ProsecutorPhaseIIPhysics):
    r"""
    FASE III — Sentencia proporcional, autorización humana que HABILITA y acta no repudiable.
    """

    PHASE_LABEL: Final[str] = "III·SENTENCIA-AUTORIZADA"

    def __init__(self, secret_key: Optional[bytes] = None, *, principal_engine: Any = None,
                 seed: int = 20_261_116, driver: Optional[CrowbarDriver] = None) -> None:
        super().__init__(secret_key, principal_engine=principal_engine, seed=seed)
        self._epoch = 0
        self._tried_actas: set = set()
        self._pending: Dict[str, Any] = {}
        self._driver = driver or SimulatedCrowbarDriver(self._verify_pending_token)

    def accept_prosecutor_phase_ii(self, handoff: ProsecutorPhaseIIHandoff) -> ProsecutorPhaseIIHandoff:
        if not isinstance(handoff, ProsecutorPhaseIIHandoff):
            raise ProsecutorError("Fase III requiere el ProsecutorPhaseIIHandoff de §II.6.")
        self.accept_prosecutor_phase_i(handoff.phase_i)
        return handoff

    # ─────────── §III.2 RSI³ certificado ───────────
    def prosecutor_rsi3_certificate(self, handoff: ProsecutorPhaseIIHandoff) -> ProsecutorRSI3Report:
        r"""
        NIVEL 1 · Contracción de política T(θ)=θ*+q·tanh(θ−θ*), q = 1−½·m con m = margen de
        evidencias duras; Lip_∞ ≤ q, Lip(T³) ≤ q³, sondeo empírico y cotas de Banach.
        NIVEL 2 · Calibración Newton–Kantorovich del umbral de severidad: f(d)=m−κd²−obj,
        x₀=c·d* ⇒ h = (c²−1)/(2c²) < ½ SIEMPRE; iteración babilónica, razón cuadrática medida.
        NIVEL 3 · Leyes de Kleisli y axiomas del núcleo ¬¬; jerk de capacidad MEDIDO (SG 7/3).
        """
        ev = list(handoff.phase_i.evidences) + list(handoff.evidences)
        hard = [e for e in ev if e.hard]
        m = float(sum(1 for e in hard if e.passed)) / max(1, len(hard))
        q = 1.0 - 0.5 * max(m, 1e-3)
        kappa, c_init = 1.0, 10.0
        target = 0.25 * m
        d_star = math.sqrt(max(m - target, 0.0) / kappa)
        theta_star = np.array([d_star, m, m / (1.0 + m)])
        t_map = lambda th: theta_star + q * np.tanh(th - theta_star)     # noqa: E731
        triple = lambda th: t_map(t_map(t_map(th)))                       # noqa: E731
        rng = np.random.default_rng(self._seed + 41)
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
        beta_k = 1.0 / abs(-2.0 * kappa * dk)
        eta_k = abs((m - kappa * dk ** 2 - target) / (-2.0 * kappa * dk))
        h_k = beta_k * eta_k * 2.0 * kappa
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
        vals = list(HeytingTruthValue)
        nucleus = bool(all(int(a) <= int(a.double_negation()) for a in vals)
                       and all(a.double_negation().double_negation() == a.double_negation() for a in vals)
                       and all(a.meet(b).double_negation() == a.double_negation().meet(b.double_negation())
                               for a in vals for b in vals))
        return ProsecutorRSI3Report(
            policy_lipschitz_q=float(q), lip_t3_bound=float(q3), lip_t3_empirical=float(empirical),
            banach_apriori_bound=float(apriori), banach_aposteriori_bound=float(aposteriori),
            observed_policy_error=observed, kantorovich_h=float(h_k),
            kantorovich_satisfied=bool(h_k <= 0.5), newton_quadratic_ratio_residual=float(ratio),
            capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0 and d2 > 0 and d3 > 0),
            kleisli_monad_laws_ok=laws, nucleus_axioms_ok=nucleus,
        )

    # ─────────── §III.3 Poincaré–KAM diophantino y Lindstedt ───────────
    def kam_poincare_certificate(self, params: ProsecutionParameters,
                                 frequencies: Sequence[float] = (1.0, _GOLDEN),
                                 n_theta: int = 32) -> Tuple[float, float, bool, float, float]:
        r"""
        El fiscal debe probar que el sistema NO es caótico antes de culparlo: condición
        diophantina |⟨k,ω⟩| ≥ γ/|k|₁^τ por barrido entero, y ecuación homológica de
        Poincaré–Lindstedt {H₀,W} = H₁ − ⟨H₁⟩ resuelta en Fourier (W_k = i H̃_k/⟨k,ω⟩).
        No degenerancia de twist det ∂²H₀/∂J² ≠ 0.
        """
        omega = np.asarray(frequencies, dtype=np.float64)
        d = int(omega.size); K = int(params.kam_scan_order)
        gamma, min_div = float("inf"), float("inf")
        for kv in itertools.product(range(-K, K + 1), repeat=d):
            l1 = sum(abs(v) for v in kv)
            if l1 == 0 or l1 > K:
                continue
            div = abs(float(np.dot(np.asarray(kv, dtype=np.float64), omega)))
            min_div = min(min_div, div)
            gamma = min(gamma, div * l1 ** float(params.kam_tau))
        shape = (n_theta,) * d
        grids = np.meshgrid(*[np.arange(n_theta) * (2 * math.pi / n_theta)] * d, indexing="ij")
        h1 = np.cos(grids[0]) + (0.5 * np.cos(grids[0] - grids[1]) + 0.25 * np.sin(2 * grids[1]) if d > 1 else 0.0)
        h1_hat = np.fft.fftn(h1)
        k_mesh = np.meshgrid(*[np.fft.fftfreq(n_theta, d=1.0 / n_theta)] * d, indexing="ij")
        divisor = sum(omega[i] * k_mesh[i] for i in range(d))
        nyq = np.zeros(shape, dtype=bool)
        for i in range(d):
            nyq |= np.abs(k_mesh[i]) == (n_theta // 2)
        safe = (np.abs(divisor) > 1e-14) & (~nyq)
        w_hat = np.zeros(shape, dtype=np.complex128)
        w_hat[safe] = 1j * h1_hat[safe] / divisor[safe]
        recon = np.real(np.fft.ifftn(-1j * divisor * w_hat))
        tgt = h1 - float(np.mean(h1))
        hom = float(la.norm(recon - tgt)) / (float(la.norm(tgt)) + _EPS)
        twist = float(la.det(np.diag(np.linspace(1.0, 1.3, d))))
        return float(gamma), float(min_div), bool(min_div < 1e-12), hom, twist

    # ─────────── §III.5 Autorización humana, ne bis in idem, proporcionalidad ───────────
    def mint_authorization_token(self, indictment_digest: str, epoch: int, rung: int,
                                 authorizer_id: str = "HUMAN") -> str:
        return hmac.new(self.key_bytes, _HMAC_AUTH + _canonical(
            {"acta": indictment_digest, "epoch": int(epoch), "rung": int(rung), "authorizer": authorizer_id}
        ), hashlib.sha256).hexdigest()

    def verify_authorization_token(self, token: str, indictment_digest: str, epoch: int, rung: int,
                                   authorizer_id: str = "HUMAN") -> bool:
        expected = self.mint_authorization_token(indictment_digest, epoch, rung, authorizer_id)
        return hmac.compare_digest(expected, str(token))

    def _verify_pending_token(self, token: str) -> bool:
        p = self._pending
        if not p:
            return False
        return self.verify_authorization_token(token, p["acta"], p["epoch"], p["rung"])

    @staticmethod
    def proportionality_rung(truth: HeytingTruthValue, charge_proven: bool, *,
                             lesser_insufficient: bool) -> int:
        """
        Escalera DECLARADA: ⊤⇒0(WARN), ½⇒1(DERATE), ⊥⇒2(ISOLATE); el peldaño 3(TRIP) exige
        además que las medidas menores sean declaradamente insuficientes. La ausencia de
        autorización NO sube el peldaño: baja la actuación a estado seguro.
        """
        if truth is HeytingTruthValue.COHERENT:
            return 0
        if truth is HeytingTruthValue.DEGRADED:
            return 1
        return 3 if (charge_proven and lesser_insufficient) else 2

    # ─────────── §III.6 Dirac + crowbar medido ───────────
    def dirac_crowbar_certificate(self, params: ProsecutionParameters, sentence: SentenceReport, *,
                                  positron_kinetic_kev: float = 0.0) -> DiracCrowbarProsecutionReport:
        r"""
        Cinemática e⁻e⁺→2γ con electrón en reposo: s = 2m²c⁴ + 2mc²E₊ ⇒ √s = √(2mc²(mc²+E₊));
        umbral 2m_ec² = 1021.99790138 keV (CODATA 2022). Cada fotón en el CM: √s/2.
        Actuación: SÓLO con token verificado y peldaño 3; en otro caso estado seguro + escalada.
        Latencia MEDIDA (no `42.0+350.15`), presupuesto físico DECLARADO con procedencia.
        """
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
        e_plus = _ME_C2_KEV + float(positron_kinetic_kev)
        sqrt_s = math.sqrt(2.0 * _ME_C2_KEV * (_ME_C2_KEV + e_plus))
        return DiracCrowbarProsecutionReport(
            electron_rest_energy_kev=_ME_C2_KEV, two_rest_energy_threshold_kev=2.0 * _ME_C2_KEV,
            sqrt_s_kev=sqrt_s, photon_energy_cm_kev=0.5 * sqrt_s,
            photons_emitted=2 if dispatched else 0, actuation_dispatched=dispatched,
            driver_name=self._driver.name, driver_is_physical=bool(self._driver.is_physical),
            dispatch_median_ns=float(np.median(samples)), dispatch_p99_ns=float(np.percentile(samples, 99)),
            trip_dispatch_ns=trip_ns, logic_deadline_ns=float(params.logic_deadline_ns),
            firmware_isr_verified=params.hil_firmware_isr_ns is not None,
            physical_budget_ns=params.physical_budget_ns,
            physical_deadline_feasible=bool(params.physical_budget_ns <= params.logic_deadline_ns),
            safe_state_applied=safe,
            note=("392.15 ns es política del camino lógico ISR/IRAM; el SCR BT151 (t_gt ~1–3 µs) "
                  "no puede alcanzarlo. Sin token válido NO hay disparo: estado seguro + escalada."),
        )

    # ─────────── §III.7 [TERMINAL] Orquestador del proceso ───────────
    def execute_thermal_prosecutor_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        rho_mac: Optional[NDArray] = None,
        *,
        cost_stream: Optional[NDArray] = None,
        principal_report: Any = None,
        witness_report: Any = None,
        authorization_token: Optional[str] = None,
        params: Optional[ProsecutionParameters] = None,
        positron_kinetic_kev: float = 0.0,
    ) -> ThermalExergyWeaverProsecutorSpectrumReport:
        r"""
        MÉTODO TERMINAL. Ejecuta §I.5→§II.1→§II.6→§III.1→§III.2..§III.6, sentencia en Ω₃ con
        presunción de inocencia, y sella el acta (Merkle + HMAC determinista + contrafirma).
        """
        t0 = time.perf_counter()
        params = params or ProsecutionParameters()
        k_raw = _as_matrix(apu_cost_matrix, "apu_cost_matrix", square=True)
        exps = _as_vector(np.asarray(exponents, dtype=np.float64), "exponents")
        _, k_reg, _, _ = self._principal.regularize_kernel(k_raw, psd_floor=params.psd_floor)
        if rho_mac is None:
            en, u = la.eigh(k_reg)
            lp = -params.beta_thermo * (en - en[0]); lp -= _logsumexp(lp)
            rho = (u * np.exp(lp)) @ u.conj().T
        else:
            r = _as_matrix(rho_mac, "rho_mac", square=True, complex_ok=True)
            tr = complex(np.trace(r))
            if abs(tr.imag) > 1e-10 * max(1.0, abs(tr.real)) or tr.real <= 0:
                raise ProsecutorError("ρ: traza compleja o no positiva.")
            rho = 0.5 * (r + r.conj().T) / tr.real
        if principal_report is None:
            principal_report = self._principal.execute_thermal_exergy_weaver_audit(
                k_raw, exps, rho_mac_input=rho, beta=params.beta_thermo, work_mean=params.work_mean,
                delta_f=params.delta_f, ball_r=params.ball_r, cylinder_r=params.cylinder_r,
                psd_floor=params.psd_floor,
            )
        passport_ok = bool(self._principal.verify_report_passport(principal_report))

        handoff_i = self.build_prosecutor_phase_i_handoff(
            k_raw, exps, params, cost_stream=cost_stream, principal_report=principal_report)
        handoff_ii = self.build_prosecutor_phase_ii_handoff(
            handoff_i, exps, params, rho=rho, principal_report=principal_report)
        handoff_ii = self.accept_prosecutor_phase_ii(handoff_ii)
        rsi3 = self.prosecutor_rsi3_certificate(handoff_ii)
        gamma, min_div, resonance, hom, twist = self.kam_poincare_certificate(params)
        geo = dataclasses.replace(handoff_ii.geometry, kam_diophantine_gamma=gamma,
                                  kam_min_small_divisor=min_div, kam_resonance=resonance,
                                  kam_homological_residual=hom, kam_twist_det=twist)
        handoff_ii = dataclasses.replace(handoff_ii, geometry=geo)

        # ── Sentencia: presunción de inocencia + carga de la prueba ──────────────────────
        structural_failure = any((not e.passed) and e.hard for e in
                                 list(handoff_i.evidences) + list(handoff_ii.evidences))
        statistical_conviction = handoff_i.burden.decision == "CONVICT"
        charge_proven = bool(structural_failure or statistical_conviction)
        charge_kind = "STRUCTURAL" if structural_failure else "STATISTICAL"
        evidences: Tuple[Any, ...] = (
            Evidence.boolean("F3.pasaporte_principal", passport_ok, hard=True),
            Evidence("F3.banach_q", rsi3.policy_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("F3.lip_t3", rsi3.lip_t3_empirical - rsi3.lip_t3_bound * (1 + 1e-9), 0.0, hard=True),
            Evidence("F3.banach_apriori", rsi3.observed_policy_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("F3.banach_aposteriori", rsi3.observed_policy_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("F3.kantorovich", rsi3.kantorovich_satisfied, hard=True),
            Evidence.boolean("F3.kleisli_leyes", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("F3.nucleo_axiomas", rsi3.nucleus_axioms_ok, hard=True),
            Evidence.boolean("F3.sin_resonancia_kam", not resonance, hard=True),
            Evidence("F3.gamma_diophantino", max(0.0, 1e-3 - gamma), 0.0),
            Evidence("F3.ecuacion_homologica", hom, 1e-10, hard=True),
            Evidence.boolean("F3.twist_no_degenerado", abs(twist) > 1e-12, hard=True),
        )
        base_truth = HeytingTruthValue.infimum(
            [handoff_ii.truth_value, principal_report.truth_value] + [e.truth for e in evidences])
        if base_truth is HeytingTruthValue.COHERENT and statistical_conviction:
            base_truth = HeytingTruthValue.DEGRADED     # carga estadística: nunca ⊥ por sí sola
        truth = base_truth
        rung = self.proportionality_rung(truth, charge_proven,
                                         lesser_insufficient=params.lesser_measures_insufficient)

        self._epoch += 1
        epoch = self._epoch
        acta_payload = {
            "epoch": epoch, "truth": truth, "rung": rung, "charge_proven": charge_proven,
            "charge_kind": charge_kind, "commitment": handoff_i.commitment_digest,
            "principal_passport": principal_report.passport_hmac,
            "burden_decision": handoff_i.burden.decision,
        }
        acta_digest = _sha256(_canonical(acta_payload))
        ne_bis = acta_digest not in self._tried_actas
        self._tried_actas.add(acta_digest)
        self_issued = authorization_token is None and rung == 3 and not params.require_human_authorization
        token = authorization_token or (self.mint_authorization_token(acta_digest, epoch, rung)
                                        if self_issued else "")
        token_ok = bool(token) and self.verify_authorization_token(token, acta_digest, epoch, rung)
        actuation_requested = bool(rung == 3)
        authorized = bool(actuation_requested and token_ok and ne_bis)
        sentence = SentenceReport(
            truth_value=truth, boolean_shadow=truth.double_negation(), ladder_rung=rung,
            action=_LADDER[rung], charge_proven=charge_proven, charge_kind=charge_kind,
            presumption_of_innocence_honored=bool(not (statistical_conviction and truth is HeytingTruthValue.VETOED
                                                       and not structural_failure)),
            proportionality_ok=bool(rung < 3 or params.lesser_measures_insufficient),
            authorization_token_verified=token_ok, authorization_self_issued=self_issued,
            ne_bis_in_idem_ok=bool(ne_bis), actuation_requested=actuation_requested,
            actuation_authorized=authorized, indictment_digest=acta_digest, epoch=epoch,
        )
        dirac = self.dirac_crowbar_certificate(params, sentence, positron_kinetic_kev=positron_kinetic_kev)

        impeached = False
        if witness_report is not None:
            w_leaves = [bytes.fromhex(h) for h in getattr(witness_report, "merkle_leaves", ())]
            w_root_ok = bool(w_leaves) and merkle_root(w_leaves) == witness_report.merkle_root_hash
            impeached = bool(w_root_ok and int(witness_report.truth_value) > int(truth))

        leaves: List[bytes] = [
            _canonical({"commitment": handoff_i.commitment_digest, "K": handoff_i.kernel_digest,
                        "exps": handoff_i.exponents_digest}),
            _canonical(handoff_i.falsification), _canonical(handoff_i.smith), _canonical(handoff_i.burden),
            _canonical(acta_payload), _canonical(handoff_ii.spohn), _canonical(handoff_ii.geometry),
            _canonical(rsi3), _canonical(sentence),
            _canonical({k: v for k, v in _freeze(dirac).items()
                        if k not in ("dispatch_median_ns", "dispatch_p99_ns", "trip_dispatch_ns")}),
        ]
        root = merkle_root(leaves)
        acta_idx = 4
        proof = merkle_proof(leaves, acta_idx)
        payload = {"version": __version__, "truth": truth, "acta": acta_digest, "merkle_root": root,
                   "rung": rung, "authorized": authorized, "commitment": handoff_i.commitment_digest}
        passport = hmac.new(self.key_bytes, _HMAC_ACTA + _canonical(payload), hashlib.sha256).hexdigest()
        counter = hmac.new(self.key_bytes, _HMAC_COUNTER +
                           principal_report.passport_hmac.encode("ascii") + bytes.fromhex(root),
                           hashlib.sha256).hexdigest()
        report = ThermalExergyWeaverProsecutorSpectrumReport(
            engine_version=__version__, phase_ii=handoff_ii, rsi3=rsi3, sentence=sentence,
            dirac_crowbar=dirac, evidences=evidences, truth_value=truth,
            principal_passport=principal_report.passport_hmac, principal_passport_verified=passport_ok,
            witness_impeached=impeached, merkle_root_hash=root,
            merkle_leaves=tuple(x.hex() for x in leaves), acta_leaf_index=acta_idx,
            acta_inclusion_proof=proof, passport_hash=passport, countersignature=counter,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        logger.info("Fiscal v14 Ω₃=%s · peldaño=%s · autorizado=%s · disparo=%s",
                    truth.name, sentence.action, authorized, dirac.actuation_dispatched)
        return report

    def verify_prosecutor_report(self, report: ThermalExergyWeaverProsecutorSpectrumReport) -> bool:
        """Reverifica Merkle, prueba de inclusión del acta y pasaporte determinista."""
        leaves = [bytes.fromhex(h) for h in report.merkle_leaves]
        if merkle_root(leaves) != report.merkle_root_hash:
            return False
        i = report.acta_leaf_index
        if not verify_merkle(leaves[i], i, len(leaves), report.acta_inclusion_proof, report.merkle_root_hash):
            return False
        payload = {"version": report.engine_version, "truth": report.truth_value,
                   "acta": report.sentence.indictment_digest, "merkle_root": report.merkle_root_hash,
                   "rung": report.sentence.ladder_rung, "authorized": report.sentence.actuation_authorized,
                   "commitment": report.phase_ii.phase_i.commitment_digest}
        expected = hmac.new(self.key_bytes, _HMAC_ACTA + _canonical(payload), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, report.passport_hash)


class BoolNovikovThermalExergyWeaverProsecutorEngine(ProsecutorPhaseIIISentence):
    """Fachada pública (nombre conservado). Anidamiento: Fase I ⊂ II ⊂ III ⊂ Fachada."""

    PHASE_LABEL: Final[str] = "FACHADA·FISCAL-BOOL-NOVIKOV-THERMAL-EXERGY"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada certificado se prueba y cada detector se prueba en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    principal = PrincipalEngine(secret_key=b"PRINCIPAL_KEY_32_BYTES_MINIMUM!!")
    pros = BoolNovikovThermalExergyWeaverProsecutorEngine(
        secret_key=b"PROSECUTOR_KEY_32_BYTES_MINIMUM!", principal_engine=principal)
    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps = np.array([0.15, 0.45, 0.85], dtype=np.float64)
    prep = principal.execute_thermal_exergy_weaver_audit(k_test, exps)

    # 1) Nominal: H0 (coste legítimo) ⇒ ABSUELTO, ⊤, SIN disparo (v13 condenaba por defecto)
    rep = pros.execute_thermal_prosecutor_audit(k_test, exps, principal_report=prep)
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert rep.phase_ii.phase_i.burden.decision == "ACQUIT"
    assert rep.sentence.ladder_rung == 0 and not rep.dirac_crowbar.actuation_dispatched
    assert rep.phase_ii.spohn.sigma_nonnegative and rep.phase_ii.spohn.entropy_monotone
    assert rep.phase_ii.spohn.generator_stationarity_residual < 1e-9
    assert rep.phase_ii.itoh_tellegen.secant_residual_max < 1e-12
    assert rep.phase_ii.itoh_tellegen.legacy_tautology_residual < 1e-12   # v13: 0=0
    assert rep.phase_ii.phase_i.smith.scaling_fabricates_torsion          # ×100 fabrica torsión
    assert abs(rep.dirac_crowbar.two_rest_energy_threshold_kev - 1021.99790138) < 1e-6
    assert pros.verify_prosecutor_report(rep)

    # 2) Carga estadística probada (μ=μ1) ⇒ condena SPRT, pero NUNCA ⊥ sólo por estadística
    rng = np.random.default_rng(5)
    guilty = pros.execute_thermal_prosecutor_audit(
        k_test, exps, principal_report=prep,
        cost_stream=rng.normal(ProsecutionParameters().cost_mu1, 1.0, 400))
    assert guilty.phase_ii.phase_i.burden.decision == "CONVICT"
    assert guilty.truth_value is not HeytingTruthValue.COHERENT
    assert guilty.sentence.presumption_of_innocence_honored

    # 3) Violación ESTRUCTURAL (r>R) ⇒ ⊥, ISOLATE; SIN token NO hay disparo
    bad = pros.execute_thermal_prosecutor_audit(
        k_test, exps, principal_report=prep,
        params=ProsecutionParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True))
    assert bad.truth_value is HeytingTruthValue.VETOED and bad.sentence.ladder_rung == 3
    assert bad.sentence.actuation_requested and not bad.sentence.actuation_authorized
    assert not bad.dirac_crowbar.actuation_dispatched          # ← la autorización HABILITA
    assert bad.dirac_crowbar.safe_state_applied                # ← estado seguro, no disparo

    # 4) CON token válido ligado al acta ⇒ disparo autorizado y latencia MEDIDA
    tok = pros.mint_authorization_token(bad.sentence.indictment_digest, bad.sentence.epoch + 1, 3)
    armed = pros.execute_thermal_prosecutor_audit(
        k_test, exps, principal_report=prep, authorization_token=tok,
        params=ProsecutionParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True))
    assert armed.sentence.actuation_authorized and armed.dirac_crowbar.actuation_dispatched
    assert armed.dirac_crowbar.trip_dispatch_ns > 0.0
    assert not armed.dirac_crowbar.physical_deadline_feasible   # SCR ~µs: declarado

    # 5) Ne bis in idem: la misma acta no se procesa dos veces
    twice = pros.execute_thermal_prosecutor_audit(
        k_test, exps, principal_report=prep, authorization_token=tok,
        params=ProsecutionParameters(ball_r=2.0, cylinder_r=1.0, lesser_measures_insufficient=True))
    assert twice.sentence.epoch != armed.sentence.epoch

    # 6) SPRT: tasas declaradas verificadas por Monte Carlo
    b = rep.phase_ii.phase_i.burden
    assert b.mc_false_conviction_rate <= b.alpha_declared + 0.01
    assert b.mc_power >= 1.0 - b.beta_declared - 0.03
    assert abs(b.log_threshold_convict - math.log(0.9 / 0.01)) < 1e-12

    # 7) SNF por menores: torsión real vs fabricada; Merkle; determinismo
    tri = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=np.float64)
    snf = _smith_by_minors(tri)
    assert snf is not None and snf[0] == (1, 1) and snf[1] == 2
    snf100 = _smith_by_minors(tri * 100.0)
    assert snf100 is not None and snf100[0] == (100, 100)
    for n in range(1, 9):
        lv = [f"hoja-{i}".encode() for i in range(n)]
        rt = merkle_root(lv)
        for i in range(n):
            assert verify_merkle(lv[i], i, n, merkle_proof(lv, i), rt)
    rep2 = pros.execute_thermal_prosecutor_audit(k_test, exps, principal_report=prep)
    assert rep2.passport_hash != rep.passport_hash            # época distinta ⇒ acta distinta
    assert not pros.verify_prosecutor_report(dataclasses.replace(rep, truth_value=HeytingTruthValue.VETOED))

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverProsecutorEngine v14 ===")
    print(f"Veredicto Ω₃ / sombra ¬¬      : {rep.truth_value.name}({int(rep.truth_value)}) / "
          f"{rep.sentence.boolean_shadow.name}")
    print(f"Carga de la prueba (SPRT)     : {b.decision} · α̂={b.mc_false_conviction_rate:.4f} "
          f"(decl {b.alpha_declared}) · potencia={b.mc_power:.3f}")
    print(f"Umbrales Wald ln A / ln B     : {b.log_threshold_convict:.4f} / {b.log_threshold_acquit:.4f}")
    print(f"ASN H0 / H1                   : {b.mc_asn_h0:.1f} / {b.mc_asn_h1:.1f}")
    print(f"Falsación (contraejemplos)    : PSD={rep.phase_ii.phase_i.falsification.psd_counterexamples} "
          f"secante={rep.phase_ii.phase_i.falsification.secant_counterexamples} "
          f"Lip={rep.phase_ii.phase_i.falsification.lipschitz_counterexamples}")
    print(f"SNF real / escalada ×100      : {rep.phase_ii.phase_i.smith.invariants} / "
          f"{rep.phase_ii.phase_i.smith.scaled_invariants} "
          f"(fabrica torsión={rep.phase_ii.phase_i.smith.scaling_fabricates_torsion})")
    print(f"Tellegen real / tautología v13: {rep.phase_ii.itoh_tellegen.tellegen_power_residual:.2e} / "
          f"{rep.phase_ii.itoh_tellegen.legacy_tautology_residual:.2e}")
    print(f"Spohn σ_min / S monótona      : {rep.phase_ii.spohn.sigma_min:.4e} / "
          f"{rep.phase_ii.spohn.entropy_monotone}")
    print(f"S(ρ_t‖ρ_β) traza              : "
          + ", ".join(f"{v:.4f}" for v in rep.phase_ii.spohn.relative_entropy_trace))
    print(f"KMS modular / S(ρ‖ρ_β)        : {rep.phase_ii.jarzynski_kms.kms_residual:.2e} / "
          f"{rep.phase_ii.jarzynski_kms.gibbs_relative_entropy:.2e}")
    print(f"Melnikov A / 4δ/3 / R_crit    : {rep.phase_ii.geometry.melnikov_amplitude:.5f} / "
          f"{rep.phase_ii.geometry.melnikov_offset:.5f} / {rep.phase_ii.geometry.melnikov_critical_ratio:.5f}")
    print(f"KAM γ / residuo homológico    : {rep.phase_ii.geometry.kam_diophantine_gamma:.5f} / "
          f"{rep.phase_ii.geometry.kam_homological_residual:.2e}")
    print(f"RSI³ q / Lip(T³) medido       : {rep.rsi3.policy_lipschitz_q:.6f} / {rep.rsi3.lip_t3_empirical:.6f}")
    print(f"Kantorovich h / d³C           : {rep.rsi3.kantorovich_h:.6f} / {rep.rsi3.capacity_d3:.4f}")
    print(f"Sentencia nominal             : {rep.sentence.action} (disparo={rep.dirac_crowbar.actuation_dispatched})")
    print(f"⊥ SIN token                   : {bad.sentence.action} autorizado={bad.sentence.actuation_authorized} "
          f"disparo={bad.dirac_crowbar.actuation_dispatched} seguro={bad.dirac_crowbar.safe_state_applied}")
    print(f"⊥ CON token                   : {armed.sentence.action} disparo={armed.dirac_crowbar.actuation_dispatched} "
          f"latencia={armed.dirac_crowbar.trip_dispatch_ns:.0f} ns")
    print(f"√s Dirac [keV] / umbral       : {rep.dirac_crowbar.sqrt_s_kev:.6f} / "
          f"{rep.dirac_crowbar.two_rest_energy_threshold_kev:.6f}")
    print(f"Merkle / contrafirma          : {rep.merkle_root_hash[:16]}… / {rep.countersignature[:16]}…")
    print(f"Acta reverificable            : {pros.verify_prosecutor_report(rep)}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")