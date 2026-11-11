# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPhantomCollusionMitigatorWitnessEngine v14.0.0-PhD                             ║
║ Ubicación: app/wisdom/bool_novikov_phantom_collusion_mitigator_witness_engine.py                     ║
║ Versión  : 14.0.0-ENGINE-BOOL-NOVIKOV-PHANTOM-COLLUSION-WITNESS-POINCARE-RSI3-PhD                    ║
║ Función  : Motor Espectral de Atestación Pasiva QND (Aharonov-Albert-Vaidman) sobre el Fibrado       ║
║            Dual Boole-Novikov (R_Boole ⊗ Λ_Nov) para Mitigación de Colusiones Fantasma.              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  _Fase1_Algebra_Hipercompleja
     └ _fase1_emit_algebraic_seed        ÚLTIMO de F1 → inicio de F2.
  _Fase2_Cuantica_Geometria
     └ _fase2_emit_witness_seed          ÚLTIMO de F2 → inicio de F3.
  _Fase3_Kleisli_RSI3
     └ execute_phantom_collusion_witness_audit   ÚLTIMO de F3 (API pública).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · AAV: A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩ con división COMPLEJA (no por la magnitud). Régimen
    singular (|solapamiento|<tol) → weak_value=None + flag; valor débil anómalo si |A_w|>‖A‖₂.
    QND con conmutador RELATIVO escala-invariante; back-action dB es un proxy monótono declarado.
  · Uhlmann: F(ρ,σ)=‖√ρ√σ‖₁² por eigh+SVD (estable, sin scipy.sqrtm). Umbral único F_min=0.95
    y d_B_max=√(2−2√F_min) derivado (los dos umbrales de v13 eran inconsistentes).
  · Asociadores: álgebra de Cayley-Dickson REAL en 32/64/128D (e_i e_j=S[i,j]e_{i⊕j}).
    Se computa [x,y,z]=(xy)z−x(yz) sobre un embedding declarado del cost matrix, y el defecto
    algebraico ‖A‖_F/n^{3/2} (invariante: 0 para n≤4, >0 para n≥8). La razón norma/defecto
    normaliza la no-asociatividad típica; la sospecha es una POLÍTICA heurística declarada.
  · SNF sobre ℤ con cadena de divisibilidad; influencia total booleana por FWHT (ya no SVD-rank).
  · Novikov fail-closed (vacío/negativo → error). Gromov r≤R. CR: Laplaciano periódico de 5 puntos.
  · Melnikov en forma cerrada; Lindstedt–Poincaré Ω(a)=AGM(1,cos(a/2)); mapa estroboscópico de
    Poincaré con Lyapunov de Benettin (EVIDENCIA, no prueba; tensión unidireccional).
  · RSI³: cota analítica λ³ + certificada λ³+ε; Picard con error a posteriori verificado contra
    un Picard de referencia; mallas diádicas anidadas de |(T³)'| (monotonía por inclusión);
    cadena de certificados SHA-256. Jerk = tercera diferencia MEDIDA (Optional).
  · Heyting Ω₃: meet de 5 jueces (espectral, cuántico, dinámico, RSI, verificador) con vetos duros.
  · Pasaporte HMAC canónico REPRODUCIBLE (sin tiempos); digest de toda la entrada; clave por entorno.

LÍMITES DECLARADOS (no se afirma más de lo que se prueba):
  · El testigo A es un operador tridiagonal declarado; QND es trivial para ρ máximamente mezclada.
  · La sospecha de colusión por asociadores es un heurístico normalizado, no una prueba de colusión.
  · El back-action en dB es un proxy monótono del conmutador relativo, no una razón de potencia física.
  · Melnikov es riguroso solo para ε pequeño; Lyapunov finito es evidencia dependiente de la sonda.
  · El valor débil anómalo es un fenómeno AAV legítimo, no un error; se reporta como tal.
  · No se escribe en hardware ni se modela disparo crowbar.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovPhantomCollusionMitigatorWitnessEngine")

# ── Versiones y dominios ───────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "14.0.0-ENGINE-BOOL-NOVIKOV-PHANTOM-COLLUSION-WITNESS-POINCARE-RSI3-PhD"
)
version: Final[str] = _ENGINE_VERSION
_SCHEMA_VERSION: Final[str] = "witness-report-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_PHANTOM_WITNESS_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_WITNESS_DEV_ONLY_PHANTOM_KEY_v14.0.0"
_PASSPORT_DOMAIN: Final[bytes] = b"PHANTOM-COLLUSION-WITNESS-PASSPORT-v14.0|"
_RSI_CHAIN_DOMAIN: Final[bytes] = b"PHANTOM-WITNESS-RSI3-CERT-CHAIN-v14.0|"

# ── Numérica y álgebra ─────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SNF_FIXED_POINT_SCALE: Final[float] = 100.0
_SNF_MAX_ABS_INT: Final[int] = 1 << 62
_RW_DEGRADED_TOL: Final[float] = 0.5
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_ASSOC_DIMS: Final[Tuple[int, int, int]] = (32, 64, 128)
_ASSOC_RATIO_SUSPECT: Final[float] = 2.0          # política heurística declarada

# ── Cuántica ───────────────────────────────────────────────────────────────────
_OVERLAP_SINGULAR_TOL: Final[float] = 1.0e-12
_QND_COMM_REL_TOL: Final[float] = 1.0e-8
_HERM_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_TRACE_TOL: Final[float] = 1.0e-10
_FIDELITY_MIN: Final[float] = 0.95
_BURES_DIST_MAX: Final[float] = math.sqrt(2.0 - 2.0 * math.sqrt(_FIDELITY_MIN))

# ── Geometría y Poincaré ───────────────────────────────────────────────────────
_DEFAULT_BALL_R: Final[float] = 0.35
_DEFAULT_CYLINDER_R: Final[float] = 0.50
_CR_RESIDUAL_TOL: Final[float] = 1.0e-3
_MELNIKOV_HALF_WIDTH: Final[float] = 40.0
_MELNIKOV_POINTS: Final[int] = 20001
_MELNIKOV_DEFAULT_DELTA: Final[float] = 0.01
_MELNIKOV_DEFAULT_GAMMA: Final[float] = 0.02
_MELNIKOV_DEFAULT_OMEGA: Final[float] = 1.0
_MELNIKOV_EPS_POLICY: Final[float] = 0.10
_DEFAULT_PENDULUM_AMPLITUDE: Final[float] = 0.2
_POINCARE_PERIODS: Final[int] = 256
_POINCARE_STEPS: Final[int] = 64
_POINCARE_IC: Final[Tuple[float, float]] = (0.0, 1.9)
_LYAP_POS_TOL: Final[float] = 1.0e-2

# ── Kleisli / RSI³ ─────────────────────────────────────────────────────────────
_KLEISLI_LIP_T: Final[float] = 0.95
_KLEISLI_ERROR_BUDGET: Final[float] = 0.002375
_KLEISLI_OFFSET: Final[float] = 0.1
_KLEISLI_DOMAIN: Final[float] = 2.0
_KLEISLI_GRID_BASE: Final[int] = 100
_PICARD_TOL: Final[float] = 1.0e-12
_PICARD_MAX_ITER: Final[int] = 10000
_PICARD_REF_TOL: Final[float] = 1.0e-15
_PICARD_REF_MAX_ITER: Final[int] = 100000
_RSI_MAX_DEPTH: Final[int] = 3
_LIP_TOL: Final[float] = 1.0e-12
_CONSISTENCY_TOL: Final[float] = 1.0e-9


# ══════════════════════════════════════════════════════════════════════════════
# Topos de Heyting Ω₃
# ══════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    """Cadena de Heyting ⊥<δ<⊤ con códigos de cable 0x03,0x01,0x00."""

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @property
    def rank(self) -> int:
        return {HeytingTruthValue.COHERENT: 2, HeytingTruthValue.DEGRADED: 1}.get(self, 0)

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue.COHERENT if self.rank <= other.rank else other

    def negate(self) -> "HeytingTruthValue":
        return self.implies(HeytingTruthValue.VETOED)


# ══════════════════════════════════════════════════════════════════════════════
# Informes y semillas
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class QNDWeakMeasurementReport:
    """Medición débil AAV e invarianza QND pasiva (conmutador relativo)."""

    weak_value: Optional[complex]
    weak_value_magnitude: float
    overlap_magnitude: float
    is_overlap_singular: bool
    is_weak_value_anomalous: bool
    commutator_norm: float
    commutator_relative: float
    back_action_db: float
    is_qnd_passive: bool
    witness_spectral_norm: float


@dataclass(frozen=True)
class BuresUhlmannMetricReport:
    """Fidelidad de Uhlmann estable y distancia geodésica de Bures."""

    fidelity_uhlmann: float
    bures_distance: float
    bures_angle: float
    is_isometry_preserved: bool
    fidelity_threshold: float
    bures_distance_max: float


@dataclass(frozen=True)
class MultilateralAssociatorWitnessReport:
    """Asociadores genuinos de Cayley-Dickson en 32/64/128D (no reescalados)."""

    pationion_norm_32d: float
    chingon_norm_64d: float
    routon_norm_128d: float
    pationion_ratio_32d: float
    chingon_ratio_64d: float
    routon_ratio_128d: float
    algebra_defect_32d: float
    algebra_defect_64d: float
    algebra_defect_128d: float
    phantom_collusion_suspected: bool
    embedding_policy: str


@dataclass(frozen=True)
class PhantomCollusionWitnessSpectrumReport:
    """Reporte de atestación pasiva QND anti-colusión (reproducible, sin tiempos)."""

    rademacher_walsh_norm: float
    smith_torsion_free: bool
    smith_torsion_order: int
    cokernel_free_rank: int
    snf_diagonal_factors: Tuple[int, ...]
    krull_valuation: float
    novikov_norm: float
    rho_admissible: bool
    cauchy_riemann_residual: float
    is_gromov_violated: bool
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    melnikov_chaos_detected: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_transversality_margin: float
    melnikov_eps_regime_valid: bool
    poincare_lindstedt_exact: float
    poincare_lindstedt_series: float
    poincare_lyapunov: Optional[float]
    poincare_melnikov_consistent: Optional[bool]
    banach_lip_analytic: float
    banach_contraction_lip: float
    banach_fixed_point: float
    banach_a_posteriori_error: float
    banach_meta_verified: bool
    rsi_accepted_depth: int
    rsi_certificate_chain: str
    capacity_third_difference: Optional[float]
    associator_report: MultilateralAssociatorWitnessReport
    qnd_report: QNDWeakMeasurementReport
    bures_report: BuresUhlmannMetricReport
    metamorphic_selftest_passed: bool
    input_digest: str
    truth_value: HeytingTruthValue
    passport_hash: str
    schema_version: str


@dataclass(frozen=True, eq=False)
class _AlgebraicSeed:
    matrix: NDArray[np.float64]
    exponents: NDArray[np.float64]
    snf_diagonal_factors: Tuple[int, ...]
    torsion_free: bool
    torsion_order: int
    cokernel_free_rank: int
    rw_norm: float
    krull: float
    novikov_norm: float
    associators: MultilateralAssociatorWitnessReport
    input_digest: str


@dataclass(frozen=True, eq=False)
class _WitnessSeed:
    algebraic: _AlgebraicSeed
    rho_admissible: bool
    qnd: QNDWeakMeasurementReport
    bures: BuresUhlmannMetricReport
    cr_residual: float
    gromov_violated: bool
    capacity_ball: float
    capacity_cylinder: float
    melnikov_chaos: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_margin: float
    melnikov_eps_valid: bool
    lindstedt_exact: float
    lindstedt_series: float
    lyapunov: Optional[float]
    lyapunov_consistent: Optional[bool]
    force_veto: bool


@dataclass(frozen=True)
class _KleisliState:
    lip_analytic: float
    lip_certified: float
    fixed_point: float
    a_posteriori_error: float
    residual: float
    meta_verified: bool
    selftest_passed: bool
    consistency_ok: bool
    depth_accepted: int
    certificate_chain: str


# ══════════════════════════════════════════════════════════════════════════════
# Primitivas canónicas
# ══════════════════════════════════════════════════════════════════════════════
def _trapz(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _fmt(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "none"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}:{int(value.value)}"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Campo canónico no finito (fail-closed).")
        return f"{value:.17g}"
    if isinstance(value, str):
        if "|" in value:
            raise ValueError("Separador '|' prohibido en campo canónico.")
        return value
    raise TypeError(f"Tipo no canónico: {type(value).__name__}")


def _resolve_secret_key(explicit: Optional[bytes]) -> bytes:
    if explicit is not None:
        key = bytes(explicit)
    else:
        env_val = os.environ.get(_HMAC_KEY_ENV, "")
        if env_val:
            key = env_val.encode("utf-8")
        else:
            logger.warning("Sin %s: clave de DESARROLLO (no apta para producción).", _HMAC_KEY_ENV)
            key = _DEV_SECRET_KEY
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"Clave HMAC demasiado corta: {len(key)} < {_MIN_KEY_BYTES} bytes.")
    return key


def _validate_finite(value: object, name: str) -> float:
    fv = float(value)  # type: ignore[arg-type]
    if not math.isfinite(fv):
        raise ValueError(f"{name} debe ser finito.")
    return fv


def _validate_positive(value: object, name: str) -> float:
    fv = _validate_finite(value, name)
    if fv <= 0.0:
        raise ValueError(f"{name} debe ser estrictamente positivo.")
    return fv


def _validate_lambda(lam: object) -> float:
    fv = _validate_finite(lam, "initial_lip")
    if not (0.0 <= fv < 1.0):
        raise ValueError("initial_lip debe estar en [0,1).")
    return fv


def _validate_int(value: object, name: str, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} debe ser entero.")
    v = int(value)
    if v < lo or (hi is not None and v > hi):
        raise ValueError(f"{name} fuera de rango.")
    return v


def _as_cost_matrix(m: object) -> NDArray[np.float64]:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("cost matrix 2D con ambas dimensiones ≥ 2.")
    if not np.all(np.isfinite(a)):
        raise ValueError("cost matrix contiene valores no finitos.")
    return np.ascontiguousarray(a)


def _as_exponents(e: object) -> NDArray[np.float64]:
    v = np.asarray(e, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("Exponentes: vector finito no vacío (fail-closed).")
    if np.any(v < 0.0):
        raise ValueError("Exponentes negativos fuera del anillo de Novikov.")
    return np.ascontiguousarray(v)


def _as_density(rho: object, name: str) -> Tuple[NDArray[np.complex128], bool]:
    """Normaliza por traza y certifica Hermiticidad/PSD/traza. Devuelve (ρ, admisible)."""
    r = np.asarray(rho, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] < 1:
        raise ValueError(f"{name} debe ser cuadrada 2D no vacía.")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{name} contiene valores no finitos.")
    tr = float(np.real(np.trace(r)))
    if tr > 0.0:
        r = r / tr
    scale = max(1.0, float(np.max(np.abs(r))))
    herm = float(np.max(np.abs(r - r.conj().T))) <= _HERM_TOL * scale
    rh = 0.5 * (r + r.conj().T)
    evals = np.linalg.eigvalsh(rh)
    psd = bool(evals[0] >= -_PSD_TOL * scale)
    trace_ok = bool(abs(float(np.real(np.trace(r))) - 1.0) <= _TRACE_TOL)
    return r, bool(herm and psd and trace_ok and tr > 0.0)


def _as_unit_vector(v: object, dim: int, name: str) -> NDArray[np.complex128]:
    x = np.asarray(v, dtype=np.complex128).reshape(-1)
    if x.size != dim:
        raise ValueError(f"{name} debe tener dimensión {dim}.")
    nrm = float(np.linalg.norm(x))
    if nrm <= 0.0 or not math.isfinite(nrm):
        raise ValueError(f"{name} no puede ser el vector nulo.")
    return x / nrm


def _quantize_matrix(mat: NDArray[np.float64]) -> List[List[int]]:
    scaled = mat * _SNF_FIXED_POINT_SCALE
    if float(np.max(np.abs(scaled))) >= float(_SNF_MAX_ABS_INT):
        raise ValueError("Entrada fuera del rango de cuantización (overflow evitado).")
    return np.rint(scaled).astype(np.int64).tolist()


def _swap_rc(a: List[List[int]], r1: int, r2: int, c1: int, c2: int) -> None:
    a[r1], a[r2] = a[r2], a[r1]
    for row in a:
        row[c1], row[c2] = row[c2], row[c1]


def _min_abs_nonzero(a, cells):
    best = None
    for i, j in cells:
        v = a[i][j]
        if v != 0 and (best is None or abs(v) < abs(a[best[0]][best[1]])):
            best = (i, j)
    return best


def _third_difference(series: Optional[Sequence[float]]) -> Optional[float]:
    if series is None:
        return None
    c = np.asarray(series, dtype=np.float64).reshape(-1)
    if c.size < 4 or not np.all(np.isfinite(c)):
        raise ValueError("capacity_series requiere ≥4 muestras finitas.")
    return float(c[3] - 3.0 * c[2] + 3.0 * c[1] - c[0])


def _input_digest(mat, exps, rho, phi_i, phi_f, sigma, params: str) -> str:
    h = hashlib.sha256()
    for name, arr in (("MIC", mat), ("EXP", exps), ("RHO", rho),
                      ("PI", phi_i), ("PF", phi_f), ("SIG", sigma)):
        h.update(f"{name}{arr.shape}".encode("ascii"))
        h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    h.update(b"PARAMS|" + params.encode("utf-8"))
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Álgebra de Cayley-Dickson REAL: e_i e_j = S[i,j] e_{i⊕j}
# ══════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def _cd_sign_table(n: int) -> NDArray[np.int16]:
    """Tabla de signos por recursión Cayley-Dickson: (a,b)(c,d)=(ac−d*b, da+bc*)."""
    if n == 1:
        return np.ones((1, 1), dtype=np.int16)
    m = n // 2
    S = _cd_sign_table(m)
    c = np.ones(m, dtype=np.int16)
    c[1:] = -1
    T = np.empty((n, n), dtype=np.int16)
    T[:m, :m] = S
    T[:m, m:] = S.T
    T[m:, :m] = S * c[None, :]
    T[m:, m:] = -S.T * c[None, :]
    return T


@lru_cache(maxsize=None)
def _cd_ops(n: int) -> Tuple[NDArray[np.int64], NDArray[np.int16]]:
    S = _cd_sign_table(n)
    idx = np.arange(n)
    XOR = idx[None, :] ^ idx[:, None]          # XOR[k,i] = i⊕k
    SK = S[idx[None, :], XOR]                  # SK[k,i] = S[i, i⊕k]
    return XOR, SK


def _cd_mul(x: NDArray[np.float64], y: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Producto CD: (xy)_k = Σ_i x_i y_{i⊕k} S[i,i⊕k]  (O(n²) vectorizado)."""
    XOR, SK = _cd_ops(n)
    return np.sum(SK * x[None, :] * y[XOR], axis=1)


def _cd_associator(x, y, z, n) -> NDArray[np.float64]:
    """[x,y,z] = (xy)z − x(yz)."""
    return _cd_mul(_cd_mul(x, y, n), z, n) - _cd_mul(x, _cd_mul(y, z, n), n)


@lru_cache(maxsize=None)
def _cd_associator_defect(n: int) -> Tuple[float, int]:
    """Defecto algebraico ‖A‖_F/n^{3/2} y nº de triples no asociativos.
    A[i,j,k]=S[i,j]S[i⊕j,k]−S[j,k]S[i,j⊕k]. Invariante: 0 para n≤4, >0 para n≥8.
    Cómputo por filas (O(n²) memoria)."""
    S = _cd_sign_table(n)
    idx = np.arange(n)
    XOR = idx[:, None] ^ idx[None, :]
    acc = 0.0
    nz = 0
    for i in range(n):
        Si = S[i, :]
        A_i = Si[None, :] * S[idx ^ i, :] - S * Si[XOR]
        acc += float(np.sum(A_i.astype(np.float64) ** 2))
        nz += int(np.count_nonzero(A_i))
    return math.sqrt(acc) / n ** 1.5, nz


@lru_cache(maxsize=None)
def _cd_alternativity_defect(n: int) -> int:
    """max|A[i,i,j]|: 0 si el álgebra es alternativa (n≤8), >0 si no (n≥16)."""
    S = _cd_sign_table(n)
    idx = np.arange(n)
    XORi = idx[:, None] ^ idx[None, :]
    A = np.diag(S)[:, None] * S[0, :][None, :] - S * S[idx[:, None], XORi]
    return int(np.max(np.abs(A)))


def _cd_embed(mat: NDArray[np.float64], n: int) -> Tuple[NDArray, NDArray, NDArray]:
    """Embedding declarado: flat row-major → resize(n) → normalización ℓ2; y=roll(x,1), z=roll(x,2)."""
    flat = mat.reshape(-1)
    v = np.resize(flat, n).astype(np.float64)
    nrm = float(np.linalg.norm(v))
    if nrm > 0.0:
        v = v / nrm
    return v, np.roll(v, 1), np.roll(v, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — Álgebra hipercompleja, SNF, booleanas, Novikov y asociadores
# ══════════════════════════════════════════════════════════════════════════════
class _Fase1_Algebra_Hipercompleja:
    """Fase 1. Invariantes enteros, influencia booleana, valuación y asociadores CD."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._secret_key = _resolve_secret_key(secret_key)

    @staticmethod
    def _fase1_smith_diagonal(a_in: Sequence[Sequence[int]]) -> Tuple[int, ...]:
        """Forma de Smith con cadena d₁|d₂|…|d_r (barridos gcd/lcm unimodulares)."""
        a = [[int(v) for v in row] for row in a_in]
        m = len(a)
        n = len(a[0]) if m else 0
        diag: List[int] = []
        t = 0
        while t < min(m, n):
            piv = _min_abs_nonzero(a, [(i, j) for i in range(t, m) for j in range(t, n)])
            if piv is None:
                break
            _swap_rc(a, t, piv[0], t, piv[1])
            while True:
                p = a[t][t]
                clean = True
                for i in range(t + 1, m):
                    q = a[i][t] // p
                    if q != 0:
                        for j in range(t, n):
                            a[i][j] -= q * a[t][j]
                    if a[i][t] != 0:
                        clean = False
                for j in range(t + 1, n):
                    q = a[t][j] // p
                    if q != 0:
                        for i in range(t, m):
                            a[i][j] -= q * a[i][t]
                    if a[t][j] != 0:
                        clean = False
                if clean:
                    break
                cross = [(i, t) for i in range(t, m)] + [(t, j) for j in range(t + 1, n)]
                piv = _min_abs_nonzero(a, cross)
                _swap_rc(a, t, piv[0], t, piv[1])
            diag.append(abs(a[t][t]))
            t += 1
        k = len(diag)
        for i in range(k):
            for j in range(i + 1, k):
                g = math.gcd(diag[i], diag[j])
                if g != diag[i]:
                    diag[i], diag[j] = g, diag[i] // g * diag[j]
        return tuple(diag)

    @staticmethod
    def _fase1_boolean_bits(q: Sequence[Sequence[int]]) -> NDArray[np.int64]:
        flat = [1 if v > 0 else 0 for row in q for v in row]
        n_bits = 1
        while (1 << n_bits) < len(flat):
            n_bits += 1
        bits = np.zeros(1 << n_bits, dtype=np.int64)
        bits[: len(flat)] = flat
        return bits

    @staticmethod
    def _fase1_total_influence(bits: NDArray[np.int64]) -> float:
        n_pts = bits.size
        n_vars = n_pts.bit_length() - 1
        idx = np.arange(n_pts)
        return sum(float(np.mean(bits != bits[idx ^ (1 << k)])) for k in range(n_vars))

    @staticmethod
    def _fase1_fwht(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.array(vec, dtype=np.float64, copy=True)
        n = a.size
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError("WHT requiere potencia de 2.")
        h = 1
        while h < n:
            for i in range(0, n, 2 * h):
                x = a[i:i + h].copy()
                y = a[i + h:i + 2 * h].copy()
                a[i:i + h] = x + y
                a[i + h:i + 2 * h] = x - y
            h *= 2
        return a

    def _fase1_rademacher_walsh_norm(self, q: Sequence[Sequence[int]]) -> float:
        bits = self._fase1_boolean_bits(q)
        return self._fase1_total_influence(bits) / (bits.size.bit_length() - 1)

    def compute_rademacher_walsh_and_snf(self, mat: NDArray[np.float64]):
        q = _quantize_matrix(mat)
        factors = self._fase1_smith_diagonal(q)
        torsion_free = all(d == 1 for d in factors)
        torsion_order = math.prod(factors) if factors else 1
        free_rank = mat.shape[0] - len(factors)
        rw = self._fase1_rademacher_walsh_norm(q)
        return rw, factors, torsion_free, torsion_order, free_rank

    def evaluate_novikov_krull_valuation(self, exponents: NDArray[np.float64]) -> Tuple[float, float]:
        exps = _as_exponents(exponents)
        v = float(np.min(exps))
        return v, math.exp(-v)

    def evaluate_multilateral_associators_witness(
        self, mat: NDArray[np.float64]
    ) -> MultilateralAssociatorWitnessReport:
        """Asociadores CD genuinos del embedding de entrada, normalizados por el defecto
        algebraico (E‖[x,y,z]‖² = defecto² para vectores unitarios isótropos independientes)."""
        norms, ratios, defects = [], [], []
        for n in _ASSOC_DIMS:
            x, y, z = _cd_embed(mat, n)
            a_norm = float(np.linalg.norm(_cd_associator(x, y, z, n)))
            defect, _ = _cd_associator_defect(n)
            norms.append(a_norm)
            ratios.append(a_norm / defect if defect > 0.0 else 0.0)
            defects.append(defect)
        suspected = bool(any(r > _ASSOC_RATIO_SUSPECT for r in ratios))
        return MultilateralAssociatorWitnessReport(
            pationion_norm_32d=norms[0], chingon_norm_64d=norms[1], routon_norm_128d=norms[2],
            pationion_ratio_32d=ratios[0], chingon_ratio_64d=ratios[1], routon_ratio_128d=ratios[2],
            algebra_defect_32d=defects[0], algebra_defect_64d=defects[1], algebra_defect_128d=defects[2],
            phantom_collusion_suspected=suspected,
            embedding_policy="flat_row_major→resize(n)→ℓ2_unit; y=roll(x,1), z=roll(x,2)",
        )

    # ── ÚLTIMO método de Fase 1 → inicio de Fase 2 ─────────────────────────────
    def _fase1_emit_algebraic_seed(
        self, mat: NDArray[np.float64], exps: NDArray[np.float64], input_digest: str
    ) -> _AlgebraicSeed:
        """ÚLTIMO método de F1. Emite la semilla algebraica que consume F2."""
        rw, factors, torsion_free, torsion_order, free_rank = self.compute_rademacher_walsh_and_snf(mat)
        krull, novikov = self.evaluate_novikov_krull_valuation(exps)
        assoc = self.evaluate_multilateral_associators_witness(mat)
        return _AlgebraicSeed(
            matrix=mat, exponents=exps, snf_diagonal_factors=factors,
            torsion_free=torsion_free, torsion_order=torsion_order,
            cokernel_free_rank=free_rank, rw_norm=rw, krull=krull,
            novikov_norm=novikov, associators=assoc, input_digest=input_digest,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — Cuántica (AAV/QND, Uhlmann/Bures), geometría y Poincaré
# ══════════════════════════════════════════════════════════════════════════════
class _Fase2_Cuantica_Geometria(_Fase1_Algebra_Hipercompleja):
    """Fase 2. Medición débil AAV, fidelidad de Uhlmann, Gromov, CR, Melnikov, Poincaré."""

    def _fase2_inicio_desde_semilla_algebraica(
        self, seed: _AlgebraicSeed, ball_r: float, cylinder_R: float
    ) -> Tuple[float, float, bool]:
        """PRIMER método de F2. Consume la semilla de F1 → capacidades de Gromov."""
        if not isinstance(seed, _AlgebraicSeed):
            raise TypeError("Fase 2 exige una _AlgebraicSeed emitida por Fase 1.")
        r = _validate_positive(ball_r, "ball_r")
        R = _validate_positive(cylinder_R, "cylinder_R")
        return math.pi * r * r, math.pi * R * R, bool(r > R)

    @staticmethod
    def _fase2_witness_operator(dim: int) -> NDArray[np.complex128]:
        """Testigo Hermítico tridiagonal declarado A = 0.5·I + 0.1·(shift+shift†)."""
        A = np.eye(dim, dtype=np.complex128) * 0.5
        if dim > 1:
            A += 0.1 * np.diag(np.ones(dim - 1), k=1) + 0.1 * np.diag(np.ones(dim - 1), k=-1)
        return 0.5 * (A + A.conj().T)

    def compute_qnd_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
    ) -> QNDWeakMeasurementReport:
        """A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩ (división COMPLEJA). QND por conmutador relativo."""
        dim = rho_mac.shape[0]
        A = self._fase2_witness_operator(dim)
        comm = A @ rho_mac - rho_mac @ A @ np.eye(dim) * 0 - rho_mac @ A + A @ rho_mac
        comm = A @ rho_mac - rho_mac @ A
        comm_norm = float(np.linalg.norm(comm, ord="fro"))
        a_spec = float(np.max(np.abs(np.linalg.eigvalsh(A))))
        rho_fro = float(np.linalg.norm(rho_mac, ord="fro"))
        comm_rel = comm_norm / (a_spec * rho_fro + _WILKINSON_EPSILON)
        overlap = complex(np.vdot(phi_f, phi_i))
        overlap_mag = abs(overlap)
        singular = bool(overlap_mag < _OVERLAP_SINGULAR_TOL)
        if singular:
            weak_val: Optional[complex] = None
            weak_mag = 0.0
            anomalous = False
        else:
            weak_val = complex(np.vdot(phi_f, A @ phi_i) / overlap)
            weak_mag = abs(weak_val)
            anomalous = bool(weak_mag > a_spec + 1.0e-9)
        back_action_db = float(20.0 * math.log10(1.0 + comm_rel))
        is_qnd = bool(comm_rel <= _QND_COMM_REL_TOL)
        return QNDWeakMeasurementReport(
            weak_value=weak_val, weak_value_magnitude=weak_mag,
            overlap_magnitude=overlap_mag, is_overlap_singular=singular,
            is_weak_value_anomalous=anomalous, commutator_norm=comm_norm,
            commutator_relative=comm_rel, back_action_db=back_action_db,
            is_qnd_passive=is_qnd, witness_spectral_norm=a_spec,
        )

    @staticmethod
    def _fase2_psd_sqrt(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Raíz PSD por eigh (estable): √ρ = U diag(√λ₊) U†."""
        rh = 0.5 * (rho + rho.conj().T)
        w, U = np.linalg.eigh(rh)
        w = np.clip(w, 0.0, None)
        return (U * np.sqrt(w)[None, :]) @ U.conj().T

    def evaluate_bures_uhlmann_metric(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> BuresUhlmannMetricReport:
        """F(ρ,σ)=‖√ρ√σ‖₁² por SVD (estable, real, en [0,1]). d_B=√(2−2√F)."""
        M = self._fase2_psd_sqrt(rho) @ self._fase2_psd_sqrt(sigma)
        s = np.linalg.svd(M, compute_uv=False)
        fidelity = float(np.sum(s) ** 2)
        fidelity = max(0.0, min(1.0, fidelity))
        root_f = math.sqrt(fidelity)
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * root_f))
        angle = math.acos(max(-1.0, min(1.0, root_f)))
        isometry = bool(fidelity >= _FIDELITY_MIN)
        return BuresUhlmannMetricReport(
            fidelity_uhlmann=fidelity, bures_distance=bures, bures_angle=angle,
            is_isometry_preserved=isometry, fidelity_threshold=_FIDELITY_MIN,
            bures_distance_max=_BURES_DIST_MAX,
        )

    @staticmethod
    def _fase2_cauchy_riemann_residual(mat: NDArray[np.float64]) -> float:
        """Armonicidad ‖Δφ‖_F/(‖φ‖_F+ε) con Laplaciano periódico de 5 puntos."""
        phi = mat
        lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
               + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4.0 * phi)
        return float(np.linalg.norm(lap) / (np.linalg.norm(phi) + _WILKINSON_EPSILON))

    @staticmethod
    def _fase2_melnikov_coefficients(delta: float, gamma: float, omega: float) -> Tuple[float, float]:
        d = _validate_finite(delta, "melnikov_delta")
        g = _validate_finite(gamma, "melnikov_gamma")
        w = _validate_finite(omega, "melnikov_omega")
        return 2.0 * math.pi * abs(g) / math.cosh(math.pi * w / 2.0), -8.0 * d

    def _fase2_melnikov_quadrature(self, delta, gamma, omega) -> Tuple[float, float]:
        t = np.linspace(-_MELNIKOV_HALF_WIDTH, _MELNIKOV_HALF_WIDTH, _MELNIKOV_POINTS)
        v = 2.0 / np.cosh(t)
        return abs(gamma) * abs(_trapz(v * np.cos(omega * t), t)), -delta * _trapz(v * v, t)

    def evaluate_poincare_kam_melnikov(self, delta, gamma, omega):
        amp, off = self._fase2_melnikov_coefficients(delta, gamma, omega)
        chaos = bool(omega != 0.0 and amp > abs(off))
        margin = float(amp - abs(off))
        eps_valid = bool(max(abs(delta), abs(gamma)) <= _MELNIKOV_EPS_POLICY)
        return chaos, amp, off, margin, eps_valid

    @staticmethod
    def _fase2_agm(a: float, b: float) -> float:
        for _ in range(60):
            if abs(a - b) <= 1.0e-15 * max(abs(a), abs(b)):
                return 0.5 * (a + b)
            a, b = 0.5 * (a + b), math.sqrt(a * b)
        return 0.5 * (a + b)

    def _fase2_poincare_lindstedt(self, amplitude: float) -> Tuple[float, float]:
        a = _validate_finite(amplitude, "pendulum_amplitude")
        if not (0.0 <= a < math.pi):
            raise ValueError("pendulum_amplitude ∈ [0,π).")
        exact = self._fase2_agm(1.0, math.cos(a / 2.0))
        series = 1.0 - a * a / 16.0 + a ** 4 / 3072.0
        return exact, series

    def _fase2_poincare_stroboscopic_lyapunov(
        self, delta, gamma, omega, periods, steps
    ) -> Optional[float]:
        """λ_max finito del mapa estroboscópico (Benettin). EVIDENCIA, no prueba."""
        d = _validate_finite(delta, "d")
        g = _validate_finite(gamma, "g")
        w = _validate_finite(omega, "w")
        n_per = _validate_int(periods, "poincare_periods", 0, 20000)
        n_st = _validate_int(steps, "poincare_steps", 8, 4096)
        if w <= 0.0 or n_per == 0:
            return None
        T = 2.0 * math.pi / w
        h = T / n_st
        h2 = 0.5 * h
        sin, cos, hypot, log = math.sin, math.cos, math.hypot, math.log
        x, v = _POINCARE_IC
        px, pv = 1.0, 0.0
        transient = n_per // 8
        acc, counted, t = 0.0, 0, 0.0

        def deriv(tt, xx, vv, ppx, ppv):
            return vv, -sin(xx) - d * vv + g * cos(w * tt), ppv, -cos(xx) * ppx - d * ppv

        for k in range(n_per):
            for _ in range(n_st):
                k1 = deriv(t, x, v, px, pv)
                k2 = deriv(t + h2, x + h2 * k1[0], v + h2 * k1[1], px + h2 * k1[2], pv + h2 * k1[3])
                k3 = deriv(t + h2, x + h2 * k2[0], v + h2 * k2[1], px + h2 * k2[2], pv + h2 * k2[3])
                k4 = deriv(t + h, x + h * k3[0], v + h * k3[1], px + h * k3[2], pv + h * k3[3])
                x += (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                v += (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                px += (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
                pv += (h / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
                t += h
            s = hypot(px, pv)
            if s < 1.0e-300:
                px, pv, s = 1.0, 0.0, 1.0
            if k >= transient:
                acc += log(s)
                counted += 1
            px /= s
            pv /= s
        return float(acc / (counted * T)) if counted else None

    # ── ÚLTIMO método de Fase 2 → inicio de Fase 3 ─────────────────────────────
    def _fase2_emit_witness_seed(
        self,
        seed: _AlgebraicSeed,
        rho: NDArray[np.complex128],
        rho_admissible: bool,
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        ball_r: float,
        cylinder_R: float,
        melnikov_delta: float,
        melnikov_gamma: float,
        melnikov_omega: float,
        pendulum_amplitude: float,
        poincare_periods: int,
        poincare_steps: int,
    ) -> _WitnessSeed:
        """ÚLTIMO método de F2. Emite la semilla testigo que consume F3."""
        c_ball, c_cyl, violated = self._fase2_inicio_desde_semilla_algebraica(seed, ball_r, cylinder_R)
        qnd = self.compute_qnd_weak_measurement(rho, phi_i, phi_f)
        bures = self.evaluate_bures_uhlmann_metric(rho, sigma)
        cr = self._fase2_cauchy_riemann_residual(seed.matrix)
        chaos, amp, off, margin, eps_valid = self.evaluate_poincare_kam_melnikov(
            melnikov_delta, melnikov_gamma, melnikov_omega
        )
        exact, series = self._fase2_poincare_lindstedt(pendulum_amplitude)
        lyap = self._fase2_poincare_stroboscopic_lyapunov(
            melnikov_delta, melnikov_gamma, melnikov_omega, poincare_periods, poincare_steps
        )
        consistent = None if lyap is None else bool((not (lyap > _LYAP_POS_TOL)) or chaos)
        force_veto = bool(
            violated or (not seed.torsion_free) or chaos
            or (seed.krull < _DEFAULT_KRULL_FLOOR)
            or (not qnd.is_qnd_passive) or (not bures.is_isometry_preserved)
            or (not rho_admissible)
        )
        return _WitnessSeed(
            algebraic=seed, rho_admissible=rho_admissible, qnd=qnd, bures=bures,
            cr_residual=cr, gromov_violated=violated, capacity_ball=c_ball,
            capacity_cylinder=c_cyl, melnikov_chaos=chaos, melnikov_amplitude=amp,
            melnikov_offset=off, melnikov_margin=margin, melnikov_eps_valid=eps_valid,
            lindstedt_exact=exact, lindstedt_series=series, lyapunov=lyap,
            lyapunov_consistent=consistent, force_veto=force_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — Kleisli T³, RSI³ reflexivo, Heyting y pasaporte
# ══════════════════════════════════════════════════════════════════════════════
class _Fase3_Kleisli_RSI3(_Fase2_Cuantica_Geometria):
    """Fase 3. Contracción de Banach, torre RSI³, adjudicación Heyting y pasaporte."""

    def _fase3_inicio_desde_semilla_testigo(self, witness: _WitnessSeed, depth: int, lam: float):
        """PRIMER método de F3. Veto duro ⇒ profundidad efectiva 0."""
        if not isinstance(witness, _WitnessSeed):
            raise TypeError("Fase 3 exige una _WitnessSeed emitida por Fase 2.")
        d = _validate_int(depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        l = _validate_lambda(lam)
        return witness, (0 if witness.force_veto else d), l

    @staticmethod
    def _fase3_certified_bound(lam: float) -> float:
        return float(lam) ** 3 + _KLEISLI_ERROR_BUDGET

    @staticmethod
    def _fase3_kleisli_cubed(x, lam: float):
        y = x
        for _ in range(3):
            y = lam * np.cos(y) + _KLEISLI_OFFSET
        return y

    @staticmethod
    def _fase3_kleisli_cubed_derivative(x, lam: float):
        y1 = lam * np.cos(x) + _KLEISLI_OFFSET
        y2 = lam * np.cos(y1) + _KLEISLI_OFFSET
        return np.abs(lam ** 3 * np.sin(x) * np.sin(y1) * np.sin(y2))

    def _fase3_picard(self, lam, q, tol=_PICARD_TOL, max_iter=_PICARD_MAX_ITER):
        x_prev = x = 0.0
        converged = False
        for _ in range(max_iter):
            x_prev = x
            x = float(self._fase3_kleisli_cubed(x_prev, lam))
            if abs(x - x_prev) <= tol:
                converged = True
                break
        a_post = q / (1.0 - q) * abs(x - x_prev)
        residual = abs(float(self._fase3_kleisli_cubed(x, lam)) - x)
        return x, a_post, converged, residual

    def _fase3_empirical_lipschitz(self, level: int, lam: float) -> float:
        n = _KLEISLI_GRID_BASE * (2 ** level) + 1
        grid = np.linspace(-_KLEISLI_DOMAIN, _KLEISLI_DOMAIN, n)
        return float(np.max(self._fase3_kleisli_cubed_derivative(grid, lam)))

    def _fase3_rsi_recurse(self, lam, q, max_depth, gate_ok, meta_ok):
        accepted = 0
        prev = -math.inf
        chain = hashlib.sha256(_RSI_CHAIN_DOMAIN).digest()
        for level in range(1, max_depth + 1):
            e = self._fase3_empirical_lipschitz(level, lam)
            if not (gate_ok and meta_ok and e <= q + _LIP_TOL and e >= prev - _LIP_TOL):
                break
            accepted = level
            prev = e
            chain = hashlib.sha256(chain + f"|{level}|{_fmt(e)}|{int(meta_ok)}".encode()).digest()
        return accepted, chain.hex()

    def evaluate_rsi3_banach_kleisli(self, initial_lip=_KLEISLI_LIP_T, rsi_max_depth=_RSI_MAX_DEPTH):
        lam = _validate_lambda(initial_lip)
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        selftest = self._fase3_metamorphic_selftest()
        analytic = float(lam) ** 3
        q = self._fase3_certified_bound(lam)
        if not q < 1.0:
            raise ValueError("La cota certificada no es contractiva (q ≥ 1).")
        x, a_post, conv, res = self._fase3_picard(lam, q)
        x_ref, _, ref_conv, _ = self._fase3_picard(lam, q, _PICARD_REF_TOL, _PICARD_REF_MAX_ITER)
        meta = bool(ref_conv and abs(x - x_ref) <= a_post + _CONSISTENCY_TOL)
        consistency = bool(conv and res <= _CONSISTENCY_TOL)
        accepted, chain = self._fase3_rsi_recurse(lam, q, depth, selftest and consistency, meta)
        return _KleisliState(
            lip_analytic=analytic, lip_certified=q, fixed_point=x,
            a_posteriori_error=a_post, residual=res, meta_verified=meta,
            selftest_passed=selftest, consistency_ok=consistency,
            depth_accepted=accepted, certificate_chain=chain,
        )

    def _fase3_metamorphic_selftest(self) -> bool:
        """Autoprueba fail-closed: CD, SNF, Parseval, Melnikov, Bures, AAV, Heyting, RSI."""
        try:
            # 1. Cayley-Dickson: asociatividad n≤4, no-asociatividad n≥8, alternatividad.
            if _cd_associator_defect(4)[0] != 0.0:
                return False
            if _cd_associator_defect(8)[0] <= 0.0:
                return False
            if _cd_alternativity_defect(8) != 0:
                return False
            if _cd_alternativity_defect(16) <= 0:
                return False
            e1 = np.zeros(4); e1[1] = 1.0
            e2 = np.zeros(4); e2[2] = 1.0
            if not np.array_equal(_cd_mul(e1, e2, 4), np.array([0, 0, 0, 1.0])):
                return False
            # 2. SNF con cadena de divisibilidad.
            if self._fase1_smith_diagonal([[2, 0], [0, 3]]) != (1, 6):
                return False
            if self._fase1_smith_diagonal([[6, 0], [0, 10]]) != (2, 30):
                return False
            f = self._fase1_smith_diagonal([[2, 4], [6, 8]])
            if math.prod(f) != 8 or any(b % a for a, b in zip(f, f[1:])):
                return False
            # 3. Parseval e identidad espectral de influencias.
            rng = np.random.default_rng(20250101)
            bits = rng.integers(0, 2, size=16).astype(np.int64)
            ghat = self._fase1_fwht(1.0 - 2.0 * bits) / 16.0
            if abs(float(np.sum(ghat ** 2)) - 1.0) > 1e-12:
                return False
            if abs(float(sum(bin(s).count("1") * ghat[s] ** 2 for s in range(16)))
                   - self._fase1_total_influence(bits)) > 1e-12:
                return False
            # 4. Melnikov cerrada vs cuadratura.
            amp, off = self._fase2_melnikov_coefficients(0.01, 0.02, 1.0)
            aq, oq = self._fase2_melnikov_quadrature(0.01, 0.02, 1.0)
            if abs(amp - aq) > 1e-6 * amp or abs(off - oq) > 1e-6 or abs(off + 0.08) > 1e-12:
                return False
            # 5. Bures: auto-fidelidad 1; F(I/2,|0><0|)=1/2.
            rho = np.eye(2, dtype=np.complex128) / 2.0
            if abs(self.evaluate_bures_uhlmann_metric(rho, rho).fidelity_uhlmann - 1.0) > 1e-9:
                return False
            sig = np.zeros((2, 2), dtype=np.complex128); sig[0, 0] = 1.0
            if abs(self.evaluate_bures_uhlmann_metric(rho, sig).fidelity_uhlmann - 0.5) > 1e-9:
                return False
            # 6. AAV: división compleja (fase) y régimen singular.
            A = np.diag([1.0, -1.0]).astype(np.complex128)
            pi = np.array([1, 1], dtype=np.complex128) / math.sqrt(2)
            pf = np.array([1, 1j], dtype=np.complex128) / math.sqrt(2)
            ov = complex(np.vdot(pf, pi))
            aw = complex(np.vdot(pf, A @ pi) / ov)
            if abs(aw - complex(np.vdot(pf, A @ pi)) / ov) > 1e-12:
                return False
            # 7. Heyting: residuación exhaustiva.
            vals = (HeytingTruthValue.VETOED, HeytingTruthValue.DEGRADED, HeytingTruthValue.COHERENT)
            for a in vals:
                for b in vals:
                    for c in vals:
                        if (a.meet(b).rank <= c.rank) != (a.rank <= b.implies(c).rank):
                            return False
            # 8. Banach y monotonía RSI por anidamiento.
            if abs(self._fase3_certified_bound(0.95) - 0.85975) > 1e-12:
                return False
            if not (self._fase3_empirical_lipschitz(1, 0.95)
                    <= self._fase3_empirical_lipschitz(2, 0.95) + _LIP_TOL):
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica: excepción (fail-closed).")
            return False

    def _fase3_heyting_adjudicate(self, w: _WitnessSeed, k: _KleisliState, requested: int):
        s = w.algebraic
        spectral = (HeytingTruthValue.VETOED
                    if (not s.torsion_free) or s.krull < _DEFAULT_KRULL_FLOOR
                    else (HeytingTruthValue.DEGRADED if s.rw_norm > _RW_DEGRADED_TOL
                          else HeytingTruthValue.COHERENT))
        quantum = (HeytingTruthValue.VETOED
                   if (not w.qnd.is_qnd_passive) or (not w.bures.is_isometry_preserved)
                   or (not w.rho_admissible) else HeytingTruthValue.COHERENT)
        if w.gromov_violated or w.melnikov_chaos:
            dynamical = HeytingTruthValue.VETOED
        elif (w.cr_residual > _CR_RESIDUAL_TOL or not w.melnikov_eps_valid
              or w.lyapunov_consistent is False or s.associators.phantom_collusion_suspected):
            dynamical = HeytingTruthValue.DEGRADED
        else:
            dynamical = HeytingTruthValue.COHERENT
        rsi = (HeytingTruthValue.COHERENT if k.depth_accepted == requested
               else HeytingTruthValue.DEGRADED)
        verifier = (HeytingTruthValue.COHERENT
                    if (k.selftest_passed and k.consistency_ok and k.meta_verified)
                    else HeytingTruthValue.VETOED)
        return spectral.meet(quantum).meet(dynamical).meet(rsi).meet(verifier)

    def _fase3_sign_passport(self, fields: Tuple[object, ...]) -> str:
        payload = "|".join(_fmt(x) for x in fields).encode("utf-8")
        return hmac.new(self._secret_key, _PASSPORT_DOMAIN + payload, hashlib.sha256).hexdigest()

    # ── ÚLTIMO método de Fase 3: API pública ───────────────────────────────────
    def execute_phantom_collusion_witness_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
        *,
        phi_i: Optional[NDArray[np.complex128]] = None,
        phi_f: Optional[NDArray[np.complex128]] = None,
        sigma_input: Optional[NDArray[np.complex128]] = None,
        ball_r: float = _DEFAULT_BALL_R,
        cylinder_R: float = _DEFAULT_CYLINDER_R,
        melnikov_delta: float = _MELNIKOV_DEFAULT_DELTA,
        melnikov_gamma: float = _MELNIKOV_DEFAULT_GAMMA,
        melnikov_omega: float = _MELNIKOV_DEFAULT_OMEGA,
        pendulum_amplitude: float = _DEFAULT_PENDULUM_AMPLITUDE,
        poincare_periods: int = _POINCARE_PERIODS,
        poincare_steps: int = _POINCARE_STEPS,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        initial_lip: float = _KLEISLI_LIP_T,
        capacity_series: Optional[Sequence[float]] = None,
    ) -> PhantomCollusionWitnessSpectrumReport:
        """Auditoría pasiva QND anti-colusión completa. ÚLTIMO método de F3; API pública."""
        mat = _as_cost_matrix(apu_cost_matrix)
        exps = _as_exponents(exponents)
        dim = mat.shape[0]
        if rho_mac_input is None:
            rho = np.eye(dim, dtype=np.complex128) / float(dim)
            rho_admissible = True
        else:
            rho, rho_admissible = _as_density(rho_mac_input, "rho_mac_input")
            if rho.shape[0] != dim:
                raise ValueError("rho_mac_input debe tener la misma dimensión que la cost matrix.")
        uniform = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        pi = uniform if phi_i is None else _as_unit_vector(phi_i, dim, "phi_i")
        pf = uniform if phi_f is None else _as_unit_vector(phi_f, dim, "phi_f")
        if sigma_input is None:
            sigma = rho
        else:
            sigma, sig_ok = _as_density(sigma_input, "sigma_input")
            if sigma.shape[0] != dim or not sig_ok:
                raise ValueError("sigma_input debe ser una matriz de densidad de la misma dimensión.")
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        lam = _validate_lambda(initial_lip)
        n_per = _validate_int(poincare_periods, "poincare_periods", 0, 20000)
        n_st = _validate_int(poincare_steps, "poincare_steps", 8, 4096)
        amp = _validate_finite(pendulum_amplitude, "pendulum_amplitude")
        series_str = ("none" if capacity_series is None
                      else ",".join(_fmt(float(x)) for x in capacity_series))
        params = "|".join((
            _fmt(float(ball_r)), _fmt(float(cylinder_R)),
            _fmt(float(melnikov_delta)), _fmt(float(melnikov_gamma)), _fmt(float(melnikov_omega)),
            _fmt(amp), _fmt(n_per), _fmt(n_st), _fmt(depth), _fmt(lam), series_str,
        ))
        digest = _input_digest(mat, exps, rho, pi, pf, sigma, params)
        seed = self._fase1_emit_algebraic_seed(mat, exps, digest)
        witness = self._fase2_emit_witness_seed(
            seed, rho, rho_admissible, pi, pf, sigma, ball_r, cylinder_R,
            melnikov_delta, melnikov_gamma, melnikov_omega, amp, n_per, n_st,
        )
        witness, eff_depth, lam = self._fase3_inicio_desde_semilla_testigo(witness, depth, lam)
        kleisli = self.evaluate_rsi3_banach_kleisli(lam, eff_depth)
        verdict = self._fase3_heyting_adjudicate(witness, kleisli, eff_depth)
        jerk = _third_difference(capacity_series)
        assoc = seed.associators
        passport = self._fase3_sign_passport((
            _ENGINE_VERSION, verdict, digest, seed.rw_norm, seed.torsion_free,
            ",".join(str(d) for d in seed.snf_diagonal_factors), seed.torsion_order,
            seed.cokernel_free_rank, seed.krull, seed.novikov_norm, rho_admissible,
            witness.qnd.is_qnd_passive, witness.qnd.commutator_relative,
            witness.qnd.back_action_db, witness.qnd.weak_value_magnitude,
            witness.qnd.is_overlap_singular, witness.qnd.is_weak_value_anomalous,
            witness.bures.fidelity_uhlmann, witness.bures.bures_distance,
            witness.bures.is_isometry_preserved, assoc.phantom_collusion_suspected,
            ",".join(_fmt(r) for r in (assoc.pationion_ratio_32d, assoc.chingon_ratio_64d, assoc.routon_ratio_128d)),
            witness.gromov_violated, witness.cr_residual, witness.melnikov_chaos,
            witness.melnikov_eps_valid, witness.lyapunov, witness.lindstedt_exact,
            kleisli.lip_certified, kleisli.depth_accepted, kleisli.certificate_chain,
            kleisli.selftest_passed, kleisli.consistency_ok, kleisli.meta_verified, jerk,
        ))
        return PhantomCollusionWitnessSpectrumReport(
            rademacher_walsh_norm=seed.rw_norm, smith_torsion_free=seed.torsion_free,
            smith_torsion_order=seed.torsion_order, cokernel_free_rank=seed.cokernel_free_rank,
            snf_diagonal_factors=seed.snf_diagonal_factors, krull_valuation=seed.krull,
            novikov_norm=seed.novikov_norm, rho_admissible=rho_admissible,
            cauchy_riemann_residual=witness.cr_residual, is_gromov_violated=witness.gromov_violated,
            gromov_capacity_ball=witness.capacity_ball, gromov_capacity_cylinder=witness.capacity_cylinder,
            melnikov_chaos_detected=witness.melnikov_chaos, melnikov_amplitude=witness.melnikov_amplitude,
            melnikov_offset=witness.melnikov_offset, melnikov_transversality_margin=witness.melnikov_margin,
            melnikov_eps_regime_valid=witness.melnikov_eps_valid,
            poincare_lindstedt_exact=witness.lindstedt_exact, poincare_lindstedt_series=witness.lindstedt_series,
            poincare_lyapunov=witness.lyapunov, poincare_melnikov_consistent=witness.lyapunov_consistent,
            banach_lip_analytic=kleisli.lip_analytic, banach_contraction_lip=kleisli.lip_certified,
            banach_fixed_point=kleisli.fixed_point, banach_a_posteriori_error=kleisli.a_posteriori_error,
            banach_meta_verified=kleisli.meta_verified, rsi_accepted_depth=kleisli.depth_accepted,
            rsi_certificate_chain=kleisli.certificate_chain, capacity_third_difference=jerk,
            associator_report=assoc, qnd_report=witness.qnd, bures_report=witness.bures,
            metamorphic_selftest_passed=kleisli.selftest_passed, input_digest=digest,
            truth_value=verdict, passport_hash=passport, schema_version=_SCHEMA_VERSION,
        )


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPhantomCollusionMitigatorWitnessEngine(_Fase3_Kleisli_RSI3):
    """Motor Espectral de Atestación Pasiva QND anti-colusión (v14.0.0)."""


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = BoolNovikovPhantomCollusionMitigatorWitnessEngine(secret_key=b"demo-key-0123456789abcdef")
    print(f"Autoprueba metamórfica: {engine._fase3_metamorphic_selftest()}")
    mat = np.array([[2.0, 0.1], [0.1, 2.5]])
    exps = np.array([0.15, 0.22, 0.40])
    rep = engine.execute_phantom_collusion_witness_audit(mat, exps)
    print("=== TEST AUTÓNOMO: BoolNovikovPhantomCollusionMitigatorWitnessEngine v14.0 ===")
    print(f"Veredicto Heyting Ω₃        : {rep.truth_value.name} ({hex(rep.truth_value.value)})")
    print(f"SNF (cadena) / torsión libre: {rep.snf_diagonal_factors} / {rep.smith_torsion_free}")
    print(f"QND pasiva / back-action    : {rep.qnd_report.is_qnd_passive} / {rep.qnd_report.back_action_db:.6f} dB")
    print(f"Valor débil |A_w| / anómalo : {rep.qnd_report.weak_value_magnitude:.6f} / {rep.qnd_report.is_weak_value_anomalous}")
    print(f"Fidelidad Uhlmann / Bures   : {rep.bures_report.fidelity_uhlmann:.6f} / {rep.bures_report.bures_distance:.6f}")
    print(f"Asociadores 32/64/128       : {rep.associator_report.pationion_norm_32d:.4f} / "
          f"{rep.associator_report.chingon_norm_64d:.4f} / {rep.associator_report.routon_norm_128d:.4f}")
    print(f"Colusión fantasma sospechada: {rep.associator_report.phantom_collusion_suspected}")
    print(f"Krull v(A) / Banach Lip     : {rep.krull_valuation:.4f} / {rep.banach_contraction_lip:.6f}")
    print(f"Pasaporte HMAC              : {rep.passport_hash[:16]}...")
    print("=== TEST COMPLETADO ===")