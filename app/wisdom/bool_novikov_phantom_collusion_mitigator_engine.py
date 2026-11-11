# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPhantomCollusionMitigatorEngine v14.0.0-PhD                                    ║
║ Ubicación: app/wisdom/bool_novikov_phantom_collusion_mitigator_engine.py                            ║
║ Versión  : 14.0.0-ENGINE-BOOL-NOVIKOV-PHANTOM-COLLUSION-MITIGATOR-POINCARE-RSI3-PhD                  ║
║ Función  : Motor raíz de detección y mitigación de colusiones fantasma multilaterales (32/64/128D)   ║
║            sobre la variedad dual Boole-Novikov (R_Boole ⊗ Λ_Nov). Sin crowbar: la actuación         ║
║            corresponde al motor fiscal (prosecutor). Este motor solo detecta, juzga y recomienda.    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  _Fase1_Algebra_Hipercompleja
     └ _fase1_emit_algebraic_seed            ÚLTIMO de F1 → inicio de F2.
  _Fase2_Geometria_Celeste
     └ _fase2_emit_mitigation_seed           ÚLTIMO de F2 → inicio de F3.
  _Fase3_Kleisli_RSI3
     └ execute_phantom_collusion_mitigation_audit   ÚLTIMO de F3 (API pública).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · SNF sobre ℤ con cadena de divisibilidad d₁|d₂|…|d_r (barridos gcd/lcm unimodulares).
    La v13 infería torsión de la "cercanía a entero" de los valores singulares: eso es ajeno
    a la torsión del cokernel. Ahora |tors| = ∏|dᵢ| y rango libre = m − r.
    `betti_number_zero` (engañoso) → `cokernel_free_rank` + `full_rank`.
  · Influencia total booleana por FWHT (la v13 usaba una diferencia cíclica normalizada).
  · Asociadores k-vías: álgebra de Cayley-Dickson REAL (e_i e_j = S[i,j] e_{i⊕j}) en 32/64/128D
    con k = 5/7/9 actores. Se computan asociadores locales [a_i,a_{i+1},a_{i+2}] (RMS/máx) y la
    discrepancia left-comb vs right-comb, normalizados por el defecto algebraico ‖A‖_F/n^{3/2}
    (invariante: 0 para n≤4, >0 para n≥8). Ya no son reescalados de la norma off-diagonal.
  · CR: Laplaciano periódico de 5 puntos. La v13 construía J con diagonal 1 (J²≠−I) y medía
    ‖(I+J)g‖/‖g‖, que vale √2 para cualquier J ortogonal antisimétrico: residuo sin información.
  · Gromov: capacidades πr², πR² y violación ⇔ r>R (comparación directa, sin cancelación).
  · Melnikov en forma cerrada; Lindstedt–Poincaré Ω(a)=AGM(1,cos(a/2)); mapa estroboscópico de
    Poincaré con Lyapunov de Benettin (EVIDENCIA; tensión unidireccional).
  · RSI³: cota analítica λ³ + certificada λ³+ε; Picard con error a posteriori verificado contra
    un Picard de referencia; mallas diádicas anidadas de |(T³)'| (monotonía por inclusión);
    cadena de certificados SHA-256. Jerk = tercera diferencia MEDIDA (Optional).
  · Dirac: 2·m_e·c² = 1021.9979 keV (511 keV por fotón en el CM). Anomalía = ‖M − I‖_F
    (política declarada: desviación respecto de la identidad = ausencia de interacción).
  · Epistemia: `is_phantom_collusion_detected` → `phantom_collusion_suspected` (heurístico,
    DEGRADED) + `mitigation_recommended` (solo fallos CERTIFICADOS). El defecto de rango real
    ya NO veta (v13 vetaba por `not betti_zero`).
  · Heyting Ω₃: meet de 5 jueces (espectral, dinámico, asociadores, RSI, verificador).
  · Pasaporte HMAC canónico REPRODUCIBLE (sin tiempos); digest de toda la entrada; clave por entorno.

LÍMITES DECLARADOS (no se afirma más de lo que se prueba):
  · El asociador k-vías es una agregación de triples locales + discrepancia de combs; NO es el
    tensor completo del asociaedro (Catalan) ni una prueba de colusión.
  · La sospecha por asociadores es un heurístico normalizado por el defecto algebraico.
  · Gromov en ℝ² es monotonía de área; el no-squeezing profundo es n≥2. Aquí opera como política r≤R.
  · Melnikov es riguroso solo para ε pequeño; Lyapunov finito es evidencia dependiente de la sonda.
  · Este motor NO modela disparo crowbar ni escribe en hardware; solo recomienda la mitigación.
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

logger = logging.getLogger("APU.Wisdom.BoolNovikovPhantomCollusionMitigatorEngine")

# ── Versiones y dominios ───────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "14.0.0-ENGINE-BOOL-NOVIKOV-PHANTOM-COLLUSION-MITIGATOR-POINCARE-RSI3-PhD"
)
version: Final[str] = _ENGINE_VERSION
_SCHEMA_VERSION: Final[str] = "mitigator-report-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_PHANTOM_MITIGATOR_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_MITIGATOR_DEV_ONLY_PHANTOM_KEY_v14.0.0"
_PASSPORT_DOMAIN: Final[bytes] = b"PHANTOM-COLLUSION-MITIGATOR-PASSPORT-v14.0|"
_RSI_CHAIN_DOMAIN: Final[bytes] = b"PHANTOM-MITIGATOR-RSI3-CERT-CHAIN-v14.0|"

# ── Numérica y álgebra ─────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SNF_FIXED_POINT_SCALE: Final[float] = 100.0      # política declarada, no derivada
_SNF_MAX_ABS_INT: Final[int] = 1 << 62
_RW_DEGRADED_TOL: Final[float] = 0.5
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_ASSOC_DIMS: Final[Tuple[int, int, int]] = (32, 64, 128)
_ASSOC_ACTORS: Final[Tuple[int, int, int]] = (5, 7, 9)
_ASSOC_RATIO_SUSPECT: Final[float] = 2.0          # política heurística declarada

# ── Geometría y Poincaré ───────────────────────────────────────────────────────
_DEFAULT_BALL_R: Final[float] = 0.50
_DEFAULT_CYLINDER_R: Final[float] = 0.85
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

# ── Dirac (sin crowbar: la actuación es del motor fiscal) ──────────────────────
_ELECTRON_REST_MASS_KEV: Final[float] = 510.99895
_DIRAC_TOTAL_KEV: Final[float] = 2.0 * _ELECTRON_REST_MASS_KEV      # 1021.9979
_ANOMALY_THRESHOLD: Final[float] = 1.0e-6

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
class MultilateralAssociatorMitigationReport:
    """Asociadores k-vías genuinos de Cayley-Dickson (5/7/9 actores en 32/64/128D)."""

    pationionic_associator_norm: float      # RMS de asociadores locales, k=5, 32D
    chingonic_associator_norm: float        # RMS, k=7, 64D
    routonic_associator_norm: float         # RMS, k=9, 128D
    pationionic_max_norm: float
    chingonic_max_norm: float
    routonic_max_norm: float
    pationionic_ratio: float                # RMS / defecto algebraico
    chingonic_ratio: float
    routonic_ratio: float
    comb_discrepancy_32d: float             # ‖left_comb − right_comb‖
    comb_discrepancy_64d: float
    comb_discrepancy_128d: float
    algebra_defect_32d: float
    algebra_defect_64d: float
    algebra_defect_128d: float
    phantom_collusion_suspected: bool       # heurístico, NO prueba
    actor_counts: Tuple[int, int, int]
    embedding_policy: str


@dataclass(frozen=True)
class PhantomCollusionSpectrumReport:
    """Reporte de mitigación anti-colusión fantasma (reproducible, sin tiempos)."""

    rademacher_walsh_norm: float
    smith_torsion_free: bool
    smith_torsion_order: int
    cokernel_free_rank: int                 # β₀ del cokernel (rango libre), NO β₁
    full_rank: bool
    snf_diagonal_factors: Tuple[int, ...]
    krull_valuation: float
    novikov_norm: float
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
    poincare_lindstedt_error: float
    poincare_lyapunov: Optional[float]
    poincare_melnikov_consistent: Optional[bool]
    banach_lip_analytic: float
    banach_contraction_lip: float
    banach_fixed_point: float
    banach_a_posteriori_error: float
    banach_meta_verified: bool
    rsi_accepted_depth: int
    rsi_empirical_lipschitz: Tuple[float, ...]
    rsi_certificate_chain: str
    capacity_third_difference: Optional[float]
    anomaly_norm: float
    dirac_annihilation_kev: float
    associator_report: MultilateralAssociatorMitigationReport
    phantom_collusion_suspected: bool       # evidencia heurística
    mitigation_recommended: bool            # solo fallos certificados
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
    full_rank: bool
    rw_norm: float
    krull: float
    novikov_norm: float
    anomaly_norm: float
    associators: MultilateralAssociatorMitigationReport
    input_digest: str


@dataclass(frozen=True, eq=False)
class _MitigationSeed:
    algebraic: _AlgebraicSeed
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
    lindstedt_error: float
    lyapunov: Optional[float]
    lyapunov_consistent: Optional[bool]
    dirac_kev: float
    mitigation_recommended: bool


@dataclass(frozen=True)
class _KleisliState:
    lip_analytic: float
    lip_certified: float
    lip_empirical: Tuple[float, ...]
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


def validate_positron(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError("positron_hmac_valid debe ser bool o None (fail-closed).")


def _validate_amplitude(value: object) -> float:
    a = _validate_finite(value, "pendulum_amplitude")
    if not (0.0 <= a < math.pi):
        raise ValueError("pendulum_amplitude ∈ [0,π).")
    return a


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


def _input_digest(mat: NDArray[np.float64], exps: NDArray[np.float64], params: str) -> str:
    h = hashlib.sha256()
    for name, arr in (("MIC", mat), ("EXP", exps)):
        h.update(f"{name}{arr.shape}".encode("ascii"))
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes(order="C"))
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


def _cd_embed_actors(mat: NDArray[np.float64], n: int, k: int) -> List[NDArray[np.float64]]:
    """Embedding declarado de k actores: flat row-major, roll(i·stride), resize(n), ℓ2 unit."""
    flat = mat.reshape(-1)
    stride = max(1, flat.size // k)
    actors: List[NDArray[np.float64]] = []
    for i in range(k):
        v = np.resize(np.roll(flat, i * stride), n).astype(np.float64)
        nrm = float(np.linalg.norm(v))
        if nrm > 0.0:
            v = v / nrm
        actors.append(v)
    return actors


def _cd_local_associator_stats(actors: Sequence[NDArray[np.float64]], n: int) -> Tuple[float, float]:
    """RMS y máximo de ‖[a_i,a_{i+1},a_{i+2}]‖ sobre triples consecutivos (k-vías local)."""
    norms = [
        float(np.linalg.norm(_cd_associator(actors[i], actors[i + 1], actors[i + 2], n)))
        for i in range(len(actors) - 2)
    ]
    if not norms:
        return 0.0, 0.0
    return math.sqrt(sum(v * v for v in norms) / len(norms)), max(norms)


def _cd_comb_discrepancy(actors: Sequence[NDArray[np.float64]], n: int) -> float:
    """‖left_comb − right_comb‖: discrepancia de parentización extrema de k actores."""
    if len(actors) < 2:
        return 0.0
    L = actors[0]
    for a in actors[1:]:
        L = _cd_mul(L, a, n)
    R = actors[-1]
    for a in reversed(actors[:-1]):
        R = _cd_mul(a, R, n)
    return float(np.linalg.norm(L - R))


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — Álgebra hipercompleja, SNF, booleanas, Novikov y asociadores k-vías
# ══════════════════════════════════════════════════════════════════════════════
class _Fase1_Algebra_Hipercompleja:
    """Fase 1. Invariantes enteros, influencia booleana, valuación y asociadores CD."""

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._secret_key = _resolve_secret_key(secret_key)

    @staticmethod
    def _fase1_smith_diagonal(a_in: Sequence[Sequence[int]]) -> Tuple[int, ...]:
        """Forma de Smith con cadena d₁|d₂|…|d_r (barridos gcd/lcm unimodulares).
        coker ≅ ⊕ ℤ/|dᵢ| ⊕ ℤ^{m−r}; torsión libre ⇔ todo |dᵢ|=1."""
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
        """b(x) = [entrada_x > 0], relleno con ceros a potencia de 2 (política de cable)."""
        flat = [1 if v > 0 else 0 for row in q for v in row]
        n_bits = 1
        while (1 << n_bits) < len(flat):
            n_bits += 1
        bits = np.zeros(1 << n_bits, dtype=np.int64)
        bits[: len(flat)] = flat
        return bits

    @staticmethod
    def _fase1_total_influence(bits: NDArray[np.int64]) -> float:
        """Influencia total Σₖ Pr_x[b(x) ≠ b(x ⊕ eₖ)] (definición combinatoria exacta)."""
        n_pts = bits.size
        n_vars = n_pts.bit_length() - 1
        idx = np.arange(n_pts)
        return sum(float(np.mean(bits != bits[idx ^ (1 << k)])) for k in range(n_vars))

    @staticmethod
    def _fase1_fwht(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transformada de Walsh–Hadamard sin normalizar (mariposa natural)."""
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
        """(rw_norm, factores SNF, torsión libre, |torsión|, rango libre, rango pleno)."""
        q = _quantize_matrix(mat)
        factors = self._fase1_smith_diagonal(q)
        torsion_free = all(d == 1 for d in factors)
        torsion_order = math.prod(factors) if factors else 1
        free_rank = mat.shape[0] - len(factors)
        full_rank = len(factors) == min(mat.shape)
        rw = self._fase1_rademacher_walsh_norm(q)
        return rw, factors, torsion_free, torsion_order, free_rank, full_rank

    def evaluate_novikov_krull_valuation(self, exponents: NDArray[np.float64]) -> Tuple[float, float]:
        """Valuación de Krull v = min λᵢ y norma ‖A‖_Nov = e^{−v}. Fail-closed en vacío/negativo."""
        exps = _as_exponents(exponents)
        v = float(np.min(exps))
        return v, math.exp(-v)

    def evaluate_multilateral_associators(
        self, mat: NDArray[np.float64]
    ) -> MultilateralAssociatorMitigationReport:
        """Asociadores k-vías genuinos (5/7/9 actores) normalizados por el defecto algebraico.
        E‖[x,y,z]‖² = defecto² para vectores unitarios isótropos independientes; el embedding
        es determinista y dependiente, por lo que la razón es un indicador declarado."""
        rms_list, max_list, ratio_list, comb_list, defect_list = [], [], [], [], []
        for n, k in zip(_ASSOC_DIMS, _ASSOC_ACTORS):
            actors = _cd_embed_actors(mat, n, k)
            rms, mx = _cd_local_associator_stats(actors, n)
            comb = _cd_comb_discrepancy(actors, n)
            defect, _ = _cd_associator_defect(n)
            rms_list.append(rms)
            max_list.append(mx)
            ratio_list.append(rms / defect if defect > 0.0 else 0.0)
            comb_list.append(comb)
            defect_list.append(defect)
        suspected = bool(any(r > _ASSOC_RATIO_SUSPECT for r in ratio_list))
        return MultilateralAssociatorMitigationReport(
            pationionic_associator_norm=rms_list[0],
            chingonic_associator_norm=rms_list[1],
            routonic_associator_norm=rms_list[2],
            pationionic_max_norm=max_list[0],
            chingonic_max_norm=max_list[1],
            routonic_max_norm=max_list[2],
            pationionic_ratio=ratio_list[0],
            chingonic_ratio=ratio_list[1],
            routonic_ratio=ratio_list[2],
            comb_discrepancy_32d=comb_list[0],
            comb_discrepancy_64d=comb_list[1],
            comb_discrepancy_128d=comb_list[2],
            algebra_defect_32d=defect_list[0],
            algebra_defect_64d=defect_list[1],
            algebra_defect_128d=defect_list[2],
            phantom_collusion_suspected=suspected,
            actor_counts=_ASSOC_ACTORS,
            embedding_policy="flat_row_major→roll(i·stride)→resize(n)→ℓ2_unit; triples locales + combs",
        )

    # ── ÚLTIMO método de Fase 1 → inicio de Fase 2 ─────────────────────────────
    def _fase1_emit_algebraic_seed(
        self, mat: NDArray[np.float64], exps: NDArray[np.float64], input_digest: str
    ) -> _AlgebraicSeed:
        """ÚLTIMO método de F1. Emite la semilla algebraica que consume F2."""
        rw, factors, torsion_free, torsion_order, free_rank, full_rank = (
            self.compute_rademacher_walsh_and_snf(mat)
        )
        krull, novikov = self.evaluate_novikov_krull_valuation(exps)
        assoc = self.evaluate_multilateral_associators(mat)
        # Política declarada: anomalía = desviación respecto de la identidad (sin interacción).
        anomaly = float(np.linalg.norm(mat - np.eye(mat.shape[0], mat.shape[1])))
        return _AlgebraicSeed(
            matrix=mat, exponents=exps, snf_diagonal_factors=factors,
            torsion_free=torsion_free, torsion_order=torsion_order,
            cokernel_free_rank=free_rank, full_rank=full_rank, rw_norm=rw,
            krull=krull, novikov_norm=novikov, anomaly_norm=anomaly,
            associators=assoc, input_digest=input_digest,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — Geometría simpléctica, Poincaré y aniquilación de Dirac
# ══════════════════════════════════════════════════════════════════════════════
class _Fase2_Geometria_Celeste(_Fase1_Algebra_Hipercompleja):
    """Fase 2. Gromov, armonicidad periódica, Melnikov, Lindstedt, Lyapunov y Dirac."""

    def _fase2_inicio_desde_semilla_algebraica(
        self, seed: _AlgebraicSeed, ball_r: float, cylinder_R: float
    ) -> Tuple[float, float, bool]:
        """PRIMER método de F2. Consume la semilla de F1 → capacidades y violación de Gromov."""
        if not isinstance(seed, _AlgebraicSeed):
            raise TypeError("Fase 2 exige una _AlgebraicSeed emitida por Fase 1.")
        r = _validate_positive(ball_r, "ball_r")
        R = _validate_positive(cylinder_R, "cylinder_R")
        return math.pi * r * r, math.pi * R * R, bool(r > R)

    @staticmethod
    def _fase2_cauchy_riemann_residual(mat: NDArray[np.float64]) -> float:
        """Armonicidad ‖Δφ‖_F/(‖φ‖_F+ε) con Laplaciano periódico de 5 puntos.
        CR-II se anula por conmutación de derivadas parciales (declarado, no medido)."""
        lap = (np.roll(mat, 1, 0) + np.roll(mat, -1, 0)
               + np.roll(mat, 1, 1) + np.roll(mat, -1, 1) - 4.0 * mat)
        return float(np.linalg.norm(lap) / (np.linalg.norm(mat) + _WILKINSON_EPSILON))

    def solve_cauchy_riemann_floer_surface(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = _DEFAULT_BALL_R,
        cylinder_R: float = _DEFAULT_CYLINDER_R,
    ) -> Tuple[float, bool]:
        """Devuelve (residuo de armonicidad, violación de Gromov)."""
        mat = _as_cost_matrix(apu_cost_matrix)
        _, _, violated = self._fase2_inicio_desde_semilla_algebraica(
            self._fase1_emit_algebraic_seed(mat, np.array([_DEFAULT_KRULL_FLOOR]), ""),
            ball_r, cylinder_R,
        )
        return self._fase2_cauchy_riemann_residual(mat), violated

    @staticmethod
    def _fase2_melnikov_coefficients(delta: float, gamma: float, omega: float) -> Tuple[float, float]:
        """Separatriz ẋ_h = 2 sech t: ∫ẋ_h² = 8 exacto; ∫ẋ_h cos(ωt) = 2π sech(πω/2)."""
        d = _validate_finite(delta, "melnikov_delta")
        g = _validate_finite(gamma, "melnikov_gamma")
        w = _validate_finite(omega, "melnikov_omega")
        return 2.0 * math.pi * abs(g) / math.cosh(math.pi * w / 2.0), -8.0 * d

    def _fase2_melnikov_quadrature(self, delta, gamma, omega) -> Tuple[float, float]:
        """Cuadratura de control (solo autoprueba metamórfica)."""
        t = np.linspace(-_MELNIKOV_HALF_WIDTH, _MELNIKOV_HALF_WIDTH, _MELNIKOV_POINTS)
        v = 2.0 / np.cosh(t)
        return abs(gamma) * abs(_trapz(v * np.cos(omega * t), t)), -delta * _trapz(v * v, t)

    def evaluate_poincare_kam_melnikov(self, delta, gamma, omega):
        """(caos, amplitud, offset, margen, régimen ε válido). Cero simple ⇔ ω≠0 ∧ A>|offset|."""
        amp, off = self._fase2_melnikov_coefficients(delta, gamma, omega)
        chaos = bool(omega != 0.0 and amp > abs(off))
        margin = float(amp - abs(off))
        eps_valid = bool(max(abs(delta), abs(gamma)) <= _MELNIKOV_EPS_POLICY)
        return chaos, amp, off, margin, eps_valid

    @staticmethod
    def _fase2_agm(a: float, b: float) -> float:
        """Media aritmético-geométrica (convergencia cuadrática)."""
        for _ in range(60):
            if abs(a - b) <= 1.0e-15 * max(abs(a), abs(b)):
                return 0.5 * (a + b)
            a, b = 0.5 * (a + b), math.sqrt(a * b)
        return 0.5 * (a + b)

    def _fase2_poincare_lindstedt(self, amplitude: float) -> Tuple[float, float, float]:
        """Ω(a) exacta = AGM(1,cos(a/2)) frente a la serie 1 − a²/16 + a⁴/3072 (error O(a⁶))."""
        a = _validate_amplitude(amplitude)
        exact = self._fase2_agm(1.0, math.cos(a / 2.0))
        series = 1.0 - a * a / 16.0 + a ** 4 / 3072.0
        return exact, series, abs(exact - series)

    def _fase2_poincare_stroboscopic_lyapunov(
        self, delta, gamma, omega, periods, steps
    ) -> Optional[float]:
        """λ_max finito del mapa estroboscópico de Poincaré (Benettin). EVIDENCIA, no prueba."""
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

    def execute_dirac_annihilation(
        self, anomaly_norm: float, positron_hmac_valid: Optional[bool]
    ) -> float:
        """e⁻ + e⁺ → 2γ: 2·m_e·c² = 1021.9979 keV (511 keV por fotón en el CM), o 0.
        Requiere positrón validado Y anomalía sobre el umbral (política declarada)."""
        positron = validate_positron(positron_hmac_valid)
        if positron is not True:
            return 0.0
        if not (math.isfinite(anomaly_norm) and anomaly_norm > _ANOMALY_THRESHOLD):
            return 0.0
        return _DIRAC_TOTAL_KEV

    # ── ÚLTIMO método de Fase 2 → inicio de Fase 3 ─────────────────────────────
    def _fase2_emit_mitigation_seed(
        self,
        seed: _AlgebraicSeed,
        ball_r: float,
        cylinder_R: float,
        melnikov_delta: float,
        melnikov_gamma: float,
        melnikov_omega: float,
        pendulum_amplitude: float,
        poincare_periods: int,
        poincare_steps: int,
        positron_hmac_valid: Optional[bool],
    ) -> _MitigationSeed:
        """ÚLTIMO método de F2. Emite la semilla de mitigación que consume F3.
        La recomendación solo usa fallos CERTIFICADOS; los asociadores son evidencia."""
        c_ball, c_cyl, violated = self._fase2_inicio_desde_semilla_algebraica(seed, ball_r, cylinder_R)
        cr = self._fase2_cauchy_riemann_residual(seed.matrix)
        chaos, amp, off, margin, eps_valid = self.evaluate_poincare_kam_melnikov(
            melnikov_delta, melnikov_gamma, melnikov_omega
        )
        exact, series, lerr = self._fase2_poincare_lindstedt(pendulum_amplitude)
        lyap = self._fase2_poincare_stroboscopic_lyapunov(
            melnikov_delta, melnikov_gamma, melnikov_omega, poincare_periods, poincare_steps
        )
        # Tensión UNIDIRECCIONAL: λ>tol sin caos de Melnikov ⇒ tensión (Melnikov es O(ε)).
        # Melnikov con caos y λ≤tol NO es tensión: el conjunto homoclínico puede no ser atractor.
        consistent = None if lyap is None else bool((not (lyap > _LYAP_POS_TOL)) or chaos)
        dirac = self.execute_dirac_annihilation(seed.anomaly_norm, positron_hmac_valid)
        recommended = bool(
            violated or (not seed.torsion_free) or chaos or (seed.krull < _DEFAULT_KRULL_FLOOR)
        )
        return _MitigationSeed(
            algebraic=seed, cr_residual=cr, gromov_violated=violated,
            capacity_ball=c_ball, capacity_cylinder=c_cyl, melnikov_chaos=chaos,
            melnikov_amplitude=amp, melnikov_offset=off, melnikov_margin=margin,
            melnikov_eps_valid=eps_valid, lindstedt_exact=exact, lindstedt_series=series,
            lindstedt_error=lerr, lyapunov=lyap, lyapunov_consistent=consistent,
            dirac_kev=dirac, mitigation_recommended=recommended,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — Kleisli T³, RSI³ reflexivo, Heyting y pasaporte
# ══════════════════════════════════════════════════════════════════════════════
class _Fase3_Kleisli_RSI3(_Fase2_Geometria_Celeste):
    """Fase 3. Contracción de Banach, torre RSI³, adjudicación Heyting y pasaporte."""

    def _fase3_inicio_desde_semilla_mitigacion(
        self, seed: _MitigationSeed, depth: int, lam: float
    ) -> Tuple[_MitigationSeed, int, float]:
        """PRIMER método de F3. Recomendación certificada ⇒ profundidad efectiva 0."""
        if not isinstance(seed, _MitigationSeed):
            raise TypeError("Fase 3 exige una _MitigationSeed emitida por Fase 2.")
        d = _validate_int(depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        l = _validate_lambda(lam)
        return seed, (0 if seed.mitigation_recommended else d), l

    @staticmethod
    def _fase3_certified_bound(lam: float) -> float:
        """Cota certificada λ³+ε. |T'|≤λ en ℝ ⇒ Lip(T³)≤λ³ (analítica global)."""
        return float(lam) ** 3 + _KLEISLI_ERROR_BUDGET

    @staticmethod
    def _fase3_kleisli_cubed(x, lam: float):
        y = x
        for _ in range(3):
            y = lam * np.cos(y) + _KLEISLI_OFFSET
        return y

    @staticmethod
    def _fase3_kleisli_cubed_derivative(x, lam: float):
        """|(T³)'(x)| = λ³|sin x · sin T(x) · sin T²(x)| (regla de la cadena exacta)."""
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
        """sup|(T³)'| en malla diádica anidada ⇒ monotonía por inclusión de conjuntos."""
        n = _KLEISLI_GRID_BASE * (2 ** level) + 1
        grid = np.linspace(-_KLEISLI_DOMAIN, _KLEISLI_DOMAIN, n)
        return float(np.max(self._fase3_kleisli_cubed_derivative(grid, lam)))

    def _fase3_rsi_recurse(self, lam, q, max_depth, gate_ok, meta_ok):
        accepted = 0
        history: List[float] = []
        prev = -math.inf
        chain = hashlib.sha256(_RSI_CHAIN_DOMAIN).digest()
        for level in range(1, max_depth + 1):
            e = self._fase3_empirical_lipschitz(level, lam)
            history.append(e)
            if not (gate_ok and meta_ok and e <= q + _LIP_TOL and e >= prev - _LIP_TOL):
                break
            accepted = level
            prev = e
            chain = hashlib.sha256(chain + f"|{level}|{_fmt(e)}|{int(meta_ok)}".encode()).digest()
        return accepted, tuple(history), chain.hex()

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
        accepted, history, chain = self._fase3_rsi_recurse(
            lam, q, depth, selftest and consistency, meta
        )
        return _KleisliState(
            lip_analytic=analytic, lip_certified=q, lip_empirical=history,
            fixed_point=x, a_posteriori_error=a_post, residual=res, meta_verified=meta,
            selftest_passed=selftest, consistency_ok=consistency,
            depth_accepted=accepted, certificate_chain=chain,
        )

    def _fase3_metamorphic_selftest(self) -> bool:
        """Autoprueba fail-closed: CD, SNF, Parseval, Melnikov, AGM, CR, Heyting, RSI, Dirac."""
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
            # 2. SNF con cadena de divisibilidad (invariante del cokernel).
            if self._fase1_smith_diagonal([[2, 0], [0, 3]]) != (1, 6):
                return False
            if self._fase1_smith_diagonal([[1, 1], [0, 1]]) != (1, 1):
                return False
            if self._fase1_smith_diagonal([[6, 0], [0, 10]]) != (2, 30):
                return False
            f = self._fase1_smith_diagonal([[2, 4], [6, 8]])
            if math.prod(f) != 8 or any(b % a for a, b in zip(f, f[1:])):
                return False
            # 3. Parseval e identidad espectral de influencias (semilla 20250101, N=16).
            rng = np.random.default_rng(20250101)
            bits = rng.integers(0, 2, size=16).astype(np.int64)
            ghat = self._fase1_fwht(1.0 - 2.0 * bits) / 16.0
            if abs(float(np.sum(ghat ** 2)) - 1.0) > 1e-12:
                return False
            if abs(float(sum(bin(s).count("1") * ghat[s] ** 2 for s in range(16)))
                   - self._fase1_total_influence(bits)) > 1e-12:
                return False
            # 4. Melnikov cerrada vs cuadratura de control.
            amp, off = self._fase2_melnikov_coefficients(0.01, 0.02, 1.0)
            aq, oq = self._fase2_melnikov_quadrature(0.01, 0.02, 1.0)
            if abs(amp - aq) > 1e-6 * amp or abs(off - oq) > 1e-6 or abs(off + 0.08) > 1e-12:
                return False
            # 5. Lindstedt–Poincaré: AGM exacta vs serie.
            om0, _, _ = self._fase2_poincare_lindstedt(0.0)
            if om0 != 1.0:
                return False
            _, _, err02 = self._fase2_poincare_lindstedt(0.2)
            if err02 > 2e-6:
                return False
            # 6. Laplaciano periódico: autofunción discreta (suma de modos 1D).
            ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
            phi = np.cos(2.0 * np.pi * ii / 8.0) + np.cos(2.0 * np.pi * jj / 8.0)
            lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
                   + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4.0 * phi)
            eig = 2.0 * (math.cos(2.0 * math.pi / 8.0) - 1.0)
            if np.linalg.norm(lap - eig * phi) > 1e-12 * np.linalg.norm(phi):
                return False
            # 7. Heyting: residuación a∧b ≤ c ⇔ a ≤ (b→c) exhaustiva.
            vals = (HeytingTruthValue.VETOED, HeytingTruthValue.DEGRADED, HeytingTruthValue.COHERENT)
            for a in vals:
                for b in vals:
                    for c in vals:
                        if (a.meet(b).rank <= c.rank) != (a.rank <= b.implies(c).rank):
                            return False
            # 8. Banach, monotonía RSI por anidamiento y energía de Dirac.
            if abs(self._fase3_certified_bound(0.95) - 0.85975) > 1e-12:
                return False
            if not (self._fase3_empirical_lipschitz(1, 0.95)
                    <= self._fase3_empirical_lipschitz(2, 0.95) + _LIP_TOL):
                return False
            if abs(_DIRAC_TOTAL_KEV - 1021.9979) > 1e-9:
                return False
            if self.execute_dirac_annihilation(1.0, True) != _DIRAC_TOTAL_KEV:
                return False
            if self.execute_dirac_annihilation(1.0, None) != 0.0:
                return False
            # 9. Novikov fail-closed y asociador nulo sobre matriz nula.
            try:
                self.evaluate_novikov_krull_valuation(np.array([]))
                return False
            except ValueError:
                pass
            zero_assoc = self.evaluate_multilateral_associators(np.zeros((2, 2)))
            if zero_assoc.pationionic_associator_norm != 0.0:
                return False
            if zero_assoc.phantom_collusion_suspected:
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica: excepción (fail-closed).")
            return False

    def _fase3_heyting_adjudicate(self, s: _MitigationSeed, k: _KleisliState, requested: int):
        """Veto: criterios certificados. Degradado: evidencia/heurística. El defecto de
        rango real NO veta (corrección explícita de v13)."""
        a = s.algebraic
        spectral = (HeytingTruthValue.VETOED
                    if (not a.torsion_free) or a.krull < _DEFAULT_KRULL_FLOOR
                    else (HeytingTruthValue.DEGRADED if a.rw_norm > _RW_DEGRADED_TOL
                          else HeytingTruthValue.COHERENT))
        if s.gromov_violated or s.melnikov_chaos:
            dynamical = HeytingTruthValue.VETOED
        elif (s.cr_residual > _CR_RESIDUAL_TOL or not s.melnikov_eps_valid
              or s.lyapunov_consistent is False):
            dynamical = HeytingTruthValue.DEGRADED
        else:
            dynamical = HeytingTruthValue.COHERENT
        assoc_judge = (HeytingTruthValue.DEGRADED
                       if a.associators.phantom_collusion_suspected
                       else HeytingTruthValue.COHERENT)
        rsi = (HeytingTruthValue.COHERENT if k.depth_accepted == requested
               else HeytingTruthValue.DEGRADED)
        verifier = (HeytingTruthValue.COHERENT
                    if (k.selftest_passed and k.consistency_ok and k.meta_verified)
                    else HeytingTruthValue.VETOED)
        return spectral.meet(dynamical).meet(assoc_judge).meet(rsi).meet(verifier)

    def _fase3_sign_passport(self, fields: Tuple[object, ...]) -> str:
        payload = "|".join(_fmt(x) for x in fields).encode("utf-8")
        return hmac.new(self._secret_key, _PASSPORT_DOMAIN + payload, hashlib.sha256).hexdigest()

    # ── ÚLTIMO método de Fase 3: API pública ───────────────────────────────────
    def execute_phantom_collusion_mitigation_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        positron_hmac_valid: Optional[bool] = None,
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
    ) -> PhantomCollusionSpectrumReport:
        """Auditoría completa de mitigación anti-colusión. ÚLTIMO método de F3; API pública."""
        mat = _as_cost_matrix(apu_cost_matrix)
        exps = _as_exponents(exponents)
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        lam = _validate_lambda(initial_lip)
        n_per = _validate_int(poincare_periods, "poincare_periods", 0, 20000)
        n_st = _validate_int(poincare_steps, "poincare_steps", 8, 4096)
        amp = _validate_amplitude(pendulum_amplitude)
        positron = validate_positron(positron_hmac_valid)
        series_str = ("none" if capacity_series is None
                      else ",".join(_fmt(float(x)) for x in capacity_series))
        params = "|".join((
            _fmt(float(ball_r)), _fmt(float(cylinder_R)),
            _fmt(float(melnikov_delta)), _fmt(float(melnikov_gamma)), _fmt(float(melnikov_omega)),
            _fmt(amp), _fmt(n_per), _fmt(n_st), _fmt(depth), _fmt(lam),
            _fmt(positron), series_str,
        ))
        digest = _input_digest(mat, exps, params)
        seed = self._fase1_emit_algebraic_seed(mat, exps, digest)
        mit = self._fase2_emit_mitigation_seed(
            seed, ball_r, cylinder_R, melnikov_delta, melnikov_gamma, melnikov_omega,
            amp, n_per, n_st, positron,
        )
        mit, eff_depth, lam = self._fase3_inicio_desde_semilla_mitigacion(mit, depth, lam)
        kleisli = self.evaluate_rsi3_banach_kleisli(lam, eff_depth)
        verdict = self._fase3_heyting_adjudicate(mit, kleisli, eff_depth)
        jerk = _third_difference(capacity_series)
        assoc = seed.associators
        passport = self._fase3_sign_passport((
            _ENGINE_VERSION, verdict, digest, seed.rw_norm, seed.torsion_free,
            ",".join(str(d) for d in seed.snf_diagonal_factors), seed.torsion_order,
            seed.cokernel_free_rank, seed.full_rank, seed.krull, seed.novikov_norm,
            seed.anomaly_norm, mit.cr_residual, mit.gromov_violated,
            mit.gromov_capacity_ball, mit.gromov_capacity_cylinder,
            mit.melnikov_chaos, mit.melnikov_eps_valid, mit.lyapunov, mit.lindstedt_exact,
            assoc.phantom_collusion_suspected,
            ",".join(_fmt(r) for r in (assoc.pationionic_ratio, assoc.chingonic_ratio, assoc.routonic_ratio)),
            mit.dirac_kev, mit.mitigation_recommended,
            kleisli.lip_analytic, kleisli.lip_certified, kleisli.depth_accepted,
            kleisli.certificate_chain,
            ",".join(_fmt(e) for e in kleisli.lip_empirical),
            kleisli.selftest_passed, kleisli.consistency_ok, kleisli.meta_verified, jerk,
        ))
        return PhantomCollusionSpectrumReport(
            rademacher_walsh_norm=seed.rw_norm, smith_torsion_free=seed.torsion_free,
            smith_torsion_order=seed.torsion_order, cokernel_free_rank=seed.cokernel_free_rank,
            full_rank=seed.full_rank, snf_diagonal_factors=seed.snf_diagonal_factors,
            krull_valuation=seed.krull, novikov_norm=seed.novikov_norm,
            cauchy_riemann_residual=mit.cr_residual, is_gromov_violated=mit.gromov_violated,
            gromov_capacity_ball=mit.capacity_ball, gromov_capacity_cylinder=mit.capacity_cylinder,
            melnikov_chaos_detected=mit.melnikov_chaos, melnikov_amplitude=mit.melnikov_amplitude,
            melnikov_offset=mit.melnikov_offset, melnikov_transversality_margin=mit.melnikov_margin,
            melnikov_eps_regime_valid=mit.melnikov_eps_valid,
            poincare_lindstedt_exact=mit.lindstedt_exact, poincare_lindstedt_series=mit.lindstedt_series,
            poincare_lindstedt_error=mit.lindstedt_error,
            poincare_lyapunov=mit.lyapunov, poincare_melnikov_consistent=mit.lyapunov_consistent,
            banach_lip_analytic=kleisli.lip_analytic, banach_contraction_lip=kleisli.lip_certified,
            banach_fixed_point=kleisli.fixed_point, banach_a_posteriori_error=kleisli.a_posteriori_error,
            banach_meta_verified=kleisli.meta_verified, rsi_accepted_depth=kleisli.depth_accepted,
            rsi_empirical_lipschitz=kleisli.lip_empirical, rsi_certificate_chain=kleisli.certificate_chain,
            capacity_third_difference=jerk, anomaly_norm=seed.anomaly_norm,
            dirac_annihilation_kev=mit.dirac_kev, associator_report=assoc,
            phantom_collusion_suspected=assoc.phantom_collusion_suspected,
            mitigation_recommended=mit.mitigation_recommended,
            metamorphic_selftest_passed=kleisli.selftest_passed, input_digest=digest,
            truth_value=verdict, passport_hash=passport, schema_version=_SCHEMA_VERSION,
        )


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPhantomCollusionMitigatorEngine(_Fase3_Kleisli_RSI3):
    """Motor raíz de detección y mitigación de colusiones fantasma (v14.0.0)."""


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = BoolNovikovPhantomCollusionMitigatorEngine(secret_key=b"demo-key-0123456789abcdef")
    print(f"Autoprueba metamórfica: {engine._fase3_metamorphic_selftest()}")
    exps_test = np.array([0.15, 0.25, 0.40, 0.85])
    # A: cuantiza a diag 105 / off 5 ⇒ gcd 5 ⇒ torsión ⇒ mitigación certificada.
    mat_a = np.eye(4) + 0.05 * np.ones((4, 4))
    # B: cuantiza a [[1,1],[0,1]] unimodular ⇒ sin torsión.
    mat_b = np.array([[0.01, 0.01], [0.0, 0.01]])
    for label, mat in (("A (interacción densa)", mat_a), ("B (unimodular)", mat_b)):
        rep = engine.execute_phantom_collusion_mitigation_audit(
            mat, exps_test, positron_hmac_valid=True, capacity_series=(1.0, 2.0, 4.0, 8.0)
        )
        ar = rep.associator_report
        print(f"\n=== Escenario {label}: {rep.truth_value.name} ({hex(rep.truth_value.value)}) ===")
        print(f"  SNF (cadena d₁|d₂|…)      : {rep.snf_diagonal_factors}  torsión libre={rep.smith_torsion_free}"
              f"  |tors|={rep.smith_torsion_order}  rango libre={rep.cokernel_free_rank}")
        print(f"  Rademacher–Walsh          : {rep.rademacher_walsh_norm:.4f}")
        print(f"  Krull v(A) / ‖A‖_Nov      : {rep.krull_valuation:.4f} / {rep.novikov_norm:.4f}")
        print(f"  Asociadores RMS 32/64/128 : {ar.pationionic_associator_norm:.4f} / "
              f"{ar.chingonic_associator_norm:.4f} / {ar.routonic_associator_norm:.4f}")
        print(f"  Razones vs defecto        : {ar.pationionic_ratio:.4f} / {ar.chingonic_ratio:.4f} / {ar.routonic_ratio:.4f}")
        print(f"  Colusión sospechada       : {rep.phantom_collusion_suspected} (heurístico)")
        print(f"  Mitigación recomendada    : {rep.mitigation_recommended} (certificado)")
        print(f"  CR residual / Gromov      : {rep.cauchy_riemann_residual:.3e} / {rep.is_gromov_violated}")
        print(f"  Melnikov caos / margen    : {rep.melnikov_chaos_detected} / {rep.melnikov_transversality_margin:.5f}")
        lyap = "None" if rep.poincare_lyapunov is None else f"{rep.poincare_lyapunov:.5f}"
        print(f"  Poincaré λ_max (evid.)    : {lyap}  consistente={rep.poincare_melnikov_consistent}")
        print(f"  Banach λ³/λ³+ε / x*       : {rep.banach_lip_analytic:.5f}/{rep.banach_contraction_lip:.5f}"
              f" / {rep.banach_fixed_point:.6f}  meta={rep.banach_meta_verified}")
        print(f"  RSI³ profundidad / jerk   : {rep.rsi_accepted_depth} / {rep.capacity_third_difference}")
        print(f"  Anomalía ‖M−I‖ / Dirac    : {rep.anomaly_norm:.4f} / {rep.dirac_annihilation_kev:.4f} keV")
        print(f"  Pasaporte HMAC            : {rep.passport_hash[:16]}...")
    print("\n=== TEST COMPLETADO ===")