# -*- coding: utf-8 -*-
r"""
Módulo   : BoolNovikovToonCartridgeWeaverEngine v15.0.0
Ubicación: app/wisdom/bool_novikov_toon_cartridge_weaver_engine.py
Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-POINCARE-RSI3-PhD

Motor espectral de tejeduría TOON por fases anidadas (herencia lineal):

  _Fase1_Cimientos      álgebra booleana, forma de Smith sobre ℤ con certificado de
                        determinante (Bareiss exacto), Novikov–Krull, Kepler con
                        salvaguarda, estado orbital independiente (vis-viva, |h|, h_z),
                        Delaunay → variables canónicas de Poincaré.
    └ _fase1_emit_spectral_seed            ÚLTIMO método de F1 → semilla que inicia F2.
  _Fase2_Geometria      Gromov (política r≤R), residuo de armonicidad discreta,
                        Melnikov–Poincaré en forma cerrada, mapa estroboscópico de
                        Poincaré con Lyapunov de Benettin (evidencia), frecuencia de
                        Lindstedt–Poincaré exacta por AGM, espectro secular de
                        Poincaré–Laplace–Lagrange, Spohn y aniquilación de Dirac.
    └ _fase2_emit_dynamical_seed           ÚLTIMO método de F2 → semilla que inicia F3.
  _Fase3_Kleisli_RSI3   RSI³ = torre reflexiva de 3 niveles con certificados SHA-256
                        encadenados y meta-certificado de procedimiento, Banach/Picard
                        con cota analítica λ³, Heyting (⊥<δ<⊤), pasaporte HMAC fail-closed.
    └ execute_toon_cartridge_weaver_audit  ÚLTIMO método de F3 (API pública).

Cambios v14.0 → v15.0 (granulares, por fase):
  F1 · Certificado det: para cuadradas no singulares, |det| = ∏|dᵢ| (Bareiss entero).
  F1 · Estado orbital: r, v en ECI (rotación 3-1-3); vis-viva v²=μ(2/r−1/a);
       |r×v| = G; (r×v)_z = H. Valida la solución de Kepler y las acciones de Delaunay.
  F2 · «Cauchy–Riemann» renombrado a residuo de armonicidad (campo conservado).
       Se retira la afirmación no medida sobre CR-II.
  F2 · Matriz secular coherente: aproximación M☉+mⱼ≈M☉ aplicada en diagonal y fuera.
  F3 · Cota certificada = λ³ (demostración). λ³+ε es margen conservador (error a posteriori).
  F3 · Adjudicación: un veto que fuerza profundidad 0 ya no figura como RSI³ coherente.
  F3 · Meta-certificado de procedimiento: la cadena SHA-256 ata las constantes de
       verificación. «Automejora» = refinamiento de verificación; sin automodificación.
  Global · allow_dev_key=False por defecto (fail-closed). Kepler: 200 iteraciones.

Límites declarados (no se afirma más de lo que se prueba):
  · Smith: sobre la matriz CUANTIZADA a escala fija 100 (política, no derivada).
    El certificado de determinante solo aplica a cuadradas no singulares.
  · «Rademacher–Walsh»: influencia total normalizada de b(x)=[entrada>0], relleno a 2ⁿ.
  · Gromov: en ℝ² es monotonía de área; el no-squeezing profundo es n≥2. Aquí es
    política de diseño r≤R.
  · Melnikov: riguroso solo para ε pequeño; fuera de régimen se DEGRADA.
  · Lyapunov finito: evidencia dependiente de la órbita sonda (IC declarado).
  · Órbita: problema de dos cuerpos sin perturbaciones; vis-viva certifica la solución
    de Kepler, no el modelo de fuerzas.
  · Secular: teoría lineal (orden e²), M☉=1, μ=4π² AU³/año², aproximación mⱼ≪M☉
    coherente; modelo juguete.
  · Kleisli: T(x)=λ·cos x+0.1 es supuesto del autor.
  · Jerk: tercera diferencia MEDIDA; no se postula signo.
  · No modela disparo crowbar ni escribe en hardware.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverEngine")

# ── Versiones ──────────────────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-POINCARE-RSI3-PhD"
)
version: Final[str] = _ENGINE_VERSION
_SCHEMA_VERSION: Final[str] = "engine-report-v15.0"

# ── Clave HMAC y dominios de separación ────────────────────────────────────────
_HMAC_KEY_ENV: Final[str] = "APU_TOON_WEAVER_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_ENGINE_DEV_ONLY_TOON_WEAVER_KEY_v15.0.0"
_PASSPORT_DOMAIN: Final[bytes] = b"TOON-WEAVER-ENGINE-PASSPORT-v15.0|"
_RSI_CHAIN_DOMAIN: Final[bytes] = b"TOON-WEAVER-RSI3-CERT-CHAIN-v15.0|"

# ── Numérica general ───────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_LIP_TOL: Final[float] = 1.0e-12
_CONSISTENCY_TOL: Final[float] = 1.0e-9

# ── Álgebra y compresión ───────────────────────────────────────────────────────
_TOON_SOURCE_TOKENS: Final[int] = 412
_TOON_BUDGET_TOKENS: Final[int] = 56
_SNF_FIXED_POINT_SCALE: Final[float] = 100.0      # política declarada, no derivada
_SNF_MAX_ABS_INT: Final[int] = 1 << 62
_RW_DEGRADED_TOL: Final[float] = 0.5
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10

# ── Geometría ──────────────────────────────────────────────────────────────────
_DEFAULT_BALL_R: Final[float] = 0.35
_DEFAULT_CYLINDER_R: Final[float] = 0.50
_HARMONICITY_TOL: Final[float] = 1.0e-3
_MELNIKOV_HALF_WIDTH: Final[float] = 40.0         # solo autoprueba de cuadratura
_MELNIKOV_POINTS: Final[int] = 20001
_MELNIKOV_DEFAULT_DELTA: Final[float] = 0.01
_MELNIKOV_DEFAULT_GAMMA: Final[float] = 0.02
_MELNIKOV_DEFAULT_OMEGA: Final[float] = 1.0
_MELNIKOV_EPS_POLICY: Final[float] = 0.10         # régimen perturbativo declarado

# ── Mecánica celeste y Poincaré ────────────────────────────────────────────────
_SYMPLECTIC_TOL: Final[float] = 1.0e-8
_COMPLEX_STEP: Final[float] = 1.0e-20             # paso complejo: derivada exacta a redondeo
_ORBIT_RESIDUAL_TOL: Final[float] = 1.0e-10
_DEFAULT_PENDULUM_AMPLITUDE: Final[float] = 0.2
_POINCARE_PERIODS: Final[int] = 256
_POINCARE_STEPS_PER_PERIOD: Final[int] = 64
_POINCARE_IC: Final[Tuple[float, float]] = (0.0, 1.9)   # sonda cerca de la separatriz
_LYAP_POS_TOL: Final[float] = 1.0e-2              # resolución finita declarada
_SECULAR_DEFAULT: Final[Tuple[Tuple[float, float], ...]] = (
    (9.5458e-4, 5.2044), (2.858e-4, 9.5826),      # (m/M☉, a[AU]) Júpiter, Saturno
)
_SECULAR_QUAD_POINTS: Final[int] = 4096
_SECULAR_SYM_TOL: Final[float] = 1.0e-10
_SECULAR_MAX_PLANETS: Final[int] = 8

# ── Entropía y aniquilación ────────────────────────────────────────────────────
_SPOHN_LAMBDA: Final[float] = 0.1
_SPOHN_TOL: Final[float] = -1.0e-12
_ANOMALY_SCALE: Final[float] = 0.1                # política: anomalía = 0.1·‖M‖_F
_ANOMALY_THRESHOLD: Final[float] = 1.0e-6
_POSITRON_REST_MASS_KEV: Final[float] = 2.0 * 510.99895   # 2·m_e·c² = 1021.9979 keV

# ── Mecánica celeste (Tierra) ──────────────────────────────────────────────────
_MU_EARTH_KM3_S2: Final[float] = 398600.4418
_DEFAULT_ORBIT: Final[Tuple[float, ...]] = (7000.0, 0.01, 0.9, 1.0, 0.5, 0.2)
_KEPLER_TOL: Final[float] = 1.0e-13
_KEPLER_MAX_ITER: Final[int] = 200
_KEPLER_RESIDUAL_TOL: Final[float] = 1.0e-10

# ── Kleisli / RSI³ ─────────────────────────────────────────────────────────────
_KLEISLI_LIP_T: Final[float] = 0.95
_KLEISLI_ERROR_BUDGET: Final[float] = 0.002375    # margen conservador declarado
_KLEISLI_OFFSET: Final[float] = 0.1
_KLEISLI_DOMAIN: Final[float] = 2.0
_KLEISLI_GRID_BASE: Final[int] = 100
_PICARD_TOL: Final[float] = 1.0e-12
_PICARD_MAX_ITER: Final[int] = 10000
_PICARD_REF_TOL: Final[float] = 1.0e-15
_PICARD_REF_MAX_ITER: Final[int] = 100000
_RSI_MAX_DEPTH: Final[int] = 3


# ══════════════════════════════════════════════════════════════════════════════
# Topos de Heyting: cadena ⊥ < δ < ⊤ (códigos de cable 0x03, 0x01, 0x00)
# ══════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    """Álgebra de Heyting de 3 elementos; residuación verificada en autoprueba."""

    COHERENT = 0x00   # ⊤
    DEGRADED = 0x01   # δ
    VETOED = 0x03     # ⊥

    @property
    def rank(self) -> int:
        if self is HeytingTruthValue.COHERENT:
            return 2
        if self is HeytingTruthValue.DEGRADED:
            return 1
        return 0

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """a → b = ⊤ si a ≤ b; en otro caso b (implicación de la cadena)."""
        return HeytingTruthValue.COHERENT if self.rank <= other.rank else other

    def negate(self) -> "HeytingTruthValue":
        return self.implies(HeytingTruthValue.VETOED)


# ══════════════════════════════════════════════════════════════════════════════
# Informes y semillas (contratos entre fases)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ToonCartridgeWeaverSpectrumReport:
    """Informe de atestación espectral y condensación TOON. Sin tiempos (reproducible)."""

    rademacher_walsh_norm: float
    smith_torsion_free: bool
    smith_torsion_order: int
    cokernel_free_rank: int               # rango libre del cokernel, NO β₁
    full_rank: bool
    snf_diagonal_factors: Tuple[int, ...]  # cadena de divisibilidad d₁|d₂|…|d_r
    snf_determinant_certificate: Optional[bool]
    krull_valuation: float
    novikov_norm: float
    raw_token_count: int
    compressed_token_count: int
    kv_cache_compression_pct: float
    cauchy_riemann_residual: float        # residuo de armonicidad discreta (ver docs)
    is_gromov_violated: bool
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    melnikov_chaos_detected: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_transversality_margin: float
    melnikov_eps_regime_valid: bool
    poincare_lyapunov: Optional[float]            # evidencia finita (Benettin)
    poincare_melnikov_consistent: Optional[bool]  # tensión unidireccional
    poincare_lindstedt_exact: float
    poincare_lindstedt_series: float
    poincare_lindstedt_error: float
    delaunay_LGH: Tuple[float, float, float]
    delaunay_valid: bool
    orbital_state_residual: Optional[float]       # vis-viva / |h| / h_z (relativo)
    orbital_state_ok: Optional[bool]
    poincare_variables: Tuple[float, float, float, float, float, float]
    poincare_canonical_residual: Optional[float]
    poincare_canonical_ok: Optional[bool]
    secular_frequencies: Tuple[float, ...]        # rad/año (Laplace–Lagrange)
    secular_symmetry_residual: float
    secular_valid: bool
    banach_lip_analytic: float                    # λ³: cota certificada (demostrada)
    banach_contraction_lip: float                 # = banach_lip_analytic (certificada)
    banach_safety_lip: float                      # λ³+ε (margen conservador)
    banach_fixed_point: float
    banach_a_posteriori_error: float
    banach_residual: float
    banach_meta_verified: bool
    rsi_accepted_depth: int
    rsi_empirical_lipschitz: Tuple[float, ...]
    rsi_certificate_chain: str
    spohn_entropy_prod: float
    capacity_third_difference: Optional[float]
    dirac_annihilation_kev: float
    metamorphic_selftest_passed: bool
    input_digest: str
    truth_value: HeytingTruthValue
    passport_hash: str
    schema_version: str


@dataclass(frozen=True)
class _SNFCertificate:
    """Resultado de la forma de Smith y de sus certificados."""

    rw_norm: float
    factors: Tuple[int, ...]
    torsion_free: bool
    full_rank: bool
    torsion_order: int
    free_rank: int
    det_ok: Optional[bool]


@dataclass(frozen=True, eq=False)
class _SpectralSeed:
    """Semilla espectral: salida de Fase 1, entrada de Fase 2."""

    matrix: NDArray[np.float64]
    exponents: NDArray[np.float64]
    snf_diagonal_factors: Tuple[int, ...]
    torsion_free: bool
    torsion_order: int
    full_rank: bool
    cokernel_free_rank: int
    snf_det_ok: Optional[bool]
    rw_norm: float
    krull: float
    novikov_norm: float
    raw_token_count: int
    compressed_token_count: int
    kv_cache_pct: float
    positron_valid: Optional[bool]
    anomaly_norm: float
    delaunay_LGH: Tuple[float, float, float]
    delaunay_valid: bool
    eccentric_anomaly: float
    orbital_residual: Optional[float]
    orbital_ok: Optional[bool]
    poincare_variables: Tuple[float, float, float, float, float, float]
    poincare_canonical_residual: Optional[float]
    poincare_canonical_ok: Optional[bool]
    input_digest: str


@dataclass(frozen=True, eq=False)
class _DynamicalSeed:
    """Semilla dinámica: salida de Fase 2, entrada de Fase 3."""

    spectral: _SpectralSeed
    capacity_ball: float
    capacity_cylinder: float
    gromov_violated: bool
    harmonicity_residual: float
    melnikov_chaos: bool
    melnikov_amplitude: float
    melnikov_offset: float
    melnikov_margin: float
    melnikov_eps_valid: bool
    poincare_lyapunov: Optional[float]
    poincare_consistent: Optional[bool]
    lindstedt_exact: float
    lindstedt_series: float
    lindstedt_error: float
    secular_frequencies: Tuple[float, ...]
    secular_sym_residual: float
    secular_valid: bool
    spohn_production: float
    dirac_kev: float
    force_veto: bool


@dataclass(frozen=True)
class _KleisliState:
    """Estado de la mónada Kleisli y de la torre reflexiva RSI³."""

    lip_T: float
    lip_analytic: float
    lip_safety: float
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
# Primitivas compartidas (validación fail-closed y serialización canónica)
# ══════════════════════════════════════════════════════════════════════════════
def _trapz(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Integración trapezoidal compatible con NumPy 1.x y 2.x."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _fmt(value: object) -> str:
    """Serialización canónica de un campo del pasaporte. Rechaza valores ambiguos."""
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
            raise ValueError("Campo canónico no finito; pasaporte abortado (fail-closed).")
        return f"{value:.17g}"
    if isinstance(value, str):
        if "|" in value:
            raise ValueError("Separador '|' prohibido en campo canónico.")
        return value
    raise TypeError(f"Tipo no canónico: {type(value).__name__}")


def _input_digest(
    mat: NDArray[np.float64], exps: NDArray[np.float64], params_canonical: str
) -> str:
    """SHA-256 de la entrada COMPLETA: forma y bytes float64 + parámetros canónicos."""
    h = hashlib.sha256()
    for name, arr in (("MIC", mat), ("EXP", exps)):
        h.update(f"{name}{arr.shape}".encode("ascii"))
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes(order="C"))
    h.update(b"PARAMS|" + params_canonical.encode("utf-8"))
    return h.hexdigest()


def _resolve_secret_key(explicit: Optional[bytes], allow_dev_key: bool) -> bytes:
    """Clave explícita > variable de entorno > clave de desarrollo (solo si se autoriza).
    Sin clave válida y sin autorización ⇒ ValueError (fail-closed)."""
    if explicit is not None:
        key = bytes(explicit)
    else:
        env_val = os.environ.get(_HMAC_KEY_ENV, "")
        if env_val:
            key = env_val.encode("utf-8")
        elif allow_dev_key:
            logger.warning("Clave de DESARROLLO activa (allow_dev_key=True). No apta para producción.")
            key = _DEV_SECRET_KEY
        else:
            raise ValueError(
                f"Sin clave HMAC: defina {_HMAC_KEY_ENV} o pase secret_key; "
                "allow_dev_key=True solo para desarrollo."
            )
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"Clave HMAC demasiado corta: {len(key)} < {_MIN_KEY_BYTES} bytes.")
    return key


def _compute_rsi_procedure_digest() -> str:
    """Meta-certificado de procedimiento: ata las constantes que definen la verificación
    RSI³. Si cambian umbrales, mallas o profundidad, la cadena cambia."""
    canon = "|".join((
        f"grid_base={_KLEISLI_GRID_BASE}",
        f"domain={_KLEISLI_DOMAIN!r}",
        f"offset={_KLEISLI_OFFSET!r}",
        f"margin={_KLEISLI_ERROR_BUDGET!r}",
        f"picard_tol={_PICARD_TOL!r}",
        f"picard_max={_PICARD_MAX_ITER}",
        f"ref_tol={_PICARD_REF_TOL!r}",
        f"ref_max={_PICARD_REF_MAX_ITER}",
        f"lip_tol={_LIP_TOL!r}",
        f"cons_tol={_CONSISTENCY_TOL!r}",
        f"max_depth={_RSI_MAX_DEPTH}",
    ))
    return hashlib.sha256(_RSI_CHAIN_DOMAIN + canon.encode("utf-8")).hexdigest()


_RSI_PROCEDURE_DIGEST: Final[str] = _compute_rsi_procedure_digest()


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
        raise ValueError("initial_lip debe estar en [0, 1).")
    return fv


def _validate_int(value: object, name: str, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} debe ser un entero.")
    v = int(value)
    if v < minimum or (maximum is not None and v > maximum):
        raise ValueError(f"{name} fuera de rango.")
    return v


def validate_positron(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError("positron_hmac_valid debe ser bool o None (fail-closed).")


def _validate_orbit(orbit: Sequence[float]) -> Tuple[float, ...]:
    vals = tuple(float(x) for x in orbit)
    if len(vals) != 6 or not all(math.isfinite(x) for x in vals):
        raise ValueError("orbital_elements: 6 valores finitos en orden (a, e, i, M, ω, Ω).")
    return vals


def _validate_amplitude(value: object) -> float:
    a = _validate_finite(value, "pendulum_amplitude")
    if not (0.0 <= a < math.pi):
        raise ValueError("pendulum_amplitude debe estar en [0, π).")
    return a


def _validate_secular_system(
    system: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, float], ...]:
    """(m/M☉, a[AU]) por planeta: masas>0, semiejes>0 distintos, 2..8 planetas."""
    pairs = tuple((float(p[0]), float(p[1])) for p in system)
    if not (2 <= len(pairs) <= _SECULAR_MAX_PLANETS):
        raise ValueError(f"secular_system admite 2..{_SECULAR_MAX_PLANETS} planetas.")
    for m, a in pairs:
        if not (math.isfinite(m) and math.isfinite(a) and m > 0.0 and a > 0.0):
            raise ValueError("secular_system: masas y semiejes finitos y positivos.")
    axes = sorted(a for _, a in pairs)
    if any(b <= a for a, b in zip(axes, axes[1:])):
        raise ValueError("secular_system: semiejes distintos (sin órbitas cruzadas).")
    return tuple(sorted(pairs, key=lambda p: p[1]))


def _as_cost_matrix(m: object) -> NDArray[np.float64]:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("La matriz de coste debe ser 2D con ambas dimensiones ≥ 2.")
    if not np.all(np.isfinite(a)):
        raise ValueError("La matriz de coste contiene valores no finitos.")
    return np.ascontiguousarray(a)


def _as_exponents(e: object) -> NDArray[np.float64]:
    v = np.asarray(e, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("Los exponentes deben ser un vector finito no vacío.")
    if np.any(v < 0.0):
        raise ValueError("Exponentes negativos: fuera del anillo de Novikov.")
    return np.ascontiguousarray(v)


def _quantize_matrix(mat: NDArray[np.float64]) -> List[List[int]]:
    """Cuantización con signo a escala fija. Rechaza desbordamientos; devuelve enteros
    de Python (precisión arbitraria) para los pivotes de Smith y Bareiss."""
    scaled = mat * _SNF_FIXED_POINT_SCALE
    if float(np.max(np.abs(scaled))) >= float(_SNF_MAX_ABS_INT):
        raise ValueError("Entrada fuera del rango de cuantización (overflow evitado).")
    return np.rint(scaled).astype(np.int64).tolist()


def _swap_rc(a: List[List[int]], r1: int, r2: int, c1: int, c2: int) -> None:
    """Intercambia filas r1↔r2 y columnas c1↔c2 (operaciones unimodulares)."""
    a[r1], a[r2] = a[r2], a[r1]
    for row in a:
        row[c1], row[c2] = row[c2], row[c1]


def _min_abs_nonzero(
    a: List[List[int]], cells: Sequence[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    for i, j in cells:
        v = a[i][j]
        if v != 0 and (best is None or abs(v) < abs(a[best[0]][best[1]])):
            best = (i, j)
    return best


def _third_difference(series: Optional[Sequence[float]]) -> Optional[float]:
    """Diferencia tercera MEDIDA con paso unitario. None si no hay serie."""
    if series is None:
        return None
    c = np.asarray(series, dtype=np.float64).reshape(-1)
    if c.size < 4 or not np.all(np.isfinite(c)):
        raise ValueError("capacity_series requiere al menos 4 muestras finitas.")
    return float(c[3] - 3.0 * c[2] + 3.0 * c[1] - c[0])


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — Cimientos: ℤ, booleanas, Novikov, Kepler, estado orbital, Poincaré
# ══════════════════════════════════════════════════════════════════════════════
class _Fase1_Cimientos:
    """Fase 1. Invariantes enteros certificados, booleanas, valuación de Krull, órbita
    de Kepler con certificado de estado independiente y variables de Poincaré."""

    def __init__(self, secret_key: Optional[bytes] = None, allow_dev_key: bool = False) -> None:
        self._secret_key: bytes = _resolve_secret_key(secret_key, allow_dev_key)

    # ── Forma de Smith completa sobre ℤ ────────────────────────────────────────
    @staticmethod
    def _fase1_smith_diagonal(a_in: Sequence[Sequence[int]]) -> Tuple[int, ...]:
        """Forma de Smith: |dᵢ| con CADENA DE DIVISIBILIDAD d₁|d₂|…|d_r.
        coker ≅ ⊕ ℤ/|dᵢ| ⊕ ℤ^{m−r}. Pivote mínimo + barridos unimodulares.
        La barrida gcd/lcm final fija d_i | d_{i+1}: tras procesar i, d_i divide a todos
        los posteriores (invariante por inducción)."""
        a: List[List[int]] = [[int(v) for v in row] for row in a_in]
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
    def _fase1_bareiss_determinant(a_in: Sequence[Sequence[int]]) -> int:
        """Determinante ENTERO exacto por Bareiss (divisiones exactas, sin fracciones)."""
        a: List[List[int]] = [[int(v) for v in row] for row in a_in]
        n = len(a)
        if n == 0:
            return 1
        sign = 1
        prev = 1
        for k in range(n - 1):
            if a[k][k] == 0:
                swap = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
                if swap is None:
                    return 0
                a[k], a[swap] = a[swap], a[k]
                sign = -sign
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
            prev = a[k][k]
        return sign * a[n - 1][n - 1]

    def _fase1_snf_determinant_certificate(
        self, q: Sequence[Sequence[int]], factors: Tuple[int, ...]
    ) -> Optional[bool]:
        """Para cuadrada no singular: |det(Q)| = |coker torsión| = ∏|dᵢ|.
        None si no aplica (rectangular o singular)."""
        if len(q) != len(q[0]):
            return None
        det = self._fase1_bareiss_determinant(q)
        if det == 0:
            return None
        return bool(abs(det) == math.prod(factors))

    # ── Booleanas y Walsh–Hadamard ─────────────────────────────────────────────
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
        total = 0.0
        for k in range(n_vars):
            partner = idx ^ (1 << k)
            total += float(np.mean(bits != bits[partner]))
        return total

    @staticmethod
    def _fase1_fwht(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transformada de Walsh–Hadamard sin normalizar (mariposa)."""
        a = np.array(vec, dtype=np.float64, copy=True)
        n = a.size
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError("WHT requiere longitud potencia de 2.")
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
        """Influencia total normalizada por n ∈ [0, 1]."""
        bits = self._fase1_boolean_bits(q)
        n_vars = bits.size.bit_length() - 1
        return self._fase1_total_influence(bits) / n_vars

    def compute_snf_certificate(self, incidence_matrix: NDArray[np.float64]) -> _SNFCertificate:
        """Forma de Smith (sobre la cuantización) + certificados: torsión, rango, det."""
        mat = _as_cost_matrix(incidence_matrix)
        q = _quantize_matrix(mat)
        factors = self._fase1_smith_diagonal(q)
        return _SNFCertificate(
            rw_norm=self._fase1_rademacher_walsh_norm(q),
            factors=factors,
            torsion_free=all(d == 1 for d in factors),
            full_rank=len(factors) == min(mat.shape),
            torsion_order=math.prod(factors),
            free_rank=mat.shape[0] - len(factors),
            det_ok=self._fase1_snf_determinant_certificate(q, factors),
        )

    # ── Novikov y TOON ─────────────────────────────────────────────────────────
    def evaluate_novikov_krull_and_toon_compression(
        self, exponents: NDArray[np.float64], raw_token_count: int = _TOON_SOURCE_TOKENS
    ) -> Tuple[float, float, float, int]:
        """Valuación de Krull v = min λᵢ, ‖A‖ = e^{−v} y compresión TOON (56 tokens).
        Soporte finito ⇒ buen orden de Novikov satisfecho."""
        exps = _as_exponents(exponents)
        raw = _validate_int(raw_token_count, "raw_token_count", _TOON_BUDGET_TOKENS)
        krull = float(np.min(exps))
        novikov_norm = math.exp(-krull)
        compression_pct = 100.0 * (1.0 - _TOON_BUDGET_TOKENS / raw)
        return krull, novikov_norm, compression_pct, _TOON_BUDGET_TOKENS

    # ── Mecánica celeste: Delaunay y Kepler ────────────────────────────────────
    def _fase1_delaunay_actions(self, a: float, e: float, i: float) -> Tuple[float, float, float]:
        """Acciones de Delaunay L = √(μa), G = L√(1−e²), H = G cos i."""
        L = math.sqrt(_MU_EARTH_KM3_S2 * a)
        G = L * math.sqrt(1.0 - e * e)
        H = G * math.cos(i)
        return L, G, H

    def _fase1_delaunay_is_physical(self, a: float, e: float, i: float) -> bool:
        """Dominio físico + identidades 0 ≤ G ≤ L, |H| ≤ G."""
        if not all(math.isfinite(x) for x in (a, e, i)):
            return False
        if a <= 0.0 or not (0.0 <= e < 1.0) or not (0.0 <= i <= math.pi):
            return False
        L, G, H = self._fase1_delaunay_actions(a, e, i)
        return bool(0.0 < L and 0.0 <= G <= L and abs(H) <= G)

    @staticmethod
    def _fase1_kepler_residual(ecc: float, mean_anomaly: float, e: float) -> float:
        return ecc - e * math.sin(ecc) - (mean_anomaly % (2.0 * math.pi))

    def _fase1_kepler_solve(self, mean_anomaly: float, e: float) -> float:
        """E − e·sinE = M. f(E) = E − e·sinE − M es estrictamente creciente (f' ≥ 1−e > 0)
        con raíz única en [0, 2π). Newton con arranque de Danby, horquilla que se estrecha
        con cada iterado y bisección de respaldo ⇒ convergencia garantizada."""
        M = _validate_finite(mean_anomaly, "mean_anomaly")
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad fuera de [0, 1).")
        m_red = M % (2.0 * math.pi)
        if m_red == 0.0 or e == 0.0:
            return m_red
        lo, hi = 0.0, 2.0 * math.pi
        ecc = m_red + 0.85 * e * (1.0 if math.sin(m_red) >= 0.0 else -1.0)
        ecc = min(max(ecc, lo), hi)
        for _ in range(_KEPLER_MAX_ITER):
            f = ecc - e * math.sin(ecc) - m_red
            if abs(f) <= _KEPLER_TOL:
                return ecc
            if f > 0.0:
                hi = ecc
            else:
                lo = ecc
            nxt = ecc - f / (1.0 - e * math.cos(ecc))
            if not (lo < nxt < hi):
                nxt = 0.5 * (lo + hi)
            if abs(nxt - ecc) <= _KEPLER_TOL:
                return nxt
            ecc = nxt
        raise RuntimeError("Kepler: sin convergencia (no debería ocurrir con salvaguarda).")

    def _fase1_kepler_check(self, mean_anomaly: float, e: float) -> Tuple[bool, float]:
        try:
            ecc = self._fase1_kepler_solve(mean_anomaly, e)
        except (ValueError, RuntimeError):
            return False, math.nan
        residual = self._fase1_kepler_residual(ecc, mean_anomaly, e)
        return bool(abs(residual) <= _KEPLER_RESIDUAL_TOL), ecc

    # ── Estado orbital independiente (certificado de mecánica celeste) ─────────
    @staticmethod
    def _fase1_rotation_313(node: float, inc: float, argp: float) -> NDArray[np.float64]:
        """R = R_z(Ω)·R_x(i)·R_z(ω): perifocal → geocéntrica (secuencia 3-1-3)."""
        cO, sO = math.cos(node), math.sin(node)
        ci, si = math.cos(inc), math.sin(inc)
        cw, sw = math.cos(argp), math.sin(argp)
        rz_o = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
        rx_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
        rz_w = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
        return rz_o @ rx_i @ rz_w

    def _fase1_orbital_state_certificate(
        self, a: float, e: float, i: float, mean_anom: float, argp: float, node: float
    ) -> Tuple[Optional[float], Optional[bool]]:
        """Construye r, v desde E (Kepler), los rota a ECI y verifica tres identidades
        independientes de las acciones de Delaunay:
          (1) vis-viva:  v² = μ(2/r − 1/a)
          (2) |r×v| = G = √(μa(1−e²))
          (3) (r×v)_z = H = G cos i
        Devuelve (residuo máximo relativo, ok). (None, None) si no es físico."""
        if not self._fase1_delaunay_is_physical(a, e, i):
            return None, None
        ok_k, ecc = self._fase1_kepler_check(mean_anom, e)
        if not ok_k:
            return None, None
        mu = _MU_EARTH_KM3_S2
        cos_e, sin_e = math.cos(ecc), math.sin(ecc)
        one_minus = 1.0 - e * cos_e
        b = a * math.sqrt(1.0 - e * e)
        r_p = np.array([a * (cos_e - e), b * sin_e, 0.0])
        e_dot = math.sqrt(mu / a ** 3) / one_minus
        v_p = np.array([-a * sin_e * e_dot, b * cos_e * e_dot, 0.0])
        rot = self._fase1_rotation_313(node, i, argp)
        r_vec = rot @ r_p
        v_vec = rot @ v_p
        r_norm = float(np.linalg.norm(r_vec))
        energy_res = abs(float(v_vec @ v_vec) - mu * (2.0 / r_norm - 1.0 / a)) / (mu / a)
        h_vec = np.cross(r_vec, v_vec)
        _, G, H = self._fase1_delaunay_actions(a, e, i)
        ang_res = abs(float(np.linalg.norm(h_vec)) - G) / G
        hz_res = abs(float(h_vec[2]) - H) / G
        residual = max(energy_res, ang_res, hz_res)
        return float(residual), bool(residual <= _ORBIT_RESIDUAL_TOL)

    # ── Variables canónicas de Poincaré (regulares en e→0, i→0) ────────────────
    @staticmethod
    def _fase1_poincare_map(u: NDArray) -> NDArray:
        """(L,G,H,l,g,h) → (Λ,ξ,p,λ,η,q) con pares canónicos (Λ,λ),(ξ,η),(p,q).
        Acepta dtype compleja (paso complejo). λ sin reducir: la reducción mod 2π
        es un recubrimiento simpléctico y no altera J."""
        L, G, H, l, g, h = u
        rho_e = np.sqrt(2.0 * (L - G))
        rho_i = np.sqrt(2.0 * (G - H))
        varpi = g + h
        return np.array([
            L,
            rho_e * np.cos(varpi),
            rho_i * np.cos(h),
            l + g + h,
            -rho_e * np.sin(varpi),
            -rho_i * np.sin(h),
        ])

    def _fase1_poincare_symplectic_residual(self, u: NDArray[np.float64]) -> float:
        """Certificado JΩJᵀ = Ω por paso complejo. Derivación analítica:
        Σ = dL∧dl+dG∧dg+dH∧dh = dΛ∧dλ + dξ∧dη + dp∧dq."""
        n = 6
        omega = np.zeros((n, n))
        for k in range(3):
            omega[k, k + 3] = 1.0
            omega[k + 3, k] = -1.0
        jac = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            du = np.zeros(n, dtype=np.complex128)
            du[j] = 1j * _COMPLEX_STEP
            mapped = self._fase1_poincare_map(u.astype(np.complex128) + du)
            jac[:, j] = np.imag(mapped) / _COMPLEX_STEP
        return float(np.max(np.abs(jac @ omega @ jac.T - omega)))

    def _fase1_poincare_certificate(
        self, a: float, e: float, i: float, mean_anom: float, argp: float, node: float
    ) -> Tuple[Tuple[float, float, float, float, float, float], Optional[float], Optional[bool]]:
        """Variables de Poincaré (Λ, λ, ξ, η, p, q) + certificado simpléctico.
        En e=0 o i=0 las acciones ρ² se anulan: certificado None (punto regular declarado)."""
        nan6 = (math.nan,) * 6
        if not self._fase1_delaunay_is_physical(a, e, i):
            return nan6, None, None
        L, G, H = self._fase1_delaunay_actions(a, e, i)
        l = mean_anom % (2.0 * math.pi)
        u = np.array([L, G, H, l, argp, node], dtype=np.float64)
        v = self._fase1_poincare_map(u)
        variables = (
            float(v[0]), float(v[3]) % (2.0 * math.pi),
            float(v[1]), float(v[4]), float(v[2]), float(v[5]),
        )
        tiny = 1.0e-12 * L
        if (L - G) <= tiny or (G - H) <= tiny:
            return variables, None, None
        residual = self._fase1_poincare_symplectic_residual(u)
        return variables, residual, bool(residual <= _SYMPLECTIC_TOL)

    # ── ÚLTIMO método de Fase 1 → inicio de Fase 2 ─────────────────────────────
    def _fase1_emit_spectral_seed(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        raw_token_count: int,
        positron_hmac_valid: Optional[bool],
        orbital_elements: Optional[Sequence[float]],
        params_canonical: str,
    ) -> _SpectralSeed:
        """ÚLTIMO método de F1. Emite la semilla que consume `_fase2_inicio_desde_semilla`."""
        mat = _as_cost_matrix(apu_cost_matrix)
        exps = _as_exponents(exponents)
        positron = validate_positron(positron_hmac_valid)
        orbit = _validate_orbit(_DEFAULT_ORBIT if orbital_elements is None else orbital_elements)
        a, e, inc, mean_anom, argp, node = orbit
        snf = self.compute_snf_certificate(mat)
        krull, novikov, pct, budget = self.evaluate_novikov_krull_and_toon_compression(
            exps, raw_token_count
        )
        physical = self._fase1_delaunay_is_physical(a, e, inc)
        kepler_ok, ecc_anom = self._fase1_kepler_check(mean_anom, e)
        lgh = self._fase1_delaunay_actions(a, e, inc) if physical else (math.nan, math.nan, math.nan)
        orb_res, orb_ok = self._fase1_orbital_state_certificate(a, e, inc, mean_anom, argp, node)
        pv, pres, pok = self._fase1_poincare_certificate(a, e, inc, mean_anom, argp, node)
        return _SpectralSeed(
            matrix=mat,
            exponents=exps,
            snf_diagonal_factors=snf.factors,
            torsion_free=snf.torsion_free,
            torsion_order=snf.torsion_order,
            full_rank=snf.full_rank,
            cokernel_free_rank=snf.free_rank,
            snf_det_ok=snf.det_ok,
            rw_norm=snf.rw_norm,
            krull=krull,
            novikov_norm=novikov,
            raw_token_count=int(raw_token_count),
            compressed_token_count=budget,
            kv_cache_pct=pct,
            positron_valid=positron,
            anomaly_norm=_ANOMALY_SCALE * float(np.linalg.norm(mat)),
            delaunay_LGH=lgh,
            delaunay_valid=bool(physical and kepler_ok and orb_ok is True),
            eccentric_anomaly=ecc_anom,
            orbital_residual=orb_res,
            orbital_ok=orb_ok,
            poincare_variables=pv,
            poincare_canonical_residual=pres,
            poincare_canonical_ok=pok,
            input_digest=_input_digest(mat, exps, params_canonical),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — Geometría y dinámica de Poincaré
# ══════════════════════════════════════════════════════════════════════════════
class _Fase2_Geometria(_Fase1_Cimientos):
    """Fase 2. Rigidez de Gromov, armonicidad discreta, caos de Melnikov–Poincaré,
    Lindstedt, teoría secular de Poincaré, entropía de Spohn y aniquilación de Dirac."""

    # ── PRIMER método de F2: consume la semilla emitida por el ÚLTIMO de F1 ────
    def _fase2_inicio_desde_semilla(
        self, seed: _SpectralSeed, ball_r: float, cylinder_R: float
    ) -> Tuple[float, float, bool]:
        """Primer método de F2. Consume la _SpectralSeed → (c_B, c_Z, violación)."""
        if not isinstance(seed, _SpectralSeed):
            raise TypeError("Fase 2 exige una _SpectralSeed emitida por Fase 1.")
        return self._fase2_capacities(ball_r, cylinder_R)

    @staticmethod
    def _fase2_capacities(ball_r: float, cylinder_R: float) -> Tuple[float, float, bool]:
        """Capacidades πr², πR². Violación ⇔ r > R. En ℝ² es monotonía de área; el
        no-squeezing de Gromov es profundo para n ≥ 2: aquí es política de diseño."""
        r = _validate_positive(ball_r, "ball_r")
        R = _validate_positive(cylinder_R, "cylinder_R")
        return math.pi * r * r, math.pi * R * R, bool(r > R)

    @staticmethod
    def _fase2_harmonicity_residual(mat: NDArray[np.float64]) -> float:
        """Residuo de armonicidad discreta ‖Δφ‖_F/(‖φ‖_F+ε) con Laplaciano de 5 puntos
        y frontera PERIÓDICA. Mide qué tan armónica es la función-matriz φ.
        (No es una condición de Cauchy–Riemann; el nombre de campo se conserva.)"""
        phi = _as_cost_matrix(mat)
        lap = (
            np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0)
            + np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1)
            - 4.0 * phi
        )
        return float(np.linalg.norm(lap) / (np.linalg.norm(phi) + _WILKINSON_EPSILON))

    def solve_pseudoholomorphic_surface_weaver(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = _DEFAULT_BALL_R,
        cylinder_R: float = _DEFAULT_CYLINDER_R,
    ) -> Tuple[float, bool]:
        """Devuelve (residuo de armonicidad, violación de Gromov)."""
        mat = _as_cost_matrix(apu_cost_matrix)
        _, _, violated = self._fase2_capacities(ball_r, cylinder_R)
        return self._fase2_harmonicity_residual(mat), violated

    # ── Melnikov–Poincaré en forma cerrada ─────────────────────────────────────
    @staticmethod
    def _fase2_melnikov_coefficients(delta: float, gamma: float, omega: float) -> Tuple[float, float]:
        """M(t₀) = offset + amplitud·cos(ωt₀) para ẍ+δẋ+sin x = γcos(ωt).
        Separatriz ẋ_h = 2 sech t: ∫ẋ_h² = 8; ∫ẋ_h cos(ωt) dt = 2π·sech(πω/2).
        offset = −8δ; amplitud = 2π|γ|·sech(πω/2)."""
        d = _validate_finite(delta, "melnikov_delta")
        g = _validate_finite(gamma, "melnikov_gamma")
        w = _validate_finite(omega, "melnikov_omega")
        amplitude = 2.0 * math.pi * abs(g) / math.cosh(math.pi * w / 2.0)
        offset = -8.0 * d
        return amplitude, offset

    def _fase2_melnikov_quadrature(self, delta: float, gamma: float, omega: float) -> Tuple[float, float]:
        """Cuadratura de control (solo autoprueba metamórfica contra la forma cerrada)."""
        t = np.linspace(-_MELNIKOV_HALF_WIDTH, _MELNIKOV_HALF_WIDTH, _MELNIKOV_POINTS)
        v = 2.0 / np.cosh(t)
        i_energy = _trapz(v * v, t)
        i_cos = _trapz(v * np.cos(omega * t), t)
        return abs(gamma) * abs(i_cos), -delta * i_energy

    def evaluate_poincare_kam_melnikov(
        self,
        delta: float = _MELNIKOV_DEFAULT_DELTA,
        gamma: float = _MELNIKOV_DEFAULT_GAMMA,
        omega: float = _MELNIKOV_DEFAULT_OMEGA,
    ) -> Tuple[bool, float, float, float, bool]:
        """(caos, amplitud, offset, margen, régimen ε válido). Cero simple de M ⇔
        ω ≠ 0 ∧ amplitud > |offset|. Validez rigurosa solo para ε pequeño."""
        d = _validate_finite(delta, "melnikov_delta")
        g = _validate_finite(gamma, "melnikov_gamma")
        w = _validate_finite(omega, "melnikov_omega")
        amplitude, offset = self._fase2_melnikov_coefficients(d, g, w)
        chaos = bool(w != 0.0 and amplitude > abs(offset))
        margin = float(amplitude - abs(offset))
        eps_valid = bool(max(abs(d), abs(g)) <= _MELNIKOV_EPS_POLICY)
        return chaos, amplitude, offset, margin, eps_valid

    # ── Lindstedt–Poincaré: frecuencia exacta del péndulo por AGM ──────────────
    @staticmethod
    def _fase2_agm(a: float, b: float) -> float:
        """Media aritmético-geométrica (convergencia cuadrática)."""
        for _ in range(60):
            if abs(a - b) <= 1.0e-15 * max(abs(a), abs(b)):
                return 0.5 * (a + b)
            a, b = 0.5 * (a + b), math.sqrt(a * b)
        return 0.5 * (a + b)

    def _fase2_poincare_lindstedt(self, amplitude: float) -> Tuple[float, float, float]:
        """Ω(a) = AGM(1, cos(a/2)) exacta (K(k)=π/(2·AGM(1,k')), k'=cos(a/2), Ω=2π/T=π/(2K)).
        Serie Lindstedt–Poincaré: Ω ≈ 1 − a²/16 + a⁴/3072 (error O(a⁶))."""
        a = _validate_amplitude(amplitude)
        exact = self._fase2_agm(1.0, math.cos(a / 2.0))
        series = 1.0 - a * a / 16.0 + a ** 4 / 3072.0
        return exact, series, abs(exact - series)

    # ── Mapa estroboscópico de Poincaré: Lyapunov de Benettin (evidencia) ──────
    def _fase2_poincare_stroboscopic_lyapunov(
        self,
        delta: float,
        gamma: float,
        omega: float,
        periods: int,
        steps_per_period: int,
    ) -> Optional[float]:
        """λ_max finito del mapa estroboscópico (T=2π/ω) de ẍ+δẋ+sin x = γcos(ωt),
        con ecuación variacional y renormalización de Benettin por periodo. RK4
        determinista; transitorio = periods//8. EVIDENCIA, no prueba.
        None si ω ≤ 0 o periods = 0."""
        d = _validate_finite(delta, "melnikov_delta")
        g = _validate_finite(gamma, "melnikov_gamma")
        w = _validate_finite(omega, "melnikov_omega")
        n_per = _validate_int(periods, "poincare_periods", 0, 20000)
        n_st = _validate_int(steps_per_period, "poincare_steps_per_period", 8, 4096)
        if w <= 0.0 or n_per == 0:
            return None
        period = 2.0 * math.pi / w
        h = period / n_st
        h2 = 0.5 * h
        sin, cos, hypot, log = math.sin, math.cos, math.hypot, math.log
        x, v = _POINCARE_IC
        px, pv = 1.0, 0.0
        transient = n_per // 8
        acc = 0.0
        counted = 0
        t = 0.0

        def deriv(tt: float, xx: float, vv: float, ppx: float, ppv: float):
            ax = -sin(xx) - d * vv + g * cos(w * tt)
            return vv, ax, ppv, -cos(xx) * ppx - d * ppv

        for k in range(n_per):
            for _ in range(n_st):
                k1x, k1v, k1px, k1pv = deriv(t, x, v, px, pv)
                k2x, k2v, k2px, k2pv = deriv(
                    t + h2, x + h2 * k1x, v + h2 * k1v, px + h2 * k1px, pv + h2 * k1pv)
                k3x, k3v, k3px, k3pv = deriv(
                    t + h2, x + h2 * k2x, v + h2 * k2v, px + h2 * k2px, pv + h2 * k2pv)
                k4x, k4v, k4px, k4pv = deriv(
                    t + h, x + h * k3x, v + h * k3v, px + h * k3px, pv + h * k3pv)
                x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                px += (h / 6.0) * (k1px + 2.0 * k2px + 2.0 * k3px + k4px)
                pv += (h / 6.0) * (k1pv + 2.0 * k2pv + 2.0 * k3pv + k4pv)
                t += h
            s = hypot(px, pv)
            if s < 1.0e-300:
                px, pv, s = 1.0, 0.0, 1.0
            if k >= transient:
                acc += log(s)
                counted += 1
            px /= s
            pv /= s
        if counted == 0:
            return None
        return float(acc / (counted * period))

    # ── Espectro secular de Poincaré–Laplace–Lagrange ──────────────────────────
    @staticmethod
    def _fase2_laplace_coefficients(alpha: float) -> Tuple[float, float]:
        """b_{3/2}^{(1)}(α), b_{3/2}^{(2)}(α) por cuadratura periódica (convergencia
        espectral para α<1): b_s^{(j)} = (1/π)∫₀^{2π} cos(jθ)/(1−2α cosθ+α²)^s dθ."""
        theta = np.linspace(0.0, 2.0 * math.pi, _SECULAR_QUAD_POINTS, endpoint=False)
        denom = (1.0 - 2.0 * alpha * np.cos(theta) + alpha * alpha) ** 1.5
        b1 = float(2.0 * np.mean(np.cos(theta) / denom))
        b2 = float(2.0 * np.mean(np.cos(2.0 * theta) / denom))
        return b1, b2

    def _fase2_poincare_secular_spectrum(
        self, system: Sequence[Sequence[float]]
    ) -> Tuple[Tuple[float, ...], float, bool]:
        """Teoría secular lineal (M☉=1, μ=4π² AU³/año²). Aproximación de primer orden
        mⱼ≪M☉ aplicada COHERENTEMENTE en diagonal y fuera de diagonal:
          A_jj = n_j Σ_{k≠j} m_k · ¼ α ᾱ b¹,   A_jk = −n_j m_k · ¼ α ᾱ b²,
        con ᾱ = α si el perturbador es exterior y 1 si es interior.
        Simetría: w_j = m_j n_j a_j² (momento angular) ⇒ W·A simétrica ⇒ S = W^{½}AW^{−½}
        simétrica ⇒ espectro real por eigvalsh. Devuelve (frecuencias, residuo relativo, válido)."""
        pairs = _validate_secular_system(system)
        n_pl = len(pairs)
        masses = np.array([p[0] for p in pairs])
        axes = np.array([p[1] for p in pairs])
        n_mean = 2.0 * math.pi / axes ** 1.5
        A = np.zeros((n_pl, n_pl))
        coeffs: dict = {}
        for j in range(n_pl):
            for k in range(n_pl):
                if j == k:
                    continue
                key = (min(j, k), max(j, k))
                if key not in coeffs:
                    alpha = axes[key[0]] / axes[key[1]]
                    coeffs[key] = (alpha, *self._fase2_laplace_coefficients(alpha))
                alpha, b1, b2 = coeffs[key]
                alpha_bar = alpha if axes[k] > axes[j] else 1.0
                common = 0.25 * alpha * alpha_bar
                A[j, j] += n_mean[j] * masses[k] * common * b1
                A[j, k] = -n_mean[j] * masses[k] * common * b2
        w = masses * n_mean * axes ** 2
        WA = w[:, None] * A
        scale = float(np.max(np.abs(WA)))
        residual = float(np.max(np.abs(WA - WA.T)) / max(scale, 1.0e-300))
        sw = np.sqrt(w)
        S = (sw[:, None] * A) / sw[None, :]
        S = 0.5 * (S + S.T)   # proyección simétrica exacta (residuo ya certificado)
        freqs = tuple(float(g) for g in np.linalg.eigvalsh(S))
        valid = bool(residual <= _SECULAR_SYM_TOL and all(math.isfinite(g) for g in freqs))
        return freqs, residual, valid

    # ── Entropía de Spohn y aniquilación de Dirac ──────────────────────────────
    @staticmethod
    def _fase2_shannon(p: NDArray[np.float64]) -> float:
        q = p[p > 0.0]
        return float(-np.sum(q * np.log(q)))

    def _fase2_spohn_production(
        self, exponents: NDArray[np.float64], lam: float = _SPOHN_LAMBDA
    ) -> float:
        """σ = H(Φ(p)) − H(p) ≥ 0 con Φ = (1−λ)I + λU doblemente estocástica.
        Si Σλᵢ = 0, p uniforme (política declarada)."""
        exps = _as_exponents(exponents)
        lam_f = _validate_finite(lam, "spohn_lambda")
        if not (0.0 <= lam_f <= 1.0):
            raise ValueError("spohn_lambda debe estar en [0, 1].")
        total = float(np.sum(exps))
        p = exps / total if total > 0.0 else np.full(exps.size, 1.0 / exps.size)
        u = np.full(p.size, 1.0 / p.size)
        p_after = (1.0 - lam_f) * p + lam_f * u
        return self._fase2_shannon(p_after) - self._fase2_shannon(p)

    def execute_dirac_annihilation_toon(
        self, anomaly_norm: float, positron_hmac_valid: Optional[bool]
    ) -> float:
        """e⁻ + e⁺ → 2γ: energía total 2·m_e·c² = 1021.9979 keV (511 keV por fotón en CM),
        o 0 sin validación positiva (fail-closed)."""
        positron = validate_positron(positron_hmac_valid)
        if positron is not True:
            return 0.0
        if not (math.isfinite(anomaly_norm) and anomaly_norm > _ANOMALY_THRESHOLD):
            return 0.0
        return _POSITRON_REST_MASS_KEV

    # ── ÚLTIMO método de Fase 2 → inicio de Fase 3 ─────────────────────────────
    def _fase2_emit_dynamical_seed(
        self,
        seed: _SpectralSeed,
        ball_r: float,
        cylinder_R: float,
        melnikov_delta: float,
        melnikov_gamma: float,
        melnikov_omega: float,
        pendulum_amplitude: float,
        poincare_periods: int,
        poincare_steps_per_period: int,
        secular_system: Sequence[Sequence[float]],
    ) -> _DynamicalSeed:
        """ÚLTIMO método de F2. Emite la semilla que consume
        `_fase3_inicio_desde_semilla_dinamica`. El veto duro usa solo criterios
        certificados; la evidencia numérica únicamente degrada."""
        c_ball, c_cyl, violated = self._fase2_inicio_desde_semilla(seed, ball_r, cylinder_R)
        harm = self._fase2_harmonicity_residual(seed.matrix)
        chaos, amp, off, margin, eps_valid = self.evaluate_poincare_kam_melnikov(
            melnikov_delta, melnikov_gamma, melnikov_omega
        )
        lyap = self._fase2_poincare_stroboscopic_lyapunov(
            melnikov_delta, melnikov_gamma, melnikov_omega,
            poincare_periods, poincare_steps_per_period,
        )
        # Tensión UNIDIRECCIONAL: λ>tol con Melnikov sin caos ⇒ tensión. Melnikov con caos
        # y λ≤tol NO es tensión: el conjunto caótico homoclínico puede no ser atractor.
        consistent: Optional[bool]
        if lyap is None:
            consistent = None
        else:
            consistent = bool((not (lyap > _LYAP_POS_TOL)) or chaos)
        exact, series, lerr = self._fase2_poincare_lindstedt(pendulum_amplitude)
        freqs, sym_res, sec_valid = self._fase2_poincare_secular_spectrum(secular_system)
        spohn = self._fase2_spohn_production(seed.exponents)
        dirac = self.execute_dirac_annihilation_toon(seed.anomaly_norm, seed.positron_valid)
        force_veto = bool(
            violated
            or (not seed.torsion_free)
            or chaos
            or (seed.krull < _DEFAULT_KRULL_FLOOR)
            or (not seed.delaunay_valid)
            or (seed.poincare_canonical_ok is False)
            or (seed.snf_det_ok is False)
        )
        return _DynamicalSeed(
            spectral=seed,
            capacity_ball=c_ball,
            capacity_cylinder=c_cyl,
            gromov_violated=violated,
            harmonicity_residual=harm,
            melnikov_chaos=chaos,
            melnikov_amplitude=amp,
            melnikov_offset=off,
            melnikov_margin=margin,
            melnikov_eps_valid=eps_valid,
            poincare_lyapunov=lyap,
            poincare_consistent=consistent,
            lindstedt_exact=exact,
            lindstedt_series=series,
            lindstedt_error=lerr,
            secular_frequencies=freqs,
            secular_sym_residual=sym_res,
            secular_valid=sec_valid,
            spohn_production=spohn,
            dirac_kev=dirac,
            force_veto=force_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — Kleisli T³, RSI³ reflexivo con certificados, Heyting y pasaporte
# ══════════════════════════════════════════════════════════════════════════════
class _Fase3_Kleisli_RSI3(_Fase2_Geometria):
    """Fase 3. Torre reflexiva RSI³ (objeto → meta → meta-meta), contracción de Banach
    de T³ con cota analítica λ³ (demostrada) y margen λ³+ε (conservador), adjudicación
    de Heyting y pasaporte HMAC canónico. RSI³ NO modifica su propio código: refina y
    re-certifica sus certificados con profundidad acotada (≤3)."""

    # ── PRIMER método de F3: consume la semilla emitida por el ÚLTIMO de F2 ────
    def _fase3_inicio_desde_semilla_dinamica(
        self, dyn: _DynamicalSeed, rsi_max_depth: int, initial_lip: float
    ) -> Tuple[_DynamicalSeed, int, float]:
        """Primer método de F3. Consume la _DynamicalSeed → (semilla, profundidad
        efectiva, λ). Veto duro ⇒ profundidad efectiva 0."""
        if not isinstance(dyn, _DynamicalSeed):
            raise TypeError("Fase 3 exige una _DynamicalSeed emitida por Fase 2.")
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        lam = _validate_lambda(initial_lip)
        effective = 0 if dyn.force_veto else depth
        return dyn, effective, lam

    # ── Cotas de Banach ────────────────────────────────────────────────────────
    @staticmethod
    def _fase3_analytic_bound(lam: float) -> float:
        """Cota DEMOSTRADA: |T'| ≤ λ en todo ℝ ⇒ Lip(T³) ≤ λ³."""
        return float(lam) ** 3

    @staticmethod
    def _fase3_safety_bound(lam: float) -> float:
        """Margen conservador λ³+ε (ε=0.002375 declarado, no derivado). Se usa en el
        error a posteriori para tolerancia adicional."""
        return float(lam) ** 3 + _KLEISLI_ERROR_BUDGET

    @staticmethod
    def _fase3_kleisli_cubed(x, lam: float):
        """T³ con T(x) = λ·cos x + 0.1 (supuesto declarado del autor del motor)."""
        y = x
        for _ in range(3):
            y = lam * np.cos(y) + _KLEISLI_OFFSET
        return y

    @staticmethod
    def _fase3_kleisli_cubed_derivative(x, lam: float):
        """|(T³)'(x)| = λ³·|sin x · sin T(x) · sin T²(x)| (regla de la cadena exacta)."""
        y1 = lam * np.cos(x) + _KLEISLI_OFFSET
        y2 = lam * np.cos(y1) + _KLEISLI_OFFSET
        return np.abs(lam ** 3 * np.sin(x) * np.sin(y1) * np.sin(y2))

    def _fase3_picard(
        self, lam: float, q: float, tol: float = _PICARD_TOL, max_iter: int = _PICARD_MAX_ITER
    ) -> Tuple[float, float, bool, float]:
        """Picard desde x₀=0 sobre T³. Devuelve (x*, error a posteriori q/(1−q)·|Δ|,
        convergencia, residuo |T³x*−x*|)."""
        x_prev = 0.0
        x = 0.0
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
        """sup |(T³)'| sobre malla diádica ANIDADA de nivel `level` (100·2^level+1
        puntos en [−2,2]). Anidamiento exacto ⇒ monotonía e_{k+1} ≥ e_k."""
        n = _KLEISLI_GRID_BASE * (2 ** level) + 1
        grid = np.linspace(-_KLEISLI_DOMAIN, _KLEISLI_DOMAIN, n)
        return float(np.max(self._fase3_kleisli_cubed_derivative(grid, lam)))

    def _fase3_rsi_recurse(
        self, lam: float, q: float, max_depth: int, gate_ok: bool, meta_ok: bool
    ) -> Tuple[int, Tuple[float, ...], str]:
        """Torre RSI³: N1 objeto (cota empírica ≤ q, monotonía), N2 meta (meta_ok: cota a
        posteriori verificada contra Picard de referencia), N3 meta-meta (gate_ok incluye
        la autoprueba metamórfica). La cadena arranca con el meta-certificado de
        procedimiento y encadena cada nivel aceptado (tamper-evident)."""
        accepted = 0
        history: List[float] = []
        prev = -math.inf
        seed_text = f"{_RSI_PROCEDURE_DIGEST}|lam={_fmt(lam)}|q={_fmt(q)}"
        chain = hashlib.sha256(_RSI_CHAIN_DOMAIN + seed_text.encode("utf-8")).digest()
        for level in range(1, max_depth + 1):
            e = self._fase3_empirical_lipschitz(level, lam)
            history.append(e)
            ok = gate_ok and meta_ok and (e <= q + _LIP_TOL) and (e >= prev - _LIP_TOL)
            if not ok:
                break
            accepted = level
            prev = e
            chain = hashlib.sha256(
                chain + f"|{level}|{_fmt(e)}|{int(meta_ok)}".encode("utf-8")
            ).digest()
        return accepted, tuple(history), chain.hex()

    def evaluate_rsi3_banach_kleisli(
        self, initial_lip: float = _KLEISLI_LIP_T, rsi_max_depth: int = _RSI_MAX_DEPTH
    ) -> _KleisliState:
        """Evalúa T³ y la torre RSI³ con verificación meta del error a posteriori."""
        lam = _validate_lambda(initial_lip)
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        selftest = self._fase3_metamorphic_selftest()
        analytic = self._fase3_analytic_bound(lam)
        safety = self._fase3_safety_bound(lam)
        if not analytic < 1.0:
            raise ValueError("La cota analítica no es contractiva (λ³ ≥ 1).")
        x_star, a_post, converged, residual = self._fase3_picard(lam, safety)
        x_ref, _, ref_conv, _ = self._fase3_picard(
            lam, safety, _PICARD_REF_TOL, _PICARD_REF_MAX_ITER
        )
        meta_ok = bool(ref_conv and abs(x_star - x_ref) <= a_post + _CONSISTENCY_TOL)
        consistency = bool(converged and residual <= _CONSISTENCY_TOL)
        accepted, history, chain = self._fase3_rsi_recurse(
            lam, analytic, depth, selftest and consistency, meta_ok
        )
        return _KleisliState(
            lip_T=lam,
            lip_analytic=analytic,
            lip_safety=safety,
            lip_empirical=history,
            fixed_point=x_star,
            a_posteriori_error=a_post,
            residual=residual,
            meta_verified=meta_ok,
            selftest_passed=selftest,
            consistency_ok=consistency,
            depth_accepted=accepted,
            certificate_chain=chain,
        )

    # ── Autoprueba metamórfica (meta-meta de RSI³) ─────────────────────────────
    def _fase3_metamorphic_selftest(self) -> bool:
        """Bloques metamórficos independientes. Cualquier excepción ⇒ False (fail-closed)."""
        try:
            # 1. Smith con cadena de divisibilidad y determinante de Bareiss
            if self._fase1_smith_diagonal([[2, 0], [0, 3]]) != (1, 6):
                return False
            if self._fase1_smith_diagonal([[1, 1], [0, 1]]) != (1, 1):
                return False
            f248 = self._fase1_smith_diagonal([[2, 4], [6, 8]])
            if math.prod(f248) != 8 or any(b % a for a, b in zip(f248, f248[1:])):
                return False
            if self._fase1_smith_diagonal([[6, 0], [0, 10]]) != (2, 30):
                return False
            if self._fase1_bareiss_determinant([[2, 4], [6, 8]]) != -8:
                return False
            if self._fase1_bareiss_determinant([[1, 2, 3], [0, 1, 4], [5, 6, 0]]) != 1:
                return False
            if self._fase1_snf_determinant_certificate([[2, 4], [6, 8]], (2, 4)) is not True:
                return False
            # 2. Parseval e identidad espectral de influencias (semilla 20250101, N=16)
            rng = np.random.default_rng(20250101)
            bits = rng.integers(0, 2, size=16).astype(np.int64)
            g = 1.0 - 2.0 * bits
            ghat = self._fase1_fwht(g) / float(g.size)
            parseval = float(np.sum(ghat ** 2))
            spectral_inf = float(sum(bin(s).count("1") * ghat[s] ** 2 for s in range(ghat.size)))
            total_inf = self._fase1_total_influence(bits)
            if abs(parseval - 1.0) > 1e-12 or abs(spectral_inf - total_inf) > 1e-12:
                return False
            # 3. Kepler safeguardado (caso regular y caso rígido e→1)
            ecc = self._fase1_kepler_solve(1.0, 0.3)
            if abs(self._fase1_kepler_residual(ecc, 1.0, 0.3)) > 1e-10:
                return False
            ecc_hard = self._fase1_kepler_solve(1.0e-6, 0.999)
            if abs(self._fase1_kepler_residual(ecc_hard, 1.0e-6, 0.999)) > 1e-9:
                return False
            # 4. Melnikov: forma cerrada frente a cuadratura de control
            amp, off = self._fase2_melnikov_coefficients(0.01, 0.02, 1.0)
            amp_q, off_q = self._fase2_melnikov_quadrature(0.01, 0.02, 1.0)
            if abs(amp - amp_q) > 1e-6 * amp or abs(off - off_q) > 1e-6:
                return False
            if abs(off + 0.08) > 1e-12:
                return False
            # 5. Spohn: no negatividad y anulación en el uniforme
            if self._fase2_spohn_production(np.array([0.15, 0.22, 0.35, 0.48])) < _SPOHN_TOL:
                return False
            if abs(self._fase2_spohn_production(np.array([1.0, 1.0, 1.0]))) > 1e-12:
                return False
            # 6. Delaunay → Poincaré y estado orbital independiente
            a, e, i, M, argp, node = _DEFAULT_ORBIT
            if not self._fase1_delaunay_is_physical(a, e, i):
                return False
            L, G, H = self._fase1_delaunay_actions(a, e, i)
            pv, pres, pok = self._fase1_poincare_certificate(a, e, i, M, argp, node)
            if pok is not True or pres is None:
                return False
            if abs(pv[0] - 0.5 * (pv[2] ** 2 + pv[3] ** 2) - G) > 1e-8 * L:
                return False
            if abs(G - 0.5 * (pv[4] ** 2 + pv[5] ** 2) - H) > 1e-8 * L:
                return False
            _, orb_ok = self._fase1_orbital_state_certificate(a, e, i, M, argp, node)
            if orb_ok is not True:
                return False
            # 7. Lindstedt–Poincaré: AGM exacta vs serie (error O(a⁶))
            om0, _, _ = self._fase2_poincare_lindstedt(0.0)
            if om0 != 1.0:
                return False
            _, _, err02 = self._fase2_poincare_lindstedt(0.2)
            if err02 > 2e-6:
                return False
            # 8. Cotas de Banach: analítica (demostrada) ≤ margen conservador
            if abs(self._fase3_analytic_bound(0.95) - 0.857375) > 1e-12:
                return False
            if abs(self._fase3_safety_bound(0.95) - 0.85975) > 1e-12:
                return False
            # 9. Heyting: residuación a∧b ≤ c ⇔ a ≤ (b→c) exhaustiva + negaciones
            vals = (HeytingTruthValue.VETOED, HeytingTruthValue.DEGRADED, HeytingTruthValue.COHERENT)
            for aa in vals:
                for bb in vals:
                    for cc in vals:
                        lhs = aa.meet(bb).rank <= cc.rank
                        rhs = aa.rank <= bb.implies(cc).rank
                        if lhs != rhs:
                            return False
            if HeytingTruthValue.COHERENT.negate() is not HeytingTruthValue.VETOED:
                return False
            if HeytingTruthValue.DEGRADED.negate() is not HeytingTruthValue.VETOED:
                return False
            if HeytingTruthValue.DEGRADED.negate().negate() is not HeytingTruthValue.COHERENT:
                return False
            # 10. Laplaciano periódico: autofunción discreta exacta de suma de modos 1D
            ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
            phi = np.cos(2.0 * np.pi * ii / 8.0) + np.cos(2.0 * np.pi * jj / 8.0)
            lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
                   + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4.0 * phi)
            eig = 2.0 * (math.cos(2.0 * math.pi / 8.0) - 1.0)
            if np.linalg.norm(lap - eig * phi) > 1e-12 * np.linalg.norm(phi):
                return False
            # 11. Secular de Poincaré: simetrización por momento angular certificada
            _, sym_res, sec_ok = self._fase2_poincare_secular_spectrum(_SECULAR_DEFAULT)
            if not sec_ok or sym_res > _SECULAR_SYM_TOL:
                return False
            # 12. RSI³ N1: monotonía por anidamiento y dominación por la cota analítica
            e1 = self._fase3_empirical_lipschitz(1, 0.95)
            e2 = self._fase3_empirical_lipschitz(2, 0.95)
            if not (e1 <= e2 + _LIP_TOL and e2 <= self._fase3_analytic_bound(0.95) + _LIP_TOL):
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica: excepción (fail-closed).")
            return False

    # ── Adjudicación de Heyting (meet de los cuatro jueces) ────────────────────
    def _fase3_heyting_adjudicate(
        self, dyn: _DynamicalSeed, kleisli: _KleisliState, requested_depth: int
    ) -> HeytingTruthValue:
        """Veto: criterios certificados. Degradado: evidencia, tensión o modelo. La evidencia
        numérica (Lyapunov) jamás veta ni asciende un veto. Profundidad: se compara contra
        la SOLICITADA; un veto que la fuerza a 0 se reporta como DEGRADED."""
        s = dyn.spectral
        if ((not s.torsion_free) or s.krull < _DEFAULT_KRULL_FLOOR
                or s.poincare_canonical_ok is False or s.snf_det_ok is False):
            spectral = HeytingTruthValue.VETOED
        elif s.rw_norm > _RW_DEGRADED_TOL:
            spectral = HeytingTruthValue.DEGRADED
        else:
            spectral = HeytingTruthValue.COHERENT
        if dyn.gromov_violated or dyn.melnikov_chaos or not s.delaunay_valid:
            dynamical = HeytingTruthValue.VETOED
        elif (dyn.harmonicity_residual > _HARMONICITY_TOL
              or not dyn.melnikov_eps_valid
              or dyn.poincare_consistent is False
              or not dyn.secular_valid):
            dynamical = HeytingTruthValue.DEGRADED
        else:
            dynamical = HeytingTruthValue.COHERENT
        rsi = (HeytingTruthValue.COHERENT
               if kleisli.depth_accepted == requested_depth
               else HeytingTruthValue.DEGRADED)
        verifier = (HeytingTruthValue.COHERENT
                    if (kleisli.selftest_passed and kleisli.consistency_ok and kleisli.meta_verified)
                    else HeytingTruthValue.VETOED)
        return spectral.meet(dynamical).meet(rsi).meet(verifier)

    def _fase3_sign_passport(self, fields: Tuple[object, ...]) -> str:
        """HMAC-SHA256 con separación de dominio sobre serialización canónica."""
        payload = "|".join(_fmt(x) for x in fields).encode("utf-8")
        return hmac.new(self._secret_key, _PASSPORT_DOMAIN + payload, hashlib.sha256).hexdigest()

    # ── ÚLTIMO método de Fase 3: API pública de auditoría ──────────────────────
    def execute_toon_cartridge_weaver_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        *,
        raw_token_count: int = _TOON_SOURCE_TOKENS,
        positron_hmac_valid: Optional[bool] = None,
        orbital_elements: Optional[Sequence[float]] = None,
        ball_r: float = _DEFAULT_BALL_R,
        cylinder_R: float = _DEFAULT_CYLINDER_R,
        melnikov_delta: float = _MELNIKOV_DEFAULT_DELTA,
        melnikov_gamma: float = _MELNIKOV_DEFAULT_GAMMA,
        melnikov_omega: float = _MELNIKOV_DEFAULT_OMEGA,
        pendulum_amplitude: float = _DEFAULT_PENDULUM_AMPLITUDE,
        poincare_periods: int = _POINCARE_PERIODS,
        poincare_steps_per_period: int = _POINCARE_STEPS_PER_PERIOD,
        secular_system: Optional[Sequence[Sequence[float]]] = None,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        initial_lip: float = _KLEISLI_LIP_T,
        capacity_series: Optional[Sequence[float]] = None,
    ) -> ToonCartridgeWeaverSpectrumReport:
        """Auditoría espectral completa. ÚLTIMO método de F3; API pública.
        Flujo: F1.emite_semilla → F2.inicio/…/F2.emite_semilla → F3.inicio/…/veredicto."""
        orbit = _validate_orbit(_DEFAULT_ORBIT if orbital_elements is None else orbital_elements)
        sec_pairs = _validate_secular_system(
            _SECULAR_DEFAULT if secular_system is None else secular_system
        )
        amp = _validate_amplitude(pendulum_amplitude)
        n_per = _validate_int(poincare_periods, "poincare_periods", 0, 20000)
        n_st = _validate_int(poincare_steps_per_period, "poincare_steps_per_period", 8, 4096)
        depth = _validate_int(rsi_max_depth, "rsi_max_depth", 0, _RSI_MAX_DEPTH)
        lam = _validate_lambda(initial_lip)
        raw = _validate_int(raw_token_count, "raw_token_count", _TOON_BUDGET_TOKENS)
        positron = validate_positron(positron_hmac_valid)
        series_str = ("none" if capacity_series is None
                      else ",".join(_fmt(float(x)) for x in capacity_series))
        params_canonical = "|".join((
            ",".join(_fmt(x) for x in orbit),
            _fmt(float(ball_r)), _fmt(float(cylinder_R)),
            _fmt(float(melnikov_delta)), _fmt(float(melnikov_gamma)), _fmt(float(melnikov_omega)),
            _fmt(amp), _fmt(n_per), _fmt(n_st),
            ";".join(f"{_fmt(m)},{_fmt(a)}" for m, a in sec_pairs),
            _fmt(depth), _fmt(lam), _fmt(raw), _fmt(positron), series_str,
        ))
        seed = self._fase1_emit_spectral_seed(
            apu_cost_matrix, exponents, raw, positron, orbit, params_canonical
        )
        dyn = self._fase2_emit_dynamical_seed(
            seed, ball_r, cylinder_R, melnikov_delta, melnikov_gamma, melnikov_omega,
            amp, n_per, n_st, sec_pairs,
        )
        dyn, eff_depth, lam = self._fase3_inicio_desde_semilla_dinamica(dyn, depth, lam)
        kleisli = self.evaluate_rsi3_banach_kleisli(lam, eff_depth)
        verdict = self._fase3_heyting_adjudicate(dyn, kleisli, depth)
        jerk = _third_difference(capacity_series)
        passport = self._fase3_sign_passport((
            _ENGINE_VERSION,
            verdict,
            seed.input_digest,
            seed.rw_norm,
            seed.torsion_free,
            ",".join(str(d) for d in seed.snf_diagonal_factors),
            seed.torsion_order,
            seed.cokernel_free_rank,
            seed.snf_det_ok,
            seed.krull,
            seed.novikov_norm,
            seed.poincare_canonical_ok,
            seed.delaunay_valid,
            seed.orbital_residual,
            seed.orbital_ok,
            dyn.harmonicity_residual,
            dyn.gromov_violated,
            dyn.melnikov_chaos,
            dyn.melnikov_eps_valid,
            dyn.poincare_lyapunov,
            dyn.poincare_consistent,
            dyn.lindstedt_exact,
            ",".join(_fmt(g) for g in dyn.secular_frequencies),
            dyn.secular_valid,
            kleisli.lip_analytic,
            kleisli.lip_safety,
            kleisli.depth_accepted,
            kleisli.certificate_chain,
            kleisli.selftest_passed,
            kleisli.consistency_ok,
            kleisli.meta_verified,
            dyn.dirac_kev,
        ))
        return ToonCartridgeWeaverSpectrumReport(
            rademacher_walsh_norm=seed.rw_norm,
            smith_torsion_free=seed.torsion_free,
            smith_torsion_order=seed.torsion_order,
            cokernel_free_rank=seed.cokernel_free_rank,
            full_rank=seed.full_rank,
            snf_diagonal_factors=seed.snf_diagonal_factors,
            snf_determinant_certificate=seed.snf_det_ok,
            krull_valuation=seed.krull,
            novikov_norm=seed.novikov_norm,
            raw_token_count=seed.raw_token_count,
            compressed_token_count=seed.compressed_token_count,
            kv_cache_compression_pct=seed.kv_cache_pct,
            cauchy_riemann_residual=dyn.harmonicity_residual,
            is_gromov_violated=dyn.gromov_violated,
            gromov_capacity_ball=dyn.capacity_ball,
            gromov_capacity_cylinder=dyn.capacity_cylinder,
            melnikov_chaos_detected=dyn.melnikov_chaos,
            melnikov_amplitude=dyn.melnikov_amplitude,
            melnikov_offset=dyn.melnikov_offset,
            melnikov_transversality_margin=dyn.melnikov_margin,
            melnikov_eps_regime_valid=dyn.melnikov_eps_valid,
            poincare_lyapunov=dyn.poincare_lyapunov,
            poincare_melnikov_consistent=dyn.poincare_consistent,
            poincare_lindstedt_exact=dyn.lindstedt_exact,
            poincare_lindstedt_series=dyn.lindstedt_series,
            poincare_lindstedt_error=dyn.lindstedt_error,
            delaunay_LGH=seed.delaunay_LGH,
            delaunay_valid=seed.delaunay_valid,
            orbital_state_residual=seed.orbital_residual,
            orbital_state_ok=seed.orbital_ok,
            poincare_variables=seed.poincare_variables,
            poincare_canonical_residual=seed.poincare_canonical_residual,
            poincare_canonical_ok=seed.poincare_canonical_ok,
            secular_frequencies=dyn.secular_frequencies,
            secular_symmetry_residual=dyn.secular_sym_residual,
            secular_valid=dyn.secular_valid,
            banach_lip_analytic=kleisli.lip_analytic,
            banach_contraction_lip=kleisli.lip_analytic,
            banach_safety_lip=kleisli.lip_safety,
            banach_fixed_point=kleisli.fixed_point,
            banach_a_posteriori_error=kleisli.a_posteriori_error,
            banach_residual=kleisli.residual,
            banach_meta_verified=kleisli.meta_verified,
            rsi_accepted_depth=kleisli.depth_accepted,
            rsi_empirical_lipschitz=kleisli.lip_empirical,
            rsi_certificate_chain=kleisli.certificate_chain,
            spohn_entropy_prod=dyn.spohn_production,
            capacity_third_difference=jerk,
            dirac_annihilation_kev=dyn.dirac_kev,
            metamorphic_selftest_passed=kleisli.selftest_passed,
            input_digest=seed.input_digest,
            truth_value=verdict,
            passport_hash=passport,
            schema_version=_SCHEMA_VERSION,
        )


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonCartridgeWeaverEngine(_Fase3_Kleisli_RSI3):
    """Motor espectral de tejeduría TOON (v15.0.0). La lógica vive en las fases anidadas.
    Pase `secret_key` explícita; `allow_dev_key=True` solo para desarrollo."""


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = BoolNovikovToonCartridgeWeaverEngine(secret_key=b"demo-key-0123456789abcdef")
    print(f"Autoprueba metamórfica: {engine._fase3_metamorphic_selftest()}")
    exps = np.array([0.15, 0.30, 0.45])
    # A: la cuantización a escala 100 produce torsión. B: unimodular [[1,1],[0,1]]-like.
    matrix_a = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.9]])
    matrix_b = np.array([[0.01, 0.01], [0.0, 0.01]])
    for label, mat, kwargs in (
        ("A", matrix_a, {}),
        ("B", matrix_b, {}),
        ("C (Melnikov caos γ=0.1)", matrix_b, {"melnikov_gamma": 0.1}),
    ):
        rep = engine.execute_toon_cartridge_weaver_audit(mat, exps, **kwargs)
        print(f"\n=== Escenario {label}: {rep.truth_value.name} ({hex(rep.truth_value.value)}) ===")
        print(f"  SNF (cadena d₁|d₂|…)  : {rep.snf_diagonal_factors}  torsión libre={rep.smith_torsion_free}"
              f"  |tors|={rep.smith_torsion_order}  rango libre={rep.cokernel_free_rank}"
              f"  cert.det={rep.snf_determinant_certificate}")
        print(f"  Influencia normalizada: {rep.rademacher_walsh_norm:.4f}")
        print(f"  Krull / ‖A‖_Nov       : {rep.krull_valuation:.3f} / {rep.novikov_norm:.4f}")
        print(f"  Compresión KV-Cache   : {rep.kv_cache_compression_pct:.4f}% ({rep.compressed_token_count} tokens)")
        print(f"  Residuo armonicidad   : {rep.cauchy_riemann_residual:.3e}")
        print(f"  Gromov violado        : {rep.is_gromov_violated}")
        print(f"  Melnikov caos/margen  : {rep.melnikov_chaos_detected} / {rep.melnikov_transversality_margin:.5f}"
              f"  (ε válido={rep.melnikov_eps_regime_valid})")
        lyap = "None" if rep.poincare_lyapunov is None else f"{rep.poincare_lyapunov:.5f}"
        print(f"  Poincaré λ_max (evid.): {lyap}  consistente={rep.poincare_melnikov_consistent}")
        print(f"  Lindstedt Ω exacta    : {rep.poincare_lindstedt_exact:.10f} (serie err={rep.poincare_lindstedt_error:.2e})")
        print(f"  Estado orbital        : ok={rep.orbital_state_ok} (res={rep.orbital_state_residual})")
        print(f"  Poincaré canónico ok  : {rep.poincare_canonical_ok} (res={rep.poincare_canonical_residual})")
        print(f"  Secular g [rad/año]   : {tuple(round(g, 8) for g in rep.secular_frequencies)}"
              f"  sim.res={rep.secular_symmetry_residual:.1e}  válido={rep.secular_valid}")
        print(f"  Banach λ³ / λ³+ε / x* : {rep.banach_lip_analytic:.5f}/{rep.banach_safety_lip:.5f}"
              f" / {rep.banach_fixed_point:.6f}  meta={rep.banach_meta_verified}")
        print(f"  RSI³ profundidad/emp  : {rep.rsi_accepted_depth} / {tuple(round(e, 5) for e in rep.rsi_empirical_lipschitz)}")
        print(f"  Cadena RSI³           : {rep.rsi_certificate_chain[:16]}...")
        print(f"  Spohn σ / Dirac keV   : {rep.spohn_entropy_prod:.6f} / {rep.dirac_annihilation_kev:.4f}")
        print(f"  Pasaporte HMAC        : {rep.passport_hash[:16]}...")
    rep_pos = engine.execute_toon_cartridge_weaver_audit(matrix_b, exps, positron_hmac_valid=True)
    print(f"\nEscenario B con positrón validado: Dirac = {rep_pos.dirac_annihilation_kev:.4f} keV")