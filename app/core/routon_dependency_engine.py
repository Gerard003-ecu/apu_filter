# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Routon Dependency Engine (Motor de Calibre Routónico 128D)          ║
║ Ruta   : app/core/routon_dependency_engine.py                                ║
║ Versión: 1.1.0-Doctoral-128D-CayleyDickson-Eneagonal-Moufang-KBN-Nested3     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:                                  ║
║ Este módulo implementa el motor de cálculo ciego en la FPU para la variedad  ║
║ de los Routons reales \mathbb{R}\mathrm{ou} (128 dimensiones), estructurada  ║
║ mediante la duplicación iterativa de Cayley-Dickson sobre el álgebra de los  ║
║ Chingones \mathbb{X}:                                                        ║
║                                                                              ║
║     \mathbb{R}\mathrm{ou} \cong \mathbb{X}\times\mathbb{X},                  ║
║     \dim_{\mathbb{R}}=128.                                                   ║
║                                                                              ║
║ Opera como resolvedor de alta fidelidad para interdependencias de 9 vías     ║
║ (8-símplices en el complejo simplicial de la Malla Agéntica), calculando:    ║
║                                                                              ║
║   1. Norma del asociador eneagonal A_9 y diámetro de Stasheff A_4.           ║
║   2. Distorsión de Moufang (identidad media, API 3.x):                       ║
║        A_M(R,S,T)=(R(ST))R-(RS)(TR).                                         ║
║      y defectos L/R/M de las tres identidades de octoniones.                 ║
║   3. Alternatividad L/R, flexibilidad y potencia.                            ║
║   4. Forma cuadrática N(R)=R\overline{R} y defecto de Banach/Hurwitz.        ║
║   5. Distancia espectral al esquema de divisores de cero                     ║
║        \mathcal{N}(\mathbb{R}\mathrm{ou})                                    ║
║        =\{\,r:\ker L_r\neq\{0\}\,\}.                                         ║
║                                                                              ║
║ AXIOMAS (álgebra de Cayley-Dickson, convención interna):                     ║
║   (CD-1) Duplicación:  \mathbb{R}\mathrm{ou}=\mathbb{X}\times\mathbb{X}.     ║
║   (CD-2) Producto:     (a,b)(c,d)=(ac-\overline{d}\,b,\ da+b\,\overline{c}). ║
║   (CD-3) Conjugación:  \overline{(a,b)}=(\overline{a},-b).                   ║
║   (CD-4) Anti-homom.:  \overline{xy}=\overline{y}\,\overline{x}.             ║
║   (CD-5) Forma cuadr.: N(x):=x\overline{x}\in\mathbb{R}\,e_0\ exacto.        ║
║   (H)    Hurwitz:      N(xy)=N(x)N(y) \Leftrightarrow \dim\le 8.             ║
║   (B)    Banach:       \|\cdot\|_2 no es submultiplicativa en \dim>8.        ║
║   (A3)   Asociador:    [x,y,z]=(xy)z-x(yz).                                  ║
║   (Alt)  Alternatividad: [x,x,y]=[x,y,y]=0  (falla en \dim\ge 16).           ║
║   (F)    Flexibilidad: [x,y,x]=0.                                            ║
║   (P)    Potencia:     [x,x,x]=0.                                            ║
║   (Mf)   Moufang:      últimas identidades exactas en \dim=8 (octoniones).   ║
║          L: (x(yx))z=x(y(xz));  R: ((zx)y)x=z((xy)x);                        ║
║          M: (xy)(zx)=(x(yz))x.                                               ║
║   (A4)   Stasheff:     diámetro de las 5 asociaciones plenas de 4 factores.  ║
║   (A9)   Eneagonal:    A_9=((((((((x_1 x_2)x_3)x_4)x_5)x_6)x_7)x_8)x_9)      ║
║                              -x_1(x_2(x_3(x_4(x_5(x_6(x_7(x_8 x_9)))))))).   ║
║   (N)    Cono nulo:    xy=0,\ x,y\neq 0 \Leftrightarrow \sigma_{\min}(L_x)=0.║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA FPU):                   ║
║   Fase 1  Observe+Orient : Ingesta 128D, C-D, norma KBN, N(x), Hurwitz.      ║
║           Morfismo terminal : observe_metrics → RoutonMetricsReport.         ║
║   Fase 2  Decide         : A_3, Moufang L/M/R, A_9, Stasheff, 8-símplex.     ║
║           Objeto inicial    : RoutonMetricsReport.                           ║
║           Morfismo terminal : decide_from_metrics_report                     ║
║                               → RoutonDecisionState.                         ║
║   Fase 3  Act            : L_r, cono nulo 128D, sello criptográfico canónico.║
║           Objeto inicial    : RoutonDecisionState.                           ║
║           Morfismo terminal : execute_eneagonal_audit                        ║
║                               → RoutonEngineState.                           ║
║                                                                              ║
║ Anidación ontológica (no meramente OOP):                                     ║
║   Phase3 ⊏ Phase2 ⊏ Phase1,  Act \circ Decide \circ Observe.                 ║
║ El último morfismo de la Fase k es el objeto inicial de la Fase k+1.         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, replace
from typing import Final, Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

__version__: Final[str] = (
    "1.1.0-Doctoral-128D-CayleyDickson-Eneagonal-Moufang-KBN-Nested3"
)

__all__ = [
    "RoutonDependencyEngine",
    "RoutonState",
    "RoutonMetricsReport",
    "RoutonEneagonalReport",
    "RoutonDecisionState",
    "RoutonEngineState",
    "NullConeReport",
    "RoutonThresholds",
    "CayleyDicksonAlgebra128",
    "KBNSummationKernel",
    "Phase1_RoutonMetricObserver",
    "Phase2_EneagonalAssociatorCalculator",
    "Phase3_RoutonNullConeEvaluator",
    "RoutonEngineError",
    "RoutonDimensionError",
    "RoutonNumericalSingularityError",
    "RoutonCompositionError",
    "RoutonNullConeSingularityError",
]

logger = logging.getLogger("APU.Physics.RoutonDependencyEngine")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_ROUTON_DIM: Final[int] = 128
_CHINGON_DIM: Final[int] = 64
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_SQRT_DBL_MAX: Final[float] = float(math.sqrt(np.finfo(np.float64).max))
_SCALE_OVERFLOW_TRIGGER: Final[float] = 1e150
_SIMPLEX_FACTORIAL: Final[int] = 40320  # 8!

if _ROUTON_DIM != 2 * _CHINGON_DIM or not (
    _ROUTON_DIM > 0 and (_ROUTON_DIM & (_ROUTON_DIM - 1)) == 0
):
    raise RuntimeError(
        "Inconsistencia dimensional Cayley-Dickson: se exige "
        "dim(Rou)=128=2·dim(X) y potencia de dos."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §A. JERARQUÍA DE EXCEPCIONES ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════
class RoutonEngineError(Exception):
    r"""Excepción raíz para errores del motor de calibre routónico 128D."""


class RoutonDimensionError(RoutonEngineError):
    r"""Detonada cuando un vector no posee estrictamente 128 dimensiones reales
    (o, en el núcleo CD, cuando la dimensión no es potencia de dos)."""


class RoutonNumericalSingularityError(RoutonEngineError):
    r"""Detonada cuando aparecen NaN, Inf o desbordamientos numéricos no auditables."""


class RoutonCompositionError(RoutonEngineError):
    r"""Detonada ante rupturas severas de la ley de composición normada de Hurwitz."""


class RoutonNullConeSingularityError(RoutonEngineError):
    r"""Detonada ante penetración crítica en la variedad de divisores de cero 128D."""


# ═══════════════════════════════════════════════════════════════════════════════
# §B. FUNCIONES AUXILIARES DE METROLOGÍA NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════
def _is_power_of_two(n: int) -> bool:
    """Verifica que una dimensión algebraica sea potencia de dos estrictamente positiva."""
    return n > 0 and (n & (n - 1)) == 0


def _log_norm(norm: float) -> float:
    """Logaritmo neperiano seguro de una norma no negativa (0 ↦ -∞)."""
    if norm <= 0.0:
        return -math.inf
    return math.log(norm)


def _safe_exp(log_value: float) -> float:
    """Exponencial saturada en el rango de float64."""
    if log_value == -math.inf:
        return 0.0
    if log_value == math.inf:
        return math.inf
    if not math.isfinite(log_value):
        return math.inf if log_value > 0.0 else 0.0
    if log_value >= _LOG_DBL_MAX:
        return math.inf
    if log_value <= _LOG_DBL_TINY:
        return 0.0
    return float(math.exp(log_value))


def _within_tolerance(
    value: float,
    target: float,
    absolute: float,
    relative: float,
) -> bool:
    """Criterio IEEE mixto: |v-t| ≤ atol + rtol·max(|v|,|t|, piso)."""
    if not (math.isfinite(value) and math.isfinite(target)):
        return False
    scale = max(abs(value), abs(target), _WILKINSON_FLOOR)
    return abs(value - target) <= absolute + relative * scale


def _canonical_float_bytes(value: float) -> bytes:
    """Serialización canónica little-endian de un escalar float64."""
    x = float(value)
    if math.isnan(x):
        return b"\x7fNAN\x00\x00\x00"
    if math.isinf(x):
        return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
    if x == 0.0:
        return struct.pack("<d", 0.0)
    return struct.pack("<d", x)


def _prepare_routon_vector(value: Sequence[float], name: str) -> np.ndarray:
    """
    Valida, normaliza y congela un vector routónico en R^{128}.

    Condiciones:
    - Forma estricta (128,); se rechaza cualquier otra inmersión.
    - Componentes estrictamente finitas (sin NaN/Inf).
    - Copia propia contigua en float64.
    - Bandera write=False para inmutabilidad en RAM.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape != (_ROUTON_DIM,):
        raise RoutonDimensionError(
            f"{name} debe ser estrictamente de dimensión {_ROUTON_DIM}. "
            f"Obtenido: shape={arr.shape}."
        )

    arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
    if not np.all(np.isfinite(arr)):
        raise RoutonNumericalSingularityError(
            f"{name} contiene NaN o Inf. La FPU no puede auditar singularidades no finitas."
        )

    arr.setflags(write=False)
    return arr


def _sha256_of_arrays(arrays: Iterable[np.ndarray]) -> str:
    """Firma SHA-256 determinista little-endian de una secuencia de arreglos float64."""
    hasher = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64), dtype=np.float64)
        hasher.update(np.asarray(a, dtype="<f8").tobytes(order="C"))
    return hasher.hexdigest()


def _apply_log_scale(vector: np.ndarray, log_scale: float) -> np.ndarray:
    """Reescala `vector * exp(log_scale)` evitando overflow intermedio."""
    if log_scale == 0.0:
        return vector

    max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
    if max_abs == 0.0:
        return vector

    factor = _safe_exp(log_scale + math.log(max_abs))
    out = (vector / max_abs) * factor
    if not np.all(np.isfinite(out)):
        raise RoutonNumericalSingularityError(
            "El reescalado logarítmico del producto CD excedió el rango float64."
        )
    return out


def _clip_nonnegative(value: float) -> float:
    """Proyección sobre [0, +∞] con saturación de residuos numéricos negativos."""
    if not math.isfinite(value):
        return value
    return value if value >= 0.0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DEL ESPACIO DE FASE ROUTÓNICO (128D)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RoutonThresholds:
    r"""
    Umbrales metrológicos inmutables del motor routónico.

    Fronteras de:
    - estabilidad de Hurwitz / defecto de Banach,
    - distorsión de Moufang L/M/R,
    - frustración eneagonal y diámetro de Stasheff,
    - incursión en el cono nulo,
    - degeneración espectral del 8-símplex agéntico.
    """
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    zero_norm_threshold: float = 1e-14

    hurwitz_absolute: float = 1e-9
    hurwitz_relative: float = 1e-8

    moufang_absolute: float = 500.0
    moufang_relative: float = 1e-2

    eneagonal_absolute: float = 10.0
    eneagonal_relative: float = 1e-2

    null_absolute: float = 1e-12
    null_relative: float = 1e-8
    null_depth_threshold: float = 0.99

    condition_number_limit: float = 1e12
    quadratic_leakage_limit: float = 1e-8

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"El umbral {name} debe ser finito y estrictamente positivo.")

        if self.null_depth_threshold > 1.0:
            raise ValueError("null_depth_threshold debe estar en (0, 1].")


@dataclass(frozen=True, slots=True)
class RoutonState:
    r"""
    Estado inmutable del multivector Routónico en \mathbb{R}^{128}.

    Orden de campos 3.x (compatibilidad):
        vector_rep, norm, chingon_pair, sha256_hash.

    Bajo el isomorfismo de Cayley-Dickson:
        R = (X_1, X_2) \in \mathbb{X}\times\mathbb{X}.

    Invariantes auditados:
        \|R\|_2,\quad N(R)=R\overline{R},\quad \mathrm{leak}(N)=\|\mathrm{Im}\,N(R)\|_2.
    """
    vector_rep: np.ndarray
    norm: float
    chingon_pair: Tuple[np.ndarray, np.ndarray]
    sha256_hash: str

    norm_squared: float = 0.0
    is_unitary: bool = False
    is_zero: bool = False
    real_part: float = 0.0
    imag_norm: float = 0.0
    quadratic_scalar: float = 0.0
    quadratic_leakage: float = 0.0


@dataclass(frozen=True, slots=True)
class RoutonMetricsReport:
    r"""
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Norma, forma cuadrática, composición de Hurwitz y defecto de Banach
    del producto routónico A·B.

    Campos 3.x: state_a, state_b, product_state,
                hurwitz_composition_error, is_hurwitz_stable.
    """
    state_a: RoutonState
    state_b: RoutonState
    product_state: RoutonState
    hurwitz_composition_error: float
    is_hurwitz_stable: bool

    product_norm: float = 0.0
    expected_norm: float = 0.0
    hurwitz_absolute_error: float = 0.0
    hurwitz_relative_error: float = 0.0
    hurwitz_signed_defect: float = 0.0
    composition_ratio: float = 1.0
    is_banach_submultiplicative: bool = True


@dataclass(frozen=True, slots=True)
class RoutonEneagonalReport:
    r"""
    Auditoría de la Fase 2: asociadores, Moufang L/M/R, Stasheff,
    espectro y topología combinatoria del 8-símplex agéntico.

    Campos 3.x: trilateral_associator_norm, moufang_distortion_norm,
                eneagonal_associator_norm, is_eneagonal_stable,
                frustration_index.
    """
    trilateral_associator_norm: float
    moufang_distortion_norm: float
    eneagonal_associator_norm: float
    is_eneagonal_stable: bool
    frustration_index: float

    moufang_relative_norm: float = 0.0
    eneagonal_relative_norm: float = 0.0
    moufang_frustration_index: float = 0.0

    moufang_threshold_used: float = 0.0
    eneagonal_threshold_used: float = 0.0

    is_moufang_stable: bool = False
    is_eneagonal_norm_stable: bool = False

    simplex_condition_number: float = math.inf
    laplacian_connectivity: float = 0.0

    diagnosis: str = ""

    left_moufang_defect: float = 0.0
    right_moufang_defect: float = 0.0
    middle_moufang_defect: float = 0.0
    alternative_strain_norm: float = 0.0
    right_alternative_strain_norm: float = 0.0
    flexibility_defect: float = 0.0
    power_associator_norm: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    simplex_volume: float = 0.0
    laplacian_spectral_gap: float = 0.0
    estimated_connected_components: int = 1


@dataclass(frozen=True, slots=True)
class RoutonDecisionState:
    r"""
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Empaqueta el reporte metrológico, el reporte eneagonal y los nueve
    estados del 8-símplex, de modo que Act no reinstancia Observe.
    """
    metrics_report: RoutonMetricsReport
    eneagonal_report: RoutonEneagonalReport
    states: Tuple[
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
        RoutonState,
    ]


@dataclass(frozen=True, slots=True)
class NullConeReport:
    r"""
    Reporte de la Fase 3: fricción, profundidad y caracterización espectral
    de la incursión en \mathcal{N}(\mathbb{R}\mathrm{ou}).
    """
    product_norm: float
    expected_norm: float

    absolute_friction: float
    relative_defect: float
    null_depth: float

    sigma_min_left_r1: float
    sigma_min_left_r2: float
    commutator_norm: float

    is_trivial_null: bool
    is_null_cone_penetrated: bool


@dataclass(frozen=True, slots=True)
class RoutonEngineState:
    r"""Certificado inmutable final del motor routónico entregado a los soberanos."""
    metrics_report: RoutonMetricsReport
    eneagonal_report: RoutonEneagonalReport
    null_report: NullConeReport

    fpu_execution_time_ms: float
    cryptographic_seal: str
    engine_version: str = __version__

    @property
    def null_cone_friction(self) -> float:
        """Compatibilidad semántica con la versión 3.x / 1.0."""
        return self.null_report.absolute_friction

    @property
    def is_null_cone_penetrated(self) -> bool:
        """Compatibilidad semántica con la versión 3.x / 1.0."""
        return self.null_report.is_null_cone_penetrated


# ═══════════════════════════════════════════════════════════════════════════════
# §D. NÚCLEO NUMÉRICO KAHAN-BABUŠKA-NEUMAIER (KBN)
# ═══════════════════════════════════════════════════════════════════════════════
class KBNSummationKernel:
    r"""
    Núcleo de sumación compensada Kahan-Babuška-Neumaier.

    Distinción axiomática (a menudo confundida):
    - Kahan clásico: compensación unidireccional `c = (t-s)-y`.
    - KBN / Neumaier: el sumando de mayor magnitud dona el residuo,
      lo que es estable cuando los sumandos cambian de escala.

    Aquí se implementa KBN genuino. `math.fsum` (Shewchuk) se usa como
    corroboración de la suma de cuadrados escalada.
    """

    __slots__ = ()

    @staticmethod
    def sum(values: np.ndarray) -> float:
        r"""Sumación compensada KBN sobre un arreglo numpy ravelado."""
        total = 0.0
        compensation = 0.0

        for raw in np.ravel(values):
            x = float(raw)
            if not math.isfinite(x):
                raise RoutonNumericalSingularityError(
                    "La sumación KBN detectó un valor no finito."
                )

            t = total + x
            if abs(total) >= abs(x):
                compensation += (total - t) + x
            else:
                compensation += (x - t) + total
            total = t

        result = total + compensation
        if not math.isfinite(result):
            raise RoutonNumericalSingularityError(
                "La sumación KBN excedió el rango representable de la FPU."
            )
        return result

    @staticmethod
    def norm(values: np.ndarray) -> float:
        r"""
        Norma euclídea robusta con escalado y sumación KBN:

            \|x\|_2 = m \sqrt{\sum_i (x_i/m)^2},\qquad m=\max_i |x_i|.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0

        if not np.all(np.isfinite(arr)):
            raise RoutonNumericalSingularityError(
                "No se puede calcular una norma KBN sobre componentes no finitas."
            )

        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            return 0.0

        scaled = arr / max_abs
        squares = scaled * scaled
        ssq = KBNSummationKernel.sum(squares)

        try:
            ssq_ref = float(math.fsum(map(float, np.ravel(squares))))
            if math.isfinite(ssq_ref) and abs(ssq - ssq_ref) > 1e-12 * max(1.0, abs(ssq_ref)):
                logger.debug(
                    "Discrepancia KBN vs fsum en suma de cuadrados: kbn=%.16e fsum=%.16e",
                    ssq,
                    ssq_ref,
                )
        except (OverflowError, ValueError):
            pass

        if ssq < 0.0:
            if ssq >= -10.0 * _MACHINE_EPS:
                ssq = 0.0
            else:
                raise RoutonNumericalSingularityError(
                    "La suma de cuadrados escalada produjo un valor negativo no físico."
                )

        if max_abs >= _SQRT_DBL_MAX and ssq > 1.0:
            raise RoutonNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )

        norm_value = float(max_abs * math.sqrt(ssq))
        if not math.isfinite(norm_value):
            raise RoutonNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )
        return norm_value


# ═══════════════════════════════════════════════════════════════════════════════
# §E. NÚCLEO ALGEBRAICO DE CAYLEY-DICKSON 128D
# ═══════════════════════════════════════════════════════════════════════════════
class CayleyDicksonAlgebra128:
    r"""
    Álgebra de Cayley-Dickson recursiva de-confinada en la FPU.

    Soporta operaciones sobre
        \mathbb{R},\mathbb{C},\mathbb{H},\mathbb{O},
        \mathbb{S},\mathbb{P},\mathbb{X},\mathbb{R}\mathrm{ou}.

    Producto (axioma CD-2):
        (a_1,a_2)(b_1,b_2)
        =(a_1 b_1-\overline{b_2} a_2,\ b_2 a_1+a_2\overline{b_1}).

    El producto de nivel superior se reescala para extinguir overflow
    intermedio; el núcleo recursivo no revalida (la validación es frontera).
    """

    __slots__ = ()

    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        """Compatibilidad con la versión 3.x: sumación compensada KBN genuina."""
        return KBNSummationKernel.sum(np.asarray(arr, dtype=np.float64))

    @staticmethod
    def _validate_power_of_two_vector(value: np.ndarray, name: str) -> np.ndarray:
        """Valida que un vector algebraico tenga dimensión 2^k, rango 1 y sea finito."""
        arr = np.asarray(value, dtype=np.float64)

        if arr.ndim != 1:
            raise RoutonDimensionError(
                f"{name} debe ser un vector algebraico unidimensional. Shape={arr.shape}."
            )
        if not _is_power_of_two(int(arr.size)):
            raise RoutonDimensionError(
                f"{name} debe tener dimensión potencia de dos. Obtenido: {arr.size}."
            )
        if not np.all(np.isfinite(arr)):
            raise RoutonNumericalSingularityError(
                f"{name} contiene componentes no finitas."
            )
        return arr

    @staticmethod
    def _conjugate_fast(a: np.ndarray) -> np.ndarray:
        r"""Conjugación interior (CD-3) sin revalidación: \overline{(a_1,a_2)}=(\overline{a_1},-a_2)."""
        out = a.copy()
        if out.size > 1:
            out[1:] = -out[1:]
        return out

    @classmethod
    def conjugate(cls, a: np.ndarray) -> np.ndarray:
        r"""
        Morfismo de conjugación sobre el álgebra de Cayley-Dickson.

        Involución antilineal: \overline{\overline{x}}=x, y anti-homomorfismo
        de álgebras \overline{xy}=\overline{y}\,\overline{x}.
        """
        arr = cls._validate_power_of_two_vector(a, "a")
        return cls._conjugate_fast(arr)

    @classmethod
    def conjugate_128(cls, R: np.ndarray) -> np.ndarray:
        """Conjugación de Routones 128D con validación dimensional explícita."""
        arr = cls._validate_power_of_two_vector(R, "R")
        if arr.size != _ROUTON_DIM:
            raise RoutonDimensionError(
                f"conjugate_128 requiere un vector de {_ROUTON_DIM} dimensiones. "
                f"Obtenido: {arr.size}."
            )
        return cls._conjugate_fast(arr)

    @classmethod
    def multiply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        Producto bilineal de Cayley-Dickson sobre álgebras de dimensión 2^k.

        Escalado de frontera: se multiplica (a/s_a)(b/s_b) y se restaura
        el factor s_a s_b en espacio logarítmico.
        """
        arr_a = cls._validate_power_of_two_vector(a, "a")
        arr_b = cls._validate_power_of_two_vector(b, "b")

        if arr_a.size != arr_b.size:
            raise RoutonDimensionError(
                "No es posible multiplicar álgebras de dimensiones distintas: "
                f"{arr_a.size} != {arr_b.size}."
            )

        max_a = float(np.max(np.abs(arr_a)))
        max_b = float(np.max(np.abs(arr_b)))

        if max_a == 0.0 or max_b == 0.0:
            return np.zeros(arr_a.size, dtype=np.float64)

        needs_scale = (
            max_a >= _SCALE_OVERFLOW_TRIGGER
            or max_b >= _SCALE_OVERFLOW_TRIGGER
            or (max_a * max_b) >= _SCALE_OVERFLOW_TRIGGER
        )

        if needs_scale:
            scaled_a = arr_a / max_a
            scaled_b = arr_b / max_b
            product = cls._multiply_recursive(scaled_a, scaled_b)
            product = _apply_log_scale(product, math.log(max_a) + math.log(max_b))
        else:
            product = cls._multiply_recursive(arr_a, arr_b)

        if not np.all(np.isfinite(product)):
            raise RoutonNumericalSingularityError(
                "El producto de Cayley-Dickson produjo componentes no finitas."
            )
        return product

    @classmethod
    def multiply_128(cls, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """Producto de Routones 128D con validación dimensional explícita."""
        arr_r1 = cls._validate_power_of_two_vector(R1, "R1")
        arr_r2 = cls._validate_power_of_two_vector(R2, "R2")
        if arr_r1.size != _ROUTON_DIM or arr_r2.size != _ROUTON_DIM:
            raise RoutonDimensionError(
                f"multiply_128 requiere vectores de {_ROUTON_DIM} dimensiones. "
                f"Obtenidos: {arr_r1.size}, {arr_r2.size}."
            )
        return cls.multiply(arr_r1, arr_r2)

    @classmethod
    def _multiply_recursive(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Núcleo recursivo del producto CD-2 con casos base n=1 y n=2."""
        n = a.size

        if n == 1:
            return np.array([a[0] * b[0]], dtype=np.float64)

        if n == 2:
            a0, a1 = float(a[0]), float(a[1])
            b0, b1 = float(b[0]), float(b[1])
            return np.array(
                [a0 * b0 - b1 * a1, b1 * a0 + a1 * b0],
                dtype=np.float64,
            )

        half = n // 2
        a1, a2 = a[:half], a[half:]
        b1, b2 = b[:half], b[half:]

        left = cls._multiply_recursive(a1, b1) - cls._multiply_recursive(
            cls._conjugate_fast(b2), a2
        )
        right = cls._multiply_recursive(b2, a1) + cls._multiply_recursive(
            a2, cls._conjugate_fast(b1)
        )

        out = np.empty(n, dtype=np.float64)
        out[:half] = left
        out[half:] = right
        return out

    @classmethod
    def quadratic_form(cls, a: np.ndarray) -> Tuple[float, float]:
        r"""
        Forma cuadrática N(a)=a\overline{a}.

        Devuelve (Re N(a), \|\mathrm{Im}\,N(a)\|_2). En aritmética exacta
        el segundo sumando es idénticamente nulo para toda álgebra CD.
        """
        conj = cls.conjugate(a)
        quad = cls.multiply(a, conj)
        scalar = float(quad[0])
        leakage = KBNSummationKernel.norm(quad[1:]) if quad.size > 1 else 0.0
        return scalar, leakage

    @classmethod
    def associator(cls, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        r"""Asociador de 3 vías: [a,b,c]=(ab)c-a(bc). Curvatura algebraica de primer orden."""
        left = cls.multiply(cls.multiply(a, b), c)
        right = cls.multiply(a, cls.multiply(b, c))
        return left - right

    @classmethod
    def alternative_strain(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Deformación alternativa izquierda: A_alt^L(X,Y)=[X,X,Y]."""
        return cls.associator(x, x, y)

    @classmethod
    def right_alternative_strain(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Deformación alternativa derecha: A_alt^R(X,Y)=[X,Y,Y]."""
        return cls.associator(x, y, y)

    @classmethod
    def moufang_distortion(cls, r: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        r"""
        Distorsión media de Moufang (API 3.x):

            A_{\mathrm{Moufang}}(R,S,T)=(R(ST))R-(RS)(TR).

        Equivale, salvo signo, al defecto de la identidad media de octoniones
        (xy)(zx)=(x(yz))x con x=R, y=S, z=T.
        """
        st = cls.multiply(s, t)
        rst = cls.multiply(r, st)
        left = cls.multiply(rst, r)

        rs = cls.multiply(r, s)
        tr = cls.multiply(t, r)
        right = cls.multiply(rs, tr)

        return left - right

    @classmethod
    def moufang_identities_defect(
        cls, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[float, float, float]:
        r"""
        Defectos de las tres identidades de Moufang:

            L:\ (x(yx))z - x(y(xz)),
            R:\ ((zx)y)x - z((xy)x),
            M:\ (xy)(zx) - (x(yz))x.

        Devuelve (\|L\|,\|R\|,\|M\|).
        """
        yx = cls.multiply(y, x)
        xy = cls.multiply(x, y)
        xz = cls.multiply(x, z)
        zx = cls.multiply(z, x)
        yz = cls.multiply(y, z)

        left_m = cls.multiply(cls.multiply(x, yx), z) - cls.multiply(x, cls.multiply(y, xz))
        right_m = cls.multiply(cls.multiply(zx, y), x) - cls.multiply(z, cls.multiply(xy, x))
        mid_m = cls.multiply(xy, zx) - cls.multiply(cls.multiply(x, yz), x)

        return (
            KBNSummationKernel.norm(left_m),
            KBNSummationKernel.norm(right_m),
            KBNSummationKernel.norm(mid_m),
        )

    @classmethod
    def eneagonal_associator(
        cls,
        x1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        x4: np.ndarray,
        x5: np.ndarray,
        x6: np.ndarray,
        x7: np.ndarray,
        x8: np.ndarray,
        x9: np.ndarray,
    ) -> np.ndarray:
        r"""
        Asociador eneagonal de 9 vías (8-símplex):

            A_9(X_1,\dots,X_9)
            =((((((((X_1 X_2)X_3)X_4)X_5)X_6)X_7)X_8)X_9)
             -X_1(X_2(X_3(X_4(X_5(X_6(X_7(X_8 X_9)))))))).
        """
        p12 = cls.multiply(x1, x2)
        p123 = cls.multiply(p12, x3)
        p1234 = cls.multiply(p123, x4)
        p12345 = cls.multiply(p1234, x5)
        p123456 = cls.multiply(p12345, x6)
        p1234567 = cls.multiply(p123456, x7)
        p12345678 = cls.multiply(p1234567, x8)
        left = cls.multiply(p12345678, x9)

        p89 = cls.multiply(x8, x9)
        p789 = cls.multiply(x7, p89)
        p6789 = cls.multiply(x6, p789)
        p56789 = cls.multiply(x5, p6789)
        p456789 = cls.multiply(x4, p56789)
        p3456789 = cls.multiply(x3, p456789)
        p23456789 = cls.multiply(x2, p3456789)
        right = cls.multiply(x1, p23456789)

        return left - right

    @classmethod
    def stasheff_pentagon_associations(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Las cinco asociaciones plenas de un producto de 4 factores
        (pentágono de Stasheff / 2-celda de A_∞):

            ((ab)c)d,\ (a(bc))d,\ (ab)(cd),\ a((bc)d),\ a(b(cd)).
        """
        ab = cls.multiply(a, b)
        bc = cls.multiply(b, c)
        cd = cls.multiply(c, d)
        return (
            cls.multiply(cls.multiply(ab, c), d),
            cls.multiply(cls.multiply(a, bc), d),
            cls.multiply(ab, cd),
            cls.multiply(a, cls.multiply(bc, d)),
            cls.multiply(a, cls.multiply(b, cd)),
        )

    @classmethod
    def left_multiplication_matrix(cls, p: np.ndarray) -> np.ndarray:
        r"""
        Operador de multiplicación a la izquierda L_p:\mathbb{R}\mathrm{ou}\to\mathbb{R}\mathrm{ou},
        y \mapsto p y, en la base canónica \{e_0,\ldots,e_{127}\}.

        p es divisor de cero izquierdo \Leftrightarrow \det L_p=0
        \Leftrightarrow \sigma_{\min}(L_p)=0.
        """
        arr = cls._validate_power_of_two_vector(p, "p")
        dim = int(arr.size)
        op = np.empty((dim, dim), dtype=np.float64)
        eye_col = np.zeros(dim, dtype=np.float64)
        for j in range(dim):
            eye_col[j] = 1.0
            op[:, j] = cls.multiply(arr, eye_col)
            eye_col[j] = 0.0
        return op


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FASE 1 — OBSERVE + ORIENT
#     Morfismo terminal: observe_metrics → RoutonMetricsReport
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_RoutonMetricObserver:
    r"""
    FASE 1 — Observe + Orient.

    Ingesta de tensores 128D, descomposición de Cayley-Dickson,
    construcción de estados inmutables, auditoría de N(x) y de la
    composición de Hurwitz en la FPU.

    Categoría fuente: \mathbf{Vec}_{128}\times\mathbf{Vec}_{128}.
    Categoría meta  : \mathbf{Metrics}.

    El método terminal `observe_metrics` es el objeto inicial de la Fase 2.
    """

    __slots__ = ("_thresholds",)

    def __init__(
        self,
        tolerance: Optional[float] = None,
        thresholds: Optional[RoutonThresholds] = None,
    ) -> None:
        if thresholds is None:
            if tolerance is not None:
                tol = float(tolerance)
                if not math.isfinite(tol) or tol <= 0.0:
                    raise ValueError("tolerance debe ser finito y estrictamente positivo.")
                thresholds = RoutonThresholds(
                    absolute_tolerance=tol,
                    relative_tolerance=max(tol * 10.0, 1e-10),
                    zero_norm_threshold=max(tol, 1e-14),
                )
            else:
                thresholds = RoutonThresholds()

        self._thresholds: Final[RoutonThresholds] = thresholds

    @property
    def thresholds(self) -> RoutonThresholds:
        """Umbral metrológico activo del motor."""
        return self._thresholds

    def build_state(self, S: Sequence[float]) -> RoutonState:
        r"""
        Instancia un `RoutonState` inmutable a partir de un vector real en R^{128}.

            S\in\mathbb{R}^{128}\ \mapsto\ R=(X_1,X_2)\in\mathbb{X}\times\mathbb{X}.

        Audita simultáneamente \|R\|_2 y la forma cuadrática N(R)=R\overline{R}.
        """
        vector = _prepare_routon_vector(S, "RoutonState.vector_rep")

        x1 = np.ascontiguousarray(vector[0:_CHINGON_DIM].copy(), dtype=np.float64)
        x2 = np.ascontiguousarray(
            vector[_CHINGON_DIM:_ROUTON_DIM].copy(), dtype=np.float64
        )
        x1.setflags(write=False)
        x2.setflags(write=False)

        norm_value = KBNSummationKernel.norm(vector)
        if not math.isfinite(norm_value):
            raise RoutonNumericalSingularityError(
                "La norma del estado routónico no es finita."
            )

        if norm_value >= _SQRT_DBL_MAX:
            norm_squared = math.inf
        else:
            norm_squared = norm_value * norm_value

        real_part = float(vector[0])
        imag_norm = KBNSummationKernel.norm(vector[1:]) if vector.size > 1 else 0.0
        quadratic_scalar, quadratic_leakage = CayleyDicksonAlgebra128.quadratic_form(vector)

        is_zero = norm_value <= self._thresholds.zero_norm_threshold
        is_unitary = _within_tolerance(
            norm_value,
            1.0,
            self._thresholds.absolute_tolerance,
            self._thresholds.relative_tolerance,
        )

        sha_hash = _sha256_of_arrays((vector, x1, x2))

        return RoutonState(
            vector_rep=vector,
            norm=norm_value,
            chingon_pair=(x1, x2),
            sha256_hash=sha_hash,
            norm_squared=norm_squared,
            is_unitary=is_unitary,
            is_zero=is_zero,
            real_part=real_part,
            imag_norm=imag_norm,
            quadratic_scalar=quadratic_scalar,
            quadratic_leakage=quadratic_leakage,
        )

    def observe_metrics(
        self,
        A_vec: Sequence[float],
        B_vec: Sequence[float],
    ) -> RoutonMetricsReport:
        r"""
        Morfismo terminal de la Fase 1.

        Ejecuta el producto routónico A·B y verifica la composición de Hurwitz:

            \delta_{\mathrm{composition}}
            =\bigl|\,\|A\cdot B\|_{\mathbb{R}\mathrm{ou}}
              -\|A\|_{\mathbb{R}\mathrm{ou}}\|B\|_{\mathbb{R}\mathrm{ou}}\bigr|.

        Firma functorial:

            \mathrm{observe\_metrics}:
                \mathbb{R}^{128}\times\mathbb{R}^{128}\longrightarrow
                \mathbf{RoutonMetricsReport}.

        El valor de retorno es el objeto inicial de
        `Phase2_EneagonalAssociatorCalculator.continue_from_metrics_report`.
        """
        state_a = self.build_state(A_vec)
        state_b = self.build_state(B_vec)

        product_vector = CayleyDicksonAlgebra128.multiply_128(
            state_a.vector_rep,
            state_b.vector_rep,
        )
        product_state = self.build_state(product_vector)

        log_expected = _log_norm(state_a.norm) + _log_norm(state_b.norm)
        expected_norm = _safe_exp(log_expected)
        product_norm = product_state.norm

        if math.isfinite(product_norm) and math.isfinite(expected_norm):
            signed_defect = product_norm - expected_norm
            absolute_error = abs(signed_defect)
        else:
            signed_defect = math.inf
            absolute_error = math.inf

        scale = (
            max(_WILKINSON_FLOOR, expected_norm)
            if math.isfinite(expected_norm)
            else _WILKINSON_FLOOR
        )
        relative_error = (
            absolute_error / scale if math.isfinite(absolute_error) else math.inf
        )

        if expected_norm > self._thresholds.zero_norm_threshold and math.isfinite(
            expected_norm
        ):
            composition_ratio = (
                product_norm / expected_norm if math.isfinite(product_norm) else math.inf
            )
        else:
            composition_ratio = (
                1.0
                if product_norm <= self._thresholds.zero_norm_threshold
                else math.inf
            )

        is_hurwitz_stable = math.isfinite(absolute_error) and (
            absolute_error
            <= self._thresholds.hurwitz_absolute
            + self._thresholds.hurwitz_relative * scale
        )

        is_banach_submultiplicative = math.isfinite(product_norm) and math.isfinite(
            expected_norm
        ) and (
            product_norm
            <= expected_norm
            + self._thresholds.hurwitz_absolute
            + self._thresholds.hurwitz_relative * scale
        )

        if product_state.quadratic_leakage > self._thresholds.quadratic_leakage_limit * max(
            1.0,
            product_state.norm_squared if math.isfinite(product_state.norm_squared) else 1.0,
        ):
            logger.debug(
                "Fase 1: fuga imaginaria de N(AB) = %.6e (norma^2 = %.6e).",
                product_state.quadratic_leakage,
                product_state.norm_squared,
            )

        if not is_hurwitz_stable:
            logger.debug(
                "Fase 1: deriva de Hurwitz. abs=%.6e rel=%.6e signed=%.6e ratio=%.6e",
                absolute_error,
                relative_error,
                signed_defect,
                composition_ratio,
            )

        return RoutonMetricsReport(
            state_a=state_a,
            state_b=state_b,
            product_state=product_state,
            hurwitz_composition_error=absolute_error,
            is_hurwitz_stable=is_hurwitz_stable,
            product_norm=product_norm,
            expected_norm=expected_norm,
            hurwitz_absolute_error=absolute_error,
            hurwitz_relative_error=relative_error,
            hurwitz_signed_defect=signed_defect,
            composition_ratio=composition_ratio,
            is_banach_submultiplicative=is_banach_submultiplicative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §G. FASE 2 — DECIDE
#     Objeto inicial = morfismo terminal de la Fase 1 (RoutonMetricsReport)
#     Morfismo terminal: decide_from_metrics_report → RoutonDecisionState
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_EneagonalAssociatorCalculator(Phase1_RoutonMetricObserver):
    r"""
    FASE 2 — Decide.

    Consume `RoutonMetricsReport` (terminal de Fase 1) y extiende la
    auditoría al 8-símplex de nueve vías: asociador trilateral, distorsión
    de Moufang L/M/R, asociador eneagonal A_9, pentágono de Stasheff,
    alternatividad, flexibilidad, potencia, volumen afín y espectro del
    laplaciano de similitud.

    El método inicial `continue_from_metrics_report` es la continuación
    formal de `Phase1.observe_metrics`.
    El método terminal `decide_from_metrics_report` produce el objeto
    inicial de la Fase 3 (`RoutonDecisionState`).
    """

    __slots__ = ()

    def continue_from_metrics_report(
        self,
        metrics_report: RoutonMetricsReport,
        r3_vec: Sequence[float],
        r4_vec: Sequence[float],
        r5_vec: Sequence[float],
        r6_vec: Sequence[float],
        r7_vec: Sequence[float],
        r8_vec: Sequence[float],
        r9_vec: Sequence[float],
        moufang_threshold: Optional[float] = None,
        eneagonal_threshold: Optional[float] = None,
    ) -> RoutonEneagonalReport:
        r"""
        Morfismo de entrada de la Fase 2 (continuación formal de
        `Phase1_RoutonMetricObserver.observe_metrics`).

            \mathrm{continue\_from\_metrics\_report}:
                \mathbf{RoutonMetricsReport}\times(\mathbb{R}^{128})^{7}
                \longrightarrow\mathbf{RoutonEneagonalReport}.
        """
        return self.decide_from_metrics_report(
            metrics_report=metrics_report,
            r3_vec=r3_vec,
            r4_vec=r4_vec,
            r5_vec=r5_vec,
            r6_vec=r6_vec,
            r7_vec=r7_vec,
            r8_vec=r8_vec,
            r9_vec=r9_vec,
            moufang_threshold=moufang_threshold,
            eneagonal_threshold=eneagonal_threshold,
        ).eneagonal_report

    def calculate_eneagonal_frustration(
        self,
        r1_vec: Sequence[float],
        r2_vec: Sequence[float],
        r3_vec: Sequence[float],
        r4_vec: Sequence[float],
        r5_vec: Sequence[float],
        r6_vec: Sequence[float],
        r7_vec: Sequence[float],
        r8_vec: Sequence[float],
        r9_vec: Sequence[float],
        eneagonal_threshold: Optional[float] = None,
        moufang_threshold: Optional[float] = None,
    ) -> RoutonEneagonalReport:
        r"""Orquesta la frustración de calibre de 9 vías sin precomputar la Fase 1."""
        states = tuple(
            self.build_state(v)
            for v in (
                r1_vec, r2_vec, r3_vec, r4_vec, r5_vec, r6_vec, r7_vec, r8_vec, r9_vec
            )
        )
        return self._eneagonal_report_from_states(
            states=states,
            moufang_threshold=moufang_threshold,
            eneagonal_threshold=eneagonal_threshold,
        )

    def compute_trilateral_associator(
        self,
        R1: RoutonState,
        R2: RoutonState,
        R3: RoutonState,
    ) -> np.ndarray:
        r"""Asociador de 3 vías [R_1,R_2,R_3]=(R_1 R_2)R_3-R_1(R_2 R_3)."""
        return CayleyDicksonAlgebra128.associator(
            R1.vector_rep, R2.vector_rep, R3.vector_rep
        )

    def compute_moufang_distortion(
        self,
        R1: RoutonState,
        R2: RoutonState,
        R3: RoutonState,
    ) -> np.ndarray:
        r"""Distorsión media de Moufang A_M(R_1,R_2,R_3)=(R_1(R_2 R_3))R_1-(R_1 R_2)(R_3 R_1)."""
        return CayleyDicksonAlgebra128.moufang_distortion(
            R1.vector_rep, R2.vector_rep, R3.vector_rep
        )

    def compute_eneagonal_associator(
        self,
        R1: RoutonState,
        R2: RoutonState,
        R3: RoutonState,
        R4: RoutonState,
        R5: RoutonState,
        R6: RoutonState,
        R7: RoutonState,
        R8: RoutonState,
        R9: RoutonState,
    ) -> np.ndarray:
        r"""Asociador eneagonal A_9 de nueve vías."""
        return CayleyDicksonAlgebra128.eneagonal_associator(
            R1.vector_rep,
            R2.vector_rep,
            R3.vector_rep,
            R4.vector_rep,
            R5.vector_rep,
            R6.vector_rep,
            R7.vector_rep,
            R8.vector_rep,
            R9.vector_rep,
        )

    def _stasheff_diameter(
        self,
        R1: RoutonState,
        R2: RoutonState,
        R3: RoutonState,
        R4: RoutonState,
    ) -> float:
        """Diámetro euclídeo del pentágono de Stasheff sobre cuatro routones."""
        associations = CayleyDicksonAlgebra128.stasheff_pentagon_associations(
            R1.vector_rep, R2.vector_rep, R3.vector_rep, R4.vector_rep
        )
        diameter = 0.0
        for i in range(len(associations)):
            for j in range(i + 1, len(associations)):
                delta = associations[i] - associations[j]
                diameter = max(diameter, KBNSummationKernel.norm(delta))
        return diameter

    def _diagnose_nine_way_simplex(
        self,
        states: Sequence[RoutonState],
    ) -> Tuple[float, float, float, float, int]:
        r"""
        Diagnóstico espectral, volumétrico y de conectividad del 8-símplex.

        Devuelve:
            1. Número de condición afín (SVD de la nube centrada por KBN).
            2. Volumen 8-dimensional (Gram / 8!).
            3. Conectividad algebraica (valor de Fiedler \lambda_2).
            4. Hueco espectral \lambda_2-\lambda_1 del laplaciano.
            5. Multiplicidad numérica de \lambda=0 (componentes conexas).
        """
        if len(states) != 9:
            raise RoutonEngineError(
                "El diagnóstico del 8-símplex requiere exactamente nueve estados routónicos."
            )

        matrix = np.vstack([state.vector_rep for state in states])

        centroid = np.empty(matrix.shape[1], dtype=np.float64)
        inv_n = 1.0 / float(matrix.shape[0])
        for j in range(matrix.shape[1]):
            centroid[j] = KBNSummationKernel.sum(matrix[:, j]) * inv_n
        centered = matrix - centroid

        try:
            singular_values = la.svdvals(centered, check_finite=True)
        except la.LinAlgError:
            singular_values = np.linalg.svd(centered, compute_uv=False)

        if singular_values.size == 0 or float(singular_values[0]) <= _WILKINSON_FLOOR:
            condition_number = math.inf
        else:
            cutoff = max(_MACHINE_EPS * float(singular_values[0]), _WILKINSON_FLOOR)
            positive = singular_values[singular_values > cutoff]
            condition_number = (
                math.inf
                if positive.size == 0
                else float(singular_values[0] / positive[-1])
            )

        origin = states[0].vector_rep
        edges = np.stack([state.vector_rep - origin for state in states[1:]])
        gram = edges @ edges.T
        gram = 0.5 * (gram + gram.T)
        sign, logabs = np.linalg.slogdet(gram)
        if sign <= 0.0 or not math.isfinite(logabs):
            simplex_volume = 0.0
        else:
            simplex_volume = _safe_exp(
                0.5 * float(logabs) - math.log(float(_SIMPLEX_FACTORIAL))
            )

        norms = np.array([state.norm for state in states], dtype=np.float64)
        valid = norms > self._thresholds.zero_norm_threshold

        if not np.any(valid):
            return condition_number, simplex_volume, 0.0, 0.0, 9

        normalized = np.zeros_like(matrix)
        normalized[valid] = matrix[valid] / norms[valid][:, np.newaxis]

        cosine = normalized @ normalized.T
        cosine = np.clip(cosine, -1.0, 1.0)
        weights = 0.5 * (1.0 + cosine)
        np.fill_diagonal(weights, 0.0)
        weights[~np.isfinite(weights)] = 0.0

        degrees = np.sum(weights, axis=1)
        laplacian = np.diag(degrees) - weights
        laplacian = 0.5 * (laplacian + laplacian.T)

        try:
            eigenvalues = la.eigvalsh(laplacian, check_finite=True)
        except la.LinAlgError:
            eigenvalues = np.linalg.eigvalsh(laplacian)

        eigenvalues = np.sort(np.real(eigenvalues))
        eig_max = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        zero_cut = max(
            float(len(states)) * _MACHINE_EPS * max(eig_max, 1.0),
            _WILKINSON_FLOOR,
        )
        nonnegative = np.maximum(eigenvalues, 0.0)

        if nonnegative.size >= 2:
            fiedler = float(nonnegative[1])
            gap = float(nonnegative[1] - nonnegative[0])
        else:
            fiedler = 0.0
            gap = 0.0

        components = int(np.sum(nonnegative <= zero_cut))
        components = min(max(components, 1), len(states))

        return condition_number, simplex_volume, fiedler, _clip_nonnegative(gap), components

    def _eneagonal_report_from_states(
        self,
        states: Sequence[RoutonState],
        moufang_threshold: Optional[float],
        eneagonal_threshold: Optional[float],
    ) -> RoutonEneagonalReport:
        """Orquesta el cálculo eneagonal completo a partir de nueve estados ya construidos."""
        if len(states) != 9:
            raise RoutonEngineError(
                "La auditoría eneagonal requiere exactamente nueve estados routónicos."
            )

        R1, R2, R3, R4, R5, R6, R7, R8, R9 = states

        assoc_3_vector = self.compute_trilateral_associator(R1, R2, R3)
        trilateral_norm = KBNSummationKernel.norm(assoc_3_vector)

        moufang_vector = self.compute_moufang_distortion(R1, R2, R3)
        moufang_norm = KBNSummationKernel.norm(moufang_vector)

        left_mf, right_mf, middle_mf = CayleyDicksonAlgebra128.moufang_identities_defect(
            R1.vector_rep, R2.vector_rep, R3.vector_rep
        )

        eneagonal_vector = self.compute_eneagonal_associator(
            R1, R2, R3, R4, R5, R6, R7, R8, R9
        )
        eneagonal_norm = KBNSummationKernel.norm(eneagonal_vector)

        alt_vector = CayleyDicksonAlgebra128.alternative_strain(
            R1.vector_rep, R2.vector_rep
        )
        alternative_norm = KBNSummationKernel.norm(alt_vector)
        right_alt_vector = CayleyDicksonAlgebra128.right_alternative_strain(
            R1.vector_rep, R2.vector_rep
        )
        right_alternative_norm = KBNSummationKernel.norm(right_alt_vector)

        flexibility_defect = KBNSummationKernel.norm(
            CayleyDicksonAlgebra128.associator(
                R1.vector_rep, R2.vector_rep, R1.vector_rep
            )
        )
        power_associator_norm = KBNSummationKernel.norm(
            CayleyDicksonAlgebra128.associator(
                R1.vector_rep, R1.vector_rep, R1.vector_rep
            )
        )
        stasheff_diameter = self._stasheff_diameter(R1, R2, R3, R4)

        log_norm_moufang = _log_norm(moufang_norm)
        log_norm_eneagonal = _log_norm(eneagonal_norm)

        log_den_moufang = (
            2.0 * _log_norm(max(R1.norm, _WILKINSON_FLOOR))
            + _log_norm(max(R2.norm, _WILKINSON_FLOOR))
            + _log_norm(max(R3.norm, _WILKINSON_FLOOR))
        )
        log_den_eneagonal = sum(
            _log_norm(max(state.norm, _WILKINSON_FLOOR)) for state in states
        )

        moufang_relative_norm = _safe_exp(log_norm_moufang - log_den_moufang)
        eneagonal_relative_norm = _safe_exp(log_norm_eneagonal - log_den_eneagonal)
        moufang_frustration_index = _safe_exp(
            log_norm_moufang - max(0.0, log_den_moufang)
        )
        frustration_index = _safe_exp(
            log_norm_eneagonal - max(0.0, log_den_eneagonal)
        )

        if moufang_threshold is None:
            moufang_threshold_abs = float(self._thresholds.moufang_absolute)
        else:
            moufang_threshold_abs = float(moufang_threshold)

        if eneagonal_threshold is None:
            eneagonal_threshold_abs = float(self._thresholds.eneagonal_absolute)
        else:
            eneagonal_threshold_abs = float(eneagonal_threshold)

        if not math.isfinite(moufang_threshold_abs) or moufang_threshold_abs < 0.0:
            raise RoutonEngineError("El umbral de Moufang debe ser finito y no negativo.")
        if not math.isfinite(eneagonal_threshold_abs) or eneagonal_threshold_abs < 0.0:
            raise RoutonEngineError("El umbral eneagonal debe ser finito y no negativo.")

        log_moufang_relative_threshold = (
            math.log(self._thresholds.moufang_relative) + log_den_moufang
        )
        log_eneagonal_relative_threshold = (
            math.log(self._thresholds.eneagonal_relative) + log_den_eneagonal
        )

        moufang_threshold_used = max(
            moufang_threshold_abs,
            _safe_exp(log_moufang_relative_threshold),
        )
        eneagonal_threshold_used = max(
            eneagonal_threshold_abs,
            _safe_exp(log_eneagonal_relative_threshold),
        )

        moufang_abs_ok = moufang_norm <= moufang_threshold_abs
        moufang_rel_ok = log_norm_moufang <= log_moufang_relative_threshold
        is_moufang_stable = math.isfinite(moufang_norm) and (
            moufang_abs_ok or moufang_rel_ok
        )

        eneagonal_abs_ok = eneagonal_norm <= eneagonal_threshold_abs
        eneagonal_rel_ok = log_norm_eneagonal <= log_eneagonal_relative_threshold
        is_eneagonal_norm_stable = math.isfinite(eneagonal_norm) and (
            eneagonal_abs_ok or eneagonal_rel_ok
        )

        is_eneagonal_stable = is_moufang_stable and is_eneagonal_norm_stable

        (
            condition_number,
            simplex_volume,
            connectivity,
            spectral_gap,
            components,
        ) = self._diagnose_nine_way_simplex(states)

        messages = []

        if not math.isfinite(trilateral_norm):
            messages.append("Asociador trilateral no finito.")
        if not math.isfinite(moufang_norm):
            messages.append("A_Moufang no finito: singularidad de Moufang detectada.")
        elif not is_moufang_stable:
            messages.append("A_Moufang fuera del umbral: deformación de Moufang activa.")
        if not math.isfinite(eneagonal_norm):
            messages.append("A_9 no finito: singularidad eneagonal detectada.")
        elif not is_eneagonal_norm_stable:
            messages.append("A_9 fuera del umbral: frustración de calibre eneagonal activa.")
        if stasheff_diameter > eneagonal_threshold_used:
            messages.append("Diámetro de Stasheff A_4 excesivo: homotopía A_∞ no controlada.")
        if condition_number > self._thresholds.condition_number_limit:
            messages.append(
                "El 8-símplex agéntico presenta condicionamiento afín degenerado."
            )
        if connectivity <= _MACHINE_EPS:
            messages.append(
                "Conectividad algebraica casi nula: el grafo de nueve vías está desconectado."
            )
        if components > 1:
            messages.append(
                f"El laplaciano estima {components} componentes conexas en el 8-símplex."
            )
        if is_eneagonal_stable and not messages:
            messages.append("Auditoría eneagonal estable.")

        diagnosis = " | ".join(messages)

        return RoutonEneagonalReport(
            trilateral_associator_norm=trilateral_norm,
            moufang_distortion_norm=moufang_norm,
            eneagonal_associator_norm=eneagonal_norm,
            is_eneagonal_stable=is_eneagonal_stable,
            frustration_index=frustration_index,
            moufang_relative_norm=moufang_relative_norm,
            eneagonal_relative_norm=eneagonal_relative_norm,
            moufang_frustration_index=moufang_frustration_index,
            moufang_threshold_used=moufang_threshold_used,
            eneagonal_threshold_used=eneagonal_threshold_used,
            is_moufang_stable=is_moufang_stable,
            is_eneagonal_norm_stable=is_eneagonal_norm_stable,
            simplex_condition_number=condition_number,
            laplacian_connectivity=connectivity,
            diagnosis=diagnosis,
            left_moufang_defect=left_mf,
            right_moufang_defect=right_mf,
            middle_moufang_defect=middle_mf,
            alternative_strain_norm=alternative_norm,
            right_alternative_strain_norm=right_alternative_norm,
            flexibility_defect=flexibility_defect,
            power_associator_norm=power_associator_norm,
            stasheff_pentagon_diameter=stasheff_diameter,
            simplex_volume=simplex_volume,
            laplacian_spectral_gap=spectral_gap,
            estimated_connected_components=components,
        )

    def decide_from_metrics_report(
        self,
        metrics_report: RoutonMetricsReport,
        r3_vec: Sequence[float],
        r4_vec: Sequence[float],
        r5_vec: Sequence[float],
        r6_vec: Sequence[float],
        r7_vec: Sequence[float],
        r8_vec: Sequence[float],
        r9_vec: Sequence[float],
        moufang_threshold: Optional[float] = None,
        eneagonal_threshold: Optional[float] = None,
    ) -> RoutonDecisionState:
        r"""
        Morfismo terminal de la Fase 2.

        Empaqueta métricas de Hurwitz, auditoría eneagonal y los nueve
        estados del 8-símplex en un `RoutonDecisionState`, objeto
        inicial de `Phase3_RoutonNullConeEvaluator.continue_from_decision_state`.
        """
        R3 = self.build_state(r3_vec)
        R4 = self.build_state(r4_vec)
        R5 = self.build_state(r5_vec)
        R6 = self.build_state(r6_vec)
        R7 = self.build_state(r7_vec)
        R8 = self.build_state(r8_vec)
        R9 = self.build_state(r9_vec)
        states = (
            metrics_report.state_a,
            metrics_report.state_b,
            R3, R4, R5, R6, R7, R8, R9,
        )
        eneagonal_report = self._eneagonal_report_from_states(
            states=states,
            moufang_threshold=moufang_threshold,
            eneagonal_threshold=eneagonal_threshold,
        )
        return RoutonDecisionState(
            metrics_report=metrics_report,
            eneagonal_report=eneagonal_report,
            states=states,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §H. FASE 3 — ACT
#     Objeto inicial = morfismo terminal de la Fase 2 (RoutonDecisionState)
#     Morfismo terminal: execute_eneagonal_audit → RoutonEngineState
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_RoutonNullConeEvaluator(Phase2_EneagonalAssociatorCalculator):
    r"""
    FASE 3 — Act.

    Evaluación espectral de divisores de cero en el Cono Nulo Routónico
    \mathcal{N}(\mathbb{R}\mathrm{ou}) y generación del certificado terminal.

    El método inicial `continue_from_decision_state` es la continuación
    formal de `Phase2.decide_from_metrics_report`.
    El método terminal `execute_eneagonal_audit` cierra el ciclo OODA.
    """

    __slots__ = ()

    def continue_from_decision_state(
        self,
        decision: RoutonDecisionState,
    ) -> RoutonEngineState:
        r"""
        Morfismo de entrada de la Fase 3 (continuación formal de
        `Phase2_EneagonalAssociatorCalculator.decide_from_metrics_report`).

            \mathrm{continue\_from\_decision\_state}:
                \mathbf{RoutonDecisionState}\longrightarrow\mathbf{RoutonEngineState}.
        """
        t_start = time.perf_counter()

        null_report = self.evaluate_null_cone(
            decision.metrics_report.state_a,
            decision.metrics_report.state_b,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        cryptographic_seal = self._compute_cryptographic_seal(
            states=decision.states,
            metrics_report=decision.metrics_report,
            eneagonal_report=decision.eneagonal_report,
            null_report=null_report,
        )

        return RoutonEngineState(
            metrics_report=decision.metrics_report,
            eneagonal_report=decision.eneagonal_report,
            null_report=null_report,
            fpu_execution_time_ms=elapsed_ms,
            cryptographic_seal=cryptographic_seal,
            engine_version=__version__,
        )

    def _sigma_min_left(self, state: RoutonState) -> float:
        r"""
        Distancia espectral de r al esquema de divisores de cero:

            \mathrm{dist}(r,\mathcal{N})\sim\sigma_{\min}(L_r).

        Se evalúa L_{\hat r} sobre el routón normalizado y se reescala
        por \|r\|_2 (homogeneidad de L).
        """
        if state.is_zero or state.norm <= self._thresholds.zero_norm_threshold:
            return 0.0

        hat = np.asarray(state.vector_rep / state.norm, dtype=np.float64)
        try:
            op = CayleyDicksonAlgebra128.left_multiplication_matrix(hat)
            singular_values = la.svdvals(op, check_finite=True)
        except (la.LinAlgError, RoutonNumericalSingularityError):
            try:
                op = CayleyDicksonAlgebra128.left_multiplication_matrix(hat)
                singular_values = np.linalg.svd(op, compute_uv=False)
            except Exception:
                return math.inf

        if singular_values.size == 0:
            return 0.0
        sigma_hat = float(np.min(singular_values))
        if not math.isfinite(sigma_hat):
            return math.inf
        return _clip_nonnegative(state.norm * sigma_hat)

    def evaluate_null_cone(
        self,
        R1_state: RoutonState,
        R2_state: RoutonState,
    ) -> NullConeReport:
        r"""
        Mide la fricción exergética de incursión en el cono de divisores de cero:

            \chi_{\mathrm{routon\_null}}
            =\bigl|\,\|R_1\cdot R_2\|_{\mathbb{R}\mathrm{ou}}
              -\|R_1\|_{\mathbb{R}\mathrm{ou}}\|R_2\|_{\mathbb{R}\mathrm{ou}}\bigr|,

            d_{\mathcal{N}}
            =\max\bigl(0,1-\|R_1 R_2\|/(\|R_1\|\|R_2\|)\bigr).

        Complemento espectral: \sigma_{\min}(L_{R_1}),\ \sigma_{\min}(L_{R_2})
        y norma del conmutador \|[R_1,R_2]\|=\|R_1 R_2-R_2 R_1\|.
        """
        product_vector = CayleyDicksonAlgebra128.multiply_128(
            R1_state.vector_rep,
            R2_state.vector_rep,
        )
        product_norm = KBNSummationKernel.norm(product_vector)

        reverse_vector = CayleyDicksonAlgebra128.multiply_128(
            R2_state.vector_rep,
            R1_state.vector_rep,
        )
        commutator_norm = KBNSummationKernel.norm(product_vector - reverse_vector)

        log_expected = _log_norm(R1_state.norm) + _log_norm(R2_state.norm)
        expected_norm = _safe_exp(log_expected)

        if math.isfinite(product_norm) and math.isfinite(expected_norm):
            absolute_friction = abs(product_norm - expected_norm)
        else:
            absolute_friction = math.inf

        scale = (
            max(_WILKINSON_FLOOR, expected_norm)
            if math.isfinite(expected_norm)
            else _WILKINSON_FLOOR
        )
        relative_defect = (
            absolute_friction / scale if math.isfinite(absolute_friction) else math.inf
        )

        trivial_null = R1_state.is_zero or R2_state.is_zero

        if (
            math.isfinite(expected_norm)
            and math.isfinite(product_norm)
            and expected_norm > self._thresholds.zero_norm_threshold
        ):
            null_depth = max(0.0, 1.0 - (product_norm / expected_norm))
        elif (
            not trivial_null
            and math.isfinite(product_norm)
            and product_norm <= self._thresholds.null_absolute
        ):
            null_depth = 1.0
        else:
            null_depth = 0.0

        if math.isfinite(expected_norm):
            penetration_bound = (
                self._thresholds.null_absolute
                + self._thresholds.null_relative * expected_norm
            )
            near_zero_product = product_norm <= penetration_bound
        else:
            near_zero_product = False

        sigma_min_r1 = self._sigma_min_left(R1_state)
        sigma_min_r2 = self._sigma_min_left(R2_state)

        is_null_cone_penetrated = (
            not trivial_null
            and math.isfinite(product_norm)
            and (
                near_zero_product
                or null_depth >= self._thresholds.null_depth_threshold
            )
        )

        if is_null_cone_penetrated:
            logger.debug(
                "Fase 3: penetración no trivial del Cono Nulo Routónico. "
                "product=%.6e expected=%.6e depth=%.6e σmin(L1)=%.6e σmin(L2)=%.6e",
                product_norm,
                expected_norm,
                null_depth,
                sigma_min_r1,
                sigma_min_r2,
            )

        return NullConeReport(
            product_norm=product_norm,
            expected_norm=expected_norm,
            absolute_friction=absolute_friction,
            relative_defect=relative_defect,
            null_depth=null_depth,
            sigma_min_left_r1=sigma_min_r1,
            sigma_min_left_r2=sigma_min_r2,
            commutator_norm=commutator_norm,
            is_trivial_null=trivial_null,
            is_null_cone_penetrated=is_null_cone_penetrated,
        )

    def evaluate_null_cone_friction(
        self,
        R1_state: RoutonState,
        R2_state: RoutonState,
    ) -> float:
        """Compatibilidad con la versión 3.x: retorna únicamente la fricción."""
        return self.evaluate_null_cone(R1_state, R2_state).absolute_friction

    def _compute_cryptographic_seal(
        self,
        states: Sequence[RoutonState],
        metrics_report: RoutonMetricsReport,
        eneagonal_report: RoutonEneagonalReport,
        null_report: NullConeReport,
    ) -> str:
        """
        Sello SHA-256 write-protected de la sesión.

        Incorpora versión del motor, umbrales empaquetados en IEEE-754
        little-endian, los nueve vectores soberanos y los escalares
        críticos de las tres fases. Toda mutación posterior invalida
        el certificado.
        """
        hasher = hashlib.sha256()
        hasher.update(__version__.encode("utf-8"))

        for name in RoutonThresholds.__dataclass_fields__:
            hasher.update(name.encode("utf-8"))
            hasher.update(_canonical_float_bytes(float(getattr(self._thresholds, name))))

        for state in states:
            hasher.update(np.asarray(state.vector_rep, dtype="<f8").tobytes(order="C"))
            hasher.update(state.sha256_hash.encode("ascii"))

        scalar_payload = (
            metrics_report.product_norm,
            metrics_report.expected_norm,
            metrics_report.hurwitz_absolute_error,
            metrics_report.hurwitz_relative_error,
            metrics_report.hurwitz_signed_defect,
            metrics_report.composition_ratio,
            float(metrics_report.is_hurwitz_stable),
            float(metrics_report.is_banach_submultiplicative),
            eneagonal_report.trilateral_associator_norm,
            eneagonal_report.moufang_distortion_norm,
            eneagonal_report.eneagonal_associator_norm,
            eneagonal_report.moufang_relative_norm,
            eneagonal_report.eneagonal_relative_norm,
            eneagonal_report.moufang_frustration_index,
            eneagonal_report.frustration_index,
            eneagonal_report.moufang_threshold_used,
            eneagonal_report.eneagonal_threshold_used,
            float(eneagonal_report.is_moufang_stable),
            float(eneagonal_report.is_eneagonal_norm_stable),
            float(eneagonal_report.is_eneagonal_stable),
            eneagonal_report.simplex_condition_number,
            eneagonal_report.laplacian_connectivity,
            eneagonal_report.left_moufang_defect,
            eneagonal_report.right_moufang_defect,
            eneagonal_report.middle_moufang_defect,
            eneagonal_report.alternative_strain_norm,
            eneagonal_report.right_alternative_strain_norm,
            eneagonal_report.flexibility_defect,
            eneagonal_report.power_associator_norm,
            eneagonal_report.stasheff_pentagon_diameter,
            eneagonal_report.simplex_volume,
            eneagonal_report.laplacian_spectral_gap,
            float(eneagonal_report.estimated_connected_components),
            null_report.product_norm,
            null_report.expected_norm,
            null_report.absolute_friction,
            null_report.relative_defect,
            null_report.null_depth,
            null_report.sigma_min_left_r1,
            null_report.sigma_min_left_r2,
            null_report.commutator_norm,
            float(null_report.is_trivial_null),
            float(null_report.is_null_cone_penetrated),
        )
        for scalar in scalar_payload:
            hasher.update(_canonical_float_bytes(float(scalar)))

        return hasher.hexdigest()

    def execute_eneagonal_audit(
        self,
        r1_vec: Sequence[float],
        r2_vec: Sequence[float],
        r3_vec: Sequence[float],
        r4_vec: Sequence[float],
        r5_vec: Sequence[float],
        r6_vec: Sequence[float],
        r7_vec: Sequence[float],
        r8_vec: Sequence[float],
        r9_vec: Sequence[float],
        eneagonal_threshold: Optional[float] = None,
        moufang_threshold: Optional[float] = None,
    ) -> RoutonEngineState:
        r"""
        Morfismo terminal de la Fase 3 y del motor.

        Orquesta el ciclo ciego completo en la FPU para auditar un
        megaconsorcio de 9 vías como composición functorial estricta:

            \mathrm{Observe}(R_1,R_2)
                \xrightarrow{\text{Fase 1}} \mathbf{Metrics}
            \xrightarrow{\text{Fase 2}} \mathbf{Decision}
            \xrightarrow{\text{Fase 3}} \mathbf{Certificate}.

        Flujo:
            1. observe_metrics(R1, R2) → RoutonMetricsReport.
            2. decide_from_metrics_report(metrics, R3..R9)
               → RoutonDecisionState.
            3. continue_from_decision_state(decision)
               → RoutonEngineState.
        """
        t_start = time.perf_counter()

        metrics_report = self.observe_metrics(r1_vec, r2_vec)

        decision = self.decide_from_metrics_report(
            metrics_report=metrics_report,
            r3_vec=r3_vec,
            r4_vec=r4_vec,
            r5_vec=r5_vec,
            r6_vec=r6_vec,
            r7_vec=r7_vec,
            r8_vec=r8_vec,
            r9_vec=r9_vec,
            moufang_threshold=moufang_threshold,
            eneagonal_threshold=eneagonal_threshold,
        )

        engine_state = self.continue_from_decision_state(decision)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        engine_state = replace(engine_state, fpu_execution_time_ms=elapsed_ms)

        logger.debug(
            "Auditoría eneagonal v%s completada en %.6f ms. "
            "Hurwitz=%s Banach=%s Eneagonal=%s NullCone=%s",
            __version__,
            elapsed_ms,
            metrics_report.is_hurwitz_stable,
            metrics_report.is_banach_submultiplicative,
            decision.eneagonal_report.is_eneagonal_stable,
            engine_state.null_report.is_null_cone_penetrated,
        )
        return engine_state


# ═══════════════════════════════════════════════════════════════════════════════
# §I. FACHADA SOBERANA DEL MOTOR
# ═══════════════════════════════════════════════════════════════════════════════
# La fachada pública es la propia Fase 3, preservando la arquitectura de
# tres fases anidadas sin introducir una cuarta capa ontológica.
# Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act contiene Decide contiene Observe.
RoutonDependencyEngine = Phase3_RoutonNullConeEvaluator