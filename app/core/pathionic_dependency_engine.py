# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pathionic Dependency Engine (Motor de Calibre Pationiónico 32D)     ║
║ Ruta   : app/core/pathionic_dependency_engine.py                             ║
║ Versión: 1.1.0-Doctoral-32D-CayleyDickson-Pentagonal-KBN-FPU-Secure-Nested3  ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:                                  ║
║ Este módulo implementa el motor de cálculo ciego en la FPU para la variedad  ║
║ de los Pathiones reales \mathbb{P} (32 dimensiones), estructurada mediante   ║
║ la duplicación iterativa de Cayley-Dickson sobre el álgebra de los           ║
║ Sedeniones \mathbb{S} (\mathbb{P} \cong \mathbb{S} \times \mathbb{S}).       ║
║                                                                              ║
║ Opera como un resolvedor de alta fidelidad para interdependencias de 5 vías  ║
║ (4-símplices en el complejo simplicial de la Malla Agéntica), calculando     ║
║ la norma del asociador pentagonal de de Rham A_5, el diámetro del pentágono  ║
║ de Stasheff A_4, la forma cuadrática N(x)=x\,\overline{x}, el defecto de     ║
║ Banach/Hurwitz y la distancia espectral al esquema de divisores de cero      ║
║ \mathcal{N}(\mathbb{P})=\{\,x\in\mathbb{P}:\ker L_x\neq\{0\}\,\}.            ║
║                                                                              ║
║ AXIOMAS (álgebra de Cayley-Dickson, convención interna):                     ║
║   (CD-1) Duplicación:  \mathbb{P}=\mathbb{S}\times\mathbb{S},\ \dim_{\mathbb{R}}=32.║
║   (CD-2) Producto:     (a,b)(c,d)=(ac-\overline{d}\,b,\ da+b\,\overline{c}). ║
║   (CD-3) Conjugación:  \overline{(a,b)}=(\overline{a},-b),\ \overline{\overline{x}}=x.║
║   (CD-4) Anti-homom.:  \overline{xy}=\overline{y}\,\overline{x}.             ║
║   (CD-5) Forma cuadr.: N(x):=x\overline{x}\in\mathbb{R}\,e_0\ \text{en aritmética exacta}.║
║   (H)    Hurwitz:      N(xy)=N(x)N(y) \Leftrightarrow \dim\le 8              ║
║                        (teorema de Hurwitz); en \mathbb{P} el defecto es dato.║
║   (B)    Banach:       \|\cdot\|_2 no es, en general, submultiplicativa en   ║
║                        \dim>8; se audita \rho=\|xy\|/(\|x\|\|y\|).           ║
║   (A3)   Asociador:    [x,y,z]=(xy)z-x(yz).                                  ║
║   (A4)   Stasheff:     diámetro de las 5 asociaciones plenas de 4 factores.  ║
║   (A5)   Pentagonal:   A_5=((((ab)c)d)e)-a(b(c(de))).                        ║
║   (N)    Cono nulo:    xy=0 con x,y\neq 0 \Leftrightarrow \sigma_{\min}(L_x)=0║
║                        para algún testigo y\in\ker L_x.                      ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA FPU):                   ║
║   Fase 1  Observe + Orient : Ingesta 32D, C-D, norma KBN, N(x), Hurwitz.     ║
║           Morfismo terminal : observe_metrics → PathionicMetricsReport.      ║
║   Fase 2  Decide           : A_3, A_5, Stasheff, espectro del 4-símplex.     ║
║           Objeto inicial    : PathionicMetricsReport.                        ║
║           Morfismo terminal : decide_from_metrics_report                     ║
║                               → PathionicDecisionState.                      ║
║   Fase 3  Act              : L_x, cono nulo, sello criptográfico canónico.   ║
║           Objeto inicial    : PathionicDecisionState.                        ║
║           Morfismo terminal : execute_pentagonal_audit                       ║
║                               → PathionicEngineState.                        ║
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
    "1.1.0-Doctoral-32D-CayleyDickson-Pentagonal-KBN-FPU-Secure-Nested3"
)

__all__ = [
    "PathionicDependencyEngine",
    "PathionicState",
    "PathionicMetricsReport",
    "PathionicPentagonalReport",
    "PathionicDecisionState",
    "PathionicEngineState",
    "NullConeReport",
    "PathionicThresholds",
    "CayleyDicksonAlgebra32",
    "KBNSummationKernel",
    "Phase1_PathionicMetricObserver",
    "Phase2_PentagonalAssociatorCalculator",
    "Phase3_PathionicNullConeEvaluator",
    "PathionicEngineError",
    "PathionicDimensionError",
    "PathionicNumericalSingularityError",
    "PathionicCompositionError",
    "PathionicNullConeSingularityError",
]

logger = logging.getLogger("APU.Core.PathionicDependencyEngine")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_PATHION_DIM: Final[int] = 32
_SEDENION_DIM: Final[int] = 16
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_SQRT_DBL_MAX: Final[float] = float(math.sqrt(np.finfo(np.float64).max))
_SCALE_OVERFLOW_TRIGGER: Final[float] = 1e150

if _PATHION_DIM != 2 * _SEDENION_DIM or not (
    _PATHION_DIM > 0 and (_PATHION_DIM & (_PATHION_DIM - 1)) == 0
):
    raise RuntimeError(
        "Inconsistencia dimensional Cayley-Dickson: se exige "
        "dim(P)=32=2·dim(S) y potencia de dos."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §A. JERARQUÍA DE EXCEPCIONES ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════
class PathionicEngineError(Exception):
    r"""Excepción raíz para violaciones algebraicas o de FPU en el motor de Pathiones."""


class PathionicDimensionError(PathionicEngineError):
    r"""Detonada cuando la señal de entrada no es estrictamente de dimensión 32
    (o, en el núcleo CD, cuando la dimensión no es potencia de dos)."""


class PathionicNumericalSingularityError(PathionicEngineError):
    r"""Detonada cuando aparecen NaN, Inf o desbordamientos numéricos no auditables."""


class PathionicCompositionError(PathionicEngineError):
    r"""Detonada cuando la composición de Hurwitz sufre una deriva de Wilkinson inaceptable."""


class PathionicNullConeSingularityError(PathionicEngineError):
    r"""Detonada cuando la trayectoria transaccional colapsa en el Cono Nulo \mathcal{N}(\mathbb{P})."""


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
    """
    Exponencial saturada en el rango de float64.

    Devuelve 0.0 ante underflow e inf ante overflow, preservando la
    trazabilidad metrológica sin detonar excepciones innecesarias.
    """
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
    """
    Serialización canónica little-endian de un escalar float64.

    Unifica ±0.0, etiqueta NaN/±Inf de forma independiente de plataforma
    y empaqueta el resto como IEEE-754 '<d'. Garantiza sello reproducible.
    """
    x = float(value)
    if math.isnan(x):
        return b"\x7fNAN\x00\x00\x00"
    if math.isinf(x):
        return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
    if x == 0.0:
        return struct.pack("<d", 0.0)
    return struct.pack("<d", x)


def _prepare_pathion_vector(value: Sequence[float], name: str) -> np.ndarray:
    """
    Valida, normaliza y congela un vector pationiónico en R^{32}.

    Condiciones:
    - Forma estricta (32,); se rechaza cualquier otra inmersión.
    - Componentes estrictamente finitas (sin NaN/Inf).
    - Copia propia contigua en float64.
    - Bandera write=False para inmutabilidad en RAM.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape != (_PATHION_DIM,):
        raise PathionicDimensionError(
            f"{name} debe ser estrictamente de dimensión {_PATHION_DIM}. "
            f"Obtenido: shape={arr.shape}."
        )

    arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
    if not np.all(np.isfinite(arr)):
        raise PathionicNumericalSingularityError(
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
        raise PathionicNumericalSingularityError(
            "El reescalado logarítmico del producto CD excedió el rango float64."
        )
    return out


def _clip_nonnegative(value: float) -> float:
    """Proyección sobre [0, +∞] con saturación de residuos numéricos negativos."""
    if not math.isfinite(value):
        return value
    return value if value >= 0.0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DEL ESPACIO DE FASE PATIONIÓNICO (32D)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PathionicThresholds:
    r"""
    Umbrales metrológicos inmutables del motor pationiónico.

    Fronteras de:
    - estabilidad de Hurwitz / defecto de Banach,
    - estabilidad pentagonal y diámetro de Stasheff,
    - incursión en el cono nulo,
    - degeneración espectral del 4-símplex agéntico.
    """
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    zero_norm_threshold: float = 1e-14

    hurwitz_absolute: float = 1e-9
    hurwitz_relative: float = 1e-8

    pentagonal_absolute: float = 5.0
    pentagonal_relative: float = 1e-2

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
class PathionicState:
    r"""
    Estado físico hipercomplejo pationiónico de 32 dimensiones en la FPU.

    Bajo el isomorfismo de Cayley-Dickson:
        P = (s_1, s_2) \in \mathbb{S} \times \mathbb{S},
    donde s_1, s_2 son sedeniones en \mathbb{R}^{16}.

    Invariantes auditados:
        \|P\|_2,\quad N(P)=P\overline{P},\quad \mathrm{leak}(N)=\|\mathrm{Im}\,N(P)\|_2.
    """
    vector_rep: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    norm: float
    norm_squared: float
    real_part: float
    imag_norm: float
    quadratic_scalar: float
    quadratic_leakage: float
    is_unitary: bool
    is_zero: bool
    sha256_hash: str


@dataclass(frozen=True, slots=True)
class PathionicMetricsReport:
    r"""
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Norma, forma cuadrática, composición de Hurwitz y defecto de Banach
    del producto pationiónico A·B.
    """
    state_a: PathionicState
    state_b: PathionicState
    product_state: PathionicState

    product_norm: float
    expected_norm: float

    hurwitz_absolute_error: float
    hurwitz_relative_error: float
    hurwitz_signed_defect: float
    composition_ratio: float

    is_hurwitz_stable: bool
    is_banach_submultiplicative: bool


@dataclass(frozen=True, slots=True)
class PathionicPentagonalReport:
    r"""
    Auditoría de la Fase 2: asociadores, Stasheff, espectro y topología
    combinatoria del 4-símplex agéntico.
    """
    trilateral_associator_norm: float
    pentagonal_associator_norm: float
    pentagonal_relative_norm: float
    frustration_index: float
    stasheff_pentagon_diameter: float
    flexibility_defect: float
    alternativity_defect: float

    pentagonal_threshold_used: float
    is_pentagonal_stable: bool

    simplex_condition_number: float
    simplex_volume: float
    laplacian_connectivity: float
    laplacian_spectral_gap: float
    estimated_connected_components: int

    diagnosis: str


@dataclass(frozen=True, slots=True)
class PathionicDecisionState:
    r"""
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Empaqueta el reporte metrológico, el reporte pentagonal y los cinco
    estados del 4-símplex, de modo que Act no reinstancia Observe.
    """
    metrics_report: PathionicMetricsReport
    pentagonal_report: PathionicPentagonalReport
    states: Tuple[
        PathionicState,
        PathionicState,
        PathionicState,
        PathionicState,
        PathionicState,
    ]


@dataclass(frozen=True, slots=True)
class NullConeReport:
    r"""
    Reporte de la Fase 3: fricción, profundidad y caracterización espectral
    de la incursión en \mathcal{N}(\mathbb{P}).
    """
    product_norm: float
    expected_norm: float

    absolute_friction: float
    relative_defect: float
    null_depth: float

    sigma_min_left_p1: float
    sigma_min_left_p2: float
    commutator_norm: float

    is_trivial_null: bool
    is_null_cone_penetrated: bool


@dataclass(frozen=True, slots=True)
class PathionicEngineState:
    r"""Certificado inmutable final del motor pationiónico entregado a los soberanos."""
    metrics_report: PathionicMetricsReport
    pentagonal_report: PathionicPentagonalReport
    null_report: NullConeReport

    fpu_execution_time_ms: float
    cryptographic_seal: str
    engine_version: str = __version__

    @property
    def null_cone_friction(self) -> float:
        """Compatibilidad semántica con la versión 3.x."""
        return self.null_report.absolute_friction

    @property
    def is_null_cone_penetrated(self) -> bool:
        """Compatibilidad semántica con la versión 3.x."""
        return self.null_report.is_null_cone_penetrated

    @property
    def decision_state(self) -> PathionicDecisionState:
        """Reconstrucción del objeto de decisión (Fase 2) a partir del certificado."""
        return PathionicDecisionState(
            metrics_report=self.metrics_report,
            pentagonal_report=self.pentagonal_report,
            states=(
                self.metrics_report.state_a,
                self.metrics_report.state_b,
                self.metrics_report.state_a,
                self.metrics_report.state_a,
                self.metrics_report.state_a,
            ),
        )


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
                raise PathionicNumericalSingularityError(
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
            raise PathionicNumericalSingularityError(
                "La sumación KBN excedió el rango representable de la FPU."
            )
        return result

    @staticmethod
    def norm(values: np.ndarray) -> float:
        r"""
        Norma euclídea robusta con escalado y sumación KBN:

            \|x\|_2 = m \sqrt{\sum_i (x_i/m)^2},\qquad m=\max_i |x_i|.

        El escalado evita overflow/underflow; KBN suprime la deriva de
        Wilkinson; `math.fsum` corrobora la suma de cuadrados.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0

        if not np.all(np.isfinite(arr)):
            raise PathionicNumericalSingularityError(
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
                raise PathionicNumericalSingularityError(
                    "La suma de cuadrados escalada produjo un valor negativo no físico."
                )

        if max_abs >= _SQRT_DBL_MAX and ssq > 1.0:
            raise PathionicNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )

        norm_value = float(max_abs * math.sqrt(ssq))
        if not math.isfinite(norm_value):
            raise PathionicNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )
        return norm_value


# ═══════════════════════════════════════════════════════════════════════════════
# §E. NÚCLEO ALGEBRAICO DE CAYLEY-DICKSON 32D
# ═══════════════════════════════════════════════════════════════════════════════
class CayleyDicksonAlgebra32:
    r"""
    Álgebra de Cayley-Dickson recursiva de-confinada en la FPU.

    Soporta operaciones sobre
        \mathbb{R},\mathbb{C},\mathbb{H},\mathbb{O},\mathbb{S},\mathbb{P}.

    Producto (axioma CD-2):
        (a_1,a_2)(b_1,b_2)
        =(a_1 b_1-\overline{b_2} a_2,\ b_2 a_1+a_2\overline{b_1}).

    El producto de nivel superior se reescala para extinguir overflow
    intermedio; el núcleo recursivo no revalida (la validación es frontera).
    """

    __slots__ = ()

    @staticmethod
    def _validate_power_of_two_vector(value: np.ndarray, name: str) -> np.ndarray:
        """Valida que un vector algebraico tenga dimensión 2^k, rango 1 y sea finito."""
        arr = np.asarray(value, dtype=np.float64)

        if arr.ndim != 1:
            raise PathionicDimensionError(
                f"{name} debe ser un vector algebraico unidimensional. Shape={arr.shape}."
            )
        if not _is_power_of_two(int(arr.size)):
            raise PathionicDimensionError(
                f"{name} debe tener dimensión potencia de dos. Obtenido: {arr.size}."
            )
        if not np.all(np.isfinite(arr)):
            raise PathionicNumericalSingularityError(
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
    def multiply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        Producto bilineal de Cayley-Dickson sobre álgebras de dimensión 2^k.

        Escalado de frontera: se multiplica (a/s_a)(b/s_b) y se restaura
        el factor s_a s_b en espacio logarítmico.
        """
        arr_a = cls._validate_power_of_two_vector(a, "a")
        arr_b = cls._validate_power_of_two_vector(b, "b")

        if arr_a.size != arr_b.size:
            raise PathionicDimensionError(
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
            raise PathionicNumericalSingularityError(
                "El producto de Cayley-Dickson produjo componentes no finitas."
            )
        return product

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
    def pentagonal_associator(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        e: np.ndarray,
    ) -> np.ndarray:
        r"""
        Asociador pentagonal de 5 vías (4-símplex):

            A_5(a,b,c,d,e)=((((ab)c)d)e)-a(b(c(de))).
        """
        p12 = cls.multiply(a, b)
        p123 = cls.multiply(p12, c)
        p1234 = cls.multiply(p123, d)
        left = cls.multiply(p1234, e)

        p45 = cls.multiply(d, e)
        p345 = cls.multiply(c, p45)
        p2345 = cls.multiply(b, p345)
        right = cls.multiply(a, p2345)

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
        (pentágono de Stasheff / 2-celda de A_\infty):

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
        Operador de multiplicación a la izquierda L_p:\mathbb{P}\to\mathbb{P},
        y \mapsto p y, en la base canónica \{e_0,\ldots,e_{31}\}.

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
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PathionicMetricObserver:
    r"""
    FASE 1 — Observe + Orient.

    Ingesta de tensores 32D, descomposición de Cayley-Dickson,
    construcción de estados inmutables, auditoría de N(x) y de la
    composición de Hurwitz en la FPU.

    Categoría fuente: \mathbf{Vec}_{32}\times\mathbf{Vec}_{32}.
    Categoría meta  : \mathbf{Metrics}.

    El método terminal `observe_metrics` es el objeto inicial de la Fase 2.
    """

    __slots__ = ("_thresholds",)

    def __init__(self, thresholds: Optional[PathionicThresholds] = None) -> None:
        self._thresholds: Final[PathionicThresholds] = thresholds or PathionicThresholds()

    @property
    def thresholds(self) -> PathionicThresholds:
        """Umbral metrológico activo del motor."""
        return self._thresholds

    def build_state(self, S: Sequence[float]) -> PathionicState:
        r"""
        Instancia un `PathionicState` inmutable a partir de un vector real en R^{32}.

            S\in\mathbb{R}^{32}\ \mapsto\ P=(s_1,s_2)\in\mathbb{S}\times\mathbb{S}.

        Audita simultáneamente \|P\|_2 y la forma cuadrática N(P)=P\overline{P}.
        """
        vector = _prepare_pathion_vector(S, "PathionicState.vector_rep")

        s1 = np.ascontiguousarray(vector[0:_SEDENION_DIM].copy(), dtype=np.float64)
        s2 = np.ascontiguousarray(
            vector[_SEDENION_DIM:_PATHION_DIM].copy(), dtype=np.float64
        )
        s1.setflags(write=False)
        s2.setflags(write=False)

        norm_value = KBNSummationKernel.norm(vector)
        if not math.isfinite(norm_value):
            raise PathionicNumericalSingularityError(
                "La norma del estado pationiónico no es finita."
            )

        if norm_value >= _SQRT_DBL_MAX:
            norm_squared = math.inf
        else:
            norm_squared = norm_value * norm_value

        real_part = float(vector[0])
        imag_norm = KBNSummationKernel.norm(vector[1:]) if vector.size > 1 else 0.0
        quadratic_scalar, quadratic_leakage = CayleyDicksonAlgebra32.quadratic_form(vector)

        is_zero = norm_value <= self._thresholds.zero_norm_threshold
        is_unitary = _within_tolerance(
            norm_value,
            1.0,
            self._thresholds.absolute_tolerance,
            self._thresholds.relative_tolerance,
        )

        sha_hash = _sha256_of_arrays((vector, s1, s2))

        return PathionicState(
            vector_rep=vector,
            s1=s1,
            s2=s2,
            norm=norm_value,
            norm_squared=norm_squared,
            real_part=real_part,
            imag_norm=imag_norm,
            quadratic_scalar=quadratic_scalar,
            quadratic_leakage=quadratic_leakage,
            is_unitary=is_unitary,
            is_zero=is_zero,
            sha256_hash=sha_hash,
        )

    def observe_metrics(
        self,
        A_vec: Sequence[float],
        B_vec: Sequence[float],
    ) -> PathionicMetricsReport:
        r"""
        Morfismo terminal de la Fase 1.

        Ejecuta el producto pationiónico A·B y verifica la composición de Hurwitz:

            \delta_{\mathrm{composition}}
            =\bigl|\,\|A\cdot B\|_{\mathbb{P}}-\|A\|_{\mathbb{P}}\|B\|_{\mathbb{P}}\bigr|.

        Firma functorial:

            \mathrm{observe\_metrics}:
                \mathbb{R}^{32}\times\mathbb{R}^{32}\longrightarrow
                \mathbf{PathionicMetricsReport}.

        El valor de retorno es el objeto inicial de
        `Phase2_PentagonalAssociatorCalculator.continue_from_metrics_report`.
        """
        state_a = self.build_state(A_vec)
        state_b = self.build_state(B_vec)

        product_vector = CayleyDicksonAlgebra32.multiply(
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
            1.0, product_state.norm_squared if math.isfinite(product_state.norm_squared) else 1.0
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

        return PathionicMetricsReport(
            state_a=state_a,
            state_b=state_b,
            product_state=product_state,
            product_norm=product_norm,
            expected_norm=expected_norm,
            hurwitz_absolute_error=absolute_error,
            hurwitz_relative_error=relative_error,
            hurwitz_signed_defect=signed_defect,
            composition_ratio=composition_ratio,
            is_hurwitz_stable=is_hurwitz_stable,
            is_banach_submultiplicative=is_banach_submultiplicative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §G. FASE 2 — DECIDE
#     Objeto inicial = morfismo terminal de la Fase 1 (PathionicMetricsReport)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_PentagonalAssociatorCalculator(Phase1_PathionicMetricObserver):
    r"""
    FASE 2 — Decide.

    Consume `PathionicMetricsReport` (terminal de Fase 1) y extiende la
    auditoría al 4-símplex de cinco vías: asociador trilateral, asociador
    pentagonal A_5, pentágono de Stasheff, defectos de flexibilidad y
    alternatividad, volumen afín y espectro del laplaciano de similitud.

    El método inicial `continue_from_metrics_report` es la continuación
    formal de `Phase1.observe_metrics`.
    El método terminal `decide_from_metrics_report` produce el objeto
    inicial de la Fase 3 (`PathionicDecisionState`).
    """

    __slots__ = ()

    def continue_from_metrics_report(
        self,
        metrics_report: PathionicMetricsReport,
        p3_vec: Sequence[float],
        p4_vec: Sequence[float],
        p5_vec: Sequence[float],
        pentagonal_threshold: Optional[float] = None,
    ) -> PathionicPentagonalReport:
        r"""
        Morfismo de entrada de la Fase 2 (continuación formal de
        `Phase1_PathionicMetricObserver.observe_metrics`).

            \mathrm{continue\_from\_metrics\_report}:
                \mathbf{PathionicMetricsReport}\times(\mathbb{R}^{32})^{3}
                \longrightarrow\mathbf{PathionicPentagonalReport}.
        """
        return self.decide_from_metrics_report(
            metrics_report=metrics_report,
            p3_vec=p3_vec,
            p4_vec=p4_vec,
            p5_vec=p5_vec,
            pentagonal_threshold=pentagonal_threshold,
        ).pentagonal_report

    def calculate_pentagonal_frustration(
        self,
        p1_vec: Sequence[float],
        p2_vec: Sequence[float],
        p3_vec: Sequence[float],
        p4_vec: Sequence[float],
        p5_vec: Sequence[float],
        pentagonal_threshold: Optional[float] = None,
    ) -> PathionicPentagonalReport:
        r"""Orquesta la frustración de calibre de 5 vías sin precomputar la Fase 1."""
        states = tuple(
            self.build_state(v)
            for v in (p1_vec, p2_vec, p3_vec, p4_vec, p5_vec)
        )
        return self._pentagonal_report_from_states(
            states=states,
            pentagonal_threshold=pentagonal_threshold,
        )

    def compute_trilateral_associator(
        self,
        P1: PathionicState,
        P2: PathionicState,
        P3: PathionicState,
    ) -> np.ndarray:
        r"""Asociador de 3 vías [P_1,P_2,P_3]=(P_1 P_2)P_3-P_1(P_2 P_3)."""
        return CayleyDicksonAlgebra32.associator(
            P1.vector_rep,
            P2.vector_rep,
            P3.vector_rep,
        )

    def compute_pentagonal_associator(
        self,
        P1: PathionicState,
        P2: PathionicState,
        P3: PathionicState,
        P4: PathionicState,
        P5: PathionicState,
    ) -> np.ndarray:
        r"""Asociador pentagonal A_5=((((P_1 P_2)P_3)P_4)P_5)-P_1(P_2(P_3(P_4 P_5)))."""
        return CayleyDicksonAlgebra32.pentagonal_associator(
            P1.vector_rep,
            P2.vector_rep,
            P3.vector_rep,
            P4.vector_rep,
            P5.vector_rep,
        )

    def _stasheff_diameter(
        self,
        P1: PathionicState,
        P2: PathionicState,
        P3: PathionicState,
        P4: PathionicState,
    ) -> float:
        """Diámetro euclídeo del pentágono de Stasheff sobre cuatro pathiones."""
        associations = CayleyDicksonAlgebra32.stasheff_pentagon_associations(
            P1.vector_rep,
            P2.vector_rep,
            P3.vector_rep,
            P4.vector_rep,
        )
        diameter = 0.0
        for i in range(len(associations)):
            for j in range(i + 1, len(associations)):
                delta = associations[i] - associations[j]
                diameter = max(diameter, KBNSummationKernel.norm(delta))
        return diameter

    def _diagnose_five_way_simplex(
        self,
        states: Sequence[PathionicState],
    ) -> Tuple[float, float, float, float, int]:
        r"""
        Diagnóstico espectral, volumétrico y de conectividad del 4-símplex.

        Devuelve:
            1. Número de condición afín (SVD de la nube centrada por KBN).
            2. Volumen 4-dimensional (Gram / 4!).
            3. Conectividad algebraica (valor de Fiedler \lambda_2).
            4. Hueco espectral \lambda_2-\lambda_1 del laplaciano.
            5. Multiplicidad numérica de \lambda=0 (componentes conexas).
        """
        if len(states) != 5:
            raise PathionicEngineError(
                "El diagnóstico del 4-símplex requiere exactamente cinco estados pationiónicos."
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
            simplex_volume = _safe_exp(0.5 * float(logabs) - math.log(24.0))

        norms = np.array([state.norm for state in states], dtype=np.float64)
        valid = norms > self._thresholds.zero_norm_threshold

        if not np.any(valid):
            return condition_number, simplex_volume, 0.0, 0.0, 5

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

    def _pentagonal_report_from_states(
        self,
        states: Sequence[PathionicState],
        pentagonal_threshold: Optional[float],
    ) -> PathionicPentagonalReport:
        """Orquesta el cálculo pentagonal completo a partir de cinco estados ya construidos."""
        if len(states) != 5:
            raise PathionicEngineError(
                "La auditoría pentagonal requiere exactamente cinco estados pationiónicos."
            )

        P1, P2, P3, P4, P5 = states

        assoc_3_vector = self.compute_trilateral_associator(P1, P2, P3)
        trilateral_norm = KBNSummationKernel.norm(assoc_3_vector)

        assoc_5_vector = self.compute_pentagonal_associator(P1, P2, P3, P4, P5)
        pentagonal_norm = KBNSummationKernel.norm(assoc_5_vector)

        stasheff_diameter = self._stasheff_diameter(P1, P2, P3, P4)

        flexibility_defect = KBNSummationKernel.norm(
            CayleyDicksonAlgebra32.associator(P1.vector_rep, P2.vector_rep, P1.vector_rep)
        )
        alt_left = KBNSummationKernel.norm(
            CayleyDicksonAlgebra32.associator(P1.vector_rep, P1.vector_rep, P2.vector_rep)
        )
        alt_right = KBNSummationKernel.norm(
            CayleyDicksonAlgebra32.associator(P1.vector_rep, P2.vector_rep, P2.vector_rep)
        )
        alternativity_defect = max(alt_left, alt_right)

        log_den = sum(_log_norm(max(state.norm, _WILKINSON_FLOOR)) for state in states)
        log_norm5 = _log_norm(pentagonal_norm)
        pentagonal_relative_norm = _safe_exp(log_norm5 - log_den)
        frustration_index = _safe_exp(log_norm5 - max(0.0, log_den))

        if pentagonal_threshold is None:
            threshold_abs = float(self._thresholds.pentagonal_absolute)
        else:
            threshold_abs = float(pentagonal_threshold)

        if not math.isfinite(threshold_abs) or threshold_abs < 0.0:
            raise PathionicEngineError(
                "El umbral pentagonal debe ser finito y no negativo."
            )

        log_relative_threshold = math.log(self._thresholds.pentagonal_relative) + log_den
        relative_threshold_equivalent = _safe_exp(log_relative_threshold)
        threshold_used = max(threshold_abs, relative_threshold_equivalent)

        absolute_ok = pentagonal_norm <= threshold_abs
        relative_ok = log_norm5 <= log_relative_threshold
        is_pentagonal_stable = math.isfinite(pentagonal_norm) and (
            absolute_ok or relative_ok
        )

        (
            condition_number,
            simplex_volume,
            connectivity,
            spectral_gap,
            components,
        ) = self._diagnose_five_way_simplex(states)

        messages = []
        if not math.isfinite(pentagonal_norm):
            messages.append("A_5 no finito: singularidad algebraico-numérica detectada.")
        elif is_pentagonal_stable:
            messages.append("A_5 dentro del umbral de estabilidad pentagonal.")
        else:
            messages.append("A_5 fuera del umbral: frustración de calibre pentagonal activa.")

        if stasheff_diameter > threshold_used:
            messages.append("Diámetro de Stasheff A_4 excesivo: homotopía A_∞ no controlada.")

        if condition_number > self._thresholds.condition_number_limit:
            messages.append("El 4-símplex agéntico presenta condicionamiento afín degenerado.")

        if connectivity <= _MACHINE_EPS:
            messages.append(
                "Conectividad algebraica casi nula: el grafo de cinco vías está desconectado."
            )

        if components > 1:
            messages.append(
                f"El laplaciano estima {components} componentes conexas en el 4-símplex."
            )

        diagnosis = " | ".join(messages) if messages else "Auditoría pentagonal estable."

        return PathionicPentagonalReport(
            trilateral_associator_norm=trilateral_norm,
            pentagonal_associator_norm=pentagonal_norm,
            pentagonal_relative_norm=pentagonal_relative_norm,
            frustration_index=frustration_index,
            stasheff_pentagon_diameter=stasheff_diameter,
            flexibility_defect=flexibility_defect,
            alternativity_defect=alternativity_defect,
            pentagonal_threshold_used=threshold_used,
            is_pentagonal_stable=is_pentagonal_stable,
            simplex_condition_number=condition_number,
            simplex_volume=simplex_volume,
            laplacian_connectivity=connectivity,
            laplacian_spectral_gap=spectral_gap,
            estimated_connected_components=components,
            diagnosis=diagnosis,
        )

    def decide_from_metrics_report(
        self,
        metrics_report: PathionicMetricsReport,
        p3_vec: Sequence[float],
        p4_vec: Sequence[float],
        p5_vec: Sequence[float],
        pentagonal_threshold: Optional[float] = None,
    ) -> PathionicDecisionState:
        r"""
        Morfismo terminal de la Fase 2.

        Empaqueta métricas de Hurwitz, auditoría pentagonal y los cinco
        estados del 4-símplex en un `PathionicDecisionState`, objeto
        inicial de `Phase3_PathionicNullConeEvaluator.continue_from_decision_state`.
        """
        P3 = self.build_state(p3_vec)
        P4 = self.build_state(p4_vec)
        P5 = self.build_state(p5_vec)
        states = (
            metrics_report.state_a,
            metrics_report.state_b,
            P3,
            P4,
            P5,
        )
        pentagonal_report = self._pentagonal_report_from_states(
            states=states,
            pentagonal_threshold=pentagonal_threshold,
        )
        return PathionicDecisionState(
            metrics_report=metrics_report,
            pentagonal_report=pentagonal_report,
            states=states,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §H. FASE 3 — ACT
#     Objeto inicial = morfismo terminal de la Fase 2 (PathionicDecisionState)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_PathionicNullConeEvaluator(Phase2_PentagonalAssociatorCalculator):
    r"""
    FASE 3 — Act.

    Evaluación espectral de divisores de cero en el Cono Nulo Pationiónico
    \mathcal{N}(\mathbb{P}) y generación del certificado terminal de-confinado.

    El método inicial `continue_from_decision_state` es la continuación
    formal de `Phase2.decide_from_metrics_report`.
    El método terminal `execute_pentagonal_audit` cierra el ciclo OODA.
    """

    __slots__ = ()

    def continue_from_decision_state(
        self,
        decision: PathionicDecisionState,
    ) -> PathionicEngineState:
        r"""
        Morfismo de entrada de la Fase 3 (continuación formal de
        `Phase2_PentagonalAssociatorCalculator.decide_from_metrics_report`).

            \mathrm{continue\_from\_decision\_state}:
                \mathbf{PathionicDecisionState}\longrightarrow\mathbf{PathionicEngineState}.
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
            pentagonal_report=decision.pentagonal_report,
            null_report=null_report,
        )

        return PathionicEngineState(
            metrics_report=decision.metrics_report,
            pentagonal_report=decision.pentagonal_report,
            null_report=null_report,
            fpu_execution_time_ms=elapsed_ms,
            cryptographic_seal=cryptographic_seal,
            engine_version=__version__,
        )

    def _sigma_min_left(self, state: PathionicState) -> float:
        r"""
        Distancia espectral de p al esquema de divisores de cero:

            \mathrm{dist}(p,\mathcal{N})\sim\sigma_{\min}(L_p).

        Se evalúa L_{\hat p} sobre el pathión normalizado y se reescala
        por \|p\|_2 (homogeneidad de L).
        """
        if state.is_zero or state.norm <= self._thresholds.zero_norm_threshold:
            return 0.0

        hat = np.asarray(state.vector_rep / state.norm, dtype=np.float64)
        try:
            op = CayleyDicksonAlgebra32.left_multiplication_matrix(hat)
            singular_values = la.svdvals(op, check_finite=True)
        except (la.LinAlgError, PathionicNumericalSingularityError):
            try:
                op = CayleyDicksonAlgebra32.left_multiplication_matrix(hat)
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
        P1_state: PathionicState,
        P2_state: PathionicState,
    ) -> NullConeReport:
        r"""
        Mide la fricción exergética de incursión en el cono de divisores de cero:

            \chi_{\mathrm{pathion\_null}}
            =\bigl|\,\|P_1\cdot P_2\|_{\mathbb{P}}
              -\|P_1\|_{\mathbb{P}}\|P_2\|_{\mathbb{P}}\bigr|,

            d_{\mathcal{N}}
            =\max\bigl(0,1-\|P_1 P_2\|/(\|P_1\|\|P_2\|)\bigr).

        Complemento espectral: \sigma_{\min}(L_{P_1}),\ \sigma_{\min}(L_{P_2})
        y norma del conmutador \|[P_1,P_2]\|=\|P_1 P_2-P_2 P_1\|.
        """
        product_vector = CayleyDicksonAlgebra32.multiply(
            P1_state.vector_rep,
            P2_state.vector_rep,
        )
        product_norm = KBNSummationKernel.norm(product_vector)

        reverse_vector = CayleyDicksonAlgebra32.multiply(
            P2_state.vector_rep,
            P1_state.vector_rep,
        )
        commutator_norm = KBNSummationKernel.norm(product_vector - reverse_vector)

        log_expected = _log_norm(P1_state.norm) + _log_norm(P2_state.norm)
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

        trivial_null = P1_state.is_zero or P2_state.is_zero

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

        sigma_min_p1 = self._sigma_min_left(P1_state)
        sigma_min_p2 = self._sigma_min_left(P2_state)

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
                "Fase 3: penetración no trivial del Cono Nulo. "
                "product=%.6e expected=%.6e depth=%.6e σmin(L1)=%.6e σmin(L2)=%.6e",
                product_norm,
                expected_norm,
                null_depth,
                sigma_min_p1,
                sigma_min_p2,
            )

        return NullConeReport(
            product_norm=product_norm,
            expected_norm=expected_norm,
            absolute_friction=absolute_friction,
            relative_defect=relative_defect,
            null_depth=null_depth,
            sigma_min_left_p1=sigma_min_p1,
            sigma_min_left_p2=sigma_min_p2,
            commutator_norm=commutator_norm,
            is_trivial_null=trivial_null,
            is_null_cone_penetrated=is_null_cone_penetrated,
        )

    def _compute_cryptographic_seal(
        self,
        states: Sequence[PathionicState],
        metrics_report: PathionicMetricsReport,
        pentagonal_report: PathionicPentagonalReport,
        null_report: NullConeReport,
    ) -> str:
        """
        Sello SHA-256 write-protected de la sesión.

        Incorpora versión del motor, umbrales empaquetados en IEEE-754
        little-endian, los cinco vectores soberanos y los escalares
        críticos de las tres fases. Toda mutación posterior invalida
        el certificado.
        """
        hasher = hashlib.sha256()
        hasher.update(__version__.encode("utf-8"))

        for name in PathionicThresholds.__dataclass_fields__:
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
            pentagonal_report.trilateral_associator_norm,
            pentagonal_report.pentagonal_associator_norm,
            pentagonal_report.pentagonal_relative_norm,
            pentagonal_report.frustration_index,
            pentagonal_report.stasheff_pentagon_diameter,
            pentagonal_report.flexibility_defect,
            pentagonal_report.alternativity_defect,
            pentagonal_report.pentagonal_threshold_used,
            float(pentagonal_report.is_pentagonal_stable),
            pentagonal_report.simplex_condition_number,
            pentagonal_report.simplex_volume,
            pentagonal_report.laplacian_connectivity,
            pentagonal_report.laplacian_spectral_gap,
            float(pentagonal_report.estimated_connected_components),
            null_report.product_norm,
            null_report.expected_norm,
            null_report.absolute_friction,
            null_report.relative_defect,
            null_report.null_depth,
            null_report.sigma_min_left_p1,
            null_report.sigma_min_left_p2,
            null_report.commutator_norm,
            float(null_report.is_trivial_null),
            float(null_report.is_null_cone_penetrated),
        )
        for scalar in scalar_payload:
            hasher.update(_canonical_float_bytes(float(scalar)))

        return hasher.hexdigest()

    def execute_pentagonal_audit(
        self,
        contractor_P1: Sequence[float],
        subcontractor_P2: Sequence[float],
        supplier_P3: Sequence[float],
        interventor_P4: Sequence[float],
        entity_P5: Sequence[float],
        pentagonal_threshold: Optional[float] = None,
    ) -> PathionicEngineState:
        r"""
        Morfismo terminal de la Fase 3 y del motor.

        Orquesta el ciclo ciego completo en la FPU para auditar un
        megaconsorcio de 5 vías como composición functorial estricta:

            \mathrm{Observe}(P_1,P_2)
                \xrightarrow{\text{Fase 1}} \mathbf{Metrics}
            \xrightarrow{\text{Fase 2}} \mathbf{Decision}
            \xrightarrow{\text{Fase 3}} \mathbf{Certificate}.

        Flujo:
            1. observe_metrics(P1, P2) → PathionicMetricsReport.
            2. decide_from_metrics_report(metrics, P3, P4, P5)
               → PathionicDecisionState.
            3. continue_from_decision_state(decision)
               → PathionicEngineState.
        """
        t_start = time.perf_counter()

        metrics_report = self.observe_metrics(contractor_P1, subcontractor_P2)

        decision = self.decide_from_metrics_report(
            metrics_report=metrics_report,
            p3_vec=supplier_P3,
            p4_vec=interventor_P4,
            p5_vec=entity_P5,
            pentagonal_threshold=pentagonal_threshold,
        )

        engine_state = self.continue_from_decision_state(decision)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        engine_state = replace(engine_state, fpu_execution_time_ms=elapsed_ms)

        logger.debug(
            "Auditoría pentagonal v%s completada en %.6f ms. "
            "Hurwitz=%s Banach=%s Pentagonal=%s NullCone=%s",
            __version__,
            elapsed_ms,
            metrics_report.is_hurwitz_stable,
            metrics_report.is_banach_submultiplicative,
            decision.pentagonal_report.is_pentagonal_stable,
            engine_state.null_report.is_null_cone_penetrated,
        )
        return engine_state


# ═══════════════════════════════════════════════════════════════════════════════
# §I. FACHADA SOBERANA DEL MOTOR
# ═══════════════════════════════════════════════════════════════════════════════
# La fachada pública es la propia Fase 3, preservando la arquitectura de
# tres fases anidadas sin introducir una cuarta capa ontológica.
# Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act contiene Decide contiene Observe.
PathionicDependencyEngine = Phase3_PathionicNullConeEvaluator