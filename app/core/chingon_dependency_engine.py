# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Chingon Dependency Engine (Motor de Calibre Chingónico 64D)         ║
║ Ruta   : app/core/chingon_dependency_engine.py                               ║
║ Versión: 4.1.0-Doctoral-64D-CayleyDickson-Heptagonal-Alternative-KBN-Nested3 ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:                                  ║
║ Este módulo implementa el motor de cálculo ciego en la FPU para la variedad  ║
║ de los Chingones reales \mathbb{X} (64 dimensiones / Sexagintaquaternions),  ║
║ estructurada mediante la duplicación iterativa de Cayley-Dickson sobre el    ║
║ álgebra de los Pathiones \mathbb{P}:                                         ║
║                                                                              ║
║     \mathbb{X} \cong \mathbb{P} \times \mathbb{P},\qquad \dim_{\mathbb{R}}=64.║
║                                                                              ║
║ Opera como resolvedor de alta fidelidad para interdependencias de 7 vías     ║
║ (6-símplices en el complejo simplicial de la Malla Agéntica), calculando:    ║
║                                                                              ║
║   1. Norma del asociador heptagonal A_7 y diámetro de Stasheff A_4.          ║
║   2. Tensores de deformación alternativa:                                    ║
║        A_alt^L(X,Y)=[X,X,Y]=(X·X)·Y-X·(X·Y),                                 ║
║        A_alt^R(X,Y)=[X,Y,Y]=(X·Y)·Y-X·(Y·Y).                                 ║
║   3. Defectos de flexibilidad, potencia y Moufang.                           ║
║   4. Forma cuadrática N(X)=X\overline{X} y defecto de Banach/Hurwitz.        ║
║   5. Distancia espectral al esquema de divisores de cero                     ║
║        \mathcal{N}(\mathbb{X})=\{\,x\in\mathbb{X}:\ker L_x\neq\{0\}\,\}.     ║
║                                                                              ║
║ AXIOMAS (álgebra de Cayley-Dickson, convención interna):                     ║
║   (CD-1) Duplicación:  \mathbb{X}=\mathbb{P}\times\mathbb{P}.                ║
║   (CD-2) Producto:     (a,b)(c,d)=(ac-\overline{d}\,b,\ da+b\,\overline{c}). ║
║   (CD-3) Conjugación:  \overline{(a,b)}=(\overline{a},-b),\ \overline{\overline{x}}=x.║
║   (CD-4) Anti-homom.:  \overline{xy}=\overline{y}\,\overline{x}.             ║
║   (CD-5) Forma cuadr.: N(x):=x\overline{x}\in\mathbb{R}\,e_0\ \text{en aritmética exacta}.║
║   (H)    Hurwitz:      N(xy)=N(x)N(y) \Leftrightarrow \dim\le 8.             ║
║   (B)    Banach:       \|\cdot\|_2 no es, en general, submultiplicativa en   ║
║                        \dim>8; se audita \rho=\|xy\|/(\|x\|\|y\|).           ║
║   (A3)   Asociador:    [x,y,z]=(xy)z-x(yz).                                  ║
║   (Alt)  Alternatividad: [x,x,y]=[x,y,y]=0  (falla en \dim\ge 16).           ║
║   (F)    Flexibilidad: [x,y,x]=0.                                            ║
║   (P)    Potencia:     [x,x,x]=0.                                            ║
║   (Mf)   Moufang:      identidades de octoniones; defectos en \mathbb{X}.    ║
║   (A4)   Stasheff:     diámetro de las 5 asociaciones plenas de 4 factores.  ║
║   (A7)   Heptagonal:   A_7=((((((x_1 x_2)x_3)x_4)x_5)x_6)x_7)                ║
║                              -x_1(x_2(x_3(x_4(x_5(x_6 x_7))))).              ║
║   (N)    Cono nulo:    xy=0,\ x,y\neq 0 \Leftrightarrow \sigma_{\min}(L_x)=0.║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA FPU):                   ║
║   Fase 1  Observe+Orient : Ingesta 64D, C-D, norma KBN, N(x), Hurwitz.       ║
║           Morfismo terminal : observe_metrics → ChingonMetricsReport.        ║
║   Fase 2  Decide         : A_3, A_alt^{L/R}, A_7, Stasheff, Moufang, 6-símp. ║
║           Objeto inicial    : ChingonMetricsReport.                          ║
║           Morfismo terminal : decide_from_metrics_report                     ║
║                               → ChingonDecisionState.                        ║
║   Fase 3  Act            : L_x, cono nulo 64D, sello criptográfico canónico. ║
║           Objeto inicial    : ChingonDecisionState.                          ║
║           Morfismo terminal : execute_heptagonal_audit                       ║
║                               → ChingonEngineState.                          ║
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
    "4.1.0-Doctoral-64D-CayleyDickson-Heptagonal-Alternative-KBN-Nested3"
)

__all__ = [
    "ChingonDependencyEngine",
    "ChingonState",
    "ChingonMetricsReport",
    "ChingonHeptagonalReport",
    "ChingonDecisionState",
    "ChingonEngineState",
    "NullConeReport",
    "ChingonThresholds",
    "CayleyDicksonAlgebra64",
    "KBNSummationKernel",
    "Phase1_ChingonMetricObserver",
    "Phase2_HeptagonalFlexibilityCalculator",
    "Phase3_ChingonNullConeEvaluator",
    "ChingonEngineError",
    "ChingonDimensionError",
    "ChingonNumericalSingularityError",
    "ChingonCompositionError",
    "ChingonNullConeSingularityError",
]

logger = logging.getLogger("APU.Core.ChingonDependencyEngine")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CHINGON_DIM: Final[int] = 64
_PATHION_DIM: Final[int] = 32
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_SQRT_DBL_MAX: Final[float] = float(math.sqrt(np.finfo(np.float64).max))
_SCALE_OVERFLOW_TRIGGER: Final[float] = 1e150
_SIMPLEX_FACTORIAL: Final[int] = 720  # 6!

if _CHINGON_DIM != 2 * _PATHION_DIM or not (
    _CHINGON_DIM > 0 and (_CHINGON_DIM & (_CHINGON_DIM - 1)) == 0
):
    raise RuntimeError(
        "Inconsistencia dimensional Cayley-Dickson: se exige "
        "dim(X)=64=2·dim(P) y potencia de dos."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §A. JERARQUÍA DE EXCEPCIONES ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════
class ChingonEngineError(Exception):
    r"""Excepción raíz para violaciones algebraicas o de FPU en el motor de Chingones."""


class ChingonDimensionError(ChingonEngineError):
    r"""Detonada cuando la señal de entrada no es estrictamente de dimensión 64
    (o, en el núcleo CD, cuando la dimensión no es potencia de dos)."""


class ChingonNumericalSingularityError(ChingonEngineError):
    r"""Detonada cuando aparecen NaN, Inf o desbordamientos numéricos no auditables."""


class ChingonCompositionError(ChingonEngineError):
    r"""Detonada cuando la composición de Hurwitz sufre una deriva de Wilkinson inaceptable."""


class ChingonNullConeSingularityError(ChingonEngineError):
    r"""Detonada cuando la trayectoria transaccional colapsa en el Cono Nulo \mathcal{N}(\mathbb{X})."""


# ═══════════════════════════════════════════════════════════════════════════════
# §B. FUNCIONES AUXILIARES DE METROLOGÍA NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════
def _is_power_of_two(n: int) -> bool:
    """Verifica que una dimensión algebraica sea potencia de dos estrictamente positiva."""
    return n > 0 and (n & (n - 1)) == 0


def _log_norm(norm: float) -> bool | float:
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


def _prepare_chingon_vector(value: Sequence[float], name: str) -> np.ndarray:
    """
    Valida, normaliza y congela un vector chingónico en R^{64}.

    Condiciones:
    - Forma estricta (64,); se rechaza cualquier otra inmersión.
    - Componentes estrictamente finitas (sin NaN/Inf).
    - Copia propia contigua en float64.
    - Bandera write=False para inmutabilidad en RAM.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape != (_CHINGON_DIM,):
        raise ChingonDimensionError(
            f"{name} debe ser estrictamente de dimensión {_CHINGON_DIM}. "
            f"Obtenido: shape={arr.shape}."
        )

    arr = np.ascontiguousarray(arr, dtype=np.float64).copy()
    if not np.all(np.isfinite(arr)):
        raise ChingonNumericalSingularityError(
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
        raise ChingonNumericalSingularityError(
            "El reescalado logarítmico del producto CD excedió el rango float64."
        )
    return out


def _clip_nonnegative(value: float) -> float:
    """Proyección sobre [0, +∞] con saturación de residuos numéricos negativos."""
    if not math.isfinite(value):
        return value
    return value if value >= 0.0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DEL ESPACIO DE FASE CHINGÓNICO (64D)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ChingonThresholds:
    r"""
    Umbrales metrológicos inmutables del motor chingónico.

    Fronteras de:
    - estabilidad de Hurwitz / defecto de Banach,
    - deformación alternativa izquierda/derecha,
    - frustración heptagonal y diámetro de Stasheff,
    - incursión en el cono nulo,
    - degeneración espectral del 6-símplex agéntico.
    """
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    zero_norm_threshold: float = 1e-14

    hurwitz_absolute: float = 1e-9
    hurwitz_relative: float = 1e-8

    alternative_absolute: float = 500.0
    alternative_relative: float = 1e-2

    heptagonal_absolute: float = 5_000_000.0
    heptagonal_relative: float = 1e-2

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
class ChingonState:
    r"""
    Estado físico hipercomplejo chingónico de 64 dimensiones en la FPU.

    Bajo el isomorfismo de Cayley-Dickson:
        X = (P_1, P_2) \in \mathbb{P} \times \mathbb{P},
    donde P_1, P_2 son pathiones en \mathbb{R}^{32}.

    Invariantes auditados:
        \|X\|_2,\quad N(X)=X\overline{X},\quad \mathrm{leak}(N)=\|\mathrm{Im}\,N(X)\|_2.
    """
    vector_rep: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
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
class ChingonMetricsReport:
    r"""
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Norma, forma cuadrática, composición de Hurwitz y defecto de Banach
    del producto chingónico A·B.
    """
    state_a: ChingonState
    state_b: ChingonState
    product_state: ChingonState

    product_norm: float
    expected_norm: float

    hurwitz_composition_error: float
    hurwitz_absolute_error: float
    hurwitz_relative_error: float
    hurwitz_signed_defect: float
    composition_ratio: float

    is_hurwitz_stable: bool
    is_banach_submultiplicative: bool


@dataclass(frozen=True, slots=True)
class ChingonHeptagonalReport:
    r"""
    Auditoría de la Fase 2: asociadores, alternatividad, Stasheff, Moufang,
    espectro y topología combinatoria del 6-símplex agéntico.
    """
    trilateral_associator_norm: float
    alternative_strain_norm: float
    heptagonal_associator_norm: float

    alternative_relative_norm: float
    heptagonal_relative_norm: float

    alternative_frustration_index: float
    heptagonal_frustration_index: float

    alternative_threshold_used: float
    heptagonal_threshold_used: float

    is_alternative_stable: bool
    is_heptagonal_norm_stable: bool
    is_heptagonal_stable: bool

    simplex_condition_number: float
    laplacian_connectivity: float

    diagnosis: str

    right_alternative_strain_norm: float = 0.0
    flexibility_defect: float = 0.0
    power_associator_norm: float = 0.0
    moufang_defect: float = 0.0
    stasheff_pentagon_diameter: float = 0.0
    simplex_volume: float = 0.0
    laplacian_spectral_gap: float = 0.0
    estimated_connected_components: int = 1


@dataclass(frozen=True, slots=True)
class ChingonDecisionState:
    r"""
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Empaqueta el reporte metrológico, el reporte heptagonal y los siete
    estados del 6-símplex, de modo que Act no reinstancia Observe.
    """
    metrics_report: ChingonMetricsReport
    heptagonal_report: ChingonHeptagonalReport
    states: Tuple[
        ChingonState,
        ChingonState,
        ChingonState,
        ChingonState,
        ChingonState,
        ChingonState,
        ChingonState,
    ]


@dataclass(frozen=True, slots=True)
class NullConeReport:
    r"""
    Reporte de la Fase 3: fricción, profundidad y caracterización espectral
    de la incursión en \mathcal{N}(\mathbb{X}).
    """
    product_norm: float
    expected_norm: float

    absolute_friction: float
    relative_defect: float
    null_depth: float

    sigma_min_left_x1: float
    sigma_min_left_x2: float
    commutator_norm: float

    is_trivial_null: bool
    is_null_cone_penetrated: bool


@dataclass(frozen=True, slots=True)
class ChingonEngineState:
    r"""Certificado inmutable final del motor chingónico entregado a los soberanos."""
    metrics_report: ChingonMetricsReport
    heptagonal_report: ChingonHeptagonalReport
    null_report: NullConeReport

    fpu_execution_time_ms: float
    cryptographic_seal: str
    engine_version: str = __version__

    @property
    def null_cone_friction(self) -> float:
        """Compatibilidad semántica con la versión 3.x / 4.0."""
        return self.null_report.absolute_friction

    @property
    def is_null_cone_penetrated(self) -> bool:
        """Compatibilidad semántica con la versión 3.x / 4.0."""
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
                raise ChingonNumericalSingularityError(
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
            raise ChingonNumericalSingularityError(
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
            raise ChingonNumericalSingularityError(
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
                raise ChingonNumericalSingularityError(
                    "La suma de cuadrados escalada produjo un valor negativo no físico."
                )

        if max_abs >= _SQRT_DBL_MAX and ssq > 1.0:
            raise ChingonNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )

        norm_value = float(max_abs * math.sqrt(ssq))
        if not math.isfinite(norm_value):
            raise ChingonNumericalSingularityError(
                "La norma KBN excedió el rango representable de la FPU."
            )
        return norm_value


# ═══════════════════════════════════════════════════════════════════════════════
# §E. NÚCLEO ALGEBRAICO DE CAYLEY-DICKSON 64D
# ═══════════════════════════════════════════════════════════════════════════════
class CayleyDicksonAlgebra64:
    r"""
    Álgebra de Cayley-Dickson recursiva de-confinada en la FPU.

    Soporta operaciones sobre
        \mathbb{R},\mathbb{C},\mathbb{H},\mathbb{O},\mathbb{S},\mathbb{P},\mathbb{X}.

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
            raise ChingonDimensionError(
                f"{name} debe ser un vector algebraico unidimensional. Shape={arr.shape}."
            )
        if not _is_power_of_two(int(arr.size)):
            raise ChingonDimensionError(
                f"{name} debe tener dimensión potencia de dos. Obtenido: {arr.size}."
            )
        if not np.all(np.isfinite(arr)):
            raise ChingonNumericalSingularityError(
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
            raise ChingonDimensionError(
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
            raise ChingonNumericalSingularityError(
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
    def alternative_strain(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        Tensor de deformación alternativa izquierda:

            A_{\mathrm{alt}}^{L}(X,Y)=[X,X,Y]=(X\cdot X)\cdot Y-X\cdot(X\cdot Y).
        """
        return cls.associator(x, x, y)

    @classmethod
    def right_alternative_strain(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        Tensor de deformación alternativa derecha:

            A_{\mathrm{alt}}^{R}(X,Y)=[X,Y,Y]=(X\cdot Y)\cdot Y-X\cdot(Y\cdot Y).
        """
        return cls.associator(x, y, y)

    @classmethod
    def heptagonal_associator(
        cls,
        x1: np.ndarray,
        x2: np.ndarray,
        x3: np.ndarray,
        x4: np.ndarray,
        x5: np.ndarray,
        x6: np.ndarray,
        x7: np.ndarray,
    ) -> np.ndarray:
        r"""
        Asociador heptagonal de 7 vías (6-símplex):

            A_7(X_1,\dots,X_7)
            =((((((X_1 X_2)X_3)X_4)X_5)X_6)X_7)
             -X_1(X_2(X_3(X_4(X_5(X_6 X_7))))).
        """
        p12 = cls.multiply(x1, x2)
        p123 = cls.multiply(p12, x3)
        p1234 = cls.multiply(p123, x4)
        p12345 = cls.multiply(p1234, x5)
        p123456 = cls.multiply(p12345, x6)
        left = cls.multiply(p123456, x7)

        p67 = cls.multiply(x6, x7)
        p567 = cls.multiply(x5, p67)
        p4567 = cls.multiply(x4, p567)
        p34567 = cls.multiply(x3, p4567)
        p234567 = cls.multiply(x2, p34567)
        right = cls.multiply(x1, p234567)

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
    def moufang_defect(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        r"""
        Defecto de Moufang (identidades de octoniones) como máximo de:

            L:\ (x(yx))z - x(y(xz)),
            R:\ ((zx)y)x - z((xy)x),
            M:\ (xy)(zx) - (x(yz))x.
        """
        yx = cls.multiply(y, x)
        xy = cls.multiply(x, y)
        xz = cls.multiply(x, z)
        zx = cls.multiply(z, x)
        yz = cls.multiply(y, z)

        left_m = cls.multiply(cls.multiply(x, yx), z) - cls.multiply(x, cls.multiply(y, xz))
        right_m = cls.multiply(cls.multiply(zx, y), x) - cls.multiply(z, cls.multiply(xy, x))
        mid_m = cls.multiply(xy, zx) - cls.multiply(cls.multiply(x, yz), x)

        return max(
            KBNSummationKernel.norm(left_m),
            KBNSummationKernel.norm(right_m),
            KBNSummationKernel.norm(mid_m),
        )

    @classmethod
    def left_multiplication_matrix(cls, p: np.ndarray) -> np.ndarray:
        r"""
        Operador de multiplicación a la izquierda L_p:\mathbb{X}\to\mathbb{X},
        y \mapsto p y, en la base canónica \{e_0,\ldots,e_{63}\}.

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
#     Morfismo terminal: observe_metrics → ChingonMetricsReport
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_ChingonMetricObserver:
    r"""
    FASE 1 — Observe + Orient.

    Ingesta de tensores 64D, descomposición de Cayley-Dickson,
    construcción de estados inmutables, auditoría de N(x) y de la
    composición de Hurwitz en la FPU.

    Categoría fuente: \mathbf{Vec}_{64}\times\mathbf{Vec}_{64}.
    Categoría meta  : \mathbf{Metrics}.

    El método terminal `observe_metrics` es el objeto inicial de la Fase 2.
    """

    __slots__ = ("_thresholds",)

    def __init__(self, thresholds: Optional[ChingonThresholds] = None) -> None:
        self._thresholds: Final[ChingonThresholds] = thresholds or ChingonThresholds()

    @property
    def thresholds(self) -> ChingonThresholds:
        """Umbral metrológico activo del motor."""
        return self._thresholds

    def build_state(self, S: Sequence[float]) -> ChingonState:
        r"""
        Instancia un `ChingonState` inmutable a partir de un vector real en R^{64}.

            S\in\mathbb{R}^{64}\ \mapsto\ X=(P_1,P_2)\in\mathbb{P}\times\mathbb{P}.

        Audita simultáneamente \|X\|_2 y la forma cuadrática N(X)=X\overline{X}.
        """
        vector = _prepare_chingon_vector(S, "ChingonState.vector_rep")

        p1 = np.ascontiguousarray(vector[0:_PATHION_DIM].copy(), dtype=np.float64)
        p2 = np.ascontiguousarray(
            vector[_PATHION_DIM:_CHINGON_DIM].copy(), dtype=np.float64
        )
        p1.setflags(write=False)
        p2.setflags(write=False)

        norm_value = KBNSummationKernel.norm(vector)
        if not math.isfinite(norm_value):
            raise ChingonNumericalSingularityError(
                "La norma del estado chingónico no es finita."
            )

        if norm_value >= _SQRT_DBL_MAX:
            norm_squared = math.inf
        else:
            norm_squared = norm_value * norm_value

        real_part = float(vector[0])
        imag_norm = KBNSummationKernel.norm(vector[1:]) if vector.size > 1 else 0.0
        quadratic_scalar, quadratic_leakage = CayleyDicksonAlgebra64.quadratic_form(vector)

        is_zero = norm_value <= self._thresholds.zero_norm_threshold
        is_unitary = _within_tolerance(
            norm_value,
            1.0,
            self._thresholds.absolute_tolerance,
            self._thresholds.relative_tolerance,
        )

        sha_hash = _sha256_of_arrays((vector, p1, p2))

        return ChingonState(
            vector_rep=vector,
            p1=p1,
            p2=p2,
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
    ) -> ChingonMetricsReport:
        r"""
        Morfismo terminal de la Fase 1.

        Ejecuta el producto chingónico A·B y verifica la composición de Hurwitz:

            \delta_{\mathrm{composition}}
            =\bigl|\,\|A\cdot B\|_{\mathbb{X}}-\|A\|_{\mathbb{X}}\|B\|_{\mathbb{X}}\bigr|.

        Firma functorial:

            \mathrm{observe\_metrics}:
                \mathbb{R}^{64}\times\mathbb{R}^{64}\longrightarrow
                \mathbf{ChingonMetricsReport}.

        El valor de retorno es el objeto inicial de
        `Phase2_HeptagonalFlexibilityCalculator.continue_from_metrics_report`.
        """
        state_a = self.build_state(A_vec)
        state_b = self.build_state(B_vec)

        product_vector = CayleyDicksonAlgebra64.multiply(
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

        return ChingonMetricsReport(
            state_a=state_a,
            state_b=state_b,
            product_state=product_state,
            product_norm=product_norm,
            expected_norm=expected_norm,
            hurwitz_composition_error=absolute_error,
            hurwitz_absolute_error=absolute_error,
            hurwitz_relative_error=relative_error,
            hurwitz_signed_defect=signed_defect,
            composition_ratio=composition_ratio,
            is_hurwitz_stable=is_hurwitz_stable,
            is_banach_submultiplicative=is_banach_submultiplicative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §G. FASE 2 — DECIDE
#     Objeto inicial = morfismo terminal de la Fase 1 (ChingonMetricsReport)
#     Morfismo terminal: decide_from_metrics_report → ChingonDecisionState
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_HeptagonalFlexibilityCalculator(Phase1_ChingonMetricObserver):
    r"""
    FASE 2 — Decide.

    Consume `ChingonMetricsReport` (terminal de Fase 1) y extiende la
    auditoría al 6-símplex de siete vías: asociador trilateral, deformación
    alternativa L/R, asociador heptagonal A_7, pentágono de Stasheff,
    defectos de flexibilidad, potencia y Moufang, volumen afín y espectro
    del laplaciano de similitud.

    El método inicial `continue_from_metrics_report` es la continuación
    formal de `Phase1.observe_metrics`.
    El método terminal `decide_from_metrics_report` produce el objeto
    inicial de la Fase 3 (`ChingonDecisionState`).
    """

    __slots__ = ()

    def continue_from_metrics_report(
        self,
        metrics_report: ChingonMetricsReport,
        x3_vec: Sequence[float],
        x4_vec: Sequence[float],
        x5_vec: Sequence[float],
        x6_vec: Sequence[float],
        x7_vec: Sequence[float],
        alternative_threshold: Optional[float] = None,
        heptagonal_threshold: Optional[float] = None,
    ) -> ChingonHeptagonalReport:
        r"""
        Morfismo de entrada de la Fase 2 (continuación formal de
        `Phase1_ChingonMetricObserver.observe_metrics`).

            \mathrm{continue\_from\_metrics\_report}:
                \mathbf{ChingonMetricsReport}\times(\mathbb{R}^{64})^{5}
                \longrightarrow\mathbf{ChingonHeptagonalReport}.
        """
        return self.decide_from_metrics_report(
            metrics_report=metrics_report,
            x3_vec=x3_vec,
            x4_vec=x4_vec,
            x5_vec=x5_vec,
            x6_vec=x6_vec,
            x7_vec=x7_vec,
            alternative_threshold=alternative_threshold,
            heptagonal_threshold=heptagonal_threshold,
        ).heptagonal_report

    def calculate_heptagonal_frustration(
        self,
        x1_vec: Sequence[float],
        x2_vec: Sequence[float],
        x3_vec: Sequence[float],
        x4_vec: Sequence[float],
        x5_vec: Sequence[float],
        x6_vec: Sequence[float],
        x7_vec: Sequence[float],
        alternative_threshold: Optional[float] = None,
        heptagonal_threshold: Optional[float] = None,
    ) -> ChingonHeptagonalReport:
        r"""Orquesta la frustración de calibre de 7 vías sin precomputar la Fase 1."""
        states = tuple(
            self.build_state(v)
            for v in (x1_vec, x2_vec, x3_vec, x4_vec, x5_vec, x6_vec, x7_vec)
        )
        return self._heptagonal_report_from_states(
            states=states,
            alternative_threshold=alternative_threshold,
            heptagonal_threshold=heptagonal_threshold,
        )

    def compute_trilateral_associator(
        self,
        X1: ChingonState,
        X2: ChingonState,
        X3: ChingonState,
    ) -> np.ndarray:
        r"""Asociador de 3 vías [X_1,X_2,X_3]=(X_1 X_2)X_3-X_1(X_2 X_3)."""
        return CayleyDicksonAlgebra64.associator(
            X1.vector_rep,
            X2.vector_rep,
            X3.vector_rep,
        )

    def compute_alternative_strain(
        self,
        X1: ChingonState,
        X2: ChingonState,
    ) -> np.ndarray:
        r"""Tensor de deformación alternativa izquierda A_alt^L(X_1,X_2)=[X_1,X_1,X_2]."""
        return CayleyDicksonAlgebra64.alternative_strain(
            X1.vector_rep,
            X2.vector_rep,
        )

    def compute_heptagonal_associator(
        self,
        X1: ChingonState,
        X2: ChingonState,
        X3: ChingonState,
        X4: ChingonState,
        X5: ChingonState,
        X6: ChingonState,
        X7: ChingonState,
    ) -> np.ndarray:
        r"""Asociador heptagonal A_7 de siete vías."""
        return CayleyDicksonAlgebra64.heptagonal_associator(
            X1.vector_rep,
            X2.vector_rep,
            X3.vector_rep,
            X4.vector_rep,
            X5.vector_rep,
            X6.vector_rep,
            X7.vector_rep,
        )

    def _stasheff_diameter(
        self,
        X1: ChingonState,
        X2: ChingonState,
        X3: ChingonState,
        X4: ChingonState,
    ) -> float:
        """Diámetro euclídeo del pentágono de Stasheff sobre cuatro chingones."""
        associations = CayleyDicksonAlgebra64.stasheff_pentagon_associations(
            X1.vector_rep,
            X2.vector_rep,
            X3.vector_rep,
            X4.vector_rep,
        )
        diameter = 0.0
        for i in range(len(associations)):
            for j in range(i + 1, len(associations)):
                delta = associations[i] - associations[j]
                diameter = max(diameter, KBNSummationKernel.norm(delta))
        return diameter

    def _diagnose_seven_way_simplex(
        self,
        states: Sequence[ChingonState],
    ) -> Tuple[float, float, float, float, int]:
        r"""
        Diagnóstico espectral, volumétrico y de conectividad del 6-símplex.

        Devuelve:
            1. Número de condición afín (SVD de la nube centrada por KBN).
            2. Volumen 6-dimensional (Gram / 6!).
            3. Conectividad algebraica (valor de Fiedler \lambda_2).
            4. Hueco espectral \lambda_2-\lambda_1 del laplaciano.
            5. Multiplicidad numérica de \lambda=0 (componentes conexas).
        """
        if len(states) != 7:
            raise ChingonEngineError(
                "El diagnóstico del 6-símplex requiere exactamente siete estados chingónicos."
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
            simplex_volume = _safe_exp(0.5 * float(logabs) - math.log(float(_SIMPLEX_FACTORIAL)))

        norms = np.array([state.norm for state in states], dtype=np.float64)
        valid = norms > self._thresholds.zero_norm_threshold

        if not np.any(valid):
            return condition_number, simplex_volume, 0.0, 0.0, 7

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

    def _heptagonal_report_from_states(
        self,
        states: Sequence[ChingonState],
        alternative_threshold: Optional[float],
        heptagonal_threshold: Optional[float],
    ) -> ChingonHeptagonalReport:
        """Orquesta el cálculo heptagonal completo a partir de siete estados ya construidos."""
        if len(states) != 7:
            raise ChingonEngineError(
                "La auditoría heptagonal requiere exactamente siete estados chingónicos."
            )

        X1, X2, X3, X4, X5, X6, X7 = states

        assoc_3_vector = self.compute_trilateral_associator(X1, X2, X3)
        trilateral_norm = KBNSummationKernel.norm(assoc_3_vector)

        alt_vector = self.compute_alternative_strain(X1, X2)
        alternative_norm = KBNSummationKernel.norm(alt_vector)

        right_alt_vector = CayleyDicksonAlgebra64.right_alternative_strain(
            X1.vector_rep, X2.vector_rep
        )
        right_alternative_norm = KBNSummationKernel.norm(right_alt_vector)

        hepta_vector = self.compute_heptagonal_associator(X1, X2, X3, X4, X5, X6, X7)
        heptagonal_norm = KBNSummationKernel.norm(hepta_vector)

        stasheff_diameter = self._stasheff_diameter(X1, X2, X3, X4)

        flexibility_defect = KBNSummationKernel.norm(
            CayleyDicksonAlgebra64.associator(
                X1.vector_rep, X2.vector_rep, X1.vector_rep
            )
        )
        power_associator_norm = KBNSummationKernel.norm(
            CayleyDicksonAlgebra64.associator(
                X1.vector_rep, X1.vector_rep, X1.vector_rep
            )
        )
        moufang_defect = CayleyDicksonAlgebra64.moufang_defect(
            X1.vector_rep, X2.vector_rep, X3.vector_rep
        )

        log_norm_alt = _log_norm(alternative_norm)
        log_norm_hepta = _log_norm(heptagonal_norm)

        log_den_alt = (
            2.0 * _log_norm(max(X1.norm, _WILKINSON_FLOOR))
            + _log_norm(max(X2.norm, _WILKINSON_FLOOR))
        )
        log_den_hepta = sum(
            _log_norm(max(state.norm, _WILKINSON_FLOOR)) for state in states
        )

        alternative_relative_norm = _safe_exp(log_norm_alt - log_den_alt)
        heptagonal_relative_norm = _safe_exp(log_norm_hepta - log_den_hepta)
        alternative_frustration_index = _safe_exp(log_norm_alt - max(0.0, log_den_alt))
        heptagonal_frustration_index = _safe_exp(log_norm_hepta - max(0.0, log_den_hepta))

        if alternative_threshold is None:
            alt_threshold_abs = float(self._thresholds.alternative_absolute)
        else:
            alt_threshold_abs = float(alternative_threshold)

        if heptagonal_threshold is None:
            hepta_threshold_abs = float(self._thresholds.heptagonal_absolute)
        else:
            hepta_threshold_abs = float(heptagonal_threshold)

        if not math.isfinite(alt_threshold_abs) or alt_threshold_abs < 0.0:
            raise ChingonEngineError(
                "El umbral alternativo debe ser finito y no negativo."
            )
        if not math.isfinite(hepta_threshold_abs) or hepta_threshold_abs < 0.0:
            raise ChingonEngineError(
                "El umbral heptagonal debe ser finito y no negativo."
            )

        log_alt_relative_threshold = (
            math.log(self._thresholds.alternative_relative) + log_den_alt
        )
        log_hepta_relative_threshold = (
            math.log(self._thresholds.heptagonal_relative) + log_den_hepta
        )

        alternative_threshold_used = max(
            alt_threshold_abs,
            _safe_exp(log_alt_relative_threshold),
        )
        heptagonal_threshold_used = max(
            hepta_threshold_abs,
            _safe_exp(log_hepta_relative_threshold),
        )

        alt_abs_ok = alternative_norm <= alt_threshold_abs
        alt_rel_ok = log_norm_alt <= log_alt_relative_threshold
        is_alternative_stable = math.isfinite(alternative_norm) and (
            alt_abs_ok or alt_rel_ok
        )

        hepta_abs_ok = heptagonal_norm <= hepta_threshold_abs
        hepta_rel_ok = log_norm_hepta <= log_hepta_relative_threshold
        is_heptagonal_norm_stable = math.isfinite(heptagonal_norm) and (
            hepta_abs_ok or hepta_rel_ok
        )

        is_heptagonal_stable = is_alternative_stable and is_heptagonal_norm_stable

        (
            condition_number,
            simplex_volume,
            connectivity,
            spectral_gap,
            components,
        ) = self._diagnose_seven_way_simplex(states)

        messages = []

        if not math.isfinite(alternative_norm):
            messages.append("A_alt no finito: singularidad alternativa detectada.")
        elif not is_alternative_stable:
            messages.append("A_alt fuera del umbral: deformación no alternativa activa.")

        if not math.isfinite(heptagonal_norm):
            messages.append("A_7 no finito: singularidad heptagonal detectada.")
        elif not is_heptagonal_norm_stable:
            messages.append("A_7 fuera del umbral: frustración de calibre heptagonal activa.")

        if stasheff_diameter > heptagonal_threshold_used:
            messages.append("Diámetro de Stasheff A_4 excesivo: homotopía A_∞ no controlada.")

        if condition_number > self._thresholds.condition_number_limit:
            messages.append(
                "El 6-símplex agéntico presenta condicionamiento afín degenerado."
            )

        if connectivity <= _MACHINE_EPS:
            messages.append(
                "Conectividad algebraica casi nula: el grafo de siete vías está desconectado."
            )

        if components > 1:
            messages.append(
                f"El laplaciano estima {components} componentes conexas en el 6-símplex."
            )

        if is_heptagonal_stable and not messages:
            messages.append("Auditoría heptagonal estable.")

        diagnosis = " | ".join(messages)

        return ChingonHeptagonalReport(
            trilateral_associator_norm=trilateral_norm,
            alternative_strain_norm=alternative_norm,
            heptagonal_associator_norm=heptagonal_norm,
            alternative_relative_norm=alternative_relative_norm,
            heptagonal_relative_norm=heptagonal_relative_norm,
            alternative_frustration_index=alternative_frustration_index,
            heptagonal_frustration_index=heptagonal_frustration_index,
            alternative_threshold_used=alternative_threshold_used,
            heptagonal_threshold_used=heptagonal_threshold_used,
            is_alternative_stable=is_alternative_stable,
            is_heptagonal_norm_stable=is_heptagonal_norm_stable,
            is_heptagonal_stable=is_heptagonal_stable,
            simplex_condition_number=condition_number,
            laplacian_connectivity=connectivity,
            diagnosis=diagnosis,
            right_alternative_strain_norm=right_alternative_norm,
            flexibility_defect=flexibility_defect,
            power_associator_norm=power_associator_norm,
            moufang_defect=moufang_defect,
            stasheff_pentagon_diameter=stasheff_diameter,
            simplex_volume=simplex_volume,
            laplacian_spectral_gap=spectral_gap,
            estimated_connected_components=components,
        )

    def decide_from_metrics_report(
        self,
        metrics_report: ChingonMetricsReport,
        x3_vec: Sequence[float],
        x4_vec: Sequence[float],
        x5_vec: Sequence[float],
        x6_vec: Sequence[float],
        x7_vec: Sequence[float],
        alternative_threshold: Optional[float] = None,
        heptagonal_threshold: Optional[float] = None,
    ) -> ChingonDecisionState:
        r"""
        Morfismo terminal de la Fase 2.

        Empaqueta métricas de Hurwitz, auditoría heptagonal y los siete
        estados del 6-símplex en un `ChingonDecisionState`, objeto
        inicial de `Phase3_ChingonNullConeEvaluator.continue_from_decision_state`.
        """
        X3 = self.build_state(x3_vec)
        X4 = self.build_state(x4_vec)
        X5 = self.build_state(x5_vec)
        X6 = self.build_state(x6_vec)
        X7 = self.build_state(x7_vec)
        states = (
            metrics_report.state_a,
            metrics_report.state_b,
            X3,
            X4,
            X5,
            X6,
            X7,
        )
        heptagonal_report = self._heptagonal_report_from_states(
            states=states,
            alternative_threshold=alternative_threshold,
            heptagonal_threshold=heptagonal_threshold,
        )
        return ChingonDecisionState(
            metrics_report=metrics_report,
            heptagonal_report=heptagonal_report,
            states=states,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §H. FASE 3 — ACT
#     Objeto inicial = morfismo terminal de la Fase 2 (ChingonDecisionState)
#     Morfismo terminal: execute_heptagonal_audit → ChingonEngineState
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_ChingonNullConeEvaluator(Phase2_HeptagonalFlexibilityCalculator):
    r"""
    FASE 3 — Act.

    Evaluación espectral de divisores de cero en el Cono Nulo Chingónico
    \mathcal{N}(\mathbb{X}) y generación del certificado terminal de-confinado.

    El método inicial `continue_from_decision_state` es la continuación
    formal de `Phase2.decide_from_metrics_report`.
    El método terminal `execute_heptagonal_audit` cierra el ciclo OODA.
    """

    __slots__ = ()

    def continue_from_decision_state(
        self,
        decision: ChingonDecisionState,
    ) -> ChingonEngineState:
        r"""
        Morfismo de entrada de la Fase 3 (continuación formal de
        `Phase2_HeptagonalFlexibilityCalculator.decide_from_metrics_report`).

            \mathrm{continue\_from\_decision\_state}:
                \mathbf{ChingonDecisionState}\longrightarrow\mathbf{ChingonEngineState}.
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
            heptagonal_report=decision.heptagonal_report,
            null_report=null_report,
        )

        return ChingonEngineState(
            metrics_report=decision.metrics_report,
            heptagonal_report=decision.heptagonal_report,
            null_report=null_report,
            fpu_execution_time_ms=elapsed_ms,
            cryptographic_seal=cryptographic_seal,
            engine_version=__version__,
        )

    def _sigma_min_left(self, state: ChingonState) -> float:
        r"""
        Distancia espectral de x al esquema de divisores de cero:

            \mathrm{dist}(x,\mathcal{N})\sim\sigma_{\min}(L_x).

        Se evalúa L_{\hat x} sobre el chingón normalizado y se reescala
        por \|x\|_2 (homogeneidad de L).
        """
        if state.is_zero or state.norm <= self._thresholds.zero_norm_threshold:
            return 0.0

        hat = np.asarray(state.vector_rep / state.norm, dtype=np.float64)
        try:
            op = CayleyDicksonAlgebra64.left_multiplication_matrix(hat)
            singular_values = la.svdvals(op, check_finite=True)
        except (la.LinAlgError, ChingonNumericalSingularityError):
            try:
                op = CayleyDicksonAlgebra64.left_multiplication_matrix(hat)
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
        X1_state: ChingonState,
        X2_state: ChingonState,
    ) -> NullConeReport:
        r"""
        Mide la fricción exergética de incursión en el cono de divisores de cero:

            \chi_{\mathrm{chingon\_null}}
            =\bigl|\,\|X_1\cdot X_2\|_{\mathbb{X}}
              -\|X_1\|_{\mathbb{X}}\|X_2\|_{\mathbb{X}}\bigr|,

            d_{\mathcal{N}}
            =\max\bigl(0,1-\|X_1 X_2\|/(\|X_1\|\|X_2\|)\bigr).

        Complemento espectral: \sigma_{\min}(L_{X_1}),\ \sigma_{\min}(L_{X_2})
        y norma del conmutador \|[X_1,X_2]\|=\|X_1 X_2-X_2 X_1\|.
        """
        product_vector = CayleyDicksonAlgebra64.multiply(
            X1_state.vector_rep,
            X2_state.vector_rep,
        )
        product_norm = KBNSummationKernel.norm(product_vector)

        reverse_vector = CayleyDicksonAlgebra64.multiply(
            X2_state.vector_rep,
            X1_state.vector_rep,
        )
        commutator_norm = KBNSummationKernel.norm(product_vector - reverse_vector)

        log_expected = _log_norm(X1_state.norm) + _log_norm(X2_state.norm)
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

        trivial_null = X1_state.is_zero or X2_state.is_zero

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

        sigma_min_x1 = self._sigma_min_left(X1_state)
        sigma_min_x2 = self._sigma_min_left(X2_state)

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
                "Fase 3: penetración no trivial del Cono Nulo Chingónico. "
                "product=%.6e expected=%.6e depth=%.6e σmin(L1)=%.6e σmin(L2)=%.6e",
                product_norm,
                expected_norm,
                null_depth,
                sigma_min_x1,
                sigma_min_x2,
            )

        return NullConeReport(
            product_norm=product_norm,
            expected_norm=expected_norm,
            absolute_friction=absolute_friction,
            relative_defect=relative_defect,
            null_depth=null_depth,
            sigma_min_left_x1=sigma_min_x1,
            sigma_min_left_x2=sigma_min_x2,
            commutator_norm=commutator_norm,
            is_trivial_null=trivial_null,
            is_null_cone_penetrated=is_null_cone_penetrated,
        )

    def _compute_cryptographic_seal(
        self,
        states: Sequence[ChingonState],
        metrics_report: ChingonMetricsReport,
        heptagonal_report: ChingonHeptagonalReport,
        null_report: NullConeReport,
    ) -> str:
        """
        Sello SHA-256 write-protected de la sesión.

        Incorpora versión del motor, umbrales empaquetados en IEEE-754
        little-endian, los siete vectores soberanos y los escalares
        críticos de las tres fases. Toda mutación posterior invalida
        el certificado.
        """
        hasher = hashlib.sha256()
        hasher.update(__version__.encode("utf-8"))

        for name in ChingonThresholds.__dataclass_fields__:
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
            heptagonal_report.trilateral_associator_norm,
            heptagonal_report.alternative_strain_norm,
            heptagonal_report.heptagonal_associator_norm,
            heptagonal_report.alternative_relative_norm,
            heptagonal_report.heptagonal_relative_norm,
            heptagonal_report.alternative_frustration_index,
            heptagonal_report.heptagonal_frustration_index,
            heptagonal_report.alternative_threshold_used,
            heptagonal_report.heptagonal_threshold_used,
            float(heptagonal_report.is_alternative_stable),
            float(heptagonal_report.is_heptagonal_norm_stable),
            float(heptagonal_report.is_heptagonal_stable),
            heptagonal_report.simplex_condition_number,
            heptagonal_report.laplacian_connectivity,
            heptagonal_report.right_alternative_strain_norm,
            heptagonal_report.flexibility_defect,
            heptagonal_report.power_associator_norm,
            heptagonal_report.moufang_defect,
            heptagonal_report.stasheff_pentagon_diameter,
            heptagonal_report.simplex_volume,
            heptagonal_report.laplacian_spectral_gap,
            float(heptagonal_report.estimated_connected_components),
            null_report.product_norm,
            null_report.expected_norm,
            null_report.absolute_friction,
            null_report.relative_defect,
            null_report.null_depth,
            null_report.sigma_min_left_x1,
            null_report.sigma_min_left_x2,
            null_report.commutator_norm,
            float(null_report.is_trivial_null),
            float(null_report.is_null_cone_penetrated),
        )
        for scalar in scalar_payload:
            hasher.update(_canonical_float_bytes(float(scalar)))

        return hasher.hexdigest()

    def execute_heptagonal_audit(
        self,
        contractor_X1: Sequence[float],
        subcontractor_X2: Sequence[float],
        supplier_X3: Sequence[float],
        interventor_X4: Sequence[float],
        fiduciary_X5: Sequence[float],
        insurer_X6: Sequence[float],
        entity_X7: Sequence[float],
        alternative_threshold: Optional[float] = None,
        heptagonal_threshold: Optional[float] = None,
    ) -> ChingonEngineState:
        r"""
        Morfismo terminal de la Fase 3 y del motor.

        Orquesta el ciclo ciego completo en la FPU para auditar un
        megaconsorcio de 7 vías como composición functorial estricta:

            \mathrm{Observe}(X_1,X_2)
                \xrightarrow{\text{Fase 1}} \mathbf{Metrics}
            \xrightarrow{\text{Fase 2}} \mathbf{Decision}
            \xrightarrow{\text{Fase 3}} \mathbf{Certificate}.

        Flujo:
            1. observe_metrics(X1, X2) → ChingonMetricsReport.
            2. decide_from_metrics_report(metrics, X3..X7)
               → ChingonDecisionState.
            3. continue_from_decision_state(decision)
               → ChingonEngineState.
        """
        t_start = time.perf_counter()

        metrics_report = self.observe_metrics(contractor_X1, subcontractor_X2)

        decision = self.decide_from_metrics_report(
            metrics_report=metrics_report,
            x3_vec=supplier_X3,
            x4_vec=interventor_X4,
            x5_vec=fiduciary_X5,
            x6_vec=insurer_X6,
            x7_vec=entity_X7,
            alternative_threshold=alternative_threshold,
            heptagonal_threshold=heptagonal_threshold,
        )

        engine_state = self.continue_from_decision_state(decision)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        engine_state = replace(engine_state, fpu_execution_time_ms=elapsed_ms)

        logger.debug(
            "Auditoría heptagonal v%s completada en %.6f ms. "
            "Hurwitz=%s Banach=%s Heptagonal=%s NullCone=%s",
            __version__,
            elapsed_ms,
            metrics_report.is_hurwitz_stable,
            metrics_report.is_banach_submultiplicative,
            decision.heptagonal_report.is_heptagonal_stable,
            engine_state.null_report.is_null_cone_penetrated,
        )
        return engine_state


# ═══════════════════════════════════════════════════════════════════════════════
# §I. FACHADA SOBERANA DEL MOTOR
# ═══════════════════════════════════════════════════════════════════════════════
# La fachada pública es la propia Fase 3, preservando la arquitectura de
# tres fases anidadas sin introducir una cuarta capa ontológica.
# Phase3 ⊏ Phase2 ⊏ Phase1  ⇒  Act contiene Decide contiene Observe.
ChingonDependencyEngine = Phase3_ChingonNullConeEvaluator