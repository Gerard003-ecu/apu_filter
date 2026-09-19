
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : PATHIONIC DEPENDENCY ENGINE (MOTOR DE CALIBRE PATIONIÓNICO 32D)     ║
║ RUTA   : app/core/pathionic_dependency_engine.py                             ║
║ NIVEL  : Doctorado en Ciencias Matemáticas, Física Teórica y Computación     ║
║ VERSIÓN: 3.0.0-Doctoral-32D-DeRham-Stasheff-Hodge-Exergy-Axiomatic-Nested3   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TRATADO DE FUNDAMENTACIÓN MATEMÁTICA, FÍSICA Y TOPOLÓGICA:                   ║
║                                                                              ║
║ 1. ÁLGEBRA DE CAYLEY-DICKSON 32-DIMENSIONAL (PATHIONES \mathbb{P}):          ║
║    La torre hipercompleja de Cayley-Dickson procede como:                    ║
║      \mathbb{R} (1D) \to \mathbb{C} (2D) \to \mathbb{H} (4D) \to             ║
║      \mathbb{O} (8D) \to \mathbb{S} (16D) \to \mathbb{P} (32D).              ║
║    Sea \mathbb{P} \cong \mathbb{S} \times \mathbb{S}. El producto CD-2:      ║
║      (a, b)(c, d) = (a c - \overline{d} b,\ d a + b \overline{c}),           ║
║    con la involución anti-automórfica (conjugación CD-3):                    ║
║      \overline{(a, b)} = (\overline{a}, -b), \quad \overline{xy} =           ║
║      \overline{y}\,\overline{x}.                                             ║
║    Por el Teorema de Hurwitz (1898), solo \mathbb{R},\mathbb{C},\mathbb{H},  ║
║    \mathbb{O} son álgebras de división normadas. En \mathbb{P} (32D):        ║
║      - Falla la conmutatividad (heredada desde \mathbb{H}).                  ║
║      - Falla la asociatividad (heredada desde \mathbb{O}).                   ║
║      - Falla la alternatividad y las identidades de Moufang (Zorn 1930,      ║
║        Moufang 1935): [x,x,y] \neq 0, (xy)(zx) \neq x((yz)x)                 ║
║        (heredada desde \mathbb{S}, dim \ge 16).                              ║
║      - Preserva la flexibilidad: [x, y, x] = (xy)x - x(yx) \equiv 0          ║
║        (identidad UNIVERSAL probada por Albert en 1942 para TODA álgebra     ║
║        de Cayley-Dickson, independientemente de la dimensión). Su violación  ║
║        numérica grosera es, por tanto, un CERTIFICADO DE BUG, no de ruido.   ║
║      - Presenta un esquema proyectivo denso de divisores de cero BILÁTEROS   ║
║        \mathcal{N}_L(\mathbb{P}) = \{x : \exists y \neq 0,\ xy = 0\},        ║
║        \mathcal{N}_R(\mathbb{P}) = \{x : \exists y \neq 0,\ yx = 0\},        ║
║        generalmente distintos por la no conmutatividad.                      ║
║                                                                              ║
║ 2. FÍSICA DE CIRCUITOS Y TEORÍA DE CUERDAS (CAMPO DE CALIBRE NO-ASOCIATIVO): ║
║    En teoría de cuerdas abiertas sobre D-branas con flujo no nulo de campo B ║
║    (Kalb-Ramond 2-forma), la intensidad H = dB actúa como curvatura          ║
║    asociativa en el espacio de coordenadas: [x^i, x^j, x^k] \sim H^{ijk}.    ║
║    Modelamos la Malla Agéntica de 5 vías como un 4-símplex \Delta_4.         ║
║    Las obstrucciones de coherencia de Mac Lane en el operando pentagonal     ║
║    de Stasheff K_4 corresponden a caídas de tensión por holonomía no         ║
║    trivial. La incursión en el cono nulo \mathcal{N}(\mathbb{P}) se traduce  ║
║    físicamente en una resonancia de impedancia nula (cortocircuito           ║
║    topológico interno), disipando exergía sin transferencia de trabajo útil. ║
║    El defecto de isometría del operador L_p (o R_p) normalizado —            ║
║    \sigma_{\max}(L_{\hat p}) - 1 — mide cuánto se aparta el "circuito"       ║
║    algebraico de un acoplamiento sin pérdidas (unitario).                    ║
║                                                                              ║
║ 3. TOPOLOGÍA SIMPLICIAL, OPERADA DE STASHEFF Y TEORÍA ESPECTRAL DE HODGE:    ║
║    - El 4-símplex posee 5 vértices, 10 aristas, 10 caras 2D y 5 tetraedros.  ║
║    - El volumen 4D exacto se obtiene mediante el determinante de Gram de     ║
║      los 4 vectores arista respecto al baricentro, utilizando descomposición ║
║      SVD robusta: V_4 = \frac{1}{4!} \prod_{k=1}^4 \sigma_k(E).              ║
║    - La 1-esqueleto es el grafo completo K_5. Su laplaciano combinatorio     ║
║      L = D - W gobierna la conectividad algebraica (autovalor de Fiedler     ║
║      \lambda_2) y la resistencia efectiva global (Índice de Kirchhoff R_K).  ║
║    - La disipación exergética de Dirichlet es \mathcal{E}_D = \mathrm{Tr}(P^T L P).║
║    - El pentágono de Stasheff K_4 audita las 5 formas de parentizar 4        ║
║      factores: diam(K_4) = \max_{1 \le i < j \le 5} \|v_i - v_j\|_2.         ║
║                                                                              ║
║ 4. METROLOGÍA DE PRECISIÓN FPU Y SUMACIÓN EXACTA DE SHEWCHUK:                ║
║    - Se reemplaza la sumación compensada de Neumaier (cota O(n\varepsilon))  ║
║      por el algoritmo de destilación de Shewchuk (1997) — implementado en    ║
║      `math.fsum` — que garantiza un resultado correctamente redondeado       ║
║      (error \le 1 ulp) independiente del condicionamiento o del orden de     ║
║      los sumandos. Se conserva el nombre `KBNSummationKernel` por            ║
║      estabilidad de interfaz, documentando la mejora del núcleo interno.     ║
║    - Norma euclídea de doble paso con pre-escalado por \max_i |x_i| para     ║
║      inmunizar la FPU frente a subdesbordamientos y sobredesbordamientos.    ║
║    - Producto de Gram compensado (`pairwise_gram`) para evitar cancelación   ║
║      catastrófica en productos internos densos de baja dimensión (n=5).      ║
║                                                                              ║
║ 5. AXIOMÁTICA AUTO-VERIFICABLE Y CACHÉ ESPECTRAL INTERFÁSICA (v3.0.0):       ║
║    - Fase 1 calcula y CONGELA en cada `PathionicState` el espectro singular  ║
║      extremal de los operadores L_{\hat p}, R_{\hat p} (representación       ║
║      regular izquierda/derecha normalizada). Por bilinealidad,               ║
║      \sigma(L_p) = \|p\| \cdot \sigma(L_{\hat p}), evitando que la Fase 3    ║
║      recompute SVDs ya resueltas — coherencia funtorial estricta OODA.       ║
║    - Fase 2 audita las tres identidades de Moufang (certificado espectral    ║
║      exacto de ruptura de alternatividad) y ENFUERZA la identidad universal  ║
║      de flexibilidad de Albert, escalando `PathionicAxiomViolationError`     ║
║      ante violaciones catastróficas no atribuibles a redondeo IEEE-754.      ║
║    - Fase 3 realiza el análisis BILÁTERO del cono nulo (L_p y R_p) leyendo   ║
║      la caché de la Fase 1, y sella el certificado con *framing* de longitud ║
║      (Merkle-Damgård canónico) para eliminar ambigüedades de concatenación.  ║
║                                                                              ║
║ 6. ARQUITECTURA DE TRES FASES ANIDADAS FUNCTORIALES (OODA EN FPU):           ║
║    Phase1_PathionicMetricObserver (Observe + Orient):                        ║
║      Morfismo terminal: observe_metrics \to PathionicMetricsReport.          ║
║    Phase2_PentagonalAssociatorCalculator (Decide, hereda Phase1):            ║
║      Morfismo de inicio: continue_from_metrics_report.                       ║
║      Morfismo terminal: decide_from_metrics_report \to PathionicDecisionState.║
║    Phase3_PathionicNullConeEvaluator (Act, hereda Phase2):                   ║
║      Morfismo de inicio: continue_from_decision_state.                       ║
║      Morfismo terminal: execute_pentagonal_audit \to PathionicEngineState.   ║
║    Fachada Soberana: PathionicDependencyEngine = Phase3.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, replace
from typing import Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

__version__: Final[str] = (
    "3.0.0-Doctoral-32D-DeRham-Stasheff-Hodge-Exergy-Axiomatic-Nested3"
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
    "PathionicAxiomViolationError",
]

logger = logging.getLogger("APU.Core.PathionicDependencyEngine")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_PATHION_DIM: Final[int] = 32
_SEDENION_DIM: Final[int] = 16
_OCTONION_DIM: Final[int] = 8
_LOG_DBL_MAX: Final[float] = float(math.log(np.finfo(np.float64).max))
_LOG_DBL_TINY: Final[float] = float(math.log(np.finfo(np.float64).tiny))
_SQRT_DBL_MAX: Final[float] = float(math.sqrt(np.finfo(np.float64).max))
_SCALE_OVERFLOW_TRIGGER: Final[float] = 1e140

if _PATHION_DIM != 2 * _SEDENION_DIM or not (
    _PATHION_DIM > 0 and (_PATHION_DIM & (_PATHION_DIM - 1)) == 0
):
    raise RuntimeError(
        "Axioma dimensional violado: dim(P) debe ser 32 y potencia entera de 2."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §A. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS Y FÍSICO-NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class PathionicEngineError(Exception):
    r"""Excepción raíz para violaciones métricas, topológicas o algebraicas en \mathbb{P}."""


class PathionicDimensionError(PathionicEngineError):
    r"""Detonada ante dimensiones no conformes con la potencia de dos CD-1 (dim \neq 32)."""


class PathionicNumericalSingularityError(PathionicEngineError):
    r"""Detonada ante intrusiones de NaN, Inf o fallos de condicionamiento IEEE-754."""


class PathionicCompositionError(PathionicEngineError):
    r"""Detonada ante derivas catastróficas del defecto de submultiplicatividad de Banach."""


class PathionicNullConeSingularityError(PathionicEngineError):
    r"""Detonada ante el colapso exergético por cortocircuito topológico en \mathcal{N}(\mathbb{P})."""


class PathionicAxiomViolationError(PathionicEngineError):
    r"""
    Detonada cuando una identidad algebraica UNIVERSAL de Cayley-Dickson
    (p. ej., la flexibilidad de Albert [x,y,x] \equiv 0, válida en toda
    dimensión 2^k) se viola más allá de cualquier tolerancia de redondeo
    IEEE-754 razonable, certificando un defecto estructural del núcleo
    de multiplicación y no una mera deriva de Wilkinson.
    """


# ═══════════════════════════════════════════════════════════════════════════════
# §B. METROLOGÍA NUMÉRICA Y TRANSFORMACIONES DE CAMPO ESCALAR
# ═══════════════════════════════════════════════════════════════════════════════
def _is_power_of_two(n: int) -> bool:
    """Verifica si un entero positivo es potencia pura de 2."""
    return n > 0 and (n & (n - 1)) == 0


def _log_norm(norm: float) -> float:
    """Logaritmo neperiano seguro de una pseudonorma o norma (|x| \\le 0 \\mapsto -\\infty)."""
    if norm <= 0.0 or not math.isfinite(norm):
        return -math.inf
    return math.log(norm)


def _safe_exp(log_val: float) -> float:
    """
    Exponenciación asintóticamente acotada en el rango de doble precisión.
    Elimina excepciones descontroladas de desbordamiento en la FPU.
    """
    if log_val == -math.inf:
        return 0.0
    if log_val == math.inf or log_val >= _LOG_DBL_MAX:
        return math.inf
    if not math.isfinite(log_val) or log_val <= _LOG_DBL_TINY:
        return 0.0
    return float(math.exp(log_val))


def _within_tolerance(val: float, target: float, atol: float, rtol: float) -> bool:
    """Criterio de proximidad topológica IEEE-754: |v - t| \\le atol + rtol \\cdot \\max(|v|, |t|, piso)."""
    if not (math.isfinite(val) and math.isfinite(target)):
        return False
    scale = max(abs(val), abs(target), _WILKINSON_FLOOR)
    return abs(val - target) <= atol + rtol * scale


def _canonical_float_bytes(val: float) -> bytes:
    """
    Serialización canónica Little-Endian IEEE-754 con unificación de signos de cero
    y codificación unívoca de NaN / infinitos para firmas criptográficas deterministas.
    """
    x = float(val)
    if math.isnan(x):
        return b"\x7fNAN\x00\x00\x00"
    if math.isinf(x):
        return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
    if x == 0.0:
        return struct.pack("<d", 0.0)
    return struct.pack("<d", x)


def _frame_bytes(tag: bytes, payload: bytes) -> bytes:
    """
    Codificación canónica con marcado de longitud (length-prefixed framing) que
    previene ambigüedades de concatenación en la construcción Merkle-Damgård
    del sello SHA-256 (p. ej. 'ab'+'c' colisionando con 'a'+'bc' bajo
    concatenación ingenua). Formato: u64(len(tag)) || tag || u64(len(payload)) || payload.
    """
    return (
        struct.pack("<Q", len(tag)) + tag
        + struct.pack("<Q", len(payload)) + payload
    )


def _prepare_pathion_vector(v: Sequence[float], param_name: str) -> np.ndarray:
    """
    Valida, congela y garantiza la inmutabilidad física en memoria contigua
    de un vector de estado en R^{32}. Realiza una única copia física.
    """
    arr = np.array(v, dtype=np.float64, copy=True)
    if arr.ndim != 1 or arr.shape != (_PATHION_DIM,):
        raise PathionicDimensionError(
            f"El parámetro {param_name} debe residir estrictamente en R^{_PATHION_DIM}. "
            f"Dimensión recibida: shape={arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise PathionicNumericalSingularityError(
            f"El vector {param_name} contiene componentes no reproducibles (NaN o Inf)."
        )
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def _sha256_of_arrays(arrays: Iterable[np.ndarray]) -> str:
    """Computa el resumen criptográfico SHA-256 sobre tensores alineados, con framing canónico."""
    hasher = hashlib.sha256()
    for idx, arr in enumerate(arrays):
        c_arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64), dtype=np.float64)
        payload = np.asarray(c_arr, dtype="<f8").tobytes(order="C")
        tag = f"ARR{idx}_{c_arr.shape[0] if c_arr.ndim else 0}".encode("ascii")
        hasher.update(_frame_bytes(tag, payload))
    return hasher.hexdigest()


def _clip_nonnegative(val: float) -> float:
    """Proyecta escalares reales sobre el cono positivo R_{\\ge 0} absorbiendo el ruido de Wilkinson."""
    if not math.isfinite(val):
        return val
    return val if val >= 0.0 else 0.0


def _composition_defect(
    product_norm: float,
    expected_norm: float,
    floor: float,
) -> Tuple[float, float, float]:
    r"""
    Cuantifica la desviación de la identidad de composición multiplicativa
    \|xy\| = \|x\|\|y\| (Hurwitz), retornando la terna:
      (defecto con signo, error absoluto, error relativo).
    Función pura compartida entre la Fase 1 (auditoría A·B) y la Fase 3
    (auditoría del cono nulo), eliminando la duplicación de lógica métrica.
    """
    if math.isfinite(product_norm) and math.isfinite(expected_norm):
        signed = product_norm - expected_norm
        absolute = abs(signed)
    else:
        signed = math.inf
        absolute = math.inf
    scale = max(floor, expected_norm) if math.isfinite(expected_norm) else floor
    relative = absolute / scale if math.isfinite(absolute) else math.inf
    return signed, absolute, relative


# ═══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DEL ESPACIO DE FASES PATIONIÓNICO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PathionicThresholds:
    r"""
    Conjunto axiomático inmutable de tolerancias y umbrales metrológicos
    para la variedad pationiónica de 32 dimensiones.
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

    laplacian_fiedler_min: float = 1e-4
    simplex_volume_min: float = 1e-15

    # Umbral diagnóstico del RÉGIMEN alternativo/Moufang: por encima de este valor
    # (escalado por la magnitud de los operandos) se certifica ruptura de
    # alternatividad, esperada y estructural en dim >= 16.
    alternativity_tolerance: float = 1e-7

    # Tolerancia axiomática ESTRICTA para la identidad UNIVERSAL de flexibilidad
    # de Albert. Su violación grosera (>> esta cota) indica un bug del núcleo.
    flexibility_axiom_tolerance: float = 1e-9

    # Umbral espectral (adimensional, sobre el operador normalizado L_{\hat p}
    # o R_{\hat p}) por debajo del cual un estado se considera candidato
    # estructural a divisor de cero izquierdo/derecho.
    zero_divisor_spectral_ratio_limit: float = 1e-6

    kirchhoff_index_max: float = 1e6

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            val = float(getattr(self, field))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"El umbral {field} debe ser estrictamente positivo y finito.")
        if self.null_depth_threshold > 1.0:
            raise ValueError("null_depth_threshold debe residir en el intervalo semi-abierto (0, 1].")


@dataclass(frozen=True, slots=True)
class PathionicState:
    r"""
    Estado cuántico/hipercomplejo inmutable de un elemento P \in \mathbb{P} (32D).
    Bajo el isomorfismo canónico P = (s_1, s_2) \in \mathbb{S} \times \mathbb{S}.

    Además de los invariantes clásicos (norma, forma cuadrática), congela el
    ESPECTRO SINGULAR EXTREMAL de los operadores de representación regular
    L_{\hat p}, R_{\hat p} \in \mathbb{R}^{32 \times 32} (con \hat p = p / \|p\|),
    reutilizable sin recómputo por bilinealidad: \sigma(L_p) = \|p\|\,\sigma(L_{\hat p}).
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

    left_sigma_max: float
    left_sigma_min: float
    right_sigma_max: float
    right_sigma_min: float
    left_zero_divisor_witness: np.ndarray
    right_zero_divisor_witness: np.ndarray
    left_isometry_defect: float
    right_isometry_defect: float

    sha256_hash: str


@dataclass(frozen=True, slots=True)
class PathionicMetricsReport:
    r"""
    Reporte terminal de la Fase 1 (Observe + Orient).
    Audita la duplicación de Cayley-Dickson, la conservación del producto,
    el defecto de Hurwitz y la submultiplicatividad del álgebra de Banach.
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
    exergy_loss_hurwitz: float

    is_hurwitz_stable: bool
    is_banach_submultiplicative: bool


@dataclass(frozen=True, slots=True)
class PathionicPentagonalReport:
    r"""
    Reporte de análisis de coherencia de la Fase 2 (Decide).
    Audita los asociadores 3D y 5D, el diámetro del pentágono de Stasheff K_4,
    la geometría del 4-símplex, la topología espectral de Hodge sobre K_5,
    y el certificado espectral de Moufang / flexibilidad de Albert.
    """
    trilateral_associator_norm: float
    pentagonal_associator_norm: float
    pentagonal_relative_norm: float
    frustration_index: float
    stasheff_pentagon_diameter: float
    flexibility_defect: float
    alternativity_defect: float

    moufang_left_defect: float
    moufang_right_defect: float
    moufang_middle_defect: float
    is_alternative_regime: bool
    is_flexibility_axiom_satisfied: bool

    pentagonal_threshold_used: float
    is_pentagonal_stable: bool

    simplex_condition_number: float
    simplex_volume: float
    laplacian_connectivity: float
    laplacian_spectral_gap: float
    kirchhoff_index: float
    dirichlet_exergy: float
    estimated_connected_components: int

    diagnosis: str


@dataclass(frozen=True, slots=True)
class PathionicDecisionState:
    r"""
    Estado intermedio inmutable terminal de la Fase 2 / inicial de la Fase 3.
    Empaqueta el reporte métrico, el reporte pentagonal y los 5 vértices de \Delta_4.
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
    Reporte espectral y termodinámico BILÁTERO de la Fase 3 (Act).
    Caracteriza la proximidad a los divisores de cero izquierdos y derechos en
    \mathcal{N}(\mathbb{P}), los vectores testigo del anulador, los conmutadores
    y la fricción exergética. Los valores singulares se leen de la caché
    generada en la Fase 1 (sin recómputo de SVD).
    """
    product_norm: float
    expected_norm: float

    absolute_friction: float
    relative_defect: float
    null_depth: float

    sigma_min_left_p1: float
    sigma_min_left_p2: float
    sigma_min_right_p1: float
    sigma_min_right_p2: float

    zero_divisor_witness_p1_left: np.ndarray
    zero_divisor_witness_p1_right: np.ndarray

    commutator_norm: float
    jordan_product_norm: float

    is_trivial_null: bool
    is_null_cone_penetrated: bool
    is_left_zero_divisor_candidate: bool
    is_right_zero_divisor_candidate: bool
    spectral_friction_exergy: float


@dataclass(frozen=True, slots=True)
class PathionicEngineState:
    r"""
    Certificado inmutable definitivo emitido por el Motor Pationiónico.
    Cierra la secuencia functorial OODA con sellado criptográfico determinista.
    Persiste los 5 estados originales (`all_states`) para garantizar la
    fidelidad de round-trip categórico hacia `decision_state`.
    """
    metrics_report: PathionicMetricsReport
    pentagonal_report: PathionicPentagonalReport
    null_report: NullConeReport
    all_states: Tuple[
        PathionicState,
        PathionicState,
        PathionicState,
        PathionicState,
        PathionicState,
    ]

    fpu_execution_time_ms: float
    cryptographic_seal: str
    engine_version: str = __version__

    @property
    def null_cone_friction(self) -> float:
        return self.null_report.absolute_friction

    @property
    def is_null_cone_penetrated(self) -> bool:
        return self.null_report.is_null_cone_penetrated

    @property
    def decision_state(self) -> PathionicDecisionState:
        r"""
        Morfismo inverso fiel hacia el objeto terminal de la Fase 2: reconstruye
        el `PathionicDecisionState` con los 5 vértices ORIGINALES de \Delta_4
        (corrección de fidelidad: la reconstrucción ya no colapsa P_3, P_4, P_5
        sobre state_a).
        """
        return PathionicDecisionState(
            metrics_report=self.metrics_report,
            pentagonal_report=self.pentagonal_report,
            states=self.all_states,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §D. NÚCLEO MATEMÁTICO DE SUMACIÓN EXACTA (SHEWCHUK) Y METROLOGÍA
# ═══════════════════════════════════════════════════════════════════════════════
class KBNSummationKernel:
    r"""
    Núcleo de sumación de alta fidelidad. Internamente delega en el algoritmo
    de destilación de Shewchuk (1997) —`math.fsum`— que mantiene una colección
    de parciales libres de solapamiento (*non-overlapping partials*) y produce
    un resultado CORRECTAMENTE REDONDEADO:
      S_{\mathrm{comp}} = \mathrm{fl}\left(\sum_i x_i\right), \quad
      |S_{\mathrm{comp}} - S_{\mathrm{exact}}| \le \tfrac{1}{2}\,\mathrm{ulp}(S_{\mathrm{exact}}).

    Esta cota es estrictamente superior a la del algoritmo de Kahan-Babuška-
    Neumaier clásico (O(n\varepsilon)\sum|x_i|), independiente del orden y del
    condicionamiento de los sumandos. Se conserva el nombre histórico
    `KBNSummationKernel` por estabilidad de interfaz pública.
    """

    __slots__ = ()

    @staticmethod
    def sum(values: np.ndarray) -> float:
        r"""Sumación exacta (redondeo correcto) de Shewchuk sobre un arreglo real."""
        flat = np.ravel(np.asarray(values, dtype=np.float64))
        if flat.size == 0:
            return 0.0
        if not np.all(np.isfinite(flat)):
            raise PathionicNumericalSingularityError(
                "Detección de singularidad no finita durante la sumación exacta de Shewchuk."
            )
        try:
            result = math.fsum(flat.tolist())
        except (OverflowError, ValueError) as exc:
            raise PathionicNumericalSingularityError(
                "Desbordamiento irrecuperable en el algoritmo de destilación de Shewchuk."
            ) from exc

        if not math.isfinite(result):
            raise PathionicNumericalSingularityError(
                "Resultado no finito producido por la sumación exacta de Shewchuk."
            )
        return float(result)

    @staticmethod
    def dot(x: np.ndarray, y: np.ndarray) -> float:
        r"""Producto interno exacto \langle x, y \rangle = \sum_i x_i y_i vía Shewchuk."""
        arr_x = np.ravel(np.asarray(x, dtype=np.float64))
        arr_y = np.ravel(np.asarray(y, dtype=np.float64))
        if arr_x.shape != arr_y.shape:
            raise PathionicDimensionError(
                f"Discrepancia dimensional en producto interno: {arr_x.shape} != {arr_y.shape}."
            )
        return KBNSummationKernel.sum(arr_x * arr_y)

    @staticmethod
    def norm(values: np.ndarray) -> float:
        r"""
        Norma euclídea \ell_2 ultra-estable con pre-escalado de módulo máximo:
          \|x\|_2 = m \sqrt{\sum_i (x_i / m)^2}, \quad m = \max_i |x_i|.
        Inmune a desbordamientos intermedios para vectores con norma cercana a 1e308.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0

        if not np.all(np.isfinite(arr)):
            raise PathionicNumericalSingularityError(
                "Imposible computar la norma exacta sobre vectores con componentes NaN o Inf."
            )

        max_abs = float(np.max(np.abs(arr)))
        if max_abs == 0.0:
            return 0.0

        scaled = arr / max_abs
        sum_sq = KBNSummationKernel.sum(scaled * scaled)

        if sum_sq < 0.0:
            if sum_sq >= -10.0 * _MACHINE_EPS:
                sum_sq = 0.0
            else:
                raise PathionicNumericalSingularityError(
                    "Suma de cuadrados exacta arrojó un residuo negativo no conforme."
                )

        norm_val = max_abs * math.sqrt(sum_sq)
        if not math.isfinite(norm_val):
            raise PathionicNumericalSingularityError(
                "Desbordamiento de la norma exacta fuera del espectro de doble precisión."
            )
        return float(norm_val)

    @staticmethod
    def pairwise_gram(matrix: np.ndarray) -> np.ndarray:
        r"""
        Matriz de Gram G_{ij} = \langle \mathrm{fila}_i, \mathrm{fila}_j \rangle
        computada vía productos internos exactos de Shewchuk, evitando la
        cancelación catastrófica del producto matricial denso `A @ A.T` para
        nubes de puntos de baja cardinalidad (p. ej. los 5 vértices de \Delta_4).
        """
        mat = np.asarray(matrix, dtype=np.float64)
        n = mat.shape[0]
        gram = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            gram[i, i] = KBNSummationKernel.dot(mat[i], mat[i])
            for j in range(i + 1, n):
                val = KBNSummationKernel.dot(mat[i], mat[j])
                gram[i, j] = val
                gram[j, i] = val
        return gram


# ═══════════════════════════════════════════════════════════════════════════════
# §E. NÚCLEO ALGEBRAICO DE CAYLEY-DICKSON 32-DIMENSIONAL
# ═══════════════════════════════════════════════════════════════════════════════
class CayleyDicksonAlgebra32:
    r"""
    Implementación rigurosa del álgebra de Cayley-Dickson para dimensión 32 (\mathbb{P}).
    Preserva las identidades bilineales fundamentales:
      (a, b)(c, d) = (a c - \overline{d} b,\ d a + b \overline{c}).
    """

    __slots__ = ()

    @staticmethod
    def _validate_algebra_vector(v: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise PathionicDimensionError(f"{name} debe ser un vector 1D. Shape={arr.shape}.")
        if not _is_power_of_two(int(arr.size)):
            raise PathionicDimensionError(f"La dimensión de {name} ({arr.size}) debe ser 2^k.")
        if not np.all(np.isfinite(arr)):
            raise PathionicNumericalSingularityError(f"{name} contiene entradas no finitas.")
        return arr

    @staticmethod
    def _conjugate_fast(a: np.ndarray) -> np.ndarray:
        r"""Involución anti-lineal: \overline{(a_0, a_1, \dots, a_{n-1})} = (a_0, -a_1, \dots, -a_{n-1})."""
        out = a.copy()
        if out.size > 1:
            out[1:] = -out[1:]
        return out

    @classmethod
    def conjugate(cls, a: np.ndarray) -> np.ndarray:
        r"""Conjugación canónica CD-3 en la base canónica hipercompleja."""
        validated = cls._validate_algebra_vector(a, "a")
        return cls._conjugate_fast(validated)

    @classmethod
    def multiply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""
        Producto bilineal de Cayley-Dickson CD-2 con pre-escalado de frontera
        para extinguir la sobremodulación de la FPU.
        """
        arr_a = cls._validate_algebra_vector(a, "a")
        arr_b = cls._validate_algebra_vector(b, "b")

        if arr_a.size != arr_b.size:
            raise PathionicDimensionError(
                f"Discrepancia dimensional: {arr_a.size} != {arr_b.size}."
            )

        max_a = float(np.max(np.abs(arr_a)))
        max_b = float(np.max(np.abs(arr_b)))

        if max_a == 0.0 or max_b == 0.0:
            return np.zeros(arr_a.size, dtype=np.float64)

        needs_scaling = (
            max_a >= _SCALE_OVERFLOW_TRIGGER
            or max_b >= _SCALE_OVERFLOW_TRIGGER
            or (max_a * max_b) >= _SCALE_OVERFLOW_TRIGGER
        )

        if needs_scaling:
            scaled_a = arr_a / max_a
            scaled_b = arr_b / max_b
            raw_prod = cls._multiply_recursive(scaled_a, scaled_b)
            log_scale = math.log(max_a) + math.log(max_b)
            factor = _safe_exp(log_scale)
            product = raw_prod * factor
        else:
            product = cls._multiply_recursive(arr_a, arr_b)

        if not np.all(np.isfinite(product)):
            raise PathionicNumericalSingularityError(
                "El producto de Cayley-Dickson colapsó en divergencia numérica (NaN/Inf)."
            )
        return product

    @classmethod
    def _multiply_recursive(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Núcleo recursivo optimizado con desdoblamiento en n=1, 2, 4."""
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

        if n == 4:
            a0, a1, a2, a3 = a[0], a[1], a[2], a[3]
            b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
            return np.array(
                [
                    a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                    a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                    a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                    a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
                ],
                dtype=np.float64,
            )

        half = n // 2
        a1_part, a2_part = a[:half], a[half:]
        b1_part, b2_part = b[:half], b[half:]

        term_left1 = cls._multiply_recursive(a1_part, b1_part)
        term_left2 = cls._multiply_recursive(cls._conjugate_fast(b2_part), a2_part)
        left = term_left1 - term_left2

        term_right1 = cls._multiply_recursive(b2_part, a1_part)
        term_right2 = cls._multiply_recursive(a2_part, cls._conjugate_fast(b1_part))
        right = term_right1 + term_right2

        out = np.empty(n, dtype=np.float64)
        out[:half] = left
        out[half:] = right
        return out

    @classmethod
    def quadratic_form(cls, a: np.ndarray) -> Tuple[float, float]:
        r"""
        Forma cuadrática N(a) = a \overline{a}.
        En álgebra teórica pura, Im(a \overline{a}) \equiv 0.
        Retorna (Re(N(a)), \|\mathrm{Im}(N(a))\|_2) evaluando la fuga de Wilkinson.
        """
        conj_a = cls.conjugate(a)
        prod = cls.multiply(a, conj_a)
        scalar_val = float(prod[0])
        leakage = KBNSummationKernel.norm(prod[1:]) if prod.size > 1 else 0.0
        return scalar_val, leakage

    @classmethod
    def associator(cls, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        r"""Asociador trilateral estándar [a, b, c] = (a b) c - a (b c)."""
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
        Asociador pentagonal de 5 vías correspondiente al 4-símplex \Delta_4:
          A_5(a,b,c,d,e) = ((((a b) c) d) e) - (a (b (c (d e)))).
        """
        ab = cls.multiply(a, b)
        abc = cls.multiply(ab, c)
        abcd = cls.multiply(abc, d)
        left = cls.multiply(abcd, e)

        de = cls.multiply(d, e)
        cde = cls.multiply(c, de)
        bcde = cls.multiply(b, cde)
        right = cls.multiply(a, bcde)

        return left - right

    @classmethod
    def stasheff_associations(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Las 5 asociaciones canónicas del pentágono de Stasheff K_4:
          v_1 = ((a b) c) d
          v_2 = (a (b c)) d
          v_3 = (a b) (c d)
          v_4 = a ((b c) d)
          v_5 = a (b (c d))
        """
        ab = cls.multiply(a, b)
        bc = cls.multiply(b, c)
        cd = cls.multiply(c, d)

        v1 = cls.multiply(cls.multiply(ab, c), d)
        v2 = cls.multiply(cls.multiply(a, bc), d)
        v3 = cls.multiply(ab, cd)
        v4 = cls.multiply(a, cls.multiply(bc, d))
        v5 = cls.multiply(a, cls.multiply(b, cd))

        return (v1, v2, v3, v4, v5)

    @classmethod
    def moufang_defects(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[float, float, float]:
        r"""
        Calcula los tres defectos de las identidades de Moufang (Zorn 1930,
        Moufang 1935) — condición necesaria y suficiente para que un loop
        multiplicativo sea alternativo:
          M_L (izquierda):  x(y(xz)) = ((xy)x)z
          M_R (derecha):    ((zx)y)x = z(x(yx))
          M_M (media):      (xy)(zx) = x((yz)x)
        Se satisfacen EXACTAMENTE en \mathbb{R}, \mathbb{C}, \mathbb{H}, \mathbb{O}
        (dim \le 8) y se rompen genéricamente en \mathbb{S}, \mathbb{P} (dim \ge 16),
        constituyendo un certificado espectral de la frontera de Hurwitz.
        """
        xz = cls.multiply(x, z)
        lhs_left = cls.multiply(x, cls.multiply(y, xz))
        rhs_left = cls.multiply(cls.multiply(cls.multiply(x, y), x), z)
        left_defect = KBNSummationKernel.norm(lhs_left - rhs_left)

        zx = cls.multiply(z, x)
        yx = cls.multiply(y, x)
        lhs_right = cls.multiply(cls.multiply(zx, y), x)
        rhs_right = cls.multiply(z, cls.multiply(x, yx))
        right_defect = KBNSummationKernel.norm(lhs_right - rhs_right)

        xy = cls.multiply(x, y)
        yz = cls.multiply(y, z)
        lhs_middle = cls.multiply(xy, zx)
        rhs_middle = cls.multiply(x, cls.multiply(yz, x))
        middle_defect = KBNSummationKernel.norm(lhs_middle - rhs_middle)

        return left_defect, right_defect, middle_defect

    @classmethod
    def left_multiplication_matrix(cls, p: np.ndarray) -> np.ndarray:
        r"""
        Operador de representación regular izquierda L_p \in \mathbb{R}^{32 \times 32},
        definido por L_p(y) = p \cdot y.
        """
        vec = cls._validate_algebra_vector(p, "p")
        dim = vec.size
        L_matrix = np.empty((dim, dim), dtype=np.float64)
        e_k = np.zeros(dim, dtype=np.float64)

        for col in range(dim):
            e_k[col] = 1.0
            L_matrix[:, col] = cls.multiply(vec, e_k)
            e_k[col] = 0.0

        return L_matrix

    @classmethod
    def right_multiplication_matrix(cls, p: np.ndarray) -> np.ndarray:
        r"""
        Operador de representación regular derecha R_p \in \mathbb{R}^{32 \times 32},
        definido por R_p(y) = y \cdot p. Generalmente distinto de L_p por la
        no conmutatividad de \mathbb{P}, indispensable para el análisis BILÁTERO
        del cono nulo \mathcal{N}(\mathbb{P}).
        """
        vec = cls._validate_algebra_vector(p, "p")
        dim = vec.size
        R_matrix = np.empty((dim, dim), dtype=np.float64)
        e_k = np.zeros(dim, dtype=np.float64)

        for col in range(dim):
            e_k[col] = 1.0
            R_matrix[:, col] = cls.multiply(e_k, vec)
            e_k[col] = 0.0

        return R_matrix

    @classmethod
    def commutator(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""Conmutador de Lie [a, b] = a b - b a."""
        return cls.multiply(a, b) - cls.multiply(b, a)

    @classmethod
    def jordan_product(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r"""Producto simétrico de Jordan a \circ b = \frac{1}{2}(a b + b a)."""
        return 0.5 * (cls.multiply(a, b) + cls.multiply(b, a))


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FASE 1 — OBSERVE + ORIENT (METROLOGÍA CD 32D, CACHÉ ESPECTRAL Y BANACH)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PathionicMetricObserver:
    r"""
    FASE 1: Observe + Orient.
    Categoría Functorial: \mathbf{Hilb}_{32} \times \mathbf{Hilb}_{32} \longrightarrow \mathbf{PathionicMetricsReport}.

    Ingesta sensorial en 32 dimensiones, ensamblaje de cuaternas y sedeniones,
    cálculo de normas euclídeas por sumación exacta de Shewchuk, evaluación de
    la fuga cuadrática, determinación del defecto de submultiplicatividad de
    Banach/Hurwitz, Y CONGELACIÓN del espectro singular extremal de los
    operadores de representación regular L_{\hat p}, R_{\hat p} — insumo que
    la Fase 3 reutilizará sin recómputo, por bilinealidad en el argumento.

    El método terminal `observe_metrics` es el punto de continuidad
    hacia la Fase 2 (`Phase2_PentagonalAssociatorCalculator.continue_from_metrics_report`).
    """

    __slots__ = ("_thresholds",)

    def __init__(self, thresholds: Optional[PathionicThresholds] = None) -> None:
        self._thresholds: Final[PathionicThresholds] = thresholds or PathionicThresholds()

    @property
    def thresholds(self) -> PathionicThresholds:
        return self._thresholds

    def _compute_operator_spectral_cache(
        self,
        vec: np.ndarray,
        norm_val: float,
        is_zero_state: bool,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, float, float]:
        r"""
        Resuelve, para el operando unitario \hat p = p / \|p\|, los operadores
        L_{\hat p}, R_{\hat p} y extrae sus valores singulares extremales junto
        con los vectores singulares derechos asociados a \sigma_{\min}
        (testigos del anulador aproximado). En una álgebra de composición
        (\mathbb{R}, \mathbb{C}, \mathbb{H}, \mathbb{O}), L_{\hat p} y R_{\hat p}
        son ISOMETRÍAS exactas: todos sus valores singulares valen 1. El
        defecto \sigma_{\max} - 1 mide la ruptura de esa propiedad en \mathbb{P}.
        """
        dim = vec.size
        if is_zero_state or norm_val <= self._thresholds.zero_norm_threshold:
            zero_w = np.zeros(dim, dtype=np.float64)
            zero_w.setflags(write=False)
            return 0.0, 0.0, 0.0, 0.0, zero_w, zero_w, 0.0, 0.0

        p_hat = vec / norm_val
        L_hat = CayleyDicksonAlgebra32.left_multiplication_matrix(p_hat)
        R_hat = CayleyDicksonAlgebra32.right_multiplication_matrix(p_hat)

        try:
            _, s_L, Vt_L = la.svd(L_hat, full_matrices=False, check_finite=True)
        except la.LinAlgError:
            _, s_L, Vt_L = np.linalg.svd(L_hat, full_matrices=False)

        try:
            _, s_R, Vt_R = la.svd(R_hat, full_matrices=False, check_finite=True)
        except la.LinAlgError:
            _, s_R, Vt_R = np.linalg.svd(R_hat, full_matrices=False)

        left_sigma_max = float(s_L[0])
        left_sigma_min = float(s_L[-1])
        right_sigma_max = float(s_R[0])
        right_sigma_min = float(s_R[-1])

        left_witness = np.ascontiguousarray(Vt_L[-1, :].copy())
        right_witness = np.ascontiguousarray(Vt_R[-1, :].copy())
        left_witness.setflags(write=False)
        right_witness.setflags(write=False)

        left_isometry_defect = abs(left_sigma_max - 1.0)
        right_isometry_defect = abs(right_sigma_max - 1.0)

        return (
            left_sigma_max,
            left_sigma_min,
            right_sigma_max,
            right_sigma_min,
            left_witness,
            right_witness,
            left_isometry_defect,
            right_isometry_defect,
        )

    def build_state(self, raw_vector: Sequence[float]) -> PathionicState:
        r"""
        Morfismo de condensación de estado:
          R^{32} \longrightarrow \mathbf{PathionicState}.
        Congela el tensor, evalúa invariantes hermíticos e hipercomplejos, y
        cachea el espectro singular extremal de L_{\hat p}, R_{\hat p}.
        """
        vec = _prepare_pathion_vector(raw_vector, "PathionicState.vector_rep")

        s1 = np.ascontiguousarray(vec[0:_SEDENION_DIM].copy(), dtype=np.float64)
        s2 = np.ascontiguousarray(vec[_SEDENION_DIM:_PATHION_DIM].copy(), dtype=np.float64)
        s1.setflags(write=False)
        s2.setflags(write=False)

        norm_val = KBNSummationKernel.norm(vec)
        if not math.isfinite(norm_val):
            raise PathionicNumericalSingularityError("Norma no finita calculada para el estado.")

        norm_sq = norm_val * norm_val if norm_val < _SQRT_DBL_MAX else math.inf
        real_comp = float(vec[0])
        imag_norm = KBNSummationKernel.norm(vec[1:]) if vec.size > 1 else 0.0

        quad_scalar, quad_leakage = CayleyDicksonAlgebra32.quadratic_form(vec)

        is_zero_state = norm_val <= self._thresholds.zero_norm_threshold
        is_unitary_state = _within_tolerance(
            norm_val,
            1.0,
            self._thresholds.absolute_tolerance,
            self._thresholds.relative_tolerance,
        )

        (
            left_sigma_max,
            left_sigma_min,
            right_sigma_max,
            right_sigma_min,
            left_witness,
            right_witness,
            left_iso_defect,
            right_iso_defect,
        ) = self._compute_operator_spectral_cache(vec, norm_val, is_zero_state)

        sha_hash = _sha256_of_arrays((vec, s1, s2))

        return PathionicState(
            vector_rep=vec,
            s1=s1,
            s2=s2,
            norm=norm_val,
            norm_squared=norm_sq,
            real_part=real_comp,
            imag_norm=imag_norm,
            quadratic_scalar=quad_scalar,
            quadratic_leakage=quad_leakage,
            is_unitary=is_unitary_state,
            is_zero=is_zero_state,
            left_sigma_max=left_sigma_max,
            left_sigma_min=left_sigma_min,
            right_sigma_max=right_sigma_max,
            right_sigma_min=right_sigma_min,
            left_zero_divisor_witness=left_witness,
            right_zero_divisor_witness=right_witness,
            left_isometry_defect=left_iso_defect,
            right_isometry_defect=right_iso_defect,
            sha256_hash=sha_hash,
        )

    def observe_metrics(
        self,
        A_vec: Sequence[float],
        B_vec: Sequence[float],
    ) -> PathionicMetricsReport:
        r"""
        MORFISMO TERMINAL DE LA FASE 1.

        Ejecuta el producto hipercomplejo pationiónico A \cdot B, mide las normas
        por sumación exacta de Shewchuk, audita la identidad de Hurwitz:
          \delta_H = \bigl| \|A \cdot B\| - \|A\| \|B\| \bigr|,
        y computa la pérdida exergética por desajuste de Banach.

        El objeto retornado es el argumento formal exigido por la Fase 2.
        """
        state_a = self.build_state(A_vec)
        state_b = self.build_state(B_vec)

        prod_vector = CayleyDicksonAlgebra32.multiply(state_a.vector_rep, state_b.vector_rep)
        product_state = self.build_state(prod_vector)

        log_exp = _log_norm(state_a.norm) + _log_norm(state_b.norm)
        expected_norm = _safe_exp(log_exp)
        product_norm = product_state.norm

        signed_defect, absolute_error, relative_error = _composition_defect(
            product_norm, expected_norm, _WILKINSON_FLOOR
        )

        scale = max(_WILKINSON_FLOOR, expected_norm) if math.isfinite(expected_norm) else _WILKINSON_FLOOR

        if expected_norm > self._thresholds.zero_norm_threshold and math.isfinite(expected_norm):
            composition_ratio = product_norm / expected_norm if math.isfinite(product_norm) else math.inf
        else:
            composition_ratio = 1.0 if product_norm <= self._thresholds.zero_norm_threshold else math.inf

        is_hurwitz_stable = math.isfinite(absolute_error) and (
            absolute_error <= self._thresholds.hurwitz_absolute + self._thresholds.hurwitz_relative * scale
        )

        is_banach_submultiplicative = math.isfinite(product_norm) and math.isfinite(expected_norm) and (
            product_norm <= expected_norm + self._thresholds.hurwitz_absolute + self._thresholds.hurwitz_relative * scale
        )

        exergy_loss = abs(product_norm * product_norm - expected_norm * expected_norm) if (
            math.isfinite(product_norm) and math.isfinite(expected_norm)
        ) else math.inf

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
            exergy_loss_hurwitz=exergy_loss,
            is_hurwitz_stable=is_hurwitz_stable,
            is_banach_submultiplicative=is_banach_submultiplicative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §G. FASE 2 — DECIDE (STASHEFF, MOUFANG, ALBERT Y HODGE-LAPLACE)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_PentagonalAssociatorCalculator(Phase1_PathionicMetricObserver):
    r"""
    FASE 2: Decide.
    Hereda ontológicamente de la Fase 1 (Observe + Orient).

    Categoría Functorial:
      \mathbf{PathionicMetricsReport} \times (R^{32})^{\times 3} \longrightarrow \mathbf{PathionicDecisionState}.

    Consume el reporte métrico terminal de la Fase 1, extiende la interacción
    al 4-símplex completo \Delta_4 mediante los estados P_3, P_4, P_5, y computa:
      1. Asociador trilateral y pentagonal de calibre.
      2. Diámetro del pentágono de Stasheff K_4.
      3. Defecto de flexibilidad de Albert [P_1, P_2, P_1] — identidad UNIVERSAL
         (se enfuerza como axioma; su ruptura grosera detona
         `PathionicAxiomViolationError`).
      4. Defectos de las tres identidades de Moufang y de alternatividad
         [P_1, P_1, P_2] — certificado espectral exacto de la frontera de
         Hurwitz (dim \le 8 alternativa vs. dim \ge 16 no-alternativa).
      5. Volumen tetradimensional del 4-símplex mediante Gram-SVD.
      6. Topología de Hodge sobre el grafo K_5: espectro del laplaciano
         (Gram compensado de Shewchuk), autovalor de Fiedler \lambda_2,
         hueco espectral, índice de Kirchhoff y disipación de Dirichlet.

    El método terminal `decide_from_metrics_report` genera el objeto inicial
    para la Fase 3.
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
        MORFISMO DE CONTINUACIÓN DE LA FASE 2.
        Toma el reporte terminal de la Fase 1 y calcula el reporte pentagonal.
        """
        return self.decide_from_metrics_report(
            metrics_report=metrics_report,
            p3_vec=p3_vec,
            p4_vec=p4_vec,
            p5_vec=p5_vec,
            pentagonal_threshold=pentagonal_threshold,
        ).pentagonal_report

    def compute_trilateral_associator(
        self,
        P1: PathionicState,
        P2: PathionicState,
        P3: PathionicState,
    ) -> np.ndarray:
        r"""Calcula [P_1, P_2, P_3] = (P_1 P_2) P_3 - P_1 (P_2 P_3)."""
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
        r"""Calcula A_5 = ((((P_1 P_2) P_3) P_4) P_5) - (P_1 (P_2 (P_3 (P_4 P_5))))."""
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
        r"""
        Calcula el diámetro métrico del politopo de Stasheff K_4:
          \mathrm{diam}(K_4) = \max_{1 \le i < j \le 5} \|v_i - v_j\|_2.
        """
        associations = CayleyDicksonAlgebra32.stasheff_associations(
            P1.vector_rep,
            P2.vector_rep,
            P3.vector_rep,
            P4.vector_rep,
        )
        max_dist = 0.0
        for i in range(len(associations)):
            for j in range(i + 1, len(associations)):
                diff = associations[i] - associations[j]
                dist = KBNSummationKernel.norm(diff)
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def _diagnose_simplex_and_hodge(
        self,
        states: Sequence[PathionicState],
    ) -> Tuple[float, float, float, float, float, float, int]:
        r"""
        Auditoría geométrica y espectral de Hodge sobre el 4-símplex \Delta_4 y su
        1-esqueleto K_5. Retorna:
          1. Número de condición afín del símplex.
          2. Volumen 4D exacto: V_4 = \frac{1}{24} \prod \sigma_k(E).
          3. Conectividad algebraica de Fiedler \lambda_2.
          4. Hueco espectral del Laplaciano \lambda_2 - \lambda_1.
          5. Índice de Kirchhoff R_K = n \sum_{k=2}^n \frac{1}{\lambda_k}.
          6. Disipación exergética de Dirichlet \mathcal{E}_D = \mathrm{Tr}(P^T L P).
          7. Multiplicidad del autovalor nulo (componentes conexas).
        """
        coords = np.vstack([s.vector_rep for s in states])
        num_vertices = len(states)

        mean_vec = np.zeros(coords.shape[1], dtype=np.float64)
        for c in range(coords.shape[1]):
            mean_vec[c] = KBNSummationKernel.sum(coords[:, c]) / float(num_vertices)
        centered = coords - mean_vec

        try:
            svals_centered = la.svdvals(centered, check_finite=True)
        except la.LinAlgError:
            svals_centered = np.linalg.svd(centered, compute_uv=False)

        if svals_centered.size == 0 or float(svals_centered[0]) <= _WILKINSON_FLOOR:
            cond_number = math.inf
        else:
            cutoff = max(_MACHINE_EPS * float(svals_centered[0]), _WILKINSON_FLOOR)
            pos_svals = svals_centered[svals_centered > cutoff]
            cond_number = float(svals_centered[0] / pos_svals[-1]) if pos_svals.size > 0 else math.inf

        origin = states[0].vector_rep
        edge_matrix = np.stack([states[k].vector_rep - origin for k in range(1, num_vertices)], axis=1)
        try:
            svals_edges = la.svdvals(edge_matrix, check_finite=True)
        except la.LinAlgError:
            svals_edges = np.linalg.svd(edge_matrix, compute_uv=False)

        if svals_edges.size < 4 or np.any(svals_edges <= _WILKINSON_FLOOR):
            simplex_vol = 0.0
        else:
            log_vol = float(np.sum(np.log(svals_edges))) - math.log(24.0)
            simplex_vol = _safe_exp(log_vol)

        norms = np.array([s.norm for s in states], dtype=np.float64)
        valid_mask = norms > self._thresholds.zero_norm_threshold

        if not np.all(valid_mask):
            return cond_number, simplex_vol, 0.0, 0.0, math.inf, 0.0, num_vertices

        normed_coords = coords / norms[:, np.newaxis]
        # Gram compensado de Shewchuk: evita cancelación catastrófica en n=5.
        cosine_matrix = np.clip(KBNSummationKernel.pairwise_gram(normed_coords), -1.0, 1.0)
        weights = 0.5 * (1.0 + cosine_matrix)
        np.fill_diagonal(weights, 0.0)

        degrees = np.sum(weights, axis=1)
        laplacian = np.diag(degrees) - weights
        laplacian = 0.5 * (laplacian + laplacian.T)

        try:
            eigvals = la.eigvalsh(laplacian, check_finite=True)
        except la.LinAlgError:
            eigvals = np.linalg.eigvalsh(laplacian)

        eigvals = np.sort(np.real(eigvals))
        nonneg_eigs = np.maximum(eigvals, 0.0)

        zero_tol = max(num_vertices * _MACHINE_EPS * float(nonneg_eigs[-1]), _WILKINSON_FLOOR)
        connected_components = int(np.sum(nonneg_eigs <= zero_tol))
        connected_components = max(1, min(connected_components, num_vertices))

        fiedler_value = float(nonneg_eigs[1]) if nonneg_eigs.size >= 2 else 0.0
        spectral_gap = _clip_nonnegative(fiedler_value - float(nonneg_eigs[0]))

        if fiedler_value > self._thresholds.laplacian_fiedler_min:
            kirchhoff = float(num_vertices * np.sum(1.0 / nonneg_eigs[1:]))
        else:
            kirchhoff = math.inf

        dirichlet_exergy = float(np.trace(coords.T @ laplacian @ coords))
        dirichlet_exergy = _clip_nonnegative(dirichlet_exergy)

        return (
            cond_number,
            simplex_vol,
            fiedler_value,
            spectral_gap,
            kirchhoff,
            dirichlet_exergy,
            connected_components,
        )

    def _pentagonal_report_from_states(
        self,
        states: Sequence[PathionicState],
        pentagonal_threshold: Optional[float],
    ) -> PathionicPentagonalReport:
        if len(states) != 5:
            raise PathionicEngineError("La evaluación pentagonal exige exactamente 5 estados pationiónicos.")

        P1, P2, P3, P4, P5 = states

        v_assoc3 = self.compute_trilateral_associator(P1, P2, P3)
        trilateral_norm = KBNSummationKernel.norm(v_assoc3)

        v_assoc5 = self.compute_pentagonal_associator(P1, P2, P3, P4, P5)
        pentagonal_norm = KBNSummationKernel.norm(v_assoc5)

        stasheff_diam = self._stasheff_diameter(P1, P2, P3, P4)

        # --- Axioma UNIVERSAL de flexibilidad de Albert (1942) ---
        flex_vec = CayleyDicksonAlgebra32.associator(P1.vector_rep, P2.vector_rep, P1.vector_rep)
        flexibility_defect = KBNSummationKernel.norm(flex_vec)

        flex_scale = max(P1.norm * P1.norm * P2.norm, _WILKINSON_FLOOR)
        flex_bound = self._thresholds.flexibility_axiom_tolerance * (1.0 + flex_scale)
        is_flexibility_axiom_satisfied = flexibility_defect <= flex_bound

        if flexibility_defect > 1.0e4 * flex_bound and flexibility_defect > 1.0e-3:
            raise PathionicAxiomViolationError(
                "Violación catastrófica de la identidad de flexibilidad de Albert: "
                f"||[P1,P2,P1]|| = {flexibility_defect:.6e} excede en más de cuatro "
                f"órdenes de magnitud la tolerancia axiomática {flex_bound:.6e}. "
                "Esta identidad es UNIVERSAL en toda álgebra de Cayley-Dickson "
                "(independiente de asociatividad/alternatividad); su ruptura grosera "
                "certifica un defecto estructural del núcleo de multiplicación."
            )

        # --- Certificado espectral de Moufang / alternatividad (dim >= 16 lo rompe) ---
        alt1 = KBNSummationKernel.norm(
            CayleyDicksonAlgebra32.associator(P1.vector_rep, P1.vector_rep, P2.vector_rep)
        )
        alt2 = KBNSummationKernel.norm(
            CayleyDicksonAlgebra32.associator(P1.vector_rep, P2.vector_rep, P2.vector_rep)
        )
        alternativity_defect = max(alt1, alt2)

        moufang_left, moufang_right, moufang_middle = CayleyDicksonAlgebra32.moufang_defects(
            P1.vector_rep, P2.vector_rep, P3.vector_rep
        )
        moufang_scale = max(P1.norm * P2.norm * P3.norm, 1.0)
        is_alternative_regime = (
            max(moufang_left, moufang_right, moufang_middle, alternativity_defect)
            <= self._thresholds.alternativity_tolerance * moufang_scale
        )

        # --- Métricas relativas del asociador pentagonal ---
        log_prod_norms = sum(_log_norm(max(s.norm, _WILKINSON_FLOOR)) for s in states)
        log_norm5 = _log_norm(pentagonal_norm)
        pentagonal_relative_norm = _safe_exp(log_norm5 - log_prod_norms)

        # Parámetro de orden de frustración SATURANTE, acotado en [0, 1):
        #   \Phi = r / (1 + r), r = pentagonal_relative_norm.
        # Análogo a la relación rapidez-velocidad; evita divergencias del cociente crudo.
        if math.isfinite(pentagonal_relative_norm):
            frustration_index = pentagonal_relative_norm / (1.0 + pentagonal_relative_norm)
        else:
            frustration_index = 1.0

        th_abs = float(
            self._thresholds.pentagonal_absolute
            if pentagonal_threshold is None
            else pentagonal_threshold
        )
        if not math.isfinite(th_abs) or th_abs < 0.0:
            raise PathionicEngineError("El umbral pentagonal debe ser finito y no negativo.")

        log_rel_th = math.log(self._thresholds.pentagonal_relative) + log_prod_norms
        th_used = max(th_abs, _safe_exp(log_rel_th))

        is_stable = math.isfinite(pentagonal_norm) and (
            pentagonal_norm <= th_abs or log_norm5 <= log_rel_th
        )

        (
            cond_num,
            simp_vol,
            fiedler,
            gap,
            kirchhoff,
            dirichlet,
            comp_count,
        ) = self._diagnose_simplex_and_hodge(states)

        diag_messages: List[str] = []
        if not math.isfinite(pentagonal_norm):
            diag_messages.append("Singularidad no finita en curvatura A_5.")
        elif is_stable:
            diag_messages.append("Calibre pentagonal en régimen de estabilidad homotópica.")
        else:
            diag_messages.append("Frustración pentagonal crítica: curvatura de calibre fuera de norma.")

        if stasheff_diam > th_used:
            diag_messages.append("Diámetro de Stasheff K_4 hipertrofiado: ruptura de coherencia A_infinito.")

        if not is_alternative_regime:
            diag_messages.append(
                "Régimen no-alternativo confirmado (identidades de Moufang violadas): "
                "comportamiento estructural esperado para dim >= 16."
            )

        if not is_flexibility_axiom_satisfied:
            diag_messages.append(
                "Alerta axiomática: defecto de flexibilidad de Albert por encima de "
                "la tolerancia nominal (posible degradación numérica del núcleo CD)."
            )

        if cond_num > self._thresholds.condition_number_limit:
            diag_messages.append("Degeneración colineal en el 4-símplex (número de condición crítico).")

        if simp_vol <= self._thresholds.simplex_volume_min:
            diag_messages.append("Colapso del volumen 4D afín del 4-símplex (símplex plano).")

        if fiedler < self._thresholds.laplacian_fiedler_min:
            diag_messages.append("Estrangulamiento espectral de Fiedler: red K_5 al borde de partición.")

        if comp_count > 1:
            diag_messages.append(f"Ruptura topológica: {comp_count} componentes disconexas detectadas.")

        diagnosis_str = " | ".join(diag_messages) if diag_messages else "Régimen pentagonal nominal."

        return PathionicPentagonalReport(
            trilateral_associator_norm=trilateral_norm,
            pentagonal_associator_norm=pentagonal_norm,
            pentagonal_relative_norm=pentagonal_relative_norm,
            frustration_index=frustration_index,
            stasheff_pentagon_diameter=stasheff_diam,
            flexibility_defect=flexibility_defect,
            alternativity_defect=alternativity_defect,
            moufang_left_defect=moufang_left,
            moufang_right_defect=moufang_right,
            moufang_middle_defect=moufang_middle,
            is_alternative_regime=is_alternative_regime,
            is_flexibility_axiom_satisfied=is_flexibility_axiom_satisfied,
            pentagonal_threshold_used=th_used,
            is_pentagonal_stable=is_stable,
            simplex_condition_number=cond_num,
            simplex_volume=simp_vol,
            laplacian_connectivity=fiedler,
            laplacian_spectral_gap=gap,
            kirchhoff_index=kirchhoff,
            dirichlet_exergy=dirichlet,
            estimated_connected_components=comp_count,
            diagnosis=diagnosis_str,
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
        MORFISMO TERMINAL DE LA FASE 2.

        Construye los estados para los tres vértices restantes P_3, P_4, P_5,
        sintetiza el reporte pentagonal y empaqueta el `PathionicDecisionState`.
        Este objeto es la entrada formal única de la Fase 3.
        """
        P3 = self.build_state(p3_vec)
        P4 = self.build_state(p4_vec)
        P5 = self.build_state(p5_vec)

        all_states = (
            metrics_report.state_a,
            metrics_report.state_b,
            P3,
            P4,
            P5,
        )

        pentagonal_report = self._pentagonal_report_from_states(
            states=all_states,
            pentagonal_threshold=pentagonal_threshold,
        )

        return PathionicDecisionState(
            metrics_report=metrics_report,
            pentagonal_report=pentagonal_report,
            states=all_states,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §H. FASE 3 — ACT (CONO NULO BILÁTERO, SELLADO CRIPTOGRÁFICO FRAMED)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_PathionicNullConeEvaluator(Phase2_PentagonalAssociatorCalculator):
    r"""
    FASE 3: Act.
    Hereda ontológicamente de la Fase 2 (Decide).

    Categoría Functorial:
      \mathbf{PathionicDecisionState} \longrightarrow \mathbf{PathionicEngineState}.

    Consume el estado decisional de la Fase 2 y resuelve el análisis espectral
    BILÁTERO del cono nulo \mathcal{N}(\mathbb{P}) LEYENDO LA CACHÉ generada en
    la Fase 1 (sin recomputar SVDs: \sigma(L_p) = \|p\|\sigma(L_{\hat p}) por
    bilinealidad), extrae los vectores testigo izquierdo/derecho, audita
    conmutadores y productos de Jordan, calcula la fricción exergética
    disipada y genera el sello criptográfico SHA-256 con *framing* canónico
    (libre de ambigüedad de concatenación).

    El método terminal `execute_pentagonal_audit` cierra la arquitectura functorial.
    """

    __slots__ = ()

    def continue_from_decision_state(
        self,
        decision: PathionicDecisionState,
    ) -> PathionicEngineState:
        r"""
        MORFISMO DE CONTINUACIÓN DE LA FASE 3.
        Toma el objeto terminal de la Fase 2 y ejecuta la acción completa.
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
            all_states=decision.states,
            fpu_execution_time_ms=elapsed_ms,
            cryptographic_seal=cryptographic_seal,
            engine_version=__version__,
        )

    def evaluate_null_cone(
        self,
        P1_state: PathionicState,
        P2_state: PathionicState,
    ) -> NullConeReport:
        r"""
        Audita el estrangulamiento BILÁTERO por divisores de cero en
        \mathcal{N}_L(\mathbb{P}) \cup \mathcal{N}_R(\mathbb{P}) y la fricción
        exergética entre dos agentes:
          \chi_{\mathrm{null}} = \bigl| \|P_1 \cdot P_2\| - \|P_1\| \|P_2\| \bigr|,
          d_{\mathcal{N}} = \max\left(0, 1 - \frac{\|P_1 \cdot P_2\|}{\|P_1\| \|P_2\|}\right).

        Los valores singulares extremales de L_p, R_p se OBTIENEN de la caché
        congelada en la Fase 1 (`PathionicState.left_sigma_min`, etc.),
        reescalados por la norma física del estado gracias a la bilinealidad
        del operador de representación regular en su argumento — sin
        recomputar ninguna SVD en esta fase.
        """
        prod_vec = CayleyDicksonAlgebra32.multiply(P1_state.vector_rep, P2_state.vector_rep)
        product_norm = KBNSummationKernel.norm(prod_vec)

        comm_vec = CayleyDicksonAlgebra32.commutator(P1_state.vector_rep, P2_state.vector_rep)
        commutator_norm = KBNSummationKernel.norm(comm_vec)

        jordan_vec = CayleyDicksonAlgebra32.jordan_product(P1_state.vector_rep, P2_state.vector_rep)
        jordan_norm = KBNSummationKernel.norm(jordan_vec)

        log_exp = _log_norm(P1_state.norm) + _log_norm(P2_state.norm)
        expected_norm = _safe_exp(log_exp)

        _, absolute_friction, relative_defect = _composition_defect(
            product_norm, expected_norm, _WILKINSON_FLOOR
        )

        is_trivial = P1_state.is_zero or P2_state.is_zero

        if math.isfinite(expected_norm) and math.isfinite(product_norm) and expected_norm > self._thresholds.zero_norm_threshold:
            null_depth = max(0.0, 1.0 - (product_norm / expected_norm))
        elif not is_trivial and math.isfinite(product_norm) and product_norm <= self._thresholds.null_absolute:
            null_depth = 1.0
        else:
            null_depth = 0.0

        # Reutilización BILÁTERA de la caché espectral de la Fase 1 (sin recómputo de SVD).
        sigma_min_left_p1 = P1_state.left_sigma_min * P1_state.norm
        sigma_min_left_p2 = P2_state.left_sigma_min * P2_state.norm
        sigma_min_right_p1 = P1_state.right_sigma_min * P1_state.norm
        sigma_min_right_p2 = P2_state.right_sigma_min * P2_state.norm

        penetration_bound = self._thresholds.null_absolute + self._thresholds.null_relative * expected_norm
        near_zero_prod = product_norm <= penetration_bound

        is_penetrated = (
            not is_trivial
            and math.isfinite(product_norm)
            and (near_zero_prod or null_depth >= self._thresholds.null_depth_threshold)
        )

        ratio_limit = self._thresholds.zero_divisor_spectral_ratio_limit
        is_left_candidate = bool(
            P1_state.left_sigma_min <= ratio_limit or P2_state.left_sigma_min <= ratio_limit
        )
        is_right_candidate = bool(
            P1_state.right_sigma_min <= ratio_limit or P2_state.right_sigma_min <= ratio_limit
        )

        spectral_friction_exergy = 0.25 * (
            abs(P1_state.norm - sigma_min_left_p1)
            + abs(P2_state.norm - sigma_min_left_p2)
            + abs(P1_state.norm - sigma_min_right_p1)
            + abs(P2_state.norm - sigma_min_right_p2)
        )

        return NullConeReport(
            product_norm=product_norm,
            expected_norm=expected_norm,
            absolute_friction=absolute_friction,
            relative_defect=relative_defect,
            null_depth=null_depth,
            sigma_min_left_p1=sigma_min_left_p1,
            sigma_min_left_p2=sigma_min_left_p2,
            sigma_min_right_p1=sigma_min_right_p1,
            sigma_min_right_p2=sigma_min_right_p2,
            zero_divisor_witness_p1_left=P1_state.left_zero_divisor_witness,
            zero_divisor_witness_p1_right=P1_state.right_zero_divisor_witness,
            commutator_norm=commutator_norm,
            jordan_product_norm=jordan_norm,
            is_trivial_null=is_trivial,
            is_null_cone_penetrated=is_penetrated,
            is_left_zero_divisor_candidate=is_left_candidate,
            is_right_zero_divisor_candidate=is_right_candidate,
            spectral_friction_exergy=spectral_friction_exergy,
        )

    def _compute_cryptographic_seal(
        self,
        states: Sequence[PathionicState],
        metrics_report: PathionicMetricsReport,
        pentagonal_report: PathionicPentagonalReport,
        null_report: NullConeReport,
    ) -> str:
        """
        Genera el sello inmutable SHA-256 de la sesión empacando, mediante
        *length-prefixed framing* canónico (§B `_frame_bytes`), todos los
        invariantes de las 3 fases. El framing elimina la ambigüedad de
        concatenación propia de una construcción Merkle-Damgård ingenua.
        """
        hasher = hashlib.sha256()
        hasher.update(_frame_bytes(b"ENGINE_VERSION", __version__.encode("utf-8")))

        for field in PathionicThresholds.__dataclass_fields__:
            payload = _canonical_float_bytes(float(getattr(self._thresholds, field)))
            hasher.update(_frame_bytes(field.encode("utf-8"), payload))

        for idx, s in enumerate(states):
            tag = f"STATE_{idx}".encode("ascii")
            payload = (
                np.asarray(s.vector_rep, dtype="<f8").tobytes(order="C")
                + s.sha256_hash.encode("ascii")
            )
            hasher.update(_frame_bytes(tag, payload))

        scalar_chain: Tuple[Tuple[str, float], ...] = (
            ("metrics.product_norm", metrics_report.product_norm),
            ("metrics.expected_norm", metrics_report.expected_norm),
            ("metrics.hurwitz_absolute_error", metrics_report.hurwitz_absolute_error),
            ("metrics.hurwitz_relative_error", metrics_report.hurwitz_relative_error),
            ("metrics.hurwitz_signed_defect", metrics_report.hurwitz_signed_defect),
            ("metrics.composition_ratio", metrics_report.composition_ratio),
            ("metrics.exergy_loss_hurwitz", metrics_report.exergy_loss_hurwitz),
            ("metrics.is_hurwitz_stable", float(metrics_report.is_hurwitz_stable)),
            ("metrics.is_banach_submultiplicative", float(metrics_report.is_banach_submultiplicative)),
            ("pent.trilateral_associator_norm", pentagonal_report.trilateral_associator_norm),
            ("pent.pentagonal_associator_norm", pentagonal_report.pentagonal_associator_norm),
            ("pent.pentagonal_relative_norm", pentagonal_report.pentagonal_relative_norm),
            ("pent.frustration_index", pentagonal_report.frustration_index),
            ("pent.stasheff_pentagon_diameter", pentagonal_report.stasheff_pentagon_diameter),
            ("pent.flexibility_defect", pentagonal_report.flexibility_defect),
            ("pent.alternativity_defect", pentagonal_report.alternativity_defect),
            ("pent.moufang_left_defect", pentagonal_report.moufang_left_defect),
            ("pent.moufang_right_defect", pentagonal_report.moufang_right_defect),
            ("pent.moufang_middle_defect", pentagonal_report.moufang_middle_defect),
            ("pent.is_alternative_regime", float(pentagonal_report.is_alternative_regime)),
            ("pent.is_flexibility_axiom_satisfied", float(pentagonal_report.is_flexibility_axiom_satisfied)),
            ("pent.pentagonal_threshold_used", pentagonal_report.pentagonal_threshold_used),
            ("pent.is_pentagonal_stable", float(pentagonal_report.is_pentagonal_stable)),
            ("pent.simplex_condition_number", pentagonal_report.simplex_condition_number),
            ("pent.simplex_volume", pentagonal_report.simplex_volume),
            ("pent.laplacian_connectivity", pentagonal_report.laplacian_connectivity),
            ("pent.laplacian_spectral_gap", pentagonal_report.laplacian_spectral_gap),
            ("pent.kirchhoff_index", pentagonal_report.kirchhoff_index),
            ("pent.dirichlet_exergy", pentagonal_report.dirichlet_exergy),
            ("pent.estimated_connected_components", float(pentagonal_report.estimated_connected_components)),
            ("null.product_norm", null_report.product_norm),
            ("null.expected_norm", null_report.expected_norm),
            ("null.absolute_friction", null_report.absolute_friction),
            ("null.relative_defect", null_report.relative_defect),
            ("null.null_depth", null_report.null_depth),
            ("null.sigma_min_left_p1", null_report.sigma_min_left_p1),
            ("null.sigma_min_left_p2", null_report.sigma_min_left_p2),
            ("null.sigma_min_right_p1", null_report.sigma_min_right_p1),
            ("null.sigma_min_right_p2", null_report.sigma_min_right_p2),
            ("null.commutator_norm", null_report.commutator_norm),
            ("null.jordan_product_norm", null_report.jordan_product_norm),
            ("null.is_trivial_null", float(null_report.is_trivial_null)),
            ("null.is_null_cone_penetrated", float(null_report.is_null_cone_penetrated)),
            ("null.is_left_zero_divisor_candidate", float(null_report.is_left_zero_divisor_candidate)),
            ("null.is_right_zero_divisor_candidate", float(null_report.is_right_zero_divisor_candidate)),
            ("null.spectral_friction_exergy", null_report.spectral_friction_exergy),
        )

        for name, scalar in scalar_chain:
            hasher.update(_frame_bytes(name.encode("ascii"), _canonical_float_bytes(scalar)))

        hasher.update(
            _frame_bytes(
                b"WITNESS_LEFT",
                np.asarray(null_report.zero_divisor_witness_p1_left, dtype="<f8").tobytes(order="C"),
            )
        )
        hasher.update(
            _frame_bytes(
                b"WITNESS_RIGHT",
                np.asarray(null_report.zero_divisor_witness_p1_right, dtype="<f8").tobytes(order="C"),
            )
        )

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
        MORFISMO TERMINAL GLOBAL DEL MOTOR PATIONIÓNICO.

        Ejecuta la orquestación soberana de las tres fases anidadas sin
        re-instanciaciones ni recómputos redundantes (incluida la reutilización
        BILÁTERA del espectro singular congelado en Fase 1 durante la Fase 3):
          \mathbf{P}_1, \mathbf{P}_2 \xrightarrow{\text{Fase 1: Observe}} \mathbf{MetricsReport}
          \xrightarrow{\text{Fase 2: Decide}(\mathbf{P}_3, \mathbf{P}_4, \mathbf{P}_5)} \mathbf{DecisionState}
          \xrightarrow{\text{Fase 3: Act}} \mathbf{EngineState}.
        """
        t_start = time.perf_counter()

        # FASE 1: Observe + Orient
        metrics = self.observe_metrics(contractor_P1, subcontractor_P2)

        # FASE 2: Decide (Encadenamiento formal directo)
        decision = self.decide_from_metrics_report(
            metrics_report=metrics,
            p3_vec=supplier_P3,
            p4_vec=interventor_P4,
            p5_vec=entity_P5,
            pentagonal_threshold=pentagonal_threshold,
        )

        # FASE 3: Act (Cierre functorial)
        engine_state = self.continue_from_decision_state(decision)

        total_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        engine_state = replace(engine_state, fpu_execution_time_ms=total_elapsed_ms)

        logger.debug(
            "Auditoría Soberana 32D ejecutada en %.4f ms | "
            "Hurwitz: %s | Pentagonal: %s | Régimen Alternativo: %s | "
            "Cono Nulo Penetrado: %s | Seal: %s",
            total_elapsed_ms,
            metrics.is_hurwitz_stable,
            decision.pentagonal_report.is_pentagonal_stable,
            decision.pentagonal_report.is_alternative_regime,
            engine_state.null_report.is_null_cone_penetrated,
            engine_state.cryptographic_seal[:12],
        )

        return engine_state


# ═══════════════════════════════════════════════════════════════════════════════
# §I. FACHADA SOBERANA DEL MOTOR
# ═══════════════════════════════════════════════════════════════════════════════
# En estricta concordancia con el principio de anidación ontológica:
# Phase3 ⊏ Phase2 ⊏ Phase1 (Act hereda de Decide, que hereda de Observe).
# La fachada pública soberana es directamente la Fase 3 completada.
PathionicDependencyEngine = Phase3_PathionicNullConeEvaluator