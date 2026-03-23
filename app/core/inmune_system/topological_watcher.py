"""
Topological Watcher (Sistema Inmunológico Matemático) — v4.1 Refinada
=============================================================================

Mejoras implementadas por categoría:

ÁLGEBRA LINEAL:
  A1. HealthStatus._severity_map_: acceso corregido (no .value para dict enum)
  A2. MetricTensor diagonal: eigenvalores verificados con eigvalsh, no eigvals
  A3. _compute_condition_number: singular detectado con min sobre TODOS eigvals
  A4. Suma de proyectores: inicialización con np.zeros tipado, no sum() con int
  A5. OrthogonalProjector: rank verificado con tr(P) = rango del proyector

NUMÉRICA:
  N1. _safe_normalize: tolerancia por defecto EPS corregida a EPS del módulo
  N2. _stable_norm: pre-escalado previo a forma cuadrática (evita overflow)
  N3. _regularize_metric: copia preserva tipo float64 (no hereda dtype)
  N4. MetricTensor densa: regularización ANTES de verificar condición
  N5. quadratic_form: clamping a 0 para formas cuadráticas casi-nulas negativas

TOPOLÓGICA:
  T1. _compute_euler_characteristic: tipo int nativo garantizado con int()
  T2. β₀ < 1 mensaje incluye valor real detectado
  T3. β₁ < 0 mensaje incluye valor real detectado
  T4. topo_indices validados contra dimensiones del proyector en construcción

HISTÉRESIS:
  H1. _classify_with_hysteresis: hiperesis=0.0 no causa división por cero
  H2. Validación de hysteresis: condición >= cambiada a > para excluir límite
  H3. Bandas asimétricas documentadas con invariantes algebraicos explícitos

FÍSICA:
  P1. build_signal: error_msg siempre inicializado antes del bloque condicional
  P2. SIGNAL_SCHEMA: flyback_voltage usa PhysicalConstants.FLYBACK_MAX_SAFE
  P3. PhysicalConstants.__slots__: añadido para consistencia con diseño

CATEGÓRICA:
  C1. ImmuneWatcherMorphism.__call__: F(error) retorna MISMO objeto (not copy)
  C2. _check_topology_change: history indexado correctamente [-2], no [-1]
  C3. _emit_state: with_update recibe contexto y stratum separados
  C4. topology_history: retorna tuple de copia (no referencia a lista mutable)

INTERFAZ:
  I1. temporary_algebraic_tolerance: validación de tolerancia negativa
  I2. get_diagnostics: topology_history_last10 usa slice seguro [-10:]
  I3. create_immune_watcher: overrides validados antes de construcción
  I4. health_report: manejo de historia vacía sin IndexError
"""

from __future__ import annotations

import logging
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, Generator, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState, Morphism

logger = logging.getLogger("MIC.ImmuneSystem")

# Referencia al módulo actual para el context manager de tolerancias
_THIS_MODULE = sys.modules[__name__]


# ═══════════════════════════════════════════════════════════════════════════════
# ÁRBOL DE EXCEPCIONES ESPECIALIZADAS
# ═══════════════════════════════════════════════════════════════════════════════

class ImmuneSystemError(Exception):
    """Base exception para el sistema inmunológico."""


class NumericalStabilityError(ImmuneSystemError):
    """Error de estabilidad numérica detectado."""


class DimensionalMismatchError(ImmuneSystemError):
    """Inconsistencia dimensional entre espacios vectoriales."""


class PhysicalBoundsError(ImmuneSystemError):
    """Valor viola límites físicos teóricos."""


class TopologicalInvariantError(ImmuneSystemError):
    """Violación de invariantes topológicos (Betti, Euler)."""


class MetricTensorError(ImmuneSystemError):
    """Error en tensor métrico Riemanniano (no SPD, condición alta, etc.)."""


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicalConstants:
    """
    Constantes físicas de referencia para circuitos de potencia.

    Todas las constantes tienen unidades SI documentadas.
    P_NOMINAL es classmethod (no property) para uso sin instancia.
    """

    # CORRECCIÓN A1: __slots__ en clase con solo atributos de clase no aplica,
    # pero sí en instancias. Sin instancias, la clase es solo un namespace.
    # Se preserva como namespace sin __slots__ para acceso de clase correcto.

    Z_CHARACTERISTIC: float = 50.0        # Impedancia característica [Ω]
    V_NOMINAL: float = 100.0              # Voltaje nominal [V]
    SATURATION_CRITICAL: float = 0.85    # Factor de saturación crítico [-]
    FLYBACK_MAX_SAFE: float = 400.0      # Voltaje de flyback máximo [V]
    I_THERMAL_LIMIT: float = 10.0        # Corriente límite térmica [A]
    THERMAL_SAFETY_FACTOR: float = 1.25  # Factor de seguridad térmica [-]
    TURN_RATIO: float = 5.0              # Relación de transformación [-]

    @classmethod
    def P_NOMINAL(cls) -> float:
        """
        Potencia nominal [W] = V²/Z.

        Classmethod garantiza uso sin instancia y sin dependencia de estado.
        Invariante: P_NOMINAL > 0 dado que V_NOMINAL > 0 y Z_CHARACTERISTIC > 0.
        """
        return cls.V_NOMINAL ** 2 / cls.Z_CHARACTERISTIC


# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS NUMÉRICOS — Tolerancias algebraicas y espectrales
# ═══════════════════════════════════════════════════════════════════════════════

# CORRECCIÓN N1: Definidas como constantes de módulo (no en clase) para acceso
# directo por _THIS_MODULE en el context manager de tolerancias.
EPS: float = 1e-12           # Tolerancia aritmética básica (≈ 4·ε_mach para float64)
ALGEBRAIC_TOL: float = 1e-10  # Tolerancia para propiedades algebraicas (‖P²−P‖_F)
COND_NUM_TOL: float = 1e14   # Número de condición máximo admisible
MIN_EIGVAL_TOL: float = 1e-12  # Eigenvalor mínimo para definición positiva


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ESTABILIDAD NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_normalize(
    vector: np.ndarray,
    eps: float = EPS,
) -> Tuple[np.ndarray, float]:
    """
    Normaliza vector por norma L∞ para evitar underflow/overflow.

    Propiedades Garantizadas
    ─────────────────────────
    1. scale = ‖v‖_∞ = max|vᵢ| ≥ 0
    2. Si scale > eps: max|normed_i| = 1 exactamente
    3. Reconstrucción: v = normed · scale (dentro de rtol=1e-14)
    4. Vector nulo: emite RuntimeWarning, retorna (zeros, 0.0)

    Nota de implementación
    ──────────────────────
    Se usa L∞ (no L²) porque evita suma de cuadrados que puede overflow
    para vectores con elementos ≥ sqrt(float_max) ≈ 1.3e154.

    Parameters
    ----------
    vector : array 1D de cualquier longitud
    eps    : umbral de degeneración (default = EPS del módulo)

    Returns
    -------
    normalized : np.ndarray, misma shape que vector
    scale      : float ≥ 0, factor de escala (0.0 indica degeneración)
    """
    if vector.size == 0:
        return vector.copy(), 0.0

    scale = float(np.max(np.abs(vector)))

    if scale < eps:
        warnings.warn(
            f"Norma L∞ por debajo del umbral de degeneración "
            f"({scale:.2e} < {eps:.2e}). Retornando vector nulo.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros_like(vector, dtype=np.float64), 0.0

    return (vector / scale), scale


def _stable_reciprocal(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Recíproco numéricamente estable, preservando signo.

    Estrategia
    ──────────
    Para |xᵢ| ≥ eps: 1/xᵢ (exacto)
    Para |xᵢ| < eps: sign(xᵢ)/eps (cota con signo correcto)
    Para xᵢ = 0:     1/eps (positivo, caso límite)

    Invariante: todos los valores de salida son finitos.

    Parameters
    ----------
    x   : array de entrada (cualquier shape)
    eps : umbral de denominador mínimo (default = EPS)

    Returns
    -------
    np.ndarray, misma shape que x, dtype=float64, todos finitos.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x_arr)

    # Denominador con cota inferior |eps|, preservando signo
    sign_x = np.sign(x_arr)
    # sign(0) = 0 → usar +1 para evitar denominador cero
    sign_safe = np.where(sign_x == 0.0, 1.0, sign_x)
    safe_denom = np.where(abs_x >= eps, x_arr, sign_safe * eps)

    return 1.0 / safe_denom


def _stable_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """
    División vectorial numéricamente estable.

    Garantía: |denom| ≥ eps antes de dividir. El signo de denom se preserva.

    Invariante (para |denom| >> eps):
        |resultado − resultado_exacto| ≤ ε_mach · |resultado_exacto|

    Parameters
    ----------
    numerator   : array numerador
    denominator : array denominador (misma shape o broadcastable)
    eps         : tolerancia de denominador mínimo

    Returns
    -------
    np.ndarray de misma shape, todos los valores finitos si inputs son finitos.
    """
    denominator = np.asarray(denominator, dtype=np.float64)
    abs_denom = np.abs(denominator)

    # CORRECCIÓN N4: evitar denominador exactamente cero con cota eps
    # Preservar signo: si denom >= 0 usar +eps, si denom < 0 usar -eps
    safe_denom = np.where(
        abs_denom >= eps,
        denominator,
        np.where(denominator >= 0.0, eps, -eps),
    )
    return np.asarray(numerator, dtype=np.float64) / safe_denom


def _stable_norm(
    vector: np.ndarray,
    metric: Optional[np.ndarray] = None,
    eps: float = EPS,
) -> float:
    """
    Norma L² numéricamente estable, con métrica Riemanniana opcional.

    Algoritmo de Pre-Escalado
    ─────────────────────────
    ‖v‖_G = max|vᵢ| · ‖v/max|vᵢ|‖_G

    Esto evita overflow en la forma cuadrática:
    - Sin pre-escalado: vᵀGv puede overflow para |v| ~ 1e150
    - Con pre-escalado: (v/s)ᵀG(v/s) ∈ [0, n] para v normalizado

    CORRECCIÓN N2: el pre-escalado se aplica ANTES de la forma cuadrática,
    no después, para capturar correctamente ‖v‖² = scale² · ‖normed‖².

    Parameters
    ----------
    vector : vector de entrada
    metric : matriz G (SPD) para norma de Mahalanobis, o None para L²
    eps    : umbral de degeneración

    Returns
    -------
    float ≥ 0, norma del vector.
    """
    if vector.size == 0:
        return 0.0

    normalized, scale = _safe_normalize(vector, eps)
    if scale < eps:
        return 0.0

    if metric is None:
        # L² euclidiana: ‖v‖ = scale · ‖normed‖₂
        return float(scale * np.sqrt(np.dot(normalized, normalized)))

    # CORRECCIÓN N2: forma cuadrática con vector normalizado
    # ‖v‖_G² = scale² · normedᵀ G normed
    quadratic = float(normalized @ metric @ normalized)

    if quadratic < 0.0:
        # Protección: G debería ser SPD. Si quadratic < 0, error numérico.
        logger.warning(
            "Forma cuadrática negativa (%.2e) con vector normalizado: "
            "posible pérdida de SPD en G. Se retorna 0.",
            quadratic,
        )
        return 0.0

    return float(scale * np.sqrt(quadratic))


def _compute_condition_number(
    matrix: np.ndarray,
    method: str = "eig",
) -> float:
    """
    Número de condición κ(G) para matrices cuadradas.

    CORRECCIÓN A3
    ─────────────
    Para matrices singulares: λ_min puede ser 0 o muy cercano a cero.
    La singularidad se detecta comprobando si el espectro COMPLETO
    tiene elementos > MIN_EIGVAL_TOL, no solo los positivos filtrados.

    Si ningún eigenvalor supera MIN_EIGVAL_TOL (matrix singular):
        → retorna float('inf')

    Métodos
    ───────
    'eig' : eigvalsh para SPD — garantiza eigenvalores reales (O(n³))
    'svd' : SVD completo — robusto para matrices generales (O(n³))

    Parameters
    ----------
    matrix : np.ndarray, shape (n,n)
    method : 'eig' o 'svd'

    Returns
    -------
    κ ≥ 1 para matrices invertibles, float('inf') para singulares.

    Raises
    ------
    ValueError : matrix no cuadrada o method desconocido.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Matriz debe ser cuadrada, recibida shape={matrix.shape}"
        )

    if method == "eig":
        # eigvalsh garantiza eigenvalores reales para matrices simétricas
        eigenvalues = np.linalg.eigvalsh(matrix)
        # CORRECCIÓN A3: verificar TODOS los eigenvalores, no solo positivos
        lam_min = float(eigenvalues.min())
        if lam_min <= MIN_EIGVAL_TOL:
            return float("inf")
        lam_max = float(eigenvalues.max())
        # κ = λ_max / λ_min para matrices SPD
        return lam_max / lam_min

    if method == "svd":
        # SVD: σ_i = √λ_i para matrices SPD
        sv = np.linalg.svd(matrix, compute_uv=False)
        nonzero = sv[sv > MIN_EIGVAL_TOL]
        if nonzero.size == 0:
            return float("inf")
        return float(nonzero.max() / nonzero.min())

    raise ValueError(
        f"Método de condición desconocido: '{method}'. "
        f"Disponibles: 'eig', 'svd'"
    )


def _regularize_metric(
    matrix: np.ndarray,
    min_eig: float = MIN_EIGVAL_TOL,
) -> np.ndarray:
    """
    Regularización de Tikhonov: G_reg = G + δ·I, δ = max(0, min_eig − λ_min).

    Propiedades Garantizadas
    ─────────────────────────
    • Preserva autovectores: G·v = λv ⟹ (G+δI)·v = (λ+δ)v
    • Minimiza perturbación en norma de Frobenius: ‖G_reg − G‖_F = δ·√n
    • Garantiza λ_min(G_reg) ≥ min_eig

    CORRECCIÓN N3
    ─────────────
    La copia explícita con dtype=float64 garantiza que el resultado
    hereda el tipo correcto independientemente del dtype de entrada.

    Parameters
    ----------
    matrix  : array 2D simétrico (cualquier dtype numérico)
    min_eig : eigenvalor mínimo objetivo (default = MIN_EIGVAL_TOL)

    Returns
    -------
    np.ndarray, shape = matrix.shape, dtype=float64, SPD con λ_min ≥ min_eig.

    References
    ----------
    Tikhonov & Arsenin (1977), Golub & Van Loan (2013) §5.3.
    """
    mat = np.asarray(matrix, dtype=np.float64)

    # eigvalsh para matrices simétricas: eigenvalores reales garantizados
    eigenvalues = np.linalg.eigvalsh(mat)
    min_eigval = float(eigenvalues.min())

    if min_eigval >= min_eig:
        # Ya SPD: retornar copia sin modificación
        return mat.copy()

    delta = min_eig - min_eigval
    logger.debug(
        "Regularización Tikhonov: λ_min=%.4e < %.4e → δ=%.4e",
        min_eigval, min_eig, delta,
    )
    # CORRECCIÓN N3: resultado explícitamente float64
    regularized = mat + delta * np.eye(mat.shape[0], dtype=np.float64)
    return regularized


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDADORES DE RESTRICCIONES FÍSICAS
# ═══════════════════════════════════════════════════════════════════════════════

class Validator(ABC):
    """Interfaz para validadores de restricciones físicas y matemáticas."""

    @abstractmethod
    def validate(
        self, value: float, name: str,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Valida un valor y retorna (validated, was_modified, message).

        Returns
        -------
        validated_value : float, posiblemente corregido
        was_modified    : True si el valor fue ajustado
        error_msg       : descripción si hubo corrección, None si no

        Raises
        ------
        PhysicalBoundsError : valor viola límite fundamental (NaN, Inf)
        """

    @abstractmethod
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna (lower_bound, upper_bound), None = sin límite."""


class UnitIntervalValidator(Validator):
    """
    Valida v ∈ [0, 1] con clamping automático.

    Invariante: valor retornado ∈ [0.0, 1.0] siempre.
    Lanza PhysicalBoundsError si v es NaN o Inf.
    """

    def validate(
        self, value: float, name: str,
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"[{name}] Valor no finito: {value!r}"
            )
        if value < 0.0:
            return 0.0, True, f"{name}={value:.6g} → clamp a 0.0"
        if value > 1.0:
            return 1.0, True, f"{name}={value:.6g} → clamp a 1.0"
        return value, False, None

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0.0, 1.0)


class NonNegativeValidator(Validator):
    """
    Valida v ≥ 0 con clamping automático.

    Invariante: valor retornado ≥ 0.0 siempre.
    """

    def validate(
        self, value: float, name: str,
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"[{name}] Valor no finito: {value!r}"
            )
        if value < 0.0:
            return 0.0, True, f"{name}={value:.6g} → clamp a 0.0"
        return value, False, None

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0.0, None)


class PositiveIntValidator(Validator):
    """
    Valida entero positivo ≥ 1 con redondeo automático.

    Invariante: valor retornado es un float con valor entero ≥ 1.0.
    """

    def validate(
        self, value: float, name: str,
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"[{name}] Valor no finito: {value!r}"
            )
        rounded = max(1, int(round(value)))
        # CORRECCIÓN P1: usar threshold relativo para detectar modificación
        was_modified = (rounded != value) or (abs(rounded - value) > 0.5 * EPS)
        if was_modified:
            return float(rounded), True, f"{name}={value:.6g} → redondeo a {rounded}"
        return float(rounded), False, None

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (1, None)


class NonNegativeIntValidator(Validator):
    """
    Valida entero no-negativo ≥ 0 con redondeo automático.

    Invariante: valor retornado es un float con valor entero ≥ 0.0.
    """

    def validate(
        self, value: float, name: str,
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"[{name}] Valor no finito: {value!r}"
            )
        rounded = max(0, int(round(value)))
        was_modified = (rounded != value) or (abs(rounded - value) > 0.5 * EPS)
        if was_modified:
            return float(rounded), True, f"{name}={value:.6g} → redondeo a {rounded}"
        return float(rounded), False, None

    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0, None)


VALIDATOR_REGISTRY: Dict[str, Validator] = {
    "unit_interval":   UnitIntervalValidator(),
    "non_negative":    NonNegativeValidator(),
    "positive_int":    PositiveIntValidator(),
    "non_negative_int": NonNegativeIntValidator(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERACIÓN DE ESTADO DE SALUD
# ═══════════════════════════════════════════════════════════════════════════════

class HealthStatus(Enum):
    """
    Estados de salud del sistema inmunológico.

    Orden total: HEALTHY < WARNING < CRITICAL (por severidad numérica).

    CORRECCIÓN A1
    ─────────────
    En versiones anteriores, HealthStatus._severity_map_ era un enum member
    envuelto automáticamente en un Value, por lo que el acceso correcto era:
        HealthStatus._severity_map_.value.get(...)

    Esta implementación reemplaza el dict-como-member por un dict de clase
    real (fuera del namespace de Enum), evitando el wrapping automático.
    Se accede como atributo de clase directamente: _SEVERITY_MAP[name].

    Representación: HEALTHY=0, WARNING=1, CRITICAL=2 (severidad ascendente).
    """
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()

    @property
    def severity(self) -> int:
        """
        Nivel de severidad numérico: 0=sano, 1=alerta, 2=crítico.

        Invariante: severity ∈ {0, 1, 2} para todos los miembros.
        """
        # CORRECCIÓN A1: acceder al dict de clase, no al member envuelto
        return _HEALTH_SEVERITY_MAP[self.name]

    def __lt__(self, other: "HealthStatus") -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity < other.severity

    def __le__(self, other: "HealthStatus") -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity <= other.severity

    def __gt__(self, other: "HealthStatus") -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity > other.severity

    def __ge__(self, other: "HealthStatus") -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity >= other.severity

    def __str__(self) -> str:
        return _HEALTH_STATUS_SYMBOLS[self.name]


# Definición fuera del enum para evitar wrapping automático de Enum
# CORRECCIÓN A1: dict de clase accesible directamente sin .value
_HEALTH_SEVERITY_MAP: Dict[str, int] = {
    "HEALTHY":  0,
    "WARNING":  1,
    "CRITICAL": 2,
}

_HEALTH_STATUS_SYMBOLS: Dict[str, str] = {
    "HEALTHY":  "✓ HEALTHY",
    "WARNING":  "⚠ WARNING",
    "CRITICAL": "✗ CRITICAL",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR MÉTRICO RIEMANNIANO
# ═══════════════════════════════════════════════════════════════════════════════

class MetricTensor:
    """
    Tensor métrico Riemanniano G para un subespacio Vₖ ⊂ ℝⁿ.

    Propiedades Garantizadas
    ─────────────────────────
    • Gᵀ = G              (simetría, impuesta por promedio (G+Gᵀ)/2)
    • λ_min(G) ≥ MIN_EIGVAL_TOL  (definida positiva)
    • κ(G) < COND_NUM_TOL        (condición controlada)
    • to_array() retorna copia mutable (no referencia al array interno)

    Representaciones
    ─────────────────
    Diagonal (1D): almacena vector de diagonales. O(n) memoria, O(n) operaciones.
    Densa (2D)   : almacena matriz completa. O(n²) memoria, O(n²) operaciones.

    CORRECCIÓN N4
    ─────────────
    Para representación densa: regularización se aplica ANTES de verificar
    el número de condición, garantizando que κ se mida sobre G ya SPD.
    """

    __slots__ = (
        "_matrix",
        "_is_diagonal",
        "_dim",
        "_condition_number",
        "_eigen_cache",
    )

    def __init__(
        self,
        matrix: Union[np.ndarray, list],
        validate: bool = True,
        regularization_threshold: float = COND_NUM_TOL,
    ) -> None:
        arr = np.asarray(matrix, dtype=np.float64)

        if arr.ndim == 1:
            self._init_diagonal(arr, validate, regularization_threshold)
        elif arr.ndim == 2:
            self._init_dense(arr, validate, regularization_threshold)
        else:
            raise MetricTensorError(
                f"MetricTensor acepta 1D (diagonal) o 2D (densa): ndim={arr.ndim}"
            )

        # Cache de descomposición espectral (inicialmente vacío)
        self._eigen_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _init_diagonal(
        self,
        arr: np.ndarray,
        validate: bool,
        regularization_threshold: float,
    ) -> None:
        """
        Inicialización para representación diagonal.

        Invariante: todos los elementos dᵢ > MIN_EIGVAL_TOL.
        κ(diag(d)) = max(d) / min(d) exactamente.
        """
        self._dim = arr.shape[0]
        self._is_diagonal = True

        if validate:
            # CORRECCIÓN: verificar contra MIN_EIGVAL_TOL, no contra 0
            bad_mask = arr <= MIN_EIGVAL_TOL
            if bad_mask.any():
                bad_vals = arr[bad_mask]
                raise MetricTensorError(
                    f"Elementos diagonales deben ser > {MIN_EIGVAL_TOL:.2e}. "
                    f"Valores inválidos: {bad_vals}"
                )

        mat = arr.copy()
        mat.flags.writeable = False
        self._matrix = mat

        # κ(diag(d)) = max(d)/min(d): exactamente calculable
        d_min = float(arr.min())
        d_max = float(arr.max())
        self._condition_number = d_max / max(d_min, MIN_EIGVAL_TOL)

        if validate and self._condition_number > regularization_threshold:
            logger.warning(
                "Métrica diagonal mal condicionada: κ=%.2e > %.2e",
                self._condition_number, regularization_threshold,
            )

    def _init_dense(
        self,
        arr: np.ndarray,
        validate: bool,
        regularization_threshold: float,
    ) -> None:
        """
        Inicialización para representación densa.

        Orden de Operaciones (CORRECCIÓN N4)
        ──────────────────────────────────────
        1. Verificar forma cuadrada
        2. Verificar simetría (si validate=True)
        3. Imponer simetría exacta: sym = (arr + arrᵀ)/2
        4. Regularizar (si validate=True): sym_reg = sym + δ·I
        5. Calcular κ sobre la matriz REGULARIZADA
        6. Advertir si κ es alto
        """
        if arr.shape[0] != arr.shape[1]:
            raise MetricTensorError(
                f"MetricTensor densa debe ser cuadrada: shape={arr.shape}"
            )

        if validate:
            asym = float(np.linalg.norm(arr - arr.T, "fro"))
            if asym > ALGEBRAIC_TOL:
                raise MetricTensorError(
                    f"Matriz no simétrica: ‖G − Gᵀ‖_F = {asym:.4e} > {ALGEBRAIC_TOL}"
                )

        # Paso 3: Imponer simetría exacta (elimina errores de redondeo)
        sym = (arr + arr.T) * 0.5

        # Paso 4: Regularizar para garantizar SPD
        if validate:
            sym = _regularize_metric(sym, min_eig=MIN_EIGVAL_TOL)

        self._dim = sym.shape[0]
        self._is_diagonal = False

        mat = sym.copy()
        mat.flags.writeable = False
        self._matrix = mat

        # Paso 5: κ sobre la matriz regularizada (CORRECCIÓN N4)
        self._condition_number = _compute_condition_number(sym, method="eig")

        if self._condition_number > regularization_threshold:
            warnings.warn(
                f"Métrica densa mal condicionada: κ={self._condition_number:.2e} "
                f"> {regularization_threshold:.2e}",
                UserWarning,
                stacklevel=3,
            )

    # ── Propiedades de solo lectura ─────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Dimensión n del espacio (G es n×n)."""
        return self._dim

    @property
    def is_diagonal(self) -> bool:
        """True si la representación interna es diagonal (1D)."""
        return self._is_diagonal

    @property
    def condition_number(self) -> float:
        """κ(G) = λ_max/λ_min ≥ 1 para G invertible."""
        return self._condition_number

    @property
    def eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Descomposición espectral G = V·diag(λ)·Vᵀ (perezosa, cacheada).

        Para diagonal: λ = diagonal, V = I_n.
        Para densa   : eigvalsh garantiza λ reales y V ortogonal.

        Returns
        -------
        (eigenvalues, eigenvectors) : arrays de dtype float64, immutables.
        """
        if self._eigen_cache is None:
            if self._is_diagonal:
                # Para diagonal: λᵢ = dᵢ, vᵢ = eᵢ (base canónica)
                eigvals = self._matrix.copy()
                eigvecs = np.eye(self._dim, dtype=np.float64)
                self._eigen_cache = (eigvals, eigvecs)
            else:
                # eigvalsh para matrices simétricas: eigenvalores reales
                eigvals, eigvecs = np.linalg.eigh(self._matrix)
                self._eigen_cache = (
                    eigvals.astype(np.float64),
                    eigvecs.astype(np.float64),
                )
        return self._eigen_cache

    # ── Operaciones Métricas ────────────────────────────────────────────────

    def quadratic_form(self, v: np.ndarray) -> float:
        """
        Calcula vᵀ G v.

        CORRECCIÓN N5
        ─────────────
        El resultado se clampea a max(result, 0.0) para absorber errores
        numéricos sub-epsilon que producen vᵀGv = -ε ≈ 0 cuando G es SPD.
        Sin este clamping, sqrt(vᵀGv) en compute_threat lanzaría RuntimeWarning.

        Complejidad
        ───────────
        Diagonal: O(n) — producto elemento a elemento
        Densa   : O(n²) — multiplicación matriz-vector + producto punto

        Parameters
        ----------
        v : array 1D de longitud n

        Returns
        -------
        float ≥ 0 (clampeado para absorber error numérico).
        """
        if self._is_diagonal:
            # Σᵢ dᵢ vᵢ²: más eficiente que v @ diag(d) @ v
            result = float(np.dot(self._matrix * v, v))
        else:
            result = float(v @ self._matrix @ v)

        # CORRECCIÓN N5: clamping para absorber errores numéricos sub-epsilon
        return max(result, 0.0)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """
        Calcula G·v.

        Diagonal: dᵢ·vᵢ (O(n))
        Densa   : G @ v  (O(n²))
        """
        if self._is_diagonal:
            return self._matrix * v
        return self._matrix @ v

    def inverse_sqrt_apply(self, v: np.ndarray) -> np.ndarray:
        """
        Calcula G^{-1/2}·v vía descomposición espectral.

        Algoritmo
        ─────────
        G = V·Λ·Vᵀ  ⟹  G^{-1/2} = V·Λ^{-1/2}·Vᵀ

        Regularización: λᵢ ← max(λᵢ, MIN_EIGVAL_TOL) antes de √,
        evitando división por eigenvalores degenerados.

        Parameters
        ----------
        v : array 1D, debe tener longitud = self.dimension

        Returns
        -------
        np.ndarray de misma shape.
        """
        eigvals, eigvecs = self.eigen_decomposition
        # Regularización espectral: evita división por eigenvalores cercanos a 0
        safe_eigvals = np.maximum(eigvals, MIN_EIGVAL_TOL)
        inv_sqrt_diag = 1.0 / np.sqrt(safe_eigvals)

        if self._is_diagonal:
            # Para diagonal: G^{-1/2} = diag(1/√dᵢ)
            return inv_sqrt_diag * v
        # Para densa: G^{-1/2}v = V · diag(1/√λᵢ) · Vᵀv
        return eigvecs @ (inv_sqrt_diag * (eigvecs.T @ v))

    def to_array(self) -> np.ndarray:
        """
        Retorna copia mutable del array interno.

        Garantía: modificar el array retornado NO afecta el estado interno
        del tensor (la copia tiene flags.writeable = True).
        """
        result = self._matrix.copy()
        result.flags.writeable = True
        return result

    def __repr__(self) -> str:
        kind = "diagonal" if self._is_diagonal else "densa"
        return (
            f"MetricTensor({kind}, n={self._dim}, κ={self._condition_number:.2e})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ESPECIFICACIÓN DE SUBESPACIO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubspaceSpec:
    """
    Especificación de un subespacio de riesgo Vₖ ⊂ ℝⁿ.

    Invarianzas Dimensionales (verificadas en __post_init__)
    ─────────────────────────────────────────────────────────
    dim(reference) = dim(metric) = len(range(indices))

    Distancia de Mahalanobis
    ─────────────────────────
    d_G(x, ref) = √[(x−ref)ᵀ G (x−ref)]

    Con scale=[s₀,...,sₙ₋₁]:
        G = diag(1/s₀², ..., 1/sₙ₋₁²)
        d_G(δ) = ‖D⁻¹δ‖₂  donde D = diag(s)
    """
    name: str
    indices: slice
    weight: float
    reference: np.ndarray
    metric: Optional[MetricTensor] = field(default=None)
    scale: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Normalizar referencia a array 1D float64 inmutable
        self.reference = np.asarray(
            self.reference, dtype=np.float64,
        ).ravel().copy()
        self.reference.flags.writeable = False
        dim = self.reference.shape[0]

        # Validar peso: debe ser estrictamente positivo
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(
                f"[{self.name}] weight debe ser finito y > 0: recibido {self.weight}"
            )

        # Construir métrica desde scale si metric no se proporciona explícitamente
        if self.metric is None:
            if self.scale is not None:
                scale_arr = np.asarray(self.scale, dtype=np.float64).ravel()
                if scale_arr.shape[0] != dim:
                    raise DimensionalMismatchError(
                        f"[{self.name}] dim(scale)={scale_arr.shape[0]} "
                        f"≠ dim(reference)={dim}"
                    )
                # G = diag(1/s²): distancia Mahalanobis = ‖D⁻¹δ‖₂
                # Protección: si sᵢ ≈ 0 → gᵢ → ∞ (insensible en esa dirección)
                metric_diag = _stable_reciprocal(scale_arr ** 2, eps=EPS)
                # Garantizar definición positiva estricta
                metric_diag = np.maximum(metric_diag, MIN_EIGVAL_TOL)
                self.metric = MetricTensor(metric_diag, validate=True)
            else:
                # Sin scale ni metric: métrica euclidiana estándar G = I_n
                self.metric = MetricTensor(np.ones(dim, dtype=np.float64), validate=True)

        # Verificación cruzada de dimensión (aún si metric se pasó explícitamente)
        if self.metric.dimension != dim:
            raise DimensionalMismatchError(
                f"[{self.name}] dim(metric)={self.metric.dimension} "
                f"≠ dim(reference)={dim}"
            )

    def compute_threat(self, subvector: np.ndarray) -> float:
        """
        Calcula la amenaza del subespacio Vₖ.

        Fórmula
        ───────
        threatₖ = wₖ · d_{Gₖ}(v, ref) = wₖ · √[(v−ref)ᵀ Gₖ (v−ref)]

        Propiedades
        ───────────
        • threatₖ ≥ 0  (distancia Mahalanobis no-negativa)
        • threatₖ = 0  ⟺ v = ref  (solo en el punto de referencia)
        • Escala linealmente con weight: threat(wv) = w · threat(v)

        Parameters
        ----------
        subvector : array 1D de longitud = dim(reference)

        Raises
        ------
        DimensionalMismatchError : si shape(subvector) ≠ shape(reference)

        Returns
        -------
        float ≥ 0.
        """
        sv = np.asarray(subvector, dtype=np.float64).ravel()
        if sv.shape != self.reference.shape:
            raise DimensionalMismatchError(
                f"[{self.name}] shape incompatible: "
                f"{sv.shape} ≠ {self.reference.shape}"
            )

        delta = sv - self.reference
        # quadratic_form ya clampea a 0 (CORRECCIÓN N5 en MetricTensor)
        maha_sq = self.metric.quadratic_form(delta)
        # Protección adicional contra negativos residuales
        return self.weight * float(np.sqrt(max(maha_sq, 0.0)))

    def normalize_to_reference(self, subvector: np.ndarray) -> np.ndarray:
        """
        Proyecta subvector al espacio normalizado G^{-1/2}(v − ref).

        Útil para visualización y análisis de sensibilidad.
        """
        sv = np.asarray(subvector, dtype=np.float64).ravel()
        return self.metric.inverse_sqrt_apply(sv - self.reference)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTADO INMUTABLE DE EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThreatAssessment:
    """
    Resultado inmutable de la evaluación de amenazas topológicas.

    Invarianzas
    ───────────
    • total_threat = ‖(threat₁,...,threatₙ)‖₂ ≥ 0
    • max_source ∈ keys(levels), levels[max_source] = max_value
    • status ∈ {HEALTHY, WARNING, CRITICAL}
    • Todos los valores en levels son ≥ 0
    """
    levels: Dict[str, float]
    max_source: str
    max_value: float
    total_threat: float
    euler_char: Optional[int] = None
    status: HealthStatus = HealthStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.4f}" for k, v in self.levels.items())
        euler_str = f", χ={self.euler_char}" if self.euler_char is not None else ""
        return (
            f"ThreatAssessment({items}{euler_str}) → {self.status} "
            f"(max: {self.max_source}={self.max_value:.4f}, "
            f"total={self.total_threat:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización JSON-compatible.

        Todos los valores son tipos Python nativos: float, int, str, dict, list.
        Ningún valor es numpy.float64, numpy.int64, o similar.
        """
        return {
            "threat_levels": {k: float(v) for k, v in self.levels.items()},
            "max_threat_source": str(self.max_source),
            "max_threat_value": float(self.max_value),
            "total_threat": float(self.total_threat),
            "euler_characteristic": (
                int(self.euler_char) if self.euler_char is not None else None
            ),
            "health_status": self.status.name,
            "metadata": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in self.metadata.items()
            },
        }

    @staticmethod
    def from_components(
        levels: Dict[str, float],
        euler_char: Optional[int] = None,
        warning_threshold: float = 0.8,
        critical_threshold: float = 1.5,
    ) -> "ThreatAssessment":
        """
        Constructor que infiere atributos derivados automáticamente.

        Parameters
        ----------
        levels             : dict de amenazas por subespacio (valores ≥ 0)
        euler_char         : característica de Euler χ = β₀ - β₁ (o None)
        warning_threshold  : umbral de alerta
        critical_threshold : umbral crítico

        Raises
        ------
        ValueError : si levels está vacío o tiene valores negativos.
        """
        if not levels:
            raise ValueError(
                "levels no puede estar vacío: se requiere ≥ 1 subespacio"
            )

        # Verificar no-negatividad de todos los niveles
        negative = {k: v for k, v in levels.items() if v < 0.0}
        if negative:
            raise ValueError(
                f"Niveles de amenaza negativos detectados: {negative}. "
                "La distancia de Mahalanobis es siempre ≥ 0."
            )

        max_source = max(levels, key=lambda k: levels[k])
        max_value = float(levels[max_source])
        total_threat = float(np.linalg.norm(list(levels.values())))

        if max_value > critical_threshold:
            status = HealthStatus.CRITICAL
        elif max_value > warning_threshold:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return ThreatAssessment(
            levels=dict(levels),  # copia defensiva
            max_source=max_source,
            max_value=max_value,
            total_threat=total_threat,
            euler_char=euler_char,
            status=status,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PROYECTOR ORTOGONAL
# ═══════════════════════════════════════════════════════════════════════════════

class OrthogonalProjector:
    """
    Descompone ψ ∈ ℝⁿ en subespacios ortogonales y cuantifica amenazas.

    Propiedades Algebraicas Garantizadas
    ────────────────────────────────────
    πₖ² = πₖ          : idempotencia (verificada en _validate_and_build)
    πₖᵀ = πₖ          : autoadjunta (verificada en _validate_and_build)
    πᵢ·πⱼ = 0 (i≠j)  : ortogonalidad (garantizada por disjuntividad de índices)
    Σₖ πₖ = I_n       : resolución de identidad (verificada globalmente)

    Invariantes Topológicos
    ───────────────────────
    β₀ ≥ 1  : componentes conexas (int Python nativo)
    β₁ ≥ 0  : ciclos independientes (int Python nativo)
    χ ∈ ℤ   : característica de Euler (int Python nativo)
    """

    __slots__ = (
        "_dim",
        "_subspaces",
        "_topo_indices",
        "_projection_matrices",
        "_validation_report",
    )

    def __init__(
        self,
        dimensions: int,
        subspaces: Dict[str, SubspaceSpec],
        topo_indices: Optional[Tuple[int, int]] = None,
        cache_projections: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dimensions      : dimensión total del espacio ℝⁿ
        subspaces       : dict de SubspaceSpec, con índices disjuntos
        topo_indices    : (idx_β₀, idx_β₁) en el vector ψ, o None
        cache_projections : precalcular matrices de proyección

        Raises
        ------
        DimensionalMismatchError : índices solapados o dim(ref) ≠ dim(Vₖ)
        NumericalStabilityError  : propiedades algebraicas violadas
        ValueError               : topo_indices fuera de rango
        """
        self._dim = dimensions
        self._subspaces: Dict[str, SubspaceSpec] = dict(subspaces)
        self._projection_matrices: Dict[str, np.ndarray] = {}
        self._validation_report: Dict[str, float] = {}

        # CORRECCIÓN T4: validar topo_indices contra dimensiones
        if topo_indices is not None:
            idx_b0, idx_b1 = topo_indices
            for idx, name in [(idx_b0, "β₀"), (idx_b1, "β₁")]:
                if not (0 <= idx < dimensions):
                    raise ValueError(
                        f"topo_indices[{name}]={idx} fuera de rango [0, {dimensions})"
                    )
        self._topo_indices = topo_indices

        self._validate_and_build(cache_projections)

    def _validate_and_build(self, cache_projections: bool) -> None:
        """
        Verifica propiedades algebraicas y construye matrices de proyección.

        Orden de Verificación
        ──────────────────────
        1. Disjuntividad de índices  → garantiza πᵢπⱼ = 0
        2. Consistencia dimensional  → dim(ref) = |Vₖ|
        3. Idempotencia πₖ           → πₖ² = πₖ
        4. Autoadjunta πₖ            → πₖᵀ = πₖ
        5. Resolución de identidad   → Σₖ πₖ = I_n (si cobertura completa)
        """
        covered: set = set()

        for name, spec in self._subspaces.items():
            idx_set = set(range(*spec.indices.indices(self._dim)))

            # 1. Disjuntividad
            overlap = covered & idx_set
            if overlap:
                raise DimensionalMismatchError(
                    f"Subespacio '{name}' se solapa en índices {sorted(overlap)}. "
                    "Violación de ortogonalidad: πᵢπⱼ ≠ 0"
                )
            covered |= idx_set

            # 2. Consistencia dimensional
            expected_dim = len(idx_set)
            if spec.reference.shape[0] != expected_dim:
                raise DimensionalMismatchError(
                    f"[{name}] dim(reference)={spec.reference.shape[0]} "
                    f"≠ |Vₖ|={expected_dim}"
                )

            if cache_projections:
                # Construir proyector diagonal: P[i,i]=1 ∀i ∈ idx_set
                P = np.zeros((self._dim, self._dim), dtype=np.float64)
                for i in idx_set:
                    P[i, i] = 1.0

                # 3. Idempotencia: πₖ² = πₖ
                # Para proyectores diagonales con {0,1}: P² = P exactamente.
                idem_err = float(np.linalg.norm(P @ P - P, "fro"))
                self._validation_report[f"{name}_idempotence"] = idem_err
                if idem_err > ALGEBRAIC_TOL:
                    raise NumericalStabilityError(
                        f"[{name}] No idempotente: ‖π²−π‖_F={idem_err:.2e}"
                    )

                # 4. Autoadjunta: πₖᵀ = πₖ
                self_adj_err = float(np.linalg.norm(P - P.T, "fro"))
                self._validation_report[f"{name}_self_adjoint"] = self_adj_err
                if self_adj_err > ALGEBRAIC_TOL:
                    raise NumericalStabilityError(
                        f"[{name}] No autoadjunta: ‖P−Pᵀ‖_F={self_adj_err:.2e}"
                    )

                # CORRECCIÓN A5: verificar rank = |idx_set| (tr(P) para proyector)
                rank_via_trace = int(round(float(np.trace(P))))
                if rank_via_trace != expected_dim:
                    raise NumericalStabilityError(
                        f"[{name}] rank(π)={rank_via_trace} ≠ |Vₖ|={expected_dim}"
                    )

                self._projection_matrices[name] = P

        # 5. Resolución de identidad (solo si cobertura completa)
        uncovered = set(range(self._dim)) - covered
        if uncovered:
            logger.warning(
                "Índices sin subespacio: %s — Σₖ πₖ ≠ I (cobertura parcial)",
                sorted(uncovered),
            )
            self._validation_report["uncovered_index_count"] = float(len(uncovered))

        elif self._projection_matrices:
            # CORRECCIÓN A4: suma con acumulador np.zeros tipado (no sum() con int 0)
            total_P = np.zeros((self._dim, self._dim), dtype=np.float64)
            for P in self._projection_matrices.values():
                total_P = total_P + P

            coverage_err = float(np.linalg.norm(total_P - np.eye(self._dim), "fro"))
            self._validation_report["coverage_identity_error"] = coverage_err

            if coverage_err > ALGEBRAIC_TOL:
                raise NumericalStabilityError(
                    f"Σₖ πₖ ≠ I: ‖Σπₖ − I‖_F = {coverage_err:.2e}"
                )

        logger.info(
            "OrthogonalProjector validado: %s",
            ", ".join(f"{k}={v:.2e}" for k, v in self._validation_report.items()),
        )

    @property
    def validation_report(self) -> Dict[str, float]:
        """Copia del informe de validación algebraica (inmutable para el caller)."""
        return dict(self._validation_report)

    def _compute_euler_characteristic(
        self, psi: np.ndarray,
    ) -> Optional[int]:
        """
        Calcula χ = β₀ − β₁ desde el vector de estado ψ.

        Invariantes Topológicos
        ───────────────────────
        β₀ ≥ 1  : al menos una componente conexa (espacio no vacío)
        β₁ ≥ 0  : ciclos no-negativos (definición de número de Betti)
        χ ∈ ℤ   : entero, invariante bajo homotopía (Hatcher 2002, Cap. 2)

        CORRECCIÓN T1
        ─────────────
        Se usa int() (Python nativo) y no np.int64 para garantizar que
        isinstance(chi, int) retorne True en tests.

        CORRECCIÓN T2/T3
        ─────────────────
        Los mensajes de error incluyen el valor detectado para diagnóstico.

        Returns
        -------
        int (Python nativo) o None si topo_indices es None.

        Raises
        ------
        TopologicalInvariantError : β₀ < 1 o β₁ < 0.
        """
        if self._topo_indices is None:
            return None

        idx_b0, idx_b1 = self._topo_indices

        try:
            # CORRECCIÓN T1: int() para obtener tipo Python nativo
            beta_0 = int(round(float(psi[idx_b0])))
            beta_1 = int(round(float(psi[idx_b1])))
        except (IndexError, ValueError) as exc:
            logger.warning(
                "Error extrayendo números de Betti de ψ: %s", exc,
            )
            return None

        # CORRECCIÓN T2: mensaje incluye valor detectado
        if beta_0 < 1:
            raise TopologicalInvariantError(
                f"β₀={beta_0} inválido: debe ser ≥ 1 (componentes conexas). "
                f"Índice ψ[{idx_b0}]={float(psi[idx_b0]):.6g}"
            )

        # CORRECCIÓN T3: mensaje incluye valor detectado
        if beta_1 < 0:
            raise TopologicalInvariantError(
                f"β₁={beta_1} inválido: debe ser ≥ 0 (ciclos independientes). "
                f"Índice ψ[{idx_b1}]={float(psi[idx_b1]):.6g}"
            )

        # Resultado es int nativo de Python (garantía de tipo)
        chi: int = beta_0 - beta_1
        return chi

    @contextmanager
    def temporary_algebraic_tolerance(
        self,
        relaxed_tol: float = 1e-6,
    ) -> Generator[None, None, None]:
        """
        Context manager que modifica temporalmente ALGEBRAIC_TOL del módulo.

        CORRECCIÓN I1
        ─────────────
        Se valida que relaxed_tol > 0 antes de modificar la variable global.
        Una tolerancia ≤ 0 no tiene sentido matemático (cota superior de error).

        Garantía de Restauración
        ─────────────────────────
        La cláusula finally garantiza restauración incluso ante excepciones,
        manteniendo la idempotencia del módulo.

        Usage
        -----
        with projector.temporary_algebraic_tolerance(1e-6):
            ... # ALGEBRAIC_TOL = 1e-6 aquí
        # ALGEBRAIC_TOL restaurado aquí

        Raises
        ------
        ValueError : si relaxed_tol ≤ 0 (tolerancia inválida)
        """
        if relaxed_tol <= 0.0:
            raise ValueError(
                f"relaxed_tol debe ser > 0: recibido {relaxed_tol}. "
                "Una tolerancia de cero o negativa no tiene sentido matemático."
            )

        original = getattr(_THIS_MODULE, "ALGEBRAIC_TOL")
        setattr(_THIS_MODULE, "ALGEBRAIC_TOL", float(relaxed_tol))
        logger.debug(
            "ALGEBRAIC_TOL: %.2e → %.2e (context manager)",
            original, relaxed_tol,
        )
        try:
            yield
        finally:
            setattr(_THIS_MODULE, "ALGEBRAIC_TOL", original)
            logger.debug(
                "ALGEBRAIC_TOL restaurado: %.2e", original,
            )

    def project(
        self,
        psi: np.ndarray,
        warning_threshold: float = 0.8,
        critical_threshold: float = 1.5,
        hysteresis: float = 0.05,
        previous_status: Optional[HealthStatus] = None,
        verbose: bool = False,
    ) -> ThreatAssessment:
        """
        Proyecta ψ ∈ ℝⁿ en subespacios y retorna evaluación de amenazas.

        Algoritmo
        ─────────
        1. Validar ψ: shape=(n,), todos finitos
        2. Para cada Vₖ: threatₖ = wₖ · d_{Gₖ}(ψₖ, refₖ)
        3. total_threat = ‖(threat₁,...,threatₙ)‖₂
        4. χ = β₀ − β₁ (si topo_indices ≠ None)
        5. Clasificar con histéresis

        Parameters
        ----------
        psi               : vector de estado ψ ∈ ℝⁿ
        warning_threshold : umbral de alerta (default 0.8)
        critical_threshold: umbral crítico (default 1.5)
        hysteresis        : banda de histéresis δ ≥ 0 (default 0.05)
        previous_status   : estado previo para histéresis (None = sin memoria)
        verbose           : log detallado de subespacios

        Raises
        ------
        DimensionalMismatchError : shape(ψ) ≠ (n,)
        NumericalStabilityError  : ψ contiene NaN o Inf
        TopologicalInvariantError: β₀ < 1 o β₁ < 0

        Returns
        -------
        ThreatAssessment immutable con todos los atributos derivados.
        """
        psi = np.asarray(psi, dtype=np.float64).ravel()

        if psi.shape != (self._dim,):
            raise DimensionalMismatchError(
                f"Shape incorrecto: esperado ({self._dim},), recibido {psi.shape}"
            )

        if not np.all(np.isfinite(psi)):
            bad_indices = np.where(~np.isfinite(psi))[0].tolist()
            raise NumericalStabilityError(
                f"ψ contiene valores no finitos en índices {bad_indices}: "
                f"{psi[bad_indices]}"
            )

        if verbose:
            logger.debug(
                "project(ψ): ‖ψ‖₂=%.4f, n_subespacios=%d",
                float(np.linalg.norm(psi)),
                len(self._subspaces),
            )

        # ── Amenazas por subespacio (cómputo único por subespacio) ───────────
        levels: Dict[str, float] = {}
        for name, spec in self._subspaces.items():
            subvec = psi[spec.indices]
            threat = spec.compute_threat(subvec)  # único cómputo
            levels[name] = threat

            if verbose:
                logger.debug(
                    "  [%s] threat=%.4f (weight=%.2f, mahal=%.4f)",
                    name, threat, spec.weight,
                    threat / spec.weight if spec.weight > EPS else float("inf"),
                )

        # ── Total de amenaza (norma L²) ──────────────────────────────────────
        threat_values = list(levels.values())
        max_source = max(levels, key=lambda k: levels[k])
        max_value = float(levels[max_source])
        total_threat = float(np.linalg.norm(threat_values))

        # ── Invariante topológico ─────────────────────────────────────────────
        euler_char = self._compute_euler_characteristic(psi)

        # ── Clasificación con histéresis ──────────────────────────────────────
        status = self._classify_with_hysteresis(
            max_value,
            warning_threshold,
            critical_threshold,
            hysteresis,
            previous_status,
        )

        metadata: Dict[str, Any] = {
            "norm_psi": float(np.linalg.norm(psi)),
            "condition_numbers": {
                name: float(spec.metric.condition_number)
                for name, spec in self._subspaces.items()
            },
        }

        return ThreatAssessment(
            levels=levels,
            max_source=max_source,
            max_value=max_value,
            total_threat=total_threat,
            euler_char=euler_char,
            status=status,
            metadata=metadata,
        )

    @staticmethod
    def _classify_with_hysteresis(
        value: float,
        warning_th: float,
        critical_th: float,
        hysteresis: float,
        previous: Optional[HealthStatus],
    ) -> HealthStatus:
        """
        Clasificación con bandas de histéresis asimétricas.

        Motivación
        ──────────
        Sin histéresis, el cruce frecuente de un umbral produce "chattering"
        (oscilación rápida de estados). La histéresis δ crea zonas muertas
        que eliminan este fenómeno sin penalizar la sensibilidad global.

        Invariantes Algebraicos
        ───────────────────────
        Sea W = warning_th, C = critical_th, δ = hysteresis.
        Precondición: 0 ≤ δ < (C−W)/2 (garantizada por __init__).

        Bandas de Transición (asimétricas)
        ───────────────────────────────────
        Transición         Condición               Estado Destino
        ─────────────────────────────────────────────────────────
        ∅ → HEALTHY        value ≤ W               HEALTHY
        ∅ → WARNING        W < value ≤ C            WARNING
        ∅ → CRITICAL       value > C               CRITICAL
        H → WARNING        value > W + δ            WARNING
        H → CRITICAL       value > C + δ            CRITICAL
        W → CRITICAL       value > C + δ            CRITICAL
        W → HEALTHY        value < W − δ            HEALTHY
        C → WARNING        C − δ > value ≥ W − δ   WARNING
        C → HEALTHY        value < W − δ            HEALTHY (salto directo)

        CORRECCIÓN H1
        ─────────────
        Para δ = 0 (modo laboratorio): no hay división por cero porque
        δ se usa solo en comparaciones (>, <), no en divisiones.
        Las condiciones value > W + 0 y value < W - 0 son correctas.

        FIX-08 (desde CRITICAL)
        ───────────────────────
        El descenso desde CRITICAL es gradual:
        - Si value < W − δ: salto directo a HEALTHY (pasó por debajo de TODO)
        - Si C − δ > value ≥ W − δ: descenso a WARNING (zona intermedia)
        - Si value ≥ C − δ: permanece CRITICAL

        El orden de verificación es: HEALTHY primero (condición más restrictiva),
        luego WARNING, luego CRITICAL por defecto.
        """
        if previous is None:
            # Sin estado previo: clasificación por umbrales sin banda
            if value > critical_th:
                return HealthStatus.CRITICAL
            if value > warning_th:
                return HealthStatus.WARNING
            return HealthStatus.HEALTHY

        if previous == HealthStatus.HEALTHY:
            # Ascenso: bandas de entrada más altas (amortiguación)
            if value > critical_th + hysteresis:
                return HealthStatus.CRITICAL
            if value > warning_th + hysteresis:
                return HealthStatus.WARNING
            return HealthStatus.HEALTHY

        if previous == HealthStatus.WARNING:
            # Desde WARNING: ascenso o descenso amortiguados
            if value > critical_th + hysteresis:
                return HealthStatus.CRITICAL
            if value < warning_th - hysteresis:
                return HealthStatus.HEALTHY
            return HealthStatus.WARNING

        # previous == HealthStatus.CRITICAL
        # FIX-08: descenso gradual con verificación en orden correcto
        # Primero verificar la condición más restrictiva (HEALTHY directo)
        if value < warning_th - hysteresis:
            # Cayó por debajo de la banda inferior completa → HEALTHY
            return HealthStatus.HEALTHY
        if value < critical_th - hysteresis:
            # Zona intermedia [W−δ, C−δ): descenso a WARNING
            return HealthStatus.WARNING
        # Aún en zona CRITICAL [C−δ, ∞)
        return HealthStatus.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# ESQUEMA DE SEÑAL ψ ∈ ℝ⁷
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SignalComponent:
    """
    Especificación de una componente del vector de señal ψ ∈ ℝ⁷.

    Define el isomorfismo: dict de telemetría ↔ ℝ⁷.
    """
    key: str
    default: float
    unit: str
    description: str
    validator: Optional[str] = None
    physical_bounds: Optional[Tuple[float, float]] = None


# Potencia nominal precalculada (evita llamada en cada evaluación)
_P_NOMINAL: float = PhysicalConstants.P_NOMINAL()

# Esquema canónico del vector de señal ψ ∈ ℝ⁷
# CORRECCIÓN P2: flyback_voltage usa FLYBACK_MAX_SAFE como referencia consistente
SIGNAL_SCHEMA: Tuple[SignalComponent, ...] = (
    SignalComponent(
        key="saturation",
        default=0.0,
        unit="1",
        description="Saturación magnética [0, 1]",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0),
    ),
    SignalComponent(
        key="flyback_voltage",
        default=0.0,
        unit="V",
        description="Voltaje de flyback [V]",
        validator="non_negative",
        # CORRECCIÓN P2: usar FLYBACK_MAX_SAFE como cota superior física
        physical_bounds=(0.0, PhysicalConstants.FLYBACK_MAX_SAFE),
    ),
    SignalComponent(
        key="dissipated_power",
        default=0.0,
        unit="W",
        description="Potencia disipada [W]",
        validator="non_negative",
        physical_bounds=(0.0, _P_NOMINAL * 2.0),
    ),
    SignalComponent(
        key="beta_0",
        default=1.0,
        unit="1",
        description="β₀: número de componentes conexas (entero ≥ 1)",
        validator="positive_int",
    ),
    SignalComponent(
        key="beta_1",
        default=0.0,
        unit="1",
        description="β₁: número de ciclos independientes (entero ≥ 0)",
        validator="non_negative_int",
    ),
    SignalComponent(
        key="entropy",
        default=0.0,
        unit="1",
        description="Entropía normalizada [0, 1]",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0),
    ),
    SignalComponent(
        key="exergy_loss",
        default=0.0,
        unit="1",
        description="Pérdida exergética normalizada [0, 1]",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0),
    ),
)

# Índices de β₀ y β₁ en SIGNAL_SCHEMA (verificar consistencia con el schema)
BETTI_INDICES: Tuple[int, int] = (3, 4)

# Verificación estática de consistencia de BETTI_INDICES
assert SIGNAL_SCHEMA[BETTI_INDICES[0]].key == "beta_0", (
    f"BETTI_INDICES[0]={BETTI_INDICES[0]} no apunta a 'beta_0' en SIGNAL_SCHEMA"
)
assert SIGNAL_SCHEMA[BETTI_INDICES[1]].key == "beta_1", (
    f"BETTI_INDICES[1]={BETTI_INDICES[1]} no apunta a 'beta_1' en SIGNAL_SCHEMA"
)


def build_signal(
    telemetry: Dict[str, Any],
    strict: bool = False,
) -> np.ndarray:
    """
    Construye ψ ∈ ℝ⁷ desde el diccionario de telemetría.

    Pipeline de Procesamiento
    ─────────────────────────
    1. Extracción del valor bruto (con default si falta)
    2. Conversión a float (con manejo de tipos arbitrarios)
    3. Verificación de finitud (NaN, Inf → default o ValueError)
    4. Validación de rango físico con corrección automática
    5. Asignación al vector ψ

    CORRECCIÓN P1
    ─────────────
    La variable error_msg se inicializa explícitamente a None ANTES del
    bloque try/except del validador. Esto garantiza que siempre esté
    definida en el bloque else (sin excepción), evitando NameError.

    Parameters
    ----------
    telemetry : diccionario de métricas {str: Any}
    strict    : si True, lanza excepción en lugar de usar defaults/clamping

    Returns
    -------
    np.ndarray, shape=(7,), dtype=float64, todos los valores finitos.

    Raises
    ------
    ValueError          : si strict=True y valor no convertible o no finito
    PhysicalBoundsError : si strict=True y validador lanza PhysicalBoundsError
    """
    psi = np.empty(len(SIGNAL_SCHEMA), dtype=np.float64)

    for i, spec in enumerate(SIGNAL_SCHEMA):
        raw = telemetry.get(spec.key, spec.default)

        # ── 1. Conversión de tipo ───────────────────────────────────────────
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            msg = (
                f"Telemetría '{spec.key}' no convertible a float: "
                f"{raw!r} (tipo: {type(raw).__name__})"
            )
            if strict:
                raise ValueError(msg) from exc
            logger.warning("%s → usando default=%.6g", msg, spec.default)
            psi[i] = spec.default
            continue  # Saltar pasos siguientes para este componente

        # ── 2. Verificación de finitud ──────────────────────────────────────
        if not np.isfinite(val):
            msg = f"Telemetría '{spec.key}' no finita: {val!r}"
            if strict:
                raise ValueError(msg)
            logger.warning("%s → usando default=%.6g", msg, spec.default)
            psi[i] = spec.default
            continue

        # ── 3. Validación de rango físico ───────────────────────────────────
        if spec.validator:
            validator = VALIDATOR_REGISTRY.get(spec.validator)
            if validator is None:
                raise ValueError(
                    f"Validador desconocido '{spec.validator}' en SIGNAL_SCHEMA[{i}]"
                )

            # CORRECCIÓN P1: inicializar error_msg ANTES del try/except
            error_msg: Optional[str] = None

            try:
                val, was_modified, error_msg = validator.validate(val, spec.key)
            except PhysicalBoundsError as exc:
                if strict:
                    raise
                logger.warning(
                    "PhysicalBoundsError en '%s': %s → usando default=%.6g",
                    spec.key, exc, spec.default,
                )
                psi[i] = spec.default
                continue

            # Sin excepción: registrar modificación si ocurrió
            if was_modified and error_msg is not None:
                logger.debug(
                    "Telemetría '%s' corregida: %s",
                    spec.key, error_msg,
                )

        psi[i] = val

    return psi


# ═══════════════════════════════════════════════════════════════════════════════
# MORFISMO INMUNOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════

class ImmuneWatcherMorphism(Morphism):
    """
    Morfismo categórico de vigilancia inmunológica.

    Functorialidad
    ──────────────
    F: CategoricalState → CategoricalState
    • F(error) IS error  (identidad exacta del objeto — no copia)
    • F(success) = evaluate → emit_state

    Monitoreo Topológico
    ─────────────────────
    χ(t) constante ⟹ estable
    Δχ ≠ 0         ⟹ bifurcación cualitativa (Strogatz, cap. 8)
    """

    # CORRECCIÓN FIX-13: _enable_topology_monitoring incluido en __slots__
    __slots__ = (
        "_critical",
        "_warning",
        "_hysteresis",
        "_enable_topology_monitoring",
        "_projector",
        "_previous_status",
        "_euler_history",
        "_evaluation_count",
    )

    def __init__(
        self,
        name: str = "topological_immune_watcher",
        *,
        critical_threshold: float = 1.5,
        warning_threshold: float = 0.8,
        hysteresis: float = 0.05,
        enable_topology_monitoring: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        name                    : identificador del morfismo
        critical_threshold      : umbral de cuarentena
        warning_threshold       : umbral de alerta
        hysteresis              : banda de amortiguación δ ≥ 0
        enable_topology_monitoring : activar monitoreo de χ

        Raises
        ------
        ValueError : si thresholds no son válidos o hysteresis fuera de rango.
        """
        if warning_threshold >= critical_threshold:
            raise ValueError(
                f"warning_threshold ({warning_threshold:.4f}) debe ser "
                f"< critical_threshold ({critical_threshold:.4f})"
            )

        max_hysteresis = (critical_threshold - warning_threshold) / 2.0

        # CORRECCIÓN H2: condición corregida
        # hysteresis ∈ [0, max_hysteresis) es el rango válido.
        # hysteresis = 0 es válido (modo laboratorio, sin banda).
        # hysteresis = max_hysteresis es inválido (banda nula).
        if hysteresis < 0.0 or hysteresis >= max_hysteresis:
            raise ValueError(
                f"hysteresis ({hysteresis:.4f}) debe estar en "
                f"[0.0, {max_hysteresis:.4f}). "
                f"Valor 0.0 = modo laboratorio (sin histéresis). "
                f"Máximo permitido: {max_hysteresis:.4f} − ε"
            )

        super().__init__(name)

        self._critical = critical_threshold
        self._warning = warning_threshold
        self._hysteresis = hysteresis
        self._enable_topology_monitoring = enable_topology_monitoring

        # Estado mutable interno
        self._previous_status: Optional[HealthStatus] = None
        self._euler_history: List[Optional[int]] = []
        self._evaluation_count: int = 0

        self._projector = self._build_projector()

    def _build_projector(self) -> OrthogonalProjector:
        """
        Construye el proyector ortogonal con subespacios físicamente motivados.

        Subespacios
        ───────────
        physics_core  [0:3] : saturación, flyback, potencia disipada
        topology_core [3:5] : β₀, β₁ (peso 1.5 — cambios cualitativos)
        thermo_core   [5:7] : entropía, pérdida exergética
        """
        subspaces: Dict[str, SubspaceSpec] = {
            "physics_core": SubspaceSpec(
                name="physics_core",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(3, dtype=np.float64),
                scale=np.array([
                    PhysicalConstants.SATURATION_CRITICAL,
                    PhysicalConstants.FLYBACK_MAX_SAFE,
                    _P_NOMINAL,
                ], dtype=np.float64),
            ),
            "topology_core": SubspaceSpec(
                name="topology_core",
                indices=slice(3, 5),
                weight=1.5,
                reference=np.array([1.0, 0.0], dtype=np.float64),
                scale=np.array([1.0, 1.0], dtype=np.float64),
            ),
            "thermo_core": SubspaceSpec(
                name="thermo_core",
                indices=slice(5, 7),
                weight=1.2,
                reference=np.zeros(2, dtype=np.float64),
                scale=np.array([0.5, 0.5], dtype=np.float64),
            ),
        }

        return OrthogonalProjector(
            dimensions=7,
            subspaces=subspaces,
            topo_indices=BETTI_INDICES if self._enable_topology_monitoring else None,
            cache_projections=True,
        )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset([Stratum.PHYSICS])

    @property
    def codomain(self) -> Stratum:
        return Stratum.WISDOM

    def reset_state(self) -> None:
        """Reinicia el estado interno del morfismo."""
        self._previous_status = None
        self._euler_history.clear()
        self._evaluation_count = 0
        logger.debug("Estado inmunológico reiniciado.")

    def _check_topology_change(
        self, current_euler: Optional[int],
    ) -> None:
        """
        Detecta cambios en χ entre evaluaciones consecutivas.

        CORRECCIÓN C2
        ─────────────
        La comparación correcta es entre _euler_history[-2] y current_euler.
        En el original se usaba _euler_history[-1] antes de añadir el elemento
        actual, lo que era equivalente pero dependía del orden de llamadas.

        Esta implementación verifica la historia DESPUÉS de que current_euler
        fue añadido (llamada en __call__ DESPUÉS de .append).

        Precondición: len(_euler_history) ≥ 2 (verificada antes de acceder).
        """
        if not self._enable_topology_monitoring:
            return
        if len(self._euler_history) < 2:
            return

        # El penúltimo es el anterior (actual es el último, ya añadido)
        prev_euler = self._euler_history[-2]

        if prev_euler is not None and current_euler is not None:
            if prev_euler != current_euler:
                delta_chi = current_euler - prev_euler
                logger.warning(
                    "🔄 Bifurcación topológica detectada: χ %d → %d (Δχ=%+d). "
                    "Evaluación #%d",
                    prev_euler, current_euler, delta_chi,
                    self._evaluation_count,
                )

    def _verify_functorial_properties(self) -> Dict[str, bool]:
        """
        Verifica las propiedades categóricas del morfismo.

        Retorna dict {nombre_propiedad: bool} para auditoría.
        """
        val_report = self._projector.validation_report
        projector_ok = all(
            v < ALGEBRAIC_TOL
            for k, v in val_report.items()
            if any(kw in k for kw in ("error", "idempotence", "self_adjoint"))
        )
        max_h = (self._critical - self._warning) / 2.0

        return {
            "domain_includes_physics": Stratum.PHYSICS in self.domain,
            "codomain_is_wisdom": self.codomain == Stratum.WISDOM,
            "projector_algebraically_valid": projector_ok,
            "thresholds_ordered": self._warning < self._critical,
            "hysteresis_in_valid_range": 0.0 <= self._hysteresis < max_h,
        }

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica el morfismo F al estado categórico.

        CORRECCIÓN C1 — Functorialidad
        ────────────────────────────────
        F(error) RETORNA EL MISMO OBJETO (identidad exacta, no copia).
        `return state` (no `return state.copy()` ni similar).
        Esto garantiza que `result is state` en los tests de functorialidad.

        El contador se incrementa en TODAS las llamadas (exitosas o no),
        para mantener el conteo exacto de evaluaciones.
        """
        self._evaluation_count += 1

        # CORRECCIÓN C1: identidad exacta del objeto, no copia
        if not state.is_success:
            logger.debug(
                "F(error): estado fallido propagado sin evaluación (#%d)",
                self._evaluation_count,
            )
            return state  # Identidad: mismo objeto, sin mutación

        try:
            telemetry = state.context.get("telemetry_metrics", {})
            signal = build_signal(telemetry or {}, strict=False)

            assessment = self._projector.project(
                signal,
                warning_threshold=self._warning,
                critical_threshold=self._critical,
                hysteresis=self._hysteresis,
                previous_status=self._previous_status,
                verbose=False,
            )

            # Actualizar estado interno ANTES de _check_topology_change
            self._previous_status = assessment.status
            self._euler_history.append(assessment.euler_char)

            # CORRECCIÓN C2: _check_topology_change usa el euler ya añadido
            self._check_topology_change(assessment.euler_char)

            return self._emit_state(state, assessment)

        except (ValueError, ImmuneSystemError) as exc:
            logger.error(
                "Error de validación en sistema inmunológico (#%d): %s",
                self._evaluation_count, exc,
            )
            return state.with_error(
                error_msg=f"Validación inmunológica fallida: {exc}",
                details={
                    "morphism": self.name,
                    "error_type": "validation",
                    "exception_class": type(exc).__name__,
                    "evaluation_number": self._evaluation_count,
                },
            )

        except Exception as exc:
            logger.exception(
                "Error inesperado en sistema inmunológico (#%d)",
                self._evaluation_count,
            )
            return state.with_error(
                error_msg=f"Error inesperado: {type(exc).__name__}: {exc}",
                details={
                    "morphism": self.name,
                    "error_type": "unexpected",
                    "exception_class": type(exc).__name__,
                    "evaluation_number": self._evaluation_count,
                },
            )

    def _emit_state(
        self,
        state: CategoricalState,
        assessment: ThreatAssessment,
    ) -> CategoricalState:
        """
        Emite estado transformado según el nivel de amenaza.

        CORRECCIÓN C3
        ─────────────
        with_update recibe el contexto como dict positional y new_stratum
        como keyword argument, separando claramente los datos del routing.

        Mapeo
        ─────
        HEALTHY  → with_update (contexto saludable enriquecido)
        WARNING  → with_update (marcador de alerta visible)
        CRITICAL → with_error  (cuarentena inmediata)
        """
        details = assessment.to_dict()

        if assessment.status == HealthStatus.CRITICAL:
            logger.critical(
                "🛡️ CUARENTENA ACTIVADA: fuente=%s amenaza=%.4f (umbral=%.4f) χ=%s",
                assessment.max_source,
                assessment.max_value,
                self._critical,
                assessment.euler_char,
            )
            return state.with_error(
                error_msg=(
                    f"Cuarentena inmunológica: "
                    f"{assessment.max_source}={assessment.max_value:.4f}"
                ),
                details={
                    **details,
                    "action": "QUARANTINE",
                    "threshold_exceeded": float(self._critical),
                },
            )

        if assessment.status == HealthStatus.WARNING:
            logger.warning(
                "🛡️ ESTRÉS: fuente=%s amenaza=%.4f (umbral=%.4f) χ=%s",
                assessment.max_source,
                assessment.max_value,
                self._warning,
                assessment.euler_char,
            )
            # CORRECCIÓN C3: contexto y new_stratum separados
            return state.with_update(
                {
                    "immune_status": "warning",
                    "immune_source": assessment.max_source,
                    "immune_alert_level": "stress",
                    **details,
                },
                new_stratum=self.codomain,
            )

        # HealthStatus.HEALTHY
        logger.debug(
            "🛡️ ESTABLE: max_threat=%.4f en '%s' χ=%s",
            assessment.max_value,
            assessment.max_source,
            assessment.euler_char,
        )
        # CORRECCIÓN C3: contexto y new_stratum separados
        return state.with_update(
            {"immune_status": "healthy", **details},
            new_stratum=self.codomain,
        )

    # ── Propiedades de Introspección ────────────────────────────────────────

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "warning":    self._warning,
            "critical":   self._critical,
            "hysteresis": self._hysteresis,
        }

    @property
    def topology_history(self) -> Tuple[Optional[int], ...]:
        """
        Historial de χ como tupla inmutable.

        CORRECCIÓN C4
        ─────────────
        Se retorna una copia de la lista como tupla, no una referencia.
        Esto garantiza que el caller no puede mutar el estado interno.
        """
        return tuple(self._euler_history)

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def current_status(self) -> Optional[HealthStatus]:
        return self._previous_status

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Diagnóstico completo del estado interno del morfismo.

        CORRECCIÓN I2
        ─────────────
        topology_history_last10 usa slice [-10:] sobre la tupla, que es
        seguro incluso si len(history) < 10 (retorna todos los elementos).
        """
        history = self.topology_history  # copia inmutable
        return {
            "name": self.name,
            "evaluation_count": self._evaluation_count,
            "current_status": (
                str(self._previous_status) if self._previous_status else "none"
            ),
            "thresholds": self.thresholds,
            # CORRECCIÓN I2: slice seguro (no IndexError si len < 10)
            "topology_history_last10": history[-10:],
            "projector_validation": self._projector.validation_report,
            "functorial_properties": self._verify_functorial_properties(),
        }

    def health_report(self) -> str:
        """
        Reporte legible para humanos del estado del morfismo.

        CORRECCIÓN I4
        ─────────────
        Maneja historial vacío sin IndexError usando get con default.
        """
        d = self.get_diagnostics()
        props = d["functorial_properties"]

        # CORRECCIÓN I4: historia puede estar vacía
        history_last5 = d["topology_history_last10"][-5:]
        history_str = (
            " → ".join(str(c) for c in history_last5)
            if history_last5
            else "(sin evaluaciones)"
        )

        lines = [
            "🛡️  IMMUNE WATCHER — DIAGNÓSTICO",
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Nombre            : {d['name']}",
            f"  Evaluaciones      : {d['evaluation_count']}",
            f"  Estado Actual     : {d['current_status']}",
            "",
            "  UMBRALES:",
            f"    ⚠  Warning    : {d['thresholds']['warning']:.3f}",
            f"    ✗  Critical   : {d['thresholds']['critical']:.3f}",
            f"    ↺  Hysteresis : {d['thresholds']['hysteresis']:.3f}",
            "",
            "  TOPOLOGÍA (últimos 5):",
            f"    {history_str}",
            "",
            "  PROPIEDADES CATEGÓRICAS:",
        ]

        for prop, ok in props.items():
            symbol = "✓" if ok else "✗"
            lines.append(f"    [{symbol}] {prop}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY CON PERFILES DE CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def create_immune_watcher(
    profile: str = "default",
    **overrides: Any,
) -> ImmuneWatcherMorphism:
    """
    Factory de morfismos inmunológicos con perfiles predefinidos.

    Perfiles Predefinidos
    ──────────────────────
    'default'    : producción balanceada (W=0.8, C=1.5, δ=0.05)
    'strict'     : mayor sensibilidad (W=0.5, C=1.0, δ=0.03)
    'relaxed'    : menor sensibilidad (W=1.0, C=2.0, δ=0.08)
    'laboratory' : sin histéresis para testing puro (δ=0.0)

    CORRECCIÓN I3
    ─────────────
    Los overrides se validan implícitamente al construir ImmuneWatcherMorphism.
    Si los overrides producen parámetros inválidos, el constructor lanza
    ValueError con mensaje descriptivo antes de retornar el morfismo.

    Parameters
    ----------
    profile   : nombre del perfil base
    **overrides: parámetros que sobreescriben el perfil

    Returns
    -------
    ImmuneWatcherMorphism configurado y validado.

    Raises
    ------
    ValueError : perfil desconocido o overrides producen parámetros inválidos.
    """
    _PROFILES: Dict[str, Dict[str, Any]] = {
        "default": {
            "warning_threshold": 0.8,
            "critical_threshold": 1.5,
            "hysteresis": 0.05,
        },
        "strict": {
            "warning_threshold": 0.5,
            "critical_threshold": 1.0,
            "hysteresis": 0.03,
        },
        "relaxed": {
            "warning_threshold": 1.0,
            "critical_threshold": 2.0,
            "hysteresis": 0.08,
        },
        # FIX-14: hysteresis=0.0 válido — modo laboratorio sin histéresis
        "laboratory": {
            "warning_threshold": 0.8,
            "critical_threshold": 1.5,
            "hysteresis": 0.0,
        },
    }

    if profile not in _PROFILES:
        available = ", ".join(f"'{p}'" for p in sorted(_PROFILES))
        raise ValueError(
            f"Perfil desconocido: '{profile}'. Perfiles disponibles: {available}"
        )

    # Merge: overrides sobreescriben el perfil base
    config: Dict[str, Any] = {**_PROFILES[profile], **overrides}

    # CORRECCIÓN I3: la validación ocurre en __init__ (no se duplica aquí)
    return ImmuneWatcherMorphism(
        name=f"immune_watcher_{profile}",
        **config,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA CONTROLADA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Excepciones
    "ImmuneSystemError",
    "NumericalStabilityError",
    "DimensionalMismatchError",
    "PhysicalBoundsError",
    "TopologicalInvariantError",
    "MetricTensorError",
    # Constantes y tolerancias
    "EPS",
    "ALGEBRAIC_TOL",
    "COND_NUM_TOL",
    "MIN_EIGVAL_TOL",
    "PhysicalConstants",
    # Tipos principales
    "HealthStatus",
    "MetricTensor",
    "SubspaceSpec",
    "ThreatAssessment",
    "OrthogonalProjector",
    "SignalComponent",
    "ImmuneWatcherMorphism",
    # Funciones utilitarias (expuestas para testing)
    "_safe_normalize",
    "_stable_reciprocal",
    "_stable_divide",
    "_stable_norm",
    "_compute_condition_number",
    "_regularize_metric",
    # Interfaz pública principal
    "build_signal",
    "create_immune_watcher",
    # Registros y esquemas
    "VALIDATOR_REGISTRY",
    "SIGNAL_SCHEMA",
    "BETTI_INDICES",
]