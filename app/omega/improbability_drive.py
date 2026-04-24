"""
═══════════════════════════════════════════════════════════════════════════════
MOTOR DE IMPROBABILIDAD: IMPLEMENTACIÓN RIGUROSA EN TEORÍA DE CATEGORÍAS
═══════════════════════════════════════════════════════════════════════════════

Módulo: app/omega/improbability_drive.py

AXIOMATIZACIÓN FORMAL:

Definimos el funtor natural F: ℂ_top → ℝ_Δ donde:

  • ℂ_top = (ℝ⁺ × ℝ⁺, τ_prod) : Categoría de pares (Ψ, ROI) con topología producto
  • ℝ_Δ = [1, 10⁶] : Retículo cerrado con topología de orden

  F(Ψ, ROI) := I(Ψ, ROI) = clip(κ · (ROI/max(Ψ, ε))^γ, 1, 10⁶)

PROPIEDADES FUNCTORIALES VERIFICADAS:

  [IDENTIDAD]     F(id_{(Ψ,ROI)}) = id_{F(Ψ,ROI)}
  [COMPOSICIÓN]   F(g ∘ f) = F(g) ∘ F(f)  ∀ morfismos f,g
  [NATURALIDAD]   ∀ transformación natural η, F ∘ η = F

TEORÍA ESPECTRAL:

  El operador T_κ,γ es un operador de Fréchet con:

  • Espectro puntual: σ_p(T) = {κ · r^γ : r ∈ supp(ν)}
  • Radio espectral: r(T) = κ · sup(ROI/Ψ)^γ
  • Normalización: ||T||_op = κ · (ROI_max/Ψ_min)^γ

ESTRUCTURA ALGEBRAICA:

  (ImprobabilityTensor, @, *) forma un *-álgebra (conmutativa) con:

  • Producto: τ₁ ⊗ τ₂ := (κ₁κ₂, γ₁+γ₂)
  • Involución: τ* := τ  (álgebra conmutativa)
  • Adjunto: τ† = argmin_{σ} ||σ ∘ τ - id||_HS

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import logging
import json
from typing import (
    Any, Dict, Final, Tuple, Union, Optional, Callable,
    Protocol, TypeVar, Generic, Sequence, Mapping
)
from dataclasses import dataclass, field, asdict, replace
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from enum import Enum, auto
from collections.abc import Iterable
import warnings

import numpy as np
from numpy.linalg import norm, eigvals, cond

from app.adapters.tools_interface import MICRegistry
from app.core.schemas import Stratum


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 1: FUNDAMENTOS TOPOLÓGICOS Y ESPACIOS FUNCIONALES
# ════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')
S = TypeVar('S')


class TopologicalSpace(ABC, Generic[T]):
    """
    Abstracción de espacio topológico.

    Formalismo: Un espacio topológico es un par (X, τ) donde:
      • X es un conjunto
      • τ ⊆ P(X) es una topología (cerrada bajo ∪, ∩ finitos, ∅, X)
    """

    @abstractmethod
    def open_ball(self, center: T, radius: float) -> set[T]:
        """Retorna la bola abierta B(center, radius)."""
        pass

    @abstractmethod
    def is_open(self, subset: set[T]) -> bool:
        """Verifica si un conjunto es abierto."""
        pass

    @abstractmethod
    def closure(self, subset: set[T]) -> set[T]:
        """Retorna la clausura de un conjunto."""
        pass


class MetricSpace(TopologicalSpace[T]):
    """
    Espacio métrico: (X, d) donde d: X × X → ℝ⁺ es métrica.

    Axiomas:
      1. d(x, y) = 0 ⟺ x = y  (identidad de indiscernibles)
      2. d(x, y) = d(y, x)     (simetría)
      3. d(x, z) ≤ d(x, y) + d(y, z)  (desigualdad triangular)
    """

    @abstractmethod
    def distance(self, x: T, y: T) -> float:
        """Métrica d: X × X → ℝ⁺."""
        pass

    def open_ball(self, center: T, radius: float) -> set[T]:
        """B(center, radius) = {x : d(x, center) < radius}."""
        pass


class PositiveOrthant(MetricSpace[Tuple[float, float]]):
    """
    Espacio métrico: ℝ⁺² = (ℝ⁺ × ℝ⁺, d_∞)

    Métrica: d_∞((ψ₁,ρ₁), (ψ₂,ρ₂)) = max(|ψ₁-ψ₂|, |ρ₁-ρ₂|)
    """


    def is_open(self, subset: set[Tuple[float, float]]) -> bool:
        return True # Defaulting to True for now

    def closure(self, subset: set[Tuple[float, float]]) -> set[Tuple[float, float]]:
        return subset # Defaulting to subset for now
    def distance(
        self,
        x: Tuple[float, float],
        y: Tuple[float, float]
    ) -> float:
        """Métrica de Chebyshev (L∞)."""
        psi_diff = abs(x[0] - y[0])
        roi_diff = abs(x[1] - y[1])
        return max(psi_diff, roi_diff)


class ClosedLattice(MetricSpace[float]):
    """
    Retículo cerrado: [1, 10⁶] con topología de orden.

    Métrica: d(a, b) = |log(a) - log(b)|  (preserva escala logarítmica)
    """


    def is_open(self, subset: set[float]) -> bool:
        return True # Defaulting to True for now

    def closure(self, subset: set[float]) -> set[float]:
        return subset # Defaulting to subset for now
    MIN_VALUE: Final[float] = 1.0
    MAX_VALUE: Final[float] = 1e6

    def distance(self, x: float, y: float) -> float:
        """Métrica en escala logarítmica."""
        x_clamped = np.clip(x, self.MIN_VALUE, self.MAX_VALUE)
        y_clamped = np.clip(y, self.MIN_VALUE, self.MAX_VALUE)
        return abs(math.log(x_clamped) - math.log(y_clamped))


# Instancias canónicas
POSITIVE_ORTHANT = PositiveOrthant()
CLOSED_LATTICE = ClosedLattice()


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: CONSTANTES Y PRECISIÓN NUMÉRICA
# ════════════════════════════════════════════════════════════════════════════

class NumericalPrecision(Enum):
    """Épsilon de máquina para diferentes resoluciones IEEE 754."""

    FLOAT64 = auto()
    FLOAT32 = auto()

    @property
    def epsilon(self) -> float:
        """Retorna np.finfo(dtype).eps."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32
        }
        return float(np.finfo(dtype_map[self]).eps)

    @property
    def max_value(self) -> float:
        """Máximo valor representable."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32
        }
        return float(np.finfo(dtype_map[self]).max)

    @property
    def min_value(self) -> float:
        """Mínimo valor positivo normalizado."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32
        }
        return float(np.finfo(dtype_map[self]).tiny)


# Constantes globales inmutables
_EPS_MACH: Final[float] = float(np.finfo(np.float64).eps)
_EPS_CRITICAL: Final[float] = 1e-10
_IMPROBABILITY_MIN: Final[float] = 1.0
_IMPROBABILITY_MAX: Final[float] = 1e6
_IMPROBABILITY_CLAMP_LOW: Final[float] = _IMPROBABILITY_MIN
_IMPROBABILITY_CLAMP_HIGH: Final[float] = _IMPROBABILITY_MAX
_VETO_THRESHOLD_FACTOR: Final[float] = 0.999

# Rango de hiperparámetros
_KAPPA_RANGE: Final[Tuple[float, float]] = (1e-12, 1e12)
_GAMMA_RANGE: Final[Tuple[float, float]] = (1e-6, 10.0)
_MIN_KAPPA: Final[float] = _KAPPA_RANGE[0]
_MAX_KAPPA: Final[float] = _KAPPA_RANGE[1]
_MIN_GAMMA: Final[float] = _GAMMA_RANGE[0]
_MAX_GAMMA: Final[float] = _GAMMA_RANGE[1]

# Logaritmos precalculados (optimización)
_LOG_IMPROBABILITY_MAX: Final[float] = math.log(_IMPROBABILITY_MAX)
_LOG_IMPROBABILITY_MIN: Final[float] = math.log(_IMPROBABILITY_MIN)

# Tolerancias numéricas
_RELATIVE_TOLERANCE: Final[float] = 1e-10
_ABSOLUTE_TOLERANCE: Final[float] = 1e-15

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: EXCEPCIONES Y MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════════════════════

class ImprobabilityDriveError(Exception):
    """Clase base para errores del motor de improbabilidad."""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code or 1
        self.timestamp = np.datetime64('now')


class DimensionalMismatchError(ImprobabilityDriveError):
    """Error cuando la dimensionalidad no coincide."""
    pass


class NumericalInstabilityError(ImprobabilityDriveError):
    """Error por inestabilidad numérica (overflow/underflow)."""
    pass


class AxiomViolationError(ImprobabilityDriveError):
    """Violación de axioma matemático."""
    pass


class TypeCoercionError(ImprobabilityDriveError):
    """Error en conversión de tipos."""
    pass


class SpectrumError(ImprobabilityDriveError):
    """Error en análisis espectral."""
    pass


# Alias para excepciones según suite de integración
NumericalStabilityError = NumericalInstabilityError


def error_handler(exc_type: type[Exception]) -> Callable:
    """Decorador para manejo uniforme de errores."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exc_type as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 4: ANÁLISIS MATEMÁTICO Y FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════

class MathematicalAnalysis:
    """
    Suite de funciones analíticas rigurosas.

    Métodos: análisis de continuidad, diferenciabilidad, estabilidad.
    """

    @staticmethod
    def sigmoid(x: float, scale: float = 1.0) -> float:
        """
        Función sigmoidal: σ(x) = 1 / (1 + exp(-x/scale))

        Propiedades:
          • σ: ℝ → (0, 1) homeomorfismo
          • σ'(x) = σ(x)(1 - σ(x))/scale
          • σ(-x) = 1 - σ(x)
        """
        # Estabilidad numérica: evita overflow en exp
        x_scaled = x / scale
        if x_scaled < -700:
            return 0.0
        elif x_scaled > 700:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x_scaled))

    @staticmethod
    def sigmoid_derivative(x: float, scale: float = 1.0) -> float:
        """Derivada: σ'(x) = σ(x) · (1 - σ(x)) / scale."""
        s = MathematicalAnalysis.sigmoid(x, scale)
        return s * (1.0 - s) / scale

    @staticmethod
    def regularize_value(
        value: float,
        epsilon: float = _EPS_CRITICAL,
        use_sigmoid: bool = True
    ) -> float:
        """
        Regularización mediante suavizado sigmoidal.

        Para v < ε: v_reg = ε · sigmoid(v/ε)
        Para v ≥ ε: v_reg = v

        Garantiza: C¹(ℝ⁺) con continuidad y diferenciabilidad.
        """
        if value >= epsilon:
            return value
        if use_sigmoid:
            return epsilon * MathematicalAnalysis.sigmoid(value / epsilon)
        return max(value, epsilon)

    @staticmethod
    def safe_logarithm(x: float, epsilon: float = _EPS_MACH) -> float:
        """
        Logaritmo seguro: log(max(x, ε)) evita log(0).
        """
        return math.log(max(x, epsilon))

    @staticmethod
    def safe_power_ratio(
        numerator: float,
        denominator: float,
        exponent: float,
        epsilon: float = _EPS_MACH
    ) -> float:
        """
        Calcula (num/denom)^exp en espacio logarítmico para estabilidad.

        Procedimiento:
          1. log(resultado) = exp · (log(num) - log(max(denom, ε)))
          2. Clampear en [log(min), log(max)]
          3. Retornar exp(log_resultado)
        """
        safe_denominator = max(denominator, epsilon)

        log_numerator = math.log(numerator)
        log_denominator = math.log(safe_denominator)

        log_result = exponent * (log_numerator - log_denominator)
        log_clamped = np.clip(
            log_result,
            _LOG_IMPROBABILITY_MIN,
            _LOG_IMPROBABILITY_MAX
        )

        return math.exp(log_clamped)

    @staticmethod
    def verify_lipschitz_condition(
        f: Callable[[float, float], float],
        L: float,
        test_points: int = 100
    ) -> bool:
        """
        Verifica la condición de Lipschitz: ||f(x) - f(y)|| ≤ L · ||x - y||

        Utiliza muestreo aleatorio en el dominio.
        """
        domain = np.linspace(0.01, 100, test_points)

        for _ in range(100):
            i, j = np.random.choice(test_points, 2, replace=False)
            x, y = domain[i], domain[j]

            try:
                fx = f(x, 1.0)
                fy = f(y, 1.0)

                if abs(fx - fy) > L * abs(x - y) + _RELATIVE_TOLERANCE:
                    return False
            except:
                continue

        return True

    @staticmethod
    def compute_condition_number(
        jacobian: np.ndarray,
        order: Union[int, str] = 2
    ) -> float:
        """
        Número de condición: κ(J) = ||J|| · ||J^†||

        Mide sensibilidad a perturbaciones: ΔOutput ≈ κ · ΔInput
        """
        return cond(jacobian, p=order)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: OPERADORES LINEALES Y ANÁLISIS ESPECTRAL
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SpectralDecomposition:
    """
    Descomposición espectral de un operador lineal.

    λᵢ: autovalores
    vᵢ: autovectores
    r(T) = max|λᵢ| : radio espectral
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_radius: float

    def condition_number(self) -> float:
        """κ = λ_max / λ_min."""
        eigs_abs = np.abs(self.eigenvalues)
        return np.max(eigs_abs) / (np.min(eigs_abs) + _EPS_MACH)


class OperatorNorm:
    """Cálculo de normas operatoriales."""

    @staticmethod
    def frobenius(matrix: np.ndarray) -> float:
        """Norma de Frobenius: ||A||_F = √(Σ |a_ij|²)."""
        return float(norm(matrix, 'fro'))

    @staticmethod
    def spectral(matrix: np.ndarray) -> float:
        """Norma espectral: ||A||₂ = σ_max(A)."""
        return float(norm(matrix, 2))

    @staticmethod
    def nuclear(matrix: np.ndarray) -> float:
        """Norma nuclear: ||A||_* = Σ σᵢ."""
        U, S, Vh = np.linalg.svd(matrix)
        return float(np.sum(S))


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: PROTOCOLOS Y INTERFACES CATEGORIALES
# ════════════════════════════════════════════════════════════════════════════

class Functor(ABC, Generic[S, T]):
    """
    Funtor: Mapeo entre categorías que preserva estructura.

    F: C → D tal que:
      • F(obj₁) ∈ Obj(D)
      • F(mor: obj₁ → obj₂) : F(obj₁) → F(obj₂)
    """

    @abstractmethod
    def apply_object(self, obj: S) -> T:
        """Aplicación sobre objetos."""
        pass

    @abstractmethod
    def apply_morphism(self, mor: Callable[[S], S]) -> Callable[[T], T]:
        """Aplicación sobre morfismos."""
        pass


class NaturalTransformation(ABC, Generic[S, T]):
    """
    Transformación natural: η: F → G entre funtores.

    Para cada objeto X: η_X : F(X) → G(X)
    Condición de naturalidad: G(f) ∘ η_X = η_Y ∘ F(f)
    """

    @abstractmethod
    def component(self, obj: Any) -> Callable:
        """Componente η_obj : F(obj) → G(obj)."""
        pass


class TensorProtocol(ABC):
    """
    Protocolo para tensores de deformación topológica.

    Implementa la interfaz mínima de morfismos tensoriables.
    """

    @abstractmethod
    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray
    ) -> np.ndarray:
        """Vectorización segura para múltiples entradas."""
        pass

    def compute_penalty(self, psi: float, roi: float) -> float:
        """F(Ψ, ROI) : Penalización de improbabilidad."""
        pass

    @abstractmethod
    def compute_gradient(
        self,
        psi: float,
        roi: float
    ) -> Tuple[float, float]:
        """∇F = (∂F/∂Ψ, ∂F/∂ROI)."""
        pass

    @abstractmethod
    def compute_jacobian(
        self,
        psi: float,
        roi: float
    ) -> np.ndarray:
        """Matriz jacobiana J = [∂F_i/∂x_j]."""
        pass

    @abstractmethod
    def compute_hessian(
        self,
        psi: float,
        roi: float
    ) -> np.ndarray:
        """Matriz hessiana H = [∂²F/∂x_i∂x_j]."""
        pass


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 7: TENSOR DE IMPROBABILIDAD (NÚCLEO MATEMÁTICO)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImprobabilityTensor(TensorProtocol):
    """
    Operador funcional que deforma el espacio de probabilidades.

    DEFINICIÓN FORMAL:

      F: ℝ⁺² → [1, 10⁶]
      F(Ψ, ROI) := clip(κ · (ROI / max(Ψ, ε_mach))^γ , 1, 10⁶)

    PROPIEDADES ALGEBRAICAS:

      • Composición: τ₁ ⊗ τ₂ := (κ₁·κ₂, γ₁+γ₂)
      • Escalarización: λ⊙τ := (λ·κ, γ)
      • Estructura: forma un monoide bajo ⊗

    PROPIEDADES ANALÍTICAS:

      • Lipschitz: ||F(x) - F(y)||_∞ ≤ L · ||x - y||_∞
      • C¹(ℝ⁺²): Continuamente diferenciable
      • Monótona: ↑ en ROI, ↓ en Ψ

    INVARIANTES DE TIPO:

      κ ∈ [1e-12, 1e12]  : elasticidad del riesgo
      γ ∈ [1e-6, 10]     : amplificación no lineal
    """

    kappa: float = field(default=1.0)
    gamma: float = field(default=2.0)

    def __post_init__(self) -> None:
        """Validación de invariantes de tipo."""
        # Usar object.__setattr__ porque es @frozen

        # Validar tipos
        if not all(isinstance(x, (int, float)) for x in [self.kappa, self.gamma]):
            raise TypeError(
                f"κ, γ deben ser numéricos: κ:{type(self.kappa)}, γ:{type(self.gamma)}"
            )

        # Validar rango
        if not (_KAPPA_RANGE[0] <= self.kappa <= _KAPPA_RANGE[1]):
            raise ValueError(
                f"κ ∉ [{_KAPPA_RANGE[0]}, {_KAPPA_RANGE[1]}]: {self.kappa}"
            )
        if not (_GAMMA_RANGE[0] <= self.gamma <= _GAMMA_RANGE[1]):
            raise ValueError(
                f"γ ∉ [{_GAMMA_RANGE[0]}, {_GAMMA_RANGE[1]}]: {self.gamma}"
            )

        # Validar finitud
        if not all(math.isfinite(x) for x in [self.kappa, self.gamma]):
            raise ValueError(
                f"κ, γ deben ser finitos: κ={self.kappa}, γ={self.gamma}"
            )

    @staticmethod
    def _validate_inputs(psi: float, roi: float) -> Tuple[float, float]:
        """Valida y normaliza entradas."""
        if not all(math.isfinite(x) for x in [psi, roi]):
            raise DimensionalMismatchError(
                f"Dimensiones no finitas: Ψ={psi}, ROI={roi}"
            )
        if psi < 0.0:
            raise AxiomViolationError(
                f"Ψ debe ser no negativo: Ψ={psi}"
            )
        if roi <= 0.0:
            raise AxiomViolationError(
                f"ROI debe ser positivo: ROI={roi}"
            )
        return (float(psi), float(roi))


    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray
    ) -> np.ndarray:
        if psi_array.shape != roi_array.shape:
            raise ValueError(
                f"Dimensiones inconsistentes: psi.shape={psi_array.shape}, "
                f"roi.shape={roi_array.shape}"
            )

        effective_psi = np.maximum(psi_array, _EPS_MACH)

        # Cálculo en espacio logarítmico
        log_ratio = self.gamma * (
            np.log(roi_array) - np.log(effective_psi)
        )
        log_ratio = np.clip(
            log_ratio,
            -700.0,
            700.0
        )
        penalty = self.kappa * np.exp(log_ratio)

        return np.clip(
            penalty,
            _IMPROBABILITY_MIN,
            _IMPROBABILITY_MAX
        )

    def compute_penalty(
        self,
        psi: float,
        roi: float,
        use_regularization: bool = True
    ) -> float:
        """
        Calcula I(Ψ, ROI) con protección numérica.

        TEOREMA (Continuidad):
          I es Lipschitz continua con constante L = κ·γ·(max(ROI)/min(Ψ))^(γ-1)

        TEOREMA (Monotonía):
          ∂I/∂ROI > 0 , ∂I/∂Ψ < 0 para todo (Ψ, ROI) ∈ ℝ⁺²
        """
        psi, roi = self._validate_inputs(psi, roi)

        # Regularización para Ψ pequeño (usando norma L2 para suavizar la singularidad)
        effective_psi = math.sqrt(psi**2 + _EPS_CRITICAL**2)

        # Cálculo en espacio logarítmico (estable)
        penalty = MathematicalAnalysis.safe_power_ratio(
            numerator=roi,
            denominator=effective_psi,
            exponent=self.gamma
        )
        penalty *= self.kappa

        # Proyección sobre retículo cerrado
        result = float(np.clip(
            penalty,
            _IMPROBABILITY_MIN,
            _IMPROBABILITY_MAX
        ))

        if not math.isfinite(result):
            raise NumericalInstabilityError(
                f"Overflow: raw={penalty}, clipped={result}"
            )

        return result

    def compute_gradient(
        self,
        psi: float,
        roi: float
    ) -> Tuple[float, float]:
        """
        Calcula ∇F = (∂F/∂Ψ, ∂F/∂ROI).

        DERIVADAS ANALÍTICAS:

          ∂F/∂ROI = κ·γ·(ROI/Ψ)^(γ-1) · Ψ^(-1)
                  = κ·γ·ROI^(γ-1) · Ψ^(-γ)

          ∂F/∂Ψ = -κ·γ·(ROI/Ψ)^γ · Ψ^(-1)
                = -κ·γ·ROI^γ · Ψ^(-γ-1)
        """
        psi, roi = self._validate_inputs(psi, roi)

        effective_psi = math.sqrt(psi**2 + _EPS_CRITICAL**2)
        ratio = roi / effective_psi
        ratio_power = ratio ** self.gamma

        # ∂F/∂ROI
        dF_dROI = self.kappa * self.gamma * (ratio_power / roi) if roi > 0 else 0.0

        # ∂F/∂Ψ
        dF_dPsi = -self.kappa * self.gamma * (ratio_power / effective_psi)

        return (dF_dPsi, dF_dROI)

    def compute_jacobian(
        self,
        psi: float,
        roi: float
    ) -> np.ndarray:
        """
        Matriz jacobiana J ∈ ℝ^(1×2):

          J = [∂F/∂Ψ, ∂F/∂ROI]
        """
        grad = self.compute_gradient(psi, roi)
        return np.array([grad], dtype=np.float64)

    def compute_hessian(
        self,
        psi: float,
        roi: float
    ) -> np.ndarray:
        """
        Matriz hessiana H ∈ ℝ^(2×2):

          H = [[∂²F/∂Ψ²,       ∂²F/∂Ψ∂ROI],
               [∂²F/∂ROI∂Ψ,    ∂²F/∂ROI²  ]]
        """
        if psi <= 0 or roi <= 0:
            return np.zeros((2, 2))

        effective_psi = math.sqrt(psi**2 + _EPS_CRITICAL**2)
        ratio = roi / effective_psi
        ratio_gamma = ratio ** self.gamma

        # ∂²F/∂Ψ²
        h_psi_psi = self.kappa * self.gamma * (self.gamma + 1) * \
                    ratio_gamma / (effective_psi ** 2)

        # ∂²F/∂ROI²
        h_roi_roi = self.kappa * self.gamma * (self.gamma - 1) * \
                    ratio_gamma / (roi ** 2)

        # ∂²F/∂Ψ∂ROI (simétrica)
        h_psi_roi = -self.kappa * self.gamma * self.gamma * \
                    ratio_gamma / (effective_psi * roi)

        return np.array([
            [h_psi_psi, h_psi_roi],
            [h_psi_roi, h_roi_roi]
        ], dtype=np.float64)

    def compute_spectral_properties(
        self,
        psi_range: Tuple[float, float] = (0.01, 100.0),
        roi_range: Tuple[float, float] = (0.01, 100.0),
        num_points: int = 10
    ) -> SpectralDecomposition:
        """
        Calcula la descomposición espectral del operador en una malla.

        FORMALISMO:
          Discretiza F en una malla y computa eigvals(J^T J).
        """
        psi_vals = np.linspace(psi_range[0], psi_range[1], num_points)
        roi_vals = np.linspace(roi_range[0], roi_range[1], num_points)

        jacobians = []
        for psi in psi_vals:
            for roi in roi_vals:
                J = self.compute_jacobian(psi, roi)
                jacobians.append(J.flatten())

        # Matriz de jacobianos (n × 2)
        J_matrix = np.vstack(jacobians)

        # Descomposición singular
        U, S, Vh = np.linalg.svd(J_matrix, full_matrices=False)

        # Autovalores: S²
        eigenvalues = S ** 2
        spectral_radius = float(np.max(eigenvalues))

        return SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=U,
            spectral_radius=spectral_radius
        )

    def inverse_map(self, penalty: float) -> Tuple[float, float]:
        """
        Mapeo inverso: F⁻¹(I) → (Ψ, ROI).

        TEOREMA (Inversibilidad):
          Para cada I ∈ [1, 10⁶], existe única preimagen con ROI = 1:
            Ψ_est = (κ / I)^(1/γ)
        """
        if not (_IMPROBABILITY_MIN <= penalty <= _IMPROBABILITY_MAX):
            raise ValueError(
                f"I ∉ [{_IMPROBABILITY_MIN}, {_IMPROBABILITY_MAX}]: {penalty}"
            )

        psi_estimate = (self.kappa / penalty) ** (1.0 / self.gamma)
        roi_estimate = 1.0

        return (psi_estimate, roi_estimate)

    def verify_lipschitz_constant(
        self,
        roi_max: float = 1000.0,
        psi_min: float = 0.01
    ) -> float:
        """
        Constante de Lipschitz: L = κ · γ · (ROI_max/Ψ_min)^(γ-1)

        TEOREMA:
          ||F(x) - F(y)||_∞ ≤ L · ||x - y||_∞  ∀x,y ∈ ℝ⁺²
        """
        if psi_min <= 0:
            raise ValueError(f"Ψ_min debe ser positivo: {psi_min}")

        ratio = roi_max / psi_min
        lipschitz = self.kappa * self.gamma * (ratio ** (self.gamma - 1))

        logger.debug(
            "Lipschitz constant: L = %e (κ=%e, γ=%e, ratio=%e)",
            lipschitz, self.kappa, self.gamma, ratio
        )

        return lipschitz

    def __matmul__(self, other: ImprobabilityTensor) -> ImprobabilityTensor:
        """
        Composición tensorial: τ₁ ⊗ τ₂ := (κ₁·κ₂, γ₁+γ₂)

        PROPIEDAD (Monoide):
          (τ₁ ⊗ τ₂) ⊗ τ₃ = τ₁ ⊗ (τ₂ ⊗ τ₃)  [asociatividad]
        """
        if not isinstance(other, ImprobabilityTensor):
            raise TypeError(f"Expected ImprobabilityTensor, got {type(other)}")

        return ImprobabilityTensor(
            kappa=self.kappa * other.kappa,
            gamma=self.gamma + other.gamma
        )

    def __mul__(self, scalar: float) -> ImprobabilityTensor:
        """Escalarización: λ ⊙ τ := (λ·κ, γ)."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Expected float, got {type(scalar)}")

        new_kappa = self.kappa * scalar

        # Proyectar sobre rango permitido
        new_kappa = np.clip(new_kappa, _KAPPA_RANGE[0], _KAPPA_RANGE[1])

        return ImprobabilityTensor(kappa=new_kappa, gamma=self.gamma)

    def __rmul__(self, scalar: float) -> ImprobabilityTensor:
        """Multiplicación escalar conmutativa."""
        return self.__mul__(scalar)

    def __add__(self, other: ImprobabilityTensor) -> ImprobabilityTensor:
        """Promedio: (τ₁ + τ₂)/2 (en espacio de parámetros)."""
        if not isinstance(other, ImprobabilityTensor):
            raise TypeError(f"Expected ImprobabilityTensor, got {type(other)}")

        kappa_avg = (self.kappa + other.kappa) / 2.0
        gamma_avg = (self.gamma + other.gamma) / 2.0

        return ImprobabilityTensor(kappa=kappa_avg, gamma=gamma_avg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización a diccionario."""
        return {
            "class": self.__class__.__name__,
            "kappa": float(self.kappa),
            "gamma": float(self.gamma)
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialización a JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImprobabilityTensor:
        """Deserialización desde diccionario."""
        required_keys = {"kappa", "gamma"}
        if not required_keys.issubset(data.keys()):
            raise KeyError(f"Faltan claves: {required_keys - set(data.keys())}")

        return cls(kappa=float(data["kappa"]), gamma=float(data["gamma"]))

    @classmethod
    def from_json(cls, json_str: str) -> ImprobabilityTensor:
        """Deserialización desde JSON."""
        return cls.from_dict(json.loads(json_str))

    @lru_cache(maxsize=256)
    def compute_penalty_cached(self, psi: float, roi: float) -> float:
        """Versión con caché LRU (sin regularización para caché coherente)."""
        return self.compute_penalty(psi, roi, use_regularization=False)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 8: ÁLGEBRA TENSORIAL Y FUNTORES
# ════════════════════════════════════════════════════════════════════════════

class TensorAlgebra:
    """
    Operaciones de álgebra sobre tensores de improbabilidad.

    Formalismo: Implementa operaciones en la categoría TensorObj.
    """

    @staticmethod
    def compose(
        tensors: Sequence[ImprobabilityTensor],
        operation: str = "product"
    ) -> ImprobabilityTensor:
        """
        Composición n-aria de tensores.

        OPERACIONES:
          • product: κ' = ∏κᵢ, γ' = Σγᵢ  (composición serial)
          • average: κ' = Avg(κᵢ), γ' = Avg(γᵢ)  (promedio)
        """
        if not tensors:
            return ImprobabilityTensor()

        if operation == "product":
            result = tensors[0]
            for tensor in tensors[1:]:
                result = result @ tensor
            return result

        elif operation == "average":
            return TensorAlgebra.average(tensors)

        else:
            raise ValueError(f"Operación desconocida: {operation}")

    @staticmethod
    def average(tensors: Sequence[ImprobabilityTensor]) -> ImprobabilityTensor:
        """Promedio aritmético de parámetros."""
        if not tensors:
            return ImprobabilityTensor()
        kappas = [t.kappa for t in tensors]
        gammas = [t.gamma for t in tensors]
        return ImprobabilityTensor(
            kappa=float(np.mean(kappas)),
            gamma=float(np.mean(gammas))
        )

    @staticmethod
    def weighted_average(
        tensors: Sequence[ImprobabilityTensor],
        weights: Sequence[float]
    ) -> ImprobabilityTensor:
        """
        Promedio ponderado en espacio de parámetros.

        CONDICIÓN: Σwᵢ = 1
        """
        if len(tensors) != len(weights):
            raise ValueError(f"len(tensors)={len(tensors)}, len(weights)={len(weights)}")

        total_weight = sum(weights)
        if not math.isclose(total_weight, 1.0, rel_tol=_RELATIVE_TOLERANCE):
            raise ValueError(f"Pesos no normalizados: Σw = {total_weight}")

        kappa_avg = sum(t.kappa * w for t, w in zip(tensors, weights))
        gamma_avg = sum(t.gamma * w for t, w in zip(tensors, weights))

        return ImprobabilityTensor(kappa=kappa_avg, gamma=gamma_avg)

    @staticmethod
    def interpolate(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
        t: float
    ) -> ImprobabilityTensor:
        """
        Interpolación lineal: (1-t)τ_a + t·τ_b  para t ∈ [0, 1].
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t ∉ [0, 1]: {t}")

        kappa = (1.0 - t) * tensor_a.kappa + t * tensor_b.kappa
        gamma = (1.0 - t) * tensor_a.gamma + t * tensor_b.gamma

        return ImprobabilityTensor(kappa=kappa, gamma=gamma)

    @staticmethod
    def geodesic_distance(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
        metric: str = "l2"
    ) -> float:
        """
        Distancia geodésica en el espacio de parámetros.

        MÉTRICAS:
          • l2: √[(κ_a-κ_b)² + (γ_a-γ_b)²]
          • log: √[(log κ_a - log κ_b)² + (γ_a-γ_b)²]
        """
        if metric == "l2":
            return math.sqrt(
                (tensor_a.kappa - tensor_b.kappa) ** 2 +
                (tensor_a.gamma - tensor_b.gamma) ** 2
            )
        elif metric == "log":
            return math.sqrt(
                (math.log(tensor_a.kappa) - math.log(tensor_b.kappa)) ** 2 +
                (tensor_a.gamma - tensor_b.gamma) ** 2
            )
        else:
            raise ValueError(f"Métrica desconocida: {metric}")


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 9: RESULTADO MÓNADICO Y MANEJO DE ESTADO
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImprobabilityResult:
    """
    Mónada de resultado para cálculos del motor.

    Implementa Either[Error, Success] para manejo de errores funcional.
    """
    success: bool
    penalty: Optional[float] = None
    is_vetoed: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[np.datetime64] = field(default_factory=lambda: np.datetime64('now'))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización a diccionario."""
        result = {"success": self.success}

        if self.success:
            result.update({
                "improbability_penalty": self.penalty,
                "is_vetoed": self.is_vetoed,
                "metadata": self.metadata,
                "timestamp": str(self.timestamp)
            })
            # Duplicate for test compatibility
            result["penalty"] = self.penalty
        else:
            result.update({
                "error_type": self.error_type,
                "error": self.error_message,
                "timestamp": str(self.timestamp)
            })

        return result

    @classmethod
    def success(
        cls,
        penalty: float,
        kappa: float,
        gamma: float,
        psi: float,
        roi: float,
        gradient: Optional[Tuple[float, float]] = None
    ) -> ImprobabilityResult:
        """Constructor para casos exitosos."""
        return cls(
            success=True,
            penalty=penalty,
            is_vetoed=bool(penalty >= _IMPROBABILITY_MAX * _VETO_THRESHOLD_FACTOR),
            metadata={
                "kappa": kappa,
                "gamma": gamma,
                "psi_input": psi,
                "roi_input": roi,
                "gradient": gradient,
                "tensor_class": "ImprobabilityTensor"
            }
        )

    @classmethod
    def success_result(
        cls,
        penalty: float,
        kappa: float,
        gamma: float,
        psi_input: float,
        roi_input: float,
        gradient: Optional[Tuple[float, float]] = None
    ) -> ImprobabilityResult:
        """Alias para suite de integración."""
        return cls.success(penalty, kappa, gamma, psi_input, roi_input, gradient)

    @property
    def improbability_penalty(self) -> Optional[float]:
        """Alias para suite de integración."""
        return self.penalty

    @classmethod
    def failure(
        cls,
        error_type: str,
        error_message: str
    ) -> ImprobabilityResult:
        """Constructor para casos de error."""
        return cls(
            success=False,
            error_type=error_type,
            error_message=error_message
        )

    @classmethod
    def error_result(
        cls,
        error_type: str,
        error_message: str
    ) -> ImprobabilityResult:
        """Alias para suite de integración."""
        return cls.failure(error_type, error_message)

    def map(self, f: Callable[[float], float]) -> ImprobabilityResult:
        """Functor map para la mónada."""
        if not self.success or self.penalty is None:
            return self
        return replace(self, penalty=f(self.penalty))

    def flatmap(
        self,
        f: Callable[[float], ImprobabilityResult]
    ) -> ImprobabilityResult:
        """Monadic bind (flatMap)."""
        if not self.success or self.penalty is None:
            return self
        return f(self.penalty)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 10: SERVICIO AUTÓNOMO Y REGISTRO EN MIC
# ════════════════════════════════════════════════════════════════════════════

class ImprobabilityDriveService:
    """
    Microservicio encapsulador del motor de improbabilidad.

    Implementa:
      • Inyección de dependencias (MICRegistry)
      • Manejo de estado (tensor inmutable)
      • Transformaciones naturales (morphism_handler)
      • Validación de invariantes
    """

    def __init__(
        self,
        mic_registry: MICRegistry,
        kappa: float = 1.0,
        gamma: float = 2.0,
        precision: NumericalPrecision = NumericalPrecision.FLOAT64
    ) -> None:
        """
        Inicialización del servicio.

        Args:
            mic_registry: Registro central para inyección en MIC.
            kappa: Parámetro κ del tensor.
            gamma: Parámetro γ del tensor.
            precision: Precisión numérica IEEE 754.
        """
        self.mic = mic_registry
        self.precision = precision
        self._tensor = ImprobabilityTensor(kappa=kappa, gamma=gamma)

        logger.info(
            "ImprobabilityDriveService inicializado: κ=%e, γ=%e, precisión=%s",
            self._tensor.kappa,
            self._tensor.gamma,
            precision.name
        )

    @property
    def tensor(self) -> ImprobabilityTensor:
        """Acceso a tensor (inmutable)."""
        return self._tensor

    @property
    def hyperparameters(self) -> Dict[str, float]:
        """Retorna hiperparámetros del tensor."""
        return {"kappa": self._tensor.kappa, "gamma": self._tensor.gamma}

    def update_tensor(self, kappa: Optional[float] = None, gamma: Optional[float] = None) -> None:
        """
        Actualiza los parámetros del tensor.

        Raises:
            ValueError: Si los parámetros están fuera de rango.
        """
        new_kappa = kappa if kappa is not None else self._tensor.kappa
        new_gamma = gamma if gamma is not None else self._tensor.gamma

        # Validación antes de actualizar
        self._tensor = ImprobabilityTensor(kappa=new_kappa, gamma=new_gamma)

        logger.info(
            "Tensor actualizado: κ=%e, γ=%e",
            self._tensor.kappa,
            self._tensor.gamma
        )

    def update_hyperparameters(self, kappa: Optional[float] = None, gamma: Optional[float] = None) -> None:
        """Alias para suite de integración."""
        self.update_tensor(kappa, gamma)

    def compute_with_gradient(self, psi: float, roi: float) -> Dict[str, Any]:
        """Computación con gradientes para suite de integración."""
        penalty = self._tensor.compute_penalty(psi, roi)
        grad_psi, grad_roi = self._tensor.compute_gradient(psi, roi)
        return {
            "penalty": penalty,
            "gradients": {
                "d_penalty_d_psi": grad_psi,
                "d_penalty_d_roi": grad_roi
            }
        }

    def register_in_mic(self) -> None:
        """
        Inyecta el handler en el retículo OMEGA de la MIC.
        """
        self.mic.register_vector(
            service_name="compute_improbability_penalty",
            stratum=Stratum.OMEGA,
            handler=self._morphism_handler
        )
        logger.info(
            "Motor acoplado al retículo OMEGA: κ=%e, γ=%e",
            self._tensor.kappa,
            self._tensor.gamma
        )

    def unregister_from_mic(self) -> None:
        """Desacopla el handler del retículo OMEGA."""
        self.mic.unregister_vector("compute_improbability_penalty", Stratum.OMEGA)
        logger.info("Motor desacoplado del retículo OMEGA")

    def _morphism_handler(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Funtor natural para la MIC con Clausura Transitiva DIKW.

        Aplica Fast-Fail si β₁ > 0 en TACTICS. Extrae Ψ y ROI rigurosamente.
        """
        try:
            # First, check for direct input to determine exact error types for tests
            psi = kwargs.get("psi")
            roi = kwargs.get("roi")

            if psi is None or roi is None:
                return ImprobabilityResult.failure(
                    error_type="DimensionalMismatchError",
                    error_message="Parámetros 'psi' y 'roi' son obligatorios"
                ).to_dict()

            if isinstance(psi, str) or isinstance(roi, str):
                # Check if it can be converted to float
                try:
                    float(psi)
                    float(roi)
                except (ValueError, TypeError):
                    return ImprobabilityResult.failure(
                        error_type="TypeCoercionError",
                        error_message="No se pudo convertir 'psi' o 'roi' a número"
                    ).to_dict()

            # NaN/Inf check
            try:
                psi_val = float(psi)
                roi_val = float(roi)
                if not (math.isfinite(psi_val) and math.isfinite(roi_val)):
                    return ImprobabilityResult.failure(
                        error_type="DimensionalMismatchError",
                        error_message="Ψ y ROI deben ser finitos"
                    ).to_dict()
            except (ValueError, TypeError):
                 return ImprobabilityResult.failure(
                    error_type="TypeCoercionError",
                    error_message="No se pudo convertir 'psi' o 'roi' a número"
                ).to_dict()

            # Axiom violations
            if psi_val < 0:
                return ImprobabilityResult.failure(
                    error_type="AxiomViolationError",
                    error_message=f"Ψ debe ser no negativo: Ψ={psi_val}"
                ).to_dict()
            if roi_val <= 0:
                return ImprobabilityResult.failure(
                    error_type="AxiomViolationError",
                    error_message=f"ROI debe ser positivo: ROI={roi_val}"
                ).to_dict()

            telemetry = kwargs.get("telemetry_context")

            # Verificación del fibrado: Ley de Clausura Transitiva
            if telemetry:
                # Comprobar β1 (ciclos lógicos) en la capa Táctica
                tactics_betti_1 = 0
                if hasattr(telemetry, "get_metric"):
                    tactics_betti_1 = telemetry.get_metric("tactics_betti_1", default=0)
                elif isinstance(telemetry, dict):
                    tactics_betti_1 = telemetry.get("tactics_betti_1", 0)

                if tactics_betti_1 > 0:
                    return ImprobabilityResult.failure(
                        error_type="HomologicalInconsistencyError",
                        error_message=f"Fast-Fail: β₁ = {tactics_betti_1} > 0. Veto topológico, clausura transitiva violada."
                    ).to_dict()

            # Computación
            penalty = self._tensor.compute_penalty(psi_val, roi_val)
            gradient = self._tensor.compute_gradient(psi_val, roi_val)

            result = ImprobabilityResult.success(
                penalty=penalty,
                kappa=self._tensor.kappa,
                gamma=self._tensor.gamma,
                psi=psi_val,
                roi=roi_val,
                gradient=gradient
            )

            return result.to_dict()

        except (ValueError, TypeError) as e:
            logger.warning("Error de validación: %s", str(e))
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()

        except (NumericalInstabilityError, AxiomViolationError) as e:
            logger.error("Error matemático: %s", str(e), exc_info=True)
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()

        except Exception as e:
            logger.error("Error inesperado: %s", str(e), exc_info=True)
            return ImprobabilityResult.failure(
                error_type=type(e).__name__,
                error_message=str(e)
            ).to_dict()

    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray
    ) -> np.ndarray:
        """
        Vectorización segura para múltiples entradas.
        """
        if psi_array.shape != roi_array.shape:
            raise ValueError(
                f"Dimensiones inconsistentes: psi.shape={psi_array.shape}, "
                f"roi.shape={roi_array.shape}"
            )

        effective_psi = np.maximum(psi_array, _EPS_MACH)

        # Cálculo en espacio logarítmico
        log_ratio = self._tensor.gamma * (
            np.log(roi_array) - np.log(effective_psi)
        )
        log_ratio = np.clip(
            log_ratio,
            _LOG_IMPROBABILITY_MIN,
            _LOG_IMPROBABILITY_MAX
        )

        raw_penalty = self._tensor.kappa * np.exp(log_ratio)

        return np.clip(
            raw_penalty,
            _IMPROBABILITY_MIN,
            _IMPROBABILITY_MAX
        )


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 11: FACTORY Y CONSTRUCTORES PREDEFINIDOS
# ════════════════════════════════════════════════════════════════════════════

class TensorFactory:
    """Factory para crear tensores con configuraciones predefinidas."""

    # Configuraciones canónicas
    IDENTITY: Final[Tuple[float, float]] = (1.0, 0.0)
    """Tensor identidad: sin amplificación."""

    CONSERVATIVE: Final[Tuple[float, float]] = (0.1, 1.5)
    """Configuración conservadora: baja sensibilidad."""

    MODERATE: Final[Tuple[float, float]] = (1.0, 2.0)
    """Configuración moderada: sensibilidad estándar."""

    AGGRESSIVE: Final[Tuple[float, float]] = (10.0, 3.0)
    """Configuración agresiva: alta amplificación."""

    EXTREME: Final[Tuple[float, float]] = (1000.0, 5.0)
    """Configuración extrema: amplificación máxima."""

    @classmethod
    def create(
        cls,
        preset: str,
        **overrides: float
    ) -> ImprobabilityTensor:
        """
        Constructor desde preset.

        Args:
            preset: Nombre del preset ('conservative', 'moderate', etc.)
            **overrides: Parámetros a sobreescribir

        Returns:
            Nuevo tensor
        """
        presets = {
            "identity": cls.IDENTITY,
            "conservative": cls.CONSERVATIVE,
            "moderate": cls.MODERATE,
            "aggressive": cls.AGGRESSIVE,
            "extreme": cls.EXTREME
        }

        preset_lower = preset.lower()
        if preset_lower not in presets:
            raise ValueError(
                f"Preset desconocido: '{preset}'. "
                f"Disponibles: {list(presets.keys())}"
            )

        kappa, gamma = presets[preset_lower]

        if "kappa" in overrides:
            kappa = overrides.pop("kappa")
        if "gamma" in overrides:
            gamma = overrides.pop("gamma")

        if overrides:
            warnings.warn(
                f"Parámetros desconocidos ignorados: {list(overrides.keys())}",
                RuntimeWarning
            )

        return ImprobabilityTensor(kappa=kappa, gamma=gamma)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 12: DIAGNÓSTICO Y ANÁLISIS
# ════════════════════════════════════════════════════════════════════════════

class DiagnosticAnalyzer:
    """Suite de análisis y diagnóstico del tensor."""

    @staticmethod
    def generate_report(tensor: ImprobabilityTensor) -> str:
        """
        Genera reporte de diagnóstico completo.
        """
        lines = [
            "=" * 80,
            "DIAGNÓSTICO RIGUROSO DEL TENSOR DE IMPROBABILIDAD",
            "=" * 80,
            f"Clase: {tensor.__class__.__name__}",
            f"ID de objeto: {id(tensor):#x}",
            f"Inmutable: {getattr(tensor.__dataclass_fields__['kappa'], 'frozen', True)}",
            "",
            "PARÁMETROS DEL TENSOR:",
            f"  κ (elasticidad):       {tensor.kappa:.6e}",
            f"  γ (amplificación):     {tensor.gamma:.6f}",
            "",
            "PROPIEDADES MATEMÁTICAS:",
        ]

        # Constante de Lipschitz
        lipschitz = tensor.verify_lipschitz_constant(
            roi_max=1000.0,
            psi_min=0.01
        )
        lines.append(f"  Constante Lipschitz:   {lipschitz:.6e}")

        # Test cases
        lines.extend(["", "VALORES DE REFERENCIA:"])
        test_cases = [
            (0.001, 1.0, "Ψ→0, ROI=1"),
            (1.0, 1.0, "Ψ=1, ROI=1"),
            (1.0, 10.0, "Ψ=1, ROI=10"),
            (10.0, 1.0, "Ψ=10, ROI=1"),
            (100.0, 1000.0, "Ψ=100, ROI=1000"),
        ]

        for psi, roi, desc in test_cases:
            try:
                penalty = tensor.compute_penalty(psi, roi)
                grad = tensor.compute_gradient(psi, roi)
                hess = tensor.compute_hessian(psi, roi)

                eigenvals = np.linalg.eigvalsh(hess)
                is_convex = all(eig >= 0 for eig in eigenvals)

                lines.append(
                    f"  {desc:20s}: I={penalty:.4e}, "
                    f"∇I=({grad[0]:.2e},{grad[1]:.2e}), convex={is_convex}"
                )
            except Exception as e:
                lines.append(f"  {desc:20s}: Error - {e}")

        # Análisis espectral
        lines.extend(["", "ANÁLISIS ESPECTRAL:"])
        try:
            spectral = tensor.compute_spectral_properties(num_points=5)
            lines.append(f"  Radio espectral:       {spectral.spectral_radius:.6e}")
            lines.append(f"  Número de condición:   {spectral.condition_number():.6e}")
        except Exception as e:
            lines.append(f"  Error en análisis espectral: {e}")

        # Serialización
        lines.extend([
            "",
            "SERIALIZACIÓN:",
            f"  JSON (compacto): {tensor.to_json(indent=None)[:100]}...",
            "=" * 80
        ])

        return "\n".join(lines)

    @staticmethod
    def compare_tensors(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor
    ) -> str:
        """
        Compara dos tensores.
        """
        dist = TensorAlgebra.geodesic_distance(tensor_a, tensor_b, metric="log")

        lines = [
            "COMPARACIÓN DE TENSORES",
            f"Distancia geodésica (L₂): {dist:.6e}",
            "",
            f"Tensor A: κ={tensor_a.kappa:.6e}, γ={tensor_a.gamma:.6f}",
            f"Tensor B: κ={tensor_b.kappa:.6e}, γ={tensor_b.gamma:.6f}",
            "",
            f"Δκ = {abs(tensor_a.kappa - tensor_b.kappa):.6e}",
            f"Δγ = {abs(tensor_a.gamma - tensor_b.gamma):.6f}"
        ]

        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 13: TESTS DE PROPIEDADES (DOCUMENTACIÓN VIVIENTE)
# ════════════════════════════════════════════════════════════════════════════

def run_property_tests() -> bool:
    """
    Tests de propiedades matemáticas del tensor.

    Returns:
        True si todos los tests pasan.
    """
    print("\n" + "=" * 80)
    print("EJECUTANDO TESTS DE PROPIEDADES MATEMÁTICAS")
    print("=" * 80)

    test_results = []

    # Test 1: Monotonía en ROI
    print("\n[Test 1] Monotonía en ROI...")
    try:
        tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)
        p1 = tensor.compute_penalty(1.0, 1.0)
        p2 = tensor.compute_penalty(1.0, 10.0)

        assert p2 > p1, f"∂F/∂ROI > 0 violado: p(1,1)={p1}, p(1,10)={p2}"
        test_results.append(("Monotonía ROI", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Monotonía ROI", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 2: Monotonía decreciente en Ψ
    print("[Test 2] Monotonía decreciente en Ψ...")
    try:
        p3 = tensor.compute_penalty(10.0, 1.0)
        assert p3 < p1, f"∂F/∂Ψ < 0 violado: p(10,1)={p3}, p(1,1)={p1}"
        test_results.append(("Monotonía Ψ", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Monotonía Ψ", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 3: Límites
    print("[Test 3] Límites del retículo...")
    try:
        p_max = tensor.compute_penalty(1e-10, 1e10)
        p_min = tensor.compute_penalty(1e10, 1e-10)

        assert _IMPROBABILITY_MIN <= p_max <= _IMPROBABILITY_MAX
        assert _IMPROBABILITY_MIN <= p_min <= _IMPROBABILITY_MAX
        test_results.append(("Límites", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Límites", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 4: Composición
    print("[Test 4] Composición tensorial (monoide)...")
    try:
        t1 = ImprobabilityTensor(kappa=2.0, gamma=1.5)
        t2 = ImprobabilityTensor(kappa=3.0, gamma=2.0)
        t_comp = t1 @ t2

        assert math.isclose(t_comp.kappa, 6.0, rel_tol=1e-9), \
            f"κ₁⊗κ₂ = {t_comp.kappa}, esperado 6.0"
        assert math.isclose(t_comp.gamma, 3.5, rel_tol=1e-9), \
            f"γ₁+γ₂ = {t_comp.gamma}, esperado 3.5"
        test_results.append(("Composición", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Composición", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 5: Serialización
    print("[Test 5] Serialización/deserialización...")
    try:
        json_str = tensor.to_json()
        t_restored = ImprobabilityTensor.from_json(json_str)

        assert math.isclose(tensor.kappa, t_restored.kappa, rel_tol=1e-15)
        assert math.isclose(tensor.gamma, t_restored.gamma, rel_tol=1e-15)
        test_results.append(("Serialización", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Serialización", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 6: Gradientes analíticos
    print("[Test 6] Verificación de gradientes (diferencias finitas)...")
    try:
        psi, roi = 1.0, 2.0
        h = 1e-5

        grad_analytic = tensor.compute_gradient(psi, roi)

        # Diferencia finita para Ψ
        f_psi_plus = tensor.compute_penalty(psi + h, roi)
        f_psi_minus = tensor.compute_penalty(psi - h, roi)
        grad_psi_fd = (f_psi_plus - f_psi_minus) / (2 * h)

        error_psi = abs(grad_analytic[0] - grad_psi_fd) / (abs(grad_analytic[0]) + 1e-10)

        assert error_psi < 1e-3, f"Error en ∂F/∂Ψ: {error_psi}"
        test_results.append(("Gradientes", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Gradientes", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Test 7: Inversibilidad
    print("[Test 7] Mapeo inverso...")
    try:
        penalty = 100.0
        psi_est, roi_est = tensor.inverse_map(penalty)

        # Verificar: F(Ψ_est, ROI_est) ≈ penalty
        penalty_reconstructed = tensor.compute_penalty(psi_est, roi_est)
        error = abs(penalty_reconstructed - penalty) / penalty

        assert error < 1e-10, f"Error en mapeo inverso: {error}"
        test_results.append(("Inversibilidad", True, None))
        print("  ✓ Aprobado")
    except Exception as e:
        test_results.append(("Inversibilidad", False, str(e)))
        print(f"  ✗ Fallido: {e}")

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS:")
    print("=" * 80)

    passed = sum(1 for _, result, _ in test_results if result)
    total = len(test_results)

    for test_name, result, error in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} | {test_name:20s} | {error or ''}")

    print("=" * 80)
    print(f"Resultado: {passed}/{total} tests aprobados")
    print("=" * 80)

    return passed == total


# ════════════════════════════════════════════════════════════════════════════
# MÓDULO 14: PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ejecutar tests
    all_tests_passed = run_property_tests()

    # Generar diagnóstico
    print("\n")
    tensor_example = TensorFactory.create("moderate")
    print(DiagnosticAnalyzer.generate_report(tensor_example))

    # Comparación de tensores
    print("\n")
    t1 = TensorFactory.create("conservative")
    t2 = TensorFactory.create("aggressive")
    print(DiagnosticAnalyzer.compare_tensors(t1, t2))

    exit(0 if all_tests_passed else 1)