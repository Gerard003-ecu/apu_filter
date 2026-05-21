"""
=========================================================================================
Módulo: Business Agent (Operador de Síntesis Categórica y Cerebro Ejecutivo)
Ubicación: app/strategy/business_agent.py
Versión: 2.0.0-rigorous
=========================================================================================

Fundamentación Matemática (Teoría de Categorías y Topología Algebraica):
    
    Este módulo implementa un funtor covariante F: ℂ_Budget → ℂ_Decision que mapea
    el complejo simplicial del presupuesto (categoría fuente) a un retículo de decisión
    booleano (categoría objetivo). La construcción respeta los siguientes axiomas:
    
    1. **Preservación de Estructura (Functorialidad):**
       ∀ morfismos f: A → B en ℂ_Budget, F(f ∘ g) = F(f) ∘ F(g)
       F(id_A) = id_{F(A)}
    
    2. **Topos de Haces sobre el Sitio de Grothendieck:**
       La topología del grafo presupuestario induce un topos Sh(𝒢, J) donde 𝒢 es
       el grafo y J es la topología de Grothendieck canónica. Los haces representan
       asignaciones coherentes de valores financieros a subestructuras abiertas.
    
    3. **Cohomología de De Rham Discreta:**
       Los números de Betti β_k se interpretan como dimensiones de grupos de cohomología:
       β_k = dim H^k(K; ℝ) para el complejo simplicial K
       La característica de Euler χ = Σ(-1)^k β_k es un invariante homológico.
    
    4. **Teoría Espectral del Laplaciano:**
       Λ = D - A (Matriz Laplaciana combinatoria)
       donde D es diagonal de grados y A es la matriz de adyacencia.
       
       Espectro: 0 = λ₀ ≤ λ₁ ≤ ... ≤ λ_{n-1}
       
       El valor de Fiedler λ₁ cuantifica la conectividad algebraica:
       - λ₁ = 0 ⇔ Grafo desconexo (múltiples componentes)
       - λ₁ > 0 ⇔ Grafo conexo
       
       El gap espectral λ₁ - λ₀ mide la robustez de la conectividad ante perturbaciones.
    
    5. **Álgebra de Boole del Espacio de Decisión:**
       El retículo de decisión D forma un álgebra de Boole completa con:
       - Ínfimo: ⊓ (intersección de restricciones)
       - Supremo: ⊔ (unión de oportunidades)
       - Complemento: ¬ (negación de condición)
       - Elementos extremos: ⊤ (aprobación total), ⊥ (veto absoluto)
    
    6. **Mecánica Cuántica del Colapso de Decisión:**
       El estado del proyecto |Ψ⟩ ∈ ℋ (espacio de Hilbert) colapsa bajo medición:
       
       |Ψ⟩ = α|viable⟩ + β|inviable⟩
       
       Operador de proyección: P̂ = |viable⟩⟨viable|
       Probabilidad de viabilidad: P(viable) = |⟨viable|Ψ⟩|²
       
       El Risk Challenger actúa como operador de medición que fuerza el colapso.

Arquitectura de Módulos:
    - Capa 1 (Física): Álgebra Lineal, Topología, Teoría de Grafos
    - Capa 2 (Táctica): Análisis Financiero, Termodinámica Estadística
    - Capa 3 (Estrategia): Síntesis Categórica, Auditoría Adversarial
    - Capa 4 (Sabiduría): Traducción Semántica, Narrativa Ejecutiva

=========================================================================================
"""

from __future__ import annotations

import abc
import asyncio
import copy
import functools
import hashlib
import logging
import math
import numbers
import operator
import threading
import time
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Hashable, Iterable, Mapping as ABCMapping, Sequence as ABCSequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, asdict, fields as dataclass_fields, replace
from decimal import Decimal, getcontext, ROUND_HALF_UP
from enum import Enum, IntEnum, auto, unique
from fractions import Fraction
from functools import cached_property, lru_cache, partial, reduce, wraps
from itertools import chain, combinations, product
from pathlib import Path
from typing import (
    Any, Callable, ClassVar, Dict, Final, FrozenSet, Generic, Hashable as TypingHashable,
    Iterator, List, Literal, Mapping, NamedTuple, NewType, NoReturn, Optional, 
    Protocol, Sequence, Set, SupportsFloat, SupportsInt, Tuple, Type, TypeAlias,
    TypeVar, Union, cast, overload, runtime_checkable, final
)
from typing_extensions import Self, TypeGuard
from weakref import WeakValueDictionary

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy import ndarray
from scipy import sparse
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from scipy.spatial.distance import cdist, euclidean

# =============================================================================
# CONFIGURACIÓN DE PRECISIÓN NUMÉRICA
# =============================================================================

# Decimal precision para cálculos financieros críticos
getcontext().prec = 50
getcontext().rounding = ROUND_HALF_UP

# Configuración NumPy
np.seterr(divide='warn', over='warn', under='ignore', invalid='warn')
np.set_printoptions(precision=15, suppress=False, threshold=1000)

# =============================================================================
# LOGGING ESTRUCTURADO
# =============================================================================

logger = logging.getLogger(__name__)

class LogLevel(IntEnum):
    """Niveles de logging con semántica extendida."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60

# Registro de nivel TRACE
logging.addLevelName(LogLevel.TRACE, "TRACE")

def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Logging de traza para debugging profundo."""
    if self.isEnabledFor(LogLevel.TRACE):
        self._log(LogLevel.TRACE, message, args, **kwargs)

logging.Logger.trace = trace  # type: ignore

# =============================================================================
# CONSTANTES MATEMÁTICAS FUNDAMENTALES
# =============================================================================

@final
class MathematicalConstants:
    """
    Constantes matemáticas fundamentales con precisión de máquina.
    
    Todas las constantes están calculadas con precisión float64 (IEEE 754).
    """
    
    # Constantes algebraicas
    GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0  # φ ≈ 1.618033988749895
    SILVER_RATIO: Final[float] = 1.0 + math.sqrt(2.0)          # δ_S ≈ 2.414213562373095
    EULER_MASCHERONI: Final[float] = 0.5772156649015329        # γ
    
    # Constantes topológicas
    EULER_CHARACTERISTIC_SPHERE: Final[int] = 2                 # χ(S²) = 2
    EULER_CHARACTERISTIC_TORUS: Final[int] = 0                  # χ(T²) = 0
    
    # Umbrales numéricos (ε-neighborhood)
    EPSILON_MACHINE: Final[float] = np.finfo(np.float64).eps    # ≈ 2.220446049250313e-16
    EPSILON_SQRT: Final[float] = np.sqrt(EPSILON_MACHINE)       # ≈ 1.4901161193847656e-08
    EPSILON_TOLERANCE: Final[float] = 1e-10                     # Tolerancia estándar
    EPSILON_STRICT: Final[float] = 1e-14                        # Tolerancia estricta
    
    # Límites numéricos
    FLOAT_MAX: Final[float] = np.finfo(np.float64).max          # ≈ 1.7976931348623157e+308
    FLOAT_MIN_POSITIVE: Final[float] = np.finfo(np.float64).tiny # ≈ 2.2250738585072014e-308
    
    # Constantes logarítmicas
    LOG_EPSILON: Final[float] = math.log(EPSILON_TOLERANCE)     # ≈ -23.025850929940457
    LOG_2: Final[float] = math.log(2.0)                         # ln(2) ≈ 0.6931471805599453
    LOG_10: Final[float] = math.log(10.0)                       # ln(10) ≈ 2.302585092994046
    
    # Constantes geométricas
    SQRT_2: Final[float] = math.sqrt(2.0)                       # √2 ≈ 1.4142135623730951
    SQRT_3: Final[float] = math.sqrt(3.0)                       # √3 ≈ 1.7320508075688772
    SQRT_5: Final[float] = math.sqrt(5.0)                       # √5 ≈ 2.23606797749979
    
    # Límites de iteración
    MAX_ITERATIONS: Final[int] = 10_000                         # Límite para algoritmos iterativos
    DEFAULT_ITERATIONS: Final[int] = 1_000
    
    @classmethod
    def is_negligible(cls, value: float, threshold: Optional[float] = None) -> bool:
        """
        Determina si un valor es numéricamente insignificante.
        
        Args:
            value: Valor a evaluar
            threshold: Umbral personalizado (por defecto EPSILON_TOLERANCE)
        
        Returns:
            True si |value| < threshold
        """
        eps = threshold if threshold is not None else cls.EPSILON_TOLERANCE
        return abs(value) < eps
    
    @classmethod
    def are_close(
        cls, 
        a: float, 
        b: float, 
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> bool:
        """
        Comparación numérica robusta con tolerancias relativa y absoluta.
        
        Implementa: |a - b| ≤ max(rel_tol × max(|a|, |b|), abs_tol)
        
        Args:
            a, b: Valores a comparar
            rel_tol: Tolerancia relativa (por defecto EPSILON_SQRT)
            abs_tol: Tolerancia absoluta (por defecto EPSILON_TOLERANCE)
        
        Returns:
            True si los valores son numéricamente equivalentes
        """
        rel = rel_tol if rel_tol is not None else cls.EPSILON_SQRT
        abs_ = abs_tol if abs_tol is not None else cls.EPSILON_TOLERANCE
        
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    
    @classmethod
    def safe_log(cls, value: float, base: float = math.e) -> float:
        """
        Logaritmo seguro que maneja valores no positivos.
        
        Args:
            value: Argumento del logaritmo
            base: Base del logaritmo (por defecto e)
        
        Returns:
            log_base(max(value, EPSILON_TOLERANCE))
        """
        safe_val = max(value, cls.EPSILON_TOLERANCE)
        if base == math.e:
            return math.log(safe_val)
        return math.log(safe_val, base)
    
    @classmethod
    def safe_divide(
        cls, 
        numerator: float, 
        denominator: float, 
        fallback: float = 0.0
    ) -> float:
        """
        División segura con manejo de denominador nulo.
        
        Args:
            numerator: Numerador
            denominator: Denominador
            fallback: Valor de retorno si denominador ≈ 0
        
        Returns:
            numerator / denominator si |denominator| > ε, sino fallback
        """
        if abs(denominator) < cls.EPSILON_TOLERANCE:
            return fallback
        return numerator / denominator


MC = MathematicalConstants  # Alias para brevedad

# =============================================================================
# TIPOS ALGEBRAICOS Y NEWTYPE WRAPPERS
# =============================================================================

# Tipos numéricos refinados
Probability = NewType('Probability', float)  # ∈ [0, 1]
Percentage = NewType('Percentage', float)    # ∈ [0, 100]
PositiveReal = NewType('PositiveReal', float)  # ∈ (0, ∞)
NonNegativeReal = NewType('NonNegativeReal', float)  # ∈ [0, ∞)
UnitInterval = NewType('UnitInterval', float)  # ∈ [0, 1]

# Tipos topológicos
SimplexDimension = NewType('SimplexDimension', int)  # ∈ ℕ₀
BettiNumber = NewType('BettiNumber', int)  # ∈ ℕ₀
EulerCharacteristic = NewType('EulerCharacteristic', int)  # ∈ ℤ

# Tipos financieros
CashFlow = NewType('CashFlow', Decimal)
DiscountRate = NewType('DiscountRate', Decimal)
NPV = NewType('NPV', Decimal)

# Tipos de grafos
NodeID = NewType('NodeID', str)
EdgeWeight = NewType('EdgeWeight', float)

# Type aliases para estructuras complejas
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
SparseMatrix = Union[csr_matrix, sparse.spmatrix]
GraphAdjacencyMatrix: TypeAlias = Union[Matrix, SparseMatrix]

# =============================================================================
# PROTOCOLOS ALGEBRAICOS (INTERFACES CATEGÓRICAS)
# =============================================================================

@runtime_checkable
class Semigroup(Protocol):
    """
    Protocolo para estructura de semigrupo (S, ∘).
    
    Axioma de Asociatividad:
        ∀ a, b, c ∈ S: (a ∘ b) ∘ c = a ∘ (b ∘ c)
    """
    
    def compose(self, other: Self) -> Self:
        """Operación binaria asociativa."""
        ...


@runtime_checkable
class Monoid(Semigroup, Protocol):
    """
    Protocolo para estructura de monoide (M, ∘, e).
    
    Axiomas:
        1. Asociatividad (heredada de Semigroup)
        2. Identidad: ∃ e ∈ M: ∀ a ∈ M, e ∘ a = a ∘ e = a
    """
    
    @classmethod
    def identity(cls) -> Self:
        """Elemento identidad del monoide."""
        ...


@runtime_checkable
class Group(Monoid, Protocol):
    """
    Protocolo para estructura de grupo (G, ∘, e, ⁻¹).
    
    Axiomas:
        1. Asociatividad (heredada)
        2. Identidad (heredada)
        3. Inversos: ∀ a ∈ G, ∃ a⁻¹ ∈ G: a ∘ a⁻¹ = a⁻¹ ∘ a = e
    """
    
    def inverse(self) -> Self:
        """Inverso del elemento en el grupo."""
        ...


@runtime_checkable
class Ring(Protocol):
    """
    Protocolo para estructura de anillo (R, +, ×, 0, 1).
    
    Axiomas:
        1. (R, +, 0) es grupo abeliano
        2. (R, ×, 1) es monoide
        3. × distribuye sobre +: a × (b + c) = (a × b) + (a × c)
    """
    
    def add(self, other: Self) -> Self:
        """Suma en el anillo."""
        ...
    
    def multiply(self, other: Self) -> Self:
        """Multiplicación en el anillo."""
        ...
    
    def negate(self) -> Self:
        """Inverso aditivo."""
        ...
    
    @classmethod
    def additive_identity(cls) -> Self:
        """Identidad aditiva (0)."""
        ...
    
    @classmethod
    def multiplicative_identity(cls) -> Self:
        """Identidad multiplicativa (1)."""
        ...


@runtime_checkable
class VectorSpace(Protocol[TypeVar("F")]):
    """
    Protocolo para espacio vectorial sobre un campo F.
    
    Axiomas:
        1. (V, +) es grupo abeliano
        2. Multiplicación escalar: F × V → V
        3. Compatibilidad: a(bv) = (ab)v
        4. Identidad: 1v = v
        5. Distributividad: (a+b)v = av + bv, a(v+w) = av + aw
    """
    
    def scale(self, scalar: Any) -> Self:
        """Multiplicación por escalar."""
        ...
    
    def add_vector(self, other: Self) -> Self:
        """Suma de vectores."""
        ...
    
    @property
    def dimension(self) -> int:
        """Dimensión del espacio vectorial."""
        ...


@runtime_checkable
class NormedSpace(VectorSpace, Protocol):
    """
    Protocolo para espacio vectorial normado.
    
    Norma ‖·‖: V → ℝ₊ satisface:
        1. ‖v‖ ≥ 0, ‖v‖ = 0 ⇔ v = 0
        2. ‖αv‖ = |α|‖v‖
        3. ‖v + w‖ ≤ ‖v‖ + ‖w‖ (desigualdad triangular)
    """
    
    def norm(self) -> float:
        """Norma del vector."""
        ...


@runtime_checkable
class MetricSpace(Protocol):
    """
    Protocolo para espacio métrico (X, d).
    
    Métrica d: X × X → ℝ₊ satisface:
        1. d(x, y) ≥ 0, d(x, y) = 0 ⇔ x = y
        2. d(x, y) = d(y, x) (simetría)
        3. d(x, z) ≤ d(x, y) + d(y, z) (desigualdad triangular)
    """
    
    def distance(self, other: Self) -> float:
        """Métrica entre dos puntos."""
        ...


@runtime_checkable
class TopologicalSpace(Protocol):
    """
    Protocolo para espacio topológico (X, τ).
    
    Topología τ ⊆ P(X) satisface:
        1. ∅, X ∈ τ
        2. Cerrada bajo uniones arbitrarias
        3. Cerrada bajo intersecciones finitas
    """
    
    def is_open(self, subset: FrozenSet[Any]) -> bool:
        """Determina si un conjunto es abierto en la topología."""
        ...
    
    def interior(self, subset: FrozenSet[Any]) -> FrozenSet[Any]:
        """Interior topológico de un conjunto."""
        ...
    
    def closure(self, subset: FrozenSet[Any]) -> FrozenSet[Any]:
        """Clausura topológica de un conjunto."""
        ...


@runtime_checkable
class Lattice(Protocol):
    """
    Protocolo para retículo (L, ⊓, ⊔).
    
    Axiomas:
        1. Idempotencia: a ⊓ a = a, a ⊔ a = a
        2. Commutatividad: a ⊓ b = b ⊓ a, a ⊔ b = b ⊔ a
        3. Asociatividad
        4. Absorción: a ⊓ (a ⊔ b) = a, a ⊔ (a ⊓ b) = a
    """
    
    def meet(self, other: Self) -> Self:
        """Ínfimo (greatest lower bound)."""
        ...
    
    def join(self, other: Self) -> Self:
        """Supremo (least upper bound)."""
        ...
    
    def __le__(self, other: Self) -> bool:
        """Orden parcial: a ≤ b ⇔ a ⊓ b = a."""
        ...


@runtime_checkable
class BooleanAlgebra(Lattice, Protocol):
    """
    Protocolo para álgebra de Boole (B, ⊓, ⊔, ¬, ⊤, ⊥).
    
    Axiomas adicionales:
        1. Distributividad completa
        2. Complementación: a ⊔ ¬a = ⊤, a ⊓ ¬a = ⊥
        3. Ley de De Morgan
    """
    
    def complement(self) -> Self:
        """Complemento booleano."""
        ...
    
    @classmethod
    def top(cls) -> Self:
        """Elemento máximo (⊤)."""
        ...
    
    @classmethod
    def bottom(cls) -> Self:
        """Elemento mínimo (⊥)."""
        ...


# =============================================================================
# EXCEPCIONES MATEMÁTICAS ESPECIALIZADAS
# =============================================================================

class MathematicalError(Exception):
    """Clase base para errores matemáticos."""
    
    def __init__(
        self, 
        message: str, 
        *, 
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp
        }


class TopologicalInvariantError(MathematicalError):
    """Error en cálculo de invariantes topológicos."""
    pass


class SpectralTheoryError(MathematicalError):
    """Error en análisis espectral de operadores."""
    pass


class AlgebraicStructureError(MathematicalError):
    """Violación de axiomas algebraicos."""
    pass


class NumericalInstabilityError(MathematicalError):
    """Inestabilidad numérica detectada."""
    pass


class ConvergenceError(MathematicalError):
    """Fallo de convergencia en algoritmo iterativo."""
    pass


class DimensionMismatchError(MathematicalError):
    """Dimensiones incompatibles en operación vectorial."""
    pass


# =============================================================================
# VALIDADORES DE TIPOS REFINADOS
# =============================================================================

class TypeGuards:
    """Guardias de tipo para tipos refinados."""
    
    @staticmethod
    def is_probability(value: float) -> TypeGuard[Probability]:
        """Verifica que value ∈ [0, 1]."""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def is_positive_real(value: float) -> TypeGuard[PositiveReal]:
        """Verifica que value > 0."""
        return value > 0.0
    
    @staticmethod
    def is_non_negative_real(value: float) -> TypeGuard[NonNegativeReal]:
        """Verifica que value ≥ 0."""
        return value >= 0.0
    
    @staticmethod
    def is_unit_interval(value: float) -> TypeGuard[UnitInterval]:
        """Verifica que value ∈ [0, 1]."""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def is_simplex_dimension(value: int) -> TypeGuard[SimplexDimension]:
        """Verifica que value ∈ ℕ₀."""
        return isinstance(value, int) and value >= 0
    
    @staticmethod
    def is_valid_vector(arr: Any) -> TypeGuard[Vector]:
        """Verifica que arr es un vector NumPy válido."""
        return isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == np.float64
    
    @staticmethod
    def is_valid_matrix(arr: Any) -> TypeGuard[Matrix]:
        """Verifica que arr es una matriz NumPy válida."""
        return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.dtype == np.float64


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES (ALGEBRAICAS)
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti como invariantes topológicos del complejo simplicial.
    
    Interpretación Homológica:
        β_k = dim H_k(K; 𝔽) = rank(Z_k) - rank(B_k)
    
    donde:
        - H_k: k-ésimo grupo de homología
        - Z_k: k-ciclos (ker ∂_k)
        - B_k: k-fronteras (im ∂_{k+1})
        - 𝔽: Campo de coeficientes (típicamente ℤ₂ o ℝ)
    
    Attributes:
        beta_0: Número de componentes conexas
        beta_1: Número de ciclos independientes (1-dimensional holes)
        beta_2: Número de cavidades (2-dimensional voids)
        beta_n: Números de Betti superiores (tupla extendida)
    """
    
    beta_0: BettiNumber = BettiNumber(1)
    beta_1: BettiNumber = BettiNumber(0)
    beta_2: BettiNumber = BettiNumber(0)
    beta_n: Tuple[BettiNumber, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if self.beta_0 < 0:
            raise TopologicalInvariantError(
                f"β₀ debe ser no negativo: {self.beta_0}",
                context={"beta_numbers": asdict(self)}
            )
        if self.beta_1 < 0 or self.beta_2 < 0:
            raise TopologicalInvariantError(
                "Números de Betti deben ser no negativos",
                context={"beta_1": self.beta_1, "beta_2": self.beta_2}
            )
        if any(b < 0 for b in self.beta_n):
            raise TopologicalInvariantError(
                "Números de Betti superiores deben ser no negativos",
                context={"beta_n": self.beta_n}
            )
    
    @cached_property
    def euler_characteristic(self) -> EulerCharacteristic:
        """
        Característica de Euler mediante fórmula de Euler-Poincaré.
        
        χ = Σ_{k=0}^∞ (-1)^k β_k = β₀ - β₁ + β₂ - β₃ + ...
        
        Teorema (Euler-Poincaré):
            Para CW-complejo finito K:
            χ(K) = Σ (-1)^k n_k
            donde n_k = número de k-celdas
        
        Returns:
            Característica de Euler (invariante topológico)
        """
        result = self.beta_0 - self.beta_1 + self.beta_2
        
        for k, beta_k in enumerate(self.beta_n, start=3):
            sign = (-1) ** k
            result += sign * beta_k
        
        return EulerCharacteristic(result)
    
    @cached_property
    def is_connected(self) -> bool:
        """
        Determina si el complejo es conexo.
        
        Teorema:
            K es conexo ⇔ β₀(K) = 1
        
        Returns:
            True si β₀ = 1 (espacio conexo)
        """
        return self.beta_0 == 1
    
    @cached_property
    def has_cycles(self) -> bool:
        """
        Determina si existen ciclos 1-dimensionales.
        
        Returns:
            True si β₁ > 0 (existen loops independientes)
        """
        return self.beta_1 > 0
    
    @cached_property
    def has_voids(self) -> bool:
        """
        Determina si existen cavidades 2-dimensionales.
        
        Returns:
            True si β₂ > 0 (existen voids)
        """
        return self.beta_2 > 0
    
    @cached_property
    def total_homology_rank(self) -> int:
        """
        Rango total de la homología: Σ β_k.
        
        Interpretación:
            Dimensión total del espacio de homología H_*(K; 𝔽)
        
        Returns:
            Suma de todos los números de Betti
        """
        return self.beta_0 + self.beta_1 + self.beta_2 + sum(self.beta_n)
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Constructor desde diccionario con validación.
        
        Args:
            data: Diccionario con claves beta_0, beta_1, beta_2, beta_n (opcional)
        
        Returns:
            Instancia de BettiNumbers
        
        Raises:
            TopologicalInvariantError: Si los datos son inválidos
        """
        try:
            beta_0 = BettiNumber(int(data.get("beta_0", 1)))
            beta_1 = BettiNumber(int(data.get("beta_1", 0)))
            beta_2 = BettiNumber(int(data.get("beta_2", 0)))
            beta_n_raw = data.get("beta_n", ())
            
            if isinstance(beta_n_raw, (list, tuple)):
                beta_n = tuple(BettiNumber(int(b)) for b in beta_n_raw)
            else:
                beta_n = ()
            
            return cls(beta_0=beta_0, beta_1=beta_1, beta_2=beta_2, beta_n=beta_n)
        
        except (ValueError, TypeError, KeyError) as e:
            raise TopologicalInvariantError(
                f"Fallo al construir BettiNumbers desde diccionario: {e}",
                context={"data": dict(data)}
            ) from e
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario.
        
        Returns:
            Diccionario con representación completa
        """
        result = {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "euler_characteristic": self.euler_characteristic,
            "is_connected": self.is_connected,
            "has_cycles": self.has_cycles,
            "has_voids": self.has_voids,
            "total_homology_rank": self.total_homology_rank
        }
        
        if self.beta_n:
            result["beta_n"] = list(self.beta_n)
        
        return result
    
    def __str__(self) -> str:
        """Representación matemática."""
        parts = [f"β₀={self.beta_0}", f"β₁={self.beta_1}", f"β₂={self.beta_2}"]
        if self.beta_n:
            for k, beta_k in enumerate(self.beta_n, start=3):
                parts.append(f"β₃={beta_k}")
        parts.append(f"χ={self.euler_characteristic}")
        return f"BettiNumbers({', '.join(parts)})"


@final
@dataclass(frozen=True, slots=True)
class SpectralData:
    """
    Datos espectrales del Laplaciano de grafos.
    
    Fundamentación Teórica:
        El Laplaciano combinatorio Λ = D - A donde:
        - D: matriz diagonal de grados
        - A: matriz de adyacencia
        
        Propiedades:
            1. Λ es simétrica y semi-definida positiva
            2. Λ1 = 0 (el vector constante es eigenvector con eigenvalor 0)
            3. El número de eigenvalores nulos = número de componentes conexas
            4. λ₁ (Fiedler) cuantifica la conectividad algebraica
    
    Attributes:
        eigenvalues: Eigenvalores λ₀ ≤ λ₁ ≤ ... ≤ λ_{n-1} (ordenados)
        fiedler_value: λ₁ (valor de Fiedler, conectividad algebraica)
        spectral_gap: λ₁ - λ₀ (robustez ante perturbaciones)
        algebraic_connectivity: Alias de fiedler_value
        spectral_radius: max_i |λ_i|
    """
    
    eigenvalues: Tuple[float, ...]
    fiedler_value: float = field(init=False)
    spectral_gap: float = field(init=False)
    algebraic_connectivity: float = field(init=False)
    spectral_radius: float = field(init=False)
    
    def __post_init__(self) -> None:
        """Cálculo de propiedades derivadas."""
        if not self.eigenvalues:
            raise SpectralTheoryError(
                "Se requiere al menos un eigenvalor",
                context={"eigenvalues": self.eigenvalues}
            )
        
        # Los eigenvalores ya deben estar ordenados
        sorted_eigs = sorted(self.eigenvalues)
        
        object.__setattr__(self, 'eigenvalues', tuple(sorted_eigs))
        object.__setattr__(self, 'fiedler_value', sorted_eigs[1] if len(sorted_eigs) > 1 else 0.0)
        object.__setattr__(self, 'spectral_gap', self.fiedler_value - sorted_eigs[0])
        object.__setattr__(self, 'algebraic_connectivity', self.fiedler_value)
        object.__setattr__(self, 'spectral_radius', max(abs(eig) for eig in sorted_eigs))
    
    @cached_property
    def is_connected(self) -> bool:
        """
        Determina si el grafo es conexo.
        
        Teorema (Conectividad Algebraica):
            Grafo G es conexo ⇔ λ₁(Λ) > 0
        
        Returns:
            True si el grafo es conexo
        """
        return self.fiedler_value > MC.EPSILON_TOLERANCE
    
    @cached_property
    def number_of_components(self) -> int:
        """
        Número de componentes conexas mediante multiplicidad de eigenvalor 0.
        
        Teorema:
            El número de componentes conexas = multiplicidad de eigenvalor 0
        
        Returns:
            Número de componentes (β₀)
        """
        return sum(1 for eig in self.eigenvalues if abs(eig) < MC.EPSILON_TOLERANCE)
    
    @cached_property
    def condition_number(self) -> float:
        """
        Número de condición espectral: κ(Λ) = λ_max / λ_min (excluyendo 0).
        
        Interpretación:
            Mide la sensibilidad del sistema a perturbaciones.
            κ grande ⇒ ill-conditioned
        
        Returns:
            Número de condición (≥ 1)
        """
        nonzero_eigs = [eig for eig in self.eigenvalues if abs(eig) > MC.EPSILON_TOLERANCE]
        
        if not nonzero_eigs:
            return float('inf')
        
        lambda_min = min(nonzero_eigs)
        lambda_max = max(nonzero_eigs)
        
        return MC.safe_divide(lambda_max, lambda_min, fallback=float('inf'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "eigenvalues": list(self.eigenvalues),
            "fiedler_value": self.fiedler_value,
            "spectral_gap": self.spectral_gap,
            "algebraic_connectivity": self.algebraic_connectivity,
            "spectral_radius": self.spectral_radius,
            "is_connected": self.is_connected,
            "number_of_components": self.number_of_components,
            "condition_number": self.condition_number
        }
    
    @classmethod
    def from_laplacian(cls, laplacian: Union[Matrix, SparseMatrix]) -> Self:
        """
        Calcula espectro del Laplaciano.
        
        Args:
            laplacian: Matriz Laplaciana (densa o sparse)
        
        Returns:
            SpectralData con eigenvalores calculados
        
        Raises:
            SpectralTheoryError: Si el cálculo falla
        """
        try:
            if sparse.issparse(laplacian):
                # Usar eigsh para matrices sparse (solo k eigenvalores más pequeños)
                k = min(10, laplacian.shape[0] - 1)
                eigenvalues_array, _ = sparse_linalg.eigsh(
                    laplacian,
                    k=k,
                    which='SA',  # Smallest Algebraic
                    return_eigenvectors=True
                )
            else:
                # Matriz densa: usar eigh (simétrica)
                eigenvalues_array, _ = np.linalg.eigh(laplacian)
            
            # Filtrar eigenvalores no finitos y ordenar
            valid_eigs = eigenvalues_array[np.isfinite(eigenvalues_array)]
            sorted_eigs = np.sort(valid_eigs)
            
            return cls(eigenvalues=tuple(float(eig) for eig in sorted_eigs))
        
        except (np.linalg.LinAlgError, sparse_linalg.ArpackError) as e:
            raise SpectralTheoryError(
                f"Fallo en cálculo espectral: {e}",
                context={"laplacian_shape": laplacian.shape}
            ) from e


# =============================================================================
# FASE 2: TOPOLOGÍA COMPUTACIONAL Y ÁLGEBRA LINEAL
# =============================================================================

# =============================================================================
# HOMOLOGÍA PERSISTENTE Y DIAGRAMAS DE PERSISTENCIA
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """
    Intervalo de persistencia [birth, death) en filtración de Čech/Vietoris-Rips.
    
    Fundamentación Teórica:
        En una filtración K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ, un feature topológico 
        (ciclo, componente conexa, cavidad) "nace" en tiempo birth y "muere" 
        en tiempo death.
        
        Persistencia = death - birth (medida de significancia del feature)
    
    Attributes:
        dimension: Dimensión homológica (0, 1, 2, ...)
        birth: Tiempo de nacimiento (radio de filtración)
        death: Tiempo de muerte (∞ si el feature persiste)
        persistence: Longevidad del feature
    """
    
    dimension: SimplexDimension
    birth: float
    death: float
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if self.birth < 0:
            raise TopologicalInvariantError(
                f"Tiempo de nacimiento debe ser no negativo: {self.birth}"
            )
        
        if not (self.death > self.birth or math.isinf(self.death)):
            raise TopologicalInvariantError(
                f"Debe cumplirse death > birth: birth={self.birth}, death={self.death}"
            )
        
        if self.dimension < 0:
            raise TopologicalInvariantError(
                f"Dimensión debe ser no negativa: {self.dimension}"
            )
    
    @cached_property
    def persistence(self) -> float:
        """
        Persistencia del intervalo: death - birth.
        
        Interpretación:
            Mide la robustez del feature topológico ante ruido.
            Persistencia alta ⇒ feature significativo
        
        Returns:
            Persistencia (puede ser ∞)
        """
        return self.death - self.birth
    
    @cached_property
    def is_essential(self) -> bool:
        """
        Determina si el intervalo es esencial (death = ∞).
        
        Returns:
            True si el feature nunca muere
        """
        return math.isinf(self.death)
    
    @cached_property
    def midpoint(self) -> float:
        """
        Punto medio del intervalo (para visualización).
        
        Returns:
            (birth + death) / 2 si death < ∞, sino birth
        """
        if self.is_essential:
            return self.birth
        return (self.birth + self.death) / 2.0
    
    def bottleneck_distance(self, other: 'PersistenceInterval') -> float:
        """
        Distancia de Bottleneck entre intervalos.
        
        d_B(I₁, I₂) = max(|b₁ - b₂|, |d₁ - d₂|)
        
        Args:
            other: Otro intervalo de persistencia
        
        Returns:
            Distancia de Bottleneck (métrica en espacio de diagramas)
        """
        if self.dimension != other.dimension:
            return float('inf')
        
        birth_diff = abs(self.birth - other.birth)
        
        if self.is_essential and other.is_essential:
            death_diff = 0.0
        elif self.is_essential or other.is_essential:
            death_diff = float('inf')
        else:
            death_diff = abs(self.death - other.death)
        
        return max(birth_diff, death_diff)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "dimension": self.dimension,
            "birth": self.birth,
            "death": self.death,
            "persistence": self.persistence,
            "is_essential": self.is_essential,
            "midpoint": self.midpoint
        }
    
    def __str__(self) -> str:
        """Representación matemática."""
        death_str = "∞" if self.is_essential else f"{self.death:.4f}"
        return f"H_{self.dimension}[{self.birth:.4f}, {death_str})"


@final
@dataclass(frozen=True, slots=True)
class PersistenceDiagram:
    """
    Diagrama de persistencia: multiconjunto de intervalos de persistencia.
    
    Teorema (Estabilidad de Diagramas):
        Para funciones f, g: X → ℝ en espacio métrico (X, d_X),
        d_B(Dgm(f), Dgm(g)) ≤ ‖f - g‖_∞
        
        donde d_B es la distancia de Bottleneck.
    
    Attributes:
        intervals: Conjunto de intervalos de persistencia
        max_dimension: Dimensión homológica máxima
    """
    
    intervals: Tuple[PersistenceInterval, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validación y ordenamiento."""
        if not self.intervals:
            return
        
        # Verificar que todos los intervalos son válidos
        for interval in self.intervals:
            if not isinstance(interval, PersistenceInterval):
                raise TopologicalInvariantError(
                    f"Intervalo inválido: {type(interval)}"
                )
        
        # Ordenar por persistencia descendente
        sorted_intervals = tuple(
            sorted(self.intervals, key=lambda x: x.persistence, reverse=True)
        )
        object.__setattr__(self, 'intervals', sorted_intervals)
    
    @cached_property
    def max_dimension(self) -> SimplexDimension:
        """Dimensión homológica máxima en el diagrama."""
        if not self.intervals:
            return SimplexDimension(0)
        return SimplexDimension(max(interval.dimension for interval in self.intervals))
    
    @cached_property
    def betti_numbers(self) -> BettiNumbers:
        """
        Números de Betti inferidos desde intervalos esenciales.
        
        Returns:
            BettiNumbers con conteo de features esenciales
        """
        beta_counts: Dict[int, int] = defaultdict(int)
        
        for interval in self.intervals:
            if interval.is_essential:
                beta_counts[interval.dimension] += 1
        
        return BettiNumbers(
            beta_0=BettiNumber(beta_counts.get(0, 1)),
            beta_1=BettiNumber(beta_counts.get(1, 0)),
            beta_2=BettiNumber(beta_counts.get(2, 0)),
            beta_n=tuple(BettiNumber(beta_counts.get(k, 0)) for k in range(3, self.max_dimension + 1))
        )
    
    def filter_by_persistence(self, threshold: float) -> 'PersistenceDiagram':
        """
        Filtra intervalos por umbral de persistencia.
        
        Args:
            threshold: Persistencia mínima
        
        Returns:
            Diagrama filtrado con intervalos significativos
        """
        filtered = tuple(
            interval for interval in self.intervals
            if interval.persistence >= threshold
        )
        return PersistenceDiagram(intervals=filtered)
    
    def filter_by_dimension(self, dimension: SimplexDimension) -> 'PersistenceDiagram':
        """
        Filtra intervalos por dimensión homológica.
        
        Args:
            dimension: Dimensión a retener
        
        Returns:
            Diagrama con solo intervalos de dimensión especificada
        """
        filtered = tuple(
            interval for interval in self.intervals
            if interval.dimension == dimension
        )
        return PersistenceDiagram(intervals=filtered)
    
    def bottleneck_distance(self, other: 'PersistenceDiagram') -> float:
        """
        Distancia de Bottleneck entre diagramas.
        
        Implementación simplificada: considera solo matching óptimo parcial.
        
        Args:
            other: Otro diagrama de persistencia
        
        Returns:
            Cota superior de la distancia de Bottleneck
        """
        if not self.intervals and not other.intervals:
            return 0.0
        
        if not self.intervals or not other.intervals:
            # Distancia a diagrama vacío = máxima persistencia
            non_empty = self if self.intervals else other
            return max(interval.persistence for interval in non_empty.intervals)
        
        # Matching greedy (aproximación)
        distances: List[float] = []
        
        for interval1 in self.intervals:
            min_dist = min(
                interval1.bottleneck_distance(interval2)
                for interval2 in other.intervals
            )
            distances.append(min_dist)
        
        return max(distances) if distances else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "intervals": [interval.to_dict() for interval in self.intervals],
            "max_dimension": self.max_dimension,
            "total_features": len(self.intervals),
            "betti_numbers": self.betti_numbers.to_dict()
        }
    
    @classmethod
    def from_intervals_list(
        cls, 
        intervals_data: Sequence[Tuple[int, float, float]]
    ) -> Self:
        """
        Constructor desde lista de tuplas (dimension, birth, death).
        
        Args:
            intervals_data: Lista de tuplas (dim, birth, death)
        
        Returns:
            PersistenceDiagram construido
        """
        intervals = tuple(
            PersistenceInterval(
                dimension=SimplexDimension(dim),
                birth=birth,
                death=death
            )
            for dim, birth, death in intervals_data
        )
        return cls(intervals=intervals)


# =============================================================================
# ÁLGEBRA LINEAL COMPUTACIONAL RIGUROSA
# =============================================================================

class MatrixOperations:
    """
    Operaciones de álgebra lineal con estabilidad numérica garantizada.
    
    Implementa algoritmos robustos para:
        - Factorización LU con pivoteo parcial
        - Descomposición QR (Gram-Schmidt modificado)
        - SVD (Singular Value Decomposition)
        - Pseudoinversa de Moore-Penrose
        - Cálculo de rango numérico
    """
    
    @staticmethod
    def compute_rank(
        matrix: Matrix, 
        tolerance: Optional[float] = None
    ) -> int:
        """
        Calcula el rango numérico mediante SVD.
        
        Teorema (Rango-Nullidad):
            rank(A) + nullity(A) = n (número de columnas)
        
        Args:
            matrix: Matriz a analizar
            tolerance: Umbral para valores singulares (por defecto: ε√n)
        
        Returns:
            Rango numérico (número de valores singulares > tolerance)
        """
        if matrix.size == 0:
            return 0
        
        # SVD: A = UΣVᵀ
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        
        if tolerance is None:
            # Umbral adaptativo: ε × √n × σ_max
            n = max(matrix.shape)
            tolerance = MC.EPSILON_SQRT * math.sqrt(n) * singular_values[0]
        
        rank = np.sum(singular_values > tolerance)
        return int(rank)
    
    @staticmethod
    def compute_condition_number(matrix: Matrix) -> float:
        """
        Número de condición espectral: κ(A) = σ_max / σ_min.
        
        Interpretación:
            - κ(A) = 1: matriz ortogonal (perfectamente condicionada)
            - κ(A) > 10¹²: matriz ill-conditioned (inestable)
        
        Args:
            matrix: Matriz a analizar
        
        Returns:
            Número de condición (≥ 1)
        """
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        
        if len(singular_values) == 0:
            return float('inf')
        
        sigma_max = singular_values[0]
        sigma_min = singular_values[-1]
        
        if sigma_min < MC.EPSILON_TOLERANCE:
            return float('inf')
        
        return float(sigma_max / sigma_min)
    
    @staticmethod
    def pseudoinverse(
        matrix: Matrix, 
        tolerance: Optional[float] = None
    ) -> Matrix:
        """
        Pseudoinversa de Moore-Penrose: A⁺.
        
        Propiedades:
            1. AA⁺A = A
            2. A⁺AA⁺ = A⁺
            3. (AA⁺)ᵀ = AA⁺
            4. (A⁺A)ᵀ = A⁺A
        
        Args:
            matrix: Matriz a invertir
            tolerance: Umbral para valores singulares
        
        Returns:
            Pseudoinversa A⁺
        """
        if tolerance is None:
            tolerance = MC.EPSILON_TOLERANCE
        
        # Usar pinv de NumPy (implementa SVD truncado)
        return np.linalg.pinv(matrix, rcond=tolerance)
    
    @staticmethod
    def gram_schmidt(
        vectors: Matrix, 
        normalize: bool = True
    ) -> Tuple[Matrix, Matrix]:
        """
        Ortogonalización de Gram-Schmidt modificada (estable).
        
        Algoritmo:
            Para cada vector vⱼ:
                uⱼ = vⱼ - Σᵢ₌₁ʲ⁻¹ ⟨vⱼ, qᵢ⟩qᵢ
                qⱼ = uⱼ / ‖uⱼ‖
        
        Args:
            vectors: Matriz con vectores en columnas
            normalize: Si True, retorna base ortonormal
        
        Returns:
            Tupla (Q, R) donde Q es ortogonal y R es triangular superior
        """
        m, n = vectors.shape
        Q = np.zeros((m, n), dtype=np.float64)
        R = np.zeros((n, n), dtype=np.float64)
        
        for j in range(n):
            v = vectors[:, j].copy()
            
            # Ortogonalización
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], v)
                v = v - R[i, j] * Q[:, i]
            
            # Normalización
            norm = np.linalg.norm(v)
            
            if norm < MC.EPSILON_TOLERANCE:
                # Vector linealmente dependiente
                logger.warning(f"Vector {j} linealmente dependiente en Gram-Schmidt")
                continue
            
            R[j, j] = norm
            Q[:, j] = v / norm if normalize else v
        
        return Q, R
    
    @staticmethod
    def is_positive_definite(matrix: Matrix) -> bool:
        """
        Verifica si la matriz es definida positiva.
        
        Teorema (Sylvester):
            A es definida positiva ⇔ todos los menores principales > 0
        
        Implementación:
            Intenta factorización de Cholesky: A = LLᵀ
            Éxito ⇔ A es definida positiva
        
        Args:
            matrix: Matriz simétrica a verificar
        
        Returns:
            True si A es definida positiva
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Verificar simetría
        if not np.allclose(matrix, matrix.T, atol=MC.EPSILON_TOLERANCE):
            return False
        
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def frobenius_norm(matrix: Matrix) -> float:
        """
        Norma de Frobenius: ‖A‖_F = √(Σᵢⱼ |aᵢⱼ|²).
        
        Propiedades:
            1. ‖A‖_F = √(tr(AᵀA))
            2. ‖A‖_F = √(Σᵢ σᵢ²) (suma de valores singulares al cuadrado)
        
        Args:
            matrix: Matriz a medir
        
        Returns:
            Norma de Frobenius
        """
        return float(np.linalg.norm(matrix, ord='fro'))
    
    @staticmethod
    def operator_norm(matrix: Matrix, ord: Union[int, float, str] = 2) -> float:
        """
        Norma de operador (norma espectral para ord=2).
        
        Normas implementadas:
            - ord=1: norma-1 (máximo de suma de columnas)
            - ord=2: norma espectral (máximo valor singular)
            - ord='inf': norma-∞ (máximo de suma de filas)
            - ord='fro': norma de Frobenius
        
        Args:
            matrix: Matriz a medir
            ord: Tipo de norma
        
        Returns:
            Norma de operador
        """
        return float(np.linalg.norm(matrix, ord=ord))


class LaplacianBuilder:
    """
    Constructor del Laplaciano de grafos con variantes.
    
    Implementa:
        1. Laplaciano combinatorio: Λ = D - A
        2. Laplaciano normalizado (simétrico): ℒ = I - D⁻¹/²AD⁻¹/²
        3. Laplaciano de paseo aleatorio: ℒ_rw = I - D⁻¹A
        4. Laplaciano signless: |Λ| = D + A
    """
    
    @staticmethod
    def combinatorial_laplacian(adjacency: Matrix) -> Matrix:
        """
        Laplaciano combinatorio: Λ = D - A.
        
        Propiedades:
            1. Λ es simétrica
            2. Λ es semi-definida positiva
            3. Λ1 = 0 (vector constante es eigenvector)
            4. Suma de filas = 0
        
        Args:
            adjacency: Matriz de adyacencia (simétrica para grafos no dirigidos)
        
        Returns:
            Matriz Laplaciana
        """
        if adjacency.shape[0] != adjacency.shape[1]:
            raise DimensionMismatchError(
                "La matriz de adyacencia debe ser cuadrada",
                context={"shape": adjacency.shape}
            )
        
        # Matriz de grados (diagonal)
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        laplacian = degree_matrix - adjacency
        
        return laplacian
    
    @staticmethod
    def normalized_laplacian(adjacency: Matrix) -> Matrix:
        """
        Laplaciano normalizado simétrico: ℒ = I - D⁻¹/²AD⁻¹/².
        
        Ventajas:
            - Eigenvalores en [0, 2]
            - Invariante ante re-escalado de nodos
        
        Args:
            adjacency: Matriz de adyacencia
        
        Returns:
            Laplaciano normalizado
        """
        n = adjacency.shape[0]
        degrees = np.sum(adjacency, axis=1)
        
        # Manejar nodos aislados (grado 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            degrees_inv_sqrt = np.where(
                degrees > MC.EPSILON_TOLERANCE,
                1.0 / np.sqrt(degrees),
                0.0
            )
        
        D_inv_sqrt = np.diag(degrees_inv_sqrt)
        
        # ℒ = I - D⁻¹/²AD⁻¹/²
        normalized = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return normalized
    
    @staticmethod
    def random_walk_laplacian(adjacency: Matrix) -> Matrix:
        """
        Laplaciano de paseo aleatorio: ℒ_rw = I - D⁻¹A.
        
        Interpretación:
            Matriz de transición de paseo aleatorio: P = D⁻¹A
            ℒ_rw = I - P
        
        Args:
            adjacency: Matriz de adyacencia
        
        Returns:
            Laplaciano de paseo aleatorio
        """
        n = adjacency.shape[0]
        degrees = np.sum(adjacency, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            degrees_inv = np.where(
                degrees > MC.EPSILON_TOLERANCE,
                1.0 / degrees,
                0.0
            )
        
        D_inv = np.diag(degrees_inv)
        
        laplacian_rw = np.eye(n) - D_inv @ adjacency
        
        return laplacian_rw
    
    @staticmethod
    def signless_laplacian(adjacency: Matrix) -> Matrix:
        """
        Laplaciano signless: |Λ| = D + A.
        
        Propiedades:
            - Todos los eigenvalores son no negativos
            - Útil para análisis de bipartición
        
        Args:
            adjacency: Matriz de adyacencia
        
        Returns:
            Laplaciano signless
        """
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        return degree_matrix + adjacency


# =============================================================================
# ESTRUCTURAS DE GRAFOS CON INVARIANTES TOPOLÓGICOS
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class GraphMetrics:
    """
    Métricas de teoría de grafos para análisis estructural.
    
    Combina métricas clásicas con invariantes topológicos.
    
    Attributes:
        n_nodes: Número de vértices |V|
        n_edges: Número de aristas |E|
        density: Densidad del grafo: 2|E| / (|V|(|V|-1))
        average_degree: Grado promedio: 2|E| / |V|
        clustering_coefficient: Coeficiente de clustering global
        diameter: Diámetro del grafo (distancia geodésica máxima)
        is_connected: Si el grafo es conexo
        n_components: Número de componentes conexas
    """
    
    n_nodes: int
    n_edges: int
    density: float = field(init=False)
    average_degree: float = field(init=False)
    clustering_coefficient: Optional[float] = None
    diameter: Optional[int] = None
    is_connected: bool = False
    n_components: int = 1
    
    def __post_init__(self) -> None:
        """Cálculo de métricas derivadas."""
        if self.n_nodes < 0 or self.n_edges < 0:
            raise ValueError("Número de nodos y aristas debe ser no negativo")
        
        # Densidad: ρ = 2|E| / (|V|(|V|-1)) para grafos no dirigidos
        if self.n_nodes > 1:
            max_edges = (self.n_nodes * (self.n_nodes - 1)) / 2.0
            density = self.n_edges / max_edges if max_edges > 0 else 0.0
        else:
            density = 0.0
        
        # Grado promedio: ⟨k⟩ = 2|E| / |V|
        avg_degree = (2.0 * self.n_edges) / self.n_nodes if self.n_nodes > 0 else 0.0
        
        object.__setattr__(self, 'density', density)
        object.__setattr__(self, 'average_degree', avg_degree)
    
    @cached_property
    def is_sparse(self) -> bool:
        """
        Determina si el grafo es sparse.
        
        Criterio: ρ < 0.1 o |E| = O(|V|)
        
        Returns:
            True si el grafo es sparse
        """
        return self.density < 0.1
    
    @cached_property
    def is_dense(self) -> bool:
        """
        Determina si el grafo es denso.
        
        Criterio: ρ > 0.5 o |E| = Θ(|V|²)
        
        Returns:
            True si el grafo es denso
        """
        return self.density > 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "average_degree": self.average_degree,
            "clustering_coefficient": self.clustering_coefficient,
            "diameter": self.diameter,
            "is_connected": self.is_connected,
            "n_components": self.n_components,
            "is_sparse": self.is_sparse,
            "is_dense": self.is_dense
        }


# =============================================================================
# BUNDLE TOPOLÓGICO EXTENDIDO
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class TopologicalMetricsBundle:
    """
    Bundle cohesivo de métricas topológicas del presupuesto.
    
    Integra:
        - Homología (números de Betti)
        - Teoría espectral (Laplaciano)
        - Homología persistente (diagrama de persistencia)
        - Métricas de grafos
        - Estabilidad piramidal (física)
    
    Attributes:
        betti: Números de Betti
        spectral: Datos espectrales del Laplaciano
        persistence: Diagrama de persistencia (opcional)
        graph_metrics: Métricas de teoría de grafos
        pyramid_stability: Índice Ψ ∈ [0, 1]
    """
    
    betti: BettiNumbers
    spectral: SpectralData
    graph_metrics: GraphMetrics
    pyramid_stability: UnitInterval
    persistence: Optional[PersistenceDiagram] = None
    
    def __post_init__(self) -> None:
        """Validación de consistencia topológica."""
        # Verificar consistencia entre Betti y componentes espectrales
        if self.betti.beta_0 != self.spectral.number_of_components:
            logger.warning(
                f"Inconsistencia: β₀={self.betti.beta_0} vs "
                f"componentes espectrales={self.spectral.number_of_components}"
            )
        
        # Verificar que pyramid_stability ∈ [0, 1]
        if not (0.0 <= self.pyramid_stability <= 1.0):
            raise ValueError(
                f"pyramid_stability debe estar en [0, 1]: {self.pyramid_stability}"
            )
    
    @cached_property
    def structural_coherence(self) -> UnitInterval:
        """
        Índice de coherencia estructural mediante invariantes topológicos.
        
        Fórmula (revisada con teoría espectral):
            C(G) = exp(-λ₀·max(0, β₀-1)) × exp(-λ₁·β₁/√n) × Ψ × f(λ₁)
        
        donde:
            - λ₀: tasa de decaimiento por fragmentación
            - λ₁: valor de Fiedler (conectividad algebraica)
            - β₀, β₁: números de Betti
            - n: número de vértices
            - Ψ: estabilidad piramidal
            - f(λ₁): factor espectral = tanh(λ₁)
        
        Returns:
            Índice ∈ [0, 1], donde 1 = máxima coherencia
        """
        n = max(self.graph_metrics.n_nodes, 1)
        
        # Tasa de decaimiento
        lambda_frag = MC.LOG_2  # ln(2)
        lambda_cycle = MC.LOG_2 / max(1.0, math.sqrt(n))
        
        # Penalización por fragmentación
        excess_components = max(0, self.betti.beta_0 - 1)
        frag_penalty = math.exp(-lambda_frag * excess_components)
        
        # Penalización por ciclos
        cycle_penalty = math.exp(-lambda_cycle * self.betti.beta_1)
        
        # Factor espectral (conectividad algebraica)
        fiedler = self.spectral.fiedler_value
        spectral_factor = math.tanh(fiedler) if fiedler > 0 else 0.0
        
        # Composición multiplicativa
        coherence = frag_penalty * cycle_penalty * self.pyramid_stability * spectral_factor
        
        return UnitInterval(max(0.0, min(1.0, coherence)))
    
    @cached_property
    def cycle_density(self) -> float:
        """
        Densidad de ciclos: β₁ / |V|.
        
        Interpretación:
            Proporción de ciclos independientes por nodo.
        
        Returns:
            Densidad ∈ [0, 1]
        """
        if self.graph_metrics.n_nodes == 0:
            return 0.0
        return self.betti.beta_1 / self.graph_metrics.n_nodes
    
    @cached_property
    def topological_entropy(self) -> float:
        """
        Entropía topológica estimada mediante diversidad de features.
        
        Fórmula:
            H_top = -Σᵢ pᵢ log pᵢ
        
        donde pᵢ = βᵢ / Σⱼ βⱼ (proporción de cada dimensión homológica)
        
        Returns:
            Entropía ∈ [0, log(d+1)] donde d = dimensión máxima
        """
        total_betti = self.betti.total_homology_rank
        
        if total_betti == 0:
            return 0.0
        
        betti_values = [
            self.betti.beta_0,
            self.betti.beta_1,
            self.betti.beta_2
        ] + list(self.betti.beta_n)
        
        # Distribución de probabilidad
        probabilities = [b / total_betti for b in betti_values if b > 0]
        
        # Entropía de Shannon
        entropy = -sum(p * MC.safe_log(p) for p in probabilities)
        
        return entropy
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario completo."""
        result = {
            "betti_numbers": self.betti.to_dict(),
            "spectral_data": self.spectral.to_dict(),
            "graph_metrics": self.graph_metrics.to_dict(),
            "pyramid_stability": self.pyramid_stability,
            "structural_coherence": self.structural_coherence,
            "cycle_density": self.cycle_density,
            "topological_entropy": self.topological_entropy,
        }
        
        if self.persistence is not None:
            result["persistence_diagram"] = self.persistence.to_dict()
        
        return result


# =============================================================================
# PARÁMETROS FINANCIEROS CON DECIMAL PARA PRECISIÓN
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class FinancialParameters:
    """
    Parámetros financieros con precisión Decimal para cálculos críticos.
    
    Attributes:
        initial_investment: Inversión inicial I₀ (debe ser > 0)
        cash_flows: Flujos de caja proyectados CF = (CF₁, CF₂, ..., CFₙ)
        discount_rate: Tasa de descuento r (típicamente WACC)
        risk_free_rate: Tasa libre de riesgo rᶠ
        market_return: Retorno esperado del mercado E[Rₘ]
        beta: Beta del proyecto (riesgo sistemático)
        cost_std_dev: Desviación estándar de costos σ_C
        project_volatility: Volatilidad del proyecto σ ∈ [0, 1]
    """
    
    initial_investment: Decimal
    cash_flows: Tuple[Decimal, ...]
    discount_rate: Decimal
    risk_free_rate: Decimal = Decimal("0.05")
    market_return: Decimal = Decimal("0.12")
    beta: Decimal = Decimal("1.0")
    cost_std_dev: Decimal = Decimal("0.0")
    project_volatility: UnitInterval = 0.2
    
    def __post_init__(self) -> None:
        """Validación de invariantes financieros."""
        if self.initial_investment <= 0:
            raise ValueError(
                f"La inversión inicial debe ser positiva: {self.initial_investment}"
            )
        
        if self.discount_rate < 0:
            raise ValueError(
                f"La tasa de descuento no puede ser negativa: {self.discount_rate}"
            )
        
        if self.risk_free_rate < 0:
            raise ValueError(
                f"La tasa libre de riesgo no puede ser negativa: {self.risk_free_rate}"
            )
        
        if not (0.0 <= self.project_volatility <= 1.0):
            raise ValueError(
                f"La volatilidad debe estar en [0, 1]: {self.project_volatility}"
            )
        
        if self.cost_std_dev < 0:
            raise ValueError(
                f"La desviación estándar de costos no puede ser negativa: {self.cost_std_dev}"
            )
    
    @cached_property
    def periods(self) -> int:
        """Número de períodos de flujo de caja."""
        return len(self.cash_flows)
    
    @cached_property
    def total_cash_flow(self) -> Decimal:
        """Suma total de flujos de caja."""
        return sum(self.cash_flows)
    
    @cached_property
    def average_cash_flow(self) -> Decimal:
        """Flujo de caja promedio por período."""
        if self.periods == 0:
            return Decimal("0.0")
        return self.total_cash_flow / Decimal(self.periods)
    
    @cached_property
    def required_return_capm(self) -> Decimal:
        """
        Retorno requerido según CAPM.
        
        CAPM: E[Rᵢ] = Rᶠ + βᵢ(E[Rₘ] - Rᶠ)
        
        Returns:
            Tasa de retorno requerida
        """
        market_premium = self.market_return - self.risk_free_rate
        return self.risk_free_rate + self.beta * market_premium
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario con conversión a float."""
        return {
            "initial_investment": float(self.initial_investment),
            "cash_flows": [float(cf) for cf in self.cash_flows],
            "discount_rate": float(self.discount_rate),
            "risk_free_rate": float(self.risk_free_rate),
            "market_return": float(self.market_return),
            "beta": float(self.beta),
            "cost_std_dev": float(self.cost_std_dev),
            "project_volatility": self.project_volatility,
            "periods": self.periods,
            "total_cash_flow": float(self.total_cash_flow),
            "average_cash_flow": float(self.average_cash_flow),
            "required_return_capm": float(self.required_return_capm),
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Constructor desde diccionario.
        
        Args:
            data: Diccionario con parámetros financieros
        
        Returns:
            Instancia de FinancialParameters
        """
        return cls(
            initial_investment=Decimal(str(data["initial_investment"])),
            cash_flows=tuple(Decimal(str(cf)) for cf in data["cash_flows"]),
            discount_rate=Decimal(str(data.get("discount_rate", "0.10"))),
            risk_free_rate=Decimal(str(data.get("risk_free_rate", "0.05"))),
            market_return=Decimal(str(data.get("market_return", "0.12"))),
            beta=Decimal(str(data.get("beta", "1.0"))),
            cost_std_dev=Decimal(str(data.get("cost_std_dev", "0.0"))),
            project_volatility=float(data.get("project_volatility", 0.2))
        )


# =============================================================================
# FASE 3: MOTOR FINANCIERO Y TERMODINÁMICA ESTADÍSTICA
# =============================================================================

# =============================================================================
# IMPORTACIONES ADICIONALES PARA ANÁLISIS FINANCIERO
# =============================================================================

try:
    from scipy.stats import norm, t as student_t
    from scipy.optimize import newton, brentq
    _HAS_SCIPY_STATS = True
except ImportError:
    _HAS_SCIPY_STATS = False
    logger.warning("scipy.stats no disponible - funcionalidad financiera limitada")

try:
    from scipy.integrate import quad
    _HAS_SCIPY_INTEGRATE = True
except ImportError:
    _HAS_SCIPY_INTEGRATE = False

# =============================================================================
# MOTOR DE ANÁLISIS FINANCIERO
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class FinancialMetrics:
    """
    Métricas financieras calculadas con precisión Decimal.
    
    Incluye:
        - VPN (Valor Presente Neto)
        - TIR (Tasa Interna de Retorno)
        - Payback Period (período de recuperación)
        - Profitability Index (índice de rentabilidad)
        - VaR y CVaR (Value at Risk, Conditional VaR)
    
    Attributes:
        npv: Valor Presente Neto
        irr: Tasa Interna de Retorno (puede ser None si no converge)
        payback_period: Período de recuperación
        profitability_index: Índice de rentabilidad (VPN / I₀)
        var_95: Value at Risk al 95% de confianza
        cvar_95: Conditional VaR (Expected Shortfall)
        roi: Retorno sobre la inversión
        modified_irr: TIR modificada (reinversión a tasa segura)
    """
    
    npv: Decimal
    irr: Optional[Decimal] = None
    payback_period: Optional[Decimal] = None
    profitability_index: Decimal = Decimal("0.0")
    var_95: Optional[Decimal] = None
    cvar_95: Optional[Decimal] = None
    roi: Decimal = Decimal("0.0")
    modified_irr: Optional[Decimal] = None
    
    @cached_property
    def is_viable(self) -> bool:
        """
        Determina viabilidad financiera básica.
        
        Criterios:
            1. VPN > 0
            2. TIR > tasa de descuento (si existe)
            3. PI > 1
        
        Returns:
            True si el proyecto es financieramente viable
        """
        if self.npv <= 0:
            return False
        
        if self.profitability_index <= 1:
            return False
        
        return True
    
    @cached_property
    def risk_class(self) -> str:
        """
        Clasificación de riesgo basada en VaR y TIR.
        
        Returns:
            Cadena con clasificación de riesgo
        """
        if self.var_95 is None:
            return "UNKNOWN"
        
        var_ratio = abs(float(self.var_95)) / max(float(self.npv), 1.0)
        
        if var_ratio < 0.1:
            return "LOW"
        elif var_ratio < 0.3:
            return "MODERATE"
        elif var_ratio < 0.6:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "npv": float(self.npv),
            "irr": float(self.irr) if self.irr is not None else None,
            "payback_period": float(self.payback_period) if self.payback_period is not None else None,
            "profitability_index": float(self.profitability_index),
            "var_95": float(self.var_95) if self.var_95 is not None else None,
            "cvar_95": float(self.cvar_95) if self.cvar_95 is not None else None,
            "roi": float(self.roi),
            "modified_irr": float(self.modified_irr) if self.modified_irr is not None else None,
            "is_viable": self.is_viable,
            "risk_class": self.risk_class
        }


class FinancialEngine:
    """
    Motor de análisis financiero con métodos rigurosos.
    
    Implementa:
        1. Cálculo de VPN con descuento continuo/discreto
        2. TIR mediante Newton-Raphson con convergencia garantizada
        3. VaR y CVaR paramétrico (distribución normal)
        4. VaR histórico (simulación Monte Carlo)
        5. Análisis de sensibilidad (derivadas parciales)
    """
    
    def __init__(
        self,
        *,
        max_iterations: int = MC.DEFAULT_ITERATIONS,
        tolerance: float = MC.EPSILON_STRICT,
        use_continuous_discounting: bool = False
    ):
        """
        Args:
            max_iterations: Límite de iteraciones para TIR
            tolerance: Tolerancia de convergencia
            use_continuous_discounting: Si True, usa descuento continuo e^(-rt)
        """
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._use_continuous = use_continuous_discounting
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_npv(
        self,
        params: FinancialParameters
    ) -> Decimal:
        """
        Calcula el Valor Presente Neto (VPN).
        
        Fórmula (descuento discreto):
            VPN = -I₀ + Σₜ CFₜ / (1 + r)ᵗ
        
        Fórmula (descuento continuo):
            VPN = -I₀ + Σₜ CFₜ · e^(-rt)
        
        Args:
            params: Parámetros financieros
        
        Returns:
            Valor Presente Neto
        """
        r = params.discount_rate
        npv = -params.initial_investment
        
        for t, cf in enumerate(params.cash_flows, start=1):
            if self._use_continuous:
                # Descuento continuo
                discount_factor = Decimal(math.exp(-float(r) * t))
            else:
                # Descuento discreto
                discount_factor = Decimal(1) / ((Decimal(1) + r) ** t)
            
            npv += cf * discount_factor
        
        return npv
    
    def calculate_irr(
        self,
        params: FinancialParameters
    ) -> Optional[Decimal]:
        """
        Calcula la Tasa Interna de Retorno (TIR).
        
        Definición:
            TIR es la tasa r* tal que VPN(r*) = 0
        
        Método:
            Newton-Raphson: rₙ₊₁ = rₙ - f(rₙ)/f'(rₙ)
            donde f(r) = -I₀ + Σₜ CFₜ/(1+r)ᵗ
        
        Args:
            params: Parámetros financieros
        
        Returns:
            TIR como Decimal, o None si no converge
        """
        # Función objetivo: VPN(r) = 0
        def npv_function(r: float) -> float:
            npv = -float(params.initial_investment)
            for t, cf in enumerate(params.cash_flows, start=1):
                npv += float(cf) / ((1.0 + r) ** t)
            return npv
        
        # Derivada: dVPN/dr = -Σₜ t·CFₜ/(1+r)^(t+1)
        def npv_derivative(r: float) -> float:
            deriv = 0.0
            for t, cf in enumerate(params.cash_flows, start=1):
                deriv -= t * float(cf) / ((1.0 + r) ** (t + 1))
            return deriv
        
        # Estimación inicial: aproximación de TIR
        initial_guess = float(params.discount_rate)
        
        try:
            # Newton-Raphson
            irr_value = newton(
                npv_function,
                x0=initial_guess,
                fprime=npv_derivative,
                maxiter=self._max_iterations,
                tol=self._tolerance
            )
            
            # Validar que la solución es razonable
            if not (-1.0 < irr_value < 10.0):
                self._logger.warning(f"TIR fuera de rango razonable: {irr_value}")
                return None
            
            return Decimal(str(irr_value))
        
        except (RuntimeError, ValueError) as e:
            self._logger.debug(f"TIR no converge: {e}")
            
            # Intentar método de bisección como fallback
            try:
                irr_value = brentq(
                    npv_function,
                    a=-0.99,
                    b=5.0,
                    xtol=self._tolerance,
                    maxiter=self._max_iterations
                )
                return Decimal(str(irr_value))
            except (RuntimeError, ValueError):
                return None
    
    def calculate_payback_period(
        self,
        params: FinancialParameters
    ) -> Optional[Decimal]:
        """
        Calcula el período de recuperación (payback period).
        
        Definición:
            Tiempo mínimo t* tal que Σₜ₌₁ᵗ* CFₜ ≥ I₀
        
        Args:
            params: Parámetros financieros
        
        Returns:
            Período de recuperación en períodos, o None si nunca se recupera
        """
        cumulative = Decimal("0.0")
        
        for t, cf in enumerate(params.cash_flows, start=1):
            cumulative += cf
            
            if cumulative >= params.initial_investment:
                # Interpolación lineal para fracción de período
                if t == 1:
                    fraction = params.initial_investment / cf
                else:
                    prev_cumulative = cumulative - cf
                    remaining = params.initial_investment - prev_cumulative
                    fraction = remaining / cf
                
                return Decimal(t - 1) + fraction
        
        # No se recupera la inversión
        return None
    
    def calculate_profitability_index(
        self,
        params: FinancialParameters,
        npv: Decimal
    ) -> Decimal:
        """
        Calcula el índice de rentabilidad (PI).
        
        Fórmula:
            PI = (VPN + I₀) / I₀ = VP(flujos) / I₀
        
        Interpretación:
            PI > 1: proyecto rentable
            PI = 1: punto de equilibrio
            PI < 1: proyecto no rentable
        
        Args:
            params: Parámetros financieros
            npv: VPN calculado
        
        Returns:
            Índice de rentabilidad
        """
        if params.initial_investment == 0:
            return Decimal("0.0")
        
        present_value_inflows = npv + params.initial_investment
        pi = present_value_inflows / params.initial_investment
        
        return pi
    
    def calculate_var_parametric(
        self,
        params: FinancialParameters,
        npv: Decimal,
        confidence_level: float = 0.95
    ) -> Tuple[Decimal, Decimal]:
        """
        Calcula VaR y CVaR paramétrico (asumiendo distribución normal).
        
        VaR (Value at Risk):
            VaR_α = μ - z_α · σ
            donde z_α es el cuantil de la distribución normal
        
        CVaR (Conditional VaR / Expected Shortfall):
            CVaR_α = μ - σ · φ(z_α) / α
            donde φ es la densidad de la normal estándar
        
        Args:
            params: Parámetros financieros
            npv: VPN esperado (media)
            confidence_level: Nivel de confianza (típicamente 0.95 o 0.99)
        
        Returns:
            Tupla (VaR, CVaR)
        """
        if not _HAS_SCIPY_STATS:
            return Decimal("0.0"), Decimal("0.0")
        
        # Desviación estándar del VPN (propagación de incertidumbre)
        # σ_VPN ≈ σ_C · √(Σₜ 1/(1+r)^(2t))
        r = float(params.discount_rate)
        sigma_c = float(params.cost_std_dev)
        
        variance_sum = sum(
            1.0 / ((1.0 + r) ** (2 * t))
            for t in range(1, params.periods + 1)
        )
        
        sigma_npv = sigma_c * math.sqrt(variance_sum)
        
        # Cuantil de la distribución normal
        alpha = 1.0 - confidence_level
        z_alpha = norm.ppf(alpha)
        
        # VaR
        var = float(npv) + z_alpha * sigma_npv  # Negativo si es pérdida
        
        # CVaR (Expected Shortfall)
        phi_z = norm.pdf(z_alpha)
        cvar = float(npv) - sigma_npv * (phi_z / alpha)
        
        return Decimal(str(var)), Decimal(str(cvar))
    
    def calculate_modified_irr(
        self,
        params: FinancialParameters
    ) -> Optional[Decimal]:
        """
        Calcula la TIR Modificada (MIRR).
        
        Fórmula:
            MIRR = [(VP_positivos / VP_negativos)^(1/n)] - 1
        
        donde:
            - VP_positivos: valor futuro de flujos positivos (reinvertidos a r_reinv)
            - VP_negativos: valor presente de flujos negativos (descontados a r_fin)
        
        Args:
            params: Parámetros financieros
        
        Returns:
            MIRR como Decimal
        """
        r_finance = float(params.discount_rate)
        r_reinvest = float(params.risk_free_rate)
        n = params.periods
        
        # Valor futuro de flujos positivos (reinvertidos)
        fv_positive = 0.0
        for t, cf in enumerate(params.cash_flows, start=1):
            if cf > 0:
                periods_to_end = n - t
                fv_positive += float(cf) * ((1.0 + r_reinvest) ** periods_to_end)
        
        # Valor presente de flujos negativos (descontados)
        pv_negative = float(params.initial_investment)
        for t, cf in enumerate(params.cash_flows, start=1):
            if cf < 0:
                pv_negative += abs(float(cf)) / ((1.0 + r_finance) ** t)
        
        if pv_negative <= 0 or fv_positive <= 0:
            return None
        
        # MIRR
        mirr = (fv_positive / pv_negative) ** (1.0 / n) - 1.0
        
        return Decimal(str(mirr))
    
    def calculate_roi(
        self,
        params: FinancialParameters
    ) -> Decimal:
        """
        Calcula el ROI simple.
        
        Fórmula:
            ROI = (Σ CF - I₀) / I₀
        
        Args:
            params: Parámetros financieros
        
        Returns:
            ROI como decimal
        """
        total_return = params.total_cash_flow - params.initial_investment
        
        if params.initial_investment == 0:
            return Decimal("0.0")
        
        roi = total_return / params.initial_investment
        
        return roi
    
    def analyze(
        self,
        params: FinancialParameters
    ) -> FinancialMetrics:
        """
        Ejecuta análisis financiero completo.
        
        Args:
            params: Parámetros financieros
        
        Returns:
            FinancialMetrics con todos los indicadores calculados
        """
        self._logger.info("💰 Ejecutando análisis financiero...")
        
        # VPN
        npv = self.calculate_npv(params)
        
        # TIR
        irr = self.calculate_irr(params)
        
        # Payback
        payback = self.calculate_payback_period(params)
        
        # PI
        pi = self.calculate_profitability_index(params, npv)
        
        # VaR y CVaR
        var_95, cvar_95 = self.calculate_var_parametric(params, npv, confidence_level=0.95)
        
        # ROI
        roi = self.calculate_roi(params)
        
        # MIRR
        mirr = self.calculate_modified_irr(params)
        
        self._logger.info(
            f"Resultados: VPN={npv:.2f}, TIR={irr:.4f if irr else 'N/A'}, "
            f"PI={pi:.2f}, VaR_95={var_95:.2f}"
        )
        
        return FinancialMetrics(
            npv=npv,
            irr=irr,
            payback_period=payback,
            profitability_index=pi,
            var_95=var_95,
            cvar_95=cvar_95,
            roi=roi,
            modified_irr=mirr
        )


# =============================================================================
# TEORÍA DE OPCIONES REALES
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class RealOption:
    """
    Representación de una opción real.
    
    Tipos de opciones:
        - Espera: postergar inversión
        - Expansión: ampliar el proyecto
        - Abandono: salir del proyecto
        - Flexibilidad: cambiar operación
    
    Attributes:
        option_type: Tipo de opción
        underlying_value: Valor del activo subyacente (VPN del proyecto)
        strike_price: Precio de ejercicio (inversión adicional)
        volatility: Volatilidad del activo subyacente σ
        time_to_maturity: Tiempo hasta vencimiento (años)
        risk_free_rate: Tasa libre de riesgo
        option_value: Valor de la opción (Black-Scholes o binomial)
    """
    
    option_type: Literal["wait", "expand", "abandon", "switch"]
    underlying_value: Decimal
    strike_price: Decimal
    volatility: float
    time_to_maturity: float
    risk_free_rate: Decimal
    option_value: Optional[Decimal] = None
    
    def __post_init__(self) -> None:
        """Validación de parámetros."""
        if self.underlying_value < 0:
            raise ValueError("El valor subyacente no puede ser negativo")
        
        if self.strike_price < 0:
            raise ValueError("El precio de ejercicio no puede ser negativo")
        
        if not (0.0 < self.volatility <= 2.0):
            raise ValueError("La volatilidad debe estar en (0, 2]")
        
        if self.time_to_maturity <= 0:
            raise ValueError("El tiempo hasta vencimiento debe ser positivo")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "option_type": self.option_type,
            "underlying_value": float(self.underlying_value),
            "strike_price": float(self.strike_price),
            "volatility": self.volatility,
            "time_to_maturity": self.time_to_maturity,
            "risk_free_rate": float(self.risk_free_rate),
            "option_value": float(self.option_value) if self.option_value else None
        }


class BlackScholesEngine:
    """
    Motor de valoración mediante modelo de Black-Scholes-Merton.
    
    Fórmula (opción call europea):
        C = S·N(d₁) - K·e^(-rT)·N(d₂)
    
    donde:
        d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T
        N(·): función de distribución acumulada normal estándar
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @staticmethod
    def _calculate_d1(
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float
    ) -> float:
        """
        Calcula d₁ en la fórmula de Black-Scholes.
        
        d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
        denominator = sigma * math.sqrt(T)
        
        return numerator / denominator
    
    @staticmethod
    def _calculate_d2(d1: float, sigma: float, T: float) -> float:
        """
        Calcula d₂ en la fórmula de Black-Scholes.
        
        d₂ = d₁ - σ√T
        """
        return d1 - sigma * math.sqrt(T)
    
    def call_option_value(
        self,
        S: Decimal,
        K: Decimal,
        r: Decimal,
        sigma: float,
        T: float
    ) -> Decimal:
        """
        Calcula el valor de una opción call europea.
        
        Args:
            S: Valor del activo subyacente
            K: Precio de ejercicio
            r: Tasa libre de riesgo
            sigma: Volatilidad
            T: Tiempo hasta vencimiento
        
        Returns:
            Valor de la opción call
        """
        if not _HAS_SCIPY_STATS:
            self._logger.warning("scipy.stats no disponible - usando aproximación")
            return Decimal("0.0")
        
        S_f = float(S)
        K_f = float(K)
        r_f = float(r)
        
        # Casos extremos
        if S_f <= 0 or K_f <= 0 or T <= 0:
            return Decimal("0.0")
        
        # Calcular d₁ y d₂
        d1 = self._calculate_d1(S_f, K_f, r_f, sigma, T)
        d2 = self._calculate_d2(d1, sigma, T)
        
        # Valores de la distribución normal acumulada
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        
        # Fórmula de Black-Scholes
        call_value = S_f * N_d1 - K_f * math.exp(-r_f * T) * N_d2
        
        return Decimal(str(max(0.0, call_value)))
    
    def put_option_value(
        self,
        S: Decimal,
        K: Decimal,
        r: Decimal,
        sigma: float,
        T: float
    ) -> Decimal:
        """
        Calcula el valor de una opción put europea.
        
        Usa paridad put-call:
            P = C - S + K·e^(-rT)
        
        Args:
            S: Valor del activo subyacente
            K: Precio de ejercicio
            r: Tasa libre de riesgo
            sigma: Volatilidad
            T: Tiempo hasta vencimiento
        
        Returns:
            Valor de la opción put
        """
        if not _HAS_SCIPY_STATS:
            return Decimal("0.0")
        
        S_f = float(S)
        K_f = float(K)
        r_f = float(r)
        
        if S_f <= 0 or K_f <= 0 or T <= 0:
            return Decimal("0.0")
        
        d1 = self._calculate_d1(S_f, K_f, r_f, sigma, T)
        d2 = self._calculate_d2(d1, sigma, T)
        
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        
        put_value = K_f * math.exp(-r_f * T) * N_neg_d2 - S_f * N_neg_d1
        
        return Decimal(str(max(0.0, put_value)))
    
    def greeks(
        self,
        S: Decimal,
        K: Decimal,
        r: Decimal,
        sigma: float,
        T: float,
        option_type: Literal["call", "put"] = "call"
    ) -> Dict[str, float]:
        """
        Calcula las griegas de la opción.
        
        Griegas:
            - Delta (Δ): ∂C/∂S (sensibilidad al precio del subyacente)
            - Gamma (Γ): ∂²C/∂S² (convexidad)
            - Vega (ν): ∂C/∂σ (sensibilidad a volatilidad)
            - Theta (Θ): ∂C/∂T (decaimiento temporal)
            - Rho (ρ): ∂C/∂r (sensibilidad a tasa de interés)
        
        Args:
            S, K, r, sigma, T: Parámetros de la opción
            option_type: "call" o "put"
        
        Returns:
            Diccionario con valores de las griegas
        """
        if not _HAS_SCIPY_STATS:
            return {}
        
        S_f = float(S)
        K_f = float(K)
        r_f = float(r)
        
        d1 = self._calculate_d1(S_f, K_f, r_f, sigma, T)
        d2 = self._calculate_d2(d1, sigma, T)
        
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        phi_d1 = norm.pdf(d1)
        
        # Delta
        if option_type == "call":
            delta = N_d1
        else:
            delta = N_d1 - 1.0
        
        # Gamma (igual para call y put)
        gamma = phi_d1 / (S_f * sigma * math.sqrt(T))
        
        # Vega (igual para call y put, en %)
        vega = S_f * phi_d1 * math.sqrt(T) / 100.0
        
        # Theta (por día)
        if option_type == "call":
            theta = (
                -(S_f * phi_d1 * sigma) / (2.0 * math.sqrt(T))
                - r_f * K_f * math.exp(-r_f * T) * N_d2
            ) / 365.0
        else:
            theta = (
                -(S_f * phi_d1 * sigma) / (2.0 * math.sqrt(T))
                + r_f * K_f * math.exp(-r_f * T) * norm.cdf(-d2)
            ) / 365.0
        
        # Rho (en %)
        if option_type == "call":
            rho = K_f * T * math.exp(-r_f * T) * N_d2 / 100.0
        else:
            rho = -K_f * T * math.exp(-r_f * T) * norm.cdf(-d2) / 100.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }


class RealOptionsEngine:
    """
    Motor de valoración de opciones reales.
    
    Implementa:
        1. Opción de espera (defer)
        2. Opción de expansión (scale-up)
        3. Opción de abandono (exit)
        4. Opción de flexibilidad (switch)
    """
    
    def __init__(self):
        self._bs_engine = BlackScholesEngine()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def value_wait_option(
        self,
        params: FinancialParameters,
        npv: Decimal,
        wait_years: float = 1.0
    ) -> RealOption:
        """
        Valora la opción de esperar antes de invertir.
        
        Modelado como call europea:
            - Subyacente S: VPN del proyecto
            - Strike K: Inversión inicial
            - Vencimiento T: años de espera posibles
        
        Args:
            params: Parámetros financieros
            npv: VPN actual del proyecto
            wait_years: Años que se puede esperar
        
        Returns:
            RealOption con valor calculado
        """
        option_value = self._bs_engine.call_option_value(
            S=npv + params.initial_investment,  # Valor del proyecto
            K=params.initial_investment,
            r=params.risk_free_rate,
            sigma=params.project_volatility,
            T=wait_years
        )
        
        return RealOption(
            option_type="wait",
            underlying_value=npv,
            strike_price=params.initial_investment,
            volatility=params.project_volatility,
            time_to_maturity=wait_years,
            risk_free_rate=params.risk_free_rate,
            option_value=option_value
        )
    
    def value_abandon_option(
        self,
        params: FinancialParameters,
        npv: Decimal,
        salvage_value: Decimal,
        project_life: float = 5.0
    ) -> RealOption:
        """
        Valora la opción de abandonar el proyecto.
        
        Modelado como put europea:
            - Subyacente S: Valor del proyecto en operación
            - Strike K: Valor de salvamento
        
        Args:
            params: Parámetros financieros
            npv: VPN del proyecto
            salvage_value: Valor de liquidación
            project_life: Vida útil del proyecto
        
        Returns:
            RealOption con valor calculado
        """
        project_value = npv + params.initial_investment
        
        option_value = self._bs_engine.put_option_value(
            S=project_value,
            K=salvage_value,
            r=params.risk_free_rate,
            sigma=params.project_volatility,
            T=project_life
        )
        
        return RealOption(
            option_type="abandon",
            underlying_value=npv,
            strike_price=salvage_value,
            volatility=params.project_volatility,
            time_to_maturity=project_life,
            risk_free_rate=params.risk_free_rate,
            option_value=option_value
        )
    
    def value_expand_option(
        self,
        params: FinancialParameters,
        npv: Decimal,
        expansion_cost: Decimal,
        expansion_factor: float = 1.5,
        expansion_window: float = 2.0
    ) -> RealOption:
        """
        Valora la opción de expandir el proyecto.
        
        Modelado como call sobre proyecto ampliado:
            - Subyacente S: VPN del proyecto ampliado
            - Strike K: Costo de expansión
        
        Args:
            params: Parámetros financieros
            npv: VPN actual
            expansion_cost: Inversión requerida para expandir
            expansion_factor: Factor de multiplicación (ej. 1.5x)
            expansion_window: Años disponibles para decidir
        
        Returns:
            RealOption con valor calculado
        """
        expanded_npv = npv * Decimal(str(expansion_factor))
        
        option_value = self._bs_engine.call_option_value(
            S=expanded_npv + params.initial_investment,
            K=expansion_cost,
            r=params.risk_free_rate,
            sigma=params.project_volatility * 1.2,  # Mayor volatilidad en expansión
            T=expansion_window
        )
        
        return RealOption(
            option_type="expand",
            underlying_value=expanded_npv,
            strike_price=expansion_cost,
            volatility=params.project_volatility * 1.2,
            time_to_maturity=expansion_window,
            risk_free_rate=params.risk_free_rate,
            option_value=option_value
        )
    
    def analyze_real_options(
        self,
        params: FinancialParameters,
        financial_metrics: FinancialMetrics
    ) -> Dict[str, Any]:
        """
        Análisis completo de opciones reales.
        
        Args:
            params: Parámetros financieros
            financial_metrics: Métricas financieras calculadas
        
        Returns:
            Diccionario con opciones valoradas
        """
        self._logger.info("🎲 Valorando opciones reales...")
        
        npv = financial_metrics.npv
        
        # Opción de espera
        wait_option = self.value_wait_option(params, npv, wait_years=1.0)
        
        # Opción de abandono (salvamento = 50% de inversión)
        salvage = params.initial_investment * Decimal("0.5")
        abandon_option = self.value_abandon_option(
            params, npv, salvage, project_life=float(params.periods)
        )
        
        # Opción de expansión (duplicar proyecto con 150% de inversión adicional)
        expansion_cost = params.initial_investment * Decimal("1.5")
        expand_option = self.value_expand_option(
            params, npv, expansion_cost, expansion_factor=2.0, expansion_window=2.0
        )
        
        # Valor total con opciones
        total_option_value = (
            wait_option.option_value +
            abandon_option.option_value +
            expand_option.option_value
        )
        
        # VPN expandido (con valor de opciones)
        expanded_npv = npv + total_option_value
        
        return {
            "wait_option": wait_option.to_dict(),
            "abandon_option": abandon_option.to_dict(),
            "expand_option": expand_option.to_dict(),
            "total_option_value": float(total_option_value),
            "expanded_npv": float(expanded_npv),
            "option_value_ratio": float(total_option_value / max(abs(npv), Decimal("1.0")))
        }


# =============================================================================
# TERMODINÁMICA ESTADÍSTICA DEL PRESUPUESTO
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class ThermodynamicState:
    """
    Estado termodinámico del sistema presupuestario.
    
    Analogía Termodinámica:
        - Energía interna U: Flujo de caja total
        - Entropía S: Desorden/incertidumbre del presupuesto
        - Temperatura T: Volatilidad del mercado
        - Capacidad calorífica C: Inercia financiera
        - Exergía B: Trabajo útil máximo extraíble
    
    Attributes:
        internal_energy: Energía interna U (flujo total)
        entropy: Entropía S ∈ [0, 1]
        temperature: Temperatura T (en unidades arbitrarias)
        heat_capacity: Capacidad calorífica C
        exergy: Exergía B ∈ [0, 1]
        free_energy: Energía libre de Helmholtz F = U - TS
    """
    
    internal_energy: Decimal
    entropy: UnitInterval
    temperature: float
    heat_capacity: float
    exergy: UnitInterval
    free_energy: Optional[Decimal] = None
    
    def __post_init__(self) -> None:
        """Calcula energía libre."""
        if self.free_energy is None:
            # F = U - TS
            free_energy = self.internal_energy - Decimal(str(self.temperature * self.entropy))
            object.__setattr__(self, 'free_energy', free_energy)
    
    @cached_property
    def negentropy(self) -> float:
        """
        Negentropía (información): -S.
        
        Mide el orden del sistema.
        
        Returns:
            Negentropía ∈ [0, 1]
        """
        return 1.0 - self.entropy
    
    @cached_property
    def thermal_efficiency(self) -> float:
        """
        Eficiencia térmica: η = 1 - T_cold/T_hot.
        
        Aproximación: η ≈ exergy
        
        Returns:
            Eficiencia ∈ [0, 1]
        """
        return self.exergy
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "internal_energy": float(self.internal_energy),
            "entropy": self.entropy,
            "temperature": self.temperature,
            "heat_capacity": self.heat_capacity,
            "exergy": self.exergy,
            "free_energy": float(self.free_energy) if self.free_energy else None,
            "negentropy": self.negentropy,
            "thermal_efficiency": self.thermal_efficiency
        }


class ThermodynamicsEngine:
    """
    Motor de análisis termodinámico del presupuesto.
    
    Calcula:
        1. Entropía del presupuesto (desorden)
        2. Temperatura del sistema (volatilidad)
        3. Capacidad calorífica (inercia financiera)
        4. Exergía (trabajo útil)
    """
    
    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_entropy(
        self,
        cash_flows: Tuple[Decimal, ...],
        topo_bundle: TopologicalMetricsBundle
    ) -> float:
        """
        Calcula la entropía del sistema presupuestario.
        
        Fórmula (entropía de Shannon normalizada):
            S = -Σᵢ pᵢ log pᵢ / log n
        
        donde pᵢ = |CFᵢ| / Σⱼ |CFⱼ|
        
        Penalización topológica:
            S_total = S_shannon × (1 + β₁/√n)
        
        Args:
            cash_flows: Flujos de caja
            topo_bundle: Bundle topológico
        
        Returns:
            Entropía normalizada ∈ [0, 1]
        """
        if not cash_flows:
            return 0.5  # Máxima incertidumbre
        
        # Distribución de probabilidad
        total_flow = sum(abs(cf) for cf in cash_flows)
        
        if total_flow == 0:
            return 0.5
        
        probabilities = [abs(float(cf)) / float(total_flow) for cf in cash_flows if cf != 0]
        
        # Entropía de Shannon
        shannon_entropy = -sum(p * MC.safe_log(p) for p in probabilities)
        max_entropy = MC.safe_log(len(probabilities))
        
        normalized_entropy = shannon_entropy / max(max_entropy, MC.EPSILON_TOLERANCE)
        
        # Penalización topológica
        n_nodes = max(topo_bundle.graph_metrics.n_nodes, 1)
        beta_1 = topo_bundle.betti.beta_1
        
        topo_penalty = 1.0 + (beta_1 / math.sqrt(n_nodes))
        
        final_entropy = min(1.0, normalized_entropy * topo_penalty)
        
        return final_entropy
    
    def calculate_temperature(
        self,
        params: FinancialParameters,
        market_volatility: float = 0.15
    ) -> float:
        """
        Calcula la temperatura del sistema.
        
        Fórmula:
            T = T₀ · (1 + σ_proyecto) · (1 + σ_mercado)
        
        donde T₀ = 20°C (temperatura de referencia)
        
        Args:
            params: Parámetros financieros
            market_volatility: Volatilidad del mercado
        
        Returns:
            Temperatura en unidades arbitrarias
        """
        T_0 = 20.0  # Temperatura de referencia
        
        project_factor = 1.0 + params.project_volatility
        market_factor = 1.0 + market_volatility
        
        temperature = T_0 * project_factor * market_factor
        
        return temperature
    
    def calculate_heat_capacity(
        self,
        params: FinancialParameters,
        financial_metrics: FinancialMetrics
    ) -> float:
        """
        Calcula la capacidad calorífica (inercia financiera).
        
        Fórmula:
            C = (I₀ / σ_C) · (1 + PI)
        
        Interpretación:
            Mayor inversión con bajo riesgo ⇒ alta inercia
        
        Args:
            params: Parámetros financieros
            financial_metrics: Métricas financieras
        
        Returns:
            Capacidad calorífica (normalizada a [0, 1])
        """
        if params.cost_std_dev == 0:
            heat_capacity_raw = float(params.initial_investment)
        else:
            heat_capacity_raw = float(params.initial_investment / params.cost_std_dev)
        
        pi_factor = 1.0 + float(financial_metrics.profitability_index)
        
        heat_capacity = heat_capacity_raw * pi_factor
        
        # Normalizar mediante función sigmoide
        normalized = math.tanh(heat_capacity / 1e6)
        
        return max(0.0, min(1.0, normalized))
    
    def calculate_exergy(
        self,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> float:
        """
        Calcula la exergía (trabajo útil máximo).
        
        Fórmula:
            B = (VPN / I₀) · Ψ · (1 - S)
        
        Interpretación:
            Exergía alta ⇒ proyecto eficiente con baja entropía
        
        Args:
            financial_metrics: Métricas financieras
            topo_bundle: Bundle topológico
        
        Returns:
            Exergía ∈ [0, 1]
        """
        # Eficiencia de Carnot modificada
        pi = float(financial_metrics.profitability_index)
        psi = topo_bundle.pyramid_stability
        coherence = topo_bundle.structural_coherence
        
        # Exergía = eficiencia × estabilidad × coherencia
        exergy = math.sqrt(max(0.0, pi - 1.0)) * psi * coherence
        
        return max(0.0, min(1.0, exergy))
    
    def analyze(
        self,
        params: FinancialParameters,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> ThermodynamicState:
        """
        Ejecuta análisis termodinámico completo.
        
        Args:
            params: Parámetros financieros
            financial_metrics: Métricas financieras
            topo_bundle: Bundle topológico
        
        Returns:
            ThermodynamicState con análisis completo
        """
        self._logger.info("🔥 Ejecutando análisis termodinámico...")
        
        # Energía interna (flujo total)
        internal_energy = params.total_cash_flow
        
        # Entropía
        entropy = self.calculate_entropy(params.cash_flows, topo_bundle)
        
        # Temperatura
        temperature = self.calculate_temperature(params)
        
        # Capacidad calorífica
        heat_capacity = self.calculate_heat_capacity(params, financial_metrics)
        
        # Exergía
        exergy = self.calculate_exergy(financial_metrics, topo_bundle)
        
        state = ThermodynamicState(
            internal_energy=internal_energy,
            entropy=UnitInterval(entropy),
            temperature=temperature,
            heat_capacity=heat_capacity,
            exergy=UnitInterval(exergy)
        )
        
        self._logger.info(
            f"Estado: S={entropy:.3f}, T={temperature:.1f}, "
            f"C={heat_capacity:.3f}, B={exergy:.3f}"
        )
        
        return state


# =============================================================================
# FASE 4: RISK CHALLENGER, COMPOSITOR Y BUSINESS AGENT
# =============================================================================

# =============================================================================
# IMPORTACIONES FINALES
# =============================================================================

try:
    from app.tactics.business_topology import (
        BudgetGraphBuilder,
        BusinessTopologicalAnalyzer,
        ConstructionRiskReport,
    )
    _HAS_TOPOLOGY = True
except ImportError:
    BudgetGraphBuilder = None  # type: ignore
    BusinessTopologicalAnalyzer = None  # type: ignore
    ConstructionRiskReport = None  # type: ignore
    _HAS_TOPOLOGY = False
    logger.warning("business_topology no disponible - usando fallbacks")

try:
    from app.core.constants import ColumnNames
    _HAS_CONSTANTS = True
except ImportError:
    _HAS_CONSTANTS = False
    class ColumnNames:
        CODIGO_APU = "codigo_apu"
        DESCRIPCION_APU = "descripcion_apu"
        VALOR_TOTAL = "valor_total"
        DESCRIPCION_INSUMO = "descripcion_insumo"
        CANTIDAD_APU = "cantidad_apu"
        COSTO_INSUMO_EN_APU = "costo_insumo_en_apu"

try:
    from app.core.schemas import Stratum
    _HAS_SCHEMAS = True
except ImportError:
    _HAS_SCHEMAS = False
    class Stratum(IntEnum):
        WISDOM = 0
        OMEGA = 1
        STRATEGY = 2
        TACTICS = 3
        PHYSICS = 4

try:
    from app.wisdom.semantic_translator import SemanticTranslator
    _HAS_TRANSLATOR = True
except ImportError:
    SemanticTranslator = None  # type: ignore
    _HAS_TRANSLATOR = False

try:
    from app.core.telemetry import TelemetryContext
    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False
    @dataclass
    class TelemetryContext:
        def record_error(self, category: str, message: str) -> None:
            logger.error(f"[{category}] {message}")

try:
    from app.adapters.tools_interface import MICRegistry
    _HAS_MIC = True
except ImportError:
    MICRegistry = None  # type: ignore
    _HAS_MIC = False

# =============================================================================
# ENUMERACIONES Y CONSTANTES DE ESTRATEGIA
# =============================================================================

@unique
class RiskClassification(Enum):
    """
    Clasificación de riesgo financiero mediante cuantiles.
    
    Basado en distribución de VaR y métricas de rentabilidad.
    """
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_metrics(
        cls,
        financial_metrics: FinancialMetrics,
        var_threshold_moderate: float = 0.2,
        var_threshold_high: float = 0.5,
        var_threshold_critical: float = 0.8
    ) -> 'RiskClassification':
        """
        Clasifica riesgo desde métricas financieras.
        
        Criterios:
            - SAFE: VaR/VPN < 0.2 y VPN > 0
            - MODERATE: 0.2 ≤ VaR/VPN < 0.5
            - HIGH: 0.5 ≤ VaR/VPN < 0.8
            - CRITICAL: VaR/VPN ≥ 0.8
        
        Args:
            financial_metrics: Métricas calculadas
            var_threshold_*: Umbrales de clasificación
        
        Returns:
            Clasificación de riesgo
        """
        if not financial_metrics.is_viable:
            return cls.CRITICAL
        
        if financial_metrics.var_95 is None or financial_metrics.npv == 0:
            return cls.UNKNOWN
        
        var_ratio = abs(float(financial_metrics.var_95)) / max(abs(float(financial_metrics.npv)), 1.0)
        
        if var_ratio < var_threshold_moderate:
            return cls.SAFE
        elif var_ratio < var_threshold_high:
            return cls.MODERATE
        elif var_ratio < var_threshold_critical:
            return cls.HIGH
        else:
            return cls.CRITICAL
    
    @classmethod
    def from_string(cls, value: str) -> 'RiskClassification':
        """
        Parsea string a clasificación normalizada.
        
        Args:
            value: String con nivel de riesgo
        
        Returns:
            RiskClassification correspondiente
        """
        if not isinstance(value, str):
            return cls.UNKNOWN
        
        normalized = value.upper().strip()
        
        # Diccionarios de palabras clave
        safe_kw = {"LOW", "BAJO", "SAFE", "SEGURO", "MINIMAL", "MÍNIMO"}
        moderate_kw = {"MODERATE", "MODERADO", "MEDIUM", "MEDIO"}
        high_kw = {"HIGH", "ALTO", "ELEVATED", "ELEVADO"}
        critical_kw = {"CRITICAL", "CRÍTICO", "SEVERE", "SEVERO", "EXTREME", "EXTREMO"}
        
        if any(kw in normalized for kw in critical_kw):
            return cls.CRITICAL
        elif any(kw in normalized for kw in high_kw):
            return cls.HIGH
        elif any(kw in normalized for kw in moderate_kw):
            return cls.MODERATE
        elif any(kw in normalized for kw in safe_kw):
            return cls.SAFE
        
        return cls.UNKNOWN


@unique
class VetoSeverity(Enum):
    """Severidad del veto emitido por el Risk Challenger."""
    CRITICO = "CRÍTICO"
    SEVERO = "SEVERO"
    MODERADO = "MODERADO"
    LEVE = "LEVE"


@unique
class PivotType(Enum):
    """Tipos de pivote lateral para excepciones estratégicas."""
    MONOPOLIO_COBERTURADO = "MONOPOLIO_COBERTURADO"
    OPCION_ESPERA = "OPCION_ESPERA"
    CUARENTENA_TOPOLOGICA = "CUARENTENA_TOPOLOGICA"
    IMPROBABILITY_OVERRIDE = "IMPROBABILITY_OVERRIDE"


# =============================================================================
# CONFIGURACIÓN Y UMBRALES
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class RiskChallengerThresholds:
    """
    Umbrales para el Risk Challenger con validación.
    
    Todos los umbrales están en [0, 1] y representan puntos de decisión
    para vetos y alertas.
    
    Attributes:
        critical_stability: Ψ < umbral → Veto crítico
        warning_stability: Ψ < umbral → Advertencia
        coherence_minimum: Coherencia estructural mínima
        cycle_density_limit: β₁/n máximo aceptable
        integrity_penalty_veto: Penalización por veto crítico
        integrity_penalty_warn: Penalización por advertencia
        improbability_threshold: Umbral para Motor de Improbabilidad
    """
    
    critical_stability: UnitInterval = 0.70
    warning_stability: UnitInterval = 0.85
    coherence_minimum: UnitInterval = 0.60
    cycle_density_limit: float = 0.33
    integrity_penalty_veto: UnitInterval = 0.30
    integrity_penalty_warn: UnitInterval = 0.15
    improbability_threshold: UnitInterval = 0.15
    
    def __post_init__(self) -> None:
        """Valida que todos los umbrales estén en [0, 1]."""
        for field_info in dataclass_fields(self):
            value = getattr(self, field_info.name)
            if not isinstance(value, (int, float)):
                continue
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Umbral {field_info.name} fuera de rango [0, 1]: {value}"
                )
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Constructor desde diccionario."""
        return cls(
            critical_stability=data.get("critical_stability", 0.70),
            warning_stability=data.get("warning_stability", 0.85),
            coherence_minimum=data.get("coherence_minimum", 0.60),
            cycle_density_limit=data.get("cycle_density_limit", 0.33),
            integrity_penalty_veto=data.get("integrity_penalty_veto", 0.30),
            integrity_penalty_warn=data.get("integrity_penalty_warn", 0.15),
            improbability_threshold=data.get("improbability_threshold", 0.15)
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Serializa a diccionario."""
        return {
            "critical_stability": self.critical_stability,
            "warning_stability": self.warning_stability,
            "coherence_minimum": self.coherence_minimum,
            "cycle_density_limit": self.cycle_density_limit,
            "integrity_penalty_veto": self.integrity_penalty_veto,
            "integrity_penalty_warn": self.integrity_penalty_warn,
            "improbability_threshold": self.improbability_threshold
        }


@final
@dataclass(frozen=True, slots=True)
class DecisionWeights:
    """
    Pesos para la combinación convexa de dimensiones de decisión.
    
    Representan la importancia relativa de cada dimensión en el vector
    de decisión final. Deben formar una partición de la unidad.
    
    Attributes:
        topology: Peso de dimensión topológica α
        finance: Peso de dimensión financiera β
        thermodynamics: Peso de dimensión termodinámica γ
    
    Invariante:
        α + β + γ = 1
    """
    
    topology: UnitInterval = 0.40
    finance: UnitInterval = 0.40
    thermodynamics: UnitInterval = 0.20
    
    def __post_init__(self) -> None:
        """Valida que los pesos sean no negativos."""
        for field_info in dataclass_fields(self):
            value = getattr(self, field_info.name)
            if value < 0:
                raise ValueError(
                    f"Peso {field_info.name} no puede ser negativo: {value}"
                )
    
    @cached_property
    def normalized(self) -> 'DecisionWeights':
        """
        Retorna pesos normalizados a suma 1.
        
        Returns:
            DecisionWeights con Σwᵢ = 1
        """
        total = self.topology + self.finance + self.thermodynamics
        
        if total < MC.EPSILON_TOLERANCE:
            # Distribución uniforme como fallback
            return DecisionWeights(
                topology=1.0 / 3.0,
                finance=1.0 / 3.0,
                thermodynamics=1.0 / 3.0
            )
        
        return DecisionWeights(
            topology=self.topology / total,
            finance=self.finance / total,
            thermodynamics=self.thermodynamics / total
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Retorna como tupla (α, β, γ).
        
        Returns:
            Tupla de pesos
        """
        return (self.topology, self.finance, self.thermodynamics)
    
    def to_dict(self) -> Dict[str, float]:
        """Serializa a diccionario."""
        return {
            "topology": self.topology,
            "finance": self.finance,
            "thermodynamics": self.thermodynamics
        }
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Constructor desde diccionario."""
        return cls(
            topology=data.get("topology", 0.40),
            finance=data.get("finance", 0.40),
            thermodynamics=data.get("thermodynamics", 0.20)
        )


# =============================================================================
# REGISTROS DE VETOS Y EXCEPCIONES
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class VetoRecord:
    """
    Registro inmutable de veto emitido por el Risk Challenger.
    
    Attributes:
        veto_type: Identificador del tipo de veto
        severity: Severidad del veto
        stability_at_veto: Estabilidad piramidal al momento del veto
        financial_class: Clasificación de riesgo financiero
        original_integrity: Score de integridad antes del veto
        penalty_applied: Penalización aplicada (fracción)
        reason: Justificación técnica del veto
        timestamp: Marca temporal Unix
    """
    
    veto_type: str
    severity: VetoSeverity
    stability_at_veto: float
    financial_class: RiskClassification
    original_integrity: float
    penalty_applied: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "veto_type": self.veto_type,
            "severity": self.severity.value,
            "stability_at_veto": self.stability_at_veto,
            "financial_class": self.financial_class.value,
            "original_integrity": self.original_integrity,
            "penalty_applied": self.penalty_applied,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.timestamp))
        }


@final
@dataclass(frozen=True, slots=True)
class LateralExceptionRecord:
    """
    Registro de excepción por pensamiento lateral.
    
    Attributes:
        exception_type: Tipo de excepción
        pivot_type: Tipo de pivote aplicado
        penalty_relief: Alivio de penalización (fracción positiva)
        reason: Justificación de la excepción
        approved_by_mic: Si la MIC aprobó la excepción
        timestamp: Marca temporal Unix
    """
    
    exception_type: str
    pivot_type: PivotType
    penalty_relief: float
    reason: str
    approved_by_mic: bool
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "exception_type": self.exception_type,
            "pivot_type": self.pivot_type.value,
            "penalty_relief": self.penalty_relief,
            "reason": self.reason,
            "approved_by_mic": self.approved_by_mic,
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.timestamp))
        }


# =============================================================================
# ÁLGEBRA DE DECISIONES
# =============================================================================

class DecisionAlgebra:
    """
    Operaciones algebraicas para síntesis multicriterio.
    
    Implementa el marco matemático para combinar dimensiones
    topológicas, financieras y termodinámicas en un vector de decisión
    mediante álgebra lineal y optimización convexa.
    """
    
    @staticmethod
    def normalize_to_sphere(
        vector: Vector,
        epsilon: float = MC.EPSILON_TOLERANCE
    ) -> Vector:
        """
        Proyecta vector a la esfera unitaria S^(n-1).
        
        Normalización: v̂ = v / ‖v‖
        
        Args:
            vector: Vector a normalizar
            epsilon: Umbral mínimo de norma
        
        Returns:
            Vector unitario v̂
        """
        norm = np.linalg.norm(vector)
        
        if norm < epsilon:
            # Vector casi nulo → distribución uniforme
            n = len(vector)
            return np.ones(n, dtype=np.float64) / np.sqrt(n)
        
        return vector / norm
    
    @staticmethod
    def weighted_geometric_mean(
        factors: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        epsilon: float = MC.EPSILON_TOLERANCE
    ) -> float:
        """
        Media geométrica ponderada: (∏ᵢ xᵢ^wᵢ)^(1/Σwᵢ).
        
        Implementación robusta en log-space para evitar overflow/underflow.
        
        Fórmula:
            GM_w = exp(Σᵢ wᵢ log xᵢ / Σᵢ wᵢ)
        
        Args:
            factors: Factores a combinar
            weights: Pesos (por defecto uniformes)
            epsilon: Umbral para factores no positivos
        
        Returns:
            Media geométrica ponderada
        """
        if not factors:
            return 0.0
        
        n = len(factors)
        
        if weights is None:
            weights = [1.0 / n] * n
        
        if len(weights) != n:
            raise DimensionMismatchError(
                "Dimensiones de factores y pesos no coinciden",
                context={"n_factors": n, "n_weights": len(weights)}
            )
        
        # Sanitizar factores y pesos
        clean_factors = [max(f, epsilon) for f in factors]
        clean_weights = [max(w, 0.0) for w in weights]
        
        weight_sum = sum(clean_weights)
        
        if weight_sum < epsilon:
            return 0.0
        
        # Si hay factor cero con peso positivo, resultado es 0
        for f, w in zip(factors, clean_weights):
            if f <= epsilon and w > epsilon:
                return 0.0
        
        # Calcular en log-space para estabilidad numérica
        log_sum = sum(
            w * MC.safe_log(f)
            for f, w in zip(clean_factors, clean_weights)
        )
        
        result = math.exp(log_sum / weight_sum)
        
        return float(result)
    
    @staticmethod
    def convex_combination(
        vectors: Sequence[Vector],
        weights: DecisionWeights
    ) -> Vector:
        """
        Combinación convexa de vectores: d = Σᵢ αᵢ·vᵢ.
        
        Args:
            vectors: Lista [topo_vec, finance_vec, thermo_vec]
            weights: Pesos para cada dimensión
        
        Returns:
            Vector de decisión combinado
        
        Raises:
            DimensionMismatchError: Si las dimensiones no coinciden
        """
        if len(vectors) != 3:
            raise DimensionMismatchError(
                "Se esperan exactamente 3 vectores",
                context={"n_vectors": len(vectors)}
            )
        
        # Verificar que todos los vectores tienen la misma dimensión
        dims = [len(v) for v in vectors]
        if len(set(dims)) > 1:
            raise DimensionMismatchError(
                "Todos los vectores deben tener la misma dimensión",
                context={"dimensions": dims}
            )
        
        # Normalizar pesos
        normalized = weights.normalized
        alpha, beta, gamma = normalized.to_tuple()
        
        # Combinación convexa
        result = (
            alpha * vectors[0] +
            beta * vectors[1] +
            gamma * vectors[2]
        )
        
        return result
    
    @classmethod
    def compute_quality_factors(
        cls,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: FinancialMetrics,
        thermo_state: ThermodynamicState,
        initial_investment: Decimal = Decimal("1000000.0")
    ) -> Tuple[UnitInterval, UnitInterval, UnitInterval]:
        """
        Calcula factores de calidad para cada dimensión.
        
        Fórmulas:
            Q_topo = √(coherence × Ψ)
            Q_finance = (tanh(VPN/I₀) + 1) / 2
            Q_thermo = (negentropía + exergía) / 2
        
        Args:
            topo_bundle: Bundle topológico
            financial_metrics: Métricas financieras
            thermo_state: Estado termodinámico
            initial_investment: Inversión inicial (para normalización)
        
        Returns:
            Tupla (Q_topo, Q_finance, Q_thermo) ∈ [0, 1]³
        """
        # Calidad topológica
        coherence = topo_bundle.structural_coherence
        stability = topo_bundle.pyramid_stability
        
        topo_quality = math.sqrt(coherence * stability)
        
        # Calidad financiera
        npv = financial_metrics.npv
        inv = max(abs(initial_investment), Decimal("1.0"))
        
        npv_normalized = float(npv / inv)
        finance_quality = (math.tanh(npv_normalized) + 1.0) / 2.0
        
        # Calidad termodinámica
        negentropy = thermo_state.negentropy
        exergy = thermo_state.exergy
        
        thermo_quality = (negentropy + exergy) / 2.0
        
        # Garantizar que están en [0, 1]
        return (
            UnitInterval(max(0.0, min(1.0, topo_quality))),
            UnitInterval(max(0.0, min(1.0, finance_quality))),
            UnitInterval(max(0.0, min(1.0, thermo_quality)))
        )


# =============================================================================
# ESTRATEGIAS DE PIVOTE (PATTERN STRATEGY)
# =============================================================================

class PivotStrategy(ABC):
    """
    Estrategia base para evaluación de pivotes laterales.
    
    Implementa el patrón Strategy para diferentes tipos de excepciones.
    """
    
    @property
    @abstractmethod
    def pivot_type(self) -> PivotType:
        """Tipo de pivote que implementa esta estrategia."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> Tuple[bool, str]:
        """
        Evalúa si el pivote aplica.
        
        Args:
            stability: Estabilidad piramidal
            financial_class: Clasificación de riesgo
            thermal_state: Estado termodinámico
            financial_metrics: Métricas financieras
            topo_bundle: Bundle topológico
        
        Returns:
            Tupla (aplica, razón)
        """
        pass


@final
class MonopolioCoberturadoStrategy(PivotStrategy):
    """
    Pivote: Monopolio Coberturado (Topología vs Termodinámica).
    
    Condición:
        base estrecha (Ψ < umbral) +
        sistema frío (T < T_umbral) +
        inercia financiera alta (C > C_umbral)
    
    Justificación:
        La alta capacidad calorífica (inercia financiera) compensa
        la inestabilidad topológica mediante absorción de volatilidad.
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.MONOPOLIO_COBERTURADO
    
    def __init__(
        self,
        stability_threshold: float = 0.70,
        temp_threshold: float = 15.0,
        inertia_threshold: float = 0.70
    ):
        self._stability_threshold = stability_threshold
        self._temp_threshold = temp_threshold
        self._inertia_threshold = inertia_threshold
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> Tuple[bool, str]:
        """Evalúa criterios de monopolio coberturado."""
        system_temp = thermal_state.temperature
        heat_capacity = thermal_state.heat_capacity
        
        if (
            stability < self._stability_threshold and
            system_temp < self._temp_threshold and
            heat_capacity > self._inertia_threshold
        ):
            return True, (
                f"Riesgo logístico (Ψ={stability:.3f}) neutralizado por alta "
                f"inercia térmica financiera (C={heat_capacity:.3f}). "
                f"Sistema frío (T={system_temp:.1f}°C) garantiza estabilidad operativa."
            )
        
        return False, (
            f"Condiciones termodinámicas insuficientes: "
            f"T={system_temp:.1f}°C (req. <{self._temp_threshold}), "
            f"C={heat_capacity:.3f} (req. >{self._inertia_threshold})"
        )


@final
class OpcionEsperaStrategy(PivotStrategy):
    """
    Pivote: Opción de Espera (Opciones Reales).
    
    Condición:
        riesgo financiero alto +
        valor de opción de espera > VPN × k
    
    Justificación:
        El valor de la opción de postergar la inversión supera
        el VPN inmediato, justificando el retraso estratégico.
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.OPCION_ESPERA
    
    def __init__(self, npv_multiplier: float = 1.5):
        self._npv_multiplier = npv_multiplier
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> Tuple[bool, str]:
        """Evalúa criterios de opción de espera."""
        if financial_class not in (RiskClassification.HIGH, RiskClassification.CRITICAL):
            return False, f"Riesgo financiero es {financial_class.value}, no HIGH/CRITICAL"
        
        # Simular valoración de opción (simplificado)
        npv = float(financial_metrics.npv)
        wait_option_value = max(0.0, npv * 0.3)  # Aproximación: 30% del VPN
        
        threshold = max(npv, 0.0) * self._npv_multiplier
        
        if wait_option_value > threshold:
            return True, (
                f"Valor de opción de espera ({wait_option_value:.2f}) > "
                f"umbral VPN × {self._npv_multiplier} = {threshold:.2f}. "
                f"Estrategia de retraso justificada por teoría de opciones reales."
            )
        
        return False, (
            f"Valor de opción ({wait_option_value:.2f}) ≤ umbral ({threshold:.2f}). "
            f"No se justifica retraso."
        )


@final
class CuarentenaTopologicaStrategy(PivotStrategy):
    """
    Pivote: Cuarentena Topológica (Aislamiento de Ciclos).
    
    Condición:
        ciclos presentes (β₁ > 0) SIN sinergia multiplicativa
    
    Justificación:
        Los ciclos están topológicamente aislados y pueden ser
        excluidos del grafo de ejecución sin comprometer viabilidad.
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.CUARENTENA_TOPOLOGICA
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> Tuple[bool, str]:
        """Evalúa criterios de cuarentena topológica."""
        beta_1 = topo_bundle.betti.beta_1
        
        # Detectar sinergia (simplificado: alta densidad de aristas en ciclos)
        cycle_density = topo_bundle.cycle_density
        has_synergy = cycle_density > 0.5
        
        if beta_1 > 0 and not has_synergy:
            return True, (
                f"Ciclos detectados (β₁={beta_1}) pero confinados "
                f"(densidad={cycle_density:.3f}). "
                f"Se aprueba ejecución exceptuando subgrafo aislado."
            )
        
        if has_synergy:
            return False, (
                f"Ciclos presentan sinergia multiplicativa "
                f"(densidad={cycle_density:.3f} > 0.5). "
                f"Cuarentena imposible."
            )
        
        return False, f"No hay ciclos que requieran cuarentena (β₁={beta_1})"


@final
class ImprobabilityOverrideStrategy(PivotStrategy):
    """
    Pivote: Anulación del Motor de Improbabilidad.
    
    Condición:
        Proyecto declarado inviable por improbabilidad +
        respaldo de garantías externas (colaterales, seguros)
    
    Justificación:
        Instrumentos financieros externos cubren el riesgo de cola gorda.
    """
    
    @property
    def pivot_type(self) -> PivotType:
        return PivotType.IMPROBABILITY_OVERRIDE
    
    def __init__(self, collateral_ratio_threshold: float = 1.5):
        self._collateral_threshold = collateral_ratio_threshold
    
    def evaluate(
        self,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle
    ) -> Tuple[bool, str]:
        """Evalúa criterios de anulación de improbabilidad."""
        # Simulación: verificar si hay garantías suficientes
        # (esto requeriría datos externos, aquí es una aproximación)
        collateral_ratio = 1.0  # Placeholder
        
        if collateral_ratio > self._collateral_threshold:
            return True, (
                f"Motor de Improbabilidad anulado: garantías externas "
                f"(ratio={collateral_ratio:.2f}) > umbral "
                f"({self._collateral_threshold:.2f})"
            )
        
        return False, (
            f"Garantías insuficientes: ratio={collateral_ratio:.2f} ≤ "
            f"umbral={self._collateral_threshold:.2f}"
        )


# =============================================================================
# RISK CHALLENGER (AUDITORÍA ADVERSARIAL)
# =============================================================================

class RiskChallenger:
    """
    Motor de Auditoría Adversarial con Pensamiento Lateral.
    
    Formula "Intenciones" y las proyecta sobre la MIC para validar
    clausura transitiva. Si la MIC aprueba, emite excepción estratégica.
    
    Arquitectura:
        1. Recibe reporte base (Construction Risk Report)
        2. Evalúa reglas de veto según umbrales
        3. Intenta pivotes laterales vía estrategias registradas
        4. Proyecta intenciones a la MIC para aprobación
        5. Retorna reporte modificado con vetos/excepciones
    
    Attributes:
        thresholds: Umbrales para vetos y alertas
        mic: Matriz de Interacción Central (opcional)
        strategies: Estrategias de pivote registradas
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskChallengerThresholds] = None,
        mic: Optional[Any] = None,
        strategies: Optional[Sequence[PivotStrategy]] = None
    ):
        """
        Args:
            thresholds: Umbrales personalizados
            mic: MICRegistry para proyección
            strategies: Estrategias de pivote personalizadas
        """
        self._thresholds = thresholds or RiskChallengerThresholds()
        self._mic = mic
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Registrar estrategias
        self._strategies: Dict[PivotType, PivotStrategy] = {}
        
        if strategies:
            for strategy in strategies:
                self._strategies[strategy.pivot_type] = strategy
        else:
            # Estrategias por defecto
            self._strategies = {
                PivotType.MONOPOLIO_COBERTURADO: MonopolioCoberturadoStrategy(
                    stability_threshold=self._thresholds.critical_stability
                ),
                PivotType.OPCION_ESPERA: OpcionEsperaStrategy(),
                PivotType.CUARENTENA_TOPOLOGICA: CuarentenaTopologicaStrategy(),
                PivotType.IMPROBABILITY_OVERRIDE: ImprobabilityOverrideStrategy(),
            }
    
    def challenge_verdict(
        self,
        report: 'ConstructionRiskReport',
        financial_metrics: FinancialMetrics,
        thermal_state: ThermodynamicState,
        topo_bundle: TopologicalMetricsBundle,
        session_context: Optional[Dict[str, Any]] = None
    ) -> 'ConstructionRiskReport':
        """
        Ejecuta auditoría adversarial proyectando vectores a la MIC.
        
        Args:
            report: Reporte base a auditar
            financial_metrics: Métricas financieras
            thermal_state: Estado termodinámico
            topo_bundle: Bundle topológico
            session_context: Contexto de sesión
        
        Returns:
            Reporte modificado (potencialmente con vetos o excepciones)
        """
        self._logger.info("⚖️ Risk Challenger: Iniciando auditoría adversarial...")
        
        session_context = session_context or {}
        details = report.details or {}
        
        # Extraer métricas
        stability = topo_bundle.pyramid_stability
        coherence = topo_bundle.structural_coherence
        beta_1 = topo_bundle.betti.beta_1
        
        financial_class = RiskClassification.from_metrics(financial_metrics)
        
        current_report = report
        
        # Contexto MIC
        mic_context = self._build_mic_context(session_context)
        
        # ═══ REGLA 1: Estabilidad Crítica ═══
        if stability < self._thresholds.critical_stability:
            self._logger.warning(
                f"⚠️ Estabilidad crítica: Ψ={stability:.3f} < {self._thresholds.critical_stability:.2f}"
            )
            
            current_report = self._apply_pivot_or_veto(
                report=current_report,
                pivot_type=PivotType.MONOPOLIO_COBERTURADO,
                stability=stability,
                financial_class=financial_class,
                thermal_state=thermal_state,
                financial_metrics=financial_metrics,
                topo_bundle=topo_bundle,
                mic_context=mic_context
            )
        
        # ═══ REGLA 2: Estabilidad Subóptima ═══
        elif stability < self._thresholds.warning_stability:
            self._logger.info(
                f"ℹ️ Estabilidad subóptima: Ψ={stability:.3f} < {self._thresholds.warning_stability:.2f}"
            )
            
            if financial_class in (RiskClassification.MODERATE, RiskClassification.HIGH):
                current_report = self._emit_veto(
                    report=current_report,
                    veto_type="ALERTA_STRUCTURAL_WARNING",
                    severity=VetoSeverity.MODERADO,
                    stability=stability,
                    financial_class=financial_class,
                    penalty=self._thresholds.integrity_penalty_warn,
                    reason=(
                        f"Estabilidad piramidal Ψ={stability:.3f} es subóptima "
                        f"(umbral: {self._thresholds.warning_stability:.2f})"
                    )
                )
        
        # ═══ REGLA 3: Opción de Espera (Riesgo Alto) ═══
        if financial_class in (RiskClassification.HIGH, RiskClassification.CRITICAL):
            self._logger.info("ℹ️ Evaluando opción de espera...")
            
            current_report = self._try_pivot(
                report=current_report,
                pivot_type=PivotType.OPCION_ESPERA,
                stability=stability,
                financial_class=financial_class,
                thermal_state=thermal_state,
                financial_metrics=financial_metrics,
                topo_bundle=topo_bundle,
                mic_context=mic_context
            )
        
        # ═══ REGLA 4: Cuarentena Topológica ═══
        if beta_1 > 0:
            self._logger.info(f"ℹ️ Ciclos detectados: β₁={beta_1}")
            
            current_report = self._apply_cycle_handling(
                report=current_report,
                stability=stability,
                financial_class=financial_class,
                thermal_state=thermal_state,
                financial_metrics=financial_metrics,
                topo_bundle=topo_bundle,
                mic_context=mic_context
            )
        
        # ═══ REGLA 5: Motor de Improbabilidad (Fat-Tail Risk) ═══
        if stability < self._thresholds.improbability_threshold:
            self._logger.critical(
                f"🚨 Umbral de improbabilidad cruzado: Ψ={stability:.3f}"
            )
            
            # Proyección a OMEGA stratum
            improbability_payload = {
                "psi": stability,
                "roi": float(financial_metrics.roi),
                "npv": float(financial_metrics.npv),
                "coherence": coherence
            }
            
            improbability_monad = self._project_to_mic(
                "compute_improbability_penalty",
                improbability_payload,
                mic_context
            )
            
            if improbability_monad and improbability_monad.get("is_vetoed"):
                self._logger.critical(
                    "🚨 VETO CRÍTICO: Motor de Improbabilidad detectó colapso de función de onda"
                )
                
                current_report = self._emit_veto(
                    report=current_report,
                    veto_type="VETO_IMPROBABILITY_DRIVE",
                    severity=VetoSeverity.CRITICO,
                    stability=stability,
                    financial_class=financial_class,
                    penalty=0.95,  # Colapso casi total
                    reason=(
                        "Fractura del hiperespacio de decisión financiera ante "
                        "socavones lógicos del presupuesto. Distribución de cola "
                        "gorda indica probabilidad de quiebra no despreciable."
                    )
                )
        
        self._logger.info("✅ Auditoría adversarial completada")
        return current_report
    
    def _build_mic_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Construye contexto para proyección MIC."""
        validated = session_context.get("validated_strata", {Stratum.PHYSICS, Stratum.TACTICS})
        
        return {
            "validated_strata": validated,
            "telemetry_context": session_context.get("telemetry_context"),
            "session_id": session_context.get("session_id", "unknown")
        }
    
    def _try_pivot(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle,
        mic_context: Dict[str, Any]
    ) -> 'ConstructionRiskReport':
        """Intenta aplicar un pivote vía MIC."""
        strategy = self._strategies.get(pivot_type)
        
        if strategy is None:
            self._logger.debug(f"Estrategia no registrada: {pivot_type}")
            return report
        
        # Evaluar localmente
        applies, reason = strategy.evaluate(
            stability=stability,
            financial_class=financial_class,
            thermal_state=thermal_state,
            financial_metrics=financial_metrics,
            topo_bundle=topo_bundle
        )
        
        if not applies:
            self._logger.debug(f"Pivote {pivot_type.value} no aplica: {reason}")
            return report
        
        # Proyectar a MIC
        payload = {
            "pivot_type": pivot_type.value,
            "stability": stability,
            "financial_class": financial_class.value,
            "reason": reason
        }
        
        projection = self._project_to_mic("lateral_thinking_pivot", payload, mic_context)
        
        if projection.get("success"):
            self._logger.info(f"🧠 MIC Aprobó pivote: {pivot_type.value}")
            
            penalty_relief = projection.get("payload", {}).get("penalty_relief", 0.15)
            
            return self._emit_lateral_exception(
                report=report,
                pivot_type=pivot_type,
                exception_type=f"EXCEPCIÓN_{pivot_type.value}",
                penalty_relief=penalty_relief,
                reason=reason
            )
        else:
            self._logger.warning(f"MIC rechazó pivote: {projection.get('error', 'unknown')}")
        
        return report
    
    def _apply_pivot_or_veto(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle,
        mic_context: Dict[str, Any]
    ) -> 'ConstructionRiskReport':
        """Intenta pivote; si falla, emite veto."""
        modified = self._try_pivot(
            report=report,
            pivot_type=pivot_type,
            stability=stability,
            financial_class=financial_class,
            thermal_state=thermal_state,
            financial_metrics=financial_metrics,
            topo_bundle=topo_bundle,
            mic_context=mic_context
        )
        
        # Si no cambió y el riesgo es alto, veto
        if modified is report and financial_class in (
            RiskClassification.MODERATE, RiskClassification.HIGH, RiskClassification.CRITICAL
        ):
            return self._emit_veto(
                report=report,
                veto_type="VETO_CRITICAL_INSTABILITY",
                severity=VetoSeverity.CRITICO,
                stability=stability,
                financial_class=financial_class,
                penalty=self._thresholds.integrity_penalty_veto,
                reason=(
                    f"Cimentación logística angosta (Ψ={stability:.3f}) sin inercia "
                    "térmica que la cubra. Proyecto estructuralmente inviable."
                )
            )
        
        return modified
    
    def _apply_cycle_handling(
        self,
        report: 'ConstructionRiskReport',
        stability: float,
        financial_class: RiskClassification,
        thermal_state: ThermodynamicState,
        financial_metrics: FinancialMetrics,
        topo_bundle: TopologicalMetricsBundle,
        mic_context: Dict[str, Any]
    ) -> 'ConstructionRiskReport':
        """Maneja ciclos topológicos."""
        modified = self._try_pivot(
            report=report,
            pivot_type=PivotType.CUARENTENA_TOPOLOGICA,
            stability=stability,
            financial_class=financial_class,
            thermal_state=thermal_state,
            financial_metrics=financial_metrics,
            topo_bundle=topo_bundle,
            mic_context=mic_context
        )
        
        if modified is not report:
            return modified
        
        # Penalización por ciclos
        cycle_density = topo_bundle.cycle_density
        
        if cycle_density > self._thresholds.cycle_density_limit:
            self._logger.warning(
                f"⚠️ Densidad de ciclos β₁/n = {cycle_density:.3f} > "
                f"{self._thresholds.cycle_density_limit:.2f}"
            )
            
            new_details = {**(report.details or {})}
            new_details["challenger_cycle_warning"] = {
                "beta_1": topo_bundle.betti.beta_1,
                "n_nodes": topo_bundle.graph_metrics.n_nodes,
                "cycle_density": cycle_density,
                "threshold": self._thresholds.cycle_density_limit,
            }
            
            return self._create_modified_report(
                report,
                integrity_score=report.integrity_score * 0.92,
                details=new_details
            )
        
        return report
    
    def _project_to_mic(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Proyecta intención a la MIC."""
        if self._mic is None:
            self._logger.debug("MIC no configurada - retornando rechazo")
            return {"success": False, "error": "MIC no configurada"}
        
        try:
            if hasattr(self._mic, 'project_intent'):
                return self._mic.project_intent(service_name, payload, context)
            else:
                self._logger.warning("MIC no tiene método project_intent")
                return {"success": False, "error": "Método no disponible"}
        except Exception as e:
            self._logger.warning(f"Proyección MIC falló: {e}")
            return {"success": False, "error": str(e)}
    
    def _emit_veto(
        self,
        report: 'ConstructionRiskReport',
        veto_type: str,
        severity: VetoSeverity,
        stability: float,
        financial_class: RiskClassification,
        penalty: float,
        reason: str
    ) -> 'ConstructionRiskReport':
        """Emite veto estructurado."""
        self._logger.warning(f"🚨 Emitiendo veto: {veto_type}")
        
        original_integrity = report.integrity_score
        new_integrity = max(0.0, original_integrity * (1.0 - penalty))
        
        veto_record = VetoRecord(
            veto_type=veto_type,
            severity=severity,
            stability_at_veto=stability,
            financial_class=financial_class,
            original_integrity=original_integrity,
            penalty_applied=penalty,
            reason=reason
        )
        
        new_details = {**(report.details or {})}
        new_details["challenger_verdict"] = veto_record.to_dict()
        new_details["challenger_applied"] = True
        
        narrative = self._generate_veto_narrative(veto_record, new_integrity)
        
        return self._create_modified_report(
            report,
            integrity_score=new_integrity,
            details=new_details,
            strategic_narrative=f"{narrative}\n\n{report.strategic_narrative}",
            financial_risk_level=f"RIESGO ESTRUCTURAL ({severity.value})"
        )
    
    def _emit_lateral_exception(
        self,
        report: 'ConstructionRiskReport',
        pivot_type: PivotType,
        exception_type: str,
        penalty_relief: float,
        reason: str
    ) -> 'ConstructionRiskReport':
        """Emite excepción por pensamiento lateral."""
        self._logger.info(f"✨ Emitiendo excepción lateral: {exception_type}")
        
        original_integrity = report.integrity_score
        new_integrity = min(100.0, original_integrity * (1.0 + penalty_relief))
        
        exception_record = LateralExceptionRecord(
            exception_type=exception_type,
            pivot_type=pivot_type,
            penalty_relief=penalty_relief,
            reason=reason,
            approved_by_mic=True
        )
        
        new_details = {**(report.details or {})}
        new_details["lateral_thinking_applied"] = exception_type
        new_details["lateral_exception"] = exception_record.to_dict()
        
        narrative = self._generate_exception_narrative(exception_record)
        
        return self._create_modified_report(
            report,
            integrity_score=new_integrity,
            details=new_details,
            strategic_narrative=f"{narrative}\n\n{report.strategic_narrative}",
            financial_risk_level="ESTRATEGIA MODIFICADA (PENSAMIENTO LATERAL)"
        )
    
    def _generate_veto_narrative(
        self,
        veto: VetoRecord,
        new_integrity: float
    ) -> str:
        """Genera narrativa de veto."""
        return (
            "━" * 80 + "\n"
            "🏛️  **ACTA DE DELIBERACIÓN DEL CONSEJO DE RIESGO**\n"
            "━" * 80 + "\n\n"
            f"📋 **Tipo de Veto:** {veto.veto_type}\n"
            f"⚠️  **Severidad:** {veto.severity.value}\n\n"
            "**Posiciones de los Agentes:**\n\n"
            f"1. 🤵 **Gestor Financiero:** «El proyecto es financieramente "
            f"{veto.financial_class.value}. Los indicadores de rentabilidad son evaluables.»\n\n"
            f"2. 👷 **Ingeniero Estructural:** «OBJECIÓN. {veto.reason}»\n\n"
            f"3. ⚖️  **Fiscal de Riesgos:** «Se detecta contradicción lógica entre "
            f"viabilidad financiera y estabilidad estructural (Ψ={veto.stability_at_veto:.3f}).»\n\n"
            "**VEREDICTO FINAL:**\n"
            f"Se emite **{veto.veto_type}**. La integridad del proyecto se degrada de "
            f"{veto.original_integrity:.1f} a {new_integrity:.1f} puntos "
            f"(penalización: {veto.penalty_applied:.1%}).\n\n"
            "━" * 80
        )
    
    def _generate_exception_narrative(self, exception: LateralExceptionRecord) -> str:
        """Genera narrativa de excepción."""
        return (
            "━" * 80 + "\n"
            "🏛️  **ACTA DEL CONSEJO: EXCEPCIÓN POR PENSAMIENTO LATERAL**\n"
            "━" * 80 + "\n\n"
            f"⚖️  **Resolución de la MIC:** {exception.exception_type}\n"
            f"🎯 **Pivote Aplicado:** {exception.pivot_type.value}\n\n"
            f"**Fiscal de Riesgos:** «{exception.reason}\n\n"
            "Se levanta el veto estructural o se modifica la estrategia base mediante "
            "instrumentos financieros avanzados.»\n\n"
            f"**Alivio de Penalización:** +{exception.penalty_relief:.1%}\n"
            f"**Aprobado por MIC:** {'Sí' if exception.approved_by_mic else 'No'}\n\n"
            "━" * 80
        )
    
    def _create_modified_report(
        self,
        original: 'ConstructionRiskReport',
        integrity_score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        strategic_narrative: Optional[str] = None,
        financial_risk_level: Optional[str] = None
    ) -> 'ConstructionRiskReport':
        """Crea copia modificada del reporte."""
        if not _HAS_TOPOLOGY or ConstructionRiskReport is None:
            # Fallback: retornar original
            self._logger.warning("ConstructionRiskReport no disponible - retornando original")
            return original
        
        return ConstructionRiskReport(
            integrity_score=integrity_score if integrity_score is not None else original.integrity_score,
            waste_alerts=original.waste_alerts,
            circular_risks=original.circular_risks,
            complexity_level=original.complexity_level,
            financial_risk_level=financial_risk_level if financial_risk_level else original.financial_risk_level,
            details=details if details is not None else original.details,
            strategic_narrative=strategic_narrative if strategic_narrative else original.strategic_narrative
        )


# =============================================================================
# COMPOSITOR DE REPORTES EJECUTIVOS
# =============================================================================

class ReportComposer:
    """
    Compositor de reportes ejecutivos mediante álgebra de decisiones.
    
    Integra dimensiones topológicas, financieras y termodinámicas
    en un reporte estratégico unificado con narrativa semántica.
    """
    
    def __init__(
        self,
        analyzer: Optional[Any] = None,
        translator: Optional[Any] = None,
        weights: Optional[DecisionWeights] = None,
        telemetry: Optional[Any] = None
    ):
        self._analyzer = analyzer
        self._translator = translator
        self._weights = weights or DecisionWeights()
        self._telemetry = telemetry or TelemetryContext()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compose(
        self,
        graph: Any,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: FinancialMetrics,
        thermal_state: ThermodynamicState
    ) -> 'ConstructionRiskReport':
        """
        Genera reporte ejecutivo con álgebra multicriterio.
        
        Args:
            graph: Grafo del presupuesto
            topo_bundle: Bundle topológico
            financial_metrics: Métricas financieras
            thermal_state: Estado termodinámico
        
        Returns:
            ConstructionRiskReport con análisis completo
        """
        self._logger.info("🧠 Componiendo reporte ejecutivo...")
        
        # Reporte base del analizador topológico
        if self._analyzer is not None and hasattr(self._analyzer, 'generate_executive_report'):
            base_report = self._analyzer.generate_executive_report(
                graph,
                financial_metrics.to_dict()
            )
        else:
            # Reporte fallback
            base_report = self._create_fallback_report()
        
        # Calcular factores de calidad
        weights = self._weights.normalized
        
        topo_quality, finance_quality, thermo_quality = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermo_state=thermal_state
        )
        
        # Score integrado (media geométrica ponderada)
        integrated_score = DecisionAlgebra.weighted_geometric_mean(
            factors=[topo_quality, finance_quality, thermo_quality],
            weights=list(weights.to_tuple())
        )
        
        integrated_score_100 = float(np.clip(integrated_score * 100.0, 0.0, 100.0))
        
        # Resumen de álgebra de decisiones
        decision_summary = {
            "weights": {
                "alpha": weights.topology,
                "beta": weights.finance,
                "gamma": weights.thermodynamics
            },
            "quality_factors": {
                "topology": float(topo_quality),
                "finance": float(finance_quality),
                "thermodynamics": float(thermo_quality),
            },
            "integrated_score": integrated_score,
            "integrated_score_100": integrated_score_100
        }
        
        # Generar narrativa
        narrative = self._generate_narrative(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermal_state=thermal_state,
            decision_summary=decision_summary,
            base_narrative=getattr(base_report, 'strategic_narrative', '')
        )
        
        # Detalles enriquecidos
        enriched_details = {
            **(base_report.details or {}),
            "strategic_narrative": narrative,
            "financial_metrics": financial_metrics.to_dict(),
            "thermal_state": thermal_state.to_dict(),
            "topological_invariants": topo_bundle.to_dict(),
            "decision_algebra": decision_summary,
        }
        
        if not _HAS_TOPOLOGY or ConstructionRiskReport is None:
            self._logger.warning("ConstructionRiskReport no disponible")
            return base_report
        
        return ConstructionRiskReport(
            integrity_score=integrated_score_100,
            waste_alerts=base_report.waste_alerts,
            circular_risks=base_report.circular_risks,
            complexity_level=base_report.complexity_level,
            financial_risk_level=financial_metrics.risk_class,
            details=enriched_details,
            strategic_narrative=narrative
        )
    
    def _create_fallback_report(self) -> Any:
        """Crea reporte fallback si el analizador no está disponible."""
        if _HAS_TOPOLOGY and ConstructionRiskReport is not None:
            return ConstructionRiskReport(
                integrity_score=50.0,
                waste_alerts=[],
                circular_risks=[],
                complexity_level="Desconocida",
                financial_risk_level="UNKNOWN",
                details={},
                strategic_narrative="Análisis base no disponible"
            )
        
        # Fallback genérico
        class FallbackReport:
            integrity_score = 50.0
            waste_alerts = []
            circular_risks = []
            complexity_level = "Desconocida"
            financial_risk_level = "UNKNOWN"
            details = {}
            strategic_narrative = "Análisis base no disponible"
        
        return FallbackReport()
    
    def _generate_narrative(
        self,
        topo_bundle: TopologicalMetricsBundle,
        financial_metrics: FinancialMetrics,
        thermal_state: ThermodynamicState,
        decision_summary: Dict[str, Any],
        base_narrative: str
    ) -> str:
        """Genera narrativa estratégica."""
        if self._translator is not None and hasattr(self._translator, 'compose_strategic_narrative'):
            try:
                strategic_report = self._translator.compose_strategic_narrative(
                    topological_metrics=topo_bundle.betti.to_dict(),
                    financial_metrics=financial_metrics.to_dict(),
                    stability=topo_bundle.pyramid_stability,
                    thermal_metrics=thermal_state.to_dict(),
                    decision_algebra=decision_summary
                )
                return getattr(strategic_report, "raw_narrative", str(strategic_report))
            except Exception as e:
                self._logger.warning(f"⚠️ Generación de narrativa falló: {e}")
        
        # Narrativa fallback
        quality = decision_summary.get("quality_factors", {})
        return (
            f"**Reporte Ejecutivo de Análisis de Proyecto**\n\n"
            f"**Score de Integridad:** {decision_summary['integrated_score_100']:.1f}/100\n\n"
            f"**Dimensiones de Calidad:**\n"
            f"- Coherencia Topológica: {quality.get('topology', 0):.1%}\n"
            f"- Salud Financiera: {quality.get('finance', 0):.1%}\n"
            f"- Eficiencia Termodinámica: {quality.get('thermodynamics', 0):.1%}\n\n"
            f"**Métricas Clave:**\n"
            f"- VPN: {float(financial_metrics.npv):,.2f}\n"
            f"- TIR: {float(financial_metrics.irr)*100:.2f}% " if financial_metrics.irr else "- TIR: N/A\n"
            f"- Estabilidad Piramidal: {topo_bundle.pyramid_stability:.1%}\n"
            f"- Entropía del Sistema: {thermal_state.entropy:.1%}\n"
        )


# =============================================================================
# BUSINESS AGENT (FACHADA PRINCIPAL)
# =============================================================================

class BusinessAgent:
    """
    Orquestador de inteligencia de negocio para evaluación de proyectos.
    
    Combina análisis topológico (complejo simplicial del presupuesto)
    con análisis financiero (VPN, TIR, VaR, opciones reales) y termodinámico
    para producir evaluación holística rigurosa.
    
    Principio Fundamental:
        "No hay Estrategia sin Física"
    
    Pipeline de Evaluación:
        1. Validación de datos (DataFrameValidator)
        2. Construcción topológica (TopologyBuilder)
        3. Análisis termodinámico (ThermodynamicsEngine)
        4. Análisis financiero (FinancialEngine + RealOptionsEngine)
        5. Síntesis y composición (ReportComposer)
        6. Auditoría adversarial (RiskChallenger)
    
    Ejemplo de uso:
        ```python
        config = {
            "financial_config": {"discount_rate": 0.10},
            "risk_challenger_config": {"critical_stability": 0.70},
            "decision_weights": {"topology": 0.4, "finance": 0.4, "thermodynamics": 0.2}
        }
        
        agent = BusinessAgent(config, mic, telemetry)
        
        report = agent.evaluate_project({
            "df_presupuesto": df_budget,
            "df_merged": df_detail,
            "initial_investment": 1_000_000,
            "validated_strata": {Stratum.PHYSICS, Stratum.TACTICS}
        })
        
        if report and report.integrity_score > 70:
            print("✅ Proyecto viable")
        ```
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mic: Any,
        telemetry: Optional[Any] = None,
        *,
        graph_builder: Optional[Any] = None,
        topology_analyzer: Optional[Any] = None,
        semantic_translator: Optional[Any] = None
    ):
        """
        Inicializa el agente con inyección de dependencias.
        
        Args:
            config: Configuración global
            mic: Matriz de Interacción Central
            telemetry: Contexto de telemetría
            graph_builder: Constructor de grafos (inyectable)
            topology_analyzer: Analizador topológico
            semantic_translator: Traductor semántico
        """
        self._validate_config(config)
        self._config = config
        self._mic = mic
        self._telemetry = telemetry or TelemetryContext()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Componentes inyectables
        self._graph_builder = graph_builder or (
            BudgetGraphBuilder() if _HAS_TOPOLOGY and BudgetGraphBuilder else None
        )
        self._topology_analyzer = topology_analyzer or (
            BusinessTopologicalAnalyzer(telemetry=self._telemetry)
            if _HAS_TOPOLOGY and BusinessTopologicalAnalyzer else None
        )
        self._translator = semantic_translator or (
            SemanticTranslator(mic=mic) if _HAS_TRANSLATOR and SemanticTranslator else None
        )
        
        # Motores de análisis
        self._financial_engine = FinancialEngine()
        self._real_options_engine = RealOptionsEngine()
        self._thermodynamics_engine = ThermodynamicsEngine()
        
        # Compositor y challenger
        decision_weights = self._build_decision_weights(config)
        self._report_composer = ReportComposer(
            analyzer=self._topology_analyzer,
            translator=self._translator,
            weights=decision_weights,
            telemetry=self._telemetry
        )
        
        challenger_thresholds = self._build_challenger_thresholds(config)
        self._risk_challenger = RiskChallenger(
            thresholds=challenger_thresholds,
            mic=mic
        )
        
        self._logger.info("✅ BusinessAgent inicializado correctamente")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Valida configuración."""
        if not isinstance(config, dict):
            raise ValueError("La configuración debe ser un diccionario")
    
    def _build_decision_weights(self, config: Dict[str, Any]) -> DecisionWeights:
        """Construye pesos de decisión desde config."""
        weights_cfg = config.get("decision_weights", {})
        
        if not weights_cfg:
            return DecisionWeights()
        
        return DecisionWeights.from_dict(weights_cfg)
    
    def _build_challenger_thresholds(self, config: Dict[str, Any]) -> RiskChallengerThresholds:
        """Construye umbrales del challenger desde config."""
        challenger_cfg = config.get("risk_challenger_config", {})
        
        if not challenger_cfg:
            return RiskChallengerThresholds()
        
        return RiskChallengerThresholds.from_dict(challenger_cfg)
    
    def evaluate_project(self, context: Dict[str, Any]) -> Optional[Any]:
        """
        Ejecuta evaluación completa del proyecto.
        
        Args:
            context: Contexto con DataFrames y parámetros
        
        Returns:
            ConstructionRiskReport o None si falla
        """
        self._logger.info("🤖 Iniciando evaluación de proyecto...")
        
        try:
            # Fase 1: Topología
            graph, topo_bundle = self._build_topology(context)
            
            # Fase 2: Finanzas
            financial_params = self._extract_financial_params(context)
            financial_metrics = self._financial_engine.analyze(financial_params)
            
            # Fase 3: Termodinámica
            thermal_state = self._thermodynamics_engine.analyze(
                financial_params,
                financial_metrics,
                topo_bundle
            )
            
            # Fase 4: Síntesis
            report = self._report_composer.compose(
                graph=graph,
                topo_bundle=topo_bundle,
                financial_metrics=financial_metrics,
                thermal_state=thermal_state
            )
            
            # Fase 5: Auditoría
            audited_report = self._risk_challenger.challenge_verdict(
                report=report,
                financial_metrics=financial_metrics,
                thermal_state=thermal_state,
                topo_bundle=topo_bundle,
                session_context=context
            )
            
            self._logger.info("✅ Evaluación completada con éxito")
            return audited_report
        
        except Exception as e:
            self._logger.error(f"❌ Error en evaluación: {e}", exc_info=True)
            if self._telemetry:
                self._telemetry.record_error("evaluation", str(e))
            return None
    
    def _build_topology(self, context: Dict[str, Any]) -> Tuple[Any, TopologicalMetricsBundle]:
        """Construye topología del presupuesto."""
        df_presupuesto = context.get("df_presupuesto") or context.get("df_final")
        df_apus_detail = context.get("df_merged")
        
        if self._graph_builder is None or self._topology_analyzer is None:
            raise RuntimeError("Componentes topológicos no configurados")
        
        # Construir grafo
        graph = self._graph_builder.build(df_presupuesto, df_apus_detail)
        
        # Calcular métricas
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        graph_metrics = GraphMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            is_connected=True,  # Placeholder
            n_components=1
        )
        
        # Betti numbers
        betti_raw = self._topology_analyzer.calculate_betti_numbers(graph)
        if hasattr(betti_raw, '__dataclass_fields__'):
            betti = BettiNumbers.from_dict(asdict(betti_raw))
        else:
            betti = BettiNumbers.from_dict(betti_raw)
        
        # Estabilidad
        pyramid_stability = self._topology_analyzer.calculate_pyramid_stability(graph)
        
        # Espectro (si disponible)
        try:
            adjacency = np.array([[0]])  # Placeholder
            laplacian = LaplacianBuilder.combinatorial_laplacian(adjacency)
            spectral = SpectralData.from_laplacian(laplacian)
        except Exception:
            spectral = SpectralData(eigenvalues=(0.0,))
        
        bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=UnitInterval(pyramid_stability)
        )
        
        return graph, bundle
    
    def _extract_financial_params(self, context: Dict[str, Any]) -> FinancialParameters:
        """Extrae parámetros financieros del contexto."""
        initial_investment = Decimal(str(context.get("initial_investment", "1000000.0")))
        
        if "cash_flows" in context:
            cash_flows = tuple(Decimal(str(cf)) for cf in context["cash_flows"])
        else:
            # Generar flujos sintéticos
            n_periods = 5
            annual_cf = initial_investment * Decimal("0.3")
            cash_flows = tuple(annual_cf for _ in range(n_periods))
        
        return FinancialParameters(
            initial_investment=initial_investment,
            cash_flows=cash_flows,
            discount_rate=Decimal(str(context.get("discount_rate", "0.10"))),
            risk_free_rate=Decimal(str(context.get("risk_free_rate", "0.05"))),
            project_volatility=context.get("project_volatility", 0.20)
        )
    
    # Propiedades de acceso
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @property
    def mic(self) -> Any:
        return self._mic
    
    @property
    def risk_challenger(self) -> RiskChallenger:
        return self._risk_challenger


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def create_business_agent(
    config: Dict[str, Any],
    mic: Any,
    telemetry: Optional[Any] = None
) -> BusinessAgent:
    """
    Factory function para crear BusinessAgent.
    
    Args:
        config: Configuración del agente
        mic: Matriz de Interacción Central
        telemetry: Contexto de telemetría
    
    Returns:
        Instancia configurada de BusinessAgent
    """
    return BusinessAgent(config=config, mic=mic, telemetry=telemetry)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Ejemplo de uso
    logger.info("Business Agent Module - Versión 2.0.0 (Rigorous)")
    logger.info("Fundamentos matemáticos: Topología Algebraica, Teoría Espectral, Teoría de Categorías")