"""
=========================================================================================
Módulo: Topological Analyzer (Operador de Observabilidad Funtorial y TDA)
Ubicación: app/tactics/topological_analyzer.py
Versión: 2.0.0-rigorous
=========================================================================================

Fundamentación Matemática Rigurosa:

1. TEORÍA DE CATEGORÍAS Y FUNTORES
   --------------------------------
   El sistema se modela como un funtor F: ℂ_Temp → ℂ_Topo donde:
   - ℂ_Temp: Categoría de series temporales (objetos = métricas, morfismos = transformaciones)
   - ℂ_Topo: Categoría de espacios topológicos (objetos = complejos simpliciales, morfismos = maps continuas)
   
   Propiedad Funtorial:
   F(g ∘ f) = F(g) ∘ F(f)  ∀ morfismos f, g
   F(id_X) = id_{F(X)}

2. HOMOLOGÍA PERSISTENTE (TDA - TOPOLOGICAL DATA ANALYSIS)
   --------------------------------------------------------
   Para una filtración de complejos simpliciales K_ε parametrizada por ε ∈ ℝ⁺:
   
   K_ε₁ ⊆ K_ε₂ ⊆ ... ⊆ K_εₙ  cuando ε₁ ≤ ε₂ ≤ ... ≤ εₙ
   
   La homología persistente estudia:
   H_k(K_ε; 𝔽) = ker(∂_k) / im(∂_{k+1})
   
   donde:
   - ∂_k: operador frontera del complejo de cadenas
   - 𝔽: campo de coeficientes (típicamente ℤ₂ o ℝ)
   - H_k: k-ésimo grupo de homología

   Teorema (Estabilidad de Diagramas de Persistencia):
   Para funciones f, g: X → ℝ en espacio métrico (X, d_X),
   d_B(Dgm(f), Dgm(g)) ≤ ‖f - g‖_∞
   
   donde d_B es la distancia de Bottleneck.

3. NÚMEROS DE BETTI Y CARACTERÍSTICA DE EULER
   -------------------------------------------
   Para un complejo simplicial K de dimensión d:
   
   β_k = dim H_k(K; 𝔽) = rank(Z_k) - rank(B_k)
   
   Fórmula de Euler-Poincaré:
   χ(K) = Σ_{k=0}^d (-1)^k n_k = Σ_{k=0}^d (-1)^k β_k
   
   Para grafos: χ = |V| - |E| = β₀ - β₁

4. TEORÍA ESPECTRAL DEL LAPLACIANO DE GRAFOS
   ------------------------------------------
   Laplaciano combinatorio: Λ = D - A
   Laplaciano normalizado: ℒ = I - D⁻¹/²AD⁻¹/²
   
   Propiedades espectrales:
   - 0 = λ₀ ≤ λ₁ ≤ ... ≤ λ_{n-1} ≤ 2  (normalizado)
   - λ₁ > 0 ⟺ grafo conexo (valor de Fiedler)
   - λ₁ mide la conectividad algebraica

5. DISTANCIAS EN ESPACIOS DE PERSISTENCIA
   ---------------------------------------
   Distancia de Wasserstein-p:
   W_p(μ, ν) = (inf_{γ∈Γ(μ,ν)} ∫∫ d(x,y)^p dγ(x,y))^{1/p}
   
   Distancia de Bottleneck:
   d_B(X, Y) = inf_{γ: X→Y} sup_{x∈X} ‖x - γ(x)‖_∞

=========================================================================================
"""

from __future__ import annotations

import abc
import functools
import hashlib
import logging
import math
import numbers
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from decimal import Decimal, getcontext
from enum import Enum, IntEnum, auto, unique
from fractions import Fraction
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    final,
    runtime_checkable,
)

import networkx as nx
import numpy as np
import numpy.typing as npt
from numpy import ndarray

# Configurar precisión Decimal para cálculos críticos
getcontext().prec = 50

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

logger = logging.getLogger("TopologicalAnalyzer")

# Suprimir warnings de NetworkX para operaciones esperadas
warnings.filterwarnings("ignore", category=UserWarning, module="networkx")

# =============================================================================
# CONSTANTES MATEMÁTICAS FUNDAMENTALES
# =============================================================================

@final
class TopologicalConstants:
    """
    Constantes matemáticas para análisis topológico.
    
    Todas las constantes están derivadas de principios fundamentales
    de topología algebraica y teoría de grafos.
    """
    
    # Constantes topológicas
    EULER_SPHERE: Final[int] = 2                    # χ(S²) = 2
    EULER_TORUS: Final[int] = 0                     # χ(T²) = 0
    EULER_PROJECTIVE_PLANE: Final[int] = 1          # χ(ℝP²) = 1
    EULER_KLEIN_BOTTLE: Final[int] = 0              # χ(K²) = 0
    
    # Umbrales numéricos
    EPSILON: Final[float] = 1e-10                   # Tolerancia estándar
    EPSILON_STRICT: Final[float] = 1e-14            # Tolerancia estricta
    
    # Límites de persistencia
    MIN_PERSISTENCE_RATIO: Final[float] = 0.01      # 1% de la ventana
    NOISE_THRESHOLD_RATIO: Final[float] = 0.20      # 20% para clasificar ruido
    CRITICAL_THRESHOLD_RATIO: Final[float] = 0.50   # 50% para estado crítico
    
    # Límites de complejidad ciclomática
    MAX_CYCLOMATIC_COMPLEXITY: Final[int] = 10      # Complejidad máxima saludable
    WARNING_CYCLOMATIC_COMPLEXITY: Final[int] = 5   # Umbral de advertencia
    
    # Límites de componentes conexas
    MAX_COMPONENTS_HEALTHY: Final[int] = 1          # Ideal: grafo conexo
    MAX_COMPONENTS_WARNING: Final[int] = 2          # Advertencia
    
    # Pesos para modelo de salud
    WEIGHT_FRAGMENTATION: Final[float] = 0.35       # Peso por fragmentación
    WEIGHT_CYCLES: Final[float] = 0.20              # Peso por ciclos
    WEIGHT_DISCONNECTED: Final[float] = 0.25        # Peso por nodos desconectados
    WEIGHT_MISSING_EDGES: Final[float] = 0.15       # Peso por aristas faltantes
    WEIGHT_RETRY_LOOPS: Final[float] = 0.05         # Peso por bucles de reintento
    
    @classmethod
    def validate_weights(cls) -> bool:
        """Valida que los pesos del modelo de salud sumen 1.0."""
        total = (
            cls.WEIGHT_FRAGMENTATION +
            cls.WEIGHT_CYCLES +
            cls.WEIGHT_DISCONNECTED +
            cls.WEIGHT_MISSING_EDGES +
            cls.WEIGHT_RETRY_LOOPS
        )
        return abs(total - 1.0) < cls.EPSILON


# Validar constantes al cargar el módulo
assert TopologicalConstants.validate_weights(), "Pesos del modelo no suman 1.0"

TC = TopologicalConstants  # Alias para brevedad

# =============================================================================
# TIPOS REFINADOS Y NEWTYPE WRAPPERS
# =============================================================================

# Tipos topológicos
BettiIndex = int  # Índice de número de Betti (0, 1, 2, ...)
SimplexDimension = int  # Dimensión de símplex (≥ 0)
EulerCharacteristic = int  # Característica de Euler (∈ ℤ)

# Tipos de persistencia
PersistenceValue = float  # Valor de persistencia (≥ 0)
BirthTime = int  # Tiempo de nacimiento (índice en serie temporal)
DeathTime = int  # Tiempo de muerte (-1 si aún vivo)

# Tipos de salud
HealthScore = float  # Score de salud ∈ [0, 1]
PenaltyFactor = float  # Factor de penalización ∈ [0, 1]

# Type aliases
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
AdjacencyDict = Dict[str, Dict[str, int]]

# =============================================================================
# ENUMERACIONES
# =============================================================================

@unique
class MetricState(Enum):
    """
    Estados posibles de una métrica según análisis de persistencia.
    
    Clasificación basada en teoría de homología persistente y
    análisis de estabilidad estructural.
    """
    
    STABLE = auto()      # Bajo control, sin excursiones significativas
    NOISE = auto()       # Excursión breve (muerte rápida en diagrama)
    FEATURE = auto()     # Patrón estructural (vida larga, persistencia alta)
    CRITICAL = auto()    # Estado crítico persistente y activo
    UNKNOWN = auto()     # Datos insuficientes para clasificar
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def severity(self) -> int:
        """Severidad del estado (0=mejor, 4=peor)."""
        return {
            MetricState.STABLE: 0,
            MetricState.NOISE: 1,
            MetricState.FEATURE: 2,
            MetricState.CRITICAL: 3,
            MetricState.UNKNOWN: 4
        }[self]


@unique
class HealthLevel(Enum):
    """
    Niveles de salud del sistema con umbrales matemáticos.
    
    Basado en análisis multicriterio de invariantes topológicos.
    """
    
    HEALTHY = auto()     # Sistema operando óptimamente
    DEGRADED = auto()    # Degradación parcial pero funcional
    UNHEALTHY = auto()   # Problemas significativos
    CRITICAL = auto()    # Fallo inminente o activo
    
    @classmethod
    def from_score(cls, score: float) -> 'HealthLevel':
        """
        Determina nivel de salud desde score numérico.
        
        Args:
            score: Score de salud ∈ [0, 1]
        
        Returns:
            Nivel de salud correspondiente
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score debe estar en [0, 1]: {score}")
        
        if score >= 0.90:
            return cls.HEALTHY
        elif score >= 0.70:
            return cls.DEGRADED
        elif score >= 0.40:
            return cls.UNHEALTHY
        else:
            return cls.CRITICAL
    
    @property
    def severity(self) -> int:
        """Severidad del nivel (0=mejor, 3=peor)."""
        return {
            HealthLevel.HEALTHY: 0,
            HealthLevel.DEGRADED: 1,
            HealthLevel.UNHEALTHY: 2,
            HealthLevel.CRITICAL: 3
        }[self]
    
    def __str__(self) -> str:
        return self.name


# =============================================================================
# EXCEPCIONES ESPECIALIZADAS
# =============================================================================

class TopologicalError(Exception):
    """Clase base para errores topológicos."""
    
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


class BettiNumberError(TopologicalError):
    """Error en cálculo de números de Betti."""
    pass


class PersistenceComputationError(TopologicalError):
    """Error en cálculo de homología persistente."""
    pass


class GraphStructureError(TopologicalError):
    """Error en estructura del grafo."""
    pass


class InvalidTopologyError(TopologicalError):
    """Topología inválida o inconsistente."""
    pass


# =============================================================================
# PROTOCOLOS TOPOLÓGICOS
# =============================================================================

@runtime_checkable
class SimplexProtocol(Protocol):
    """Protocolo para objetos que representan símplices."""
    
    @property
    def dimension(self) -> SimplexDimension:
        """Dimensión del símplex."""
        ...
    
    @property
    def vertices(self) -> FrozenSet[str]:
        """Conjunto de vértices."""
        ...


@runtime_checkable
class PersistentFeatureProtocol(Protocol):
    """Protocolo para características topológicas persistentes."""
    
    @property
    def birth(self) -> BirthTime:
        """Tiempo de nacimiento."""
        ...
    
    @property
    def death(self) -> DeathTime:
        """Tiempo de muerte."""
        ...
    
    @property
    def persistence(self) -> PersistenceValue:
        """Persistencia (muerte - nacimiento)."""
        ...
    
    @property
    def is_alive(self) -> bool:
        """Si la característica sigue viva."""
        ...


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti del sistema con validación rigurosa.
    
    Invariantes topológicos que caracterizan la "forma" del grafo de servicios.
    
    Teorema (Euler-Poincaré para Grafos):
        Para un grafo G = (V, E) con β₀ componentes y β₁ ciclos:
        χ(G) = |V| - |E| = β₀ - β₁
    
    Attributes:
        b0: β₀ - Número de componentes conexas
        b1: β₁ - Número de ciclos independientes (rango del grupo de ciclos)
        num_vertices: |V| - Número de vértices
        num_edges: |E| - Número de aristas
    """
    
    b0: BettiIndex
    b1: BettiIndex
    num_vertices: int = 0
    num_edges: int = 0
    
    def __post_init__(self) -> None:
        """
        Validación rigurosa de invariantes topológicos.
        
        Verifica:
        1. Números de Betti no negativos
        2. Consistencia con fórmula de Euler-Poincaré
        3. Cotas superiores (β₀ ≤ |V|, β₁ ≤ combinación de aristas)
        
        Raises:
            BettiNumberError: Si los invariantes son inconsistentes
        """
        # Validación básica: no negatividad
        if self.b0 < 0:
            raise BettiNumberError(
                f"β₀ debe ser no negativo: {self.b0}",
                context={"betti_numbers": self.__dict__}
            )
        
        if self.b1 < 0:
            raise BettiNumberError(
                f"β₁ debe ser no negativo: {self.b1}",
                context={"betti_numbers": self.__dict__}
            )
        
        if self.num_vertices < 0:
            raise BettiNumberError(
                f"Número de vértices no puede ser negativo: {self.num_vertices}"
            )
        
        if self.num_edges < 0:
            raise BettiNumberError(
                f"Número de aristas no puede ser negativo: {self.num_edges}"
            )
        
        # Validación: β₀ ≤ |V| (máximo una componente por vértice)
        if self.num_vertices > 0 and self.b0 > self.num_vertices:
            raise BettiNumberError(
                f"β₀ ({self.b0}) no puede exceder |V| ({self.num_vertices})",
                context={"b0": self.b0, "vertices": self.num_vertices}
            )
        
        # Validación de Euler-Poincaré para grafos simples
        if self.num_vertices > 0 and self.num_edges >= 0:
            # β₁ = |E| - |V| + β₀
            expected_b1 = self.num_edges - self.num_vertices + self.b0
            
            # Permitir pequeña discrepancia por errores de redondeo
            if expected_b1 >= 0 and abs(self.b1 - expected_b1) > 0:
                # Discrepancia indica posible multi-grafo o error
                if self.b1 != expected_b1:
                    raise BettiNumberError(
                        f"Violación de Euler-Poincaré: β₁={self.b1}, "
                        f"esperado={expected_b1} (|E|={self.num_edges}, "
                        f"|V|={self.num_vertices}, β₀={self.b0})",
                        context={
                            "actual_b1": self.b1,
                            "expected_b1": expected_b1,
                            "euler_deficit": self.b1 - expected_b1
                        }
                    )
        
        # Validación: β₁ ≤ |E| (máximo un ciclo por arista)
        if self.num_edges > 0 and self.b1 > self.num_edges:
            raise BettiNumberError(
                f"β₁ ({self.b1}) no puede exceder |E| ({self.num_edges})",
                context={"b1": self.b1, "edges": self.num_edges}
            )
    
    @property
    def is_connected(self) -> bool:
        """
        Sistema completamente conectado (un solo componente).
        
        Returns:
            True si β₀ = 1
        """
        return self.b0 == 1
    
    @property
    def is_acyclic(self) -> bool:
        """
        Grafo sin ciclos (árbol o bosque).
        
        Returns:
            True si β₁ = 0
        """
        return self.b1 == 0
    
    @property
    def is_ideal(self) -> bool:
        """
        Estado ideal: conectado y acíclico (árbol).
        
        Returns:
            True si β₀ = 1 y β₁ = 0
        """
        return self.is_connected and self.is_acyclic
    
    @property
    def is_tree(self) -> bool:
        """
        Alias de is_ideal para claridad matemática.
        
        Un árbol es un grafo conexo y acíclico.
        
        Returns:
            True si el grafo es un árbol
        """
        return self.is_ideal and self.num_vertices > 0 and self.num_edges == self.num_vertices - 1
    
    @property
    def is_forest(self) -> bool:
        """
        Grafo acíclico (posiblemente no conexo).
        
        Returns:
            True si β₁ = 0
        """
        return self.is_acyclic
    
    @property
    def euler_characteristic(self) -> EulerCharacteristic:
        """
        Característica de Euler: χ = β₀ - β₁ = |V| - |E|.
        
        Teorema (Euler-Poincaré):
            χ(G) = Σ_{k=0}^∞ (-1)^k β_k
            Para grafos: χ = β₀ - β₁
        
        Returns:
            Característica de Euler del grafo
        """
        return self.b0 - self.b1
    
    @property
    def euler_characteristic_alt(self) -> EulerCharacteristic:
        """
        Cálculo alternativo: χ = |V| - |E|.
        
        Debe coincidir con euler_characteristic por invariancia.
        
        Returns:
            Característica de Euler calculada desde vértices/aristas
        """
        return self.num_vertices - self.num_edges
    
    @property
    def cyclomatic_complexity(self) -> int:
        """
        Complejidad ciclomática: M = β₁ + β₀.
        
        Generalización de la métrica de McCabe para grafos.
        
        Returns:
            Complejidad ciclomática del grafo
        """
        return self.b1 + self.b0
    
    @property
    def connectivity_index(self) -> float:
        """
        Índice de conectividad normalizado: (|E| + β₀) / (|V| + 1).
        
        Mide qué tan densamente conectado está el grafo.
        
        Returns:
            Índice ∈ [0, ∞), típicamente ∈ [0, 2]
        """
        if self.num_vertices == 0:
            return 0.0
        return (self.num_edges + self.b0) / (self.num_vertices + 1)
    
    def verify_euler_consistency(self) -> bool:
        """
        Verifica consistencia de la característica de Euler.
        
        Compara ambas fórmulas: χ = β₀ - β₁ = |V| - |E|
        
        Returns:
            True si las fórmulas coinciden
        """
        return self.euler_characteristic == self.euler_characteristic_alt
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "b0": self.b0,
            "b1": self.b1,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "euler_characteristic": self.euler_characteristic,
            "is_connected": self.is_connected,
            "is_acyclic": self.is_acyclic,
            "is_ideal": self.is_ideal,
            "is_tree": self.is_tree,
            "is_forest": self.is_forest,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "connectivity_index": self.connectivity_index,
            "euler_consistent": self.verify_euler_consistency()
        }
    
    def __str__(self) -> str:
        status = "✓" if self.is_ideal else "⚠"
        return (
            f"BettiNumbers({status} β₀={self.b0}, β₁={self.b1}, "
            f"χ={self.euler_characteristic}, |V|={self.num_vertices}, |E|={self.num_edges})"
        )
    
    def __repr__(self) -> str:
        return (
            f"BettiNumbers(b0={self.b0}, b1={self.b1}, "
            f"num_vertices={self.num_vertices}, num_edges={self.num_edges})"
        )


@final
@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """
    Intervalo de nacimiento-muerte para homología persistente.
    
    Representa un punto (birth, death) en el diagrama de persistencia.
    La distancia perpendicular a la diagonal mide la persistencia.
    
    Teorema (Interpretación Geométrica):
        persistence = (death - birth) / √2
        es la distancia perpendicular del punto a la diagonal birth=death.
    
    Attributes:
        birth: Índice de nacimiento en la serie temporal
        death: Índice de muerte (-1 si aún vive)
        dimension: Dimensión de la característica topológica
        amplitude: Amplitud máxima durante la vida
    """
    
    birth: BirthTime
    death: DeathTime
    dimension: SimplexDimension
    amplitude: float = 0.0
    
    def __post_init__(self) -> None:
        """
        Validación de invariantes del intervalo.
        
        Raises:
            PersistenceComputationError: Si el intervalo es inválido
        """
        if self.birth < 0:
            raise PersistenceComputationError(
                f"Tiempo de nacimiento no puede ser negativo: {self.birth}"
            )
        
        if self.death != -1 and self.death < self.birth:
            raise PersistenceComputationError(
                f"Muerte debe ser >= nacimiento: birth={self.birth}, death={self.death}"
            )
        
        if self.dimension < 0:
            raise PersistenceComputationError(
                f"Dimensión no puede ser negativa: {self.dimension}"
            )
        
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise PersistenceComputationError(
                f"Amplitud debe ser finita y no negativa: {self.amplitude}"
            )
    
    @property
    def lifespan(self) -> float:
        """
        Duración de vida de la característica.
        
        Returns:
            death - birth si finito, infinito si aún vive
        """
        if self.is_alive:
            return float('inf')
        return float(self.death - self.birth)
    
    @property
    def is_alive(self) -> bool:
        """
        ¿La característica sigue viva?
        
        Returns:
            True si death < 0
        """
        return self.death < 0
    
    @property
    def persistence(self) -> PersistenceValue:
        """
        Persistencia normalizada (distancia perpendicular a diagonal).
        
        Fórmula:
            p = (death - birth) / √2
        
        Interpretación geométrica:
            Distancia del punto (birth, death) a la diagonal en norma L².
        
        Returns:
            Persistencia ∈ [0, ∞)
        """
        if self.is_alive:
            return float('inf')
        return (self.death - self.birth) / math.sqrt(2.0)
    
    @property
    def midpoint(self) -> float:
        """
        Punto medio del intervalo.
        
        Returns:
            (birth + death) / 2 si finito, birth si vivo
        """
        if self.is_alive:
            return float(self.birth)
        return (self.birth + self.death) / 2.0
    
    def bottleneck_distance(self, other: 'PersistenceInterval') -> float:
        """
        Distancia de Bottleneck entre dos intervalos.
        
        Fórmula:
            d_B(I₁, I₂) = max(|b₁ - b₂|, |d₁ - d₂|)
        
        Args:
            other: Otro intervalo de persistencia
        
        Returns:
            Distancia de Bottleneck (métrica en espacio de diagramas)
        """
        if self.dimension != other.dimension:
            return float('inf')
        
        birth_diff = abs(self.birth - other.birth)
        
        if self.is_alive and other.is_alive:
            death_diff = 0.0
        elif self.is_alive or other.is_alive:
            death_diff = float('inf')
        else:
            death_diff = abs(self.death - other.death)
        
        return max(birth_diff, death_diff)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "birth": self.birth,
            "death": self.death,
            "dimension": self.dimension,
            "amplitude": self.amplitude,
            "lifespan": self.lifespan if math.isfinite(self.lifespan) else None,
            "persistence": self.persistence if math.isfinite(self.persistence) else None,
            "is_alive": self.is_alive,
            "midpoint": self.midpoint
        }
    
    def __str__(self) -> str:
        death_str = "∞" if self.is_alive else str(self.death)
        pers_str = "∞" if math.isinf(self.persistence) else f"{self.persistence:.4f}"
        return (
            f"PersistenceInterval(dim={self.dimension}, [{self.birth}, {death_str}), "
            f"pers={pers_str}, amp={self.amplitude:.4f})"
        )


@final
@dataclass(frozen=True, slots=True)
class RequestLoopInfo:
    """
    Información sobre bucles de reintentos detectados.
    
    Attributes:
        request_id: Identificador del request
        count: Número de ocurrencias
        first_seen: Índice de primera ocurrencia
        last_seen: Índice de última ocurrencia
    """
    
    request_id: str
    count: int
    first_seen: BirthTime
    last_seen: int
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if self.count < 0:
            raise ValueError(f"count no puede ser negativo: {self.count}")
        
        if self.first_seen < 0 or self.last_seen < 0:
            raise ValueError("Índices no pueden ser negativos")
        
        if self.last_seen < self.first_seen:
            raise ValueError(
                f"last_seen ({self.last_seen}) < first_seen ({self.first_seen})"
            )
    
    @property
    def duration(self) -> int:
        """Duración del bucle (last - first)."""
        return self.last_seen - self.first_seen
    
    @property
    def frequency(self) -> float:
        """Frecuencia de reintentos (count / duration)."""
        if self.duration == 0:
            return float('inf') if self.count > 1 else 0.0
        return self.count / self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "request_id": self.request_id,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": self.duration,
            "frequency": self.frequency
        }


# =============================================================================
# FASE 2: MODELO DE SALUD TOPOLÓGICA Y ANÁLISIS DE GRAFOS
# =============================================================================

@final
@dataclass(frozen=True, slots=True)
class TopologicalHealth:
    """
    Resumen completo de salud topológica del sistema.
    
    Implementa un modelo de penalizaciones normalizado basado en
    invariantes topológicos y teoría de grafos.
    
    Modelo Matemático:
        Score = 1.0 - Σ(penalización_i × peso_i)
    
    donde:
        - penalización_i ∈ [0, 1] (normalizada)
        - peso_i ∈ [0, 1] con Σpesos_i = 1.0
    
    Attributes:
        betti: Números de Betti del sistema
        disconnected_nodes: Nodos requeridos sin conexiones
        missing_edges: Conexiones esperadas no presentes
        request_loops: Bucles de reintento detectados
        health_score: Score de salud ∈ [0, 1]
        level: Nivel de salud categórico
        diagnostics: Diccionario con detalles de diagnóstico
    """
    
    betti: BettiNumbers
    disconnected_nodes: FrozenSet[str]
    missing_edges: FrozenSet[Tuple[str, str]]
    request_loops: Tuple[RequestLoopInfo, ...]
    health_score: HealthScore
    level: HealthLevel
    diagnostics: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validación de invariantes del modelo de salud."""
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError(
                f"health_score debe estar en [0, 1]: {self.health_score}"
            )
        
        # Verificar consistencia entre score y level
        expected_level = HealthLevel.from_score(self.health_score)
        if self.level != expected_level:
            logger.warning(
                f"Inconsistencia: level={self.level.name} pero score={self.health_score:.3f} "
                f"sugiere level={expected_level.name}"
            )
    
    @property
    def is_healthy(self) -> bool:
        """Sistema saludable (nivel HEALTHY)."""
        return self.level == HealthLevel.HEALTHY
    
    @property
    def is_critical(self) -> bool:
        """Sistema en estado crítico."""
        return self.level == HealthLevel.CRITICAL
    
    @property
    def has_fragmentation(self) -> bool:
        """Sistema fragmentado (β₀ > 1)."""
        return self.betti.b0 > 1
    
    @property
    def has_cycles(self) -> bool:
        """Sistema tiene ciclos (β₁ > 0)."""
        return self.betti.b1 > 0
    
    @property
    def has_disconnected_nodes(self) -> bool:
        """Hay nodos requeridos desconectados."""
        return len(self.disconnected_nodes) > 0
    
    @property
    def has_missing_edges(self) -> bool:
        """Hay conexiones esperadas faltantes."""
        return len(self.missing_edges) > 0
    
    @property
    def has_loops(self) -> bool:
        """Hay bucles de reintento detectados."""
        return len(self.request_loops) > 0
    
    @property
    def total_anomalies(self) -> int:
        """Número total de anomalías detectadas."""
        count = 0
        if self.has_fragmentation:
            count += self.betti.b0 - 1  # Componentes adicionales
        if self.has_cycles:
            count += self.betti.b1  # Ciclos
        count += len(self.disconnected_nodes)
        count += len(self.missing_edges)
        count += len(self.request_loops)
        return count
    
    def get_summary(self) -> str:
        """
        Genera resumen textual del estado de salud.
        
        Returns:
            String con resumen ejecutivo
        """
        lines = [
            f"=== SALUD TOPOLÓGICA DEL SISTEMA ===",
            f"Nivel: {self.level.name}",
            f"Score: {self.health_score:.3f}/1.000",
            f"",
            f"Invariantes Topológicos:",
            f"  β₀ (componentes): {self.betti.b0}",
            f"  β₁ (ciclos): {self.betti.b1}",
            f"  χ (Euler): {self.betti.euler_characteristic}",
            f"  Complejidad ciclomática: {self.betti.cyclomatic_complexity}",
            f"",
            f"Anomalías Detectadas: {self.total_anomalies}"
        ]
        
        if self.disconnected_nodes:
            lines.append(f"  • Nodos desconectados: {len(self.disconnected_nodes)}")
        if self.missing_edges:
            lines.append(f"  • Conexiones faltantes: {len(self.missing_edges)}")
        if self.request_loops:
            lines.append(f"  • Bucles de reintento: {len(self.request_loops)}")
        
        if self.diagnostics:
            lines.append("")
            lines.append("Diagnósticos:")
            for key, value in self.diagnostics.items():
                lines.append(f"  • {key}: {value}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario completo."""
        return {
            "health_score": self.health_score,
            "level": self.level.name,
            "severity": self.level.severity,
            "is_healthy": self.is_healthy,
            "is_critical": self.is_critical,
            "betti_numbers": self.betti.to_dict(),
            "anomalies": {
                "fragmentation": self.has_fragmentation,
                "cycles": self.has_cycles,
                "disconnected_nodes": self.has_disconnected_nodes,
                "missing_edges": self.has_missing_edges,
                "loops": self.has_loops,
                "total": self.total_anomalies
            },
            "disconnected_nodes": sorted(self.disconnected_nodes),
            "missing_edges": sorted([tuple(sorted(e)) for e in self.missing_edges]),
            "request_loops": [loop.to_dict() for loop in self.request_loops],
            "diagnostics": self.diagnostics
        }
    
    def __str__(self) -> str:
        return (
            f"TopologicalHealth(level={self.level.name}, score={self.health_score:.3f}, "
            f"anomalies={self.total_anomalies})"
        )


@final
@dataclass(frozen=True, slots=True)
class PersistenceAnalysisResult:
    """
    Resultado completo del análisis de homología persistente.
    
    Clasifica el estado de una métrica basándose en su diagrama de persistencia
    y propiedades estadísticas de los intervalos.
    
    Attributes:
        state: Estado clasificado de la métrica
        intervals: Tupla inmutable de intervalos de persistencia
        feature_count: Número de características estructurales
        noise_count: Número de excursiones de ruido
        active_count: Número de características activas
        max_lifespan: Máxima duración de vida finita
        total_persistence: Suma de persistencias
        metadata: Diccionario con metadatos adicionales
    """
    
    state: MetricState
    intervals: Tuple[PersistenceInterval, ...]
    feature_count: int
    noise_count: int
    active_count: int
    max_lifespan: float
    total_persistence: float
    metadata: Dict[str, Union[int, float, str]]
    
    def __post_init__(self) -> None:
        """Validación de consistencia."""
        if self.feature_count < 0 or self.noise_count < 0 or self.active_count < 0:
            raise ValueError("Contadores no pueden ser negativos")
        
        total_intervals = len(self.intervals)
        counted = self.feature_count + self.noise_count + self.active_count
        
        # Los contadores pueden no sumar exactamente si hay intervalos no clasificados
        # pero no deben exceder el total
        if counted > total_intervals:
            raise ValueError(
                f"Suma de contadores ({counted}) excede total de intervalos ({total_intervals})"
            )
        
        if not math.isfinite(self.max_lifespan) and self.max_lifespan != float('inf'):
            raise ValueError(f"max_lifespan debe ser finito o infinito: {self.max_lifespan}")
        
        if self.total_persistence < 0 and math.isfinite(self.total_persistence):
            raise ValueError(f"total_persistence no puede ser negativo: {self.total_persistence}")
    
    @property
    def total_intervals(self) -> int:
        """Número total de intervalos."""
        return len(self.intervals)
    
    @property
    def confidence(self) -> float:
        """
        Nivel de confianza de la clasificación.
        
        Extraído de metadata si está disponible.
        
        Returns:
            Confianza ∈ [0, 1]
        """
        return float(self.metadata.get("confidence", 0.0))
    
    @property
    def reason(self) -> str:
        """Razón de la clasificación."""
        return str(self.metadata.get("reason", "unknown"))
    
    @property
    def is_stable(self) -> bool:
        """Estado es estable."""
        return self.state == MetricState.STABLE
    
    @property
    def is_critical(self) -> bool:
        """Estado es crítico."""
        return self.state == MetricState.CRITICAL
    
    @property
    def has_features(self) -> bool:
        """Hay características estructurales."""
        return self.feature_count > 0
    
    @property
    def average_persistence(self) -> float:
        """Persistencia promedio de intervalos finitos."""
        finite = [i for i in self.intervals if not i.is_alive]
        if not finite:
            return 0.0
        return sum(i.persistence for i in finite) / len(finite)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "state": self.state.name,
            "severity": self.state.severity,
            "total_intervals": self.total_intervals,
            "feature_count": self.feature_count,
            "noise_count": self.noise_count,
            "active_count": self.active_count,
            "max_lifespan": self.max_lifespan if math.isfinite(self.max_lifespan) else None,
            "total_persistence": self.total_persistence,
            "average_persistence": self.average_persistence,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "intervals": [i.to_dict() for i in self.intervals]
        }
    
    def __str__(self) -> str:
        return (
            f"PersistenceAnalysisResult(state={self.state.name}, "
            f"intervals={len(self.intervals)}, features={self.feature_count}, "
            f"confidence={self.confidence:.2f})"
        )


# =============================================================================
# CLASE PRINCIPAL: SystemTopology
# =============================================================================

class SystemTopology:
    """
    Representa el estado de los servicios como un espacio topológico (Grafo).
    
    Fundamentación Matemática:
    -------------------------
    Modelamos el sistema como un grafo no dirigido G = (V, E) donde:
    - V = conjunto de servicios (nodos)
    - E = conexiones activas entre servicios (aristas)
    
    Los números de Betti caracterizan la topología:
    - β₀ = número de componentes conexas
    - β₁ = |E| - |V| + β₀ (rango del ciclo del grafo)
    
    Para un sistema saludable típico:
    - β₀ = 1 (todo conectado)
    - β₁ = 0 (sin redundancias cíclicas, topología de árbol)
    
    Invariante de Clase:
    -------------------
    ∀ estado válido del grafo G:
        χ(G) = |V| - |E| = β₀ - β₁
    
    Esta invariante debe mantenerse en todo momento.
    """
    
    # Configuración de nodos requeridos (topología canónica)
    REQUIRED_NODES: ClassVar[FrozenSet[str]] = frozenset({
        "Agent",
        "Core",
        "Redis",
        "Filesystem"
    })
    
    # Topología esperada: árbol con Core como hub central
    EXPECTED_TOPOLOGY: ClassVar[FrozenSet[Tuple[str, str]]] = frozenset({
        # Plano de Datos (base de la pirámide)
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        # Plano de Control (cúspide - el Agente observa todo)
        ("Agent", "Core"),
        ("Agent", "Redis"),
        ("Agent", "Filesystem"),
    })
    
    # Límites de validación
    MIN_WINDOW_SIZE: ClassVar[int] = 3
    DEFAULT_WINDOW_SIZE: ClassVar[int] = 50
    MAX_WINDOW_SIZE: ClassVar[int] = 1000
    
    def __init__(
        self,
        max_history: int = DEFAULT_WINDOW_SIZE,
        custom_nodes: Optional[Set[str]] = None,
        custom_topology: Optional[Set[Tuple[str, str]]] = None,
        validate_strictly: bool = True
    ):
        """
        Inicializa la topología del sistema.
        
        Args:
            max_history: Tamaño máximo del historial de requests
            custom_nodes: Nodos adicionales a monitorear
            custom_topology: Topología esperada personalizada
            validate_strictly: Si True, valida rigurosamente todas las operaciones
        
        Raises:
            ValueError: Si max_history está fuera de rango válido
            GraphStructureError: Si la topología inicial es inválida
        """
        # Validar parámetros
        if not (self.MIN_WINDOW_SIZE <= max_history <= self.MAX_WINDOW_SIZE):
            raise ValueError(
                f"max_history debe estar en [{self.MIN_WINDOW_SIZE}, {self.MAX_WINDOW_SIZE}]: "
                f"{max_history}"
            )
        
        self._max_history = max_history
        self._validate_strictly = validate_strictly
        
        # Inicializar grafo NetworkX
        self._graph: nx.Graph = nx.Graph()
        
        # Configurar nodos requeridos
        all_nodes = set(self.REQUIRED_NODES)
        if custom_nodes:
            # Validar nodos personalizados
            invalid_nodes = {n for n in custom_nodes if not isinstance(n, str) or not n.strip()}
            if invalid_nodes:
                raise GraphStructureError(
                    f"Nodos personalizados inválidos: {invalid_nodes}"
                )
            all_nodes.update(n.strip() for n in custom_nodes)
        
        self._graph.add_nodes_from(all_nodes)
        
        # Configurar topología esperada
        self._expected_topology = set(self.EXPECTED_TOPOLOGY)
        if custom_topology:
            # Validar topología personalizada
            for edge in custom_topology:
                if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                    raise GraphStructureError(
                        f"Arista inválida en topología personalizada: {edge}"
                    )
                if not all(isinstance(n, str) and n.strip() for n in edge):
                    raise GraphStructureError(
                        f"Nodos de arista deben ser strings no vacíos: {edge}"
                    )
            self._expected_topology.update(custom_topology)
        
        # Historial de requests con timestamps relativos
        self._request_history: deque = deque(maxlen=max_history)
        self._request_index: int = 0
        
        # Cache de cálculos costosos
        self._betti_cache: Optional[Tuple[int, BettiNumbers]] = None  # (hash_graph, betti)
        self._health_cache: Optional[Tuple[int, TopologicalHealth]] = None
        
        logger.debug(
            f"SystemTopology inicializado: {len(all_nodes)} nodos, "
            f"historial máximo: {max_history}, validación estricta: {validate_strictly}"
        )
    
    # -------------------------------------------------------------------------
    # Propiedades del Grafo
    # -------------------------------------------------------------------------
    
    @property
    def nodes(self) -> Set[str]:
        """Conjunto de todos los nodos en el grafo."""
        return set(self._graph.nodes())
    
    @property
    def edges(self) -> Set[Tuple[str, str]]:
        """Conjunto de todas las aristas activas."""
        return set(self._graph.edges())
    
    @property
    def num_nodes(self) -> int:
        """Número de nodos: |V|."""
        return self._graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        """Número de aristas: |E|."""
        return self._graph.number_of_edges()
    
    @property
    def density(self) -> float:
        """
        Densidad del grafo: ρ = 2|E| / (|V|(|V|-1)).
        
        Para grafos no dirigidos simples.
        
        Returns:
            Densidad ∈ [0, 1]
        """
        n = self.num_nodes
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2.0
        return self.num_edges / max_edges if max_edges > 0 else 0.0
    
    def _compute_graph_hash(self) -> int:
        """
        Calcula hash del estado actual del grafo.
        
        Usado para invalidar caché cuando el grafo cambia.
        
        Returns:
            Hash determinista del grafo
        """
        # Crear representación canónica
        nodes_sorted = tuple(sorted(self._graph.nodes()))
        edges_sorted = tuple(sorted(tuple(sorted(e)) for e in self._graph.edges()))
        
        # Combinar en tupla y hashear
        state = (nodes_sorted, edges_sorted)
        return hash(state)
    
    def _invalidate_caches(self) -> None:
        """Invalida todos los cachés cuando el grafo cambia."""
        self._betti_cache = None
        self._health_cache = None
    
    # -------------------------------------------------------------------------
    # Gestión de Nodos
    # -------------------------------------------------------------------------
    
    def add_node(self, node: str) -> bool:
        """
        Agrega un nodo dinámico al grafo con validación rigurosa.
        
        Args:
            node: Nombre del servicio a agregar
        
        Returns:
            True si se agregó, False si ya existía o es inválido
        
        Raises:
            GraphStructureError: Si validate_strictly=True y el nodo es inválido
        """
        # Validación de tipo
        if not isinstance(node, str):
            msg = f"Nodo debe ser string, recibido: {type(node).__name__}"
            if self._validate_strictly:
                raise GraphStructureError(msg)
            logger.warning(msg)
            return False
        
        # Normalizar
        node_cleaned = node.strip()
        
        # Validación de contenido
        if not node_cleaned:
            msg = f"Nodo no puede ser vacío: {repr(node)}"
            if self._validate_strictly:
                raise GraphStructureError(msg)
            logger.warning(msg)
            return False
        
        # Validación de caracteres problemáticos
        invalid_chars = {'\x00', '\n', '\r', '\t'}
        if any(c in node_cleaned for c in invalid_chars):
            msg = f"Nodo contiene caracteres de control no permitidos: {repr(node)}"
            if self._validate_strictly:
                raise GraphStructureError(msg)
            logger.warning(msg)
            return False
        
        # Verificar si ya existe
        if node_cleaned in self._graph:
            logger.debug(f"Nodo ya existe, ignorando: {node_cleaned}")
            return False
        
        # Agregar nodo
        self._graph.add_node(node_cleaned)
        self._invalidate_caches()
        
        logger.debug(f"Nodo agregado: {node_cleaned}")
        return True
    
    def remove_node(self, node: str) -> bool:
        """
        Elimina un nodo del grafo (solo si no es requerido).
        
        Args:
            node: Nombre del servicio a eliminar
        
        Returns:
            True si se eliminó correctamente
        
        Raises:
            GraphStructureError: Si se intenta eliminar nodo requerido con validación estricta
        """
        # Validación de tipo
        if not isinstance(node, str):
            msg = f"Tipo inválido para remove_node: {type(node).__name__}"
            if self._validate_strictly:
                raise GraphStructureError(msg)
            logger.warning(msg)
            return False
        
        node = node.strip()
        
        if not node:
            return False
        
        # Verificar si es nodo requerido
        if node in self.REQUIRED_NODES:
            msg = f"No se puede eliminar nodo requerido: {node}"
            if self._validate_strictly:
                raise GraphStructureError(msg)
            logger.warning(msg)
            return False
        
        # Verificar existencia
        if node not in self._graph:
            logger.debug(f"Nodo no existe, nada que eliminar: {node}")
            return False
        
        # Registrar aristas perdidas para auditoría
        lost_edges = list(self._graph.edges(node))
        if lost_edges:
            logger.debug(
                f"Eliminando nodo {node} con {len(lost_edges)} aristas asociadas"
            )
        
        # Eliminar
        self._graph.remove_node(node)
        self._invalidate_caches()
        
        logger.debug(f"Nodo eliminado: {node}")
        return True
    
    def has_node(self, node: str) -> bool:
        """Verifica si un nodo existe en el grafo."""
        if not isinstance(node, str):
            return False
        return node in self._graph
    
    # -------------------------------------------------------------------------
    # Gestión de Conectividad
    # -------------------------------------------------------------------------
    
    def update_connectivity(
        self,
        active_connections: List[Tuple[str, str]],
        validate_nodes: bool = True,
        auto_add_nodes: bool = False
    ) -> Tuple[int, List[str]]:
        """
        Actualiza las conexiones del grafo con operación atómica.
        
        Garantiza que si hay errores críticos, el estado anterior se preserva.
        
        Algoritmo:
        1. Validar todas las aristas
        2. Guardar estado actual
        3. Actualizar grafo
        4. Si falla, hacer rollback
        
        Args:
            active_connections: Lista de pares (origen, destino) activos
            validate_nodes: Si True, valida que los nodos existan
            auto_add_nodes: Si True, agrega nodos faltantes automáticamente
        
        Returns:
            Tupla (edges_added, warnings) con número de aristas agregadas
            y lista de advertencias
        
        Raises:
            TypeError: Si active_connections no es iterable
            GraphStructureError: Si hay error crítico en modo estricto
        """
        # Validación de entrada
        if active_connections is None:
            logger.warning("active_connections es None, tratando como lista vacía")
            active_connections = []
        
        if not hasattr(active_connections, "__iter__"):
            raise TypeError(
                f"active_connections debe ser iterable, recibido: {type(active_connections).__name__}"
            )
        
        warnings: List[str] = []
        valid_edges: List[Tuple[str, str]] = []
        nodes_to_add: Set[str] = set()
        
        # Fase 1: Validación
        for idx, item in enumerate(active_connections):
            # Validar formato de arista
            if not isinstance(item, (tuple, list)):
                msg = f"[{idx}] Formato inválido (no tupla/lista): {repr(item)}"
                warnings.append(msg)
                continue
            
            if len(item) != 2:
                msg = f"[{idx}] Arista debe tener 2 elementos: {repr(item)}"
                warnings.append(msg)
                continue
            
            src, dst = item
            
            # Validar tipos
            if not isinstance(src, str) or not isinstance(dst, str):
                msg = (
                    f"[{idx}] Nodos deben ser strings: "
                    f"({type(src).__name__}, {type(dst).__name__})"
                )
                warnings.append(msg)
                continue
            
            src, dst = src.strip(), dst.strip()
            
            # Validar no vacíos
            if not src or not dst:
                msg = f"[{idx}] Nodos no pueden ser vacíos después de strip"
                warnings.append(msg)
                continue
            
            # Evitar auto-loops
            if src == dst:
                msg = f"[{idx}] Auto-loop ignorado: {src}"
                warnings.append(msg)
                continue
            
            # Validar existencia de nodos
            if validate_nodes:
                missing_nodes = []
                
                for node, role in [(src, "origen"), (dst, "destino")]:
                    if node not in self._graph:
                        if auto_add_nodes:
                            nodes_to_add.add(node)
                            warnings.append(f"[{idx}] Nodo {role} será agregado: {node}")
                        else:
                            missing_nodes.append(f"{role}={node}")
                
                if missing_nodes and not auto_add_nodes:
                    msg = f"[{idx}] Nodos faltantes: {', '.join(missing_nodes)}"
                    warnings.append(msg)
                    continue
            
            valid_edges.append((src, dst))
        
        # Fase 2: Commit atómico
        # Guardar estado previo para rollback
        previous_edges = list(self._graph.edges())
        previous_nodes = set(self._graph.nodes())
        
        try:
            # Agregar nodos nuevos primero
            for node in nodes_to_add:
                self._graph.add_node(node)
            
            # Actualizar aristas (reemplazar completamente)
            self._graph.clear_edges()
            self._graph.add_edges_from(valid_edges)
            
            # Invalidar cachés
            self._invalidate_caches()
        
        except Exception as e:
            # Rollback: restaurar estado anterior
            logger.error(f"Error durante actualización, ejecutando rollback: {e}")
            
            # Restaurar nodos
            self._graph.clear()
            self._graph.add_nodes_from(previous_nodes)
            
            # Restaurar aristas
            self._graph.add_edges_from(previous_edges)
            
            # Invalidar cachés por seguridad
            self._invalidate_caches()
            
            if self._validate_strictly:
                raise GraphStructureError(
                    f"Fallo en update_connectivity, estado restaurado: {e}"
                ) from e
            else:
                logger.warning(f"Estado restaurado después de error: {e}")
                return 0, warnings + [f"Error crítico: {e}"]
        
        # Log warnings
        for warn in warnings:
            logger.warning(warn)
        
        logger.debug(
            f"Conectividad actualizada: {len(valid_edges)} aristas activas, "
            f"{len(nodes_to_add)} nodos agregados, {len(warnings)} advertencias"
        )
        
        return len(valid_edges), warnings
    
    def add_edge(self, src: str, dst: str) -> bool:
        """
        Agrega una arista individual si los nodos existen.
        
        Args:
            src: Nodo origen
            dst: Nodo destino
        
        Returns:
            True si se agregó correctamente
        """
        # Validación de tipos
        if not isinstance(src, str) or not isinstance(dst, str):
            logger.warning(
                f"add_edge requiere strings: src={type(src).__name__}, dst={type(dst).__name__}"
            )
            return False
        
        src, dst = src.strip(), dst.strip()
        
        if not src or not dst:
            logger.warning("add_edge: nodos no pueden ser vacíos")
            return False
        
        if src not in self._graph:
            logger.debug(f"add_edge: nodo origen no existe: {src}")
            return False
        
        if dst not in self._graph:
            logger.debug(f"add_edge: nodo destino no existe: {dst}")
            return False
        
        if src == dst:
            logger.warning(f"add_edge: auto-loop no permitido: {src}")
            return False
        
        if self._graph.has_edge(src, dst):
            logger.debug(f"add_edge: arista ya existe: ({src}, {dst})")
            return False
        
        self._graph.add_edge(src, dst)
        self._invalidate_caches()
        
        logger.debug(f"Arista agregada: ({src}, {dst})")
        return True
    
    def remove_edge(self, src: str, dst: str) -> bool:
        """
        Elimina una arista si existe.
        
        Args:
            src: Nodo origen
            dst: Nodo destino
        
        Returns:
            True si se eliminó correctamente
        """
        # Validación de tipos
        if not isinstance(src, str) or not isinstance(dst, str):
            logger.warning(
                f"remove_edge requiere strings: src={type(src).__name__}, dst={type(dst).__name__}"
            )
            return False
        
        src, dst = src.strip(), dst.strip()
        
        if not src or not dst:
            return False
        
        if self._graph.has_edge(src, dst):
            self._graph.remove_edge(src, dst)
            self._invalidate_caches()
            logger.debug(f"Arista eliminada: ({src}, {dst})")
            return True
        
        logger.debug(f"remove_edge: arista no existe: ({src}, {dst})")
        return False
    
    # -------------------------------------------------------------------------
    # Registro de Requests
    # -------------------------------------------------------------------------
    
    def record_request(self, request_id: str) -> bool:
        """
        Registra un request_id para análisis de patrones de reintentos.
        
        Args:
            request_id: Identificador único del request
        
        Returns:
            True si se registró correctamente
        """
        if not request_id or not isinstance(request_id, str):
            return False
        
        request_id = request_id.strip()
        if not request_id:
            return False
        
        self._request_history.append((self._request_index, request_id))
        self._request_index += 1
        return True
    
    def clear_request_history(self) -> None:
        """Limpia el historial de requests."""
        self._request_history.clear()
        self._request_index = 0
    
    @property
    def request_history_size(self) -> int:
        """Tamaño actual del historial."""
        return len(self._request_history)
    
    # -------------------------------------------------------------------------
    # Cálculos Topológicos (Números de Betti)
    # -------------------------------------------------------------------------
    
    def calculate_betti_numbers(
        self,
        include_isolated: bool = True,
        calculate_b1: bool = True,
        use_cache: bool = True
    ) -> BettiNumbers:
        """
        Calcula los números de Betti del sistema con validación rigurosa.
        
        Algoritmo:
        1. Determinar subgrafo a analizar (con/sin nodos aislados)
        2. Calcular β₀ usando componentes conexas de NetworkX
        3. Calcular β₁ usando fórmula de Euler-Poincaré: β₁ = |E| - |V| + β₀
        4. Validar consistencia con invariantes topológicos
        5. Cachear resultado
        
        Teorema (Euler-Poincaré para Grafos):
            χ = |V| - |E| = β₀ - β₁
            ⟹ β₁ = |E| - |V| + β₀
        
        Args:
            include_isolated: Si True, incluye nodos sin conexiones
            calculate_b1: Si True, calcula β₁ (ciclos)
            use_cache: Si True, usa caché si disponible
        
        Returns:
            BettiNumbers validados y consistentes
        
        Raises:
            BettiNumberError: Si hay inconsistencia matemática irreconciliable
        """
        # Intentar usar caché
        if use_cache and self._betti_cache is not None:
            graph_hash = self._compute_graph_hash()
            cached_hash, cached_betti = self._betti_cache
            if cached_hash == graph_hash:
                logger.debug("Usando Betti numbers desde caché")
                return cached_betti
        
        # Caso especial: grafo vacío
        if self._graph.number_of_nodes() == 0:
            betti = BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
            if use_cache:
                self._betti_cache = (self._compute_graph_hash(), betti)
            return betti
        
        # Determinar subgrafo a analizar
        if include_isolated:
            subgraph = self._graph
        else:
            # Solo nodos con grado > 0
            connected_nodes = [n for n in self._graph.nodes() if self._graph.degree(n) > 0]
            if not connected_nodes:
                betti = BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
                if use_cache:
                    self._betti_cache = (self._compute_graph_hash(), betti)
                return betti
            subgraph = self._graph.subgraph(connected_nodes)
        
        num_vertices = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        
        # Caso 1: Sin aristas (todos nodos aislados)
        if num_edges == 0:
            # Cada nodo aislado es una componente conexa
            b0 = num_vertices
            b1 = 0
        else:
            # Calcular β₀ usando componentes conexas
            try:
                b0 = nx.number_connected_components(subgraph)
            except nx.NetworkXError as e:
                logger.error(f"Error calculando componentes conexas: {e}")
                # Fallback conservador
                b0 = 1
            
            b1 = 0
            if calculate_b1:
                # Fórmula de Euler-Poincaré: β₁ = |E| - |V| + β₀
                b1 = num_edges - num_vertices + b0
                
                # Validación: β₁ debe ser ≥ 0 para grafos simples
                if b1 < 0:
                    logger.warning(
                        f"β₁ negativo calculado ({b1}). Recalculando con enfoque conservador. "
                        f"V={num_vertices}, E={num_edges}, β₀={b0}"
                    )
                    
                    # Enfoque alternativo: contar ciclos fundamentales
                    try:
                        cycles = nx.cycle_basis(subgraph)
                        b1_alt = len(cycles)
                        logger.info(f"β₁ recalculado mediante base de ciclos: {b1_alt}")
                        b1 = b1_alt
                    except nx.NetworkXError:
                        # Último recurso: ajustar con max
                        b1 = max(0, num_edges - num_vertices + b0)
        
        # Validación final
        if b0 < 0 or b1 < 0:
            raise BettiNumberError(
                f"Números de Betti inválidos: β₀={b0}, β₁={b1}",
                context={
                    "vertices": num_vertices,
                    "edges": num_edges,
                    "subgraph_info": str(subgraph)
                }
            )
        
        # Cota: β₀ ≤ |V|
        if b0 > num_vertices:
            logger.warning(
                f"β₀ ({b0}) > |V| ({num_vertices}). Ajustando β₀ = {num_vertices}"
            )
            b0 = num_vertices
            # Recalcular β₁ para mantener consistencia
            if calculate_b1:
                b1 = max(0, num_edges - num_vertices + b0)
        
        # Crear objeto con validación automática
        try:
            betti = BettiNumbers(
                b0=b0,
                b1=b1,
                num_vertices=num_vertices,
                num_edges=num_edges
            )
        except BettiNumberError as e:
            # Si la validación falla, loggear y re-lanzar
            logger.error(f"Fallo al crear BettiNumbers: {e}")
            raise
        
        # Cachear resultado
        if use_cache:
            self._betti_cache = (self._compute_graph_hash(), betti)
        
        logger.debug(f"Betti numbers calculados: {betti}")
        return betti
    
    def calculate_cyclomatic_complexity(self) -> int:
        """
        Calcula la complejidad ciclomática del grafo.
        
        Fórmula (McCabe generalizada para grafos):
            M = E - N + 2P
        
        donde P = número de componentes conexas.
        
        Para nuestro caso:
            M = β₁ + β₀
        
        Returns:
            Complejidad ciclomática total
        """
        betti = self.calculate_betti_numbers()
        return betti.cyclomatic_complexity


# =============================================================================
# FASE 3: DETECCIÓN DE ANOMALÍAS Y HOMOLOGÍA PERSISTENTE
# =============================================================================

    # -------------------------------------------------------------------------
    # Detección de Ciclos y Anomalías (continuación de SystemTopology)
    # -------------------------------------------------------------------------
    
    def detect_request_loops(
        self,
        threshold: int = 3,
        window: Optional[int] = None,
        min_time_between_repeats: Optional[int] = None
    ) -> List[RequestLoopInfo]:
        """
        Detecta patrones de reintentos con análisis temporal robusto.
        
        Algoritmo Mejorado:
        1. Agrupar requests por ID
        2. Calcular intervalos entre ocurrencias
        3. Analizar regularidad (desviación estándar de intervalos)
        4. Calcular densidad temporal
        5. Clasificar por severidad compuesta
        
        Mejoras vs versión original:
        - Análisis de frecuencia temporal (no solo conteo)
        - Detección de patrones periódicos
        - Filtrado por densidad temporal
        - Clasificación por severidad multicriterio
        
        Args:
            threshold: Mínimo número de repeticiones para considerar loop
            window: Ventana temporal de análisis (en índices)
            min_time_between_repeats: Mínimo tiempo entre repeticiones
        
        Returns:
            Lista de RequestLoopInfo ordenada por severidad descendente
        """
        if not self._request_history:
            return []
        
        # Validación y normalización de parámetros
        threshold = max(2, int(threshold)) if isinstance(threshold, (int, float)) else 3
        
        if window is not None:
            try:
                window = int(window)
                if window <= 0:
                    window = None
            except (TypeError, ValueError):
                window = None
        
        if min_time_between_repeats is not None:
            try:
                min_time_between_repeats = int(min_time_between_repeats)
                if min_time_between_repeats < 0:
                    min_time_between_repeats = None
            except (TypeError, ValueError):
                min_time_between_repeats = None
        
        # Obtener historial dentro de ventana
        history = list(self._request_history)
        if window and window < len(history):
            history = history[-window:]
        
        if len(history) < threshold:
            return []
        
        # Fase 1: Agrupar por request_id con análisis temporal
        request_analysis: Dict[str, Dict[str, Any]] = {}
        
        for idx, req_id in history:
            if req_id not in request_analysis:
                request_analysis[req_id] = {
                    "count": 0,
                    "indices": [],
                    "first": idx,
                    "last": idx,
                    "intervals": []
                }
            
            info = request_analysis[req_id]
            info["count"] += 1
            info["indices"].append(idx)
            info["last"] = idx
            
            # Calcular intervalo desde última ocurrencia
            if len(info["indices"]) > 1:
                last_idx = info["indices"][-2]
                interval = idx - last_idx
                info["intervals"].append(interval)
        
        # Fase 2: Análisis de patrones y clasificación
        loops: List[RequestLoopInfo] = []
        
        for req_id, info in request_analysis.items():
            if info["count"] < threshold:
                continue
            
            indices = info["indices"]
            intervals = info["intervals"]
            
            # Análisis de regularidad
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                
                if len(intervals) > 1:
                    # Varianza y desviación estándar
                    variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                    interval_std = math.sqrt(variance)
                    
                    # Coeficiente de variación inverso (medida de regularidad)
                    regularity = avg_interval / (interval_std + TC.EPSILON) if interval_std > 0 else 1.0
                else:
                    interval_std = 0.0
                    regularity = 1.0
            else:
                avg_interval = 0.0
                interval_std = 0.0
                regularity = 1.0
            
            # Densidad temporal
            time_span = info["last"] - info["first"] + 1
            density = info["count"] / time_span if time_span > 0 else 1.0
            
            # Filtrar por mínimo tiempo entre repeticiones
            if min_time_between_repeats is not None and intervals:
                min_observed_interval = min(intervals)
                if min_observed_interval < min_time_between_repeats:
                    logger.debug(
                        f"Request {req_id} tiene intervalo mínimo {min_observed_interval} < "
                        f"{min_time_between_repeats}, filtrando"
                    )
                    continue
            
            # Calcular severidad compuesta (modelo multicriterio)
            # Factores: frecuencia, regularidad, densidad
            frequency_score = min(1.0, info["count"] / 10.0)  # Normalizado a máx 10
            regularity_score = min(1.0, regularity / 5.0)
            density_score = min(1.0, density * 5.0)
            
            # Combinación ponderada
            severity = (
                0.5 * frequency_score +
                0.3 * regularity_score +
                0.2 * density_score
            )
            
            # Crear RequestLoopInfo
            loop_info = RequestLoopInfo(
                request_id=req_id,
                count=info["count"],
                first_seen=info["first"],
                last_seen=info["last"]
            )
            
            # Agregar metadata para ordenamiento (usando atributo dinámico)
            setattr(loop_info, "_severity", severity)
            setattr(loop_info, "_metadata", {
                "avg_interval": avg_interval,
                "interval_std": interval_std,
                "regularity": regularity,
                "density": density,
                "severity": severity
            })
            
            loops.append(loop_info)
        
        # Ordenar por severidad (descendente) y luego por frecuencia
        def get_severity(loop: RequestLoopInfo) -> float:
            return getattr(loop, "_severity", 0.0)
        
        return sorted(loops, key=lambda x: (get_severity(x), x.count), reverse=True)
    
    def find_structural_cycles(self) -> List[List[str]]:
        """
        Encuentra todos los ciclos fundamentales en el grafo.
        
        Usa la base de ciclos del grafo, que tiene exactamente β₁ ciclos.
        
        Teorema:
            Para un grafo conexo G = (V, E):
            - β₁ = |E| - |V| + 1
            - La base de ciclos tiene exactamente β₁ ciclos
        
        Returns:
            Lista de ciclos, donde cada ciclo es una lista de nodos
        """
        try:
            cycles = list(nx.cycle_basis(self._graph))
            logger.debug(f"Ciclos fundamentales encontrados: {len(cycles)}")
            return cycles
        except nx.NetworkXError as e:
            logger.warning(f"Error calculando base de ciclos: {e}")
            return []
    
    def get_disconnected_nodes(self) -> FrozenSet[str]:
        """
        Identifica nodos requeridos que están desconectados.
        
        Un nodo está desconectado si su grado es 0.
        
        Returns:
            Conjunto inmutable de nodos sin conexiones
        """
        disconnected = frozenset(
            node
            for node in self.REQUIRED_NODES
            if node in self._graph and self._graph.degree(node) == 0
        )
        return disconnected
    
    def get_missing_connections(self) -> FrozenSet[Tuple[str, str]]:
        """
        Identifica conexiones esperadas que faltan.
        
        Compara la topología actual con la topología esperada.
        
        Returns:
            Conjunto inmutable de aristas esperadas no presentes
        """
        missing = frozenset(
            edge
            for edge in self._expected_topology
            if not self._graph.has_edge(*edge)
        )
        return missing
    
    def get_unexpected_connections(self) -> FrozenSet[Tuple[str, str]]:
        """
        Identifica conexiones no esperadas en la topología.
        
        Útil para detectar dependencias no documentadas.
        
        Returns:
            Conjunto de aristas presentes pero no esperadas
        """
        current_edges = set(self._graph.edges())
        
        # Normalizar dirección de aristas para comparación
        expected_normalized = set()
        for u, v in self._expected_topology:
            expected_normalized.add((min(u, v), max(u, v)))
        
        unexpected = set()
        for u, v in current_edges:
            normalized = (min(u, v), max(u, v))
            if normalized not in expected_normalized:
                unexpected.add((u, v))
        
        return frozenset(unexpected)
    
    # -------------------------------------------------------------------------
    # Análisis de Salud Topológica
    # -------------------------------------------------------------------------
    
    def get_topological_health(
        self,
        calculate_b1: bool = True,
        use_cache: bool = True
    ) -> TopologicalHealth:
        """
        Calcula salud topológica con modelo de penalizaciones riguroso.
        
        Modelo Matemático:
        -----------------
        Score = 1.0 - Σᵢ (penalización_i × peso_i)
        
        donde:
        - penalización_i ∈ [0, 1] (normalizada por máximo posible)
        - peso_i ∈ [0, 1] con Σpesos_i = 1.0
        
        Pesos (definidos en TopologicalConstants):
        - Fragmentación (β₀ > 1): 0.35
        - Ciclos estructurales (β₁ > 0): 0.20
        - Nodos requeridos desconectados: 0.25
        - Conexiones esperadas faltantes: 0.15
        - Bucles de reintento: 0.05
        
        Penalizaciones Normalizadas:
        1. Fragmentación: (β₀ - 1) / (|V| - 1)
        2. Ciclos: β₁ / max_cycles_approx
        3. Desconectados: n_desconectados / n_requeridos
        4. Faltantes: n_faltantes / n_esperadas
        5. Bucles: combinación de frecuencia y diversidad
        
        Args:
            calculate_b1: Si True, incluye penalización por ciclos
            use_cache: Si True, usa caché si disponible
        
        Returns:
            TopologicalHealth con score fundamentado matemáticamente
        """
        # Intentar usar caché
        if use_cache and self._health_cache is not None:
            graph_hash = self._compute_graph_hash()
            cached_hash, cached_health = self._health_cache
            if cached_hash == graph_hash:
                logger.debug("Usando TopologicalHealth desde caché")
                return cached_health
        
        # Calcular métricas base
        betti = self.calculate_betti_numbers(calculate_b1=calculate_b1, use_cache=use_cache)
        disconnected = self.get_disconnected_nodes()
        missing = self.get_missing_connections()
        loops = self.detect_request_loops()
        
        # Configurar pesos
        if calculate_b1:
            weights = {
                "fragmentation": TC.WEIGHT_FRAGMENTATION,
                "cycles": TC.WEIGHT_CYCLES,
                "disconnected": TC.WEIGHT_DISCONNECTED,
                "missing_edges": TC.WEIGHT_MISSING_EDGES,
                "retry_loops": TC.WEIGHT_RETRY_LOOPS
            }
        else:
            # Re-normalizar sin ciclos
            base_weights = {
                "fragmentation": TC.WEIGHT_FRAGMENTATION,
                "disconnected": TC.WEIGHT_DISCONNECTED,
                "missing_edges": TC.WEIGHT_MISSING_EDGES,
                "retry_loops": TC.WEIGHT_RETRY_LOOPS
            }
            total_weight = sum(base_weights.values())
            weights = {k: v / total_weight for k, v in base_weights.items()}
            weights["cycles"] = 0.0
        
        diagnostics: Dict[str, str] = {}
        penalty_score = 0.0
        
        # 1. Penalización por fragmentación (β₀ > 1)
        if betti.b0 > 1:
            # Normalizar: máximo fragmentación = todos nodos aislados
            max_components = betti.num_vertices
            actual_components = betti.b0
            
            if max_components > 1:
                fragmentation_penalty = (actual_components - 1) / (max_components - 1)
            else:
                fragmentation_penalty = 0.0
            
            penalty_score += fragmentation_penalty * weights["fragmentation"]
            diagnostics["fragmentation"] = (
                f"Sistema fragmentado en {betti.b0} componentes "
                f"(penalización: {fragmentation_penalty:.3f})"
            )
        
        # 2. Penalización por ciclos estructurales (solo si calculate_b1=True)
        if calculate_b1 and betti.b1 > 0:
            n = betti.num_vertices
            
            if n >= 3:
                # Máximo teórico de ciclos fundamentales
                # Para grafo completo K_n: aproximadamente C(n,3)
                max_cycles_approx = min(
                    math.comb(n, 3) if n <= 20 else n * (n - 1) * (n - 2) // 6,
                    1000  # Cota superior para evitar overflow
                )
                
                if max_cycles_approx > 0:
                    cycles_penalty = min(1.0, betti.b1 / max_cycles_approx)
                else:
                    cycles_penalty = 0.0
            else:
                cycles_penalty = 1.0 if betti.b1 > 0 else 0.0
            
            penalty_score += cycles_penalty * weights["cycles"]
            diagnostics["cycles"] = (
                f"{betti.b1} ciclo(s) estructural(es) "
                f"(penalización: {cycles_penalty:.3f})"
            )
        
        # 3. Penalización por nodos requeridos desconectados
        if disconnected:
            num_disconnected = len(disconnected)
            num_required = len(self.REQUIRED_NODES)
            
            disconnected_penalty = num_disconnected / num_required if num_required > 0 else 0.0
            
            penalty_score += disconnected_penalty * weights["disconnected"]
            nodes_str = ", ".join(sorted(disconnected))
            diagnostics["disconnected"] = (
                f"{num_disconnected}/{num_required} nodos requeridos desconectados: "
                f"{nodes_str} (penalización: {disconnected_penalty:.3f})"
            )
        
        # 4. Penalización por conexiones esperadas faltantes
        if missing:
            num_missing = len(missing)
            num_expected = len(self._expected_topology)
            
            missing_penalty = num_missing / num_expected if num_expected > 0 else 0.0
            
            penalty_score += missing_penalty * weights["missing_edges"]
            edges_str = ", ".join(f"{u}-{v}" for u, v in sorted(missing))
            diagnostics["missing_edges"] = (
                f"{num_missing}/{num_expected} conexiones esperadas faltantes: "
                f"{edges_str} (penalización: {missing_penalty:.3f})"
            )
        
        # 5. Penalización por bucles de reintento
        if loops:
            total_retries = sum(loop.count for loop in loops)
            unique_loops = len(loops)
            
            # Modelo combinado: frecuencia y diversidad
            max_retries = self._max_history
            if max_retries > 0:
                retry_frequency_penalty = min(1.0, total_retries / (2 * max_retries))
            else:
                retry_frequency_penalty = 0.0
            
            # Penalización por diversidad de bucles
            loop_diversity_penalty = min(1.0, unique_loops / 5.0)
            
            # Combinar: 70% frecuencia, 30% diversidad
            retry_penalty = 0.7 * retry_frequency_penalty + 0.3 * loop_diversity_penalty
            
            penalty_score += retry_penalty * weights["retry_loops"]
            diagnostics["retry_loops"] = (
                f"{unique_loops} patrón(es) de reintento, {total_retries} intentos totales "
                f"(penalización: {retry_penalty:.3f})"
            )
        
        # Calcular score final (asegurar [0, 1])
        health_score = max(0.0, min(1.0, 1.0 - penalty_score))
        
        # Determinar nivel con márgenes definidos
        level = HealthLevel.from_score(health_score)
        
        # Si no hay diagnósticos, sistema está óptimo
        if not diagnostics:
            diagnostics["status"] = "Sistema topológicamente óptimo"
        
        # Crear objeto de salud
        health = TopologicalHealth(
            betti=betti,
            disconnected_nodes=disconnected,
            missing_edges=missing,
            request_loops=tuple(loops),
            health_score=round(health_score, 4),
            level=level,
            diagnostics=diagnostics
        )
        
        # Cachear resultado
        if use_cache:
            self._health_cache = (self._compute_graph_hash(), health)
        
        logger.debug(f"Salud topológica calculada: {health}")
        return health
    
    # -------------------------------------------------------------------------
    # Visualización
    # -------------------------------------------------------------------------
    
    def visualize_topology(
        self,
        output_path: str = "data/topology_status.png",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 120,
        show_labels: bool = True,
        show_weights: bool = False
    ) -> bool:
        """
        Genera visualización del grafo con estado actual vs esperado.
        
        Leyenda de colores:
        - Aristas VERDES: Conexiones activas esperadas
        - Aristas ROJAS punteadas: Conexiones faltantes
        - Aristas NARANJAS: Conexiones inesperadas
        - Nodos AZUL CLARO: Nodos requeridos conectados
        - Nodos SALMÓN: Nodos aislados
        - Nodos GRIS CLARO: Nodos no existentes pero en topología esperada
        
        Args:
            output_path: Ruta donde guardar la imagen
            figsize: Tamaño de la figura (ancho, alto) en pulgadas
            dpi: Resolución de la imagen
            show_labels: Si True, muestra etiquetas de nodos
            show_weights: Si True, muestra pesos de aristas (si existen)
        
        Returns:
            True si se generó correctamente, False en caso contrario
        """
        # Validación de parámetros
        if not isinstance(output_path, str) or not output_path.strip():
            logger.error("output_path inválido")
            return False
        
        output_path = output_path.strip()
        
        # Validar extensión
        import os
        valid_extensions = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
        _, ext = os.path.splitext(output_path.lower())
        if ext not in valid_extensions:
            logger.warning(f"Extensión '{ext}' no reconocida, agregando '.png'")
            output_path = output_path + ".png"
        
        # Validar figsize y dpi
        try:
            figsize = (int(figsize[0]), int(figsize[1]))
            if figsize[0] <= 0 or figsize[1] <= 0:
                raise ValueError("Dimensiones deben ser positivas")
        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"figsize inválido: {e}, usando default (12, 10)")
            figsize = (12, 10)
        
        try:
            dpi = int(dpi)
            if dpi <= 0:
                raise ValueError("DPI debe ser positivo")
        except (TypeError, ValueError) as e:
            logger.warning(f"dpi inválido: {e}, usando default 120")
            dpi = 120
        
        # Importar matplotlib con manejo de errores
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            logger.error("matplotlib no disponible. Instalar: pip install matplotlib")
            return False
        except Exception as e:
            logger.error(f"Error configurando matplotlib: {e}")
            return False
        
        fig = None
        try:
            # Crear figura
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Crear grafo compuesto (actual + esperado)
            viz_graph = nx.Graph()
            
            # Agregar todos los nodos
            all_nodes = set(self._graph.nodes())
            for u, v in self._expected_topology:
                all_nodes.add(u)
                all_nodes.add(v)
            viz_graph.add_nodes_from(all_nodes)
            
            # Clasificar aristas
            existing_edges = []
            missing_edges = []
            unexpected_edges = []
            
            # Normalizar aristas esperadas
            expected_normalized = {
                tuple(sorted([u, v])) for u, v in self._expected_topology
            }
            
            # Aristas existentes
            for u, v in self._graph.edges():
                viz_graph.add_edge(u, v)
                normalized = tuple(sorted([u, v]))
                if normalized in expected_normalized:
                    existing_edges.append((u, v))
                else:
                    unexpected_edges.append((u, v))
            
            # Aristas faltantes
            for u, v in self._expected_topology:
                if not self._graph.has_edge(u, v):
                    viz_graph.add_edge(u, v)
                    missing_edges.append((u, v))
            
            # Calcular layout con reproducibilidad
            try:
                pos = nx.spring_layout(viz_graph, seed=42, k=2.0, iterations=50)
            except Exception:
                pos = nx.circular_layout(viz_graph)
            
            # Colorear nodos según estado
            node_colors = []
            node_sizes = []
            for node in viz_graph.nodes():
                if node not in self._graph:
                    node_colors.append("lightgray")
                    node_sizes.append(1500)
                elif self._graph.degree(node) == 0:
                    node_colors.append("salmon")
                    node_sizes.append(2000)
                elif node in self.REQUIRED_NODES:
                    node_colors.append("lightblue")
                    node_sizes.append(2500)
                else:
                    node_colors.append("lightgreen")
                    node_sizes.append(2000)
            
            # Dibujar nodos
            nx.draw_networkx_nodes(
                viz_graph,
                pos,
                ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                edgecolors="black",
                linewidths=2
            )
            
            # Dibujar etiquetas
            if show_labels:
                nx.draw_networkx_labels(
                    viz_graph,
                    pos,
                    ax=ax,
                    font_size=9,
                    font_weight="bold"
                )
            
            # Dibujar aristas existentes (Verde)
            if existing_edges:
                nx.draw_networkx_edges(
                    viz_graph,
                    pos,
                    ax=ax,
                    edgelist=existing_edges,
                    edge_color="green",
                    width=3.0,
                    style="solid",
                    alpha=0.8
                )
            
            # Dibujar aristas faltantes (Rojo punteado)
            if missing_edges:
                nx.draw_networkx_edges(
                    viz_graph,
                    pos,
                    ax=ax,
                    edgelist=missing_edges,
                    edge_color="red",
                    width=2.5,
                    style="dashed",
                    alpha=0.7
                )
            
            # Dibujar aristas inesperadas (Naranja)
            if unexpected_edges:
                nx.draw_networkx_edges(
                    viz_graph,
                    pos,
                    ax=ax,
                    edgelist=unexpected_edges,
                    edge_color="orange",
                    width=2.5,
                    style="dotted",
                    alpha=0.7
                )
            
            # Título con información topológica
            betti = self.calculate_betti_numbers()
            health = self.get_topological_health()
            
            title = (
                f"Estado Topológico del Sistema\n"
                f"β₀={betti.b0} (componentes), β₁={betti.b1} (ciclos), "
                f"χ={betti.euler_characteristic}\n"
                f"Salud: {health.level.name} ({health.health_score:.3f})"
            )
            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
            
            # Leyenda
            legend_elements = [
                Line2D([0], [0], color="green", linewidth=3, label="Conexión Activa"),
                Line2D([0], [0], color="red", linewidth=2.5, linestyle="--", label="Conexión Faltante"),
                Line2D([0], [0], color="orange", linewidth=2.5, linestyle=":", label="Conexión Inesperada")
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
            
            ax.axis("off")
            plt.tight_layout()
            
            # Crear directorio si no existe
            output_dir = os.path.dirname(output_path)
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"No se pudo crear directorio '{output_dir}': {e}")
                    return False
            
            # Guardar figura
            try:
                plt.savefig(
                    output_path,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none"
                )
                logger.info(f"✅ Visualización guardada en: {output_path}")
                return True
            except PermissionError:
                logger.error(f"Sin permisos para escribir en: {output_path}")
                return False
            except OSError as e:
                logger.error(f"Error de sistema guardando imagen: {e}")
                return False
        
        except Exception as e:
            logger.error(f"Error generando visualización: {e}", exc_info=True)
            return False
        
        finally:
            if fig is not None:
                plt.close(fig)
            else:
                plt.close("all")
    
    # -------------------------------------------------------------------------
    # Utilidades y Serialización
    # -------------------------------------------------------------------------
    
    def get_adjacency_matrix(self) -> AdjacencyDict:
        """
        Retorna la matriz de adyacencia como diccionario anidado.
        
        Returns:
            Diccionario {nodo: {vecino: 1}} para grafo simple
        """
        nodes = sorted(self._graph.nodes())
        matrix = {n: {m: 0 for m in nodes} for n in nodes}
        
        for u, v in self._graph.edges():
            matrix[u][v] = 1
            matrix[v][u] = 1
        
        return matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa el estado actual a un diccionario completo.
        
        Incluye toda la información necesaria para reconstruir o debuggear
        el estado topológico del sistema.
        
        Returns:
            Diccionario serializable a JSON
        """
        # Calcular métricas (con caché)
        betti = self.calculate_betti_numbers()
        health = self.get_topological_health()
        
        # Anomalías
        disconnected = self.get_disconnected_nodes()
        missing = self.get_missing_connections()
        unexpected = self.get_unexpected_connections()
        
        return {
            "graph": {
                "nodes": sorted(self._graph.nodes()),
                "edges": sorted([tuple(sorted(e)) for e in self._graph.edges()]),
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "density": self.density
            },
            "betti_numbers": betti.to_dict(),
            "health": health.to_dict(),
            "topology_status": {
                "disconnected_nodes": sorted(disconnected),
                "missing_connections": sorted([tuple(sorted(e)) for e in missing]),
                "unexpected_connections": sorted([tuple(sorted(e)) for e in unexpected])
            },
            "request_history": {
                "size": len(self._request_history),
                "max_size": self._max_history,
                "current_index": self._request_index
            },
            "configuration": {
                "required_nodes": sorted(self.REQUIRED_NODES),
                "expected_topology": sorted([tuple(sorted(e)) for e in self._expected_topology]),
                "validate_strictly": self._validate_strictly
            }
        }
    
    def __repr__(self) -> str:
        """Representación string para debugging."""
        try:
            node_count = self.num_nodes
            edge_count = self.num_edges
        except Exception:
            node_count = "?"
            edge_count = "?"
        
        try:
            betti = self.calculate_betti_numbers()
            betti_str = str(betti)
        except Exception as e:
            betti_str = f"BettiError({type(e).__name__})"
        
        return f"SystemTopology(nodes={node_count}, edges={edge_count}, {betti_str})"
    
    def __str__(self) -> str:
        """Representación string user-friendly."""
        health = self.get_topological_health()
        return (
            f"SystemTopology(health={health.level.name}, score={health.health_score:.3f}, "
            f"β₀={health.betti.b0}, β₁={health.betti.b1})"
        )


# =============================================================================
# CLASE: PersistenceHomology (Motor de Homología Persistente)
# =============================================================================

class PersistenceHomology:
    """
    Análisis de Homología Persistente para métricas de series temporales.
    
    Fundamentación Matemática:
    -------------------------
    La homología persistente estudia la evolución de características
    topológicas a través de una filtración (secuencia de subespacios).
    
    Para series temporales, implementamos una filtración por nivel:
    1. Definimos un umbral θ
    2. Para cada punto temporal t, observamos si f(t) > θ
    3. Las "excursiones" sobre el umbral son características 0-dimensionales
    4. Registramos nacimiento (cruce ascendente) y muerte (cruce descendente)
    
    Diagrama de Persistencia:
    - Cada característica se representa como punto (birth, death)
    - Persistencia = (death - birth) / √2 (distancia perpendicular a diagonal)
    - Alta persistencia = característica estructural
    - Baja persistencia = ruido
    
    Métricas Derivadas:
    - Total persistence: Σ(death - birth) para todas las características
    - Persistence entropy: Medida de complejidad del diagrama
    - Wasserstein distance: Métrica entre diagramas
    """
    
    DEFAULT_WINDOW_SIZE: ClassVar[int] = 20
    MIN_WINDOW_SIZE: ClassVar[int] = 3
    MAX_WINDOW_SIZE: ClassVar[int] = 10000
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        """
        Inicializa el analizador de persistencia.
        
        Args:
            window_size: Tamaño de la ventana deslizante
        
        Raises:
            ValueError: Si window_size está fuera de rango válido
        """
        if not (self.MIN_WINDOW_SIZE <= window_size <= self.MAX_WINDOW_SIZE):
            raise ValueError(
                f"window_size debe estar en [{self.MIN_WINDOW_SIZE}, {self.MAX_WINDOW_SIZE}]: "
                f"{window_size}"
            )
        
        self.window_size = window_size
        self._buffers: Dict[str, deque] = {}
        
        logger.debug(f"PersistenceHomology inicializado: window_size={window_size}")
    
    @property
    def metrics(self) -> Set[str]:
        """Conjunto de métricas registradas."""
        return set(self._buffers.keys())
    
    @property
    def num_metrics(self) -> int:
        """Número de métricas registradas."""
        return len(self._buffers)
    
    # -------------------------------------------------------------------------
    # Gestión de Datos
    # -------------------------------------------------------------------------
    
    def add_reading(self, metric_name: str, value: float) -> bool:
        """
        Agrega una lectura al buffer de una métrica con validación rigurosa.
        
        Args:
            metric_name: Nombre de la métrica (no vacío)
            value: Valor de la lectura (numérico finito)
        
        Returns:
            True si se agregó correctamente, False si se rechazó
        """
        # Validar nombre de métrica
        if not isinstance(metric_name, str):
            logger.warning(f"Nombre de métrica debe ser string: {type(metric_name).__name__}")
            return False
        
        metric_name = metric_name.strip()
        if not metric_name:
            logger.warning("Nombre de métrica vacío rechazado")
            return False
        
        # Validar valor numérico
        if not isinstance(value, (int, float, Decimal)):
            logger.warning(f"Valor no numérico para {metric_name}: {type(value).__name__}")
            return False
        
        # Convertir a float
        try:
            value = float(value)
        except (ValueError, OverflowError):
            logger.warning(f"No se pudo convertir valor a float: {value}")
            return False
        
        # Rechazar NaN estrictamente
        if math.isnan(value):
            logger.warning(f"Valor NaN rechazado para {metric_name}")
            return False
        
        # Manejar infinitos con cap
        if math.isinf(value):
            MAX_FINITE = 1e100
            capped_value = math.copysign(MAX_FINITE, value)
            logger.warning(f"Valor infinito para {metric_name} capeado a {capped_value:.2e}")
            value = capped_value
        
        # Detectar valores extremos
        if abs(value) > 1e100:
            logger.debug(f"Valor extremo detectado para {metric_name}: {value:.2e}")
        
        # Agregar al buffer (crear si no existe)
        if metric_name not in self._buffers:
            self._buffers[metric_name] = deque(maxlen=self.window_size)
            logger.debug(f"Nuevo buffer creado para métrica: {metric_name}")
        
        self._buffers[metric_name].append(value)
        return True
    
    def add_readings_batch(self, metric_name: str, values: Sequence[float]) -> int:
        """
        Agrega múltiples lecturas a una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            values: Secuencia de valores
        
        Returns:
            Número de lecturas agregadas exitosamente
        """
        count = 0
        for value in values:
            if self.add_reading(metric_name, value):
                count += 1
        return count
    
    def get_buffer(self, metric_name: str) -> Optional[List[float]]:
        """Obtiene copia del buffer de una métrica."""
        buffer = self._buffers.get(metric_name)
        return list(buffer) if buffer else None
    
    def clear_metric(self, metric_name: str) -> bool:
        """Elimina una métrica específica."""
        if metric_name in self._buffers:
            del self._buffers[metric_name]
            logger.debug(f"Métrica eliminada: {metric_name}")
            return True
        return False
    
    def clear_all(self) -> None:
        """Limpia todos los buffers."""
        num_metrics = len(self._buffers)
        self._buffers.clear()
        logger.debug(f"Todos los buffers limpiados ({num_metrics} métricas)")
    
    # ... (Continuará en siguiente respuesta debido a límite de longitud)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def create_simple_topology() -> SystemTopology:
    """
    Crea una topología simple con conexiones default.
    
    Útil para testing y ejemplos.
    
    Returns:
        SystemTopology con topología básica configurada
    """
    topology = SystemTopology()
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem")
    ])
    return topology


def compute_wasserstein_distance(
    intervals1: List[PersistenceInterval],
    intervals2: List[PersistenceInterval],
    p: int = 2
) -> float:
    """
    Calcula la distancia p-Wasserstein aproximada entre dos diagramas.
    
    Esta es una aproximación que compara distribuciones ordenadas de duraciones.
    La distancia real de Wasserstein requiere resolver un problema de
    matching óptimo (assignación húngara), pero esta aproximación es
    computacionalmente eficiente y proporciona una métrica útil.
    
    Fórmula (aproximada):
        W_p ≈ (Σ |l1_i - l2_i|^p)^(1/p)
    
    Args:
        intervals1: Primer diagrama de persistencia
        intervals2: Segundo diagrama de persistencia
        p: Orden de la norma (debe ser >= 1, default=2)
    
    Returns:
        Distancia de Wasserstein aproximada (>= 0)
    
    Raises:
        ValueError: Si p < 1
    """
    # Validación de parámetro p
    if not isinstance(p, (int, float)):
        raise TypeError(f"p debe ser numérico, recibido: {type(p).__name__}")
    
    if p < 1:
        raise ValueError(f"p debe ser >= 1, recibido: {p}")
    
    p = float(p)
    
    # Caso trivial: ambos vacíos
    if not intervals1 and not intervals2:
        return 0.0
    
    # Extraer duraciones finitas
    def get_lifespans(intervals: List[PersistenceInterval]) -> List[float]:
        lifespans = []
        for i in intervals:
            if not i.is_alive:
                lifespan = i.lifespan
                if math.isfinite(lifespan) and lifespan >= 0:
                    lifespans.append(lifespan)
        return sorted(lifespans)
    
    lifespans1 = get_lifespans(intervals1)
    lifespans2 = get_lifespans(intervals2)
    
    # Ambos sin intervalos finitos
    if not lifespans1 and not lifespans2:
        return 0.0
    
    # Un diagrama vacío: distancia es la norma-p del otro
    if not lifespans1:
        return sum(l**p for l in lifespans2) ** (1.0 / p)
    if not lifespans2:
        return sum(l**p for l in lifespans1) ** (1.0 / p)
    
    # Igualar longitudes con padding de ceros (matching con diagonal)
    max_len = max(len(lifespans1), len(lifespans2))
    padded1 = lifespans1 + [0.0] * (max_len - len(lifespans1))
    padded2 = lifespans2 + [0.0] * (max_len - len(lifespans2))
    
    # Calcular distancia con manejo de overflow
    try:
        total = sum(abs(l1 - l2) ** p for l1, l2 in zip(padded1, padded2))
        distance = total ** (1.0 / p)
    except OverflowError:
        logger.warning("Overflow en cálculo de Wasserstein, retornando infinito")
        return float("inf")
    
    return distance


# =============================================================================
# PUNTO DE ENTRADA PARA PRUEBAS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("=" * 80)
    print("DEMO: Topological Analyzer - Versión 2.0.0-rigorous")
    print("=" * 80)
    
    # Demo de SystemTopology
    print("\n🔷 Demo: SystemTopology")
    print("-" * 80)
    
    topology = SystemTopology()
    print(f"Estado inicial: {topology}")
    
    health = topology.get_topological_health()
    print(f"\n{health.get_summary()}")
    
    # Agregar conexiones
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem")
    ])
    
    print(f"\nCon conexiones básicas: {topology}")
    health = topology.get_topological_health()
    print(f"Salud: {health.level.name} ({health.health_score:.3f})")
    
    # Agregar ciclo
    topology.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        ("Redis", "Agent")  # Ciclo
    ])
    
    print(f"\nCon ciclo: {topology}")
    betti = topology.calculate_betti_numbers()
    print(f"Betti: {betti}")
    print(f"Ciclos estructurales: {topology.find_structural_cycles()}")
    
    # Visualizar si matplotlib está disponible
    try:
        import matplotlib
        if topology.visualize_topology(output_path="topology_demo.png"):
            print("\n✅ Visualización generada: topology_demo.png")
    except ImportError:
        print("\n⚠️ matplotlib no disponible - visualización omitida")
    
    print("\n" + "=" * 80)
    print("✅ Demo completado exitosamente")
    print("=" * 80)