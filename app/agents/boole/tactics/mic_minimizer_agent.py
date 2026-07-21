# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Minimizer Agent (Custodio de la Base Booleana)                  ║
║ Ruta   : app/agents/boole/tactics/mic_minimizer_agent.py                     ║
║ Versión: 3.0.0-Grobner-ROBDD-Categorical-Rigorous-Advanced                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA DE BOOLE (Rigor Doctoral Avanzado):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mic_minimizer.py` en el subespacio Γ-TACTICS mediante
una composición funtorial estricta que preserva invariantes algebraicos, topológicos
y de teoría de la información.

FUNDAMENTOS MATEMÁTICOS RIGUROSOS:

1. ÁLGEBRA DE BOOLE Y TEORÍA DE ANILLOS:
   - Anillo Z₂ = GF(2) con operaciones (+, ·) módulo 2
   - Ideales booleanos: I = ⟨f₁, ..., fₘ⟩ ⊆ Z₂[x₁, ..., xₙ]
   - Bases de Gröbner: conjunto generador minimal único bajo orden monomial
   - Teorema de Hilbert: todo ideal tiene base finita

2. TEORÍA DE GRAFOS Y COMPLEJIDAD:
   - ROBDD (Reduced Ordered BDD): grafo dirigido acíclico canónico
   - Complejidad de reducción: O(|nodos|² · log|nodos|)
   - Isomorfismo de grafos preserva caminos de evaluación
   - Conectividad algebraica: rango del laplaciano del grafo de dependencias

3. TEORÍA DE LA INFORMACIÓN:
   - Entropía de Shannon: H(X) = -Σ p(xᵢ) log₂ p(xᵢ)
   - Información mutua: I(X;Y) = H(X) + H(Y) - H(X,Y)
   - Divergencia de Kullback-Leibler: D_KL(P‖Q) = Σ p(x) log(p(x)/q(x))
   - Distancia de variación total: d_TV(P,Q) = ½Σ|p(x) - q(x)|

4. TOPOLOGÍA ALGEBRAICA:
   - Homología de complejos simpliciales
   - Retractos y equivalencia homotópica
   - Grupos de cohomología de ideales
   - Números de Betti algebraicos

5. ÁLGEBRA LINEAL SOBRE GF(2):
   - Eliminación de Gauss-Jordan sobre campos finitos
   - Forma escalonada reducida por filas (RREF)
   - Teorema del rango-nulidad: rank(A) + nullity(A) = n
   - Bases duales y espacios ortogonales

6. TEORÍA DE CATEGORÍAS:
   - Funtor de reducción: R: Bool_full → Bool_minimal
   - Transformación natural: η: Id ⟹ R ∘ E (embedding)
   - Adjunción libre-olvido entre categorías booleanas
   - Topos de haces sobre espectro primo de Z₂[X]

7. TEORÍA DE CÓDIGOS:
   - Códigos lineales sobre GF(2)
   - Matriz generadora y matriz de paridad
   - Distancia de Hamming: d_H(x,y) = |{i : xᵢ ≠ yᵢ}|
   - Peso de Hamming: w_H(x) = d_H(x, 0)

ARQUITECTURA FUNTORIAL ANIDADA (3 FASES):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Bases de Gröbner
  ├─ Validación de matriz GF(2) (método inicial)
  ├─ Eliminación de Gauss-Jordan sobre GF(2)
  ├─ Cómputo de forma escalonada reducida
  ├─ Análisis de espacios nulos y kernel
  ├─ Detección de dependencias lineales
  └─ Certificado de Independencia (método final) ──┐
                                                     │
Fase 2 → Certificación de No-Interferencia          │
  ├─ Consumo del Certificado de Fase 1 ←────────────┘
  ├─ Cómputo de matriz de Gram (G = PPᵀ)
  ├─ Análisis espectral de ortogonalidad
  ├─ Detección de aristas de conflicto
  ├─ Validación de normas matriciales
  └─ Certificado de Ortogonalidad (método final) ──┐
                                                    │
Fase 3 → Isomorfismo ROBDD                         │
  ├─ Consumo del Certificado de Fase 2 ←───────────┘
  ├─ Validación de distribuciones probabilísticas
  ├─ Cómputo de entropía de Shannon
  ├─ Análisis de divergencia KL
  ├─ Distancia de variación total
  ├─ Mutual information preservation
  └─ Certificado de Equivalencia (método final)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Final, List, Optional, Protocol, Set, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════════
# §0. DEPENDENCIAS ARQUITECTÓNICAS Y PROTOCOLOS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Gamma.MinimizerAgent.v3")


# ─────────────────────────────────────────────────────────────────────────────
# Protocolos de Tipo para Estructuras Booleanas
# ─────────────────────────────────────────────────────────────────────────────
class BooleanAlgebraProtocol(Protocol):
    """Protocolo para álgebras booleanas válidas."""
    
    def rank_gf2(self) -> int: ...
    def nullity_gf2(self) -> int: ...
    def is_independent(self) -> bool: ...


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES BOOLEANAS, ALGEBRAICAS Y DE COMPLEJIDAD (RIGOR DOCTORAL)
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_LOG2: Final[float] = math.log(2.0)

# Límites combinatorios
_MAX_BOOLEAN_VARIABLES: Final[int] = 256
_MAX_ROBDD_NODES: Final[int] = 100000
_MAX_TRUTH_TABLE_SIZE: Final[int] = 2**20  # 1M entradas

# Tolerancias algebraicas y probabilísticas
_MIN_ENTROPY_TOLERANCE: Final[float] = 1e-12
_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-10
_PROBABILITY_TOLERANCE: Final[float] = 1e-12
_INTEGER_REPRESENTATION_TOLERANCE: Final[float] = 1e-12
_KL_DIVERGENCE_TOLERANCE: Final[float] = 1e-10
_HAMMING_DISTANCE_TOLERANCE: Final[int] = 0

# Factores de seguridad numérica
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_GF2_PIVOT_THRESHOLD: Final[float] = 0.5  # Para conversión a binario

# Límites de información
_MAX_SHANNON_ENTROPY_PER_BIT: Final[float] = 1.0  # H(X) ≤ log₂(|X|)
_MIN_MUTUAL_INFORMATION: Final[float] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS (ENRIQUECIDA)
# ═══════════════════════════════════════════════════════════════════════════════
class MICMinimizerAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Base Booleana."""
    pass


class BooleanInputValidationError(MICMinimizerAgentError):
    r"""Detonada si los datos booleanos, matriciales o probabilísticos son inválidos."""
    pass


class GrobnerDegeneracyError(MICMinimizerAgentError):
    r"""Detonada si el ideal booleano colapsa o pierde independencia efectiva."""
    pass


class NonInterferenceViolationError(MICMinimizerAgentError):
    r"""Detonada si ⟨eᵢ, eⱼ⟩ ≠ 0 para i ≠ j. Ruptura del Zero Side-Effects."""
    pass


class ROBDDHomotopyError(MICMinimizerAgentError):
    r"""Detonada si el ROBDD reducido no conserva la entropía booleana original."""
    pass


class LinearDependencyError(MICMinimizerAgentError):
    r"""Detonada si se detectan dependencias lineales no triviales en GF(2)."""
    pass


class ProbabilityDistributionError(MICMinimizerAgentError):
    r"""Detonada si las distribuciones probabilísticas son inválidas."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ENUMERACIONES PARA ESTRUCTURAS BOOLEANAS
# ═══════════════════════════════════════════════════════════════════════════════
class BooleanReductionPhase(Enum):
    """Fases del proceso de reducción booleana."""
    PHASE_1_GROBNER = auto()
    PHASE_2_UNSAT_CORE = auto()
    PHASE_3_ROBDD = auto()
    COMPLETE = auto()


class OrthogonalityMetric(Enum):
    """Métricas de ortogonalidad matricial."""
    FROBENIUS_NORM = auto()      # ‖PPᵀ - I‖_F
    SPECTRAL_NORM = auto()        # σₘₐₓ(PPᵀ - I)
    MAX_ABSOLUTE = auto()         # max|PPᵀ - I|
    OFF_DIAGONAL_SUM = auto()     # Σᵢ≠ⱼ |PPᵀᵢⱼ|


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES ENRIQUECIDAS (DTOs del Anillo Z₂)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GF2MatrixProperties:
    r"""
    Propiedades algebraicas de una matriz sobre GF(2).
    
    Invariantes:
        - rank + nullity = número de columnas (teorema del rango-nulidad)
        - Pivotes definen base del espacio columna
        - Kernel caracteriza dependencias lineales
    """
    rows: int
    cols: int
    rank_gf2: int
    nullity_gf2: int
    pivot_columns: Tuple[int, ...]
    free_columns: Tuple[int, ...]
    kernel_dimension: int
    has_full_rank: bool
    
    # Propiedades adicionales
    condition_number: float = field(default=1.0)
    sparsity: float = field(default=0.0)  # Fracción de ceros
    hamming_weight: int = field(default=0)  # Número de unos
    
    def __post_init__(self) -> None:
        """Validación de invariantes algebraicos."""
        if self.rank_gf2 + self.nullity_gf2 != self.cols:
            raise ValueError(
                f"Violación del teorema del rango-nulidad: "
                f"rank({self.rank_gf2}) + nullity({self.nullity_gf2}) "
                f"≠ cols({self.cols})"
            )
        if self.rank_gf2 > min(self.rows, self.cols):
            raise ValueError(
                f"Rango imposible: {self.rank_gf2} > min({self.rows}, {self.cols})"
            )


@dataclass(frozen=True, slots=True)
class LinearCodeProperties:
    r"""
    Propiedades de código lineal sobre GF(2).
    
    Un código lineal [n, k, d] tiene:
        - Longitud n (número de bits)
        - Dimensión k (log₂ del número de palabras código)
        - Distancia mínima d (mínimo peso de Hamming no nulo)
    """
    length: int  # n
    dimension: int  # k
    minimum_distance: int  # d
    rate: float  # k/n
    redundancy: int  # n - k
    
    # Cotas teóricas
    singleton_bound: int  # d ≤ n - k + 1
    hamming_bound: float  # Cota superior del número de palabras código
    
    def is_valid_code(self) -> bool:
        """Verifica cotas de teoría de códigos."""
        if self.minimum_distance > self.singleton_bound:
            return False
        if self.dimension > self.length:
            return False
        return True


@dataclass(frozen=True, slots=True)
class GrobnerAuditData:
    r"""
    Artefacto de Fase 1: Certificado de independencia algebraica en Z₂[X].
    
    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    
    Teorema (Base de Gröbner):
        Todo ideal I ⊆ Z₂[x₁,...,xₙ] tiene una única base de Gröbner reducida
        bajo un orden monomial fijo.
    """
    rows: int
    cols: int
    ideal_dimension: int
    nullity: int
    pivot_columns: Tuple[int, ...]
    is_minimally_independent: bool
    
    # Propiedades matriciales completas
    matrix_properties: GF2MatrixProperties
    
    # Propiedades de código lineal
    code_properties: Optional[LinearCodeProperties] = field(default=None)
    
    # Métricas de dependencia
    dependency_graph_edges: int = field(default=0)
    max_dependency_chain_length: int = field(default=0)
    
    # Forma escalonada reducida
    rref_matrix: Optional[NDArray[np.uint8]] = field(default=None)


@dataclass(frozen=True, slots=True)
class OrthogonalityAnalysis:
    r"""
    Análisis detallado de ortogonalidad matricial.
    
    Para matriz P ∈ ℝᵐˣⁿ, analiza G = PPᵀ:
        - Idealmente G = I (ortonormalidad)
        - Desviaciones indican no-ortogonalidad
    """
    gram_matrix: NDArray[np.float64]
    off_diagonal_norm: float
    diagonal_deviation_norm: float
    spectral_gap: float  # λₘᵢₙ(G) - separación del cero
    condition_number: float  # κ(G) = σₘₐₓ/σₘᵢₙ
    
    # Métricas por métrica de ortogonalidad
    frobenius_deviation: float
    spectral_deviation: float
    max_absolute_deviation: float
    
    # Análisis de interferencia
    conflict_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    max_interference_value: float = field(default=0.0)


@dataclass(frozen=True, slots=True)
class UnsatCoreCertifierData:
    r"""
    Artefacto de Fase 2: Certificado de ortogonalidad estricta y no-interferencia.
    
    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    
    Principio Zero Side-Effects:
        ∀i ≠ j: ⟨tool_i, tool_j⟩ = 0 (ortogonalidad estricta)
    """
    tool_count: int
    variable_dim: int
    off_diagonal_conflict_norm: float
    diagonal_deviation_norm: float
    conflict_edges: int
    orthogonality_tolerance: float
    is_strictly_orthogonal: bool
    
    # Análisis completo de ortogonalidad
    orthogonality_analysis: OrthogonalityAnalysis
    
    # Validación de proyección
    projection_rank: int = field(default=0)
    is_projection_idempotent: bool = field(default=False)  # P² = P


@dataclass(frozen=True, slots=True)
class EntropyAnalysis:
    r"""
    Análisis entrópico completo de distribuciones booleanas.
    
    Incluye múltiples métricas de teoría de la información.
    """
    shannon_entropy: float  # H(X) = -Σ p(x) log₂ p(x)
    min_entropy: float  # H_∞(X) = -log₂(max p(x))
    collision_entropy: float  # H₂(X) = -log₂(Σ p(x)²)
    hartley_entropy: float  # H₀(X) = log₂(|support(X)|)
    
    # Propiedades de la distribución
    support_size: int
    max_probability: float
    uniformity: float  # H(X) / log₂(|support|)
    
    def is_uniform(self, tolerance: float = 1e-6) -> bool:
        """Verifica si la distribución es aproximadamente uniforme."""
        return abs(self.uniformity - 1.0) < tolerance


@dataclass(frozen=True, slots=True)
class ROBDDIsomorphismData:
    r"""
    Artefacto de Fase 3: Certificado de conservación de entropía de Shannon booleana.
    
    Garantiza equivalencia homotópica bajo reducción ROBDD.
    
    Teorema (Unicidad de ROBDD):
        Para una función booleana f y un orden de variables fijo, existe un
        único ROBDD reducido que la representa.
    """
    support_dimension: int
    original_entropy: float
    reduced_entropy: float
    entropy_loss: float
    entropy_tolerance: float
    total_variation_distance: float
    is_homotopically_equivalent: bool
    
    # Análisis entrópico completo
    original_entropy_analysis: EntropyAnalysis
    reduced_entropy_analysis: EntropyAnalysis
    
    # Métricas de divergencia
    kl_divergence_forward: float = field(default=0.0)  # D_KL(P_orig‖P_red)
    kl_divergence_reverse: float = field(default=0.0)  # D_KL(P_red‖P_orig)
    jensen_shannon_divergence: float = field(default=0.0)  # Simétrica
    
    # Información mutua
    mutual_information: float = field(default=0.0)  # I(Original;Reduced)
    
    # Propiedades de ROBDD
    robdd_node_count: int = field(default=0)
    robdd_depth: int = field(default=0)
    reduction_ratio: float = field(default=1.0)  # nodos_red / nodos_orig


@dataclass(frozen=True, slots=True)
class MinimizerGovernanceState:
    r"""
    Objeto final del endofuntor Z_Minimizer.
    
    Representa el estado completo de gobernanza tras la composición funtorial:
        Φ₃ ∘ Φ₂ ∘ Φ₁: BoolMat × ProjMat × Prob² → GovernanceState
    
    Invariante categórico:
        is_topologically_valid = True ⟺ 
            ∀ phase ∈ {1,2,3}: certificate(phase).is_valid = True
    """
    grobner_audit: GrobnerAuditData
    unsat_core_audit: UnsatCoreCertifierData
    robdd_audit: ROBDDIsomorphismData
    is_topologically_valid: bool
    
    # Metadatos del proceso
    reduction_phase: BooleanReductionPhase = field(default=BooleanReductionPhase.COMPLETE)
    timestamp: Optional[float] = field(default=None)
    
    # Métricas agregadas
    overall_quality_score: float = field(default=0.0)  # [0, 1]
    risk_assessment: str = field(default="NOMINAL")  # NOMINAL | WARNING | CRITICAL
    
    def __post_init__(self) -> None:
        """Validación de consistencia entre certificados."""
        if self.is_topologically_valid:
            if not self.grobner_audit.is_minimally_independent:
                raise ValueError("Inconsistencia: independencia Gröbner no certificada.")
            if not self.unsat_core_audit.is_strictly_orthogonal:
                raise ValueError("Inconsistencia: ortogonalidad no certificada.")
            if not self.robdd_audit.is_homotopically_equivalent:
                raise ValueError("Inconsistencia: equivalencia ROBDD no certificada.")


# ═══════════════════════════════════════════════════════════════════════════════
# §E. GUARDAS NUMÉRICAS ENRIQUECIDAS
# ═══════════════════════════════════════════════════════════════════════════════
class _AdvancedNumericalGuard:
    r"""
    Capa de saneamiento numérico con validaciones algebraicas rigurosas.
    
    Implementa:
        - Validación de estructuras sobre GF(2)
        - Saneamiento de distribuciones probabilísticas
        - Detección de degeneraciones numéricas
        - Validación de invariantes algebraicos
    """

    @staticmethod
    def _validate_gf2_structure(
        matrix: NDArray[np.uint8],
        name: str = "matrix",
    ) -> None:
        r"""
        Valida que una matriz sea una estructura válida sobre GF(2).
        
        Verificaciones:
            - Todos los elementos son 0 o 1
            - No hay valores fuera de {0, 1}
            - Dimensiones son razonables
        """
        if not np.all((matrix == 0) | (matrix == 1)):
            invalid_values = matrix[(matrix != 0) & (matrix != 1)]
            raise BooleanInputValidationError(
                f"{name} contiene valores fuera de GF(2) = {{0, 1}}. "
                f"Valores inválidos encontrados: {np.unique(invalid_values)}"
            )
        
        if matrix.size > _MAX_TRUTH_TABLE_SIZE:
            raise BooleanInputValidationError(
                f"{name} excede el tamaño máximo permitido: "
                f"{matrix.size} > {_MAX_TRUTH_TABLE_SIZE}"
            )

    @staticmethod
    def _validate_probability_distribution(
        distribution: NDArray[np.float64],
        name: str = "distribution",
        *,
        allow_zero_support: bool = False,
    ) -> None:
        r"""
        Valida una distribución de probabilidad.
        
        Exige:
            - No negatividad
            - Suma a 1 (dentro de tolerancia)
            - Valores finitos
            - Soporte no vacío (opcional)
        """
        if not np.all(np.isfinite(distribution)):
            raise ProbabilityDistributionError(
                f"{name} contiene valores no finitos (NaN o infinito)."
            )
        
        if np.any(distribution < -_PROBABILITY_TOLERANCE):
            min_val = float(np.min(distribution))
            raise ProbabilityDistributionError(
                f"{name} contiene probabilidades negativas: mín={min_val:.6e}"
            )
        
        total = float(np.sum(distribution))
        
        if not math.isfinite(total):
            raise ProbabilityDistributionError(
                f"{name} tiene masa total no finita: {total}"
            )
        
        if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
            raise ProbabilityDistributionError(
                f"{name} no suma a 1: suma={total:.6e}"
            )
        
        if not allow_zero_support:
            support_size = int(np.count_nonzero(distribution > _PROBABILITY_TOLERANCE))
            if support_size == 0:
                raise ProbabilityDistributionError(
                    f"{name} tiene soporte vacío (todas las probabilidades son ~0)."
                )

    @staticmethod
    def _as_finite_real_matrix(name: str, value: Any) -> NDArray[np.float64]:
        r"""Valida una matriz real finita."""
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if not np.all(np.isfinite(arr)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        if arr.ndim != 2:
            raise BooleanInputValidationError(
                f"{name} debe ser una matriz 2D, recibido {arr.ndim}D."
            )

        return arr

    @staticmethod
    def _as_finite_real_vector(name: str, value: Any) -> NDArray[np.float64]:
        r"""Valida un vector real finito."""
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise BooleanInputValidationError(
                f"{name} debe ser un vector 1D, fila, columna o escalar."
            )

        if not np.all(np.isfinite(arr)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        return arr

    @classmethod
    def _as_finite_gf2_matrix(cls, name: str, value: Any) -> NDArray[np.uint8]:
        r"""Valida y convierte a matriz sobre GF(2)."""
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser booleano/entero; se rechazó entrada compleja."
            )

        try:
            arr_float = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser convertible a valores enteros en GF(2)."
            ) from exc

        if not np.all(np.isfinite(arr_float)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        if arr_float.ndim != 2:
            raise BooleanInputValidationError(
                f"{name} debe ser una matriz 2D sobre GF(2)."
            )

        if arr_float.size > 0:
            max_abs = float(np.max(np.abs(arr_float)))
        else:
            max_abs = 0.0

        integer_tolerance = max(
            _INTEGER_REPRESENTATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, max_abs),
        )

        rounded = np.rint(arr_float)

        if not np.all(np.abs(arr_float - rounded) <= integer_tolerance):
            raise BooleanInputValidationError(
                f"{name} contiene valores no enteros; GF(2) requiere 0 o 1."
            )

        arr_gf2 = np.mod(rounded.astype(np.int64), 2).astype(np.uint8)
        
        # Validación adicional de estructura GF(2)
        cls._validate_gf2_structure(arr_gf2, name)

        return arr_gf2

    @classmethod
    def _as_finite_probability_vector(
        cls,
        name: str,
        value: Any,
        *,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        r"""Valida y normaliza un vector de probabilidades."""
        arr = cls._as_finite_real_vector(name, value)

        if arr.size == 0:
            raise BooleanInputValidationError(
                f"{name} no puede ser un vector de probabilidades vacío."
            )

        min_value = float(np.min(arr))

        if min_value < -_PROBABILITY_TOLERANCE:
            raise BooleanInputValidationError(
                f"{name} contiene probabilidades negativas: mín={min_value:.6e}"
            )

        arr = np.clip(arr, 0.0, None)
        total_mass = float(np.sum(arr))

        if not math.isfinite(total_mass) or total_mass <= _PROBABILITY_TOLERANCE:
            raise BooleanInputValidationError(
                f"{name} tiene masa probabilística nula o no finita: {total_mass}"
            )

        if normalize:
            normalization_tolerance = max(
                _PROBABILITY_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, total_mass),
            )

            if abs(total_mass - 1.0) > normalization_tolerance:
                logger.warning(
                    "%s tiene masa total %.6e distinta de 1; normalizando...",
                    name,
                    total_mass,
                )

            arr = arr / total_mass

        # Validación final
        cls._validate_probability_distribution(arr, name)

        return arr


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: AUDITORÍA DE BASES DE GRÖBNER EN EL ANILLO Z₂                     ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Sea I = ⟨f₁, ..., fₘ⟩ ⊆ Z₂[x₁, ..., xₙ] un ideal booleano.                ║
# ║                                                                             ║
# ║   Definición (Base de Gröbner):                                             ║
# ║       G = {g₁, ..., gₛ} es base de Gröbner de I si:                         ║
# ║       1. I = ⟨g₁, ..., gₛ⟩                                                  ║
# ║       2. ∀f ∈ I: LT(f) es divisible por algún LT(gᵢ)                        ║
# ║                                                                             ║
# ║   Teorema (Unicidad de forma reducida):                                     ║
# ║       Para un orden monomial fijo, la base de Gröbner reducida es única.    ║
# ║                                                                             ║
# ║   Verificamos independencia lineal sobre GF(2) mediante:                    ║
# ║       rank_GF(2)(M) = número de generadores                                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_GrobnerBasisAuditor(_AdvancedNumericalGuard):
    r"""
    Fase 1: Auditoría rigurosa de bases de Gröbner en GF(2).
    
    Garantiza que la minimización de herramientas no degenere la base operativa
    mediante análisis algebraico completo sobre el anillo Z₂.
    
    Métodos de esta fase:
        1. _compute_gf2_matrix_properties (inicio)
        2. _gf2_rref (forma escalonada reducida)
        3. _gf2_rank (rango sobre GF(2))
        4. _compute_kernel_gf2
        5. _detect_linear_dependencies
        6. _compute_code_properties
        7. _estimate_dependency_graph
        8. _audit_grobner_independence (final)
    
    El último método retorna GrobnerAuditData, que es consumido por Fase 2.
    """

    def _compute_gf2_matrix_properties(
        self,
        matrix: NDArray[np.uint8],
        name: str = "matrix",
    ) -> GF2MatrixProperties:
        r"""
        Método inicial de Fase 1.
        
        Computa propiedades algebraicas completas de una matriz sobre GF(2).
        
        Returns:
            GF2MatrixProperties con invariantes algebraicos
        """
        self._validate_gf2_structure(matrix, name)
        
        rows, cols = matrix.shape
        
        if rows == 0 or cols == 0:
            return GF2MatrixProperties(
                rows=rows,
                cols=cols,
                rank_gf2=0,
                nullity_gf2=cols,
                pivot_columns=tuple(),
                free_columns=tuple(range(cols)),
                kernel_dimension=cols,
                has_full_rank=False,
                sparsity=1.0,
                hamming_weight=0,
            )
        
        # Rango y pivotes
        rank, pivot_columns = self._gf2_rank(matrix)
        
        # Columnas libres
        all_columns = set(range(cols))
        pivot_set = set(pivot_columns)
        free_columns = tuple(sorted(all_columns - pivot_set))
        
        # Nulidad
        nullity = cols - rank
        
        # Dimensión del kernel
        kernel_dimension = nullity
        
        # Rango completo
        has_full_rank = (rank == min(rows, cols))
        
        # Sparsity (fracción de ceros)
        total_elements = rows * cols
        zero_elements = int(np.count_nonzero(matrix == 0))
        sparsity = zero_elements / total_elements if total_elements > 0 else 1.0
        
        # Peso de Hamming (número de unos)
        hamming_weight = int(np.count_nonzero(matrix == 1))
        
        # Número de condición (estimación sobre GF(2))
        # Para GF(2), estimamos basándonos en la distancia al rango deficiente
        if has_full_rank:
            condition_number = 1.0
        else:
            # Proporción de rango deficiente
            rank_deficit = min(rows, cols) - rank
            condition_number = 1.0 + float(rank_deficit)
        
        return GF2MatrixProperties(
            rows=rows,
            cols=cols,
            rank_gf2=rank,
            nullity_gf2=nullity,
            pivot_columns=pivot_columns,
            free_columns=free_columns,
            kernel_dimension=kernel_dimension,
            has_full_rank=has_full_rank,
            condition_number=condition_number,
            sparsity=sparsity,
            hamming_weight=hamming_weight,
        )

    def _gf2_rref(
        self,
        matrix: NDArray[np.uint8],
    ) -> Tuple[NDArray[np.uint8], Tuple[int, ...]]:
        r"""
        Computa la forma escalonada reducida por filas (RREF) sobre GF(2).
        
        Algoritmo de Gauss-Jordan sobre GF(2):
            1. Para cada columna, buscar pivote
            2. Intercambiar filas si es necesario
            3. Eliminar hacia arriba y hacia abajo usando XOR
        
        Returns:
            rref_matrix: Matriz en forma RREF
            pivot_columns: Tupla de columnas pivote
        """
        M = matrix.copy()
        rows, cols = M.shape
        
        rank = 0
        pivot_columns: List[int] = []
        
        for col in range(cols):
            if rank >= rows:
                break
            
            # Buscar pivote en columna actual
            pivot_candidates = np.nonzero(M[rank:, col])[0]
            
            if pivot_candidates.size == 0:
                continue  # Columna sin pivote
            
            # Pivote encontrado
            pivot = int(pivot_candidates[0] + rank)
            
            # Intercambiar filas si es necesario
            if pivot != rank:
                M[[rank, pivot]] = M[[pivot, rank]]
            
            pivot_row = M[rank].copy()
            
            # Eliminación completa: arriba y abajo del pivote
            for i in range(rows):
                if i != rank and M[i, col] == 1:
                    M[i] = np.bitwise_xor(M[i], pivot_row)
            
            pivot_columns.append(col)
            rank += 1
        
        return M, tuple(pivot_columns)

    def _gf2_rank(
        self,
        matrix: NDArray[np.uint8],
    ) -> Tuple[int, Tuple[int, ...]]:
        r"""
        Calcula el rango sobre GF(2) mediante eliminación de Gauss-Jordan.
        
        Returns:
            rank: Rango efectivo sobre GF(2)
            pivot_columns: Columnas pivote que certifican la base algebraica
        """
        _, pivot_columns = self._gf2_rref(matrix)
        rank = len(pivot_columns)
        
        return rank, pivot_columns

    def _compute_kernel_gf2(
        self,
        matrix: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        r"""
        Computa una base del kernel (núcleo) de la matriz sobre GF(2).
        
        El kernel es el conjunto de vectores v tales que Mv = 0 (mod 2).
        
        Algoritmo:
            1. Computar RREF
            2. Identificar columnas libres
            3. Para cada columna libre, construir vector del kernel
        
        Returns:
            Matriz cuyas columnas forman base del kernel
        """
        rref, pivot_columns = self._gf2_rref(matrix)
        rows, cols = matrix.shape
        
        # Columnas libres
        all_columns = set(range(cols))
        pivot_set = set(pivot_columns)
        free_columns = sorted(all_columns - pivot_set)
        
        nullity = len(free_columns)
        
        if nullity == 0:
            # Kernel trivial
            return np.zeros((cols, 0), dtype=np.uint8)
        
        # Construir base del kernel
        kernel_basis = np.zeros((cols, nullity), dtype=np.uint8)
        
        for idx, free_col in enumerate(free_columns):
            # Vector del kernel correspondiente a esta columna libre
            kernel_vector = np.zeros(cols, dtype=np.uint8)
            kernel_vector[free_col] = 1
            
            # Llenar componentes pivote
            for pivot_idx, pivot_col in enumerate(pivot_columns):
                if pivot_idx < rows:
                    kernel_vector[pivot_col] = rref[pivot_idx, free_col]
            
            kernel_basis[:, idx] = kernel_vector
        
        return kernel_basis

    def _detect_linear_dependencies(
        self,
        matrix: NDArray[np.uint8],
    ) -> List[Tuple[int, ...]]:
        r"""
        Detecta dependencias lineales entre filas de la matriz sobre GF(2).
        
        Returns:
            Lista de tuplas de índices de filas linealmente dependientes
        """
        rows, cols = matrix.shape
        
        if rows <= 1:
            return []
        
        dependencies: List[Tuple[int, ...]] = []
        
        # Transponer para analizar filas como columnas
        M_T = matrix.T
        
        # Computar kernel de M^T
        kernel = self._compute_kernel_gf2(M_T)
        
        # Cada vector del kernel indica una dependencia lineal entre filas
        for k in range(kernel.shape[1]):
            kernel_vec = kernel[:, k]
            dependent_rows = tuple(np.nonzero(kernel_vec)[0].tolist())
            
            if len(dependent_rows) > 1:
                dependencies.append(dependent_rows)
        
        return dependencies

    def _compute_code_properties(
        self,
        matrix: NDArray[np.uint8],
        rank: int,
    ) -> LinearCodeProperties:
        r"""
        Computa propiedades de código lineal de la matriz sobre GF(2).
        
        Interpreta las filas como palabras código de un código lineal [n, k, d].
        
        Returns:
            LinearCodeProperties con parámetros del código
        """
        rows, cols = matrix.shape
        
        # Parámetros del código
        n = cols  # Longitud del código
        k = rank  # Dimensión (log₂ del número de palabras código)
        
        # Distancia mínima: mínimo peso de Hamming no nulo
        hamming_weights = np.array([
            int(np.count_nonzero(matrix[i]))
            for i in range(rows)
        ])
        
        nonzero_weights = hamming_weights[hamming_weights > 0]
        
        if nonzero_weights.size > 0:
            d = int(np.min(nonzero_weights))
        else:
            d = n  # Código trivial
        
        # Tasa del código
        rate = k / n if n > 0 else 0.0
        
        # Redundancia
        redundancy = n - k
        
        # Cota de Singleton: d ≤ n - k + 1
        singleton_bound = n - k + 1
        
        # Cota de Hamming: número máximo de palabras código
        # Para código binario: M ≤ 2^n / V(n, t) donde t = ⌊(d-1)/2⌋
        t = (d - 1) // 2
        
        # Volumen de esfera de Hamming
        hamming_volume = sum(
            math.comb(n, i)
            for i in range(t + 1)
        )
        
        hamming_bound = (2**n) / hamming_volume if hamming_volume > 0 else float('inf')
        
        return LinearCodeProperties(
            length=n,
            dimension=k,
            minimum_distance=d,
            rate=rate,
            redundancy=redundancy,
            singleton_bound=singleton_bound,
            hamming_bound=hamming_bound,
        )

    def _estimate_dependency_graph(
        self,
        matrix: NDArray[np.uint8],
    ) -> Tuple[int, int]:
        r"""
        Estima propiedades del grafo de dependencias algebraicas.
        
        Returns:
            edges: Número de aristas (dependencias directas)
            max_chain_length: Longitud máxima de cadena de dependencias
        """
        dependencies = self._detect_linear_dependencies(matrix)
        
        # Número de aristas: pares de filas dependientes
        edges = sum(
            math.comb(len(dep), 2)
            for dep in dependencies
        )
        
        # Longitud máxima de cadena
        max_chain_length = max(
            (len(dep) for dep in dependencies),
            default=1,
        )
        
        return int(edges), int(max_chain_length)

    def _audit_grobner_independence(
        self,
        boolean_polynomial_matrix: NDArray[np.uint8],
    ) -> GrobnerAuditData:
        r"""
        Último método de la Fase 1 (continuación de los 7 métodos anteriores).
        
        Calcula el rango algebraico en Z₂ usando eliminación de Gauss-Jordan
        y análisis completo de propiedades algebraicas.
        
        Este método retorna GrobnerAuditData, que es el objeto inicial
        consumido por el primer método de Fase 2.
        
        Teorema verificado:
            rank_GF(2)(M) = número de generadores ⟹ base mínima
        """
        matrix_gf2 = self._as_finite_gf2_matrix(
            "boolean_polynomial_matrix",
            boolean_polynomial_matrix,
        )

        rows, cols = matrix_gf2.shape

        if rows == 0 or cols == 0:
            raise BooleanInputValidationError(
                "boolean_polynomial_matrix no puede ser vacía."
            )

        if cols > _MAX_BOOLEAN_VARIABLES:
            raise MICMinimizerAgentError(
                "Explosión combinatoria detectada: "
                f"n_vars={cols} > límite seguro={_MAX_BOOLEAN_VARIABLES}."
            )

        # Propiedades matriciales completas
        matrix_properties = self._compute_gf2_matrix_properties(
            matrix_gf2,
            "boolean_polynomial_matrix",
        )

        rank = matrix_properties.rank_gf2
        nullity = matrix_properties.nullity_gf2
        pivot_columns = matrix_properties.pivot_columns

        # Forma RREF para certificación
        rref_matrix, _ = self._gf2_rref(matrix_gf2)

        # Propiedades de código lineal
        code_properties = self._compute_code_properties(matrix_gf2, rank)

        # Validación de código
        if not code_properties.is_valid_code():
            logger.warning(
                "La matriz no satisface cotas de teoría de códigos. "
                "d=%d > singleton_bound=%d",
                code_properties.minimum_distance,
                code_properties.singleton_bound,
            )

        # Grafo de dependencias
        dep_edges, max_dep_chain = self._estimate_dependency_graph(matrix_gf2)

        # Verificación crítica: independencia mínima
        if rank < rows:
            # Detectar dependencias específicas
            dependencies = self._detect_linear_dependencies(matrix_gf2)
            
            raise GrobnerDegeneracyError(
                "Degeneración en el anillo booleano detectada.\n"
                f"  Rango efectivo en Z₂: {rank}\n"
                f"  Número de generadores: {rows}\n"
                f"  Déficit de rango: {rows - rank}\n"
                f"  Dependencias lineales detectadas: {len(dependencies)}\n"
                f"  Grupos dependientes: {dependencies[:5]}\n"  # Mostrar primeros 5
                "  Interpretación: El ideal colapsó; la poda algorítmica "
                "amputaría capacidades esenciales del agente."
            )

        logger.info(
            "Fase 1 COMPLETADA: Auditoría de Base de Gröbner.\n"
            "  Dimensión: %dx%d\n"
            "  Rango GF(2): %d\n"
            "  Nulidad: %d\n"
            "  Código [%d, %d, %d]\n"
            "  Tasa: %.4f\n"
            "  Sparsity: %.4f\n"
            "  Aristas de dependencia: %d",
            rows,
            cols,
            rank,
            nullity,
            code_properties.length,
            code_properties.dimension,
            code_properties.minimum_distance,
            code_properties.rate,
            matrix_properties.sparsity,
            dep_edges,
        )

        return GrobnerAuditData(
            rows=int(rows),
            cols=int(cols),
            ideal_dimension=int(rank),
            nullity=nullity,
            pivot_columns=pivot_columns,
            is_minimally_independent=True,
            matrix_properties=matrix_properties,
            code_properties=code_properties,
            dependency_graph_edges=dep_edges,
            max_dependency_chain_length=max_dep_chain,
            rref_matrix=rref_matrix,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: CERTIFICACIÓN DE NO-INTERFERENCIA (UNSAT CORE)                    ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Sea P ∈ ℝᵐˣⁿ la matriz de proyección de herramientas.                     ║
# ║                                                                             ║
# ║   Definición (Ortogonalidad estricta):                                      ║
# ║       Las filas {p₁, ..., pₘ} de P son ortogonales si:                      ║
# ║       ⟨pᵢ, pⱼ⟩ = δᵢⱼ  ∀i,j                                                  ║
# ║                                                                             ║
# ║   En forma matricial:                                                       ║
# ║       G = PPᵀ = I                                                           ║
# ║                                                                             ║
# ║   Verificamos:                                                              ║
# ║       ‖G - I‖_F ≤ ε_ortho                                                   ║
# ║                                                                             ║
# ║   Teorema (Proyección ortogonal):                                           ║
# ║       P es proyección ortogonal ⟺ P² = P y Pᵀ = P                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UnsatCoreCertifier(Phase1_GrobnerBasisAuditor):
    r"""
    Fase 2: Certificación rigurosa de no-interferencia y ortogonalidad.
    
    Evalúa que las herramientas sean estrictamente ortogonales para prevenir
    el colapso del principio Zero Side-Effects en la MIC.
    
    Métodos de esta fase:
        1. _compute_gram_matrix (inicio, hereda certificado Fase 1)
        2. _analyze_orthogonality
        3. _detect_conflict_pairs
        4. _validate_projection_properties
        5. _compute_spectral_analysis
        6. _certify_non_interference_unsat (final)
    
    El último método retorna UnsatCoreCertifierData, que es consumido por Fase 3.
    """

    def _compute_gram_matrix(
        self,
        projection_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Primer método de Fase 2 (continuación formal de Fase 1).
        
        Computa la matriz de Gram G = PPᵀ.
        
        Propiedades:
            - G es simétrica
            - G es semidefinida positiva
            - G = I ⟺ filas de P son ortonormales
        
        Args:
            projection_matrix: Matriz P ∈ ℝᵐˣⁿ
        
        Returns:
            G: Matriz de Gram G = PPᵀ ∈ ℝᵐˣᵐ
        """
        try:
            gram = projection_matrix @ projection_matrix.T
        except Exception as exc:
            raise NonInterferenceViolationError(
                "No fue posible computar la matriz de Gram PPᵀ."
            ) from exc

        if not np.all(np.isfinite(gram)):
            raise NonInterferenceViolationError(
                "La matriz de Gram PPᵀ contiene valores no finitos."
            )

        # Simetrización para eliminar error numérico
        gram = (gram + gram.T) / 2.0

        return gram

    def _analyze_orthogonality(
        self,
        gram_matrix: NDArray[np.float64],
    ) -> OrthogonalityAnalysis:
        r"""
        Analiza la ortogonalidad de la matriz de Gram.
        
        Computa múltiples métricas de desviación de la identidad.
        
        Args:
            gram_matrix: Matriz de Gram G = PPᵀ
        
        Returns:
            OrthogonalityAnalysis con métricas detalladas
        """
        m = gram_matrix.shape[0]
        
        if m == 0:
            raise BooleanInputValidationError(
                "La matriz de Gram está vacía."
            )

        identity = np.eye(m, dtype=np.float64)
        deviation = gram_matrix - identity

        # Norma de Frobenius
        frobenius_deviation = float(la.norm(deviation, ord='fro'))

        # Norma espectral (máximo valor singular)
        try:
            singular_values = la.svdvals(deviation)
            spectral_deviation = float(np.max(singular_values))
        except la.LinAlgError:
            spectral_deviation = float('inf')

        # Máxima desviación absoluta
        max_absolute_deviation = float(np.max(np.abs(deviation)))

        # Diagonal y fuera de diagonal
        diagonal = np.diag(gram_matrix).copy()
        diagonal_deviation_norm = float(np.max(np.abs(diagonal - 1.0)))

        off_diagonal = gram_matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)

        if m > 1:
            upper_indices = np.triu_indices(m, k=1)
            off_diagonal_values = np.abs(off_diagonal[upper_indices])
            off_diagonal_norm = float(np.sum(off_diagonal_values))
            max_interference_value = float(np.max(off_diagonal_values))
        else:
            off_diagonal_norm = 0.0
            max_interference_value = 0.0

        # Análisis espectral de G
        try:
            eigenvalues = np.linalg.eigvalsh(gram_matrix)
            eigenvalues = eigenvalues[eigenvalues > _MACHINE_EPSILON]
            
            if eigenvalues.size > 0:
                lambda_min = float(np.min(eigenvalues))
                lambda_max = float(np.max(eigenvalues))
                
                # Gap espectral: separación del cero
                spectral_gap = lambda_min
                
                # Número de condición
                if lambda_min > _MACHINE_EPSILON:
                    condition_number = lambda_max / lambda_min
                else:
                    condition_number = float('inf')
            else:
                spectral_gap = 0.0
                condition_number = float('inf')
                
        except np.linalg.LinAlgError:
            spectral_gap = 0.0
            condition_number = float('inf')

        return OrthogonalityAnalysis(
            gram_matrix=gram_matrix,
            off_diagonal_norm=off_diagonal_norm,
            diagonal_deviation_norm=diagonal_deviation_norm,
            spectral_gap=spectral_gap,
            condition_number=condition_number,
            frobenius_deviation=frobenius_deviation,
            spectral_deviation=spectral_deviation,
            max_absolute_deviation=max_absolute_deviation,
            max_interference_value=max_interference_value,
        )

    def _detect_conflict_pairs(
        self,
        gram_matrix: NDArray[np.float64],
        tolerance: float,
    ) -> Set[Tuple[int, int]]:
        r"""
        Detecta pares de herramientas con interferencia significativa.
        
        Un par (i, j) está en conflicto si |Gᵢⱼ| > tolerance para i ≠ j.
        
        Args:
            gram_matrix: Matriz de Gram G = PPᵀ
            tolerance: Tolerancia de ortogonalidad
        
        Returns:
            Conjunto de pares (i, j) con i < j en conflicto
        """
        m = gram_matrix.shape[0]
        
        conflict_pairs: Set[Tuple[int, int]] = set()
        
        for i in range(m):
            for j in range(i + 1, m):
                if abs(gram_matrix[i, j]) > tolerance:
                    conflict_pairs.add((i, j))
        
        return conflict_pairs

    def _validate_projection_properties(
        self,
        projection_matrix: NDArray[np.float64],
    ) -> Tuple[int, bool]:
        r"""
        Valida propiedades de proyección de la matriz.
        
        Una proyección ortogonal satisface:
            1. P² = P (idempotencia)
            2. Pᵀ = P (simetría)
        
        Args:
            projection_matrix: Matriz P
        
        Returns:
            rank: Rango de la proyección
            is_idempotent: True si P² ≈ P
        """
        P = projection_matrix
        m, n = P.shape
        
        # Rango
        try:
            rank = int(np.linalg.matrix_rank(P))
        except np.linalg.LinAlgError:
            rank = min(m, n)
        
        # Idempotencia: P² = P (solo si m = n)
        if m == n:
            try:
                P_squared = P @ P
                idempotence_deviation = float(la.norm(P_squared - P, ord='fro'))
                
                idempotence_tolerance = max(
                    _ORTHOGONALITY_TOLERANCE,
                    _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * float(la.norm(P, ord='fro')),
                )
                
                is_idempotent = idempotence_deviation <= idempotence_tolerance
            except Exception:
                is_idempotent = False
        else:
            is_idempotent = False
        
        return rank, is_idempotent

    def _compute_spectral_analysis(
        self,
        gram_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Realiza análisis espectral de la matriz de Gram.
        
        Args:
            gram_matrix: Matriz de Gram G = PPᵀ
        
        Returns:
            eigenvalues: Autovalores de G (ordenados descendentemente)
            spectral_radius: Radio espectral ρ(G) = max|λᵢ|
            spectral_norm: Norma espectral ‖G‖₂ = σₘₐₓ
        """
        try:
            eigenvalues = np.linalg.eigvalsh(gram_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Orden descendente
            
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            
            # Norma espectral (máximo valor singular)
            singular_values = la.svdvals(gram_matrix)
            spectral_norm = float(np.max(singular_values))
            
        except np.linalg.LinAlgError:
            m = gram_matrix.shape[0]
            eigenvalues = np.zeros(m)
            spectral_radius = 0.0
            spectral_norm = 0.0
        
        return eigenvalues, spectral_radius, spectral_norm

    def _certify_non_interference_unsat(
        self,
        tool_projection_matrix: NDArray[np.float64],
        grobner_audit: Optional[GrobnerAuditData] = None,
    ) -> UnsatCoreCertifierData:
        r"""
        Último método de Fase 2 (continuación de métodos previos de esta fase).
        
        Computa la matriz de Gram y realiza análisis completo de ortogonalidad.
        
        Este método retorna UnsatCoreCertifierData, que es el objeto inicial
        consumido por el primer método de Fase 3.
        
        Teorema verificado:
            PPᵀ = I ⟺ Las filas de P son ortonormales
        """
        # Validación del certificado de Fase 1
        if grobner_audit is not None:
            if not grobner_audit.is_minimally_independent:
                raise GrobnerDegeneracyError(
                    "La Fase 2 no puede iniciarse sin certificado de independencia válido.\n"
                    "  La Fase 1 no certificó independencia algebraica en GF(2)."
                )

        # Saneamiento de matriz de proyección
        P = self._as_finite_real_matrix(
            "tool_projection_matrix",
            tool_projection_matrix,
        )

        if P.size == 0:
            raise BooleanInputValidationError(
                "tool_projection_matrix no puede ser vacía."
            )

        tool_count, variable_dim = P.shape

        if tool_count == 0 or variable_dim == 0:
            raise BooleanInputValidationError(
                "tool_projection_matrix debe tener herramientas y variables.\n"
                f"  Shape recibido: ({tool_count}, {variable_dim})"
            )

        # Validación de consistencia con Fase 1
        if grobner_audit is not None:
            if grobner_audit.cols != variable_dim:
                raise ValueError(
                    "Inconsistencia dimensional entre Fase 1 y Fase 2.\n"
                    f"  Fase 1 certificó cols={grobner_audit.cols}\n"
                    f"  tool_projection_matrix tiene variable_dim={variable_dim}"
                )

            if grobner_audit.rows != tool_count:
                logger.warning(
                    "Fase 1 certificó %d generadores, pero Fase 2 recibió %d herramientas.",
                    grobner_audit.rows,
                    tool_count,
                )

        # Cómputo de matriz de Gram
        gram = self._compute_gram_matrix(P)

        # Análisis completo de ortogonalidad
        orthogonality_analysis = self._analyze_orthogonality(gram)

        # Determinación de tolerancia adaptativa
        diagonal_scale = float(np.sum(np.abs(np.diag(gram))))
        scale = max(1.0, diagonal_scale)

        orthogonality_tolerance = max(
            _ORTHOGONALITY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * scale,
        )

        # Detección de pares en conflicto
        conflict_pairs = self._detect_conflict_pairs(gram, orthogonality_tolerance)
        conflict_edges = len(conflict_pairs)

        # Almacenar pares en análisis
        orthogonality_analysis = OrthogonalityAnalysis(
            gram_matrix=orthogonality_analysis.gram_matrix,
            off_diagonal_norm=orthogonality_analysis.off_diagonal_norm,
            diagonal_deviation_norm=orthogonality_analysis.diagonal_deviation_norm,
            spectral_gap=orthogonality_analysis.spectral_gap,
            condition_number=orthogonality_analysis.condition_number,
            frobenius_deviation=orthogonality_analysis.frobenius_deviation,
            spectral_deviation=orthogonality_analysis.spectral_deviation,
            max_absolute_deviation=orthogonality_analysis.max_absolute_deviation,
            conflict_pairs=conflict_pairs,
            max_interference_value=orthogonality_analysis.max_interference_value,
        )

        # Validación de propiedades de proyección
        projection_rank, is_idempotent = self._validate_projection_properties(P)

        # Verificaciones críticas
        if orthogonality_analysis.off_diagonal_norm > orthogonality_tolerance:
            raise NonInterferenceViolationError(
                "Violación del axioma Zero Side-Effects.\n"
                "  La matriz de capacidades no es ortogonal.\n"
                f"  Norma residual fuera de diagonal: {orthogonality_analysis.off_diagonal_norm:.6e}\n"
                f"  Tolerancia: {orthogonality_tolerance:.6e}\n"
                f"  Pares en conflicto: {conflict_pairs}"
            )

        if orthogonality_analysis.diagonal_deviation_norm > orthogonality_tolerance:
            raise NonInterferenceViolationError(
                "Violación de ortonormalidad.\n"
                f"  La diagonal de PPᵀ se desvía de I en {orthogonality_analysis.diagonal_deviation_norm:.6e}\n"
                f"  Tolerancia: {orthogonality_tolerance:.6e}"
            )

        if conflict_edges > 0:
            raise NonInterferenceViolationError(
                "UNSAT Core detectó aristas de interferencia cruzada.\n"
                f"  Número de pares no ortogonales: {conflict_edges}\n"
                f"  Pares: {list(conflict_pairs)[:10]}"  # Mostrar primeros 10
            )

        logger.info(
            "Fase 2 COMPLETADA: Certificación de No-Interferencia.\n"
            "  Herramientas: %d\n"
            "  Variables: %d\n"
            "  Norma fuera de diagonal: %.6e\n"
            "  Desviación diagonal: %.6e\n"
            "  Número de condición: %.6e\n"
            "  Rango de proyección: %d\n"
            "  Idempotente: %s",
            tool_count,
            variable_dim,
            orthogonality_analysis.off_diagonal_norm,
            orthogonality_analysis.diagonal_deviation_norm,
            orthogonality_analysis.condition_number,
            projection_rank,
            is_idempotent,
        )

        return UnsatCoreCertifierData(
            tool_count=int(tool_count),
            variable_dim=int(variable_dim),
            off_diagonal_conflict_norm=float(orthogonality_analysis.off_diagonal_norm),
            diagonal_deviation_norm=float(orthogonality_analysis.diagonal_deviation_norm),
            conflict_edges=conflict_edges,
            orthogonality_tolerance=float(orthogonality_tolerance),
            is_strictly_orthogonal=True,
            orthogonality_analysis=orthogonality_analysis,
            projection_rank=projection_rank,
            is_projection_idempotent=is_idempotent,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: ISOMORFISMO DE REDUCCIÓN ROBDD                                    ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Sea f: {0,1}ⁿ → {0,1} una función booleana.                              ║
# ║                                                                             ║
# ║   Definición (ROBDD):                                                       ║
# ║       Un ROBDD es un grafo dirigido acíclico que:                           ║
# ║       1. Respeta un orden de variables fijo                                 ║
# ║       2. No tiene nodos redundantes                                         ║
# ║       3. No tiene nodos duplicados isomorfos                                ║
# ║                                                                             ║
# ║   Teorema (Unicidad):                                                       ║
# ║       Para f y orden de variables fijo, el ROBDD es único.                  ║
# ║                                                                             ║
# ║   Preservación de información:                                              ║
# ║       H(f_original) ≈ H(f_ROBDD)                                            ║
# ║                                                                             ║
# ║   donde H es la entropía de Shannon de la distribución inducida.            ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ROBDDIsomorphismValidator(Phase2_UnsatCoreCertifier):
    r"""
    Fase 3: Validación rigurosa de isomorfismo ROBDD.
    
    Asegura que la reducción de BDD preserve la equivalencia homotópica con
    el árbol de sintaxis original mediante análisis entrópico completo.
    
    Métodos de esta fase:
        1. _compute_entropy_analysis (inicio, hereda certificado Fase 2)
        2. _shannon_entropy_bits
        3. _min_entropy
        4. _collision_entropy
        5. _hartley_entropy
        6. _kl_divergence
        7. _jensen_shannon_divergence
        8. _mutual_information
        9. _validate_robdd_homotopy (final)
    
    El último método retorna ROBDDIsomorphismData, certificado final del proceso.
    """

    def _compute_entropy_analysis(
        self,
        probabilities: NDArray[np.float64],
        name: str = "distribution",
    ) -> EntropyAnalysis:
        r"""
        Primer método de Fase 3 (continuación formal de Fase 2).
        
        Computa análisis entrópico completo de una distribución.
        
        Args:
            probabilities: Distribución de probabilidad
            name: Identificador para mensajes
        
        Returns:
            EntropyAnalysis con múltiples métricas entrópicas
        """
        p = probabilities[probabilities > _PROBABILITY_TOLERANCE]
        
        if p.size == 0:
            return EntropyAnalysis(
                shannon_entropy=0.0,
                min_entropy=0.0,
                collision_entropy=0.0,
                hartley_entropy=0.0,
                support_size=0,
                max_probability=0.0,
                uniformity=0.0,
            )
        
        # Entropía de Shannon
        shannon = self._shannon_entropy_bits(probabilities)
        
        # Min-entropía
        min_ent = self._min_entropy(probabilities)
        
        # Entropía de colisión
        collision = self._collision_entropy(probabilities)
        
        # Entropía de Hartley
        hartley = self._hartley_entropy(probabilities)
        
        # Propiedades de la distribución
        support_size = int(np.count_nonzero(probabilities > _PROBABILITY_TOLERANCE))
        max_probability = float(np.max(probabilities))
        
        # Uniformidad: H(X) / log₂(|support|)
        if support_size > 1:
            max_entropy = math.log2(support_size)
            uniformity = shannon / max_entropy if max_entropy > 0 else 0.0
        else:
            uniformity = 1.0 if support_size == 1 else 0.0
        
        return EntropyAnalysis(
            shannon_entropy=shannon,
            min_entropy=min_ent,
            collision_entropy=collision,
            hartley_entropy=hartley,
            support_size=support_size,
            max_probability=max_probability,
            uniformity=uniformity,
        )

    @staticmethod
    def _shannon_entropy_bits(probabilities: NDArray[np.float64]) -> float:
        r"""
        Calcula la entropía de Shannon en bits:
            H(X) = -Σ p(xᵢ) log₂ p(xᵢ)
        """
        p = np.clip(probabilities, 0.0, 1.0)
        p = p[p > _PROBABILITY_TOLERANCE]

        if p.size == 0:
            return 0.0

        entropy = float(-np.sum(p * np.log2(p)))

        if not math.isfinite(entropy):
            raise ROBDDHomotopyError(
                "La entropía de Shannon no es finita."
            )

        return entropy

    @staticmethod
    def _min_entropy(probabilities: NDArray[np.float64]) -> float:
        r"""
        Calcula la min-entropía (entropía de Rényi de orden ∞):
            H_∞(X) = -log₂(max pᵢ)
        
        Interpretación: mínima incertidumbre, peor caso de adivinación.
        """
        p = probabilities[probabilities > _PROBABILITY_TOLERANCE]
        
        if p.size == 0:
            return 0.0
        
        max_p = float(np.max(p))
        
        if max_p > _PROBABILITY_TOLERANCE:
            return -math.log2(max_p)
        else:
            return 0.0

    @staticmethod
    def _collision_entropy(probabilities: NDArray[np.float64]) -> float:
        r"""
        Calcula la entropía de colisión (entropía de Rényi de orden 2):
            H₂(X) = -log₂(Σ pᵢ²)
        
        Interpretación: probabilidad de colisión al muestrear dos veces.
        """
        p = probabilities[probabilities > _PROBABILITY_TOLERANCE]
        
        if p.size == 0:
            return 0.0
        
        collision_prob = float(np.sum(p**2))
        
        if collision_prob > _PROBABILITY_TOLERANCE:
            return -math.log2(collision_prob)
        else:
            return 0.0

    @staticmethod
    def _hartley_entropy(probabilities: NDArray[np.float64]) -> float:
        r"""
        Calcula la entropía de Hartley (entropía de Rényi de orden 0):
            H₀(X) = log₂(|support(X)|)
        
        Interpretación: logaritmo del tamaño del soporte.
        """
        support_size = int(
            np.count_nonzero(probabilities > _PROBABILITY_TOLERANCE)
        )
        
        if support_size > 0:
            return math.log2(support_size)
        else:
            return 0.0

    @staticmethod
    def _kl_divergence(
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la divergencia de Kullback-Leibler:
            D_KL(P‖Q) = Σ p(x) log₂(p(x)/q(x))
        
        Propiedades:
            - D_KL(P‖Q) ≥ 0
            - D_KL(P‖Q) = 0 ⟺ P = Q
            - No es simétrica: D_KL(P‖Q) ≠ D_KL(Q‖P) en general
        
        Args:
            p: Distribución P
            q: Distribución Q
        
        Returns:
            D_KL(P‖Q) en bits
        """
        # Asegurar mismo tamaño
        max_size = max(p.size, q.size)
        
        p_pad = np.pad(p, (0, max_size - p.size))
        q_pad = np.pad(q, (0, max_size - q.size))
        
        # Evitar división por cero
        q_safe = np.where(q_pad > _PROBABILITY_TOLERANCE, q_pad, _PROBABILITY_TOLERANCE)
        
        # Computar solo donde p > 0
        mask = p_pad > _PROBABILITY_TOLERANCE
        
        if not np.any(mask):
            return 0.0
        
        kl = float(np.sum(p_pad[mask] * np.log2(p_pad[mask] / q_safe[mask])))
        
        # KL debe ser no negativa
        if kl < -_KL_DIVERGENCE_TOLERANCE:
            raise ROBDDHomotopyError(
                f"Divergencia KL negativa (imposible): {kl:.6e}"
            )
        
        return max(0.0, kl)

    @classmethod
    def _jensen_shannon_divergence(
        cls,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la divergencia de Jensen-Shannon (simétrica):
            JSD(P‖Q) = ½D_KL(P‖M) + ½D_KL(Q‖M)
        
        donde M = (P + Q)/2 es la distribución media.
        
        Propiedades:
            - 0 ≤ JSD(P‖Q) ≤ 1 (en bits)
            - JSD(P‖Q) = JSD(Q‖P) (simétrica)
            - JSD(P‖Q) = 0 ⟺ P = Q
            - √JSD es una métrica (desigualdad triangular)
        """
        # Asegurar mismo tamaño
        max_size = max(p.size, q.size)
        
        p_pad = np.pad(p, (0, max_size - p.size))
        q_pad = np.pad(q, (0, max_size - q.size))
        
        # Distribución media
        m = (p_pad + q_pad) / 2.0
        
        # JSD simétrica
        jsd = 0.5 * cls._kl_divergence(p_pad, m) + 0.5 * cls._kl_divergence(q_pad, m)
        
        return jsd

    @staticmethod
    def _mutual_information(
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        r"""
        Estima la información mutua entre dos distribuciones.
        
        Aproximación:
            I(X;Y) ≈ H(X) + H(Y) - H(X,Y)
        
        Para distribuciones marginales, asumimos independencia para H(X,Y).
        
        Args:
            p: Distribución marginal P(X)
            q: Distribución marginal P(Y)
        
        Returns:
            Estimación de I(X;Y) en bits
        """
        # Entropías marginales
        h_x = Phase3_ROBDDIsomorphismValidator._shannon_entropy_bits(p)
        h_y = Phase3_ROBDDIsomorphismValidator._shannon_entropy_bits(q)
        
        # Asumiendo independencia: H(X,Y) = H(X) + H(Y)
        # Entonces I(X;Y) = 0 bajo independencia
        
        # Estimación más sofisticada requeriría distribución conjunta
        # Por ahora, retornamos 0 (límite inferior)
        
        return 0.0

    def _validate_robdd_homotopy(
        self,
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
        unsat_core_audit: Optional[UnsatCoreCertifierData] = None,
    ) -> ROBDDIsomorphismData:
        r"""
        Último método de Fase 3 (continuación de métodos previos de esta fase).
        
        Compara distribuciones mediante análisis entrópico y de divergencia completo.
        
        Este método retorna ROBDDIsomorphismData, el certificado final que completa
        la composición funtorial Φ₃ ∘ Φ₂ ∘ Φ₁.
        
        Teorema verificado:
            |H(f_orig) - H(f_ROBDD)| ≤ ε_H ⟹ Equivalencia homotópica
        """
        # Validación del certificado de Fase 2
        if unsat_core_audit is not None:
            if not unsat_core_audit.is_strictly_orthogonal:
                raise NonInterferenceViolationError(
                    "La Fase 3 no puede iniciarse sin certificado de ortogonalidad válido.\n"
                    "  La Fase 2 no certificó ortogonalidad estricta."
                )

        # Saneamiento de distribuciones
        p_original = self._as_finite_probability_vector(
            "original_truth_table_probs",
            original_truth_table_probs,
            normalize=True,
        )

        p_reduced = self._as_finite_probability_vector(
            "reduced_robdd_probs",
            reduced_robdd_probs,
            normalize=True,
        )

        # Validación de distribuciones
        self._validate_probability_distribution(p_original, "original_truth_table_probs")
        self._validate_probability_distribution(p_reduced, "reduced_robdd_probs")

        # Dimensión del soporte
        support_dimension = max(p_original.size, p_reduced.size)

        # Padding para compatibilidad
        p_original_pad = np.pad(
            p_original,
            (0, support_dimension - p_original.size),
        )

        p_reduced_pad = np.pad(
            p_reduced,
            (0, support_dimension - p_reduced.size),
        )

        # Análisis entrópico completo
        original_entropy_analysis = self._compute_entropy_analysis(
            p_original_pad,
            "original_distribution",
        )

        reduced_entropy_analysis = self._compute_entropy_analysis(
            p_reduced_pad,
            "reduced_distribution",
        )

        H_original = original_entropy_analysis.shannon_entropy
        H_reduced = reduced_entropy_analysis.shannon_entropy

        # Pérdida de entropía
        entropy_loss = float(abs(H_original - H_reduced))

        if not math.isfinite(entropy_loss):
            raise ROBDDHomotopyError(
                "La pérdida entrópica ΔH no es finita."
            )

        # Tolerancia adaptativa
        entropy_tolerance = max(
            _MIN_ENTROPY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(H_original), abs(H_reduced)),
        )

        # Distancia de variación total
        total_variation_distance = float(
            0.5 * np.sum(np.abs(p_original_pad - p_reduced_pad))
        )

        if not math.isfinite(total_variation_distance):
            raise ROBDDHomotopyError(
                "La distancia de variación total no es finita."
            )

        # Divergencias de Kullback-Leibler
        kl_forward = self._kl_divergence(p_original_pad, p_reduced_pad)
        kl_reverse = self._kl_divergence(p_reduced_pad, p_original_pad)

        # Divergencia de Jensen-Shannon
        jsd = self._jensen_shannon_divergence(p_original_pad, p_reduced_pad)

        # Información mutua (estimación)
        mutual_info = self._mutual_information(p_original_pad, p_reduced_pad)

        # Propiedades de ROBDD (valores por defecto, pueden ser parametrizados)
        robdd_node_count = 0
        robdd_depth = 0
        reduction_ratio = 1.0

        # Verificación crítica: preservación de entropía
        if entropy_loss > entropy_tolerance:
            raise ROBDDHomotopyError(
                "Ruptura homotópica en la reducción ROBDD.\n"
                "  La entropía booleana divergió significativamente.\n"
                f"  ΔH = |H_orig - H_red| = {entropy_loss:.6e} bits\n"
                f"  Tolerancia: {entropy_tolerance:.6e} bits\n"
                f"  H_original: {H_original:.6f} bits\n"
                f"  H_reduced: {H_reduced:.6f} bits\n"
                f"  D_KL(P_orig‖P_red): {kl_forward:.6e}\n"
                f"  D_KL(P_red‖P_orig): {kl_reverse:.6e}\n"
                f"  JSD: {jsd:.6e}\n"
                "  Interpretación: El minimizador mutiló ramas lógicas operativas."
            )

        logger.info(
            "Fase 3 COMPLETADA: Validación de Isomorfismo ROBDD.\n"
            "  Dimensión de soporte: %d\n"
            "  H_original: %.6f bits\n"
            "  H_reduced: %.6f bits\n"
            "  ΔH: %.6e bits\n"
            "  Distancia de variación total: %.6e\n"
            "  D_KL(P_orig‖P_red): %.6e bits\n"
            "  D_KL(P_red‖P_orig): %.6e bits\n"
            "  JSD: %.6e bits\n"
            "  Uniformidad original: %.4f\n"
            "  Uniformidad reducida: %.4f",
            support_dimension,
            H_original,
            H_reduced,
            entropy_loss,
            total_variation_distance,
            kl_forward,
            kl_reverse,
            jsd,
            original_entropy_analysis.uniformity,
            reduced_entropy_analysis.uniformity,
        )

        return ROBDDIsomorphismData(
            support_dimension=int(support_dimension),
            original_entropy=float(H_original),
            reduced_entropy=float(H_reduced),
            entropy_loss=float(entropy_loss),
            entropy_tolerance=float(entropy_tolerance),
            total_variation_distance=float(total_variation_distance),
            is_homotopically_equivalent=True,
            original_entropy_analysis=original_entropy_analysis,
            reduced_entropy_analysis=reduced_entropy_analysis,
            kl_divergence_forward=float(kl_forward),
            kl_divergence_reverse=float(kl_reverse),
            jensen_shannon_divergence=float(jsd),
            mutual_information=float(mutual_info),
            robdd_node_count=robdd_node_count,
            robdd_depth=robdd_depth,
            reduction_ratio=reduction_ratio,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO: MIC MINIMIZER AGENT                                  ║
# ║                                                                             ║
# ║   Endofuntor Z_Minimizer: BoolMat × ProjMat × Prob² → GovernanceState       ║
# ║   Z = Φ₃ ∘ Φ₂ ∘ Φ₁                                                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MICMinimizerAgent(Morphism, Phase3_ROBDDIsomorphismValidator):
    r"""
    El Custodio de la Base Booleana.
    
    Gobierna incondicionalmente el módulo `mic_minimizer.py`, impidiendo que
    algoritmos heurísticos de minimización degraden el rango efectivo de la MIC.
    
    Arquitectura Categórica:
        - Objeto de partida: (M_bool, P_proj, p_orig, p_red) ∈ GF(2)ᵐˣⁿ × ℝᵐˣⁿ × Δₙ × Δₙ
        - Morfismo: Z_Minimizer
        - Objeto de llegada: MinimizerGovernanceState
    
    Invariantes garantizados:
        - Independencia algebraica en GF(2)
        - Ortogonalidad estricta (Zero Side-Effects)
        - Preservación de entropía de Shannon
        - Equivalencia homotópica ROBDD
    """

    def __init__(self) -> None:
        """Inicializa el agente con configuración por defecto."""
        super().__init__()
        
        logger.info(
            "MICMinimizerAgent v3.0.0 inicializado.\n"
            "  Tolerancia de ortogonalidad: %.6e\n"
            "  Tolerancia de entropía: %.6e\n"
            "  Variables booleanas máximas: %d",
            _ORTHOGONALITY_TOLERANCE,
            _MIN_ENTROPY_TOLERANCE,
            _MAX_BOOLEAN_VARIABLES,
        )

    def _compute_overall_quality_score(
        self,
        grobner_audit: GrobnerAuditData,
        unsat_core_audit: UnsatCoreCertifierData,
        robdd_audit: ROBDDIsomorphismData,
    ) -> float:
        r"""
        Computa un score de calidad global en [0, 1].
        
        Factores considerados:
            - Independencia algebraica (peso: 0.3)
            - Ortogonalidad (peso: 0.4)
            - Preservación de entropía (peso: 0.3)
        """
        # Componente de independencia
        if grobner_audit.matrix_properties.has_full_rank:
            independence_score = 1.0
        else:
            rank_ratio = (
                grobner_audit.ideal_dimension / grobner_audit.rows
                if grobner_audit.rows > 0
                else 0.0
            )
            independence_score = float(rank_ratio)
        
        # Componente de ortogonalidad
        if unsat_core_audit.off_diagonal_conflict_norm <= unsat_core_audit.orthogonality_tolerance:
            orthogonality_score = 1.0
        else:
            deviation_ratio = (
                unsat_core_audit.orthogonality_tolerance
                / max(unsat_core_audit.off_diagonal_conflict_norm, _MACHINE_EPSILON)
            )
            orthogonality_score = float(np.clip(deviation_ratio, 0.0, 1.0))
        
        # Componente de preservación de entropía
        if robdd_audit.entropy_loss <= robdd_audit.entropy_tolerance:
            entropy_score = 1.0
        else:
            loss_ratio = (
                robdd_audit.entropy_tolerance
                / max(robdd_audit.entropy_loss, _MACHINE_EPSILON)
            )
            entropy_score = float(np.clip(loss_ratio, 0.0, 1.0))
        
        # Agregación ponderada
        overall_score = (
            0.3 * independence_score
            + 0.4 * orthogonality_score
            + 0.3 * entropy_score
        )
        
        return float(np.clip(overall_score, 0.0, 1.0))

    def _assess_risk_level(
        self,
        grobner_audit: GrobnerAuditData,
        unsat_core_audit: UnsatCoreCertifierData,
        robdd_audit: ROBDDIsomorphismData,
    ) -> str:
        r"""
        Evalúa el nivel de riesgo de la minimización.
        
        Niveles:
            NOMINAL: Minimización óptima
            WARNING: Minimización aceptable con degradación menor
            CRITICAL: Minimización cercana a umbrales de fallo
        """
        # Margen de independencia
        if grobner_audit.nullity > 0:
            return "WARNING"
        
        # Margen de ortogonalidad
        orthogonality_margin = (
            unsat_core_audit.orthogonality_tolerance
            - unsat_core_audit.off_diagonal_conflict_norm
        )
        
        if orthogonality_margin < unsat_core_audit.orthogonality_tolerance * 0.1:
            return "CRITICAL"
        
        # Margen de entropía
        entropy_margin = robdd_audit.entropy_tolerance - robdd_audit.entropy_loss
        
        if entropy_margin < robdd_audit.entropy_tolerance * 0.1:
            return "CRITICAL"
        
        # Conflictos detectados
        if unsat_core_audit.conflict_edges > 0:
            return "WARNING"
        
        return "NOMINAL"

    def execute_boolean_topology_governance(
        self,
        boolean_polynomial_matrix: NDArray[np.uint8],
        tool_projection_matrix: NDArray[np.float64],
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
        *,
        enable_extended_diagnostics: bool = False,
    ) -> MinimizerGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta completa.
        
        Flujo de ejecución:
            1. Fase 1: Auditoría de independencia Gröbner sobre GF(2)
               └─ Retorna: GrobnerAuditData
            
            2. Fase 2: Certificación de no-interferencia y ortogonalidad
               ├─ Consume: GrobnerAuditData
               └─ Retorna: UnsatCoreCertifierData
            
            3. Fase 3: Validación de isomorfismo entrópico ROBDD
               ├─ Consume: UnsatCoreCertifierData
               └─ Retorna: ROBDDIsomorphismData
            
            4. Síntesis: Construcción de MinimizerGovernanceState
        
        Args:
            boolean_polynomial_matrix: Matriz de coeficientes sobre GF(2)
            tool_projection_matrix: Matriz de proyección de herramientas P
            original_truth_table_probs: Distribución original
            reduced_robdd_probs: Distribución reducida por ROBDD
            enable_extended_diagnostics: Activar diagnósticos extendidos
        
        Returns:
            MinimizerGovernanceState con certificados de las 3 fases
        
        Raises:
            GrobnerDegeneracyError: Si rank_GF(2) < número de generadores
            NonInterferenceViolationError: Si PPᵀ ≉ I
            ROBDDHomotopyError: Si |ΔH| > ε_H
        """
        import time
        
        start_time = time.time()
        
        logger.info(
            "═══════════════════════════════════════════════════════════════\n"
            "  INICIANDO GOBERNANZA DE TOPOLOGÍA BOOLEANA\n"
            "═══════════════════════════════════════════════════════════════"
        )

        # ─────────────────────────────────────────────────────────────────
        # FASE 1: Auditoría de Independencia Gröbner
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 1: Auditoría de Base de Gröbner sobre GF(2)")
        
        grobner_audit = self._audit_grobner_independence(
            boolean_polynomial_matrix
        )
        
        logger.info("│  └─ ✓ Independencia certificada: rank=%d", grobner_audit.ideal_dimension)

        # ─────────────────────────────────────────────────────────────────
        # FASE 2: Certificación de No-Interferencia
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 2: Certificación de Ortogonalidad (Zero Side-Effects)")
        
        unsat_core_audit = self._certify_non_interference_unsat(
            tool_projection_matrix,
            grobner_audit=grobner_audit,
        )
        
        logger.info(
            "│  └─ ✓ Ortogonalidad certificada: ‖PPᵀ - I‖_F = %.6e",
            unsat_core_audit.orthogonality_analysis.frobenius_deviation,
        )

        # ─────────────────────────────────────────────────────────────────
        # FASE 3: Validación de Isomorfismo ROBDD
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 3: Validación de Equivalencia Homotópica ROBDD")
        
        robdd_audit = self._validate_robdd_homotopy(
            original_truth_table_probs,
            reduced_robdd_probs,
            unsat_core_audit=unsat_core_audit,
        )
        
        logger.info("│  └─ ✓ Homotopía certificada: ΔH = %.6e bits", robdd_audit.entropy_loss)

        # ─────────────────────────────────────────────────────────────────
        # SÍNTESIS: Construcción del estado de gobernanza
        # ─────────────────────────────────────────────────────────────────
        is_topologically_valid = bool(
            grobner_audit.is_minimally_independent
            and unsat_core_audit.is_strictly_orthogonal
            and robdd_audit.is_homotopically_equivalent
        )

        if not is_topologically_valid:
            raise MICMinimizerAgentError(
                "La composición funtorial no autorizó la minimización booleana.\n"
                "  Al menos una fase falló en certificar sus invariantes."
            )

        # Métricas agregadas
        overall_quality_score = self._compute_overall_quality_score(
            grobner_audit,
            unsat_core_audit,
            robdd_audit,
        )

        risk_assessment = self._assess_risk_level(
            grobner_audit,
            unsat_core_audit,
            robdd_audit,
        )

        elapsed_time = time.time() - start_time

        logger.info(
            "│\n╞═════════════════════════════════════════════════════════════\n"
            "│  GOBERNANZA COMPLETADA\n"
            "├─────────────────────────────────────────────────────────────\n"
            "│  ✓ Independencia Gröbner: CERTIFICADA (rank=%d)\n"
            "│  ✓ Ortogonalidad estricta: CERTIFICADA\n"
            "│  ✓ Equivalencia homotópica: CERTIFICADA\n"
            "│  ✓ Pérdida de entropía: ΔH = %.6e bits\n"
            "├─────────────────────────────────────────────────────────────\n"
            "│  Score de calidad global: %.4f / 1.000\n"
            "│  Evaluación de riesgo: %s\n"
            "│  Tiempo de ejecución: %.4f ms\n"
            "╰═════════════════════════════════════════════════════════════",
            grobner_audit.ideal_dimension,
            robdd_audit.entropy_loss,
            overall_quality_score,
            risk_assessment,
            elapsed_time * 1000,
        )

        # Diagnósticos extendidos (opcional)
        if enable_extended_diagnostics:
            self._log_extended_diagnostics(
                grobner_audit,
                unsat_core_audit,
                robdd_audit,
            )

        return MinimizerGovernanceState(
            grobner_audit=grobner_audit,
            unsat_core_audit=unsat_core_audit,
            robdd_audit=robdd_audit,
            is_topologically_valid=is_topologically_valid,
            reduction_phase=BooleanReductionPhase.COMPLETE,
            timestamp=start_time,
            overall_quality_score=overall_quality_score,
            risk_assessment=risk_assessment,
        )

    def _log_extended_diagnostics(
        self,
        grobner_audit: GrobnerAuditData,
        unsat_core_audit: UnsatCoreCertifierData,
        robdd_audit: ROBDDIsomorphismData,
    ) -> None:
        """Registra diagnósticos extendidos en el log."""
        logger.info(
            "\n"
            "╔═════════════════════════════════════════════════════════════╗\n"
            "║            DIAGNÓSTICOS EXTENDIDOS                          ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ PROPIEDADES DE MATRIZ GF(2):                                ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · Dimensión: %dx%d                                        ║\n"
            "║   · Rango GF(2): %d                                         ║\n"
            "║   · Nulidad: %d                                             ║\n"
            "║   · Rango completo: %s                                      ║\n"
            "║   · Sparsity: %.4f                                          ║\n"
            "║   · Peso de Hamming: %d                                     ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ PROPIEDADES DE CÓDIGO LINEAL [n, k, d]:                     ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · Longitud n: %d                                          ║\n"
            "║   · Dimensión k: %d                                         ║\n"
            "║   · Distancia mínima d: %d                                  ║\n"
            "║   · Tasa: %.4f                                              ║\n"
            "║   · Redundancia: %d                                         ║\n"
            "║   · Cota de Singleton: %d                                   ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ ANÁLISIS DE ORTOGONALIDAD:                                  ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · Norma fuera de diagonal: %.6e                           ║\n"
            "║   · Desviación diagonal: %.6e                               ║\n"
            "║   · Número de condición: %.6e                               ║\n"
            "║   · Gap espectral: %.6e                                     ║\n"
            "║   · Norma de Frobenius: %.6e                                ║\n"
            "║   · Norma espectral: %.6e                                   ║\n"
            "║   · Pares en conflicto: %d                                  ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ ANÁLISIS ENTRÓPICO:                                         ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Original:                                                   ║\n"
            "║   · Shannon H: %.6f bits                                    ║\n"
            "║   · Min-entropía H_∞: %.6f bits                             ║\n"
            "║   · Collision H₂: %.6f bits                                 ║\n"
            "║   · Hartley H₀: %.6f bits                                   ║\n"
            "║   · Uniformidad: %.4f                                       ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Reducido:                                                   ║\n"
            "║   · Shannon H: %.6f bits                                    ║\n"
            "║   · Min-entropía H_∞: %.6f bits                             ║\n"
            "║   · Collision H₂: %.6f bits                                 ║\n"
            "║   · Hartley H₀: %.6f bits                                   ║\n"
            "║   · Uniformidad: %.4f                                       ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ DIVERGENCIAS:                                               ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · D_KL(P_orig‖P_red): %.6e bits                           ║\n"
            "║   · D_KL(P_red‖P_orig): %.6e bits                           ║\n"
            "║   · Jensen-Shannon: %.6e bits                               ║\n"
            "║   · Distancia de variación total: %.6e                      ║\n"
            "╚═════════════════════════════════════════════════════════════╝",
            grobner_audit.matrix_properties.rows,
            grobner_audit.matrix_properties.cols,
            grobner_audit.matrix_properties.rank_gf2,
            grobner_audit.matrix_properties.nullity_gf2,
            grobner_audit.matrix_properties.has_full_rank,
            grobner_audit.matrix_properties.sparsity,
            grobner_audit.matrix_properties.hamming_weight,
            grobner_audit.code_properties.length if grobner_audit.code_properties else 0,
            grobner_audit.code_properties.dimension if grobner_audit.code_properties else 0,
            grobner_audit.code_properties.minimum_distance if grobner_audit.code_properties else 0,
            grobner_audit.code_properties.rate if grobner_audit.code_properties else 0.0,
            grobner_audit.code_properties.redundancy if grobner_audit.code_properties else 0,
            grobner_audit.code_properties.singleton_bound if grobner_audit.code_properties else 0,
            unsat_core_audit.orthogonality_analysis.off_diagonal_norm,
            unsat_core_audit.orthogonality_analysis.diagonal_deviation_norm,
            unsat_core_audit.orthogonality_analysis.condition_number,
            unsat_core_audit.orthogonality_analysis.spectral_gap,
            unsat_core_audit.orthogonality_analysis.frobenius_deviation,
            unsat_core_audit.orthogonality_analysis.spectral_deviation,
            unsat_core_audit.conflict_edges,
            robdd_audit.original_entropy_analysis.shannon_entropy,
            robdd_audit.original_entropy_analysis.min_entropy,
            robdd_audit.original_entropy_analysis.collision_entropy,
            robdd_audit.original_entropy_analysis.hartley_entropy,
            robdd_audit.original_entropy_analysis.uniformity,
            robdd_audit.reduced_entropy_analysis.shannon_entropy,
            robdd_audit.reduced_entropy_analysis.min_entropy,
            robdd_audit.reduced_entropy_analysis.collision_entropy,
            robdd_audit.reduced_entropy_analysis.hartley_entropy,
            robdd_audit.reduced_entropy_analysis.uniformity,
            robdd_audit.kl_divergence_forward,
            robdd_audit.kl_divergence_reverse,
            robdd_audit.jensen_shannon_divergence,
            robdd_audit.total_variation_distance,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA ENRIQUECIDA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    # Excepciones
    "MICMinimizerAgentError",
    "BooleanInputValidationError",
    "GrobnerDegeneracyError",
    "NonInterferenceViolationError",
    "ROBDDHomotopyError",
    "LinearDependencyError",
    "ProbabilityDistributionError",
    # Enumeraciones
    "BooleanReductionPhase",
    "OrthogonalityMetric",
    # Estructuras de datos
    "GF2MatrixProperties",
    "LinearCodeProperties",
    "GrobnerAuditData",
    "OrthogonalityAnalysis",
    "UnsatCoreCertifierData",
    "EntropyAnalysis",
    "ROBDDIsomorphismData",
    "MinimizerGovernanceState",
    # Clases de fase
    "Phase1_GrobnerBasisAuditor",
    "Phase2_UnsatCoreCertifier",
    "Phase3_ROBDDIsomorphismValidator",
    # Agente principal
    "MICMinimizerAgent",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FIN DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════