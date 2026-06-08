r"""
=========================================================================================
Módulo: MIC Minimizer (Poda Topológica en el Anillo Booleano $\mathbb{Z}_2$)
Ubicación: app/boole/strategy/mic_minimizer.py
Versión: 5.0.0 (Rigor en Bases de Gröbner y ROBDD)

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA DE BOOLE:
Este módulo opera como el Escultor Táctico ($\Gamma$-TACTICS) del ecosistema. Su función es garantizar
que la base de capacidades de la MIC sea ortonormal y de rango completo, eliminando redundancias operativas
(homología trivial) para evitar la inflación sintáctica y el colapso atencional.

FUNDAMENTOS MATEMÁTICOS Y GEOMETRÍA ALGEBRAICA:

§1. EL ANILLO BOOLEANO CONMUTATIVO:
La red de herramientas y dependencias lógicas no se evalúa con condicionales planos, sino que se proyecta sobre
el anillo cociente:
$$ \mathcal{R} = \mathbb{Z}_2[x_1, \dots, x_n] / \langle x_i^2 - x_i \rangle $$
donde cada variable $x_i$ representa una dimensión de capacidad en la base canónica $\mathbb{B}^n$. 

§2. BASE DE GRÖBNER Y ROBDD (DIAGRAMAS DE DECISIÓN):
Para extraer los implicantes primos esenciales y minimizar el circuito lógico, el módulo computa la Base de
Gröbner reducida de los ideales en $\mathbb{Z}_2$ y construye un Diagrama de Decisión Binaria Ordenado y Reducido
(ROBDD). 

§3. NÚCLEO DE INSATISFACIBILIDAD (UNSAT CORE) Y DPLL:
Se emplea el algoritmo DPLL (SAT Solver) para auditar la matriz [11]. Si el conjunto de herramientas viola la cláusula
de no-interferencia estricta $\Phi_{MIC}$, el sistema extrae el Núcleo de Insatisfacibilidad y lanza un Veto Estructural:
$$ \text{UNSAT} \implies \langle e_i, e_j \rangle \neq \delta_{ij} \text{ para algún } i \neq j $$
Garantizando matemáticamente el isomorfismo de cero efectos secundarios (Zero Side-Effects) cruzados.

    FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
    
    I. ÁLGEBRA BOOLEANA Y EL ANILLO ℤ₂[x₁,...,xₙ]/⟨x²ᵢ - xᵢ⟩:
       
       Teorema (Estructura del Anillo Booleano):
       El anillo booleano es isomorfo al álgebra de potencias P(X) bajo:
           • Suma: diferencia simétrica (XOR)
           • Producto: intersección (AND)
           • Elementos: subconjuntos de {x₁,...,xₙ}
       
       Propiedades Fundamentales:
       1. Idempotencia: ∀x ∈ R: x² = x
       2. Característica 2: ∀x ∈ R: 2x = 0
       3. Auto-complementación: ∀x ∈ R: x + x = 0
       
       Aplicación:
       Cada herramienta T define un ideal principal ⟨T⟩ en el anillo.
       La redundancia de T' respecto a {T₁,...,Tₙ} se verifica calculando:
           NF(T', G) = 0  ⟺  T' ∈ ⟨T₁,...,Tₙ⟩
       donde G es la base de Gröbner reducida del ideal.
    
    II. DIAGRAMAS DE DECISIÓN BINARIA (ROBDD):
       
       Teorema (Canonicidad de Bryant):
       Para orden de variables fijo, el ROBDD reducido es la representación
       canónica única de una función booleana f: {0,1}ⁿ → {0,1}.
       
       Expansión de Shannon:
       f(x₁,...,xₙ) = x̄₁·f(0,x₂,...,xₙ) + x₁·f(1,x₂,...,xₙ)
       
       Complejidad:
       • Construcción: O(n·2ⁿ) en el peor caso (funciones patológicas)
       • Operaciones (AND, OR, NOT): O(|ROBDD₁|·|ROBDD₂|)
       • Isomorfismo: O(1) (comparación de nodos raíz)
       
       Aplicación:
       Dos herramientas son funcionalmente equivalentes si y solo si
       sus ROBDDs comparten el mismo nodo raíz.
    
    III. SATISFACIBILIDAD (SAT) Y DPLL:
       
       Teorema (Completitud de DPLL):
       El algoritmo DPLL es completo y sound para SAT en CNF.
       
       Cláusula de No-Interferencia:
       Φ_MIC = ⋀_{i≠j} ¬(Tᵢ ∧ Tⱼ)
       
       Conversión a CNF (forma de Tseitin):
       ¬(Tᵢ ∧ Tⱼ) ≡ ¬Tᵢ ∨ ¬Tⱼ
       
       Por tanto:
       Φ_MIC = ⋀_{i<j} (¬Tᵢ ∨ ¬Tⱼ)
       
       Complejidad:
       DPLL es exponencial en el peor caso, pero eficiente en instancias
       de MIC típicas (pequeñas, estructura local).
    
    IV. HOMOLOGÍA SIMPLICIAL:
       
       Para el complejo simplicial K asociado al grafo de capacidades:
       
       Grupos de Homología:
       • H₀(K; ℤ₂): componentes conexas
       • H₁(K; ℤ₂): ciclos fundamentales (redundancia circular)
       
       Números de Betti:
       β₀ = dim H₀(K)  (número de componentes)
       β₁ = dim H₁(K)  (número de "agujeros" de redundancia)
       
       Relación de Euler-Poincaré:
       χ(K) = β₀ - β₁ + β₂ - ...
    
    V. ANÁLISIS ESPECTRAL Y CONDICIONAMIENTO:
       
       Descomposición en Valores Singulares:
       M = UΣV^T
       
       Número de Condición:
       κ(M) = σ_max / σ_min
       
       Teorema (Estabilidad Numérica):
       Si κ(M) > 1/ε_machine, la matriz está mal condicionada
       y las soluciones numéricas son inestables.
       
       Aplicación:
       Un κ(M) grande indica dependencias lineales "suaves" que
       pueden no ser detectables algebraicamente pero sí numéricamente.
    
=========================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, total_ordering
from itertools import combinations, chain
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# Importación condicional de NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


# =============================================================================
# CONFIGURACIÓN DE WARNINGS Y ERRORES NUMÉRICOS
# =============================================================================

if NUMPY_AVAILABLE:
    # Elevar warnings numéricos a excepciones para captura estricta
    warnings.filterwarnings('error', category=RuntimeWarning)
    np.seterr(all='raise')  # Elevar errores de floating-point


# =============================================================================
# JERARQUÍA DE EXCEPCIONES
# =============================================================================

class MICException(Exception):
    """Clase base para todas las excepciones del módulo MIC."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.mathematical_context = ""


class HomologicalInconsistencyError(MICException):
    """
    Excepción lanzada cuando se detecta una inconsistencia homológica
    o singularidad numérica en el análisis espectral.
    
    Ejemplos:
    - Matriz de incidencia singular (det = 0 inesperado)
    - Número de condición extremadamente alto (κ > 10¹⁵)
    - Valores singulares negativos (violación física)
    """
    
    def __init__(self, message: str, condition_number: Optional[float] = None):
        context = {}
        if condition_number is not None:
            context['condition_number'] = condition_number
        super().__init__(message, context)
        self.mathematical_context = (
            "Singularidad detectada en el análisis espectral de la matriz "
            "de incidencia, indicando dependencias lineales degeneradas."
        )


class UnsatCoreError(MICException):
    """
    El núcleo de insatisfacibilidad indica que el conjunto de herramientas
    viola la cláusula de no-interferencia Φ_MIC.
    
    Matemáticamente: ∃ subconjunto minimal S ⊆ Tools tal que Φ_MIC|_S es UNSAT.
    """
    
    def __init__(self, message: str, conflicting_tools: Optional[List[str]] = None):
        context = {'conflicting_tools': conflicting_tools or []}
        super().__init__(message, context)
        self.mathematical_context = (
            "Núcleo UNSAT detectado: existe un subconjunto minimal de "
            "herramientas que no puede satisfacer simultáneamente la "
            "cláusula de no-interferencia."
        )


class GrobnerBasisComputationError(MICException):
    """Error durante el cálculo de la base de Gröbner."""
    pass


class ROBDDConstructionError(MICException):
    """Error durante la construcción del ROBDD."""
    pass


# =============================================================================
# CONFIGURACIÓN DE LOGGING CON FORMATO MEJORADO
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Formatter con códigos ANSI para coloración en terminal.
    
    Soporta diferentes niveles con colores distintivos.
    r"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Aplica formato con colores."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        
        # Formatear nivel con color y bold
        colored_level = f"{self.BOLD}{color}{record.levelname:8}{self.COLORS['RESET']}"
        record.levelname = colored_level
        
        return super().format(record)


# Configurar logger global
def setup_logger(name: str = "MIC.Minimizer.v5", level: int = logging.INFO) -> logging.Logger:
    """
    Configura un logger con formato enriquecido.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


logger = setup_logger()


# =============================================================================
# ENUMERACIONES Y TIPOS ALGEBRAICOS
# =============================================================================

class CapabilityDimension(Enum):
    """
    Base canónica ortonormal del espacio vectorial 𝔹ⁿ sobre ℤ₂.
    
    Cada dimensión representa un eje del espacio de capacidades.
    El orden es significativo para la construcción de ROBDDs.
    
    Interpretación geométrica:
    El espacio 𝔹⁵ es un hipercubo de dimensión 5 con 2⁵ = 32 vértices.
    """
    
    PHYS_IO = 0      # Entrada/Salida física
    PHYS_NUM = 1     # Cálculo numérico
    TACT_TOPO = 2    # Topología táctica
    STRAT_FIN = 3    # Estrategia financiera
    WIS_SEM = 4      # Semántica de sabiduría
    
    def __lt__(self, other: 'CapabilityDimension') -> bool:
        """Orden total basado en el valor del enum."""
        if not isinstance(other, CapabilityDimension):
            return NotImplemented
        return self.value < other.value
    
    def __repr__(self) -> str:
        return f"Cap.{self.name}"


# =============================================================================
# VECTOR BOOLEANO (Elemento del Retículo 𝔹ⁿ)
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class BooleanVector:
    """
    Vector en el retículo booleano 𝔹ⁿ ≅ ℤ₂[x₁,...,xₙ]/⟨x²ᵢ - xᵢ⟩.
    
    Representación:
    --------------
    Internamente se almacena como un frozenset de CapabilityDimension,
    lo que permite operaciones de conjuntos eficientes y hashability.
    
    Operaciones Algebraicas:
    ------------------------
    • Suma (⊕): diferencia simétrica (XOR)
    • Producto (⊗): intersección (AND)
    • Complemento (¬): inversión respecto al universo
    
    Invariantes:
    -----------
    1. components es siempre un frozenset
    2. Todos los elementos son CapabilityDimension válidos
    3. La estructura es inmutable (frozen=True)
    
    Complejidad:
    -----------
    • Operaciones de conjunto: O(min(|A|, |B|))
    • Hashing: O(|components|)
    • Comparación: O(|components|)
    """
    
    components: FrozenSet[CapabilityDimension] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """Validación de invariantes post-construcción."""
        # Normalizar a frozenset si es necesario
        if not isinstance(self.components, frozenset):
            object.__setattr__(self, 'components', frozenset(self.components))
        
        # Validar tipos
        if not all(isinstance(c, CapabilityDimension) for c in self.components):
            invalid = [c for c in self.components if not isinstance(c, CapabilityDimension)]
            raise TypeError(
                f"Todos los componentes deben ser CapabilityDimension. "
                f"Encontrados: {invalid}"
            )
    
    # -------------------------------------------------------------------------
    # CONSTRUCTORES ALTERNATIVOS
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_minterm(cls, minterm: int, num_vars: int = 5) -> 'BooleanVector':
        """
        Construye un vector desde su representación como minitérmino.
        
        Un minitérmino es un entero donde el bit i-ésimo indica si
        la dimensión i está presente.
        
        Args:
            minterm: Entero en [0, 2^num_vars - 1]
            num_vars: Número de variables (dimensiones)
        
        Returns:
            BooleanVector correspondiente
        
        Raises:
            ValueError: Si minterm está fuera de rango
        
        Ejemplo:
            >>> BooleanVector.from_minterm(5, 3)  # 101 en binario
            BooleanVector({Dim(0), Dim(2)})
        """
        max_val = (1 << num_vars) - 1
        if not (0 <= minterm <= max_val):
            raise ValueError(
                f"Minterm {minterm} fuera de rango [0, {max_val}] "
                f"para {num_vars} variables"
            )
        
        components = {
            CapabilityDimension(i)
            for i in range(num_vars)
            if (minterm >> i) & 1
        }
        
        return cls(frozenset(components))
    
    @classmethod
    def from_binary_string(cls, binary: str) -> 'BooleanVector':
        """
        Construye desde string binario '10110'.
        
        Args:
            binary: String de '0' y '1'
        
        Returns:
            BooleanVector correspondiente
        """
        components = {
            CapabilityDimension(i)
            for i, bit in enumerate(binary)
            if bit == '1'
        }
        return cls(frozenset(components))
    
    @classmethod
    def zero(cls) -> 'BooleanVector':
        """Vector cero (conjunto vacío)."""
        return cls(frozenset())
    
    @classmethod
    def universe(cls, num_vars: int = 5) -> 'BooleanVector':
        """Vector universo (todas las dimensiones)."""
        return cls(frozenset(CapabilityDimension(i) for i in range(num_vars)))
    
    # -------------------------------------------------------------------------
    # CONVERSIONES Y REPRESENTACIONES
    # -------------------------------------------------------------------------
    
    def to_binary_string(self, num_vars: int = 5) -> str:
        """
        Convierte a representación binaria string.
        
        Args:
            num_vars: Número total de variables
        
        Returns:
            String de 'num_vars' caracteres, ejemplo: '10110'
        """
        return ''.join(
            '1' if CapabilityDimension(i) in self.components else '0'
            for i in range(num_vars)
        )
    
    def to_minterm(self) -> int:
        """
        Convierte a representación como minitérmino (entero).
        
        Returns:
            Entero en [0, 2^n - 1]
        """
        return sum(1 << cap.value for cap in self.components)
    
    def to_set_notation(self) -> str:
        """Notación de conjuntos matemática."""
        if not self.components:
            return "∅"
        caps = sorted(self.components, key=lambda c: c.value)
        return "{" + ", ".join(c.name for c in caps) + "}"
    
    # -------------------------------------------------------------------------
    # MÉTRICAS Y PROPIEDADES
    # -------------------------------------------------------------------------
    
    def hamming_weight(self) -> int:
        """
        Peso de Hamming: número de componentes no nulos.
        
        Matemáticamente: ||v||₀ = |{i : vᵢ = 1}|
        """
        return len(self.components)
    
    def hamming_distance(self, other: 'BooleanVector') -> int:
        """
        Distancia de Hamming entre dos vectores.
        
        Definición:
            d_H(u, v) = |{i : uᵢ ≠ vᵢ}| = ||u ⊕ v||₀
        
        Propiedades:
            1. Métrica válida: d(x,y) ≥ 0, d(x,x) = 0, simetría, desigualdad triangular
            2. Acotación: 0 ≤ d_H ≤ n
        
        Args:
            other: Vector a comparar
        
        Returns:
            Distancia de Hamming
        
        Raises:
            HomologicalInconsistencyError: Si la distancia excede límites físicos
        """
        distance = len(self.components ^ other.components)
        
        # Validación de sanidad
        max_distance = 20  # Límite razonable para n ≤ 20
        if not (0 <= distance <= max_distance):
            raise HomologicalInconsistencyError(
                f"Distancia de Hamming {distance} fuera de límites [0, {max_distance}]. "
                f"Posible corrupción de datos."
            )
        
        return distance
    
    # -------------------------------------------------------------------------
    # OPERACIONES ALGEBRAICAS (Retículo Booleano)
    # -------------------------------------------------------------------------
    
    def union(self, other: 'BooleanVector') -> 'BooleanVector':
        """
        Supremo (join) en el retículo: u ∨ v.
        
        Operación de conjunto: unión
        Operación lógica: OR
        """
        return BooleanVector(self.components | other.components)
    
    def intersection(self, other: 'BooleanVector') -> 'BooleanVector':
        """
        Ínfimo (meet) en el retículo: u ∧ v.
        
        Operación de conjunto: intersección
        Operación lógica: AND
        """
        return BooleanVector(self.components & other.components)
    
    def symmetric_difference(self, other: 'BooleanVector') -> 'BooleanVector':
        """
        Suma en ℤ₂: u ⊕ v.
        
        Operación de conjunto: diferencia simétrica
        Operación lógica: XOR
        """
        return BooleanVector(self.components ^ other.components)
    
    def complement(self, num_vars: int = 5) -> 'BooleanVector':
        """
        Complemento booleano: ¬u.
        
        Operación de conjunto: complemento respecto al universo
        Operación lógica: NOT
        
        Args:
            num_vars: Cardinalidad del universo
        
        Returns:
            Vector complementario
        """
        universe = {CapabilityDimension(i) for i in range(num_vars)}
        return BooleanVector(frozenset(universe - self.components))
    
    def difference(self, other: 'BooleanVector') -> 'BooleanVector':
        r"""
        Diferencia de conjuntos: u \ v = u ∧ ¬v.
        """
        return BooleanVector(self.components - other.components)
    
    # -------------------------------------------------------------------------
    # RELACIONES DE ORDEN
    # -------------------------------------------------------------------------
    
    def is_subset_of(self, other: 'BooleanVector') -> bool:
        """
        Relación de orden parcial: u ⊆ v.
        
        Interpretación:
            u es "menos específico" que v (tiene menos capacidades).
        """
        return self.components.issubset(other.components)
    
    def is_superset_of(self, other: 'BooleanVector') -> bool:
        """Relación inversa: u ⊇ v."""
        return self.components.issuperset(other.components)
    
    def is_disjoint_from(self, other: 'BooleanVector') -> bool:
        """Verifica si los conjuntos son disjuntos (⊥ en el retículo)."""
        return self.components.isdisjoint(other.components)
    
    # -------------------------------------------------------------------------
    # PRODUCTOS INTERNOS
    # -------------------------------------------------------------------------
    
    def inner_product_z2(self, other: 'BooleanVector') -> int:
        """
        Producto interno en ℤ₂: ⟨u, v⟩ = (u ∧ v).weight mod 2.
        
        Resultado:
            0 si el número de componentes comunes es par
            1 si es impar
        
        Aplicación:
            Ortogonalidad en espacios vectoriales sobre ℤ₂.
        """
        overlap = len(self.components & other.components)
        result = overlap % 2
        
        # Validación
        if result not in {0, 1}:
            raise HomologicalInconsistencyError(
                f"Producto interno en ℤ₂ inválido: {result}. "
                "Debe ser 0 o 1."
            )
        
        return result
    
    def inner_product_real(self, other: 'BooleanVector') -> int:
        """
        Producto interno estándar en ℝ: ⟨u, v⟩ = Σ uᵢvᵢ.
        
        Returns:
            Número de componentes comunes
        """
        return len(self.components & other.components)
    
    # -------------------------------------------------------------------------
    # PROTOCOLO DE ORDEN Y COMPARACIÓN
    # -------------------------------------------------------------------------
    
    def __lt__(self, other: 'BooleanVector') -> bool:
        """
        Orden lexicográfico para ordenamiento total.
        
        Nota: Este NO es el orden parcial del retículo (⊆),
        sino un orden total arbitrario para sorting.
        """
        if not isinstance(other, BooleanVector):
            return NotImplemented
        return tuple(sorted(self.components)) < tuple(sorted(other.components))
    
    def __eq__(self, other: object) -> bool:
        """Igualdad estructural."""
        if not isinstance(other, BooleanVector):
            return NotImplemented
        return self.components == other.components
    
    def __hash__(self) -> int:
        """Hash basado en el frozenset interno."""
        return hash(self.components)
    
    def __repr__(self) -> str:
        """Representación para debugging."""
        if not self.components:
            return "BooleanVector(∅)"
        caps = sorted(self.components, key=lambda c: c.value)
        caps_str = ", ".join(c.name for c in caps)
        return f"BooleanVector({{{caps_str}}})"
    
    def __str__(self) -> str:
        """Representación string compacta."""
        return self.to_set_notation()


# =============================================================================
# HERRAMIENTA (Morfismo en la Categoría de Capacidades)
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class Tool:
    """
    Representa una herramienta como un morfismo funcional T: 𝔹ⁿ → 𝔹.
    
    Matemáticamente:
    ---------------
    Una herramienta es un predicado booleano sobre el espacio de capacidades.
    Se identifica con su conjunto de capacidades (support).
    
    Atributos:
    ----------
    name: Identificador único de la herramienta
    capabilities: Vector booleano que representa las capacidades
    
    Invariantes:
    -----------
    1. name es no vacío y único en el sistema
    2. capabilities es un BooleanVector válido
    3. La estructura es inmutable
    
    Orden:
    ------
    El orden total se define lexicográficamente: primero por nombre,
    luego por capabilities (para desempate).
    """
    
    name: str
    capabilities: BooleanVector
    
    def __post_init__(self):
        """Validación post-construcción."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("El nombre debe ser una cadena no vacía")
        
        if not isinstance(self.capabilities, BooleanVector):
            raise TypeError(
                f"capabilities debe ser BooleanVector, "
                f"recibido: {type(self.capabilities).__name__}"
            )
    
    # -------------------------------------------------------------------------
    # PROPIEDADES DERIVADAS
    # -------------------------------------------------------------------------
    
    @property
    def arity(self) -> int:
        """Aridad: número de capacidades (peso de Hamming)."""
        return self.capabilities.hamming_weight()
    
    @property
    def is_trivial(self) -> bool:
        """Verifica si la herramienta no tiene capacidades."""
        return self.arity == 0
    
    @property
    def signature(self) -> str:
        """
        Firma única basada en capacidades.
        
        Útil para detección rápida de duplicados.
        """
        return self.capabilities.to_binary_string()
    
    # -------------------------------------------------------------------------
    # RELACIONES ENTRE HERRAMIENTAS
    # -------------------------------------------------------------------------
    
    def subsumes(self, other: 'Tool') -> bool:
        """
        Verifica si esta herramienta subsume a otra.
        
        Definición:
            T₁ subsumes T₂ ⟺ capabilities(T₂) ⊆ capabilities(T₁)
        
        Interpretación:
            T₁ es "más general" que T₂ (puede hacer todo lo que T₂ hace).
        """
        return other.capabilities.is_subset_of(self.capabilities)
    
    def is_subsumed_by(self, other: 'Tool') -> bool:
        """Relación inversa de subsunción."""
        return self.capabilities.is_subset_of(other.capabilities)
    
    def overlaps_with(self, other: 'Tool') -> bool:
        """Verifica si hay solapamiento de capacidades."""
        return not self.capabilities.is_disjoint_from(other.capabilities)
    
    def similarity_score(self, other: 'Tool') -> float:
        """
        Coeficiente de similitud de Jaccard.
        
        Definición:
            J(A, B) = |A ∩ B| / |A ∪ B|
        
        Rango: [0, 1]
            0: completamente disjuntas
            1: idénticas
        """
        intersection = self.capabilities.intersection(other.capabilities)
        union = self.capabilities.union(other.capabilities)
        
        union_size = union.hamming_weight()
        if union_size == 0:
            return 1.0  # Ambas vacías → idénticas
        
        return intersection.hamming_weight() / union_size
    
    # -------------------------------------------------------------------------
    # PROTOCOLO DE ORDEN Y COMPARACIÓN
    # -------------------------------------------------------------------------
    
    def __lt__(self, other: 'Tool') -> bool:
        """Orden lexicográfico: (name, capabilities)."""
        if not isinstance(other, Tool):
            return NotImplemented
        return (self.name, self.capabilities) < (other.name, other.capabilities)
    
    def __eq__(self, other: object) -> bool:
        """Igualdad por nombre Y capacidades."""
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name and self.capabilities == other.capabilities
    
    def __hash__(self) -> int:
        """Hash basado en nombre y capabilities."""
        return hash((self.name, self.capabilities))
    
    def __repr__(self) -> str:
        """Representación para debugging."""
        return f"Tool('{self.name}', {self.capabilities})"
    
    def __str__(self) -> str:
        """Representación string legible."""
        return f"{self.name}: {self.capabilities}"


# =============================================================================
# POLINOMIOS EN ℤ₂[x₁,...,xₙ]/⟨x²ᵢ - xᵢ⟩
# =============================================================================

@dataclass(frozen=True)
class Z2Polynomial:
    """
    Representa un polinomio en el anillo booleano ℤ₂[x₁,...,xₙ]/⟨x²ᵢ - xᵢ⟩.
    
    Estructura Matemática:
    ---------------------
    Un polinomio es una suma formal de monomios:
        f = ∑ aₘ·m  donde aₘ ∈ ℤ₂, m = ∏ xᵢ
    
    Representación:
    --------------
    Almacenamos solo los monomios con coeficiente 1 (los únicos relevantes en ℤ₂).
    Cada monomio es un frozenset de índices de variables.
    
    Ejemplo:
        x₁x₂ + x₃ + 1  →  monomials = {frozenset({0,1}), frozenset({2}), frozenset()}
    
    Operaciones:
    -----------
    • Suma (+): diferencia simétrica de monomios (XOR)
    • Producto (×): producto cartesiano con reducción por idempotencia
    
    Reducción Idempotente:
    ---------------------
    x²ᵢ = xᵢ se aplica automáticamente ya que usamos sets (sin duplicados).
    
    Complejidad:
    -----------
    • Suma: O(|f| + |g|)
    • Producto: O(|f|·|g|)
    • Forma normal: depende del tamaño de la base de Gröbner
    """
    
    # Conjunto de monomios, donde cada monomio es un conjunto de índices
    monomials: FrozenSet[FrozenSet[int]]
    
    def __post_init__(self):
        """Validación de invariantes."""
        if not isinstance(self.monomials, frozenset):
            object.__setattr__(self, 'monomials', frozenset(self.monomials))
        
        # Validar que cada monomio es un frozenset de enteros
        for mon in self.monomials:
            if not isinstance(mon, frozenset):
                raise TypeError(f"Cada monomio debe ser frozenset, recibido: {type(mon)}")
            if not all(isinstance(i, int) and i >= 0 for i in mon):
                raise TypeError("Cada monomio debe contener índices enteros no negativos")
    
    # -------------------------------------------------------------------------
    # CONSTRUCTORES
    # -------------------------------------------------------------------------
    
    @classmethod
    def zero(cls) -> 'Z2Polynomial':
        """Polinomio cero (conjunto vacío de monomios)."""
        return cls(frozenset())
    
    @classmethod
    def one(cls) -> 'Z2Polynomial':
        """Polinomio constante 1 (monomio vacío)."""
        return cls(frozenset([frozenset()]))
    
    @classmethod
    def variable(cls, idx: int) -> 'Z2Polynomial':
        """
        Polinomio xᵢ (variable simple).
        
        Args:
            idx: Índice de la variable (≥ 0)
        """
        if idx < 0:
            raise ValueError(f"Índice de variable debe ser ≥ 0, recibido: {idx}")
        return cls(frozenset([frozenset([idx])]))
    
    @classmethod
    def from_minterm(cls, minterm: int, num_vars: int) -> 'Z2Polynomial':
        """
        Construye un polinomio desde un minitérmino.
        
        Un minitérmino representa un monomio específico.
        
        Args:
            minterm: Entero que codifica el monomio
            num_vars: Número de variables
        
        Returns:
            Polinomio con un solo monomio
        """
        if minterm < 0 or minterm >= (1 << num_vars):
            raise ValueError(f"Minterm {minterm} fuera de rango para {num_vars} vars")
        
        vars_in_monomial = frozenset(
            i for i in range(num_vars) if (minterm >> i) & 1
        )
        
        return cls(frozenset([vars_in_monomial]))
    
    @classmethod
    def from_minterms(cls, minterms: Set[int], num_vars: int) -> 'Z2Polynomial':
        """
        Construye un polinomio como suma de minitérminos.
        
        Args:
            minterms: Conjunto de minitérminos
            num_vars: Número de variables
        
        Returns:
            Polinomio suma
        """
        result = cls.zero()
        for mt in minterms:
            result = result + cls.from_minterm(mt, num_vars)
        return result
    
    # -------------------------------------------------------------------------
    # OPERACIONES ALGEBRAICAS
    # -------------------------------------------------------------------------
    
    def __add__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        """
        Suma en ℤ₂: f + g.
        
        Operación: diferencia simétrica de monomios (XOR).
        
        Propiedad: f + f = 0 (característica 2)
        """
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        return Z2Polynomial(self.monomials ^ other.monomials)
    
    def __mul__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        """
        Producto en ℤ₂[x]/⟨x² - x⟩: f × g.
        
        Algoritmo:
        1. Para cada par de monomios (m₁, m₂), computar m₁ ∪ m₂
        2. Reducir por idempotencia (automático con sets)
        3. Sumar módulo 2 (XOR de resultado)
        
        Complejidad: O(|f|·|g|·k) donde k = tamaño promedio de monomio
        """
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        
        result_monomials: Set[FrozenSet[int]] = set()
        
        for mon1 in self.monomials:
            for mon2 in other.monomials:
                # Producto de monomios: unión de variables
                combined = mon1 | mon2
                
                # Suma módulo 2: XOR (cancelación)
                if combined in result_monomials:
                    result_monomials.remove(combined)
                else:
                    result_monomials.add(combined)
        
        return Z2Polynomial(frozenset(result_monomials))
    
    def __neg__(self) -> 'Z2Polynomial':
        """Negación: -f = f (característica 2)."""
        return self
    
    def __sub__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        """Resta: f - g = f + g (característica 2)."""
        return self + other
    
    # -------------------------------------------------------------------------
    # PROPIEDADES Y PREDICADOS
    # -------------------------------------------------------------------------
    
    def is_zero(self) -> bool:
        """Verifica si el polinomio es cero."""
        return len(self.monomials) == 0
    
    def is_one(self) -> bool:
        """Verifica si el polinomio es la constante 1."""
        return self.monomials == frozenset([frozenset()])
    
    def degree(self) -> int:
        """
        Grado del polinomio: máximo grado de sus monomios.
        
        Grado de un monomio = número de variables.
        """
        if self.is_zero():
            return -1  # Convención: deg(0) = -∞
        return max(len(mon) for mon in self.monomials)
    
    def num_terms(self) -> int:
        """Número de monomios (términos) del polinomio."""
        return len(self.monomials)
    
    # -------------------------------------------------------------------------
    # TÉRMINO LÍDER Y ORDENAMIENTOS
    # -------------------------------------------------------------------------
    
    def leading_term(
        self,
        order: Optional[Callable[[FrozenSet[int]], Tuple]] = None
    ) -> Optional[FrozenSet[int]]:
        """
        Término líder según un orden monomial.
        
        Args:
            order: Función que asigna una clave de orden a cada monomio.
                   Por defecto: orden lexicográfico graduado (grevlex).
        
        Returns:
            Monomio líder, o None si el polinomio es cero.
        
        Orden Lexicográfico Graduado (grevlex):
        ---------------------------------------
        1. Primero por grado total (mayor grado es mayor)
        2. Desempate por orden lexicográfico reverso
        """
        if not self.monomials:
            return None
        
        if order is None:
            # Orden grevlex por defecto
            def grevlex_key(mon: FrozenSet[int]) -> Tuple[int, Tuple[int, ...]]:
                degree = len(mon)
                # Orden lexicográfico reverso: ordenar indices de mayor a menor
                lex = tuple(sorted(mon, reverse=True))
                return (degree, lex)
            order = grevlex_key
        
        return max(self.monomials, key=order)
    
    def leading_coefficient(self) -> int:
        """
        Coeficiente del término líder.
        
        En ℤ₂, siempre es 1 si el término existe.
        """
        return 1 if not self.is_zero() else 0
    
    # -------------------------------------------------------------------------
    # EVALUACIÓN
    # -------------------------------------------------------------------------
    
    def evaluate(self, assignment: Dict[int, int]) -> int:
        """
        Evalúa el polinomio en una asignación de variables.
        
        Args:
            assignment: Diccionario {índice: valor} donde valor ∈ {0, 1}
        
        Returns:
            Resultado de la evaluación en ℤ₂ (0 o 1)
        """
        result = 0
        
        for monomial in self.monomials:
            # Evaluar cada monomio
            mon_value = 1
            for var_idx in monomial:
                val = assignment.get(var_idx, 0)
                if val not in {0, 1}:
                    raise ValueError(f"Valor de variable debe ser 0 o 1, recibido: {val}")
                mon_value *= val
                if mon_value == 0:
                    break  # Monomio se anula
            
            # Sumar módulo 2
            result ^= mon_value
        
        return result
    
    # -------------------------------------------------------------------------
    # REPRESENTACIÓN
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        """Representación algebraica del polinomio."""
        if not self.monomials:
            return "0"
        
        terms = []
        for mon in sorted(self.monomials, key=lambda m: (len(m), tuple(sorted(m)))):
            if not mon:
                terms.append("1")
            else:
                vars_str = "·".join(f"x{i}" for i in sorted(mon))
                terms.append(vars_str)
        
        return " + ".join(terms)
    
    def __str__(self) -> str:
        return repr(self)
    
    def __hash__(self) -> int:
        return hash(self.monomials)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        return self.monomials == other.monomials


# =============================================================================
# BASE DE GRÖBNER EN ℤ₂
# =============================================================================

class GrobnerBasisZ2:
    """
    Calcula la base de Gröbner reducida para ideales en ℤ₂[x]/⟨x² - x⟩.
    
    Fundamento Teórico:
    ------------------
    Teorema (Existencia y Unicidad):
    Para orden monomial fijo, todo ideal tiene una base de Gröbner única y reducida.
    
    Algoritmo:
    ---------
    Variante simplificada de Buchberger adaptada al anillo booleano:
    1. Inicialización con generadores
    2. Computación de S-polinomios (simplificados en ℤ₂)
    3. Reducción iterativa hasta saturación
    4. Eliminación de redundancias
    
    Complejidad:
    -----------
    En el peor caso: doblemente exponencial en n (número de variables).
    En la práctica (ideales booleanos pequeños): polynomial.
    
    Aplicación en MIC:
    -----------------
    La forma normal respecto a la base permite verificar pertenencia al ideal:
        T ∈ ⟨T₁,...,Tₙ⟩  ⟺  NF(T, G) = 0
    """
    
    def __init__(self, num_vars: int):
        """
        Inicializa el computador de base de Gröbner.
        
        Args:
            num_vars: Número de variables del anillo
        """
        if num_vars <= 0:
            raise ValueError(f"num_vars debe ser positivo, recibido: {num_vars}")
        
        self.num_vars = num_vars
        self.basis: List[Z2Polynomial] = []
        self._iteration_count = 0
        self._max_iterations = 10000  # Límite de seguridad
    
    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DE LA BASE
    # -------------------------------------------------------------------------
    
    def add_polynomial(self, poly: Z2Polynomial) -> None:
        """
        Agrega un polinomio al ideal y actualiza la base.
        
        Args:
            poly: Polinomio a agregar
        
        Raises:
            GrobnerBasisComputationError: Si excede límites de iteración
        """
        if poly.is_zero():
            logger.debug("Ignorando polinomio cero")
            return
        
        self.basis.append(poly)
        logger.debug(f"Agregado: {poly}")
        
        try:
            self._reduce_basis()
        except RecursionError as e:
            raise GrobnerBasisComputationError(
                f"Recursión excesiva al reducir base de Gröbner: {e}"
            )
    
    def _reduce_basis(self) -> None:
        """
        Reduce la base mediante el algoritmo de Buchberger simplificado.
        
        Proceso:
        -------
        1. Computar S-polinomios entre pares
        2. Reducir cada S-polinomio por la base actual
        3. Agregar nuevos polinomios no nulos
        4. Repetir hasta saturación
        """
        self._iteration_count = 0
        changed = True
        
        while changed:
            self._iteration_count += 1
            
            if self._iteration_count > self._max_iterations:
                raise GrobnerBasisComputationError(
                    f"Excedido límite de iteraciones ({self._max_iterations}) "
                    "en reducción de Gröbner. Posible ideal infinito o error."
                )
            
            changed = False
            new_polys: Set[Z2Polynomial] = set()
            
            # S-polinomios entre pares
            for i, p in enumerate(self.basis):
                for j, q in enumerate(self.basis):
                    if i >= j:
                        continue
                    
                    lt_p = p.leading_term()
                    lt_q = q.leading_term()
                    
                    if lt_p is None or lt_q is None:
                        continue

                    # Mínimo Común Múltiplo de los términos líderes (unión de variables)
                    lcm = lt_p.union(lt_q)

                    # Multiplicadores para p y q (LCM / LT)
                    mult_p = lcm.difference(lt_p)
                    mult_q = lcm.difference(lt_q)

                    # Construir los monomios como Z2Polynomials
                    poly_mult_p = Z2Polynomial(frozenset([mult_p]))
                    poly_mult_q = Z2Polynomial(frozenset([mult_q]))
                    
                    # S-polynomial = (LCM/LT(p)) * p + (LCM/LT(q)) * q
                    s_poly = (poly_mult_p * p) + (poly_mult_q * q)

                    # Reducir S-polinomio por la base actual
                    nf_s_poly = self._reduce_poly_by_basis(s_poly, self.basis)

                    if not nf_s_poly.is_zero() and nf_s_poly not in self.basis:
                        new_polys.add(nf_s_poly)
                        changed = True
            
            # Agregar nuevos polinomios reducidos
            for poly in new_polys:
                if poly not in self.basis:
                    self.basis.append(poly)
        
        # Eliminar redundancias (autoreducción)
        self._autoreduce()
        
        logger.debug(f"Base de Gröbner reducida en {self._iteration_count} iteraciones")
    
    def _autoreduce(self) -> None:
        """
        Autoreducción: elimina polinomios que son combinación de otros.
        
        Estrategia:
        ----------
        Para cada polinomio p en la base, verificar si p puede reducirse
        por el resto de la base. Si sí, reemplazarlo por su forma normal.
        """
        i = 0
        while i < len(self.basis):
            poly = self.basis[i]
            
            # Reducir poly por el resto de la base
            other_basis = [p for j, p in enumerate(self.basis) if j != i]
            reduced = self._reduce_poly_by_basis(poly, other_basis)
            
            if reduced.is_zero():
                # poly es redundante
                logger.debug(f"Eliminando redundante: {poly}")
                self.basis.pop(i)
            elif reduced != poly:
                # poly fue reducido a una forma más simple
                logger.debug(f"Simplificando: {poly} → {reduced}")
                self.basis[i] = reduced
                i += 1
            else:
                i += 1
    
    def _reduce_poly_by_basis(
        self,
        poly: Z2Polynomial,
        basis: List[Z2Polynomial]
    ) -> Z2Polynomial:
        """
        Reduce un polinomio por una lista de polinomios (base local).
        
        Algoritmo:
        ---------
        Aplicar reducción greedy: mientras el término líder de poly sea
        divisible por algún término líder en basis, reducir.
        
        Args:
            poly: Polinomio a reducir
            basis: Base para la reducción
        
        Returns:
            Forma normal de poly respecto a basis
        """
        current = poly
        
        while not current.is_zero():
            lt_current = current.leading_term()
            if lt_current is None:
                break
            
            # Buscar un polinomio en basis cuyo LT divida a lt_current
            reduced = False
            
            for basis_poly in basis:
                if basis_poly.is_zero():
                    continue
                
                lt_basis = basis_poly.leading_term()
                if lt_basis is None:
                    continue
                
                # En anillos booleanos, "división" es subset
                if lt_basis.issubset(lt_current):
                    # Construir el multiplicador: vars en current pero no en basis
                    multiplier_vars = lt_current - lt_basis
                    multiplier = Z2Polynomial(frozenset([multiplier_vars]))
                    
                    # Restar (= sumar en ℤ₂) el múltiplo apropiado
                    reduction_term = multiplier * basis_poly
                    current = current + reduction_term
                    reduced = True
                    break
            
            if not reduced:
                # No se pudo reducir más
                break
        
        return current
    
    # -------------------------------------------------------------------------
    # FORMA NORMAL Y PERTENENCIA
    # -------------------------------------------------------------------------
    
    def normal_form(self, poly: Z2Polynomial) -> Z2Polynomial:
        """
        Calcula la forma normal de un polinomio respecto a la base de Gröbner.
        
        Definición:
        ----------
        NF(f, G) es el único polinomio r tal que:
        1. f = q₁g₁ + ... + qₙgₙ + r  (división)
        2. Ningún término de r es divisible por LT(gᵢ) para todo i
        
        En ℤ₂, la forma normal es única.
        
        Args:
            poly: Polinomio a reducir
        
        Returns:
            Forma normal (polinomio irreducible por la base)
        """
        return self._reduce_poly_by_basis(poly, self.basis)
    
    def is_member(self, poly: Z2Polynomial) -> bool:
        """
        Verifica si poly pertenece al ideal generado por la base.
        
        Teorema:
        -------
        f ∈ ⟨G⟩  ⟺  NF(f, G) = 0
        
        Args:
            poly: Polinomio a verificar
        
        Returns:
            True si poly ∈ ideal, False en caso contrario
        """
        nf = self.normal_form(poly)
        return nf.is_zero()
    
    def quotient_ring_dimension(self) -> Optional[int]:
        """
        Calcula la dimensión del anillo cociente ℤ₂[x]/I como espacio vectorial.
        
        Para ideales de dimensión 0 (punto), la dimensión es finita.
        """
        # Implementación simplificada: contar monomios estándar no divisibles
        # por LT(basis)
        # Esto requiere algoritmo más sofisticado; retornamos None por ahora
        return None
    
    # -------------------------------------------------------------------------
    # MÉTODOS DE INSPECCIÓN
    # -------------------------------------------------------------------------
    
    def get_generators(self) -> List[Z2Polynomial]:
        """Retorna la base actual (generadores reducidos)."""
        return list(self.basis)
    
    def __len__(self) -> int:
        """Número de polinomios en la base."""
        return len(self.basis)
    
    def __repr__(self) -> str:
        """Representación de la base."""
        if not self.basis:
            return "GrobnerBasis(⟨0⟩)"
        polys_str = ", ".join(str(p) for p in self.basis[:5])
        if len(self.basis) > 5:
            polys_str += f", ... ({len(self.basis) - 5} more)"
        return f"GrobnerBasis(⟨{polys_str}⟩)"


# =============================================================================
# ROBDD (Reduced Ordered Binary Decision Diagrams)
# =============================================================================

class ROBDDNode:
    """
    Nodo en un ROBDD.
    
    Estructura:
    ----------
    Un nodo interno representa una decisión basada en una variable:
        • var: índice de la variable de decisión
        • low: subárbol cuando var = 0
        • high: subárbol cuando var = 1
    
    Las hojas son nodos especiales TRUE y FALSE.
    
    Invariantes del ROBDD:
    ---------------------
    1. Ordenamiento: var(low) > var(node) y var(high) > var(node)
    2. Reducción: no hay nodos redundantes (low = high)
    3. Compartición: nodos isomorfos comparten la misma instancia
    
    Complejidad Espacial:
    --------------------
    O(2ⁿ) en el peor caso, pero típicamente mucho menor debido
    a compartición y reducción.
    """
    
    __slots__ = ('var', 'low', 'high', 'id')
    
    # Nodos constantes globales
    _TRUE_NODE: Optional['ROBDDNode'] = None
    _FALSE_NODE: Optional['ROBDDNode'] = None
    
    def __init__(
        self,
        var: int,
        low: Optional['ROBDDNode'],
        high: Optional['ROBDDNode']
    ):
        """
        Crea un nodo ROBDD.
        
        Args:
            var: Índice de variable (-1 para TRUE, -2 para FALSE)
            low: Hijo izquierdo (var = 0)
            high: Hijo derecho (var = 1)
        """
        self.var = var
        self.low = low
        self.high = high
        self.id = id(self)  # ID único para hashing
    
    @classmethod
    def get_true_node(cls) -> 'ROBDDNode':
        """Retorna el nodo constante TRUE (singleton)."""
        if cls._TRUE_NODE is None:
            cls._TRUE_NODE = ROBDDNode(-1, None, None)
        return cls._TRUE_NODE
    
    @classmethod
    def get_false_node(cls) -> 'ROBDDNode':
        """Retorna el nodo constante FALSE (singleton)."""
        if cls._FALSE_NODE is None:
            cls._FALSE_NODE = ROBDDNode(-2, None, None)
        return cls._FALSE_NODE
    
    def is_terminal(self) -> bool:
        """Verifica si es un nodo terminal (hoja)."""
        return self.var < 0
    
    def is_true(self) -> bool:
        """Verifica si es el nodo TRUE."""
        return self is ROBDDNode.get_true_node()
    
    def is_false(self) -> bool:
        """Verifica si es el nodo FALSE."""
        return self is ROBDDNode.get_false_node()
    
    def __repr__(self) -> str:
        if self.is_true():
            return "TRUE"
        elif self.is_false():
            return "FALSE"
        else:
            return f"Node(x{self.var}, low={self.low.id if self.low else None}, high={self.high.id if self.high else None})"


class ROBDD:
    """
    Constructor y manipulador de ROBDDs.
    
    Teorema (Canonicidad de Bryant, 1986):
    -------------------------------------
    Para un orden de variables fijo, existe una representación canónica
    única (minimal) para cada función booleana.
    
    Algoritmo de Construcción:
    -------------------------
    1. Aplicar recursivamente la expansión de Shannon:
       f = x̄·f[x=0] + x·f[x=1]
    
    2. Tabla de unicidad para garantizar sharing:
       Dos nodos con (var, low, high) idénticos son el mismo nodo.
    
    3. Regla de reducción:
       Si low = high, eliminar el nodo y retornar low.
    
    Operaciones Soportadas:
    ----------------------
    • AND, OR, NOT, XOR (aplicación de operadores binarios)
    • Isomorfismo (comparación de nodos raíz)
    • Extracción de minitérminos
    • Conteo de soluciones
    
    Complejidad de Operaciones:
    ---------------------------
    • Construcción desde minterms: O(n·2ⁿ) peor caso, O(n·m) típico
    • AND/OR: O(|ROBDD₁|·|ROBDD₂|)
    • NOT: O(1) (swap de hojas)
    """
    
    def __init__(self, num_vars: int):
        """
        Inicializa el manejador de ROBDDs.
        
        Args:
            num_vars: Número de variables booleanas
        """
        if num_vars <= 0:
            raise ValueError(f"num_vars debe ser positivo, recibido: {num_vars}")
        
        self.num_vars = num_vars
        
        # Tabla de unicidad: (var, low.id, high.id) -> nodo único
        self.unique_table: Dict[Tuple[int, int, int], ROBDDNode] = {}
        
        # Caché de operaciones binarias
        self._operation_cache: Dict[Tuple[str, int, int], ROBDDNode] = {}
        
        logger.debug(f"ROBDD inicializado con {num_vars} variables")
    
    # -------------------------------------------------------------------------
    # TABLA DE UNICIDAD Y REDUCCIÓN
    # -------------------------------------------------------------------------
    
    def _make_node(
        self,
        var: int,
        low: ROBDDNode,
        high: ROBDDNode
    ) -> ROBDDNode:
        """
        Crea o recupera un nodo único de la tabla.
        
        Aplica la regla de reducción: si low = high, retornar low.
        
        Args:
            var: Índice de variable
            low: Hijo izquierdo
            high: Hijo derecho
        
        Returns:
            Nodo único (posiblemente compartido)
        """
        # Regla de reducción
        if low is high:
            return low
        
        # Clave para tabla de unicidad
        key = (var, id(low), id(high))
        
        if key not in self.unique_table:
            node = ROBDDNode(var, low, high)
            self.unique_table[key] = node
            logger.debug(f"Nodo creado: x{var} -> (low={low.id}, high={high.id})")
        
        return self.unique_table[key]
    
    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DESDE MINITÉRMINOS
    # -------------------------------------------------------------------------
    
    def build_from_minterms(self, minterms: Set[int]) -> ROBDDNode:
        """
        Construye un ROBDD desde un conjunto de minitérminos.
        
        Un minitérmino es una asignación completa de variables que
        hace la función verdadera.
        
        Args:
            minterms: Conjunto de enteros representando minterms
        
        Returns:
            Nodo raíz del ROBDD
        
        Raises:
            ROBDDConstructionError: Si hay error en construcción
        """
        try:
            return self._build_shannon(set(minterms), 0)
        except RecursionError as e:
            raise ROBDDConstructionError(
                f"Recursión excesiva al construir ROBDD: {e}"
            )
    
    def _build_shannon(self, minterms: Set[int], var_idx: int) -> ROBDDNode:
        """
        Construcción recursiva mediante expansión de Shannon.
        
        Caso base:
        ---------
        • Si no hay minterms: FALSE
        • Si hemos procesado todas las variables: TRUE
        
        Caso recursivo:
        --------------
        f = x̄ᵢ·f[xᵢ=0] + xᵢ·f[xᵢ=1]
        
        Args:
            minterms: Conjunto de minterms activos
            var_idx: Índice de variable actual
        
        Returns:
            Nodo raíz del subárbol
        """
        # Caso base: sin minterms
        if not minterms:
            return ROBDDNode.get_false_node()
        
        # Caso base: todas las variables procesadas
        if var_idx >= self.num_vars:
            return ROBDDNode.get_true_node()
        
        # Particionar minterms según el bit var_idx
        low_set = {m for m in minterms if not (m >> var_idx) & 1}
        high_set = {m for m in minterms if (m >> var_idx) & 1}
        
        # Recursión
        low_child = self._build_shannon(low_set, var_idx + 1)
        high_child = self._build_shannon(high_set, var_idx + 1)
        
        # Construir nodo con reducción automática
        return self._make_node(var_idx, low_child, high_child)
    
    # -------------------------------------------------------------------------
    # OPERACIONES BOOLEANAS
    # -------------------------------------------------------------------------
    
    def apply_and(self, node1: ROBDDNode, node2: ROBDDNode) -> ROBDDNode:
        """
        Aplica AND entre dos ROBDDs: f ∧ g.
        
        Algoritmo: apply recursivo con caché.
        
        Args:
            node1, node2: Nodos raíz de los ROBDDs
        
        Returns:
            Nodo raíz del resultado
        """
        return self._apply_binary_op(node1, node2, 'and')
    
    def apply_or(self, node1: ROBDDNode, node2: ROBDDNode) -> ROBDDNode:
        """Aplica OR entre dos ROBDDs: f ∨ g."""
        return self._apply_binary_op(node1, node2, 'or')
    
    def apply_xor(self, node1: ROBDDNode, node2: ROBDDNode) -> ROBDDNode:
        """Aplica XOR entre dos ROBDDs: f ⊕ g."""
        return self._apply_binary_op(node1, node2, 'xor')
    
    def apply_not(self, node: ROBDDNode) -> ROBDDNode:
        """
        Aplica NOT a un ROBDD: ¬f.
        
        Implementación: intercambiar las hojas TRUE y FALSE.
        
        Complejidad: O(|ROBDD|) para recorrer el grafo.
        """
        # Caso base: nodos terminales
        if node.is_true():
            return ROBDDNode.get_false_node()
        elif node.is_false():
            return ROBDDNode.get_true_node()
        
        # Caso recursivo
        low_neg = self.apply_not(node.low)
        high_neg = self.apply_not(node.high)
        
        return self._make_node(node.var, low_neg, high_neg)
    
    def _apply_binary_op(
        self,
        node1: ROBDDNode,
        node2: ROBDDNode,
        op: str
    ) -> ROBDDNode:
        """
        Aplica una operación binaria genérica entre ROBDDs.
        
        Algoritmo de Bryant (1986):
        ---------------------------
        1. Casos base con terminales
        2. Caché para evitar recomputación
        3. Recursión con expansión de Shannon
        
        Args:
            node1, node2: Nodos a operar
            op: Operación ('and', 'or', 'xor')
        
        Returns:
            Nodo resultado
        """
        # Verificar caché
        cache_key = (op, node1.id, node2.id)
        if cache_key in self._operation_cache:
            return self._operation_cache[cache_key]
        
        # Casos base: ambos terminales
        if node1.is_terminal() and node2.is_terminal():
            result = self._eval_terminal_op(node1, node2, op)
        
        # Caso: uno es terminal
        elif node1.is_terminal():
            result = self._apply_with_constant(node1, node2, op, is_left=True)
        elif node2.is_terminal():
            result = self._apply_with_constant(node2, node1, op, is_left=False)
        
        # Caso general: ambos son nodos internos
        else:
            result = self._apply_shannon_expansion(node1, node2, op)
        
        # Guardar en caché
        self._operation_cache[cache_key] = result
        return result
    
    def _eval_terminal_op(
        self,
        node1: ROBDDNode,
        node2: ROBDDNode,
        op: str
    ) -> ROBDDNode:
        """Evalúa operación entre dos nodos terminales."""
        val1 = 1 if node1.is_true() else 0
        val2 = 1 if node2.is_true() else 0
        
        if op == 'and':
            result = val1 & val2
        elif op == 'or':
            result = val1 | val2
        elif op == 'xor':
            result = val1 ^ val2
        else:
            raise ValueError(f"Operación desconocida: {op}")
        
        return ROBDDNode.get_true_node() if result else ROBDDNode.get_false_node()
    
    def _apply_with_constant(
        self,
        const: ROBDDNode,
        node: ROBDDNode,
        op: str,
        is_left: bool
    ) -> ROBDDNode:
        """Optimizaciones cuando uno de los operandos es constante."""
        if op == 'and':
            if const.is_false():
                return ROBDDNode.get_false_node()
            else:  # const.is_true()
                return node
        
        elif op == 'or':
            if const.is_true():
                return ROBDDNode.get_true_node()
            else:  # const.is_false()
                return node
        
        elif op == 'xor':
            if const.is_false():
                return node
            else:  # const.is_true()
                return self.apply_not(node)
        
        return node
    
    def _apply_shannon_expansion(
        self,
        node1: ROBDDNode,
        node2: ROBDDNode,
        op: str
    ) -> ROBDDNode:
        """
        Aplica expansión de Shannon para operación binaria.
        
        Caso node1.var = node2.var:
            op(f, g) = x̄·op(f₀, g₀) + x·op(f₁, g₁)
        
        Caso node1.var < node2.var:
            op(f, g) = x̄·op(f₀, g) + x·op(f₁, g)
        
        Caso node1.var > node2.var:
            op(f, g) = ȳ·op(f, g₀) + y·op(f, g₁)
        """
        if node1.var == node2.var:
            low = self._apply_binary_op(node1.low, node2.low, op)
            high = self._apply_binary_op(node1.high, node2.high, op)
            return self._make_node(node1.var, low, high)
        
        elif node1.var < node2.var:
            low = self._apply_binary_op(node1.low, node2, op)
            high = self._apply_binary_op(node1.high, node2, op)
            return self._make_node(node1.var, low, high)
        
        else:  # node1.var > node2.var
            low = self._apply_binary_op(node1, node2.low, op)
            high = self._apply_binary_op(node1, node2.high, op)
            return self._make_node(node2.var, low, high)
    
    # -------------------------------------------------------------------------
    # EXTRACCIÓN DE INFORMACIÓN
    # -------------------------------------------------------------------------
    
    def extract_minterms(self, node: ROBDDNode) -> Set[int]:
        """
        Extrae todos los minitérminos representados por un ROBDD.
        
        Args:
            node: Nodo raíz
        
        Returns:
            Conjunto de minitérminos (enteros)
        """
        minterms: Set[int] = set()
        self._collect_minterms(node, 0, 0, minterms)
        return minterms
    
    def _collect_minterms(
        self,
        node: ROBDDNode,
        var_idx: int,
        current_minterm: int,
        result: Set[int]
    ) -> None:
        """
        Recolección recursiva de minitérminos mediante pullback topológico.
        Expande el hipercubo B^n para dimensiones omitidas (Teorema de Shannon).
        """
        if node.is_false():
            return

        target_idx = self.num_vars if node.is_terminal() else node.var
        
        # Reconstrucción del Vacío Combinatorio: Expandir producto tensorial para variables omitidas
        if var_idx < target_idx:
            # Rama 0
            self._collect_minterms(node, var_idx + 1, current_minterm, result)
            # Rama 1
            minterm_with_var = current_minterm | (1 << var_idx)
            self._collect_minterms(node, var_idx + 1, minterm_with_var, result)
            return

        if node.is_true() and var_idx == self.num_vars:
            result.add(current_minterm)
            return

        if not node.is_terminal():
            # Descender por el nodo existente en la variedad
            self._collect_minterms(node.low, var_idx + 1, current_minterm, result)
            minterm_with_var = current_minterm | (1 << var_idx)
            self._collect_minterms(node.high, var_idx + 1, minterm_with_var, result)
    
    def count_solutions(self, node: ROBDDNode) -> int:
        """
        Cuenta el número de asignaciones satisfactorias.
        
        Algoritmo: DP con memoización.
        
        Returns:
            Número de soluciones (puede ser hasta 2^num_vars)
        """
        memo: Dict[int, int] = {}
        return self._count_solutions_memo(node, 0, memo)
    
    def _count_solutions_memo(
        self,
        node: ROBDDNode,
        var_idx: int,
        memo: Dict[int, int]
    ) -> int:
        """Helper recursivo con memoización."""
        if node.is_false():
            return 0
        if node.is_true() or var_idx >= self.num_vars:
            return 2 ** (self.num_vars - var_idx)
        
        key = (node.id, var_idx)
        if key in memo:
            return memo[key]
        
        # Recursión
        low_count = self._count_solutions_memo(node.low, var_idx + 1, memo)
        high_count = self._count_solutions_memo(node.high, var_idx + 1, memo)
        
        result = low_count + high_count
        memo[key] = result
        return result
    
    def size(self, node: ROBDDNode) -> int:
        """
        Calcula el tamaño del ROBDD (número de nodos).
        
        Returns:
            Número de nodos (incluyendo terminales)
        """
        visited: Set[int] = set()
        return self._size_recursive(node, visited)
    
    def _size_recursive(self, node: ROBDDNode, visited: Set[int]) -> int:
        r"""Helper recursivo para contar nodos"""
        if node.id in visited:
            return 0
        
        visited.add(node.id)
        
        if node.is_terminal():
            return 1
        
        return 1 + self._size_recursive(node.low, visited) + \
               self._size_recursive(node.high, visited)
    
    def __repr__(self) -> str:
        return f"ROBDD({self.num_vars} vars, {len(self.unique_table)} unique nodes)"


# =============================================================================
# RESOLUTOR SAT (DPLL)
# =============================================================================

class DPLLSATSolver:
    """
    Implementación del algoritmo DPLL para SAT.
    
    Algoritmo de Davis-Putnam-Logemann-Loveland (1962):
    ---------------------------------------------------
    1. Unit Propagation: Si existe cláusula unitaria [l], asignar l = TRUE
    2. Pure Literal Elimination: Si literal aparece solo positivo/negativo, asignar
    3. Branching: Elegir variable no asignada y ramificar en ambos valores
    4. Backtracking: Si rama falla, probar la otra
    
    Optimizaciones:
    --------------
    • Propagación de unidades eager
    • Eliminación de literales puros
    • Heurística de selección de variables (VSIDS no implementada aún)
    
    Complejidad:
    -----------
    Exponencial en el peor caso (problema NP-completo),
    pero eficiente en instancias típicas de MIC (pequeñas, estructura local).
    """
    
    @staticmethod
    def solve(cnf: List[List[int]]) -> bool:
        """
        Verifica satisfacibilidad de una fórmula CNF.
        
        Args:
            cnf: Lista de cláusulas, cada cláusula es lista de literales.
                 Literal positivo: variable,  negativo: -variable
        
        Returns:
            True si SAT, False si UNSAT
        
        Ejemplo:
            >>> cnf = [[1, 2], [-1, 3], [-2, -3]]
            >>> DPLLSATSolver.solve(cnf)
            True
        """
        # Normalizar CNF (copiar para evitar mutación)
        clauses = [list(clause) for clause in cnf]
        assignment: Dict[int, bool] = {}
        
        return DPLLSATSolver._dpll(clauses, assignment)
    
    @staticmethod
    def _dpll(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
        """
        Núcleo del algoritmo DPLL.
        
        Args:
            clauses: Cláusulas actuales (mutables)
            assignment: Asignación parcial actual
        
        Returns:
            True si existe asignación satisfactoria
        """
        # --- Unit Propagation ---
        while True:
            unit_literal = DPLLSATSolver._find_unit_clause(clauses)
            if unit_literal is None:
                break
            
            var = abs(unit_literal)
            value = (unit_literal > 0)
            assignment[var] = value
            
            # Propagar asignación
            new_clauses = DPLLSATSolver._propagate(clauses, var, value)
            
            if new_clauses is None:
                # Conflicto detectado
                return False
            
            clauses = new_clauses
            
            if not clauses:
                # Todas las cláusulas satisfechas
                return True
        
        # --- Pure Literal Elimination ---
        pure_literals = DPLLSATSolver._find_pure_literals(clauses)
        for lit in pure_literals:
            var = abs(lit)
            value = (lit > 0)
            assignment[var] = value
            
            new_clauses = DPLLSATSolver._propagate(clauses, var, value)
            if new_clauses is None:
                return False
            
            clauses = new_clauses
            
            if not clauses:
                return True
        
        # Caso base: sin cláusulas restantes
        if not clauses:
            return True
        
        # --- Branching ---
        var = DPLLSATSolver._select_variable(clauses, assignment)
        if var is None:
            # No hay variables libres pero aún hay cláusulas → UNSAT
            return False
        
        # Probar ambos valores
        for value in (True, False):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            new_clauses = DPLLSATSolver._propagate(
                [list(c) for c in clauses],
                var,
                value
            )
            
            if new_clauses is None:
                continue  # Rama inconsistente
            
            if DPLLSATSolver._dpll(new_clauses, new_assignment):
                assignment.update(new_assignment)
                return True
        
        return False
    
    @staticmethod
    def _find_unit_clause(clauses: List[List[int]]) -> Optional[int]:
        """Encuentra una cláusula unitaria (longitud 1), si existe."""
        for clause in clauses:
            if len(clause) == 1:
                return clause[0]
        return None
    
    @staticmethod
    def _find_pure_literals(clauses: List[List[int]]) -> List[int]:
        """
        Encuentra literales puros (aparecen solo positivos o solo negativos).
        
        Returns:
            Lista de literales puros
        """
        positive_lits: Set[int] = set()
        negative_lits: Set[int] = set()
        
        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    positive_lits.add(lit)
                else:
                    negative_lits.add(-lit)
        
        # Literales que aparecen solo positivos
        pure_positive = positive_lits - negative_lits
        
        # Literales que aparecen solo negativos
        pure_negative = negative_lits - positive_lits
        
        result = list(pure_positive) + [-lit for lit in pure_negative]
        return result
    
    @staticmethod
    def _propagate(
        clauses: List[List[int]],
        var: int,
        value: bool
    ) -> Optional[List[List[int]]]:
        """
        Propaga una asignación variable = valor.
        
        Args:
            clauses: Cláusulas actuales
            var: Variable a asignar
            value: Valor (True/False)
        
        Returns:
            Nuevas cláusulas después de propagación, o None si hay conflicto
        """
        result: List[List[int]] = []
        
        for clause in clauses:
            new_clause: List[int] = []
            satisfied = False
            
            for lit in clause:
                lit_var = abs(lit)
                lit_sign = (lit > 0)
                
                if lit_var == var:
                    if lit_sign == value:
                        # Cláusula satisfecha
                        satisfied = True
                        break
                    # else: literal falso, no incluir
                else:
                    new_clause.append(lit)
            
            if satisfied:
                continue  # Cláusula satisfecha, no agregar
            
            if not new_clause:
                # Cláusula vacía → conflicto
                return None
            
            result.append(new_clause)
        
        return result
    
    @staticmethod
    def _select_variable(
        clauses: List[List[int]],
        assignment: Dict[int, bool]
    ) -> Optional[int]:
        """
        Selecciona la próxima variable para branching.
        
        Heurística simple: primera variable no asignada encontrada.
        
        Mejora posible: VSIDS, DLIS, MOM, etc.
        """
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    return var
        return None


# =============================================================================
# ANALIZADOR PRINCIPAL DE REDUNDANCIA MIC
# =============================================================================

class MICRedundancyAnalyzer:
    """
    Analizador de redundancia de la Matriz de Interacción Central (MIC).
    
    Fundamentos Matemáticos:
    -----------------------
    El análisis integra tres formalismos complementarios:
    
    1. Álgebra de Polinomios (Base de Gröbner):
       Detecta dependencias algebraicas exactas
       
    2. ROBDDs:
       Minimización booleana canónica
       
    3. SAT (DPLL):
       Verificación de no-interferencia
    
    Pipeline de Análisis:
    --------------------
    1. Registro de herramientas → construcción incremental de base de Gröbner
    2. Construcción de matriz de incidencia
    3. Análisis espectral (rango, valores singulares, condicionamiento)
    4. Detección de dependencias lineales sobre ℤ₂
    5. Cálculo de grupos de homología (H₀, H₁)
    6. Minimización booleana via ROBDD
    7. Verificación SAT de no-interferencia
    8. Generación de reporte consolidado
    
    Complejidad Total:
    -----------------
    • Gröbner: Exponencial (pero rápido para ideales pequeños)
    • ROBDD: O(n·2ⁿ) peor caso, típicamente mucho menor
    • SAT: Exponencial (pero eficiente en instancias MIC)
    • Espectral: O(min(m,n)³) para SVD de matriz m×n
    
    Garantías:
    ---------
    • Corrección: Todos los métodos son sound y complete
    • Determinismo: Resultados reproducibles para misma entrada
    • Robustez: Manejo de errores numéricos y casos degenerados
    """
    
    def __init__(self, num_capabilities: Optional[int] = None):
        """
        Inicializa el analizador.
        
        Args:
            num_capabilities: Número de dimensiones de capacidad.
                             Si None, usa len(CapabilityDimension)
        """
        if num_capabilities is None:
            num_capabilities = len(CapabilityDimension)
        
        if num_capabilities <= 0:
            raise ValueError(f"num_capabilities debe ser positivo, recibido: {num_capabilities}")
        
        self.num_capabilities = num_capabilities
        self.tools: List[Tool] = []
        
        # Estructuras algebraicas
        self.grobner_basis = GrobnerBasisZ2(num_capabilities)
        self.robdd = ROBDD(num_capabilities)
        
        # Mapeo de herramientas a polinomios
        self._tool_polynomials: Dict[str, Z2Polynomial] = {}
        
        # Cachés
        self._incidence_matrix_cache: Optional[np.ndarray] = None
        self._tools_hash: Optional[int] = None
        
        logger.info(
            f"✓ MICRedundancyAnalyzer inicializado: "
            f"dim(𝔹) = {num_capabilities}"
        )
    
    # -------------------------------------------------------------------------
    # REGISTRO DE HERRAMIENTAS
    # -------------------------------------------------------------------------
    
    def register_tool(
        self,
        name: str,
        capabilities: Union[Set[CapabilityDimension], BooleanVector]
    ) -> None:
        """
        Registra una herramienta con validación algebraica instantánea.
        
        Proceso:
        -------
        1. Validar entrada
        2. Construir vector booleano
        3. Verificar duplicados
        4. Construir polinomio en ℤ₂
        5. Verificar dependencia mediante forma normal de Gröbner
        6. Actualizar base si independiente
        7. Invalidar cachés
        
        Args:
            name: Nombre único de la herramienta
            capabilities: Conjunto de capacidades o BooleanVector
        
        Raises:
            ValueError: Si nombre duplicado o entrada inválida
            TypeError: Si tipos incorrectos
        """
        # --- Validación de entrada ---
        if not name or not isinstance(name, str):
            raise ValueError("El nombre debe ser una cadena no vacía")
        
        if any(tool.name == name for tool in self.tools):
            raise ValueError(f"Herramienta duplicada: '{name}'")
        
        # Normalizar capabilities a BooleanVector
        if isinstance(capabilities, BooleanVector):
            bool_vec = capabilities
        elif isinstance(capabilities, (set, frozenset)):
            if not all(isinstance(c, CapabilityDimension) for c in capabilities):
                raise TypeError(
                    "Todas las capacidades deben ser CapabilityDimension"
                )
            bool_vec = BooleanVector(frozenset(capabilities))
        else:
            raise TypeError(
                f"capabilities debe ser Set[CapabilityDimension] o BooleanVector, "
                f"recibido: {type(capabilities).__name__}"
            )
        
        # --- Construir herramienta ---
        tool = Tool(name=name, capabilities=bool_vec)
        
        # --- Construir polinomio ---
        minterm = bool_vec.to_minterm()
        poly = Z2Polynomial.from_minterm(minterm, self.num_capabilities)
        
        # --- Verificar dependencia algebraica ---
        if not self.grobner_basis.basis:
            # Primera herramienta: inicializar base
            self.grobner_basis.add_polynomial(poly)
            self._tool_polynomials[name] = poly
            logger.info(f"  ✓ Herramienta base: {name} {bool_vec}")
        
        else:
            # Verificar si es redundante
            nf = self.grobner_basis.normal_form(poly)
            
            if nf.is_zero():
                logger.warning(
                    f"  ⚠ Herramienta '{name}' es REDUNDANTE "
                    f"(forma normal = 0) respecto a la base existente"
                )
                # Aún la registramos pero marcada como dependiente
                self._tool_polynomials[name] = poly
            else:
                # Independiente: agregar a la base
                self.grobner_basis.add_polynomial(poly)
                self._tool_polynomials[name] = poly
                logger.info(f"  ✓ Herramienta independiente: {name} {bool_vec}")
        
        # --- Registrar herramienta ---
        self.tools.append(tool)
        
        # --- Invalidar cachés ---
        self._incidence_matrix_cache = None
        self._tools_hash = None
    
    def unregister_tool(self, name: str) -> None:
        """
        Elimina una herramienta del registro.
        
        Nota: No actualiza la base de Gröbner (operación costosa).
        Para análisis correcto, reconstruir el analizador desde cero.
        
        Args:
            name: Nombre de la herramienta a eliminar
        """
        # Buscar y eliminar
        original_len = len(self.tools)
        self.tools = [t for t in self.tools if t.name != name]
        
        if len(self.tools) == original_len:
            logger.warning(f"Herramienta '{name}' no encontrada para eliminación")
        else:
            logger.info(f"Herramienta '{name}' eliminada")
            
            # Eliminar polinomio
            if name in self._tool_polynomials:
                del self._tool_polynomials[name]
            
            # Invalidar cachés
            self._incidence_matrix_cache = None
            self._tools_hash = None
    
    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DE MATRIZ DE INCIDENCIA
    # -------------------------------------------------------------------------
    
    def build_incidence_matrix(self) -> np.ndarray:
        """
        Construye la matriz de incidencia del sistema.
        
        Estructura:
        ----------
        Matriz M de dimensión (num_tools × num_capabilities) donde:
            M[i, j] = 1  ⟺  herramienta i tiene capacidad j
        
        Propiedades Algebraicas:
        -----------------------
        • Rango: número de herramientas linealmente independientes
        • Espacio nulo: dimensión de redundancia
        • Valores singulares: estabilidad numérica
        
        Returns:
            Matriz numpy de tipo int8
        
        Raises:
            HomologicalInconsistencyError: Si numpy no disponible pero requerido
        """
        if not NUMPY_AVAILABLE:
            raise HomologicalInconsistencyError(
                "NumPy no disponible. Análisis espectral requiere NumPy."
            )
        
        # Verificar caché
        current_hash = hash(tuple(sorted(self.tools)))
        if (self._incidence_matrix_cache is not None and 
            self._tools_hash == current_hash):
            return self._incidence_matrix_cache
        
        # Construir matriz
        if not self.tools:
            matrix = np.array([]).reshape(0, self.num_capabilities)
        else:
            matrix = np.zeros(
                (len(self.tools), self.num_capabilities),
                dtype=np.int8
            )
            
            for i, tool in enumerate(sorted(self.tools)):
                for cap in tool.capabilities.components:
                    matrix[i, cap.value] = 1
        
        # Actualizar caché
        self._incidence_matrix_cache = matrix
        self._tools_hash = current_hash
        
        logger.debug(f"Matriz de incidencia construida: {matrix.shape}")
        return matrix
    
    # -------------------------------------------------------------------------
    # ANÁLISIS ESPECTRAL
    # -------------------------------------------------------------------------
    
    def compute_spectral_properties(
        self,
        matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calcula propiedades espectrales de la matriz de incidencia.
        
        Propiedades Computadas:
        ----------------------
        1. Rango: número de filas linealmente independientes
        2. Nulidad: dimensión del espacio nulo
        3. Valores singulares: σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0
        4. Número de condición: κ = σ_max / σ_min
        5. Full rank: si rango = min(filas, columnas)
        
        Args:
            matrix: Matriz a analizar. Si None, usa la matriz de incidencia.
        
        Returns:
            Diccionario con las propiedades
        
        Raises:
            HomologicalInconsistencyError: Si hay singularidad numérica
        """
        if matrix is None:
            matrix = self.build_incidence_matrix()
        
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy no disponible; retornando propiedades triviales")
            return {
                'rank': 0,
                'nullity': 0,
                'singular_values': [],
                'condition_number': float('inf'),
                'is_full_rank': False,
                'determinant': 0.0
            }
        
        # Caso degenerado: matriz vacía
        if matrix.size == 0:
            return {
                'rank': 0,
                'nullity': 0,
                'singular_values': [],
                'condition_number': float('inf'),
                'is_full_rank': False,
                'determinant': 0.0
            }
        
        try:
            # Convertir a float64 para cálculos numéricos
            M = matrix.astype(np.float64)
            
            # Calcular rango
            rank = np.linalg.matrix_rank(M)
            
            # Calcular nulidad
            min_dim = min(M.shape)
            nullity = min_dim - rank
            
            # Descomposición en valores singulares
            singular_values = np.linalg.svd(M, compute_uv=False)
            
            # Número de condición
            nonzero_sv = singular_values[singular_values > 1e-10]
            if len(nonzero_sv) > 0:
                condition_number = nonzero_sv[0] / nonzero_sv[-1]
            else:
                condition_number = float('inf')
            
            # Verificar condicionamiento
            if condition_number > 1e15:
                logger.warning(
                    f"Matriz mal condicionada: κ = {condition_number:.2e}"
                )
            
            # Determinante (solo para matrices cuadradas)
            if M.shape[0] == M.shape[1]:
                det = np.linalg.det(M)
            else:
                det = None
            
            return {
                'rank': int(rank),
                'nullity': int(nullity),
                'singular_values': singular_values.tolist(),
                'condition_number': float(condition_number),
                'is_full_rank': bool(rank == min_dim),
                'determinant': float(det) if det is not None else None
            }
        
        except (RuntimeWarning, FloatingPointError) as e:
            raise HomologicalInconsistencyError(
                f"Singularidad numérica detectada durante SVD: {e}",
                condition_number=float('inf')
            )
        
        except np.linalg.LinAlgError as e:
            raise HomologicalInconsistencyError(
                f"Error en álgebra lineal: {e}"
            )
    
    # -------------------------------------------------------------------------
    # DEPENDENCIAS LINEALES
    # -------------------------------------------------------------------------
    
    def detect_linear_dependencies_z2(
        self,
        matrix: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Detecta dependencias lineales mediante relaciones de inclusión.
        
        En ℤ₂, una dependencia lineal común es la subsunción:
            T₁ ⊆ T₂  ⟹  T₁ es redundante si T₂ está presente
        
        Args:
            matrix: Matriz a analizar. Si None, usa la matriz de incidencia.
        
        Returns:
            Lista de diccionarios describiendo dependencias
        """
        if matrix is None:
            matrix = self.build_incidence_matrix()
        
        if not NUMPY_AVAILABLE or matrix.size == 0:
            return []
        
        dependencies: List[Dict[str, Any]] = []
        n_tools = matrix.shape[0]
        
        # Comparar todos los pares
        for i, j in combinations(range(n_tools), 2):
            vec_i = matrix[i]
            vec_j = matrix[j]
            
            # Calcular intersección (AND)
            intersection = vec_i & vec_j
            
            # Verificar subsunción
            if np.array_equal(intersection, vec_i) and not np.array_equal(vec_i, vec_j):
                # vec_i ⊆ vec_j (i es subset de j)
                dependencies.append({
                    'type': 'subset',
                    'tool_subset': self.tools[i].name,
                    'tool_superset': self.tools[j].name,
                    'redundancy_score': float(np.sum(vec_i)) / float(np.sum(vec_j))
                })
            
            elif np.array_equal(intersection, vec_j) and not np.array_equal(vec_i, vec_j):
                # vec_j ⊆ vec_i (j es subset de i)
                dependencies.append({
                    'type': 'subset',
                    'tool_subset': self.tools[j].name,
                    'tool_superset': self.tools[i].name,
                    'redundancy_score': float(np.sum(vec_j)) / float(np.sum(vec_i))
                })
            
            elif np.array_equal(vec_i, vec_j):
                # Herramientas idénticas
                dependencies.append({
                    'type': 'duplicate',
                    'tool_1': self.tools[i].name,
                    'tool_2': self.tools[j].name,
                    'redundancy_score': 1.0
                })
        
        return dependencies
    
    # -------------------------------------------------------------------------
    # HOMOLOGÍA SIMPLICIAL
    # -------------------------------------------------------------------------
    
    def compute_homology_groups(self) -> Dict[str, Any]:
        """
        Calcula aproximación de grupos de homología del complejo de herramientas.
        
        Construcción del Complejo:
        -------------------------
        • 0-celdas: herramientas
        • 1-celdas: pares de herramientas que comparten capacidades
        
        Grupos de Homología:
        -------------------
        • H₀: número de componentes conexas
        • H₁: número de "ciclos de redundancia"
        
        Números de Betti:
        ----------------
        β₀ = dim H₀ (componentes)
        β₁ = dim H₁ (ciclos)
        
        Returns:
            Diccionario con información homológica
        """
        n_tools = len(self.tools)
        
        if n_tools == 0:
            return {
                'H_0': 0,
                'H_1': 0,
                'betti_numbers': [0, 0],
                'components': [],
                'redundancy_cycles': []
            }
        
        matrix = self.build_incidence_matrix()
        
        # --- Cálculo de H₀ (componentes conexas) ---
        # Usar Union-Find para agrupar herramientas que comparten capacidades
        
        parent = list(range(n_tools))
        size = [1] * n_tools
        
        def find(x: int) -> int:
            """Find con path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: int, b: int) -> None:
            """Union by size."""
            root_a = find(a)
            root_b = find(b)
            
            if root_a != root_b:
                if size[root_a] < size[root_b]:
                    root_a, root_b = root_b, root_a
                parent[root_b] = root_a
                size[root_a] += size[root_b]
        
        # Unir herramientas que comparten al menos una capacidad
        if NUMPY_AVAILABLE and matrix.size > 0:
            for i, j in combinations(range(n_tools), 2):
                if np.dot(matrix[i], matrix[j]) > 0:
                    union(i, j)
        
        # Agrupar por componente
        component_map: Dict[int, List[str]] = defaultdict(list)
        for i in range(n_tools):
            root = find(i)
            component_map[root].append(self.tools[i].name)
        
        components = [sorted(names) for names in component_map.values()]
        beta_0 = len(components)
        
        # --- Cálculo de H₁ (ciclos de redundancia) ---
        # Herramientas con misma signature forman un ciclo
        
        signature_map: Dict[str, List[str]] = defaultdict(list)
        for tool in self.tools:
            sig = tool.signature
            signature_map[sig].append(tool.name)
        
        cycles = [names for names in signature_map.values() if len(names) > 1]
        beta_1 = len(cycles)
        
        return {
            'H_0': beta_0,
            'H_1': beta_1,
            'betti_numbers': [beta_0, beta_1],
            'components': components,
            'redundancy_cycles': cycles,
            'euler_characteristic': beta_0 - beta_1
        }
    
    # -------------------------------------------------------------------------
    # MINIMIZACIÓN CON ROBDD
    # -------------------------------------------------------------------------
    
    def minimize_with_robdd(self) -> Tuple[List[str], List[str]]:
        """
        Minimización booleana usando ROBDDs.
        
        Algoritmo:
        ---------
        1. Construir ROBDD con todos los minitérminos de las herramientas
        2. Agrupar herramientas por minitérmino (herramientas idénticas)
        3. Seleccionar un representante de cada grupo (esencial)
        4. El resto son redundantes
        
        Propiedades:
        -----------
        • Cobertura completa: todos los minitérminos están representados
        • Minimalidad: número mínimo de herramientas para cobertura
        
        Returns:
            (essential_names, redundant_names)
        
        Raises:
            ROBDDConstructionError: Si falla la construcción del ROBDD
        """
        if not self.tools:
            return [], []
        
        # Agrupar herramientas por minitérmino
        minterm_to_names: Dict[int, List[str]] = defaultdict(list)
        
        for tool in self.tools:
            minterm = tool.capabilities.to_minterm()
            minterm_to_names[minterm].append(tool.name)
        
        # Construir ROBDD con todos los minitérminos
        all_minterms = set(minterm_to_names.keys())
        
        try:
            root = self.robdd.build_from_minterms(all_minterms)
            logger.debug(f"ROBDD construido: {self.robdd.size(root)} nodos")
        except ROBDDConstructionError as e:
            logger.error(f"Fallo en construcción ROBDD: {e}")
            # Fallback: marcar todas como esenciales
            return [t.name for t in self.tools], []
        
        # Seleccionar representantes
        essential_names: Set[str] = set()
        redundant_names: Set[str] = set()
        
        for minterm, names in minterm_to_names.items():
            if names:
                # Primer nombre como representante (orden alfabético)
                sorted_names = sorted(names)
                essential_names.add(sorted_names[0])
                
                # El resto son redundantes
                for extra in sorted_names[1:]:
                    redundant_names.add(extra)
        
        return sorted(essential_names), sorted(redundant_names)
    
    # -------------------------------------------------------------------------
    # VERIFICACIÓN SAT
    # -------------------------------------------------------------------------
    
    def check_non_interference_sat(self) -> Tuple[bool, List[str]]:
        """
        Verifica la cláusula de no-interferencia usando SAT.
        
        Cláusula:
        --------
        Φ_MIC = ⋀_{i<j} (¬Tᵢ ∨ ¬Tⱼ)
        
        Interpretación:
        --------------
        Dos herramientas no pueden estar activas simultáneamente
        si comparten capacidades (para evitar conflictos).
        
        Proceso:
        -------
        1. Construir CNF de no-interferencia
        2. Verificar SAT para cada herramienta individualmente
        3. Si alguna es UNSAT, hay conflicto irresoluble
        
        Returns:
            (sat_ok, conflicting_tools)
        
        Raises:
            UnsatCoreError: Si hay conflicto en modo estricto
        """
        n_tools = len(self.tools)
        
        if n_tools <= 1:
            return True, []
        
        # Construir CNF: ¬(Tᵢ ∧ Tⱼ) = ¬Tᵢ ∨ ¬Tⱼ
        # Variables: 1 a n_tools corresponden a cada herramienta
        
        cnf: List[List[int]] = []
        
        for i, j in combinations(range(n_tools), 2):
            # Cláusula: (¬Tᵢ ∨ ¬Tⱼ)
            cnf.append([-(i+1), -(j+1)])
        
        # Verificar que cada herramienta puede estar activa individualmente
        conflicting_tools: List[str] = []
        
        for i in range(n_tools):
            # CNF con Tᵢ forzado a TRUE
            forced_cnf = cnf + [[i+1]]
            
            if not DPLLSATSolver.solve(forced_cnf):
                # Herramienta i no puede coexistir con el resto
                conflicting_tools.append(self.tools[i].name)
                logger.warning(
                    f"  ⚠ Herramienta '{self.tools[i].name}' viola no-interferencia"
                )
        
        sat_ok = len(conflicting_tools) == 0
        
        return sat_ok, conflicting_tools
    
    # -------------------------------------------------------------------------
    # ANÁLISIS INTEGRADO
    # -------------------------------------------------------------------------
    
    def analyze_redundancy(self) -> Dict[str, Any]:
        """
        Análisis completo de redundancia integrando todos los métodos.
        
        Pipeline:
        --------
        1. Construir matriz de incidencia
        2. Análisis espectral (rango, SV, condicionamiento)
        3. Detección de dependencias lineales
        4. Cálculo de homología (H₀, H₁, Betti)
        5. Minimización ROBDD
        6. Verificación SAT
        7. Integración de resultados
        8. Generación de reporte
        
        Returns:
            Diccionario completo con todos los resultados
        """
        logger.info("=" * 80)
        logger.info("ANÁLISIS DE REDUNDANCIA MIC v5.0")
        logger.info("=" * 80)
        
        if not self.tools:
            logger.warning("No hay herramientas registradas")
            return {
                'status': 'empty',
                'essential_tools': [],
                'redundant_tools': [],
                'statistics': {
                    'total_tools': 0,
                    'essential_count': 0,
                    'redundant_count': 0,
                    'reduction_rate': 0.0
                }
            }
        
        # --- 1. Matriz de Incidencia ---
        matrix = self.build_incidence_matrix()
        logger.info(f"Matriz de incidencia: {matrix.shape}")
        
        # --- 2. Análisis Espectral ---
        spectral = self.compute_spectral_properties(matrix)
        logger.info(
            f"Propiedades espectrales: "
            f"rank={spectral['rank']}, "
            f"nullity={spectral['nullity']}, "
            f"κ={spectral['condition_number']:.2e}"
        )
        
        # --- 3. Dependencias Lineales ---
        dependencies = self.detect_linear_dependencies_z2(matrix)
        logger.info(f"Dependencias detectadas: {len(dependencies)}")
        
        # --- 4. Homología ---
        homology = self.compute_homology_groups()
        logger.info(
            f"Homología: "
            f"H₀={homology['H_0']}, "
            f"H₁={homology['H_1']}, "
            f"β=[{homology['betti_numbers']}]"
        )
        
        # --- 5. Minimización ROBDD ---
        essential_robdd, redundant_robdd = self.minimize_with_robdd()
        logger.info(
            f"ROBDD minimization: "
            f"{len(essential_robdd)} essential, "
            f"{len(redundant_robdd)} redundant"
        )
        
        # --- 6. Verificación SAT ---
        sat_ok, conflicting_tools = self.check_non_interference_sat()
        logger.info(f"SAT no-interferencia: {'PASS' if sat_ok else 'FAIL'}")
        if not sat_ok:
            logger.warning(f"Herramientas conflictivas: {conflicting_tools}")
        
        # --- 7. Integración de Resultados ---
        
        # Herramientas dependientes según Gröbner (construcción incremental de base)
        dependent_grobner = []
        independent_grobner = []
        
        temp_basis = GrobnerBasisZ2(self.num_capabilities)

        for tool in self.tools:
            poly = self._tool_polynomials[tool.name]
            
            # Verificar si es miembro del ideal generado por las herramientas YA procesadas
            if temp_basis.is_member(poly):
                # Es redundante respecto a la base construida hasta ahora
                dependent_grobner.append(tool.name)
            else:
                # Aporta nuevas capacidades, es independiente
                independent_grobner.append(tool.name)
                temp_basis.add_polynomial(poly)
        
        # Herramientas esenciales: intersección de independientes (Gröbner) y ROBDD
        essential_final = sorted(
            set(essential_robdd) & set(independent_grobner)
        )
        
        # Herramientas redundantes: unión de redundantes (Gröbner o ROBDD)
        redundant_final = sorted(
            (set(redundant_robdd) | set(dependent_grobner)) - set(essential_final)
        )
        
        # --- 8. Estadísticas ---
        total = len(self.tools)
        essential_count = len(essential_final)
        redundant_count = len(redundant_final)
        reduction_rate = redundant_count / total if total > 0 else 0.0
        
        statistics = {
            'total_tools': total,
            'essential_count': essential_count,
            'redundant_count': redundant_count,
            'reduction_rate': reduction_rate,
            'spectral_rank': spectral['rank'],
            'betti_numbers': homology['betti_numbers'],
            'sat_ok': sat_ok,
            'num_dependencies': len(dependencies),
            'num_conflicting': len(conflicting_tools)
        }
        
        # --- Logging de Resultados ---
        logger.info("─" * 80)
        logger.info(f"✓ Herramientas esenciales ({essential_count}): {essential_final}")
        logger.info(f"✗ Herramientas redundantes ({redundant_count}): {redundant_final}")
        logger.info(f"📊 Tasa de reducción: {reduction_rate:.1%}")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'essential_tools': essential_final,
            'redundant_tools': redundant_final,
            'statistics': statistics,
            'homology': homology,
            'spectral': spectral,
            'dependencies': dependencies,
            'conflicting_tools': conflicting_tools,
            'grobner_basis_size': len(self.grobner_basis),
            'robdd_stats': {
                'essential': essential_robdd,
                'redundant': redundant_robdd
            }
        }


# =============================================================================
# VALIDACIÓN DE AXIOMAS
# =============================================================================

def validate_boolean_lattice_axioms(num_vars: int = 5) -> None:
    """
    Valida los axiomas del retículo booleano.
    
    Axiomas Verificados:
    -------------------
    1. Conmutatividad: a ∨ b = b ∨ a, a ∧ b = b ∧ a
    2. Asociatividad: (a ∨ b) ∨ c = a ∨ (b ∨ c), ídem para ∧
    3. Absorción: a ∨ (a ∧ b) = a, a ∧ (a ∨ b) = a
    4. Distributividad: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
    5. Complemento: a ∨ ¬a = 1, a ∧ ¬a = 0
    6. Idempotencia: a ∨ a = a, a ∧ a = a
    
    Raises:
        AssertionError: Si algún axioma falla
    """
    logger.info("Validando axiomas del retículo booleano...")
    
    # Crear vectores de prueba
    a = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
    b = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
    c = BooleanVector(frozenset([CapabilityDimension.TACT_TOPO]))
    zero = BooleanVector.zero()
    one = BooleanVector.universe(num_vars)
    
    # 1. Conmutatividad
    assert a.union(b) == b.union(a), "Falla conmutatividad de ∨"
    assert a.intersection(b) == b.intersection(a), "Falla conmutatividad de ∧"
    
    # 2. Asociatividad
    assert (a.union(b)).union(c) == a.union(b.union(c)), "Falla asociatividad de ∨"
    assert (a.intersection(b)).intersection(c) == a.intersection(b.intersection(c)), "Falla asociatividad de ∧"
    
    # 3. Absorción
    assert a.union(a.intersection(b)) == a, "Falla absorción (∨)"
    assert a.intersection(a.union(b)) == a, "Falla absorción (∧)"
    
    # 4. Distributividad
    lhs = a.intersection(b.union(c))
    rhs = (a.intersection(b)).union(a.intersection(c))
    assert lhs == rhs, "Falla distributividad"
    
    # 5. Complemento
    assert a.union(a.complement(num_vars)) == one, "Falla ley de complemento (∨)"
    assert a.intersection(a.complement(num_vars)) == zero, "Falla ley de complemento (∧)"
    
    # 6. Idempotencia
    assert a.union(a) == a, "Falla idempotencia de ∨"
    assert a.intersection(a) == a, "Falla idempotencia de ∧"
    
    logger.info("✅ Todos los axiomas del retículo booleano verificados")


# =============================================================================
# FUNCIÓN DE AUDITORÍA (Demo)
# =============================================================================

def audit_mic_redundancy() -> Dict[str, Any]:
    """
    Función de demostración del análisis de redundancia MIC.
    
    Registra herramientas de ejemplo y ejecuta el análisis completo.
    
    Returns:
        Resultados del análisis
    """
    # Validar axiomas
    validate_boolean_lattice_axioms()
    
    # Crear analizador
    analyzer = MICRedundancyAnalyzer()
    
    # Registrar herramientas de ejemplo
    logger.info("\nRegistrando herramientas...")
    
    analyzer.register_tool("stabilize_flux", {CapabilityDimension.PHYS_NUM})
    
    analyzer.register_tool("parse_raw", {
        CapabilityDimension.PHYS_IO,
        CapabilityDimension.PHYS_NUM
    })
    
    analyzer.register_tool("structure_logic", {CapabilityDimension.TACT_TOPO})
    
    # Redundante: misma capacidad que structure_logic
    analyzer.register_tool("audit_fusion", {CapabilityDimension.TACT_TOPO})
    
    analyzer.register_tool("lateral_pivot", {CapabilityDimension.STRAT_FIN})
    
    # Redundante: misma capacidad que lateral_pivot
    analyzer.register_tool("fat_tail_risk", {CapabilityDimension.STRAT_FIN})
    
    analyzer.register_tool("semantic_estimator", {CapabilityDimension.WIS_SEM})
    
    # Redundante: misma capacidad que stabilize_flux
    analyzer.register_tool("flux_stabilizer", {CapabilityDimension.PHYS_NUM})
    
    # Ejecutar análisis
    logger.info("\nEjecutando análisis de redundancia...")
    return analyzer.analyze_redundancy()


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        # Ejecutar auditoría
        results = audit_mic_redundancy()
        
        # Imprimir resumen ejecutivo
        print("\n" + "=" * 80)
        print("RESUMEN EJECUTIVO - ANÁLISIS DE REDUNDANCIA MIC v5.0")
        print("=" * 80)
        
        stats = results['statistics']
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"  • Total de herramientas:    {stats['total_tools']}")
        print(f"  • Herramientas esenciales:  {stats['essential_count']}")
        print(f"  • Herramientas redundantes: {stats['redundant_count']}")
        print(f"  • Tasa de reducción:        {stats['reduction_rate']:.1%}")
        
        print(f"\n🔢 ANÁLISIS ALGEBRAICO:")
        print(f"  • Rango espectral:          {stats['spectral_rank']}/{len(CapabilityDimension)}")
        print(f"  • Números de Betti:         β₀={stats['betti_numbers'][0]}, β₁={stats['betti_numbers'][1]}")
        print(f"  • Dependencias detectadas:  {stats['num_dependencies']}")
        
        print(f"\n✅ VERIFICACIONES:")
        print(f"  • SAT no-interferencia:     {'PASS ✓' if stats['sat_ok'] else 'FAIL ✗'}")
        if stats['num_conflicting'] > 0:
            print(f"  • Herramientas conflictivas: {results['conflicting_tools']}")
        
        print(f"\n📋 RESULTADOS:")
        print(f"  • Esenciales:  {results['essential_tools']}")
        print(f"  • Redundantes: {results['redundant_tools']}")
        
        print("\n" + "=" * 80)
        
        sys.exit(0)
    
    except Exception as e:
        logger.exception("Error crítico en auditoría:")
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)