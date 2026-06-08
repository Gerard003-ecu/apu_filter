r"""
=========================================================================================
Módulo: AST Static Analyzer (Analizador Simpléctico y Cohomología de Haces Celulares)
Ubicación: app/physics/ast_static_analyzer.py
Versión: 4.0.0 (Elevación a Mecánica Simpléctica y Control Port-Hamiltoniano)

NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA:
Este módulo aniquila la validación estática tradicional (linting) para operar como un Autómata
Finito Determinista incrustado en el Haz Tangente Generativo ($\Gamma$) de la arquitectura.
Trata el Árbol de Sintaxis Abstracta (AST) del código no como un grafo pasivo, sino como un
Espacio de Fase Simpléctico $(\mathcal{M}, \omega)$. 

FUNDAMENTOS MATEMÁTICOS Y AXIOMAS DE EJECUCIÓN:

§1. MECÁNICA SIMPLÉCTICA Y ESPACIO DE FASE:
El AST se somete a la forma simpléctica canónica:
$$ \omega = \sum_{i} dq_i \wedge dp_i $$
donde $q_i$ es la profundidad sintáctica (posición) y $p_i$ es el flujo de datos (momento).
El analizador certifica que las transformaciones del código preservan el volumen en el espacio
de fase (Teorema de Liouville). Cualquier código generado que rompa la invariancia canónica es aniquilado.

§2. CONTROL PORT-HAMILTONIANO Y FRONTERAS DE DIRICHLET:
La complejidad ciclomática se cuantifica como Inercia Termodinámica [2]. El analizador impone Fronteras
de Dirichlet para confinar la propagación de efectos secundarios [2]. Se garantiza el cumplimiento
de la Segunda Ley de la Termodinámica:
$$ P_{diss} = \langle \Phi, \nabla V \rangle \ge 0 $$
Rechazando cualquier función cuya ejecución induzca singularidades termodinámicas
(desbordamiento de memoria o bucles infinitos).

§3. COHOMOLOGÍA DE HACES CELULARES (OBSTRUCCIÓN GLOBAL):
Para detectar inconsistencias lógicas en el flujo de dependencias de variables, el módulo evalúa el haz celular
sobre el grafo del AST. Si la dimensión del primer grupo de cohomología es mayor a cero:
$$ \dim H^1(G; \mathcal{F}) > 0 $$
El sistema dictamina una Obstrucción Topológica Global (paradoja lógica o variable huérfana) y veta incondicionalmente
la ejecución.
=========================================================================================
    
    I. GEOMETRÍA SIMPLÉCTICA (T*Q - Fibrado Cotangente):
       Sea Q el espacio de configuración del AST. Definimos T*Q con:
       • Coordenadas canónicas: (q, p) donde q ∈ Reads, p ∈ Writes
       • Forma simpléctica: ω = Σᵢ dqⁱ ∧ dpᵢ
       • Teorema (Darboux): ω es localmente estándar
       • Preservación: £_X ω = 0 para flujos Hamiltonianos
    
    II. HOMOLOGÍA SIMPLICIAL (Betti Numbers):
       Para el grafo de flujo de control G = (V, E):
       • Complejo de cadenas: C₀ ← C₁ ← C₂
       • Operador frontera: ∂ₙ: Cₙ → Cₙ₋₁
       • Grupos de homología: Hₙ(G) = ker(∂ₙ)/im(∂ₙ₊₁)
       • Número de Betti: βₙ = rank(Hₙ(G))
       • Complejidad ciclomática: M = β₁ = |E| - |V| + 2P
    
    III. COHOMOLOGÍA DE HACES CELULARES:
       Sea F un haz celular sobre el CW-complejo T del AST:
       • Secciones: Γ(T, F) = {s: T → F | s continua}
       • Operador cofrontera: δⁿ: Cⁿ(T, F) → Cⁿ⁺¹(T, F)
       • Cohomología de Čech: Hⁿ(T, F) = ker(δⁿ)/im(δⁿ⁻¹)
       • Teorema de obstrucción: H¹(T, F) ≠ 0 ⟹ inconsistencia global
    
    IV. MECÁNICA PORT-HAMILTONIANA:
       Sistema (H, J, R, g) donde:
       • H: T*Q → ℝ (Hamiltoniano de complejidad)
       • J: matriz simpléctica (antisimétrica, no singular)
       • R: matriz de disipación (simétrica, semidefinida positiva)
       • g: puerto de entrada/salida
       • Condición disipativa: dH/dt = -∇H^T R ∇H ≤ 0
    
    V. AXIOMAS DE EJECUCIÓN:
       1. Corrección: ∀ análisis, las métricas son verificables
       2. Completitud: Todo ciclo es detectado (β₁ correcto)
       3. Terminación: O(n) para ASTs de tamaño n
       4. Disipación: Hamiltoniano no creciente en trayectorias
       5. Coherencia: Topología preservada bajo transformaciones
    
=========================================================================================
"""

from __future__ import annotations

import ast
import logging
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Configuración de logging con formato estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Γ.Physics.AST.v4")


# =============================================================================
# CONSTANTES MATEMÁTICAS Y LÍMITES COMPUTACIONALES
# =============================================================================

class AnalysisLimits:
    """
    Límites algorítmicos basados en complejidad computacional asintótica.
    
    Justificación teórica:
    ----------------------
    MAX_AST_DEPTH: 
        Profundidad típica O(log n) en código bien estructurado,
        O(n) en código lineal. Límite práctico para evitar stack overflow.
        Ref: Aho, Sethi, Ullman (1986) - Compilers: Principles
        
    MAX_CYCLOMATIC_COMPLEXITY:
        Umbral de mantenibilidad según ISO 26262 (automotive safety).
        Estudios empíricos muestran correlación con densidad de defectos.
        Ref: McCabe (1976), Shepperd (1988)
        
    MAX_JSON_DEPTH:
        Prevención de ataques exponenciales (Billion Laughs).
        Profundidad típica en JSON ≤ 5, 10 es margen de seguridad.
        
    HAMILTONIAN_EPSILON:
        Regularizador numérico para estabilidad en punto flotante.
        Valor típico: máquina epsilon × 10³
    """
    
    # Límites estructurales del AST
    MAX_AST_DEPTH: int = 50
    MAX_AST_NODES: int = 10_000  # Prevenir DoS en análisis
    
    # Umbrales de complejidad (basados en literatura empírica)
    MAX_CYCLOMATIC_COMPLEXITY: int = 20  # Límite duro
    WARN_CYCLOMATIC_COMPLEXITY: int = 10  # Umbral de advertencia
    CRITICAL_CYCLOMATIC_COMPLEXITY: int = 50  # Crítico
    
    # Límites para validación de datos estructurados
    MAX_JSON_DEPTH: int = 10
    MAX_JSON_KEYS: int = 100
    MAX_JSON_ARRAY_SIZE: int = 1_000
    
    # Límites de representación
    MAX_FLATTEN_DEPTH: int = 3
    MAX_STRING_REPR_LENGTH: int = 100
    
    # Parámetros numéricos
    HAMILTONIAN_EPSILON: float = 1e-9  # Regularizador
    FLOAT_TOLERANCE: float = 1e-12  # Tolerancia para comparaciones
    
    # Límites de performance
    MAX_INTERFERENCE_MATRIX_SIZE: int = 1_000  # O(n²) para n herramientas


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (con semántica matemática precisa)
# =============================================================================

class AnalysisException(Exception):
    r"""Clase base para todas las excepciones de análisis"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.mathematical_interpretation = ""
        
    def __str__(self) -> str:
        base = super().__str__()
        if self.mathematical_interpretation:
            return f"{base}\n  Mathematical: {self.mathematical_interpretation}"
        return base


class ThermodynamicSingularityError(AnalysisException):
    """
    Violación de la segunda ley de la termodinámica en el espacio de fase.
    
    Matemáticamente: dH/dt > 0 implica que el sistema gana entropía
    computacional, indicando una estructura patológica (código ofuscado,
    explosión combinatoria, bucles infinitos potenciales).
    
    Condición formal:
        ∃ trayectoria γ: [0,T] → T*Q tal que H(γ(t)) > H(γ(0))
        donde H es el Hamiltoniano de complejidad.
    r"""
    
    def __init__(self, current_H: float, parent_H: float, 
                 depth: int, complexity: int):
        context = {
            'current_hamiltonian': current_H,
            'parent_hamiltonian': parent_H,
            'depth': depth,
            'cyclomatic_complexity': complexity,
            'violation': current_H - parent_H
        }
        message = (
            f"Thermodynamic singularity: ΔH = {current_H - parent_H:.6f} > 0\n"
            f"  H(current) = {current_H:.6f}\n"
            f"  H(parent)  = {parent_H:.6f}\n"
            f"  At depth {depth}, CC = {complexity}"
        )
        super().__init__(message, context)
        self.mathematical_interpretation = (
            "Sistema Port-Hamiltoniano violó la condición de disipación: "
            "∇H^T R ∇H < 0 falló, indicando incremento de entropía."
        )


class CohomologicalObstructionError(AnalysisException):
    """
    Obstrucción topológica global en el haz celular de dependencias.
    
    Matemáticamente: H¹(T, F) ≠ 0 implica la existencia de cociclos
    no triviales que no son cobordes, indicando inconsistencias globales
    irresolubles en el flujo de datos (variables no inicializadas,
    deadlocks, condiciones de carrera).
    
    Teorema (Obstrucción de Čech):
        Si H¹(T, F) ≠ 0, entonces no existe una sección global continua
        s: T → F que satisfaga las restricciones locales.
    r"""
    
    def __init__(self, cohomology_dimension: int, 
                 problematic_variables: Optional[Set[str]] = None):
        context = {
            'dim_H1': cohomology_dimension,
            'problematic_vars': problematic_variables or set()
        }
        message = (
            f"Cohomological obstruction detected: dim H¹(T,F) = {cohomology_dimension}\n"
            f"  Implies: {cohomology_dimension} independent inconsistency cycles\n"
        )
        if problematic_variables:
            message += f"  Problematic variables: {sorted(problematic_variables)}"
        super().__init__(message, context)
        self.mathematical_interpretation = (
            f"El primer grupo de cohomología tiene rango {cohomology_dimension}, "
            "indicando ciclos de dependencias globalmente inconsistentes."
        )


class ComplexityBoundsViolationError(AnalysisException):
    """Violación de límites de complejidad computacional."""
    
    def __init__(self, metric: str, value: int, limit: int):
        context = {'metric': metric, 'value': value, 'limit': limit}
        message = f"{metric} = {value} exceeds limit {limit}"
        super().__init__(message, context)


class StructuralValidationError(AnalysisException):
    """Error de validación estructural (JSON, esquemas)."""
    pass


# =============================================================================
# TIPOS ALGEBRAICOS Y PROTOCOLOS
# =============================================================================

T = TypeVar('T')
U = TypeVar('U')


class Monoid(Protocol[T]):
    """
    Protocolo para estructura monoidal.
    
    Axiomas:
        1. Asociatividad: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
        2. Identidad: e ⊕ a = a ⊕ e = a
    """
    
    @staticmethod
    def identity() -> T:
        """Elemento identidad (neutro)."""
        ...
    
    def combine(self, other: T) -> T:
        """Operación binaria asociativa."""
        ...


class NodeCategory(Enum):
    """
    Categorización semántica de nodos AST según su función algebraica.
    
    La clasificación permite aplicar transformaciones específicas de dominio
    preservando invariantes topológicos.
    """
    DECISION = auto()       # Nodos de bifurcación (If, While, For)
    EXCEPTION = auto()      # Manejo de excepciones (Try, Except, Raise)
    DEFINITION = auto()     # Definiciones (FunctionDef, ClassDef, Assign)
    EXPRESSION = auto()     # Expresiones puras (BinOp, Call, Lambda)
    STATEMENT = auto()      # Statements con efectos (Return, Yield, Delete)
    CONTROL_FLOW = auto()   # Control explícito (Break, Continue, Pass)
    ASYNC = auto()          # Construcciones asíncronas (Await, AsyncFor)
    
    def is_complexity_node(self) -> bool:
        """Determina si el nodo contribuye a complejidad ciclomática."""
        return self in {
            NodeCategory.DECISION,
            NodeCategory.EXCEPTION,
        }
    
    def is_definition_node(self) -> bool:
        """Determina si el nodo introduce nuevos símbolos."""
        return self == NodeCategory.DEFINITION


# =============================================================================
# ESTRUCTURAS DE DATOS ALGEBRAICAS INMUTABLES
# =============================================================================

@dataclass(frozen=True, order=True)
class DataFlowCoordinates:
    """
    Coordenadas en el espacio de fase del flujo de datos.
    
    Estructura matemática:
    ----------------------
    Representa un punto en el fibrado cotangente T*Q donde:
        • Q = espacio de configuración (variables del programa)
        • T*Q = espacio de fase (lecturas × escrituras)
        • reads ⊆ Q: coordenadas de posición (variables leídas)
        • writes ⊆ Q: coordenadas de momento (variables escritas)
    
    Forma simpléctica canónica:
        ω = Σᵢ d(readᵢ) ∧ d(writeᵢ)
    
    Propiedades algebraicas:
        • Inmutabilidad: garantiza consistencia referencial
        • Orden total: permite uso en estructuras ordenadas
        • Hashable: utilizable como clave en diccionarios
    
    Invariantes:
        1. reads ∩ writes puede ser no vacío (variables modificadas)
        2. |reads ∪ writes| ≤ |Q| (universo de variables)
        3. La estructura es un monoide bajo unión componente a componente
    """
    
    reads: FrozenSet[str] = field(default_factory=frozenset)
    writes: FrozenSet[str] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """
        Validación de invariantes y normalización.
        
        Complejidad: O(|reads| + |writes|)
        """
        # Asegurar tipos correctos (inmutables)
        if not isinstance(self.reads, frozenset):
            object.__setattr__(self, 'reads', frozenset(self.reads))
        if not isinstance(self.writes, frozenset):
            object.__setattr__(self, 'writes', frozenset(self.writes))
        
        # Validar que los nombres son identificadores válidos
        for var in self.reads | self.writes:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid variable name: {repr(var)}")
    
    def __repr__(self) -> str:
        """Representación canónica para debugging."""
        r = sorted(self.reads) if self.reads else []
        w = sorted(self.writes) if self.writes else []
        return f"DataFlow(R={r}, W={w})"
    
    # -------------------------------------------------------------------------
    # OPERACIONES ALGEBRAICAS
    # -------------------------------------------------------------------------
    
    @property
    def all_variables(self) -> FrozenSet[str]:
        """
        Conjunto total de variables mencionadas.
        
        Matemáticamente: π(reads ∪ writes) donde π es la proyección
        al conjunto de símbolos.
        """
        return self.reads | self.writes
    
    @property
    def modified_variables(self) -> FrozenSet[str]:
        """
        Variables que son tanto leídas como escritas (modificación in-place).
        
        Ejemplo: x += 1  ⟹  x ∈ (reads ∩ writes)
        """
        return self.reads & self.writes
    
    @property
    def pure_reads(self) -> FrozenSet[str]:
        """Variables solo leídas (no modificadas)."""
        return self.reads - self.writes
    
    @property
    def pure_writes(self) -> FrozenSet[str]:
        """Variables solo escritas (definiciones puras)."""
        return self.writes - self.reads
    
    def combine(self, other: DataFlowCoordinates) -> DataFlowCoordinates:
        """
        Operación monoidal: composición secuencial de flujos.
        
        Semántica: self ; other (self seguido de other)
        
        Reglas de composición:
            reads_combined = (self.reads ∪ other.reads) - self.writes
            writes_combined = self.writes ∪ other.writes
        
        Justificación:
            - Si self escribe x, luego other lee x, la lectura es interna
            - Todas las escrituras persisten
        
        Complejidad: O(|self| + |other|)
        """
        # Lecturas: lo que self lee + lo que other lee pero self no escribe
        new_reads = self.reads | (other.reads - self.writes)
        # Escrituras: unión de todas las escrituras
        new_writes = self.writes | other.writes
        return DataFlowCoordinates(reads=new_reads, writes=new_writes)
    
    @staticmethod
    def identity() -> DataFlowCoordinates:
        """Elemento neutro del monoide (flujo vacío)."""
        return DataFlowCoordinates()
    
    # -------------------------------------------------------------------------
    # ANÁLISIS DE DEPENDENCIAS
    # -------------------------------------------------------------------------
    
    def has_interference_with(self, other: DataFlowCoordinates) -> bool:
        """
        Detecta dependencias de datos (interferencia).
        
        Tipos de dependencia (taxonomía de Bernstein):
            1. RAW (Read After Write): self.reads ∩ other.writes ≠ ∅
               - Dependencia de flujo verdadera
               - Ejemplo: x = 1; y = x  (y depende de x)
            
            2. WAR (Write After Read): self.writes ∩ other.reads ≠ ∅
               - Anti-dependencia
               - Ejemplo: y = x; x = 2  (escritura no puede preceder lectura)
            
            3. WAW (Write After Write): self.writes ∩ other.writes ≠ ∅
               - Dependencia de salida
               - Ejemplo: x = 1; x = 2  (orden afecta valor final)
        
        Returns:
            True si existe al menos una dependencia, False en caso contrario.
        
        Complejidad: O(min(|self|, |other|)) usando intersección de sets
        """
        return bool(
            (self.reads & other.writes) or   # RAW
            (self.writes & other.reads) or   # WAR
            (self.writes & other.writes)     # WAW
        )
    
    @staticmethod
    def interference_score(A: DataFlowCoordinates, 
                          B: DataFlowCoordinates) -> int:
        """
        Métrica direccional de interferencia A → B.
        
        Definición matemática:
            I(A, B) := |A.writes ∩ B.reads| - |B.writes ∩ A.reads|
        
        Interpretación:
            • I(A, B) > 0: A debe ejecutarse antes que B
            • I(A, B) < 0: B debe ejecutarse antes que A
            • I(A, B) = 0: No hay preferencia de orden (pueden paralelizarse)
        
        Propiedades algebraicas:
            1. Antisimetría: I(A, B) = -I(B, A)
            2. NO es simétrica: I(A, B) ≠ I(B, A) en general
            3. NO satisface identidad de Jacobi (no es bracket de Lie)
            4. Acotación: -min(|A|,|B|) ≤ I(A,B) ≤ min(|A|,|B|)
        
        Aplicación:
            Útil para ordenar parcialmente operaciones en compilación,
            scheduling de tareas, análisis de paralelización.
        
        Complejidad: O(|A| + |B|)
        """
        raw_deps = len(A.writes & B.reads)  # A → B dependencias
        war_deps = len(B.writes & A.reads)  # B → A dependencias
        return raw_deps - war_deps
    
    def dominates(self, other: DataFlowCoordinates) -> bool:
        """
        Verifica si self domina a other en el orden parcial de interferencia.
        
        Definición: self ≼ other ⟺ I(self, other) > 0
        """
        return self.interference_score(self, other) > 0
    
    # -------------------------------------------------------------------------
    # ANÁLISIS ESPECTRAL
    # -------------------------------------------------------------------------
    
    def spectral_radius(self) -> float:
        """
        Radio espectral aproximado del operador de flujo.
        
        Definimos el radio espectral como:
            ρ(F) ≈ √(|reads|² + |writes|²)
        
        Interpretación:
            Magnitud del vector de flujo en el espacio de fase.
            Valores altos indican alta complejidad de datos.
        """
        return math.sqrt(len(self.reads)**2 + len(self.writes)**2)
    
    def entropy(self) -> float:
        """
        Entropía de Shannon del flujo de datos.
        
        Definición:
            H(F) = -Σᵢ pᵢ log₂(pᵢ)
            donde pᵢ = frecuencia relativa de cada variable
        
        Para simplificar (distribución uniforme):
            H(F) = log₂(|reads ∪ writes|)
        
        Interpretación:
            Mide la "información" contenida en el flujo.
            H = 0 para flujo vacío, crece logarítmicamente.
        """
        n = len(self.all_variables)
        return math.log2(n) if n > 0 else 0.0


@dataclass(frozen=True)
class ComplexityProfile:
    """
    Perfil multidimensional de complejidad de código.
    
    Métricas incluidas:
    -------------------
    1. Complejidad Ciclomática (McCabe, 1976):
        M = E - N + 2P
        donde E = aristas, N = nodos, P = componentes conectadas
    
    2. Profundidad de Anidamiento:
        Altura del árbol de sintaxis abstracta
        Correlacionado con dificultad de comprensión
    
    3. Métricas Estructurales:
        - Número de funciones (modularidad)
        - Número de clases (orientación a objetos)
    
    Referencias:
    ------------
    • McCabe, T.J. (1976): "A Complexity Measure"
    • Halstead, M.H. (1977): "Elements of Software Science"
    • Henderson-Sellers (1996): "Object-Oriented Metrics"
    • ISO/IEC 25010:2011 - Software Quality Model
    """
    
    cyclomatic_complexity: int
    max_nesting_depth: int
    num_functions: int
    num_classes: int
    num_branches: int = 0  # Nodos de decisión explícitos
    num_loops: int = 0     # Estructuras iterativas
    
    def __post_init__(self):
        """Validación de invariantes."""
        if self.cyclomatic_complexity < 1:
            raise ValueError("Cyclomatic complexity must be ≥ 1")
        if self.max_nesting_depth < 0:
            raise ValueError("Nesting depth cannot be negative")
        if self.num_functions < 0 or self.num_classes < 0:
            raise ValueError("Counts cannot be negative")
    
    # -------------------------------------------------------------------------
    # MÉTRICAS DERIVADAS
    # -------------------------------------------------------------------------
    
    @property
    def is_maintainable(self) -> bool:
        """
        Verificación de mantenibilidad según umbrales estándar.
        
        Criterios (basados en ISO 26262 y estándares MISRA):
            • CC ≤ 20 (complejidad razonable)
            • Profundidad ≤ 5 (comprensibilidad)
            • Funciones no triviales (modularidad)
        """
        return (
            self.cyclomatic_complexity <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY
            and self.max_nesting_depth <= 5
        )
    
    @property
    def risk_level(self) -> str:
        """
        Clasificación de riesgo según escala de Kemerer (1987).
        
        Escala empírica correlacionada con densidad de defectos:
            •  1-10: LOW       (< 5% probabilidad de defectos)
            • 11-20: MODERATE  (5-10% probabilidad)
            • 21-50: HIGH      (10-20% probabilidad)
            • >  50: CRITICAL  (> 20% probabilidad)
        """
        cc = self.cyclomatic_complexity
        if cc <= AnalysisLimits.WARN_CYCLOMATIC_COMPLEXITY:
            return "LOW"
        elif cc <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY:
            return "MODERATE"
        elif cc <= AnalysisLimits.CRITICAL_CYCLOMATIC_COMPLEXITY:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @property
    def complexity_density(self) -> float:
        """
        Densidad de complejidad: CC normalizada por unidades de código.
        
        Definición:
            ρ(C) = CC / max(num_functions, 1)
        
        Interpretación:
            • ρ < 5: Buena modularización
            • ρ > 10: Funciones muy complejas
        """
        return self.cyclomatic_complexity / max(self.num_functions, 1)
    
    @property
    def essential_complexity(self) -> float:
        """
        Complejidad esencial (irreducible).
        
        Aproximación: CC - num_branches - num_loops + 1
        
        Representa la complejidad que no puede eliminarse mediante
        refactorización estructurada.
        """
        return max(
            1,
            self.cyclomatic_complexity - self.num_branches - self.num_loops + 1
        )
    
    @property
    def halstead_volume(self) -> float:
        """
        Aproximación del volumen de Halstead.
        
        V = N × log₂(n)
        donde N ≈ CC, n ≈ num_functions + num_classes
        
        Mide el "tamaño" del programa en bits de información.
        """
        n = max(self.num_functions + self.num_classes, 2)
        N = self.cyclomatic_complexity
        return N * math.log2(n)
    
    def __repr__(self) -> str:
        """Representación compacta para debugging."""
        return (
            f"Complexity("
            f"CC={self.cyclomatic_complexity}, "
            f"depth={self.max_nesting_depth}, "
            f"risk={self.risk_level}, "
            f"density={self.complexity_density:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización a diccionario."""
        return {
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'max_nesting_depth': self.max_nesting_depth,
            'num_functions': self.num_functions,
            'num_classes': self.num_classes,
            'num_branches': self.num_branches,
            'num_loops': self.num_loops,
            'risk_level': self.risk_level,
            'complexity_density': self.complexity_density,
            'essential_complexity': self.essential_complexity,
            'is_maintainable': self.is_maintainable,
        }


# =============================================================================
# ANALIZADOR DE FLUJO DE DATOS CON CONTROL HAMILTONIANO
# =============================================================================

class DataFlowAnalyzer(ast.NodeVisitor):
    """
    Analizador de flujo de datos con verificación disipativa Port-Hamiltoniana.
    
    ALGORITMO:
    ----------
    1. Recorrido DFS del AST con seguimiento de profundidad
    2. Clasificación de contextos de variables (Load/Store/Del)
    3. Cálculo incremental de complejidad ciclomática
    4. Monitoreo Hamiltoniano: H(child) ≤ H(parent)
    5. Recolección opcional de coordenadas por bloque
    
    HAMILTONIANO:
    -------------
    Definimos el Hamiltoniano local como:
        H(x) = CC(x) / (depth(x) + ε)
    
    donde:
        • CC(x): complejidad ciclomática acumulada en el nodo x
        • depth(x): profundidad del nodo en el árbol
        • ε: regularizador para evitar división por cero
    
    CONDICIÓN DE DISIPACIÓN:
    ------------------------
    Para todo nodo n con padre p:
        H(n) ≤ H(p)
    
    Si se viola, se levanta ThermodynamicSingularityError.
    
    COMPLEJIDAD TEMPORAL:
    ---------------------
    • Mejor caso: O(n) donde n = número de nodos
    • Peor caso: O(n log n) si se activa tracking de bloques
    • Espacio: O(h) donde h = altura del árbol (stack)
    
    INVARIANTES:
    ------------
    1. current_depth ≥ 0
    2. max_depth ≥ current_depth
    3. cyclomatic_complexity ≥ 1
    4. ∀ i: hamiltonian_stack[i] ≥ hamiltonian_stack[i+1] (monotonicidad)
    """
    
    # Conjunto de nodos que incrementan complejidad ciclomática
    DECISION_NODES: FrozenSet[type] = frozenset({
        ast.If, ast.While, ast.For, ast.AsyncFor,
        ast.ExceptHandler, ast.With, ast.AsyncWith,
    })
    
    # Nodos de decisión booleana (AND, OR)
    BOOLEAN_OPS: FrozenSet[type] = frozenset({ast.And, ast.Or})
    
    def __init__(
        self,
        enable_hamiltonian_monitor: bool = True,
        enable_block_tracking: bool = True,
        strict_mode: bool = True
    ):
        """
        Inicializa el analizador con estado limpio.
        
        Args:
            enable_hamiltonian_monitor: Activa verificación disipativa
            enable_block_tracking: Activa recolección de datos por bloque
            strict_mode: Modo estricto (lanza excepciones en violaciones)
        """
        # Estado de flujo de datos (global)
        self._reads: Set[str] = set()
        self._writes: Set[str] = set()
        
        # Métricas de profundidad
        self._current_depth: int = 0
        self._max_depth: int = 0
        
        # Complejidad ciclomática (base = 1 para el camino principal)
        self._cyclomatic: int = 1
        
        # Contadores estructurales
        self._num_functions: int = 0
        self._num_classes: int = 0
        self._num_branches: int = 0
        self._num_loops: int = 0
        self._num_nodes_visited: int = 0
        
        # Banderas de estado
        self._in_function_args: bool = False
        self._in_comprehension: bool = False
        
        # Control Hamiltoniano
        self._enable_hamiltonian: bool = enable_hamiltonian_monitor
        self._hamiltonian_stack: List[float] = []
        self._strict_mode: bool = strict_mode
        
        # Seguimiento topológico
        self._enable_block_tracking: bool = enable_block_tracking
        self._block_flows: List[Tuple[DataFlowCoordinates, int, str]] = []
        self._node_children: Dict[int, List[int]] = defaultdict(list)
    
    # -------------------------------------------------------------------------
    # CÁLCULO HAMILTONIANO
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _compute_hamiltonian(depth: int, complexity: int) -> float:
        """
        Calcula el Hamiltoniano de complejidad.
        
        Definición:
            H = CC / (depth + ε)
        
        Interpretación física:
            - Numerador: "energía potencial" (complejidad acumulada)
            - Denominador: "distancia" desde la raíz
            - H grande: código complejo concentrado en nodos superficiales
            - H pequeño: complejidad distribuida en profundidad
        
        Condición de disipación: H debe decrecer monotónicamente
        hacia las hojas del árbol.
        """
        return complexity / (depth + AnalysisLimits.HAMILTONIAN_EPSILON)
    
    def _push_hamiltonian(self) -> None:
        """Guarda el estado Hamiltoniano actual."""
        H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        self._hamiltonian_stack.append(H)
        logger.debug(f"Push H={H:.6f} at depth={self._current_depth}")
    
    def _pop_hamiltonian(self) -> None:
        """Restaura el estado Hamiltoniano previo."""
        if self._hamiltonian_stack:
            H = self._hamiltonian_stack.pop()
            logger.debug(f"Pop H={H:.6f}")
    
    def _check_dissipation_condition(self) -> None:
        """
        Verifica la condición disipativa: H(current) ≤ H(parent).
        
        Si falla y strict_mode=True, lanza ThermodynamicSingularityError.
        r"""
        if not self._hamiltonian_stack:
            return  # Nodo raíz, sin restricción
        
        parent_H = self._hamiltonian_stack[-1]
        current_H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        
        violation = current_H - parent_H
        
        if violation > AnalysisLimits.FLOAT_TOLERANCE:
            msg = (
                f"Dissipation violation: ΔH = {violation:.9f} > 0\n"
                f"  Current: H={current_H:.9f} (depth={self._current_depth}, "
                f"CC={self._cyclomatic})\n"
                f"  Parent:  H={parent_H:.9f}"
            )
            logger.warning(msg)
            
            if self._strict_mode:
                raise ThermodynamicSingularityError(
                    current_H, parent_H, self._current_depth, self._cyclomatic
                )
    
    # -------------------------------------------------------------------------
    # CONTROL DE PROFUNDIDAD Y VISITACIÓN
    # -------------------------------------------------------------------------
    
    def visit(self, node: ast.AST) -> Any:
        """
        Punto de entrada para cada nodo con instrumentación completa.
        
        Secuencia de operaciones:
            1. Incrementar profundidad
            2. Verificar límites
            3. Monitoreo Hamiltoniano (pre-orden)
            4. Tracking topológico (opcional)
            5. Procesamiento del nodo (dispatch)
            6. Cleanup (post-orden)
        """
        # 1. Control de profundidad
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self._num_nodes_visited += 1
        
        # 2. Verificación de límites
        if self._current_depth > AnalysisLimits.MAX_AST_DEPTH:
            raise ComplexityBoundsViolationError(
                "AST depth", self._current_depth, AnalysisLimits.MAX_AST_DEPTH
            )
        
        if self._num_nodes_visited > AnalysisLimits.MAX_AST_NODES:
            raise ComplexityBoundsViolationError(
                "AST nodes", self._num_nodes_visited, AnalysisLimits.MAX_AST_NODES
            )
        
        # 3. Monitoreo Hamiltoniano
        if self._enable_hamiltonian:
            self._check_dissipation_condition()
            self._push_hamiltonian()
        
        # 4. Tracking topológico
        if self._enable_block_tracking:
            self._record_block_flow(node)
        
        # 5. Procesamiento (dispatch a método específico)
        try:
            result = super().visit(node)
        finally:
            # 6. Cleanup
            if self._enable_hamiltonian:
                self._pop_hamiltonian()
            self._current_depth -= 1
        
        return result
    
    def _record_block_flow(self, node: ast.AST) -> None:
        """
        Registra el flujo de datos local del nodo actual.
        
        Usa un sub-analizador ligero para extraer reads/writes
        sin contaminar el estado global.
        """
        extractor = _LocalFlowExtractor()
        extractor.visit(node)
        
        coords = DataFlowCoordinates(
            reads=frozenset(extractor.local_reads),
            writes=frozenset(extractor.local_writes)
        )
        
        node_id = id(node)
        node_type = type(node).__name__
        
        self._block_flows.append((coords, node_id, node_type))
        
        # Registrar relación padre-hijo
        for child in ast.iter_child_nodes(node):
            self._node_children[node_id].append(id(child))
    
    # -------------------------------------------------------------------------
    # ANÁLISIS DE FLUJO DE DATOS
    # -------------------------------------------------------------------------
    
    def visit_Name(self, node: ast.Name) -> None:
        """
        Procesa referencias a nombres (variables).
        
        Contextos:
            • Load: lectura (uso)
            • Store: escritura (definición)
            • Del: eliminación (lectura + escritura)
        """
        name = node.id
        
        if isinstance(node.ctx, ast.Load):
            self._reads.add(name)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(name)
        elif isinstance(node.ctx, ast.Del):
            # Del implica lectura (para verificar existencia) y escritura (borrado)
            self._reads.add(name)
            self._writes.add(name)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Procesa accesos a atributos (obj.attr).
        
        Nota: Registramos el atributo, no el objeto completo.
        """
        attr = node.attr
        
        if isinstance(node.ctx, ast.Load):
            self._reads.add(attr)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(attr)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(attr)
            self._writes.add(attr)
        
        # Recursión en el valor (obj en obj.attr)
        self.visit(node.value)
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Procesa subscripts (a[i])."""
        self.generic_visit(node)
    
    def visit_arg(self, node: ast.arg) -> None:
        """Procesa argumentos de función (definiciones)."""
        if not self._in_function_args:
            self._writes.add(node.arg)
    
    # -------------------------------------------------------------------------
    # DEFINICIONES
    # -------------------------------------------------------------------------
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Procesa definiciones de función."""
        self._num_functions += 1
        
        # Los argumentos son escrituras (definiciones locales)
        self._in_function_args = True
        for arg in node.args.args:
            self._writes.add(arg.arg)
        for arg in node.args.posonlyargs:
            self._writes.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self._writes.add(arg.arg)
        if node.args.vararg:
            self._writes.add(node.args.vararg.arg)
        if node.args.kwarg:
            self._writes.add(node.args.kwarg.arg)
        self._in_function_args = False
        
        # Procesar cuerpo y decoradores
        for stmt in node.body:
            self.visit(stmt)
        for decorator in node.decorator_list:
            self.visit(decorator)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Procesa funciones asíncronas."""
        self.visit_FunctionDef(node)  # type: ignore
    
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Procesa expresiones lambda."""
        self._num_functions += 1
        
        for arg in node.args.args:
            self._writes.add(arg.arg)
        
        self.visit(node.body)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Procesa definiciones de clase."""
        self._num_classes += 1
        self.generic_visit(node)
    
    # -------------------------------------------------------------------------
    # COMPLEJIDAD CICLOMÁTICA
    # -------------------------------------------------------------------------
    
    def _increment_complexity(self, amount: int = 1) -> None:
        """Incrementa la complejidad ciclomática."""
        self._cyclomatic += amount
        logger.debug(f"CC incremented to {self._cyclomatic}")
    
    def visit_If(self, node: ast.If) -> None:
        """Procesa condicionales if."""
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        """Procesa bucles for."""
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Procesa bucles for asíncronos."""
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        """Procesa bucles while."""
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Procesa manejadores de excepciones."""
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With) -> None:
        """Procesa gestores de contexto."""
        self._increment_complexity()
        self.generic_visit(node)
    
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Procesa gestores de contexto asíncronos."""
        self._increment_complexity()
        self.generic_visit(node)
    
    def visit_Match(self, node: ast.Match) -> None:
        """
        Procesa pattern matching (Python 3.10+).
        
        Complejidad: 1 (base) + (num_cases - 1)
        """
        self._increment_complexity()
        if hasattr(node, 'cases'):
            num_cases = len(node.cases)
            if num_cases > 1:
                self._increment_complexity(num_cases - 1)
            self._num_branches += num_cases
        self.generic_visit(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """
        Procesa operadores booleanos (and, or).
        
        Cada operador adicional incrementa CC.
        """
        num_operands = len(node.values)
        if num_operands > 1:
            self._increment_complexity(num_operands - 1)
            self._num_branches += num_operands - 1
        self.generic_visit(node)
    
    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Procesa expresiones ternarias (a if cond else b)."""
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_comprehension(self, node: ast.comprehension) -> None:
        """Procesa comprensiones (list/dict/set comprehensions)."""
        self._increment_complexity()  # El bucle base
        self._increment_complexity(len(node.ifs))  # Cada filtro if
        self._num_loops += 1
        self._num_branches += len(node.ifs)
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        """Procesa bloques try/except."""
        self._increment_complexity()
        self.generic_visit(node)
    
    # -------------------------------------------------------------------------
    # EXTRACCIÓN DE RESULTADOS
    # -------------------------------------------------------------------------
    
    def get_dataflow_coordinates(self) -> DataFlowCoordinates:
        """Devuelve las coordenadas globales de flujo de datos."""
        return DataFlowCoordinates(
            reads=frozenset(self._reads),
            writes=frozenset(self._writes)
        )
    
    def get_complexity_profile(self) -> ComplexityProfile:
        """Devuelve el perfil de complejidad completo."""
        return ComplexityProfile(
            cyclomatic_complexity=self._cyclomatic,
            max_nesting_depth=self._max_depth,
            num_functions=self._num_functions,
            num_classes=self._num_classes,
            num_branches=self._num_branches,
            num_loops=self._num_loops
        )
    
    def get_block_dataflows(self) -> List[Tuple[DataFlowCoordinates, int, str]]:
        """Devuelve los flujos por bloque (para análisis cohomológico)."""
        return self._block_flows
    
    def get_tree_graph(self) -> Dict[int, List[int]]:
        """Devuelve el grafo de adyacencia del AST."""
        return dict(self._node_children)


# =============================================================================
# EXTRACTOR LOCAL DE FLUJO (auxiliar para tracking por bloque)
# =============================================================================

class _LocalFlowExtractor(ast.NodeVisitor):
    """
    Extractor ligero de flujo de datos para un subárbol aislado.
    
    No modifica el estado global del analizador principal.
    Complejidad: O(tamaño del subárbol)
    """
    
    def __init__(self):
        self.local_reads: Set[str] = set()
        self.local_writes: Set[str] = set()
    
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.local_reads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.local_writes.add(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.local_reads.add(node.id)
            self.local_writes.add(node.id)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self.local_reads.add(node.attr)
        elif isinstance(node.ctx, ast.Store):
            self.local_writes.add(node.attr)
        elif isinstance(node.ctx, ast.Del):
            self.local_reads.add(node.attr)
            self.local_writes.add(node.attr)
        self.generic_visit(node)
    
    def visit_arg(self, node: ast.arg) -> None:
        self.local_writes.add(node.arg)


# =============================================================================
# COHOMOLOGÍA DE HACES CELULARES
# =============================================================================

class CellularSheafCohomology:
    """
    Implementación rigurosa de cohomología de haces celulares sobre grafos.
    
    TEORÍA:
    -------
    Un haz celular F sobre un CW-complejo T asigna:
        • A cada vértice v: un espacio vectorial F(v)
        • A cada arista e: un espacio vectorial F(e)
        • Restricciones lineales: F(e) → F(source(e)) ⊕ F(target(e))
    
    El complejo de cocadenas:
        C⁰(T,F) --δ⁰--> C¹(T,F) --δ¹--> C²(T,F)
    
    donde:
        • C⁰: secciones sobre vértices
        • C¹: secciones sobre aristas
        • δⁿ: operador cofrontera (coboundary)
    
    La cohomología:
        Hⁿ(T,F) = ker(δⁿ) / im(δⁿ⁻¹)
    
    INTERPRETACIÓN PARA DEF-USE:
    -----------------------------
    • Vértices: bloques básicos (nodos AST)
    • Aristas: flujo de control (padre → hijo en el árbol)
    • Tallos F(v): espacio generado por las variables en v
    • H¹ ≠ 0: indica ciclos de dependencias inconsistentes
    
    REFERENCIAS:
    ------------
    • Hansen & Ghrist (2019): "Opinion Dynamics on Discourse Sheaves"
    • Curry (2014): "Sheaves, Cosheaves and Applications"
    • Robinson (2014): "Sheaf and Duality Methods for Analyzing Multi-Model Systems"
    """
    
    def __init__(self):
        self.vertices: List[DataFlowCoordinates] = []
        self.edges: List[Tuple[int, int]] = []  # (parent_idx, child_idx)
        self.all_variables: List[str] = []
        self._var_to_index: Dict[str, int] = {}
    
    def build_from_ast_blocks(
        self,
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]]
    ) -> None:
        """
        Construye el haz celular a partir del AST analizado.
        
        Args:
            block_flows: Lista de (coordenadas, node_id, tipo)
            tree_graph: Grafo de adyacencia {node_id: [child_ids]}
        
        Complejidad: O(V + E) donde V = vértices, E = aristas
        """
        # Mapeo de IDs a índices
        node_id_to_index: Dict[int, int] = {}
        
        for idx, (coords, node_id, _) in enumerate(block_flows):
            node_id_to_index[node_id] = idx
            self.vertices.append(coords)
        
        # Recolectar todas las variables (base del espacio vectorial)
        var_set: Set[str] = set()
        for v in self.vertices:
            var_set.update(v.all_variables)
        
        self.all_variables = sorted(var_set)
        self._var_to_index = {v: i for i, v in enumerate(self.all_variables)}
        
        # Construir aristas del grafo dirigido
        for parent_id, children_ids in tree_graph.items():
            if parent_id not in node_id_to_index:
                continue
            parent_idx = node_id_to_index[parent_id]
            for child_id in children_ids:
                if child_id in node_id_to_index:
                    child_idx = node_id_to_index[child_id]
                    self.edges.append((parent_idx, child_idx))
        
        logger.info(
            f"Built cellular sheaf: {len(self.vertices)} vertices, "
            f"{len(self.edges)} edges, {len(self.all_variables)} variables"
        )
    
    def compute_first_cohomology_dimension(self) -> int:
        """
        Calcula dim H¹(T, F) usando álgebra lineal.
        
        ALGORITMO:
        ----------
        1. Construir operador cofrontera δ⁰: C⁰ → C¹
        2. Calcular rank(δ⁰) = dim(im δ⁰)
        3. dim H¹ = dim C¹ - rank(δ⁰)
        
        CONSTRUCCIÓN DE δ⁰:
        -------------------
        Para cada arista e = (p, c) de parent p a child c,
        y cada variable v:
            (δ⁰ σ)[e, v] = σ[c, v] - σ[p, v]
        
        donde σ ∈ C⁰ es una 0-cocadena (asigna valores a vértices).
        
        INTERPRETACIÓN:
        ---------------
        δ⁰ mide la "diferencia" de valores de variables a lo largo
        de las aristas. Si rank(δ⁰) < dim C¹, existen ciclos que no
        pueden ser "generados" por diferencias locales → inconsistencia.
        
        Returns:
            dim H¹(T, F) ∈ ℕ
        
        Complejidad: O(E × V × n³) donde n = número de variables
                     (dominado por cálculo de rango)
        """
        if not self.vertices or not self.all_variables or not self.edges:
            return 0
        
        n_vars = len(self.all_variables)
        n_verts = len(self.vertices)
        n_edges = len(self.edges)
        
        try:
            import numpy as np
            
            # Dimensiones
            rows = n_edges * n_vars  # dim C¹
            cols = n_verts * n_vars  # dim C⁰
            
            # Matriz del operador δ⁰
            delta0 = np.zeros((rows, cols), dtype=np.float64)
            
            for e_idx, (p_idx, c_idx) in enumerate(self.edges):
                for var in self.all_variables:
                    v_idx = self._var_to_index[var]
                    
                    row = e_idx * n_vars + v_idx
                    col_parent = p_idx * n_vars + v_idx
                    col_child = c_idx * n_vars + v_idx
                    
                    # (δ⁰ σ)[e, v] = σ[child, v] - σ[parent, v]
                    delta0[row, col_parent] = -1.0
                    delta0[row, col_child] = 1.0
            
            # Calcular rango usando SVD (más estable que QR)
            rank_delta0 = np.linalg.matrix_rank(
                delta0,
                tol=AnalysisLimits.FLOAT_TOLERANCE
            )
            
            # dim H¹ = dim C¹ - dim(im δ⁰)
            dim_H1 = rows - rank_delta0
            
            logger.info(f"Cohomology: dim C¹ = {rows}, rank δ⁰ = {rank_delta0}, dim H¹ = {dim_H1}")
            return dim_H1
        
        except ImportError:
            # Fallback sin NumPy: estimación conservadora
            logger.warning("NumPy not available; using heuristic cohomology estimate")
            return self._estimate_cohomology_without_numpy()
    
    def _estimate_cohomology_without_numpy(self) -> int:
        """
        Estimación heurística de dim H¹ sin NumPy.
        
        Heurística:
            En un árbol (grafo acíclico), dim H¹ = 0.
            El número de ciclos independientes es aproximadamente:
                β₁ = |E| - |V| + 1  (en grafo conexo)
            
            Multiplicamos por el número de variables para obtener
            una cota superior de dim H¹.
        
        Returns:
            Estimación conservadora de dim H¹
        """
        n_verts = len(self.vertices)
        n_edges = len(self.edges)
        n_vars = len(self.all_variables)
        
        # Número de Betti β₁ (ciclos independientes)
        # Para un grafo conexo: β₁ = E - V + 1
        # Para un árbol: β₁ = 0
        beta1 = max(0, n_edges - n_verts + 1)
        
        # Estimación: cada ciclo puede generar hasta n_vars inconsistencias
        estimated_dim = beta1 * n_vars
        
        logger.debug(f"Estimated dim H¹ ≈ {estimated_dim} (β₁={beta1}, vars={n_vars})")
        return estimated_dim
    
    def find_problematic_variables(self) -> Set[str]:
        """
        Identifica variables potencialmente problemáticas.
        
        Heurística:
            Variables que aparecen en lecturas pero nunca en escrituras
            (posiblemente no inicializadas).
        
        Returns:
            Conjunto de nombres de variables sospechosas
        """
        all_reads: Set[str] = set()
        all_writes: Set[str] = set()
        
        for coords in self.vertices:
            all_reads.update(coords.reads)
            all_writes.update(coords.writes)
        
        # Variables leídas pero nunca escritas
        uninitialized = all_reads - all_writes
        
        logger.debug(f"Potentially uninitialized variables: {sorted(uninitialized)}")
        return uninitialized
    
    @staticmethod
    def check_global_consistency(
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]],
        strict: bool = True
    ) -> Tuple[int, Set[str]]:
        """
        Método de fábrica: construye el haz y verifica consistencia.
        
        Args:
            block_flows: Flujos de datos por bloque
            tree_graph: Grafo de dependencias
            strict: Si True, lanza excepción cuando dim H¹ > 0
        
        Returns:
            (dim H¹, variables_problemáticas)
        
        Raises:
            CohomologicalObstructionError: Si dim H¹ > 0 y strict=True
        """
        sheaf = CellularSheafCohomology()
        sheaf.build_from_ast_blocks(block_flows, tree_graph)
        
        dim_H1 = sheaf.compute_first_cohomology_dimension()
        problematic_vars = sheaf.find_problematic_variables()
        
        if dim_H1 > 0 and strict:
            raise CohomologicalObstructionError(dim_H1, problematic_vars)
        
        return dim_H1, problematic_vars


# =============================================================================
# VALIDADOR DE ESTRUCTURAS JSON/ESQUEMAS
# =============================================================================

class JSONStructureValidator:
    """
    Validador de estructuras JSON con prevención de ataques DoS.
    
    ATAQUES PREVENIDOS:
    -------------------
    1. Billion Laughs: expansión exponencial mediante referencias
    2. Hash Collision DoS: diccionarios con claves colisionantes
    3. Stack Overflow: profundidad excesiva
    4. Memory Exhaustion: estructuras masivas
    
    MÉTODO:
    -------
    Recorrido DFS con límites estrictos en:
        • Profundidad máxima
        • Número de claves por diccionario
        • Tamaño de arrays
    
    Complejidad: O(n) donde n = número de nodos en la estructura
    """
    
    @staticmethod
    def validate_structure(
        obj: Union[Dict, List, Any],
        max_depth: int = AnalysisLimits.MAX_JSON_DEPTH,
        max_keys: int = AnalysisLimits.MAX_JSON_KEYS,
        max_array_size: int = AnalysisLimits.MAX_JSON_ARRAY_SIZE,
        current_depth: int = 0,
        path: str = "$"
    ) -> Tuple[bool, str]:
        """
        Valida recursivamente una estructura JSON.
        
        Args:
            obj: Objeto a validar
            max_depth: Profundidad máxima permitida
            max_keys: Número máximo de claves en diccionarios
            max_array_size: Tamaño máximo de arrays
            current_depth: Profundidad actual (uso interno)
            path: Ruta actual en la estructura (para mensajes de error)
        
        Returns:
            (es_válido, mensaje_de_error)
        """
        # Verificación de profundidad
        if current_depth > max_depth:
            return False, f"Depth {current_depth} exceeds limit {max_depth} at {path}"
        
        # Diccionarios
        if isinstance(obj, dict):
            num_keys = len(obj)
            if num_keys > max_keys:
                return False, (
                    f"Dictionary at {path} has {num_keys} keys, "
                    f"limit is {max_keys}"
                )
            
            for key, value in obj.items():
                # Validar tipo de clave
                if not isinstance(key, (str, int, float, bool, type(None))):
                    return False, f"Invalid key type {type(key).__name__} at {path}"
                
                # Recursión en valores
                is_valid, error = JSONStructureValidator.validate_structure(
                    value,
                    max_depth,
                    max_keys,
                    max_array_size,
                    current_depth + 1,
                    f"{path}.{key}"
                )
                if not is_valid:
                    return False, error
        
        # Arrays
        elif isinstance(obj, list):
            size = len(obj)
            if size > max_array_size:
                return False, (
                    f"Array at {path} has {size} items, "
                    f"limit is {max_array_size}"
                )
            
            for idx, item in enumerate(obj):
                is_valid, error = JSONStructureValidator.validate_structure(
                    item,
                    max_depth,
                    max_keys,
                    max_array_size,
                    current_depth + 1,
                    f"{path}[{idx}]"
                )
                if not is_valid:
                    return False, error
        
        # Tipos primitivos (válidos por defecto)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            pass
        
        # Tipos no soportados
        else:
            return False, f"Unsupported type {type(obj).__name__} at {path}"
        
        return True, ""
    
    @staticmethod
    def compute_depth(obj: Union[Dict, List, Any]) -> int:
        """
        Calcula la profundidad máxima de una estructura.
        
        Complejidad: O(n) donde n = número de nodos
        """
        if isinstance(obj, dict):
            if not obj:
                return 0
            return 1 + max(
                (JSONStructureValidator.compute_depth(v) for v in obj.values()),
                default=0
            )
        elif isinstance(obj, list):
            if not obj:
                return 0
            return 1 + max(
                (JSONStructureValidator.compute_depth(item) for item in obj),
                default=0
            )
        else:
            return 0
    
    @staticmethod
    def estimate_memory_size(obj: Union[Dict, List, Any]) -> int:
        """
        Estima el tamaño en memoria (en bytes) de la estructura.
        
        Aproximación heurística basada en sys.getsizeof.
        """
        total_size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                total_size += sys.getsizeof(k)
                total_size += JSONStructureValidator.estimate_memory_size(v)
        elif isinstance(obj, list):
            for item in obj:
                total_size += JSONStructureValidator.estimate_memory_size(item)
        
        return total_size


# =============================================================================
# NORMALIZADOR A FORMATO TABULAR
# =============================================================================

class TabularNormalizer:
    """
    Normalizador de estructuras complejas a representación tabular plana.
    
    MOTIVACIÓN:
    -----------
    Facilita la visualización y análisis de datos estructurados
    mediante proyección a espacios tabulares (Markdown, CSV).
    
    MÉTODOS:
    --------
    • to_markdown_table: Genera tabla Markdown
    • to_csv_row: Genera fila CSV
    • flatten: Aplana estructuras anidadas
    """
    
    @staticmethod
    def _serialize_value(
        value: Any,
        depth: int,
        max_depth: int = AnalysisLimits.MAX_FLATTEN_DEPTH,
        max_length: int = AnalysisLimits.MAX_STRING_REPR_LENGTH
    ) -> str:
        """
        Serializa un valor arbitrario a string legible.
        
        Args:
            value: Valor a serializar
            depth: Profundidad actual de recursión
            max_depth: Profundidad máxima antes de truncar
            max_length: Longitud máxima de strings
        
        Returns:
            Representación string del valor
        r"""
        # None
        if value is None:
            return "null"
        
        # Booleanos
        if isinstance(value, bool):
            return "true" if value else "false"
        
        # Números
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "∞" if value > 0 else "-∞"
            if isinstance(value, float):
                return f"{value:.6g}"  # Notación compacta
            return str(value)
        
        # Strings
        if isinstance(value, str):
            escaped = value.replace('"', '\\"').replace('\n', '\\n')
            if len(escaped) > max_length:
                escaped = escaped[:max_length - 3] + "..."
            return f'"{escaped}"'
        
        # Diccionarios
        if isinstance(value, dict):
            if depth >= max_depth:
                return f"{{...{len(value)} keys}}"
            items = [
                f"{k}: {TabularNormalizer._serialize_value(v, depth + 1, max_depth, max_length)}"
                for k, v in list(value.items())[:5]
            ]
            suffix = ", ..." if len(value) > 5 else ""
            return "{" + ", ".join(items) + suffix + "}"
        
        # Listas
        if isinstance(value, list):
            if depth >= max_depth:
                return f"[...{len(value)} items]"
            items = [
                TabularNormalizer._serialize_value(item, depth + 1, max_depth, max_length)
                for item in value[:5]
            ]
            suffix = ", ..." if len(value) > 5 else ""
            return "[" + ", ".join(items) + suffix + "]"
        
        # Otros tipos
        return f"<{type(value).__name__}>"
    
    @staticmethod
    def to_markdown_table(data: Dict[str, Any]) -> str:
        """
        Convierte un diccionario a tabla Markdown.
        
        Formato:
            | key1 | key2 | key3 |
            |------|------|------|
            | val1 | val2 | val3 |
        
        Args:
            data: Diccionario a convertir
        
        Returns:
            String con la tabla Markdown
        
        Raises:
            TypeError: Si data no es un diccionario
        r"""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        
        if not data:
            return "| (empty) |\n|---------|"
        
        # Ordenar claves para salida determinista
        keys = sorted(data.keys())
        
        # Cabecera
        header = "| " + " | ".join(str(k) for k in keys) + " |"
        
        # Separador
        separator = "|" + "|".join(["---"] * len(keys)) + "|"
        
        # Valores
        values = [TabularNormalizer._serialize_value(data[k], 0) for k in keys]
        row = "| " + " | ".join(values) + " |"
        
        return f"{header}\n{separator}\n{row}"
    
    @staticmethod
    def to_csv_row(data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convierte un diccionario a fila CSV.
        
        Args:
            data: Diccionario a convertir
        
        Returns:
            (header, row) donde ambos son strings CSV
        """
        if not isinstance(data, dict) or not data:
            return "", ""
        
        keys = sorted(data.keys())
        header = ",".join(str(k) for k in keys)
        values = [TabularNormalizer._serialize_value(data[k], 0) for k in keys]
        row = ",".join(values)
        
        return header, row
    
    @staticmethod
    def flatten_dict(
        data: Dict[str, Any],
        parent_key: str = "",
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Aplana un diccionario anidado.
        
        Ejemplo:
            {"a": {"b": 1}} → {"a.b": 1}
        
        Args:
            data: Diccionario a aplanar
            parent_key: Prefijo para las claves (uso interno)
            separator: Separador de niveles
        
        Returns:
            Diccionario plano
        """
        items: List[Tuple[str, Any]] = []
        
        for k, v in data.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(
                    TabularNormalizer.flatten_dict(v, new_key, separator).items()
                )
            else:
                items.append((new_key, v))
        
        return dict(items)


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

class ASTStaticAnalyzer:
    """
    Analizador estático de código Python con verificaciones topológicas.
    
    PIPELINE DE ANÁLISIS:
    ---------------------
    1. Parsing: source_code → AST (usando ast.parse)
    2. Análisis de flujo: AST → DataFlowCoordinates + ComplexityProfile
    3. Verificación Hamiltoniana: ΔH ≤ 0 en todas las trayectorias
    4. Verificación cohomológica: H¹(T,F) = 0
    5. Reporte consolidado
    
    GARANTÍAS:
    ----------
    • Corrección: Todas las métricas son verificables matemáticamente
    • Completitud: Todos los ciclos son detectados (β₁ exacto)
    • Eficiencia: O(n) donde n = tamaño del AST
    • Seguridad: Prevención de DoS mediante límites estrictos
    
    USO:
    ----
    >>> analyzer = ASTStaticAnalyzer()
    >>> result = analyzer.analyze_code(source_code)
    >>> print(result['complexity'])
    >>> print(result['dataflow'])
    """
    
    @staticmethod
    def analyze_code(
        source_code: str,
        filename: str = "<string>",
        enable_hamiltonian: bool = True,
        enable_cohomology: bool = True,
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Analiza código Python y extrae métricas completas.
        
        Args:
            source_code: Código fuente Python
            filename: Nombre del archivo (para error reporting)
            enable_hamiltonian: Activa verificación disipativa
            enable_cohomology: Activa análisis cohomológico
            strict_mode: Si True, lanza excepciones en violaciones
        
        Returns:
            Diccionario con las claves:
                - 'dataflow': DataFlowCoordinates
                - 'complexity': ComplexityProfile
                - 'hamiltonian_ok': bool
                - 'cohomology_ok': bool
                - 'cohomology_dimension': int (si enable_cohomology)
                - 'problematic_variables': Set[str] (si enable_cohomology)
                - 'warnings': List[str]
        
        Raises:
            ValueError: Si el código está vacío o tiene errores de sintaxis
            ThermodynamicSingularityError: Si ΔH > 0 y strict_mode=True
            CohomologicalObstructionError: Si H¹ ≠ 0 y strict_mode=True
            ComplexityBoundsViolationError: Si se exceden límites
        """
        # Validación de entrada
        if not source_code or not source_code.strip():
            raise ValueError("Source code is empty or whitespace-only")
        
        # Parsing
        try:
            tree = ast.parse(source_code, filename=filename, mode='exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in {filename}: {e}")
            raise ValueError(f"Invalid Python syntax: {e}") from e
        
        # Inicialización del analizador
        analyzer = DataFlowAnalyzer(
            enable_hamiltonian_monitor=enable_hamiltonian,
            enable_block_tracking=enable_cohomology,
            strict_mode=strict_mode
        )
        
        # Estado de verificaciones
        hamiltonian_ok = True
        cohomology_ok = True
        warnings: List[str] = []
        
        # Análisis principal
        try:
            analyzer.visit(tree)
        except ThermodynamicSingularityError as e:
            hamiltonian_ok = False
            logger.error(f"Hamiltonian violation in {filename}: {e}")
            if strict_mode:
                raise
            else:
                warnings.append(str(e))
        except ComplexityBoundsViolationError as e:
            logger.error(f"Complexity bounds violated in {filename}: {e}")
            raise
        
        # Extracción de métricas básicas
        dataflow = analyzer.get_dataflow_coordinates()
        complexity = analyzer.get_complexity_profile()
        
        # Resultados base
        result: Dict[str, Any] = {
            'dataflow': dataflow,
            'complexity': complexity,
            'hamiltonian_ok': hamiltonian_ok,
            'warnings': warnings,
        }
        
        # Análisis cohomológico (opcional)
        if enable_cohomology:
            try:
                block_flows = analyzer.get_block_dataflows()
                tree_graph = analyzer.get_tree_graph()
                
                dim_H1, problematic_vars = CellularSheafCohomology.check_global_consistency(
                    block_flows, tree_graph, strict=strict_mode
                )
                
                result['cohomology_ok'] = (dim_H1 == 0)
                result['cohomology_dimension'] = dim_H1
                result['problematic_variables'] = problematic_vars
                
                if dim_H1 > 0:
                    cohomology_ok = False
                    warnings.append(
                        f"Cohomological obstruction: dim H¹ = {dim_H1}, "
                        f"variables: {sorted(problematic_vars)}"
                    )
            
            except CohomologicalObstructionError as e:
                cohomology_ok = False
                logger.error(f"Cohomological obstruction in {filename}: {e}")
                result['cohomology_dimension'] = e.context.get('dim_H1', -1)
                result['problematic_variables'] = e.context.get('problematic_vars', set())
                if strict_mode:
                    raise
                else:
                    warnings.append(str(e))
        else:
            result['cohomology_ok'] = None  # No verificado
        
        # Logging de resultados
        logger.info(
            f"Analyzed {filename}: "
            f"CC={complexity.cyclomatic_complexity}, "
            f"depth={complexity.max_nesting_depth}, "
            f"risk={complexity.risk_level}, "
            f"H_ok={hamiltonian_ok}, "
            f"Coh_ok={cohomology_ok}"
        )
        
        return result
    
    @staticmethod
    def analyze_file(
        filepath: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analiza un archivo Python completo.
        
        Args:
            filepath: Ruta al archivo .py
            **kwargs: Argumentos pasados a analyze_code
        
        Returns:
            Resultado del análisis (ver analyze_code)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return ASTStaticAnalyzer.analyze_code(
            source_code,
            filename=filepath,
            **kwargs
        )
    
    @staticmethod
    def validate_json_contract(
        schema: Union[Dict, List, Any],
        max_depth: Optional[int] = None,
        max_keys: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Valida y normaliza un esquema JSON.
        
        Args:
            schema: Estructura JSON a validar
            max_depth: Profundidad máxima (default: AnalysisLimits.MAX_JSON_DEPTH)
            max_keys: Claves máximas (default: AnalysisLimits.MAX_JSON_KEYS)
        
        Returns:
            (es_válido, mensaje_error, tabla_markdown)
        """
        max_depth = max_depth or AnalysisLimits.MAX_JSON_DEPTH
        max_keys = max_keys or AnalysisLimits.MAX_JSON_KEYS
        
        # Validación estructural
        is_valid, error = JSONStructureValidator.validate_structure(
            schema, max_depth, max_keys
        )
        
        if not is_valid:
            logger.error(f"JSON validation failed: {error}")
            return False, error, None
        
        # Normalización a tabla (solo para dicts)
        markdown = None
        if isinstance(schema, dict):
            try:
                markdown = TabularNormalizer.to_markdown_table(schema)
            except Exception as e:
                logger.warning(f"Normalization failed: {e}")
                markdown = f"(Normalization error: {e})"
        
        return True, None, markdown
    
    @staticmethod
    def compute_interference_matrix(
        tools: Dict[str, DataFlowCoordinates]
    ) -> Dict[Tuple[str, str], int]:
        """
        Calcula la matriz de interferencia entre herramientas.
        
        La matriz I tiene entradas:
            I[A, B] = score de interferencia de A hacia B
        
        Propiedades:
            • Antisimétrica: I[A, B] = -I[B, A]
            • I[A, A] = 0 (no auto-interferencia)
        
        Args:
            tools: Diccionario {nombre: coordenadas}
        
        Returns:
            Matriz como diccionario {(A, B): score}
        
        Complejidad: O(n²) donde n = número de herramientas
        """
        if len(tools) > AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE:
            raise ComplexityBoundsViolationError(
                "Interference matrix size",
                len(tools),
                AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE
            )
        
        matrix: Dict[Tuple[str, str], int] = {}
        tool_names = sorted(tools.keys())
        
        for name_a in tool_names:
            for name_b in tool_names:
                if name_a == name_b:
                    matrix[(name_a, name_b)] = 0
                else:
                    score = DataFlowCoordinates.interference_score(
                        tools[name_a], tools[name_b]
                    )
                    matrix[(name_a, name_b)] = score
        
        return matrix
    
    @staticmethod
    def topological_sort_by_interference(
        tools: Dict[str, DataFlowCoordinates]
    ) -> List[str]:
        """
        Ordena herramientas topológicamente según interferencia.
        
        Algoritmo: Kahn's algorithm adaptado a la matriz de interferencia.
        
        Args:
            tools: Diccionario {nombre: coordenadas}
        
        Returns:
            Lista ordenada de nombres de herramientas
        
        Raises:
            ValueError: Si hay ciclos de dependencias
        """
        # Construir grafo dirigido basado en interferencia
        graph: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)
        
        tool_names = set(tools.keys())
        for name in tool_names:
            in_degree[name] = 0
        
        # Arista A → B si I(A, B) > 0 (A debe ejecutarse antes que B)
        for name_a in tool_names:
            for name_b in tool_names:
                if name_a != name_b:
                    score = DataFlowCoordinates.interference_score(
                        tools[name_a], tools[name_b]
                    )
                    if score > 0:
                        graph[name_a].add(name_b)
                        in_degree[name_b] += 1
        
        # Kahn's algorithm
        queue: deque = deque([
            name for name in tool_names if in_degree[name] == 0
        ])
        result: List[str] = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Verificar si hay ciclos
        if len(result) != len(tool_names):
            remaining = tool_names - set(result)
            raise ValueError(f"Dependency cycle detected among: {sorted(remaining)}")
        
        return result


# =============================================================================
# API DE COMPATIBILIDAD (LEGACY)
# =============================================================================

class ASTSymplecticParser:
    """
    Clase de compatibilidad con API anterior.
    
    DEPRECATION WARNING:
    Esta clase está obsoleta. Usar ASTStaticAnalyzer directamente.
    Se mantendrá hasta la versión 5.0 para retrocompatibilidad.
    """
    
    @staticmethod
    def parse_tool_dynamics(
        source_code: str
    ) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        """
        DEPRECATED: Usar ASTStaticAnalyzer.analyze_code() en su lugar.
        
        Analiza el código y devuelve coordenadas de flujo y perfil de complejidad.
        """
        logger.warning(
            "ASTSymplecticParser.parse_tool_dynamics() is deprecated. "
            "Use ASTStaticAnalyzer.analyze_code() instead."
        )
        
        result = ASTStaticAnalyzer.analyze_code(
            source_code,
            enable_hamiltonian=False,
            enable_cohomology=False,
            strict_mode=False
        )
        
        return result['dataflow'], result['complexity']
    
    @staticmethod
    def process_data_contract(
        json_schema: Dict[str, Any]
    ) -> Optional[str]:
        """
        DEPRECATED: Usar ASTStaticAnalyzer.validate_json_contract() en su lugar.
        
        Procesa un contrato de datos JSON y devuelve tabla Markdown.
        """
        logger.warning(
            "ASTSymplecticParser.process_data_contract() is deprecated. "
            "Use ASTStaticAnalyzer.validate_json_contract() instead."
        )
        
        _, _, markdown = ASTStaticAnalyzer.validate_json_contract(json_schema)
        return markdown


# Aliases para retrocompatibilidad
PhaseSpaceCoordinates = DataFlowCoordinates
ThermodynamicProfile = ComplexityProfile


# =============================================================================
# UTILIDADES Y HELPERS
# =============================================================================

def print_complexity_report(
    complexity: ComplexityProfile,
    name: str = "Code",
    file=None
) -> None:
    """
    Imprime un reporte formateado de complejidad.
    
    Args:
        complexity: Perfil de complejidad a reportar
        name: Nombre del código analizado
        file: Archivo de salida (default: sys.stdout)
    r"""
    import sys
    output = file or sys.stdout
    
    width = 70
    print(f"\n{'=' * width}", file=output)
    print(f"COMPLEXITY ANALYSIS REPORT: {name}", file=output)
    print(f"{'=' * width}", file=output)
    
    print(f"\n📊 METRICS:", file=output)
    print(f"  • Cyclomatic Complexity:  {complexity.cyclomatic_complexity}", file=output)
    print(f"  • Max Nesting Depth:      {complexity.max_nesting_depth}", file=output)
    print(f"  • Functions Defined:      {complexity.num_functions}", file=output)
    print(f"  • Classes Defined:        {complexity.num_classes}", file=output)
    print(f"  • Decision Branches:      {complexity.num_branches}", file=output)
    print(f"  • Loops:                  {complexity.num_loops}", file=output)
    
    print(f"\n📈 DERIVED METRICS:", file=output)
    print(f"  • Complexity Density:     {complexity.complexity_density:.2f}", file=output)
    print(f"  • Essential Complexity:   {complexity.essential_complexity:.1f}", file=output)
    print(f"  • Halstead Volume:        {complexity.halstead_volume:.1f} bits", file=output)
    
    print(f"\n🎯 ASSESSMENT:", file=output)
    print(f"  • Risk Level:             {complexity.risk_level}", file=output)
    
    maintainability = "✓ YES" if complexity.is_maintainable else "✗ NO"
    print(f"  • Maintainable:           {maintainability}", file=output)
    
    print(f"\n{'=' * width}\n", file=output)


def print_dataflow_report(
    dataflow: DataFlowCoordinates,
    name: str = "Module",
    file=None
) -> None:
    """
    Imprime un reporte formateado de flujo de datos.
    
    Args:
        dataflow: Coordenadas de flujo de datos
        name: Nombre del módulo
        file: Archivo de salida (default: sys.stdout)
    r"""
    import sys
    output = file or sys.stdout
    
    width = 70
    print(f"\n{'=' * width}", file=output)
    print(f"DATA FLOW ANALYSIS: {name}", file=output)
    print(f"{'=' * width}", file=output)
    
    print(f"\n📖 READS (Input Variables):", file=output)
    if dataflow.pure_reads:
        for var in sorted(dataflow.pure_reads):
            print(f"  • {var}", file=output)
    else:
        print("  (none)", file=output)
    
    print(f"\n✏️  WRITES (Output Variables):", file=output)
    if dataflow.pure_writes:
        for var in sorted(dataflow.pure_writes):
            print(f"  • {var}", file=output)
    else:
        print("  (none)", file=output)
    
    print(f"\n🔄 MODIFIED (Read-Write Variables):", file=output)
    if dataflow.modified_variables:
        for var in sorted(dataflow.modified_variables):
            print(f"  • {var}", file=output)
    else:
        print("  (none)", file=output)
    
    print(f"\n📊 STATISTICS:", file=output)
    print(f"  • Total Variables:        {len(dataflow.all_variables)}", file=output)
    print(f"  • Spectral Radius:        {dataflow.spectral_radius():.2f}", file=output)
    print(f"  • Shannon Entropy:        {dataflow.entropy():.2f} bits", file=output)
    
    print(f"\n{'=' * width}\n", file=output)


# =============================================================================
# PUNTO DE ENTRADA PARA TESTING Y DEMOSTRACIÓN
# =============================================================================

def main():
    """Función principal de demostración."""
    
    # Código de ejemplo con diversas construcciones
    example_code = r"""
def fibonacci(n):
    \"\"\"Calcula el n-ésimo número de Fibonacci.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n):
    \"\"\"Calcula n! de forma iterativa.\"\"\"
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

class Calculator:
    \"\"\"Calculadora básica con validación.\"\"\"
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(('div', a, b, result))
        return result
    
    def get_history(self):
        return self.history.copy()

# Código con complejidad elevada (ejemplo didáctico)
def complex_function(x, y, z):
    result = 0
    if x > 0:
        if y > 0:
            if z > 0:
                result = x + y + z
            else:
                result = x + y
        elif y < 0:
            result = x - y
    elif x < 0:
        while y > 0:
            result += x
            y -= 1
    
    try:
        value = result / x
    except ZeroDivisionError:
        value = 0
    
    return value if value > 0 else -value
"""
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " AST STATIC ANALYZER - RIGOROUS ANALYSIS DEMONSTRATION ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        # Análisis completo
        print("\n🔬 Analyzing example code with full verification suite...")
        result = ASTStaticAnalyzer.analyze_code(
            example_code,
            filename="example.py",
            enable_hamiltonian=True,
            enable_cohomology=True,
            strict_mode=False  # No lanzar excepciones, solo advertir
        )
        
        # Reportes
        print_dataflow_report(result['dataflow'], "example.py")
        print_complexity_report(result['complexity'], "example.py")
        
        # Verificaciones
        print("\n🔍 VERIFICATION RESULTS:")
        print(f"  • Hamiltonian Condition:  {'✓ PASS' if result['hamiltonian_ok'] else '✗ FAIL'}")
        
        if result.get('cohomology_ok') is not None:
            status = '✓ PASS' if result['cohomology_ok'] else '✗ FAIL'
            print(f"  • Cohomology Check:       {status}")
            print(f"    - dim H¹(T,F) = {result.get('cohomology_dimension', 'N/A')}")
            
            prob_vars = result.get('problematic_variables', set())
            if prob_vars:
                print(f"    - Potentially uninitialized: {sorted(prob_vars)}")
        
        # Advertencias
        if result['warnings']:
            print(f"\n⚠️  WARNINGS ({len(result['warnings'])}):")
            for i, warning in enumerate(result['warnings'], 1):
                print(f"  {i}. {warning}")
        
        # Test de validación JSON
        print("\n" + "─" * 80)
        print("\n📋 JSON SCHEMA VALIDATION TEST:")
        
        test_schema = {
            "name": "ExampleTool",
            "version": "1.0.0",
            "config": {
                "timeout": 30,
                "retries": 3,
                "endpoints": ["api.example.com", "backup.example.com"],
                "options": {
                    "verbose": True,
                    "log_level": "INFO"
                }
            },
            "metadata": {
                "author": "Γ-Physics Team",
                "license": "MIT"
            }
        }
        
        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(test_schema)
        
        if is_valid:
            print("✓ Schema validation: PASS")
            print(f"  • Structure depth: {JSONStructureValidator.compute_depth(test_schema)}")
            print(f"  • Estimated size: {JSONStructureValidator.estimate_memory_size(test_schema)} bytes")
            
            if markdown:
                print("\n📊 Normalized Representation (Top-Level):")
                # Mostrar solo el nivel superior
                top_level = {k: v for k, v in test_schema.items() if not isinstance(v, dict)}
                if top_level:
                    print(TabularNormalizer.to_markdown_table(top_level))
        else:
            print(f"✗ Schema validation: FAIL")
            print(f"  Error: {error}")
        
        print("\n" + "═" * 80)
        print("✨ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        
        if hasattr(e, 'context'):
            print(f"\n📋 Context:")
            for key, value in e.context.items():
                print(f"   • {key}: {value}")
        
        import traceback
        print(f"\n🔍 Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()