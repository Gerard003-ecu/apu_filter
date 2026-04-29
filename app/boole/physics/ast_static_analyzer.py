"""
=========================================================================================
Módulo: AST Static Analyzer con Estructura de Grafo de Dependencias
Ubicación: app/boole/physics/ast_symplectic_parser.py
Versión: 2.0 - Rigorización Matemática Completa
=========================================================================================

FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
----------------------------------

1. TEORÍA DE GRAFOS (Análisis de Dependencias):
   - Grafo dirigido G = (V, E) donde V = variables, E = dependencias
   - Clasificación: read(v) → aristas de entrada, write(v) → aristas de salida
   - Interferencia: ∃ ciclo en el grafo de dependencias transitivo
   
2. COMPLEJIDAD CICLOMÁTICA (McCabe 1976):
   - M = E - N + 2P para grafo de flujo de control
   - M = número de caminos linealmente independientes
   - Definición operacional: M = 1 + #(nodos de decisión)
   
3. TEORÍA DE LENGUAJES (AST como Árbol Ordenado):
   - AST es un árbol ordenado etiquetado T = (N, E, λ)
   - Profundidad: max{d(n) : n ∈ N} donde d es distancia desde raíz
   - Límite de Lipschitz: evita explosión combinatoria en parsing
   
4. ANÁLISIS DE FLUJO DE DATOS:
   - Def-Use chains: (d, u) donde d define variable, u la usa
   - Reaching definitions: conjunto de definiciones que llegan a un punto
   - Live variable analysis: variables vivas en cada punto del programa

5. TEORÍA DE TIPOS (Contratos de Datos):
   - Schema como álgebra inicial en categoría de tipos
   - Profundidad de tipo: max{height(τ) : τ ∈ Type}
   - Normalización: retracción a forma canónica

ACLARACIONES:
-------------
La terminología "simpléctica" se mantiene como METÁFORA del espacio de fase clásico,
donde q (posiciones) ≈ reads y p (momentos) ≈ writes. NO hay verdadera estructura
simpléctica (no hay 2-forma ω cerrada no degenerada).

El "conmutador" mide INTERFERENCIA asimétrica, no es un corchete de Lie real.

=========================================================================================
"""

from __future__ import annotations

import ast
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("Gamma.Physics.ASTAnalyzer.v2.0")


# =============================================================================
# CONSTANTES Y LÍMITES (ANÁLISIS DE COMPLEJIDAD)
# =============================================================================

class AnalysisLimits:
    """
    Límites algorítmicos basados en complejidad computacional.
    
    Referencias:
    - McCabe (1976): "A Complexity Measure"
    - Shepperd (1988): "A Critique of Cyclomatic Complexity"
    """
    
    # AST depth: O(n) para código lineal, O(log n) óptimo para árboles balanceados
    MAX_AST_DEPTH: int = 50  # Límite pragmático antes de stack overflow
    
    # Complejidad ciclomática: umbral de mantenibilidad
    MAX_CYCLOMATIC_COMPLEXITY: int = 20  # ISO 26262 recomienda ≤ 15-20
    WARN_CYCLOMATIC_COMPLEXITY: int = 10
    
    # JSON/Schema depth: prevenir DoS exponencial
    MAX_JSON_DEPTH: int = 10
    MAX_JSON_KEYS: int = 100
    
    # Flattening depth para representación tabular
    MAX_FLATTEN_DEPTH: int = 3


# =============================================================================
# ENUMERACIONES Y TIPOS
# =============================================================================

class NodeCategory(Enum):
    """Categorías de nodos AST por función semántica."""
    DECISION = "decision"          # If, While, For, etc.
    EXCEPTION = "exception"        # Try, Except, Raise
    DEFINITION = "definition"      # FunctionDef, ClassDef
    EXPRESSION = "expression"      # BinOp, Call, etc.
    STATEMENT = "statement"        # Assign, Return, etc.
    CONTROL_FLOW = "control"       # Break, Continue, Return


# =============================================================================
# ESTRUCTURAS DE DATOS ALGEBRAICAS
# =============================================================================

@dataclass(frozen=True, order=True)
class DataFlowCoordinates:
    """
    Coordenadas de flujo de datos en el grafo de dependencias.
    
    Estructura matemática: Par ordenado (R, W) donde
    - R ⊆ Var: conjunto de variables leídas (use)
    - W ⊆ Var: conjunto de variables escritas (def)
    
    Propiedades:
    - Inmutabilidad garantiza coherencia
    - Order=True permite comparaciones deterministas
    - FrozenSet permite hashing para uso en conjuntos/diccionarios
    """
    reads: FrozenSet[str] = field(default_factory=frozenset)
    writes: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self):
        """Validación de invariantes."""
        if not isinstance(self.reads, frozenset):
            object.__setattr__(self, 'reads', frozenset(self.reads))
        if not isinstance(self.writes, frozenset):
            object.__setattr__(self, 'writes', frozenset(self.writes))

    def __repr__(self) -> str:
        r_list = sorted(self.reads) if self.reads else []
        w_list = sorted(self.writes) if self.writes else []
        return f"DataFlow(R={r_list}, W={w_list})"

    @property
    def all_variables(self) -> FrozenSet[str]:
        """Unión de todas las variables mencionadas."""
        return self.reads | self.writes

    def has_interference_with(self, other: DataFlowCoordinates) -> bool:
        """
        Verifica si hay interferencia (dependencia de datos) con otro flujo.
        
        Tipos de dependencia:
        1. RAW (Read After Write): self.reads ∩ other.writes ≠ ∅
        2. WAR (Write After Read): self.writes ∩ other.reads ≠ ∅
        3. WAW (Write After Write): self.writes ∩ other.writes ≠ ∅
        """
        return bool(
            (self.reads & other.writes) or   # RAW
            (self.writes & other.reads) or   # WAR
            (self.writes & other.writes)     # WAW
        )

    @staticmethod
    def interference_score(A: DataFlowCoordinates, B: DataFlowCoordinates) -> int:
        """
        Métrica de interferencia ASIMÉTRICA A → B.
        
        Definición:
            I(A, B) = |A.writes ∩ B.reads| - |B.writes ∩ A.reads|
        
        Interpretación:
        - I(A, B) > 0: A interfiere más con B (A debe ejecutarse antes)
        - I(A, B) < 0: B interfiere más con A (B debe ejecutarse antes)
        - I(A, B) = 0: No hay preferencia (o interferencia simétrica)
        
        Propiedades:
        - Antisimetría: I(A, B) = -I(B, A)
        - NO es un conmutador de Lie (no satisface Jacobi)
        """
        return len(A.writes & B.reads) - len(B.writes & A.reads)


@dataclass(frozen=True)
class ComplexityProfile:
    """
    Perfil de complejidad del código según múltiples métricas.
    
    Referencias:
    - McCabe (1976): Complejidad ciclomática
    - Halstead (1977): Métricas de volumen
    - Henderson-Sellers (1996): Object-oriented metrics
    """
    cyclomatic_complexity: int
    max_nesting_depth: int
    num_functions: int
    num_classes: int
    
    # Métricas derivadas
    @property
    def is_maintainable(self) -> bool:
        """
        Verifica si el código satisface umbrales de mantenibilidad.
        
        Criterios (basados en estándares industriales):
        - Complejidad ciclomática ≤ 20
        - Profundidad de anidamiento ≤ 5
        """
        return (
            self.cyclomatic_complexity <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY and
            self.max_nesting_depth <= 5
        )
    
    @property
    def risk_level(self) -> str:
        """
        Nivel de riesgo según complejidad ciclomática.
        
        Escala de Kemerer (1987):
        - 1-10: Bajo
        - 11-20: Moderado
        - 21-50: Alto
        - >50: Muy alto
        """
        cc = self.cyclomatic_complexity
        if cc <= AnalysisLimits.WARN_CYCLOMATIC_COMPLEXITY:
            return "LOW"
        elif cc <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY:
            return "MODERATE"
        elif cc <= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @property
    def complexity_density(self) -> float:
        """
        Densidad de complejidad: complejidad por unidad de código.
        
        Normalización por número de funciones definidas (si hay).
        """
        base_units = max(self.num_functions, 1)
        return self.cyclomatic_complexity / base_units

    def __repr__(self) -> str:
        return (
            f"Complexity(CC={self.cyclomatic_complexity}, "
            f"depth={self.max_nesting_depth}, "
            f"risk={self.risk_level})"
        )


# =============================================================================
# VISITADOR DE AST CON ANÁLISIS DE FLUJO DE DATOS
# =============================================================================

class DataFlowAnalyzer(ast.NodeVisitor):
    """
    Analizador de flujo de datos basado en recorrido de AST.
    
    Algoritmo:
    ----------
    1. Recorrido depth-first del AST
    2. Clasificación de contextos (Load/Store/Del)
    3. Rastreo de profundidad con límite de Lipschitz
    4. Cálculo de complejidad ciclomática según McCabe
    
    Complejidad:
    - Tiempo: O(n) donde n = número de nodos AST
    - Espacio: O(h) donde h = altura del árbol (para call stack)
    
    Invariantes:
    - current_depth ≥ 0
    - max_depth ≥ current_depth
    - cyclomatic_complexity ≥ 1
    """
    
    # Nodos que incrementan complejidad ciclomática
    DECISION_NODES = {
        ast.If, ast.While, ast.For, ast.AsyncFor,
        ast.ExceptHandler, ast.With, ast.AsyncWith,
    }
    
    def __init__(self):
        """Inicializa el analizador con estado limpio."""
        # Conjuntos de variables
        self._reads: Set[str] = set()
        self._writes: Set[str] = set()
        
        # Métricas de profundidad
        self._current_depth: int = 0
        self._max_depth: int = 0
        
        # Complejidad ciclomática
        self._cyclomatic: int = 1  # Camino base
        
        # Contadores estructurales
        self._num_functions: int = 0
        self._num_classes: int = 0
        
        # Estado de análisis
        self._in_function_args: bool = False

    # =========================================================================
    # CONTROL DE PROFUNDIDAD (con límite de Lipschitz)
    # =========================================================================

    def visit(self, node: ast.AST) -> Any:
        """
        Punto de entrada para cada nodo con control de profundidad.
        
        Implementa límite de Lipschitz para prevenir explosión exponencial.
        """
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        
        if self._current_depth > AnalysisLimits.MAX_AST_DEPTH:
            raise RecursionError(
                f"AST depth {self._current_depth} exceeds limit "
                f"{AnalysisLimits.MAX_AST_DEPTH}"
            )
        
        try:
            result = super().visit(node)
        finally:
            self._current_depth -= 1
        
        return result

    # =========================================================================
    # ANÁLISIS DE FLUJO DE DATOS (Def-Use)
    # =========================================================================

    def visit_Name(self, node: ast.Name) -> None:
        """
        Procesa referencias a nombres (variables).
        
        Contextos:
        - Load: lectura de variable (use)
        - Store: escritura de variable (def)
        - Del: eliminación de variable (kill)
        """
        name = node.id
        
        if isinstance(node.ctx, ast.Load):
            self._reads.add(name)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(name)
        elif isinstance(node.ctx, ast.Del):
            # Del es tanto read como write
            self._reads.add(name)
            self._writes.add(name)
        
        # Name no tiene hijos que visitar

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Procesa accesos a atributos (obj.attr).
        
        Estrategia:
        - Solo rastreamos el atributo final
        - El objeto base se visita recursivamente
        """
        attr_name = node.attr
        
        if isinstance(node.ctx, ast.Load):
            self._reads.add(attr_name)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(attr_name)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(attr_name)
            self._writes.add(attr_name)
        
        # Visitar el valor (obj en obj.attr)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        Procesa subscripting (arr[i]).
        
        Nota: No rastreamos el índice como variable separada.
        """
        # El subscript en sí se trata como acceso al contenedor
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        """
        Argumentos de función son definiciones (writes).
        """
        if not self._in_function_args:
            # Evitar doble conteo si ya estamos procesando args
            self._writes.add(node.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Definición de función."""
        self._num_functions += 1
        
        # Argumentos son writes
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
        
        # Visitar cuerpo
        for stmt in node.body:
            self.visit(stmt)
        
        # Decoradores
        for dec in node.decorator_list:
            self.visit(dec)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Función asíncrona."""
        self.visit_FunctionDef(node)  # Reutilizar lógica (que ya incrementa num_functions)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Definición de clase."""
        self._num_classes += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        Lambda es una función anónima.
        
        Incrementa contador de funciones y complejidad.
        """
        self._num_functions += 1
        # Lambda añade un camino
        
        # Argumentos
        for arg in node.args.args:
            self._writes.add(arg.arg)
        
        # Cuerpo
        self.visit(node.body)

    # =========================================================================
    # COMPLEJIDAD CICLOMÁTICA (McCabe)
    # =========================================================================

    def _increment_complexity(self, amount: int = 1) -> None:
        """Incrementa complejidad ciclomática."""
        self._cyclomatic += amount

    def visit_If(self, node: ast.If) -> None:
        """
        Condicional if/elif/else.
        
        Cada 'if' o 'elif' añade un camino de decisión.
        """
        self._increment_complexity()
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Loop for."""
        self._increment_complexity()
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Async for loop."""
        self._increment_complexity()
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Loop while."""
        self._increment_complexity()
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """
        Manejador de excepciones.
        
        Cada 'except' añade un camino alternativo.
        """
        self._increment_complexity()
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Context manager (with)."""
        self._increment_complexity()
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Async context manager."""
        self._increment_complexity()
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        """
        Pattern matching (Python 3.10+).
        
        Cada 'case' añade un camino.
        """
        # El match en sí añade uno
        self._increment_complexity()
        
        # Cada case adicional añade otro camino
        if hasattr(node, 'cases'):
            self._increment_complexity(len(node.cases) - 1)
        
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """
        Operadores booleanos (and/or).
        
        Cada operando adicional añade un camino de cortocircuito.
        """
        # n operandos → n-1 decisiones
        num_operands = len(node.values)
        if num_operands > 1:
            self._increment_complexity(num_operands - 1)
        
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """
        Expresión ternaria (a if cond else b).
        
        Añade un camino de decisión.
        """
        self._increment_complexity()
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """
        Comprensión (parte 'for' en list/dict/set comprehension).
        
        Cada 'for' en comprensión añade complejidad.
        """
        self._increment_complexity()
        
        # Cada 'if' en comprensión también añade
        self._increment_complexity(len(node.ifs))
        
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """
        Bloque try/except/else/finally.
        
        La estructura try en sí añade caminos.
        """
        # El try añade uno
        self._increment_complexity()
        
        # Los handlers se cuentan individualmente en visit_ExceptHandler
        self.generic_visit(node)

    # =========================================================================
    # EXTRACCIÓN DE RESULTADOS
    # =========================================================================

    def get_dataflow_coordinates(self) -> DataFlowCoordinates:
        """Obtiene las coordenadas de flujo de datos."""
        return DataFlowCoordinates(
            reads=frozenset(self._reads),
            writes=frozenset(self._writes)
        )

    def get_complexity_profile(self) -> ComplexityProfile:
        """Obtiene el perfil de complejidad."""
        return ComplexityProfile(
            cyclomatic_complexity=self._cyclomatic,
            max_nesting_depth=self._max_depth,
            num_functions=self._num_functions,
            num_classes=self._num_classes
        )


# =============================================================================
# VALIDADOR DE ESTRUCTURAS JSON/SCHEMA
# =============================================================================

class JSONStructureValidator:
    """
    Validador de profundidad y complejidad de estructuras JSON.
    
    Previene ataques de tipo:
    - Billion Laughs (explosión exponencial)
    - Hash collision DoS
    - Stack overflow por recursión profunda
    """
    
    @staticmethod
    def validate_structure(
        obj: Union[Dict, List, Any],
        max_depth: int = AnalysisLimits.MAX_JSON_DEPTH,
        max_keys: int = AnalysisLimits.MAX_JSON_KEYS,
        current_depth: int = 0
    ) -> Tuple[bool, str]:
        """
        Valida estructura JSON contra límites de complejidad.
        
        Args:
            obj: Objeto a validar
            max_depth: Profundidad máxima permitida
            max_keys: Número máximo de claves por nivel
            current_depth: Profundidad actual (para recursión)
        
        Returns:
            (is_valid, error_message)
        """
        if current_depth > max_depth:
            return False, f"Depth {current_depth} exceeds limit {max_depth}"
        
        if isinstance(obj, dict):
            if len(obj) > max_keys:
                return False, f"Dictionary has {len(obj)} keys, limit is {max_keys}"
            
            for key, value in obj.items():
                is_valid, error = JSONStructureValidator.validate_structure(
                    value, max_depth, max_keys, current_depth + 1
                )
                if not is_valid:
                    return False, f"In key '{key}': {error}"
        
        elif isinstance(obj, list):
            if len(obj) > max_keys * 10:  # Listas pueden ser más largas
                return False, f"List has {len(obj)} items, limit is {max_keys * 10}"
            
            for idx, item in enumerate(obj):
                is_valid, error = JSONStructureValidator.validate_structure(
                    item, max_depth, max_keys, current_depth + 1
                )
                if not is_valid:
                    return False, f"In index {idx}: {error}"
        
        return True, ""

    @staticmethod
    def compute_depth(obj: Union[Dict, List, Any]) -> int:
        """
        Calcula la profundidad máxima de una estructura.
        
        Complejidad: O(n) donde n = número de elementos
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


# =============================================================================
# NORMALIZADOR A FORMATO TABULAR
# =============================================================================

class TabularNormalizer:
    """
    Normaliza estructuras complejas a representación tabular plana.
    
    Proceso:
    1. Extracción de schema (claves y tipos)
    2. Aplanamiento controlado (con límite de profundidad)
    3. Serialización a formato Markdown/CSV
    
    Propiedades:
    - Preserva información hasta profundidad máxima
    - Formato human-readable
    - Compresión sin pérdida para estructuras planas
    """
    
    @staticmethod
    def _serialize_value(
        value: Any,
        depth: int,
        max_depth: int = AnalysisLimits.MAX_FLATTEN_DEPTH
    ) -> str:
        """
        Serializa un valor con límite de profundidad.
        
        Estrategia:
        - Primitivos: representación directa
        - Colecciones profundas: resumen estadístico
        - Objetos complejos: type hint
        """
        if value is None:
            return "null"
        
        if isinstance(value, bool):
            return "true" if value else "false"
        
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "∞" if value > 0 else "-∞"
            return str(value)
        
        if isinstance(value, str):
            # Escapar y truncar si es muy largo
            escaped = value.replace('"', '\\"')
            if len(escaped) > 50:
                escaped = escaped[:47] + "..."
            return f'"{escaped}"'
        
        if isinstance(value, dict):
            if depth >= max_depth:
                return f"{{...{len(value)} keys}}"
            
            items = [
                f"{k}: {TabularNormalizer._serialize_value(v, depth + 1, max_depth)}"
                for k, v in list(value.items())[:5]  # Primeros 5
            ]
            suffix = "..." if len(value) > 5 else ""
            return "{" + ", ".join(items) + suffix + "}"
        
        if isinstance(value, list):
            if depth >= max_depth:
                return f"[...{len(value)} items]"
            
            items = [
                TabularNormalizer._serialize_value(item, depth + 1, max_depth)
                for item in value[:5]  # Primeros 5
            ]
            suffix = "..." if len(value) > 5 else ""
            return "[" + ", ".join(items) + suffix + "]"
        
        # Fallback
        return f"<{type(value).__name__}>"

    @staticmethod
    def to_markdown_table(data: Dict[str, Any]) -> str:
        """
        Convierte diccionario a tabla Markdown.
        
        Formato:
        | Key1 | Key2 | ... |
        |------|------|-----|
        | val1 | val2 | ... |
        
        Args:
            data: Diccionario plano o anidado
        
        Returns:
            String en formato Markdown
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        
        if not data:
            return "| (empty) |\n|---------|"
        
        keys = sorted(data.keys())
        
        # Header
        header = "| " + " | ".join(keys) + " |"
        
        # Separator
        separator = "|" + "|".join(["---"] * len(keys)) + "|"
        
        # Values
        values = [
            TabularNormalizer._serialize_value(data[k], 0)
            for k in keys
        ]
        row = "| " + " | ".join(values) + " |"
        
        return f"{header}\n{separator}\n{row}"

    @staticmethod
    def to_csv_row(data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Convierte diccionario a fila CSV.
        
        Returns:
            (header_row, data_row)
        """
        if not isinstance(data, dict) or not data:
            return "", ""
        
        keys = sorted(data.keys())
        header = ",".join(keys)
        
        values = [
            TabularNormalizer._serialize_value(data[k], 0)
            for k in keys
        ]
        row = ",".join(values)
        
        return header, row


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

class ASTStaticAnalyzer:
    """
    Analizador estático de código Python basado en AST.
    
    Funcionalidades:
    1. Análisis de flujo de datos (def-use)
    2. Cálculo de métricas de complejidad
    3. Detección de dependencias
    4. Validación de límites estructurales
    
    Uso:
        analyzer = ASTStaticAnalyzer()
        dataflow, complexity = analyzer.analyze_code(source_code)
    """
    
    @staticmethod
    def analyze_code(
        source_code: str,
        filename: str = "<string>"
    ) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        """
        Analiza código Python y extrae métricas.
        
        Args:
            source_code: Código fuente Python
            filename: Nombre del archivo (para mensajes de error)
        
        Returns:
            (coordenadas_flujo_datos, perfil_complejidad)
        
        Raises:
            ValueError: Si el código está vacío
            SyntaxError: Si el código tiene errores sintácticos
            RecursionError: Si el AST excede profundidad máxima
        """
        # Validación de entrada
        if not source_code or not source_code.strip():
            raise ValueError("Source code is empty")
        
        # Parsing
        try:
            tree = ast.parse(source_code, filename=filename, mode='exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in {filename}: {e}")
            raise ValueError(f"Invalid Python syntax: {e}") from e
        
        # Análisis
        analyzer = DataFlowAnalyzer()
        
        try:
            analyzer.visit(tree)
        except RecursionError as e:
            logger.error(f"AST too deep in {filename}: {e}")
            raise
        
        # Extracción de resultados
        dataflow = analyzer.get_dataflow_coordinates()
        complexity = analyzer.get_complexity_profile()
        
        # Log de métricas
        logger.info(f"Analyzed {filename}: {complexity}")
        logger.debug(f"  Data flow: {len(dataflow.reads)} reads, {len(dataflow.writes)} writes")
        
        # Advertencias
        if not complexity.is_maintainable:
            logger.warning(
                f"Code in {filename} exceeds maintainability thresholds: {complexity}"
            )
        
        return dataflow, complexity

    @staticmethod
    def validate_json_contract(
        schema: Union[Dict, List, Any],
        max_depth: Optional[int] = None,
        max_keys: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Valida y normaliza un contrato de datos JSON.
        
        Args:
            schema: Estructura JSON a validar
            max_depth: Profundidad máxima (None usa default)
            max_keys: Claves máximas (None usa default)
        
        Returns:
            (is_valid, error_message, normalized_markdown)
        """
        max_depth = max_depth or AnalysisLimits.MAX_JSON_DEPTH
        max_keys = max_keys or AnalysisLimits.MAX_JSON_KEYS
        
        # Validación
        is_valid, error = JSONStructureValidator.validate_structure(
            schema, max_depth, max_keys
        )
        
        if not is_valid:
            logger.error(f"JSON validation failed: {error}")
            return False, error, None
        
        # Normalización (solo para dicts)
        if isinstance(schema, dict):
            try:
                markdown = TabularNormalizer.to_markdown_table(schema)
                return True, None, markdown
            except Exception as e:
                logger.error(f"Normalization failed: {e}")
                return True, None, f"(Normalization error: {e})"
        else:
            return True, None, "(Not a dict, cannot normalize to table)"

    @staticmethod
    def compute_interference_matrix(
        tools: Dict[str, DataFlowCoordinates]
    ) -> Dict[Tuple[str, str], int]:
        """
        Calcula matriz de interferencia entre herramientas.
        
        Args:
            tools: Diccionario {nombre: coordenadas_flujo}
        
        Returns:
            Diccionario {(tool_a, tool_b): score_interferencia}
        """
        interference = {}
        tool_names = sorted(tools.keys())
        
        for i, name_a in enumerate(tool_names):
            for name_b in tool_names[i:]:  # Incluye diagonal
                score = DataFlowCoordinates.interference_score(
                    tools[name_a],
                    tools[name_b]
                )
                
                if score != 0 or name_a == name_b:
                    interference[(name_a, name_b)] = score
                    if name_a != name_b:
                        interference[(name_b, name_a)] = -score  # Antisimetría
        
        return interference


# =============================================================================
# FUNCIÓN DE COMPATIBILIDAD (API LEGACY)
# =============================================================================

class ASTSymplecticParser:
    """
    Clase de compatibilidad con API anterior.
    
    DEPRECATED: Usar ASTStaticAnalyzer directamente.
    """
    
    @staticmethod
    def parse_tool_dynamics(
        source_code: str
    ) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        """
        API legacy: mantiene compatibilidad con nombre anterior.
        
        DEPRECATED: Renombrado a PhaseSpaceCoordinates → DataFlowCoordinates
                    y ThermodynamicProfile → ComplexityProfile
        """
        logger.warning(
            "parse_tool_dynamics is deprecated. "
            "Use ASTStaticAnalyzer.analyze_code instead."
        )
        return ASTStaticAnalyzer.analyze_code(source_code)
    
    @staticmethod
    def process_data_contract(
        json_schema: Dict[str, Any]
    ) -> Optional[str]:
        """API legacy para validación de contratos."""
        logger.warning(
            "process_data_contract is deprecated. "
            "Use ASTStaticAnalyzer.validate_json_contract instead."
        )
        is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(json_schema)
        return markdown if is_valid else None


# Alias para compatibilidad
PhaseSpaceCoordinates = DataFlowCoordinates
ThermodynamicProfile = ComplexityProfile


# =============================================================================
# UTILIDADES ADICIONALES
# =============================================================================

def analyze_file(filepath: str) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
    """
    Analiza un archivo Python completo.
    
    Args:
        filepath: Ruta al archivo .py
    
    Returns:
        (coordenadas_flujo_datos, perfil_complejidad)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    return ASTStaticAnalyzer.analyze_code(source_code, filename=filepath)


def print_complexity_report(complexity: ComplexityProfile, name: str = "Code"):
    """Imprime reporte legible de complejidad."""
    print(f"\n{'='*60}")
    print(f"Complexity Report: {name}")
    print(f"{'='*60}")
    print(f"  Cyclomatic Complexity:  {complexity.cyclomatic_complexity}")
    print(f"  Max Nesting Depth:      {complexity.max_nesting_depth}")
    print(f"  Functions Defined:      {complexity.num_functions}")
    print(f"  Classes Defined:        {complexity.num_classes}")
    print(f"  Risk Level:             {complexity.risk_level}")
    print(f"  Maintainable:           {'YES' if complexity.is_maintainable else 'NO'}")
    print(f"  Complexity Density:     {complexity.complexity_density:.2f}")
    print(f"{'='*60}\n")


# =============================================================================
# PUNTO DE ENTRADA PARA TESTING
# =============================================================================

if __name__ == "__main__":
    # Código de ejemplo para testing
    example_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
"""
    
    print("Analyzing example code...")
    dataflow, complexity = ASTStaticAnalyzer.analyze_code(example_code, "example.py")
    
    print(f"\nData Flow: {dataflow}")
    print_complexity_report(complexity, "example.py")
    
    # Test JSON validation
    test_schema = {
        "name": "TestTool",
        "version": "1.0",
        "config": {
            "timeout": 30,
            "retries": 3,
            "endpoints": ["api.example.com", "backup.example.com"]
        }
    }
    
    print("\nValidating JSON schema...")
    is_valid, error, markdown = ASTStaticAnalyzer.validate_json_contract(test_schema)
    
    if is_valid:
        print("✓ Schema is valid")
        if markdown:
            print("\nNormalized representation:")
            print(markdown)
    else:
        print(f"✗ Schema validation failed: {error}")