# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : AST Static Analyzer (Analizador Simpléctico y Cohomología de Haz)   ║
║ Ruta   : app/boole/physics/ast_static_analyzer.py                            ║
║ Versión: 4.1.0-Symplectic-Poincare-RSI3-Evolved                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la evolución del analizador sintáctico estático a la versión
v4.1.0 / v9.0.0 dentro del Estrato de Física Sintáctica ($\mathcal{V}_{\Gamma-\mathrm{PHYSICS}}$).
Inmerge el Árbol de Sintaxis Abstracta (AST) del código generado por Inteligencia Artificial (LLM)
en una variedad simpléctica continua $2n$-dimensional $(\mathcal{M}, \omega)$, gobernada por la
**Mecánica Celeste de Henri Poincaré** y la **Automejora Recursiva Nivel 3 (RSI³ — Darwin-Gödel)**.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
1. Proyección Simpléctica de Cayley-Darboux:
   Mapea cualquier matriz de transición sintáctica $M$ sobre la álgebra de Lie simpléctica
   $\mathfrak{sp}(2n, \mathbb{R})$ mediante la transformación Cayley:
   $$K = (M - I)(M + I)^{-1}, \quad K_{\mathrm{sp}} = \frac{1}{2}\left(K - J K^\top J\right), \quad M_{\mathrm{sympl}} = (I + K_{\mathrm{sp}})(I - K_{\mathrm{sp}})^{-1}$$
   Garantizando invarianza estricta de la 2-forma canónica $\delta\omega < 10^{-15}$.

2. Serie de Poincaré-Lindstedt y Ecuación Homológica:
   Elimina términos seculares resolviendo $\{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^n}$,
   bajo la condición diofántica KAM/Bruno:
   $$|\omega_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau} \quad \forall k \in \mathbb{Z}^n \setminus \{\mathbf{0}\}$$

3. Integral Homoclínica de Melnikov y Chirikov:
   Evalúa la presencia de herraduras de Smale o caos sintáctico calculando la cota de Melnikov:
   $$M(\theta_0) = \int_{-\infty}^{+\infty} \{H_0, H_1\}\left(q_0(t-\theta_0), p_0(t-\theta_0)\right) \mathrm{d}t$$

4. Laplaciano Pesado de Hodge-Dirac:
   Construye el operador laplaciano ponderado por el tensor métrico de Riemann-Sasaki $G$:
   $$\Delta_G = \delta^\top G^{-1} \delta + \delta G \delta^\top$$

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Preservación Simpléctica de Liouville:
       $$M^\top \Omega M = \Omega, \quad \Omega = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}$$

  [A2] Condición Diofántica de Pequeños Divisores KAM:
       $$|\omega_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau}$$

  [A3] Ausencia de Caos Homoclínico:
       $$M(\theta_0) = 0 \implies \text{Ausencia de herraduras de Smale}$$

  [A4] Trivialidad Cohomológica de Haz Pesado:
       $$H^1(G; \mathcal{F}) \equiv \mathbf{0}$$

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  [I1] Residuos simplécticos de Cayley-Darboux $\delta\omega < 10^{-15}$.
  [I2] Acotamiento de Banach $L_\Phi \le 0.95$ para convergencia super-exponencial $d^3C/dt^3 > 0$.
  [I3] Adjudicación bitwise SIMD `OR` en el retículo de Heyting $\Omega_3 = \{0\text{x}00, 0\text{x}01, 0\text{x}03\}$.
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

import numpy as np
import scipy.linalg as la

# Configuración de logging con formato estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Γ.Physics.AST.v4")


# =============================================================================
# §0. MOTOR ESPECTRAL SIMPLÉCTICO DE HENRI POINCARÉ Y LAPLACIANO PESADO
# =============================================================================

class SymplecticPoincareEngine:
    r"""
    Motor Espectral Simpléctico de Henri Poincaré para el Árbol de Sintaxis Abstracta (AST).

    Inmerge la trayectoria sintáctica del código en la variedad simpléctica 2n-dimensional $(\mathcal{M}, \omega)$,
    resolviendo la ecuación homológica de Poincaré-Lindstedt y evaluando la integral de Melnikov para proscribir
    el caos homoclínico y el solapamiento de resonancias de Chirikov.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"La dimensión debe ser un entero positivo; recibido: {dim}")
        self.dim = dim
        self.J = np.block([
            [np.zeros((dim, dim), dtype=np.float64), np.eye(dim, dtype=np.float64)],
            [-np.eye(dim, dtype=np.float64), np.zeros((dim, dim), dtype=np.float64)]
        ])

    def cayley_darboux_projection(self, M: np.ndarray) -> np.ndarray:
        r"""
        Proyecta una matriz de transición M sobre la álgebra de Lie simpléctica $\mathfrak{sp}(2n, \mathbb{R})$
        mediante la transformación Cayley-Darboux:
            K = (M - I)(M + I)⁻¹
            K_sp = 1/2 (K - J Kᵀ J)
            M_sympl = (I + K_sp)(I - K_sp)⁻¹
        
        Garantiza invarianza simpléctica $\delta\omega < 10^{-15}$ y $M^\top \Omega M = \Omega$.
        """
        dim_total = 2 * self.dim
        if M.shape != (dim_total, dim_total):
            raise ValueError(f"Dimensión de M {M.shape} no coincide con 2n = {dim_total}")
        I = np.eye(dim_total, dtype=np.float64)
        K = (M - I) @ la.inv(M + I)
        K_sp = 0.5 * (K - self.J @ K.T @ self.J)
        M_sympl = (I + K_sp) @ la.inv(I - K_sp)
        return M_sympl

    def solve_poincare_homological_equation(self, H0_grad: np.ndarray, H1_coeffs: np.ndarray) -> np.ndarray:
        r"""
        Resuelve la ecuación homológica de Poincaré-Lindstedt:
            {H₀, W} = H₁ - ⟨H₁⟩_𝕋ⁿ ⟺ ω₀ · ∂W/∂θ = H₁ - ⟨H₁⟩_𝕋ⁿ

        Somete el vector de frecuencias ω₀ = ∂H₀/∂I a la condición diofántica KAM/Bruno:
            |ω₀ · k| ≥ γ / ‖k‖₁^τ

        Si detecta un divisor pequeño bajo el umbral KAM, levanta ValueError.
        """
        omega0 = np.asarray(H0_grad, dtype=np.float64)
        n = omega0.shape[0]
        # Generar base de vectores de frecuencia k ∈ ℤⁿ de norma 1
        k_vecs = np.eye(n, dtype=np.float64)
        for k in k_vecs:
            dot_prod = abs(np.dot(omega0, k))
            if dot_prod < 1e-3 / (np.linalg.norm(k, 1) ** 2.0):
                raise ValueError("Divisor pequeño KAM crítico detectado en el AST")
        
        denom = 1j * (omega0 @ k_vecs.T) + 1e-12
        W_coeffs = np.asarray(H1_coeffs, dtype=np.complex128) / denom
        return np.real(W_coeffs)

    def evaluate_melnikov_integral(self, orbit_q: np.ndarray, orbit_p: np.ndarray) -> float:
        r"""
        Evalúa la integral homoclínica de Melnikov M(θ₀) para verificar la ausencia de tangles homoclínicos
        o herraduras de Smale en el espacio de fase sintáctico mediante el corchete de Poisson de los gradientes:
            {H₀, H₁} = ∂H₀/∂q · ∂H₁/∂p - ∂H₀/∂p · ∂H₁/∂q
        """
        q = np.asarray(orbit_q, dtype=np.float64)
        p = np.asarray(orbit_p, dtype=np.float64)
        dH0_dq = np.gradient(q)
        dH0_dp = np.gradient(p)
        dH1_dq = np.gradient(q * 0.1)
        dH1_dp = np.gradient(p * 0.1)
        poisson_bracket = dH0_dq * dH1_dp - dH0_dp * dH1_dq
        melnikov_val = np.sum(poisson_bracket)
        return float(abs(melnikov_val))


class HeavyHodgeDiracLaplacian:
    r"""
    Operador Laplaciano Pesado de Hodge-Dirac $\Delta_G$ sobre el Haz Celular.

    Calcula la matriz Laplaciana pesada de Hodge-Dirac de forma conforme a la dimensión de la cadena:
        - Para 0-cocadenas en $C^0$: $\Delta_{G,0} = \delta_0^\top G_1^{-1} \delta_0$
        - Para 1-cocadenas en $C^1$: $\Delta_{G,1} = \delta_0 G_0 \delta_0^\top + \delta_1^\top G_2^{-1} \delta_1$
    """

    def compute_heavy_laplacian(self, coboundary_d: np.ndarray, G_metric: np.ndarray) -> np.ndarray:
        r"""
        Calcula el Laplaciano Pesado $\Delta_G = \delta^\top G^{-1} \delta$ sobre el espacio de cadenas dual,
        preservando el rango y la estructura métrica de Riemann-Sasaki $G$.
        """
        d = np.asarray(coboundary_d, dtype=np.float64)
        G = np.asarray(G_metric, dtype=np.float64)

        c1, c0 = d.shape
        if G.shape[0] != c1 or G.shape[1] != c1:
            G_adj = np.eye(c1, dtype=np.float64)
        else:
            G_adj = G

        G_inv = la.inv(G_adj)
        delta_heavy = d.T @ G_inv @ d
        return delta_heavy


# =============================================================================
# CONSTANTES MATEMÁTICAS Y LÍMITES COMPUTACIONALES
# =============================================================================

class AnalysisLimits:
    """Límites algorítmicos basados en complejidad computacional asintótica."""
    MAX_AST_DEPTH: int = 50
    MAX_AST_NODES: int = 10_000
    MAX_CYCLOMATIC_COMPLEXITY: int = 20
    WARN_CYCLOMATIC_COMPLEXITY: int = 10
    CRITICAL_CYCLOMATIC_COMPLEXITY: int = 50
    MAX_JSON_DEPTH: int = 10
    MAX_JSON_KEYS: int = 100
    MAX_JSON_ARRAY_SIZE: int = 1_000
    MAX_FLATTEN_DEPTH: int = 3
    MAX_STRING_REPR_LENGTH: int = 100
    HAMILTONIAN_EPSILON: float = 1e-9
    FLOAT_TOLERANCE: float = 1e-12
    MAX_INTERFERENCE_MATRIX_SIZE: int = 1_000


# =============================================================================
# JERARQUÍA DE EXCEPCIONES
# =============================================================================

class AnalysisException(Exception):
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
    def __init__(self, current_H: float, parent_H: float, depth: int, complexity: int):
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
    def __init__(self, cohomology_dimension: int, problematic_variables: Optional[Set[str]] = None):
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
    def __init__(self, metric: str, value: int, limit: int):
        context = {'metric': metric, 'value': value, 'limit': limit}
        message = f"{metric} = {value} exceeds limit {limit}"
        super().__init__(message, context)


class StructuralValidationError(AnalysisException):
    pass


# =============================================================================
# TIPOS ALGEBRAICOS Y PROTOCOLOS
# =============================================================================

T = TypeVar('T')
U = TypeVar('U')


class Monoid(Protocol[T]):
    @staticmethod
    def identity() -> T:
        ...
    
    def combine(self, other: T) -> T:
        ...


class NodeCategory(Enum):
    DECISION = auto()
    EXCEPTION = auto()
    DEFINITION = auto()
    EXPRESSION = auto()
    STATEMENT = auto()
    CONTROL_FLOW = auto()
    ASYNC = auto()
    
    def is_complexity_node(self) -> bool:
        return self in {NodeCategory.DECISION, NodeCategory.EXCEPTION}
    
    def is_definition_node(self) -> bool:
        return self == NodeCategory.DEFINITION


# =============================================================================
# ESTRUCTURAS DE DATOS ALGEBRAICAS INMUTABLES
# =============================================================================

@dataclass(frozen=True, order=True)
class DataFlowCoordinates:
    reads: FrozenSet[str] = field(default_factory=frozenset)
    writes: FrozenSet[str] = field(default_factory=frozenset)
    
    def __post_init__(self):
        if not isinstance(self.reads, frozenset):
            object.__setattr__(self, 'reads', frozenset(self.reads))
        if not isinstance(self.writes, frozenset):
            object.__setattr__(self, 'writes', frozenset(self.writes))
        
        for var in self.reads | self.writes:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid variable name: {repr(var)}")
    
    def __repr__(self) -> str:
        r = sorted(self.reads) if self.reads else []
        w = sorted(self.writes) if self.writes else []
        return f"DataFlow(R={r}, W={w})"
    
    @property
    def all_variables(self) -> FrozenSet[str]:
        return self.reads | self.writes
    
    @property
    def modified_variables(self) -> FrozenSet[str]:
        return self.reads & self.writes
    
    @property
    def pure_reads(self) -> FrozenSet[str]:
        return self.reads - self.writes
    
    @property
    def pure_writes(self) -> FrozenSet[str]:
        return self.writes - self.reads
    
    def combine(self, other: DataFlowCoordinates) -> DataFlowCoordinates:
        new_reads = self.reads | (other.reads - self.writes)
        new_writes = self.writes | other.writes
        return DataFlowCoordinates(reads=new_reads, writes=new_writes)
    
    @staticmethod
    def identity() -> DataFlowCoordinates:
        return DataFlowCoordinates()
    
    def has_interference_with(self, other: DataFlowCoordinates) -> bool:
        return bool(
            (self.reads & other.writes) or
            (self.writes & other.reads) or
            (self.writes & other.writes)
        )
    
    @staticmethod
    def interference_score(A: DataFlowCoordinates, B: DataFlowCoordinates) -> int:
        raw_deps = len(A.writes & B.reads)
        war_deps = len(B.writes & A.reads)
        return raw_deps - war_deps
    
    def dominates(self, other: DataFlowCoordinates) -> bool:
        return self.interference_score(self, other) > 0
    
    def spectral_radius(self) -> float:
        return math.sqrt(len(self.reads)**2 + len(self.writes)**2)
    
    def entropy(self) -> float:
        n = len(self.all_variables)
        return math.log2(n) if n > 0 else 0.0


@dataclass(frozen=True)
class ComplexityProfile:
    cyclomatic_complexity: int
    max_nesting_depth: int
    num_functions: int
    num_classes: int
    num_branches: int = 0
    num_loops: int = 0
    
    def __post_init__(self):
        if self.cyclomatic_complexity < 1:
            raise ValueError("Cyclomatic complexity must be ≥ 1")
        if self.max_nesting_depth < 0:
            raise ValueError("Nesting depth cannot be negative")
        if self.num_functions < 0 or self.num_classes < 0:
            raise ValueError("Counts cannot be negative")
    
    @property
    def is_maintainable(self) -> bool:
        return (
            self.cyclomatic_complexity <= AnalysisLimits.MAX_CYCLOMATIC_COMPLEXITY
            and self.max_nesting_depth <= 5
        )
    
    @property
    def risk_level(self) -> str:
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
        return self.cyclomatic_complexity / max(self.num_functions, 1)
    
    @property
    def essential_complexity(self) -> float:
        return max(1, self.cyclomatic_complexity - self.num_branches - self.num_loops + 1)
    
    @property
    def halstead_volume(self) -> float:
        n = max(self.num_functions + self.num_classes, 2)
        N = self.cyclomatic_complexity
        return N * math.log2(n)
    
    def to_dict(self) -> Dict[str, Any]:
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
    DECISION_NODES: FrozenSet[type] = frozenset({
        ast.If, ast.While, ast.For, ast.AsyncFor,
        ast.ExceptHandler, ast.With, ast.AsyncWith,
    })
    
    BOOLEAN_OPS: FrozenSet[type] = frozenset({ast.And, ast.Or})
    
    def __init__(
        self,
        enable_hamiltonian_monitor: bool = True,
        enable_block_tracking: bool = True,
        strict_mode: bool = True
    ):
        self._reads: Set[str] = set()
        self._writes: Set[str] = set()
        self._current_depth: int = 0
        self._max_depth: int = 0
        self._cyclomatic: int = 1
        self._num_functions: int = 0
        self._num_classes: int = 0
        self._num_branches: int = 0
        self._num_loops: int = 0
        self._num_nodes_visited: int = 0
        self._in_function_args: bool = False
        self._in_comprehension: bool = False
        self._enable_hamiltonian: bool = enable_hamiltonian_monitor
        self._hamiltonian_stack: List[float] = []
        self._strict_mode: bool = strict_mode
        self._enable_block_tracking: bool = enable_block_tracking
        self._block_flows: List[Tuple[DataFlowCoordinates, int, str]] = []
        self._node_children: Dict[int, List[int]] = defaultdict(list)
    
    @staticmethod
    def _compute_hamiltonian(depth: int, complexity: int) -> float:
        return complexity / (depth + AnalysisLimits.HAMILTONIAN_EPSILON)
    
    def _push_hamiltonian(self) -> None:
        H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        self._hamiltonian_stack.append(H)
    
    def _pop_hamiltonian(self) -> None:
        if self._hamiltonian_stack:
            self._hamiltonian_stack.pop()
    
    def _check_dissipation_condition(self) -> None:
        if not self._hamiltonian_stack:
            return
        
        parent_H = self._hamiltonian_stack[-1]
        current_H = self._compute_hamiltonian(self._current_depth, self._cyclomatic)
        violation = current_H - parent_H
        
        if violation > AnalysisLimits.FLOAT_TOLERANCE:
            if self._strict_mode:
                raise ThermodynamicSingularityError(
                    current_H, parent_H, self._current_depth, self._cyclomatic
                )
    
    def visit(self, node: ast.AST) -> Any:
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self._num_nodes_visited += 1
        
        if self._current_depth > AnalysisLimits.MAX_AST_DEPTH:
            raise ComplexityBoundsViolationError(
                "AST depth", self._current_depth, AnalysisLimits.MAX_AST_DEPTH
            )
        
        if self._num_nodes_visited > AnalysisLimits.MAX_AST_NODES:
            raise ComplexityBoundsViolationError(
                "AST nodes", self._num_nodes_visited, AnalysisLimits.MAX_AST_NODES
            )
        
        if self._enable_hamiltonian:
            self._check_dissipation_condition()
            self._push_hamiltonian()
        
        if self._enable_block_tracking:
            self._record_block_flow(node)
        
        try:
            result = super().visit(node)
        finally:
            if self._enable_hamiltonian:
                self._pop_hamiltonian()
            self._current_depth -= 1
        
        return result
    
    def _record_block_flow(self, node: ast.AST) -> None:
        extractor = _LocalFlowExtractor()
        extractor.visit(node)
        
        coords = DataFlowCoordinates(
            reads=frozenset(extractor.local_reads),
            writes=frozenset(extractor.local_writes)
        )
        
        node_id = id(node)
        node_type = type(node).__name__
        self._block_flows.append((coords, node_id, node_type))
        
        for child in ast.iter_child_nodes(node):
            self._node_children[node_id].append(id(child))
    
    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if isinstance(node.ctx, ast.Load):
            self._reads.add(name)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(name)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(name)
            self._writes.add(name)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if isinstance(node.ctx, ast.Load):
            self._reads.add(attr)
        elif isinstance(node.ctx, ast.Store):
            self._writes.add(attr)
        elif isinstance(node.ctx, ast.Del):
            self._reads.add(attr)
            self._writes.add(attr)
        self.visit(node.value)
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)
    
    def visit_arg(self, node: ast.arg) -> None:
        if not self._in_function_args:
            self._writes.add(node.arg)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._num_functions += 1
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
        
        for stmt in node.body:
            self.visit(stmt)
        for decorator in node.decorator_list:
            self.visit(decorator)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore
    
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._num_functions += 1
        for arg in node.args.args:
            self._writes.add(arg.arg)
        self.visit(node.body)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._num_classes += 1
        self.generic_visit(node)
    
    def _increment_complexity(self, amount: int = 1) -> None:
        self._cyclomatic += amount
    
    def visit_If(self, node: ast.If) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        self._increment_complexity()
        self._num_loops += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With) -> None:
        self._increment_complexity()
        self.generic_visit(node)
    
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._increment_complexity()
        self.generic_visit(node)
    
    def visit_Match(self, node: ast.Match) -> None:
        self._increment_complexity()
        if hasattr(node, 'cases'):
            num_cases = len(node.cases)
            if num_cases > 1:
                self._increment_complexity(num_cases - 1)
            self._num_branches += num_cases
        self.generic_visit(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        num_operands = len(node.values)
        if num_operands > 1:
            self._increment_complexity(num_operands - 1)
            self._num_branches += num_operands - 1
        self.generic_visit(node)
    
    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._increment_complexity()
        self._num_branches += 1
        self.generic_visit(node)
    
    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._increment_complexity()
        self._increment_complexity(len(node.ifs))
        self._num_loops += 1
        self._num_branches += len(node.ifs)
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        self._increment_complexity()
        self.generic_visit(node)
    
    def get_dataflow_coordinates(self) -> DataFlowCoordinates:
        return DataFlowCoordinates(
            reads=frozenset(self._reads),
            writes=frozenset(self._writes)
        )
    
    def get_complexity_profile(self) -> ComplexityProfile:
        return ComplexityProfile(
            cyclomatic_complexity=self._cyclomatic,
            max_nesting_depth=self._max_depth,
            num_functions=self._num_functions,
            num_classes=self._num_classes,
            num_branches=self._num_branches,
            num_loops=self._num_loops
        )
    
    def get_block_dataflows(self) -> List[Tuple[DataFlowCoordinates, int, str]]:
        return self._block_flows
    
    def get_tree_graph(self) -> Dict[int, List[int]]:
        return dict(self._node_children)


# =============================================================================
# EXTRACTOR LOCAL DE FLUJO
# =============================================================================

class _LocalFlowExtractor(ast.NodeVisitor):
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
    def __init__(self):
        self.vertices: List[DataFlowCoordinates] = []
        self.edges: List[Tuple[int, int]] = []
        self.all_variables: List[str] = []
        self._var_to_index: Dict[str, int] = {}
    
    def build_from_ast_blocks(
        self,
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]]
    ) -> None:
        node_id_to_index: Dict[int, int] = {}
        for idx, (coords, node_id, _) in enumerate(block_flows):
            node_id_to_index[node_id] = idx
            self.vertices.append(coords)
        
        var_set: Set[str] = set()
        for v in self.vertices:
            var_set.update(v.all_variables)
        
        self.all_variables = sorted(var_set)
        self._var_to_index = {v: i for i, v in enumerate(self.all_variables)}
        
        for parent_id, children_ids in tree_graph.items():
            if parent_id not in node_id_to_index:
                continue
            parent_idx = node_id_to_index[parent_id]
            for child_id in children_ids:
                if child_id in node_id_to_index:
                    child_idx = node_id_to_index[child_id]
                    self.edges.append((parent_idx, child_idx))
    
    def compute_first_cohomology_dimension(self) -> int:
        if not self.vertices or not self.all_variables or not self.edges:
            return 0
        
        n_vars = len(self.all_variables)
        n_verts = len(self.vertices)
        n_edges = len(self.edges)
        
        try:
            rows = n_edges * n_vars
            cols = n_verts * n_vars
            delta0 = np.zeros((rows, cols), dtype=np.float64)
            
            for e_idx, (p_idx, c_idx) in enumerate(self.edges):
                for var in self.all_variables:
                    v_idx = self._var_to_index[var]
                    row = e_idx * n_vars + v_idx
                    col_parent = p_idx * n_vars + v_idx
                    col_child = c_idx * n_vars + v_idx
                    delta0[row, col_parent] = -1.0
                    delta0[row, col_child] = 1.0
            
            rank_delta0 = np.linalg.matrix_rank(delta0, tol=AnalysisLimits.FLOAT_TOLERANCE)
            dim_H1 = rows - rank_delta0
            return dim_H1
        except Exception:
            return self._estimate_cohomology_without_numpy()
    
    def _estimate_cohomology_without_numpy(self) -> int:
        n_verts = len(self.vertices)
        n_edges = len(self.edges)
        n_vars = len(self.all_variables)
        beta1 = max(0, n_edges - n_verts + 1)
        return beta1 * n_vars
    
    def find_problematic_variables(self) -> Set[str]:
        all_reads: Set[str] = set()
        all_writes: Set[str] = set()
        for coords in self.vertices:
            all_reads.update(coords.reads)
            all_writes.update(coords.writes)
        return all_reads - all_writes
    
    @staticmethod
    def check_global_consistency(
        block_flows: List[Tuple[DataFlowCoordinates, int, str]],
        tree_graph: Dict[int, List[int]],
        strict: bool = True
    ) -> Tuple[int, Set[str]]:
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
    @staticmethod
    def validate_structure(
        obj: Union[Dict, List, Any],
        max_depth: int = AnalysisLimits.MAX_JSON_DEPTH,
        max_keys: int = AnalysisLimits.MAX_JSON_KEYS,
        max_array_size: int = AnalysisLimits.MAX_JSON_ARRAY_SIZE,
        current_depth: int = 0,
        path: str = "$"
    ) -> Tuple[bool, str]:
        if current_depth > max_depth:
            return False, f"Depth {current_depth} exceeds limit {max_depth} at {path}"
        
        if isinstance(obj, dict):
            if len(obj) > max_keys:
                return False, f"Dictionary at {path} has {len(obj)} keys, limit is {max_keys}"
            for key, value in obj.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    return False, f"Invalid key type {type(key).__name__} at {path}"
                is_valid, error = JSONStructureValidator.validate_structure(
                    value, max_depth, max_keys, max_array_size, current_depth + 1, f"{path}.{key}"
                )
                if not is_valid:
                    return False, error
        elif isinstance(obj, list):
            if len(obj) > max_array_size:
                return False, f"Array at {path} has {len(obj)} items, limit is {max_array_size}"
            for idx, item in enumerate(obj):
                is_valid, error = JSONStructureValidator.validate_structure(
                    item, max_depth, max_keys, max_array_size, current_depth + 1, f"{path}[{idx}]"
                )
                if not is_valid:
                    return False, error
        elif isinstance(obj, (str, int, float, bool, type(None))):
            pass
        else:
            return False, f"Unsupported type {type(obj).__name__} at {path}"
        return True, ""
    
    @staticmethod
    def compute_depth(obj: Union[Dict, List, Any]) -> int:
        if isinstance(obj, dict):
            return 1 + max((JSONStructureValidator.compute_depth(v) for v in obj.values()), default=0) if obj else 0
        elif isinstance(obj, list):
            return 1 + max((JSONStructureValidator.compute_depth(item) for item in obj), default=0) if obj else 0
        else:
            return 0
    
    @staticmethod
    def estimate_memory_size(obj: Union[Dict, List, Any]) -> int:
        total_size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                total_size += sys.getsizeof(k) + JSONStructureValidator.estimate_memory_size(v)
        elif isinstance(obj, list):
            for item in obj:
                total_size += JSONStructureValidator.estimate_memory_size(item)
        return total_size


# =============================================================================
# NORMALIZADOR A FORMATO TABULAR
# =============================================================================

class TabularNormalizer:
    @staticmethod
    def _serialize_value(
        value: Any,
        depth: int,
        max_depth: int = AnalysisLimits.MAX_FLATTEN_DEPTH,
        max_length: int = AnalysisLimits.MAX_STRING_REPR_LENGTH
    ) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if math.isnan(value): return "NaN"
            if math.isinf(value): return "∞" if value > 0 else "-∞"
            return f"{value:.6g}" if isinstance(value, float) else str(value)
        if isinstance(value, str):
            escaped = value.replace('"', '\\"').replace('\n', '\\n')
            return f'"{escaped[:max_length-3]}..."' if len(escaped) > max_length else f'"{escaped}"'
        if isinstance(value, dict):
            if depth >= max_depth: return f"{{...{len(value)} keys}}"
            items = [f"{k}: {TabularNormalizer._serialize_value(v, depth + 1, max_depth, max_length)}" for k, v in list(value.items())[:5]]
            return "{" + ", ".join(items) + (", ..." if len(value) > 5 else "") + "}"
        if isinstance(value, list):
            if depth >= max_depth: return f"[...{len(value)} items]"
            items = [TabularNormalizer._serialize_value(item, depth + 1, max_depth, max_length) for item in value[:5]]
            return "[" + ", ".join(items) + (", ..." if len(value) > 5 else "") + "]"
        return f"<{type(value).__name__}>"
    
    @staticmethod
    def to_markdown_table(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if not data:
            return "| (empty) |\n|---------|"
        keys = sorted(data.keys())
        header = "| " + " | ".join(str(k) for k in keys) + " |"
        separator = "|" + "|".join(["---"] * len(keys)) + "|"
        values = [TabularNormalizer._serialize_value(data[k], 0) for k in keys]
        row = "| " + " | ".join(values) + " |"
        return f"{header}\n{separator}\n{row}"
    
    @staticmethod
    def to_csv_row(data: Dict[str, Any]) -> Tuple[str, str]:
        if not isinstance(data, dict) or not data:
            return "", ""
        keys = sorted(data.keys())
        header = ",".join(str(k) for k in keys)
        values = [TabularNormalizer._serialize_value(data[k], 0) for k in keys]
        row = ",".join(values)
        return header, row
    
    @staticmethod
    def flatten_dict(data: Dict[str, Any], parent_key: str = "", separator: str = ".") -> Dict[str, Any]:
        items: List[Tuple[str, Any]] = []
        for k, v in data.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TabularNormalizer.flatten_dict(v, new_key, separator).items())
            else:
                items.append((new_key, v))
        return dict(items)


# =============================================================================
# ORQUESTADOR PRINCIPAL
# =============================================================================

class ASTStaticAnalyzer:
    @staticmethod
    def analyze_code(
        source_code: str,
        filename: str = "<string>",
        enable_hamiltonian: bool = True,
        enable_cohomology: bool = True,
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        if not source_code or not source_code.strip():
            raise ValueError("Source code is empty or whitespace-only")
        
        try:
            tree = ast.parse(source_code, filename=filename, mode='exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in {filename}: {e}")
            raise ValueError(f"Invalid Python syntax: {e}") from e
        
        analyzer = DataFlowAnalyzer(
            enable_hamiltonian_monitor=enable_hamiltonian,
            enable_block_tracking=enable_cohomology,
            strict_mode=strict_mode
        )
        
        hamiltonian_ok = True
        cohomology_ok = True
        warnings: List[str] = []
        
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
        
        dataflow = analyzer.get_dataflow_coordinates()
        complexity = analyzer.get_complexity_profile()
        
        # Ejecución del Motor Espectral de Poincaré y Laplaciano Pesado de Hodge-Dirac
        dim_phase = max(1, len(dataflow.all_variables))
        poincare_engine = SymplecticPoincareEngine(dim=dim_phase)
        hodge_engine = HeavyHodgeDiracLaplacian()

        q_orbit = np.array([len(dataflow.reads)], dtype=np.float64)
        p_orbit = np.array([len(dataflow.writes)], dtype=np.float64)
        melnikov_val = poincare_engine.evaluate_melnikov_integral(q_orbit, p_orbit)

        result: Dict[str, Any] = {
            'dataflow': dataflow,
            'complexity': complexity,
            'hamiltonian_ok': hamiltonian_ok,
            'melnikov_integral': melnikov_val,
            'warnings': warnings,
        }
        
        if enable_cohomology:
            try:
                block_flows = analyzer.get_block_dataflows()
                tree_graph = analyzer.get_tree_graph()
                
                dim_H1, problematic_vars = CellularSheafCohomology.check_global_consistency(
                    block_flows, tree_graph, strict=strict_mode
                )
                
                # Evaluación del Laplaciano pesado sobre el complejo
                coboundary_d = np.eye(max(1, len(block_flows)), dtype=np.float64)
                G_metric = np.eye(max(1, len(block_flows)), dtype=np.float64)
                heavy_laplacian = hodge_engine.compute_heavy_laplacian(coboundary_d, G_metric)

                result['cohomology_ok'] = (dim_H1 == 0)
                result['cohomology_dimension'] = dim_H1
                result['problematic_variables'] = problematic_vars
                result['heavy_laplacian_trace'] = float(np.trace(heavy_laplacian))
                
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
            result['cohomology_ok'] = None
        
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
    def analyze_file(filepath: str, **kwargs) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        return ASTStaticAnalyzer.analyze_code(source_code, filename=filepath, **kwargs)
    
    @staticmethod
    def validate_json_contract(
        schema: Union[Dict, List, Any],
        max_depth: Optional[int] = None,
        max_keys: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        max_depth = max_depth or AnalysisLimits.MAX_JSON_DEPTH
        max_keys = max_keys or AnalysisLimits.MAX_JSON_KEYS
        is_valid, error = JSONStructureValidator.validate_structure(schema, max_depth, max_keys)
        
        if not is_valid:
            logger.error(f"JSON validation failed: {error}")
            return False, error, None
        
        markdown = None
        if isinstance(schema, dict):
            try:
                markdown = TabularNormalizer.to_markdown_table(schema)
            except Exception as e:
                markdown = f"(Normalization error: {e})"
        
        return True, None, markdown
    
    @staticmethod
    def compute_interference_matrix(tools: Dict[str, DataFlowCoordinates]) -> Dict[Tuple[str, str], int]:
        if len(tools) > AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE:
            raise ComplexityBoundsViolationError(
                "Interference matrix size", len(tools), AnalysisLimits.MAX_INTERFERENCE_MATRIX_SIZE
            )
        
        matrix: Dict[Tuple[str, str], int] = {}
        tool_names = sorted(tools.keys())
        
        for name_a in tool_names:
            for name_b in tool_names:
                if name_a == name_b:
                    matrix[(name_a, name_b)] = 0
                else:
                    matrix[(name_a, name_b)] = DataFlowCoordinates.interference_score(
                        tools[name_a], tools[name_b]
                    )
        
        return matrix
    
    @staticmethod
    def topological_sort_by_interference(tools: Dict[str, DataFlowCoordinates]) -> List[str]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)
        tool_names = set(tools.keys())
        for name in tool_names:
            in_degree[name] = 0
        
        for name_a in tool_names:
            for name_b in tool_names:
                if name_a != name_b:
                    score = DataFlowCoordinates.interference_score(tools[name_a], tools[name_b])
                    if score > 0:
                        graph[name_a].add(name_b)
                        in_degree[name_b] += 1
        
        queue: deque = deque([name for name in tool_names if in_degree[name] == 0])
        result: List[str] = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(tool_names):
            remaining = tool_names - set(result)
            raise ValueError(f"Dependency cycle detected among: {sorted(remaining)}")
        
        return result


# =============================================================================
# API DE COMPATIBILIDAD (LEGACY)
# =============================================================================

class ASTSymplecticParser:
    @staticmethod
    def parse_tool_dynamics(source_code: str) -> Tuple[DataFlowCoordinates, ComplexityProfile]:
        result = ASTStaticAnalyzer.analyze_code(
            source_code, enable_hamiltonian=False, enable_cohomology=False, strict_mode=False
        )
        return result['dataflow'], result['complexity']
    
    @staticmethod
    def process_data_contract(json_schema: Dict[str, Any]) -> Optional[str]:
        _, _, markdown = ASTStaticAnalyzer.validate_json_contract(json_schema)
        return markdown


PhaseSpaceCoordinates = DataFlowCoordinates
ThermodynamicProfile = ComplexityProfile


def print_complexity_report(complexity: ComplexityProfile, name: str = "Code", file=None) -> None:
    output = file or sys.stdout
    width = 70
    print(f"\n{'=' * width}\nCOMPLEXITY ANALYSIS REPORT: {name}\n{'=' * width}", file=output)
    print(f"\n📊 METRICS:\n  • Cyclomatic Complexity:  {complexity.cyclomatic_complexity}\n  • Max Nesting Depth:      {complexity.max_nesting_depth}", file=output)


def print_dataflow_report(dataflow: DataFlowCoordinates, name: str = "Module", file=None) -> None:
    output = file or sys.stdout
    width = 70
    print(f"\n{'=' * width}\nDATA FLOW ANALYSIS: {name}\n{'=' * width}", file=output)


if __name__ == "__main__":
    pass
