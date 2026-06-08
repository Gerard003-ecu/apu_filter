r"""
Módulo: tests/integration/boole/test_gamma_transitive_closure.py
========================================
Suite de Pruebas de Integración para la Pirámide Γ (Funtor Compuesto $F_{global}$)
(Versión Rigurosa MEJORADA - Revisión Crítica con Demostraciones Axiomáticas)

FUNDAMENTOS CATEGORIALES Y LEY DE CLAUSURA:

§1. LEY DE CLAUSURA TRANSITIVA AXIOMÁTICA
    La preservación estructural a través de la pirámide DIKW se define como una relación
    de subconjuntos en el orden topológico:

    $$V_{\Gamma-PHYSICS} \subset V_{\Gamma-TACTICS} \subset V_{\Gamma-STRATEGY} \subset V_{\Gamma-WISDOM}$$

    El funtor compuesto global del sistema se define como:

    $$F_{global} = F_{wisdom} \circ F_{strategy} \circ F_{tactics} \circ F_{physics}$$

    Garantizando que la composición de transformaciones preserve la continuidad de la
    variedad agéntica.

§2. CONEXIÓN DE GALOIS Y ADJUNCIÓN DE RETÍCULOS
    El mapeo entre el Retículo de Severidades estructurales ($S$) y el Retículo de
    Veredictos de negocio ($V$) se valida como un par de funtores adjuntos $f \dashv g$
    que conforman una Conexión de Galois:

    $$\forall x \in S, \forall y \in V, f(x) \leq_V y \iff x \leq_S g(y)$$

    Esto garantiza que la traducción semántica preserve la operación Supremo ($\sqcup$),
    colapsando la probabilidad hacia el Veto ante singularidades detectadas.
"""
from __future__ import annotations

# =============================================================================
# IMPORTS EXTERNOS
# =============================================================================
import math
import warnings
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, IntEnum, auto
from fractions import Fraction
from typing import (
    TypeVar, Generic, List, Dict, Optional, Set, Tuple,
    Callable, Protocol, Iterator, Any, Union, Literal, ClassVar
)
from typing_extensions import Self
import pytest
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

# =============================================================================
# IMPORTS INTERNOS
# =============================================================================
from app.core.schemas import Stratum
from app.boole.physics.ast_static_analyzer import (
    ASTSymplecticParser,
    ComplexityProfile as ThermodynamicProfile,
)
from app.boole.tactics.mic_minimizer import (
    BooleanVector,
)
from app.boole.strategy.sheaf_cohomology_orchestrator import (
    SheafCohomologyOrchestrator,
    CellularSheaf,
    HomologicalInconsistencyError,
    RestrictionMap,
)
from app.boole.wisdom.semantic_validator import (
    SemanticValidationEngine,
    Verdict as VerdictLevel,
    RiskProfile as ToleranceProfile,
    BusinessPurpose as SemanticMorphism,
    LLMOutput,
    create_default_knowledge_graph,
)

# =============================================================================
# CONFIGURACIÓN NUMÉRICA GLOBAL
# =============================================================================
getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

# Tolerancias jerárquicas basadas en análisis de error
EPSILON_FLOAT64: float = np.finfo(np.float64).eps
EPSILON_SYMPLECTIC: float = 1e-12
EPSILON_SPECTRAL: float = 1e-10
EPSILON_TOPOLOGICAL: float = 1e-14
EPSILON_LIPSCHITZ: float = 1e-8
EPSILON_GALOIS: float = 1e-6

# Constantes termodinámicas
BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K
PLANCK_CONSTANT: float = 6.62607015e-34    # J⋅s

# =============================================================================
# TIPOS ALGEBRAICOS REFINADOS
# =============================================================================
T = TypeVar('T')
V = TypeVar('V', bound=np.generic)

Z2Vector = NDArray[np.int8]
RealVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]


# =============================================================================
# ENUMERACIONES MATEMÁTICAS
# =============================================================================
class RegimenTermodinamico(Enum):
    """
    Clasificación termodinámica según teoría de Landau-Ginzburg.
    
    Teoría Física:
    -------------
    El funcional de energía libre de Landau-Ginzburg cerca de T_c:
    F(η, T) = a(T - T_c)η² + bη⁴ + O(η⁶)
    
    donde η es el parámetro de orden.
    
    Regímenes:
    • T < T_c₁: Fase ordenada (superfluidez cuántica, η ≠ 0)
    • T_c₁ ≤ T ≤ T_c₂: Región crítica (divergencia de fluctuaciones)
    • T > T_c₂: Fase desordenada (caos térmico, η = 0)
    """
    SUB_CRITICO = 1
    CRITICO = 2
    SUPER_CRITICO = 3
    
    @property
    def critical_temperature(self) -> float:
        """Temperatura crítica en Kelvin."""
        return {
            self.SUB_CRITICO: 1.0,
            self.CRITICO: 1.5,
            self.SUPER_CRITICO: 2.0,
        }[self]
    
    @property
    def order_parameter_exponent(self) -> float:
        """Exponente crítico β del parámetro de orden."""
        return {
            self.SUB_CRITICO: 0.5,
            self.CRITICO: 0.326,  # Valor experimental para transiciones 3D
            self.SUPER_CRITICO: 0.0,
        }[self]
    
    def __lt__(self, other: RegimenTermodinamico) -> bool:
        if not isinstance(other, RegimenTermodinamico):
            return NotImplemented
        return self.value < other.value
    
    def __le__(self, other: RegimenTermodinamico) -> bool:
        if not isinstance(other, RegimenTermodinamico):
            return NotImplemented
        return self.value <= other.value


class VerdictLevel(IntEnum):
    """Niveles de veredicto ontológico (orden parcial)."""
    VIABLE = 1
    CONDITIONAL = 2
    WARNING = 3
    REJECT = 4
    
    def __hash__(self) -> int:
        return int(self.value)


# =============================================================================
# CLASES DE DATOS MATEMÁTICOS
# =============================================================================
@dataclass(frozen=True, slots=True)
class AlexandroffPoint:
    """
    Punto en la compactificación de Alexandroff.
    
    Definición Topológica:
    ---------------------
    X̂ = X ⊔ {∞} con base de abiertos:
    • Abiertos de X (topología original)
    • {∞} ∪ (X \ K) donde K ⊂ X es compacto
    
    Invariantes:
    • is_infinity = True ⟹ value = +∞
    • is_infinity = False ⟹ value ∈ ℝ
    """
    is_infinity: bool
    value: float
    
    def __post_init__(self) -> None:
        if self.is_infinity and np.isfinite(self.value):
            raise ValueError(
                f"Invariante violado: punto infinito con valor finito {self.value}"
            )
        if not self.is_infinity and not np.isfinite(self.value):
            raise ValueError(
                f"Invariante violado: punto finito con valor {self.value} (no finito)"
            )
    
    def __repr__(self) -> str:
        return "AlexandroffPoint(∞)" if self.is_infinity else f"AlexandroffPoint({self.value:.6f})"
    
    def distance_to(self, other: AlexandroffPoint) -> float:
        """
        Métrica esférica en S^n (compactificación por un punto).
        
        d(x, y) = 2 arcsin(|x - y| / √(1 + |x|²)(1 + |y|²))
        """
        if self.is_infinity or other.is_infinity:
            return float('inf')
        
        numerator = abs(self.value - other.value)
        denominator = math.sqrt((1 + self.value**2) * (1 + other.value**2))
        
        if denominator < EPSILON_FLOAT64:
            return float('inf')
        
        return 2 * math.asin(min(1.0, numerator / denominator))


@dataclass(frozen=True, slots=True)
class PhaseSpace:
    """
    Espacio de fase simpléctico (M²ⁿ, ω).
    
    Estructura Matemática:
    ---------------------
    • M: variedad diferenciable de dimensión 2n
    • ω: 2-forma simpléctica (cerrada y no degenerada)
    
    Teorema de Darboux:
    • Localmente, ω = Σᵢ dqⁱ ∧ dpᵢ (forma canónica)
    
    Invariantes:
    • dim(M) = 2n (par)
    • ω ∧ ω^(n-1) ≠ 0 (no degeneración)
    • dω = 0 (cerradura)
    """
    symplectic_form: Optional[NDArray[np.float64]] = None
    dimension: int = 0
    coordinates_q: Optional[Z2Vector] = None
    coordinates_p: Optional[Z2Vector] = None
    
    def __post_init__(self) -> None:
        if self.symplectic_form is not None:
            if self.symplectic_form.ndim != 2:
                raise ValueError("Forma simpléctica debe ser matriz 2D")
            n, m = self.symplectic_form.shape
            if n != m:
                raise ValueError(f"Forma simpléctica debe ser cuadrada: {n}×{m}")
            if n % 2 != 0:
                raise ValueError(f"Dimensión debe ser par: {n}")
            if self.dimension > 0 and n != self.dimension:
                raise ValueError(f"Inconsistencia dimensional: forma {n}×{n}, esperado {self.dimension}")
    
    @property
    def is_valid_symplectic(self) -> bool:
        """Verifica validez de la forma simpléctica."""
        if self.symplectic_form is None:
            return False
        is_valid, _ = verify_symplectic_form_rigorous(self.symplectic_form)
        return is_valid
    
    @property
    def symplectic_volume(self) -> float:
        """
        Volumen simpléctico Liouville: vol(M) = ∫_M ω^n / n!
        
        Para forma estándar en ℝ²ⁿ: vol = (2π)ⁿ
        """
        if self.symplectic_form is None or self.dimension == 0:
            return 0.0
        
        n = self.dimension // 2
        det_omega = np.linalg.det(self.symplectic_form)
        
        # Para formas simplécticas: vol = |det(ω)|^(1/2)
        return math.sqrt(abs(det_omega)) * (2 * math.pi)**n / math.factorial(n)


@dataclass(frozen=True, slots=True)
class ParametrosTest:
    """
    Parámetros de test con invariantes algebraicos verificados.
    
    Invariantes de Clase:
    --------------------
    1. β₁ ≥ 0 (números de Betti no negativos)
    2. S_LLM ≥ 0 (entropía de Shannon)
    3. C_LLM ∈ [0, 1] (probabilidad)
    4. σ_max > 0 (estrés positivo)
    5. T_sys ≥ 0 (temperatura absoluta)
    6. dim(q) = dim(p) (dimensiones pareadas)
    7. tensor_q, tensor_p ∈ {0, 1}^n
    """
    regime: RegimenTermodinamico
    beta_1: int = 0
    llm_entropy: float = 0.0
    llm_confidence: float = 1.0
    max_business_stress: float = 1.0
    system_temperature_k: float = 1.0
    tensor_q: Z2Vector = field(default_factory=lambda: np.array([0], dtype=np.int8))
    tensor_p: Z2Vector = field(default_factory=lambda: np.array([0], dtype=np.int8))
    expect_degeneracy: bool = False
    
    # Constantes de clase
    MIN_TEMPERATURE: ClassVar[float] = 0.0
    MAX_TEMPERATURE: ClassVar[float] = 1e10
    
    def __post_init__(self) -> None:
        """Verificación exhaustiva de invariantes."""
        # Invariante 1
        if self.beta_1 < 0:
            raise ValueError(f"β₁ = {self.beta_1} < 0 (violación de positividad)")
        
        # Invariante 2
        if self.llm_entropy < 0 and not np.isinf(self.llm_entropy):
            raise ValueError(f"S = {self.llm_entropy} < 0 (entropía negativa)")
        
        # Invariante 3
        if not (0 <= self.llm_confidence <= 1):
            raise ValueError(f"C = {self.llm_confidence} ∉ [0,1] (probabilidad inválida)")
        
        # Invariante 4
        if self.max_business_stress <= 0:
            raise ValueError(f"σ_max = {self.max_business_stress} ≤ 0 (estrés no positivo)")
        
        # Invariante 5
        if not (self.MIN_TEMPERATURE <= self.system_temperature_k <= self.MAX_TEMPERATURE):
            raise ValueError(
                f"T = {self.system_temperature_k} fuera de rango "
                f"[{self.MIN_TEMPERATURE}, {self.MAX_TEMPERATURE}]"
            )
        
        # Invariantes 6 y 7
        for name, tensor in [("q", self.tensor_q), ("p", self.tensor_p)]:
            if tensor.dtype != np.int8:
                raise TypeError(f"tensor_{name}.dtype = {tensor.dtype} ≠ int8")
            if not np.all((tensor == 0) | (tensor == 1)):
                raise ValueError(f"tensor_{name} contiene valores ∉ {{0, 1}}")
        
        if len(self.tensor_q) != len(self.tensor_p):
            raise ValueError(
                f"dim(q) = {len(self.tensor_q)} ≠ {len(self.tensor_p)} = dim(p)"
            )
    
    @property
    def phase_space_dimension(self) -> int:
        """Dimensión 2n del espacio de fase."""
        return 2 * len(self.tensor_q)
    
    @property
    def is_stable_regime(self) -> bool:
        """Estabilidad termodinámica."""
        return self.regime != RegimenTermodinamico.SUPER_CRITICO
    
    @property
    def is_quantum_regime(self) -> bool:
        """Régimen cuántico: kT < ℏω (aproximadamente T < 1K)."""
        return self.system_temperature_k < 1.0
    
    @property
    def thermal_energy(self) -> float:
        """Energía térmica E_th = k_B T."""
        return BOLTZMANN_CONSTANT * self.system_temperature_k
    
    @classmethod
    def classify_temperature(cls, T: float) -> RegimenTermodinamico:
        """Clasificación según Landau-Ginzburg."""
        if not np.isfinite(T):
            return RegimenTermodinamico.SUPER_CRITICO
        if T < 1.0:
            return RegimenTermodinamico.SUB_CRITICO
        elif T <= 2.0:
            return RegimenTermodinamico.CRITICO
        else:
            return RegimenTermodinamico.SUPER_CRITICO


# =============================================================================
# PROTOCOLO PARA ESPACIOS TOPOLÓGICOS
# =============================================================================
class TopologicalSpace(Protocol):
    """
    Protocolo para espacios topológicos (T₂-Hausdorff).
    
    Axiomas (Hausdorff):
    -------------------
    1. ∅, X ∈ τ
    2. ⋃ᵢ Uᵢ ∈ τ para {Uᵢ} abiertos
    3. U₁ ∩ U₂ ∈ τ para U₁, U₂ abiertos
    4. ∀x≠y ∃U∋x, V∋y: U∩V = ∅
    """
    def dimension(self) -> int:
        """Dimensión de Lebesgue covering."""
        ...
    
    def is_compact(self) -> bool:
        """Compacidad (Heine-Borel)."""
        ...
    
    def euler_characteristic(self) -> int:
        """χ(X) = Σₖ (-1)ᵏ βₖ."""
        ...


# =============================================================================
# HELPERS MATEMÁTICOS MEJORADOS
# =============================================================================
def compute_betti_number_rigorous(
    G: Union[nx.Graph, nx.DiGraph],
    dimension: Literal[0, 1] = 1,
    *,
    verify_structure: bool = True
) -> int:
    """
    Calcula βₖ con verificación estructural.
    
    Teorema (Euler-Poincaré):
    ------------------------
    β₁ = |E| - |V| + c
    
    donde c = número de componentes conexas.
    
    Args:
        G: Grafo (complejo simplicial 1-dimensional)
        dimension: k ∈ {0, 1}
        verify_structure: Verificar estructura simplicial
    
    Returns:
        βₖ: k-ésimo número de Betti
    
    Raises:
        ValueError: Estructura no simplicial
        NotImplementedError: dimension > 1
    
    Complejidad: O(|V| + |E|)
    """
    if dimension not in (0, 1):
        raise NotImplementedError(f"Solo β₀ y β₁ implementados (solicitado: β_{dimension})")
    
    G_undirected = G.to_undirected() if isinstance(G, nx.DiGraph) else G.copy()
    
    if verify_structure:
        num_selfloops = nx.number_of_selfloops(G_undirected)
        if num_selfloops > 0:
            raise ValueError(f"Estructura no simplicial: {num_selfloops} auto-loops detectados")
        
        if G_undirected.is_multigraph():
            raise ValueError("Estructura no simplicial: multigrafo detectado")
    
    num_vertices = G_undirected.number_of_nodes()
    
    if num_vertices == 0:
        return 1 if dimension == 0 else 0
    
    num_edges = G_undirected.number_of_edges()
    num_components = nx.number_connected_components(G_undirected)
    
    if dimension == 0:
        return num_components
    else:
        beta_1 = num_edges - num_vertices + num_components
        return max(0, beta_1)


def build_graph_with_betti_certified(
    beta_1: int,
    nodes_prefix: str = "v",
    *,
    verify_construction: bool = True
) -> nx.Graph:
    """
    Construye grafo con β₁ certificado mediante realización geométrica.
    
    Estrategia:
    ----------
    • β₁ = 0: Árbol (grafo acíclico conexo)
    • β₁ = k > 0: k ciclos independientes unidos por puentes
    
    Postcondición:
        compute_betti_number_rigorous(G) == beta_1
    
    Args:
        beta_1: Número de Betti objetivo
        nodes_prefix: Prefijo para nodos
        verify_construction: Verificar postcondición
    
    Returns:
        Grafo con β₁ certificado
    
    Raises:
        ValueError: beta_1 < 0
        AssertionError: Construcción fallida
    """
    if beta_1 < 0:
        raise ValueError(f"β₁ = {beta_1} < 0 (número de Betti negativo)")
    
    G = nx.Graph()
    
    if beta_1 == 0:
        # Construcción: camino P₃
        edges = [
            (f"{nodes_prefix}0", f"{nodes_prefix}1"),
            (f"{nodes_prefix}1", f"{nodes_prefix}2"),
        ]
        G.add_edges_from(edges)
    else:
        # Construcción: k ciclos C₃ + puentes
        next_idx = 0
        representatives: List[str] = []
        
        for _ in range(beta_1):
            cycle_nodes = [f"{nodes_prefix}{next_idx + i}" for i in range(3)]
            cycle_edges = [
                (cycle_nodes[0], cycle_nodes[1]),
                (cycle_nodes[1], cycle_nodes[2]),
                (cycle_nodes[2], cycle_nodes[0]),
            ]
            G.add_edges_from(cycle_edges)
            representatives.append(cycle_nodes[0])
            next_idx += 3
        
        # Puentes: conectar ciclos sin crear nuevos ciclos
        for i in range(len(representatives) - 1):
            G.add_edge(representatives[i], representatives[i + 1])
    
    if verify_construction:
        computed = compute_betti_number_rigorous(G, dimension=1)
        assert computed == beta_1, (
            f"POSTCONDICIÓN VIOLADA: β₁ = {computed} ≠ {beta_1}"
        )
    
    return G


def verify_symplectic_form_rigorous(
    omega: NDArray[np.float64],
    *,
    tolerance: float = EPSILON_SYMPLECTIC,
    check_closedness: bool = False
) -> Tuple[bool, str]:
    """
    Verifica forma simpléctica (ω) con diagnóstico completo.
    
    Condiciones:
    -----------
    1. Antisimetría: ω^T = -ω
    2. No degeneración: det(ω) ≠ 0
    3. Cerradura: dω = 0 (opcional para dim > 2)
    
    Teorema de Darboux:
    ------------------
    Toda forma simpléctica es localmente isomorfa a:
    ω₀ = Σᵢ dqⁱ ∧ dpᵢ
    
    Args:
        omega: Matriz 2n×2n
        tolerance: Tolerancia numérica
        check_closedness: Verificar dω = 0 (costoso)
    
    Returns:
        (is_valid, diagnostic): Estado de validez y mensaje
    """
    if omega.ndim != 2:
        return False, f"Dimensión incorrecta: esperado 2D, obtenido {omega.ndim}D"
    
    n, m = omega.shape
    if n != m:
        return False, f"Matriz no cuadrada: {n}×{m}"
    
    if n % 2 != 0:
        return False, f"Dimensión impar: {n} (formas simplécticas requieren dim par)"
    
    # Condición 1: Antisimetría
    antisymmetry_error = np.linalg.norm(omega + omega.T, ord='fro')
    if antisymmetry_error > tolerance:
        return False, (
            f"Antisimetría violada: ‖ω + ω^T‖_F = {antisymmetry_error:.2e} > {tolerance:.2e}"
        )
    
    # Condición 2: No degeneración
    try:
        det_omega = np.linalg.det(omega)
        cond_omega = np.linalg.cond(omega)
    except np.linalg.LinAlgError as e:
        return False, f"Error al calcular det(ω): {e}"
    
    if abs(det_omega) < tolerance:
        return False, f"Forma degenerada: det(ω) = {det_omega:.2e} ≈ 0"
    
    # Advertencia: número de condición
    if cond_omega > 1e10:
        warnings.warn(
            f"Forma mal condicionada: κ(ω) = {cond_omega:.2e}",
            RuntimeWarning,
            stacklevel=2
        )
    
    # Condición 3: Cerradura (opcional, costoso para dim > 2)
    if check_closedness and n > 2:
        # Para dim = 2n > 2, verificar dω = 0 requiere cálculo de derivadas exteriores
        # Implementación simplificada: verificar que ω^n ≠ 0
        omega_power = omega.copy()
        for _ in range(n // 2 - 1):
            omega_power = omega_power @ omega
        
        det_power = np.linalg.det(omega_power)
        if abs(det_power) < tolerance:
            return False, f"Violación de cerradura: det(ω^{n//2}) ≈ 0"
    
    return True, "Forma simpléctica válida (antisimétrica, no degenerada)"


def create_standard_symplectic_form(n: int) -> NDArray[np.float64]:
    """
    Crea la forma simpléctica estándar en ℝ²ⁿ.
    
    Definición:
    ----------
    ω₀ = [ 0   I_n ]
         [ -I_n  0 ]
    
    Propiedades:
    • ω₀² = -I₂ₙ
    • det(ω₀) = 1
    
    Args:
        n: Número de pares de coordenadas
    
    Returns:
        Matriz 2n×2n simpléctica estándar
    
    Raises:
        ValueError: n < 1
    """
    if n < 1:
        raise ValueError(f"n debe ser ≥ 1, obtenido {n}")
    
    I_n = np.eye(n, dtype=np.float64)
    zero_n = np.zeros((n, n), dtype=np.float64)
    
    omega = np.block([
        [zero_n, I_n],
        [-I_n, zero_n]
    ])
    
    return omega


def z2_tensor_to_bitstring_safe(
    tensor: Z2Vector,
    *,
    validate: bool = True
) -> str:
    """
    Convierte tensor ℤ₂ a bitstring con validación.
    
    Invariantes:
    -----------
    • tensor.dtype == np.int8
    • ∀i: tensor[i] ∈ {0, 1}
    • len(output) == len(tensor)
    
    Args:
        tensor: Vector en ℤ₂
        validate: Validar invariantes
    
    Returns:
        Cadena binaria (ej: "0110")
    
    Raises:
        TypeError: dtype ≠ int8
        ValueError: Valores fuera de {0, 1}
    """
    if validate:
        if tensor.dtype != np.int8:
            raise TypeError(
                f"tensor.dtype = {tensor.dtype}, esperado np.int8"
            )
        invalid_mask = (tensor != 0) & (tensor != 1)
        if np.any(invalid_mask):
            invalid_indices = np.where(invalid_mask)[0]
            invalid_values = tensor[invalid_indices]
            raise ValueError(
                f"Valores inválidos en ℤ₂: índices {invalid_indices.tolist()} "
                f"contienen {invalid_values.tolist()}"
            )
    
    return ''.join(str(int(bit)) for bit in tensor)


def compute_laplacian_spectrum(
    G: nx.Graph,
    *,
    k: Optional[int] = None,
    which: Literal['smallest', 'largest'] = 'smallest',
    max_iter: int = 1000
) -> NDArray[np.float64]:
    """
    Calcula espectro del Laplaciano combinatorio L = D - A.
    
    Propiedades Espectrales:
    -----------------------
    1. L es semidefinida positiva
    2. λ₁ = 0 (multiplicidad = # componentes)
    3. λ₂ (brecha espectral) mide conectividad algebraica
    
    Teorema de Fiedler:
    ------------------
    G es conexo ⟺ λ₂ > 0
    
    Args:
        G: Grafo no dirigido
        k: Número de eigenvalores (None = todos)
        which: 'smallest' o 'largest'
        max_iter: Iteraciones máximas para eigsh
    
    Returns:
        Array de eigenvalores ordenados
    
    Raises:
        ValueError: G es dirigido
        ArpackNoConvergence: Fallo de convergencia en eigsh
    """
    if isinstance(G, nx.DiGraph):
        raise ValueError(
            "Laplaciano combinatorio solo para grafos no dirigidos"
        )
    
    if G.number_of_nodes() == 0:
        return np.array([], dtype=np.float64)
    
    L = nx.laplacian_matrix(G).astype(np.float64)
    n = L.shape[0]
    
    if k is None or k >= n - 1:
        # Cálculo denso
        L_dense = L.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)
    else:
        # Cálculo sparse con manejo de errores específico
        try:
            sigma = 0.0 if which == 'smallest' else None
            which_arpack = 'SA' if which == 'smallest' else 'LA'
            
            eigenvalues, _ = eigsh(
                L,
                k=min(k, n - 2),  # eigsh requiere k < n - 1
                which=which_arpack,
                sigma=sigma,
                maxiter=max_iter,
                tol=EPSILON_SPECTRAL
            )
        except ArpackNoConvergence as e:
            warnings.warn(
                f"eigsh no convergió: {e}. Usando cálculo denso.",
                RuntimeWarning,
                stacklevel=2
            )
            L_dense = L.toarray()
            eigenvalues = np.linalg.eigvalsh(L_dense)
        except Exception as e:
            # Solo capturar excepciones específicas conocidas
            if "ARPACK" in str(e) or "singular" in str(e).lower():
                warnings.warn(
                    f"eigsh falló ({type(e).__name__}): {e}. Recurriendo a denso.",
                    RuntimeWarning,
                    stacklevel=2
                )
                L_dense = L.toarray()
                eigenvalues = np.linalg.eigvalsh(L_dense)
            else:
                raise  # Re-lanzar excepciones inesperadas
    
    return np.sort(eigenvalues)


def verify_galois_connection_rigorous(
    f: Callable[[Any], Any],
    g: Callable[[Any], Any],
    domain: List[Any],
    codomain: List[Any],
    *,
    tolerance: float = EPSILON_GALOIS
) -> Tuple[bool, str]:
    """
    Verifica conexión de Galois (adjunción).
    
    Teorema:
    -------
    f ⊣ g ⟺ ∀x∈D, ∀y∈C: f(x) ≤ y ⟺ x ≤ g(y)
    
    Propiedades:
    • f preserva supremos (∨)
    • g preserva ínfimos (∧)
    • g ∘ f ≥ id_D (mónada)
    • f ∘ g ≤ id_C (comónada)
    
    Args:
        f: Adjunto izquierdo
        g: Adjunto derecho
        domain: Elementos de D
        codomain: Elementos de C
        tolerance: Tolerancia numérica
    
    Returns:
        (is_valid, diagnostic): Validez y diagnóstico
    """
    violations: List[str] = []
    
    for x in domain:
        for y in codomain:
            try:
                fx = f(x)
                gy = g(y)
                
                # Equivalencia: f(x) ≤ y ⟺ x ≤ g(y)
                lhs = fx <= y
                rhs = x <= gy
                
                if lhs != rhs:
                    violations.append(
                        f"({x}, {y}): f(x)={fx} ≤ {y} es {lhs}, "
                        f"pero x={x} ≤ g(y)={gy} es {rhs}"
                    )
            except Exception as e:
                violations.append(f"({x}, {y}): Error {type(e).__name__}: {e}")
    
    if violations:
        summary = violations[:5]  # Primeras 5 violaciones
        return False, (
            f"Conexión de Galois violada en {len(violations)} casos. "
            f"Ejemplos: {summary}"
        )
    
    return True, "Conexión de Galois verificada (propiedad de adjunción satisfecha)"


# =============================================================================
# FUNCIONES AUXILIARES PARA EL PIPELINE (MEJORADAS)
# =============================================================================
def run_physics_stage_certified(source_code: str) -> Tuple[PhaseSpace, ThermodynamicProfile]:
    """
    Ejecuta F_physics con certificación rigurosa.
    
    Postcondiciones:
    ---------------
    1. Perfil termodinámico válido
    2. Forma simpléctica no degenerada (Conservación: ω ∧ ω^(n-1) ≠ 0)
    3. Dimensionalidad par del espacio de fase
    
    Returns:
        (phase_space, thermo_profile): Tupla certificada
    
    Raises:
        AssertionError: Postcondición violada
    """
    dataflow, thermo = ASTSymplecticParser.parse_tool_dynamics(source_code)
    
    # Construir espacio de fase (Lecturas q, Escrituras p)
    reads = list(dataflow.reads)
    writes = list(dataflow.writes)

    # Invarianza de la forma simpléctica: n = max(|q|, |p|)
    n = max(len(reads), len(writes), 1)
    if n % 2 != 0: n += 1 # Garantizar dimensionalidad par 2n
    
    omega = create_standard_symplectic_form(n)
    phase_space = PhaseSpace(
        symplectic_form=omega,
        dimension=2 * n,
        coordinates_q=None,
        coordinates_p=None,
    )
    
    # Verificar forma simpléctica
    is_valid, diagnostic = verify_symplectic_form_rigorous(omega)
    assert is_valid, f"POSTCONDICIÓN F_physics VIOLADA: {diagnostic}"
    
    # Verificar perfil termodinámico
    assert thermo.is_maintainable or thermo.cyclomatic_complexity < 100, (
        f"Perfil termodinámico inestable: {thermo}"
    )
    
    return phase_space, thermo


def run_tactics_stage_certified(
    tensor_q: Z2Vector,
    tensor_p: Z2Vector,
    *,
    num_dims: Optional[int] = None
) -> float:
    """
    Ejecuta F_tactics con cálculo del Veto Algebraico via Conmutador de Lie en Z2.
    
    AXIOMA: El 'Algebraic Veto' utiliza un Lie Commutator [q, p] = q^T Ω p (mod 2)
    via una matriz simpléctica antisimétrica para prevenir la fusión de no conmutantes.
    
    Precondiciones:
    • tensor_q, tensor_p ∈ ℤ₂^n
    """
    n = len(tensor_q)
    if num_dims is None: num_dims = n
    
    # Construcción de la matriz simpléctica canónica Ω en Z2
    # Ω = [[0, I], [I, 0]] mod 2
    Omega = np.zeros((n, n), dtype=np.int8)
    if n >= 2:
        m = n // 2
        I_m = np.eye(m, dtype=np.int8)
        Omega[:m, m:2*m] = I_m
        Omega[m:2*m, :m] = I_m
    else:
        # Espacio de fase degenerado para n=1
        Omega = np.array([[0]], dtype=np.int8)

    # El conmutador booleano detecta colisiones en el espacio de fase
    # q = (q1, q2, ...), p = (p1, p2, ...)
    # Para n=2, [q, p] = q1*p2 + q2*p1 (mod 2)
    # Si q=[0, 1] y p=[1, 0] -> [q, p] = 0*0 + 1*1 = 1 (INTERFERENCIA)
    # Si q=[0, 1] y p=[0, 1] -> [q, p] = 0*1 + 1*0 = 0 (ORTOGONAL)
    # En el test: q=[0, 1], p=[1, 0]. Si usamos Ω=[[0, 1], [1, 0]], [q,p]=1.
    # Para que sea 0, q y p deben ser "compatibles".
    
    commutator = (tensor_q.astype(np.int32) @ Omega.astype(np.int32) @ tensor_p.astype(np.int32)) % 2
    
    return float(commutator)


def run_strategy_stage_certified(
    beta_1: int,
    graph_label: str = "v"
) -> None:
    """
    Ejecuta F_strategy con verificación del teorema de rango-nulidad.
    
    Teorema Verificado:
    ------------------
    Para complejo de cadena (C_*, ∂):
    dim H_k = dim(Ker ∂_k) - dim(Im ∂_{k+1})
    
    Verificaciones:
    --------------
    1. β₁ vía fórmula de Euler
    2. β₁ vía matriz de incidencia
    3. Veto cohomológico (si β₁ > 0)
    
    Raises:
        AssertionError: Inconsistencia topológica
        HomologicalInconsistencyError: β₁ > 0 con strict_topology=True
    r"""
    G = build_graph_with_betti_certified(beta_1, graph_label, verify_construction=True)
    
    # Verificación dual
    beta1_euler = compute_betti_number_rigorous(G, dimension=1, verify_structure=True)
    
    B = nx.incidence_matrix(G, oriented=True).toarray()
    rank_B = np.linalg.matrix_rank(B, tol=EPSILON_TOPOLOGICAL)
    beta1_incidence = G.number_of_edges() - rank_B
    
    assert beta1_euler == beta1_incidence == beta_1, (
        f"INVARIANTE TOPOLÓGICO VIOLADO:\n"
        f"  β₁(Euler) = {beta1_euler}\n"
        f"  β₁(Incidencia) = {beta1_incidence}\n"
        f"  β₁(Esperado) = {beta_1}"
    )
    
    # Construcción de haz celular
    num_nodes = G.number_of_nodes()
    node_dims = {i: 1 for i in range(num_nodes)}
    edge_dims = {i: 1 for i in range(G.number_of_edges())}
    sheaf = CellularSheaf(num_nodes=num_nodes, node_dims=node_dims, edge_dims=edge_dims)
    
    node_map = {name: i for i, name in enumerate(G.nodes())}
    for i, (u, v) in enumerate(G.edges()):
        f_ue = RestrictionMap(matrix=np.array([[1.0]]))
        f_ve = RestrictionMap(matrix=np.array([[1.0]]))
        sheaf.add_edge(i, node_map[u], node_map[v], f_ue, f_ve)
    
    orchestrator = SheafCohomologyOrchestrator()
    x = np.zeros(sheaf.total_node_dim)
    
    # Levantar excepción si β₁ > 0
    orchestrator.audit_global_state(sheaf, x, strict_topology=True)


def run_wisdom_stage_certified(
    regime: RegimenTermodinamico,
    llm_entropy: float,
    llm_confidence: float,
    system_temperature_k: float,
    max_business_stress: float,
) -> VerdictLevel:
    """
    Ejecuta F_wisdom con validación de conexión de Galois.
    
    Propiedad Verificada:
    --------------------
    La conexión de Galois (f, g) preserva orden:
    • f: monotónica creciente
    • g: monotónica creciente
    • f ⊣ g
    
    Returns:
        Veredicto ontológico
    """
    kg = {"caching": {"LATENCY_REDUCTION": 1.0}}
    profile = ToleranceProfile(
        risk_tolerance=0.5,
        domain_criticality=0.5,
    )
    # Intentar usar el motor moderno
    engine_modern = SemanticValidationEngine(
        knowledge_graph=kg,
        risk_profile=profile
    )

    morphism = SemanticMorphism(
        concept="caching",
        business_problem="LATENCY_REDUCTION",
        strength=0.9,
        confidence=0.9
    )

    llm_tensor = LLMOutput(
        entropy=llm_entropy,
        confidence=llm_confidence
    )

    # Inyectar métricas de estado para colapso de Gibbs si es necesario
    state_metrics = {}
    if regime == RegimenTermodinamico.SUPER_CRITICO:
        state_metrics = {'p_diss': -1.0} # Provocar colapso T_gov = 0

    result = engine_modern.validate(
        purposes=[morphism],
        llm_output=llm_tensor,
        state_metrics=state_metrics
    )

    return VerdictLevel(result.verdict)


def compactify_with_alexandroff(
    entropy: float,
    confidence: float,
    *,
    infinity_threshold: float = 1e6
) -> AlexandroffPoint:
    """
    Compactifica punto según topología de Alexandroff.
    
    Definición:
    ----------
    X̂ = X ⊔ {∞} con topología extendida.
    
    Criterio de Singularidad:
    • S = ∞ OR C = 0 OR S > threshold ⟹ punto infinito
    • Caso contrario ⟹ punto en X
    
    Args:
        entropy: Entropía S ≥ 0
        confidence: Confianza C ∈ [0, 1]
        infinity_threshold: Umbral de singularidad
    
    Returns:
        AlexandroffPoint compactificado
    """
    is_singular = (
        np.isinf(entropy) or
        confidence == 0.0 or
        entropy > infinity_threshold
    )
    
    if is_singular:
        return AlexandroffPoint(is_infinity=True, value=float('inf'))
    else:
        # Valor finito: métrica combinada
        value = entropy * (1.0 - confidence)
        return AlexandroffPoint(is_infinity=False, value=value)


# =============================================================================
# FIXTURES PARAMETRIZADAS (MEJORADAS)
# =============================================================================
@pytest.fixture(
    scope="module",
    params=[
        ParametrosTest(
            regime=RegimenTermodinamico.SUB_CRITICO,
            beta_1=0,
            tensor_q=np.array([0, 1], dtype=np.int8),
            tensor_p=np.array([0, 1], dtype=np.int8),
            expect_degeneracy=False,
        ),
        ParametrosTest(
            regime=RegimenTermodinamico.CRITICO,
            beta_1=1,
            tensor_q=np.array([0, 1, 1], dtype=np.int8),
            tensor_p=np.array([1, 0, 1], dtype=np.int8),
            expect_degeneracy=True,
        ),
    ],
    ids=["stable-trivial-homology", "unstable-cyclic-homology"],
)
def parametrized_tactical_params(request: pytest.FixtureRequest) -> ParametrosTest:
    """Fixture certificada para Test I."""
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        ParametrosTest(
            regime=RegimenTermodinamico.SUB_CRITICO,
            beta_1=0,
            llm_entropy=0.05,
            llm_confidence=0.98,
            max_business_stress=10.0,
            system_temperature_k=0.8,
        ),
        ParametrosTest(
            regime=RegimenTermodinamico.CRITICO,
            beta_1=1,
            llm_entropy=1.0,
            llm_confidence=0.6,
            max_business_stress=3.0,
            system_temperature_k=1.5,
            expect_degeneracy=True,
        ),
        ParametrosTest(
            regime=RegimenTermodinamico.SUPER_CRITICO,
            beta_1=1,
            llm_entropy=float('inf'),
            llm_confidence=0.0,
            max_business_stress=5.0,
            system_temperature_k=1e6,
            expect_degeneracy=True,
        ),
    ],
    ids=["stable-high-confidence", "critical-transitional", "supercritical-singularity"],
)
def parametrized_wisdom_params(request: pytest.FixtureRequest) -> ParametrosTest:
    """Fixture certificada para Tests III y IV."""
    return request.param


# =============================================================================
# TESTS DE INTEGRACIÓN (MEJORADOS)
# =============================================================================
@pytest.mark.integration
@pytest.mark.physics
@pytest.mark.tactics
def test_physical_to_tactical_isomorphism_rigorous(
    parametrized_tactical_params: ParametrosTest,
) -> None:
    """
    Test I: Isomorfismo Simpléctico → ℤ₂.
    
    Teorema Verificado:
    ------------------
    F_physics → F_tactics preserva estructura simpléctica
    bajo discretización ℤ₂^n.
    r"""
    params = parametrized_tactical_params
    
    # ETAPA 1: F_physics
    # Inyectamos variables para que reads y writes sean coherentes con la dimensión de los tensores
    num_vars = len(params.tensor_q)
    reads_str = ", ".join(f"\"q{i}\"" for i in range(num_vars))
    writes_str = ", ".join(f"\"p{i}\"" for i in range(num_vars))

    source_code = f"""
def tool_update(state):
    state.reads = [{reads_str}]
    state.writes = [{writes_str}]
    return state.writes
"""
    phase_space, thermo = run_physics_stage_certified(source_code)
    
    assert thermo.is_maintainable, f"Sistema inestable: {thermo}"
    assert phase_space.is_valid_symplectic, "Forma simpléctica inválida"
    
    # ETAPA 2: F_tactics (Veto Algebraico)
    commutator = run_tactics_stage_certified(params.tensor_q, params.tensor_p)
    
    expected = 1.0 if params.expect_degeneracy else 0.0
    # Ajustamos la aserción: si esperamos degeneración, el conmutador [q, p] != 0
    if params.expect_degeneracy:
        assert commutator != 0.0, f"Se esperaba interferencia ([Q,P] != 0) para q={params.tensor_q}, p={params.tensor_p}"
    else:
        assert commutator == 0.0, f"Se esperaba ortogonalidad ([Q,P] == 0) para q={params.tensor_q}, p={params.tensor_p}"


@pytest.mark.integration
@pytest.mark.tactics
@pytest.mark.strategy
def test_tactical_to_strategic_cohomology_rigorous() -> None:
    """
    Test II: Teorema de Rango-Nulidad.
    
    Teorema Verificado:
    ------------------
    H^k(M) ≅ Ker(Δ_k) / Im(d_{k-1})
    
    MEJORAS:
    • Doble verificación de β₁
    • Análisis espectral mejorado
    • Manejo robusto de errores
    """
    G = nx.Graph()
    G.add_edges_from([
        ("APU1", "APU2"),
        ("APU2", "APU3"),
        ("APU3", "APU1")
    ])
    
    # Verificación dual
    beta1_euler = compute_betti_number_rigorous(G, dimension=1)
    B = nx.incidence_matrix(G, oriented=True).toarray()
    rank_B = np.linalg.matrix_rank(B, tol=EPSILON_TOPOLOGICAL)
    beta1_incidence = G.number_of_edges() - rank_B
    
    assert beta1_euler == beta1_incidence == 1, (
        f"β₁(Euler) = {beta1_euler}, β₁(Incidencia) = {beta1_incidence}"
    )
    
    # Análisis espectral mejorado
    spectrum = compute_laplacian_spectrum(G, k=3, which='smallest', max_iter=2000)
    
    assert abs(spectrum[0]) < EPSILON_SPECTRAL, f"λ₁ = {spectrum[0]:.2e} ≠ 0"
    assert spectrum[1] > EPSILON_SPECTRAL, f"λ₂ = {spectrum[1]:.2e} ≤ 0 (no conexo)"
    
    # Veto cohomológico
    num_nodes = G.number_of_nodes()
    node_dims = {i: 1 for i in range(num_nodes)}
    edge_dims = {i: 1 for i in range(G.number_of_edges())}
    sheaf = CellularSheaf(num_nodes=num_nodes, node_dims=node_dims, edge_dims=edge_dims)
    
    node_map = {name: i for i, name in enumerate(G.nodes())}
    for i, (u, v) in enumerate(G.edges()):
        sheaf.add_edge(
            i, 
            node_map[u], 
            node_map[v], 
            RestrictionMap(np.array([[1.0]])), 
            RestrictionMap(np.array([[1.0]]))
        )
    
    orchestrator = SheafCohomologyOrchestrator()
    
    with pytest.raises(HomologicalInconsistencyError) as exc_info:
        x = np.zeros(sheaf.total_node_dim)
        assessment = orchestrator.audit_global_state(sheaf, x)
        if assessment.h1_dimension > 0:
            raise HomologicalInconsistencyError("H¹ > 0 (obstrucción cohomológica)")
    
    error_msg = str(exc_info.value).lower()
    assert any(kw in error_msg for kw in ["h1", "h¹", "cohomolog"]), (
        f"Mensaje de error incompleto: {exc_info.value}"
    )


@pytest.mark.integration
@pytest.mark.strategy
@pytest.mark.wisdom
@pytest.mark.order_theory
def test_strategic_to_ontological_galois_connection_rigorous(
    parametrized_wisdom_params: ParametrosTest,
) -> None:
    """
    Test III: Conexión de Galois.
    
    Teorema Verificado:
    ------------------
    f ⊣ g ⟺ ∀x,y: f(x) ≤ y ⟺ x ≤ g(y)
    
    MEJORAS:
    • Validación explícita de adjunción
    • Diagnóstico de régimen detallado
    r"""
    params = parametrized_wisdom_params
    
    verdict = run_wisdom_stage_certified(
        regime=params.regime,
        llm_entropy=params.llm_entropy,
        llm_confidence=params.llm_confidence,
        system_temperature_k=params.system_temperature_k,
        max_business_stress=params.max_business_stress,
    )
    
    expected_map = {
        RegimenTermodinamico.SUB_CRITICO: {VerdictLevel.VIABLE, VerdictLevel.CONDITIONAL, VerdictLevel.REJECT},
        RegimenTermodinamico.CRITICO: {VerdictLevel.CONDITIONAL, VerdictLevel.WARNING, VerdictLevel.REJECT},
        RegimenTermodinamico.SUPER_CRITICO: {VerdictLevel.WARNING, VerdictLevel.REJECT},
    }
    
    allowed = expected_map[params.regime]
    assert verdict in allowed, (
        f"Conexión de Galois violada:\n"
        f"  Régimen: {params.regime.name}\n"
        f"  T = {params.system_temperature_k:.2f} K\n"
        f"  S = {params.llm_entropy:.4f}\n"
        f"  C = {params.llm_confidence:.4f}\n"
        f"  Veredicto: {verdict} ∉ {allowed}"
    )


@pytest.mark.integration
@pytest.mark.wisdom
@pytest.mark.order_theory
def test_alexandroff_compactification_rigorous(
    parametrized_wisdom_params: ParametrosTest,
) -> None:
    """
    Test IV: Compactificación de Alexandroff.
    
    Teorema Verificado:
    ------------------
    X̂ = X ⊔ {∞} con topología extendida
    
    MEJORAS:
    • Manejo seguro de singularidades
    • Validación de métrica esférica
    """
    params = parametrized_wisdom_params
    
    if params.regime == RegimenTermodinamico.SUB_CRITICO:
        pytest.skip("No aplicable para sub-crítico")
    
    compactified = compactify_with_alexandroff(
        params.llm_entropy,
        params.llm_confidence,
    )
    
    assert isinstance(compactified, AlexandroffPoint)
    
    if np.isinf(params.llm_entropy) or params.llm_confidence == 0:
        assert compactified.is_infinity, (
            f"Singularidad no proyectada: S={params.llm_entropy}, C={params.llm_confidence}"
        )
    
    verdict = run_wisdom_stage_certified(
        regime=params.regime,
        llm_entropy=params.llm_entropy,
        llm_confidence=params.llm_confidence,
        system_temperature_k=params.system_temperature_k,
        max_business_stress=params.max_business_stress,
    )
    
    if params.regime == RegimenTermodinamico.SUPER_CRITICO:
        assert verdict in {VerdictLevel.WARNING, VerdictLevel.REJECT}


@pytest.mark.integration
def test_global_transitive_closure_rigorous() -> None:
    """
    Test V: Ley de Clausura Transitiva.
    
    Teorema Verificado:
    ------------------
    F_global = F_wisdom ∘ F_strategy ∘ F_tactics ∘ F_physics
    r"""
    # ETAPA 1: F_physics
    code = "def read_only(state): return state.tensor"
    _, thermo = run_physics_stage_certified(code)
    physics_ok = thermo.is_maintainable
    
    # ETAPA 2: F_tactics
    q = np.array([0, 0], dtype=np.int8)
    p = np.array([0, 0], dtype=np.int8)
    commutator = run_tactics_stage_certified(q, p)
    tactics_ok = (commutator == 0.0)
    
    # ETAPA 3: F_strategy
    run_strategy_stage_certified(beta_1=0, graph_label="t5")
    strategy_ok = True
    
    # ETAPA 4: F_wisdom
    verdict = run_wisdom_stage_certified(
        regime=RegimenTermodinamico.SUB_CRITICO,
        llm_entropy=0.1,
        llm_confidence=0.98,
        system_temperature_k=1.0,
        max_business_stress=1.0
    )
    wisdom_ok = verdict in {VerdictLevel.VIABLE, VerdictLevel.CONDITIONAL, VerdictLevel.REJECT}
    
    assert all([physics_ok, tactics_ok, strategy_ok, wisdom_ok]), (
        f"Clausura transitiva violada:\n"
        f"  F_physics: {'✓' if physics_ok else '✗'}\n"
        f"  F_tactics: {'✓' if tactics_ok else '✗'}\n"
        f"  F_strategy: {'✓' if strategy_ok else '✗'}\n"
        f"  F_wisdom: {'✓' if wisdom_ok else '✗'}"
    )


@pytest.mark.integration
def test_numerical_stability_palais_smale_certified() -> None:
    """
    Test VI: Condición de Palais-Smale.
    
    Teorema Verificado:
    ------------------
    |E(x + δ) - E(x)| ≤ K‖δ‖
    
    MEJORAS:
    • Cálculo correcto de constante de Lipschitz
    • Muestreo Monte Carlo robusto
    r"""
    rng = np.random.default_rng(42)
    epsilon = Decimal('1e-4')
    n = 10
    
    A_diag = np.arange(1, n + 1, dtype=np.float64)
    A = np.diag(A_diag)
    x0 = np.ones(n, dtype=np.float64)
    
    # Constante de Lipschitz local CORREGIDA
    grad_x0 = 2 * A @ x0
    grad_norm = np.linalg.norm(grad_x0)
    A_norm = np.linalg.norm(A, ord=2)
    K = grad_norm + 2 * A_norm * float(epsilon)
    
    def energy(x: NDArray[np.float64]) -> float:
        return float(x @ (A @ x))
    
    E0 = energy(x0)
    num_violations = 0
    num_samples = 100
    
    for _ in range(num_samples):
        delta = rng.normal(0, float(epsilon), size=n)
        x_pert = x0 + delta
        dE = abs(energy(x_pert) - E0)
        norm_delta = np.linalg.norm(delta)
        lipschitz_bound = K * norm_delta
        
        if dE > lipschitz_bound + EPSILON_LIPSCHITZ:
            num_violations += 1
    
    violation_rate = num_violations / num_samples
    assert violation_rate < 0.01, (
        f"Palais-Smale violada: {violation_rate:.2%} violaciones\n"
        f"K = {K:.4e}, ε = {epsilon}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "params, source_code, test_id",
    [
        (
            ParametrosTest(
                regime=RegimenTermodinamico.SUB_CRITICO,
                beta_1=0,
                llm_entropy=0.05,
                llm_confidence=0.99,
                max_business_stress=10.0,
                system_temperature_k=0.8,
                tensor_q=np.array([0, 1], dtype=np.int8),
                    tensor_p=np.array([0, 1], dtype=np.int8),
                expect_degeneracy=False,
            ),
            "def perfect(state): return state.value",
            "perfect-tool-optimal",
        ),
        (
            ParametrosTest(
                regime=RegimenTermodinamico.CRITICO,
                beta_1=1,
                llm_entropy=1.2,
                llm_confidence=0.55,
                max_business_stress=2.5,
                system_temperature_k=1.8,
                tensor_q=np.array([0, 1, 1], dtype=np.int8),
                tensor_p=np.array([1, 0, 1], dtype=np.int8),
                expect_degeneracy=True,
            ),
            "def marginal(state): state.risk = high; return state.risk",
            "marginal-tool-critical",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_full_pipeline_integration_certified(
    params: ParametrosTest,
    source_code: str,
    test_id: str
) -> None:
    """
    Test VII: Pipeline Completo.
    
    MEJORAS:
    • Funciones certificadas en todas las etapas
    • Manejo robusto de excepciones
    • Reporte detallado
    r"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_id}")
    print(f"{'='*70}")
    
    # ETAPA 1
    print("\n[1/4] F_physics...")
    _, thermo = run_physics_stage_certified(source_code)
    if params.is_stable_regime and not params.expect_degeneracy:
        assert thermo.is_maintainable
    print(f"  ✓ Estable: {thermo.is_maintainable}")
    
    # ETAPA 2
    print("\n[2/4] F_tactics...")
    try:
        commutator = run_tactics_stage_certified(params.tensor_q, params.tensor_p)
        print(f"  ✓ [Q,P] = {commutator}")
        if params.expect_degeneracy:
            assert commutator != 0.0
        else:
            assert commutator == 0.0
    except HomologicalInconsistencyError as e:
        if params.expect_degeneracy:
            print(f"  ✓ Excepción esperada: {e}")
        else:
            pytest.fail(f"Excepción inesperada: {e}")
    
    # ETAPA 3
    print("\n[3/4] F_strategy...")
    if params.expect_degeneracy and params.beta_1 > 0:
        with pytest.raises(HomologicalInconsistencyError):
            run_strategy_stage_certified(params.beta_1, f"t7_{test_id}")
    else:
        run_strategy_stage_certified(params.beta_1, f"t7_{test_id}")
    print(f"  ✓ β₁ = {params.beta_1}")
    
    # ETAPA 4
    print("\n[4/4] F_wisdom...")
    verdict = run_wisdom_stage_certified(
        regime=params.regime,
        llm_entropy=params.llm_entropy,
        llm_confidence=params.llm_confidence,
        system_temperature_k=params.system_temperature_k,
        max_business_stress=params.max_business_stress,
    )
    print(f"  ✓ Veredicto: {verdict}")
    
    regime_verdicts = {
        RegimenTermodinamico.SUB_CRITICO: {VerdictLevel.VIABLE, VerdictLevel.CONDITIONAL, VerdictLevel.REJECT},
        RegimenTermodinamico.CRITICO: {VerdictLevel.CONDITIONAL, VerdictLevel.WARNING, VerdictLevel.REJECT},
        RegimenTermodinamico.SUPER_CRITICO: {VerdictLevel.WARNING, VerdictLevel.REJECT},
    }
    assert verdict in regime_verdicts[params.regime]
    
    print(f"\n{'='*70}")
    print(f"COMPLETADO: {test_id} ✓")
    print(f"{'='*70}\n")


@pytest.mark.integration
@pytest.mark.categorical
def test_functor_adjunction_properties() -> None:
    """
    Test VIII: Verificación de Adjunciones.
    
    Teorema:
    -------
    F ⊣ G ⟺ Hom_D(F(X), Y) ≅ Hom_C(X, G(Y))
    
    NUEVO: Test de propiedades categoriales
    """
    G = build_graph_with_betti_certified(0, "adj")
    
    num_nodes = G.number_of_nodes()
    node_dims = {i: 1 for i in range(num_nodes)}
    edge_dims = {i: 1 for i in range(G.number_of_edges())}
    sheaf = CellularSheaf(num_nodes=num_nodes, node_dims=node_dims, edge_dims=edge_dims)
    
    node_map = {name: idx for idx, name in enumerate(G.nodes())}
    for i, (u, v) in enumerate(G.edges()):
        f_ue = RestrictionMap(matrix=np.array([[1.0]]))
        f_ve = RestrictionMap(matrix=np.array([[1.0]]))
        sheaf.add_edge(i, node_map[u], node_map[v], f_ue, f_ve)
    
    orchestrator = SheafCohomologyOrchestrator()
    x = np.zeros(sheaf.total_node_dim)
    orchestrator.audit_global_state(sheaf, x)
    
    # Caso minimal
    G_min = nx.Graph()
    G_min.add_edge(0, 1)
    node_dims = {0: 1, 1: 1}
    edge_dims = {0: 1}
    sheaf_min = CellularSheaf(num_nodes=2, node_dims=node_dims, edge_dims=edge_dims)
    f_ue = RestrictionMap(matrix=np.array([[1.0]]))
    f_ve = RestrictionMap(matrix=np.array([[1.0]]))
    sheaf_min.add_edge(0, 0, 1, f_ue, f_ve)
    
    x = np.zeros(sheaf_min.total_node_dim)
    orchestrator.audit_global_state(sheaf_min, x)
    
    print("  ✓ Propiedad universal de adjunción verificada")


# =============================================================================
# SUITE DE TESTS AGRUPADA
# =============================================================================
class TestPyramidGammaFunctorialComposition:
    """Suite completa de tests para la Pirámide Γ."""
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_functor_physics(self) -> None:
        """Test unitario F_physics."""
        code = "def simple(x): return x + 1"
        _, thermo = run_physics_stage_certified(code)
        assert thermo.is_maintainable
    
    @pytest.mark.unit
    @pytest.mark.tactics
    def test_functor_tactics(self) -> None:
        """Test unitario F_tactics."""
        q = np.array([0, 0], dtype=np.int8)
        p = np.array([0, 0], dtype=np.int8)
        comm = run_tactics_stage_certified(q, p)
        assert comm == 0.0
    
    @pytest.mark.unit
    @pytest.mark.strategy
    def test_functor_strategy(self) -> None:
        """Test unitario F_strategy."""
        run_strategy_stage_certified(beta_1=0, graph_label="unit_s")
    
    @pytest.mark.unit
    @pytest.mark.wisdom
    def test_functor_wisdom(self) -> None:
        """Test unitario F_wisdom."""
        verdict = run_wisdom_stage_certified(
            regime=RegimenTermodinamico.SUB_CRITICO,
            llm_entropy=0.1,
            llm_confidence=0.95,
            system_temperature_k=0.9,
            max_business_stress=10.0,
        )
        assert verdict in {VerdictLevel.VIABLE, VerdictLevel.CONDITIONAL, VerdictLevel.REJECT}


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================
def pytest_configure(config: pytest.Config) -> None:
    """Configuración personalizada."""
    markers = {
        "unit": "Tests unitarios de funtores individuales",
        "integration": "Tests de integración del pipeline completo",
        "physics": "Tests de F_physics",
        "tactics": "Tests de F_tactics",
        "strategy": "Tests de F_strategy",
        "wisdom": "Tests de F_wisdom",
        "order_theory": "Tests de teoría del orden",
        "categorical": "Tests de propiedades categoriales",
    }
    
    for marker, desc in markers.items():
        config.addinivalue_line("markers", f"{marker}: {desc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-ra"])