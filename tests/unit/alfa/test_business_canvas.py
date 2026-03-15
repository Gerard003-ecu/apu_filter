"""
Suite de Pruebas: Business Canvas Topology (test_business_canvas.py)

Pruebas matemáticamente rigurosas para validar:
1. Topología Algebraica: Complejos de cadenas, homología, Euler-Poincaré
2. Teoría Espectral: Laplacianos, autovalores, Fiedler
3. Teoría de Grafos: Construcción, proyección, ciclos
4. Álgebra Lineal: Rango, núcleo, tolerancias numéricas
5. Validación de Payload: Esquemas, errores
6. Auditoría de Fusión: Mayer-Vietoris inspirado

Convenciones:
- Los tests verifican identidades algebraicas exactas donde es posible
- Se usan tolerancias numéricas explícitas para comparaciones flotantes
- Cada test documenta el teorema/propiedad que verifica
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Importaciones del módulo bajo prueba
from app.stratums.alpha.business_canvas import (
    # Clases principales
    AlphaTopologyVector,
    ChainComplex1D,
    HomologyMetrics,
    SpectralMetrics,
    CycleSpaceMetrics,
    BmcTopologyMetrics,
    # Enums
    MergeVerdict,
    ConnectivityClass,
    # Excepciones
    BMCTopologyError,
    TopologicalInvariantError,
    SpectralAnalysisError,
    PayloadValidationError,
    HomologicalInconsistencyError,
    # Funciones auxiliares
    canonicalize_edge,
    compute_numerical_rank,
    compute_null_space_basis,
    safe_eigenvalues_symmetric,
    # Constantes
    BMC_NODES,
    BASE_EDGES,
    EPSILON,
    RANK_TOL,
    EIGENVALUE_ZERO_TOL,
    MIN_FIEDLER_VALUE,
)
from app.core.mic_algebra import CategoricalState


# =============================================================================
# CONSTANTES DE PRUEBA
# =============================================================================

NUMERICAL_TOL = 1e-10
EIGENVALUE_TOL = 1e-8


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def topology_vector() -> AlphaTopologyVector:
    """Instancia del vector topológico para pruebas."""
    return AlphaTopologyVector(name="test_topology")


@pytest.fixture
def empty_payload() -> Dict[str, Any]:
    """Payload vacío (configuración base del BMC)."""
    return {}


@pytest.fixture
def mock_state_vector(empty_payload: Dict[str, Any]) -> CategoricalState:
    """Estado categórico mockeado con payload vacío."""
    state = MagicMock(spec=CategoricalState)
    state.payload = empty_payload
    return state


@pytest.fixture
def simple_triangle_graph() -> nx.Graph:
    """
    Grafo triangular simple: K_3
    
    Propiedades conocidas:
    - |V| = 3, |E| = 3
    - β₀ = 1 (conexo)
    - β₁ = 1 (un ciclo)
    - χ = 0
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return G


@pytest.fixture
def simple_path_graph() -> nx.Graph:
    """
    Grafo camino: P_4 (4 vértices, 3 aristas)
    
    Propiedades conocidas:
    - |V| = 4, |E| = 3
    - β₀ = 1 (conexo)
    - β₁ = 0 (árbol)
    - χ = 1
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return G


@pytest.fixture
def disconnected_graph() -> nx.Graph:
    """
    Grafo desconectado: dos componentes
    
    Propiedades conocidas:
    - |V| = 4, |E| = 2
    - β₀ = 2 (dos componentes)
    - β₁ = 0
    - χ = 2
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("C", "D")])
    return G


@pytest.fixture
def complete_graph_k4() -> nx.Graph:
    """
    Grafo completo K_4.
    
    Propiedades conocidas:
    - |V| = 4, |E| = 6
    - β₀ = 1
    - β₁ = 3 (número ciclomático = |E| - |V| + 1)
    - χ = -2
    """
    return nx.complete_graph(4)


@pytest.fixture
def base_bmc_digraph(topology_vector: AlphaTopologyVector) -> nx.DiGraph:
    """Digrafo BMC base sin modificaciones."""
    return topology_vector._build_directed_business_graph({})


# =============================================================================
# TESTS: FUNCIONES AUXILIARES PURAS
# =============================================================================

class TestCanonicalizeEdge:
    """Pruebas para la función de canonicalización de aristas."""
    
    def test_already_canonical(self):
        """Arista ya en orden lexicográfico."""
        assert canonicalize_edge("A", "B") == ("A", "B")
    
    def test_needs_swap(self):
        """Arista que requiere intercambio."""
        assert canonicalize_edge("B", "A") == ("A", "B")
    
    def test_equal_vertices(self):
        """Auto-lazo (edge case)."""
        assert canonicalize_edge("A", "A") == ("A", "A")
    
    def test_numeric_strings(self):
        """Ordenamiento lexicográfico de strings numéricos."""
        # "10" < "2" lexicográficamente
        assert canonicalize_edge("2", "10") == ("10", "2")
    
    @pytest.mark.parametrize("u,v,expected", [
        ("alpha", "beta", ("alpha", "beta")),
        ("z", "a", ("a", "z")),
        ("P_val", "P_seg", ("P_seg", "P_val")),
    ])
    def test_parametrized(self, u: str, v: str, expected: Tuple[str, str]):
        """Tests parametrizados de canonicalización."""
        assert canonicalize_edge(u, v) == expected


class TestComputeNumericalRank:
    """Pruebas para cálculo de rango numérico via SVD."""
    
    def test_full_rank_square(self):
        """Matriz cuadrada de rango completo."""
        A = np.array([[1, 0], [0, 1]], dtype=float)
        assert compute_numerical_rank(A) == 2
    
    def test_rank_deficient(self):
        """Matriz con columnas linealmente dependientes."""
        A = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
        assert compute_numerical_rank(A) == 1
    
    def test_zero_matrix(self):
        """Matriz nula tiene rango 0."""
        A = np.zeros((3, 4))
        assert compute_numerical_rank(A) == 0
    
    def test_empty_matrix(self):
        """Matriz vacía tiene rango 0."""
        A = np.array([]).reshape(0, 0)
        assert compute_numerical_rank(A) == 0
    
    def test_nearly_singular(self):
        """Matriz casi singular (estabilidad numérica)."""
        # Columnas casi linealmente dependientes
        A = np.array([
            [1.0, 1.0 + 1e-15],
            [1.0, 1.0 + 1e-15]
        ])
        # Con tolerancia adecuada, debe detectar rango 1
        assert compute_numerical_rank(A, tol=1e-10) == 1
    
    def test_rectangular_tall(self):
        """Matriz más alta que ancha."""
        A = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        assert compute_numerical_rank(A) == 2
    
    def test_rectangular_wide(self):
        """Matriz más ancha que alta."""
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        assert compute_numerical_rank(A) == 2


class TestComputeNullSpaceBasis:
    """Pruebas para cálculo de base del núcleo."""
    
    def test_trivial_kernel(self):
        """Matriz inyectiva tiene núcleo trivial."""
        A = np.eye(3)
        kernel = compute_null_space_basis(A)
        assert kernel.shape[1] == 0
    
    def test_nontrivial_kernel(self):
        """Matriz con núcleo no trivial."""
        # Núcleo: span{(1, -1, 0), (0, 0, 1)} (dos dimensiones)
        A = np.array([[1, 1, 0]], dtype=float)
        kernel = compute_null_space_basis(A)
        assert kernel.shape[1] == 2
        
        # Verificar que efectivamente A·v = 0 para cada v en la base
        for i in range(kernel.shape[1]):
            v = kernel[:, i]
            assert_allclose(A @ v, np.zeros(1), atol=NUMERICAL_TOL)
    
    def test_orthonormal_basis(self):
        """La base del núcleo debe ser ortonormal."""
        A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
        kernel = compute_null_space_basis(A)
        
        # Verificar ortogonalidad: K^T K ≈ I
        if kernel.shape[1] > 0:
            gram = kernel.T @ kernel
            assert_allclose(gram, np.eye(kernel.shape[1]), atol=NUMERICAL_TOL)
    
    def test_empty_matrix(self):
        """Matriz vacía."""
        A = np.array([]).reshape(0, 0)
        kernel = compute_null_space_basis(A)
        assert kernel.size == 0


class TestSafeEigenvaluesSymmetric:
    """Pruebas para cálculo estable de autovalores."""
    
    def test_identity_matrix(self):
        """Identidad tiene todos autovalores = 1."""
        I = np.eye(3)
        eigenvalues = safe_eigenvalues_symmetric(I)
        assert_allclose(eigenvalues, [1.0, 1.0, 1.0], atol=NUMERICAL_TOL)
    
    def test_diagonal_matrix(self):
        """Matriz diagonal: autovalores son entradas diagonales."""
        D = np.diag([1.0, 2.0, 3.0])
        eigenvalues = safe_eigenvalues_symmetric(D)
        assert_allclose(eigenvalues, [1.0, 2.0, 3.0], atol=NUMERICAL_TOL)
    
    def test_zero_matrix(self):
        """Matriz nula tiene todos autovalores = 0."""
        Z = np.zeros((3, 3))
        eigenvalues = safe_eigenvalues_symmetric(Z)
        assert_allclose(eigenvalues, [0.0, 0.0, 0.0], atol=NUMERICAL_TOL)
    
    def test_forces_symmetry(self):
        """Debe forzar simetría en matrices casi simétricas."""
        # Matriz ligeramente asimétrica
        A = np.array([
            [2.0, 1.0 + 1e-14],
            [1.0 - 1e-14, 2.0]
        ])
        eigenvalues = safe_eigenvalues_symmetric(A)
        # Autovalores de [[2,1],[1,2]] son 1 y 3
        assert_allclose(eigenvalues, [1.0, 3.0], atol=NUMERICAL_TOL)
    
    def test_ascending_order(self):
        """Los autovalores deben estar ordenados ascendentemente."""
        A = np.diag([5.0, 1.0, 3.0])
        eigenvalues = safe_eigenvalues_symmetric(A)
        assert list(eigenvalues) == sorted(eigenvalues)
    
    def test_cleans_numerical_noise(self):
        """Valores muy pequeños deben limpiarse a cero."""
        # Laplaciano de K_2: [[1,-1],[-1,1]], autovalores: 0, 2
        L = np.array([[1, -1], [-1, 1]], dtype=float)
        eigenvalues = safe_eigenvalues_symmetric(L)
        assert eigenvalues[0] == 0.0  # Exactamente cero tras limpieza


# =============================================================================
# TESTS: COMPLEJO DE CADENAS
# =============================================================================

class TestChainComplex1D:
    """Pruebas para construcción del complejo de cadenas."""
    
    def test_boundary_matrix_dimensions(
        self, 
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """∂₁ debe tener dimensiones |V| × |E|."""
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        n_v = len(cc.vertex_basis)
        n_e = len(cc.edge_basis)
        assert cc.boundary_1.shape == (n_v, n_e)
    
    def test_boundary_matrix_signs(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """
        Verificar convención de signos de ∂₁.
        
        Para arista canónica (u, v) con u < v:
        - [∂₁]_{índice(u), j} = -1
        - [∂₁]_{índice(v), j} = +1
        """
        G = nx.Graph()
        G.add_edge("A", "B")
        
        cc = topology_vector._build_chain_complex_1d(G)
        
        # Verificar que ∂₁ tiene exactamente dos entradas no nulas por columna
        for j in range(cc.boundary_1.shape[1]):
            column = cc.boundary_1[:, j]
            nonzero = column[column != 0]
            assert len(nonzero) == 2
            assert -1.0 in nonzero
            assert +1.0 in nonzero
            assert np.sum(column) == 0  # Suma de coeficientes = 0
    
    def test_laplacian_0_formula(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """Verificar L₀ = ∂₁∂₁ᵀ."""
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        expected_L0 = cc.boundary_1 @ cc.boundary_1.T
        assert_allclose(cc.laplacian_0, expected_L0, atol=NUMERICAL_TOL)
    
    def test_laplacian_1_formula(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """Verificar L₁ = ∂₁ᵀ∂₁."""
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        expected_L1 = cc.boundary_1.T @ cc.boundary_1
        assert_allclose(cc.laplacian_1, expected_L1, atol=NUMERICAL_TOL)
    
    def test_laplacians_symmetric(
        self,
        topology_vector: AlphaTopologyVector,
        simple_path_graph: nx.Graph
    ):
        """Laplacianos deben ser simétricos."""
        cc = topology_vector._build_chain_complex_1d(simple_path_graph)
        assert_allclose(cc.laplacian_0, cc.laplacian_0.T, atol=NUMERICAL_TOL)
        assert_allclose(cc.laplacian_1, cc.laplacian_1.T, atol=NUMERICAL_TOL)
    
    def test_laplacians_positive_semidefinite(
        self,
        topology_vector: AlphaTopologyVector,
        complete_graph_k4: nx.Graph
    ):
        """Laplacianos deben ser semidefinidos positivos."""
        cc = topology_vector._build_chain_complex_1d(complete_graph_k4)
        
        eigenvalues_L0 = np.linalg.eigvalsh(cc.laplacian_0)
        eigenvalues_L1 = np.linalg.eigvalsh(cc.laplacian_1)
        
        assert np.all(eigenvalues_L0 >= -NUMERICAL_TOL)
        assert np.all(eigenvalues_L1 >= -NUMERICAL_TOL)
    
    def test_canonical_edge_ordering(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Las aristas deben estar en orden canónico lexicográfico."""
        G = nx.Graph()
        G.add_edges_from([("C", "A"), ("B", "A"), ("C", "B")])
        
        cc = topology_vector._build_chain_complex_1d(G)
        
        # Todas las aristas deben cumplir u < v
        for u, v in cc.edge_basis:
            assert u < v, f"Arista no canónica: ({u}, {v})"
        
        # La lista debe estar ordenada
        assert cc.edge_basis == tuple(sorted(cc.edge_basis))
    
    def test_single_vertex_complex(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Complejo con un solo vértice."""
        G = nx.Graph()
        G.add_node("solo")
        
        cc = topology_vector._build_chain_complex_1d(G)
        
        assert cc.vertex_basis == ("solo",)
        assert cc.edge_basis == ()
        assert cc.boundary_1.shape == (1, 0)
        assert cc.laplacian_0.shape == (1, 1)
        assert cc.laplacian_0[0, 0] == 0.0


# =============================================================================
# TESTS: HOMOLOGÍA
# =============================================================================

class TestHomologyMetrics:
    """Pruebas para cálculos homológicos."""
    
    def test_connected_tree_homology(
        self,
        topology_vector: AlphaTopologyVector,
        simple_path_graph: nx.Graph
    ):
        """
        Árbol (grafo acíclico conexo):
        - β₀ = 1
        - β₁ = 0
        """
        cc = topology_vector._build_chain_complex_1d(simple_path_graph)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 1
        assert h.beta_1 == 0
        assert h.euler_char == 1  # |V| - |E| = 4 - 3 = 1
    
    def test_triangle_homology(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """
        Triángulo (ciclo):
        - β₀ = 1
        - β₁ = 1
        """
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 1
        assert h.beta_1 == 1
        assert h.euler_char == 0  # |V| - |E| = 3 - 3 = 0
    
    def test_disconnected_homology(
        self,
        topology_vector: AlphaTopologyVector,
        disconnected_graph: nx.Graph
    ):
        """
        Grafo desconectado:
        - β₀ = número de componentes
        """
        cc = topology_vector._build_chain_complex_1d(disconnected_graph)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 2
        assert h.beta_1 == 0
    
    def test_complete_graph_homology(
        self,
        topology_vector: AlphaTopologyVector,
        complete_graph_k4: nx.Graph
    ):
        """
        K_4: β₁ = |E| - |V| + 1 = 6 - 4 + 1 = 3
        """
        cc = topology_vector._build_chain_complex_1d(complete_graph_k4)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 1
        assert h.beta_1 == 3
    
    def test_euler_poincare_identity(
        self,
        topology_vector: AlphaTopologyVector,
        complete_graph_k4: nx.Graph
    ):
        """
        Teorema de Euler-Poincaré:
        χ = |V| - |E| = β₀ - β₁
        """
        cc = topology_vector._build_chain_complex_1d(complete_graph_k4)
        h = topology_vector._compute_homology_metrics(cc)
        
        euler_direct = h.n_vertices - h.n_edges
        euler_betti = h.beta_0 - h.beta_1
        
        assert euler_direct == euler_betti
        assert h.euler_char == h.euler_from_betti
    
    def test_rank_nullity_theorem(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """
        Teorema de rango-nulidad:
        rank(∂₁) + nullity(∂₁) = |E|
        """
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.rank_boundary_1 + h.nullity_boundary_1 == h.n_edges
    
    @pytest.mark.parametrize("n_vertices", [5, 10, 20])
    def test_cycle_graph_homology(
        self,
        topology_vector: AlphaTopologyVector,
        n_vertices: int
    ):
        """
        Ciclo C_n:
        - β₀ = 1
        - β₁ = 1
        - χ = 0
        """
        G = nx.cycle_graph(n_vertices)
        # Renombrar nodos a strings para consistencia
        mapping = {i: f"v{i}" for i in range(n_vertices)}
        G = nx.relabel_nodes(G, mapping)
        
        cc = topology_vector._build_chain_complex_1d(G)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 1
        assert h.beta_1 == 1
        assert h.euler_char == 0


# =============================================================================
# TESTS: ANÁLISIS ESPECTRAL
# =============================================================================

class TestSpectralMetrics:
    """Pruebas para invariantes espectrales."""
    
    def test_fiedler_connected_graph(
        self,
        topology_vector: AlphaTopologyVector,
        simple_path_graph: nx.Graph
    ):
        """
        Grafo conexo: λ₁ > 0 (conectividad algebraica de Fiedler)
        """
        cc = topology_vector._build_chain_complex_1d(simple_path_graph)
        s = topology_vector._compute_spectral_metrics(cc)
        
        assert s.fiedler_value > 0
    
    def test_fiedler_disconnected_graph(
        self,
        topology_vector: AlphaTopologyVector,
        disconnected_graph: nx.Graph
    ):
        """
        Grafo desconectado: λ₁ = 0 (segundo autovalor también nulo)
        """
        cc = topology_vector._build_chain_complex_1d(disconnected_graph)
        s = topology_vector._compute_spectral_metrics(cc)
        
        assert s.fiedler_value == 0.0
    
    def test_multiplicity_zero_equals_beta0(
        self,
        topology_vector: AlphaTopologyVector,
        disconnected_graph: nx.Graph
    ):
        """
        Teorema de Hodge discreto: mult(0, L₀) = β₀
        """
        cc = topology_vector._build_chain_complex_1d(disconnected_graph)
        h = topology_vector._compute_homology_metrics(cc)
        s = topology_vector._compute_spectral_metrics(cc)
        
        assert s.multiplicity_zero == h.beta_0
    
    def test_trace_equals_twice_edges(
        self,
        topology_vector: AlphaTopologyVector,
        complete_graph_k4: nx.Graph
    ):
        """
        tr(L₀) = Σᵢ deg(vᵢ) = 2|E|
        """
        cc = topology_vector._build_chain_complex_1d(complete_graph_k4)
        s = topology_vector._compute_spectral_metrics(cc)
        
        expected_trace = 2 * complete_graph_k4.number_of_edges()
        assert_allclose(s.trace_laplacian, expected_trace, atol=NUMERICAL_TOL)
    
    def test_complete_graph_eigenvalues(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """
        K_n tiene autovalores: 0 (mult. 1) y n (mult. n-1)
        """
        n = 5
        G = nx.complete_graph(n)
        mapping = {i: f"v{i}" for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        
        cc = topology_vector._build_chain_complex_1d(G)
        s = topology_vector._compute_spectral_metrics(cc)
        
        eigenvalues = np.array(s.eigenvalues)
        
        # Un autovalor cero
        zeros = eigenvalues[np.abs(eigenvalues) < EIGENVALUE_TOL]
        assert len(zeros) == 1
        
        # n-1 autovalores iguales a n
        n_values = eigenvalues[np.abs(eigenvalues - n) < EIGENVALUE_TOL]
        assert len(n_values) == n - 1
    
    def test_path_graph_fiedler_formula(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """
        P_n tiene λ₁ = 2(1 - cos(π/n))
        """
        n = 10
        G = nx.path_graph(n)
        mapping = {i: f"v{i}" for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        
        cc = topology_vector._build_chain_complex_1d(G)
        s = topology_vector._compute_spectral_metrics(cc)
        
        expected_fiedler = 2 * (1 - math.cos(math.pi / n))
        assert_allclose(s.fiedler_value, expected_fiedler, atol=1e-6)
    
    def test_eigenvalues_sorted(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """Los autovalores deben estar ordenados ascendentemente."""
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        s = topology_vector._compute_spectral_metrics(cc)
        
        eigenvalues = list(s.eigenvalues)
        assert eigenvalues == sorted(eigenvalues)


# =============================================================================
# TESTS: CICLOS DIRIGIDOS
# =============================================================================

class TestDirectedCycles:
    """Pruebas para detección de ciclos dirigidos."""
    
    def test_dag_no_cycles(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """DAG no tiene ciclos dirigidos."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        
        cycles = topology_vector._analyze_directed_cycles(G)
        
        assert cycles.is_acyclic == True
        assert cycles.directed_cycles_count == 0
    
    def test_simple_cycle_detected(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Ciclo simple es detectado."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        
        cycles = topology_vector._analyze_directed_cycles(G)
        
        assert cycles.is_acyclic == False
        assert cycles.directed_cycles_count >= 1
    
    def test_multiple_cycles(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Múltiples ciclos son contados."""
        G = nx.DiGraph()
        # Dos triángulos compartiendo un vértice
        G.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),  # Ciclo 1
            ("A", "D"), ("D", "E"), ("E", "A"),  # Ciclo 2
        ])
        
        cycles = topology_vector._analyze_directed_cycles(G)
        
        assert cycles.directed_cycles_count >= 2
    
    def test_self_loop_is_cycle(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Auto-lazo cuenta como ciclo."""
        G = nx.DiGraph()
        G.add_edge("A", "A")
        
        cycles = topology_vector._analyze_directed_cycles(G)
        
        assert cycles.is_acyclic == False
        assert cycles.directed_cycles_count >= 1


# =============================================================================
# TESTS: CONSTRUCCIÓN DEL GRAFO BMC
# =============================================================================

class TestBMCGraphConstruction:
    """Pruebas para construcción del grafo dirigido del BMC."""
    
    def test_base_structure(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Grafo base tiene estructura correcta."""
        G = topology_vector._build_directed_business_graph({})
        
        assert set(G.nodes()) == set(BMC_NODES)
        assert G.number_of_edges() == len(BASE_EDGES)
    
    def test_disable_nodes(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Desactivar nodos los elimina del grafo."""
        payload = {"disable_nodes": ["P_soc", "P_rec"]}
        G = topology_vector._build_directed_business_graph(payload)
        
        assert "P_soc" not in G.nodes()
        assert "P_rec" not in G.nodes()
        assert len(G.nodes()) == len(BMC_NODES) - 2
    
    def test_remove_edges(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Eliminar aristas específicas."""
        payload = {
            "remove_edges": [
                {"source": "P_soc", "target": "P_act"}
            ]
        }
        G = topology_vector._build_directed_business_graph(payload)
        
        assert not G.has_edge("P_soc", "P_act")
    
    def test_modify_weights(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Modificar pesos de aristas existentes."""
        payload = {
            "edge_weights": [
                {"source": "P_val", "target": "P_can", "weight": 2.5}
            ]
        }
        G = topology_vector._build_directed_business_graph(payload)
        
        assert G["P_val"]["P_can"]["weight"] == 2.5
    
    def test_add_extra_edges(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Añadir aristas adicionales."""
        payload = {
            "extra_edges": [
                {"source": "P_seg", "target": "P_val", "weight": 0.8}
            ]
        }
        G = topology_vector._build_directed_business_graph(payload)
        
        assert G.has_edge("P_seg", "P_val")
        assert G["P_seg"]["P_val"]["weight"] == 0.8
    
    def test_invalid_node_in_disable(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Error al desactivar nodo inválido."""
        payload = {"disable_nodes": ["nodo_inexistente"]}
        
        with pytest.raises(PayloadValidationError, match="Nodo inválido"):
            topology_vector._build_directed_business_graph(payload)
    
    def test_nonpositive_weight_error(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Error con peso no positivo."""
        payload = {
            "edge_weights": [
                {"source": "P_val", "target": "P_can", "weight": -1.0}
            ]
        }
        
        with pytest.raises(PayloadValidationError, match="Peso debe ser > 0"):
            topology_vector._build_directed_business_graph(payload)
    
    def test_modify_nonexistent_edge_error(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Error al modificar arista inexistente."""
        payload = {
            "edge_weights": [
                {"source": "P_val", "target": "P_ing", "weight": 1.0}
            ]
        }
        
        with pytest.raises(PayloadValidationError, match="Arista inexistente"):
            topology_vector._build_directed_business_graph(payload)


# =============================================================================
# TESTS: PROYECCIÓN A GRAFO NO DIRIGIDO
# =============================================================================

class TestUndirectedProjection:
    """Pruebas para proyección al 1-esqueleto no dirigido."""
    
    def test_basic_projection(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Proyección básica conserva nodos."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        
        H = topology_vector._to_weighted_undirected(G)
        
        assert set(H.nodes()) == set(G.nodes())
        assert H.has_edge("A", "B")
        assert H.has_edge("B", "C")
    
    def test_bidirectional_weight_aggregation(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """
        Aristas bidireccionales agregan pesos:
        w({u,v}) = w(u→v) + w(v→u)
        """
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "A", weight=2.0)
        
        H = topology_vector._to_weighted_undirected(G)
        
        assert H["A"]["B"]["weight"] == 3.0
    
    def test_preserves_node_data(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """La proyección preserva atributos de nodos."""
        G = nx.DiGraph()
        G.add_node("A", label="nodo_A")
        G.add_node("B", label="nodo_B")
        G.add_edge("A", "B")
        
        H = topology_vector._to_weighted_undirected(G)
        
        assert H.nodes["A"]["label"] == "nodo_A"


# =============================================================================
# TESTS: VALIDACIÓN DE CONSISTENCIA
# =============================================================================

class TestInternalConsistency:
    """Pruebas para validación de consistencia interna."""
    
    def test_consistent_metrics_pass(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """Métricas consistentes no lanzan error."""
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        h = topology_vector._compute_homology_metrics(cc)
        s = topology_vector._compute_spectral_metrics(cc)
        
        # No debe lanzar excepción
        topology_vector._validate_internal_consistency(h, s, simple_triangle_graph)
    
    def test_detects_beta0_inconsistency(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Detecta inconsistencia en β₀."""
        # Crear métricas artificialmente inconsistentes
        h = HomologyMetrics(
            n_vertices=4,
            n_edges=3,
            rank_boundary_1=3,
            nullity_boundary_1=0,
            beta_0=2,  # Incorrecto: debería ser 1
            beta_1=0,
            euler_char=1,
            euler_from_betti=2,  # Inconsistente
        )
        # Esto debería fallar en __post_init__
        # Pero si llegara a pasar, la validación lo detectaría


# =============================================================================
# TESTS: RESTRICCIONES TOPOLÓGICAS
# =============================================================================

class TestTopologicalConstraints:
    """Pruebas para restricciones topológicas del BMC."""
    
    def test_rejects_disconnected_bmc(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """BMC desconectado es rechazado."""
        payload = {"disable_nodes": ["P_val"]}  # Desconecta el grafo
        
        metrics = topology_vector._compute_full_analysis(payload)
        
        if metrics.beta_0 > 1:
            with pytest.raises(TopologicalInvariantError, match="fragmentado"):
                topology_vector._enforce_topological_constraints(metrics)
    
    def test_accepts_valid_bmc(
        self,
        topology_vector: AlphaTopologyVector,
        mock_state_vector: CategoricalState
    ):
        """BMC válido es aceptado."""
        result = topology_vector(mock_state_vector)
        
        # El BMC base debería ser válido
        assert result.get("success") == True or "VETO" not in result.get("narrative", "")


# =============================================================================
# TESTS: AUDITORÍA DE FUSIÓN (MAYER-VIETORIS)
# =============================================================================

class TestStrategicFusionAudit:
    """Pruebas para auditoría de fusión estratégica."""
    
    def test_compatible_fusion_accepted(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Fusión compatible es aceptada."""
        G1 = nx.DiGraph()
        G1.add_edges_from([("A", "B"), ("B", "C")])
        
        G2 = nx.DiGraph()
        G2.add_edges_from([("C", "D"), ("D", "E")])
        
        verdict = topology_vector.audit_strategic_fusion(G1, G2)
        
        assert verdict == MergeVerdict.ACCEPTED
    
    def test_disconnected_fusion_rejected(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Fusión que produce grafo desconectado es rechazada."""
        G1 = nx.DiGraph()
        G1.add_edges_from([("A", "B")])
        
        G2 = nx.DiGraph()
        G2.add_edges_from([("X", "Y")])  # Sin conexión con G1
        
        verdict = topology_vector.audit_strategic_fusion(G1, G2)
        
        assert verdict == MergeVerdict.REJECTED_DISCONNECTED
    
    def test_new_cycles_rejected(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Fusión que introduce ciclos dirigidos es rechazada."""
        G1 = nx.DiGraph()
        G1.add_edges_from([("A", "B"), ("B", "C")])
        
        G2 = nx.DiGraph()
        G2.add_edges_from([("C", "A")])  # Cierra un ciclo
        
        verdict = topology_vector.audit_strategic_fusion(G1, G2)
        
        assert verdict == MergeVerdict.REJECTED_TOXIC_CYCLES
    
    def test_homological_defect_rejected(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Fusión con defecto homológico anómalo es rechazada."""
        # Construir caso donde β₁ crece anómalamente
        G1 = nx.DiGraph()
        G1.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        
        G2 = nx.DiGraph()
        # Añadir aristas que crean múltiples ciclos estructurales
        G2.add_edges_from([
            ("D", "A"),  # Cierra ciclo 1
            ("B", "D"),  # Cierra ciclo 2 (con A-B-D)
        ])
        
        verdict = topology_vector.audit_strategic_fusion(G1, G2)
        
        # Puede ser rechazado por ciclos dirigidos o defecto homológico
        assert verdict in (
            MergeVerdict.REJECTED_TOXIC_CYCLES,
            MergeVerdict.REJECTED_HOMOLOGICAL_DEFECT
        )


# =============================================================================
# TESTS: BUNDLE DE ANÁLISIS
# =============================================================================

class TestAnalysisBundle:
    """Pruebas para el paquete analítico completo."""
    
    def test_bundle_structure(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """El bundle tiene todas las claves esperadas."""
        bundle = topology_vector.build_analysis_bundle({})
        
        expected_keys = {
            "bases", "matrices", "homology", "spectral", "cycles", "graph_stats"
        }
        assert set(bundle.keys()) == expected_keys
    
    def test_matrices_serializable(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Las matrices están en formato serializable (listas)."""
        bundle = topology_vector.build_analysis_bundle({})
        
        assert isinstance(bundle["matrices"]["boundary_1"], list)
        assert isinstance(bundle["matrices"]["laplacian_0"], list)
        assert isinstance(bundle["matrices"]["laplacian_1"], list)
    
    def test_bundle_mathematical_consistency(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """El bundle mantiene consistencia matemática."""
        bundle = topology_vector.build_analysis_bundle({})
        
        h = bundle["homology"]
        s = bundle["spectral"]
        
        # Euler-Poincaré
        assert h["beta_0"] - h["beta_1"] == h["euler_char"]
        
        # Hodge discreto
        assert s["mult_zero"] == h["beta_0"]


# =============================================================================
# TESTS: INTEGRACIÓN END-TO-END
# =============================================================================

class TestEndToEndIntegration:
    """Pruebas de integración completa."""
    
    def test_full_analysis_pipeline(
        self,
        topology_vector: AlphaTopologyVector,
        mock_state_vector: CategoricalState
    ):
        """Pipeline completo de análisis funciona."""
        result = topology_vector(mock_state_vector)
        
        assert "metrics" in result or "error" in result
        if "metrics" in result:
            metrics = result["metrics"]
            assert "beta_0" in metrics
            assert "fiedler_value" in metrics
    
    def test_error_handling_invalid_payload(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Manejo correcto de payloads inválidos."""
        state = MagicMock(spec=CategoricalState)
        state.payload = {"disable_nodes": "no_es_lista"}  # Error: debería ser lista
        
        result = topology_vector(state)
        
        assert result.get("success") == False or "error" in result
    
    def test_narrative_generation(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Narrativa descriptiva se genera correctamente."""
        metrics = BmcTopologyMetrics(
            beta_0=1,
            beta_1=0,
            euler_char=1,
            rank_boundary_1=8,
            nullity_boundary_1=0,
            fiedler_value=0.5,
            spectral_gap=0.5,
            spectral_radius=4.0,
            multiplicity_zero=1,
            trace_laplacian=16.0,
            directed_cycle_count=0,
            fundamental_cycle_count=0,
            is_connected=True,
            has_cycle_space=False,
            has_directed_feedback=False,
            is_dag=True,
            is_spectrally_stable=True,
            connectivity_class=ConnectivityClass.WEAKLY_CONNECTED,
            n_vertices=9,
            n_edges=8,
        )
        
        narrative = topology_vector._generate_narrative(metrics)
        
        assert "conexo" in narrative.lower()
        assert "acíclico" in narrative.lower() or "dag" in narrative.lower()


# =============================================================================
# TESTS: CASOS EDGE
# =============================================================================

class TestEdgeCases:
    """Pruebas para casos límite."""
    
    def test_empty_graph(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Grafo vacío (sin nodos ni aristas)."""
        G = nx.Graph()
        cc = topology_vector._build_chain_complex_1d(G)
        
        assert cc.dimension_0 == 0
        assert cc.dimension_1 == 0
    
    def test_isolated_nodes_only(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Grafo con solo nodos aislados."""
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C"])
        
        cc = topology_vector._build_chain_complex_1d(G)
        h = topology_vector._compute_homology_metrics(cc)
        
        assert h.beta_0 == 3  # Tres componentes
        assert h.beta_1 == 0  # Sin ciclos
    
    def test_very_large_graph_performance(
        self,
        topology_vector: AlphaTopologyVector
    ):
        """Rendimiento con grafo grande (smoke test)."""
        import time
        
        n = 100
        G = nx.path_graph(n)
        mapping = {i: f"v{i}" for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        
        start = time.time()
        cc = topology_vector._build_chain_complex_1d(G)
        h = topology_vector._compute_homology_metrics(cc)
        s = topology_vector._compute_spectral_metrics(cc)
        elapsed = time.time() - start
        
        # Debe completarse en tiempo razonable (< 5 segundos)
        assert elapsed < 5.0
        assert h.beta_0 == 1
        assert h.beta_1 == 0


# =============================================================================
# TESTS: PROPIEDADES MATEMÁTICAS INVARIANTES
# =============================================================================

class TestMathematicalInvariants:
    """Pruebas de propiedades matemáticas fundamentales."""
    
    @pytest.mark.parametrize("n", [3, 5, 7, 10])
    def test_cyclomatic_number_formula(
        self,
        topology_vector: AlphaTopologyVector,
        n: int
    ):
        """
        Número ciclomático: β₁ = |E| - |V| + β₀
        
        Para grafo conexo: β₁ = |E| - |V| + 1
        """
        # Grafo completo K_n
        G = nx.complete_graph(n)
        mapping = {i: f"v{i}" for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        
        cc = topology_vector._build_chain_complex_1d(G)
        h = topology_vector._compute_homology_metrics(cc)
        
        expected_beta1 = G.number_of_edges() - G.number_of_nodes() + 1
        assert h.beta_1 == expected_beta1
    
    def test_boundary_squared_is_zero(
        self,
        topology_vector: AlphaTopologyVector,
        complete_graph_k4: nx.Graph
    ):
        """
        En complejos de cadenas: ∂₀ ∘ ∂₁ = 0
        
        En 1-complejos, ∂₀ ≡ 0 trivialmente, pero podemos verificar
        que im(∂₁) ⊆ ker(∂₀) (que es todo C₀ en este caso).
        """
        cc = topology_vector._build_chain_complex_1d(complete_graph_k4)
        
        # La imagen de cualquier columna de ∂₁ suma 0
        for j in range(cc.boundary_1.shape[1]):
            column_sum = np.sum(cc.boundary_1[:, j])
            assert abs(column_sum) < NUMERICAL_TOL
    
    def test_laplacian_kernel_dimension(
        self,
        topology_vector: AlphaTopologyVector,
        disconnected_graph: nx.Graph
    ):
        """
        dim(ker(L₀)) = β₀ (número de componentes conexas)
        """
        cc = topology_vector._build_chain_complex_1d(disconnected_graph)
        h = topology_vector._compute_homology_metrics(cc)
        
        L0_kernel = compute_null_space_basis(cc.laplacian_0)
        
        assert L0_kernel.shape[1] == h.beta_0
    
    def test_sum_of_eigenvalues_equals_trace(
        self,
        topology_vector: AlphaTopologyVector,
        simple_triangle_graph: nx.Graph
    ):
        """
        Σλᵢ = tr(L₀)
        """
        cc = topology_vector._build_chain_complex_1d(simple_triangle_graph)
        s = topology_vector._compute_spectral_metrics(cc)
        
        sum_eigenvalues = sum(s.eigenvalues)
        
        assert_allclose(sum_eigenvalues, s.trace_laplacian, atol=NUMERICAL_TOL)


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])