"""
=============================================================================
test_logistics_manifold.py — Suite de Pruebas Rigurosas para LogisticsManifold
=============================================================================

Organización por secciones:

    §0.  Infraestructura de pruebas (fixtures, mocks, helpers)
    §1.  Validación de entradas y sanitización
    §2.  Construcción de operadores discretos (B₁, C)
    §3.  Conservación discreta (ecuación de continuidad)
    §4.  Descomposición de Hodge discreta
    §5.  Política solenoidal
    §6.  Geodésicas riemannianas discretas
    §7.  Análisis espectral (valor de Fiedler)
    §8.  Centralidades estructurales
    §9.  Renormalización de masa (polarones logísticos)
    §10. Política regenerativa DPP
    §11. Ejecución completa del funtor (__call__)
    §12. Casos límite y robustez numérica

Convenciones:
    - Cada test documenta la propiedad matemática que verifica.
    - Los grafos de prueba son canónicos y reproducibles.
    - Las tolerancias numéricas se fijan en 1e-8 salvo justificación.

Referencias:
    [1] Lim, L.-H. "Hodge Laplacians on Graphs." SIAM Review, 62(3), 2020.
    [2] Mohar, B. "The Laplacian spectrum of graphs." 1991.
    [3] Jiang et al. "Statistical ranking and combinatorial Hodge theory." 2011.

=============================================================================
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

# ── Importaciones del módulo bajo prueba ─────────────────────────────────
# Ajustar la ruta de importación según la estructura del proyecto.
from app.tactics.logistics_manifold import (
    ContinuityReport,
    CycleData,
    HodgeDecomposition,
    IncidenceData,
    LogisticsManifold,
)

# =========================================================================
# §0. Infraestructura de pruebas
# =========================================================================

# ── Tolerancia numérica global ───────────────────────────────────────────
ATOL = 1e-8
RTOL = 1e-7


# ── Fixture: instancia del manifold ──────────────────────────────────────

@pytest.fixture
def manifold() -> LogisticsManifold:
    """Instancia estándar de LogisticsManifold con parámetros por defecto."""
    return LogisticsManifold(
        name="test_router",
        tolerance=1e-9,
        regularization=1e-10,
        max_ortho_iters=5,
    )


@pytest.fixture
def strict_manifold() -> LogisticsManifold:
    """Instancia con tolerancia más estricta para pruebas de precisión."""
    return LogisticsManifold(
        name="strict_router",
        tolerance=1e-12,
        regularization=1e-14,
        max_ortho_iters=10,
    )


# ── Grafos canónicos de prueba ───────────────────────────────────────────

@pytest.fixture
def simple_path_graph() -> nx.DiGraph:
    """
    Grafo camino: A → B → C

    Propiedades:
        n=3, m=2, c=1, β₁=0
        Sin ciclos.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", time=1.0, cost=2.0, risk=0.1, flow=5.0)
    G.add_edge("B", "C", time=2.0, cost=1.0, risk=0.2, flow=5.0)
    # Fuente en A, sumidero en C, balanceado en B
    G.nodes["A"]["sink_source"] = 5.0
    G.nodes["B"]["sink_source"] = 0.0
    G.nodes["C"]["sink_source"] = -5.0
    return G


@pytest.fixture
def triangle_graph() -> nx.DiGraph:
    """
    Grafo triángulo dirigido: A → B → C → A

    Propiedades:
        n=3, m=3, c=1, β₁=1
        Un ciclo fundamental.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", time=1.0, cost=1.0, risk=0.0, flow=1.0)
    G.add_edge("B", "C", time=1.0, cost=1.0, risk=0.0, flow=1.0)
    G.add_edge("C", "A", time=1.0, cost=1.0, risk=0.0, flow=1.0)
    for n in ["A", "B", "C"]:
        G.nodes[n]["sink_source"] = 0.0
    return G


@pytest.fixture
def diamond_graph() -> nx.DiGraph:
    """
    Grafo diamante:
        A → B, A → C, B → D, C → D

    Propiedades:
        n=4, m=4, c=1, β₁=1
        Un ciclo fundamental: A→B→D←C←A.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", time=1.0, cost=1.0, risk=0.0, flow=3.0)
    G.add_edge("A", "C", time=2.0, cost=1.5, risk=0.1, flow=2.0)
    G.add_edge("B", "D", time=1.0, cost=1.0, risk=0.0, flow=3.0)
    G.add_edge("C", "D", time=1.5, cost=1.0, risk=0.05, flow=2.0)
    G.nodes["A"]["sink_source"] = 5.0
    G.nodes["B"]["sink_source"] = 0.0
    G.nodes["C"]["sink_source"] = 0.0
    G.nodes["D"]["sink_source"] = -5.0
    return G


@pytest.fixture
def disconnected_graph() -> nx.DiGraph:
    """
    Grafo con dos componentes conexas:
        Componente 1: A → B
        Componente 2: C → D

    Propiedades:
        n=4, m=2, c=2, β₁=0
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", flow=1.0)
    G.add_edge("C", "D", flow=1.0)
    G.nodes["A"]["sink_source"] = 1.0
    G.nodes["B"]["sink_source"] = -1.0
    G.nodes["C"]["sink_source"] = 1.0
    G.nodes["D"]["sink_source"] = -1.0
    return G


@pytest.fixture
def single_edge_graph() -> nx.DiGraph:
    """Grafo mínimo: un solo nodo fuente y un sumidero."""
    G = nx.DiGraph()
    G.add_edge("S", "T", time=1.0, cost=1.0, risk=0.0, flow=10.0)
    G.nodes["S"]["sink_source"] = 10.0
    G.nodes["T"]["sink_source"] = -10.0
    return G


@pytest.fixture
def graph_with_delays() -> nx.DiGraph:
    """Grafo con atributos de delay en nodos para pruebas de polarones."""
    G = nx.DiGraph()
    G.add_edge("A", "B", flow=3.0)
    G.add_edge("B", "C", flow=3.0)
    G.add_edge("A", "C", flow=0.0)
    G.nodes["A"]["sink_source"] = 3.0
    G.nodes["B"]["sink_source"] = 0.0
    G.nodes["C"]["sink_source"] = -3.0
    G.nodes["A"]["delay"] = 1.0
    G.nodes["B"]["delay"] = 2.0
    G.nodes["C"]["delay"] = 0.5
    return G


@pytest.fixture
def large_grid_graph() -> nx.DiGraph:
    """
    Grafo grilla 8x8 dirigido (n=64, m≈224).
    Para pruebas de escalabilidad del análisis espectral.
    """
    grid = nx.grid_2d_graph(8, 8)
    G = nx.DiGraph()
    for u, v in grid.edges():
        u_str = f"{u[0]}_{u[1]}"
        v_str = f"{v[0]}_{v[1]}"
        G.add_edge(u_str, v_str, time=1.0, cost=1.0, risk=0.0, flow=0.0)
        G.add_edge(v_str, u_str, time=1.0, cost=1.0, risk=0.0, flow=0.0)
    for n in G.nodes:
        G.nodes[n]["sink_source"] = 0.0
    return G


# ── Mock del CategoricalState ───────────────────────────────────────────

def make_mock_state(context: Dict[str, Any]) -> MagicMock:
    """
    Crea un mock de CategoricalState con el contexto dado.

    El mock soporta:
        - state.context → dict
        - state.with_update(new_context=...) → nuevo mock
        - state.with_error(error_msg=...) → nuevo mock con error
    """
    state = MagicMock()
    state.context = context

    def with_update(new_context: Dict[str, Any]) -> MagicMock:
        result = MagicMock()
        result.context = {**context, **new_context}
        result.error = None
        return result

    def with_error(error_msg: str) -> MagicMock:
        result = MagicMock()
        result.context = context
        result.error = error_msg
        return result

    state.with_update = MagicMock(side_effect=with_update)
    state.with_error = MagicMock(side_effect=with_error)
    return state


# ── Helpers matemáticos ──────────────────────────────────────────────────

def assert_vectors_orthogonal(
    a: np.ndarray, b: np.ndarray, tol: float = ATOL
) -> None:
    """Verifica que ⟨a, b⟩ ≈ 0."""
    inner = abs(float(np.dot(a.ravel(), b.ravel())))
    assert inner < tol, (
        f"Vectores no ortogonales: |⟨a, b⟩| = {inner:.2e} > {tol:.2e}"
    )


def assert_vector_in_image(
    v: np.ndarray, A: sp.spmatrix, tol: float = ATOL
) -> None:
    """Verifica que v ∈ Im(A) comprobando que ‖v - proj_{Im(A)}(v)‖ ≈ 0."""
    from scipy.sparse.linalg import lsqr

    x = lsqr(A, v, atol=1e-12, btol=1e-12)[0]
    proj = A @ x
    residual = float(np.linalg.norm(v - proj))
    assert residual < tol, (
        f"Vector no pertenece a Im(A): residual = {residual:.2e} > {tol:.2e}"
    )


def assert_psd(matrix: np.ndarray, tol: float = ATOL) -> None:
    """Verifica que una matriz sea simétrica semidefinida positiva."""
    # Simetría
    asym = float(np.linalg.norm(matrix - matrix.T))
    assert asym < tol, f"Matriz no simétrica: ‖G - Gᵀ‖ = {asym:.2e}"
    # Eigenvalores no negativos
    evals = np.linalg.eigvalsh(matrix)
    min_eval = float(np.min(evals))
    assert min_eval >= -tol, f"Eigenvalor negativo: λ_min = {min_eval:.2e}"


# =========================================================================
# §1. Validación de entradas y sanitización
# =========================================================================

class TestValidation:
    """Pruebas de validación de grafos y sanitización del tensor métrico."""

    # ── Validación del grafo ─────────────────────────────────────────

    def test_validate_graph_none_raises(self, manifold: LogisticsManifold):
        """Grafo None debe lanzar TypeError."""
        with pytest.raises(TypeError, match="nx.DiGraph"):
            manifold._validate_graph(None)

    def test_validate_graph_wrong_type_raises(self, manifold: LogisticsManifold):
        """Tipo incorrecto (no DiGraph) debe lanzar TypeError."""
        with pytest.raises(TypeError):
            manifold._validate_graph("not_a_graph")

    def test_validate_graph_undirected_raises(self, manifold: LogisticsManifold):
        """Un Graph (no dirigido) debe lanzar TypeError."""
        with pytest.raises(TypeError):
            manifold._validate_graph(nx.Graph())

    def test_validate_graph_empty_nodes_raises(self, manifold: LogisticsManifold):
        """Grafo sin nodos debe lanzar ValueError."""
        with pytest.raises(ValueError, match="vacío"):
            manifold._validate_graph(nx.DiGraph())

    def test_validate_graph_no_edges_raises(self, manifold: LogisticsManifold):
        """Grafo con nodos pero sin aristas debe lanzar ValueError."""
        G = nx.DiGraph()
        G.add_node("lonely")
        with pytest.raises(ValueError, match="aristas"):
            manifold._validate_graph(G)

    def test_validate_graph_valid(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Grafo válido debe retornarse sin modificación."""
        result = manifold._validate_graph(simple_path_graph)
        assert result is simple_path_graph

    # ── Sanitización del tensor métrico ──────────────────────────────

    def test_sanitize_metric_identity_when_none(self, manifold: LogisticsManifold):
        """Sin tensor → debe retornar identidad 3x3 (o fallback)."""
        with patch(
            "app.tactics.logistics_manifold.G_PHYSICS", None
        ), patch(
            "app.tactics.logistics_manifold.MetricTensorFactory"
        ) as mock_factory:
            mock_factory.build.side_effect = Exception("no factory")
            metric = manifold._sanitize_metric(None)
            assert metric.shape == (3, 3)
            np.testing.assert_array_almost_equal(metric, np.eye(3))

    def test_sanitize_metric_symmetrization(self, manifold: LogisticsManifold):
        """
        Propiedad: G_out = ½(G_in + G_inᵀ)
        Una matriz asimétrica debe simetrizarse.
        """
        asymmetric = np.array([[1.0, 2.0], [0.0, 1.0]])
        result = manifold._sanitize_metric(asymmetric)
        np.testing.assert_array_almost_equal(result, result.T, decimal=12)

    def test_sanitize_metric_psd_projection(self, manifold: LogisticsManifold):
        """
        Propiedad: eigenvalores negativos se proyectan a 0.
        Resultado debe ser PSD.
        """
        indefinite = np.array([[1.0, 0.0], [0.0, -2.0]])
        result = manifold._sanitize_metric(indefinite)
        assert_psd(result)
        # El eigenvalor -2 debe mapearse a 0
        evals = np.linalg.eigvalsh(result)
        assert float(np.min(evals)) >= -ATOL

    def test_sanitize_metric_already_psd(self, manifold: LogisticsManifold):
        """Matriz PSD válida debe preservarse (hasta error numérico)."""
        psd = np.array([[4.0, 2.0], [2.0, 3.0]])
        result = manifold._sanitize_metric(psd)
        assert_psd(result)
        np.testing.assert_array_almost_equal(result, psd, decimal=10)

    def test_sanitize_metric_invalid_shape_fallback(
        self, manifold: LogisticsManifold
    ):
        """Forma no cuadrada → fallback a identidad."""
        rectangular = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = manifold._sanitize_metric(rectangular)
        assert result.shape[0] == result.shape[1]
        assert_psd(result)

    def test_sanitize_metric_1d_fallback(self, manifold: LogisticsManifold):
        """Array 1D → fallback a identidad."""
        result = manifold._sanitize_metric(np.array([1.0, 2.0, 3.0]))
        assert result.ndim == 2
        assert result.shape[0] == result.shape[1]

    def test_sanitize_metric_zero_matrix(self, manifold: LogisticsManifold):
        """Matriz cero es PSD (semidefinida, no definida)."""
        result = manifold._sanitize_metric(np.zeros((3, 3)))
        assert_psd(result)
        np.testing.assert_array_almost_equal(result, np.zeros((3, 3)))

    # ── Validación de parámetros del constructor ─────────────────────

    def test_constructor_negative_tolerance_raises(self):
        """Tolerancia negativa debe fallar."""
        with pytest.raises(ValueError, match="tolerance"):
            LogisticsManifold(tolerance=-1e-9)

    def test_constructor_zero_tolerance_raises(self):
        """Tolerancia cero debe fallar (debe ser estrictamente positiva)."""
        with pytest.raises(ValueError, match="tolerance"):
            LogisticsManifold(tolerance=0.0)

    def test_constructor_negative_regularization_raises(self):
        """Regularización negativa debe fallar."""
        with pytest.raises(ValueError, match="regularization"):
            LogisticsManifold(regularization=-1e-10)

    def test_constructor_zero_ortho_iters_raises(self):
        """max_ortho_iters < 1 debe fallar."""
        with pytest.raises(ValueError, match="max_ortho_iters"):
            LogisticsManifold(max_ortho_iters=0)


# =========================================================================
# §2. Construcción de operadores discretos (B₁, C)
# =========================================================================

class TestIncidenceMatrix:
    """
    Pruebas de la matriz de incidencia B₁.

    Propiedades algebraicas fundamentales:
        1. B₁ ∈ ℝⁿˣᵐ
        2. Cada columna tiene exactamente un +1 y un -1
        3. B₁ 𝟏 = 0 (columnas suman cero)
        4. rank(B₁) = n - c donde c = componentes conexas
    """

    def test_shape(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """B₁ debe tener forma (n_nodes, n_edges)."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        n = simple_path_graph.number_of_nodes()
        m = simple_path_graph.number_of_edges()
        assert inc.B1.shape == (n, m)

    def test_column_sum_zero(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Propiedad: cada columna de B₁ suma exactamente 0."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        col_sums = np.asarray(inc.B1.sum(axis=0)).ravel()
        np.testing.assert_array_almost_equal(col_sums, 0.0, decimal=12)

    def test_column_entries(
        self, manifold: LogisticsManifold, single_edge_graph: nx.DiGraph
    ):
        """
        Para arista (S, T):
            B₁[S, 0] = +1, B₁[T, 0] = -1
        """
        inc = manifold._build_incidence_matrix(single_edge_graph)
        B1_dense = inc.B1.toarray()

        s_idx = inc.node_idx["S"]
        t_idx = inc.node_idx["T"]

        assert B1_dense[s_idx, 0] == pytest.approx(+1.0)
        assert B1_dense[t_idx, 0] == pytest.approx(-1.0)

    def test_each_column_has_one_plus_one_minus(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Cada columna tiene exactamente un +1 y un -1."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        B1_dense = inc.B1.toarray()
        for j in range(B1_dense.shape[1]):
            col = B1_dense[:, j]
            assert np.sum(col == 1.0) == 1, f"Columna {j}: falta +1"
            assert np.sum(col == -1.0) == 1, f"Columna {j}: falta -1"
            assert np.sum(np.abs(col) > 0) == 2, f"Columna {j}: entradas extra"

    def test_rank_equals_n_minus_components(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """
        Propiedad: rank(B₁) = n - c
        Para grafo desconexo con 2 componentes: rank = 4 - 2 = 2.
        """
        inc = manifold._build_incidence_matrix(disconnected_graph)
        rank = np.linalg.matrix_rank(inc.B1.toarray())
        n = disconnected_graph.number_of_nodes()
        c = nx.number_connected_components(disconnected_graph.to_undirected())
        assert rank == n - c

    def test_node_edge_indices_bijective(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Los mapas node_idx y edge_idx deben ser biyecciones."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        # Inyectividad y sobreyectividad de node_idx
        assert len(inc.node_idx) == len(inc.nodes)
        assert set(inc.node_idx.values()) == set(range(len(inc.nodes)))
        # Inyectividad y sobreyectividad de edge_idx
        assert len(inc.edge_idx) == len(inc.edges)
        assert set(inc.edge_idx.values()) == set(range(len(inc.edges)))


class TestCycleMatrix:
    """
    Pruebas de la matriz de ciclos C.

    Propiedades:
        1. C ∈ ℝᵐˣᵏ donde k = número de ciclos fundamentales
        2. β₁ = m - n + c
        3. rank(C) = β₁
        4. Im(C) ⊆ ker(B₁)
    """

    def test_acyclic_graph_has_no_cycles(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Grafo camino (árbol): β₁ = 0, C tiene 0 columnas."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        cyc = manifold._build_cycle_matrix(simple_path_graph, inc.edge_idx, inc.B1)
        assert cyc.betti_1 == 0
        assert cyc.C.shape[1] == 0
        assert cyc.rank == 0

    def test_triangle_has_one_cycle(
        self, manifold: LogisticsManifold, triangle_graph: nx.DiGraph
    ):
        """Triángulo: β₁ = 3 - 3 + 1 = 1."""
        inc = manifold._build_incidence_matrix(triangle_graph)
        cyc = manifold._build_cycle_matrix(triangle_graph, inc.edge_idx, inc.B1)
        assert cyc.betti_1 == 1
        assert cyc.rank == 1

    def test_diamond_has_one_cycle(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Diamante: β₁ = 4 - 4 + 1 = 1."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        assert cyc.betti_1 == 1
        assert cyc.rank == 1

    def test_betti_1_formula(
        self, manifold: LogisticsManifold, large_grid_graph: nx.DiGraph
    ):
        """
        Verifica β₁ = m - n + c para la grilla.
        """
        inc = manifold._build_incidence_matrix(large_grid_graph)
        cyc = manifold._build_cycle_matrix(large_grid_graph, inc.edge_idx, inc.B1)
        n = large_grid_graph.number_of_nodes()
        m = large_grid_graph.number_of_edges()
        c = nx.number_connected_components(large_grid_graph.to_undirected())
        expected_betti = m - n + c
        assert cyc.betti_1 == expected_betti

    def test_cycle_columns_in_kernel_of_B1(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """
        Propiedad fundamental: Im(C) ⊆ ker(B₁).
        Es decir, B₁ C = 0.
        """
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        product = inc.B1 @ cyc.C
        product_dense = product.toarray()
        np.testing.assert_array_almost_equal(
            product_dense, 0.0, decimal=10,
            err_msg="B₁ C ≠ 0: las columnas de C no están en ker(B₁)"
        )

    def test_disconnected_graph_betti(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """Grafo desconexo sin ciclos: β₁ = 2 - 4 + 2 = 0."""
        inc = manifold._build_incidence_matrix(disconnected_graph)
        cyc = manifold._build_cycle_matrix(disconnected_graph, inc.edge_idx, inc.B1)
        assert cyc.betti_1 == 0


# =========================================================================
# §3. Conservación discreta
# =========================================================================

class TestDiscreteConservation:
    """
    Pruebas de la ecuación de continuidad discreta B₁f = s.
    """

    def test_conservation_satisfied(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """
        Grafo camino con flujo conservativo:
            A→B (flow=5), B→C (flow=5)
            s = [5, 0, -5]
        Debe pasar sin errores.
        """
        inc = manifold._build_incidence_matrix(simple_path_graph)
        edges = inc.edges
        f = np.array([simple_path_graph.edges[e]["flow"] for e in edges])
        s = np.array([
            simple_path_graph.nodes[n]["sink_source"] for n in inc.nodes
        ])
        report = manifold._enforce_discrete_continuity(inc.B1, f, s, strict=True)
        assert report.residual_inf < ATOL
        assert report.residual_l2 < ATOL
        assert report.mass_imbalance < ATOL

    def test_conservation_violated_strict_raises(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Flujo no conservativo + strict=True debe lanzar ValueError."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.0, 3.0])  # Flujo inconsistente
        s = np.array([5.0, 0.0, -5.0])
        with pytest.raises(ValueError, match="conservación"):
            manifold._enforce_discrete_continuity(inc.B1, f, s, strict=True)

    def test_mass_imbalance_strict_raises(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """∑sᵢ ≠ 0 + strict=True debe lanzar ValueError."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.0, 5.0])
        s = np.array([5.0, 0.0, -3.0])  # sum = 2.0 ≠ 0
        with pytest.raises(ValueError, match="masa"):
            manifold._enforce_discrete_continuity(inc.B1, f, s, strict=True)

    def test_conservation_violated_non_strict_warns(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """strict=False debe advertir pero no lanzar."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.0, 3.0])
        s = np.array([5.0, 0.0, -5.0])
        # No debe lanzar
        report = manifold._enforce_discrete_continuity(
            inc.B1, f, s, strict=False
        )
        assert report.residual_inf > 0

    def test_dimension_mismatch_f_raises(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Dimensión de f incompatible con B₁ debe lanzar ValueError."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        with pytest.raises(ValueError, match="incompatible"):
            manifold._enforce_discrete_continuity(
                inc.B1, np.array([1.0]), np.zeros(3), strict=True
            )

    def test_dimension_mismatch_s_raises(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Dimensión de s incompatible con B₁ debe lanzar ValueError."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        with pytest.raises(ValueError, match="incompatible"):
            manifold._enforce_discrete_continuity(
                inc.B1, np.zeros(2), np.array([1.0, 2.0]), strict=True
            )

    def test_integrality_defect_computed(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Flujos no enteros deben reportar defecto de integralidad."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.3, 5.3])
        s = np.array([5.3, 0.0, -5.3])
        report = manifold._enforce_discrete_continuity(
            inc.B1, f, s, strict=True
        )
        assert report.integrality_defect > 0

    def test_integer_flows_zero_defect(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Flujos enteros deben tener defecto de integralidad ≈ 0."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.0, 5.0])
        s = np.array([5.0, 0.0, -5.0])
        report = manifold._enforce_discrete_continuity(
            inc.B1, f, s, strict=True
        )
        assert report.integrality_defect < ATOL

    def test_report_is_dataclass(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """El reporte debe ser un ContinuityReport (dataclass)."""
        inc = manifold._build_incidence_matrix(simple_path_graph)
        f = np.array([5.0, 5.0])
        s = np.array([5.0, 0.0, -5.0])
        report = manifold._enforce_discrete_continuity(inc.B1, f, s)
        assert isinstance(report, ContinuityReport)
        assert isinstance(report.to_dict(), dict)


# =========================================================================
# §4. Descomposición de Hodge discreta
# =========================================================================

class TestHodgeDecomposition:
    """
    Pruebas de la descomposición de Hodge:
        f = f_grad + f_curl + f_harm

    Propiedades verificadas:
        1. Reconstrucción: f_grad + f_curl + f_harm = f
        2. Ortogonalidad mutua (hasta tolerancia)
        3. Identidad de Parseval: ‖f‖² = ‖f_grad‖² + ‖f_curl‖² + ‖f_harm‖²
        4. f_grad ∈ Im(B₁ᵀ)
        5. f_curl ∈ Im(C)  (si β₁ > 0)
        6. Caso trivial: f=0 ⟹ todas las componentes son 0
    """

    def test_reconstruction(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """f_grad + f_curl + f_harm = f (identidad de descomposición)."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.array([
            diamond_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        reconstructed = hodge.f_grad + hodge.f_curl + hodge.f_harm
        np.testing.assert_allclose(
            reconstructed, f, atol=ATOL,
            err_msg="Falla de reconstrucción: f_grad + f_curl + f_harm ≠ f"
        )

    def test_orthogonality(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Ortogonalidad mutua de las tres componentes."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.array([
            diamond_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        assert_vectors_orthogonal(hodge.f_grad, hodge.f_curl)
        assert_vectors_orthogonal(hodge.f_grad, hodge.f_harm)
        assert_vectors_orthogonal(hodge.f_curl, hodge.f_harm)

    def test_parseval_identity(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """‖f‖² = ‖f_grad‖² + ‖f_curl‖² + ‖f_harm‖²."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.array([
            diamond_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        lhs = float(np.dot(f, f))
        rhs = hodge.grad_energy + hodge.curl_energy + hodge.harm_energy
        assert lhs == pytest.approx(rhs, abs=ATOL), (
            f"Parseval falla: ‖f‖²={lhs:.6e} ≠ "
            f"‖f_grad‖²+‖f_curl‖²+‖f_harm‖²={rhs:.6e}"
        )

    def test_gradient_in_image_of_B1T(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """f_grad ∈ Im(B₁ᵀ)."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.array([
            diamond_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        assert_vector_in_image(hodge.f_grad, inc.B1.T, tol=ATOL)

    def test_curl_in_image_of_C(
        self, manifold: LogisticsManifold, triangle_graph: nx.DiGraph
    ):
        """f_curl ∈ Im(C) cuando β₁ > 0."""
        inc = manifold._build_incidence_matrix(triangle_graph)
        cyc = manifold._build_cycle_matrix(triangle_graph, inc.edge_idx, inc.B1)
        f = np.array([
            triangle_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        if cyc.betti_1 > 0 and np.linalg.norm(hodge.f_curl) > ATOL:
            assert_vector_in_image(hodge.f_curl, cyc.C, tol=ATOL)

    def test_zero_flow(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """f = 0 ⟹ f_grad = f_curl = f_harm = 0."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.zeros(len(inc.edges))

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        np.testing.assert_array_almost_equal(hodge.f_grad, 0.0)
        np.testing.assert_array_almost_equal(hodge.f_curl, 0.0)
        np.testing.assert_array_almost_equal(hodge.f_harm, 0.0)
        assert hodge.total_energy == pytest.approx(0.0)

    def test_pure_gradient_flow(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """
        En un árbol (β₁ = 0), todo flujo es gradiente puro:
            f_curl = 0, f_harm = 0.
        """
        inc = manifold._build_incidence_matrix(simple_path_graph)
        cyc = manifold._build_cycle_matrix(simple_path_graph, inc.edge_idx, inc.B1)
        f = np.array([
            simple_path_graph.edges[e]["flow"] for e in inc.edges
        ], dtype=float)

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        np.testing.assert_allclose(hodge.f_grad, f, atol=ATOL)
        np.testing.assert_array_almost_equal(hodge.f_curl, 0.0)
        assert hodge.curl_energy < ATOL

    def test_pure_cycle_flow(
        self, manifold: LogisticsManifold, triangle_graph: nx.DiGraph
    ):
        """
        Flujo circulante uniforme en triángulo: todo solenoidal.
        s = 0 ⟹ B₁f = 0 ⟹ f ∈ ker(B₁) = Im(C).
        Entonces f_grad = 0 y f_curl = f (módulo componente armónica).
        """
        inc = manifold._build_incidence_matrix(triangle_graph)
        cyc = manifold._build_cycle_matrix(triangle_graph, inc.edge_idx, inc.B1)
        f = np.array([1.0, 1.0, 1.0])

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        assert hodge.grad_energy < ATOL
        # La energía solenoidal + armónica debe ser ≈ total
        assert (hodge.curl_energy + hodge.harm_energy) == pytest.approx(
            hodge.total_energy, abs=ATOL
        )

    def test_orthogonality_defect_reported(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """El defecto de ortogonalidad debe ser un float no negativo."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.array([3.0, 2.0, 3.0, 2.0])

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        assert isinstance(hodge, HodgeDecomposition)
        assert hodge.orthogonality_defect >= 0.0
        assert isinstance(hodge.to_report_dict(), dict)

    def test_random_flow_reconstruction(
        self, manifold: LogisticsManifold, large_grid_graph: nx.DiGraph
    ):
        """
        Reconstrucción para flujo aleatorio en grafo grande.
        Verifica escalabilidad y corrección numérica.
        """
        inc = manifold._build_incidence_matrix(large_grid_graph)
        cyc = manifold._build_cycle_matrix(large_grid_graph, inc.edge_idx, inc.B1)

        rng = np.random.default_rng(42)
        f = rng.standard_normal(len(inc.edges))

        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        reconstructed = hodge.f_grad + hodge.f_curl + hodge.f_harm
        np.testing.assert_allclose(reconstructed, f, atol=1e-6)


# =========================================================================
# §5. Política solenoidal
# =========================================================================

class TestSolenoidalPolicy:
    """
    Pruebas de la política de veto/aceptación del componente solenoidal.
    """

    def test_zero_curl_energy_returns_none(
        self, manifold: LogisticsManifold
    ):
        """Energía solenoidal ≈ 0 → sin vórtice → None."""
        result = manifold._validate_solenoidal_policy(
            curl_energy=0.0, cycle_rank=1, is_regenerative=False
        )
        assert result is None

    def test_regenerative_allows_curl(
        self, manifold: LogisticsManifold
    ):
        """Con política regenerativa, ciclos son permitidos → None."""
        result = manifold._validate_solenoidal_policy(
            curl_energy=10.0, cycle_rank=3, is_regenerative=True
        )
        assert result is None

    def test_nonzero_curl_non_regenerative_raises(
        self, manifold: LogisticsManifold
    ):
        """Energía solenoidal significativa sin regeneración → veto."""
        with pytest.raises(ValueError, match="parasitario"):
            manifold._validate_solenoidal_policy(
                curl_energy=1.0, cycle_rank=1, is_regenerative=False
            )

    def test_tiny_curl_below_tolerance_returns_none(
        self, manifold: LogisticsManifold
    ):
        """Energía solenoidal por debajo de tolerancia → aceptable."""
        result = manifold._validate_solenoidal_policy(
            curl_energy=1e-15, cycle_rank=1, is_regenerative=False
        )
        assert result is None


# =========================================================================
# §6. Geodésicas riemannianas discretas
# =========================================================================

class TestGeodesics:
    """
    Pruebas de cómputo de geodésicas riemannianas discretas.

    Propiedad: la geodésica minimiza la longitud riemanniana discreta
        L(γ) = ∑_{e ∈ γ} √(xₑᵀ G xₑ)
    """

    def test_geodesic_path_graph(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """En un grafo camino, la geodésica es el único camino."""
        metric = np.eye(3)
        path = manifold._compute_logistical_geodesics(
            metric, simple_path_graph, "A", "C"
        )
        assert path == ["A", "B", "C"]

    def test_geodesic_diamond_prefers_cheaper(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """
        Diamante con tensor identidad:
            Ruta A→B→D: √(1²+1²+0²) + √(1²+1²+0²) = 2√2
            Ruta A→C→D: √(4+2.25+0.01) + √(2.25+1+0.0025) > 2√2
        La ruta por B debe ser preferida.
        """
        metric = np.eye(3)
        path = manifold._compute_logistical_geodesics(
            metric, diamond_graph, "A", "D"
        )
        assert path == ["A", "B", "D"]

    def test_geodesic_metric_sensitivity(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """
        Un tensor que penalice fuertemente el costo (dimensión 1)
        puede cambiar la ruta óptima.
        """
        # Penalizar cost (índice 1) fuertemente
        metric = np.diag([0.01, 100.0, 0.01])
        path = manifold._compute_logistical_geodesics(
            metric, diamond_graph, "A", "D"
        )
        # Ambas rutas tienen cost=1.0 en su primer tramo
        # A→B: cost=1.0, A→C: cost=1.5
        # B→D: cost=1.0, C→D: cost=1.0
        # Total cost A→B→D: 2.0, A→C→D: 2.5
        # Con métrica dominada por cost, A→B→D sigue siendo mejor
        assert path == ["A", "B", "D"]

    def test_geodesic_nonexistent_node_raises(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Nodo inexistente debe lanzar ValueError."""
        metric = np.eye(3)
        with pytest.raises(ValueError, match="inválido"):
            manifold._compute_logistical_geodesics(
                metric, simple_path_graph, "A", "Z"
            )

    def test_geodesic_no_path_raises(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """Sin camino entre componentes debe lanzar ValueError."""
        metric = np.eye(3)
        with pytest.raises(ValueError, match="trayectoria"):
            manifold._compute_logistical_geodesics(
                metric, disconnected_graph, "A", "C"
            )

    def test_geodesic_same_source_target(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Origen = destino → camino trivial de un nodo."""
        metric = np.eye(3)
        path = manifold._compute_logistical_geodesics(
            metric, simple_path_graph, "A", "A"
        )
        assert path == ["A"]

    def test_geodesic_with_psd_metric(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Tensor métrico PSD (no definido) debe funcionar correctamente."""
        # Métrica singular (rank 2)
        metric = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0]])
        path = manifold._compute_logistical_geodesics(
            metric, simple_path_graph, "A", "C"
        )
        assert len(path) >= 2
        assert path[0] == "A"
        assert path[-1] == "C"


# =========================================================================
# §7. Análisis espectral (valor de Fiedler)
# =========================================================================

class TestFiedlerValue:
    """
    Pruebas del cómputo de λ₂ del Laplaciano.

    Propiedades:
        1. λ₂ ≥ 0  siempre.
        2. λ₂ = 0  ⟺ grafo desconexo.
        3. λ₂ > 0  ⟺ grafo conexo.
        4. Para el grafo completo Kₙ: λ₂ = n.
        5. Para el ciclo Cₙ: λ₂ = 2(1 - cos(2π/n)).
    """

    def test_fiedler_connected_positive(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Grafo conexo ⟹ λ₂ > 0."""
        undirected = simple_path_graph.to_undirected()
        fiedler = manifold._compute_fiedler_value(undirected)
        assert fiedler > 0

    def test_fiedler_disconnected_zero(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """Grafo desconexo ⟹ λ₂ = 0."""
        undirected = disconnected_graph.to_undirected()
        fiedler = manifold._compute_fiedler_value(undirected)
        assert fiedler == pytest.approx(0.0)

    def test_fiedler_single_node(self, manifold: LogisticsManifold):
        """Grafo con un solo nodo: λ₂ = 0 (no definido, por convención)."""
        G = nx.Graph()
        G.add_node("solo")
        assert manifold._compute_fiedler_value(G) == 0.0

    def test_fiedler_complete_graph(self, manifold: LogisticsManifold):
        """
        Para Kₙ: λ₂ = n.
        K₅: λ₂ = 5.
        """
        K5 = nx.complete_graph(5)
        fiedler = manifold._compute_fiedler_value(K5)
        assert fiedler == pytest.approx(5.0, abs=ATOL)

    def test_fiedler_cycle_graph(self, manifold: LogisticsManifold):
        """
        Para Cₙ: λ₂ = 2(1 - cos(2π/n)).
        C₆: λ₂ = 2(1 - cos(π/3)) = 2(1 - 0.5) = 1.0.
        """
        C6 = nx.cycle_graph(6)
        fiedler = manifold._compute_fiedler_value(C6)
        expected = 2.0 * (1.0 - math.cos(2.0 * math.pi / 6.0))
        assert fiedler == pytest.approx(expected, abs=ATOL)

    def test_fiedler_path_graph_analytic(self, manifold: LogisticsManifold):
        """
        Para Pₙ (camino): λ₂ = 2(1 - cos(π/n)).
        P₅: λ₂ = 2(1 - cos(π/5)).
        """
        P5 = nx.path_graph(5)
        fiedler = manifold._compute_fiedler_value(P5)
        expected = 2.0 * (1.0 - math.cos(math.pi / 5.0))
        assert fiedler == pytest.approx(expected, abs=ATOL)

    def test_fiedler_nonnegative(self, manifold: LogisticsManifold):
        """λ₂ debe ser ≥ 0 para cualquier grafo."""
        for n in [3, 5, 10, 20]:
            G = nx.erdos_renyi_graph(n, 0.3, seed=42)
            fiedler = manifold._compute_fiedler_value(G)
            assert fiedler >= 0.0

    def test_fiedler_large_graph_uses_sparse(
        self, manifold: LogisticsManifold
    ):
        """
        Grafo con n > _DENSE_EIGENVALUE_THRESHOLD debe usar eigsh.
        Verificamos que el resultado es correcto comparando con dense.
        """
        n = manifold._DENSE_EIGENVALUE_THRESHOLD + 10
        G = nx.cycle_graph(n)
        fiedler = manifold._compute_fiedler_value(G)
        expected = 2.0 * (1.0 - math.cos(2.0 * math.pi / n))
        assert fiedler == pytest.approx(expected, abs=1e-4)


# =========================================================================
# §8. Centralidades estructurales
# =========================================================================

class TestCentralities:
    """
    Pruebas de centralidad para renormalización de masa.
    """

    def test_centrality_returns_all_nodes(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Debe retornar un valor para cada nodo."""
        centralities = manifold._compute_structural_centralities(diamond_graph)
        assert set(centralities.keys()) == set(diamond_graph.nodes)

    def test_centrality_values_nonnegative(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Todas las centralidades deben ser ≥ 0."""
        centralities = manifold._compute_structural_centralities(diamond_graph)
        for v in centralities.values():
            assert v >= 0.0

    def test_centrality_empty_graph(self, manifold: LogisticsManifold):
        """Grafo sin nodos → diccionario vacío."""
        G = nx.DiGraph()
        # Añadimos una arista para pasar validación, luego limpiamos
        centralities = manifold._compute_structural_centralities(G)
        assert centralities == {}

    def test_centrality_symmetric_graph(
        self, manifold: LogisticsManifold
    ):
        """
        En un grafo simétrico (todos los nodos equivalentes por automorfismo),
        todas las centralidades deben ser iguales.
        Ejemplo: ciclo C₄ dirigido.
        """
        G = nx.DiGraph()
        nodes = ["A", "B", "C", "D"]
        for i in range(4):
            G.add_edge(nodes[i], nodes[(i + 1) % 4])
        centralities = manifold._compute_structural_centralities(G)
        vals = list(centralities.values())
        for v in vals:
            assert v == pytest.approx(vals[0], abs=ATOL)


# =========================================================================
# §9. Renormalización de masa (polarones logísticos)
# =========================================================================

class TestPolarons:
    """
    Pruebas de la renormalización de masa efectiva:
        m_eff = m₀ · (1 + α/6),   α = c / (λ₂ + ε)

    Propiedades:
        1. m_eff ≥ m₀
        2. m_eff(m₀=0) = 0
        3. m_eff es monótonamente creciente en c
        4. m_eff es monótonamente decreciente en λ₂
        5. m_eff es finito
    """

    def test_effective_mass_geq_base(self, manifold: LogisticsManifold):
        """m_eff ≥ m₀ siempre (α ≥ 0)."""
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=1.0, centrality=0.5, base_mass=3.0
        )
        assert m_eff >= 3.0

    def test_zero_base_mass(self, manifold: LogisticsManifold):
        """m₀ = 0 ⟹ m_eff = 0."""
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=1.0, centrality=0.5, base_mass=0.0
        )
        assert m_eff == pytest.approx(0.0)

    def test_zero_centrality(self, manifold: LogisticsManifold):
        """c = 0 ⟹ α = 0 ⟹ m_eff = m₀."""
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=1.0, centrality=0.0, base_mass=5.0
        )
        assert m_eff == pytest.approx(5.0)

    def test_monotone_in_centrality(self, manifold: LogisticsManifold):
        """m_eff crece con c (fijando λ₂ y m₀)."""
        m1 = manifold._quantize_logistical_polarons(1.0, 0.1, 1.0)
        m2 = manifold._quantize_logistical_polarons(1.0, 0.5, 1.0)
        m3 = manifold._quantize_logistical_polarons(1.0, 1.0, 1.0)
        assert m1 < m2 < m3

    def test_monotone_in_fiedler(self, manifold: LogisticsManifold):
        """m_eff decrece con λ₂ (fijando c y m₀)."""
        m1 = manifold._quantize_logistical_polarons(0.1, 0.5, 1.0)
        m2 = manifold._quantize_logistical_polarons(1.0, 0.5, 1.0)
        m3 = manifold._quantize_logistical_polarons(10.0, 0.5, 1.0)
        assert m1 > m2 > m3

    def test_exact_value(self, manifold: LogisticsManifold):
        """
        Verificación analítica:
            λ₂ = 2.0, c = 0.6, m₀ = 3.0, ε = 1e-9
            α = 0.6 / (2.0 + 1e-9) ≈ 0.3
            m_eff = 3.0 * (1 + 0.3/6) = 3.0 * 1.05 = 3.15
        """
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=2.0, centrality=0.6, base_mass=3.0
        )
        alpha = 0.6 / (2.0 + manifold.tolerance)
        expected = 3.0 * (1.0 + alpha / 6.0)
        assert m_eff == pytest.approx(expected, rel=RTOL)

    def test_negative_inputs_clamped(self, manifold: LogisticsManifold):
        """Entradas negativas se clampean a 0."""
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=-1.0, centrality=-0.5, base_mass=-3.0
        )
        assert m_eff >= 0.0
        assert np.isfinite(m_eff)

    def test_very_small_fiedler(self, manifold: LogisticsManifold):
        """λ₂ ≈ 0 no debe causar divergencia (denominador protegido)."""
        m_eff = manifold._quantize_logistical_polarons(
            fiedler_val=0.0, centrality=1.0, base_mass=1.0
        )
        assert np.isfinite(m_eff)
        assert m_eff > 1.0  # α grande pero finito


# =========================================================================
# §10. Política regenerativa DPP
# =========================================================================

class TestRegenerativePolicy:
    """
    Pruebas de la política de pasaporte de producto digital.
    """

    def test_no_dpp_returns_false(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Sin dpp_circularity → no regenerativo."""
        result = manifold._evaluate_regenerative_policy(
            simple_path_graph, {"dpp_circularity": False}
        )
        assert result is False

    def test_missing_dpp_returns_false(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Clave ausente → no regenerativo."""
        result = manifold._evaluate_regenerative_policy(
            simple_path_graph, {}
        )
        assert result is False

    def test_dpp_with_valid_dissipation(
        self, manifold: LogisticsManifold
    ):
        """DPP con disipación ≥ 0 → regenerativo."""
        G = nx.DiGraph()
        G.add_edge("A", "B", p_diss=0.5)
        G.add_edge("B", "A", p_diss=0.3)
        result = manifold._evaluate_regenerative_policy(
            G, {"dpp_circularity": True}
        )
        assert result is True

    def test_dpp_with_negative_dissipation_raises(
        self, manifold: LogisticsManifold
    ):
        """DPP con disipación negativa → violación termodinámica."""
        G = nx.DiGraph()
        G.add_edge("A", "B", p_diss=-1.0)
        with pytest.raises(ValueError, match="termodinámica"):
            manifold._evaluate_regenerative_policy(
                G, {"dpp_circularity": True}
            )

    def test_dpp_zero_dissipation_allowed(
        self, manifold: LogisticsManifold
    ):
        """Disipación cero es termodinámicamente válida."""
        G = nx.DiGraph()
        G.add_edge("A", "B", p_diss=0.0)
        result = manifold._evaluate_regenerative_policy(
            G, {"dpp_circularity": True}
        )
        assert result is True


# =========================================================================
# §11. Ejecución completa del funtor (__call__)
# =========================================================================

class TestFullExecution:
    """
    Pruebas de integración del funtor __call__.
    """

    def test_successful_execution_path_graph(
        self, manifold: LogisticsManifold, simple_path_graph: nx.DiGraph
    ):
        """Ejecución exitosa con grafo camino (sin ciclos)."""
        state = make_mock_state({
            "logistics_graph": simple_path_graph,
        })
        result = manifold(state)
        state.with_update.assert_called_once()
        ctx = result.context
        assert "logistics_graph" in ctx
        assert ctx["betti_1"] == 0
        assert ctx["cycle_rank"] == 0
        assert ctx["fiedler_value"] > 0

    def test_successful_execution_diamond(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Ejecución exitosa con grafo diamante (β₁ = 1, flujo conservativo)."""
        state = make_mock_state({
            "logistics_graph": diamond_graph,
            "dpp_circularity": True,  # Permitir ciclos
        })
        result = manifold(state)
        state.with_update.assert_called_once()
        ctx = result.context
        assert ctx["betti_1"] == 1
        assert "hodge_report" in ctx
        assert "continuity_report" in ctx

    def test_execution_with_geodesic(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Ejecución con solicitud de geodésica."""
        state = make_mock_state({
            "logistics_graph": diamond_graph,
            "route_source": "A",
            "route_target": "D",
            "dpp_circularity": True,
        })
        result = manifold(state)
        ctx = result.context
        assert "geodesic_path" in ctx
        assert ctx["geodesic_path"][0] == "A"
        assert ctx["geodesic_path"][-1] == "D"

    def test_execution_with_delays(
        self, manifold: LogisticsManifold, graph_with_delays: nx.DiGraph
    ):
        """Ejecución con nodos que tienen delay → masas efectivas."""
        state = make_mock_state({
            "logistics_graph": graph_with_delays,
            "dpp_circularity": True,
        })
        result = manifold(state)
        ctx = result.context
        G = ctx["logistics_graph"]
        # Nodos con delay > 0 deben tener effective_mass
        for n in ["A", "B", "C"]:
            if graph_with_delays.nodes[n].get("delay", 0) > 0:
                assert "effective_mass" in G.nodes[n]
                assert G.nodes[n]["effective_mass"] >= graph_with_delays.nodes[n]["delay"]

    def test_execution_returns_error_on_invalid_graph(
        self, manifold: LogisticsManifold
    ):
        """Grafo inválido debe retornar estado con error."""
        state = make_mock_state({"logistics_graph": None})
        result = manifold(state)
        state.with_error.assert_called_once()
        assert result.error is not None

    def test_execution_returns_error_on_conservation_failure(
        self, manifold: LogisticsManifold
    ):
        """Flujo no conservativo → error."""
        G = nx.DiGraph()
        G.add_edge("A", "B", flow=5.0)
        G.add_edge("B", "C", flow=3.0)  # Violación
        G.nodes["A"]["sink_source"] = 5.0
        G.nodes["B"]["sink_source"] = 0.0
        G.nodes["C"]["sink_source"] = -5.0
        state = make_mock_state({"logistics_graph": G})
        result = manifold(state)
        state.with_error.assert_called_once()

    def test_execution_veto_on_parasitic_cycle(
        self, manifold: LogisticsManifold, triangle_graph: nx.DiGraph
    ):
        """
        Flujo circular sin DPP debe vetar (energía solenoidal > 0).
        """
        state = make_mock_state({
            "logistics_graph": triangle_graph,
            "dpp_circularity": False,
        })
        result = manifold(state)
        state.with_error.assert_called_once()

    def test_euler_characteristic(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """
        χ = c - β₁.
        Diamante: c=1, β₁=1, χ=0.
        """
        state = make_mock_state({
            "logistics_graph": diamond_graph,
            "dpp_circularity": True,
        })
        result = manifold(state)
        ctx = result.context
        assert ctx["euler_characteristic"] == 0
        assert ctx["betti_0"] == 1
        assert ctx["betti_1"] == 1

    def test_euler_characteristic_disconnected(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """
        Grafo desconexo: c=2, β₁=0, χ=2.
        """
        state = make_mock_state({
            "logistics_graph": disconnected_graph,
        })
        result = manifold(state)
        ctx = result.context
        assert ctx["betti_0"] == 2
        assert ctx["betti_1"] == 0
        assert ctx["euler_characteristic"] == 2

    def test_hodge_annotations_on_edges(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Cada arista debe tener flow_grad, flow_curl, flow_harm."""
        state = make_mock_state({
            "logistics_graph": diamond_graph,
            "dpp_circularity": True,
        })
        result = manifold(state)
        G = result.context["logistics_graph"]
        for u, v in G.edges:
            assert "flow_grad" in G.edges[u, v]
            assert "flow_curl" in G.edges[u, v]
            assert "flow_harm" in G.edges[u, v]

    def test_domain_and_codomain(self, manifold: LogisticsManifold):
        """Verificar la interfaz categórica."""
        from app.core.schemas import Stratum
        assert Stratum.PHYSICS in manifold.domain
        assert manifold.codomain == Stratum.TACTICS

    def test_geodesic_error_does_not_crash(
        self, manifold: LogisticsManifold, disconnected_graph: nx.DiGraph
    ):
        """
        Solicitar geodésica imposible (nodos en componentes distintas)
        no debe romper el funtor, solo registrar el error.
        """
        state = make_mock_state({
            "logistics_graph": disconnected_graph,
            "route_source": "A",
            "route_target": "C",
        })
        result = manifold(state)
        ctx = result.context
        # El funtor debe tener éxito (with_update), con geodesic_error
        assert "geodesic_error" in ctx


# =========================================================================
# §12. Casos límite y robustez numérica
# =========================================================================

class TestEdgeCasesAndRobustness:
    """
    Pruebas de robustez para casos extremos y condicionamiento numérico.
    """

    def test_very_large_flows(self, manifold: LogisticsManifold):
        """Flujos del orden 10¹² deben manejarse correctamente."""
        G = nx.DiGraph()
        G.add_edge("A", "B", flow=1e12)
        G.nodes["A"]["sink_source"] = 1e12
        G.nodes["B"]["sink_source"] = -1e12
        inc = manifold._build_incidence_matrix(G)
        f = np.array([1e12])
        s = np.array([1e12, -1e12])
        report = manifold._enforce_discrete_continuity(inc.B1, f, s)
        assert report.residual_inf < 1.0  # Tolerancia absoluta laxa

    def test_very_small_flows(self, manifold: LogisticsManifold):
        """Flujos del orden 10⁻¹² deben manejarse correctamente."""
        G = nx.DiGraph()
        G.add_edge("A", "B", flow=1e-12)
        G.nodes["A"]["sink_source"] = 1e-12
        G.nodes["B"]["sink_source"] = -1e-12
        inc = manifold._build_incidence_matrix(G)
        f = np.array([1e-12])
        s = np.array([1e-12, -1e-12])
        report = manifold._enforce_discrete_continuity(inc.B1, f, s)
        assert report.residual_inf < ATOL

    def test_projection_onto_empty_image(self, manifold: LogisticsManifold):
        """Proyección con A de 0 columnas → vector cero."""
        A = sp.csr_matrix((5, 0))
        y = np.ones(5)
        proj = manifold._orthogonal_projection_onto_image(A, y)
        np.testing.assert_array_almost_equal(proj, 0.0)

    def test_projection_identity(self, manifold: LogisticsManifold):
        """
        Si A = I (identidad), proj_{Im(I)}(y) = y.
        """
        A = sp.eye(5, format="csr")
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        proj = manifold._orthogonal_projection_onto_image(A, y)
        np.testing.assert_allclose(proj, y, atol=ATOL)

    def test_projection_onto_subspace(self, manifold: LogisticsManifold):
        """
        Proyección de [1, 1, 1] sobre el subespacio generado por [1, 0, 0]:
            proj = [1, 0, 0]
        """
        A = sp.csr_matrix(np.array([[1.0], [0.0], [0.0]]))
        y = np.array([1.0, 1.0, 1.0])
        proj = manifold._orthogonal_projection_onto_image(A, y)
        np.testing.assert_allclose(proj, [1.0, 0.0, 0.0], atol=ATOL)

    def test_projection_idempotent(self, manifold: LogisticsManifold):
        """
        Propiedad de proyección: P² = P.
        proj(proj(y)) = proj(y).
        """
        A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        y = np.array([3.0, 4.0, 5.0])
        p1 = manifold._orthogonal_projection_onto_image(A, y)
        p2 = manifold._orthogonal_projection_onto_image(A, p1)
        np.testing.assert_allclose(p1, p2, atol=ATOL)

    def test_hodge_with_all_zero_edges(
        self, manifold: LogisticsManifold, diamond_graph: nx.DiGraph
    ):
        """Todas las aristas con flow=0 debe dar descomposición trivial."""
        inc = manifold._build_incidence_matrix(diamond_graph)
        cyc = manifold._build_cycle_matrix(diamond_graph, inc.edge_idx, inc.B1)
        f = np.zeros(len(inc.edges))
        hodge = manifold._compute_hodge_decomposition(f, inc.B1, cyc.C)
        assert hodge.total_energy == pytest.approx(0.0)
        assert hodge.orthogonality_defect == pytest.approx(0.0)

    def test_dataclass_immutability(self):
        """Las dataclasses de resultado deben ser inmutables (frozen)."""
        report = ContinuityReport(
            residual_inf=0.1,
            residual_l2=0.2,
            mass_imbalance=0.0,
            integrality_defect=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.residual_inf = 999.0  # type: ignore

    def test_incidence_data_immutability(self):
        """IncidenceData debe ser frozen."""
        inc = IncidenceData(
            nodes=["A"],
            edges=[("A", "B")],
            node_idx={"A": 0},
            edge_idx={("A", "B"): 0},
            B1=sp.csr_matrix(np.array([[1.0]])),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            inc.nodes = ["X"]  # type: ignore

    def test_manifold_name(self, manifold: LogisticsManifold):
        """El nombre debe ser el proporcionado al constructor."""
        assert manifold.name == "test_router"

    def test_large_graph_execution(
        self, manifold: LogisticsManifold, large_grid_graph: nx.DiGraph
    ):
        """
        Ejecución en grafo grande (64 nodos) para verificar
        que no hay errores de rendimiento o convergencia.
        """
        state = make_mock_state({
            "logistics_graph": large_grid_graph,
        })
        result = manifold(state)
        state.with_update.assert_called_once()
        ctx = result.context
        assert ctx["betti_0"] == 1
        assert ctx["fiedler_value"] > 0

    def test_self_loop_handling(self, manifold: LogisticsManifold):
        """
        Un grafo con auto-bucle: la columna de B₁ correspondiente
        tiene +1 y -1 en la misma fila ⟹ columna cero.
        """
        G = nx.DiGraph()
        G.add_edge("A", "A", flow=0.0)
        G.add_edge("A", "B", flow=1.0)
        G.nodes["A"]["sink_source"] = 1.0
        G.nodes["B"]["sink_source"] = -1.0
        # El auto-bucle tiene flujo 0, no afecta conservación
        inc = manifold._build_incidence_matrix(G)
        # La columna del auto-bucle debe sumar 0 (ya se verifica internamente)
        B1_dense = inc.B1.toarray()
        loop_idx = inc.edge_idx[("A", "A")]
        # +1 y -1 en la misma fila → la columna es cero
        np.testing.assert_array_almost_equal(
            B1_dense[:, loop_idx], 0.0
        )


# =========================================================================
# Ejecución directa
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])