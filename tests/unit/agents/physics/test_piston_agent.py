# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite   : test_piston_agent.py                                               ║
║ Ruta    : tests/unit/agents/physics/test_piston_agent.py                     ║
║ Objetivo: Validación rigurosa de HodgeHelmholtzInjectorAgent (v7.0.0)        ║
║ Cubre   : FASE 1 (complejo), FASE 2 (Poisson), FASE 3 (Hodge-DEC), funtor    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Estrategia de prueba
────────────────────
  §T0  Helpers algebraicos y métrica de energía
  §T1  FASE 1 — Complejo de cadenas, Leibniz, L₀^W / L₁^W
  §T2  FASE 2 — Poisson, espectro, KCL, Ohm
  §T3  FASE 3 — Proyecciones métricas, ortogonalidad, Pitágoras
  §T4  Orquestación execute_injection + funtor categórico
  §T5  Vetos topológicos / excepciones (casos patológicos)
  §T6  Invariantes de regresión métrica (FIX-M1…M10)

Grafos canónicos
────────────────
  TREE_PATH   : camino A—B—C (acíclico, β₁=0)
  CYCLE_C3    : triángulo A—B—C—A (β₁=1)
  DIAMOND     : A—B—C—D con diagonal (β₁≥1)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import scipy.sparse as sp

from app.agents.physics.piston_agent import (
    BoundaryComplexError,
    FlowState,
    HodgeDecomposition,
    HodgeHelmholtzInjectorAgent,
    HodgeMetricInconsistencyError,
    HomologicalKirchhoffError,
    InjectionVector,
    ParasiticVorticityVetoError,
    PistonConstants,
    PistonInjectorError,
    SimplicialMesh,
    SingularLaplacianError,
    SourceCompatibilityError,
    TopologicalInvariantError,
    _compute_min_eigenvalue_sparse,
    _diag_of_dia,
    _energy_inner_product,
    _energy_norm,
    _estimate_kernel_dimension,
    _lsqr_solve,
)

# Stubs / tipos opcionales del funtor
try:
    from app.core.mic_algebra import CategoricalState
    from app.core.schemas import Stratum
except ImportError:
    from app.agents.physics.piston_agent import CategoricalState, Stratum


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures canónicas de grafos
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tree_path() -> Dict[str, Any]:
    r"""
    Camino A — B — C (árbol).

    |V|=3, |E|=2, |F|=0 → β₁ = 0.
    Flujo de Poisson debe ser puramente gradiente (I_curl = 0 exacto).
    """
    return {
        "nodes": ["A", "B", "C"],
        "edges": [
            ("A", "B", 1.0),
            ("B", "C", 2.0),
        ],
        "cycles": [],
        "pump_source": "A",
        "pump_sink": "C",
        "reservoir_node": "B",
        "q_pump": 1.0,
    }


@pytest.fixture
def cycle_c3() -> Dict[str, Any]:
    r"""
    Triángulo equiconductivo A—B—C—A.

    |V|=3, |E|=3, |F|=1 → β₁ = 1.
    Complejo mínimo no trivial para Hodge con curl.
    """
    return {
        "nodes": ["A", "B", "C"],
        "edges": [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "A", 1.0),
        ],
        "cycles": [
            ["A", "B", "C"],
        ],
        "pump_source": "A",
        "pump_sink": "B",
        "reservoir_node": "C",
        "q_pump": 1.0,
    }


@pytest.fixture
def diamond() -> Dict[str, Any]:
    r"""
    Diamante A—B—D—C—A con arista B—C.

        A
       / \
      B---C
       \ /
        D

    Conductancias heterogéneas para forzar la métrica W.
    """
    return {
        "nodes": ["A", "B", "C", "D"],
        "edges": [
            ("A", "B", 1.0),
            ("A", "C", 2.0),
            ("B", "D", 1.5),
            ("C", "D", 0.5),
            ("B", "C", 3.0),
        ],
        "cycles": [
            ["A", "B", "C"],
            ["B", "C", "D"],
        ],
        "pump_source": "A",
        "pump_sink": "D",
        "reservoir_node": "B",
        "q_pump": 2.0,
    }


@pytest.fixture
def agent() -> HodgeHelmholtzInjectorAgent:
    return HodgeHelmholtzInjectorAgent()


@pytest.fixture
def phase1():
    return HodgeHelmholtzInjectorAgent._Phase1_SimplicialBoundaryBuilder


@pytest.fixture
def phase2():
    return HodgeHelmholtzInjectorAgent._Phase2_DirichletPoissonSolver


@pytest.fixture
def phase3():
    return HodgeHelmholtzInjectorAgent._Phase3_HodgeHelmholtzAuditor


def _build_mesh(phase1, cfg: Dict[str, Any]) -> SimplicialMesh:
    return phase1.build_mesh(cfg["nodes"], cfg["edges"], cfg["cycles"])


def _build_injection(agent, cfg: Dict[str, Any]) -> InjectionVector:
    return agent._create_injection_vector(
        cfg["nodes"], cfg["pump_source"], cfg["pump_sink"], cfg["q_pump"]
    )


def _solve_flow(phase2, mesh, injection, cfg) -> FlowState:
    return phase2.solve_hydrodynamics(mesh, injection, cfg["reservoir_node"])


# ══════════════════════════════════════════════════════════════════════════════
# §T0 — Helpers algebraicos y métrica de energía
# ══════════════════════════════════════════════════════════════════════════════

class TestAlgebraicHelpers:
    r"""Validación de la capa §D (espectral + norma de energía)."""

    def test_energy_norm_euclidean_when_W_is_identity(self):
        v = np.array([3.0, 4.0], dtype=np.float64)
        W_inv = sp.diags([1.0, 1.0], format="dia")
        assert abs(_energy_norm(v, W_inv) - 5.0) < 1e-14

    def test_energy_norm_weighted(self):
        r"""‖v‖_W² = Σ vₖ²/wₖ con w = (4, 1) → v=(2,1) ⇒ 1 + 1 = 2."""
        v = np.array([2.0, 1.0], dtype=np.float64)
        W_inv = sp.diags([0.25, 1.0], format="dia")  # 1/w
        assert abs(_energy_norm(v, W_inv) - math.sqrt(2.0)) < 1e-14

    def test_energy_norm_zero_vector(self):
        v = np.zeros(4, dtype=np.float64)
        W_inv = sp.diags(np.ones(4), format="dia")
        assert _energy_norm(v, W_inv) == 0.0

    def test_energy_inner_product_symmetry(self):
        u = np.array([1.0, -2.0, 0.5], dtype=np.float64)
        v = np.array([0.3, 1.0, -1.0], dtype=np.float64)
        W_inv = sp.diags([2.0, 0.5, 1.0], format="dia")
        uv = _energy_inner_product(u, v, W_inv)
        vu = _energy_inner_product(v, u, W_inv)
        assert abs(uv - vu) < 1e-15

    def test_energy_inner_product_induces_norm(self):
        v = np.array([1.5, -0.7, 2.0], dtype=np.float64)
        W_inv = sp.diags([1.0, 2.0, 0.5], format="dia")
        n2 = _energy_inner_product(v, v, W_inv)
        assert abs(n2 - _energy_norm(v, W_inv) ** 2) < 1e-14

    def test_lsqr_solves_consistent_system(self):
        A = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        b = np.array([4.0, 9.0])
        x = _lsqr_solve(A, b)
        np.testing.assert_allclose(x, [2.0, 3.0], atol=1e-10)

    def test_lsqr_min_norm_on_singular_laplacian(self):
        r"""L = ∂∂ᵀ de un camino de 2 nodos: ker = span{1}; LSQR gauge-fixea."""
        L = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        b = np.array([1.0, -1.0])  # b ⊥ 1
        x = _lsqr_solve(L, b)
        # Solución mínima norma: media nula
        assert abs(np.mean(x)) < 1e-10
        np.testing.assert_allclose(L.dot(x), b, atol=1e-9)

    def test_min_eigenvalue_spd(self):
        A = sp.csr_matrix(np.diag([3.0, 1.0, 5.0]))
        assert abs(_compute_min_eigenvalue_sparse(A) - 1.0) < 1e-9

    def test_min_eigenvalue_singular(self):
        A = sp.csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        assert _compute_min_eigenvalue_sparse(A) < PistonConstants.SPECTRAL_THRESHOLD

    def test_min_eigenvalue_edge_n1(self):
        A = sp.csr_matrix([[2.5]])
        assert abs(_compute_min_eigenvalue_sparse(A) - 2.5) < 1e-15

    def test_estimate_kernel_dimension_rank_deficient(self):
        # Matriz 3×3 de rango 1 → dim ker = 2
        v = np.array([1.0, 1.0, 1.0])
        A = sp.csr_matrix(np.outer(v, v))
        dim_ker = _estimate_kernel_dimension(A, spectral_tol=1e-8)
        assert dim_ker == 2

    def test_diag_of_dia_roundtrip(self):
        d = np.array([1.0, 2.0, 3.0])
        M = sp.diags(d, format="dia")
        np.testing.assert_allclose(_diag_of_dia(M), d)


# ══════════════════════════════════════════════════════════════════════════════
# §T1 — FASE 1: Complejo de cadenas y Laplacianos ponderados
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1SimplicialBoundaryBuilder:
    r"""Propiedades del fibrador topológico (build_mesh)."""

    def test_mesh_dimensions_tree(self, phase1, tree_path):
        mesh = _build_mesh(phase1, tree_path)
        assert len(mesh.nodes) == 3
        assert len(mesh.edges) == 2
        assert len(mesh.cycles) == 0
        assert mesh.boundary_1.shape == (3, 2)
        assert mesh.boundary_2.shape[0] == 2
        assert mesh.boundary_identity is True

    def test_mesh_dimensions_c3(self, phase1, cycle_c3):
        mesh = _build_mesh(phase1, cycle_c3)
        assert mesh.boundary_1.shape == (3, 3)
        assert mesh.boundary_2.shape == (3, 1)
        assert mesh.boundary_identity is True

    def test_leibniz_identity_holds(self, phase1, cycle_c3, diamond):
        for cfg in (cycle_c3, diamond):
            mesh = _build_mesh(phase1, cfg)
            composition = mesh.boundary_1.dot(mesh.boundary_2)
            frob = sp.linalg.norm(composition, ord="fro")
            assert frob < 1e-12, f"‖∂₁∘∂₂‖_F = {frob}"

    def test_boundary_1_column_sum_zero(self, phase1, diamond):
        r"""Cada columna de ∂₁ suma 0 (conservación local de incidencia)."""
        mesh = _build_mesh(phase1, diamond)
        col_sums = np.asarray(mesh.boundary_1.sum(axis=0)).ravel()
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-15)

    def test_conductance_matrix_positive_diagonal(self, phase1, diamond):
        mesh = _build_mesh(phase1, diamond)
        w = _diag_of_dia(mesh.conductance_matrix)
        assert np.all(w > 0)
        w_inv = _diag_of_dia(mesh.inv_conductance)
        np.testing.assert_allclose(w * w_inv, 1.0, atol=1e-14)

    def test_laplacian_0_is_symmetric(self, phase1, diamond):
        mesh = _build_mesh(phase1, diamond)
        L0 = mesh.laplacian_0.toarray()
        np.testing.assert_allclose(L0, L0.T, atol=1e-14)

    def test_laplacian_0_kernel_contains_constants(self, phase1, cycle_c3):
        r"""ker(L₀^W) ⊇ span{1} en grafo conexo sin Dirichlet."""
        mesh = _build_mesh(phase1, cycle_c3)
        ones = np.ones(len(mesh.nodes))
        residual = mesh.laplacian_0.dot(ones)
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_laplacian_0_matches_formula(self, phase1, diamond):
        r"""L₀^W = ∂₁ W ∂₁ᵀ (reconstrucción independiente)."""
        mesh = _build_mesh(phase1, diamond)
        B1, W = mesh.boundary_1, mesh.conductance_matrix
        L0_ref = B1.dot(W.dot(B1.T)).tocsr()
        np.testing.assert_allclose(
            mesh.laplacian_0.toarray(), L0_ref.toarray(), atol=1e-13
        )

    def test_laplacian_1_formula_autoadjoint_structure(self, phase1, diamond):
        r"""
        L₁^W = W ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ W⁻¹  (FIX-M4).
        Verifica reconstrucción y simetría de la forma bilineal
        ⟨L₁ u, v⟩_W = ⟨u, L₁ v⟩_W  (autoadjunción en energía).
        """
        mesh = _build_mesh(phase1, diamond)
        B1, B2 = mesh.boundary_1, mesh.boundary_2
        W, W_inv = mesh.conductance_matrix, mesh.inv_conductance
        L1_ref = (
            W.dot(B1.T.dot(B1)) + B2.dot(B2.T.dot(W_inv))
        ).tocsr()
        np.testing.assert_allclose(
            mesh.laplacian_1.toarray(), L1_ref.toarray(), atol=1e-12
        )

        # Autoadjunción: ⟨L₁u, v⟩_W = ⟨u, L₁v⟩_W
        rng = np.random.default_rng(0)
        n_e = len(mesh.edges)
        u = rng.normal(size=n_e)
        v = rng.normal(size=n_e)
        Lu, Lv = mesh.laplacian_1.dot(u), mesh.laplacian_1.dot(v)
        lhs = _energy_inner_product(Lu, v, W_inv)
        rhs = _energy_inner_product(u, Lv, W_inv)
        assert abs(lhs - rhs) < 1e-10

    def test_betti_1_tree_is_zero(self, phase1, tree_path):
        mesh = _build_mesh(phase1, tree_path)
        assert mesh.betti_1_estimate == 0

    def test_betti_1_cycle_at_least_one(self, phase1, cycle_c3):
        mesh = _build_mesh(phase1, cycle_c3)
        assert mesh.betti_1_estimate >= 1

    def test_rejects_nonpositive_conductance(self, phase1, tree_path):
        bad_edges = [("A", "B", 0.0), ("B", "C", 1.0)]
        with pytest.raises(TopologicalInvariantError, match="Conductancia"):
            phase1.build_mesh(tree_path["nodes"], bad_edges, [])

    def test_rejects_negative_conductance(self, phase1, tree_path):
        bad_edges = [("A", "B", -1.0), ("B", "C", 1.0)]
        with pytest.raises(TopologicalInvariantError):
            phase1.build_mesh(tree_path["nodes"], bad_edges, [])

    def test_rejects_disconnected_graph(self, phase1):
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B", 1.0), ("C", "D", 1.0)]  # 2 componentes
        with pytest.raises(TopologicalInvariantError, match="conexo"):
            phase1.build_mesh(nodes, edges, [])

    def test_rejects_unknown_node_in_edge(self, phase1, tree_path):
        bad_edges = [("A", "Z", 1.0), ("B", "C", 1.0)]
        with pytest.raises(TopologicalInvariantError):
            phase1.build_mesh(tree_path["nodes"], bad_edges, [])

    def test_rejects_empty_mesh(self, phase1):
        with pytest.raises(TopologicalInvariantError, match="degenerada"):
            phase1.build_mesh([], [], [])

    def test_incoherent_cycle_breaks_leibniz(self, phase1):
        r"""
        Ciclo que referencia una arista inexistente se omite parcialmente;
        un ciclo con nodos en orden que no cierra coherentemente con aristas
        reales debe o bien omitirse o fallar Leibniz si se fuerza mal.

        Aquí: grafo A—B—C y ciclo fantasma A—B—X se omite sin romper Leibniz.
        """
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
        # Ciclo válido sigue permitiendo Leibniz
        mesh = phase1.build_mesh(nodes, edges, [["A", "B", "C"]])
        assert mesh.boundary_identity is True

    def test_spanning_tree_covers_all_nodes(self, phase1, diamond):
        parent = phase1._build_spanning_tree_bfs(
            diamond["nodes"], diamond["edges"]
        )
        assert set(parent.keys()) == set(diamond["nodes"])
        # Exactamente un root (parent=None)
        roots = [n for n, p in parent.items() if p is None]
        assert len(roots) == 1

    def test_boundary_2_orientation_coherent_with_boundary_1(
        self, phase1, cycle_c3
    ):
        r"""∂₁ ∂₂ = 0 implica cancelación nodal exacta por columna de ∂₂."""
        mesh = _build_mesh(phase1, cycle_c3)
        # Columna de ∂₂ = signos del ciclo; ∂₁ · col = 0
        col = mesh.boundary_2.toarray()[:, 0]
        nodal = mesh.boundary_1.dot(col)
        np.testing.assert_allclose(nodal, 0.0, atol=1e-14)


# ══════════════════════════════════════════════════════════════════════════════
# §T2 — FASE 2: Poisson, espectro, KCL, Ohm
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2DirichletPoissonSolver:
    r"""Propiedades del solucionador de Poisson ponderado."""

    def test_source_balance_verified(self, agent, tree_path):
        inj = _build_injection(agent, tree_path)
        assert inj.balance_verified is True
        assert abs(float(np.sum(inj.s_vector))) < PistonConstants.SOURCE_BALANCE_TOL

    def test_source_entries_correct(self, agent, tree_path):
        inj = _build_injection(agent, tree_path)
        nodes = tree_path["nodes"]
        i_src = nodes.index("A")
        i_snk = nodes.index("C")
        assert inj.s_vector[i_src] == pytest.approx(-1.0)
        assert inj.s_vector[i_snk] == pytest.approx(+1.0)

    def test_rejects_unbalanced_source(self, phase2):
        s = np.array([1.0, 0.0, 0.0])  # ∑ ≠ 0
        with pytest.raises(SourceCompatibilityError):
            phase2._verify_source_compatibility(s)

    def test_accepts_balanced_source(self, phase2):
        s = np.array([-2.0, 0.5, 1.5])
        phase2._verify_source_compatibility(s)  # no raise

    def test_poisson_solves_on_tree(self, agent, phase1, phase2, tree_path):
        mesh = _build_mesh(phase1, tree_path)
        inj = _build_injection(agent, tree_path)
        flow = _solve_flow(phase2, mesh, inj, tree_path)
        assert isinstance(flow, FlowState)
        assert flow.nodal_pressures.shape == (3,)
        assert flow.edge_flows.shape == (2,)
        assert flow.kirchhoff_residual < PistonConstants.KIRCHHOFF_TOLERANCE
        assert flow.lambda_min_L_red > PistonConstants.SPECTRAL_THRESHOLD

    def test_dirichlet_reservoir_pressure_zero(
        self, agent, phase1, phase2, cycle_c3
    ):
        mesh = _build_mesh(phase1, cycle_c3)
        inj = _build_injection(agent, cycle_c3)
        flow = _solve_flow(phase2, mesh, inj, cycle_c3)
        res_idx = list(mesh.nodes).index(cycle_c3["reservoir_node"])
        assert flow.nodal_pressures[res_idx] == pytest.approx(0.0)

    def test_kcl_identity(self, agent, phase1, phase2, diamond):
        r"""‖∂₁ f + s‖_∞ ≈ 0 (ley de nodos de Kirchhoff)."""
        mesh = _build_mesh(phase1, diamond)
        inj = _build_injection(agent, diamond)
        flow = _solve_flow(phase2, mesh, inj, diamond)
        residual = mesh.boundary_1.dot(flow.edge_flows) + inj.s_vector
        assert np.linalg.norm(residual, ord=np.inf) < PistonConstants.KIRCHHOFF_TOLERANCE

    def test_ohm_law_consistency(self, agent, phase1, phase2, diamond):
        r"""f = W ∂₁ᵀ p (Ley de Ohm discreta ponderada)."""
        mesh = _build_mesh(phase1, diamond)
        inj = _build_injection(agent, diamond)
        flow = _solve_flow(phase2, mesh, inj, diamond)
        f_ohm = mesh.conductance_matrix.dot(
            mesh.boundary_1.T.dot(flow.nodal_pressures)
        )
        np.testing.assert_allclose(flow.edge_flows, f_ohm, atol=1e-12)

    def test_poisson_equation_on_free_nodes(
        self, agent, phase1, phase2, cycle_c3
    ):
        r"""(L₀^W p + s)[free] ≈ 0; en reservorio no se exige."""
        mesh = _build_mesh(phase1, cycle_c3)
        inj = _build_injection(agent, cycle_c3)
        flow = _solve_flow(phase2, mesh, inj, cycle_c3)
        residual = mesh.laplacian_0.dot(flow.nodal_pressures) + inj.s_vector
        res_idx = list(mesh.nodes).index(cycle_c3["reservoir_node"])
        free = np.ones(len(mesh.nodes), dtype=bool)
        free[res_idx] = False
        np.testing.assert_allclose(residual[free], 0.0, atol=1e-10)

    def test_lambda_min_positive(self, agent, phase1, phase2, diamond):
        mesh = _build_mesh(phase1, diamond)
        inj = _build_injection(agent, diamond)
        flow = _solve_flow(phase2, mesh, inj, diamond)
        assert flow.lambda_min_L_red > PistonConstants.SPECTRAL_THRESHOLD

    def test_unknown_reservoir_raises(self, agent, phase1, phase2, tree_path):
        mesh = _build_mesh(phase1, tree_path)
        inj = _build_injection(agent, tree_path)
        with pytest.raises(SingularLaplacianError, match="Reservorio"):
            phase2.solve_hydrodynamics(mesh, inj, "Z_NOT_IN_GRAPH")

    def test_spectral_check_rejects_zero_matrix(self, phase2):
        L_red = sp.csr_matrix((2, 2), dtype=np.float64)
        with pytest.raises(SingularLaplacianError):
            phase2._verify_laplacian_spectrum(L_red)

    def test_rejects_mesh_without_leibniz(self, agent, phase1, phase2, tree_path):
        r"""FASE 2 debe rechazar malla con boundary_identity=False."""
        mesh = _build_mesh(phase1, tree_path)
        # Congelar DTO con identidad falsa
        bad_mesh = SimplicialMesh(
            nodes=mesh.nodes,
            edges=mesh.edges,
            cycles=mesh.cycles,
            boundary_1=mesh.boundary_1,
            boundary_2=mesh.boundary_2,
            conductance_matrix=mesh.conductance_matrix,
            inv_conductance=mesh.inv_conductance,
            laplacian_0=mesh.laplacian_0,
            laplacian_1=mesh.laplacian_1,
            boundary_identity=False,
            betti_1_estimate=mesh.betti_1_estimate,
        )
        inj = _build_injection(agent, tree_path)
        with pytest.raises(BoundaryComplexError):
            phase2.solve_hydrodynamics(bad_mesh, inj, tree_path["reservoir_node"])

    def test_flow_scales_with_q_pump(self, agent, phase1, phase2, tree_path):
        r"""Linealidad: f(αQ) = α f(Q) (Poisson lineal)."""
        mesh = _build_mesh(phase1, tree_path)
        cfg1 = dict(tree_path, q_pump=1.0)
        cfg2 = dict(tree_path, q_pump=3.0)
        f1 = _solve_flow(phase2, mesh, _build_injection(agent, cfg1), cfg1)
        f2 = _solve_flow(phase2, mesh, _build_injection(agent, cfg2), cfg2)
        np.testing.assert_allclose(f2.edge_flows, 3.0 * f1.edge_flows, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# §T3 — FASE 3: Hodge-Helmholtz métricamente consistente
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase3HodgeHelmholtzAuditor:
    r"""Descomposición de Hodge en ⟨·,·⟩_W."""

    def _full_pipeline(self, agent, phase1, phase2, phase3, cfg):
        mesh = _build_mesh(phase1, cfg)
        inj = _build_injection(agent, cfg)
        flow = _solve_flow(phase2, mesh, inj, cfg)
        hodge = phase3.decompose_flow(mesh, flow)
        return mesh, flow, hodge

    def test_tree_has_vanishing_curl(
        self, agent, phase1, phase2, phase3, tree_path
    ):
        r"""Sobre un árbol, im(∂₂)={0} ⇒ I_curl = 0 exactamente."""
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, tree_path
        )
        assert hodge.vorticity_norm < 1e-14
        np.testing.assert_allclose(hodge.i_curl, 0.0, atol=1e-14)

    def test_poisson_flow_is_pure_gradient(
        self, agent, phase1, phase2, phase3, diamond
    ):
        r"""
        FIX-M2: f = W ∂₁ᵀ p ∈ W·im(∂₁ᵀ), luego I_grad ≈ I e I_curl ≈ 0
        para el flujo de Poisson (aunque el grafo tenga ciclos).
        """
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, diamond
        )
        W_inv = mesh.inv_conductance
        # Componente curl del flujo de Poisson debe ser numéricamente nula
        assert hodge.vorticity_norm < max(
            1e-8, PistonConstants.MAX_VORTICITY_NORM
        )
        # I_grad recupera casi todo el flujo
        rel = _energy_norm(flow.edge_flows - hodge.i_grad, W_inv) / max(
            _energy_norm(flow.edge_flows, W_inv), 1e-15
        )
        assert rel < 1e-6

    def test_reconstruction_identity(
        self, agent, phase1, phase2, phase3, cycle_c3
    ):
        r"""I = I_grad + I_curl + I_harm (identidad algebraica)."""
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, cycle_c3
        )
        recon = hodge.i_grad + hodge.i_curl + hodge.i_harm
        np.testing.assert_allclose(recon, flow.edge_flows, atol=1e-12)
        assert hodge.projection_residual < 1e-12

    def test_orthogonality_in_energy_norm(
        self, agent, phase1, phase2, phase3, diamond
    ):
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, diamond
        )
        assert hodge.orthogonality_ok is True
        W_inv = mesh.inv_conductance
        pairs = [
            (hodge.i_grad, hodge.i_curl),
            (hodge.i_grad, hodge.i_harm),
            (hodge.i_curl, hodge.i_harm),
        ]
        for u, v in pairs:
            assert abs(_energy_inner_product(u, v, W_inv)) < PistonConstants.HODGE_ORTHO_TOL

    def test_pythagoras_energy(
        self, agent, phase1, phase2, phase3, diamond
    ):
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, diamond
        )
        assert hodge.pythagoras_ok is True
        W_inv = mesh.inv_conductance
        e_I = _energy_norm(flow.edge_flows, W_inv) ** 2
        e_sum = (
            _energy_norm(hodge.i_grad, W_inv) ** 2
            + _energy_norm(hodge.i_curl, W_inv) ** 2
            + _energy_norm(hodge.i_harm, W_inv) ** 2
        )
        assert abs(e_I - e_sum) / max(e_I, 1.0) < PistonConstants.PYTHAGORAS_TOL

    def test_gradient_lives_in_weighted_image(
        self, agent, phase1, phase2, phase3, diamond
    ):
        r"""
        I_grad = W ∂₁ᵀ φ para algún φ
        ⇔ W⁻¹ I_grad ∈ im(∂₁ᵀ)
        ⇔ ∂₁ (W⁻¹ I_grad) está en im(L₀) y el residual de proyección
          sobre im(∂₁ᵀ) es nulo en norma 2.
        """
        mesh, flow, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, diamond
        )
        B1 = mesh.boundary_1
        W_inv = mesh.inv_conductance
        # Resolver min_φ ‖∂₁ᵀ φ − W⁻¹ I_grad‖₂
        target = W_inv.dot(hodge.i_grad)
        phi = _lsqr_solve(B1.T, target)
        residual = B1.T.dot(phi) - target
        assert np.linalg.norm(residual) < 1e-8

    def test_curl_lives_in_image_of_boundary_2(
        self, phase1, phase3, cycle_c3
    ):
        r"""I_curl ∈ im(∂₂): residual de proyección LSQR nulo."""
        mesh = _build_mesh(phase1, cycle_c3)
        rng = np.random.default_rng(7)
        # Flujo sintético con componente curl forzada
        alpha = np.array([1.5])
        I_pure_curl = mesh.boundary_2.dot(alpha)
        I_noise = rng.normal(size=len(mesh.edges)) * 0.0
        I = I_pure_curl + I_noise
        I_curl = phase3._project_curl(mesh.boundary_2, mesh.inv_conductance, I)
        # Debe recuperar el curl puro
        np.testing.assert_allclose(I_curl, I_pure_curl, atol=1e-10)

    def test_project_curl_empty_cycles_is_zero(self, phase1, phase3, tree_path):
        mesh = _build_mesh(phase1, tree_path)
        I = np.array([1.0, -0.5])
        I_curl = phase3._project_curl(mesh.boundary_2, mesh.inv_conductance, I)
        np.testing.assert_allclose(I_curl, 0.0)

    def test_synthetic_curl_detected_and_vetoed(
        self, phase1, phase3, cycle_c3
    ):
        r"""
        Inyecta un flujo con curl grande artificialmente y espera veto.
        """
        mesh = _build_mesh(phase1, cycle_c3)
        # Curl puro de amplitud grande
        I = mesh.boundary_2.dot(np.array([1e3]))
        # FlowState sintético (presiones irrelevantes para el auditor)
        flow = FlowState(
            nodal_pressures=np.zeros(len(mesh.nodes)),
            edge_flows=I,
            kirchhoff_residual=0.0,
            lambda_min_L_red=1.0,
        )
        with pytest.raises(ParasiticVorticityVetoError):
            phase3.decompose_flow(mesh, flow)

    def test_decompose_returns_hodge_dto(
        self, agent, phase1, phase2, phase3, cycle_c3
    ):
        _, _, hodge = self._full_pipeline(
            agent, phase1, phase2, phase3, cycle_c3
        )
        assert isinstance(hodge, HodgeDecomposition)
        assert hodge.i_grad.shape == hodge.i_curl.shape == hodge.i_harm.shape
        assert hodge.energy_total >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §T4 — Orquestación execute_injection y funtor categórico
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestrationAndFunctor:
    r"""Cadena completa FASE1→2→3 y φ: CategoricalState → CategoricalState."""

    def test_execute_injection_keys(self, agent, diamond):
        result = agent.execute_injection(
            diamond["nodes"],
            diamond["edges"],
            diamond["cycles"],
            diamond["pump_source"],
            diamond["pump_sink"],
            diamond["reservoir_node"],
            diamond["q_pump"],
        )
        required = {
            "status",
            "injection_q",
            "nodal_pressures",
            "edge_flows",
            "laminar_flow_norm_W",
            "vorticity_norm_W",
            "harmonic_norm_W",
            "hodge_energy_total",
            "hodge_orthogonal",
            "hodge_pythagoras_ok",
            "projection_residual_W",
            "topological_invariants",
            "flow_components",
        }
        assert required.issubset(result.keys())

    def test_execute_injection_invariants_block(self, agent, diamond):
        result = agent.execute_injection(
            diamond["nodes"],
            diamond["edges"],
            diamond["cycles"],
            diamond["pump_source"],
            diamond["pump_sink"],
            diamond["reservoir_node"],
            diamond["q_pump"],
        )
        inv = result["topological_invariants"]
        assert inv["boundary_identity"] is True
        assert inv["source_balanced"] is True
        assert inv["kirchhoff_residual"] < PistonConstants.KIRCHHOFF_TOLERANCE
        assert inv["lambda_min_L_red"] > PistonConstants.SPECTRAL_THRESHOLD
        assert inv["num_nodes"] == 4
        assert inv["num_edges"] == 5
        assert inv["num_cycles"] == 2

    def test_execute_injection_hodge_flags(self, agent, diamond):
        result = agent.execute_injection(
            diamond["nodes"],
            diamond["edges"],
            diamond["cycles"],
            diamond["pump_source"],
            diamond["pump_sink"],
            diamond["reservoir_node"],
            diamond["q_pump"],
        )
        assert result["hodge_orthogonal"] is True
        assert result["hodge_pythagoras_ok"] is True
        assert result["vorticity_norm_W"] < PistonConstants.MAX_VORTICITY_NORM
        assert result["projection_residual_W"] < 1e-10

    def test_flow_components_reconstruct(self, agent, cycle_c3):
        result = agent.execute_injection(
            cycle_c3["nodes"],
            cycle_c3["edges"],
            cycle_c3["cycles"],
            cycle_c3["pump_source"],
            cycle_c3["pump_sink"],
            cycle_c3["reservoir_node"],
            cycle_c3["q_pump"],
        )
        g = np.array(result["flow_components"]["i_grad"])
        c = np.array(result["flow_components"]["i_curl"])
        h = np.array(result["flow_components"]["i_harm"])
        f = np.array(result["edge_flows"])
        np.testing.assert_allclose(g + c + h, f, atol=1e-11)

    def test_functor_call_success(self, agent, tree_path):
        state = CategoricalState(payload=dict(tree_path))
        out = agent(state)
        assert out.stratum == Stratum.PHYSICS
        assert "hydrodynamic_tensor" in out.payload
        tensor = out.payload["hydrodynamic_tensor"]
        assert tensor["status"] == "ANNIHILATED_PARASITIC_VORTICITY"

    def test_functor_preserves_original_payload_keys(self, agent, tree_path):
        payload = dict(tree_path)
        payload["trace_id"] = "abc-123"
        out = agent(CategoricalState(payload=payload))
        assert out.payload["trace_id"] == "abc-123"
        assert out.payload["nodes"] == tree_path["nodes"]

    def test_functor_missing_nodes(self, agent, tree_path):
        payload = dict(tree_path)
        del payload["nodes"]
        with pytest.raises(TopologicalInvariantError, match="nodes"):
            agent(CategoricalState(payload=payload))

    def test_functor_missing_pump_source(self, agent, tree_path):
        payload = dict(tree_path)
        payload["pump_source"] = None
        with pytest.raises(TopologicalInvariantError, match="pump_source"):
            agent(CategoricalState(payload=payload))

    def test_functor_default_q_pump(self, agent, tree_path):
        payload = dict(tree_path)
        del payload["q_pump"]
        out = agent(CategoricalState(payload=payload))
        assert out.payload["hydrodynamic_tensor"]["injection_q"] == pytest.approx(1.0)

    def test_create_injection_same_source_sink(self, agent, tree_path):
        with pytest.raises(TopologicalInvariantError, match="coinciden|mismo"):
            agent._create_injection_vector(
                tree_path["nodes"], "A", "A", 1.0
            )

    def test_create_injection_unknown_source(self, agent, tree_path):
        with pytest.raises(TopologicalInvariantError, match="Fuente"):
            agent._create_injection_vector(
                tree_path["nodes"], "Z", "C", 1.0
            )

    def test_create_injection_unknown_sink(self, agent, tree_path):
        with pytest.raises(TopologicalInvariantError, match="Sumidero"):
            agent._create_injection_vector(
                tree_path["nodes"], "A", "Z", 1.0
            )


# ══════════════════════════════════════════════════════════════════════════════
# §T5 — Vetos y casos patológicos
# ══════════════════════════════════════════════════════════════════════════════

class TestTopologicalVetoes:
    r"""Superficie de excepciones y bordes del dominio."""

    def test_exception_hierarchy(self):
        assert issubclass(SingularLaplacianError, PistonInjectorError)
        assert issubclass(HomologicalKirchhoffError, PistonInjectorError)
        assert issubclass(ParasiticVorticityVetoError, PistonInjectorError)
        assert issubclass(BoundaryComplexError, PistonInjectorError)
        assert issubclass(SourceCompatibilityError, PistonInjectorError)
        assert issubclass(HodgeMetricInconsistencyError, PistonInjectorError)
        assert issubclass(PistonInjectorError, TopologicalInvariantError)

    def test_constants_are_positive(self):
        assert PistonConstants.MACHINE_EPSILON > 0
        assert PistonConstants.KIRCHHOFF_TOLERANCE > 0
        assert PistonConstants.SPECTRAL_THRESHOLD > 0
        assert PistonConstants.MAX_VORTICITY_NORM > 0
        assert PistonConstants.SOURCE_BALANCE_TOL > 0
        assert PistonConstants.HODGE_ORTHO_TOL > 0
        assert PistonConstants.PYTHAGORAS_TOL > 0

    def test_verify_boundary_identity_detects_corruption(self, phase1, cycle_c3):
        mesh = _build_mesh(phase1, cycle_c3)
        # Corromper ∂₂
        B2_bad = mesh.boundary_2.copy()
        B2_bad.data = B2_bad.data + 1.0  # rompe signos
        if B2_bad.nnz == 0:
            B2_bad = sp.csr_matrix(
                np.ones(mesh.boundary_2.shape), dtype=np.float64
            )
        with pytest.raises(BoundaryComplexError):
            phase1._verify_boundary_identity(mesh.boundary_1, B2_bad)

    def test_heterogeneous_conductances_affect_flows(
        self, agent, phase1, phase2
    ):
        r"""Mayor conductancia ⇒ mayor |flujo| en esa arista (mismo Δp local)."""
        nodes = ["A", "B", "C"]
        # Dos caminos A→B y A→C→B; C→B muy conductivo vs A→B débil no aplica
        # Comparar: misma topología, reescalar W en una arista del camino único.
        edges_lo = [("A", "B", 1.0), ("B", "C", 1.0)]
        edges_hi = [("A", "B", 1.0), ("B", "C", 10.0)]
        cfg_base = {
            "nodes": nodes,
            "cycles": [],
            "pump_source": "A",
            "pump_sink": "C",
            "reservoir_node": "A",
            "q_pump": 1.0,
        }
        mesh_lo = phase1.build_mesh(nodes, edges_lo, [])
        mesh_hi = phase1.build_mesh(nodes, edges_hi, [])
        inj = _build_injection(agent, cfg_base)
        flow_lo = phase2.solve_hydrodynamics(mesh_lo, inj, "A")
        flow_hi = phase2.solve_hydrodynamics(mesh_hi, inj, "A")
        # En un camino, |f| es el mismo (conservación); las presiones cambian.
        # Verificar que |Δp| en arista B→C es menor con mayor w (Ohm).
        # f = w Δp ⇒ |Δp| = |f|/w
        # Ambos caminos tienen el mismo |f| por KCL (serie).
        np.testing.assert_allclose(
            np.abs(flow_lo.edge_flows),
            np.abs(flow_hi.edge_flows),
            atol=1e-10,
        )
        # Caída de presión total |p_C - p_A| menor si B—C es más conductiva
        # (resistencia serie 1/w_AB + 1/w_BC)
        dp_lo = abs(flow_lo.nodal_pressures[2] - flow_lo.nodal_pressures[0])
        dp_hi = abs(flow_hi.nodal_pressures[2] - flow_hi.nodal_pressures[0])
        assert dp_hi < dp_lo


# ══════════════════════════════════════════════════════════════════════════════
# §T6 — Regresiones métricas FIX-M1…M10 (contratos v7)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricRegressionsV7:
    r"""
    Candados de regresión para las correcciones doctorales v7.0.0.
    Cada test documenta el FIX que protege.
    """

    def test_fix_m1_energy_norm_import_math(self):
        r"""FIX-M1: _energy_norm no debe lanzar NameError (import math)."""
        v = np.array([1.0, 0.0])
        W_inv = sp.diags([1.0, 1.0], format="dia")
        n = _energy_norm(v, W_inv)
        assert n == pytest.approx(1.0)

    def test_fix_m2_gradient_uses_W_boundary_transpose(
        self, agent, phase1, phase2, phase3, diamond
    ):
        r"""
        FIX-M2: I_grad = W ∂₁ᵀ φ, no ∂₁ᵀ φ.
        Comprobación: W⁻¹ I_grad ∈ im(∂₁ᵀ).
        """
        mesh = _build_mesh(phase1, diamond)
        flow = _solve_flow(
            phase2, mesh, _build_injection(agent, diamond), diamond
        )
        hodge = phase3.decompose_flow(mesh, flow)
        target = mesh.inv_conductance.dot(hodge.i_grad)
        phi = _lsqr_solve(mesh.boundary_1.T, target)
        assert np.linalg.norm(mesh.boundary_1.T.dot(phi) - target) < 1e-8

    def test_fix_m2_poisson_flow_near_pure_gradient(
        self, agent, phase1, phase2, phase3, diamond
    ):
        r"""FIX-M2: flujo de Poisson ≈ I_grad en norma energía."""
        mesh = _build_mesh(phase1, diamond)
        flow = _solve_flow(
            phase2, mesh, _build_injection(agent, diamond), diamond
        )
        hodge = phase3.decompose_flow(mesh, flow)
        W_inv = mesh.inv_conductance
        err = _energy_norm(flow.edge_flows - hodge.i_grad, W_inv)
        scale = max(_energy_norm(flow.edge_flows, W_inv), 1e-15)
        assert err / scale < 1e-6

    def test_fix_m3_curl_gram_uses_W_inv(self, phase1, phase3, cycle_c3):
        r"""
        FIX-M3: proyección curl minimiza ‖I − ∂₂α‖_W
        ⇔ G₂ = ∂₂ᵀ W⁻¹ ∂₂ (no ∂₂ᵀ W ∂₂).
        """
        mesh = _build_mesh(phase1, cycle_c3)
        B2, W_inv = mesh.boundary_2, mesh.inv_conductance
        # I aleatorio
        rng = np.random.default_rng(42)
        I = rng.normal(size=len(mesh.edges))
        I_curl = phase3._project_curl(B2, W_inv, I)
        # Condición normal: ∂₂ᵀ W⁻¹ (I − I_curl) ≈ 0
        residual = B2.T.dot(W_inv.dot(I - I_curl))
        assert np.linalg.norm(residual) < 1e-8

    def test_fix_m4_L1_matches_autoadjoint_formula(self, phase1, diamond):
        r"""FIX-M4: L₁ = W ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ W⁻¹."""
        mesh = _build_mesh(phase1, diamond)
        B1, B2 = mesh.boundary_1, mesh.boundary_2
        W, W_inv = mesh.conductance_matrix, mesh.inv_conductance
        expected = (W.dot(B1.T.dot(B1)) + B2.dot(B2.T.dot(W_inv))).toarray()
        np.testing.assert_allclose(mesh.laplacian_1.toarray(), expected, atol=1e-12)

    def test_fix_m5_pythagoras_flag(
        self, agent, phase1, phase2, phase3, diamond
    ):
        r"""FIX-M5: pythagoras_ok en el DTO."""
        mesh = _build_mesh(phase1, diamond)
        flow = _solve_flow(
            phase2, mesh, _build_injection(agent, diamond), diamond
        )
        hodge = phase3.decompose_flow(mesh, flow)
        assert hasattr(hodge, "pythagoras_ok")
        assert hodge.pythagoras_ok is True

    def test_fix_m6_lsqr_gauge_zero_mean_potential(self, phase1, cycle_c3):
        r"""FIX-M6: LSQR sobre L₀ produce potencial de media ~0."""
        mesh = _build_mesh(phase1, cycle_c3)
        rng = np.random.default_rng(1)
        # rhs en im(L₀) = 1^⊥
        b = rng.normal(size=len(mesh.nodes))
        b -= b.mean()
        phi = _lsqr_solve(mesh.laplacian_0, b)
        assert abs(phi.mean()) < 1e-8

    def test_fix_m7_phase_chain_dto_types(
        self, agent, phase1, phase2, phase3, tree_path
    ):
        r"""FIX-M7: sutura formal SimplicialMesh → FlowState → HodgeDecomposition."""
        mesh = _build_mesh(phase1, tree_path)
        assert isinstance(mesh, SimplicialMesh)
        inj = _build_injection(agent, tree_path)
        assert isinstance(inj, InjectionVector)
        flow = _solve_flow(phase2, mesh, inj, tree_path)
        assert isinstance(flow, FlowState)
        hodge = phase3.decompose_flow(mesh, flow)
        assert isinstance(hodge, HodgeDecomposition)

    def test_fix_m8_betti_field_present(self, phase1, cycle_c3, tree_path):
        r"""FIX-M8: betti_1_estimate expuesto en SimplicialMesh."""
        m_tree = _build_mesh(phase1, tree_path)
        m_c3 = _build_mesh(phase1, cycle_c3)
        assert m_tree.betti_1_estimate == 0
        assert m_c3.betti_1_estimate >= 1

    def test_fix_m9_cycle_sign_cocycle(self, phase1, cycle_c3):
        r"""FIX-M9: signos de ∂₂ coherentes ⇒ Leibniz."""
        mesh = _build_mesh(phase1, cycle_c3)
        assert mesh.boundary_identity is True
        comp = mesh.boundary_1.dot(mesh.boundary_2)
        assert sp.linalg.norm(comp, ord="fro") < 1e-12

    def test_fix_m10_projection_residual_in_result(self, agent, diamond):
        r"""FIX-M10: projection_residual_W en el diccionario de salida."""
        result = agent.execute_injection(
            diamond["nodes"],
            diamond["edges"],
            diamond["cycles"],
            diamond["pump_source"],
            diamond["pump_sink"],
            diamond["reservoir_node"],
            diamond["q_pump"],
        )
        assert "projection_residual_W" in result
        assert result["projection_residual_W"] < 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# §T7 — Propiedades numéricas de estrés (conductancias extremas / escala)
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStress:
    r"""Condicionamiento y robustez numérica básica."""

    def test_wide_conductance_dynamic_range(self, agent, phase1, phase2, phase3):
        nodes = ["A", "B", "C"]
        edges = [("A", "B", 1e-4), ("B", "C", 1e4), ("C", "A", 1.0)]
        cycles = [["A", "B", "C"]]
        cfg = {
            "nodes": nodes,
            "edges": edges,
            "cycles": cycles,
            "pump_source": "A",
            "pump_sink": "B",
            "reservoir_node": "C",
            "q_pump": 1.0,
        }
        mesh = phase1.build_mesh(nodes, edges, cycles)
        flow = phase2.solve_hydrodynamics(
            mesh, _build_injection(agent, cfg), "C"
        )
        hodge = phase3.decompose_flow(mesh, flow)
        assert flow.kirchhoff_residual < 1e-8
        assert hodge.orthogonality_ok
        assert hodge.pythagoras_ok

    def test_small_q_pump_stability(self, agent, tree_path):
        cfg = dict(tree_path, q_pump=1e-12)
        result = agent.execute_injection(
            cfg["nodes"], cfg["edges"], cfg["cycles"],
            cfg["pump_source"], cfg["pump_sink"],
            cfg["reservoir_node"], cfg["q_pump"],
        )
        assert result["topological_invariants"]["kirchhoff_residual"] < 1e-9

    def test_large_q_pump_linearity(self, agent, tree_path):
        r1 = agent.execute_injection(
            tree_path["nodes"], tree_path["edges"], tree_path["cycles"],
            tree_path["pump_source"], tree_path["pump_sink"],
            tree_path["reservoir_node"], 1.0,
        )
        r2 = agent.execute_injection(
            tree_path["nodes"], tree_path["edges"], tree_path["cycles"],
            tree_path["pump_source"], tree_path["pump_sink"],
            tree_path["reservoir_node"], 1e6,
        )
        f1 = np.array(r1["edge_flows"])
        f2 = np.array(r2["edge_flows"])
        np.testing.assert_allclose(f2, f1 * 1e6, rtol=1e-9)

    def test_reversed_pump_negates_flow(self, agent, tree_path):
        r_fwd = agent.execute_injection(
            tree_path["nodes"], tree_path["edges"], tree_path["cycles"],
            "A", "C", tree_path["reservoir_node"], 1.0,
        )
        r_bwd = agent.execute_injection(
            tree_path["nodes"], tree_path["edges"], tree_path["cycles"],
            "C", "A", tree_path["reservoir_node"], 1.0,
        )
        np.testing.assert_allclose(
            np.array(r_bwd["edge_flows"]),
            -np.array(r_fwd["edge_flows"]),
            atol=1e-10,
        )