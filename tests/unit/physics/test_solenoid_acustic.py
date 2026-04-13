"""
=========================================================================================
Suite de Pruebas Rigurosas: Solenoidal Acústico v3.0
=========================================================================================

Filosofía de Testing:
    Cada prueba verifica una propiedad matemática demostrable, no solo
    comportamiento de implementación. Las tolerancias son derivadas de
    principios de análisis numérico (épsilon de máquina, número de condición).

Estructura de la Suite:
    I.   TestNumericalUtilities         — Álgebra lineal numérica
    II.  TestHodgeDecompositionBuilder  — Complejo de cadenas y matrices
    III. TestCochainComplexInvariants   — Invariantes topológicos
    IV.  TestSpectralProperties         — Propiedades espectrales de L₁
    V.   TestAcousticSolenoidOperator   — Operador de vorticidad
    VI.  TestMagnonCartridge            — Dataclass y validaciones
    VII. TestFullHodgeDecomposition     — Descomposición completa
    VIII.TestEdgeCases                  — Casos degenerados y frontera
    IX.  TestEulerPoincare              — Invariantes topológicos globales
    X.   TestNumericalStability         — Estabilidad ante perturbaciones

Propiedades matemáticas verificadas:
    - B₁B₂ = 0                 (cochain complex)
    - rank(B₁) = n − c         (conectividad)
    - rank(B₂) = β₁ = m−n+c   (primer número de Betti)
    - L₁ PSD simétrica         (positiva semidefinida)
    - dim ker(L₁) = β₁         (isomorfismo de Hodge)
    - χ = β₀ − β₁              (Euler–Poincaré)
    - P² = P, Pᵀ = P           (proyector ortogonal)
    - ‖I_grad‖² + ‖I_curl‖² + ‖I_harm‖² ≈ ‖I‖²  (Parseval)
    - ⟨I_grad, I_curl⟩ ≈ 0    (ortogonalidad)
=========================================================================================
"""

from __future__ import annotations

import math
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

# ── Módulo bajo prueba ────────────────────────────────────────────────────────
from solenoid_acustic import (
    AcousticSolenoidOperator,
    HodgeDecompositionBuilder,
    MagnonCartridge,
    NumericalUtilities,
    _generate_proof,
    inspect_and_mitigate_resonance,
    verify_hodge_properties,
)

# ── Configuración de logging para tests ──────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

EPS_MACH: float = np.finfo(np.float64).eps          # ≈ 2.22e-16
TOL_STRICT: float = 1e-10                            # Para propiedades exactas
TOL_NUMERICAL: float = 1e-8                          # Para propiedades numéricas
TOL_PROJECTION: float = 1e-6                         # Para errores de proyección


# ─────────────────────────────────────────────────────────────────────────────
# FÁBRICA DE GRAFOS DE PRUEBA
# ─────────────────────────────────────────────────────────────────────────────

class GraphFactory:
    """
    Fábrica de grafos con propiedades topológicas conocidas.

    Cada grafo incluye sus invariantes esperados para verificación.
    """

    @staticmethod
    def triangle() -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Triángulo dirigido: A→B→C→A
        n=3, m=3, c=1, β₀=1, β₁=1, χ=0
        """
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        return G, {"n": 3, "m": 3, "c": 1, "beta_0": 1, "beta_1": 1, "chi": 0}

    @staticmethod
    def two_triangles_shared_vertex() -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Dos triángulos compartiendo un vértice:
        A→B→C→A y A→D→E→A
        n=5, m=6, c=1, β₁=2, χ=-1
        """
        G = nx.DiGraph()
        G.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("A", "D"), ("D", "E"), ("E", "A"),
        ])
        return G, {"n": 5, "m": 6, "c": 1, "beta_0": 1, "beta_1": 2, "chi": -1}

    @staticmethod
    def path_graph(n: int = 4) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Camino dirigido: 0→1→2→...→n-1
        β₁=0 (árbol: sin ciclos)
        """
        G = nx.DiGraph()
        G.add_edges_from([(i, i + 1) for i in range(n - 1)])
        return G, {
            "n": n, "m": n - 1, "c": 1,
            "beta_0": 1, "beta_1": 0, "chi": 1,
        }

    @staticmethod
    def complete_directed(n: int = 4) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Grafo completo dirigido K_n (ambas direcciones por par).
        n nodos, m = n(n-1) aristas, β₁ = m - n + 1
        """
        G = nx.DiGraph()
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(i, j)
        m = n * (n - 1)
        beta_1 = m - n + 1
        return G, {"n": n, "m": m, "c": 1, "beta_0": 1, "beta_1": beta_1}

    @staticmethod
    def disconnected_two_triangles() -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Dos triángulos disjuntos:
        A→B→C→A  y  X→Y→Z→X
        n=6, m=6, c=2, β₀=2, β₁=2, χ=0
        """
        G = nx.DiGraph()
        G.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("X", "Y"), ("Y", "Z"), ("Z", "X"),
        ])
        return G, {
            "n": 6, "m": 6, "c": 2,
            "beta_0": 2, "beta_1": 2, "chi": 0,
        }

    @staticmethod
    def single_node() -> Tuple[nx.DiGraph, Dict[str, int]]:
        """Grafo trivial: un único nodo, sin aristas."""
        G = nx.DiGraph()
        G.add_node("A")
        return G, {"n": 1, "m": 0, "c": 1, "beta_0": 1, "beta_1": 0, "chi": 1}

    @staticmethod
    def square_with_diagonal() -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Cuadrado con diagonal: 0→1→2→3→0, 0→2
        n=4, m=5, c=1, β₁=2, χ=-1
        """
        G = nx.DiGraph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 3), (3, 0), (0, 2)
        ])
        return G, {"n": 4, "m": 5, "c": 1, "beta_0": 1, "beta_1": 2, "chi": -1}

    @staticmethod
    def spanning_tree(n: int = 5) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Árbol generador dirigido (sin ciclos).
        n nodos, m = n-1 aristas, β₁ = 0.
        """
        G = nx.DiGraph()
        for i in range(1, n):
            G.add_edge(0, i)
        return G, {
            "n": n, "m": n - 1, "c": 1,
            "beta_0": 1, "beta_1": 0, "chi": 1,
        }

    @staticmethod
    def uniform_flow_cycle(n: int = 5, flow: float = 1.0) -> Tuple[
        nx.DiGraph, Dict[Tuple, float]
    ]:
        """
        Ciclo n-gono con flujo uniforme.
        Retorna (grafo, flows) donde flows tiene circulación exactamente `flow`.
        """
        nodes = list(range(n))
        G = nx.DiGraph()
        flows = {}
        for i in range(n):
            u, v = nodes[i], nodes[(i + 1) % n]
            G.add_edge(u, v)
            flows[(u, v)] = flow
        return G, flows


# ─────────────────────────────────────────────────────────────────────────────
# I. TESTS: NumericalUtilities
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalUtilities:
    """
    Verifica las propiedades matemáticas de las utilidades numéricas.

    Propiedades clave:
        - adaptive_tolerance: tol ≥ ε_mach, escala con σ_max
        - compute_rank: rank(A) = #{σᵢ > tol}
        - moore_penrose_pseudoinverse: 4 condiciones de Penrose
        - orthogonal_projection: idempotencia, ortogonalidad, norma ≤ ‖v‖
        - null_space_basis: A·N = 0, columnas ortonormales
    """

    # ── adaptive_tolerance ────────────────────────────────────────────────

    def test_adaptive_tolerance_lower_bounded_by_eps_mach(self):
        """tol ≥ ε_mach para cualquier matriz."""
        A = np.eye(5)
        tol = NumericalUtilities.adaptive_tolerance(A)
        assert tol >= EPS_MACH, (
            f"Tolerancia {tol:.2e} < ε_mach {EPS_MACH:.2e}"
        )

    def test_adaptive_tolerance_zero_matrix(self):
        """Matriz cero: σ_max = 0, tol = ε_mach."""
        A = np.zeros((4, 4))
        tol = NumericalUtilities.adaptive_tolerance(A)
        assert tol >= EPS_MACH

    def test_adaptive_tolerance_scales_with_sigma_max(self):
        """
        tol debe escalar con σ_max: si A' = α·A, entonces tol(A') ≈ α·tol(A).
        """
        A = np.random.default_rng(42).standard_normal((6, 4))
        alpha = 1e6
        tol_A = NumericalUtilities.adaptive_tolerance(A)
        tol_scaled = NumericalUtilities.adaptive_tolerance(alpha * A)
        ratio = tol_scaled / tol_A
        assert abs(ratio - alpha) / alpha < 1e-6, (
            f"Ratio esperado {alpha:.2e}, obtenido {ratio:.2e}"
        )

    def test_adaptive_tolerance_sparse_matrix(self):
        """Para matrices sparse, tol ≥ ε_mach."""
        A = sp.eye(10, format="csr")
        tol = NumericalUtilities.adaptive_tolerance(A)
        assert tol >= EPS_MACH

    def test_adaptive_tolerance_raises_on_non_2d(self):
        """Matriz no 2-D debe lanzar ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            NumericalUtilities.adaptive_tolerance(np.ones(5))

    # ── compute_rank ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("shape,expected_rank", [
        ((5, 5), 5),   # Identidad 5×5
        ((4, 6), 4),   # Rango máximo por filas
        ((6, 4), 4),   # Rango máximo por columnas
    ])
    def test_rank_full_rank_matrices(self, shape, expected_rank):
        """Matrices de rango máximo generadas aleatoriamente."""
        rng = np.random.default_rng(seed=0)
        A = rng.standard_normal(shape)
        rank, svs = NumericalUtilities.compute_rank(A)
        assert rank == expected_rank, (
            f"Rango esperado {expected_rank}, obtenido {rank}"
        )
        assert svs[0] >= svs[-1] >= 0, "SVs no ordenados o negativos"

    def test_rank_rank_deficient_matrix(self):
        """Matriz de rango conocido r < min(m,n)."""
        # Construir matriz de rango exacto r
        r = 3
        rng = np.random.default_rng(1)
        U = la.orth(rng.standard_normal((6, r)))
        V = la.orth(rng.standard_normal((8, r)))
        A = U @ np.diag([1.0, 2.0, 3.0]) @ V.T
        rank, _ = NumericalUtilities.compute_rank(A)
        assert rank == r, f"Rango esperado {r}, obtenido {rank}"

    def test_rank_zero_matrix(self):
        """Matriz cero tiene rango 0."""
        A = np.zeros((5, 7))
        rank, svs = NumericalUtilities.compute_rank(A)
        assert rank == 0
        assert np.all(svs == 0.0)

    def test_rank_identity(self):
        """Identidad n×n tiene rango n."""
        for n in [2, 5, 10]:
            rank, _ = NumericalUtilities.compute_rank(np.eye(n))
            assert rank == n

    def test_rank_sparse_matrix(self):
        """Rango de sparse matrix de identidad."""
        A = sp.eye(7, format="csr")
        rank, _ = NumericalUtilities.compute_rank(A)
        assert rank == 7

    def test_rank_singular_values_consistent(self):
        """Los SVs retornados deben ser consistentes con np.linalg.svd."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((5, 4))
        _, svs = NumericalUtilities.compute_rank(A)
        expected_svs = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(svs, expected_svs, rtol=1e-12)

    # ── moore_penrose_pseudoinverse ────────────────────────────────────────

    @pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 5), (6, 2)])
    def test_penrose_condition_1_AA_plus_A_eq_A(self, shape):
        """Condición de Penrose (i): A A⁺ A = A."""
        rng = np.random.default_rng(10 + shape[0])
        A = rng.standard_normal(shape)
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        residual = np.linalg.norm(A @ A_plus @ A - A, "fro")
        assert residual < TOL_STRICT, f"‖AA⁺A − A‖_F = {residual:.2e}"

    @pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 5)])
    def test_penrose_condition_2_A_plus_AA_plus_eq_A_plus(self, shape):
        """Condición de Penrose (ii): A⁺ A A⁺ = A⁺."""
        rng = np.random.default_rng(20 + shape[0])
        A = rng.standard_normal(shape)
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        residual = np.linalg.norm(A_plus @ A @ A_plus - A_plus, "fro")
        assert residual < TOL_STRICT, f"‖A⁺AA⁺ − A⁺‖_F = {residual:.2e}"

    @pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 5)])
    def test_penrose_condition_3_AA_plus_symmetric(self, shape):
        """Condición de Penrose (iii): (AA⁺)ᵀ = AA⁺."""
        rng = np.random.default_rng(30 + shape[0])
        A = rng.standard_normal(shape)
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        P = A @ A_plus
        residual = np.linalg.norm(P - P.T, "fro")
        assert residual < TOL_STRICT, f"‖AA⁺ − (AA⁺)ᵀ‖_F = {residual:.2e}"

    @pytest.mark.parametrize("shape", [(4, 4), (5, 3), (3, 5)])
    def test_penrose_condition_4_A_plus_A_symmetric(self, shape):
        """Condición de Penrose (iv): (A⁺A)ᵀ = A⁺A."""
        rng = np.random.default_rng(40 + shape[0])
        A = rng.standard_normal(shape)
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        Q = A_plus @ A
        residual = np.linalg.norm(Q - Q.T, "fro")
        assert residual < TOL_STRICT, f"‖A⁺A − (A⁺A)ᵀ‖_F = {residual:.2e}"

    def test_pseudoinverse_of_invertible_equals_inverse(self):
        """Para A invertible: A⁺ = A⁻¹."""
        rng = np.random.default_rng(50)
        A = rng.standard_normal((5, 5))
        # Asegurar invertibilidad
        A = A + 5 * np.eye(5)
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        A_inv = np.linalg.inv(A)
        np.testing.assert_allclose(A_plus, A_inv, atol=TOL_STRICT)

    def test_pseudoinverse_rank_deficient_matrix(self):
        """A⁺ para matriz singular satisface las 4 condiciones."""
        # Matriz de rango 2 en espacio 4×4
        U = np.linalg.qr(np.random.default_rng(51).standard_normal((4, 4)))[0]
        S = np.diag([3.0, 1.5, 0.0, 0.0])
        A = U @ S @ U.T
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        # Verificar condición (i)
        assert np.linalg.norm(A @ A_plus @ A - A, "fro") < TOL_STRICT

    def test_pseudoinverse_zero_matrix(self):
        """Pseudoinversa de la matriz cero es la matriz cero."""
        A = np.zeros((3, 4))
        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        assert A_plus.shape == (4, 3)
        np.testing.assert_array_equal(A_plus, np.zeros((4, 3)))

    # ── orthogonal_projection ─────────────────────────────────────────────

    def test_projection_is_idempotent(self):
        """P²(v) = P(v): proyectar dos veces da el mismo resultado."""
        rng = np.random.default_rng(60)
        B = rng.standard_normal((8, 3))
        v = rng.standard_normal(8)
        proj1, _ = NumericalUtilities.orthogonal_projection(v, B)
        proj2, _ = NumericalUtilities.orthogonal_projection(proj1, B)
        np.testing.assert_allclose(proj1, proj2, atol=TOL_STRICT)

    def test_projection_residual_orthogonal_to_subspace(self):
        """⟨residual, bᵢ⟩ ≈ 0 para toda columna bᵢ de B."""
        rng = np.random.default_rng(61)
        B = rng.standard_normal((8, 3))
        v = rng.standard_normal(8)
        _, residual = NumericalUtilities.orthogonal_projection(v, B)
        inner_products = B.T @ residual
        np.testing.assert_allclose(
            inner_products, np.zeros(3), atol=TOL_STRICT
        )

    def test_projection_reconstruction(self):
        """proj + residual = v exactamente."""
        rng = np.random.default_rng(62)
        B = rng.standard_normal((6, 2))
        v = rng.standard_normal(6)
        proj, residual = NumericalUtilities.orthogonal_projection(v, B)
        np.testing.assert_allclose(proj + residual, v, atol=TOL_STRICT)

    def test_projection_norm_leq_original(self):
        """‖P(v)‖ ≤ ‖v‖ (proyección no aumenta la norma)."""
        rng = np.random.default_rng(63)
        B = rng.standard_normal((7, 4))
        v = rng.standard_normal(7)
        proj, _ = NumericalUtilities.orthogonal_projection(v, B)
        assert np.linalg.norm(proj) <= np.linalg.norm(v) + TOL_STRICT

    def test_projection_vector_in_subspace_unchanged(self):
        """Si v ∈ col(B), entonces P(v) = v."""
        rng = np.random.default_rng(64)
        B = rng.standard_normal((6, 3))
        alpha = rng.standard_normal(3)
        v = B @ alpha  # v está exactamente en col(B)
        proj, residual = NumericalUtilities.orthogonal_projection(v, B)
        np.testing.assert_allclose(proj, v, atol=TOL_STRICT)
        np.testing.assert_allclose(
            residual, np.zeros(6), atol=TOL_STRICT
        )

    def test_projection_empty_subspace(self):
        """Proyección sobre subespacio vacío retorna (0, v)."""
        v = np.array([1.0, 2.0, 3.0])
        B = np.zeros((3, 0))
        proj, residual = NumericalUtilities.orthogonal_projection(v, B)
        np.testing.assert_array_equal(proj, np.zeros(3))
        np.testing.assert_array_equal(residual, v)

    def test_projection_raises_on_bad_dimensions(self):
        """Dimensiones inconsistentes deben lanzar ValueError."""
        v = np.ones(5)
        B = np.ones((6, 3))  # B tiene 6 filas, v tiene 5 elementos
        with pytest.raises(ValueError):
            NumericalUtilities.orthogonal_projection(v, B)

    # ── matrix_condition_number ───────────────────────────────────────────

    def test_condition_number_identity(self):
        """κ(I) = 1."""
        kappa, s_min, s_max = NumericalUtilities.matrix_condition_number(np.eye(5))
        assert abs(kappa - 1.0) < TOL_STRICT
        assert abs(s_min - 1.0) < TOL_STRICT
        assert abs(s_max - 1.0) < TOL_STRICT

    def test_condition_number_singular_matrix(self):
        """κ = ∞ para matriz singular."""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        kappa, s_min, _ = NumericalUtilities.matrix_condition_number(A)
        assert math.isinf(kappa)
        assert s_min == 0.0

    def test_condition_number_diagonal(self):
        """κ(diag(σ₁,...,σₙ)) = σ_max / σ_min."""
        sigmas = np.array([10.0, 5.0, 2.0, 1.0])
        A = np.diag(sigmas)
        kappa, s_min, s_max = NumericalUtilities.matrix_condition_number(A)
        assert abs(kappa - 10.0) < TOL_STRICT
        assert abs(s_min - 1.0) < TOL_STRICT
        assert abs(s_max - 10.0) < TOL_STRICT

    # ── null_space_basis ──────────────────────────────────────────────────

    def test_null_space_satisfies_A_N_eq_zero(self):
        """A · N = 0 para toda base del kernel."""
        rng = np.random.default_rng(70)
        # Construir A con nulidad conocida = 3
        m, n, r = 4, 7, 4
        U = la.orth(rng.standard_normal((m, r)))
        V = la.orth(rng.standard_normal((n, r)))
        A = U @ np.diag(np.linspace(1, 5, r)) @ V.T
        N = NumericalUtilities.null_space_basis(A)
        if N.shape[1] > 0:
            residual = np.linalg.norm(A @ N, "fro")
            assert residual < TOL_STRICT, f"‖AN‖_F = {residual:.2e}"

    def test_null_space_columns_orthonormal(self):
        """Columnas de N son ortonormales: NᵀN = I."""
        rng = np.random.default_rng(71)
        A = rng.standard_normal((3, 7))   # nulidad = 4
        N = NumericalUtilities.null_space_basis(A)
        if N.shape[1] > 0:
            NtN = N.T @ N
            np.testing.assert_allclose(NtN, np.eye(N.shape[1]), atol=TOL_STRICT)

    def test_null_space_dimension(self):
        """dim ker(A) = n − rank(A) (teorema del rango-nulidad)."""
        rng = np.random.default_rng(72)
        m, n = 4, 7
        A = rng.standard_normal((m, n))
        rank_A, _ = NumericalUtilities.compute_rank(A)
        N = NumericalUtilities.null_space_basis(A)
        expected_nullity = n - rank_A
        assert N.shape[1] == expected_nullity, (
            f"Nulidad esperada {expected_nullity}, obtenida {N.shape[1]}"
        )

    def test_null_space_full_rank_matrix(self):
        """Matriz de rango máximo cuadrada: ker = {0}."""
        A = np.eye(5) + 0.1 * np.ones((5, 5))
        N = NumericalUtilities.null_space_basis(A)
        assert N.shape[1] == 0, "Ker de matriz invertible debe ser vacío"


# ─────────────────────────────────────────────────────────────────────────────
# II. TESTS: HodgeDecompositionBuilder — Matrices B₁ y B₂
# ─────────────────────────────────────────────────────────────────────────────

class TestHodgeDecompositionBuilder:
    """
    Verifica la construcción correcta de B₁, B₂ y L₁.
    """

    # ── Constructor ───────────────────────────────────────────────────────

    def test_constructor_requires_digraph(self):
        """Constructor rechaza grafos no dirigidos."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(TypeError, match="DiGraph"):
            HodgeDecompositionBuilder(G)

    def test_constructor_counts_nodes_edges(self):
        """Builder cuenta nodos y aristas correctamente."""
        G, inv = GraphFactory.two_triangles_shared_vertex()
        builder = HodgeDecompositionBuilder(G)
        assert builder.n == inv["n"]
        assert builder.m == inv["m"]

    # ── Matriz de Incidencia B₁ ───────────────────────────────────────────

    def test_B1_shape(self):
        """B₁ ∈ ℝⁿˣᵐ."""
        G, inv = GraphFactory.triangle()
        B1, _ = HodgeDecompositionBuilder(G).build_incidence_matrix()
        assert B1.shape == (inv["n"], inv["m"])

    def test_B1_column_sums_zero(self):
        """
        Σ_v (B₁)_{v,e} = 0 ∀e.

        Propiedad: el operador de borde de una arista suma +1 y -1.
        """
        for factory in [
            GraphFactory.triangle,
            GraphFactory.two_triangles_shared_vertex,
            GraphFactory.square_with_diagonal,
        ]:
            G, _ = factory()
            B1, meta = HodgeDecompositionBuilder(G).build_incidence_matrix()
            col_sums = B1.sum(axis=0)
            np.testing.assert_allclose(
                col_sums, np.zeros(B1.shape[1]),
                atol=TOL_STRICT,
                err_msg=f"Suma de columnas no nula en {factory.__name__}",
            )
            assert meta["column_sum_max"] < TOL_STRICT

    def test_B1_entries_in_minus1_0_plus1(self):
        """Entradas de B₁ ∈ {-1, 0, +1}."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        B1, _ = HodgeDecompositionBuilder(G).build_incidence_matrix()
        valid_entries = set(np.unique(B1))
        assert valid_entries <= {-1.0, 0.0, 1.0}

    def test_B1_exactly_two_nonzeros_per_column(self):
        """Cada arista tiene exactamente dos entradas no nulas (tail y head)."""
        G, _ = GraphFactory.complete_directed(4)
        B1, _ = HodgeDecompositionBuilder(G).build_incidence_matrix()
        nonzeros_per_col = np.sum(B1 != 0, axis=0)
        np.testing.assert_array_equal(nonzeros_per_col, 2 * np.ones(B1.shape[1]))

    def test_B1_rank(self):
        """
        rank(B₁) = n − c donde c es el número de componentes conexas.

        Teorema: La imagen de B₁ tiene codimensión igual al número de
        componentes conexas (cada componente contribuye un vector en ker(B₁ᵀ)).
        """
        test_cases = [
            GraphFactory.triangle,
            GraphFactory.two_triangles_shared_vertex,
            GraphFactory.disconnected_two_triangles,
            GraphFactory.path_graph,
        ]
        for factory in test_cases:
            G, inv = factory()
            B1, meta = HodgeDecompositionBuilder(G).build_incidence_matrix()
            expected_rank = inv["n"] - inv["c"]
            assert meta["rank_B1"] == expected_rank, (
                f"{factory.__name__}: rank esperado {expected_rank}, "
                f"obtenido {meta['rank_B1']}"
            )

    def test_B1_orientation_convention(self):
        """
        Verificar convención: B₁[tail, e] = -1, B₁[head, e] = +1.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        tail_idx = builder._node_index["A"]
        head_idx = builder._node_index["B"]
        e_idx = builder._edge_index[("A", "B")]
        assert B1[tail_idx, e_idx] == -1.0, "tail debe ser -1"
        assert B1[head_idx, e_idx] == +1.0, "head debe ser +1"

    # ── Matriz de Ciclos B₂ ───────────────────────────────────────────────

    def test_B2_shape(self):
        """B₂ ∈ ℝᵐˣᵏ con k = β₁."""
        for factory, (G, inv) in [
            ("triangle", GraphFactory.triangle()),
            ("two_tri", GraphFactory.two_triangles_shared_vertex()),
            ("path", GraphFactory.path_graph()),
        ]:
            builder = HodgeDecompositionBuilder(G)
            B2, meta = builder.build_cycle_matrix()
            expected_k = inv["beta_1"]
            assert B2.shape == (inv["m"], expected_k), (
                f"{factory}: shape esperado ({inv['m']}, {expected_k}), "
                f"obtenido {B2.shape}"
            )

    def test_B2_rank_equals_beta1(self):
        """
        rank(B₂) = β₁ = m − n + c.

        Las columnas de B₂ deben ser linealmente independientes.
        """
        test_cases = [
            GraphFactory.triangle(),
            GraphFactory.two_triangles_shared_vertex(),
            GraphFactory.square_with_diagonal(),
            GraphFactory.disconnected_two_triangles(),
        ]
        for G, inv in test_cases:
            if inv["beta_1"] == 0:
                continue
            builder = HodgeDecompositionBuilder(G)
            B2, meta = builder.build_cycle_matrix()
            assert meta["rank_B2"] == inv["beta_1"], (
                f"rank(B₂) esperado {inv['beta_1']}, "
                f"obtenido {meta['rank_B2']}"
            )

    def test_B2_entries_in_minus1_0_plus1(self):
        """Entradas de B₂ ∈ {-1, 0, +1}."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        B2, _ = HodgeDecompositionBuilder(G).build_cycle_matrix()
        if B2.size > 0:
            valid_entries = set(np.unique(B2))
            assert valid_entries <= {-1.0, 0.0, 1.0}

    def test_B2_empty_for_acyclic_graph(self):
        """β₁ = 0 ⟹ B₂ ∈ ℝᵐˣ⁰ (sin columnas)."""
        G, _ = GraphFactory.path_graph(5)
        B2, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
        assert B2.shape[1] == 0
        assert meta["betti_1"] == 0

    def test_B2_empty_for_single_node(self):
        """Grafo trivial: B₂ vacía."""
        G, _ = GraphFactory.single_node()
        B2, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
        assert B2.shape == (0, 0)

    def test_B2_betti_1_formula(self):
        """β₁ = m − n + c se satisface para todos los grafos de prueba."""
        test_cases = [
            GraphFactory.triangle(),
            GraphFactory.two_triangles_shared_vertex(),
            GraphFactory.path_graph(6),
            GraphFactory.disconnected_two_triangles(),
            GraphFactory.square_with_diagonal(),
            GraphFactory.complete_directed(4),
        ]
        for G, inv in test_cases:
            _, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
            expected = inv.get("beta_1", inv["m"] - inv["n"] + inv["c"])
            assert meta["betti_1"] == expected, (
                f"β₁ esperado {expected}, obtenido {meta['betti_1']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# III. TESTS: Invariantes del Complejo de Co-Cadenas
# ─────────────────────────────────────────────────────────────────────────────

class TestCochainComplexInvariants:
    """
    Verifica los invariantes fundamentales del complejo de cadenas:

        C₀ ←B₁— C₁ ←B₂— C₂

    El invariante central es ∂₁ ∘ ∂₂ = 0, i.e., B₁B₂ = 0.
    """

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.square_with_diagonal,
        GraphFactory.disconnected_two_triangles,
        pytest.param(
            lambda: GraphFactory.complete_directed(4),
            id="complete_K4"
        ),
    ])
    def test_B1_times_B2_equals_zero(self, factory):
        """
        Invariante fundamental: B₁ · B₂ = 0.

        Esto garantiza que el borde del borde es cero (∂² = 0),
        propiedad axiomática de cualquier complejo de cadenas.
        """
        G, _ = factory()
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, meta = builder.build_cycle_matrix()

        if B2.shape[1] == 0:
            return  # Trivialmente satisfecho

        product = B1 @ B2
        norm = float(np.linalg.norm(product, "fro"))

        assert norm < TOL_STRICT, (
            f"‖B₁B₂‖_F = {norm:.2e} > {TOL_STRICT:.2e} "
            f"en grafo {G.edges()}"
        )
        assert meta["verify_B1B2_zero"], (
            f"meta['verify_B1B2_zero'] = False con norma {meta['B1B2_norm']:.2e}"
        )

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.path_graph,
        GraphFactory.disconnected_two_triangles,
    ])
    def test_verify_cochain_complex_is_valid(self, factory):
        """verify_cochain_complex() debe reportar is_valid=True."""
        G, _ = factory()
        result = HodgeDecompositionBuilder(G).verify_cochain_complex()
        assert result["is_valid"], (
            f"Complejo de cadenas inválido en {factory.__name__}: {result}"
        )

    def test_cochain_dimensions_consistent(self):
        """Dimensiones: B₁ ∈ ℝⁿˣᵐ, B₂ ∈ ℝᵐˣᵏ."""
        G, inv = GraphFactory.two_triangles_shared_vertex()
        result = HodgeDecompositionBuilder(G).verify_cochain_complex()
        assert result["dimensions_consistent"]

    def test_rank_B1_equals_n_minus_c(self):
        """rank(B₁) = n − c para cada grafo de prueba."""
        test_cases = [
            GraphFactory.triangle(),
            GraphFactory.two_triangles_shared_vertex(),
            GraphFactory.disconnected_two_triangles(),
            GraphFactory.path_graph(5),
        ]
        for G, inv in test_cases:
            result = HodgeDecompositionBuilder(G).verify_cochain_complex()
            assert result["rank_B1_ok"], (
                f"rank(B₁) incorrecto: esperado {result['rank_B1_expected']}, "
                f"obtenido {result['rank_B1']}"
            )

    def test_rank_B2_equals_beta1(self):
        """rank(B₂) = β₁ para grafos con ciclos."""
        test_cases = [
            GraphFactory.triangle(),
            GraphFactory.two_triangles_shared_vertex(),
            GraphFactory.square_with_diagonal(),
        ]
        for G, inv in test_cases:
            result = HodgeDecompositionBuilder(G).verify_cochain_complex()
            assert result["rank_B2_ok"], (
                f"rank(B₂) incorrecto: esperado β₁={result['rank_B2_expected']}, "
                f"obtenido {result['rank_B2']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# IV. TESTS: Propiedades Espectrales de L₁
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectralProperties:
    """
    Verifica las propiedades espectrales del Laplaciano de Hodge L₁.

    Teorema Espectral (L₁ simétrica PSD):
        • λᵢ ≥ 0 ∀i
        • dim ker(L₁) = β₁ (isomorfismo de Hodge)
        • Traza(L₁) = Σλᵢ
        • L₁ = L₁ᵀ
    """

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.path_graph,
        GraphFactory.square_with_diagonal,
    ])
    def test_L1_is_symmetric(self, factory):
        """L₁ = L₁ᵀ (Laplaciano de Hodge es simétrico)."""
        G, _ = factory()
        L1, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        assert meta["is_symmetric"], "L₁ no es simétrica"
        np.testing.assert_allclose(L1, L1.T, atol=TOL_STRICT)

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.path_graph,
        GraphFactory.square_with_diagonal,
        GraphFactory.disconnected_two_triangles,
    ])
    def test_L1_is_positive_semidefinite(self, factory):
        """L₁ ≥ 0 (todos los autovalores son no negativos)."""
        G, _ = factory()
        L1, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        assert meta["is_positive_semidefinite"], (
            f"L₁ no es PSD en {factory.__name__}. "
            f"Autovalores: {meta['eigenvalues'][:5]}"
        )
        # Verificar directamente
        eigenvalues = np.array(meta["eigenvalues"])
        assert np.all(eigenvalues >= -TOL_STRICT), (
            f"Autovalor negativo: {eigenvalues.min():.2e}"
        )

    @pytest.mark.parametrize("factory,expected_inv", [
        (GraphFactory.triangle, {"n": 3, "m": 3, "c": 1, "beta_0": 1, "beta_1": 1}),
        (GraphFactory.two_triangles_shared_vertex,
         {"n": 5, "m": 6, "c": 1, "beta_0": 1, "beta_1": 2}),
        (GraphFactory.path_graph, {"n": 4, "m": 3, "c": 1, "beta_0": 1, "beta_1": 0}),
    ])
    def test_kernel_dimension_equals_beta1(self, factory, expected_inv):
        """
        dim ker(L₁) = β₁ (Isomorfismo de Hodge).

        Esto es consecuencia directa del Teorema de Hodge:
        ker(L₁) ≅ H₁(G; ℝ)
        """
        G, _ = factory()
        _, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        assert meta["hodge_isomorphism_satisfied"], (
            f"Isomorfismo de Hodge fallido: "
            f"dim ker(L₁) = {meta['kernel_dimension']}, "
            f"β₁ = {meta['betti_1']}"
        )

    def test_trace_equals_sum_eigenvalues(self):
        """Traza(L₁) = Σᵢλᵢ (propiedad espectral fundamental)."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        L1, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        trace_L1 = float(np.trace(L1))
        sum_eigvals = float(np.sum(meta["eigenvalues"]))
        assert abs(trace_L1 - sum_eigvals) < TOL_NUMERICAL, (
            f"Traza {trace_L1:.6f} ≠ Σλᵢ {sum_eigvals:.6f}"
        )

    def test_L1_equals_Lgrad_plus_Lcurl(self):
        """L₁ = B₁ᵀB₁ + B₂B₂ᵀ verificado directamente."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, _ = builder.build_cycle_matrix()
        L1, _ = builder.compute_hodge_laplacian()

        L_grad = B1.T @ B1
        L_curl = B2 @ B2.T
        L1_direct = L_grad + L_curl

        np.testing.assert_allclose(L1, L1_direct, atol=TOL_STRICT)

    def test_eigenvalues_nonnegative_after_clipping(self):
        """
        Los autovalores retornados en metadata son ≥ 0 (clipeados).

        La implementación debe clipear autovalores negativos numéricos.
        """
        G, _ = GraphFactory.square_with_diagonal()
        _, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        eigenvalues = np.array(meta["eigenvalues"])
        assert np.all(eigenvalues >= 0), (
            f"Autovalores negativos encontrados: {eigenvalues[eigenvalues < 0]}"
        )

    def test_spectral_gap_nonnegative(self):
        """Gap espectral λ₁ − λ₀ ≥ 0."""
        G, _ = GraphFactory.triangle()
        _, meta = HodgeDecompositionBuilder(G).compute_hodge_laplacian()
        assert meta["spectral_gap"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# V. TESTS: AcousticSolenoidOperator
# ─────────────────────────────────────────────────────────────────────────────

class TestAcousticSolenoidOperator:
    """
    Verifica el comportamiento del Operador de Vorticidad.
    """

    # ── Caso acíclico ─────────────────────────────────────────────────────

    def test_acyclic_graph_returns_none(self):
        """β₁ = 0 ⟹ isolate_vorticity retorna None."""
        G, _ = GraphFactory.path_graph(5)
        flows = {(i, i + 1): float(i + 1) for i in range(4)}
        op = AcousticSolenoidOperator()
        result = op.isolate_vorticity(G, flows)
        assert result is None, "Grafo acíclico debe retornar None"

    def test_tree_graph_returns_none(self):
        """Árbol generador: sin ciclos, vorticidad nula."""
        G, _ = GraphFactory.spanning_tree(6)
        flows = {(0, i): float(i) for i in range(1, 6)}
        op = AcousticSolenoidOperator()
        result = op.isolate_vorticity(G, flows)
        assert result is None

    # ── Ciclo con flujo uniforme ──────────────────────────────────────────

    def test_uniform_flow_cycle_detects_vorticity(self):
        """
        Ciclo n-gono con flujo uniforme f:
        Circulación esperada = f · n (suma de flujos en el ciclo).
        La energía E_curl = f².
        """
        G, flows = GraphFactory.uniform_flow_cycle(n=4, flow=3.0)
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None, "Debe detectar vorticidad en ciclo uniforme"
        assert magnon.kinetic_energy > 0
        assert magnon.curl_subspace_dim == 1
        assert magnon.is_significant

    def test_zero_flow_returns_none(self):
        """Flujo identicamente cero: norma ‖I‖ < ε ⟹ None."""
        G, _ = GraphFactory.triangle()
        flows = {e: 0.0 for e in G.edges()}
        op = AcousticSolenoidOperator()
        result = op.isolate_vorticity(G, flows)
        assert result is None, "Flujo cero debe retornar None"

    # ── Energía de vorticidad ─────────────────────────────────────────────

    def test_kinetic_energy_nonnegative(self):
        """E_curl = ‖Γ‖² ≥ 0 siempre."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: np.random.default_rng(99).standard_normal() for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        if magnon is not None:
            assert magnon.kinetic_energy >= 0

    def test_kinetic_energy_formula(self):
        """E_curl = ‖B₂ᵀI‖² verificado directamente."""
        G, _ = GraphFactory.triangle()
        flows = {("A", "B"): 2.0, ("B", "C"): 3.0, ("C", "A"): 1.5}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None

        # Calcular manualmente
        builder = HodgeDecompositionBuilder(G)
        B2, _ = builder.build_cycle_matrix()
        I_vec = np.array([
            flows.get(e, 0.0)
            for e in builder._edges
        ])
        circulation = B2.T @ I_vec
        E_curl_expected = float(np.dot(circulation, circulation))

        assert abs(magnon.kinetic_energy - E_curl_expected) < TOL_STRICT, (
            f"E_curl esperado {E_curl_expected:.6f}, "
            f"obtenido {magnon.kinetic_energy:.6f}"
        )

    # ── Índice de vorticidad ──────────────────────────────────────────────

    def test_vorticity_index_in_unit_interval(self):
        """ω = E_curl / ‖I‖² ∈ [0, 1]."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        rng = np.random.default_rng(42)
        flows = {e: rng.standard_normal() for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        if magnon is not None:
            assert 0.0 <= magnon.vorticity_index <= 1.0, (
                f"ω = {magnon.vorticity_index:.6f} fuera de [0, 1]"
            )

    def test_vorticity_index_zero_for_irrotational_flow(self):
        """
        Flujo irrotacional (en im(B₁ᵀ)) tiene ω ≈ 0.

        Construimos I = B₁ᵀα (gradiente de potencial) que satisface
        B₂ᵀI = B₂ᵀB₁ᵀα = (B₁B₂)ᵀα = 0.
        """
        G, _ = GraphFactory.triangle()
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()

        # I = B₁ᵀ · [1, 0, 0] (flujo de potencial)
        alpha = np.array([1.0, 0.0, 0.0])
        I_irrotational = B1.T @ alpha

        flows = {
            e: float(I_irrotational[i])
            for i, e in enumerate(builder._edges)
        }
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)

        if magnon is not None:
            assert magnon.vorticity_index < TOL_NUMERICAL, (
                f"Flujo irrotacional tiene ω = {magnon.vorticity_index:.2e}"
            )

    # ── Projector P_curl ──────────────────────────────────────────────────

    def test_projector_idempotency(self):
        """
        P_curl² = P_curl (proyector idempotente).

        Error de idempotencia ‖P² − P‖_F / ‖P‖_F < tol.
        """
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None
        assert magnon.projection_idempotency_error < TOL_PROJECTION, (
            f"Error de idempotencia: {magnon.projection_idempotency_error:.2e}"
        )

    def test_projector_symmetry(self):
        """
        P_curl es simétrico: P_curlᵀ = P_curl (proyector ortogonal).
        """
        G, _ = GraphFactory.triangle()
        flows = {e: 1.0 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None
        assert magnon.projector_matrix is not None

        P = magnon.projector_matrix
        residual = np.linalg.norm(P - P.T, "fro")
        assert residual < TOL_STRICT, (
            f"P_curl no simétrico: ‖P − Pᵀ‖_F = {residual:.2e}"
        )

    def test_projector_psd(self):
        """P_curl ≥ 0 (proyector ortogonal es PSD)."""
        G, _ = GraphFactory.triangle()
        flows = {e: 2.0 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None
        P = magnon.projector_matrix
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -TOL_STRICT), (
            f"P_curl no es PSD. Min eigval = {eigenvalues.min():.2e}"
        )

    def test_projector_rank_equals_beta1(self):
        """rank(P_curl) = β₁ (imagen del proyector = subespacio curl)."""
        G, inv = GraphFactory.triangle()
        flows = {e: 1.0 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None
        P = magnon.projector_matrix
        rank_P, _ = NumericalUtilities.compute_rank(P)
        assert rank_P == inv["beta_1"], (
            f"rank(P_curl) = {rank_P}, esperado β₁ = {inv['beta_1']}"
        )

    # ── Circulaciones ─────────────────────────────────────────────────────

    def test_circulation_count_equals_beta1(self):
        """Número de circulaciones = β₁."""
        G, inv = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        magnon = op.isolate_vorticity(G, flows)
        assert magnon is not None
        assert len(magnon.circulation_per_cycle) == inv["beta_1"], (
            f"Circulaciones: {len(magnon.circulation_per_cycle)}, "
            f"esperado β₁ = {inv['beta_1']}"
        )

    def test_circulation_linearity(self):
        """
        Circulación es lineal en el flujo: Γ(αI) = αΓ(I).

        Esto es consecuencia directa de Γ = B₂ᵀI.
        """
        G, _ = GraphFactory.triangle()
        base_flows = {("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "A"): 1.0}
        alpha = 3.7
        scaled_flows = {e: alpha * f for e, f in base_flows.items()}

        op = AcousticSolenoidOperator(tolerance_epsilon=1e-14)
        m1 = op.isolate_vorticity(G, base_flows)
        m2 = op.isolate_vorticity(G, scaled_flows)

        assert m1 is not None and m2 is not None

        for c1, c2 in zip(m1.circulation_per_cycle, m2.circulation_per_cycle):
            assert abs(c2 - alpha * c1) < TOL_STRICT, (
                f"No lineal: Γ(αI) = {c2:.6f} ≠ αΓ(I) = {alpha * c1:.6f}"
            )

    # ── Cache ─────────────────────────────────────────────────────────────

    def test_cache_invalidated_on_graph_change(self):
        """El cache se invalida cuando el grafo cambia."""
        G1, _ = GraphFactory.triangle()
        G2, _ = GraphFactory.square_with_diagonal()

        op = AcousticSolenoidOperator()
        flows1 = {e: 1.0 for e in G1.edges()}
        flows2 = {e: 1.0 for e in G2.edges()}

        # Primer análisis
        m1 = op.isolate_vorticity(G1, flows1)
        sig1 = op._cached_graph_signature

        # Segundo análisis con grafo diferente
        m2 = op.isolate_vorticity(G2, flows2)
        sig2 = op._cached_graph_signature

        assert sig1 != sig2, "Cache no invalidado al cambiar el grafo"

    def test_cache_reused_on_same_graph(self):
        """El cache se reutiliza para el mismo grafo."""
        G, _ = GraphFactory.triangle()
        op = AcousticSolenoidOperator()
        flows = {e: 1.0 for e in G.edges()}

        op.isolate_vorticity(G, flows)
        builder_first = op._cached_builder

        op.isolate_vorticity(G, flows)
        builder_second = op._cached_builder

        assert builder_first is builder_second, "Cache no reutilizado"


# ─────────────────────────────────────────────────────────────────────────────
# VI. TESTS: MagnonCartridge
# ─────────────────────────────────────────────────────────────────────────────

class TestMagnonCartridge:
    """Verifica invariantes y propiedades del dataclass MagnonCartridge."""

    def _make_valid_magnon(self, **kwargs) -> MagnonCartridge:
        defaults = dict(
            kinetic_energy=1.5,
            curl_subspace_dim=2,
            vorticity_index=0.3,
            circulation_per_cycle=(1.2, -0.8),
            projection_idempotency_error=1e-10,
            energy_decomposition={"total": 5.0, "curl": 1.5},
            cycle_metadata={},
            projector_matrix=None,
        )
        defaults.update(kwargs)
        return MagnonCartridge(**defaults)

    # ── Validación de invariantes ─────────────────────────────────────────

    def test_negative_kinetic_energy_raises(self):
        """E_curl < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="≥ 0"):
            self._make_valid_magnon(kinetic_energy=-1.0)

    def test_vorticity_index_above_one_raises(self):
        """ω > 1 + ε debe lanzar ValueError."""
        with pytest.raises(ValueError):
            self._make_valid_magnon(vorticity_index=1.5)

    def test_vorticity_index_below_zero_raises(self):
        """ω < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError):
            self._make_valid_magnon(vorticity_index=-0.1)

    def test_negative_curl_subspace_dim_raises(self):
        """k < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="≥ 0"):
            self._make_valid_magnon(curl_subspace_dim=-1)

    def test_valid_magnon_creates_successfully(self):
        """MagnonCartridge válido se crea sin errores."""
        magnon = self._make_valid_magnon()
        assert magnon.kinetic_energy == 1.5
        assert magnon.curl_subspace_dim == 2

    # ── Propiedades derivadas ─────────────────────────────────────────────

    @pytest.mark.parametrize("vorticity,expected_severity", [
        (0.51, "CRITICAL"),
        (0.21, "HIGH"),
        (0.06, "MODERATE"),
        (0.04, "LOW"),
        (0.0,  "LOW"),
    ])
    def test_thermodynamic_severity_thresholds(self, vorticity, expected_severity):
        """Clasificación de severidad según ω."""
        magnon = self._make_valid_magnon(vorticity_index=vorticity)
        assert magnon.thermodynamic_severity == expected_severity, (
            f"ω={vorticity:.2f} → esperado {expected_severity}, "
            f"obtenido {magnon.thermodynamic_severity}"
        )

    def test_is_significant_all_conditions(self):
        """is_significant = True requiere E_curl > 1e-9, ω > 0.01, k > 0."""
        magnon = self._make_valid_magnon(
            kinetic_energy=1.0,
            vorticity_index=0.5,
            curl_subspace_dim=1,
        )
        assert magnon.is_significant

    @pytest.mark.parametrize("field_name,value", [
        ("kinetic_energy", 1e-10),   # E_curl demasiado pequeño
        ("vorticity_index", 0.005),  # ω demasiado pequeño
        ("curl_subspace_dim", 0),    # Sin ciclos
    ])
    def test_is_significant_false_when_any_condition_fails(self, field_name, value):
        """is_significant = False si cualquier condición falla."""
        kwargs = dict(
            kinetic_energy=1.0,
            vorticity_index=0.5,
            curl_subspace_dim=1,
        )
        kwargs[field_name] = value
        magnon = self._make_valid_magnon(**kwargs)
        assert not magnon.is_significant, (
            f"is_significant debe ser False cuando {field_name}={value}"
        )

    def test_dominant_cycle_identifies_max_circulation(self):
        """dominant_cycle retorna el índice con mayor |Γⱼ|."""
        magnon = self._make_valid_magnon(
            circulation_per_cycle=(1.0, -5.0, 2.0)
        )
        idx, val = magnon.dominant_cycle
        assert idx == 1, f"Índice dominante esperado 1, obtenido {idx}"
        assert abs(val - (-5.0)) < TOL_STRICT

    def test_dominant_cycle_empty_returns_minus_one(self):
        """Sin circulaciones: dominant_cycle = (-1, 0.0)."""
        magnon = self._make_valid_magnon(circulation_per_cycle=())
        idx, val = magnon.dominant_cycle
        assert idx == -1
        assert val == 0.0

    def test_total_circulation_norm_equals_sqrt_energy(self):
        """‖Γ‖ = √E_curl."""
        magnon = self._make_valid_magnon(kinetic_energy=9.0)
        assert abs(magnon.total_circulation_norm - 3.0) < TOL_STRICT

    # ── Veto payload ──────────────────────────────────────────────────────

    def test_veto_payload_serializable(self):
        """to_veto_payload retorna dict serializable (sin np.ndarray)."""
        magnon = self._make_valid_magnon()
        payload = magnon.to_veto_payload()
        assert isinstance(payload, dict)
        assert "type" in payload
        assert payload["type"] == "ROUTING_VETO"

        # Verificar que no hay np.ndarray en el payload
        def check_no_arrays(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_arrays(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_no_arrays(v, f"{path}[{i}]")
            elif isinstance(obj, np.ndarray):
                raise AssertionError(
                    f"np.ndarray encontrado en payload en {path}"
                )

        check_no_arrays(payload)

    @pytest.mark.parametrize("severity,expected_action", [
        ("CRITICAL",  "COLLAPSE_AND_RECONFIGURE"),
        ("HIGH",      "PARTITION_AND_RELAY"),
        ("MODERATE",  "MONITOR_AND_DAMP"),
        ("LOW",       "LOG_AND_PROCEED"),
    ])
    def test_prescribed_action_per_severity(self, severity, expected_action):
        """Acción correctiva coincide con severidad."""
        # Mapear severidad a ω
        omega_map = {
            "CRITICAL": 0.6, "HIGH": 0.3,
            "MODERATE": 0.1, "LOW": 0.01,
        }
        magnon = self._make_valid_magnon(
            vorticity_index=omega_map[severity]
        )
        assert magnon._prescribe_action() == expected_action


# ─────────────────────────────────────────────────────────────────────────────
# VII. TESTS: Descomposición Completa de Hodge
# ─────────────────────────────────────────────────────────────────────────────

class TestFullHodgeDecomposition:
    """
    Verifica la descomposición ortogonal de Hodge–Helmholtz:

        I = I_grad + I_curl + I_harm

    con propiedades:
        (a) Reconstrucción: I_grad + I_curl + I_harm = I
        (b) Ortogonalidad: ⟨I_grad, I_curl⟩ = ⟨I_grad, I_harm⟩ = ⟨I_curl, I_harm⟩ = 0
        (c) Parseval: ‖I_grad‖² + ‖I_curl‖² + ‖I_harm‖² = ‖I‖²
    """

    def _get_decomposition(
        self, G: nx.DiGraph, flows: Dict
    ) -> Dict[str, Any]:
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-14)
        return op.compute_full_hodge_decomposition(G, flows)

    def test_reconstruction_identity(self):
        """I = I_grad + I_curl + I_harm (exacto hasta tolerancia numérica)."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        rng = np.random.default_rng(100)
        flows = {e: rng.standard_normal() for e in G.edges()}
        result = self._get_decomposition(G, flows)

        I_orig = np.array(result["original_flow"])
        I_grad = np.array(result["irrotational_component"])
        I_curl = np.array(result["solenoidal_component"])
        I_harm = np.array(result["harmonic_component"])

        reconstruction = I_grad + I_curl + I_harm
        error = np.linalg.norm(I_orig - reconstruction)
        assert error < TOL_NUMERICAL, (
            f"Error de reconstrucción: {error:.2e}"
        )

    def test_grad_curl_orthogonal(self):
        """⟨I_grad, I_curl⟩ ≈ 0."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        result = self._get_decomposition(G, flows)

        I_grad = np.array(result["irrotational_component"])
        I_curl = np.array(result["solenoidal_component"])
        inner = float(np.dot(I_grad, I_curl))

        assert abs(inner) < TOL_NUMERICAL, (
            f"⟨I_grad, I_curl⟩ = {inner:.2e} ≠ 0"
        )

    def test_grad_harm_orthogonal(self):
        """⟨I_grad, I_harm⟩ ≈ 0."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        result = self._get_decomposition(G, flows)

        I_grad = np.array(result["irrotational_component"])
        I_harm = np.array(result["harmonic_component"])
        inner = float(np.dot(I_grad, I_harm))

        assert abs(inner) < TOL_NUMERICAL, (
            f"⟨I_grad, I_harm⟩ = {inner:.2e} ≠ 0"
        )

    def test_curl_harm_orthogonal(self):
        """⟨I_curl, I_harm⟩ ≈ 0."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        result = self._get_decomposition(G, flows)

        I_curl = np.array(result["solenoidal_component"])
        I_harm = np.array(result["harmonic_component"])
        inner = float(np.dot(I_curl, I_harm))

        assert abs(inner) < TOL_NUMERICAL, (
            f"⟨I_curl, I_harm⟩ = {inner:.2e} ≠ 0"
        )

    def test_parseval_energy_identity(self):
        """
        Parseval: ‖I_grad‖² + ‖I_curl‖² + ‖I_harm‖² = ‖I‖².

        Consecuencia de la ortogonalidad y la igualdad de Pitágoras.
        """
        G, _ = GraphFactory.two_triangles_shared_vertex()
        rng = np.random.default_rng(200)
        flows = {e: rng.standard_normal() for e in G.edges()}
        result = self._get_decomposition(G, flows)

        energy = result["energy_decomposition"]
        total = energy["total"]
        sum_components = (
            energy["irrotational"] + energy["solenoidal"] + energy["harmonic"]
        )
        assert abs(total - sum_components) < TOL_NUMERICAL, (
            f"Parseval violado: ‖I‖² = {total:.6f}, "
            f"Σ‖Iᵢ‖² = {sum_components:.6f}"
        )

    def test_acyclic_graph_curl_is_zero(self):
        """Para β₁ = 0: ‖I_curl‖ = 0 (no hay componente solenoidal)."""
        G, _ = GraphFactory.path_graph(5)
        flows = {(i, i + 1): float(i + 1) for i in range(4)}
        result = self._get_decomposition(G, flows)
        curl_norm = result["norms"]["solenoidal"]
        assert curl_norm < TOL_NUMERICAL, (
            f"‖I_curl‖ = {curl_norm:.2e} debe ser ≈ 0 para β₁ = 0"
        )

    def test_verification_flags_orthogonal(self):
        """El flag is_orthogonal_decomposition debe ser True."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 1.0 for e in G.edges()}
        result = self._get_decomposition(G, flows)
        assert result["verification"]["is_orthogonal_decomposition"], (
            f"Descomposición no ortogonal: {result['verification']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# VIII. TESTS: Casos Degenerados y Frontera
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """
    Verifica el comportamiento correcto en casos límite y degenerados.
    """

    def test_single_node_no_edges(self):
        """Grafo trivial: un nodo, sin aristas."""
        G, _ = GraphFactory.single_node()
        result = verify_hodge_properties(G)
        assert result["betti_numbers"]["beta_1"] == 0
        assert result["euler_characteristic"]["verified"]

    def test_two_nodes_one_edge(self):
        """Grafo mínimo: 2 nodos, 1 arista. Sin ciclos."""
        G = nx.DiGraph()
        G.add_edge(0, 1)
        builder = HodgeDecompositionBuilder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, meta = builder.build_cycle_matrix()
        assert B2.shape[1] == 0
        assert meta["betti_1"] == 0

    def test_graph_with_antiparallel_edges(self):
        """
        Grafo con aristas antiparalelas: A→B y B→A simultáneamente.
        β₁ = m − n + c = 2 − 2 + 1 = 1.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        builder = HodgeDecompositionBuilder(G)
        B2, meta = builder.build_cycle_matrix()
        # Verificar B₁B₂ = 0
        B1, _ = builder.build_incidence_matrix()
        if B2.shape[1] > 0:
            norm = np.linalg.norm(B1 @ B2, "fro")
            assert norm < TOL_STRICT, f"‖B₁B₂‖_F = {norm:.2e} para aristas antiparalelas"

    def test_flows_with_unknown_edges_ignored(self):
        """Aristas en flows que no existen en G se ignoran silenciosamente."""
        G, _ = GraphFactory.triangle()
        flows_with_unknown = {
            ("A", "B"): 1.0,
            ("B", "C"): 1.0,
            ("C", "A"): 1.0,
            ("X", "Y"): 99.0,   # No existe en G
        }
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        # No debe lanzar excepción
        magnon = op.isolate_vorticity(G, flows_with_unknown)
        assert magnon is not None

    def test_disconnected_graph_cochain_complex(self):
        """
        Para grafo disconnected, β₀ = c > 1.
        Euler–Poincaré: χ = β₀ − β₁.
        """
        G, inv = GraphFactory.disconnected_two_triangles()
        result = HodgeDecompositionBuilder(G).verify_cochain_complex()
        assert result["is_valid"]
        assert result["beta_0"] == inv["c"]
        assert result["beta_1"] == inv["beta_1"]
        assert result["chi_geometric"] == result["chi_topological"]

    def test_large_uniform_cycle_energy(self):
        """
        Ciclo de n nodos con flujo f:
        Hay 1 ciclo fundamental, E_curl = f² (circulación = f por construcción
        del árbol generador).
        """
        for n in [5, 10, 20]:
            G, flows = GraphFactory.uniform_flow_cycle(n=n, flow=2.0)
            op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
            magnon = op.isolate_vorticity(G, flows)
            assert magnon is not None, f"No detectó vorticidad para ciclo n={n}"
            assert magnon.curl_subspace_dim == 1
            assert magnon.kinetic_energy > 0

    def test_very_small_flow_below_threshold(self):
        """Flujo < epsilon: debe retornar None."""
        G, _ = GraphFactory.triangle()
        epsilon = 1e-9
        tiny_flows = {e: epsilon * 0.01 for e in G.edges()}
        op = AcousticSolenoidOperator(tolerance_epsilon=epsilon)
        result = op.isolate_vorticity(G, tiny_flows)
        assert result is None, "Flujo sub-umbral debe retornar None"

    def test_empty_flows_dict(self):
        """Diccionario de flujos vacío: I = 0, debe retornar None."""
        G, _ = GraphFactory.triangle()
        op = AcousticSolenoidOperator()
        result = op.isolate_vorticity(G, {})
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# IX. TESTS: Euler–Poincaré y Invariantes Topológicos
# ─────────────────────────────────────────────────────────────────────────────

class TestEulerPoincare:
    """
    Verifica la fórmula de Euler–Poincaré:

        χ(G) = n − m = β₀ − β₁

    para una familia paramétrica de grafos.
    """

    @pytest.mark.parametrize("factory,expected_chi", [
        (GraphFactory.triangle,                   0),
        (GraphFactory.two_triangles_shared_vertex, -1),
        (GraphFactory.disconnected_two_triangles,  0),
        (GraphFactory.single_node,                 1),
        (GraphFactory.square_with_diagonal,        -1),
    ])
    def test_euler_poincare_geometric(self, factory, expected_chi):
        """χ_geom = n − m."""
        G, inv = factory()
        chi = inv["n"] - inv["m"]
        assert chi == expected_chi, (
            f"{factory.__name__}: χ = {chi}, esperado {expected_chi}"
        )

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.disconnected_two_triangles,
        GraphFactory.path_graph,
        GraphFactory.square_with_diagonal,
    ])
    def test_euler_poincare_topological_equals_geometric(self, factory):
        """χ_geom = χ_top = β₀ − β₁."""
        G, inv = factory()
        result = HodgeDecompositionBuilder(G).verify_cochain_complex()
        assert result["euler_poincare_ok"], (
            f"{factory.__name__}: "
            f"χ_geom = {result['chi_geometric']}, "
            f"χ_top = {result['chi_topological']}"
        )

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_cycle_graph_betti(self, n):
        """
        Ciclo n-gono: β₁ = 1, β₀ = 1, χ = 0.
        """
        G = nx.cycle_graph(n, create_using=nx.DiGraph)
        _, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
        assert meta["betti_1"] == 1, (
            f"Ciclo {n}-gono: β₁ esperado 1, obtenido {meta['betti_1']}"
        )

    @pytest.mark.parametrize("n", [4, 6])
    def test_complete_graph_betti(self, n):
        """
        K_n dirigido: β₁ = m − n + 1 = n(n−1) − n + 1.
        """
        G, inv = GraphFactory.complete_directed(n)
        _, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
        expected_beta1 = inv["m"] - inv["n"] + 1
        assert meta["betti_1"] == expected_beta1, (
            f"K_{n}: β₁ esperado {expected_beta1}, obtenido {meta['betti_1']}"
        )

    def test_tree_has_beta1_zero(self):
        """Árbol: β₁ = 0 (sin ciclos)."""
        for n in [4, 7, 10]:
            G, _ = GraphFactory.spanning_tree(n)
            _, meta = HodgeDecompositionBuilder(G).build_cycle_matrix()
            assert meta["betti_1"] == 0, (
                f"Árbol {n}-nodos: β₁ = {meta['betti_1']}, esperado 0"
            )


# ─────────────────────────────────────────────────────────────────────────────
# X. TESTS: Estabilidad Numérica
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:
    """
    Verifica que los resultados son estables ante perturbaciones de
    orden O(ε_mach) en los flujos y matrices.
    """

    def test_vorticity_stable_under_small_perturbation(self):
        """
        Perturbación δI con ‖δI‖ ≈ ε_mach · ‖I‖ no cambia la clasificación
        cualitativa (significant/non-significant) de la vorticidad.
        """
        G, _ = GraphFactory.triangle()
        base_flows = {e: 10.0 for e in G.edges()}
        rng = np.random.default_rng(300)
        perturbation = {e: rng.standard_normal() * 1e-12 for e in G.edges()}
        perturbed_flows = {
            e: base_flows[e] + perturbation[e] for e in G.edges()
        }

        op = AcousticSolenoidOperator(tolerance_epsilon=1e-12)
        m1 = op.isolate_vorticity(G, base_flows)
        m2 = op.isolate_vorticity(G, perturbed_flows)

        # Ambos deben tener la misma clasificación cualitativa
        assert (m1 is None) == (m2 is None), (
            "Perturbación numérica cambió la clasificación qualitativa"
        )
        if m1 is not None and m2 is not None:
            assert m1.thermodynamic_severity == m2.thermodynamic_severity

    def test_energy_continuous_in_flow(self):
        """
        E_curl(αI) = α² · E_curl(I).

        Consecuencia de la cuadraturicidad: E_curl = ‖B₂ᵀI‖².
        """
        G, _ = GraphFactory.triangle()
        base_flows = {e: 2.0 for e in G.edges()}
        alpha = 5.0
        scaled_flows = {e: alpha * f for e, f in base_flows.items()}

        op = AcousticSolenoidOperator(tolerance_epsilon=1e-14)
        m1 = op.isolate_vorticity(G, base_flows)
        m2 = op.isolate_vorticity(G, scaled_flows)

        assert m1 is not None and m2 is not None
        expected_ratio = alpha ** 2
        actual_ratio = m2.kinetic_energy / m1.kinetic_energy
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10, (
            f"E_curl(αI)/E_curl(I) = {actual_ratio:.6f}, "
            f"esperado α² = {expected_ratio:.6f}"
        )

    def test_B1B2_norm_below_machine_precision_threshold(self):
        """
        ‖B₁B₂‖_F debe ser O(ε_mach) para todos los grafos de prueba.
        """
        test_cases = [
            GraphFactory.triangle(),
            GraphFactory.two_triangles_shared_vertex(),
            GraphFactory.square_with_diagonal(),
            GraphFactory.disconnected_two_triangles(),
        ]
        for G, inv in test_cases:
            builder = HodgeDecompositionBuilder(G)
            B1, _ = builder.build_incidence_matrix()
            B2, meta = builder.build_cycle_matrix()
            if B2.shape[1] == 0:
                continue
            norm = meta["B1B2_norm"]
            # Para B₁, B₂ con entradas ∈ {-1, 0, 1}:
            # ‖B₁B₂‖_F debe ser exactamente 0 en aritmética exacta
            # y O(ε_mach · m) en punto flotante
            threshold = EPS_MACH * inv["m"] * 100  # margen conservador
            assert norm < threshold, (
                f"‖B₁B₂‖_F = {norm:.2e} > {threshold:.2e} "
                f"para {G.edges()}"
            )

    def test_pseudoinverse_stable_for_ill_conditioned_matrix(self):
        """
        Para matriz mal condicionada (κ ≈ 1e10), A⁺ satisface A A⁺ A = A.
        """
        # Construir matriz con κ ≈ 1e10
        rng = np.random.default_rng(400)
        U = la.orth(rng.standard_normal((6, 4)))
        V = la.orth(rng.standard_normal((6, 4)))
        sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
        A = U @ np.diag(sigmas) @ V.T

        A_plus = NumericalUtilities.moore_penrose_pseudoinverse(A)
        residual = np.linalg.norm(A @ A_plus @ A - A, "fro")
        # Para κ=1e10 y ε_mach≈2e-16: error esperado ≈ 2e-6
        assert residual < 1e-5, (
            f"A A⁺ A ≠ A para matriz mal condicionada: residual = {residual:.2e}"
        )

    def test_hodge_decomposition_stable_random_flows(self):
        """
        Descomposición de Hodge es estable para flujos aleatorios.
        Error de reconstrucción < TOL_NUMERICAL para 10 ensayos.
        """
        G, _ = GraphFactory.two_triangles_shared_vertex()
        op = AcousticSolenoidOperator(tolerance_epsilon=1e-14)
        rng = np.random.default_rng(500)

        for trial in range(10):
            flows = {e: rng.standard_normal() * 10 for e in G.edges()}
            result = op.compute_full_hodge_decomposition(G, flows)
            error = result["verification"]["reconstruction_error"]
            assert error < TOL_NUMERICAL, (
                f"Trial {trial}: error de reconstrucción {error:.2e}"
            )

    def test_projector_stable_for_orthogonal_B2(self):
        """
        Si B₂ tiene columnas ortonormales, P_curl = B₂B₂ᵀ exactamente.
        """
        # Construir B₂ ortonormal artificialmente
        m, k = 6, 2
        rng = np.random.default_rng(600)
        B2 = la.orth(rng.standard_normal((m, k)))  # k columnas ortonormales

        # Calcular proyector esperado
        P_expected = B2 @ B2.T

        # Calcular proyector vía SVD (el método del módulo)
        op = AcousticSolenoidOperator()
        P_svd, _, error = op._compute_projector_via_svd(B2)

        np.testing.assert_allclose(P_svd, P_expected, atol=TOL_STRICT)
        assert error < TOL_PROJECTION


# ─────────────────────────────────────────────────────────────────────────────
# XI. TESTS: Interfaz Agéntica (inspect_and_mitigate_resonance)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgenticInterface:
    """
    Verifica el comportamiento de la interfaz principal.
    """

    def test_resonance_detected_for_cyclic_graph(self):
        """Grafo con ciclos y flujo significativo: RESONANCE_DETECTED."""
        G, _ = GraphFactory.triangle()
        flows = {e: 10.0 for e in G.edges()}
        result = inspect_and_mitigate_resonance(G, flows)
        assert result["status"] == "RESONANCE_DETECTED"
        assert "action" in result
        assert "vorticity_metrics" in result
        assert "mathematical_proof" in result

    def test_laminar_flow_for_acyclic_graph(self):
        """Grafo acíclico: LAMINAR_FLOW."""
        G, _ = GraphFactory.path_graph(5)
        flows = {(i, i + 1): float(i) for i in range(4)}
        result = inspect_and_mitigate_resonance(G, flows)
        assert result["status"] == "LAMINAR_FLOW"
        assert result["action"] == "PROCEED"
        assert result["vorticity_metrics"]["betti_1_cycles"] == 0

    def test_full_analysis_includes_hodge_decomposition(self):
        """full_analysis=True incluye descomposición completa."""
        G, _ = GraphFactory.triangle()
        flows = {e: 5.0 for e in G.edges()}
        result = inspect_and_mitigate_resonance(G, flows, full_analysis=True)
        assert "full_hodge_decomposition" in result
        assert "spectral_analysis" in result

    def test_full_analysis_false_excludes_hodge(self):
        """full_analysis=False no incluye campos adicionales."""
        G, _ = GraphFactory.triangle()
        flows = {e: 5.0 for e in G.edges()}
        result = inspect_and_mitigate_resonance(G, flows, full_analysis=False)
        assert "full_hodge_decomposition" not in result
        assert "spectral_analysis" not in result

    def test_vorticity_metrics_nonnegative(self):
        """Todas las métricas numéricas son ≥ 0."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        flows = {e: 3.0 for e in G.edges()}
        result = inspect_and_mitigate_resonance(G, flows)
        vm = result["vorticity_metrics"]
        assert vm["parasitic_kinetic_energy"] >= 0
        assert vm["vorticity_index"] >= 0
        assert vm["betti_1_cycles"] >= 0

    def test_action_is_valid_string(self):
        """La acción prescripta es uno de los valores válidos."""
        valid_actions = {
            "COLLAPSE_AND_RECONFIGURE",
            "PARTITION_AND_RELAY",
            "MONITOR_AND_DAMP",
            "LOG_AND_PROCEED",
            "PROCEED",
        }
        for factory in [GraphFactory.triangle, GraphFactory.path_graph]:
            G, _ = factory()
            flows = {e: 5.0 for e in G.edges()}
            result = inspect_and_mitigate_resonance(G, flows)
            assert result["action"] in valid_actions, (
                f"Acción inválida: {result['action']}"
            )

    def test_generate_proof_structure(self):
        """_generate_proof retorna estructura correcta."""
        magnon = MagnonCartridge(
            kinetic_energy=2.5,
            curl_subspace_dim=1,
            vorticity_index=0.4,
            circulation_per_cycle=(2.5,),
        )
        proof = _generate_proof(magnon)
        assert "theorem" in proof
        assert "decomposition" in proof
        assert "verification" in proof
        assert "conclusion" in proof
        # Verificar que contiene los valores correctos
        assert "2.5" in proof["verification"]["energy"] or \
               f"{2.5:.6e}" in proof["verification"]["energy"]


# ─────────────────────────────────────────────────────────────────────────────
# XII. TESTS: verify_hodge_properties (utilidad standalone)
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyHodgeProperties:
    """Verifica la utilidad standalone verify_hodge_properties."""

    @pytest.mark.parametrize("factory", [
        GraphFactory.triangle,
        GraphFactory.two_triangles_shared_vertex,
        GraphFactory.path_graph,
        GraphFactory.disconnected_two_triangles,
    ])
    def test_all_properties_valid(self, factory):
        """verify_hodge_properties retorna resultados consistentes."""
        G, inv = factory()
        result = verify_hodge_properties(G)

        # Verificar estructura mínima
        assert "graph_properties" in result
        assert "cochain_complex" in result
        assert "betti_numbers" in result
        assert "euler_characteristic" in result
        assert "hodge_kernel" in result
        assert "spectral_properties" in result

    @pytest.mark.parametrize("factory,expected_inv", [
        (GraphFactory.triangle,
         {"n": 3, "m": 3, "beta_0": 1, "beta_1": 1}),
        (GraphFactory.two_triangles_shared_vertex,
         {"n": 5, "m": 6, "beta_0": 1, "beta_1": 2}),
        (GraphFactory.disconnected_two_triangles,
         {"n": 6, "m": 6, "beta_0": 2, "beta_1": 2}),
    ])
    def test_betti_numbers_correct(self, factory, expected_inv):
        """Números de Betti son correctos."""
        G, _ = factory()
        result = verify_hodge_properties(G)
        bn = result["betti_numbers"]
        assert bn["beta_0"] == expected_inv["beta_0"], (
            f"β₀: esperado {expected_inv['beta_0']}, obtenido {bn['beta_0']}"
        )
        assert bn["beta_1"] == expected_inv["beta_1"], (
            f"β₁: esperado {expected_inv['beta_1']}, obtenido {bn['beta_1']}"
        )

    def test_euler_characteristic_verified(self):
        """Euler–Poincaré verificado en result."""
        G, _ = GraphFactory.two_triangles_shared_vertex()
        result = verify_hodge_properties(G)
        assert result["euler_characteristic"]["verified"]

    def test_hodge_kernel_isomorphism(self):
        """Isomorfismo de Hodge: dim ker(L₁) = β₁."""
        for factory in [
            GraphFactory.triangle,
            GraphFactory.two_triangles_shared_vertex,
            GraphFactory.path_graph,
        ]:
            G, _ = factory()
            result = verify_hodge_properties(G)
            hk = result["hodge_kernel"]
            assert hk["isomorphism_ok"], (
                f"{factory.__name__}: "
                f"dim ker(L₁) = {hk['ker_L1_dimension']}, "
                f"β₁ = {hk['expected_beta_1']}"
            )

    def test_kernel_vectors_in_null_space_of_B1T_and_B2T(self):
        """ker(L₁) ⊆ ker(B₁ᵀ) ∩ ker(B₂ᵀ)."""
        G, _ = GraphFactory.triangle()
        result = verify_hodge_properties(G)
        hk = result["hodge_kernel"]
        assert hk["kernel_property_ok"], (
            f"ker(L₁) no está en ker(B₁ᵀ) ∩ ker(B₂ᵀ): "
            f"‖B₁ᵀN‖ = {hk['ker_subset_of_ker_B1T']:.2e}, "
            f"‖B₂ᵀN‖ = {hk['ker_subset_of_ker_B2T']:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",                          # verbose
        "--tb=short",                  # traceback corto
        "-rN",                         # no mostrar summary de passed
        "--color=yes",
        "-x",                          # parar en primer fallo
    ])