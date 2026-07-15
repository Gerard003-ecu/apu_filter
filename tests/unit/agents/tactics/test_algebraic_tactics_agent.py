r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite  : test_algebraic_tactics_agent.py                                     ║
║ Ruta   : tests/unit/agents/tactics/test_algebraic_tactics_agent.py           ║
║ Versión: 2.1.0-Rigorous-Spectral-Categorical-Homological-Ring                ║
║ Objetivo: Validación granular y rigurosa del Endofuntor AlgebraicTacticsAgent║
║           (Fases 1 → 2 → 3 anidadas)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de la suite (espejo de las fases del SUT):
  §T0  Constantes, Mónada Option, DTOs e jerarquía de excepciones
  §T1  Fase 1 — Homogeneidad del anillo conmutativo ℛ y saneamiento monádico
  §T2  Fase 2 — Homología simplicial, Laplaciano y valor de Fiedler
  §T3  Fase 3 — Orquestador AlgebraicTacticsAgent (composición funtorial)
  §T4  Integración de punta a punta, invariantes y casos límite
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# ─── SUT ─────────────────────────────────────────────────────────────────────
from app.tactics.algebraic_tactics_agent import (
    AlgebraicConstants,
    OptionMonad,
    AlgebraicTacticsError,
    RingSymmetryViolation,
    RingDegeneracyError,
    TopologicalIslandError,
    HomologicalInvariantError,
    EmptyCostTensorError,
    RingHomogeneityValidation,
    SimplicialSkeletonAudit,
    Phase1_AlgebraicRingAuditor,
    Phase2_TopologicalSkeletonAuditor,
    AlgebraicTacticsAgent,
)

try:
    from app.core.mic_algebra import CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    from app.tactics.algebraic_tactics_agent import (  # type: ignore[attr-defined]
        CategoricalState,
        TopologicalInvariantError,
        Stratum,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES GLOBALES Y HELPERS DE MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def _connected_path_matrix(n: int = 4, d: int = 2) -> NDArray:
    """
    Tensor n×d cuyas filas generan un grafo camino conexo vía Gram.
    Fila i ≈ e_{min(i,d-1)} + ruido direccional suave para solapamiento
    entre vecinos consecutivos.
    """
    rng = np.random.default_rng(0)
    M = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        M[i, i % d] = 1.0 + 0.1 * i
        if d > 1:
            M[i, (i + 1) % d] = 0.5  # solapamiento con vecino
    # Pequeña perturbación positiva para estabilidad numérica
    M += 1e-3 * rng.standard_normal(M.shape)
    return M


def _disconnected_two_islands() -> NDArray:
    """
    Dos bloques ortogonales → Gram bloque-diagonal → β₀ = 2.
    Islas: filas 0..1 en span{e0}, filas 2..3 en span{e1}.
    """
    return np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )


def _fully_connected_clique(n: int = 5, d: int = 3) -> NDArray:
    """Filas con soporte completo → grafo denso casi completo."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, d)) + 1.0  # sesgo positivo → productos internos > 0
    return M.astype(np.float64)


def _single_vertex() -> NDArray:
    return np.array([[1.0, 2.0, 3.0]], dtype=np.float64)


def _rank_one_matrix() -> NDArray:
    """Todas las filas son múltiplos de la misma dirección → grafo completo conexo."""
    v = np.array([1.0, 2.0, 3.0])
    return np.array([v, 2 * v, 0.5 * v, -v], dtype=np.float64)


def _zero_matrix(n: int = 3, d: int = 2) -> NDArray:
    return np.zeros((n, d), dtype=np.float64)


def _matrix_with_singularities() -> NDArray:
    return np.array(
        [
            [1.0, np.nan],
            [np.inf, 2.0],
            [-np.inf, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def raw_ast_dummy() -> List[Dict[str, Any]]:
    return [{"item": "A", "qty": 1}, {"item": "B", "qty": 2}]


@pytest.fixture
def connected_matrix() -> NDArray:
    return _connected_path_matrix(5, 3)


@pytest.fixture
def disconnected_matrix() -> NDArray:
    return _disconnected_two_islands()


@pytest.fixture
def clique_matrix() -> NDArray:
    return _fully_connected_clique(4, 2)


@pytest.fixture
def phase1() -> type:
    return Phase1_AlgebraicRingAuditor


@pytest.fixture
def phase2() -> type:
    return Phase2_TopologicalSkeletonAuditor


@pytest.fixture
def agent_default() -> AlgebraicTacticsAgent:
    return AlgebraicTacticsAgent()


def _patch_processor(matrix: NDArray):
    """Context manager factory: parchea APUProcessor.process → matrix fija."""
    return patch(
        "app.tactics.algebraic_tactics_agent.APUProcessor"
    )


def _mock_processor_returning(matrix: NDArray):
    """
    Devuelve un patch activo cuyo process retorna `matrix`.
    Uso:
        with _mock_processor_returning(M) as MockProc:
            ...
    """
    p = patch("app.tactics.algebraic_tactics_agent.APUProcessor")
    mock_cls = p.start()
    mock_cls.return_value.process.return_value = matrix
    return p, mock_cls


# ══════════════════════════════════════════════════════════════════════════════
# §T0 — CONSTANTES, MÓNADA, DTOs Y JERARQUÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════

class TestAlgebraicConstants:
    """Invariantes numéricos de AlgebraicConstants."""

    def test_machine_eps_positive(self):
        assert AlgebraicConstants.MACHINE_EPS > 0.0
        assert AlgebraicConstants.MACHINE_EPS == pytest.approx(np.finfo(np.float64).eps)

    def test_ring_tolerance_positive(self):
        assert AlgebraicConstants.RING_TOLERANCE > 0.0

    def test_fiedler_tolerance_positive(self):
        assert AlgebraicConstants.FIEDLER_TOLERANCE > 0.0

    def test_sample_size_positive(self):
        assert AlgebraicConstants.RING_SAMPLE_SIZE >= 1

    def test_max_condition_number_large(self):
        assert AlgebraicConstants.MAX_CONDITION_NUMBER >= 1e6

    def test_min_vertices_at_least_one(self):
        assert AlgebraicConstants.MIN_VERTICES >= 1

    def test_seed_is_int(self):
        assert isinstance(AlgebraicConstants.RING_SAMPLE_SEED, int)


class TestOptionMonad:
    """Mónada Option: unit, bind, map, get_or_else, nothing."""

    def test_unit_finite_float(self):
        m = OptionMonad.unit(3.14)
        assert m.is_singular is False
        assert m.value == pytest.approx(3.14)

    def test_unit_nan_is_singular(self):
        m = OptionMonad.unit(float("nan"))
        assert m.is_singular is True
        assert m.value is None

    def test_unit_pos_inf_is_singular(self):
        m = OptionMonad.unit(float("inf"))
        assert m.is_singular is True

    def test_unit_neg_inf_is_singular(self):
        m = OptionMonad.unit(float("-inf"))
        assert m.is_singular is True

    def test_unit_none_is_singular(self):
        m = OptionMonad.unit(None)  # type: ignore[arg-type]
        assert m.is_singular is True

    def test_unit_numpy_nan(self):
        m = OptionMonad.unit(np.float64("nan"))
        assert m.is_singular is True

    def test_nothing_factory(self):
        m = OptionMonad.nothing()
        assert m.is_singular is True
        assert m.value is None

    def test_pure_factory(self):
        m = OptionMonad.pure(42)
        assert m.is_singular is False
        assert m.value == 42

    def test_bind_on_just(self):
        m = OptionMonad.unit(2.0)
        result = m.bind(lambda x: OptionMonad.unit(x * 3))
        assert result.is_singular is False
        assert result.value == pytest.approx(6.0)

    def test_bind_on_nothing_short_circuits(self):
        m = OptionMonad.nothing()
        called = []

        def f(x):
            called.append(x)
            return OptionMonad.unit(x)

        result = m.bind(f)
        assert result.is_singular is True
        assert called == []

    def test_bind_exception_becomes_nothing(self):
        m = OptionMonad.unit(1.0)
        result = m.bind(lambda x: (_ for _ in ()).throw(ValueError("boom")))
        # bind captura excepciones → Nothing
        assert result.is_singular is True

    def test_map_functorial(self):
        m = OptionMonad.unit(5)
        result = m.map(lambda x: x + 1)
        assert result.value == 6
        assert result.is_singular is False

    def test_map_on_nothing(self):
        m = OptionMonad.nothing()
        result = m.map(lambda x: x + 1)
        assert result.is_singular is True

    def test_get_or_else_just(self):
        assert OptionMonad.unit(7).get_or_else(0) == 7

    def test_get_or_else_nothing(self):
        assert OptionMonad.nothing().get_or_else(99) == 99

    def test_frozen_immutability(self):
        m = OptionMonad.unit(1.0)
        with pytest.raises(Exception):
            m.value = 2.0  # type: ignore[misc]

    def test_repr_just(self):
        r = repr(OptionMonad.unit(1.5))
        assert "Just" in r

    def test_repr_nothing(self):
        r = repr(OptionMonad.nothing())
        assert "Nothing" in r

    def test_complex_non_finite(self):
        m = OptionMonad.unit(complex(float("inf"), 0.0))
        assert m.is_singular is True

    def test_complex_finite(self):
        m = OptionMonad.unit(complex(1.0, 2.0))
        assert m.is_singular is False


class TestExceptionHierarchy:
    """Jerarquía de vetos algebraicos y topológicos."""

    def test_root_is_topological(self):
        assert issubclass(AlgebraicTacticsError, TopologicalInvariantError)

    def test_ring_symmetry_is_tactics(self):
        assert issubclass(RingSymmetryViolation, AlgebraicTacticsError)

    def test_ring_degeneracy_is_tactics(self):
        assert issubclass(RingDegeneracyError, AlgebraicTacticsError)

    def test_topological_island_is_tactics(self):
        assert issubclass(TopologicalIslandError, AlgebraicTacticsError)

    def test_homological_is_tactics(self):
        assert issubclass(HomologicalInvariantError, AlgebraicTacticsError)

    def test_empty_cost_is_tactics(self):
        assert issubclass(EmptyCostTensorError, AlgebraicTacticsError)

    def test_catchable_as_root(self):
        with pytest.raises(AlgebraicTacticsError):
            raise TopologicalIslandError("isla")
        with pytest.raises(TopologicalInvariantError):
            raise RingSymmetryViolation("anillo")


class TestDTOImmutability:
    """Contratos frozen entre fases."""

    def test_ring_validation_is_frozen(self, phase1, raw_ast_dummy, connected_matrix):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        assert isinstance(ring, RingHomogeneityValidation)
        with pytest.raises(Exception):
            ring.is_closed = False  # type: ignore[misc]

    def test_ring_validation_fields(self, phase1, raw_ast_dummy, connected_matrix):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        for attr in (
            "is_closed", "is_commutative", "is_associative", "is_distributive",
            "has_additive_identity", "has_multiplicative_identity",
            "cost_matrix", "monadic_failures", "frobenius_norm", "spectral_norm",
            "condition_number", "matrix_rank", "singular_values", "ring_check_log",
        ):
            assert hasattr(ring, attr), f"falta campo {attr}"

    def test_skeleton_audit_is_frozen(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert isinstance(skel, SimplicialSkeletonAudit)
        with pytest.raises(Exception):
            skel.betti_0 = 99  # type: ignore[misc]

    def test_skeleton_embeds_ring_validation(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.ring_validation is ring


# ══════════════════════════════════════════════════════════════════════════════
# §T1 — FASE 1: HOMOGENEIDAD DEL ANILLO ℛ Y SANEAMIENTO MONÁDICO
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1SanitizeMonadic:
    """Saneamiento monádico NaN/Inf → 0."""

    def test_no_singularities(self, phase1):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        sane, n = phase1._sanitize_matrix_monadic(M)
        assert n == 0
        assert_allclose(sane, M)

    def test_absorbs_nan_inf(self, phase1):
        M = _matrix_with_singularities()
        sane, n = phase1._sanitize_matrix_monadic(M)
        assert n == 3  # nan, +inf, -inf
        assert np.all(np.isfinite(sane))
        assert sane[0, 1] == 0.0
        assert sane[1, 0] == 0.0
        assert sane[2, 0] == 0.0
        assert sane[3, 0] == pytest.approx(4.0)

    def test_preserves_shape(self, phase1):
        M = np.ones((7, 5)) * np.nan
        sane, n = phase1._sanitize_matrix_monadic(M)
        assert sane.shape == (7, 5)
        assert n == 35
        assert_allclose(sane, np.zeros((7, 5)))


class TestPhase1SpectralProfile:
    """SVD: rango, normas, condición."""

    def test_identity_like(self, phase1):
        M = np.eye(3)
        sv, rank, fro, spectral, cond = phase1._spectral_profile(M)
        assert rank == 3
        assert fro == pytest.approx(math.sqrt(3.0))
        assert spectral == pytest.approx(1.0)
        assert cond == pytest.approx(1.0, abs=1e-6)

    def test_rank_one(self, phase1):
        M = _rank_one_matrix()
        sv, rank, fro, spectral, cond = phase1._spectral_profile(M)
        assert rank == 1
        assert spectral > 0.0
        assert fro > 0.0

    def test_empty_matrix(self, phase1):
        M = np.zeros((0, 0))
        sv, rank, fro, spectral, cond = phase1._spectral_profile(M)
        assert rank == 0
        assert fro == 0.0
        assert math.isinf(cond) or cond == 0.0 or len(sv) == 0

    def test_zero_matrix_rank_zero(self, phase1):
        M = _zero_matrix(3, 2)
        sv, rank, fro, spectral, cond = phase1._spectral_profile(M)
        assert rank == 0
        assert fro == pytest.approx(0.0)
        assert spectral == pytest.approx(0.0)


class TestPhase1SampleIndices:
    """Muestreo determinista reproducible."""

    def test_reproducibility(self, phase1):
        a = phase1._sample_column_indices(20, 8)
        b = phase1._sample_column_indices(20, 8)
        assert_array_equal(a, b)

    def test_size_capped(self, phase1):
        idx = phase1._sample_column_indices(5, 100)
        assert len(idx) == 5

    def test_empty_cols(self, phase1):
        idx = phase1._sample_column_indices(0, 5)
        assert len(idx) == 0

    def test_unique_indices(self, phase1):
        idx = phase1._sample_column_indices(15, 10)
        assert len(set(idx.tolist())) == len(idx)


class TestPhase1RingAxioms:
    """Los 6 axiomas del anillo ℛ sobre columnas."""

    def test_additive_closure_finite(self, phase1, connected_matrix):
        assert phase1._verify_additive_closure(connected_matrix) is True

    def test_commutativity(self, phase1, connected_matrix):
        assert phase1._verify_commutativity(connected_matrix) is True

    def test_associativity(self, phase1, connected_matrix):
        assert phase1._verify_associativity(connected_matrix) is True

    def test_distributivity(self, phase1, connected_matrix):
        assert phase1._verify_distributivity(connected_matrix) is True

    def test_additive_identity(self, phase1, connected_matrix):
        assert phase1._verify_additive_identity(connected_matrix) is True

    def test_multiplicative_identity(self, phase1, connected_matrix):
        assert phase1._verify_multiplicative_identity(connected_matrix) is True

    def test_single_column_axioms_hold(self, phase1):
        M = np.array([[1.0], [2.0], [3.0]])
        assert phase1._verify_additive_closure(M) is True
        assert phase1._verify_commutativity(M) is True
        assert phase1._verify_associativity(M) is True
        assert phase1._verify_distributivity(M) is True
        assert phase1._verify_additive_identity(M) is True
        assert phase1._verify_multiplicative_identity(M) is True

    def test_distributivity_false_on_empty(self, phase1):
        M = np.zeros((3, 0))
        assert phase1._verify_distributivity(M) is False


class TestPhase1AuditRingManifold:
    """Método terminal de la Fase 1."""

    def test_success_connected(
        self, phase1, raw_ast_dummy, connected_matrix
    ):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()

        assert isinstance(ring, RingHomogeneityValidation)
        assert ring.is_closed is True
        assert ring.is_commutative is True
        assert ring.is_associative is True
        assert ring.is_distributive is True
        assert ring.has_additive_identity is True
        assert ring.has_multiplicative_identity is True
        assert ring.matrix_rank >= 1
        assert ring.frobenius_norm > 0.0
        assert ring.cost_matrix.shape == connected_matrix.shape
        assert "OK" in ring.ring_check_log

    def test_absorbs_singularities_count(
        self, phase1, raw_ast_dummy
    ):
        M = _matrix_with_singularities()
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        assert ring.monadic_failures == 3
        assert np.all(np.isfinite(ring.cost_matrix))

    def test_empty_tensor_raises(self, phase1, raw_ast_dummy):
        p, _ = _mock_processor_returning(np.zeros((0, 2)))
        try:
            with pytest.raises(EmptyCostTensorError):
                phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()

    def test_zero_rank_raises(self, phase1, raw_ast_dummy):
        p, _ = _mock_processor_returning(_zero_matrix(3, 2))
        try:
            with pytest.raises(RingDegeneracyError):
                phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()

    def test_non_ndarray_raises(self, phase1, raw_ast_dummy):
        p, _ = _mock_processor_returning([[1, 2], [3, 4]])  # type: ignore[arg-type]
        # process retorna list → no ndarray
        try:
            # Forzar retorno de lista
            with patch(
                "app.tactics.algebraic_tactics_agent.APUProcessor"
            ) as MockProc:
                MockProc.return_value.process.return_value = [[1.0, 2.0]]
                with pytest.raises(RingSymmetryViolation):
                    phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()

    def test_ndim_not_2_raises(self, phase1, raw_ast_dummy):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = np.array([1.0, 2.0, 3.0])
            with pytest.raises(RingSymmetryViolation):
                phase1.audit_ring_manifold(raw_ast_dummy)

    def test_processor_exception_wrapped(self, phase1, raw_ast_dummy):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.side_effect = RuntimeError("APU crash")
            with pytest.raises(RingSymmetryViolation) as ei:
                phase1.audit_ring_manifold(raw_ast_dummy)
        assert "APU" in str(ei.value) or "procesador" in str(ei.value).lower() or "Colapso" in str(ei.value)

    def test_singular_values_sorted_descending(
        self, phase1, raw_ast_dummy, clique_matrix
    ):
        p, _ = _mock_processor_returning(clique_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        sv = ring.singular_values
        if len(sv) > 1:
            assert np.all(sv[:-1] >= sv[1:] - 1e-12)

    def test_condition_number_finite_for_full_rank(
        self, phase1, raw_ast_dummy
    ):
        M = np.eye(4, 3)  # rank 3
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        assert ring.matrix_rank == 3
        assert math.isfinite(ring.condition_number)
        assert ring.condition_number >= 1.0 - 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# §T2 — FASE 2: HOMOLÓGICA Y ESPECTRAL DEL 1-ESQUELETO
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2InduceAdjacency:
    """Grafo de adyacencia inducido por Gram."""

    def test_symmetric(self, phase2, connected_matrix):
        A = phase2._induce_adjacency(connected_matrix)
        assert_allclose(A, A.T)
        assert A.shape[0] == connected_matrix.shape[0]

    def test_zero_diagonal(self, phase2, connected_matrix):
        A = phase2._induce_adjacency(connected_matrix)
        assert_allclose(np.diag(A), np.zeros(A.shape[0]))

    def test_disconnected_has_block_structure(self, phase2, disconnected_matrix):
        A = phase2._induce_adjacency(disconnected_matrix)
        # Bloques {0,1} y {2,3}: no debe haber arista entre bloques
        assert A[0, 2] == 0.0
        assert A[0, 3] == 0.0
        assert A[1, 2] == 0.0
        assert A[1, 3] == 0.0
        # Dentro del bloque ortogonal a e1, filas paralelas a e0 se conectan
        assert A[0, 1] == 1.0
        assert A[2, 3] == 1.0

    def test_rank_one_is_complete(self, phase2):
        M = _rank_one_matrix()
        A = phase2._induce_adjacency(M)
        n = M.shape[0]
        # Grafo completo K_n (sin lazos)
        assert int(A.sum()) == n * (n - 1)


class TestPhase2OrientedIncidence:
    """Matriz de incidencia orientada B₁."""

    def test_path_graph_incidence(self, phase2):
        # Camino 0—1—2
        A = np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64
        )
        B, edges = phase2._build_oriented_incidence(A)
        assert B.shape == (3, 2)
        assert len(edges) == 2
        # Cada columna tiene exactamente un +1 y un −1
        for col in range(B.shape[1]):
            assert_allclose(B[:, col].sum(), 0.0)
            assert np.count_nonzero(B[:, col]) == 2

    def test_no_edges(self, phase2):
        A = np.zeros((4, 4))
        B, edges = phase2._build_oriented_incidence(A)
        assert B.shape == (4, 0)
        assert edges == []

    def test_column_orientation_i_lt_j(self, phase2):
        A = np.array([[0, 1], [1, 0]], dtype=np.float64)
        B, edges = phase2._build_oriented_incidence(A)
        assert edges == [(0, 1)]
        assert B[0, 0] == 1.0
        assert B[1, 0] == -1.0


class TestPhase2BettiFromIncidence:
    """Fórmulas β₀ = |V|−rank(B), β₁ = |E|−rank(B)."""

    def test_tree(self, phase2):
        # Árbol: |V|=4, |E|=3, rank=3 → β₀=1, β₁=0
        b0, b1 = phase2._betti_from_incidence(4, 3, 3)
        assert b0 == 1
        assert b1 == 0

    def test_cycle(self, phase2):
        # C_3: |V|=3, |E|=3, rank=2 → β₀=1, β₁=1
        b0, b1 = phase2._betti_from_incidence(3, 3, 2)
        assert b0 == 1
        assert b1 == 1

    def test_two_components(self, phase2):
        # Dos aristas disjuntas: |V|=4, |E|=2, rank=2 → β₀=2, β₁=0
        b0, b1 = phase2._betti_from_incidence(4, 2, 2)
        assert b0 == 2
        assert b1 == 0

    def test_isolated_vertices(self, phase2):
        b0, b1 = phase2._betti_from_incidence(5, 0, 0)
        assert b0 == 5
        assert b1 == 0


class TestPhase2LaplacianSpectrum:
    """Espectro de L = BBᵀ y Fiedler."""

    def test_path_connected_fiedler_positive(self, phase2):
        A = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        B, _ = phase2._build_oriented_incidence(A)
        spectrum, fiedler, beta0 = phase2._laplacian_spectrum(B, 4)
        assert beta0 == 1
        assert fiedler > AlgebraicConstants.FIEDLER_TOLERANCE
        assert spectrum[0] == pytest.approx(0.0, abs=AlgebraicConstants.FIEDLER_TOLERANCE)
        assert len(spectrum) == 4
        # Espectro no decreciente
        assert np.all(spectrum[:-1] <= spectrum[1:] + 1e-12)

    def test_no_edges_all_zero_spectrum(self, phase2):
        B = np.zeros((3, 0))
        spectrum, fiedler, beta0 = phase2._laplacian_spectrum(B, 3)
        assert beta0 == 3
        assert fiedler == pytest.approx(0.0)
        assert_allclose(spectrum, np.zeros(3))

    def test_complete_graph_high_fiedler(self, phase2):
        n = 4
        A = np.ones((n, n)) - np.eye(n)
        B, _ = phase2._build_oriented_incidence(A)
        spectrum, fiedler, beta0 = phase2._laplacian_spectrum(B, n)
        assert beta0 == 1
        # Para K_n, λ₂ = n
        assert fiedler == pytest.approx(float(n), abs=1e-6)


class TestPhase2AuditSimplicialSkeleton:
    """Método principal de la Fase 2 (continuación de RingHomogeneityValidation)."""

    def _ring_from_matrix(self, phase1, raw_ast, matrix):
        p, _ = _mock_processor_returning(matrix)
        try:
            return phase1.audit_ring_manifold(raw_ast)
        finally:
            p.stop()

    def test_success_connected(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, connected_matrix)
        skel = phase2.audit_simplicial_skeleton(ring)

        assert isinstance(skel, SimplicialSkeletonAudit)
        assert skel.is_connected is True
        assert skel.betti_0 == 1
        assert skel.fiedler_value > AlgebraicConstants.FIEDLER_TOLERANCE
        assert skel.num_vertices == connected_matrix.shape[0]
        assert skel.incidence_rank >= 1
        assert skel.euler_characteristic == skel.num_vertices - skel.num_edges
        assert 0.0 <= skel.adjacency_density <= 1.0
        assert skel.ring_validation is ring

    def test_disconnected_raises_island(
        self, phase1, phase2, raw_ast_dummy, disconnected_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, disconnected_matrix)
        with pytest.raises(TopologicalIslandError) as ei:
            phase2.audit_simplicial_skeleton(ring)
        msg = str(ei.value)
        assert "β₀" in msg or "beta" in msg.lower() or "inconexa" in msg.lower() or "topológic" in msg.lower()

    def test_single_vertex_trivial(
        self, phase1, phase2, raw_ast_dummy
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, _single_vertex())
        skel = phase2.audit_simplicial_skeleton(ring)
        assert skel.betti_0 == 1
        assert skel.betti_1 == 0
        assert skel.is_connected is True
        assert skel.num_edges == 0
        assert math.isinf(skel.fiedler_value)

    def test_clique_connected(
        self, phase1, phase2, raw_ast_dummy, clique_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, clique_matrix)
        skel = phase2.audit_simplicial_skeleton(ring)
        assert skel.is_connected is True
        assert skel.betti_0 == 1
        assert skel.fiedler_value > 0.0

    def test_rank_one_connected(
        self, phase1, phase2, raw_ast_dummy
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, _rank_one_matrix())
        skel = phase2.audit_simplicial_skeleton(ring)
        assert skel.betti_0 == 1
        assert skel.is_connected is True
        # K_4 → β₁ = |E|−rank = 6−3 = 3
        assert skel.num_edges == 6
        assert skel.betti_1 == 3

    def test_invalid_ring_guard(self, phase2, connected_matrix):
        """Anillo artificialmente inválido → RingSymmetryViolation en Fase 2."""
        bad = RingHomogeneityValidation(
            is_closed=False,
            is_commutative=True,
            is_associative=True,
            is_distributive=True,
            has_additive_identity=True,
            has_multiplicative_identity=True,
            cost_matrix=connected_matrix,
            monadic_failures=0,
            frobenius_norm=1.0,
            spectral_norm=1.0,
            condition_number=1.0,
            matrix_rank=1,
            singular_values=np.array([1.0]),
            ring_check_log="FORCED_FAIL",
        )
        with pytest.raises(RingSymmetryViolation):
            phase2.audit_simplicial_skeleton(bad)

    def test_euler_characteristic_consistency(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, connected_matrix)
        skel = phase2.audit_simplicial_skeleton(ring)
        assert skel.euler_characteristic == skel.num_vertices - skel.num_edges
        # χ = β₀ − β₁ en complejos de dim ≤ 1 conexos de tipo grafo
        assert skel.euler_characteristic == skel.betti_0 - skel.betti_1

    def test_laplacian_spectrum_length(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, connected_matrix)
        skel = phase2.audit_simplicial_skeleton(ring)
        assert len(skel.laplacian_spectrum) == skel.num_vertices
        assert skel.laplacian_spectrum[0] == pytest.approx(
            0.0, abs=AlgebraicConstants.FIEDLER_TOLERANCE
        )

    def test_homology_log_nonempty(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        ring = self._ring_from_matrix(phase1, raw_ast_dummy, connected_matrix)
        skel = phase2.audit_simplicial_skeleton(ring)
        assert isinstance(skel.homology_log, str)
        assert len(skel.homology_log) > 0


# ══════════════════════════════════════════════════════════════════════════════
# §T3 — FASE 3: ORQUESTADOR ALGEBRAIC TACTICS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestAlgebraicTacticsAgentInit:
    """Construcción y parámetros del endofuntor."""

    def test_default_tolerances(self, agent_default):
        assert agent_default._fiedler_tol == pytest.approx(
            AlgebraicConstants.FIEDLER_TOLERANCE
        )
        assert agent_default._ring_tol == pytest.approx(
            AlgebraicConstants.RING_TOLERANCE
        )

    def test_custom_tolerances(self):
        agent = AlgebraicTacticsAgent(
            fiedler_tolerance=1e-5, ring_tolerance=1e-8
        )
        assert agent._fiedler_tol == pytest.approx(1e-5)
        assert agent._ring_tol == pytest.approx(1e-8)

    def test_is_subclass_of_phase2(self):
        assert issubclass(AlgebraicTacticsAgent, Phase2_TopologicalSkeletonAuditor)

    def test_is_subclass_of_phase1(self):
        assert issubclass(AlgebraicTacticsAgent, Phase1_AlgebraicRingAuditor)


class TestAlgebraicTacticsAgentCall:
    """Composición funtorial completa: raw_ast → CategoricalState."""

    def test_success_returns_categorical_state(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)

        assert isinstance(state, CategoricalState)
        assert state.stratum == Stratum.TACTICS

    def test_payload_keys(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)

        required = {
            "cost_matrix",
            "fiedler_value",
            "betti_0",
            "betti_1",
            "incidence_rank",
            "num_vertices",
            "num_edges",
            "euler_characteristic",
            "adjacency_density",
            "laplacian_spectrum",
            "frobenius_norm",
            "spectral_norm",
            "condition_number",
            "matrix_rank",
            "singular_values",
            "ring_check_log",
            "homology_log",
        }
        assert required.issubset(state.payload.keys())

    def test_context_keys(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)

        ctx = state.context
        assert ctx.get("is_connected") is True
        assert ctx.get("is_distributive") is True
        assert ctx.get("is_closed") is True
        assert ctx.get("is_commutative") is True
        assert ctx.get("is_associative") is True
        assert "monadic_failures_absorbed" in ctx
        assert "fiedler_tolerance_applied" in ctx
        assert "ring_tolerance_applied" in ctx

    def test_betti_zero_is_one_when_connected(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)
        assert state.payload["betti_0"] == 1

    def test_fiedler_positive(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)
        assert state.payload["fiedler_value"] > 0.0

    def test_type_error_on_non_list(self, agent_default):
        with pytest.raises(TypeError):
            agent_default("not a list")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agent_default(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            agent_default(42)  # type: ignore[arg-type]

    def test_disconnected_raises(
        self, agent_default, raw_ast_dummy, disconnected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = disconnected_matrix
            with pytest.raises(TopologicalIslandError):
                agent_default(raw_ast_dummy)

    def test_zero_matrix_raises(
        self, agent_default, raw_ast_dummy
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = _zero_matrix()
            with pytest.raises(RingDegeneracyError):
                agent_default(raw_ast_dummy)

    def test_singularities_absorbed_in_context(
        self, agent_default, raw_ast_dummy
    ):
        M = _matrix_with_singularities()
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = M
            # Puede ser conexo o no tras saneamiento; solo verificamos que
            # no reviente por NaN y que el contador monádico se propague.
            try:
                state = agent_default(raw_ast_dummy)
                assert state.context["monadic_failures_absorbed"] == 3
            except TopologicalIslandError:
                # Si tras absorber singularidades el grafo se desconecta, es válido
                pass

    def test_cost_matrix_in_payload_matches_shape(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)
        cm = np.array(state.payload["cost_matrix"])
        assert cm.shape == connected_matrix.shape

    def test_tolerances_propagated_to_context(
        self, raw_ast_dummy, connected_matrix
    ):
        agent = AlgebraicTacticsAgent(
            fiedler_tolerance=1e-6, ring_tolerance=1e-10
        )
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent(raw_ast_dummy)
        assert state.context["fiedler_tolerance_applied"] == pytest.approx(1e-6)
        assert state.context["ring_tolerance_applied"] == pytest.approx(1e-10)

    def test_num_vertices_matches(
        self, agent_default, raw_ast_dummy, connected_matrix
    ):
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default(raw_ast_dummy)
        assert state.payload["num_vertices"] == connected_matrix.shape[0]
        assert len(state.payload["laplacian_spectrum"]) == connected_matrix.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
# §T4 — INTEGRACIÓN, INVARIANTES Y CASOS LÍMITE
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndInvariants:
    """Propiedades preservadas a través de la composición funtorial."""

    def test_phase_composition_identity_of_ring(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.ring_validation is ring

    def test_agent_matches_manual_composition(
        self, raw_ast_dummy, connected_matrix
    ):
        agent = AlgebraicTacticsAgent()
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state_agent = agent(raw_ast_dummy)

            ring = Phase1_AlgebraicRingAuditor.audit_ring_manifold(raw_ast_dummy)
            skel = Phase2_TopologicalSkeletonAuditor.audit_simplicial_skeleton(ring)

        assert state_agent.payload["betti_0"] == skel.betti_0
        assert state_agent.payload["betti_1"] == skel.betti_1
        assert state_agent.payload["fiedler_value"] == pytest.approx(skel.fiedler_value)
        assert state_agent.payload["incidence_rank"] == skel.incidence_rank
        assert state_agent.payload["matrix_rank"] == ring.matrix_rank

    def test_exports_all_public_symbols(self):
        import app.tactics.algebraic_tactics_agent as mod

        expected = {
            "AlgebraicConstants",
            "OptionMonad",
            "AlgebraicTacticsError",
            "RingSymmetryViolation",
            "RingDegeneracyError",
            "TopologicalIslandError",
            "HomologicalInvariantError",
            "EmptyCostTensorError",
            "RingHomogeneityValidation",
            "SimplicialSkeletonAudit",
            "Phase1_AlgebraicRingAuditor",
            "Phase2_TopologicalSkeletonAuditor",
            "AlgebraicTacticsAgent",
        }
        assert expected.issubset(set(mod.__all__))

    def test_fiedler_leq_spectral_radius_laplacian(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        """λ₂ ≤ λ_max siempre."""
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.fiedler_value <= skel.laplacian_spectrum[-1] + 1e-12

    def test_rank_B_equals_n_minus_beta0(
        self, phase1, phase2, raw_ast_dummy, connected_matrix
    ):
        p, _ = _mock_processor_returning(connected_matrix)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.incidence_rank == skel.num_vertices - skel.betti_0


class TestEdgeCasesAndRobustness:
    """Casos límite y robustez numérica."""

    def test_two_vertices_connected(self, phase1, phase2, raw_ast_dummy):
        M = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.betti_0 == 1
        assert skel.num_edges == 1
        assert skel.betti_1 == 0

    def test_two_vertices_orthogonal_disconnected(
        self, phase1, phase2, raw_ast_dummy
    ):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            with pytest.raises(TopologicalIslandError):
                phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()

    def test_large_connected_random(self, phase1, phase2, raw_ast_dummy):
        rng = np.random.default_rng(123)
        # Matriz con sesgo → casi seguro conexo
        M = rng.standard_normal((20, 5)) + 0.5
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.is_connected is True
        assert skel.betti_0 == 1
        assert skel.num_vertices == 20

    def test_negative_entries_still_form_ring(
        self, phase1, raw_ast_dummy
    ):
        M = np.array(
            [[-1.0, 2.0], [3.0, -4.0], [-5.0, 6.0]],
            dtype=np.float64,
        )
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        assert ring.is_distributive is True
        assert ring.is_closed is True

    def test_very_small_values_near_eps(self, phase1, raw_ast_dummy):
        M = np.array(
            [
                [1e-15, 1.0],
                [1.0, 1e-15],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            assert ring.matrix_rank >= 1
        finally:
            p.stop()

    def test_integer_matrix_coerced_to_float(
        self, phase1, raw_ast_dummy
    ):
        M = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
        finally:
            p.stop()
        assert ring.cost_matrix.dtype == np.float64

    def test_single_feature_column(self, phase1, phase2, raw_ast_dummy):
        """Una sola columna de features: filas escalares → grafo según signos/magnitudes."""
        M = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        p, _ = _mock_processor_returning(M)
        try:
            ring = phase1.audit_ring_manifold(raw_ast_dummy)
            # Productos internos todos positivos → K_4 conexo
            skel = phase2.audit_simplicial_skeleton(ring)
        finally:
            p.stop()
        assert skel.betti_0 == 1
        assert skel.is_connected is True

    def test_agent_empty_ast_list_still_calls_processor(
        self, agent_default, connected_matrix
    ):
        """AST vacío es lista válida; el processor decide la matriz."""
        with patch(
            "app.tactics.algebraic_tactics_agent.APUProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = connected_matrix
            state = agent_default([])
        assert state.stratum == Stratum.TACTICS
        MockProc.return_value.process.assert_called_once_with([])


class TestOptionMonadLaws:
    """Leyes monádicas (izquierda/derecha unitarias y asociatividad) sobre floats finitos."""

    def test_left_identity(self):
        """unit(a).bind(f) ≡ f(a)"""
        a = 3.0
        f = lambda x: OptionMonad.unit(x * 2)
        assert OptionMonad.unit(a).bind(f).value == pytest.approx(f(a).value)

    def test_right_identity(self):
        """m.bind(unit) ≡ m"""
        m = OptionMonad.unit(5.0)
        assert m.bind(OptionMonad.unit).value == pytest.approx(m.value)

    def test_associativity(self):
        """m.bind(f).bind(g) ≡ m.bind(λx. f(x).bind(g))"""
        m = OptionMonad.unit(2.0)
        f = lambda x: OptionMonad.unit(x + 1)
        g = lambda x: OptionMonad.unit(x * 3)
        left = m.bind(f).bind(g)
        right = m.bind(lambda x: f(x).bind(g))
        assert left.value == pytest.approx(right.value)

    def test_nothing_absorbing(self):
        """Nothing es absorbente por la izquierda para bind."""
        f = lambda x: OptionMonad.unit(x + 1)
        assert OptionMonad.nothing().bind(f).is_singular is True


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZACIÓN ADICIONAL
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "value,is_sing",
    [
        (0.0, False),
        (1.0, False),
        (-3.5, False),
        (float("nan"), True),
        (float("inf"), True),
        (float("-inf"), True),
        (None, True),
    ],
)
def test_option_unit_parametrized(value, is_sing):
    m = OptionMonad.unit(value)  # type: ignore[arg-type]
    assert m.is_singular is is_sing


@pytest.mark.parametrize(
    "n,d",
    [
        (2, 1),
        (3, 2),
        (5, 3),
        (8, 4),
    ],
)
def test_connected_path_parametrized(n, d, phase1, phase2, raw_ast_dummy):
    M = _connected_path_matrix(n, d)
    p, _ = _mock_processor_returning(M)
    try:
        ring = phase1.audit_ring_manifold(raw_ast_dummy)
        skel = phase2.audit_simplicial_skeleton(ring)
    finally:
        p.stop()
    assert skel.is_connected is True
    assert skel.betti_0 == 1
    assert skel.num_vertices == n


@pytest.mark.parametrize(
    "matrix_factory",
    [
        lambda: _connected_path_matrix(4, 2),
        lambda: _fully_connected_clique(5, 3),
        lambda: _rank_one_matrix(),
        lambda: _single_vertex(),
    ],
)
def test_ring_axioms_hold_parametrized(matrix_factory, phase1, raw_ast_dummy):
    M = matrix_factory()
    p, _ = _mock_processor_returning(M)
    try:
        ring = phase1.audit_ring_manifold(raw_ast_dummy)
    finally:
        p.stop()
    assert ring.is_closed
    assert ring.is_commutative
    assert ring.is_associative
    assert ring.is_distributive
    assert ring.has_additive_identity
    assert ring.has_multiplicative_identity