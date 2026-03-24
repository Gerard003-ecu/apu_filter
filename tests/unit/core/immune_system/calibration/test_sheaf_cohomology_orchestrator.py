"""
Suite de Pruebas Rigurosas: Sheaf Cohomology Orchestrator
Ubicación: tests/test_sheaf_cohomology_orchestrator.py

Cobertura Matemática:
─────────────────────
1.  Estructuras inmutables (RestrictionMap, SheafEdge, SpectralInvariants)
2.  Validación de nodos y aristas (CellularSheaf.__init__, add_edge)
3.  Ensamblaje del operador de cofrontera δ: C⁰ → C¹
4.  Propiedades algebraicas de δ (dimensiones, estructura por bloques)
5.  Laplaciano del haz L = δᵀδ (simetría, semi-positividad)
6.  Energía de Dirichlet E(x) = ‖δx‖²
7.  Análisis espectral denso y disperso
8.  Orquestador: auditoría de estados globales
9.  Consistencia cohomológica (H⁰, brecha espectral)
10. Casos límite y patologías numéricas
11. Invariantes del cono semi-definido positivo
12. Haces con fibras de dimensión heterogénea
13. Constantes y excepciones del módulo

Convenciones:
─────────────
- Cada clase agrupa un aspecto matemático o funcional coherente
- Los nombres siguen: test_<qué>_<condición>_<resultado_esperado>
- Tolerancias numéricas justificadas explícitamente
- Tests autocontenidos (sin dependencia inter-test)
- Semillas fijas para reproducibilidad estocástica
"""
from __future__ import annotations

import os

# FASE 1: Certificación Espectral en el Vacío Termodinámico
# Bloquear la entropía termodinámica forzando ejecución estrictamente en un solo hilo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from typing import Dict, Final

import numpy as np
import pytest
import scipy.sparse as sp

from app.core.immune_system.calibration.sheaf_cohomology_orchestrator import (
    CellularSheaf,
    GlobalFrustrationAssessment,
    HomologicalInconsistencyError,
    RestrictionMap,
    SheafCohomologyError,
    SheafCohomologyOrchestrator,
    SheafDegeneracyError,
    SheafEdge,
    SpectralComputationError,
    SpectralInvariants,
    _SpectralAnalyzer,
    _ARPACK_TOLERANCE,
    _DENSE_SPECTRAL_MAX_DIM,
    _FRUSTRATION_TOLERANCE,
    _SPARSE_MAX_EIGENVALUES,
    _SPECTRAL_TOLERANCE,
    _SYMMETRY_TOLERANCE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

_FLOAT64_RTOL: Final[float] = 1e-12
_FLOAT64_ATOL: Final[float] = 1e-14


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE CONSTRUCCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _build_simple_sheaf_2_nodes_1_edge(
    d_nodes: int = 2,
    d_edge: int = 2,
    F_u: np.ndarray | None = None,
    F_v: np.ndarray | None = None,
) -> CellularSheaf:
    """
    Construye un haz minimal: 2 nodos, 1 arista, fibras uniformes.

    Grafo: 0 ——(e=0)——> 1

    Si no se especifican mapas, se usan identidades.
    """
    if F_u is None:
        F_u = np.eye(d_edge, d_nodes, dtype=np.float64)
    if F_v is None:
        F_v = np.eye(d_edge, d_nodes, dtype=np.float64)

    sheaf = CellularSheaf(
        num_nodes=2,
        node_dims={0: d_nodes, 1: d_nodes},
        edge_dims={0: d_edge},
    )
    sheaf.add_edge(
        edge_id=0, u=0, v=1,
        F_ue=RestrictionMap(F_u),
        F_ve=RestrictionMap(F_v),
    )
    return sheaf


def _build_triangle_sheaf(
    dim: int = 1,
) -> CellularSheaf:
    """
    Construye un haz sobre el grafo triangular K₃.

    Grafo:
        0 ——(e=0)——> 1
        1 ——(e=1)——> 2
        0 ——(e=2)——> 2

    Todos los espacios de fibra tienen dimensión `dim`.
    Todos los mapas de restricción son la identidad dim×dim.
    """
    I = np.eye(dim, dtype=np.float64)

    sheaf = CellularSheaf(
        num_nodes=3,
        node_dims={0: dim, 1: dim, 2: dim},
        edge_dims={0: dim, 1: dim, 2: dim},
    )
    sheaf.add_edge(0, u=0, v=1, F_ue=RestrictionMap(I), F_ve=RestrictionMap(I))
    sheaf.add_edge(1, u=1, v=2, F_ue=RestrictionMap(I), F_ve=RestrictionMap(I))
    sheaf.add_edge(2, u=0, v=2, F_ue=RestrictionMap(I), F_ve=RestrictionMap(I))
    return sheaf


def _build_path_sheaf(
    n_nodes: int,
    dim: int = 1,
) -> CellularSheaf:
    """
    Construye un haz sobre el grafo camino P_n.

    Grafo: 0 → 1 → 2 → ... → (n-1)

    n-1 aristas, todas con identidad dim×dim.
    """
    I = np.eye(dim, dtype=np.float64)

    sheaf = CellularSheaf(
        num_nodes=n_nodes,
        node_dims={i: dim for i in range(n_nodes)},
        edge_dims={i: dim for i in range(n_nodes - 1)},
    )
    for i in range(n_nodes - 1):
        sheaf.add_edge(
            i, u=i, v=i + 1,
            F_ue=RestrictionMap(I),
            F_ve=RestrictionMap(I),
        )
    return sheaf


def _build_heterogeneous_sheaf() -> CellularSheaf:
    """
    Construye un haz con fibras de dimensión heterogénea.

    Nodo 0: dim 2
    Nodo 1: dim 3
    Arista 0: dim 2

    F_{0▷e}: ℝ² → ℝ² (identidad 2×2)
    F_{1▷e}: ℝ³ → ℝ² (proyección, primeras 2 componentes)
    """
    F_0 = np.eye(2, dtype=np.float64)
    F_1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    sheaf = CellularSheaf(
        num_nodes=2,
        node_dims={0: 2, 1: 3},
        edge_dims={0: 2},
    )
    sheaf.add_edge(
        0, u=0, v=1,
        F_ue=RestrictionMap(F_0),
        F_ve=RestrictionMap(F_1),
    )
    return sheaf


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_sheaf():
    """Haz minimal: 2 nodos, 1 arista, fibras dim=2, mapas identidad."""
    return _build_simple_sheaf_2_nodes_1_edge()


@pytest.fixture
def triangle_sheaf():
    """Haz sobre K₃ con fibras dim=1 y mapas identidad."""
    return _build_triangle_sheaf(dim=1)


@pytest.fixture
def path_sheaf_4():
    """Haz sobre camino P₄ con fibras dim=1 y mapas identidad."""
    return _build_path_sheaf(n_nodes=4, dim=1)


@pytest.fixture
def heterogeneous_sheaf():
    """Haz con fibras de dimensión no uniforme."""
    return _build_heterogeneous_sheaf()


@pytest.fixture
def orchestrator():
    """Instancia del orquestador."""
    return SheafCohomologyOrchestrator


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 1: RESTRICTION MAP
# ═══════════════════════════════════════════════════════════════════════════════

class TestRestrictionMap:
    """Verifica la construcción, validación e inmutabilidad de RestrictionMap."""

    def test_valid_identity_map(self):
        """Una identidad 2×2 se acepta correctamente."""
        rm = RestrictionMap(np.eye(2, dtype=np.float64))
        assert rm.matrix.shape == (2, 2)
        assert rm.domain_dim == 2
        assert rm.codomain_dim == 2

    def test_valid_rectangular_map(self):
        """Un mapa rectangular (m×n con m≠n) se acepta."""
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        rm = RestrictionMap(M)
        assert rm.domain_dim == 3
        assert rm.codomain_dim == 2

    def test_matrix_is_float64(self):
        """La matriz interna debe ser float64."""
        rm = RestrictionMap(np.eye(2, dtype=np.int32))
        assert rm.matrix.dtype == np.float64

    def test_matrix_is_immutable(self):
        """La matriz interna debe ser read-only."""
        rm = RestrictionMap(np.eye(2))
        with pytest.raises(ValueError, match="read-only|not writeable"):
            rm.matrix[0, 0] = 999.0

    def test_matrix_is_independent_copy(self):
        """Modificar la entrada original no afecta al RestrictionMap."""
        M = np.eye(2, dtype=np.float64)
        rm = RestrictionMap(M)
        M[0, 0] = 999.0
        assert rm.matrix[0, 0] == 1.0

    def test_rejects_1d_array(self):
        """Un vector 1D no es un mapa lineal (requiere 2D)."""
        with pytest.raises(SheafDegeneracyError, match="2D"):
            RestrictionMap(np.array([1.0, 2.0]))

    def test_rejects_3d_array(self):
        """Un tensor 3D no es un mapa lineal."""
        with pytest.raises(SheafDegeneracyError, match="2D"):
            RestrictionMap(np.ones((2, 2, 2)))

    def test_rejects_nan(self):
        """Entradas NaN son rechazadas."""
        M = np.eye(2, dtype=np.float64)
        M[0, 1] = np.nan
        with pytest.raises(SheafDegeneracyError, match="no finita"):
            RestrictionMap(M)

    def test_rejects_inf(self):
        """Entradas ±∞ son rechazadas."""
        M = np.eye(2, dtype=np.float64)
        M[1, 1] = np.inf
        with pytest.raises(SheafDegeneracyError, match="no finita"):
            RestrictionMap(M)

    def test_rejects_empty_matrix(self):
        """Matriz 0×n o n×0 es degenerada."""
        with pytest.raises(SheafDegeneracyError, match="degenerada"):
            RestrictionMap(np.array([]).reshape(0, 2))

    def test_rejects_non_convertible_input(self):
        """Entrada no convertible a array lanza error."""
        with pytest.raises(SheafDegeneracyError, match="convertible"):
            RestrictionMap("not a matrix")

    def test_frozen_dataclass(self):
        """RestrictionMap es inmutable (frozen)."""
        rm = RestrictionMap(np.eye(2))
        with pytest.raises(AttributeError):
            rm.matrix = np.eye(3)

    def test_accepts_list_of_lists(self):
        """Acepta listas anidadas convertibles."""
        rm = RestrictionMap([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(rm.matrix, np.eye(2))


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 2: CELLULAR SHEAF - CONSTRUCCIÓN Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestCellularSheafConstruction:
    """Verifica la construcción y validación del CellularSheaf."""

    def test_valid_construction(self):
        """Construcción válida con 2 nodos y 1 arista."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 2, 1: 3},
            edge_dims={0: 2},
        )
        assert sheaf.num_nodes == 2
        assert sheaf.total_node_dim == 5
        assert sheaf.total_edge_dim == 2

    def test_rejects_zero_nodes(self):
        """0 nodos es inválido."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=0, node_dims={}, edge_dims={0: 1})

    def test_rejects_negative_nodes(self):
        """Número negativo de nodos es inválido."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=-1, node_dims={}, edge_dims={0: 1})

    def test_rejects_float_nodes(self):
        """Número flotante de nodos es inválido."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=2.5, node_dims={0: 1, 1: 1}, edge_dims={0: 1})

    def test_rejects_missing_node_dim(self):
        """Nodo sin dimensión especificada lanza error."""
        with pytest.raises(SheafDegeneracyError, match="Faltan"):
            CellularSheaf(num_nodes=3, node_dims={0: 1, 1: 1}, edge_dims={0: 1})

    def test_rejects_extra_node_keys(self):
        """Claves espurias en node_dims lanza error."""
        with pytest.raises(SheafDegeneracyError, match="fuera del rango"):
            CellularSheaf(
                num_nodes=2,
                node_dims={0: 1, 1: 1, 5: 1},
                edge_dims={0: 1},
            )

    def test_rejects_zero_node_dim(self):
        """Dimensión 0 para un nodo es inválida."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=2, node_dims={0: 1, 1: 0}, edge_dims={0: 1})

    def test_rejects_negative_node_dim(self):
        """Dimensión negativa para un nodo es inválida."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=1, node_dims={0: -1}, edge_dims={0: 1})

    def test_rejects_empty_edge_dims(self):
        """Un haz sin aristas es degenerado."""
        with pytest.raises(SheafDegeneracyError, match="vacío"):
            CellularSheaf(num_nodes=2, node_dims={0: 1, 1: 1}, edge_dims={})

    def test_rejects_negative_edge_id(self):
        """Identificador de arista negativo es inválido."""
        with pytest.raises(SheafDegeneracyError, match="Identificador"):
            CellularSheaf(num_nodes=2, node_dims={0: 1, 1: 1}, edge_dims={-1: 1})

    def test_rejects_zero_edge_dim(self):
        """Dimensión 0 para una arista es inválida."""
        with pytest.raises(SheafDegeneracyError, match="entero positivo"):
            CellularSheaf(num_nodes=2, node_dims={0: 1, 1: 1}, edge_dims={0: 0})

    def test_rejects_node_dims_not_dict(self):
        """node_dims que no es dict lanza error."""
        with pytest.raises(SheafDegeneracyError, match="diccionario"):
            CellularSheaf(num_nodes=1, node_dims=[1], edge_dims={0: 1})

    def test_rejects_edge_dims_not_dict(self):
        """edge_dims que no es dict lanza error."""
        with pytest.raises(SheafDegeneracyError, match="diccionario"):
            CellularSheaf(num_nodes=2, node_dims={0: 1, 1: 1}, edge_dims=[1])

    def test_node_dims_returns_copy(self):
        """La propiedad node_dims retorna copia defensiva."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1},
        )
        dims = sheaf.node_dims
        dims[0] = 999
        assert sheaf.node_dims[0] == 1

    def test_edge_dims_returns_copy(self):
        """La propiedad edge_dims retorna copia defensiva."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1},
        )
        dims = sheaf.edge_dims
        dims[0] = 999
        assert sheaf.edge_dims[0] == 1

    def test_is_fully_assembled_initially_false(self):
        """Antes de añadir aristas, el haz no está ensamblado."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1},
        )
        assert not sheaf.is_fully_assembled

    def test_is_fully_assembled_after_all_edges(self, simple_sheaf):
        """Después de añadir todas las aristas, el haz está ensamblado."""
        assert simple_sheaf.is_fully_assembled

    def test_num_edges_tracking(self):
        """Conteo de aristas añadidas vs esperadas."""
        sheaf = CellularSheaf(
            num_nodes=3,
            node_dims={0: 1, 1: 1, 2: 1},
            edge_dims={0: 1, 1: 1},
        )
        assert sheaf.num_edges_added == 0
        assert sheaf.num_edges_expected == 2

        I = np.eye(1, dtype=np.float64)
        sheaf.add_edge(0, 0, 1, RestrictionMap(I), RestrictionMap(I))
        assert sheaf.num_edges_added == 1
        assert not sheaf.is_fully_assembled

        sheaf.add_edge(1, 1, 2, RestrictionMap(I), RestrictionMap(I))
        assert sheaf.num_edges_added == 2
        assert sheaf.is_fully_assembled


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 3: ADD_EDGE - VALIDACIONES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddEdge:
    """Verifica todas las validaciones de add_edge."""

    def _make_sheaf(self, n_nodes=3, node_dim=1, edge_ids=None):
        """Helper para crear un haz base."""
        if edge_ids is None:
            edge_ids = [0, 1]
        return CellularSheaf(
            num_nodes=n_nodes,
            node_dims={i: node_dim for i in range(n_nodes)},
            edge_dims={eid: node_dim for eid in edge_ids},
        )

    def test_rejects_unknown_edge_id(self):
        """edge_id no declarado en edge_dims es rechazado."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        with pytest.raises(SheafDegeneracyError, match="no existe"):
            sheaf.add_edge(99, 0, 1, I, I)

    def test_rejects_duplicate_edge_id(self):
        """edge_id duplicado es rechazado."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        sheaf.add_edge(0, 0, 1, I, I)
        with pytest.raises(SheafDegeneracyError, match="ya fue añadida"):
            sheaf.add_edge(0, 0, 2, I, I)

    def test_rejects_node_out_of_range(self):
        """Nodo fuera de [0, num_nodes) es rechazado."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        with pytest.raises(SheafDegeneracyError, match="fuera de rango"):
            sheaf.add_edge(0, 0, 10, I, I)

    def test_rejects_negative_node(self):
        """Nodo negativo es rechazado."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        with pytest.raises(SheafDegeneracyError, match="fuera de rango"):
            sheaf.add_edge(0, -1, 1, I, I)

    def test_rejects_self_loop(self):
        """Lazo u=v es rechazado."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        with pytest.raises(SheafDegeneracyError, match="lazo"):
            sheaf.add_edge(0, 1, 1, I, I)

    def test_rejects_duplicate_node_pair(self):
        """Par {u,v} duplicado es rechazado (grafo simple)."""
        sheaf = self._make_sheaf()
        I = RestrictionMap(np.eye(1))
        sheaf.add_edge(0, 0, 1, I, I)
        with pytest.raises(SheafDegeneracyError, match="Ya existe"):
            sheaf.add_edge(1, 1, 0, I, I)

    def test_rejects_incompatible_restriction_shape_u(self):
        """Mapa F_ue con forma incorrecta es rechazado."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 2, 1: 3},
            edge_dims={0: 2},
        )
        wrong_shape = RestrictionMap(np.eye(2, 3, dtype=np.float64))
        correct = RestrictionMap(np.eye(2, 3, dtype=np.float64))
        with pytest.raises(SheafDegeneracyError, match="Incoherencia dimensional.*u"):
            sheaf.add_edge(0, 0, 1, wrong_shape, correct)

    def test_rejects_incompatible_restriction_shape_v(self):
        """Mapa F_ve con forma incorrecta es rechazado."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 2, 1: 3},
            edge_dims={0: 2},
        )
        correct_u = RestrictionMap(np.eye(2, dtype=np.float64))
        wrong_v = RestrictionMap(np.eye(2, dtype=np.float64))
        with pytest.raises(SheafDegeneracyError, match="Incoherencia dimensional.*v"):
            sheaf.add_edge(0, 0, 1, correct_u, wrong_v)

    def test_invalidates_cache_on_add(self):
        """Añadir una arista invalida el caché de δ."""
        sheaf = CellularSheaf(
            num_nodes=3,
            node_dims={0: 1, 1: 1, 2: 1},
            edge_dims={0: 1, 1: 1},
        )
        I = RestrictionMap(np.eye(1))
        sheaf.add_edge(0, 0, 1, I, I)
        with pytest.raises(SheafDegeneracyError, match="no está completamente ensamblado"):
             _ = sheaf.build_coboundary_operator()

        sheaf.add_edge(1, 1, 2, I, I)
        _ = sheaf.build_coboundary_operator()
        assert sheaf._cached_coboundary is not None

        # Add an extra edge to invalidate
        sheaf._edge_dims[2] = 1 # Hack to allow adding
        sheaf.add_edge(2, 0, 2, I, I)
        assert sheaf._cached_coboundary is None


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 4: OPERADOR DE COFRONTERA δ
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoboundaryOperator:
    """
    Verifica la construcción y propiedades algebraicas del operador δ.

    Para el haz constante con F(v) = F(e) = ℝ y mapas identidad,
    δ coincide con la matriz de incidencia orientada del grafo.
    """

    def test_shape_correct(self, simple_sheaf):
        """δ tiene forma (total_edge_dim, total_node_dim)."""
        delta = simple_sheaf.build_coboundary_operator()
        assert delta.shape == (
            simple_sheaf.total_edge_dim,
            simple_sheaf.total_node_dim,
        )

    def test_is_sparse_csc(self, simple_sheaf):
        """δ es una matriz CSC dispersa."""
        delta = simple_sheaf.build_coboundary_operator()
        assert isinstance(delta, sp.csc_matrix)

    def test_dtype_is_float64(self, simple_sheaf):
        """δ tiene dtype float64."""
        delta = simple_sheaf.build_coboundary_operator()
        assert delta.dtype == np.float64

    def test_all_entries_finite(self, simple_sheaf):
        """Todas las entradas de δ son finitas."""
        delta = simple_sheaf.build_coboundary_operator()
        if delta.nnz > 0:
            assert np.all(np.isfinite(delta.data))

    def test_incidence_matrix_for_constant_sheaf_path(self):
        """
        Para el haz constante sobre P₃ (camino 0→1→2) con dim=1:
        δ = [[-1, 1, 0],
             [ 0,-1, 1]]
        """
        sheaf = _build_path_sheaf(n_nodes=3, dim=1)
        delta = sheaf.build_coboundary_operator()
        expected = np.array([
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
        ], dtype=np.float64)
        np.testing.assert_array_equal(delta.toarray(), expected)

    def test_incidence_matrix_for_constant_sheaf_triangle(self, triangle_sheaf):
        """
        Para K₃ con aristas (0→1, 1→2, 0→2) y dim=1:
        δ = [[-1, 1, 0],   # e=0: v=1 − u=0
             [ 0,-1, 1],   # e=1: v=2 − u=1
             [-1, 0, 1]]   # e=2: v=2 − u=0
        """
        delta = triangle_sheaf.build_coboundary_operator()
        expected = np.array([
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ], dtype=np.float64)
        np.testing.assert_array_equal(delta.toarray(), expected)

    def test_block_structure_heterogeneous_sheaf(self, heterogeneous_sheaf):
        """
        Para el haz heterogéneo con nodo 0 (dim 2), nodo 1 (dim 3),
        arista 0 (dim 2):

        δ tiene forma (2, 5) con bloques:
          columnas [0,1]: -F₀ = -I₂
          columnas [2,3,4]: +F₁ = +[I₂ | 0]
        """
        delta = heterogeneous_sheaf.build_coboundary_operator()
        assert delta.shape == (2, 5)

        D = delta.toarray()
        # Bloque u (nodo 0): -I₂ en columnas 0,1
        np.testing.assert_array_equal(D[:, :2], -np.eye(2))
        # Bloque v (nodo 1): proyección en columnas 2,3,4
        expected_v = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_array_equal(D[:, 2:], expected_v)

    def test_rejects_incomplete_sheaf(self):
        """Construir δ sin todas las aristas añadidas lanza error."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1, 1: 1},
        )
        I = RestrictionMap(np.eye(1))
        sheaf.add_edge(0, 0, 1, I, I)
        with pytest.raises(SheafDegeneracyError, match="no está completamente"):
            sheaf.build_coboundary_operator()

    def test_caching_returns_same_object(self, simple_sheaf):
        """Llamadas repetidas a build_coboundary_operator retornan el mismo objeto."""
        delta1 = simple_sheaf.build_coboundary_operator()
        delta2 = simple_sheaf.build_coboundary_operator()
        assert delta1 is delta2

    def test_delta_x_consensus_is_zero(self, triangle_sheaf):
        """
        Para el haz constante, si todos los nodos tienen el mismo valor
        (consenso), entonces δx = 0.
        """
        delta = triangle_sheaf.build_coboundary_operator()
        # x = (c, c, c) para c = 3.14
        x = np.full(3, 3.14, dtype=np.float64)
        residual = delta.dot(x)
        np.testing.assert_allclose(
            residual, np.zeros(3), atol=_FLOAT64_ATOL
        )

    def test_delta_x_disagreement_nonzero(self, simple_sheaf):
        """
        Para el haz simple con identidades, si x_0 ≠ x_1 entonces δx ≠ 0.
        """
        delta = simple_sheaf.build_coboundary_operator()
        x = np.array([1.0, 0.0, 0.0, 2.0], dtype=np.float64)
        residual = delta.dot(x)
        assert np.linalg.norm(residual) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 5: LAPLACIANO DEL HAZ L = δᵀδ
# ═══════════════════════════════════════════════════════════════════════════════

class TestSheafLaplacian:
    """
    Verifica propiedades algebraicas del Laplaciano L = δᵀδ.
    """

    def test_shape_square(self, simple_sheaf):
        """L es cuadrada de dimensión total_node_dim."""
        L = simple_sheaf.compute_sheaf_laplacian()
        n = simple_sheaf.total_node_dim
        assert L.shape == (n, n)

    def test_is_symmetric(self, triangle_sheaf):
        """L = δᵀδ es simétrica."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        L_dense = L.toarray()
        np.testing.assert_allclose(
            L_dense, L_dense.T, atol=_SYMMETRY_TOLERANCE
        )

    def test_is_semidefinite_positive(self, triangle_sheaf):
        """Todos los eigenvalores de L son ≥ 0."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        eigvals = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigvals >= -_SPECTRAL_TOLERANCE)

    def test_kernel_contains_consensus(self, triangle_sheaf):
        """
        Para el haz constante, el vector de consenso (1,1,...,1) está
        en ker(L), es decir, L·1 = 0.
        """
        L = triangle_sheaf.compute_sheaf_laplacian()
        ones = np.ones(triangle_sheaf.total_node_dim, dtype=np.float64)
        result = L.dot(ones)
        np.testing.assert_allclose(
            result, np.zeros_like(result), atol=_FLOAT64_ATOL
        )

    def test_kernel_dimension_connected_constant_sheaf(self, triangle_sheaf):
        """
        Para un grafo conexo con haz constante escalar, dim ker(L) = 1.
        """
        L = triangle_sheaf.compute_sheaf_laplacian()
        eigvals = np.linalg.eigvalsh(L.toarray())
        nullity = int(np.sum(np.abs(eigvals) <= _SPECTRAL_TOLERANCE))
        assert nullity == 1

    def test_dirichlet_energy_equals_quadratic_form(self, triangle_sheaf):
        """
        E(x) = ‖δx‖² = xᵀLx para todo x.
        """
        delta = triangle_sheaf.build_coboundary_operator()
        L = triangle_sheaf.compute_sheaf_laplacian()

        rng = np.random.default_rng(seed=42)
        n = triangle_sheaf.total_node_dim
        for _ in range(50):
            x = rng.standard_normal(n)
            residual = delta.dot(x)
            energy_delta = float(np.dot(residual, residual))
            energy_L = float(x @ L.toarray() @ x)
            np.testing.assert_allclose(
                energy_delta, energy_L,
                rtol=_FLOAT64_RTOL, atol=_FLOAT64_ATOL,
            )

    def test_laplacian_path_graph(self, path_sheaf_4):
        """
        Para P₄ con haz constante escalar:
        L = δᵀδ es el Laplaciano del grafo camino.

        L = [[ 1, -1,  0,  0],
             [-1,  2, -1,  0],
             [ 0, -1,  2, -1],
             [ 0,  0, -1,  1]]
        """
        L = path_sheaf_4.compute_sheaf_laplacian()
        expected = np.array([
            [1, -1, 0, 0],
            [-1, 2, -1, 0],
            [0, -1, 2, -1],
            [0, 0, -1, 1],
        ], dtype=np.float64)
        np.testing.assert_allclose(
            L.toarray(), expected, atol=_FLOAT64_ATOL
        )

    def test_trace_equals_sum_of_squared_norms(self, triangle_sheaf):
        """
        tr(L) = tr(δᵀδ) = ‖δ‖²_F = Σᵢⱼ δᵢⱼ².
        """
        delta = triangle_sheaf.build_coboundary_operator()
        L = triangle_sheaf.compute_sheaf_laplacian()
        trace_L = float(L.diagonal().sum())
        frobenius_sq = float(np.sum(delta.toarray() ** 2))
        np.testing.assert_allclose(
            trace_L, frobenius_sq, rtol=_FLOAT64_RTOL
        )

    def test_laplacian_all_entries_finite(self, heterogeneous_sheaf):
        """Todas las entradas de L son finitas."""
        L = heterogeneous_sheaf.compute_sheaf_laplacian()
        if L.nnz > 0:
            assert np.all(np.isfinite(L.data))


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 6: ENERGÍA DE FRUSTRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrustrationEnergy:
    """
    Verifica el cálculo de E(x) = ‖δx‖².
    """

    def test_zero_for_consensus(self, triangle_sheaf, orchestrator):
        """E(x) = 0 para estado de consenso."""
        delta = triangle_sheaf.build_coboundary_operator()
        x = np.ones(3, dtype=np.float64)
        energy, norm = orchestrator._compute_frustration_energy(delta, x)
        assert energy == 0.0
        assert norm == 0.0

    def test_positive_for_disagreement(self, simple_sheaf, orchestrator):
        """E(x) > 0 para estado de desacuerdo."""
        delta = simple_sheaf.build_coboundary_operator()
        x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        energy, norm = orchestrator._compute_frustration_energy(delta, x)
        assert energy > 0
        assert norm > 0

    def test_returns_tuple(self, simple_sheaf, orchestrator):
        """_compute_frustration_energy retorna (energy, norm)."""
        delta = simple_sheaf.build_coboundary_operator()
        x = np.ones(simple_sheaf.total_node_dim, dtype=np.float64)
        result = orchestrator._compute_frustration_energy(delta, x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_energy_equals_norm_squared(self, triangle_sheaf, orchestrator):
        """E(x) = (‖δx‖)² = residual_norm²."""
        delta = triangle_sheaf.build_coboundary_operator()
        rng = np.random.default_rng(seed=42)
        x = rng.standard_normal(triangle_sheaf.total_node_dim)
        energy, norm = orchestrator._compute_frustration_energy(delta, x)
        np.testing.assert_allclose(
            energy, norm ** 2, rtol=_FLOAT64_RTOL
        )

    def test_energy_non_negative(self, triangle_sheaf, orchestrator):
        """E(x) ≥ 0 para todo x."""
        delta = triangle_sheaf.build_coboundary_operator()
        rng = np.random.default_rng(seed=42)
        for _ in range(100):
            x = rng.standard_normal(triangle_sheaf.total_node_dim)
            energy, _ = orchestrator._compute_frustration_energy(delta, x)
            assert energy >= 0.0

    def test_energy_scales_quadratically(self, simple_sheaf, orchestrator):
        """E(cx) = c² · E(x) (homogeneidad cuadrática)."""
        delta = simple_sheaf.build_coboundary_operator()
        rng = np.random.default_rng(seed=42)
        x = rng.standard_normal(simple_sheaf.total_node_dim)
        energy_x, _ = orchestrator._compute_frustration_energy(delta, x)

        for c in [0.5, 2.0, 3.0]:
            energy_cx, _ = orchestrator._compute_frustration_energy(delta, c * x)
            np.testing.assert_allclose(
                energy_cx, c ** 2 * energy_x,
                rtol=_FLOAT64_RTOL,
            )

    def test_energy_satisfies_parallelogram_law(
        self, triangle_sheaf, orchestrator,
    ):
        """
        La forma bilineal B(x,y) = xᵀLy satisface la ley del paralelogramo:
        ‖x+y‖² + ‖x-y‖² = 2(‖x‖² + ‖y‖²)
        donde ‖·‖ = ‖δ·‖.
        """
        delta = triangle_sheaf.build_coboundary_operator()
        rng = np.random.default_rng(seed=42)
        n = triangle_sheaf.total_node_dim

        for _ in range(50):
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)

            e_sum, _ = orchestrator._compute_frustration_energy(delta, x + y)
            e_diff, _ = orchestrator._compute_frustration_energy(delta, x - y)
            e_x, _ = orchestrator._compute_frustration_energy(delta, x)
            e_y, _ = orchestrator._compute_frustration_energy(delta, y)

            np.testing.assert_allclose(
                e_sum + e_diff,
                2.0 * (e_x + e_y),
                rtol=1e-10,
                err_msg="Ley del paralelogramo violada",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 7: VALIDACIÓN DEL ESTADO GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalStateValidation:
    """Verifica _validate_global_state_vector."""

    def test_valid_vector(self, simple_sheaf, orchestrator):
        """Vector válido se acepta."""
        x = np.ones(simple_sheaf.total_node_dim, dtype=np.float64)
        result = orchestrator._validate_global_state_vector(simple_sheaf, x)
        assert result.shape == (simple_sheaf.total_node_dim,)
        assert result.dtype == np.float64

    def test_rejects_wrong_length(self, simple_sheaf, orchestrator):
        """Vector con longitud incorrecta es rechazado."""
        x = np.ones(simple_sheaf.total_node_dim + 1)
        with pytest.raises(SheafDegeneracyError, match="Dimensión incompatible"):
            orchestrator._validate_global_state_vector(simple_sheaf, x)

    def test_rejects_2d_array(self, simple_sheaf, orchestrator):
        """Matriz 2D es rechazada (se requiere 1D)."""
        n = simple_sheaf.total_node_dim
        x = np.ones((n, 1))
        with pytest.raises(SheafDegeneracyError, match="1D"):
            orchestrator._validate_global_state_vector(simple_sheaf, x)

    def test_rejects_nan(self, simple_sheaf, orchestrator):
        """Vector con NaN es rechazado."""
        x = np.ones(simple_sheaf.total_node_dim)
        x[0] = np.nan
        with pytest.raises(SheafDegeneracyError, match="no finita"):
            orchestrator._validate_global_state_vector(simple_sheaf, x)

    def test_rejects_inf(self, simple_sheaf, orchestrator):
        """Vector con ∞ es rechazado."""
        x = np.ones(simple_sheaf.total_node_dim)
        x[-1] = np.inf
        with pytest.raises(SheafDegeneracyError, match="no finita"):
            orchestrator._validate_global_state_vector(simple_sheaf, x)

    def test_accepts_list(self, simple_sheaf, orchestrator):
        """Acepta listas convertibles a array."""
        x_list = [1.0] * simple_sheaf.total_node_dim
        result = orchestrator._validate_global_state_vector(simple_sheaf, x_list)
        assert isinstance(result, np.ndarray)

    def test_rejects_non_convertible(self, simple_sheaf, orchestrator):
        """Entrada no convertible lanza error."""
        with pytest.raises(SheafDegeneracyError, match="convertible"):
            orchestrator._validate_global_state_vector(simple_sheaf, "not array")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 8: ANÁLISIS ESPECTRAL DENSO
# ═══════════════════════════════════════════════════════════════════════════════

class TestDenseSpectralAnalysis:
    """Verifica _SpectralAnalyzer.compute_dense."""

    def test_identity_laplacian(self):
        """L = I tiene todos eigenvalores = 1, h0_dim = 0."""
        L = sp.csc_matrix(np.eye(3, dtype=np.float64))
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 0
        assert result.method == "dense"
        np.testing.assert_allclose(result.spectral_gap, 1.0, rtol=_FLOAT64_RTOL)

    def test_zero_laplacian(self):
        """L = 0 tiene todos eigenvalores = 0, h0_dim = n."""
        L = sp.csc_matrix((3, 3), dtype=np.float64)
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 3
        assert result.spectral_gap == 0.0

    def test_path_graph_laplacian_h0(self, path_sheaf_4):
        """El camino P₄ (conexo) tiene dim H⁰ = 1."""
        L = path_sheaf_4.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 1
        assert result.spectral_gap > 0.0

    def test_disconnected_graph_h0(self):
        """
        Un grafo con 2 componentes conexas y haz constante escalar
        tiene dim H⁰ = 2.

        Grafo: {0-1} ∪ {2-3} (dos aristas disjuntas)
        """
        I = np.eye(1, dtype=np.float64)
        sheaf = CellularSheaf(
            num_nodes=4,
            node_dims={0: 1, 1: 1, 2: 1, 3: 1},
            edge_dims={0: 1, 1: 1},
        )
        sheaf.add_edge(0, 0, 1, RestrictionMap(I), RestrictionMap(I))
        sheaf.add_edge(1, 2, 3, RestrictionMap(I), RestrictionMap(I))

        L = sheaf.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 2

    def test_triangle_spectral_gap(self, triangle_sheaf):
        """
        Para K₃ con haz constante, los eigenvalores de L son {0, 3, 3}
        (0 con multiplicidad 1, 3 con multiplicidad 2).
        """
        L = triangle_sheaf.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_dense(L)

        # FASE 1: asertar que h0_dimension coincide exactamente con la multiplicidad algebraica de λ=0
        multiplicity_zero = int(np.sum(np.abs(result.smallest_eigenvalues) <= _SPECTRAL_TOLERANCE))
        assert result.h0_dimension == multiplicity_zero, "Multiplicidad algebraica no coincide con h0_dimension"

        assert result.h0_dimension == 1
        np.testing.assert_allclose(
            result.spectral_gap, 3.0, rtol=1e-10
        )

    def test_eigenvalues_immutable(self):
        """Los eigenvalores en SpectralInvariants son read-only."""
        L = sp.csc_matrix(np.eye(2, dtype=np.float64))
        result = _SpectralAnalyzer.compute_dense(L)
        with pytest.raises(ValueError, match="read-only|not writeable"):
            result.smallest_eigenvalues[0] = 999.0

    def test_rejects_non_symmetric(self):
        """Laplaciano no simétrico es rechazado."""
        L_dense = np.array([[1.0, 0.5], [0.0, 1.0]])
        L = sp.csc_matrix(L_dense)
        with pytest.raises(SheafCohomologyError, match="simétrico"):
            _SpectralAnalyzer.compute_dense(L)

    def test_rejects_negative_definite(self):
        """Laplaciano negativo definido es rechazado."""
        L = sp.csc_matrix(-np.eye(2, dtype=np.float64))
        with pytest.raises(SpectralComputationError, match="semidefinido"):
            _SpectralAnalyzer.compute_dense(L)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 9: ANÁLISIS ESPECTRAL DISPERSO
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparseSpectralAnalysis:
    """Verifica _SpectralAnalyzer.compute_sparse."""

    def test_zero_dimensional(self):
        """Matriz 0×0 retorna invariantes triviales."""
        L = sp.csc_matrix((0, 0), dtype=np.float64)
        result = _SpectralAnalyzer.compute_sparse(L)
        assert result.h0_dimension == 0
        assert result.spectral_gap == 0.0
        assert result.method == "sparse"

    def test_one_dimensional(self):
        """Matriz 1×1 con L[0,0] = 0 tiene h0_dim = 1."""
        L = sp.csc_matrix(np.array([[0.0]]))
        result = _SpectralAnalyzer.compute_sparse(L)
        assert result.h0_dimension == 1

    def test_one_dimensional_nonzero(self):
        """Matriz 1×1 con L[0,0] > 0 tiene h0_dim = 0."""
        L = sp.csc_matrix(np.array([[5.0]]))
        result = _SpectralAnalyzer.compute_sparse(L)
        assert result.h0_dimension == 0
        np.testing.assert_allclose(result.spectral_gap, 5.0, rtol=1e-6)

    def test_path_graph_sparse(self, path_sheaf_4):
        """El camino P₄ tiene dim H⁰ = 1 (también via disperso)."""
        L = path_sheaf_4.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_sparse(L)
        assert result.h0_dimension == 1
        assert result.spectral_gap > 0.0

    def test_eigenvalues_immutable_sparse(self, path_sheaf_4):
        """Eigenvalores son inmutables en modo disperso."""
        L = path_sheaf_4.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_sparse(L)
        with pytest.raises(ValueError, match="read-only|not writeable"):
            result.smallest_eigenvalues[0] = 999.0

    def test_rejects_negative_eigenvalues(self):
        """Eigenvalores negativos severos lanzan error."""
        L = sp.csc_matrix(-5.0 * np.eye(3, dtype=np.float64))
        with pytest.raises(SpectralComputationError, match="semidefinido"):
            _SpectralAnalyzer.compute_sparse(L)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 10: ESTRATEGIA HÍBRIDA
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridSpectralStrategy:
    """Verifica que compute() selecciona correctamente dense vs sparse."""

    def test_small_uses_dense(self, triangle_sheaf):
        """Matrices pequeñas (n ≤ threshold) usan método denso."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        assert L.shape[0] <= _DENSE_SPECTRAL_MAX_DIM
        result = _SpectralAnalyzer.compute(L)
        assert result.method == "dense"

    def test_dense_and_sparse_agree_on_h0(self, path_sheaf_4):
        """Ambos métodos coinciden en dim H⁰ para problemas pequeños."""
        L = path_sheaf_4.compute_sheaf_laplacian()
        dense = _SpectralAnalyzer.compute_dense(L)
        sparse = _SpectralAnalyzer.compute_sparse(L)
        assert dense.h0_dimension == sparse.h0_dimension

    def test_dense_and_sparse_agree_on_gap(self, triangle_sheaf):
        """Ambos métodos coinciden en brecha espectral."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        dense = _SpectralAnalyzer.compute_dense(L)
        sparse = _SpectralAnalyzer.compute_sparse(L)
        np.testing.assert_allclose(
            dense.spectral_gap, sparse.spectral_gap, rtol=1e-4
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 11: SPECTRAL INVARIANTS DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpectralInvariants:
    """Verifica la estructura SpectralInvariants."""

    def test_valid_construction(self):
        """Construcción válida de SpectralInvariants."""
        eigs = np.array([0.0, 1.0, 3.0], dtype=np.float64)
        eigs.setflags(write=False)
        si = SpectralInvariants(
            h0_dimension=1,
            spectral_gap=1.0,
            smallest_eigenvalues=eigs,
            method="dense",
        )
        assert si.h0_dimension == 1
        assert si.spectral_gap == 1.0
        assert si.method == "dense"

    def test_rejects_negative_h0(self):
        """h0_dimension negativo es rechazado."""
        eigs = np.array([0.0], dtype=np.float64)
        eigs.setflags(write=False)
        with pytest.raises(ValueError, match="no negativo"):
            SpectralInvariants(
                h0_dimension=-1,
                spectral_gap=0.0,
                smallest_eigenvalues=eigs,
                method="dense",
            )

    def test_rejects_negative_gap(self):
        """spectral_gap negativo es rechazado."""
        eigs = np.array([0.0], dtype=np.float64)
        eigs.setflags(write=False)
        with pytest.raises(ValueError, match="no negativo"):
            SpectralInvariants(
                h0_dimension=0,
                spectral_gap=-0.1,
                smallest_eigenvalues=eigs,
                method="dense",
            )

    def test_rejects_invalid_method(self):
        """Método no reconocido es rechazado."""
        eigs = np.array([0.0], dtype=np.float64)
        eigs.setflags(write=False)
        with pytest.raises(ValueError, match="method"):
            SpectralInvariants(
                h0_dimension=0,
                spectral_gap=0.0,
                smallest_eigenvalues=eigs,
                method="unknown",
            )

    def test_frozen(self):
        """SpectralInvariants es inmutable."""
        eigs = np.array([0.0], dtype=np.float64)
        eigs.setflags(write=False)
        si = SpectralInvariants(
            h0_dimension=0,
            spectral_gap=0.0,
            smallest_eigenvalues=eigs,
            method="dense",
        )
        with pytest.raises(AttributeError):
            si.h0_dimension = 5


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 12: ORQUESTADOR - AUDITORÍA COMPLETA
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestrator:
    """Verifica el pipeline completo de audit_global_state."""

    def test_coherent_consensus_state(self, triangle_sheaf, orchestrator):
        """Estado de consenso pasa auditoría."""
        x = np.ones(triangle_sheaf.total_node_dim, dtype=np.float64) * 2.5
        result = orchestrator.audit_global_state(triangle_sheaf, x)

        assert isinstance(result, GlobalFrustrationAssessment)
        assert result.is_coherent is True
        assert result.frustration_energy <= _FRUSTRATION_TOLERANCE
        assert result.h0_dimension >= 1
        assert result.spectral_gap >= 0.0
        assert result.residual_norm >= 0.0
        assert result.spectral_method in ("dense", "sparse")

    def test_incoherent_state_raises(self, triangle_sheaf, orchestrator):
        """Estado incoherente lanza HomologicalInconsistencyError."""
        x = np.array([1.0, 100.0, -50.0], dtype=np.float64)
        with pytest.raises(
            HomologicalInconsistencyError,
            match="sección global compatible",
        ):
            orchestrator.audit_global_state(triangle_sheaf, x)

    def test_rejects_incomplete_sheaf(self, orchestrator):
        """Haz incompleto lanza error."""
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1, 1: 1},
        )
        I = RestrictionMap(np.eye(1))
        sheaf.add_edge(0, 0, 1, I, I)

        x = np.ones(2, dtype=np.float64)
        with pytest.raises(SheafDegeneracyError, match="no está completamente"):
            orchestrator.audit_global_state(sheaf, x)

    def test_zero_vector_is_coherent(self, simple_sheaf, orchestrator):
        """El vector cero es trivialmente coherente: δ·0 = 0."""
        x = np.zeros(simple_sheaf.total_node_dim, dtype=np.float64)
        result = orchestrator.audit_global_state(simple_sheaf, x)
        assert result.is_coherent is True
        assert result.frustration_energy == 0.0
        assert result.residual_norm == 0.0

    def test_heterogeneous_sheaf_consensus(
        self, heterogeneous_sheaf, orchestrator,
    ):
        """
        Para el haz heterogéneo (nodo 0: dim 2, nodo 1: dim 3),
        el estado x = (a, b, a, b, 0) satisface las restricciones:
        F_0·(a,b) = (a,b) y F_1·(a,b,0) = (a,b).
        """
        x = np.array([1.0, 2.0, 1.0, 2.0, 0.0], dtype=np.float64)
        result = orchestrator.audit_global_state(heterogeneous_sheaf, x)
        assert result.is_coherent is True
        np.testing.assert_allclose(
            result.frustration_energy, 0.0, atol=_FRUSTRATION_TOLERANCE
        )

    def test_heterogeneous_sheaf_inconsistent(
        self, heterogeneous_sheaf, orchestrator,
    ):
        """
        Para el haz heterogéneo, x = (1, 2, 3, 4, 5) viola restricciones:
        F_0·(1,2) = (1,2) pero F_1·(3,4,5) = (3,4) ≠ (1,2).
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        with pytest.raises(HomologicalInconsistencyError):
            orchestrator.audit_global_state(heterogeneous_sheaf, x)

    def test_path_graph_consistent_gradient(self, path_sheaf_4, orchestrator):
        """
        Para P₄ con haz constante: solo x constante es coherente.
        Un gradiente x = (0, 1, 2, 3) NO es coherente.
        """
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(HomologicalInconsistencyError):
            orchestrator.audit_global_state(path_sheaf_4, x)

    def test_assessment_fields_types(self, simple_sheaf, orchestrator):
        """Verifica tipos de todos los campos del assessment."""
        x = np.ones(simple_sheaf.total_node_dim, dtype=np.float64)
        result = orchestrator.audit_global_state(simple_sheaf, x)

        assert isinstance(result.frustration_energy, float)
        assert isinstance(result.h0_dimension, int)
        assert isinstance(result.is_coherent, bool)
        assert isinstance(result.spectral_gap, float)
        assert isinstance(result.residual_norm, float)
        assert isinstance(result.spectral_method, str)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 13: INVARIANTES COHOMOLÓGICOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCohomologicalInvariants:
    """
    Verifica propiedades de la cohomología del haz.

    H⁰(G; F) = ker(δ) = ker(L) son las secciones globales.
    dim H⁰ para el haz constante = número de componentes conexas.
    """

    def test_connected_graph_h0_equals_1(self, triangle_sheaf, orchestrator):
        """Grafo conexo con haz constante tiene dim H⁰ = 1."""
        x = np.ones(3, dtype=np.float64)
        result = orchestrator.audit_global_state(triangle_sheaf, x)
        assert result.h0_dimension == 1

    def test_path_h0_equals_1(self, path_sheaf_4, orchestrator):
        """P₄ (conexo) tiene dim H⁰ = 1."""
        x = np.ones(4, dtype=np.float64)
        result = orchestrator.audit_global_state(path_sheaf_4, x)
        assert result.h0_dimension == 1

    def test_two_components_h0_equals_2(self, orchestrator):
        """
        Grafo con 2 componentes conexas: dim H⁰ = 2.
        Nodos {0,1} conectados, nodos {2,3} conectados.
        """
        I = np.eye(1, dtype=np.float64)
        sheaf = CellularSheaf(
            num_nodes=4,
            node_dims={i: 1 for i in range(4)},
            edge_dims={0: 1, 1: 1},
        )
        sheaf.add_edge(0, 0, 1, RestrictionMap(I), RestrictionMap(I))
        sheaf.add_edge(1, 2, 3, RestrictionMap(I), RestrictionMap(I))

        # Consenso por componente: (c₁, c₁, c₂, c₂)
        x = np.array([5.0, 5.0, -3.0, -3.0], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x)
        assert result.h0_dimension == 2

    def test_spectral_gap_positive_for_connected(
        self, triangle_sheaf, orchestrator,
    ):
        """
        Grafo conexo tiene brecha espectral λ₁ > 0.
        La brecha mide la tasa de convergencia al consenso.
        """
        x = np.ones(3, dtype=np.float64)
        result = orchestrator.audit_global_state(triangle_sheaf, x)
        assert result.spectral_gap > 0.0

    def test_spectral_gap_k3_equals_3(self, triangle_sheaf, orchestrator):
        """
        Para K₃ con haz constante escalar, la brecha espectral es 3.
        Eigenvalores de L(K₃): {0, 3, 3}.
        """
        x = np.ones(3, dtype=np.float64)
        result = orchestrator.audit_global_state(triangle_sheaf, x)
        np.testing.assert_allclose(
            result.spectral_gap, 3.0, rtol=1e-8
        )

    def test_h0_dim_multidimensional_fiber(self):
        """
        Haz constante con fibra ℝ² sobre P₂ (0→1):
        dim H⁰ = dim(fibra) = 2 (cada componente del vector debe concordar).
        """
        I2 = np.eye(2, dtype=np.float64)
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 2, 1: 2},
            edge_dims={0: 2},
        )
        sheaf.add_edge(0, 0, 1, RestrictionMap(I2), RestrictionMap(I2))

        L = sheaf.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 2

    def test_nontrivial_restriction_changes_h0(self):
        """
        Haz NO constante puede tener dim H⁰ = 0 (sin secciones globales).

        Ejemplo: 2 nodos dim=1, arista dim=1.
        F_0 = [1], F_1 = [2].
        δx = 2·x₁ - 1·x₀. ker(δ) = {x : x₀ = 2x₁}.
        Si δx=0 necesitamos x₀ = 2x₁, que tiene dim 1.
        Pero si F_0 = [1], F_1 = [-1]:
        δx = -x₁ - x₀. ker(δ) = {x : x₀ = -x₁}, dim=1.

        Para hacer dim H⁰ = 0, necesitamos que ker(δ) = {0}.
        Esto ocurre con F_0 = [1, 0], F_1 = [0, 1] (arista dim=2, nodos dim=1).
        Entonces δx = [-x₀, x₁]ᵀ. ker(δ) = {x : x₀=0, x₁=0} = {0}.
        """
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 2},
        )
        F_0 = RestrictionMap(np.array([[1.0], [0.0]]))
        F_1 = RestrictionMap(np.array([[0.0], [1.0]]))
        sheaf.add_edge(0, 0, 1, F_0, F_1)

        L = sheaf.compute_sheaf_laplacian()
        result = _SpectralAnalyzer.compute_dense(L)
        assert result.h0_dimension == 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 14: HACES CON FIBRAS HETEROGÉNEAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeterogeneousFibers:
    """
    Verifica haces donde los espacios de fibra tienen dimensiones distintas.
    """

    def test_offsets_computed_correctly(self, heterogeneous_sheaf):
        """Los offsets nodales reflejan dimensiones acumuladas."""
        # Nodo 0: dim 2, offset 0
        # Nodo 1: dim 3, offset 2
        # Total: 5
        assert heterogeneous_sheaf.total_node_dim == 5
        assert heterogeneous_sheaf.total_edge_dim == 2

    def test_coboundary_correct_shape(self, heterogeneous_sheaf):
        """δ tiene forma (2, 5) para el haz heterogéneo."""
        delta = heterogeneous_sheaf.build_coboundary_operator()
        assert delta.shape == (2, 5)

    def test_laplacian_correct_shape(self, heterogeneous_sheaf):
        """L tiene forma (5, 5) para el haz heterogéneo."""
        L = heterogeneous_sheaf.compute_sheaf_laplacian()
        assert L.shape == (5, 5)

    def test_three_different_fiber_dims(self):
        """
        Haz con 3 nodos de dimensiones 1, 2, 3 y 2 aristas.
        """
        sheaf = CellularSheaf(
            num_nodes=3,
            node_dims={0: 1, 1: 2, 2: 3},
            edge_dims={0: 1, 1: 2},
        )

        # Arista 0: (0→1), F_0: 1×1, F_1: 1×2
        F_0e0 = RestrictionMap(np.array([[1.0]]))
        F_1e0 = RestrictionMap(np.array([[1.0, 0.0]]))
        sheaf.add_edge(0, 0, 1, F_0e0, F_1e0)

        # Arista 1: (1→2), F_1: 2×2, F_2: 2×3
        F_1e1 = RestrictionMap(np.eye(2, dtype=np.float64))
        F_2e1 = RestrictionMap(np.eye(2, 3, dtype=np.float64))
        sheaf.add_edge(1, 1, 2, F_1e1, F_2e1)

        delta = sheaf.build_coboundary_operator()
        assert delta.shape == (3, 6)  # total_edge_dim=3, total_node_dim=6

        L = sheaf.compute_sheaf_laplacian()
        assert L.shape == (6, 6)

        # Verificar simetría
        np.testing.assert_allclose(
            L.toarray(), L.toarray().T, atol=_SYMMETRY_TOLERANCE
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 15: GLOBAL FRUSTRATION ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalFrustrationAssessment:
    """Verifica la estructura GlobalFrustrationAssessment."""

    def test_frozen(self):
        """Es inmutable."""
        gfa = GlobalFrustrationAssessment(
            frustration_energy=0.0,
            h0_dimension=1,
            is_coherent=True,
            spectral_gap=1.0,
            residual_norm=0.0,
            spectral_method="dense",
        )
        with pytest.raises(AttributeError):
            gfa.frustration_energy = 999.0

    def test_all_fields_accessible(self):
        """Todos los campos son accesibles."""
        gfa = GlobalFrustrationAssessment(
            frustration_energy=0.001,
            h0_dimension=2,
            is_coherent=True,
            spectral_gap=0.5,
            residual_norm=0.0316,
            spectral_method="sparse",
        )
        assert gfa.frustration_energy == 0.001
        assert gfa.h0_dimension == 2
        assert gfa.is_coherent is True
        assert gfa.spectral_gap == 0.5
        assert gfa.residual_norm == 0.0316
        assert gfa.spectral_method == "sparse"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 16: SHEAF EDGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSheafEdge:
    """Verifica la estructura SheafEdge."""

    def test_frozen(self):
        """SheafEdge es inmutable."""
        I = RestrictionMap(np.eye(2))
        edge = SheafEdge(
            edge_id=0, u=0, v=1,
            restriction_u=I, restriction_v=I,
        )
        with pytest.raises(AttributeError):
            edge.edge_id = 5

    def test_fields_accessible(self):
        """Todos los campos son accesibles."""
        F_u = RestrictionMap(np.eye(2))
        F_v = RestrictionMap(np.ones((2, 3)))
        edge = SheafEdge(
            edge_id=7, u=3, v=5,
            restriction_u=F_u, restriction_v=F_v,
        )
        assert edge.edge_id == 7
        assert edge.u == 3
        assert edge.v == 5
        assert edge.restriction_u is F_u
        assert edge.restriction_v is F_v


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 17: CASOS LÍMITE Y PATOLOGÍAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Verifica comportamiento en situaciones extremas."""

    def test_single_edge_graph(self, orchestrator):
        """Grafo minimal: 2 nodos, 1 arista, fibra dim=1."""
        sheaf = _build_simple_sheaf_2_nodes_1_edge(d_nodes=1, d_edge=1)
        x = np.array([3.14, 3.14], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x)
        assert result.is_coherent is True
        assert result.h0_dimension == 1

    def test_large_equal_state(self, triangle_sheaf, orchestrator):
        """Consenso con valor grande sigue siendo coherente."""
        x = np.ones(3, dtype=np.float64) * 1e12
        result = orchestrator.audit_global_state(triangle_sheaf, x)
        assert result.is_coherent is True

    def test_large_disagreement_energy(self, simple_sheaf, orchestrator):
        """Desacuerdo grande produce alta energía de frustración."""
        n = simple_sheaf.total_node_dim
        x = np.zeros(n, dtype=np.float64)
        x[0] = 1e6
        with pytest.raises(HomologicalInconsistencyError):
            orchestrator.audit_global_state(simple_sheaf, x)

    def test_zero_restriction_maps(self, orchestrator):
        """
        Mapas de restricción cero: F_u = 0, F_v = 0.
        Entonces δx = 0 para todo x, y dim H⁰ = dim C⁰.
        """
        Z = np.zeros((1, 1), dtype=np.float64)
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1},
        )
        sheaf.add_edge(0, 0, 1, RestrictionMap(Z), RestrictionMap(Z))

        x = np.array([42.0, -17.0], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x)
        assert result.is_coherent is True
        assert result.h0_dimension == 2

    def test_scaling_restriction_maps(self, orchestrator):
        """
        F_0 = [2], F_1 = [2].
        δx = 2x₁ - 2x₀ = 2(x₁-x₀).
        Coherente iff x₀ = x₁.
        """
        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 1, 1: 1},
            edge_dims={0: 1},
        )
        F = RestrictionMap(np.array([[2.0]]))
        sheaf.add_edge(0, 0, 1, F, F)

        # Coherente
        x_good = np.array([5.0, 5.0], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x_good)
        assert result.is_coherent

        # Incoherente
        x_bad = np.array([5.0, 6.0], dtype=np.float64)
        with pytest.raises(HomologicalInconsistencyError):
            orchestrator.audit_global_state(sheaf, x_bad)

    def test_near_coherent_boundary(self, orchestrator):
        """
        Estado con energía justo en la frontera de tolerancia.
        Construir δx con ‖δx‖² ≈ _FRUSTRATION_TOLERANCE.
        """
        # Para P₂ con identidad: δx = x₁ - x₀.
        # ‖δx‖² = (x₁-x₀)². Para ‖δx‖² = tol, x₁-x₀ = √tol.
        sheaf = _build_path_sheaf(2, dim=1)
        epsilon = np.sqrt(_FRUSTRATION_TOLERANCE) * 0.99
        x = np.array([0.0, epsilon], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x)
        assert result.is_coherent is True

    def test_just_above_tolerance_rejects(self, orchestrator):
        """Estado con energía justo por encima de tolerancia es rechazado."""
        sheaf = _build_path_sheaf(2, dim=1)
        epsilon = np.sqrt(_FRUSTRATION_TOLERANCE) * 1.1
        x = np.array([0.0, epsilon], dtype=np.float64)
        with pytest.raises(HomologicalInconsistencyError):
            orchestrator.audit_global_state(sheaf, x)

    def test_orthogonal_restriction_maps(self, orchestrator):
        """
        Mapas de restricción ortogonales.
        F_u = [1, 0; 0, 1], F_v = [0, -1; 1, 0] (rotación 90°).
        Para coherencia: F_v·x_v = F_u·x_u, es decir Rx_v = x_u.
        """
        F_u = RestrictionMap(np.eye(2, dtype=np.float64))
        R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        F_v = RestrictionMap(R)

        sheaf = CellularSheaf(
            num_nodes=2,
            node_dims={0: 2, 1: 2},
            edge_dims={0: 2},
        )
        sheaf.add_edge(0, 0, 1, F_u, F_v)

        # x_u = (1, 0), x_v tal que R·x_v = x_u → x_v = R⁻¹(1,0) = Rᵀ(1,0) = (0,1)
        # Note: the test specifies R·x_v = x_u, so x_v should be (0, 1) as
        # R = [[0, -1], [1, 0]], R(0, 1) = (-1, 0)
        # Wait, if R = [[0, -1], [1, 0]], then R(0, 1) = [-1, 0]. But x_u is (1, 0).
        # We want R * x_v = x_u = (1, 0). So [[0, -1], [1, 0]] * [x_v0, x_v1] = [1, 0]
        # -x_v1 = 1 => x_v1 = -1
        # x_v0 = 0
        # So x_v should be (0, -1). Let's fix this in the test.
        x = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float64)
        result = orchestrator.audit_global_state(sheaf, x)
        assert result.is_coherent


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 18: PROPIEDADES DEL CONO SEMIDEFINIDO POSITIVO
# ═══════════════════════════════════════════════════════════════════════════════

class TestSDPConeProperties:
    """
    Verifica propiedades del cono de matrices semidefinidas positivas
    para el Laplaciano del haz.
    """

    def test_laplacian_in_sdp_cone(self, triangle_sheaf):
        """L ∈ S⁺ⁿ (semidefinido positivo)."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        eigvals = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigvals >= -_SPECTRAL_TOLERANCE)

    def test_sum_of_laplacians_is_sdp(self):
        """
        Si L₁, L₂ ∈ S⁺ⁿ entonces L₁ + L₂ ∈ S⁺ⁿ.
        (Cono convexo cerrado.)
        """
        sheaf1 = _build_path_sheaf(3, dim=1)
        sheaf2 = _build_triangle_sheaf(dim=1)

        L1 = sheaf1.compute_sheaf_laplacian()
        L2 = sheaf2.compute_sheaf_laplacian()

        L_sum = L1 + L2
        eigvals = np.linalg.eigvalsh(L_sum.toarray())
        assert np.all(eigvals >= -_SPECTRAL_TOLERANCE)

    def test_scalar_multiple_is_sdp(self, triangle_sheaf):
        """αL ∈ S⁺ⁿ para α ≥ 0."""
        L = triangle_sheaf.compute_sheaf_laplacian()
        for alpha in [0.0, 0.5, 1.0, 100.0]:
            eigvals = np.linalg.eigvalsh((alpha * L).toarray())
            assert np.all(eigvals >= -_SPECTRAL_TOLERANCE)

    def test_kernel_is_subspace(self, triangle_sheaf):
        """
        ker(L) es un subespacio vectorial:
        si Lx = 0 y Ly = 0, entonces L(αx + βy) = 0.
        """
        L = triangle_sheaf.compute_sheaf_laplacian()
        # El consenso está en ker(L)
        ones = np.ones(3, dtype=np.float64)
        assert np.linalg.norm(L.dot(ones)) < _FLOAT64_ATOL
        assert np.linalg.norm(L.dot(3.0 * ones)) < _FLOAT64_ATOL
        assert np.linalg.norm(L.dot(-7.5 * ones)) < _FLOAT64_ATOL


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 19: EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy:
    """Verifica la jerarquía de excepciones."""

    def test_base_is_exception(self):
        """SheafCohomologyError hereda de Exception."""
        assert issubclass(SheafCohomologyError, Exception)

    def test_homological_is_base(self):
        """HomologicalInconsistencyError hereda de SheafCohomologyError."""
        assert issubclass(HomologicalInconsistencyError, SheafCohomologyError)

    def test_degeneracy_is_base(self):
        """SheafDegeneracyError hereda de SheafCohomologyError."""
        assert issubclass(SheafDegeneracyError, SheafCohomologyError)

    def test_spectral_is_base(self):
        """SpectralComputationError hereda de SheafCohomologyError."""
        assert issubclass(SpectralComputationError, SheafCohomologyError)

    def test_can_catch_all_with_base(self):
        """Todas las excepciones del módulo se capturan con la base."""
        for exc_class in [
            HomologicalInconsistencyError,
            SheafDegeneracyError,
            SpectralComputationError,
        ]:
            with pytest.raises(SheafCohomologyError):
                raise exc_class("test")

    def test_exceptions_carry_messages(self):
        """Las excepciones preservan mensajes."""
        msg = "Error específico de prueba"
        exc = HomologicalInconsistencyError(msg)
        assert msg in str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 20: CONSTANTES DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleConstants:
    """Verifica coherencia de las constantes del módulo."""

    def test_frustration_tolerance_positive(self):
        """Tolerancia de frustración es positiva."""
        assert _FRUSTRATION_TOLERANCE > 0

    def test_symmetry_tolerance_positive(self):
        """Tolerancia de simetría es positiva."""
        assert _SYMMETRY_TOLERANCE > 0

    def test_spectral_tolerance_positive(self):
        """Tolerancia espectral es positiva."""
        assert _SPECTRAL_TOLERANCE > 0

    def test_dense_max_dim_positive(self):
        """Umbral de dimensión densa es positivo."""
        assert _DENSE_SPECTRAL_MAX_DIM > 0
        assert isinstance(_DENSE_SPECTRAL_MAX_DIM, int)

    def test_sparse_max_eigenvalues_positive(self):
        """Número máximo de eigenvalores dispersos es positivo."""
        assert _SPARSE_MAX_EIGENVALUES > 0
        assert isinstance(_SPARSE_MAX_EIGENVALUES, int)

    def test_arpack_tolerance_positive(self):
        """Tolerancia ARPACK es positiva."""
        assert _ARPACK_TOLERANCE > 0

    def test_tolerances_are_small(self):
        """Todas las tolerancias son mucho menores que 1."""
        assert _FRUSTRATION_TOLERANCE < 1e-3
        assert _SYMMETRY_TOLERANCE < 1e-6
        assert _SPECTRAL_TOLERANCE < 1e-3
        assert _ARPACK_TOLERANCE < 1e-3

    def test_spectral_and_frustration_tolerances_independent(self):
        """
        Las tolerancias espectral y de frustración pueden ser diferentes,
        reflejando que son criterios independientes.
        """
        # Solo verificar que ambas existen y son positivas
        assert _SPECTRAL_TOLERANCE > 0
        assert _FRUSTRATION_TOLERANCE > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 21: DETERMINISMO Y REPRODUCIBILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Verifica que los cálculos son deterministas."""

    def test_coboundary_deterministic(self, triangle_sheaf):
        """Construcciones repetidas de δ producen el mismo resultado."""
        # Invalidar caché construyendo un haz nuevo
        sheaf1 = _build_triangle_sheaf(dim=1)
        sheaf2 = _build_triangle_sheaf(dim=1)

        delta1 = sheaf1.build_coboundary_operator()
        delta2 = sheaf2.build_coboundary_operator()

        np.testing.assert_array_equal(
            delta1.toarray(), delta2.toarray()
        )

    def test_laplacian_deterministic(self, triangle_sheaf):
        """Construcciones repetidas de L producen el mismo resultado."""
        sheaf1 = _build_triangle_sheaf(dim=1)
        sheaf2 = _build_triangle_sheaf(dim=1)

        L1 = sheaf1.compute_sheaf_laplacian()
        L2 = sheaf2.compute_sheaf_laplacian()

        np.testing.assert_array_equal(
            L1.toarray(), L2.toarray()
        )

    def test_audit_deterministic(self, triangle_sheaf, orchestrator):
        """Auditorías repetidas del mismo estado producen el mismo resultado."""
        x = np.ones(3, dtype=np.float64) * 2.0

        result1 = orchestrator.audit_global_state(triangle_sheaf, x)
        result2 = orchestrator.audit_global_state(triangle_sheaf, x)

        assert result1.frustration_energy == result2.frustration_energy
        assert result1.h0_dimension == result2.h0_dimension
        assert result1.spectral_gap == result2.spectral_gap
        assert result1.residual_norm == result2.residual_norm


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 22: PROPIEDADES ALGEBRAICAS DE δ
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoboundaryAlgebraicProperties:
    """
    Verifica propiedades algebraicas del operador de cofrontera δ.
    """

    def test_delta_is_linear(self, triangle_sheaf):
        """δ es un operador lineal: δ(αx + βy) = αδx + βδy."""
        delta = triangle_sheaf.build_coboundary_operator()
        n = triangle_sheaf.total_node_dim

        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            alpha, beta = rng.standard_normal(2)

            lhs = delta.dot(alpha * x + beta * y)
            rhs = alpha * delta.dot(x) + beta * delta.dot(y)

            np.testing.assert_allclose(lhs, rhs, atol=_FLOAT64_ATOL)

    def test_rank_of_delta_path(self):
        """
        Para P_n con haz constante escalar, rank(δ) = n-1.
        δ es (n-1)×n, y tiene rango completo por filas para un camino.
        """
        for n in [2, 3, 5, 10]:
            sheaf = _build_path_sheaf(n, dim=1)
            delta = sheaf.build_coboundary_operator()
            rank = np.linalg.matrix_rank(delta.toarray())
            assert rank == n - 1

    def test_nullity_of_delta_path(self):
        """
        Para P_n con haz constante escalar, dim ker(δ) = 1 (consenso).
        Por rank-nullity: dim ker(δ) = n - rank(δ) = n - (n-1) = 1.
        """
        for n in [2, 3, 5, 10]:
            sheaf = _build_path_sheaf(n, dim=1)
            delta = sheaf.build_coboundary_operator()
            nullity = n - np.linalg.matrix_rank(delta.toarray())
            assert nullity == 1

    def test_rank_nullity_theorem(self, triangle_sheaf):
        """
        Teorema rank-nullity: dim(C⁰) = rank(δ) + dim ker(δ).
        """
        delta = triangle_sheaf.build_coboundary_operator()
        n = triangle_sheaf.total_node_dim
        rank = np.linalg.matrix_rank(delta.toarray())

        L = triangle_sheaf.compute_sheaf_laplacian()
        eigvals = np.linalg.eigvalsh(L.toarray())
        nullity = int(np.sum(np.abs(eigvals) <= _SPECTRAL_TOLERANCE))

        assert rank + nullity == n

    def test_image_orthogonal_to_kernel_of_transpose(self, triangle_sheaf):
        """
        Im(δ) ⊥ ker(δᵀ) en C¹.
        Para y ∈ ker(δᵀ) y cualquier x: ⟨δx, y⟩ = ⟨x, δᵀy⟩ = ⟨x, 0⟩ = 0.
        """
        delta = triangle_sheaf.build_coboundary_operator()
        delta_dense = delta.toarray()
        delta_T = delta_dense.T

        # Encontrar ker(δᵀ)
        _, s, Vh = np.linalg.svd(delta_T)
        null_mask = s < _FLOAT64_ATOL
        # Las filas de Vh correspondientes a valores singulares ≈ 0
        # forman una base de ker(δᵀ)
        kernel_vectors = Vh[len(s) - int(np.sum(null_mask)):]

        if kernel_vectors.shape[0] > 0:
            rng = np.random.default_rng(seed=42)
            for _ in range(20):
                x = rng.standard_normal(triangle_sheaf.total_node_dim)
                delta_x = delta_dense @ x
                for y in kernel_vectors:
                    dot = np.dot(delta_x, y)
                    np.testing.assert_allclose(
                        dot, 0.0, atol=_FLOAT64_ATOL,
                        err_msg="Im(δ) no ortogonal a ker(δᵀ)",
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 23: TESTS DE LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogging:
    """Verifica que los mensajes de logging se emiten correctamente."""

    def test_coherent_state_logs_info(
        self, triangle_sheaf, orchestrator, caplog,
    ):
        """Estado coherente emite log INFO con diagnóstico."""
        x = np.ones(3, dtype=np.float64)
        with caplog.at_level(
            logging.INFO, logger="MIC.ImmuneSystem.SheafCohomology"
        ):
            orchestrator.audit_global_state(triangle_sheaf, x)
        assert "Auditoría cohomológica exitosa" in caplog.text

    def test_incoherent_state_logs_critical(
        self, triangle_sheaf, orchestrator, caplog,
    ):
        """Estado incoherente emite log CRITICAL."""
        x = np.array([1.0, 100.0, -50.0], dtype=np.float64)
        with caplog.at_level(
            logging.CRITICAL, logger="MIC.ImmuneSystem.SheafCohomology"
        ):
            with pytest.raises(HomologicalInconsistencyError):
                orchestrator.audit_global_state(triangle_sheaf, x)
        assert "FRUSTRACIÓN DE HAZ" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 24: INTEGRACIÓN COMPLETA
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Tests de integración que ejercitan el pipeline completo."""

    def test_full_pipeline_small_sheaf(self, orchestrator):
        """Pipeline completo para un haz pequeño bien construido."""
        sheaf = _build_triangle_sheaf(dim=2)
        n = sheaf.total_node_dim  # 3 nodos × 2 dims = 6

        # Consenso: todos los nodos con el mismo vector (1, 2)
        x = np.tile([1.0, 2.0], 3)
        result = orchestrator.audit_global_state(sheaf, x)

        assert result.is_coherent
        assert result.frustration_energy < _FRUSTRATION_TOLERANCE
        assert result.h0_dimension == 2  # dim(fibra) para grafo conexo
        assert result.spectral_gap > 0
        assert result.spectral_method == "dense"

    def test_full_pipeline_heterogeneous(
        self, heterogeneous_sheaf, orchestrator,
    ):
        """Pipeline completo para haz heterogéneo."""
        # Estado coherente: (a,b, a,b,0)
        x = np.array([3.0, -1.0, 3.0, -1.0, 0.0], dtype=np.float64)
        result = orchestrator.audit_global_state(heterogeneous_sheaf, x)
        assert result.is_coherent

    def test_build_audit_cycle(self, orchestrator):
        """
        Ciclo completo: construir haz → ensamblar → auditar → interpretar.
        """
        # Haz sobre K₄ completo con fibra ℝ¹
        n = 4
        I = np.eye(1, dtype=np.float64)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        sheaf = CellularSheaf(
            num_nodes=n,
            node_dims={i: 1 for i in range(n)},
            edge_dims={k: 1 for k in range(len(edges))},
        )

        for k, (u, v) in enumerate(edges):
            sheaf.add_edge(k, u, v, RestrictionMap(I), RestrictionMap(I))

        assert sheaf.is_fully_assembled
        assert sheaf.num_edges_added == 6  # C(4,2) = 6

        # Consenso
        x = np.ones(n, dtype=np.float64) * np.pi
        result = orchestrator.audit_global_state(sheaf, x)

        assert result.is_coherent
        assert result.h0_dimension == 1
        # K₄ tiene brecha espectral = n = 4
        np.testing.assert_allclose(
            result.spectral_gap, 4.0, rtol=1e-8
        )