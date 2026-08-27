# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Business Canvas Solver (Estrato Alfa)                    ║
║ Ruta: tests/unit/alfa/test_business_canvas.py                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np
import networkx as nx

from app.alfa.business_canvas import (
    AlphaTopologyVector,
    canonicalize_edge,
    compute_numerical_rank,
    compute_null_space_basis,
    safe_eigenvalues_symmetric,
    ChainComplex1D,
    HomologyMetrics,
    SpectralMetrics,
    BmcTopologyMetrics,
    BMCTopologyError,
    TopologicalInvariantError,
    PayloadValidationError,
    HomologicalInconsistencyError,
    BMC_NODES,
    BASE_EDGES,
)
from app.core.mic_algebra import CategoricalState
from app.core.schemas import Stratum


class TestBusinessCanvasPureFunctions:
    """Pruebas unitarias para funciones algebraicas puras en el complejo 1D."""

    def test_canonicalize_edge(self):
        """Verifica la orientación canónica lexicográfica."""
        assert canonicalize_edge("P_val", "P_can") == ("P_can", "P_val")
        assert canonicalize_edge("P_act", "P_rec") == ("P_act", "P_rec")

    def test_compute_numerical_rank(self):
        """Calcula el rango numérico mediante SVD."""
        M = np.array([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 6.0],
                      [0.0, 1.0, 1.0]], dtype=np.float64)
        assert compute_numerical_rank(M) == 2

    def test_compute_null_space_basis(self):
        """Calcula base ortonormal del kernel."""
        M = np.array([[1.0, 1.0]], dtype=np.float64)
        null_basis = compute_null_space_basis(M)
        assert null_basis.shape == (2, 1)
        assert np.allclose(M @ null_basis, 0.0)

    def test_safe_eigenvalues_symmetric(self):
        """Calcula autovalores de matriz simétrica limpios de ruido."""
        S = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        evals = safe_eigenvalues_symmetric(S)
        assert np.allclose(evals, [1.0, 3.0])


class TestAlphaTopologyVector:
    """Evaluación del morfismo de topología simplicial del Business Model Canvas."""

    def setup_method(self):
        self.vector = AlphaTopologyVector()

    def test_build_directed_business_graph_default(self):
        """Construye el digrafo base del BMC con los 9 bloques y 11 aristas base."""
        payload = {}
        G = self.vector._build_directed_business_graph(payload)
        assert len(G.nodes) == 9
        assert len(G.edges) == len(BASE_EDGES)

    def test_chain_complex_1d_construction(self):
        """Verifica la construcción del complejo de cadenas C_*(K)."""
        payload = {}
        G = self.vector._build_directed_business_graph(payload)
        H = self.vector._to_weighted_undirected(G)
        complex_1d = self.vector._build_chain_complex_1d(H)

        assert isinstance(complex_1d, ChainComplex1D)
        assert complex_1d.dimension_0 == 9
        assert complex_1d.dimension_1 == len(BASE_EDGES)
        assert complex_1d.boundary_1.shape == (9, len(BASE_EDGES))

    def test_call_morphism_acyclic_tree(self):
        """Ejecuta la invocación sobre un árbol acíclico (beta_1 = 0, chi = 1)."""
        payload = {
            "remove_edges": [
                {"source": "P_soc", "target": "P_cost"},
                {"source": "P_rec", "target": "P_cost"},
                {"source": "P_rel", "target": "P_seg"},
            ]
        }
        state = CategoricalState(
            payload=payload,
            validated_strata=frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}),
        )
        res = self.vector(state)
        assert res["success"] is True
        assert res["stratum"] == Stratum.ALPHA or res["stratum"] == Stratum.ALPHA.name
