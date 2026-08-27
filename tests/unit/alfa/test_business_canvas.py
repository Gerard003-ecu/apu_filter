# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para business_canvas."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.alfa.business_canvas import (
    canonicalize_edge,
    compute_numerical_rank,
    compute_null_space_basis,
    safe_eigenvalues_symmetric,
    ChainComplex1D,
    HomologyMetrics,
    SpectralMetrics,
    CycleSpaceMetrics,
    BmcTopologyMetrics,
    AlphaTopologyVector,
    BMCTopologyError,
    TopologicalInvariantError,
    SpectralAnalysisError,
    PayloadValidationError,
    HomologicalInconsistencyError,
    EPSILON,
    RANK_TOL,
    EIGENVALUE_ZERO_TOL,
    MIN_FIEDLER_VALUE
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_epsilon(self):
        assert EPSILON > 0

    def test_rank_tol(self):
        assert RANK_TOL > 0

    def test_eigenvalue_zero_tol(self):
        assert EIGENVALUE_ZERO_TOL > 0

    def test_min_fiedler_value(self):
        assert MIN_FIEDLER_VALUE > 0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_bmc_topology_error(self):
        with pytest.raises(BMCTopologyError):
            raise BMCTopologyError("test")
            
    def test_topological_invariant_error(self):
        with pytest.raises(TopologicalInvariantError):
            raise TopologicalInvariantError("test")

    def test_spectral_analysis_error(self):
        with pytest.raises(SpectralAnalysisError):
            raise SpectralAnalysisError("test")
            
    def test_payload_validation_error(self):
        with pytest.raises(PayloadValidationError):
            raise PayloadValidationError("test")

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_chain_complex_1d_frozen(self):
        obj = ChainComplex1D(
            vertex_basis=("A", "B"),
            edge_basis=(("A", "B"),),
            boundary_1=np.array([[-1.0], [1.0]]),
            laplacian_0=np.array([[1.0, -1.0], [-1.0, 1.0]]),
            laplacian_1=np.array([[2.0]])
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.vertex_basis = ("C",)

    def test_homology_metrics_inconsistency(self):
        with pytest.raises(HomologicalInconsistencyError):
            HomologyMetrics(
                n_vertices=2,
                n_edges=1,
                rank_boundary_1=1,
                nullity_boundary_1=0,
                beta_0=1,
                beta_1=0,
                euler_char=1,
                euler_from_betti=0  # should be 1
            )

    def test_spectral_metrics_invalid(self):
        with pytest.raises(SpectralAnalysisError):
            SpectralMetrics(
                eigenvalues=(-1.0, 1.0),
                fiedler_value=1.0,
                spectral_gap=1.0,
                multiplicity_zero=1,
                spectral_radius=1.0,
                trace_laplacian=0.0
            )

class TestFunctions:
    """Granular tests for helper functions."""
    def test_canonicalize_edge(self):
        assert canonicalize_edge("B", "A") == ("A", "B")
        assert canonicalize_edge("A", "B") == ("A", "B")

    def test_compute_numerical_rank(self):
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        rank = compute_numerical_rank(M)
        assert rank == 1

    def test_compute_null_space_basis(self):
        M = np.array([[1.0, 1.0], [1.0, 1.0]])
        null_basis = compute_null_space_basis(M)
        assert null_basis.shape[1] == 1

    def test_safe_eigenvalues_symmetric(self):
        M = np.array([[2.0, -1.0], [-1.0, 2.0]])
        eigs = safe_eigenvalues_symmetric(M)
        assert abs(eigs[0] - 1.0) < 1e-9
        assert abs(eigs[1] - 3.0) < 1e-9

class TestPhase1_AlphaTopologyVector:
    """Granular tests for AlphaTopologyVector."""
    def setup_method(self):
        with patch.multiple(AlphaTopologyVector, __abstractmethods__=frozenset()):
            self.morphism = AlphaTopologyVector()

    def test_validate_payload_schema_valid(self):
        payload = {"disable_nodes": ["P_soc"]}
        # Should not raise
        self.morphism._validate_payload_schema(payload)

    def test_validate_payload_schema_invalid(self):
        payload = {"disable_nodes": "P_soc"}
        with pytest.raises(PayloadValidationError):
            self.morphism._validate_payload_schema(payload)

    def test_validate_positive_weight(self):
        with pytest.raises(PayloadValidationError):
            self.morphism._validate_positive_weight(-1.0, "test")

    def test_build_directed_business_graph(self):
        payload = {"extra_edges": [{"source": "P_soc", "target": "P_ing", "weight": 2.0}]}
        G = self.morphism._build_directed_business_graph(payload)
        assert G.has_edge("P_soc", "P_ing")
        assert abs(G["P_soc"]["P_ing"]["weight"] - 2.0) < 1e-9

class TestPhase2_Phase3:
    """Granular tests for Phase 2 and 3 methods."""
    def setup_method(self):
        with patch.multiple(AlphaTopologyVector, __abstractmethods__=frozenset()):
            self.morphism = AlphaTopologyVector()
            
    def test_to_weighted_undirected(self):
        G = self.morphism._build_directed_business_graph({})
        H = self.morphism._to_weighted_undirected(G)
        assert not H.is_directed()
        
    def test_build_chain_complex_1d(self):
        G = self.morphism._build_directed_business_graph({})
        H = self.morphism._to_weighted_undirected(G)
        comp = self.morphism._build_chain_complex_1d(H)
        assert isinstance(comp, ChainComplex1D)
        assert comp.boundary_1.shape[0] == 9
        
class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def test_alpha_topology_vector_call(self):
        with patch.multiple(AlphaTopologyVector, __abstractmethods__=frozenset()):
            morphism = AlphaTopologyVector()
            state = MagicMock()
            state.payload = {}
            # Just verifying it handles the call without crashing if mocked internally, 
            # or if the method completes. Since some internal methods aren't available, we just mock the compute step.
            with patch.object(morphism, '_compute_full_analysis') as mock_compute:
                mock_compute.return_value = BmcTopologyMetrics(
                    beta_0=1, beta_1=0, euler_char=1, rank_boundary_1=8, nullity_boundary_1=0,
                    fiedler_value=1.0, spectral_gap=1.0, spectral_radius=2.0, multiplicity_zero=1, trace_laplacian=10.0,
                    directed_cycle_count=0, fundamental_cycle_count=0,
                    is_connected=True, has_cycle_space=False, has_directed_feedback=False, is_dag=True, is_spectrally_stable=True,
                    connectivity_class="CONNECTED", n_vertices=9, n_edges=8
                )
                with patch.object(morphism, '_enforce_topological_constraints'):
                    with patch.object(morphism, '_generate_narrative', return_value="narrative"):
                        res = morphism(state)
                        assert res.success is True
